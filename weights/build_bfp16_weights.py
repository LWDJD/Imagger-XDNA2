"""build_bfp16_weights.py -- pack the whole transformer into ONE BlockFP16 file.

THE GOAL
--------
A single `weights.bfp16` holding every shard's weights in AMD's kernel-ready
`bfp16ebs8` layout: 1.125 bytes per parameter, so 6,143,056,384 parameters
become 6.44 GiB instead of the 22.93 GiB of fp32 safetensors.

The encoding is `pack_bfp16.pack`, which reproduces a vendor `.rai` byte for
byte on all 12 Z-Image shapes (whole-region identity, 0 differing bytes).

FORMAT
------
    [8-byte little-endian header length][JSON header][packed payloads]

Header:
    {"format": "bfp16ebs8", "total_params": N, "total_bytes": B,
     "tensors": [{"name", "checkpoint", "shape": [N,K], "params",
                  "tile_rows", "offset", "length"}]}

`offset`/`length` locate each tensor's packed bytes.  Reconstruct with
`pack_bfp16.unpack(blob, N, K, tile_rows)`.

WHY tile_rows IS RECORDED PER TENSOR
------------------------------------
`tile_rows` is not a function of (N,K) alone -- it also depends on the GEMM's
token count M, because M changes the compiler's subvolume tiling.  The current
model is exported for M=64 (the 128x128 geometry), which is what
`pack_bfp16.KNOWN_TILE_ROWS` was measured at.  Recording it per tensor means
the file stays usable even if a different M needs a different value later.

Usage:
    python build_bfp16_weights.py
    python build_bfp16_weights.py --out _probe_out/weights.bfp16
"""

from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent                      # the repository root
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MANIFEST = HERE / "skeleton_manifest.json"     # the plan, committed
DEFAULT_OUT = ROOT / "weights.bfp16"
# Both are set by main() from the command line: there is no sensible
# default for where the upstream model happens to live.
INDEX = None      # --index : diffusion_pytorch_model.safetensors.index.json
SHARDS = None     # --shards: the directory holding the *.safetensors


def _force_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass


def read_checkpoint_tensor(ckpt: str, name: str, shape: tuple[int, ...]
                           ) -> np.ndarray:
    """Read one tensor straight out of the safetensors header offsets.

    `safetensors.get_tensor` crashes on this machine (access violation inside
    torch/storage.py), so go through the documented on-disk layout instead --
    the same reader `shard_lib.WeightStreamer` uses.
    """
    if SHARDS is None:
        raise RuntimeError("the shard directory is unset: pass --shards")
    root = pathlib.Path(SHARDS)
    with open(str(root / ckpt), "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(hlen))
        base = 8 + hlen
        e = meta[name]
        o0, o1 = e["data_offsets"]
        f.seek(base + o0)
        raw = f.read(o1 - o0)
    if e.get("dtype") != "F32":
        raise NotImplementedError(f"{name}: dtype {e.get('dtype')}")
    return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()


def main() -> int:
    _force_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--index", default=None,
                    help="the upstream diffusion_pytorch_model.safetensors"
                         ".index.json (tensor name -> shard)")
    ap.add_argument("--shards", default=None,
                    help="directory holding the diffusion_pytorch_model-"
                         "0000N-of-00003.safetensors shards")
    ap.add_argument("--m", type=int, default=64,
                    help="token count the model is exported for; the "
                         "KNOWN_TILE_ROWS table was measured at this M")
    args = ap.parse_args()

    global INDEX, SHARDS
    if not args.index:
        ap.error("--index is required: point it at the upstream " 
                 "diffusion_pytorch_model.safetensors.index.json")
    if not args.shards:
        ap.error("--shards is required: the directory with the safetensors")
    INDEX = pathlib.Path(args.index)
    SHARDS = pathlib.Path(args.shards)

    from pack_bfp16 import KNOWN_TILE_ROWS, pack

    manifest = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))
    # weight_map.json is committed next to the manifest, so the plan travels
    # with the code and the original skeleton_dir is not needed.
    wm_path = pathlib.Path(args.manifest).parent / "weight_map.json"
    if not wm_path.exists() and manifest.get("skeleton_dir"):
        wm_path = pathlib.Path(manifest["skeleton_dir"]) / "weight_map.json"
    if not wm_path.exists():
        print(f"[FAIL] {wm_path} missing -- run the weight-map derivation first")
        return 1
    wmap = json.loads(wm_path.read_text(encoding="utf-8"))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("build_bfp16_weights -- one BlockFP16 file for the whole transformer")
    print("=" * 78)
    print(f"  out  : {out}")
    print(f"  M    : {args.m} tokens (tile_rows table is measured at this M)")
    print(f"  shards: {len(wmap.get('shards', {}))}")

    index = json.loads(INDEX.read_text(encoding="utf-8"))
    ckpt_of = index["weight_map"]

    # ---- pass 1: lay out the file WITHOUT reading any tensor data ------
    # The header must be written BEFORE the payloads: writing it afterwards
    # overwrote the first 88 KB of payload (the header is larger than the 8
    # bytes reserved for its length prefix), which showed up downstream as
    # "short read" on a late tensor and garbage exponents on unpack.
    # Offsets are pure arithmetic -- params x bytes-per-param -- so this pass
    # needs no reads at all.
    plan = []
    off = 0
    for shard_name, srec in wmap.get("shards", {}).items():
        for p in srec.get("pairs", []):
            shape = tuple(p["shape"])
            params = int(np.prod(shape))
            packed = len(shape) == 2
            if packed:
                N, K = shape
                tr = KNOWN_TILE_ROWS.get((N, K))
                if tr is None:
                    raise KeyError(
                        f"no known tile_rows for (N={N}, K={K}) at "
                        f"M={args.m}; measure it with detect_tile_rows")
                ln = N * K * 9 // 8
            else:
                tr = None
                ln = params * 4
            plan.append({"name": p["checkpoint"], "shard": shard_name,
                         "shape": list(shape), "params": params,
                         "tile_rows": tr, "packed_bfp16": packed,
                         "offset": off, "length": ln})
            off += ln
    total_bytes = off
    total_params = sum(r["params"] for r in plan)
    packed_params = sum(r["params"] for r in plan if r["packed_bfp16"])

    header = {
        "format": "bfp16ebs8", "m_tokens": args.m,
        "total_params": total_params, "total_bytes": total_bytes,
        "bytes_per_param": round(total_bytes / total_params, 6),
        "tensors": plan,
    }
    hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")

    print(f"  layout     : {len(plan)} tensors, {total_bytes:,} B payload, "
          f"{len(hjson):,} B header")

    # ---- pass 2: write header, then stream the payloads ----------------
    t0 = time.time()
    with open(str(out), "wb") as fh:
        fh.write(struct.pack("<Q", len(hjson)))
        fh.write(hjson)
        assert fh.tell() == 8 + len(hjson), fh.tell()

        for shard_name, srec in wmap.get("shards", {}).items():
            t_sh = time.time()
            packed_here = 0
            for p in srec.get("pairs", []):
                name = p["checkpoint"]
                shape = tuple(p["shape"])
                ck = ckpt_of.get(name)
                if ck is None:
                    raise KeyError(f"{name} is not in the safetensors index")
                W = read_checkpoint_tensor(ck, name, shape)
                if len(shape) == 2:
                    blob = pack(W, tile_rows=KNOWN_TILE_ROWS[shape])
                else:
                    blob = np.ascontiguousarray(
                        W, dtype=np.float32).tobytes()
                rec = next(r for r in plan if r["name"] == name)
                if len(blob) != rec["length"]:
                    raise AssertionError(
                        f"{name}: wrote {len(blob)} B, layout said "
                        f"{rec['length']}")
                fh.write(blob)
                packed_here += len(blob)
                del W, blob
            print(f"  {shard_name:16s} {len(srec.get('pairs', [])):3d} tensors  "
                  f"{packed_here / 2**20:8.1f} MiB  "
                  f"{time.time() - t_sh:5.1f}s")
        assert fh.tell() == 8 + len(hjson) + total_bytes, fh.tell()

    records = plan

    dt = time.time() - t0
    size = out.stat().st_size
    unpacked_params = total_params - packed_params
    expected = packed_params * 9 // 8 + unpacked_params * 4
    print()
    print(f"  tensors      : {len(records)}  "
          f"(GEMM packed {sum(1 for r in records if r['packed_bfp16'])}, "
          f"raw {sum(1 for r in records if not r['packed_bfp16'])})")
    print(f"  parameters   : {total_params:,}  "
          f"(BlockFP16 {packed_params:,} + fp32 {unpacked_params:,})")
    print(f"  packed bytes : {total_bytes:,}   expected {expected:,}")
    print(f"  file size    : {size:,} = {size / 2**30:.3f} GiB")
    print(f"  fp32 source  : {total_params * 4:,} = "
          f"{total_params * 4 / 2**30:.3f} GiB")
    print(f"  compression  : {total_params * 4 / size:.3f}x")
    print(f"  wall time    : {dt:.0f}s")

    assert total_bytes == expected, (total_bytes, expected)
    assert abs(packed_params * 9 // 8 / packed_params - 1.125) < 1e-6, \
        "BFP16 must be 1.125 B/param, not 2.0 (that would be bf16)"
    frac = packed_params / total_params
    print(f"\n  OK: {frac:.4%} of parameters are BlockFP16 at 1.125 B/param")
    print(f"      the remaining {unpacked_params:,} are 1-D norms/biases kept "
          f"as fp32 ({unpacked_params * 4 / 2**20:.1f} MiB, "
          f"{(1 - frac):.4%}) -- they are not GEMM operands so the kernel "
          f"tiling does not apply, and they are precision-sensitive")
    return 0


if __name__ == "__main__":
    sys.exit(main())
