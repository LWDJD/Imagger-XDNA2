"""verify_bfp16_weights.py -- round-trip the packed file against the checkpoint.

`build_bfp16_weights.py` checks that the byte ACCOUNTING closes; that is not the
same as the CONTENT being right.  This unpacks every tensor out of
`weights.bfp16` and compares it with the value read straight from the
safetensors checkpoint.

Two different tolerances apply, and conflating them would hide a bug:

  * packed (BlockFP16) tensors -- the round trip is lossy by design.  The floor
    is 2^-7 = 0.0078 relative per element; anything materially above that means
    the packing is wrong.
  * raw tensors (1-D norms/biases) -- stored as fp32, so the round trip must be
    BIT EXACT.  Any difference is a bug.

Usage:
    python verify_bfp16_weights.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys

import numpy as np

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent                      # the repository root
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_FILE = ROOT / "weights.bfp16"
INDEX = None      # --index : diffusion_pytorch_model.safetensors.index.json
SHARDS = None     # --shards: the directory holding the *.safetensors


def _force_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass


def read_checkpoint_tensor(ckpt: str, name: str, shape) -> np.ndarray:
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
    return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()


def main() -> int:
    _force_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(DEFAULT_FILE))
    ap.add_argument("--index", default=None,
                    help="the upstream diffusion_pytorch_model.safetensors"
                         ".index.json")
    ap.add_argument("--shards", default=None,
                    help="directory holding the safetensors shards")
    ap.add_argument("--tol", type=float, default=2 ** -6,
                    help="relative tolerance for packed tensors")
    args = ap.parse_args()

    global INDEX, SHARDS
    INDEX = pathlib.Path(args.index) if args.index else None
    SHARDS = pathlib.Path(args.shards) if args.shards else None

    from pack_bfp16 import unpack

    path = pathlib.Path(args.file)
    blob = path.read_bytes()
    hlen = struct.unpack("<Q", blob[:8])[0]
    header = json.loads(blob[8:8 + hlen].decode("utf-8"))
    base = 8 + hlen
    tensors = header["tensors"]

    if INDEX is None:
        ap.error("--index is required to check against the source weights")
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    ckpt_of = index["weight_map"]

    print("=" * 78)
    print("verify_bfp16_weights -- unpack and compare against the checkpoint")
    print("=" * 78)
    print(f"  file     : {path}")
    print(f"  size     : {len(blob):,} B = {len(blob) / 2**30:.3f} GiB")
    print(f"  header   : {hlen:,} B, {len(tensors)} tensors")
    print(f"  format   : {header.get('format')}   M={header.get('m_tokens')}")

    bad_pack = []
    bad_raw = []
    worst = (0.0, None)
    n_pack = n_raw = 0
    sum_pack_err = 0.0
    checked_bytes = 0

    for t in tensors:
        name = t["name"]
        shape = tuple(t["shape"])
        off, ln = t["offset"], t["length"]
        seg = blob[base + off:base + off + ln]
        if len(seg) != ln:
            raise IOError(f"{name}: short read")
        checked_bytes += ln
        ck = ckpt_of[name]
        ref = read_checkpoint_tensor(ck, name, shape)

        if t.get("packed_bfp16"):
            n_pack += 1
            got = unpack(seg, shape[0], shape[1], tile_rows=t["tile_rows"])
            scale = float(np.abs(ref).max()) + 1e-30
            rel = float(np.abs(got - ref).max()) / scale
            mean_rel = float(np.abs(got - ref).mean()) / scale
            sum_pack_err += rel
            if rel > worst[0]:
                worst = (rel, name)
            if rel > args.tol:
                bad_pack.append((name, rel))
        else:
            n_raw += 1
            got = np.frombuffer(seg, dtype=np.float32).reshape(shape)
            if not np.array_equal(got, ref):
                bad_raw.append(name)

    print()
    print(f"  packed (BlockFP16) tensors : {n_pack}")
    print(f"  raw    (fp32) tensors      : {n_raw}")
    print(f"  bytes verified             : {checked_bytes:,} "
          f"({checked_bytes / len(blob):.4%} of the file)")
    print(f"  worst packed rel error     : {worst[0]:.4e}  ({worst[1]})")
    print(f"  mean of per-tensor worst   : {sum_pack_err / max(1, n_pack):.4e}")
    print(f"  tolerance                  : {args.tol:.4e} "
          f"(BFP16 floor is {2 ** -7:.4e})")
    print(f"  raw tensors bit-exact      : "
          f"{'YES' if not bad_raw else f'NO -- {len(bad_raw)} differ'}")

    ok = not bad_pack and not bad_raw
    if ok:
        verdict = ("PASS -- every packed tensor round-trips inside the BFP16 "
                   "tolerance and every raw tensor is bit-exact.")
    else:
        verdict = (f"FAIL -- {len(bad_pack)} packed tensor(s) over tolerance, "
                   f"{len(bad_raw)} raw tensor(s) not bit-exact")
        for n, r in bad_pack[:8]:
            print(f"    packed  {n:50s} rel {r:.4e}")
        for n in bad_raw[:8]:
            print(f"    raw     {n}")
    print(f"\n  verdict: {verdict}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
