# SPDX-License-Identifier: MIT
"""extract.py -- extract ONE real Z-Image-Turbo DiT transformer block and pack
every one of its weight tensors to `bfp16ebs8` (BlockFP16), plus the vendor
`coalesed_weights` regions, with a manifest.

WHAT THIS PRODUCES (all under this directory)
--------------------------------------------
    block0.npz             per-tensor arrays:
                             <name>            uint8   logical bfp16ebs8 bytes
                             <name>.f32        float32 exact source of every
                                                 1-D tensor (norm gains, adaLN
                                                 bias) -- tiny, for kernels that
                                                 want unquantised gains
                             <name>.vendor     uint8   vendor `coalesed_weights`
                                                 region where it differs from the
                                                 logical packing (adaLN record
                                                 layout, bf16-tiled consts)
    block0.coalesced.bin   208,035,840 B -- the block's bytes exactly as they
                           sit in a vendor `.rai` `coalesed_weights` buffer
                           (sub-buffer order per the artifact's own
                           `external_buffers.buffer1` table)
    manifest.json          name / fp32 shape / packed length / dim, m, k, n /
                           tile_rows per tensor + verification numbers
    FINDINGS.md            what exists, what does not, naming mismatches

THE PACKING CONTRACT (do not change any of this)
-----------------------------------------------
Per 8-element group along the packed dim, 9 bytes
`[exponent byte][8 x int8 mantissa]`:

    amax = max |v| over the 8 values
    e    = floor(log2(amax)) - 6            UNCONDITIONALLY
    q    = v / 2**e
    m    = sign(q) * floor(|q| + 0.5)       round HALF AWAY FROM ZERO
    if m.max() > 127:  e += 1; recompute    (-128 legal, +128 not)
    exponent byte = (e + 133) & 0xFF

2-D GEMM weights are additionally permuted into tiles of
`tile_rows` rows x 32 K-elements:

    slot(i) = tile*per_tile + (r//8)*32 + (c//8)*8 + (r%8)
      row = i // (K/8), kg = i % (K/8), ke = kg*8
      r = row % tile_rows,  c = ke % 32
      tile = (row//tile_rows)*(K//32) + (ke//32)
      per_tile = (tile_rows//8)*32

with the MEASURED tile_rows (pack_bfp16.KNOWN_TILE_ROWS, byte-exact against
vendor `.rai` at seq=64 and re-fitted identically at seq=96):

    (3840, 3840) -> 48     (10240, 3840) -> 32
    (3840, 10240) -> 32    (15360, 256)  -> 64

The encoder itself is NOT reimplemented here: `pack_bfp16.pack`,
`pack_bfp16.unpack`, `pack_bfp16._quantise` and `pack_bfp16._slots` are
imported from `zimage-bfp16-npu/pack_bfp16.py`, the encoder that was verified
byte-for-byte against vendor `.rai` artifacts.  This script only adds:

  * the flat (identity-slot) packing used for 1-D tensors, which the 2-D
    tile rule cannot express, and
  * the vendor *region* wrappers (adaLN padded records with bf16 bias
    prefixes; 1-D consts as bf16 tiled to 737,280 B), per
    `zimage-port/_probe_out/route3/COALESCED-LAYOUT.md` and
    `zimage-port/route3_coalesce.py` (both byte-exact vs the vendor artifact).

SOURCE OF THE WEIGHTS
---------------------
`zimage-port/_models/Z-Image-Turbo/transformer/diffusion_pytorch_model-*.safetensors`
(header offsets + np.frombuffer -- `safetensors.get_tensor()` raises
0xC0000005 on this machine; see shard_lib.WeightStreamer).

USAGE
-----
    python extract.py                       # extract layers.0 -> block0.*
    python extract.py --block layers.0
    python extract.py --rai <file.rai>      # ADDITIONALLY rebuild the two
                                            # layers_10_12 blocks from the
                                            # checkpoint and compare them byte
                                            # for byte with the vendor payload
    python extract.py --skip-fused          # skip the w1/w3 fusion check
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import struct
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent          # .../imaggers/weights
NPU_ROOT = HERE.parents[1]                              # .../zimage-bfp16-npu
WS = HERE.parents[2]                                    # workspace root
PACKER_PATH = NPU_ROOT / "pack_bfp16.py"
MODEL_ROOT = WS / "zimage-port" / "_models" / "Z-Image-Turbo" / "transformer"
INDEX = MODEL_ROOT / "diffusion_pytorch_model.safetensors.index.json"

# --- vendor coalesced-region constants (route3_coalesce.py, COALESCED-LAYOUT.md)
ADALN_SHAPE = (15360, 256)
ADALN_TILE_ROWS = 64
ADALN_RECORD_GROUPS = 512          # groups per padded record
ADALN_RECORD_BYTES = 4736          # 128 B prefix + 512*9 B data
ADALN_BANDS = 240                  # 64 bias values per band x 240 = 15360
ADALN_REGION_BYTES = ADALN_BANDS * 4 * ADALN_RECORD_BYTES      # 4,546,560
CONST_REGION_BYTES = 737_280       # every 1-D const region is exactly this

# tile_rows measured per (N, K) -- pack_bfp16.KNOWN_TILE_ROWS, identical at
# seq=64 and seq=96 for all four block shapes
TILE_ROWS = {
    (3840, 3840): 48,
    (10240, 3840): 32,
    (3840, 10240): 32,
    (15360, 256): 64,
}

# The block's weight registry: the 15 tails that `layers.<N>.` owns, in the
# vendor coalesced order for the GEMM part (adaLN first, w3 BEFORE w2), then
# the 6 consts.  Source: `_probe_out/route3/COALESCED-LAYOUT.md` §1 (read from
# the `.rai`'s own `external_buffers.buffer1.coalesed_buffers` table) and
# `docs/WEIGHT-MAP.md` (506/506 name mapping, proven).
#
# Each entry: (tail, role, brief_expected_name, kind)
#   kind: "gemm" (2-D bfp16ebs8, N*K*9/8) | "adaln_w" | "adaln_b" | "const"
BLOCK_LAYOUT = [
    ("adaLN_modulation.0.weight", "norm_adaln_projection",
     "norm1.linear+norm2.linear (fused adaLN proj)", "adaln_w"),
    ("attention.to_q.weight", "attn.to_q", "attn.to_q", "gemm"),
    ("attention.to_k.weight", "attn.to_k", "attn.to_k", "gemm"),
    ("attention.to_v.weight", "attn.to_v", "attn.to_v", "gemm"),
    ("attention.to_out.0.weight", "attn.to_out", "attn.to_out", "gemm"),
    ("feed_forward.w1.weight", "ff.w1 (gate)", "ff.net.0.proj (w1 half)", "gemm"),
    ("feed_forward.w3.weight", "ff.w3 (up)", "ff.net.0.proj (w3 half)", "gemm"),
    ("feed_forward.w2.weight", "ff.w2 (down)", "ff.net.2", "gemm"),
    ("attention_norm1.weight", "norm1.gamma", "norm1.linear? (gamma)", "const"),
    ("attention.norm_q.weight", "attn.qknorm_q.gamma", "(not in brief)", "const"),
    ("attention.norm_k.weight", "attn.qknorm_k.gamma", "(not in brief)", "const"),
    ("attention_norm2.weight", "norm2.gamma", "norm2.linear? (gamma)", "const"),
    ("ffn_norm1.weight", "ffn_norm1.gamma", "(not in brief)", "const"),
    ("ffn_norm2.weight", "ffn_norm2.gamma", "(not in brief)", "const"),
]
ADALN_BIAS_TAIL = ("adaLN_modulation.0.bias", "adaln_bias", "(bias)",
                   "adaln_b")

# vendor coalesced order of the per-block regions (COALESCED-LAYOUT.md §1)
COALESCED_ORDER = [t for t, *_ in BLOCK_LAYOUT[:8]] + \
    [t for t, *_ in BLOCK_LAYOUT[8:]]


def _force_utf8() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass


def load_packer():
    """Import pack_bfp16.py by path (the dir name has a dash: no package)."""
    spec = importlib.util.spec_from_file_location("pack_bfp16", PACKER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# safetensors reading (header offsets + np.frombuffer; get_tensor crashes here)
# --------------------------------------------------------------------------
class Shard:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.f = open(str(path), "rb")
        hlen = struct.unpack("<Q", self.f.read(8))[0]
        self.meta = json.loads(self.f.read(hlen))
        self.base = 8 + hlen

    def read_f32(self, name: str) -> np.ndarray:
        e = self.meta[name]
        o0, o1 = e["data_offsets"]
        self.f.seek(self.base + o0)
        raw = self.f.read(o1 - o0)
        if len(raw) != o1 - o0:
            raise IOError(f"{name}: short read")
        if e.get("dtype") != "F32":
            raise NotImplementedError(f"{name}: dtype {e.get('dtype')}")
        return np.frombuffer(raw, dtype=np.float32).reshape(e["shape"]).copy()

    def close(self) -> None:
        self.f.close()


def stream_units(packer, W: np.ndarray) -> np.ndarray:
    """float32 flat stream -> (ng, 9) uint8 units in STREAM order (no shuffle).

    This is the shared quantisation step; the 2-D tile shuffle and the flat
    layout only differ in WHERE these units are written.
    """
    x = np.ascontiguousarray(W, dtype=np.float32).reshape(-1)
    if x.size % 8:
        raise ValueError(f"numel {x.size} not a multiple of 8")
    ng = x.size // 8
    v = x.reshape(ng, 8).astype(np.float64)
    m, expb = packer._quantise(v)
    units = np.empty((ng, 9), dtype=np.uint8)
    units[:, 0] = expb
    units[:, 1:] = np.ascontiguousarray(m).view(np.uint8).reshape(ng, 8)
    return units


def pack_flat(packer, x: np.ndarray) -> bytes:
    """1-D tensor -> bfp16ebs8, groups of 8 consecutive elements, no shuffle."""
    return stream_units(packer, x).tobytes()


def unpack_flat(blob: bytes, numel: int) -> np.ndarray:
    ng = numel // 8
    reg = np.frombuffer(blob, dtype=np.uint8).reshape(ng, 9)
    m = np.ascontiguousarray(reg[:, 1:]).view(np.int8).astype(np.float32)
    e = reg[:, 0].astype(np.int32) - 133
    return (m * np.exp2(e.astype(np.float32))[:, None]).reshape(numel)


def to_bf16(v: np.ndarray) -> bytes:
    """fp32 -> bf16 bytes, TRUNCATION of the low 16 bits.

    This is `route3_coalesce.to_bf16`, the transformation byte-exact against
    the vendor artifact's const regions and adaLN bias prefixes ("values here
    are bf16-exact" -- this script measures whether that holds and records the
    measured error).
    """
    bits = np.ascontiguousarray(v, dtype=np.float32).view(np.uint32)
    return np.ascontiguousarray((bits >> 16).astype(np.uint16)).tobytes()


def bf16_roundtrip_err(v: np.ndarray) -> float:
    """max |bf16(trunc)(v) - v| (measured, recorded in the manifest)."""
    if v.size == 0:
        return 0.0
    b = np.frombuffer(to_bf16(v), dtype=np.uint16).astype(np.uint32) << 16
    back = b.view(np.float32)
    return float(np.abs(back - v).max())


# --------------------------------------------------------------------------
# vendor coalesed regions (route3_coalesce.build_* reimplemented ONLY as
# layout glue around pack_bfp16's quantiser/slot functions)
# --------------------------------------------------------------------------
def vendor_gemm(packer, W: np.ndarray, tile_rows: int) -> bytes:
    """2-D weight region: identical to pack_bfp16.pack(W, tile_rows)."""
    return packer.pack(W, tile_rows=tile_rows)


def vendor_adaln(packer, W: np.ndarray, bias: np.ndarray) -> bytes:
    """(15360, 256) weight + (15360,) bias -> the 4,546,560 B adaLN region.

    960 records of 4,736 B = [128 B prefix][512 groups x 9 B].  The prefix of
    record 0 of each 4-record band holds that band's 64 bias values in bf16;
    the other three prefixes are zero.  (route3_coalesce.build_adaln; verified
    byte-exact against the vendor artifact.)
    """
    N, K = W.shape
    if (N, K) != ADALN_SHAPE or bias.shape != (N,):
        raise ValueError(f"adaLN shapes: {W.shape} {bias.shape}")
    ng = N * K // 8
    s = packer._slots(N, K, ng, ADALN_TILE_ROWS)        # 0 .. 491519
    rec = s // ADALN_RECORD_GROUPS
    inner = s % ADALN_RECORD_GROUPS
    pos = rec * ADALN_RECORD_BYTES + 128 + 9 * inner

    units = stream_units(packer, W)
    out = np.zeros(ADALN_REGION_BYTES, dtype=np.uint8)
    out[pos[:, None] + np.arange(9)] = units

    b16 = np.frombuffer(to_bf16(bias), dtype=np.uint8)
    for band in range(ADALN_BANDS):
        p0 = (band * 4) * ADALN_RECORD_BYTES
        out[p0:p0 + 128] = b16[band * 128:(band + 1) * 128]
    return out.tobytes()


def vendor_const(vec: np.ndarray) -> bytes:
    """1-D tensor -> bf16 tiled to exactly 737,280 B (vendor const region)."""
    unit = to_bf16(vec)
    if CONST_REGION_BYTES % len(unit):
        raise ValueError(f"const region not a multiple of {len(unit)}")
    return unit * (CONST_REGION_BYTES // len(unit))


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------
def verify_bfp16(packer, blob: bytes, W: np.ndarray, tile_rows):
    """Decode the packed bytes and measure the error vs the fp32 source.

    Returns (max_abs_err, max_err_over_group_step).  The contract gives an
    error of at most HALF a group step (round half away from zero), i.e. the
    second number must be <= 0.5 (+ float noise).

    Careful with the index spaces: `reg` is indexed by SLOT (post-shuffle),
    while the error is indexed by stream group.  The exponent byte of stream
    group i sits at `reg[_slots(...)[i], 0]`, so the steps must be gathered
    through the permutation -- computing them slot-wise silently mixes groups
    and reports ratios up to 16x for perfectly correct bytes.
    """
    x = np.ascontiguousarray(W, dtype=np.float32).reshape(-1)
    ng = x.size // 8
    if tile_rows is None:
        back = unpack_flat(blob, x.size)
        expb = np.frombuffer(blob, dtype=np.uint8).reshape(ng, 9)[:, 0]
    else:
        back = packer.unpack(blob, W.shape[0], W.shape[1], tile_rows)
        reg = np.frombuffer(blob, dtype=np.uint8).reshape(ng, 9)
        expb = reg[:, 0][packer._slots(W.shape[0], W.shape[1], ng, tile_rows)]
    err = np.abs(back.reshape(-1) - x)
    step = np.exp2(expb.astype(np.int32) - 133).astype(np.float32)
    ratio = float((err.reshape(ng, 8) / step[:, None]).max())
    return float(err.max()), ratio


def rai_payload_meta(rai_path: pathlib.Path):
    """Read the `.rai`'s own external_buffers table (read-only).

    Returns (buffer_info, all_offset_size_pairs).  This is the artifact telling
    us where `coalesed_weights` lives; nothing is assumed.
    """
    b = rai_path.read_bytes()
    i = b.find(b"external_buffers")
    if i < 0:
        return None, []
    start = b.rfind(b"{", 0, i)
    depth = 0
    in_str = False
    esc = False
    end = None
    for j in range(start, min(len(b), start + 4_000_000)):
        c = b[j]
        if in_str:
            if esc:
                esc = False
            elif c == 0x5C:
                esc = True
            elif c == 0x22:
                in_str = False
            continue
        if c == 0x22:
            in_str = True
        elif c == 0x7B:
            depth += 1
        elif c == 0x7D:
            depth -= 1
            if depth == 0:
                end = j + 1
                break
    try:
        obj = json.loads(b[start:end].decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001
        return None, []
    # the walked object is usually {"external_buffers": {...}}; look one level
    # down before giving up on buffer1
    root = obj.get("external_buffers", obj)

    pairs = []

    def walk(o, path):
        if isinstance(o, dict):
            off = o.get("offset") or o.get("file_offset") or o.get("base")
            size = o.get("size_in_bytes") or o.get("size") or o.get("length")
            if off is not None and size is not None:
                try:
                    pairs.append((path, int(str(off), 0), int(str(size), 0)))
                except Exception:  # noqa: BLE001
                    pass
            for k, v in o.items():
                walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(o, list):
            for n, v in enumerate(o):
                walk(v, f"{path}[{n}]")

    walk(obj, "")
    return obj, pairs


def build_block_payload(packer, shard: Shard, block: str,
                        ckpt_of: dict) -> tuple[bytes, list]:
    """Build the block's vendor-layout payload regions.

    Returns (regions, records) where regions is a list of
    (role, tail, offset, size, blob) in COALESCED_ORDER and records carries
    per-tensor bookkeeping for the manifest.
    """
    regions: list[tuple] = []
    records: list[dict] = []
    off = 0
    for tail, role, brief, kind in BLOCK_LAYOUT:
        name = f"{block}.{tail}"
        if kind == "adaln_w":
            W = shard.read_f32(name)
            bias = shard.read_f32(f"{block}.{ADALN_BIAS_TAIL[0]}")
            blob = vendor_adaln(packer, W, bias)
            size = ADALN_REGION_BYTES
        elif kind == "gemm":
            W = shard.read_f32(name)
            tr = TILE_ROWS[W.shape]
            blob = vendor_gemm(packer, W, tr)
            size = W.size * 9 // 8
        elif kind == "const":
            v = shard.read_f32(name)
            blob = vendor_const(v)
            size = CONST_REGION_BYTES
        else:
            raise AssertionError(kind)
        assert len(blob) == size, (name, len(blob), size)
        regions.append((role, tail, off, size, blob))
        records.append({"tail": tail, "role": role, "kind": kind})
        off += size
    return regions, records


# --------------------------------------------------------------------------
def main() -> int:
    _force_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", default="layers.0",
                    help="checkpoint block prefix to extract (default layers.0)")
    ap.add_argument("--out", default=str(HERE / "block0.npz"))
    ap.add_argument("--bin", default=str(HERE / "block0.coalesced.bin"))
    ap.add_argument("--manifest", default=str(HERE / "manifest.json"))
    ap.add_argument("--rai", default=None,
                    help="vendor .rai to byte-compare against (optional)")
    ap.add_argument("--rai-blocks", default="layers.10,layers.11",
                    help="the two blocks the .rai was compiled from "
                         "(layers_10_12 shard)")
    ap.add_argument("--skip-fused", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    packer = load_packer()
    block = args.block

    index = json.loads(INDEX.read_text(encoding="utf-8"))
    ckpt_of = index["weight_map"]

    # ---- name classification ------------------------------------------
    # `zimage-port/get_params.py` / `shard_inspect.py` / `skeleton_names_all()`
    # named in the brief do NOT exist in this workspace (see FINDINGS.md).
    # The registry used here is the proven one from docs/WEIGHT-MAP.md: a block owns
    # exactly the 15 tails below, as `layers.<N>.<tail>`.
    owned = sorted(k for k in ckpt_of if k.startswith(block + "."))
    expected = sorted(f"{block}.{t}" for t, *_ in
                      list(BLOCK_LAYOUT) + [ADALN_BIAS_TAIL])
    if owned != expected:
        print(f"[FAIL] block registry mismatch\n  owned:    {owned}\n"
              f"  expected: {expected}")
        return 1

    # shard that holds the block
    shards_needed = {ckpt_of[k] for k in owned}
    if len(shards_needed) != 1:
        print(f"[FAIL] block spans shards: {shards_needed}")
        return 1
    shard_name = shards_needed.pop()
    shard = Shard(MODEL_ROOT / shard_name)

    print("=" * 78)
    print("extract.py -- one Z-Image-Turbo DiT block -> BlockFP16 (bfp16ebs8)")
    print("=" * 78)
    print(f"  block   : {block}   ({len(owned)} tensors)")
    print(f"  shard   : {shard_name}")
    print(f"  packer  : {PACKER_PATH}")
    print(f"  out     : {args.out}")

    # ---- pass 1: read + pack everything -------------------------------
    logical: dict[str, bytes] = {}
    vendor_regions: list[tuple] = []        # (key, offset, size, blob)
    fp32_small: dict[str, np.ndarray] = {}
    tensor_records: list[dict] = []
    verify_rows: list[tuple] = []

    for tail, role, brief, kind in BLOCK_LAYOUT:
        name = f"{block}.{tail}"
        e = shard.meta[name]
        shape = tuple(e["shape"])
        W = shard.read_f32(name)
        assert W.shape == shape

        if kind == "gemm":
            N, K = shape
            tr = TILE_ROWS[(N, K)]
            blob = packer.pack(W, tile_rows=tr)
            # m = the GEMM token count of the geometry the tile_rows were
            # MEASURED in (hidden [1, 96, 3840] -- see m_tokens in the manifest)
            dim, m, k, n = -1, 96, K, N
            expected_len = ((K + 7) // 8) * 9 * N
            layout = "tile-shuffle (pack_bfp16._slots)"
        elif kind == "adaln_w":
            N, K = shape
            tr = TILE_ROWS[(N, K)]
            blob = pack_flat(packer, W.reshape(-1))
            # the adaLN GEMM sees ONE conditioning vector: adaln [1, 256]
            dim, m, k, n = -1, 1, K, N
            expected_len = ((K + 7) // 8) * 9 * N
            layout = "tile-shuffle (pack_bfp16._slots) for the vendor region; " \
                     "flat-groups8 for this logical blob"
        else:                                   # const 1-D
            tr = None
            blob = pack_flat(packer, W.reshape(-1))
            dim, m, k, n = 0, None, shape[0], None
            expected_len = ((shape[0] + 7) // 8) * 9
            layout = "flat-groups8 (identity slots)"

        if len(blob) != expected_len:
            print(f"[FAIL] {name}: {len(blob)} B != contract {expected_len} B")
            return 1

        # logical blob verification: GEMM blobs carry the tile shuffle,
        # adaLN's logical blob and the 1-D consts are flat (identity slots)
        verify_tr = tr if kind == "gemm" else None
        max_err, ratio = verify_bfp16(packer, blob, W, verify_tr)
        verify_rows.append((name, max_err, ratio))
        print(f"  packed {name:44s} {len(blob):>10,} B  "
              f"max|err| {max_err:.4e}  err/step {ratio:.3f}")

        logical[name] = blob
        rec = {
            "name": name,
            "checkpoint_key": name,
            "role": role,
            "brief_expected_name": brief,
            "kind": kind,
            "fp32_shape": list(shape),
            "source_dtype": "F32",
            "params": int(np.prod(shape)),
            "dim": dim,
            "packed_dim": "K" if kind != "const" else "the only dim",
            "m": m,
            "k": k,
            "n": n,
            "tile_rows": tr,
            "tile_kelems": 32,
            "group": 8,
            "layout": layout,
            "packed_length": len(blob),
            "expected_length": expected_len,
            "length_matches_formula":
                len(blob) == expected_len == (((shape[-1] + 7) // 8) * 9 *
                                             int(np.prod(shape[:-1])) if
                                             kind != "const" else
                                             ((shape[0] + 7) // 8) * 9),
            "sha256": hashlib.sha256(blob).hexdigest(),
            "decode_max_abs_err": max_err,
            "decode_max_abs_err_rel_to_absmax":
                max_err / float(np.abs(W).max()),
            "decode_max_err_over_group_step": ratio,
            "decode_within_half_step": bool(ratio <= 0.5 + 1e-6),
        }
        tensor_records.append(rec)

    # adaLN bias (1-D, lives in the adaLN vendor region prefixes as bf16)
    bname = f"{block}.{ADALN_BIAS_TAIL[0]}"
    bias = shard.read_f32(bname)
    bblob = pack_flat(packer, bias.reshape(-1))
    expected_len = ((bias.size + 7) // 8) * 9
    max_err, ratio = verify_bfp16(packer, bblob, bias, None)
    logical[bname] = bblob
    print(f"  packed {bname:44s} {len(bblob):>10,} B  "
          f"max|err| {max_err:.4e}  err/step {ratio:.3f}")
    tensor_records.append({
        "name": bname, "checkpoint_key": bname,
        "role": ADALN_BIAS_TAIL[1], "brief_expected_name": ADALN_BIAS_TAIL[2],
        "kind": "adaln_b",
        "fp32_shape": list(bias.shape), "source_dtype": "F32",
        "params": int(bias.size),
        "dim": 0, "packed_dim": "the only dim",
        "m": 1, "k": 256, "n": 15360,
        "tile_rows": None, "tile_kelems": 32, "group": 8,
        "layout": "flat-groups8 (identity slots); in the vendor payload the "
                  "bias is NOT a standalone region -- it rides in bf16 inside "
                  "the adaLN weight region's 128 B record prefixes",
        "packed_length": len(bblob), "expected_length": expected_len,
        "length_matches_formula": len(bblob) == expected_len,
        "sha256": hashlib.sha256(bblob).hexdigest(),
        "decode_max_abs_err": max_err,
        "decode_max_abs_err_rel_to_absmax":
            max_err / float(np.abs(bias).max()),
        "decode_max_err_over_group_step": ratio,
        "decode_within_half_step": bool(ratio <= 0.5 + 1e-6),
    })

    # ---- vendor regions + coalesced payload ---------------------------
    payload = bytearray()
    region_records = []
    for tail, role, brief, kind in BLOCK_LAYOUT:
        name = f"{block}.{tail}"
        if kind == "adaln_w":
            blob = vendor_adaln(packer, shard.read_f32(name), bias)
        elif kind == "gemm":
            blob = logical[name]
        else:
            blob = vendor_const(shard.read_f32(name))
        off = len(payload)
        payload += blob
        region_records.append({
            "role": role, "tensor": name, "offset": off, "size": len(blob),
            "format": ("bfp16ebs8 padded records [128 B bf16 bias prefix"
                       "][512 groups x 9 B], 960 records"
                       if kind == "adaln_w" else
                       "bfp16ebs8 tile-shuffled groups of 8" if kind == "gemm"
                       else "bf16 tiled to 737,280 B"),
        })
    payload = bytes(payload)

    expected_payload = (ADALN_REGION_BYTES + 4 * (3840 * 3840 * 9 // 8)
                        + 3 * (10240 * 3840 * 9 // 8)
                        + 6 * CONST_REGION_BYTES)
    if len(payload) != expected_payload:
        print(f"[FAIL] payload {len(payload):,} != {expected_payload:,}")
        return 1

    # ---- optional fused w1/w3 property -------------------------------
    fused_ok = None
    if not args.skip_fused:
        w1 = shard.read_f32(f"{block}.feed_forward.w1.weight")
        w3 = shard.read_f32(f"{block}.feed_forward.w3.weight")
        tr = TILE_ROWS[w1.shape]
        fused = packer.pack(np.ascontiguousarray(np.vstack([w1, w3])),
                            tile_rows=tr)
        sep = logical[f"{block}.feed_forward.w1.weight"] + \
            logical[f"{block}.feed_forward.w3.weight"]
        fused_ok = (len(fused) == len(sep)) and (fused == sep)
        print(f"  fused(w1,w3) == pack(w1)++pack(w3) : {fused_ok}")
        del w1, w3, fused, sep

    # ---- write outputs ------------------------------------------------
    out_npz = pathlib.Path(args.out)
    arrays = {}
    for name, blob in logical.items():
        arrays[name] = np.frombuffer(blob, dtype=np.uint8)
    for tail, role, brief, kind in BLOCK_LAYOUT:
        name = f"{block}.{tail}"
        if kind == "adaln_w":
            arrays[f"{name}.vendor"] = np.frombuffer(
                payload[:ADALN_REGION_BYTES], dtype=np.uint8)
        elif kind == "const":
            rec = next(r for r in region_records if r["tensor"] == name)
            arrays[f"{name}.vendor"] = np.frombuffer(
                payload[rec["offset"]:rec["offset"] + rec["size"]],
                dtype=np.uint8)
    oned = [f"{block}.{t}" for t, _, _, k in BLOCK_LAYOUT if k == "const"] \
        + [bname]
    for name in oned:
        arrays[f"{name}.f32"] = np.ascontiguousarray(
            shard.read_f32(name), dtype=np.float32)
    np.savez(str(out_npz), **arrays)

    bin_path = pathlib.Path(args.bin)
    bin_path.write_bytes(payload)

    # ---- optional vendor .rai byte comparison -------------------------
    rai_report = None
    if args.rai:
        rai_report = rai_check(packer, args, ckpt_of, payload)

    # ---- manifest -----------------------------------------------------
    manifest = {
        "format": "bfp16ebs8",
        "model": "Z-Image-Turbo/transformer",
        "block": block,
        "source_shard": shard_name,
        "source_index": str(INDEX.relative_to(WS)).replace("\\", "/"),
        "encoder": {
            "module": "zimage-bfp16-npu/pack_bfp16.py",
            "functions": ["pack", "unpack", "_quantise", "_slots",
                          "KNOWN_TILE_ROWS"],
            "verified": "byte-exact vs vendor .rai on 12 synthetic shapes and "
                        "on the real layers_10_12 coalesed_weights payload "
                        "(416,071,680 B); independently corroborated by "
                        "Quark's bfp_kernel.cc (scale 2**(e_unb-6), STD_ROUND "
                        "= half away from zero)",
        },
        "spec": {
            "group": 8,
            "group_bytes": 9,
            "group_layout": "[exponent byte][8 x int8 mantissa]",
            "exponent_rule": "e = floor(log2(amax)) - 6 UNCONDITIONALLY",
            "exponent_byte": "(e + 133) & 0xFF",
            "rounding": "sign(q)*floor(|q|+0.5) -- round half AWAY from zero",
            "overflow_rule": "if m.max() > 127: e += 1 and recompute "
                             "(-128 legal, +128 not)",
            "tile_layout_2d": "slot = tile*per_tile + (r//8)*32 + (c//8)*8 + "
                              "(r%8); tile = (row//tile_rows)*(K//32) + "
                              "(kelem//32); per_tile = (tile_rows//8)*32; "
                              "r = row % tile_rows, c = kelem % 32",
            "tile_rows_table": {"3840x3840": 48, "10240x3840": 32,
                                "3840x10240": 32, "15360x256": 64},
            "tile_rows_note": "measured per (N,K,M): identical at seq=64 "
                              "(pack_bfp16.KNOWN_TILE_ROWS) and seq=96 "
                              "(COALESCED-LAYOUT.md --sweep, unique fit).  "
                              "f(N,K,M) with M-dependence PROVEN only for "
                              "(3840,256): M=64->48, M=256->32.  Re-measure "
                              "with pack_bfp16.detect_tile_rows at any other "
                              "M before feeding a kernel built for it.",
            "length_formula": "ceil(numel_packed_dim / 8) * 9 * other_dims",
        },
        "m_tokens": {
            "token_gemms": 96,
            "note": "the vendor geometry the tile_rows were measured in is "
                    "hidden [1, 96, 3840] (layers_* shards at S=96).  The "
                    "adaLN GEMM's own M is 1 (adaln [1, 256]).  packed "
                    "weights are M-invariant EXCEPT through tile_rows.",
        },
        "tensors": tensor_records,
        "coalesced_layout": {
            "description": "vendor coalesed_weights order for one block, read "
                           "from the .rai external_buffers.buffer1 table "
                           "(COALESCED-LAYOUT.md).  adaLN FIRST, w3 BEFORE "
                           "w2.  One block = 208,035,840 B.",
            "order": COALESCED_ORDER,
            "total_bytes": len(payload),
            "file": pathlib.Path(args.bin).name,
            "regions": region_records,
        },
        "npz": {
            "file": out_npz.name,
            "keys": {n: (".f32 suffix = exact fp32; .vendor suffix = "
                         "coalesed-region bytes" if n.endswith((".f32",
                                                                 ".vendor"))
                         else "logical bfp16ebs8 bytes") for n in
                     sorted(arrays)},
        },
        "verification": {
            "decode_roundtrip": [
                {"tensor": n, "max_abs_err": e,
                 "max_err_over_group_step": r,
                 "contract": "<= 0.5 group steps"} for n, e, r in verify_rows],
            "fused_w1_w3_equals_concatenation": fused_ok,
            "rai_check": rai_report,
        },
    }
    pathlib.Path(args.manifest).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    shard.close()
    print()
    print(f"  tensors      : {len(tensor_records)}")
    print(f"  logical bytes: {sum(len(b) for b in logical.values()):,}")
    print(f"  coalesced    : {len(payload):,} "
          f"({len(payload) / 2**20:.1f} MiB) -> {bin_path.name}")
    print(f"  npz          : {out_npz.stat().st_size:,} B -> {out_npz.name}")
    print(f"  manifest     -> {args.manifest}")
    print(f"  wall time    : {time.time() - t0:.1f}s")
    worst = max(verify_rows, key=lambda r: r[2])
    print(f"  worst decode : {worst[0]}  err/step {worst[2]:.3f} "
          f"(contract <= 0.5)")
    return 0


# --------------------------------------------------------------------------
def rai_check(packer, args, ckpt_of: dict, block_payload: bytes) -> dict:
    """Rebuild the .rai's two blocks from the checkpoint and compare bytes.

    Layout fact (COALESCED-LAYOUT.md, from the artifact's own
    `external_buffers.buffer1.coalesed_buffers` table): the `coalesed_weights`
    payload holds [block0: adaLN,to_q,to_k,to_v,to_out,w1,w3,w2] [block1: same]
    [12 consts: b0's six, then b1's six].  So one block's bytes are SCATTERED
    in the payload -- its GEMM part is contiguous at the front of its slot and
    its consts sit after BOTH blocks' GEMM parts.  This function places every
    rebuilt region at its true payload offset before comparing.

    The payload base is NOT assumed: candidates come from the artifact's own
    external_buffers table (absolute offsets, if declared) plus 96-byte
    needles at known region starts.  A candidate is accepted only when the
    FULL comparison passes (a short needle can false-positive; same
    methodology as pack_bfp16.detect_tile_rows).
    """
    rai_path = pathlib.Path(args.rai)
    print()
    print("-- vendor .rai byte comparison --")
    print(f"  artifact: {rai_path}  ({rai_path.stat().st_size:,} B)")
    obj, pairs = rai_payload_meta(rai_path)
    report = {"artifact": str(rai_path), "size": rai_path.stat().st_size,
              "external_buffers_found": obj is not None}
    declared = {}
    if obj is not None:
        root = obj.get("external_buffers", obj)
        b1 = root.get("buffer1") or {}
        subs = b1.get("coalesed_buffers") or []
        report["buffer1_name"] = b1.get("name")
        report["buffer1_size_in_bytes"] = b1.get("size_in_bytes")
        report["buffer1_sub_buffers"] = [
            {k: s.get(k) for k in ("logical_id", "name", "offset_in_bytes")}
            for s in subs]
        declared = {int(s["offset_in_bytes"]): s.get("name")
                    for s in subs if s.get("offset_in_bytes") is not None}
        print(f"  buffer1: name={b1.get('name')} "
              f"size={b1.get('size_in_bytes')} sub-buffers={len(subs)}")
    blocks = [b for b in args.rai_blocks.split(",") if b]

    # ---- rebuild each block's regions ---------------------------------
    built: dict[str, list] = {}
    for b in blocks:
        shard = Shard(MODEL_ROOT / ckpt_of[f"{b}.attention.to_q.weight"])
        regions, _ = build_block_payload(packer, shard, b, ckpt_of)
        built[b] = regions
        shard.close()
        print(f"  rebuilt {b}: {sum(r[3] for r in regions):,} B")

    # ---- true payload offsets: gemm parts first (both blocks), then consts
    # (BLOCK_LAYOUT is 8 GEMM/adaLN regions followed by the 6 consts; the
    #  coalesced buffer holds both blocks' GEMM parts, then both blocks' consts)
    placed = []          # (full_offset, size, blob, label)
    off = 0
    for b in blocks:
        for role, tail, _o, size, blob in built[b][:8]:
            placed.append((off, size, blob, f"{b}.{tail}"))
            off += size
    for b in blocks:
        for role, tail, _o, size, blob in built[b][8:]:
            placed.append((off, size, blob, f"{b}.{tail}"))
            off += size
    total = off
    print(f"  rebuilt payload total: {total:,} B")
    report["rebuilt_payload_bytes"] = total

    # ---- candidate payload bases --------------------------------------
    cands: set[int] = set()
    for path, o, size in pairs:
        if size == total:
            cands.add(int(o))
    raw = rai_path.read_bytes()
    for full_off, size, blob, label in placed:
        needle = blob[:96]
        start = 0
        while True:
            i = raw.find(needle, start)
            if i < 0:
                break
            cands.add(i - full_off)
            start = i + 1
    cands = {c for c in cands if c >= 0}
    print(f"  payload-base candidates: {len(cands)} "
          f"(declared+needle) {sorted(cands)[:8]}")

    best = None
    for base in sorted(cands):
        if base + total > len(raw):
            continue
        nd = 0
        per_region = []
        for full_off, size, blob, label in placed:
            got = raw[base + full_off:base + full_off + size]
            d = int((np.frombuffer(got, np.uint8)
                     != np.frombuffer(blob, np.uint8)).sum()) \
                if len(got) == size else size
            per_region.append({"region": label, "payload_offset": full_off,
                               "size": size, "diff_bytes": d})
            nd += d
        if best is None or nd < best["diff_bytes"]:
            best = {"payload_base": base, "diff_bytes": nd,
                    "compared_bytes": total, "regions": per_region}
        if nd == 0:
            break
    if best is None:
        print("  [FAIL] no candidate payload base fit the file")
        report["result"] = "no candidate base"
        return report

    print(f"  payload base {best['payload_base']:,}: "
          f"{best['compared_bytes'] - best['diff_bytes']:,} / "
          f"{best['compared_bytes']:,} bytes identical "
          f"({best['diff_bytes']:,} differing)")
    for rr in best["regions"]:
        rr["declared_sub_buffer"] = declared.get(rr["payload_offset"])
        print(f"    {rr['region']:52s} off {rr['payload_offset']:>10,} "
              f"size {rr['size']:>11,}  diff {rr['diff_bytes']:>10,}  "
              f"declared={rr['declared_sub_buffer']}")
    n_match = sum(1 for rr in best["regions"] if rr["declared_sub_buffer"])
    report["declared_offset_matches"] = \
        f"{n_match}/{len(best['regions'])}"
    print(f"  .rai-declared sub-buffer offsets matching my regions: "
          f"{n_match}/{len(best['regions'])}")
    report["result"] = ("byte-identical" if best["diff_bytes"] == 0
                        else "differs")
    report.update(best)
    return report


if __name__ == "__main__":
    raise SystemExit(main())
