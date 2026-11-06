# SPDX-License-Identifier: MIT
"""t3d_codec.py -- the HOST CODECS the T3d loop needs, in PURE NUMPY.

Why this file exists: t3_real.py's device stage runs on
<python310> (3.10), which has numpy but
NOT the MLIR-AIE "aie" package.  Until now every bfp16ebs8 / blocked-layout pack
happened in the py3.12 prep leg, one dispatch per process.  An in-loop image run
packages the payloads BETWEEN dispatches, so the codecs have to exist on the
py3.10 side.

Three codecs, each with a byte-equality gate against the library object it
replaces (t3d_codec_gate.py, run on py3.12):

  1. bf16_rne(x)          -> uint16       == np.asarray(ml_dtypes.bfloat16(x))
  2. bfp_encode(x)        -> uint8        == aie.utils.bfp.encode(x)  ("floor")
  3. pack_blocked(x)      -> bytes        == aie.iron.kernels.mm_bfp(...)
                                             .contract.layouts[2].pack(x)
     unpack_blocked(b)    -> float32      == ... .layouts[2].unpack(b)
  4. pack_b_nk(x)         -> uint8        == t3b_ip_bfp_run.pack_b(x)

pack_blocked/unpack_blocked take a PERMUTATION TABLE extracted from the library
itself (t3d_perm_96x128.npy), so the layout is not re-derived by hand: the
extraction (t3d_codec_gate.py --emit-perm) proves the table is a permutation and
that layouts[0] (A, dim_k=128) and layouts[2] (C) are the SAME permutation at
(96,128) -- the identity step 3's whole design rests on.

Nothing here decides arithmetic: every one of these is a permutation or the
library's own encoder, ported.
"""
from __future__ import annotations

import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PERM_FILE = HERE / "t3d_perm_96x128.npy"
BLOCK = 8
BLOCK_BYTES = 9
_MANTISSA_SHIFT = 23 - 7 + 1


# ---------------------------------------------------------------------------
# 1. bf16 round-to-nearest-even, numpy only
# ---------------------------------------------------------------------------
def bf16_rne(x) -> np.ndarray:
    """float32 -> the uint16 bf16 bit pattern, RNE (what bfloat16() does)."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    r = ((u >> 16) & 1).astype(np.uint32) + np.uint32(0x7FFF)
    return ((u + r) >> 16).astype(np.uint16)


def bf16_bytes(x) -> bytes:
    return bf16_rne(x).tobytes()


def bf16_to_f32(b) -> np.ndarray:
    u = np.frombuffer(bytes(b), np.uint16)
    return (u.astype(np.uint32) << 16).view(np.float32)


# ---------------------------------------------------------------------------
# 2. bfp16ebs8 encode -- a numpy port of aie.utils.bfp.encode(rounding="floor")
# ---------------------------------------------------------------------------
def bfp_encode(x) -> np.ndarray:
    """float32 (..., n) with n % 8 == 0 -> uint8 (..., n*9//8).

    Port of aie/utils/bfp.py: blocks along the LAST axis; one shared exponent
    (the block max); each mantissa truncated toward negative infinity in two's
    complement; anything more than 31 binades down becomes 0 / -1.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    n = x.shape[-1]
    if n % BLOCK:
        raise ValueError(f"bfp_encode: last axis {n} is not a multiple of {BLOCK}")
    if not np.isfinite(x).all():
        raise ValueError("bfp_encode: inputs must be finite")
    lead = x.shape[:-1]
    bits = x.view(np.uint32).reshape(*lead, n // BLOCK, BLOCK)
    sign = (bits >> 31).astype(bool)
    exp = (bits >> 23) & 0xFF
    mant = (bits & 0x7FFFFF) | np.where(exp != 0, np.uint32(0x800000),
                                        np.uint32(0))
    max_exp = exp.max(axis=-1, keepdims=True)
    shift = (max_exp - exp).astype(np.int64)
    far = shift >= 32
    m32 = np.where(sign, (-mant.astype(np.int64)) & 0xFFFFFFFF,
                   mant).astype(np.uint32)
    v = (((m32 >> _MANTISSA_SHIFT) & 0xFF).astype(np.uint8).view(np.int8)
         .astype(np.int32))
    v = np.right_shift(v, np.minimum(shift, 31).astype(np.int32))
    blk = np.empty(lead + (n // BLOCK,), dtype=np.dtype(
        [("exponent", np.uint8), ("mantissas", np.int8, (BLOCK,))]))
    blk["exponent"] = max_exp[..., 0]
    blk["mantissas"] = np.where(far, np.where(sign, -1, 0), v)
    return blk.view(np.uint8).reshape(*lead, n // BLOCK * BLOCK_BYTES)


def bfp_decode(b) -> np.ndarray:
    """uint8 (..., n*9//8) -> float32 (..., n): the exact inverse map of a block.

    Port of aie/utils/bfp.py decode: value = mantissa * 2**(exponent - 127 - 6).
    Used only by closure gates (what did the device actually multiply?), never to
    produce a payload.
    """
    b = np.ascontiguousarray(b, dtype=np.uint8)
    nb = b.shape[-1]
    if nb % BLOCK_BYTES:
        raise ValueError(f"bfp_decode: last axis {nb} is not a multiple of "
                         f"{BLOCK_BYTES}")
    lead = b.shape[:-1]
    dt = np.dtype([("exponent", np.uint8), ("mantissas", np.int8, (BLOCK,))])
    blk = b.view(dt)
    scale = np.ldexp(1.0, blk["exponent"].astype(np.int32) - 127 - 6)
    vals = blk["mantissas"].astype(np.float64) * scale[..., None]
    return vals.astype(np.float32).reshape(*lead, nb // BLOCK_BYTES * BLOCK)


def unpack_b_nk(b, calls: int, n: int, k: int) -> np.ndarray:
    """The inverse of pack_b_nk: uint8 -> (calls, n, k) float32."""
    nb, kb = n // BLOCK, k // BLOCK
    flat = np.ascontiguousarray(b, np.uint8).reshape(
        calls, nb * kb, BLOCK * BLOCK_BYTES)
    dec = bfp_decode(flat)                          # (calls, nb*kb, 8 n x 8 k)
    dec = dec.reshape(calls, nb, kb, BLOCK, BLOCK)  # (nb, kb, n8, k8)
    return np.ascontiguousarray(dec.transpose(0, 1, 3, 2, 4)).reshape(calls, n, k)


def pack_b_nk(calls_nk: np.ndarray) -> np.ndarray:
    """(calls, n, k) float32 -> uint8 (calls, n*k//8*9): the B image mm_bfp reads.

    Same block order as t3b_ip_bfp_run.pack_b (the codec behind the gated chain):
    per call the (n, k) matrix is cut into (8n x 8k) tiles in (t, s) order and
    each tile is 8 groups of 8 k-values, one 9-byte block per group.
    """
    x = np.ascontiguousarray(calls_nk, np.float32)
    calls, n, k = x.shape
    nb, kb = n // BLOCK, k // BLOCK
    tiles = x.reshape(calls, nb, BLOCK, kb, BLOCK).transpose(0, 1, 3, 2, 4)
    tiles = np.ascontiguousarray(tiles).reshape(calls * nb * kb, BLOCK, BLOCK)
    enc = bfp_encode(tiles)                      # (calls*nb*kb, 72)
    return enc.reshape(calls, nb * kb * BLOCK * BLOCK_BYTES)


# ---------------------------------------------------------------------------
# 3. the (96,128) blocked layout: a permutation, extracted from the library
# ---------------------------------------------------------------------------
def load_perm(path=None) -> np.ndarray:
    """The (96,128) blocked-layout permutation, as EXTRACTED FROM THE LIBRARY.

    Two committed forms, same numbers: t3d_perm_96x128.json (git-tracked, what
    the py3.10 loop reads) and t3d_perm_96x128.npy (the py3.12 gate's scratch,
    git-ignored because *.npy is).  The permutation is a property of the
    installed toolchain's layout codec, not of this design, so it is extracted
    by t3d_codec_gate.py and gated there against lay.pack byte for byte.
    """
    p = pathlib.Path(path) if path else None
    if p is None:
        j = HERE / "t3d_perm_96x128.json"
        if j.exists():
            import json as _json
            return np.asarray(_json.loads(j.read_text())["perm"], np.int64)
        p = PERM_FILE
    if p.suffix == ".json":
        import json as _json
        return np.asarray(_json.loads(p.read_text())["perm"], np.int64)
    return np.load(p).astype(np.int64)


def load_perm_inv(path=None) -> np.ndarray:
    return np.argsort(load_perm(path)).astype(np.int64)


def pack_blocked(x2d, perm) -> bytes:
    """logical (rows, m*n) float32 -> the blocked bf16 bytes the object holds.

    The layout is a permutation of the LAST axis applied per row (the library's
    own lay.pack does exactly this: (1, 12288) -> (1, 12288))."""
    x = np.ascontiguousarray(x2d, np.float32)
    return bf16_rne(x[..., perm]).tobytes()


def unpack_blocked(raw, inv_perm, shape) -> np.ndarray:
    """blocked bf16 bytes -> logical float32 of the given (rows, m*n) shape."""
    f = bf16_to_f32(raw).reshape(shape[0], -1)
    return np.ascontiguousarray(f[:, inv_perm]).reshape(shape)


# ---------------------------------------------------------------------------
# 4. the projection's X image: [g][s][m rows][k] bf16
# ---------------------------------------------------------------------------
def x_image(x_mk: np.ndarray, k: int = 8) -> bytes:
    """(m, K) float32 -> the proj core's X object bytes (NG * BK*m*k*2)."""
    m, K = x_mk.shape
    NG = K // (k * BLOCK)
    a = np.ascontiguousarray(
        np.asarray(x_mk, np.float32).reshape(m, K // k, k).transpose(1, 0, 2))
    return bf16_rne(a).tobytes()
