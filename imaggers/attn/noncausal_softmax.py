# SPDX-License-Identifier: MIT
"""noncausal_softmax.py -- numpy reference + byte-level kernel contract.

Bidirectional (NON-causal) attention softmax for the Z-Image DiT block on
AMD XDNA2 (AIE2P), as needed by a 30-head / head_dim-128 / 96-token block.
This is a DESIGN artifact: a runnable numpy reference and the exact byte
contract an AIE kernel call must satisfy.  Nothing here compiles or touches
the NPU.

Why this file exists
--------------------
``mha_softmax`` (``_pylibs/mlir_aie/python/aie/iron/kernels/linalg.py:1200``)
masks CAUSALLY (its reference keeps ``cols <= rows`` on the diagonal block,
linalg.py:1276-1277, and skips key blocks past the query block, linalg.py:1278).
A diffusion DiT needs every token to attend to every other token.

The unmasked building block ALREADY exists in C++:
``softmax_rows_bf16`` in
``_pylibs/mlir_aie/include/aie_kernels/activation/softmax_aie2p.h:222-229`` --
``softmax_rows_causal_bf16`` (same header, 232-241) minus its single mask
call at line 237.  XDNA2 compiles exactly that header
(``activation/softmax.cc:10-14`` selects ``softmax_aie2p.h`` unless
``AIE_TUNED_AIE2``).  What is missing is the python binding (the factory
``activation.py:376`` binds only ``softmax_bf16``, one flat vector per call)
and this file pins down the tile/byte contract such a binding must declare.

Reference semantics
-------------------
``noncausal_softmax(scores)`` computes, per query row i over all key columns j::

    P[i, j] = exp(scale * S[i, j] - m_i) / sum_k exp(scale * S[i, k] - m_i)
    m_i     = max_k(scale * S[i, k])          (bf16-rounded, as the kernel stores)

No masking of any kind is applied: P[i, j] > 0 for every finite input, above
and below the diagonal alike.  Internals are float64, like
``softmax_ref`` (activation.py:1063-1081); the device instead runs
exp2((x - m) * log2e) with log2e = 1.4453125 (softmax_aie2p.h:15-16, 88-98),
which is mathematically the same softmax and numerically within the 2.94e-2
relative envelope measured on npu2 (activation.py:400-407).

Byte-level kernel contract (see KERNEL_CONTRACT below and DESIGN.md)
--------------------------------------------------------------------
Proposed call, the symbol that already exists in the AIE2P object built
from ``softmax.cc``::

    void softmax_rows_bf16(bfloat16 *input, bfloat16 *output,
                           const int32_t rows, const int32_t row_len);
    // softmax_aie2p.h:222-223

    input   : rows * row_len bfloat16, row-major contiguous.
              For Z-Image scores: (96, 96) -> 9216 elems -> 18432 bytes.
    output  : same shape, same dtype, same byte count.
    rows    : 96 (int32 scalar).
    row_len : 96 (int32 scalar); MUST be a multiple of 32 -- the kernel's
              vector step (softmax_aie2p.h:15 ``SM_VEC_LEN 32``,
              :53 ``elem_iters = vector_size / SM_VEC_LEN``, no tail).
              96 = 3 * 32, so the Z-Image row fits exactly.

Fallback call (factory ``activation.py:376`` as shipped) ::

    void softmax_bf16(bfloat16 *input, bfloat16 *output,
                      const int32_t input_size);
    // softmax_aie2p.h:194-195
    input/output : one flat row, (96,) -> 192 bytes each, 96 calls per head.

bfloat16 element: IEEE-754 binary32 truncated to 1 sign / 8 exponent /
7 mantissa bits, 2 bytes little-endian (ml_dtypes.bfloat16, linalg.py:18).
Padding, if a design ever needs it, is filled with bf16 ``lowest`` =
bit pattern 0xFF7F, as ``mask_bf16`` / ``partial_softmax`` do
(softmax_aie2p.h:212-219, mha.cc:564-571).  Z-Image needs NO padding:
96 % 32 == 0.

The 1/sqrt(head_dim) scale of attention is NOT in this kernel
(``softmax_simple_bf16`` hard-codes plain log2e, softmax_aie2p.h:57).
Fold 1/sqrt(128) into Q before QK^T (bilinear, exact-ish) or scale the S
tile before the call; ``scale=`` below models that pre-scale.

bfp16ebs8 note
--------------
The softmax kernel consumes plain bfloat16 (softmax_aie2p.h:20-22).  The
bfp16ebs8 block format (9 bytes per 8 values, ``aie.utils.bfp``
module docstring, bfp.py:6-27) matters only for the QK^T / P*V matmul
operands if the design emulates bf16 mmul with bfp16 (``mm_bfp``,
linalg.py:920-1006; 8x8x8 micro-tile, linalg.py:916).  It is modelled here
through ``aie.utils.bfp`` as required, with byte layouts derived from it.

Run:  py -3.12 noncausal_softmax.py        (self-test, no NPU, no compile)
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

# ---------------------------------------------------------------------------
# sys.path exactly as the rest of zimage-bfp16-npu does (bfp16_gemm_iron.py:28-31):
# <root>/_pylibs and <root>/_pylibs/mlir_aie/python.  Python 3.12.
# ---------------------------------------------------------------------------
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent  # .../public_work
PYLIBS = ROOT / "_pylibs"
sys.path.append(str(PYLIBS))
sys.path.append(str(PYLIBS / "mlir_aie" / "python"))

# ---------------------------------------------------------------------------
# Z-Image geometry (fixed by the model, see task statement)
# ---------------------------------------------------------------------------
HIDDEN = 3840
NUM_HEADS = 30
HEAD_DIM = 128  # = QK^T reduction depth and P*V output width
SEQ = 96  # tokens == query rows == key columns per head
ATTN_SCALE = float(1.0 / np.sqrt(HEAD_DIM))  # 1/sqrt(d) to pre-fold into Q

# ---------------------------------------------------------------------------
# bfloat16 helpers -- pure numpy, RNE, so the reference needs no ml_dtypes.
# (Kernel-side the type is ml_dtypes.bfloat16, linalg.py:18.)
# ---------------------------------------------------------------------------
BF16_LOWEST_BITS = 0xFF7F  # std::numeric_limits<bfloat16>::lowest(); mask fill
BF16_BYTES = 2


def round_bf16(x) -> np.ndarray:
    """float -> bfloat16 values (kept as float32 arrays), round-to-nearest-even.

    Same convention as the kernel's conv_even stores
    (mha.cc:27 ``ROUNDING_MODE``, softmax_aie2p.h:23-24).  Finite inputs only.
    """
    xf = np.ascontiguousarray(x, dtype=np.float32)
    if not np.isfinite(xf).all():
        raise ValueError("round_bf16: inputs must be finite")
    u = xf.view(np.uint32)
    # round half to even: add the round bit (bit 16) plus 0x7FFF tie mask
    rounding_bias = ((u >> 16) & 1) + 0x7FFF
    u2 = (u + rounding_bias) & np.uint32(0xFFFF0000)
    return u2.view(np.float32)


def bf16_bytes(x) -> np.ndarray:
    """float array -> uint8 bytes, the exact buffer the kernel reads.

    Every value becomes 2 bytes little-endian; element (i, j) of an (R, C)
    tile lives at byte offset (i * C + j) * 2.
    """
    rounded_u32 = np.ascontiguousarray(round_bf16(x)).view(np.uint32)
    u16 = ((rounded_u32 >> 16) & np.uint32(0xFFFF)).astype(np.uint16)
    return np.ascontiguousarray(u16).view(np.uint8)


def lowest_bf16(shape) -> np.ndarray:
    """Tile filled with bf16 ``lowest`` (0xFF7F): the kernel's mask/pad value."""
    out = np.empty(shape, dtype=np.float32)
    bits = np.full(shape, BF16_LOWEST_BITS, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32)


# ---------------------------------------------------------------------------
# THE REFERENCE
# ---------------------------------------------------------------------------
def noncausal_softmax(scores, *, scale: float = 1.0, out_dtype="bf16"):
    """Bidirectional softmax over the LAST axis (the key axis) of ``scores``.

    Args:
        scores: (..., L) score matrix -- e.g. one head's (96, 96) QK^T tile,
            or (30, 96, 96) for a whole block's heads.  Finite values.
        scale: pre-applied multiplier, models folding 1/sqrt(128) into Q or
            an S-scale step before the kernel call.  The kernel itself has
            no scale parameter (softmax_aie2p.h:20-22).
        out_dtype: "bf16" (what the kernel stores), "f32" or "f64".

    Returns:
        Same shape as ``scores``; every row sums to 1 up to the output
        rounding.  NO position is masked: all L keys get positive weight
        for every query row.
    """
    s = np.asarray(scores, dtype=np.float64) * float(scale)
    if s.ndim < 1:
        raise ValueError("noncausal_softmax: scores must be at least 1-D")
    if not np.isfinite(s).all():
        raise ValueError("noncausal_softmax: scores must be finite")
    m = s.max(axis=-1, keepdims=True)  # bf16 rounding of m is immaterial in f64
    e = np.exp(s - m)
    p = e / e.sum(axis=-1, keepdims=True)
    if out_dtype == "bf16":
        return round_bf16(p)
    if out_dtype == "f32":
        return p.astype(np.float32)
    if out_dtype == "f64":
        return p
    raise ValueError(f"noncausal_softmax: unknown out_dtype {out_dtype!r}")


def softmax_rows_ref(tile, rows: int, row_len: int, *, scale: float = 1.0):
    """Reference for ONE ``softmax_rows_bf16`` call (softmax_aie2p.h:222-229).

    ``tile`` is the flat input buffer of ``rows * row_len`` bf16 values
    (row-major); returns the flat output buffer the kernel writes.  Each row
    is softmaxed independently over its full ``row_len`` keys -- the whole
    row must be in this one call: softmax has no cross-call state, so a row
    split across key chunks cannot be recombined without the online
    machinery of ``partial_softmax`` / ``rescale_O`` (mha.cc).
    """
    rows, row_len = int(rows), int(row_len)
    if rows <= 0 or row_len <= 0:
        raise ValueError("softmax_rows_ref: rows and row_len must be positive")
    if row_len % 32:
        raise ValueError(
            f"softmax_rows_ref: row_len={row_len} is not a multiple of the "
            "kernel's 32-element vector step (softmax_aie2p.h:15,53)"
        )
    t = np.asarray(tile)
    if t.size != rows * row_len:
        raise ValueError(
            f"softmax_rows_ref: tile has {t.size} values, expected {rows * row_len}"
        )
    out = noncausal_softmax(t.reshape(rows, row_len), scale=scale, out_dtype="bf16")
    return out.reshape(-1)


def causal_softmax_rows_ref(tile, rows: int, row_len: int, *, scale: float = 1.0):
    """What the CAUSAL sibling does (for contrast only): row r keeps cols <= r.

    Mirrors ``mha_softmax_ref``'s on-diagonal rule (linalg.py:1276-1277) and
    ``softmax_rows_causal_bf16``'s ``mask_bf16(input + r*row_len, r + 1,
    row_len)`` (softmax_aie2p.h:237).  Differs from softmax_rows_ref on every
    strictly-upper-triangular entry -- which is exactly the Z-Image bug.
    """
    rows, row_len = int(rows), int(row_len)
    t = np.asarray(tile).reshape(rows, row_len).astype(np.float64) * float(scale)
    cols = np.arange(row_len)
    keep = cols[None, :] <= np.arange(rows)[:, None]
    m = np.where(keep, t, -np.inf).max(axis=1, keepdims=True)
    e = np.exp(np.where(keep, t - m, -np.inf))
    p = e / e.sum(axis=1, keepdims=True)
    return round_bf16(p).reshape(-1)


# ---------------------------------------------------------------------------
# BYTE-LEVEL KERNEL CONTRACT
# ---------------------------------------------------------------------------
KERNEL_CONTRACT = {
    # The kernel we recommend binding: it already exists in the object built
    # from activation/softmax.cc on XDNA2 (softmax_aie2p.h:222-229).
    "kernel_symbol": "softmax_rows_bf16",
    "kernel_source": "_pylibs/mlir_aie/include/aie_kernels/activation/"
    "softmax_aie2p.h:222-229 (unmasked); causal sibling at :232-241 "
    "(mask call at :237)",
    "arch": "aie2p (XDNA2); softmax.cc:10-14 selects softmax_aie2p.h",
    # Call signature, in order:
    #   void softmax_rows_bf16(bfloat16 *input, bfloat16 *output,
    #                          const int32_t rows, const int32_t row_len);
    "arg_order": ["input", "output", "rows", "row_len"],
    # Input tile
    "in_shape": (SEQ, SEQ),  # (96 query rows, 96 key cols) score tile per head
    "in_dtype": "bfloat16 (ml_dtypes; 1 sign / 8 exp / 7 mantissa, 2 bytes LE)",
    "in_elems": SEQ * SEQ,
    "in_bytes": SEQ * SEQ * BF16_BYTES,  # 18432
    "in_layout": "row-major contiguous; element (i,j) at byte (i*96 + j)*2",
    # Output tile
    "out_shape": (SEQ, SEQ),
    "out_dtype": "bfloat16",
    "out_elems": SEQ * SEQ,
    "out_bytes": SEQ * SEQ * BF16_BYTES,  # 18432
    "out_layout": "identical to input layout (in-place-safe row by row: the "
    "kernel writes output only after the row's max/sum are read, "
    "softmax_aie2p.h:33-110)",
    # Scalar params
    "params": {"rows": 96, "row_len": 96, "dtype": "int32"},
    # Constraints
    "alignment": "row_len % 32 == 0 (softmax_aie2p.h:15 SM_VEC_LEN, :53 "
    "elem_iters = vector_size / 32, tail-free loop)",
    "padding_value": "bf16 lowest = 0xFF7F LE per element (mask fill, "
    "softmax_aie2p.h:212-219, mha.cc:564-571); NOT needed at seq 96",
    "scale": "not a kernel parameter; fold 1/sqrt(128) into Q or pre-scale S "
    "(softmax_simple_bf16 hard-codes log2e, softmax_aie2p.h:57)",
    # Whole-block working set (per head), for the dataflow budget
    "per_head_buffers": {
        "Q (96x128 bf16)": 96 * HEAD_DIM * BF16_BYTES,
        "K (96x128 bf16)": 96 * HEAD_DIM * BF16_BYTES,
        "V (96x128 bf16)": 96 * HEAD_DIM * BF16_BYTES,
        "S (96x96 bf16)": SEQ * SEQ * BF16_BYTES,
        "P (96x96 bf16)": SEQ * SEQ * BF16_BYTES,
        "O (96x128 bf16)": 96 * HEAD_DIM * BF16_BYTES,
    },
    # Fallback: factory-bound softmax_bf16 (activation.py:376-391)
    "fallback_symbol": "softmax_bf16",
    "fallback_signature": "void softmax_bf16(bfloat16 *input, bfloat16 *output, "
    "const int32_t input_size)  (softmax_aie2p.h:194-195)",
    "fallback_call": "one flat (96,) tile = 192 bytes per call, "
    "tile_size=96 satisfies the factory's %32 rule "
    "(activation.py:386 -> _common.py:518-533); 96 calls per head, "
    "2880 calls for 30 heads",
}


def describe_contract() -> str:
    lines = ["AIE kernel byte contract (bidirectional attention softmax):"]
    for k, v in KERNEL_CONTRACT.items():
        lines.append(f"  {k:20s} : {v}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# bfp16ebs8 model, through aie.utils.bfp as required.
# (9 bytes per 8 values: [exponent u8, m0..m7 i8]; blocks along the last
# axis; see bfp.py:6-27 and npdtypes.py:29-36 block_size=8, block_bytes=9.)
# ---------------------------------------------------------------------------
def _import_bfp():
    """``from aie.utils import bfp``, with a documented workaround.

    This workspace's Python 3.12 has numpy but not ml_dtypes, and
    ``aie/utils/__init__.py`` imports hostruntime which imports
    ml_dtypes.bfloat16 at module scope.  The bfp codec itself (bfp.py) is
    pure numpy and never dereferences bfloat16, so a placeholder attribute
    is enough to let the REAL ``aie.utils.bfp`` codec load.  If real
    ml_dtypes is installed, nothing is stubbed.
    """
    try:
        from aie.utils import bfp  # noqa: WPS433 (deliberate runtime import)
        return bfp
    except ModuleNotFoundError as exc:
        if "ml_dtypes" not in str(exc):
            raise
        import types

        stub = types.ModuleType("ml_dtypes")

        class bfloat16:  # placeholder: import-time annotation use only
            """Stand-in for ml_dtypes.bfloat16 (codec never calls it)."""

        stub.bfloat16 = bfloat16
        sys.modules["ml_dtypes"] = stub
        from aie.utils import bfp  # noqa: WPS433
        return bfp


BFP_BLOCK = 8  # values sharing one exponent (npdtypes.py:36)
BFP_BLOCK_BYTES = 9  # [exp, m0..m7] (npdtypes.py:36, bfp.py:35-37)


def bfp_encode_tile(x) -> np.ndarray:
    """(R, C) float tile -> (R, C*9//8) uint8, blocks along C.

    C must be a multiple of 8.  96x96 scores -> (96, 108) = 10368 bytes.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.shape[-1] % BFP_BLOCK:
        raise ValueError("bfp_encode_tile: last axis must be a multiple of 8")
    return _import_bfp().encode(x)


def bfp_decode_tile(b) -> np.ndarray:
    """(R, C*9//8) uint8 -> (R, C) float32; the exact inverse per block."""
    return _import_bfp().decode(b)


def bfp_quantize_tile(x) -> np.ndarray:
    """What a bfp16ebs8 matmul kernel reads of ``x`` (bfp.quantize)."""
    return _import_bfp().quantize(x)


def bfp_shuffle_tile(b, width, height, tile_width, tile_height, unshuffle=False):
    """Into (or out of) the mmul 8x8 block layout for ``mm_bfp`` operands.

    width/tile_width count VALUES (multiples of 8); this is the host-side
    shuffle the block-datatypes matmul needs (bfp.py:22-27, 188-199).
    """
    return _import_bfp().shuffle(
        b, width, height, tile_width, tile_height, unshuffle=unshuffle
    )


# ---------------------------------------------------------------------------
# Self-test (pure numpy; proves the reference and pins the byte contract)
# ---------------------------------------------------------------------------
def _self_test() -> None:
    rng = np.random.default_rng(0)

    # --- reference semantics -------------------------------------------
    s = rng.standard_normal((SEQ, SEQ)) * 4.0
    p = noncausal_softmax(s, scale=ATTN_SCALE, out_dtype="f64")
    assert p.shape == (SEQ, SEQ)
    assert np.all(p > 0.0), "noncausal softmax must give every key positive weight"
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-12)

    # whole-matrix softmax must equal per-row calls of the tile reference
    flat = round_bf16(s * ATTN_SCALE).reshape(-1)
    rows_out = softmax_rows_ref(flat, SEQ, SEQ, scale=1.0).reshape(SEQ, SEQ)
    one_row = softmax_rows_ref(round_bf16(s[7] * ATTN_SCALE), 1, SEQ)
    assert np.array_equal(rows_out[7], one_row), "per-row call must match the tile call"

    # causal vs noncausal: strictly-upper triangle live here, dead there
    causal = causal_softmax_rows_ref(flat, SEQ, SEQ).reshape(SEQ, SEQ)
    iu = np.triu_indices(SEQ, k=1)
    assert np.all(causal[iu] == 0.0)
    assert np.all(rows_out[iu] > 0.0), "the gap under test: upper triangle is live"
    assert np.allclose(causal.sum(axis=1), 1.0, atol=2e-2), "causal rows normalize too"

    # --- byte contract -------------------------------------------------
    b = bf16_bytes(p)
    assert b.size == KERNEL_CONTRACT["in_bytes"] == 18432
    assert b.dtype == np.uint8
    pad = bf16_bytes(lowest_bf16((4,)))
    assert np.array_equal(pad.reshape(4, 2), np.tile([0x7F, 0xFF], (4, 1))), (
        "mask fill must encode as 0xFF7F little-endian"
    )
    # round trip through the exact bytes
    back = b.view(np.uint16).astype(np.uint32) << 16
    back = back.view(np.float32)
    assert np.array_equal(back, round_bf16(p)), "byte round trip must be exact"

    # row_len % 32 guard
    try:
        softmax_rows_ref(np.zeros(95), 1, 95)
    except ValueError:
        pass
    else:
        raise AssertionError("row_len=95 must be rejected (vector step 32)")

    # --- bfp16ebs8 model (via aie.utils.bfp) ---------------------------
    x = np.ascontiguousarray(round_bf16(rng.standard_normal((SEQ, SEQ))))
    enc = bfp_encode_tile(x)
    assert enc.shape == (SEQ, SEQ * BFP_BLOCK_BYTES // BFP_BLOCK) == (96, 108)
    assert enc.size == SEQ * SEQ * BFP_BLOCK_BYTES // BFP_BLOCK  # 10368 bytes
    dec = bfp_decode_tile(enc)
    assert dec.shape == (SEQ, SEQ)
    # Truncation costs at most one LSB of the block's shared scale:
    # LSB = 2^(floor(log2(block_max)) - 6) (decode map, bfp.py:141-143).
    blk = x.reshape(SEQ, SEQ // BFP_BLOCK, BFP_BLOCK)
    bmax = np.abs(blk).max(axis=-1)
    lsb = np.where(
        bmax > 0.0, np.exp2(np.floor(np.log2(np.maximum(bmax, 2.0**-149))) - 6), 0.0
    )
    err = np.abs(dec - x).reshape(blk.shape).max(axis=-1)
    assert np.all(err <= lsb + 2.0**-149), "bfp decode within one block LSB"
    q = bfp_quantize_tile(x)
    assert q.shape == x.shape
    # shuffle into the mmul layout and back must be a byte permutation
    sh = bfp_shuffle_tile(enc, SEQ, SEQ, SEQ, SEQ)
    assert np.array_equal(
        bfp_shuffle_tile(sh, SEQ, SEQ, SEQ, SEQ, unshuffle=True), enc
    ), "shuffle must be invertible"

    print("noncausal_softmax self-test: PASS")
    print(f"  score tile            : ({SEQ}, {SEQ}) bf16 = {b.size} bytes")
    print(f"  rows sum to 1         : min {p.astype(np.float64).sum(1).min():.6f}, "
          f"max {p.astype(np.float64).sum(1).max():.6f}")
    print(f"  upper-triangle mass   : {rows_out[iu].sum():.3f} (causal: 0.000)")
    print(f"  bfp16ebs8 encoded     : {enc.shape} uint8 = {enc.size} bytes")


if __name__ == "__main__":
    print(describe_contract())
    print()
    _self_test()
