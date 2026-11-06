# SPDX-License-Identifier: MIT
"""refs.py -- numpy references + byte contracts for the two missing Z-Image
DiT building blocks: WEIGHTED RMSNorm and adaLN MODULATION.

Z-Image DiT block (task statement):

    x = x + attn( adaLN_modulate(norm1(x), cond) )       # scale/shift
    x = x + ff(   adaLN_modulate(norm2(x), cond) )

with ``adaLN_modulate(x, s, b) = x * (1 + s) + b``, per-token scale/shift.

This is a DESIGN artifact: runnable numpy references and the exact byte
contracts an AIE kernel call must satisfy.  Nothing here compiles or touches
the NPU.

Why these two blocks are missing
--------------------------------
* The library norms are UNWEIGHTED: ``rms_norm`` / ``rms_norm_eps`` /
  ``layer_norm`` (``_pylibs/mlir_aie/python/aie/iron/kernels/norm.py:85/:114/
  :139``) take no gamma/beta argument at all; ``layer_norm_aie2p.h:63`` says
  "gamma = 1 and beta = 0 here, so the affine pair is not applied at all".
* The only weighted RMSNorm arithmetic in tree is the Gemma4 decode helper
  ``template <int D> rms_norm(y, x, w)``
  (``_pylibs/mlir_aie/include/aie_kernels/flm_gemma4/rms_norm.h:54-68``),
  reached only through the geometry-locked factories
  (``flm_gemma4.py:2157`` wraps the fused decode kernel; row type is
  ``(geometry.model_dim,)``, ``flm_gemma4.py:2166``, model_dim 1536/2560,
  ``flm_gemma4.py:903/:919``).  No factory exposes weighted RMSNorm at
  hidden 3840.
* ``affine_cast`` (``datamovement.py:211``) is per-COLUMN gamma*x+beta with
  gamma/beta a baked tensor Param (``datamovement.py:245`` roles ``Param``),
  f32 in / bf16 out (``datamovement.py:236-238``); adaLN needs per-TOKEN
  scale/shift on bf16.
* ``eltwise.mul_add`` computes ``a*b`` OR ``a+b`` per call, selected by
  ``is_mul`` -- NOT a fused ``a*b+c``
  (``eltwise.py:229-235``; C++ ``scale_shift.cc:17-27``).  So a two-pass
  ``mul`` then ``add`` is the best composition from existing kernels.

Reference semantics (float64 internals, one bf16 rounding at each store, the
same RNE convention as the kernels' ``conv_even`` stores -- see
``norm.py:34-43`` and ``eltwise.py:33-48`` tolerance notes):

1. ``rms_norm_weighted_ref(x, gamma)``::

       y = x / sqrt(mean(x**2) + eps) * gamma        (per row)

   The proposed kernel accumulates sum-of-squares in f32
   (``rms_norm_aie2p.h:123-138``), computes inv_rms in f32 scalar
   (``rms_norm_aie2p.h:140-142``) and rounds once on the store.  gamma is
   bf16, matching the Gemma4 helper (``rms_norm.h:55-60``).

2. ``adaln_modulate_ref(x, scale, shift)``::

       y = x * (1 + scale) + shift                    (per token row)

   The proposed kernel computes ``acc = x; acc = mac(acc, x, scale);
   acc = add(acc, shift)`` in one f32 accumulator and rounds once on the
   store -- so ``(1 + scale)`` never lands in bf16.

Both also come in a ``_two_pass`` flavor that models what the EXISTING
kernels give: each pass stores bf16 between kernels, so the composition
rounds one extra time (and needs ``1 + scale`` as an actual bf16 tile).

Byte-level kernel contracts: ``WEIGHTED_RMSNORM_CONTRACT`` and
``ADALN_CONTRACT`` below (see also DESIGN.md / RECOMMENDATION.md).

Run:  py -3.12 refs.py        (self-test, no NPU, no compile)
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Z-Image geometry (fixed by the model, see task statement)
# ---------------------------------------------------------------------------
HIDDEN = 3840  # hidden size; row length for every op below
SEQ = 96  # tokens == rows of the (96, 3840) residual stream
NUM_HEADS = 30
HEAD_DIM = 128
FFN_INNER = 10240

BF16_BYTES = 2

# Tiling facts the contracts cite (evidence in DESIGN.md):
ELTWISE_FIXED_TILE = 1024  # eltwise.py:27; mul.cc:137-139, add.cc:143-145
NORM_CHUNK_AIE2P = 64  # rms_norm_aie2p.h:78 kLanes (tuned path)
NORM_CHUNK_PORTABLE = 32  # rms_norm_aie2p.h:180 (rms_norm<bfloat16, 32>)
VEC_BF16_AIE2P = 32  # aie_arch.h:30 AIE_BF16_LANES on aie2p/XDNA2

# 3840 = 60 * 64 = 120 * 32 = 3 * 1024 + 768  (1024 does NOT divide 3840)
assert HIDDEN % NORM_CHUNK_AIE2P == 0
assert HIDDEN % VEC_BF16_AIE2P == 0
assert HIDDEN % ELTWISE_FIXED_TILE != 0
# Flat over the whole (96, 3840) tensor the fixed 1024 tile is exact:
assert (SEQ * HIDDEN) % ELTWISE_FIXED_TILE == 0
FLAT_1024_TILES = SEQ * HIDDEN // ELTWISE_FIXED_TILE  # 360


# ---------------------------------------------------------------------------
# bfloat16 helpers -- pure numpy, RNE.  Kernel-side the type is
# ml_dtypes.bfloat16 (eltwise.py:12); 1 sign / 8 exponent / 7 mantissa bits,
# 2 bytes little-endian.
# ---------------------------------------------------------------------------
def round_bf16(x) -> np.ndarray:
    """float -> bfloat16 values (kept as float32 arrays), round-to-nearest-even.

    Same convention as the kernels' conv_even stores (norm.py:25 imports
    ``conv_even`` as a norm setup, eltwise.py:33-48 documents one rounding
    on the store).  Finite inputs only.
    """
    xf = np.ascontiguousarray(x, dtype=np.float32)
    if not np.isfinite(xf).all():
        raise ValueError("round_bf16: inputs must be finite")
    u = xf.view(np.uint32)
    rounding_bias = ((u >> 16) & 1) + 0x7FFF
    u2 = (u + rounding_bias) & np.uint32(0xFFFF0000)
    return u2.view(np.float32)


def bf16_bytes(x) -> np.ndarray:
    """float array -> flat uint8 byte buffer, exactly what the kernel reads.

    Element (i, j) of an (R, C) tile lives at byte offset (i * C + j) * 2 of
    the returned 1-D buffer (the buffer a kernel's bf16 pointer sees).
    """
    rounded_u32 = np.ascontiguousarray(round_bf16(x)).view(np.uint32)
    u16 = ((rounded_u32 >> 16) & np.uint32(0xFFFF)).astype(np.uint16)
    return np.ascontiguousarray(u16).reshape(-1).view(np.uint8)


# ---------------------------------------------------------------------------
# (a) WEIGHTED RMSNORM:  y = x / rms(x) * gamma
# ---------------------------------------------------------------------------
def rms_norm_weighted_ref(x, gamma, *, eps: float = 1e-5, out_dtype="bf16"):
    """Weighted RMSNorm over the LAST axis: ``y = x / sqrt(mean(x**2)+eps) * gamma``.

    Args:
        x: (..., HIDDEN) -- e.g. one token row (3840,) or the whole
            (96, 3840) residual stream.
        gamma: (HIDDEN,) per-channel weight, one per column, SHARED by all
            rows (the Gemma4 helper's ``w``: rms_norm.h:55-68).
        eps: added inside the sqrt, as rms_norm's default 1e-5
            (norm.py:86; rms_norm_aie2p.h:141-142 adds it before invsqrt).
            Z-Image's own eps is "not determinable from the sources read".
        out_dtype: "bf16" (what the kernel stores), "f32", "f64".

    One bf16 rounding total, at the store: the kernel keeps sum-of-squares in
    f32 (rms_norm_aie2p.h:123-138) and the x*gamma*(s_hi+s_lo) product chain
    in f32 accumulators (rms_norm_aie2p.h:143-159, flm_gemma4/rms_norm.h:61-62).
    """
    xf = np.asarray(x, dtype=np.float64)
    g = np.asarray(gamma, dtype=np.float64)
    if xf.shape[-1] != g.shape[-1] or g.ndim != 1:
        raise ValueError(
            f"rms_norm_weighted_ref: gamma must be ({xf.shape[-1]},), got {g.shape}"
        )
    if float(eps) <= 0:
        raise ValueError("rms_norm_weighted_ref: eps must be positive")
    rms = np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + float(eps))
    y = xf / rms * g
    return _cast_out(y, out_dtype)


def rms_norm_weighted_two_pass_ref(x, gamma, *, eps: float = 1e-5):
    """What ``rms_norm`` (norm.py:85) then ``mul``/``mul_sized`` (eltwise.py:210/:322) computes.

    Two bf16 stores: the unweighted norm rounds once (norm.py:167-171
    reference), then the gamma multiply rounds again (mul.cc:37 MUL_ONE
    accumulates in f32 and converts on the store).  The difference from
    ``rms_norm_weighted_ref`` is exactly that intermediate rounding.
    """
    xf = np.asarray(x, dtype=np.float64)
    g = np.asarray(gamma, dtype=np.float64)
    rms = np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + float(eps))
    x_hat = round_bf16(xf / rms)  # rms_norm store
    y = round_bf16(x_hat.astype(np.float64) * g)  # mul store
    return y


# ---------------------------------------------------------------------------
# (b) adaLN MODULATION:  y = x * (1 + scale) + shift   (per token)
# ---------------------------------------------------------------------------
def adaln_modulate_ref(x, scale, shift, *, out_dtype="bf16"):
    """adaLN modulation over the LAST axis: ``y = x * (1 + scale) + shift``.

    Args:
        x: (..., HIDDEN) token rows, e.g. (96, 3840).
        scale: SAME shape as x -- per-token, per-channel (96, 3840).
        shift: SAME shape as x -- per-token, per-channel (96, 3840).
        out_dtype: "bf16" (what the kernel stores), "f32", "f64".

    One bf16 rounding total, at the store: the fused kernel computes
    ``acc = x; acc = mac(acc, x, scale); acc = add(acc, shift)`` in one f32
    accumulator, so ``1 + scale`` is never materialized (the pattern of
    add.cc:47-49's acc-from-vector + mac, plus one add).
    """
    xf = np.asarray(x, dtype=np.float64)
    s = np.asarray(scale, dtype=np.float64)
    b = np.asarray(shift, dtype=np.float64)
    if s.shape != xf.shape or b.shape != xf.shape:
        raise ValueError(
            f"adaln_modulate_ref: scale/shift must match x {xf.shape}, "
            f"got {s.shape} / {b.shape}"
        )
    y = xf * (1.0 + s) + b
    return _cast_out(y, out_dtype)


def adaln_two_pass_ref(x, scale, shift):
    """What the EXISTING kernels can compute: mul then add, bf16 between passes.

    Best 2-pass composition, with ``g = 1 + scale`` formed as its own bf16
    tile first (an ``add_sized`` pass against a ones tile -- or precomputed on
    the host when scale arrives host-side):

        g   = bf16(1 + scale)     # add pass (eltwise.py:295 add_sized)
        tmp = bf16(x * g)         # mul pass (eltwise.py:322 mul_sized)
        y   = bf16(tmp + shift)   # add pass

    Three bf16 stores.  If the design also goes through the fixed-1024
    ``mul_add`` symbol (eltwise.py:229) the passes are ``is_mul=1`` then
    ``is_mul=0`` (scale_shift.cc:17-27) -- same three roundings.  Raw
    scale/shift with NO pre-add pass would need a fused a*b+c, which does
    not exist in the library (scale_shift.cc is a*b OR a+b).
    """
    xf = np.asarray(x, dtype=np.float64)
    s = np.asarray(scale, dtype=np.float64)
    b = np.asarray(shift, dtype=np.float64)
    g = round_bf16(1.0 + s)  # pass 0: ones + scale
    tmp = round_bf16(xf * g.astype(np.float64))  # pass 1: x * g
    y = round_bf16(tmp.astype(np.float64) + b)  # pass 2: tmp + shift
    return y


def _cast_out(y: np.ndarray, out_dtype: str) -> np.ndarray:
    if out_dtype == "bf16":
        return round_bf16(y)
    if out_dtype == "f32":
        return y.astype(np.float32)
    if out_dtype == "f64":
        return y
    raise ValueError(f"unknown out_dtype {out_dtype!r}")


# ---------------------------------------------------------------------------
# BYTE-LEVEL KERNEL CONTRACTS
# ---------------------------------------------------------------------------
_ROW_BYTES = HIDDEN * BF16_BYTES  # 7680
_STREAM_BYTES = SEQ * HIDDEN * BF16_BYTES  # 737280

WEIGHTED_RMSNORM_CONTRACT = {
    "status": "MISSING in mlir-aie; propose new kernel (DESIGN.md section c)",
    "kernel_symbol": "rms_norm_weighted (new)",
    "closest_template": "norm/rms_norm_aie2p.h:114-170 rms_norm_aie2p + the "
    "weighted loop of flm_gemma4/rms_norm.h:54-68 (y = x*rsqrt*w)",
    "call": "void rms_norm_weighted(bfloat16 *x, bfloat16 *gamma, "
    "bfloat16 *y, int32_t cols, float epsilon)",
    # Per call: ONE token row (the norm reduction is along the row, so the
    # full 3840 must be in one call -- norm.py:85-91 allows any row length).
    "per_call": {
        "x": f"({HIDDEN},) bf16 = {_ROW_BYTES} B, row-major contiguous",
        "gamma": f"({HIDDEN},) bf16 = {_ROW_BYTES} B, per-channel, SHARED "
        "across all 96 calls (may be a core-resident Param)",
        "y": f"({HIDDEN},) bf16 = {_ROW_BYTES} B",
        "cols": "int32 = 3840 (Param, cf. norm.py:101 parameter_bindings)",
        "epsilon": "float32 scalar (default 1e-5, norm.py:86; Z-Image's own "
        "value not determinable from the sources read)",
        "buffer_bytes_per_call": 3 * _ROW_BYTES,  # 23040
    },
    "per_block": {
        "calls": SEQ,
        "x": f"({SEQ}, {HIDDEN}) bf16 = {_STREAM_BYTES} B",
        "y": f"({SEQ}, {HIDDEN}) bf16 = {_STREAM_BYTES} B",
        "gamma": f"({HIDDEN},) bf16 = {_ROW_BYTES} B (once)",
    },
    "tiles": f"{NORM_CHUNK_AIE2P}-element chunks on the aie2p tuned path "
    f"(kLanes, rms_norm_aie2p.h:78) -> {HIDDEN // NORM_CHUNK_AIE2P} chunks/row, "
    f"no tail; {NORM_CHUNK_PORTABLE}-element vectors on the portable path "
    f"(rms_norm_aie2p.h:180) -> {HIDDEN // NORM_CHUNK_PORTABLE}/row. "
    f"3840 needs no padding either way.",
    "accumulators": "f32 (sum of squares rms_norm_aie2p.h:123-138; product "
    "chain rms_norm_aie2p.h:156-162); ONE bf16 rounding on the store",
    # Composition from existing kernels (no new C++):
    "composed_existing": {
        "pass_1": "rms_norm(tile_size=3840) -- norm.py:85-111, symbol "
        "rms_norm(x, y, cols) rms_norm_aie2p.h:174-182; 96 calls",
        "pass_2": "mul_sized(tile_size=3840) -- eltwise.py:322-346, symbol "
        "eltwise_mul_bf16_vector_size(a=x_hat, b=gamma, c=y, size=3840) "
        "mul.cc:141-144; 96 calls. Alternative: fixed mul(tile_size=1024) "
        "over the FLAT (96,3840) stream = 360 calls with gamma replicated "
        "per token (737280 B host tensor) -- 3840 = 3*1024+768 so a "
        "PER-ROW 1024 tiling does not exist",
        "caveats": "mul/mul_sized are bf16-only (eltwise.py:76-80); an f32 "
        "gamma needs convert_copy first (datamovement.py:167-208). Two-pass "
        "rounds bf16 twice (see rms_norm_weighted_two_pass_ref).",
        "calls_per_block": "192 (96+96) at 3840-row tiles, or 456 "
        "(96 + 360) at flat 1024 tiles",
    },
}

ADALN_CONTRACT = {
    "status": "MISSING in mlir-aie; propose new kernel (DESIGN.md section c)",
    "kernel_symbol": "adaln_modulate (new)",
    "closest_template": "eltwise/scale_shift.cc (named for exactly this "
    "scale->shift two-phase) + add.cc:39-51 eltwise_vadd_mac acc idiom",
    "call": "void adaln_modulate(bfloat16 *x, bfloat16 *scale, "
    "bfloat16 *shift, bfloat16 *y, int32_t cols)",
    # Per call: ONE token row of the (96, 3840) tensors.
    "per_call": {
        "x": f"({HIDDEN},) bf16 = {_ROW_BYTES} B",
        "scale": f"({HIDDEN},) bf16 = {_ROW_BYTES} B -- PER-TOKEN (token t "
        "reads scale[t]); unlike affine_cast's per-column baked Param "
        "(datamovement.py:245-246)",
        "shift": f"({HIDDEN},) bf16 = {_ROW_BYTES} B -- PER-TOKEN",
        "y": f"({HIDDEN},) bf16 = {_ROW_BYTES} B = bf16(x*(1+scale)+shift)",
        "cols": "int32 = 3840 (or compile-time N like MUL_ELEMS, mul.cc:18-21)",
        "buffer_bytes_per_call": 4 * _ROW_BYTES,  # 30720
    },
    "per_block": {
        "calls": SEQ,
        "x / y": f"({SEQ}, {HIDDEN}) bf16 = {_STREAM_BYTES} B each",
        "scale / shift": f"({SEQ}, {HIDDEN}) bf16 = {_STREAM_BYTES} B each "
        "-- produced per token by the cond projection (a GEMM, linalg.py:920)",
    },
    "tiles": f"{VEC_BF16_AIE2P}-lane vectors (aie2p AIE_BF16_LANES, "
    f"aie_arch.h:30; the eltwise loops stride by vec_factor, mul.cc:44) -> "
    f"{HIDDEN // VEC_BF16_AIE2P} vectors/row, no tail. Eltwise FIXED tiles "
    f"are {ELTWISE_FIXED_TILE} elements (eltwise.py:27): 3840 = 3*1024+768 "
    f"per row (does not fit), but the flat (96,3840) stream = "
    f"{FLAT_1024_TILES} exact 1024-tiles (368640/1024).",
    "accumulators": "f32; ONE bf16 rounding on the store "
    "(acc = x; mac(acc, x, scale); add(acc, shift); store)",
    "composed_existing": {
        "pass_0_optional": "g = bf16(1+scale): add_sized(3840) of a ones "
        "tile and scale (eltwise.py:295-319), 96 calls -- skippable only if "
        "the host supplies 1+scale directly. mm_bfp has no bias to fold it "
        "into (linalg.py:986-1005 args are A, B, C only)",
        "pass_1": "tmp = x*g via mul_sized(3840) (eltwise.py:322) or "
        "mul_add(1024, is_mul=1) (eltwise.py:229-258); 96 or 360 calls",
        "pass_2": "y = tmp+shift via add_sized(3840) (eltwise.py:295) or "
        "mul_add(1024, is_mul=0); 96 or 360 calls",
        "is_mul_note": "mul_add is a*b OR a+b per call, NOT a*b+c "
        "(scale_shift.cc:17-27, eltwise.py:230) -- >=2 calls per tile even "
        "in the best case, 3 with the +1 pass",
        "tmp_bytes": f"{_ROW_BYTES} B per row-tile between passes "
        f"({_STREAM_BYTES} B if materialized whole)",
        "calls_per_block": "288 at row tiles (3*96), 1080 at flat 1024 "
        "tiles (3*360), vs 96 for the fused kernel",
    },
    "layout_note": "row-major in/out. The GEMM A operand wants the (8,8,8) "
    "micro-tile block layout (mm_bfp_mixed.cc:15-22; linalg.py:928-930) -- "
    "the corner turn sits at the DMA hop into the GEMM (linalg.py:404) or "
    "is fused into this kernel's store loop (DESIGN.md section d)",
}


def describe_contracts() -> str:
    lines = ["AIE kernel byte contracts (Z-Image adaLN building blocks):"]
    for name, c in (
        ("WEIGHTED_RMSNORM", WEIGHTED_RMSNORM_CONTRACT),
        ("ADALN_MODULATE", ADALN_CONTRACT),
    ):
        lines.append(f"\n[{name}]")
        for k, v in c.items():
            lines.append(f"  {k:20s} : {v}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test (pure numpy; proves the references and pins the byte contract)
# ---------------------------------------------------------------------------
def _self_test() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((SEQ, HIDDEN)) * 3.0
    gamma = rng.standard_normal(HIDDEN) * 0.5 + 1.0
    scale = rng.standard_normal((SEQ, HIDDEN)) * 0.2
    shift = rng.standard_normal((SEQ, HIDDEN)) * 0.1

    # --- (a) weighted RMSNorm -----------------------------------------
    y = rms_norm_weighted_ref(x, gamma, out_dtype="f64")
    assert y.shape == (SEQ, HIDDEN)
    # Defining property: normalize then weight, per row
    rms = np.sqrt((x.astype(np.float64) ** 2).mean(1, keepdims=True) + 1e-5)
    assert np.allclose(y, x / rms * gamma, rtol=0, atol=0)
    # gamma=1 recovers the UNWEIGHTED rms_norm reference (norm.py:167-171)
    one = rms_norm_weighted_ref(x, np.ones(HIDDEN), out_dtype="f64")
    xf = x.astype(np.float32)
    ms = np.mean(xf * xf, axis=-1, keepdims=True)
    lib = (xf / np.sqrt(ms + 1e-5)).astype(np.float64)
    assert np.allclose(one, lib, rtol=1e-6, atol=1e-6), "gamma=1 == rms_norm"
    # the two-pass composition loses one extra bf16 rounding, no more
    two = rms_norm_weighted_two_pass_ref(x, gamma)
    exact = round_bf16(y)
    ulp = np.abs(two.astype(np.float64) - exact.astype(np.float64))
    rel = ulp / np.maximum(np.abs(exact.astype(np.float64)), 2.0**-126)
    assert np.all(rel < 2.0**-6), "two-pass within ~2 bf16 ulps of fused"

    # --- (b) adaLN ----------------------------------------------------
    m = adaln_modulate_ref(x, scale, shift, out_dtype="f64")
    assert m.shape == (SEQ, HIDDEN)
    assert np.allclose(m, x * (1.0 + scale) + shift, rtol=0, atol=0)
    # scale=0, shift=0 is identity
    ident = adaln_modulate_ref(x, np.zeros_like(x), np.zeros_like(x))
    assert np.array_equal(ident, round_bf16(x))
    # per-token really is per-token: two rows, same x, different s -> different y
    x2 = np.stack([x[0], x[0]])
    s2 = np.stack([scale[0], scale[1]])
    b2 = np.stack([shift[0], shift[1]])
    m2 = adaln_modulate_ref(x2, s2, b2, out_dtype="f64")
    assert not np.allclose(m2[0], m2[1]), "per-token modulation must vary"
    # two-pass composition vs fused: THREE bf16 stores (g, tmp, y) against
    # the fused one, so the error is up to ~3.5 half-ulps (2^-8 each) of
    # |x*(1+s)| + |shift| -- NOT relative to y (cancellation in y can make y
    # tiny while the intermediates are not).  Measured max 3.40 half-ulps on
    # this seed; 4 gives headroom.
    m_tp = adaln_two_pass_ref(x, scale, shift)
    err = np.abs(m_tp.astype(np.float64) - round_bf16(m).astype(np.float64))
    mag = np.abs(x * (1.0 + scale)).astype(np.float64) + np.abs(shift).astype(
        np.float64
    )
    assert np.all(err <= 4 * 2.0**-8 * mag + 2.0**-126), "two-pass within 4 ulp"
    # without the +1 (raw scale) a two-pass CANNOT reproduce the reference:
    naive = round_bf16(
        round_bf16(x * scale).astype(np.float64) + shift
    ).astype(np.float64)
    assert np.max(np.abs(naive - round_bf16(m).astype(np.float64))) > 1e-3, (
        "x*scale+shift is NOT x*(1+scale)+shift -- the +1 matters"
    )

    # --- byte contract ------------------------------------------------
    row_b = bf16_bytes(x[0])
    assert row_b.size == _ROW_BYTES == 7680
    stream_b = bf16_bytes(x)
    assert stream_b.size == _STREAM_BYTES == 737280
    back = stream_b.view(np.uint16).astype(np.uint32) << 16
    assert np.array_equal(
        back.view(np.float32), round_bf16(x).reshape(-1)
    ), "byte round trip must be exact"
    # element (t, j) offset
    t, j = 5, 1234
    off = (t * HIDDEN + j) * BF16_BYTES
    assert np.array_equal(stream_b[off : off + 2], bf16_bytes(x[t, j])[0:2])

    # --- tiling arithmetic -------------------------------------------
    assert HIDDEN == 3 * ELTWISE_FIXED_TILE + 768  # no per-row 1024 tiling
    assert FLAT_1024_TILES == 360  # flat tiling of (96,3840) is exact
    assert HIDDEN % NORM_CHUNK_AIE2P == 0  # 60 norm chunks, no tail

    print("adaln refs self-test: PASS")
    print(f"  weighted RMSNorm row   : ({HIDDEN},) bf16 = {_ROW_BYTES} B, "
          f"{HIDDEN // NORM_CHUNK_AIE2P} x 64-elem norm chunks")
    print(f"  adaLN row              : 4 buffers x {_ROW_BYTES} B = "
          f"{4 * _ROW_BYTES} B per call")
    print(f"  (96, 3840) stream      : {_STREAM_BYTES} B = "
          f"{SEQ * HIDDEN} elems -> {FLAT_1024_TILES} fixed-1024 tiles flat")
    print(f"  per-row 1024 tiling    : 3840 = 3*1024 + 768 (does not divide)")


if __name__ == "__main__":
    print(describe_contracts())
    print()
    _self_test()
