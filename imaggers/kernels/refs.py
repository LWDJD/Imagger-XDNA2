# SPDX-License-Identifier: MIT
"""refs.py -- numpy references for the two new Z-Image DiT AIE2P kernels.

    rms_norm_weighted :  y = x / sqrt(mean(x**2) + eps) * gamma     (per row)
    adaln_modulate    :  y = x * (1 + scale) + shift                (per row)

Both kernels keep f32 accumulators and round to bfloat16 ONCE, at the store
(conv_even = round-to-nearest-even, set inside each kernel's entry).  These
references therefore compute in float64 and round to bf16 once with the same
RNE rule, so a correct device result is within ~1 bf16 ulp of ``ref``.

Sanity order (per the project's hard-won constraints): FIRST the plain
float32 semantics (``*_f32_exact`` identities below, no rounding model at
all), THEN the bf16 model.  The device f32 entry points
(``rms_norm_weighted_f32`` / ``adaln_modulate_f32``) are judged against the
same formulas at f32-level tolerance; the bf16 entry points against the bf16
model at bf16-level tolerance.

Self-test:  python refs.py        (pure numpy; no NPU, no compile)
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# bfloat16 helpers -- pure numpy, RNE.  (Bit-twiddle round_bf16 proven in
# imaggers/adaln/refs.py; ml_dtypes is NOT needed and NOT present in every
# interpreter on this machine.)
# ---------------------------------------------------------------------------
BF16_EXP_BIAS = 127
BF16_MANT_BITS = 7  # stored fraction bits; 8 significant bits with the implicit 1


def round_bf16(x) -> np.ndarray:
    """float -> bfloat16 values (kept as float32 arrays), round-to-nearest-even.

    Matches the kernels' ``conv_even`` stores and ml_dtypes' f32->bf16 cast.
    Finite inputs only.
    """
    xf = np.ascontiguousarray(x, dtype=np.float32)
    if not np.isfinite(xf).all():
        raise ValueError("round_bf16: inputs must be finite")
    u = xf.view(np.uint32)
    rounding_bias = ((u >> 16) & 1) + 0x7FFF
    u2 = (u + rounding_bias) & np.uint32(0xFFFF0000)
    return u2.view(np.float32)


def bf16_ulp(x) -> np.ndarray:
    """Per-element ulp of |x| in the bf16 grid: 2**(exp(x) - 7).

    ulp(v) for v = 1.f x 2**e is 2**(e-7) (8 significant bits).
    """
    xf = np.asarray(x, dtype=np.float64)
    xf = np.where(xf == 0.0, np.float64(2.0**-126), np.abs(xf))
    exp = np.floor(np.log2(xf)).astype(np.int64)
    return np.exp2(exp - BF16_MANT_BITS)


def bf16_bytes(x) -> np.ndarray:
    """float array -> flat uint8 byte buffer, exactly what the kernel reads."""
    rounded_u32 = np.ascontiguousarray(round_bf16(x)).view(np.uint32)
    u16 = ((rounded_u32 >> 16) & np.uint32(0xFFFF)).astype(np.uint16)
    return np.ascontiguousarray(u16).reshape(-1).view(np.uint8)


def bytes_to_f32(raw: np.ndarray) -> np.ndarray:
    """Raw bf16 bytes -> float32 values (the inverse view of bf16_bytes)."""
    u16 = np.ascontiguousarray(raw, np.uint8).view(np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32)


# ---------------------------------------------------------------------------
# (a) WEIGHTED RMSNORM:  y = x / rms(x) * gamma
# ---------------------------------------------------------------------------
def rms_norm_weighted_ref(x, gamma, *, eps: float = 1e-5, out_dtype="bf16"):
    """``y = x / sqrt(mean(x**2) + eps) * gamma`` over the LAST axis.

    Args:
        x: (..., H) rows -- one token row (H,) or a block (M, H).
        gamma: (H,) per-channel weight, SHARED across rows.
        eps: added inside the sqrt (kernel default 1e-5, as
            ``rms_norm_aie2p.h:141-142``; Z-Image's own value is a model
            property, parameterized here).
        out_dtype: "bf16" (one RNE rounding -- what the kernel stores),
            "f32", "f64".

    The kernel's statistics accumulate in f32 and its product chain is exact
    (x*gamma needs at most 16 significand bits), so this f64 formula is the
    arithmetic model; the device difference is dominated by the single bf16
    store rounding.
    """
    xf = np.asarray(x, dtype=np.float64)
    g = np.asarray(gamma, dtype=np.float64)
    if g.ndim != 1 or xf.shape[-1] != g.shape[0]:
        raise ValueError(
            f"rms_norm_weighted_ref: gamma must be ({xf.shape[-1]},), got {g.shape}"
        )
    if not float(eps) > 0:
        raise ValueError("rms_norm_weighted_ref: eps must be positive")
    ms = np.mean(xf * xf, axis=-1, keepdims=True)
    rms = np.sqrt(ms + float(eps))
    return _cast_out(xf / rms * g, out_dtype)


def rms_norm_weighted_f32_exact(x, gamma, *, eps: float = 1e-5) -> np.ndarray:
    """f32 semantics of the f32 entry point: same formula, float32 throughout.

    Used for the np.float32 sanity stage; NOT bit-exact against the device
    (the device's f32 sum-of-squares reduction order differs from numpy's),
    so judge with an f32-level tolerance.
    """
    xf = np.asarray(x, dtype=np.float32)
    g = np.asarray(gamma, dtype=np.float32)
    ms = np.mean(xf * xf, axis=-1, keepdims=True)
    rms = np.sqrt(ms + np.float32(eps))
    return (xf / rms) * g


# ---------------------------------------------------------------------------
# (b) adaLN MODULATION:  y = x * (1 + scale) + shift   (per token row)
# ---------------------------------------------------------------------------
def adaln_modulate_ref(x, scale, shift, *, out_dtype="bf16"):
    """``y = x * (1 + scale) + shift`` over the LAST axis, per-token s/b.

    One f32 accumulator (``acc = x; acc += x*scale; acc += shift``) and ONE
    bf16 rounding at the store, so ``1 + scale`` is never materialized in
    bf16.  scale/shift are per-token: row t reads scale[t], shift[t].
    """
    xf = np.asarray(x, dtype=np.float64)
    s = np.asarray(scale, dtype=np.float64)
    b = np.asarray(shift, dtype=np.float64)
    if s.shape != xf.shape or b.shape != xf.shape:
        raise ValueError(
            f"adaln_modulate_ref: scale/shift must match x {xf.shape}, "
            f"got {s.shape} / {b.shape}"
        )
    return _cast_out(xf * (1.0 + s) + b, out_dtype)


def adaln_modulate_f32_exact(x, scale, shift) -> np.ndarray:
    """f32 semantics of the f32 entry point (float32 throughout)."""
    xf = np.asarray(x, dtype=np.float32)
    s = np.asarray(scale, dtype=np.float32)
    b = np.asarray(shift, dtype=np.float32)
    return xf * (np.float32(1.0) + s) + b


def adaln_magnitude(x, scale, shift) -> np.ndarray:
    """Per-element magnitude bound |x*(1+s)| + |shift| -- the right error
    scale for a kernel that rounds the SUM once (y itself can cancel to
    zero while the terms do not)."""
    xf = np.asarray(x, dtype=np.float64)
    s = np.asarray(scale, dtype=np.float64)
    b = np.asarray(shift, dtype=np.float64)
    return np.abs(xf * (1.0 + s)) + np.abs(b)


def _cast_out(y: np.ndarray, out_dtype: str) -> np.ndarray:
    if out_dtype == "bf16":
        return round_bf16(y)
    if out_dtype == "f32":
        return y.astype(np.float32)
    if out_dtype == "f64":
        return y
    raise ValueError(f"unknown out_dtype {out_dtype!r}")


# ---------------------------------------------------------------------------
# Tolerances used by run.py's verification (bf16-level; declared up front).
#
# Both kernels have ONE visible rounding: the bf16 store (half-ulp = 2^-8
# relative).  The scale split / f32 chains contribute <= 2^-16 relative
# (see FINDINGS.md), so the device result can land at most 1 bf16 grid step
# from bf16(exact) when a tie flips.  Declared budget: 2 bf16 ulps of the
# reference element (ulp = 2**(exp-7)), plus a 2^-26 relative floor for
# elementwise chains measured against the magnitude bound.
# ---------------------------------------------------------------------------
TOL_ULPS = 2.0  # in bf16 ulps of the per-element reference
TOL_ULP_FLOOR = 2.0**-126  # smallest normal bf16, absolute floor
F32_RTOL = 2.0**-18  # f32 sanity stage: f32 rounding + reduction-order slack
F32_ATOL = 2.0**-20


def check_within_ulps(dev, ref, ulps=TOL_ULPS, scale=None):
    """Return (ok, max_abs_err, max_ulp_err) of dev vs ref on the bf16 grid.

    ``scale`` replaces the reference as the error's magnitude base -- pass
    adaln_magnitude() for adaln_modulate, whose single-rounded SUM can cancel
    near zero while its terms do not.
    """
    d = np.asarray(dev, np.float64)
    r = np.asarray(ref, np.float64)
    err = np.abs(d - r)
    base = r if scale is None else np.asarray(scale, np.float64)
    u = np.maximum(bf16_ulp(base), TOL_ULP_FLOOR)
    ulp_err = err / u
    return bool(ulp_err.max() <= ulps), float(err.max()), float(ulp_err.max())


# ---------------------------------------------------------------------------
# Self-test: f32 semantics FIRST, then the bf16 model.
# ---------------------------------------------------------------------------
def _self_test() -> None:
    rng = np.random.default_rng(20261015)
    M, H = 96, 3840
    x = rng.standard_normal((M, H)) * 3.0
    gamma = rng.standard_normal(H) * 0.5 + 1.0
    scale = rng.standard_normal((M, H)) * 0.2
    shift = rng.standard_normal((M, H)) * 0.1

    # ---- stage 0: float32 semantics first (no rounding model) ------------
    xf32 = x.astype(np.float32)
    g32 = gamma.astype(np.float32)
    y32 = rms_norm_weighted_f32_exact(xf32, g32)
    ms = np.mean(xf32 * xf32, axis=-1, keepdims=True)
    rms = np.sqrt(ms + np.float32(1e-5))
    assert np.allclose(y32, (xf32 / rms) * g32, rtol=0, atol=0), "definitional"
    # gamma = 1 is the unweighted RMSNorm
    one32 = rms_norm_weighted_f32_exact(xf32, np.ones(H, np.float32))
    assert np.allclose(one32, xf32 / rms, rtol=0, atol=0), "gamma=1 == rms_norm"
    m32 = adaln_modulate_f32_exact(
        xf32, scale.astype(np.float32), shift.astype(np.float32)
    )
    assert np.allclose(
        m32, xf32 * (np.float32(1.0) + scale.astype(np.float32))
        + shift.astype(np.float32),
        rtol=0,
        atol=0,
    ), "adaln definitional"
    # per-token really is per-token
    m2 = adaln_modulate_f32_exact(
        np.stack([xf32[0], xf32[0]]),
        np.stack([scale[0], scale[1]]).astype(np.float32),
        np.stack([shift[0], shift[1]]).astype(np.float32),
    )
    assert not np.allclose(m2[0], m2[1]), "per-token modulation must vary"
    # scale=0, shift=0 is identity
    ident = adaln_modulate_f32_exact(xf32, np.zeros_like(xf32), np.zeros_like(xf32))
    assert np.array_equal(ident, xf32), "s=0,b=0 -> identity"

    # ---- stage 1: bf16 model -------------------------------------------
    y = rms_norm_weighted_ref(x, gamma, out_dtype="f64")
    rms64 = np.sqrt((x.astype(np.float64) ** 2).mean(1, keepdims=True) + 1e-5)
    assert np.allclose(y, x / rms64 * gamma, rtol=0, atol=0)
    one = rms_norm_weighted_ref(x, np.ones(H), out_dtype="f64")
    lib = (xf32 / rms).astype(np.float64)  # library rms_norm_ref at f32 stats
    assert np.allclose(one, lib, rtol=1e-5, atol=1e-5), "gamma=1 == unweighted ref"
    # one bf16 rounding: the stored values sit on the bf16 grid
    yb = rms_norm_weighted_ref(x, gamma, out_dtype="bf16")
    assert np.array_equal(yb, round_bf16(y)), "bf16 out is one RNE rounding"

    m = adaln_modulate_ref(x, scale, shift, out_dtype="f64")
    assert np.allclose(m, x * (1.0 + scale) + shift, rtol=0, atol=0)
    mb = adaln_modulate_ref(x, scale, shift, out_dtype="bf16")
    assert np.array_equal(mb, round_bf16(m))
    ident_b = adaln_modulate_ref(x, np.zeros_like(x), np.zeros_like(x))
    assert np.array_equal(ident_b, round_bf16(x))
    # the +1 matters: x*scale+shift is NOT x*(1+scale)+shift
    naive = round_bf16(round_bf16(x * scale).astype(np.float64) + shift)
    assert np.max(np.abs(naive - mb)) > 1e-3, "the +1 must not be dropped"

    # ---- stage 2: byte contract ----------------------------------------
    row_b = bf16_bytes(x[0])
    assert row_b.size == H * 2
    back = bytes_to_f32(row_b)
    assert np.array_equal(back, round_bf16(x[0])), "byte round trip exact"
    flat = bf16_bytes(x)
    assert flat.size == M * H * 2
    assert np.array_equal(
        bytes_to_f32(flat).reshape(M, H), round_bf16(x)
    ), "block byte layout is row-major"

    # ---- stage 3: rounding-bound sanity for the declared tolerance ------
    # A device whose pre-rounding value is within half a bf16 ulp (2^-9
    # relative) of the exact one must land within 1 grid step of the
    # reference; the declared 2-ulp budget covers a tie flip plus the
    # kernel's <= 2^-16 relative scale-split slack.
    for pert in (2.0**-9, -(2.0**-9)):
        ok, mx, mu = check_within_ulps(round_bf16(y * (1.0 + pert)), yb)
        assert ok and mu <= 2.0, (pert, mx, mu)

    print("kernels refs self-test: PASS")
    print(f"  shapes: x {(M, H)}  gamma {(H,)}  scale/shift {(M, H)}")
    print(f"  bf16 ulp model: ulp(v) = 2**(floor(log2|v|) - 7); tol = {TOL_ULPS} ulps")
    print("  f32-first stages: definitional identities PASS")


if __name__ == "__main__":
    _self_test()
