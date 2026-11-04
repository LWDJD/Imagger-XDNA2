# SPDX-License-Identifier: MIT
"""refs.py -- numpy references for NON-CAUSAL (bidirectional) attention on XDNA2.

The design being verified (`attention.py`) computes, per head, at Z-Image
geometry (heads=30, head_dim=128, seq=96, start small and grow):

    O = softmax_rows_bf16(Qs @ K^T) @ V        with Qs = bf16(Q / sqrt(head_dim))

Three levels of reference live here, loosest to tightest:

1. ``attention_ref``          -- the true math in float64.  NO causal mask, no
   masking of any kind: every query row attends to every key column.  This is
   the definition of "non-causal attention" the device must match.
2. ``attention_pipeline_model`` -- what the DEVICE pipeline computes, step by
   step: host bf16 rounding, the scalar ``mm`` kernel's f32 accumulation order,
   the ``cast_f32_bf16_row`` conv_even store, the ``softmax_rows_bf16``
   arithmetic, and the second GEMM.  Only the hardware exp2 inside
   ``softmax_simple_bf16`` (softmax_aie2p.h:94 -> exp2_bf16.h:59-61,
   ``aie::exp2<bfloat16>``) is modelled as exact 2^x; its error is measured on
   npu2 at 2.94e-2 relative (activation.py:400-407) and is the documented gap.
3. ``softmax_rows_kernel_model`` alone, for step-level debugging.

Byte/arith conventions copied from `imaggers/attn/noncausal_softmax.py`
(same workspace, authored for this task): bf16 is IEEE-754 binary32 truncated
to 1/8/7, round-to-nearest-even, and every device store that lands in bf16
uses conv_even (the kernels set it themselves: softmax_aie2p.h:23-24,
cast_f32_bf16.cc per datamovement.py:175-178).

QK-norm (new requirement from the parent agent, see FINDINGS.md):
Z-Image applies ``attention.norm_q`` / ``attention.norm_k`` (weighted
RMSNorm over head_dim) to Q and K AFTER the to_q/to_k projection and BEFORE
RoPE -- zimage-port/export_dit.py:410-431:
    query = attn.to_q(hidden); key = attn.to_k(hidden); value = attn.to_v(...)
    query = query.unflatten(...); key = key.unflatten(...); value = ...
    if attn.norm_q is not None: query = attn.norm_q(query)
    if attn.norm_k is not None: key   = attn.norm_k(key)
    ... apply_rotary_emb_real(query, cos, sin); apply_rotary_emb_real(key, ...)
so the full chain is  to_q/to_k/to_v -> unflatten -> QK-norm -> RoPE ->
QK^T -> softmax -> P*V.  ``qk_norm_ref`` below is the reference for that
stage; ``attention_pipeline_model(..., qk_norm=True)`` runs it too.

Run:  py -3.12 refs.py      (self-test only; no NPU, no compile)
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Geometry constants (the real block; small configs are just smaller)
# ---------------------------------------------------------------------------
HIDDEN = 3840
NUM_HEADS = 30
HEAD_DIM = 128
SEQ = 96
QK_NORM_EPS = 1e-5  # zimage-port/export_dit.py:64 norm_eps=1e-5

LOG2E = 1.4453125  # softmax_aie2p.h:16 -- the kernel's EXACT bf16 constant

# ---------------------------------------------------------------------------
# bfloat16 helpers -- pure numpy, RNE (conv_even)
# ---------------------------------------------------------------------------
def round_bf16(x) -> np.ndarray:
    """float -> bfloat16 values (kept as float32 arrays), round-half-to-even."""
    xf = np.ascontiguousarray(x, dtype=np.float32)
    if not np.isfinite(xf).all():
        raise ValueError("round_bf16: inputs must be finite")
    u = xf.view(np.uint32)
    rounding_bias = ((u >> 16) & 1) + 0x7FFF
    u2 = (u + rounding_bias) & np.uint32(0xFFFF0000)
    return u2.view(np.float32)


def bf16_ulp(x) -> np.ndarray:
    """bf16 ulp of each value (used to report errors in ulps)."""
    x = np.asarray(x, np.float32)
    bits = x.view(np.uint32)
    exp = ((bits >> 23) & 0xFF).astype(np.int64) - 127
    return np.exp2(np.maximum(exp - 7, -126)).astype(np.float64)


# ---------------------------------------------------------------------------
# Reference 1: the true math
# ---------------------------------------------------------------------------
def attention_ref(Q, K, V, *, scale: float | None = None):
    """Non-causal attention in float64: O = softmax(scale * Q K^T) @ V.

    Q, K, V: (heads, seq, head_dim) arrays (any float dtype).  ``scale``
    defaults to 1/sqrt(head_dim).  NO mask: P[i, j] > 0 for every (i, j).
    Returns (O, S, P) with O (heads, seq, head_dim) and S, P (heads, seq, seq).
    """
    Q = np.asarray(Q, np.float64)
    K = np.asarray(K, np.float64)
    V = np.asarray(V, np.float64)
    if Q.shape != K.shape or Q.shape != V.shape or Q.ndim != 3:
        raise ValueError(f"attention_ref: need equal (heads, seq, hd) shapes, "
                         f"got {Q.shape}, {K.shape}, {V.shape}")
    heads, seq, hd = Q.shape
    if scale is None:
        scale = 1.0 / np.sqrt(hd)
    S = scale * (Q @ np.swapaxes(K, -1, -2))
    m = S.max(axis=-1, keepdims=True)
    e = np.exp(S - m)
    P = e / e.sum(axis=-1, keepdims=True)
    O = P @ V
    return O, S, P


# ---------------------------------------------------------------------------
# Reference 2: the device pipeline, step by step
# ---------------------------------------------------------------------------
def f32_accum_gemm(A16, B16, *, b_col_maj: bool = False,
                   product: str = "bf16_floor") -> np.ndarray:
    """One ``mm(vectorized=False)`` call's C = A @ B, device arithmetic.

    ``matmul_scalar`` (mm_aie2p.h:21-48) computes, per output element, a
    running sum over i in order with ``T_out=float`` accumulation for the
    bf16->f32 combo this design uses (matmul_scalar_bf16_f32), then one f32
    store.

    MEASURED (attn_impl/FINDINGS.md, raw f32 P*V dump at (1,32,32)):
    the scalar ``bfloat16 * bfloat16`` product is rounded to bf16 with the
    core's DEFAULT rounding mode, which boots at floor (toward -inf) --
    bit-exact on 81% of elements and within 1 f32 ulp on the rest, vs 0%
    bit-exact for exact-f32 products.  ``product`` selects the model:

        "bf16_floor"  device-faithful (default)
        "bf16_rne"    products rounded conv_even (what conv_even before the
                      GEMM would give)
        "f32"         exact f32 products (optimistic)

    ``b_col_maj`` (mm compiled with -DB_COL_MAJ, mm_aie2p.h:293-297) reads B
    as stored (n, k): element (i, col) is b[col * k + i].  This design uses
    it for QK^T so K stays row-major (seq, head_dim) -- the layout QK-norm
    needs (see module docstring).
    """
    A16 = np.asarray(A16, np.float32)
    B16 = np.asarray(B16, np.float32)
    m, k = A16.shape
    if b_col_maj:
        n, kB = B16.shape
        assert kB == k, (B16.shape, A16.shape)
    else:
        k2, n = B16.shape
        assert k2 == k, (B16.shape, A16.shape)

    def prod(p):
        if product == "f32":
            return p
        if product == "bf16_rne":
            return round_bf16(p)
        if product == "bf16_floor":
            u = np.ascontiguousarray(p, np.float32).view(np.uint32)
            low = u & np.uint32(0xFFFF)
            bump = ((u >> np.uint32(31)) == np.uint32(1)) & (low != 0)
            return ((u & np.uint32(0xFFFF0000))
                    + (bump.astype(np.uint32) << np.uint32(16))).view(np.float32)
        raise ValueError(f"unknown product mode {product!r}")

    C = np.zeros((m, n), np.float32)
    for i in range(k):
        a = A16[:, i]  # (m,)
        b = B16[:, i] if b_col_maj else B16[i, :]  # (n,)
        p = prod((a[:, None] * b[None, :]).astype(np.float32))
        C = (C + p).astype(np.float32)
    return C


def softmax_rows_kernel_model(S16) -> np.ndarray:
    """``softmax_rows_bf16`` arithmetic (softmax_aie2p.h:20-115, 222-229).

    Models, per row: exact bf16 max; max_scaled = m * log2e in f32;
    e_in = x * log2e - max_scaled in f32; exp2 -> bf16; f32 sum; bf16
    reciprocal of the sum; out = bf16(e * inv).  The ONLY unmodelled piece is
    the hardware exp2 (``aie::exp2<bfloat16>``, exp2_bf16.h:59-61), modelled
    as exact 2^x -- its measured error on npu2 is 2.94e-2 relative
    (activation.py:400-407) and is the tolerance floor of everything below.
    """
    S = np.ascontiguousarray(S16, np.float32)
    S = round_bf16(S)  # input is bf16
    rows, row_len = S.shape
    m16 = S.max(axis=1, keepdims=True)  # bf16 max of bf16 values: exact
    max_scaled = (m16.astype(np.float32) * np.float32(LOG2E)).astype(np.float32)
    e_in = (S * np.float32(LOG2E)).astype(np.float32) - max_scaled
    e = round_bf16(np.exp2(e_in.astype(np.float64)))  # exp2_bf16 surrogate
    esum = np.zeros((rows, 1), np.float32)
    for j in range(row_len):  # sequential f32 accumulation
        esum = (esum + e[:, j : j + 1]).astype(np.float32)
    inv = round_bf16(1.0 / esum.astype(np.float64))  # (bfloat16) aie::inv
    P = round_bf16((e * inv).astype(np.float32))
    return P


def qk_norm_ref(x, w, *, eps: float = QK_NORM_EPS):
    """Weighted RMSNorm over the last axis: y = x / sqrt(mean(x^2) + eps) * w.

    The Z-Image QK-norm (attention.norm_q / attention.norm_k).  ``w`` is the
    (head_dim,) per-column weight.  Float64 here; the device kernel rounds y
    to bf16 once (FINDINGS.md documents the target kernel's arithmetic).
    """
    x64 = np.asarray(x, np.float64)
    w64 = np.asarray(w, np.float64)
    ms = (x64 * x64).mean(axis=-1, keepdims=True)
    return x64 / np.sqrt(ms + eps) * w64


def qk_norm_model(x16, w16, *, eps: float = QK_NORM_EPS):
    """The device qk_norm.cc arithmetic on one block of rows.

    Per row: sum_sq = f32 sum of exact bf16 squares; ms = scalar_mul(sum_sq,
    inv(cols)); rms = ms + eps; inv = scalar_invsqrt(rms); each output is
    one conv_even store of f32(x_i * w_i) * inv (products exact in f32).
    The only gap to the silicon is the last bits of ``aie::invsqrt``
    (~1 ulp) and the f32 summation order of sum_sq.
    """
    x = round_bf16(np.ascontiguousarray(x16, np.float32))
    w = round_bf16(np.ascontiguousarray(w16, np.float32))
    x2 = (x * x).astype(np.float32)  # bf16 squares are exact in f32
    sum_sq = np.zeros(x.shape[:-1] + (1,), np.float32)
    for j in range(x.shape[-1]):
        sum_sq = (sum_sq + x2[..., j : j + 1]).astype(np.float32)
    inv_cols = np.float32(1.0 / x.shape[-1])  # aie::inv(aie::to_float(cols))
    ms = (sum_sq * inv_cols).astype(np.float32)  # scalar_mul
    rms = (ms + np.float32(eps)).astype(np.float32)
    inv = (np.float32(1.0) / np.sqrt(rms).astype(np.float32)).astype(np.float32)
    y = ((x * w).astype(np.float32) * inv).astype(np.float32)
    return round_bf16(y)


def attention_pipeline_model(Q, K, V, *, m: int, scale: float | None = None,
                             qk_norm_w=(None, None), product: str = "bf16_floor"):
    """Bit-level emulation of attention.py's pipeline (minus hardware exp2).

    Q, K, V: (heads, seq, hd) float arrays -- the host rounds them to bf16
    exactly as run.py does.  ``m`` is the query row-block of the design (the
    device re-rounds C to f32 per block, which this reproduces).  ``scale``
    defaults to 1/sqrt(hd) and is folded into Q on the host.

    ``qk_norm_w = (w_q, w_k)``: when both are not None, apply the QK-norm
    stage to Q and K rows first (device order: norm, then GEMM).

    Returns dict with the device-visible outputs and intermediates:
    ``O`` (bf16 values), ``O32``, ``S16``, ``P`` per block, and inputs used.
    """
    Q = np.asarray(Q, np.float64)
    K = np.asarray(K, np.float64)
    V = np.asarray(V, np.float64)
    heads, seq, hd = Q.shape
    if seq % m:
        raise ValueError(f"m={m} must divide seq={seq}")
    if scale is None:
        scale = 1.0 / np.sqrt(hd)

    # host inputs, exactly what the design receives
    Qs = round_bf16(Q * scale)   # 1/sqrt(hd) folded into Q (DESIGN.md d)
    K16 = round_bf16(K)
    V16 = round_bf16(V)

    w_q, w_k = qk_norm_w
    if (w_q is None) != (w_k is None):
        raise ValueError("qk_norm_w: give both weights or neither")
    if w_q is not None:
        # QK-norm is SCALE-INVARIANT (it divides by the row rms), so the
        # 1/sqrt(hd) fold must NOT sit on Q before the norm -- it would
        # cancel exactly.  The design folds it into the norm WEIGHT:
        # norm(Q) * (scale * w) == scale * (norm(Q) * w), and both the norm
        # kernel and the reference see the scaled weight once.
        Q16 = round_bf16(Q)
        K16raw = round_bf16(K)
        wq_dev = round_bf16(np.asarray(w_q, np.float64) * scale)
        wk_dev = round_bf16(np.asarray(w_k, np.float64))
        Qs = np.empty_like(Q16)
        K16 = np.empty_like(K16raw)
        for h in range(heads):
            Qs[h] = qk_norm_model(Q16[h], wq_dev)
            K16[h] = qk_norm_model(K16raw[h], wk_dev)

    blocks = seq // m
    O = np.empty((heads, seq, hd), np.float32)
    dbg = {"Qs": Qs, "K16": K16, "V16": V16, "S16": [], "P": []}
    for h in range(heads):
        for b in range(blocks):
            r0 = b * m
            q_blk = Qs[h, r0 : r0 + m]            # (m, hd) row-major
            # QK^T: mm(m, hd, seq) with B = K stored (seq, hd) = b_col_maj
            S32 = f32_accum_gemm(q_blk, K16[h], b_col_maj=True, product=product)
            S16 = round_bf16(S32)                 # cast_f32_bf16_row (conv_even)
            P = softmax_rows_kernel_model(S16)    # softmax_rows_bf16, in place
            # P*V: mm(m, seq, hd) with B = V (seq, hd) row-major
            O32 = f32_accum_gemm(P, V16[h], product=product)
            O[h, r0 : r0 + m] = O32
            dbg["S16"].append(S16)
            dbg["P"].append(P)
    dbg["O32"] = O
    dbg["O"] = round_bf16(O)                      # cast_f32_bf16_row of O
    return dbg


def attention_budget(Qs, K16, V16, P_ref, O_ref, *, p_rel: float,
                     product: str = "bf16_floor") -> np.ndarray:
    """Per-element error budget for O vs the float64 non-causal reference.

    Sums the four measured/derived error mechanisms of the pipeline
    (see FINDINGS.md for the measurements behind each factor):

    1. softmax envelope: ``p_rel`` relative per P element (measured on npu2
       via the identity-V trick) prices through P@V as
       ``p_rel * sum_j P[i,j] |V[j,k] - O[i,k]|`` -- rows of dP sum to ~0,
       so the P-weighted SPREAD of V bounds it, not its magnitude.
    2. QK^T product rounding: each scalar bf16 product rounds toward -inf
       (measured; factor 2^-7 per product), so
       ``|dS[i,j]| <= 2^-7 * sum_l |Q[i,l] K[j,l]|``; through the softmax
       Jacobian (|dP_j| <= 2 P_j max|dS|) this prices at
       ``2 * max_j|dS[i,j]| * sum_j P[i,j] |V[j,k]|``.
    3. P*V product rounding: ``2^-7 * sum_j P[i,j] |V[j,k]|``.
    4. the final bf16 store: 4 ulp of O.
    """
    Qs = np.asarray(Qs, np.float64)
    K16 = np.asarray(K16, np.float64)
    V16 = np.asarray(V16, np.float64)
    P_ref = np.asarray(P_ref, np.float64)
    O_ref = np.asarray(O_ref, np.float64)
    heads, seq, hd = Qs.shape
    if product not in ("bf16_floor", "bf16_rne"):
        raise ValueError("budget factors assume bf16-rounded products")
    prod_err = 2.0 ** -7 if product == "bf16_floor" else 2.0 ** -8

    pv_mag = (P_ref[:, :, :, None] * np.abs(V16)[:, None, :, :]).sum(axis=2)  # Σ_j P_j |V_jk|
    # Σ_j P[i,j] |V[j,k] - O[i,k]|, row by row (a 4-D broadcast would be
    # heads*seq*seq*hd floats -- 283 MB at full Z-Image geometry)
    spread = np.empty_like(pv_mag)
    for h in range(heads):
        for i in range(seq):
            spread[h, i] = (
                P_ref[h, i, :, None] * np.abs(V16[h] - O_ref[h, i, None, :])
            ).sum(axis=0)
    # QK^T per-entry product noise, max over keys per query row
    s_abs = np.abs(Qs) @ np.abs(K16).swapaxes(-1, -2)  # Σ_l |Q_il K_jl|
    s_noise = prod_err * s_abs.max(axis=2)             # (heads, seq)

    budget = p_rel * spread + (2.0 * s_noise[:, :, None] + prod_err) * pv_mag
    ulp_o = np.exp2(np.maximum(
        (O_ref.view(np.uint64) >> 52 & 0x7FF).astype(np.int64) - 1023 - 7, -126))
    budget += 4.0 * ulp_o
    return budget


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _self_test() -> None:
    rng = np.random.default_rng(7)
    heads, seq, hd, m = 1, 32, 32, 16

    Q = rng.standard_normal((heads, seq, hd)) * 2.0
    K = rng.standard_normal((heads, seq, hd)) * 2.0
    V = rng.standard_normal((heads, seq, hd)) * 2.0

    # Reference 1: every key gets positive weight, rows of P sum to 1.
    O, S, P = attention_ref(Q, K, V)
    assert np.all(P > 0.0), "non-causal: the upper triangle must be live"
    assert np.allclose(P.sum(axis=-1), 1.0, atol=1e-12)
    iu = np.triu_indices(seq, k=1)
    assert P[0][iu].min() > 0.0

    # Reference 2: pipeline model close to the math.
    mdl = attention_pipeline_model(Q, K, V, m=m)
    Om = mdl["O"].astype(np.float64)
    # relative error only where O is significant: P@V with random V cancels
    # to near zero on some elements, where no relative claim is meaningful.
    sig = np.abs(O) >= 0.05 * np.abs(O).max()
    rel = np.abs(Om - O)[sig] / np.abs(O)[sig]
    print(f"[self] pipeline model vs f64 math: max rel err (significant) "
          f"{rel.max():.4e}  (softmax exp2 envelope 2.94e-2)")
    # P's bf16 store alone is 2^-9 relative per element and amplifies through
    # P@V cancellation to ~1e-1 on significant elements of random-V O; that
    # is the amplification run.py's spread-derived budget prices exactly.
    assert rel.max() < 0.25, "pipeline model must sit near the f64 math"

    # The derived budget (run.py's criterion) must price the model-vs-math
    # gap: softmax envelope + GEMM product roundings + stores (helper above).
    P_REL = 0.0294 + 2.0 ** -9
    P_full = np.concatenate(mdl["P"]).reshape(heads, seq, seq).astype(np.float64)
    Qs_full = mdl["Qs"]
    K_full = mdl["K16"]
    budget = attention_budget(Qs_full, K_full, mdl["V16"], P_full, O,
                              p_rel=P_REL)
    inside = float((np.abs(Om - O) <= budget).mean())
    print(f"[self] model inside derived budget for {100 * inside:.2f}% "
          f"of elements (aim 100%)")
    assert inside == 1.0, "derived budget must price the model-vs-math gap"

    # Per-row softmax normalization survives the kernel model.
    Pm = mdl["P"][0]
    rsums = Pm.astype(np.float64).sum(axis=1)
    print(f"[self] model P row sums: min {rsums.min():.6f} max {rsums.max():.6f}")
    assert np.allclose(rsums, 1.0, atol=2e-2)

    # The GEMM model must match an exact matmul at f32 rounding level when
    # products are exact ("f32" mode).
    A = round_bf16(rng.standard_normal((4, 8)) * 3)
    B = round_bf16(rng.standard_normal((8, 6)) * 3)
    C = f32_accum_gemm(A, B, product="f32")
    C_exact = (A.astype(np.float64) @ B.astype(np.float64))
    assert np.abs(C - C_exact).max() <= 1e-5 * max(1.0, np.abs(C_exact).max())

    # b_col_maj reads B as (n, k): A @ B_stored.T must equal the row-major call.
    Bs = round_bf16(rng.standard_normal((6, 8)) * 3)
    C_cm = f32_accum_gemm(A, Bs, b_col_maj=True)
    C_rm = f32_accum_gemm(A, Bs.T.copy())
    assert np.allclose(C_cm, C_rm), "b_col_maj must equal the transposed call"

    # QK-norm: every output row has rms ~ |w| pattern, weights applied.
    w = round_bf16(rng.standard_normal(hd) * 0.5 + 1.0)
    x = rng.standard_normal((seq, hd)) * 4.0
    y = qk_norm_ref(x, w)
    rms = np.sqrt(((y / w) ** 2).mean(axis=-1))
    assert np.allclose(rms, 1.0, atol=1e-4), "qk_norm must unit-rms each row"

    print("refs.py self-test: PASS")


if __name__ == "__main__":
    _self_test()
