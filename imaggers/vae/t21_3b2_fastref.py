# t21_3b2_fastref.py -*- Python -*-
"""U0: vectorized device model for the conv gate reference.

Why this file exists
--------------------
The gate reference for a conv row is vae_npu.bfp_gemm_model, whose inner loop
calls aie.iron.kernels.linalg.mm_bfp_tile_ref once per (m-tile, n-tile, k-step):
19.1M Python calls for the largest 3b2 geometry (M=65536, K=2336, N=256),
measured at 0.0588 ms each -> ~19 min of gate per row and 172 min for the 55
conv rows.  That cost, not the compile, dominates the batch.

What is and is not the same arithmetic
--------------------------------------
mm_bfp_tile_ref re-quantizes both operands on every call:

    aq = bfp.quantize(a, rounding="conv_even")     # A is bf16, core converts
    bq = bfp.quantize(b.T).T                       # B is host-encoded floor

Quantization is per 8-value block along K, and every tile boundary is
8-aligned (k = 8), so quantizing the whole A / B once gives the *same
operands*.  What cannot be batched is the K reduction: the kernel rounds C
back to bf16 on every k-call, so the loop over kk stays sequential and in
order.  Each BLAS call is then a (rows, k) x (k, N) contraction, whose
per-element summation order over k matches the per-tile one -- that is the
claim the bit-identity evidence in T21-3B2-BATCH.md tests, not an assumption.

Evidence (all bit-exact, 0 differing elements):
  * the 5 stored conv io.npz on disk: this model reproduces their ref_model
    (and ref_quant) exactly, and those arrays are the ones the device matched
    at 0 ulp when those xclbins were gated;
  * a freshly generated row-11 case, scalar vs this model on the same
    operands;
  * a geometry sweep over the 55 conv shapes, scalar vs this model.

The scalar path stays one env var away (T21_3B2_FAST_REF=0).

Note this module deliberately lives outside t21_3b2_plan.py: that file is
guarded by a source-level contract (pitfall 1) which asserts the harness
never calls bfp.quantize itself -- the operand path there must be exactly
_pack_b() (floor) plus the bf16 cast.  Keeping the model here keeps that
tripwire intact instead of loosening it.
"""
from __future__ import annotations

import numpy as np


def bfp_gemm_model_v(A_b, W, *, m: int = 16, k: int = 8, n: int = 16,
                     bqT=None, want_quant: bool = False,
                     row_block: int = 16384):
    """Vectorized bfp_gemm_model; same arithmetic, ~20-50x faster.

    The per-tile reference (vae_npu.bfp_gemm_model -> mm_bfp_tile_ref) calls
    bfp.quantize once per (m,k)/(k,n) tile.  Quantization is per 8-value block
    along K and every tile boundary is 8-aligned, so quantizing the whole
    matrix once yields the same operands.  What must stay sequential is the
    K reduction (one bf16 rounding per k-call), and it does: the k loop is
    unchanged and each BLAS call is a (rows, k) x (k, N) contraction whose
    per-element k order matches the per-tile one (verified bit-exact against
    5 stored device-verified ref_model arrays and against the scalar model on
    the real geometries; see T21-3B2-BATCH.md).

    bqT is the (N, K) float32 form of quantize(W.T); the caller may pass
    decode(packed B) directly, which is the very operand the device reads.
    """
    from aie.utils import bfp
    from ml_dtypes import bfloat16

    A_b = np.ascontiguousarray(A_b)
    M, K = A_b.shape
    N = W.shape[1] if bqT is None else bqT.shape[0]
    aq = bfp.quantize(np.ascontiguousarray(A_b, np.float32),
                      rounding="conv_even")
    if bqT is None:
        bqT = bfp.quantize(np.ascontiguousarray(np.asarray(W, np.float32).T))
    bqT = np.ascontiguousarray(bqT)
    assert bqT.shape == (N, K), (bqT.shape, (N, K))
    ref_model = np.empty((M, N), np.float64)
    ref_quant = np.empty((M, N), np.float64) if want_quant else None
    kb = K // k
    for r0 in range(0, M, row_block):
        r1 = min(r0 + row_block, M)
        accm = np.zeros((r1 - r0, N), np.float32)
        acc64 = np.zeros((r1 - r0, N), np.float64) if want_quant else None
        src = aq[r0:r1]
        for kk in range(kb):
            ab = np.ascontiguousarray(src[:, kk * k:(kk + 1) * k], np.float64)
            b = np.ascontiguousarray(bqT[:, kk * k:(kk + 1) * k].T, np.float64)
            r = ab @ b
            if want_quant:
                acc64 += r
            accm = (accm + r.astype(np.float32)).astype(bfloat16).astype(
                np.float32)
        ref_model[r0:r1] = accm.astype(np.float64)
        if want_quant:
            ref_quant[r0:r1] = acc64
    return ref_quant, ref_model


def unpack_b_bytes(B_bytes, K: int, N: int, k: int = 8, n: int = 16):
    """pack_b_blocks' inverse: packed bytes -> float32 (N, K) = quantize(W.T).

    decode() of the packed stream is exactly what the core reads, so the
    device model can be built from the operand that is actually sent.
    """
    from aie.utils import bfp as _bfp
    raw = (np.frombuffer(B_bytes, np.uint8) if isinstance(B_bytes, (bytes,
                                                                    bytearray))
           else np.asarray(B_bytes, np.uint8))
    blocks = raw.reshape(N // n, K // k, n, k // 8, 9)
    enc = blocks.transpose(0, 2, 1, 3, 4).reshape(N, K // 8 * 9)
    return _bfp.decode(enc)



def compare_bf16(got, want) -> dict:
    """bit-exactness of two float arrays after the bf16 rounding the gate uses."""
    from ml_dtypes import bfloat16
    g = np.asarray(got, np.float64).astype(bfloat16)
    w = np.asarray(want, np.float64).astype(bfloat16)
    eq = (g == w)
    return dict(bit_exact_frac=float(eq.mean()),
                n_bad=int(eq.size - int(eq.sum())), n=int(eq.size),
                max_abs_diff=float(np.abs(np.asarray(got, np.float64)
                                          - np.asarray(want, np.float64)).max()))
