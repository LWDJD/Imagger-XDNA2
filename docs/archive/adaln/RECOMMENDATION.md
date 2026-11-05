# RECOMMENDATION — weighted RMSNorm + adaLN for the Z-Image DiT block

**Verdict: write exactly TWO small new kernels — `rms_norm_weighted` (~10–15
lines over `rms_norm_aie2p.h`) and `adaln_modulate` (~20 lines over
`scale_shift.cc`) — and use the existing-kernel composition (`rms_norm` +
`mul_sized`, `mul_sized` + `add_sized`) only as the bring-up oracle.**  The
composition is *possible today* (`rms_norm` takes any row length,
`norm.py:85-91`; `mul_sized`/`add_sized` take a compiled-in 3840,
`eltwise.py:295-346`) but it is the wrong steady state: the fixed 1024
eltwise tiles do not divide a 3840 row (3840 = 3×1024 + 768,
`eltwise.py:27`, `mul.cc:137-139`), the two- or three-pass chains materialize
bf16 intermediates and round 2–3× where the fused form rounds once
(`refs.py` measurements: adaLN two-pass ≤ 3.4 half-ulps of
`|x(1+s)|+|shift|` above fused), and adaLN raw `scale` even forces a pass to
build `1+scale` because `mul_add` is `a*b` OR `a+b`, never `a*b+c`
(`scale_shift.cc:17-27`, `eltwise.py:230`).  At 96 tokens the call counts are
288–1080 per modulate composed vs 96 fused; the fused kernels also save the
intermediate buffer and (for adaLN) can absorb the GEMM corner turn in their
store loop.  Bring-up first with the composed path — it validates shapes,
eps and tolerances against shipped, measured kernels (`norm.py:34-43`,
`eltwise.py:43-48`) — then swap in the two fused kernels behind the same
numpy references.

## Concrete op sequence, one DiT block (96 × 3840 stream)

| # | Op | Kernel | Tiling / calls |
|---|----|--------|----------------|
| 1 | `x̂₁ = rms_norm(x) * γ₁` | **new** `rms_norm_weighted` (bring-up: `rms_norm(3840)` + `mul_sized(3840)`) | 1 row/call → 96 calls |
| 2 | `m₁ = x̂₁*(1+s₁)+b₁` | **new** `adaln_modulate` (bring-up: `add_sized` ones→`g`, `mul_sized`, `add_sized`) | 1 row/call → 96 calls |
| 3 | corner turn + QKV | `mm_bfp(mixed=True)` (`linalg.py:920`); A via `to_stream` `tile((8,8))` (`linalg.py:404`) | (96,3840)×(3840,≤3840·3); 96%16, 3840%8 OK (`mm_bfp_mixed.cc:126-128`) |
| 4 | attention | QKᵀ, softmax_rows-style unmasked (see `attn/DESIGN.md`), P·V | 30 heads × (96×128) |
| 5 | `x = x + attn_out` | `add_sized(3840)` (`eltwise.py:295`) | 96 calls |
| 6 | `x̂₂ = rms_norm(x) * γ₂` | **new** `rms_norm_weighted` | 96 calls |
| 7 | `m₂ = x̂₂*(1+s₂)+b₂` | **new** `adaln_modulate` | 96 calls |
| 8 | FFN up+gate, SwiGLU, down | `mm_bfp` 3840→10240 (×2), `swiglu` (10×1024 tiles, `activation.py:505`), `mm_bfp` 10240→3840 | B held bfp16ebs8 |
| 9 | `x = x + ffn_out` | `add_sized(3840)` | 96 calls |

Side input: one cond-projection GEMM producing `(s₁,b₁,s₂,b₂)` per token
(4 × (96, 3840) bf16 tiles — head count for gates/extra modulations: **not
determinable from the sources read**).  γ₁, γ₂ are per-channel bf16 (7 680 B
each; f32 checkpoint → `convert_copy`, `datamovement.py:167-208`).

## Why not stay composed

1. **Tiling friction.** The fixed-1024 `mul`/`add`/`mul_add`
   (`eltwise.py:27`) cannot tile 3840 per row; every composed scheme either
   switches to `*_sized` anyway (so no kernel is "saved") or tiles flat over
   the 96-row stream with per-token-replicated gamma (737 KB host tensor).
2. **Roundings.** Composed adaLN stores bf16 for `g`, `t` and `y`; fused
   stores once in f32-accumulators (`add.cc:39-51` idiom).  Same for
   norm+gamma (2 stores vs 1).
3. **Passes & bytes.** Fused adaLN is 1 pass over x/s/b → y; composed is 3
   (or 2 with host-prepared `1+s`) with a `t` intermediate.  Weighted norm
   fused reads x once; composed reads it twice and writes `x̂`.
4. **Layout.** The corner turn between row-major norm/adaLN and the GEMM's
   (8,8,8) A layout is unavoidable (`FINDINGS.md:77-99`), but it can live
   fused in `adaln_modulate`'s store loop instead of a dedicated pass —
   impossible with composed shipped kernels.

## Why not reuse `affine_cast` / Gemma4's weighted norm

`affine_cast` is per-column static gamma/beta as a baked `Param`, f32→bf16
(`datamovement.py:211-260`) — wrong axis, wrong lifetime, wrong dtype for
per-token modulation.  Gemma4's `rms_norm(y, x, w)` (`flm_gemma4/
rms_norm.h:53-68`) is the *right arithmetic* but only reachable through
geometry-locked fused decode kernels (model_dim 1536/2560,
`flm_gemma4.py:903/:919`); port its loop body into the generic
`rms_norm_aie2p.h` shape instead (runtime `cols`, tail handling), which is
exactly the proposed `rms_norm_weighted`.

## Fallbacks / milestones

* **M0 (no C++):** composed path end-to-end at (96, 3840) — `rms_norm(3840)`
  + `mul_sized(3840)` and `add_sized`+`mul_sized`+`add_sized`; judge with
  `refs.py` two-pass models.
* **M1:** `adaln_modulate` fused (bigger win: 3 passes → 1; also hosts the
  corner turn if desired).
* **M2:** `rms_norm_weighted` fused.
* Keep the composed path as the regression oracle for both.
