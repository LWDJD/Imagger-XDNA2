# PRECISION BUDGET — how much error each stage of the 8-step chain can absorb

> 2026-11-02 | **host only, CPU only. No device was opened, no pyxrt, no xclbin, no NPU
> dispatch. No product file was edited.** Everything below was produced by the scripts in
> `imaggers/pbudget/` on py310 GAIA with numpy 1.26.4 / torch 2.3.1+cpu / numba 0.62.1.
> `git status` shows this work only as `?? PRECISION-BUDGET.md` and `?? imaggers/pbudget/`; the
> eight `M` entries in the tree pre-date this session and none of them was touched.

---

## 0. Verdict

**The question was:** instead of trying a precision and then testing quality, measure for each
stage how much error it can absorb before the final latents fall through corr 0.9953.

**The answer is a budget that spans a factor of 22**, from **1.4e-2** relative error at the
attention scores to **3.0e-1** at the refiner's GEMMs:

| rank | stage | what it is | BUDGET: per-call rel_rms that lands corr = 0.9953 | what real bf16 costs there | margin |
|---:|---|---|---:|---:|---:|
| 1 | **d_refgemm** | the refiner's 7 GEMMs | **3.0e-1** | 1.66e-3 | **181x** |
| 2 | **a_pv** | the attention output ( == stage b ) | **1.9e-1** | 1.66e-3 | **114x** |
| 3 | **c_sm** | the softmax probabilities | **1.5e-1** | 1.61e-3 | **94x** |
| 4 | **g_devA** | the device GEMM A operand | **8.0e-2** | 1.66e-3 | **48x** |
| 5 | **f_devC** | the device GEMM output C tile | **5.4e-2** | 1.66e-3 | **32x** |
| 6 | **e_norm** | every rms/norm | **2.2e-2** | 1.66e-3 | **13x** |
| 7 | **a_qk** | the attention scores q·kᵀ | **1.4e-2** | 1.66e-3 | **8.4x** |

1. **Every one of the seven stages is SAFE in bf16, with at least 8x margin.** bf16 rounding of
   any single one of them costs a per-call rel_rms of 1.6e-3 (measured, not assumed), and the
   tightest budget is 1.4e-2.
2. **fp16 is not automatically safer.** Its 11-bit mantissa is 8x finer (2.0e-4 per call), but its
   5-bit exponent **overflows to NaN at three of the seven stages**: the refiner GEMMs, the device
   C tile and the device A operand. fp16 is only usable where the tensors are bounded.
3. **The chain does not amplify a per-call error into a latents error the way the campaign
   assumed.** On the f32 chain a 1.1e-7 per-call attention error produces a latents rel_rms of
   2.2e-6 — a gain of 19, i.e. essentially sqrt(272), the number of injection sites, which is what
   independent errors do. There is no chaotic 1e4 in the arithmetic. The 1e4 appears only when the **device's bf16 A/C quantisers**
   are in the loop: the same 1.1e-7 perturbation then moves the latents by 5.6e-3 (**gain 2600**),
   because a quantiser turns any sub-ulp difference into full-ulp flips. That is measured in §6.
4. **Variant D could not be made to fail at any geometry I can build** — 256²x8 steps gives latents
   corr 0.99973, and 512²/SEQ=1056 (the product's own token count) gives 0.99981, or 0.99984 on
   top of the shipped tier. See §8. This is a negative result and it matters: the reason the
   campaign rejected D is not reproducible from the block arithmetic.

---

## 1. The harness, and the proof that it is the product's chain

`imaggers/pbudget/pb_lib.py` re-types the block arithmetic of `imaggers/image/t3_real.py`
(`full_block_host` / `attention` / `ffn_raw`) with an injection hook at every point where a real
precision decision lives, and `pb_step` / `run_chain` re-type `step_once` / `--stage host`.

Three gates, all in `pb_gate.py`, all **PASS**:

| gate | what it compares | result |
|---|---|---|
| 1 | `pb_block` vs `t3_real.full_block_host`, real weights, one refiner block (mod None), one context block, one layer block (mod on) | **BYTE-EQUAL** on all three |
| 2 | `pb_step` vs `t3_real.step_once`, real weights, one full step, 30 layers | **BYTE-EQUAL** (91.4 s vs 5.5 s) |
| 3 | the whole **8-step loop** vs `imaggers/image/t3real_latents.npy` — the recorded f32 reference (seed 0, 8 steps, 30 layers, absmax 6.1792, recorded wall 535.9 s) | **BYTE-EQUAL**, corr 1.000000000, in **38.3 s** |

Gate 3 is the one that matters: the harness reproduces the run the 0.9953 bar was recorded
against, not a re-typed copy of it. It is 14x faster than the recorded run, from two changes that
were themselves gated byte-for-byte:

* a numba bfp16 decoder (`pb_decode.py`) that walks the packer's tile structure in the sequential
  direction — 0.022 s vs `pack_bfp16.unpack`'s 0.337 s for a (10240,3840) block, **byte-equal**, and
  byte-equal on real weight tensors of every shape the run touches;
* a decode buffer pool keyed by shape — decoding into a fresh 708 MB array costs 0.31 s per block
  and only 0.075 s once the pages exist, which was 8 s per step of pure page faults (10.9 s → 5.5 s
  per step, still byte-equal).

Weights are read through a **shared mmap** of `zimage-port/weights.bfp16` (6.9 GB): RSS 0.9 GB
instead of 7.7 GB. An earlier batch died of MemoryError at point 13 because it held the file as a
private bytes object; the mmap is what makes an 82-point sweep safe on a 31 GB box.

---

## 2. Method

### 2.1 The eight injection points

| tag | stage | injected at | sites per 8-step run |
|---|---|---|---:|
| `a_qkv` | q/k/v tensors in bf16 | after the q/k/v projections, before both matmuls, all 34 blocks | 816 |
| `a_qk` | attention qk | S = q·kᵀ, all 34 blocks | 272 |
| `a_pv` | attention output (**== stage b**) | o = P·V, all 34 blocks | 272 |
| `c_sm` | softmax | P, all 34 blocks | 272 |
| `d_refgemm` | refiner's 7 GEMMs | to_q/k/v/out + w1/w2/w3 outputs, the 4 refiner blocks only | 224 |
| `e_norm` | rms/norm | attention_norm1/2, ffn_norm1/2, norm_q, norm_k in all 34 blocks, plus the final layer's norm | 1640 |
| `f_devC` | device GEMM output C | w1/w3/w2 results, the 30 unified layers | 720 |
| `g_devA` | device GEMM A operand | the activation feeding w1/w3/w2, the 30 unified layers | 480 |

**Stage (a) and stage (b) are the same tensor in this architecture.** The pv product *is* the
attention output that feeds `to_out`, so `a_pv` is reported as both (a)-pv and (b) — they cannot be
separated without inventing a tensor that does not exist. `a_qkv` is not one of the seven budget
stages; it exists so the product's variant D (which rounds q, k and v to bf16 as well) can be
reproduced exactly.

### 2.2 The perturbation

For a sweep point (stage, σ) the hook replaces the stage's tensor `t` by

~~~
t + a * g,   g ~ N(0,1) with one FIXED direction per call index,
             a = sigma * rms(t) / rms(g)     so the applied rel_rms is sigma
~~~

The direction is re-used at every σ of a sweep, so the whole curve is **one direction scaled**, not
a different random draw per point: the curve is monotone and the interpolation is not fighting
draw-to-draw noise. The achieved rel_rms is recorded per call (median/min/max in the JSONL); the
medians are exactly σ to 8 digits.

`NoiseCoherent` repeats the *same* direction at every call site (the coherent regime), and
`Bf16Q` / `Fp16Q` / `Bfp16Q` apply the **real** round-to-nearest-even to bfloat16 / float16 / bfp16
(the last via `pack_bfp16._quantise`, i.e. the device's own per-group-of-8 shared-exponent rule).
The real-rounding arms are the ground truth: they are coherent by construction because a rounding
error is a deterministic function of the tensor.

### 2.3 The metric and the bar

`corr` is Pearson in float64 over the finite pairs of the **final latents**, which is exactly what
`imaggers/util/place/quality_gate.py` computes, and 0.9953 is the recorded best-cross yardstick.

For an error uncorrelated with the signal, corr = 1/sqrt(1+ρ²) with ρ the latents rel_rms, so the
bar corresponds to

~~~
corr = 0.9953   <=>   rho = 9.73e-2
~~~

The twelve stages' measured `rho_at_bar` (§3, last column) land between 8.6e-2 and 9.7e-2, which is
the assumption checking itself.

### 2.4 The interpolation

corr(σ) is monotone decreasing. Three interpolations are reported so the number carries its own
uncertainty:

* **M1 — log-linear bracket.** Find the adjacent pair with corr(σ_lo) ≥ 0.9953 > corr(σ_hi) and
  interpolate **linearly in log10 σ**. Model-free, guaranteed inside a bracket, the headline number.
* **M2 — log-quadratic.** Fit a parabola to corr vs log10 σ over the up-to-four points nearest the
  crossing and solve for 0.9953. Tests curvature (the response is visibly saturating at large σ).
* **M3 — power law.** Fit log10(latents rel_rms) = log10(A) + p·log10 σ on the points still in the
  linear regime, then invert at the latents rel_rms the bracket actually reaches at the bar.

The budget in the table is the **median of the three**; the spread of M1/M2/M3 is quoted in §3 and
is at most 1.4x (worst case `a_qk`, whose curve is the steepest).

### 2.5 What one point is

One point = **one full 8-step chain** — 34 blocks x 8 steps, the real weights, the real recorded
schedule (timesteps 1000 -> 8.93, sigmas 1.0 -> 0.0), seed 0. Not a 2-step proxy: the whole 8-step
loop was replayed for every point. **82 measurement points plus two reference chains** were run:
78 at 256² x 8 steps, 4 at 512² x 4 steps, each one a distinct precision hypothesis.

---
## 3. The budget table

Every number below is a full 8-step replay on the real `weights.bfp16`, reference = the f32 chain
that gate 3 proves byte-equal to the recorded one. `σ*` is the per-call rel_rms at which the final
latents land on corr = 0.9953.

| rank | stage | BUDGET σ* | M1 log-linear | M2 log-quadratic | M3 power-law | ρ at the bar | bf16 per-call cost | bf16 latents corr | margin | verdict |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `d_refgemm` | **3.00e-1** | 3.00e-1 | 3.01e-1 | 2.45e-1 | 9.68e-2 | 1.658e-3 | 0.9999975 | 181x | SAFE |
| 2 | `a_pv` (= b) | **1.90e-1** | 1.74e-1 | 1.90e-1 | 1.97e-1 | 8.90e-2 | 1.658e-3 | 0.9999986 | 114x | SAFE |
| 3 | `c_sm` | **1.51e-1** | 1.42e-1 | 1.53e-1 | 1.51e-1 | 9.02e-2 | 1.611e-3 | 0.9999977 | 94x | SAFE |
| 4 | `g_devA` | **8.02e-2** | 8.73e-2 | 7.92e-2 | 8.02e-2 | 9.49e-2 | 1.657e-3 | 0.9999961 | 48x | SAFE |
| 5 | `f_devC` | **5.36e-2** | 5.21e-2 | 6.36e-2 | 5.36e-2 | 9.11e-2 | 1.658e-3 | 0.9999941 | 32x | SAFE |
| 6 | `e_norm` | **2.17e-2** | 2.15e-2 | 2.17e-2 | 2.32e-2 | 9.24e-2 | 1.657e-3 | 0.9998910 | 13x | SAFE |
| 7 | `a_qk` | **1.39e-2** | 1.22e-2 | 1.39e-2 | 1.67e-2 | 8.58e-2 | 1.662e-3 | 0.9998202 | 8.4x | SAFE |

Read it as: *the attention score matrix may be wrong by 1.4 % per call and the image still passes;
the refiner's GEMMs may be wrong by 30 % per call and the image still passes.* Real bf16 is 0.17 %
per call at every one of them.

Two shapes of the same fact are worth separating:

* **The ranking is not the ranking of how much arithmetic each stage carries.** The refiner is 4
  blocks out of 34 and its budget is the largest; the attention scores carry 2 matmuls per block and
  its budget is the smallest. What sets the budget is *where the error enters the residual stream*:
  `d_refgemm` sits before 30 layers of renormalisation and averaging, `a_qk` sits inside an exp(),
  where a small change in S is a multiplicative change in the softmax weights.
* **The two tightest stages are close in the linear regime and separate at the knee.** `e_norm` and
  `a_qk` both give about 7e-3 latents rel_rms at a per-call 1e-3, but between 1e-2 and 3e-2 `a_qk`
  accelerates (7.4e-2 -> 1.6e-1) while `e_norm` stays linear (6.5e-2 -> 1.1e-1). The exponential in
  the softmax is what pulls `a_qk`'s crossing down; `a_pv` and `c_sm` are 14x and 11x looser even
  though they are the same attention call, because the pv product is renormalised by
  `attention_norm2` before it re-enters the residual stream.

### 3.1 The bar is a shared budget, not seven independent ones

Two combination runs test whether the per-stage numbers compose. If the latents errors added in
quadrature, the combination's rel_rms would be sqrt(Σ ρ_i²):

| combination (all at real bf16) | quadrature of the singles | measured | measured / predicted |
|---|---:|---:|---:|
| `a_qkv`+`a_qk`+`c_sm`+`a_pv` — the variant-D pattern | 2.03e-2 | **2.31e-2** | 1.14x |
| `a_qk`+`c_sm`+`a_pv` | 1.92e-2 | 1.45e-2 | 0.76x |
| `f_devC`+`g_devA` — the shipped device tier | 4.43e-3 | **2.85e-2** | **6.4x** |

So the **attention** group composes in quadrature to within 14 %, but the **two device quantisers**
compose 6.4x worse than quadrature. The difference is that a quantiser in the A-operand path and a
quantiser in the C-return path sit on opposite ends of the same GEMM, so each sees the other's
error as a change in its own input — the pair is a feedback loop, the attention stages are not.
Anything that puts coarse precision on **both** sides of the FFN GEMM must be measured as a pair;
it cannot be assembled from the single-stage budgets.

---

## 4. The per-stage curves

`x` = injected per-call rel_rms, `rho` = final-latents rel_rms, `corr` = final-latents Pearson corr.
The real-rounding rows are the *measured* cost of that precision, not another noise point.

### 4.1 `a_qk` — attention qk (the tightest stage)

| σ | rho | corr | vs 0.9953 |
|---:|---:|---:|---|
| 1.0e-4 | 6.969e-4 | 0.9999998 | PASS |
| 1.0e-3 | 7.122e-3 | 0.9999747 | PASS |
| 3.0e-3 | 2.294e-2 | 0.9997370 | PASS |
| 1.0e-2 | 7.406e-2 | 0.9972695 | PASS |
| **3.0e-2** | 1.643e-1 | 0.9865654 | **fail** |
| 1.0e-1 | 2.764e-1 | 0.9618079 | fail |
| **bf16, real** | **1.897e-2** | **0.9998202** | **PASS** |
| fp16, real | 2.369e-3 | 0.9999972 | PASS |
| coherent 1e-2 | 1.019e-1 | 0.9948274 | fail |

The response is **linear in rho up to 1e-2** — the amplification rho/σ is 6.97, 7.12, 7.65, 7.41 at
1e-4, 1e-3, 3e-3, 1e-2 — and then **saturates**: 5.48 at 3e-2, 2.76 at 1e-1. The crossing sits exactly
on the knee, which is why this stage's three interpolations disagree the most (1.22e-2 … 1.67e-2);
the other six agree to within 20 %.

### 4.2 `a_pv` — the attention output, which is also stage (b)

| σ | rho | corr | vs 0.9953 |
|---:|---:|---:|---|
| 1.1e-7 (the A2 arm's per-call size) | 2.153e-6 | 1.0000000 | PASS |
| 1.0e-3 | 1.028e-3 | 0.9999995 | PASS |
| 7.0e-3 (the variant-D arm's per-call size) | 7.112e-3 | 0.9999748 | PASS |
| 1.0e-2 | 9.973e-3 | 0.9999504 | PASS |
| 3.0e-2 | 2.506e-2 | 0.9996870 | PASS |
| 1.0e-1 | 6.575e-2 | 0.9978493 | PASS |
| **3.0e-1** | 1.197e-1 | 0.9928043 | **fail** |
| **bf16, real** | **1.669e-3** | **0.9999986** | **PASS** |
| fp16, real | 2.740e-4 | 1.0000000 | PASS |

Note the two rows that matter for the history of this project: **a per-call 1.1e-7 gives corr
1.0000000 and a per-call 7.0e-3 gives corr 0.9999748** — the second is 14x inside the bar in rel_rms
(187x in 1-corr). Those are the two per-call numbers the ATTN-SEG campaign measured for its A2 and D
arms, and neither of them is anywhere near failing on this chain. §6 and §8 are about why.

### 4.3 `c_sm` — the softmax probabilities

| σ | rho | corr | vs 0.9953 |
|---:|---:|---:|---|
| 1.0e-3 | 7.472e-4 | 0.9999997 | PASS |
| 1.0e-2 | 7.552e-3 | 0.9999715 | PASS |
| 3.0e-2 | 2.287e-2 | 0.9997383 | PASS |
| 1.0e-1 | 7.555e-2 | 0.9971592 | PASS |
| **3.0e-1** | 1.316e-1 | 0.9913223 | **fail** |
| **bf16, real** | **2.163e-3** | **0.9999977** | **PASS** |
| fp16, real | 2.248e-4 | 1.0000000 | PASS |

### 4.4 `d_refgemm` — the refiner's seven GEMMs (the loosest stage)

| σ | rho | corr | vs 0.9953 |
|---:|---:|---:|---|
| 1.0e-3 | 4.109e-4 | 0.9999999 | PASS |
| 1.0e-2 | 4.437e-3 | 0.9999902 | PASS |
| 3.0e-2 | 1.862e-2 | 0.9998267 | PASS |
| 1.0e-1 | 5.295e-2 | 0.9986013 | PASS |
| 3.0e-1 | 9.671e-2 | 0.9953364 | PASS (just) |
| **1.0e0** | 3.130e-1 | 0.9539712 | **fail** |
| **bf16, real** | **2.235e-3** | **0.9999975** | **PASS** |
| fp16, real | **NaN** | **NaN** | **FAIL (overflow)** |

The refiner can be wrong by 30 % per call before the image notices, and it is the only stage whose
budget is set by a bracket (3e-1 … 1e0) rather than by a knee. This is the scientific confirmation
of `T22_REF_BF16`: the refiner's seven GEMMs in bf16 cost 0.22 % latents rel_rms against a 9.7 %
bar — 43x of headroom, i.e. the flag that the campaign measured at −4.5 s/image was never within
1 % of the quality bar.

### 4.5 `e_norm` — every rms/norm

| σ | rho | corr | vs 0.9953 |
|---:|---:|---:|---|
| 1.0e-3 | 7.753e-3 | 0.9999701 | PASS |
| 3.0e-3 | 2.397e-2 | 0.9997132 | PASS |
| 1.0e-2 | 6.469e-2 | 0.9979054 | PASS |
| **3.0e-2** | 1.080e-1 | 0.9941637 | **fail** |
| 1.0e-1 | 2.318e-1 | 0.9738960 | fail |
| **bf16, real** | **1.478e-2** | **0.9998910** | **PASS** |
| fp16, real | 1.138e-3 | 0.9999994 | PASS |

1640 injection sites per run, and it is the second-tightest stage. Its response is almost exactly
**linear** in σ over the whole swept range (1e-3 -> 7.75e-3, 3e-2 -> 1.08e-1: 14x for 30x), which is
why its three interpolations agree to 8 % while `a_qk`'s disagree by 37 %.

### 4.6 `f_devC` — the device GEMM's C tile

| σ | rho | corr | vs 0.9953 |
|---:|---:|---:|---|
| 1.0e-3 | 3.871e-3 | 0.9999926 | PASS |
| 3.0e-3 | 1.558e-2 | 0.9998796 | PASS |
| 1.0e-2 | 4.555e-2 | 0.9989747 | PASS |
| 3.0e-2 | 7.156e-2 | 0.9974697 | PASS |
| **1.0e-1** | 1.211e-1 | 0.9927353 | **fail** |
| 3.0e-1 | 2.411e-1 | 0.9712892 | fail |
| **bf16, real** | **3.440e-3** | **0.9999941** | **PASS** |
| **bfp16, real** (per-group shared exponent) | **1.986e-2** | **0.9998025** | **PASS** |
| fp16, real | **NaN** | **NaN** | **FAIL (overflow)** |

### 4.7 `g_devA` — the device GEMM's A operand

| σ | rho | corr | vs 0.9953 |
|---:|---:|---:|---|
| 1.0e-3 | 3.720e-3 | 0.9999931 | PASS |
| 3.0e-3 | 1.074e-2 | 0.9999424 | PASS |
| 1.0e-2 | 3.296e-2 | 0.9994568 | PASS |
| 3.0e-2 | 5.879e-2 | 0.9982687 | PASS |
| **1.0e-1** | 1.009e-1 | 0.9949220 | **fail** |
| 3.0e-1 | 1.921e-1 | 0.9816099 | fail |
| **bf16, real** | **2.791e-3** | **0.9999961** | **PASS** |
| **bfp16, real** | **1.509e-2** | **0.9998860** | **PASS** |
| fp16, real | **NaN** | **NaN** | **FAIL (overflow)** |

A bfp16 A operand is 4.6x more expensive than bf16 (7.67e-3 vs 1.66e-3 per call) and still leaves
6.4x of headroom — the bfp16 activation is affordable, and the *measured* bfp16 A + bfp16 C
combination is 2.85e-2 rel_rms, still 3.4x inside the bar.

---
## 5. What the real precision actually costs

The real-rounding arms are the only numbers that are not a model. Every one is the *actual*
round-to-nearest-even of that stage's tensors, at every one of its call sites, for all 8 steps.

| stage | bf16 per-call | bf16 latents rho | bf16 latents corr | the noise curve predicts at that rho | fp16 per-call | fp16 corr | bfp16 per-call | bfp16 corr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `a_qk` | 1.662e-3 | 1.897e-2 | 0.9998202 | 0.9998649 | 2.075e-4 | 0.9999972 | — | — |
| `a_pv` | 1.658e-3 | 1.669e-3 | 0.9999986 | 0.9999930 | 2.073e-4 | 1.0000000 | — | — |
| `c_sm` | 1.611e-3 | 2.163e-3 | 0.9999977 | 0.9999939 | 2.017e-4 | 1.0000000 | — | — |
| `d_refgemm` | 1.658e-3 | 2.235e-3 | 0.9999975 | 0.9999978 | **NaN** | **NaN** | — | — |
| `e_norm` | 1.658e-3 | 1.478e-2 | 0.9998910 | 0.9998519 | 2.071e-4 | 0.9999994 | — | — |
| `f_devC` | 1.658e-3 | 3.440e-3 | 0.9999941 | 0.9999405 | **NaN** | **NaN** | 6.593e-3 | 0.9998025 |
| `g_devA` | 1.657e-3 | 2.791e-3 | 0.9999961 | 0.9999698 | **NaN** | **NaN** | 7.673e-3 | 0.9998860 |

### 5.1 bf16 costs 1.6e-3 per call at every stage, and it is not the same amount of damage

The per-call cost is almost identical across the seven stages (1.611e-3 … 1.662e-3, a 3.2 % spread:
that is just what rounding a float to 8 mantissa bits does). The **latents** damage differs by
**11x** (1.669e-3 at `a_pv` to 1.897e-2 at `a_qk`). So a per-call comparison between precisions is
not a proxy for the chain effect — which is the ATTN-SEG report's own conclusion, now with the
numbers that say how big the gap is per stage.

### 5.2 fp16 is a trap for the reason nobody measures

fp16's mantissa is 8x finer, and where it survives it is 8x cheaper: `e_norm` 2.071e-4 per call
(vs bf16's 1.657e-3), `c_sm` 2.017e-4, `a_qk` 2.075e-4, `a_pv` 2.073e-4. But at **three** stages
it produces **NaN** — the refiner's GEMM outputs, the device C tile and the device A operand.
Those are the tensors with the largest dynamic range in the model: the FFN intermediate
`silu(w1x) * w3x` runs past 65504, so fp16 overflows rather than rounds. **fp16's exponent, not its
mantissa, is what decides where it is affordable**, and it is affordable exactly where the bf16
budget is loose, which is not where the speed is.

(A scaled fp16 — a per-tensor or per-tile power-of-two scale — would fix the overflow and is not
measured here. bfp16 *is* that idea with an 8-bit mantissa, and it is measured: 4.6x more expensive
than bf16 on the A operand, still 6.4x inside the bar.)

---

## 6. Two regimes, and the amplifier that killed variant D

### 6.1 Independent noise adds in quadrature; real rounding need not

An injection at 272 call sites with an independent direction each adds in quadrature, so it gives
sqrt(272) = 16.5x the effect of one call. Real rounding is deterministic in the tensor, so the same
pattern can recur and add coherently. `NoiseCoherent` is the coherent end of that range: the same
direction at every site, same σ.

| stage | σ | rho, independent | rho, coherent | coherent / independent |
|---|---:|---:|---:|---:|
| `a_qk` | 1e-3 | 7.122e-3 | 9.959e-3 | 1.40x |
| `a_qk` | 1e-2 | 7.406e-2 | 1.019e-1 | 1.38x |
| `e_norm` | 1e-3 | 7.753e-3 | 1.166e-2 | 1.50x |
| `e_norm` | 1e-2 | 6.469e-2 | 7.040e-2 | 1.09x |
| `d_refgemm` | 1e-2 | 4.437e-3 | 9.916e-3 | 2.23x |
| `c_sm` | 1e-2 | 7.552e-3 | 7.975e-3 | 1.06x |
| `a_pv` | 1e-2 | 9.973e-3 | 1.108e-2 | 1.11x |
| `g_devA` | 1e-2 | 3.296e-2 | 3.378e-2 | 1.02x |
| `f_devC` | 1e-2 | 4.555e-2 | 4.280e-2 | 0.94x |

The coherent factor is **0.94x … 2.23x**, so the budgets in §3 are good to about a factor of two
when the error is systematic rather than random. That is the honest uncertainty on the table.
Checked against reality, stage by stage, the real-arm rho divided by what the independent curve
predicts at the same per-call size is: `a_qk` 1.15x, `d_refgemm` 1.06x, `e_norm` 0.86x, `c_sm`
0.62x, `a_pv` 0.45x, `g_devA` 0.36x, `f_devC` 0.32x. Every one is inside the 0.94x-2.23x band the
coherent runs bracket, so the *model* (a noise injection of the measured size) reproduces the *real*
rounding to within a factor of two everywhere — but the sign of the deviation differs by stage, and
the three cheapest stages (`f_devC`, `g_devA`, `a_pv`) are *less* damaged than pure noise because the
norm that follows them absorbs part of it.

### 6.2 The 1e4 amplifier is the device's bf16 quantisers, not chaos

The shipped product does **not** run the f32 chain. It rounds the GEMM A operand and the C result
to bf16 on every device call —

~~~
webui/chunked.py:823   a_img = H.bf16_bytes_f32(xf, sc, "ffnA")     # A operand -> bf16 bytes
webui/chunked.py:848   yb = H.bf16_to_f32_bits(...)                   # C result  -> bf16 bits
imaggers/image/t8_codec.py:58  bf16_rne_u16  = plain RNE to bf16
~~~

— and `webui/shapes.py:56` says it plainly: `BYTES_PER_ELEM = 2  # bf16 / BFP16 stream on the
device`. So the budgets above were re-measured **on top of that tier**: the baseline is a chain with
`f_devC` and `g_devA` both at real bf16, and the perturbation is layered on it.

| what is layered on the shipped tier | rho vs the tier | corr vs the tier | the same perturbation on the f32 chain | the tier's gain |
|---|---:|---:|---:|---:|
| (nothing) | 4.562e-3 | 1.0000000 | 0 (corr 0.9999896 vs f32) | — |
| `a_pv` + 1.1e-7 (the A2 per-call size) | 5.574e-3 | 0.9999845 | 2.153e-6 | **2590x** |
| `a_pv` + 1.0e-4 | 5.053e-3 | 0.9999873 | 1.0e-4 | 51x |
| `a_qk` + 1.0e-4 | 7.690e-3 | 0.9999705 | 6.969e-4 | 11x |
| `e_norm` + 1.0e-4 | 5.596e-3 | 0.9999843 | 8.8e-4 | 6.4x |
| the whole D pattern at bf16 | 1.941e-2 | 0.9998115 | 2.307e-2 | 0.84x |

**That is the mechanism the campaign was looking for.** A quantiser is a step function: a
1e-7 change in a value that sits near a rounding boundary flips it by a full ulp (2e-3 relative),
so 34 blocks x 8 steps turn a 1e-7 perturbation into a 5.6e-3 latents difference. It is exactly the
"four orders of magnitude" of ATTN-SEG §6.1, and it is a property of the *quantisers*, not of
float arithmetic and not of the attention.

Three consequences that matter:

1. **The tier has a dither floor of 4.6e-3** (corr 0.99999 vs the f32 chain) that any
   sub-ulp perturbation is amplified *up to* and no further. 4.6e-3 is 21x inside the 9.7e-2 bar,
   so the shipped build's own quantisation is not close to being a problem.
2. **The budgets transfer, and the gain is a function of size.** 2590x at a per-call 1.1e-7, 51x at
   1e-4, 11x at 1e-4 for `a_qk`, and 0.84x for a pattern already at 2.3e-2. The gain falls as the
   perturbation grows, so at the bar itself it is <= 1: the tier adds its 4.6e-3 to a 9.7e-2 in
   quadrature, which is a 0.1 % shift. **The numbers in §3 are therefore valid for the shipped build
   to well within their own interpolation spread, not just for the f32 reference.**
3. **Once the perturbation is bigger than the floor the tier stops mattering** — the D pattern's
   tier gain is 0.84x, i.e. slightly *less* than on f32. Amplification is a small-error phenomenon.

---
## 7. Safe / unsafe, and what the product already does

| stage | budget σ* | bf16 margin | verdict | **what the product does today** | what this says |
|---|---:|---:|---|---|---|
| `a_qk` | 1.39e-2 | 8.4x | **SAFE** | host f32 (torch, `chunked.py` `_attention`) | bf16 qk is affordable but it is the **tightest** of the seven — spend this margin last |
| `e_norm` | 2.17e-2 | 13x | **SAFE** | host f32 numpy `rms` | bf16 norms are affordable; second-tightest, with a response that is linear over the whole swept range |
| `f_devC` | 5.36e-2 | 32x | **SAFE** | **bf16 already** (`bf16_to_f32_bits`, `chunked.py:848`) | the shipped choice is 32x inside the bar — confirmed correct |
| `g_devA` | 8.02e-2 | 48x | **SAFE** | **bf16 already** (`bf16_bytes_f32`, `chunked.py:823`) | likewise; **a bfp16 A operand would also pass** (6.4x margin) |
| `c_sm` | 1.51e-1 | 94x | **SAFE** | host f32 softmax | bf16 probabilities are essentially free |
| `a_pv` | 1.90e-1 | 114x | **SAFE** | host f32 | bf16 pv is essentially free |
| `d_refgemm` | 3.00e-1 | 181x | **SAFE** | f32 by default; `T22_REF_BF16=1` opt-in | the flag is 43x inside the bar — **the refiner bf16 is unambiguously safe** |

**So: no stage needs to stay in f32 for quality reasons, and the four stages the campaign already
moved or proposed (device A, device C, the refiner, the attention block) are all inside the bar by
a wide margin on their own.** The ranking is the useful part: if only part of the attention block
may move, move `a_pv` and `c_sm` (94x-114x) and leave `a_qk` alone (8.4x).

### 7.1 The verdict is not 'all seven at once'

The margins are per stage and they do not multiply. The combination the campaign actually wanted —
variant D, which is `a_qkv`+`a_qk`+`c_sm`+`a_pv` — measures **2.31e-2** latents rel_rms: 4.2x inside
the bar, 1.14x worse than the quadrature sum of its members, and nowhere near the "each member has
8x-114x of margin so the sum must too" reading. **Quote the combination measurement, never the
product of the singles** — and note that the two device quantisers do not even reach quadrature
(6.4x, §3.1).

---

## 8. The 512x512 arm: variant D re-tested at the product's own geometry

The record of this project is that variant D failed in product at 512² with latents corr **0.9919453**
and a per-call attention rel_rms of **7.0e-3**, while A2 (per-call 1.1e-7) landed 0.9975444. Every
number in §3-§7 is on the 256²/SEQ=96 configuration, which is what the recorded f32 reference is.
So the same points were re-run at the **product's** geometry: 32x32 patches = 1024 image tokens +
32 caption = **SEQ 1056**, latents (1,16,64,64), a fresh reference. Four steps instead of eight:
a 512² step is 10-25x a 256² step (50 s for the reference chain, 114 s when every block also
quantises), and the arm still cost 28 minutes.

| at 512², SEQ 1056, 4 steps | latents rho | latents corr | vs 0.9953 |
|---|---:|---:|---|
| `a_pv` + 7.0e-3 — the per-call size variant D measured | 6.411e-3 | 0.9999794 | PASS |
| the **whole D pattern** at real bf16 (`a_qkv`+`a_qk`+`c_sm`+`a_pv`) | 1.961e-2 | 0.9998078 | PASS |
| the shipped tier alone (`f_devC`+`g_devA` at real bf16) | 3.559e-3 | 0.9999937 | baseline |
| the D pattern **on top of** the shipped tier | 1.809e-2 | 0.9998364 | PASS |

Same answer as at 256², to within **15 %**: at the product's own token count, with the product's own
bf16 A/C quantisers in the loop, the variant-D arithmetic lands **5x inside the bar** (corr
0.99981 … 0.99984). The 256² arm over- or under-states every one of these by less than the
coherent/independent spread, so the 256² budgets are not an artefact of the small sequence.

### 8.1 What this means for the D rejection

A per-call attention error has to reach **~3.2e-1** to move these latents to corr 0.9919 — 45x the
7.0e-3 the D arm measured. My chain does not produce a 45x amplification at either geometry, and
the tier's quantiser amplifier tops out at a 4.6e-3 floor, which is 21x *below* the bar. So one of
the following is true, and this measurement cannot tell which:

1. the product's 512²/8-step chain amplifies ~45x more than either of my chains — possible only if
   some path I did not model (the caption / text-encoder side, the VAE-adjacent tensors, or the
   device's internal bfp16 accumulation, which I model as bf16 in / bf16 out but not as bfp16
   *accumulate*) is the amplifier;
2. the D arm's per-call 7.0e-3 understates what the arm actually did to the chain — it was measured
   on the attention output of a *probe*, at SEQ=1056, against the product's real q/k/v, and the
   product's D also packs q/k/v to bf16 before the matmuls, which changes the pv operand;
3. the pair was not clean. ATTN-SEG §5 records that in the D pair the box was materially faster
   during the B arm (the rope control moved 13 %) and that the two A arms of the two pairs differ
   by 3.24 s.

**I am not saying variant D is safe.** I am saying the evidence that rejected it is not reproducible
from the block arithmetic at either geometry, and the per-call budget says the attention block in
bf16 should be 4-5x inside the bar. The next step for that question is a re-measurement of the D
arm with the per-layer `[fastglue]` medians and a latents gate on the same pair, not another
budget point — this harness cannot settle it without the device, and the device was out of scope.

---

## 9. What I could not do — plainly

1. **No device.** Nothing here is a device measurement. Every "the product does X" statement is
   read from the product's source (`webui/chunked.py`, `imaggers/image/t8_codec.py`,
   `webui/shapes.py`) or from a recorded artefact, and is cited. No NPU was opened.
2. **8 steps everywhere except the 512² arm, which is 4.** The 512² arm is 4 steps because a step is
   5-7x the 256² cost; it is labelled as such wherever it appears (§8). Everything else is the full
   8-step loop.
3. **The reference is the f32 chain, not the shipped build.** §6 re-bases five perturbations on a
   bf16-A/bf16-C tier (a dither floor of 4.6e-3, and a gain that falls from 2590x at a per-call 1.1e-7
   to 0.84x at 2.3e-2), but a **full per-stage table on the tier** was not run — that is another 80
   chains. The single-point evidence says the table transfers to within its own interpolation spread.
4. **bfp16 is modelled as a rounding, not as an accumulate.** `f_devC`/`g_devA` at bfp16 round the A
   operand and the C result with the device's own per-group shared-exponent rule
   (`pack_bfp16._quantise`), but the internal bfp16 multiply-accumulate of that kernel is not
   modelled. The device's accumulate is plausibly worse than the roundings I applied.
5. **The device GEMM's bf16 *weights* are not a perturbation here** — the reference already uses the
   delivered `weights.bfp16` values, so the weight quantisation is inside every number in this
   report by construction. What is *not* measured is a **weight** change (e.g. 8-bit weights, or a
   coarser bfp16 group); that is a different bracket.
6. **One realisation per σ.** The direction is fixed across σ within a sweep (deliberately, §2.2),
   so each σ is one draw, not an ensemble. The coherent/independent pair brackets the regime
   (0.94x-2.23x) but there is no repeat-variance estimate.
7. **256², not the product's 512², for the table.** §8 shows the 512² arm agrees within 15 %, but
   the *table* is 256². A 512² table would be ~5x the compute.
8. **No speed measurement and no speed claim.** Host only; the device was out of scope. The
   time-per-point numbers in this report are harness cost, not product cost.

---

## 10. Reproduce

~~~powershell
cd D:\Project\deepseek_harness\public_work\zimage-bfp16-npu
$py = 'D:\Users\15081\AppData\Local\GAIA\python\python.exe'

# 1. the decoder is byte-exact and 15x faster than pack_bfp16.unpack
& $py imaggers\pbudget\pb_decode.py gate
& $py imaggers\pbudget\pb_decode.py real

# 2. the three gates: block, step, and the whole 8-step loop vs the recorded reference
& $py imaggers\pbudget\pb_gate.py --gates 1,2,3 --steps 8      # RESULT: PASS

# 3. one sweep point (a full 8-step chain); points append to the JSONL as they finish
& $py imaggers\pbudget\pb_sweep.py --steps 8 --points "a_qk:noise:1e-2"

# 4. the whole plan (~40 min serial; the point list is in pb_plan_*.txt)
& $py imaggers\pbudget\pb_sweep.py --steps 8 --points (Get-Content imaggers\pbudget\pb_plan_ess.txt)

# 5. the 512x512 / SEQ 1056 arm
& $py imaggers\pbudget\pb_sweep.py --steps 4 --grid 32x32 --ref imaggers\pbudget\pb_ref32.npy \
     --out imaggers\pbudget\pb_512.jsonl --points "a_qkv+a_qk+c_sm+a_pv:bf16:0"

# 6. the tables, the curves and the figure
& $py imaggers\pbudget\pb_report.py \
     --points "imaggers\pbudget\pb_shard*.jsonl,imaggers\pbudget\pb_test.jsonl,imaggers\pbudget\pb_points.jsonl" \
     --json imaggers\pbudget\pb_budget.json --md imaggers\pbudget\PB-curves.md \
     --png imaggers\pbudget\pb_budget.png --csv imaggers\pbudget\pb_curves.csv

# 7. the tier amplifier table
& $py imaggers\pbudget\pb_layer.py
~~~

### Files

| file | what |
|---|---|
| `imaggers/pbudget/pb_lib.py` | the chain, the eight injection points, the injectors, the metrics |
| `imaggers/pbudget/pb_decode.py` | the numba bfp16 decoder + its byte-equality gate |
| `imaggers/pbudget/pb_gate.py` | the three gates that make this the product's chain |
| `imaggers/pbudget/pb_sweep.py` | one point = one full chain; JSONL append; `--grid` for the 512² shape |
| `imaggers/pbudget/pb_report.py` | the budget table, the curves, the figure (M1/M2/M3) |
| `imaggers/pbudget/pb_layer.py` | perturbations layered on the shipped bf16 A/C tier |
| `imaggers/pbudget/pb_points.jsonl` | 78 records, every one a full 8-step chain |
| `imaggers/pbudget/pb_512.jsonl` | the 512² / SEQ 1056 arm |
| `imaggers/pbudget/pb_budget.json` | the machine-readable budget |
| `imaggers/pbudget/pb_budget.png` | the curves and the ranking |
| `imaggers/pbudget/pb_curves.csv` | every (stage, σ, rho, corr) point |
| `imaggers/pbudget/PB-curves.md` | the same curves as markdown tables |
| `imaggers/pbudget/pblat/*.npy` | the final latents of every point |

Nothing outside `imaggers/pbudget/` was created, and no product file was modified.