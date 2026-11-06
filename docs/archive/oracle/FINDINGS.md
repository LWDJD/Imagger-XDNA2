# FINDINGS — the Z-Image DiT block oracle: float truth vs device model

Working directory: `zimage-bfp16-npu\_scratch\oracle\`.  Files:
`dit_block.py` (the two variants + device-model primitives),
`test_dit_block.py` (self-test, small AND full geometry),
`_test_full_log.txt` (full-geometry run transcript).

The deliverable is one function per variant of a complete Z-Image DiT
transformer block forward pass:

* `dit_block_float`  — float64 throughout, no rounding.  THE TRUTH.
* `dit_block_device` — the precision/dtype model of the AIE assembly
  (`mm_bfp_mixed` GEMMs, bf16 tensors, bfp16ebs8 operands, the softmax /
  RoPE / SwiGLU kernel numerics).  **`float - device` is the gap the NPU
  work is allowed to lose.**

Both are parameterised by `(hidden, heads, head_dim, ffn_inner, seq)` via
`Geometry` (defaults 3840/30/128/10240/96); the device model's knobs are
`DeviceCfg`, every default being the CONSERVATIVE choice (the one that makes
the device model look worse).

## 1. RESULTS

Seed 0, gaussian inputs/weights (1/sqrt(fan_in) scaled fp32 "checkpoint"),
conservative `DeviceCfg` defaults (B codec `floor`, `gemm_k_chunk=8`,
interleaved RoPE, two-pass norm/adaLN).  "rel_to_absmax" is
`max|Δ| / max|float|` — the repo's "rel" convention (cf. KNOWN-ISSUES.md
"rel 1.011607e-02").  "max_rel" is `max |Δ|/|float|` over elements with
`|float| >= 1% of absmax`; it is dominated by outputs that pass near zero
(cancellation), where even a bf16 rounding flips the value across 0.

| geometry | max abs err | max rel err (gated) | rel_to_absmax | rms | float wall | device wall |
|---|---|---|---|---|---|---|
| small (64/2/32/128/32) | 9.353e-02 | 1.519e+00 | **2.243e-02** | 2.393e-02 | 0.001 s | 0.005 s |
| **FULL Z-Image (3840/30/128/10240/96)** | **3.202e-01** | **2.392e+00** | **5.363e-02** | 4.374e-02 | **0.54 s** | **17.6 s** |

Full-geometry reference output absmax 5.972, so max abs err 0.320 is 5.4% of
the output range.  Per-stage divergence at full geometry (each stage's own
absmax):

| stage | max abs | rel_to_absmax |
|---|---|---|
| x_in (bf16 entry rounding only) | 1.55e-02 | 3.24e-03 |
| adaln1 (norm + modulation) | 6.32e-02 | 1.12e-02 |
| q_rope (QKV GEMM + RoPE) | 2.82e-01 | 5.48e-02 |
| k_rope | 3.33e-01 | 6.50e-02 |
| attn_o (QK^T, scale, softmax, P@V) | 1.39e-01 | 7.95e-02 |
| x_mid (out proj + residual) | 6.96e-02 | 1.43e-02 |
| adaln2 | 9.08e-02 | 1.59e-02 |
| swiglu | 9.63e-01 | 9.90e-02 |
| x_out (FFN down + residual) | **3.20e-01** | **5.36e-02** |

Sensitivity at full geometry (which conservative choice costs what):

| variant | rel_to_absmax | wall |
|---|---|---|
| **conservative default** (B codec floor, bf16 C re-round every 8 of K) | **5.363e-02** | 17.6 s |
| B codec `conv_even` instead of `floor` | 5.179e-02 | 17.6 s |
| one GEMM call per matmul (single bf16 C store per GEMM) | 1.980e-02 | ~5 s |

So the **dominant term is the per-call bf16 re-rounding of C**, exactly the
effect measured on the device in `_scratch/dit2/FINDINGS.md:80-95` (a K/k-call
accumulation carries K/k intermediate bf16 roundings; 13 ulps at K=1792).
The B-codec rounding choice is worth only ~4% of the gap; the call
granularity is worth 2.7x.  The conservative default is (by test assertion)
the worst of the three.

**Wall time at full geometry** (this machine, `ryzen-ai-1.8.0` python):
float oracle **0.54 s**, device model **17.6 s** (per-chunk Python/BLAS loop
with a bf16 store per chunk — CPU-side model cost, NOT a prediction of NPU
time), full self-test 41.3 s (device model runs 3x with sensitivities).
A single block is ~34 GFLOP; 34 blocks per step is ~1.2 TFLOP of reference
math, i.e. minutes of CPU time for a full multi-block oracle comparison.

## 2. THE DEVICE MODEL, STAGE BY STAGE

| stage | device model | evidence | determinable? |
|---|---|---|---|
| tensor entry | x, s, b rounded to bf16 | residual stream / GEMM outputs are bf16 (`mm_bfp_mixed.cc` C type) | yes |
| rms_norm_weighted | sum-of-squares sequential f32, f32 inv_rms, TWO bf16 stores (norm, then gamma mul) | `rms_norm_aie2p.h:123-142`; but fused kernel **MISSING** (`refs.py` contract "MISSING in mlir-aie") -> conservative two-pass (`refs.py:164-177`) | kernel missing -> worst plausible build |
| adaln_modulate | g=bf16(1+s); tmp=bf16(x*g); y=bf16(tmp+shift) — 3 bf16 stores | fused kernel **MISSING**; `refs.py:209-232`, `adaln/DESIGN.md:165` (proposed fused = 1 store) | kernel missing -> worst plausible build |
| QKV/FFN/oproj GEMM | operands bfp16ebs8 in groups of 8 along K (A conv_even on-core, B floor host codec); per call `C = bf16(C + A_q@B_q)` in f32 MAC; call = 8 of K | `mm_bfp_mixed.cc:49-52,80-89,130-135`; dit2 bit-exact vs this model at K=32..1792 | call granularity at hidden 3840 NOT determinable -> 8 |
| RoPE | interleaved pairs, cos/sin bf16 LUT, products in accumulator, ONE bf16 store per half | `rope.cc:29-39`; convention `rope_real.py:39-61` | yes (except convention conflict, below) |
| QK^T | same GEMM model, K=head_dim 128 | as above | as above |
| 1/sqrt(head_dim) | extra bf16 store on S (`S = bf16(S * bf16(1/sqrt(d)))`) | kernel has no scale param (`attn/DESIGN.md:44-45,205`); placement open (`attn/DESIGN.md:71-73`) | NOT determinable -> extra store (worse than folding into Q) |
| softmax | `exp2((x-m)*log2e)`, log2e=1.4453125, f32, sequential row sum, one bf16 store | `softmax_aie2p.h:15-16,57,88-98,222-229` | arith yes; **exp2 polynomial NO** (envelope below) |
| P@V | same GEMM model, reduction = 96 keys | as above | as above |
| SwiGLU | sig narrowed to bf16; `silu_out=bf16_ftz(max(gate,-8)*sig)`; `out=bf16_ftz(up*silu_out)` (0 where silu_out=0) | `activation.py:907-941` (`_swiglu_model`), `swiglu.cc` narrows after every multiply | structure yes; **sigmoid build NO** (envelope below) |
| residual adds | one bf16 store (`add_sized`) | `refs.py` ADALN_CONTRACT accumulators note | yes |

## 3. NOT DETERMINABLE — and the conservative choice made

1. **GEMM call granularity at hidden 3840.**  dit2's measured model had one
   bf16 C store per 8-of-K call and was bit-exact against the device up to
   K=1792 (`dit2/FINDINGS.md:84-95`).  The 3840 design (`dit2/FINDINGS.md:222-233`
   recommends a memtile hop) is not built yet, so the number of roundings per
   matmul is unknown.  Conservative: `gemm_k_chunk=8` (most roundings).
   COST: 2.7x of the gap (5.36e-2 vs 1.98e-2).  If the assembly calls
   `mm_bfp` with a larger K, its true gap is SMALLER — do not let a passing
   NPU be argued up to the 5.4e-2 number without measuring its call pattern.
2. **Host-side B codec rounding for bfp16ebs8 weights/operands.**
   `aie.utils.bfp.encode` offers only `floor` (DEFAULT, truncating toward
   -inf, biased) or `conv_even` (`bfp.py:76-88`); `pack_bfp16._quantise`
   (VitisAI-verified) rounds half-away-from-zero.  Which one the assembly
   feeds `mm_bfp` is unknown.  Conservative: `floor`.  COST: ~4% of the gap
   (5.36e-2 vs 5.18e-2).  The model's quantizer matches
   `pack_bfp16._quantise` bit-exactly in `half_away` mode (test-asserted) and
   `bfp.quantize`'s documented semantics in the other two.
3. **Fused vs composed rms_norm_weighted / adaln_modulate.**  Both fused
   kernels are MISSING from mlir-aie (`refs.py` contracts; `adaln/DESIGN.md`
   proposes them).  Conservative: the two-pass/three-pass compositions with
   their extra bf16 stores (up to ~3.5 half-ulps more than fused,
   `refs.py:411-421`).  If the assembly lands the fused kernels, its true gap
   shrinks slightly.
4. **Attention scale placement** — fold 1/sqrt(128) into Q, or pre-scale the
   S tile (`attn/DESIGN.md:71-73`): not determinable.  Conservative: pre-scale
   S with a separate bf16 store (the fold into Q would round once, on Q).
5. **Softmax exp2 error is NOT modelled.**  The kernel's exp2/log2e arithmetic
   is modelled exactly, but the AIE2P exp instruction's own approximation is
   only bounded by measurement: **2.94e-2 relative through the softmax
   normalisation** (`activation.py:400-407`).  Its shape is unknown, so the
   device model does not fabricate it.  **Judging must add this envelope** to
   P (and thus to attn_o): a correct NPU may legitimately differ from
   `dit_block_device` by up to ~3% relative on softmax outputs.
6. **SwiGLU sigmoid build is NOT determinable.**  `swiglu(use_lut=True)` is
   an exact table model (`activation.py:923-931`, judged at 1 ulp) but the
   default build uses AIE2P `vtanh`, "a coarser approximation with no
   published spec" (`activation.py:956-960`), measured at **0.08 rel /
   0.35 abs** for swiglu (`activation.py:144-148`).  The device model uses an
   exact sigmoid plus the determinable bf16 narrowing/clamp/ftz structure;
   **judging must add the vtanh envelope** (or the assembly should build
   `use_lut=True`, which removes it).  Cost is visible in the per-stage table:
   the swiglu stage already carries the largest stage-local rel (9.9e-2)
   from structure alone.
7. **RoPE convention conflict (this one is a bug in the plan, not a
   tolerance).**  ASSEMBLY.md:18 selects `rope(..., two_halves)`, i.e.
   HuggingFace rotate-half (`datamovement.py:322-323`, `rope.cc`
   `rope_two_halves`).  Z-Image is the INTERLEAVED convention — adjacent
   pairs — proven against the installed diffusers source in
   `zimage-bfp16-repo/rope_real.py:39-61`, with a negative control where
   rotate-half is wrong by 6.288 vs 4.8e-7.  The oracle computes interleaved
   (`DeviceCfg.rope_convention='two_halves'` is available and demonstrably
   different).  **The assembly must call `rope` WITHOUT `two_halves`.**
8. **rms eps** — Z-Image's own value is not determinable
   (`refs.py:143-144`, `adaln/DESIGN.md:274`); modelled 1e-5 everywhere.
   Same eps in both variants, so the gap is unaffected; the truth is not.
9. **RoPE position ids** — how the 96 tokens map to the three rope axes
   (axes_dims [32,48,48], `rope_real.py:58`) is not determinable; the oracle
   uses `ids[t] = (t, t, t)`.  Both variants share the table, so the gap is
   unaffected.
10. **s1,b1,s2,b2 production** — the conditioning projection GEMM is outside
    this block's scope (`ASSEMBLY.md:26-28`); the oracle takes s/b as inputs
    and the device model rounds them to bf16 on entry.
11. **Intra-call f32 accumulation order** (numpy's blocked sgemm vs the
    core's fixed MAC order) — below one f32 ulp per add, i.e. ~2^-23
    relative, utterly dominated by the bf16 store (2^-8).  The device was
    measured bit-exact against a numpy f32-accumulation model at
    K=32..1792 (`dit2/FINDINGS.md:89-95`), so this is safe.
12. **gamma dtype** (bf16 assumed per `refs.py:48`; f32 in the checkpoint is
    possible, `adaln/DESIGN.md:288-290`) — the device model rounds gamma to
    bf16, one extra rounding if the checkpoint is f32.

## 4. HOW TO JUDGE THE NPU AGAINST THIS ORACLE

    budget(stage) = |dit_block_device - dit_block_float|      (measured above)
                  + exp2 envelope on softmax outputs          (2.94e-2 rel, item 5)
                  + vtanh envelope on swiglu outputs          (0.08 rel / 0.35 abs, item 6)
                                                       [only if those builds are used]

At full geometry the measured term is **max abs 0.320, 5.4e-2 of output
absmax** for the conservative call pattern.  An NPU implementation that
matches `dit_block_device` bit-for-bit is doing as well as this model; the
model is deliberately NOT optimistic, but it is also not a substitute for
the two envelopes in items 5-6 — those two approximation errors are real
budget the NPU may spend and this file cannot predict their elementwise
shape.  Cheapest way to remove both envelopes: build softmax rows via
`softmax_rows_bf16` (its exp2 is what the 2.94e-2 measures — no way around
it) and build `swiglu(use_lut=True)` (exact table model, judged at 1 ulp).

Caveats on the numbers: synthetic gaussian data, one block only, one seed;
the divergence is data-distribution dependent (real Z-Image activations
reach 3.1e5, `rope_real.py:57-58`, which stresses exponents more than
N(0,1) does).  Re-run `test_dit_block.py` with other seeds before treating
the budget as final.

## 5. REPRODUCTION

    "<python312>\python.exe" ^
        zimage-bfp16-npu\_scratch\oracle\test_dit_block.py               (both)
    ... test_dit_block.py --geometry full --no-sensitivity               (fast full)
    ... test_dit_block.py --geometry small

No NPU, no compile, pure numpy; `refs.py` / `noncausal_softmax.py` /
`pack_bfp16.py` are imported read-only and cross-checked by assertions.

## 6. SUCCESS CRITERIA

* **MINIMUM (both variants run, self-test passes at a small geometry): MET.**
  25 primitive assertions + block checks pass at 64/2/32/128/32
  (`test_dit_block.py --geometry small`).
* **TARGET (both run at full Z-Image geometry; max abs and max rel
  reported): MET.**  Full geometry 3840/30/128/10240/96 runs both variants;
  max abs error **3.202e-01**, max rel error (gated at 1% of absmax)
  **2.392**, rel to output absmax **5.363e-02**.
* **Wall time at full geometry: REPORTED.**  float 0.54 s, device model
  17.6 s, per-block; the assembly can budget ~1 s of CPU per block reference
  (float) or ~18 s (full device model) and should note the device-model wall
  is model overhead, not device time.
