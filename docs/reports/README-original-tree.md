# Z-Image BFP16 on AMD XDNA2 NPU — open IRON + XRT route

A self-contained, reproducible collection of everything needed to **feed
BlockFP16 (`bfp16ebs8`) weights straight to the AMD XDNA2 NPU** without the
VitisAI EP, without an fp32 expansion, and without per-call conversion.

Target: AMD Ryzen AI 9 H 365 (Radeon 880M), NPU **Strix** (XDNA2 / AIE2P).

---

## Why this route

The goal is to keep weights as **BlockFP16 all the way to the hardware**:

| requirement | why |
|---|---|
| BFP16 **storage** | 1.125 B/param instead of 4 B/param fp32 |
| BFP16 **residency** | the same 1.125 B/param in RAM |
| **no runtime overhead** | no per-call unpack, no fp32 conversion, no reload |

Anything that materialises fp32 loses the whole advantage over an iGPU, so the
conversion must never happen on the host.

The VitisAI EP path cannot do this: it converts on load, its `wgt_formatting`
tool rejects BFP16, and its compiled `.rai` artifacts carry an integrity
signature that is not recomputable.  The **open** path does not have any of
those problems.

---

## What is in here

| file | what it is |
|---|---|
| `pack_bfp16.py` | the `bfp16ebs8` encoder.  Verified byte-for-byte against a vendor `.rai` on 12 synthetic shapes **and** a real 16-tensor transformer shard (416,071,680 B region, 0 differing bytes). |
| `bfp16_gemm_iron.py` | the first open IRON BFP16 matmul: `mm_bfp` (aie2p) + ObjectFIFOs + Worker + `pipeline`. |
| `run_with_xrt.py` | the run side: loads `xclbin`/`insts.bin` through `pyxrt` and feeds BFP16 buffers straight to the NPU. |
| `verify_packer.py` | checks `pack_bfp16` against AMD's own reference implementation (`aie.utils.bfp`). |
| `ENVIRONMENT.md` | the exact paths and versions this was built and run against. |
| `REPRODUCE.md` | step-by-step reproduction. |
| `KNOWN-ISSUES.md` | what is unfinished and what has already been ruled out. |

---

## Format (settled — do not re-derive)

Block float, 8 elements sharing one exponent, **9 bytes per group of 8**:

```
group of 8 consecutive elements along K
    -> [exponent byte][8 x int8 mantissa]

e   = floor(log2(amax)) - 6        UNCONDITIONALLY
      (there is NO special case for an exact power of two; that variant is wrong)
m   = sign(v) * floor(|v| / 2^e + 0.5)     round half AWAY from zero
if m.max() > 127: e += 1; recompute         -128 is legal, +128 is not
exponent byte = (e + 133) & 0xFF             IEEE bias 127
```

Tiling for the AMD kernel layout: tiles of `tile_rows` rows x 32 K-elements;
inside a tile

```
slot = (r//8)*32 + (c//8)*8 + (r%8)     r = row % tile_rows,  c = kelem % 32
```

`tile_rows` is a function of (N, K, M) where M is the token count.  At seq=96
the measured values are 48 / 32 / 32 / 64 for the four matrix shapes of the
Z-Image transformer.

**Independent confirmation of `block_size = 8`** from four AMD sources:
`quark/torch/quantization/config/config.py` (`group_size=8`),
`quark/torch/kernel/hw_emulation/hw_emulation_interface.py` (`block_size = 8`),
`aie.utils.bfp` (`BLOCK=8, BLOCK_BYTES=9`), and Quark's
`BFPAttributes: {block_size: 8}`.  The often-quoted "32 elements share an
exponent" is wrong for `bfp16ebs8` — it confuses the `q4nx` group of 32.

**AMD's own reference codec** (from
`quark/onnx/operators/custom_ops/src/bfp/cpu/bfp_kernel.cc`) agrees with the
encoder above:

```c
shared_exp_value = max_biased_exponent - 127;
m_bits           = bit_width - 9;                        // 16 -> 7
scale            = 2 ** (shared_exp_value - (m_bits-1)); // == 2**(e_unb - 6)
max_v            = 2 ** (shared_exp_value + 1) - scale;
// rounding: STD_ROUND = std::round  (half away from zero)
```

---

### Weight residency — the goal, measured

`weights.bfp16` is 6,913,965,970 B holding 6,143,056,384 parameters in 506
tensors (270 packed `bfp16ebs8` + 236 raw fp32 1-D), across 17 shards of
~407 MB each.

    resident (bfp16ebs8 + raw fp32)   6,913,878,016 B
    fp32 equivalent                  24,572,225,536 B
    compression                                    3.554x
    saved                            17,658,347,520 B  (71.9%)

And the property the whole exercise exists for, instrumented rather than
asserted: over N=3 steady-state iterations of the block's access pattern,
`unpack` is called **0 times**.  Decoding per call instead would mean 8 unpack
calls and 203,489,280 B of decode work per three iterations.  See
`imaggers/residency/residency.py` and `residency_report.md`.

### Assembly: stage 1 is VERIFIED on the NPU

The task-compliant three-kernel chain

    rms_norm_weighted (gamma != 1) -> adaln_modulate (per-token scale/shift)
      -> mm_bfp(mixed=True)         [bf16 activations, BFP16 weights]

runs at M=32 K=3840 N=32, tile (16,8,16), on the memtile2 architecture:

    4/4 runs ERT_CMD_STATE_COMPLETED (5.8-6.5 ms), run-to-run byte-identical
    max abs err 128.0 = exactly 1 bf16 ulp;  max ulp 1.000 at both seeds
    99.90% bit-exact (1023/1024);  <=2 bf16 ulps criterion PASS

Artifact `imaggers/block/out/s1w_M32_K3840_N32_m16_k8_n16_d111.xclbin`
(21,304 B).  Gamma is folded into one packed `[scale|shift|gamma]` stream
element because `core_dma_channels_in == 2` bars a third input fifo; L1 at the
norm core measures 65,280 of 65,536 B.

Two traps that were costing whole rounds, now settled:
  * **use `set_current_device(NPU2())`**, not `from_name("npu2")` — the latter
    exposes only one shim tile to the placer and refuses any multi-fill design,
    with an error message that reads like a hardware limit but is not
  * `pack_bfp16`'s API is `pack(W, tile_rows=None) -> bytes`; there is no
    `quantize(...)` and no `unpack_and_quantize(...)`

Next: stage 2, the attention branch (`to_q/to_k/to_v -> unflatten ->
norm_q/norm_k -> RoPE -> QK^T -> softmax_rows_bf16 -> P*V -> to_out`), reusing
`imaggers/attn_impl/` which already verifies at the full Z-Image geometry.

### Assembly has begun: three kernels in one design

`imaggers/block/step1_design.py` chains `rms_norm` -> `adaln_modulate` ->
`mm_bfp(mixed=True)` on the memtile2 architecture and compiles:

    step1.xclbin   17,480 B
    step1.bin         900 B

The invocation that works (the earlier failure was the call form, not the
design):

    spec = S.memtile_worker.specialize(M=..., K=..., ..., use_cache=False)
    xcl, ins = spec.compile(xclbin_path=..., inst_path=...)

Next: run it on the NPU and verify against `imaggers/kernels/refs.py` within
2 bf16 ulps.

### The 34-block stack: enumerated and measured

`imaggers/pipeline/stack34.py` walks the whole transformer in forward order —
`noise_refiner.0/1`, `context_refiner.0/1`, `layers.0..29` — and measures the
residency property at full scale:

    506 tensors   6,913,878,016 B resident   6,143,056,384 params
    one pass: 506 reads, **0 unpack calls**

So the property the project exists for is confirmed on the real stack: the
weights stay as `bfp16ebs8` blocks and are never re-decoded.

## The pipeline is BATCHED across three phases, not one image at a time

The weights are not a constraint: DiT 7.1B + text 4B + VAE 0.5B ≈ 11.6B
params, so BFP16 (1.125 B/param) is about 13 GB for the WHOLE pipeline.  Even
if memory is tight the phases can be swapped in and out — and load time is
short enough that it does not matter.

The throughput win comes from batching at the phase boundary:

    text encoder  : batch over PROMPTS    -> pe (L, 2560) each
    DiT, 34 x 8   : batch over (prompt, seed) -> latents
    VAE           : batch over LATENTS    -> images

Each phase reads its weights ONCE and amortises them over every sample.
Because the DiT's per-dispatch traffic is 97.29% weight bytes, batching N
samples leaves the weight traffic unchanged and multiplies the compute by N —
so the arithmetic intensity rises again by N.

Arithmetic intensity is the token count, and it grows with the prompt:

    prompt length    tokens    FLOP/byte (Qwen3-4B forward)
    short               13          13.6
    medium             256         268
    long             1,024       1,074
    very long        4,096       4,294

The text encoder is a SINGLE FORWARD PASS — no autoregressive decoding, no KV
cache management, all tokens parallel.  That is the best possible load shape
for a systolic array, and at a thousand tokens it is as compute-bound as the
denoiser at 512x512.  **The text encoder belongs on the NPU too**; the earlier
"keep it on the CPU" advice was based on two arithmetic errors and is
withdrawn.

## The scope rule: compute per byte decides what goes on the NPU

Estimates per 128x128 image, 8 denoising steps:

| component | bytes | FLOP per image | FLOP per byte | where it runs |
|---|---|---|---|---|
| 34 DiT blocks | 6.14 GB | ~9.25 TFLOP | **~1,500** | **NPU** |
| VAE decoder | 0.34 GB | ~10 GFLOP | ~29 | CPU |
| text encoder | 7.67 GB | ~1 GFLOP | **~0.13** | CPU |

The denoiser is ~99.9% of the compute and is the only part worth putting on the
NPU — and it is exactly the part the BlockFP16 residency requirement is about.
The text encoder is the largest tensor set in the checkpoint but the
lightest per byte: moving it to the NPU would cost 6.7 GB of BFP16 residency
for no throughput gain.

So the split is deliberate, not a shortcut: **BFP16 weights and the NPU are
for the denoiser; the text encoder stays on the CPU.**

**Update — the VAE goes on the NPU too, in BFP16.**  The owner's call: faster
to keep the whole image path on the NPU than to round-trip through the iGPU,
and BFP16 (`bfp16ebs8`, 1.125 B/param) — explicitly *not* bf16 — keeps one
residency story.  The VAE is ~0.38 GB in BFP16 and ~10 GFLOP per image, so it
adds little residency and no meaningful load.  It reuses the same
`mm_bfp(mixed=True)` pipeline as the DiT.

The text encoder runs on the **GPU via the existing llama.cpp** — the owner's
call: no new encoder implementation, no rebuilding llama.cpp.  The machine
already has a Vulkan build at
`<retired-llama-build>\` with `llama-embedding.exe`,
plus `llama.cpp-0.5.0`, `prism-llama.cpp` and an `.ollama` tree.

So the final placement is:

| component | where | format |
|---|---|---|
| 34 DiT blocks | NPU | **BFP16** (`bfp16ebs8`) |
| VAE decoder | NPU | **BFP16** (`bfp16ebs8`) |
| text encoder | GPU | llama.cpp (Vulkan), reused as-is |

## Where this stands

**The hard part is done**: BFP16 weights are packed, the open toolchain compiles
them for aie2p, and the NPU executes a BFP16 kernel.  The remaining work is
assembly, not research.

### Kernel coverage for a DiT block — `COVERAGE.md`

7 of the 8 operators a Z-Image block needs are already in mlir-aie's IRON
library:

| operator | status |
|---|---|
| BFP16 GEMM (`mm_bfp`, incl. `mixed=True` = bf16 activations + BFP16 weights) | found |
| RMSNorm / LayerNorm | found (unweighted) |
| Softmax | found |
| RoPE (incl. HF split-half form) | found |
| SwiGLU / SiLU / GELU | found |
| elementwise add / mul / mul_add | found |
| attention toolkit (`mha`) | found |
| **adaLN modulation** | **missing** — composable from mul + add |

The four gaps are all *composition* gaps, not missing kernels:

  1. adaLN — `mul` + `add` with a per-token conditioning vector
  2. RMSNorm with learnable gamma — `rms_norm` then a separate `mul`
  3. **bidirectional attention softmax — SOLVED, no new C++ needed**
     (`attn/DESIGN.md`).  The causal mask is a single call site,
     `softmax_aie2p.h:237`; removing it yields `softmax_rows_bf16`
     (`softmax_aie2p.h:222-229`), which already exists UNMASKED in the same
     object and only lacks a Python binding.  Contract:
     `softmax_rows_bf16(bfloat16 *in, bfloat16 *out, rows=96, row_len=96)`,
     (96,96) bf16 row-major, 18,432 B each, `row_len % 32` hard.
     `mha.cc`'s online-softmax cannot simply drop its mask (causality is a
     derived diagonal plus three hard `kv>q` skips), so the recommendation is
     QK^T -> `softmax_rows_bf16` -> P*V with `1/sqrt(128)` folded into Q.
  4. flash attention at head_dim 128 — `prefill_fv` covers 256/512 only,
     though the `mha` micro-tile toolkit does accept 128

`_examples/block_datatypes/matrix_multiplication/whole_array_shuffle/whole_array.py`
is the most composed design available and the right skeleton to copy;
`single_core_mixed/` and `whole_array_mixed/` show `mm_bfp(mixed=True)`, which
is exactly the BlockFP16-weights configuration.

### What is left, honestly

| milestone | state |
|---|---|
| BFP16 encoder byte-exact vs the vendor format | **done** |
| single `weights.bfp16` file (6.439 GiB) | **done** |
| open toolchain compiles BFP16 for aie2p | **done** |
| **BFP16 matmul VERIFIED on the NPU** | **done** — rel 1.01e-02, the quantisation floor |
| **two-op pipeline (RMSNorm -> GEMM) on the NPU** | **done** — 100.00% bit-exact |
| **corner turn expressible by DMA** | **done** — measured, hidden to 1792 |
| attention softmax + adaLN + weighted RMSNorm designs | **done** (`attn/`, `adaln/`) |
| hidden 3840 (the real Z-Image dim) | path documented in `dit2/FINDINGS.md`, not built |
| full DiT block in IRON | next |
| 34-block transformer, VAE, end-to-end image | not started |

### The architecture is now proven end to end at small scale

`dit2/two_worker.py` runs RMSNorm on one core and `mm_bfp(mixed=True)` on
another, with the corner turn between them expressed **as a DMA `to_stream`
permutation** — no custom kernel.  Verified **100.00% bit-exact** against a
numpy RMSNorm-then-matmul model at hidden 32 … 1792, `ERT_CMD_STATE_COMPLETED`
and byte-identical on reruns.

The trick: at per-GEMM `k=8`, an `(m,8)` A tile's `(8,8,8)` micro-tile layout
**is** row-major, so the corner turn collapses to permuting
`(m, K/8, 8) -> (K/8, m, 8)` — one three-dimensional `to_stream` walk, proven
offline with `tap.gather()` before compiling.

Hard limits found on the way (all with verbatim error text in
`dit2/FINDINGS.md`):
  * compute-tile DMA descriptors take **at most 3 dimensions**
  * bfp16ebs8 strides must be **4-byte multiples** (9-byte blocks need packing)
  * **L1 is 64 KB per tile** — K=1792 uses 64,768 of 65,536 B
  * a single DMA descriptor caps at **16383 words**, which is what refuses
    hidden 2048 and 3840

> **2026-10-20 note**: every hardware/platform fact used by this repo (L1 64 KB,
> 16 KB program region, 16 BDs, ≤7 data arguments, state-8, the ~7 s ERT
> watchdog, BFP16 vs BF16 …) is consolidated — with provenance (report:line /
> commit) and design impact — in `docs/KNOWN-PLATFORM-FACTS.md`.

To reach hidden 3840: keep the `(m,K)` tile in a **memtile** (512 KB), turn the
corner on the memtile hop, and use asymmetric transfer granularity so the GEMM
core receives ready `(m,k)` slices — then neither core holds `(m,K)`.

**Reference-model trap worth knowing:** `mm_bfp_mixed.cc` re-rounds C to bf16
after *every* GEMM call.  A numpy reference must round-trip through bf16 per
accumulation; a plain f64 sum drifts up to 13 ulps at K=1792 and looks like a
bug when it is not.

    XRT 2.21.0            ->  xrt-smi examine  ->  NPU Strix
    pyxrt (Python 3.10)   ->  pyxrt.device(0)

The compile side and the run side use different Python versions; that is the
natural layering, not a workaround:

    compile : Python 3.12 + mlir_aie   ->  aiecc  ->  xclbin + insts.bin
    run     : Python 3.10 + pyxrt      ->  device(0) / bo / kernel / run

See `ENVIRONMENT.md` for the exact interpreter and DLL paths.
