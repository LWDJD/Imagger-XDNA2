# The iGPU, honestly

**The product does not use the GPU.** Everything below is what was measured, so the
next person does not repeat the search.

## 1. Why torch cannot help

Both interpreters ship a **CPU-only** build:

```
torch 2.3.1+cpu / 2.4.1+cpu    cuda available False    hip None    device_count 0
```

So every tensor operation the product performs -- attention's two matmuls, every
LayerNorm, RoPE, SiLU, the refiner, the VAE -- has no GPU backend to go to. That is
the whole reason the CPU is busy.

## 2. The three GPU routes

| route | state | measured |
|---|---|---|
| `torch-directml` | **not installed, not installable here** | when present: attention **42.31 ms/layer-step (405 GFLOP/s, rel_rms 6.3e-07)** vs CPU 59.82 ms (286 GFLOP/s) |
| **ggml + Vulkan** | **works today** | 512^2 VAE decode **2.264 s vs 9.104 s CPU = 4.02x**, `unsupported ops: 0/321` |
| hand-written Vulkan (`_gpuwork`) | **broken** | fp32 GEMM writes only columns 0..71 of 128 |

`torch-directml` is absent from all 179/145 distributions on both interpreters, there
is no cached wheel, and the package index is unreachable from the reference box. **No
step in this repository may depend on it.**

## 3. The working path: ggml + Vulkan

```
ggml_vulkan: 0 = AMD Radeon(TM) 880M Graphics (AMD proprietary driver) | uma: 1
  fp16: 1 | bf16: 1 | fp4: 0 | warp size: 64 | shared memory: 32768
  int dot: 1 | matrix cores: KHR_coopmat
```

Driver: `sdvae-bench/vbench.exe`, built from `vae_ggml.cpp`.

```powershell
# The backend name is CASE-SENSITIVE ('Vulkan', not 'vulkan') because make_backend
# matches with strstr, and --lat takes the project's own .bin, not a .npy.
.\vbench.exe --mode decode --backend Vulkan --lat lat_512.bin --reps 2
```

**`KHR_coopmat` is why this backend wins** where the hand-written shader lost: it can
use cooperative-matrix (tensor-core) paths, which a 64-accumulator scalar shader
cannot.

## 4. The broken path, and why its numbers are void

`_gpuwork/v3/gemm_f32.comp` declares local_size 16x16, BM=BN=128, and 64 f32
accumulators per invocation. Measured at 128^3, one dispatch, with bind() called
before every run():

* columns **0..71 are exact**
* columns **72..127 are never written**
* 56 = (16 - 9) x 8: `tx = 0..8` reach the epilogue, `tx = 9..15` never do

The identical behaviour occurs at every shape tried, so it is not a tail effect.

**Consequence:** the record's 'hand-written Vulkan SGEMM = 1.05 TF/s, slower than the
CPU' was measured on this kernel. It computed 56 % of its output, so it is **not**
evidence that Vulkan is slower, and **not** evidence that it is faster. An attention
built on it returned 27.04 ms/layer-step at rel_rms = 1.0 -- a fast wrong answer.
Those numbers are discarded, not reported.

## 5. What it would take

1. **Preferred:** build attention from ggml ops (`ggml_mul_mat` twice plus a softmax)
   and dispatch it to `Vulkan0`. The VAE result says the backend is ~4x the CPU on a
   real workload, and `vae_ggml.cpp:154/158` already contains the attention-shaped
   calls.
2. **Or** obtain a working `torch-directml` for the target torch version.
3. **Not** the hand-written path until section 4 is fixed -- and if it is fixed,
   re-verify EVERY output element at several shapes before timing anything.

## 6. What is on the CPU today, for scale

At 512^2 per image: attention ~18 s, noise-refiner pre-stage ~16.6 s,
norms/rope/packing/glue ~20 s -- about **58 s of a ~90 s loop**. The GPU prize is
therefore large, and it is the only remaining large item.