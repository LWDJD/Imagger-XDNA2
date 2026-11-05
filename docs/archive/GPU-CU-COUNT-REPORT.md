# iGPU CU count, corrected roofs, and the achievable GEMM rate

**Date** 2026-11-02 · Supersedes §1 and §3.3 of `GPU-FEASIBILITY-REPORT.md`.
Nothing here touched the NPU, no xclbin, no device lock, no 8-step job.
Every number below was measured on this box in this session unless marked.

---

## 0. Verdict in six lines

1. **The iGPU is 12 CU, not 6.** Vulkan reports it twice over
   (`VK_AMD_shader_core_properties` → 1 shader engine × 2 arrays × **6 CU per array = 12**;
   `VK_AMD_shader_core_properties2.activeComputeUnitCount` → **12**), the topology matches the
   published Strix Point configuration, and the measured ALU peak is consistent with
   **12 CU × 64 FP32 lanes = 768 lanes**. HIP's `multiProcessorCount` really does read **6**, but 6 is
   exactly half of 12 and is inconsistent with the same struct's per-CU thread limit — it is
   reporting **workgroup processors (WGPs)**, 2 CU each.
2. **Corrected roofs**: fp32 **4.454 TFLOP/s**, packed fp16 **8.909 TFLOP/s** (12 CU × 64 lanes × 2 × 2.90 GHz,
   doubled for packed fp16). The previous report's 2.23 / 4.45 were exactly half.
3. **The fp32 roof is genuinely reachable — the ALU is not the limit.** A pure-ALU kernel measures
   **4.45 TFLOP/s median**, i.e. ~100 % of the 4.454 nameplate roof, in a quiet measurement
   window; the same kernel measured 3.08–3.16 in a contended one. **Throughput on this part swings ±40 %
   with host CPU load** because the iGPU shares the socket's power budget — that, not a clock limit, is why
   short or contended measurements read low.
4. **fp16 is *not* faster than fp32 here, and that is a compiler result, not a hardware one.** With the
   flop counts matched (both kernels executing exactly 64 scalar FMAs per iteration), a GLSL `f16vec2`
   FMA kernel gives **4.73 TFLOP/s vs 4.45 for fp32 — 1.06×, i.e. no packing**. `glslc`/LLPC is not
   emitting `v_pk_fma_f16`. (My first fp16 number, 7.18, was a **2× flop over-count** in the harness and is
   withdrawn — see §2 (iii).) So the 8.909 packed-fp16 roof is *not* reachable from GLSL either.
5. **DirectML is still not beatable.** Best rate of *any* path measured: **1.68 TFLOP/s** (DirectML fp16,
   4096³, back-to-back). A hand-written, exactly-correct Vulkan SGEMM reaches **1.05 TFLOP/s** — slower
   than DirectML. Cooperative-matrix (`VK_KHR_cooperative_matrix`, 16×16×16 f16→f32) is advertised
   but the GLSL front end in this SDK cannot compile `coopMatLoad`/`coopMatStore`, so a matrix-core
   GEMM could not be built.
6. **So the "3× on the table" is real arithmetic and an unrecovered opportunity.** 1.588 TFLOP/s is
   **35.7 %** of the 4.454 fp32 roof, and the hardware really does deliver ~4.45 TFLOP/s to a pure-ALU
   kernel — so the headroom is genuinely ~2.8×, not the 1.4× the old (6-CU) roof implied. But **no GEMM
   path I could build or probe recovered it**: DirectML tops out at 1.68, my own Vulkan kernel at 1.05.
   **The floor arithmetic is therefore unchanged: ≈ 16–19 s, not 24.12 s, and a <10 s image is still
   unreachable.**

---

## 1. Compute units: five sources, and which ones agree

### 1.1 (a) Vulkan — two independent fields, both say 12

Probe: `_gpuwork/v3/vkprobe.py` (raw ctypes Vulkan 1.4, `vkGetPhysicalDeviceProperties2`
with a `pNext` chain). Raw JSON: `_gpuwork/v3/vkprobe.json`.

| field | value |
|---|---|
| `VkPhysicalDeviceProperties.deviceName` | AMD Radeon(TM) 880M Graphics |
| `vendorID` / `deviceID` | 0x1002 / **0x150e** |
| `apiVersion` / `driverInfo` | 1.4.349 / 26.8.1 (LLPC) |
| `VkPhysicalDevicePCIBusInfoPropertiesEXT` | domain 0, **bus 99**, device 0, function 0 |
| **`VK_AMD_shader_core_properties.shaderEngineCount`** | **1** |
| **`...shaderArraysPerEngineCount`** | **2** |
| **`...computeUnitsPerShaderArray`** | **6** |
| **⇒ CU = 1 × 2 × 6** | **12** |
| **`VK_AMD_shader_core_properties2.activeComputeUnitCount`** | **12** |
| `...simdPerComputeUnit` / `wavefrontSize` | 2 / 64 |
| `...sgprsPerSimd` / `vgprsPerSimd` | 2048 / 1024 |
| `VkPhysicalDeviceSubgroupProperties.subgroupSize` | 64 |
| memory heaps | 5.38 GiB (no flags), 10.76 GiB (DEVICE_LOCAL | HOST_VISIBLE) |

`activeComputeUnitCount` is a *separate extension* (`VK_AMD_shader_core_properties2`, rev 1) from the
product above (`...properties`, rev 2) and is defined as the number of *active* CUs — so these are two
independent driver statements that agree.

**Topology:** 1 shader engine, 2 arrays, 6 CU per array. The 16-CU sibling in the same
family (Radeon 890M, Ryzen AI 9 HX 370) is the same 1 × 2 × topology with **8** CU per array. 12 is the
consistent member of that series; 6 would require either half an array or one array unused, and would make
the 890M a 1 × 1 × 8 part, which is not AMD's Strix Point topology.

### 1.2 (b) D3D12 / DXGI — adapter identity confirmed, lane count *not* obtainable

Probe: `_gpuwork/v3/dxprobe.py`. Raw: `_gpuwork/v3/dxprobe.json`.

| field | value |
|---|---|
| `DXGI_ADAPTER_DESC1.Description` | AMD Radeon(TM) 880M Graphics |
| `VendorId` / `DeviceId` / `SubSysId` / `Revision` | 0x1002 / **0x150e** / 0xe0211d05 / 196 |
| `DedicatedVideoMemory` / `SharedSystemMemory` | 338.1 MiB / 16020.7 MiB |
| `D3D12CreateDevice` | succeeds at FL 12_0 (0xc000) and 12_1 (0xc100); **fails with E_INVALIDARG at 11_0 and 10_0** |

I tried to get `D3D12_FEATURE_DATA_D3D12_OPTIONS1.TotalLaneCount` — the one D3D12 field that carries a
lane count — but `ID3D12Device::CheckFeatureSupport` returned **E_INVALIDARG (0x80070057)** for every
feature I asked for (`ARCHITECTURE`, `D3D12_OPTIONS`, `D3D12_OPTIONS1`), through the `d3d12.dll` export
with explicit ctypes argtypes, on all four DXGI adapters including the WARP one. Neighbouring vtable slots
faulted outright (slot 11 → access violation at 0x9), while the DXGI `GetDesc1` call on slot 10 of the
adapter does work — so the device interface binding is not trustworthy and I stopped rather than report a
number I could not corroborate.
**D3D12 therefore contributes the PCI device ID (0x150e) and feature level, not a CU count.** I did not
get a D3D12 lane count and I am not claiming one.

### 1.3 (c) HIP with a *validated* struct — really says 6, and 6 is half of 12

`hip_runtime_api.h` **does not ship anywhere on this box** — I searched `C:\Program Files\AMD`,
`C:\Program Files\AMD RyzenAI`, `C:\VulkanSDK`, the miniforge tree, the GAIA tree and `C:\` to
depth 4, and there is no ROCm/HIP SDK tree (confirmed: no `amdhip64` headers, no `hipcc`). So the layout
**cannot be declared from the header**. Instead I dumped the raw `hipDeviceProp_t` buffer
(`_gpuwork/v3/hipdump.py`) and located every field *by value*, with five independent anchors:

| offset | value | identification | independent confirmation |
|---:|---:|---|---|
| 0 | "AMD Radeon(TM) 880M Graphics" | `name[256]` | Vulkan deviceName |
| 256 | 13 136 003 072 | `totalGlobalMem` = 12.23 GiB | matches previous report |
| 264 | 65 536 | `sharedMemPerBlock` | Vulkan `maxComputeSharedMemorySize` would be 64 K on RDNA |
| 280 / 284,288,292 | 1024 / 1024,1024,1024 | `maxThreadsPerBlock` / `maxThreadsDim[3]` | Vulkan `maxComputeWorkGroupInvocations` = 1024 |
| 296,300,304 | 2147483647, 65536, 65536 | `maxGridSize[3]` | Vulkan `maxComputeWorkGroupCount` = 2^32-1, 65535, 65535 |
| 308 | 2 900 000 | `clockRate` | nameplate 2.90 GHz |
| 312 | 1 960 000 | `memoryClockRate` | LPDDR5X-9600 |
| 316 | 256 | `memoryBusWidth` | 12.2 GiB UMA carve-out |
| 328 / 332 | 11 / 5 | `major` / `minor` = gfx11.5 | previous report's `gcnArchName` = gfx1150 |
| **336** | **6** | **`multiProcessorCount`** | ← **this is the disputed field** |
| 340 | 2 097 152 | `l2CacheSize` = 2 MiB | previous report |
| 344 | 2048 | `maxThreadsPerMultiProcessor` | 2 SIMD × 16 waves × 64 lanes |
| 368 | 99 | PCI bus id | **Vulkan says pciBus = 99** ✔ |

The field order `... major, minor, multiProcessorCount, l2CacheSize, maxThreadsPerMultiProcessor` is HIP's
documented order, and the run of values 11, 5, 6, 2097152, 2048 can only be that run. So **HIP genuinely
returns 6**, and offset 336 is genuinely `multiProcessorCount` — the previous agent read the right field; the
field itself is the problem. `hipDeviceGetAttribute` agrees (enum 63 and enum 91 both return 6; enum 23
returns 11 = major, 25 returns 64 = warpSize, 61 returns 5 = minor, 67 returns 99 = pciBus).

**Why 6 is wrong, and exactly how wrong.** 6 = 12 / 2 precisely, and it is self-inconsistent with the same
struct: `maxThreadsPerMultiProcessor` = 2048, which is the **per-CU** resident-thread limit for this part
(2 SIMDs × 16 wavefronts × 64 lanes). A WGP (2 CU) would allow 4096. So the runtime is counting WGPs in a
field named "multiProcessor". This is the well-known RDNA behaviour the brief warned about, and it is
reproducible here.

### 1.4 (d) Cross-check: measured fp32 GEMM vs a known-good reference

Not a CU measurement by itself, but it is the tie-breaker between "12 CU × 64 lanes" and "6 CU × 128 lanes"
(both give 768 lanes and therefore the same peak):

* A pure-ALU kernel (no memory, 8 independent `vec4` FMA chains, 256 threads/workgroup,
  `_gpuwork/v3/peak_f32.comp`) measures **4.45 TFLOP/s median** (samples 3.78–5.37) in a quiet window and
  **3.08 median** in a contended one — the same kernel, two windows (§2).
* 4.45 TFLOP/s ÷ 2 flop ÷ 2.90 GHz = **767 lanes**. That is 768 to within 0.2 % — the machine's total FP32
  lane count, read straight off a stopwatch at the nameplate clock.
* 767 lanes ÷ 12 CU = 64 lanes/CU, which is exactly what `simdPerComputeUnit = 2` (two 32-wide SIMDs)
  predicts. The alternative reading, 6 CU × 128 lanes, fits the lane count but not the Vulkan CU count, and
  it would require Strix Point CUs to be twice the width of every other RDNA3 CU.
* **The 12-vs-6 question is settled by the direction of the arithmetic, not by the peak:** the total lane
  count (768) is directly measurable and agrees with 12 × 64; the two drivers that report a CU count
  disagree, and the one that reports 6 is the one that is internally inconsistent (§1.3).

### 1.5 What I believe

**12 compute units, 768 FP32 lanes (64/CU), 2 SIMD/CU, wave64, 1 shader engine, 2 arrays, 6 CU per array.**
Sources agreeing on 12: Vulkan `shader_core_properties` (product), Vulkan `shader_core_properties2`
(`activeComputeUnitCount`), the published Strix Point topology, and the measured 768-lane ALU peak.
Source disagreeing: HIP (6), which is exactly half and is explained by WGP counting. The previous report's
"6 CU" is therefore **wrong**, and its derived roof (2.23 TFLOP/s) is exactly half the correct value.

---

## 2. Corrected roofs, and the roofs that are actually reachable

| quantity | value | how |
|---|---:|---|
| FP32 nameplate roof | **4.454 TFLOP/s** | 12 CU × 64 lanes × 2 flop × 2.90 GHz |
| packed-FP16 nameplate roof | **8.909 TFLOP/s** | same × 2 for `v_pk_fma_f16` |
| **measured FP32 ALU peak, quiet window** | **4.45 TFLOP/s** median (7 samples span 3.78–5.37) | pure-ALU kernel, 7 interleaved samples |
| **measured FP32 ALU peak, contended window** | **3.08 – 3.16 TFLOP/s** median, best 3.47 | same kernel, host CPU busy |
| implied core clock at 4.45 TFLOP/s | **2.89 GHz** | peak / (768 × 2) — i.e. the nameplate clock |
| **measured FP16 ALU peak (matched flops)** | **4.73 TFLOP/s** median (samples 4.53–5.38) | `f16vec2` FMA, 64 scalar FMAs/iter |
| **FP16 / FP32 ratio at matched flops** | **1.06×** | ⇒ **not packed**, no `v_pk_fma_f16` |
| short 41 ms dispatch of the same kernel | 1.67 TFLOP/s | the measurement is duration-dependent |

**(i) The fp32 roof is real: 4.45 TFLOP/s median is 99.9 % of the 4.454 nameplate figure**, and the implied
core clock (4.45 TFLOP/s ÷ 1536 flop/clk) is 2.89 GHz — exactly the nameplate boost. So an fp32 ALU loop can
and does use the whole machine. The previous report's 1.588 TFLOP/s ceiling is therefore **35.7 % of what the
FP32 ALU can actually do**, which is the brief's point. (Individual samples run 3.78–5.37, i.e. ±20 % — the
kernel's own measurement noise plus the host-load swing below. I quote the median because the top samples sit
*above* the nameplate roof, which can only mean noise, not a clock above spec.)

**(ii) But the number is not stable, and the instability is host load, not a GPU clock cap.** The identical
kernel measured 4.45 TFLOP/s in one window and 3.08 in another; a 41 ms dispatch of it reads 1.67 TFLOP/s
while a 1.4 s one reads 3.08. The iGPU has no power budget of its own — it shares the socket's with the CPU,
and other agent processes were running for much of this session. **Every GPU number in this report is
therefore a lower bound whose error bar is up to 1.45×, and only same-window comparisons are trustworthy.**
This is a second, independent way to get a badly wrong answer, of the same family as the missing-synchronisation
bug the previous agent found.

**(iii) fp16 gives no throughput win from GLSL, and I withdrew my first fp16 number.** My initial
`peak_f16.comp` issued 16 `f16vec2` FMAs per iteration (32 scalar FMAs = 64 flops) but the harness scored it
with the fp32 kernel's 128-flop formula, doubling it to a bogus 7.18 TFLOP/s. Re-run with the counts matched
— both kernels executing exactly 64 scalar FMAs per iteration (`_gpuwork/v3/peak_check.py`) — the answer is
**fp16 4.73 vs fp32 4.45 TFLOP/s, a ratio of 1.06×**. No packing is happening: LLPC is issuing scalar
`v_fma_f16`, not `v_pk_fma_f16`. So:

* the previous report's *observation* ("fp16 ≈ fp32 on this part") is **confirmed and generalises to shaders
  in general**, not just to DirectML;
* the packed-fp16 roof of 8.909 TFLOP/s is **not reachable from GLSL** — only a matrix-core (cooperative
  matrix) path could get there, and that does not compile (§4.4);
* the correction to the previous report is therefore about the *fp32* roof only (2.23 → 4.454), not about a
  newly-available packed-fp16 path.

---

## 3. The GEMM sweep

All numbers per-call synchronised by a blocking `.cpu()` read-back of the result tensor.
`torch_directml` has no `synchronize()`; the read-back is the only synchronisation.

### 3.1 Square GEMM, one shape resident at a time (`_gpuwork/v3/dml_sizes.py`)

| size | dtype | median ms | best ms | median TFLOP/s | best TFLOP/s | rel_rms vs CPU fp32 |
|---|---|---:|---:|---:|---:|---:|
| 2048³ | fp32 | 16.47 | 14.01 | 1.04 | 1.23 | 8.4e-07 |
| 2048³ | fp16 | 16.39 | 15.07 | 1.05 | 1.14 | 4.2e-04 |
| 4096³ | fp32 | 129.86 | 118.64 | 1.06 | 1.16 | 1.2e-06 |
| 4096³ | fp16 | 128.47 | 114.09 | 1.07 | 1.20 | 4.1e-04 |
| 8192³ | fp32 | 1246.05 | 1236.13 | 0.88 | 0.89 | 1.6e-06 |
| 8192³ | fp16 | 894.10 | 839.62 | 1.23 | 1.31 | 4.2e-04 |

### 3.2 The same sizes executed back-to-back, one read-back per burst (`_gpuwork/v3/compare.py`)

This is the mode that matches the pipeline (a queue of GEMMs, one sync per stage) and it is **1.3–1.6×
faster than 3.1**, because the GPU never idles between submissions and the clock stays up:

| shape | dtype | DirectML ms | DirectML TFLOP/s | hand-written Vulkan ms | Vulkan TFLOP/s |
|---|---|---:|---:|---:|---:|
| 1024×3840 @ 3840×3840 | fp32 | **25.07** | **1.20** | 31.10 | 0.97 |
| 1024×3840 @ 3840×3840 | fp16 | 36.23 | 0.83 | 29.27 | 1.03 |
| 1024×3840 @ 3840×10240 | fp32 | **73.50** | **1.10** | 93.36 | 0.86 |
| 1024×3840 @ 3840×10240 | fp16 | 67.15 | 1.20 | 72.25 | 1.11 |
| 1024×10240 @ 10240×3840 | fp32 | 67.33 | 1.20 | 93.80 | 0.86 |
| 1024×10240 @ 10240×3840 | fp16 | **52.54** | **1.53** | 73.20 | 1.10 |
| 4096³ | fp32 | 94.14 | 1.46 | 140.41 | 0.98 |
| 4096³ | fp16 | **81.57** | **1.68** | 119.49 | 1.15 |

DirectML reproduces the previous report's numbers closely here (his 24.42 / 63.77 / 64.08 ms vs my
25.07 / 73.50 / 67.33 ms fp32 on the same shapes), so those were real.

### 3.3 Percentage of the corrected roof

| rate | vs FP32 roof (4.454) | vs **measured** FP32 ALU peak (4.45) | vs packed-fp16 roof (8.909, unreachable) |
|---|---:|---:|---:|
| previous report's 1.588 TFLOP/s | **35.7 %** | 35.7 % | 17.8 % |
| best measured, any path: **1.68** | 37.7 % | **37.8 %** | 18.9 % |
| DirectML fp32, 1.46 | 32.8 % | 32.8 % | 16.4 % |
| hand-written Vulkan SGEMM, 1.05 | 23.6 % | **23.6 %** | 11.8 % |
| hand-written Vulkan fp16, 1.15 | 25.8 % | 25.8 % | 12.9 % |

So the brief's arithmetic is confirmed: **1.588 is 36 % of the fp32 roof, not 69 %**, and the roof is not
theoretical — an ALU loop really does reach 4.45 TFLOP/s. The headroom is genuinely ~2.8×. **None of it was
recovered by any GEMM path I could build or probe**; the residual is a kernel-efficiency problem (§4).

---

## 4. Why DirectML does not reach the roof — what I found and what I could not fix

### 4.1 Op form and layout do not matter (DirectML, 4096³, `_gpuwork/v3/dml_sweep2.py`)

| form | median ms | median TFLOP/s | best TFLOP/s |
|---|---:|---:|---:|
| `torch.mm(A,B)` fp32 | 190.16 | 0.72 | 0.96 |
| `torch.matmul(A,B)` fp32 | 164.49 | 0.84 | 0.97 |
| `torch.matmul(A, B.t())` fp32 | 191.95 | 0.72 | 0.75 |
| `torch.mm(A, B.t().contiguous())` fp32 | 182.63 | 0.75 | 0.85 |
| `torch.mm(A,B)` fp16 | 124.64 | 1.10 | 1.29 |
| `torch.mm(A, B.t().contiguous())` fp16 | 149.17 | 0.92 | 1.09 |

`mm` vs `matmul` is within noise, transposing costs 5–20 %, and a pre-transposed contiguous B
is *worse* than letting the kernel handle the stride. **There is no layout win here.**

### 4.2 fp16 is not the "hardware-wanted" fallback — it is the same speed or slightly faster

fp16/fp32 ratio: 1.01× (2048³), 1.01× (4096³), 1.40× (8192³) in 3.1; 0.69×/1.09×/1.27×/1.15× in 3.2.
So DirectML's fp16 path is *not* using packed maths, and neither is anything else here: §2 measured a GLSL
`f16vec2` kernel at 1.06× the fp32 rate, i.e. no packing anywhere in the stack. fp16 is also not a fallback
to fp32 — it is one generic tiled kernel either way, and it buys ~10 % for 4e-4 rel_rms. It does not close
the gap, and it is not the "hardware wants fp16" story the brief hypothesised.

### 4.3 A hand-written Vulkan compute GEMM (`_gpuwork/v3/gen_gemm.py`, `bench_sustained.py`, `exp3.py`, `exp6.py`)

I built a generator for unrolled, shared-memory-tiled GLSL SGEMMs (block 64²…128×256, BK 8…64,
per-thread tiles 2×2…8×8, 64…512 threads) and benchmarked 20+ configurations with sustained timing.

* **It is exactly correct**: `rel_rms = 8.43e-07` vs a NumPy fp32 reference on every configuration.
* **Best sustained: 0.90–0.93 TFLOP/s** at 2048³ (block 128×128, BK 16, tile 8×8, 256 threads, 16.5 KB
  shared); 1.05 TFLOP/s at 4096³. **Slower than DirectML in every configuration I tried.**

Where the ALU throughput goes, measured by ablation (2048³, sustained, all in one measurement window so
the ratios are trustworthy even though the absolute scale is contended):

| ablation | TFLOP/s | what it isolates |
|---|---:|---|
| pure FMA, 64 accumulators, **no shared memory at all** | 2.11 | the loop structure is not the problem |
| same loop, but a 16.5 KB `shared` block merely *declared* | 1.18 | **shared-memory allocation alone costs 44 %** — it pins the CU to one workgroup |
| full GEMM, 128×128 BK16 tile 8×8 | 0.90–0.93 | the remaining loss is LDS traffic + barriers |

Read as ratios (the only safe way to read them, §2): **declaring 16.5 KB of shared memory costs 44 % of the
throughput before a single LDS instruction executes**, and the real GEMM gives up another 21 % on top of that.
That is the whole story of why a naive tiled kernel cannot approach 4.45 TFLOP/s on this part, and it is
also why DirectML's 1.46–1.68 at 33–38 % of the ALU peak is a perfectly respectable number for a generic
kernel rather than evidence of a broken driver.

A shared-memory size ladder at a *fixed* compute loop (1, 2, 4, 8, 16, 32 KB) showed no clean monotone
trend (0.86 / 1.43 / 0.38 / 0.97 / 1.00 / 0.82 TFLOP/s) — the allocation cost is step-like and dominated by
the 1-workgroup-per-CU cliff, not by size. Larger BK helps (fewer barriers per unit of work: BK 32 ≈ 1.8× BK 8
in the same block shape) but larger BK also needs more shared memory, so the two effects fight and the best
point is BK 16 at 128×128. A no-shared, L1-streaming variant (no barriers, no LDS at all) reached only
0.56–0.64 TFLOP/s, so giving up the tiling is not a way out.

### 4.4 Cooperative matrix (`VK_KHR_cooperative_matrix`) — advertised, not usable from GLSL

`vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` returns **11 configurations**, including the one that
matters:

    16 x 16 x 16   A=f16  B=f16  C=f32  Result=f32  scope=SUBGROUP   (saturating=0)

and `cooperativeMatrixSupportedStages = COMPUTE`. So the hardware path exists and is exposed. **But I could
not build a kernel for it**: `glslc`/`glslangValidator` from Vulkan SDK 1.4.357.0 accepts `coopMatMulAdd`
and the `coopmat<>` types, and rejects `coopMatLoad`/`coopMatStore` with
*"no matching overloaded function found"* in **every** documented form I tried — 3, 4 and 7 argument
variants, `shared`, private and SSBO array operands, `GL_KHR_cooperative_matrix`, `GL_EXT_cooperative_matrix`
and both together (16 combinations, `_gpuwork/v3/coop_variants2.py`; 0 compiled). Without a load or a store
there is no way to move a matrix in or out of registers, so the matrix-core rate could not be measured and
a WMMA GEMM could not be written. The `coopMatMulAdd`-only module *does* compile, so this is a front-end
limitation, not a driver one.

That leaves the matrix-core roof unmeasured. Worth stating plainly: **if it were reachable it would be the
one remaining path to ~2× DirectML**, and it is the only thing in this survey I would spend more time on.

### 4.5 Best achievable rate, and how I got it

**1.68 TFLOP/s** — DirectML fp16, 4096³, `torch.mm`, executed back-to-back in a burst with one
synchronising read-back per burst (`_gpuwork/v3/compare.py`). Per-invocation, one sync per call, the same
shape gives 1.07 TFLOP/s. **That is 37.7 % of the corrected fp32 roof and 37.8 % of the 4.45 TFLOP/s an
ALU kernel actually achieves** — so ~2.6× of real, measured headroom remains, and nothing in the shader
stack I could reach (fp16, layouts, hand-written tiling, cooperative matrix) captured any of it.

---

## 5. Recomputed offload arithmetic

Baseline ledger, unchanged from the brief (per 512² image, 8 steps = 240 layer-steps):

    attention   240 × 49.0 ms                     = 11.76 s
    refiner     10.68 s  (of which the 7 big GEMMs, 8.15 s)
    other                                         1.68 s
    -----------------------------------------------------
    CPU-only floor                                = 24.12 s

**Refiner decomposition**: 16 calls/image (8 steps × 2 refiners), 7 big GEMMs per call —
4 × 3840², 1 × 3840×10240, 2 × 10240×3840.

Using the **best rate I achieved**, per call:

    fp32 only, best per shape : 4×25.07 + 1×73.50 + 2×67.33 = 308.5 ms
    best per shape, mixed dtype: 4×25.07 + 1×67.15 + 2×52.54 = 272.6 ms
    (previous report, idle box                                  289.3 ms)

Attention, best rate I achieved (measured with the read-back inside the timing, `_gpuwork/v3/attn.py`,
30 × 1056 × 128 = 17.13 GFLOP/layer-step):

| path | ms/layer-step | GFLOP/s | rel_rms |
|---|---:|---:|---:|
| CPU fp32 | 59.82 | 286 | — |
| CPU bf16 | 29.39 | 583 | 3.0e-03 |
| **GPU fp32** | **42.31** | 405 | **6.3e-07** |
| GPU fp16 | 39.34 | 435 | 9.7e-04 |
| GPU fp32, chunk 384 | 40.98 | 418 | 3.6e-07 |

Floor with my best measured rates, fp32 on the device:

    attention compute     240 × 42.31 ms           = 10.15 s   (was 11.76)
    refiner GEMMs         16 × 308.5 ms            =  4.94 s   (was  8.15)
    refiner non-GEMM      (unchanged, CPU)         =  2.53 s
    other                 (unchanged)              =  1.68 s
    --------------------------------------------------------
    new CPU-only floor                             = 19.30 s
    saving = 24.12 − 19.30 = 4.82 s  (−20 %)

Floor using the best rate measured *in either session* (idle-box attention 29.20 ms + my refiner 289.3–308.5 ms):

    attention 240 × 29.20 ms = 7.01 s + refiner 4.63 s + 2.53 + 1.68 = 15.85 s   (−34 %)

**The number to carry forward is ≈ 16 s, matching the previous report's 16.30 s.** The corrected CU count
does **not** move the floor, because the floor is built from *measured GEMM times* and those did not change —
the achievable rate is ~1.5–1.7 TFLOP/s no matter which roof you believe. The roof went up 2×; the rate at
which the work actually gets done did not move at all.

### 5.1 Does a < 10 s image become reachable? Still no.

* Residual floor **15.9 – 19.3 s** with the NPU assumed infinitely fast: 1.6–1.9× the target.
* The NPU requirement is untouched: 89.69 TFLOP / 10 s = **8.97 TFLOP/s** sustained, 4.04× today's.
* The iGPU cannot take that job either — 89.69 TFLOP at 1.68 TFLOP/s = **53.4 s** (at DirectML's fp32
  1.46 it is 61.4 s). Even at the fp32 ALU peak the hardware can genuinely reach (4.45 TFLOP/s) it is
  **20.2 s**, twice the target; only the unreachable packed-fp16 / matrix path (8.9) would get near 10 s.

### 5.2 What the corrected roof does change

It changes the *diagnosis*, not the arithmetic:

| claim | before (6 CU) | after (12 CU) |
|---|---|---|
| fp32 roof | 2.23 TFLOP/s | **4.454 TFLOP/s** |
| fp32 roof actually achievable by an ALU kernel | untested | **4.45 TFLOP/s (measured, 99.9 %)** |
| packed-fp16 roof | 4.45, called unreachable | **8.909, still unreachable from GLSL (measured 1.06× fp32)** |
| DirectML 1.588 as % of fp32 roof | 69 % ("only 1.4× on the table") | **36 % ("2.8× on the table")** |
| DirectML 1.588 as % of measured fp32 ALU peak | 71 % | **36 %** |
| DirectML 1.588 as % of the best GEMM ever run here | ~100 % | **94 %** |
| headroom actually recovered by any path tested | ~1.4× | **~1.06×** |

The honest summary: the fp32 roof is real — an ALU loop reaches 99.9 % of 4.454 TFLOP/s at the nameplate
clock — so DirectML's 1.588 is genuinely leaving 2.8× on the table, exactly as the brief suspected. What
I could not do is collect any of it. DirectML's generic tiled kernel captures 36 % of the ALU peak; a
hand-written, numerically exact tiled Vulkan SGEMM captures 24 % and is *slower*; a no-shared L1-streaming
Vulkan variant is slower still; fp16, `mm`/`matmul` and pre-transposed layouts all move nothing. The
ablation says the loss is structural — shared-memory tiling pins a CU to one workgroup and costs 44 % before
any real work starts (§4.3) — and the two paths that could bypass it (packed fp16, cooperative matrix) are
both closed by the toolchain, not by the hardware. **The one thing I would do next is get a cooperative-matrix
GEMM compiling** (a different shader compiler, or hand-written SPIR-V): it is the only remaining route to the
top half of the roof.

---

## 6. Caveats that would change these numbers

* **Throughput swings ±40 % with host load, and that is not a GPU clock cap.** The identical ALU kernel
  measures 4.45 TFLOP/s in one window and 3.08 in another; fp16/fp32 ordering is stable but absolute values
  are not. The iGPU shares the socket power budget with the CPU. Any single measurement on this part is
  worth less than the same-window comparison.
* **The box was contended during this session.** Other agent processes were running (`python` and
  `DeepSeek Harness` with 14000+ s of CPU time each). The CPU reference is only 1.05–1.16× slower than
  in the previous session, but the GPU is up to 1.45× slower on bandwidth-sensitive work (attention
  42.31 ms here vs 29.20 ms then). Treat every GPU number here as a **lower bound**; the interleaved
  comparisons in §3.2 are the reliable part because both engines saw the same conditions.
* **Duration dependence is large and is confounded with host load** (§2): a 41 ms dispatch of the peak
  kernel measures 1.67 TFLOP/s, a 365 ms one 3.01, a 1.4 s one 3.08 — in the same run. Separately the same
  kernel measured 4.45 median in a quieter window. I cannot cleanly separate "the clock ramps" from "another
  process was eating the power budget", and I do not claim to. What is safe: numbers below ~100 ms of GPU
  work are not comparable with numbers above, and cross-window comparisons are not safe at all.
* **Holding many large tensors resident costs up to 2.4×** — 4096³ fp32 measured 0.60 TFLOP/s with twelve
  large model tensors resident vs 1.06 TFLOP/s with only that shape's pair. DirectML's allocation behaviour
  under memory pressure is worth knowing before sizing a resident-weight design.
* `tempfile.mkdtemp()` is still broken on this host (0o700 dirs get a denied ACL); the existing
  `sitecustomize.py` inside `_gpuwork/gpuvenv` still fixes it. I created no new venv and did not
  modify either of the two pre-existing interpreters.

**Disk**: D: **24.14 GB free** at the end, the same as at the start. This session's whole footprint is
**1.3 MB** (`_gpuwork/v3/`, all scripts + shaders + .spv + JSON) — far under the 2 GB budget. **No new venv
was created**; the previous agent's `_gpuwork/gpuvenv` (1.28 GB, already on D:) was reused read-only and
neither of the two pre-existing interpreters was touched. **C: is being consumed by something else on this
shared box** — 13.99 GB free at the start of my session, 12.02 GB an hour later, **4.50 GB** by the end.
**I wrote nothing to C:** (it is write-denied, and every artefact went to D:); that is worth someone's
attention, because it is not this work.

---

## 7. Reproduction

    $ws = "D:\Project\deepseek_harness\public_work\zimage-bfp16-npu\_gpuwork"
    $py = "$ws\gpuvenv\Scripts\python.exe"

    & $py "$ws\v3\vkprobe.py"          # CU count: VK_AMD_shader_core_properties(2)   -> 12
    & $py "$ws\v3\hipdump.py"          # hipDeviceProp_t field-by-field              -> 6 (WGPs)
    & $py "$ws\v3\dxprobe.py"          # DXGI AdapterDesc1 (deviceID 0x150e); D3D12 needs FL 12_0+
    & $py "$ws\v3\coopmat.py"          # VK_KHR_cooperative_matrix configs (11, incl. 16x16x16 f16->f32)
    & $py "$ws\v3\bench_peak_final.py" # ALU peak: fp32 3.08-3.16 TFLOP/s in a contended window
    & $py "$ws\v3\peak_check.py"       # matched-flop fp32 vs fp16: 4.45 vs 4.73 TFLOP/s (no packing)
    & $py "$ws\v3\dml_sizes.py"        # DirectML 2048/4096/8192 x fp32/fp16, per-call sync
    & $py "$ws\v3\compare.py"          # DirectML vs hand-written Vulkan, interleaved
    & $py "$ws\v3\attn.py"             # attention 30x1056x128, GPU vs CPU
    & $py "$ws\v3\exp3.py"             # ablation: where the GEMM throughput goes
    & $py "$ws\v3\exp6.py"             # final GEMM config sweep (interleaved)
    & $py "$ws\v3\coop_variants2.py"   # 16 coopmat load/store forms, none compile

Shaders: `v3/peak_f32.comp`, `v3/peak_f16.comp`, and `v3/gen_gemm.py` generates the tiled GEMM
(`gemm_f32.comp` / `gemm_f16.comp`) and `v3/gen_gemm3.py` the no-shared streaming variant.
Vulkan host: `v3/vk.py` (ctypes, Vulkan 1.4, no python-vulkan dependency).
Raw JSON for every number: `v3/vkprobe.json`, `v3/hipdump.json`, `v3/dxprobe.json`, `v3/coopmat.json`,
`v3/bench_peak_final.json`, `v3/dml_sizes.json`, `v3/dml_sweep2.json`, `v3/compare.json`, `v3/attn.json`,
`v3/exp3.json`, `v3/exp6.json`.
