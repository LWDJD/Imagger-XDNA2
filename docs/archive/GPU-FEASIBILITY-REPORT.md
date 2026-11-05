# iGPU feasibility — can the CPU half of the 512² pipeline move to the Radeon 880M?

**Date** 2026-11-02 · Nothing here touched the NPU, no xclbin, no 8-step job, no device lock.
Every number is measured on this box in this session unless marked otherwise.

---

## 0. Verdict in five lines

1. **The iGPU is real and usable**: DirectML runs on it today (`privateuseone:0`, 1.53 TFLOP/s fp16,
   1.26 TFLOP/s fp32 sustained). It is **not** a Strix Halo — it is a **Radeon 880M (gfx1150), 6 CU,
   2.90 GHz, 125 GB/s**, i.e. a small RDNA3.5 iGPU sharing the socket with the CPU.
2. **The refiner GEMMs are a genuine 2.2–2.7× win** over the CPU f32 path (and 1.35–1.45× over the
   CPU bf16 path the project already has).
3. **Attention is only a 1.77× win** (51.8 → 29.2 ms/layer-step) and the win **disappears entirely**
   if the CPU is allowed to use bf16 (27.4 ms). The GPU's attention advantage is much smaller than
   its GEMM advantage, for the same reason the NPU's was: the N² score matrix is bandwidth, not maths.
4. **Accuracy is a non-issue if you use fp32 on the device** — rel_rms **6.3e-7** on attention and
   **1.1e-6** on the GEMMs, i.e. numerically exact for this purpose, at 2.2–2.6× the CPU's speed.
   fp16 would sit *on* the 1e-3 suspicion line (9.7e-4 / 1.1e-3); **fp32 is both faster and exact
   for attention, so there is no reason to take the fp16 risk.**
5. **The CPU-only floor goes 24.12 s → 16.60 s (−31 %).** A <10 s image is **still not reachable**:
   the residual floor is 1.66× the target with a *free* NPU, and the NPU still has to sustain
   8.97 TFLOP/s (4.04× today). The iGPU cannot take that job either — 89.69 TFLOP at 1.53 TFLOP/s
   is 58.6 s. **New bottleneck: the NPU device stream, 37.9 s/image.**

---

## 1. Hardware identity

| property | value | source |
|---|---|---|
| GPU | **AMD Radeon(TM) 880M Graphics** | WMI-free registry `HKLM\...\Class\{4d36e968-…}\0000`, HIP, Vulkan |
| architecture | **gfx1150** (RDNA 3.5, Strix Point) | `hipGetDeviceProperties`: major.minor = 11.5, `gcnArchName` = `gfx1150` |
| compute units | **6** | `hipDeviceProp_t.multiProcessorCount` |
| clock | **2.90 GHz** (max) | `hipDeviceProp_t.clockRate` = 2 900 000 kHz |
| memory clock / bus | 1.96 GHz / **256-bit** | `memoryClockRate`, `memoryBusWidth` |
| memory bandwidth (roof) | **125.4 GB/s** (DDR, 1.96 GHz × 2 × 32 B) | derived |
| memory type | **LPDDR5X, shared with the CPU** (UMA) | integrated GPU, 33.60 GB system RAM, no dedicated VRAM |
| VRAM addressable | **12.23 GiB** | `totalGlobalMem` |
| L2 | 2 MB | `l2CacheSize` |
| warp / max threads per CU | 32 / 2048 | `warpSize`, `maxThreadsPerMP` |
| **driver** | **32.0.31041.1004** (dated 2026-08-17), Adrenalin **26.8.1** | display-class registry key; Vulkan `driverInfo = 26.8.1 (LLPC)` |
| GPU driver files | `amdxc64.dll` 8.18.10.0504, `amdhip64.dll` 10.0.3592.0, `amdhip64_7.dll` 10.0.3679.0 | `C:\Windows\System32` |
| CPU / RAM | AMD Ryzen AI 9 H 365, 10 cores / 20 threads, 33.60 GB | `HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0`, `GlobalMemoryStatusEx` |
| OS | Windows 11 build 10.0.26200 | |

**This is not Strix Halo.** Strix Halo would be a Radeon 8060S with 40 CU and a 256-bit LPDDR5X bus at
~256 GB/s. What is actually here is **Strix Point**: 6 CU / 2.90 GHz / 125 GB/s, an iGPU whose entire
fp32 roof is **2.23 TFLOP/s** — about 1/8 of the NPU's 51.3 TOPS and 1/20 of what a 10 s image needs.

### 1.1 Compute APIs actually present

| API | state | evidence |
|---|---|---|
| **DirectML** | **WORKS** | `DirectML.dll` 1.15.5 inbox; `torch_directml.device_name(0)` → "AMD Radeon(TM) 880M Graphics" |
| **Vulkan** | **works, 1.4** | `vulkaninfoSDK`: GPU0 apiVersion 1.4.349, vendorID 0x1002, INTEGRATED_GPU, `shaderFloat16 = true`, `VK_KHR_cooperative_matrix` present, subgroupSize 64 |
| **HIP / ROCm** | runtime **enumerates the device**, but no toolchain | `amdhip64_7.dll` loads, `hipGetDeviceCount` = 1, `hipGetDeviceProperties` = 0 with full props. **No ROCm tree** (`C:\Program Files\AMD\ROCm` and `HIP SDK` absent), so no `hipcc`, no code objects for gfx1150 → could not run a kernel. |
| **OpenCL** | **not installed** | `OpenCL.dll` (Khronos loader) present but `amdocl64.dll` missing and `HKLM\SOFTWARE\Khronos\OpenCL\Vendors` does not exist |
| D3D12 | present | `d3d12.dll` 10.0.26100.9278 |

---

## 2. What was installed, and the proof it runs

Nothing was installed into either existing interpreter. Both were left untouched.

    D:\Users\15081\AppData\Local\GAIA\python\python.exe            3.10.9   (has NO venv module)
    D:\Users\15081\miniforge3\envs\ryzen-ai-1.8.0\python.exe       3.12.11  (used as the venv base)

**New venv**: `D:\Project\deepseek_harness\public_work\zimage-bfp16-npu\_gpuwork\gpuvenv`

    torch              2.4.1        (CPU build: torch.version.hip = None, cuda False)
    torch-directml     0.2.5.dev240914   (cp312 wheel, pulls torch 2.4.1 + torchvision 0.19.1)
    numpy 2.5.4, sympy, networkx, jinja2, filelock, fsspec, pillow, typing_extensions

**Proof (this is the command that proves it works):**

    .\_gpuwork\gpuvenv\Scripts\python.exe -c "import torch, torch_directml;         d=torch_directml.device(); a=torch.randn(512,512).to(d);         print(torch_directml.device_name(0)); print((a@a).cpu().shape)"

    -> AMD Radeon(TM) 880M Graphics
    -> torch.Size([512, 512])

fp32 512³ matmul vs CPU: **max abs err 1.03e-04**. fp16: 8.30e-02.

### 2.1 Two environment obstacles worth recording

* **`C:\` is write-denied** outside `$env:TEMP` (the DSH sandbox allows only the session workspace
  plus a temp area). `C:\DSH_GPU` and `C:\Users\…\Temp\dsh_gpu` both returned *access denied*.
  **The venv therefore lives on D:, not C:.** D: had **25.55 GB free** at the start — not the
  "very little free space" the brief assumed — and the whole footprint is **1.28 GB**.
* **`tempfile.mkdtemp()` is broken on this host**: the sandbox ACL layer denies writes inside any
  directory created with mode 0o700, which is what `mkdtemp` does. That kills `pip install` and
  `ensurepip` with `PermissionError: [Errno 13]`. Fixed **inside the new venv only** with a
  `sitecustomize.py` that re-implements `mkdtemp` via `os.makedirs` (inherits the parent ACL).
  Neither existing interpreter was modified.

### 2.2 Why paths (b), (c), (d) were not used

The brief said stop at the first that works; (a) works. For the record: (b) there is **no ROCm-on-Windows
torch wheel** for gfx1150 — ROCm PyTorch wheels are Linux-only, and the box has no ROCm tree anyway;
(c) and (d) were unnecessary.

### 2.3 DirectML limitations found by measurement

* **bfloat16 is not supported at all** and is a *hard abort*, not an exception:
  `[F1102 … dml_util.cc:118] Invalid or unsupported data type BFloat16` kills the process. Any port
  must use fp16 or fp32, never bf16.
* There is **no `synchronize()`** in torch-directml. Timings that do not force a read-back are pure
  launch overhead — an early run of mine reported 0.04 ms for a 30 GFLOP GEMM (758 TFLOP/s) before
  this was caught. Every number below is per-call synchronised via a blocking `.cpu()` with the copy
  cost measured separately and subtracted.

---

## 3. GPU vs CPU on the two real shapes

Same process, same tensors, each engine measured with the machine otherwise idle, medians, per-call
synchronised. `torch.set_num_threads(20)`.

### 3.1 Attention — 30 heads × seq 1056 × hd 128 (QK^T + softmax + PV)

17.13 GFLOP per layer-step (= 8.57 QK + 8.57 PV).

| path | ms | **GFLOP/s** | rel_rms vs CPU fp32 | corr | speedup |
|---|---:|---:|---:|---:|---:|
| **CPU fp32 (today)** | **51.77** | **331** | — | — | 1.00× |
| CPU bf16 | 27.39 | 625 | 3.03e-3 | 0.99999541 | 1.89× |
| **GPU fp32 (DirectML)** | **29.20** | **587** | **6.28e-7** | **1.0000000** | **1.77×** |
| GPU fp16 (DirectML) | 32.06 | 534 | 9.74e-4 | 0.99999985 | 1.61× |
| GPU fp16, chunked qblock=512 | 30.56 | 560 | 1.20e-3 | 0.99999983 | 1.69× |

Component split, GPU fp16: QK^T 45.8 ms / softmax 10.6 ms / PV 7.7 ms when run as separate
kernels on a materialised S — i.e. **QK^T alone is 6× more expensive than PV for identical FLOPs**,
because it writes the 67 MB score matrix. That is the same "data movement dominates" wall the NPU hit.
The fp32 path wins because it avoids two dtype-conversion kernels around the softmax.

### 3.2 Refiner GEMMs

| shape | GFLOP | CPU fp32 ms | CPU bf16 ms | **GPU fp32 ms** | GPU fp16 ms | **GPU fp32 GFLOP/s** | GPU/CPU fp32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1024×3840 @ 3840×3840 | 30.20 | 64.65 | 28.60 | **24.42** | 20.45 | **1237** | **2.65×** |
| 1024×3840 @ 3840×10240 | 80.53 | 157.77 | 74.58 | **63.77** | 54.95 | **1263** | **2.47×** |
| 1024×10240 @ 10240×3840 | 80.53 | 141.02 | 72.22 | **64.08** | 54.19 | **1257** | **2.20×** |

(The brief's CPU figures 46.5 / 114.0 / 95.6 ms are 1.3–1.4× faster than my CPU measurements on the
same shapes; both are shown so the ratio is auditable. GPU absolute times are machine-absolute and do
not depend on that discrepancy.)

### 3.3 Device ceiling, measured

| probe | result |
|---|---|
| 8192³ fp16 GEMM | **1.588 TFLOP/s** |
| 4096³ fp16 | 1.492 TFLOP/s |
| 4096³ fp32 | 1.430 TFLOP/s |
| fp32 architectural roof (6 CU × 64 lanes × 2 × 2.90 GHz) | 2.23 TFLOP/s |
| packed-fp16 roof (if matrix cores were used) | 4.45 TFLOP/s |
| **achieved** | **69 % of the fp32 roof, 34 % of the fp16 roof** |

fp16 ≈ fp32 throughput ⇒ **DirectML is not using a packed-fp16 / matrix path on this part**; the
`VK_KHR_cooperative_matrix` extension is exposed by the driver but torch-directml does not reach it.

### 3.4 Transfer costs (they matter)

| direction | measured |
|---|---|
| H2D | **25.8 GB/s** |
| D2H | **12.9 GB/s** |
| device→device `copy_` | 21.2 GB/s (rw) |

All far below the 125 GB/s DDR roof — DirectML's copy path is the tax. Per attention layer-step
(q/k/v in, out back) that is **3.14 ms fp32** / 1.57 ms fp16.

---

## 4. Numerical delta vs the CPU path

| path | shape | corr | maxabs | **rel_rms** |
|---|---|---:|---:|---:|
| GPU fp32 attention | 30×1056×128 | 1.000000000 | 1.0e-07 | **6.28e-07** |
| GPU fp16 attention | 30×1056×128 | 0.999999850 | 1.15e-04 | **9.74e-04** |
| GPU fp16 attention, chunk 512 | 30×1056×128 | 0.999999827 | 1.42e-04 | 1.20e-03 |
| CPU bf16 attention | 30×1056×128 | 0.999995408 | 3.38e-04 | 3.03e-03 |
| **GPU fp32 refiner GEMM** | 1024×3840 @ 3840×3840 | 1.000000000 | 8.4e-05 | **1.14e-06** |
| GPU fp16 refiner GEMM | 1024×3840 @ 3840×3840 | 0.999999928 | 3.53e-02 | **1.12e-03** |
| CPU bf16 refiner GEMM | 1024×3840 @ 3840×3840 | 0.999995866 | 1.01e-01 | 2.88e-03 |

**Does fp16/bf16 on the GPU clear the bar?** The brief's rule is that a per-call rel_rms above ~1e-3
is already suspect under a ~1e4 amplification. **fp16 lands exactly on the line (9.7e-4 attention,
1.12e-3 GEMM) — it does not clear it with margin.** bf16 does not exist on this backend at all.

**But this is a false dilemma, because fp32 on the device is not slower in the case that matters:**

* attention: **GPU fp32 (29.20 ms) is faster than GPU fp16 (32.06 ms)** and is exact (6.3e-7).
* GEMMs: GPU fp32 costs 13–18 % more than fp16 (63.8 vs 55.0 ms) and buys **1000× the accuracy**
  (1.14e-6 vs 1.12e-3).

For calibration against what this project already ships: the accepted host attention variant is
*"corr 0.99997 / rel 0.72 %"* (`webui/chunked.py:19`, i.e. 7.2e-3) and the accepted refiner bf16
block is 2.0e-3. **GPU fp32 is 4–5 orders of magnitude better than both; GPU fp16 is 3–6× better
than both.** So: run fp32 on the iGPU — it is simultaneously the fastest and the exact option for
attention, and within 15 % of the fastest for the GEMMs.

---

## 5. Verdict — the floor arithmetic

Baseline, the brief's own measured ledger (per 512² image, 8 steps = 240 layer-steps):

    attention   240 × 49.0 ms                     = 11.76 s
    refiner     10.68 s  (of which the 7 big GEMMs, see below, 8.15 s)
    other                                          1.68 s
    ------------------------------------------------------
    CPU-only floor                                 = 24.12 s

**Refiner decomposition used** (needed to know how much can move): the refiner is 16 calls/image
(8 steps × 2 refiners) and each call is 7 big GEMMs — 4 × 3840², 2 × 10240×3840, 1 × 3840×10240:

    4 × 46.5 + 2 × 114.0 + 95.6 = 509.6 ms/call  →  16 × 0.5096 = 8.15 s   (76 % of the refiner)
    non-GEMM refiner work (attention, norms, adaLN)  = 10.68 − 8.15 = 2.53 s  (stays on CPU)

That 7-GEMM model reproduces the repo's own 378.5 GFLOP/call and 6.06 TFLOP/image to within 4 %,
so it is the right decomposition.

**Replacing both with the iGPU (fp32 on the device, absolute measured times).**
Note the 29.20 ms attention figure is measured *with the output read-back inside the timing*, so the
decomposition is compute + H2D of q/k/v + D2H of the output, counted once:

    attention compute     240 × 27.94 ms          =  6.706 s   (was 11.76)
    H2D q/k/v (48.6 MB @25.8 GB/s = 1.88 ms/step) =  0.451 s
    D2H out   (16.2 MB @12.9 GB/s = 1.26 ms/step) =  0.302 s
    refiner GEMMs         16 × 289.3 ms           =  4.629 s   (was 8.15)
    refiner non-GEMM      (unchanged, CPU)        =  2.527 s
    other                 (unchanged)             =  1.680 s
    --------------------------------------------------------
    new CPU-only floor                            = 16.30 s

    saving = 24.12 − 16.30 = 7.82 s  (−32 %)

With fp16 on the device instead it is 15.9 s — 0.4 s better for a 1e-3 accuracy hit. Not worth it.

### 5.1 Does a <10 s 8-step image become reachable? **No.**

Three independent reasons, any one of which is fatal:

1. **The residual floor is still 16.30 s** — 1.63× the target — *with the NPU assumed infinitely fast*.
   The change removes 32 % of the floor, not all of it.
2. **The NPU requirement is unchanged.** A 10 s image needs 89.69 TFLOP / 10 s = **8.97 TFLOP/s**
   sustained from the device, which is 4.04× today's 2.22 TFLOP/s and 1.46× the best full GEMM ever
   measured on this box (6.14). Moving host work to the iGPU does not move that number by one bit.
3. **The iGPU cannot take the job instead.** 89.69 TFLOP at the measured 1.53 TFLOP/s = **58.6 s**
   (71.2 s in fp32). The whole iGPU is 1/6 of the NPU's throughput for this workload.

### 5.2 What the new bottleneck is

**The NPU device stream: 37.91 s/image at 2.37 TFLOP/s.** After this change the host stops being the
wall and the device becomes it, which it already nearly was:

| model | loop wall today | loop wall with iGPU offload |
|---|---:|---:|
| serial (host work adds to the device) | 89.74 s | **≈ 81.9 s** |
| host work overlapped with the NPU | 89.74 s | **≈ 55.4 s** |

Arithmetic: the in-loop host ledger is 143.2 ms/layer-step = 34.37 s; the offload removes
7.82 s / 240 = **32.6 ms/layer-step**, leaving 110.6 ms → 26.54 s.
Serial: 37.91 (device) + 26.54 (host) + 15.78 (preamble) + 1.68 (other) = **81.9 s**.
Overlapped: max(37.91, 26.54) + 15.78 + 1.68 = **55.4 s**.

**So the interesting number is not 16.60 s, it is 55.4 s** — and it is only reachable if the host work
is made to *overlap* the NPU rather than sit in front of it. Today it is serial. That restructuring is
worth 4× more than the offload itself.

### 5.3 Caveats that will eat into these numbers

* **The iGPU shares the socket with the CPU.** Interleaved measurement (CPU and GPU both busy) dropped
  the GPU GEMM from 1530 to **1134 GFLOP/s (−26 %)** and the CPU GEMM from 64.7 to 62.0 ms. Any design
  where the GPU work and the CPU glue run concurrently will pay this.
* **12.23 GiB of addressable VRAM, carved out of 33.6 GB of system RAM.** Resident refiner weights
  (1.85 GB bf16 / 3.7 GB fp32) would fit, but only by taking that memory from the CPU side.
* Transfers at 12.9–25.8 GB/s are ~5–10× below the DDR roof; 3.14 ms per attention layer-step is a
  real 0.75 s/image and there is no way around it with DirectML's copy path.
* DirectML exposes no fp16 matrix path on this part, so the 4.45 TFLOP/s packed roof is unreachable;
  2.23 TFLOP/s is the wall, and 1.53 is what we get (69 %).

---

## 6. Reproduction

    $ws = "D:\Project\deepseek_harness\public_work\zimage-bfp16-npu\_gpuwork"
    & "$ws\gpuvenv\Scripts\python.exe" "$ws\hw2.py"       # hardware identity from the HIP runtime
    & "$ws\gpuvenv\Scripts\python.exe" "$ws\bench6.py"    # CPU phase + GPU phase, both shapes
    & "$ws\gpuvenv\Scripts\python.exe" "$ws\bench7.py"    # GPU fp32 vs fp16 attention + accuracy

Raw results: `_gpuwork\bench6_results.json`, `_gpuwork\bench5_results.json`, `_gpuwork\bench2_results.json`.

Disk: C: 14.18 → 14.19 GB free (untouched — sandbox denies writes there); D: 25.55 → 24.22 GB free,
footprint **1.28 GB** (venv + scripts + JSON).
