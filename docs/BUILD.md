# Build and environment

Everything below was measured on the reference box (`docs/reports/ENVIRONMENT.md`);
versions are exact because this project has been bitten by "the same" version that
was not.

## 1. The two-interpreter split is real, not stylistic

The product hard-codes **two** interpreters (`webui/config.py`), and the reason is
that the two scientific stacks do not coexist:

| | Python 3.10.9 | Python 3.12.11 |
|---|---|---|
| numpy | 1.26.4 | 1.26.4 |
| torch | **2.3.1 (+cpu)** | **2.4.1 (+cpu)** |
| numba / llvmlite | **0.62.1 / 0.45.1** | absent |
| ml_dtypes | absent | **0.5.4** |
| onnx | — | 1.22.0 |
| llvm-aie | — | **1.8.0** |
| diffusers | — | 0.36.0 |

* **3.10** runs the device worker and the numba glue: `numba` exists only here.
* **3.12** runs the planner: `ml_dtypes` and `llvm-aie` exist only here.

Override the paths with `IMAGGER_PY310` / `IMAGGER_PY312`. The preflight check P01
verifies both actually run.

`torch` is a **CPU-only** build on both. There is no CUDA/HIP backend on this
platform, and `torch-directml` is **not installed and not installable** in the
reference environment (absent from all 179/145 distributions; the index is not
reachable). **No document in this repository may depend on it.**

## 2. The NPU stack

| component | version | notes |
|---|---|---|
| XRT | **2.21.0** | from `npu_mcdm_stack_prod`; `IMAGGER_XRT` overrides the path |
| MLIR-AIE | **1.4.4** | shipped as an *unzipped wheel*, not a pip distribution |
| Peano | **1.8.0** | the AIE compiler backend |
| driver | 376 | reported by `xrt-smi` |

You need this stack only to **rebuild** kernels. This repository ships the compiled
images (**66 files, ~8.8 MB**, in `webui/cache/<shape>/` and `imaggers/*/out/`), so a
fresh clone can run without any of it.

**Open question worth settling before you trust `IMAGGER_XRT`:** `ENVIRONMENT.md`
and `webui/config.py:24` describe the driver directory differently. Resolve with
`xrt-smi examine --report` plus hashes of the three `xrt_coreutil.dll` copies.

## 3. Rebuilding a device image

The kernels are MLIR-AIE designs compiled to `xclbin` plus an instruction stream:

```
<name>.mlir  --(aiecc / Peano)-->  <name>.xclbin + <name>.bin
```

The build helpers are in `imaggers/attn_impl/` (`t18_attn_compile.py`, `t5_build_win.py`)
and `imaggers/block/` (`t4_compile.py`, `t4_build_win.py`). Each writes a
`*.compile.json` beside the image; those are the provenance records and they are
committed.

**A hard constraint that shapes every kernel:** the images are compiled for a fixed
number of row tiles (**RT**). An RT=11 image computes 1056 rows and *silently leaves
the rest of the drain untouched*. The projection guard (`proj_block_rows`) exists
because of this — see §5 of the README. Never dispatch a fixed-tile image for more
rows than it has tiles.

## 4. Optional: the GPU

The iGPU (Radeon 880M) is not used by the product. Two paths exist and neither is
wired in:

* **ggml + Vulkan** — present and working as `ggml-vulkan.dll` beside
  `sdvae-bench/vbench.exe`; measured **4.02x faster than the CPU** on a 512² VAE
  decode (2.264 s vs 9.104 s, `unsupported ops: 0/321`). Device reports
  `matrix cores: KHR_coopmat`.
* **hand-written Vulkan compute** in `_gpuwork/` — **broken**: its fp32 GEMM writes
  only columns 0..71 of 128, i.e. 56 % of its output. Any number ever derived from it
  is void, in either direction.

The Vulkan SDK (1.4.357.0, `glslangValidator`/`glslc`) is needed only for the second.

## 5. Third-party trees that are NOT here

Referenced by path in the build scripts, never redistributed:

| tree | role |
|---|---|
| `llama.cpp` (Vulkan build) | provides `ggml.dll`, `ggml-vulkan.dll` for the text encoder worker |
| `k2h-llama.cpp` | the build used for those DLLs |
| `sdvae-bench` | the ggml VAE benchmark that measured the 4x |
| AIE / Peano / XRT | the NPU toolchain, §2 |

## 6. Reproducing a number

See `docs/MEASUREMENTS.md`. The short version: the loop wall resolves to only ±2 s,
so the **per-section columns are the evidence** and the wall is corroboration; and
every arm needs its own 8-step quality gate, because a per-call correlation cannot
predict an 8-step one.
