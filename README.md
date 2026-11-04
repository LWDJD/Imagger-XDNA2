# Imagger-XDNA2

**Z-Image-Turbo 512²–1080p image generation on the AMD XDNA2 NPU**, with a WebUI,
a BlockFP16 weight format, and a measured performance record.

This repository is the **clean, runnable core**. It is not the research tree: every
file here is reachable from `webui.py` or from a build script in
[docs/guides/BUILD.md](docs/guides/BUILD.md).

> **Status:** runs on Ryzen AI 9 H 365 / Radeon 880M (XDNA2 "Strix", 32 AIE tiles).
> Verified byte-reproducible: 512² x8 steps -> latents `2abf91331200d069…`,
> reproduced from **this** tree after the move.

---

## 1. What this does

| | |
|---|---|
| Model | Z-Image-Turbo, a 30-layer DiT + text encoder + VAE, distilled to 8 steps |
| Weights | **BlockFP16 (BFP16)** — **9 bytes per 8 values**: one scale byte plus 8 int8-range values, `value = m x 2^(byte-133)`. Peak device rate **49.152 TOPS** = 32 cores x 512 MAC/cycle x 2 x 1.5 GHz |
| Backend | AMD XDNA2 NPU via XRT + MLIR-AIE compiled `xclbin` images |
| Host | the CPU does attention, the refiner pre-stage, norms/rope/packing |
| Resolutions | 512², 1024², 1920x1080, 1080x1920 and other multiples of 64 (see §6) |
| Batch | `--batch B`; with the two-chain engine two images are in flight at once |

**BFP16 is an 8-bit format. It is not bf16.** See [docs/guides/BFP16.md](docs/guides/BFP16.md).

## 2. Measured performance (512², 8 steps, AC power)

| configuration | loop | note |
|---|---:|---|
| delivered default | ~90–94 s | the numbers in this README's tables |
| + `ZIMAGE_SKIP_STEPS=7` | ~78–81 s | one step skipped, latents corr 0.9999748 |
| + fast attention (`T22_ATTN_*`) | ~81 s | corr 0.9975444 |
| + `--batch 2` two chains | 70.5 s/image | −19.7 %, byte-identical to serial |

Wall clock resolves to only ±2 s; the per-section columns in the logs are the real
evidence ([docs/guides/MEASUREMENTS.md](docs/guides/MEASUREMENTS.md)). Device utilisation: 45 %
single-chain, 56–59 % two-chain.

## 3. Requirements

### Hardware
* AMD Ryzen AI (XDNA2) NPU with 4+ columns. Verified on Ryzen AI 9 H 365 (32 tiles).
* ~13 GB free commit at 512²; **17 GB** for 1080p.

### Software that this repository does **not** contain

Nothing in this list is redistributed here — install it yourself.

| dependency | why | where |
|---|---|---|
| **AMD NPU driver + XRT** | the device runtime | [docs/guides/BUILD.md](docs/guides/BUILD.md) §2 |
| **MLIR-AIE / Peano** | compiles the `xclbin` images | only to rebuild kernels; this repo ships the built ones |
| **Python 3.10 and 3.12** | the product hard-codes two interpreters | `webui/config.py`, override with env vars, §5 |
| `torch` (CPU build) | the host maths | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| `numpy`, `numba` | host maths, fused softmax | pip (numba on 3.10 only) |
| **Vulkan SDK** (optional) | only for the GPU experiments | `glslangValidator`, `glslc` |

`torch` here is a **CPU-only** build. There is no CUDA/HIP backend on this platform;
the GPU is reachable only through `ggml`+Vulkan or `torch-directml`, neither of which
the product uses yet. See [docs/guides/GPU.md](docs/guides/GPU.md).

### Artifacts you must supply (not in git — too large)

| artifact | size | how to get it |
|---|---:|---|
| `webui/models/zimage-te-qwen3-4b-bf16.gguf` | 8.05 GB | **download** — the Qwen3-4B text encoder |
| `weights.bfp16` (in `zimage-port/`, outside this repo) | 6.91 GB | **build or download** — the wpack's only input |
| `weights/block*.npz` (30 files) | 6.38 GB | **build** from the upstream model |
| `webui/cache/wpack/wpack_l30_v2.bin` | 1.90 GB | **build** from `weights.bfp16`, ~33 s |
| `brepack_cache/brepack_30_k32_*.bin` | 5.31 GB | **build once**, content-addressed |

A fresh clone cannot generate an image until those are in place. Everything else —
including the compiled NPU images (**66 files, ~8.8 MB**) — is in this repository.
See [docs/guides/WEIGHTS.md](docs/guides/WEIGHTS.md), which also records the fact that the upstream
transformer shards are **gone** on the reference box, so `weights/block*.npz` and
`weights.bfp16` currently cannot be rebuilt there.

## 4. Quick start

```powershell
git clone https://github.com/LWDJD/Imagger-XDNA2
cd Imagger-XDNA2
git checkout dev

# 1. put the artifacts from section 3 in place
# 2. point the product at your interpreters and driver
$env:IMAGGER_PY310 = "C:\path\to\python310.exe"
$env:IMAGGER_PY312 = "C:\path\to\python312.exe"
$env:IMAGGER_XRT   = "C:\path\to\npu_mcdm_stack_prod"

# 3. prove the path before generating anything (host-only, no device)
.\imaggers\repro\t22_run.ps1 -Action preflight -Size 512 -Steps 8

# 4. generate
.\imaggers\repro\t22_run.ps1 -Size 512 -Steps 8 -Tag first
# or the WebUI:
python webui.py     # then open http://127.0.0.1:4168
```

See [scripts/README.md](scripts/README.md) for the three commands you will actually
use.

## 5. Configuration

Hard-coded paths are overridable by environment variable:

| variable | default | meaning |
|---|---|---|
| `IMAGGER_PY310` | the shipped absolute path | interpreter for the device worker |
| `IMAGGER_PY312` | the shipped absolute path | interpreter for the planner |
| `IMAGGER_XRT` | the shipped absolute path | the XRT driver stack |
| `IMAGGER_PEANO` | `_pylibs/llvm-aie/llvm-aie` | MLIR-AIE toolchain (build only) |

Runtime switches are environment flags, documented one per entry in
[docs/reference/KNOBS.md](docs/reference/KNOBS.md) — 177 variables, classified by whether they can move a
byte. The important ones:

| flag | default | effect |
|---|---|---|
| `T22_FASTGLUE` | 1 in the product recipe | installs the 8 host/device hooks |
| `T22_BREPACK_CACHE` | 1 in the product recipe | maps the 5.31 GB B-repack cache; startup 47.2 -> 17.2 s |
| `T22_PROJ_BLOCK` | `1` = 1056 rows | projection block size; **must stay on above 1056 rows** (§7) |
| `ZIMAGE_SKIP_STEPS` | unset | e.g. `7`; −9…−12.5 s at corr 0.9999748 |
| `T22_ATTN_REUSE` | 0 | fast attention; corr 0.9975444 |
| `T22_CHAIN2B` | 0 | two-chain batch engine; byte-identical, −19.7 % |

## 6. Resolution support

The NPU image is compiled for a fixed number of row tiles (RT). **A fixed-tile image
cannot compute more rows than it has tiles**, so the projection runs in blocks sized
to the image's own tile count (1056 rows = RT 11). This is enforced — the guard
refuses to run rather than read stale memory.

| resolution | rows (M) | path | status |
|---|---:|---|---|
| 512x512 | 1056 | 1 projection launch | byte-reproducible |
| 1024x1024 | 4128 | 4 blocks | byte-identical to the reference |
| 1920x1080 / 1080x1920 | 8256 | 8 blocks | generates a correct image |
| anything else | `ceil(h/8)*ceil(w/8)+32` | same rule | each side must be 64..2048 |

## 7. Things that will bite you

* **Rows above the tile count.** A single launch of an 11-tile image handed 86 tiles
  of rows leaves 75 tiles of *stale* buffer — a plausible-looking image that is wrong
  above row 1056. This is a real, measured failure mode; the projection block guard
  exists for it.
* **The B-repack cache is content-addressed over the payload k-block order.** A cache
  built in a different order changes the arithmetic in the last bits, which amplifies
  ~1.55x per layer and reaches 4.19e-2 by step 8. Do not mix caches.
* **Two chains need one shared read-only context set.** Every `pyxrt.kernel()` is its
  own hardware context; a chain needs 18 and the ~27th is refused with `0xc01e0009`.
  Two processes, and two independently registered `Dev`s, are both impossible — see
  [docs/guides/ARCHITECTURE.md](docs/guides/ARCHITECTURE.md) §4.
* **A flag that never reached the process looks exactly like a flag that did
  nothing.** See [docs/guides/MEASUREMENTS.md](docs/guides/MEASUREMENTS.md) §5.

## 8. Documentation

| file | contents |
|---|---|
**[docs/README.md](docs/README.md) is the index of the whole documentation tree.**

| folder | files | what it is |
|---|---:|---|
| [docs/guides/](docs/guides/) | 6 | written for this repository, current by construction |
| [docs/reference/](docs/reference/) | 2 | [KNOBS.md](docs/reference/KNOBS.md) (177 flags) and [API.md](docs/reference/API.md) |
| [docs/reports/](docs/reports/) | 24 | curated engineering records + [INDEX.md](docs/reports/INDEX.md) |
| [docs/archive/](docs/archive/) | 450 | everything else, paths mirroring the original tree |

The guides:

| file | answers |
|---|---|
| [ARCHITECTURE.md](docs/guides/ARCHITECTURE.md) | the 20-segment chain, where time goes, the device model |
| [BUILD.md](docs/guides/BUILD.md) | the two interpreters, the NPU toolchain, how the `xclbin` images were built |
| [WEIGHTS.md](docs/guides/WEIGHTS.md) | upstream model -> `block*.npz` -> `wpack` -> `brepack`, **and the blocker** |
| [BFP16.md](docs/guides/BFP16.md) | the 9-byte/8-value format and why it is not bf16 |
| [MEASUREMENTS.md](docs/guides/MEASUREMENTS.md) | how to measure without fooling yourself |
| [GPU.md](docs/guides/GPU.md) | the iGPU situation, honestly |

## 9. Verification status

This is a research-driven port, so the repository records what is *proven*:

* 512²x8 -> `2abf91331200d069…`, reproduced across sessions **and from this tree after
  the move** (preflight 12/12 PASS, loop 94.09 s, device released cleanly).
* 1024²x1 -> `af993cadfecbc419…`, byte-identical to an independent chunked path.
* 1920x1080 -> a real image, device peak 12.71 GB.
* Fast attention, skip-step and two-chain arms each carry an 8-step quality gate and
  a **negative control that is shown to fail**.

## 10. License

MIT — see [LICENSE](LICENSE).