# BUILD — how to obtain everything this project needs

**This repository contains only original work.**  No third-party source, no
compiled binaries and no vendor artifacts are committed.  Everything below is
obtained separately and placed at the paths the scripts expect.

## 1. Python interpreters (two of them, on purpose)

| role | interpreter | why |
|---|---|---|
| compile / encode | `<python312>\python.exe` (3.12) | `aie.utils.bfp`, `ml_dtypes` and the IRON API live here |
| run (pyxrt) | `<python310>\python.exe` (3.10) | `pyxrt.pyd` links `python310.dll` |

Any Python 3.12 with `numpy`, `ml_dtypes` and the packages below works for the
compile side; the run side must be a 3.10 matching the installed XRT.

## 2. mlir-aie + Peano (the open AIE toolchain)

    pip install mlir_aie==1.4.4

then the matching Peano.  The pin is in the mlir-aie repo at tag v1.4.4:

    utils/peano-requirements.txt  ->  llvm-aie==22.0.0.2026092101+0006955e

    pip install llvm-aie==22.0.0.2026092101+0006955e \
        -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly

**The Peano version matters.**  The 21.0.x nightlies fail to legalise
`mm_bfp.cc`'s `G_BUILD_VECTOR <8 x s8>`; 22.0.0.2026092101 compiles it.  See
`KNOWN-ISSUES.md`.

Set before importing `aie`:

    os.environ["PEANO_INSTALL_DIR"] = <the llvm-aie install>/llvm-aie
    os.environ["NPU_CACHE_HOME"]    = <a writable cache dir>

`get_peano_pin.py` and `get_pinned_peano.py` automate the lookup and download.

## 3. XRT and the NPU driver

AMD XRT 2.21.0 (or newer) plus the NPU driver.  `xrt-smi examine` must list
the NPU.  Two tools are needed on `PATH` for aiecc's packaging step:

    xclbinutil        # not on PATH by default; add its directory
    xrt-smi

`pyxrt` comes with XRT and must be importable from the run-side interpreter.

## 4. llama.cpp (the text encoder runs here)

**Not built from this repository and not committed.**  Use an existing build.
The one this project was developed against is a Vulkan build:

    <retired-llama-build>\llama-embedding.exe

any recent llama.cpp with a Vulkan (or CPU) backend and `llama-embedding.exe`
will do.  Get it from https://github.com/ggml-org/llama.cpp and build with
`-DGGML_VULKAN=ON` for the iGPU, or use a prebuilt release.

## 5. The model

Z-Image-Turbo from the Hugging Face hub, in diffusers layout:

    _models/Z-Image-Turbo/
        transformer/      3 safetensors shards   ~23.5 GB
        text_encoder/     3 safetensors shards   ~7.7 GB
        vae/
        scheduler/

`safetensors.get_tensor()` raises an access violation on this machine; read
tensors via the header's byte offsets and `np.frombuffer` (see
`build_bfp16_weights.py`'s `read_checkpoint_tensor`).

## 6. The BlockFP16 weight file

Built from the model, never committed (it is 6.4 GiB of derived data):

    python build_bfp16_weights.py --out zimage-port/weights.bfp16

`pack_bfp16.py` is the encoder and is original work in this repository.  Its
contract is `pack(W, tile_rows=None) -> bytes`, `unpack(blob, N, K, tile_rows)`,
`KNOWN_TILE_ROWS`, `detect_tile_rows`.  There is no `quantize(...)` and no
`unpack_and_quantize(...)`.

## 7. Verifying the build

    python final_audit.py          # the tree is complete
    python api_check.py            # every module's API composes
    python weights_integrity.py    # the weight payload tiles exactly
    python -c "import pack_bfp16; pack_bfp16.selftest()"

## What IS in this repository

* `pack_bfp16.py` and the other scripts — original
* `kernels/rms_norm_weighted.cc`, `kernels/adaln_modulate.cc`,
  `attn_impl/qk_norm.cc` — original C++ kernels for AIE2P
* the IRON designs under `block/`, `memtile2/`, `dit2/` — original
* the numpy references and the oracle — original
* the documents (`README`, `ASSEMBLY`, `PIPELINE`, `STAGE3`, `COVERAGE`,
  `KNOWN-ISSUES`, `RESUME`, `ENVIRONMENT`, `REPRODUCE`) — original

Everything else — mlir-aie, llvm-aie, XRT, llama.cpp, the model, the compiled
xclbins, the vendor `.rai` — is obtained separately and is not distributed
here.
