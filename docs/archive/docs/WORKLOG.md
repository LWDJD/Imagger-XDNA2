# WORKLOG — Z-Image-Turbo → AMD XDNA2 NPU

Durable record so a future session can resume without this conversation.
Append dated entries.  State what was measured, not what was hoped.

Related: `HANDOFF-2026-10-12.md` (project handoff),
`zimage-port/_probe_out/route3/*.json` (raw measurements).

---

## 2026-10-13 — ADDENDUM 3: the deliverable tree is audited and complete

`zimage-bfp16-npu\final_audit.py` checks every document, script, directory and
artifact.  Last run: **9 documents, 12 scripts, 14 directories, all present and
parsing.**  Key artifacts verified:

    weights.bfp16                6,913,965,970 B   506 tensors, 3.554x compression
    vendor_artifacts/*.rai         418,881,856 B   sha256 recorded, irreplaceable
    pipeline/oracle.png             35,958 B       the reference image (128x128)
    block/out/s1w_*.xclbin          21,304 B       stage 1, verified on the NPU

The documents: `README.md` (route, format, milestones), `ASSEMBLY.md` (the
block's stages, the judging budget, every corrected ordering), `PIPELINE.md`
(the loop, the residency numbers, the block contract), `STAGE3.md` (the FFN
spec), `RESUME.md` (how to pick this up cold), `COVERAGE.md` (kernel
inventory), `KNOWN-ISSUES.md` (every trap, every ruled-out path),
`ENVIRONMENT.md`, `REPRODUCE.md`.

Stage 2 (attention) and stage 3 (the FFN branch) are in flight in parallel,
scoped to `step2_*` and `step3_*` respectively so they cannot collide.

## 2026-10-13 — ADDENDUM 2: the host side is complete; only the block is left

**Everything except the NPU block body is done and measured.**

| piece | evidence |
|---|---|
| 34-block stack, forward order | `stack34.py`: 34 calls in 19.4 s (572 ms/call) |
| weight residency at full scale | 506 reads, **0 unpack calls**, 6,913,878,016 B resident |
| reference image | `pipeline/oracle.png`, 128x128, seed 0, 8 steps |
| denoising loop | `FlowMatchEulerDiscreteScheduler` (shift 3.0), the checkpoint's own schedule |
| VAE decode | CPU, `latents / scaling_factor + shift_factor` per `ref_native_128.py:580-582` |
| block contract | `block_adapter.py`: `block(latents, t, embeds)`, patchify 16x16 latent -> 64 tokens |

**Geometry correction that was wrong in earlier notes:** `vae_scale_factor = 8`
(measured: 8x8 latent -> 64x64 image), so a 128x128 image is a 16x16 latent =
64 DiT tokens at a 2x2 patch.

**Stage 1 of the block is verified on the NPU:**
`rms_norm_weighted -> adaln_modulate -> mm_bfp(mixed)` at K=3840, 4/4 runs
`ERT_CMD_STATE_COMPLETED`, **max ulp 1.000**, 99.90% bit-exact, <=2 bf16 ulps
PASS.  Gamma is folded into one packed `[scale|shift|gamma]` stream element
because `core_dma_channels_in == 2`.

**Stage 2 (attention) is the only thing outstanding.**  `_scratch/attn_impl/`
already verifies at the full Z-Image geometry (30 heads / 128 / 96) and is
being integrated into the block.

Two traps that cost rounds and are now settled:
* `set_current_device(NPU2())`, NOT `from_name("npu2")` — the latter exposes
  one shim tile to the placer and refuses multi-fill designs with an error
  that reads like a hardware limit
* `pack_bfp16`'s API is `pack(W, tile_rows=None) -> bytes`; there is no
  `quantize(...)` and no `unpack_and_quantize(...)`

## 2026-10-13 — ADDENDUM: the host-interface budget, and where assembly stands

**New hardware fact, measured and non-negotiable:**

    error: no ShimNOCTile on the device has 0 input/1 output DMA channel(s)
           free: all 1 ShimNOCTile(s) are at 0/2 input, 2/2 output channels used
    note: this is the device's total ShimNOCTile DMA budget, not a placement
          choice; fan traffic through a MemTile, or reduce the number of
          DMA-attached ObjectFIFOs feeding this tile

A compute tile also has `core_dma_channels_in == 2`.  **Every DiT stage must be
planned as at most two host-fed input streams plus one output.**  Per-token
parameters ride inside ONE packed stream element (`sb_row_layout`), never as
their own channel.

**Assembly status:**
* the unmodified `memtile2` base verifies **100.00% bit-exact** at
  M=32 K=3840 N=32, tile 16/8/16, chunked producer — architecture, corner
  turn, segmented producer and host ABI all confirmed at this geometry
* inserting adaLN into that base produced garbage (482/1024 NaN/Inf); since the
  base is exact, the fault is in the inserted ~20 lines
* the task requires `rms_norm_weighted`, not the unweighted `rms_norm` the
  first attempt used; fold gamma into the packed stream element so Worker A
  keeps its two input fifos
* the adaLN stream as a THIRD host input is refused by the SHIM budget; the
  sanctioned fix is to merge A and the params into one stream, or fan through a
  MemTile

**Still to do:** merge the streams; get `rms_norm_weighted -> adaln_modulate ->
mm_bfp(mixed)` verifying within 2 bf16 ulps; then attention; then the FFN
branch; then two blocks chained (residency on silicon); then 34 blocks; then
the loop; then the VAE on CPU for the first image.

## 2026-10-13 — STATE SNAPSHOT: all pieces verified, assembly is the remaining work

For a future session, this is where things stand and what is left.

**Done and measured (each verified on the NPU against numpy):**

| piece | result |
|---|---|
| BFP16 encoder `pack_bfp16.py` | byte-exact vs the vendor `.rai` |
| `weights.bfp16` | 6,913,965,970 B, 506 tensors, 6,143,056,384 params, 3.554x compression, regenerated with the fixed quantiser and verified |
| weight residency | **resident 6.91 GB vs fp32 24.57 GB (3.554x, 71.9% saved); steady-state unpack calls = 0** |
| BFP16 matmul | rel 1.01e-02 = the quantisation floor |
| RMSNorm -> GEMM pipeline | 100.00% bit-exact, hidden to 1792 |
| `rms_norm_weighted` | 96x3840 bf16, worst 1.0 ulp |
| `adaln_modulate` | 96x3840 bf16, bit-identical at all 368,640 elements |
| non-causal attention + QK-norm | full Z-Image geometry (30/128/96), 368,640/368,640 in budget, P*V 100% bit-exact |
| hidden-3840 corner turn + producer | **100.00% bit-exact**, max ulp 0.000, swept to K=24,576 |
| DiT block weights | 15 tensors, byte-identical to the vendor `.rai` payload (416,071,680/416,071,680) |
| oracle | `dit_block_float` / `dit_block_device`; acceptance `rel_to_absmax <= 5.4e-02` |

**The base for assembly is `_scratch/memtile2/memtile_worker.py`**, re-verified
at hidden 3840: compile 10.1 s, 2/2 `ERT_CMD_STATE_COMPLETED`, 100.00%
bit-exact vs the per-call-bf16 model.

**Remaining, in order:**
1. insert `adaln_modulate` (and switch to `rms_norm_weighted`) into the memtile2
   design — `_scratch/block/`, insertion points are `memtile_worker.py` lines
   ~292 (norm kernel), ~305/~309 (`inA`/`outA`), ~371 (in-place rms_norm),
   ~388/~417 (`worker_a`)
2. + RoPE + QK-norm + QK^T + softmax + P*V + to_out  (attention,
   `_scratch/attn_impl/`)
3. + the FFN branch and both residuals
4. two blocks chained — this is where weight residency is proven on silicon
5. 34 blocks + the 8-step flow-matching loop (`PIPELINE.md`)
6. VAE on CPU -> first image

**Corrected orderings, settled from source:**
`to_q/to_k/to_v -> unflatten -> norm_q/norm_k -> RoPE -> attention`; RoPE is
INTERLEAVED (`rope` WITHOUT `two_halves`).  Weight order per block:
`adaLN, to_q, to_k, to_v, to_out, w1, w3, w2` then 6 consts — adaLN FIRST, w3
BEFORE w2 — and a block is not contiguous in the vendor buffer.

**Hardware facts measured and wrong in earlier notes:**
* memtile DMA BD cap is NOT 16383 words (that is a compute-tile limit)
* `delegate_tile` is dead on npu2; a join output can never carry a corner walk
* **AIE2P vector loads silently floor misaligned addresses to 64 B** — wrong
  numbers, no error.  Enforce `contract.alignments`
* `core_dma_channels_in == 2`; `aie::mul` never mixes operand classes
* the scalar bf16 GEMM rounds each product toward -inf unless `conv_even` is
  set first

**Irreplaceable:** `zimage-bfp16-npu\vendor_artifacts\39e41383350b8c53e90996269478cfb7.rai`
(418,881,856 B, sha256 `dce407a3c5c5ffcf9a369c31309e29aa8af182711b814f0cbdb1936dcd412a10`)
is the only surviving vendor artifact.  Keep `zimage-port\_probe_out\fresh_cache`
intact.

---

All six parallel subagent streams landed.  The status is that the DiT block
needs **assembly, not invention**.

**Verified on the NPU, each measured against numpy:**

| piece | result |
|---|---|
| BFP16 matmul | rel 1.01e-02 = the quantisation floor |
| RMSNorm -> GEMM (two-op pipeline) | 100.00% bit-exact, hidden to 1792 |
| `rms_norm_weighted` (new C++) | 96x3840 bf16, worst element 1.0 ulp |
| `adaln_modulate` (new C++) | 96x3840 bf16, **bit-identical** at all 368,640 elements |
| non-causal attention + QK-norm | **full Z-Image geometry** (30 heads / 128 / 96), 368,640/368,640 in budget, P*V 100% bit-exact |
| hidden-3840 corner turn, real producer | **100.00% bit-exact**, max ulp 0.000; swept to K=24,576 |

**The oracle** (`_scratch/oracle/`) gives the acceptance rule for the assembled
block: `rel_to_absmax <= 5.4e-02` against the f64 truth, raised by the softmax
exp2 error (2.94e-2) and optionally the swiglu vtanh error (0.08, removable
with `use_lut=True`).

**The block weights** are extracted and byte-exact: the same pipeline
reproduces the surviving vendor `.rai` payload 416,071,680 / 416,071,680 bytes,
0 differing.  `weights.bfp16` was regenerated with the fixed quantiser
(6,913,965,970 B, 506 tensors, 3.554x compression) and verified.

**Order and naming, corrected from source and now settled:**
`to_q/to_k/to_v -> unflatten -> norm_q/norm_k -> RoPE -> attention`, with RoPE
in the INTERLEAVED convention (`rope` WITHOUT `two_halves`).  Per block the
weights are `adaLN, to_q, to_k, to_v, to_out, w1, w3, w2` then 6 consts -- adaLN
FIRST and w3 BEFORE w2 -- and a block is not contiguous in the vendor buffer.

**Hardware facts that were wrong before and are now measured:**
* the memtile DMA BD cap is NOT 16383 words (that is a compute-tile limit);
  30,720-word BDs run at a memtile
* `delegate_tile` is dead on npu2: a core cannot write a memtile-resident pool
* a fifo cannot join AND feed a forward hop; a join output's walk covers one
  input segment, so a corner turn can never live there
* **AIE2P vector loads silently floor misaligned addresses to 64 B** -- wrong
  numbers, no error.  Enforce `contract.alignments`
* `core_dma_channels_in == 2`; `aie::mul` never mixes operand classes
* the scalar bf16 GEMM rounds each product toward -inf unless `conv_even` is
  set first

Next: assemble the block (stage 4 of `ASSEMBLY.md`), then two blocks chained to
measure weight residency, then 34 blocks, then the loop, then the VAE on CPU for
the first image.

---

Goal armed as `goal-c043d2ea-94bd-473f-bff9-7b9bb1c73c96` (256 rounds):
run Z-Image-Turbo on XDNA2 with the open IRON + XRT stack, all weights
BlockFP16, no fp32 expansion and no per-call overhead.

**Established before this phase — do not re-derive:**
* BFP16 encoder byte-exact; `weights.bfp16` 6.439 GiB single file
* toolchain: mlir_aie 1.4.4 + **llvm-aie 22.0.0.2026092101** (the pin from
  `utils/peano-requirements.txt` at tag v1.4.4; the 21.0.x nightlies fail to
  legalise `mm_bfp.cc`'s `G_BUILD_VECTOR <8 x s8>`)
* BFP16 matmul verified on the NPU at the quantisation floor (rel 1.01e-2)
* RMSNorm -> mm_bfp(mixed) two-op pipeline **100.00% bit-exact**, hidden to
  1792; corner turn expressed as a DMA `to_stream` walk, proven with
  `tap.gather()` before compiling
* kernel coverage 7/8 (`COVERAGE.md`); adaLN is the only missing symbol
* designs complete with file:line evidence for non-causal softmax
  (`attn/DESIGN.md`), adaLN and weighted RMSNorm (`adaln/DESIGN.md`)

**Running in parallel**, each strictly scoped to its own `_scratch/<topic>/`:
* `86f27262` `_scratch/kernels/` — implement `rms_norm_weighted` and
  `adaln_modulate` as C++ kernels + IRON wrappers + NPU verification
* `2bae3c2e` `_scratch/attn_impl/` — bind `softmax_rows_bf16` (the unmasked
  variant already present behind `softmax_aie2p.h:237`) and build non-causal
  attention QK^T -> softmax -> P*V
* `a519eeb6` `_scratch/memtile/` — reach hidden 3840 via the documented memtile
  path (512 KB tile, corner turn on the memtile hop, asymmetric transfer
  granularity so neither core holds `(m,K)`)

**Then:** assemble the full DiT block (the 9-step sequence in
`_scratch/adaln/RECOMMENDATION.md`), 34 blocks, the VAE decoder, and the
8-step denoising loop.

**Traps for whoever continues** (all measured, all in `KNOWN-ISSUES.md`):
* `mm_bfp_mixed.cc` re-rounds C to bf16 after EVERY GEMM call — numpy
  references must round-trip through bf16 per accumulation, or K=1792 drifts
  13 ulps and looks like a bug
* compute-tile DMA BDs accept at most 3 dimensions; memtiles accept 4
* bfp16ebs8 strides must be 4-byte multiples (9-byte blocks need repacking)
* L1 = 64 KB per compute tile (4x16 KB); memtile = 512 KB; a single DMA BD caps
  at 16383 32-bit words — that is what refuses hidden 2048/3840
* `os.mkdir(path, 0o700)` makes a directory the sandbox token cannot re-enter
  and `tempfile` uses exactly that mode, breaking `prefix_symbols_in_object`.
  Proven workaround: a process-local `os.mkdir` shim widening to 0o777 plus
  `use_cache=False`.  The ACL is NOT the cause
  (`diagnose-windows-sandbox-acl` returned NOT_THIS_CLASS)
* `xclbinutil` is off-PATH; add
  `<xrt>`

---

`xrt-smi examine` (from `<xrt-parent>\
npu_mcdm_stack_prod\xrt-smi.exe`) reports:

    XRT Version          : 2.21.0   (hash 15e6319be8de1e76a6150111a3861729)
    NPU Driver Version   : 32.0.20102.3930
    NPU Firmware Version : 1.1.2.64
    Device(s) Present
      |[0064:00:01.1]|NPU Strix|

Machine: AMD Ryzen AI 9 H 365 w/ Radeon 880M, 20 cores, 32 GB.

**This ends the "there is no Windows NPU runtime" conclusion.**  XRT is present
and functional; it simply was not on PATH and `aie.doctor`'s "pyxrt is not
importable" was a *binding version* problem, not a missing runtime.

Where the pieces live:

| file | path |
|---|---|
| `xrt_coreutil.dll` | `C:\Windows\System32\`, `C:\Program Files\RyzenAI\1.6.0\xrt\`, `C:\AMD\AMD-Software-Installer\Packages\Drivers\AMDNPUMCDM\WT64A\` |
| `xrt_core.dll` | `...\AMDNPUMCDM\WT64A\`, `C:\Windows\System32\DriverStore\FileRepository\kipudrv.inf_amd64_*\` |
| `pyxrt.pyd` | `C:\Windows\System32\AMD\`, `...\AMDNPUMCDM\WT64A\`, `test7\WStmp\driver376\npu_mcdm_stack_prod\` |
| `xrt-smi.exe` | `test7\WStmp\driver376\npu_mcdm_stack_prod\xrt-smi.exe` (2,292,072 B) |
| `xrt_smi_strx.a` | `...\AMDNPUMCDM\WT64A\Runner\` (36 MB) |

**Only blocker for the Python API:** `pyxrt.pyd` imports `python310.dll` while
the working interpreter is Python 3.12.  `xrt_coreutil.dll` itself imports only
system DLLs, so the C++ API is usable from ctypes or from a 3.10 interpreter.
No Python 3.10 interpreter was found on the machine (only embedded
`python310.dll` copies inside an IDE / JetBrains / GAIA).

### What this unblocks

The open source route is now complete end to end on this machine:

    mlir_aie 1.4.4 (aiecc / aie-opt / aie-translate / bootgen)   [win_amd64 wheel]
    llvm-aie 1.8.0 (Peano ISA backend)                          [bundled]
    aie_kernels/linalg/mm_bfp.cc  +  aie_api/.../mmul_bfp16_bfp16.hpp   [BFP16 GEMM]
    IRON Python API, NPU2 device class                           [imports]
    XRT 2.21.0 + NPU Strix                                       [device visible]

BFP16 weights can therefore be fed to the hardware directly, without the
VitisAI EP and without its fp32 conversion -- which is exactly the goal
(BFP16 storage + BFP16 residency + no runtime conversion overhead).

---

## 2026-10-13 — hand-written operators live in `onnxruntime_vitis_ai_custom_ops.dll`

PE strings of the 2.7 MB DLL reveal the operator sources:

    G:\xmc\src\voe\vaip\onnxruntime_vitis_ai_custom_ops\src\main.cpp
      custom_op_mha.cpp   custom_op_gqa.cpp   custom_op_gqo.cpp
      custom_op_sslrn.cpp custom_op_ssmlp.cpp custom_op_prefill_gqa.cpp
      custom_op_rope.cpp  custom_op_mm.cpp

Runtime strings: `running AIE kernel`, `bf16KernelOutputAlloc`,
`ORTKernelInputBf16ToFloat32`, `ORTKernelAllocInput`, `ORTBuiltInKernelCompute`,
`DEBUG_CUSTOM_OP_REGISTRATION`, `Failed to register custom ops: `.

It imports `xrt_coreutil.dll` and `dyn_dispatch_core.dll`.  So **AMD's own NPU
operators are ONNX custom ops that run AIE kernels through XRT** -- the same
shape as the open route.

---

## 2026-10-13 — `.rai` integrity: settled

| fact | evidence |
|---|---|
| trailer = `XCLBIN_MIRROR_DATA_END\0` + flatbuffer vtable + 2 length-prefixed 32-hex strings | hexdump |
| **trailer digest #1 == `config.onnxMd5File` == md5(ModelProto)** | `route3_link.py`, 5 confirmations |
| trailer digest #2 == `original-info-signature.txt`, constant `84963b6ab78b5e82df87bf513e35a22a` | `route3_signatures.py` |
| `config.cacheKey` == `config.onnxMd5A` == the `.rai` file name | `route3_link.py` |
| **`onnxMd5A` and `onnxMd5B` are STRUCTURAL — unchanged when weight values change** | `route3_diff.py` on two same-graph different-weight models |
| `config.mepTable[296]` = checksums of prebuilt PSA/PSA2/PSA3 kernel variants | `route3_meptable.py` |
| **one-bit flips in the weight payload are ACCEPTED** (4 offsets: GEMM body, adaLN prefix, adaLN weights, const region) | `route3_bit.py`, on a CLEAN baseline |
| the payload IS the weights: compiled payload -> rel 2.116e-02 vs torch reference; noise payload -> run-time crash | `route3_payload_live.py`, `route3_single.py` |
| `buffer1` of `external_buffers` = the weight buffer, 28 sub-buffers named `compute_graph.Layer_*_wts_ddr`, `offset_in_bytes` 0..415,334,400 | `route3_extbuf.py`, `route3_buf1.py` |

**CAUTION — earlier "byte-identical rebuild" claims are VOID.**  `route3_findbase.py`
compared the rebuild against a `.rai` that had already been overwritten with the
same bytes, so its 0-diff result was trivial.  The honest comparison
(`route3_verify_clean.py`, against `route3/CLEAN_baseline.rai`) gives
**410,902,718 / 416,071,680 differing bytes (98.76%)**.  The rebuild does NOT
yet match the compiler's output; the sub-buffer ordering must be read from
`buffer1`'s declared `offset_in_bytes` instead of the assumed `REGION_ORDER`.

Also: a contaminated baseline (`..._pristine.rai` saved from an already patched
copy) made every "patched .rai is rejected" conclusion wrong for a while.  Always
run `route3_bit.py --offset none` first.

---

## 2026-10-13 — BFP16 format, settled

`pack_bfp16.py` reproduces the vendor encoding; verified byte for byte on 12
synthetic shapes.

    group of 8 along K  ->  9 bytes = [exponent byte][8 x int8 mantissa]
    e = floor(log2(amax)) - 6        UNCONDITIONALLY
        (the earlier -7 branch for exact powers of two is wrong; it hit ~0.4%
         of real-weight groups)
    mantissa = sign(v) * floor(|v| / 2^e + 0.5)     round half AWAY from zero
    if m.max() > 127: e += 1; recompute             -128 legal, +128 not
    exponent byte = (e + 133) & 0xFF

Tiling: `tile_rows` rows x 32 K-elements per tile; inside a tile
`slot = (r//8)*32 + (c//8)*8 + r%8`, `r = row % tile_rows`, `c = kelem % 32`.
`tile_rows` is f(N,K,M).  At seq=96: 48 / 32 / 32 / 64 for the four shapes.

`block_size = 8` confirmed by four independent AMD sources
(`quark/torch/.../config.py` `group_size=8`; `hw_emulation_interface.py`
`block_size = 8`; `aie.utils.bfp` `BLOCK=8, BLOCK_BYTES=9`; Quark's
`BFPAttributes: block_size: 8`).  The Zhihu article's "32 elements share an
exponent" is wrong for `bfp16ebs8` -- it confuses the q4nx group of 32.

---

## 2026-10-13 — disk pressure

D: fell to 1.25 GiB and the ONNX export began failing with `OSError: Errno 28`,
producing a 0-byte `model.onnx` that then fails every compile with "ModelProto
does not have a graph".  **A 0-byte model.onnx means the disk is full -- check
free space first.**

`_models` exists in three trees (30.637 GiB each) and they are verified
identical AND already hard-linked, so they cost one copy.  8.88 GiB of
regenerable artefacts (ONNX exports, `dd_iter` scratch, Quark's BFP16 ONNX,
VAIP/route3 caches) was deleted; D: free 1.26 -> 10.14 GiB.

---

## 2026-10-13 — open-source toolchain assessment (corrected)

Earlier this was written as "Windows has no open runtime".  **Wrong** -- see the
XRT entry above.  What is true:

* mlir-aie ships official `win_amd64` wheels (`mlir_aie-1.4.4-cp312`);
  `aie.doctor` reports `toolchain: mlir_aie 1.4.4, llvm-aie 1.8.0` ok.
* XRT's own docs (`system_requirements.html`) say "Host Platform for AMD NPU:
  **Windows or Linux**"; the install/build pages are for PCIe cards only.
* The RyzenAI NuGet (`RyzenAI_Deployment.1.8.0`) is deployment DLLs only -- no
  headers, no import libs.
* `amd/RyzenAI-SW` is examples only.
* Ryzen AI docs list supported quantization types as **BF16, A16W8, A8W8,
  XINT8**; BFP16 appears only as `MatMulNBits (W4ABF16/W4ABFP16)` (4-bit weights
  with BFP16 activations).  So BFP16 *weights* are not a documented
  developer-facing format.

---

## 2026-10-13 — OPEN IRON + XRT ROUTE IS COMPLETE ON THIS MACHINE

`pyxrt` works under a Python 3.10 interpreter that was on the machine all along:

    <python310>\python.exe        (3.10.9)
    <user>\AppData\Roaming\an IDE\ModularData\ai-agent\vm\tools\python\python.exe

Earlier searches missed them because the paths contain no "310"/"3.10" text.
Verified:

    >>> import pyxrt; pyxrt.device(0)
    <pyxrt.device object>

with `<xrt>` on
`sys.path` / `os.add_dll_directory`.  That folder holds `pyxrt.pyd` (533,864 B),
`xrt_core.dll`, `xrt_coreutil.dll`, `xrt-smi.exe`.

`pyxrt` exports: `device`, `bo`, `kernel`, `module`, `program`, `run`,
`runlist`, `hw_context`, `elf`, `xclbin`, `uuid`, `enumerate_devices`, ...

**Full stack now available:**

| component | python | status |
|---|---|---|
| mlir_aie 1.4.4 (aiecc / aie-opt / aie-translate / bootgen) | 3.12 | ok |
| llvm-aie 1.8.0 (Peano) | 3.12 | ok |
| `aie_kernels/linalg/mm_bfp.cc`, `aie_api/.../mmul_bfp16_bfp16.hpp` | — | BFP16 GEMM source |
| IRON (`aie.iron`, `NPU2`) | 3.12 | imports |
| XRT 2.21.0, `xrt-smi` sees `NPU Strix` | — | ok |
| `pyxrt` + `pyxrt.device(0)` | **3.10** | ok |

**The version split is natural, not a problem:**

    compile side : Python 3.12 + mlir_aie  ->  aiecc  ->  xclbin + insts.bin
    run side     : Python 3.10 + pyxrt     ->  device(0) / bo / kernel / run

Next: a Python 3.10 loader that takes `xclbin` + `insts.bin` and feeds BFP16
weights as `pyxrt.bo` buffers straight into the NPU.  That satisfies all three
requirements at once -- BFP16 storage (done), BFP16 residency (bo holds the
packed bytes, no fp32 expansion), and no per-call unpack/convert overhead.

---

## 2026-10-13 — first open IRON BFP16 program: MLIR generation works

`zimage-port/bfp16_gemm_iron.py` builds a BFP16 matmul through the open IRON
API.  Iterated through the API and now the design resolves to MLIR and reaches
verifier.  What is established:

* `mm_bfp(dim_m, dim_k, dim_n, mixed=False)` needs the **current device** set
  first: `aie.iron.set_current_device(NPU2())`.  Otherwise
  `NotImplementedError: mm_bfp: bfp16ebs8 is an AIE2P type`.
* `numpy` reference `mm_bfp_ref` vs float64 matmul: rel 1.79e-2 at 64x64x64 --
  BFP16 precision level.  `aie.utils.bfp`: `BLOCK=8`, `BLOCK_BYTES=9`.
* Worker construction (from `iron/algorithms/_pipeline.py`):
  `Worker(core_fn, [f.cons() for f,_ in inputs] + [f.prod() for f in outputs] + list(constants))`
  where `core_fn` is a closure that acquires/releases and calls the kernel.
  The **Kernel must be a Worker argument** or "Kernel must be resolved before it
  can be called".
* `Stage(body, inputs=[(fifo, count), ...], outputs=[fifo, ...], constants=[...])`
  -- note `inputs` are pairs, `outputs` are bare fifos.
* `pipeline(stages, host_types, transfers, ...)` from
  `aie.iron.algorithms.kernel_design`; `transfers` are
  `(fifo, "fill"|"drain", host_index)` triples.
* **Operand type must be the block-float type**: the kernel wants
  `memref<512x!aiex.bfp<"v8bfp16ebs8">>` for a 64x64 operand (512 = 64*64/8
  blocks, each block 9 bytes).  So the annotation is
  `np.ndarray[(M, K//8), v8bfp16ebs8]`, NOT `uint8`.

**Current blocker:** with that annotation, `program.py:200` raises
`IndexError: tuple index out of range`.  Next step is to look at
`_pylibs/mlir_aie/python/aie/iron/program.py` around line 200 and at
`aie/iron/program.py::device_body` to see what shape/dtype form it expects;
`aie.helpers.npdtypes` documents `v8bfp16ebs8` as "a marker for the type an
ndarray annotation carries", so the annotation form is the thing to fix.

---

`_pylibs/quark/onnx/operators/custom_ops/src/bfp/cpu/bfp_kernel.cc`:

    shared_exp   = max over the block of (bits & 0x7f800000) >> 23
    shared_exp_value = shared_exp - 127
    m_bits       = bit_width - 9            # 16 -> 7
    scale        = 2 ** (shared_exp_value - (m_bits - 1))   == 2**(e_unb - 6)
    max_v        = 2 ** (shared_exp_value + 1) - scale

    enum rounding_mode_enum { STD_ROUND, DPU_ROUND, PY3_ROUND };
    dpu_round(x) = ((x<0) && (x-floor(x)==0.5)) ? ceil(x) : round(x)
    py3_round(x) = ties-to-even        # == mlir-aie's conv_even

**This confirms `pack_bfp16` on both disputed points**: the exponent rule is
`e_unb - 6` UNCONDITIONALLY (no power-of-two branch), and the vendor rounding is
`STD_ROUND` = `std::round` = half away from zero, matching the measured
`-50.5 -> -51`.  `PY3_ROUND`/`conv_even` is a different mode the toolchain does
not use for these weights.

Custom-op pattern (Quark's `custom_op_library.cc`):
`Ort::CustomOpBase<...>` with `CreateKernel`/`GetName`/`GetVersion`/
`GetExecutionProviderType`; domain `com.amd.quark`; ops
`BFPQuantizeDequantize`, `MXQuantizeDequantize`, `ExtendedQuantizeLinear`, ...
That one targets CPU/ROCM/CUDA.  The NPU counterpart is
`onnxruntime_vitis_ai_custom_ops.dll` (`custom_op_mm.cpp` etc., which links
`xrt_coreutil.dll` and "run AIE kernel").

---
