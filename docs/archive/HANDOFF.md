# HANDOFF — Z-Image-Turbo on AMD XDNA2 NPU

**Read this first.  It is the complete state.  Everything else is detail.**

Written for a fresh session (PTC mode).  The previous session ended after the
throughput architecture was measured and before the whole-chip utilisation
analysis landed.

---

## 0. THE GOAL, VERBATIM

> Z-Image-Turbo on the AMD XDNA2 NPU via the **open IRON + XRT toolchain**,
> all weights in **BlockFP16 (`bfp16ebs8`)**, resident, with **no runtime
> unpack / convert / reload overhead**, producing **images**.

Owner's three acceptance criteria, all **measured and met**:
1. BFP16 storage — `weights.bfp16` 6,913,965,970 B, 506 tensors,
   6,143,056,384 params, **3.554x** compression, 1.125 B/param
2. BFP16 residency — **0 unpack calls** in steady state; on silicon, weight BOs
   written once and byte-identical after **272 dispatches**
3. It generates images — `image_npu.png`, 128x128, **8 dispatches**
   (8 steps x 1 block/step, 8/8 COMPLETED).  The separate 34-block run is
   `image_b34_npu.png` at **272 dispatches** (8 steps x 34 blocks/step, mean
   326.1 ms, loop 250.6 s) — `IMAGE.md` §7.  **These are two experiments; do
   not conflate them.**

Working directory: `D:\Project\deepseek_harness\public_work`
Deliverable tree: `zimage-bfp16-npu/` (mirrored to `zimage-port/`)

---

## 1. THE ONE NUMBER THAT DECIDES EVERYTHING

```
PEAK      512 MAC/instr x 2 x 1.5 GHz x 32 tiles = 49.152 TOPS  (BFP16)
NOW       187.64 GFLOP/s = 0.38% of peak
CLOSED    a closed-source route runs this model in ~8 s on the same silicon,
          reportedly at 512 or 128 resolution and NOT saturating the device
```

Owner confirms the device's BFP16 rate is **>= 45 TOPS**.  The 49.152
derivation matches.  **Do not use 25 TOPS — that figure is wrong and was
corrected.**

Model is **Z-Image-Turbo: 6,143,056,384 params (6.1B), 34 blocks**, hidden
3840, heads 30, head_dim 128, ffn 10240, seq 96 = 32 context + 64 image.
**Not** Qwen-Image-2.1 (7.1B / 32 blocks) — that was a separate survey and the
two were briefly conflated.  Per-block ~34 GFLOP at 96 tokens.

Estimated wall at 1024^2 / 8 steps: ~395 TFLOP / 49 TOPS = **8.1 s at 100%**.
Owner's "8 seconds" is consistent with 512 or 128 resolution.

---

## 2. THE 500x GAP IS FULLY DECOMPOSED

```
8x   (30 of 32 compute tiles idle)   x
25x  (per-call overhead, NOW FIXED)  x
2.5x (residual)
= ~500x
```

* **8x idle tiles**: `Device(AIEDevice.npu2).cols == 8`, `core_rows == [2,3,4,5]`
  = 32 compute tiles.  But `device_from_args` resolves `n_cols="auto"` to a
  SINGLE COLUMN (its own docstring calls that *"the safest 'I haven't thought
  about this yet' choice"*, `_pylibs/mlir_aie/python/aie/utils/hostruntime/
  argparse.py:298-304`).  The design places exactly `Tile(0,2)`/`Tile(0,3)`.
  **Reaching the other 30 tiles is a NEW DESIGN, not a parameter change.**
* **25x per-call overhead — FIXED**: it was the norm core's scalar chunk copy
  loop (`te_npu_chunked_design.py:384-385`), a bitwise copy of a contiguous
  1280-byte region written as `2*Kc` scalar ops, 128 iterations per GEMM call
  at any `n`.  Replaced with one 512-bit-vector kernel call (`perfc_copy.cc`).
  **Measured 2.0017 -> 1.2367 us = 1.62x at the same point; 187.64 GFLOP/s with
  the retile = 2.82x.**  100.00% bit-exact, max abs error 0.0, 7/7 COMPLETED,
  byte-identical.

---

## 3. WHAT IS DONE AND VERIFIED (do not redo)

| item | evidence |
|---|---|
| BFP16 encoder, byte-exact vs the vendor `.rai` | 416,071,680/416,071,680 bytes, 0 diff |
| `weights.bfp16` 6.91 GB, 506 tensors | `weights_integrity.py`: payload tiles exactly, 0 gaps/overlaps |
| **weight residency** | `STEP5.md`: 272 dispatches, weight BOs written once, byte-identical after |
| stage 1 `rms_norm_weighted -> adaln_modulate -> mm_bfp(mixed)` | `STEP1.md`, 2/2, **max ulp 1.000** |
| stage 3 FFN + residual | `STEP3.md`, 2/2, `rel_to_absmax` 5.3489e-02 |
| **assembled DiT block** | `STEP4.md`, **5.0760e-02 <= 5.4e-02 PASS** vs the CORRECTED truth |
| VAE decoder, every arithmetic op has an NPU path | `vae/VAEG.md`, PSNR 37.32 dB |
| text encoder on GPU via llama.cpp | `textenc/FINDINGS.md`, cosine 0.99999 |
| host loop + scheduler + VAE | `pipeline/`, `oracle96.png` byte-reproducible |
| first NPU image | `image/IMAGE.md`, `image_npu.png`, PSNR 39.17 dB |
| throughput architecture | `perf/PERF.md`, 14 device configs |
| per-call fix | `perfc/PERFC.md`, 2.82x |

---

## 4. WHAT IS NOT DONE

1. **Whole-chip utilisation** — `58dab9ed` was analysing this when the session
   ended.  `zimage-bfp16-npu/_scratch/util/UTIL.md` may be partial or absent.
2. **8-column replication** (the 8x) — a new design, not a knob.
3. **Operator fusion** — norm+GEMM, adaLN+GEMM, residual+C write, SwiGLU
   triple, QK^T+softmax+P*V.  Each removes one pass over activation bytes.
4. **Real-geometry attention on the NPU** — `block/STEP6.md`: designed, audit
   clean, but no dispatch completes (see the data-arg limit below) and the
   scalar `mm` is ~123 s against a 7 s ERT timeout.  **Needs the vectorised
   GEMM family.**
5. **Text encoder on the NPU with BFP16** — weights packed and verified
   (3.807 GiB) but the device path has an open view-address question.
   Not a blocker: the GPU path works.
6. **34 blocks x 8 steps with REAL weights** — the current 34x8 image uses
   synthetic weights whose trajectory amplifies (PSNR 4.95 dB); the comparable
   result is the 1-block image at 39.17 dB.

---

## 5. HARD CONSTRAINTS — every one cost time

These are measured.  Violating any of them wastes a build or silently corrupts
numbers.

> **2026-10-20 note**: every constraint below now has its provenance
> (report:line / commit) and its **design impact** collected in
> `docs/KNOWN-PLATFORM-FACTS.md` (cards F-05…F-09, F-16).  Add or correct a
> constraint there first, then reference it from here.

### Toolchain / codegen
* **`llc -O1 --regalloc=greedy`, injected via `aiecc --cut="objects_{0}.o"
  --checkpoint` -> swap `objects_main_core_0_2.o` -> `--resume`.**
  See `_scratch/block/step2d_build_win.py`, `step4_build_win.py`.
* **A plain `aiecc -O0` build COMPLETES 2/2 and is WRONG and NOT
  byte-identical.**  The worst kind of failure: it looks green.
  **Acceptance triple, stated precisely:**
  1. **ACQ-clobber gate clean** — `step2d_audit.py` on the ELF reports
     `6 acquires, 0 ACQ CLOBBER, 1 ARG exposure` and PASS.  The ARG count is
     **informational** (the script's own docstring), so "audit clean" does NOT
     mean zero exposures.  The **ORDER gate currently FAILS (7 divergences)**
     because `check_order`'s IR-event extractor returns an empty list — a known
     tool defect recorded in `KNOWN-ISSUES.md`; it is NOT an acceptance leg
     until `check_order` is fixed.
     > **2026-10-20 (t22_factcards2)**: extractor fixed (5 defects; IR 36 events
     > vs ASM 36, multiset identical) and the ORDER gate **retired** as an
     > acceptance leg — the linear-sequence comparison is invalid (8 differences
     > with an identical multiset = basic-block layout).  Keep the multiset
     > invariant; the per-lock CFG check is a separate 0.5-1 person-day unit.
     > `docs/T22-QUICKFIX-REPORT.md` §2.
  2. **2/2 `ERT_CMD_STATE_COMPLETED`**
  3. **run-to-run byte-identical**
  Never trust completion alone.  `step2d_audit.py` has two modes: ELF-only
  (ACQ + ARG) and `--ll peano-linked_main_core_0_2.ll` (adds B1/B2 and ORDER).
  The `.ll` ships WITH the build, next to the ELF in the same `.prj` directory —
  no extra step is needed to obtain it.
* **AIE2P `jl`/`j` have DELAY SLOTS** — printed-after argument setups execute
  BEFORE the callee.  Disassembly order is NOT execution order.
* Per-core `program (RX)` is **16,384 B and it is HARDWARE** (generated ld
  script `LENGTH=0x4000`, aiecc's own "covers ALL of the program memory", the
  hardware register DB, `regdb.py`).  `-O2` overflows by 20992 B, `-O1` by
  17040 B.  **Only `-O0` fits in aiecc's own flow — which is why the
  `llc -O1` injection exists.**  Direct `llc -O1` core .text is 3,024 B vs
  `-O0`'s 4,816 B; the "-O1 wall" is aiecc's IR-level `opt -O1` blow-up.
* `llc -O0` HARD-REFUSES non-fast allocators:
  `"Must use fast (default) register allocator for unoptimized regalloc"`.

### Data movement
* **Declared data arguments <= 7.**  10/9/8 all `ERT_CMD_STATE_ABORT`; 7
  completes.  Largest previously proven was 6.  **Concatenate same-shape
  streams into one buffer (`np.concatenate`) rather than adding arguments.**
* **A ShimNOCTile has 4 DMA channels (2 in / 2 out)** and that is the DEVICE
  TOTAL, not a placement choice.  Per-token / per-channel parameters MUST ride
  a packed stream element, not a second input fifo.  This is why `[x | gamma]`
  is packed into one element.
* **`wait=True` on EVERY fill.**  `TaskGroup.finish()` frees unwaited transfers
  and `dma_free_task` does NOT wait -> hang.  Push the drain FIRST into its own
  group awaited LAST.
* **`memref.view` needs a 1-D i8 source.**  A malformed one aborts the tracing
  context mid-region and **masquerades as `Unbalanced Location enter/exit`**.
  On that error, check every `memref.view` source type FIRST.
* **The object's declared type and the walk's tensor shape must be the SAME
  quantity** — `_check_walk_size` requires `sprod(tensor_dims) == transfer`.
  A row-major pitch with gamma gaps therefore CANNOT be walked.
* A non-outermost zero-stride fill dim expands to **one BD per repetition**;
  the shim BD budget is **16**.
* Compute-tile DMA BD: <= 3 dimensions, <= 16,383 words.  `m*Kc*2 <= 30,720 B`.
  `Kc` must DIVIDE `K` (2560 -> 640, 4096 -> 512, 9728 -> 608).
* AIE2P vector loads **silently floor misaligned addresses to 64 B**.

### Device / runtime
* `set_current_device(NPU2())`, NOT `from_name("npu2")` (which is `npu2_1col`
  and exposes one shim tile, refusing multi-fill designs).
* **Registering a SECOND xclbin in one process is a native `0xC0000005`.  One
  image per process.**  A switch-only process can die at teardown AFTER writing
  its results — **the artifact is the measurement, not the exit code.**
* ABI: kernel `MLIR_AIE`, `k(3, bo_instr, ninstr_words, *data_bos)`;
  `bo_instr` = `pyxrt.bo.flags.cacheable` @ `group_id(1)`;
  data = `host_only` @ `group_id(3+)`;
  `dev.register_xclbin(x)` BEFORE `pyxrt.kernel(dev, uuid, name)`.
  Use the `pyxrt.bo.flags.*` ENUM, not `XCL_BO_FLAGS_*` ints.
* IRON: `spec = worker.specialize(...)`, then `spec.compile(xclbin_path=,
  inst_path=)`.  NOT `worker(...)`.
* `--dynamic-objFifos` sizes a fifo's pool from each `acquire`'s ELEMENT COUNT,
  overriding the declared depth.  **Use ONE `acquire(n)`, not n of
  `acquire(1)`** — the latter gives a 2-object pool and deadlocks on the third.
* **`Kc` must be solved against `2*m*(2*Kc)*2` when `[x | gamma]` is packed**,
  not `m*Kc*2` — the packed element doubles the A-side buffer.

### Numerics
* **RMSNorm is scale-invariant**: fold `1/sqrt(head_dim)` into the Q weight
  ONLY.  Folding into both Q and K is a NO-OP and leaves S `1/scale` too large.
  Measured cost 1.5852e-01 vs 5.0760e-02.  Diagnostic: `q_rope` absmax 6.25 vs
  the model's 0.5508 = sqrt(128).
* QK-norm comes AFTER the projections and BEFORE RoPE.
* **RoPE conventions are OPPOSITE between models**: Z-Image DiT is INTERLEAVED
  (call `rope` WITHOUT `two_halves`); Qwen3 is `two_halves`/rotate-half.
  Getting it backwards still produces plausible output.
* Z-Image RoPE is three-axis: `axes_dims=[32,48,48]`,
  `axes_lens=[1536,512,512]`, `cos/sin [1,96,64]`, zero-centred.
* `mm_bfp_mixed` re-rounds C to bf16 after EVERY call — references must model it.
* `mm_bfp_mixed.cc` static_asserts `m%16==0` and `n%16==0`.
* The device consumes `memtile2/run.py`'s `pack_b_blocks` layout, NOT
  `pack_bfp16.pack` — verified at production shape, 560x better.
* `oracle.make_weights` produces NO `attention.norm_q/k.weight` — a synthetic
  block silently runs `w=1` (5.87e-02 instead of 5.08e-02) and looks like an
  assembly defect.

### The truth reference
* **`step4_oracle.py` is the corrected truth.**  `dit_block.py`'s
  `dit_block_float` has NO QK-norm and uses placeholder position ids
  (`ids[t]=(t,t,t)`); the real model has both.  Uncorrected vs corrected differ
  by 2.1737e-01.

---

## 6. METHOD RULES — learned expensively

1. **Read the interface before planning the change.**  Three separate reverts
   came from discovering a constraint at trace time that one `grep` would have
   shown.
2. **`mean`/`max` carry no positional information** — report `median` and the
   concentration alongside.  A banded or two-population defect is invisible to
   mean and max.  Concentration (e.g. "83.02% of elements carry 99.95% of the
   error") is the most informative single summary.
3. **Go to bin width 1 before calling anything a stripe** — binning created a
   false "striped" structure.
4. **A split must change ONE thing.**
5. **Force a fresh tag before trusting any before/after** — a cached xclbin
   masquerades as "the change did nothing".
6. **A diagnostic switch in the tree IS part of the artefact — assert it,
   never default it.**  A bisection cannot outrun a stale global.
7. **Verify the anchor is unique before editing a shared file**, and use the
   literal edit tool against a re-read region, not regex over the whole file.
8. **Verify packing and layout against REAL weights at the production shape.**
   Both wrong answers on the text-encoder branch came from synthetic
   small-integer test data where the alternatives cannot separate.
9. **Build the instrument that produces a MATCHABLE number, not merely a
   smaller one.**  A ratio matching a known constant identifies the bug; a
   generic "wrong" does not.
10. **A strong measurement finds ONE cause, not THE cause** — twice this
    produced an over-claim.
11. **The all-ones probe** is the transferable artefact: the only measurement
    whose correct answer is a known constant independent of the reference.
    It found a scalar defect in one run after eight layout hypotheses failed.
12. **Report measurements, not hopes.  A negative result is a result.**

---

## 7. THE NEXT WORK, IN ORDER

Owner's direction: **maximise whole-chip utilisation, do not lock the
direction, use the on-chip cache well.**  Owner also notes that at small
shapes the scheduling overhead dominates, so 512 and 1024 can take similar
wall time at different utilisation — quantify the fixed-overhead fraction.

Ranked by TOPS gained per unit of work:

1. **Instrument and fix the remaining per-call path** (done: 2.82x; the design
   is now DMA-throughput-bound at ~5.0 GB/s).
2. **Replicate the 2-worker pair across all 8 columns** (the 8x).  L1 budget is
   per compute tile and does not change.  Ask: what breaks first — the shim's
   2-in/2-out, the memtile, or the instruction memory?
3. **Operator fusion** — norm+GEMM, adaLN+GEMM, residual+C write, SwiGLU
   triple, QK^T+softmax+P*V.  Each removes one pass over activation bytes and
   the design is byte-bound.
4. **On-chip cache residency across time** — what stays in L1 across calls,
   across the K sweep, across the 8 denoising steps (weights do not change
   between steps)?  64 KB per compute tile, 512 KB per memtile.
5. **Batching** — weights read once, compute scales by N.
6. **Arithmetic intensity** — B bytes fall as 1/m; `m=64` is refused by aiecc.
7. **Pure-bfp16 MMA** (`mmul_bfp16_bfp16<8,8,16,...>`) — 2x the MMA floor at
   the cost of bf16 activation numerics (already priced in
   `textenc/TE_NPU.md`).

---

## 8. THE ENVIRONMENT

* Compile / encode Python 3.12:
  `D:\Users\15081\miniforge3\envs\ryzen-ai-1.8.0\python.exe` (**NOT `py -3.12`**,
  no ml_dtypes there)
* Run (pyxrt) Python 3.10:
  `D:\Users\15081\AppData\Local\GAIA\python\python.exe` (`pyxrt.pyd` links
  `python310.dll`)
* Before importing `aie`:
  `PEANO_INSTALL_DIR = D:\Project\deepseek_harness\public_work\_pylibs\llvm-aie\llvm-aie`
  `NPU_CACHE_HOME` = a FRESH dir per attempt (`os.mkdir(path, 0o700)` poisons
  the object cache; use the shim from `_scratch/dit_block_v1_run.py`)
* `PATH` += `D:\Project\deepseek_harness\test7\WStmp\driver376\npu_mcdm_stack_prod`
  (xclbinutil, xrt-smi, pyxrt)
* Toolchain: `mlir_aie==1.4.4`, `llvm-aie==22.0.0.2026092101+0006955e`
  (the pin in the mlir-aie repo's `utils/peano-requirements.txt` at tag v1.4.4).
  Other Peano versions fail to legalise `mm_bfp.cc`.
* `kernels/refs.py` and `attn_impl/refs.py` share a module name — put `kernels`
  LAST in `sys.path` insertion.

**No downloads.  No contacting AMD/Xilinx.**  Keep third-party binaries and
model weights OUT of the git repo.

---

## 9. THE REPO

`zimage-bfp16-npu/` is a git repo.  MIT `LICENSE` + `NOTICE` reproducing the
AMD/Xilinx MIT notice for the three derived kernels + SPDX headers.

**Compliance rule: only original work is committed.**  No third-party source,
no compiled binaries, no vendor artifacts, no model weights.  `.gitignore`
excludes `*.xclbin *.bin *.bfp16 *.safetensors *.gguf *.onnx *.pt *.pth
*.so *.dll *.exe *.rai` and more.  `BUILD.md` documents how to obtain
everything.

Key documents:
* `README.md` — route, scope rule, batching, compute-per-byte
* `ASSEMBLY.md` — the block's stages and every corrected ordering
* `PIPELINE.md` — the loop, residency, geometry (96 = 32 context + 64 image)
* `KNOWN-ISSUES.md` — every trap, every ruled-out path
* `RESUME.md` — how to pick this up cold
* `BUILD.md` — how to obtain every dependency
* `ENVIRONMENT.md`, `REPRODUCE.md`, `COVERAGE.md`

Measured records (all under `_scratch/`):
`block/STEP1.md`, `STEP2.md`, `STEP3.md`, `STEP4.md`, `STEP5.md`, `STEP6.md`
`vae/FINDINGS.md`, `VAEG.md`
`textenc/FINDINGS.md`, `TE_NPU.md`, `API.md`
`image/IMAGE.md`
`perf/PERF.md`, `perfc/PERFC.md`
`qwenimg/ARCHITECTURE.md` (a separate survey, NOT the model being ported)

Reusable tools:
`step2d_audit.py` — the codegen acceptance gate (run on every build)
`step5_chain.py` — silicon residency measurement
`weights_integrity.py`, `api_check.py`, `final_audit.py`
`perf/roofline.py`, `perf/tile_space.py`
`perfc/perfc_probe.py` — the ablation harness + hardware trace

---

## 10. HOW TO WORK

* Use subagents with STRICT scope (one directory per agent; collisions have
  happened twice).
* **Do critical steps yourself.**
* Require `STEP*.md` written AS YOU GO, so progress survives an interruption.
* Report only on major problems; the owner does not want a running commentary.
* Record every verbatim refusal and every ruled-out path — they are what stops
  the next session repeating work.
* The owner values **honest negatives over optimistic claims**.  Two agents
  were explicitly praised for withdrawing an over-claim and for reporting a
  failed restore rather than hiding it.

---

## 11. OPEN QUESTIONS WORTH PASSING ON

* **Where does the ~2 us per-call cost go?**  The copy was 41% and is fixed;
  `mm_bfp`'s body is 445 cycles of a 3521-cycle interval (12.6%).  The rest is
  A/B object delivery at ~5 GB/s.  Is that the DMA floor or is there slack?
* **What does the closed-source 8-second route do with the on-chip cache?**
  It reaches ~40-80% utilisation where we are at 0.38%.  Residency across the
  8 denoising steps is the obvious unexploited opportunity.
* **The `K/Kc` axis is uncharacterised** — two controls give opposite-signed
  exponents.
* **Text encoder view-address question** (`textenc/TE_NPU.md`): chunk 0 recovers
  `INV` correctly while chunk 1 collapses.  Address-keyed, arithmetic-invariant.
  Not a blocker.
