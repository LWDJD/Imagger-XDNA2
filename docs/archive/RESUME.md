# Resume runbook — how to pick this project up cold

Written for a future session (or a compacted one) that has none of this
conversation in context.  Everything here is measured, not assumed.

## The goal, in one line

Run Z-Image-Turbo on the AMD XDNA2 NPU with the **open IRON + XRT** stack, all
weights stored and resident as **BlockFP16 (`bfp16ebs8`)**, with no fp32
expansion, no per-call unpack and no repeated loads.

## The machines and interpreters

| what | where |
|---|---|
| compile-side Python | `<python312>\python.exe` (3.12) |
| run-side Python | `<python310>\python.exe` (3.10) |
| mlir_aie | `_pylibs\mlir_aie\python` (put on `sys.path`) |
| **Peano** | `_pylibs\llvm-aie\llvm-aie` — set `PEANO_INSTALL_DIR` to it |
| XRT / pyxrt | `<xrt>` |
| `xclbinutil` | same XRT directory — **not on PATH, must be added** |
| device | `pyxrt.device(0)`, `xrt-smi` reports NPU Strix at `0064:00:01.1` |

Two interpreters are required, not a mistake: `pyxrt.pyd` links `python310.dll`
and `aie.utils.bfp` only imports on 3.12.  Encode on 3.12 into an `.npz`, run
on 3.10.

Before importing `aie`, set:

    os.environ["PEANO_INSTALL_DIR"] = r"...\_pylibs\llvm-aie\llvm-aie"
    os.environ["NPU_CACHE_HOME"]  = r"...\zimage-bfp16-npu\_npu_cache"

`NPU_CACHE_HOME` is read at import time.

## The deliverable tree, audited

`final_audit.py` checks that every document, script, directory and artifact is
present and that the scripts parse.  Last run: **9 documents, 12 scripts, 14
directories, all OK.**  Key artifacts:

    weights.bfp16                6,913,965,970 B    506 tensors, 3.554x
    vendor_artifacts/*.rai         418,881,856 B    sha256 recorded, irreplaceable
    pipeline/oracle.png             35,958 B        the reference image
    block/out/s1w_*.xclbin          21,304 B        stage 1, verified on the NPU

Re-run `python final_audit.py` at any time to confirm the tree is intact.

## What is done — do not re-derive

* BFP16 encoder `pack_bfp16.py`, byte-exact against the vendor `.rai`
* toolchain compiles BFP16 for aie2p; the pin is
  **`llvm-aie==22.0.0.2026092101`** from `utils/peano-requirements.txt` at
  mlir_aie tag v1.4.4.  The 21.0.x nightlies fail to legalise
  `mm_bfp.cc`'s `G_BUILD_VECTOR <8 x s8>`.
* BFP16 matmul verified on the NPU at the quantisation floor (rel 1.01e-2)
* RMSNorm -> mm_bfp(mixed) two-op pipeline, **100.00% bit-exact**, hidden to
  1792; the corner turn is expressed as a DMA `to_stream` walk
* kernel coverage 7/8 (`COVERAGE.md`); adaLN is the only missing symbol
* designs with file:line evidence: `attn/DESIGN.md`, `adaln/DESIGN.md`

## Traps that cost real time

* **`mm_bfp_mixed.cc` re-rounds C to bf16 after every GEMM call.**  A numpy
  reference that accumulates in f64 drifts up to 13 ulps at hidden 1792 and
  looks like a kernel bug.  Round-trip through bf16 per accumulation.
* **`pack_bfp16`'s API is `pack(W, tile_rows=None) -> bytes`.**  There is no
  `quantize(...)` and no `unpack_and_quantize(...)`.  Read the module.
* `mm_bfp_mixed.cc` static_asserts **m%16==0 and n%16==0**, though the Python
  factory only checks %8.
* Host encoding is `bfp.shuffle(bfp.encode(v), w, h, tw, th)` with `(tw,th)` the
  PER-CORE TILE, not the whole matrix.  Wrong here gives plausible numbers and
  no error.
* Compute-tile DMA descriptors take **at most 3 dimensions**; memtiles take 4.
* bfp16ebs8 strides must be **4-byte multiples**; 9-byte blocks need repacking.
* L1 is **64 KB** per compute tile; a memtile is **512 KB**; a single DMA
  descriptor caps at **16383 32-bit words**.
* **`os.mkdir(path, 0o700)` produces a directory the sandbox token cannot
  re-enter**, and `tempfile` uses exactly that mode, which breaks
  `prefix_symbols_in_object`.  Workaround: a process-local `os.mkdir` shim
  widening 0o700 -> 0o777 plus `use_cache=False`.  This is NOT an ACL problem
  (`diagnose-windows-sandbox-acl` returned NOT_THIS_CLASS).
* `safetensors.get_tensor()` raises 0xC0000005 here.  Read via the header's
  byte offsets and `np.frombuffer`.
* D: has ~5 GiB free.  `_models` (30.6 GiB) and `_probe_out` (35.5 GiB) are
  each hard-linked across three trees; deleting a copy frees nothing.  If space
  runs out, delete `_probe_out` subdirectories — they are regenerable.

## Documents, in reading order

| file | what it is |
|---|---|
| `README.md` | the route, the format spec, the hardware, the milestone table |
| `COVERAGE.md` | which kernels exist for a DiT block and which do not |
| `ASSEMBLY.md` | the 10-step block and the 7-stage integration order |
| `PIPELINE.md` | the denoising loop and what to measure |
| `KNOWN-ISSUES.md` | every trap, every ruled-out path, every correction |
| `ENVIRONMENT.md` | exact paths and versions |
| `docs/WORKLOG.md` | dated record of what was tried |

## Where the work stands

Parallel subagents were implementing: `rms_norm_weighted` + `adaln_modulate`,
non-causal attention, the hidden-3840 memtile path, block weight extraction,
and a numpy oracle for the full DiT block.  Their outputs live under
`zimage-bfp16-npu\_scratch\<topic>\` and are promoted into the deliverable tree
with `promote_scratch.py` once reviewed.

Next step after those land: `ASSEMBLY.md` stages 1-2 — one GEMM at hidden 3840
(the load-bearing piece), then norm -> adaLN -> GEMM.  Do not skip stage 2 of
`PIPELINE.md`; weight residency is the whole point and it is invisible until
two blocks run back to back.
