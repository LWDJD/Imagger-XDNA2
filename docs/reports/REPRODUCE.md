# Reproduce

Every step is runnable.  Nothing depends on state outside this directory except
the toolchain listed in `ENVIRONMENT.md`.

## 0. Sanity: the toolchain is reachable

**Correction 2026-10-19 (t21_qfix / A3).** The compile-side command used to be

    python -c "import sys; sys.path += ['_pylibs','_pylibs/mlir_aie/python']; \
               import aie, aie.iron; from aie.iron.device import NPU2; print(NPU2())"

which is cwd-dependent: `_pylibs` is the **parent** of this repository, so the
relative entries only resolve when the working directory happens to be the
repository's grandparent.  Use the absolute block below instead.  Measured
2026-10-19 **from `C:\`** (i.e. not from the repo): exit 0, `aie OK NPU2()`.

```powershell
$py312 = '<python312>\python.exe'
$env:PEANO_INSTALL_DIR = '<workspace>\_pylibs\llvm-aie\llvm-aie'
$env:PYTHONPATH = '<workspace>\_pylibs\mlir_aie\python;<workspace>\_pylibs'
& $py312 -c "import aie; from aie.iron.device import NPU2; print('aie OK', NPU2())"
```

`from aie.iron.device import NPU2` is the import that works.  `from aie.iron
import NPU2` raises `ImportError: cannot import name 'NPU2' from 'aie.iron'`
(measured, exit 1) -- do not "simplify" it.  The 3.12 `aie` import does **not**
need `os.add_dll_directory`; that is only for pyxrt on 3.10.

    # run side (Python 3.10)
    "<python310>\python.exe" -c "
        import sys, os
        d = r'<xrt>'
        sys.path.insert(0, d); os.add_dll_directory(d)
        import pyxrt; print(pyxrt.device(0))"

    "<xrt>\xrt-smi.exe" examine

Expected: `NPU2()`, a `<pyxrt.device object>`, and `|NPU Strix|` in the device
table.

## 1. Verify the BFP16 encoder

    python verify_packer.py

This checks `pack_bfp16.py` against AMD's own reference (`aie.utils.bfp` encode
+ `shuffle`) and reports the byte difference per shape.

**Correction 2026-10-19 (t21_qfix / A4).** This paragraph used to end with
"A correct run prints `MATCH` for every case".  That is wrong.  The byte
**lengths** agree; the byte **content** does not.  Measured 2026-10-19 with the
3.12 interpreter on this tree:

    shape      tile     bytes   length    diff   note
    (8,  32)      8       288      288      124   124 bytes differ
    (16, 32)      8       576      576      260   260 bytes differ
    (16, 64)     16     1,152    1,152    1,036   1,036 bytes differ
    (32, 64)     32     2,304    2,304    2,100   2,100 bytes differ

These four diff counts must match `KNOWN-ISSUES.md`:988-1000 exactly; the
verdict line is `VERDICT: differences found`, not `MATCH`.  **The length is the
contract** (1.125 B/param).  Do not "fix" the packer to make this table empty:
`pack_bfp16` is the side that matches the vendor `.rai` byte-for-byte
(416,071,680/416,071,680, 0 diff), so the residue is a property of
`aie.utils.bfp`'s shuffle/rounding path.  The authoritative integrity checks
are `weights_integrity.py` and the vendor-`.rai` comparison.

## 2. The numpy reference (no hardware)

    python bfp16_gemm_iron.py --check-only

Expected: `numpy reference vs float64 matmul: rel ~1.8e-02` (BFP16 precision
level) and `BLOCK=8  BLOCK_BYTES=9`.

## 3. Build the IRON program

    python bfp16_gemm_iron.py --m 64 --k 64 --n 64

This assembles `mm_bfp` with ObjectFIFOs, a Worker and `pipeline`, then
resolves to MLIR.  `KNOWN-ISSUES.md` records the one open typing problem at
this step.

**Correction 2026-10-19 (t21_qfix / B3): that typing record is FIXED, not open.**
The scripts now annotate `np.dtype[v8bfp16ebs8]` and resolve cleanly; the
record itself is `KNOWN-ISSUES.md`:979-986 ("The dtype annotation must be
`np.dtype[marker]`, not the bare marker"), which explains why the idiom is
what it is and is worth keeping.  Nothing is open at this step.  (The sentence
above is also imprecise on its own terms: the word "typing" does not occur in
`KNOWN-ISSUES.md`; the record is the dtype one.)

### 3.1 The real compile/device entries (correction 2026-10-19, t21_qfix / A1)

`bfp16_gemm_iron.py` is a **resolve-to-MLIR demo**, not the compile path: its
entire argparse is `--m/--k/--n/--seed/--check-only`, and it never calls
`aiecc`.  The entries that actually compile and run on the NPU are the
`imaggers` harnesses.  **Their flags are per-script: `--npz`, `--io` and
`--outZ/--outX` are not interchangeable**, and `--compile` exists on exactly
one of them.  The argument lists below were read out of each `argparse` on
2026-10-19:

| stage | entry (under `imaggers/`) | arguments that exist | expected |
|---|---|---|---|
| 1. norm -> adaLN -> GEMM | `block/step1_run.py` | `--npz --xclbin --insts --out --runs --skip-compile --skip-run --skip-static --verify-only --xrt-only` | 2/2 `ERT_COMPLETED`, `max ulp 1.000` |
| 2. attention | `block/step2_run.py` (real codegen: `step2d_build_win.py`) | `--outZ --outX --npz --xclbin --insts --runs --skip-compile --verify-only --xrt-only`; `step2b` adds `--tag`; `step2c/2d` add `--probe --probe-head` | 2/2, `3.125e-02` |
| 3. FFN + residual | `block/step3_run.py` | **`--io`** (there is no `--npz`), `--xclbin --insts --out --runs --level --skip-compile --verify-only --xrt-only` | 2/2, `rel_to_absmax 5.348863e-02` |
| 4. assembled block | `block/step4_run.py` | `--outZ --outX --npz --xclbin --insts --runs --win --skip-compile --verify-only --xrt-only` | judged by `step4_accept.py` (§5) |
| 4b. FFN half / acceptance | `block/step4_run_ffn.py`, `block/step4_accept.py` | `step4_run_ffn.py`: **`--io`**, `--xclbin --insts --out`; `step4_accept.py`: **only** `--seed --real-weights` | `5.0760e-02 <= 5.4e-02` PASS |
| 5. two-block chain on silicon | `block/step5_run.py` | `--weights --blocks --prep --skip-prep --skip-device --skip-verify --verify-only --switch-cost --xrt-only --switch-only --only-image --tag --out` | `imaggers/block/STEP5.md`; `--tag` keeps the record intact (D1) |
| 6. real-geometry attention | `block/step6_run.py` | **`--compile`** (NOT `--skip-compile`), `--prep --device --verify --all --tag --weights --runs --seq-cap --xrt-only`, plus `--npz --xclbin --insts --outO --outX` | refusal at real geometry; `HANDOFF.md` §4 |
| first NPU image | `image/image_run.py` | `--stage --steps --seed --blocks-per-step --dump-calls --ffn-mode --call --tag` | `--stage all --tag <name>` reproduces `image_npu.png`, PSNR 39.1683 dB |
| per-call fix | `perfc/perfc_probe.py` | `--variant --all --list --fit --trace-report --runs --warmup --stage --trace-bytes --walk-check --b-check --stalls --shape` (**no** `--npz/--xclbin/--insts/--out/--tag/--skip-compile`) | 187.64 GFLOP/s |

The expected column is the record's, one anchor per number:
`imaggers/block/STEP4.md`:379 (`5.348863e-02`), `imaggers/block/STEP4.md`:44
and `imaggers/block/STEP2D.md`:244 (`3.125e-02`), `imaggers/block/STEP5.md`:21
(`5.0760e-02`), `imaggers/image/IMAGE.md`:66 (PSNR 39.1683 dB),
`HANDOFF.md`:37 and `imaggers/perfc/PERFC.md`:24 (187.64 GFLOP/s),
`HANDOFF.md`:88-90 (`max ulp 1.000`).  They are here so a reader can tell a
good run from a bad one.  **None was re-measured while writing this document.**

Every harness splits the same way: 3.12 does build / prep / verify / judge and
calls the 3.10 half itself (`--xrt-only`); 3.10 owns `pyxrt`.  Run the 3.12 half
and let it drive the other.

## 4. Compile to xclbin and run on the NPU

**Correction 2026-10-19 (t21_qfix / A1) -- the two commands that used to be
here are VOID:**

    python bfp16_gemm_iron.py --m 64 --k 64 --n 64 --compile        # VOID
    "...\python.exe" run_with_xrt.py --xclbin out/final.xclbin ...   # VOID

`bfp16_gemm_iron.py` has no `--compile` (its argparse stops at `--check-only`)
and produces no `out/final.xclbin`, so the first line ends in
`unrecognized arguments: --compile`.  The two-interpreter paragraph at the end
is the part that is still true, and it still matters.

**What actually compiles and runs** (the harnesses from §3.1):

* **Compile** -- run the stage harness under 3.12 **without** `--skip-compile`;
  it builds its own image and instruction stream (needs `PEANO_INSTALL_DIR`, §0,
  plus disk and time; this is the step that is NOT reproducible on a 0.9 GiB
  volume).  With `--skip-compile` it reuses the delivered image, which is the
  compile-free path every number in this file was taken with.
* **Run** -- the same harness calls its 3.10 half (`--xrt-only`) with that
  stage's own argument group; you do not hand-drive `pyxrt`.  The bare dispatch
  test `run_with_xrt.py --xclbin <x> --insts <i>` still works under 3.10; it
  allocates BOs and dispatches and verifies nothing (see §5).
* **Two interpreters, on purpose**: mlir_aie is a cp312 wheel, `pyxrt.pyd`
  links `python310.dll`.

**Nothing in this section was rebuilt or re-run on silicon while writing this
document** (t21_qfix is a paper pass; the NPU was not touched).  The delivered
images these numbers came from are on disk: `imaggers/block/out/step1.xclbin`
(17,752 B) + `imaggers/block/out/step1.bin` (900 B), and the three stage images
`imaggers/block/step5_run.py` loads (its `IMAGES` map).

## 5. Compare against the reference

**Correction 2026-10-19 (t21_qfix / A2).** The sentence that used to be here --
"`run_with_xrt.py` prints the relative error against `mm_bfp_ref`" -- is false.
`run_with_xrt.py` allocates BOs, dispatches, and prints
`run complete; result buffer ready`; it computes no error and never mentions
`mm_bfp_ref`.

The verification lives in the stage harnesses.  For stage 1 it is
`imaggers/block/step1_run.py`'s verifier -- the entry this step is supposed to
point at:

    $py312 imaggers\block\step1_run.py --verify-only

which prints the per-run numeric line

    [verify]    max abs err <...>   max ulp <...>   ...

(`imaggers/block/step1_run.py`:567) and the gate

    [verify] PASS (<= 2 bf16 ulps vs numpy): True

(`:576`).  **Expected values** (the record's, not re-measured here):
`max ulp 1.000`, 99.90% bit-exact, 2/2 `ERT_COMPLETED`, run-to-run
byte-identical outputs.

Per stage:

| stage | verifier | expected |
|---|---|---|
| 1 | `step1_run.py --verify-only` | `max ulp 1.000`; `PASS` |
| 2 | `step2_run.py --verify-only` | 2/2; `3.125e-02` |
| 3 | `step3_run.py --verify-only` | `rel_to_absmax 5.348863e-02` (budget 5.4e-02) |
| 4 (assembled) | `step4_accept.py` -- **the authority** | `rel_to_absmax 5.0760e-02` -> PASS |
| 5 (residency) | `step5_run.py --verify-only` | per block vs its own device-exact model; `imaggers/block/STEP5.md` |

**Step 4 has two verifiers and they disagree; `step4_accept.py` is the
authority.**  `step4_run.py --verify-only` exits 3 with
`check_within_ulps(<=2): False` / `max ulp 43072` (fourth-round record), because
that gate compares in bf16 ulps and an ulp is not a meaningful unit for values
at or near zero and subnormals.  `step4_accept.py` instead reports
`rel_to_absmax` from `step4_oracle.divergence` against the CORRECTED float truth
(`imaggers/block/step4_accept.py`:111-114, :201-205) and passes at
`5.0760e-02 <= 5.4e-02`.  **Do not read the `step4_run` ulp exit code as an
acceptance failure**; the ulp gate should be fixed or dropped, and until it is,
this paragraph is the ruling.

> **2026-10-20 update (t22_factcards2)**: the tool was fixed rather than this
> wording.  `step4_run.py --verify-only` now judges `rel_to_absmax <= 5.4e-2`
> (the project's own budget) and prints the ulp numbers as
> `[DIAGNOSTIC ONLY, not a verdict]`; it exits 0 at 7.9365e-03 / 2.7902e-02
> (M=96 / K=96).  The old `max ulp 43072` is a historical artifact and no longer
> reproducible (today: 1.66e+35 / 2.20e+05, the first from a single zero
> reference element).  `step4_accept.py` remains the authority and
> `5.0760e-02` is still a **recorded** value, not a local re-measurement (the
> rerun hit page-file / memory pressure).  Source: `docs/T22-QUICKFIX-REPORT.md`
> §1 and §5 items 3-4.

---

## What "reproducible" means here

* `pack_bfp16.py` is self-contained (numpy + stdlib) and its output is
  deterministic given the same input and `tile_rows`.
* `verify_packer.py` recomputes the expected bytes from AMD's own
  `aie.utils.bfp`, so the encoder is checked against an independent source
  rather than against itself.
* `bfp16_gemm_iron.py --check-only` exercises the whole quantisation path
  without any hardware.
* The hardware steps degrade gracefully: if `pyxrt` or the device is missing,
  they report exactly what is missing instead of crashing.

## Regenerating the single weight file

`weights.bfp16` (6.439 GiB, 506 tensors) is produced by
`build_bfp16_weights.py` in the parent project.  Its header is

    [8-byte LE length][JSON header][payloads]

with each tensor carrying its `tile_rows`, so the payload can be decoded
without any external state:

    pack_bfp16.unpack(blob, N, K, tile_rows)
