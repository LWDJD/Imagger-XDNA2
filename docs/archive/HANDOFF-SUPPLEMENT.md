# HANDOFF-SUPPLEMENT — answers to the cold-start questions

**Purpose**: close the gaps in `HANDOFF.md` that a zero-modification cold start
exposed.  Read `HANDOFF.md` first, then this.

**Headline**: the cold start is correct and the project runs.  The broken part
is the top-level `REPRODUCE.md` / `README.md`, which were written early and
never re-pointed at the real harnesses.  **The real entries are the `_scratch`
scripts**; the top-level demos stop at MLIR.

Priority answers: **A1, A2, A6, C3, D1** (marked **[P]** below).

---

## A. Top-level documents vs implementation

### **[P] A1. `REPRODUCE.md` step 4's `--compile` does not exist** — CONFIRMED

`bfp16_gemm_iron.py` is a **resolve-to-MLIR demo only** (135 lines,
`--m/--k/--n/--seed/--check-only`).  It never calls aiecc.  That is by design:
it proves the IRON assembly resolves.

**The real compile/device entries are** (all take `--skip-compile`, `--runs`,
`--npz/--xclbin/--insts/--out`):

> **【作废 2026-10-19 / VOID（t21_qfix）】上面那句"all take ..."和下面这张表都不能
> 照抄。** 实测各脚本 argparse:`step3_run.py`/`step4_run_ffn.py` 用 **`--io`**
> (没有 `--npz`);`step1_run.py` 用 `--npz` 且输出是 **`--out`**;`step2*` 与
> `step4_run.py` 输出是 **`--outZ/--outX`**;`step6_run.py` 用 **`--compile`**
> (根本没有 `--skip-compile`);`step4_accept.py` 只有 `--seed/--real-weights`;
> `perfc_probe.py` 没有 `--npz/--xclbin/--insts/--out/--tag/--skip-compile`。
> 正确参数表:**`HANDOFF-SUPPLEMENT-v2.md`:53-62**(已更正 step1 行)与
> **`REPRODUCE.md` §3.1**(2026-10-19 逐脚本读出)。下表 expected 一列的数值仍有效。

| stage | script | expected |
|---|---|---|
| 1. norm -> adaLN -> GEMM | `_scratch/block/step1_run.py` | 2/2, **max ulp 1.000** |
| 2. attention | `_scratch/block/step2_run.py` (see `step2d_build_win.py` for the codegen) | 2/2, 3.125e-02 |
| 3. FFN + residual | `_scratch/block/step3_run.py` | 2/2, 5.3489e-02 |
| assembled block | `_scratch/block/step4_run.py`, `step4_accept.py` | 5.0760e-02 <= 5.4e-02 |
| residency (272 dispatch) | `_scratch/block/step5_run.py` | see D1 |
| real-geometry attention | `_scratch/block/step6_*` | refusal; see HANDOFF §4 |
| first NPU image | `_scratch/image/image_run.py --stage all --tag <tag>` | PSNR 39.1683 dB |
| per-call fix | `_scratch/perfc/perfc_probe.py` | 187.64 GFLOP/s |

**Action**: rewrite `REPRODUCE.md` steps 3-5 as the commands above, or mark the
top-level demo "MLIR only; compile and device runs are under `_scratch/`".

### **[P] A2. `REPRODUCE.md` step 5's `mm_bfp_ref` comparison does not exist** — CONFIRMED

`run_with_xrt.py` only allocates BOs and dispatches.  **The verification lives
in the stage harnesses**, e.g. `step1_run.py` prints `max ulp` against
`kernels/refs.py`, and `step4_accept.py` judges against `step4_oracle.py`.

**Expected numbers to quote**:
* stage 1: `max ulp 1.000`, 99.90% bit-exact
* stage 3: `rel_to_absmax 5.3489e-02`
* assembled block: `5.0760e-02` vs the CORRECTED truth

**Action**: point `REPRODUCE.md` step 5 at `_scratch/block/step1_run.py`'s
verify output.

### A3. `REPRODUCE.md` step 0 relative paths only work from a specific cwd — CONFIRMED

`_pylibs` lives at `<workspace>\_pylibs`, i.e. the PARENT of the repo.  The
in-tree scripts use `HERE.parent / "_pylibs"`.

**Sanity command from the repo root**:

> **【作废 2026-10-19 / VOID（t21_qfix）】下面这个代码块的 import 是错的,
> 逐字执行会失败。** 保留原文只为留痕,不要照抄。
> 实测:第 66 行 `from aie.iron import NPU2` → `ImportError: cannot import name
> 'NPU2' from 'aie.iron'`(exit 1);改成 `from aie.iron.device import NPU2` →
> `aie OK NPU2()`(exit 0,从 `C:\` 运行,cwd 无关)。
> **可用块见 `HANDOFF-SUPPLEMENT-v2.md`:19-27,或 `REPRODUCE.md` §0。**

```powershell
cd <checkout>
$py312 = '<python312>\python.exe'
$env:PEANO_INSTALL_DIR = '<workspace>\_pylibs\llvm-aie\llvm-aie'
$env:PYTHONPATH = '<workspace>\_pylibs\mlir_aie\python;<workspace>\_pylibs'
& $py312 -c "import aie; from aie.iron import NPU2; print('aie OK')"
```

**Action (VOID 2026-10-19)**: do NOT put this block in `REPRODUCE.md` step 0 --
`REPRODUCE.md` §0 already carries the corrected, cwd-independent, measured block.

### A4. `verify_packer.py` expectations vs `KNOWN-ISSUES.md` "expected divergence" — CONFIRMED CONTRADICTION

`KNOWN-ISSUES.md` already records that `pack_bfp16` and `aie.utils.bfp` have
**the same length but different content**, with the exact per-shape diffs
(124/260/1036/2100 bytes).  `verify_packer.py`'s "expect MATCH" is the stale
text.

**Correct expectation**:
> `verify_packer.py` reports `bytes == length` for every shape (the length is
> the contract) and a non-zero `diff` for shapes where `pack_bfp16` and
> `aie.utils.bfp` disagree.  **The diff counts must match the
> `KNOWN-ISSUES.md` table exactly.**  The authoritative check is
> `weights_integrity.py` (payload tiles exactly) and the byte-exact comparison
> against the vendor `.rai` (416,071,680/416,071,680, 0 diff).

**Action**: fix the wording in `README.md` / `REPRODUCE.md` to match
`KNOWN-ISSUES.md`; do NOT "fix" the packer — it is the one that matches the
vendor.

### A5. `README.md` / `REPRODUCE.md` never point at the real harness — CONFIRMED

**The real entry list is now in `HANDOFF.md` §9 (Reusable tools) and §3.**
Add a "real entries" section to `README.md` that points at `HANDOFF.md`.

### **[P] A6. `HANDOFF.md` §0 acceptance #3 conflates two artifacts** — CONFIRMED, AND IT IS MY ERROR

You are right.  They are two different experiments:

| artifact | dispatches | what it is |
|---|---|---|
| `image_npu.png` | **8** (8 steps x 1 block/step), 8/8 COMPLETED | the 1-block image |
| `image_b34_npu.png` | **272** (8 steps x 34 blocks/step), 272/272 COMPLETED | the 34-block image, mean 326.1 ms, loop 250.6 s |

`IMAGE.md` §7 records the 272-dispatch run separately.  **`HANDOFF.md` §0 and
§3 must say: `image_npu.png` = 8 dispatches; `image_b34_npu.png` = 272
dispatches.**  The 250.6 s figure belongs to the 272-dispatch run.

**Action**: correct `HANDOFF.md` §0 and §3.  (Already accurate in
`IMAGE.md`.)

---

## B. `KNOWN-ISSUES.md` gaps

### B1. `verify_packer.py`'s 5th case — EXPECTED FAILURE, not a doc gap

`(8,32) tile=32` is an **illegal input** for `pack_bfp16`: the tiling rule
requires `N%32==0 and K%32==0`.  `N=8` violates it, so the packer refuses
correctly.  **This is the packer doing its job**, not a missing record.

**Action**: annotate `verify_packer.py`'s expectation for that case as
"expected refusal".

### B2. top-level `bfp16_gemm_jit.py` fails at CDO — KNOWN DEAD ROUTE

The verbatim error

```
aiecc: --get-cdo/-pdi/-xclbin require an NPU device, but 'main' is not NPU
```

is the same class as the `set_current_device(NPU2())` trap.  That script does
not set the device, so its `@jit` path cannot reach CDO.  **It is a legacy
exploration, not a maintained path.**

**Action**: mark `bfp16_gemm_jit.py` deprecated in its file header and in
`REPRODUCE.md`.  Do not invest in fixing it — the `_scratch` harnesses are the
maintained path.

### B3. `REPRODUCE.md` step 3's typing problem is FIXED — STALE RECORD

The scripts now use `np.dtype[v8bfp16ebs8]` and resolve cleanly.  **Mark that
record "fixed" rather than deleting it** — the history explains why the type
idiom is what it is.

---

## C. Artifacts and paths

### C1. `weights.bfp16` existence — TRUST THE FILE, THE NOTE IS STALE

**`weights.bfp16` exists at 6,913,965,970 B** (`zimage-port/weights.bfp16`) and
`weights_integrity.py` verifies it tiles exactly.  The `KNOWN-ISSUES.md` note
about "deleted in a disk cleanup, needs rebuilding" refers to an older incident
that was already repaired.

**If it ever needs rebuilding**:
```
python build_bfp16_weights.py --out zimage-port/weights.bfp16
```
`build_bfp16_weights.py` is in `<workspace>\zimage-port\`.  It reads the
**transformer** shards (`_models/Z-Image-Turbo/transformer/`, 3 safetensors)
via `read_checkpoint_tensor` (header byte offsets + `np.frombuffer`, because
`safetensors.get_tensor()` raises an access violation on this machine).

**Action**: mark the stale note "resolved".

### C2. `final_audit.py`'s `_out/step1.*` expectation is WRONG — CONFIRMED

The real files are `_scratch/block/out/step1.xclbin` (17,752 B) and
`step1.bin` (900 B).  `_out/` is a stale expectation from an earlier layout.

**Action**: fix `final_audit.py` to look at `_scratch/block/out/`.  Low risk —
it is a read-only checker and the MISS is already not counted.

### **[P] C3. `step2d_audit.py` needs `merged_core.ll` for the B1/B2 and ORDER gates**

**Where `.ll` comes from**: the `llc -O1` injection path produces it.  In
`step2d_build_win.py` / `step4_build_win.py` the `aiecc --cut` step writes the
merged core IR before the object is swapped.  The file is named
`peano-linked_main_core_0_2.ll` in the build's `.prj` directory.

**To make the audit complete**, keep that `.ll` next to the ELF:

```
<build>.prj/peano-linked_main_core_0_2.ll     <- feed to --ll
<build>.prj/elfs_main_core_0_2/elfs_main_core_0_2.elf
```

**Without it the audit covers only ACQ-clobber and ARG** — which is exactly
what you measured.  That is a real limitation and should be stated.

**Action**: document in `HANDOFF.md` §5 that `step2d_audit.py` has two modes:
* **ELF-only** (what you ran): ACQ-clobber + ARG
* **with `--ll`**: adds the B1/B2 and ORDER gates

and have the injection scripts copy the `.ll` alongside the ELF so the full
gate is reproducible.

### C4. "audit clean" must be stated precisely — CONFIRMED

The winning O1 build reports **"6 acquires, 0 ACQ CLOBBER, 1 ARG exposure,
PASS"**.  The ARG check is an **informational** gate (the script's own
docstring says so).  "Audit clean" therefore means:

> **ACQ-clobber gate clean (0 exposures); 1 informational ARG exposure, which
> the gate does not fail on.**

**Action**: replace every bare "audit clean" in `HANDOFF.md` with that
sentence.  This is exactly the kind of imprecision that produces a wrong
acceptance later.

### C5. resolved — no action

---

## **[P]** D1. `step5_run.py` cannot be re-run without overwriting the record — CONFIRMED, AND IT MATTERS

This is the most important item for reproducibility of acceptance criterion #2.

**The problem**: `OUT = _scratch/block/step5_out/*` is hard-coded and that
directory holds the measured record.

**Two options, pick one**:

1. **Add `--tag` / `--out`** to `step5_run.py`, defaulting to the current
   names, so a re-run writes `step5_out/<tag>/`.  This matches what
   `image_run.py` already does (`--tag r3v` produced a clean, non-overwriting
   reproduction — good design).
2. **Document a no-overwrite procedure** that copies `step5_out/` aside first.

**Recommendation: option 1.**  It is a small change and it is what made the
image path reproducible.

Until that lands, the safe re-run procedure is:

```powershell
# preserve the record
Copy-Item _scratch\block\step5_out _scratch\block\step5_out.bak -Recurse
# then run
<py312> _scratch\block\step5_run.py ...
# compare, then restore
```

---

## E. Leftover files from the verification — recommendation

`r3v_*` (48 files, 171 MB) is a **complete, byte-identical reproduction** —
keep it or delete it, both are defensible.  **Recommendation: DELETE**; the
conclusion is in the F table and the reproducibility is established by the
record, not by keeping 171 MB of outputs.

`r3img_*` (45 files, 171 MB) is a **partial** run (device half done, no final
PNG).  **Recommendation: DELETE** — a partial run proves nothing.

**Keep neither.**  They are regenerable with one command and the git tree is
for source, not payloads (`.npy`/`.npz` are gitignored).

---

## F. What the cold start established, and what it means

The cold start's F table is correct and is the strongest external validation
this project has had:

* audit PASS on the shipped ELF
* `dit1.xclbin` 2/2 COMPLETED, run-to-run byte-identical
* `image_run.py --stage all --tag r3v` reproduces `image_npu.png`
  **byte-identically** (sha256 `5f59dfaa871cdff2`, 35,866 B)
* PSNR 39.1683 dB vs `oracle96.png`; the host half is byte-identical to
  `oracle96.png`; Euler 8/8 bit-exact
* numerics PASS: `range_frac 6.205e-03 <= 7.813e-03`, max ulp 1.000,
  device-vs-model `rel_to_absmax` max 4.8780e-02, 84.41% bit-exact

**That is the project running.**  The remaining work is documentation
hygiene (A1-A6, C2-C4) plus two small code changes (C2, D1), and then the
performance work in `HANDOFF.md` §7.

---

## SUMMARY OF ACTIONS

| id | action | priority |
|---|---|---|
| **A6** | correct `HANDOFF.md` §0/§3: `image_npu.png` = 8 dispatch, `image_b34_npu.png` = 272 | **[P]** |
| **A1** | rewrite `REPRODUCE.md` steps 3-5 against the `_scratch` harnesses | **[P]** |
| **A2** | point step 5 at `step1_run.py`'s verify with expected numbers | **[P]** |
| **C3** | document the `.ll` source and the two audit modes | **[P]** |
| **D1** | add `--tag`/`--out` to `step5_run.py` | **[P]** |
| C4 | replace bare "audit clean" with the precise sentence | high |
| A4 | align `verify_packer.py`'s expectation with `KNOWN-ISSUES.md` | high |
| A5 | add a real-entries section to `README.md` | medium |
| A3 | add the cwd-independent sanity block to `REPRODUCE.md` | medium |
| C1 | mark the stale `weights.bfp16` note resolved | medium |
| C2 | fix `final_audit.py`'s `_out/` path | low |
| B1 | annotate `verify_packer.py`'s 5th case as expected refusal | low |
| B2 | mark `bfp16_gemm_jit.py` deprecated | low |
| B3 | mark the typing record "fixed" | low |
| E | delete `r3v_*` and `r3img_*` | low |
