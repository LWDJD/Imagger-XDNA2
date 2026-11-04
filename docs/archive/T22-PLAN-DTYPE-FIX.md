# T22_PLAN_FFN × T22_FASTGLUE: the silent-garbage FFN input, fixed and gated

**2026-11-03 night · 512² · plan arm (T22_PLAN_FFN=1) · house recipe (T22_FASTGLUE=1) · AC**

`@
VERDICT: the defect is FIXED (the plan arm is now bit-identical with the hooks ON and OFF, and its
delivered comparison went from corr -0.013689961 to 0.998478950), and the class of bug is GATED in
two places -- a refusal at the executor boundary and a seconds-long host gate. T22_PLAN_FUSE_GU is
RECORDED now (its own env_id) and still FAILS the correctness gate, so it stays 0.
`@

---

## 1. The defect, as measured (not as argued)

With the house recipe exactly as shipped (T22_FASTGLUE=1, hooks 8/8, block=on(4 hunks)), one hunk
changes the TYPE of the FFN input:

| hunk | file:line | what it does to the FFN input |
|---|---|---|
| h3-ffn-norm1-bf16 | _scratch/util/glue/install.py:291-300 | _xn becomes fast_glue.norm_bf16(xm, ffn_norm1, mod[2]) -- a **uint16 BF16 IMAGE** (the same values as a bit pattern, 2 B each) |

devkill_ffn consumes that image correctly (it hands the bytes to the image), but the PLAN executor
began with

    x = np.asarray(xn, np.float32)

which is a **numeric cast**: bf16 1.0 is the bit pattern 0x3F80, and that call turns it into the
integer 16256.0. The FFN then computed a different function for 30 layers.

**Measured at the three boundaries in the product (512² x1, AC, house recipe), from the job logs:**

| point | hooks ON (T22_FASTGLUE=1) | hooks OFF (T22_FASTGLUE=0) |
|---|---|---|
| t18_run call site (before plan_ffn) | dtype=**uint16** absmax=49463 sha=f461ef95ed6ee4a6 | dtype=float32 absmax=33.8248 sha=f924a378ddf54b0a |
| plan_ffn.ffn() entry | dtype=**uint16** absmax=49463 sha=f461ef95ed6ee4a6 | dtype=float32 absmax=33.8248 sha=f924a378ddf54b0a |
| s2_ffn32.ffn32_run entry | xn_dtype=**uint16** -> dtype=float32 absmax=**33.75** sha=8571b618a2c70586 | xn_dtype=float32 -> dtype=float32 absmax=33.8248 sha=f924a378ddf54b0a |
| the A payload the device gets | **8b661de20f6831fd** | **8b661de20f6831fd** |

The last row is the whole point: after the fix the two hook states produce the SAME A payload, and
everything downstream of it (gate/up/h/dn shas, latents) is identical. Before the fix the same line
read absmax 49171.0 and the A payload was a different array.

**The numeric consequence, on the same 30-layer stack:**

| arm | latents (npy sha16) | vs the hooks-off arm | job 'latents' sha |
|---|---|---|---|
| ph1  plan, hooks ON, **old code** | e5f2bd390ab85aed | **corr -0.013689961, maxabs 6.22** | 06a3a5d1eaf59594 |
| u1   plan, hooks OFF (recorded) | 1e1b8420bb9beda3 | -- | 32720d2b96534e5c |
| planON1  plan, hooks ON, **fixed** | **1e1b8420bb9beda3** | **maxabs 0, exact** | **32720d2b96534e5c** |

The broken arm reported 'done' -- that is the silent part. It is also why every earlier
"plan vs delivered" number in this tree was taken with the hooks OFF.

## 2. The fix (one line, plus the contract that makes it impossible to re-add)

_scratch/util/place/s2_ffn32.py, at the executor entry:

    -    x = np.asarray(xn, np.float32)
    +    x = as_a_f32(xn, "ffn32_run")

with three new module-level pieces:

* bf16_image_f32(img) -- uint16 -> float32 by **bit view**:
  (np.ascontiguousarray(img).astype(np.uint32) << np.uint32(16)).view(np.float32).
  That is the expression the DELIVERED arm's devkill_ffn already uses for its own bf16 drains
  (devkill_ffn.py:404, and decode_fast's np.left_shift(b, 16, out=out.view(np.uint32))), so the two
  arms decode a bf16 image identically **by construction**, not by agreement.
* as_a_f32(xn, who) -- float32 returned **as-is** (no copy: the executor must read the caller's
  buffer, which the DET-BISECT probe asserts), uint16 decoded by the bit view above, everything else
  **raises**.
* ffn32_reference() uses the same contract: a reference that decoded differently from the executor
  would hide exactly this defect.

## 3. The refusal (the deliverable)

Any other dtype raises, naming the dtype, the shape, the bytes, the absmax and the sha:

    TypeError: ffn32_run: xn has dtype int32, which this executor does not cast.  It takes float32
    (the normalised activation) or uint16 (a bf16 A image from fast_glue.norm_bf16, decoded by BIT
    VIEW).  Got shape=(96, 3840) bytes=1474560 absmax=0 sha16=6e3a8d0a1f7c8b21.  Refusing: a numeric
    uint16->float32 cast reads bf16 bit patterns as integers (0x3F80 -> 16256.0), which decorrelates
    every layer of the FFN and still reports 'done' -- T22-PLAN-FUSE-GU-ARM.md section 6.

Measured: int32 / uint8 / float64 / float16 were all **silently accepted** before (each produced a
finite, plausible y); they now raise before any A write.

## 4. The gates that keep it that way

1. **The refusal above** -- the executor cannot be handed an image or an integer it does not
   understand; the failure is loud at the boundary, where there is nothing downstream to blame.
2. **gate_ffn32.py --dtypes** (new, seconds, no device, and wired into --all):

       py312 _scratch/util/place/gate_ffn32.py --dtypes --m 1056

   12 assertions per M, PASS at M=96 and M=1056, for the PURE gu and the FUSED gu:
   * the uint16 bf16 image packs the **same** A payload as the float32 activation (bit view);
   * the **numeric cast packs a different** one (the negative control -- without it the check would
     be blind if both arms agreed);
   * int32 / uint8 / float64 / float16 each **raise** a TypeError naming dtype+sha before any A write.
   It needs no GEMM: it swaps the gu A BO for one that records the payload and unwinds.
3. **probe_dtype.py** (_scratch/t22dtype/probe_dtype.py, host-only, M=96): the whole FFN through
   gate_ffn32's functional fake device, in one table -- f32 arm / numeric cast / current code /
   four bad dtypes. Before the fix it printed corr 0.098369663, maxabs 8.686e13 for the image; after,
   **maxabs 0** against the f32 arm and the cast arm still collapses.

## 5. Correctness with the hooks ON (the house recipe)

512², seed 7, prompt 'a mountain lake at sunrise', **AC** (P12 ACLineStatus=1 on every run), device
idle, one job at a time.

| arm | tag / job | latents npy | job sha16 | loop_s |
|---|---|---|---|---|
| delivered default (house recipe) | del1 / 20261103-220513-7b8c6f | 96ec12f7e4c3c8c9 | **395177106cea950b == the reference** | 12.27 |
| plan, hooks ON (fixed) | planON1 / 20261103-220629-989fc8 | **1e1b8420bb9beda3** | 32720d2b96534e5c | 17.13 |
| plan, hooks ON (self-repro) | planON1b / 20261103-221143-fa7145 | **1e1b8420bb9beda3** | 32720d2b96534e5c | -- |
| plan, hooks OFF | planOFF1 / 20261103-220947-f3c40f | **1e1b8420bb9beda3** | 32720d2b96534e5c | 18.21 |

**plan vs DELIVERED, same harness, same window: corr 0.998478950, maxabs 0.930301** (gate >= 0.9953).
Identical to the comparison of the recorded hooks-off plan arm (u1, w210713d938fc) against the same
delivered run -- because the two arms ARE the same bytes.

* self-reproducibility: planON1 vs planON1b **corr 1.000000000, maxabs 0, exact**;
  and planON1 == planOFF1 == u1, maxabs 0.
* the delivered default is untouched: its latents and PNG both reproduce the reference
  (Q01-Q19 all PASS, 39.8 s).

**The recorded-env number (0.995965046).** The gate as written was measured on run_step arms
(_scratch/util/place/run_step.ps1, hooks OFF by that script's design), plan vs delivered:
**0.995965046**. The fixed arm is bit-identical to that arm (row 2 of the table above), so the value
is unchanged by identity -- and the house-recipe pair above is the same statement with both arms in
the FIXED env. [PENDING device window: a fresh run_step pair re-measures it in-window.]

## 6. T22_PLAN_FUSE_GU on the corrected base

[PENDING device window -- the fused re-test was interrupted by the 1080p acceptance run. What is
already established: the fusion is a COMPILE-TIME knob of the gu image (it changes the plan_key, the
image, and the dispatch count), it was measured at 0.995239938 vs the 0.9953 gate with hooks OFF,
and the corrected base does not change a hooks-OFF arm numerically (as_a_f32 returns float32
untouched) -- so the expectation is that it still fails by 6.0e-5 and stays 0. The re-test is queued
in _scratch/t22dtype/finish.ps1.]

## 7. The env record: the fused and unfused arms now have distinct env_ids

T22_PLAN_FUSE_GU was in neither the repro tool's RECIPE nor webui/config.py's LAUNCH_FLAGS, so both
arms ran under env_id f65cd982141f9400 and the only record of which one ran was a log line.

* webui/config.py: LAUNCH_FLAGS += "T22_PLAN_FUSE_GU" (recording only; it selects nothing).
* _scratch/repro/t22_repro.py: RECIPE += ("T22_PLAN_FUSE_GU", "0", reason, source).

| arm | env_id | where it is visible |
|---|---|---|
| recipe / unfused | **cfc618522d695a67** | preflight P06, postflight Q11, job.json env_flags |
| T22_PLAN_FUSE_GU=1 | **d6fd51b8c8b63b6a** | the same three |

The recipe's env_id therefore MOVES (f65cd982141f9400 -> cfc618522d695a67); every earlier record keeps
the id it was measured under, and RUNBOOK.md was regenerated (t22_repro.py doc --check is clean).

## 8. The sibling sites, flagged not fixed

The SAME numeric cast still stands at two other FFN entries: _scratch/util/place/s2_dev.py:456
(xn = np.asarray(xn, np.float32)) and s2_product.py:72. They belong to the T22_S2_FFN32 executor
family, which is on the repro tool's FORBID list and is NOT the plan path, so they are out of scope
here -- but if that family is ever armed on a shape where the fastglue hooks run, it has the same
silent-garbage failure mode and the same one-line fix (s2_ffn32.as_a_f32). Flagged, not touched:
changing a second executor in the same session would put two numerics in one diff.

## 9. Files

| file | what |
|---|---|
| _scratch/util/place/s2_ffn32.py | the fix (as_a_f32 / bf16_image_f32), the refusal, the reference contract, dtype in the DET probe line |
| _scratch/util/place/gate_ffn32.py | --dtypes: the 12-assertion A-image contract gate (also in --all) |
| _scratch/t22dtype/probe_dtype.py | the host-only before/after table through the fake device |
| _scratch/t22dtype/show.py | the compact run-record reader used for the tables above |
| _scratch/t22dtype/finish.ps1 | the queued device runs (recorded-env pair, fused re-test, 8 steps) |
| webui/config.py | LAUNCH_FLAGS += T22_PLAN_FUSE_GU |
| _scratch/repro/t22_repro.py | RECIPE += T22_PLAN_FUSE_GU (distinct env_id per arm) |
| _scratch/repro/RUNBOOK.md | regenerated by 't22_repro.py doc' |
| this file | the record |

**No product default was flipped. The delivered path is byte-identical (latents 395177106cea950b,
PNG fb3aebde, both == the reference), T22_PLAN_FFN stays on the FORBID list, and T22_PLAN_FUSE_GU
stays 0.**
