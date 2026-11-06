# PREOVL-REPORT — the preamble's context half: 2.05 s measured, and it is a CONSTANT of the checkpoint

**Date** 2026-11-02 · Labels: **[M]** measured, **[A]** arithmetic on measured
values, **[S]** structural (code path), **[X]** not measured.

Scope: the 512^2 / 8-step arm (`_scratch/devkill/run_arm.ps1` through
`_scratch/preovl/run_preovl.ps1`), `T22_FASTGLUE=1 T22_REF_SMNB=1
ZIMAGE_CHUNK_THREADS=16`, **fast memory mode in every arm** (recorded per run,
section 5.1). One product file changed (`_scratch/image/t18_run.py`), one new
flag (`T22_PREOVL`, default **0**), no arithmetic touched.

---

## 0. Verdict in six lines

1. **The context half is 2.05 s per 8-step image, not the 3-6 s the brief
   budgeted** [M]: 0.75 s of arithmetic (2 x 45 ms x 8 steps) plus a 1.31 s
   one-off refiner-cache read at step 0. It is **10.9 % of the preamble**
   (R = 18.9 s) and **2.2 % of the loop** (94 s of step walls).
2. **The brief's "~40 % of the refiner" is 4.8 %** [M]: `pre.refiner2` 0.75 s
   against `pre.refiner` 15.5 s. The refiner GEMMs are compute-bound and scale
   with the token count, and the caption is 32 tokens against the image's 1024:
   **45 ms per context block against 962 ms per image block** — 21x the cost for
   32x the rows. The 3-6 s estimate assumed the wrong scaling law, not a wrong
   dependency.
3. **The dependency claim is stronger than the brief's** [S]: the context half
   does not depend on `step_t` either. `cap_embedding(H)` is 32 rows of the
   checkpoint's own learned pad token, and **neither context_refiner block has
   an adaLN** (their `adaLN_modulation` weights are absent from the checkpoint,
   so `modulation()` returns `None` before it reads the adaln vector). The
   buffer is the **same bytes on all 8 steps** — proven on the weights and
   numerically (section 4, the `adaln` test).
4. **So the schedule is not "step i+1's context half under step i's loop" but
   "read its weights under the startup, compute it once, read the buffer 7
   times"** — the same dependency argument, one fewer parallel region, and it
   also removes step 0's copy, which a loop-overlap version cannot.
5. **Measured, two interleaved pairs, fast mode both arms: `loop_s` 91.33 ->
   88.59 and 90.98 -> 88.80 = -2.46 s mean, geomean ratio 0.9731 (-2.69 %)** [M].
   Every `[fastglue]`/`[devkill]` column is within 0.2 ms/layer and the
   latents are `2abf91331200` in **all eight** runs, so nothing moved but the
   schedule.
6. **And the preamble that is left (16.7 s) is 90 % the noise refiner** [M],
   which consumes the latent and is therefore on the chain. The context half was
   the last thing in R that could be moved without changing arithmetic.

---

## 1. What the context half actually costs

One GlueProf run sizes the prize, as the brief asked: the `[t22g]` ledger books
every host segment exclusively, so `pre.refiner2` (the two context blocks) and
`pre.refiner` (the two image blocks) are read off directly. Four delivered-mode
runs, 8 steps each:

| bucket (s) | pv_a1 | pv_a2 | pv_a3 | pv_a4 | calls |
|---|---:|---:|---:|---:|---:|
| `pre.refiner`   | 15.459 | 15.382 | 15.581 | 15.468 | 16 |
| `pre.refiner2`  | **0.726** | **0.741** | **0.774** | **0.747** | 16 |
| `pre.refblock2` | **1.310** | **1.274** | **1.335** | **1.306** | 16 |
| `pre.capemb`    | 0.005 | 0.004 | 0.005 | 0.042 | 8 |
| `pre.refblock`  | 1.247 | 1.338 | 1.151 | 0.870 | 16 |
| `pre.xemb`      | 0.135 | 0.135 | 0.136 | 0.135 | 8 |
| `pre.temb`      | 0.007 | 0.009 | 0.009 | 0.008 | 8 |
| `step.us`       | 0.044 | 0.045 | 0.045 | 0.046 | 8 |
| **R (all of it)** | **18.93** | **18.93** | **19.04** | **18.62** | |

* **the context half = `refiner2 + refblock2 + capemb` = 2.02-2.11 s** (median
  **2.06 s**), i.e. 10.9 % of R and 2.2 % of the loop's 94 s of step walls [A].
* **`pre.refiner2` is 4.8 % of `pre.refiner`, not 40 %** [M]. Per call: 45.4 /
  46.3 / 48.4 / 46.7 ms per context block against 966 / 961 / 974 / 967 ms per
  image block. A refiner block is seven GEMMs of (S,3840) against the same
  0.66 GB of weights; at S=32 they run at ~250 GFLOP/s, at S=1024 at ~350, so
  the cost tracks S almost linearly and the caption is 1/32 of the image [A].
* **`pre.refblock2` is 1.31 s of one-off I/O, not per-step work**: 16 calls, of
  which 2 (step 0) do the 2 x 0.66 GB `np.load` and 14 are dict hits. It is the
  single largest piece of the context half and it is *pure I/O* [M].
* `pre.temb` — the brief groups `t_emb` with the context half — is **9 ms per
  image** (1.1 ms per step). It is the only genuinely step-dependent piece and it
  is not worth a schedule change [M].

---

## 2. Why it is a constant of the checkpoint

The delivered preamble (`t18_run.run()`'s `pre()`, now
`_preovl_preamble(...)["pre"]`) is, in order:

    adaln = T3.t_emb(step_t, H)                     # step-dependent
    xb    = T3.x_embedding(lat[b], H)               # latent-dependent
    xb    = 2 x full_block_host(xb, Wb, mod(adaln), x_cos, x_sin)
    cb    = T3.cap_embedding(H)                     # H ONLY
    cb    = 2 x full_block_host(cb, Wb, mod(adaln), c_cos, c_sin)

and `t3_real.modulation()` is

    if Wb["adaln_w"] is None:
        return None                                 # <- before adaln_vec is read

The refiner blocks have no `adaLN_modulation` tensors at all — the product's own
refiner cache says so in its manifest (`["norm_q", ..., "!adaln_w",
"!adaln_b"]`, where `!` marks the absent ones) — so `mod` is `None` for both
context blocks and **neither `step_t` nor the latent reaches the arithmetic**.
`cap_embedding` is `np.repeat(cap_pad_token, 32)`: the checkpoint's own learned
pad row, 32 times.

The whole optimisation rests on that one fact, so it is checked rather than
assumed, three ways: on the weights at run time (`_ctx_weights()` raises if
either block ever grows an adaLN, and the flag then falls back), in the gate
(`modulation() None for both context blocks: [True, True]`), and numerically —
the gate computes `cb` with a **random** adaln vector and with `None` and gets
the same bytes as the delivered path (`1cf3f9f8f7854de3` in all three cases).

---

## 3. The design, and the trap that reshaped it

### 3.1 What was implemented

`T22_PREOVL=1` does two things and nothing else:

1. **The two context blocks' refiner-cache read runs on a worker thread under
   the `--cache-all` startup pass.** `submit()` is called right before the
   payload section; the worker does `L.refblock("context_refiner.0/1")`, i.e.
   2 x 0.66 GB of `np.load`, measured at **0.57 / 0.60 s** in the two clean arms
   (1.18 s in the excluded slow-box arm); `_ctx_join()` at step 0's preamble
   waits **0.000 s** in every arm — the read is always finished
   before the loop reaches it.
2. **The buffer is memoised.** Step 0 runs the delivered `_ctx_step()` (so
   `cap_embedding` is called exactly where it always was — see 3.2) and steps
   1..7 read the same array. `[preovl] ... 1 computation + 7 reuse(s)`.

With the flag at its default 0: no thread, no pool, no branch taken, and
`pre()` runs `_ctx_step()`'s statements in the delivered order. The factory
is only a seam so that the gate can drive the same code twice in one process.

### 3.2 The trap: `loop_s` is anchored on the first `cap_embedding` call

The first implementation had the worker compute the whole context half. It was
byte-identical and it removed 2.0 s of buckets — and the product reported
**`loop_s` +23.5 s** for it. The reason is in `webui/bridge.py:197-207`: the
bridge monkeypatches `t3_real.cap_embedding`, and that wrapper emits the
`startup_done` event **on the first call**. `loop_s` is the span
`startup_done -> last step` (`webui/runner.py:1443`), so moving the first
`cap_embedding` ~2.5 s earlier re-defines the bucket: the whole remaining
startup wall is booked as loop.

| v1 arm (worker computes) | `loop_s` | step walls | `total_s` | `pre.refblock2` | `pre.refiner2` |
|---|---:|---:|---:|---:|---:|
| pv_a2 (delivered) | 90.70 | 93.91 | 125.52 | 1.274 | 0.741 |
| pv_b2 (v1) | **114.18** | **92.27** | 124.59 | 0 | 0 |

-1.6 s of real loop, +23.5 s of reported loop. **This is a second trap of the
same family as TRAP 1** (the fast/low memory switch): the number the campaign
tracks is produced by a hook that is sensitive to *when* a preamble function is
first called. Anyone moving preamble work should check the `startup_done`
anchor first.

The shipped design keeps `cap_embedding` where the delivered path calls it, so
`loop_s` and the step walls move together — measured in every pair below.

### 3.3 The epsilon question

CONTENTION-REPORT's rule is that the device pays for the **length and width of
a continuous parallel region**, not for bytes: an 18-thread 74-86 ms attention
pass costs the device +15.3 %, an 88.5 MB `bo.write` costs +0.0 %. The region
this change adds is **one thread doing `np.load`** — no BLAS, no numba, no GEMM,
nothing that spawns a team — for 0.6 s, and it runs entirely **before the loop**,
when the device is idle (the `--cache-all` pass is pure host work). The measured
device cost is in 5.5: every device column is within 0.2 ms/layer.

The cost it does have is host-side and lands on the startup pass: `[t13]
full-memory cache: 3.71 GB in 2.3 s` -> `2.9 s` in the same pair. That is the
honest price of moving 1.32 GB of reads, and it is paid **outside** `loop_s`.

---

## 4. The gate (host only, no XRT)

    py310 _scratch\preovl\gate_preovl.py

One process, the **real** refiner weights from the product's own refiner cache
(2.7 GB), the **real** `_preovl_preamble()` the product calls, the 512^2
schedule, 8 steps per arm. Six arms, compared on the sha256 of `adaln`, `xb`,
`cb` and the concatenated `u` for every step:

| arm | what it is | vs `ref` |
|---|---|---|
| `ref` | the DELIVERED statements, transcribed from git HEAD | — |
| `off` | the factory with `T22_PREOVL=0` (the shipped default) | **8/8 equal** |
| `on` | the flag on: worker preload + memo (the weights re-read from disk) | **8/8 equal** |
| `load` | as `on`, with a (1024,3840)@(3840,3840) f32 GEMM on the main thread *while* the worker reads | **8/8 equal** |
| `nowin` | the flag on with no startup window (low memory mode): the preload must be refused | **8/8 equal** |
| `negctl` | the memo perturbed by ONE ULP after its first use | 1/8 — **the gate bites** |

Plus two structural checks: the `adaln` test above, and
`no-window path: PASS (preloaded=False, hits=7, thread=no pool)` — in low
memory mode no thread is created, the preload is refused, and the memo still
removes 7 of the 8 computations.

`off` vs `ref` is the "flag off changes nothing" proof: the delivered
statements moved into `_ctx_step()` produce the same bytes as the
transcription of HEAD.

---

## 5. The device A/B

### 5.1 Protocol

Two interleaved pairs, one launcher at a time, `_scratch/preovl/run_preovl.ps1`
(`-GlueProf 1` on **both** arms, so the ledger decomposes the wall), with the
delivered env plus nothing but `T22_PREOVL=1` on arm B. Per run the script
records, before it starts: the live `python.exe` count and core use (a live
python that is not a device job is a co-tenant and needs `-AllowCotenant`), the
commit headroom, the memory mode the run actually got, and any **foreign job
directory created while the arm ran**. Both arms of both pairs: **fast mode,
co-tenant none, foreign jobs none**.

### 5.2 The arms

| tag | arm | `loop_s` | `total_s` | startup | step walls | mode | latents | step 0 |
|---|---|---:|---:|---:|---:|---|---|---:|
| pv_a3 | A | 91.325 | 125.488 | 32.32 | 94.331 | fast | `2abf91331200` | 13.99 |
| pv_b3 | B | **88.586** | 124.143 | 33.58 | 91.705 | fast | `2abf91331200` | 12.59 |
| pv_a4 | A | 90.978 | 124.896 | 31.98 | 93.743 | fast | `2abf91331200` | 13.81 |
| pv_b4 | B | **88.804** | 124.856 | 34.05 | 91.929 | fast | `2abf91331200` | 12.56 |
| pv_a1 | A | 90.804 | 125.567 | — | 93.998 | fast | `2abf91331200` | 14.02 |
| pv_a2 | A | 90.697 | 125.523 | — | 93.914 | fast | `2abf91331200` | 14.24 |
| pv_a6 | A | 98.578 | 134.308 | — | — | fast | `2abf91331200` | 15.61 |
| pv_b5 | B | 95.623 | 143.150 | — | — | fast | `2abf91331200` | 15.22 |

`pv_a1/a2` are extra delivered-mode runs (used for the sizing table in 1);
`pv_a6` and `pv_b5` are **excluded from the timing**: every bucket in them is
7-10 % slower (`pre.refiner` 16.15/17.09 against 15.4-15.6, `blk.ffn` 28.0
against 26.1, `blk.norm1` +13 %), there was no foreign device job and no
co-tenant python at launch, so it is a host-side slowdown that landed on the
whole run, not an effect of the flag. They are kept because they are two more
independent confirmations that the flag produces the identical latent.

### 5.3 Pair ratios of `loop_s`

    [pair 1] pv_a3 91.325 -> pv_b3 88.586   ratio 0.9700  (-2.74 s)
    [pair 2] pv_a4 90.978 -> pv_b4 88.804   ratio 0.9761  (-2.17 s)
    geomean ratio B/A = 0.9731  (-2.69 %);  mean -2.46 s

The four delivered-mode fast runs span **90.70-91.33 (0.63 s)**, so this A/B
**resolves**: the effect is 3.5-4x the A-arm spread and both B arms sit below
every A arm. (OVL-REPORT's design-1 A/B could not do this: its A arms spanned
2.3 s against a 2.3 s effect.)

`total_s` moves less than `loop_s` — -1.35 s and -0.04 s (mean -0.69 s) —
because the startup grows by 1.26/2.07 s: +0.4-0.6 s of that is measured
contention in the `--cache-all` pass (2.3 -> 2.9 s) and +0.4 s in the noise
refiner's own cache read; the rest is startup-phase scatter (two identical A
runs differ by 0.34 s there). The preload converts ~1.3 s of startup into
~1.3 s of loop; the loop is the metric this campaign tracks.

### 5.4 The ledger: where the 2.2 s came from **[M]**

| bucket (s) | a3 | b3 | a4 | b4 | delta (mean) |
|---|---:|---:|---:|---:|---:|
| `pre.refiner` (image half) | 15.581 | 15.070 | 15.468 | 15.291 | -0.34 (drift) |
| `pre.refiner2` (context half) | 0.774 /16 | **0.100 /2** | 0.747 /16 | **0.092 /2** | **-0.66** |
| `pre.refblock2` (its weight read) | 1.335 | **0.000** | 1.306 | **0.000** | **-1.32** |
| `pre.capemb` | 0.005 /8 | 0.006 /1 | 0.042 /8 | 0.005 /1 | -0.02 |
| `pre.refblock` (image weights) | 1.151 | 1.262 | 0.870 | 1.272 | +0.26 (drift) |
| **PREAMBLE total** | **19.036** | **16.620** | **18.622** | **16.850** | **-2.09** |
| `blk.ffn` | 26.062 | 26.148 | 26.127 | 26.192 | +0.08 |
| `blk.ffnpost` | 2.221 | 2.195 | 2.185 | 2.210 | 0.00 |
| `blk.norm1`/`norm2` | 0.954 | 0.954 | 0.925 | 0.969 | +0.02 |
| `step.final`, `step.other`, `step.us` | 1.05 | 0.98 | 0.99 | 0.96 | -0.05 |
| **ledger walls (the loop)** | **94.331** | **91.705** | **93.743** | **91.929** | **-2.22** |

Three things to read off it:

1. **`pre.refblock2` 1.31 -> 0 and `pre.refiner2` 0.75 (16 calls) -> 0.10
   (2 calls).** The 2 remaining `refiner2` calls are step 0's, which is where
   the delivered path computes the buffer for the first time; the other 14 are
   the memo hits. `pre.capemb` 8 calls -> 1 for the same reason.
2. **Nothing else moved.** `blk.ffn` +0.08 s over 240 dispatches,
   `blk.ffnpost` 0.00, the norms +0.02, `step.final`+`step.other`+`step.us`
   -0.05. The device columns below are identical to +-0.2 ms/layer, so d is
   zero at this resolution again: a single-threaded `np.load` beside a device
   dispatch is below it, exactly as the 88.5 MB `bo.write` was.
3. **The residual is 0.000 s in all four arms** — the ledger still closes, so
   nothing leaked out of a bucket into `step.other`.

### 5.5 Section medians (240 samples per column per run) **[M]**

| column (ms/layer) | a3 | b3 | a4 | b4 |
|---|---:|---:|---:|---:|
| `proj_qkv` | 0.0 | 0.0 | 0.0 | 0.0 |
| `rope+build` | 74.0 | 73.0 | 73.0 | 73.0 |
| `attn` | 81.0 | 81.0 | 81.0 | 81.0 |
| `proj_o` | 29.0 | 29.0 | 29.0 | 29.0 |
| `pack` | 5.7 | 5.5 | 5.6 | 5.7 |
| `wgu` | 4.3 | 4.3 | 4.35 | 4.3 |
| `gu` | 55.9 | 55.9 | 56.05 | 56.0 |
| `dn` | 38.1 | 38.1 | 38.1 | 38.1 |
| `read` | 2.4 | 2.2 | 2.3 | 2.3 |
| `dec` | 1.1 | 1.0 | 1.1 | 1.0 |
| `[devkill] total` | 107.8 | 107.5 | 107.9 | 107.8 |

Every device column is within 0.2 ms/layer between arms: **the flag costs the
device nothing measurable** (epsilon = 0 at this resolution).

### 5.6 Latents

    pv_a1 2abf91331200   pv_a2 2abf91331200   pv_a3 2abf91331200
    pv_a4 2abf91331200   pv_a6 2abf91331200   pv_b3 2abf91331200
    pv_b4 2abf91331200   pv_b5 2abf91331200
    ALL ARMS IDENTICAL (2abf91331200) -- no arithmetic moved

`2abf91331200` is the delivered **fast-mode** baseline OVL-REPORT records, so
these runs are also a cross-check that the delivered path is where it was.

---

## 6. What it is worth, and what is left

* **Measured: -2.46 s of `loop_s` per 8-step image (geomean -2.69 %), -0.69 s
  of `total_s`, 0.0 of device time, 0 of arithmetic.** [M]
* **The preamble is now 16.7 s and 15.1 s of it (90 %) is the noise refiner**
  [M], which is `x_embedding(lat_i)` followed by two blocks: it consumes the
  latent and is therefore *on* the chain. Its 1.15-1.27 s one-off weight read is
  the only other movable piece — and it happens **before** the `startup_done`
  anchor, so moving it into the startup window would buy `total_s` (about -0.8 s
  [A]) and **nothing** on `loop_s`. Not implemented here, on purpose: this report
  is about one attributable change.
* **The general lesson for this graph**: the `loop_s` bucket is anchored on the
  first `cap_embedding` call (`webui/bridge.py:206`). Any change that moves
  that call re-defines the metric by the whole remaining startup wall — 23.5 s
  in the v1 arm. Check it before believing a loop number.
* **A note on the brief's premise**: "step i+1's context half under step i's
  layer loop" is legal but strictly worse than what the dependency allows. The
  half is step-invariant, so the cheapest correct schedule is *one* computation
  under the startup (no device work in flight, one short single-threaded
  region, step 0 included) rather than 7 overlapping regions under the loop, each
  paying epsilon beside a device dispatch.

---

## 7. Files, flags, reproduce

| file | status | what |
|---|---|---|
| `_scratch/image/t18_run.py` | **changed** (+271/-34 vs HEAD, all of it this work) | flag `T22_PREOVL`; `_preovl_preamble()` (the preamble as a factory: `_ctx_step` = the delivered statements, `_ctx_weights` = the preload, `submit`/`_ctx_join`/memo/`report`); `run()` calls it and nothing else changed |
| `_scratch/preovl/gate_preovl.py` | **new** | host-only byte gate, 6 arms + negative control (section 4) |
| `_scratch/preovl/run_preovl.ps1` | **new** | one A/B arm of the delivered harness, with the device preflight, the memory-mode gate and the foreign-job check |
| `_scratch/preovl/analyze.py` | **new** | pair ratios, the `[t22g]` ledger table, section medians, latents |
| `_scratch/preovl/probe_env.py` | **new** | the environment facts the design needs (numba threading layer; a f32 GEMM called from a worker thread is byte-identical) |
| `PREOVL-REPORT.md` | **new** | this file |

| flag | file | default | effect |
|---|---|---|---|
| `T22_PREOVL` | t18_run.py | **0** | master switch: preload the two context blocks' weights on a worker under the startup + memoise the context half |

Run byproducts (safe to delete): `_scratch/preovl/arm_*.txt`,
`_scratch/preovl/pv_*.bridge.log`, `_scratch/bigpipe2/pv_*.run.log`, and the
`webui/jobs/20261102-20*` job dirs of these eight runs.

```powershell
cd <checkout>

# 1. the host-only gate (no device; ~2 min)
py310 _scratch\preovl\gate_preovl.py

# 2. one interleaved pair (device; one launcher at a time; ~2.2 min each)
powershell -NoProfile -ExecutionPolicy Bypass -File _scratch\preovl\run_preovl.ps1 -Arm A -Tag pv_a3 -Steps 8 -GlueProf 1
powershell -NoProfile -ExecutionPolicy Bypass -File _scratch\preovl\run_preovl.ps1 -Arm B -Tag pv_b3 -Steps 8 -GlueProf 1

# 3. the numbers
py310 _scratch\preovl\analyze.py pv_a3 pv_b3 pv_a4 pv_b4
```

Arm B is arm A plus `T22_PREOVL=1` and nothing else.

The whole measurement in one line:

```powershell
cd <checkout>; powershell -NoProfile -ExecutionPolicy Bypass -File _scratch\preovl\run_preovl.ps1 -Arm A -Tag pv_a3 -Steps 8 -GlueProf 1; powershell -NoProfile -ExecutionPolicy Bypass -File _scratch\preovl\run_preovl.ps1 -Arm B -Tag pv_b3 -Steps 8 -GlueProf 1; py310 _scratch\preovl\analyze.py pv_a3 pv_b3
```
