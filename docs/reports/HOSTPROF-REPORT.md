# HOSTPROF-REPORT — the Pareto of NON-KERNEL time, 8-step 512²

One env-gated profiler, one 8-step run (240 layer-steps), every host call counted.
**No product file was edited; the instrument is off unless `T22_HOSTPROF=1`.**

The profiled run is **byte-identical** to the unprofiled reference:
`hp_p1_latents.npy` sha256 prefix `2abf91331200d069` == `w11412507efe9p0_latents.npy`
(the unprofiled harness run `hp_h1`, same env, same day) == the sha that ~35 other runs of this
config produced today; and every per-layer `a / xm / y` absmax trace matches line for line
(`imaggers/hostprof/_latent_cmp.py` re-checks it). So the table below is the product's own host
path, not a model of it.

---

## 0. TL;DR

| | |
|---|---|
| per layer-step (this box, today) | **374 ms** = 158 device + **143 host** + 66 refiner + 7 other |
| in-loop non-kernel total | **34.37 s** of the 89.8 s loop (38 %), 143.2 ms/layer |
| 80 % of it needs | **10 labels**, not 5 (the top 5 are only 63 %) |
| biggest single lever | the host attention: `torch.matmul` 34.1 + `torch.softmax` 23.9 = **58 ms/layer (40 %)** |
| outside the loop | a **19.9 s one-off B repack** and **15.8 s of refiner** — 30 % of the 122 s process wall |

The 71.8 ms/layer "misc" from the brief is now resolved into 20 rows below; the single
largest hidden item in it was **`rope+build` 63 ms/layer** at the time of the brief, which this
run measures as **`rope.rope_fused` 4.6 + `bf16.widen_t` 1.9** because the fastglue attention
hook replaces the delivered `heads()` transposes (see section 9, correction 1).

---

## 1. What was measured, and the two caveats

* Run `hp_p1`: 8 steps × 30 layers, 512×512, seed 7, py3.10 device child.
* Env (the product's fast path, exactly the brief's): `T22_FASTGLUE=1 T22_FAST_GLUE=1 T22_GLUE2=1 T22_GLUE3=1`
  `T22_DEVKILL=1 T22_HOSTKILL=1 T22_BIGM_DEVHANDOFF=1 ZIMAGE_BIGM=1 ZIMAGE_BIGM_PROJ=1`
  `ZIMAGE_CHUNK_PIPE=0 ZIMAGE_CHUNK_THREADS=16 T22_REF_SMNB=1 T22_GLUE_PROF=0`.
* Loop wall: `[t3d] wall 89.80 s` (the brief's reference is 84.65 s; three same-config runs on
  this box today gave 88.60 / 89.80 / 92.33 s, so the box is ~5 % slower than on 2026-11-01 and the
  run-to-run band is ±2 %).

**Caveat A (one run, not two).** The brief asks for two runs for a spread. This box is shared and the
device was handed to another agent mid-task, so run 2 was not taken; section 6 gives the spread from the three
same-config runs and from the per-label p10/p90 instead, and section 11 gives the one-line command.

**Caveat B (one row is an instrument artefact).** The profiler wrapped `t18_run.block_step` before
`chunked.install()`, and fastglue's block hook does `inspect.getsource(T.block_step)` — it saw the
wrapper, refused ("block=hunk-failed: block_step source does not start at the def") and the run
continued with **7/8 hooks**. That is fixed in `hostprof.py` (block_step is now patched in "phase 2",
after every install), but it means row 3 (`loop.block_step` 10.26 ms) is *the price of the four
missing fused norm hunks*, not a cost the product pays. Section 5 gives the corrected ranking.

---

## 2. The instrument

`imaggers/util/place/hostprof.py` — a nesting-aware (inclusive **and** exclusive) per-call timer.
83 symbols patched, **0 missed, 0 errors**, **543 ns** of overhead per profiled call
(measured: 200 k calls, wrapped vs unwrapped, `imaggers/hostprof/_overhead.py`), i.e. ≈16 ms over
this run's ~30 k calls — **< 0.02 %**, far below the ±2 % run-to-run band. The profiled run was in
fact 0.5 % *below* the mean of the three same-config runs, i.e. the cost is invisible.

Coverage: the numba codecs and the fused-glue kernels (norm/rope/bf16/pack/decode/proj_read_cv), the
attention, every `pyxrt` BO write/read/sync and every `run.wait()`, the allocation and copy
primitives (`np.ascontiguousarray/empty/zeros/concatenate/copyto/frombuffer/save`),
`torch.matmul/softmax/from_numpy`, `builtins.print`, the python loop bodies
(`block_step`, `fast_bigm_ffn`, `block`, `bigm_attention`) and the per-step preamble.

---

## 3. Accounting closure (no residual guessing)

| | s | ms/layer |
|---|---:|---:|
| device wait (`pyxrt.run.wait`) — **the NPU computing** | 37.91 | 158.0 |
| **in-loop host** | **34.37** | **143.2** |
| in-loop total | 72.28 | 301.2 |
| per-step preamble (refiner, t_emb, x_emb, modulation, final_layer) | 15.78 | (1 972 ms/step) |
| startup one-offs (B repack 19.91 + device open/warm/preload 7.75) | 27.66 | — |
| **`t18_run.run()` wall** | **114.90** | — |
| residual (code no wrapper saw) | **−0.82** | **−0.7 %** |

In-loop closes to −0.7 % of the run wall, so the self column below *is* the wall, not an estimate.

---

## 4. The table (self = exclusive, sorted, 43 in-loop labels)

```
 #  label                    count   ms/layer  share%   cum%   med us   p90 us
 1  torch.matmul              1440      34.10    23.8   23.8    7252    8695
 2  torch.softmax              720      23.85    16.7   40.5   10557   12368
 3  loop.block_step *          240      10.26     7.2   47.6  296811  308442
 4  att.torch                  240      10.25     7.2   54.8   67014   73094
 5  bo.write                  2645      10.06     7.0   61.8     568    2166
 6  ffn.block                  240       6.41     4.5   66.3  107368  113471
 7  cdec.proj_read_cv          241       6.07     4.2   70.5    5742    7500
 8  bo.read                   1200       6.01     4.2   74.7    1118    1676
 9  pack.pack_a                242       5.46     3.8   78.5    5296    5922
10  norm.norm_f32               961       5.18     3.6   82.2    1294    1374
11  rope.rope_fused             481       4.64     3.2   85.4    2265    2793
12  att.attention               240       3.82     2.7   88.1  170031  178805
13  glue.absmax                1443       2.85     2.0   90.1     930    1076
14  bo.sync                    3605       2.47     1.7   91.8     109     223
15  bf16.widen_t                241       1.87     1.3   93.1    1806    2093
16  att.dispatch_proj           960       1.54     1.1   94.2   20522   23983
17  bf16.rne                    481       1.46     1.0   95.2     765     895
18  np.ascontiguousarray       8328       1.17     0.8   96.0     1.8     5.4
19  cdec.decode_y               241       1.01     0.7   96.7     919    1197
20  glue.make_sbg               240       0.98     0.7   97.4    1028    1144
21  np.copyto                    30       0.86     0.6   98.0    6344    9503
22  bf16.bf16_of                241       0.69     0.5   98.5     623     679
23  ffn.bigm_ffn2               240       0.55     0.4   98.8  108529  114783
24  ffn.dispatch                240       0.25     0.2   99.0   35580   40554
25  np.empty                   1647       0.24     0.2   99.2      35      52
26  io.print                    904       0.17     0.1   99.3      48      58
   ... 17 more rows, together 1.1 %   (see the CSV)
```

`*` row 3 is the 7/8-hook artefact of section 1B — see section 5.

Raw artefacts: `imaggers/hostprof/hp_p1_90052.json` and `.csv` (per-label, with per-call
median/p10/p90/max and the byte count), `imaggers/hostprof/pareto_hp_p1.csv` / `.json`
(ranked, with cumulative %).

---

## 5. The Pareto, and the corrected view

**As measured** (7/8 hooks): the 80 % cut lands at **11 labels (81.7 %)**.

**Corrected for a normal 8/8 run.** The four fused block hunks would replace row 3 with ~1.5 ms/layer
(they fold the six 16 MB in-place `np.multiply/np.add` passes into the norm kernel). Subtracting
8.8 ms/layer from both the row and the total (34.37 → 32.27 s = 134.5 ms/layer):

| # | label | ms/layer | share | cum |
|---|---|---:|---:|---:|
| 1 | `torch.matmul` | 34.10 | 25.4 % | 25.4 % |
| 2 | `torch.softmax` | 23.85 | 17.7 % | 43.1 % |
| 3 | `att.torch` (copies inside `_attention`) | 10.25 | 7.6 % | 50.7 % |
| 4 | `bo.write` | 10.06 | 7.5 % | 58.2 % |
| 5 | `ffn.block` (python + thread + submits) | 6.41 | 4.8 % | 63.0 % |
| 6 | `cdec.proj_read_cv` | 6.07 | 4.5 % | 67.5 % |
| 7 | `bo.read` | 6.01 | 4.5 % | 72.0 % |
| 8 | `pack.pack_a` | 5.46 | 4.1 % | 76.1 % |
| 9 | `norm.norm_f32` | 5.18 | 3.9 % | 80.0 % |
| 10 | `rope.rope_fused` | 4.64 | 3.5 % | 83.5 % |

**So the answer to "which 5 items are 80 % of the host time" is: none — there are 10.**
The curve is flat: the top item is 25 %, the tenth 3.5 %. Any plan that only fixes one thing
caps at a quarter of the host cost; the first five fixes together reach 63 %.

---

## 6. Run-to-run spread

No second profiled run (section 1A), so the spread is bounded from three directions:

* **whole-run**: three same-config runs today — `[t3d] wall` 88.60 s (unprofiled harness),
  89.80 s (this profiled run), 92.33 s (unprofiled harness) → **±2.0 %** band (mean 90.24 s); the
  profiled run sits 0.5 % below the mean and 2.7 % below the slowest, i.e. no measurable cost.
* **step to step inside the run**: 10.11 / 10.66 / 10.82 / 10.90 / 10.98 / 10.78 / 10.81 s after the
  warm step → σ = 0.28 s = **2.6 %**.
* **call to call inside the run** (per-label p10/median/p90 in the CSV): `torch.matmul`
  865 / 7252 / 8695 µs, `torch.softmax` 830 / 10557 / 12368, `cdec.proj_read_cv` 5592 / 5742 / 7500,
  `pack.pack_a` 5025 / 5296 / 5922, `bo.read` 895 / 1118 / 1676. The big kernels are stable to a few %;
  `bo.write` is not (324 / 568 / 2166 µs) because it is a mixture of 0.7 MB and 88.5 MB writes — always
  compare it as a total, never as a median.

**No ranking in section 5 is inside the noise**: #1 and #10 differ by 7.4×, and #5 vs #6 differ by 6 %.

---

## 7. The top five — one fix each

### 1 + 2. `torch.matmul` 34.1 + `torch.softmax` 23.9 = 58.0 ms/layer (43 % of host)
Two separable sub-fixes, **both already written and measured in `webui/chunked.py`**, both behind an env var:

* **`T22_ATTN_E=1`** — QK^T on AVX512-BF16 + the fused in-place numba f32 softmax
  (`bcore_tail.sm_bf16_f32`, zero-copy through a uint16 view) + the f32 PV matmul. The file's own
  isolated medians, per 512-row block: qk f32 11.0 → **4.56 ms**, softmax torch-f32 30.0 → **16.9 ms**,
  pv kept at 9.89. **Estimate here: −20 to −22 ms/layer ⇒ −5 s per image.**
  *Risk:* QK in bf16 changes the arithmetic; it must pass `quality_gate.py` (corr ≥ 0.9953) and the
  chain's absmax trace must be re-baselined. *Cost:* one env var, zero code.
* **`T22_ATTN_A2=1`** — the **exact** fallback: f32 matmul + the same softmax fused in place
  (no 130 MB copy out). Recorded in-file as "92.9 ms/layer vs 107.9, corr 1.000000000, relrms 1.3e-7".
  **Estimate: −8 to −11 ms/layer ⇒ −2 to −2.6 s**, no numeric risk at all.

*Why cheap:* nothing to write, nothing to compile, both variants were already benched in-tree.
*Why risky:* only the bf16 arm can move the image numerically; and the attention file is the one
another agent is A/B-ing right now, so land the env-var flip, not an edit.

### 3. `att.torch` 10.25 ms/layer (copies inside `_attention`)
What it is (measured, not guessed): `np.empty((seq,heads,hd))` once per layer plus
`out[q0:q1] = o.numpy().transpose(1,0,2)` — a **strided 7.9 MB store per query block** (3 blocks),
plus three `np.ascontiguousarray` before `torch.from_numpy`.
**Fix:** keep one persistent `(heads, seq, hd)` output buffer and have the PV result written straight
into `out[:, q0:q1]` through a transposed *view* (`np.moveaxis`, or transpose on the torch side before
`.numpy()`), so the per-block transpose copy disappears; q/k/v are already contiguous, so
`torch.from_numpy(qh)` is free.
**Estimate: −6 ms/layer ⇒ −1.4 s.** *Risk:* low — pure movement of the same bytes into the same
positions, gated by the latents. *Why cheap:* ~15 lines in one function.

### 4. `bo.write` 10.06 ms/layer (52.6 GB per image!)
11 writes per layer totalling **240 MB**: FFN W_gu 88.5 + A 8.1 + W_dn 44.2 + 4 × (X 8.1 + W 16.6).
That is **24 GB/s**, i.e. already at memory bandwidth — the only fix is to write **less**.
**Fix:** make the payloads BO-resident for the whole run, exactly like T13 U-4b `--bo-resident` already
does for the delivered FFN's BG/B3. The bytes are identical on all 8 steps, so **7 of every 8 writes are
pure duplication**. Cheapest half: the four projection W payloads, 4 × 16.6 MB/layer = **−2.9 ms/layer
(−0.7 s)** for 2.0 GB of device BOs. Full: + the FFN gu/dn, **−8.3 ms/layer (−2.0 s)** for ~6 GB.
*Risk:* medium-high — 6 GB of device BOs must exist, and a wrong group-id binding is a **silent**
numeric error (the F-08 rule). *Why cheap to try:* the projection-W half is 120 BOs of one size, and
the existing `--bo-resident` code is the template.

### 5. `ffn.block` 6.41 ms/layer (python + a per-layer thread + 2 submits)
Every layer builds `threading.Thread(target=_write_sync, ...)`, starts it, joins it, and makes two
`im["k"](...)` submissions; the thread exists to hide the 44 MB W_dn write under the gu dispatch, and
`wdn=0.0` in the product's own debug line says the hiding works — the *thread machinery* is the cost.
**Fix:** keep one module-level `ThreadPoolExecutor(max_workers=1)` (T22_PFILL already uses that pattern)
plus a pre-created `threading.Event`, so the per-layer cost is a submit + an event wait instead of a
thread create/join.
**Estimate: −2 to −3 ms/layer ⇒ −0.5 to −0.7 s**, and it is the least certain number in this report —
instrument the thread ops first (they are not currently wrapped).
*Risk:* low numerically, medium behaviourally (the write must still complete before the dn dispatch reads it).

### Next five, if the first five land
`cdec.proj_read_cv` 6.07 — **the clearest remaining win**: one call/layer, 5.74 ms median, and it is
the only kernel in the chain that is *not* parallel (`_widen_abs_k` runs with `_threads(0)`). 4.05 M u16 →
f32 plus an absmax is 0.7 ms of memory traffic at the measured 24 GB/s; prange it like `_widen_kp` and
reduce the absmax from per-thread partials (**same value**: a selection, not arithmetic) → **−4.5 ms/layer,
−1.1 s**. Then `bo.read` 6.01 (1200 reads of 737 KB–8 MB; `bo.read` hands back a uint8 array, so every
call already copies — reading through `bo.map()` into a preallocated buffer would halve it),
`pack.pack_a` 5.46 (8 MB permute, parallel over only 11 row blocks — split the prange over the k-groups
instead), `norm.norm_f32` 5.18 (4 calls/layer at 1.29 ms; the 8/8 hunks already replace 3 of them),
`rope.rope_fused` 4.64 (2 calls/layer at 2.27 ms for 16 MB in / 16 MB out — 4× off the memory bound;
`T22_FASTGLUE_THREADS` is 8 on a 20-thread box, so raise it and measure).

---

## 8. Outside the layer loop — 45 s of the 122 s process wall

| item | s | note |
|---|---:|---|
| **B repack** (`glue.cached_payloads`, first call) | **19.91** | re-derives the same 3.98 GB permutation on **every** run |
| **refiner** (`step.refiner` + `t3_rms` + `t3_apply`) | **15.78** | 1 972 ms/step × 8 |
| device open + `--warm` + payload-preload join | 7.75 | |
| bridge/driver startup before `run()` | ~7 | |

* The 19.9 s repack is a pure function of the weights. `wpack` already caches the *projection* images
  to a 1.85 GB file; the same treatment for the repacked BG/B3 (**19.9 s → one background read**) is the
  single biggest non-loop win available, and it is bigger than any single loop fix.
  `T22_DEVKILL_BREPACK=0` is *not* the answer — it only moves the 19.9 s into step 0.
* The refiner is 13 % of the process wall and `T22_REF_BF16=1` is already measured at **−4.5 s**; it is
  not on by default. That is the cheapest 4.5 s in the whole system.

---

## 9. Corrections to the brief (all measured today)

1. **`misc 71.8 ms/layer` did not survive.** With the fastglue hooks actually installed, the delivered
   `heads()` transposes and the 63 ms `rope+build` collapse to `rope.rope_fused` 4.6 + `bf16.widen_t` 1.9;
   what is left of "misc" is `bo.write/read/sync` 18.5, `proj_read_cv` 6.1, `pack_a` 5.5, `norm_f32` 5.2,
   the `block_step` python body 10.3 (only when the block hook is missing) and 17 tiny rows.
2. **`T22_FASTGLUE` and `T22_FAST_GLUE` are two different switches.** `-FastGlue 1` on `run_arm.ps1`
   only sets `T22_FAST_GLUE` (t18_run's rms/rope buffers); the 8 fused-glue hooks are gated on
   `T22_FASTGLUE`. Without both, the run is **+37 s** (measured: 122.8 s vs 88.6 s loop) and nothing
   says so except the `[fastglue] installed 0/8` line in the job log.
3. **The block hook fails open on any wrapper.** `build_block_step` refuses when
   `inspect.getsource(T.block_step)` does not start with `def block_step(`, and the run then keeps
   **+9 ms/layer** for the four fused norm hunks while still printing `installed 8/8` (it prints
   `block=hunk-failed: ...` one line earlier). Anything that decorates `block_step` — a profiler, a
   tracer, an `lru_cache` — silently costs 2.2 s per image.
4. **This box is ~5 % slower than on 2026-11-01** (88.6 s vs 84.65 s loop for the same config), power
   plan "Balanced", AC online. Compare *shares*, not absolute ms, against the brief's numbers.

---

## 10. Files

**New (nothing else was modified):**

| file | what |
|---|---|
| `imaggers/util/place/hostprof.py` | the profiler (83 symbols, exclusive+inclusive, per-thread, JSON/CSV dump). Off unless `T22_HOSTPROF=1` |
| `imaggers/util/place/hostprof_site/sitecustomize.py` | injection point for interpreters that DO run `site` (the py312 server). Inert without the env var |
| `imaggers/devkill/run_arm_prof.ps1` | `run_arm.ps1` + the correct `T22_FASTGLUE` env + the profiler switches (`-Prof 0` = control arm) |
| `imaggers/devkill/hostprof_report.py` | reads 1..N artifacts → closure, spread, Pareto, CSV+JSON |
| `imaggers/hostprof/run_bridge_prof.py` | runs `webui/bridge.py` in-process with the profiler armed (same argv, same env, same lock) — needed because the py310 GAIA interpreter is an **embeddable** dist (`python310._pth` ⇒ `site` is never imported and `PYTHONPATH` is ignored) |
| `imaggers/hostprof/run_bridge_prof.ps1` | its harness (device lock, env, stderr dup2) |
| `imaggers/hostprof/hp_p1_90052.json` / `.csv` | **the raw table** |
| `imaggers/hostprof/pareto_hp_p1.csv` / `.json` | the ranked Pareto with cumulative % |
| `imaggers/hostprof/_preflight.py`, `_inspect.py`, `_overhead.py` | resolve the SPEC against the real modules; print one artifact; measure the 543 ns |
| `HOSTPROF-REPORT.md` | this file |

**Byproducts of the runs** (safe to delete): `imaggers/image/hp_p1_latents.npy`,
`imaggers/image/hp_p1_ckpt*.{npy,json}`, `imaggers/hostprof/hp_p1{,.err}.log`.

---

## 11. Reproduce

```powershell
# control arm (no profiler, same env)  -> loop ~89 s
powershell -NoProfile -ExecutionPolicy Bypass -File imaggers\devkill\run_arm_prof.ps1 -Tag hp_c1 -Steps 8 -GlueProf 0 -FastGlue 1 -Glue2 1 -Glue3 1 -Prof 0

# profiled arm -- take the device lock first, i.e. only when YOU own the NPU
powershell -NoProfile -ExecutionPolicy Bypass -File imaggers\hostprof\run_bridge_prof.ps1 -Tag hp_p2 -Job webui\jobs\20261102-114125-07efe9 -Steps 8 -Prof 1 -Threads 16

# report (1..N runs; 2 runs add the spread column)
py imaggers\devkill\hostprof_report.py --runs hp_p1 hp_p2 --nlayer 240
```

Note: `run_arm_prof.ps1` cannot arm the profiler inside the *device* child (embeddable python, section 10),
so it measures the control arm only; `run_bridge_prof.ps1` is the profiled one.
