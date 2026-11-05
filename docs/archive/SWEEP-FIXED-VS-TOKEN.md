# SWEEP — fixed overhead vs per-token cost, measured

**Date** 2026-11-02 21:14–21:50 (+08:00) · **Window** one continuous block, no other
launcher alive (python.exe count checked to 0 before **every** run) · **Host** 20 logical
cores, 31.29 GB RAM, commit headroom 10.5–12.1 GB → **memory mode `fast` in every run**
(the `--no-cache-all` trap never fired; the reason string is quoted per run below).

**What this is.** The one measurement the project never had: how much of an 8-step run is
*fixed* and how much *scales with token count*. Six resolutions were asked for; **only two of
them (256², 512²) run as asked**. 384² and 640² die in the fast configuration, 640² runs on
the un-glued product path instead, and 768²/1024² cannot run in either configuration — three
distinct hard limits, each reproduced from the driver log (§6.4). Nine completed arms in the
fast configuration (five sizes, two of them repeated) plus three product-mode arms that reach
N=1632.

Everything below is a number a run wrote. Nothing is extrapolated except where it says so.

---

## 0. Verdict in six lines

1. **44.3 s of every loop is fixed.** `loop_s = 44.3 s + 46.3 ms × N` (R² = 0.987) — at 512²
   (N=1056) the loop is **44.3 s that does not care how big the image is + 48.9 s that does**
   (48 % / 52 %); at N=480 the same 44.3 s is 66 % of the loop.
2. **The whole run is ~64 % fixed**: `total_s = 83.2 s + 44.3 ms × N` (R² = 0.916).
3. **Only one segment really scales**: the attention block
   (`14.8 s + 29.8 ms × N`, R² = 0.991, 68 % scaling). **FFN is 24.0 s + 4.9 ms × N**
   (R² = 0.986, **82 % constant**) and the **preamble is 7.3 s + 10.8 ms × N** (R² = 0.919,
   61 % scaling but only 19 s of the 93 s loop).
4. **Per layer, 2.2× the tokens buys 1.05× the time** — except attention: FFN 103.9 → 108.9 ms,
   projection+rope 68 → 73 ms, host attention **25 → 83 ms** (480 → 1056 tokens).
5. **The floor is 50.8 s**, measured at 128×128 (N=96, the smallest shape the product
   supports): 27.1 s startup + 21.6 s loop. **No resolution reaches 10 s** — the target is
   below the *fixed* cost alone at every N ≥ 0.
6. **Shrinking is not a lever; the fast path's shape lock is.** The devkill image set is
   sized for exactly 11 row-tiles (N ≤ 1056), so 640²/768²/1024² cannot run at all in the
   fast configuration — the sweep's upper half is not *slow*, it is *unreachable*.

---

## 1. Harness, window and mode for every run

Launcher: `_scratch/devkill/run_arm_size.ps1` — a **copy** of `run_arm.ps1`'s env block with
the size exposed (`run_arm.ps1` hardcodes `--h 512 --w 512` on line 25 and has **no size
flag**; `-?` does not exist, its `param()` block is the whole flag list, and `-Stag` is dead).
It calls the same driver `run_arm.ps1` calls (`_scratch/bigpipe2/run_job_py.py`), so the code
path is the delivered one; only the size is new.

```powershell
$env:T22_HOSTKILL="1" $env:T22_DEVKILL="1" $env:T22_BIGM_DEVHANDOFF="1" $env:T22_BIGM_DEBUG="1"
$env:T22_FAST_GLUE="1" $env:T22_GLUE2="1" $env:T22_GLUE3="1" $env:T22_GLUE_PROF=<0|1>
$env:T22_FASTGLUE="1"    # <- the switch run_arm.ps1 does NOT set; without it the 8 fused
                         #    hooks do not install (+37 s) while printing "installed 0/8"
$env:T22_REF_SMNB="1" $env:ZIMAGE_CHUNK_THREADS="16"
```

Every fast-mode run printed `[fastglue] installed 8/8 hooks (threads=8)` — the hooks were
live in all of them. (`threads=8` is the numpy pool, `T22_FASTGLUE_THREADS`; the torch
attention pool is the separate `ZIMAGE_CHUNK_THREADS=16` knob, `webui/chunked.py:85`.)

**Profiler.** `T22_GLUE_PROF=1` was on for every arm except one 512² control. It costs
**0.2 s in 94 s (0.2 %)**: 512² measured 94.15 s (prof 0) vs 93.94 s (prof 1), same latents
sha. It is on everywhere else because it is the only source of the per-step preamble
(`pre.*`) split — without it a run has no ledger at all.

**Determinism.** 448×256 twice → identical latents (`7864d22bb09b`); 512² with and without
the profiler → identical (`2abf91331200`). Product-mode arms are byte-*different* from
fast-mode arms (different kernel family) — cross-mode comparisons below are **cost only**.

---

## 2. The raw table

`startup`/`loop`/`total` are `job.timing.{startup_s,loop_s,total_s}`; the three segment
columns are the exclusive `[t22g]` ledger (sums to the loop wall, residual 0.000 s);
`s/step` is the mean of steps 1–7 (step 0 excluded: it carries first-use cost, +2.4 s).

| # | tag | shape | N | mode | startup s | loop s | total s | preamble s | FFN s | attn+ s | s/step |
|---|-----|-------|---|------|----------:|-------:|--------:|-----------:|------:|--------:|-------:|
| 1 | n96a | 128×128 | 96 | native | 27.11 | 21.61 | **50.80** | 5.39 | 6.40 | 10.95 | 2.286 |
| 2 | sw256a | 256×256 | 288 | native | 31.19 | 61.13 | **94.78** | 8.81 | 23.86 | 29.68 | 7.118 |
| 3 | n480a | 448×256 | 480 | chunked | 34.68 | 66.99 | **103.73** | 12.54 | 26.31 | 29.38 | 8.367 |
| 4 | n480b | 448×256 | 480 | chunked | 36.15 | 67.85 | **105.97** | 13.34 | 26.39 | 29.49 | 8.537 |
| 5 | n672a | 640×256 | 672 | chunked | 36.45 | 73.57 | **112.06** | 13.37 | 27.50 | 34.28 | 9.191 |
| 6 | n864a | 832×256 | 864 | chunked | 36.20 | 82.58 | **120.89** | 16.37 | 28.05 | 39.63 | 10.335 |
| 7 | sw512a | 512×512 | 1056 | chunked | 28.68 | 94.15 | **124.88** | (no ledger) | | | 11.852 |
| 8 | n1056b | 512×512 | 1056 | chunked | 39.28 | 93.94 | **135.74** | 19.29 | 29.29 | 47.05 | 11.830 |
| — | sw384a | 384×384 | 608 | chunked | 33.76 | **REFUSED** | — | — | — | — | — |
| — | n1248a | 608×512 | 1248 | chunked | 38.86 | **REFUSED** | — | — | — | — | — |

Product-mode arms (no `T22_DEVKILL`/`T22_HOSTKILL`, i.e. no fused glue, no 11-tile images —
they exist to reach N above the fast path's ceiling):

| # | tag | shape | N | mode | startup s | loop s | total s | preamble | FFN | attn+ | s/step |
|---|-----|-------|---|------|----------:|-------:|--------:|---------:|----:|------:|-------:|
| P1 | prod480 | 448×256 | 480 | chunked | 11.21 | 160.41 | **173.62** | 11.25 | 58.38 | 91.90 | 16.530 |
| P2 | prod1056 | 512×512 | 1056 | chunked | 12.06 | 229.26 | **243.25** | 19.06 | 68.52 | 142.50 | 25.060 |
| P3 | prod640 | 640×640 | 1632 | chunked | 12.66 | 314.17 | **328.77** | 29.06 | 80.56 | 205.21 | 35.412 |

Memory mode, per run (`job.memory_mode` + the reason string): **fast** in all 11 —
`auto: commit 10.5/10.9/11.0/11.3/11.8/12.0 GB >= 9.0 GB`. Core count: **20 logical** in all.
Runs 7 and 8 are the same shape twice: **loop 94.15 / 93.94 (0.2 % spread), total 124.88 /
135.74 (8.7 % spread)** — the spread is entirely `startup_s` (28.68 vs 39.28), and inside it
entirely the `B repack` prologue (14.5 vs 21.7 s). Runs 3 and 4: loop 66.99 / 67.85 (1.3 %).

---

## 3. The three fits

OLS on `y = F + c·N`, R² = 1 − SS_res/SS_tot. **Excluded points and why** are listed after.

### (a) `total_s = F_all + c_all·N`

| point set | F_all (s) | c_all (ms/token) | R² | n |
|-----------|----------:|-----------------:|---:|---:|
| fast mode, chunked arms 480–1056 | **83.19** | **44.29** | **0.9158** | 6 |
| fast mode, all completed arms 96–1056 | 64.24 | 67.08 | 0.8335 | 8 |
| product mode, 480–1632 | 106.32 | 134.68 | 0.9965 | 3 |

Residuals (chunked set, s): −0.72, +1.52, −0.90, −0.58, −5.09, +5.77. The ±5 s pair is the
two 512² runs — i.e. **`total_s` is limited by startup noise, not by the fit**. Fitting the
loop instead (§b) lifts R² from 0.916 to 0.987 with the same data.

### (b) `loop_s = F_loop + c_loop·N` — the fixed part *inside* the loop

| point set | F_loop (s) | c_loop (ms/token) | R² | n |
|-----------|-----------:|------------------:|---:|---:|
| fast mode, chunked 480–1056 | **44.32** | **46.26** | **0.9871** | 6 |
| fast mode, all 96–1056 | 31.88 | 61.45 | 0.8700 | 8 |
| product mode 480–1632 | 93.66 | 133.47 | 0.9964 | 3 |

Same law per step: `steady_step_s = 5.497 + 5.89 ms × N` (R² = 0.983, n=6). At 512² that is
**5.50 s/step fixed + 6.22 s/step scaling**.

`startup_s` fitted the same way gives **c = −2.4 ms/token, R² = 0.033** — flat, as it must be:
startup does not see N at all (it is weight/registration work). Its measured spread is
27.1–39.3 s and it is **pure box noise**, which is why it wrecks fit (a).

### (c) Per segment — which ones are constant, which scale

Fast mode, chunked arms **with a ledger** (5 points; `sw512a` excluded, `T22_GLUE_PROF=0`):

| segment | F (s) | c (ms/token) | R² | at N=1056 | verdict |
|---------|------:|-------------:|---:|----------:|---------|
| preamble / refiner (`pre.*`) | 7.32 | 10.79 | 0.9192 | 18.7 s | **mostly scaling** (61 % of it) |
| FFN (`blk.ffn`+`blk.ffnpost`) | **24.01** | **4.92** | 0.9859 | 29.2 s | **82 % constant** |
| attention block + norms + glue | 14.80 | **29.79** | 0.9912 | 46.3 s | **68 % scaling — the only strong scaler** |
| step scaffolding | 0.59 | 1.26 | 0.9379 | 1.9 s | negligible |
| **sum** | **46.7** | **46.8** | — | **96.1** | vs loop fit 44.3 + 48.9 = 93.2 |

Within the attention block, per layer at N=1056 (medians of 240 `[fastglue]` lines):

| part | ms/layer @480 | ms/layer @1056 | ×(2.2 tokens) | what it is |
|------|--------------:|---------------:|--------------:|------------|
| `rope+build` | 68 | **73** | **1.07×** | the q/k/v projections + rope + widen — 3 dispatches × 16.59 MB of weights |
| `attn` (host f32) | 25 | **83** | **3.32×** | torch SDPA on 30 heads — *the* token-scaled term |
| `proj_o` | 25 | **30** | **1.20×** | 1 dispatch × 16.59 MB |
| FFN `[devkill]` total | 103.9 | **108.9** | **1.05×** | gu 56 (88.47 MB W) + dn 38 (44.24 MB W) + pack/dec |

**Excluded**: `sw384a` and `n1248a` (loop never started — fit only completed arms);
`sw512a` from every *segment* fit (no ledger); the native arms (96, 288) from the
chunked fits, because N ≤ 304 selects a **different executor** (§6.1) — they are shown as
their own family and in the "all arms" rows above.

---

## 4. The 512² split

Measured: **124.88 s** (prof 0) and **135.74 s** (prof 1); fitted 129.97 s.

> **83.2 s of the 130 s is fixed; 46.8 s scales with tokens** — 64 % fixed, 36 % scaling.

Broken down (fitted total 129.97 s):

| bucket | s | share | what |
|--------|---:|------:|------|
| startup + tail | **38.9** | 30 % | fitted as F_all − F_loop; measured startup alone is 28.7–39.3 s (mean 34.0). Inside it: B repack 14.5–21.7, cache-all 4.6–5.6, devkill prewarm pack 2.7–3.6, refiner weights 2.4, wpack map+verify 1.2–1.4, registration+import ~1 |
| loop, fixed | **44.3** | 34 % | 5.50 s/step × 8: FFN weight stream, projection weight stream, preamble fixed part, dispatch latency |
| loop, scaling | **48.9** | 38 % | 6.22 s/step × 8: 29.8 ms/token attention block + 10.8 preamble + 4.9 FFN |

Inside the loop only: **44.3 s fixed + 48.9 s scaling** — the loop is very nearly half fixed
at 512², and the fixed half grows with nothing.

---

## 5. The floor — and the 10 s verdict

**Smallest shape the product supports: 128×128, N=96** (`webui/shapes.py` BUILT, M=96,
`ready=True`; 64×64 is below the FFN row tile).

| | value |
|---|---|
| total, 8 steps, N=96 | **50.80 s** |
| of which startup | **27.11 s** (19.41 s of it the devkill `B repack`) |
| of which loop | **21.61 s** (2.29 s/step — essentially all marginal: 1 row-block) |
| loops needed to reach 10 s | the loop alone is 2.2× the target; startup alone is 2.7× |

**Fitted floor.** `total = 83.19 + 0.0443·N`; setting total = 10 s gives **N = −1653**.
There is no resolution, including N = 0, at which this pipeline reaches 10 s. The most
optimistic measured configuration — product-mode startup (11.2 s, no devkill prologue) with
the fast loop — still starts at ≈ 33 s before the first denoise step.

**Plainly: no. Shrinking the image cannot reach 10 s.** The image tokens are *not* what the
run is made of. At the smallest resolution the model supports, **5.3 of every 10 seconds are
startup** and the remaining 4.7 are 240 layer-steps that cost 2.29 s each *with only 96
tokens in them* — 24 device dispatches per layer whose latency does not shrink with the
picture. 10 s would need the fixed cost to fall ~5× *and* the per-token cost to go to zero;
that is a different execution model, not a smaller image.

---

## 6. Where it is not linear, and why (from the design)

### 6.1 A mode switch at N = 304, not a kink in a curve

`chunk_gate()` refuses the chunked path for `m ≤ 304` ("inside the delivered ip ceiling") and
`resolve()` then runs the **native delivered image** instead. The two executors have
opposite cost shapes:

* native (N=96, 288): **F ≈ 0, c ≈ 25.2 ms/token/step** — the cost is the 24 device
  dispatches per layer (3 qkv + 18 ip query-block + 3 o), and it is *all* marginal. At N=96
  that is 2.29 s/step; at N=288, 7.12 s/step (3 row-blocks → 3× the dispatches).
* chunked (N ≥ 480): **F = 5.50 s/step, c = 5.89 ms/token/step** — 5.5 s of the step is
  paid before a single token is looked at.

They cross at ≈ N 290, which is why 256² (native, 7.12 s/step) lands almost exactly on the
chunked line's prediction for 288 (7.20 s/step) while 128² (native, 2.29 s/step) is 2.6×
below it. **This is why pooling the native arms into one fit drops R² from 0.987 to 0.870** —
it is a discontinuity in the cost model, not noise.

### 6.2 The FFN is bandwidth-bound, so its 24 s is *fixed*

FFN per layer is 104–109 ms from N=480 to N=1056 (+5 %) while the row-block count goes 5 → 11
(+120 %). Each layer streams **BG 88.47 MB + B3 44.24 MB = 132.7 MB** of weights regardless of
M (×30 layers = 3.98 GB per step). 3.27 s/step measured ÷ 3.98 GB ⇒ **≈1.2 GB/s effective
weight rate**, i.e. the FFN is sitting on the weight stream, not on the tokens. The same
arithmetic holds one layer down for the projection: 3 dispatches × (16.59 MB W + 8.1 MB X)
= 74 MB in 73 ms ⇒ **≈1.0 GB/s**. Its fitted "slope" (4.9 ms/token) is the small per-row
pack/decode tail, not the GEMM.

### 6.3 The projection is dispatch/weight-bound too

`rope+build` = 68 → 73 ms (×1.07) for ×2.2 tokens: `_proj_post` writes the W object
(16.59 MB, `BIGM_SIZES`) and the X buffer before **every** launch, so the three q/k/v
dispatches move ~74 MB and pay three launches + waits regardless of M — which is why four
dispatches cost ~100 ms/layer at *any* M. This is also the documented reason the projection is the worst-rate dispatch in the
machine (1.56–1.89 TFLOP/s vs 3.56 for FFN gate/up — `FLOOR-VERDICT.md` §1.2): `plain15`
uses **15 of 32 cores** on the projection.

### 6.4 The sweep's upper half is *unreachable*, for two independent reasons

Both were produced by the driver, not inferred:

1. **`N % 96 ≠ 0` → `ValueError: widen_t: input 2580480 elements, expected 2334720`**
   (384×384, N=608, at the *first* attention block). The chunked attention hook gets
   `raw = dispatch_raw(..., nt)` = `ceil(N/96)·96` rows and hands it to `widen_t(raw, seq)`
   = `N` rows. At 512² the two coincide (11×96 = 1056 = N) — that is the *only* reason 512²
   works. **The chunked path is implemented for N ≡ 0 (mod 96) only.**
2. **`N > 1056` → `RuntimeError: attempting to write past buffer size`** (608×512 N=1248,
   640×640 N=1632). `_scratch/image/t18_run.py:377` sizes the devkill projection input BO at
   `8_110_080 B = 11 tiles × 96 rows × 3840 × 2 B = N ≤ 1056`. The product images are bigger
   (`proj`/`swiglu`/`down` = 12,533,760 B = **17 tiles = N ≤ 1632**) — which is exactly why
   640² *runs* in product mode (P3) and dies in fast mode.

| shape | N | 96-row blocks | 32-row tiles | 32-slot passes | product gate says | actually runs |
|-------|---|--------------:|-------------:|---------------:|-------------------|---------------|
| 128×128 | 96 | 1 | 3 | 1 | native | ✅ native |
| 256×256 | 288 | 3 | 9 | 1 | native | ✅ native |
| 256×448 | 480 | 5 | 15 | 1 | chunked | ✅ |
| 384×384 | **608** | 7 | 19 | 1 | chunked | ❌ **no** — N % 96 |
| 256×640 | 672 | 7 | 21 | 1 | chunked | ✅ |
| 256×832 | 864 | 9 | 27 | 1 | chunked | ✅ |
| 512×512 | 1056 | 11 | **33** | **2** | chunked | ✅ (fast path ceiling) |
| 640×640 | 1632 | 17 | 51 | 2 | chunked | ❌ fast / ✅ **product** |
| 768×768 | **2336** | 25 | 73 | 3 | chunked | ❌ **no** — N%96 *and* > BO |
| 1024×1024 | **4128** | 43 | 129 | 5 | chunked | ❌ **no** — > every BO |

`chunk_gate(1024,1024)` returns **runnable, with a 1.5 h estimate**. It is not runnable: the
first projection writes 4128 rows into a 1632-row BO. The gate models L1 and host memory and
never looks at `BIGM_SIZES`.

### 6.5 The 33-tiles-over-32-slots kink is *not* active here

N=1056 is the one point in the sweep that crosses it (33 row-tiles into a 32-slot image, the
`ffn_design_t33` family of `KK32-E2E-REPORT.md`). The FFN fit shows **no step at 1056**:
residual +0.08 s against a 29.2 s value. The reason is that the devkill FFN images
(`swiglu11`/`dn11`) are 11-tile images, not 32-slot ones — this harness never builds the
33-tile image, so the documented 33/32 tax cannot appear in these numbers. Anyone chasing it
should know it is invisible by construction here.

### 6.6 The one genuinely superlinear term

Attention-block increments per +192 tokens: **+4.90, +5.34, +7.42 s** (480→672→864→1056) —
the last is 51 % bigger than the first. Host attention is O(N²) on 30 heads in f32
(`4·N²·128·30` FLOPs); at N=1056 it is already 83 ms of the 299 ms layer, and it is the only
segment whose slope grows. Extrapolating the *scaling* part to 1024² (N=4128) would put the
attention block at ~14.8 + 0.0298·4128 ≈ 138 s per step, i.e. ~18 min/step — which is
consistent with the gate's 1.5 h estimate for 8 steps, and is the reason the product's
"runnable" verdict is wrong for a different reason than the BO.

---

## 7. What would have to change (measured, not proposed)

| lever | effect on the fixed 83 s | effect on the 44.3 ms/token | evidence |
|---|---|---|---|
| cache the B repack (repo knows how: `wpack` maps + verifies in 1.19 s) | **−14…−21 s** | 0 | `[devkill] B repack at startup: 14.5–21.7 s` vs prod-mode startup 11.2–12.7 s |
| run the product path instead of devkill (P1–P3) | −26 s startup | **+89 ms/token** (133 vs 46) | 3-point product fit, R² = 0.9965 |
| remove the preamble | −7.3 s | −10.8 ms | preamble fit |
| move host attention off the CPU | −14.8 s | **−29.8 ms** (192 → 117 ms/token left) | attention-block fit; `FLOOR-VERDICT` measured 462.8 ms/layer for device attention |
| shrink the image | 0 | −44.3 ms/token | this sweep: F is invariant, c is the only thing N touches |

Even zeroing **every** fixed term leaves `46.3 ms/token` — 49 s at 512² and 4.4 s at N=96 —
before startup. 10 s is not in this design's reach by image size.

---

## 8. Reproduce

```powershell
cd D:\Project\deepseek_harness\public_work\zimage-bfp16-npu
# one launcher at a time; check for ANY python.exe first
(Get-Process python -ErrorAction SilentlyContinue | Measure-Object).Count   # must be 0
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File _scratch\devkill\run_arm_size.ps1 `
    -Tag n672a -W 640 -H 256 -Steps 8 -GlueProf 1
#   product-mode arm (no devkill images, no fused glue -- reaches N=1632):
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File _scratch\devkill\run_arm_size.ps1 `
    -Tag prod640 -W 640 -H 640 -Steps 8 -GlueProf 1 -Hostkill 0 -Devkill 0
```

Artifacts: `_scratch/devkill/sweep_results.json` (per-run record), `sweep_extract.py`
(reads `webui/jobs/<id>/{job.json,job.log}`), `sweep_fit.py`, `sweep_report.py`,
`gate_table.py` (the §6.4 table, no device). No product file was changed.
