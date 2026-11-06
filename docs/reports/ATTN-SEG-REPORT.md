# The attention segment: the four variants, the fixed cost, and one real 8-step A/B

**Date** 2026-11-02 · **device** taken in two clean windows, released at the end
**Deliverable** `webui/chunked.py` (five new `T22_ATTN_*` flags, all default 0) + the files in §9
**Nothing outside `webui/chunked.py` was edited.** Every path the product ships today is
byte-for-byte unchanged (proved in §2).

---

## 0. Verdict in one paragraph

Variant D is implemented exactly as A2/E were and it is by far the fastest attention this box
can run (35.7 vs 59.1 ms/layer isolated, and **-44% of the segment in-product**), **but it does
not survive the quality gate**: latents corr **0.9919453 < 0.9953**, so it is **REJECTED** and
must not be re-proposed as a default. The arm that *does* clear the bar is **variant A2 + the
buffer-reuse eliminations + numba threads 20 + qblock 1056**, which is exact to f32 rounding
(attention-output corr 1.0000000, rel_rms 1.1e-7) and measured **loop 82.896 -> 79.210 s,
pair ratio 0.9555, attn segment 66.0 -> 49.0 ms/layer, attn_sum 16.17 -> 11.88 s, latents
corr 0.9975444 = PASS**. Two things also came out of the measurement that change what this
project should do next: the fused softmax is *slower than torch* at the process thread count
the product sets (numba 8), and the 8-step chain amplifies a per-call attention error by about
four orders of magnitude, so per-call corr cannot be used to accept a change.

---

## 1. What was implemented (all inside `webui/chunked.py`)

| flag | default | what it does |
|---|---|---|
| `T22_ATTN_D=1` | 0 | **variant D.** bf16 qk matmul -> the score matrix is viewed as `uint16` (zero copy, it is contiguous) -> `bcore_tail.sm_bf16_bf16` writes bf16 bits into a **reused** uint16 buffer -> bf16 pv. The probabilities never exist as f32, so pv runs at the bf16 rate too. |
| `T22_ATTN_REUSE=1` | 0 | the fixed-cost eliminations of §3: both matmuls write into reused `out=` buffers, the `(heads,q,hd)->(q,heads,hd)` store goes through torch instead of a numpy strided assign, and the `(seq,heads,hd)` result buffer is reused. **Bit-exact.** |
| `T22_ATTN_NBTHREADS=N` | 0 | numba threads for the fused softmax only (0 = leave the process setting, which fastglue pins at 8). Restored before `_attention` returns. |
| `T22_ATTN_QB=N` | 0 | query-block override. **Changes no arithmetic** (the softmax is over the full key axis) — see §2. |
| `T22_ATTN_DUMP=prefix`, `T22_ATTN_DUMP_N=k` | off | saves the `qh/kh/vh` a product call actually saw plus its own output, so the isolated table can be replayed on real product tensors (§4). Reads only. |

Plus `_attn_prewarm()`, called from `install()`: it compiles the fused kernels **before the
loop**. Measured first-call cost in a fresh py310 process: `sm_bf16_bf16` **1.14-1.92 s**,
`sm_bf16_f32` 0.45 s, `sm_f32_inplace` 0.28 s, plus 0.27 s to import `bcore_tail`. numba
refuses to disk-cache the two bf16 kernels (they use `get_num_threads/get_thread_id`, which are
dynamic globals), so paid lazily that lands on step 0 and therefore on `loop_s`. The in-product
proof it works: the B arm's log line `[attnprew] sm_bf16_bf16 1.92s (… paid at install not in the
loop)` and `startup_s` 35.8 s (B) vs 35.9 s (A).

---

## 2. The bit-identity gate (`imaggers/attnseg/gate_flags.py`, exit 0)

One command, and it imports **both** implementations: the current `webui/chunked.py` and
`imaggers/attnseg/_frozen_head.py`, which is the `_attention` of **git HEAD**
(rev `d2ac9c9`, blob `1fadbef`), copied verbatim and checked verbatim by
`_check_frozen.py` (sha256 of the extracted text compared against `git show HEAD:…`).

```
== 1. DEFAULT PATH (all flags unset) vs frozen git-HEAD ==
  shape (30, 1056, 128)  sha c0c2a7245774  IDENTICAL
  shape (30, 1056, 128)  sha 545ac3fe851f  IDENTICAL
  shape (30, 1056, 128)  sha 19ba49c4c7a9  IDENTICAL
  shape (8, 320, 64)     sha da646285d33e  IDENTICAL
  qblock=256 / 352 / 1056                   IDENTICAL  (all sha 91cd19d1c201)

== 2. each flag: product code vs the attn_variants model ==
  T22_ATTN_REUSE=1           91cd19d1c201  corr 1.0000000  rel_rms 0.00e+00  yes (bytes)
  T22_ATTN_A2=1              1f54a6bca51c  corr 1.0000000  rel_rms 1.08e-07  yes (bytes)
  T22_ATTN_D=1               f9537c176918  corr 0.9997547  rel_rms 2.22e-02  yes (bytes)
  T22_ATTN_E=1               c2080e9c9d7b  corr 0.9997579  rel_rms 2.20e-02  yes (bytes)
  T22_ATTN_QB=1056           91cd19d1c201  corr 1.0000000  rel_rms 0.00e+00  yes (bytes)
  D  + REUSE + NBTHREADS=20  f9537c176918  (the B arm of pair 1)
  A2 + REUSE + NBTHREADS=20  1f54a6bca51c  (the B arm of pair 2)

== 3. T22_ATTN_DUMP and the install() prewarm ==
  dump does not change the result      : yes
  dump wrote qh/kh/vh/o                : [True, True, True, True]
  dumped o == the returned array       : yes
  D output == the D model (bytes)      : yes
  _attn_prewarm() with D on            : {'sm_bf16_bf16': 0.0}
  prewarm does not change the result   : yes
  shipping path again after all that   : yes
  _attn_prewarm() with every flag off  : {} (no-op)
```

So: the shipped path is **byte-identical**, `T22_ATTN_REUSE` alone is **byte-identical to
HEAD**, `T22_ATTN_QB` alone is **byte-identical to HEAD**, and every variant flag produces
exactly the bytes the measurement model produces.

### 2.1 The product really ran what §4 measured (`verify_product_replay.py`)

The dump from the real runs was replayed through the model:

```
arm                      product_o    model_o      shipped_o     corr
at1  shipped             777f00e0b840 777f00e0b840 777f00e0b840 1.0000000
at3  shipped             777f00e0b840 777f00e0b840 777f00e0b840 1.0000000
bt1  D+R+NB20+QB1056     7a4fd4daf58c 7a4fd4daf58c 777f00e0b840 0.9999758
bt3  A2+R+NB20+QB1056    1d4320c3c6f8 1d4320c3c6f8 777f00e0b840 1.0000000
[gate] PASS: the model reproduces every product attention output byte for byte
```

All four rows were measured. D: has only 2.2 GB free, so the **pair-1 dumps
(`real_at1_*`, `real_bt1_*`, 124 MB) were deleted after the run** and the script now skips a
missing prefix; the retained pair-2 dumps (`real_at3_*`, `real_bt3_*`, 124 MB) reproduce the
gate above on their own, and any dump can be regenerated by setting `T22_ATTN_DUMP=<prefix>`.

---

## 3. Where the fixed cost actually was (measured, not assumed)

All numbers per 512-row q-block at (30, 1056, 128), torch 16 threads, one process:

| item | measured | verdict |
|---|---|---|
| `.numpy()` / `torch.from_numpy` / `.view(uint16)` round trips | **2-4 us each**, 6 per layer | **negligible** — 0.02 ms/layer, 0.03 % of the segment. Nothing to eliminate. |
| `torch.empty` of a 65 MB score buffer | 0.003 ms (lazy) | the *allocation call* is free; the **first touch** is not: `matmul(new)+softmax(new)` 27.22 ms vs `matmul(out=)+softmax(new)` 26.69 ms -> **~0.5 ms/block** |
| the store `(30,512,128)->out[q0:q1]` | numpy strided assign **1.07 ms**, torch `copy_` **0.29 ms** | **the one real fixed cost**; torch copy into the same numpy buffer is bit-exact and 3.7x faster |
| the q slice `qt[:, q0:q1]` | f32 matmul 9.93 ms strided vs 8.84 ms contiguous | a strided batch; with `QB=1056` there is one block and the slice is the whole tensor |
| numpy->bf16 conversion of q/k/v (variant D/E only) | 6.58 ms/layer (3 x 16.2 MB) | eats most of bf16's qk+pv gain — this is why D is not 2x |

Net effect of the bit-exact part alone (`T22_ATTN_REUSE=1`): **59.10 -> 54.44 ms/layer
(0.927x)** with **IDENTICAL bytes**. That is the whole of the "fixed cost" this task asked for,
and it is worth having on its own.

---

## 4. The four-variant table

Product shape `HEADS=30, SEQ=1056, HD=128`, `qblock=512` exactly as `_attention` runs it,
**real product tensors** (`T22_ATTN_DUMP` from run `at3`), one process, 15 repetitions round
robin, `med ratio` = median over reps of (this / shipped in the same rep).
Source: `imaggers/attnseg/isolated_real.json` / `.csv`; synthetic-input version:
`isolated_synthetic.json`.

### 4.1 As the four variants would run today (numba at the process default, 8)

| config | med ms | min ms | med ratio | conv | qk | softmax | pv | store | bytes | attn corr | rel_rms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| today | 59.10 | 57.30 | **1.000x** | 0.12 | 18.27 | 20.68 | 15.29 | 3.67 | IDENTICAL | 1.0000000 | 0.00e+00 |
| A2 | 78.94 | 68.95 | **1.333x** | 0.09 | 17.61 | 40.79 | 14.94 | 3.62 | differs | 1.0000000 | 1.06e-07 |
| D | 67.36 | 58.52 | **1.118x** | 4.51 | 7.25 | 43.27 | 6.73 | 4.35 | differs | 0.9999758 | 6.96e-03 |
| E | 71.94 | 63.72 | **1.213x** | 2.98 | 7.20 | 42.74 | 14.78 | 3.68 | differs | 0.9999777 | 6.68e-03 |

**All three variants LOSE as-is**, and the reason is one column: the fused numba softmax costs
40-43 ms at 8 threads against torch's 20.7 ms. The campaign's earlier "A2 -14 %, E -21 %"
isolated numbers were not taken at the numba thread count the product actually runs
(`install.py:494`, `T22_FASTGLUE_THREADS=8` -> numba 8), so they do not describe the shipped
configuration. **This is the single most important number in this report**: a variant flag on
its own is a regression, and a reviewer who tests one flag at a time will conclude "attention
work does not pay".

### 4.2 With the two knobs that matter (numba threads 20, buffer reuse, qblock 1056)

| config | med ms | min ms | med ratio | conv | qk | softmax | pv | store | bytes | attn corr | rel_rms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| today | 59.10 | 57.30 | **1.000x** | 0.12 | 18.27 | 20.68 | 15.29 | 3.67 | IDENTICAL | 1.0000000 | 0.00e+00 |
| today+R | 54.44 | 52.73 | **0.927x** | 0.07 | 17.64 | 20.60 | 14.89 | 0.98 | **IDENTICAL** | 1.0000000 | 0.00e+00 |
| A2 nb20 | 57.36 | 55.06 | 0.965x | 0.09 | 17.49 | 19.63 | 15.07 | 3.62 | differs | 1.0000000 | 1.06e-07 |
| A2 nb20+R | 54.82 | 51.13 | 0.920x | 0.05 | 17.69 | 20.32 | 15.50 | 1.00 | differs | 1.0000000 | 1.06e-07 |
| **A2 nb20+R+QB1056** | **49.71** | **47.24** | **0.843x** | 0.04 | 16.50 | 18.11 | 14.00 | 0.84 | differs | **1.0000000** | **1.06e-07** |
| D nb20+R+QB1056 | 35.65 | 35.15 | **0.607x** | 4.52 | 5.59 | 18.70 | 5.59 | 0.98 | differs | 0.9999758 | 6.96e-03 |
| E nb20+R+QB1056 | 42.09 | 41.07 | 0.715x | 3.02 | 5.51 | 18.55 | 13.95 | 0.84 | differs | 0.9999777 | 6.68e-03 |

Corr here is the **attention output** against the shipped path on the same real tensors, one
call. §6 shows why it is a misleading statistic on its own.

### 4.3 The thread knob, measured (`imaggers/attnseg/nb_sweep.py`, real tensors, interleaved)

Torch pinned at 16 in every cell; the same-process control (the shipped path, numba 8) measured
**64.72 ms/layer** in this run, and it is measured in **every** repetition, so the ratios below
are drift-free. `nb` = `T22_ATTN_NBTHREADS`, i.e. numba threads for the fused softmax only.

| config | nb4 | nb8 | nb12 | nb16 | **nb20** |
|---|---:|---:|---:|---:|---:|
| shipped path (control), ms | 64.72 | 64.72 | 64.72 | 64.72 | 64.72 |
| A2+R+QB1056, ms | 91.14 | 67.87 | 56.27 | 49.45 | **46.65** |
| A2+R+QB1056, ratio | 1.408x | 1.049x | 0.869x | 0.764x | **0.721x** |
| D+R+QB1056, ms | 102.50 | 60.18 | 46.21 | 36.45 | **35.26** |
| D+R+QB1056, ratio | 1.584x | 0.930x | 0.714x | 0.563x | **0.545x** |
| E+R+QB1056, ms | 79.34 | 56.95 | 47.12 | 44.52 | **41.33** |
| E+R+QB1056, ratio | 1.226x | 0.880x | 0.728x | 0.688x | **0.639x** |

At the product's numba setting (**8**) the fused softmax is a *net loss* for A2 (1.049x) and only
a small win for D/E. The softmax alone goes 40.8 -> 18.1 ms/block between 8 and 20 threads
(**2.25x**), because it is bandwidth bound. `install.py:494` pins numba at
`T22_FASTGLUE_THREADS=8` for fastglue's kernels, so `T22_ATTN_NBTHREADS` is scoped to the
softmax call and restored before `_attention` returns.

---

## 5. The in-product A/B

Two pairs, 8 steps, `T22_FASTGLUE=1 T22_REF_SMNB=1 ZIMAGE_CHUNK_THREADS=16`, delivered harness
(`imaggers/devkill/run_arm.ps1 … -Steps 8 -GlueProf 0 -FastGlue 1 -Glue2 1 -Glue3 1`), one
launcher at a time, each launch preceded by a check that `imaggers/device.lock` was absent and
that **no** `python.exe` was running.

* `attn` = median over the 240 per-layer `[fastglue] layers.N … attn <s> …` lines the install.py
  hook prints (the product prints **seconds**; converted here), `attn_sum` = their sum.
* `rope` is the same code in both arms, so it is the **control** for machine drift.

### Pair 2 — the accepted arm (variant A2). Window 12:01:07 -> 12:05:17.

| arm | tag | loop_s | attn ms/layer | attn_sum s | rope ms | latents sha12 |
|---|---|---:|---:|---:|---:|---|
| A shipped | at3 | **82.896** | **66.0** | **16.17** | 71.0 | `4cbf4f4a1b98` |
| B A2+R+NB20+QB1056 | bt3 | **79.210** | **49.0** | **11.88** | 71.0 | `fdbdcd74dafd` |

**pair ratio 79.210 / 82.896 = 0.9555 (-4.45 %)**, attn segment **-25.8 %**, attn_sum
**-4.29 s**. The rope control is **identical to 0.1 ms** in both arms, so this pair is not
contaminated by drift. `total_s` 119.455 -> 116.425; `startup_s` 34.5 -> 35.2 (the +0.7 s is
the `[attnprew] sm_f32_inplace 0.70s` compile, deliberately moved out of the loop).

### Pair 1 — the D arm. REJECTED ON QUALITY. Window 11:49:26 -> 11:54:04.

| arm | tag | loop_s | attn ms/layer | attn_sum s | rope ms | latents sha12 |
|---|---|---:|---:|---:|---:|---|
| A shipped | at1 | **86.133** | **70.0** | **17.29** | 70.0 | `4cbf4f4a1b98` |
| B D+R+NB20+QB1056 | bt1 | **72.341** | **39.0** | **9.51** | 61.0 | `2fd2f3a773c2` |

pair ratio 72.341 / 86.133 = **0.840** (-13.79 s), attn segment **-44 %**, attn_sum -7.78 s.
**But the rope control moved 70.0 -> 61.0 (-13 %), i.e. the box was materially faster during the
B arm**, so 0.840 overstates the change and is only usable as an upper bound. The attention
segment comparison (attn/rope: 0.639 vs 1.000) is the defensible part: **about -36 %**.

### Drift between pairs, stated because it matters

The A arm of pair 1 (86.133 s) and the A arm of pair 2 (82.896 s) are the **same code** and
differ by **-3.24 s**. The box was measurably faster during pair 2. Both A arms produced
**byte-identical latents** (`4cbf4f4a1b98`, the recorded baseline), so the difference is speed,
not behaviour. **Use the pair ratios, not the absolute seconds.**

### One dead attempt, excluded

`at2` (11:55:24) was killed from outside at 11:55:56, before the loop (job log stops at
`[t13] cache-all … 8.21 GB free after`; no `job.json`, no FINAL line, no traceback). At
11:56:09 a foreign `hostprof` lock appeared and a GAIA python started. Excluded; pair 2 was
re-run as `at3`/`bt3` in the clean window. No number in this report comes from `at2`.

---

## 6. Quality

`imaggers/util/place/quality_gate.py` on the latents (the only quality statement that counts):

| pair | arm | sha256[:12] | NaN/inf | absmax | corr | maxabs | nrmse | verdict vs 0.9953 |
|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | A at1 | `4cbf4f4a1b98` | 0/0 | 3.97612 | 1 | 0 | 0 | — |
| 1 | **B bt1 (D)** | `2fd2f3a773c2` | 0/0 | 3.96242 | **0.9919453** | 2.368 | 0.1150 | **FAIL — REJECTED** |
| 2 | A at3 | `4cbf4f4a1b98` | 0/0 | 3.97612 | 1 | 0 | 0 | — |
| 2 | **B bt3 (A2)** | `fdbdcd74dafd` | 0/0 | 3.95549 | **0.9975444** | 1.759 | 0.0634 | **PASS** |

The shipped arm reproduced the recorded baseline **byte-identically in both pairs**
(`4cbf4f4a1b98`), so the A arm is the delivered path, not an approximation of it.

### 6.1 The finding that matters more than either variant: the chain amplifies ~1e4

| | per-call attention corr (real tensors) | 8-step latents corr | factor in rel_rms |
|---|---:|---:|---:|
| A2 | 1.0000000 (rel_rms 1.1e-7) | 0.9975444 | 1.1e-7 -> ~6e-3 |
| D | 0.9999758 (rel_rms 7.0e-3) | 0.9919453 | 7.0e-3 -> ~6e-2 |
| E | 0.9999777 | (not run in-product) | |

A per-call attention difference of 1e-7 relative — f32 rounding, nothing else — becomes a
**0.25 % latents difference**, and a 0.7 % per-call difference becomes **0.8 %** and falls below
the 0.9953 bar. **Do not accept or reject an attention change on its per-call corr.** The only
usable evidence is a full 8-step latents gate, which is why §5 exists.

---

## 7. Reproduce, one command each

```powershell
cd <checkout>

# (a) the gate: default path vs git HEAD, every flag vs the model, dump + prewarm
py310 imaggers\attnseg\gate_flags.py --threads 16

# (b) the four-variant table on REAL product tensors -> json + csv
py310 imaggers\attnseg\isolated_table.py --reps 15 --threads 16 ^
    --dump imaggers\attnseg\real_at3 --json imaggers\attnseg\isolated_real.json ^
    --csv imaggers\attnseg\isolated_real.csv

# (c) the model reproduces the product byte for byte on the product's own tensors
py310 imaggers\attnseg\verify_product_replay.py

# (d) THE WINNING ARM, in product, exactly as pair 2 ran it
$env:T22_FASTGLUE="1"; $env:T22_REF_SMNB="1"; $env:ZIMAGE_CHUNK_THREADS="16"
$env:T22_ATTN_A2="1"; $env:T22_ATTN_REUSE="1"
$env:T22_ATTN_NBTHREADS="20"; $env:T22_ATTN_QB="1056"
powershell -NoProfile -ExecutionPolicy Bypass -File imaggers\devkill\run_arm.ps1 ^
    -Tag mytag -Steps 8 -GlueProf 0 -FastGlue 1 -Glue2 1 -Glue3 1
# or, with nothing to copy-paste wrong:
powershell -NoProfile -ExecutionPolicy Bypass -File imaggers\attnseg\run_ab.ps1 ^
    -Arm B -Var A2 -Tag mytag

# (e) the A/B table + the quality gate from the artefacts
py310 imaggers\attnseg\ab_report.py at3 bt3
py310 imaggers\util\place\quality_gate.py ^
    webui\outputs\zimage_1201083215d9_p0_latents.npy ^
    webui\outputs\zimage_120320c03b53_p0_latents.npy --label "pair 2"
```

---

## 8. What not to re-propose

1. **Variant D as a default.** Fastest and rejected: latents corr 0.9919453 < 0.9953. Its
   per-call error (rel_rms 7e-3) is three times the shipped host-attention tier.
2. **The fused numba softmax at the product's numba thread count (8).** It is 2x slower than
   torch there (40.8 vs 20.7 ms/block). Every variant flag is a regression unless
   `T22_ATTN_NBTHREADS` is raised too — that is why the earlier isolated "-14 %/-21 %" numbers
   did not reproduce.
3. **`T22_ATTN_QB=1056` on a large shape.** Memory is `qblock*SEQ*HEADS*4 B` (133 MB at 512²),
   so it is only safe because SEQ=1056 here.

## 9. Files (all new; nothing tracked was edited except `webui/chunked.py`)

| file | role |
|---|---|
| `ATTN-SEG-REPORT.md` | this report |
| `imaggers/attnseg/_frozen_head.py` | `_attention` at git HEAD, verbatim (provenance in the header) |
| `imaggers/attnseg/_check_frozen.py` | proves the frozen copy is the HEAD text |
| `imaggers/attnseg/attn_variants.py` | the four variants + reuse/ktc/qblock/thread knobs, with per-segment timers |
| `imaggers/attnseg/gate_flags.py` | the bit-identity gate (§2) |
| `imaggers/attnseg/isolated_table.py` | the four-variant table, JSON + CSV (§4) |
| `imaggers/attnseg/verify_product_replay.py` | model == product on real tensors (§2.1) |
| `imaggers/attnseg/run_ab.ps1` | one A/B arm on the delivered harness (`-Arm A|B -Var D|A2|E|REUSE|D512`) |
| `imaggers/attnseg/ab_report.py` | pair ratios + in-product attn medians + quality, from the artefacts |
| `imaggers/attnseg/seg_stats.py` | per-layer segment table from a product job log |
| `imaggers/attnseg/micro_attn.py`, `bench_variants.py`, `bench2.py`, `bench3.py`, `prof_variants.py`, `jit_cost.py`, `probe_torch.py` | the primitive measurements of §3/§4.3 |
| `imaggers/attnseg/isolated_real.{json,csv}`, `isolated_synthetic.{json,csv}` | the tables as data |
| `imaggers/attnseg/arm_<tag>.txt` | the exact env each arm ran with |
| `imaggers/attnseg/real_at3_*`, `real_bt3_*` | the product tensors + product outputs of pair 2 (8 x 15.5 MB = 124 MB; the pair-1 dumps were deleted for disk, D: has 2.2 GB free) |

Run artefacts: `imaggers/bigpipe2/{at1,bt1,at3,bt3}.{run.log,job.json}`,
`webui/jobs/<id>/job.log` (240 per-layer lines each), `webui/outputs/zimage_<id>_p0_latents.npy`.

## 10. Device

Taken twice and released: 11:49:26-11:54:04 (pair 1) and 12:01:07-12:05:17 (pair 2). One run
(`at2`) was killed from outside by a foreign launcher before the loop and is excluded. No
`imaggers/device.lock` and no python process of this task is left running.
