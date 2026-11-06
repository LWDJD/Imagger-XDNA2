# OVL-REPORT — design 1, implemented and gated: where the loop serialises, and what overlap is actually worth

**Date** 2026-11-02 · Labels: **[M]** measured, **[A]** arithmetic on measured values,
**[S]** structural (code path), **[X]** not measured.

Scope: the 512^2 / 8-step arm (`imaggers/devkill/run_arm.ps1`, `T22_FASTGLUE=1`
`T22_REF_SMNB=1 ZIMAGE_CHUNK_THREADS=16`). Two `imaggers` modules changed, each
behind its own flag, all defaulting to 0. No product file, no image, no arithmetic.

---

## 0. Verdict in six lines

1. **The in-loop host work and the device dispatches do not form two streams, they
   form a chain** [S]: every host segment's output is the next dispatch's input, in
   exactly one order, for all 30 layers. The sum of the section columns is the wall
   (291 ms/layer of sections against a 383 ms layer wall in the clean window, 317
   against 437 under contention; the rest is the per-step preamble). CONTENTION-REPORT 4.2 prices design 1 as `max(D(1+d), H(1+e))` = **61.17 s**;
   that is the wall of a graph this product does not have. On this graph the wall is
   `D + H`, and 61.17 s is the **floor of any schedule that removes the chain**, not
   the result of re-ordering it.
2. **Exactly two host segments are off the chain, and neither was overlapped before**
   [S]: `rope(q)` (the k projection does not read `qh`) and `rope(k)` (v does not read
   `kh`) — 4.5 ms/layer — and **layer i+1's 88.5 MB gu W payload**, which is a
   per-layer constant the current layer's `dn` dispatch does not read — 4.3 ms/layer.
   Everything else (attention 84.5, norm1/2/ffn_norm1/ffnpost ~6, A pack 6.4, drain
   reads 4.0, decode 1.2) feeds the very next dispatch and cannot move.
3. **Implemented both, and they land in every pair**: `rope+build` 73.5 -> 69.0 ms/layer
   and `[devkill] wgu` 4.3 -> 0.8 ms/layer in the clean window (84.0 -> 78.0 and
   5.2 -> 0.9 under contention), with the device columns untouched (`gu` 56.2 -> 56.2,
   `dn + wgn` 38.2 -> 38.2: the hoisted write is *perfectly* hidden) [M]. The
   wall-effective saving is **8.0 ms/layer = 1.9 s per 8-step image** [A].
4. **The loop wall resolves it in a clean window** [M]: with the box to itself, two
   interleaved pairs gave 91.89 -> 89.74 s and 91.63 -> 89.83 s (geomean **0.9785,
   -2.15 %**), and the two A arms agree to 0.26 s, so the effect is ~8x the band. In a
   **contended** window (another job at 5-18 cores of 20) the same three pairs read
   0.9720 / 1.0269 / 0.9671 (geomean 0.9883) against a 2.3 s band: there the wall
   resolves nothing and the section columns are the evidence. Both windows agree on
   the sections to 0.5 ms, which is the point.
5. **Latents are byte-identical in all ten fast-mode arms** (`2abf91331200`), and the
   host-only gate shows the OFF path, the ON path and the proj-only path produce the same
   bytes **and the same (BO, W-BO, W-BO-sha) dispatch order**, with a negative control
   that bites [M].
6. **Therefore: design 1 is real but it is 1.9-2.0 s, not 28.57 s.** The 26 s gap is not a
   modelling error in the coefficients (d = 0.153 / e = 0.165 are confirmed here: the
   overlapped host kernel costs the device **nothing**); it is the assumption that `D`
   and `H` are concurrent. See section 7 for what would have to change.

---

## 1. Where the loop serialises host work and device dispatches

### 1.1 The chain, per layer-step, with the code sites

`t18_run.block_step` [S: `imaggers/image/t18_run.py:1869`, and its generated copy in
`install.py`] calls, in this order, with **no pipelining anywhere**:

| # | segment | code | ms/layer [M] | consumes | next dispatch | needs it? |
|---|---|---|---:|---|---|---|
| 1 | `attention_norm1` + adaln | `t18_run.py:1879` (hunk h1) | 1.5 | `u` (prev layer) | q | **yes** |
| 2 | q projection dispatch | `install.py:385` `_proj_post` | 21.0 | x1, Wq | k | no (W BO only) |
| 3 | `rope_fused(q)` | `install.py:483` | 2.3 | q drain | **k** | **NO -> hidden here** |
| 4 | k projection dispatch | `install.py:480` | 21.0 | x1, Wk | v | no |
| 5 | `rope_fused(k)` | `install.py:492` | 2.3 | k drain | **v** | **NO -> hidden here** |
| 6 | v projection dispatch | `install.py:489` | 21.0 | x1, Wv | attention | no |
| 7 | `widen_t(v)` | `install.py:499` | 1.9 | v drain | attention | **yes** (vh) |
| 8 | **host attention** | `install.py:522` -> `webui/chunked.py:223` | **84.5** | qh, kh, vh | o | **yes** (o is its output) |
| 9 | `bf16_of(o)` | `install.py:525` | 0.7 | attention out | o | **yes** |
| 10 | o projection dispatch | `install.py:528` | 28.0 | o16, Wo | gu | **yes** (via 11-14) |
| 11 | `proj_read_cv(o drain)` + absmax | `install.py:555` | 3.0 | o drain | gu | **yes** |
| 12 | `attention_norm2` + residual | `t18_run.py:1897` (hunk h2) | 1.5 | a, u | gu | **yes** |
| 13 | `ffn_norm1` -> bf16 | `t18_run.py:1899` (hunk h3) | 1.5 | xm | gu | **yes** |
| 14 | A image pack (8.1 MB) | `devkill_ffn.py:311` `a_pack_fast` | 6.4 | xn | gu | **yes** |
| 15 | **gu W payload write** (88.5 MB) | `devkill_ffn.py:317` | 5.2 | nothing (constant) | gu | **it IS the weights -> i+1 is free** |
| 16 | gu dispatch (+ dn W write on a thread) | `devkill_ffn.py:358` | 57.5 | A, W_gu | dn | **yes** (y) |
| 17 | gu drain read + decode | `devkill_ffn.py:390` | 4.0 | y | dn | **yes** |
| 18 | dn dispatch | `devkill_ffn.py:378` | 38.6 | y, W_dn | ffnpost | **yes** |
| 19 | `ffn_norm2` + residual (hunk h4) | `t18_run.py:2032` | 1.5 | y, xm | next norm1 | **yes** |
| 20 | absmax x3, print, python body | `t18_run.py:2047`, `:2528` | ~10 | — | — | no, already memoised/small |

The ms above are the **contended-window** medians (ov_a2/a3/a4); the clean window (5.7)
is 4-6 % lower (attention 80.5, `gu` 56.2, `dn` 38.2, `rope+build` 73.5).

Device busy: 21+21+21+28+57.5+38.6 = **187 ms/layer**. Host in-loop: **~130 ms/layer**.
Wall: **437 ms/layer** contended, **383 ms** clean (11.5 s per 30-layer step). The device
is idle **57 % of the loop**, and every idle millisecond is a row of this table.

The two rows marked *hidden here* are the whole of design 1's in-loop opportunity.
Row 15 is the third: the write is 88.5 MB of **constant payload**, so it does not have
to happen in its own layer at all.

### 1.2 The proof that nothing else overlaps today **[M]**

Run `ov_a2` (delivered config, fast mode, 8 steps): the four `[fastglue]` sections plus
`[devkill] total` sum to **317.2 ms/layer**, and the step wall is 13.1-13.4 s per 30
layers = **437-447 ms/layer**. The 120 ms difference is the per-step preamble (refiner
+ t_emb + x_emb + final_layer, 66 ms/layer-equivalent) plus the four fused norms and the
python body.

If the device and the host overlapped at all, the sections would sum to MORE than the
wall. They sum to less, by exactly the preamble and the small unmatched glue. **The
overlap fraction is 0**, and it is 0 by construction, not by accident: rows 1-20 are a
single chain.

---

## 2. What was implemented (and what was deliberately not)

### 2.1 The mechanism: launch-ahead by one, no BO double buffering

`_dispatch_proj` was split into `_proj_post` (write X+W, submit) and `_proj_join` (wait,
copy the drain out) -- `imaggers/util/glue/install.py:385/402/421`. With
`_dispatch_proj = _proj_join(_proj_post(...))` the delivered path is unchanged call for
call. The overlap branch (`install.py:467`) then orders the q/k/v block as:

    post q ; join q ; post k ; rope(q) ; join k ; post v ; rope(k) ; join v ; widen(v)

so the rope of dispatch n runs while dispatch n+1 is on the device, and **the same
single X/W/drain BO pair carries every dispatch**: `_proj_join` copies the drain to host
memory before the next `_proj_post` rewrites X and W, which is the ordering the device
already guarantees (the wait returns when it has finished reading them).

### 2.2 The FFN weight hoist: layer i+1's gu payload under layer i's dn

`imaggers/devkill/devkill_ffn.py:206-266` adds `_ovl_wgu()` / `_wgu_pre_hit()` /
`_next_gu_payload()` / `_wgu_hoist()`, and `block()` (:297) uses them in two places:

* :320 -- **skip** the 88.5 MB write when the payload in hand is the one the previous
  layer pre-wrote (`_WGU['ptr']/['nb']`, i.e. the *address and size of the payload*, not
  a flag);
* :378 -- submit `dn` **first**, then write layer i+1's payload into the gu W BO, then
  wait. `dn` reads its own alt W BO (`_alt_bo`) and its own output BO, and this layer's
  `gu` has already been waited for, so the gu W BO is free. The next reader is layer
  i+1's `gu` dispatch, issued after this returns on the same thread.

The next layer's payload comes from `t18_run.CACHE_BG[i+1]` (a row of the resident
30-layer cache, already in image order -- `devkill_ffn._layer_of` already relies on that
identity). In **low memory mode** (`--no-cache-all`) there is no resident cache, so
`_next_gu_payload` returns None and the hoist **fails open** -- which is what happened in
the first pair I ran (section 5.5).

### 2.3 The host-pool cap

`T22_OVL_THREADS=N` (default 0 = keep `T22_FASTGLUE_THREADS`) sets the numba pool for the
**overlapped** kernels only (`th_o = _OVL_TH or th`, `install.py:476`). That is deliberate:
the overlapped kernel is the only host work that runs beside the device, so it is the
only place where e (CONTENTION-REPORT: +0.165 at 18 threads) can be paid.

### 2.4 What was NOT touched, on purpose

The attention, the o dispatch, `proj_read_cv`, the four norms, the A pack, the gu/dn
drain reads and the decode all stay exactly where they are, because each one is the input
of the dispatch that follows it. Reordering them would be a **data-flow change** (a new
image or a new buffer), not a scheduling change, and would have to be gated against
arithmetic that this report cannot certify byte-identical.

---

## 3. Files

| file | status | what |
|---|---|---|
| `imaggers/util/glue/install.py` | **changed** (+148/-11 vs HEAD, all of it this work) | new flags `_OVL/_OVL_WGU/_OVL_TH`; `_dispatch_proj` split into `_proj_post`/`_proj_join` (the delivered path composes them back); the `if _OVL:` branch in `bigm_attention`; one install-time line `[fastglue] T22_OVL=1: ...` |
| `imaggers/devkill/devkill_ffn.py` | **changed** (+168/-24 vs HEAD, which also carries the earlier HOSTWIN3 `T22_FFN_TP` edit; this work is ~62 of the insertions) | `_WGU` state + `_ovl_wgu`/`_wgu_pre_hit`/`_next_gu_payload`/`_wgu_hoist`/`_wgu_report`; the write is skipped on a proven hit; `dn` is submitted before the hoist and waited after; the debug line gains `wgn=%.1f` (the hoisted write's own time, charged separately so `pack+wgu+gu+wdn+dn+read+dec+wgn` still closes to `total`) |
| `imaggers/ovl/gate_ovl.py` | **new** | host-only byte/order gate, 4 arms, negative control |
| `imaggers/ovl/run_ovl.ps1` | **new** | one A/B arm of the delivered harness, with the device preflight and the memory-mode gate |
| `imaggers/ovl/analyze.py` | **new** | pair ratios, section medians, deltas, latents |
| `OVL-REPORT.md` | **new** | this file |

Run byproducts (safe to delete): `imaggers/ovl/arm_ov_*.txt`, `imaggers/ovl/ov_*.bridge.log`,
`imaggers/ovl/_gate_*.json`, `imaggers/bigpipe2/ov_*.run.log`, the `webui/jobs/20261102-19*`
job dirs and their `webui/outputs/zimage_19*_p0_latents.npy`.

No product file was edited. `webui/`, `t18_run.py`, `prod_loop_bigm.py`, `prod_devkill.py`
and every xclbin are untouched by this work.

---

## 4. The gate (host only, no XRT)

```
py310 imaggers\ovl\gate_ovl.py --all
```

Four arms run in four child processes (`install()` cannot run twice in one process: it
replaces `block_step` with a generated copy and `inspect.getsource` then fails, which is the
documented HOSTPROF 1B trap). Each arm is the **real product path** --
`t18_run.block_step`, `prod_devkill.bigm_attention`, `devkill_ffn.block`, the real fastglue
hooks -- on `selftest_path.py`'s fake device, whose C depends on BOTH X and W and whose
`bo.read()` returns a uint8 ndarray like the bridge does. 3 layers, 18 dispatches, and the
gu-W hoist engages for layers 0 and 1 and is refused for the last.

| arm | env | activation bytes | dispatch order | hoists |
|---|---|---|---|---:|
| `off` | — | `871dcf2f8b63e6e1` | 18 records | 0 |
| `on` | `T22_OVL=1` | `871dcf2f8b63e6e1` | **identical** | 2/2 |
| `proj` | `T22_OVL=1 T22_OVL_WGU=0` | `871dcf2f8b63e6e1` | **identical** | 0 |
| `negctl` | hoist broken (never pre-write, always skip) | `7bbfad828fe3fa21` | **DIFF** | 0 |

The **order record** is the interesting half: for every dispatch it stores the output BO,
the W BO and the sha of the W BO's contents *at the moment the kernel is told to run*.
That is the property that matters -- the gu W payload must be in its BO before gu's wait,
and `dn` must read its own W BO, not the gu one. The negative control (skip the write but
do not pre-write) changes both the bytes and the order, so the gate is not blind.

---

## 5. The device A/B

### 5.1 Protocol

Three interleaved 8-step pairs, A = delivered, B = `T22_OVL=1`, one launcher at a time,
the delivered harness (`imaggers/devkill/run_arm.ps1 -Steps 8 -GlueProf 0 -FastGlue 1
-Glue2 1 -Glue3 1`) with `T22_FASTGLUE=1 T22_REF_SMNB=1 ZIMAGE_CHUNK_THREADS=16`.

Two environmental controls were added **after the first pair** (see 5.5), both recorded
per arm:

* **memory mode**: `webui/config.py:433` picks fast(`--cache-all`) vs low(`--no-cache-all`)
  from the commit headroom at submit time, and the two are not the same experiment (low
  mode loads every layer's payload from disk, has no resident cache for the gu-W hoist,
  and produces a *different* latent baseline). `run_ovl.ps1` waits for >= 9.6 GB of commit
  headroom so auto picks fast, and records the mode it actually got.
* **co-tenant**: another agent's host-only job ran throughout at **4.9-18.0 cores of 20**
  and 7-8 GB. Every arm file records its core count. This is the same condition
  CONTENTION-REPORT 2.1 documents; the interleaving is the mitigation.

### 5.2 The six arms

| tag | arm | loop_s | total_s | co-tenant cores | mode | latents sha |
|---|---|---:|---:|---:|---|---|
| `ov_a2` | A | 107.23 | 156.6 | 10.27 | fast | `2abf91331200` |
| `ov_b2` | B | **104.23** | 152.3 | 6.15 | fast | `2abf91331200` |
| `ov_a3` | A | 106.85 | 155.0 | 5.78 | fast | `2abf91331200` |
| `ov_b3` | B | 109.72 | 161.4 | 7.43 | fast | `2abf91331200` |
| `ov_a4` | A | 104.92 | 152.4 | 6.95 | fast | `2abf91331200` |
| `ov_b4` | B | **101.47** | 147.1 | 4.90 | fast | `2abf91331200` |
| `ov_a5` | A | 91.89 | 128.8 | **none** | fast | `2abf91331200` |
| `ov_b5` | B | **89.74** | 124.7 | **none** | fast | `2abf91331200` |
| `ov_a6` | A | 91.63 | 125.9 | **none** | fast | `2abf91331200` |
| `ov_b6` | B | **89.83** | 124.9 | **none** | fast | `2abf91331200` |

Rows 5-10 are the **clean window** (20:05-20:16, §5.7); rows 1-6 are the contended window.
Two earlier arms (`ov_a1` 199.31, `ov_b1` 196.66) were taken in **low memory mode**
(commit headroom below 9 GB at submit time) and are reported only in 5.5.

### 5.3 Pair ratios of loop_s

```
[pair 1] ov_a2 107.23 -> ov_b2 104.23   ratio 0.9720  (-3.01 s)
[pair 2] ov_a3 106.85 -> ov_b3 109.72   ratio 1.0269  (+2.87 s)
[pair 3] ov_a4 104.92 -> ov_b4 101.47   ratio 0.9671  (-3.45 s)
geomean ratio B/A = 0.9883  (-1.17 %);  mean -1.19 s;  median -3.01 s
```

clean window (section 5.7):

```
[pair 4] ov_a5 91.89 -> ov_b5 89.74   ratio 0.9766  (-2.15 s)
[pair 5] ov_a6 91.63 -> ov_b6 89.83   ratio 0.9804  (-1.80 s)
geomean ratio B/A = 0.9785  (-2.15 %);  the two A arms differ by 0.26 s
```

In the **contended** window the A arms span 104.92-107.23 (**2.3 s**), so the wall cannot
resolve a 2.3 s effect there -- the same conclusion HOSTWIN3 3.1 reached for its 1.3 s.
In the clean window the A arms span 0.26 s and the effect is **8x the band**. Pair 2 is
the one that inverts,
and it is also the pair whose B arm carries the run's only outlier (its `attn` median is
93.0 ms against 84.0-85.0 in every other run, and its step 0 took 19.9 s against
15.5-17.5: a heavier co-tenant window landed inside it).

### 5.4 Section medians -- the evidence, contended window **[M]**

The **clean-window** version of this table is 5.7 and is the one to quote; this one is
kept because it shows the same two deltas surviving a 5-18 core co-tenant.

240 samples per column per run (one line per layer-step). `[fastglue]` prints seconds,
`[devkill]` prints ms; both are normalised to ms/layer here.

| column | a2 | b2 | a3 | b3 | a4 | b4 | **A med** | **B med** | **delta** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `proj_qkv` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| `rope+build` | 84.5 | 80.0 | 84.0 | 78.0 | 83.0 | 77.5 | **84.0** | **78.0** | **-6.0** |
| `attn` | 84.5 | 85.0 | 85.0 | 93.0 | 84.0 | 84.0 | **84.5** | **85.0** | **+0.5** |
| `proj_o` | 32.0 | 32.0 | 32.0 | 33.0 | 32.0 | 32.0 | 32.0 | 32.0 | 0 |
| `pack` | 6.5 | 6.4 | 6.4 | 6.8 | 6.3 | 6.5 | 6.4 | 6.5 | +0.1 |
| `wgu` | 5.2 | 0.9 | 5.2 | 0.9 | 5.1 | 0.9 | **5.2** | **0.9** | **-4.3** |
| `gu` | 58.0 | 58.1 | 57.5 | 57.3 | 57.2 | 57.3 | 57.5 | 57.3 | -0.2 |
| `dn` | 39.0 | 32.9 | 38.6 | 32.8 | 38.6 | 32.5 | 38.6 | 32.8 | -5.9 |
| `read` | 2.7 | 2.8 | 3.0 | 3.0 | 2.6 | 2.8 | 2.7 | 2.8 | +0.1 |
| `dec` | 1.2 | 1.2 | 1.2 | 1.3 | 1.1 | 1.2 | 1.2 | 1.2 | 0.0 |
| `wgn` (new) | 0.0 | 6.5 | 0.0 | 6.5 | 0.0 | 6.1 | **0.0** | **6.5** | +6.5 |
| `total` | 116.2 | 111.3 | 116.3 | 112.5 | 115.0 | 110.0 | **116.2** | **111.3** | **-4.9** |
| sections sum | 317.2 | 308.4 | 317.4 | 316.6 | 313.9 | 303.4 | 316.7 | 306.3 | **-10.4** |

Three things to read off it:

1. **`rope+build` -6.0 in every pair.** That column is 3 dispatches + `rope(q)` +
   `rope(k)` + `widen(v)`. The two ropes (2.27 ms each on a clean box) now run under the k
   and v dispatches. The measured -6.0 is a little larger than the 4.5 the overlap is
   designed for; `widen(v)` is **not** overlapped (attention needs it), and the column's
   1 ms quantisation accounts for the rest.
2. **`wgu` 5.2 -> 0.9 and `wgn` 0.0 -> 6.5.** The 88.5 MB write moved out of the critical
   path into the `dn` window, where it costs 6.5 ms instead of 5.2 (it now competes with
   the device's own DMA) but is hidden: `dn` falls to 32.8 and `dn + wgn` = 39.3 against
   38.6 before, i.e. **the window grew 0.7 ms**. So the wall saving of this half is
   5.2 - 0.9 - 0.7 = **3.6 ms/layer**, not 4.3.
3. **Nothing else moved.** `gu` (a device dispatch) -0.2, `proj_o` 0.0, `pack`, `read`,
   `dec` +0.1 or less, `attn` +0.5. In particular the two overlapped kernels -- 8 numba
   threads for 2.3-4 ms, and an 88.5 MB `bo.write` -- cost the device **nothing
   measurable**: d = 0.000 +- 0.004 over `gu`, against CONTENTION-REPORT 2.3's d = +0.153
   for an 18-thread 74-86 ms attention pass. That is the report's own mechanism confirmed
   from the other side: **it is the length and width of the parallel region, not the
   bytes, that costs the device**, and 4 ms at 8 threads is below its resolution.

Wall arithmetic: 6.0 (rope, 1:1) + 3.6 (W hoist, 1:1 after the window adjustment) =
**9.6 ms/layer = 2.30 s per 8-step image** in the *contended* window; in the clean window
(5.7) the same two halves read 4.5 + 3.5 = **8.0 ms/layer = 1.92 s**, and the wall moved
-1.98 s (median of the two clean pairs) against that 1.92 s prediction. The three
*contended* pair ratios average -1.19 s and resolve nothing on their own; the two clean
pairs read -2.15 s and -1.80 s (5.7), which is the measurement of record.

### 5.5 The low-memory pair -- and a trap worth writing down

`ov_a1` / `ov_b1` ran before the memory gate existed: **199.31 s and 196.66 s** of loop
for the same command, because `webui/config.py` had flipped the job to `--no-cache-all` (2x
slower loop, no resident payload cache). In that mode:

* the gu-W hoist **cannot engage** (`_next_gu_payload` -> None, `wgn` stays 0.0, the atexit
  line says `0 hoist(s), N not-hoistable`) -- it fails open, which is why the code is
  written to prove the payload is already there instead of trusting a flag;
* `rope+build` still moved (125.5 -> 103.0) and `attn` moved the other way (125.0 -> 158.0)
  because arm B ran under 18.0 cores of co-tenant against arm A's 12.75 -- **a single
  pair with a 40 % load mismatch attributes nothing**, which is why the numbers above
  come from the fast-mode pairs with matched load (three contended in 5.2-5.4, two
  clean in 5.7).

**The trap:** `webui/config.py:433` decides fast vs low from the *commit headroom at
submit time* and says nothing in the driver log. Two runs of the identical command can
differ by 2x in `loop_s` purely on how much memory another process happened to hold. Any
A/B on this box must record the mode; `run_ovl.ps1` now gates on it.

> **Correction 2026-11-03 -- "different latents" here was a hash-convention error.**
> This paragraph used to end "... and produce **different latents** (`2abf91331200` low
> vs `4cbf4f4a1b98` fast)". Both strings are the SAME 512x512/8-step latents:
> `2abf91331200` is `sha256` of the **.npy file bytes** (what `job.json:latents_sha256`
> records -- true for 304 of 304 surviving jobs) and `4cbf4f4a1b98` is `sha256` of the
> **array `.tobytes()`** (the "sha16" the reports quote). They differ only by the
> 128-byte `.npy` header, and both memory modes produce both forms. Section 6 of this
> report (line 420) already said so; the error was this parenthetical, and it propagated
> to `imaggers/util/place/PLAN-FFN-512-AB.md:24`, `imaggers/ovl/run_ovl.ps1:87` and
> `imaggers/util/place/compare_512_ab.py:35`. Proof:
> `imaggers/par/memmode/MEMORY-MODE-NEUTRALITY.md` §5; the audit and the helper that make
> the mistake impossible: `imaggers/par/hashconv/`.

### 5.6 The host-pool cap arm **[M, not resolved]**

`ov_c1` = the B configuration with `T22_OVL_THREADS=4` (the overlapped kernels run on 4
numba threads instead of 8). Against `ov_b2`, run 15 minutes earlier:

| column | b2 (8 thr) | c1 (4 thr) | delta |
|---|---:|---:|---:|
| `rope+build` | 80.0 | 77.0 | -3.0 |
| `attn` | 85.0 | 83.0 | -2.0 |
| `gu` | 58.1 | 56.9 | -1.2 |
| `dn` | 32.9 | 33.2 | +0.3 |
| `total` | 111.3 | 108.8 | -2.5 |
| `loop_s` | 104.23 | 101.20 | -3.03 |

**Every column improved, including two (`attn`, `gu`) the flag cannot touch**, so this is
run-wide drift, not an effect of the cap. The cap is **not resolvable** in this
environment. What it does say is negative and useful: at 4 threads the overlapped rope
still fits inside the 21 ms dispatch (`rope+build` did not grow), and at 8 threads it
already costs the device nothing (5.4) -- so on this product **e is not paid at all**, and
the cap has no epsilon to buy.


### 5.7 The clean window — the decisive run **[M]**

The co-tenant ended at 20:03. Four runs were taken back to back with **no other
`python.exe` alive** (`cotenant none` in every arm file), all in fast mode, and the two
A arms reproduce the recorded clean-box sections (rope+build 73.5 against HOSTWIN3's
73.0, `wgu` 4.3 against 4.7, `gu` 56.2 against 57.6, `dn` 38.2 against 38.5).

| tag | arm | loop_s | total_s | per-step s | latents sha |
|---|---|---:|---:|---|---|
| `ov_a5` | A | 91.89 | 128.8 | 13.97 11.71 11.60 11.61 11.50 11.73 11.44 11.43 | `2abf91331200` |
| `ov_b5` | B | **89.74** | 124.7 | 14.15 11.21 11.30 11.33 11.27 11.31 11.13 11.26 | `2abf91331200` |
| `ov_a6` | A | 91.63 | 125.9 | 14.25 11.55 11.78 11.58 11.49 11.39 11.39 11.38 | `2abf91331200` |
| `ov_b6` | B | **89.83** | 124.9 | 13.83 11.30 11.36 11.38 11.31 11.26 11.30 11.20 | `2abf91331200` |

**loop_s: -2.15 s and -1.80 s (geomean -2.15 %), against a 0.26 s A-arm spread.**
Every steady-state step of both B arms is below every steady-state step of both A arms
(11.13-11.36 against 11.38-11.78). This is the run the design should be judged on.

| column | a5 | b5 | a6 | b6 | **A med** | **B med** | **delta** |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rope+build` | 74.0 | 69.0 | 73.0 | 69.0 | **73.5** | **69.0** | **-4.5** |
| `attn` | 81.0 | 81.0 | 80.0 | 81.0 | 80.5 | 81.0 | +0.5 |
| `proj_o` | 29.0 | 29.0 | 29.0 | 29.0 | 29.0 | 29.0 | 0.0 |
| `pack` | 5.8 | 6.0 | 5.7 | 5.7 | 5.8 | 5.8 | +0.1 |
| `wgu` | 4.3 | 0.8 | 4.3 | 0.8 | **4.3** | **0.8** | **-3.5** |
| `gu` | 56.1 | 56.2 | 56.2 | 56.1 | 56.2 | 56.2 | **0.0** |
| `dn` | 38.2 | 33.8 | 38.1 | 33.7 | 38.2 | 33.8 | -4.4 |
| `wgn` | 0.0 | 4.4 | 0.0 | 4.4 | **0.0** | **4.4** | +4.4 |
| `total` | 108.4 | 105.0 | 107.9 | 104.7 | **108.2** | **104.8** | **-3.3** |
| sections sum | 292.4 | 284.0 | 289.9 | 283.7 | 291.2 | 283.9 | **-7.3** |

Four things this pins down:

1. **`rope+build` -4.5, exactly the design value** (rope(q) 2.27 + rope(k) 2.27;
   `widen(v)` is not overlapped). In the contended window the same column read -6.0,
   i.e. the contended number was 1.5 ms optimistic -- which is why the clean run is the
   one to quote.
2. **`dn + wgn` = 33.8 + 4.4 = 38.2 = the delivered `dn` exactly.** The 88.5 MB write is
   *perfectly* hidden inside the dn dispatch: the window did not grow by one sample.
   The FFN half is therefore worth the full 4.3 - 0.8 = 3.5 ms/layer.
3. **`gu` 56.2 -> 56.2 and `attn` +0.5.** The overlapped kernels cost the device and the
   attention nothing. d = 0.000 over 240 samples.
4. **The wall moves 1:1 with the sections**: -7.3 ms/layer x 240 = -1.75 s predicted,
   -1.98 s median measured (-1.80 to -2.15). That 1:1 is the same fact as section 1.2:
   nothing overlaps, so a millisecond removed from a section is a millisecond off the
   wall.

**Headline number: -1.9 s +- 0.2 s per 8-step image (-2.15 %), 8.0 ms/layer, with zero
cost to any other column and byte-identical latents.**

---

## 6. Quality

**All ten fast-mode arms (5.2, 5.7) and both low-mode arms (5.5) produce `2abf91331200`**
(`webui/outputs/zimage_19*_p0_latents.npy`, 262,144 B, sha256 prefix). That is the sha
HOSTPROF-REPORT records for this configuration (`hp_p1_latents.npy`, `2abf91331200d069`,
"the sha that ~35 other runs of this config produced today").

> This line is the correct statement; 5.5's parenthetical said the opposite and was the
> hash-convention error (see the correction note there).

Three independent reasons the arithmetic cannot move, in order of strength:

1. **Sampled identity**: the same X bytes, the same W objects, the same k-group order, the
   same kernels, the same destination buffers. Only the *instant* at which a host kernel
   or a BO write is issued changed.
2. **The gate** (section 4): the product path with the flag on is byte-equal to the flag
   off, and the dispatch-order record is identical.
3. **The device A/B** (section 5.2): 6/6 arms agree.

---

## 7. What this means for the roadmap

1. **Design 1 is worth 1.9-2.0 s, not 28.57 s, and the difference is the chain.** The
   report's coefficients are not the problem -- this A/B *confirms* them (the overlapped
   kernel costs the device 0.000 +- 0.004, well inside the d = +0.153 budget). The
   problem is `wall = max(D(1+d), H(1+e))`: it needs two independent streams, and
   `block_step` is one chain of 20 alternating segments. The same objection applies to
   design 2's *in-loop* half. The exception is the preamble: `cap_embedding` + the two
   `context_refiner` blocks + `t_emb` depend only on `step_t`, not on the latent, so
   step i+1's context half **is** genuinely off-chain and can hide under step i's layer
   loop. CONTENTION-REPORT 4.1 puts R at 15.78 s of the 89.74 s loop; how much of it is
   the context half is **[X]** here (not implemented, not measured) and is the next thing
   worth pricing.
2. **61.17 s is a floor, not a schedule.** It is what you get if the host segments stop
   existing. Getting there needs images that absorb them (BIGPIPE's P3-1: norm -> qkv ->
   attn -> o -> ffn on one image, intermediate state never leaving the tile), which is a
   data-flow change that also removes 1,650 dispatches/step. No re-ordering of the
   current dispatch sequence can do it.
3. **Every remaining host millisecond is 1:1 on the wall.** `attn` 84.5, the A pack 6.4,
   the drain reads 4.0, the four norms 6, the python body ~10: the HOSTPROF Pareto is
   still the right list, and its items are worth exactly what they say. CONTENTION-REPORT
   5's ordering ("overlap first, host wins second, kernel last") should therefore be
   **host wins first**: on this graph they are the only lever that pays 1:1 without a new
   image.
4. **The device is idle 57 % of the loop and will stay idle.** At 187 ms/layer of device
   time against a 437 ms layer, even a 2x faster kernel buys 0 s unless the host comes
   down first -- CONTENTION-REPORT 4.5's conclusion, reached here from the schedule rather
   than from the coefficients.

---

## 8. Reproduce

```powershell
cd <checkout>

# 1. the host-only gate (no device; ~1 min)
py310 imaggers\ovl\gate_ovl.py --all

# 2. three interleaved 8-step pairs (device; one launcher at a time; ~2.5 min each)
powershell -NoProfile -ExecutionPolicy Bypass -File imaggers\ovl\run_ovl.ps1 -Arm A -Tag ov_a2 -Steps 8 -AllowCotenant
powershell -NoProfile -ExecutionPolicy Bypass -File imaggers\ovl\run_ovl.ps1 -Arm B -Tag ov_b2 -Steps 8 -AllowCotenant
powershell -NoProfile -ExecutionPolicy Bypass -File imaggers\ovl\run_ovl.ps1 -Arm A -Tag ov_a3 -Steps 8 -AllowCotenant
powershell -NoProfile -ExecutionPolicy Bypass -File imaggers\ovl\run_ovl.ps1 -Arm B -Tag ov_b3 -Steps 8 -AllowCotenant
powershell -NoProfile -ExecutionPolicy Bypass -File imaggers\ovl\run_ovl.ps1 -Arm A -Tag ov_a4 -Steps 8 -AllowCotenant
powershell -NoProfile -ExecutionPolicy Bypass -File imaggers\ovl\run_ovl.ps1 -Arm B -Tag ov_b4 -Steps 8 -AllowCotenant

# 3. the numbers
py310 imaggers\ovl\analyze.py ov_a2 ov_b2 ov_a3 ov_b3 ov_a4 ov_b4

# 4. the clean-window pairs (section 5.7) -- run these with NO other python.exe alive,
#    they are the measurement of record: -2.15 s and -1.80 s, A-arm spread 0.26 s
powershell -NoProfile -ExecutionPolicy Bypass -File imaggers\ovl\run_ovl.ps1 -Arm A -Tag ov_a5 -Steps 8
powershell -NoProfile -ExecutionPolicy Bypass -File imaggers\ovl\run_ovl.ps1 -Arm B -Tag ov_b5 -Steps 8
powershell -NoProfile -ExecutionPolicy Bypass -File imaggers\ovl\run_ovl.ps1 -Arm A -Tag ov_a6 -Steps 8
powershell -NoProfile -ExecutionPolicy Bypass -File imaggers\ovl\run_ovl.ps1 -Arm B -Tag ov_b6 -Steps 8
py310 imaggers\ovl\analyze.py ov_a5 ov_b5 ov_a6 ov_b6
```

Arm B is arm A plus `T22_OVL=1` and nothing else. Drop `-AllowCotenant` to make the script
refuse while any `python.exe` is alive (it already refuses while a *device* job is: a TCP
connect on 127.0.0.1:7874, or a `imaggers/device.lock`).

### Flag summary

| flag | file | default | effect |
|---|---|---|---|
| `T22_OVL` | install.py, devkill_ffn.py | **0** | master switch: launch-ahead projections + gu-W hoist |
| `T22_OVL_WGU` | devkill_ffn.py | **1** (when OVL) | `0` disables only the gu-W hoist (bisect) |
| `T22_OVL_THREADS` | install.py | **0** (off) | numba pool for the overlapped kernels only |

With every flag at its default the only difference from the delivered code is that
`_dispatch_proj` calls `_proj_post` then `_proj_join` instead of doing both inline, and
that `block()` calls a predicate that returns False. The delivered arm of this A/B
(`ov_a2/a3/a4`) is that path, and it reproduces `2abf91331200`.
