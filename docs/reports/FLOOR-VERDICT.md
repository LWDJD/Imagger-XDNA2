# FLOOR-VERDICT — what the 512² 8-step pipeline can actually reach, and which one change moves it most

**Date** 2026-11-02 · **Scope** host-only analysis. No device opened, no xclbin loaded, no
`run_arm.ps1` invocation, no `device.lock`. Every number below is either a **measured**
value from this repo (tag **[M]**), an **arithmetic consequence** of measured values (**[A]**),
or an explicitly labelled **prediction** (**[P]**). Sources are given inline; nothing is
re-derived that the brief already fixed.

---

## 0. Verdict in five lines

1. **The floor table** (`loop_s / total_s`, today's code shape, 240 layer-steps, 32.3 s of
   start-up): **122.0 / 97.0 / 84.4 / 73.7 / 69.7** for {today, +all verified host wins,
   +kernel 1.5×, +kernel 2.6×, +kernel 3.6×}.
2. **A 10 s total needs 8.97 TFLOP/s of device GEMM** (9.99 if the refiner also moves to the
   device) — **4.04× today's sustained 2.22 TFLOP/s, 1.46× the best full GEMM ever measured
   on this box (6.14), 1.15× the best persistent-BO burst (7.83), and 30.7 % of the
   resident-kernel microbenchmark (29.2)**. It also needs everything that is *not* the device
   to take zero seconds, and the CPU-only floor is **24.1 s**.
3. **The single change that moves the total most is not in the loop**: the **18.6 s B repack**
   at start-up, worth **−16.2 s** for a cache that the repo already knows how to build
   (measured sibling: the 1.85 GB `wpack` cache maps + verifies in 1.19 s).
4. **The single change that moves the loop most is the kernel, and it is the one with the
   worst in-product record**: 1.5× on the device is **−12.6 s**, but the 1.5× number exists
   only at M=1024, and the same family at M=1056 measured **+42…+61 ms/layer-step** in product.
5. **<10 s is not reachable, and not by a small margin.** The loop's CPU work alone — host
   attention 49 ms + refiner 44.5 ms + 7 ms — is **24.1 s**, i.e. **2.4× the whole target**,
   before a single GEMM runs. The smallest number the measurements support is **≈ 70 s**
   (3.6× kernel + every verified host win + cached repack); **≈ 55 s** if the device *and* the
   glue stop being the wall; below that the refiner and the host attention must leave the CPU,
   and both have a measured negative on the device (attention 462.8 ms/layer).

---

## 1. Units — what "today" is, and where every millisecond comes from

### 1.1 The ledger that closes to the wall

| bucket | ms / layer-step | s / image | source |
|---|---:|---:|---|
| device (`pyxrt.run.wait` — the NPU computing) | **158.0** | 37.91 | HOSTPROF-REPORT §3 **[M]** |
| in-loop host | **143.2** | 34.37 | HOSTPROF-REPORT §3 **[M]** |
| per-step preamble (noise refiner ×2, t_emb, x_emb, modulation, final_layer) | **65.7** | 15.78 | HOSTPROF-REPORT §3 **[M]** |
| other / residual | **7.0** | 1.68 | **[A]** = wall − the three above (residual −0.7 %) |
| **sum** | **373.9** | **89.74** | **≈ the measured `[t3d] wall 89.80` [M]** |
| outside the loop | — | **32.30** | **[A]** = 122.10 (delivered total, n≥9) − 89.80 |
| **total** | — | **122.04** | HANDOVER-2026-11-01 §1: total 122.10 **[M]** |

**240 layer-steps** = 8 steps × 30 layers. The preamble is per *step* (1.972 s) and is
expressed per layer-step above (1.972 / 30 = 65.7 ms) so that everything is in one column.

Run-to-run band on this box: the *same* code measured `loop_s` 82.90 (job `at3`,
`20261102-120108-3215d9`) and 89.80 (`hp_h1`) on the same day, and the delivered n=10 median
is 84.8–84.9. So **every number in this document carries ±4 % of box state**, and §1.3 shows a
further ±19 % *inside* a single run.

### 1.2 Where the 158 ms of device time is (6 dispatches/layer-step)

| dispatch | shape | GFLOP | ms/layer-step | TFLOP/s | best measured on this shape |
|---|---|---:|---:|---:|---|
| proj q, k, v, o (4×) | M=1056 K=3840 N=3840 | 31.14 each | 16.5 – 20.0 each (total 66–80) | **1.56 – 1.89** | **9.88 ms** = 3.15 TFLOP/s (t33 kk32 n320, verified corr 0.99989, T33-REPORT §3) |
| FFN gate/up, silu-fused `swiglu11` | M=1056 K=3840 N=20480 | 166.09 | **46.7 – 55.9** | **2.97 – 3.56** | 2 × 22.06 = 44.1 ms = 3.77 (t33B, verified) — but that loses the fusion |
| FFN down `dn11` | M=1056 K=10240 N=3840 | 83.05 | **31.3 – 38.3** | **2.17 – 2.65** | not measured separately |
| **total** | | **373.7** | **158** | **2.37** | |

Sourcing: `[devkill] rows=1056 pack=… gu=46.7 … dn=31.3 …` lines of job `at3` **[M]**;
`[fastglue] layers.N proj_qkv 0.000 rope+build 0.067 attn 0.069 proj_o 0.030` **[M]** —
`rope+build` carries the three q/k/v dispatches, `proj_o` the fourth, so a dispatch is
16.5 ms in the fast state and 20 ms in the slow state; the 158 ms ledger sits between them.

**The projection is the worst-rate dispatch in the machine**: 1.56–1.89 TFLOP/s against
3.56 for the FFN gate/up, because `plain15` uses **15 of 32 cores** (HANDOVER-2026-11-01 §8:
"同形状 32 核 9.2 ms vs 产品 16.1 ms → 1.75×" **[M]**). That single fact is why the *average*
device rate (2.37 TFLOP/s) is 42 % below the best single-dispatch rate (4.09).

### 1.3 The device throttles inside a run — a measured, not modelled, fact

Per-step medians of the 180 `[devkill]` lines in job `at3` **[M]**:

| step | gu ms | dn ms | proj 3-dispatch segment | attn ms |
|---|---:|---:|---:|---:|
| 0 | 46.9 | 31.3 | 63 | 68 |
| 1 | 46.8 | 31.3 | 60 | 66 |
| 2 | 47.2 | 31.4 | 61 | 66 |
| **3** | **55.9** | **38.3** | **72** | 66 |
| 4 | 55.9 | 38.3 | 72 | 65.5 |
| 5 | 55.8 | 38.3 | 72 | 66 |

The identical binary, identical data, **+19 % device time from step 3 onward** while the host
attention stays flat (66.0 → 66.0). Any "kernel ×k" projection therefore has a ±19 %
denominator; this is also why "4.09 TFLOP/s shipped" and "158 ms/layer-step" do not
reconcile without the mix in §1.2.

### 1.4 The 32.3 s outside the loop, itemised (job `at3`, `startup_s = 34.54`) **[M]**

| item | s |
|---|---:|
| **B repack** ("mapped [n-group][k-tile][n-sub]; 30 layers repacked in place" -- the LOG's word: `T22_BREPACK_CACHE=0`, side-car `inplace`; the reports' "mapped B" is the opposite, `cache-map`) | **18.61** |
| t13 full-memory weight cache read (3.71 GB) | 4.1 |
| device open / image register (19 images) / driver warm | ≈ 5.3 |
| `devkill` prewarm of the 120 G=15 projection images | 3.17 |
| `t3d` warm of the 30 layers' projection images | 1.4 |
| `wpack` map + sha256 verify (1.85 GB) | 1.19 |
| prewarm wait (pack was a cache hit this run) | 0.57 |
| bridge import | 0.43 |

The 18.61 s repack is a pure function of the weights and is paid **every run**; a 23.5 s
first-build pack is already cached for the *projection* images (`wpack_l30_v2.bin`), which is
the exact mechanism a B-repack cache would reuse.

---

## 2. The floor table

### 2.1 Algebra

```
loop_s  = 240 × ( d + h + r + o ) / 1000        d = device wait, h = in-loop host,
total_s = loop_s + outside                     r = per-step preamble, o = everything else
```

Today `240 × 0.3739 = 89.74` (measured 89.80). Host wins act on `h` and `r`; a kernel factor
`k` divides only `d`. `outside` drops only by the repack cache.

### 2.2 The five rows

| row | d | h | r | o | loop_s | outside | **total_s** |
|---|---:|---:|---:|---:|---:|---:|---:|
| **today** | 158.0 | 143.2 | 65.7 | 7.0 | **89.7** | 32.3 | **122.0** |
| **+ all verified host wins** | 158.0 | 127.84 | 44.50 | 7.0 | **80.96** | 16.14 | **97.1** |
| **+ kernel 1.5×** | 105.33 | 127.84 | 44.50 | 7.0 | **68.32** | 16.14 | **84.4** |
| **+ kernel 2.6×** | 60.77 | 127.84 | 44.50 | 7.0 | **57.63** | 16.14 | **73.7** |
| **+ kernel 3.6×** | 43.89 | 127.84 | 44.50 | 7.0 | **53.58** | 16.14 | **69.7** |

Line-by-line:

* **Row 2 = row 1 plus three measured in-loop wins and one predicted start-up win.**
  `h`: −15.36 ms = attention arm A2+REUSE+NB20+QB1056, measured in product as
  `loop 82.896 → 79.210` (−3.686 s) with the rope control flat and latents corr **0.9975444
  (PASS)** (ATTN-SEG-REPORT §5/§6 **[M]**). `r`: −18.87 ms refiner bf16 (block-level 1.32×,
  2427 → 1861 ms/step, −4.53 s; REFINER-BF16-REPORT §2.3 **[M]**) and −2.33 ms bit-exact
  context-refiner cache (−0.56 s, gate 8/8 byte-equal; REFINER-AUDIT §5 **[M]**).
  `outside`: −16.2 s for a cached B repack **[P]** (§4, row 2). Without that one prediction
  the row is `loop 80.96 / total 113.3` — **that is the "measured-only" column**.
* **Rows 3–5** divide `d` by k and change nothing else. What each k means in the machine's own
  currency is in §2.3 — and it is the whole argument.

### 2.3 What "kernel ×k" is in TFLOP/s, and whether it has ever been measured

The device does **89.69 TFLOP** of GEMM per image (373.7 GFLOP × 240; attention's 4.11 TFLOP
are on the host and not in `d`).

| k | device rate required | what that number is, on this box |
|---|---:|---|
| 1.0 | **2.37 TFLOP/s** | today's mix — and **3.56 already achieved by the gu dispatch alone** |
| 1.5 | **3.55 TFLOP/s** | ≈ what gate/up already sustains. The 1.5× *image* (kk16 pure d2, 13.118 ms = 6.14) exists only at **M=1024** |
| 2.6 | **6.15 TFLOP/s** | **the best verified full-GEMM rate ever measured here** (kk32, M=1024, n=512, 13.187 ms, corr 0.999915) |
| 3.6 | **8.51 TFLOP/s** | **above every full GEMM ever measured.** Only the persistent-BO burst (10.277 ms = 7.83) comes close; nothing else does |

**This is the honest reading of the ladder.** The brief's 1.5×/2.6×/3.6× are *inner-loop
density* ratios (78.8 → 128 → 207 → 282 MAC/cyc). Density has never converted 1:1:
kk=32 density is 1.62× and the verified full GEMM is 1.49×; kk=64 density is 2.63× and the
only full-GEMM data point at that depth is **3.80 TFLOP/s = 0.93× the shipped rate**
(SWEEP-MN §2, §4 **[M]**). The physical reason is in INNER-LOOP-REPORT §7: the loop is at
**103 % of its own 26-bundle issue ceiling** (81 measured vs 78.8 MAC/cyc), but the dispatch is
bounded by the weight stream — 46 % duty cycle, 30.2 GB/s of the measured 39–42.5 GB/s roof —
so raising density moves the 16.2 ms of issue work to 4.5 ms and leaves a 35.1 ms dispatch
where it was.

### 2.4 Assumption register (each one can move a row)

| # | assumption | evidence for / against | direction |
|---|---|---|---|
| A1 | host wins are **additive** | The project's own rule: "不许把三笔收益相加" (fastglue's biggest block was the proj hook, overlapping the 32-core projection). Isolated 2.28× projection → 0.3 % in-product. | rows 2+ are **optimistic** |
| A2 | a kernel factor leaves the **host** cost unchanged | kk=32 in product **added** 12.7 ms of host silu + 3 extra dispatches at 16.4 ms of B re-read each | rows 3–5 **optimistic** |
| A3 | the kernel factor is **uniform** over the 6 dispatches | projection is 50 % of `d` at 1.56–1.89 TFLOP/s; the FFN is at 2.65–3.56. A 1.5× on the gate/up dispatch alone = **−3.7 s**, on both FFN dispatches = −6.2 s, on the projection alone = −6.4 s — none of them is −12.6 | rows 3–5 **optimistic** |
| A4 | refiner bf16 holds end-to-end | 1.32× measured on the real block with real weights; the 8-step latents gate has **not** been run | row 2 ± |
| A5 | the B-repack cache is 16.2 s | measured sibling: 1.85 GB `wpack` maps + sha-verifies in 1.19 s; 3.71 GB at 1.55 GB/s = 2.4 s | **[P]** |
| A6 | no new bottleneck at low device time | at k=3.6 the host is 78 % of the loop; the host is already the wall | rows 3–5 **optimistic** |
| A7 | box state | ±4 % run-to-run, ±19 % intra-run throttle (§1.3) | all rows ±5 % |

---

## 3. The throughput a 10 s total needs

### 3.1 The exact number

```
device GEMM per image          = 89.69 TFLOP   [A: 373.7 GFLOP/layer × 240]
+ refiner if it also moves     = 6.06 TFLOP    [A: 8 × 2 × 378.5 GFLOP]
+ host attention (stays CPU)   = 4.11 TFLOP    (not a device requirement)

10 s, device-only denominator  ->  89.69 / 10 = 8.97 TFLOP/s
10 s, everything on device     ->  99.86 / 10 = 9.99 TFLOP/s
```

**8.97 TFLOP/s sustained** is the answer to "what must the device sustain".

### 3.2 How far that is from every reference on this machine

| reference | TFLOP/s | 8.97 / it |
|---|---:|---:|
| **today's sustained rate** (89.69 TFLOP / 40.4 s busy) | **2.22** | **4.04×** |
| shipped single dispatch, kk=8, N=10240 (19.675 ms) | 4.09 | 2.19× |
| **best verified full GEMM ever** (kk32, 13.187 ms, corr 0.999915) | **6.14** | **1.46×** |
| best persistent-BO burst (kk16 pure d2, 10.277 ms, timing-only) | 7.83 | 1.15× |
| **resident-kernel microbenchmark** (913 GFLOP/s/core × 32) | **29.2** | **30.7 %** |
| architectural peak (51.3 TOPS measured by `xrt-smi`) | 51.3 | 17.5 % |

So the requirement is **not absurd against the silicon** (30.7 % of a measured marginal rate)
and **absurd against every structure that has ever produced a full GEMM here** (1.46× the
best one). The gap between those two columns *is* the project: the 29.2 TFLOP/s figure needs
B resident on-chip ("常驻内核"), and 512² cannot feed it — 512² gives AI ≈ 490 FLOP/B against
the ~1,260 FLOP/B the 39–42 GB/s inlet needs (UTIL-CAMPAIGN §9, INNER-LOOP §7 **[M]**).

### 3.3 The part that makes the throughput question moot

| CPU-only floor, measured inputs | s per image |
|---|---:|
| host attention at the best **accepted** variant (49 ms × 240) | **11.76** |
| refiner at bf16 + context cache (44.5 ms × 240) | **10.68** |
| everything else in the loop (7 ms × 240) | 1.68 |
| **CPU floor, device and glue both free** | **24.12** |
| + the host glue that actually exists (143.2 − 66.0 = **77.2** ms/layer today; 78.8 in row-2 terms, −17.8 after its own fixes ⇒ 61.0) | **+14.6** |
| **realistic CPU floor of the loop** | **≈ 38.4** |

The brief's "nothing below 26 s without a faster kernel" is **confirmed to within 8 %**: my
measured CPU-only floor is **24.1 s**. But the figure that matters is the **second** one — with
the glue the pipeline actually pays, the loop cannot go below **~38.4 s** even with an infinitely
fast NPU, because the host attention (11.8 s) and the refiner (10.7 s) are serial with each
other and with the device.

---

## 4. The ranked top 5 by (seconds saved) / (risk)

### 4.1 The record the risk score is read off (the nine that died)

| isolated win | what it claimed in isolation | what it did in product | which row it prices |
|---|---|---|---|
| **P33 33-tile projection hook** | 2.28× (90.5 → 39.7 ms for the four names) | **no-op** — the hook was never called (bit-identical latents); when reachable, 6.7 % of proj → **0.3 % of loop** (pair ratio 1.0027) | row 5 |
| **kk=32 FFN gate/up** | 1.49× (13.19 vs 19.68 ms) | **+42…+61 ms/layer-step** + 0.89 s/layer of B repack (33 tiles over 32 slots) | row 3 |
| **shared-B / single B stream** | 8× fewer B bytes, one shim channel | device **15.948 vs 13.443 ms = −18.6 %** (the brief's −28 %); the bytes were never the wall | row 3 |
| **attention D (bf16 qk/pv)** | −44 % of the segment, −13.8 s of loop (drift-inflated) | latents **0.9919453 < 0.9953 — rejected**, keep-out | bounds row 2 |
| **kk=16** | "one more k-block" | **no density change** — 52 slots for 8 vmac = the same 78.8 MAC/cyc | row 3 |
| **m/n tile plane** | a better tile | **flat**: shape B within 3.6 % over n = 256…640; m > 32 loses | row 3 |
| **CPU/device overlap** | −60 ms/layer | **177.0 vs 169.6 ms = +4 %**, dispatch 59.2 → 93.4 | kills a whole class |
| **numba rms (refiner)** | 29.3× | **+5…+6 s**: 6 calls/layer × (15 MB alloc + parallel-region start) | row 4 |
| **old 32-core projection** | 5.68× | **119 vs 106 ms/layer** (device −37, host +83) | row 5 |

The two that survived — fastglue (−27.5 %, byte-deterministic) and the refiner's fused softmax
(−1.9 %, corr 0.9953) — share one property: **in-place, zero new allocation, few calls**. That
is what the risk score rewards, and it is why a *cache* outranks a *kernel* here.

### 4.2 The risk scale

| score | meaning |
|---:|---|
| 1 | measured **in product**, 8 steps, quality-gated — no new engineering |
| 2 | measured at block level on real weights/inputs, **or** a mechanism with a working sibling in this repo; needs one 8-step gate |
| 3 | measured in isolation only; no in-product precedent either way |
| 4 | measured in isolation **and** this family already failed in product once |
| 5 | measured in isolation **and** the as-is route already failed in product in this exact shape |

| # | change | s saved | risk | s/risk | what the record says |
|---|---|---:|---:|---:|---|
| **1** | **B-repack cache** (write the repacked weight order once, mmap it) | **−16.2** **[P]** | 2 | **8.1** | Caches are the only category that has *never* died here: `wpack` (1.85 GB, 1.19 s), t13 full-memory cache (3.71 GB, 4.1 s), refiner context cache (8/8 byte-equal). The repack is a pure permutation of unchanged blocks, so the gate is the existing latents sha256 |
| **2** | **Attention A2 stack** (`T22_ATTN_A2` + `REUSE` + `NBTHREADS=20` + `QB=1056`) | **−3.69** **[M]** | 1 | **3.7** | The only *in-product, quality-gated, 8-step* win on the shelf: 82.896 → 79.210, rope control flat, latents 0.9975444 PASS. Zero new code — four env vars |
| **3** | **Kernel 1.5× (kk family)** | **−12.6** **[A]** | 5 | **2.5** | The fastest thing ever measured (6.14 TFLOP/s) lives at **M=1024**; the product is M=1056 = 33 tiles over 32 slots, and there the same family measured **+42…+61 ms/layer-step + 0.89 s/layer of B repack** (KK32-E2E §1/§2). The 33-tile fix exists and was measured on the product shape: **22.06 ms per half = 3.77 TFLOP/s**, i.e. 1.06× the fused image once the freed host silu is paid back (T33-REPORT §3) |
| **4** | **Refiner bf16** (`T22_REF_BF16=1`) | **−4.53** **[M]** | 3 | **1.5** | 1.32× median on 6 alternating block runs with real weights; per-call GEMM corr ≥ 0.999997. Open risk is the chain: a 1.1e-3 per-call difference amplifies ~1e4 into latents (§6.1 of ATTN-SEG), and the gate has not been run |
| **5** | **Projection placement** (33-tile kk=32, one dispatch) | **−6.4** **[M]** | 5 | **1.3** | Isolated **verified 2.28×** (90.5 → 39.7 ms for the four names; t33A 9.88 vs 11.35 ms). In product: **a no-op** (dead hook, bit-identical latents) and, when reachable, **0.3 % loop** (pair ratio 1.0027) — the host bill ate it |

*Next two, if the first five are taken:* `proj_read_cv` prange (−4.5 ms/layer = **−1.08 s**,
predicted, risk 2 → 0.54) and `bo.write` BO-residency (−2.9…−8.3 ms/layer, F-08 silent-wrong
hazard, risk 3 → 0.4).

**Which single change moves it most.** By absolute seconds: the **B repack** (16.2 s, 58 % of
the entire outside-loop budget, 13 % of the total). By *measured-in-product* seconds: the
**attention A2 stack** (3.7 s). By loop seconds: the **kernel** — 12.6 s at a 1.5× that has been
measured somewhere, 27.4 s at a 3.6× that has never been measured anywhere — and it is the one
line in this table whose in-product result is already known and known-negative.

---

## 5. Verdict

**<10 s is not reachable.** Not "hard" and not "another 20 %": the arithmetic misses by **12×**
on the total (122 → 10), by **2.4×** on the CPU-only floor (24.1 → 10), and by **1.46×** on the
best kernel ever measured on this box.

* **Blocker 1 — the CPU floor is 24.1 s** (attention 11.76 + refiner 10.68 + other 1.68) with
  the device and the glue both free. That is already 2.4× the target. The refiner's 6.06 TFLOP
  of host GEMM at the measured 330–450 GFLOPS (bf16 ≈ 600) is 10–14 s *by itself*.
* **Blocker 2 — the glue is real**: 77.2 ms/layer of in-loop host outside attention
  (18.5 s). The identified fixes remove ~17.8 ms/layer of it, leaving ~14.3 s.
* **Blocker 3 — the device would need 8.97 TFLOP/s**, 4.04× today's sustained rate and 1.46×
  the best full GEMM ever measured here, *and* would still be only 30 % of the resident-kernel
  microbenchmark — which no full GEMM on this box has ever approached because 512²'s arithmetic
  intensity (~490 FLOP/B) is below the ~1,260 FLOP/B the inlet needs.

**The smallest number the measurements support:**

| stack | total_s |
|---|---:|
| 3.6× kernel + every verified host win + cached repack (**row 5**) | **69.7** |
| … and the host glue squeezed by its own identified fixes (−18 ms/layer) | **65.4** |
| … and the device made free (the floor of the code as it exists) | **54.9** |
| … glue squeezed, device at 3.6×, **and** the refiner + attention moved off the CPU (both have a measured negative) | **28.3** |
| … the same, in a resident worker that does not re-pay start-up | **17.2** |

**The chain that supports 69.7 s**: 89.7 s loop today (HOSTPROF closure, residual −0.7 %) →
−3.69 s attention (measured pair, PASS) → −4.53 s refiner bf16 (block-level 1.32×) → −0.56 s
context cache (bit-exact) → **−27.4 s device** (158 → 43.9 ms/layer-step, i.e. 8.51 TFLOP/s)
→ **+16.1 s outside** (32.3 measured − 16.2 cached repack). Each link is either measured or
labelled; the soft links are exactly A1–A4 in §2.4.

**Where the 10 s idea could survive**: only if the device absorbed *everything* — the 89.69 TFLOP
of GEMM, the host attention (462.8 ms/layer when it was put on the device, i.e. 111 s), and the
refiner (441 ms/block, an **estimate**, not a measurement) — at ≥ 9.99 TFLOP/s. That is one
number against one number: 9.99 required vs 6.14 best-ever-verified, and the 462.8 ms/layer
attention result is the counter-example that says the device will not take the CPU's work at a
profit.

---

## 6. What I did not do, and what would settle the open questions

* **No device.** Nothing in this document was produced by a dispatch, an xclbin load, or a
  `run_arm.ps1` invocation. The two A/Bs I lean on hardest (attention A2, refiner bf16) are
  cited from artefacts, not re-run.
* **The four numbers that would tighten the table**, in value order:
  1. **The B-repack cache** — build it, run the 8-step A/B, expect `startup_s` 34.54 → ≈ 18
     and `loop_s` unchanged, latents sha256 unchanged (`4cbf4f4a1b98`). This is the only
     ≥15 s item on the board and it is a *cache*, not a kernel.
  2. **Refiner bf16, 8 steps, 3 runs, median** — the gate the block measurement lacks.
  3. **The attention A2 arm on the delivered harness** — 4 env vars, already measured once.
  4. **A 33-tile kk=32 FFN gate/up at N=10240 with the *verified* packer** — t33B measured
     22.06 ms for one half; whether 2 × 22.06 minus the fusion beats 46.7 has not been measured
     end-to-end, and it is the only route that keeps 33 tiles in one dispatch.
* **The one arithmetic that would flip the verdict**: a verified full GEMM at **≥ 6.2 TFLOP/s
  on the product's own M=1056 shape** (not M=1024). Nothing in the repo has that.
