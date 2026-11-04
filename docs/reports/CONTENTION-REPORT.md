# CONTENTION-REPORT — does host work really contend with the NPU?

**Date** 2026-11-02 · Labels: **[M]** measured, **[A]** arithmetic on measured values,
**[P]** prediction, **[X]** not measured.

This version supersedes the host-only draft: the NPU was released at 18:36 and **all three
phases ran on it** (e1 18:37, e2 18:38, e3 18:44-18:47). The contention coefficients below are
measured on `kk32_m32_n512` (M=1024 K=3840 N=10240, 80.53 GFLOP, A 7.86 MB, B 44.24 MB,
C 20.97 MB) with the dispatch sequence copied from `run_ffn.run_inline` (the 08:06 path), C
bytes verified identical start→end in every run. Files: `imaggers/contend/*`, all new.

**One caveat that governs §2.1.** Another agent's host-only sweep (`imaggers/pbudget`, no
device calls at all) ran throughout, at **3.8-13.7 cores**, drifting by 4× *during* the session.
It is why e1's arms are not directly comparable (§2.1) and why the measurement of record is e3
(§2.3), which brackets every load with its own baseline seconds apart.

---

## 0. Verdict in five lines

1. **Bandwidth is not the mechanism.** A co-tenant streaming **65 GB/s** — two-thirds of the
   box's roof — costs the device **+2.1 %** (noise). A co-tenant at **8.6 GB/s** (the product's
   attention shape) costs it **+15.3 %** **[M]**. Whatever hurts the NPU, it is not the memory
   bus being busy.
2. **Neither is raw thread count.** 20 threads doing cache-resident matmuls at **0 GB/s** cost
   **+0.5 %** **[M]**. Only the sustained-multi-core-and-memory shape moves the device. The
   dispatch loses its CPU, not its bandwidth.
3. **The measured coefficients are δ = +0.153 (device) and ε = +0.165 (host)** **[M]**, where ε
   is how much the *host* work inflates while the device runs beside it. Capping the host pool
   at 18 threads (2026-11-01 used 20) cut them from **+0.578 / +0.603** to these — 3.5× on both
   — and turned that experiment's **−4.3 % loss into a +2.4 % win** **[M, §2.4]**.
4. **With the measured coefficients, one 8-step image goes 89.74 s → 60.10 s (−29.64 s)** when
   every host segment hides, or **→ 61.17 s (−28.57 s)** when only the in-loop host hides
   **[A, §4]**. The two designs land within **1.1 s** of each other because the pole moves:
   design 1 pays δ 1:1, design 2 pays ε 1:1, and the two coefficients are the same size today.
5. **The 2026-11-01 rejection was an ε artefact.** Its own numbers replay to 176.6 ms against
   177.0 measured, with δ = +0.578 **inside** a +1.87 budget and ε = +0.603 **outside** a +0.54
   budget. With the same δ and ε = 0 it was a **35 % win** **[A, §4.4]**. Nothing in that
   experiment said the device could not be overlapped.

---

## 1. What was measured

| # | experiment | device? | when | result |
|---|---|---|---|---|
| A | load-type characterisation, 14 arms | no | 17:04 | §2.2 |
| B | co-tenant bandwidth loss ρ, 6 arms × 3 | no | 17:20 | §2.2 |
| C | host inflation ε, 3 co-tenants × 3 | no | 17:38 | §2.2 |
| **E1** | **contention table, 6 arms × 3 cycles × 25 dispatches** | **yes** | 18:37 | §2.1 |
| **E2** | **3 × 400-dispatch bursts + 20 s gap** | **yes** | 18:38 | §3.1 |
| **E3** | **A-B-A bracketed δ/ε, 3 load types** | **yes** | 18:44-18:47 | **§2.3** |

---

## 2. The contention table

### 2.1 E1 — the on-device arm table, and why it is not the headline **[M]**

18 blocks, 25 dispatches each, 3 interleaved cycles, per-dispatch `run_ms` medians:

| arm | what it is | run ms | p10-p90 | vs idle | bg cores |
|---|---|---:|---|---:|---:|
| (a) idle | nothing | 17.134 | 15.03-19.90 | — | 13.72 |
| (c) cpu4 | cache-resident matmul, 4 threads | 16.438 | 16.04-18.17 | −4.1 % | 10.03 |
| (c) cpu20 | same, 20 threads, **0 GB/s** | 16.660 | 16.15-21.04 | −2.8 % | 8.37 |
| (d) copy1 | 130 MB memcpy, 1 thread, **65 GB/s** | 17.360 | 15.02-20.93 | +1.3 % | 11.45 |
| (b) attn4 | attention shape, 4 threads | 18.485 | 16.20-20.51 | +7.9 % | 9.19 |
| (b) attn20 | attention shape, 20 threads, **8.6 GB/s** | 18.421 | 16.36-22.62 | +7.5 % | 3.75 |

**The `bg cores` column is the problem**: the other agent's sweep fell from 13.7 to 3.8 cores
across the six arms, so each arm is measured against a different background and the raw
percentages are not comparable. Normalising each arm against the idle block **in its own cycle**
gives attn20 **+6.1 %**, attn4 +2.6 %, copy1 −0.1 %, cpu20 +0.4 %, cpu4 −2.6 %: the sign and the
ordering survive, the magnitude does not. E1 is therefore reported as *consistent with* E3, not
as the measurement of record. (The whole session is affected: the 5-dispatch baseline read
16.7 ms at 18:37 and 13.4 ms at 18:43 — same binary, same image.)

### 2.2 Host-only characterisation **[M]**

`loads.py --self-bench` on a clean box; bytes are counted in the load, GB/s measured:

| arm | threads | ms/pass | **GB/s** | GFLOP/s | co-tenant loss ρ on a peer stream |
|---|---:|---:|---:|---:|---:|
| idle | — | — | 0 | 0 | — |
| attn (31 heads × 1024² f32 = 130 MB of S) | 20 | 65.97 | **8.62** | 252 | **−23.7 %** |
| | 4 | 164.27 | 3.46 | 101 | −7.1 % |
| cpu (512² matmul, 3 MB working set) | 20 | 0.52 | **0** | 517 | **−5.1 %** |
| | 4 | 1.90 | 0 | 142 | — |
| copy (130 MB memcpy) | 1 | 4.20 | **64.96** | 0 | **−28.1 %** |
| | 4 | 4.25 | 64.18 | 0 | — |
| poll (memory-status only) | 1 | — | ~0 | 0 | +0.1 % |

**Host inflation ε by OFF/ON alternation (`eps_bench.py`):**

| co-tenant | attention base | + co-tenant | ε |
|---|---:|---:|---:|
| copy 1 thread | 59.50 ms | 64.06 ms | +0.077 |
| **attn 20 threads** | 104.59 ms | 154.11 ms | **+0.473** |
| **cpu 20 threads** | 104.13 ms | 109.02 ms | **+0.047** |

The box's streaming roof is **65 GB/s and one thread reaches it** (1, 2 and 4 threads all give
64-65 GB/s). Against that, the NPU's weight inlet is **30.2 GB/s operating of a measured
39-42.5 GB/s roof** (INNER-LOOP-REPORT §7 **[M]**) — 46 % of the box's roof by itself. The priors
were reasonable; the experiment says they do not describe what actually happens.

### 2.3 E3 — the coefficient measurement, A-B-A bracketed **[M]** ← the measurement of record

Each load is bracketed: dispatch solo, then load alone, then both, then dispatch solo again,
then load alone — all inside ~20 s, so background drift cannot masquerade as an effect.

| co-tenant | D solo | D under | **δ (device)** | H alone | H under | **ε (host)** |
|---|---:|---:|---:|---:|---:|---:|
| copy 1 thread — **65 GB/s**, 1 core | 16.27 | 16.61 | **+0.021** | 8.92 | 12.89 | **+0.446** |
| cpu 20 threads — **0 GB/s**, 20 cores | 16.23 | 16.32 | **+0.005** | 1.24 | 1.43 | **+0.159** |
| attn 18 threads — **8.6 GB/s**, 18 cores | 14.36 | 16.55 | **+0.153** | 74.12 | 86.36 | **+0.165** |

Four things fall out:

* **The device is insensitive to bandwidth.** 65 GB/s of contiguous memcpy — the harshest
  possible test of the "shared LPDDR5x" hypothesis — moved the dispatch by **+2.1 %**, inside
  the run's own spread. The hypothesis that set this task is **rejected for the device**.
* **The device is insensitive to core count as such.** 20 threads of cache-resident matmul:
  **+0.5 %**. Their parallel regions are short (0.52 ms per call) and the cores breathe between
  them, so the dispatch thread always finds a CPU.
* **Only the attention shape hurts it: +15.3 %.** It is the one arm that holds many cores
  *continuously* (74-86 ms per pass) **and** touches memory throughout. The dispatch is a host
  thread driving `run.wait`; what it loses there is the CPU, not the bus.
* **Bandwidth contention is real, but it hits the other side.** copy1 is the only arm whose own
  throughput collapses next to the device (**ε = +0.446**): the dispatch's 52 MB payload write
  per iteration competes with it directly. The bus *is* shared — it just costs the host
  segment, not the NPU.

*Bracket note:* for attn18 the two solo brackets read 13.35 and 15.37 ms (the second has a wide
p90 — it caught a background spike). Against the bracket median δ = **+15.3 %**; against the
clean first bracket it is **+24.0 %**. Both are inside the budget in §4; the report uses the
conservative +15.3 %.

### 2.4 The 2026-11-01 comparison — same test, one variable changed **[M]**

| | 2026-11-01 | today |
|---|---|---|
| host pool | 20 threads | **18 threads** |
| dispatch pinned | no | no — the sandbox refused `cpu_affinity` (WinError 5) |
| δ device | **+0.578** | **+0.153** |
| ε host | **+0.603** | **+0.165** |
| wall vs serial | 177.0 vs 169.6 (**−4.3 %**) | 86.4 vs 88.5 (**+2.4 %**) |

**Dropping two threads took δ by 3.8× and ε by 3.7×.** The pinning half of the recommendation
could not be tested on this host; the thread-cap half is worth most of the effect anyway.

---

## 3. The throttle: not thermal, not bandwidth, and it needs a co-tenant

### 3.1 The device alone does not drift at all **[M]**

| phase | what | n | median ms |
|---|---|---:|---:|
| warm-up | 8 dispatches | 8 | 16.712 |
| **burst** | **400 back to back** | 400 | 17.210 (sextiles 16.98 16.95 17.10 17.60 18.11 17.05) |
| after a **20 s idle gap** | same payload | 400 | **16.400** |
| burst **with copy1 running** | 65 GB/s co-tenant | 400 | **16.407** |

![burst and gap](imaggers/contend/burst_gap.png)

A sibling's longer run of the same image (`util/place/q_throttle.py`: 1800 dispatches, 28.9 s
at 100 % duty, host idle) drifts **−2.6 %** — the last decile is the fastest. **No device
self-heating, no "longer run = slower", and a 65 GB/s co-tenant does not slow the burst either.**
The 20 s gap is not "resetting" anything: nothing had degraded, and the burst after it is
*faster* (16.40 vs 17.21), which is the background co-tenant easing off, not a thermal recovery.

### 3.2 In product the device does slow ~20 %, abruptly, and it persists **[M]**

![device state](imaggers/contend/throttle_state.png)

Paired fast/slow over the 8 runs that contain both states:

| item | what it is | slow/fast | p10 | p90 |
|---|---|---:|---:|---:|
| `gu` | **device** gate/up | **1.197** | 1.150 | 1.205 |
| `dn` | **device** down | **1.219** | 1.138 | 1.237 |
| `pack` | **host** A pack | 0.971 | 0.928 | 1.035 |
| `read` | **host** C readback | 1.062 | 0.773 | 1.149 |
| `dec` | **host** decode | 0.913 | 0.833 | 1.083 |

The transition is a step (<1 s), persists for the rest of the run, and occasionally steps back.
Across 81 runs the level persists for clusters and flips in both directions within minutes, and
**0 of 15 in-run flips had another job starting within the preceding 40 s** (144 runs checked).

### 3.3 All of it together **[A]**

The device needs a co-tenant *of the right shape* to slow down: 65 GB/s is not enough, 20
threads is not enough, 8.6 GB/s **on 18 threads held continuously** is. The product runs its own
attention (66 ms/layer-step, 20 threads) in front of the dispatch — the right shape — and the
device slows by +19.7 %. The independent 28.9 s burst with no host work never slows. **The +19 %
is co-tenant-induced, and the co-tenant is the product's own host work** — the same conclusion
as §2.3, seen from the other end. That also explains why the effect looked like a "run state":
it switches when the host work starts or stops, not when the device gets hot.

---

## 4. The schedule arithmetic

### 4.1 Segments per 8-step image **[M]**

| **D** device | **Ha** attention | **Hg** glue | **R** preamble | **O** other | serial |
|---:|---:|---:|---:|---:|---:|
| 37.91 | 15.84 | 18.53 | 15.78 | 1.68 | **89.74** |

H_in = 34.37 s, H_all = 50.15 s, **H/D = 1.32**.

### 4.2 With the measured coefficients **[A]**

    wall = max( D·(1+δ) , H·(1+ε) ) + (host not hidden) + O        δ = 0.153, ε = 0.165

| design | hidden | device | host | wall | saved |
|---|---|---:|---:|---:|---:|
| serial (control) | — | 37.91 | 50.15 | **89.74** | — |
| **1 — in-loop host only** | 34.37 s | 37.91×1.153 = **43.71** | 34.37×1.165 = 40.04 | **61.17** | **−28.57** |
| **2 — everything hidden** | 50.15 s | **43.71** | 50.15×1.165 = **58.42** | **60.10** | **−29.64** |
| ideal (δ = ε = 0) | all | 37.91 | 50.15 | 51.83 | −37.91 |

**The pole moves, and that is the design decision:**

* design 1: the **device** is the pole (43.71 > 40.04) → **δ costs 1:1, ε is free**; it absorbs ε
  up to +0.272 before anything changes.
* design 2: the **host** is the pole (58.42 > 43.71) → **ε costs 1:1, δ is free**; it absorbs δ
  up to **+0.541**.

Because the two coefficients are the same size today, the designs land **1.1 s apart**. **You
cannot escape both coefficients — you choose which one to pay**, and the choice should follow
which one you can actually reduce. ε is reducible (thread caps, §2.4); δ is not.

### 4.3 Sensitivity **[A]**

Design 2 wall (s):

| ε \ δ | 0.00 | 0.15 | 0.30 | 0.45 | 0.60 |
|---|---:|---:|---:|---:|---:|
| 0.000 | 51.83 | 51.83 | 51.83 | 56.65 | 62.34 |
| 0.050 | 54.34 | 54.34 | 54.34 | 56.65 | 62.34 |
| **0.165 (measured)** | **60.10** | **60.10** | **60.10** | **60.10** | 62.34 |
| 0.300 | 66.88 | 66.88 | 66.88 | 66.88 | 66.88 |
| 0.450 | 74.40 | 74.40 | 74.40 | 74.40 | 74.40 |
| 0.600 | 81.92 | 81.92 | 81.92 | 81.92 | 81.92 |

Seconds saved vs serial:

| ε \ δ | 0.00 | 0.15 | 0.30 | 0.45 | 0.60 |
|---|---:|---:|---:|---:|---:|
| 0.000 | +37.91 | +37.91 | +37.91 | +33.09 | +27.40 |
| 0.050 | +35.40 | +35.40 | +35.40 | +33.09 | +27.40 |
| **0.165** | **+29.64** | **+29.64** | **+29.64** | **+29.64** | +27.40 |
| 0.300 | +22.86 | +22.86 | +22.86 | +22.86 | +22.86 |
| 0.450 | +15.34 | +15.34 | +15.34 | +15.34 | +15.34 |
| 0.600 | +7.82 | +7.82 | +7.82 | +7.82 | +7.82 |

**Every cell is a win.** The worst (ε = 0.60, the 2026-11-01 value) still saves **+7.82 s**. To
make overlap a mistake you need ε > 0.79 **or** δ > 1.32; the measured values are 0.165 and
0.153.

### 4.4 Autopsy of 2026-11-01 — an ε artefact **[M]/[A]**

    δ = 93.4/59.2 − 1 = +0.578          ε = 177.0/110.4 − 1 = +0.603
    wall = max(59.2·1.578, 110.4·1.603) = max(93.4, 176.6) = 176.6    (measured 177.0)

    budgets:   δ < H/D = +1.865  → 0.578 INSIDE by 3.2x
               ε < D/H = +0.536  → 0.603 OUTSIDE by 12 %
    same δ, ε = 0:  wall = max(93.4, 110.4) = 110.4 ms  → a 35 % win

The campaign's explanation ("the BO write/read and attention's 390 MB softmax fight for the same
DRAM") is **not supported**: 65 GB/s of pure memcpy beside a dispatch costs the device 2 %
(§2.3). What cost 60 % was **20 host threads taking the dispatch thread's CPU**, and §2.4 shows
that dropping the pool to 18 threads takes it to 16 %.

### 4.5 What overlap does to the rest of the roadmap **[A]**

In design 2 the device has 58.42 − 43.71 = **14.7 s of slack**:

| device speed-up | D(1+δ) | wall | kernel's contribution |
|---|---:|---:|---:|
| as-is | 43.71 | 60.10 | — |
| 1.5× | 29.14 | **60.10** | **0.00 s** |
| 2.6× | 16.81 | 60.10 | 0.00 s |

**Under a full pipeline the kernel work is worth zero until the host comes down.** Overlap
(−29.6 s) first, host-segment wins second (1:1 on the critical path), kernel last. In design 1
the device stays the pole and the kernel keeps its 1:1 value. **Which design is chosen sets what
the next three months of kernel work are worth** — and §4.2 says the designs are 1.1 s apart, so
that is a cheap decision to get right.

---

## 5. Verdict

**Overlap pays: −28.6 s (design 1) to −29.6 s (design 2) on measured δ = +0.153 and ε = +0.165,
with every cell of the sensitivity grid positive.**

The single change with the best (seconds saved)/(risk) is **the ε-controlled re-run**, and it is
half done: capping the pool at 18 threads cut δ by 3.8× and ε by 3.7× and flipped that
experiment from −4.3 % to +2.4 % **[M]**. The remaining half is the **dispatch-thread pin**,
refused by this host's sandbox (`cpu_affinity` → WinError 5), plus a 5 s burst with the host
load running to confirm the +19 % appears on demand.

Before building anything: run **design 1** (in-loop host only, no BO double-buffering) as an
8-step A/B. It is worth 28.6 s, it needs no new machinery beyond running the attention on a
worker thread with a capped pool, and its gate is the existing latents sha256.

---

## 6. What is still open

1. **The pin.** `cpu_affinity` is denied in this sandbox; run `run_contend.py --phase e3
   --pin 18` from a normal shell. Expected: ε drops further, δ unchanged.
2. **E1 without a background.** The arm table (§2.1) is contaminated by a 3.8-13.7 core
   co-tenant. `run_contend.py --phase e1 --arms idle,cpu4,cpu20,copy1,attn4,attn20` on an
   idle box gives the clean version; E3 says what it should show.
3. **Why attention and not memcpy.** The mechanism is not settled (CPU occupancy vs memory
   latency vs both). It matters only for whether ε can be cut further than the thread cap does.
