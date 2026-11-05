# Where the plan path's 2-3x goes, and the three fixes that remove it

**2026-11-03 evening · 512² · plan arm vs campaign arm, back to back · py310 + numba 0.62.1 + pyxrt 0.62.1**

~~~
BEFORE   plan 27.80 s/step   campaign 17.60 s/step   ratio 1.58x   (AC, 512^2, 8 steps)
AFTER    plan 19.60 s/step   campaign 17.90 s/step   ratio 1.09x   (AC, 512^2, 8 steps)
         plan -29.5 %      8-step loop wall 235.63 s -> 158.87 s (-32.6 %)
         startup          ~31.7 s ->   5.2 s  (-26.5 s: the dead 3.71 GB delivered cache
                                                and its 25.7 s in-place B repack, replaced
                                                by a 5.2 s plan-payload prewarm)

CORRECTNESS IS UNCHANGED, AND THAT IS MEASURED, NOT ARGUED:
  1 step  plan after == plan before  corr 1.000000000 maxabs 0 exact True  (sha 3fd6236e608c31be)
          plan vs DELIVERED          corr 0.995965046  (gate >= 0.9953)  -- the recorded number, unchanged
  8 steps plan after == plan before  corr 1.000000000 maxabs 0 exact True  (sha 2de08e0127a1587e)
          plan vs campaign           corr 0.994423088  -- the recorded number, unchanged
~~~

Every change below is **bit-identical on the latents**; the two 1-step and 8-step arms above
are byte-equal to the pre-change arms of DET-BISECT-REPORT-2, so the correctness proof there
carries over without being re-derived.

---

## 1. The per-section table, plan vs campaign, ON THE SAME RUN

Both arms: `webui/bridge.py` via `run_step.ps1`, 512², 8 steps, seed 7, **AC**, the campaign env
(`ZIMAGE_BIGM=1 ZIMAGE_BIGM_PROJ=1 ZIMAGE_CHUNK_PIPE=0 T22_DEVKILL=1 T22_BIGM_DEVHANDOFF=1
T22_HOSTKILL=1`), no fastglue hooks, `T22_BIGM_DEBUG=1`.  The ONLY difference is `T22_PLAN_FFN`.
Columns are the product's own per-layer lines — `[planffn]` for the plan arm, `[devkill] rows=`
for the campaign arm — which print the SAME column names on purpose (`plan_ffn.py:294`).

`_scratch/util/place/section_table.py` produced this from the logs; medians over 240 lines
(30 layers x 8 steps).  Runs: `q1plan` / `q1camp`.

| column (median ms/layer) | PLAN | CAMPAIGN | delta |
|---|---:|---:|---:|
| `pack`   A pack (host)        |  15.20 |   9.90 |  +5.30 |
| `wgu`    write gate/up A+W    |   4.20 |   4.50 |  -0.30 |
| `gu`     gate+up dispatch     |  19.80 |  48.40 | -28.60 |
| `swiglu` silu(gate)*up        | **157.70** | **0.00** | **+157.70** |
| `wdn`    write dn A+W         |   3.10 |   0.00 |  +3.10 |
| `dn`     down dispatch        |  20.30 |  31.10 | -10.80 |
| `read`   read C from the BO   |  23.30 |   3.00 | +20.30 |
| `dec`    drain/untile C       |  30.40 |   1.30 | +29.10 |
| `barrier` same-image barrier  |  24.90 |   0.00 | +24.90 |
| **FFN section total**         | **301.10** | **99.35** | **+201.75** |
| `proj_qkv` 4 plain15 launches |  90 | 100 | -10 |
| `rope`                        |  20 |  20 |   0 |
| `attn`   host f32 attention   |  90 | 100 | -10 |
| `proj_o` 4 more plain15       |  60 |  70 | -10 |
| **attention total**           | **260** | **290** | **-30** |
| **B repack (NOT in any column)** | **~170** | **0** | **+170** |
| **step wall**                 | **27 800 ms** | **17 600 ms** | **+10 200** |

The closure is exact: 301.10 - 99.35 = +201.75 ms/layer = +6.05 s/step of FFN section, and
27.80 - 9.03 (FFN) - 7.80 (attention) = 11.00 s/step of remainder for the plan arm against
17.60 - 2.98 - 8.70 = 5.92 for the campaign: **+5.1 s/step that no section line shows**.
`probe_plan_speed.py` prices exactly that remainder at **209 ms/layer** (`payloads_for` =
`load_blocks` 190.7 ms of which 117.5 ms is the `weights/block<N>.npz` disk read, plus
`pack_payloads` 32.6 ms) = 6.3 s/step — it lands in the step wall because `plan_ffn.ffn()`
calls `payloads()` BEFORE `s2_ffn32.ffn32_run()` starts its timer.

### So where the 2-3x goes

| share of the 10.2 s/step gap | s/step | what it is |
|---|---:|---|
| the host silu(gate)*up | 4.73 | the plan's `gu` image is a PURE GEMM, so the swiglu the delivered `swiglu11` image folds into its epilogue runs on the host over 10.8 M elements |
| the B payload re-derivation | 5.10 | `T22_PLAN_B_LAYERS` defaulted to **3** slots for **30** layers: every layer MISSES the LRU every step and re-reads 3.98 GB of `.npz` per step |
| the barrier | 0.75 | the same-image relaunch guard between the gate and up launches (F-01) |
| the host C read+drain | 1.48 | `read` + `dec` + `pack`: the plan's C is the 32-core tile permutation, the delivered image writes row-major C |
| the plan's device time | **-1.45** | the plan's 32-core kk=32 images are genuinely FASTER on the device |

**The plan path is not slow because of its dispatch structure — it is slow because its image
family is a pure GEMM, so everything the delivered image does in silicon (swiglu, the C
layout) and everything the delivered cache does for free (a resident B payload) lands on the
host, once per layer, every step.**

## 2. The structural difference, counted and priced

| | plan (`xd_m48_kk32_Mblk1056`) | delivered/campaign (RT=11, G=15) |
|---|---|---|
| geometry | kk=32, m=48, n=320 (gu) / 384 (dn), 22 of 32 tiles | kk=8, m=96, n=512/384, 11 blocks |
| FFN launches per layer | **4** (gate, barrier, up, dn) | **2** (swiglu11, dn11) |
| projection launches per layer | 4 (`plain15` q,k,v,o) | 4 (identical code) |
| device ms per layer (FFN) | **40.1** (gu 19.8 + dn 20.3) | **79.5** (gu 48.4 + dn 31.1) |
| device ms per layer (proj) | 62 (4 x 15.5) | 62 (4 x 15.5) |

The plan does **2 more launches per layer** and they cost **~35 ms/layer**: the barrier is
24.9 ms (measured, its own column) and the second `gu` launch is inside the 19.8 ms `gu`
column (two launches, ~10 ms each).  That is 1.05 s/step, i.e. **10 % of the gap**.  Meanwhile
the plan's FFN device time is **39 ms/layer LOWER**.  Dispatch count is not the problem; the
per-dispatch price is not the problem; the plan is *cheaper* on the device.

## 3. The C drain and the A pack, measured separately

`probe_plan_speed.py` (host only, py310, median of 5):

| component | ms | x per layer | x 30 layers |
|---|---:|---:|---:|
| `drain_c` -> `unpack_c_plan_numba` | 18.42 | 2 (gu C + dn C) | 1.11 s/step |
| `drain_c` -> `unpack_c_plan_fast` (numpy bulk) | 77.49 | 2 | 4.65 s/step |
| `drain_c` -> `unpack_c_plan` (gather spec) | 110.15 | 2 | 6.61 s/step |
| `pack_a_kk` (A, Mblk x 3840) | 3.60 | 2 | 0.22 s/step |
| `pack_b_kk` w1 / w2 | 10.82 / 11.31 | 3 | 0.98 s/step |
| `load_blocks` (npz read + 3 tile gathers) | 190.66 | 1 | 5.72 s/step |
| `payloads_for` (what the product calls) | 209.22 | 1 | **6.28 s/step** |

So the C drain is 0.9 s/step and the A pack 0.2 s/step — together 11 % of the plan's step.
The **B payload path is 6.3 s/step, 3x the drain and 30x the A pack**, and it was invisible.

## 4. Is numba actually used in the product (py310)?  YES — all three kernels

`probe_plan_speed.py` prints the state it found in the product's own interpreter:

~~~
[ps] numba: {'version': '0.62.1', 'vendor_gather': True, 'unpack_c': True,
             'pack_a_kk': True, 'drain_mode': 1}
~~~

i.e. `vendor_blocks` uses the tile-gather kernel (11.05 ms vs **57.94 ms** for the numpy
fancy index), `drain_c` uses `unpack_c_plan_numba` (18.42 ms vs 77.49 for the bulk
transpose), and `pack_a_kk` uses its kernel.  **There is no silent pure-python fallback** —
the constant was the npz re-read, not a fallback.

## 5. What I changed

### 5.1 `_scratch/util/place/plan_ffn.py` — the B payload stays resident (the 5.1 s/step)

`T22_PLAN_B_LAYERS` defaulted to **3** and a model has **30** layers, so the LRU evicted a
layer's payload before the next step could reuse it: 30 misses per step, 240 per 8-step run.
It is now a **byte budget** (`T22_PLAN_B_BYTES`, default 6 GiB) plus an optional count limit
(`T22_PLAN_B_LAYERS=0` = no count limit, the new default).  30 x 132.7 MB = 3.71 GiB, which
fits.  New `prewarm()` fills it at startup (fail-open on MemoryError).

### 5.2 `_scratch/image/t18_run.py` — stop paying for a cache the plan never reads (the memory)

With `T22_PLAN_FFN=1`, `block_step` sets `_c = _b = None`: the delivered 3.71 GB resident B
cache and the 5.31 GB in-place B-order repack are **never read**.  They were still built —
3.71 GB of RAM and 25.7 s of startup — so the plan's own 3.71 GiB had nowhere to live.
Both are now skipped for the plan arm, and the plan's payload prewarm takes that memory.
`bg, b3 = None, None` for `PLAN_FFN` (they are unused by the BIGM FFN branch), and the
`CACHE_ALL` branch gained a `CACHE_BG is not None` guard.

### 5.3 `_scratch/util/place/s2_ffn32.py` — the host swiglu, 4.6x, bit-identical

`h = _bf16(_bf16_sigmoid_silu(gate) * up)` materialised **thirteen** 43.3 MB temporaries.
`swiglu_host()` does the same operations in the same order on two reused scratch buffers:
numpy's own `exp` ufunc (in place — bit-identical by construction, it IS the ufunc the
shipped line calls) plus one numba pass that finishes `silu*up` and the RNE bf16 rounding.

| variant | ms/layer | exact vs shipped |
|---|---:|---|
| shipped numpy expression | 105.5 | - |
| numpy in-place scratch | 43.5 | **exact** (ndiff 0 / 10 813 440) |
| **`swiglu_host` (shipped now)** | **19.8** | **exact** (ndiff 0) |
| all-numba (exp inside) | 7.4 | 16 / 10.8 M differ by 1 bf16 ulp — **NOT used** |

The all-numba form is 2.7x faster again and is deliberately rejected: numba's `exp` is not
numpy's, and 1 ulp of f32 flips a bf16 rounding on ~1.5e-5 of elements.  This path must stay
bit-identical to the expression DET-BISECT-REPORT-2 section 6 was proved on.

Also in this file: `unpack_c_plan_numba` no longer copies the C payload a second time
(`bytes(memoryview(raw).cast("B"))` on an object that is already `bytes`): 31.5 MB per
dispatch, 63 MB per layer, 1.9 GB per step, for nothing — the kernel only reads it.

## 6. The new numbers

**AC** (the launcher records `power=Online`; section 12 of RUNBOOK.md: AC and battery numbers
are never compared to each other, and corr/maxabs/sha are immune to the state anyway).

| | before | after |
|---|---:|---:|
| 512² 8-step loop wall | 235.63 s | **158.87 s** |
| 512² steady s/step (median of steps 1..7) | 27.80 | **19.60** |
| 512² step 0 | 36.72 s | **20.73 s** |
| 512² 1-step step 0, AC only | (not measured on AC before) | 22.76 s |
| FFN section | 301.10 ms/layer | **186.00 ms/layer** |
| `swiglu` | 157.70 ms/layer | **34.15 ms/layer** |
| `dec` | 30.40 ms/layer | **19.60 ms/layer** |
| B repack (invisible) | ~170 ms/layer | **0** |
| startup: delivered cache + B repack | 6.0 + 25.7 s | **0** |
| startup: plan payload prewarm | 0 (lazy, in-loop) | 5.2 s |

### the same table AFTER, measured as a within-pair on the same run conditions

Both arms re-run back to back on AC with the same 8-step schedule and the same pe
(`q3plan` / `q3camp`), i.e. the same pair that produced the "before" table above, so the
two pairs differ only in the pe file (the 13-token caption embedding).  The campaign arm's
own step wall moved 17.60 -> 17.90 s across that change, which bounds the confound at ~2 %.

| column (median ms/layer) | PLAN after | CAMPAIGN after | delta |
|---|---:|---:|---:|
| `pack`   A pack             |  11.40 |  10.50 |  +0.90 |
| `wgu`                       |  11.90 |   4.40 |  +7.50 |
| `gu`     gate+up dispatch   |  24.20 |  48.20 | -24.00 |
| `swiglu` silu(gate)*up      | **34.15** | **0.00** | **+34.15** |
| `wdn`                       |   6.20 |   0.00 |  +6.20 |
| `dn`     down dispatch      |  23.80 |  30.70 |  -6.90 |
| `read`   read C             |  26.80 |   3.10 | +23.70 |
| `dec`    drain C            |  19.60 |   1.30 | +18.30 |
| `barrier`                   |  30.30 |   0.00 | +30.30 |
| **FFN section total**       | **186.00** | **99.00** | **+87.00** |
| **attention total**         | **300** | **300** | **0** |
| **B repack (invisible)**    | **0** | **0** | **0** |
| **step wall**               | **19 600 ms** | **17 900 ms** | **+1 700** |

Closure: 19.60 - 5.58 (FFN) - 9.00 (attention) = 5.02 s/step of remainder for the plan arm;
17.90 - 2.97 - 9.00 = 5.93 for the campaign.  **The plan arm's invisible remainder is now
0.9 s/step SMALLER than the campaign's**, and the whole remaining difference is the 87 ms/layer
of FFN host work the delivered image does in silicon.

The attention row moved 260 -> 300 ms/layer between the two pairs in BOTH arms (the
`[plain15]` run median 15.5 -> 18.1 ms) because another session held 11.8 GB and a device job
during `q3plan`; it is ambient load, not code, and it cancels inside each pair.

**So: 27.80 -> 19.60 s/step, and plan/campaign 1.58x -> 1.09x.**  Against the recorded
campaign arm at the same shape and env (`p6camp8`, **battery**, 16.80 s/step) the ratio is
1.17x, but that is a battery number and is quoted only as a magnitude.

**Correctness, re-measured after the change:**

| pair | corr | maxabs | verdict |
|---|---:|---:|---|
| plan after vs plan before, 1 step | 1.000000000 | 0 | exact (sha 3fd6236e608c31be both) |
| plan after vs plan before, 8 steps | 1.000000000 | 0 | exact (sha 2de08e0127a1587e both) |
| **plan vs DELIVERED, 512², 1 step** | **0.995965046** | 1.24808 | **gate >= 0.9953 PASS** |
| plan vs campaign, 512², 8 steps | 0.994423088 | 1.43012 | unchanged |
| plan vs campaign, 512², 1 step | 0.998417455 | 0.631815 | unchanged |
| **campaign after vs campaign before, 1 step** | **1.000000000** | **0** | **exact -- the delivered path is untouched** |

## 7. Files touched

**Modified (3, all product code)**

| file | what |
|---|---|
| `_scratch/util/place/plan_ffn.py` | byte-budgeted resident payload cache (default: every layer), `prewarm()`, `_lru_clear`/`_lru_trim`/`_payload_bytes` |
| `_scratch/image/t18_run.py` | `PLAN_FFN` skips the delivered `--cache-all` pass and the in-place B-order repack (and prewarms the plan's payloads instead); `bg, b3 = None, None` for `PLAN_FFN`; `CACHE_ALL` branch guarded on `CACHE_BG is not None` |
| `_scratch/util/place/s2_ffn32.py` | `swiglu_host()` + `_swiglu_tail_kernel()` + `_swiglu_bufs()`, used by `ffn32_run`; `unpack_c_plan_numba` no longer copies the C payload |

**New (measurement only, no product path)**

| file | what |
|---|---|
| `_scratch/util/place/probe_plan_speed.py` | prices every host constant of the plan path in the product's interpreter, including the ones no section line shows |
| `_scratch/util/place/probe_swiglu.py` | the swiglu variants with a bit-exactness column |
| `_scratch/util/place/section_table.py` | the per-section table above, from a run log |
| `_scratch/util/place/probe_dummy_rule.py` | proves the `gate_numba_equiv` dummy-slot check is the gate's bug (section 9) |
| `_scratch/util/place/run_pair.ps1` | plan arm then campaign arm, back to back, same power state, `T22_BIGM_DEBUG=1` |
| `_scratch/util/place/probe_payload_cache.py` | the resident payload cache: hit identity, byte accounting, budget eviction |
| `PLAN-SPEED-REPORT.md` (repo root) | this report |

**No shipped flag default changed on the delivered path.**  `T22_PLAN_B_LAYERS` default moved
3 -> 0 ("no count limit") and `T22_PLAN_B_BYTES` (new, 6 GiB) and `T22_PLAN_PREWARM` (new, 1)
exist only inside the plan arm.  The delivered/campaign path is untouched: `s2_ffn32` is
never imported there, `plan_ffn` is never imported there, and every `t18_run` edit is behind
`PLAN_FFN` or is a `None` guard that cannot fire when `CACHE_BG` is set (it always is on the
delivered path).  PROVED, not argued: the campaign arm was re-run after the change and its
latents are **byte-identical** to the pre-change arm -- `q2camp` == `p3camp`, sha
`81068771d7865661` both, corr 1.000000000 maxabs 0.  And `q2camp` vs `q2plan` is
0.998417455, the recorded 1-step plan-vs-campaign number.

## 8. Reproduce

~~~powershell
cd D:\Project\deepseek_harness\public_work\zimage-bfp16-npu
py310 _scratch\util\place\probe_plan_speed.py --m 1056     # where the host seconds go
py310 _scratch\util\place\probe_swiglu.py --m 1056          # the swiglu, with exactness

# the pair (device).  -Sched is the 8-step schedule; the pe stays at run_step's default,
# which is what p1plan / p5plan8 / p6camp8 / pGdel used.
pwsh _scratch\util\place\run_pair.ps1 -Size 512 -Steps 8 -Pre q4 `
     -Sched webui\jobs\20261103-152409-e8e997\schedule.npz
py310 _scratch\util\place\section_table.py _scratch\image\_run_q4plan.log `
                                             _scratch\image\_run_q4camp.log
py310 _scratch\util\place\lat_cmp.py q4plan p5plan8 p6camp8
~~~

## 9. Two defects found on the way (not fixed — reported)

1. **`gate_numba_equiv.py:96-101` asserts the wrong dummy-slot rule** and therefore reports
   22 FAILURES at every M ("dummy **BAD**").  It compares every dummy slot's A leaf against
   tile 0's leaf at **k-step 0**; `pack_kk.pack_a_kk_ref` (pack_kk.py:229-231) puts tile 0's
   leaf at the slot's **own k-step**.  `probe_dummy_rule.py` prints both: the spec rule holds
   (`True`), the gate's rule does not (`False`).  The numba A pack is byte-identical to the
   numpy spec in all 22 rows — the gate is red for a reason that has nothing to do with the
   packer.  Left alone deliberately: a speed task must not edit the gate it is measured by.
2. **`T22_PLAN_B_LAYERS`'s default of 3 was a silent 6.3 s/step tax** with no log line, because
   `payloads()` runs outside `ffn32_run`'s timer.  Now resident, and `prewarm()` prints the
   layer count and the GB so the state is visible in the log.

## 10. What is left, if anyone wants the last 2 s/step

| item | ms/layer | s/step | how |
|---|---:|---:|---|
| the barrier | 24.9 | 0.75 | only `T22_PLAN_FUSE_GU=1` removes it (one gu launch, silu on device). The fused image for M=1056 **is already built** (`ad_xd_m48k32_M1056_n160f_a1b1_gu.xclbin`); it is a DIFFERENT numeric arm, so it needs its own correctness run |
| `read` + `dec` + `swiglu` fused into one numba pass | ~40 | 1.2 | would have to take exp out of numpy, i.e. give up bit-identity |
| the projection's 8 launches/layer | 0 | 0 | `bigm_attention` calls `bigm_proj4(...)` for q,k,v AND again for o, so 4 of the 8 dispatches are computed and discarded — in BOTH arms, so it is not part of this gap |