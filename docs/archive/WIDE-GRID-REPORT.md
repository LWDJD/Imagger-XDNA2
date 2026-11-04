# WIDE-GRID REPORT — landscape / portrait aspect ratios on the adaptive NPU path

**2026-11-03 · T22 wide · new module + new plan manifests + new images only; the delivered default path is untouched**

```
THE ONE SENTENCE: 1920x1080 is not a square and not even a multiple of the
16-px patch grid -- it pads to 1920x1088, becomes M=8192, gets a legal 32-core
placement (xd m=64 kk=32 Mblk=2048 x4, L1 58,624 B), BUILDS three images and
runs on the device at corr 0.99999835 with the negative control collapsed.
Eleven of the twelve requested sizes are planned AND built; 4K is refused for
two exact reasons.  The cost model says the NPU is no longer the wall at these
sizes: host attention is 74 % of the step at 1080p.
```

---

## 0. The user's requirement, and what was actually missing

> "I also need other NON-SQUARE resolutions, e.g. 1920x1080, and other
> non-integer ones."

The planner already placed ANY M in 64..16,416, and the synthesizer built any
plan.  What had never been done:

| # | gap | why it mattered |
|---|---|---|
| 1 | every validated size was square or a small 2:1/3:2 rectangle (256x512, 384x512, 512x768) | no true 16:9 had ever been planned, built or run |
| 2 | the pad rule was documented in `webui/shapes.py` but never stated as a LAW with its exact safe set | "which sizes can I type?" had no one-sentence answer |
| 3 | the planner's cost model is the NPU GEMM half only | at M=8192 the host attention term is ~3x the device term, so the plan's number is the optimistic half of the truth |
| 4 | nobody had run a plan with ndisp > 2 on the device | 1920x1080 is 4 dispatches of 2048 rows |

---

## 1. THE PADDING LAW

### 1.1 The rule (each side independently, then crop the output back)

```
H_run = ceil(H / 16) * 16              # the image grid is 16-px patches
W_run = ceil(W / 16) * 16
gh, gw = H_run / 16, W_run / 16        # patch cells
T_img  = gh * gw                       # image tokens
T_pad  = ceil(T_img / 32) * 32         # SEQ_MULTI_OF = 32 (x_pad_token rows)
M      = 32 + T_pad                    # 32 caption rows + the image rows
deliver: decode H_run x W_run, keep [0:H, 0:W]      # "bottom-right trim"
```

Two roundings, and they are independent: the **geometry** rounds each
side up to 16 (because a patch is 16 px), and the **sequence** rounds the token
count up to 32 (because the product's `SEQ_MULTI_OF` is 32).  A shape can need
either, both, or neither.

### 1.2 1920x1080, worked to the byte

| step | arithmetic | value |
|---|---|---|
| height on the patch grid | 1080 / 16 = **67.5** -> ceil -> 68 | 1088 (8 px added) |
| width on the patch grid | 1920 / 16 = 120 exactly | 1920 (no pad) |
| patch grid | 120 x 68 | 8160 cells |
| image tokens | 120 * 68 | 8160 |
| token pad to 32 | 8160 / 32 = 255 exactly | 8160 (no pad) |
| unified sequence | 32 caption + 8160 | **M = 8192** |
| delivery | decode 1920x1088, keep [0:1080, 0:1920] | 1920x1080 |

So 1920x1080 is *exactly* the case the user named: one side lands on a half
patch, the other does not, and the token count happens to be 32-aligned.  A
16:9 that needs BOTH roundings is 1600x900 (900 -> 912, 5700 -> 5728 -> M=5760);
one that needs neither is 1280x720 (80 x 45 = 3600 -> 3616 -> M=3648 -- the
*token* pad fires even though both sides are already 16-aligned).

### 1.3 Exactly which sides and values are safe

| rule | statement |
|---|---|
| **per side** | **any integer in 64..2048 inclusive** (`MIN_HW`..`MAX_HW`). Not required to be a multiple of 16, 32 or anything else. |
| **token budget** | `ceil(H/16) * ceil(W/16) <= 16384` (then padded to 32, then +32 caption) |
| **equivalently** | `M <= 16416`, the product's `MAX_M` |
| **corner** | 2048x2048 -> 128 x 128 = 16,384 tokens -> M = 16,416 (exactly the ceiling) |
| **16:9 corner** | 2048x1080 -> 128 x 68 = 8,704 -> M = 8,736; the *side* limit (2048), not the token budget, is what stops a wider 16:9 |
| **not safe** | any side < 64; any side > 2048 (**4K is refused here**); `ceil(H/16)*ceil(W/16) > 16384` |
| **delivery** | decode the padded frame, keep `[0:H, 0:W]`; a 16-multiple request makes no crop call at all, so its PNG bytes and sha256 are unchanged |

Every side value is safe in the sense that **no requested pixel is lost**: the
run frame is always >= the request, and the crop is the top-left rectangle.

### 1.4 The law is the product's own arithmetic (not a second implementation)

`_scratch/util/place/_wide_law_check.py` compares `npu_wide.pad_law()` against
`webui.shapes.budget()` / `m_from_hw()` on the 12 requested sizes, the envelope corners
(2048x2048, 2048x16, 64x2048, 2048x2064, 3840x2160, 2048x1088, 64x64, 2049x2049,
63x1024) and **2000
random (H, W) pairs** drawn from 64..2048:

```
== the wide grid ==            (12 sizes, all agree)
== envelope corners ==         (9 corners, all agree)
== random sample (2000 pairs) ==
0 disagreement(s)
```

`run_h/run_w`, `image_tokens`, `padded_tokens`, `M`, `pad_applied` and the crop box agree on all
2,021 cases.

### 1.5 The product ALREADY reports "padded" and "crop" in the job record

No new plumbing was needed for the delivery half -- it shipped with T22 pad_crop
(2026-10-21) and there is a real landscape job in this tree that proves it.

**The code contract** (`webui/runner.py`):

| line | what it does |
|---|---|
| 261-266 | the job record carries three separate size records: `requested`, `padded`, `output` |
| 203-219 | `_crop_png_top_left()` -- `im.crop((0, 0, out_w, out_h))`; a 16-multiple run returns before touching the file |
| 1621-1667 | the crop block: per image `rec["crop"] = {policy, box, run_*, output_*, seconds}`, the file renamed `*_crop<W>x<H>.png`, the sha256 re-sealed, and a job-level `crop` record with `semantics` |
| 1850-1853 | the params carry `pad_applied`, `crop_applied`, `pad_height/width`, `output_height/width` |

`webui/shapes.py:867-869` is where the flag is decided:
`plan["crop_applied"] = bool(b["pad_applied"])` and
`plan["crop"] = {"policy": "bottom-right-trim", "box": [0, 0, w, h], ...}`.

**The record** -- `webui/jobs/20261021-170817-cd6a58/job.json`, a real 480x854
(landscape) request:

```json
"requested": {"height": 480, "width": 854},
"padded":    {"height": 480, "width": 864},      <- 854 -> 864 (one 16-px cell)
"output":    {"height": 480, "width": 854},
"crop": {"policy": "bottom-right-trim", "box": [0, 0, 854, 480],
         "requested_width": 854, "requested_height": 480,
         "run_width": 864, "run_height": 480,
         "pad_width": 864, "pad_height": 480, "seconds": 0.0878,
         "semantics": "generate the padded 16-px frame, then keep the
                       requested top-left H x W rectangle; right/bottom
                       excess is discarded"}
"images": [{"file": "chunk480_170817cd6a58_p0_s7_crop854x480.png",
            "width": 854, "height": 480,
            "run_width": 864, "run_height": 480,
            "crop": {"policy": "bottom-right-trim", "box": [0, 0, 854, 480],
                     "run_width": 864, "seconds": 0.0878},
            "sha256": "af4ccd81102ae820..."}]
```

The delivered PNG is 854x480, not 864x480: the crop is executed, not just
recorded.  (The job-level `crop_applied` flag lives in `params.crop_applied` and in
the `/api/plan` echo; the top-level record carries `requested/padded/output` and the
`crop` block.)

---

## 2. THE WIDE GRID: planned AND built

Twelve sizes, planned by `webui/npu_plan.plan(M)` (with the built-plan affinity the
product uses) and built by `_scratch/util/place/npu_wide.py --build`.  **BUILT** below
means three aiecc images (gu / dn / proj) exist and a `plan_<key>.json` manifest
records their sha256 -- a claim without a build is not evidence.

| request | run (padded) | grid | img tok | padded | M | plan | n gu/dn/proj | L1 B (compiler) | built |
|---|---|---|---|---|---|---|---|---|---|
| **1920x1080** | 1920x1088 | 120x68 | 8160 | 8160 | **8192** | xd m=64 kk=32 Mblk=2048 x4 T=1 | 320/320/320 | 58624 (=58624) | **BUILT** |
| 1280x720 | 1280x720 | 80x45 | 3600 | 3616 | 3648 | t33 m=64 kk=32 Mblk=3648 x1 T=2 | 160/160/160 | 56960 (=52864) | **BUILT** |
| 1024x576 | 1024x576 | 64x36 | 2304 | 2304 | 2336 | t33 m=16 kk=32 Mblk=2336 x1 T=5 | 256/256/256 | 57344 (=53248) | **BUILT** (cached: 768x768 plan) |
| 1600x900 | 1600x912 | 100x57 | 5700 | 5728 | 5760 | xd m=64 kk=32 Mblk=1920 x3 T=1 | 320/320/320 | 58624 (=58624) | **BUILT** |
| 2048x1080 | 2048x1088 | 128x68 | 8704 | 8704 | 8736 | xd m=48 kk=32 Mblk=1488 x6 T=1 | 320/384/384 | 55808 (=55808) | **BUILT** |
| 1080x1920 | 1088x1920 | 68x120 | 8160 | 8160 | 8192 | xd m=64 kk=32 Mblk=2048 x4 T=1 | 320/320/320 | 58624 (=58624) | **BUILT** (same images as 1920x1080) |
| 3840x2160 | -- | -- | -- | -- | (32448) | **REFUSED** | -- | -- | **side envelope** (height=3840 > 2048) AND **token envelope** (M=32448 > 16416) |
| 512x288 | 512x288 | 32x18 | 576 | 576 | 608 | xd m=32 kk=32 Mblk=608 x1 T=1 | 512/480/480 | 55296 (=55296) | **BUILT** (cached: 384x384 plan) |
| 288x512 | 288x512 | 18x32 | 576 | 576 | 608 | xd m=32 kk=32 Mblk=608 x1 T=1 | 512/480/480 | 55296 (=55296) | **BUILT** (same images as 512x288) |
| 1366x768 | 1376x768 | 86x48 | 4128 | 4128 | 4160 | t33 m=48 kk=32 Mblk=4176 x1 T=3 | 160/160/160 | 63104 (=56960) | **BUILT** |
| 1080x1080 | 1088x1088 | 68x68 | 4624 | 4640 | 4672 | t33 m=32 kk=32 Mblk=4672 x1 T=5 | 128/128/128 | 57856 (=49664) | **BUILT** |
| 1440x1080 | 1440x1088 | 90x68 | 6120 | 6144 | 6176 | t33 m=16 kk=16 Mblk=6176 x1 T=13 | 128/128/128 | 64256 (=58112) | **BUILT** |

**11 of 12 planned and BUILT; 9 distinct plan keys** (the two 608s share one, and
1920x1080 / 1080x1920 share one -- the plan key is the cache key, so a portrait
request costs no second build).

The L1 column is the planner's number with the compiler's own printed value in
parentheses.  Where they differ it is the `t33` design's own short formula
(it charges `a_leaf` where the placer must fit `T * a_leaf`); the difference is
exactly `(T-1) * a_leaf`:

| size | T | a_leaf = m*kk*2 | (T-1)*a_leaf | plan - compiler |
|---|---|---|---|---|
| 1280x720 | 2 | 4,096 | 4,096 | 4,096 |
| 1024x576 | 5 | 1,024 | 4,096 | 4,096 |
| 1366x768 | 3 | 3,072 | 6,144 | 6,144 |
| 1080x1080 | 5 | 2,048 | 8,192 | 8,192 |
| 1440x1080 | 13 | 512 | 6,144 | 6,144 |

aiecc accepted every one of them, so the **corrected** gate is the one that
matches the toolchain.  The `xd` plans print the planner's number exactly.

### 2.1 4K: the exact refusal

```
3840x2160: each side must be 64..2048 (MIN_HW..MAX_HW); height=3840 is outside
```

and if the side limit did not exist it would still be refused on the tokens:
240 x 135 = 32,400 -> padded 32,416 -> **M = 32,448 > 16,416**.

### 2.2 What was built, byte for byte

| plan key | M | Mblk | images (gu/dn/proj) | xclbin | insts |
|---|---|---|---|---|---|
| `xd_m64_kk32_Mblk2048_n320-320-320_653b0019` | 8192 | 2048 | `ad_xd_m64k32_M2048_n320_a1b1_{gu,dn,proj}` | 94,734 B | 79,248 / 28,608 / 28,608 B |
| `t33_m64_kk32_Mblk3648_n160-160-160_7337da64` | 3648 | 3648 | `ad_t33_m64k32_M3648_n160_a1b1cw_*` | 99,934 B | 233,744 / 86,544 / 86,544 B |
| `xd_m64_kk32_Mblk1920_n320-320-320_13278b50` | 5760 | 1920 | `ad_xd_m64k32_M1920_n320_a1b1_*` | 94,734 B | 79,248 / 28,608 / 28,608 B |
| `xd_m48_kk32_Mblk1488_n384-320-384_63a5c5a7` | 8736 | 1488 | `ad_xd_m48k32_M1488_n{320,384,384}_a1b1_*` | 94,734 B | 79,248 / 23,544 / 23,544 B |
| `t33_m48_kk32_Mblk4176_n160-160-160_c87b5458` | 4160 | 4176 | `ad_t33_m48k32_M4176_n160_a1b1cw_*` | 105,070 B | 233,744 / 86,544 / 86,544 B |
| `t33_m32_kk32_Mblk4672_n128-128-128_40569abe` | 4672 | 4672 | `ad_t33_m32k32_M4672_n128_a1b1cw_*` | 116,046 B | 292,624 / 108,624 / 108,624 B |
| `t33_m16_kk16_Mblk6176_n128-128-128_c9030a37` | 6176 | 6176 | `ad_t33_m16k16_M6176_n128_a1b1cw_*` | 129,774 B | 292,624 / 108,624 / 108,624 B |
| `xd_m64_kk32_Mblk1856_n320-320-320_d55f76dc` | 9248 | 1856 | `ad_xd_m64k32_M1856_n320_a1b1_*` (2048x1152, section 5.3) | 94,734 B | 79,248 / 28,608 / 28,608 B |

`.prj` directories were deleted after every build; the single file kept in each is
`kernels_main.json` (the BO-count record `run_ffn` reads).  Eight
`webui/shape_manifests/plan_<key>.json` manifests were written (one per plan key),
each carrying the three images' sha256 and `built_ok: true`.

---

## 3. THE PER-STEP COST MODEL: the NPU half and the host half

### 3.1 The planner's estimate covers only half the step

`npu_plan.layer_ms()` is the **32-core GEMM half**:

```
dispatch_ms = max( T * 2*m*K*N / (rate(kk)/32) ,  (ntiles*a_bytes + b_bytes + c_bytes) / 42 GB/s ) + 1.0 ms
layer_ms    = (2*gu + 1*dn + 4*proj) * ndisp
step_s      = layer_ms * 30 / 1000
```

`rate = {8: 3.78, 16: 4.60, 32: 5.12}` TFLOP/s (measured), BW = 42 GB/s (measured),
1.0 ms per dispatch.  Attention is **not** in it -- and above SEQ=304 there is no
attention image at all, so attention runs on the host in f32 torch, one 96-row
query block at a time.  Its cost is the same anchor `webui/shapes.chunk_estimate()`
uses (measured 190-700 GFLOP/s, 600 used):

```
attn_ms_per_layer = 4 * N^2 * HD(128) * HEADS(30) / 600e9 * 1e3
attn_step_s       = attn_ms_per_layer * 30 / 1000
total_step_s      = step_s + attn_step_s          # what npu_wide reports
```

So the number a user should believe for a wide size is `total_step_s`, not
`step_s`.  `_scratch/util/place/npu_wide.py` prints both, plus the share.

### 3.2 The estimate, size by size

| size | M | 96-row blocks | device s/step | host attention s/step | total s/step | attention share | attention vs 512² | verdict |
|---|---|---|---|---|---|---|---|---|
| 512x512 (baseline) | 1056 | 11 | 3.40 | 0.86 | 4.26 | 20 % | 1.00x | NPU |
| 512x288 | 608 | 7 | 2.33 | 0.28 | 2.61 | 11 % | 0.33x | **NPU** |
| 288x512 | 608 | 7 | 2.33 | 0.28 | 2.61 | 11 % | 0.33x | **NPU** |
| 1024x576 | 2336 | 25 | 5.52 | 4.19 | 9.71 | 43 % | 4.87x | **NPU** |
| 1280x720 | 3648 | 38 | 8.70 | 10.22 | 18.92 | 54 % | 11.88x | attention |
| 1366x768 | 4160 | 44 | 9.77 | 13.29 | 23.06 | 58 % | 15.45x | attention |
| 1080x1080 | 4672 | 49 | 10.83 | 16.76 | 27.59 | 61 % | 19.49x | attention |
| 1600x900 | 5760 | 60 | 13.37 | 25.48 | 38.85 | 66 % | 29.63x | attention |
| 1440x1080 | 6176 | 65 | 15.57 | 29.29 | 44.86 | 65 % | 34.06x | attention |
| **1920x1080** | **8192** | **86** | **17.83** | **51.54** | **69.37** | **74 %** | **59.93x** | **attention** |
| 1080x1920 | 8192 | 86 | 17.83 | 51.54 | 69.37 | 74 % | 59.93x | attention |
| 2048x1080 | 8736 | 91 | 20.37 | 58.61 | 78.98 | 74 % | 68.15x | attention |

Every number is an ESTIMATE anchored on measured rates, never a promise.

### 3.3 Which sizes are dominated by attention, plainly

**Eight of the eleven runnable sizes are attention-dominated**; the three that are
not are the small ones (512x288, 288x512, 1024x576).

The crossover is arithmetic, not opinion.  Per row of sequence:

* the device costs **2.18-2.52 ms/row** for every size above M=2,000 (17.83 s /
  8192 rows, 13.37 / 5760, 10.83 / 4672; the 608-token size is 3.83 ms/row because
  its fixed dispatch cost dominates);
* the host attention costs **`7.68e-7 * M` ms/row** (it is O(M) per row because
  the total is O(M^2)).

```
7.68e-7 * M = 2.3e-3   ->   M* = 2,995
```

**Above M ~ 3,000 tokens (~680x680) the host attention is the wall, not the NPU.**
At M=8192 it is 6.29 ms/row against the device's 2.18 -- **2.9x the NPU**, and
**60x the 512² attention term** (N=8192 vs N=1056; (8192/1056)^2 = 60.2).

The "96-row blocks" framing is the delivery detail that makes this concrete: the
host path walks the sequence in 96-row query blocks, so 1080p is **86 blocks per
layer** (2,580 blocks per 30-layer step) against 11 at 512².  The block count
grows linearly and the work inside each block grows linearly too -- hence the
quadratic total.

### 3.4 The product's own number is a different (and larger) estimate

`webui/shapes.chunk_gate()` reports **398.45 s/step** for 1920x1080, not 69.37.
Both are honest; they describe different paths:

| term (per layer) | delivered chunked path | adaptive plan |
|---|---|---|
| projections | 4 x 86 blocks x 5.62 ms = 1,933 ms | in the plan |
| FFN | 86 blocks x 40.3 ms = 3,466 ms | in the plan |
| barriers | 85 x 1.7 ms = 145 ms | -- |
| host glue | 86 blocks x 70 ms = **6,020 ms** | -- |
| host attention | 1,718 ms | 1,718 ms |
| **per layer** | **13,282 ms -> 398 s/step** | **2,312 ms -> 69.4 s/step** |

The delivered path spends 84 % of its time in 96-row staging and host glue
(430 dispatches per layer); the adaptive plan does the same linear work in
**4 dispatches of 2,048 rows**.  Replacing that half is the wiring
`ADAPTIVE-NPU-REPORT.md` section 9 still lists as open -- it is not a planning
problem and this report does not claim it is done.

---

## 4. DEVICE TESTS: four wide sizes, real layer-0 weights

`_scratch/util/adaptive/npu_dev.py` (py310), one launcher at a time.  Weights are the
model's own BFP16 blocks (`weights/block0.npz`: `layers.0.feed_forward.w1`,
`.w2`, `layers.0.attention.to_q`), reshaped only, never re-encoded.  A is a
seeded random activation, bf16-rounded.  The reference is the per-call-bf16 model
(the kernel's C operand is bf16, so the running sum is rounded K/kk times) with A
quantised to bfp16 (conv_even).

The negative control permutes A's **k-steps inside the packed payload** -- a real
k-order error, not a packing mistake -- and must collapse.

### 4.1 Results (12/12 PASS)

| size | M | plan | GEMM | n | corr | maxabs | rel_max | exact_frac | ms/dispatch | TFLOP/s (true) | neg control |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **1920x1080** | 8192 | xd m=64 kk=32 Mblk=2048 **x4** | gu | 320 | **0.99999835** | 0.500 | 1.57 % | 0.905 | 24.27 | 6.64 | collapsed |
| | | | dn | 320 | 0.99999702 | 0.875 | 2.01 % | 0.868 | 22.66 | 7.11 | collapsed |
| | | | proj | 320 | 0.99999836 | 0.500 | 1.64 % | 0.905 | 10.71 | 5.64 | collapsed |
| 1280x720 | 3648 | t33 m=64 kk=32 Mblk=3648 x1 T=2 | gu | 160 | 0.99999835 | 0.500 | 1.69 % | 0.905 | 57.33 | 5.00 | collapsed |
| | | | dn | 160 | 0.99999703 | 0.875 | 2.03 % | 0.868 | 55.83 | 5.14 | collapsed |
| | | | proj | 160 | 0.99999835 | 0.375 | 1.23 % | 0.905 | 23.09 | 4.66 | collapsed |
| 1600x900 | 5760 | xd m=64 kk=32 Mblk=1920 **x3** | gu | 320 | 0.99999835 | 0.500 | 1.69 % | 0.904 | 23.76 | 6.36 | collapsed |
| | | | dn | 320 | 0.99999703 | 0.875 | 2.01 % | 0.869 | 23.25 | 6.49 | collapsed |
| | | | proj | 320 | 0.99999835 | 0.375 | 1.23 % | 0.904 | 11.22 | 5.05 | collapsed |
| 2048x1080 | 8736 | xd m=48 kk=32 Mblk=1488 **x6** | gu | 320 | 0.99999836 | 0.500 | 1.57 % | 0.905 | 21.40 | 5.47 | collapsed |
| | | | dn | 384 | 0.99999703 | 0.875 | 2.01 % | 0.868 | 21.32 | 5.49 | collapsed |
| | | | proj | 384 | 0.99999836 | 0.500 | 1.64 % | 0.905 | 10.49 | 4.18 | collapsed |

* **corr > 0.999 on all 12 rows**; the worst is 0.99999702 (dn), which is the
  normal bf16 accumulation-order difference, not a layout error -- maxabs is
  0.875 against a reference whose own max is 43.5 (2 %), and 0.500 against 31.9
  for gu.
* **every negative control collapsed** (corr ~ 0 against the correct reference).
* `exact_frac ~ 0.90` (0.868 for dn) is the 1-2 ulp bf16 accumulation spread, the
  same as every previously gated size.
* 1920x1080 and 2048x1080 are the first **multi-dispatch** plans gated beyond
  ndisp=2: 4 and 6 dispatches, each covering Mblk rows, with the tail zero-padded
  rows checked too (their C must be zero).

### 4.2 The cost model vs the device

| size | GEMM | model dispatch_ms | measured ms | model / measured |
|---|---|---|---|---|
| 1920x1080 | gu | 32.46 | 24.27 | 1.34 |
| | dn | 32.46 | 22.66 | 1.43 |
| | proj | 12.80 | 10.71 | 1.19 |
| 1280x720 | gu | 63.91 | 57.33 | 1.11 |
| | dn | 63.91 | 55.83 | 1.14 |
| | proj | 24.59 | 23.09 | 1.07 |
| 1600x900 | gu | 32.46 | 23.76 | 1.37 |
| | dn | 32.46 | 23.25 | 1.40 |
| | proj | 12.80 | 11.22 | 1.14 |
| 2048x1080 | gu | 24.59 | 21.40 | 1.15 |
| | dn | 24.59 | 21.32 | 1.15 |
| | proj | 9.85 | 10.49 | **0.94** |

The model is an upper bound on 11 of the 12 dispatches, as designed (it charges
the saturated per-core rate).  The one miss is 2048x1080's `proj`: 9.85 predicted
vs 10.49 measured (-6 %), because that dispatch is the shortest and the model's
`DISP_MS = 1.0` fixed charge is a midpoint between the measured 2.5 ms one-shot
staging and 0 in a burst.  It does not move any size's verdict.

Per step (device half only), measured vs planned:
1920x1080 **13.69 s** vs 17.83, 1280x720 **7.89** vs 8.70, 1600x900 **10.41** vs
13.37, 2048x1080 **19.10** vs 20.37.

### 4.3 New measurement: m=64 xd is the fastest geometry on this array

| plan form | measured TFLOP/s (gu / dn / proj) |
|---|---|
| xd m=64 kk=32 n=320 | **6.64 / 7.11 / 5.64** (1920x1080), **6.36 / 6.49 / 5.05** (1600x900) |
| t33 m=64 kk=32 n=160 | 5.00 / 5.14 / 4.66 (1280x720) |
| xd m=48 kk=32 n=320/384 | 5.47 / 5.49 / 4.18 (2048x1080) |

The cost model's kk=32 rate is 5.12 TFLOP/s; the xd m=64 form beats it by
24-39 %, which is exactly why its prediction is 1.3-1.4x the measurement.  The
planner is conservative in the safe direction, and m=64 is a better geometry than
the 1024² campaign used (t33 m=48, 5.30 TFLOP/s).

### 4.4 A reporting bug in the device harness (not in the numbers)

`npu_dev.run_gemm()` computes `TFLOP_s = 2*M*K*N / wall` where `M` is the FULL
sequence but `wall` is ONE dispatch's median.  For any multi-dispatch plan the
printed rate is inflated by `M / Mblk` (exactly `ndisp` when `M` is a multiple of
`Mblk`, and `ndisp` rounded up otherwise):

| size | ndisp | harness TFLOP/s | true per-dispatch |
|---|---|---|---|
| 1920x1080 gu | 4 | 26.545 | 6.636 (= 26.545 / 4) |
| 1600x900 gu | 3 | 19.065 | 6.355 (= 19.065 / 3) |
| 2048x1080 gu | 6 | 32.105 | 5.469 (= 32.105 / 5.871; M/Mblk = 8736/1488) |
| 1280x720 gu | 1 | 5.004 | 5.004 |

The same inflation is in `ADAPTIVE-NPU-REPORT.md` section 8's 512x768 row (6.53 TFLOP/s at
ndisp=2, M/Mblk = 1568/800, is 3.33 per dispatch).  It does not affect `corr`, `maxabs` or the
pass/fail gate -- only the speed column.  This report quotes the true rate.

---

## 5. DOES 1920x1080 HIT A LIMIT? NO -- and here are the margins

### 5.1 Every gate, with its margin

| gate | 1920x1080 | limit | used |
|---|---|---|---|
| side envelope | 1080 / 1920 | 64..2048 | 52.7 % / 93.8 % |
| token envelope | M = 8192 | 16,416 | 49.9 % |
| per-core L1 | 58,624 B | 65,536 B | 89.5 % (6,912 B spare) |
| c_leaf vs one BD | 40,960 B | 65,532 B | 62.5 % |
| host BO `x` | 62,914,560 B (60 MiB) | 2 GiB | 2.9 % |
| host BO `ff` | 167,772,160 B (160 MiB) | 2 GiB | 7.8 % |
| host attention RAM | 906 MB (864 MiB) | 8 GiB | 10.5 % |
| 8-step ceiling (delivered model) | 3,187 s (0.89 h) | 21,600 s (6 h) | 14.8 % |
| ERT watchdog per dispatch | 24.3 ms | 7,000 ms | 0.35 % |
| device dispatches per layer | 4 | -- | -- |

**1920x1080 exceeds no limit.**  It is not near the L1 wall in any dangerous way
(10.5 % spare, and aiecc accepted the image), it is at half the token budget, and
its host BOs are 3-8 % of the ceiling.

### 5.2 What actually binds at 1080p: attention (a speed limit, not a wall)

The binding constraint is the **host attention**: 51.54 of the 69.37 s/step (74 %),
2.9x the NPU's per-row cost, 60x the 512² attention term.  That is a *throughput*
limit, and the fix is the T22 campaign-1 key-blocked/online-softmax attention
image -- the same open item `ADAPTIVE-NPU-REPORT.md` section 9 already lists.  On the
delivered chunked path the wall is even further from the NPU: 84 % of its step is
96-row staging and host glue (section 3.4).

### 5.3 What DOES refuse, and exactly what it would take

| request | refused by | what it would take |
|---|---|---|
| **3840x2160 (4K)** | (a) `MIN_HW..MAX_HW`: height 3840 > 2048; (b) `MAX_M`: M = 32,448 > 16,416 | widen the side envelope to 3840 **and** the token budget to 32,448 (2x).  Nothing else: with `MAX_M` lifted, `plan(32448)` resolves to **the same plan key as 1920x1080** (`xd_m64_kk32_Mblk2048_n320-320-320_653b0019`, L1 58,624 B) with **ndisp = 16 instead of 4** -- the 1080p images already built would serve it.  Host attention at N=32,448 would be 26.9 s/layer = 808 s/step, and its f32 working set 2.33 GiB (still under the 8 GiB budget) |
| **a 16:9 wider than 2048** | the side envelope only | the token budget still has 7,168 tokens spare at 2048x1152 (M=9,248).  So at 16:9 it is the **side** limit that binds, not the token budget -- the opposite of the square case, where 2048x2048 hits both at once |
| 2048x1152 (the largest exact 16:9 the envelope admits) | **nothing -- and it is BUILT** | it plans to `xd m=64 kk=32 Mblk=1856 x5` (M=9,248, L1 58,624 B) and `npu_wide.py --hw 2048 1152 --build` produced it: 3 images, `xd_m64_kk32_Mblk1856_n320-320-320_d55f76dc`, dev 22.28 s/step + host attention 65.68 = **87.96 s/step** (74.7 % attention).  It is the widest exact 16:9 the product admits, and any other pair in 64..2048 is one `--hw` invocation away |

### 5.4 The honest summary for the user

* 1920x1080, 1280x720, 1600x900, 2048x1080, 1080x1920, 1024x576, 1366x768,
  1080x1080, 1440x1080, 512x288, 288x512: **planned, BUILT, and (four of them)
  run on the device**.
* 2048x1152 (the largest exact 16:9 in the envelope): **planned and BUILT** as a
  bonus, through the same `--hw` path any other pair in 64..2048 uses.
* 4K: refused by two product envelope constants, with the exact arithmetic above.
* The step time at 1080p is **~69 s/step on the adaptive model** (17.8 device +
  51.5 host attention) and **~398 s/step on the delivered chunked model**
  (mostly 96-row staging).  Both are estimates; the device half is now measured
  at 13.7 s/step for the same work.

---

## 6. WHAT THIS DOES NOT PROVE

| claim | status |
|---|---|
| the three GEMM images compute the right numbers at M=8192 / 8736 / 5760 / 3648 on real layer-0 weights | **PROVEN** (12/12, corr > 0.999997, negative controls collapsed) |
| a legal 32-core placement exists for each of the 11 sizes, and aiecc builds it | **PROVEN** (10 plan keys, 30 images, sha256 in the manifests) |
| the product's pad + crop contract covers non-16-multiples | **PROVEN** (existing behaviour + a real 480x854 job record) |
| a 30-layer step at 1920x1080 runs end to end | **NOT PROVEN** -- `s2_ffn32.ffn32_run` is still `NotImplementedError` (INTEGRATION-SPEC 8); the images are built and gated, the per-layer loop is not chained |
| the 51.5 s/step host attention at M=8192 is real | **ESTIMATE** -- the 600 GFLOP/s anchor was measured at 480p/1080p, not at M=8192.  The device half (13.7 s/step) is measured |
| the product's own step estimate (398 s/step) is measured | **ESTIMATE** -- it is the delivered 96-row model; the adaptive half of it is what this report measured |
| the crop path was exercised at 1920x1080 | **NOT RUN HERE** -- the crop is existing product behaviour, verified against the 480x854 job record and the code, not re-executed at 1080p |

Two operational notes for anyone re-running this:

* **the device lock is a wait, never a kill.**  The 2048x1080 run printed
  `[dev] waiting for 4 other python.exe [26684, 56400, 107020, 112608]` and
  started 21 s later when they exited.  Any python.exe means the device is busy;
  `device.lock` itself is unlinked at startup and is not a mutex.
* **do not compare these numbers with a WebUI run without checking the cache
  mode.**  `webui/config.py:433` (`MEMORY_FAST_THRESHOLD_GB = 9.0`) silently
  selects the `--no-cache-all` path when commit headroom is under 9 GB: that is
  2x slower AND a different latent baseline, so a WebUI timing taken in `low`
  mode is not comparable to the device numbers here.  The job record carries
  `memory_mode` and the auto-reason; check it before quoting a time.

---

## 7. REPRODUCE

```powershell
# 1. the padding law against the product's own arithmetic (2000 random pairs + corners)
py312 _scratch/util/place/_wide_law_check.py

# 2. the wide grid: plan only (no build), then plan + build every size
py312 _scratch/util/place/npu_wide.py --plan  --json out/wide_plan.json
py312 _scratch/util/place/npu_wide.py --build --json out/wide_build.json --md out/wide_grid.md

# 3. the product's own verdict for each size (py310 -- chunk_gate needs torch)
$env:ZIMAGE_ADAPT='1'
py310 _scratch/util/place/_wide_product_check.py

# 4. the device gate, ONE launcher at a time, py310
py310 _scratch/util/adaptive/npu_dev.py --hw 1920 1080 --gemm gu dn proj --json out/wide_dev_1920x1080.json
py310 _scratch/util/adaptive/npu_dev.py --hw 1280  720 --gemm gu dn proj --json out/wide_dev_1280x720.json
py310 _scratch/util/adaptive/npu_dev.py --hw 1600  900 --gemm gu dn proj --json out/wide_dev_1600x900.json
py310 _scratch/util/adaptive/npu_dev.py --hw 2048 1080 --gemm gu dn proj --json out/wide_dev_2048x1080.json

# 5. the cost model vs the measurements
py312 _scratch/util/place/_wide_model_check.py
```

The product's verdict for all eleven runnable sizes is `exec_mode = chunked` with
`plan_built = true`; `ready` stays False because `ready` means "a gated native
image set exists for this exact shape", and these are adaptive plans, not
delivered shapes.  That is the same meaning it had before this work.

---

## 8. FILES TOUCHED

**New**

| file | what |
|---|---|
| `_scratch/util/place/npu_wide.py` | the padding law, the wide grid, the per-step cost model (NPU + host attention), `--plan` / `--build` / `--json` / `--md` |
| `_scratch/util/place/_wide_law_check.py` | the law vs `webui.shapes` on 2,021 cases |
| `_scratch/util/place/_wide_product_check.py` | what `shapes.resolve()` says for each wide size (py310) |
| `_scratch/util/place/_wide_model_check.py` | model dispatch_ms vs measured, and the true per-dispatch TFLOP/s |
| `_scratch/util/place/_wide_build_summary.py` | the build record, one line per plan |
| `_scratch/util/place/_wide_images.py` | the image set (tag, bytes, sha256) per plan |
| `_scratch/util/place/_wide_4k_probe.py` | what 4K would need (the envelope widened in-process) |
| `WIDE-GRID-REPORT.md` | this file |

**Artefacts**

* `webui/shape_manifests/plan_<key>.json` -- **8 new plan manifests**
  (`xd_m64_Mblk2048`, `xd_m64_Mblk1920`, `xd_m64_Mblk1856`, `xd_m48_Mblk1488`,
  `t33_m64_Mblk3648`, `t33_m48_Mblk4176`, `t33_m32_Mblk4672`,
  `t33_m16k16_Mblk6176`), all `built_ok: true`
* `_scratch/util/place/out/ad_*` -- **24 new images** (8 plans x 3 GEMMs);
  the other 2 of the 10 keys were already built and are reported as CACHED
* `_scratch/util/place/out/wide_{plan,build}.json`, `wide_grid.md`,
  `_scratch/util/adaptive/out/wide_dev_*.json` -- the records behind every
  number in this report
* `.prj` directories were deleted after every build; the single file kept in
  each is `kernels_main.json`

**Not touched**: the delivered default path.  No shipped module was modified; the
new module is additive and every plan is keyed by the plan, so an existing shape
resolves exactly as it did before.
