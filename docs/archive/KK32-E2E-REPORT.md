# kk=32 GEMM WIRED INTO THE PRODUCT FFN GATE/UP: MEASURED, AND WHY IT LOSES

2026-11-02.  Everything below is a measurement or an arithmetic consequence of
one, except where explicitly marked "predicted".

**Headline.**  The k-blocked GEMM is real and it is fast -- 13.19 ms / 6.11
TFLOP/s with the product's own layer-0 w1 blocks, corr 0.999915 -- but it
**cannot shorten the 512^2 image** through the FFN gate/up path.  Wired into the
product and dispatched with real weights it costs **+42 to +61 ms per
layer-step** plus a **~0.89 s/layer** B repack, against the shipped silu-fused
swiglu image.  The cause is structural, not the kernel: SEQ = 1056 rows is
**33 M-tiles of 32** and the image has **32 (column, core) slots**, so one
layer-step needs ceil(33/32) = 2 row chunks x 2 halves = **4 dispatches**, and
the fused silu + the device handoff both have to move to the host.

Deliverables:

| file | what |
|---|---|
| `_scratch/util/place/kk_prod.py` | the product module: registration, the u16 A packer, the fused silu(gate)*up -> down-image A pass, gu_kk32() |
| `_scratch/devkill/devkill_ffn.py` | `T22_FFN_KK32=1` (default 0) branch in block() -- the product's gu dispatch site |
| `_scratch/util/place/kk_prod_check.py` | DEVICE proof: real layer-N w1 blocks, per-M-tile corr, negative controls, dispatch timing |
| `_scratch/util/place/kk_prod_selftest.py` | HOST proof: pack_a_u16 == pack_a_kk == pack_a_kk_ref byte-for-byte; the fused pass == the direct [rt][obj][12][4][8][8] construction, 0 of 10,813,440 words differ |
| `_scratch/util/place/kk_prod_devtest.py` | DEVICE proof of gu_kk32 in isolation: dispatch count, per-dispatch ms, corr vs a numpy silu(g)*u |
| `_scratch/util/kk32/run_kk_ab.ps1` | the A/B launcher (one arm per invocation; `T22_FFN_KK32` is the only variable) |
| `out/kk_prod_check.json` | the device numbers, machine readable |

---

## 1. THE ARITHMETIC: 33 TILES DO NOT FIT 32 SLOTS

    SEQ           = 32 caption + 32x32 image tokens = 1056 rows
    image m       = 32 rows per M-tile
    M-tiles       = 1056 / 32 = 33
    image slots   = cols * cpc = 8 * 4 = 32        (tile_map: 4 per column)
    row chunks    = ceil(33 / 32) = 2              (1024 rows + 32 rows)
    halves        = 2                              (gate = w1, up = w3)
    DISPATCHES    = 2 x 2 = 4 per layer-step                  <- the whole story

Measured dispatch costs (device, this host, same image `out/kk32_m32_n512`):

| measurement | ms | note |
|---|---|---|
| 1 dispatch, A/B/C already resident, median of 7 | **13.187** (min 13.094 max 13.236) | 6.107 TFLOP/s -- reproduces the 13.21 / 6.095 record |
| 1 dispatch, immediately after its B was rewritten (in-product case) | **16.3 - 16.4** | `kk_prod_devtest`, per-dispatch [16.41, 16.35, 16.26, 35.95] |
| 4 dispatches, sum | **85.0** | 65.5 at the median of the four |
| 4 dispatches, the 4th | 35.95 | the one that follows the host silu pass; unexplained, and it is the pessimistic reading |

NOTE: the 16.4 ms in-product figure, not 13.19, is the one that applies: the
product rewrites the 44.2 MB B payload for every half of every layer-step, so
the device's B read is always cold.  A/B/C are the same byte counts the shipped
image moves (A 7,864,320 / B 44,236,800 / C 20,971,520).

So the device side alone is 4 x 16.4 = 65.5 ms and the pessimistic sum is
85.0 ms, against the shipped **48.0 ms** for one silu-fused dispatch.

---

## 2. WHAT THE SHIPPED IMAGE'S FUSION IS WORTH (measured, per layer-step)

`swiglu11` (M=1056, K=3840, N=20480, silu(gate)*up fused in the epilogue) does
in ONE dispatch what the kk=32 image needs four for, and it hands its
1056 x 10240 drain **straight into the down image's A BO** (t22_bigpipe3 device
handoff) so the intermediate never crosses the bus.

Per-layer-step ledger, both arms measured on the same harness
(`T22_FASTGLUE=1 T22_REF_SMNB=1 ZIMAGE_CHUNK_THREADS=16`), 30 layer-steps each:

| component | A: shipped swiglu11 | B: `T22_FFN_KK32=1` | delta |
|---|---|---|---|
| A pack | 6.4 (a_pack_fast) | 7.5 (pack_a_u16, 2 chunks) | **+1.1** |
| gate/up W write | 4.1 (88.5 MB, once) | 14.8 (88.5 MB B + 15.7 MB A) | **+10.7** |
| gate/up dispatch | **48.0** (1 dispatch) | **65.5 - 85.0** (4 dispatches) | **+17.5 .. +37.0** |
| read + silu + dn-A pack | **0.0** (device handoff) | 12.7 (fused host pass) | **+12.7** |
| down dispatch | 31.4 | 31.1 | 0 |
| **per layer-step** | **95.0 - 99.8** | 1131 - 4530 as run (repack on the path); **~126** with the repack pre-paid | **+42 .. +61 steady** |

What the fusion + handoff buys, itemised:

* **silu on device: free.** Doing it on the host costs **12.7 ms** of a
  10,813,440-element pass (`_swiglu_to_dn_a`, numba, 16 threads) that also has
  to re-tile the result into the down image's A layout.
* **the handoff: free.** Reading the two 21.0 MB C BOs is what forces the host
  pass at all; the shipped route reads nothing (measured read=4.9 dec=2.3 on
  the *down* drain only, identical in both arms).
* **one dispatch instead of four** saves 3 x 16.4 = 49.2 ms of B re-read for a
  payload that did not change between the gate and the up half -- kk changes no
  byte of B (KK-GEMM-REPORT 3: "kk is a PERMUTATION of B"), so the second half
  re-reads 44.2 MB to learn the same thing.

### The B repack is a second, larger bill

The kk image eats B in **model-block order** (N, K/8, 9).  The product's
resident cache is the **delivered container** order, so it must be converted --
and it may not be re-encoded (0.3 % of the checkpoint's blocks are
non-canonical).  Measured in-product on the 1-step run:

    [devkill] rows=1056 ... gu=4478.1 ms    <- layer 0: JIT + repack
    [devkill] rows=1056 ... gu=1494.9 ms
    [devkill] rows=1056 ... gu=1107.1 ms    <- median of the next 29 layers: 1016 ms
    ...
    warm gu_kk32 (repack cached, devtest):  125.7 ms

=> **~890 ms per layer** of repack (load 2 x 44.2 MB of stored blocks + two
pack_b_kk permutations) that the shipped path does not pay, because the shipped
image's order **is** the resident order.  30 layers = **+26.7 s once**, and
2.65 GB of extra resident memory if it is pre-packed instead (the shipped
startup repack is 12.58 s for its own order).

---

## 3. PUTTING IT ON THE CALL PATH (proof, not "installed")

The last attempt in this project (T22_P33) printed "installed" and was a no-op.
This one is proven by three independent signals from a real 1-step product run
(`webui/jobs/20261102-114521-6592d9`, 30 layer-steps, real layer weights from
the resident cache):

1. **the flag reached the device worker** and the image registered:

       [devkill] T22_FFN_KK32=1: FFN gate/up -> the kk=32 image kk32_m32_n512 (4 dispatches per layer-step)
       [kk32] kk32_m32_n512 registered (68,800 insts) in 0.02 s -- A 7,864,320 / B 44,236,800 / C 20,971,520 B

2. **the shipped dispatch is gone, not merely shadowed.**  All 30 lines read
   `pack=0.0 wgu=0.0`: a_pack_fast and the 88.5 MB bw.write(w_gu) are only
   skipped in the kk32 branch (arm A always shows pack=6.4 wgu=4.1).  The
   branch does not call dev.imgs["swiglu11"] at all.
3. **the dispatch count.**  gu_kk32 was measured in isolation to issue exactly
   **4** dispatches (`kk_prod_devtest`: "gu_kk32 (measured): 4 dispatches ...
   per-dispatch [16.41, 16.35, 16.26, 35.95]"), and the 1-step run has exactly
   30 layer-steps -> **120 kk=32 dispatches in one step, 960 in an 8-step run**.
   `kk_prod.report()` (atexit) did not get to print because dev_worker kills the
   child with TerminateProcess, which skips atexit; the count above is measured
   per call, not inferred.

**The latents changed.**  Step 0 is deterministic per arm (same seed, same
resume checkpoint, same schedule; verified: three separate A-arm variants each
reproduce their own value to the last digit), and the kk=32 arm produces a
fourth, different value:

| arm | step-0 pred_absmax | runs |
|---|---|---|
| A: `T22_FASTGLUE=1 T22_REF_SMNB=1` (asm_1) | 6.595890522003174 | 1 |
| A + T22_P33 attention (p33ac_a1/a2/a3) | 6.591831207275391 | 3, identical |
| A + the P33 b variant (p33ac_b1/b2/b3) | 6.599099159240723 | 3, identical |
| **B: `T22_FFN_KK32=1`** | **6.576844692230225** | 1 |

and, at the FFN output level on the device, the kk=32 gate/up differs from the
shipped path by **corr 0.999830 / maxabs 11.05** (`kk_prod_devtest`, real layer-0
w1+w3), so the stream cannot be bit-identical.  *(A 1-step A-arm run for a direct
latents-file comparison was not spent: the device was handed to the attention
agent.  The step-0 column above is reproducible and already separates the arms.)*

### Correct by construction, and checked

| check | result |
|---|---|
| pack_a_u16(bf16 u16) vs pack_a_kk(widen) vs pack_a_kk_ref | **byte-identical**, 1024-row and 32-row chunks (`kk_prod_selftest`) |
| fused silu(g)*u -> dn-A pass vs the direct [rt][obj][12][4][8][8] construction | **0 of 10,813,440 words differ**; decodes back exactly |
| pack_b_kk is a permutation of the stored blocks | asserted on every layer (multiset of 9-byte units, exact) |
| C per M-tile, real layer-0 w1, device | corr **0.999915** (bfp-A) / 0.999893 (bf16-A); worst of 32 tiles **0.999913**; maxabs 0.5625 |
| negative control, A M-tiles 0/1 swapped | **-0.0077** (collapses) |
| gu_kk32 end-to-end, 1056 rows, device | corr **0.999830**; every 96-row tile 0.999829-0.999834; the 33rd tile (rows 1024:1056, the 2nd chunk) **0.999830** |
| the down image's dispatch in arm B | 30.2-34.8 ms, i.e. unchanged -> the host-produced A layout is accepted |

---

## 4. WHAT WOULD MAKE kk=32 PAY, WITH NUMBERS

The prize is real: at the kk=32 inner loop the gate/up GEMM is 1.49x the kk=8
image (13.19 vs 19.70 ms), and 1056x3840x20480 at 6.11 TFLOP/s would be
**27.2 ms** instead of 48.0.  Two separate obstacles stand in the way.

### 4a. The 33-tile image: it FITS, but the measured t33 rate does not pay

`ffn_design_t33` already builds M=1056 (33 tiles over 32 slots, per-column
[5,4,4,4,4,4,4,4]) at kk=32; `out/kk32_t33_m32_n320.xclbin` is on disk.  L1 for
the gate/up shape (m=32, kk=32, n=320):

    C   2 leaves x m*n*2   = 2 x 32*320*2 = 40,960   (column 0's core runs 5 tiles)
    B   kk*n/8*9           = 32*320/8*9   = 11,520
    A   m*kk*2             = 32*32*2      =  2,048
    stack                                 =  2,048
                                          = 56,576  <= 65,536   FITS

(exactly the built image's own accounting, 56,576, per KK-GEMM-REPORT 1).  At
N=10240 the payloads stay the SAME bytes as today -- A 7,864,320, B 44,236,800,
C 21,626,880 (33 tiles x 32 ntiles x 32*320*2) -- so it is a drop-in for the
same BOs, and it needs ntiles=32.

And n cannot be pushed to 512: with 2 C leaves,
`128n + 36n + 2,048 + 2,048 <= 65,536` gives **n <= 374**.  So the 33-tile
form is stuck at n<=320, i.e. a different (smaller-n) inner loop than the 6.11
TFLOP/s shape.

**What it has to beat:** swiglu11's 48.0 ms **minus** the 12.7 ms host silu and
the dn-A write that the fusion gives back for free -- i.e. both halves must run
in **< 35 ms**, i.e. **<= 17.6 ms per half at M=1056**, i.e. **>= 4.7 TFLOP/s**.

**What the only measured 33-tile kk=32 image actually does:** 10.31 ms / **3.02
TFLOP/s** at N=3840 (ntiles=12).  Scaled to N=10240 (32 ntiles) that is
10.31 x 32/12 = 27.5 ms per half -> **55 ms for both halves**, plus the host
silu: a loss of ~20 ms against 48.0.  So the 33-tile image is a *necessary* fix
for the dispatch-count problem but, at the rate that shape measures, not a
sufficient one.

=> **The minimal change that would pay is an n=512-class 33-tile inner loop**,
and L1 refuses it (2 C leaves at n=512 need 73,728 B; that is exactly the
kk64_b1_m32_n512-style refusal in KK-GEMM-REPORT 1).  The realistic options, in
order of expected value:

1. **Make the FFN see a 1024-row slab.**  The 33rd tile is the 32 caption
   tokens.  If the FFN ran on a **1024-row token slab** and the 32 caption rows
   used the existing sw_rt1 image, the kk=32 path would be 1 dispatch per half:
   2 x 13.2 = 26.4 ms.  Cost: one extra 88.5 MB container B write for sw_rt1
   (+4.1 ms) unless the caption slab is hoisted out of the layer loop.  This is
   the only variant that reaches 26.4 ms.
2. **A 33-tile kk=32 image at n=320, built and measured.**  Fits, drops to 2
   dispatches, needs >= 4.7 TFLOP/s to win; the one t33 measurement (3.02) says
   it will not get there, but that image is N=3840 and a gate/up build would be
   the first fair test.  Highest risk on this list.
3. **Nothing else.**  kk=64 is L1-refused at n>=448, and kk=16 measures 52 slots
   (1.62x is the kk=32 number).

### 4b. kk=32 without the restructuring

Separating the two axes the kk campaign changed:

* **kk is the whole compute win and it costs nothing in bytes.** kk is a pure
  *permutation* of B and a re-blocking of A: A stays 7,864,320 B, B stays
  44,236,800 B, C stays 20,971,520 B (KK-GEMM-REPORT 3).  The 1.49x is purely
  the inner loop: 26 bundles per 16x16 tile -> 64, 78.8 -> 128.0 MAC/cyc.
* **the bfeed="shared" restructuring buys ZERO slots.** The kk=8 control on the
  same new entry (kk8_b1_m32_n512) measures **26** slots -- the shipped loop.
  Its only effect is 8x fewer B *bytes per dispatch* (353.9 -> 44.2 MB), which
  is a DMA/BO-footprint win.  This shape is compute-bound (48.0 ms moving 118 MB
  = 2.5 GB/s, vs the ~30 GB/s these images sustain), so shared-B would buy
  nothing here.

=> **Keep kk=32; drop the multi-dispatch restructuring.**  Where the row count
is a multiple of 1024 there is no restructuring to do at all: the kk32 image is
a **direct drop-in for `ffn_gu_m32`** in the S2 FFN path -- identical BO sizes
(A 7,864,320 / B 44,236,800 / C 20,971,520), identical byte counts, one dispatch
per half.  There the A/B is clean and the win is the full
**2 x (19.675 - 13.19) = -12.97 ms per layer-step** (39.35 -> 26.38 ms),
i.e. **-3.1 s on an 85 s loop**, with no fusion and no handoff to give up
because that path already does the silu on the host.

---

## 5. WHY THE WHOLE THING LOSES -- ONE PARAGRAPH

The shipped gate/up is not one GEMM, it is one dispatch that carries a fused
silu epilogue and a device-resident handoff into the down image, at 48.0 ms for
1.66e11 FLOP (3.46 TFLOP/s).  The kk=32 kernel is 1.77x that rate, which would
make the same work 27.2 ms -- but 1056 rows is 33 tiles against 32 slots, so the
work is issued as four dispatches that each re-read a 44.2 MB B payload they
already read (65.5-85.0 ms), the silu comes back to the host (12.7 ms), the
intermediate crosses the bus again (15.7 MB written, 42 MB read), and the B must
be re-ordered per layer because the resident cache is in the *other* order
(~0.89 s/layer, one-off).  Net: +42 to +61 ms per layer-step steady, +26.7 s of
one-off repack, and the 33-tile image that would remove the four-dispatch tax
cannot carry the n=512 loop that produced the 1.49x in the first place.

**Recommended next step, if kk=32 is to be used at all: put it where it is a
drop-in** -- the S2 ffn_gu_m32 gate/up (2 dispatches, -12.97 ms/layer-step),
not the silu-fused big-M image, and not through a 32-slot image on a 33-tile
problem.

---

## 6. DEVICE HYGIENE NOTE

My isolated device work windows were ~11:20-11:22 (`kk_prod_check.py`) and
~11:39-11:44 (`kk_prod_devtest.py`, two attempts).  The four 8-step runs
`20261102-113131`, `-113449`, `-113827`, `-114125` (240/240 layer-steps
each) overlap the second window for two of them: **113827 (11:38:27) and 114125
(11:41:25) may be contaminated** by ~0.5 s of foreign dispatches and an xclbin
registration.  113131 and 113449 are outside it.  `20261102-114704` (11:47:04)
stopped after 84/240 layer-steps -- not mine (my only product run was
114521, 11:45:23-11:46:54, tag kk32smoke1).
