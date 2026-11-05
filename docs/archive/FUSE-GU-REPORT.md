# silu(gate)*up is fused into the plan's gu image

**2026-11-03 · T22 adaptive, part 3 · new flag `T22_PLAN_FUSE_GU` (default 0) ·
the delivered default path is untouched (proven, section 7)**

~~~
THE ONE SENTENCE: the plan-generated gu image now computes gate, up AND
silu(gate)*up in ONE dispatch -- the delivered swiglu11 kernel body, at the
plan's own kk=32 / m / tile map, with a B object that is the [gate|up] pair --
so the 50-140 ms host elementwise pass the plan path was paying disappears, and
the fused image's output is bit-identical to (pure-GEMM gu) then host
silu(gate)*up on the model's own blocks.
~~~

| at | fused gu (one dispatch) | pure gu (two dispatches) | dn | per layer, fused | per layer, delivered |
|---|---|---|---|---|---|
| 512² (M=1056) | **36.5 ms**, 4.55 TFLOP/s | 19.7 ms x2 | 19.8 ms | **90.2 ms** | 102.4 ms (measured) |
| 1920x1080 (M=8192) | **44.6 ms**, 7.22 TFLOP/s | 21.6 ms x2 | 19.9 ms | **513.4 ms** | 819.2 ms (8 chunks) |

corr vs a numpy reference on the model's blocks: **0.9998** fused, 0.9999 pure;
the swapped-pair negative control collapses to **0.504**.  The 8-step FFN delta
is **-2.9 s at 512²** and **-73.4 s at 1920x1080** -- see section 6 for why the
512² number is small and the task's -14 s is not reachable there.

---

## 1. The fusion mechanism, and where it already exists

The delivered design does this today.  `_scratch/devkill/swiglu_rt11.py` is the
shipped `swiglu11` image (M=1056, K=3840, N=20480, G=10 cores, 48.0 ms measured,
the one `devkill_ffn.block` and `kk_prod.gu_kk32` dispatch as the FFN gate/up).
Its three data-flow places are the whole of the fusion:

1. **B**: ONE object per (n-tile, k-step) is the pair `[gate slice | up slice]`,
   `2 x 9*kk*n_slice/8` bytes, in ONE ObjectFifo.  The two matmuls of a k-step
   read the SAME A object, so gate and up can never be handed each other's
   weights by a scheduling accident -- the pairing is a property of the image,
   not of two dispatches agreeing.
2. **two mm_bfp calls per k-step**: `mm_bfp(dim_m=m, dim_k=kk, dim_n=n_slice,
   mixed=True)` at B offset 0 -> the gate tile, at B offset `b_obj` -> the up
   tile (`memref.view(bm, eb, constant(B_OBJ))`, the delivered line).
3. **the epilogue is on the core**: `silu_sized(m*n)` gate -> dst, then
   `mul_sized(m*n)` dst * up -> dst.  The DRAINED object is the fused tile
   (m x n_slice), which is exactly what a GEMM of (m, n_slice) would have
   produced -- same 8x8 block order, same `m*n*2` bytes per leaf.  That is why
   the host drain (`s2_ffn32.unpack_c_plan`) needs no change at all.

**It does not assume kk=8.**  The delivered image's own assert is
`k in (8, 32, 64)` and the shipped arm runs **k=64**.  `dim_k` is `kk`; the
fusion only changes the B object's shape and the epilogue.  The new image
`_scratch/util/place/ffn_design_xf.py` (design `ffn_xf`) is
`ffn_design_xb`'s 32-core placement (one M-tile per core, ONE shared B fifo so
the pair crosses the shim once per n-tile) with exactly those three places
replaced, and it runs at **kk=32**.

### Why the fused image needs its own n

The epilogue keeps **three** C tiles alive at once (gate, up, and the silu
destination), so the C term triples:

~~~
pure   L1 = C(m,n) + B(kk,n) + A(m,kk) + stack
fused  L1 = 3*C(m,n) + 2*B(kk,n) + A(m,kk) + stack
~~~

At m=48 / kk=32 the plan's own gu n=320 needs 122,880 B (2x the 64 KiB budget),
so a fused image at the plan's n cannot exist.  `webui/npu_plan.pick_n_fused`
therefore picks the largest n that fits, and ntiles = N/n grows:

| plan | m | kk | pure gu n | fused gu n | fused L1 | fill | ntiles |
|---|---|---|---|---|---|---|---|
| M=1056 (512²) | 48 | 32 | 320 | **160** | 62,720 B | 95.7 % | 64 |
| M=8192 (1920x1080) | 64 | 32 | 320 | **128** | 64,512 B | 98.4 % | 80 |

Both built (32 core ELFs each, 11.5 s and 12.2 s of aiecc).  The fill is high on
purpose: aiecc accepted 95.7 % and 98.4 %, i.e. the placer did not need the
margin the plan reserves, and the synthesizer builds what the planner chose.

The zero/accumulate trap the delivered design documents is inherited and handled
the same way: two `zero()` calls in a row collapse (the FIRST is dropped), so a
SACRIFICIAL first zero goes on the tile the epilogue overwrites before reading
(here the drained element itself, which silu writes entirely).

## 2. What was added, and how the knob stays honest

* `webui/npu_plan.py`: `plan(M, fuse_gu=False, fuse_fill=1.0)`.
  `l1_bytes_fused`, `pick_n_fused`, `dispatch_ms_fused` (2x the FLOPs, 2x the B
  bytes, the same A traffic per dispatch), `layer_ms(plan, fused)` counts **one**
  gu.  The fusion narrows the candidate strategies to `xd` (the only executor
  that exists) and adds `"fuse_gu": 1` to `plan_key` -- **only when it is on**,
  so every pre-existing plan_key stays byte-identical.  All 22 baseline keys
  reproduce exactly (verified), and the 24 built manifests stay cache hits.
* `npu_synth.py`: a fused gu is its own design (`ffn_xf`) and its tag carries
  the fusion: `ad_xd_m48k32_M1056_n160f_a1b1_gu` -- the `f` on the n field.  A
  fused and a pure gu of the same n can never share an xclbin name.  The
  manifest records `fuse_gu`, `gu_fused`, `gu_l1`.
* `ffn_design_xf.py`: `ffn_xf` (see section 1) + `bytes_of`.
* `build_sweep.py`: one `elif` registering the design -- no built-in VARIANTS
  entry uses it.
* `pack_kk.py`: `pair_b` / `unpair_b` / `b_pair_bytes`.  The pair is a pure
  reshape of two `pack_b_kk` payloads into `[n-tile][k-step][gate leaf | up
  leaf]`, so the fused B is still a PERMUTATION of the model's 9-byte blocks
  (never a re-encode), and `unpair_b(pair)` is exact.
* `s2_ffn32.py`: `_ffn32_run_fused` -- one gu launch, no barrier, no host silu,
  the same contract (y is the FFN output before ffn_norm2, row-local) and the
  same stats columns (`swiglu` and `barrier` are 0.0).  `pack_payloads` returns
  the pair for a fused plan (3-tuple kept, `b_gate is b_up`), and `plan_for`
  takes `fuse_gu`.
* `plan_ffn.py`: `T22_PLAN_FUSE_GU=1` (default 0) plans, registers and runs it;
  the payload disk cache stores the pair once (2 files, 132.7 MB/layer, the same
  total as the pure path's 3).
* `gate_ffn32.py`: the fused fake device + four fused arms (section 3).

## 3. The host gate (no device): bit-identical, and it bites

```
py312 _scratch/util/place/gate_ffn32.py --all
```

The "device" is functional, not a stub: it decodes the A and pair-B bytes the
BOs actually hold (`unpack_a_kk` / `unpair_b`), runs the two GEMMs with the
kernel's own accumulation model (per-k-step bf16 rounding of C, A quantised to
bfp16 by conv_even as `mm_bfp_mixed.cc` does), applies the epilogue, and writes
the C BO in the device's 8x8 C order.  The reference is the numpy FFN on the
model's own blocks -- it uses neither the packers nor the tile map, so an error
in either cannot cancel.  The epilogue is modelled with the HOST's silu on
purpose: the assertion is that the FUSION PAIRS THE RIGHT TILES, so both sides
must use the same activation (the device's own silu is an approximation -- see
section 5).

~~~
[gate] new      M=96    layers.0 wall  6.28 s corr 1.000000000 maxabs 0 exact 1.0000 record(rep/dev) True/True barriers 1
[gate] new      M=1056  layers.0 wall 54.87 s corr 1.000000000 maxabs 0 exact 1.0000 record(rep/dev) True/True barriers 1
[gate] w        M=1056  layers.0 corr -0.001311850 maxabs 3224
[gate] a        M=1056  layers.0 corr  0.007712145 maxabs 2288
[gate] fused    M=96    layers.0 wall  6.18 s corr 1.000000000 maxabs 0 exact 1.0000 record(rep/dev) True/True barriers 0
[gate] fused    M=1056  layers.0 wall 51.59 s corr 1.000000000 maxabs 0 exact 1.0000 record(rep/dev) True/True barriers 0
[gate] swap     M=1056  layers.0 wall 37.98 s corr 0.503780916 maxabs 1572
[gate] w        M=1056  layers.0 wall 1122.66 s corr -0.001311850 maxabs 3224
[gate] glue_fused M=1056 layers.0 wall 43.36 s corr 1.000000000 maxabs 0 exact 1.0000 barriers 0
...
[gate] ALL PASS -- layouts exact, the record is the plan's, and the gate bites   (1853.4 s)
~~~

45 assertions, 0 failures.  (The `w` arm's 1122 s is this box thrashing, not the
gate: the same arm took 35 s in isolation.)

1. **bit-identity**: at M=96 (6 of 32 slots real, so the dummy-tile path runs)
   and at M=1056 (512²), the fused image's y equals the numpy FFN **exactly**:
   corr 1.0, maxabs 0, exact_frac 1.0.  Not "close" -- the same bits.
2. **the record**: one gu dispatch per ndisp (the pair W BO), one dn, observed
   == expected, twice (reported and observed-from-the-BO-contents).
3. **0 barriers** and 2 x ndisp dispatches: a fused plan is not a same-image
   relaunch, so the +1.7..6.5 ms barrier is gone.
4. **the pair is a permutation** of the model's blocks (unpair + multiset
   identity on both halves).
5. **the gate bites**: `fused_swap` (the pair's halves swapped, i.e.
   silu(up)*gate) collapses to **corr 0.5038**, and `fused_w` (up := gate
   inside the pair) to **corr -0.001311850** -- the SAME number the existing
   pure-path control gives, which is the control the task asked to keep.
   `negctl_w` (-0.001312) and `negctl_a` (0.0077, a real dn k-order error) are
   unchanged on the pure plan.
6. **the product entry point**, not just the executor: the `glue_fused` arm
   calls `plan_ffn.ffn()` -- plan cache, payload LRU (a real pack from
   `weights/block0.npz`), section line -- and is also bit-identical.
7. **the drain** (section 4) is asserted bit-identical to the gather drain for
   the fused, pure and dn geometries.

`check_additive.py` still reports **0 unannounced changes** across
`t18_run.py`, `chunked.py`, `pack_kk.py`, `s2_ffn32.py`, and now also checks
that `T22_PLAN_FUSE_GU` defaults to 0.

## 4. A second finding: the plan path's C drain was 8-12x the device time

Measuring the per-layer ledger exposed a cost no previous report had measured:
`s2_ffn32.unpack_c_plan` decodes the C BO with a fancy-index GATHER per
(m, n) leaf, and that gather costs **12-322 ms per dispatch** on the plan's
geometries -- more than the dispatch itself at 1920x1080.

The 8x8 C permutation is a BLOCK transpose (`[m/8][n/8][8][8] -> [m/8][8][n/8][8]`),
so one reshape + transpose does the same work in bulk memory order.  Added
`unpack_c_plan_fast` (numpy, bit-identical) and `unpack_c_plan_numba` (the same
walk in C, used when numba is importable), behind `drain_c()`; both are asserted
equal to the gather on every geometry the executors meet, including the 22-of-32
dummy tile map.  `T22_PLAN_FAST_DRAIN=0` restores the gather.

| per dispatch | gather (before) | bulk transpose | numba (device leg) |
|---|---|---|---|
| fused gu C, 31.4 MB (M=1056) | 96.3 ms | 11.1 ms | **11.1 ms** |
| pure gu C, 2 x 31.4 MB | 104.2 ms | 22.1 ms | **22.1 ms** |
| fused gu C, 41.9 MB (M=8192) | 185.9 ms | 27.3 ms | **27.3 ms** |
| dn C, 11.8 MB | 44.7 ms | 5.0 ms | **5.0 ms** |

Without it the fused path would be *slower* than the delivered one (the 512²
ledger reads ~176 ms/layer instead of 90.2, and ~1.1 s/layer at 1920x1080).  It
is an independent win for every shape the plan serves, and it is why the
arithmetic in section 6 closes at all.

## 5. Device numbers

```
py310 _scratch/util/place/fuse_gu_devtest.py --m 1056            --runs 9
py310 _scratch/util/place/fuse_gu_devtest.py --hw 1080 1920      --runs 9
```

One process per arm (the gate_ffn32 --all pattern), the model's own layer-0
blocks, a deterministic bf16 A, and a plain f32 matmul of the decoded blocks as
the reference.  Every repeat returned ERT state 4.

### 512² -- M=1056, m=48, kk=32, ndisp=1

| arm | image | ms (median of 9) | min..max | TFLOP/s | corr |
|---|---|---|---|---|---|
| **fused gu** | `ad_xd_m48k32_M1056_n160f_a1b1_gu` | **36.47** | 35.78..42.99 | **4.55** | **0.999825685** |
| pure gu | `ad_xd_m48k32_M1056_n320_a1b1_gu` | 19.71 | 17.96..21.11 | 4.21 | 0.999916525 |
| swap (negctl) | the fused image, pair halves swapped | 36.81 | 35.54..43.38 | 4.51 | **0.504034617** |
| dn | `ad_xd_m48k32_M1056_n384_a1b1_dn` | 19.81 | 17.79..21.23 | 4.19 | (timing) |

### 1920x1080 -- M=8192, m=64, kk=32, ndisp=4

| arm | image | ms (median of 9) | min..max | TFLOP/s | corr |
|---|---|---|---|---|---|
| **fused gu** | `ad_xd_m64k32_M2048_n128f_a1b1_gu` | **44.60** | 43.43..49.94 | **7.22** | **0.999825602** |
| pure gu | `ad_xd_m64k32_M2048_n320_a1b1_gu` | 21.57 | 20.41..24.34 | 7.47 | 0.999916486 |
| swap (negctl) | the fused image, pair halves swapped | 43.40 | 40.08..49.69 | 7.42 | **0.504366653** |
| dn | `ad_xd_m64k32_M2048_n320_a1b1_dn` | 19.87 | 19.70..23.32 | 8.11 | (timing) |

The per-n-tile and per-m-tile correlations equal the whole-array correlation in
both rows (0.99982 / 0.99983 / 0.99983 at 512²), so **no n-tile is corrupted** --
that is the check that would catch the zero/accumulate collapse the delivered
design documents, and it passes.

**Why 0.9998 and not 1.0.**  The *pure* gu image, which has no activation at
all, is 0.99992 against the same reference: the gap is the kernel's per-k-step
bf16 C rounding plus the bfp16 quantisation of A, not the fusion.  The fused
image's extra 1e-4 is the device's silu: `silu_sized` is `silu.cc`'s
tanh kernel (`x * 0.5 * (1 + tanh(x/2))` in bf16), while the reference uses an
f32 sigmoid.  The delivered `swiglu11` image runs the very same kernel, and
the host gate is where bit-identity is proven (both sides run the host's
epilogue there).

**Against the task's device target**: the fused gu lands at **36.5 ms** (512²)
and **44.6 ms** (per 2048 rows at 1920x1080) against the delivered gu's 45.9 ms,
and dn at **19.8 / 19.9 ms** against 31.2 ms.  Both targets met.

## 6. The per-layer arithmetic, and the honest 8-step delta

Every number below is measured on this box except the two marked (extrapolated).

### 512² (M=1056, ndisp=1), ms per layer

| column | delivered (measured, recorded 8-step medians) | plan, unfused | plan, **fused** |
|---|---|---|---|
| gu dispatch | 45.9 (one fused swiglu11, G=10) | 2 x 19.7 = 39.4 | **36.5 (one)** |
| barrier | 0 | ~6 (same-image relaunch, NOT in the 247.0 total) | **0** |
| host silu(gate)*up | **0 (on device)** | **139.3** | **0 (on device)** |
| dn dispatch | 31.2 | 19.8 | 19.8 |
| pack A | 8.5 | 2.4 + 9.4 = 11.8 | 11.8 |
| gu W write | 3.6 | 7.2 (two 44.2 MB) | 3.6 (one 88.5 MB pair) |
| C read | 2.4 | 2.4 | 2.4 |
| C drain | 10.7 (numba, 8x8) | 22.1 (bulk) / 104.2 (gather) | **11.1** |
| **total** | **102.4** | **247.0** | **90.2** |

```
fused vs unfused  : -156.9 ms/layer  -> -37.7 s per 8 steps (240 layer-steps)
fused vs delivered:  -12.2 ms/layer  ->  -2.9 s per 8 steps
```

(The unfused column does not charge the ~6 ms same-image barrier it would also
need -- charging it would make the fusion's win 6 ms/layer larger, so the
comparison above is conservative in the fusion's favour only by that amount.)

### 1920x1080 (M=8192, ndisp=4), ms per layer

| column | delivered (8 chunks x measured 512² layer) | plan, unfused | plan, **fused** |
|---|---|---|---|
| dispatches (fused/pure gu + dn) | 8 x 77.1 = 616.8 | 4 x (2x21.6+19.9+302.6) = 1462.2 | 4 x (44.6+19.9) = **257.9** |
| host silu | 0 | **1210.0** (4 x 302.6) | **0** |
| pack / drain | 8 x 19.2 = 153.6 | 306.8 | 231.5 |
| gu W write + C read | 8 x 6.0 = 48.0 | 38.4 | 24.0 |
| **total** | **819.2** (extrapolated) | **1807.4** | **513.4** |

```
fused vs unfused  : -1294.0 ms/layer -> -310.6 s per 8 steps
fused vs delivered:  -305.8 ms/layer ->  -73.4 s per 8 steps
```

### On the "-14 s" this was supposed to buy

At **512² it is -2.9 s, not -14 s**, and the reason is arithmetic, not a defect:

* the delivered path at 512² is at its *best*: its G=10 fused image covers all
  1056 rows in one dispatch (45.9 ms) and its dn image is 31.2 ms, i.e. 77.1 ms
  of dispatch.  The plan's fused pair is 56.3 ms, so the device win is 20.8 ms.
* the plan path's *extra* host work eats 8.7 ms of it (the dn A must be packed
  from the fused h -- the delivered path hands its C BO over zero-copy -- and
  two drains instead of one), and the delivered host columns are already cheap
  (25.2 ms).
* the fusion's own contribution is huge (**-156.9 ms/layer**, because it deletes
  a 139.3 ms host pass), and it is what turns a **+144.6 ms/layer regression
  into a -12.2 ms/layer win**.  The regression the previous report measured was
  +64 ms/layer because it estimated the host pass at ~50 ms; measured here it is
  **139.3 ms** at 512² and **1210 ms/layer** at 1920x1080.

At **1920x1080 the same change is worth -73.4 s per 8 steps**, because there the
delivered path needs 8 chunk-dispatches per layer (1056 rows each) while the
plan needs 4 (2048 rows each), and the host silu it deletes is 1.2 s per layer.

## 7. What is proven, and what is not

**Proven (host, no device)**
* the fused gu image's output is BIT-IDENTICAL to (pure-GEMM gu) then host
  silu(gate)*up, at M=96 (dummy tiles) and M=1056, through the product entry
  point as well;
* the pair is a permutation of the model's blocks; unpair is exact;
* the dispatch record is the plan's, and the gate bites on two mis-wirings
  (swapped halves 0.5038, up:=gate -0.001311850);
* the delivered path is intact: `check_additive.py` 0 unannounced changes, the
  flag defaults to 0, all 22 baseline plan_keys reproduce, the pure plan's
  selftest is unchanged;
* the bulk/numba drains equal the gather drain bit for bit.

**Proven (device)**
* corr 0.9998 vs the numpy reference at both sizes, with the per-n-tile and
  per-m-tile correlations equal to it (no tile corrupted);
* the swapped-pair control collapses on the device too (0.5040);
* 36.5 / 44.6 ms and 4.55 / 7.22 TFLOP/s for the fused gu; dn 19.8 / 19.9 ms.

**Not proven**
* **the 8-step runs themselves.** Both deltas are per-layer arithmetic from
  measured columns; the only authority is the A/B in
  `_scratch/util/place/PLAN-FFN-512-AB.md` (arm B now needs
  `T22_PLAN_FUSE_GU=1`), and the 1920x1080 delivered baseline is an
  extrapolation (8 x the measured 512² layer; the delivered path re-writes the
  gu W per chunk, so if anything that baseline is optimistic *for the delivered
  path*).
* **the device's silu vs the host's**: the fused image inherits the delivered
  image family's tanh-approximated silu.  Its error is inside the 0.9998
  correlation but it is not bit-identical to the host reference; if the product
  ever needs that, the epilogue kernel is the thing to change, not the fusion.
* **attention, the projection and the t33 shapes** are untouched by this change
  (the same gaps PLAN-FFN-WIRING-REPORT.md section 6 lists).  The fusion covers
  the FFN gate/up of every shape the planner serves with `xd`.
* **the timing spread** on this box is 10-20 % between invocations (the box is
  shared); the medians of 9 are quoted, and the conclusions do not depend on
  which sample is taken.

## 8. Files

**New**

| file | what |
|---|---|
| `_scratch/util/place/ffn_design_xf.py` | the fused gu image (ffn_xb placement + the delivered swiglu11 kernel body) |
| `_scratch/util/place/fuse_gu_devtest.py` | the device test (fused / pure / swap / dn / host / hostcost arms) + the per-layer ledger |
| `webui/shape_manifests/plan_xd_m48_kk32_Mblk1056_n384-160-384_c2bae462.json` | the M=1056 fused plan record |
| `webui/shape_manifests/plan_xd_m64_kk32_Mblk2048_n320-128-320_892e4417.json` | the M=8192 fused plan record |
| `_scratch/util/place/out/ad_xd_m48k32_M1056_n160f_a1b1_gu.{xclbin,bin,log}` | the 512² fused image |
| `_scratch/util/place/out/ad_xd_m64k32_M2048_n128f_a1b1_gu.{xclbin,bin,log}` | the 1920x1080 fused image |
| `_scratch/util/place/out/ad_xd_m16k32_M96_n320f_a1b1_gu.{xclbin,bin,log}` | the M=96 gate fixture |
| `_scratch/util/place/out/gen_*_c2bae462.json`, `gen_*_892e4417.json` | the generated VARIANT tables (npu_synth) |
| `_scratch/util/place/out/fuse_gu_dev_m{1056,8192}.json` | the device results |
| `FUSE-GU-REPORT.md` | this report |

**Modified**

| file | change |
|---|---|
| `webui/npu_plan.py` | `fuse_gu` / `fuse_fill`; `l1_bytes_fused`, `pick_n_fused`, `dispatch_ms_fused`; `layer_ms(plan, fused)`; plan_key carries the fusion only when on; selftest + CLI |
| `_scratch/util/place/npu_synth.py` | `DESIGN_FUSED_GU`, the `f` tag flag, the fused VARIANT branch, manifest `fuse_gu`/`gu_fused`/`gu_l1`, `--fuse` / `--fuse-fill` |
| `_scratch/util/place/build_sweep.py` | one `elif` registering `ffn_xf` |
| `_scratch/util/place/pack_kk.py` | `pair_b`, `unpair_b`, `b_pair_bytes` (+ `__all__ +=`) |
| `_scratch/util/place/s2_ffn32.py` | `plan_for(fuse_gu)`, `pack_payloads` fused branch, `_ffn32_run_fused`, `unpack_c_plan_fast` / `unpack_c_plan_numba` / `drain_c`, `import os` |
| `_scratch/util/place/gate_ffn32.py` | the fused fake device, four fused arms, the drain-identity checks |
| `_scratch/util/place/plan_ffn.py` | `T22_PLAN_FUSE_GU` (default 0), fused payload cache, fused register-check, fused selftest |
| `_scratch/util/place/check_additive.py` | the new flag's default-0 check |

## 9. Reproduce

~~~powershell
$py312 = D:\Project\deepseek_harness\test7\.winml_venv\Scripts\python.exe
$py310 = D:\Users\15081\AppData\Local\GAIA\python\python.exe

# 1. the plan (pure output must be unchanged; the fused plan is the new one)
& $py312 webui\npu_plan.py --selftest
& $py312 webui\npu_plan.py --grid --fuse

# 2. build the fused image for a shape (one aiecc per image, ~12 s)
& $py312 _scratch\util\place\npu_synth.py --m 1056 --fuse
& $py312 _scratch\util\place\npu_synth.py --hw 1080 1920 --fuse

# 3. the host gate: bit-identity, the record, the controls, the drains (~30 min
#    on a loaded box; the drain checks alone are seconds)
& $py312 _scratch\util\place\gate_ffn32.py --drains
& $py312 _scratch\util\place\gate_ffn32.py --all

# 4. the additive proof + the product selftests
& $py312 _scratch\util\place\check_additive.py
& $py312 _scratch\util\place\plan_ffn.py --selftest --fuse
$env:T22_PLAN_FUSE_GU="1"; & $py312 _scratch\util\place\plan_ffn.py --m 1056 --register-check

# 5. the device (owns the NPU; one process per arm)
& $py310 _scratch\util\place\fuse_gu_devtest.py --m 1056 --runs 9
& $py310 _scratch\util\place\fuse_gu_devtest.py --hw 1080 1920 --runs 9

# 6. the product arm (the only authority for the 8-step delta):
#    ZIMAGE_BIGM=1 T22_DEVKILL=1 ZIMAGE_CHUNK_PIPE=0 T22_PLAN_FFN=1
#    T22_PLAN_FUSE_GU=1     (see PLAN-FFN-512-AB.md)
~~~
