# Adaptive NPU path — planner, synthesizer, BO sizing, cache, refusal, validation

**2026-11-03 · T22 adaptive · new modules + new COMPILE entries + new manifest keys only, behind flags**

```
THE ONE SENTENCE: any (H, W) in the product envelope now resolves to a legal,
built, cached 32-core placement — or to a refusal that quotes the exact binding
constraint in bytes, slots or rows.  Twenty sizes were planned and BUILT, and
five of them were run on the device against numpy on real layer weights.
```

---

## 0. What was actually wrong

Three hardcoded facts decided whether a request could run, and a fourth told the
user it could when it could not.

| # | where | what it said | what it cost |
|---|---|---|---|
| 1 | `webui/build_shape.py:51` `COMPILE` | recipes for exactly M = 128 / 160 / 288 | every other size was "no build in this tree" |
| 2 | `imaggers/image/t18_run.py:377` | projection input BO = 8,110,080 B = 1056 rows ("11 tiles") | 1024² (M=4128, 31,703,040 B needed) died with *attempting to write past buffer size* |
| 3 | `imaggers/util/place/ffn_design*` | one tile per core, 96-row blocks | 384² (M=608) and 768² (M=2336) died on `N % 96 != 0` |
| 4 | `webui/shapes.py chunk_gate()` | "runnable" from an L1 + host-RAM model | it never looked at 1–3, so 1024² was promised at "~1.5 h" |

The fix is not a bigger table.  It is a **pure planner** that resolves any M to a
legal geometry, a **synthesizer** that builds it, a **BO derivation** that cannot
disagree with the plan, a **cache keyed on the plan**, and a **gate that quotes
the constraint**.

---

## 1. THE PLANNER — `webui/npu_plan.py`

### 1.1 Signature

```python
def plan(M, *, m_choices=(16, 32, 48, 64), kk_choices=(8, 16, 32),
         strategies=("xd", "t33"), mixed=True, adepth=1, bdepth=1,
         prefer=None) -> dict
```

Returns:

```python
{"ok": bool,          # is there a legal placement for this M
 "plan": {...} | None, # the chosen geometry (see 1.3)
 "rejected": [{"strategy", "m", "kk", "constraint", "detail"}, ...],
 "binding": str|None, # the constraint that killed the last family
 "reason": str}       # one sentence, "" when ok
```

It is **pure**: no filesystem, no numpy, no toolchain, no device, no randomness.
Every number is a measured constant, and `npu_plan.py --selftest` re-derives
the measured L1 points to prove the model has not drifted.

### 1.2 The constraint list (all measured, none guessed)

| id | constraint | source | value |
|---|---|---|---|
| C1 | `m % 16 == 0` | `mm_bfp.cc:303` `static_assert(m % (2*r))`, r = 8 | m ∈ {16, 32, 48, 64} |
| C2 | `n % 16 == 0` and `N % n == 0` | `mm_bfp.cc:305` + `ffn_design_xb` `assert N // n == ntiles` | n divides 10240 **and** 3840 |
| C3 | `kk % 8 == 0`, `K % kk == 0` | `mm_bfp.cc:304` | kk ∈ {8, 16, 32} (kk = 64 excluded, see below) |
| C4 | per-core L1 ≤ 65,536 B | `ffn_design_xb.py:163`, `t33_plan.worst_l1` (corrected, §7.1) | `T*m*n*2 + 9*kk*n/8 + T*m*kk*2 + 2,048` |
| C5 | slots = cols × cpc = 32 | the array | `xd`: nm ≤ 32; `t33`: T = ceil(nm/32) |
| C6 | `c_leaf ≤ 65,532 B` | `aie.dma_bd`: 16,383 × 32-bit words | implied by C4, kept as a belt |
| C7 | rows a dispatch covers = Mblk | the designs' `M` is compile-time | ndisp = ceil(M / Mblk), tail zero-padded |
| C8 | kk > 32 is DMA-gated | measured 3.78 / 5.12 / 5.59 TFLOP/s at N=10240 | do not plan with it |

Two strategies, both already in the tree:

* **xd** — `ffn_design_xb.ffn_xb`: one M-tile per core, ndisp dispatches of
  Mblk = m·ceil(M / (m·ndisp)) rows, ndisp = ceil(M / 32m).
* **t33** — `ffn_design_t33.ffn_t33`: ONE dispatch, T = ceil(nm/32) tiles per
  core, per-column tile counts from `tile_map_t(..., rot=7)` (the verified form:
  rot=0 corrupted every column-0 block at odd n-tiles).

### 1.3 What a plan carries

```python
{"M", "m", "kk", "Mblk", "nm", "T", "ndisp", "strategy", "cols": 8, "cpc": 4,
 "mixed", "adepth", "bdepth", "cwait", "pad_rows", "per_col", "per_core",
 "tile_map", "dispatches": [[row_start, rows], ...],
 "gemms": {"gu"|"dn"|"proj": {"K", "N", "n", "ntiles", "nkk", "l1",
                              "a_bytes", "b_bytes", "c_bytes", "dispatch_ms"}},
 "l1_max", "layer_ms", "step_s", "s_8step", "h_8step", "plan_key"}
```

`a_bytes`/`c_bytes` are the **design's own BO sizes**, not the useful-work
size: `ffn_xb` always allocates all 32 slots (`cols*nkk*cpc*a_leaf`,
`cols*ntiles*cpc*c_leaf`), while `ffn_t33` allocates exactly the Mblk rows.
Getting this wrong is how a plan silently under-sizes a BO.

### 1.4 Cost model (an estimate, never a promise)

```
t_dispatch = max( T * 2*m*K*N / (rate(kk)/32) ,  (ntiles*a_bytes + b_bytes + c_bytes) / 42 GB/s ) + 1.0 ms
layer_ms   = (2*gu + 1*dn + 4*proj) * ndisp          # gate+up, down, q/k/v/o
```

`rate = {8: 3.78, 16: 4.60, 32: 5.12} TFLOP/s` (measured at N=10240; kk=16 is the
interpolated midpoint), BW = 42 GB/s (measured), 1.0 ms per dispatch (the
measured one-shot staging penalty is 2.5 ms, a burst is 0; 1.0 is the midpoint).
A partially filled array is charged the **saturated** per-core rate, so the
estimate is an upper bound — the honest direction.

Ties inside 0.5 ms are broken by `_rank`: ffn_xb before ffn_t33, fewer
dispatches, smaller T, larger m.  A plan that is already built wins a tie inside
15 % of the best cost (`prefer`), which is what makes the cache sticky.

### 1.5 Why kk = 32 and not 64

The inner-loop density depends only on `colA = kk/8` (26 / 64 / 79 issue slots at
kk = 8 / 32 / 64), but the measured array rate is 3.78 / 5.12 / **5.59**
TFLOP/s: kk = 64 buys 9 % over kk = 32 while halving the largest legal n
(L1: `n ≤ (63488 − 2*m*kk) / (2*m*T + 9*kk/8)`), which multiplies the A
re-stream.  The planner therefore caps at kk = 32.

---

## 2. THE SYNTHESIZER — `imaggers/util/place/npu_synth.py`

### 2.1 Entry point

```
py312 npu_synth.py --m 608                  # plan -> variants -> build all three
py312 npu_synth.py --hw 384 384             # same, from a requested (H, W)
py312 npu_synth.py --grid --json out/grid_build.json
py312 npu_synth.py --list --m 4128          # show the VARIANTs, build nothing
```

```python
def synth(M, force=False, prefer_built=True, note="") -> dict   # plan+build+manifest
def variants_for(plan) -> {tag: {"design", "kw"}}               # build_sweep entries
def build_plan(plan, force=False) -> {"ok", "images", "gen", ...}
def write_manifest(plan, rec) -> pathlib.Path                    # plan_<key>.json
def manifest_for(plan) -> dict | None                            # the cache hit test
```

### 2.2 How a plan becomes a VARIANT

One entry per GEMM image, and nothing is re-derived:

```python
ad_t33_m48k32_M4128_n160_a1b1cw_gu   design=ffn_t33
  kw = {M: 4128, K: 3840, N: 10240, m: 48, kk: 32, n: 160, ntiles: 64,
        cols: 8, cpc: 4, mixed: true, adepth: 1, bdepth: 1, rot: 7, cwait: true}
```

The tag encodes **every** compile-time knob (including `cwait`, see §7.2), so two
different images can never share an xclbin name.

`build_sweep.py` gained exactly one option, `--gen <json>`, which merges a
generated table into its `VARIANTS` dict and refuses a collision.  Without
`--gen` the module is byte-for-byte what it was.

### 2.3 Packers wired

| operand | packer | note |
|---|---|---|
| A, one tile/core | `pack_kk.pack_a_kk` | kk-parametric, 179 bit-verified checks |
| A, multi-tile | `verify_t33.pack_a_t33` | the T-tile column element form |
| B, both | `pack_kk.pack_b_kk` | **permutation only** — the model's own BFP16 blocks are re-ordered, never re-encoded (0.297 % of them are non-canonical) |

---

## 3. BO SIZING — `imaggers/image/t18_run.py`

Every M-dependent host BO was a 512² literal.  Each is `rows × width × 2`, so the
derivation is exact and the delivered path is unchanged:

| BO | old literal | now | at the delivered size |
|---|---|---|---|
| whole-shape x / out | 8,110,080 | `max(1056, SEQ) * 3840 * 2` | identical |
| whole-shape ff | 21,626,880 | `max(1056, SEQ) * 10240 * 2` | identical |
| rt17 chunk x / out | 12,533,760 | `BIGM_CHUNK_ROWS * 3840 * 2` (1632) | identical |
| rt17 chunk ff | 33,423,360 | `BIGM_CHUNK_ROWS * 10240 * 2` (1632) | identical |

```python
def bigm_bo_bytes(rows, width): return int(rows) * int(width) * 2
BIGM_WHOLE_ROWS = int(os.environ.get("T22_ADAPT_M", "1056") or 1056)
def refresh_bigm_sizes(seq):     # called from configure() once SEQ is known
    rows = max(1056, int(seq))   # the RT=11 image still needs its own 1056 rows
```

and a module-level `assert (_BX, _BFF) == (8_110_080, 21_626_880)` proves the
delivered size is byte-for-byte the old literal.

**A BO is necessary but not sufficient.**  The image is compiled for RT row
tiles, so a shape that needs more rows than its image has must run the adaptive
image set, not a wider BO.  That is exactly what §6's refusal now says.

---

## 4. CACHE — keyed on the PLAN

`webui/build_shape.py` already had the `SHAPE_MANIFESTS` mechanism; the
adaptive path reuses it with a different key:

* manifest file: `webui/shape_manifests/plan_<plan_key>.json`
* `plan_key` covers **every** image-defining parameter — strategy, m, kk, Mblk,
  T, cols, cpc, mixed, adepth, bdepth, cwait, and the three n values — and
  nothing that is shape-specific (H, W, M, ndisp, pad are not in the key).
* a second size that lands on the same plan **merges** into the same entry
  (`used_by` / `ndisp_by_M`) instead of forking it.

Measured on the 20-size grid: **16 distinct plan keys for 20 sizes**.

| shared plan | sizes |
|---|---|
| `xd_m16_kk32_Mblk288_n640-768-768_...` | 256×256, 128×512 |
| `xd_m32_kk32_Mblk608_n480-512-480_...` | 384×384, 320×448 |
| `xd_m32_kk32_Mblk800_n480-512-480_...` | 384×512, 512×768 |

---

## 5. REFUSAL — the gate that no longer over-promises

`chunk_gate()` now asks, in this order, and quotes the binding constraint:

1. **the token envelope** (M ∈ 64..16,416) — same limit as before, new wording;
2. **a legal placement must exist** — `npu_plan.plan(M)`; if not, the refusal
   quotes the L1 / slot arithmetic;
3. **the host BOs must cover M** — `npu_plan.bo_sizes(M)`, and when a whole-shape
   image is in play (`T22_HOSTKILL` / `T22_DEVKILL`) the BO must be at least
   `max(1056, M)` rows, or the sentence names `T22_ADAPT_M`;
4. **the adaptive path must be BUILT** when `ZIMAGE_ADAPT=1` — a claim without a
   build is not evidence;
5. the delivered chunked-path limits (host attention RAM, 8-step ceiling, torch).

Every refusal goes through one function so the front end and the builder cannot
describe the same shape two different ways:

```python
def refusal(h, w, why, m=None, b=None):
    return "%s (M=%d) 未构建：%s。" % (b["run_key"], m, why)
```

and `build_shape._no_build()` now *adds the plan* instead of ending on "no
recipe":

> 1024x1024 (M=4128) 未构建：… 自适应放置已规划：plan
> t33_m48_kk32_Mblk4128_n160-160-160_9606ce93（t33 m=48 kk=32 Mblk=4128，1 次
> dispatch，L1<=63104 B，pad 0 行）；构建入口 py312
> imaggers/util/place/npu_synth.py --m 4128。

The estimate a runnable size gets is now plan-anchored too: `chunk_estimate()`
carries `plan`, `plan_key`, `plan_l1_max`, `plan_rows`,
`plan_dispatches`, `plan_per_col`, `bo_x_bytes`, `bo_ff_bytes`,
so the UI can show the geometry the NPU would actually run.  For 1024² the
honest number is **0.41 h for 8 steps** (187 s/step, the DELIVERED chunked
model: 171 s/step of 96-row-block staging + host glue, plus 13 s/step of host
f32 attention) — with the plan's own device GEMM named separately at
9.8 s/step, because replacing that device half is exactly the wiring §9 still
lists.  Not the old "~1.5 h", and not "runnable because the arithmetic fits".

---

## 6. VALIDATION TABLE

Every size was planned, built and checked.  `BUILT` means an aiecc build
produced the xclbin; the L1 column is the planner's number with the
**compiler's own printed value** in parentheses, so the two cannot drift.

| size | tokens | M | plan | n gu/dn/proj | L1 B (compiler) | per-column tiles | images built | verdict |
|---|---|---|---|---|---|---|---|---|
| 128 x 128 | 64 | 96 | xd m=16 kk=32 Mblk=96 x1 T=1 | 640/768/768 | 55296 (=46592,55296) | 1,1,1,1,1,1,0,0 | 3/3 | OK |
| 160 x 160 | 100 | 160 | xd m=16 kk=32 Mblk=160 x1 T=1 | 640/768/768 | 55296 (=46592,55296) | 2,2,1,1,1,1,1,1 | 3/3 | OK |
| 192 x 192 | 144 | 192 | xd m=16 kk=32 Mblk=192 x1 T=1 | 640/768/768 | 55296 (=46592,55296) | 2,2,2,2,1,1,1,1 | 3/3 | OK |
| 224 x 224 | 196 | 256 | xd m=16 kk=32 Mblk=256 x1 T=1 | 640/768/768 | 55296 (=46592,55296) | 2,2,2,2,2,2,2,2 | 3/3 | OK |
| 256 x 256 | 256 | 288 | xd m=16 kk=32 Mblk=288 x1 T=1 | 640/768/768 | 55296 (=46592,55296) | 3,3,2,2,2,2,2,2 | 3/3 | OK |
| 288 x 288 | 324 | 384 | xd m=16 kk=32 Mblk=384 x1 T=1 | 640/768/768 | 55296 (=46592,55296) | 3,3,3,3,3,3,3,3 | 3/3 | OK |
| 320 x 320 | 400 | 448 | xd m=16 kk=32 Mblk=448 x1 T=1 | 640/768/768 | 55296 (=46592,55296) | 4,4,4,4,3,3,3,3 | 3/3 | OK |
| 384 x 384 | 576 | 608 | xd m=32 kk=32 Mblk=608 x1 T=1 | 512/480/480 | 55296 (=52096,55296) | 3,3,3,2,2,2,2,2 | 3/3 | OK |
| 448 x 448 | 784 | 832 | xd m=32 kk=32 Mblk=832 x1 T=1 | 512/480/480 | 55296 (=52096,55296) | 4,4,3,3,3,3,3,3 | 3/3 | OK |
| 512 x 512 | 1024 | 1056 | xd m=48 kk=32 Mblk=1056 x1 T=1 | 320/384/384 | 55808 (=47360,55808) | 3,3,3,3,3,3,2,2 | 3/3 | OK |
| 640 x 640 | 1600 | 1632 | xd m=32 kk=32 Mblk=832 x2 T=1 | 512/480/480 | 55296 (=52096,55296) | 4,4,3,3,3,3,3,3 | 3/3 | OK |
| 768 x 768 | 2304 | 2336 | t33 m=16 kk=32 Mblk=2336 x1 T=5 | 256/256/256 | 57344 (=53248) | 19,18,18,18,18,18,18,19 | 3/3 | OK (compiler prints its design's short L1 53248; the A element is T*a_leaf -- section 7.1) |
| 896 x 896 | 3136 | 3168 | t33 m=16 kk=32 Mblk=3168 x1 T=7 | 160/192/192 | 59136 (=44672,52992) | 25,25,25,25,25,24,24,25 | 3/3 | OK (compiler prints its design's short L1 44672,52992; the A element is T*a_leaf -- section 7.1) |
| 1024 x 1024 | 4096 | 4128 | t33 m=48 kk=32 Mblk=4128 x1 T=3 | 160/160/160 | 63104 (=56960) | 11,11,11,11,11,10,10,11 | 3/3 | OK (compiler prints its design's short L1 56960; the A element is T*a_leaf -- section 7.1) |
| 192 x 384 | 288 | 320 | xd m=16 kk=32 Mblk=320 x1 T=1 | 640/768/768 | 55296 (=46592,55296) | 3,3,3,3,2,2,2,2 | 3/3 | OK |
| 256 x 512 | 512 | 544 | xd m=32 kk=32 Mblk=544 x1 T=1 | 512/480/480 | 55296 (=52096,55296) | 3,2,2,2,2,2,2,2 | 3/3 | OK |
| 384 x 512 | 768 | 800 | xd m=32 kk=32 Mblk=800 x1 T=1 | 512/480/480 | 55296 (=52096,55296) | 4,3,3,3,3,3,3,3 | 3/3 | OK |
| 128 x 512 | 256 | 288 | xd m=16 kk=32 Mblk=288 x1 T=1 | 640/768/768 | 55296 (=46592,55296) | 3,3,2,2,2,2,2,2 | 3/3 | OK |
| 512 x 768 | 1536 | 1568 | xd m=32 kk=32 Mblk=800 x2 T=1 | 512/480/480 | 55296 (=52096,55296) | 4,3,3,3,3,3,3,3 | 3/3 | OK |
| 320 x 448 | 560 | 608 | xd m=32 kk=32 Mblk=608 x1 T=1 | 512/480/480 | 55296 (=52096,55296) | 3,3,3,2,2,2,2,2 | 3/3 | OK |

### 6.1 Refusals (what cannot be planned)

| size | M | plan | binding constraint |
|---|---|---|---|
| 48 x 48 | 64 | REFUSED | shape out of range: each side must be 64..2048 |
| 4096 x 4096 | 65568 | REFUSED | shape out of range: each side must be 64..2048 |
| 2048 x 4096 | 32800 | REFUSED | shape out of range: each side must be 64..2048 |

### 6.2 The planner's own M-envelope refusal

M=32,800 (a 2048 x 4096 request): ok=False, binding=token envelope

> M=32800 is outside the product envelope 64..16416

### 6.3 The L1 gate at its boundary

| m | kk | T | largest legal n | L1 B | next n | L1 B | L1 cap |
|---|---|---|---|---|---|---|---|
| 32 | 32 | 1 | 512 | 55,296 | 640 | 68,096 | 614 |
| 32 | 64 | 1 | 384 | 58,368 | 512 | 75,776 | 436 |
| 16 | 32 | 7 | 192 | 59,136 | 224 | 67,456 | 216 |
| 48 | 32 | 3 | 160 | 63,104 | 192 | 73,472 | 167 |

Inside the envelope the planner never refuses: `n = 16` always fits L1 at every
legal m, so a legal placement exists for every M ∈ 64..16,416.  The refusals
that are real are (a) the envelope, (b) the measured L1 gate at a candidate
geometry, (c) the product BO coverage, (d) the build itself — which is the final
arbiter and did refuse once (§7.1).

---

## 7. NEW MEASURED FINDINGS

### 7.1 The t33 L1 model was 2,048·(T−1) bytes short — and it cost a build

`t33_plan.worst_l1()` charges `adepth * a_leaf` for A.  A core that owns T
M-tiles acquires ONE A element of **T·a_leaf** bytes (the split hands it
`sum(T[:p]) * a_leaf`), so the object the placer must fit is T·a_leaf.

Proof, from this session's build log (M=3168, m=16, kk=32, n=240, T=7):

```
[ffn_t33] … L1=65,472B                       <- the short formula, "fits"
error: 'aie.buffer' op could not be placed: buffer "rowA7_0_cons_buff_0"
       needs 7168 bytes and this tile has no room left for it
```

7 × 1,024 = 7,168 — the A element, not one leaf.  The corrected L1 is 71,616 B
and the planner now refuses that geometry, picking n=192 (L1 59,136 B) which
built.  `npu_plan.l1_bytes` is the corrected gate and its docstring carries all
six measured points.

**Also**: the literature's "m=32 kk=64 n=512 → 73,728 B" omits the 2,048 B
stack; the design's own assert prints **75,776**, which is what the planner
reproduces.

### 7.2 ffn_t33's per-n-tile C drain corrupts the LAST n-tile

The design awaits only the last column's C drain per n-tile
(`wait=(cwait or c == cols - 1)`).  At M=4128 (86 tiles, 64 n-tiles) the final
n-tile came back **18 % wrong** while every other n-tile was exact to one bf16
ulp:

| image | corr | maxabs (of 29.5) | worst n-tile |
|---|---|---|---|
| t33, `cwait=False` (delivered form) | 0.999266 | 20.875 | nt=63, 119,605 elements > 1e-3 |
| t33, `cwait=True` | **0.99999835** | **0.5** | none (all n-tiles ≤ 0.5) |

`cwait=True` is the design's own knob for this bug class (SWEEP-T33-REPORT
§4.1).  The planner now sets it for every t33 plan, it is part of `plan_key`,
and the per-n-tile diagnostic that found it is kept as
`imaggers/util/adaptive/_diag_1024.py`.  The `xd` form (ffn_xb) has no
such knob and no measurement shows the effect at its ntiles ≤ 20.

### 7.3 The planner rediscovers the delivered one-dispatch shapes

* M=1056 → **xd m=48**, 22 tiles, ONE dispatch — the shape `ffn_gu_m48` already
  uses, at kk=32/n=320 instead of kk=8/n=512.
* M=608 → xd m=32, 19 tiles, one dispatch (today it dies on `N % 96 != 0`).
* M=2336 → t33 m=16, 146 tiles, T=5, one dispatch.

---

## 8. DEVICE TESTS (py310, real layer-0 weights, one launcher at a time)

Weights are the model's own BFP16 blocks (`weights/block0.npz`:
`layers.0.feed_forward.w1`, `.w2`, `layers.0.attention.to_q`), reshaped
only.  A is a seeded random activation, bf16-rounded.  The reference is the
per-call-bf16 model (the kernel's C operand is a bf16 tile, so the running sum
is rounded K/kk times) with A quantised to bfp16 (conv_even), exactly as
`mm_bfp_tile_ref` does.

The negative control permutes A's **k-steps inside the packed payload** — a real
k-order error, not a packing mistake — and must collapse.

| size | M | plan | GEMM | corr | maxabs (ref max) | ms / dispatch | TFLOP/s | negative control |
|---|---|---|---|---|---|---|---|---|
| 384 x 384 | 608 | xd m=32 kk=32 Mblk=608 x1 | gu | 0.99999827 | 0.4375 (29.5) | 19.15 | 2.50 | collapsed |
| 768 x 768 | 2336 | t33 m=16 kk=32 Mblk=2336 x1 T=5 | gu | 0.99999833 | 0.500 (29.5) | 30.23 | 6.08 | collapsed |
| | | | dn | 0.99999702 | 0.875 (43.0) | 28.50 | 6.45 | collapsed |
| | | | proj | 0.99999833 | 0.375 (29.3) | 13.28 | 5.19 | collapsed |
| 896 x 896 | 3168 | t33 m=16 kk=32 Mblk=3168 x1 T=7 | gu | 0.99999834 | 0.500 (29.5) | 49.57 | 5.03 | collapsed |
| | | | dn | 0.99999704 | 0.875 (43.0) | 41.46 | 6.01 | collapsed |
| | | | proj | 0.99999834 | 0.375 (29.3) | 18.89 | 4.95 | collapsed |
| 1024 x 1024 | 4128 | t33 m=48 kk=32 Mblk=4128 x1 T=3 | gu | 0.99999835 | 0.500 (29.5) | 61.28 | 5.30 | collapsed |
| | | | dn | 0.99999704 | 0.875 (43.0) | 58.91 | 5.51 | collapsed |
| | | | proj | 0.99999835 | 0.375 (29.3) | 24.81 | 4.91 | collapsed |
| 512 x 768 | 1568 | xd m=32 kk=32 Mblk=800 **x2 dispatches** (pad 32) | gu | 0.99999832 | 0.4375 (29.5) | 18.88 | 6.53 | collapsed |
| | | | dn | 0.99999703 | 0.875 (43.0) | 18.67 | 6.61 | collapsed |
| | | | proj | 0.99999832 | 0.375 (29.3) | 9.08 | 5.09 | collapsed |

Each GEMM row is ONE dispatch of one image (the t33 sizes) or TWO dispatches
(the multi-dispatch row, whose second dispatch carries the 32 zero-pad rows);
`ms / dispatch` is the median of the device's own wall clock, so `TFLOP/s` is
the real array rate for the full M.  The plan's `dispatch_ms` is an upper bound,
as designed: 40.3 ms predicted vs 30.2 measured at 768², 71.8 vs 61.3 at 1024².

All controls collapsed (`neg_collapsed` true for every row: corr ≈ 0 against
the correct reference).  `exact_frac` ≈ 0.90 at every size is the normal bf16
accumulation-order difference (1–4 ulps), not a layout error: the per-tile
diagnostic of §7.2 shows every one of the 86 M-tiles at corr ≥ 0.999997 with a
per-element max error ≤ 0.5.

---

## 9. WHAT STILL CANNOT BE PLANNED

| gap | why | what it would take |
|---|---|---|
| **the attention (ip) image above SEQ=304** | dense attention needs the whole key axis in L1; the key-blocked + online-softmax kernel is designed but ungated | the T22 campaign-1 kernel + its four gates.  Until then the chunked path runs attention on the host in f32, and the plan covers only the linear half |
| **the product loop dispatch** | `s2_ffn32.ffn32_run` is still `NotImplementedError` (INTEGRATION-SPEC §8): the images are built, validated and cached, but the per-layer loop does not yet chain gu/dn/proj | wiring the plan's images into `t18_run` behind `ZIMAGE_ADAPT=1` — a host-glue job, not a planning job |
| **the pure-BFP16 MMA path** (`mixed=False`) | no verified A packer at kk>8 (`verify_t33.pack_a_t33` raises for bfp16 A) | an A encoder + a C decoder for the 9/8-byte path; measured gain 8–16 % |
| **kk > 32** | measured but DMA-gated (5.59 vs 5.12 TFLOP/s) | a DMA-side fix, not a tile-geometry one |
| **m < 16** | `mm_bfp.cc:303` `static_assert(m % 16)` | a different micro-kernel |
| **depth 2 FIFOs** | measured +13.5 %, but halves the largest legal n; the planner keeps depth 1 so the feasible set is maximal | a cost-model revision once the product can afford the extra L1 |
| **shapes above 2048 per side** | the product envelope (VAE/text-encoder path), not the planner | a different product decision |

---

## 10. FILES TOUCHED

**New**

| file | what |
|---|---|
| `webui/npu_plan.py` | the planner (pure) + `bo_sizes()` + selftest |
| `imaggers/util/place/npu_synth.py` | plan → VARIANT → build → plan-keyed manifest |
| `imaggers/util/adaptive/npu_dev.py` | the device gate: pack → dispatch → unpack → corr + negative control |
| `imaggers/util/adaptive/validate.py` | the validation table |
| `imaggers/util/adaptive/_diag_1024.py` | per-tile / per-n-tile error localiser (§7.2) |
| `ADAPTIVE-NPU-REPORT.md` | this file |

**Modified (all additive, all behind a flag or a pure derivation)**

| file | change |
|---|---|
| `webui/shapes.py` | `chunk_gate` now consults the planner, the BO coverage and (opt-in) the build; new `refusal()`, `npu_plan_for()`, `built_plan_keys()`, `plan_manifest()`, `plan_summary()`; `resolve()` carries `npu_plan*`; `chunk_reason` names the plan |
| `webui/build_shape.py` | `build_adaptive()` + `--adaptive`; `_no_build()` names the plan; `ADAPT` flag routes an unbuilt M |
| `imaggers/image/t18_run.py` | the four M-dependent BO literals are derived (§3); `refresh_bigm_sizes()` called from `configure()` |
| `imaggers/util/place/build_sweep.py` | `--gen <json>` merges generated VARIANTs (opt-in, collision-refusing) |

**Artefacts** — `webui/shape_manifests/plan_*.json` (16 plan manifests),
`imaggers/util/place/out/ad_*.{xclbin,bin}` (48 images),
`imaggers/util/place/out/grid_build.json`.  `.prj` dirs were deleted
after every build; `kernels_main.json` was kept (184 of them).

---

## 11. REPRODUCE

```powershell
# 1. the planner's own checks (the six measured L1 points + the delivered 512²)
py312 webui/npu_plan.py --selftest

# 2. plan + build one size (all three GEMM images), then the manifest
py312 imaggers/util/place/npu_synth.py --hw 384 384

# 3. the whole validation grid
py312 imaggers/util/place/npu_synth.py --grid --json out/grid_build.json
py310 imaggers/util/adaptive/validate.py --md out/VALIDATION.md

# 4. the device gate (ONE launcher; it refuses to start next to any python.exe)
py310 imaggers/util/adaptive/npu_dev.py --hw 1024 1024 --gemm gu dn proj
```
