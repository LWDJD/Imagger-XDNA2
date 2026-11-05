# DESIGN — weighted RMSNorm + adaLN modulation for the Z-Image DiT block on XDNA2

Scope: the two missing building blocks of a Z-Image DiT block

```
x = x + attn( adaLN_modulate(norm1(x), cond) )       # scale/shift
x = x + ff(   adaLN_modulate(norm2(x), cond) )
```

with `adaLN_modulate(x, s, b) = x * (1 + s) + b`, per-token `(96, 3840)`
scale/shift (hidden 3840, heads 30, head_dim 128, ffn inner 10240, seq 96).
Everything below is derived from files read in this workspace; paths are
relative to the workspace root (`_pylibs/...` = the installed mlir-aie tree,
`zimage-bfp16-npu/...` = this project).  Where a fact could not be
established from those files it is marked **"not determinable from the
sources read"** (collected at the end).

Companion numpy references + byte contracts (runnable, self-testing):
`_scratch/adaln/refs.py`.

What exists (baseline, per the completed coverage survey and re-verified
here):

* `rms_norm` / `rms_norm_eps` / `layer_norm` are **unweighted** — no gamma or
  beta argument at all (`_pylibs/mlir_aie/python/aie/iron/kernels/norm.py:85,
  :114, :139`; the C++ entry points take `(input, output, cols[, epsilon])`,
  `norm/rms_norm_aie2p.h:174-191`).  `layer_norm_aie2p.h:63`: "gamma = 1 and
  beta = 0 here, so the affine pair is not applied at all".
* Weighted RMSNorm arithmetic exists only inside the Gemma4 decode helper
  `template <int D> rms_norm(y, x, w)` — `y = x * rsqrt(mean(x^2)+eps) * w`
  (`flm_gemma4/rms_norm.h:53-68`), reached only through geometry-locked
  factories (`flm_gemma4.py:2157`, row type `(geometry.model_dim,)` at
  `flm_gemma4.py:2166`, model_dim 1536/2560 at `flm_gemma4.py:903/:919`; the
  factories "build only those" geometries, `flm_gemma4.py:854-857`).  The
  header itself says "Independent of the model geometry" (`rms_norm.h:8`) —
  the *arithmetic* is reusable as a pattern; no factory exposes it at 3840.
* `affine_cast` (`datamovement.py:211`) is per-**column** `gamma*x+beta`:
  gamma/beta arrive packed in one `2*cols` buffer as a baked tensor `Param`
  (`datamovement.py:237, :245-246`), the same columns applied to every row
  (`datamovement.py:68-72`), f32 in / bf16 out (`datamovement.py:236-238`).
  `layer_norm_affine_cast` (`transformer.py:145-173`) is the same static
  gamma/beta fused into a LayerNorm.  Neither is per-token conditioned.
* `add`/`mul`/`mul_add`/`add_sized`/`mul_sized` (`eltwise.py:191/:210/:229/
  :295/:322`).  `mul_add` is `a*b` **or** `a+b` per call selected by `is_mul`
  (`eltwise.py:230`; C++ `scale_shift.cc:17-27`) — **not** a fused `a*b+c`.
* `mm_bfp(dim_m, dim_k, dim_n, mixed=True)` (`linalg.py:920`): bf16 A/C,
  bfp16ebs8 B (`linalg.py:957-961`); C++ `mm_bfp_mixed.cc` static_asserts
  `m % (2*r) == 0`, `k % s == 0`, `n % (2*t) == 0` with `r=s=t=8`
  (`mm_bfp_mixed.cc:118-128`), i.e. **m % 16 == 0, k % 8 == 0, n % 16 == 0**
  — stricter than the factory's "multiple of 8" (`linalg.py:950-954`).

---

## a. Weighted RMSNorm from EXISTING kernels: `rms_norm` then `mul` by gamma?

**Yes, functionally — two library kernels, no new C++ — with one extra bf16
rounding.**

Semantics: `rms_norm_ref` is `x / sqrt(mean(x**2) + eps)` (`norm.py:167-171`)
and `mul_ref` is `a * b` (`eltwise.py:56-58`), so `mul(rms_norm(x), gamma)`
is exactly `x / rms(x) * gamma`.  `refs.rms_norm_weighted_two_pass_ref` models
the device chain (bf16 stored between the two calls) against the fused
`rms_norm_weighted_ref`; the self-test bounds the difference at ~3 half-ulps.

**Tiling at hidden 3840.**

* `rms_norm` runs **one whole row per call**: `tile_size`/`cols` may be any
  positive length ("Positive row lengths need not be vector-aligned: the
  kernel handles scalar tails", `norm.py:88-89`), and the factory takes
  `tile_size=3840` with no 1024 restriction (`norm.py:85-91`, `_row_size`
  `norm.py:72-82`).  On the XDNA2 (aie2p) tuned path the kernel walks
  `kLanes = 64`-element chunks (`rms_norm_aie2p.h:77-79, :118-121`):
  **3840 = 60 × 64**, no tail; the portable path uses 32-lane vectors
  (`rms_norm_aie2p.h:178-180`), 120 per row, tail handled
  (`rms_norm_aie2p.h:84-110`).  96 calls per (96, 3840) block.
* The gamma `mul` against the **fixed 1024-element** tiles: **3840 = 3 × 1024
  + 768 — does NOT fit.**  `mul`/`add`/`mul_add` are hard-wired to 1024
  (`eltwise.py:27`, `_require_fixed_tile_size` `_common.py:507-515`; C++
  `eltwise_mul_bf16_vector` compiled at N=1024, `mul.cc:137-139`), and a 4th
  call would run its 1024-wide loop over the 768-element remainder.  Three
  escapes:
  1. `mul_sized(tile_size=3840)` (`eltwise.py:322-346`, symbol
     `eltwise_mul_bf16_vector_size`, `mul.cc:141-144`, `-DMUL_ELEMS=3840`,
     `eltwise.py:335`): **one call per row**, 3840 = 120 whole 32-lane
     vectors (loop bound `MUL_ELEMS / vec_factor`, `mul.cc:86-91`), scalar
     tail machinery present but unused (`mul.cc:115-127`).
  2. Flat tiling over the whole `(96, 3840)` stream: 368 640 elements =
     **360 exact 1024-tiles** (elementwise ops don't care about row
     boundaries) — but gamma is per-**column**, so each tile needs its
     gamma segment with wraparound at every 3840 elements; present gamma
     replicated per token as a `(96, 3840)` host buffer (737 280 B) and the
     flat pairing is exact.
  3. `mul_sized` sub-row tiles (e.g. 3 × 1280 per row).
* gamma must be **bf16**: `mul` raises for anything else ("Only the bf16
  variant is available", `eltwise.py:76-80`); an f32 gamma needs
  `convert_copy` first (`datamovement.py:167-208`).

Cost: 192 calls per block (96 norm + 96 `mul_sized`) or 456 (96 + 360 flat);
the intermediate `x̂ = x/rms` is materialized as bf16 between the calls, so
two roundings instead of one.

## b. adaLN from EXISTING kernels: `mul` then `add`?  Is `mul_add` enough?

**Two passes are the minimum with existing kernels; `mul_add` alone is NOT
enough — and the `1 +` forces a third pass (or a host-side prep).**

* `mul_add` computes `c = a*b` **or** `c = a+b`, chosen per call by `is_mul`
  (`eltwise.py:230`; the if/else in `scale_shift.cc:17-27`, dispatch
  `scale_shift.cc:108-115`).  Its own docstring names the two-phase
  pattern: `A * B` with `is_mul = 1` then `+ C` with `is_mul = 0` on the same
  workers (`eltwise.py:232-235`).  So **yes, 2 calls per tile minimum** —
  `a*b+c` in one call does not exist anywhere in `aie_kernels` (the only
  fused affine is `saxpy`, `z = a*x + y` with a **scalar** a,
  `datamovement.py:42-48, :123-164`; `axpy.cc:49` broadcasts `aL`).
* The `1 + scale` is a real operation: `y = x*(1+s)+b` needs `g = 1+s` as a
  tile before the mul pass.  `mm_bfp` has no bias operand to fold it into
  (its args are A, B, C only, `linalg.py:986-989`), so with raw on-device
  scale it costs one `add_sized` pass against a ones tile: **3 passes
  total** (`g = ones + s`; `t = x*g`; `y = t + b`).  If scale arrives from
  the host, `1+s` is free; if scale is a GEMM output on device it is not.
* Tiling for `(96, 3840)`: identical arithmetic to (a) — the fixed 1024 tile
  does not divide a 3840 row (3840 = 3×1024+768), while the **flat stream
  divides exactly (360 tiles of 1024)** and `*_sized(3840)` gives 96 calls
  per pass.  Per-pass call counts: 96 (row tiles) or 360 (flat) → **288 or
  1080 calls per modulate** (3 passes) versus 96 for a fused kernel.
* Intermediates: `g` and `t` are bf16 tiles (7 680 B each per row call,
  737 280 B if the whole block is materialized between passes), and the
  chain rounds bf16 **three times** (g, t, y) where the fused form rounds
  once (`refs.adaln_two_pass_ref` vs `refs.adaln_modulate_ref`; measured
  ≤ 3.4 half-ulps of `|x(1+s)| + |shift|`, which can be far larger than an
  ulp of `y` when `x(1+s)` and `shift` cancel).

## c. Smallest NEW C++ kernel for each

**(1) `rms_norm_weighted(x, gamma, y, cols, epsilon)` — closest template:
`rms_norm_aie2p.h:114-170` (`rms_norm_aie2p`), with the weighted multiply
pattern of `flm_gemma4/rms_norm.h:54-68`.**  Changes over the shipped
kernel, all in the normalize (second) loop:

* one extra `const bfloat16 *gamma` parameter on the `extern "C"` entry
  (`rms_norm_aie2p.h:174-182` gains a sibling symbol);
* per chunk: load gamma and form `xg = x * gamma` in f32, then apply the
  existing `s_hi + s_lo = inv_rms` mac pair to `xg` instead of `x`
  (the loop body at `rms_norm_aie2p.h:156-162`); `flm_gemma4/rms_norm.h:59-63`
  already shows the exact `aie::mul(x, w)`-then-scale order;
* the tail path likewise (`rms_norm_aie2p.h:164-168`, gamma tail alongside
  the input tail);
* optionally an `rms_norm_weighted` factory beside `norm.py:85` with args
  `[tile, tile, tile, np.int32, np.float32]` (cf. `rms_norm_eps`'s
  `[tile, tile, np.int32, np.float32]`, `norm.py:118-121`).

Net: **~10–15 lines of C++**.  The sum-of-squares pass
(`rms_norm_aie2p.h:123-138`) is unchanged, and the result has ONE bf16
rounding (f32 accumulators throughout, `rms_norm_aie2p.h:143-159`).

**(2) `adaln_modulate(x, scale, shift, y, cols)` — closest template:
`aie_kernels/eltwise/scale_shift.cc`** (the file is literally the
scale→shift two-phase building block, `eltwise.py:232-235`).  Changes:

* make the loop **tri-input** and fused — the body becomes the
  `acc-from-vector + mac` idiom of `eltwise_vadd_mac` (`add.cc:39-51`) plus
  one more accumulate:

  ```c++
  // y = x * (1 + s) + b, one f32 accumulator, one bf16 rounding on store
  auto X = *pX++;  auto S = *pS++;  auto B = *pB++;
  aie::accum<accfloat, vec_factor> acc;
  acc.from_vector(X);                       // + x
  acc = aie::mac(acc, X, S);                // + x*s
  acc = aie::add(acc, B);                   // + shift   (cf. layer_norm_aie2p.h:39)
  *pC++ = acc.template to_vector<bfloat16>();
  ```

* drop the `is_mul` branch (`scale_shift.cc:17-27, :108-115`); the pointer
  walking/vector scaffolding is `scale_shift.cc:50-62` as-is;
* optionally `-DADALN_ELEMS` compile-time count like `MUL_ELEMS`
  (`mul.cc:18-21`) plus a scalar tail like `mul.cc:121-126`.

Net: **~20 lines of C++**.  The `1 +` is free in the accumulator (it is
`acc = x` before the mac), which is exactly what the three-pass composition
cannot express.  One bf16 rounding matches the library's eltwise tolerance
(`_BF16_ROUNDTRIP`, `eltwise.py:43-48`).

`affine_cast` is **not** the right template for either: it is per-column
static gamma/beta (`datamovement.py:245-246`) and f32→bf16
(`datamovement.py:236-238`); adaLN is per-token bf16→bf16.

## d. Does the GEMM's layout constraint apply?  Where does the corner turn go?

**Yes — and the corner turn must sit exactly once, on the boundary between
the row-major world (norm / adaLN / residual adds) and the GEMM's block
world.**  It cannot live inside `rms_norm`, and it need not be a separate
kernel.

* `mm_bfp(mixed=True)` reads A in the **(r, s, t) = (8, 8, 8) micro-tile
  block layout** (`mm_bfp_mixed.cc:15-22` walks `colA` consecutive 64-lane
  micro-tiles per row-block, `:57-65`; `r = s = t = 8`, `:118-120`).  The
  contract declares A's layout as the DMA permutation
  `TensorAccessPattern.full((m, k)).tile((r, s))` (`linalg.py:404`, bound at
  `linalg.py:973`), and `mm_stream_dims` warns "a plain row-major stream
  yields wrong numbers, not an error" (`linalg.py:391-393`).  C is drained
  with the **inverse** walk (`linalg.py:410-411`, bound at
  `linalg.py:979-984`).
* In that layout a logical hidden row is contiguous **only at k == 8** (one
  micro-tile row); at k = 3840 each row is scattered over 480 segments of 8
  (established in `_scratch/FINDINGS.md:77-89` and asserted in
  `_scratch/dit_block_v1.py:10-20`, whose k == 8 design is the degenerate
  case where the block layout *is* row-major).  So `rms_norm` (one contiguous
  row per call, `norm.py:85-91`) and the adaLN kernels **must see row-major
  rows**, while the GEMM must see blocks.
* Consequently the corner turn happens **at the DMA hop into each GEMM's A
  operand and out of its C** — `to_stream` with `streams.A`
  (`TensorLayout.stream` is exactly "the DMA transform a design applies on
  the hop that feeds this operand", `_common.py:47-50`), which the `dit_block_v1`
  pipeline already uses on the C side (`dit_block_v1.py:125-126`).  The
  two-worker variant (row-wise core → fifo with the permutation → GEMM core)
  is spelled out at `FINDINGS.md:92-93`.  There is **no in-core corner-turn
  kernel in the library** for bf16 (`FINDINGS.md:94-96`: `transpose` only
  moves 8×8 blocks in place, `datamovement.py:103-117`;
  `mm_bfp_shuffle` permutes 9-byte bfp blocks only, `linalg.py:1009-1018`).
* The one *optional* optimization: since adaLN is elementwise, the new
  `adaln_modulate` kernel can perform the corner turn **in its store loop**
  (write each 64-element `(8, 8)` output block contiguously while reading 8
  contiguous 8-element row runs) and hand the GEMM a block-layout A with a
  plain stream — eliminating one DMA permutation.  That changes the call
  granularity from one row to at least one 8-row × k strip (per-call buffer
  shapes change accordingly); whether the strip fits a core's object fifos
  is **not determinable from the sources read** (no memory-size figure was
  found in the sources read).

For the block geometry: `m = 96`, `k = 3840`, `n = 3840 | 10240` all satisfy
the kernel's asserts — 96 % 16 == 0, 3840 % 8 == 0 (`mm_bfp_mixed.cc:126-128`)
— and the norm row 3840 needs no padding on either vector path (section a).

---

## Evidence index (file:line)

| Fact | Where |
|---|---|
| rms_norm / rms_norm_eps / layer_norm factories, unweighted | `norm.py:85-111, :114-136, :139-164` |
| "gamma = 1 and beta = 0 … the affine pair is not applied at all" | `layer_norm_aie2p.h:63` |
| rms_norm C++ symbols `(input, output, cols[, epsilon])` | `rms_norm_aie2p.h:173-191` |
| norm row length arbitrary; scalar tails | `norm.py:88-89`; `rms_norm_aie2p.h:84-110` |
| aie2p norm chunks 64-lane (tuned) / 32 (portable) | `rms_norm_aie2p.h:77-79, :118-121, :178-180` |
| f32 sum-of-squares + f32 scalar inv_rms, single store rounding | `rms_norm_aie2p.h:123-142, :143-159` |
| weighted RMSNorm arithmetic exists only as Gemma4 `rms_norm<D>(y,x,w)` | `flm_gemma4/rms_norm.h:53-68` |
| Gemma4 factory geometry-locked (model_dim 1536/2560) | `flm_gemma4.py:2157, :2166, :903, :919, :854-857` |
| affine_cast per-column gamma/beta baked Param, f32→bf16 | `datamovement.py:211-260` (esp. `:236-238, :245-246`), ref `:61-72` |
| layer_norm_affine_cast: static gamma/beta fused into LayerNorm | `transformer.py:145-173` |
| add/mul fixed 1024 (C++ N=1024) | `eltwise.py:27, :191-226`; `mul.cc:137-139`, `add.cc:143-145`; `_common.py:507-515` |
| mul_add is a*b OR a+b per call (is_mul), two-phase scale_shift pattern | `eltwise.py:229-235, :261-264`; `scale_shift.cc:17-27, :108-115` |
| add_sized / mul_sized: compiled-in count, scalar tails | `eltwise.py:295-319, :322-346`; `mul.cc:18-21, :115-127` |
| only fused affine is scalar-a saxpy | `datamovement.py:42-48, :123-164`; `axpy.cc:49` |
| bf16-only eltwise ("Only the bf16 variant is available") | `eltwise.py:76-80` |
| convert_copy f32→bf16 (for an f32 gamma) | `datamovement.py:167-208` |
| one-ulp eltwise tolerance for a fused kernel | `eltwise.py:33-48` |
| mm_bfp factory; mixed = bf16 A/C + bfp16ebs8 B | `linalg.py:920-1006` (esp. `:928-930, :957-961`) |
| mm_bfp_mixed asserts m%16, k%8, n%16 | `mm_bfp_mixed.cc:118-128` |
| factory checks looser (multiples of 8) | `linalg.py:950-954` |
| A layout = to_stream `tile((r,s))`; C = inverse; "wrong numbers, not an error" | `linalg.py:380-412` (esp. `:391-393, :404, :410-411`), `:973, :979-984` |
| TensorLayout.stream = DMA transform on the feeding hop | `_common.py:41-53` |
| (8,8,8) micro-tile; A walks 64-lane micro-tiles | `linalg.py:916`; `mm_bfp_mixed.cc:20-22, :57-65` |
| rows contiguous only at k == 8; corner turn gap | `FINDINGS.md:77-99`; `dit_block_v1.py:10-25, :114-126` |
| no in-core corner-turn kernel in library | `FINDINGS.md:94-96` |
| AIE_BF16_LANES 32 on aie2p (XDNA2) | `aie_arch.h:27-35` |
| add loop idiom acc-from-vector + mac | `add.cc:39-51`; `scale_shift.cc:32-48` |
| bf16 = 2 bytes LE, 1/8/7 fields | `noncausal_softmax.py:65-66` |

## Left undetermined (from the sources read)

1. **Z-Image's RMSNorm epsilon** — the library default is 1e-5
   (`norm.py:86`, `rms_norm_aie2p.h:15`), the Gemma4 helper uses 1e-6
   (`flm_gemma4/rms_norm.h:27`); Z-Image's own value: not determinable from
   the sources read (parameterized in `refs.py`).
2. **Core (L1) memory budget** — whether one AIE2P core can hold the
   per-call buffers (weighted norm: 23 040 B; fused adaLN: 30 720 B; plus
   the two-pass `g`/`t` intermediates) or an 8-row corner-turn strip: no
   memory-size figure appears in the sources read — not determinable.
3. **Whether the Z-Image block also gates attn/FF outputs** (some DiTs use
   `x + gate*attn(...)` with a third modulation head): the task formula
   shows scale/shift only; the model's actual heads: not determinable from
   the sources read.
4. **Where the cond → (scale, shift) projection runs and its dtype** — one
   `(96, 2×3840)` GEMM vs two, on-device vs host, bf16 vs bfp16 weights:
   not determinable from the sources read (`mm_bfp` can produce the tiles as
   bf16 C, `linalg.py:959-960`).
5. **gamma dtype in the Z-Image checkpoint** (bf16 vs f32) — the kernel
   designs above assume bf16 (`flm_gemma4/rms_norm.h:55-60` pattern);
   `convert_copy` covers f32 (`datamovement.py:167-208`): not determinable
   which the model needs.
6. **Per-row vs per-call buffer presentation in the Iron dataflow** —
   whether a Worker can feed `mul_sized`/`adaln_modulate` individual rows of
   a (96, 3840) tile per call (cf. the same open question for softmax rows,
   `attn/DESIGN.md:226-228`): not determinable from the sources read.
7. **Measured accuracy/tolerance for the two new fused kernels on npu2** —
   the refs give the arithmetic model and expected roundings (1-ulp style,
   `norm.py:34-43`, `eltwise.py:43-48`), but nothing was run on hardware in
   this task by scope: not determinable.
8. **Corner-turn DMA feasibility at k = 3840** — `to_stream` permutations
   are declared (`linalg.py:404`) but the only assembled pipeline in this
   project exercised k == 8 (`dit_block_v1.py:83-90`); a (96, 3840) A
   permutation was not built: not determinable from the sources read.
