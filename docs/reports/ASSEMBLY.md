# DiT block assembly plan — what to build and in what order

Written so that the moment the four parallel pieces land, assembly is
mechanical rather than exploratory.  Every step names the kernel it uses, the
shape it sees, and the thing that can go wrong.

Z-Image-Turbo geometry: hidden 3840, heads 30, head_dim 128, ffn inner 10240,
sequence 96 tokens (128x128 image), 34 transformer blocks
(`noise_refiner` + `context_refiner` + 15 shards x 2).

## The judging budget (measured by the oracle — use this to accept/reject)

`imaggers/oracle/` computes the block twice: `dit_block_float` (f64 truth) and
`dit_block_device` (the dtype model of the AIE assembly).  Measured at the full
Z-Image geometry (3840/30/128/10240/96), conservative defaults:

    max abs err                        3.202e-01
    rel to output absmax               5.363e-02      <-- the acceptance number
    rms                                4.374e-02
    output absmax                      5.972

**So the NPU assembly is in tolerance when `rel_to_absmax <= 5.4e-02`** against
the float oracle.  That number must be RAISED for two errors the device model
cannot capture:

  * softmax's AIE2P `exp2` error — 2.94e-2 rel (`activation.py:400-407`)
  * `swiglu`'s `vtanh` error — 0.08 rel / 0.35 abs (`activation.py:144-148`);
    removable by building `swiglu(use_lut=True)` (exact table model, 1 ulp)

Sensitivity, measured: the dominant term is the **per-call bf16 C re-rounding**
(one call per matmul gives 1.980e-02 vs 5.363e-02 with a finer schedule).  The
B codec's floor-vs-conv_even choice is minor (5.363e-02 vs 5.179e-02).

The oracle's device model was cross-checked against measured device behaviour:
per-call bf16 accumulation reproduces `imaggers/dit2/FINDINGS.md:80-95`
(bit-exact at hidden 32..1792), and its bfp16ebs8 quantizer is bit-exact vs
`pack_bfp16._quantise` in half-away mode.

Wall time at full geometry: float oracle 0.49 s, device model 17.9 s per block
(that is Python/BLAS modelling overhead, NOT NPU time).  ~34 GFLOP per block.

## The 9-step block, and where each piece comes from

| step | op | kernel | shape | source |
|---|---|---|---|---|
| 1 | weighted RMSNorm | `rms_norm_weighted` | (96, 3840) | `imaggers/kernels/` |
| 2 | adaLN modulate | `adaln_modulate` | (96, 3840) | `imaggers/kernels/` |
| 3 | corner turn + QKV GEMM | DMA `to_stream` walk + `mm_bfp(mixed=True)` | (96,3840) -> (96,3*3840) | `imaggers/memtile/` |
| 4 | RoPE on Q,K | `rope` (`datamovement.py:317`) — **NOT `two_halves`** | (96,30,128) | existing |
| 5 | attention | QK^T -> `softmax_rows_bf16` -> P*V | (30,96,128) | `imaggers/attn_impl/` |
| 6 | output proj + residual | `mm_bfp` + `add_sized` | (96,3840) | existing |
| 7 | weighted RMSNorm | `rms_norm_weighted` | (96, 3840) | `imaggers/kernels/` |
| 8 | adaLN modulate | `adaln_modulate` | (96, 3840) | `imaggers/kernels/` |
| 9 | FFN | `mm_bfp` 3840->10240 x2, `swiglu`, `mm_bfp` 10240->3840 | | existing |
| 10 | residual | `add_sized` | (96,3840) | existing |

(adaLN produces scale+shift for both the attention branch and the FFN branch
from the same conditioning vector, so steps 2 and 8 are the same kernel with
different slices of the modulation tensor.)

## RoPE: the INTERLEAVED convention, not rotate-half

`two_halves` selects HuggingFace rotate-half (`datamovement.py:322-323`,
`rope.cc` symbol `rope_two_halves`, pairs `(k, k+d/2)`).  **Z-Image does not use
that.**  Its RoPE is the interleaved convention — adjacent pairs `(2k, 2k+1)` —
proven in `zimage-bfp16-repo/rope_real.py:39-61` against the installed diffusers
source, with a negative control in `test_rope_parity.py` showing rotate-half is
wrong (error 6.288 vs 4.8e-7).

So the kernel to call is **`rope` WITHOUT `two_halves`** (`rope.cc:29-39`,
`x_even`/`x_odd` pairs).  Precision is determinable: products stay in the
accumulator with one bf16 rounding at the store, and cos/sin arrive as a bf16
LUT.

## THE ALIGNMENT TRAP — silent wrong numbers, no error

AIE2P vector loads/stores **silently floor misaligned addresses to 64 B**
(`aie_api/detail/ld_st.hpp:33-41` for 512/1024-bit vectors; "automatic HW
truncation", `:74-76`).  It produced wrong numbers with no error and cost real
time to find (the f32 adaLN read at 8192 when asked for 4108).

**Rule for every buffer and every row handed to a vector loop: the start address
must be a multiple of 64 B.**  The kernel wrappers now declare
`contract.alignments = ((i, 64), ...)`, so a misaligned `memref.view` raises at
design time instead of silently misreading.

## What the two new kernels measured

Both run on the NPU and verify at the full block geometry (96 x 3840):

| kernel | stage | M x H | max abs err | max ulp | verdict |
|---|---|---|---|---|---|
| `rms_norm_weighted` | f32 | 8x1027 | 4.77e-07 | 0.0 | PASS |
| `adaln_modulate` | f32 | 8x1027 | 1.43e-06 | 0.0 | PASS |
| `rms_norm_weighted` | bf16 | 96x3840 | 3.125e-02 | **1.00** | PASS |
| `adaln_modulate` | bf16 | 96x3840 | **0.0** | **0.00** | PASS (bit-identical) |

Tolerance declared up front: 2 bf16 ulps, `ulp = 2**(exp-7)`, floor `2**-126`.
Two runs per xclbin, byte-identical.  The f32 geometry H=1027 exercises the
scalar tails.

Kernel arithmetic:
  * `rms_norm_weighted` = `rms_norm_aie2p.h:114-170` plus one `x*gamma`
    multiply per 64-lane chunk and tail (`flm_gemma4/rms_norm.h:53-68`
    pattern), one bf16 rounding at the store
  * `adaln_modulate` = `acc=x; mac(acc,x,s); add(acc,b)` over `scale_shift.cc`'s
    scaffolding, one bf16 rounding, and **`1+scale` is never materialised**

Two more constraints that shape the block design:
  * `npu2` `core_dma_channels_in == 2` — a Worker can be fed by **at most 2
    input ObjectFifos**.  adaLN packs `[scale|pad|shift]` per token into ONE
    i8-arena stream element to stay within it.  Any block design with >2 input
    streams must pack or time-multiplex.
  * `aie::mul` pairs operands of ONE class (`mul.hpp:50-91`) — never bf16 x f32.
    `rms_norm_weighted` widens the `inv_rms` limbs to f32 to scale the exact
    f32 `x*gamma` product.

## QK-norm runs BEFORE RoPE, not after — order corrected from source

`export_dit.py:410-431` gives the real sequence:

    to_q / to_k / to_v  ->  unflatten  ->  norm_q / norm_k  ->  RoPE  ->  attention

so QK-norm is applied AFTER the projections and **BEFORE** RoPE.  The earlier
note in this file had it between RoPE and QK^T; that was wrong.

The attention implementation handles this with a new `qk_norm.cc` (weighted
RMSNorm, `y = x*w*rsqrt(mean(x^2)+1e-5)`, one conv_even store) on Q rows
`(m, hd)` and K rows `(seq, hd)`.

Two things that make it cheap:
* **`1/sqrt(head_dim)` folds into the norm WEIGHT** — RMSNorm is
  scale-invariant, so `norm(Q)*(scale*w) == scale*norm(Q)*w`.  No extra pass.
* the norm weights ride an i8-arena prefix of the inB K/V objects via
  `memref.view`, because a compute tile has only **2 input DMA channels**
  (`"tile (0,2) requires 4 input/1 output DMA channels, but only 2 input/2
  output available"`).

## Attention: verified at the full Z-Image geometry

    heads=30, head_dim=128, seq=96, m=8
    compile 13.2 s (13.9 s with qk_norm), 755 ms/run (30 heads x 12 row blocks)
    ERT_CMD_STATE_COMPLETED, run-to-run byte-identical
    368,640 / 368,640 outputs inside the derived budget (worst 0.126)
    max abs err 0.238 vs absmax 8.14  = 2.9% of range

Sharpest stage check: **P*V against the device's own measured P is 100.00%
bit-exact, max ulp 0.00** on every config — the whole non-softmax pipeline is
bit-exact to the model.

**The only error source is the shipped softmax's hardware `exp2`/`inv`**:
measured via an identity-V run (where O/2 equals the device P) at 4.6-7.8e-2
relative on significant P entries, median ~1.8e-2.  Rows normalise to within
0.3%.  That matches the budget item already flagged by the oracle.

Non-causality is confirmed live: with identity V, the strictly-upper-triangular
P entries are non-zero (min 2.5e-11 / 3.9e-10 / 9.0e-12) and rows sum to
0.997-1.004.

**Design choice to know about:** the GEMMs are scalar `mm` (bf16 -> f32, with
`-DB_COL_MAJ` on QK^T so K stays row-major for QK-norm), deliberately not
`mm_bfp`/vectorised mm, whose blocked layouts would need corner-turn DMA walks
and a memtile hop.  755 ms per 30 heads is scalar-GEMM-bound — roughly 10x
headroom if the block wants the `mm_bfp` route of `attn/DESIGN.md` later.

**Another trap:** `matmul_scalar_bf16_f32`'s scalar bfloat16 multiply rounds
EACH PRODUCT toward -inf (the core boots in floor and nothing sets conv_even
before the GEMM), then accumulates in f32.  Seen at 81% bit-exact / <=1 f32 ulp
on raw-f32 dumps.  Call `conv_even` before the GEMMs to remove it.

Also: `aie::bfloat16` does not exist; it is the global `::bfloat16` (`__bf16`).

The compiler's own note names the fixes:

    note: this is the device's total ShimNOCTile DMA budget, not a placement
    choice -- no column has spare capacity to pin to; fan traffic through a
    MemTile, or reduce the number of DMA-attached ObjectFIFOs feeding this tile

So the two sanctioned routes are:
  1. **fan the traffic through a MemTile** — put the extra stream on a memtile
     hop, which is exactly what the corner turn already does for `outA`
  2. **reduce the DMA-attached ObjectFIFOs** — merge A and SB into ONE stream,
     since the norm core already re-sends each row once per chunk pass and can
     carry `[row | params]` interleaved

Option 2 keeps the host interface at the two streams the device allows and is
the better default.  Plan every later stage (attention, FFN) around at most TWO
host-fed input streams plus one output.

## HARD LIMIT: the SHIM tile has only 2 input DMA channels

Adding the adaLN stream as a third host input is refused at placement:

    error: no ShimNOCTile on the device has 0 input/1 output DMA channel(s)
           free: all 1 ShimNOCTile(s) are at 0/2 input, 2/2 output channels used

So a design cannot have three host-fed input streams (A, B, SB) plus an output.
`core_dma_channels_in == 2` applied to the COMPUTE tile; this is the SHIM
limit, and it bites before the compute one does.

Ways out, in order of least disruption:
  1. **time-multiplex** A and SB onto one shim channel — the norm core already
     re-sends each row once per chunk pass, so a single stream carrying
     `[row | params]` interleaved is expressible
  2. merge B into the same stream if its schedule aligns
  3. use a second shim tile if the device exposes one (the error says
     "all 1 ShimNOCTile(s)", so probably not)

This is the same class of constraint as `core_dma_channels_in == 2` and will
recur for every stage that adds a host-fed tensor.  Plan the block's host
interface around at most TWO input streams from the start.

## STEP 1 VERIFIED ON THE NPU — the task-compliant chain

    [static] ALL PASS: True
    [prep] N_ref absmax 4.2812   Y absmax 6.1250   ref_model absmax 25344.0
    2/2 runs ERT_CMD_STATE_COMPLETED (5.92 / 6.38 ms)
    [verify] vs per-call-bf16 ref_model: max abs err 1.28e+02  max ulp 1.000
             bit-exact 99.90% of 1024 elements
    [verify] refs.check_within_ulps(<= 2 bf16 ulps): True
    [verify] PASS (<= 2 bf16 ulps vs numpy): True

The chain is `rms_norm_weighted` (gamma != 1) -> `adaln_modulate` (per-token
scale/shift) -> `mm_bfp(mixed=True)` (bf16 activations, BFP16 weights) at
M=32 K=3840 N=32, tile (16,8,16).  Gamma is folded into a packed
`[scale | shift | gamma]` stream element so Worker A keeps exactly two input
ObjectFifos.

Two things made this work:
  * `set_current_device(NPU2())` — `from_name("npu2")` exposes only ONE shim
    tile to the placer and refuses the 3-fill design; `NPU2()` shows the
    per-column shim tiles and it compiles.  See `KNOWN-ISSUES.md`.
  * gamma packed into the params stream rather than given its own channel.

Artifact: `imaggers/block/out/s1w_M32_K3840_N32_m16_k8_n16_d111.xclbin`
(21,304 B) + `...bin` (900 B).

**Next stage** is `ASSEMBLY.md` step 2: + RoPE + QK-norm + QK^T + softmax +
P*V + to_out, using `imaggers/attn_impl/` (which already verifies at the full
Z-Image geometry, heads 30 / head_dim 128 / seq 96).

## The 34-block weight census is COMPLETE

`imaggers/residency/block_census.py` enumerates every block's tensors from
`weights.bfp16`.  34 blocks, 506 tensors, **0 ungrouped**:

    layers.*            15 tensors/block   203,489,280 packed B   180,910,336 params
    context_refiner.*   13 tensors/block   199,065,600 packed B   176,962,816 params
    noise_refiner.*     15 tensors/block   203,489,280 packed B   180,910,336 params
    TOTAL                            506   6,909,788,160 packed   6,143,056,384
                                                  + 4,089,856 raw fp32

Per block (`layers.0`, 15 tensors):

    feed_forward.w1 / w2 / w3        [10240,3840] [3840,10240] [10240,3840]   44,236,800 B each  tr=32
    attention.to_q / to_k / to_v / to_out.0   [3840,3840]                     16,588,800 B each  tr=48
    adaLN_modulation.0.weight        [15360,256]                               4,423,680 B       tr=64
    adaLN_modulation.0.bias          [15360]                                     61,440 B
    attention_norm1/2, ffn_norm1/2   [3840]                                     15,360 B each
    attention.norm_q                 [128]                                         512 B

So the load list for every block is fully determined and the 34-block assembly
has no unknowns on the weight side.  Note `context_refiner.*` has 13 tensors,
not 15 — it lacks two of the norms.

## Coordinate before writing in `imaggers/block/` — two agents collided here

Two subagents were writing the same files at the same time and one was stopped.
The one that continued has the plan worth keeping:

  * **the task requires `rms_norm_weighted`**, not the unweighted `rms_norm`
    that `imaggers/block/step1_design.py` uses.  The earlier attempt deferred it
    on the grounds that `rms_norm_weighted` would need gamma as a third input
    stream and `core_dma_channels_in == 2` forbids that.
  * **the fix is to fold gamma into the packed stream element** — extend the
    `kernels/` idiom from `[scale | shift]` to `[scale | shift | gamma]` so
    Worker A still has exactly two input fifos: `inA` (x rows) and `inSBG`
    (packed per-token params).
  * L1 at tile (0,2) then reads 7,680 + 23,040 + 30,720 + 3,840 (stack) =
    55,280 of 65,536 B — it fits.

`imaggers/block/step1_norm_adaln_gemm.py` is the deliverable name for the
weighted design; `step1_design.py` / `step1_build.py` / `step1_prep.py` /
`step1_verify.py` were the earlier unweighted attempt and are kept for
comparison only.

## ISOLATED: the base design is exact, the adaLN insertion is the fault

Control run of the UNMODIFIED `imaggers/memtile2/memtile_worker.py` at the same
geometry (M=32, K=3840, N=32, tile 16/8/16, chunked producer):

    [static] ALL PASS: True
    [verify] vs per-call-bf16 model: max abs err 0.0000  max ulp 0.000
    [verify] bit-exact vs ref_model: 100.00% of 1024 elements
    [verify] PASS (ref_model 1-ulp criterion): True

So the architecture, the corner turn, the segmented producer and the host ABI
are all correct at this exact geometry.  The garbage output comes from the
adaLN stage that was inserted into `imaggers/block/step1_design.py`.

Search space is now the ~20 inserted lines:

    norm(in_row, in_row, K)
    adaln(in_row, s_row, b_row, in_row, K)     # suspect: x/y aliasing vs `restrict`
    src = memref.view(chunk_memref_ty, elem_in, src_off, [])
    dst = memref.view(chunk_memref_ty, elem_out, constant(r*Kc*2), [])

The base works WITHOUT any of that, so each line is a candidate: the `inSB`
element views, the aliasing of `in_row`, and the `src_off`/`dst` offsets.
A/B each one against the base rather than guessing.

## The failure is a MISALIGNMENT, not a small error

Raw output decoded as bf16 at M=32, N=32:

    1024 elements: 482 NaN/Inf, 542 finite
    finite absmax 3.2e38   absmin 7.5e17   455 distinct values
    first 32 f32: nan, 1.86e19, -inf, 3.55e20, nan, 2.21e19, ...

Values of magnitude 1e17..1e20 with exponents all over the map are what you get
when a bf16 vector is read from the wrong byte offset — the exponent field
lands on mantissa bits.  This is NOT a precision problem and not a small
reference mismatch.

The most likely mechanism is the AIE2P 64 B floor: a vector load from a
non-multiple-of-64 address is silently read from the floored address
(`aie_api/detail/ld_st.hpp:33-41`, "automatic HW truncation").  The
`require_64B_alignment` contract covers the kernel's buffer ARGUMENTS, but the
`memref.view` offsets inside `norm_core` are a separate thing — `sb_off` is
7,680 which is a 64 B multiple, but `src_off` (the dynamic chunk offset
`c * Kc * 2` with `Kc=960`, so 1,920 bytes per step) is only 64 B aligned
because `Kc*2 % 64 == 0`.  Verify every `memref.view` offset in the loop is a
64 B multiple AT RUNTIME, including the `r * Kc * 2` destination offset.

Test to run: the same design with `Kc` chosen so every offset is a 64 B
multiple (e.g. `Kc=1024` -> 2,048 B steps), and a ramp-filled A so a misread is
immediately visible.

## The aliasing question is NOT settled — test it, do not assume

`step1_design.py` currently contains a comment asserting the in-place call is
safe because the kernel does load -> acc -> store per vector:

    norm(in_row, in_row, K)
    adaln(in_row, s_row, b_row, in_row, K)     # comment claims y aliases x safely

but `adaln_modulate.cc:43` marks both `x` and `y` `restrict`, which forbids
that.  **Both claims can be true in practice** — the loop may genuinely be
element-safe while the `restrict` promise still lets the optimiser do something
unexpected.  Do not assume either way; settle it with a measurement:

  * run once with the in-place call and once with a separate output row
    (the norm core's L1 has room: 30,720 B chunk + 2 x 7,680 B rows < 64 KB)
  * if the outputs differ, `restrict` is binding and the separate buffer is
    required; if they match, the in-place form is fine and the bug is elsewhere

That single A/B test decides it.  Everything else about the call is confirmed
correct against `designs.make_adaln_design`: arity 5, `sb_off` a BYTE offset,
`sb_elem_ty`, `SB_ty`, `sb_tap`.

## ROOT CAUSE CONFIRMED: `restrict` forbids the aliasing

`imaggers/kernels/adaln_modulate.cc:43-52`:

    void adaln_modulate_rows(const T *restrict x, const T *restrict scale,
                             const T *restrict shift, T *restrict y,
                             int32_t cols)
    auto pX = ::aie::begin_restrict_vector<vec_factor>(x);
    ...
    auto pY = ::aie::begin_restrict_vector<vec_factor>(y);

`restrict` on both `x` and `y` means the compiler assumes they never overlap.
The assembled design passes the SAME `in_row` for both, which is undefined
behaviour — that is the repeating `-2` pattern and the NaN/Inf.

**The fix:** give `adaln` its own output object.  Options, in order of least
disruption:
  1. keep `norm(in_row, in_row, K)` in place (that kernel IS alias-safe, per
     `rms_norm_aie2p.h`) and call `adaln(in_row, s_row, b_row, OUT_row, K)`
     where `OUT_row` is a separate L1 row buffer, then copy from `OUT_row`
     into the chunk object
  2. or write the modulated row straight into the chunk object's row slot and
     skip the copy

The norm core's L1 at K=3840 is 30,720 B for the chunk + 7,680 B for one row;
one more 7,680 B row fits comfortably inside the 64 KB tile budget.

## The adaLN call form is RIGHT — the aliasing is the suspect

`designs.make_adaln_design` (lines 148-215) is the authoritative reference and
its core loop is

    x = of_x.acquire(1);  sb = of_sb.acquire(1);  y = of_y.acquire(1)
    s = memref.view(row_memref_ty, sb, constant(0), [])        # offset 0
    b = memref.view(row_memref_ty, sb, constant(sb_off), [])   # sb_off is BYTES
    k(x, s, b, y, H)                                           # 5 args, H IS the 5th

so `adaln(in_row, s_row, b_row, in_row, K)` has the right ARITY and the right
offsets (`sb_off=7680` bytes at K=3840).  What differs is the aliasing: the
reference passes **separate** `x` and `y` objects, while the assembled design
passes `in_row` for both.  `adaln_modulate` computes `y = x*(1+scale)+shift` and
must read `x` before writing `y`; if the kernel is not alias-safe, in-place
produces the repeating pattern and NaN/Inf seen in the output.

The fix is to give `adaln` its own output object (the chunk copy already
targets `elem_out`, so the natural place is to write the modulated row into
`elem_out` directly and let the chunk copy run from there), or to confirm from
`adaln_modulate.cc` that the loop reads before it writes and keep in-place.

Also matching the reference exactly: `sb_elem_ty = np.ndarray[(sb_row_elems *
elem_size,), np.int8]`, `SB_ty = np.ndarray[(M * sb_row_elems,), np_dt]`,
`sb_tap = TensorAccessPattern.full((M*sb_row_elems,)).tile((sb_row_elems,))`.

## STEP 1 WITH THE CORRECT adaLN LAYOUT: runs, still wrong numbers

    sb_row_layout: off=7680 pad=0 row_elems=7680     (K=3840, bf16)
    A 245,760 B   SB 491,520 B   B 245,760 B   C 2,048 B
    2/2 ERT_CMD_STATE_COMPLETED (4.77 / 4.18 ms), run-to-run byte-identical
    decoded C absmax NaN, ref absmax 11.4375          <- still garbage

The packed `[scale | pad | shift]` stream is now wired correctly
(`sb_row_layout` gives `pad=0` at K=3840 so the shift half already starts on a
64 B boundary), the design compiles to `step1.xclbin` 17,752 B, and the NPU
runs it deterministically.  The output is still wrong.

What the output looks like: a repeating pattern of small negative bf16 values
(`81ff 80ff 81ff 807f 81ff 817f ...`, i.e. about -2) plus NaN/Inf, against a
reference absmax of 11.4.  A constant repeating pattern suggests the kernel is
reading one address repeatedly rather than walking the row.

Likely suspects, in order:
  1. the `memref.view` offsets for `s_row` / `b_row` inside `norm_core` — the
     split is `sb_off` bytes, and `sb_off` is a BYTE offset, not an element
     index
  2. the `sb_elem_ty` element width: `sb_row_elems * 2` bytes is right for
     bf16, but the view type must agree with what `adaln_modulate`'s
     `KernelContract` declares
  3. `adaln`'s argument list — `designs.make_adaln_design` is 2-in/1-out
     (`inX`, `inSB`), while `kernel_wrappers.adaln_modulate` is an
     `ExternalFunction`; the call `adaln(in_row, s_row, b_row, in_row, K)` has
     five arguments and `K` is probably not one of them

**Proven and unaffected:** the three-kernel chain compiles and executes on the
NPU with byte-identical results across runs.  The defect is in the data the
adaLN kernel reads, not in the architecture.

## STEP 1 WITH THE CORRECT adaLN LAYOUT COMPILES

    SPECIALIZE OK  CallableDesign
    COMPILE OK  step1.xclbin  17,752 B   (+272 B over the broken version)
                step1.bin       900 B

The packed `[scale | pad | shift]` layout is now wired through `inSB`
(`sb_elem_ty = np.ndarray[(sb_row_elems*2,), np.int8]`, `sb_row_elems` from
`sb_row_layout(K, is_f32=False)`), with `s_row` / `b_row` carved out of one
element inside `norm_core`.  The host side declares

    SB_ty = np.ndarray[(M * sb_row_elems,), np.dtype[bfloat16]]
    sb_host_tap(M, sb_row_elems) = TensorAccessPattern.full((M*sb_row_elems,))
                                   .tile((sb_row_elems,))
    sequence(a, b, sb, c, inA_h, inB_h, inSB_h, outC_h)

so the runtime order is A, B, SB, C.

Remaining: run on the NPU and verify.  The prep must emit `SB` as
`(M * sb_row_elems)` bf16 elements, not a plain `(M, K)` scale row.

## TWO adaLN implementations exist — pick one deliberately

`imaggers/kernels/` ships two wrappers for the same kernel and their input
counts differ:

| wrapper | inputs | source |
|---|---|---|
| `designs.make_adaln_design` | **2** — `inX` (rows), `inSB` (packed `[scale\|shift]`) -> `outY` | `designs.py:6`, `:148` |
| `kernel_wrappers.adaln_modulate` | an `ExternalFunction` + `KernelContract`; check its contract before calling | `kernel_wrappers.py:171` |

`designs.make_adaln_design` is the one that already solves the
`core_dma_channels_in == 2` problem by packing scale and shift into ONE stream
element via `sb_row_layout`.  A design that carves `s_row` / `b_row` out of one
packed element and calls a 3-argument form is a DIFFERENT contract — verify
which the compiled kernel actually expects before assuming either.

The call that is definitely wrong (it produced 475/1024 NaN/Inf):

    adaln(in_row, s_row, s_row, in_row, K)

because `K` is not a positional argument and both scale and shift pointed at the
same buffer.

## STEP 1 RUNS BUT COMPUTES GARBAGE — the adaLN input layout is wrong

    2/2 ERT_CMD_STATE_COMPLETED (3.87 / 3.60 ms), run-to-run byte-identical
    decoded C: 475 of 1024 elements NaN/Inf, finite absmax 3.3e38   <- garbage

The three-kernel chain compiles and runs, but `adaln_modulate` is being fed the
wrong bytes.  `kernels/designs.py` defines `sb_row_layout` — the packed
`[scale|pad|shift]` element that `core_dma_channels_in == 2` requires — and the
call must use it.  `step1_design.py` currently passes a plain K-element row:

    adaln(in_row, s_row, s_row, in_row, K)      # WRONG

It needs the packed layout from `sb_row_layout`, with `scale` and `shift` as
distinct slices, and the correct argument list for the `KernelContract`
(three inputs plus the output; `K` is not a positional argument).

**The exact fix** (from `imaggers/kernels/designs.py`):

    from designs import sb_row_layout
    sb_off, pad_elems, sb_row_elems = sb_row_layout(K, is_f32=False)
    sb_raw_ty = np.ndarray[(sb_row_elems * 2,), np.dtype[np.uint8]]

then
  * `inS = ObjectFifo(sb_raw_ty, name="inS", depth=1)` — NOT `row_raw_ty`
  * inside `norm_core`, after `norm(in_row, in_row, K)`:

        sb_row = memref.view(sb_memref_ty, sb_elem, constant(0, index=True), [])
        adaln(in_row, sb_row, in_row)

    with `sb_memref_ty = np_ndarray_type_to_memref_type(sb_raw_ty)`
  * host tensor type is `sb_raw_ty`; `s_tap` walks one whole packed element per
    token: `TensorAccessPattern.full((M, sb_row_elems*2)).tile((1, sb_row_elems*2))`

`designs.py:148 make_adaln_design` is the authoritative 2-in / 1-out reference.
Note the file can be overwritten by a parallel agent — re-apply this if the
`sb_raw_ty`/`sb_memref_ty` names disappear.

**What is nevertheless proven by this run:** the whole chain
`rms_norm -> adaln_modulate -> mm_bfp(mixed)` compiles to an xclbin (17,480 B)
and executes on the NPU with deterministic, byte-identical results across runs.
The remaining defect is a data-layout one, not an architectural one.

## STEP 1 COMPLETE: norm -> adaLN -> GEMM compiles to an xclbin

    SPECIALIZE OK  CallableDesign
    COMPILE OK  imaggers/block/out/step1.xclbin   17,480 B
                imaggers/block/out/step1.bin         900 B

Three kernels in one design — `rms_norm` -> `adaln_modulate` ->
`mm_bfp(mixed=True)` — with the segmented producer and the memtile corner turn
in between.  `imaggers/block/step1_design.py` is the modified memtile2 design;
`imaggers/block/step1_build.py` builds it.

The invocation that works (this is the part that was wrong before):

    spec = S.memtile_worker.specialize(
        M=..., K=..., N=..., m=..., k=..., n=...,
        inA_depth=2, mid_depth=1, producer="chunked", Kc=0,
        use_cache=False)
    xcl, ins = spec.compile(xclbin_path=..., inst_path=...)

NOT `memtile_worker(...)` — that returns a design whose `Runtime` still expects
the four host tensors and raises "expects 4 tensor argument(s), but received 0".

Still to do for step 1: run it on the NPU and verify against
`imaggers/kernels/refs.py` within 2 bf16 ulps.

## STEP 1 STATUS: adaLN insertion COMPILES

The modified `imaggers/block/step1_design.py` (memtile2 + `adaln_modulate`
inserted between the norm and the segmented producer) builds and runs the full
aiecc pipeline through to `final.xclbin`.  The only remaining error is at
host-invocation time:

    RuntimeError: Design 'memtile_worker' expects 4 tensor argument(s),
                  but received 0

which is expected — the design now declares four host tensors
(`A_ty, S_ty, B_ty, C_ty`) and the test harness called it with none.  Supply
them the way `memtile2/run.py` does and the compile completes.

Environment facts that had to be re-discovered on the way:
* the object cache dir gets POISONED by the sandbox's `os.mkdir(path, 0o700)`
  — once a `aie-symbol-map-*` staging dir exists, `_cleanup_failed_compilation`
  cannot remove it and every later fetch fails.  Use a FRESH
  `NPU_CACHE_HOME` per attempt, plus the `os.mkdir` shim (0o700 -> 0o777).
* `xclbinutil` must be on PATH or aiecc dies at `final.xclbin` with
  `ShellCommand: tool 'xclbinutil' not found in search paths or PATH`.
* the design must `set_current_device(NPU2())` before building, or `mm_bfp`
  raises `NotImplementedError: bfp16ebs8 is an AIE2P type; select an NPU2
  device`.

## Where adaLN goes in the base design (exact call form)

`imaggers/kernels/kernel_wrappers.py` exports `adaln_modulate(tile_size=1024,
*, cols=None)` as an `ExternalFunction` with a `KernelContract`; it consumes
`(x, scale, shift)`.  In `imaggers/memtile2/memtile_worker.py` the insertion
is:

    line ~292   norm_kernel = require_64B_alignment(kernels.rms_norm(tile_size=K), ...)
    line ~305   inA  = ObjectFifo(row_raw_ty, name="inA", depth=inA_depth)
    line ~309   outA = ObjectFifo(...)            <- the segmented producer's input
    line ~374   norm(in_row, in_row, K)           <- INSERT adaln AFTER THIS
    line ~388   worker_a = Worker(norm_core, [inA.cons(), outA.prod(), norm_kernel], ...)

The change is:

  * `norm_core(of_a, of_s, of_out, norm, adaln)`
  * `[inA.cons(), inS.cons(), outA.prod(), norm_kernel, adaln_kernel]`
  * after `norm(in_row, in_row, K)`, call `adaln(in_row, scale_row, shift_row,
    in_row, K)`

`inS` is the SECOND input ObjectFifo — allowed, because `core_dma_channels_in
== 2`.  `kernels/` packs `[scale|pad|shift]` into one stream element, so it is
one fifo rather than two.

The HOST side needs the matching change (`step1_design.py` lines ~452-489):

    A_rows  = a_host_tap(M, K, m)          # TensorAccessPattern.full((M,K)).tile((1,K))
    b_tap   = b_host_tap(N, K, k, n)
    C_tiles = c_host_tap(M, N, m, n)

    def sequence(a, b, c, inA_h, inB_h, outC_h):
        ...
        inA_h.fill(a, tap=a_tap, group=tgs[-1])
        inB_h.fill(b, tap=b_tap, group=tgs[-1])
        outC_h.drain(c, tap=c_tap, group=tgs[-1], wait=True)

becomes

    def sequence(a, s, b, c, inA_h, inS_h, inB_h, outC_h):
        ...
        inA_h.fill(a, tap=a_tap, group=tgs[-1])
        inS_h.fill(s, tap=s_tap, group=tgs[-1])     # one [scale|pad|shift] row per token
        inB_h.fill(b, tap=b_tap, group=tgs[-1])
        outC_h.drain(c, tap=c_tap, group=tgs[-1], wait=True)

and `Runtime(...)` gains the `S_ty` host buffer.  `s_tap` is
`TensorAccessPattern.full((M, S_elem)).tile((1, S_elem))` where `S_elem` is the
packed `[scale|pad|shift]` element width that `kernels/` defines
(`sb_row_layout` in `imaggers/kernels/designs.py`).

Note the base design uses the UNWEIGHTED `kernels.rms_norm`; the real block
needs `rms_norm_weighted` (which is `x/rms(x)*gamma`).  Switching is optional
for step 1 and required by step 3.

## STAGE 1 COMPLETE — hidden 3840 with a real producer, 100% bit-exact

    K=3840, M=N=32, tile (16,8,16), depths inA=2 mid=1
    compile 7.6 s warm / 151 s cold
    2/2 runs ERT_CMD_STATE_COMPLETED, run-to-run byte-identical
    100.00% bit-exact vs the per-call-bf16 numpy model, max ulp distance 0.000

**`delegate_tile` is DEAD on npu2** — measured, both with separate Tiles and
with one shared Tile:

    error: 'aie.objectfifo.allocate' op objectfifo has no shared memory access
           to delegate tile's memory module

A compute core cannot write a memtile-resident pool, so the norm core can never
produce a whole `(m,K)` object.  (`ObjectFifoAllocateOp` documents it: "Only
works if all the objectfifo's tiles ... have shared memory access to the
specified tile's memory module".)

**The fix is to segment the producer BD into COLUMN chunks.**  At K=3840 use
`Kc=960` (4 chunks):

* the corner walk decomposes over contiguous `kk` ranges — walking each
  `(m,Kc)` chunk with `corner_turn_tap(m,Kc)` at the memtile and concatenating
  the streams in chunk order **is** `corner_turn_tap(m,K)` exactly
* the GEMM consumes slices in `kk` order, so chunk boundaries are invisible
* the producer BD drops from 30,720 words to **7,680** (compute cap 16383)
* the norm core's L1 holds `(m,Kc)` = 30,720 B plus one 7,680 B row, instead
  of 122,880 B
* `rms_norm` runs in place on each full K row (alias-safe) before the chunk
  copy; rows are re-sent per chunk pass via the fill tap `.repeat(n_chunks)`

`imaggers/memtile2/memtile_worker.py` has this as the default producer, with
the plain and `delegate_tile` modes kept only to capture their refusals
verbatim.

## STAGE 1 IS PROVEN — hidden 3840 corner turn compiles and runs

`imaggers/memtile/probe_cap.py`: host -> `(m,K)=(16,3840)` i8 object
(122,880 B = 30,720 32-bit words) -> 1:1 link at memtile (0,1) ->
`to_stream = corner_turn_tap` -> `consumer_obj_type=(m*k*2,) i8` -> worker
drains slices.

    xclbin compiles (5.3 s), 2/2 runs ERT_CMD_STATE_COMPLETED,
    run-to-run byte-identical, corner turn 100% bit-exact vs numpy at K=3840

**Two earlier assumptions were wrong:**
* the memtile DMA BD cap is **NOT** 16383 words — 30,720-word BDs at the
  memtile verify and run.  The 16383 cap is a **compute-tile** BD limit
* a fifo **cannot** join AND feed a forward hop
  (`'aie.objectfifo' op objectfifo cannot be in more than one ObjectFifoLinkOp`),
  so the walk cannot ride a hop after a join.  A join output's transfers are
  per-input-segment anyway, so the walk can never live on a join output.

**Net architecture for the block** (measured, not inferred):

    norm core  ->  outA ((m,K) i8; the corner walk lives on the 1:1 link's
                   output at the memtile)  ->  GEMM core acquiring (m,k) slices

with the `(m,K)` buffer resident at the memtile via the link, and a
`delegate_tile` option for the producer-side pool.  `consumer_obj_type` hands
the GEMM core ready 256-B `(m,k)` slice objects (`memref<256xi8>`).

## WHERE THIS STANDS — every stage has a verified piece

| stage | state | evidence |
|---|---|---|
| 1. one GEMM at hidden 3840 | **PROVEN** | `imaggers/memtile/probe_cap.py`, 100% bit-exact at K=3840 |
| 2. norm -> adaLN -> GEMM | **PROVEN as parts** | `imaggers/kernels/` verify at 96x3840 (adaLN bit-identical, rmsw 1 ulp) |
| 3. attention in isolation | **PROVEN** | `imaggers/attn_impl/`, full Z-Image geometry, 100% of outputs in budget |
| 4. the whole block | **next** | the oracle at `imaggers/oracle/` is the judge (rel_to_absmax <= 5.4e-02) |
| 5. two blocks chained | next | weight residency is measured here |
| 6. 34 blocks + the loop | next | `PIPELINE.md` |
| 7. VAE on CPU -> first image | next | `KNOWN-ISSUES.md` scope note |

What remains for stage 4 is **assembly**, not invention: every kernel exists,
the corner turn works, the weights are extracted and byte-exact against the
vendor, and the oracle tells you when to accept.

The one gap is the memtile PRODUCER side (`imaggers/memtile2/`, in progress):
the norm core cannot hold a `(m,K)` object at K=3840 because its L1 is 64 KB.
The consumer side and the corner turn are already proven.

1. **One GEMM at hidden 3840.** `imaggers/memtile/` is doing this.  Until it
   passes, nothing above it is worth assembling.  It is the load-bearing piece.
2. **Norm -> adaLN -> GEMM** (steps 1-3).  This is `imaggers/dit_block_v1.py`
   plus the two new kernels plus the memtile path.  Verify against numpy with
   the bf16 round-trip model (`mm_bfp_mixed` re-rounds C after every call).
3. **Attention in isolation** (step 5), at 1 head / small head_dim first, then
   30 heads / head_dim 128.  `imaggers/attn_impl/` is doing this.
4. **The whole block** (1-10), one seed, one token count.  Compare against a
   numpy reference of the block built from the checkpoint's fp32 weights.
5. **Two blocks chained.**  This is where the residency question is answered:
   weights must live in BFP16 and stay there between calls.
6. **All 34 blocks + the denoising loop.**  At this point the only thing on the
   CPU is the sampler arithmetic.
7. **VAE on the CPU** for the first image.  Port it later.

## Traps, restated because they cost real time

* `mm_bfp_mixed.cc` re-rounds C to bf16 after EVERY GEMM call.  A numpy
  reference that accumulates in f64 will drift up to 13 ulps at K=1792 and look
  like a kernel bug.  Round-trip through bf16 per accumulation.
* Compute-tile DMA descriptors accept at most 3 dimensions.  Memtiles accept 4.
* bfp16ebs8 strides must be 4-byte multiples; 9-byte blocks need repacking into
  `(jj, kk, row)` order.
* L1 is 64 KB per compute tile.  A memtile is 512 KB.  A single DMA descriptor
  caps at 16383 32-bit words.
* `os.mkdir(path, 0o700)` produces a directory the sandbox token cannot
  re-enter, and `tempfile` uses exactly that mode.  Copy the `os.mkdir` shim
  and `use_cache=False` from `imaggers/dit_block_v1_run.py`.  This is not an ACL
  problem (`diagnose-windows-sandbox-acl` returned NOT_THIS_CLASS).
* `xclbinutil` is off-PATH; add
  `D:\Project\deepseek_harness\test7\WStmp\driver376\npu_mcdm_stack_prod`.
* Host encoding is `bfp.shuffle(bfp.encode(v), w, h, tw, th)` with `(tw,th)` the
  PER-CORE TILE, not the whole matrix.  Get this wrong and the numbers are
  plausible but wrong, with no error.
* `mm_bfp_mixed.cc` static_asserts m%16==0 and n%16==0 even though the Python
  factory only checks %8.

## Real tensor names in the checkpoint (measured — do not re-guess)

From the actual transformer shard (`layers.0.*`), the block owns **15 tensors**:

    attention.norm_q.weight      attention.norm_k.weight       <- QK-norm!
    attention.to_q.weight         attention.to_k.weight        attention.to_v.weight
    attention.to_out.0.weight                                    <- note the .0. infix
    feed_forward.w1.weight        feed_forward.w2.weight       feed_forward.w3.weight
    attention_norm1.weight        attention_norm2.weight
    ffn_norm1.weight              ffn_norm2.weight
    adaLN_modulation.0.weight  [15360, 256]   adaLN_modulation.0.bias [15360]

Corrections to the table above, which was written from the usual DiT naming
rather than this checkpoint:

  * there is **no** `ff.net.0.proj` fused tensor — `w1` and `w3` are separate
    SwiGLU tensors, so the FFN is three GEMMs, not two
  * there is **no** `norm1.linear` / `norm2.linear` — the adaLN projection is
    a single `adaLN_modulation.0` producing all of (scale, shift) for both
    branches, 15360 = 4 x 3840
  * `to_out` carries a `.0.` infix
  * **`attention.norm_q` / `attention.norm_k` exist** — Q and K are
    normalised BEFORE the dot product.  `export_dit.py:410-421` pins the
    order: `to_q`/`to_k` projection first, then `norm_q`/`norm_k`.  Add that
    to the attention path; it is another `rms_norm_weighted`-shaped call on
    (96, head_dim) rows per head.  `shard_lib.py:368/:389` shows where RoPE's
    cos/sin enter relative to the projections — read it rather than inferring.
  * **`adaLN_modulation.0` is `Linear(256, 15360)`** — the modulation is
    projected from a **256-dim conditioning vector** (the timestep/label
    embedding), not from `x`.  15360 = 4 x 3840 = (scale1, shift1, scale2,
    shift2), so one GEMM per block per step feeds both branches.  This makes
    step 2/8 a per-token affine with tensors of shape (96, 3840) each, and the
    projection itself is a cheap 256->15360 GEMM.
  * the norms around each branch are `attention_norm1/2` and `ffn_norm1/2`,
    not `norm1/norm2`

The point of the whole exercise: one `weights.bfp16` file, loaded once, kept as
`bfp16ebs8` blocks in memory and on the device, with no fp32 expansion and no
per-call unpack.  Step 5 is where that gets tested: two blocks chained, and
measure that the weights are read, not re-decoded.
