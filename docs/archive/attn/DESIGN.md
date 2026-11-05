# DESIGN — non-causal (bidirectional) attention softmax for Z-Image on XDNA2

Scope: one missing kernel for the Z-Image DiT attention block
(hidden 3840, heads 30, head_dim 128, seq 96).  Everything below is derived
from files read in this workspace; paths are relative to the workspace root.
Where a fact could not be established from those files it is marked
**"not determinable from the sources read"** (collected at the end).

The gap: `mha_softmax` (``_pylibs/mlir_aie/python/aie/iron/kernels/linalg.py:1200``)
is causal — its reference keeps `cols <= rows` on the diagonal block
(`linalg.py:1276-1277`) and skips key blocks past the query block
(`linalg.py:1278`).  A diffusion DiT needs every token to attend to every
other token.

Companion numpy reference + byte contract:
`_scratch/attn/noncausal_softmax.py` (runnable, self-testing).

---

## a. Can the EXISTING unmasked `softmax` (`activation.py:376`) be reused?

**What it is.**  `softmax(tile_size)` binds `softmax_bf16`
(`_pylibs/mlir_aie/python/aie/iron/kernels/activation.py:376-391`), which is
`softmax_simple_bf16` from
`_pylibs/mlir_aie/include/aie_kernels/activation/softmax_aie2p.h:20-115`
(dispatched by `activation/softmax.cc:10-14`: `AIE_TUNED_AIE2` → aie2 header,
else **softmax_aie2p.h — the XDNA2 path**).  It is a genuinely **unmasked**
3-pass softmax (max → exp → normalize) over **one flat vector** of
`input_size` elements: no mask, no causal logic, no cross-row state anywhere
in `softmax_simple_bf16`.  Constraints:

* `tile_size` must be a **positive multiple of 32** — `activation.py:386`
  calls `_require_vector_alignment(..., _RUNTIME_VECTOR_WIDTH)` with
  `_RUNTIME_VECTOR_WIDTH = 32` (`activation.py:36`); the check is
  `_common.py:518-533`.  The C++ step matches: `SM_VEC_LEN 32`
  (`softmax_aie2p.h:15`) and `elem_iters = vector_size / SM_VEC_LEN`
  (`softmax_aie2p.h:53`) — a tail-free loop, so the multiple is hard.
* The call normalizes over the **whole tile as one row**
  (`softmax_ref` reshapes to `(-1, tile_size)` and softmaxes each,
  `activation.py:1077-1080`; the C++ reduces one flat vector,
  `softmax_aie2p.h:59-103`).  There is no row/stride parameter: each call
  consumes one contiguous flat tile of `tile_size` bf16 values
  (`activation.py:387-391`).
* No `1/sqrt(d)` scale parameter (the only log factor is the hard-coded
  `log2e` broadcast, `softmax_aie2p.h:57`).

**Tiling for 96 tokens × 128 head_dim.**  Per head the softmax input is the
score tile `S = Q·Kᵀ` of shape **(96 queries, 96 keys)**; `head_dim = 128`
never enters the softmax tile — it sizes Q/K/V and constrains only the
matmuls (`mm_bfp` takes any multiple of 8, `linalg.py:950-954`; `mha()`'s
micro-tile wants dim_k a multiple of 8, `linalg.py:1131-1139`).  Two tilings:

1. **One query row per call**: `tile_size = 96` (one row of 96 key scores).
   `96 = 3 × 32`, **fits** the %32 rule exactly.  Cost: 96 calls per head,
   **2880 calls** for 30 heads per DiT block.  Each call must receive its
   row as its own flat (96,) tile = 192 bytes in / 192 bytes out.
2. **One row per call, whole tile in one call** — needs a different symbol:
   the same object file already exports `softmax_rows_bf16(input, output,
   rows, row_len)` (`softmax_aie2p.h:222-229`), a plain loop of
   `softmax_simple_bf16` over `rows` rows of `row_len`.  Bind it with
   `rows=96, row_len=96`: **1 call per head**, row_len 96 satisfies the same
   %32 rule.

**Does it fit?**  Yes, functionally: 96 keys per row is a whole number of
32-lane vectors, the entire 96-key row lives in one call (softmax has no
cross-call state, so a row split across key chunks could not be recombined —
not an issue at 96), and no padding is needed (`96 % 32 == 0`).  Caveats:
(i) the factory binds only the single-vector symbol (`activation.py:388-391`)
so tiling 1 costs 2880 kernel invocations per block unless `softmax_rows_bf16`
is bound (the `fn.object_file.bind` mechanism is documented at
`linalg.py:1112-1117` and used at `fused.py:457`); (ii) fold `1/sqrt(128)`
into Q before QKᵀ (bilinear) or scale S before the call, since the kernel has
no scale parameter; (iii) whether the Iron dataflow can present individual
rows of a (96, 96) tile as per-call buffers is **not determinable from the
sources read**.

## b. If not — the smallest kernel to ADD (derived from `softmax.cc`)

`softmax.cc` itself is a 14-line dispatcher: it carries **no** mask; it
selects `softmax_aie2.h` or `softmax_aie2p.h` (`softmax.cc:10-14`).  In the
XDNA2 header the causal mask is exactly two things:

* **`mask_bf16`, `softmax_aie2p.h:212-219`** — fills `[unmasked_size,
  total_size)` with bf16 `lowest` ("causal / padding mask" per its comment);
* **its single call site, `softmax_aie2p.h:237`**, inside
  `softmax_rows_causal_bf16` (`softmax_aie2p.h:232-241`):
  `mask_bf16(input + r * row_len, row_offset + r + 1, row_len);`
  — row `r` keeps columns `< row_offset + r + 1`.

**The unmasked variant is literally that function with line 237 deleted** —
and it already exists: `softmax_rows_bf16`
(`softmax_aie2p.h:222-229`) is `softmax_rows_causal_bf16` minus the mask
call, a loop of `softmax_simple_bf16` over `rows` rows.  So the **smallest
kernel to ADD on XDNA2 is: none in C++** — only a Python binding of the
existing `softmax_rows_bf16` symbol (arg types `[tile, tile, int32, int32]`,
cf. the `softmax_bf16` binding `[tile, tile, int32]` at
`activation.py:387-391`).  If a fused `1/sqrt(d)` is wanted inside the call
(as `partial_softmax_alias_bf16` has: `scale` parameter at
`softmax_aie2p.h:125-130`, broadcast instead of `log2e` at `:151`), the
smallest *new* C++ is `softmax_rows_bf16` with the hard-coded `log2e`
broadcast (`softmax_aie2p.h:57`) replaced by a `scale` argument — one new
parameter on `softmax_simple_bf16`; otherwise pre-scale S.  (The aie2/XDNA1
header has only `softmax_bf16` and `mask_bf16`, `softmax_aie2.h:93-113` — no
rows variants — but XDNA2 compiles the aie2p header, `softmax.cc:10-14`.)

For completeness, the causal masks of the two attention toolkits (not the
base softmax) are: `mha.cc`'s `diagonal` machinery (`mha.cc:526-532`,
`587-611`, `221-231`) and `flash_attn_prefill.h`'s
`apply_mask_and_get_max` (`flash_attn_prefill.h:88-111`, causal + sliding
window) — the latter is head_dim 256/512 only (`linalg.py:1295`, `:1325-1326`),
so it is out of the running for head_dim 128.

## c. Does `mha`'s `partial_softmax` + `rescale_O` implement online softmax, and can it just drop the mask?

**Yes to the structure.**  `mha.cc` implements the streaming (online)
two-phase softmax completely:

* `init_scale_buffer` zeroes the state: `m_{-1} = lowest`, `l = 0`
  (`mha.cc:645-654`, `init_state_w` `mha.cc:409-423`);
* per key block, `partial_softmax` (`mha.cc:520-643`) computes the running
  max `m_i = max(m_{i-1}, block max)` (`mha.cc:259-262` fast path,
  `softmax_aie2p.h:165-167` alias path), writes **unnormalized**
  `P = exp2(A·s − m_i)` (`mha.cc:304-310`, `softmax_aie2p.h:171-180`), and
  folds the block sum into `l_i = exp2(m_{i-1} − m_i)·l_{i-1} + ΣP_block`
  (`advance_state_w`, `mha.cc:386-407`, called at `:638-640`);
* `matmul_PV` rescales the prior `O` by `exp2(m_{i-1} − m_i)` before
  accumulating `P·V` (`mha.cc:489-497`, `scale_blocked_rows` `:93-109`);
* `rescale_O` inverts the sums and applies `1/l_i` at the end
  (`mha.cc:502-518`, `invert_sums_w` `:425-437`).

**No to "run without the causal mask by simply not applying it."**  The mask
is not a switch you can leave off; it is compiled into one derived value and
three unconditional gates:

1. `diagonal = q_block_idx*B_q − kv_block_idx*B_kv` is derived from `idx`
   (`mha.cc:526-532`) — there is no mask argument;
2. on-diagonal blocks apply the suffix cut: `suffix_mask`
   (`mha.cc:128-137`, applied `:221-231` on the fast path,
   `:587-611` on the slow path);
3. key blocks past the query block are **hard-skipped** in three places:
   `matmul_bf16_bf16_wrapper` (`mha.cc:450-457`), `partial_softmax` itself
   (`mha.cc:533-537`, P zeroed when `diagonal + B_q <= 0`) and `matmul_PV`
   (`mha.cc:478-482`).

However, the row masking is *conditional*: `suffix_mask` keeps everything
when `diagonal >= B_kv` ("a diagonal of VECTOR_LENGTH or more keeps them
all", `mha.cc:128-129`; guards at `mha.cc:221` and `:587`), and all three
gates pass when `idx[0] <= idx[1]` and `idx[1]*B_q + B_q > idx[0]*DIM_K`.
So driving every block pair with a constant **`idx = (kv=0, q=1)`** (giving
`diagonal = B_q >= B_kv − 1`) runs the whole toolkit **unmasked**.  The catch
is that the padding-tail extents are derived from the same `idx` and the
`S_q_eff`/`S_kv_eff` parameters (`mha.cc:540-550`), so each physical block
needs its own `(idx, S_q_eff, S_kv_eff)` encoding to keep tail masking right
(e.g. one 96×96 call: `idx=(0,1)`, `S_q_eff = S_kv_eff = 192` yields the full
unmasked 96×96 with no tails).  Further adaptations: the contract's
`scale` is bound to `log2(e)/sqrt(64)` (`linalg.py:1220`) but Z-Image needs
`log2(e)/sqrt(128)` (arg 4 is runtime-bindable, `linalg.py:1224-1229`); the
factory fixes 64×64 blocks (`_MHA_BLOCK`, `linalg.py:1197`, compile flags
`:1225`) and the fast path requires `B_kv == 64` (`mha.cc:576`,
`VECTOR_LENGTH 64` `mha.cc:25`), so a 96-key block takes the slow per-row
path (row width a multiple of 32, `softmax_aie2p.h:15,53` — 96 qualifies).
The shipped contract reference also hard-codes the causal rule
(`linalg.py:1276-1278`), so it would need a non-causal sibling.

**Bottom line for (c):** the online-softmax two-phase structure is all
there and is causality-agnostic in its math; the causality lives in the
block gates and the derived `diagonal`.  A clean non-causal `partial_softmax`
is a *tiny* change (gate the three skips on a flag and pass
`diagonal >= B_kv`), but "simply not applying" the mask — with the code as
shipped — requires the per-block `(idx, S_eff)` encoding described above.

## d. Recommendation

**Reuse, with a minimal binding — no new C++ on XDNA2.**  At seq 96 the whole
score tile fits: per head, QKᵀ with `mm`/`mm_bfp` (96×128×96; `mm_bfp`
takes dim_k % 8, `linalg.py:950-954`; bf16 emulation of mmul via bfp16 is
supported, `linalg.py:533-538`) → `S` (96, 96) bf16 = 18432 B; fold
`1/sqrt(128)` into Q; then **one call of the already-present unmasked
`softmax_rows_bf16`** (`softmax_aie2p.h:222-229`) with `(rows=96,
row_len=96)` → `P` (96, 96); then P·V with `mm`/`mm_bfp` (96×96×128) → `O`.
This is (a)-answer reuse of the unmasked softmax family via the multi-row
entry point of (b) — the "kernel to add" is only the Python binding
(`fn.object_file.bind`, `linalg.py:1112-1117`) — and it avoids (c)'s mask
surgery entirely.  Fallback if the dataflow cannot tile per row: 96 calls of
the factory-bound `softmax_bf16` at `tile_size = 96` (2880 calls per block —
works, but control-heavy).  Only if the sequence grows past what one score
tile can hold, or block streaming becomes necessary, add the minimal
flag-gated non-causal `partial_softmax` variant of (c) (three gates + a
forced `diagonal`), which buys the online-softmax memory profile at the cost
of the `mha.cc` change and a non-causal contract reference.

---

## Evidence index (file:line)

| Fact | Where |
|---|---|
| `softmax` factory, one flat tile per call, args `[tile, tile, int32]` | `activation.py:376-391` |
| tile_size % 32 rule | `activation.py:36,386`; `_common.py:518-533` |
| softmax normalizes the whole flat tile | `activation.py:1063-1081` (esp. `:1077-1080`); `softmax_aie2p.h:59-103` |
| header dispatch, aie2p on XDNA2 | `softmax.cc:10-14` |
| unmasked multi-row softmax already exists | `softmax_aie2p.h:222-229` |
| causal mask = `mask_bf16` + one call site | `softmax_aie2p.h:212-219`, `:237` (fn `:232-241`) |
| `mask_bf16` fill value bf16 `lowest` | `softmax_aie2p.h:217`; `mha.cc:564-571` |
| plain softmax has no `scale` (log2e hard-coded) | `softmax_aie2p.h:57`; scale param exists only in `partial_softmax_alias_bf16` `:125-130,151` |
| exp2/log2e accuracy note (2.94e-2 rel on npu2) | `activation.py:400-407`; `softmax_aie2p.h:16` |
| `mha_softmax` 64×64, scale `log2(e)/sqrt(64)` | `linalg.py:1197,1220,1225` |
| `mha_softmax` causal semantics (`idx`, `cols<=rows`, kv>q skip) | `linalg.py:1211-1214,1273-1278` |
| `mha` toolkit symbols, bind-via-object-file | `linalg.py:1086-1117` |
| `partial_softmax` online state + P + sums | `mha.cc:520-643`, `:386-407` |
| online rescale of O in `matmul_PV` | `mha.cc:489-497` |
| final `1/l_i` in `rescale_O` | `mha.cc:502-518`, `:425-437` |
| causal diagonal + three hard skips | `mha.cc:526-537`, `:450-457`, `:478-482` |
| suffix mask; unmasked when `diagonal >= B_kv` | `mha.cc:128-137,221-231,587-611` |
| fast path needs `B_kv == 64` | `mha.cc:576`, `:25`; `linalg.py:1197` |
| `prefill_fv` head_dim 256/512 only | `linalg.py:1295,1325-1326` |
| `flash_attn_prefill` causal + window mask | `flash_attn_prefill.h:88-111` |
| `mm_bfp` dims % 8, (8,8,8) micro-tile, bfp16ebs8 | `linalg.py:916,920-1006,950-954` |
| bfp16ebs8 = 9 bytes / 8 values, codec + shuffle | `bfp.py:6-27,35-37,76-224`; `npdtypes.py:29-36` |

## Left undetermined (from the sources read)

1. **Core memory budget** — whether one AIE2P core's object fifos can hold
   the 18432 B S tile + 18432 B P tile (plus Q/K/V staging) for the one-call
   plan: not determinable from the sources read.
2. **Per-row buffer presentation** — whether the Iron dataflow can hand
   `softmax_bf16` one row of a (96, 96) tile per call (strided/offset
   buffers) for the 96-calls fallback: not determinable from the sources read.
3. **Measured accuracy of `softmax_rows_bf16` at (96, 96)** on npu2 — the
   2.94e-2 relative envelope (`activation.py:400-407`) was measured for the
   flat `softmax_bf16`; the rows wrapper shares `softmax_simple_bf16` so the
   same envelope is expected, but no measurement for this shape was found:
   not determinable from the sources read.
4. **`fn.object_file.bind` contract machinery** — the API exists
   (`fused.py:457`, `linalg.py:1524-1527`) but how a `KernelContract`
   (tolerance/reference) attaches to a hand-bound symbol was not fully read:
   not determinable from the sources read.
5. **`mha(dim_m=dim_k=dim_n=96)` viability** — the factory validates
   multiples of 16/8/16 (`linalg.py:1131-1139`) and `partial_softmax`'s slow
   path accepts row widths % 32, but no test of a 96-block toolkit
   configuration was found: not determinable from the sources read.
6. **Layout compatibility of `scale_blocked_rows`** (assumes the 8-row /
   64-column blocked GEMM output, `mha.cc:88-109`) with a `mm_bfp`-produced
   O tile in a mixed design: not determinable from the sources read.
7. **Z-Image's upstream scale convention** — whether `1/sqrt(128)` is
   already folded into the Q projection or must be applied here: not
   determinable from the sources read.
8. **Dispatch cost of 2880 small calls** per 30-head block on the NPU
   runtime: not determinable from the sources read.
