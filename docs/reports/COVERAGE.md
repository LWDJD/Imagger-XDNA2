# mlir-aie IRON kernel coverage for a Z-Image-Turbo DiT block (BlockFP16)

Read-only survey of the mlir-aie tree vendored at `_pylibs/mlir_aie/`
(Python wrappers in `python/aie/iron/kernels/`, C++ sources in
`include/aie_kernels/`). Nothing was compiled or run; every statement below is
from files actually read.

Target geometry: hidden 3840, heads 30, head_dim 128, ffn inner 10240,
sequence 96 tokens. Target: XDNA2 NPU (aie2p / "npu2" — required for
`bfp16ebs8`, which `mm_bfp` refuses on aie2).

## Coverage table

| Operator | Status | file:symbol | Notes / constraints |
|---|---|---|---|
| BFP16 GEMM (`mm_bfp`) | **FOUND** | `_pylibs/mlir_aie/python/aie/iron/kernels/linalg.py:920` `mm_bfp`; shuffle helper `mm_bfp_shuffle` at `linalg.py:1009`. C++: `_pylibs/mlir_aie/include/aie_kernels/linalg/mm_bfp.cc` (`matmul_vectorized_bfp16`), `mm_bfp_mixed.cc` (`matmul_vectorized_different_datatypes`) | `dim_m/dim_k/dim_n` each **positive multiples of 8** (validated `linalg.py:950-954`); aie2p only (`NotImplementedError` otherwise, `linalg.py:946`). `mixed=False`: A,B,C all `v8bfp16ebs8`, pre-shuffled to mmul layout on the host (`aie.utils.bfp`); `mixed=True`: **A bf16, B bfp16ebs8, C bf16** — this is the "BlockFP16 weights" form. C is accumulate (`+=`), zero-init with `kernels.zero`; B is held transposed (N,K). Micro-tile (8,8,8). For Z-Image: 3840/10240/96/128 all satisfy %8. |
| RMSNorm / LayerNorm | **FOUND** (but no learnable weight) | `kernels/norm.py:85` `rms_norm`, `norm.py:114` `rms_norm_eps`, `norm.py:139` `layer_norm`; `kernels/transformer.py:118` `layer_norm_f32`, `transformer.py:145` `layer_norm_affine_cast`. C++: `aie_kernels/norm/rms_norm.cc`, `norm/layer_norm.cc`, `transformer/layer_norm_f32.cc` | One bf16 row per call. `rms_norm`: any positive row length (scalar tails OK); eps fixed 1e-5, or runtime eps with `rms_norm_eps`. **gamma=1 / beta=0 are hard-wired** — `layer_norm_aie2p.h:63` ("the affine pair is not applied at all"); there is no weight argument. `layer_norm`: cols multiple of 16, also gamma=1/beta=0. `layer_norm_affine_cast` takes gamma/beta but as a **baked tensor Param** (static per column) with f32 input. Z-Image's weighted RMSNorm therefore needs `rms_norm` + a separate `mul` by gamma (or `layer_norm_affine_cast`'s pattern). |
| Softmax | **FOUND** | `kernels/activation.py:376` `softmax` (C++ `aie_kernels/activation/softmax.cc`, symbol `softmax_bf16`). Attention-specific online softmax: `kernels/linalg.py:1200` `mha_softmax` (C++ `linalg/mha.cc`, `partial_softmax`) | `softmax`: one row per call, `tile_size` a **positive multiple of 32** (`_RUNTIME_VECTOR_WIDTH = 32`, `activation.py:36`); row of 96 fits (96 = 3×32). Normal (non-causal) softmax over the row. `mha_softmax` is fixed **64×64 blocks** with causal masking and running max state (see attention row). |
| RoPE | **FOUND** | `kernels/datamovement.py:317` `rope` (`two_halves` flag) — C++ `aie_kernels/datamovement/rope.cc`, symbols `rope` (Llama interleave) and `rope_two_halves` (HuggingFace split-half); re-exported at `kernels/transformer.py:35`. Gemma4-specific variants: `kernels/flm_gemma4.py:1873` `flm_gemma4_decode_rope`, `:1985` `flm_gemma4_rope_core` (C++ `aie_kernels/flm_gemma4/decode_rope.cc`, `decode_rope_core.cc`) | ABI `(in, lut, out, dims)` bf16; each input row streams its own interleaved (cos, sin) LUT. Row length a positive **multiple of 16** (interleaved) or **32** (`two_halves`); head_dim 128 fits both. The flm_gemma4 rope is locked to the Gemma4 decode geometry (q/k norm fused in) — not the generic path. |
| SwiGLU / SiLU / GELU | **FOUND** | `kernels/activation.py:505` `swiglu` (C++ `aie_kernels/activation/swiglu.cc`, `swiglu_bf16`), `:435` `silu` / `:461` `silu_sized` (`activation/silu.cc`), `:418` `gelu` / `:482` `gelu_sized` (`activation/gelu.cc`). Fused GEMM epilogues: `kernels/fused.py:54` `fused_mm` (C++ `aie_kernels/fused/fused_mm_tile.cc`), `kernels/transformer.py:176` `mm_activation_epilogue` (C++ `transformer/mm_activation_epilogue.cc`) | `swiglu(x, w1, w2) = (x*w1) * silu(x*w2)` elementwise on already-projected tiles — the w1/w3/w2 GEMMs are separate. `swiglu`/`silu`/`gelu` take **fixed 1024-element tiles** (`_require_fixed_tile_size(..., _LUT_FIXED_TILE)`; Z-Image FFN inner 10240 → 10 calls of 1024). `silu_sized`/`gelu_sized` accept a compiled-in count, multiples of 16 (aie2) / 32 (aie2p). `fused_mm` offers GEMM + fused gelu/silu/sigmoid epilogue (and optional clamp) on one core, including `bfp16_b=True` (B as bfp16ebs8 blocks, aie2p). `mm_activation_epilogue` is an f32 row epilogue with runtime mode (identity/SiLU/GELU/ReLU). |
| Elementwise add / mul | **FOUND** | `kernels/eltwise.py:191` `add`, `:210` `mul`, `:229` `mul_add`, `:295` `add_sized`, `:322` `mul_sized`. C++ `aie_kernels/eltwise/add.cc`, `mul.cc`, `scale_shift.cc` (`eltwise_mul_add_bf16_vector`) | bf16. `add`/`mul`/`mul_add` are **fixed 1024 elements per call**; `add_sized`/`mul_sized` take any positive compiled-in size with scalar tails. `mul_add` computes `a*b` **or** `a+b` per call selected by `is_mul` — the `programming_examples/ml/scale_shift` two-phase pattern (multiply then add) is the building block for `x*scale + shift`. |
| Attention as a fused op (`mha`, `mha_softmax`) | **FOUND (toolkit, not a single fused call)** | `kernels/linalg.py:1086` `mha` (C++ `aie_kernels/linalg/mha.cc`: `matmul_bf16_bf16_wrapper` = QK^T, `matmul_bf16_bf16_rowmaj` = P·V; same object binds `matmul_bf16_bf16_wrapper_scalar`, `partial_softmax`, `matmul_PV`, `rescale_O`, `init_scale_buffer`), `linalg.py:1200` `mha_softmax`. Also `linalg.py:1299` `prefill_fv` (C++ `linalg/flash_attn_prefill.cc`) | `mha` dims: `dim_m`/`dim_n` multiples of **16**, `dim_k` multiple of **8**; bf16 micro-tiles accumulating f32 into C (zero-init via `kernels.zero`); dataflow for a full attention is composed by the design from these entry points. `mha_softmax` is one fixed **64×64** block (`_MHA_BLOCK = 64`) of online softmax with running `scale_buffer`; masking is **causal** (diagonal keep, key-block > query-block skipped) with tail padding via `S_q_eff`/`S_kv_eff` — a DiT needs *bidirectional* attention, so this masking would need bypass/rework. `prefill_fv`/`flash_attn_prefill` is a fused flash-attention prefill but only for **head_dim 512 or 256** (`_PREFILL_GEOM`, `linalg.py:1295`) — head_dim 128 is not covered. The `flm_gemma4` prefill/decode attention kernels (`flm_gemma4_attn_prefill`, `flm_gemma4_decode_attn_qk/kv`, …) are Gemma4-geometry-locked (model_dim 1536/2560, `flm_gemma4.py:902,918`). |
| adaLN modulation (scale/shift from a conditioning vector) | **MISSING** | — | No symbol or file matching adaLN/adaln/modulation exists anywhere in `python/aie/` or `include/aie_kernels/`. Closest building blocks, none of which is per-token conditioned modulation: `kernels/datamovement.py:211` `affine_cast` (`out = bf16(in*gamma+beta)`, C++ `aie_kernels/datamovement/affine_cast_f32_bf16.cc`) — per-**column** gamma/beta supplied as a baked tensor Param (f32 in, cols multiple of 16); `transformer.py:145` `layer_norm_affine_cast` — static gamma/beta fused into LayerNorm; and `eltwise.mul`/`mul_add` composition for `x*(1+scale)+shift`. A new kernel (or two-pass mul+add over bf16 tiles) is needed; the conditioning-vector MLP itself is just GEMM. |

## C++ kernels under `include/aie_kernels/` for the operators asked about

Every kernel in the requested directories has a Python wrapper in
`aie/iron/kernels/` (nothing was found unwrapped):

- `activation/`: `softmax.cc` → `softmax`; `silu.cc` → `silu`, `silu_sized`;
  `gelu.cc` → `gelu`, `gelu_sized`; `swiglu.cc` → `swiglu`; also
  `bf16_exp.cc`, `exp2f_vec.cc`, `tanh.cc`, `sigmoid.cc`, `leaky_relu.cc`,
  all wrapped in `activation.py`.
- `norm/`: `rms_norm.cc` → `rms_norm`, `rms_norm_eps`; `layer_norm.cc` →
  `layer_norm`. (No gamma/beta entry point in either.)
- `linalg/`: `mm_bfp.cc` → `mm_bfp`, `mm_bfp_shuffle`; `mm_bfp_mixed.cc` →
  `mm_bfp(mixed=True)`; `mha.cc` → `mha`, `mha_softmax`; `flash_attn_prefill.cc`
  → `prefill_fv`; `mm.cc` → `mm`; `mv_bf16.cc`/`mv_i16.cc` → `mv`;
  `cascade_mm.cc` → `cascade_mm`.
- `fused/`: `fused_mm_tile.cc` → `fused.py:fused_mm` (GEMM + activation
  epilogue on one core; supports `bfp16_b` BFP16 weights).
- `transformer/`: `layer_norm_f32.cc` → `layer_norm_f32`, and the
  `layer_norm_affine_cast` symbol (gamma/beta variant); `mm_activation_epilogue.cc`
  → `mm_activation_epilogue`. That is the entire directory.
- Also relevant: `datamovement/rope.cc` → `rope`; `datamovement/affine_cast_f32_bf16.cc`
  → `affine_cast`; `eltwise/scale_shift.cc` → `mul_add`; `flm_gemma4/decode_rms_residual.cc`
  → `flm_gemma4_rms_residual_core` (the one RMSNorm **with** weight + residual
  add in the tree, but geometry-locked to Gemma4 model_dim and eps 1e-6).

No adaLN / modulation C++ kernel exists in any directory.

## Examples under `zimage-bfp16-npu/_examples/`

Only `block_datatypes/` is present (BFP16 GEMM designs, all `@iron.jit`, default
device `npu2`). The best structure to copy:

- `_examples/block_datatypes/matrix_multiplication/whole_array_shuffle/whole_array.py`
  — the most composed design in the tree: `kernels.mm_bfp` +
  `kernels.zero` + `kernels.mm_bfp_shuffle` in one worker pipeline over the
  whole array (in-core shuffle before the GEMM).
- `.../single_core_mixed/single_core.py` and `.../whole_array_mixed/whole_array.py`
  — `mm_bfp(mixed=True)` (bf16 activations, bfp16ebs8 weights) + `zero`; the
  closest match to a BlockFP16-weight GEMM with bf16 activations.
- `.../single_core/single_core.py` — minimal `mm_bfp` + `zero` single-core form.
- `.../in_core_shuffle/single_core.py` — the shuffle alone as a building block.
- `gemm_asymmetric_tile_buffering/` contains only `README.md` + `CMakeLists.txt`
  in this snapshot (no design source to copy; it points at the UCLA-VAST repo).

No example here composes multiple *transformer* kernels (norm/softmax/RoPE/FFN)
into a pipeline — the most multi-kernel example is shuffle+zero+GEMM. A DiT
block pipeline would have to be assembled from the kernel factories above using
the `whole_array*` designs as the ObjectFifo/Worker/Runtime skeleton.

## Dimension constraints collected (Z-Image fit)

| Kernel | Constraint | Z-Image fit |
|---|---|---|
| `mm_bfp` | dim_m/k/n multiples of 8; aie2p only; B transposed; host-side bfp shuffle | 3840/10240/128/96 all OK |
| `rms_norm` / `rms_norm_eps` | any positive row length; gamma fixed 1 | hidden 3840 OK, needs separate gamma mul |
| `layer_norm` | cols multiple of 16 | 3840 OK |
| `softmax` | tile multiple of 32, one row per call | 96 OK |
| `mha` | dim_m/dim_n multiples of 16, dim_k multiple of 8 | head_dim 128, seq 96 OK |
| `mha_softmax` | fixed 64×64, **causal** mask | 96 = 2 blocks w/ 32-row tail; causal only — wrong mask for DiT |
| `prefill_fv` | head_dim ∈ {256, 512} | head_dim 128 **not supported** |
| `rope` | row multiple of 16 (or 32 for two_halves) | head_dim 128 OK |
| `swiglu` / `silu` / `gelu` | fixed 1024 elements | FFN inner 10240 → 1024-sized tiles |
| `add` / `mul` / `mul_add` | fixed 1024 elements (use `*_sized` for other sizes) | tile hidden/ffn rows accordingly |
| `affine_cast` | rows ≥ 1, cols multiple of 16, f32 in | usable shape-wise but gamma/beta baked, not per-token |
| `fused_mm` | dim_m % band_m, band_m % 2r, dim_n % 2t, dim_k % chunk_k, chunk_k % s, out_chunk % 16, … (`fused.py:132-139`) | bounded single-core tile, not the streaming whole-matrix path |

## Bottom line

FOUND: BFP16 GEMM (`mm_bfp`, incl. mixed bf16-activation/bfp16-weight form),
RMSNorm+LayerNorm (unweighted), softmax (row and attention-block forms), RoPE
(two layouts), SwiGLU/SiLU/GELU, elementwise add/mul/mul_add, and a
flash-attention toolkit (`mha` + `mha_softmax`).

MISSING: (1) adaLN modulation of any kind; (2) RMSNorm/LayerNorm **with a
learnable weight** at arbitrary geometry (only the Gemma4-locked
`flm_gemma4_rms_residual_core` has one); (3) a non-causal (bidirectional)
attention softmax — `mha_softmax` masks causally; (4) attention/flash kernels
at head_dim 128 (`prefill_fv` supports 256/512 only); (5) any multi-op
transformer-pipeline example to copy.
