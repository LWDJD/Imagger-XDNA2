# Z-Image-Turbo Transformer - Weight Tensor Inventory

Mechanical extraction from safetensors JSON headers. No tensor data was read; every number below
comes from the header JSON (`dtype`, `shape`, `data_offsets`) of the three shards plus
`diffusion_pytorch_model.safetensors.index.json`.

## 0. Source and method

Directory: `<workspace>\zimage-port\_models\Z-Image-Turbo\transformer\`

| File | Bytes |
|---|---:|
| `diffusion_pytorch_model-00001-of-00003.safetensors` | 9,973,693,184 |
| `diffusion_pytorch_model-00002-of-00003.safetensors` | 9,973,714,824 |
| `diffusion_pytorch_model-00003-of-00003.safetensors` | 4,672,282,880 |
| `diffusion_pytorch_model.safetensors.index.json` | 48,969 |
| `config.json` | 473 |
| `SHA256SUMS.modelscope.json` | 501 |

Method per shard: read 8 bytes (little-endian u64 header length), then read exactly that many
bytes and `json.loads` them. Header lengths: shard 1 = 23,032 B, shard 2 = 22,400 B, shard 3 = 10,488 B.
Each shard header carries `__metadata__: {"format": "pt"}` plus its tensors.

Index cross-check: index `weight_map` has 521 entries; the three headers contain exactly the same
521 names - no name missing on either side, and every index entry points at the shard whose header
actually contains the tensor. Index `metadata.total_size` = 24,619,634,944.

Byte-layout cross-check (header `data_offsets` vs. file sizes):

| Shard | tensors | first offset | last data end | sum of tensor bytes | contiguous spans | 8 + header + last end = file size |
|---|---:|---:|---:|---:|---|---|
| `00001-of-00003` | 214 | 0 | 9,973,670,144 | 9,973,670,144 | yes (gaps=0) | 9,973,693,184 = 9,973,693,184 |
| `00002-of-00003` | 209 | 0 | 9,973,692,416 | 9,973,692,416 | yes (gaps=0) | 9,973,714,824 = 9,973,714,824 |
| `00003-of-00003` | 98 | 0 | 4,672,272,384 | 4,672,272,384 | yes (gaps=0) | 4,672,282,880 = 4,672,282,880 |

Sum of the three file sizes = 24,619,690,888 B; that is 55,944 B more than the tensor bytes, and
55,944 = (8 + 23,032) + (8 + 22,400) + (8 + 10,488) exactly - i.e. the three length prefixes plus
the three JSON headers. Accounting is exact, byte for byte.

Classification rules applied exactly as specified:

* `GEMM-WEIGHT` - name ends in `.weight`, ndim == 2, not an embedding name
* `BIAS` - ndim == 1 and name ends in `.bias`
* `NORM` - ndim == 1 `.weight` belonging to a norm (`norm`, `norm_q`, `norm_k`, `attention_norm*`, `ffn_norm*`)
* `EMBED` - name contains `x_embedder`, `cap_embedder`, `t_embedder`, or `cap_pad_token` (checked first)
* `ROPE` / `OTHER` - anything else

All 521 tensors are `F32`; there is not a single other dtype. Byte size = numel x 4 for every tensor.

## 1. Totals and cross-check

| Class | Tensors | Parameters | Bytes |
|---|---:|---:|---:|
| GEMM-WEIGHT | 272 | 6,143,262,720 | 24,573,050,880 |
| BIAS | 34 | 495,424 | 1,981,696 |
| NORM | 204 | 530,944 | 2,123,776 |
| EMBED | 10 | 10,615,808 | 42,463,232 |
| OTHER | 1 | 3,840 | 15,360 |
| **TOTAL** | **521** | **6,154,908,736** | **24,619,634,944** |

Cross-check results:

* Parameter count **6,154,908,736** - matches the documented figure **exactly** (difference 0).
* Byte count **24,619,634,944** - matches the documented fp32 figure **exactly** (difference 0),
  and equals `index metadata.total_size` exactly.
* 6,154,908,736 x 4 = 24,619,634,944; the fp32 figure is simply 4x the parameter count.

**Verdict: both documented figures are right, to the parameter/byte. Nothing is off.**

Per-shard totals:

| Shard | Tensors | Parameters | Bytes |
|---|---:|---:|---:|
| `00001-of-00003` | 214 | 2,493,417,536 | 9,973,670,144 |
| `00002-of-00003` | 209 | 2,493,423,104 | 9,973,692,416 |
| `00003-of-00003` | 98 | 1,168,068,096 | 4,672,272,384 |

## 2. Distinct (N, K) shapes among GEMM-WEIGHT  [key output for a weight packer]

N = out_features = shape[0], K = in_features = shape[1]. 272 GEMM tensors, 6,143,262,720 params,
24,573,050,880 bytes. There are exactly **6 distinct shapes**:

| N (out) | K (in) | Tensors | Params | Bytes | Where used |
|---:|---:|---:|---:|---:|---|
| 10,240 | 3,840 | 68 | 2,673,868,800 | 10,695,475,200 | `feed_forward.w1`, `feed_forward.w3` in every one of the 34 transformer blocks |
| 3,840 | 3,840 | 136 | 2,005,401,600 | 8,021,606,400 | `attention.to_q`, `to_k`, `to_v`, `to_out.0` in every one of the 34 blocks |
| 3,840 | 10,240 | 34 | 1,336,934,400 | 5,347,737,600 | `feed_forward.w2` in every one of the 34 blocks |
| 15,360 | 256 | 32 | 125,829,120 | 503,316,480 | `adaLN_modulation.0` in `noise_refiner.{0,1}` and `layers.{0..29}` (32 blocks) |
| 3,840 | 256 | 1 | 983,040 | 3,932,160 | `all_final_layer.2-1.adaLN_modulation.1` (final adaLN) |
| 64 | 3,840 | 1 | 245,760 | 983,040 | `all_final_layer.2-1.linear` (final projection back to patch pixels) |
| **total** | | **272** | **6,143,262,720** | **24,573,050,880** | |

A weight packer must therefore support exactly six (N, K) pairs; the three large ones cover
97.9% of the GEMM parameters (6,016,204,800 / 6,143,262,720).

Supplementary - 2-D `.weight` matrices that are *excluded* from GEMM-WEIGHT only because the
embedding rule claims them first (they are ordinary linear operands and a packer will meet them):

| Name | (N, K) | Params | Bytes | Class by the stated rules |
|---|---|---:|---:|---|
| `all_x_embedder.2-1.weight` | (3840, 64) | 245,760 | 983,040 | EMBED |
| `cap_embedder.1.weight` | (3840, 2560) | 9,830,400 | 39,321,600 | EMBED |
| `t_embedder.mlp.0.weight` | (1024, 256) | 262,144 | 1,048,576 | EMBED |
| `t_embedder.mlp.2.weight` | (256, 1024) | 262,144 | 1,048,576 | EMBED |

If these four are packed as GEMMs too, the packer additionally needs (3840, 64), (3840, 2560),
(1024, 256) and (256, 1024) - one tensor each.

## 3. GEMM ordering per module block

Order below is the deterministic order of the index/header (lexicographic by full name).
Execution order within attention is to_q -> to_k -> to_v -> to_out.0.

### noise_refiner.0 and noise_refiner.1  (8 GEMMs, 180,879,360 params each)

| # | Name suffix | (N, K) |
|---:|---|---|
| 1 | `adaLN_modulation.0.weight` | (15360, 256) |
| 2 | `attention.to_k.weight` | (3840, 3840) |
| 3 | `attention.to_out.0.weight` | (3840, 3840) |
| 4 | `attention.to_q.weight` | (3840, 3840) |
| 5 | `attention.to_v.weight` | (3840, 3840) |
| 6 | `feed_forward.w1.weight` | (10240, 3840) |
| 7 | `feed_forward.w2.weight` | (3840, 10240) |
| 8 | `feed_forward.w3.weight` | (10240, 3840) |

`noise_refiner.1` is structurally identical: same suffix list, same order, same shapes.

### context_refiner.0 and context_refiner.1  (7 GEMMs, 176,947,200 params each)

| # | Name suffix | (N, K) |
|---:|---|---|
| 1 | `attention.to_k.weight` | (3840, 3840) |
| 2 | `attention.to_out.0.weight` | (3840, 3840) |
| 3 | `attention.to_q.weight` | (3840, 3840) |
| 4 | `attention.to_v.weight` | (3840, 3840) |
| 5 | `feed_forward.w1.weight` | (10240, 3840) |
| 6 | `feed_forward.w2.weight` | (3840, 10240) |
| 7 | `feed_forward.w3.weight` | (10240, 3840) |

`context_refiner.1` is structurally identical. Note: no `adaLN_modulation` at all in these two
blocks - that is the only per-block GEMM difference.

### layers.0 ... layers.29  (8 GEMMs each, 180,879,360 params each; 30 blocks)

| # | Name suffix | (N, K) |
|---:|---|---|
| 1 | `adaLN_modulation.0.weight` | (15360, 256) |
| 2 | `attention.to_k.weight` | (3840, 3840) |
| 3 | `attention.to_out.0.weight` | (3840, 3840) |
| 4 | `attention.to_q.weight` | (3840, 3840) |
| 5 | `attention.to_v.weight` | (3840, 3840) |
| 6 | `feed_forward.w1.weight` | (10240, 3840) |
| 7 | `feed_forward.w2.weight` | (3840, 10240) |
| 8 | `feed_forward.w3.weight` | (10240, 3840) |

Verified programmatically: all 30 layers carry the identical suffix list in identical order,
and identical shapes. Full names are `layers.<i>.<suffix>` with i = 0..29.

### final_layer  (2 GEMMs, 1,228,800 params)

| # | Full name | (N, K) |
|---:|---|---|
| 1 | `all_final_layer.2-1.adaLN_modulation.1.weight` | (3840, 256) |
| 2 | `all_final_layer.2-1.linear.weight` | (64, 3840) |

Block totals: 2 x 180,879,360 + 2 x 176,947,200 + 30 x 180,879,360 + 1,228,800 = 6,143,262,720 params
(the GEMM total). Per-shard placement of complete layers:

| Shard | Layer tensors held |
|---|---|
| `00001-of-00003` | layers 0-8 complete (15 tensors each) + 8 tensors of layer 9 (`attention.norm_k`, `attention.norm_q`, `attention.to_k`, `attention.to_out.0`, `attention.to_q`, `attention.to_v`, `feed_forward.w1`, `feed_forward.w2`) |
| `00002-of-00003` | 7 tensors of layer 9 (`adaLN_modulation.0.{weight,bias}`, `attention_norm1`, `attention_norm2`, `feed_forward.w3`, `ffn_norm1`, `ffn_norm2`) + layers 10-22 complete + 7 tensors of layer 23 (`attention.norm_k`, `attention.norm_q`, `attention.to_k`, `attention.to_out.0`, `attention.to_q`, `attention.to_v`, `feed_forward.w1`) |
| `00003-of-00003` | 8 tensors of layer 23 (`adaLN_modulation.0.{weight,bias}`, `attention_norm1`, `attention_norm2`, `feed_forward.w2`, `feed_forward.w3`, `ffn_norm1`, `ffn_norm2`) + layers 24-29 complete |

All non-`layers` tensors (embedders, both refiners, final layer, pad tokens) live in `00001-of-00003`.

## 4. Tensors that do not fit the (out, in) convention

Zero tensors have ndim >= 3 (ndim distribution: {1: 243, 2: 278}).
No 2-D GEMM looks transposed: every (N, K) matches the module semantics (w1/w3 = 10240x3840 out x in,
w2 = 3840x10240, q/k/v/out = 3840x3840). The following tensors are the ones that are *not*
plain (out, in) linear weights:

| Name | Shape | DType | Shard | Bytes | Class | Why it is odd |
|---|---|---|---|---:|---|---|
| `cap_pad_token` | [1, 3840] | F32 | 00001 | 15,360 | EMBED | 2-D but no `.weight`/`.bias` suffix; a learned pad-token vector stored as a row |
| `x_pad_token` | [1, 3840] | F32 | 00001 | 15,360 | OTHER | same shape as `cap_pad_token`, but its name is not in the EMBED rule, so by the stated rules it lands in OTHER |
| `cap_embedder.0.weight` | [2560] | F32 | 00001 | 10,240 | EMBED | 1-D weight (norm-shaped, almost certainly the LayerNorm of `cap_embedder`) carrying no `norm` in its name |
| `attention.norm_q.weight` (x34) | [128] | F32 | various | 512 ea. | NORM | 1-D norm over head_dim 128 (= 3840 / 30 heads), not over dim 3840 |
| `attention.norm_k.weight` (x34) | [128] | F32 | various | 512 ea. | NORM | same as above |
| `all_x_embedder.2-1.weight` | [3840, 64] | F32 | 00001 | 983,040 | EMBED | 2-D linear (64 = 16 channels x 2 x 2 patch) claimed by the embed rule |
| `t_embedder.mlp.0.weight` | [1024, 256] | F32 | 00001 | 1,048,576 | EMBED | 2-D linear inside t_embedder |
| `t_embedder.mlp.2.weight` | [256, 1024] | F32 | 00001 | 1,048,576 | EMBED | 2-D linear inside t_embedder |
| `cap_embedder.1.weight` | [3840, 2560] | F32 | 00001 | 39,321,600 | EMBED | 2-D linear (2560 = config `cap_feat_dim`) claimed by the embed rule |

## 5. Surprising / inconsistent things

1. **Both headline figures are exact.** 6,154,908,736 params and 24,619,634,944 bytes both match
   the extraction with zero difference. No rounding or off-by-one anywhere.
2. **Everything is F32.** All 521 tensors, no bf16/fp16 anywhere - hence bytes = 4 x params exactly.
3. **Weird module names with a `2-1` infix:** the output layer is `all_final_layer.2-1.*` and the
   patch embedder is `all_x_embedder.2-1.*`. There is no plain `final_layer.*` or `x_embedder.*` key.
   Any loader expecting the standard Z-Image names must remap these two prefixes.
4. **adaLN index inconsistency:** in `noise_refiner.*` and `layers.*` the modulation linear is
   `adaLN_modulation.0.{weight,bias}` (index 0), but in `all_final_layer.2-1` it is
   `adaLN_modulation.1.{weight,bias}` (index 1) - i.e. wrapped in a Sequential with a non-parameter
   module at slot 0 in one place but not the other. A key-indexed loader must handle both indices.
5. **`context_refiner.{0,1}` have no adaLN at all** (7 GEMMs vs 8 everywhere else).
6. **No bias on any attention or feed-forward projection.** The 34 BIAS-class tensors are exactly:
   32 x `adaLN_modulation.0.bias` [15360] (30 layers + 2 noise_refiner), 1 x
   `all_final_layer.2-1.adaLN_modulation.1.bias` [3840], 1 x `all_final_layer.2-1.linear.bias` [64].
   (`to_q/to_k/to_v/to_out.0` and `w1/w2/w3` are all bias-free.) Four further 1-D `.bias` vectors
   exist inside embedder modules - `all_x_embedder.2-1.bias` [3840], `cap_embedder.1.bias` [3840],
   `t_embedder.mlp.0.bias` [1024], `t_embedder.mlp.2.bias` [256] - but the EMBED name rule is applied
   first, so they are classed EMBED, not BIAS.
7. **No ROPE tensors exist** (class `ROPE` is empty). Config `axes_dims: [32, 48, 48]` sums to 128 =
   head_dim (3840/30), `axes_lens: [1536, 512, 512]`, `rope_theta: 256.0` - RoPE is computed at
   runtime from the config, not stored. This is consistent, but worth stating for a packer.
8. **`x_pad_token` vs `cap_pad_token`:** two pad tokens of identical shape [1, 3840]. The stated
   EMBED rule names only `cap_pad_token`, so `x_pad_token` (3,840 params) is classified OTHER.
   If the intent was that both pad tokens are embeddings, move those 3,840 params from OTHER to
   EMBED: EMBED becomes 11 tensors / 10,619,648 params and OTHER becomes empty. Totals unchanged.
9. **`cap_embedder.0.weight` is a 1-D [2560] weight inside an EMBED module** - almost certainly a
   LayerNorm, but it has no `norm` in its name, so the rules classify it EMBED, not NORM.
10. **Shard boundaries cut layers 9 and 23 in half** (see section 3 table): layer 9 is split 8/7
    between shards 1/2, layer 23 is split 7/8 between shards 2/3. A packer streaming one shard at
    a time cannot assume a layer lives in a single file.
11. **Byte order inside each shard is lexicographic by tensor name** (verified: for each of the three
    shards, `data_offsets` are strictly increasing along lexicographically sorted names),
    which is *not* module/execution order: inside a layer `to_k` precedes `to_q`, and shard 2 begins
    with `layers.10` and ends with `layers.9` because "layers.9" sorts after "layers.23".
12. **Config cross-check (corroborating, from `config.json`):** dim=3840, n_heads=30 -> head_dim=128
    matches the [128] qk-norms; cap_feat_dim=2560 matches `cap_embedder.1` K; in_channels=16 and
    patch_size 2 -> 64 matches `all_x_embedder` K and final N; n_layers=30 and n_refiner_layers=2
    match the block counts. adaLN N=15360 = 4 x 3840 (shift+scale for attention and FFN).

## Appendix A. Complete tensor inventory (521 tensors, sorted by name)

Shard legend: `00001-of-00003` = `diffusion_pytorch_model-00001-of-00003.safetensors`, etc.
Byte size = product(shape) x 4 (all F32).

| # | Name | Shape | DType | Shard | Bytes | Class |
|---:|---|---|---|---|---:|---|
| 1 | `all_final_layer.2-1.adaLN_modulation.1.bias` | [3840] | F32 | 00001-of-00003 | 15,360 | BIAS |
| 2 | `all_final_layer.2-1.adaLN_modulation.1.weight` | [3840, 256] | F32 | 00001-of-00003 | 3,932,160 | GEMM-WEIGHT |
| 3 | `all_final_layer.2-1.linear.bias` | [64] | F32 | 00001-of-00003 | 256 | BIAS |
| 4 | `all_final_layer.2-1.linear.weight` | [64, 3840] | F32 | 00001-of-00003 | 983,040 | GEMM-WEIGHT |
| 5 | `all_x_embedder.2-1.bias` | [3840] | F32 | 00001-of-00003 | 15,360 | EMBED |
| 6 | `all_x_embedder.2-1.weight` | [3840, 64] | F32 | 00001-of-00003 | 983,040 | EMBED |
| 7 | `cap_embedder.0.weight` | [2560] | F32 | 00001-of-00003 | 10,240 | EMBED |
| 8 | `cap_embedder.1.bias` | [3840] | F32 | 00001-of-00003 | 15,360 | EMBED |
| 9 | `cap_embedder.1.weight` | [3840, 2560] | F32 | 00001-of-00003 | 39,321,600 | EMBED |
| 10 | `cap_pad_token` | [1, 3840] | F32 | 00001-of-00003 | 15,360 | EMBED |
| 11 | `context_refiner.0.attention.norm_k.weight` | [128] | F32 | 00001-of-00003 | 512 | NORM |
| 12 | `context_refiner.0.attention.norm_q.weight` | [128] | F32 | 00001-of-00003 | 512 | NORM |
| 13 | `context_refiner.0.attention.to_k.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 14 | `context_refiner.0.attention.to_out.0.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 15 | `context_refiner.0.attention.to_q.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 16 | `context_refiner.0.attention.to_v.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 17 | `context_refiner.0.attention_norm1.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 18 | `context_refiner.0.attention_norm2.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 19 | `context_refiner.0.feed_forward.w1.weight` | [10240, 3840] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 20 | `context_refiner.0.feed_forward.w2.weight` | [3840, 10240] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 21 | `context_refiner.0.feed_forward.w3.weight` | [10240, 3840] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 22 | `context_refiner.0.ffn_norm1.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 23 | `context_refiner.0.ffn_norm2.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 24 | `context_refiner.1.attention.norm_k.weight` | [128] | F32 | 00001-of-00003 | 512 | NORM |
| 25 | `context_refiner.1.attention.norm_q.weight` | [128] | F32 | 00001-of-00003 | 512 | NORM |
| 26 | `context_refiner.1.attention.to_k.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 27 | `context_refiner.1.attention.to_out.0.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 28 | `context_refiner.1.attention.to_q.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 29 | `context_refiner.1.attention.to_v.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 30 | `context_refiner.1.attention_norm1.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 31 | `context_refiner.1.attention_norm2.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 32 | `context_refiner.1.feed_forward.w1.weight` | [10240, 3840] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 33 | `context_refiner.1.feed_forward.w2.weight` | [3840, 10240] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 34 | `context_refiner.1.feed_forward.w3.weight` | [10240, 3840] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 35 | `context_refiner.1.ffn_norm1.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 36 | `context_refiner.1.ffn_norm2.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 37 | `layers.0.adaLN_modulation.0.bias` | [15360] | F32 | 00001-of-00003 | 61,440 | BIAS |
| 38 | `layers.0.adaLN_modulation.0.weight` | [15360, 256] | F32 | 00001-of-00003 | 15,728,640 | GEMM-WEIGHT |
| 39 | `layers.0.attention.norm_k.weight` | [128] | F32 | 00001-of-00003 | 512 | NORM |
| 40 | `layers.0.attention.norm_q.weight` | [128] | F32 | 00001-of-00003 | 512 | NORM |
| 41 | `layers.0.attention.to_k.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 42 | `layers.0.attention.to_out.0.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 43 | `layers.0.attention.to_q.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 44 | `layers.0.attention.to_v.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 45 | `layers.0.attention_norm1.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 46 | `layers.0.attention_norm2.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 47 | `layers.0.feed_forward.w1.weight` | [10240, 3840] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 48 | `layers.0.feed_forward.w2.weight` | [3840, 10240] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 49 | `layers.0.feed_forward.w3.weight` | [10240, 3840] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 50 | `layers.0.ffn_norm1.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 51 | `layers.0.ffn_norm2.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 52 | `layers.1.adaLN_modulation.0.bias` | [15360] | F32 | 00001-of-00003 | 61,440 | BIAS |
| 53 | `layers.1.adaLN_modulation.0.weight` | [15360, 256] | F32 | 00001-of-00003 | 15,728,640 | GEMM-WEIGHT |
| 54 | `layers.1.attention.norm_k.weight` | [128] | F32 | 00001-of-00003 | 512 | NORM |
| 55 | `layers.1.attention.norm_q.weight` | [128] | F32 | 00001-of-00003 | 512 | NORM |
| 56 | `layers.1.attention.to_k.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 57 | `layers.1.attention.to_out.0.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 58 | `layers.1.attention.to_q.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 59 | `layers.1.attention.to_v.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 60 | `layers.1.attention_norm1.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 61 | `layers.1.attention_norm2.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 62 | `layers.1.feed_forward.w1.weight` | [10240, 3840] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 63 | `layers.1.feed_forward.w2.weight` | [3840, 10240] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 64 | `layers.1.feed_forward.w3.weight` | [10240, 3840] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 65 | `layers.1.ffn_norm1.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 66 | `layers.1.ffn_norm2.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 67 | `layers.10.adaLN_modulation.0.bias` | [15360] | F32 | 00002-of-00003 | 61,440 | BIAS |
| 68 | `layers.10.adaLN_modulation.0.weight` | [15360, 256] | F32 | 00002-of-00003 | 15,728,640 | GEMM-WEIGHT |
| 69 | `layers.10.attention.norm_k.weight` | [128] | F32 | 00002-of-00003 | 512 | NORM |
| 70 | `layers.10.attention.norm_q.weight` | [128] | F32 | 00002-of-00003 | 512 | NORM |
| 71 | `layers.10.attention.to_k.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 72 | `layers.10.attention.to_out.0.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 73 | `layers.10.attention.to_q.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 74 | `layers.10.attention.to_v.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 75 | `layers.10.attention_norm1.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 76 | `layers.10.attention_norm2.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 77 | `layers.10.feed_forward.w1.weight` | [10240, 3840] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 78 | `layers.10.feed_forward.w2.weight` | [3840, 10240] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 79 | `layers.10.feed_forward.w3.weight` | [10240, 3840] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 80 | `layers.10.ffn_norm1.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 81 | `layers.10.ffn_norm2.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 82 | `layers.11.adaLN_modulation.0.bias` | [15360] | F32 | 00002-of-00003 | 61,440 | BIAS |
| 83 | `layers.11.adaLN_modulation.0.weight` | [15360, 256] | F32 | 00002-of-00003 | 15,728,640 | GEMM-WEIGHT |
| 84 | `layers.11.attention.norm_k.weight` | [128] | F32 | 00002-of-00003 | 512 | NORM |
| 85 | `layers.11.attention.norm_q.weight` | [128] | F32 | 00002-of-00003 | 512 | NORM |
| 86 | `layers.11.attention.to_k.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 87 | `layers.11.attention.to_out.0.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 88 | `layers.11.attention.to_q.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 89 | `layers.11.attention.to_v.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 90 | `layers.11.attention_norm1.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 91 | `layers.11.attention_norm2.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 92 | `layers.11.feed_forward.w1.weight` | [10240, 3840] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 93 | `layers.11.feed_forward.w2.weight` | [3840, 10240] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 94 | `layers.11.feed_forward.w3.weight` | [10240, 3840] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 95 | `layers.11.ffn_norm1.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 96 | `layers.11.ffn_norm2.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 97 | `layers.12.adaLN_modulation.0.bias` | [15360] | F32 | 00002-of-00003 | 61,440 | BIAS |
| 98 | `layers.12.adaLN_modulation.0.weight` | [15360, 256] | F32 | 00002-of-00003 | 15,728,640 | GEMM-WEIGHT |
| 99 | `layers.12.attention.norm_k.weight` | [128] | F32 | 00002-of-00003 | 512 | NORM |
| 100 | `layers.12.attention.norm_q.weight` | [128] | F32 | 00002-of-00003 | 512 | NORM |
| 101 | `layers.12.attention.to_k.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 102 | `layers.12.attention.to_out.0.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 103 | `layers.12.attention.to_q.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 104 | `layers.12.attention.to_v.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 105 | `layers.12.attention_norm1.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 106 | `layers.12.attention_norm2.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 107 | `layers.12.feed_forward.w1.weight` | [10240, 3840] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 108 | `layers.12.feed_forward.w2.weight` | [3840, 10240] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 109 | `layers.12.feed_forward.w3.weight` | [10240, 3840] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 110 | `layers.12.ffn_norm1.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 111 | `layers.12.ffn_norm2.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 112 | `layers.13.adaLN_modulation.0.bias` | [15360] | F32 | 00002-of-00003 | 61,440 | BIAS |
| 113 | `layers.13.adaLN_modulation.0.weight` | [15360, 256] | F32 | 00002-of-00003 | 15,728,640 | GEMM-WEIGHT |
| 114 | `layers.13.attention.norm_k.weight` | [128] | F32 | 00002-of-00003 | 512 | NORM |
| 115 | `layers.13.attention.norm_q.weight` | [128] | F32 | 00002-of-00003 | 512 | NORM |
| 116 | `layers.13.attention.to_k.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 117 | `layers.13.attention.to_out.0.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 118 | `layers.13.attention.to_q.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 119 | `layers.13.attention.to_v.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 120 | `layers.13.attention_norm1.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 121 | `layers.13.attention_norm2.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 122 | `layers.13.feed_forward.w1.weight` | [10240, 3840] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 123 | `layers.13.feed_forward.w2.weight` | [3840, 10240] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 124 | `layers.13.feed_forward.w3.weight` | [10240, 3840] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 125 | `layers.13.ffn_norm1.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 126 | `layers.13.ffn_norm2.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 127 | `layers.14.adaLN_modulation.0.bias` | [15360] | F32 | 00002-of-00003 | 61,440 | BIAS |
| 128 | `layers.14.adaLN_modulation.0.weight` | [15360, 256] | F32 | 00002-of-00003 | 15,728,640 | GEMM-WEIGHT |
| 129 | `layers.14.attention.norm_k.weight` | [128] | F32 | 00002-of-00003 | 512 | NORM |
| 130 | `layers.14.attention.norm_q.weight` | [128] | F32 | 00002-of-00003 | 512 | NORM |
| 131 | `layers.14.attention.to_k.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 132 | `layers.14.attention.to_out.0.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 133 | `layers.14.attention.to_q.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 134 | `layers.14.attention.to_v.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 135 | `layers.14.attention_norm1.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 136 | `layers.14.attention_norm2.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 137 | `layers.14.feed_forward.w1.weight` | [10240, 3840] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 138 | `layers.14.feed_forward.w2.weight` | [3840, 10240] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 139 | `layers.14.feed_forward.w3.weight` | [10240, 3840] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 140 | `layers.14.ffn_norm1.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 141 | `layers.14.ffn_norm2.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 142 | `layers.15.adaLN_modulation.0.bias` | [15360] | F32 | 00002-of-00003 | 61,440 | BIAS |
| 143 | `layers.15.adaLN_modulation.0.weight` | [15360, 256] | F32 | 00002-of-00003 | 15,728,640 | GEMM-WEIGHT |
| 144 | `layers.15.attention.norm_k.weight` | [128] | F32 | 00002-of-00003 | 512 | NORM |
| 145 | `layers.15.attention.norm_q.weight` | [128] | F32 | 00002-of-00003 | 512 | NORM |
| 146 | `layers.15.attention.to_k.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 147 | `layers.15.attention.to_out.0.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 148 | `layers.15.attention.to_q.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 149 | `layers.15.attention.to_v.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 150 | `layers.15.attention_norm1.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 151 | `layers.15.attention_norm2.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 152 | `layers.15.feed_forward.w1.weight` | [10240, 3840] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 153 | `layers.15.feed_forward.w2.weight` | [3840, 10240] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 154 | `layers.15.feed_forward.w3.weight` | [10240, 3840] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 155 | `layers.15.ffn_norm1.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 156 | `layers.15.ffn_norm2.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 157 | `layers.16.adaLN_modulation.0.bias` | [15360] | F32 | 00002-of-00003 | 61,440 | BIAS |
| 158 | `layers.16.adaLN_modulation.0.weight` | [15360, 256] | F32 | 00002-of-00003 | 15,728,640 | GEMM-WEIGHT |
| 159 | `layers.16.attention.norm_k.weight` | [128] | F32 | 00002-of-00003 | 512 | NORM |
| 160 | `layers.16.attention.norm_q.weight` | [128] | F32 | 00002-of-00003 | 512 | NORM |
| 161 | `layers.16.attention.to_k.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 162 | `layers.16.attention.to_out.0.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 163 | `layers.16.attention.to_q.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 164 | `layers.16.attention.to_v.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 165 | `layers.16.attention_norm1.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 166 | `layers.16.attention_norm2.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 167 | `layers.16.feed_forward.w1.weight` | [10240, 3840] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 168 | `layers.16.feed_forward.w2.weight` | [3840, 10240] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 169 | `layers.16.feed_forward.w3.weight` | [10240, 3840] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 170 | `layers.16.ffn_norm1.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 171 | `layers.16.ffn_norm2.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 172 | `layers.17.adaLN_modulation.0.bias` | [15360] | F32 | 00002-of-00003 | 61,440 | BIAS |
| 173 | `layers.17.adaLN_modulation.0.weight` | [15360, 256] | F32 | 00002-of-00003 | 15,728,640 | GEMM-WEIGHT |
| 174 | `layers.17.attention.norm_k.weight` | [128] | F32 | 00002-of-00003 | 512 | NORM |
| 175 | `layers.17.attention.norm_q.weight` | [128] | F32 | 00002-of-00003 | 512 | NORM |
| 176 | `layers.17.attention.to_k.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 177 | `layers.17.attention.to_out.0.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 178 | `layers.17.attention.to_q.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 179 | `layers.17.attention.to_v.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 180 | `layers.17.attention_norm1.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 181 | `layers.17.attention_norm2.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 182 | `layers.17.feed_forward.w1.weight` | [10240, 3840] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 183 | `layers.17.feed_forward.w2.weight` | [3840, 10240] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 184 | `layers.17.feed_forward.w3.weight` | [10240, 3840] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 185 | `layers.17.ffn_norm1.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 186 | `layers.17.ffn_norm2.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 187 | `layers.18.adaLN_modulation.0.bias` | [15360] | F32 | 00002-of-00003 | 61,440 | BIAS |
| 188 | `layers.18.adaLN_modulation.0.weight` | [15360, 256] | F32 | 00002-of-00003 | 15,728,640 | GEMM-WEIGHT |
| 189 | `layers.18.attention.norm_k.weight` | [128] | F32 | 00002-of-00003 | 512 | NORM |
| 190 | `layers.18.attention.norm_q.weight` | [128] | F32 | 00002-of-00003 | 512 | NORM |
| 191 | `layers.18.attention.to_k.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 192 | `layers.18.attention.to_out.0.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 193 | `layers.18.attention.to_q.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 194 | `layers.18.attention.to_v.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 195 | `layers.18.attention_norm1.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 196 | `layers.18.attention_norm2.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 197 | `layers.18.feed_forward.w1.weight` | [10240, 3840] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 198 | `layers.18.feed_forward.w2.weight` | [3840, 10240] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 199 | `layers.18.feed_forward.w3.weight` | [10240, 3840] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 200 | `layers.18.ffn_norm1.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 201 | `layers.18.ffn_norm2.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 202 | `layers.19.adaLN_modulation.0.bias` | [15360] | F32 | 00002-of-00003 | 61,440 | BIAS |
| 203 | `layers.19.adaLN_modulation.0.weight` | [15360, 256] | F32 | 00002-of-00003 | 15,728,640 | GEMM-WEIGHT |
| 204 | `layers.19.attention.norm_k.weight` | [128] | F32 | 00002-of-00003 | 512 | NORM |
| 205 | `layers.19.attention.norm_q.weight` | [128] | F32 | 00002-of-00003 | 512 | NORM |
| 206 | `layers.19.attention.to_k.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 207 | `layers.19.attention.to_out.0.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 208 | `layers.19.attention.to_q.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 209 | `layers.19.attention.to_v.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 210 | `layers.19.attention_norm1.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 211 | `layers.19.attention_norm2.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 212 | `layers.19.feed_forward.w1.weight` | [10240, 3840] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 213 | `layers.19.feed_forward.w2.weight` | [3840, 10240] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 214 | `layers.19.feed_forward.w3.weight` | [10240, 3840] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 215 | `layers.19.ffn_norm1.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 216 | `layers.19.ffn_norm2.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 217 | `layers.2.adaLN_modulation.0.bias` | [15360] | F32 | 00001-of-00003 | 61,440 | BIAS |
| 218 | `layers.2.adaLN_modulation.0.weight` | [15360, 256] | F32 | 00001-of-00003 | 15,728,640 | GEMM-WEIGHT |
| 219 | `layers.2.attention.norm_k.weight` | [128] | F32 | 00001-of-00003 | 512 | NORM |
| 220 | `layers.2.attention.norm_q.weight` | [128] | F32 | 00001-of-00003 | 512 | NORM |
| 221 | `layers.2.attention.to_k.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 222 | `layers.2.attention.to_out.0.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 223 | `layers.2.attention.to_q.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 224 | `layers.2.attention.to_v.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 225 | `layers.2.attention_norm1.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 226 | `layers.2.attention_norm2.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 227 | `layers.2.feed_forward.w1.weight` | [10240, 3840] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 228 | `layers.2.feed_forward.w2.weight` | [3840, 10240] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 229 | `layers.2.feed_forward.w3.weight` | [10240, 3840] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 230 | `layers.2.ffn_norm1.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 231 | `layers.2.ffn_norm2.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 232 | `layers.20.adaLN_modulation.0.bias` | [15360] | F32 | 00002-of-00003 | 61,440 | BIAS |
| 233 | `layers.20.adaLN_modulation.0.weight` | [15360, 256] | F32 | 00002-of-00003 | 15,728,640 | GEMM-WEIGHT |
| 234 | `layers.20.attention.norm_k.weight` | [128] | F32 | 00002-of-00003 | 512 | NORM |
| 235 | `layers.20.attention.norm_q.weight` | [128] | F32 | 00002-of-00003 | 512 | NORM |
| 236 | `layers.20.attention.to_k.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 237 | `layers.20.attention.to_out.0.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 238 | `layers.20.attention.to_q.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 239 | `layers.20.attention.to_v.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 240 | `layers.20.attention_norm1.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 241 | `layers.20.attention_norm2.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 242 | `layers.20.feed_forward.w1.weight` | [10240, 3840] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 243 | `layers.20.feed_forward.w2.weight` | [3840, 10240] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 244 | `layers.20.feed_forward.w3.weight` | [10240, 3840] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 245 | `layers.20.ffn_norm1.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 246 | `layers.20.ffn_norm2.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 247 | `layers.21.adaLN_modulation.0.bias` | [15360] | F32 | 00002-of-00003 | 61,440 | BIAS |
| 248 | `layers.21.adaLN_modulation.0.weight` | [15360, 256] | F32 | 00002-of-00003 | 15,728,640 | GEMM-WEIGHT |
| 249 | `layers.21.attention.norm_k.weight` | [128] | F32 | 00002-of-00003 | 512 | NORM |
| 250 | `layers.21.attention.norm_q.weight` | [128] | F32 | 00002-of-00003 | 512 | NORM |
| 251 | `layers.21.attention.to_k.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 252 | `layers.21.attention.to_out.0.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 253 | `layers.21.attention.to_q.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 254 | `layers.21.attention.to_v.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 255 | `layers.21.attention_norm1.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 256 | `layers.21.attention_norm2.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 257 | `layers.21.feed_forward.w1.weight` | [10240, 3840] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 258 | `layers.21.feed_forward.w2.weight` | [3840, 10240] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 259 | `layers.21.feed_forward.w3.weight` | [10240, 3840] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 260 | `layers.21.ffn_norm1.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 261 | `layers.21.ffn_norm2.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 262 | `layers.22.adaLN_modulation.0.bias` | [15360] | F32 | 00002-of-00003 | 61,440 | BIAS |
| 263 | `layers.22.adaLN_modulation.0.weight` | [15360, 256] | F32 | 00002-of-00003 | 15,728,640 | GEMM-WEIGHT |
| 264 | `layers.22.attention.norm_k.weight` | [128] | F32 | 00002-of-00003 | 512 | NORM |
| 265 | `layers.22.attention.norm_q.weight` | [128] | F32 | 00002-of-00003 | 512 | NORM |
| 266 | `layers.22.attention.to_k.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 267 | `layers.22.attention.to_out.0.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 268 | `layers.22.attention.to_q.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 269 | `layers.22.attention.to_v.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 270 | `layers.22.attention_norm1.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 271 | `layers.22.attention_norm2.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 272 | `layers.22.feed_forward.w1.weight` | [10240, 3840] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 273 | `layers.22.feed_forward.w2.weight` | [3840, 10240] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 274 | `layers.22.feed_forward.w3.weight` | [10240, 3840] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 275 | `layers.22.ffn_norm1.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 276 | `layers.22.ffn_norm2.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 277 | `layers.23.adaLN_modulation.0.bias` | [15360] | F32 | 00003-of-00003 | 61,440 | BIAS |
| 278 | `layers.23.adaLN_modulation.0.weight` | [15360, 256] | F32 | 00003-of-00003 | 15,728,640 | GEMM-WEIGHT |
| 279 | `layers.23.attention.norm_k.weight` | [128] | F32 | 00002-of-00003 | 512 | NORM |
| 280 | `layers.23.attention.norm_q.weight` | [128] | F32 | 00002-of-00003 | 512 | NORM |
| 281 | `layers.23.attention.to_k.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 282 | `layers.23.attention.to_out.0.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 283 | `layers.23.attention.to_q.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 284 | `layers.23.attention.to_v.weight` | [3840, 3840] | F32 | 00002-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 285 | `layers.23.attention_norm1.weight` | [3840] | F32 | 00003-of-00003 | 15,360 | NORM |
| 286 | `layers.23.attention_norm2.weight` | [3840] | F32 | 00003-of-00003 | 15,360 | NORM |
| 287 | `layers.23.feed_forward.w1.weight` | [10240, 3840] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 288 | `layers.23.feed_forward.w2.weight` | [3840, 10240] | F32 | 00003-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 289 | `layers.23.feed_forward.w3.weight` | [10240, 3840] | F32 | 00003-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 290 | `layers.23.ffn_norm1.weight` | [3840] | F32 | 00003-of-00003 | 15,360 | NORM |
| 291 | `layers.23.ffn_norm2.weight` | [3840] | F32 | 00003-of-00003 | 15,360 | NORM |
| 292 | `layers.24.adaLN_modulation.0.bias` | [15360] | F32 | 00003-of-00003 | 61,440 | BIAS |
| 293 | `layers.24.adaLN_modulation.0.weight` | [15360, 256] | F32 | 00003-of-00003 | 15,728,640 | GEMM-WEIGHT |
| 294 | `layers.24.attention.norm_k.weight` | [128] | F32 | 00003-of-00003 | 512 | NORM |
| 295 | `layers.24.attention.norm_q.weight` | [128] | F32 | 00003-of-00003 | 512 | NORM |
| 296 | `layers.24.attention.to_k.weight` | [3840, 3840] | F32 | 00003-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 297 | `layers.24.attention.to_out.0.weight` | [3840, 3840] | F32 | 00003-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 298 | `layers.24.attention.to_q.weight` | [3840, 3840] | F32 | 00003-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 299 | `layers.24.attention.to_v.weight` | [3840, 3840] | F32 | 00003-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 300 | `layers.24.attention_norm1.weight` | [3840] | F32 | 00003-of-00003 | 15,360 | NORM |
| 301 | `layers.24.attention_norm2.weight` | [3840] | F32 | 00003-of-00003 | 15,360 | NORM |
| 302 | `layers.24.feed_forward.w1.weight` | [10240, 3840] | F32 | 00003-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 303 | `layers.24.feed_forward.w2.weight` | [3840, 10240] | F32 | 00003-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 304 | `layers.24.feed_forward.w3.weight` | [10240, 3840] | F32 | 00003-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 305 | `layers.24.ffn_norm1.weight` | [3840] | F32 | 00003-of-00003 | 15,360 | NORM |
| 306 | `layers.24.ffn_norm2.weight` | [3840] | F32 | 00003-of-00003 | 15,360 | NORM |
| 307 | `layers.25.adaLN_modulation.0.bias` | [15360] | F32 | 00003-of-00003 | 61,440 | BIAS |
| 308 | `layers.25.adaLN_modulation.0.weight` | [15360, 256] | F32 | 00003-of-00003 | 15,728,640 | GEMM-WEIGHT |
| 309 | `layers.25.attention.norm_k.weight` | [128] | F32 | 00003-of-00003 | 512 | NORM |
| 310 | `layers.25.attention.norm_q.weight` | [128] | F32 | 00003-of-00003 | 512 | NORM |
| 311 | `layers.25.attention.to_k.weight` | [3840, 3840] | F32 | 00003-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 312 | `layers.25.attention.to_out.0.weight` | [3840, 3840] | F32 | 00003-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 313 | `layers.25.attention.to_q.weight` | [3840, 3840] | F32 | 00003-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 314 | `layers.25.attention.to_v.weight` | [3840, 3840] | F32 | 00003-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 315 | `layers.25.attention_norm1.weight` | [3840] | F32 | 00003-of-00003 | 15,360 | NORM |
| 316 | `layers.25.attention_norm2.weight` | [3840] | F32 | 00003-of-00003 | 15,360 | NORM |
| 317 | `layers.25.feed_forward.w1.weight` | [10240, 3840] | F32 | 00003-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 318 | `layers.25.feed_forward.w2.weight` | [3840, 10240] | F32 | 00003-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 319 | `layers.25.feed_forward.w3.weight` | [10240, 3840] | F32 | 00003-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 320 | `layers.25.ffn_norm1.weight` | [3840] | F32 | 00003-of-00003 | 15,360 | NORM |
| 321 | `layers.25.ffn_norm2.weight` | [3840] | F32 | 00003-of-00003 | 15,360 | NORM |
| 322 | `layers.26.adaLN_modulation.0.bias` | [15360] | F32 | 00003-of-00003 | 61,440 | BIAS |
| 323 | `layers.26.adaLN_modulation.0.weight` | [15360, 256] | F32 | 00003-of-00003 | 15,728,640 | GEMM-WEIGHT |
| 324 | `layers.26.attention.norm_k.weight` | [128] | F32 | 00003-of-00003 | 512 | NORM |
| 325 | `layers.26.attention.norm_q.weight` | [128] | F32 | 00003-of-00003 | 512 | NORM |
| 326 | `layers.26.attention.to_k.weight` | [3840, 3840] | F32 | 00003-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 327 | `layers.26.attention.to_out.0.weight` | [3840, 3840] | F32 | 00003-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 328 | `layers.26.attention.to_q.weight` | [3840, 3840] | F32 | 00003-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 329 | `layers.26.attention.to_v.weight` | [3840, 3840] | F32 | 00003-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 330 | `layers.26.attention_norm1.weight` | [3840] | F32 | 00003-of-00003 | 15,360 | NORM |
| 331 | `layers.26.attention_norm2.weight` | [3840] | F32 | 00003-of-00003 | 15,360 | NORM |
| 332 | `layers.26.feed_forward.w1.weight` | [10240, 3840] | F32 | 00003-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 333 | `layers.26.feed_forward.w2.weight` | [3840, 10240] | F32 | 00003-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 334 | `layers.26.feed_forward.w3.weight` | [10240, 3840] | F32 | 00003-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 335 | `layers.26.ffn_norm1.weight` | [3840] | F32 | 00003-of-00003 | 15,360 | NORM |
| 336 | `layers.26.ffn_norm2.weight` | [3840] | F32 | 00003-of-00003 | 15,360 | NORM |
| 337 | `layers.27.adaLN_modulation.0.bias` | [15360] | F32 | 00003-of-00003 | 61,440 | BIAS |
| 338 | `layers.27.adaLN_modulation.0.weight` | [15360, 256] | F32 | 00003-of-00003 | 15,728,640 | GEMM-WEIGHT |
| 339 | `layers.27.attention.norm_k.weight` | [128] | F32 | 00003-of-00003 | 512 | NORM |
| 340 | `layers.27.attention.norm_q.weight` | [128] | F32 | 00003-of-00003 | 512 | NORM |
| 341 | `layers.27.attention.to_k.weight` | [3840, 3840] | F32 | 00003-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 342 | `layers.27.attention.to_out.0.weight` | [3840, 3840] | F32 | 00003-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 343 | `layers.27.attention.to_q.weight` | [3840, 3840] | F32 | 00003-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 344 | `layers.27.attention.to_v.weight` | [3840, 3840] | F32 | 00003-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 345 | `layers.27.attention_norm1.weight` | [3840] | F32 | 00003-of-00003 | 15,360 | NORM |
| 346 | `layers.27.attention_norm2.weight` | [3840] | F32 | 00003-of-00003 | 15,360 | NORM |
| 347 | `layers.27.feed_forward.w1.weight` | [10240, 3840] | F32 | 00003-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 348 | `layers.27.feed_forward.w2.weight` | [3840, 10240] | F32 | 00003-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 349 | `layers.27.feed_forward.w3.weight` | [10240, 3840] | F32 | 00003-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 350 | `layers.27.ffn_norm1.weight` | [3840] | F32 | 00003-of-00003 | 15,360 | NORM |
| 351 | `layers.27.ffn_norm2.weight` | [3840] | F32 | 00003-of-00003 | 15,360 | NORM |
| 352 | `layers.28.adaLN_modulation.0.bias` | [15360] | F32 | 00003-of-00003 | 61,440 | BIAS |
| 353 | `layers.28.adaLN_modulation.0.weight` | [15360, 256] | F32 | 00003-of-00003 | 15,728,640 | GEMM-WEIGHT |
| 354 | `layers.28.attention.norm_k.weight` | [128] | F32 | 00003-of-00003 | 512 | NORM |
| 355 | `layers.28.attention.norm_q.weight` | [128] | F32 | 00003-of-00003 | 512 | NORM |
| 356 | `layers.28.attention.to_k.weight` | [3840, 3840] | F32 | 00003-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 357 | `layers.28.attention.to_out.0.weight` | [3840, 3840] | F32 | 00003-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 358 | `layers.28.attention.to_q.weight` | [3840, 3840] | F32 | 00003-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 359 | `layers.28.attention.to_v.weight` | [3840, 3840] | F32 | 00003-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 360 | `layers.28.attention_norm1.weight` | [3840] | F32 | 00003-of-00003 | 15,360 | NORM |
| 361 | `layers.28.attention_norm2.weight` | [3840] | F32 | 00003-of-00003 | 15,360 | NORM |
| 362 | `layers.28.feed_forward.w1.weight` | [10240, 3840] | F32 | 00003-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 363 | `layers.28.feed_forward.w2.weight` | [3840, 10240] | F32 | 00003-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 364 | `layers.28.feed_forward.w3.weight` | [10240, 3840] | F32 | 00003-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 365 | `layers.28.ffn_norm1.weight` | [3840] | F32 | 00003-of-00003 | 15,360 | NORM |
| 366 | `layers.28.ffn_norm2.weight` | [3840] | F32 | 00003-of-00003 | 15,360 | NORM |
| 367 | `layers.29.adaLN_modulation.0.bias` | [15360] | F32 | 00003-of-00003 | 61,440 | BIAS |
| 368 | `layers.29.adaLN_modulation.0.weight` | [15360, 256] | F32 | 00003-of-00003 | 15,728,640 | GEMM-WEIGHT |
| 369 | `layers.29.attention.norm_k.weight` | [128] | F32 | 00003-of-00003 | 512 | NORM |
| 370 | `layers.29.attention.norm_q.weight` | [128] | F32 | 00003-of-00003 | 512 | NORM |
| 371 | `layers.29.attention.to_k.weight` | [3840, 3840] | F32 | 00003-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 372 | `layers.29.attention.to_out.0.weight` | [3840, 3840] | F32 | 00003-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 373 | `layers.29.attention.to_q.weight` | [3840, 3840] | F32 | 00003-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 374 | `layers.29.attention.to_v.weight` | [3840, 3840] | F32 | 00003-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 375 | `layers.29.attention_norm1.weight` | [3840] | F32 | 00003-of-00003 | 15,360 | NORM |
| 376 | `layers.29.attention_norm2.weight` | [3840] | F32 | 00003-of-00003 | 15,360 | NORM |
| 377 | `layers.29.feed_forward.w1.weight` | [10240, 3840] | F32 | 00003-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 378 | `layers.29.feed_forward.w2.weight` | [3840, 10240] | F32 | 00003-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 379 | `layers.29.feed_forward.w3.weight` | [10240, 3840] | F32 | 00003-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 380 | `layers.29.ffn_norm1.weight` | [3840] | F32 | 00003-of-00003 | 15,360 | NORM |
| 381 | `layers.29.ffn_norm2.weight` | [3840] | F32 | 00003-of-00003 | 15,360 | NORM |
| 382 | `layers.3.adaLN_modulation.0.bias` | [15360] | F32 | 00001-of-00003 | 61,440 | BIAS |
| 383 | `layers.3.adaLN_modulation.0.weight` | [15360, 256] | F32 | 00001-of-00003 | 15,728,640 | GEMM-WEIGHT |
| 384 | `layers.3.attention.norm_k.weight` | [128] | F32 | 00001-of-00003 | 512 | NORM |
| 385 | `layers.3.attention.norm_q.weight` | [128] | F32 | 00001-of-00003 | 512 | NORM |
| 386 | `layers.3.attention.to_k.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 387 | `layers.3.attention.to_out.0.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 388 | `layers.3.attention.to_q.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 389 | `layers.3.attention.to_v.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 390 | `layers.3.attention_norm1.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 391 | `layers.3.attention_norm2.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 392 | `layers.3.feed_forward.w1.weight` | [10240, 3840] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 393 | `layers.3.feed_forward.w2.weight` | [3840, 10240] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 394 | `layers.3.feed_forward.w3.weight` | [10240, 3840] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 395 | `layers.3.ffn_norm1.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 396 | `layers.3.ffn_norm2.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 397 | `layers.4.adaLN_modulation.0.bias` | [15360] | F32 | 00001-of-00003 | 61,440 | BIAS |
| 398 | `layers.4.adaLN_modulation.0.weight` | [15360, 256] | F32 | 00001-of-00003 | 15,728,640 | GEMM-WEIGHT |
| 399 | `layers.4.attention.norm_k.weight` | [128] | F32 | 00001-of-00003 | 512 | NORM |
| 400 | `layers.4.attention.norm_q.weight` | [128] | F32 | 00001-of-00003 | 512 | NORM |
| 401 | `layers.4.attention.to_k.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 402 | `layers.4.attention.to_out.0.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 403 | `layers.4.attention.to_q.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 404 | `layers.4.attention.to_v.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 405 | `layers.4.attention_norm1.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 406 | `layers.4.attention_norm2.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 407 | `layers.4.feed_forward.w1.weight` | [10240, 3840] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 408 | `layers.4.feed_forward.w2.weight` | [3840, 10240] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 409 | `layers.4.feed_forward.w3.weight` | [10240, 3840] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 410 | `layers.4.ffn_norm1.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 411 | `layers.4.ffn_norm2.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 412 | `layers.5.adaLN_modulation.0.bias` | [15360] | F32 | 00001-of-00003 | 61,440 | BIAS |
| 413 | `layers.5.adaLN_modulation.0.weight` | [15360, 256] | F32 | 00001-of-00003 | 15,728,640 | GEMM-WEIGHT |
| 414 | `layers.5.attention.norm_k.weight` | [128] | F32 | 00001-of-00003 | 512 | NORM |
| 415 | `layers.5.attention.norm_q.weight` | [128] | F32 | 00001-of-00003 | 512 | NORM |
| 416 | `layers.5.attention.to_k.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 417 | `layers.5.attention.to_out.0.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 418 | `layers.5.attention.to_q.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 419 | `layers.5.attention.to_v.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 420 | `layers.5.attention_norm1.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 421 | `layers.5.attention_norm2.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 422 | `layers.5.feed_forward.w1.weight` | [10240, 3840] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 423 | `layers.5.feed_forward.w2.weight` | [3840, 10240] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 424 | `layers.5.feed_forward.w3.weight` | [10240, 3840] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 425 | `layers.5.ffn_norm1.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 426 | `layers.5.ffn_norm2.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 427 | `layers.6.adaLN_modulation.0.bias` | [15360] | F32 | 00001-of-00003 | 61,440 | BIAS |
| 428 | `layers.6.adaLN_modulation.0.weight` | [15360, 256] | F32 | 00001-of-00003 | 15,728,640 | GEMM-WEIGHT |
| 429 | `layers.6.attention.norm_k.weight` | [128] | F32 | 00001-of-00003 | 512 | NORM |
| 430 | `layers.6.attention.norm_q.weight` | [128] | F32 | 00001-of-00003 | 512 | NORM |
| 431 | `layers.6.attention.to_k.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 432 | `layers.6.attention.to_out.0.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 433 | `layers.6.attention.to_q.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 434 | `layers.6.attention.to_v.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 435 | `layers.6.attention_norm1.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 436 | `layers.6.attention_norm2.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 437 | `layers.6.feed_forward.w1.weight` | [10240, 3840] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 438 | `layers.6.feed_forward.w2.weight` | [3840, 10240] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 439 | `layers.6.feed_forward.w3.weight` | [10240, 3840] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 440 | `layers.6.ffn_norm1.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 441 | `layers.6.ffn_norm2.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 442 | `layers.7.adaLN_modulation.0.bias` | [15360] | F32 | 00001-of-00003 | 61,440 | BIAS |
| 443 | `layers.7.adaLN_modulation.0.weight` | [15360, 256] | F32 | 00001-of-00003 | 15,728,640 | GEMM-WEIGHT |
| 444 | `layers.7.attention.norm_k.weight` | [128] | F32 | 00001-of-00003 | 512 | NORM |
| 445 | `layers.7.attention.norm_q.weight` | [128] | F32 | 00001-of-00003 | 512 | NORM |
| 446 | `layers.7.attention.to_k.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 447 | `layers.7.attention.to_out.0.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 448 | `layers.7.attention.to_q.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 449 | `layers.7.attention.to_v.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 450 | `layers.7.attention_norm1.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 451 | `layers.7.attention_norm2.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 452 | `layers.7.feed_forward.w1.weight` | [10240, 3840] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 453 | `layers.7.feed_forward.w2.weight` | [3840, 10240] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 454 | `layers.7.feed_forward.w3.weight` | [10240, 3840] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 455 | `layers.7.ffn_norm1.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 456 | `layers.7.ffn_norm2.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 457 | `layers.8.adaLN_modulation.0.bias` | [15360] | F32 | 00001-of-00003 | 61,440 | BIAS |
| 458 | `layers.8.adaLN_modulation.0.weight` | [15360, 256] | F32 | 00001-of-00003 | 15,728,640 | GEMM-WEIGHT |
| 459 | `layers.8.attention.norm_k.weight` | [128] | F32 | 00001-of-00003 | 512 | NORM |
| 460 | `layers.8.attention.norm_q.weight` | [128] | F32 | 00001-of-00003 | 512 | NORM |
| 461 | `layers.8.attention.to_k.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 462 | `layers.8.attention.to_out.0.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 463 | `layers.8.attention.to_q.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 464 | `layers.8.attention.to_v.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 465 | `layers.8.attention_norm1.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 466 | `layers.8.attention_norm2.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 467 | `layers.8.feed_forward.w1.weight` | [10240, 3840] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 468 | `layers.8.feed_forward.w2.weight` | [3840, 10240] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 469 | `layers.8.feed_forward.w3.weight` | [10240, 3840] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 470 | `layers.8.ffn_norm1.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 471 | `layers.8.ffn_norm2.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 472 | `layers.9.adaLN_modulation.0.bias` | [15360] | F32 | 00002-of-00003 | 61,440 | BIAS |
| 473 | `layers.9.adaLN_modulation.0.weight` | [15360, 256] | F32 | 00002-of-00003 | 15,728,640 | GEMM-WEIGHT |
| 474 | `layers.9.attention.norm_k.weight` | [128] | F32 | 00001-of-00003 | 512 | NORM |
| 475 | `layers.9.attention.norm_q.weight` | [128] | F32 | 00001-of-00003 | 512 | NORM |
| 476 | `layers.9.attention.to_k.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 477 | `layers.9.attention.to_out.0.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 478 | `layers.9.attention.to_q.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 479 | `layers.9.attention.to_v.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 480 | `layers.9.attention_norm1.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 481 | `layers.9.attention_norm2.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 482 | `layers.9.feed_forward.w1.weight` | [10240, 3840] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 483 | `layers.9.feed_forward.w2.weight` | [3840, 10240] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 484 | `layers.9.feed_forward.w3.weight` | [10240, 3840] | F32 | 00002-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 485 | `layers.9.ffn_norm1.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 486 | `layers.9.ffn_norm2.weight` | [3840] | F32 | 00002-of-00003 | 15,360 | NORM |
| 487 | `noise_refiner.0.adaLN_modulation.0.bias` | [15360] | F32 | 00001-of-00003 | 61,440 | BIAS |
| 488 | `noise_refiner.0.adaLN_modulation.0.weight` | [15360, 256] | F32 | 00001-of-00003 | 15,728,640 | GEMM-WEIGHT |
| 489 | `noise_refiner.0.attention.norm_k.weight` | [128] | F32 | 00001-of-00003 | 512 | NORM |
| 490 | `noise_refiner.0.attention.norm_q.weight` | [128] | F32 | 00001-of-00003 | 512 | NORM |
| 491 | `noise_refiner.0.attention.to_k.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 492 | `noise_refiner.0.attention.to_out.0.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 493 | `noise_refiner.0.attention.to_q.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 494 | `noise_refiner.0.attention.to_v.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 495 | `noise_refiner.0.attention_norm1.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 496 | `noise_refiner.0.attention_norm2.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 497 | `noise_refiner.0.feed_forward.w1.weight` | [10240, 3840] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 498 | `noise_refiner.0.feed_forward.w2.weight` | [3840, 10240] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 499 | `noise_refiner.0.feed_forward.w3.weight` | [10240, 3840] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 500 | `noise_refiner.0.ffn_norm1.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 501 | `noise_refiner.0.ffn_norm2.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 502 | `noise_refiner.1.adaLN_modulation.0.bias` | [15360] | F32 | 00001-of-00003 | 61,440 | BIAS |
| 503 | `noise_refiner.1.adaLN_modulation.0.weight` | [15360, 256] | F32 | 00001-of-00003 | 15,728,640 | GEMM-WEIGHT |
| 504 | `noise_refiner.1.attention.norm_k.weight` | [128] | F32 | 00001-of-00003 | 512 | NORM |
| 505 | `noise_refiner.1.attention.norm_q.weight` | [128] | F32 | 00001-of-00003 | 512 | NORM |
| 506 | `noise_refiner.1.attention.to_k.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 507 | `noise_refiner.1.attention.to_out.0.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 508 | `noise_refiner.1.attention.to_q.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 509 | `noise_refiner.1.attention.to_v.weight` | [3840, 3840] | F32 | 00001-of-00003 | 58,982,400 | GEMM-WEIGHT |
| 510 | `noise_refiner.1.attention_norm1.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 511 | `noise_refiner.1.attention_norm2.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 512 | `noise_refiner.1.feed_forward.w1.weight` | [10240, 3840] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 513 | `noise_refiner.1.feed_forward.w2.weight` | [3840, 10240] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 514 | `noise_refiner.1.feed_forward.w3.weight` | [10240, 3840] | F32 | 00001-of-00003 | 157,286,400 | GEMM-WEIGHT |
| 515 | `noise_refiner.1.ffn_norm1.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 516 | `noise_refiner.1.ffn_norm2.weight` | [3840] | F32 | 00001-of-00003 | 15,360 | NORM |
| 517 | `t_embedder.mlp.0.bias` | [1024] | F32 | 00001-of-00003 | 4,096 | EMBED |
| 518 | `t_embedder.mlp.0.weight` | [1024, 256] | F32 | 00001-of-00003 | 1,048,576 | EMBED |
| 519 | `t_embedder.mlp.2.bias` | [256] | F32 | 00001-of-00003 | 1,024 | EMBED |
| 520 | `t_embedder.mlp.2.weight` | [256, 1024] | F32 | 00001-of-00003 | 1,048,576 | EMBED |
| 521 | `x_pad_token` | [1, 3840] | F32 | 00001-of-00003 | 15,360 | OTHER |

## Appendix B. Non-GEMM class rosters

NORM (204 tensors, 530,944 params): 6 per block x 34 blocks - `attention.norm_q.weight` [128],
`attention.norm_k.weight` [128], `attention_norm1.weight` [3840], `attention_norm2.weight` [3840],
`ffn_norm1.weight` [3840], `ffn_norm2.weight` [3840]. Params: 34 x (128+128+4x3840) = 530,944.

EMBED (10 tensors): `all_x_embedder.2-1.weight` [3840, 64], `all_x_embedder.2-1.bias` [3840],
`cap_embedder.0.weight` [2560], `cap_embedder.1.weight` [3840, 2560], `cap_embedder.1.bias` [3840],
`cap_pad_token` [1, 3840], `t_embedder.mlp.0.weight` [1024, 256], `t_embedder.mlp.0.bias` [1024],
`t_embedder.mlp.2.weight` [256, 1024], `t_embedder.mlp.2.bias` [256].

OTHER (1 tensor): `x_pad_token` [1, 3840].

ROPE: empty.

Class totals recombined: 6,143,262,720 (GEMM) + 495,424 (BIAS) + 530,944 (NORM) + 10,615,808 (EMBED)
+ 3,840 (OTHER) = 6,154,908,736 params. Note that the 34 BIAS-class tensors exclude the 4 bias
vectors inside embedder modules (`all_x_embedder.2-1.bias`, `cap_embedder.1.bias`,
`t_embedder.mlp.0.bias`, `t_embedder.mlp.2.bias`) because the EMBED name rule is applied first.
