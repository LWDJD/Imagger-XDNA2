# WEIGHT-MAP — skeleton.onnx graph inputs → safetensors checkpoint keys

Scope: the 17 exported shards under `zimage-port/_probe_out/skeleton_all_v2/`
(shard plan `shard_lib.shard_specs(blocks_per_shard=2)` from
`zimage-port/shard_lib.py`). Machine-readable companion:
`zimage-port/_probe_out/skeleton_all_v2/weight_map.json` (every ordered pair,
with shape and param count). Derivation/verification script:
`derive_weight_map.py` (run with the `ryzen-ai-1.8.0` conda Python, which has
onnx 1.22 / torch 2.4.1 / diffusers 0.36).

**Result: 506/506 weight inputs map 1:1 to checkpoint keys. No unmapped graph
inputs, no dropped checkpoint tensors, no shape mismatches.**

---

## 1. The mapping rule (proven, not assumed)

```
onnx graph INPUT : blocks.<N>.<tail>
checkpoint key   : <group>.<lo + N>.<tail>
```

where `<group>` and `<lo>` are the shard's `ShardSpec.group` / `ShardSpec.lo`,
`<N>` is the LOCAL block index inside the shard (0 or 1), and `<tail>` is
**byte-identical on both sides** (e.g. `attention.to_out.0.weight`,
`adaLN_modulation.0.bias`). Equivalently: a pure prefix swap
`blocks.<N>.` → `<group>.<lo+N>.`; nothing else changes.

### How it was proved (three-way equality, per shard)

1. **Checkpoint side** — `shard_lib.build_meta_model()` (meta device, 0 bytes)
   + `shard_param_names(model, spec)` produces exactly the names
   `export_skeleton`/`materialize` used to fetch weights. All **506** names
   (15 per adaLN block × 32 blocks + 13 per context block × 2) exist verbatim
   in `diffusion_pytorch_model.safetensors.index.json` (`missing = 0`).
2. **Wrapper side** — the ACTUAL `skeleton.onnx` graph input list read with
   `onnx.load(..., load_external_data=False)`. For every shard,
   `{blocks.<i-lo>.<tail>}` (derived from the names in step 1) equals the set
   of non-activation graph inputs **exactly** — 34 inputs for adaLN shards
   (4 activations + 30 weights), 29 for `context_refiner` (3 + 26).
   Every one of the 506 pairs resolved by **exact** string match
   (`match_types: {"exact": 506}`).
3. **Round trip** — each graph input maps to exactly one checkpoint key and
   each checkpoint key is used exactly once across all 17 shards
   (globally unique; no duplication between shards).

### The `/` ↔ `.` hazard: checked, never fires

`export_skeleton.py:225` falls back to `k.replace("/", ".") == nm` when
resolving initializer names against the state dict. Empirically:

* **0** keys in the safetensors index contain `/` (all 521 are dot-separated);
* **0** `skeleton.onnx` graph input names contain `/` (torch.onnx preserved the
  `blocks.N.…` names verbatim);
* so the fallback is dead code for these 17 shards, and the mapping is the pure
  prefix swap shown above. Nothing renames `attention.to_out.0.weight` or
  `adaLN_modulation.0.*` on the way through.

Activation inputs are NOT weights and are excluded from the map:
`hidden`, `cos`, `sin`, plus `adaln` on adaLN shards.
Shapes: `hidden [1, S, 3840]`, `cos/sin [1, S, 64]`, `adaln [1, 256]`,
with S = 64 (`noise_refiner`), 32 (`context_refiner`), 96 (all `layers_*`).
All activation shapes were verified against this contract.

---

## 2. Ordered pair lists

Pairs below are in `skeleton.onnx` graph-input order (activations first, then
weights). The weight tails are the same 15 (adaLN) or 13 (context) suffixes in
the same order in every shard; only the block index shifts.

### `noise_refiner` (noise_refiner[0:2], S=64, adaLN=yes) — 30 pairs

| # | onnx_input | checkpoint | shape | params |
|---:|---|---|---|---:|
| 1 | blocks.0.attention.norm_q.weight | noise_refiner.0.attention.norm_q.weight | [128] | 128 |
| 2 | blocks.0.attention.norm_k.weight | noise_refiner.0.attention.norm_k.weight | [128] | 128 |
| 3 | blocks.0.attention.to_q.weight | noise_refiner.0.attention.to_q.weight | [3840, 3840] | 14,745,600 |
| 4 | blocks.0.attention.to_k.weight | noise_refiner.0.attention.to_k.weight | [3840, 3840] | 14,745,600 |
| 5 | blocks.0.attention.to_v.weight | noise_refiner.0.attention.to_v.weight | [3840, 3840] | 14,745,600 |
| 6 | blocks.0.attention.to_out.0.weight | noise_refiner.0.attention.to_out.0.weight | [3840, 3840] | 14,745,600 |
| 7 | blocks.0.feed_forward.w1.weight | noise_refiner.0.feed_forward.w1.weight | [10240, 3840] | 39,321,600 |
| 8 | blocks.0.feed_forward.w2.weight | noise_refiner.0.feed_forward.w2.weight | [3840, 10240] | 39,321,600 |
| 9 | blocks.0.feed_forward.w3.weight | noise_refiner.0.feed_forward.w3.weight | [10240, 3840] | 39,321,600 |
| 10 | blocks.0.attention_norm1.weight | noise_refiner.0.attention_norm1.weight | [3840] | 3,840 |
| 11 | blocks.0.ffn_norm1.weight | noise_refiner.0.ffn_norm1.weight | [3840] | 3,840 |
| 12 | blocks.0.attention_norm2.weight | noise_refiner.0.attention_norm2.weight | [3840] | 3,840 |
| 13 | blocks.0.ffn_norm2.weight | noise_refiner.0.ffn_norm2.weight | [3840] | 3,840 |
| 14 | blocks.0.adaLN_modulation.0.weight | noise_refiner.0.adaLN_modulation.0.weight | [15360, 256] | 3,932,160 |
| 15 | blocks.0.adaLN_modulation.0.bias | noise_refiner.0.adaLN_modulation.0.bias | [15360] | 15,360 |
| 16–30 | blocks.1.<tail> (same 15 tails, same order) | noise_refiner.1.<tail> | same | 180,910,336 |

### `context_refiner` (context_refiner[0:2], S=32, adaLN=no) — 26 pairs

Same 13 tails as above **minus** `adaLN_modulation.0.{weight,bias}`:

| # | onnx_input | checkpoint | shape | params |
|---:|---|---|---|---:|
| 1–9 | blocks.0.{attention.norm_q, attention.norm_k, attention.to_q, attention.to_k, attention.to_v, attention.to_out.0, feed_forward.w1, feed_forward.w2, feed_forward.w3}.weight | context_refiner.0.<same tail> | as above | 176,947,200 |
| 10–13 | blocks.0.{attention_norm1, ffn_norm1, attention_norm2, ffn_norm2}.weight | context_refiner.0.<same tail> | [3840] | 15,360 |
| 14–26 | blocks.1.<tail> (same 13 tails) | context_refiner.1.<tail> | same | 176,962,816 |

### `layers_00_02` … `layers_28_30` (layers[lo:lo+2], S=96, adaLN=yes) — 30 pairs each

Identical 15-tail structure as `noise_refiner`; the ONLY difference is the
index offset. Explicit offsets (all verified against the skeleton inputs):

| shard | group | lo | hi | blocks.0 → | blocks.1 → |
|---|---|---:|---:|---|---|
| layers_00_02 | layers | 0 | 2 | layers.0.* | layers.1.* |
| layers_02_04 | layers | 2 | 4 | layers.2.* | layers.3.* |
| layers_04_06 | layers | 4 | 6 | layers.4.* | layers.5.* |
| layers_06_08 | layers | 6 | 8 | layers.6.* | layers.7.* |
| layers_08_10 | layers | 8 | 10 | layers.8.* | layers.9.* |
| layers_10_12 | layers | 10 | 12 | layers.10.* | layers.11.* |
| layers_12_14 | layers | 12 | 14 | layers.12.* | layers.13.* |
| layers_14_16 | layers | 14 | 16 | layers.14.* | layers.15.* |
| layers_16_18 | layers | 16 | 18 | layers.16.* | layers.17.* |
| layers_18_20 | layers | 18 | 20 | layers.18.* | layers.19.* |
| layers_20_22 | layers | 20 | 22 | layers.20.* | layers.21.* |
| layers_22_24 | layers | 22 | 24 | layers.22.* | layers.23.* |
| layers_24_26 | layers | 24 | 26 | layers.24.* | layers.25.* |
| layers_26_28 | layers | 26 | 28 | layers.26.* | layers.27.* |
| layers_28_30 | layers | 28 | 30 | layers.28.* | layers.29.* |

The full ordered 506-pair enumeration with shapes and param counts is in
`weight_map.json` under `shards.<name>.pairs`.

---

## 3. Unmapped / mismatched names

**None.** Concretely:

| check | result |
|---|---|
| graph inputs not mapping to a checkpoint key | 0 (all 506 map; activations excluded: hidden/cos/sin/adaln) |
| checkpoint tensors in a shard's range with no graph input (dropped weights) | 0 (all 506 block-group keys covered) |
| checkpoint keys outside every shard's range | 0 of 506 block-group keys; the other 15 keys are host-side by design |
| shape mismatches onnx vs `_analysis.json` (safetensors header scan) | 0 |
| names needing the `/`↔`.` fallback | 0 (`match_types: {"exact": 506}`) |
| duplicate use of one checkpoint key | 0 (each used exactly once across all shards) |

Host-side keys (correctly NOT graph inputs; `11,852,352` params): the 10
EMBED tensors (`all_x_embedder.2-1.*`, `cap_embedder.*`, `t_embedder.mlp.*`),
`x_pad_token` (OTHER), `cap_pad_token`, and `all_final_layer.2-1.*` (4
tensors). Note the inventory's quirk stands: the host prefixes are
`all_x_embedder.2-1` / `all_final_layer.2-1`, not plain `x_embedder` /
`final_layer` — irrelevant for the shard map but relevant to any host loader.

---

## 4. Parameter counts

Per-block: adaLN block = 15 tensors / **180,910,336** params
(GEMM 180,879,360 + norms 15,616 + adaLN bias 15,360);
context block = 13 tensors / **176,962,816** params (GEMM 176,947,200 + norms
15,616).

| shard | group[lo:hi] | tensors | params | params fp32 (B) |
|---|---|---:|---:|---:|
| noise_refiner | noise_refiner[0:2] | 30 | 361,820,672 | 1,447,282,688 |
| context_refiner | context_refiner[0:2] | 26 | 353,925,632 | 1,415,702,528 |
| layers_00_02 | layers[0:2] | 30 | 361,820,672 | 1,447,282,688 |
| layers_02_04 | layers[2:4] | 30 | 361,820,672 | 1,447,282,688 |
| layers_04_06 | layers[4:6] | 30 | 361,820,672 | 1,447,282,688 |
| layers_06_08 | layers[6:8] | 30 | 361,820,672 | 1,447,282,688 |
| layers_08_10 | layers[8:10] | 30 | 361,820,672 | 1,447,282,688 |
| layers_10_12 | layers[10:12] | 30 | 361,820,672 | 1,447,282,688 |
| layers_12_14 | layers[12:14] | 30 | 361,820,672 | 1,447,282,688 |
| layers_14_16 | layers[14:16] | 30 | 361,820,672 | 1,447,282,688 |
| layers_16_18 | layers[16:18] | 30 | 361,820,672 | 1,447,282,688 |
| layers_18_20 | layers[18:20] | 30 | 361,820,672 | 1,447,282,688 |
| layers_20_22 | layers[20:22] | 30 | 361,820,672 | 1,447,282,688 |
| layers_22_24 | layers[22:24] | 30 | 361,820,672 | 1,447,282,688 |
| layers_24_26 | layers[24:26] | 30 | 361,820,672 | 1,447,282,688 |
| layers_26_28 | layers[26:28] | 30 | 361,820,672 | 1,447,282,688 |
| layers_28_30 | layers[28:30] | 30 | 361,820,672 | 1,447,282,688 |
| **sum (17 shards)** | | **506** | **6,143,056,384** | **24,572,225,536** |

Cross-checks (all delta 0):

| figure | value | source | delta |
|---|---:|---|---:|
| sum over 17 shards | 6,143,056,384 | derived here (onnx shapes, cross-checked vs `_analysis.json` numerals) | — |
| block-group keys in index | 6,143,056,384 | `_analysis.json` over `layers.*`/`noise_refiner.*`/`context_refiner.*` (506 keys) | 0 |
| host-side params | 11,852,352 | 15 non-block keys (EMBED 10,615,808 + OTHER 3,840 + final_layer 1,232,704) | — |
| **whole model** | **6,154,908,736** | 6,143,056,384 + 11,852,352 vs ZIMAGE-TENSOR-INVENTORY.md | **0** |
| GEMM-only, whole model | 6,143,262,720 | `_analysis.json` cls=GEMM-WEIGHT vs ZIMAGE-TENSOR-INVENTORY.md | **0** |
| GEMM-only, in shards | 6,142,033,920 | = 6,143,262,720 − 1,228,800 (final-layer GEMMs are host-side) | 0 |
| in-shard by class | GEMM 6,142,033,920 + NORM 530,944 + BIAS 491,520 | sum = 6,143,056,384 | 0 |
| manifest byte totals | 1,447,282,688 / 1,415,702,528 | `skeleton_all_v2/manifest.json` `was_initializer_bytes` = 4 B × per-shard params | 0 |

So the 17 shards carry **99.807%** of the model
(6,143,056,384 / 6,154,908,736); the remainder is the 15 host-side tensors.

---

## 5. How to use the map at load time

For shard `S` with spec `(group, lo)`, feed `skeleton.onnx` from
`weight_map.json` → `shards[S].pairs`:

```
feed[pair.onnx_input] = safetensors[pair.checkpoint]      # fp32, shape pair.shape
```

plus the activations (`hidden`, `cos`, `sin`, `adaln` where present). All
tensors are F32 in the checkpoint (`index metadata.total_size =
24,619,634,944 = 6,154,908,736 × 4`), so no dtype conversion is needed.
