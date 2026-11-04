# FINDINGS — one Z-Image-Turbo DiT block extracted to BlockFP16 (`bfp16ebs8`)

Task: extract the real weights of ONE complete Z-Image-Turbo DiT transformer
block so an NPU kernel assembly can be verified against the actual model.
Everything below is **measured** in this session by
`zimage-bfp16-npu/_scratch/weights/extract.py` (reproducible; re-run it to
regenerate every number here).

## 0. Deliverables (all created under `_scratch/weights/` only)

| file | what it is |
|---|---|
| `extract.py` | the extraction/packing script (imports the verified `pack_bfp16.py`, does not reimplement the encoder) |
| `manifest.json` | name / fp32 shape / packed length / `dim, m, k, n` / `tile_rows` per tensor + verification numbers |
| `block0.npz` | per-tensor bytes: `<name>` = logical bfp16ebs8 (all 15 tensors), `<name>.f32` = exact fp32 of the 1-D tensors, `<name>.vendor` = vendor coalesced-region bytes where they differ from the logical packing |
| `block0.coalesced.bin` | 208,035,840 B — the block's bytes exactly as they sit in a vendor `.rai` `coalesed_weights` buffer |
| `FINDINGS.md` | this file |

Block extracted: **`layers.0`** (all 30 `layers.N` blocks have identical
shapes; the byte-identity verification in §3 was run on `layers.10`/`layers.11`
because that is the pair a vendor `.rai` exists for).

## 1. What the block owns — 15 tensors, nothing else

Read from the shard header
(`diffusion_pytorch_model-00001-of-00003.safetensors`, header offsets +
`np.frombuffer`; `safetensors.get_tensor()` raises 0xC0000005 here). All F32.
Sorted-index order of the checkpoint:

| brief name (from the task) | real checkpoint key | fp32 shape | dim | m | k | n | tile_rows | logical `bfp16ebs8` B | vendor region B | role |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `attn.to_q` | `layers.0.attention.to_q.weight` | [3840, 3840] | -1 | 96 | 3840 | 3840 | 48 | 16,588,800 | = logical | Q proj |
| `attn.to_k` | `layers.0.attention.to_k.weight` | [3840, 3840] | -1 | 96 | 3840 | 3840 | 48 | 16,588,800 | = logical | K proj |
| `attn.to_v` | `layers.0.attention.to_v.weight` | [3840, 3840] | -1 | 96 | 3840 | 3840 | 48 | 16,588,800 | = logical | V proj |
| `attn.to_out` | `layers.0.attention.to_out.0.weight` | [3840, 3840] | -1 | 96 | 3840 | 3840 | 48 | 16,588,800 | = logical | out proj |
| `ff.net.0.proj` (gate half) | `layers.0.feed_forward.w1.weight` | [10240, 3840] | -1 | 96 | 3840 | 10240 | 32 | 44,236,800 | = logical | SwiGLU gate |
| `ff.net.0.proj` (up half) | `layers.0.feed_forward.w3.weight` | [10240, 3840] | -1 | 96 | 3840 | 10240 | 32 | 44,236,800 | = logical | SwiGLU up |
| `ff.net.2` | `layers.0.feed_forward.w2.weight` | [3840, 10240] | -1 | 96 | 10240 | 3840 | 32 | 44,236,800 | = logical | down proj |
| `norm1.linear`/`norm2.linear` **do not exist** | `layers.0.adaLN_modulation.0.weight` | [15360, 256] | -1 | 1 | 256 | 15360 | 64 | 4,423,680 | **4,546,560** (padded records + bias prefixes) | adaLN modulation proj |
| (bias) | `layers.0.adaLN_modulation.0.bias` | [15360] | 0 | 1 | — | 15360 | — | 17,280 | **inside the adaLN region prefixes** (bf16) | adaLN bias |
| `norm1` gamma | `layers.0.attention_norm1.weight` | [3840] | 0 | — | 3840 | — | — | 4,320 | **737,280** (bf16 tiled) | RMSNorm gain, attn in |
| `norm2` gamma | `layers.0.attention_norm2.weight` | [3840] | 0 | — | 3840 | — | — | 4,320 | **737,280** (bf16 tiled) | RMSNorm gain, attn out |
| (not in brief) | `layers.0.ffn_norm1.weight` | [3840] | 0 | — | 3840 | — | — | 4,320 | **737,280** (bf16 tiled) | RMSNorm gain, FFN in |
| (not in brief) | `layers.0.ffn_norm2.weight` | [3840] | 0 | — | 3840 | — | — | 4,320 | **737,280** (bf16 tiled) | RMSNorm gain, FFN out |
| (not in brief) | `layers.0.attention.norm_q.weight` | [128] | 0 | — | 128 | — | — | 144 | **737,280** (bf16 tiled) | per-head-dim Q norm |
| (not in brief) | `layers.0.attention.norm_k.weight` | [128] | 0 | — | 128 | — | — | 144 | **737,280** (bf16 tiled) | per-head-dim K norm |

* logical totals: **203,524,128 B** (15 tensors) — every length is exactly
  `ceil(numel_packed_dim/8) * 9 * other_dims`.
* vendor coalesced block total: **208,035,840 B** (= 2 × 208,035,840 for both
  blocks of the `layers_10_12` shard = the artifact's 416,071,680 B payload).
* params per block: 180,910,336 (matches `docs/WEIGHT-MAP.md` §4 exactly).

**Nothing else belongs to the block.** Checked, none present: biases on
`to_q/to_k/to_v/to_out/w1/w2/w3` (all Linear(bias=False); the ONLY bias in the
block is `adaLN_modulation.0.bias`), no per-block RoPE tables (cos/sin are
graph ACTIVATIONS `[1, S, 64]`, RoPE is computed — `input_output_shape.json`:
hidden `[1 96 3840]`, cos/sin `[1 96 64]`, adaln `[1 256]`), no extra norm
parameters. Block param count cross-check: 3840×3840×4 + 10240×3840×2 +
3840×10240 + 15360×256 + 15360 + 3840×4 + 128×2 = 180,910,336 ✓.

## 2. Where the model's naming did NOT match the brief

| brief expectation | reality in the checkpoint | impact |
|---|---|---|
| `attn.to_q` / `to_k` / `to_v` / `to_out` | `attention.to_q` / `to_k` / `to_v` / `attention.to_out.**0**.weight` | prefix `attn.` → `attention.`; `to_out` has a `.0.` infix (it is `Sequential`-wrapped) |
| `ff.net.0.proj` as ONE fused w1/w3 tensor | **no fused tensor exists**: `feed_forward.w1.weight` [10240,3840] and `feed_forward.w3.weight` [10240,3840] are separate SwiGLU gate/up tensors | fusion is a consumer-side concat. MEASURED: `pack(vstack(w1,w3), tile_rows=32) == pack(w1,32) ++ pack(w3,32)` (0 differing bytes), because 10240 % 32 == 0 and the tiles do not cross the boundary. So a fused [20480, 3840] tensor is byte-identical to the concatenation of the two regions I ship |
| `ff.net.2` | `feed_forward.w2.weight` [3840, 10240] | rename only |
| `norm1.linear` / `norm2.linear` (per-block adaLN scale/shift projections) | **absent.** The modulation comes from ONE fused projection `adaLN_modulation.0.{weight [15360,256], bias [15360]}` fed by the `adaln [1,256]` conditioning vector (15360 = 4 × 3840). The per-channel **gammas** are plain norm weights `attention_norm1/2`, `ffn_norm1/2` [3840] | the "linear that carries gamma" is `adaLN_modulation.0.*`; gamma itself rides in the 4 norm vectors |
| (not mentioned in the brief) | `attention.norm_q.weight`, `attention.norm_k.weight` [128] — qk-norm gains (`qk_norm: true` in config) | owned by the block; added to the manifest |
| naming consistency of adaLN modules | in BLOCKS the Linear is `adaLN_modulation.**0**`, in the host FinalLayer it is `all_final_layer.2-1.adaLN_modulation.**1**` (`Sequential(SiLU, Linear)`) — verified by `zimage-port/verify_adaln_order.py` | do not derive one index from the other |
| `zimage-port/get_params.py` (17-token classifier) | **does not exist anywhere in the workspace** | name classification was done against the proven registry in `docs/WEIGHT-MAP.md` (506/506 names verified three ways) + the shard header |
| `zimage-port/shard_inspect.py` | **does not exist** | used the same technique its description gives: header offsets + `np.frombuffer`, as implemented in `shard_lib.WeightStreamer.read` |
| `skeleton_names_all()` registry | **no such symbol anywhere** | equivalent data: `zimage-port/_probe_out/skeleton_all_v2/weight_map.json` (506 pairs) + `docs/WEIGHT-MAP.md` |
| `verify_rai_rebuild.py` | **does not exist** (closest: `route3_coalesce.py`, `route3_verify_clean.py`, `verify_rai_weights.py`) | re-verified independently in §3 instead |
| "`zimage-port/weights.bfp16` does not exist (deleted)" | **it exists** (`zimage-port\weights.bfp16`, 6,913,965,970 B, and `zimage-port\_probe_out\weights.bfp16`, same size) — see §4 | no regeneration needed, see §4 |

## 3. Verification — measurements

### 3a. Packed length contract (TARGET criterion)

Every one of the 15 tensors: `packed_length == ceil(numel_packed_dim/8) * 9 *
other_dims`, recorded per tensor in `manifest.json`
(`length_matches_formula: true` for all 15). Totals: 203,524,128 B logical.

### 3b. Decode round trip (all 15 tensors, not just one)

Decoding the packed bytes back (inverse permutation + `m * 2^(byte-133)`) and
comparing to the fp32 source:

| tensor | max abs err (fp32) | max err / group-step |
|---|---:|---:|
| adaLN_modulation.0.weight | 7.8125e-03 | **0.500** |
| attention.to_q.weight | 7.8125e-03 | 0.500 |
| attention.to_k.weight | 1.5625e-02 | 0.500 |
| attention.to_v.weight | 7.8125e-03 | 0.500 |
| attention.to_out.0.weight | 7.8125e-03 | 0.500 |
| feed_forward.w1.weight | 1.5381e-02 | 0.500 |
| feed_forward.w3.weight | 1.5625e-02 | 0.500 |
| feed_forward.w2.weight | 3.1250e-02 | 0.500 |
| attention_norm1.weight | 1.5625e-02 | 0.500 |
| attention.norm_q.weight | 7.8125e-03 | 0.500 |
| attention.norm_k.weight | 7.8125e-03 | 0.500 |
| attention_norm2.weight | 7.8125e-03 | 0.500 |
| ffn_norm1.weight | 1.5625e-02 | 0.500 |
| ffn_norm2.weight | 1.5625e-02 | 0.500 |
| adaLN_modulation.0.bias | 1.5381e-02 | 0.500 |

`max err / group-step = 0.500` on all 15: the error is **exactly the contract
ceiling of half a group step** (round-half-away-from-zero attains 0.5 at ties)
— i.e. one bfp16 step scale, never worse. Note the earlier draft of this check
computed the ratio in slot order instead of stream order and reported up to
16x; the fixed verifier gathers the exponent bytes through `_slots`
(`verify_bfp16`, see the docstring there).

### 3c. Byte identity against the real vendor `.rai` (the strongest check)

`extract.py --rai zimage-port/_probe_out/fresh_cache/39e41383350b8c53e90996269478cfb7/39e41383350b8c53e90996269478cfb7.rai`
rebuilds the two `layers_10_12` blocks (`layers.10`, `layers.11`) from the
safetensors checkpoint and compares with the compiled artifact's
`coalesed_weights` payload:

* **416,071,680 / 416,071,680 bytes identical — 0 differing bytes**, all 28
  sub-buffers (16 GEMM + 2 adaLN-with-bias + 12 consts).
* Payload base located by needle candidates + FULL-region acceptance:
  **file offset 282,432** (2,880 candidates; a short probe alone is not
  accepted — same discipline as `pack_bfp16.detect_tile_rows`).
* The artifact's own `external_buffers.buffer1.coalesed_buffers` table
  (`compute_graph.Layer_*_wts_ddr`, `compute_graph.const_ifm_ddr*`) declares
  **28/28 sub-buffer offsets that match my region offsets exactly**
  (e.g. `Layer_15_wts_ddr`@0 = adaLN, `Layer_25_wts_ddr`@4,546,560 = to_q,
  `const_ifm_ddr_4`@407,224,320 = b0 attention_norm1 ...).
* One complete block (the deliverable unit) = 208,035,840 B of those bytes:
  its 8 GEMM/adaLN regions at payload offsets 0..203,612,160 and its 6 consts
  at 407,224,320..411,648,000 (the payload holds both blocks' GEMM parts
  first, then both blocks' consts — one block is NOT contiguous).

Provenance caveat (documented in `docs/WORKLOG.md`, ".rai integrity: settled"):
an earlier byte-identity claim was void because a `.rai` copy had already been
overwritten with rebuild bytes. The surviving artifact used here is the fresh
vendor compile produced by `route3_fresh.py` (`fresh_cache`, 418,881,856 B);
the independent corroboration above (artifact-declared offsets 28/28, full
416 MB identity, per-region identity) is what this report stands on. The
`_probe_out/route3/` evidence folder (incl. `CLEAN_baseline.rai`) was deleted
by someone else DURING this session (it existed at session start and is gone
now), so the earlier baselines can no longer be re-compared.

### 3d. Fused w1/w3 property (for a fused-gate-up kernel)

`pack(vstack(w1,w3), tile_rows=32) == pack(w1,32) ++ pack(w3,32)`: **True**,
0 differing bytes. A consumer wanting a fused `[20480, 3840]` gate/up matrix
can concatenate the two shipped regions without repacking.

### 3e. `weights.bfp16` currency check (the file the parent said was deleted)

`zimage-port\weights.bfp16` (and `zimage-port\_probe_out\weights.bfp16`) both
exist: 6,913,965,970 B file = 87,946 B JSON header + 6,913,878,016 B payload,
506 tensors, `m_tokens: 64`. Sampled 8 block-0 tensors (all four block shapes
+ the 1-D norms/bias) against the CURRENT `pack_bfp16`: **0 differing bytes**
on every one, and its header's `tile_rows` table is exactly {48, 32, 32, 64}.
So this file was built with the corrected quantiser (it is NOT the stale
pre-fix file `KNOWN-ISSUES.md` warns about) and does not need regeneration.
Caveat: 8/506 tensors sampled, not the whole file. 1-D tensors are stored
there as raw fp32 (`packed_bfp16: false`), matching `build_bfp16_weights.py`.

## 4. Vendor layout facts the NPU side needs

Source: `_probe_out/route3/COALESCED-LAYOUT.md` + re-verified byte-exact here
(§3c). The `.rai` `coalesed_weights` region is 28 sub-buffers, in this order:

```
block0: adaLN, to_q, to_k, to_v, to_out, w1, w3, w2   (note: adaLN FIRST,
block1: adaLN, to_q, to_k, to_v, to_out, w1, w3, w2    w3 BEFORE w2)
consts: b0.{attention_norm1, norm_q, norm_k, attention_norm2, ffn_norm1,
        ffn_norm2}, then the same six for b1
```

Per-tensor byte layouts:

1. **GEMM tensors** — `pack_bfp16.pack` exactly: 9-byte groups
   `[exponent byte][8 x int8 mantissa]`, groups of 8 along K, scattered into
   tiles of `tile_rows` rows x 32 K-elements,
   `slot = tile*per_tile + (r//8)*32 + (c//8)*8 + (r%8)`.
   Region size `N*K*9/8` bytes.
2. **adaLN weight+bias** — same slot permutation (`tile_rows=64`) but the
   stream is broken into 960 records of 4,736 B = `[128 B prefix][512 groups
   x 9 B]`; the prefix of record 0 of each 4-record band (64 rows = 1 band)
   holds `bias[64*band : 64*band+64]` as **bf16** (truncation of the fp32
   bits); the other three prefixes are zero. Region size 4,546,560 B =
   4,423,680 (weight) + 122,880 (prefixes). The bias appears NOWHERE ELSE in
   the payload.
3. **1-D consts** — NOT bfp16ebs8 in the vendor payload: **bf16 tiled** to
   exactly 737,280 B per region (`tile(bf16(vec), 737280/(2*len))`, i.e. 96
   copies of a 3840-vector or 2880 copies of a 128-vector).

Because of (2)/(3), I ship **two representations** for the affected tensors:
the logical `bfp16ebs8` bytes (the brief's `ceil(numel/8)*9*others` contract —
what a pure-BlockFP16 kernel wants) and the vendor region bytes (`*.vendor` in
the npz, and what `block0.coalesed.bin` is assembled from). If the NPU kernel
assembly targets the vendor-compatible buffer, feed `block0.coalesced.bin`
(its per-region offsets are in `manifest.json → coalesced_layout.regions`).

`block0.npz` also carries `<name>.f32` for the 7 one-dimensional tensors
(4 RMSNorm gains + 2 qk-norm gains + adaLN bias). Rationale:
`build_bfp16_weights.py` (the whole-model tool) keeps 1-D tensors as raw fp32
because they are not GEMM operands; the bfp16 step on a 3840-vector is large
(7.8e-3..1.6e-2 measured), so a kernel that wants exact gains can use the
`.f32` arrays directly.

## 5. `tile_rows` and the token count M — read before repacking

`tile_rows` is `f(N, K, M)` (compiler subvolume tiling, `L2ToL3TilingAnalysisPass`;
no predictive rule, non-monotonic). Measured values used here (all four block
shapes):

| (N, K) | tile_rows | measured at |
|---|---|---|
| (3840, 3840) | 48 | seq=64 (`pack_bfp16.KNOWN_TILE_ROWS`) AND seq=96 (`COALESCED-LAYOUT.md --sweep`, unique fit: 32/64 leave 1.83M mismatching groups, 48 leaves 0) |
| (10240, 3840) | 32 | same |
| (3840, 10240) | 32 | same |
| (15360, 256) | 64 | same |

The M-dependence is **proven only for (3840, 256)** (M=64 → 48, M=256 → 32,
`docs/HANDOFF-2026-10-12.md`); for the four shapes above, seq=64 and seq=96 agree
and "whether tile_rows ever changes with M is unproven"
(`COALESCED-LAYOUT.md` §5). **If the NPU kernel runs at any other M, re-measure
with `pack_bfp16.detect_tile_rows(W, blob)` (full-region acceptance) before
using these bytes.** The packed weights themselves are M-invariant; only the
slot permutation depends on `tile_rows`. `manifest.json` records
`tile_rows` per tensor plus `m` per tensor (96 for the token GEMMs — the
geometry the vendor measurement lives in, hidden `[1, 96, 3840]`; 1 for the
adaLN GEMM whose input is `adaln [1, 256]`).

## 6. Success criteria

* **MINIMUM — met.** `manifest.json` lists every weight the block owns: all
  15 tensors found in the shard header (none dropped, none invented), with
  fp32 shapes, source dtype F32 and params (sum 180,910,336 = the
  `docs/WEIGHT-MAP.md` figure).
* **TARGET — met.** All 15 tensors are packed to `bfp16ebs8`; for every one,
  `packed_length == ceil(numel_packed_dim/8) * 9 * other_dims`
  (15/15 `length_matches_formula: true`).
* **Decode verification — met, on all 15 tensors.** max abs err per tensor is
  in the table above (7.8e-3 .. 3.1e-2, i.e. one bfp16 step scale); the
  stronger measure, error in units of the group step, is **0.500 (the exact
  half-step ceiling of the round-half-away rule)** on every tensor.
* **Beyond target:** the complete block was reproduced **byte-for-byte inside
  the real vendor `.rai`** (208,035,840 B, part of a 416,071,680/416,071,680
  B identity run), with the artifact's own sub-buffer table agreeing 28/28.

## 7. Scope compliance

* Created ONLY under `zimage-bfp16-npu/_scratch/weights/`:
  `extract.py`, `manifest.json`, `block0.npz` (212,626,930 B),
  `block0.coalesced.bin` (208,035,840 B), `FINDINGS.md`.
* Nothing else was modified or deleted; no compile/NPU work was run.
* The full-model `zimage-port\weights.bfp16` was NOT regenerated (out of my
  write scope); §3e shows it exists and is already consistent with the current
  packer, so regeneration is not required for correctness as sampled.
