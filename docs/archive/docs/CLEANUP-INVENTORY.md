# DISK CLEANUP INVENTORY — `<workspace>`

**Survey mode: READ-ONLY.** Nothing was deleted, moved, or modified. This file is the only file created.
**Survey time:** 2026/10/11 ~20:25–20:35 (local). **Units:** sizes shown as **GiB** (bytes / 2^30, i.e. PowerShell `1GB`), with exact bytes where it matters. "GB" in Windows Explorer ≈ GiB.

> ⚠️ **Active-process warning:** `_probe_out\rootcure\` was being written *while this survey ran* (it changed from 3 files/0.44 GiB → 4 files → 6 files/0.11 GiB between scans at 20:25–20:29), and `original-info-signature.txt` / `original-model-signature.txt` appeared at the workspace root at 20:24. **Some experiment is running right now. Do not delete anything it owns (rootcure), and re-check sizes before any actual cleanup.**

---

## 1. Space context

- Drive D: 652.9 GB total, **12.0 GB free** (user-stated; sandbox denied `fsutil`/`Get-PSDrive` verification).
- Needed to regenerate NPU artifacts: **~22.93 GiB ONNX shards + up to 15 GiB VAIP compile cache ≈ 37.9 GiB**. Does not fit today.
- **Deleting the 15 items ranked in §6 frees 31,671,935,714 B = 29.50 GiB (31.67 GB decimal)** → D: free becomes ~41.5 GiB, enough for the 37.9 GiB regeneration with ~3.5 GiB margin. Adding the optional log groups (§7) or `single` (§5.4, not recommended) widens the margin.

---

## 2. Total bytes per top-level area

| Area | Bytes | GiB | Files | Notes |
|---|---:|---:|---:|---|
| `zimage-port\_probe_out\` (whole tree) | 40,112,247,031 | 37.36 | 12,130 | 214 subdirs + 227 loose log files; the big one |
| `zimage-port\_models\` | 32,935,930,734 | 30.67 | 46 | **IRREPLACEABLE model data — never delete** |
| `zimage-port\` top-level loose files (*.py/*.log/*.txt/*.md + misc) | 198,701,298 | 0.19 | 2,708 | 868 .py (4.28 MB) + 208 .md (1.89 MB) = code/docs KEEP; 660 .log (98.9 MiB) + 959 .txt (84.7 MiB) = probe logs |
| `zimage-port\.git\` | ~23,000,000 | 0.02 | 2,704 | git metadata, KEEP |
| `zimage-port\tmp*` (41 dirs) | **0** | 0 | 0 | **ALL EMPTY** — disposable shells |
| root `tmp*` (14 dirs) | **0** | 0 | 0 | **ALL EMPTY** — disposable shells |
| root top-level loose files | 11,747,655 | 0.01 | ~76 | largest: `build-vulkan.log` 10.2 MiB |
| (context) other root dirs: `llama.cpp-0.5.0` 0.82 GiB, `k2h-llama.cpp` 0.69 GiB, `models\K2-Horizon-1B-Q4_K_M.gguf` 0.62 GiB, `research` 0.10 GiB, rest < 0.1 GiB | ~2.4 GiB | | | outside survey scope; sizes listed for context |

`_probe_out` largest subdirectories (all 214 measured; top of the tail):

| Subdir | GiB | Files | Last modified |
|---|---:|---:|---|
| `zimage_loop` | 16.801 | 8,377 | 2026/10/11 05:53 |
| `bfp16_weights_v2` | 6.449 | 2 | 2026/10/09 16:37 |
| `dd_iter` | 4.044 | 10 | 2026/10/10 20:25 |
| `single` | 2.053 | 42 | 2026/10/09 13:08 |
| `merge_probe` | 1.348 | 2 | 2026/10/09 18:54 |
| `p2_fold` | 1.348 | 1 | 2026/10/09 17:29 |
| `p2_wholegraph` | 0.759 | 130 | 2026/10/09 17:42 |
| `bfp16_deadlock` | 0.512 | 8 | 2026/10/09 19:51 |
| `gl_work` | 0.411 | 257 | 2026/10/10 20:33 |
| `rootcure` | 0.11* | 6* | 2026/10/11 20:29 (*actively changing) |
| `dd85` | 0.255 | 12 | 2026/10/10 17:38 |
| `dd81` | 0.170 | 7 | 2026/10/10 17:36 |
| `dd_swap` | 0.167 | 8 | 2026/10/10 10:22 |
| `subv_search` | 0.164 | 296 | 2026/10/10 09:00 |
| `waic_gate2` | 0.148 | 63 | 2026/10/10 09:00 |
| `dd_swap2` | 0.120 | 37 | 2026/10/10 10:25 |
| `dd_live` | 0.111 | 15 | 2026/10/10 10:30 |
| (all other ~197 subdirs combined) | ~1.3 | ~2,600 | — |

Dominant file types in `_probe_out`: `.const` 3,451 files / 15.6 GiB (VAIP compile-cache weight blobs), `.bfp16` 1 / 6.0 GiB, `.onnx` 66 / 4.5 GiB, `.bin` 140 / 2.6 GiB, `.zip` 114 / 2.1 GiB, `.weight` 36 / 1.9 GiB, `.data` 42 / 1.2 GiB.

---

## 3. Model data that survives (determines what can be regenerated)

`zimage-port\_models\Z-Image-Turbo\` is **PRESENT and COMPLETE**:

| Component | Files | Bytes | Verified against |
|---|---:|---:|---|
| `transformer\diffusion_pytorch_model-0000{1,2,3}-of-00003.safetensors` | 3 | 24,619,690,888 (22.93 GiB) | index `total_size` 24,619,634,944 ✓ (diff = safetensors headers). Sizes 9,511.66 / 9,511.68 / 4,455.84 MiB. Plus `config.json`, `*.index.json`, `SHA256SUMS.modelscope.json`. Last written 2026/10/09 12:41–13:01 (re-verified/re-downloaded from ModelScope; `_models\_raw\*.progress` + `SHA256SUMS.json` are the only download remnants) |
| `text_encoder\model-0000{1,2,3}-of-00003.safetensors` | 3 | 8,044,982,000 (7.49 GiB) | index `total_size` 8,044,936,192 ✓ |
| `vae\diffusion_pytorch_model.safetensors` | 1 | 159.9 MiB | present |
| `tokenizer\`, `scheduler\`, `model_index.json`, `host_tensors.safetensors` (45.2 MiB) | ~10 | ~62 MiB | present |
| **Z-Image-Turbo total** | 46 | **32.9 GB (30.67 GiB)** | everything except `transformer\` = 7.71 GiB ✓ matches expectation |

**Conclusion:** the transformer safetensors (the input for `export_shards_fp32.py`, `encode_zimage_bfp16_v2.py`, and all shard/VAIP regeneration) survive intact. All IRREPLACEABLE data in this workspace is inside `_models\` (plus `models\K2-Horizon-1B-Q4_K_M.gguf`, a downloaded GGUF, outside the surveyed paths). **Everything else in the survey is derived data and is at least theoretically regenerable.**

Also relevant: `_probe_out\shards_i1024_c32\` is where the ONNX shards get exported and it is **effectively empty** — only `io.npz` (31.46 MiB) and an **empty** `layers_00_02\` directory. The ~22.93 GiB shard set is already gone (previously deleted), which is exactly why it must be regenerated. Keep `io.npz` (probe reference).

---

## 4. The workspace author's own cleanup guidance (found on disk)

- `zimage-port\_gen_cleanlist.py` — a read-only cleanup-list generator (writes `_CLEANUP-LIST.txt`, deletes nothing). It encodes the author's classification:
  - **NEVER delete:** `shards*`, `vaip_cache*`, `single`, `_probe_out\aiert_build`, `cast_nb3`, `dd103t`, `plugins_clean`, `zimage_loop\{out_op1, out_op2, out_fix2, out_r1, t_a4, t_w1, g_ca1, out_nomul2, out_nomul2_keep}`, all `.md` docs, implementation code, export scripts, `_out_512\` reference images.
  - **Cleanup candidates per author:** `bfp16_weights_v2` (regenerable via `encode_zimage_bfp16_v2.py`, slow), `dd_iter` (regenerable via `_dd4_iter.py` / `_dd5_vi.py`, slow), `merge_probe` (probe output, conclusions recorded), and all `zimage_loop` subdirs except the KEEP set (regenerable via `partition`, recipe in `_GL2-FINAL-HANDOVER.md`).
- This inventory independently confirms those classifications (see §5) and adds the ones the author's list predates (`gl_work`, `dd8x`, `dd_swap*`, `subv_search`, `waic_gate2`, `dd_live`, `bfp16_deadlock`, `p2_*`).

---

## 5. Detailed items (>100 MB) — content, recoverability, recommendation

### 5.1 `_probe_out\zimage_loop\` — 16.801 GiB, 8,377 files, 45 subdirs (modified 2026/10/10–10/11)
**What it is:** the DD-route ("DD 路线 A–L") iteration loop. Two kinds of subdirs:
- `out_*` experiments: `partition.log` + `replaced.onnx` (0.15 MiB) + `replaced.onnx.data` (~62 MiB patched external-data graph) + `cache\` (VAIP compile cache: `*_meta.json` + `*.const` weight blobs, 16–43 MiB each).
- `gx1..gx3`, `gxd1`, `g_ca1`, `g_g2`, `g_g3`, `t_a4`, `t_w1`: full-graph variants holding `dd_metastate_*.ddbundle/.fconst/.state` triplets (per-op metastate bundles, ~105 MiB per op) + `cache\`.
- `DONE.json` = `{"round": 1, "out": "...\\out_r1", "added": []}` — loop-harness checkpoint: round 1 emitted `out_r1` and added nothing.
**Recoverability:** **REGENERABLE-SLOW** — each subdir is one `partition` run of a graph variant; the recipes are documented in `zimage-port\_GL2-FINAL-HANDOVER.md`. The whole set is ~1.5 days of accumulated iterations; any single one costs minutes to tens of minutes.
**Recommendation:** **DELETABLE-CANDIDATE for the 36 non-KEEP subdirs = 14,486,073,968 B (13.491 GiB), 6,350 files.** KEEP the 8 author-KEEP baselines (`out_op1, out_fix2, out_r1, t_a4, t_w1, g_ca1, out_nomul2, out_nomul2_keep`) = 3.55 GiB, 2,020 files. ⚠️ `zimage_loop\plugins\` (85 MiB, 10 files) is inside the delete set per the author's rule but see §8 (unsure).

### 5.2 `_probe_out\bfp16_weights_v2\` — 6.449 GiB, 2 files (2026/10/09 16:37)
**What it is:** exactly 2 files:
- `weights.bfp16` — 6,924,272,328 B: ALL 521 transformer tensors encoded into the official-IR interleaved BFP16 layout (B as [K,N], grouped along N by 8; 8 mantissa bytes + 1 exp byte = 1.125 B/param; 6,154,908,736 params).
- `manifest.json` — full layout spec + per-tensor offsets/shapes/exp_groups (the format is self-documenting).
**Producer:** `zimage-port\encode_zimage_bfp16_v2.py` (+ `_bfp16_pack.py`), deterministic pure-NumPy, reads `_models\Z-Image-Turbo\transformer\*.safetensors` (survives). Header doc says v1 layout was wrong; v2 matches the official layout.
**Recoverability:** **REGENERABLE-SLOW** — NOT the only copy in any meaningful sense: the encoder, its manifest, and the source weights all survive; it is a deterministic re-encode (author: "耗时较长" / takes a while; expect tens of minutes of CPU for 6.15B params). It is **not an input** to ONNX-shard/VAIP regeneration.
**Recommendation:** **DELETABLE-CANDIDATE** (6.449 GiB). Precise classification: *derived weight encoding, fully regenerable* — safe to delete if you can afford one encoder run later.

### 5.3 `_probe_out\dd_iter\` — 4.044 GiB, 10 files (2026/10/10 09:37–20:25)
**What it is:** `fold\folded.onnx` (1.38 GiB), `fold\folded_vi.onnx` (1.38 GiB), `fold\weights.bin` (1.38 GiB merged weight blob), `fold\model.onnx` (0.13 MiB), plus tiny `fold\_fold.py`, `fix_vi.py`, and three `partition.log`s / `cache\` stubs (`out_raw`, `out_folded`, `out_vi`).
**Recoverability:** **REGENERABLE-SLOW** — author: rebuild via `_dd4_iter.py` / `_dd5_vi.py` (slow).
**Recommendation:** **DELETABLE-CANDIDATE** — but keep the two tiny `.py` files (`fold\_fold.py`, `fix_vi.py`) before deleting the three 1.38 GiB artifacts.

### 5.4 `_probe_out\single\` — 2.053 GiB, 42 files (2026/10/09 13:08)
**What it is:** `layers_00_03\` — per-tensor fp32 dumps of transformer blocks 0–2 (9 × 150 MiB `feed_forward.w{1,2,3}.weight`, 12 × 56.25 MiB attention `to_{q,k,v,out}`, adaLN 15 MiB each, norms), plus `model.onnx` (0.14 MiB) and `io.npz` (31.46 MiB probe inputs + CPU reference outputs).
**Producer:** `zimage-port\export_shards_fp32.py` (writes the `.weight` sidecars and `io.npz` via `np.savez`) from the surviving transformer safetensors.
**Recoverability:** **REGENERABLE-FAST** in principle (minutes; script + source on disk). **BUT** it is the live fixture referenced by 10 `p2_probe_*.py` scripts and is on the author's NEVER-delete list.
**Recommendation:** **KEEP.** Only 2 GiB; deleting it breaks the probe family for marginal space.

### 5.5 `_probe_out\p2_fold\` — 1.348 GiB, 1 file (2026/10/09 17:29)
`ort_optimized.onnx` (1,447,321,654 B) = ONNX Runtime-optimized folded-graph export (plus empty `meta\`). **REGENERABLE-FAST** (re-run the ORT optimization step, minutes). **DELETABLE-CANDIDATE.**

### 5.6 `_probe_out\merge_probe\` — 1.348 GiB, 2 files (2026/10/09 12:41)
`merge_1shard_2blk\weights.bin` (1.38 GiB) + `model.onnx` (0.13 MiB) — output of the shard-merge probe (`_probe_merge_shards.py` survives). **REGENERABLE-SLOW** (re-merge after re-exporting shards). Author: conclusions recorded, safe to drop. **DELETABLE-CANDIDATE.**

### 5.7 `_probe_out\p2_wholegraph\` — 0.759 GiB, 130 files (2026/10/09 17:42)
`w_*` / `m_*` `.const` blobs (42.19 MiB each) — whole-graph const-weight dumps produced by partition runs of the p2 probes. **REGENERABLE-FAST** (minutes: re-run the partition probes). **DELETABLE-CANDIDATE.**

### 5.8 `_probe_out\bfp16_deadlock\` — 0.512 GiB, 8 files (2026/10/09 19:51)
`fail_{8,16,32}_{48,64,96,128}*.txt` — deadlock-reproduction run logs (up to 140 MiB of text each). Conclusions are recorded in the `_gl*_REPORT.md` / handover docs. **DISPOSABLE** (log output). **DELETABLE-CANDIDATE.**

### 5.9 `_probe_out\rootcure\` — size fluctuating (0.44 → 0.11 GiB during survey), 3–6 files (2026/10/11 20:24–20:29)
`gemm_{real,zeros,other}.onnx` (150 MiB each at first sighting; some removed since) + `weight_source.json` — an **in-flight** GEMM "weight-source ambiguity / cache-key" experiment (json shows `weight_source: "ambiguous"`, VitisAIExecutionProvider build/run timings). **REGENERABLE-FAST** (each ONNX builds in ~42 s) but **ACTIVE RIGHT NOW**.
**Recommendation:** **KEEP — DO NOT TOUCH** (another process owns it; also see root `original-*-signature.txt`).

### 5.10 `_probe_out\gl_work\` — 0.411 GiB, 257 files (2026/10/10 20:33)
GL-route working directory: `cache\` with 173 `.const` VAIP-cache blobs (~43 MiB each). Compiled artifacts from the gl16/gl17/gl20 series (see `gl8_adf`, `gl16_pack`, `gl20_restore2`, `gl_plug`, `gl17_plugins` siblings and `_GL2-FINAL-HANDOVER.md`). **REGENERABLE-SLOW** (re-partition). **DELETABLE-CANDIDATE**, ⚠️ but see §8 (unsure — today's GL-route state).

### 5.11 `_probe_out\dd85\` — 0.255 GiB, 12 files (2026/10/10 17:38)
Three identical sets (`nb4\`, `nb5\`, `nb6\`) of `sd_fastpm_DiT.zip` (59.79 MiB), `sd_fastpm_Unet.zip` (14.79 MiB), `sd_preemption.zip` (12.4 MiB), `sd_fastpm_mine.zip` (0.01 MiB). Zip contents = `sd_fastpm/*.bin` NPU instruction/ctrl bundles + `*_meta.json` (vendor test bundles). The same zips are duplicated in `dd81` and `dd_swap` (up to 6 copies of DiT.zip total). **DISPOSABLE (duplicate copies)**. **DELETABLE-CANDIDATE** (consider retaining one set, e.g. `dd85\nb4` — see §8).

### 5.12 `_probe_out\dd81\` — 0.170 GiB, 7 files (2026/10/10 17:36)
Two more sets (`A_only_vendor\`, `B_vendor_cast\`) of the same `sd_fastpm_*` zips. **DISPOSABLE (duplicates)**. **DELETABLE-CANDIDATE.**

### 5.13 `_probe_out\dd_swap\` — 0.167 GiB, 8 files (2026/10/10 10:22)
Another set of the same zips (`broot\` + `sd_fastpm_DiT_B.zip`) + 2 `.npy` outputs (12 MiB each) + 2 tiny `.py`. **DISPOSABLE (duplicates)**. **DELETABLE-CANDIDATE.**

### 5.14 `_probe_out\subv_search\` — 0.164 GiB, 296 files (2026/10/10 09:00)
AIE dataflow subvolume search: many repeated `dflow.log` copies (14.1 MiB each) + small outputs. **DISPOSABLE** (search logs; results summarized in notes). **DELETABLE-CANDIDATE.**

### 5.15 `_probe_out\waic_gate2\` — 0.148 GiB, 63 files (2026/10/10 09:00)
`gemm_model*.onnx` (37.5 MiB × 4) + `WAIC_Outputs\{Quant_0,Dequant_0}\` dataflow source (small `.py/.hpp/.h`). **REGENERABLE-FAST** for the ONNX; the code is small and worth keeping. **DELETABLE-CANDIDATE** (delete the 4 `.onnx`, keep the source).

### 5.16 `_probe_out\dd_swap2\` — 0.120 GiB, 37 files (2026/10/10 10:25)
`cache_A_vendor\`, `cache_B_mine\`, `cache_C_mine_corrupt\` = 3 copies of the same `ln0_*.const` cache sets (10.27 MiB each) — cache-swap A/B/C experiment. **DISPOSABLE (duplicates)**. **DELETABLE-CANDIDATE.**

### 5.17 `_probe_out\dd_live\` — 0.111 GiB, 15 files (2026/10/10 10:30)
`ctrl2.onnx` (72 MiB control graph) + `ln0_*.const` / `dd_metastate_*` fragments. **REGENERABLE-FAST** (re-run the dd_live probe). **DELETABLE-CANDIDATE.**

### 5.18 File group: `zimage-port\` top-level `*.py/*.log/*.txt/*.md` — 198.6 MB, 2,695 files
- `*.py` (868 files, 4.28 MB) + `*.md` (208 files, 1.89 MB): **code + docs — KEEP.**
- `*.log` (660 files, 98.9 MiB) + `*.txt` (959 files, 84.7 MiB): probe logs/outputs. Largest: `_p0_d.log` 37.4 MiB, `_p0_d.txt` 20.0 MiB, `_p1_ac_run.txt` 10.2 MiB, `_p1_ac_run.log` 9.2 MiB, `_vaip3.log` 6.1 MiB, then ~15 × ~5 MiB `_g*/_gh*/_glC*` logs. **REGENERABLE-FAST / DISPOSABLE.** Group exceeds 100 MB but no single file does. **DELETABLE-CANDIDATE** as a group (~184 MiB) if needed; cheap either way.

### 5.19 Below-threshold but noted
- `_probe_out\` root loose logs: 227 files, 62 MiB (`round2*.log`, `mha_*.txt`, …) — disposable logs.
- Root loose files: 11.2 MiB total (`build-vulkan.log` 10.2 MiB disposable).
- Root `tmp*` 14 dirs + `zimage-port\tmp*` 41 dirs: **all empty (0 bytes)** — DISPOSABLE, trivial.
- `shards_i1024_c32\`: 31.46 MiB (`io.npz` + empty `layers_00_02\`) — **KEEP** (probe reference; destination for the re-export).

---

## 6. Top 15 DELETABLE-CANDIDATES ranked by size freed

| # | Item (path under `zimage-port\_probe_out\`) | Bytes freed | GiB | Class | Regeneration |
|---:|---|---:|---:|---|---|
| 1 | `zimage_loop\` — 36 non-KEEP subdirs | 14,486,073,968 | 13.491 | REGENERABLE-SLOW | `partition` per `_GL2-FINAL-HANDOVER.md` recipe |
| 2 | `bfp16_weights_v2\` | 6,924,417,834 | 6.449 | REGENERABLE-SLOW | `encode_zimage_bfp16_v2.py` |
| 3 | `dd_iter\` (keep 2 tiny `.py`) | 4,342,209,036 | 4.044 | REGENERABLE-SLOW | `_dd4_iter.py` / `_dd5_vi.py` |
| 4 | `merge_probe\` | 1,447,418,023 | 1.348 | REGENERABLE-SLOW | `_probe_merge_shards.py` |
| 5 | `p2_fold\` | 1,447,321,654 | 1.348 | REGENERABLE-FAST | re-run ORT optimization |
| 6 | `p2_wholegraph\` | 814,842,660 | 0.759 | REGENERABLE-FAST | re-run partition probes |
| 7 | `bfp16_deadlock\` | 550,264,384 | 0.512 | DISPOSABLE | log output (conclusions recorded) |
| 8 | `gl_work\` ⚠️ (see §8) | 441,278,209 | 0.411 | REGENERABLE-SLOW | re-partition gl-route graph |
| 9 | `dd85\` | 273,671,708 | 0.255 | DISPOSABLE | duplicate zips |
| 10 | `dd81\` | 182,439,182 | 0.170 | DISPOSABLE | duplicate zips |
| 11 | `dd_swap\` | 179,087,406 | 0.167 | DISPOSABLE | duplicate zips |
| 12 | `subv_search\` | 175,613,559 | 0.164 | DISPOSABLE | search logs |
| 13 | `waic_gate2\` (delete `.onnx`, keep source) | 158,673,854 | 0.148 | REGENERABLE-FAST | rebuild ONNX |
| 14 | `dd_swap2\` | 129,266,685 | 0.120 | DISPOSABLE | triplicated cache sets |
| 15 | `dd_live\` | 119,357,552 | 0.111 | REGENERABLE-FAST | re-run probe |

**Total: deleting these 15 items frees 31,671,935,714 B = 29.50 GiB (31.67 GB decimal).**

Optional extras (not counted): `zimage-port\` log/txt group ~184 MiB, `_probe_out\` loose logs 62 MiB, empty tmp dirs 0 B.

Space math: 12.0 GB free + 29.50 GiB freed ≈ **41.5 GiB free** vs **37.9 GiB** needed for 22.93 GiB shards + 15 GiB VAIP cache → **fits with ~3.5 GiB margin**. If more margin is wanted, delete the optional log groups first (no risk), and only then consider `single` (§5.4; not recommended).

---

## 7. Explicitly KEPT (do not delete)

| Item | GiB | Why |
|---|---:|---|
| `zimage-port\_models\Z-Image-Turbo\` (all, esp. `transformer\` 22.93 GiB, `text_encoder\` 7.49 GiB) | 30.67 | **IRREPLACEABLE** downloaded/produced model data; source of all regeneration |
| `zimage_loop\` KEEP set: `out_op1, out_fix2, out_r1, t_a4, t_w1, g_ca1, out_nomul2, out_nomul2_keep` | 3.31 | DD-route baseline graphs (author's never-delete list) |
| `single\` | 2.05 | live probe fixture (author's never-delete list), 10 scripts depend on it |
| `rootcure\` | ~0.1–0.4 | **ACTIVE experiment** |
| `shards_i1024_c32\io.npz` + `aiert_build`, `cast_nb3`, `dd103t`, `plugins_clean`, `_out_512\` | ~0.1 | author's never-delete list / reference images |
| all `*.py`, `*.md`, `export_shards_fp32.py`, `_bfp16_pack.py`, handover docs, `.git` | ~0.05 | code + documentation needed to regenerate everything else |
| root `models\K2-Horizon-1B-Q4_K_M.gguf` | 0.62 | downloaded model weight (outside survey scope) |

---

## 8. Items whose classification I am NOT fully sure about

1. **`gl_work\` (0.411 GiB)** — it is today's (10/10–10/11) GL-route working state and the newest non-`rootcure` work in `_probe_out`. The `.const`/cache blobs are regenerable, but if the gl20 "restore" experiment's compiled state is not reproducible from the handover notes, this could matter. **Verify `_GL2-FINAL-HANDOVER.md` gl17/gl20 status before deleting.**
2. **`zimage_loop\plugins\` (85 MiB, 10 files)** — falls inside the non-KEEP delete set per the author's rule, but sibling dirs (`ca_plugins*`, `gx_plugins1`, `gl_plug`, …) suggest compiled plugin binaries were exchanged between experiments; keep it if any KEEP subdir loads plugins from it.
3. **`sd_fastpm_*.zip` bundles in dd85/dd81/dd_swap** — 6 duplicate sets exist so deleting copies is safe, but their origin is unknown (possibly a vendor sample bundle that is not re-downloadable). Safest: delete all but one set (`dd85\nb4`).
4. **`zimage_loop\out_nomul2_mf\` / `out_nomul2_norm\` (0.411 GiB each)** — created at the same second as the kept `out_nomul2`/`out_nomul2_keep`; they look like paired arms of one experiment. The author's rule deletes them; if the paired-arm comparison is still open, keep them.
5. **`rootcure\`** — actively changing; classification "REGENERABLE-FAST" is based on `weight_source.json` timings, but the experiment is not finished.
6. **`zimage-port\` large `*.txt` logs** (e.g. `_p0_d.txt` 20 MiB) — presumed logs/dumps; a few may be captured tool output that was never summarized. Low risk at 184 MiB total.

---

*End of inventory. No file was deleted, moved, or modified during this survey; this report is the only artifact created.*
