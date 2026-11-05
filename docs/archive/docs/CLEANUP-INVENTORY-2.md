# DISK CLEANUP INVENTORY 2 — `D:\Project\deepseek_harness\public_work`

**Survey mode: READ-ONLY.** Nothing was deleted, moved, or modified. This report is the only file created.
**Survey time:** 2026/10/12 ~11:15–11:35 (local). Supersedes `CLEANUP-INVENTORY.md` (2026/10/11 ~20:30). **Units:** exact bytes; GiB = bytes / 2^30.
**Sizing method:** per-directory recursive enumeration (PowerShell `Get-ChildItem -Recurse -File` + `Measure-Object Length -Sum`), each big number cross-checked against per-file listings.

---

## 1. Free space — D: IS CRITICAL AND VOLATILE

| Drive | Total | Free (11:20) | Free (11:28) |
|---|---:|---:|---:|
| **D:** | 652.88 GiB (≈701 GB dec.) | **3.898 GiB** | **5.386 GiB** |
| C: | 300.00 GiB | 10.636 GiB | 10.659 GiB |

- **D: free space dropped from ~12 GB (yesterday's inventory) to ~3.9–5.4 GB** — it is now genuinely near-full.
- ⚠️ **Free space moved +1.5 GiB in the 8 minutes between my two measurements.** Some other process is still creating/deleting things on D: right now (consistent with the earlier report of caches repeatedly created and deleted). **Re-measure free space immediately before any actual cleanup.**
- (Verification note: `fsutil`/`Get-PSDrive`/`Get-CimInstance` were denied by the sandbox; numbers above come from `[System.IO.DriveInfo]`.)

---

## 2. What changed since the old inventory (verified)

- `_probe_out\` whole tree: **42,726,873,534 B (39.79 GiB), 12,156 files** (was 40,112,247,031 / 37.36 GiB / 12,130 files) → **+2.43 GiB**, all of it the new 10/11–10/12 experiments.
- New experiment group (21 dirs from the 10/11 20:47 → 10/12 11:00 session): **3,086,703,553 B (2.875 GiB)** total.
- **Correction to the briefing:** the new `*_cache` VAIP/RAI compile caches are **NOT 100+ MB each** — the biggest is 44.58 MB. The 100+ MB items are the experiment output dirs (`skeleton` 1.35 GiB, `packverify` 0.64 GiB, `noinit` 0.29 GiB). All 8 `*_cache` dirs combined = **151,945,787 B (144.9 MiB)**.
- `_acl_recovery\` is tiny as expected: **2,811 B, 1 file** (`acl-report-*.jsonl`, 2026/10/12 10:48). Note: it lives at the **workspace root**, not `zimage-port\_acl_recovery\` (that path does not exist). Harmless either way.
- The 10/11 `rootcure` experiment has **finished** (last write 2026/10/11 20:29; conclusions in `weight_source.json`). It is no longer "active".
- Most recent activity anywhere: `export_skeleton.py` run logging to `_probe_out\skel_ctx.log` (last write **2026/10/12 11:00:51**). It started exporting `skeleton_all\context_refiner` but **no output for that shard exists yet** and no python process was visible at 11:28 → the run either crashed or is unlisted. **The "skeleton" work stream may still be live — treat `skeleton\`, `skeleton_all\`, `skeleton_cache\`, `export_skeleton.py` as an in-flight project (see §8).**
- Every old big item (`zimage_loop`, `bfp16_weights_v2`, `dd_iter`, `single`, `merge_probe`, `p2_fold`, `p2_wholegraph`, `bfp16_deadlock`, `gl_work`, `dd8x`, `dd_swap*`, `subv_search`, `waic_gate2`, `dd_live`) is **byte-for-byte unchanged** in size since yesterday's survey.

---

## 3. Every `_probe_out\` directory > 50 MB — size, files, newest mtime, contents, recoverability

| Directory | Bytes | GiB | Files | Newest mtime | Appears to contain | RECOVERABILITY CLASS |
|---|---:|---:|---:|---|---|---|
| `zimage_loop` | 18,039,768,447 | 16.801 | 8,377 | 10/10 21:12 | DD-route iteration loop: 42 `out_*`/`g*`/`t*` graph-variant dirs, each with `partition.log` + `replaced.onnx(.data)` + VAIP `cache\` (`.const` blobs); 8 author-KEEP baselines = 3,553,601,626 B / 2,020 files; **36 non-KEEP = 14,486,073,968 B / 6,350 files** | REGENERABLE-SLOW (recipes in `_GL2-FINAL-HANDOVER.md`; whole set ≈1.5 days of iterations) |
| `bfp16_weights_v2` | 6,924,417,834 | 6.449 | 2 | 10/09 16:37 | `weights.bfp16` (6,924,272,328 B) + `manifest.json` — all 521 transformer tensors in official interleaved BFP16 layout | REGENERABLE-SLOW (`encode_zimage_bfp16_v2.py`, deterministic; tens of minutes–hours of CPU, unverified) |
| `dd_iter` | 4,342,209,036 | 4.044 | 10 | 10/10 20:25 | `fold\folded.onnx`, `folded_vi.onnx`, `weights.bin` (1.38 GiB each) + 2 tiny `.py` | REGENERABLE-SLOW (`_dd4_iter.py` / `_dd5_vi.py`; **keep the 2 tiny .py**) |
| `single` | 2,204,054,023 | 2.053 | 42 | 10/09 13:08 | fp32 per-tensor dumps of blocks 0–2 + `io.npz` live fixture used by 10 `p2_probe_*.py` | REGENERABLE-FAST but **KEEP** (author NEVER-delete list; live probe fixture) |
| `skeleton` ⚠️ | 1,450,350,887 | 1.351 | 5 | 10/12 09:25 | NEW: `layers_00_02\weights.npz` 1,447,291,506 B + `io.npz` 3 MB + 58 KB `skeleton.onnx` + manifest/verify JSON. Weightless-skeleton export ("weights fed as inputs") — verdict recorded in `verify.json`: *"ARCHITECTURE PROVEN ON A REAL BLOCK"*, verify build 296 s | REGENERABLE-FAST (`export_skeleton.py` re-extracts from surviving safetensors, minutes) — but **part of an in-flight project** |
| `merge_probe` | 1,447,418,023 | 1.348 | 2 | 10/10 09:37 | `merge_1shard_2blk\weights.bin` + `model.onnx` — shard-merge probe output | REGENERABLE-SLOW (re-merge after re-export; author: conclusions recorded) |
| `p2_fold` | 1,447,321,654 | 1.348 | 1 | 10/09 17:29 | `ort_optimized.onnx` — ORT-optimized folded graph | REGENERABLE-FAST (re-run ORT step, minutes) |
| `p2_wholegraph` | 814,842,660 | 0.759 | 130 | 10/09 19:06 | `w_*`/`m_*` `.const` whole-graph weight dumps | REGENERABLE-FAST (re-run partition probes, minutes) |
| `packverify` ⚠️ | 689,339,858 | 0.642 | 10 | 10/12 08:40 | NEW: 8 per-op GEMM `gemm_*.onnx` (ffn w1/w2/w3 157/157/236 MB, attn 59+59 MB…) + `packverify.json` (verdict: packing **MISMATCH**, conclusion captured) + `perm_rule.json` | REGENERABLE-FAST (`verify_pack_shapes.py`/`pack_bfp16.py`; builds 28–43 s each per the JSON) |
| `bfp16_deadlock` | 550,264,384 | 0.512 | 8 | 10/09 19:51 | `fail_*.txt` deadlock-repro logs (up to 140 MiB text) | DISPOSABLE (logs; conclusions in report docs) |
| `gl_work` | 441,278,209 | 0.411 | 257 | 10/10 20:33 | GL-route working dir: `cache\` 173 `.const` VAIP blobs (~43 MiB each) | REGENERABLE-SLOW (⚠️ GL-route state — see §8) |
| `noinit` ⚠️ | 314,657,555 | 0.293 | 7 | 10/11 21:24 | NEW: `gemm_{absent,external,inline}.onnx` + `w.bin` 157 MB + `absent_init.json`/`feed_vs_rai.json` (weight-init/feeding experiment, conclusions captured) | REGENERABLE-FAST (GEMM build ~50 s) |
| `dd85` | 273,671,708 | 0.255 | 12 | 10/10 17:38 | 3 identical sets of `sd_fastpm_*.zip` vendor bundles | DISPOSABLE (duplicates; keep 1 set — §8) |
| `feedcost` ⚠️ | 201,910,024 | 0.188 | 4 | 10/11 21:47 | NEW: `gemm_inline.onnx` 157 MB + `gemm_absent.onnx` + `cache_inline\*.rai` 44,581,236 B + `feedcost.json` (verdict **"feed is cheap"**, timings: inline build 51.6 s, feed build 0.4 s) | REGENERABLE-FAST; the `.rai` cache = DISPOSABLE |
| `dd81` | 182,439,182 | 0.170 | 7 | 10/10 17:36 | 2 more sets of the same `sd_fastpm_*.zip` | DISPOSABLE (duplicates) |
| `dd_swap` | 179,087,406 | 0.167 | 8 | 10/10 10:22 | same zips + 2 `.npy` 12 MB + 2 tiny `.py` | DISPOSABLE (duplicates) |
| `subv_search` | 175,613,559 | 0.164 | 296 | 10/10 09:00 | many repeated `dflow.log` copies (14.1 MB each) | DISPOSABLE (search logs; results summarized) |
| `waic_gate2` | 158,673,854 | 0.148 | 63 | 10/10 09:00 | 4 × `gemm_model*.onnx` (37.5 MB) + small dataflow source | REGENERABLE-FAST (delete `.onnx`, keep source) |
| `bfp16order` ⚠️ | 157,328,220 | 0.147 | 2 | 10/11 20:48 | NEW: `gemm_probe.onnx` 157 MB + `byteorder.json` (verdict: *"byte order NOT solved"*, 21 candidates tried, build 43.4 s) | REGENERABLE-FAST |
| `dd_swap2` | 129,266,685 | 0.120 | 37 | 10/10 10:25 | 3 copies of same `ln0_*.const` cache set (cache-swap A/B/C) | DISPOSABLE (triplicates) |
| `dd_live` | 119,357,552 | 0.111 | 15 | 10/10 10:30 | `ctrl2.onnx` 72 MB + cache fragments | REGENERABLE-FAST (re-run probe) |
| `rootcure` ⚠️ | 118,005,013 | 0.110 | 6 | 10/11 20:29 | NEW (final size): 3 × `gemm_*.onnx` 39.3 MB + `weight_source*.json` (conclusion: `weight_source: "ambiguous"`, cold build 23.7 s) — **experiment FINISHED** | REGENERABLE-FAST |

⚠️ = created by the 10/11–10/12 experiment session (all new items listed in the briefing are accounted for: `rootcure`, `rootcure_cache`, `noinit`, `noinit_cache`, `feedcost`, `crashcause`, `crashcause_cache`, `ceiling`, `ceiling_cache`, `bfp16order`, `bfp16order_cache`, `bfp16phase`, `bfp16fin`, `bfp16exp`, `inject`, `inject_cache`, `packverify`, `packverify_cache`, `skeleton`, `skeleton_all`, plus `skeleton_cache`).

Below 50 MB but part of the same session (all REGENERABLE-FAST/DISPOSABLE, all with conclusions already in JSON):

| Directory | Bytes | Files | Contents / class |
|---|---:|---:|---|
| `bfp16order_cache` | 44,581,216 | 1 | VAIP `.rai` cache — **DISPOSABLE** |
| `noinit_cache` | 44,581,208 | 1 | VAIP `.rai` cache — **DISPOSABLE** |
| `inject_cache` | 44,550,144 | 1 | VAIP `.rai` cache — **DISPOSABLE** |
| `rootcure_cache` | 11,403,592 | 1 | VAIP `.rai` cache — **DISPOSABLE** |
| `ceiling_cache` | 3,240,011 | 9 | `.rai` caches + gops/graph JSON — **DISPOSABLE** |
| `skeleton_cache` | 1,518,748 | 1 | `.rai` cache — DISPOSABLE (but tied to live skeleton project) |
| `crashcause_cache` | 1,474,136 | 3 | `.rai` caches — **DISPOSABLE** |
| `packverify_cache` | 596,732 | 1 | `.rai` cache — **DISPOSABLE** |
| `skeleton_all` ⚠️ | 2,059,054 | 2 | `noise_refiner\{io.npz, skeleton.onnx}` — KEEP while project live (see §8) |
| `crashcause` | 1,008,663 | 11 | crash-cause probe ONNX + logs + `crashcause.json` — DISPOSABLE (concluded 10/12 07:55) |
| `ceiling` | 84,877 | 10 | chain-length ceiling probe + `ceiling.json` — DISPOSABLE |
| `bfp16fin` | 9,633 | 5 | result JSONs — DISPOSABLE |
| `bfp16phase` | 3,775 | 1 | `phase.json` — DISPOSABLE |
| `inject` | 207 | 1 | `gemm_absent.onnx` stub — DISPOSABLE |
| `bfp16exp` | 0 | 0 | **empty dir** — DISPOSABLE |

**Notable:** `bfp16order_cache`, `noinit_cache`, `inject_cache`, and `feedcost\cache_inline` (44,581,236 B) are **four near-identical copies of the same compiled cache** (same hash dir `40df9993dd37e123f93539d2f574630f`, ±31 KB). Classic scratch duplication.

---

## 4. Re-verification of yesterday's three flagged items

| Item | Yesterday | Now | Status |
|---|---|---|---|
| `_models\Z-Image-Turbo\` | ~30.67 GiB, IRREPLACEABLE | **32,895,719,879 B = 30.637 GiB, 42 files** (46 incl. `_models\_raw` remnants), newest 10/09 13:01 | ✅ **UNCHANGED and COMPLETE.** transformer shards 9,973,693,184 + 9,973,714,824 + 4,672,282,880 = **24,619,690,888 B** ✓; text_encoder 8,044,982,000 B ✓; VAE 167,666,902 B ✓. **IRREPLACEABLE — the safetensors are the only source of all weights. NEVER delete.** |
| `_probe_out\bfp16_weights_v2\` | 6.45 GiB | **6,924,417,834 B = 6.449 GiB**, 2 files (`weights.bfp16` 6,924,272,328 + `manifest.json` 145,506), 10/09 16:37 | ✅ unchanged. Producer `encode_zimage_bfp16_v2.py` + `_bfp16_pack.py` still on disk; deterministic re-encode from surviving safetensors → **REGENERABLE-SLOW** |
| `_probe_out\zimage_loop\` | ~16.8 GiB | **18,039,768,447 B = 16.801 GiB**, 8,377 files + 7 loose `.py/.json`, newest 10/10 21:12 | ✅ unchanged. KEEP set (out_op1, out_fix2, out_r1, t_a4, t_w1, g_ca1, out_nomul2, out_nomul2_keep) = 3,553,601,626 B / 2,020 files; **non-KEEP 36 subdirs = 14,486,073,968 B / 6,350 files** (matches yesterday exactly) |

Also re-verified unchanged: `dd_iter` 4,342,209,036 · `single` 2,204,054,023 · `merge_probe` 1,447,418,023 · `p2_fold` 1,447,321,654 · `p2_wholegraph` 814,842,660 · `bfp16_deadlock` 550,264,384 · `gl_work` 441,278,209 · `shards_i1024_c32` 32,983,238 (only `io.npz`, KEEP).

---

## 5. Outside `zimage-port`

| Item | Bytes | Files | Class |
|---|---:|---:|---|
| root `tmp*` dirs | **0** | 0 | 18 empty dirs — DISPOSABLE (trivial) |
| `zimage-port\tmp*` dirs | **0** | 0 | 41 empty dirs — DISPOSABLE (trivial) |
| root loose logs: `build-vulkan.log` 10,720,134 · `infer-test.log` 373,228 · `k2h-build.log` 71,752 · `xdna2-log-1/2.txt` 173,172 · `AIECompiler.log` 8,494 | 11,346,779 | 6 | DISPOSABLE build/test logs |
| `zimage-port\` loose `*.log` (660) + `*.txt` (959) | 192,473,190 | 1,619 | DISPOSABLE/REGENERABLE-FAST logs (largest: `_p0_d.log` 39.2 MB, `_p0_d.txt` 21.0 MB) — cheap either way |
| `_probe_out\` loose root logs (229 `round*.log` etc.) | 64,970,160 | 229 | DISPOSABLE logs |
| `zimage-port\` loose `*.py` (887) + `*.md` (208) | 6,340,237 | 1,095 | **KEEP** — the code/docs that regenerate everything |
| root `_acl_recovery\` | 2,811 | 1 | DISPOSABLE diagnostic (tiny) |
| (context) `llama.cpp-0.5.0` 878,266,196 · `k2h-llama.cpp` 736,509,318 · `models\K2-Horizon-1B-Q4_K_M.gguf` 666,184,064 · `research` 110,183,990 | ≈2.4 GiB | | outside survey scope; the GGUF is downloaded weight (KEEP) |

`zimage-port\` total: 75,861,708,002 B (70.65 GiB) = `_models` 32,895,737,888 + `_probe_out` 42,726,873,534 + ~0.19 GiB code/logs + `.git`.

---

## 6. **NEW PRIORITY — pure scratch: VAIP/RAI compile caches (`*_cache`)**

All 8 `*_cache` dirs + `feedcost\cache_inline` = **196,527,023 B (187.4 MiB)** and are **DISPOSABLE**: each is one `<hash>\<hash>.rai` compiled-cache blob (+tiny metadata), fully regenerated the next time the corresponding graph is compiled. Nothing references them; the experiments that produced them are concluded (JSON verdicts recorded).

Rebuild cost, measured from the experiments' own JSONs:

| Cache to regenerate | Measured rebuild time |
|---|---|
| single-GEMM `.rai` (rootcure/noinit/inject/bfp16order/feecost class) | **cold compile 23.7–51.6 s** (`rootcure` 23.72 s cold / 0.55 s warm; `feedcost` 51.58 s; `bfp16order` 43.38 s) |
| `packverify_cache` (final_linear GEMM) | 27.8 s (`build_s` in `packverify.json`) |
| `skeleton_cache` (2-block shard verify session) | 295.97 s (`build_s` in `skeleton\verify.json`) |
| `crashcause_cache` / `ceiling_cache` (tiny synthetic graphs) | seconds each |

Also pure scratch from the same session (all REGENERABLE-FAST or concluded): the GEMM `gemm_*.onnx` probe graphs in `rootcure`/`noinit`/`feedcost`/`bfp16order`/`packverify` rebuild in **~24–52 s each** (their `build_s` fields), and `skeleton\layers_00_02\weights.npz` (1,447,291,506 B) re-extracts from the safetensors via `export_skeleton.py` in minutes (the shard verify itself took 296 s). **But do not touch the skeleton family while its project is live (§8).**

---

## 7. Ranked deletion candidates (top N totals)

Ranked by bytes freed; **class** per the recoverability taxonomy. All sizes exact.

| # | Item | Bytes | GiB | Class | Why deletable |
|---:|---|---:|---:|---|---|
| 1 | `zimage_loop\` 36 non-KEEP subdirs | 14,486,073,968 | 13.491 | REGENERABLE-SLOW | Completed DD-route iterations; author's own cleanup list; recipes in `_GL2-FINAL-HANDOVER.md` (⚠️ see §8 re `plugins`, `out_nomul2_mf/norm`) |
| 2 | `bfp16_weights_v2\` | 6,924,417,834 | 6.449 | REGENERABLE-SLOW | Deterministic re-encode via `encode_zimage_bfp16_v2.py`; not an input to shard/VAIP regeneration |
| 3 | `dd_iter\` (keep 2 tiny `.py`) | 4,342,209,036 | 4.044 | REGENERABLE-SLOW | Rebuild via `_dd4_iter.py`/`_dd5_vi.py` |
| 4 | `merge_probe\` | 1,447,418,023 | 1.348 | REGENERABLE-SLOW | Probe output; conclusions recorded |
| 5 | `p2_fold\` | 1,447,321,654 | 1.348 | REGENERABLE-FAST | ORT-optimized export; re-run in minutes |
| 6 | `p2_wholegraph\` | 814,842,660 | 0.759 | REGENERABLE-FAST | Const dumps; re-run probes in minutes |
| 7 | `packverify\` (+`packverify_cache`) | 689,339,858 | 0.642 | REGENERABLE-FAST | GEMM graphs rebuild in 28–43 s each; MISMATCH verdict already recorded |
| 8 | `bfp16_deadlock\` | 550,264,384 | 0.512 | DISPOSABLE | Repro logs; conclusions in report docs |
| 9 | `noinit\` (+`noinit_cache`) | 314,657,555 | 0.293 | REGENERABLE-FAST | Concluded probe (`feed_vs_rai.json`); GEMM rebuild ~50 s |
| 10 | `dd85\` | 273,671,708 | 0.255 | DISPOSABLE | Duplicate vendor zips (6 copies exist across dd85/dd81/dd_swap) |
| 11 | `feedcost\` (incl. `cache_inline` 44,581,236) | 201,910,024 | 0.188 | REGENERABLE-FAST | Verdict "feed is cheap" recorded; build 51 s |
| 12 | `dd81\` | 182,439,182 | 0.170 | DISPOSABLE | Duplicate zips |
| 13 | `dd_swap\` | 179,087,406 | 0.167 | DISPOSABLE | Duplicate zips |
| 14 | `subv_search\` | 175,613,559 | 0.164 | DISPOSABLE | Repeated search logs; results summarized |
| 15 | `waic_gate2\` (delete `.onnx`, keep source) | 158,673,854 | 0.148 | REGENERABLE-FAST | Rebuild ONNX in minutes |
| 16 | `bfp16order\` (+`bfp16order_cache`) | 157,328,220 | 0.147 | REGENERABLE-FAST | "byte order NOT solved" verdict recorded; build 43 s |
| 17 | `dd_swap2\` | 129,266,685 | 0.120 | DISPOSABLE | Triplicated cache sets |
| 18 | `dd_live\` | 119,357,552 | 0.111 | REGENERABLE-FAST | Re-run probe |
| 19 | `rootcure\` (+`rootcure_cache` 11,403,592) | 118,005,013 | 0.110 | REGENERABLE-FAST | Experiment FINISHED 10/11 20:29; json conclusions captured; GEMM build 24 s |
| 20 | all remaining new-session scratch (`inject_cache` 44,550,144, `crashcause` 1,008,663, `crashcause_cache` 1,474,136, `ceiling` 84,877, `ceiling_cache` 3,240,011, `bfp16fin` 9,633, `bfp16phase` 3,775, `inject` 207, `bfp16exp` 0) | 50,853,069 | 0.047 | DISPOSABLE | Tiny concluded probes + cache copies |
| | **Optional log groups (no risk):** zimage-port `*.log`+`*.txt` 192,473,190 + `_probe_out` loose logs 64,970,160 + root build logs 11,346,779 | 268,790,129 | 0.250 | DISPOSABLE | Logs only |

**Totals:**

| Delete set | Bytes freed | GiB | D: free after (≈5.4 GiB now) |
|---|---:|---:|---|
| Top 5 (rows 1–5) | 28,647,440,515 | **26.68** | ≈32.1 GiB |
| Top 10 (rows 1–10) | 31,290,216,680 | **29.14** | ≈34.5 GiB |
| Top 15 (rows 1–15) | 32,187,940,705 | **29.98** | ≈35.4 GiB |
| **All 20 rows (recommended scratch set)** | **32,863,432,369** | **30.61** | **≈36.0 GiB** |
| + optional log groups | 33,132,222,498 | 30.86 | ≈36.3 GiB |
| + unsure items `gl_work` + `skeleton*` (NOT recommended yet) | +1,895,206,898 | +1.76 | — |

Space math: the regeneration need quoted yesterday was **~37.9 GiB** (22.93 GiB shards + ≤15 GiB VAIP cache). Deleting the full recommended scratch set (30.61 GiB) gives ≈**36.0 GiB** free — just short of 37.9 GiB; adding the log groups and/or `gl_work` (if cleared, §8) crosses the line comfortably. Deleting only the top 10 gives ≈34.5 GiB.

---

## 8. UNSURE / do-not-touch-yet (explicit uncertainty list)

1. **⚠️ ACTIVE WORK STREAM: the "skeleton" family** (`skeleton\`, `skeleton_all\`, `skeleton_cache\`, `export_skeleton.py`, `verify_skeleton.py`, `skel_ctx.log`). Last activity **11:00:51 today** — `export_skeleton.py` had just started exporting shard `context_refiner` into `skeleton_all\` and produced no output yet (crashed or still running invisibly). `skeleton\` (1.35 GiB) is technically REGENERABLE-FAST, but it is ~2 hours old and its project may still need it. **Recommendation: keep the whole family until the project owner confirms the weightless-skeleton work is done; the 1.35 GiB `weights.npz` is the only meaningful piece and re-exporting costs only minutes.**
2. **`gl_work\` (441,278,209 B)** — unchanged since yesterday's survey; same caveat as before: GL-route compiled state may not be fully reproducible from notes. Verify `_GL2-FINAL-HANDOVER.md` gl17/gl20 status before deleting.
3. **`zimage_loop\plugins\` (91,311,027 B)** and **`zimage_loop\out_nomul2_mf\` / `out_nomul2_norm\` (440,852,934 / 440,834,098 B)** — inside the row-1 non-KEEP set, but `plugins` may be loaded by KEEP subdirs and the `_mf`/`_norm` dirs are paired arms of the kept `out_nomul2` experiment. ~0.96 GiB of the row-1 total; keep these three if unsure.
4. **`sd_fastpm_*.zip` vendor bundles (dd85/dd81/dd_swap)** — 6 duplicate sets; duplicates are safe to delete, but origin unknown (possibly not re-downloadable). Safest: retain one set (`dd85\nb4`) before deleting the rest.
5. **Free-space volatility** — D: free changed by +1.5 GiB during this 20-minute survey. Another process is churning disk. **Re-measure before cleanup, and re-run a size check on any dir right before deleting it** (the first survey's `rootcure` shrank mid-survey the same way).
6. `bfp16_weights_v2` rebuild time ("tens of minutes–hours") is **estimated**, not measured; only its determinism is verified.
7. The `zimage-port\` large `*.txt` logs (~88.8 MB) are presumed dumps; a few may be unsummarized captured output. Low risk at this size.

---

*End of inventory. No file was deleted, moved, or modified during this survey; this report is the only artifact created.*
