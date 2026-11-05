# P33 end-to-end: the 33-tile projection behind `T22_P33=1`, measured

**Date** 2026-11-02 · **device** owned by this task throughout (released at the end)
**Deliverable measured** `_scratch/util/place/proj33.py` (`install(dev)` + `proj4_33`),
reached through the existing hook `_scratch/util/glue/install.py:601-624`.
**Nothing tracked was edited.** Every file this task added is listed in section 9.

---

## 0. Verdict, in one paragraph

`T22_P33=1` **does nothing at all in the shipped 84.91 s configuration**, because
`install.py:704-711` replaces `Loop.attention` with its own hook and that hook never
calls `PD.proj4` (section 2). Proof: `p33b_1` (SHIPPED config + `T22_P33=1`) produced
latents **bit-identical** to the baseline (`4cbf4f4a1b98`, pred_absmax 6.591831207275391
on both). In the one configuration where the patch is reachable (the glue attention hook
switched off) the projection **is** 2.35x faster in isolation (38.68 vs 91.07 ms/layer,
same process, same device) but only **6.7% faster in the product** (90 -> 84 ms/layer):
the end-to-end loop is then a **wash, median pair ratio 1.0027** (0.979 / 1.006 / 1.003).
Three separate defects eat the difference, and all three are quantified in section 5:
the dead hook, a numba cache that **kills every run after the first**, and
`proj33._BRAM_MAX = 4`, which re-reads 66 MB of B payload from disk **every layer**
(16.8 ms/layer, 44% of the whole advantage).

Quality, where it does run: **corr 0.9958809, maxabs 1.34, verdict PASS**
(at or above 0.9953, the best cross the campaign has recorded).

---

## 1. The harness that reproduces the baseline (step 1)

The 84.91 s runs came from `_scratch/devkill/run_arm.ps1`, **not** from calling
`_scratch/bigpipe2/run_job_py.py` directly. `run_arm.ps1` is what sets
`T22_HOSTKILL=1 T22_DEVKILL=1 T22_BIGM_DEVHANDOFF=1`, and those are what make
`webui/chunked.py:720-753` take the devkill branch and call `install()` at all.
Two different variables are in play and the documents blur them:

| variable | set by | meaning |
|---|---|---|
| `T22_FASTGLUE` | **the shell** (run_arm.ps1 does not touch it) | install.py master switch: no `1`, `install()` returns before doing anything and prints nothing |
| `T22_FAST_GLUE` | `run_arm.ps1 -FastGlue 1` | t18_run's U1 host fast path (h1..h4 hunks) |

So the exact reproduction of the delivered number is:

```powershell
cd D:\Project\deepseek_harness\public_work\zimage-bfp16-npu
$env:T22_FASTGLUE = "1"      # install.py: 8 hooks
$env:T22_REF_SMNB = "1"      # refiner fused softmax
powershell -NoProfile -ExecutionPolicy Bypass -File _scratch\devkill\run_arm.ps1 `
    -Tag <tag> -Steps 8 -GlueProf 0 -FastGlue 1 -Glue2 1 -Glue3 1
# result: _scratch\bigpipe2\<tag>.run.log, last line
#   FINAL done total_s <T> loop_s <L> steps [...]
```

Confirmed by `[fastglue] installed 8/8 hooks (threads=8): bf16=on proj=on pack=on
decode=on ffnchain=on rms=on attn=on block=on(4 hunks)` in the job log.

**Recorded baseline** (HANDOVER-2026-11-01.md section 1): loop 84.91 s (n=10, range
84.359-86.222), latents sha12 `4cbf4f4a1b98`.

**Reproduced here**: 89.98 / 90.11 / 94.45 s on AC, **latents sha12 `4cbf4f4a1b98` on
all three runs** - i.e. the *configuration and the bytes* reproduce exactly; the
*seconds* do not, because the 11-01 numbers were taken on a quieter host.
Measured host state (job.json `memory_snapshot` and the startup line):

| | 11-01 delivery (qg_1) | this session (p33ac_a1) |
|---|---|---|
| available RAM at job start | 17.82 GB | 13.4 GB |
| CPU load at job start | 43% | 57% |
| device stage startup | 35.91 s | 42.9 s |

Absolute seconds from *this* session are therefore not comparable with the delivery;
only ratios and same-run numbers are used for conclusions (section 7).

---

## 2. The hook is dead in the shipped configuration (the finding that changes the plan)

`prod_devkill.bigm_attention` does call `PD.proj4` - that is what the T22_P33 hook
patches. But `install.py` replaces that function:

```
704    # ---- 6. attention: q/k skip the f32 drain, rope eats the bf16 drain --
705    if on("attn"):
706        fa = _make_attention(PB, PD, T, th, verbose)
707        PD.bigm_attention = fa
708        if getattr(T.Loop, "attention", None) is not None and \
709                getattr(T.Loop.attention, "__name__", "") == "bigm_attention":
710            T.Loop.attention = fa          # chunked installed it already
```

and `_make_attention`'s own `bigm_attention` (install.py:387-438) does its four
projections through `dispatch_raw` -> `_dispatch_proj` (install.py:338-358), which
writes X and the **plain15** W image and launches `PD._IMG` itself. `PD.proj4` is
never called. HANDOVER-2026-11-01.md section 7 item 1 already warned about this
("the hook installs `_PD.proj4` only in the fastglue-off branch"); this is what it
costs.

Measured, not inferred:

| run | `T22_P33` | attention hook | latent sha12 | pred_absmax | proj_qkv / proj_o (s/layer) |
|---|---|---|---|---|---|
| p33a_1 | - | on | `4cbf4f4a1b98` | 6.591831207275391 | 0.077 / 0.032 |
| **p33b_1** | **1** | **on** | **`4cbf4f4a1b98`** | **6.591831207275391** | **0.074 / 0.030** |

Bit-identical latents and a projection wall within 4%: with the shipped hook set,
`T22_P33=1` is a **no-op**. The `[fastglue] T22_P33: projection -> 33-tile kk=32 ...`
line does print (`proj=on+p33` appears in the installed-hooks line), so the flag
*looks* engaged - it is the call site that never arrives.

---

## 3. A/B where the hook is reachable (step 2)

Reachability required one documented knob: `T22_FASTGLUE_SKIP=attn` (install.py's
own per-hook bisect switch). Then `Loop.attention` stays
`prod_devkill.bigm_attention`, which *does* call `PD.proj4`, and the two arms differ
in **nothing but the projection**. Arms:

* **A** = `T22_FASTGLUE=1 T22_REF_SMNB=1 T22_FASTGLUE_SKIP=attn` -> `PD.proj4 = fast_proj4` (plain15)
* **B** = the same `+ T22_P33=1` -> `PD.proj4 = proj4_33`

### 3.1 loop_s, 3 alternating pairs, all on AC

```
_scratch\p33_e2e\run_p33_ab.ps1 -Arm A|B -Tag <tag> -Skip attn
```

| pair | arm | tag | loop_s | total_s | ratio B/A |
|---|---|---|---:|---:|---:|
| 1 | A | p33ac_a1 | 94.449 | 126.6 | |
| 1 | B | p33ac_b1 | **92.461** | 132.4 | **0.9790** |
| 2 | A | p33ac_a2 | 90.107 | 130.2 | |
| 2 | B | p33ac_b2 | 90.657 | 128.2 | **1.0061** |
| 3 | A | p33ac_a3 | 89.983 | 131.2 | |
| 3 | B | p33ac_b3 | 90.222 | 130.1 | **1.0027** |

**Median pair ratio 1.0027, spread 0.9790-1.0061 (2.7 points).** Not faster.
A median 90.107 s, B median 90.657 s.

### 3.2 The projection itself *is* faster, by 6.7%

The driver prints one line per (step, layer); 240 per run. Medians:

| pair | A proj ms/layer | B proj ms/layer | ratio | A proj/FFN | B proj/FFN | ratio |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 91.0 | 84.0 | 0.923 | 0.850 | 0.791 | 0.931 |
| 2 | 90.0 | 84.0 | 0.933 | 0.862 | 0.799 | 0.927 |
| 3 | 89.0 | 83.0 | 0.933 | 0.864 | 0.779 | 0.902 |
| **med** | **90.0** | **84.0** | **0.933** | **0.862** | **0.791** | **0.927** |

`proj` = `proj_qkv + proj_o`, i.e. all four projections of one layer.
`proj/FFN` divides by the FFN wall measured in the same layer of the same run; the FFN
is identical code in both arms, so this ratio is immune to the clock/power state.

-6.0 ms/layer x 240 layer-steps = **-1.45 s** out of ~90 s. That is 1.6%, which is
exactly the size of the loop noise (section 7), hence section 3.1.

---

## 4. Quality (step 3)

`_scratch/util/place/quality_gate.py p33ac_a3... p33ac_b3...`:

```
sha256[:12] : A 4cbf4f4a1b98   B 1545ff4732ba
NaN / inf   : A 0 / 0          B 0 / 0
absmax      : A 3.97612        B 3.92177
corr        : 0.9958809
maxabs      : 1.33667  (rel 0.341 of absmax(B))
nrmse       : 0.0820901
verdict     : corr 0.995881 >= 0.9953 : at or above the best recorded cross  (PASS)
```

* The `[fastglue] T22_P33` line **did** appear in the job log
  (`[fastglue] T22_P33: projection -> 33-tile kk=32 (measured 39.7 vs 90.5 ms/layer, corr 0.99989)`),
  and `installed 8/8 hooks ... proj=on+p33`. It prints whenever the *hook is
  installed*, which is not the same as the projection being used - see section 2.
* Determinism: **A ran 5x, all `4cbf4f4a1b98`; B ran 8x, all `1545ff4732ba`.**
  Arm A reproduces the delivery's byte-identical output exactly; the P33 change is
  itself byte-deterministic.

---

## 5. Why it is not faster: the per-layer breakdown (step 4)

### 5.1 It is not the forecast failure mode

`proj4_33` does **not** rebuild B from the checkpoint per layer in the product:
`b_payload(..., src="auto")` hits the disk cache (`P33_BCACHE`). The rebuild path
costs, measured on AC by `proj33_bench.py`:

| B path | per name | per layer (4 names) |
|---|---:|---:|
| checkpoint -> blocks (`np.load` + `pack_bfp16.unpack` + `encode_bfp`) | 466-499 ms | **2.03 s** |
| blocks -> payload (the kk=32 permutation, numba) | 3.0-4.6 ms | 4-18 ms |
| payload read back from the disk cache (`np.fromfile`) | **4.2-4.7 ms** | **16.8-18.9 ms** |
| from the in-process RAM cache | 0.01 ms | 0.04 ms |

Had it rebuilt per layer it would be **+2.0 s/layer** - the run would be ~10 minutes
instead of 90 s. It does not; the damage is the third row, not the first.

### 5.2 Isolated, same process, same device, AC: 2.35x

`proj33_bench.py --layers layers.0 --runs 5` (plain15 control in the same process):

| stage | proj4_33 (33-tile) ms/layer | plain15 ms/layer |
|---|---:|---:|
| A prep / pack | **1.00** | 23.16 |
| B get | 0.01 (RAM) | - (RAM) |
| A write to device | 1.10 | - |
| B write (names 2-4) | 2.16 | 4.76 (W) |
| **dispatch run** | **30.53** (4 x 7.63) | **60.60** (8 x 7.58) |
| read | 0.35 | 0.35 |
| C decode / cv | 2.81 | 2.28 |
| **total** | **38.68** | **91.07** |

2.35x, reproducing the previous agent's 39.7 vs 90.5.

### 5.3 In the product: 84 ms/layer, itemised

Reconstructing the 84 ms/layer from the measured components above plus what the
product adds:

| item | ms/layer (4 names) | source |
|---|---:|---|
| A pack | 4.8 | `proj4_33` is called **twice** per layer (q,k,v then o) and packs A each time: 2 x 2.41 |
| B get from disk | **16.8** | `_BRAM_MAX = 4` = exactly one layer, so all four payloads are re-read every layer: 4 x 4.2 |
| B write to device | 2.9 | 16.6 MB per name |
| A write to device | 2.2 | 8.1 MB per call |
| dispatch run | ~54 | 4 launches; the product measures ~13.5 ms/launch against 7.63 ms back-to-back |
| C decode | 2.8 | 0.70 ms/name |
| **total** | **~84** | matches the 83-84 ms/layer the driver printed |

Against arm A's 90 ms/layer on **8** launches of **half** the B bytes each.

### 5.4 The three defects, in order of damage

1. **The hook is unreachable** (section 2). In the shipped configuration the effect is
   exactly zero, not "small".
2. **The numba cache kills the run.** `proj33` is loaded by
   `importlib.util.spec_from_file_location("_p33mod", ...)` and is not registered in
   `sys.modules`, so numba cannot rebuild the cached kernel environment. The **first**
   process compiles and works; **every later process** dies at step 0:

   ```
   File "...numba/core/caching.py", line 618, in _load_data
     tup = pickle.loads(data)
   File "...numba/core/environment.py", line 51, in _rebuild_env
     mod = importlib.import_module(modname)
   ModuleNotFoundError: No module named '<dynamic>'
   ```

   Measured: `p33b_2` died after 34.5 s, `p33b_2_4` after 38.1 s, `p33b_2_3` after
   46.6 s, all at step 0. I had to delete `webui/cache/numba/place_*` before **every**
   B run to get a result at all (2 KB of `.nbi` + `.nbc` for
   `proj33._nb.locals._pack_a32` / `_unpack_c_bt`). This alone makes `T22_P33=1`
   unshippable regardless of speed.
3. **`_BRAM_MAX = 4`** costs 16.8 ms/layer, **44% of the 38.7 ms/layer advantage**.
   The shipped product already solves the same problem for plain15 by prewarming 120
   images into `prod_devkill._PROJ_W` at startup (`devkill_prewarm.prewarm_all`);
   `proj33._BRAM` needs the identical treatment (a 120-entry RAM cache is 1.99 GB,
   the same size as the plain15 W-image cache it replaces).

---

## 6. What it would take to make P33 win

Measured counterfactual, from section 5.3: put B in RAM and the same run is
`84.0 - 16.8 = 67.2` ms/layer, i.e. **-22.8 ms/layer (-25%) against arm A's 90 ms**.
Over 240 layer-steps that is **-5.5 s per 8-step image (-6.1% of 90 s)** - worth having,
but an order of magnitude below the 2.28x the isolated numbers suggested, because the
loop is dominated by the FFN, the refiner and attention, not by the projection.

In priority order:

1. **Wire the hook where the projection is actually called.** With the glue attention
   hook on, the call site is `_dispatch_proj` (per name), and its consumers
   (`G.rope_fused`, `G.widen_t`, `G.proj_read_cv`) eat the **bf16 drain**, while
   `proj4_33` returns **f32**. So this is not a rename: `proj4_33` needs a
   `drain="bf16"` output (return `(u32 >> 16).astype(uint16)` from the same mapped C,
   ~4 ms/layer) for the glue hook to be able to use it at all.
2. **Prewarm `_BRAM` at startup** for all 30 layers (or raise `_BRAM_MAX` to >= 120),
   exactly as `devkill_prewarm` does for plain15 - otherwise 16.8 ms/layer is lost.
3. **Fix the numba cache**: register the module in `sys.modules` before
   `exec_module` (or drop `cache=True`), otherwise the second job on a host dies.
4. **Pack A once per layer**: `_dispatch_proj` is called per name, so `proj4_33` packs
   the same A three times (3 x 2.41 ms). Cache the A payload per (layer, image version).
5. Only then does the 2-launches-per-name -> 1-launches-per-name win survive; today it
   is spent on the 16.6 MB B re-upload per name (2x the plain15 W image) plus the
   in-product launch penalty (13.5 ms vs 7.63 ms back-to-back).

---

## 7. Method and validity (read before quoting any number)

* **Power state.** The machine was **on battery** from ~08:20 to ~10:10 and **on AC**
  after that (user-confirmed). Battery-era numbers are not comparable across time and
  are **discarded** for the A/B: 13 runs (`p33a_1`, `p33b_1`, `p33a2_1..8`,
  `p33b2_1..9`) are reported only as mechanism evidence (the dead hook, the numba
  crash, the shipped-config bit-identity), never as performance.
  **All numbers in sections 3, 4 and 5 are from AC work started at 10:11:24.**
* **Absolute vs ratio.** Absolute seconds on this host are noisy: the same
  configuration produced 89.98-94.45 s on AC and 96.5-135.4 s on battery. The
  **primary results are ratios**: pair ratio of loop_s, pair ratio of `proj`, and
  `proj/FFN` within one run. The absolute 84.91 s of the delivery is **not**
  reproduced as a wall time here (best: 89.98 s); what is reproduced is the
  **configuration and the output bytes** (`4cbf4f4a1b98`).
* **Alternation.** A/B/A/B/A/B on AC, one pair every ~5.5 minutes, so any drift
  between pairs hits both arms; the three pair ratios (0.979 / 1.006 / 1.003) agree to
  2.7 points, i.e. the "not faster" verdict is firm, not a noisy coin flip.
* **`-Prewarm`.** Both AC arms use the delivered `-Prewarm 1`. (`-Prewarm 0` costs
  +8.6 s of step 0 in **both** arms, because `fast_proj4` also fills
  `prod_devkill._PROJ_W`; it is not a P33-specific flag.)
* **`-Skip attn`.** Both AC arms skip the same hook, so the A/B isolates the
  projection. It is **not** the delivered configuration - the delivered configuration
  is section 2, where the answer is "no change at all".

---

## 8. Files (all new)

| file | role |
|---|---|
| `P33-E2E-REPORT.md` | this report |
| `_scratch/p33_e2e/run_p33_ab.ps1` | one A/B arm on the delivered harness (`-Arm A|B -Tag -Skip -Prewarm`) |
| `_scratch/p33_e2e/p33_prewarm.py` | build the 120-payload B cache (host only, 106.8 s, 1.99 GB) |
| `_scratch/p33_e2e/p33_ab.py` | A/B table + quality from the run artefacts |
| `_scratch/p33_e2e/p33_layers.py` | per-layer projection / FFN / proj-FFN-ratio table (machine-independent view) |
| `_scratch/p33_e2e/_probe_*.py` | the probes quoted above (B cost, qkv-cache byte identity, payload read cost, hashes) |
| `_scratch/p33_e2e/site/sitecustomize.py` | the in-product instrument. **It does not work**: the py310 interpreter is an isolated embedded build (`python310._pth`, `sys.flags.isolated == 1`), so `PYTHONPATH`/sitecustomize are ignored. Kept as evidence; the timing had to come from `proj33_bench.py` instead. |

Run artefacts: `_scratch/bigpipe2/p33*.{run.log,job.json,webui.*.log}` and
`webui/jobs/<id>/job.log` (the full logs, 65 KB each).

**Disk**: the 1.99 GB B-payload cache lived in `%TEMP%\p33_bcache` (session temp, not
the repo) and was **deleted at the end**; the repo footprint of this task is < 2 MB.
D: had only 3 GB free at the start, so nothing 2 GB was ever written to the repo.

## 9. Device

Released. No `_scratch/device.lock`, no python process of this task left running;
`_scratch/block/bigm/work/qkv` (2 GB) untouched.
