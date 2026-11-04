# The WebUI says what it can actually do (and what it cannot)

**2026-11-03 · sizes, paths, walls and flags, all derived from the code that
decides them · no adaptive path was enabled · no table was written by hand**

```
THE ONE SENTENCE: the page now answers "can I render this?" from
shapes.resolve + chunk_gate + plan_manifest + webui/jobs/*/job.json, it names
the binding constraint when the answer is no, and the two sizes the brief named
are BOTH refused for a measured reason (1920x1080: refiner f32 matrix 7.44 GiB,
measured OOM at 42 s; 1024x1024: runnable, but no 8-step run has ever completed
in this tree and both attempts failed for named reasons).
```

---

## 0. One screen

| | before | after |
|---|---|---|
| 1920x1080 | accepted ("0.41 h"-class estimate from a model that does not see the refiner); measured: **device loop ran out of memory after 42 s** | **refused at submit** with the arithmetic: `30 x 8160 x 8160 x 4 B = 7.44 GiB > 6 GiB ceiling`, quoting the job that died |
| 1024x1024 | accepted; the only two 8-step attempts in the tree both failed (23 s BO overflow in the fast config; OOM in low mode) and the page did not say so | **accepted with the truth attached**: model 187 s/step / 0.41 h, the measured 1-step band 111-209 s, and "no completed 8-step job at this exact size; 4 failed runs recorded, last: device loop failed (rc=2) (23 s)" |
| 384x384 (adaptive) | "the plan is built" read as "it works" | **runs, and is slower**: first end-to-end 8-step run at a new size (loop 160.6 s vs 119.4 s on the delivered path) and its image does not match (latents corr 0.35).  Default stays OFF |
| every size | a hand-written table + a model estimate | one row per size from `py312 -m webui.capability --table`, with the path, the model, the measured wall + job id, and the flags |

---

## 1. Three ways the old answer lied, each with the run that proves it

**(1) The noise refiner is not in the model.**  `chunk_gate()` reasoned about
L1, host BOs and the host attention working set -- never about the CPU noise
refiner, which materialises `heads x S x S x float32` of scores per step in the
device child.  At 1088x1920 (S=8160) that is **7.44 GiB** on top of the 3.71 GB
cache-all image set and the 1.85 GB wpack mapping:

    job 20261103-150606-10d726  1920x1080 -> 1088x1920  M=8192
      "fast cache-all failed with MemoryError; fallback --no-cache-all"
      "device loop ran out of memory"                      41.5 s, no image

The old gate called this shape runnable and quoted 0.89 h for 8 steps.

**(2) The fast configuration's projection BO is not in the model either.**  With
`T22_FASTGLUE=1` (the env the product's own best 512^2 job used) the projection
is dispatched through the rt17 hook, whose host BO holds 1632 rows.  At M=4128:

    job 20261103-063607-d60112  1024x1024  M=4128
      RuntimeError: attempting to write past buffer size: invalid argument
      "device loop failed (rc=2)"                          23.2 s, no image

That is the "24-second crash" in the brief.  It is a *BO* constraint, so it is now
gated by BO arithmetic, not by a special case.

**(3) "Built" was read as "usable", and a 1-step wall as an 8-step wall.**  The
plan manifests prove the images exist; they do not prove the loop can dispatch
them at a profit.  Measured today at 384x384 (M=608), 8 steps, seed 7:

| arm | env | loop s | total s | latents |
|---|---|---:|---:|---|
| delivered (product default) | none | **119.44** | 140.25 | 3e57599f |
| campaign fast path | BIGM/DEVKILL/HOSTKILL/DEVHANDOFF, CHUNK_PIPE=0 | 92.35 | 143.37 | 2fb57c78 |
| campaign fast path **+ the plan's FFN** | + T22_PLAN_FFN=1 | **160.58** | 211.52 | 2b6b27ef |

* the plan path **completes** an 8-step run at a new size for the first time
  (job 20261103-152409-e8e997) -- and is **34 % slower on the loop** than the
  delivered path, because the plan's images are pure GEMMs and the
  silu(gate)*up pass lands on the host (measured 79.7 ms/layer of the 195.4 ms
  FFN block);
* it does **not** reproduce the delivered image: latents corr **0.3536**
  (maxabs 7.59); against the same campaign env without the flag, corr **0.4709**;
* and the campaign env itself does not reproduce the delivered path at this size
  (corr **0.6329**) -- it is tuned for M=1056 and is not validated at M=608.

So the adaptive path is **left off**, exactly as instructed: the one thing that
could have flipped it (a new size that completes **and passes quality**) is
precisely what failed the second half.

---

## 2. What the page answers now

`GET /api/capability?height=H&width=W` -> one shape, `?table=1` -> every size.
Both are `webui/capability.py`, which only reads:

| source | what it contributes |
|---|---|
| `shapes.resolve(h,w)` | ready / exec_mode / padded run size / crop rule / chunk estimate / plan + manifest |
| `shapes.chunk_gate(h,w)` | runnable, and the binding constraint when not (now incl. the refiner and fastglue BO gates) |
| `shapes.plan_manifest(plan_key)` | is the adaptive image set BUILT, when, and which M it serves |
| `webui/npu_plan.plan(M)` | the geometry the NPU would run (strategy, Mblk, ndisp, L1) |
| `webui/jobs/*/job.json` | the wall that size **actually took**, with the job id, date, step count and memory mode |

The panel on the page renders that JSON verbatim (answer line, path, M, model vs
measured, plan, flags that change it) and a scrollable table of every size the
product can speak about.  The page does no arithmetic of its own.

### The two answers the brief named

    $ py312 -m webui.capability --hw 1080 1920
    1920x1080 -> runs 1920x1088 (16-px pad) and crops back (M=8192): 不可用 —
    CPU noise refiner needs a 30x8160x8160x4 B = 7.44 GiB f32 score matrix
    (S=8160 image tokens) > 6 GiB ceiling; measured: S=4096 (1.88 GiB) completes,
    S=8160 (7.44 GiB) dies with 'device loop ran out of memory' 42 s in
    (job 20261103-150606-10d726).  Free memory and raise
    ZIMAGE_CHUNK_REFINER_GB if you accept the risk。

    $ py312 -m webui.capability --hw 1024 1024
    1024x1024 (M=4128): 可跑 but slow — 分块 path (delivered M=96 images + host
    f32 attention); model 187 s/step, 8 steps 0.41 h; no completed 8-step job at
    this exact size in this tree; 4 failed run(s) recorded, last: device loop
    failed (rc=2) (23 s).

and the same two through the submit path (the first is refused, the second
accepted):

    POST /api/jobs {"height":1080,"width":1920,...} -> 400
      1088x1920 (M=8192): CPU noise refiner needs ... 7.44 GiB ... (job ...10d726)
    POST /api/jobs {"height":1024,"width":1024,...} -> 202
      job 20261103-152053-eca758  exec=chunked  est_step=186.52

---

## 3. The size/answer table the WebUI produces

`$ py312 -m webui.capability --table` (33 rows; the envelope header first).
"分块·慢" = runnable on the delivered 96-row blocks + host f32 attention;
"专用镜像" = a gated per-shape image set; "不可用" = refused at submit.

```
envelope: padded image tokens <= 7327  (refiner score matrix <= 6.00 GiB)
          max square side 1360 ; native images up to M=288 ; 26 adaptive plans BUILT

480x854     M=1664   分块·慢      model 75 s/step, 8 steps 0.17 h   measured 359 s 20261023-105451-305858 (8)
720x1280    M=3648   分块·慢      model 163 s/step, 8 steps 0.36 h
1080x1920   M=8192   不可用       model 398 s/step, 8 steps 0.89 h
1152x2048   M=9248   不可用       model 457 s/step, 8 steps 1.02 h
854x480     M=1664   分块·慢      model 75 s/step, 8 steps 0.17 h
1280x720    M=3648   分块·慢      model 163 s/step, 8 steps 0.36 h
1920x1080   M=8192   不可用       model 398 s/step, 8 steps 0.89 h
2048x1152   M=9248   不可用       model 457 s/step, 8 steps 1.02 h
480x480     M=960    分块·慢      model 41 s/step, 8 steps 0.09 h
720x720     M=2080   分块·慢      model 92 s/step, 8 steps 0.20 h
1080x1080   M=4672   分块·慢      model 214 s/step, 8 steps 0.48 h
2048x2048   M=16416  不可用       model 897 s/step, 8 steps 1.99 h
128x128     M=96     专用镜像      -                                measured 32 s  20261023-192108-49507a (8)
128x208     M=160    专用镜像      -                                measured 59 s  20261021-154336-493a0d (8)
128x256     M=160    专用镜像      -
144x128     M=128    专用镜像      -
144x144     M=128    专用镜像      -                                measured 40 s  20261021-152449-5e8420 (8)
192x128     M=128    专用镜像      -
208x128     M=160    专用镜像      -                                measured 100 s 20261021-151731-9e6e88 (8)
256x256     M=288    专用镜像      -                                measured 77 s  20261022-153137-903f35 (8)
64x256      M=96     专用镜像      -
256x448     M=480    分块·慢      model 20 s/step, 8 steps 0.05 h   measured 104 s 20261102-212109-97ff9b (8)
256x640     M=672    分块·慢      model 29 s/step, 8 steps 0.06 h   measured 112 s 20261102-212327-8f3602 (8)
256x832     M=864    分块·慢      model 37 s/step, 8 steps 0.08 h   measured 121 s 20261102-212530-c8d015 (8)
384x384     M=608    分块·慢      model 28 s/step, 8 steps 0.06 h   measured 140 s 20261103-152812-d44f10 (8)
480x864     M=1664   分块·慢      model 75 s/step, 8 steps 0.17 h   measured 359 s 20261023-105451-305858 (8)
512x512     M=1056   分块·慢      model 45 s/step, 8 steps 0.10 h   measured 110 s 20261102-115213-3fca3c (8)
512x608     M=1248   分块·慢      model 54 s/step, 8 steps 0.12 h
640x640     M=1632   分块·慢      model 71 s/step, 8 steps 0.16 h   measured 329 s 20261102-213349-02cc8d (8)
768x768     M=2336   分块·慢      model 105 s/step, 8 steps 0.23 h  measured 68 s  20261030-221509-c28f1d (1)
1024x1024   M=4128   分块·慢      model 187 s/step, 8 steps 0.41 h  measured 112 s 20261030-221644-f60a3b (1)
1088x1920   M=8192   不可用       model 398 s/step, 8 steps 0.89 h
1920x1920   M=14432  不可用       model 769 s/step, 8 steps 1.71 h
```

**How to read it.**  The two columns are different kinds of number and the page
says so on every row: `model` is `chunk_estimate()` (an anchor-based estimate,
`not_a_speed_claim`), `measured` is a real job with its id, its date and its step
count.  A 1-step run is **never** presented as the wall of an 8-step shape -- be
suspicious of the 768²/1024² rows, which have no 8-step run at all.

**Where the measured column comes from.**  `webui/jobs/<id>/job.json`, which
`.gitignore` excludes -- the ids are cited as every earlier report cites them,
and `py312 -m webui.capability --table` rebuilds the column from whatever this
checkout has actually run.

**A caveat the table now records.**  The measured column mixes launcher envs:
jobs from 2026-11-02 and earlier do not say which flags were set, and the same
shape moves 2.5x between them (512^2 8-step: 110 s in the campaign config with
VAE off, 208 s in the product default with VAE on).  From this change on every
job record carries `env_flags` (ZIMAGE_BIGM, T22_DEVKILL, T22_FASTGLUE,
ZIMAGE_CHUNK_PIPE, the thread knobs, ...), so the next table can group by path
instead of guessing.

---

## 4. What a user can actually get today (the answer, in one paragraph)

**Renders today, no flags:** every native "专用镜像" shape (128x128, 64x256,
128x256, 144x128, 144x144, 192x128, 208x128, 128x208, 256x256 -- 32-100 s each,
measured), and any size whose padded image tokens are <= 7327 on the delivered
chunked path (measured at 384², 448x256, 512², 640x640, 854x480 ...; the model
puts 0.10 h at 512², 0.41 h at 1024², 0.89 h at 1080p -- the last two are
estimates, see the table).  **Does not render today:** 1920x1080 and anything
bigger (refiner memory, refused at submit), and any size above 1360² even if the
memory were there.  **Renders but slowly, with the plan built and unused:** the
adaptive path covers essentially every M in the envelope (26 plan image sets
BUILT) -- it is reachable only behind four env flags, it was slower in the one
end-to-end run, and its image does not match, so the UI offers the *estimate* and
never claims the plan as a path.

---

## 5. Files touched

**New**

| file | what |
|---|---|
| `webui/capability.py` | `describe(h,w)`, `table()`, `envelope()` + a CLI; every field sourced from resolve/chunk_gate/plan_manifest/npu_plan/jobs |
| `webui/vae_lean.py` | the landed VAE loader (task 1; see VAE-THREADS-LANDED.md) |
| `imaggers/vaeopt/api_probe.py` | the HTTP probe used to prove the refusal (1080p -> 400) and the acceptance (1024² -> 202) |
| `imaggers/vaeopt/adapt_cmp.py`, `adapt_cmp2.py` | the 384² three-way latent comparison |

**Modified**

| file | change | gated by |
|---|---|---|
| `webui/shapes.py` | `CHUNK_REFINER_MAX_BYTES` + `chunk_refiner_bytes()` gate; `fastglue_proj_rows()`/`fastglue_projection_active()` gate; `est["refiner_bytes"/"refiner_gib"/"fastglue_proj_rows"]` | `ZIMAGE_CHUNK_REFINER_GB` raises the ceiling; `T22_ADAPT_CHUNK` raises the BO |
| `webui/runner.py` | the submit refusal quotes `chunk_gate_reason` (the binding constraint) instead of "needs a device build"; `detail` carries both; `job["env_flags"]` records the launcher env | refusal only fires where it already fired |
| `webui.py` | `GET /api/capability` (`?height&width`, `?table=1`, bare = envelope) | additive route |
| `webui/static/index.html` | a "这个尺寸今天到底能不能跑" panel: answer line, path/M/model/measured/plan rows, the flags, and the size table; refreshed with the shape input | additive; no existing element changed |
| `webui/config.py` | `LAUNCH_FLAGS` + `launch_flags()` (record only) | n/a |

---

## 6. Reproduce

```powershell
py312 -m webui.capability --hw 1080 1920        # the refusal, with arithmetic
py312 -m webui.capability --hw 1024 1024        # the honest accept
py312 -m webui.capability --table               # the table above
py312 webui.py --host 127.0.0.1 --port 7860     # the panel is in the 生成参数 card
py312 imaggers/vaeopt/api_probe.py http://127.0.0.1:7860   # 1080p 400 / 1024 202
```
