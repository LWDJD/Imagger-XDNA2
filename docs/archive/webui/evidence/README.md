# E2E evidence — prompt-to-PNG and seed gate

## 00000) T22 glue (2026-10-21): the in-loop ledger, U1 pass-through, and the stop-loss account

`T22-GLUE.md` is the record. **NOT SHIPPABLE** -- this session had no device
window (process creation is denied), so it delivers the exclusive in-loop
ledger (`T22_GLUE_PROF=1` / `--prof`: 32 leaf buckets with `*other` residuals,
sum == step wall by construction), the bridge pass-through for U1 / glue-2
(default OFF), and an indirect account of the existing archives: of the 12.4 s
"glue" in the 128^2 loop, **4.48 s (15.7 %) sat outside every pmark**; two old
phase values are contaminated (`att.unpack_o` contained the ip dispatch;
`post.sbg` shared norm2's t0).  Predicted ceiling of host-only work: -1.7 s
(U1, measured) to -3.5 s (~28 %), below the 30 % stop-loss.

## 0000) T22 closurefix (2026-10-20): four process-cost buckets, and what the 16:06 FAIL really was

`T22-CLOSUREFIX.md` is the record (numbers: `t22_closurefix/summary.json`).
The closure table grows **15 -> 19 buckets**: `bridge_import_s` (py310
interpreter + bridge imports, split OUT of `startup_s` so the two add up to the
old `startup_s` exactly), `vae_spawn_s` (the one-shot py312 `vae_worker.py`),
`prewarm_pack_s` (the cold-build **wait**, moved out of `startup_s`; the
packer's own wall stays out of every bucket because it overlaps TE/VAE), and
`worker_drop_s` -- the fourth segment the task did not name: killing a LIVE
resident worker before a low-memory job.  That last one is what the 16:06 job's
**+1.1236 s FAIL** actually was: its `prewarm_wait_s` was **0.000** and its
artifact was **reused** (`pack_s=36.363` is the artifact's build stamp, not
time that job spent), so it was never the pack.

Seven real jobs in one session (product HTTP API): product default
**+0.0444 / +0.0335 s** in the same server process (p10 default on / off; the second
one is the warm-worker steady state, startup 1.221 s) and **+0.0960 / +0.0624 s** on
a repeat pair,
delivered bridge with the worker dropped **+0.0416 s** (the 16:06 shape; the same
job under the 18-bucket table is +0.6366), fallback `vae_worker` + cold build
**+0.0372 s** (`prewarm_pack_s=17.028`, `vae_spawn_s=1.147`).  A reconstruction
built from the RAW anchors (not the runner's bucket dict) closes 7/7 with a
**0.000 s** difference from the sealed sums.  latents `9f86aee9` on all five,
PNG `e7aef10c` on the NPU-chain paths and `104ff564` on the py312 fallback
(the two are not comparable -- PROD-REPORT §5), 960 dispatches / 0 retries,
`t3d_baseline --check` green before and after.  The cold rebuild produced a
byte-identical pack (`da2f45a6`), so the artifact on disk is the delivered one.

## 000) T21-3B3 (2026-10-19): the VAE decoder runs on the NPU for all six sizes

T21-3B3-VAE.md is the record.  Six product sizes (128x128, 64x256, 144x128,
192x128, 128x256, 256x256) decode through the real AutoencoderKL with
**conv / GroupNorm / upsample on the NPU** (the 92 gated xclbins) and SiLU /
add / attention on host fp32 (declared per stage in section 3).  Quality vs
the independent CPU VAE (diffusers fp32): 37.70-38.22 dB; run-to-run
byte-identical on every size; 1146 conv launches / 837.6 GMAC.  The 64x64
self-test reproduces the 3b1 figure to +0.089 dB with the residual localised
in the host stack by a single-variable control.  The WebUI now tries the NPU
chain first and falls back to the resident CPU decoder; a real 2-image job
closes at unaccounted 0.062 s and batch image 0 is byte-identical to its
single-image decode.  Speed is deliberately NOT claimed: 96-98 s per 128x128
image, dominated by one py310 process per launch.

Side-by-side: T21-3B3-<size>-side-by-side.png (NPU | CPU reference | |diff|x8)
and T21-3B3-official-128-side-by-side.png (official | CPU | NPU | diff).

## 00) T19 (2026-10-19): 256x256 closed — the last outstanding shape

`T19-256x256.md` is the record: two independent 8-step device runs landed on
**byte-identical** latents (`683050a95f4bd357`) although their intermittent FFN
ERT state-8 retries differed (200 vs 264, distributed differently step by
step); the chaos-free one-step comparison against the host f32 reference is
**corr 0.999964** (the delivered 128x128 path measures 0.999959 with the same
metric); audit 13/17 (delivered baseline) + ip 3/3; timing closure +0.05 s;
side-by-side `T19-256x256-host-vs-npu.png`; cache 938 KiB and
`webui/shape_manifests/256x256.json`; `/api/shapes` 256x256 -> ready=true.
The 8-step image is corr 0.90531 / 17.19 dB; T19-256x256.md §3.3 shows with a
control (host f32 loop seeded from the device's step-0 latent: 0.861 from the
host's own final, 0.948 from the device's) that this spread is the diffusion
sampler's sensitivity, not device error.

## 0) T16 (2026-10-18): the image bug, the fix, and the new evidence

**Supersedes §1's picture.** The 8-step PNG in §1 (39,282 B) is the *broken*
image: `t3_real.patchify` flattened each 2x2 patch channel-major while the
checkpoint's `all_x_embedder` was trained patch-major. Every block self-check
still passed because it compared NPU against the same host reference.

* Diagnosis + fix: T16-REPORT.md §1; fix is
  `webui/bridge.py::_install_token_layout_fix()`.
* Side-by-side (official 128² ZImagePipeline / host f32 before / WebUI before /
  WebUI after): T16-P0-1-side-by-side.png.
* Fixed 8-step job 20261018-170459-ed2a32: PNG 22,926 B, sha256
  fd7722e14cbba30ae1ad3eb4553ee784ed1b59eca024ec12f1c1365455d9accf,
  latent rms 0.939 (was 1.434), timing closure PASS (unaccounted 0.771 s).
* Batch cap 4 -> 16. The post-fix batch byte gate **PASSED**: batch 5 (all 5
  images) and batch 8 (all 8 images) are byte-identical to their own single-seed
  runs (`_scratch/image/t16_batch_gate.json`, `gate.pass=true`). The device stack
  did wedge once (~20 processes, deterministic 0xC0000005 in
  `xrt_coreutil.dll+0x1228ce`) and recovered; the gap-fill reran seeds 4/6/7 and
  batches 5/8. See T16-REPORT.md §3.3 for the curve (loop/image 26.4 -> 22.8 s).
* Product checklist: PRODUCT-CHECKLIST.md.
* §5/§6 below (batch-4 cold-start failures) are historical: a later 4x128² job
  (20261018-163636-f54297) did complete after two cold-start retries, and its
  262.96 s is fully accounted in T16-REPORT.md §3.2.

## 1) Final 8-step (post-reboot, 2026-10-18) — SUPERSEDED by §0

Job: 20261018-121643-c13f4e
Prompt: a mountain lake at sunrise
Steps: 8
Seed: 0
Batch: 1
Resolution: 128x128 (M=96)
Memory mode: fast (--cache-all; commit 17 GB at the time)
Text encoder: GPU, 10.86 s (load 7.80 + decode 1.57)
Startup: 37.53 s
Loop: 27.70 s (device 14.98 + staging 1.98; 1440 dispatches, 0 retries)
VAE: load 20.86 + decode 0.43 = 21.31 s
Total: 199.03 s (two cold-start retries 49.72 + 47.12 s; successful process 64.45 s)
PNG: E2E-prompt-8step.png, 128x128, 39,282 B
sha256: 90f0e2d628f5064e9cc548d944d4530246b1953f3a8973d30537611336f96ef8
calls.json: steps_rec=8, dispatches=1440, retries=0

The image is the real 8-step chain output for the encoded prompt. It is NOT
comparable to the T13 null-caption baseline (changing the caption changes the
latents by design); latents corr 0.907 and PNG PSNR 14.39 dB vs that baseline
only quantify the expected caption change.

## 2) Seed reproducibility (two independent complete runs)

Run A: job 20261018-121643-c13f4e, successful pass w121643c13f4ep0r2
Run B: job 20261018-122043-b83527, pass0 w122043b83527p0r1
Both: same prompt, steps=8, seed=0, batch=1, TE backend GPU, memory fast.

latents file sha256 (both): a25920e724ab983c292ee86f0e9e70dcad0a60428a882b3095ce1b66d44fe786
array_equal: True, shape (1,16,16,16)
PE npz file sha256 (both): 437634b665835e3f268cc4cefe29efa884061d9b643ed137e58bf8ff72dd1bdf
PE array sha256 (both): 90514d089ec4b35662d81a8ae000ea909dacb39347c4d25d389cd804bbe9a04c

Conclusion: reproducibility PASS.

Caveat: the automated gate job's second pass did not start (four fresh
processes all crashed at the first dispatch with 0xC0000005), so the job's gate
field is FAIL (infrastructure), not a latents mismatch. The two independent
completed runs above are the PASS evidence. See E2E-seed-gate.log.txt.

## 3) Safe mode / root causes isolated after reboot

* direct t13_run.py WITH --fast-codec -> first-dispatch 0xC0000005; WITHOUT -> success.
* bridge WITH refiner mmap -> first-dispatch 0xC0000005; WITHOUT -> success.

So the WebUI now defaults to fast-codec OFF and refiner mmap OFF; fast-host and
wpack-array remain ON. The safe-mode loop is 27.70 s (T13 fast path was ~24.5 s).
Both fast paths are opt-in (bridge --fast-codec / --refiner-mmap) and need
revalidation before they are turned on. Cold-start 0xC0000005 is still
occasional; the runner auto-retries up to 3 fresh processes and the bridge uses
os._exit after artifacts, so successful jobs exit 0.

## 4) Legacy 1-step smoke (earlier session)

E2E-prompt-1step.png / .log.txt: 1-step, seed 123, 72.74 s, 180 dispatches.
It is retained only as proof of the prompt->latents->PNG plumbing; it is not the
quality target and is superseded by the 8-step run above.


## 5) Batch-4 attempt (optional, 2026-10-18 12:27)

Job: 20261018-122752-721a59
Params: batch=4, steps=8, seed=0, prompt=a mountain lake at sunrise,
128x128, memory fast, VAE on.

Result: no image. Four fresh processes all crashed at the first dispatch with
0xC0000005 (rc 3221225477), steps=0 each:

    w122752721a59p0    49.53 s
    w122752721a59p0r1  43.60 s
    w122752721a59p0r2  47.53 s
    w122752721a59p0r3  45.65 s

This is the same cold-start class that also affects batch=1 occasionally; the
batch=1 8-step job succeeded after 2 retries, but batch=4 did not get a
successful dispatch in 4 tries. It is not evidence of a batch math error (the
runner never reached a layer). The unedited log is
webui/evidence/E2E-batch4-attempt.log.txt. Batch=1 remains the verified path.


## 6) Batch-4 low-memory retest (2026-10-18 12:50)

Job: 20261018-125018-974f83, batch=4, steps=8, seed=0, memory low
(--no-cache-all), VAE on. Four fresh processes all crashed at the first dispatch
with 0xC0000005 (rc 3221225477), steps=0, no images; total 147.5 s.
So batch 4 failed in both cache-all/fast and low-memory arms (8 attempts total).
Batch=1 remains the verified path. Full matrix: CONFIG-MATRIX.md.
