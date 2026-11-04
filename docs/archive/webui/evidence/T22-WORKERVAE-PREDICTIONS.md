# T22 workervae: predictions (written BEFORE the device runs)

Task: worker x VAE coexistence (t22_workervae).  Plan: T22-VAE-REPORT.md §4.2
C-1 — the VAE dispatches in the py310 process that already holds the DiT
device context (dev_worker.py VAE link + t22_vae_srv connect mode).

Everything below is falsifiable and the gate is fixed here, before the runs.

## The shape being tested

* one pyxrt.device handle: the VAE Srv is handed a shim whose device()
  returns t18_run.Dev.dev (state-8: no second context, no second handle on
  any xclbin — the VAE images are a disjoint set from the four DiT images);
* the DiT loop and every VAE launch are serialized by one lock;
* the VAE Srv registers xclbins lazily, on the first VAE request;
* one launch per request, run.wait(TIMEOUT_MS) each (F-01/F-04/F-22),
  sync(FROM_DEVICE) before the output read (F-02 v2).

## Predictions

| # | prediction | falsified by |
|---|---|---|
| P1 | the link costs the cold start <= 0.05 s and the hot job startup stays 4.0-7.0 s in BOTH arms (link on / link off) | a startup outside 4-7 s, or >= 0.5 s between arms |
| P2 | the DiT bytes do not move: latents sha16 9f86aee9, pe sha16 90514d08 on every job | any other sha |
| P3 | the first 128x128 VAE decode through the worker succeeds (impl=npu) and its PNG is byte-identical to the same decode in spawn mode | crash, impl=cpu/mixed, or a different PNG |
| P4 | 4 alternating rounds (DiT -> VAE) all succeed, the worker survives, and the hot startup does not drift out of 4-7 s | any crash, any CPU fallback, drift |
| P5 | all six VAE sizes are byte-identical to the FROZEN 3b3 npu.npy when served by a fresh worker each (4 DiT + ~40 VAE images in one process) | any DIFF |
| P6 | the device-context budget does NOT bite at ~44 images (the report measured 20 OK, 40 untested); if it bites it does so at register_xclbin with 'Failed to create context virtual'/0xc01e0009, i.e. before any VAE launch | a failure that is not a context/registration error |
| P7 | the runner's DiT timing closure stays <= 1 s and the VAE stage keeps its own accounting (the chain is untouched) | closure > 1 s |
| P8 | the VAE BOs fit the fast-memory budget: the worker's commit stays under the 9 GB threshold, and the process does not exceed ~10 GB RSS | MemoryError, or the runner's memory gate flips to low-memory |

## Instrumentation (already in place)

* _scratch/vae/t22_workervae_ab.py: startup (cold + hot A/B), dit (bytes),
  coexist (alternating rounds + PNG sha), sixsize (one fresh worker per size,
  chain CLI in connect mode), vae (spawn-mode baseline in a fresh process).
  Every result JSON goes to _scratch/vae/out/t22_workervae/.
* worker log per run under _scratch/vae/out/t22_workervae/logs/.
* on any death the harness records worker_rc_after and the log tail.

## Decision rules (fixed now)

1. P2 or P5 fails -> the C-1 shape is REJECTED; the runner change is not
   made (it comes last) and the byte diff is recorded.
2. The worker dies inside the VAE (no error frame, rc=0xC0000005) -> per
   stop-line ①: 30 min of localization, then report the death cause; do not
   ship the runner change.
3. register_xclbin fails with a context error -> the image budget is the
   wall; report the exact image count at failure and stop (no workaround
   that weakens a byte gate).
4. Cost > 1.5 design-days -> stop and report with the data.

## What is NOT claimed

* not tested: two different VAE sizes decoded by the SAME worker (the product
  worker is shape-pinned to 128x128, so this is not a product path); the
  six-size gate uses a fresh worker per size by design.
* not tested: concurrent DiT + VAE (the product is sequential, and the lock
  forbids it).
* the CPU fallback path is unchanged; a link failure must land there, not in
  a crash.
