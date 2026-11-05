# Post-reboot configuration matrix — 1-step cold-start probe

Date: 2026-10-18 (after machine restart)
Host: NPU Strix, XRT 2.21.0, NPU driver 32.0.20102.3930
Probe: webui/bridge.py, steps=1, seed=0, batch=1, prompt pe_test2, fresh process
per row. "OK" means calls/latents written; "crash" means rc 3221225477
(0xC0000005) at the first dispatch before any layer line.

All successful probes produced the same 1-step latents:
sha256[:16] = 065b52398a8d8e35, equal_ref=True. So the flag matrix did not
change the arithmetic; only process survival differed.

## Matrix (each row = independent cold-start process)

| id | flags | runs | OK | first-dispatch crash | success latents |
|---|---|---:|---:|---:|---|
| A | safe low: no fast-codec, no mmap, --no-cache-all | 4 | 2 | 2 | 065b52398a8d8e35 |
| B | fast-codec only, --no-cache-all | 4 | 0 | 4 | — |
| C | refiner mmap only, --no-cache-all | 4 | 2 | 2 | 065b52398a8d8e35 |
| D | fast-codec + refiner mmap, --no-cache-all | 4 | 2 | 2 | 065b52398a8d8e35 |
| F | fast-codec + cache-all | 2 | 1 | 1 | 065b52398a8d8e35 |
| E | 60 BO + wpack + refiner mmap | 2 | 1 | 1 | 065b52398a8d8e35 |
| E0 | 60 BO + wpack (no mmap) | 2 | 1 | 1 | 065b52398a8d8e35 |
| G | cache-all + refiner mmap (no fast-codec) | 2 | 0 | 2 | — |
| H | cache-all safe (no fast-codec, no mmap) | 2 | 0 | 2 | — |

Notes:
- A/C/D/E/E0 show the same ~50% cold-start loss; no flag is immune.
- B (fast-codec, low) is the worst at 0/4, but F shows fast-codec + cache-all can
  succeed (1/2); the sample is too small to call it deterministic.
- H/G (cache-all) were 0/4 in this window, but the official 8-step safe job and
  gate pass0 both succeeded with cache-all after retries. The window is noisy.
- 60 BO + wpack + mmap can allocate and complete sometimes (1/2); it did not
  produce a distinct memory-threshold failure signature.

## Longer runs / gates (same session)

| job | config | attempts | result |
|---|---|---:|---|
| 20261018-121643-c13f4e | 8-step safe, cache-all, batch1, seed0 | 3 | PASS: 8 steps, 1440 dispatch, 0 retry, loop 27.70 s |
| 20261018-122043-b83527 (pass0) | 8-step safe, cache-all, batch1, seed0 | 2 | PASS: latents == official |
| 20261018-122043-b83527 (pass1) | 8-step safe, cache-all, batch1, seed0 | 4 | FAIL: all first dispatch 0xC0000005 |
| 20261018-122752-721a59 | 8-step fast-codec, cache-all, batch4 | 4 | FAIL: all first dispatch 0xC0000005 |
| 20261018-125018-974f83 | 8-step safe, --no-cache-all, batch4 | 4 | FAIL: all first dispatch 0xC0000005 |

## Conclusion / recommendation

No single deterministic root cause was isolated. The failure is a process-level
cold-start 0xC0000005 at the first dispatch that occurs in every tested config;
fast-codec is the strongest correlating factor in the low-memory arm (0/4), but
it can succeed with cache-all (1/2). The evidence does not justify switching a
flag "on" to restore the 24.5 s fast path.

Final recommended defaults (unchanged from the delivered safe mode):

- fast-codec OFF
- refiner mmap OFF
- fast-host ON
- wpack-array ON
- memory_mode auto (fast cache-all when commit >= 9 GB; otherwise low)
- keep auto-retry (3 fresh processes) and os._exit after artifacts
- keep fast-codec / refiner-mmap as opt-in only; revalidate only on a clean power
  cycle with a larger sample

Measured safe-mode 8-step: loop 27.70 s (device 14.98 + staging 1.98), 1440
dispatch/0 retry; 24.5 s fast path NOT restored.

Remaining optional work:
1. batch-4 post-reboot success (both cache-all fast and low failed 4/4 each).
2. larger-sample cold-start matrix (>=10 processes/config) to separate the
   fast-codec and cache-all effects from the ambient ~50% failure rate.
3. re-enable fast-codec/refiner-mmap only after (2) is clean and latents remain
   byte-identical.
