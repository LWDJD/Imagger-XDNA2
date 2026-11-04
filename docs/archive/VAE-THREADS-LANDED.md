# The two VAE wins, landed and proved with 8-step runs

**2026-11-03 · 512x512 · 8 steps · seed 7 · batch 1 · prompt 场景：苹果树、山顶、夕阳
· every arm recorded its memory mode (all `fast`) and its latents sha**

> **2026-11-04 ANNOTATION — this note is history, not the current VAE-thread rule.**
> It is kept verbatim; where a sentence is now false it is struck through in place
> and the replacement is named under it.  Nothing was deleted or silently rewritten.
>
> * **What still stands:** the lean loader (row 1: load 19.3 -> 3.9 s, PNG
>   byte-identical); the 512^2 1-LSB measurement (4-7 px of 786,432, 1 LSB); the
>   8-step A/B in section 2 *as measured that day*; and the derivation in section 4
>   (122.08 - 16.07 + 3.92 = 109.93 s).
> * **What is superseded:** the "20 threads is an env opt-in / the default stays the
>   shipped bytes" framing (row 2 and section 3), the 512^2 decode pair
>   `6.4 -> 4.0 s`, the `4.01 s instead of 6.19-6.45 s` figure, and the 1088x1920
>   decode pair `39.53 vs 85.34 s`.  The decode pool is now a **geometry rule**, and
>   the 1088x1920 10-thread sample was **contended**.
> * **What replaced it:** `webui/config.py` now decouples `vae_threads()` /
>   `vae_decode_threads(h, w)` from `loop_threads()`; the decode runs at
>   `VAE_DECODE_THREADS = 20` iff `latent_H * latent_W % 5 == 0`, else at
>   `VAE_DECODE_THREADS_GUARD = 10` (`VAE_SILU_MODULUS = 5`,
>   `VAE_THREAD_RULE_DEFAULT = True`).  Evidence:
>   `_scratch/par/vaethreads/REPORT.md`, `_scratch/par/vae1lsb/REPORT.md`.
> * **Where the current numbers live:** `_scratch/par/final/STATE.md`.

```
THE ONE SENTENCE: the diffusers-free decoder is landed as the product default and
is byte-identical end-to-end (same latents, same PNG bytes), the load bucket drops
19.3 -> 3.9 s mean, and the 20-thread decode is a one-flag opt-in because at 512^2
it moves 4-7 pixels of 786,432 by 1 LSB -- it is NOT byte-identical.
```

> **2026-11-04:** "the 20-thread decode is a one-flag opt-in" is no longer true of
> the product.  The thread count is a function of the **latent geometry**, not of a
> flag: 20 threads at `latent_H*latent_W % 5 == 0`, 10 otherwise.  At **512^2**
> (latent 64x64 = 4096, `4096 % 5 == 1`) the rule itself selects **10**, so the
> 512^2 sentence above still describes the bytes the product ships at 512^2 -- it
> is the *mechanism* that changed, not that image.  Taking 20 threads at 512^2 now
> needs an explicit pin (`ZIMAGE_VAE_THREADS=20`), which is exactly what section 3
> describes.

---

## 0. One screen

| | what | measured | verdict |
|---|---|---|---|
| 1 | **lean VAE loader** (`webui/vae_lean.py`) | load 19.3 s -> 3.9 s (mean of 2+2 runs); PNG **byte-identical** (a131cc55 both ways), latents identical | **landed, default ON** |
| 2 | **VAE decode threads = 20** | ~~decode 6.4 -> 4.0 s~~ **re-based 2026-11-04:** 512^2 decode **5.74-6.10 s at 10** -> **4.64-4.79 s at 20**; PNG **not** byte-identical: 4-7 px of 786,432 move by 1 LSB | ~~**opt-in** (`ZIMAGE_VAE_THREADS=20`), default stays the shipped bytes~~ **superseded 2026-11-04: default = the geometry rule** (20 iff `latent_H*latent_W % 5 == 0`, else 10) -- see section 3 |
| 3 | 8-step wall, my baseline | 220.7 s -> 208.2 s mean (**-12.6 s, -5.7 %**) | |
| 4 | the BOM's 122.08 s job | 122.08 - 12.2 = **~109.9 s** | **not ~106 s** -- see section 4 |

---

## 1. What was actually landed (files, and what each line does)

| file | change | rollback |
|---|---|---|
| `webui/vae_lean.py` | **new**. `_scratch/vaeopt/vae_lean.py` verbatim (class bodies byte-identical, checked) + a repo-relative checkpoint path + a `--gate` CLI that re-derives the diffusers comparison in-tree | nothing imports it unless the flag is on |
| `webui/vae_local.py` | `ZIMAGE_VAE_FAST` **defaults to 1**; the shipped loader is kept line for line in `_load_shipped()` and is used on ANY lean failure; the decode applies `config.vae_threads()`; the result dict gains `loader`/`threads` | `ZIMAGE_VAE_FAST=0` |
| `webui/config.py` | `loop_threads()`, `vae_threads()`, `VAE_DECODE_THREADS=20`, `LAUNCH_FLAGS`/`launch_flags()` (one home for the thread rule) | both stay 0 = torch default |
| `webui/chunked.py` | `_torch()` reads the thread count through `config.loop_threads()`; **behaviour identical** (0 still means "do not call set_num_threads") | n/a (refactor) |
| `webui/vae_worker.py` | the one-shot fallback gets the same loader + thread rule, so the fallback cannot silently diverge | `ZIMAGE_VAE_FAST=0` |
| `webui/runner.py` | records the launcher env per job (`env_flags`) -- additive; the 2026-11-03 audit needed it | n/a |

The loader is a pure *load* change: the decoder is rebuilt from the same bf16
checkpoint with the same op order, so `decode()` is bit-identical
(`py312 -m webui.vae_lean --gate _scratch/image/w0630319dd130p0_latents.npy`:
equal=True maxabs=0 rel_rms=0 png 571c96cc386285e7 both ways, 49,545,475 params).

---

## 2. The 8-step A/B (interleaved, same box, same config)

Loop config in every arm: **product defaults** (no ZIMAGE_BIGM / T22_* flags), so
the *only* difference between the arms is the VAE. Memory mode was `fast` in all
five runs (`config.choose_memory_mode` auto: commit 11.1-11.6 GB >= 9.0 GB), and
the loop's latents are identical in every arm: **086c9ee8f0520655**.

| arm | VAE loader | threads | startup s | loop s | **vae_load s** | vae_decode s | **total s** | PNG sha16 | job |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| **A1** shipped | diffusers | 10 (default) | 13.61 | 179.19 | **22.13** | 6.45 | **223.66** | a131cc55 | 20261103-144345-de7620 |
| **A2** shipped | diffusers | 10 (default) | 12.12 | 182.07 | **16.55** | 4.74 | **217.81** | a131cc55 | 20261103-145954-8c65cf |
| **B0** landed, threads shipped | lean | 10 | 13.85 | 181.85 | **3.61** | 6.19 | **207.85** | a131cc55 | 20261103-145128-56b1f5 |
| **C1** landed, product default | lean | 10 | 13.54 | 182.43 | **4.22** | 6.00 | **208.48** | a131cc55 | 20261103-145602-8a3eee |
| **B1** landed + 20 threads | lean | 20 | 14.22 | 179.22 | **4.18** | 4.00 | **204.34** | 0f689f8b | 20261103-144746-ad9978 |

* **A vs B0/C1: the PNG is byte-identical** (`a131cc5566d2da3e`, 0 pixels differ,
  `open(A,'rb').read() == open(C,'rb').read()` True).  The loader is byte-neutral
  end-to-end, not just in a unit test.
* **A vs B1**: 4 pixels of 786,432 differ, each by **-1 LSB** (f32: 432,950 values
  move by <= 1.76e-06, PSNR 143.5 dB).  Same latent, same loader -- the thread
  count alone moves the bytes, reproducibly, in both decoders (diffusers at 20
  threads gives the same 0f689f8b).
* The loop is untouched: loop_s 179.2-182.4 across all five arms (+-1.7 %).

### The +-2 s caveat is optimistic on this box today

The same env in two arms gave startup 12.12 vs 13.61 s and vae_load 16.55 vs
22.13 s.  The box is shared and slower than at 06:30 (job 063031 ran the loop in
86.1 s; the same code today needs 179-182 s).  So the **bucket** delta is the
number to quote, not the total:

    vae_load  mean(A) 19.34 s  ->  mean(B0,C1) 3.92 s   = -15.4 s
    vae_decode mean(A) 5.59 s  ->  mean(B0,C1) 6.10 s   = +0.5 s (noise, same 10 threads)
    total     mean(A) 220.74 s ->  mean(B0,C1) 208.16 s = -12.6 s  (-5.7 %)
    total     B1 (20 threads)  204.34 s                  = -16.4 s vs mean(A), -3.8 s vs the default

---

## 3. Why the VAE threads are NOT the default

The campaign measured "10 of 1,572,864 values move by 1 LSB, same PNG".  In the
product at 512^2 the PNG is **not** the same: 7 pixels (t5 probe) and 4 pixels
(B1 arm) of 786,432 move by exactly 1 LSB.  That is a product decision, not a
measurement, so the default stays the shipped byte path and the win is one
environment variable away:

    ZIMAGE_VAE_THREADS=20        # 4.01 s instead of 6.19-6.45 s at 512^2
    ZIMAGE_CHUNK_THREADS=20      # same thing via the loop's knob (the VAE follows it)   <-- 2026-11-04: FALSE, see the note below

> **2026-11-04 — the second line is FALSE; it is the coupling that was removed.**
> The VAE decode **no longer** follows the loop's knob.  `webui/config.py` decouples
> `vae_threads()` / `vae_decode_threads(h, w)` from `loop_threads()`, deliberately:
> the decode's bytes move with the thread count, so a loop knob must not be able to
> move an image.  `ZIMAGE_CHUNK_THREADS=20` now sets the loop's pool and **nothing
> else**.  The decode pool comes from a geometry rule: `vae_decode_threads(h, w)`
> returns **20 threads iff `latent_H * latent_W % 5 == 0**, else **10**.
>
> *Why the modulus is 5:* the decoder's 29 SiLU activations have numel 512*X,
> 2048*X, 4096*X, 8192*X, 16384*X with X = `latent_H*latent_W`.  torch's CPU SiLU
> runs a vectorised body and finishes each thread range with a **scalar remainder**,
> so the remainder positions -- and the last 1-2 ulp of a handful of elements --
> move with T.  All five numels are multiples of 320 exactly when `X % 5 == 0`, and
> then the remainder is identical at T=10 and T=20.  The other 88 arithmetic ops
> (oneDNN conv, GroupNorm, GEMM, SDPA, upsample, add) are bit-identical across
> thread counts: measured, not assumed.
>
> Verified end to end, 0 f32 values moved, identical PNG: 320^2, 640^2, 800^2,
> 960^2, 1088x1920, 1280x720, 480^2.  The control 576^2 (`5184 % 5 == 4`) moved
> 563,028 values / 15 px by 1 LSB -- i.e. the predicate was tested on both sides.
> `ZIMAGE_VAE_THREADS` still pins an arm and an explicit pin still wins;
> `ZIMAGE_VAE_THREAD_RULE=0` restores this note's behaviour (decode follows the
> explicit env, else torch's default).  Sources:
> `_scratch/par/vaethreads/REPORT.md` 3.1-3.4, `_scratch/par/vae1lsb/REPORT.md`.
>
> **2026-11-04 — re-based decode numbers for 512^2** (the row-2 figure above):
> 10 threads **5.74 / 6.10 s**, 20 threads **4.64 / 4.67 / 4.79 s**, PNG
> `571c96cc386285e7` at 10 and `3fae09be69664af8` at 20.  So the line above should
> read "4.6-4.8 s instead of 5.7-6.1 s at 512^2", not "4.01 vs 6.19-6.45".

~~At 1088x1920 the campaign measured the 20-thread decode **bit-identical**
(maxabs 0, 39.53 s vs 85.34 s), so large sizes should take the flag even under
this project's byte-identity rule.~~

> **2026-11-04 — the claim survives, its speed pair does not.**  1088x1920 (latent
> 136x240 = 32,640; `32640 % 5 == 0`) **is** bit-identical at 20 threads -- maxabs 0,
> 0 of 6,266,880 f32 values moved, PNG `3c813528c457d468` in all six runs -- so large
> sizes do take the win.  But it is no longer "the flag": the rule selects it.
>
> The **`39.53 vs 85.34 s`** pair is **superseded**: the 10-thread sample was taken
> on a **CONTENDED** box (two processes OOM-killed during that sweep, and the same
> report notes timings moving +-40 % on a shared box).  The honest measured pair is
> **10 threads 51.24 / 58.01 s vs 20 threads 39.81 / 40.32 / 40.42 / 43.98 s**, i.e.
> **-11 to -13 s on the 1080p stage (1.27-1.46x), NOT -46 s (2.16x)**.  The
> 20-thread side of the old pair (39.53 s) agrees with the new samples to 2 %; it is
> the 10-thread side that was wrong.  Evidence:
> `_scratch/par/vaethreads/REPORT.md` 3.1 and 4.


---

## 4. The new 512^2 wall: ~110 s, not ~106 s

`_scratch/vaeopt/VAE-OPT-REPORT.md` predicted 122.08 -> 106.22 s by swapping the
16.07 s in-product load for **0.21 s**.  That 0.21 s is the **torch-warm** load
(the campaign's `b_warm_ready_s`).  The product's py312 server has NOT imported
torch when the VAE stage starts, so it pays the torch import inside the job:

| | load |
|---|---:|
| shipped in-product (job 063031) | 16.07 s |
| lean, **torch cold** (this A/B, mean of 4) | **3.92 s** |
| lean, torch warm (campaign, t1d_loadab) | 0.21 s |

    122.08 - 16.07 + 3.92 = 109.93 s      (+-2 s wall, +-1 s box drift)
    and on my own baseline: 220.74 -> 208.16 s (-12.6 s, -5.7 %)

If a future server imports torch before the first VAE job (e.g. by touching
`webui.vae_lean` at boot) the load falls to ~0.3 s and the same change is worth
-15.9 s, i.e. 106.2 s.  That is a different change and it is not in this one.

---

## 5. Reproduce

```powershell
# the loader gate (diffusers oracle, in-tree)
py312 -m webui.vae_lean --gate _scratch/image/w0630319dd130p0_latents.npy

# the decode-only thread A/B (product decoder, one process, interleaved)
py312 _scratch/vaeopt/t4_dec_threads.py

# the product VAE entry point, both ways (writes a job-shaped record)
$env:ZIMAGE_VAE_FAST='0'; $env:ZIMAGE_VAE_THREADS='0'
py312 _scratch/vaeopt/t5_vae_local.py --tag ship
Remove-Item Env:ZIMAGE_VAE_FAST,Env:ZIMAGE_VAE_THREADS
py312 _scratch/vaeopt/t5_vae_local.py --tag landed

# one 8-step arm through the product runner (the A/B above)
$env:ZIMAGE_VAE_FAST='0'; $env:ZIMAGE_VAE_THREADS='0'
py312 _scratch/vaeopt/ab8.py --tag A1
```

**Where the records live.**  The job records quoted above are
`webui/jobs/<id>/job.json`; `.gitignore` excludes `webui/jobs/`, so a fresh
checkout cannot re-open them -- the ids are cited the way every earlier report in
this tree cites them.  The A/B rows are also mirrored (numbers only) in
`_scratch/vaeopt/out/ab8_<tag>.json`, and `_scratch/vaeopt/ab8.py` re-runs any
arm.

## 6. Also measured on the way (and not caused by this change)

With the campaign env (`T22_HOSTKILL=1 T22_DEVKILL=1 ZIMAGE_BIGM=1
ZIMAGE_CHUNK_PIPE=0`) the loop is **not reproducible**: two 1-step runs with the
same env, same seed, same `latents0` (2d87cfbb) produced different final latents
(604964e7 vs f9516369).  The product-default config is deterministic (two 1-step
runs: c81a6fd6 both).  That is why this A/B uses the default config, and it is a
finding the campaign config's byte-identity claims need to answer.
