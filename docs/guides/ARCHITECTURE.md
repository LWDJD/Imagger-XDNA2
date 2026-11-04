# Architecture

## 1. What runs where

Z-Image-Turbo is a **DiT with 30 layers**, distilled to 8 steps. One 512² image is
**240 layer-steps**, and each layer-step issues **6 device dispatches** — 1440 per
image.

| stage | on the NPU | on the CPU |
|---|---|---|
| text encode | — | encoder worker (ggml/Vulkan *optionally*) |
| context refiner | — | torch f32, 2 layers |
| **noise refiner** | — | torch f32, 2 layers — **~16.6 s, the second largest CPU item** |
| qkv projection | **yes** (RT=11 plain 15) | pack/unpack |
| attention | — | **torch f32, ~18 s/image — the largest CPU item** |
| o projection | **yes** | pack |
| FFN gu | **yes** (RT=11) | staging |
| FFN dn | **yes** | staging |
| rope / norms | — | torch |
| VAE decode | — | torch (~4.3 s) or lean loader |

## 2. Why attention is on the CPU

Not a choice: at M=1056 the attention stage needs **196,096 B of L1** and a core has
**65,536 B**. It does not fit, at any tiling the project tried. See
`docs/reports/ATTN-SEG-REPORT.md` for the four variants and the one that was
rejected on quality (variant D: fast, corr 0.9919453, below the 0.9953 gate).

The CPU attention is at its own arithmetic ceiling: ~223–286 GFLOP/s measured,
and SDPA / blocking / permute all gave no improvement
(`docs/reports/HANDOVER-2026-11-01.md:63`).

## 3. The chain, and why the wall is D + H

The loop is **20 segments per block-step: 6 device + 14 host**, and every host
segment's output is the next dispatch's input. The wall is therefore
**`D + H`, not `max(D, H)`** — a sum, not a maximum.

Measured on 512²×8 (`docs/reports/HOSTPROF-REPORT.md`, `OVL-REPORT.md`):

| | |
|---|---:|
| wall | 98.10 s (older window; ~90 s today) |
| NPU busy | 39.99 s (40.8 %) |
| NPU idle | 58.11 s (59.2 %) |

Idle by cause: host attention 21.56, step preamble/refiner 20.91, python glue 5.91,
BO writes 3.69, codec 3.38, FFN staging 2.58.

**Only two host segments are off the chain** and can be hidden under a dispatch:
`rope(q)`, `rope(k)` (4.5 ms/layer) and the next layer's gu W payload (4.3 ms/layer).
Both are implemented (`T22_OVL`, default off) and worth **1.9 s/image**. Everything
else feeds the very next dispatch.

## 4. The device model — the constraint that shapes everything

**Every `pyxrt.kernel()` call creates its own hardware context** with its own
group-id space. Consequences, all measured:

| fact | evidence |
|---|---|
| one chain needs **18 contexts** | the delivered image set |
| the **~27th** context in a process is refused with `0xc01e0009` | `probe_ctx.py` |
| **two processes are impossible** — one XRT client per host | second chain dies 9.6 s in, `0xc01e0009` |
| **two `Dev`s each registering their own xclbins is impossible** — a uuid registered on handle 1 does not exist on handle 2 | `probe_two_dev_product.py` |
| **two chains in ONE process work**, if they share one read-only context set and each owns a complete private BO table | `Dev(share=)`, `probe_shared_ctx.py` |
| two dispatches never overlap in wall time — the array is serial | `B_done − A_done` = B's solo time |

The last line is the important one for expectations: **two chains do not buy device
concurrency.** They buy the ability to fill one chain's idle host time with the
other's device work. Measured: **−19.7 % per image**, device utilisation 44 % → 56 %.

**A fixed-tile image computes only its own tile count.** `plain_g15_rt11` is RT=11 =
1056 rows; handing it 8256 rows leaves 7200 rows of stale buffer. The projection
block guard (`proj_block_rows`, default 1056 rows) exists for this and must not be
disabled above M=1056.

## 5. Resolution independence

The row count is `M = ceil(h/8) * ceil(w/8) + 32` (32 caption tokens). The planner
sizes the projection in blocks of one image's tile count, so 512² takes one launch,
1024² takes four, 1080p takes eight. Each side must be 64..2048; anything outside is
refused with an explanation rather than an exception.

## 6. Reading order

1. this file
2. `docs/reports/ASSEMBLY.md` — the long-form structural description
3. `docs/reports/BOTTLENECK-MAP.md` — where the wall goes
4. `docs/reports/OVL-REPORT.md` — the overlap analysis, including what cannot move
5. `docs/reports/KNOWN-ISSUES.md` — before debugging anything
