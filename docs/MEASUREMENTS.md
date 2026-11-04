# How to measure this project without fooling yourself

This file exists because the project spent a long time producing confident numbers
that were measured on the wrong array. Every rule below is there because breaking it
produced a wrong conclusion.

## 1. The wall clock is not the evidence

**Loop wall resolves to only +-2 s.** Any effect smaller than that is invisible in the
wall, and you will 'confirm' whatever you expected.

* Report the **per-section columns** (`[fastglue]`, `[devkill]`, `[ffnprof]`) as the
  evidence, and the wall as corroboration.
* A step line's `host` field is **not** a wall clock -- it runs 1.43-1.57x the real
  wall because it double-counts. `dev` is a full device interval only when
  `T22_OVL=0`.

## 2. Cross-session drift is larger than most effects

Measured device-state error between sessions: **23-32 %**. The correct comparison is
**round-robin inside one process/session**, interleaved A/B/A, at least two pairs.
Never compare a number from this morning with a number from this afternoon.

## 3. The quality gate is 8 steps, and a per-call correlation cannot predict it

Accept a change only if, over **8 steps**:

* the latents are **byte-identical**, or
* `corr >= 0.9953` against the reference.

A per-call attention error is amplified by roughly **four orders of magnitude** by
the 8-step chain, so a per-layer check that looks perfect says nothing about the
image.

**Known references** (512x512 x8 unless stated):

| shape | latents sha256 |
|---|---|
| 512x512 x8 | `2abf91331200d069...` (file) / `4cbf4f4a1b980c80` (array) |
| 1024x1024 x1 | `af993cadfecbc419...` |
| 1920x1080 x1 | `7ba858a69f68e0bd...` |

## 4. The provenance floor

Two runs of the *same* arm differ by exactly 0. Two runs of arms that differ in a
knob the record did not capture differ by:

| steps | floor |
|---|---:|
| 1 | **8.915e-4** |
| 8 | **1.056e-2** |

A margin smaller than the floor proves nothing. The switch that rejected
`T22_PLAN_FUSE_GU` had a **6.0e-5** margin -- **14.8x below** the 1-step floor.

## 5. Prove the flag reached the process

*(This one has bitten this project repeatedly.)*

`job.json`'s `env_flags` is **`LAUNCH_FLAGS`** -- a fixed list in `webui/config.py`.
A flag not in that list will never appear there, so its absence proves nothing. And
`t22_run.ps1 -Extra` only passes names that are **already in the recipe**
(`RECIPE_ENV`); anything else is silently filtered out.

Check properly:

* `imaggers/repro/last/<tag>.preflight.json` -> `extras` (authoritative)
* the job log -> one `[extra] ARM OVERRIDE NAME=VALUE` line per arm flag

A run that produced the reference bytes with an arm flag 'on' is the signature of a
flag that never landed.

## 6. Numbers are only comparable at the same power state

Battery vs AC, measured: **loop x1.075, startup x1.498, total x1.234**. Quote AC
only, and check that preflight P12 says `ac`.

## 7. Every claim needs a negative control

An arm that passes proves nothing unless the gate is shown to be able to fail. Two
controls that bite:

* `T22_CHAIN2B_SHARE=1` -- makes both chains share one scratch set; both images move.
* a deliberately too-small FFN BO -- makes the tile guard refuse.

## 8. Hash conventions are not interchangeable

Two different 'sha' values exist for the same latents and both are correct:

| convention | 512x512 x8 | what it hashes |
|---|---|---|
| file form | `2abf91331200d069` | the whole .npy file (128 B header + payload) |
| array form | `4cbf4f4a1b980c80` | the array bytes only |

`imaggers/util/hashconv.py` normalises, and **refuses** to compare across
conventions rather than silently reporting a mismatch.

## 9. 'Done' is not evidence

A job that reports `done` may have written garbage: this record contains an image
that was a black band plus vertical stripes, produced by a projection dispatch that
could not compute the rows it was handed. **Decode the image and look at it**, or at
minimum check its content statistics.