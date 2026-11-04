# Denoising loop and end-to-end orchestration

This is step 6 of `ASSEMBLY.md`.  Written before the pieces land so the loop is
not designed under pressure.  The sampler is `FlowMatchEulerDiscreteScheduler`
(`zimage-port/ref_probe_env.py:68`), so the loop is short and its arithmetic
is trivial — the whole question is where each tensor lives.

## The loop

```
latents = randn(seed) * init_noise_sigma          # (1, 16, 16, 16) for 128x128
for t in scheduler.timesteps:                     # 8 steps
    latent_in = latents
    noise_pred = transformer(latent_in, t, text_embeds)   # 34 blocks
    latents = scheduler.step(noise_pred, t, latents)      # one Euler step
image = vae.decode(latents / scaling_factor + shift_factor)
```

`transformer` is the part that must run on the NPU with BlockFP16 weights.  The
scheduler step and the VAE can stay on the CPU for the first image; see
`KNOWN-ISSUES.md` for the scope decision.

## The host-interface budget: at most TWO input streams per stage

Measured on this device and recorded in `ASSEMBLY.md`:

    error: no ShimNOCTile on the device has 0 input/1 output DMA channel(s)
           free: all 1 ShimNOCTile(s) are at 0/2 input, 2/2 output channels used
    note: this is the device's total ShimNOCTile DMA budget, not a placement
          choice; fan traffic through a MemTile, or reduce the number of
          DMA-attached ObjectFIFOs feeding this tile

A compute tile also has `core_dma_channels_in == 2`.  So **every stage of the
DiT block must be planned as at most two host-fed input streams plus one
output.**  Extra per-token parameters (adaLN scale/shift/gamma) ride inside one
packed stream element rather than getting their own channel, and anything that
genuinely needs a third stream has to fan through a MemTile.

This constraint shapes the block's whole host interface, and it is the reason
the adaLN parameters are packed rather than streamed separately.

## CORRECTION: `vae_scale_factor` is 8, not 16

Measured, not assumed — `imaggers/pipeline/vae_decode.py --self-test`:

    vae 83,819,683 params   scaling 0.3611   shift 0.1159
    latents (1, 16, 8, 8) -> image (64, 64, 3)

so an 8x8 latent gives a 64x64 image: **`vae_scale_factor = 8`**, i.e.
`block_out_channels` has 4 entries, not 5.  The earlier note in this file said
16 and that was wrong.

The correct geometry:

    128x128 image  ->  latent 16x16  ->  16x16 / (2x2 patch) = 64 DiT tokens

which matches the `img_tokens=64` figure recorded elsewhere.  `denoise_loop.py`
must be run with `--vae-scale 8` and `--height/--width` giving latents of
`(H/8, W/8)`.

## The stack runs at the VENDOR geometry too

    34 block calls at seq=96 in 20.9 s (614 ms/call)
    output absmax 72.0941

Compared with 565 ms/call at seq=64, the 96-token geometry costs about 9%
more per call.  So the whole 34-block transformer is exercisable at exactly
the shape the vendor measurement lives in (`hidden [1, 96, 3840]`).

## The 96-token reference image is BYTE-REPRODUCIBLE

    oracle96.png       35,845 B   sha256 919e547110e4205c...
    oracle96_repro.png 35,845 B   sha256 919e547110e4205c...
    reproducible: True

Produced at the vendor geometry (`seq=96`, 32 context + 64 image tokens),
run via `run_full_host.py --seq 96`.  It differs from the seq=64 run
(35,958 B), so the geometry switch is confirmed to take effect.

## The reference image is BYTE-REPRODUCIBLE

    oracle.png       35,958 B   sha256 317bf014811100e9...
    oracle_repro.png 35,958 B   sha256 317bf014811100e9...
    reproducible: True

Two runs at the same seed produce identical bytes, so `pipeline/oracle.png` is
a strict oracle: the NPU block can be judged against it without tolerance
ambiguity on the host side.

## The two conditioning tensors are DIFFERENT things

A distinction the text-encoder work surfaced, which the earlier notes blurred:

| tensor | shape | source | consumer |
|---|---|---|---|
| `pe` | `(1, L, 2560)` float32 | text encoder's `hidden_states[-2]`, bf16 on device | the 32 CONTEXT tokens of the 96-token stream |
| `t_emb` | `(1, 256)` float32 | computed from the **timestep** (`generate.py:193`, `t_emb`, `:128-151`) | `adaLN_modulation.0 = Linear(256, 15360)` |

So the `embeds` argument in `block_adapter.py` is the **timestep** embedding,
not the text one, and current consumers pass `np.zeros((1, 256))` as a
placeholder.  The text encoder's output is a separate tensor that populates the
context half of the token stream.

`pipeline_z_image.py:236-247` emits `hidden_states[-2]` as `(L, 2560)` bf16;
`zimage-port/generate.py:532-536` stores it as `pe` float32 `(1, L, 2560)` in
the `--embeds` npz alongside scalar `prompt`/`cfg` and optional `ne`.

## GEOMETRY: 96 tokens is the block's design value, 64 is the image token count

There are TWO different token counts in this project and they are not in
conflict:

* **96** — the block's design geometry.  The vendor measurement lives at
  `hidden [1, 96, 3840]` with `cos/sin [1, 96, 64]` and `adaln [1, 256]`
  (`weights/FINDINGS.md:230-231`), so every kernel was verified at seq=96 and
  `tile_rows` was measured at both seq=64 and seq=96.
* **64** — the image token count for a 128x128 image: a 16x16 latent at a 2x2
  patch is 8x8 = 64 tokens.

The gap is **confirmed** by `zimage_pipeline.py:95-114`:

    def position_ids(S_CAP, grid):
        axes = [torch.arange(x0, x0 + span) ...]
        grids = torch.meshgrid(axes, indexing="ij")
        cap_ids = coordinate_grid(size=(cap_len, 1, 1), ...)
        img_ids = coordinate_grid(size=(F_t, H_t, W_t), ...)
        img_ids = torch.cat([img_ids, pad], dim=0)
        return cap_ids, img_ids

so the model keeps `cap_ids` (context) and `img_ids` (image) separate and the
96 tokens are `S_CAP + image tokens`.  **Measured through
`block_adapter.rope_position_ids(32, (8, 8, 1))`:**

    cap_ids (32, 3)   img_ids (64, 3)   total tokens 96

so `S_CAP = 32` and a 128x128 image is 64 tokens — 32 + 64 = 96, exactly the
vendor's `hidden [1, 96, 3840]`.  RoPE is **three-axis**:
`axes_dims = [32, 48, 48]` summing to head_dim 128, `axes_lens = [1536, 512,
512]`, with `cos/sin [1, 96, 64]` (= head_dim / 2).  `zimage_pipeline.py:328-333`
shows the two id sets fed to separate `RopeTables` lookups.

## The multi-step path is stable

Two full passes through the 34-block stack:

    residency over 2 passes: reads 1012, unpack calls 0
    forward: 68 block calls (2 steps x 34 blocks) in 38.4 s (565 ms/call)
    output absmax 195.9873

So the zero-decode property holds across steps, not just for one, and the
per-call cost is stable (565-572 ms across runs).

## The 34-block stack runs

`imaggers/pipeline/stack34.py --steps 1` drives the whole transformer in
forward order:

    34 block calls (1 step x 34 blocks) in 19.4 s (572 ms/call)
    output absmax 83.7536
    residency: 506 reads, 0 unpack calls, 6,913,878,016 B resident

So the stacked-transformer path is verified on the host side, and the
residency property holds at full scale.  When the NPU block lands it replaces
the oracle block inside this loop, and the 34-block residency claim becomes a
measured silicon result rather than a host one.

## The reference IMAGE is produced

`imaggers/pipeline/run_full_host.py` runs the whole host path with the oracle
block and writes a real PNG:

    block  : oracle dit_block_float
    scheduler shift 3.0   timesteps [1000.0 947.5 882.8 800.8 693.8 548.0 338.0 8.9]
    latent absmax 3.73 -> 1.92 over 8 steps (a sensible convergence trajectory)
    loop wall 4.4 s
    VAE -> (128, 128, 3) -> oracle.png (35,958 B)

So `oracle.png` is the reference the NPU block must reproduce at the same seed.
The chain is `seed -> FlowMatchEulerDiscreteScheduler -> 8 x dit_block_float ->
VAE`, every stage the reference's arithmetic.

**What remains for an NPU image is exactly one substitution**: replace the
oracle block with the NPU block behind `block_adapter`'s call signature and
compare.  Everything else in the path is done and measured.

## The block contract is fixed and verified

`imaggers/pipeline/block_adapter.py` defines the interface the NPU block must
honour:

    block(latents, t, embeds) -> same shape as latents
        latents : (1, 16, H/8, W/8) float32
        t       : scalar timestep from the scheduler
        embeds  : (1, 256) float32 conditioning

and it patchifies before calling the block: a 16x16 latent at a 2x2 patch is
**8x8 = 64 tokens** of 64 values each, which is the `img_tokens=64` figure.

Verified end to end against the oracle's `dit_block_float`:

    latents (1, 16, 16, 16) -> numpy block -> (1, 16, 16, 16) absmax 4.8050
    ADAPTER OK

So the host reference path (`denoise_loop` -> `block_adapter` ->
`dit_block_float`) is complete, and the NPU block drops in behind the same
call signature.

## The FULL host path produces a PNG

    seed -> FlowMatchEulerDiscreteScheduler -> 8 steps -> VAE -> PNG
    VAE decode: image (128, 128, 3) -> standin.png (37,592 B)

`imaggers/pipeline/denooise_loop.py` now runs the entire host path end to end
and writes a real PNG.  The block callable is still the stand-in, so the image
is noise — but **every other stage is the reference's**: the scheduler is the
checkpoint's, the timestep schedule is the one diffusers produces, the latent
geometry is 16x16 for a 128x128 image (`vae_scale_factor = 8`), and the VAE
decode is `latents / scaling_factor + shift_factor` exactly as
`ref_native_128.py:580-582` does it.

So when the DiT block lands, the remaining work is one flag:

    python denoise_loop.py --block <module:callable> --steps 8 --seed 0

and then compare against `zimage-port/ref_native_128.py` at the same seed.

## The loop runs on the REFERENCE scheduler

`imaggers/pipeline/denoise_loop.py` now drives the real
`FlowMatchEulerDiscreteScheduler` from the checkpoint:

    scheduler shift 3.0  dynamic False
    timesteps [1000.000 947.543 882.788 800.837 693.793 548.046 337.972 8.929]
    image 128x128  vae_scale 8  ->  latents (16, 16, 16)
    8 steps, DENOISE LOOP COMPLETE

So the sampler arithmetic is the reference's, not a placeholder.  When the
block lands, the remaining work is only: swap `--block` to the real callable,
wire `imaggers/pipeline/vae_decode.py` instead of the stub, and compare the PNG
against `zimage-port/ref_native_128.py` at the same seed.

## The loop skeleton runs today

`imaggers/pipeline/denoise_loop.py` exercises the whole host loop against a
pluggable block callable:

    python denoise_loop.py --block standin --steps 8 --seed 0
    image 128x128  vae_scale 16  ->  latents (16, 8, 8)
    timesteps [1.0 ... 0.125]    flow-matching Euler
    8 steps, DENOISE LOOP COMPLETE

`--block module:callable` swaps in the real NPU block unchanged.  The VAE
decode is a stub on purpose — the scope decision puts the first image's decoder
on the CPU.

So the loop is verified independently of the block assembly and does not gate
on it.  When the block lands, the remaining work is: wire the reference
scheduler's timestep schedule instead of the placeholder linear one, wire the
CPU VAE decode, and compare the PNG against `zimage-port/ref_native_128.py` at
the same seed.

## Where each tensor lives

| tensor | lifetime | form |
|---|---|---|
| weights | the whole run | **`bfp16ebs8` blocks, loaded once** |
| text embeds | the whole run | bf16, from `--embeds` (precomputed) |
| latents | the whole run | bf16 |
| per-block activations | transient | bf16 on device |
| modulations (scale/shift) | per block per step | bf16 |

The point of the exercise is the first row: **one `weights.bfp16`, read, never
re-decoded.**  Step 5 of the assembly plan is where that is measured.

## What to measure at the end

1. **residency** — the weights must be loaded once.  Instrument the loader to
   count BFP16 decode calls; it must be zero after startup.
2. **NPU share** — wall time on the NPU vs wall time anywhere else.  The
   original failure mode was CPU > 95%; the target is the reverse.
3. **image quality** — compare against the CPU reference
   (`zimage-port/ref_native_128.py`) at the same seed and prompt.  Not
   bit-exact — the model is a flow model and bf16 is lossy — but recognisably
   the same picture.

## Where the reference comes from

`zimage-port/ref_native_128.py` runs the full pipeline in PyTorch at 128x128
with a fixed seed and writes a PNG.  That is the oracle for step 3 above.  It
already exists and works; do not rebuild it.

`zimage-port/generate.py` runs the same loop through the VitisAI EP and is the
abandoned path (CPU dominates).  Useful only as a second opinion on the image.

## The order that avoids rework

1. one block, one seed, compare activations against numpy
2. two blocks chained — proves weight residency
3. 34 blocks, 1 step — proves the transformer
4. 34 blocks, 8 steps — proves the loop
5. VAE on CPU — first image
6. then instrument, then optimise, then consider porting the VAE

Do not skip step 2.  Residency is the goal, and it is invisible until two
blocks run back to back.
