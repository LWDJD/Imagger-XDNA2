# Stage 3 — the FFN branch and both residuals

Written while stage 2 (attention) is in flight, so stage 3 is a specification
rather than a discovery exercise when it starts.

## What stage 3 adds

After stage 2 gives

    x = x + attn( adaln_modulate( rms_norm_weighted(x, gamma1), s1, b1 ) )

stage 3 completes the block:

    x = x + ff( adaln_modulate( rms_norm_weighted(x, gamma2), s2, b2 ) )

with

    ff(x) = ( silu(x @ W1) * (x @ W3) ) @ W2        SwiGLU

## Weights (from `imaggers/residency/block_census.py`, `layers.0`)

    feed_forward.w1.weight   [10240, 3840]   44,236,800 B   tile_rows 32
    feed_forward.w3.weight   [10240, 3840]   44,236,800 B   tile_rows 32
    feed_forward.w2.weight   [ 3840,10240]   44,236,800 B   tile_rows 32
    attention_norm2.weight   [   3840]          15,360 B
    ffn_norm2.weight         [   3840]          15,360 B

`w1` and `w3` are SEPARATE tensors, and `pack(vstack(w1,w3), 32) ==
pack(w1) ++ pack(w3)` was MEASURED, so a fused [20480,3840] is just the two
regions concatenated if a single GEMM is wanted.

`adaLN_modulation.0` is `Linear(256, 15360)` and 15360 = 4 x 3840 = (scale1,
shift1, scale2, shift2) — one projection feeds BOTH branches.

## Op sequence

    [6] rms_norm_weighted(x, gamma2)          attention_norm2 / ffn_norm2
    [7] adaln_modulate(x, s2, b2)             the 3rd and 4th quarters of adaLN
    [8] mm_bfp(mixed) 3840 -> 10240  x2       w1 and w3
    [9] swiglu                                 activation.py:505, fixed 1024-elem tiles
    [10] mm_bfp(mixed) 10240 -> 3840          w2
    [11] add_sized residual

## Constraints that already bit earlier stages

* `core_dma_channels_in == 2` per Worker — fold the second branch's
  `[scale|shift|gamma]` into the SAME packed stream element as the first, or
  reuse the stream with a different slice of the adaLN output
* `set_current_device(NPU2())`, NOT `from_name("npu2")`
* `mm_bfp_mixed` re-rounds C to bf16 after EVERY GEMM call
* AIE2P vector loads silently floor misaligned addresses to 64 B
* `swiglu` takes FIXED 1024-element tiles; ffn inner 10240 = 10 x 1024
* `swiglu`'s `vtanh` has 0.08 rel / 0.35 abs error not modelable by the oracle;
  use `swiglu(use_lut=True)` to remove it when judging
* bfp16ebs8 strides must be 4-byte multiples

## Judging

`imaggers/oracle/dit_block_float` is the truth.  Acceptance is

    rel_to_absmax <= 5.4e-02   (the measured float-vs-device gap)

raised by the softmax exp2 error (+2.94e-2) and, unless `use_lut=True`, the
swiglu vtanh error (+0.08).

## What stage 3 must NOT redo

Everything in `imaggers/kernels/` (`rms_norm_weighted`, `adaln_modulate`),
`imaggers/memtile2/` (segmented producer + memtile corner turn) and
`imaggers/attn_impl/` (attention) is verified and should be composed, not
rewritten.
