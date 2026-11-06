# SPDX-License-Identifier: MIT
"""t13_run.py -- T11 + T13 U-4: the full-memory (--cache-all) driver.

T13 adds exactly one mode to t11_run.py, and it is the DEFAULT.  Everything
else is verbatim.

  --cache-all   DEFAULT.  Read all 30 layers' BG (88.5 MB) + B3 (44.2 MB)
                payloads ONCE,
                permute BG into the NH=5 per-core order ONCE, and keep them in
                two flat uint8 arrays (3.98 GB).  The loop takes them by
                reference: the per-step np.load (50-70 ms/layer) and the
                per-step permutation (11-22 ms/layer) both disappear.  The
                arrays are STABLE for the whole run, so the T9 double-buffer
                hazard (an async BO write over a reused scratch buffer) cannot
                occur here by construction -- the double buffer is kept on the
                non-cached path.
                Gate: latents and PNG byte-identical to t11b, PSNR unchanged.
  --no-cache-all
                The T11 path, unchanged: np.load + permutation per layer with
                the prefetch thread.  This is the A arm of every T13 A/B.

--- t11_run.py header follows ---
t11_run.py -- T11: the T10 batch driver on the T11 FFN image with a BATCH dimension.

T10 unit U-1's measurement arm.  Added to t8_run.py (everything else verbatim):

  --batch B   run B independent images through the same 8 x 30 block-step loop.
              Image b uses seed+b (so batch=1 is the delivered run, byte for
              byte) and the images share EXACTLY three things, all of them
              values that do not depend on the image:
                * the layer's BG/B3 payload bytes (132.7 MB/layer, read once)
                * the layer's SBG bytes (one build per layer, not per image)
                * the loop's structure and its 1440 dispatches/layer-set
              Everything numerical stays per image: latents, u, rms, RoPE, the
              FFN A image, the attention drains.  The red line is that each
              image's latents are BYTE-IDENTICAL to its own batch=1 run, which
              t10_cmp.py checks (and that implies its PSNR is unchanged, since
              the VAE decode is a function of the latents alone).

This is the CONTROL arm the task asks for: with the M=96 image a batch of B
still costs B dispatches per layer, so the device side must be exactly linear
and the only gain is the host-side sharing above.  The large-M image (one
dispatch eating M=96*B) is priced separately by t10_mprobe.

--- t8_run.py's own header follows ---
t8_run.py -- T7: the T6 product loop with the attention image SPLIT IN TWO.

T7 U-1: --att-image is now the PROJECTION-ONLY image and --att-ip-image the
IP-ONLY one.  The five attention dispatches keep their identity and their fills
(d1/d2/d3 = to_q/to_k/to_v with X+W, d4 = the ip with QP/KB/VB, d5 = to_out with
X+W), so the HOST side of the loop is the T6 one; what changes is which image
each dispatch runs, and the fact that the projection's redundant second run (d4)
and the ip's four redundant runs (d1,d2,d3,d5) are gone.
Empty --att-ip-image = the T6 fused behaviour, which is the A arm.

--- t5_run.py's own header follows ---
t5_run.py -- T5: the T3d product loop with a T5 attention image.

T5 unit U1: identical to t4_run.py except that the ATTENTION image can be
pointed at another build (--att-image/--att-dir), so the wall-clock delta of a
new attention image is attributable to that image alone (same FFN image, same
payload, same session).  Everything else -- the host glue, the accounting
buckets, the checkpoint protocol, the SBG assertion -- is unchanged.

    py310 t5_run.py --steps 8 --layers 30 --warm --dimk64 --tag t5a \
        --att-image t5_pipe3 --att-dir ../attn_impl/t5_out --prefetch

--- t4_run.py's own header follows ---
T4: the T3d product loop with a T4 FFN image.

Identical to t3d_run.py (the t3d5 delivery driver) except for ONE thing: the
stage-3 FFN image is taken from imaggers/block/t4_out/<--ffn-image>.xclbin
instead of the shipped block/out/step3_M96...xclbin.  Everything else -- the
attention image (t3d_pipe2, dim_k=64), the host glue, the accounting buckets,
the checkpoint protocol -- is unchanged, so a wall-clock delta is attributable
to the FFN image alone.

    py310 t4_run.py --steps 8 --layers 30 --warm --dimk64 --tag t4a --host-ref

The architecture this runs (measured, not assumed -- see _t3d_twoimage_probe.py):
a single py3.10 process registers THREE xclbins and dispatches them alternately:

    proj   t3c_chain_q.xclbin   mode q of the attention chain: one 1.415 GMAC
                                projection per dispatch (30 heads x 60 k-groups),
                                Q bit-exact gate in t3c_chain_q_verify.json
    ip     t3c_fused_o.xclbin   qk -> HOP1 -> sm -> HOP2 -> pv, O bit-identical
                                gate in t3c_fused_o_verify.json
    ffn    step3_M96_K3840_N110240_m16_k8_m64_G4x6L10.xclbin   the stage-3 FFN

Per block-step: 4 projection dispatches (to_q, to_k, to_v, to_out) + 1 ip
dispatch + 1 FFN dispatch = 6.  The HOST between dispatches does exactly what it
did in the f32 chain -- rms(qk_norm) + RoPE + the bf16/bfp16ebs8 packaging -- and
nothing else: every matmul in attention and in the FFN is on the NPU.

The packaging is done with t3d_codec.py (pure numpy), whose every codec is gated
byte-for-byte against the library object it replaces (t3d_codec_gate.py,
ALL_PASS True).  That gate is the precondition for believing any number here.

Wall-clock accounting is kept in four buckets per block-step so the NPU share is
a measured ratio, not a claim:
    t_dev   sum of run.wait() walls (the NPU actually working)
    t_write host-side BO writes (payload staging -- the price of bfp16 weights)
    t_host  host arithmetic (norms, RoPE, pack/unpack)
    t_loop  wall clock of the block-step

Driver 376 / XRT 2.21.0: two (here three) registered xclbins in one process work
and produce byte-identical output to the same image dispatched alone; the old
one-process-one-xclbin rule is version-scoped and this file records the version.

Run (py3.10, GAIA):
    py310 t3d_run.py --gate            # one block-step vs the host f32 block
    py310 t3d_run.py --steps 8 --resume
"""
from __future__ import annotations

import argparse
import concurrent.futures as _fut
import json
import os
import pathlib
import sys
import threading
import time

import numpy as np

sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
NPU = HERE.parent.parent
ATT = NPU / "imaggers" / "attn_impl"
OUT_A = ATT / "t3b_out"
BLOCK = NPU / "imaggers" / "block"
STAGE3 = BLOCK / "out" / "step3_M96_K3840_N110240_m16_k8_n64_G4x6L10"
T4_OUT = BLOCK / "t4_out"
# T4-F: the t4_vs FFN image takes ONE [scale|shift|gamma] per row-tile, which is
# exact only when all 96 SBG rows are equal.  Checked in block_step (module-level,
# hence the global) so the image can never be silently wrong.
ASSERT_SBG = True
NH_BG_DEFAULT = 0
# ---- T6 (this file's three deltas from t5_run.py) -------------------------
#   --fast-codec      t6_codec.install(): pack_b_nk 1.78x, unpack_blocked 1.06x,
#                     byte-identical (t6_codec_gate.py: 40/40 pack cases + all
#                     unpack cases + pack_w64; end-to-end gate = the latents)
#   --refiner-cache D the four refiner blocks' decoded f32 (2.83 GB) persisted;
#                     step 0 pays a 2.83 GB read instead of a 5.5 s decode
#   --wpack-array     hold the 30 layers' projection images (2.0 GB) in ONE
#                     preallocated uint8 array instead of 120 bytes objects
REFCACHE = None
# T7 U-2 knobs (one variable each; both gated by BYTES):
#   --fast-codec2  t7_codec.install(): pack_b_nk 1.48x offline (K), 1.10x (V)
#   --fast-glue    the q/k path stays in the drain's own (HEADS, M, HD) layout,
#                  so the 4 ascontiguousarray transposes around rms+RoPE go away
GLUE_FAST = False
WPACK = None            # None | the one big uint8 array
WPACK_SLOTS = {}        # (key, nm) -> index into WPACK
WPACK_SECONDS = [0.0, 0]
# T13 U-4: full-memory payload cache.  CACHE_BG is ALREADY permuted into the
# per-core NH order when NH_BG > 1, so ffn() must not permute it again.
CACHE_ALL = False
CACHE_BG = None
CACHE_B3 = None
BG_PREPERM = False
# T13 U-4b: 30 BG + 30 B3 BOs, written once.  Implies no host cache (the same
# 3.98 GB lives in BOs instead of RAM).
BO_RESIDENT = False
BG_BOS = None
B3_BOS = None

# ---------------------------------------------------------------------------
# T22 (campaign 2, NPU utilisation): two opt-in, arithmetic-neutral changes.
# Both are OFF by default, so the delivered path is byte-for-byte the T21 one;
# both are gated by the latents (and the T22 report has the same-session A/B).
#
#   T22_FAST_GLUE=1
#     * the three per-block RMS norms and the two residual adds run through
#       rms_into() with reused buffers instead of T3.rms()+temporaries.
#       MEASURED (py310, (96,3840) f32): T3.rms 1.016 ms vs rms_into 0.182 ms,
#       np.array_equal TRUE; the add chain 1.574 vs 0.205 ms.
#     * make_sbg() builds ONE 23,040 B bf16 row and tiles it, instead of
#       zeroing a (96, 11,520) f32 array and encoding 4.4 MB.  The image
#       requires all rows equal (asserted), so this is the same bytes by
#       construction; t22_gate.py checks it over all 240 (step, layer) pairs.
#     * the SBG "all rows equal" assert runs on the first layer of each step
#       (it is a property of make_sbg, not of the layer); T22_SBG_ASSERT=all
#       restores the per-layer check, =off disables it.
#   T22_PFILL=1
#     * the FFN's 132.7 MB payload is written into a SECOND BG/B3 BO pair by a
#       worker thread while layer i's FFN runs.  The device never reads the
#       pair being written (one-layer lookahead, and each FFN dispatch is
#       waited before the next submit).
#     MEASURED NEGATIVE (t22a2 + t22_wthread.py): pyxrt's bo.write holds the
#     GIL for the whole copy, so the worker starves the main thread's glue.
#     The 1.22 s staging bucket simply moved into the between-dispatch gaps
#     (wall 30.23 -> 30.16, +-noise); glue alone 0.043 s -> 12.36 s with a
#     writer thread.  Kept OFF as the recorded A/B arm, not as a win.
T22_FAST_GLUE = os.environ.get("T22_FAST_GLUE", "0") == "1"
# T22 glue (campaign t22_glue): the IN-LOOP GLUE LEDGER.  Every host-side
# segment between two dispatches gets one EXCLUSIVE bucket, and every nesting
# level contributes a "*other" residual, so the sum closes to the loop wall by
# construction.  Off = one None check per call site, no arithmetic.
T22_GLUE_PROF = os.environ.get("T22_GLUE_PROF", "0") == "1"
# T22 glue-2: copy-free drain/pack fast path (memview slices, direct-into
# destination unpack, reused X block).  Same kernels, same element order, same
# destination values; gated by latents byte equality.
T22_GLUE2 = os.environ.get("T22_GLUE2", "0") == "1"
# T22 glue-3: |x|.max() without the 491 KB temporary (3x/layer = 353 MB/image).
# Same value for finite data (max(max, -min)); gated by latents byte equality.
T22_GLUE3 = os.environ.get("T22_GLUE3", "0") == "1"
# T22_PREOVL: the preamble's CONTEXT half is a CONSTANT OF THE CHECKPOINT, so
# read its weights under the startup and compute it once, instead of 8 times
# inside the loop.  [S] cap_embedding(H) is 32 rows of the checkpoint's own
# learned pad token and NEITHER context_refiner block has an adaLN (their
# "adaln_w" is absent from the checkpoint -- modulation() returns None before it
# reads the adaln vector), so (capemb, context_refiner.0, context_refiner.1) is
# a function of H alone: the same bytes on all 8 steps, which is why the 8-step
# loop pays for it 8 times.  [M] 91 ms per step (2 x 45 ms) plus a one-off 1.3 s
# refiner-cache read at step 0 -- 2.0 s of the loop's wall per 8-step image.
# This is SCHEDULING: the same calls in the same order on the same weights, and
# the buffer is read again rather than recomputed, so every downstream byte is
# unchanged (gated: imaggers/preovl/gate_preovl.py).  Default 0 runs the
# delivered statements where they have always run.
T22_PREOVL = os.environ.get("T22_PREOVL", "0") == "1"
# t22_util S2: 32-core FFN GEMM segment (gu main+tail / dn main+tail, swiglu on
# host).  Opt-in only: default 0 keeps the FFN call site byte-for-byte unchanged.
T22_S2_FFN32 = os.environ.get("T22_S2_FFN32", "0") == "1"
T22_S2_FFN32_DIR = os.environ.get("T22_S2_FFN32_DIR", "")
# t22_adaptive (2026-11-03): the PLAN's 32-core image set drives the FFN
# segment (gu gate + gu up -> silu*up on host -> dn), one dispatch set per
# Mblk rows.  The plan comes from webui/npu_plan.py, the images from
# imaggers/util/place/npu_synth.py, and the glue from plan_ffn.py + s2_ffn32.
# Opt-in ONLY: with T22_PLAN_FFN unset nothing below runs, no image is added to
# BIGM_IMGS and the delivered path is byte-for-byte what it was.
#   needs ZIMAGE_BIGM=1 (this is block_step's device-FFN branch) and
#         ZIMAGE_CHUNK_PIPE=0 (the 96-row pipeline calls the delivered ffn image)
PLAN_FFN = os.environ.get("T22_PLAN_FFN", "0") == "1"
T22_PFILL = os.environ.get("T22_PFILL", "0") == "1"
T22_SBG_ASSERT = os.environ.get("T22_SBG_ASSERT", "step")
# t22_breadout: first-call-only dump of the FFN input/output (see block_step)
_DUMP_DONE = False
_DUMP_FIRST = False
T22_PFILL_STATS = {"n": 0, "s": 0.0}
_SBG_ROW = None          # (3*D,) f32, reused
_SBG_OUT = None          # (SEQ*3*D,) uint16, reused

# ---------------------------------------------------------------------------
# T22 qkv (campaign t22_qkv_prod, 2026-10-20): the FUSED q/k/v projection image
# as an OPT-IN of this driver.  Both switches are OFF by default, so the
# delivered path is byte-for-byte the T21 one.
#
#   T22_QKV=1        / --qkv          the three projection launches of a row
#                                     block become ONE dispatch (X, WQ, WK, WV
#                                     -> OUTQKV = [Q | K | V])
#   T22_QKV_XCLBIN   / --qkv-xclbin   the fused image (default: the t22_qkv
#                                     delivery in imaggers/block/qkv/out/)
#   T22_QKV_BIN      / --qkv-bin      its instruction stream
#   T22_XPACK_ONCE=1 / --xpack-once   the delivered loop packs X once per row
#                                     block instead of once per projection
#
# THE CONTRACT OF THE FUSED IMAGE (imaggers/block/qkv/QKV-DESIGN.md): each
# output element depends only on (X, W_p) and the k-group order, not on the
# tile, the worker count or the heads per core, so the three slices are
# byte-identical to the delivered t7_p5 projection on the same inputs (gate:
# 100.0000 % on each of Q/K/V, plus run-to-run).  What that buys is two fewer
# dispatches per layer and the three projections running in parallel.
#
# The W payloads are the SAME Loop.w_images(key) objects the delivered launches
# write, and the X bytes are the SAME H.x_image64_bytes() output, so "same
# inputs" is a construction here, not a hope.  The image inherits the delivered
# projection's M == 96 constraint; SEQ > 96 runs one fused launch per 96-row
# block, exactly as the delivered image runs once per block.
QKV = os.environ.get("T22_QKV", "0") == "1"
QKV_XCLBIN = os.environ.get("T22_QKV_XCLBIN", "")
QKV_BIN = os.environ.get("T22_QKV_BIN", "")
XPACK_ONCE = os.environ.get("T22_XPACK_ONCE", "0") == "1"
QKV_DEFAULT_OUT = BLOCK / "qkv" / "out"
QKV_DEFAULT_TAG = "qkv_par15m"

# ---------------------------------------------------------------------------
# T22 p10 (campaign t22_p15prod, 2026-10-20): the o PROJECTION on a SEPARATE
# multi-stream image.  Opt-in, OFF by default, ORTHOGONAL to T22_QKV.
#
#   T22_P10=1        / --p10           the `att/o` dispatch -- the only
#                                      projection still on the 5-stream
#                                      delivered image once qkv is on -- runs
#                                      on its own image
#   T22_P10_XCLBIN   / --p10-xclbin    the image (default: the t22_proj
#                                      delivery imaggers/attn_impl/proj/out/)
#   T22_P10_BIN      / --p10-bin       its instruction stream
#
# THE CONTRACT (T22-PROJ §3.1, PROJ-REPORT.md): the image is the same
# projection arithmetic -- same per-(head, k-group) mm_bfp calls in the same
# k-ascending order, same C accumulation -- with the delivered 5 worker tile
# streams replaced by G parallel W streams and X multicast.  Each output
# element depends only on (X, W) and that k order, NOT on the tile, the worker
# count or the heads per core, so the 737,280 B output is byte-identical to the
# delivered t7_p5 projection on the same inputs (gate: 100.0000 %, 0 B, plus
# step2d_audit on every core).  What it buys is bandwidth: the delivered
# 5-stream dispatch runs at 3.65 GB/s on 9-12 KB objects, the 10/15-stream
# images sit at the 6.4-6.6 GB/s wire ceiling (PROJ-REPORT §2), so one o
# dispatch goes 5.8 -> 3.8 ms and the o bucket 240 x that per figure.
#
# The payloads are the SAME X2 bytes and the SAME W["o"] object the delivered
# launch writes, so "same inputs" is a construction here, not a hope.  The
# image declares the same three data arguments (X, W -> OUT) in the same order,
# so its BOs are the delivered sizes on ITS OWN group ids (F-08).
#
# NAMING: the task specified "--p15" (campaign id t22_p15prod); the G scan
# (commit b95cc2c) moved the recommendation to 10 streams, so the flag is
# --p10 and the default image is p10.  --p15 is accepted as an alias: the
# stream count is a property of the IMAGE the flag points at (p10 and p15 are
# layout-compatible, both 3 declared args), never of the flag.
P10 = os.environ.get("T22_P10", "0") == "1"
# t22_prodasm: the big-M path (whole-shape projection + swiglu + device norm).
# DEFAULT 0 -- with ZIMAGE_BIGM unset nothing below runs and the shipped path is
# byte-for-byte what it was.  Three more images on the SAME device, registered
# exactly like qkv/attout above (t18_run.py:859-897).
BIGM = os.environ.get("ZIMAGE_BIGM", "0") == "1"
BIGM_DIR = T4_OUT.parent
BIGM_IMGS = {
    "norm": (BIGM_DIR / "normcore" / "out" / "nc_s3a.xclbin",
             BIGM_DIR / "normcore" / "out" / "nc_s3a.bin"),
    "proj": (BIGM_DIR / "bigm" / "out" / "rt17.xclbin",
             BIGM_DIR / "bigm" / "out" / "rt17.bin"),
    "swiglu": (BIGM_DIR / "swiglu" / "out" / "sw64d2.xclbin",
               BIGM_DIR / "swiglu" / "out" / "sw64d2.bin"),
    "down": (BIGM_DIR / "bigm" / "out" / "ffn_dn_n64.xclbin",
             BIGM_DIR / "bigm" / "out" / "ffn_dn_n64.bin"),
    # RT=1 pair: one 96-row tile, i.e. 128^2 (SEQ=96) and any single-tile call.
    # Same payloads as the RT=17 pair (measured: w_bytes identical), so the
    # containers and b3 streams serve both generations.
    "swiglu1": (BIGM_DIR / "swiglu" / "out" / "sw_rt1.xclbin",
                BIGM_DIR / "swiglu" / "out" / "sw_rt1.bin"),
    "down1": (BIGM_DIR / "bigm" / "out" / "ffn_dn_rt1.xclbin",
              BIGM_DIR / "bigm" / "out" / "ffn_dn_rt1.bin"),
    # k=8 generation (PRODASM-STEP345 110): the product instantiates every
    # mm_bfp with dim_k=8; k=64 made each call compute only 1/8 of the k axis.
    "swiglu8": (BIGM_DIR / "bigm" / "out" / "sw8.xclbin",
                BIGM_DIR / "bigm" / "out" / "sw8.bin"),
    "down8": (BIGM_DIR / "bigm" / "out" / "dn8.xclbin",
              BIGM_DIR / "bigm" / "out" / "dn8.bin"),
    # t22_bigpipe3: k=32 down image -- consumes the gu output BO directly
    # (A layout == C layout at k=32), so the swiglu intermediate never
    # touches the host.  Only built when T22_BIGM_DEVHANDOFF=1.
    "dn32": (BIGM_DIR / "swiglu" / "out" / "dn32.xclbin",
             BIGM_DIR / "swiglu" / "out" / "dn32.bin"),
    "dn32_1": (BIGM_DIR / "swiglu" / "out" / "dn32_1.xclbin",
               BIGM_DIR / "swiglu" / "out" / "dn32_1.bin"),
    # dim_k sweep (PRODASM-STEP345 121): find the smallest k the fused loop
    # accepts.  k=8 returns all zeros; 16 and 32 are the intermediate points.
    "sw16": (BIGM_DIR / "bigm" / "out" / "sw16.xclbin",
             BIGM_DIR / "bigm" / "out" / "sw16.bin"),
    "sw32": (BIGM_DIR / "bigm" / "out" / "sw32.xclbin",
             BIGM_DIR / "bigm" / "out" / "sw32.bin"),
}
# t22_hostkill: the DMA-side A-pack/C-un-tile projection image.  OFF by default;
# only registered when the campaign flag is on, so the product image set is
# byte-for-byte unchanged without it.
if os.environ.get("T22_HOSTKILL", "0") == "1":
    BIGM_IMGS["plain"] = (BIGM_DIR / "bigm" / "out" / "plain_rt11.xclbin",
                          BIGM_DIR / "bigm" / "out" / "plain_rt11.bin")

# t22_devkill: G=15 plain projection + RT=11 FFN pair.  Flag-gated; the default
# image set above is byte-for-byte unchanged when T22_DEVKILL is unset.
if os.environ.get("T22_DEVKILL", "0") == "1":
    _DV = BIGM_DIR.parent / "devkill" / "out"
    BIGM_IMGS["plain15"] = (_DV / "plain_g15_rt11.xclbin", _DV / "plain_g15_rt11.bin")
    BIGM_IMGS["swiglu11"] = (_DV / "sw_rt11.xclbin", _DV / "sw_rt11.bin")
    BIGM_IMGS["dn11"] = (_DV / "dn_rt11.xclbin", _DV / "dn_rt11.bin")
# ---------------------------------------------------------------------------
# T22 adaptive (2026-11-03): every M-DEPENDENT host BO is DERIVED, not a 512^2
# literal.  The numbers this replaces were all measured at one size:
#
#     8,110,080 B  = 1056 rows x 3840 x 2   ("11 tiles")  whole-shape x / out
#    21,626,880 B  = 1056 rows x 10240 x 2                whole-shape ff
#    12,533,760 B  = 1632 rows x 3840 x 2   ("17 tiles")  rt17 chunk x / out
#    33,423,360 B  = 1632 rows x 10240 x 2                rt17 chunk ff
#
# Each one is rows x width x 2, so deriving it is exact: with the default row
# counts below the table is BYTE-FOR-BYTE what it was, and configure() widens
# the whole-shape entries only when SEQ is larger than the image they run.
#
# A BO is necessary but NOT sufficient above the image's own row count: the
# image is compiled for RT row tiles, so a shape that needs more rows than the
# image has must run the ADAPTIVE image set (webui/npu_plan.py +
# imaggers/util/place/npu_synth.py), not a wider BO.  Deriving the size is what
# turns "attempting to write past buffer size" into an honest refusal.
#
# T22_ADAPT_M overrides the whole-shape row count; T22_ADAPT_CHUNK the rt17
# chunk.  Neither is set on the delivered path.
# ---------------------------------------------------------------------------
def bigm_bo_bytes(rows, width):
    """rows x width x 2 B -- one host BO of the big-M image set."""
    return int(rows) * int(width) * 2


BIGM_WHOLE_ROWS = int(os.environ.get("T22_ADAPT_M", "1056") or 1056)
BIGM_CHUNK_ROWS = int(os.environ.get("T22_ADAPT_CHUNK", "1632") or 1632)
_BX = bigm_bo_bytes(BIGM_WHOLE_ROWS, 3840)      # 8,110,080 at 1056 rows
_BFF = bigm_bo_bytes(BIGM_WHOLE_ROWS, 10240)    # 21,626,880 at 1056 rows
_CX = bigm_bo_bytes(BIGM_CHUNK_ROWS, 3840)      # 12,533,760 at 1632 rows
_CFF = bigm_bo_bytes(BIGM_CHUNK_ROWS, 10240)    # 33,423,360 at 1632 rows
if "T22_ADAPT_M" not in os.environ:
    assert (_BX, _BFF) == (8_110_080, 21_626_880), (_BX, _BFF)
if "T22_ADAPT_CHUNK" not in os.environ:
    assert (_CX, _CFF) == (12_533_760, 33_423_360), (_CX, _CFF)


# measured BO sizes (PRODASM-STEP345 37): (x, w, out) per image
BIGM_SIZES = {
    # norm / *1 / *8 / sw16 / sw32 / dn32_1 are IMAGE-fixed (96 or 32 rows),
    # so their sizes do not move with SEQ and are left as measured.
    "norm": (737_280, 4_423_680, 737_280),
    "proj": (_CX, 16_588_800, _CX),
    "swiglu": (_CX, 88_473_600, _CFF),
    "down": (_CFF, 44_236_800, _CX),
    "swiglu1": (737_280, 88_473_600, 1_966_080),
    "down1": (1_966_080, 44_236_800, 737_280),
    "swiglu8": (737_280, 88_473_600, 3_932_160),
    "sw16": (737_280, 88_473_600, 3_932_160),
    "sw32": (737_280, 88_473_600, 3_932_160),
    "down8": (1_966_080, 44_236_800, 737_280),
    "dn32": (_CFF, 44_236_800, _CX),
    "dn32_1": (1_966_080, 44_236_800, 737_280),
}
if os.environ.get("T22_HOSTKILL", "0") == "1":
    # (x, w, out) of bigm_proj_plain RT=11; rows derived (see above)
    BIGM_SIZES["plain"] = (_BX, 16_588_800, _BX)

if os.environ.get("T22_DEVKILL", "0") == "1":
    BIGM_SIZES["plain15"] = (_BX, 16_588_800, _BX)
    BIGM_SIZES["swiglu11"] = (_BX, 88_473_600, _BFF)
    BIGM_SIZES["dn11"] = (_BFF, 44_236_800, _BX)


# ---------------------------------------------------------------------------
# T22 proj-block (2026-11-04): THE PROJECTION RUNS IN 96-ROW BLOCKS ABOVE 1056.
#
# WHY (measured, not assumed): every big-M projection image is compiled for a
# FIXED number of 96-row tiles -- plain15 / plain are RT=11 (11 tiles = 1056
# rows), the rt17 pair is 17.  ONE launch therefore computes AT MOST 1056 rows
# whatever the X BO holds.  Measured on this box (probe_plain15_m.py): 2112 rows
# written into b_plain15_x, one launch, and rows 1056..2111 of the drain are
# ZERO and do not move when the X bytes above row 1056 change.  So above 1056
# rows the whole-shape dispatch was not only 60 MB of BO per operand: it
# computed 1056 rows and handed the consumer stale BO memory for the rest.
#
# ROWS ARE INDEPENDENT IN THIS GEMM.  out[r] = x[r] @ W: every output element
# is a dot product over k of ONE X row with ONE W column.  The k-group order,
# the C rounding and the W image are fixed by the image and identical for every
# row; no row reads another row and no block boundary splits a k reduction.  So
# splitting the row axis changes WHICH ROWS ARE IN FLIGHT, never one FLOP's
# order, and the concatenation of the blocks is the whole-shape result bit for
# bit (a block is the same program on a row range).  The one thing a block
# DOES need that the whole shape did not is a zero pad for a partial last
# tile: before, the BO tail supplied 96-row multiples by accident.
# ---------------------------------------------------------------------------
# DEFAULT = 1056 = ONE IMAGE'S FULL TILE COUNT (RT=11), i.e. ONE DISPATCH PER
# LAUNCH IS THE WHOLE IMAGE.  96 (one tile) is kept as the A/B control and for
# memory-critical runs; both are byte-identical (measured at 1024^2: 58.9 s at
# 1056 vs 133.0 s at 96 vs 66.5 s for the delivered chunked arm, same latents
# af993cadfecbc419, +28 MiB of BO for the 1056 table).
PROJ_BLOCK_ROWS = 1056        # default block: every tile the image has
PROJ_BLOCK_MIN = 96           # one tile: the smallest legal block
PROJ_BLOCK_CEILING = 1056     # the delivered ceiling: <= it, ONE dispatch (today)
PROJ_BLOCK_INSTALLED = False  # set by glue.install(); verification only


def img_tiles(name):
    """How many 96-row row tiles the image was COMPILED for (its RT).

    Read from the registered file name -- plain_g15_rt11.xclbin, rt17.xclbin,
    sw_rt1.xclbin -- because the file name is the only authoritative record of
    the number of tiles in the instruction stream.  Returns 0 when the name
    carries no rtN marker (the FFN pair's sw64d2 / ffn_dn_n64, whose chunk is
    derived by fast_bigm_ffn's own endswith("11") rule).
    """
    import re as _re
    try:
        fn = str(BIGM_IMGS[str(name)][0])
    except Exception:                                         # noqa: BLE001
        return 0
    m = _re.search(r"rt(\d+)", fn, _re.I)
    return int(m.group(1)) if m else 0


def proj_block_rows(seq):
    """Rows per projection dispatch, or 0 for "one dispatch for the whole Q".

    ONE definition for the BO (sized here) and the consumer (glue/install.py),
    so the two cannot drift: configure() sizes the BO from this and
    glue.install() refuses to run the delivered dispatcher against it.

    The env test is the same one install.py uses for T22_FASTGLUE, because the
    blocked dispatcher EXISTS ONLY THERE.  With the fastglue attention hook off,
    prod_devkill.proj4 (the delivered 1056-row chunk loop) consumes the BO and
    the old whole-shape rule is correct -- so the answer is 0.
    """
    raw = (os.environ.get("T22_PROJ_BLOCK", "1") or "1").strip().lower()
    if raw in ("", "0", "off", "false", "no"):
        return 0                       # the delivered whole-shape dispatch
    if os.environ.get("T22_FASTGLUE", "0") != "1":
        return 0
    skip = {s.strip() for s in
            os.environ.get("T22_FASTGLUE_SKIP", "").split(",") if s.strip()}
    if "attn" in skip:
        return 0
    if int(seq) <= PROJ_BLOCK_CEILING:
        return 0
    if raw in ("1", "on", "true", "yes"):
        return PROJ_BLOCK_ROWS
    try:
        n = int(raw)
    except ValueError:
        n = PROJ_BLOCK_ROWS
    return max(PROJ_BLOCK_MIN, ((n + 95) // 96) * 96)


def refresh_bigm_sizes(seq):
    """Size every M-dependent big-M host BO for this SEQ and ITS consumer.

    Called from configure() once SEQ is known (BEFORE the glue installs, which
    is why the block decision is an env predicate; glue.install() then asserts
    that the blocked dispatcher is the one that got installed).

    Two regimes, and the delivered shapes are unchanged in both:

      * BLOCKED (fastglue attention on, SEQ > 1056): the projection consumer
        reads exactly ONE 96-row tile, so the projection BOs (plain15, plain,
        proj) are ONE tile -- 737,280 B instead of ceil(SEQ/96)*96 rows
        (63,406,080 B at 1080p).  Nothing else needs the whole shape: the FFN
        images keep their own compiled chunk (11 or 17 tiles), which is what
        _BX / _BFF already are at the default T22_ADAPT_M=1056.
      * UNBLOCKED: exactly the previous rule, rows = max(1056, TILE-PADDED
        SEQ), for the whole-shape dispatcher that reads nt*96 rows.

    Then the guard: every BO a consumer drains must hold at least the rows THAT
    consumer reads, taken from the same constants the consumer uses.  A guard
    that compares a number with itself can never fire, and a guard that assumes
    one dispatch covers the whole shape is exactly what 1080p disproved.
    """
    global BIGM_WHOLE_ROWS, BIGM_CHUNK_ROWS
    seq = int(seq)
    blk = proj_block_rows(seq)
    padded = ((seq + 95) // 96) * 96
    if blk:
        # Every M-dependent BO is pinned to what its SURVIVING consumer reads:
        #   plain/plain15      96 rows   one tile per dispatch (this change)
        #   swiglu11 / dn11    1056 rows the FFN's own compiled chunk (11 tiles)
        #   proj / swiglu / down / dn32  1632 rows  the RT=17 images' 17 tiles
        # T22_ADAPT_M / T22_ADAPT_CHUNK exist ONLY to widen the BO for the
        # whole-shape dispatch this change removes, so they are not honoured
        # here -- honouring them would re-add the memory that was the point.
        before = sum(sum(v) for v in BIGM_SIZES.values())
        px = bigm_bo_bytes(blk, 3840)
        fx = bigm_bo_bytes(11 * 96, 3840)
        ff = bigm_bo_bytes(11 * 96, 10240)
        cx = bigm_bo_bytes(17 * 96, 3840)
        cff = bigm_bo_bytes(17 * 96, 10240)
        BIGM_SIZES.update({
            "plain": (px, 16_588_800, px),
            "plain15": (px, 16_588_800, px),
            "proj": (cx, 16_588_800, cx),
            "swiglu": (cx, 88_473_600, cff),
            "down": (cff, 44_236_800, cx),
            "dn32": (cff, 44_236_800, cx),
            "swiglu11": (fx, 88_473_600, ff),
            "dn11": (ff, 44_236_800, fx),
        })
        BIGM_WHOLE_ROWS = 11 * 96
        BIGM_CHUNK_ROWS = 17 * 96
        rows = BIGM_WHOLE_ROWS
        after = sum(sum(v) for v in BIGM_SIZES.values())
        print("[t18] big-M projection blocked: %d-row BOs (%d B), %d block(s) "
              "for SEQ=%d; FFN BOs 1056 rows, rt17 BOs 1632 rows; BO table "
              "%.3f -> %.3f GiB (-%.3f GiB); T22_ADAPT_M/CHUNK ignored while "
              "blocked"
              % (blk, px, (seq + blk - 1) // blk, seq, before / 2**30,
                 after / 2**30, (before - after) / 2**30), flush=True)
    else:
        rows = max(1056, padded)
        if rows != BIGM_WHOLE_ROWS:
            BIGM_WHOLE_ROWS = rows
            x, ff = bigm_bo_bytes(rows, 3840), bigm_bo_bytes(rows, 10240)
            if "plain" in BIGM_SIZES:
                BIGM_SIZES["plain"] = (x, 16_588_800, x)
            if "plain15" in BIGM_SIZES:
                BIGM_SIZES["plain15"] = (x, 16_588_800, x)
                BIGM_SIZES["swiglu11"] = (x, 88_473_600, ff)
                BIGM_SIZES["dn11"] = (ff, 44_236_800, x)
            print("[t18] big-M BOs derived for %d rows: x/out %d B, ff %d B"
                  % (rows, x, ff), flush=True)
    # The consumer's row count, per image.  install.py:418 (_proj_join) drains
    # exactly the rows the dispatcher issued; the FFN hands its image one
    # compiled chunk (11 tiles for the RT=11 pair, 17 for the RT=17 pair).
    _proj_need = blk or padded
    _need_rows = {"plain": _proj_need,      # devkill / hostkill projection
                  "plain15": _proj_need,
                  # the rt17 images are compiled for 17 tiles and are consumed
                  # by prod_loop_bigm's own loops (17-tile chunks); their BO is
                  # the 1632-row T22_ADAPT_CHUNK default, NOT the whole shape.
                  "proj": 17 * 96, "swiglu": 17 * 96, "down": 17 * 96,
                  "dn32": 17 * 96,
                  "swiglu11": 11 * 96, "dn11": 11 * 96}
    for _n, _nr in _need_rows.items():
        if _n in BIGM_SIZES:
            _x, _w, _o = BIGM_SIZES[_n]
            need = bigm_bo_bytes(_nr, 3840)
            assert min(_x, _o) >= need, (
                "big-M BO %r holds %d/%d B (x/o) but its consumer reads %d "
                "rows = %d B (SEQ=%d, blocked projection=%s, block=%d).  "
                "Re-introduce a smaller BO and step 0 dies with 'attempting "
                "to read past buffer size'; it must fail HERE instead."
                % (_n, _x, _o, _nr, need, seq, bool(blk), blk))
    return rows
P10_XCLBIN = os.environ.get("T22_P10_XCLBIN", "")
P10_BIN = os.environ.get("T22_P10_BIN", "")
P10_DEFAULT_OUT = ATT / "proj" / "out"
P10_DEFAULT_TAG = "p10"


def _mem_gb():
    """Available physical GB (ctypes; no psutil dependency in py310)."""
    import ctypes
    from ctypes import wintypes

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    m = MS()
    m.dwLength = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2**30
PREP = HERE / "t3real_prep"
# Third-party locations: resolved, never hard-coded (imaggers/paths.py).
_root = pathlib.Path(__file__).resolve().parent
while not (_root / 'imaggers' / 'paths.py').is_file():
    if _root.parent == _root:
        raise RuntimeError('cannot locate the checkout root')
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from imaggers import paths as _paths  # noqa: E402

XRT_DIR = _paths.xrt_dir()

sys.path.insert(0, str(HERE))
sys.path.append(str(ATT))

import t3d_codec as K                                             # noqa: E402
import t8_codec as H                                              # noqa: E402
import t9_permute as _P9                                          # noqa: E402  (T9 U-4b)
import t11_permute as _P11                                        # noqa: E402  (T11 U-1)

# T9 U-4b: the n_hold FFN image consumes BG as [pass][kk][j]; the payload on
# disk is the delivered [sweep][kk].  0 = off (the delivered path, byte-for-byte
# what t8b runs); 2 = the n_hold=2 image (t9_nh2b).  The permutation is host
# payload plumbing -- it changes no arithmetic, and it is charged to the pack
# bucket so the ledger stays honest.
NH_BG = 0
# T11 U-1: with --nh-bg N, --nh-bg-core writes the PER-CORE contiguous order
# [w][pass][kk][j] that the per-core-fill NH image reads (each gate/up core owns
# its own shim fill, so its bytes must be one contiguous slice of the BO).
NH_BG_CORE = False
_BG_SCRATCH = None
_BG_FLIP = 0

D, HEADS, HD, FF, SEQ, CAP = 3840, 30, 128, 10240, 96, 32
M = SEQ
KD, ND = HD, M
HEADS_ = HEADS
ACT_BYTES = SEQ * D * 2                      # 737,280
SBG_BYTES = SEQ * 3 * D * 2
BG_BYTES = 2 * FF * D * 9 // 8               # 88,473,600
B3_BYTES = D * FF * 9 // 8                   # 44,236,800
KB_BYTES = KD * ND // 8 * 9                  # 13,824
W_BYTES = HEADS * (D * HD // 8 * 9)          # one projection's bfp image
assert W_BYTES == 16_588_800, W_BYTES

# ---------------------------------------------------------------------------
# T18: arbitrary sequence length (SEQ = 32 caption + H/16 * W/16 image tokens)
# ---------------------------------------------------------------------------
# What scales with SEQ, and what does not:
#   * FFN: ONE image per M (t18_m128 / t18_m288 -- the t11_ffn_nh5 SOURCE,
#     verbatim; --M only changes row_tiles, i.e. how many times the runtime
#     sequence replays the same per-row-tile code, so no arithmetic moves).
#     A / SBG / X / Y BOs scale with SEQ.
#   * projection (t7_p5, m=96): run once per 96-row block.  A projection row
#     is independent of every other row (same k-group order, same C rounding),
#     so concatenating blocks is exact -- no image change, byte-identical.
#   * ip (t7_ip): attention is DENSE, so the image must see ALL SEQ keys; its
#     L1 budget QP + KB + C + stack picks the query-block height:
#         SEQ= 96 -> M=96, ND= 96  (delivered t7_ip;  62,976 B)
#         SEQ=128 -> M=64, ND=128  (t18_ip_m64nd128;  57,344 B)
#         SEQ=288 -> M=16, ND=288  (t18_ip_m16nd288;  60,928 B)
#   * blocked-layout perms: a function of (rows, K), extracted by t18_perm.py.
PROJ_M = 96
IP_MQ = 96
IP_ND = 96
GH, GW, TOKENS = 8, 8, 64
PROJ_X_BYTES = D * PROJ_M * 2                 # 737,280
PROJ_OUT_BYTES = HEADS * PROJ_M * HD * 2      # 737,280
IP_QP_BYTES = HEADS * IP_MQ * HD * 2
IP_OUT_BYTES = HEADS * IP_MQ * HD * 2
FFN_M = 96
FFN_BLOCKS = 1
FFN_A_BYTES = ACT_BYTES
FFN_SBG_BYTES = SBG_BYTES
# T20 U-1: see Dev.barrier().  On by default; T20_FFN_BARRIER=0 is the A arm.
FFN_BARRIER = os.environ.get("T20_FFN_BARRIER", "1") != "0"
PERM_Q = {96: "t3d_perm_96x128.json", 64: "t18_perm_64x128.json",
          32: "t18_perm_32x128.json", 16: "t18_perm_16x128.json"}
# T18 padding branch (zimage_pipeline / diffusers patchify_and_embed):
#     image_padding_len = (-image_tokens) % 32
#     padded rows: x_pad_token (after all_x_embedder), pos id (0,0,0)
#     caption rows come AFTER the padded image block; M = 32 + padded
# The device sees SEQ = 32 + padded rows; unpatchify keeps only the real tokens.
REAL_TOKENS = 64
PAD_ROWS = 0
_T18_CFG = {"done": False, "gh": 8, "gw": 8, "orig_real_ids": None,
            "orig_x_embed": None}


CHUNKED = False


def configure(height, width, ip_mq=0, ffn_m=0):
    """Set every SEQ-dependent global.  Call before Dev() and Loop().

    T22 chunk_path: a SEQ above the delivered ip ceiling (304) is not an
    error any more.  The delivered M=96 images are row-independent for the
    projection / FFN / o projection, so configure() selects them in 96-row
    blocks and marks CHUNKED; t18_run.main() then installs the host attention
    (webui/chunked.py).  Every host constant below -- IMG_GRID (RoPE ids),
    x_pad_token rows, SBG rows, patch/unpatchify grid -- is generated for the
    requested geometry exactly as for the delivered shapes.
    """
    global SEQ, M, ND, ACT_BYTES, SBG_BYTES, KB_BYTES, GH, GW, TOKENS
    global IP_MQ, IP_ND, IP_QP_BYTES, IP_OUT_BYTES
    global FFN_M, FFN_BLOCKS, FFN_A_BYTES, FFN_SBG_BYTES
    global REAL_TOKENS, PAD_ROWS, CHUNKED
    h, w = int(height), int(width)
    gh, gw = h // 16, w // 16
    tokens = gh * gw
    padded = ((tokens + 31) // 32) * 32
    seq = CAP + padded
    CHUNKED = False
    if seq not in (96, 128, 160, 288) and not ip_mq and not ffn_m:
        # ---- chunk_path: M=96 kernels in 96-row blocks, host attention ----
        mq, nd = 96, 96
        kb = KD * nd // 8 * 9
        l1 = mq * KD * 2 + kb + mq * nd * 2 + 0x1800
        if l1 > 65536:
            raise SystemExit("[t18] chunked dummy ip L1 %d > 65536" % l1)
        SEQ, M, ND = seq, seq, nd
        ACT_BYTES = seq * D * 2
        SBG_BYTES = seq * 3 * D * 2
        KB_BYTES = kb
        GH, GW, TOKENS = gh, gw, tokens
        REAL_TOKENS, PAD_ROWS = tokens, padded - tokens
        IP_MQ, IP_ND = mq, nd
        IP_QP_BYTES = HEADS * mq * HD * 2
        IP_OUT_BYTES = HEADS * mq * HD * 2
        FFN_M = 96
        FFN_BLOCKS = (seq + FFN_M - 1) // FFN_M
        FFN_A_BYTES = FFN_M * D * 2
        FFN_SBG_BYTES = FFN_M * 3 * D * 2
        CHUNKED = True
        print("[t18] CHUNKED SEQ=%d (cap %d + image %d real + %d pad, grid %dx%d)"
              "  proj m=96 x %d block(s)  host attention  ffn M=96 x %d block(s)"
              % (SEQ, CAP, tokens, PAD_ROWS, gh, gw,
                 (seq + 95) // 96, FFN_BLOCKS), flush=True)
    else:
        if ip_mq:
            mq = int(ip_mq)
        else:
            mq = {96: 96, 128: 64, 160: 32, 288: 16}.get(seq, 0)
        if mq not in PERM_Q:
            raise SystemExit("[t18] no ip query-block image / perm for SEQ=%d" % seq)
        kb = KD * seq // 8 * 9
        l1 = mq * KD * 2 + kb + mq * seq * 2 + 0x1800
        if l1 > 65536:
            raise SystemExit("[t18] ip L1 %d > 65536 at M=%d, ND=%d" % (l1, mq, seq))
        if seq % mq:
            raise SystemExit("[t18] SEQ=%d is not a multiple of the ip block M=%d"
                             % (seq, mq))
        SEQ, M, ND = seq, seq, seq
        ACT_BYTES = SEQ * D * 2
        SBG_BYTES = SEQ * 3 * D * 2
        KB_BYTES = kb
        GH, GW, TOKENS = gh, gw, tokens
        REAL_TOKENS, PAD_ROWS = tokens, padded - tokens
        IP_MQ, IP_ND = mq, seq
        IP_QP_BYTES = HEADS * mq * HD * 2
        IP_OUT_BYTES = HEADS * mq * HD * 2
        # T18: the FFN image is row-independent, but its RUNTIME sequence grows
        # with M.  The M=288 image (18 row tiles, 36,144 B of instructions) does
        # not complete on the device (ERT_CMD_STATE_TIMEOUT, reproduced
        # standalone), while M=128 (8 tiles, 15,264 B) does.  So the FFN is
        # dispatched in blocks of FFN_M rows with an image compiled for exactly
        # that M:
        #     SEQ= 96 -> FFN_M= 96, delivered t11_nh5, 1 block
        #     SEQ=128 -> FFN_M=128, t18_m128,          1 block
        #     SEQ=160 -> FFN_M= 96, delivered t11_nh5, 2 blocks (T20, 128x256)
        #     SEQ=288 -> FFN_M= 96, delivered t11_nh5, 3 blocks (exact)
        # Every multi-block shape inserts the T20 relaunch barrier (Dev.barrier).
        fm = int(ffn_m) if ffn_m else {96: 96, 128: 128, 160: 96,
                                         288: 96}.get(seq, 0)
        if not fm or fm % 16:
            raise SystemExit("[t18] no FFN block size for SEQ=%d" % seq)
        FFN_M = fm
        FFN_BLOCKS = (SEQ + fm - 1) // fm
        FFN_A_BYTES = fm * D * 2
        FFN_SBG_BYTES = fm * 3 * D * 2
        print("[t18] SEQ=%d (cap %d + image %d real + %d pad, grid %dx%d)"
              "  proj m=%d x %d block(s)  ip M=%d ND=%d x %d block(s)"
              "  ffn M=%d x %d block(s)  A/SBG/Y %d/%d/%d B"
              % (SEQ, CAP, tokens, PAD_ROWS, gh, gw, PROJ_M,
                 (padded + PROJ_M - 1) // PROJ_M, IP_MQ, IP_ND,
                 (SEQ + IP_MQ - 1) // IP_MQ, FFN_M, FFN_BLOCKS,
                 ACT_BYTES, SBG_BYTES, ACT_BYTES), flush=True)
    # T22 adaptive: the whole-shape big-M host BOs are derived from this SEQ
    # (a no-op at the delivered 96/128/160/288 sizes).
    refresh_bigm_sizes(SEQ)
    if PLAN_FFN:
        # The plan's images + BO sizes must be in the registry BEFORE Dev()
        # (called from run(), i.e. after this), because Dev.__init__ is what
        # registers the xclbin and allocates b_<tag>_{x,w,o}.
        import sys as _sys_pf
        import pathlib as _pl_pf
        _pf_dir = _pl_pf.Path(__file__).resolve().parents[1] / "util" / "place"
        if str(_pf_dir) not in _sys_pf.path:
            _sys_pf.path.insert(0, str(_pf_dir))
        import plan_ffn as _PFF
        _PFF.register(SEQ)
    import t3_real as _T3
    _T3.IMG_GRID = (1, gh, gw)
    _T18_CFG["gh"], _T18_CFG["gw"] = gh, gw
    if not _T18_CFG["done"]:
        _orig_unpatch = _T3.unpatchify
        _T18_CFG["orig_real_ids"] = _T3.real_ids
        _T18_CFG["orig_x_embed"] = _T3.x_embedding

        def unpatchify_shape(tok, chans, _gh=None, _gw=None):
            return _orig_unpatch(tok, chans, _T18_CFG["gh"], _T18_CFG["gw"])

        def real_ids_padded():
            # zimage_pipeline: the padded image rows carry pos id (0,0,0) and
            # sit at the END of the image block, before the caption rows.
            cap, img = _T18_CFG["orig_real_ids"]()
            if PAD_ROWS:
                img = np.concatenate(
                    [img, np.zeros((PAD_ROWS, 3), np.int64)], axis=0)
            return cap, img

        def x_embedding_padded(lat, H):
            # all_x_embedder -> then _apply_pad_token: the padded rows are the
            # learned x_pad_token, NOT the (meaningless) patch embedding.
            xb = _T18_CFG["orig_x_embed"](lat, H)
            if PAD_ROWS:
                pad = np.repeat(np.asarray(H["x_pad_token"], np.float32
                                           ).reshape(1, -1), PAD_ROWS, axis=0)
                xb = np.concatenate([xb, pad], axis=0)
            return xb

        _T3.unpatchify = unpatchify_shape
        _T3.real_ids = real_ids_padded
        _T3.x_embedding = x_embedding_padded
        _T18_CFG["done"] = True
    return SEQ

COMPLETED = 4
TIMEOUT_MS = int(os.environ.get("T3_TIMEOUT_MS", "300000"))
# T20 U-1: the 10 s first backoff was masking a *measured* recovery cost.  A
# command that trips the ERT watchdog (state 8) leaves the array needing one
# reset; an immediate relaunch pays it (~1.0-1.5 s, t20_bench.py) and completes.
# So the first backoff only has to clear the host side: 2 s, not 10 s.
RETRY_SLEEPS = (2, 5, 10, 20, 30)
PROJ_ARGS = ("x", "w", "kb", "vb")
FFN_ORDER = ("A", "SBG", "BG", "B3", "X", "Y")
FFN_GROUPS = {"A": 3, "SBG": 4, "BG": 5, "B3": 6, "X": 7, "Y": 8}


def _trace(msg):
    """T18: stage breadcrumbs for hard-crash bisection (env T18_TRACE=1)."""
    if os.environ.get("T18_TRACE"):
        print("[trace] " + msg, flush=True)


def act(x):
    return np.asarray(x, np.float32)


def bf16f32(x):
    """bf16(x) back in float32 -- the grid the device's operands live on."""
    return (K.bf16_rne(x).astype(np.uint32) << 16).view(np.float32)


def rope_rms_hmd(x, w, cos, sin, T3):
    """rms + RoPE in the (HEADS, M, HD) layout the drain already produces.

    t3_real.rms reduces over the LAST axis and t3_real.apply_rope rotates the
    0::2 / 1::2 pairs of the last axis with cos/sin indexed by the SEQUENCE
    POSITION (axis -2 of (M, HEADS, HD)).  In the drained layout the sequence is
    axis 1, so the very same element-wise expressions apply with cos[None] --
    same operands, same order, same rounding.  The only thing that changes is
    that no (M, HEADS, HD) copy is ever materialised: 4 x 1.47 MB per layer.
    """
    y = T3.rms(x, w)                       # (HEADS, M, HD), reduction over HD
    c, s = cos[None, :, :], sin[None, :, :]
    xe, xo = y[..., 0::2], y[..., 1::2]
    out = np.empty_like(y)
    out[..., 0::2] = xe * c - xo * s
    out[..., 1::2] = xe * s + xo * c
    return out


# ---------------------------------------------------------------------------
# weight packing: to_q/to_k/to_v/to_out -> the proj image's W payload
# ---------------------------------------------------------------------------
FAST_HOST = False
SCRATCH = None


PROF = None


def pmark(name, t0):
    if PROF is not None:
        PROF[name] = PROF.get(name, 0.0) + 1e3 * (time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# T22 glue: the in-loop ledger (T22_GLUE_PROF / --prof)
# ---------------------------------------------------------------------------
GLUE = None          # None | {name: [seconds, calls]}


def gadd(name, dt):
    if GLUE is None:
        return
    v = GLUE.get(name)
    if v is None:
        GLUE[name] = [float(dt), 1]
    else:
        v[0] += float(dt)
        v[1] += 1


def gmark(name, t0):
    if GLUE is not None:
        gadd(name, time.perf_counter() - t0)


def gtotal():
    return sum(v[0] for v in GLUE.values()) if GLUE is not None else 0.0


def absmax_fast(x):
    """max|x| without np.abs(x)'s full-size temporary.

    For finite data max(max(x), -min(x)) is exactly np.abs(x).max() (the same
    f32 value, widened to float64 for the print).  Saves one 491 KB allocation
    per call -- block_step calls it three times per layer (353 MB/image).
    """
    hi = float(x.max())
    lo = float(x.min())
    return hi if hi >= -lo else -lo


GLUE_OUTSIDE = ("step.lat", "step.ckpt")   # written between two step walls


def glue_inside():
    """Sum of the buckets that live INSIDE one_step's wall."""
    return sum(v[0] for k, v in GLUE.items() if k not in GLUE_OUTSIDE)


def glue_print(total, nlayer):
    """T22 glue: print the in-loop table.  Every row is EXCLUSIVE; the
    residual line is step-wall - sum and must be ~0 (it is what the ledger
    missed).  step.lat/step.ckpt are charged between two steps, so they are
    listed but excluded from the closure."""
    if GLUE is None:
        return
    s = glue_inside()
    print("[t22g] in-loop ledger: step-walls %.3f s  sum %.3f s  residual %+.3f s"
          % (total, s, total - s), flush=True)
    print("[t22g] %-18s %9s %9s %7s" % ("bucket", "s", "ms/layer", "calls"),
          flush=True)
    for k, v in sorted(GLUE.items(), key=lambda kv: -kv[1][0]):
        if v[0] < 5e-4:
            continue
        print("[t22g] %-18s %9.3f %9.3f %7d"
              % (k, v[0], 1e3 * v[0] / nlayer if nlayer else 0.0, v[1]),
              flush=True)


def _u16_view(raw):
    """bytes | memoryview -> uint16 view, WITHOUT the bytes() copy."""
    if isinstance(raw, memoryview):
        return np.frombuffer(raw, dtype=np.uint16)
    return np.frombuffer(bytes(raw), np.uint16)


def unpack_bytes_mv(raw, inv, shape, scratch=None, key="unpack"):
    """t3d_codec.unpack_bytes, but a memoryview drain costs no copy."""
    u = _u16_view(raw)
    rows = shape[0]
    cols = u.size // rows
    if scratch is None:
        out = np.empty((rows, cols), np.float32)
    else:
        out = scratch.get(key, (rows, cols), np.float32)
    H.unpack_blocked(u.reshape(-1), inv, out)
    return out.reshape(shape)


def unpack_bytes_t_into(raw, inv, hd, out):
    """t3d_codec.unpack_bytes_t, straight into the caller's (M, H*HD) buffer.

    Same kernel, same inv, same element order: only the destination address
    changes (the scratch copy in between disappears).
    """
    H.unpack_blocked_t(_u16_view(raw).reshape(-1), inv, out, hd)
    return out


def rms_into(x, w, out):
    """T3.rms into a caller-supplied buffer: same ops, same order, no new page.

    t3_real.rms is  y = (x / sqrt(mean(x*x) + eps)).astype(f32); * w, all in
    float32.  Writing through np.divide/np.multiply with out= computes exactly
    the same float32 expressions (the .astype is a no-op on float32 input); what
    disappears is two 1.47 MB allocations per call.
    """
    import t3_real as T3
    np.multiply(x, x, out=out)
    v = out.mean(-1, keepdims=True, dtype=np.float32)
    np.add(v, np.float32(T3.EPS), out=v)
    np.sqrt(v, out=v)
    np.divide(x, v, out=out)
    np.multiply(out, np.asarray(w, np.float32), out=out)
    return out


def rope_rms_jit(x, w, cos, sin, out, tmp):
    """rms -> rope, both into reusable buffers (no copy between them)."""
    rms_into(x, w, tmp)
    H.rope_f32(tmp, cos, sin, out)
    return out


def _xblock(L, src, r0, rows):
    """T22 glue-2: the 96-row projection block without a fresh np.zeros.

    The image reads pm[p] over ALL PROJ_M rows, so a short block's padding rows
    must be zero.  With T22_GLUE2 the buffer is allocated once per process; the
    tail is re-zeroed only when the block is short (SEQ % 96 == 0 -- the
    delivered 96 and 288 -- has no tail at all).  Bytes handed to the kernel are
    identical either way: same copy, same zeros.
    """
    if not T22_GLUE2:
        xb = np.zeros((PROJ_M, D), np.float32)
        xb[:rows] = src[r0:r0 + rows]
        return xb
    xb = L.xblk
    if rows < PROJ_M:
        xb[rows:].fill(0.0)
    np.copyto(xb[:rows], src[r0:r0 + rows])
    return xb


def pack_w(Wh: np.ndarray) -> np.ndarray:
    """W (out=D, in=D) -> the 16,588,800 B image the proj core reads.

    Block order = t3c_chain_run.pack_w (the packer behind the gated mode-q
    image), vectorized: the caller's 14,400 python-level bfp.encode calls of one
    (16,8,8) tile each become one encode of a (8*16, 8, 8) stack per (head,
    k-group).  Byte-equality with the slow packer is proven by t3d_wgate.py; the
    slow one must not be trusted either, so t3d_wgate also re-derives the whole
    projection against the k-slice reference.
    """
    K_ = Wh.shape[1]
    NG = K_ // 64
    out = np.empty((HEADS, NG, 8 * 16 * 72), np.uint8)
    for h in range(HEADS):
        Wn = np.ascontiguousarray(Wh[h * HD:(h + 1) * HD, :], np.float32)
        for g in range(NG):
            # (nb=16, n8=8, s=8, k8=8) -> (s, nb, n8, k8): the packer's order is
            # [s][nb] with each tile an (8 n, 8 k) block of 8 groups of 9 B.
            sl = Wn[:, g * 64:(g + 1) * 64].reshape(16, 8, 8, 8)
            blk = np.ascontiguousarray(sl.transpose(2, 0, 1, 3)).reshape(128, 8, 8)
            out[h, g] = K.bfp_encode(blk).reshape(-1)
    return out


def pack_w64(Wh: np.ndarray) -> np.ndarray:
    """W (out=D, in=D) -> the payload of the dim_k=64 image: per (head, k-group)
    the (n=128, k=64) B image, in pack_b_nk's [nb][kb] order.  Same 16,588,800 B
    total as the k=8 packer, different internal grouping."""
    out = np.empty((HEADS, D // 64, 128 * 64 // 8 * 9), np.uint8)
    for h in range(HEADS):
        for g in range(D // 64):
            blk = np.ascontiguousarray(Wh[h * HD:(h + 1) * HD, g * 64:(g + 1) * 64])
            assert blk.shape == (HD, 64), blk.shape
            out[h, g] = K.pack_b_nk(blk[None])[0]
    return out


def x_image64(x_mk: np.ndarray, perm64: np.ndarray) -> bytes:
    """(96, K) float32 -> the dim_k=64 X object: 60 groups of the blocked (96,64)."""
    m, Kd = x_mk.shape
    parts = [K.pack_blocked(np.ascontiguousarray(x_mk[:, g * 64:(g + 1) * 64])
                            .reshape(1, -1), perm64) for g in range(Kd // 64)]
    return b"".join(parts)


def make_sbg(P, mod):
    """The FFN image's [scale|shift|gamma] stream, bf16, one row per token.

    Verbatim the block that used to live in block_step: it depends only on the
    (step, layer) modulation and on ffn_norm1 (a weight), NEVER on the image --
    which is why T10 builds it once per layer and hands the same bytes to every
    image of the batch.  batch=1 gets the identical bytes from the identical
    expressions, so nothing about the delivered run changes.
    """
    if T22_FAST_GLUE and FAST_HOST:
        # T22: one row, encoded once, tiled.  bf16 RNE is elementwise, and every
        # row of the old (96, 3D) array held exactly these 3D f32 values, so the
        # bytes are identical by construction (t22_gate.py checks 240 pairs).
        global _SBG_ROW, _SBG_OUT
        if _SBG_ROW is None:
            _SBG_ROW = np.empty(3 * D, np.float32)
        row = _SBG_ROW
        row[:D] = (mod[2] - np.float32(1.0)) if mod is not None else 0.0
        row[D:2 * D] = 0.0
        row[2 * D:] = P["ffn_norm1"]
        enc = H.bf16_bytes_f32(row.reshape(1, -1), SCRATCH, "sbgrow")
        n = SEQ * 3 * D
        if _SBG_OUT is None or _SBG_OUT.size != n:
            _SBG_OUT = np.empty(n, np.uint16)
        _SBG_OUT.reshape(SEQ, 3 * D)[:] = enc
        return _SBG_OUT
    SBG = np.zeros((SEQ, 3 * D), np.float32)
    if mod is not None:
        SBG[:, :D] = (mod[2] - np.float32(1.0))
    SBG[:, D:2 * D] = 0.0
    SBG[:, 2 * D:] = P["ffn_norm1"]
    if FAST_HOST:
        return H.bf16_bytes_f32(SBG, SCRATCH, "sbg")
    return np.frombuffer(K.bf16_bytes(SBG), np.uint8)


# T10 U-1b: with a batch, layer L's BG/B3 bytes are already staged in the BO
# after image 0's dispatch; rewriting them for images 1..B-1 moves the same
# bytes to the same place.  --share-weights skips those writes (the red line is
# the latents: byte-identical to the batch=1 run).
SHARE_W = False


# ---------------------------------------------------------------------------
_BIGM_DEV = {"d": None}


# T22_CHAIN2B (2026-11-04): the chain THIS MODULE INSTANCE drives.  0 is the
# delivered single-chain value; the two-chain driver loads a second instance of
# this file as its own module (own SEQ/D/tims/PROF/Loop/block_step) and sets
# CHAIN_ID=1 on it.  Nothing reads it unless T22_CHAIN2B=1.
CHAIN_ID = 0
_CHAIN2B_CHILD = False


class Dev:
    """The three images, their BOs, and the retry/measure protocol."""

    def __init__(self, tag="t3d", dimk64=False, ffn_image="t4_sbg",
                 att_image="", att_dir=None, att_ip_image="", share=None):
        """T22_CHAIN2B: share=<another Dev> builds a SECOND chain in the same
        process on the FIRST chain's registered xclbins/kernels and gives it a
        complete BO table of its own.

        WHY NOT its own registration: measured (chain2b/probe_two_dev_product.py,
        probe_ctx.py) every pyxrt.kernel() call is a NEW HARDWARE CONTEXT with
        its own group-id space, the product needs EIGHTEEN of them per chain,
        and the twenty-seventh in one process dies with
          RuntimeError: Failed to create context virtual (0xc01e0009).
        Two chains x 18 contexts is therefore NOT available on this driver.  The
        contexts are read-only (xclbin + instruction BO, both synced in and never
        written again), so sharing them is a sharing of CONSTANTS; everything a
        dispatch writes -- the X, W and out BOs -- is built fresh below, per Dev.
        Measured with a twin BO table on shared kernels: two host threads, 48
        dispatches, every drain byte-identical to the solo arm
        (chain2b/logs/probe_shared_ctx.log).
        """
        for d in (XRT_DIR, pathlib.Path(r"C:\Windows\System32\AMD")):
            if d.exists():
                sys.path.insert(0, str(d))
                try:
                    os.add_dll_directory(str(d))
                except Exception:  # noqa: BLE001
                    pass
        global pyxrt
        import pyxrt
        self.pyxrt = pyxrt
        # A shared chain still opens nothing: the donor's handle owns the
        # contexts and BOs are allocated from it.
        self.dev = share.dev if share is not None else pyxrt.device(0)
        self.share = share
        self.tag = tag
        self.imgs = {}
        att_x = (att_image, att_image) if att_image else (
            ("t3d_pipe2", "t3d_pipe2") if dimk64 else ("t3d_pipe", "t3d_pipe"))
        att_base = pathlib.Path(att_dir) if att_dir else OUT_A
        if att_dir and not att_base.is_absolute():
            att_base = (HERE / att_base).resolve()
        # T7: the split pair.  "att" is the projection-only image when an ip
        # image is named, and the T6 fused image otherwise (the A arm).
        self.has_ip = bool(att_ip_image)
        regs = [("att", att_base / (att_x[0] + ".xclbin"),
                 att_base / (att_x[1] + ".bin"))]
        if self.has_ip:
            regs.append(("attip", att_base / (att_ip_image + ".xclbin"),
                         att_base / (att_ip_image + ".bin")))
        regs.append(("ffn", T4_OUT / (ffn_image + ".xclbin"),
                     T4_OUT / (ffn_image + ".bin")))
        for name, xcl, inst in (() if share is not None else regs):
            t0 = time.perf_counter()
            xc = pyxrt.xclbin(str(xcl))
            uuid = self.dev.register_xclbin(xc)
            kk = pyxrt.kernel(self.dev, uuid, "MLIR_AIE")
            ib = inst.read_bytes()
            bo_i = pyxrt.bo(self.dev, len(ib), pyxrt.bo.flags.cacheable,
                            kk.group_id(1))
            bo_i.write(ib, 0)
            bo_i.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self.imgs[name] = {"k": kk, "inst": ib, "bo_i": bo_i,
                               "reg_s": time.perf_counter() - t0}
        if QKV and share is None:
            # T22 qkv: a FOURTH image, registered exactly like the other three.
            # Five data args (X, WQ, WK, WV -> OUTQKV), so it needs its own
            # BOs with ITS group ids (F-08: bound BOs must match the declared
            # parameters one for one).
            if not (QKV_XCLBIN and QKV_BIN):
                raise SystemExit("[qkv] --qkv needs --qkv-xclbin and --qkv-bin")
            t0 = time.perf_counter()
            xc = pyxrt.xclbin(str(QKV_XCLBIN))
            uuid = self.dev.register_xclbin(xc)
            kk = pyxrt.kernel(self.dev, uuid, "MLIR_AIE")
            ib = pathlib.Path(QKV_BIN).read_bytes()
            bo_i = pyxrt.bo(self.dev, len(ib), pyxrt.bo.flags.cacheable,
                            kk.group_id(1))
            bo_i.write(ib, 0)
            bo_i.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self.imgs["qkv"] = {"k": kk, "inst": ib, "bo_i": bo_i,
                                "reg_s": time.perf_counter() - t0}
            print("[qkv] fused image %s registered (%s B insts) in %.2f s"
                  % (pathlib.Path(QKV_XCLBIN).name, format(len(ib), ","),
                     self.imgs["qkv"]["reg_s"]), flush=True)
        if P10 and share is None:
            # T22 p10: a FIFTH image, the o projection, registered exactly like
            # the other four.  It declares the SAME three data arguments as
            # t7_p5 (X, W -> OUT), so its BOs are the delivered sizes on ITS
            # OWN group ids (F-08: the ERT maps the BO list onto the declared
            # sequence of THAT kernel).
            if not (P10_XCLBIN and P10_BIN):
                raise SystemExit("[p10] --p10 needs --p10-xclbin and --p10-bin")
            t0 = time.perf_counter()
            xc = pyxrt.xclbin(str(P10_XCLBIN))
            uuid = self.dev.register_xclbin(xc)
            kk = pyxrt.kernel(self.dev, uuid, "MLIR_AIE")
            ib = pathlib.Path(P10_BIN).read_bytes()
            bo_i = pyxrt.bo(self.dev, len(ib), pyxrt.bo.flags.cacheable,
                            kk.group_id(1))
            bo_i.write(ib, 0)
            bo_i.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self.imgs["attout"] = {"k": kk, "inst": ib, "bo_i": bo_i,
                                   "reg_s": time.perf_counter() - t0}
            print("[p10] o image %s registered (%s B insts) in %.2f s"
                  % (pathlib.Path(P10_XCLBIN).name, format(len(ib), ","),
                     self.imgs["attout"]["reg_s"]), flush=True)
        if BIGM and share is None:
            # t22_prodasm: three more images, registered exactly like qkv/attout.
            for _n, (_xcl, _inst) in BIGM_IMGS.items():
                _t0 = time.perf_counter()
                _xc = pyxrt.xclbin(str(_xcl))
                _uuid = self.dev.register_xclbin(_xc)
                _kk = pyxrt.kernel(self.dev, _uuid, "MLIR_AIE")
                _ib = pathlib.Path(_inst).read_bytes()
                _bo_i = pyxrt.bo(self.dev, len(_ib), pyxrt.bo.flags.cacheable,
                                 _kk.group_id(1))
                _bo_i.write(_ib, 0)
                _bo_i.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
                self.imgs[_n] = {"k": _kk, "inst": _ib, "bo_i": _bo_i,
                                 "reg_s": time.perf_counter() - _t0}
                print("[bigm] %s registered (%s B insts) in %.2f s"
                      % (_n, format(len(_ib), ","),
                         self.imgs[_n]["reg_s"]), flush=True)
        if share is not None:
            # The kernels and instruction BOs are read-only constants; the group
            # ids below are the donor's, which is why a shared chain's BO table
            # is built with THIS Dev's pyxrt.bo() calls on the SAME ids.
            self.imgs = dict(share.imgs)
            print("[chain2b] dev %r shares %d kernels/contexts with %r "
                  "(no new hw context: the driver admits ~26 per process and "
                  "one chain needs 18)" % (tag, len(self.imgs), share.tag),
                  flush=True)
        fl = pyxrt.bo.flags.host_only
        g = lambda im, i: self.imgs[im]["k"].group_id(i)
        self.bo = {}
        # the attention image's 7 data args, in declaration order
        #   x(3) w(4) qp(5) kb(6) vb(7) -> outq(8) outo(9)
        if self.has_ip:
            # T7 split pair: the projection image declares (X, W -> OUTQ) and the
            # ip image (QP, KB, VB -> OUTO); each argument needs a BO carrying
            # THAT kernel's group id.  T18: the projection runs 96-row blocks
            # and the ip runs (IP_MQ rows x IP_ND keys) blocks, so the BO sizes
            # are the IMAGE's object sizes, not SEQ x D.
            self.bo["a_x"] = pyxrt.bo(self.dev, PROJ_X_BYTES, fl, g("att", 3))
            self.bo["a_w"] = pyxrt.bo(self.dev, W_BYTES, fl, g("att", 4))
            self.bo["a_outq"] = pyxrt.bo(self.dev, PROJ_OUT_BYTES, fl, g("att", 5))
            self.bo["a_qp"] = pyxrt.bo(self.dev, IP_QP_BYTES, fl, g("attip", 3))
            self.bo["a_kb"] = pyxrt.bo(self.dev, KB_BYTES * HEADS, fl,
                                       g("attip", 4))
            self.bo["a_vb"] = pyxrt.bo(self.dev, KB_BYTES * HEADS, fl,
                                       g("attip", 5))
            self.bo["a_outo"] = pyxrt.bo(self.dev, IP_OUT_BYTES, fl,
                                         g("attip", 6))
        else:
            for i, n in enumerate(("x", "w", "qp", "kb", "vb")):
                sz = {"x": PROJ_X_BYTES, "w": W_BYTES, "qp": ACT_BYTES,
                      "kb": KB_BYTES * HEADS, "vb": KB_BYTES * HEADS}[n]
                self.bo["a_" + n] = pyxrt.bo(self.dev, sz, fl, g("att", 3 + i))
            self.bo["a_outq"] = pyxrt.bo(self.dev, PROJ_OUT_BYTES, fl,
                                         g("att", 8))
            self.bo["a_outo"] = pyxrt.bo(self.dev, ACT_BYTES, fl, g("att", 9))
        for n in ("SBG", "BG", "B3"):
            sz = {"SBG": FFN_SBG_BYTES, "BG": BG_BYTES, "B3": B3_BYTES}[n]
            self.bo["f_" + n] = pyxrt.bo(self.dev, sz, fl, g("ffn", FFN_GROUPS[n]))
        # T22 PFILL: a second BG/B3 pair so layer i+1's 132.7 MB payload can be
        # written while layer i's dispatch runs.  +132.7 MB of BO memory.
        self.pf_pair = 0
        if T22_PFILL:
            for n in ("BG", "B3"):
                sz = {"BG": BG_BYTES, "B3": B3_BYTES}[n]
                self.bo["f_" + n + "2"] = pyxrt.bo(self.dev, sz, fl,
                                                   g("ffn", FFN_GROUPS[n]))
            self.pf_bg = [self.bo["f_BG"], self.bo["f_BG2"]]
            self.pf_b3 = [self.bo["f_B3"], self.bo["f_B32"]]
        for n in ("A", "X", "Y"):
            self.bo["f_" + n] = pyxrt.bo(self.dev, FFN_A_BYTES, fl,
                                        g("ffn", FFN_GROUPS[n]))
        if QKV:
            # T22 qkv: the fused image's five arguments, in declaration order
            #   x(3) wq(4) wk(5) wv(6) -> out(7)
            for nm, sz, gid in (("x", PROJ_X_BYTES, 3), ("wq", W_BYTES, 4),
                                ("wk", W_BYTES, 5), ("wv", W_BYTES, 6),
                                ("out", 3 * PROJ_OUT_BYTES, 7)):
                self.bo["q_" + nm] = pyxrt.bo(self.dev, sz, fl, g("qkv", gid))
        if P10:
            # T22 p10: the o image's three arguments, in declaration order
            #   x(3) w(4) -> out(5)
            for nm, sz, gid in (("x", PROJ_X_BYTES, 3), ("w", W_BYTES, 4),
                                ("out", PROJ_OUT_BYTES, 5)):
                self.bo["o_" + nm] = pyxrt.bo(self.dev, sz, fl,
                                              g("attout", gid))
        if BIGM:
            # t22_prodasm: (x, w -> out) on each image's own group ids 3/4/5
            for _n, _sz in BIGM_SIZES.items():
                for _k, _s, _gid in (("x", _sz[0], 3), ("w", _sz[1], 4),
                                     ("o", _sz[2], 5)):
                    self.bo["b_%s_%s" % (_n, _k)] = pyxrt.bo(
                        self.dev, _s, fl, g(_n, _gid))
        self.stats = {"dev_ms": {}, "write_ms": {}, "calls": 0,
                      "attempts": [], "retries": 0}

    # -- one dispatch with the contention protocol -------------------------
    def _run(self, name, args, out_keys, tag):
        im = self.imgs[name]
        t0 = time.perf_counter()
        try:
            run = im["k"](3, im["bo_i"], len(im["inst"]) // 4, *args)
            state = int(run.wait(TIMEOUT_MS))
        except Exception as exc:  # noqa: BLE001 (0xc00002b6 and friends)
            state = -1
            tag = "%s (XRT %s)" % (tag, type(exc).__name__)
        wall = 1e3 * (time.perf_counter() - t0)
        t0 = time.perf_counter()
        for b in out_keys:
            b.sync(self.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        pmark("host.osync", t0)
        gmark("host.osync", t0)
        if PROF is not None:
            PROF["host.wait"] = PROF.get("host.wait", 0.0) + wall
        # T22 glue: split by dispatch family, so blk.ffn's contents are
        # separable (the ffn dispatch is 85 % device+staging, 15 % host CPU).
        gadd("host.wait.ffn" if name == "ffn" else "host.wait.att", wall / 1e3)
        return state, wall, tag

    def dispatch(self, name, fills, outs, tag=""):
        """fills: [(bo, bytes)] written + synced; outs: [(bo, nbytes)] read."""
        pyxrt = self.pyxrt
        t0 = time.perf_counter()
        for bo, payload in fills:
            _trace("%s fill %d B" % (name, len(payload)))
            # t22_prodasm DIAGNOSTIC HOOK (env-gated, not a product path): capture
            # the exact host bytes written into f_X.  Reading the BO back is
            # useless -- host_only + sync(TO_DEVICE) means a post-dispatch read
            # returns the device side (zeros), not what the host wrote.
            _cap = os.environ.get("T18_CAP_X")
            if _cap:
                for _nm in ("f_A", "f_X", "f_BG"):
                    if bo is self.bo.get(_nm):
                        with open("%s.%s" % (_cap, _nm), "wb") as _fh:
                            _fh.write(bytes(payload))
            bo.write(payload, 0)
            bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        _trace("%s fills done" % name)
        tw = 1e3 * (time.perf_counter() - t0)
        pmark("host.fill", t0)
        gadd("host.fill.ffn" if name == "ffn" else "host.fill.att", tw / 1e3)
        if name == "att":
            if self.has_ip:
                args = [self.bo["a_" + n] for n in ("x", "w", "outq")]
                out_keys = [self.bo["a_outq"]]
            else:
                args = [self.bo["a_" + n] for n in ("x", "w", "qp", "kb", "vb")]
                out_keys = [self.bo["a_outq"], self.bo["a_outo"]]
        elif name == "attip":
            args = [self.bo["a_" + n] for n in ("qp", "kb", "vb", "outo")]
            out_keys = [self.bo["a_outo"]]
        elif name == "qkv":
            # T22 qkv: the fused image declares FIVE data args, in order
            #   X, WQ, WK, WV, OUTQKV
            # and every one of them is bound (the T18 rule above: the ERT maps
            # the BO list onto the declared sequence).  The single output BO
            # carries the three projections back to back.
            args = [self.bo["q_" + n] for n in ("x", "wq", "wk", "wv", "out")]
            out_keys = [self.bo["q_out"]]
        elif name == "attout":
            # T22 p10: the o image declares the same three arguments as t7_p5
            #   X, W -> OUT
            args = [self.bo["o_x"], self.bo["o_w"], self.bo["o_out"]]
            out_keys = [self.bo["o_out"]]
        elif name in ("norm", "proj", "swiglu", "down", "swiglu1", "down1"):
            # t22_prodasm: (x, w -> out), same three-argument shape as attout
            args = [self.bo["b_%s_x" % name], self.bo["b_%s_w" % name],
                    self.bo["b_%s_o" % name]]
            out_keys = [self.bo["b_%s_o" % name]]
        else:
            args = [self.bo["f_" + n] for n in FFN_ORDER]
            out_keys = [self.bo["f_Y"]]
        # EVERY declared data argument must be bound: the ERT maps the BOs to
        # the runtime sequence's arguments and a short list leaves the tail
        # unbound (measured: att hung for the whole 300 s timeout when the two
        # outputs were omitted, while the outputs' group ids were still
        # allocated -- so "it completed anyway" is not evidence).
        #
        # T18 FIX: pass EXACTLY the declared arguments -- do NOT append the
        # output BOs a second time (the delivered driver did).  The delivered
        # M=96 FFN image tolerates the extra BO; the same source compiled at
        # M=128 aborts inside the ERT launch (0xC0000005), reproduced standalone:
        #   t18_m128, 6 args -> ERT_CMD_STATE_COMPLETED (51.8 ms)
        #   t18_m128, 7 args -> process dies at the kernel call
        if not args:
            args = list(out_keys)
        if outs:
            pass
        elif name == "att":
            outs = ([(self.bo["a_outq"], PROJ_OUT_BYTES),
                     (self.bo["a_outo"], IP_OUT_BYTES)] if not self.has_ip
                    else [(self.bo["a_outq"], PROJ_OUT_BYTES)])
        elif name == "attip":
            outs = [(self.bo["a_outo"], IP_OUT_BYTES)]
        elif name == "qkv":
            # T22 qkv: ONE read of the joint buffer; the Loop slices it into
            # Q / K / V (each 737,280 B for the delivered shape).
            outs = [(self.bo["q_out"], 3 * PROJ_OUT_BYTES)]
        elif name == "attout":
            # T22 p10: ONE read of the o image's single output (737,280 B)
            outs = [(self.bo["o_out"], PROJ_OUT_BYTES)]
        else:
            outs = [(self.bo["f_Y"], FFN_A_BYTES)]
        for i in range(len(RETRY_SLEEPS) + 1):
            _trace("%s kernel call" % name)
            state, wall, tag2 = self._run(name, args, out_keys, tag)
            _trace("%s kernel state=%s wall=%.2f ms" % (name, state, wall))
            self.stats["attempts"].append({"img": name, "state": state,
                                           "wall_ms": wall})
            if state == COMPLETED:
                # T4-G: per-DISPATCH device time (att/q, att/k, att/v, att/ip,
                # att/o, ffn), so the attention's 28 s can be attributed to its
                # five dispatches instead of being one number.
                dbt = self.stats.setdefault("dev_by_tag", {})
                dbt[tag] = dbt.get(tag, 0.0) + wall
                self.stats["dev_ms"][name] = self.stats["dev_ms"].get(
                    name, 0.0) + wall
                self.stats["write_ms"][name] = self.stats["write_ms"].get(
                    name, 0.0) + tw
                self.stats["calls"] += 1
                t0 = time.perf_counter()
                raws = [bytes(bo.read(n, 0)) for bo, n in outs]
                pmark("host.read", t0)
                gadd("host.read.ffn" if name == "ffn" else "host.read.att",
                     time.perf_counter() - t0)
                return raws, wall, tw
            self.stats["retries"] += 1
            if i < len(RETRY_SLEEPS):
                print("[t3d]   %s RETRY %d/%d after %.1f ms (%s)"
                      % (tag2, i + 1, len(RETRY_SLEEPS), wall, state),
                      flush=True)
                time.sleep(RETRY_SLEEPS[i])
        raise SystemExit("[t3d] REFUSAL: %s never completed (%s); checkpoint "
                         "holds every dispatch already measured" % (tag, state))

    # -- T20 U-1: the relaunch barrier -------------------------------------
    def barrier(self, name="attip"):
        """One launch of a DIFFERENT xclbin between two FFN launches.

        Measured (t20_bench.py, 2026-10-19):

          * the FFN image t11_nh5 does NOT survive a same-image relaunch: after
            one COMPLETED launch, the next launch of the SAME image never
            raises completion -- the ERT's own watchdog reports state 8 at
            ~7.0 s (run.state() stays 8), and the launch after it completes in
            ~1.0-1.5 s (the watchdog's array reset) with the SAME output bytes;
          * host spacing does not matter: gaps of 0 / 10 / 30 / 100 / 500 /
            2000 ms all stall at the same rate (1 of 2 launches);
          * one launch of a DIFFERENT xclbin in between removes it completely:
            the ip image (1.6-1.8 ms) and the projection image (5.3-6.5 ms)
            both give 0 stalls over 4/4 and 3/3 launches at the full 39-40 ms;
          * a second kernel handle on the SAME xclbin is NOT a fix: it returns
            COMPLETED in 2-4 ms, i.e. it reports a completion without doing the
            132 MB weight pass (a false completion -- do not use it).

        Why this is free in the delivered loops: 128x128 / 192x128 run the FFN
        once per layer with attention dispatches (other xclbins) in between, so
        the image is never relaunched back-to-back -- 240 launches, 0 retries.
        At 256x256 (SEQ=288 > FFN_M=96) the FFN runs 3x in a row, and exactly
        the 2 relaunches stall: measured 2 retries/layer, 200-264 retries per
        8-step run.  This barrier puts one ip launch between the blocks: same
        arithmetic (the ip kernel re-runs on the BOs it already holds and the
        result is discarded -- the attention output was drained long before),
        +1.7 ms per gap.

        Cost when FFN_BLOCKS == 1: never called.
        """
        im = self.imgs[name]
        if name == "attip":
            args = [self.bo["a_" + n] for n in ("qp", "kb", "vb", "outo")]
        elif self.has_ip:
            args = [self.bo["a_" + n] for n in ("x", "w", "outq")]
        else:
            args = [self.bo["a_" + n] for n in ("x", "w", "qp", "kb", "vb")]
        t0 = time.perf_counter()
        try:
            run = im["k"](3, im["bo_i"], len(im["inst"]) // 4, *args)
            state = int(run.wait(TIMEOUT_MS))
        except Exception:                                     # noqa: BLE001
            state = -1
        wall = 1e3 * (time.perf_counter() - t0)
        st = self.stats
        st["barrier_calls"] = st.get("barrier_calls", 0) + 1
        st["barrier_ms"] = st.get("barrier_ms", 0.0) + wall
        st["barrier_states"] = st.get("barrier_states", {})
        st["barrier_states"][state] = st["barrier_states"].get(state, 0) + 1
        if state != COMPLETED:
            raise SystemExit("[t20] REFUSAL: barrier (%s) did not complete (%s)"
                             % (name, state))
        return wall


# ---------------------------------------------------------------------------
# T6: the four refiner blocks' decoded f32 weights, persisted
# ---------------------------------------------------------------------------
_RC_HIT = [False]


def _refblock_cached(RW, key):
    """RW.block(key), but the decoded f32 tensors survive the process.

    The decode is 5.5 s per process (4 x ~1.4 s, 2.83 GB) and the values are
    identical on every run, so they are saved as .npy (exact for f32) and read
    back.  The gate is BYTES: t6_refcache_gate.py compares every tensor of all
    four blocks against a fresh decode, and the delivered latents are compared
    byte-for-byte with t3d5's.
    """
    _RC_HIT[0] = False
    if REFCACHE is None:
        return RW.block(key)
    d = pathlib.Path(REFCACHE) / key
    man = d / "manifest.json"
    if man.exists():
        try:
            names = json.loads(man.read_text())
            out = {}
            for nm in names:
                if nm.startswith("!"):
                    out[nm[1:]] = None
                else:
                    out[nm] = np.load(d / (nm + ".npy"))
            _RC_HIT[0] = True
            return out
        except Exception as exc:                              # noqa: BLE001
            print("[t6] refiner cache unreadable (%s); decoding" % exc,
                  flush=True)
    out = RW.block(key)
    try:
        d.mkdir(parents=True, exist_ok=True)
        names = []
        for nm, v in out.items():
            if v is None:
                names.append("!" + nm)
            else:
                np.save(d / (nm + ".npy"), v)
                names.append(nm)
        (d / "manifest.json").write_text(json.dumps(names))
    except Exception as exc:                                  # noqa: BLE001
        print("[t6] refiner cache write failed (%s); running uncached" % exc,
              flush=True)
    return out


# ---------------------------------------------------------------------------
# the block on the NPU
# ---------------------------------------------------------------------------
class Loop:
    def __init__(self, dev: Dev, tag="t3d", dimk64=False, chain=0):
        # T22_CHAIN2B: the chain owns this Loop's host scratch (n1buf/n2buf/
        # n3buf, qm_buf, y_buf) and is what PD.bigm_attention dispatches on.
        self.chain = int(chain)
        import t3_real as T3
        self.T3 = T3
        self.dev = dev
        self.RW = T3.Real()
        self.dimk64 = dimk64
        self.perm = K.load_perm(HERE / "t3d_perm_96x128.json")
        self.inv = K.load_perm_inv(HERE / "t3d_perm_96x128.json")
        self.perm64 = K.load_perm(HERE / "t3d_perm_96x64.json") if dimk64 else None
        # T18: the ip query-block layout is a function of (IP_MQ, HD); SEQ=96
        # reuses the committed 96x128 table unchanged.
        self.perm_q = K.load_perm(HERE / PERM_Q[IP_MQ])
        self.inv_q = K.load_perm_inv(HERE / PERM_Q[IP_MQ])
        # T22 glue-2: the int64 index vectors the nJIT gather wants, built once
        # (the delivered path re-casts them on every one of the 960 calls), and
        # the one 96-row X block buffer (see _xblock).
        self.inv64 = self.inv.astype(np.int64)
        self.inv_q64 = self.inv_q.astype(np.int64)
        self.xblk = np.zeros((PROJ_M, D), np.float32) if T22_GLUE2 else None
        # T22 glue-2b: the per-layer activation buffers.  attention_fast made
        # 5 fresh arrays per layer (3 x 491 KB Q/K/V + o_flat + a_flat = 2.45 MB)
        # and ffn one more (y, 491 KB); the measured att.other bucket (1.767 s
        # on t22g_L) is what that allocation churn costs.  Reused here, same
        # values written in the same order -- bit-identical by construction.
        if T22_GLUE2:
            self.qm_buf = {nm: np.empty((HEADS, SEQ, HD), np.float32)
                           for nm in ("q", "k", "v")}
            self.o_flat_buf = np.empty((SEQ, D), np.float32)
            self.a_flat_buf = np.empty((SEQ, D), np.float32)
            self.y_buf = np.empty((SEQ, D), np.float32)
        else:
            self.qm_buf = self.o_flat_buf = self.a_flat_buf = self.y_buf = None
        self.fold = np.float32(1.0 / np.sqrt(HD))
        self.wcache = {}
        # T18: the FFN X stream is one image-block's worth (M*D*2), not SEQ's.
        self.zero_x = bytes(FFN_A_BYTES)
        self.written = set()
        # T22 fast glue: three (SEQ, D) f32 workspaces (4.4 MB total).  Only
        # x1/xm/p live here; the returned `out` is fresh because it escapes to
        # the next layer (and, with a batch, to the next image).
        self.n1buf = np.empty((SEQ, D), np.float32)
        self.n2buf = np.empty((SEQ, D), np.float32)
        self.n3buf = np.empty((SEQ, D), np.float32)
        self.tag = tag

    # -- weight images (packed once per layer, kept in RAM) ---------------
    def ximg(self, x2d):
        """The projection's X payload, in whichever k-grouping this run uses."""
        if self.dimk64:
            return x_image64(np.asarray(x2d, np.float32), self.perm64)
        return K.x_image(np.asarray(x2d, np.float32))

    def w_images(self, key):
        if key in self.wcache:
            return self.wcache[key]
        t0 = time.perf_counter()
        out = {}
        for nm, tail in (("q", "attention.to_q.weight"),
                         ("k", "attention.to_k.weight"),
                         ("v", "attention.to_v.weight"),
                         ("o", "attention.to_out.0.weight")):
            W = self.RW.get(key + "." + tail)
            img = np.ascontiguousarray(
                pack_w64(W) if self.dimk64 else pack_w(W))
            del W
            if WPACK is None:
                out[nm] = img.tobytes()
            else:
                # T6: one 2.0 GB array instead of 120 x 16.6 MB bytes objects
                slot = (len(WPACK_SLOTS))
                WPACK_SLOTS[(key, nm)] = slot
                buf = WPACK[slot * W_BYTES:(slot + 1) * W_BYTES]
                buf[:] = img.reshape(-1)
                out[nm] = buf
        out["_s"] = time.perf_counter() - t0
        WPACK_SECONDS[0] += out["_s"]
        WPACK_SECONDS[1] += 1
        self.wcache[key] = out
        return out

    def params(self, key):
        if key not in self._pcache:
            g = lambda t: self.RW.get(key + "." + t)
            self._pcache[key] = {
                "norm_q": g("attention.norm_q.weight"),
                "norm_k": g("attention.norm_k.weight"),
                "attention_norm1": g("attention_norm1.weight"),
                "attention_norm2": g("attention_norm2.weight"),
                "ffn_norm1": g("ffn_norm1.weight"),
                "ffn_norm2": g("ffn_norm2.weight"),
                "adaln_w": (g("adaLN_modulation.0.weight")
                            if key + ".adaLN_modulation.0.weight" in self.RW.rec
                            else None),
                "adaln_b": (g("adaLN_modulation.0.bias")
                            if key + ".adaLN_modulation.0.bias" in self.RW.rec
                            else None),
            }
        return self._pcache[key]

    _pcache = {}
    _rcache = {}

    def refblock(self, key):
        """The four refiner blocks' weights, decoded ONCE per process.

        They are identical on every one of the 8 steps, and RW.block() decodes
        ~708 MB of f32 (w1|w3|w2 157 MB each + to_q/k/v/out 59 MB each) every
        time it is called -- which is 16.2 s of the 30 s a step used to spend
        outside the NPU.  Caching 4 blocks costs ~2.8 GB of RAM; the gate is
        that the whole run stays BIT-IDENTICAL to the uncached one.
        """
        if key not in self._rcache:
            t0 = time.perf_counter()
            self._rcache[key] = _refblock_cached(self.RW, key)
            print("[t3d] refiner weights %-14s %-18s %5.1f s (%.2f GB)"
                  % ("loaded" if _RC_HIT[0] else "decoded", key,
                     time.perf_counter() - t0,
                     sum(v.nbytes for v in self._rcache[key].values()
                         if v is not None) / 2**30), flush=True)
        return self._rcache[key]

    # -- the attention branch: 5 dispatches of ONE image ------------------
    def att_dispatch(self, fills, tag, tim, img="att"):
        raw, wall, tw = self.dev.dispatch(img, fills, [], tag=tag)
        tim["dev"] += wall
        tim["write"] += tw
        if img == "attip":
            return None, raw[0]
        return raw[0], (raw[1] if len(raw) > 1 else None)

    def attention(self, x1, key, cos, sin, tim):
        if FAST_HOST:
            return self.attention_fast(x1, key, cos, sin, tim)
        return self.attention_ref(x1, key, cos, sin, tim)

    def attention_fast(self, x1, key, cos, sin, tim):
        """4 projections + the ip chain, in row blocks (T18).

        SEQ == 96 reduces to the delivered five dispatches exactly (one
        projection block of 96 rows, one ip query block of 96): that is the
        A/B arm this driver is gated on.  For SEQ > 96:
          * the projection (m=96 image) is row-independent -> 96-row blocks;
          * the ip image has ND = SEQ keys, M = IP_MQ queries -> query blocks.
        Every block boundary is a whole number of 16-row tiles and of 8x8
        blocked-layout blocks, so no value is split or re-associated.
        """
        T3 = self.T3
        bo = self.dev.bo
        sc = SCRATCH
        P = self.params(key)
        W = self.w_images(key)
        _g_in = gtotal()
        _g_t = time.perf_counter()
        n_proj = (SEQ + PROJ_M - 1) // PROJ_M
        n_ip = (SEQ + IP_MQ - 1) // IP_MQ
        assert SEQ % IP_MQ == 0, (SEQ, IP_MQ)
        _trace("att %s start seq=%d n_proj=%d n_ip=%d" % (key, SEQ, n_proj, n_ip))
        # T18 FIX: assign each block straight into the (HEADS, SEQ, HD) array.
        # The first version appended H.unpack_bytes(...) results to a list, but
        # unpack_bytes returns a VIEW of the shared Scratch buffer -- block 1
        # overwrote block 0, so rows 0..95 became block 1's padded rows (the
        # host-ref gate showed rel 1.00 for rows 0..95 and 0.008 for 96..127,
        # exactly this aliasing).  SEQ == 96 (n_proj == 1) never saw it.
        _t_alloc = time.perf_counter()
        Qm = (self.qm_buf if T22_GLUE2 else
              {nm: np.empty((HEADS, SEQ, HD), np.float32)
               for nm in ("q", "k", "v")})
        gmark("att.alloc", _t_alloc)
        if QKV:
            # ---- T22 qkv: ONE fused dispatch per 96-row block --------------
            # Same X bytes, the same W objects, the same k-group order; the
            # image writes [Q | K | V] and each slice is the delivered
            # projection's own output (byte gate: qkv/QKV-DESIGN.md §4.1).
            for bi in range(n_proj):
                r0 = bi * PROJ_M
                rows = min(PROJ_M, SEQ - r0)
                _t_xb = time.perf_counter()
                xb = _xblock(self, x1, r0, rows)
                gmark("att.xblk", _t_xb)
                t0 = time.perf_counter()
                Xb = H.x_image64_bytes(xb, self.perm64, sc, "ximg")
                pmark("att.ximg", t0)
                gmark("att.qkv.img", t0)
                tim["pack"] += 1e3 * (time.perf_counter() - t0)
                raws, wall, tw = self.dev.dispatch(
                    "qkv", [(bo["q_x"], Xb), (bo["q_wq"], W["q"]),
                            (bo["q_wk"], W["k"]), (bo["q_wv"], W["v"])],
                    [], tag="att/qkv")
                tim["dev"] += wall
                tim["write"] += tw
                out = raws[0]
                _mv = memoryview(out) if T22_GLUE2 else None
                for pi, nm in enumerate(("q", "k", "v")):
                    t0 = time.perf_counter()
                    qraw = (_mv[pi * PROJ_OUT_BYTES:(pi + 1) * PROJ_OUT_BYTES]
                            if T22_GLUE2 else
                            out[pi * PROJ_OUT_BYTES:(pi + 1) * PROJ_OUT_BYTES])
                    Qm[nm][:, r0:r0 + rows] = (
                        unpack_bytes_mv(qraw, self.inv64,
                                        (HEADS, PROJ_M * HD), sc, "u" + nm)
                        if T22_GLUE2 else
                        H.unpack_bytes(qraw, self.inv, (HEADS, PROJ_M * HD),
                                       sc, "u" + nm)
                    ).reshape(HEADS, PROJ_M, HD)[:, :rows]
                    pmark("att.unpack3", t0)
                    gmark("att.unpack3", t0)
                _trace("att %s qkv block %d done" % (key, bi))
        elif XPACK_ONCE:
            # ---- T22 xpack-once: X is packed once, used by q, k and v -----
            # Purely host-side: the three launches still receive the same
            # 737,280 B X image, so the device arithmetic is untouched.  The
            # delivered loop re-packs the identical bytes three times per row
            # block (QKV-DESIGN.md §6.7).
            for bi in range(n_proj):
                r0 = bi * PROJ_M
                rows = min(PROJ_M, SEQ - r0)
                _t_xb = time.perf_counter()
                xb = _xblock(self, x1, r0, rows)
                gmark("att.xblk", _t_xb)
                t0 = time.perf_counter()
                Xb = H.x_image64_bytes(xb, self.perm64, sc, "ximg")
                pmark("att.ximg", t0)
                gmark("att.qkv.img", t0)
                tim["pack"] += 1e3 * (time.perf_counter() - t0)
                for nm in ("q", "k", "v"):
                    raw = self.att_dispatch([(bo["a_x"], Xb),
                                             (bo["a_w"], W[nm])],
                                            "att/" + nm, tim)
                    t0 = time.perf_counter()
                    Qm[nm][:, r0:r0 + rows] = H.unpack_bytes(
                        raw[0], self.inv, (HEADS, PROJ_M * HD), sc, "u" + nm
                    ).reshape(HEADS, PROJ_M, HD)[:, :rows]
                    pmark("att.unpack3", t0)
                    _trace("att %s proj %s done" % (key, nm))
        else:
            for nm in ("q", "k", "v"):
                for bi in range(n_proj):
                    r0 = bi * PROJ_M
                    rows = min(PROJ_M, SEQ - r0)
                    _t_xb = time.perf_counter()
                    xb = _xblock(self, x1, r0, rows)
                    gmark("att.xblk", _t_xb)
                    t0 = time.perf_counter()
                    Xb = H.x_image64_bytes(xb, self.perm64, sc, "ximg")
                    pmark("att.ximg", t0)
                    gmark("att.qkv.img", t0)
                    tim["pack"] += 1e3 * (time.perf_counter() - t0)
                    raw = self.att_dispatch([(bo["a_x"], Xb),
                                             (bo["a_w"], W[nm])],
                                            "att/" + nm, tim)
                    t0 = time.perf_counter()
                    Qm[nm][:, r0:r0 + rows] = H.unpack_bytes(
                        raw[0], self.inv, (HEADS, PROJ_M * HD), sc, "u" + nm
                    ).reshape(HEADS, PROJ_M, HD)[:, :rows]
                    pmark("att.unpack3", t0)
                _trace("att %s proj %s done" % (key, nm))
        t0 = time.perf_counter()
        wq = bf16f32(P["norm_q"] * self.fold)
        kh = rope_rms_jit(Qm["k"], P["norm_k"], cos, sin,
                          sc.get("kh", (HEADS, SEQ, HD), np.float32),
                          sc.get("rms0", (HEADS, SEQ, HD), np.float32))
        qh = rope_rms_jit(Qm["q"], wq, cos, sin,
                          sc.get("qh", (HEADS, SEQ, HD), np.float32),
                          sc.get("rms1", (HEADS, SEQ, HD), np.float32))
        pmark("att.rms_rope", t0)
        gmark("att.rms_rope", t0)
        t0 = time.perf_counter()
        kb = H.pack_b_nk_f32(kh, sc, "kb")
        pmark("att.packkb", t0)
        gmark("att.packkb", t0)
        t0 = time.perf_counter()
        vb = H.pack_b_nk_t_f32(Qm["v"], sc, "vb")
        pmark("att.packvb", t0)
        gmark("att.packvb", t0)
        tim["pack"] += time.perf_counter() - t0
        _trace("att %s kv packed" % key)
        o_flat = (self.o_flat_buf if T22_GLUE2
                  else np.empty((SEQ, D), np.float32))
        for bi in range(n_ip):
            _trace("att %s ip block %d" % (key, bi))
            q0 = bi * IP_MQ
            t0 = time.perf_counter()
            a_bytes = H.pack_blocked_bytes(qh[:, q0:q0 + IP_MQ].reshape(
                HEADS, IP_MQ * HD), self.perm_q, sc, "qp")
            pmark("att.packq", t0)
            gmark("att.packq", t0)
            _, raw_o = self.att_dispatch(
                [(bo["a_qp"], a_bytes), (bo["a_kb"], kb), (bo["a_vb"], vb)],
                "att/ip", tim, img=("attip" if self.dev.has_ip else "att"))
            t0 = time.perf_counter()
            if T22_GLUE2 and (SEQ - q0) >= IP_MQ:
                # straight into the destination rows: same kernel, same inv,
                # same element order -- only the scratch copy disappears.
                # ONLY for a FULL block: unpack_blocked_t derives its head count
                # from out.shape, so a partial block would misread the drain.
                unpack_bytes_t_into(raw_o, self.inv_q64, HD,
                                    o_flat[q0:q0 + IP_MQ])
            else:
                o_flat[q0:q0 + IP_MQ] = H.unpack_bytes_t(
                    raw_o, self.inv_q, (HEADS, IP_MQ * HD), HD, sc, "ot")
            pmark("att.unpack_o", t0)
            gmark("att.unpack_o", t0)
        _trace("att %s ip done" % key)
        if os.environ.get("T18_DUMP") and key == "layers.0":
            self.last_qm = Qm
            _trace("dumped Q/K/V for %s" % key)
        a_flat = (self.a_flat_buf if T22_GLUE2
                  else np.empty((SEQ, D), np.float32))
        for bi in range(n_proj):
            _trace("att %s o block %d" % (key, bi))
            r0 = bi * PROJ_M
            rows = min(PROJ_M, SEQ - r0)
            _t_xb = time.perf_counter()
            xb = _xblock(self, o_flat, r0, rows)
            gmark("att.xblk", _t_xb)
            t0 = time.perf_counter()
            X2 = H.x_image64_bytes(xb, self.perm64, sc, "ximg2")
            tim["pack"] += time.perf_counter() - t0
            gmark("att.o.img", t0)
            if P10:
                # T22 p10: the SAME X2 bytes and the SAME W["o"] object, the
                # SAME "att/o" tag (so the ledger compares 1:1), a different
                # image and therefore its own BOs.
                raw_q, _ = self.att_dispatch([(bo["o_x"], X2),
                                              (bo["o_w"], W["o"])], "att/o",
                                             tim, img="attout")
            else:
                raw_q, _ = self.att_dispatch([(bo["a_x"], X2),
                                              (bo["a_w"], W["o"])], "att/o", tim)
            t0 = time.perf_counter()
            if T22_GLUE2 and rows == PROJ_M:
                unpack_bytes_t_into(raw_q, self.inv64, HD,
                                    a_flat[r0:r0 + rows])
            else:
                a_flat[r0:r0 + rows] = H.unpack_bytes_t(
                    raw_q, self.inv, (HEADS, PROJ_M * HD), HD, sc, "at")[:rows]
            pmark("att.unpack_a", t0)
            gmark("att.unpack_a", t0)
            tim["pack"] += time.perf_counter() - t0
        if GLUE is not None:
            gadd("att.other", max(0.0, (time.perf_counter() - _g_t)
                                  - (gtotal() - _g_in)))
        return a_flat

    def attention_ref(self, x1, key, cos, sin, tim):
        """4 projections + the ip, all on the NPU, in one image.

        d1..d3  W = to_q / to_k / to_v, X filled once  -> Q, K, V drains
        host    rms(qk_norm) + RoPE + bf16/bfp16ebs8 packaging (t3d_codec)
        d4      QP/KB/VB filled -> O drain (its proj drain is discarded)
        d5      X = the bf16 image of o, W = to_out -> the attention output
        """
        T3 = self.T3
        bo = self.dev.bo
        P = self.params(key)
        W = self.w_images(key)
        t0 = time.perf_counter()
        Xb = self.ximg(x1)
        tim["pack"] += 1e3 * (time.perf_counter() - t0)
        raw = {}
        for i, nm in enumerate(("q", "k", "v")):
            fills = []
            if i == 0:
                fills.append((bo["a_x"], Xb))
            fills.append((bo["a_w"], W[nm]))
            t0 = time.perf_counter()
            raw[nm] = self.att_dispatch(fills, "att/" + nm, tim)
            pass
        t0 = time.perf_counter()
        # T3.rms / T3.apply_rope are (seq, heads, hd) -- t3_real.attention's own
        # layout -- so the device drains are transposed into it and back.
        Qm = {nm: K.unpack_blocked(raw[nm][0], self.inv,
                                   (HEADS, M * HD)).reshape(HEADS, M, HD)
              for nm in ("q", "k", "v")}
        wq = bf16f32(P["norm_q"] * self.fold)
        if GLUE_FAST:
            kh = rope_rms_hmd(Qm["k"], P["norm_k"], cos, sin, T3)
            qh = rope_rms_hmd(Qm["q"], wq, cos, sin, T3)
        else:
            qm = np.ascontiguousarray(Qm["q"].transpose(1, 0, 2))
            km = np.ascontiguousarray(Qm["k"].transpose(1, 0, 2))
            q = np.asarray(T3.apply_rope(T3.rms(qm, wq), cos, sin), np.float32)
            k = np.asarray(T3.apply_rope(T3.rms(km, P["norm_k"]), cos, sin),
                           np.float32)
            qh = np.ascontiguousarray(q.transpose(1, 0, 2))
            kh = np.ascontiguousarray(k.transpose(1, 0, 2))
        a_bytes = K.pack_blocked(qh.reshape(HEADS, M * HD), self.perm)
        kb = K.pack_b_nk(kh)
        vb = K.pack_b_nk(np.ascontiguousarray(
            Qm["v"].transpose(0, 2, 1), np.float32))
        tim["pack"] += time.perf_counter() - t0
        # d4: the ip.  w still holds to_v and x the attention input: no rewrite.
        t0 = time.perf_counter()
        # att_dispatch returns (OUTQ, OUTO).  The ip's answer is OUTO, and OUTQ
        # of this dispatch is the *projection* (X @ to_v) -- reading raw[0] here
        # is the bug that made the whole image 11.65 dB: every block-step fed
        # x1 @ to_v into to_out as if it were o.  Measured cost of the slip:
        # attention output 5.5 % rmse, image 11.65 dB, and a false attribution to
        # the projection's C cadence (t3d_acc.json's first version).
        # T7 split: the ip-only image takes QP/KB/VB from ITS OWN BOs; the fused
        # image (A arm) takes them from the projection image's BOs.
        _, raw_o = self.att_dispatch(
            [(bo["a_qp"], a_bytes), (bo["a_kb"], kb.tobytes()),
             (bo["a_vb"], vb.tobytes())], "att/ip", tim,
            img=("attip" if self.dev.has_ip else "att"))
        o = K.unpack_blocked(raw_o, self.inv,
                             (HEADS, M * HD)).reshape(HEADS, M, HD)
        o_flat = np.ascontiguousarray(o.transpose(1, 0, 2)).reshape(M, D)
        X2 = self.ximg(o_flat)
        tim["pack"] += time.perf_counter() - t0
        # d5: to_out.  qp/kb/vb keep d4's bytes, so the ip re-runs on them and
        # its O is discarded (2.3 ms of the 25 ms this dispatch costs anyway).
        raw_q, _ = self.att_dispatch([(bo["a_x"], X2), (bo["a_w"], W["o"])],
                                     "att/o", tim)
        t0 = time.perf_counter()
        A = K.unpack_blocked(raw_q, self.inv,
                             (HEADS, M * HD)).reshape(HEADS, M, HD)
        a = np.ascontiguousarray(A.transpose(1, 0, 2)).reshape(M, D)
        tim["pack"] += time.perf_counter() - t0
        return a

    # -- the FFN, unchanged from the M2 image ----------------------------
    def ffn(self, x_mid, key, mod, tim, bg, b3, sbg_bytes):
        dev = self.dev
        _g_in = gtotal()
        _g_t = time.perf_counter()
        t0 = time.perf_counter()
        if NH_BG > 1 and not BG_PREPERM and not BO_RESIDENT:
            # T9 U-4b route (a): one pass over 88.5 MB, charged to `pack`.
            # T13: skipped when --cache-all pre-permuted it at startup, or when
            # --bo-resident already wrote the permuted payload into the BO.
            global _BG_SCRATCH, _BG_FLIP
            n_bg = len(bg)
            if _BG_SCRATCH is None or _BG_SCRATCH.size != n_bg:
                # DOUBLE buffered on purpose: the BO write is asynchronous, so
                # the layer-i+1 permutation must not overwrite the bytes the
                # layer-i dispatch may still be streaming (that cost two runs).
                _BG_SCRATCH = np.empty((2, n_bg), np.uint8)
                _BG_FLIP = 0
            _BG_FLIP ^= 1
            dst = _BG_SCRATCH[_BG_FLIP]
            src = (np.frombuffer(bg, np.uint8)
                   if isinstance(bg, (bytes, bytearray))
                   else np.ascontiguousarray(bg, np.uint8))
            (_P11.permute_bg if NH_BG_CORE else _P9.permute_bg)(
                src, dst, NH_BG)
            bg = dst
        # T18: FFN in blocks of FFN_M rows (the image's compiled M).  Every
        # block is the image's own runtime sequence on that row range; the
        # rows are independent, so the concatenation is exact.  SEQ=96/128 is
        # one block.  The SBG rows are all equal (asserted in block_step), so
        # the single ff_n-M SBG element is the same bytes for every block.
        y = (self.y_buf if T22_GLUE2 else np.empty((SEQ, D), np.float32))
        # sbg_bytes is a uint16 ELEMENT array (2 B/element) when FAST_HOST is on,
        # a bytes object otherwise: slice by bytes.
        sbg_m = (sbg_bytes[:FFN_SBG_BYTES // 2] if sbg_bytes.dtype == np.uint16
                 else sbg_bytes[:FFN_SBG_BYTES])
        for bi in range(FFN_BLOCKS):
            if bi and FFN_BARRIER:
                # T20 U-1: never relaunch the FFN image back-to-back (see
                # Dev.barrier -- the second launch never completes without a
                # different xclbin in between).
                self.dev.barrier("attip" if self.dev.has_ip else "att")
            r0 = bi * FFN_M
            rows = min(FFN_M, SEQ - r0)
            xb = np.zeros((FFN_M, D), np.float32)
            xb[:rows] = x_mid[r0:r0 + rows]
            t0 = time.perf_counter()
            a_img = (H.bf16_bytes_f32(xb, SCRATCH, "ffnA")
                     if FAST_HOST else
                     np.frombuffer(K.bf16_bytes(xb), np.uint8))
            fills = [(dev.bo["f_A"], a_img), (dev.bo["f_SBG"], sbg_m)]
            if T22_PFILL and CACHE_ALL and not BO_RESIDENT:
                # T22 PFILL: one_step already wrote this layer's payload into
                # this pair (and waited for it); point the argument names at
                # the pair the kernel must read and skip the 132.7 MB fill.
                dev.bo["f_BG"] = dev.pf_bg[dev.pf_pair]
                dev.bo["f_B3"] = dev.pf_b3[dev.pf_pair]
            elif not BO_RESIDENT:
                # T13 U-4b: the layer's BG/B3 BOs already hold the payload.
                fills += [(dev.bo["f_BG"], bg), (dev.bo["f_B3"], b3)]
            if "fx" not in self.written:
                fills.append((dev.bo["f_X"], self.zero_x))
                self.written.add("fx")
            tim["pack"] += time.perf_counter() - t0
            gmark("ffn.aimg", t0)
            _trace("ffn %s block %d dispatch" % (key, bi))
            raw, wall, tw = dev.dispatch(
                "ffn", fills, [(dev.bo["f_Y"], FFN_A_BYTES)], tag="ffn")
            _trace("ffn %s block %d returned" % (key, bi))
            tim["dev"] += wall
            tim["write"] += tw
            t0 = time.perf_counter()
            yb = (H.bf16_to_f32_bits(np.frombuffer(raw[0], np.uint16),
                                     (FFN_M, D), SCRATCH, "ffnY")
                  if FAST_HOST else K.bf16_to_f32(raw[0]).reshape(FFN_M, D))
            y[r0:r0 + rows] = yb[:rows]
            tim["pack"] += time.perf_counter() - t0
            gmark("ffn.ymap", t0)
        _trace("ffn %s done" % key)
        return y


# ---------------------------------------------------------------------------
# one block-step: attention on the NPU + FFN on the NPU
# ---------------------------------------------------------------------------
_PFILL = {"pool": None, "fut": None}


def _pfill_worker(dev, pair, bg, b3):
    """One layer payload -> the pair the *next* dispatch will read."""
    t0 = time.perf_counter()
    bo_bg = dev.pf_bg[pair]
    bo_b3 = dev.pf_b3[pair]
    bo_bg.write(bg, 0)
    bo_bg.sync(dev.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_b3.write(b3, 0)
    bo_b3.sync(dev.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    return time.perf_counter() - t0


def _pfill_ready(dev, li, layers):
    """Join layer li payload, then submit layer li+1 on the other pair.

    The join happens BEFORE the FFN dispatch of layer li, and the next submit
    only after it returned, so the device never reads a BO being written.
    """
    import concurrent.futures as _cf
    if _PFILL["pool"] is None:
        _PFILL["pool"] = _cf.ThreadPoolExecutor(max_workers=1)
    if _PFILL["fut"] is None:
        _PFILL["fut"] = _PFILL["pool"].submit(
            _pfill_worker, dev, li % 2, CACHE_BG[li], CACHE_B3[li])
    T22_PFILL_STATS["s"] += _PFILL["fut"].result()
    T22_PFILL_STATS["n"] += 1
    dev.pf_pair = li % 2
    nxt = (li + 1) % layers
    _PFILL["fut"] = _PFILL["pool"].submit(
        _pfill_worker, dev, nxt % 2, CACHE_BG[nxt], CACHE_B3[nxt])


def block_step(L: Loop, u, key, mod, cos, sin, tim, bg, b3, host_ref=False,
               sbg_bytes=None):
    T3 = L.T3
    P = L.params(key)
    _g_in = gtotal()
    _g_t = time.perf_counter()
    t0 = time.perf_counter()
    if T22_FAST_GLUE:
        # T22: same expressions, reused buffers, no temporaries (rms_into is
        # bit-equal to T3.rms -- t22_rmsbench.py asserts np.array_equal).
        x1 = rms_into(u, P["attention_norm1"], L.n1buf)
        if mod is not None:
            np.multiply(x1, mod[0], out=x1)
        n1 = x1
    else:
        n1 = T3.rms(u, P["attention_norm1"])
        x1 = n1 if mod is None else (n1 * mod[0]).astype(np.float32)
    tim["host"] += 1e3 * (time.perf_counter() - t0)
    pmark("post.norm1", t0)
    gmark("blk.norm1", t0)
    _trace("block %s attention start" % key)
    if os.environ.get("T18_SKIP_ATT"):
        a = x1                      # T18 probe only: FFN in the product context
    else:
        a = L.attention(x1, key, cos, sin, tim)
    _trace("block %s attention done" % key)
    t0 = time.perf_counter()
    if T22_FAST_GLUE:
        a2 = rms_into(a, P["attention_norm2"], L.n2buf)
        if mod is None:
            xm = np.add(u, a2, out=a2)
        else:
            np.multiply(a2, mod[1], out=a2)
            xm = np.add(u, a2, out=a2)
    else:
        a2 = T3.rms(a, P["attention_norm2"])
        xm = (u + a2) if mod is None else (u + mod[1] * a2)
        xm = xm.astype(np.float32)
    pmark("post.norm2", t0)
    gmark("blk.norm2", t0)
    # ---- FFN: SBG exactly as the M2 driver built it --------------------
    # T10: the batch driver builds it ONCE per layer and passes it in (it does
    # not depend on the image); batch=1 calls make_sbg(P, mod), the same
    # expressions on the same buffers as the delivered run.
    _t_sbg = time.perf_counter()
    if sbg_bytes is None:
        sbg_bytes = make_sbg(P, mod)
    # T4-F: the FFN image's norm core takes the [scale|shift|gamma] element once
    # per row-tile, which is exact ONLY if all 96 rows of SBG are equal -- true
    # by construction here (scale/shift are per-block constants, gamma a weight),
    # and ASSERTED rather than assumed: 96 x 23,040 B compare, ~1 ms/layer.
    # T22: the invariant is a property of make_sbg, not of the layer, so the
    # fast-glue path checks the first layer of each step by default
    # (T22_SBG_ASSERT=all restores the per-layer check, =off skips it).
    if ASSERT_SBG and (not T22_FAST_GLUE or T22_SBG_ASSERT == "all"
                       or (T22_SBG_ASSERT == "step" and key == "layers.0")):
        _rows = (np.ascontiguousarray(sbg_bytes).view(np.uint16)
                 if not FAST_HOST else sbg_bytes).reshape(SEQ, -1)
        if not bool((_rows == _rows[0]).all()):
            raise SystemExit(
                "[t4] REFUSAL: SBG rows differ -- the t4_vs FFN image assumes "
                "one [scale|shift|gamma] per block; rebuild/choose the C-only "
                "image (t4_vcopy2) for this payload")
    pmark("post.sbg", t0)
    gmark("blk.sbg", _t_sbg)
    t0 = time.perf_counter()
    _t_ffn = time.perf_counter()
    _g_ffn = gtotal()
    global _DUMP_DONE, _DUMP_FIRST
    _DUMP = os.environ.get("T22_DUMP_FFN", "")
    if _DUMP and key == "layers.0" and not _DUMP_DONE:
        _DUMP_DONE = True
        _DUMP_FIRST = True
        # t22_breadout: the product IS the ground truth.  Dump the real FFN
        # input (the residual x_mid, before either branch touches it) so the
        # delivered and big-M arms can be compared at the same call site
        # instead of through a re-implemented harness (which failed 3x).
        np.save(_DUMP + "_xmid.npy", np.asarray(xm, np.float32))
    if BIGM:
        # t22_prodasm: the FFN segment on device (gu+swiglu -> down).  The
        # contract is unchanged: y is (SEQ, D) float32, and everything after
        # this line (ffn_norm2, the residual) runs exactly as before.
        import sys as _sys, os as _os
        _bgm = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "block", "bigm")
        if _bgm not in _sys.path:
            _sys.path.insert(0, _bgm)
        import prod_loop_bigm as _PB
        # t22_prodasm: the delivered FFN image does ffn_norm1 itself (its SBG BO
        # carries [scale|shift|gamma]).  sw64d2 has NO SBG -- it is GEMM+swiglu
        # only -- so the norm must be applied on the host first, or the GEMM sees
        # the raw residual stream instead of the normalised value (~100x).
        # t3_real.py:303-306 is the oracle, verbatim:
        #     n2 = rms(x, Wb['ffn_norm1']);  return n2 * mod[2]
        # MULTIPLY ONLY -- there is no shift term here.  mod[3] is gate_mlp and
        # belongs to the residual (x + mod[3]*p), NOT to the norm input.
        _xn = rms_into(xm, P["ffn_norm1"], L.n1buf)
        if mod is not None:
            np.multiply(_xn, mod[2], out=_xn)
        # DIAGNOSTIC (DET-BISECT, default OFF): the FFN input EVERY arm sees, so
        # the plan arm's call-site hash and the campaign arm's can be compared
        # byte for byte.  T22_PLAN_FFN_PROBE=1 only; no other effect.
        if (os.environ.get("T22_PLAN_FFN_PROBE", "0") == "1"
                and key == "layers.0"):
            import hashlib as _hlp
            _ab = np.ascontiguousarray(_xn)
            _mod2 = (float(np.abs(np.asarray(mod[2])).max())
                     if mod is not None else -1.0)
            print("[pffnprobe] t18-xn key=%s dtype=%s absmax=%.6g sha=%s "
                  "mod2_absmax=%.6g xm_absmax=%.6g"
                  % (key, _ab.dtype, float(np.abs(_ab).max()),
                     _hlp.sha256(memoryview(_ab).cast("B")).hexdigest()[:16],
                     _mod2, float(np.abs(xm).max())), flush=True)
            _pd = os.environ.get("T22_PLAN_FFN_PROBE_DUMP", "")
            if _pd:
                np.save(_pd + "_xn.npy", np.asarray(_xn, np.float32))
                np.save(_pd + "_xm.npy", np.asarray(xm, np.float32))
                if mod is not None:
                    np.save(_pd + "_mod2.npy", np.asarray(mod[2], np.float32))
                print("[pffnprobe] t18-xn dumped to %s_{xn,xm,mod2}.npy" % _pd,
                      flush=True)
        if _DUMP and key == "layers.0" and _DUMP_FIRST:
            np.save(_DUMP + "_xn.npy", np.asarray(_xn, np.float32))
            np.save(_DUMP + "_mod.npy",
                    np.asarray(mod[2] if mod is not None else np.ones(3840), np.float32))
        # t22_breadout: DO NOT clobber xm.  The residual add after this branch
        # (x + mod[3]*rms(y, ffn_norm2)) must use the ORIGINAL residual stream --
        # the delivered branch L.ffn(xm, ...) never modifies xm either.  The
        # shipped line was "xm = _xn", which replaced the residual with the
        # normalised FFN input for every layer, so the whole 30-layer stack ran
        # on a corrupted stream: measured 128^2 latents corr 0.0990 with the FFN
        # itself at 0.999890 against the numpy oracle.
        # t22_480ab: with --cache-all the product ALREADY holds every layer
        # (3.98 GB).  load_payloads() keeps a SECOND copy of all 30 and the
        # 480p arm died with ArrayMemoryError (84.4 MiB, 153600x576) inside
        # apply_bg_map.  Repack the resident rows in place once (3.6 s for the
        # whole cache, validated by imaggers/block/ab480/repack_check.py) and
        # hand them out: 0 extra GB instead of 3.98 GB.
        if PLAN_FFN:
            # t22_adaptive: the plan's B payloads are in the PLAN's order
            # (pack_kk at the plan's kk), so neither the resident cache nor
            # load_payloads can serve them -- plan_ffn repacks them from the
            # model's own BFP16 blocks.  Skipping both here also skips the
            # 30-layer in-place repack the delivered cache would pay for.
            _c = _b = None
        elif CACHE_ALL and CACHE_BG is not None:
            _c, _b = _PB.cached_payloads(key, CACHE_BG, CACHE_B3, nh_bg=NH_BG)
        else:
            _c, _b = _PB.load_payloads(key)
        # the image pair must match the row-tile count: RT=1 for SEQ=96
        if _DUMP and _c is not None and key == "layers.0" and _DUMP_FIRST:
            import hashlib as _hl
            import pathlib as _pl
            _pl.Path(_DUMP + "_wsha.txt").write_text(
                _hl.sha256(bytes(_c)).hexdigest() + " " + _hl.sha256(bytes(_b)).hexdigest(),
                encoding="utf-8")
        _pair = ("swiglu1", "down1") if _xn.shape[0] == 96 else ("swiglu", "down")
        # DIAGNOSTIC (DET-BISECT, default OFF): hand the FFN branch a DIFFERENT
        # input without editing the math, so the image's own input contract can
        # be MEASURED instead of argued:
        #   T22_PLAN_FFN_PROBE_ALT=xm   -> the RAW residual (no ffn_norm1)
        #   T22_PLAN_FFN_PROBE_ALT=xn2  -> _xn scaled by 2 (a normalising image
        #                                  is scale-invariant; a pure GEMM is not)
        _alt = os.environ.get("T22_PLAN_FFN_PROBE_ALT", "")
        _xfin = _xn
        if _alt == "xm":
            _xfin = np.ascontiguousarray(xm, np.float32)
        elif _alt == "xn2":
            _xfin = (_xn * np.float32(2.0)).astype(np.float32)
        if _alt:
            print("[pffnprobe] ALT input=%s absmax=%.6g (xn absmax=%.6g)"
                  % (_alt, float(np.abs(_xfin).max()),
                     float(np.abs(_xn).max())), flush=True)
        if PLAN_FFN:
            # t22_adaptive: the plan's image set (gu gate, gu up, dn), one
            # dispatch set per Mblk rows.  Same contract as the branches below:
            # y is the FFN output BEFORE ffn_norm2.
            import plan_ffn as _PFF
            y = _PFF.ffn(_BIGM_DEV["d"], _xfin, key, tim)
        elif T22_S2_FFN32:
            # t22_util S2: 32-core FFN (4 dispatches: gu main/tail, dn main/tail).
            # Lazy import so the default path pays nothing; INTEGRATION-SPEC §5.
            import sys as _sys3
            _s2d = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "util", "place")
            if _s2d not in _sys3.path:
                _sys3.path.insert(0, _s2d)
            # s2_product owns the S2Dev singleton (lock, cache, slabs, atexit).
            import s2_product as _S2P
            y = _S2P.ffn32_run(_BIGM_DEV["d"], _xn, key, tim)
        elif os.environ.get("T22_DEVKILL", "0") == "1":
            # t22_devkill: RT=11 pair (no 6 dead row-tiles) + fast A pack.
            import sys as _sys2
            _dv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "devkill")
            if _dv not in _sys2.path:
                _sys2.path.insert(0, _dv)
            import devkill_ffn as _DVF
            if _xn.shape[0] == 96:
                y = _DVF.bigm_ffn(_BIGM_DEV["d"], _xfin, _c, _b, tim, imgs=("swiglu1", "down1"))
            else:
                y = _DVF.bigm_ffn(_BIGM_DEV["d"], _xfin, _c, _b, tim, imgs=("swiglu11", "dn11"))
        else:
            y = _PB.bigm_ffn(_BIGM_DEV["d"], _xfin, _c, _b, tim, imgs=_pair)
    else:
        y = L.ffn(xm, key, mod, tim, bg, b3, sbg_bytes)
    if _DUMP and key == "layers.0" and _DUMP_FIRST:
        np.save(_DUMP + "_y.npy", np.asarray(y, np.float32))
        _DUMP_FIRST = False
    # T22 GLUE FIX (2026-10-21 09:4x): L.ffn contains dev.dispatch(), whose
    # host.wait/fill/read are ALREADY leaf buckets.  A plain gmark here made
    # blk.ffn a PARENT and inflated the ledger by exactly that overlap
    # (measured -9.556 s on t22g_L).  Keep only the exclusive remainder.
    gadd("blk.ffn", max(0.0, (time.perf_counter() - _t_ffn)
                        - (gtotal() - _g_ffn)))
    _t_fp = time.perf_counter()
    if T22_FAST_GLUE:
        p = rms_into(y, P["ffn_norm2"], L.n3buf)
        out = np.empty((SEQ, D), np.float32)   # escapes to the next layer
        if mod is None:
            np.add(xm, p, out=out)
        else:
            np.multiply(p, mod[3], out=p)
            np.add(xm, p, out=out)
    else:
        p = T3.rms(y, P["ffn_norm2"])
        out = (xm + (p if mod is None else mod[3] * p)).astype(np.float32)
    pmark("post.ffnpost", t0)
    gmark("blk.ffnpost", _t_fp)
    tim["host"] += 1e3 * (time.perf_counter() - t0)
    _t_rec = time.perf_counter()
    if T22_GLUE3:
        rec = {"layer": key, "a_absmax": absmax_fast(a),
               "xm_absmax": absmax_fast(xm), "y_absmax": absmax_fast(y)}
    else:
        rec = {"layer": key, "a_absmax": float(np.abs(a).max()),
               "xm_absmax": float(np.abs(xm).max()),
               "y_absmax": float(np.abs(y).max())}
    gmark("blk.rec", _t_rec)
    if host_ref:
        t0 = time.perf_counter()
        Wb = L.RW.block(key)
        a_ref = T3.attention(bf16f32(x1), Wb, cos, sin)
        y_ref = T3.ffn_raw(T3.ffn_input(xm, Wb, mod), Wb)
        if os.environ.get("T18_DUMP") and key == "layers.0" and hasattr(L, "last_qm"):
            _qk = L.last_qm
            for nm, tail in (("q", "to_q"), ("k", "to_k"), ("v", "to_v")):
                ref = T3.lin(bf16f32(x1), Wb[tail]).reshape(SEQ, HEADS, HD)
                dev = _qk[nm].transpose(1, 0, 2)
                rows = []
                for r0 in range(0, SEQ, 32):
                    e_ = np.abs(dev[r0:r0 + 32] - ref[r0:r0 + 32])
                    rows.append(round(float(np.sqrt((e_ ** 2).mean()) /
                                            max(np.sqrt((ref[r0:r0 + 32] ** 2).mean()), 1e-9)), 4))
                print("      [qkv %s] rel by 32 rows %s" % (nm, rows), flush=True)
        e = np.abs(a - a_ref)
        # T18 probe: WHICH rows are wrong?  Projection blocks are 96 rows
        # ([0:96], [96:SEQ]); ip query blocks are IP_MQ rows.
        rb = []
        for r0 in range(0, SEQ, 16):
            seg = e[r0:r0 + 16]
            rb.append(round(float(np.sqrt((seg ** 2).mean())), 3))
        print("      [rowrms] %s = %s" % (key, rb), flush=True)
        rec["attn_row_rms16"] = rb
        rec["attn_dev_vs_host"] = {
            "max_abs": float(e.max()),
            "rel_to_absmax": float(e.max() / max(np.abs(a_ref).max(), 1e-30)),
            "rms": float(np.sqrt((e ** 2).mean())),
            "ref_absmax": float(np.abs(a_ref).max())}
        e2 = np.abs(y - y_ref)
        rec["ffn_dev_vs_host"] = {
            "max_abs": float(e2.max()),
            "rel_to_absmax": float(e2.max() / max(np.abs(y_ref).max(), 1e-30)),
            "rms": float(np.sqrt((e2 ** 2).mean()))}
        del Wb, a_ref, y_ref
        rec["host_ref_s"] = time.perf_counter() - t0
    if GLUE is not None:
        gadd("blk.other", max(0.0, (time.perf_counter() - _g_t)
                              - (gtotal() - _g_in)))
    return out, rec


# ---------------------------------------------------------------------------
# T21 2b-b: payload pre-swap on load (the read+permute moves off the critical path)
# ---------------------------------------------------------------------------
_PAYLOAD_PRE = {"th": None, "err": None, "s": None, "wait_s": None}

# t22_brepack: the content-addressed B-repack cache (brepack_cache.py).  None
# unless T22_BREPACK_CACHE=1; every use of it is guarded by that None.
_BRC = None


def _brepack_prefetch(args):
    """Start the B-repack cache's map+verify on a worker thread, or None.

    The guard is the same one the consumer site uses (--cache-all without
    --bo-resident, which is the default), and it is off unless
    T22_BREPACK_CACHE=1.  Any failure returns None and the shipped path runs.
    """
    if os.environ.get("T22_BREPACK_CACHE", "0") != "1":
        return None
    if not (getattr(args, "cache_all", False)
            and not getattr(args, "bo_resident", False)):
        return None
    try:
        _bm = pathlib.Path(__file__).resolve().parent.parent / "block" / "bigm"
        if str(_bm) not in sys.path:
            sys.path.insert(0, str(_bm))
        import brepack_cache as _BRCmod
        _BRCmod.start(PREP, int(args.layers), NH_BG)
        return _BRCmod
    except Exception as _e:                                     # noqa: BLE001
        print("[brepack] cache unavailable (%r)" % (_e,), flush=True)
        return None


def _preload_payload(args, at="start"):
    """Put the 3.71 GB payload into RAM and permute it, "as it lands".

    T13 moved the per-step np.load + NH permutation into a one-off startup pass
    (--cache-all).  That pass used to read each .npy into an ARRAY, copy it to
    bytes (.tobytes()) and copy it AGAIN into the permuted destination -- three
    passes over 3.71 GB.  Here the permutation reads the loaded array directly,
    so only read+write remain (measured 3.6/2.7 s -> 2.1/2.2 s on this host).

    T18_PAYLOAD_ORDER picks where the pass runs:
      "overlap" (default) start a worker thread before --warm packs the 30
                projection images; the cache-all site joins it.  The device is
                not involved, and the arrays are the same bytes in the same
                order, so latents cannot change.
      "first"   run it inline BEFORE the W pack (payload is in anonymous RAM
                before the 8 GB weight read can evict its .npy pages).
      "last"    the T13 order (inline at the cache-all site) -- the A arm.
    """
    if not (args.cache_all and not args.bo_resident):
        return None
    if BG_PREPERM and CACHE_BG is not None and CACHE_B3 is not None:
        return None            # already resident (resident-worker reuse)
    # default "last": the sequential fused pass.  The "overlap" mode is kept
    # as an A/B knob but is a measured LOSS on this host (see
    # webui/evidence/T21-2A2B2-WARM.md: the extra 3.71 GB resident during the
    # W pack pushes the working set past the commit headroom and step 0 pays
    # ~15 s of paging).
    mode = os.environ.get("T18_PAYLOAD_ORDER", "last")

    def work():
        global CACHE_BG, CACHE_B3, BG_PREPERM
        t0 = time.perf_counter()
        try:
            cache_bg = np.empty((args.layers, BG_BYTES), np.uint8)
            cache_b3 = np.empty((args.layers, B3_BYTES), np.uint8)
            for li in range(args.layers):
                key = "layers.%d" % li
                # No .tobytes(): the permutation reads the loaded uint8 array
                # itself (same bytes, two fewer full-size copies).
                src = np.load(PREP / (key + "_bg.npy"))
                if not src.flags.c_contiguous:
                    src = np.ascontiguousarray(src)
                if NH_BG > 1:
                    (_P11.permute_bg if NH_BG_CORE else _P9.permute_bg)(
                        src.reshape(-1), cache_bg[li], NH_BG)
                else:
                    np.copyto(cache_bg[li], src.reshape(-1))
                b3 = np.load(PREP / (key + "_b3.npy"))
                np.copyto(cache_b3[li], b3.reshape(-1))
                del src, b3
            CACHE_BG, CACHE_B3 = cache_bg, cache_b3
            BG_PREPERM = True
        except BaseException as exc:                          # noqa: BLE001
            _PAYLOAD_PRE["err"] = exc
        finally:
            _PAYLOAD_PRE["s"] = time.perf_counter() - t0

    if at == "cache":
        work()                 # "last" (or a retry): do it here, synchronously
        return None
    if mode == "last":
        return None            # the cache-all site runs it after --warm
    if mode == "first":
        work()
        return None
    th = threading.Thread(target=work, name="t18-payload-preload", daemon=True)
    _PAYLOAD_PRE["th"] = th
    th.start()
    return th


def _preovl_preamble(T3, L, H, B, x_cos, x_sin, c_cos, c_sin, window=True):
    """The step preamble, flag-aware: {"pre", "submit", "report", "state"}.

    pre(step_t, lat_) -> (adaln, [xb per image], cb) is the delivered step_once
    preamble statement for statement.  T22_PREOVL touches ONE thing in it: the
    context half (cap_embedding + context_refiner.0/1) is a function of H alone
    -- see the flag's note at the top of the file -- and the delivered path
    recomputes it, weights included, inside every step's preamble.  With the
    flag on:

      * the two context blocks' 1.3 GB refiner-cache read is submitted to a
        worker thread before the loop and joined at step 0's preamble, so it
        runs under the --cache-all startup pass instead of on the loop's wall;
      * the buffer that read feeds is MEMOISED: steps 1..7 read it instead of
        running the two blocks again.

    WHY THE WORKER LOADS WEIGHTS AND DOES NOT COMPUTE: the bridge's
    "startup_done" anchor -- and therefore the product's whole loop_s bucket --
    is emitted by its cap_embedding() monkeypatch on the FIRST call.  Computing
    the context half on the worker calls cap_embedding ~2.5 s early, which
    re-defines loop_s (measured: +23.5 s on the arm that did it, for a real
    -1.9 s of loop).  So cap_embedding stays exactly where the delivered path
    calls it, the arithmetic stays on the main thread, and only the I/O moves.

    window: True when run() is about to do the --cache-all startup pass, i.e.
    when there is a window for the worker to hide in.  False = do not submit
    the preload (low memory mode has no such pass): the context half is then
    computed at step 0 exactly as delivered and MEMOISED anyway, so the flag
    degrades to the half of itself that needs no thread.
    """
    S = {"pool": None, "fut": None, "cb": None, "ok": False, "why": "",
         "preloaded": False, "t_work": 0.0, "t_wait": 0.0, "hits": 0,
         "window": bool(window)}

    def _ctx_step(adaln):
        """THE DELIVERED STATEMENTS, verbatim (the off path and the fallback)."""
        _t = time.perf_counter()
        cb = T3.cap_embedding(H)
        gmark("pre.capemb", _t)
        for key in T3.REFINERS[2:]:
            _t = time.perf_counter()
            Wb = L.refblock(key)
            gmark("pre.refblock2", _t)
            _t = time.perf_counter()
            cb = T3.full_block_host(cb, Wb, T3.modulation(adaln, Wb),
                                    c_cos, c_sin)
            gmark("pre.refiner2", _t)
        return cb

    def _ctx_weights():
        """The two context blocks' refiner-cache read, off the loop's wall.

        Pure I/O -- np.load of 2 x 0.66 GB, no BLAS, no numba, no GEMM -- so
        the parallel region this adds is ONE thread reading page-cached files:
        the shape CONTENTION-REPORT measured at +0.0 % device cost for an
        88.5 MB bo.write, not the 18-thread attention pass that costs +15.3 %.

        The adaLN precondition is checked on the weights themselves: if either
        context block ever grows an adaLN the optimisation is not valid, and
        the flag must fall back rather than memoise a buffer that can move.
        """
        out = []
        for key in T3.REFINERS[2:]:
            Wb = L.refblock(key)
            if Wb["adaln_w"] is not None:
                raise RuntimeError(
                    "preovl: %s has an adaLN -- the context half is not "
                    "step-invariant" % key)
            out.append(Wb)
        return out

    def submit():
        """Start the weight preload on a worker thread; False (fail open)."""
        if not T22_PREOVL:
            return False
        if not S["window"]:
            # the window this thread hides in IS the --cache-all startup pass
            # (3.71 GB, ~2.3 s).  In low memory mode there is no such pass, the
            # join below would simply block, and the flag would be a no-op with
            # more moving parts -- so it refuses instead.
            print("[preovl] preload not submitted: no --cache-all startup "
                  "window (low memory mode).  The context half is computed at "
                  "step 0 as delivered and MEMOISED for steps 1..n.", flush=True)
            return False
        if REFCACHE is None or not all(
                (pathlib.Path(REFCACHE) / k / "manifest.json").exists()
                for k in T3.REFINERS[2:]):
            # The worker must never touch RW: Real.raw() is a seek+read on ONE
            # file handle, so two threads on it would race.  With the refiner
            # cache present L.refblock() is an np.load and nothing else.
            print("[preovl] not submitted: no refiner cache for the context "
                  "blocks; the delivered per-step path runs", flush=True)
            return False
        try:
            pool = S["pool"]
            if pool is None:
                pool = S["pool"] = _fut.ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="t22-preovl")
            S["t_work"] = 0.0

            def work():
                t0 = time.perf_counter()
                w = _ctx_weights()
                S["t_work"] = time.perf_counter() - t0
                return w

            S["fut"] = pool.submit(work)
            S["ok"] = True
            S["preloaded"] = True
            print("[preovl] context-half weights submitted on a worker thread "
                  "(2 x 0.66 GB, under the --cache-all pass)", flush=True)
            return True
        except BaseException as exc:                          # noqa: BLE001
            S["why"] = "%r" % (exc,)
            print("[preovl] submit failed (%r); the delivered per-step path "
                  "runs" % (exc,), flush=True)
            return False

    def _ctx_join():
        """Join the preload once (idempotent; a no-op when not submitted)."""
        fut, S["fut"] = S["fut"], None
        if fut is None:
            return
        t0 = time.perf_counter()
        try:
            fut.result()
        except BaseException as exc:                          # noqa: BLE001
            S["ok"] = False
            S["why"] = "%s: %s" % (type(exc).__name__, exc)
            print("[preovl] preload failed (%s); the delivered per-step path "
                  "runs from here" % (S["why"],), flush=True)
            return
        S["t_wait"] = time.perf_counter() - t0

    def pre(step_t, lat_):
        """(adaln, [xb per image], cb): B=1 reproduces the delivered order."""
        _t = time.perf_counter()
        adaln = T3.t_emb(step_t, H)
        gmark("pre.temb", _t)
        xbs = []
        for b in range(B):
            _t = time.perf_counter()
            xb = T3.x_embedding(lat_[b], H)
            gmark("pre.xemb", _t)
            for key in T3.REFINERS[:2]:
                # split the WEIGHT LOAD from the f32 forward: refblock() is the
                # 2.83 GB refiner cache read on step 0 (one-off), full_block_host
                # is the per-step arithmetic.
                _t = time.perf_counter()
                Wb = L.refblock(key)
                gmark("pre.refblock", _t)
                _t = time.perf_counter()
                xb = T3.full_block_host(xb, Wb, T3.modulation(adaln, Wb),
                                        x_cos, x_sin)
                gmark("pre.refiner", _t)
            xbs.append(xb)
        if T22_PREOVL:
            # the weights are (or will be, after one wait) resident: the
            # delivered _ctx_step below then pays only the arithmetic, once
            _t = time.perf_counter()
            _ctx_join()
            gmark("pre.ctxjoin", _t)
            if S["cb"] is not None:
                _t = time.perf_counter()
                cb = S["cb"]
                S["hits"] += 1
                gmark("pre.ctxreuse", _t)
            else:
                cb = _ctx_step(adaln)
                S["cb"] = cb
        else:
            cb = _ctx_step(adaln)
        return adaln, xbs, cb

    def report():
        """One line at exit: what the flag actually did."""
        if not T22_PREOVL:
            return
        if S["cb"] is None:
            print("[preovl] context half: NOT USED (%s)"
                  % (S["why"] or "no window / not submitted"), flush=True)
            return
        how = ("weights preloaded in %.3f s on the worker (joined in %.3f s)"
               % (S["t_work"], S["t_wait"])) if S["preloaded"] else (
                   "preload not submitted (%s)" % (S["why"] or "no window"))
        print("[preovl] context half: %s; 1 computation + %d reuse(s)"
              % (how, S["hits"]), flush=True)

    return {"pre": pre, "submit": submit, "report": report, "state": S}



def run(args) -> int:
    import t3_real as T3
    global _BIGM_DEV
    # T22 glue: any harness that turns PROF on (t22_prof.py sets D.PROF = {})
    # gets the exclusive ledger as well -- no second switch to remember.
    global GLUE
    if PROF is not None and GLUE is None:
        GLUE = {}
    dev = Dev(args.tag, dimk64=args.dimk64, ffn_image=args.ffn_image,
              att_image=args.att_image, att_dir=args.att_dir,
              att_ip_image=args.att_ip_image)
    _BIGM_DEV["d"] = dev
    # t22_brepack: resolve + mmap + verify the repacked BG/B3 payloads on a
    # worker thread, here, so the (measured 1.3 s) verification hides under the
    # device open and the --warm pack that follow it.  The 19.91 s in-place
    # repack it replaces used to run at the very end of startup
    # (HOSTPROF-REPORT.md 8; "glue.cached_payloads", first call).  Off unless
    # T22_BREPACK_CACHE=1; nothing here changes a byte the device sees.
    global _BRC
    _BRC = _brepack_prefetch(args)
    # T21 2b-b: the payload read+permute starts here, while --warm packs the
    # projection images; the cache-all site below only joins it.
    _preload_payload(args)
    print("[t3d] registered %d images: %s" % (len(dev.imgs), "  ".join(
        "%s %.2fs" % (n, v["reg_s"]) for n, v in dev.imgs.items())), flush=True)
    print("[t3d] BOs: %s  ffn(A,SBG,BG,B3,X)->Y"
          % ("att(x,w)->Q  attip(qp,kb,vb)->O" if dev.has_ip
             else "att(x,w,qp,kb,vb)->Q,O"), flush=True)
    for n in (("att", "attip", "ffn") if dev.has_ip else ("att", "ffn")):
        k = dev.imgs[n]["k"]
        print("[t3d]   %-4s group ids %s" % (n, [k.group_id(i)
                                                 for i in range(4)]), flush=True)
    L = Loop(dev, args.tag, dimk64=args.dimk64, chain=CHAIN_ID)
    H = T3.host_tensors(args.tag)
    RW = L.RW
    timesteps, sigmas = T3.load_schedule(args)
    # T10: one (1,16,16,16) latent PER IMAGE (seed+b).  Kept as a list of the
    # delivered shape so every downstream op sees exactly the tensor it saw in
    # the batch=1 run -- the batch never concatenates the images' activations.
    B = int(args.batch)
    lat = [np.random.default_rng(args.seed + b).standard_normal(
        (1, 16, GH * 2, GW * 2)).astype(np.float32) for b in range(B)]
    lat0 = [x.copy() for x in lat]

    ck_lat = HERE / ("%s_ckpt_lat.npy" % args.tag)
    ck_js = HERE / ("%s_ckpt.json" % args.tag)
    calls, steps_rec = [], []
    start = 0
    if args.resume and ck_lat.exists() and ck_js.exists():
        ck = json.loads(ck_js.read_text())
        if (ck.get("steps") == args.steps and ck.get("layers") == args.layers
                and int(ck.get("batch", 1)) == B):
            lat = [np.asarray(z, np.float32) for z in np.load(ck_lat)]
            lat0 = [np.asarray(z, np.float32).reshape(lat[b].shape)
                    for b, z in enumerate(ck["lat0"])]
            calls, steps_rec = ck["calls"], ck["steps_rec"]
            start = int(ck["next_step"])
            print("[t3d] RESUME at step %d" % start, flush=True)

    if args.warm:
        t0 = time.perf_counter()
        for li in range(args.layers):
            L.w_images("layers.%d" % li)
            L.params("layers.%d" % li)
        print("[t3d] warm: %d layers' projection images packed in %.1f s "
              "(startup, outside the loop's wall)"
              % (args.layers, time.perf_counter() - t0), flush=True)

    # ---- the host-only preamble of step_once, verbatim -------------------
    cap_ids, img_ids = T3.real_ids()
    c_cos, c_sin = T3.rope_cos_sin(cap_ids)
    x_cos, x_sin = T3.rope_cos_sin(img_ids)

    # ---- T22_PREOVL: the preamble, context half included ---------------
    # _preovl_preamble() is the delivered step_once preamble, verbatim, with
    # the context half routed through the PREOVL machinery when the flag is
    # on (see the flag's note at the top of this file).  It is a factory so
    # that imaggers/preovl/gate_preovl.py can drive THE SAME statements
    # twice in one process -- flag off and flag on -- without a device.
    PRE = _preovl_preamble(T3, L, H, B, x_cos, x_sin, c_cos, c_sin,
                           window=bool(args.cache_all and not args.bo_resident))
    pre = PRE['pre']
    # Start the context half HERE, so it runs under the --cache-all pass
    # below (3.71 GB, ~2.2 s on this box) and step 0's preamble joins a
    # finished future instead of computing it.  Off by default: this is a
    # flag test that returns False.
    PRE['submit']()

    # ---- T4-D: the per-layer weight payload (132.7 MB on disk) -----------
    # MEASURED (t4_hostprof.py, py3.12, no device): reading layer L's BG (88.5 MB)
    # + B3 (44.2 MB) costs 1.59-3.82 s per STEP (30 layers), i.e. 13-30 s of the
    # 60 s host bucket -- the per-layer refiner f32 forward that the brief
    # suspected is only 0.26 s/STEP.  Both fixes here move the SAME bytes into
    # the SAME BOs, so they are bit-identical by construction:
    #   --prefetch       read layer i+1 on a worker thread during layer i's
    #                    dispatch (zero extra RAM; the 269 ms dispatch is a
    #                    wide enough window for a 53 ms read)
    #   --cache-weights  read all 30 layers once before the loop (3.98 GB RAM;
    #                    only if the machine has the headroom)
    _pool = (_fut.ThreadPoolExecutor(max_workers=1)
             if (args.prefetch and not args.cache_all
                 and not args.bo_resident) else None)
    _wcache = {}

    def read_payload(key):
        return (np.load(PREP / ("%s_bg.npy" % key)).tobytes(),
                np.load(PREP / ("%s_b3.npy" % key)).tobytes())

    if args.cache_weights:
        t0 = time.perf_counter()
        nb = 0
        for li in range(args.layers):
            key = "layers.%d" % li
            _wcache[key] = read_payload(key)
            nb += len(_wcache[key][0]) + len(_wcache[key][1])
        print("[t3d] weight payloads cached: %d layers, %.2f GB in %.1f s "
              "(startup, outside the loop's wall)"
              % (args.layers, nb / 2**30, time.perf_counter() - t0), flush=True)

    if PLAN_FFN and args.cache_all and not args.bo_resident:
        # t22_adaptive: the plan arm's FFN never reads the delivered B payload
        # (block_step's PLAN_FFN branch sets _c = _b = None -- the plan's B is a
        # DIFFERENT permutation, at its own kk), so the 3.71 GB resident cache
        # below and the 5.31 GB in-place B-order repack after it are pure
        # startup cost here.  Measured on job 20261103-165433-6ab50a: 3.2-4.4 s
        # for the read + 14.6-26.1 s for the repack.  Skipping them also frees
        # the 3.71 GB that lets the PLAN's own payload cache stay resident
        # (plan_ffn.prewarm, 3.98 GB for 30 layers) -- which is what removes the
        # 3.98 GB/step of npz re-reads that dominated the plan arm's step wall.
        print("[t13] cache-all SKIPPED for the plan arm: the delivered B payload "
              "is never read by T22_PLAN_FFN (block_step sets _c = _b = None), "
              "so neither the 3.71 GB resident cache nor the in-place B-order "
              "repack is done; that memory holds the plan's own payloads "
              "instead (plan_ffn.prewarm)", flush=True)
        if os.environ.get("T22_PLAN_PREWARM", "1") != "0":
            try:
                import plan_ffn as _PFF_pw
                _PFF_pw.prewarm(["layers.%d" % _i
                                 for _i in range(args.layers)])
            except Exception as _e_pw:                        # noqa: BLE001
                print("[planffn] prewarm skipped: %r -- the loop repacks each "
                      "layer lazily" % (_e_pw,), flush=True)

    if args.cache_all and not args.bo_resident and not PLAN_FFN:
        # T13 U-4: ONE resident pass, allocated after the W-image array and the
        # refiner cache (the device is already open; measured to fit: 5.1-7.9 GB
        # free after the allocation).  Fill sequentially, hand the loop
        # references.  BG is permuted ONCE here; ffn() sees BG_PREPERM and
        # skips its per-step permutation.
        #
        # T21 2b-b: the pass was already started in run() (before --warm), so
        # this site normally only JOINS it.  Identical arrays, identical bytes,
        # identical layer order -- the wall simply overlaps the W pack.
        global CACHE_BG, CACHE_B3, BG_PREPERM
        t0 = time.perf_counter()
        th = _PAYLOAD_PRE["th"]
        if _BRC is not None and th is None:
            # t22_brepack: the mapped rows are ALREADY in the big-M image order,
            # which is exactly the state the in-place repack leaves the resident
            # cache in -- so the 5.8 s read+permute below is redundant too and
            # 3.71 GB of anonymous RAM is never allocated.  BG_PREPERM is set
            # only because that is the flag t18_run's own bookkeeping reads.
            _hit = _BRC.join()
            if _hit is not None:
                CACHE_BG, CACHE_B3 = _hit["bg"], _hit["b3"]
                BG_PREPERM = True
                _BRC.arm(_hit)
                print("[t13] cache-all: B payloads mapped from %s (%.2f GB, "
                      "%.1f s); the 3.71 GB read+permute is skipped"
                      % (_hit["name"], _hit["bytes"] / 2**30,
                         _hit["ident_s"] + _hit["header_s"]
                         + _hit["verify_s"]), flush=True)
        if BG_PREPERM and CACHE_BG is not None and CACHE_B3 is not None:
            # T21 2b-b "first" mode or a resident worker: the payload is
            # already in RAM, permuted.  Nothing to do here.
            print("[t13] cache-all: payload already resident (%d layers, "
                  "%.2f GB); 0.0 s at this site"
                  % (args.layers, (CACHE_BG.nbytes + CACHE_B3.nbytes) / 2**30),
                  flush=True)
        elif th is not None:
            print("[t13] cache-all: payload preload is on a worker thread "
                  "(%.2f GB free before join)" % _mem_gb(), flush=True)
            th.join()
            if _PAYLOAD_PRE["err"] is not None:
                raise _PAYLOAD_PRE["err"]
            _PAYLOAD_PRE["wait_s"] = time.perf_counter() - t0
            nb = CACHE_BG.nbytes + CACHE_B3.nbytes
            print("[t13] full-memory cache: %d layers, %.2f GB read+permuted on "
                  "a worker thread in %.1f s; %.1f s on the critical path "
                  "(startup, outside the loop's wall); %.2f GB free after"
                  % (args.layers, nb / 2**30, _PAYLOAD_PRE["s"],
                     _PAYLOAD_PRE["wait_s"], _mem_gb()), flush=True)
        else:
            print("[t13] cache-all: %.2f GB free before allocation"
                  % _mem_gb(), flush=True)
            _preload_payload(args, at="cache")   # fused read+permute
            if _PAYLOAD_PRE["err"] is not None:
                raise _PAYLOAD_PRE["err"]
            nb = CACHE_BG.nbytes + CACHE_B3.nbytes
            print("[t13] full-memory cache: %d layers, %.2f GB in %.1f s "
                  "(startup, outside the loop's wall); %.2f GB free after"
                  % (args.layers, nb / 2**30, time.perf_counter() - t0,
                     _mem_gb()), flush=True)

    # t22_devkill: the big-M B repack used to run on the FIRST FFN call inside
    # step 0 -- 30 layers x (88 MB BG + 44 MB B3) plus the second down
    # generation, ~5.2 GB of conversion.  Measured on job
    # 20261031-190252-1edc9b: blk.ffn = 25.08 s in a one-step run and ~20 s of
    # that is this repack.  SAME bytes, SAME order, only the instant changes,
    # so the loop latents are unchanged; it is weight preprocessing and belongs
    # with [t13] cache-all / [t22] wpack (both already startup, outside the
    # loop's wall).  T22_DEVKILL_BREPACK=0 restores the lazy arm.
    if (os.environ.get("T22_DEVKILL", "0") == "1" and CACHE_BG is not None
            and os.environ.get("T22_DEVKILL_BREPACK", "1") != "0"):
        _t_rp = time.perf_counter()
        _bm_rp = pathlib.Path(__file__).resolve().parent.parent / "block" / "bigm"
        if str(_bm_rp) not in sys.path:
            sys.path.insert(0, str(_bm_rp))
        import prod_loop_bigm as _PBRP
        _PBRP.cached_payloads("layers.0", CACHE_BG, CACHE_B3, nh_bg=NH_BG)
        print("[devkill] B repack at startup: %.2f s (outside the loop's wall)"
              % (time.perf_counter() - _t_rp), flush=True)

    if args.bo_resident:
        # T13 U-4b: one BO per (layer, BG/B3), written ONCE.  The loop selects
        # the layer's BOs and writes only A/SBG/X -- staging drops from 49.4 GB
        # to ~17.3 GB per image.  No host cache is kept (the same 3.98 GB lives
        # in BOs), so memory is not doubled versus --cache-all.
        global BO_RESIDENT, BG_BOS, B3_BOS
        t0 = time.perf_counter()
        kf = dev.imgs["ffn"]["k"]
        print("[t13] BO-resident: %d BOs, %.2f GB, %.2f GB free before"
              % (2 * args.layers,
                 (BG_BYTES + B3_BYTES) * args.layers / 2**30, _mem_gb()),
              flush=True)
        BG_BOS = [dev.pyxrt.bo(dev.dev, BG_BYTES,
                               dev.pyxrt.bo.flags.host_only, kf.group_id(5))
                  for _ in range(args.layers)]
        B3_BOS = [dev.pyxrt.bo(dev.dev, B3_BYTES,
                               dev.pyxrt.bo.flags.host_only, kf.group_id(6))
                  for _ in range(args.layers)]
        scratch_bg = np.empty(BG_BYTES, np.uint8)
        for li in range(args.layers):
            key = "layers.%d" % li
            bg = np.load(PREP / (key + "_bg.npy")).tobytes()
            b3 = np.load(PREP / (key + "_b3.npy")).tobytes()
            src = np.frombuffer(bg, np.uint8)
            if NH_BG > 1:
                (_P11.permute_bg if NH_BG_CORE else _P9.permute_bg)(
                    src, scratch_bg, NH_BG)
            else:
                scratch_bg[:] = src
            BG_BOS[li].write(scratch_bg, 0)
            BG_BOS[li].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            B3_BOS[li].write(b3, 0)
            B3_BOS[li].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            del bg, b3, src
        dev.bo["f_BG"] = BG_BOS[0]
        dev.bo["f_B3"] = B3_BOS[0]
        BO_RESIDENT = True
        print("[t13] BO-resident: %d layers written in %.1f s (startup, "
              "outside the loop's wall); %.2f GB free after"
              % (args.layers, time.perf_counter() - t0, _mem_gb()), flush=True)
    if args.prefetch:
        print("[t3d] weight payload prefetch: layer i+1 is read on a worker "
              "thread during layer i's dispatch", flush=True)

    def one_step(i, t_step, lat_, host_ref):
        """B images through one layer loop: one payload read, one SBG, B blocks."""
        t_step0 = time.perf_counter()
        _g_in = gtotal()
        tims = [{"dev": 0.0, "write": 0.0, "pack": 0.0, "host": 0.0}
                for _ in range(B)]
        tpre = time.perf_counter()
        adaln, xbs, cb = pre(t_step, lat_)
        _t_us = time.perf_counter()
        us = [np.concatenate([xbs[b], cb], axis=0).astype(np.float32)
              for b in range(B)]
        u_cos = np.concatenate([x_cos, c_cos], axis=0)
        u_sin = np.concatenate([x_sin, c_sin], axis=0)
        gmark("step.us", _t_us)
        tims[0]["host"] += 1e3 * (time.perf_counter() - tpre)
        rec = {"t": float(t_step), "layers": []}
        if _pool is not None:
            _pending = _pool.submit(read_payload, "layers.0")
        for li in range(args.layers):
            key = "layers.%d" % li
            P = L.params(key)
            tp = time.perf_counter()
            mod = T3.modulation(adaln, P)
            pmark("loop.mod", tp)
            gmark("loop.mod", tp)
            tp = time.perf_counter()
            if BO_RESIDENT:
                dev.bo["f_BG"] = BG_BOS[li]
                dev.bo["f_B3"] = B3_BOS[li]
                bg, b3 = None, None
            elif PLAN_FFN:
                # t22_adaptive: the plan's B payloads come from plan_ffn, never
                # from the delivered cache -- and with the cache-all pass
                # skipped above there is nothing here to read (a disk read of
                # 3.71 GB/step would be the alternative).
                bg, b3 = None, None
            elif CACHE_ALL and CACHE_BG is not None:
                bg, b3 = CACHE_BG[li], CACHE_B3[li]
            elif key in _wcache:
                bg, b3 = _wcache[key]
            elif _pending is not None:
                bg, b3 = _pending.result()
                nxt = li + 1
                _pending = (_pool.submit(read_payload, "layers.%d" % nxt)
                            if nxt < args.layers else None)
            else:
                bg, b3 = read_payload(key)
            pmark("loop.payload", tp)
            gmark("loop.payload", tp)
            tp = time.perf_counter()
            # T10: ONE SBG for the whole batch (per (step, layer), not per image)
            sbg_b = make_sbg(P, mod)
            pmark("loop.sbg", tp)
            gmark("loop.sbg", tp)
            if T22_PFILL and CACHE_ALL and not BO_RESIDENT:
                _pfill_ready(dev, li, args.layers)
            first = []
            for b in range(B):
                t0 = time.perf_counter()
                us[b], r = block_step(L, us[b], key, mod, u_cos, u_sin,
                                      tims[b], bg, b3,
                                      host_ref=(host_ref and b == 0),
                                      sbg_bytes=sbg_b)
                tims[b]["host"] += 1e3 * (time.perf_counter() - t0)
                rec["layers"].append(r)
                if li == 0:
                    first.append(dict(tims[b]))
                _t_pr = time.perf_counter()
                print("      %-10s img %d/%d  a %.3f xm %.3f y %.1f%s"
                      % (key, b + 1, B, r["a_absmax"], r["xm_absmax"],
                         r["y_absmax"],
                         ("  attn rel %.2e ffn rel %.2e"
                          % (r["attn_dev_vs_host"]["rel_to_absmax"],
                             r["ffn_dev_vs_host"]["rel_to_absmax"]))
                         if (host_ref and b == 0) else ""), flush=True)
                gmark("step.print", _t_pr)
            del bg, b3
            if li == 0:
                for b in range(B):
                    print("      [tim] img %d/%d first layer: dev %.0f ms  "
                          "write %.0f ms  pack %.0f ms  host %.0f ms"
                          % (b + 1, B, first[b]["dev"], first[b]["write"],
                             first[b]["pack"], first[b]["host"]), flush=True)
        tfin = time.perf_counter()
        preds = []
        for b in range(B):
            out = T3.final_layer(us[b], adaln, H)
            preds.append(T3.unpatchify(out[:REAL_TOKENS], 16, GH, GW))
        tims[0]["host"] += 1e3 * (time.perf_counter() - tfin)
        gmark("step.final", tfin)
        rec["pred_absmax"] = float(max(np.abs(p).max() for p in preds))
        rec["pred_absmax_by_image"] = [float(np.abs(p).max()) for p in preds]
        rec["tim"] = {k: sum(t[k] for t in tims) for k in tims[0]}
        rec["tim_by_image"] = [dict(t) for t in tims]
        rec["wall_s"] = time.perf_counter() - t_step0
        rec["wall_s_per_image"] = rec["wall_s"] / B
        if GLUE is not None:
            gadd("step.other", max(0.0, rec["wall_s"] - (gtotal() - _g_in)))
        return preds, rec

    # T23 SKIP7: the skip set is a function of the SCHEDULE, so it is decided
    # here, before any latent exists, and never consulted again.  Requested-but-
    # unusable is a hard error, never a silent fallback to the base path.
    _skip = None
    _held, _held_at = None, None
    _skip_spec = os.environ.get("ZIMAGE_SKIP_STEPS", "").strip()
    if _skip_spec not in ("", "0", "off"):
        import sys as _sys
        _sp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "par", "skip7")
        if _sp not in _sys.path:
            _sys.path.insert(0, _sp)
        try:
            import stepskip as _SS
        except Exception as _e:                                # noqa: BLE001
            raise SystemExit("[skip] ZIMAGE_SKIP_STEPS=%r but stepskip is not "
                             "importable (%r); refusing to run the base path"
                             % (_skip_spec, _e))
        _skip = _SS.resolve(_skip_spec, sigmas, len(timesteps))
        if not len(_skip):
            raise SystemExit("[skip] ZIMAGE_SKIP_STEPS=%r resolved to an EMPTY "
                             "pattern; refusing to run the base path silently"
                             % _skip_spec)
        print("[skip] %s -> pattern %s (policy %s): %s"
              % (_skip.mode, list(_skip.pattern), _skip.policy,
                 (_skip.assertion or {}).get("why", "")), flush=True)

    for i, t in enumerate(timesteps):
        if i < start:
            continue
        if _skip is not None and i in _skip and _held is not None:
            preds = _held
            # tim["x"] is read unconditionally by the per-step print and by the
            # job record, so a skipped step must carry the SAME KEYS with zero
            # values -- it does no device, write, pack or host work at all.
            import collections as _col
            rec = {"step": i, "skipped": True, "held_from_step": _held_at,
                   "pred_absmax": float(max(np.abs(p).max() for p in preds)),
                   "pred_absmax_by_image": [float(np.abs(p).max())
                                            for p in preds],
                   "tim": _col.defaultdict(float),
                   "tim_by_image": [_col.defaultdict(float) for _ in preds],
                   "wall_s": 0.0, "wall_s_per_image": 0.0}
        else:
            preds, rec = one_step(i, t, lat, args.host_ref and i == 0)
            _held, _held_at = preds, i
            rec["skipped"] = False
            rec["held_from_step"] = None
        _t = time.perf_counter()
        d = np.float32(sigmas[i + 1] - sigmas[i])
        for b in range(B):
            lat[b] = (lat[b] + d * (-preds[b])).astype(np.float32)
        gmark("step.lat", _t)
        steps_rec.append({"step": i, "t": float(t),
                          "pred_absmax": rec["pred_absmax"],
                          "pred_absmax_by_image": rec["pred_absmax_by_image"],
                          "lat_absmax_by_image": [float(np.abs(x).max())
                                                  for x in lat],
                          "lat_absmax": float(max(np.abs(x).max()
                                                  for x in lat)),
                          "seconds": rec["wall_s"],
                          "seconds_per_image": rec["wall_s_per_image"],
                          "batch": B,
                          "tim": rec["tim"]})
        calls.append(rec)
        _t = time.perf_counter()
        np.save(ck_lat, lat[0] if B == 1 else np.stack(lat))
        ck_js.write_text(json.dumps(
            {"next_step": i + 1, "steps": args.steps, "layers": args.layers,
             "batch": B,
             "lat0": [np.asarray(x, np.float32).ravel().tolist() for x in lat0],
             "calls": calls, "steps_rec": steps_rec}), encoding="utf-8")
        gmark("step.ckpt", _t)
        print("  step %d: t=%8.3f  pred absmax %8.4f  lat absmax %9.4f  "
              "%.1f s (%.2f s/image)  "
              "(dev %.1f / write %.1f / pack %.1f / host %.1f ms)"
              % (i, t, rec["pred_absmax"],
                 float(max(np.abs(x).max() for x in lat)),
                 rec["wall_s"], rec["wall_s_per_image"],
                 rec["tim"]["dev"], rec["tim"]["write"],
                 rec["tim"]["pack"], rec["tim"]["host"]), flush=True)

    st = dev.stats
    walls = [a["wall_ms"] for a in st["attempts"] if a["state"] == COMPLETED]
    dev_sum = 1e3 * sum(st["dev_ms"].values()) if False else sum(st["dev_ms"].values())
    write_sum = sum(st["write_ms"].values())
    total = sum(s["seconds"] for s in steps_rec)
    lat_saved = lat[0] if B == 1 else np.stack(lat)
    np.save(HERE / ("%s_latents.npy" % args.tag), lat_saved)
    np.savez(HERE / ("%s_traj.npz" % args.tag),
             latents0=np.stack(lat0), latents=np.stack(lat),
             timesteps=timesteps, sigmas=sigmas)
    rep = {"seed": args.seed, "steps": args.steps, "layers": args.layers,
           "batch": B,
           "images_per_minute": 60.0 * B / total if total > 0 else 0.0,
           "seconds_per_image": total / B if B > 0 else 0.0,
           "dispatches": st["calls"], "retries": st["retries"],
           "barrier_calls": st.get("barrier_calls", 0),
           "barrier_ms": st.get("barrier_ms", 0.0),
           "barrier_states": st.get("barrier_states", {}),
           "ffn_barrier": bool(FFN_BARRIER),
           "images": [args.att_image or (("t3d_pipe2" if args.dimk64
                                          else "t3d_pipe") + ".xclbin"),
                      (args.att_ip_image + ".xclbin") if args.att_ip_image else "",
                      args.ffn_image + ".xclbin"],
           "qkv": bool(QKV),
           "qkv_image": (pathlib.Path(QKV_XCLBIN).name if QKV else ""),
           "p10": bool(P10),
           "p10_image": (pathlib.Path(P10_XCLBIN).name if P10 else ""),
           "xpack_once": bool(XPACK_ONCE),
           "projection_dim_k": 64 if args.dimk64 else 8,
           "dev_ms_total": float(sum(st["dev_ms"].values())),
           "dev_ms_by_image": st["dev_ms"],
           "write_ms_total": float(write_sum),
           "write_ms_by_image": st["write_ms"],
           "wall_s_total": float(total),
           "npu_share_dispatch_only": float(
               sum(st["dev_ms"].values()) / (1e3 * total)),
           "npu_share_dispatch_plus_staging": float(
               (sum(st["dev_ms"].values()) + write_sum) / (1e3 * total)),
           "pfill_n": int(T22_PFILL_STATS["n"]),
           "pfill_s": float(T22_PFILL_STATS["s"]),
           "steps_rec": steps_rec, "calls": calls,
           "driver": "driver376 npu_mcdm_stack_prod; three xclbins, one process"}
    if GLUE is not None:
        rep["glue"] = {k: {"s": round(v[0], 4), "n": int(v[1])}
                       for k, v in sorted(GLUE.items(),
                                          key=lambda kv: -kv[1][0])}
        rep["glue_wall_s"] = float(total)
        rep["glue_residual_s"] = round(total - glue_inside(), 4)
        try:
            # the table as its own artifact: one file per run, easy to diff
            (HERE / ("%s_glue.json" % args.tag)).write_text(json.dumps(
                {"tag": args.tag, "steps": args.steps, "layers": args.layers,
                 "batch": B, "wall_s": float(total),
                 "residual_s": rep["glue_residual_s"],
                 "buckets": rep["glue"]}, indent=1), encoding="utf-8")
        except Exception:                                     # noqa: BLE001
            pass
    (HERE / ("%s_calls.json" % args.tag)).write_text(
        json.dumps(rep, indent=1), encoding="utf-8")
    rep["dev_ms_by_tag"] = st.get("dev_by_tag", {})
    for k in ("dev_ms_by_image", "write_ms_by_image"):
        print("[t3d] %s: %s" % (k, {n: round(v, 0) for n, v in rep[k].items()}))
    print("[t3d] barriers %d (%.1f ms, states %s)"
          % (st.get("barrier_calls", 0), st.get("barrier_ms", 0.0),
             st.get("barrier_states", {})))
    print("[t3d] dispatches %d (%d retries)  device %.1f s  staging %.1f s  "
          "wall %.1f s" % (st["calls"], st["retries"],
                           rep["dev_ms_total"] / 1e3, write_sum / 1e3, total))
    print("[t10] share_weights %s" % SHARE_W)
    print("[t10] batch %d: wall %.2f s  ->  %.3f s/image  %.4f images/min  "
          "(device %.2f s = %.3f s/image, staging %.2f s = %.3f s/image)"
          % (B, total, total / B, rep["images_per_minute"],
             rep["dev_ms_total"] / 1e3, rep["dev_ms_total"] / 1e3 / B,
             write_sum / 1e3, write_sum / 1e3 / B))
    print("[t3d] dev by dispatch tag: %s"
          % {n: round(v, 1) for n, v in rep["dev_ms_by_tag"].items()})
    if QKV:
        _qms = rep["dev_ms_by_tag"].get("att/qkv", 0.0)
        _nq = args.steps * args.layers * ((SEQ + PROJ_M - 1) // PROJ_M)
        print("[qkv] fused q/k/v: %d dispatch(es), %.1f ms total, %.2f ms "
              "per 96-row block"
              % (_nq, _qms, _qms / _nq if _nq else 0.0))
    if P10:
        _oms = rep["dev_ms_by_tag"].get("att/o", 0.0)
        _no = args.steps * args.layers * ((SEQ + PROJ_M - 1) // PROJ_M)
        print("[p10] o projection on %s: %d dispatch(es), %.1f ms total, "
              "%.2f ms per 96-row block"
              % (rep["p10_image"], _no, _oms, _oms / _no if _no else 0.0))
    print("[t3d] NPU share: %.1f %% of the wall is the NPU computing; "
          "%.1f %% counting the host-side staging it forces"
          % (100 * rep["npu_share_dispatch_only"],
             100 * rep["npu_share_dispatch_plus_staging"]))
    if PROF is not None:
        nlayer = float(args.steps * args.layers)
        print("[prof] %-16s %9s %9s %8s" % ("phase", "total s", "ms/layer", "%wall"))
        for kk in sorted(PROF, key=lambda z: -PROF[z]):
            print("[prof] %-16s %9.3f %9.3f %7.1f%%"
                  % (kk, PROF[kk] / 1e3, PROF[kk] / nlayer,
                     100 * PROF[kk] / (1e3 * total)))
        print("[prof] %-16s %9.3f %9.3f %7.1f%%"
              % ("WALL", total, total * 1e3 / nlayer, 100.0))
    glue_print(total, float(args.steps * args.layers * B))
    PRE['report']()
    print("[t3d] latents -> %s_latents.npy (VAE decode is the py3.12 stage)"
          % args.tag, flush=True)
    return 0


def main(argv=None) -> int:
    """argv=None is the delivered entry (sys.argv); the two-chain driver passes
    each chain its own list so two runs cannot race on sys.argv."""
    if os.environ.get("T22_CHAIN2B", "0") == "1" and not _CHAIN2B_CHILD:
        # T22_CHAIN2B: ONE process, TWO chains.  Routed here (not at __main__)
        # because webui/bridge.py calls DRV.main() directly, so this is the only
        # place every caller passes through.  Default OFF => one chain, exactly
        # the delivered code path.
        import chain2b
        return chain2b.run_main(argv)
    global FAST_HOST, SCRATCH, PROF, GLUE, T22_GLUE_PROF, T22_FAST_GLUE, T22_GLUE2
    global T22_GLUE3
    # T22 qkv: declared before the parser reads them as defaults.
    global QKV, QKV_XCLBIN, QKV_BIN, XPACK_ONCE
    global P10, P10_XCLBIN, P10_BIN
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="t4a")
    # T18: the shape this run drives.  The defaults (0,0) keep the delivered
    # M=96 behaviour, so the M=96 A/B arm needs no extra flags.
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--ffn-image", dest="ffn_image", default="t4_sbg",
                    help="FFN xclbin stem under imaggers/block/t4_out")
    ap.add_argument("--att-image", dest="att_image", default="",
                    help="ATTENTION xclbin stem (T5 U1); empty = the T3d image "
                         "selected by --dimk64")
    ap.add_argument("--att-dir", dest="att_dir", default="",
                    help="directory holding the attention image (default the "
                         "T3d t3b_out)")
    ap.add_argument("--att-ip-image", dest="att_ip_image", default="",
                    help="T7 U-1: the IP-ONLY xclbin stem; when given, "
                         "--att-image must be the PROJECTION-ONLY image and the "
                         "loop runs 4 projections + 1 ip instead of 5 fused "
                         "dispatches.  Empty = the T6 fused behaviour (A arm)")
    ap.add_argument("--share-weights", dest="share_weights",
                    action="store_true",
                    help="T10 U-1b: with --batch, stage the layer's BG/B3 into "
                         "the BO once instead of once per image (same bytes, "
                         "gated by the per-image latents)")
    ap.add_argument("--batch", type=int, default=1,
                    help="T10: B independent images (seeds seed..seed+B-1) per "
                         "pass; they share the layer's BG/B3 and SBG bytes and "
                         "nothing else (each image's latents must stay "
                         "byte-identical to its batch=1 run)")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--layers", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dimk64", action="store_true",
                    help="use the dim_k=64 projection image (t3d_pipe2): 60 C "
                         "roundings per projection instead of 480")
    ap.add_argument("--warm", action="store_true",
                    help="pack every layer's projection weights before the loop "
                         "(reported as startup, never inside the loop's wall)")
    ap.add_argument("--assert-sbg", dest="assert_sbg", action="store_true",
                    default=True,
                    help="check that all 96 SBG rows are equal (required by the "
                         "t4_vs FFN image); on by default")
    ap.add_argument("--no-assert-sbg", dest="assert_sbg", action="store_false")
    ap.add_argument("--nh-bg", dest="nh_bg", type=int, default=0,
                    help="T9 U-4b route (a): permute the per-layer BG payload to "
                         "the n_hold image's [pass][kk][j] order (0 = off = the "
                         "delivered path; 2 = for --ffn-image t9_nh2b)")
    ap.add_argument("--nh-bg-core", dest="nh_bg_core", action="store_true",
                    help="T11 U-1: with --nh-bg N, write the PER-CORE order "
                         "[w][pass][kk][j] (t11_permute.py) that the "
                         "per-core-fill image reads; off = the T9 [pass][kk][j] "
                         "order, i.e. the delivered path")
    ap.add_argument("--prefetch", action="store_true",
                    help="read layer i+1's BG/B3 on a worker thread during layer "
                         "i's dispatch (T4-D; zero extra RAM)")
    ap.add_argument("--cache-weights", dest="cache_weights", action="store_true",
                    help="read all 30 layers' BG/B3 once before the loop (T4-D; "
                         "3.98 GB RAM)")
    ap.add_argument("--host-ref", action="store_true",
                    help="compute the host f32 attention+FFN on step 0's first "
                         "layer only (the numerical gate)")
    ap.add_argument("--fast-codec", dest="fast_codec", action="store_true",
                    help="T6: t6_codec's byte-identical, lower-traffic codecs")
    ap.add_argument("--refiner-cache", dest="refiner_cache", default="",
                    help="T6: directory for the decoded refiner f32 weights "
                         "(2.83 GB); empty = decode every run")
    ap.add_argument("--fast-codec2", dest="fast_codec2", action="store_true",
                    help="T7 U-2: t7_codec's pack_b_nk (fewer full-size "
                         "temporaries; 40-case byte gate in t7_codec_gate.py)")
    ap.add_argument("--prof", dest="prof", action="store_true",
                    help="T8: per-phase in-loop attribution (prints s and "
                         "ms/layer at exit; no effect on the arithmetic)")
    # T22 glue (campaign t22_glue): the exclusive in-loop ledger and the two
    # host fast paths.  All three are NO-OPs for the arithmetic; the env vars
    # are what the WebUI bridge forwards, the flags are what the gates use.
    ap.add_argument("--glue-prof", dest="glue_prof", action="store_true",
                    default=T22_GLUE_PROF,
                    help="T22 glue: exclusive in-loop ledger (T22_GLUE_PROF=1); "
                         "--prof turns it on as well")
    ap.add_argument("--t22-fast-glue", dest="t22_fast_glue",
                    action="store_true", default=T22_FAST_GLUE,
                    help="T22 U1: rms_into + one-row SBG + rare SBG assert "
                         "(T22_FAST_GLUE=1; latents byte-gated)")
    ap.add_argument("--no-t22-fast-glue", dest="t22_fast_glue",
                    action="store_false")
    ap.add_argument("--t22-glue2", dest="t22_glue2", action="store_true",
                    default=T22_GLUE2,
                    help="T22 glue-2: copy-free drain/pack fast path "
                         "(T22_GLUE2=1; latents byte-gated)")
    ap.add_argument("--no-t22-glue2", dest="t22_glue2", action="store_false")
    ap.add_argument("--t22-glue3", dest="t22_glue3", action="store_true",
                    default=T22_GLUE3,
                    help="T22 glue-3: absmax without the temporary "
                         "(T22_GLUE3=1; latents byte-gated)")
    ap.add_argument("--no-t22-glue3", dest="t22_glue3", action="store_false")
    ap.add_argument("--fast-host", dest="fast_host", action="store_true",
                    help="T8 U-1: the attention/FFN host glue on numba nJIT "
                         "kernels with one persistent workspace; byte-equal, "
                         "gated by t8_codec_gate.py")
    ap.add_argument("--fast-glue", dest="fast_glue", action="store_true",
                    help="T7 U-2: rms+RoPE in the drain's own layout, so the "
                         "four q/k transposes disappear (latents are the gate)")
    ap.add_argument("--cache-all", dest="cache_all", action="store_true",
                    default=True,
                    help="T13 U-4 (DEFAULT since T13b): read+permute all 30 "
                         "layers' BG/B3 ONCE into 3.98 GB of RAM; the loop then "
                         "does no np.load and no permutation (gate: latents "
                         "byte-identical)")
    ap.add_argument("--no-cache-all", dest="cache_all", action="store_false",
                    help="T13 U-4 A arm: the T11 path (np.load + permutation per "
                         "layer, prefetch thread)")
    ap.add_argument("--bo-resident", dest="bo_resident", action="store_true",
                    help="T13 U-4b: 30 BG + 30 B3 BOs written once (3.98 GB of "
                         "BO memory); the loop writes only A/SBG/X.  Implies "
                         "--no-cache-all (same 3.98 GB, in BOs not RAM).")
    ap.add_argument("--wpack-array", dest="wpack_array", action="store_true",
                    help="T6: one preallocated 2.0 GB array for the projection "
                         "images instead of 120 x 16.6 MB bytes objects, "
                         "allocated BEFORE the device is opened")
    # T22 qkv (campaign t22_qkv_prod): the fused projection image is an OPT-IN.
    # The env var is the WebUI-facing switch (bridge.py forwards it); the CLI
    # flags are what the gate scripts use.
    ap.add_argument("--qkv", dest="qkv", action="store_true", default=QKV,
                    help="T22: run the FUSED q/k/v image -- one dispatch per "
                         "96-row block instead of three projection launches "
                         "(default off; image via --qkv-xclbin/--qkv-bin)")
    ap.add_argument("--no-qkv", dest="qkv", action="store_false")
    ap.add_argument("--qkv-xclbin", dest="qkv_xclbin", default=QKV_XCLBIN,
                    help="T22 qkv: the fused xclbin (default: the t22_qkv "
                         "delivery imaggers/block/qkv/out/qkv_par15m.xclbin)")
    ap.add_argument("--qkv-bin", dest="qkv_bin", default=QKV_BIN,
                    help="T22 qkv: the fused instruction stream (default: "
                         "qkv_par15m.bin next to the xclbin)")
    # T22 p10 (campaign t22_p15prod): the o projection's OWN image is an OPT-IN
    # and is ORTHOGONAL to --qkv (the two replace different dispatches).  --p15
    # is an alias kept from the campaign id: the stream count lives in the
    # IMAGE, not in the flag.
    ap.add_argument("--p10", "--p15", dest="p10", action="store_true",
                    default=P10,
                    help="T22: run the o projection (att/o) on its own "
                         "multi-stream image instead of the delivered 5-stream "
                         "t7_p5 (default off; image via --p10-xclbin/--p10-bin)")
    ap.add_argument("--no-p10", "--no-p15", dest="p10", action="store_false")
    ap.add_argument("--p10-xclbin", dest="p10_xclbin", default=P10_XCLBIN,
                    help="T22 p10: the o-projection xclbin (default: the "
                         "t22_proj delivery imaggers/attn_impl/proj/out/p10.xclbin)")
    ap.add_argument("--p10-bin", dest="p10_bin", default=P10_BIN,
                    help="T22 p10: its instruction stream (default: p10.bin "
                         "next to the xclbin)")
    ap.add_argument("--xpack-once", dest="xpack_once", action="store_true",
                    default=XPACK_ONCE,
                    help="T22: pack the projection X image once per row block "
                         "instead of once per projection (host-side only; the "
                         "three launches still receive the same bytes)")
    ap.add_argument("--no-xpack-once", dest="xpack_once", action="store_false")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if a.height and a.width:
        configure(a.height, a.width)
    if CHUNKED:
        # chunk_path uses the delivered projection/FFN/o images only; the
        # optional fused qkv / p10 images are not part of its cache manifest.
        a.qkv = False
        a.p10 = False
    global ASSERT_SBG, REFCACHE, WPACK, NH_BG, NH_BG_CORE
    ASSERT_SBG = bool(a.assert_sbg)
    global CACHE_ALL
    CACHE_ALL = bool(getattr(a, "cache_all", True))
    if getattr(a, "bo_resident", False):
        CACHE_ALL = False
    NH_BG = int(getattr(a, "nh_bg", 0))
    NH_BG_CORE = bool(getattr(a, "nh_bg_core", False))
    if a.fast_codec:
        import t6_codec
        t6_codec.install()
        print("[t6] fast codec installed: pack_b_nk 1.78x, unpack_blocked "
              "1.06x (byte-identical, t6_codec_gate.py ALL_PASS)", flush=True)
    if a.fast_codec2:
        import t7_codec
        t7_codec.install()
        print("[t7] fast codec 2 installed: pack_b_nk K 1.48x / V 1.10x, "
              "unpack from t6 (t7_codec_gate.py ALL_PASS)", flush=True)
    FAST_HOST = bool(a.fast_host)
    T22_FAST_GLUE = bool(getattr(a, "t22_fast_glue", T22_FAST_GLUE))
    T22_GLUE2 = bool(getattr(a, "t22_glue2", T22_GLUE2))
    T22_GLUE3 = bool(getattr(a, "t22_glue3", T22_GLUE3))
    T22_GLUE_PROF = bool(getattr(a, "glue_prof", T22_GLUE_PROF))
    QKV = bool(a.qkv)
    XPACK_ONCE = bool(a.xpack_once)
    if QKV:
        if not a.qkv_xclbin:
            QKV_XCLBIN = str(QKV_DEFAULT_OUT / (QKV_DEFAULT_TAG + ".xclbin"))
            QKV_BIN = str(QKV_DEFAULT_OUT / (QKV_DEFAULT_TAG + ".bin"))
        else:
            QKV_XCLBIN = str(a.qkv_xclbin)
            QKV_BIN = (str(a.qkv_bin) if a.qkv_bin else
                       str(pathlib.Path(a.qkv_xclbin).with_suffix(".bin")))
        for _p in (QKV_XCLBIN, QKV_BIN):
            if not pathlib.Path(_p).is_file():
                raise SystemExit("[qkv] missing image file: %s" % _p)
        if not a.fast_host:
            raise SystemExit("[qkv] --qkv patches Loop.attention_fast only; "
                             "add --fast-host (the delivered attention_ref "
                             "path is not fused)")
        print("[qkv] fused q/k/v image: %s  (+ %s)"
              % (QKV_XCLBIN, QKV_BIN), flush=True)
    P10 = bool(a.p10)
    if P10:
        if not a.p10_xclbin:
            P10_XCLBIN = str(P10_DEFAULT_OUT / (P10_DEFAULT_TAG + ".xclbin"))
            P10_BIN = str(P10_DEFAULT_OUT / (P10_DEFAULT_TAG + ".bin"))
        else:
            P10_XCLBIN = str(a.p10_xclbin)
            P10_BIN = (str(a.p10_bin) if a.p10_bin else
                       str(pathlib.Path(a.p10_xclbin).with_suffix(".bin")))
        for _p in (P10_XCLBIN, P10_BIN):
            if not pathlib.Path(_p).is_file():
                raise SystemExit("[p10] missing image file: %s" % _p)
        if not a.fast_host:
            raise SystemExit("[p10] --p10 patches Loop.attention_fast only; "
                             "add --fast-host")
        print("[p10] o projection image: %s  (+ %s)"
              % (P10_XCLBIN, P10_BIN), flush=True)
    if XPACK_ONCE:
        print("[qkv] xpack-once: X packed once per row block "
              "(host-side only; the launches get the same bytes)", flush=True)
    if a.prof or T22_GLUE_PROF:
        PROF = {}
        GLUE = {}
    if FAST_HOST:
        # Every codec this loop calls becomes the nJIT kernel, and every
        # in-loop buffer comes from one persistent workspace.  The gate is
        # byte equality against t3d_codec on the 40 T6/T7 adversarial cases
        # (t8_codec_gate.py), plus the delivered latents.
        H.install()
        SCRATCH = H.Scratch()
        print("[t8] fast host installed: pack_b_nk %.2fx, unpack %.2fx, "
              "pack_blocked %.2fx, x_image64 %.2fx, bf16 %.2fx, rope %.2fx "
              "(byte-equal, t8_codec_gate.py ALL_PASS)"
              % (9.6, 4.3, 16.7, 4.2, 40.9, 4.4), flush=True)
    global GLUE_FAST
    GLUE_FAST = bool(a.fast_glue)
    if GLUE_FAST:
        print("[t7] fast glue: rms+RoPE stay in the drained (HEADS, M, HD) "
              "layout (no q/k transpose)", flush=True)
    REFCACHE = a.refiner_cache or None
    if REFCACHE:
        pathlib.Path(REFCACHE).mkdir(parents=True, exist_ok=True)
        print("[t6] refiner cache dir: %s" % pathlib.Path(REFCACHE).resolve(),
              flush=True)
    if a.wpack_array:
        n = a.layers * 4
        print("[t6] allocating %.2f GB for the projection images BEFORE the "
              "device is opened" % (n * W_BYTES / 2**30), flush=True)
        WPACK = np.empty(n * W_BYTES, np.uint8)
    if CHUNKED:
        from webui.chunked import install as _install_chunked
        # T22_CHAIN2B: install on THIS module instance.  chunked.install() used
        # to resolve sys.modules["t18_run"], which is module A -- so a second
        # chain would have patched the FIRST chain's Loop class and block_step.
        info = _install_chunked(sys.modules[__name__])
        print("[t18] chunk_path installed: %s (proj/ffn/o = delivered M=96 "
              "row blocks; attention = host f32)" % info.get("attention"),
              flush=True)
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
