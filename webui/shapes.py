# SPDX-License-Identifier: MIT
"""Arbitrary (H, W) shape arithmetic, validation and budget (T16 P0-5).

The WebUI used to expose one fixed profile (128x128, M=96).  The product owner
wants to type any width/height and switch at will.  This module is the single
place that decides what "any" means:

    H_pad, W_pad = ceil(H/16)*16, ceil(W/16)*16   # internal 16-px grid
    latent       = (H_pad/8, W_pad/8)             # the Z-Image VAE is 8x down
    image tokens = (H_pad/16) * (W_pad/16)        # patch size 2 on the latent
    padded       = ceil(tokens / 32) * 32         # SEQ_MULTI_OF, x_pad_token rows
    M            = 32 + padded                    # 32 caption rows + image rows

The checkpoint is shape-agnostic; the NPU is not.  The delivered images
(t7_p5 attention, t7_ip attention-IP, t11_nh5 FFN) were compiled for M=96 and
the delivered host driver (t13_run.py / t3_real.py) hardcodes the 8x8 patch
grid, the 96x128 and 96x64 permutations, the SBG walk and the 30-layer
constants.  A new (H, W) is therefore a BUILD, not a UI setting: six images
(FFN, 4 projections, IP) + permutations + host constants.  This module returns
a precise plan for every shape instead of pretending a dropdown can conjure it.

The paper budget (§ T16-P0-5-BUDGET.md) fixes the limits below from the actual
object sizes, the m=16 FFN row tiles, SEQ_MULTI_OF=32 padding and the 64 KB L1
/ BD budgets recorded in imaggers/block/T13.md and T10.md.

T22 prodlarge (2026-10-20) widens the *product* envelope to the whole
128²..2048² spectrum: MIN_HW/MAX_HW stay the same kind of limit (64..**2048**;
requests are any integer, the internal grid is 16 px) and MAX_M becomes
32+16384 = **16416** (2048² = 128x128 image
tokens).

T22 pad_crop (2026-10-21): a request may be ANY integer in 64..2048.  Every
plan is computed on the internal 16-px grid (H_pad = ceil(H/16)*16, W_pad
likewise), and the plan carries BOTH the requested and the padded size.  The
delivery rule is fixed: run the padded frame, then keep the requested
[0:H, 0:W] rectangle ("bottom-right trim"), so the output is exactly H x W.  No new device image is built here -- the ceiling of the DELIVERED
attention image is still SEQ=304 (272 image tokens, T22-BUDGET.md §1), so every
shape above that is reported ready=false with the precise arithmetic.
`t22_budget()` adds the campaign-1 size budget (M, projected dispatches,
projected device time) using the exact anchors and formulae of
imaggers/attn_impl/T22-BUDGET.md / t22_large_budget.py, so the product panel and
the paper table cannot drift apart.
"""
from __future__ import annotations

import json
import math
import os
import pathlib

CAP_TOKENS = 32
SEQ_MULTI_OF = 32
PATCH = 2
VAE_SCALE = 8
HIDDEN = 3840
FFN_INNER = 10240
BYTES_PER_ELEM = 2          # bf16 / BFP16 stream on the device

MIN_HW = 64
MAX_HW = 2048               # T22 prodlarge: 128²..2048² is the product envelope
MAX_M = 32 + 16384          # 2048x2048: 128x128 image tokens -> M=16416
MIN_M = 64                  # 64x64: 4x4 tokens -> 16 -> padded 32 -> M=64
FFN_ROWS = 96               # one FFN/projection dispatch covers 96 rows


class ShapeError(ValueError):
    pass


PAD_TO = 16


def pad16(value) -> int:
    """Round one side UP to the internal 16-px grid (64 -> 64, 130 -> 144)."""
    return (int(value) + PAD_TO - 1) // PAD_TO * PAD_TO


def run_dims(h, w):
    """(H_pad, W_pad) for a request; validates the REQUESTED 64..2048 range."""
    h, w = int(h), int(w)
    if not (MIN_HW <= h <= MAX_HW and MIN_HW <= w <= MAX_HW):
        raise ShapeError(f"{h}x{w}: each side must be {MIN_HW}..{MAX_HW}")
    return pad16(h), pad16(w)


def image_tokens(h: int, w: int) -> int:
    """Image tokens of the padded run (requested H/W are accepted as-is)."""
    hp, wp = run_dims(h, w)
    return (hp // 16) * (wp // 16)


def pad_tokens(t: int) -> int:
    return int(math.ceil(t / SEQ_MULTI_OF) * SEQ_MULTI_OF)


def m_from_hw(h: int, w: int) -> int:
    """The unified sequence length the device image would be compiled for."""
    t = pad_tokens(image_tokens(h, w))
    m = CAP_TOKENS + t
    if m % 16:
        raise ShapeError(f"{h}x{w}: M={m} is not a multiple of 16 (FFN row tile)")
    if not (MIN_M <= m <= MAX_M):
        raise ShapeError(f"{h}x{w}: M={m} is outside {MIN_M}..{MAX_M}")
    return m


def profile_key(h: int, w: int) -> str:
    """Build/cache key: always the padded run size."""
    hp, wp = run_dims(h, w)
    return f"{hp}x{wp}"


def requested_key(h: int, w: int) -> str:
    return f"{int(h)}x{int(w)}"


def budget(h: int, w: int) -> dict:
    """Paper budget for one shape: object sizes, weight re-reads, feasibility.

    Every internal number is computed on (H_pad, W_pad); the record also carries
    the requested pair and the pad flag so the API/UI can echo both.
    """
    h, w = int(h), int(w)
    hp, wp = run_dims(h, w)
    t = image_tokens(h, w)
    tp = pad_tokens(t)
    m = CAP_TOKENS + tp
    latent = (hp // VAE_SCALE, wp // VAE_SCALE)
    grid = (hp // 16, wp // 16)
    a_bytes = m * HIDDEN * BYTES_PER_ELEM          # one A row per token
    sbg_bytes = m * 3 * HIDDEN * BYTES_PER_ELEM    # 3 planes of the block input
    y_bytes = m * HIDDEN * BYTES_PER_ELEM          # block output
    bg_bytes = 2 * FFN_INNER * HIDDEN * 9 // 8     # per-layer w1/w3 stream
    b3_bytes = HIDDEN * FFN_INNER * 9 // 8         # per-layer w2 stream
    # 128x128 is the measured reference point (loop 27.7 s; 24.95 s/img at batch 4)
    scale = m / 96.0
    return {
        "height": hp, "width": wp,
        "requested_height": h, "requested_width": w,
        "pad_height": hp, "pad_width": wp,
        "pad_applied": bool(hp != h or wp != w),
        "requested_key": requested_key(h, w), "run_key": f"{hp}x{wp}",
        "latent": list(latent), "patch_grid": list(grid),
        "image_tokens": t, "padded_tokens": tp, "m": m,
        "m_scale_vs_128": round(scale, 3),
        "a_row_bytes": a_bytes // m if m else 0,
        "a_bytes": a_bytes, "sbg_bytes": sbg_bytes, "y_bytes": y_bytes,
        "weights_per_layer_bg_bytes": bg_bytes, "weights_per_layer_b3_bytes": b3_bytes,
        "weights_per_layer_total_bytes": bg_bytes + b3_bytes,
        "loop_estimate_s_per_image": round(27.7 * scale, 1),
        "bd_note": ("A is one row-object per token (7680 B); SBG is 3 planes; "
                    "per-layer weight streams stay constant (BG 88.47 MB + B3 44.24 MB)"),
        "rx16k_note": ("the per-core RX/L1 budget is independent of M; the M-dependent "
                       "objects are host BOs and the memtile C walk"),
    }


# ---------------------------------------------------------------------------
# T22 campaign 1: the full-spectrum size budget (paper stage, no device).
# Ported 1:1 from imaggers/attn_impl/t22_large_budget.py so the product panel
# and T22-BUDGET.md cannot diverge: same anchors, same formulae, same labels.
# Every time column is an ESTIMATE from measured unit costs (T22-BUDGET.md §0);
# every L1 / dispatch column is exact arithmetic.
# ---------------------------------------------------------------------------
T22_HD = 128                 # head dim
T22_HEADS = 30               # attention heads per layer
T22_LAYERS = 30              # dit blocks in Z-Image-Turbo
T22_L1 = 65536               # per-core L1 data budget (bytes)
T22_STACK = 0x1800           # 6,144 B stack reservation of the delivered ip
T22_MEMTILE = 512 * 1024     # memtile per column
T22_LK = 192                 # blocked design: keys per block
T22_MQ = 32                  # blocked design: query rows per block
T22_PROJ_MS_PER_96 = 5.62    # measured: one projection, 96 rows
T22_FFN_MS_PER_96 = 40.3     # measured: one FFN block, 96 rows
T22_IP_GFLOPS = 40.0         # measured ip throughput (3 points)
T22_IP_BW_GBPS = 1.5         # device object stream implied by the same dispatches
T22_DISP_MS = 1.0            # per-dispatch overhead used by campaign 1
T22_WATCHDOG_S = 7.0         # ERT per-dispatch watchdog (platform fact)
# SEQ values the DELIVERED images serve: seq -> (Mq, ND)
T22_DELIVERED = {96: (96, 96), 128: (64, 128), 160: (32, 160), 288: (16, 288)}
# Rows one FFN dispatch covers, i.e. the row tiling of the image each delivered
# SEQ actually runs (T18 added an M=128 image; everything else runs the
# delivered 96-row t11_nh5).  Measured per-layer dispatch counts of the six
# delivered sizes: 128x128 6, 144x128/192x128 11, 128x256 15, 256x256 33
# (webui/evidence T18 §5.2, T20 §3, T19 §3) -- 4*nproj + nip + nffn with
# nffn = ceil(SEQ/ffn_rows).
T22_FFN_ROWS_DELIVERED = {96: 96, 128: 128, 160: 96, 288: 96}


def t22_cur_ip_l1(mq: int, seq: int) -> int:
    """t7_ip's L1: QP(Mq*128*2) + KB(128*SEQ/8*9) + C(Mq*SEQ*2) + stack."""
    return mq * T22_HD * 2 + T22_HD * seq // 8 * 9 + mq * seq * 2 + T22_STACK


def t22_cur_ip_mq(seq: int) -> int:
    """Largest query block today's image admits at this SEQ (0 = infeasible)."""
    for mq in (96, 64, 32, 16):
        if seq % mq:
            continue
        if (t22_cur_ip_l1(mq, seq) <= T22_L1
                and 2 * mq * seq * 2 <= T22_MEMTILE):
            return mq
    return 0


def t22_blocks(seq: int, mq: int = T22_MQ, lk: int = T22_LK):
    """Query/key block counts of the blocked (key-tiled) attention schedule."""
    return ((seq + mq - 1) // mq, (seq + lk - 1) // lk)


def t22_attn_ms(seq: int, mq: int = T22_MQ, lk: int = T22_LK,
                gflops: float = T22_IP_GFLOPS) -> dict:
    """QK+PV time for one layer under the Q-outer blocked schedule.

    compute : 4*SEQ*HD*SEQ*HEADS FLOPs at the measured ip rate
    data    : nq * HEADS*SEQ*288 B (KB+VB re-read per query block)
    S/P     : 2*HEADS*SEQ*SEQ*2 B (off-core softmax staging)
    """
    nq, nk = t22_blocks(seq, mq, lk)
    flops = 4 * seq * T22_HD * seq * T22_HEADS
    compute_ms = flops / (gflops * 1e9) * 1e3
    kv_ms = nq * T22_HEADS * seq * 288 / (T22_IP_BW_GBPS * 1e9) * 1e3
    sp_ms = 2 * T22_HEADS * seq * seq * 2 / (T22_IP_BW_GBPS * 1e9) * 1e3
    ndisp = 2 * nq * nk
    return {"nq": nq, "nk": nk, "flops": flops, "compute_ms": compute_ms,
            "kv_ms": kv_ms, "sp_ms": sp_ms, "ndisp": ndisp,
            "disp_ms": T22_DISP_MS * ndisp,
            "ms": max(compute_ms, kv_ms) + sp_ms + T22_DISP_MS * ndisp}


def t22_ffn_rows(seq: int) -> int:
    """Rows one FFN dispatch covers for this SEQ (the image's row tiling)."""
    return T22_FFN_ROWS_DELIVERED.get(int(seq), FFN_ROWS)


def t22_ip_dispatches(seq: int) -> int:
    """ip dispatches per layer for the regime this SEQ lands in.

    One dispatch per query block, always.  The regime only decides what a
    query block is:
      * the SEQ is delivered -> the delivered image's own Mq (96/64/32/16),
        ONE dispatch against all keys (they fit in L1): ceil(SEQ/Mq);
      * today's ip still compiles here -> its largest legal Mq, same rule;
      * blocked design -> every query block is split by key blocks, so
        2*nq*nk dispatches (this is T22-BUDGET.md §6's number).
    The delivered branch is what a real run reports: 128x128, 8 steps ->
    4 projections + 1 ip + 1 FFN = 6 per layer = 180 per step = 1440
    dispatches (the delivered calls.json value).
    """
    if seq in T22_DELIVERED:
        mq = T22_DELIVERED[seq][0]
        return (seq + mq - 1) // mq
    mq = t22_cur_ip_mq(seq)
    if mq:
        return (seq + mq - 1) // mq
    nq, nk = t22_blocks(seq)
    return 2 * nq * nk


def t22_budget(m: int) -> dict:
    """The campaign-1 budget for one unified sequence length M (= SEQ)."""
    seq = int(m)
    nblk = (seq + FFN_ROWS - 1) // FFN_ROWS
    a = t22_attn_ms(seq)
    mq_cur = t22_cur_ip_mq(seq)
    l1_cur = t22_cur_ip_l1(mq_cur or 16, seq)
    proj_ms = 4 * nblk * T22_PROJ_MS_PER_96
    ffn_ms = nblk * T22_FFN_MS_PER_96
    step_s = (a["ms"] + proj_ms + ffn_ms) * T22_LAYERS / 1e3
    disp_proj = 4 * nblk
    disp_attn = t22_ip_dispatches(seq)
    disp_ffn = (seq + t22_ffn_rows(seq) - 1) // t22_ffn_rows(seq)
    # Per layer, then per step (30 layers) and per 8-step run.  NB: campaign 1's
    # table column called "total/step" is the per-LAYER sum (it is the sum of
    # that table's own per-layer columns, and T22-BUDGET.md §7 multiplies it by
    # 30); the fields below are the unambiguous per-step numbers.  Validation:
    # 128x128 -> 6 per layer -> 180 per step -> 1440 per run, exactly the
    # delivered 8-step calls.json "dispatches" value.
    disp_layer = disp_proj + disp_attn + disp_ffn
    disp_step = disp_layer * T22_LAYERS
    return {
        "t22_source": "imaggers/attn_impl/T22-BUDGET.md (campaign 1)",
        "t22_seq": seq,
        "t22_layers": T22_LAYERS,
        "t22_lk": T22_LK,
        "t22_mq": T22_MQ,
        "t22_nq": a["nq"],
        "t22_nk": a["nk"],
        "t22_attn_compute_ms": round(a["compute_ms"], 2),
        "t22_attn_kv_ms": round(a["kv_ms"], 2),
        "t22_attn_sp_ms": round(a["sp_ms"], 2),
        "t22_attn_dispatch_ms": round(a["disp_ms"], 2),
        # attention per layer = max(compute, KV) + S/P staging + dispatch
        "t22_attn_ms_per_layer": round(a["ms"], 2),
        "t22_proj_ms_per_layer": round(proj_ms, 2),
        "t22_ffn_ms_per_layer": round(ffn_ms, 2),
        "t22_dev_s_per_step": round(step_s, 2),
        "t22_dev_h_8step": round(step_s * 8 / 3600.0, 2),
        # dispatches per layer / step / 8-step run (T22-BUDGET.md §6)
        "t22_dispatches_proj_per_layer": disp_proj,
        "t22_dispatches_attn_per_layer": disp_attn,
        "t22_dispatches_ffn_per_layer": disp_ffn,
        "t22_dispatches_per_layer": disp_layer,
        "t22_dispatches_step": disp_step,
        "t22_dispatches_8step": disp_step * 8,
        # state-8: a second back-to-back launch of the same FFN image never
        # completes, so a different xclbin launch sits between FFN blocks.
        "t22_barriers_8step": T22_LAYERS * 8 * max(0, nblk - 1),
        "t22_longest_dispatch_ms": round(max(
            4 * T22_MQ * T22_HD * T22_LK * T22_HEADS / (T22_IP_GFLOPS * 1e9) * 1e3,
            (T22_HD * T22_LK // 8 * 9) / (T22_IP_BW_GBPS * 1e9) * 1e3,
            T22_FFN_MS_PER_96), 2),
        # what TODAY's delivered image can do at this SEQ (exact arithmetic)
        "t22_cur_ip": ("delivered" if seq in T22_DELIVERED
                       else ("compilable" if mq_cur else "none")),
        "t22_cur_mq": mq_cur,
        "t22_cur_ip_l1_bytes": l1_cur,
        "t22_l1_limit_bytes": T22_L1,
        "t22_l1_over": l1_cur > T22_L1,
        "t22_watchdog_s": T22_WATCHDOG_S,
        "t22_note": ("device-only estimate from the measured T22-BUDGET anchors "
                     "(no host glue, no VAE); the delivered sizes measure ~10-20% "
                     "below the model (2.0 vs 2.2 s/step at 128x128); dispatch "
                     "counts are one per query block in every regime; NOT "
                     "SHIPPABLE -- no blocked image has been gated"),
    }


# ---------------------------------------------------------------------------
# T22 chunk_path (2026-10-21): arbitrary M on the delivered M=96 kernels.
#
# The product path's ip image keeps the whole key axis in L1 (SEQ <= 304).  For
# every larger shape the linear row-wise work -- projection, FFN, o projection
# -- still fits the delivered M=96 images, dispatched in 96-row blocks.  The
# non-linear part (attention) runs on the host in f32 (torch, AVX-512/OpenMP),
# fed by the device projections.  No image is compiled and no delivered path is
# touched.  The estimate below is anchored on the 480x848 prototype (M=1632):
# 65.4 s/step measured = device 27.8 s + staging 4.7 s + host 32.9 s; the model
# below is deliberately conservative (~10-15 % above that measurement).
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# T22 adaptive (2026-11-03): THE HONESTY GATE.
#
# chunk_gate() used to answer "runnable" from an L1 + host-RAM model that never
# looked at whether a 32-core placement EXISTS for this M, whether its images
# are BUILT, or whether the product's host BOs COVER M.  1024^2 was therefore
# promised at "~1.5 h" while the whole-shape projection BO was sized for
# 11 x 96 = 1056 rows (imaggers/image/t18_run.py:377) and the run died with
# "attempting to write past buffer size".
#
# The gate now asks webui/npu_plan.py (pure function) for a plan first, and
# quotes the binding constraint when there is none.  Every field it adds to the
# estimate is prefixed plan_ so nothing that already reads chunk_estimate()
# changes meaning.
# ---------------------------------------------------------------------------
PLAN_CACHE = {}
HOST_BO_MAX = 2 << 30          # one host BO above 2 GiB is not runnable here
WHOLE_SHAPE_FLAGS = ("T22_HOSTKILL", "T22_DEVKILL")   # whole-M BO paths


def built_plan_keys() -> set:
    """Every plan key whose image set is BUILT (a complete plan manifest).

    The product and the builder must agree on WHICH plan a shape resolves to, or
    the UI would name a plan that the builder would not produce.  npu_plan.plan()
    takes this set as its cache-affinity tie-break (a plan that is already built
    wins a tie inside 15 % of the best cost), so passing it here makes
    shapes.resolve() return exactly what npu_synth.synth() would build.
    """
    keys = set()
    d = _WEBUI_DIR / "shape_manifests"
    if not d.is_dir():
        return keys
    for p in d.glob("plan_*.json"):
        try:
            man = json.loads(p.read_text(encoding="utf-8"))
        except Exception:                                      # noqa: BLE001
            continue
        if man.get("built_ok"):
            keys.add(man.get("plan_key"))
    return keys


def npu_plan_for(m) -> dict:
    """webui/npu_plan.plan(M) with the built-plan affinity tie-break.

    Not memoised: the planner is ~1 ms per call and a cached answer would go
    stale the moment a new plan is built.
    """
    from . import npu_plan
    return npu_plan.plan(int(m), prefer=built_plan_keys())


def plan_manifest(m=None, plan_key=None) -> dict:
    """The build record of a plan, or None.  The same file npu_synth writes.

    shapes.py reads it directly (it must not import imaggers), so the manifest
    path + the plan_key field are the whole contract between the two sides.
    """
    if plan_key is None:
        res = npu_plan_for(m)
        if not res.get("ok"):
            return None
        plan_key = res["plan"]["plan_key"]
    p = _WEBUI_DIR / "shape_manifests" / ("plan_%s.json" % plan_key)
    try:
        man = json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                          # noqa: BLE001
        return None
    if man.get("plan_key") != plan_key or not man.get("built_ok"):
        return None
    for img in (man.get("images") or {}).values():
        if img.get("status") not in ("BUILT", "CACHED"):
            return None
        if not pathlib.Path(img.get("xclbin") or "").is_file():
            return None
    return man


def _whole_shape_rows():
    """Rows the whole-shape big-M BOs cover (t18_run T22_ADAPT_M, default 1056)."""
    try:
        return int(os.environ.get("T22_ADAPT_M", "1056") or 1056)
    except Exception:                                          # noqa: BLE001
        return 1056


def plan_summary(plan) -> dict:
    """The compact plan record the API/UI carries (never the whole tile map)."""
    return {
        "plan_key": plan["plan_key"], "M": plan["M"], "strategy": plan["strategy"],
        "m": plan["m"], "kk": plan["kk"], "Mblk": plan["Mblk"],
        "ndisp": plan["ndisp"], "T": plan["T"], "nm": plan["nm"],
        "per_col": plan["per_col"], "l1_max": plan["l1_max"],
        "pad_rows": plan["pad_rows"],
        "n": {k: plan["gemms"][k]["n"] for k in plan["gemms"]},
        "ntiles": {k: plan["gemms"][k]["ntiles"] for k in plan["gemms"]},
        "layer_ms": plan["layer_ms"], "step_s": plan["step_s"],
        "s_8step": plan["s_8step"], "h_8step": plan["h_8step"],
    }


CHUNK_MIN_SEQ = 305            # first padded SEQ above the delivered ip ceiling
CHUNK_ATTN_GFLOPS = 600.0      # measured torch f32 matmul+softmax, 480p/1080p
CHUNK_HOST_MS_PER_BLOCK = 70.0 # measured host glue per 96-row block per layer
CHUNK_BARRIER_MS = 1.7         # one dummy ip launch between FFN row blocks
CHUNK_MAX_8STEP_S = 6 * 3600.0 # honest device-time ceiling for one job
CHUNK_MAX_BYTES = 8 << 30      # host attention working set ceiling

# ---------------------------------------------------------------- T22 honest-2
# Two constraints that the L1 / BO / attention model above does NOT see, both
# measured on this box on 2026-11-03 by running the product:
#
#  (1) THE NOISE REFINER's f32 score matrix.  The refiner is CPU numpy (never
#      device-ised) and materialises (heads, S, S) float32 with S = padded image
#      tokens per step.  Measured: S=4096 (1024^2) => 1.88 GiB and an 8-step job
#      completes (job 20261031-134742, 3166.8 s); S=8160 (1088x1920) => 7.44 GiB
#      and the device loop dies with "ran out of memory" 42 s in (job
#      20261103-150606-10d726); S=14400 (1920^2) => 23.17 GiB, same death at
#      34 s (job 20261103-150555-...).  The device child already holds the
#      3.71 GB cache-all image set plus the 1.85 GB wpack mapping, so the line
#      is drawn at 6 GiB: 1.88 runs, 7.44 does not.  Raise it deliberately with
#      ZIMAGE_CHUNK_REFINER_GB after freeing memory (memory_mode=low).
CHUNK_REFINER_MAX_BYTES = 6 << 30


def chunk_refiner_bytes(m: int, tokens: int = None) -> int:
    """f32 score matrix the CPU noise refiner materialises for this shape.

    heads x S x S x 4 B, S = the padded image-token count (the 32 caption rows
    are not in the refiner).  An estimate of the dominant term, not of the
    whole refiner working set.

    This is what the DELIVERED expression allocates.  Since 2026-11-03 the
    refiner query-blocks above REF_QCEIL_GB, so the tensor a job actually
    commits is chunk_refiner_peak_bytes() -- see the refchunk block below.
    """
    if tokens is None:
        seq = int(m) - CAP_TOKENS
        tokens = max(1, seq)
    return int(T22_HEADS) * int(tokens) * int(tokens) * 4


# ---------------------------------------------------------------- refchunk
# 2026-11-03: the noise refiner's attention is QUERY-BLOCKED above a ceiling
# (t3_real._ref_attn -> imaggers/image/ref_attn_block.py), so a shape no longer
# has to pay for a (heads, S, S) f32 tensor: the block bounds it to
# heads x rows x S x 4 B and the softmax still runs over the FULL key axis, so
# no arithmetic changes (gate_refchunk.py: maxabs 0.000e+00 against the
# delivered expression at S=1024 and S=4096, and between three block sizes at
# S=8160).  That is why S=8160 -- 1080p, refused here since 2026-11-03 -- is
# runnable: its peak becomes the S=4096 peak instead of 4x it.
#
# These two numbers and the three env names are the SAME ones
# ref_attn_block.py defines; gate_refchunk.py asserts the two agree, because a
# second copy of this policy that drifted would admit a shape the refiner then
# OOMs on.
REF_QCHUNK_ENV = "T22_REF_QCHUNK"
REF_QCEIL_ENV = "T22_REF_QCHUNK_GB"
REF_QBLOCK_ENV = "T22_REF_QBLOCK_GB"
REF_QCEIL_GB = 2.0            # block above this score matrix (1.88 GiB = S=4096
#                              still runs the delivered expression, as measured)
REF_QBLOCK_GB = 0.5           # the per-block budget: S=8160 -> 512 rows


def _env_gb(name: str, default: float) -> float:
    try:
        v = float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return float(default)
    return v if v > 0 else float(default)


def chunk_refiner_blocked() -> bool:
    """True when the refiner will query-block this job (T22_REF_QCHUNK=off
    restores the delivered S x S expression, i.e. the 2026-11-03 refusals)."""
    return (os.environ.get(REF_QCHUNK_ENV, "auto") or "auto").strip(
        ).lower() not in ("0", "off", "no", "false", "none")


def chunk_refiner_rows(tokens: int) -> int:
    """Query rows per block the refiner will use at this S.

    Mirrors ref_attn_block.block_rows() exactly: a multiple of 64, at least 64,
    never more than S, derived from the REF_QBLOCK_GB byte budget.  gate_refchunk
    asserts the two agree (it compares against ref_attn_block.block_rows).
    """
    seq = max(1, int(tokens))
    per_row = max(1, int(T22_HEADS) * seq * 4)
    n = int(_env_gb(REF_QBLOCK_ENV, REF_QBLOCK_GB) * 2 ** 30 // per_row)
    n = max(64, (n // 64) * 64)
    return max(1, min(seq, n))


def chunk_refiner_peak_bytes(m: int, tokens: int = None) -> int:
    """The score-matrix tensor the CPU noise refiner will ACTUALLY materialise.

    Above REF_QCEIL_GB the query axis is blocked, so it is one
    (heads, rows, S) block instead of the whole S x S.  This is the number the
    admission gate must compare against the ceiling: the ceiling exists to keep
    the largest single allocation (and therefore the process's peak commit,
    which measures at ~4x it) inside this box's headroom.
    """
    raw = chunk_refiner_bytes(m, tokens)
    if not chunk_refiner_blocked():
        return raw
    if raw <= _env_gb(REF_QCEIL_ENV, REF_QCEIL_GB) * 2 ** 30:
        return raw
    if tokens is None:
        tokens = max(1, int(m) - CAP_TOKENS)
    return int(T22_HEADS) * chunk_refiner_rows(int(tokens)) * int(tokens) * 4


# (2) The FASTGLUE projection hook.  With T22_FASTGLUE=1 + T22_DEVKILL=1 +
#     ZIMAGE_BIGM=1 the projection is dispatched by imaggers/util/glue/install.py
#     through the rt17 chunk images, whose host BO holds BIGM_CHUNK_ROWS rows
#     (imaggers/image/t18_run.py:397, default 1632).  M beyond that is an
#     instant "attempting to write past buffer size": measured on 1024^2
#     (M=4128) in job 20261103-063607-d60112, dead 20.9 s in.  The delivered
#     chunked path (flag off) has no such BO and is unaffected.
FASTGLUE_PROJ_ROWS_ENV = "T22_ADAPT_CHUNK"
FASTGLUE_PROJ_ROWS_DEFAULT = 1632


def fastglue_proj_rows() -> int:
    try:
        return int(os.environ.get(FASTGLUE_PROJ_ROWS_ENV,
                                  str(FASTGLUE_PROJ_ROWS_DEFAULT)) or
                   FASTGLUE_PROJ_ROWS_DEFAULT)
    except Exception:                                          # noqa: BLE001
        return FASTGLUE_PROJ_ROWS_DEFAULT


def fastglue_projection_active() -> bool:
    """True when the env selects the fastglue/devkill projection hook."""
    on = lambda k: os.environ.get(k, "0") not in ("0", "", "false", "False")  # noqa: E731
    return (on("ZIMAGE_BIGM") and on("T22_HOSTKILL") and on("T22_DEVKILL")
            and on("T22_FASTGLUE"))


# ---------------------------------------------------------------------------
# (2b) T22 PROJ-BLOCK (2026-11-04): the projection is dispatched in BLOCKS.
#
# The big-M projection image is compiled for a FIXED number of 96-row tiles
# (plain15 / plain: RT=11 = 1056 rows; the rt17 pair: 17).  ONE launch computes
# its tile count and nothing more, so a whole-shape dispatch above 1056 rows
# computed 1056 rows and read STALE BO memory for the rest (measured on this
# box: plain15, 2112 rows in, rows 1056..2111 zero and unmoved when the X above
# 1056 changed).  imaggers/image/t18_run.py now dispatches one block per launch
# above PROJ_BLOCK_CEILING, so the projection consumer reads ONE block and the
# host BO has to hold ONE block -- not the whole shape.
#
# These constants and the predicate MIRROR t18_run.proj_block_rows().  They are
# duplicated (not imported) because this module is imported by the py312
# planner, which has no numba and cannot import the driver; the two must be
# changed together, and the preflight reads the driver's own printed line.
# ---------------------------------------------------------------------------
PROJ_BLOCK_ENV = "T22_PROJ_BLOCK"
PROJ_BLOCK_ROWS_DEFAULT = 1056   # one image's full tile count (RT=11)
PROJ_BLOCK_MIN = 96
PROJ_BLOCK_CEILING = 1056        # <= it: one dispatch, no block


def proj_block_rows(m: int) -> int:
    """Rows per projection dispatch: 0 = one whole-shape dispatch."""
    raw = (os.environ.get(PROJ_BLOCK_ENV, "1") or "1").strip().lower()
    if raw in ("", "0", "off", "false", "no"):
        return 0
    if os.environ.get("T22_FASTGLUE", "0") != "1":
        return 0
    skip = {s.strip() for s in
            os.environ.get("T22_FASTGLUE_SKIP", "").split(",") if s.strip()}
    if "attn" in skip:
        return 0
    if int(m) <= PROJ_BLOCK_CEILING:
        return 0
    if raw in ("1", "on", "true", "yes"):
        return PROJ_BLOCK_ROWS_DEFAULT
    try:
        n = int(raw)
    except ValueError:
        n = PROJ_BLOCK_ROWS_DEFAULT
    return max(PROJ_BLOCK_MIN, ((n + 95) // 96) * 96)


def chunk_attention_bytes(m: int) -> int:
    """Host f32 working set: q/k/v + one score block (torch SDPA bound)."""
    qb = 512 if int(m) <= 8192 else 256
    return (qb * int(m) * T22_HEADS * 4
            + 3 * int(m) * T22_HEADS * T22_HD * 4)


def chunk_estimate(m: int, steps: int = 8, layers: int = T22_LAYERS) -> dict:
    """Honest machine-time estimate for the chunked path (device + host).

    Every term is an anchor, not a promise:
      projection 4 x 5.62 ms / 96 rows   (T22-BUDGET section 0)
      FFN        40.3 ms / 96 rows       (same)
      host glue  70 ms / 96 rows         (480p prototype, 2026-10-21)
      attention  4*M^2*HD*HEADS / 600 GFLOPS (torch f32, measured 190-700)
      barrier    1.7 ms per extra FFN block (T20)
    """
    seq = int(m)
    blocks = (seq + FFN_ROWS - 1) // FFN_ROWS
    proj_ms = 4.0 * blocks * T22_PROJ_MS_PER_96
    ffn_ms = blocks * T22_FFN_MS_PER_96
    barrier_ms = max(0, blocks - 1) * CHUNK_BARRIER_MS
    host_ms = blocks * CHUNK_HOST_MS_PER_BLOCK
    attn_ms = (4.0 * seq * seq * T22_HD * T22_HEADS
               / (CHUNK_ATTN_GFLOPS * 1e9) * 1e3)
    layer_ms = proj_ms + ffn_ms + barrier_ms + host_ms + attn_ms
    step_s = layer_ms * float(layers) / 1e3
    return {
        "mode": "chunked",
        "seq": seq,
        "blocks": blocks,
        "proj_ms_per_layer": round(proj_ms, 1),
        "ffn_ms_per_layer": round(ffn_ms, 1),
        "barrier_ms_per_layer": round(barrier_ms, 1),
        "host_ms_per_layer": round(host_ms, 1),
        "attn_ms_per_layer": round(attn_ms, 1),
        "layer_ms": round(layer_ms, 1),
        "step_s": round(step_s, 2),
        "s_8step": round(step_s * 8.0, 1),
        "h_8step": round(step_s * 8.0 / 3600.0, 2),
        "dispatches_per_step": int(layers * 5 * blocks),
        "dispatches_8step": int(layers * 5 * blocks * 8),
        "attention": "host f32 torch (QK^T+softmax+PV)",
        "device": "delivered M=96 projection + FFN + o projection, 96-row blocks",
        "anchor": ("480x848/M=1632 prototype 65.4 s/step (device 27.8, "
                   "staging 4.7, host 32.9); model +10-15 %"),
        "not_a_speed_claim": True,
    }


def refusal(h, w, why, m=None, b=None):
    """The one sentence an un-runnable shape gets (the _no_build shape).

    <run shape> (M=<m>) 未构建：<why, with the exact bytes/slots/rows>。
    build_shape._no_build() uses the same head, so the front end and the builder
    cannot describe the same shape two different ways.
    """
    b = b or budget(h, w)
    m = int(b["m"]) if m is None else int(m)
    return "%s (M=%d) 未构建：%s。" % (b.get("run_key") or requested_key(h, w),
                                      m, why)


def chunk_gate(h: int, w: int, b: dict = None, steps: int = 8):
    """(runnable, reason, estimate) for the chunked path at this shape.

    T22 adaptive: the FIRST question is no longer "does the host RAM hold it",
    it is "is there a legal 32-core placement for this M, are its images built,
    and do the product's host BOs cover it".  Every refusal quotes the binding
    constraint in bytes / slots / rows.
    """
    b = b or budget(h, w)
    m = int(b["m"])
    if m <= 304:
        return (False, refusal(h, w, ("M=%d is inside the delivered ip ceiling "
                                      "(SEQ<=304), so the native path is the "
                                      "one to use" % m), m, b), None)
    if m > MAX_M:
        return (False, refusal(h, w, ("M=%d is outside the product envelope "
                                      "%d..%d" % (m, MIN_M, MAX_M)), m, b), None)
    mem = chunk_attention_bytes(m)
    est = chunk_estimate(m, steps=steps)
    # ---- 1. a legal placement must exist ---------------------------------
    res = npu_plan_for(m)
    est["plan_key"] = res["plan"]["plan_key"] if res.get("ok") else None
    est["plan"] = plan_summary(res["plan"]) if res.get("ok") else None
    if not res.get("ok"):
        return (False, refusal(h, w, "无合法 32 核放置：%s" % res["reason"],
                               m, b), est)
    plan = res["plan"]
    est["plan_l1_max"] = plan["l1_max"]
    est["plan_rows"] = plan["Mblk"]
    est["plan_dispatches"] = plan["ndisp"]
    est["plan_tiles"] = plan["nm"]
    est["plan_per_col"] = plan["per_col"]
    # ---- 2. the host BOs must cover M ------------------------------------
    from . import npu_plan
    bo = npu_plan.bo_sizes(m)
    est["bo_x_bytes"] = bo["x_bytes"]
    est["bo_ff_bytes"] = bo["ff_bytes"]
    est["bo_rows"] = bo["rows_padded"]
    if bo["x_bytes"] > HOST_BO_MAX or bo["ff_bytes"] > HOST_BO_MAX:
        return (False, refusal(h, w, (
            "host BO %.2f GiB / %.2f GiB exceeds the %.0f GiB single-BO "
            "ceiling (M=%d x 3840 x 2 = %d B)"
            % (bo["x_bytes"] / 2**30, bo["ff_bytes"] / 2**30,
               HOST_BO_MAX / 2**30, m, bo["x_bytes"])), m, b), est)
    if any(os.environ.get(k, "0") == "1" for k in WHOLE_SHAPE_FLAGS):
        # T22 proj-block: above the ceiling the projection is dispatched in
        # BLOCKS, so the consumer reads ONE block and the BO has to hold ONE
        # block -- the whole-shape rule below is only right for the delivered
        # whole-shape dispatcher (T22_PROJ_BLOCK=0), and is kept for it.
        blk = proj_block_rows(m)
        have = _whole_shape_rows()
        est["proj_block_rows"] = blk or 0
        need = blk if blk else max(1056, ((m + 95) // 96) * 96)
        if have < need:
            if blk:
                why = ("投影镜像的 host BO 只覆盖 %d 行（%d B），本形状的投影块是 %d 行"
                       "（%d B = %d 行 x 3840 x 2）；启动时须置 T22_ADAPT_M=%d"
                       % (have, have * 3840 * 2, need, need * 3840 * 2, need, need))
            else:
                why = ("整形状镜像的 host BO 只覆盖 %d 行（%d B），本形状需要 %d 行"
                       "（%d B = %d 行 x 3840 x 2）；启动时须置 T22_ADAPT_M=%d"
                       % (have, have * 3840 * 2, need, need * 3840 * 2, need, need))
            return (False, refusal(h, w, why, m, b), est)
    # ---- 3. the adaptive path must be BUILT (a claim needs a build) ------
    if os.environ.get("ZIMAGE_ADAPT", "0") == "1":
        if plan_manifest(plan_key=plan["plan_key"]) is None:
            return (False, refusal(h, w, (
                "自适应镜像集 plan %s 未构建（gu/dn/proj 三个镜像，Mblk=%d，"
                "L1<=%d B）" % (plan["plan_key"], plan["Mblk"],
                               plan["l1_max"])), m, b), est)
    # ---- 3b. the two constraints the model above cannot see (2026-11-03) --
    # The refiner term compares the tensor the refiner will ACTUALLY allocate
    # (chunk_refiner_peak_bytes) against the ceiling, not the S x S it would
    # allocate without query blocking: 1080p's 7.44 GiB becomes one 0.5 GiB
    # block, and the 6 GiB ceiling is a statement about the largest single
    # allocation (the process peak commit measures ~4x it, which is why 7.44
    # died and 1.88 did not).
    tokens = int(b.get("image_tokens") or max(1, m - CAP_TOKENS))
    ref = chunk_refiner_bytes(m, tokens)
    ref_eff = chunk_refiner_peak_bytes(m, tokens)
    est["refiner_bytes"] = ref
    est["refiner_gib"] = round(ref / 2**30, 2)
    est["refiner_block_gib"] = round(ref_eff / 2**30, 2)
    est["refiner_blocked"] = bool(ref_eff != ref)
    ref_max = int(float(os.environ.get("ZIMAGE_CHUNK_REFINER_GB", "0") or 0) * 2**30
                  ) if os.environ.get("ZIMAGE_CHUNK_REFINER_GB") else CHUNK_REFINER_MAX_BYTES
    if ref_eff > ref_max:
        if not chunk_refiner_blocked():
            _how = ("a %dx%dx%dx4 B = %.2f GiB f32 score matrix (S=%d image "
                    "tokens), and T22_REF_QCHUNK=off disables the query "
                    "blocking that would bound it to %.2f GiB"
                    % (int(T22_HEADS), tokens, tokens, ref / 2**30, tokens,
                       _env_gb(REF_QBLOCK_ENV, REF_QBLOCK_GB)))
        else:
            _how = ("%.2f GiB for one query block of its score matrix (S=%d "
                    "image tokens, T22_REF_QBLOCK_GB=%g); it would need %.2f "
                    "GiB unblocked"
                    % (ref_eff / 2**30, tokens,
                       _env_gb(REF_QBLOCK_ENV, REF_QBLOCK_GB), ref / 2**30))
        return (False, refusal(h, w, (
            "CPU noise refiner needs %s, over the %.0f GiB single-allocation "
            "ceiling; measured: S=4096 (1.88 GiB tensor, 8.39 GiB peak commit) "
            "completes, S=8160 (7.44 GiB) dies with 'device loop ran out of "
            "memory' 42 s in (job 20261103-150606-10d726).  Free memory and "
            "raise ZIMAGE_CHUNK_REFINER_GB if you accept the risk"
            % (_how, ref_max / 2**30)), m, b), est)
    if fastglue_projection_active():
        blk = proj_block_rows(m)
        rows = blk if blk else fastglue_proj_rows()
        est["fastglue_proj_rows"] = rows
        if not blk and m > rows:
            # Only the WHOLE-SHAPE dispatcher hands the image more rows than it
            # has: its host BO must hold ceil(M/96)*96.  With the block on
            # (default) no M-dependent requirement remains -- each launch is one
            # block and _proj_join drains exactly that block.
            return (False, refusal(h, w, (
                "T22_FASTGLUE=1 dispatches the projection through the rt17 hook, "
                "whose host BO holds %d rows (%d x 3840 x 2 = %d B); this shape "
                "needs M=%d rows -> 'attempting to write past buffer size' "
                "(measured: job 20261103-063607-d60112 died 20.9 s in).  Unset "
                "T22_FASTGLUE (the delivered chunked path covers any M), set "
                "T22_PROJ_BLOCK=1 (the default: one block per dispatch) or raise "
                "T22_ADAPT_CHUNK=%d"
                % (rows, rows, rows * 3840 * 2, m, m)), m, b), est)
    # ---- 4. the delivered chunked-path limits ----------------------------
    if mem > CHUNK_MAX_BYTES:
        return (False, refusal(h, w, (
            "host attention needs %.1f GiB > %.0f GiB budget"
            % (mem / 2**30, CHUNK_MAX_BYTES / 2**30)), m, b), est)
    if est["s_8step"] > CHUNK_MAX_8STEP_S:
        return (False, refusal(h, w, (
            "projected %.0f s for 8 steps > %.0f s ceiling"
            % (est["s_8step"], CHUNK_MAX_8STEP_S)), m, b), est)
    try:
        import importlib.util
        if importlib.util.find_spec("torch") is None:
            return (False, refusal(h, w, "torch is not importable in this "
                                         "interpreter", m, b), est)
    except Exception:                                          # noqa: BLE001
        return (False, refusal(h, w, "torch availability check failed",
                               m, b), est)
    return True, "", est


def chunk_reason(h: int, w: int, est: dict) -> str:
    h, w = int(h), int(w)
    pl = est.get("plan") or {}
    plan_bit = ""
    if pl:
        plan_bit = ("放置 plan %s（m=%d kk=%d，Mblk=%d，%d 次 dispatch，"
                    "L1<=%d B，%s）。"
                    % (pl["plan_key"], pl["m"], pl["kk"], pl["Mblk"],
                       pl["ndisp"], pl["l1_max"],
                       "已构建" if est.get("plan_built") else "未构建"))
    return (f"{h}×{w} 可跑（分块·慢）：无专用内核，但可用已交付 M=96 的 "
            f"投影/FFN/o 投影镜像逐 96 行块执行，attention 在 host f32 计算；"
            f"预估 {est['step_s']:.0f} s/步、8 步约 {est['h_8step']:.2f} h"
            f"（{est['dispatches_8step']:,} dispatch）。{plan_bit}"
            f"非专用内核，未过 480p 以外的门；机时是估计，不是速度声明。")


def t22_fmt_l1(n: int) -> str:
    """Thousands-separated form (196,096 B for 512², 83,456 B for 256×384)."""
    return f"{int(n):,}"


# The profiles that actually have device images + host constants in this tree.
BUILT = {
    "128x128": {
        "height": 128, "width": 128, "m": 96,
        "label": "128 x 128 (M=96, delivered)",
        "ready": True,
        "images": ["t7_p5", "t7_ip", "fs_ffn_j5"],
        "reason": "",
    },
    # T16 P0-5 slice 1: same M=96 (64 image tokens), only the host 2D geometry
    # changes (patch grid 4x16, latent 8x32, RoPE ids, unpatchify, initial
    # latent).  No image is rebuilt; the delivered M=96 images are reused.
    "64x256": {
        "height": 64, "width": 256, "m": 96,
        "label": "64 x 256 (M=96, host geometry build)",
        "ready": True,
        "images": ["t7_p5", "t7_ip", "fs_ffn_j5"],
        "reason": "",
        "note": "same M as 128x128; host patch grid 4x16; no new images",
    },
    # T18 P0-5: the first NEW-M shape.  The delivered FFN source compiles at
    # M=128 with its shipped knobs (--kgu 8); the projection is the delivered
    # m=96 image run in 96-row blocks; the ip image is compiled with ND=128
    # keys (64 queries per block).  Gates (see webui/evidence/T18-*.md):
    # run-to-run latents byte-identical; vs host f32 corr 0.99422 / 29.86 dB.
    "192x128": {
        "height": 192, "width": 128, "m": 128,
        "label": "192 x 128 (M=128, row-block build)",
        "ready": True,
        "images": ["t7_p5", "t18_ip_m64nd128", "t18_m128"],
        "reason": "",
        "note": "projection 2x96-row blocks; ip 2x64 queries against ND=128 keys",
    },
    # T18 padding branch: 72 real image tokens + 24 x_pad_token rows (pos 0,0,0)
    # -> padded 96 -> SEQ=128, so it reuses the M=128 images above; only the
    # host token padding/unpatchify changes.  Gates (T18 doc §5.3):
    # run-to-run sha16 d1a5f5fa92faaff1; vs host f32 corr 0.98241 / 25.00 dB.
    "144x128": {
        "height": 144, "width": 128, "m": 128,
        "label": "144 x 128 (M=128, 72+24 padded tokens)",
        "ready": True,
        "images": ["t7_p5", "t18_ip_m64nd128", "t18_m128"],
        "reason": "",
        "note": "image-token padding: 72 real + 24 x_pad_token rows; unpatchify 72",
    },
    # T22 pad_crop gate shapes (2026-10-21): the two run sizes the crop gate
    # uses.  Both reuse already-gated M images; the bridge derives the 2D
    # geometry from H/W, and the shapes ran 8 steps on the device during the
    # gate (144x144 latents == its 130x130 request; 208x128 == its 200x120
    # request, byte-identical).  They are recorded here so a fresh checkout can
    # prepare their cache without an untracked manifest.
    "144x144": {
        "height": 144, "width": 144, "m": 128,
        "label": "144 x 144 (M=128, pad_crop gate shape)",
        "ready": True,
        "images": ["t7_p5", "t18_ip_m64nd128", "t18_m128"],
        "reason": "",
        "note": "T22 pad_crop gate shape; reuses the gated M=128 images",
    },
    "128x208": {
        "height": 128, "width": 208, "m": 160,
        "label": "128 x 208 (M=160, pad_crop gate shape)",
        "ready": True,
        "images": ["t7_p5", "t18_ip_m32nd160", "fs_ffn_j5"],
        "reason": "",
        "note": "T22 pad_crop gate shape; landscape 200x120 WxH request",
    },
    "208x128": {
        "height": 208, "width": 128, "m": 160,
        "label": "208 x 128 (M=160, pad_crop gate shape)",
        "ready": True,
        "images": ["t7_p5", "t18_ip_m32nd160", "fs_ffn_j5"],
        "reason": "",
        "note": "T22 pad_crop gate shape; reuses the gated M=160 images",
    },
    # T20: 128x256 -> 8x16 patch grid -> 128 image tokens (already a multiple
    # of 32, no padding) -> SEQ = 32 + 128 = 160.  Everything that scales:
    #   ip    ND=160 keys with Mq=32 (L1 8,192+23,040+10,240+6,144 = 57,616 B
    #         <= 64 KB; 5 query blocks), perm (32,128) from the kernel;
    #   FFN   2 x 96-row blocks of the delivered t11_nh5, with the T20 relaunch
    #         barrier between them (t18_run.Dev.barrier -- without it the second
    #         launch of the same image never completes).
    # Gates: webui/evidence/T20-128x256.md.
    "128x256": {
        "height": 128, "width": 256, "m": 160,
        "label": "128 x 256 (M=160, row-block build)",
        "ready": True,
        "images": ["t7_p5", "t18_ip_m32nd160", "fs_ffn_j5"],
        "reason": "",
        "note": ("projection 2x96 rows; ip 5x32 queries against ND=160 keys; "
                 "FFN 2x96 rows + T20 relaunch barrier"),
    },
    # T19: the last outstanding shape.  A single M=288 FFN dispatch does not
    # complete on this device (T18 §6: standalone ERT_CMD_STATE_TIMEOUT), so
    # the FFN runs the delivered M=96 image in 3 row blocks; each block
    # dispatches through the intermittent ERT state-8 (retry + pass).  Two
    # independent 8-step runs landed on byte-identical latents
    # (683050a95f4bd357) even though their retry patterns were completely
    # different (200 vs 264 retries).  Numerically the shape matches the host
    # f32 reference after one step to corr 0.999964 -- the delivered 128x128
    # path measures 0.999959 with the same metric (T19 doc §3).
    "256x256": {
        "height": 256, "width": 256, "m": 288,
        "label": "256 x 256 (M=288, FFN 3x96-row blocks)",
        "ready": True,
        "images": ["t7_p5", "t18_ip_m16nd288", "fs_ffn_j5"],
        "reason": "",
        "note": ("projection 3x96 rows; ip 18x16 queries against ND=288 keys; "
                 "FFN 3x96 rows (the M=288 image times out); run-to-run "
                 "683050a95f4bd357; one-step vs host f32 corr 0.999964"),
    },
}

PRESETS = [
    {"label": "128 x 128", "height": 128, "width": 128},
    {"label": "256 x 256", "height": 256, "width": 256},
    {"label": "512 x 512", "height": 512, "width": 512},
    # T22 prodlarge: the spectrum end points.  Both are ready=false today --
    # the delivered ip ceiling is SEQ=304 (T22-BUDGET.md §1) and everything
    # above needs the key-blocked schedule -- but the product must be able to
    # SAY that, with the exact M / L1 / dispatch arithmetic.
    {"label": "1024 x 1024", "height": 1024, "width": 1024},
    {"label": "2048 x 2048", "height": 2048, "width": 2048},
    {"label": "192 x 128", "height": 192, "width": 128},
    {"label": "144 x 128 (padding)", "height": 144, "width": 128},
    {"label": "128 x 256", "height": 128, "width": 256},
    {"label": "64 x 256", "height": 64, "width": 256},
    {"label": "256 x 384", "height": 256, "width": 384},
]


def _legacy_reason(h, w) -> str:
    """The frozen paper reason of a legacy fixed profile (config.RESOLUTIONS).

    The 512² sentence ("512² 需新内核设计，暂不可用：…") is the product's
    reference wording for an unbuilt size: the front end quotes it and the
    README points at it.  It therefore keeps exactly ONE home, config.py, and
    is imported lazily here -- config is a leaf module and must not import
    shapes.
    """
    if h != w:
        return ""
    try:
        from . import config
        meta = (getattr(config, "RESOLUTIONS", {}) or {}).get(int(h)) or {}
    except Exception:  # noqa: BLE001
        return ""
    if meta.get("ready") is False and meta.get("reason"):
        return str(meta["reason"])
    return ""


def reason_text(h, w, b=None) -> str:
    """The precise user-facing reason this shape has no build in this tree.

    One sentence in the 512² spec's shape: <shape> + what is missing + the
    exact arithmetic + "本树未构建，不可用。".  Three regimes, each with its own
    truth (never a generic "needs 6 images" for a shape that does not):
      A. the SEQ is already delivered (M in 96/128/160/288): the device images
         exist, only the host geometry (perm table + constants) is missing;
      B. today's ip image still compiles at this SEQ with a smaller query
         block: one compile + a perm table + the four gates are missing;
      C. nothing compiles: the key-blocked / online-softmax design is missing
         (the T22 campaign-1 design, still ungated -> NOT SHIPPABLE).
    """
    h, w = int(h), int(w)
    b = b or budget(h, w)
    hp, wp = run_dims(h, w)
    if (hp, wp) != (h, w):
        # The padded run has its own exact plan sentence; the request sentence
        # adds the delivery rule: decode the padded frame, keep [0:H,0:W].
        base = reason_text(hp, wp, b)
        return (f"请求 {h}×{w} → 实跑 {hp}×{wp}（内部补零到 16 倍数），"
                f"解码后按 [0:H,0:W] 裁回 {h}×{w}（右下裁掉多余像素）。"
                f"实跑尺寸的计划：{base}")
    legacy = _legacy_reason(h, w)
    if legacy:
        return legacy
    head = f"{h}²" if h == w else f"{h}×{w}"
    m, tok = int(b["m"]), int(b["image_tokens"])
    gh, gw = b["patch_grid"]
    if m in T22_DELIVERED:
        return (f"{head} 需要新的 host 几何，暂不可用：M={m}（{tok} image tokens "
                f"+ 32 caption）与已交付的 SEQ={m} 镜像同 M，可复用该 M 的 "
                f"FFN/ip/projection 镜像，但本树没有 patch grid {gh}×{gw} 的 perm "
                f"表与 host 常量（RoPE、SBG/walk、image-token padding），也未过"
                f"四门禁，本树未构建，不可用。")
    l1 = t22_cur_ip_l1(16, m)
    mq = t22_cur_ip_mq(m)
    if mq:
        return (f"{head} 需要新的 perm 与 ip，暂不可用：M={m}（{tok} image "
                f"tokens + 32 caption）不在已交付的 SEQ 96/128/160/288 之内；今天"
                f"的 ip 镜像可在 SEQ={m} 用 Mq={mq} 重编译（L1 {t22_fmt_l1(l1)} B "
                f"≤ 64 KB），但本树既没编译、也没有 {mq}x128 的 perm 表，四门禁"
                f"未跑，本树未构建，不可用。")
    return (f"{head} 需新内核设计，暂不可用：M={m}（{tok} image tokens + 32 "
            f"caption）远超已编译的 M=96 镜像；ip 需按 key 分块 + 在线 "
            f"softmax 的新内核（M={m} 的 L1 预算 {t22_fmt_l1(l1)} B > 64 KB），"
            f"本树未构建，不可用。")


_WEBUI_DIR = pathlib.Path(__file__).resolve().parent


def _built_from_manifest(h, w):
    """A shape registry row written by webui.build_shape after its audit.

    T22 pad_crop: the static BUILT table cannot know about a shape built after
    this file shipped (144x144 / 208x128 are the two gate shapes).  A shape
    manifest is the builder's own record: it exists only after the per-shape
    cache + audit, and it names the exact FFN/ip/projection images.
    """
    key = f"{int(h)}x{int(w)}"
    p = _WEBUI_DIR / "shape_manifests" / (key + ".json")
    try:
        man = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if str(man.get("shape") or "") != key:
        return None
    try:
        b = budget(h, w)
        m = int(man.get("m"))
    except Exception:  # noqa: BLE001
        return None
    if m != int(b["m"]):
        return None
    imgs = man.get("images") or {}
    return {"height": int(h), "width": int(w), "m": m,
            "label": f"{int(h)} x {int(w)} (M={m}, built shape manifest)",
            "ready": True,
            "images": [str(v) for v in imgs.values()] or ["shape manifest"],
            "reason": ""}


def resolve(h, w) -> dict:
    """Validate a request and return its plan (built or build_required)."""
    h, w = int(h), int(w)
    b = budget(h, w)
    key = profile_key(h, w)
    built = BUILT.get(key) or _built_from_manifest(int(b["height"]),
                                                   int(b["width"]))
    plan = dict(b)
    plan["key"] = key
    plan["requested_key"] = requested_key(h, w)
    plan["run_key"] = key
    plan["ready"] = bool(built and built.get("ready"))
    plan["output_height"] = h
    plan["output_width"] = w
    plan["crop_applied"] = bool(b["pad_applied"])
    plan["crop"] = {"policy": "bottom-right-trim", "box": [0, 0, w, h],
                    "run_height": b["height"], "run_width": b["width"]}
    # T22 campaign 1: M, projected dispatches and projected device time.
    # Present on EVERY row (ready or not) so the panel can compare a delivered
    # size with a spectrum size on the same anchors.
    plan["budget_t22"] = t22_budget(b["m"])
    m_ = int(b["m"])
    # T22 chunk_path: a shape with no dedicated image can still run on the
    # delivered M=96 kernels + host attention.  This is a separate execution
    # mode, not "ready": ready still means a gated native image lives here.
    chunk_ok, chunk_why, chunk_est = chunk_gate(h, w, b)
    plan["chunked_runnable"] = bool(chunk_ok)
    plan["chunk_gate_reason"] = chunk_why
    plan["chunk_estimate"] = chunk_est
    # T22 adaptive: the 32-core placement this M resolves to, and whether its
    # image set is BUILT.  Present on EVERY row (ready or not) so the panel can
    # show the geometry the NPU would actually run.
    _est = chunk_est or {}
    plan["npu_plan"] = _est.get("plan")
    plan["npu_plan_key"] = _est.get("plan_key")
    plan["npu_plan_built"] = bool(_est.get("plan_key")
                                  and plan_manifest(plan_key=_est["plan_key"]))
    if chunk_est:
        chunk_est["plan_built"] = plan["npu_plan_built"]
    plan["exec_mode"] = ("native" if plan["ready"]
                         else ("chunked" if chunk_ok else "none"))
    if chunk_ok:
        plan["chunk_reason"] = chunk_reason(h, w, chunk_est)
    if plan["ready"]:
        plan.update({k: built[k] for k in ("label", "images", "reason")})
        return plan
    rh, rw = int(b["height"]), int(b["width"])
    run_note = (f"{h} x {w} -> {rh} x {rw}"
                if b["pad_applied"] else f"{h} x {w}")
    if plan["chunked_runnable"]:
        try:
            from . import config
            ffn = config.FFN_IMAGE
        except Exception:  # noqa: BLE001
            ffn = "fs_ffn_j5"
        plan["label"] = f"{run_note} (M={m_}, chunked 96-row path)"
        plan["images"] = ["t7_p5", "t7_ip", ffn]
        plan["missing"] = []
        plan["needs_build"] = False
        plan["reason"] = plan["chunk_reason"]
        return plan
    plan["label"] = (f"{run_note} (M={m_}, needs a host-geometry build)"
                     if m_ in T22_DELIVERED else
                     (f"{run_note} (M={m_}, needs one ip compile + perms)"
                      if t22_cur_ip_mq(m_) else
                      f"{run_note} (M={m_}, needs a rebuild)"))
    plan["images"] = ["FFN", "4 projections", "IP"]
    # T22 prodlarge: the missing list follows the SAME three regimes as
    # reason_text().  A shape whose SEQ is already delivered must not be told
    # it needs six new images it does not need.
    m, gh, gw = int(b["m"]), b["patch_grid"][0], b["patch_grid"][1]
    host = ("host constants: patch grid %dx%d, RoPE index map, SBG/walk, "
            "image-token padding to %d (x_pad_token, pos (0,0,0))"
            % (gh, gw, b["padded_tokens"]))
    gates = ("per-shape model gate (bit-exact), audit all cores, run-to-run "
             "byte gate, and a first image side-by-side with the host f32 "
             "reference")
    if m in T22_DELIVERED:
        plan["images"] = ["reuse the delivered SEQ=%d images (FFN + ip + "
                          "projection)" % m]
        plan["missing"] = [
            "permutations for patch grid %dx%d (the delivered ones are "
            "96x128 / 96x64)" % (gh, gw),
            host,
            gates,
        ]
    elif t22_cur_ip_mq(m):
        mq = t22_cur_ip_mq(m)
        plan["missing"] = [
            "1 ip image at ND=%d, Mq=%d (t18_attn_compile.py --M %d --ND %d)"
            % (m, mq, mq, m),
            "permutations for Mq=%d (the delivered ones are 96x128 / 96x64)"
            % mq,
            host,
            gates,
        ]
    else:
        plan["missing"] = [
            "6 device images at M=%d (FFN + 4 projections + IP)" % m,
            "permutations for M=%d (the delivered ones are 96x128 / 96x64)" % m,
            host,
            gates,
        ]
    if b["m"] > 96:
        # T16 reported an L1 placement wall at M=128 for the delivered FFN.
        # T18 re-measured it: that build passed --M 128 but used t4_compile's
        # default kgu=64, whose midB consumer object is m*kgu*2 = 2,048 B.
        # The SHIPPED knob set (--kgu 8, the one t11_nh5 ships with) compiles
        # at M=128 and M=288 unchanged: same source, only row_tiles changes.
        plan["missing"].append(
            "NOTE (T18): the T16 'FFN does not place at M=128' result was a "
            "kgu=64 artifact (2,048 B midB object). The shipped kgu=8 design "
            "compiles unchanged at M=128/288 (t18_m128 / t18_m288).")
    if b["m"] == 288:
        plan["missing"].append(
            "T18 status: 256x256's images (ip ND=288), perms and cache ARE built "
            "and audited, but the device run is blocked by intermittent ERT "
            "state-8 on FFN block dispatches (~2 of 3 blocks per layer; retries "
            "converge at ~34 s/layer, ~2.3 h for 8 steps). Not ready until that "
            "is resolved and the four gates are re-run.")
    plan["reason"] = reason_text(h, w, b)
    return plan


def list_shapes() -> list:
    out = []
    for p in PRESETS:
        try:
            out.append(resolve(p["height"], p["width"]))
        except ShapeError as exc:
            out.append({"height": p["height"], "width": p["width"],
                        "label": p["label"], "ready": False,
                        "error": str(exc)})
    return out


# ---------------------------------------------------------------------------
# T22 seed_res (2026-10-21) / T22 pad_crop (2026-10-21): product presets.
#
# The product owner wants the usual tier names (480p / 720p / 1080p / 2K) in
# the three common aspect ratios.  A tier now names its STANDARD resolution
# (1920x1080, 854x480, ...), not a pre-snapped one: the request is the standard
# size, the internal run size is H_pad/W_pad = ceil(side/16)*16 (1080 -> 1088,
# 854 -> 864).  9:16 is the exact width/height swap of 16:9; 1:1 uses the
# square edge of the tier.  The old nearest-16 mapping is kept in snap16() for
# callers that still need it, but preset_pairs() no longer applies it.
#
# DELIVERY: a padded run is decoded at H_pad x W_pad and the PNG is trimmed to
# [0:H, 0:W], so the output is exactly the requested size.  Rows still report
# readiness and the full exact budget of the padded run; the standard request
# is never rounded down to a smaller frame.
#
# HONESTY: no preset below is a delivered device image.  The built envelope is
# 128x128 (M=96) plus 64x256 / 192x128 / 144x128 / 128x256 / 256x256; every
# 480p+ row is a BUILD.  preset_list() says so per row, it never marks a
# preset ready just because its arithmetic fits.
# ---------------------------------------------------------------------------
PRESET_GROUPS = ("16:9", "9:16", "1:1")
# nominal (standard) tier sizes: (wide_w, wide_h, square_edge)
PRESET_TIERS = (
    {"label": "480p", "wide": (854, 480), "square": 480},
    {"label": "720p", "wide": (1280, 720), "square": 720},
    {"label": "1080p", "wide": (1920, 1080), "square": 1080},
    {"label": "2K", "wide": (2048, 1152), "square": 2048},
)

FEASIBILITY_LABELS = {
    "runnable": "可跑",
    "chunked_runnable": "可跑(分块·慢)",
    "needs_build": "需构建",
    "out_of_range": "超界",
}


def snap16(value) -> int:
    """LEGACY nearest multiple of 16 (1080 -> 1088, 854 -> 848).

    Neither preset_pairs() nor a job request uses this any more: arbitrary
    requests are padded UP by pad16().  Kept because older callers/tests name
    the function and its half-up contract.
    """
    return (int(value) + 8) // 16 * 16


def preset_pairs() -> list:
    """[(group, label, height, width), ...] as STANDARD requested sizes."""
    out = []
    for group in PRESET_GROUPS:
        for tier in PRESET_TIERS:
            ww, wh = tier["wide"]
            if group == "16:9":
                w, h = ww, wh
            elif group == "9:16":
                w, h = wh, ww
            else:  # 1:1
                w = h = tier["square"]
            hp, wp = run_dims(h, w)          # validates 64..2048 and pads
            if not (MIN_HW <= hp <= MAX_HW and MIN_HW <= wp <= MAX_HW):
                raise ShapeError(f"preset {group} {tier['label']}: "
                                 f"{wp}x{hp} is outside {MIN_HW}..{MAX_HW}")
            out.append((group, tier["label"], h, w))
    return out


def preset_feasibility(h, w) -> dict:
    """Classify one preset: runnable / needs_build / out_of_range.

    "runnable" means a gated device image exists in THIS tree for the PADDED
    run shape (BUILT + ready=True).  The request itself may be non-16-multiple;
    its output is the requested H x W after the crop.
    """
    try:
        plan = resolve(int(h), int(w))
        b = budget(int(h), int(w))
    except ShapeError as exc:
        return {"feasibility": "out_of_range",
                "feasibility_label": FEASIBILITY_LABELS["out_of_range"],
                "ready": False, "m": None, "reason": str(exc),
                "budget": None}
    m = int(plan["m"])
    if not (MIN_M <= m <= MAX_M):
        return {"feasibility": "out_of_range",
                "feasibility_label": FEASIBILITY_LABELS["out_of_range"],
                "ready": False, "m": m,
                "reason": f"M={m} is outside {MIN_M}..{MAX_M}",
                "budget": b}
    if plan.get("ready"):
        status = "runnable"
    elif plan.get("chunked_runnable"):
        status = "chunked_runnable"
    else:
        status = "needs_build"
    return {"feasibility": status,
            "feasibility_label": FEASIBILITY_LABELS[status],
            "ready": bool(plan.get("ready")),
            "chunked_runnable": bool(plan.get("chunked_runnable")),
            "exec_mode": plan.get("exec_mode", "none"),
            "chunk_estimate": plan.get("chunk_estimate"),
            "m": m,
            "reason": str(plan.get("reason") or ""),
            "budget": b}


def preset_list() -> list:
    """GET /api/presets payload: 3 aspect groups x 4 tiers, budget annotated.

    Contract fields group/label/height/width are now the STANDARD requested
    size.  Every row also carries:
      native_requested  true (the tier's standard resolution is what is asked)
      requested         {height,width} the client should put in the form
      padded            {height,width} the internal 16-px run size
      pad_applied       requested != padded
      run_height/run_width  same as padded (flat convenience fields)
    plus m/ready/feasibility/reason/budget and the compact t22 projection.
    """
    rows = []
    for group, label, h, w in preset_pairs():
        try:
            plan = resolve(h, w)
        except ShapeError:
            plan = {}
        info = preset_feasibility(h, w)
        t22 = plan.get("budget_t22") or {}
        b = info.get("budget") or {}
        rh, rw = int(b.get("height") or h), int(b.get("width") or w)
        rows.append({
            "group": group,
            "label": label,
            "height": h,
            "width": w,
            "native_requested": True,
            "requested": {"height": h, "width": w},
            "padded": {"height": rh, "width": rw},
            "run_height": rh,
            "run_width": rw,
            "pad_applied": bool(b.get("pad_applied", (rh, rw) != (h, w))),
            "m": info.get("m"),
            "ready": bool(info.get("ready")),
            "chunked_runnable": bool(info.get("chunked_runnable")),
            "exec_mode": info.get("exec_mode", "none"),
            "feasibility": info["feasibility"],
            "feasibility_label": info["feasibility_label"],
            "reason": info.get("reason") or "",
            "chunk": info.get("chunk_estimate"),
            "budget": info.get("budget"),
            "t22": {
                "seq": t22.get("t22_seq"),
                "dev_s_per_step": t22.get("t22_dev_s_per_step"),
                "dev_h_8step": t22.get("t22_dev_h_8step"),
                "dispatches_8step": t22.get("t22_dispatches_8step"),
                "cur_ip": t22.get("t22_cur_ip"),
                "l1_over": t22.get("t22_l1_over"),
                "note": t22.get("t22_note", ""),
            } if t22 else None,
        })
    return rows
