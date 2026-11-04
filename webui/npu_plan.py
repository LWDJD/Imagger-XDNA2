# SPDX-License-Identifier: MIT
"""npu_plan.py -- the resolution-adaptive NPU tile planner (PURE FUNCTION).

WHY THIS FILE EXISTS
====================
Before it, three hardcoded facts decided whether a (H, W) request could run on
the NPU:

  * webui/build_shape.py:51 COMPILE had recipes for exactly M = 128 / 160 / 288;
  * webui/shapes.py chunk_gate() answered "runnable" from an L1 + host-RAM model
    that never looked at those recipes or at the product's host BO sizes, so
    1024^2 was promised at "~1.5 h" while the projection input BO was sized for
    11 x 96 = 1056 rows (imaggers/image/t18_run.py:377) and the run died with
    "attempting to write past buffer size";
  * 384^2 (M=608) and 768^2 (M=2336) died before that on "N % 96 != 0", i.e. on
    a 96-row assumption nobody had made explicit.

This module answers, for ANY token count M, in one place and with no side
effects:

    is there a legal 32-core placement for this M?
    if yes -> which (m, n, kk, tile map, dispatch tiling, L1 bytes, BO bytes)?
    if no  -> which constraint binds, quoted in bytes / slots / tokens?

It is a PURE FUNCTION: no filesystem, no numpy, no toolchain, no device, no
randomness.  Every number it uses is a MEASURED constant (see CONSTANTS) and
_selftest() re-derives the four measured L1 points to prove the model still
reproduces them.

THE THREE CONSUMERS
===================
    webui/shapes.py            chunk_gate() -> an honest yes/no + the reason
    imaggers/util/place/npu_synth.py
                               plan -> build_sweep.py VARIANT entries -> build
    imaggers/image/t18_run.py  bo_sizes() -> host BO bytes instead of the
                               512^2 literals

GEOMETRY VOCABULARY (the same letters the designs use)
======================================================
    M       rows of the GEMM = unified sequence length = 32 caption + image tokens
    K       reduction length: 3840 (projections, FFN gate/up), 10240 (FFN down)
    N       output width:     10240 (FFN gate/up), 3840 (down, projections)
    m       rows per M-tile (dim_m of the mm micro-kernel)
    n       output columns per tile (dim_n); ntiles = N / n
    kk      K blocking = dim_k of the micro-kernel; nkk = K / kk k-steps
    cols    8 columns; cpc 4 cores per column -> 32 (column, core) slots
    T       how many M-tiles one core owns (1 for ffn_xb, >1 for ffn_t33)
    Mblk    rows one DISPATCH covers; ndisp = ceil(M / Mblk)
"""
from __future__ import annotations

import hashlib
import json

# ===========================================================================
# CONSTANTS -- every one of them is measured, none is a guess
# ===========================================================================
L1_BYTES = 64 * 1024          # per-core L1 data budget (T22_L1 / ffn_design_xb)
STACK_BYTES = 0x800           # 2,048 B stack reservation of the delivered ip/FFN
BD_MAX_BYTES = 16383 * 4      # aie.dma_bd: 16,383 32-bit words is the maximum
COLS = 8                      # the array is 8 columns
CPC = 4                       # 4 GEMM cores per column
SLOTS = COLS * CPC            # 32 (column, core) slots
M_ALIGN = 16                  # mm_bfp.cc static_assert(dim_m % 16 == 0)
N_ALIGN = 16                  # mm_bfp.cc static_assert(dim_n % 16 == 0)
KK_MEASURED = (8, 16, 32)     # verified kk values whose packers exist (pack_kk)
KK_DMA_GATED = 64             # measured, but DMA-gated: do not plan with it

# The three GEMMs one DiT layer needs from the placement image set.
# name, K, N.  "gu" serves gate AND up (same shape, different weights);
# "proj" serves all four projections (q, k, v, o).
GEMMS = (("gu", 3840, 10240), ("dn", 10240, 3840), ("proj", 3840, 3840))

# Measured array rates at N=10240 (Q1Q2Q3-REPORT.md / the k-blocked round):
# kk=8 -> 3.78, kk=32 -> 5.12, kk=64 -> 5.59 TFLOP/s.  kk=16 was not measured
# at N=10240; 4.60 is the midpoint of 3.78/5.12 and is used only for costing.
RATE_TFLOPS = {8: 3.78, 16: 4.60, 32: 5.12}
BW_GBPS = 42.0                # measured DRAM stream (SWEEP-MN-REPORT.md 6)
DISP_MS = 1.0                 # per-dispatch charge (BO staging is 2.5 ms in the
                              # product path, 0 in a burst; 1.0 is the midpoint)
LAYERS = 30                   # DiT blocks per step (T22_LAYERS)

# The product envelope (kept identical to webui/shapes.py so the two cannot
# drift; shapes.py checks it in its own selftest).
MIN_M = 64
MAX_M = 32 + 16384            # 2048^2 -> 128x128 image tokens -> M=16416


class PlanError(ValueError):
    """Raised only by the strict entry points; plan() returns a refusal."""


# ===========================================================================
# L1 and payload arithmetic -- the specification the designs assert
# ===========================================================================
def c_leaf_bytes(m, n, mixed=True):
    """Bytes of one (m, n) C tile: bf16 when mixed, 9/8 BFP16 when pure."""
    return m * n * 2 if mixed else 9 * (m * n // 8)


def a_leaf_bytes(m, kk, mixed=True):
    """Bytes of one (m, kk) A tile: bf16 when mixed, 9/8 BFP16 when pure."""
    return m * kk * 2 if mixed else 9 * (m * kk // 8)


def b_obj_bytes(kk, n):
    """Bytes of one (kk, n) B object: always v8bfp16ebs8, 9 bytes per 8 values."""
    return 9 * (kk * n // 8)


def l1_bytes(m, n, kk, T=1, *, mixed=True, adepth=1, bdepth=1):
    """Per-core L1 for the busiest core: T C leaves + depth B + depth A + stack.

    THE A TERM IS T*a_leaf, NOT a_leaf.  A core that owns T M-tiles acquires ONE
    A element of T*a_leaf bytes (ffn_design_t33._core_fn splits the parent
    element at sum(T[:p])*a_leaf), so the object the placer must fit is
    T*a_leaf.  ffn_design_xb's assert is a_leaf because there T == 1 by
    construction; t33_plan.worst_l1() writes adepth*a_leaf as well and is
    therefore (T-1)*a_leaf SHORT -- at T=2 by 2,048 B, at T=7 by 6,144 B.
    That is not a theory: M=3168 m=16 kk=32 n=240 T=7 was planned at 65,472 B
    by the short formula and aiecc refused it with
        "rowA7_0_cons_buff_0 needs 7168 bytes and this tile has no room left"
    (7*1,024 = 7,168).  This function is the corrected gate.

    The measured points reproduce exactly:

        l1_bytes(32, 512, 32)          == 55,296   (accepted, kk32_m32_n512)
        l1_bytes(32, 384, 64)          == 58,368   (accepted, kk64_m32_n384)
        l1_bytes(32, 640, 32)          == 68,096   (REFUSED, q1_kk32_n640_gu)
        l1_bytes(32, 512, 64)          == 75,776   (REFUSED, kk64_b1_m32_n512;
                                                    the literature quotes
                                                    73,728 -- it omits the
                                                    2,048 B stack)
        l1_bytes(32, 320, 32, T=2)     == 58,624   (accepted, the delivered
                                                    t33 form; the design's own
                                                    worst_l1 says 56,576)
        l1_bytes(16, 240, 32, T=7)     == 71,616   (REFUSED by aiecc, M=3168)
    """
    return (T * c_leaf_bytes(m, n, mixed)
            + bdepth * b_obj_bytes(kk, n)
            + adepth * T * a_leaf_bytes(m, kk, mixed)
            + STACK_BYTES)


def n_cap(m, kk, T, *, mixed=True, adepth=1, bdepth=1, margin=0):
    """Largest n the L1 budget admits at this (m, kk, T).

    margin leaves room for the placer's bank fragmentation (a buffer may not
    straddle the 4 x 16 KB banks freely); 0 reproduces the design's own
    budget, which every accepted image in the record satisfies.
    """
    per_n = T * (2 * m if mixed else 9 * (m // 8)) + bdepth * 9 * kk // 8
    room = (L1_BYTES - STACK_BYTES - margin
            - adepth * T * a_leaf_bytes(m, kk, mixed))
    return max(0, room // per_n)


def n_cap_bd(m, mixed=True):
    """Largest n whose C leaf still fits ONE buffer descriptor."""
    return BD_MAX_BYTES // (2 * m if mixed else 9 * (m // 8))


# ===========================================================================
# THE FUSED gu (silu(gate)*up inside the image) -- what it costs in L1
# ===========================================================================
# The delivered design does this fusion today (imaggers/devkill/swiglu_rt11.py):
# ONE image whose B object is the pair [gate slice | up slice], two mm_bfp
# calls on the SAME A object (offset 0 = gate, offset B_OBJ = up), and the
# epilogue silu_sized(gate -> out) + mul_sized(out, up -> out).  It does NOT
# assume kk == 8 -- its assert is "k in (8, 32, 64)" and the shipped image runs
# k=64 -- so the fusion is legal at kk=32; what it changes is L1, because three
# C tiles are alive at once (gate, up, and the silu destination):
#
#   pure gu    C(m,n) + B(kk,n) + A(m,kk) + stack
#   fused gu   3 C(m,W) + 2 B(kk,W) + A(m,kk) + stack    W = fused width / tile
#
# At m=48, kk=32 the plan's own gu n=320 gives 3*30,720 + 2*11,520 + 3,072 +
# 2,048 = 122,880 B, so a fused image at the plan's n cannot exist and the
# fused plan MUST pick a smaller n.  ntiles = N/W then grows, which doubles the
# A re-streaming per dispatch (the xd design re-fills A per n-tile), so the
# cost model below charges it and the planner decides.
FUSED_C_TILES = 3             # gate, up, and the silu/mul destination


def l1_bytes_fused(m, n, kk, T=1, *, mixed=True, adepth=1, bdepth=1):
    """Per-core L1 of a FUSED gu image (see FUSED_C_TILES).

    The pair B object is 2 x b_obj_bytes(kk, n): the gate and up slices of the
    same n columns, contiguous per (n-tile, k-step) -- the delivered layout.
    """
    return (FUSED_C_TILES * T * c_leaf_bytes(m, n, mixed)
            + bdepth * 2 * b_obj_bytes(kk, n)
            + adepth * T * a_leaf_bytes(m, kk, mixed)
            + STACK_BYTES)


def divisors_of_n(N):
    """Every legal n for this output width: multiple of 16 and dividing N."""
    return [n for n in range(N_ALIGN, N + 1, N_ALIGN) if N % n == 0]


def pick_n(m, kk, T, N, *, mixed=True, adepth=1, bdepth=1):
    """Largest legal n for one GEMM, or 0 when even n=16 does not fit."""
    cap = min(n_cap(m, kk, T, mixed=mixed, adepth=adepth, bdepth=bdepth),
              n_cap_bd(m, mixed), N)
    best = 0
    for n in divisors_of_n(N):
        if n <= cap:
            best = n
    return best


def pick_n_fused(m, kk, T, N, *, mixed=True, adepth=1, bdepth=1, fill=1.0):
    """Largest legal n for the FUSED gu, or 0 when even n=16 does not fit.

    Same rules as pick_n (a multiple of 16, divides N, one BD per C leaf)
    against the fused L1 model.  fill optionally tightens the budget (1.0 = the
    design's own 64 KiB); the aiecc fill the placer actually accepts is
    recorded in FUSE-GU-REPORT.md and is what the synthesizer passes here.
    """
    budget = int(L1_BYTES * float(fill))
    best = 0
    for n in divisors_of_n(N):
        if n > n_cap_bd(m, mixed):
            continue
        if l1_bytes_fused(m, n, kk, T, mixed=mixed, adepth=adepth,
                          bdepth=bdepth) <= budget:
            best = n
    return best


# ===========================================================================
# tile maps (the same two the designs use)
# ===========================================================================
def tile_map(M, m, cols=COLS, cpc=CPC):
    """[cols][cpc] M-tile index; -1 = dummy.  ffn_design_x/ffn_xb's map.

    Requires nm = M // m <= cols * cpc: the design builds cols*cpc slots and
    asserts nxt == nm, so a bigger nm raises (and with cols cut it would
    silently DROP M-tiles -- the trap SWEEP-MN-REPORT.md hit).
    """
    nm = M // m
    base, rem = divmod(nm, cols)
    out, nxt = [], 0
    for c in range(cols):
        cnt = base + (1 if c < rem else 0)
        out.append([nxt + p if p < cnt else -1 for p in range(cpc)])
        nxt += cnt
    if nxt != nm:
        raise PlanError("tile_map: %d M-tiles do not fit %d slots"
                        % (nm, cols * cpc))
    return out


def tile_map_t(M, m, cols=COLS, cpc=CPC, rot=7):
    """[cols][cpc][n_t] M-tile indices; a core may own several.  ffn_t33's map.

    rot=7 (the last column carries the extra tile) is the VERIFIED form:
    rot=0 corrupted every column-0 block at odd n-tiles (SWEEP-T33-REPORT.md).
    """
    nm = M // m
    if nm < cols:
        raise PlanError("tile_map_t: %d M-tiles < %d columns (a column with "
                        "zero tiles would get a 0-byte ObjectFifo)" % (nm, cols))
    base, rem = divmod(nm, cols)
    per_col = [base + (1 if c < rem else 0) for c in range(cols)]
    if rot:
        per_col = per_col[-rot:] + per_col[:-rot]
    out, nxt = [], 0
    for c in range(cols):
        q, r = divmod(per_col[c], cpc)
        row = []
        for p in range(cpc):
            cnt = q + (1 if p < r else 0)
            row.append(list(range(nxt, nxt + cnt)))
            nxt += cnt
        out.append(row)
    if nxt != nm:
        raise PlanError("tile_map_t: %d of %d tiles placed" % (nxt, nm))
    return out


def _is_flat(tm):
    """True for ffn_xb's [cols][cpc] int map, False for ffn_t33's list map."""
    return isinstance(tm[0][0], int)


def per_col_counts(tm):
    """Tiles per column, for the plan record and the UI."""
    if _is_flat(tm):
        return [sum(1 for t in col if t >= 0) for col in tm]
    return [sum(len(r) for r in col) for col in tm]


def per_core_counts(tm):
    """Tiles per (column, core) slot, flattened to the design's [cols][cpc]."""
    if _is_flat(tm):
        return [[1 if t >= 0 else 0 for t in col] for col in tm]
    return [[len(r) for r in col] for col in tm]


# ===========================================================================
# cost model -- an estimate, anchored on measured rates, never a promise
# ===========================================================================
def dispatch_ms(m, kk, T, K, N, n, a_bytes, c_bytes):
    """Estimated wall time of ONE dispatch of this image (ms).

    Two terms, whichever binds:
      compute  the busiest core runs T M-tiles x ntiles n-tiles x (m, kk, n),
               i.e. T*m*K*N MACs at the measured per-core rate;
      DMA      A is re-streamed once per n-tile (the designs re-fill A per
               n-tile), B is read once per dispatch, C is written once.
    A partially filled array is charged the saturated per-core rate, i.e. the
    estimate is an upper bound, which is the honest direction.
    """
    r_core = RATE_TFLOPS[kk] / SLOTS * 1e12                  # FLOP/s per core
    t_comp = T * 2.0 * m * K * N / r_core * 1e3              # ms
    ntiles = N // n
    traffic = ntiles * a_bytes + (9 * K * N // 8) + c_bytes
    t_dma = traffic / (BW_GBPS * 1e9) * 1e3                  # ms
    return max(t_comp, t_dma) + DISP_MS


def dispatch_ms_fused(m, kk, T, K, N, n, a_bytes, c_bytes):
    """Estimated wall time of ONE FUSED gu dispatch (gate and up in one pass).

    Twice the FLOPs of a pure gu dispatch at the same n and twice the B bytes
    (the [gate|up] pair), with exactly twice as many n-tiles -- so the A
    traffic per dispatch is the same as the pure image's (ntiles = N/n grows as
    n shrinks, and it is ntiles x a_bytes that the sequence re-fills).
    """
    r_core = RATE_TFLOPS[kk] / SLOTS * 1e12                  # FLOP/s per core
    t_comp = T * 2.0 * 2.0 * m * K * N / r_core * 1e3        # ms (gate AND up)
    ntiles = N // n
    traffic = ntiles * a_bytes + 2 * (9 * K * N // 8) + c_bytes
    t_dma = traffic / (BW_GBPS * 1e9) * 1e3                  # ms
    return max(t_comp, t_dma) + DISP_MS


def layer_ms(plan, fused=False):
    """Estimated device ms for one DiT layer.

    pure: 2x gu + 1x dn + 4x proj.  fused: the ONE fused gu already IS gate and
    up, so it is 1x gu_fused + 1x dn + 4x proj.
    """
    per = plan["gemms"]
    n_gu = 1.0 if fused else 2.0
    return (n_gu * per["gu"]["dispatch_ms"] + per["dn"]["dispatch_ms"]
            + 4.0 * per["proj"]["dispatch_ms"]) * plan["ndisp"]


# ===========================================================================
# the planner
# ===========================================================================
def _candidates(M, m_choices, strategies):
    """Every (strategy, m, Mblk, ndisp, nm, T) shape that can cover M.

    xd   one tile per core (ffn_xb):  nm <= 32, ndisp = ceil(M / (32*m)),
         Mblk = m * ceil(M / (m*ndisp)) so the pad is the SMALLEST multiple of m
         that still covers M in ndisp dispatches.
    t33  one dispatch, T = ceil(nm/32) tiles per core (ffn_t33).  Only for
         nm >= cols, because a column with zero tiles would get a 0-byte
         ObjectFifo.
    """
    out = []
    for m in m_choices:
        if m % M_ALIGN:
            continue
        ndisp = -(-M // (SLOTS * m))
        mblk = m * (-(-M // (m * ndisp)))
        nm = mblk // m
        if 1 <= nm <= SLOTS:
            out.append(dict(strategy="xd", m=m, mblk=mblk, ndisp=ndisp,
                            nm=nm, T=1))
        if "t33" in strategies:
            mblk = m * (-(-M // m))
            nm = mblk // m
            if nm >= COLS:
                out.append(dict(strategy="t33", m=m, mblk=mblk, ndisp=1,
                                nm=nm, T=-(-nm // SLOTS)))
    return out


COST_TIE_MS = 0.5          # two plans within this are a tie -> _rank decides


def _rank(p):
    """Tie-break inside COST_TIE_MS: the simplest proven machinery first.

    1. ffn_xb (one tile per core) before ffn_t33 (multi-tile cores): one less
       mechanism, and it is the design the (m, n) sweep measured.
    2. fewer dispatches.
    3. fewer tiles per core (T) -- T=1 has no multi-tile leaf offsetting.
    4. larger m (fewer tiles for the same rows).
    """
    return (0 if p["strategy"] == "xd" else 1, p["ndisp"], p["T"], -p["m"])


def _better(new, cur, prefer=None):
    """Is NEW a better plan than CUR?  Cost first, then _rank, then prefer."""
    if prefer:
        nb, cb = new["plan_key"] in prefer, cur["plan_key"] in prefer
        if nb != cb and abs(new["cost"] - cur["cost"]) <= cur["cost"] * 0.15:
            return nb
    if new["cost"] < cur["cost"] - COST_TIE_MS:
        return True
    if new["cost"] > cur["cost"] + COST_TIE_MS:
        return False
    return _rank(new) < _rank(cur)


def plan(M, *, m_choices=(16, 32, 48, 64), kk_choices=KK_MEASURED,
         strategies=("xd", "t33"), mixed=True, adepth=1, bdepth=1,
         prefer=None, fuse_gu=False, fuse_fill=1.0):
    """Plan the 32-core placement for M rows.  PURE.

    Returns a dict:
        ok        bool
        plan      the chosen plan (None when ok is False)
        rejected  [{"strategy","m","kk","constraint","detail"}, ...] -- one row
                  per rejected family, deduped, so a UI can explain itself
        binding   the constraint that killed the last candidate family, or None
        reason    a one-sentence refusal in the 512^2 wording ("" when ok)

    prefer is an optional plan-key set (e.g. the plans already built); a
    candidate whose key is in it wins a tie inside 15 % of the best cost.  The
    function itself stays pure -- the caller supplies the set.

    fuse_gu (default False -> the delivered planner output, byte for byte)
    plans a gu image that fuses silu(gate)*up, so ONE dispatch replaces the
    gate and up dispatches and the host elementwise pass disappears.  Only the
    xd strategy has that image (one M-tile per core, the executor s2_ffn32
    supports), so the fusion narrows the candidate strategies to xd.  The fused
    image needs FUSED_C_TILES C tiles, which forces a SMALLER n than the pure
    image's and therefore twice as many n-tiles: fuse_fill is the fraction of
    the 64 KiB L1 budget the pick may use (see pick_n_fused).
    """
    M = int(M)
    if fuse_gu:
        strategies = tuple(s for s in strategies if s == "xd") or ("xd",)
    rejected = []
    if M < MIN_M or M > MAX_M:
        return {"ok": False, "plan": None, "rejected": rejected,
                "binding": "token envelope",
                "reason": ("M=%d is outside the product envelope %d..%d"
                           % (M, MIN_M, MAX_M))}
    if M % M_ALIGN:
        # not fatal: Mblk is a multiple of m >= 16, so the tail is padded
        rejected.append({"strategy": "-", "m": 0, "kk": 0,
                         "constraint": "row alignment",
                         "detail": ("M=%d is not a multiple of %d; the tail "
                                    "dispatch is zero-padded to Mblk and the "
                                    "pad rows are discarded" % (M, M_ALIGN))})

    best = None
    for cand in _candidates(M, m_choices, strategies):
        m, nm, T, mblk, ndisp = (cand["m"], cand["nm"], cand["T"],
                                 cand["mblk"], cand["ndisp"])
        for kk in kk_choices:
            if kk > KK_DMA_GATED:
                continue
            gemms, ok, why = {}, True, None
            for name, K, N in GEMMS:
                if K % kk:
                    ok, why = False, ("K=%d is not a multiple of kk=%d"
                                      % (K, kk))
                    break
                fused = bool(fuse_gu) and name == "gu"
                if fused:
                    n = pick_n_fused(m, kk, T, N, mixed=mixed, adepth=adepth,
                                     bdepth=bdepth, fill=fuse_fill)
                else:
                    n = pick_n(m, kk, T, N, mixed=mixed, adepth=adepth,
                               bdepth=bdepth)
                if not n:
                    ok = False
                    why = ("no legal %sn for N=%d: even n=16 needs %d B > %d B "
                           "L1 (T=%d)"
                           % ("fused " if fused else "", N,
                              (l1_bytes_fused if fused else l1_bytes)(
                                  m, 16, kk, T, mixed=mixed, adepth=adepth,
                                  bdepth=bdepth),
                              L1_BYTES, T))
                    break
                ntiles = N // n
                cl = c_leaf_bytes(m, n, mixed)
                # The BO sizes are the DESIGN's, not the useful-work size:
                #   ffn_xb  A_ty = (cols*nkk*cpc*a_leaf) -- all 32 slots, so a
                #           short image still fills 32 tiles (dummy cores read
                #           tile 0), and C_ty = (cols*ntiles*cpc*c_leaf).
                #   ffn_t33 A/C are the per-column elements, i.e. exactly the
                #           Mblk rows (no dummies).
                if cand["strategy"] == "xd":
                    a_bytes = SLOTS * m * K * 2
                    c_bytes = SLOTS * ntiles * cl
                else:
                    a_bytes = mblk * K * 2
                    c_bytes = nm * ntiles * cl
                gemms[name] = {
                    "K": K, "N": N, "n": n, "ntiles": ntiles,
                    "nkk": K // kk,
                    "l1": (l1_bytes_fused if fused else l1_bytes)(
                        m, n, kk, T, mixed=mixed, adepth=adepth,
                        bdepth=bdepth),
                    "a_bytes": a_bytes,
                    # a fused gu reads BOTH matrices: one [gate|up] pair object
                    # per (n-tile, k-step), 2 x 9*K*N/8 bytes
                    "b_bytes": (2 if fused else 1) * 9 * K * N // 8,
                    "c_bytes": c_bytes,
                    "a_useful": mblk * K * 2,
                    "dispatch_ms": round(
                        (dispatch_ms_fused if fused else dispatch_ms)(
                            m, kk, T, K, N, n, a_bytes, c_bytes), 3),
                }
                if fused:
                    gemms[name]["fused"] = 1
            if not ok:
                rejected.append({"strategy": cand["strategy"], "m": m,
                                 "kk": kk, "constraint": "L1 / N divisibility",
                                 "detail": why})
                continue
            tm = (tile_map(mblk, m) if cand["strategy"] == "xd"
                  else tile_map_t(mblk, m, rot=7))
            p = {
                "M": M, "m": m, "kk": kk, "Mblk": mblk, "nm": nm, "T": T,
                "ndisp": ndisp, "strategy": cand["strategy"],
                "cols": COLS, "cpc": CPC, "mixed": bool(mixed),
                "adepth": adepth, "bdepth": bdepth,
                # ffn_t33's drain await.  The delivered t33 form awaits only
                # the LAST column's C drain per n-tile (wait=(c == cols-1));
                # measured 2026-11-03 at M=4128: the final n-tile came back
                # 18 % wrong (maxabs 20.9 of 29.5) while every other n-tile was
                # exact to 1 ulp.  cwait=True awaits EVERY column's drain,
                # which is the design's own knob for this bug class
                # (SWEEP-T33-REPORT 4.1).  xd (ffn_xb) has no such knob and is
                # unaffected in every measurement so far.
                "cwait": bool(cand["strategy"] == "t33"),
                "pad_rows": mblk * ndisp - M,
                "pad_frac": round((mblk * ndisp - M) / float(mblk * ndisp), 6),
                "per_col": per_col_counts(tm), "per_core": per_core_counts(tm),
                "tile_map": tm,
                "dispatches": [[i * mblk, min(mblk, M - i * mblk)]
                               for i in range(ndisp)],
                "gemms": gemms,
                "l1_max": max(g["l1"] for g in gemms.values()),
            }
            if fuse_gu:
                # only when it is ON, so every pre-existing plan_key stays
                # byte-identical (the 24 built manifests must stay cache hits)
                p["fuse_gu"] = 1
                p["fuse_fill"] = float(fuse_fill)
            p["layer_ms"] = round(layer_ms(p, fused=bool(fuse_gu)), 2)
            p["step_s"] = round(layer_ms(p) * LAYERS / 1e3, 2)
            p["s_8step"] = round(layer_ms(p) * LAYERS * 8 / 1e3, 1)
            p["h_8step"] = round(layer_ms(p) * LAYERS * 8 / 3.6e6, 2)
            p["plan_key"] = plan_key(p)
            p["cost"] = p["layer_ms"]
            if best is None or _better(p, best, prefer):
                best = p
    if best is None:
        return {"ok": False, "plan": None, "rejected": rejected,
                "binding": "L1 / slot count",
                "reason": ("M=%d (%.0f image tokens) has no legal 32-core "
                           "placement: every (m, kk, T) family either fails "
                           "the 65,536 B per-core L1 budget or cannot divide "
                           "N=10240/3840 by a legal n.  L1 model: "
                           "T*m*n*2 + 9*kk*n/8 + m*kk*2 + %d B stack."
                           % (M, M - 32, STACK_BYTES))}
    return {"ok": True, "plan": best, "rejected": rejected, "binding": None,
            "reason": ""}


def plan_key(p):
    """A stable key over EVERY plan parameter (not over H, W or M).

    Two shapes that produce the same plan share this key and therefore share
    one build, one manifest entry and one cache copy.
    """
    core = {
        "m": p["m"], "kk": p["kk"], "Mblk": p["Mblk"], "T": p["T"],
        "strategy": p["strategy"], "cols": p["cols"], "cpc": p["cpc"],
        "mixed": p["mixed"], "adepth": p["adepth"], "bdepth": p["bdepth"],
        "cwait": bool(p.get("cwait")),
        "n": {k: p["gemms"][k]["n"] for k in sorted(p["gemms"])},
    }
    if p.get("fuse_gu"):
        # the fusion is a COMPILE-TIME knob of the gu image (its B object is a
        # pair, its epilogue is a silu+mul, its C leaf is half as wide), so it
        # must be in the key.  It is added only when on: the off-plan key is
        # the one every built manifest already carries.
        core["fuse_gu"] = 1
    blob = json.dumps(core, sort_keys=True, separators=(",", ":"))
    return "%s_m%d_kk%d_Mblk%d_n%s" % (
        p["strategy"], p["m"], p["kk"], p["Mblk"],
        "-".join(str(core["n"][k]) for k in sorted(core["n"]))) + \
        "_" + hashlib.sha256(blob.encode()).hexdigest()[:8]


# ===========================================================================
# host BO sizes -- the replacement for the 11-tile / 17-tile literals
# ===========================================================================
def bo_sizes(M, K=3840, FF=10240, block=96):
    """Host BO bytes the product path needs for M rows.  Derived, not tiled.

    imaggers/image/t18_run.py used to carry two literals measured at 512^2:
        x   BO 8,110,080 B  = 1056 rows x 3840 x 2  ("11 tiles")
        ff  BO 21,626,880 B = 1056 rows x 10240 x 2
    For M=1056 those are exactly M*K*2 and M*FF*2, so deriving them changes
    nothing at the delivered size and is correct at every other one.
    """
    M = int(M)
    return {
        "M": M,
        "x_bytes": M * K * 2,
        "ff_bytes": M * FF * 2,
        "out_bytes": M * K * 2,
        "tiles": -(-M // block),          # 96-row tiles, for the ledger
        "block": block,
        "rows_padded": -(-M // block) * block,
    }


# ===========================================================================
# refusal wording -- the same shape as webui/shapes._no_build
# ===========================================================================
def refusal_text(h, w, res):
    """One sentence: what is missing + the exact arithmetic + 不可用."""
    head = "%d²" % h if h == w else "%d×%d" % (h, w)
    b = res.get("binding") or "L1 / slot count"
    return ("%s 需新的 32 核放置，暂不可用：%s（绑定约束：%s）；本树没有该 "
            "(m, n, kk) 组合的编译配方，四门禁未跑，本树未构建，不可用。"
            % (head, res["reason"], b))


# ===========================================================================
# selftest -- the model must reproduce the four measured L1 points
# ===========================================================================
MEASURED_L1 = (
    # (m, n, kk, T, bytes, accepted?)
    (32, 512, 32, 1, 55296, True),
    (32, 384, 64, 1, 58368, True),
    (32, 640, 32, 1, 68096, False),
    (32, 512, 64, 1, 75776, False),
    (32, 320, 32, 2, 58624, True),      # the delivered t33 form (T=2)
    (16, 240, 32, 7, 71616, False),     # aiecc refused it at M=3168
)


def _selftest():
    bad = []
    for m, n, kk, T, want, ok in MEASURED_L1:
        got = l1_bytes(m, n, kk, T)
        if got != want:
            bad.append("l1_bytes(%d,%d,%d) = %d, measured %d"
                       % (m, n, kk, got, want))
        if (got <= L1_BYTES) != ok:
            bad.append("l1_bytes(%d,%d,%d) legality disagrees with the build"
                       % (m, n, kk))
    r = plan(1056)
    if not r["ok"]:
        bad.append("M=1056 has no plan: %s" % r["reason"])
    r96 = plan(96)
    if not r96["ok"]:
        bad.append("M=96 has no plan: %s" % r96["reason"])
    for M in (608, 2336, 4128, 16416):
        r = plan(M)
        if not r["ok"]:
            bad.append("M=%d has no plan: %s" % (M, r["reason"]))
            continue
        p = r["plan"]
        for name, K, N in GEMMS:
            g = p["gemms"][name]
            if N % g["n"] or g["n"] % N_ALIGN:
                bad.append("M=%d %s: illegal n=%d" % (M, name, g["n"]))
            if g["l1"] > L1_BYTES:
                bad.append("M=%d %s: L1 %d > %d"
                           % (M, name, g["l1"], L1_BYTES))
        if p["Mblk"] % p["m"]:
            bad.append("M=%d: Mblk not a multiple of m" % M)
    b = bo_sizes(1056)
    if b["x_bytes"] != 8_110_080 or b["ff_bytes"] != 21_626_880:
        bad.append("bo_sizes(1056) = %d/%d, delivered 8,110,080/21,626,880"
                   % (b["x_bytes"], b["ff_bytes"]))
    # --- the fused gu: legal at kk=32, and it must NOT collide with the pure
    # plan's key (the fusion is a compile-time knob of the image) -------------
    for M in (96, 608, 1056, 4128, 8192, 16416):
        rf = plan(M, fuse_gu=True)
        if not rf["ok"]:
            bad.append("M=%d has no fused plan: %s" % (M, rf["reason"]))
            continue
        pf = rf["plan"]
        rp = plan(M)
        if pf["strategy"] != "xd":
            bad.append("M=%d: fused plan is %s, only xd has the fusion"
                       % (M, pf["strategy"]))
        if pf["plan_key"] == (rp["plan"]["plan_key"] if rp["ok"] else None):
            bad.append("M=%d: the fused and pure plans share a plan_key" % M)
        g = pf["gemms"]["gu"]
        if not g.get("fused") or g["b_bytes"] != 2 * (9 * g["K"] * g["N"] // 8):
            bad.append("M=%d: fused gu does not carry the pair object" % M)
        if l1_bytes_fused(pf["m"], g["n"], pf["kk"]) > L1_BYTES:
            bad.append("M=%d: fused gu L1 %d > %d"
                       % (M, l1_bytes_fused(pf["m"], g["n"], pf["kk"]),
                          L1_BYTES))
        if N_ALIGN and (g["N"] % g["n"] or g["n"] % N_ALIGN):
            bad.append("M=%d: fused gu n=%d does not divide N=%d"
                       % (M, g["n"], g["N"]))
        # the fused gu must be charged for BOTH matrices in one dispatch: it is
        # never cheaper than the PURE formula at the same geometry (that would
        # mean the model priced one GEMM where the image does two)
        pure_same = dispatch_ms(pf["m"], pf["kk"], pf["T"], g["K"], g["N"],
                                g["n"], g["a_bytes"], g["c_bytes"])
        if g["dispatch_ms"] < pure_same:
            bad.append("M=%d: the fused gu is priced as one GEMM (%.3f < %.3f)"
                       % (M, g["dispatch_ms"], pure_same))
    return bad


def main() -> int:
    import argparse
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=None)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fuse", action="store_true",
                    help="plan the FUSED gu (silu(gate)*up inside the image)")
    ap.add_argument("--fuse-fill", type=float, default=1.0,
                    help="fraction of the 64 KiB L1 budget the fused n may use")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if a.selftest:
        bad = _selftest()
        for b in bad:
            print("FAIL", b)
        print("selftest:", "OK" if not bad else "%d failures" % len(bad))
        return 0 if not bad else 2
    if a.grid:
        for M in (96, 160, 192, 256, 288, 384, 448, 608, 832, 1056, 1632,
                  2336, 3168, 4128, 8192):
            r = plan(M, fuse_gu=a.fuse, fuse_fill=a.fuse_fill)
            if not r["ok"]:
                print("%5d REFUSED %s" % (M, r["reason"]))
                continue
            p = r["plan"]
            print("%5d %-4s m=%2d kk=%2d Mblk=%4d nm=%3d T=%d nd=%d "
                  "n=%s L1=%6d pad=%4d layer=%7.1f ms%s"
                  % (M, p["strategy"], p["m"], p["kk"], p["Mblk"], p["nm"],
                     p["T"], p["ndisp"],
                     "/".join(str(p["gemms"][k]["n"]) for k in ("gu", "dn",
                                                                "proj")),
                     p["l1_max"], p["pad_rows"], p["layer_ms"],
                     "  FUSED" if p.get("fuse_gu") else ""))
        return 0
    if a.m is not None:
        print(json.dumps(plan(a.m, fuse_gu=a.fuse, fuse_fill=a.fuse_fill),
                         indent=1))
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
