# vaeg_design.py -*- Python -*-
#
# IRON designs for the two missing VAE decoder ops:
#
#   1. GroupNorm(32, eps=1e-6) -- `group_norm.cc` driven one group per call
#      (`make_gn_single`) or as streamed chunks in two passes over the data
#      (`make_gn_stream`, for the groups too big to hold in + out at once).
#   2. Nearest-2x upsample -- PURE DMA, no kernel and no worker: the
#      replicate is a stride pattern on the producer side (a host gather),
#      proven/refuted per axis in vaeg_up.py.
#
# Dataflow follows the proven `imaggers/kernels/designs.py` rmsw design
# (one Worker, depth-1 ObjectFifos, gamma held for the whole run) and
# `imaggers/vae/vae_npu.py` (memref.view over 1-D i8 arenas, dynamic
# offsets, ObjectFifoLink for pure-DMA hops).
#
# GB arena (1-D i8 ObjectFifo object, held by the Worker for the run):
#     [0, 2C)          gamma, C bf16
#     [2C, 4C)         beta,  C bf16
#     [stats_off, ..)  float[4] stats  (stream design only; 64-B aligned)
# gamma/beta/stats are read/written SCALAR-only in the kernel, so only x/y
# carry 64-B alignment contracts.
from __future__ import annotations

import os
import sys
from pathlib import Path

# mlir_aie setup: package roots on sys.path, PEANO + cache BEFORE `import aie`
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent.parent                       # ...\public_work
_PYLIBS = _ROOT / "_pylibs"
sys.path.append(str(_HERE))
sys.path.append(str(_PYLIBS))
sys.path.append(str(_PYLIBS / "mlir_aie" / "python"))
os.environ.setdefault("PEANO_INSTALL_DIR", str(_PYLIBS / "llvm-aie" / "llvm-aie"))
os.environ.setdefault("NPU_CACHE_HOME", str(_HERE / "_npucache_vaeg_default"))

import numpy as np
import aie.iron as iron
from aie.dialects import memref
from aie.extras.dialects.arith import addi, constant, muli
from aie.helpers.taplib import TensorAccessPattern
from aie.helpers.util import np_ndarray_type_to_memref_type
from aie.iron import (
    CompileTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    TaskGroup,
    Worker,
)
from aie.iron.controlflow import range_
from aie.iron.dataflow import ObjectFifoLink
from aie.iron.device import Tile
from ml_dtypes import bfloat16

import vaeg_kernels as K
import t21_3b2vf_geom as VG     # t21_3b2_vf: the design-variant geometry

GROUPS = 32


def gb_layout(C: int, with_stats: bool = True):
    """(gb_bytes, stats_off): the GB arena layout (see module docstring)."""
    stats_off = (4 * C + 63) // 64 * 64
    gb_bytes = stats_off + 16 if with_stats else 4 * C
    return gb_bytes, stats_off


def pick_chunk(Cs_max: int, group_len: int, spatial: int) -> int:
    """Largest Cs <= Cs_max with Cs a whole number of channels and Cs | group_len."""
    cs = min(Cs_max, group_len)
    while cs >= spatial and (group_len % cs or cs % spatial):
        cs -= spatial
    if cs < spatial:
        raise ValueError(f"no chunk size for L={group_len} S={spatial}")
    return cs


# ---------------------------------------------------------------------------
# GroupNorm: ONE CALL PER GROUP (whole group in + out; group_len <= 8192)
# ---------------------------------------------------------------------------

def make_gn_single(C: int, H: int, W: int, groups: int = GROUPS):
    """Design factory: group_norm_group per group, whole-group objects."""
    assert C % groups == 0
    cg = C // groups
    S = H * W
    L = cg * S
    assert 2 * 2 * L + 4096 <= 64 * 1024, (
        f"single-call design needs 2*L bytes in + out in 64 KB L1, got L={L}"
    )
    gb_bytes, _ = gb_layout(C, with_stats=False)

    @iron.jit(aiecc_flags=["--dynamic-objFifos"])
    def gn_single(
        X: In,
        GB: In,
        Y: Out,
        *,
        eps: CompileTime[float] = 1e-6,
    ):
        x_ty = np.ndarray[(L,), np.dtype[bfloat16]]
        y_ty = np.ndarray[(L,), np.dtype[bfloat16]]
        gb_ty = np.ndarray[(gb_bytes,), np.dtype[np.int8]]
        gam_ty = np.ndarray[(cg,), np.dtype[bfloat16]]
        bet_ty = np.ndarray[(cg,), np.dtype[bfloat16]]

        kern = K.group_norm_group(L, cg)

        inX = ObjectFifo(x_ty, name="inX", depth=1)
        inGB = ObjectFifo(gb_ty, name="inGB", depth=1)
        outY = ObjectFifo(y_ty, name="outY", depth=1)

        def core_fn(of_x, of_gb, of_y, k):
            gam_mt = np_ndarray_type_to_memref_type(gam_ty)
            bet_mt = np_ndarray_type_to_memref_type(bet_ty)
            gb = of_gb.acquire(1)          # held for the whole run (rmsw idiom)
            for gi in range_(groups):
                gbase = muli(gi, constant(cg * 2, index=True))
                gam = memref.view(gam_mt, gb, gbase, [])
                bet = memref.view(
                    bet_mt, gb, addi(gbase, constant(C * 2, index=True)), []
                )
                x = of_x.acquire(1)
                y = of_y.acquire(1)
                k(x, y, gam, bet, L, S, eps)
                of_x.release(1)
                of_y.release(1)
            of_gb.release(1)

        worker = Worker(
            core_fn,
            [inX.cons(), inGB.cons(), outY.prod(), kern],
            stack_size=0x1000,
        )

        X_ty = np.ndarray[(groups * L,), np.dtype[bfloat16]]
        GB_ty = np.ndarray[(gb_bytes,), np.dtype[np.int8]]
        Y_ty = np.ndarray[(groups * L,), np.dtype[bfloat16]]
        gb_tap = TensorAccessPattern.full((gb_bytes,)).tile((gb_bytes,))
        x_tap = TensorAccessPattern.full((groups * L,)).tile((L,))
        y_tap = TensorAccessPattern.full((groups * L,)).tile((L,))

        def sequence(x, gb, y, inX_h, inGB_h, outY_h):
            tgs = [TaskGroup()]
            inGB_h.fill(gb, tap=gb_tap, group=tgs[0])
            inX_h.fill(x, tap=x_tap, group=tgs[0])
            outY_h.drain(y, tap=y_tap, group=tgs[0], wait=True)
            tgs[0].finish()
            del tgs[0]

        rt = Runtime(
            sequence,
            [X_ty, GB_ty, Y_ty, inX.prod(), inGB.prod(), outY.cons()],
        )
        return Program(
            iron.get_current_device(), rt, workers=[worker]
        ).resolve_program()

    return gn_single


# ---------------------------------------------------------------------------
# GroupNorm: STREAMED CHUNKS, TWO PASSES (any group size)
# ---------------------------------------------------------------------------

def make_gn_stream(C: int, H: int, W: int, groups: int = GROUPS,
                   cs_max: int = 4096, repeat_tap: bool = False):
    """Design factory: stats pass + finalize + apply pass over Cs chunks.

    The host streams each group's chunks TWICE (two passes per group).  With
    ``repeat_tap`` the interleaved order [g0 p1][g0 p2][g1 p1][g1 p2]... is
    ONE fill whose walk repeats every group twice (a zero-stride dim) --
    MEASURED REFUSAL: the dynamic lowering expands that walk into one BD per
    (group, repetition) -- 64 BDs at (512,8,8) -- and aiecc refuses with
    "Too many simultaneously active buffer descriptors on tile (0,0), which
    supports up to 16".  The default therefore emits two fills per group in
    batched TaskGroups (<= 8 pushes each; each fill is one 2-dim BD).
    The Worker parks the running statistics in the GB arena's stats slot.
    """
    assert C % groups == 0
    cg = C // groups
    S = H * W
    L = cg * S
    Cs = pick_chunk(cs_max, L, S)
    n_chunks = L // Cs
    chpc = Cs // S                     # channels per chunk (whole runs)
    gb_bytes, stats_off = gb_layout(C, with_stats=True)

    @iron.jit(aiecc_flags=["--dynamic-objFifos"])
    def gn_stream(
        X: In,
        GB: In,
        Y: Out,
        *,
        eps: CompileTime[float] = 1e-6,
    ):
        x_ty = np.ndarray[(Cs,), np.dtype[bfloat16]]
        y_ty = np.ndarray[(Cs,), np.dtype[bfloat16]]
        gb_ty = np.ndarray[(gb_bytes,), np.dtype[np.int8]]
        gam_ty = np.ndarray[(chpc,), np.dtype[bfloat16]]
        bet_ty = np.ndarray[(chpc,), np.dtype[bfloat16]]
        st_ty = np.ndarray[(4,), np.dtype[np.float32]]

        k_obj = K._obj_name("stream", Cs, chpc)
        k_stats = K.group_norm_stats(Cs, k_obj)
        k_fin = K.group_norm_finalize(k_obj)
        k_app = K.group_norm_apply(Cs, chpc, k_obj)

        inX = ObjectFifo(x_ty, name="inX", depth=1)
        inGB = ObjectFifo(gb_ty, name="inGB", depth=1)
        outY = ObjectFifo(y_ty, name="outY", depth=1)

        def core_fn(of_x, of_gb, of_y, ks, kf, ka):
            gam_mt = np_ndarray_type_to_memref_type(gam_ty)
            bet_mt = np_ndarray_type_to_memref_type(bet_ty)
            st_mt = np_ndarray_type_to_memref_type(st_ty)
            gb = of_gb.acquire(1)          # held for the whole run
            st = memref.view(st_mt, gb, constant(stats_off, index=True), [])
            for gi in range_(groups):
                gbase = muli(gi, constant(cg * 2, index=True))
                # pass 1: accumulate sum / sum-of-squares
                for _c in range(n_chunks):     # unrolled: constant call sites
                    x = of_x.acquire(1)
                    ks(x, st, Cs)
                    of_x.release(1)
                kf(st, L, eps)
                # pass 2: normalize
                for c in range(n_chunks):
                    x = of_x.acquire(1)
                    y = of_y.acquire(1)
                    coff = addi(gbase, constant(c * chpc * 2, index=True))
                    gam = memref.view(gam_mt, gb, coff, [])
                    bet = memref.view(
                        bet_mt,
                        gb,
                        addi(coff, constant(C * 2, index=True)),
                        [],
                    )
                    ka(x, y, gam, bet, st, Cs, S)
                    of_x.release(1)
                    of_y.release(1)
            of_gb.release(1)

        worker = Worker(
            core_fn,
            [inX.cons(), inGB.cons(), outY.prod(), k_stats, k_fin, k_app],
            stack_size=0x1000,
        )

        X_ty = np.ndarray[(groups * L,), np.dtype[bfloat16]]
        GB_ty = np.ndarray[(gb_bytes,), np.dtype[np.int8]]
        Y_ty = np.ndarray[(groups * L,), np.dtype[bfloat16]]
        gb_tap = TensorAccessPattern.full((gb_bytes,)).tile((gb_bytes,))
        y_tap = TensorAccessPattern.full((groups * L,)).tile((Cs,))
        if repeat_tap:
            # one fill: [group, repeat, chunk, elem] with a zero-stride repeat
            x_tap = TensorAccessPattern(
                (groups * L,), 0,
                [groups, 2, n_chunks, Cs], [L, 0, Cs, 1],
            )

        def sequence(x, gb, y, inX_h, inGB_h, outY_h):
            if repeat_tap:
                # Kept for reproduction of the MEASURED REFUSAL (see the
                # factory docstring): this one fill's walk is expanded by the
                # dynamic lowering into one BD per (group, repetition).
                tgs = [TaskGroup()]
                outY_h.drain(y, tap=y_tap, group=tgs[0], wait=True)
                inGB_h.fill(gb, tap=gb_tap, group=tgs[0], wait=True)
                inX_h.fill(x, tap=x_tap, group=tgs[0], wait=True)
                tgs[0].finish()
                del tgs[0]
                return
            # The DRAIN is pushed FIRST into its own group, awaited LAST: the
            # shim channels are independent and the objfifo locks order the
            # data, so the drain consumes y while the fills are still being
            # pushed.  This matters twice over (both measured):
            #   * TaskGroup.finish() FREES non-waited transfers, and
            #     dma_free_task does not wait for completion -- a batch of
            #     unwaited fills freed at close is a race (hangs, ERT
            #     ERT_CMD_STATE_TIMEOUT);
            #   * awaiting fills before the drain is pushed deadlocks on
            #     outY backpressure (the worker blocks once the y pool is
            #     full and no fill can complete any more).
            # Every fill is therefore waited (each batch closes with awaits),
            # and batches keep <= 8 simultaneously active shim BDs (the shim
            # supports 16).
            drain_tg = TaskGroup()
            outY_h.drain(y, tap=y_tap, group=drain_tg, wait=True)

            tgs = [TaskGroup()]
            inGB_h.fill(gb, tap=gb_tap, group=tgs[-1], wait=True)
            pushes = 1
            for g in range(groups):
                g_tap = (
                    TensorAccessPattern.full((groups * L,))
                    [g * L:(g + 1) * L].tile((Cs,))
                )
                for _p in range(2):        # pass 1, then pass 2
                    inX_h.fill(x, tap=g_tap, group=tgs[-1], wait=True)
                    pushes += 1
                    if pushes >= 8:
                        tgs[-1].finish()
                        del tgs[-1]
                        tgs.append(TaskGroup())
                        pushes = 0
            tgs[-1].finish()          # always close the trailing group
            del tgs[-1]
            drain_tg.finish()
            del drain_tg

        rt = Runtime(
            sequence,
            [X_ty, GB_ty, Y_ty, inX.prod(), inGB.prod(), outY.cons()],
        )
        return Program(
            iron.get_current_device(), rt, workers=[worker]
        ).resolve_program()

    return gn_stream


# ---------------------------------------------------------------------------
# Nearest-2x upsample: PURE DMA designs (no kernel, no Worker)
# ---------------------------------------------------------------------------

def up_tap(C: int, H: int, W: int, mode: str, channel: int):
    """Producer-side (host gather) tap of ONE channel's replicate walk.

    mode "full": every element twice and every row twice -- the exact
        nearest-2x stream: sizes [H, 2, W, 2], strides [W, 0, 1, 0].
    mode "rows": every row twice only: sizes [H, 2, W], strides [W, 0, 1].
    mode "elems": every element twice only (flat): sizes [H*W, 2], strides
        [1, 0].
    """
    off = channel * H * W
    if mode == "full":
        return TensorAccessPattern(
            (C * H * W,), off, [H, 2, W, 2], [W, 0, 1, 0]
        )
    if mode == "rows":
        return TensorAccessPattern((C * H * W,), off, [H, 2, W], [W, 0, 1])
    if mode == "elems":
        return TensorAccessPattern((C * H * W,), off, [H * W, 2], [1, 0])
    raise ValueError(f"unknown upsample tap mode {mode!r}")


def make_up_tap(C: int, H: int, W: int, mode: str = "full", batch: int = 8):
    """Pure-DMA upsample attempt: host --gather tap--> memtile hop --> host.

    No Worker and no kernel: inX's producer-side fill walks the host input
    with the replicate tap; an ObjectFifoLink at the memtile forwards the
    stream; outX drains it linearly to the host output.  ``batch`` is the
    number of fills per TaskGroup close (a zero-stride tap dim is expanded
    into one BD per repetition by the lowering, so a wide tap needs
    batch=1); the shim holds 16 active BDs.
    """
    if mode == "full":
        plane_out = 4 * H * W
    elif mode == "rows":
        plane_out = 2 * H * W
    elif mode == "elems":
        plane_out = 2 * H * W
    else:
        raise ValueError(mode)

    @iron.jit(aiecc_flags=["--dynamic-objFifos"])
    def up_design(X: In, Y: Out):
        plane_ty = np.ndarray[(plane_out,), np.dtype[bfloat16]]
        inX = ObjectFifo(plane_ty, name="inX", depth=1)
        outX = ObjectFifo(plane_ty, name="outX", depth=1)
        # pure-DMA hop at the memtile (vae_npu's link idiom, identity walk)
        _ = ObjectFifoLink(inX.cons(), outX.prod(), Tile(0, 1), [], [])

        X_ty = np.ndarray[(C * H * W,), np.dtype[bfloat16]]
        Y_ty = np.ndarray[(C * plane_out,), np.dtype[bfloat16]]
        y_tap = TensorAccessPattern.full((C * plane_out,)).tile((plane_out,))

        def sequence(x, y, inX_h, outX_h):
            # drain first, awaited last (see make_gn_stream's comment: the
            # shim channels are independent; unwaited fills freed at a
            # TaskGroup close are a dma_free_task race, and awaiting fills
            # before the drain is pushed deadlocks on backpressure).
            drain_tg = TaskGroup()
            outX_h.drain(y, tap=y_tap, group=drain_tg, wait=True)
            tgs = [TaskGroup()]
            pushes = 0
            for c in range(C):
                inX_h.fill(x, tap=up_tap(C, H, W, mode, c), group=tgs[-1],
                           wait=True)
                pushes += 1
                if pushes >= batch:
                    tgs[-1].finish()
                    del tgs[-1]
                    tgs.append(TaskGroup())
                    pushes = 0
            tgs[-1].finish()          # always close the trailing group
            del tgs[-1]
            drain_tg.finish()
            del drain_tg

        rt = Runtime(sequence, [X_ty, Y_ty, inX.prod(), outX.cons()])
        return Program(
            iron.get_current_device(), rt, workers=[]
        ).resolve_program()

    return up_design


def make_up_kernel(C: int, H: int, W: int):
    """Complete nearest-2x upsample on device: `upsample_nearest2` per plane.

    Pure data movement (a half-word shuffle; zero arithmetic): one Worker,
    one call per channel plane, n = H*W in, 4*H*W out.
    """
    plane_in = H * W
    plane_out = 4 * H * W

    @iron.jit(aiecc_flags=["--dynamic-objFifos"])
    def up_design(X: In, Y: Out):
        x_ty = np.ndarray[(plane_in,), np.dtype[bfloat16]]
        y_ty = np.ndarray[(plane_out,), np.dtype[bfloat16]]
        kern = K.upsample_nearest2(plane_in, W)

        inX = ObjectFifo(x_ty, name="inX", depth=1)
        outY = ObjectFifo(y_ty, name="outY", depth=1)

        def core_fn(of_x, of_y, k):
            for _ in range_(C):
                x = of_x.acquire(1)
                y = of_y.acquire(1)
                k(x, y, plane_in, W)
                of_x.release(1)
                of_y.release(1)

        worker = Worker(
            core_fn, [inX.cons(), outY.prod(), kern], stack_size=0x1000
        )

        X_ty = np.ndarray[(C * plane_in,), np.dtype[bfloat16]]
        Y_ty = np.ndarray[(C * plane_out,), np.dtype[bfloat16]]
        x_tap = TensorAccessPattern.full((C * plane_in,)).tile((plane_in,))
        y_tap = TensorAccessPattern.full((C * plane_out,)).tile((plane_out,))

        def sequence(x, y, inX_h, outY_h):
            # designs.py's proven single-group shape (fills unwaited, drain
            # waited): finish() awaits the drain first, then frees the fills
            # -- by then they have long completed.
            tgs = [TaskGroup()]
            inX_h.fill(x, tap=x_tap, group=tgs[0])
            outY_h.drain(y, tap=y_tap, group=tgs[0], wait=True)
            tgs[0].finish()
            del tgs[0]

        rt = Runtime(sequence, [X_ty, Y_ty, inX.prod(), outY.cons()])
        return Program(
            iron.get_current_device(), rt, workers=[worker]
        ).resolve_program()

    return up_design


# ---------------------------------------------------------------------------
# t21_3b2_vf VARIANTS (T21-3B2VF-PLAN.md)
#
# Both factories below are ADDITIVE: the 3b2 factories above are byte-for-byte
# unchanged, so every artifact already built stays compatible.
# ---------------------------------------------------------------------------

def make_gn_stream_sub(C: int, H: int, W: int, groups: int = GROUPS,
                       cs_max: int = 4096):
    """Design factory: stats + finalize + apply over SUB-PLANE chunks.

    The 3b2 make_gn_stream chunks by WHOLE channel planes (Cs a multiple of
    S), so a geometry whose plane exceeds the chunk cap has no chunk at all
    (S > cs_max -> pick_chunk raises).  Here a chunk may be a run INSIDE one
    channel: Cs < S, Cs | S.  The group segment is contiguous either way (the
    chunk is always [c*Cs, (c+1)*Cs) of the group), so the host tap, the
    kernel entry points and the per-chunk element count are unchanged.  The
    two differences are:

      * gamma/beta: a whole-plane chunk views chpc = Cs/S consecutive channel
        values, a sub-plane chunk views ONE value (memref.view at the chunk's
        first channel).  The channel base is an OFFSET, not a parameter --
        group_norm_apply already spends all 7 of the platform's data
        parameters (x, y, gamma, beta, stats, n, spatial).
      * spatial: S for whole planes, Cs for a sub-plane chunk (the kernel
        derives nch = n/spatial, so nch == 1 there).

    When S <= cs_max and a whole-plane chunk exists this reduces to
    make_gn_stream's chunking exactly (checked by the plan printer).
    """
    assert C % groups == 0
    cg = C // groups
    S = H * W
    L = cg * S
    Cs, nch, spatial = VG.gn_chunk_sub(C, S, cs_max)
    n_chunks = L // Cs
    gb_bytes, stats_off = gb_layout(C, with_stats=True)
    assert 4 * Cs + 2 * 0x1000 + 4096 <= 64 * 1024, (
        f"in+out chunk objects are {4 * Cs} B, over the 64 KB L1")

    @iron.jit(aiecc_flags=["--dynamic-objFifos"])
    def gn_stream_sub(
        X: In,
        GB: In,
        Y: Out,
        *,
        eps: CompileTime[float] = 1e-6,
    ):
        x_ty = np.ndarray[(Cs,), np.dtype[bfloat16]]
        y_ty = np.ndarray[(Cs,), np.dtype[bfloat16]]
        gb_ty = np.ndarray[(gb_bytes,), np.dtype[np.int8]]
        gam_ty = np.ndarray[(nch,), np.dtype[bfloat16]]
        bet_ty = np.ndarray[(nch,), np.dtype[bfloat16]]
        st_ty = np.ndarray[(4,), np.dtype[np.float32]]

        k_obj = K._obj_name("sub", Cs, nch)
        k_stats = K.group_norm_stats(Cs, k_obj)
        k_fin = K.group_norm_finalize(k_obj)
        k_app = K.group_norm_apply(Cs, nch, k_obj)

        inX = ObjectFifo(x_ty, name="inX", depth=1)
        inGB = ObjectFifo(gb_ty, name="inGB", depth=1)
        outY = ObjectFifo(y_ty, name="outY", depth=1)

        chan = [VG.gn_chan_offset(C, S, Cs, c) for c in range(n_chunks)]

        def core_fn(of_x, of_gb, of_y, ks, kf, ka):
            gam_mt = np_ndarray_type_to_memref_type(gam_ty)
            bet_mt = np_ndarray_type_to_memref_type(bet_ty)
            st_mt = np_ndarray_type_to_memref_type(st_ty)
            gb = of_gb.acquire(1)          # held for the whole run
            st = memref.view(st_mt, gb, constant(stats_off, index=True), [])
            for gi in range_(groups):
                gbase = muli(gi, constant(cg * 2, index=True))
                # pass 1: accumulate sum / sum-of-squares over every chunk
                for _c in range(n_chunks):     # unrolled: constant call sites
                    x = of_x.acquire(1)
                    ks(x, st, Cs)
                    of_x.release(1)
                kf(st, L, eps)
                # pass 2: normalize
                for c in range(n_chunks):
                    x = of_x.acquire(1)
                    y = of_y.acquire(1)
                    coff = addi(gbase, constant(chan[c] * 2, index=True))
                    gam = memref.view(gam_mt, gb, coff, [])
                    bet = memref.view(
                        bet_mt,
                        gb,
                        addi(coff, constant(C * 2, index=True)),
                        [],
                    )
                    ka(x, y, gam, bet, st, Cs, spatial)
                    of_x.release(1)
                    of_y.release(1)
            of_gb.release(1)

        worker = Worker(
            core_fn,
            [inX.cons(), inGB.cons(), outY.prod(), k_stats, k_fin, k_app],
            stack_size=0x1000,
        )

        X_ty = np.ndarray[(groups * L,), np.dtype[bfloat16]]
        GB_ty = np.ndarray[(gb_bytes,), np.dtype[np.int8]]
        Y_ty = np.ndarray[(groups * L,), np.dtype[bfloat16]]
        gb_tap = TensorAccessPattern.full((gb_bytes,)).tile((gb_bytes,))
        y_tap = TensorAccessPattern.full((groups * L,)).tile((Cs,))

        def sequence(x, gb, y, inX_h, inGB_h, outY_h):
            # Same choreography as make_gn_stream (drain pushed first and
            # awaited last; fills in TaskGroups of <= 8 so no more than 8
            # shim BDs are active).  See that factory for the two measured
            # failure modes this avoids.
            drain_tg = TaskGroup()
            outY_h.drain(y, tap=y_tap, group=drain_tg, wait=True)

            tgs = [TaskGroup()]
            inGB_h.fill(gb, tap=gb_tap, group=tgs[-1], wait=True)
            pushes = 1
            for g in range(groups):
                g_tap = (
                    TensorAccessPattern.full((groups * L,))
                    [g * L:(g + 1) * L].tile((Cs,))
                )
                for _p in range(2):        # pass 1, then pass 2
                    inX_h.fill(x, tap=g_tap, group=tgs[-1], wait=True)
                    pushes += 1
                    if pushes >= 8:
                        tgs[-1].finish()
                        del tgs[-1]
                        tgs.append(TaskGroup())
                        pushes = 0
            tgs[-1].finish()          # always close the trailing group
            del tgs[-1]
            drain_tg.finish()
            del drain_tg

        rt = Runtime(
            sequence,
            [X_ty, GB_ty, Y_ty, inX.prod(), inGB.prod(), outY.cons()],
        )
        return Program(
            iron.get_current_device(), rt, workers=[worker]
        ).resolve_program()

    return gn_stream_sub


def make_up_kernel_strip(C: int, H: int, W: int, rows: int | None = None,
                         batch_tiles: int = 512):
    """Nearest-2x upsample in ROW STRIPS: same kernel, smaller plane object.

    A plane of S = H*W elements needs 2S bytes of input object and 8S of
    output object in the 64 KB L1 (10*S + 8 KB stack); S = 6144..16384 costs
    60..160 KB and does not fit.  A strip of R rows is R*W elements in and
    4*R*W out, and nearest-2x maps input rows [r0, r1) to output rows
    [2r0, 2r1) -- a CONTIGUOUS run of the (2H, 2W) output plane.  So the
    whole op is C*H/R strip calls, and both the host fill and the host drain
    stay a single 1-dim tiled walk (in tile order: channel, then strip).

    batch_tiles caps the number of tiles per fill/drain TaskGroup: 512 is
    the largest walk any built upsample row uses today (C=512, S=1024), and
    the GD/shim descriptor plumbing is the same code path.
    """
    R = int(rows or VG.up_strip_rows(H, W))
    if H % R:
        raise ValueError(f"strip rows {R} do not divide H={H}")
    if not (10 * R * W + 2 * 0x1000 + 4096 <= 64 * 1024):
        raise ValueError(
            f"strip {R}x{W}: in+out objects are {10 * R * W} B, over the "
            f"64 KB L1")
    n = R * W
    ncall = C * H // R
    tiles = C * (H // R)

    @iron.jit(aiecc_flags=["--dynamic-objFifos"])
    def up_strip(X: In, Y: Out):
        x_ty = np.ndarray[(n,), np.dtype[bfloat16]]
        y_ty = np.ndarray[(4 * n,), np.dtype[bfloat16]]
        kern = K.upsample_nearest2(n, W)

        inX = ObjectFifo(x_ty, name="inX", depth=1)
        outY = ObjectFifo(y_ty, name="outY", depth=1)

        def core_fn(of_x, of_y, k):
            for _ in range_(ncall):
                x = of_x.acquire(1)
                y = of_y.acquire(1)
                k(x, y, n, W)
                of_x.release(1)
                of_y.release(1)

        worker = Worker(
            core_fn, [inX.cons(), outY.prod(), kern], stack_size=0x1000
        )

        X_ty = np.ndarray[(C * H * W,), np.dtype[bfloat16]]
        Y_ty = np.ndarray[(C * 4 * H * W,), np.dtype[bfloat16]]

        def sequence(x, y, inX_h, outY_h):
            # Per batch: fill (unwaited) then drain (waited), one TaskGroup,
            # the proven make_up_kernel shape -- one batch is one contiguous
            # run of BOTH buffers, so the tiles line up element for element.
            lo = 0
            while lo < tiles:
                hi = min(tiles, lo + batch_tiles)
                x_tap = (TensorAccessPattern.full((C * H * W,))
                         [lo * n:hi * n].tile((n,)))
                y_tap = (TensorAccessPattern.full((C * 4 * H * W,))
                         [lo * 4 * n:hi * 4 * n].tile((4 * n,)))
                tgs = TaskGroup()
                inX_h.fill(x, tap=x_tap, group=tgs)
                outY_h.drain(y, tap=y_tap, group=tgs, wait=True)
                tgs.finish()
                del tgs
                lo = hi

        rt = Runtime(sequence, [X_ty, Y_ty, inX.prod(), outY.cons()])
        return Program(
            iron.get_current_device(), rt, workers=[worker]
        ).resolve_program()

    return up_strip
