# SPDX-License-Identifier: MIT
# t7_attn_split.py -*- Python -*-
#
# T7 U-1: THE FUSED ATTENTION IMAGE IS SPLIT IN TWO -- projection-only (t7_proj)
# and ip-only (t7_ip).  Both are the delivered image's own code, verbatim; what
# changes is only WHICH PART IS IN THE IMAGE.
#
# WHY (this is the T5/T6 boundary turned into a design decision): the delivered
# image t5_u1 runs the projection AND the ip chain in every dispatch, so the
# product loop's five attention dispatches (q, k, v, ip, o) cost 5 x 11.98 ms
# even though only FOUR projections and ONE ip are useful: dispatch d4 (att/ip)
# re-runs the to_v projection and throws its Q away, and d5 (att/o) re-runs the
# whole ip and throws its O away (t6_run.py's own comment: "qp/kb/vb keep d4's
# bytes, so the ip re-runs on them and its O is discarded").  The ip is hidden
# behind the projection -- but the projection is NOT hidden behind the ip, and
# that is the part that costs.
#
# T5 section 10 measured the two halves in isolation, on the same payload:
#     projection alone, PROJ=2  11.00 ms   (PROJ=3 7.61, PROJ=5 5.21)
#     ip chain alone             3.23 ms   (T3d, ip image)
# so the split price is 4 x proj(PROJ) + 1 x ip:
#     PROJ=2: 4 x 11.00 + 3.23 = 47.2 ms/layer   (vs 5 x 11.98 = 59.9 fused)
#     PROJ=3: 4 x  7.61 + 3.23 = 33.7 ms/layer   (-26 ms/layer = -6.3 s / 240 layers)
#     PROJ=5: 4 x  5.21 + 3.23 = 24.1 ms/layer   (-36 ms/layer = -8.6 s / 240 layers)
# and it does NOT need the ip to coexist with PROJ>=3, which is the one thing
# that five rewrites (T5) plus two builds (T6) could not make work.
#
# ARITHMETIC: untouched by construction.  A head's Q is zeroed and then
# accumulated over its 60 k-groups in ascending order by ONE tile; the ip chain
# (qk -> outS -> sm -> pvA -> pv, each memtile forward verbatim) is the
# delivered one, head by head.  So the referees are the delivered image's own
# bytes:
#     t7_proj:  Q == t3b_out/t3d_pipe2_rawq.npy      byte-identical
#     t7_ip:    O == t3b_out/t3c_fused_o_raw.npy     byte-identical
# (both files are READ-ONLY references; t3d_baseline.py --check gates them.)
#
# Data arguments: t7_proj 3 (X, W -> OUTQ); t7_ip 4 (QP, KB, VB -> OUTO).  Both
# are under the ERT limit of 7, and every one is 64 B aligned.
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PYLIBS = _HERE.parent.parent.parent / "_pylibs"
import os
sys.path.append(str(_HERE))
sys.path.append(str(_PYLIBS))
sys.path.append(str(_PYLIBS / "mlir_aie" / "python"))
os.environ["PEANO_INSTALL_DIR"] = str(_PYLIBS / "llvm-aie" / "llvm-aie")
os.environ.setdefault("NPU_CACHE_HOME", str(_HERE / "t3b_cache"))

import aie.iron as iron
import aie.iron.kernels as kernels
import numpy as np
from aie.dialects.aiex import v8bfp16ebs8
from aie.extras.dialects.arith import addi, constant, shli
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (CompileTime, In, ObjectFifo, Out, Program, Runtime,
                      TaskGroup, Worker)
from aie.iron.controlflow import range_
from aie.iron.device import AnyMemTile, Tile
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli
from ml_dtypes import bfloat16

import aie.utils.compile.utils as _compile_utils

_orig_compile_cxx_core_function = _compile_utils.compile_cxx_core_function
_orig_cxx_core_compile_command = _compile_utils.cxx_core_compile_command


def _cxx_core_compile_command_shrunk(*args, **kwargs):
    return _orig_cxx_core_compile_command(*args, **kwargs) + ["-Oz"]


_compile_utils.cxx_core_compile_command = _cxx_core_compile_command_shrunk

_TRIM_SECTIONS = {
    "mm.cc": [".text.matmul_vectorized*"],
    "softmax.cc": [
        ".text.partial_softmax_bf16",
        ".text._Z26partial_softmax_alias_bf16P8bfloat16S0_S0_iiiS_",
        ".text.mask_bf16",
        ".text.softmax_rows_causal_bf16",
    ],
}


def _trim_object(output_path, drops):
    script = Path(str(output_path) + ".trim.ld")
    script.write_text(
        "SECTIONS {\n  /DISCARD/ : {\n"
        + "".join(f"    *({sec})\n" for sec in drops)
        + "} }\n"
    )
    trimmed = Path(str(output_path) + ".trim.o")
    r = subprocess.run(
        [str(_PYLIBS / "llvm-aie" / "llvm-aie" / "bin" / "ld.lld.exe"), "-r",
         f"--script={script}", "-o", str(trimmed), str(output_path)],
        capture_output=True)
    if r.returncode == 0:
        trimmed.replace(output_path)
    else:
        print(f"[trim] {Path(output_path).name}: "
              f"{r.stderr.decode(errors='replace')[:160]}")


def _compile_cxx_core_function_trimmed(source_path, target_arch, output_path,
                                      *args, **kwargs):
    res = _orig_compile_cxx_core_function(source_path, target_arch, output_path,
                                          *args, **kwargs)
    drops = _TRIM_SECTIONS.get(Path(source_path).name)
    if drops and str(output_path).endswith(".o"):
        _trim_object(output_path, drops)
    return res


_compile_utils.compile_cxx_core_function = _compile_cxx_core_function_trimmed

STACK_BYTES = 0x1800


# ---------------------------------------------------------------------------
# (a) the projection alone:   X, W -> Q
# ---------------------------------------------------------------------------
@iron.jit(aiecc_flags=["-O", "0"])
def t7_proj(
    X: In,
    W: In,
    OUTQ: Out,
    *,
    HEADS: CompileTime[int] = 30,
    M: CompileTime[int] = 96,
    KW: CompileTime[int] = 3840,
    m: CompileTime[int] = 96,
    k: CompileTime[int] = 64,
    n: CompileTime[int] = 128,
    PROJ: CompileTime[int] = 2,
    DEPTH: CompileTime[int] = 1,
):
    """The projection of t5_attn_pipe3.py with the ip chain removed.

    ONE variable against the delivered image: the fourth load (qk/sm/pv, their
    fills and their drain) is gone.  The projection cores, the k-group order,
    the fill/drain grouping and the tile map are the delivered ones verbatim.
    """
    assert k in (8, 64) and m % 16 == 0 and n == 128 and M == m
    assert KW % k == 0
    assert HEADS % PROJ == 0, "PROJ must divide HEADS"
    HALF = HEADS // PROJ
    NG = KW // k                          # 60 k-groups
    A_OBJ = m * k * 2                     # 12,288
    B_OBJ = k * n // 8 * 9                # 9,216
    Q_BYTES = m * n * 2                   # 24,576
    assert A_OBJ + B_OBJ + Q_BYTES + STACK_BYTES <= 65536, "proj L1"

    x_ty = np.ndarray[(A_OBJ,), np.dtype[np.int8]]
    w_ty = np.ndarray[(B_OBJ,), np.dtype[np.int8]]
    q_ty = np.ndarray[(m * n,), np.dtype[bfloat16]]

    fx = [ObjectFifo(x_ty, name=f"inX{i}", depth=DEPTH) for i in range(PROJ)]
    inW = [ObjectFifo(w_ty, name=f"inW{i}", depth=DEPTH) for i in range(PROJ)]
    outQ = [ObjectFifo(q_ty, name=f"outQ{i}", depth=DEPTH) for i in range(PROJ)]

    proj = kernels.mm_bfp(dim_m=m, dim_k=k, dim_n=n, mixed=True)
    zeroQ = kernels.zero(m * n, bfloat16)

    def core_proj(ox, ow, oq, matmul, zeroC):
        from aie.dialects import memref
        from aie.helpers.util import np_ndarray_type_to_memref_type
        am = np_ndarray_type_to_memref_type(
            np.ndarray[(m * k,), np.dtype[bfloat16]])
        bm = np_ndarray_type_to_memref_type(
            np.ndarray[(k * n // 8,), np.dtype[v8bfp16ebs8]])
        z0 = constant(0, index=True)
        for _h in range_(HALF) if HALF > 1 else range(1):
            ce = oq.acquire(1)
            zeroC(ce)
            for _g in range_(NG) if NG > 1 else range(1):
                we = ow.acquire(1)
                xe = ox.acquire(1)
                matmul(memref.view(am, xe, z0, []),
                       memref.view(bm, we, z0, []), ce)
                ox.release(1)
                ow.release(1)
            oq.release(1)

    PROJ_TILES = [Tile(0, 2)] + [Tile(1, 1 + i) for i in range(1, PROJ)]
    assert PROJ <= 5, "col 1 rows 2..5 hold at most 4 more projection tiles"
    workers = []
    for i in range(PROJ):
        workers.append(Worker(core_proj,
                              [fx[i].cons(), inW[i].cons(), outQ[i].prod(),
                               proj, zeroQ],
                              tile=PROJ_TILES[i], stack_size=STACK_BYTES))

    X_ty = np.ndarray[(NG * A_OBJ,), np.dtype[np.int8]]
    W_ty = np.ndarray[(HEADS * NG * B_OBJ,), np.dtype[np.int8]]
    OUTQ_ty = np.ndarray[(HEADS * m * n,), np.dtype[bfloat16]]
    X_tap = TensorAccessPattern.full((NG * A_OBJ,)).tile((A_OBJ,))
    W_tap = TensorAccessPattern.full((HEADS * NG * B_OBJ,)).tile((B_OBJ,))
    OUTQ_tap = TensorAccessPattern.full((HEADS * m * n,)).tile((m * n,))

    def sequence(x, w, outq, *handles):
        hx = handles[:PROJ]
        hw = handles[PROJ:2 * PROJ]
        hq = handles[2 * PROJ:3 * PROJ]
        for r in range(HALF):
            for g in range(NG):
                tg = TaskGroup()
                for i in range(PROJ):
                    hw[i].fill(w, tap=W_tap[(r + i * HALF) * NG + g],
                               group=tg, wait=True)
                    hx[i].fill(x, tap=X_tap[g], group=tg, wait=True)
                tg.finish()
                del tg
            # ONE TaskGroup PER DRAINED OBJECT (the delivered image's own rule:
            # three objects of one depth-1 fifo in one group deadlock).
            for i in range(PROJ):
                tg = TaskGroup()
                hq[i].drain(outq, tap=OUTQ_tap[r + i * HALF], group=tg,
                            wait=True)
                tg.finish()
                del tg

    rt_args = [X_ty, W_ty, OUTQ_ty]
    rt_args += [f.prod() for f in fx] + [f.prod() for f in inW]
    rt_args += [f.cons() for f in outQ]
    rt = Runtime(sequence, rt_args)
    return Program(iron.get_current_device(), rt,
                   workers=workers).resolve_program()


# ---------------------------------------------------------------------------
# (b) the ip chain alone:    QP, KB, VB -> O
# ---------------------------------------------------------------------------
@iron.jit(aiecc_flags=["-O", "0"])
def t7_ip(
    QP: In,
    KB: In,
    VB: In,
    OUTO: Out,
    *,
    HEADS: CompileTime[int] = 30,
    M: CompileTime[int] = 96,
    KD: CompileTime[int] = 128,
    ND: CompileTime[int] = 96,
):
    """The ip chain of t5_attn_pipe3.py with the projection removed.

    qk -> outS (memtile forward) -> sm -> pvA (memtile forward) -> pv, depth 1,
    tiles (0,3)/(0,4)/(0,5): the delivered code verbatim, including both
    forwards and the per-head fifo discipline.
    """
    KB_BYTES = KD * ND // 8 * 9           # 13,824
    S_BYTES = M * ND * 2                  # 18,432
    VB_BYTES = ND * KD // 8 * 9           # 13,824
    O_BYTES = M * KD * 2                  # 24,576
    assert KD % 8 == 0 and ND % 8 == 0
    assert 2 * S_BYTES <= 65536, "memtile L1"

    qp_ty = np.ndarray[(M * KD * 2,), np.dtype[np.int8]]
    kb_ty = np.ndarray[(KB_BYTES,), np.dtype[np.int8]]
    vb_ty = np.ndarray[(VB_BYTES,), np.dtype[np.int8]]
    s_ty = np.ndarray[(M * ND,), np.dtype[bfloat16]]
    o_ty = np.ndarray[(M * KD,), np.dtype[bfloat16]]

    inQP = ObjectFifo(qp_ty, name="inQP", depth=1)
    inKB = ObjectFifo(kb_ty, name="inKB", depth=1)
    inVB = ObjectFifo(vb_ty, name="inVB", depth=1)

    qk = kernels.mm_bfp(dim_m=M, dim_k=KD, dim_n=ND, mixed=True)
    zeroS = kernels.zero(M * ND, bfloat16)
    midS = ObjectFifo(s_ty, name="midS", depth=1)
    outS = midS.cons().forward(tile=AnyMemTile, name="outS",
                               to_stream=qk.stream_dims.C, depth=1)

    def core_qk(oa, ob, oc, matmul, zeroC):
        from aie.dialects import memref
        from aie.helpers.util import np_ndarray_type_to_memref_type
        am = np_ndarray_type_to_memref_type(
            np.ndarray[(M * KD,), np.dtype[bfloat16]])
        bm = np_ndarray_type_to_memref_type(
            np.ndarray[(KD * ND // 8,), np.dtype[v8bfp16ebs8]])
        z0 = constant(0, index=True)
        for _h in range_(HEADS) if HEADS > 1 else range(1):
            ae = oa.acquire(1)
            be = ob.acquire(1)
            ce = oc.acquire(1)
            zeroC(ce)
            matmul(memref.view(am, ae, z0, []), memref.view(bm, be, z0, []), ce)
            oa.release(1)
            ob.release(1)
            oc.release(1)

    workers = [Worker(core_qk, [inQP.cons(), inKB.cons(), midS.prod(), qk,
                                zeroS],
                      tile=Tile(0, 3), stack_size=STACK_BYTES)]

    sm_flat = kernels.softmax(tile_size=M * ND)
    sm_ty = np.ndarray[(M * ND,), np.dtype[bfloat16]]
    sm_rows = sm_flat.object_file.bind("softmax_rows_bf16",
                                       [sm_ty, sm_ty, np.int32, np.int32])
    midP = ObjectFifo(s_ty, name="midP", depth=1)

    def core_sm(os_, op, sm_rows):
        for _h in range_(HEADS) if HEADS > 1 else range(1):
            se = os_.acquire(1)
            pe = op.acquire(1)
            sm_rows(se, pe, M, ND)
            os_.release(1)
            op.release(1)

    workers.append(Worker(core_sm, [outS.cons(), midP.prod(), sm_rows],
                          tile=Tile(0, 4), stack_size=STACK_BYTES))

    pv = kernels.mm_bfp(dim_m=M, dim_k=ND, dim_n=KD, mixed=True)
    zeroO = kernels.zero(M * KD, bfloat16)
    pvA = midP.cons().forward(tile=AnyMemTile, name="pvA",
                              to_stream=pv.stream_dims.A, depth=1)
    outO = ObjectFifo(o_ty, name="outO", depth=1)

    def core_pv(oa, ob, oc, matmul, zeroC):
        from aie.dialects import memref
        from aie.helpers.util import np_ndarray_type_to_memref_type
        bm = np_ndarray_type_to_memref_type(
            np.ndarray[(ND * KD // 8,), np.dtype[v8bfp16ebs8]])
        z0 = constant(0, index=True)
        for _h in range_(HEADS) if HEADS > 1 else range(1):
            ae = oa.acquire(1)
            be = ob.acquire(1)
            ce = oc.acquire(1)
            zeroC(ce)
            matmul(ae, memref.view(bm, be, z0, []), ce)
            oa.release(1)
            ob.release(1)
            oc.release(1)

    workers.append(Worker(core_pv, [pvA.cons(), inVB.cons(), outO.prod(), pv,
                                    zeroO],
                          tile=Tile(0, 5), stack_size=STACK_BYTES))

    QP_ty = np.ndarray[(HEADS * M * KD * 2,), np.dtype[np.int8]]
    KB_ty = np.ndarray[(HEADS * KB_BYTES,), np.dtype[np.int8]]
    VB_ty = np.ndarray[(HEADS * VB_BYTES,), np.dtype[np.int8]]
    OUTO_ty = np.ndarray[(HEADS * M * KD,), np.dtype[bfloat16]]
    QP_tap = TensorAccessPattern.full((HEADS * M * KD * 2,)).tile((M * KD * 2,))
    KB_tap = TensorAccessPattern.full((HEADS * KB_BYTES,)).tile((KB_BYTES,))
    VB_tap = TensorAccessPattern.full((HEADS * VB_BYTES,)).tile((VB_BYTES,))
    OUTO_tap = TensorAccessPattern.full((HEADS * M * KD,)).tile((M * KD,))

    def sequence(qp, kb, vb, outo, *handles):
        hqp, hkb, hvb, ho = handles
        for h in range(HEADS):
            # the delivered per-head fill group (qp, kb, vb together)
            tg = TaskGroup()
            hqp.fill(qp, tap=QP_tap[h], group=tg, wait=True)
            hkb.fill(kb, tap=KB_tap[h], group=tg, wait=True)
            hvb.fill(vb, tap=VB_tap[h], group=tg, wait=True)
            tg.finish()
            del tg
            tg = TaskGroup()
            ho.drain(outo, tap=OUTO_tap[h], group=tg, wait=True)
            tg.finish()
            del tg

    rt_args = [QP_ty, KB_ty, VB_ty, OUTO_ty]
    rt_args += [inQP.prod(), inKB.prod(), inVB.prod(), outO.cons()]
    rt = Runtime(sequence, rt_args)
    return Program(iron.get_current_device(), rt,
                   workers=workers).resolve_program()


def _make_argparser(prog):
    p = argparse.ArgumentParser(prog=prog)
    add_compile_args(p, default_dev="npu2")
    p.add_argument("--heads", type=int, default=30)
    p.add_argument("--proj", type=int, default=2)
    p.add_argument("--depth", type=int, default=1)
    return p


def main():
    part = sys.argv[1] if len(sys.argv) > 1 else "proj"
    if part in ("proj", "ip"):
        sys.argv = [sys.argv[0]] + sys.argv[2:]
    if part == "ip":
        opts = _make_argparser("t7 ip").parse_args()
        run_design_cli(t7_ip, opts,
                       compile_kwargs=dict(HEADS=opts.heads),
                       device=device_from_args)
    else:
        opts = _make_argparser("t7 proj").parse_args()
        run_design_cli(t7_proj, opts,
                       compile_kwargs=dict(HEADS=opts.heads, PROJ=opts.proj,
                                           DEPTH=opts.depth),
                       device=device_from_args)


if __name__ == "__main__":
    main()
