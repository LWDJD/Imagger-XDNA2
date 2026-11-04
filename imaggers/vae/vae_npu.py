# vae_npu.py -*- Python -*-
#
# The Z-Image VAE decoder conv-equivalent for AMD XDNA2 (AIE2P / npu2), with
# BlockFP16 (`bfp16ebs8`) weights -- the project owner's decision: the VAE runs
# on the NPU too, in the same BFP16 residency story as the DiT, NOT bf16 and
# NOT the iGPU/DirectML path.
#
# THE FORM USED FOR THE CONV
# --------------------------
# Explicit GEMM after im2col.  A stride-1 k x k conv
#
#     Y[n, o, y, x] = b[o] + sum_{c,dy,dx} X[n, c, y+dy, x+dx] * W[o, c, dy, dx]
#
# is exactly one matrix product
#
#     C (M, N)  =  A (M, K) @ Wmat (K, N)  +  b
#     M = H*W          (one row per OUTPUT PIXEL)
#     K = Cin*kh*kw    (one col per (c, dy, dx); im2col row-major c-major)
#     N = Cout
#
# * im2col is a pure gather permutation (no arithmetic): done on the HOST into
#   the A buffer.  Everything that is ARITHMETIC runs on the NPU.
# * the bias is FOLDED INTO THE GEMM: A carries one extra k-block whose first
#   column is 1.0 and whose remaining 7 columns are 0; Wmat's matching k-block
#   holds [b, 0, 0, 0, 0, 0, 0, 0].  One mm_bfp call over that block adds b to
#   every output row, so the conv has no host-side epilogue at all.
# * the WEIGHTS are BFP16: the B operand is `v8bfp16ebs8` blocks (shared
#   exponent per 8 K-elements, 9 bytes / 8 values) -- `mm_bfp(mixed=True)` is
#   bf16 A/C + BlockFP16 B, i.e. exactly the "BFP16 weights" form.
# * K is padded to a multiple of 32 with zero columns (harmless: they
#   contribute nothing and keep every row view 64 B aligned).
#
# DATAFLOW (the proven `imaggers/memtile2/memtile_worker.py` architecture,
# minus the RMSNorm core: Worker A is a pure row-tile assembler here, because
# im2col lives on the host):
#
#   host A (M x K bf16 rows, ONE ROW per transfer)
#      |
#      v  inA fifo (row objects, K*2 bytes i8)
#   Worker A "tile assembler" @ (0,2): copies each row's Kc columns into the
#      |                              current (m, Kc) column-chunk object
#      v  outA fifo: object = ONE (m, Kc) column chunk (raw i8)
#      |  1:1 ObjectFifoLink @ memtile (0,1): outA.cons() -> midB.prod()
#      |  *** THE MEMTILE CORNER TURN *** (proven 100% bit-exact)
#      v  midB fifo: to_stream = corner_turn_tap(m, Kc),
#      |       consumer_obj_type = (m*k*2,) i8 -> ready (m, k) A slices
#   Worker B "GEMM core" @ (0,3): zero(C); for kk in K/8:
#      |                            mm_bfp(m, 8, n, mixed=True) on slice kk
#      v  memC fifo -> memtile forward hop (to_stream = stream_dims.C) -> host
#   host C (M x N bf16 row-major)
#
# MEASURED FACTS THIS RESTS ON (from imaggers/memtile2/FINDINGS.md):
#   * memtile DMA BDs accept 30,720 32-bit words; the COMPUTE cap is 16383.
#     chunk_cols() keeps the producer BD (8*Kc words) under it.
#   * mm_bfp_mixed.cc static_asserts m % 16 == 0, k % 8 == 0, n % 16 == 0 and
#     re-rounds C to bf16 after EVERY call -> the numpy reference models it.
#   * AIE2P vector loads silently floor misaligned addresses to 64 B; every
#     kernel view is pinned via contract.alignments.
#   * core_dma_channels_in == 2 per Worker: per-row params ride along in the
#     same stream (here: none; the bias is inside the GEMM).
#   * compute-tile DMA BDs take <= 3 dims, memtiles 4 -> the 4-dim C inverse
#     walk must live on the memtile hop.
#   * bfp16ebs8 strides must be 4-byte multiples: B blocks move as 9*n-byte
#     segments with n % 4 == 0 (n is the tile's N width, 16 here).
#
# The A/mid objects move as raw i8; typed rows/slices are memref.view'ed out of
# the 1-D i8 sources.

import argparse
import dataclasses
import os
import sys
from pathlib import Path

# mlir_aie setup: both package roots on sys.path, and the two environment
# variables must be in place BEFORE `import aie`.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent.parent                       # ...\public_work
_PYLIBS = _ROOT / "_pylibs"
sys.path.append(str(_PYLIBS))
sys.path.append(str(_PYLIBS / "mlir_aie" / "python"))
os.environ["PEANO_INSTALL_DIR"] = str(_PYLIBS / "llvm-aie" / "llvm-aie")
# FRESH NPU_CACHE_HOME per session (see run.py: a 0o700 mkdir poisons the
# shared object cache on this machine)
os.environ.setdefault("NPU_CACHE_HOME", str(_HERE / "_npucache"))

import aie.iron as iron
import aie.iron.kernels as kernels
import numpy as np
from aie.dialects import memref
from aie.dialects.aiex import v8bfp16ebs8
from aie.extras.dialects.arith import constant, muli
from aie.helpers.util import np_ndarray_type_to_memref_type
from aie.helpers.taplib import TensorAccessPattern
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


# --------------------------------------------------------------------------
# im2col + the conv reference arithmetic (numpy; the ORACLE is torch below)
# --------------------------------------------------------------------------

def im2col(x: np.ndarray, kh: int = 3, kw: int = 3,
           pad: int = 1) -> np.ndarray:
    """(1, Cin, H, W) -> (H*W, Cin*kh*kw), row p = output pixel (y, x).

    Column order is c-major then dy then dx: col = c*kh*kw + dy*kw + dx.
    Out-of-image taps are zero (zero padding), matching Conv2d(padding=1).
    """
    x = np.asarray(x, np.float32)
    if x.ndim == 4:
        x = x[0]
    C, H, W = x.shape
    xp = np.zeros((C, H + 2 * pad, W + 2 * pad), np.float32)
    xp[:, pad:pad + H, pad:pad + W] = x
    out = np.empty((H * W, C * kh * kw), np.float32)
    for y in range(H):
        for xcol in range(W):
            patch = xp[:, y:y + kh, xcol:xcol + kw]     # (C, kh, kw)
            out[y * W + xcol, :] = patch.reshape(-1)
    return out


def conv_matmul_ref(x: np.ndarray, W: np.ndarray, b: np.ndarray | None,
                    *, kh: int = 3, kw: int = 3, pad: int = 1,
                    acc_dtype=np.float64) -> np.ndarray:
    """Float reference of the conv-as-GEMM: out = im2col(x) @ Wmat + b.

    W is the checkpoint layout (Cout, Cin, kh, kw); Wmat = W.reshape(Cout, -1).T
    is (Cin*kh*kw, Cout).  Returns (1, Cout, H, W).
    """
    x = np.asarray(x, np.float32)
    C, H, Wd = x.shape[1], x.shape[2], x.shape[3]
    A = im2col(x, kh, kw, pad).astype(acc_dtype)
    Wmat = np.asarray(W, acc_dtype).reshape(W.shape[0], -1).T   # (K, N)
    Y = A @ Wmat
    if b is not None:
        Y = Y + np.asarray(b, acc_dtype)[None, :]
    return Y.T.reshape(W.shape[0], H, Wd)[None]


def pad_kn(Wmat: np.ndarray, K: int, N: int, bias_row: np.ndarray | None):
    """(K_real, N_real) conv/linear matrix -> (K, N) with the bias folded.

    Layout of the padded K rows:
        [0, K_real)               real im2col rows
        [K_real, K_real+8)        one k-block: row K_real = bias, rows +1..+7 = 0
                                  (only when bias_row is given)
        rest                      zero rows
    Padded N columns beyond N_real are zero (their outputs are discarded).
    """
    Kr, Nr = Wmat.shape
    out = np.zeros((K, N), np.float32)
    out[:Kr, :Nr] = Wmat
    used = Kr
    if bias_row is not None:
        assert used + 8 <= K, f"no room for the bias block: K={K} used={used}"
        out[used, :Nr] = np.asarray(bias_row, np.float32)
        used += 8
    return out, used


def pad_a(A: np.ndarray, K: int, k_bias_col: int | None):
    """(M, K_real) im2col matrix -> (M, K): zeros + the ones column for the
    folded bias block (column k_bias_col == 1.0)."""
    M, Kr = A.shape
    out = np.zeros((M, K), np.float32)
    out[:, :Kr] = A
    if k_bias_col is not None:
        out[:, k_bias_col] = 1.0
    return out


# --------------------------------------------------------------------------
# the DMA walks (copied VERBATIM from imaggers/memtile2/memtile_worker.py,
# where they are proven bit-exact; see that file's docstrings)
# --------------------------------------------------------------------------


def corner_turn_tap(m: int, K: int) -> TensorAccessPattern:
    """(m, K) row-major -> slice-major: for kk, for r: 16 bytes = 8 bf16."""
    assert K % 8 == 0, f"K must be a multiple of 8, got {K}"
    assert m >= 1 and K >= 8
    return TensorAccessPattern((m, 2 * K), 0, [K // 8, m, 16], [16, 2 * K, 1])


def a_host_tap(M: int, K: int, m: int) -> TensorAccessPattern:
    """Host A (tokens x hidden, bf16 row-major) -> one (1, K) row object."""
    return TensorAccessPattern.full((M, K)).tile((1, K))


def b_host_tap(N: int, K: int, k: int, n: int) -> TensorAccessPattern:
    """Host B blocks -> per-GEMM (k, n) B tiles in ((jj, kk), kk fastest) order."""
    assert k == 8, f"b_host_tap's pack order assumes k==8, got {k}"
    return TensorAccessPattern.full((K * N // 8,)).tile((k * n // 8,))


def c_host_tap(M: int, N: int, m: int, n: int) -> TensorAccessPattern:
    """Host C (tokens x out, bf16 row-major) <- one (m, n) tile per transfer."""
    return TensorAccessPattern.full((M, N)).tile((m, n))


def require_64B_alignment(kernel, pairs):
    """Enforce `contract.alignments` = ((arg, 64), ...) on a kernel."""
    kernel.contract = dataclasses.replace(
        kernel.contract, alignments=tuple(pairs)
    )
    return kernel


def chunk_cols(K: int, cap: int = 960) -> int:
    """Largest Kc that divides K with Kc % 32 == 0 and Kc <= cap."""
    if K % 32:
        raise ValueError(f"K must be a multiple of 32, got {K}")
    kc = min(K, cap) // 32 * 32
    while kc > 32 and K % kc:
        kc -= 32
    return kc


# --------------------------------------------------------------------------
# B host encoding: bfp16ebs8 blocks in (jj, kk, row) order.
# Copied VERBATIM from imaggers/memtile2/run.py (proven there).
# --------------------------------------------------------------------------


def pack_b_blocks(W, K, N, k, n) -> bytes:
    """Encode W (K, N) as bfp16ebs8 blocks and pack them in (jj, kk, row)
    order: flat block (jj*K_div_k + kk)*n + r holds W^T row jj*n+r, hidden
    block kk.
    """
    assert k == 8, f"pack_b_blocks assumes k==8, got {k}"
    WT = np.ascontiguousarray(np.asarray(W, np.float32).T.copy())
    from aie.utils import bfp as B
    enc = B.encode(WT.reshape(-1))
    enc = (
        np.asarray(enc, dtype=np.uint8)
        if isinstance(enc, np.ndarray)
        else np.frombuffer(bytes(enc), np.uint8)
    )
    blocks = enc.reshape(N, K // 8, 9)
    packed = (
        blocks.reshape(N // n, n, K // k, k // 8, 9)
        .transpose(0, 2, 1, 3, 4)
        .reshape(-1)
    )
    return np.ascontiguousarray(packed).tobytes()


# --------------------------------------------------------------------------
# the design
# --------------------------------------------------------------------------


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def conv_gemm(
    A: In,
    B: In,
    C: Out,
    *,
    M: CompileTime[int] = 64,
    K: CompileTime[int] = 160,
    N: CompileTime[int] = 512,
    m: CompileTime[int] = 16,
    k: CompileTime[int] = 8,
    n: CompileTime[int] = 16,
    inA_depth: CompileTime[int] = 2,
    mid_depth: CompileTime[int] = 1,
    Kc: CompileTime[int] = 0,
    rep_cap: CompileTime[int] = 0,
):
    """C (M x N bf16) = A (M x K bf16) @ B (bfp16ebs8), BFP16 weights.

    A is the host-side im2col of the conv input (plus the folded-bias ones
    column); B is the packed conv weight matrix.  The conv's bias lives in B.
    """
    # --- contract checks (fail at design time, not in Peano) -----------------
    assert k == 8, (
        f"conv_gemm: per-GEMM k must be 8 (the micro-tile s dim) so every A "
        f"slice is contiguous and the corner turn stays a 3-dim DMA walk; "
        f"got k={k}"
    )
    assert K % k == 0 and K >= k, f"K must be a positive multiple of k, got K={K}"
    assert K % 32 == 0, (
        f"K must be a multiple of 32 so every row view (byte offset r*K*2) is "
        f"64 B aligned (AIE2P floors misaligned vector accesses to 64 B "
        f"silently); got K={K}"
    )
    assert m % 16 == 0, f"mm_bfp_mixed.cc: m % 16 == 0 required, got m={m}"
    assert n % 16 == 0, f"mm_bfp_mixed.cc: n % 16 == 0 required, got n={n}"
    assert M % m == 0 and N % n == 0, "M % m and N % n must be 0"
    assert M > 0 and N > 0

    if Kc <= 0:
        Kc = chunk_cols(K)
    assert K % Kc == 0 and Kc >= 32, f"Kc must divide K, got K={K} Kc={Kc}"
    assert Kc % 32 == 0, f"Kc must be a multiple of 32, got {Kc}"
    assert 8 * Kc <= 16383, (
        f"producer BD is 8*Kc = {8 * Kc} words; the compute-tile cap is 16383"
    )
    n_chunks = K // Kc

    K_div_k = K // k
    M_div_m = M // m
    N_div_n = N // n
    tiles = M_div_m * N_div_n

    a_raw_ty = np.ndarray[(m * Kc * 2,), np.dtype[np.int8]]
    row_raw_ty = np.ndarray[(K * 2,), np.dtype[np.int8]]
    row_ty = np.ndarray[(K,), np.dtype[bfloat16]]
    chunk_row_ty = np.ndarray[(Kc,), np.dtype[bfloat16]]
    slice_raw_ty = np.ndarray[(m * k * 2,), np.dtype[np.int8]]
    slice_ty = np.ndarray[(m * k,), np.dtype[bfloat16]]
    b_ty = np.ndarray[(k * n // 8,), np.dtype[v8bfp16ebs8]]
    c_ty = np.ndarray[(m * n,), np.dtype[bfloat16]]

    matmul_kernel = require_64B_alignment(
        kernels.mm_bfp(dim_m=m, dim_k=k, dim_n=n, mixed=True),
        ((0, 64), (2, 64)),
    )
    zero_kernel = require_64B_alignment(kernels.zero(m * n, bfloat16), ((0, 64),))

    # --- data movement ------------------------------------------------------
    inA = ObjectFifo(row_raw_ty, name="inA", depth=inA_depth)
    outA = ObjectFifo(a_raw_ty, name="outA", depth=mid_depth)
    midB = ObjectFifo(
        a_raw_ty,
        name="midB",
        depth=mid_depth,
        to_stream=corner_turn_tap(m, Kc),
        consumer_obj_type=slice_raw_ty,
    )
    # 1:1 hop at the memtile (0,1): outA's and midB's buffers are ONE shared
    # pool at the memtile; the corner walk is midB's memtile MM2S DMA over it.
    _ = ObjectFifoLink(outA.cons(), midB.prod(), Tile(0, 1), [], [])
    a_cons = midB.cons()

    inB = ObjectFifo(b_ty, name="inB")
    # C leaves in mm_bfp's blocked layout; the 4-dim inverse walk runs on a
    # memtile forward hop (compute-tile DMA BDs take at most 3 dims).
    memC = ObjectFifo(c_ty, name="memC")
    c_dims = matmul_kernel.stream_dims.C
    outC = memC.cons().forward(name="outC", to_stream=c_dims)

    def tile_assembler(of_a, of_out):
        # No kernel here: the host's im2col is already in each row; this core
        # only assembles rows into (m, Kc) column chunks.  MLIR types are
        # context-bound: build them during worker tracing.
        row_memref_ty = np_ndarray_type_to_memref_type(row_ty)
        chunk_memref_ty = np_ndarray_type_to_memref_type(chunk_row_ty)
        loop = range_(tiles) if tiles > 1 else range(1)
        for _ in loop:
            cloop = range_(n_chunks) if n_chunks > 1 else range(1)
            for c in cloop:
                elem_out = of_out.acquire(1)   # one (m, Kc) chunk object
                if n_chunks > 1:
                    src_off = muli(c, constant(Kc * 2, index=True))
                else:
                    src_off = constant(0, index=True)
                for r in range(m):             # static
                    elem_in = of_a.acquire(1)  # one (K*2) row object
                    src = memref.view(chunk_memref_ty, elem_in, src_off, [])
                    dst = memref.view(
                        chunk_memref_ty,
                        elem_out,
                        constant(r * Kc * 2, index=True),
                        [],
                    )
                    jloop = range_(Kc) if Kc > 1 else range(1)
                    for j in jloop:
                        memref.store(memref.load(src, [j]), dst, [j])
                    of_a.release(1)
                of_out.release(1)

    worker_a = Worker(
        tile_assembler,
        [inA.cons(), outA.prod()],
        tile=Tile(0, 2),
        stack_size=0xF00,
    )

    def gemm_core(of_a, of_b, of_c, zero, matmul):
        slice_memref_ty = np_ndarray_type_to_memref_type(slice_ty)
        loop = range_(tiles) if tiles > 1 else range(1)
        for _ in loop:
            elem_c = of_c.acquire(1)
            zero(elem_c)
            kloop = range_(K_div_k) if K_div_k > 1 else range(1)
            for _kk in kloop:
                elem_b = of_b.acquire(1)
                elem_a = of_a.acquire(1)
                a_slice = memref.view(
                    slice_memref_ty, elem_a, constant(0, index=True), []
                )
                matmul(a_slice, elem_b, elem_c)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    worker_b = Worker(
        gemm_core,
        [a_cons, inB.cons(), memC.prod(), zero_kernel, matmul_kernel],
        tile=Tile(0, 3),
        stack_size=0xF00,
    )

    # --- runtime sequence (dit2/memtile2's proven TaskGroup choreography) ---
    A_ty = np.ndarray[(M * K,), np.dtype[bfloat16]]
    B_ty = np.ndarray[(K * N // 8,), np.dtype[v8bfp16ebs8]]
    C_ty = np.ndarray[(M * N,), np.dtype[bfloat16]]

    rows_per_block = 4

    A_rows = a_host_tap(M, K, m)
    b_tap = b_host_tap(N, K, k, n)
    C_tiles = c_host_tap(M, N, m, n)
    row_repeats = N_div_n * n_chunks

    def sequence(a, b, c, inA_h, inB_h, outC_h):
        tgs = []
        for tile_row_block in range(iron.ceildiv(M_div_m, rows_per_block)):
            for pingpong in [0, 1]:
                row_base = (
                    tile_row_block * rows_per_block + pingpong * rows_per_block // 2
                )
                num_tile_rows = min([rows_per_block // 2, M_div_m - row_base])
                if num_tile_rows <= 0:
                    break
                tgs.append(TaskGroup())
                for tile_row in range(num_tile_rows):
                    r0 = (row_base + tile_row) * m
                    # t21_3b2_vf: rep_cap > 0 splits the IN-A fill's repeat
                    # walk into ceil(row_repeats/rep_cap) fills over the SAME
                    # m-row tape in the SAME order, so the object stream -- and
                    # therefore every value -- is identical; only the number of
                    # DMA descriptors changes.  Default 0 = the 3b2 emission,
                    # byte for byte (proved by a recompile diff).
                    if rep_cap and row_repeats > rep_cap:
                        tape = A_rows[r0 : r0 + m]
                        done = 0
                        while done < row_repeats:
                            nrep = min(int(rep_cap), row_repeats - done)
                            inA_h.fill(a, tap=tape.repeat(nrep), group=tgs[-1])
                            done += nrep
                    else:
                        a_tap = A_rows[r0 : r0 + m].repeat(row_repeats)
                        inA_h.fill(a, tap=a_tap, group=tgs[-1])
                    inB_h.fill(b, tap=b_tap, group=tgs[-1])
                c_tap = C_tiles[row_base : row_base + num_tile_rows]
                outC_h.drain(c, tap=c_tap, group=tgs[-1], wait=True)
                if tile_row_block > 0 or (tile_row_block == 0 and pingpong > 0):
                    tgs[-2].finish()
                    del tgs[-2]
        tgs[-1].finish()
        del tgs[-1]

    rt = Runtime(
        sequence,
        [A_ty, B_ty, C_ty, inA.prod(), inB.prod(), outC.cons()],
    )

    return Program(
        iron.get_current_device(), rt, workers=[worker_a, worker_b]
    ).resolve_program()


# --------------------------------------------------------------------------
# the numpy device model: what mm_bfp(mixed=True) computes, exactly.
# (copied in structure from imaggers/memtile2/run.py's ref_model loop; the
#  tile oracle is aie.iron.kernels.linalg.mm_bfp_tile_ref)
# --------------------------------------------------------------------------


def bfp_gemm_model(A_b: np.ndarray, W: np.ndarray, *, m: int, k: int, n: int,
                   use_tile_ref: bool = True):
    """Device model of C = A @ W with mm_bfp(mixed=True) semantics.

    A_b is the EXACT bf16 operand the host sends (already rounded), W is the
    fp32 matrix whose bfp16ebs8 encoding is the B operand.  Per GEMM call the
    mixed kernel loads C (bf16), MACs over an f32 accumulator and stores C back
    as bf16 -> every one of the K/k calls contributes one bf16 rounding.
    Returns (ref_quant, ref_model) both float64:
        ref_quant  sum of the per-call products without intermediate stores
        ref_model  the per-call-bf16 accumulation (what the device computes)
    """
    from aie.iron.kernels.linalg import mm_bfp_tile_ref

    M, K = A_b.shape
    N = W.shape[1]
    ref_quant = np.zeros((M, N), np.float64)
    ref_model = np.zeros((M, N), np.float64)
    for ii in range(M // m):
        for jj in range(N // n):
            acc64 = np.zeros((m, n), np.float64)
            accm = np.zeros((m, n), np.float32)
            for kk in range(K // k):
                if use_tile_ref:
                    r = mm_bfp_tile_ref(
                        A_b[ii * m:(ii + 1) * m, kk * k:(kk + 1) * k],
                        W[kk * k:(kk + 1) * k, jj * n:(jj + 1) * n],
                        dim_m=m, dim_k=k, dim_n=n, mixed=True,
                    ).reshape(m, n)
                else:  # manual model of the same arithmetic (cross-check)
                    At = A_b[ii * m:(ii + 1) * m, kk * k:(kk + 1) * k]
                    Bt = W[kk * k:(kk + 1) * k, jj * n:(jj + 1) * n]
                    r = At.astype(np.float32) @ Bt.astype(np.float32)
                acc64 += r
                accm = (accm + r.astype(np.float32)).astype(bfloat16).astype(
                    np.float32
                )
            ref_quant[ii * m:(ii + 1) * m, jj * n:(jj + 1) * n] = acc64
            ref_model[ii * m:(ii + 1) * m, jj * n:(jj + 1) * n] = (
                accm.astype(np.float64)
            )
    return ref_quant, ref_model


def bf16_ulp(x):
    x = np.asarray(x, np.float64)
    bits = x.view(np.uint64)
    exp = ((bits >> 52) & 0x7FF).astype(np.int64) - 1023
    return np.exp2(np.maximum(exp - 7, -126))


def _make_argparser():
    p = argparse.ArgumentParser(
        prog="conv_gemm: VAE conv-equivalent (im2col GEMM) with BFP16 weights",
        description=(
            "Host im2col + NPU mm_bfp(mixed=True): bf16 A/C, bfp16ebs8 B, "
            "bias folded into the GEMM. NPU2."
        ),
    )
    p.add_argument("--dev", default="npu2")
    p.add_argument("-M", type=int, default=64)
    p.add_argument("-K", type=int, default=160)
    p.add_argument("-N", type=int, default=512)
    p.add_argument("-m", type=int, default=16)
    p.add_argument("-k", type=int, default=8)
    p.add_argument("-n", type=int, default=16)
    p.add_argument("--Kc", type=int, default=0, help="producer chunk cols (0=auto)")
    return p


if __name__ == "__main__":
    opts = _make_argparser().parse_args()
    print(f"conv_gemm geometry M={opts.M} K={opts.K} N={opts.N} "
          f"tile ({opts.m},{opts.k},{opts.n})")
