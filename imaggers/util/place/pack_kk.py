# SPDX-License-Identifier: MIT
"""pack_kk.py -- kk-parametric HOST packers for the GEMM A and B operands.

The shipped packers (s2_ffn32.pack_a / s2_ffn32.pack_b_from_blocks) and the
t33 copies (verify_t33.pack_a_t33 / pack_b_t33) are hard-wired to KK=8 in the
*batch* axis: their per-element layout happens to collapse to row-major only
when dim_k == 8, because at dim_k=8 every (8 rows x 8 values) mmul sub-tile is
exactly one row of the tile.  At dim_k >= 16 the micro-kernel walks the tile in
8x8 sub-tiles and the row-major image is WRONG BYTES (not a wrong answer: the
kernel reads a different matrix).  This module emits the bytes the kernel
actually reads, at dim_k = 8, 16, 32, 64 (any multiple of 8).

Everything below is derived from the library source, not guessed.  Line numbers
are for the checkout under <workspace>/_pylibs as of 2026-11-02.

1. What mm_bfp declares
-----------------------
_pylibs/mlir_aie/python/aie/iron/kernels/linalg.py

  mm_bfp()                                      (l.920, layouts at l.971-985)
      A : _block_layout((dim_m, dim_k))                     when mixed=False
          _tile_layout((dim_m, dim_k), streams.A, block=(8,8))   when mixed=True
      B : _block_layout((dim_k, dim_n), axes=(1, 0))          always
      C : _tile_layout((dim_m, dim_n), streams.C, inverse=True, block=(8,8))
                                                              when mixed=True
          _block_layout((dim_m, dim_n))                       when mixed=False
  _block_layout(shape, axes=...).pack          (l.252-278)
      per tile:  bfp.shuffle(bfp.encode(ascontiguousarray(value)), w, h, w, h)
      with h, w = the (optionally transposed) shape.  Because tile_* == (h, w)
      the whole tile is one shuffle tile, so (aie/utils/bfp.py:188-224,
      ty = tx = 1) the byte image is
          [subY][subX][8 rows][9 bytes]        subY = h//8, subX = w//8,
      i.e. 8-row x 8-value sub-tiles (72 B) contiguous in raster order.
      axes=(1,0) on (dim_k, dim_n) transposes to (dim_n, dim_k) *before*
      encoding, so B's shared exponent and its 9-byte block run along K, and
      the sub-tile grid is [n/8][k/8].
  _tile_layout(shape, dims, block=(8,8))       (l.228-249)
      pack is a pure element gather by 'order'; for A, dims = streams.A =
      mm_stream_dims(m, k, n, (8,8,8)).A = TensorAccessPattern.full((m,k))
      .tile((8,8))                            (l.380-412), i.e. the usual
      (8,8) micro-tile blocking: [m/8][k/8][8][8] elements.

2. What the kernels read  (the tie-breaker)
-------------------------------------------
_pylibs/mlir_aie/include/aie_kernels/linalg/mm_bfp.cc:180-264
  matmul_vectorized_2x2_bfp16<rowA, colA, colB, 8, 8, 8> with
  rowA = dim_m/8, colA = dim_k/8, colB = dim_n/8.
      pAIn.seek(z * colA);  pBIn.seek(j * colA);
  A is addressed as sub-tile  z*colA + kb   (z = 8-row group, kb = k-block),
  B as                       j*colA + kb   (j = 8-column group, kb = k-block),
  and one pop() is one 64-value block vector = one 8x8 sub-tile (72 B for
  bfp16ebs8).  pop_seek(inc) is "pop, then advance inc vectors"
  (aie_api/detail/aie2p/array_helpers.hpp:567-591), which is what makes the
  even-colA body read k-blocks (i, i+1) of the same two row/column groups.

_pylibs/mlir_aie/include/aie_kernels/linalg/mm_bfp_mixed.cc:15-97
  same 2x2 walk with bf16 A: pA2 = pArow + colA*sizeA, pA1 += sizeA with
  sizeA = r*s = 64 bf16 elements, so A is the same [m/8][k/8][8][8] grid with
  2-byte elements; B is unchanged (seek(j*colA), one bfp16 sub-tile per pop).

So, for ONE kernel call tile (dim_m x dim_k x dim_n), from byte 0:

  A (bf16, mixed=True), dim_m*dim_k*2 bytes
      off = ((y*nbx + kb) * 128) + i*16 + j*2      -> bf16 A[y*8+i][kb*8+j]
      y < m/8, kb < nbx = dim_k/8, i < 8, j < 8
  A (bfp16ebs8, mixed=False), 9*(dim_m*dim_k/8) bytes
      off = ((y*nbx + kb) * 72) + i*9 + b          -> block (i, b) of
            bfp.encode(A rows y*8..y*8+7, k-cols kb*8..kb*8+7)
  B (bfp16ebs8), 9*(dim_k*dim_n/8) bytes, B stored as B^T (n, k)
      off = ((j*nbx + kb) * 72) + i*9 + b          -> block (i, b) of the
            bfp16 block of B[kb*8 .. kb*8+7][j*8+i]

with nbx = dim_k/8 throughout.  Note that at dim_k = 8 (nbx = 1) the A and B
images degenerate to the row-major (m, 8) / (n, 8) matrices of 9-byte (or
2-byte) rows, which is exactly why the shipped KK=8 packers are right and why
nothing else was.

3. What this module adds
------------------------
The design (ffn_design_x.ffn_x) sizes its buffers as

    A_ty = (cols, nkk, a_par)  with a_par = cpc*a_leaf, split at p*a_leaf
    B_ty = (ntiles, nkk, kk*n//8)  , full((ntiles, nkk, ...))
    a_leaf = m*kk*2 (mixed) or 9*(m*kk//8), b_leaf = 9*(kk*n//8)

so the global order is [col][k-step][core][one kernel tile] for A and
[n-tile][k-step][one kernel tile] for B.  At kk=8 that is byte-identical to
s2_ffn32.pack_a / s2_ffn32.pack_b_from_blocks (the regression anchor); at
kk>8 the k-step slice is kk wide, so each leaf holds nbx = kk/8 sub-tiles
instead of one, in the order proved above.

Public API
----------
    pack_a_kk(x, m, kk, cols=8, cpc=4, out=None)   -> uint8 (M*K*2,) bf16
    pack_b_kk(blocks, K, N, n, kk, out=None)       -> uint8 (N*K*9//8,) bfp16

HOST ONLY: numpy + numba.  No aie, no pyxrt, no xclbin, no device.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "BLOCK", "BLOCK_BYTES", "A_ELEM_BYTES", "B_BLOCK_BYTES", "VERIFIED_KK",
    "LAYOUT", "tile_map", "a_leaf_bytes", "b_leaf_bytes",
    "a_payload_bytes", "b_payload_bytes",
    "pack_a_kk", "pack_b_kk", "pack_a_kk_ref", "pack_b_kk_ref",
    "unpack_a_kk", "unpack_b_kk", "have_numba",
]
#: the FUSED gu's B object (the [gate|up] pair); additive, see pair_b below
__all__ += ["b_pair_bytes", "pair_b", "unpair_b"]

BLOCK = 8                    # values per v8bfp16ebs8 block (blocks run along K)
BLOCK_BYTES = 9              # [exponent, m0..m7]
A_ELEM_BYTES = 2             # bf16 A (mm_bfp(mixed=True), mm_bfp_mixed.cc)
B_BLOCK_BYTES = 9
SUBTILE_VALUES = 64          # 8x8: one block_vector pop()
VERIFIED_KK = (8, 16, 32, 64)

#: machine-readable form of the derivation in the module docstring.
LAYOUT = {
    "a_leaf": "((y*nbx + kb)*%d) + i*%d + j*%d  -> A[y*8+i][kb*8+j]"
              % (BLOCK * BLOCK * A_ELEM_BYTES, BLOCK * A_ELEM_BYTES, A_ELEM_BYTES),
    "b_leaf": "((j*nbx + kb)*%d) + i*%d + b  -> block of B[kb*8..kb*8+7][j*8+i]"
              % (BLOCK * BLOCK_BYTES, BLOCK_BYTES),
    "a_payload": "[col][k-step][core][leaf]",
    "b_payload": "[n-tile][k-step][leaf]",
    "a_kernel": "aie_kernels/linalg/mm_bfp_mixed.cc:41 pB1bfp16.seek(j*colA); "
                "pA2 = pArow + colA*sizeA; pA1 += sizeA",
    "b_kernel": "aie_kernels/linalg/mm_bfp.cc:210 pBIn.seek(j*colA)",
    "layouts": "aie/iron/kernels/linalg.py:971-985 _block_layout/_tile_layout",
}

# ---------------------------------------------------------------------------
# numba kernels (optional; the numpy path below is the specification)
# ---------------------------------------------------------------------------
_NB = None


def _numba_kernels():
    global _NB
    if _NB is None:
        try:
            from numba import njit, prange
        except Exception:                                   # pragma: no cover
            _NB = False
            return None

        @njit(cache=True, parallel=True, nogil=True, boundscheck=False,
              fastmath=False)
        def _pack_a_k(xu, tm, out16, m, kk, cols, cpc, nkk, K):
            """xu uint32 view of (M,K) f32; out16 uint16 view of the payload.

            One prange task per output leaf, [col][k-step][core], each writing
            one contiguous run of m*kk bf16 values in [y][kb][i][j] order.
            """
            nby = m >> 3
            nbx = kk >> 3
            per = m * kk
            for q in prange(cols * nkk * cpc):
                p = q % cpc
                rest = q // cpc
                s = rest % nkk
                c = rest // nkk
                t = tm[c, p]
                if t < 0:
                    t = 0
                c0 = s * kk
                base = q * per
                for sy in range(nby):
                    r0 = t * m + sy * 8
                    for sx in range(nbx):
                        c1 = c0 + sx * 8
                        ob = base + ((sy * nbx + sx) << 6)
                        for i in range(8):
                            src = (r0 + i) * K + c1
                            dst = ob + (i << 3)
                            for j in range(8):
                                u = xu[src + j]
                                lsb = (u >> 16) & 1
                                out16[dst + j] = (u + 0x7FFF + lsb) >> 16

        @njit(cache=True, parallel=True, nogil=True, boundscheck=False,
              fastmath=False)
        def _pack_b_k(blocks, out, nt, nkk, n, kk, kblocks):
            """blocks (N, K//8, 9) uint8 -> out, one leaf per (n-tile, k-step).

            Leaf order [j (n/8)][kb (kk/8)][8 n-rows][9 B]; a sub-tile is the
            8 blocks of 8 consecutive n rows at one k-block, 72 contiguous
            bytes, which is what one block_vector pop() of B reads.
            """
            kbp = kk >> 3
            jmax = n >> 3
            per = n * kbp * BLOCK_BYTES
            for q in prange(nt * nkk):
                s = q % nkk
                t = q // nkk
                base = q * per
                kb0 = s * kbp
                for j in range(jmax):
                    row0 = t * n + j * 8
                    for kb in range(kbp):
                        ob = base + (j * kbp + kb) * 72
                        for i in range(8):
                            src = ((row0 + i) * kblocks + kb0 + kb) * 9
                            dst = ob + i * 9
                            for b in range(9):
                                out[dst + b] = blocks[src + b]

        _NB = (_pack_a_k, _pack_b_k)
    # ``_NB`` is False once the numba import failed, and False is NOT a kernel
    # pair: returning it made ``pack_a_kk``/``pack_b_kk`` do ``nb[0]`` on a bool
    # (TypeError: 'bool' object is not subscriptable) on every call AFTER the
    # first, i.e. exactly the host-only (py312, no numba) leg.  ``or None`` keeps
    # the documented contract "None means use the numpy reference".
    return _NB or None


def have_numba() -> bool:
    """Whether the numba fast path is importable in this interpreter."""
    return _numba_kernels() is not None


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def tile_map(M, m, cols, cpc):
    """M-tile index per (column, core); contiguous per column, -1 = dummy.

    Identical to s2_ffn32.tile_map (and ffn_design_x.tile_map), repeated here
    so this module stays import-free.
    """
    nm = M // m
    base, rem = divmod(nm, cols)
    out, nxt = [], 0
    for c in range(cols):
        cnt = base + (1 if c < rem else 0)
        out.append([nxt + p if p < cnt else -1 for p in range(cpc)])
        nxt += cnt
    if nxt != nm:
        raise ValueError(
            "tile_map: %d M-tiles do not fit %d columns x %d cores"
            % (nm, cols, cpc))
    return out


def a_leaf_bytes(m, kk):
    """Bytes of one (m, kk) bf16 A leaf: 2 per value.  == a_par/cpc."""
    return m * kk * A_ELEM_BYTES


def b_leaf_bytes(n, kk):
    """Bytes of one (kk, n) bfp16ebs8 B leaf: 9 per 8 values.  == b_ty entry."""
    return (n * kk // BLOCK) * B_BLOCK_BYTES


def a_payload_bytes(M, K, m, kk, cols, cpc):
    return cols * (K // kk) * cpc * a_leaf_bytes(m, kk)


def b_payload_bytes(K, N, n, kk):
    """9*K*N/8 -- independent of kk, as it must be (a permutation only)."""
    return (N // n) * (K // kk) * b_leaf_bytes(n, kk)


# ---------------------------------------------------------------------------
# reference (numpy) packers -- the specification
# ---------------------------------------------------------------------------
def _bf16_bits(x):
    """f32 -> the uint16 bit pattern of its bf16 rounding (RNE), as s2_ffn32."""
    u = np.ascontiguousarray(np.asarray(x, np.float32)).view(np.uint32)
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16).astype(np.uint16)


def pack_a_kk_ref(x, m, kk, cols=8, cpc=4, out=None):
    """numpy reference for pack_a_kk (see that function for the contract)."""
    xf = np.ascontiguousarray(np.asarray(x, np.float32))
    if xf.ndim != 2:
        raise ValueError("pack_a_kk: x must be 2-D (M, K), got %r" % (xf.shape,))
    M, K = xf.shape
    _check_a_geom(M, K, m, kk, cols, cpc)
    nkk = K // kk
    t = tile_map(M, m, cols, cpc)
    need = a_payload_bytes(M, K, m, kk, cols, cpc)
    dst = _prep_out(out, need, "pack_a_kk")

    u = _bf16_bits(xf)                                   # (M, K) uint16
    nm = M // m
    # [tile][k-step][y][kb][i][j] -- the 8x8 sub-tile order of the kernel
    v = np.ascontiguousarray(
        u.reshape(nm, m // 8, 8, nkk, kk // 8, 8).transpose(0, 3, 1, 4, 2, 5)
    ).reshape(nm, nkk, m * kk)
    idx = np.array([[0 if q < 0 else q for q in row] for row in t], np.int64)
    seg = np.ascontiguousarray(v[idx].transpose(0, 2, 1, 3))   # [c][s][p][..]
    dst[:] = seg.view(np.uint8).reshape(-1)
    return dst


def pack_b_kk_ref(blocks, K, N, n, kk, out=None):
    """numpy reference for pack_b_kk (see that function for the contract)."""
    b = np.ascontiguousarray(blocks, np.uint8)
    _check_b_geom(b.shape, K, N, n, kk)
    nkk = K // kk
    kbp = kk // 8
    nt = N // n
    need = b_payload_bytes(K, N, n, kk)
    dst = _prep_out(out, need, "pack_b_kk")
    v = (b.reshape(nt, n // 8, 8, nkk, kbp, 9)
         .transpose(0, 3, 1, 4, 2, 5))                    # [nt][s][j][kb][i][9]
    dst[:] = np.ascontiguousarray(v).reshape(-1)
    return dst


# ---------------------------------------------------------------------------
# validation / allocation helpers
# ---------------------------------------------------------------------------
def _check_kk(kk):
    kk = int(kk)
    if kk <= 0 or kk % BLOCK:
        raise ValueError(
            "pack_kk: kk must be a positive multiple of %d (dim_k; the "
            "micro-kernel's k-block), got %r" % (BLOCK, kk))
    return kk


def _check_a_geom(M, K, m, kk, cols, cpc):
    m, kk, cols, cpc = int(m), _check_kk(kk), int(cols), int(cpc)
    if m <= 0 or m % BLOCK:
        raise ValueError("pack_a_kk: m must be a positive multiple of 8, got %r" % (m,))
    if cols <= 0 or cpc <= 0:
        raise ValueError("pack_a_kk: cols and cpc must be positive, got %r %r"
                         % (cols, cpc))
    if M % m:
        raise ValueError("pack_a_kk: M %d is not a multiple of m %d" % (M, m))
    if K % kk:
        raise ValueError("pack_a_kk: K %d is not a multiple of kk %d" % (K, kk))
    if M // m > cols * cpc:
        raise ValueError(
            "pack_a_kk: %d M-tiles do not fit %d slots (cols*cpc)"
            % (M // m, cols * cpc))
    return m, kk, cols, cpc


def _check_b_geom(shape, K, N, n, kk):
    kk = _check_kk(kk)
    n = int(n)
    if tuple(shape) != (N, K // BLOCK, BLOCK_BYTES):
        raise ValueError(
            "pack_b_kk: blocks must be (N, K//8, 9) = (%d, %d, 9), got %r"
            % (N, K // BLOCK, tuple(shape)))
    if K % kk:
        raise ValueError("pack_b_kk: K %d is not a multiple of kk %d" % (K, kk))
    if n <= 0 or n % BLOCK:
        raise ValueError("pack_b_kk: n must be a positive multiple of 8, got %r" % (n,))
    if N % n:
        raise ValueError("pack_b_kk: N %d is not a multiple of n %d" % (N, n))
    return n, kk


def _prep_out(out, need, who):
    if out is None:
        return np.empty(need, np.uint8)
    a = out if isinstance(out, np.ndarray) else np.asarray(out)
    if a.dtype != np.uint8:
        raise ValueError("%s: out must be uint8, got %s" % (who, a.dtype))
    if not a.flags.c_contiguous:
        raise ValueError("%s: out must be C-contiguous" % who)
    if a.size != need:
        raise ValueError("%s: out has %d B, layout needs %d B"
                         % (who, a.size, need))
    return a.reshape(-1)


# ---------------------------------------------------------------------------
# public packers
# ---------------------------------------------------------------------------
def pack_a_kk(x, m, kk, cols=8, cpc=4, out=None):
    """Pack a GEMM A operand at kernel k-block ``kk``: (M,K) f32 -> uint8.

    Byte-identical to ``s2_ffn32.pack_a(x, m, kk, cols, cpc)`` at kk == 8.

    Layout: ``[col][k-step][core][leaf]`` where one leaf is the (m, kk) A tile
    of one core in the micro-kernel's 8x8 sub-tile order
    ``[m/8][kk/8][8 rows][8 bf16]`` (see the module docstring).

    Parameters
    ----------
    x : (M, K) float32-like
        Activations.  Values are rounded to bf16 (RNE), which is the element
        type mm_bfp(mixed=True) declares for A; the core converts bf16 ->
        bfp16ebs8 itself (mm_bfp_mixed.cc:67-72).
    m : int
        Kernel rows (dim_m), a multiple of 8 (the kernel needs a multiple of
        16: mm_bfp.cc:303 static_assert m % 16).
    kk : int
        Kernel tile columns (dim_k), a positive multiple of 8.  8/16/32/64 are
        the verified values.
    cols, cpc : int
        Design tile grid: cols*cpc M-tiles, tile c*cpc+p owned by core p of
        column c (s2_ffn32.tile_map).  Slots past M//m are dummies and carry
        tile 0's bytes (as the shipped packer does).
    out : uint8 array of M*K*2 bytes, optional
        Preallocated destination, returned as-is.

    Returns
    -------
    uint8 array, (M*K*2,)  == cols*(K//kk)*cpc*(m*kk*2)
    """
    xf = np.ascontiguousarray(np.asarray(x, np.float32))
    if xf.ndim != 2:
        raise ValueError("pack_a_kk: x must be 2-D (M, K), got %r" % (xf.shape,))
    M, K = xf.shape
    m, kk, cols, cpc = _check_a_geom(M, K, m, kk, cols, cpc)
    need = a_payload_bytes(M, K, m, kk, cols, cpc)
    dst = _prep_out(out, need, "pack_a_kk")
    nkk = K // kk
    nb = _numba_kernels()
    if nb is None or need == 0:
        return pack_a_kk_ref(xf, m, kk, cols, cpc, out=dst)
    if dst.ctypes.data % 2:
        # a uint16 view of an odd address would be misaligned: fall back
        return pack_a_kk_ref(xf, m, kk, cols, cpc, out=dst)
    tm = np.array(tile_map(M, m, cols, cpc), np.int64)
    nb[0](xf.view(np.uint32).reshape(-1), tm, dst.view(np.uint16),
          m, kk, cols, cpc, nkk, K)
    return dst


def pack_b_kk(blocks, K, N, n, kk, out=None):
    """Repack the model's BFP16 blocks at kernel k-block ``kk``.

    Byte-identical to ``s2_ffn32.pack_b_from_blocks(blocks, K, N, n)`` at
    kk == 8.  No re-quantization: the model's bytes are already BFP16, so only
    the block order changes -- the payload is a permutation of the input.

    Layout: ``[n-tile][k-step][leaf]``, one leaf = the (kk, n) B tile (B held
    transposed, (n, k)) in the micro-kernel's sub-tile order
    ``[n/8][kk/8][8 n-rows][9 B]``.

    Parameters
    ----------
    blocks : (N, K//8, 9) uint8
        The model's BFP16 blocks, blocks of 8 running along K (s2_ffn32
        .encode_bfp / the .npz weights are already in this order).
    K, N : int
        Reduction length and output width.
    n : int
        Kernel columns (dim_n), a multiple of 8 (the kernel needs a multiple
        of 16: mm_bfp.cc:305 static_assert n % 16), dividing N.
    kk : int
        Kernel tile rows of B = A's dim_k.
    out : uint8 array of 9*N*K//8 bytes, optional

    Returns
    -------
    uint8 array, (N*K*9//8,)  == (N//n)*(K//kk)*(n*kk*9//8)
    """
    b = np.ascontiguousarray(blocks, np.uint8)
    if b.ndim != 3:
        raise ValueError("pack_b_kk: blocks must be (N, K//8, 9), got %r"
                         % (b.shape,))
    n, kk = _check_b_geom(b.shape, K, N, n, kk)
    need = b_payload_bytes(K, N, n, kk)
    dst = _prep_out(out, need, "pack_b_kk")
    nkk = K // kk
    nb = _numba_kernels()
    if nb is None or need == 0:
        return pack_b_kk_ref(b, K, N, n, kk, out=dst)
    nb[1](b.reshape(-1), dst, N // n, nkk, n, kk, K // BLOCK)
    return dst


# ---------------------------------------------------------------------------
# inverses (used by the verification harness)
# ---------------------------------------------------------------------------
def unpack_a_kk(buf, M, K, m, kk, cols=8, cpc=4):
    """payload -> (M, K) float32 (the packed bf16 values).  Inverse of pack_a_kk."""
    m, kk, cols, cpc = _check_a_geom(M, K, m, kk, cols, cpc)
    nkk = K // kk
    nb_y, nb_x = m // 8, kk // 8
    seg = np.asarray(buf, np.uint8).reshape(cols, nkk, cpc, m * kk, 2)
    v = np.ascontiguousarray(seg.view(np.uint16)[..., 0])       # [c][s][p][..]
    leaf = v.reshape(cols, nkk, cpc, nb_y, nb_x, 8, 8)          # [c][s][p][y][kb][i][j]
    tm = tile_map(M, m, cols, cpc)
    out = np.zeros((M, K), np.uint16)
    w = out.reshape(M // m, nb_y, 8, nkk, nb_x, 8)   # [t][y][i][s][kb][j]
    for c in range(cols):
        for p in range(cpc):
            t = tm[c][p]
            if t < 0:
                continue
            # [s][y][kb][i][j] -> [y][i][s][kb][j]
            w[t] = leaf[c, :, p].transpose(1, 3, 0, 2, 4)
    return (out.astype(np.uint32) << 16).view(np.float32)


def unpack_b_kk(buf, K, N, n, kk):
    """payload -> (N, K//8, 9) blocks in model order.  Inverse of pack_b_kk."""
    n, kk = _check_b_geom((N, K // BLOCK, BLOCK_BYTES), K, N, n, kk)
    nt, nkk, kbp = N // n, K // kk, kk // 8
    raw = np.asarray(buf, np.uint8).reshape(nt, nkk, n // 8, kbp, 8, 9)
    # [nt][s][j][kb][i][b] -> [nt][j][i][s][kb][b] == (N, K//8, 9)
    blocks = raw.transpose(0, 2, 4, 1, 3, 5).reshape(N, K // BLOCK, 9)
    return np.ascontiguousarray(blocks)


# ---------------------------------------------------------------------------
# the FUSED gu's B object: the [gate | up] PAIR
# ---------------------------------------------------------------------------
# The fused image (ffn_design_xf.ffn_xf, the delivered swiglu_rt11 body) reads
# ONE B element per (n-tile, k-step) that holds BOTH slices of the same n
# columns, gate first, so the two mm_bfp calls of a k-step cannot be handed
# mismatched weights.  The pair is a pure RESHAPE of two pack_b_kk payloads:
# every 9-byte block keeps its bytes and its order inside its own leaf, so the
# pair is still a PERMUTATION of the model's blocks (never a re-encode), and
# the two halves are recovered exactly by unpair_b.
def b_pair_bytes(K, N, n, kk):
    """Bytes of the fused gu's B payload: 2 x 9*K*N/8."""
    return 2 * b_payload_bytes(K, N, n, kk)


def pair_b(b_gate, b_up, K, N, n, kk, out=None):
    """[gate payload | up payload] -> the fused image's [gate|up] pair payload.

    Layout: [n-tile][k-step][gate leaf | up leaf], leaf = (kk, n) BFP16 blocks
    in the micro-kernel's sub-tile order -- i.e. pack_b_kk's own leaf, twice.
    """
    n, kk = _check_b_geom((N, K // BLOCK, BLOCK_BYTES), K, N, n, kk)
    nt, nkk = N // n, K // kk
    leaf = b_leaf_bytes(n, kk)
    g = _prep_out(b_gate, b_payload_bytes(K, N, n, kk), "pair_b(gate)")
    u = _prep_out(b_up, b_payload_bytes(K, N, n, kk), "pair_b(up)")
    dst = _prep_out(out, 2 * b_payload_bytes(K, N, n, kk), "pair_b")
    v = dst.reshape(nt, nkk, 2, leaf)
    v[:, :, 0] = g.reshape(nt, nkk, leaf)
    v[:, :, 1] = u.reshape(nt, nkk, leaf)
    return dst


def unpair_b(pair, K, N, n, kk):
    """The fused B payload -> (gate payload, up payload).  Inverse of pair_b."""
    n, kk = _check_b_geom((N, K // BLOCK, BLOCK_BYTES), K, N, n, kk)
    nt, nkk = N // n, K // kk
    leaf = b_leaf_bytes(n, kk)
    p = _prep_out(pair, 2 * b_payload_bytes(K, N, n, kk), "unpair_b")
    v = p.reshape(nt, nkk, 2, leaf)
    g = np.ascontiguousarray(v[:, :, 0]).reshape(-1)
    u = np.ascontiguousarray(v[:, :, 1]).reshape(-1)
    return g, u
