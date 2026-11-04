# SPDX-License-Identifier: MIT
"""fast_glue.py -- S1: the product's host glue, ONE pass per byte, pure CPU.

WHY
---
512^2 / 8 steps = 121.50 s (job 20261031-193227-8a64e4), steady step 14.81 s.
Per layer the host glue is 95 ms (proj4) + 25.3 ms (FFN) + 19 ms (rope) and the
block-level numpy work (norms/residuals/absmax) is another ~100 ms.  Every one
of those bodies is a chain of full-size numpy temporaries:

    prod_devkill.proj4        np.zeros(pad) + pad[:]=x        48.7 MB
                              PB._bf16_bytes(pad)            ~97 MB (6 temps)
                              .tobytes()                      16.2 MB
    devkill_ffn.a_pack_fast   np.ascontiguousarray(...)         32.4 MB
                              .tobytes()                        16.2 MB
    devkill_ffn.decode_fast   transpose copy                    16.2 MB
                              np.left_shift                     24.3 MB
    t18_run.rope_rms_jit      heads() transpose                 32.4 MB
                              rms_into = 6 numpy passes        ~81 MB
                              H.rope_f32                        32.4 MB

Each function here is a PORT of the expression it replaces: same operands, same
order, same rounding, one read + one write, no temporary.  Equality is asserted
against the PRODUCT functions (t18_run.rms_into / t18_run.rope_rms_jit /
devkill_ffn.a_pack_fast / devkill_ffn.decode_fast / prod_loop_bigm._bf16_bytes /
np.left_shift), on the real product bytes -- see bench.py.

DEVICE: none.  No pyxrt, no xrt, no BO, no xclbin.  Inputs are plain arrays /
memoryviews, i.e. exactly what pyxrt's bo.read() returns and bo.write() accepts
(webui/chunked.py:_launch already hands bo.write an ndarray, so the .tobytes()
in the current path is a pure extra copy).

DROP-IN MAP (what to call instead of what)
------------------------------------------
  prod_devkill._run        raw -> y            proj_read_cv(raw, rows, 3840)
  prod_loop_bigm._bf16_bytes + a_pack_fast     pack_a(x_f32)            (fused)
  devkill_ffn.a_pack_fast(x_bf16)              pack_a(x_u16, from_bf16=True)
  devkill_ffn.decode_fast                      decode_y(raw, rt, rows)
  t18_run.rope_rms_jit(heads(q), ...)          rope_fused(u16_2d, w, cos, sin)
  heads(v) (widen only)                        widen_t(u16_2d, ...)
  rms_into + np.multiply + bf16                norm_bf16(u, w, scale)
  rms_into + np.multiply + np.add              norm_residual(a, u, w, scale)
  rms_into + np.multiply + _bf16_bytes+pack    pack_a_norm(x, w, scale)
  rms_into + mult + add + absmax               post_ffn(y, xm, w, scale)
"""
from __future__ import annotations

import os

import numpy as np

try:                                              # py310 (the bridge) has numba
    from numba import njit, prange, set_num_threads, get_num_threads
    HAVE_NUMBA = True
except Exception:                                 # pragma: no cover - py312
    HAVE_NUMBA = False

    def njit(*a, **k):                            # type: ignore
        def deco(f):
            return f
        if a and callable(a[0]):
            return a[0]
        return deco

    def prange(x):                                # type: ignore
        return range(x)

    def set_num_threads(n):                       # type: ignore
        return None

    def get_num_threads():                        # type: ignore
        return 1

# the same flags t8_codec uses (t8_codec.py:44).  fastmath OFF on purpose: the
# rope byte gate is only exact without FMA contraction.
_FAST = dict(cache=True, nogil=True, boundscheck=False, fastmath=False)
_FAST_P = dict(_FAST, parallel=True)

EPS = np.float32(1e-5)          # t3_real.EPS == 1e-05 (verified, see GLUE-AUDIT)

_TLS = {"n": 0}


def _as_u16(x):
    """any drain (ndarray | memoryview | bytes) -> a flat uint16 array.

    MEASURED (parent's device run, arm B): pyxrt's bo.read() hands back a
    **uint8 numpy array**, not a memoryview.  np.ascontiguousarray(x, uint16)
    would then *convert* 8,110,080 bytes into 8,110,080 uint16 elements (twice
    the elements the drain has) and every kernel reads garbage.  Reinterpret,
    never convert: a 1-byte buffer of even length views as uint16 in place.
    """
    if isinstance(x, np.ndarray):
        a = np.ascontiguousarray(x)
        if a.dtype == np.uint16:
            return a                      # keep the 2-D shape a caller handed in
        if a.size % 2:
            raise ValueError("_as_u16: odd byte count %d" % a.size)
        return a.reshape(-1).view(np.uint16)   # byte buffer -> uint16, in place
    if isinstance(x, memoryview):
        return np.frombuffer(x, np.uint16)
    return np.frombuffer(bytes(x), np.uint16)


def _threads(n):
    """0 = leave the numba default alone (1 for a non-parallel kernel)."""
    if n and HAVE_NUMBA:
        set_num_threads(int(n))
        _TLS["n"] = int(n)
    return _TLS["n"]


# ---------------------------------------------------------------------------
# 1. bf16 <-> f32 : the exact product bit maps
# ---------------------------------------------------------------------------
@njit(**_FAST)
def _widen_k(u, ob, n):
    for i in range(n):
        ob[i] = np.uint32(u[i]) << np.uint32(16)


@njit(**_FAST_P)
def _widen_kp(u, ob, n):
    for i in prange(n):
        ob[i] = np.uint32(u[i]) << np.uint32(16)


@njit(**_FAST)
def _widen_abs_k(u, out, ob, n, stats):
    """widen + max/min of the SAME float32 elements absmax_fast() would see.

    numpy's absmax_fast(x) is max(x.max(), -x.min()); both are selections of an
    existing float32 element, so tracking them here is byte-identical (no
    arithmetic).  out and ob alias (ob = out.view(uint32))."""
    ob[0] = np.uint32(u[0]) << np.uint32(16)
    hi = out[0]
    lo = out[0]
    for i in range(1, n):
        ob[i] = np.uint32(u[i]) << np.uint32(16)
        f = out[i]
        if f > hi:
            hi = f
        if f < lo:
            lo = f
    stats[0] = hi
    stats[1] = lo


# T22_GLUE_ABSMAXP: the absmax half of the projection drain, on prange.
#
# TRAP, measured the hard way (2026-11-02, imaggers/hostwin/_probe6-8.py): a
# PRANGE version of _widen_abs_k that still widens through an aliased
# out.view(uint32) and then reads the float back from out[i] produces WRONG
# hi/lo: the parallel kernel is vectorised, and the read of out[i] is hoisted
# above the ob[i] store (numba assumes the two arguments do not alias).  The
# widened BYTES stayed exact, which is what makes it a silent trap.  The serial
# _widen_abs_k is safe because it is not vectorised that way.
#
# So the parallel path is TWO passes, each of which is a shape numba handles
# correctly and each of which is already gated elsewhere:
#   1. widen_u16(..., threads=n)  -> the existing _widen_kp (writes only)
#   2. _minmax_kp                 -> reads the widened f32 array, writes only
#                                    ceil(n/bs) block partials
# max/min are SELECTIONS of existing float32 elements, so the value does not
# depend on the order they are visited in; block 0 still seeds hi/lo from
# out[0] exactly like the serial loop, so even a NaN / +-inf / -inf seed behaves
# identically (verified on n = 1 .. 4.05 M and at 1/2/4/8/20 threads).
_ABS_BS = 65536


@njit(**_FAST_P)
def _minmax_kp(out, n, phi, plo, bs, stats):
    """max/min of out[:n] from ceil(n/bs) block partials, then one reduce."""
    nb = (n + bs - 1) // bs
    for b in prange(nb):
        i0 = b * bs
        i1 = n if n < i0 + bs else i0 + bs
        if b == 0:
            hi = out[0]
            lo = out[0]
            start = 1
        else:
            hi = np.float32(-np.inf)
            lo = np.float32(np.inf)
            start = i0
        for i in range(start, i1):
            f = out[i]
            if f > hi:
                hi = f
            if f < lo:
                lo = f
        phi[b] = hi
        plo[b] = lo
    hi = phi[0]
    lo = plo[0]
    for b in range(1, nb):
        if phi[b] > hi:
            hi = phi[b]
        if plo[b] < lo:
            lo = plo[b]
    stats[0] = hi
    stats[1] = lo


def widen_u16(u16, out=None, threads=0):
    """bf16 bit patterns -> float32, one pass (np.left_shift(u16,16,out=y.view(u32)))."""
    u = np.ascontiguousarray(u16, np.uint16).reshape(-1)
    if out is None:
        out = np.empty(u.size, np.float32)
    ob = out.reshape(-1).view(np.uint32)
    assert ob.size == u.size
    _threads(threads)
    (_widen_kp if threads > 1 else _widen_k)(u, ob, u.size)
    return out


@njit(**_FAST)
def _rne_k(xb, out, n):
    for i in range(n):
        u = xb[i]
        r = ((u >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
        out[i] = np.uint16((u + r) >> np.uint32(16))


@njit(**_FAST_P)
def _rne_kp(xb, out, n):
    for i in prange(n):
        u = xb[i]
        r = ((u >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
        out[i] = np.uint16((u + r) >> np.uint32(16))


def bf16_rne(x, out=None, threads=0):
    """float32 -> bf16 RNE bits (prod_loop_bigm._bf16_bytes), one pass."""
    xa = np.ascontiguousarray(x, np.float32)
    n = xa.size
    if out is None:
        out = np.empty(n, np.uint16)
    _threads(threads)
    (_rne_kp if threads > 1 else _rne_k)(xa.reshape(-1).view(np.uint32),
                                         out.reshape(-1), n)
    return out


# ---------------------------------------------------------------------------
# 2. projection C drain  (prod_devkill._run: bo.read + np.left_shift)
# ---------------------------------------------------------------------------
def proj_read_cv(raw, rows, cols=3840, out=None, threads=0, copy=False,
                 want_absmax=False):
    """raw (memoryview|bytes|u16 array) -> (rows, cols) float32, one pass.

    copy=True reproduces the product's extra bytes(bo.read(...)) copy (8.1 MB)
    -- it exists only so the audit can price that copy separately.
    want_absmax=True also returns max(max, -min) (t18_run.absmax_fast) for the
    tensor it just wrote; the value is computed from the same float32 elements.
    """
    if copy:
        raw = bytes(raw)
    u = _as_u16(raw).reshape(-1)                   # uint8 ndarray -> uint16 view
    n = rows * cols
    assert u.size >= n, (u.size, n)
    if out is None:
        out = np.empty((rows, cols), np.float32)
    o = out.reshape(-1)
    if not want_absmax:
        widen_u16(u[:n], o, threads=threads)
        return out
    st = np.zeros(2, np.float32)
    # T22_GLUE_ABSMAXP=1: the ONE kernel in the chain that was still serial.
    # 4.05 M u16 -> f32 plus an absmax is 24.3 MB of traffic (~1 ms at this
    # box's 24 GB/s); measured 5.74 ms serial => 4.2 GB/s.  Both passes are
    # the same bytes and the same two f32 values (see _minmax_kp and the trap
    # note above it).  Default 0 keeps the serial kernel and _threads(0).
    if os.environ.get("T22_GLUE_ABSMAXP", "0") == "1":
        widen_u16(u[:n], o, threads=threads)      # sets _threads(threads)
        nb = (n + _ABS_BS - 1) // _ABS_BS
        _minmax_kp(o, n, np.empty(nb, np.float32), np.empty(nb, np.float32),
                   _ABS_BS, st)
    else:
        _threads(0)
        _widen_abs_k(u[:n], o, o.view(np.uint32), n, st)
    hi, lo = float(st[0]), float(st[1])
    return out, (hi if hi >= -lo else -lo)


# ---------------------------------------------------------------------------
# 3. FFN A image  (devkill_ffn.a_pack_fast(_bf16_bytes(x)), fused)
# ---------------------------------------------------------------------------
@njit(**_FAST)
def _pack_a_k(xb, out, nrows, cols):
    nk = cols // 64
    for mb in range(nrows // 96):
        for kb in range(nk):
            base = ((mb * nk + kb) * 12) * 512
            for ri in range(12):
                for cb in range(8):
                    for ri2 in range(8):
                        r = mb * 96 + ri * 8 + ri2
                        o = base + ((ri * 8 + cb) * 8 + ri2) * 8
                        c0 = kb * 64 + cb * 8
                        for ci in range(8):
                            u = xb[r * cols + c0 + ci]
                            rr = ((u >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
                            out[o + ci] = np.uint16((u + rr) >> np.uint32(16))


@njit(**_FAST_P)
def _pack_a_kp(xb, out, nrows, cols):
    nk = cols // 64
    for mb in prange(nrows // 96):
        for kb in range(nk):
            base = ((mb * nk + kb) * 12) * 512
            for ri in range(12):
                for cb in range(8):
                    for ri2 in range(8):
                        r = mb * 96 + ri * 8 + ri2
                        o = base + ((ri * 8 + cb) * 8 + ri2) * 8
                        c0 = kb * 64 + cb * 8
                        for ci in range(8):
                            u = xb[r * cols + c0 + ci]
                            rr = ((u >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
                            out[o + ci] = np.uint16((u + rr) >> np.uint32(16))


@njit(**_FAST)
def _pack_a_u16_k(x, out, nrows, cols):
    nk = cols // 64
    for mb in range(nrows // 96):
        for kb in range(nk):
            base = ((mb * nk + kb) * 12) * 512
            for ri in range(12):
                for cb in range(8):
                    for ri2 in range(8):
                        r = mb * 96 + ri * 8 + ri2
                        o = base + ((ri * 8 + cb) * 8 + ri2) * 8
                        c0 = kb * 64 + cb * 8
                        for ci in range(8):
                            out[o + ci] = x[r * cols + c0 + ci]


def pack_a(x, out=None, from_bf16=False, threads=0):
    """The FFN A image in one pass (RNE and/or block transpose together).

    f32 in  == devkill_ffn.a_pack_fast(prod_loop_bigm._bf16_bytes(x))
    u16 in  == devkill_ffn.a_pack_fast(x)
    Returns a flat uint16 array: hand it to bo.write directly (no .tobytes()).
    """
    if from_bf16:
        xa = _as_u16(x)
        xb = None
    else:
        xa = np.ascontiguousarray(x, np.float32)
        xb = xa.reshape(-1).view(np.uint32)
    nrows, cols = xa.shape
    assert nrows % 96 == 0 and cols % 64 == 0, xa.shape
    if out is None:
        out = np.empty(nrows * cols, np.uint16)
    o = out.reshape(-1)
    _threads(threads)
    if from_bf16:
        _pack_a_u16_k(xa.reshape(-1), o, nrows, cols)
    else:
        (_pack_a_kp if threads > 1 else _pack_a_k)(xb, o, nrows, cols)
    return o


@njit(**_FAST)
def _pack_a_norm_k(x, out, w, sc, hd, eps, nrows, scr):
    """rms+scale of xm, then bf16 RNE straight into the A image (one pass).

    out[mb,kb,ri,cb,ri2,ci] = bf16( ((xm/v)*w)*sc )[row mb*96+ri*8+ri2, col ...]
    Reference: pack_a(_bf16_bytes(rms_into(xm,w,tmp)*sc)).
    """
    nk = (nrows * hd) // (96 * hd)      # unused placeholder, keeps shape explicit
    box = np.empty(1, np.uint32)
    boxf = box.view(np.float32)
    for r in range(nrows):
        base = r * hd
        s = _pwsq(x[base:base + hd], hd, scr)
        vv = np.sqrt(np.float32(s / np.float32(hd)) + eps)
        mb = r // 96
        rr = r % 96
        ri = rr // 8
        ri2 = rr % 8
        for kb in range(hd // 64):
            for cb in range(8):
                o = ((((mb * (hd // 64) + kb) * 12 + ri) * 8 + cb) * 8 + ri2) * 8
                c0 = kb * 64 + cb * 8
                for ci in range(8):
                    t = x[base + c0 + ci] / vv
                    t = t * w[c0 + ci]
                    if sc is not None:
                        t = t * sc[c0 + ci]
                    boxf[0] = t
                    u = box[0]
                    q = ((u >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
                    out[o + ci] = np.uint16((u + q) >> np.uint32(16))


def pack_a_norm(x, w, scale=None, out=None, threads=0):
    """rms_into + multiply + _bf16_bytes + a_pack_fast, all in ONE pass."""
    xa = np.ascontiguousarray(x, np.float32)
    nrows, hd = xa.shape
    assert nrows % 96 == 0 and hd % 64 == 0
    if out is None:
        out = np.empty(nrows * hd, np.uint16)
    wa = np.ascontiguousarray(w, np.float32).reshape(-1)
    sa = None if scale is None else np.ascontiguousarray(scale, np.float32).reshape(-1)
    _pack_a_norm_k(xa.reshape(-1), out.reshape(-1), wa, sa, hd, EPS, nrows,
                   np.empty(64, np.float32))
    return out


# ---------------------------------------------------------------------------
# 4. FFN Y drain  (devkill_ffn.decode_fast, one pass)
# ---------------------------------------------------------------------------
@njit(**_FAST)
def _decode_y_k(u, ob, rt, cols):
    nobj = cols // 128
    for t in range(rt):
        for rb in range(12):
            for ri in range(8):
                orow = ((t * 12 + rb) * 8 + ri) * cols
                for obj in range(nobj):
                    for cb in range(16):
                        i0 = ((((t * nobj + obj) * 12 + rb) * 16 + cb) * 8 + ri) * 8
                        o0 = orow + obj * 128 + cb * 8
                        for ci in range(8):
                            ob[o0 + ci] = np.uint32(u[i0 + ci]) << np.uint32(16)


@njit(**_FAST_P)
def _decode_y_kp(u, ob, rt, cols):
    nobj = cols // 128
    for t in prange(rt):
        for rb in range(12):
            for ri in range(8):
                orow = ((t * 12 + rb) * 8 + ri) * cols
                for obj in range(nobj):
                    for cb in range(16):
                        i0 = ((((t * nobj + obj) * 12 + rb) * 16 + cb) * 8 + ri) * 8
                        o0 = orow + obj * 128 + cb * 8
                        for ci in range(8):
                            ob[o0 + ci] = np.uint32(u[i0 + ci]) << np.uint32(16)


def decode_y(raw, rt, rows, cols=3840, out=None, threads=0):
    """blocked bf16 drain -> row-major f32 (devkill_ffn.decode_fast)."""
    u = _as_u16(raw).reshape(-1)
    if out is None:
        out = np.empty((rows, cols), np.float32)
    ob = out.reshape(-1).view(np.uint32)
    _threads(threads)
    (_decode_y_kp if threads > 1 else _decode_y_k)(u, ob, rt, cols)
    return out[:rows]


# ---------------------------------------------------------------------------
# 5. transpose+widen for V  (t18_run.heads() + widen)
# ---------------------------------------------------------------------------
@njit(**_FAST)
def _widen_t_k(u, ob, seq, heads, hd, out_seq, row0):
    d = heads * hd
    for h in range(heads):
        hbase = h * hd
        obase = (h * out_seq + row0) * hd
        for m in range(seq):
            s0 = m * d + hbase
            o0 = obase + m * hd
            for p in range(hd):
                ob[o0 + p] = np.uint32(u[s0 + p]) << np.uint32(16)


@njit(**_FAST_P)
def _widen_t_kp(u, ob, seq, heads, hd, out_seq, row0):
    d = heads * hd
    for h in prange(heads):
        hbase = h * hd
        obase = (h * out_seq + row0) * hd
        for m in range(seq):
            s0 = m * d + hbase
            o0 = obase + m * hd
            for p in range(hd):
                ob[o0 + p] = np.uint32(u[s0 + p]) << np.uint32(16)


def widen_t(u16_2d, seq, heads, hd, out=None, threads=0, row0=0, out_seq=None):
    """(seq, heads*hd) bf16 -> (heads, seq, hd) f32 == heads() + widen.

    row0/out_seq (T22 proj-block) write ONE block into its rows of a larger
    (heads, out_seq, hd) buffer -- same per-row indices as row0=0/out_seq=seq.
    """
    u = _as_u16(u16_2d).reshape(-1)
    if out is None:
        out = np.empty((heads, seq, hd), np.float32)
        out_seq = seq
    elif out_seq is None:
        out_seq = out.shape[1]
    if u.size != seq * heads * hd:
        raise ValueError("widen_t: input %d elements, expected %d"
                         % (u.size, seq * heads * hd))
    if out.size != heads * int(out_seq) * hd:
        raise ValueError("widen_t: out %d elements, expected %d for (%d, %d, %d)"
                         % (out.size, heads * int(out_seq) * hd, heads,
                            int(out_seq), hd))
    ob = out.reshape(-1).view(np.uint32)
    _threads(threads)
    (_widen_t_kp if threads > 1 else _widen_t_k)(u, ob, seq, heads, hd,
                                                 int(out_seq), int(row0))
    return out


# ---------------------------------------------------------------------------
# 6. rms + scale (+ optional bf16 emit) -- numpy pairwise sum reproduced
# ---------------------------------------------------------------------------
@njit(inline="always")
def _pw8(v, n):
    """numpy's PW_BLOCKSIZE branch: 8 accumulators, stride 8, + tail.

    loops.c.src:  if (n < 8) plain loop; else if (n <= 128) { 8 accs over
    n - n%8, combine ((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7)), then the tail }.
    """
    if n < 8:
        s = np.float32(0.0)
        for i in range(n):
            t = v[i]
            s = s + t * t
        return s
    r0 = v[0] * v[0]; r1 = v[1] * v[1]; r2 = v[2] * v[2]; r3 = v[3] * v[3]
    r4 = v[4] * v[4]; r5 = v[5] * v[5]; r6 = v[6] * v[6]; r7 = v[7] * v[7]
    i = 8
    lim = n - (n % 8)
    while i < lim:
        r0 += v[i] * v[i]; r1 += v[i + 1] * v[i + 1]
        r2 += v[i + 2] * v[i + 2]; r3 += v[i + 3] * v[i + 3]
        r4 += v[i + 4] * v[i + 4]; r5 += v[i + 5] * v[i + 5]
        r6 += v[i + 6] * v[i + 6]; r7 += v[i + 7] * v[i + 7]
        i += 8
    s = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        s = s + v[i] * v[i]
        i += 1
    return s


@njit(inline="always")
def _leafsize(n):
    """numpy splits while n > 128: n2 = n/2, n2 -= n2 % 8, left = n2.

    For every width the product reduces over (128, 3840) the recursion is
    perfectly balanced, so this returns the common leaf size.
    """
    m = n
    while m > 128:
        h = m // 2
        m = h - (h % 8)
    return m


@njit(inline="always")
def _pwsq(v, n, scr):
    """numpy pairwise_sum of v[i]*v[i] (== np.multiply(x,x).mean(-1) numerator).

    MEASURED (py310, numpy 1.26.4): for n = 3840 the 8-accumulator scheme alone
    is wrong on 29/29 real rows (rel ~5e-7); the balanced recursion is exact on
    29/29.  Bottom-up over the leaves is the same balanced binary tree in the
    same operand order, so it reproduces numpy bit for bit.
    """
    if n <= 128:
        return _pw8(v, n)
    leaf = _leafsize(n)
    cnt = n // leaf
    for i in range(cnt):
        scr[i] = _pw8(v[i * leaf:(i + 1) * leaf], leaf)
    m = cnt
    while m > 1:
        for i in range(m // 2):
            scr[i] = scr[2 * i] + scr[2 * i + 1]
        m = m // 2
    return scr[0]


@njit(**_FAST)
def _norm_out_k(x, out, w, sc, hd, eps, nrows, add_u, scr):
    """out = ((x/v)*w)*sc [+u];  v = sqrt(mean(x^2)+eps), numpy-exact."""
    for r in range(nrows):
        base = r * hd
        s = _pwsq(x[base:base + hd], hd, scr)
        vv = np.sqrt(np.float32(s / np.float32(hd)) + eps)
        for p in range(hd):
            t = x[base + p] / vv
            t = t * w[p]
            if sc is not None:
                t = t * sc[p]
            if add_u is not None:
                t = add_u[base + p] + t
            out[base + p] = t


@njit(**_FAST_P)
def _norm_out_kp(x, out, w, sc, hd, eps, nrows, add_u):
    for r in prange(nrows):
        c = np.empty(64, np.float32)        # per-thread reduction scratch
        base = r * hd
        s = _pwsq(x[base:base + hd], hd, c)
        vv = np.sqrt(np.float32(s / np.float32(hd)) + eps)
        for p in range(hd):
            t = x[base + p] / vv
            t = t * w[p]
            if sc is not None:
                t = t * sc[p]
            if add_u is not None:
                t = add_u[base + p] + t
            out[base + p] = t


@njit(**_FAST)
def _norm_b16_k(x, b16, w, sc, hd, eps, nrows, scr):
    """rms+scale -> bf16 RNE bits, ONE pass, no f32 image materialised."""
    box = np.empty(1, np.uint32)
    boxf = box.view(np.float32)
    for r in range(nrows):
        base = r * hd
        s = _pwsq(x[base:base + hd], hd, scr)
        vv = np.sqrt(np.float32(s / np.float32(hd)) + eps)
        for p in range(hd):
            t = x[base + p] / vv
            t = t * w[p]
            if sc is not None:
                t = t * sc[p]
            boxf[0] = t                       # the f32 the product would store
            u = box[0]                        # ... and its bits
            rr = ((u >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
            b16[base + p] = np.uint16((u + rr) >> np.uint32(16))


@njit(**_FAST_P)
def _norm_b16_kp(x, b16, w, sc, hd, eps, nrows):
    for r in prange(nrows):
        scr = np.empty(64, np.float32)
        box = np.empty(1, np.uint32)
        boxf = box.view(np.float32)
        base = r * hd
        s = _pwsq(x[base:base + hd], hd, scr)
        vv = np.sqrt(np.float32(s / np.float32(hd)) + eps)
        for p in range(hd):
            t = x[base + p] / vv
            t = t * w[p]
            if sc is not None:
                t = t * sc[p]
            boxf[0] = t
            u = box[0]
            rr = ((u >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
            b16[base + p] = np.uint16((u + rr) >> np.uint32(16))


def _norm_core(x, w, scale, out, threads, add=None):
    xa = np.ascontiguousarray(x, np.float32)
    hd = xa.shape[-1]
    nrows = xa.size // hd
    if add is not None and np.size(add) != xa.size:
        raise ValueError("norm_residual: residual has %d elements, x has %d"
                         % (np.size(add), xa.size))
    if out is None:
        out = np.empty(xa.shape, np.float32)
    of = out.reshape(-1)
    wa = np.ascontiguousarray(w, np.float32).reshape(-1)
    sa = None if scale is None else np.ascontiguousarray(scale, np.float32).reshape(-1)
    au = None if add is None else np.ascontiguousarray(add, np.float32).reshape(-1)
    _threads(threads)
    if threads > 1:
        _norm_out_kp(xa.reshape(-1), of, wa, sa, hd, EPS, nrows, au)
    else:
        _norm_out_k(xa.reshape(-1), of, wa, sa, hd, EPS, nrows, au,
                    np.empty(64, np.float32))
    return out


def norm_bf16(x, w, scale=None, out_bf16=None, threads=0):
    """rms+scale+bf16 in ONE pass: the projection's X image, ready to bo.write.

    Reference: np.multiply(t18_run.rms_into(x, w, tmp), scale, out=tmp) then
    prod_loop_bigm._bf16_bytes(tmp).  Emits ONLY the u16 image (8.1 MB); the f32
    image is not a product consumer of attention_norm1's output (x1 feeds only
    L.attention -> prod_devkill.proj4, t18_run.py:1875-1889).
    """
    xa = np.ascontiguousarray(x, np.float32)
    hd = xa.shape[-1]
    nrows = xa.size // hd
    if out_bf16 is None:
        out_bf16 = np.empty(xa.shape, np.uint16)      # 2-D, same shape as x
    ob = np.ascontiguousarray(out_bf16).reshape(-1)
    wa = np.ascontiguousarray(w, np.float32).reshape(-1)
    sa = None if scale is None else np.ascontiguousarray(scale, np.float32).reshape(-1)
    _threads(threads)
    if threads > 1:
        _norm_b16_kp(xa.reshape(-1), ob, wa, sa, hd, EPS, nrows)
    else:
        _norm_b16_k(xa.reshape(-1), ob, wa, sa, hd, EPS, nrows,
                    np.empty(64, np.float32))
    return np.asarray(out_bf16).reshape(xa.shape)


def norm_f32(x, w, scale=None, out=None, threads=0):
    """rms_into + np.multiply(...,out=) fused, f32 out."""
    return _norm_core(x, w, scale, out, threads, add=None)


def norm_residual(a, u, w, scale=None, out=None, threads=0):
    """out = u + ((a/v)*w)*scale  == rms_into + multiply + np.add(u, a2)."""
    return _norm_core(a, w, scale, out, threads, add=u)


# ---------------------------------------------------------------------------
# 7. post-FFN: rms + scale, residual add, absmax of the three tensors
# ---------------------------------------------------------------------------
@njit(**_FAST)
def _post_k(y, xm, out, w, sc, hd, eps, nrows, stats, scr):
    """out = xm + ((y/v)*w)*sc  and the absmax of xm and of y, one pass."""
    for r in range(nrows):
        base = r * hd
        s = _pwsq(y[base:base + hd], hd, scr)
        vv = np.sqrt(np.float32(s / np.float32(hd)) + eps)
        for p in range(hd):
            t = y[base + p] / vv
            t = t * w[p]
            if sc is not None:
                t = t * sc[p]
            xv = xm[base + p]
            out[base + p] = xv + t
            a = xv if xv >= 0 else -xv
            if a > stats[0]:
                stats[0] = a
            a = y[base + p]
            a = a if a >= 0 else -a
            if a > stats[1]:
                stats[1] = a


def post_ffn(y, xm, w, scale=None, out=None, threads=0):
    """rms_into + multiply + np.add(xm, p, out) + absmax(xm) + absmax(y).

    Returns (out, xm_absmax, y_absmax).  The product's third stat, a_absmax,
    is on the ATTENTION output -- fuse it into the o->bf16 pass instead
    (bf16_of(..., want_absmax=True)); it is not a function of y or xm.
    """
    ya = np.ascontiguousarray(y, np.float32)
    xa = np.ascontiguousarray(xm, np.float32)
    if xa.size != ya.size:
        raise ValueError("post_ffn: xm has %d elements, y has %d"
                         % (xa.size, ya.size))
    hd = ya.shape[-1]
    nrows = ya.size // hd
    if out is None:
        out = np.empty(ya.shape, np.float32)
    of = out.reshape(-1)
    wa = np.ascontiguousarray(w, np.float32).reshape(-1)
    sa = None if scale is None else np.ascontiguousarray(scale, np.float32).reshape(-1)
    st = np.zeros(2, np.float32)
    _post_k(ya.reshape(-1), xa.reshape(-1), of, wa, sa, hd, EPS, nrows, st,
            np.empty(64, np.float32))
    return out, (float(st[0]), float(st[1]))


# ---------------------------------------------------------------------------
# 7b. o -> bf16 (+ absmax(a)) : the attention output's only two consumers
# ---------------------------------------------------------------------------
@njit(**_FAST)
def _bf16_abs_k(x, b16, n, stats):
    box = np.empty(1, np.uint32)
    boxf = box.view(np.float32)
    hi = np.float32(0.0)
    lo = np.float32(0.0)
    for i in range(n):
        v = x[i]
        if v > hi:
            hi = v
        if v < lo:
            lo = v
        boxf[0] = v
        u = box[0]
        rr = ((u >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
        b16[i] = np.uint16((u + rr) >> np.uint32(16))
    stats[0] = hi
    stats[1] = lo


def bf16_of(x, out=None, want_absmax=False, threads=0):
    """bf16 RNE of a f32 tensor; optionally also max(x) / min(x) (absmax_fast)."""
    xa = np.ascontiguousarray(x, np.float32).reshape(-1)
    if out is None:
        out = np.empty(xa.size, np.uint16)
    if not want_absmax:
        _threads(threads)
        (_rne_kp if threads > 1 else _rne_k)(xa.view(np.uint32), out.reshape(-1),
                                             xa.size)
        return out
    st = np.zeros(2, np.float32)
    _bf16_abs_k(xa, out.reshape(-1), xa.size, st)
    hi, lo = float(st[0]), float(st[1])
    return out, (hi if hi >= -lo else -lo)


# ---------------------------------------------------------------------------
# 8. RoPE: bf16 C drain (seq, heads*hd) -> (heads, seq, hd) f32, rms+rope fused
# ---------------------------------------------------------------------------
@njit(**_FAST)
def _rope_k(u, out, w, cos, sin, seq, heads, hd, eps, out_seq, row0):
    d = heads * hd
    buf = np.empty(hd, np.uint32)
    bf = buf.view(np.float32)                      # ONE view per call
    scr = np.empty(64, np.float32)
    half = hd // 2
    for h in range(heads):
        for m in range(seq):
            s0 = m * d + h * hd
            for p in range(hd):
                buf[p] = np.uint32(u[s0 + p]) << np.uint32(16)
            s = _pwsq(bf, hd, scr)
            vv = np.sqrt(np.float32(s / np.float32(hd)) + eps)
            o0 = (h * out_seq + row0 + m) * hd
            c0 = (row0 + m) * half
            for j in range(half):
                xe = (bf[2 * j] / vv) * w[2 * j]
                xo = (bf[2 * j + 1] / vv) * w[2 * j + 1]
                c = cos[c0 + j]
                sn = sin[c0 + j]
                out[o0 + 2 * j] = xe * c - xo * sn
                out[o0 + 2 * j + 1] = xe * sn + xo * c


@njit(**_FAST_P)
def _rope_kp(u, out, w, cos, sin, seq, heads, hd, eps, out_seq, row0):
    d = heads * hd
    half = hd // 2
    for h in prange(heads):
        buf = np.empty(hd, np.uint32)
        bf = buf.view(np.float32)
        scr = np.empty(64, np.float32)
        for m in range(seq):
            s0 = m * d + h * hd
            for p in range(hd):
                buf[p] = np.uint32(u[s0 + p]) << np.uint32(16)
            s = _pwsq(bf, hd, scr)
            vv = np.sqrt(np.float32(s / np.float32(hd)) + eps)
            o0 = (h * out_seq + row0 + m) * hd
            c0 = (row0 + m) * half
            for j in range(half):
                xe = (bf[2 * j] / vv) * w[2 * j]
                xo = (bf[2 * j + 1] / vv) * w[2 * j + 1]
                c = cos[c0 + j]
                sn = sin[c0 + j]
                out[o0 + 2 * j] = xe * c - xo * sn
                out[o0 + 2 * j + 1] = xe * sn + xo * c


def rope_fused(u16_2d, w, cos, sin, seq=None, heads=None, hd=None, out=None,
               threads=0, row0=0, out_seq=None, rows=None):
    """t18_run.rope_rms_jit(heads(x), w, cos, sin, ...) with no intermediate.

    u16_2d : (seq, heads*hd) bf16 C drain (plain15 row-major output)
    cos/sin: (seq, hd//2) float32 tables, exactly as block_step passes them
    Returns (heads, seq, hd) float32 -- the array _attention() wants.

    T22 proj-block (2026-11-04): row0/out_seq/rows let ONE 96-row projection
    block be written straight into its rows of a (heads, out_seq, hd) output.
    The kernel indexes out[(h*out_seq + row0 + m)*hd + p] and cos[(row0+m)*half]
    -- the SAME per-row expressions as before with row0=0/out_seq=seq, so the
    default call is byte-for-byte the old one, and a block is the identical
    arithmetic on a different row range (the rms reduction is over HD, inside
    one row; the rope pairs are inside one row; no row reads another row).
    """
    u = _as_u16(u16_2d)
    n_in = np.size(u)
    if u.ndim == 1:
        u = u.reshape(-1, int(np.size(u) // max(1, (seq or 1))))
    cols = u.shape[-1]
    if out is not None:
        heads, oseq, hd = out.shape
        out_seq = oseq if out_seq is None else int(out_seq)
        if rows is None:
            # ROWS IN u, not u.shape[0]: a 1-D drain (bo.read() -> _as_u16) is
            # reshaped to (1, N) by the line above -- shape[0] is 1, and the
            # historical code never read it because it took the row count from
            # out.shape[1] instead.  Size/(heads*hd) is that same count for
            # every real call (2-D (rows, heads*hd) or 1-D bytes), and min()
            # keeps the old crop for a partial last tile (nt*96 > seq).
            _cols = int(heads) * int(hd)
            k = u.shape[0] if (u.ndim == 2 and u.shape[-1] == _cols) \
                else n_in // _cols
            rows = min(k, out_seq - row0)
    if rows is None:
        rows = u.shape[0] if seq is None else seq
    if hd is None:
        hd = 2 * int(np.shape(cos)[-1])
    if heads is None:
        heads = cols // int(hd)
    if out is None:
        out = np.empty((heads, rows, hd), np.float32)
        out_seq = rows
    of = out.reshape(-1)
    wa = np.ascontiguousarray(w, np.float32).reshape(-1)
    ca = np.ascontiguousarray(cos, np.float32).reshape(-1)
    sa = np.ascontiguousarray(sin, np.float32).reshape(-1)
    _threads(threads)
    (_rope_kp if threads > 1 else _rope_k)(u.reshape(-1), of, wa, ca, sa,
                                           int(rows), heads, hd, EPS,
                                           int(out_seq), int(row0))
    return out


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def _bench():
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import bench
    return bench.main(["--quick"])


if __name__ == "__main__":
    import sys
    if "--bench" in sys.argv:
        _bench()
    else:
        print(__doc__)
        print("numba:", HAVE_NUMBA, "threads:", get_num_threads())
