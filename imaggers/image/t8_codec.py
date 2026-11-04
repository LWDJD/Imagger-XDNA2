# SPDX-License-Identifier: MIT
"""t8_codec.py -- T8 U-1: the same BYTES, out of the numpy interpreter.

WHY: T7 §4.2 measures the host glue at 46.4 ms/layer (11.1 s / 8 steps), more
than the attention DEVICE time (5.76 s).  Every one of those codecs moves
1-4 MB per call, so their numpy plans run at 300-1000 MB/s -- two orders below
this machine's memory bandwidth.  The reason is not algorithm, it is that each
call is a chain of 5-15 full-size numpy temporaries executed by the numpy
interpreter, and inside the loop each temporary is a brand-new page.

WHAT CHANGES: the same integer/permutation maps, written as numba nJIT kernels
that read the source array once and write the destination once, with no
temporary array at all.  NOTHING about the arithmetic changes: every kernel
below is a port of a t3d_codec / t7_codec function that is itself byte-gated
against the library object it replaces.  The gate (t8_codec_gate.py) is the
SAME 40 adversarial cases as T6/T7 (power-of-two amax, +128 overflow, far
branch, zeros, denormals, ties) and it is byte equality against t3d_codec, not
against T6/T7.

    py310 imaggers/image/t8_codec_gate.py     # py310 = the device-stage python,
                                              # i.e. the interpreter the glue runs on

Kernels (all "in -> out", out supplied by the caller so a loop can reuse one
workspace instead of faulting in new pages every layer):

    pack_b_nk(x, out)                 bfp16ebs8 (calls,n,k) f32 -> u8
    unpack_blocked(u16, inv, out)     blocked bf16 -> f32 (rows, cols)
    pack_blocked(x, perm, out_u16)    f32 -> blocked bf16 (bit patterns)
    x_image64(x, perm64, out_u16)     60 x (96,64) group packs in ONE pass
    bf16_rne_u16(x, out_u16)          f32 -> bf16 RNE bit patterns
    bf16_to_f32_u16(u16, out)         bf16 bit patterns -> f32
    rope_f32(y, cos, sin, out)        RoPE on the (HEADS, M, HD) drain layout
"""
from __future__ import annotations

import numpy as np
import numba
from numba import njit

GROUP = 8
GB = 9
MANT_SHIFT = 23 - 7 + 1              # 17, same constant as t3d_codec/t7_codec

_FAST = dict(cache=True, nogil=True, boundscheck=False, fastmath=False)


# ---------------------------------------------------------------------------
# bf16 RNE: the t3d_codec bit map, scalar for scalar
# ---------------------------------------------------------------------------
@njit(**_FAST)
def _bf16_one(u):
    """uint32 (float32 bits) -> uint16 bf16 bits, RNE (== bfloat16())."""
    r = ((u >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
    return np.uint16((u + r) >> np.uint32(16))


@njit(**_FAST)
def bf16_rne_u16(x, out):
    """(n,) float32 -> (n,) uint16 bf16 bits."""
    xb = x.view(np.uint32)
    for i in range(x.size):
        out[i] = _bf16_one(xb[i])


@njit(**_FAST)
def bf16_to_f32_u16(u, out):
    """(n,) uint16 bf16 bits -> (n,) float32 (exact widening)."""
    ob = out.view(np.uint32)
    for i in range(u.size):
        ob[i] = np.uint32(u[i]) << np.uint32(16)


# ---------------------------------------------------------------------------
# bfp16ebs8 (the B image): t7_codec.pack_b_nk, ported
# ---------------------------------------------------------------------------
@njit(**_FAST)
def pack_b_nk(x, out):
    """(calls, n, k) f32 -> flat uint8, the mm_bfp B image.

    Group = 8 consecutive k at one n.  One shared exponent per group (the group
    max), each element truncated toward -inf in two's complement, and
    min(shift, 7) because m32 >> 17 always lands in [-128, 127] (T7's identity,
    proven by the 40-case gate in T7 and re-proven here against the reference).
    """
    calls, n, k = x.shape
    nb, kb = n // GROUP, k // GROUP
    xb = x.view(np.uint32)
    for c in range(calls):
        for b in range(nb):
            for g in range(kb):
                for i in range(GROUP):
                    # the group is 8 consecutive k at ONE n (t3d_codec: exp has
                    # shape (calls, n, kb, GROUP), max over the LAST axis), NOT
                    # an 8x8 tile -- the tile grouping is (nb, kb), the group is
                    # (n, kb).
                    maxe = np.uint32(0)
                    for j in range(GROUP):
                        e = (xb[c, b * GROUP + i, g * GROUP + j]
                             >> np.uint32(23)) & np.uint32(0xFF)
                        if e > maxe:
                            maxe = e
                    o = (((c * nb + b) * kb + g) * GROUP + i) * GB
                    out[o] = np.uint8(maxe)
                    for j in range(GROUP):
                        bits = xb[c, b * GROUP + i, g * GROUP + j]
                        e = (bits >> np.uint32(23)) & np.uint32(0xFF)
                        mant = bits & np.uint32(0x7FFFFF)
                        if e != np.uint32(0):
                            mant = mant | np.uint32(0x800000)
                        sh = maxe - e
                        if sh > np.uint32(7):
                            sh = np.uint32(7)
                        if (bits >> np.uint32(31)) != np.uint32(0):
                            mant = (~mant) + np.uint32(1)
                        v = (mant >> (np.uint32(MANT_SHIFT) + sh)) & np.uint32(0xFF)
                        out[o + 1 + j] = np.uint8(v)


# ---------------------------------------------------------------------------
# the blocked layouts: a permutation, applied per row
# ---------------------------------------------------------------------------
@njit(**_FAST)
def unpack_blocked(u16, inv, out):
    """(rows, cols) uint16 bf16 -> (rows, cols) float32, out[r,j]=f(u[r,inv[j]])."""
    rows, cols = out.shape
    ob = out.reshape(-1).view(np.uint32)          # FLAT: a 2-D .view would
    for r in range(rows):                          # index rows, not elements
        base = r * cols
        for j in range(cols):
            v = u16[base + inv[j]]
            ob[base + j] = np.uint32(v) << np.uint32(16)


@njit(**_FAST)
def unpack_blocked_t(u16, inv, out, hd):
    """blocked bf16 -> float32 in the (M, H*HD) layout, in ONE pass.

    The delivered loop drains (H, M*HD) and then does
        np.ascontiguousarray(x.reshape(H, M, HD).transpose(1, 0, 2)).reshape(M, D)
    which is a 1.47 MB transpose + copy (measured: 3 transposes + 2 reshapes =
    1.57 ms/layer, T6 §1.1).  Since the gather below is already random access,
    writing straight into the transposed layout is free.
    """
    m = out.shape[0]                      # M rows of the (M, H*HD) result
    cols = m * hd                         # elements per drained head row
    rows = u16.size // cols
    ob = out.reshape(-1).view(np.uint32)
    d = out.shape[1]
    for h in range(rows):
        base = h * cols
        for mm in range(m):
            o = mm * d + h * hd
            for p in range(hd):
                ob[o + p] = np.uint32(u16[base + inv[mm * hd + p]]) << np.uint32(16)


@njit(**_FAST)
def pack_b_nk_t(x, out):
    """pack_b_nk on a TRANSPOSED view: x is (calls, k, n), packed as (n, k).

    t7_run transposes V first (np.ascontiguousarray(Qm["v"].transpose(0, 2, 1)),
    a 1.47 MB copy) only so pack_b_nk can read it; the kernel can read the
    8x8 tile in place instead.  Same bytes by construction (same elements, same
    group, same arithmetic) -- the gate compares against the reference.
    """
    calls, k, n = x.shape
    nb, kb = n // GROUP, k // GROUP
    xb = x.view(np.uint32)
    for c in range(calls):
        for b in range(nb):
            for g in range(kb):
                for i in range(GROUP):
                    maxe = np.uint32(0)
                    for j in range(GROUP):
                        e = (xb[c, g * GROUP + j, b * GROUP + i]
                             >> np.uint32(23)) & np.uint32(0xFF)
                        if e > maxe:
                            maxe = e
                    o = (((c * nb + b) * kb + g) * GROUP + i) * GB
                    out[o] = np.uint8(maxe)
                    for j in range(GROUP):
                        bits = xb[c, g * GROUP + j, b * GROUP + i]
                        e = (bits >> np.uint32(23)) & np.uint32(0xFF)
                        mant = bits & np.uint32(0x7FFFFF)
                        if e != np.uint32(0):
                            mant = mant | np.uint32(0x800000)
                        sh = maxe - e
                        if sh > np.uint32(7):
                            sh = np.uint32(7)
                        if (bits >> np.uint32(31)) != np.uint32(0):
                            mant = (~mant) + np.uint32(1)
                        v = (mant >> (np.uint32(MANT_SHIFT) + sh)) & np.uint32(0xFF)
                        out[o + 1 + j] = np.uint8(v)


@njit(**_FAST)
def pack_blocked(x, perm, out_u16):
    """(rows, cols) f32 -> (rows, cols) uint16 bf16 with out[r,p]=bf16(x[r,perm[p]])."""
    rows = x.shape[0]
    cols = out_u16.shape[1]
    xb = x.view(np.uint32)
    ob = out_u16.reshape(-1)              # FLAT: A[i] on a 2-D array indexes a
    for r in range(rows):                 # row, so every store below would have
        base = r * cols                   # been an out-of-bounds write
        for p in range(cols):
            ob[base + p] = _bf16_one(xb[r, perm[p]])


@njit(**_FAST)
def x_image64(x, pm, pj, out_u16):
    """(m, K) f32 -> (K//64, m, 64) uint16: 60 group packs in one pass.

    t7_run.x_image64 does this as 60 separate pack_blocked calls, each on a
    (1, 6144) FLATTENED (m=96, 64) slice -- so perm64 is a permutation of the
    whole 6144-element block, not of a row.  pm/pj are perm64 // 64 and
    perm64 % 64, i.e. the (row, column) the p-th output byte comes from.
    """
    m, K = x.shape
    ng = K // 64
    xb = x.view(np.uint32)
    ob = out_u16.reshape(-1)              # FLAT, same reason as pack_blocked
    for g in range(ng):
        base = g * m * 64
        for p in range(64 * m):
            ob[base + p] = _bf16_one(xb[pm[p], g * 64 + pj[p]])


# ---------------------------------------------------------------------------
# RoPE on the drain layout
# ---------------------------------------------------------------------------
@njit(**_FAST)
def rope_f32(y, cos, sin, out):
    """(H, M, D) f32, out[...,0::2]=xe*c-xo*s, out[...,1::2]=xe*s+xo*c.

    Same elements, same order, same rounding as t3_real.apply_rope with
    cos[None]/sin[None] (T7 §3.2's regrouping): no (M,H,D) copy is made.
    """
    H, M, D = y.shape
    for h in range(H):
        for m in range(M):
            for j in range(D // 2):
                xe = y[h, m, 2 * j]
                xo = y[h, m, 2 * j + 1]
                c = cos[m, j]              # the table is (M, HD//2), one entry
                s = sin[m, j]              # per rotated PAIR
                out[h, m, 2 * j] = xe * c - xo * s
                out[h, m, 2 * j + 1] = xe * s + xo * c


# ---------------------------------------------------------------------------
# numpy-shaped wrappers (allocate, like the reference API) + workspace reuse
# ---------------------------------------------------------------------------
class Scratch:
    """One buffer per (name, size), reused across layers.

    The in-loop cost of a numpy codec is 1.2-2.4x its idle cost because every
    call returns a fresh page (T6 §1.2).  A kernel that writes a persistent
    buffer never touches a new page -- that is half the win, and it is why the
    wrappers below take an explicit output.
    """

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.buf = {}

    def get(self, key, shape, dtype):
        if not self.enabled:
            return np.empty(shape, dtype)
        b = self.buf.get(key)
        if b is None or b.size < int(np.prod(shape)):
            b = np.empty(shape, dtype)
            self.buf[key] = b
        return b.reshape(shape) if b.shape != shape else b


def pack_b_nk_f32(x, scratch=None, key="b_nk"):
    x = np.ascontiguousarray(x, np.float32)
    calls, n, k = x.shape
    size = calls * (n // GROUP) * (k // GROUP) * GROUP * GB
    if scratch is None:
        out = np.empty(size, np.uint8)
    else:
        out = scratch.get(key, (size,), np.uint8)
    pack_b_nk(x, out)
    return out.reshape(calls, size // calls)


def unpack_bytes(raw, inv, shape, scratch=None, key="unpack"):
    u = np.frombuffer(bytes(raw), np.uint16)
    rows = shape[0]
    cols = u.size // rows
    if scratch is None:
        out = np.empty((rows, cols), np.float32)
    else:
        out = scratch.get(key, (rows, cols), np.float32)
    unpack_blocked(u.reshape(-1), inv.astype(np.int64), out)
    return out.reshape(shape)


def unpack_bytes_t(raw, inv, shape, hd, scratch=None, key="unpack_t"):
    """raw -> (M, H*HD) float32 straight from the blocked bytes."""
    u = np.frombuffer(bytes(raw), np.uint16)
    rows = shape[0]
    cols = u.size // rows
    m = cols // hd
    if scratch is None:
        out = np.empty((m, rows * hd), np.float32)
    else:
        out = scratch.get(key, (m, rows * hd), np.float32)
    unpack_blocked_t(u.reshape(-1), inv.astype(np.int64), out, hd)
    return out


def pack_b_nk_t_f32(x_kn, scratch=None, key="b_nk_t"):
    """x is (calls, k, n) -- the untransposed V drain."""
    x = np.ascontiguousarray(x_kn, np.float32)
    calls, k, n = x.shape
    size = calls * (n // GROUP) * (k // GROUP) * GROUP * GB
    if scratch is None:
        out = np.empty(size, np.uint8)
    else:
        out = scratch.get(key, (size,), np.uint8)
    pack_b_nk_t(x, out)
    return out.reshape(calls, size // calls)


def pack_blocked_bytes(x2d, perm, scratch=None, key="pblk"):
    x = np.ascontiguousarray(x2d, np.float32)
    rows, cols = x.shape
    if scratch is None:
        out = np.empty((rows, cols), np.uint16)
    else:
        out = scratch.get(key, (rows, cols), np.uint16)
    return _pack_blocked_into(x, perm, out)


def _pack_blocked_into(x, perm, out):
    pack_blocked(x, perm.astype(np.int64), out)
    return out


_PMJ = {}


def _pmj(perm64):
    key = id(perm64)
    hit = _PMJ.get(key)
    if hit is None or hit[0] is not perm64:
        p = np.ascontiguousarray(perm64, np.int64)
        hit = (perm64, p // 64, p % 64)
        _PMJ[key] = hit
    return hit[1], hit[2]


def x_image64_bytes(x_mk, perm64, scratch=None, key="ximg"):
    x = np.ascontiguousarray(x_mk, np.float32)
    m, K = x.shape
    if scratch is None:
        out = np.empty((K // 64, m, 64), np.uint16)
    else:
        out = scratch.get(key, (K // 64, m, 64), np.uint16)
    pm, pj = _pmj(perm64)
    x_image64(x, pm, pj, out)
    return out


def bf16_bytes_f32(x, scratch=None, key="bf16"):
    x = np.ascontiguousarray(x, np.float32)
    if scratch is None:
        out = np.empty(x.size, np.uint16)
    else:
        out = scratch.get(key, (x.size,), np.uint16)
    bf16_rne_u16(x.reshape(-1), out)
    return out


def bf16_to_f32_bits(u16, shape=None, scratch=None, key="f32"):
    u = np.ascontiguousarray(u16, np.uint16).reshape(-1)
    if scratch is None:
        out = np.empty(u.size, np.float32)
    else:
        out = scratch.get(key, (u.size,), np.float32)
    bf16_to_f32_u16(u, out)
    return out if shape is None else out.reshape(shape)


def install(scratch=None):
    """Patch t3d_codec's codecs with the JIT ports (BYTE-EQUAL, gated)."""
    import t3d_codec as K

    K.pack_b_nk = lambda a: pack_b_nk_f32(a)
    K.unpack_blocked = lambda raw, inv, shape: unpack_bytes(raw, inv, shape)
    K.pack_blocked = lambda x2d, perm: pack_blocked_bytes(x2d, perm).tobytes()
    K.bf16_bytes = lambda x: bf16_bytes_f32(x).tobytes()
    K.bf16_to_f32 = lambda b: bf16_to_f32_bits(np.frombuffer(bytes(b), np.uint16))
    # bf16_rne stays numpy on purpose: it is the 3-line bit map and its callers
    # inside t3_real are not on this loop's hot path.
    return K