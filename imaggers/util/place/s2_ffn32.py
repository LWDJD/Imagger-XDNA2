# SPDX-License-Identifier: MIT
"""s2_ffn32.py -- the 32-core FFN GEMM segment: executor + CPU dry path.

Pure numpy (runs on the py310 device leg AND on py312); no mlir_aie import.
The only toolchain-derived artifact it needs is the baked C permutation
(c_perm_<m>x<n>.npy, produced by gen_c_perm.py from t1_design.c_layout).

    py310 python s2_ffn32.py --selftest          # layout math, byte level
    py310 python s2_ffn32.py --dry --key layers.0 # real weights, CPU reference
    py310 python s2_ffn32.py --bench-b --layers 30

What it guarantees:
  * pack_a / unpack_a are exact inverses on bf16 input;
  * pack_b_from_blocks / unpack_b are exact inverses on the model's own BFP16
    blocks (no re-quantization: the model's bytes are already BFP16);
  * pack_c / unpack_c are exact inverses, and unpack_c is the DEVICE's C order
    (baked from the toolchain's own layout object);
  * convert_a(product_bytes) == pack_a(unpack_product_a(product_bytes)).

The dry executor runs the real FFN shape (M=1056, gate+up+down) entirely in
numpy through those same layout functions, so a layout bug shows up as a
non-zero difference against the direct reference -- on the CPU, no device.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
D = 3840            # model width
FF = 10240          # FFN hidden width
M_TILE = 32
KK = 8
COLS = 8
CPC = 4

# the four deployed images (tag -> geometry)
IMAGES = {
    "gu_main": dict(tag="ffn_gu_m32",    M=1024, K=3840,  N=10240, n=512, ntiles=20, cpc=4),
    "gu_tail": dict(tag="ffn_tail_m32",  M=32,   K=3840,  N=10240, n=320, ntiles=1,  cpc=4),
    "dn_main": dict(tag="ffn_dn2_m32",   M=1024, K=10240, N=3840,  n=384, ntiles=10, cpc=4),
    "dn_tail": dict(tag="ffn_tail_dn2",  M=32,   K=10240, N=3840,  n=240, ntiles=1,  cpc=2),
}


# ===========================================================================
# BFP16 (v8bfp16ebs8) codec -- numpy ports of aie.utils.bfp, BLOCK = 8
# ===========================================================================
_BLOCK = 8
_BLOCK_BYTES = 9
_MANTISSA_SHIFT = 23 - 7 + 1


def encode_bfp(x, *, rounding="floor"):
    """float32 (..., n%8==0) -> uint8 (..., n*9//8), blocks along the last axis."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    n = x.shape[-1]
    if n % _BLOCK:
        raise ValueError(f"encode_bfp: last axis {n} is not a multiple of 8")
    lead = x.shape[:-1]
    bits = x.view(np.uint32).reshape(*lead, n // _BLOCK, _BLOCK)
    sign = (bits >> 31).astype(bool)
    exp = (bits >> 23) & 0xFF
    mant = (bits & 0x7FFFFF) | np.where(exp != 0, np.uint32(0x800000), np.uint32(0))
    max_exp = exp.max(axis=-1, keepdims=True)
    shift = (max_exp - exp).astype(np.int64)
    far = shift >= 32
    if rounding == "conv_even":
        signed = np.where(sign, -mant.astype(np.int64), mant.astype(np.int64))
        v = np.rint(signed / np.exp2(np.minimum(_MANTISSA_SHIFT + shift, 62)))
        v = np.clip(v, -128, 127)
    else:
        m32 = np.where(sign, (-mant.astype(np.int64)) & 0xFFFFFFFF, mant).astype(np.uint32)
        v = ((m32 >> _MANTISSA_SHIFT) & 0xFF).astype(np.uint8).view(np.int8).astype(np.int32)
        v = np.right_shift(v, np.minimum(shift, 31).astype(np.int32))
    out = np.empty(lead + (n // _BLOCK,), dtype=np.dtype(
        [("exponent", np.uint8), ("mantissas", np.int8, (_BLOCK,))]))
    out["exponent"] = max_exp[..., 0]
    out["mantissas"] = np.where(far, np.where(sign, -1, 0), v)
    return out.view(np.uint8).reshape(*lead, n // _BLOCK * _BLOCK_BYTES)


def decode_bfp(b):
    """uint8 (..., n*9//8) -> float32 (..., n); exact inverse of the block map."""
    b = np.ascontiguousarray(b, dtype=np.uint8)
    nb = b.shape[-1]
    if nb % _BLOCK_BYTES:
        raise ValueError("decode_bfp: not a multiple of 9 bytes")
    lead = b.shape[:-1]
    dt = np.dtype([("exponent", np.uint8), ("mantissas", np.int8, (_BLOCK,))])
    blk = b.view(dt)
    scale = np.ldexp(1.0, blk["exponent"].astype(np.int32) - 127 - 6)
    vals = blk["mantissas"].astype(np.float64) * scale[..., None]
    return vals.astype(np.float32).reshape(*lead, nb // _BLOCK_BYTES * _BLOCK)


# ===========================================================================
# A: product layout (a_pack 96x64) and the 32-core image layout
# ===========================================================================
def _bf16_bits(x):
    """f32 -> the uint16 bit pattern of its bf16 rounding (2 bytes/value).

    parent fix 2026-11-01: the old version rounded by building a whole float32
    array (16 MB write + read per call).  The rounding is pure integer work on
    the bit pattern, so stay in uint32 and emit the high half directly -- same
    bytes, ~2x faster.
    """
    u = np.ascontiguousarray(np.asarray(x, np.float32)).view(np.uint32)
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16).astype(np.uint16)


def _from_bf16_bytes(b, shape):
    v = np.frombuffer(np.ascontiguousarray(b).tobytes(), np.uint16)
    return (v.astype(np.uint32) << 16).view(np.float32).reshape(shape)


def _bf16(x):
    """round f32 -> bf16 precision (RNE) and back to f32."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    lsb = (u >> 16) & 1
    r = (u + 0x7FFF + lsb) >> 16
    return (r.astype(np.uint32) << 16).view(np.float32)


def tile_map(M, m=M_TILE, cols=COLS, cpc=CPC):
    """M-tile index per (column, core); contiguous per column, -1 = dummy."""
    nm = M // m
    base, rem = divmod(nm, cols)
    out, nxt = [], 0
    for c in range(cols):
        cnt = base + (1 if c < rem else 0)
        out.append([nxt + p if p < cnt else -1 for p in range(cpc)])
        nxt += cnt
    assert nxt == nm, (nxt, nm)
    return out


_PA_NB = None
_PA_TM = {}


def _pack_a_kernel():
    # 2026-11-01 parent: one fused pass (read f32 bits, RNE-round, scatter into the
    # device order) instead of the numpy reshape/transpose/gather chain.
    # MEASURED identical bytes (incl. NaN / inf / -0.0 / denormal) at 26.6 -> 2.1 ms
    # for the 1056x3840 projection A, i.e. 9.36x.
    global _PA_NB
    if _PA_NB is None:
        try:
            from numba import njit, prange

            @njit(parallel=True, cache=True)
            def _k(uf, tm, m, kk, cols, cpc, K, out):
                nkk = K // kk
                blk = m * kk
                for q in prange(cols * nkk * cpc):
                    p = q % cpc
                    rest = q // cpc
                    kbi = rest % nkk
                    c = rest // nkk
                    t = tm[c, p]
                    if t < 0:
                        t = 0
                    base = q * blk
                    for i in range(m):
                        src = (t * m + i) * K + kbi * kk
                        dst = base + i * kk
                        for j in range(kk):
                            u = uf[src + j]
                            lsb = (u >> 16) & 1
                            out[dst + j] = (u + 0x7FFF + lsb) >> 16
            _PA_NB = _k
        except Exception:
            _PA_NB = False
    return _PA_NB or None


def _tm_array(M, m, cols, cpc):
    key = (M, m, cols, cpc)
    if key not in _PA_TM:
        _PA_TM[key] = np.array(tile_map(M, m, cols, cpc), np.int64)
    return _PA_TM[key]


def pack_a_fast(x, m=M_TILE, kk=KK, cols=COLS, cpc=CPC):
    k = _pack_a_kernel()
    if k is None:
        return None
    xf = np.ascontiguousarray(np.asarray(x, np.float32))
    M, K = xf.shape
    if M % m or K % kk:
        return None
    out = np.empty(cols * (K // kk) * cpc * m * kk, np.uint16)
    k(xf.view(np.uint32).reshape(-1), _tm_array(M, m, cols, cpc), m, kk, cols, cpc, K, out)
    return out.view(np.uint8)


def pack_a(x, m=M_TILE, kk=KK, cols=COLS, cpc=CPC):
    """(M,K) float -> [cols][nkk][cpc][m*kk*2] int8, the gu/dn main A payload."""
    _fast = pack_a_fast(x, m, kk, cols, cpc)
    if _fast is not None:
        return _fast
    # parent fix 2026-11-01: _bf16_bits() already performs the RNE rounding, so
    # the extra float32 round-trip here only cost a 16 MB alloc + copy.
    xf = np.ascontiguousarray(np.asarray(x, np.float32))
    M, K = xf.shape
    assert M % m == 0 and K % kk == 0
    nkk = K // kk
    tm = tile_map(M, m, cols, cpc)
    # parent fix 2026-11-01: the nested loops below ran 32*8*60 = 15,360 numpy
    # calls per A pack (209 ms per product layer).  The payload is exactly
    # [col][kblock][core][m*k] with each (m, kk) block row-major, so one
    # transpose over the whole matrix produces every block at once.
    U = _bf16_bits(xf)                                  # (M, K) uint16
    nm = M // m
    V = np.ascontiguousarray(
        U.reshape(nm, m, nkk, kk).transpose(0, 2, 1, 3)
    ).reshape(nm, nkk, m * kk)
    # parent fix 2026-11-01: one gather over the tile axis instead of 256
    # slice copies (188 MB of memcpy -> 8 MB of gather + 8 MB transpose).
    idx = np.array([[0 if t < 0 else t for t in row] for row in tm], np.int64)
    seg = np.ascontiguousarray(V[idx].transpose(0, 2, 1, 3))
    return seg.view(np.uint8).reshape(-1)


def unpack_a(buf, M, K, m=M_TILE, kk=KK, cols=COLS, cpc=CPC):
    """inverse of pack_a: -> (M,K) float32 (bf16 values)."""
    nkk = K // kk
    seg = np.asarray(buf, np.uint8).reshape(cols, nkk, cpc, m * kk * 2)
    tm = tile_map(M, m, cols, cpc)
    out = np.zeros((M, K), np.float32)
    for c in range(cols):
        for p in range(cpc):
            t = tm[c][p]
            if t < 0:
                continue
            for kb in range(nkk):
                out[t * m:(t + 1) * m, kb * kk:(kb + 1) * kk] = _from_bf16_bytes(
                    seg[c, kb, p], (m, kk))
    return out


def pack_a_tail(x, m=M_TILE, kk=KK):
    """(m,K) float -> [nkk][m*kk*2] int8, the tail's A payload."""
    xf = np.ascontiguousarray(np.asarray(x, np.float32))
    M, K = xf.shape
    assert M == m and K % kk == 0
    nkk = K // kk
    # parent fix 2026-11-01: vectorized (was 60 numpy calls per pack)
    U = _bf16_bits(xf)                                  # (m, K) uint16
    out = np.ascontiguousarray(
        U.reshape(m, nkk, kk).transpose(1, 0, 2)).reshape(nkk, m * kk)
    return np.ascontiguousarray(out).view(np.uint8).reshape(-1)


def unpack_a_tail(buf, K, m=M_TILE, kk=KK):
    nkk = K // kk
    seg = np.asarray(buf, np.uint8).reshape(nkk, m * kk * 2)
    out = np.zeros((m, K), np.float32)
    for kb in range(nkk):
        out[:, kb * kk:(kb + 1) * kk] = _from_bf16_bytes(seg[kb], (m, kk))
    return out


# ---- the product's A layout (prod_loop_bigm.a_pack, 96x64 tiles) ----------
_PROD_ORDER = None


def _prod_order():
    global _PROD_ORDER
    if _PROD_ORDER is None:
        p = HERE.parent.parent / "block" / "bigm" / "a_order_96x64.npy"
        _PROD_ORDER = np.load(p).astype(np.int64)
        assert _PROD_ORDER.shape == (96 * 64,), _PROD_ORDER.shape
    return _PROD_ORDER


def pack_product_a(x, th=96, tw=64):
    """The product's A payload (prod_loop_bigm.a_pack), verbatim geometry."""
    order = _prod_order()
    xa = np.ascontiguousarray(np.asarray(x))
    rows, cols = xa.shape
    assert rows % th == 0 and cols % tw == 0, (rows, cols)
    u = _bf16_bits(xa)                     # bf16 elements, 2 B each
    t = u.reshape(rows // th, th, cols // tw, tw).transpose(0, 2, 1, 3)
    return np.ascontiguousarray(t.reshape(rows // th, cols // tw, -1)[..., order]).tobytes()


def unpack_product_a(b, rows, cols, th=96, tw=64):
    """inverse of pack_product_a -> (rows, cols) float32."""
    inv = np.argsort(_prod_order())
    u = np.frombuffer(b, np.uint16).reshape(rows // th, cols // tw, th * tw)
    u = u[..., inv]
    t = u.reshape(rows // th, cols // tw, th, tw).transpose(0, 2, 1, 3)
    v = (np.ascontiguousarray(t).reshape(-1).astype(np.uint32) << 16)
    return v.view(np.float32).reshape(rows, cols)


def convert_a(product_bytes, rows, cols, m=M_TILE, kk=KK, cols_=COLS, cpc=CPC):
    """product a_pack bytes -> (main payload, tail payload or None).

    The product ships ONE payload for all SEQ rows; this line needs the
    32-tile main image (rows 0..1023) and the 32-row tail image (rows
    1024..1055) as two separate BOs.
    """
    x = unpack_product_a(product_bytes, rows, cols)
    main_rows = min(rows, 1024)
    a_main = pack_a(x[:main_rows], m=m, kk=kk, cols=cols_, cpc=cpc)
    a_tail = (pack_a_tail(x[main_rows:], m=m, kk=kk)
              if rows > main_rows else None)
    return a_main, a_tail


# ===========================================================================
# B: the model's own BFP16 blocks -> the image's n-slice order
# ===========================================================================
def pack_b_from_blocks(blocks, K, N, n, cpc=1):
    """(N, K//8, 9) uint8 blocks (model order, blocks along K) -> payload.

    This is pack_b_blocks' tail with NO re-quantization: the model's bytes are
    already BFP16, so only the block order changes.
    """
    blocks = np.ascontiguousarray(blocks, np.uint8)
    assert blocks.shape == (N, K // 8, 9), blocks.shape
    assert N % n == 0 and n % 8 == 0
    cA = 1
    packed = (blocks.reshape(N // n, n // 8, 8, K // KK, cA, 9)
              .transpose(0, 3, 1, 4, 2, 5).reshape(-1))
    return np.ascontiguousarray(packed)


def pack_b_tail_from_blocks(blocks, K, N, n, cols=COLS, cpc=CPC):
    """tail: [cols][nkk][cpc][n*9], core (c,p) gets n-tile c*cpc+p."""
    blocks = np.ascontiguousarray(blocks, np.uint8)
    assert blocks.shape == (N, K // 8, 9), blocks.shape
    nt = N // n
    assert nt == cols * cpc, (nt, cols, cpc)
    per = pack_b_from_blocks(blocks, K, N, n).reshape(nt, K // KK, n * 9)
    out = np.empty((cols, K // KK, cpc, n * 9), np.uint8)
    for c in range(cols):
        for p in range(cpc):
            out[c, :, p] = per[c * cpc + p]
    return out.reshape(-1)


def unpack_b(buf, K, N, n):
    """payload -> (K,N) float32 (BFP16 dequantized)."""
    nt = N // n
    raw = np.asarray(buf, np.uint8).reshape(nt, K // KK, n // 8, 1, 8, 9)
    blocks = raw.transpose(0, 2, 4, 1, 3, 5).reshape(N, K // 8, 9)
    return decode_bfp(blocks.reshape(N, K // 8 * 9)).reshape(N, K).T


def unpack_b_tail(buf, K, N, n, cols=COLS, cpc=CPC):
    nt = cols * cpc
    raw = np.asarray(buf, np.uint8).reshape(cols, K // KK, cpc, n * 9)
    # memory order is [c][kb][p]; the per-n-tile order is (c,p) == c*cpc+p, so
    # the p axis has to come inside c and outside kb before flattening.
    per = raw.transpose(0, 2, 1, 3).reshape(cols * cpc, K // KK, n * 9)
    return unpack_b(per.reshape(-1), K, N, n)


# ===========================================================================
# C: the device's joined 4-leaf stream
# ===========================================================================
_PERMS = {}


def _cperm(m, n):
    key = (m, n)
    if key not in _PERMS:
        p = HERE / f"c_perm_{m}x{n}.npy"
        _PERMS[key] = np.load(p).astype(np.int64) if p.is_file() else None
    return _PERMS[key]


def unpack_c(raw, cols, ntiles, cpc, m, n):
    """device C -> (cols*cpc*m, ntiles*n) float32 (bf16 values)."""
    perm = _cperm(m, n)
    c_leaf = m * n * 2
    a = np.asarray(raw, np.uint8).reshape(cols, ntiles, cpc, c_leaf)
    out = np.zeros((cols * cpc * m, ntiles * n), np.float32)
    for c in range(cols):
        for p in range(cpc):
            base = (c * cpc + p) * m
            for nt in range(ntiles):
                v = _from_bf16_bytes(a[c, nt, p], (m * n,))
                if perm is not None:
                    v = v[perm]
                out[base:base + m, nt * n:(nt + 1) * n] = v.reshape(m, n)
    return out


def unpack_c_tail(raw, cols, cpc, m, n):
    """tail C -> (m, cols*cpc*n): core (c,p) owns columns (c*cpc+p)*n."""
    perm = _cperm(m, n)
    c_leaf = m * n * 2
    a = np.asarray(raw, np.uint8).reshape(cols, cpc, c_leaf)
    out = np.zeros((m, cols * cpc * n), np.float32)
    for c in range(cols):
        for p in range(cpc):
            v = _from_bf16_bytes(a[c, p], (m * n,))
            if perm is not None:
                v = v[perm]
            out[:, (c * cpc + p) * n:(c * cpc + p + 1) * n] = v.reshape(m, n)
    return out


def pack_c_tail(x, cols, cpc, m, n):
    """inverse of unpack_c_tail."""
    perm = _cperm(m, n)
    c_leaf = m * n * 2
    out = np.zeros((cols, cpc, c_leaf), np.uint8)
    for c in range(cols):
        for p in range(cpc):
            v = _bf16_bits(x[:, (c * cpc + p) * n:(c * cpc + p + 1) * n]).reshape(-1)
            pk = np.empty_like(v)
            if perm is not None:
                pk[perm] = v
            else:
                pk = v
            out[c, p] = np.frombuffer(pk.tobytes(), np.uint8)
    return out.reshape(-1)


def pack_c(c_tiles, cols, ntiles, cpc, m, n):
    """inverse of unpack_c: (cols*cpc*m, ntiles*n) float -> device bytes."""
    perm = _cperm(m, n)
    c_leaf = m * n * 2
    out = np.zeros((cols, ntiles, cpc, c_leaf), np.uint8)
    x = np.asarray(c_tiles, np.float32)
    for c in range(cols):
        for p in range(cpc):
            base = (c * cpc + p) * m
            for nt in range(ntiles):
                v = _bf16_bits(x[base:base + m, nt * n:(nt + 1) * n]).reshape(-1)
                packed = np.empty_like(v)
                if perm is not None:
                    packed[perm] = v
                else:
                    packed = v
                out[c, nt, p] = np.frombuffer(packed.tobytes(), np.uint8)
    return out.reshape(-1)

# ===========================================================================
# executor: CPU dry path + device path
# ===========================================================================
def _decode_blocks(blocks, K, N):
    """model BFP16 blocks (N, K//8, 9) -> (K, N) float32."""
    return decode_bfp(np.ascontiguousarray(blocks, np.uint8).reshape(
        N, K // 8 * 9)).reshape(N, K).T


def gemm_dry(x, blocks, K, N, n, tail=False, m=M_TILE, cols=COLS, cpc=CPC):
    """One GEMM (main or tail) THROUGH the image layouts; returns (C, info)."""
    xq = _bf16(np.asarray(x, np.float32))
    rows = xq.shape[0]
    if tail:
        assert rows == m
        a_buf = pack_a_tail(xq, m=m, kk=KK)
        a_back = unpack_a_tail(a_buf, K, m=m, kk=KK)
        b_buf = pack_b_tail_from_blocks(blocks, K, N, n, cols=cols, cpc=cpc)
        b_back = unpack_b_tail(b_buf, K, N, n, cols=cols, cpc=cpc)
        ntiles = 1
    else:
        a_buf = pack_a(xq, m=m, kk=KK, cols=cols, cpc=cpc)
        a_back = unpack_a(a_buf, rows, K, m=m, kk=KK, cols=cols, cpc=cpc)
        b_buf = pack_b_from_blocks(blocks, K, N, n)
        b_back = unpack_b(b_buf, K, N, n)
        ntiles = N // n
    w_full = _decode_blocks(blocks, K, N)
    info = {"rows": int(rows), "a_equal": bool(np.array_equal(a_back, xq)),
            "b_equal": bool(np.array_equal(b_back, w_full)),
            "a_bytes": int(a_buf.size), "b_bytes": int(b_buf.size)}
    c = a_back @ b_back
    if tail:
        c_buf = pack_c_tail(c, cols, cpc, m, n)
        c_back = unpack_c_tail(c_buf, cols, cpc, m, n)
    else:
        c_buf = pack_c(c, cols, ntiles, cpc, m, n)
        c_back = unpack_c(c_buf, cols, ntiles, cpc, m, n)
    info["c_equal"] = bool(np.array_equal(c_back, _bf16(c)))
    return c_back[:rows], info


def ffn32_dry(xn, w1, w3, w2, *, key="layers.0", stages=None):
    """The whole FFN segment on CPU through the same layouts.  Returns (y, info).

    xn: (M, 3840) float; w1/w3/w2: the model's BFP16 blocks, each
    (out, in//8, 9) uint8 -- blocks along the contraction axis.
    """
    M = xn.shape[0]
    info = {"key": key, "M": int(M), "parts": {}}
    K, N, n = 3840, 10240, 512
    gate = np.zeros((M, N), np.float32)
    up = np.zeros((M, N), np.float32)
    for tag, tgt, w in (("w1", gate, w1), ("w3", up, w3)):
        if M > 1024:
            c_main, i1 = gemm_dry(xn[:1024], w, K, N, n)
            # the tail image splits N over 32 cores: n = N/32 (320 for gu)
            c_tail, i2 = gemm_dry(xn[1024:], w, K, N, N // (COLS * CPC), tail=True)
            tgt[:1024], tgt[1024:] = c_main, c_tail
            info["parts"][tag] = {"main": i1, "tail": i2}
        else:
            c, i1 = gemm_dry(xn, w, K, N, n)
            tgt[:] = c
            info["parts"][tag] = {"main": i1}
    h = _bf16(gate * (1.0 / (1.0 + np.exp(-gate))) * up)
    if stages is not None:
        stages.update(gate=gate, up=up, h=h)
    y = np.zeros((M, 3840), np.float32)
    if M > 1024:
        y[:1024], i1 = gemm_dry(h[:1024], w2, 10240, 3840, 384)
        y[1024:], i2 = gemm_dry(h[1024:], w2, 10240, 3840, 240, tail=True, cpc=2)
        info["parts"]["w2"] = {"main": i1, "tail": i2}
    else:
        y[:], i1 = gemm_dry(h, w2, 10240, 3840, 384)
        info["parts"]["w2"] = {"main": i1}
    return y, info


def ffn32_reference(xn, w1, w3, w2):
    """Straight numpy FFN on the same quantized weights (no layout detour).

    The device stores every C tile as bf16, so the reference rounds there too
    -- otherwise the comparison measures the bf16 output rounding, not the
    layouts (0.47 % maxabs of the tile max, corr 0.99999).
    """
    # the SAME A-image contract as ffn32_run (f32 as-is, uint16 bf16 image by
    # bit view, anything else refuses): a reference that decoded differently
    # from the executor would hide exactly the defect as_a_f32 exists for.
    xq = _bf16(as_a_f32(xn, "ffn32_reference"))
    gate = _bf16(xq @ _decode_blocks(w1, 3840, 10240))
    up = _bf16(xq @ _decode_blocks(w3, 3840, 10240))
    h = _bf16(gate * (1.0 / (1.0 + np.exp(-gate))) * up)
    return _bf16(h @ _decode_blocks(w2, 10240, 3840))


# ===========================================================================
# THE PLAN-DRIVEN DEVICE PATH ('xd'): one M-tile per core, ndisp dispatches
# ===========================================================================
# WHAT WAS MISSING AND IS NOW HERE
#   The plan (webui/npu_plan.plan + imaggers/util/place/npu_synth.py) says
#   WHICH images to run and at which geometry; it carries no BO names, no
#   packers, no per-layer payload, no drain and no host glue.  This section is
#   that glue, and it is the only place that knows the four answers:
#
#     BO set        dev.bo["b_<tag>_x"|"_w"|"_o"], tag = npu_synth.tag_for(plan,
#                   gemm).  The PRODUCT builds them (t18_run.Dev.__init__ loops
#                   BIGM_SIZES), so the sizes must be the plan's a/b/c_bytes.
#                   Two extra BOs are allocated lazily on group ids 4 and 5: the
#                   up-projection's W and its C, so gate and up are both in
#                   flight (one W BO would be overwritten under the other's
#                   read).
#     packers       pack_kk.pack_a_kk (A, bf16) and pack_kk.pack_b_kk (B, a
#                   PERMUTATION of the model's BFP16 blocks -- never a
#                   re-encode: 0.297 % of the model's blocks are non-canonical).
#     per-layer B   payloads_for()/load_blocks(): the model's own
#                   weights/block<N>.npz, repacked at the plan's kk.  The
#                   product's resident CACHE_BG is in the DELIVERED image's
#                   order, so it cannot be reused here (see plan_ffn.py).
#     drain         unpack_c_plan(): the device's 8x8-block C order, derived in
#                   closed form.  s2_ffn32._cperm() reads a BAKED
#                   c_perm_<m>x<n>.npy and returns None (identity!) when the
#                   file is absent -- fine for m=32, silently wrong for
#                   m=16/48.  The closed form is verified equal to all seven
#                   baked files by gate_ffn32.py.
#     host glue     the silu(gate)*up between the two GEMMs, the Mblk row
#                   padding per dispatch, the ndisp loop and the pad-row drop.
#
# WHY THE GU IMAGE IS DISPATCHED TWICE, WITH A BARRIER BETWEEN
#   gate and up are the SAME shape (K=3840, N=10240) and the same image; the
#   plan has one "gu" image, so the second launch is a SAME-IMAGE RELAUNCH.
#   t18_run.Dev.barrier documents the measured law: the FFN image never
#   survives a same-image relaunch (a ~7 s state-8 stall, 30/30 layers), and
#   one launch of a DIFFERENT xclbin in between removes it completely, while a
#   second kernel handle on the same xclbin is NOT a fix (it reports a false
#   completion).  So the delivered fix is used: dev.barrier(<other image>)
#   costs ~1.7-6.5 ms.  T22_PLAN_FFN_BARRIER=0 disables it for an A/B.
# ===========================================================================
_PLAN_CACHE = {}
_TAG_CACHE = {}
_CPERM = {}
_ALT_BO = {}
#: T22_PLAN_FFN_PROBE=1 one-shot record (DET-BISECT); empty and unused when off
_PROBED = set()
_PROBE_YREF = None


def repo_root():
    return HERE.parents[2]


def _add_path(p):
    import sys as _sys
    p = str(p)
    if p not in _sys.path:
        _sys.path.insert(0, p)
    return p


def pack_kk():
    """The kk-parametric host packers (same directory as this file)."""
    _add_path(HERE)
    import pack_kk as _PK
    return _PK


def plan_for(M, prefer=None, fuse_gu=False, fuse_fill=1.0):
    """The planner's answer for M rows (cached).  Pure: no device, no files.

    fuse_gu plans the FUSED gu image (silu(gate)*up inside the dispatch); the
    plan then carries fuse_gu=1, a smaller gu n and the pair B object, and the
    executor is _ffn32_run_fused instead of the two-dispatch form.
    """
    key = (int(M), bool(prefer), bool(fuse_gu), float(fuse_fill))
    if key not in _PLAN_CACHE:
        _add_path(repo_root())
        from webui import npu_plan as _P
        res = _P.plan(int(M), prefer=prefer, fuse_gu=bool(fuse_gu),
                      fuse_fill=float(fuse_fill))
        if not res["ok"]:
            raise RuntimeError("npu_plan refused M=%d: %s" % (M, res["reason"]))
        _PLAN_CACHE[key] = res["plan"]
    return _PLAN_CACHE[key]


def tags_for(plan):
    """{gemm: image tag} -- npu_synth.tag_for is the ONE naming authority.

    The tag is also the xclbin stem, so this cannot be re-derived here without
    risking a name the build never produced.
    """
    pk = plan["plan_key"]
    if pk not in _TAG_CACHE:
        _add_path(HERE)
        import npu_synth as _SYN
        _TAG_CACHE[pk] = {g: _SYN.tag_for(plan, g) for g in ("gu", "dn", "proj")}
    return _TAG_CACHE[pk]


def cperm(m, n):
    """rowmajor -> the device's C order: [m/8][n/8][8][8] (t1_design.c_layout).

    Closed form, so it exists for EVERY (m, n) the planner can emit -- unlike
    the baked c_perm_<m>x<n>.npy files (m=32 only).  gate_ffn32.py asserts it
    equals all seven baked files byte for byte.
    """
    key = (int(m), int(n))
    if key not in _CPERM:
        mm, nn = key
        r = np.arange(mm)[:, None]
        c = np.arange(nn)[None, :]
        _CPERM[key] = (((r // 8) * (nn // 8) + (c // 8)) * 64
                       + (r % 8) * 8 + (c % 8)).reshape(-1)
    return _CPERM[key]


def unpack_c_plan(raw, plan, gemm):
    """device C bytes -> (Mblk, N) float32 (bf16 values), plan's own tile map."""
    m = int(plan["m"])
    g = plan["gemms"][gemm]
    n, ntiles = int(g["n"]), int(g["ntiles"])
    cols, cpc = int(plan["cols"]), int(plan["cpc"])
    c_leaf = m * n * 2
    perm = cperm(m, n)
    # bytes / bytearray / memoryview (bo.read) or an ndarray (the fake device)
    if isinstance(raw, np.ndarray):
        buf = np.ascontiguousarray(raw, np.uint8).reshape(-1)
    else:
        buf = np.frombuffer(bytes(memoryview(raw).cast("B")), np.uint8)
    a = buf.reshape(cols, ntiles, cpc, c_leaf)
    out = np.zeros((int(plan["Mblk"]), int(g["N"])), np.float32)
    tm = plan["tile_map"]
    for c in range(cols):
        for p in range(cpc):
            t = tm[c][p]
            if t < 0:
                continue
            base = t * m
            for nt in range(ntiles):
                v = _from_bf16_bytes(a[c, nt, p], (m * n,))[perm]
                out[base:base + m, nt * n:(nt + 1) * n] = v.reshape(m, n)
    return out


def unpack_c_plan_fast(raw, plan, gemm):
    """unpack_c_plan without the per-element gather -- BIT-IDENTICAL output.

    The 8x8 C permutation is a BLOCK transpose: cperm()'s index is
    ((r//8)*(n//8) + (c//8))*64 + (r%8)*8 + (c%8), i.e. the payload is
    [m/8][n/8][8][8] and the value array is [m/8][8][n/8][8].  So one reshape
    plus transpose per (column, core) does what unpack_c_plan does with a
    fancy-index gather of m*n elements -- 1024-2048 of those per dispatch, which
    is why the gather costs 57-250 ms at 512^2..1920x1080 where this costs a
    few ms.

    Same contract, same values: gate_ffn32.py asserts array_equal against
    unpack_c_plan for every baked (m, n) and for the fused gu's own geometry.
    """
    m = int(plan["m"])
    g = plan["gemms"][gemm]
    n, ntiles = int(g["n"]), int(g["ntiles"])
    cols, cpc = int(plan["cols"]), int(plan["cpc"])
    c_leaf = m * n * 2
    if isinstance(raw, np.ndarray):
        buf = np.ascontiguousarray(raw, np.uint8).reshape(-1)
    else:
        buf = np.frombuffer(bytes(memoryview(raw).cast("B")), np.uint8)
    # [c][nt][p][m/8][8][n/8][8] -> [c][nt][p][m/8][n/8][8][8] (the block
    # transpose), in ONE contiguous copy, still in uint16 (the bf16 bits)
    v = np.ascontiguousarray(
        buf.view(np.uint16).reshape(cols, ntiles, cpc, m // 8, n // 8, 8, 8)
        .transpose(0, 1, 2, 3, 5, 4, 6)).reshape(cols, ntiles, cpc, m, n)
    u16 = np.zeros((int(plan["Mblk"]), int(g["N"])), np.uint16)
    tm = plan["tile_map"]
    for c in range(cols):
        for p in range(cpc):
            t = tm[c][p]
            if t < 0:
                continue
            # this core's ntiles tiles of (m, n) -> its Mblk rows, in n order
            u16[t * m:(t + 1) * m, :] = v[c, :, p].transpose(1, 0, 2).reshape(
                m, ntiles * n)
    return (u16.astype(np.uint32) << 16).view(np.float32)


_UC_NB = None


def _unpack_c_kernel():
    """A numba kernel for the same drain: the gather in C, 64 values a task.

    The numpy bulk transpose moves the whole payload twice (a 31 MB transpose
    copy, then a 43 MB widening) and measured 42-100 ms per dispatch; this walks
    the payload once and writes f32 directly.  Same values, same order.
    """
    global _UC_NB
    if _UC_NB is None:
        try:
            from numba import njit, prange
        except Exception:                                        # noqa: BLE001
            _UC_NB = False
            return None

        @njit(parallel=True, cache=True, nogil=True)
        def _k(raw16, out32, tm, cols, cpc, ntiles, m, n, N, nb_r, nb_c):
            for q in prange(cols * cpc * ntiles * nb_r * nb_c):
                bc = q % nb_c
                r1 = q // nb_c
                br = r1 % nb_r
                r2 = r1 // nb_r
                nt = r2 % ntiles
                r3 = r2 // ntiles
                p = r3 % cpc
                c = r3 // cpc
                t = tm[c, p]
                if t < 0:
                    continue
                base = ((c * ntiles + nt) * cpc + p) * (m * n) + (br * nb_c + bc) * 64
                r0 = t * m + br * 8
                c0 = nt * n + bc * 8
                for i in range(8):
                    dst = (r0 + i) * N + c0
                    src = base + i * 8
                    for j in range(8):
                        out32[dst + j] = np.uint32(raw16[src + j]) << np.uint32(16)

        _UC_NB = _k
    return _UC_NB or None


def unpack_c_plan_numba(raw, plan, gemm):
    """unpack_c_plan through the numba kernel (falls back when numba is absent)."""
    k = _unpack_c_kernel()
    m = int(plan["m"])
    g = plan["gemms"][gemm]
    n, ntiles = int(g["n"]), int(g["ntiles"])
    cols, cpc = int(plan["cols"]), int(plan["cpc"])
    Mblk, N = int(plan["Mblk"]), int(g["N"])
    if k is None:
        return unpack_c_plan_fast(raw, plan, gemm)
    if isinstance(raw, np.ndarray):
        buf = np.ascontiguousarray(raw, np.uint8).reshape(-1)
    elif isinstance(raw, (bytes, bytearray, memoryview)):
        # raw IS already a contiguous B buffer (ffn32_run reads the BO with
        # bytes(...)).  The old bytes(memoryview(raw).cast("B")) copied it
        # AGAIN -- 31.5 MB per dispatch, 63 MB per layer, 1.9 GB per step --
        # for nothing: the kernel only reads it.
        buf = np.frombuffer(raw, np.uint8)
    else:
        buf = np.frombuffer(bytes(memoryview(raw).cast("B")), np.uint8)
    u16 = buf.view(np.uint16)
    out = np.zeros(Mblk * N, np.uint32)
    tm = np.ascontiguousarray(np.asarray(plan["tile_map"], np.int64))
    k(u16, out, tm, cols, cpc, ntiles, m, n, N, m // 8, n // 8)
    return out.view(np.float32).reshape(Mblk, N)


#: T22_PLAN_FAST_DRAIN=0 restores the gather drain (A/B and fallback);
#: =2 forces the numpy bulk transpose; =3 forces the numba kernel
FAST_DRAIN = int(os.environ.get("T22_PLAN_FAST_DRAIN", "1") or 1)


def drain_c(raw, plan, gemm):
    """The C drain the executors use (all three agree bit for bit)."""
    if FAST_DRAIN == 0:
        return unpack_c_plan(raw, plan, gemm)
    if FAST_DRAIN == 2:
        return unpack_c_plan_fast(raw, plan, gemm)
    return unpack_c_plan_numba(raw, plan, gemm)


_PBF = None
_VB_K = None
_SLOTS = {}          # (N, K, tile_rows) -> the group -> slot gather index


def _vendor_gather_kernel():
    """numba gather for vendor_blocks -- a PURE PERMUTATION, byte-identical.

    Returns None when numba is absent (the product's py310 has it; the gates'
    py312 does not), in which case vendor_blocks uses the numpy fancy index.
    """
    global _VB_K
    if _VB_K is None:
        try:
            from numba import njit, prange
        except Exception:                                    # noqa: BLE001
            _VB_K = False
            return None

        @njit(parallel=True, cache=True, nogil=True, boundscheck=False,
              fastmath=False)
        def _k(src, dst, slots, ng):
            for q in prange(ng):
                s = slots[q]
                b0 = q * 9
                s0 = s * 9
                for b in range(9):
                    dst[b0 + b] = src[s0 + b]

        _VB_K = _k
    return _VB_K or None


def _pack_bfp16():
    """pack_bfp16 (repo root): the ONE authority on the vendor weight image."""
    global _PBF
    if _PBF is None:
        import sys as _sys
        root = str(repo_root())
        if root not in _sys.path:
            _sys.path.insert(0, root)
        import pack_bfp16 as _P
        _PBF = _P
    return _PBF


def vendor_blocks(blob, N, K, tile_rows=None):
    """npz weight blob -> (N, K//8, 9) BFP16 blocks in the MODEL order.

    NOT A RESHAPE.  The .npz does not store the groups row-major: pack_bfp16
    section 3 (verified byte for byte over 4.9 M groups against a vendor .rai)
    lays the 8-value groups out in 32-row x 32-K TILES,

        tile         = (row // 32) * (K // 32) + (kelem // 32)
        slot_in_tile = (r // 8) * 32 + (c // 8) * 8 + (r % 8)
        slot         = tile * 128 + slot_in_tile

    and pack() SCATTERS unit(g) to reg[slot(g)], so the inverse is the GATHER
    unit(g) = reg[slot(g)] (pack_bfp16.unpack:258-261).  Reshaping the blob
    row-major instead returns A DIFFERENT MATRIX -- the same 9-byte groups
    wired to the wrong (row, k) slots -- which is why every multiset gate in the
    tree ("99.754 % identical", check_payload_identity.py) passed while the plan
    path computed another function.

    MEASURED (layers.0, _xn bit-identical, 512^2, on the product device):
        row-major reshape : y  2672, corr vs the product y 0.0017
        this gather       : y 98304, corr vs the product y 0.9999367

    A PURE PERMUTATION of the file 9-byte blocks: no decode, no re-encode, so
    the model quantisation is preserved exactly.
    """
    P = _pack_bfp16()
    blob = np.ascontiguousarray(blob, np.uint8).reshape(-1)
    ng = int(N) * int(K) // P.GROUP
    if blob.size != ng * 9:
        raise ValueError("vendor_blocks: %d B for (%d,%d) -- expected %d"
                         % (blob.size, N, K, ng * 9))
    if tile_rows is None:
        tile_rows = P.KNOWN_TILE_ROWS.get((int(N), int(K)), 32)
    # the index itself is 39 MB of int64 arithmetic -- once per shape, not once
    # per layer (30 of them per step)
    sk = (int(N), int(K), int(tile_rows))
    slots = _SLOTS.get(sk)
    if slots is None:
        slots = np.ascontiguousarray(P._slots(int(N), int(K), ng, tile_rows),
                                     np.int64)
        _SLOTS[sk] = slots
    k = _vendor_gather_kernel()
    if k is None:
        units = blob.reshape(ng, 9)[slots]
    else:
        # 44 MB of gathers: the numpy fancy index measured 0.75 s/layer (30 of
        # them per step), the kernel 0.06 s.  A pure byte-for-byte permutation
        # -- 9 contiguous bytes per group, no arithmetic -- so the bytes are the
        # numpy path's by construction (gate_plan_weights.py asserts it).
        dst = np.empty(ng * 9, np.uint8)
        k(blob, dst, slots, ng)
        units = dst.reshape(ng, 9)
    return np.ascontiguousarray(units.reshape(int(N), int(K) // P.GROUP, 9))


def load_blocks(key):
    """The model BFP16 blocks for one layer: {w1,w3,w2} (N, K//8, 9).

    weights/block<N>.npz stores each matrix as the vendor bfp16ebs8 image, so
    the blocks come out of vendor_blocks() -- a tile-aware GATHER, never a plain
    reshape.  See that function for the measurement that proved it.

    T22_LOAD_BLOCKS_ROWMAJOR=1 restores the old row-major reshape.  It is NOT a
    fallback: the resulting blocks are a DIFFERENT MATRIX (corr 0.0017 with the
    product); it exists only to reproduce the defect in an A/B.
    """
    li = int(str(key).split(".")[1])
    p = repo_root() / "weights" / ("block%d.npz" % li)
    z = np.load(p, allow_pickle=False)
    rowmajor = os.environ.get("T22_LOAD_BLOCKS_ROWMAJOR", "0") == "1"
    out = {}
    for nm, shp in (("w1", (FF, D)), ("w3", (FF, D)), ("w2", (D, FF))):
        b = z["%s.feed_forward.%s.weight" % (key, nm)]
        if rowmajor:
            out[nm] = np.ascontiguousarray(b.reshape(shp[0], shp[1] // 8, 9))
        else:
            out[nm] = vendor_blocks(b, shp[0], shp[1])
    return out


def pack_payloads(blocks, plan):
    """(w1,w3,w2) blocks -> the B payloads this plan's images declare.

    A PURE plan declares three: gate, up, dn.  A FUSED plan declares TWO -- the
    gu image reads one [gate|up] pair object per (n-tile, k-step) -- so the
    tuple stays 3 long with b_gate IS b_up, the pair (one array, not a copy):
    every consumer of the 3-tuple contract keeps working, and the pair reaches
    the gu W BO once (see _ffn32_run_fused).
    """
    gu, dn, kk = plan["gemms"]["gu"], plan["gemms"]["dn"], int(plan["kk"])
    PK = pack_kk()
    if plan.get("fuse_gu"):
        pair = PK.pair_b(PK.pack_b_kk(blocks["w1"], gu["K"], gu["N"], gu["n"], kk),
                         PK.pack_b_kk(blocks["w3"], gu["K"], gu["N"], gu["n"], kk),
                         gu["K"], gu["N"], gu["n"], kk)
        return (pair, pair,
                PK.pack_b_kk(blocks["w2"], dn["K"], dn["N"], dn["n"], kk))
    return (PK.pack_b_kk(blocks["w1"], gu["K"], gu["N"], gu["n"], kk),
            PK.pack_b_kk(blocks["w3"], gu["K"], gu["N"], gu["n"], kk),
            PK.pack_b_kk(blocks["w2"], dn["K"], dn["N"], dn["n"], kk))


def payloads_for(key, plan, blocks=None):
    """{key: (b_gate, b_up, b_dn)} for one layer (no cache -- see plan_ffn.py)."""
    return pack_payloads(blocks if blocks is not None else load_blocks(key), plan)


def _alt_bo(dev, tag, slot, nbytes, name=None):
    """A second BO for (image, arg group) so a write never races a read.

    slot 0 is the product's own BO (dev.bo["b_<tag>_w"|"_o"]); 4 = W, 5 = C.
    Same pattern as devkill_ffn._alt_bo, which is device-proven.

    'name' is a LABEL only (pyxrt BOs carry none): the gate's fake device
    reports the name of the BO it is told to read, so the dispatch record can be
    compared against the plan's without a side table.  A real pyxrt BO rejects
    the attribute, which is fine.
    """
    if slot == 0:
        return None
    k = (id(dev), tag, slot)
    hit = _ALT_BO.get(k)
    if hit is None or hit[0] is not dev:
        im = dev.imgs[tag]
        bo = dev.pyxrt.bo(dev.dev, int(nbytes),
                          dev.pyxrt.bo.flags.host_only,
                          im["k"].group_id(slot))
        if name:
            try:
                bo.name = name
            except Exception:                                # noqa: BLE001
                pass
        _ALT_BO[k] = (dev, bo)
    return _ALT_BO[k][1]


def _require(dev, plan, tags):
    """Refuse loudly when the plan's images are not registered on this device."""
    miss = []
    for gemm in ("gu", "dn"):
        t = tags[gemm]
        if t not in getattr(dev, "imgs", {}):
            miss.append("imgs[%r]" % t)
        for s in ("x", "w", "o"):
            if "b_%s_%s" % (t, s) not in getattr(dev, "bo", {}):
                miss.append("bo[b_%s_%s]" % (t, s))
    if miss:
        raise RuntimeError(
            "ffn32_run: the plan's image set is not registered on this device "
            "(%s missing).  Build it with\n"
            "    py312 imaggers/util/place/npu_synth.py --m %d\n"
            "and register it with plan_ffn.register(%d) before Dev().  plan=%s"
            % (", ".join(miss[:6]), int(plan["M"]), int(plan["M"]),
               plan["plan_key"]))


def _bf16_sigmoid_silu(g):
    """silu(g) = g / (1 + exp(-g)) in f32, saturating like the device."""
    with np.errstate(over="ignore"):
        return g * (1.0 / (1.0 + np.exp(-g)))


# ---------------------------------------------------------------------------
# THE HOST SWIGLU -- the plan arm's single largest host constant
# ---------------------------------------------------------------------------
# The plan's gu image is a PURE GEMM, so silu(gate)*up runs on the host over
# (Mblk, N) = (1056, 10240) f32 = 10.8 M elements, once per layer.  The
# expression the correctness proof was taken on is
#
#     h = _bf16(_bf16_sigmoid_silu(gate) * up)
#
# which materialises THIRTEEN full-size temporaries (43.3 MB each): -g, exp,
# 1+, 1/, g*, *up, then _bf16's (u>>16), &1, +0x7FFF, +lsb, >>16, astype, <<16.
# It measured 133.7 ms/layer here (probe_swiglu.py) = 4.0 s/step at 30 layers,
# i.e. 14 % of the plan arm's whole step wall -- the largest single item after
# the B payload repack.
#
# swiglu_host() performs THE SAME OPERATIONS IN THE SAME ORDER on two reusable
# scratch buffers: numpy's own exp ufunc (bit-identical -- it IS the ufunc the
# shipped line calls) writes in place, and one numba pass finishes silu*up and
# the RNE bf16 rounding in a single sweep.
#
#   shipped numpy expression      133.7 ms   13 temporaries
#   numpy in-place, no numba       45.7 ms   exact
#   numpy exp + this numba tail    31.0 ms   exact   <-- used
#   all-numba (exp inside)          7.1 ms   16 of 10.8 M elements differ
#                                             by 1 bf16 ulp: numba's exp is not
#                                             numpy's, and 1 ulp of f32 flips a
#                                             bf16 rounding on ~1.5e-5 of the
#                                             elements.  NOT used -- this path
#                                             must stay bit-identical to the
#                                             expression DET-BISECT-REPORT-2
#                                             section 6 was proved on.
# All numbers: py310 + numba 0.62.1, 1056x10240 f32, median of 5
# (imaggers/util/place/probe_swiglu.py).
_SW = {"k": None, "buf": {}}


def _swiglu_tail_kernel():
    """numba: exp(-gate) is already in e; finish silu*up and the bf16 round.

    Writes f32 into o32 and its bit pattern into ou -- the caller passes the
    SAME buffer as both, so the rounding is the exact integer sequence
    (u + 0x7FFF + ((u >> 16) & 1)) >> 16 << 16 that _bf16 performs.
    """
    if _SW["k"] is None:
        try:
            from numba import njit, prange
        except Exception:                                     # noqa: BLE001
            _SW["k"] = False
            return None

        @njit(parallel=True, cache=True, nogil=True, fastmath=False)
        def _k(g, u, e, o32, ou, n):
            for i in prange(n):
                s = np.float32(1.0) / (np.float32(1.0) + e[i])
                o32[i] = (g[i] * s) * u[i]
                b = ou[i]
                ou[i] = ((b + np.uint32(0x7FFF)
                          + ((b >> np.uint32(16)) & np.uint32(1)))
                         >> np.uint32(16)) << np.uint32(16)

        _SW["k"] = _k
    return _SW["k"] or None


def _swiglu_bufs(shape):
    """Two reusable (shape,) f32 scratch buffers, one pair per shape."""
    b = _SW["buf"].get(shape)
    if b is None:
        b = (np.empty(shape, np.float32), np.empty(shape, np.float32))
        _SW["buf"][shape] = b
    return b


def swiglu_host(gate, up):
    """bf16(silu(gate) * up) -- the SAME BYTES as _bf16(_bf16_sigmoid_silu(g)*u).

    Same operands, same order, same rounding; only the number of full-size
    temporaries changes (13 -> 2, both reused across layers).  Falls back to the
    shipped expression when numba is absent, so the result is never in doubt.
    """
    g = np.ascontiguousarray(gate, np.float32)
    u = np.ascontiguousarray(up, np.float32)
    k = _swiglu_tail_kernel()
    if k is None:
        return _bf16(_bf16_sigmoid_silu(g) * u)
    s, h = _swiglu_bufs(g.shape)
    np.negative(g, out=s)
    with np.errstate(over="ignore"):
        np.exp(s, out=s)              # numpy's ufunc, in place: bit-identical
    k(g.reshape(-1), u.reshape(-1), s.reshape(-1), h.reshape(-1),
      h.view(np.uint32).reshape(-1), g.size)
    return h


# ---------------------------------------------------------------------------
# THE A-IMAGE CONTRACT -- the one place that decides what the FFN's xn IS
# ---------------------------------------------------------------------------
# The FFN input arrives in ONE of TWO forms, and they are NOT interchangeable:
#
#     float32   the normalised activation                          (hooks OFF)
#     uint16    a BF16 IMAGE: the same values as a bit pattern,    (hooks ON:
#               2 B each                                           install.py
#                                                                  h3-ffn-norm1-bf16
#                                                                  -> fast_glue
#                                                                  .norm_bf16)
#
# np.asarray(u16, np.float32) is a NUMERIC CAST, not a reinterpretation: bf16
# 1.0 is the bit pattern 0x3F80, and that call turns it into the integer
# 16256.0.  MEASURED (T22-PLAN-FUSE-GU-ARM.md 6, house recipe, 512^2 x1): with
# T22_PLAN_FFN=1 the plan arm read its input through that cast, computed a
# different FFN for 30 layers, produced decorrelated latents (e5f2bd390ab85aed
# vs 1e1b8420bb9beda3 with the hooks off; corr -0.013689961, maxabs 6.22) and
# the job still reported 'done'.
#
# The decode below is a BIT VIEW and it is the expression the DELIVERED arm's
# devkill_ffn already uses for the device's own bf16 drains (devkill_ffn.py:404
# "(_u16.astype(np.uint32) << np.uint32(16)).view(np.float32)"; decode_fast's
# "np.left_shift(b, 16, out=out.view(np.uint32))"), so the plan arm and the
# delivered arm decode a bf16 image identically by construction.
def bf16_image_f32(img):
    """uint16 bf16 image -> float32 by BIT REINTERPRETATION, never a cast.

    The same bytes devkill_ffn.py:404 produces for the delivered arm's bf16
    drains: shift the 16-bit pattern into the high half of a float32 and view
    it.  Exact, lossless, and it round-trips through _bf16() unchanged.
    """
    return (np.ascontiguousarray(img).astype(np.uint32)
            << np.uint32(16)).view(np.float32)


def _absmax_f32(a):
    """absmax of any numeric buffer, for a REFUSAL message (never raises)."""
    try:
        return "%.6g" % float(np.abs(a.astype(np.float32)).max())
    except Exception:                                            # noqa: BLE001
        return "?"


def as_a_f32(xn, who):
    """The FFN's A input as float32, or a LOUD refusal.

    float32 is returned AS-IS (no copy: the executor must read the caller's
    buffer, which the DET-BISECT probe asserts); the uint16 bf16 image is
    decoded by bit view.  Every other dtype RAISES with the dtype and the sha.

    The refusal is the point.  A numeric cast produces a plausible array, so a
    dtype mistake used to decorrelate the whole FFN and still report 'done'
    (see the block comment above); there is no downstream assertion that can
    see it.  A loud TypeError at the boundary is the only gate that cannot be
    mistaken for a result.
    """
    a = np.asarray(xn)
    if a.dtype == np.float32:
        return a
    if a.dtype == np.uint16:
        return bf16_image_f32(a)
    raise TypeError(
        "%s: xn has dtype %s, which this executor does not cast.  It takes "
        "float32 (the normalised activation) or uint16 (a bf16 A image from "
        "fast_glue.norm_bf16, decoded by BIT VIEW).  Got shape=%r bytes=%d "
        "absmax=%s sha16=%s.  Refusing: a numeric uint16->float32 cast reads "
        "bf16 bit patterns as integers (0x3F80 -> 16256.0), which decorrelates "
        "every layer of the FFN and still reports 'done' -- "
        "T22-PLAN-FUSE-GU-ARM.md section 6."
        % (who, a.dtype, tuple(a.shape), int(a.nbytes), _absmax_f32(a),
           _sha16(a)))


def ffn32_run(dev, xn, key, tim=None, b_cache=None, plan=None, blocks=None,
              stats=None, record=None):
    """xn (rows, 3840) float32 or uint16 -> y (rows, 3840) float32.  THE FFN SEGMENT.

    The product contract (t18_run.block_step's BIGM branch): y is the FFN
    output BEFORE ffn_norm2, row-local, so a padded row changes no real row.

    dev     the product's Dev (prod_loop_bigm / t18_run): .imgs[name] =
            {"k","bo_i","inst"}, .bo["b_<tag>_x"|"_w"|"_o"], .pyxrt, .dev,
            and optionally .barrier(name).
    xn      the A image: float32 (the normalised activation) or the uint16
            BF16 IMAGE the fastglue hooks emit (install.py h3-ffn-norm1-bf16),
            decoded by BIT VIEW.  Any other dtype RAISES -- see as_a_f32.
    key     "layers.<N>" -- the layer, for the payload and for the record.
    b_cache {key: (b_gate, b_up, b_dn)} or the tuple itself.  None -> build it
            from weights/block<N>.npz (0.2 s/layer, no cache).
    plan    the plan dict (default: npu_plan.plan(rows)); must be the plan the
            registered images were built for.
    stats   optional dict, filled with the section columns in ms (the same
            names the delivered [devkill] line prints, plus "swiglu").
    record  optional list; when given, one {"gemm","layer","tag","out_bo",
            "w_bo","wsha","w_bytes"} entry per dispatch is appended -- the
            (BO, W-BO, sha-of-W) record gate_ffn32.py asserts on.  Off by
            default because the shas cost 0.26 s/layer.
    tim     the product's timing dict; only tim["dev"] is added to (the device
            wall), exactly as devkill_ffn.block does.
    """
    t_all = time.perf_counter()
    # THE A-IMAGE CONTRACT.  This line used to be np.asarray(xn, np.float32),
    # a NUMERIC cast: handed the fastglue hooks' uint16 bf16 image it read
    # 0x3F80 (bf16 1.0) as 16256.0, decorrelated the whole FFN and still
    # reported 'done' (T22-PLAN-FUSE-GU-ARM.md 6).  as_a_f32 decodes that
    # image by BIT VIEW in devkill_ffn's own expression and REFUSES every
    # other dtype with the dtype and the sha, so the mistake cannot be
    # silent again.
    x = as_a_f32(xn, "ffn32_run")
    if x.ndim != 2 or x.shape[1] != D:
        raise ValueError("ffn32_run: xn must be (rows, %d), got %r"
                         % (D, x.shape))
    # T22_PLAN_FFN_PROBE=1: the DET-BISECT test 2 at the EXECUTOR entry.  It
    # records the identity of the buffer ffn32_run actually reads (a REFERENCE,
    # not a copy: np.asarray keeps the caller's memory) so the call site's hash
    # and this one can be compared byte for byte.  OFF by default: no hashing,
    # no prints, not a product path.
    _probe = (os.environ.get("T22_PLAN_FFN_PROBE", "0") == "1"
              and key == "layers.0" and key not in _PROBED)
    if _probe:
        _PROBED.add(key)
        print("[pffnprobe] exec-entry key=%s rows=%d xn_dtype=%s dtype=%s "
              "id(x)=%d shares_mem_with_xn=%s own_data=%s absmax=%.6g sha=%s"
              % (key, int(x.shape[0]), np.asarray(xn).dtype, x.dtype, id(x),
                 bool(np.shares_memory(x, xn)),
                 bool(x.base is None or not np.shares_memory(x, xn)),
                 float(np.abs(x).max()), _sha16(x)), flush=True)
    seq = int(x.shape[0])
    pl = plan if plan is not None else plan_for(seq)
    if int(pl["Mblk"]) * int(pl["ndisp"]) < seq:
        # a plan for a smaller M would leave rows un-computed (and y is zeroed)
        raise ValueError(
            "ffn32_run: plan %s covers %d rows (%d x %d) but xn has %d"
            % (pl["plan_key"], int(pl["Mblk"]) * int(pl["ndisp"]),
               int(pl["Mblk"]), int(pl["ndisp"]), seq))
    if pl.get("fuse_gu"):
        # the fused gu: ONE dispatch for gate+up+silu, no host elementwise pass
        return _ffn32_run_fused(dev, x, key, pl, tim=tim, b_cache=b_cache,
                                stats=stats, record=record, blocks=blocks)
    if pl["strategy"] != "xd":
        raise NotImplementedError(
            "ffn32_run: strategy %r is not wired (only 'xd' is); the t33 form "
            "needs per-column A elements and a drain await per n-tile"
            % (pl["strategy"],))
    tags = tags_for(pl)
    _require(dev, pl, tags)
    m, kk = int(pl["m"]), int(pl["kk"])
    cols, cpc = int(pl["cols"]), int(pl["cpc"])
    Mblk = int(pl["Mblk"])
    g_gu, g_dn = pl["gemms"]["gu"], pl["gemms"]["dn"]
    t_gu, t_dn = tags["gu"], tags["dn"]
    PK = pack_kk()

    if b_cache is None:
        b_cache = payloads_for(key, pl, blocks=blocks)
    elif isinstance(b_cache, dict):
        b_cache = b_cache[key]
    b_gate, b_up, b_dn = (np.asarray(b, np.uint8) for b in b_cache)
    for nm, b, g in (("gate", b_gate, g_gu), ("up", b_up, g_gu),
                     ("dn", b_dn, g_dn)):
        if b.size != int(g["b_bytes"]):
            raise ValueError("ffn32_run: %s payload is %d B, the plan's %s image "
                             "declares %d B" % (nm, b.size, nm, g["b_bytes"]))

    # ---- BOs: the product's three + the up-projection's own W and C --------
    bo_x = dev.bo["b_%s_x" % t_gu]
    bo_w = dev.bo["b_%s_w" % t_gu]
    bo_o = dev.bo["b_%s_o" % t_gu]
    bo_wu = _alt_bo(dev, t_gu, 4, g_gu["b_bytes"], "b_%s_w+up" % t_gu)
    bo_ou = _alt_bo(dev, t_gu, 5, g_gu["c_bytes"], "b_%s_o+up" % t_gu)
    bo_dx = dev.bo["b_%s_x" % t_dn]
    bo_dw = dev.bo["b_%s_w" % t_dn]
    bo_do = dev.bo["b_%s_o" % t_dn]
    dto = dev.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    im_gu, im_dn = dev.imgs[t_gu], dev.imgs[t_dn]
    n_gu, n_dn = len(im_gu["inst"]) // 4, len(im_dn["inst"]) // 4

    st = {"rows": seq, "Mblk": Mblk, "ndisp": int(pl["ndisp"]),
          "plan": pl["plan_key"], "pack": 0.0, "wgu": 0.0, "gu": 0.0,
          "swiglu": 0.0, "wdn": 0.0, "dn": 0.0, "read": 0.0, "dec": 0.0,
          "barrier": 0.0, "states": [], "n_disp": 0}
    dev_ms = 0.0
    y = np.zeros((seq, D), np.float32)

    def _sync_write(bo, payload):
        bo.write(payload, 0)
        bo.sync(dto)

    def _wait(run, tag):
        state = int(run.wait(300000))
        st["states"].append(state)
        st["n_disp"] += 1
        if state != 4:
            raise RuntimeError("ffn32_run %s %s: ERT state %d"
                               % (key, tag, state))

    def _rec(gemm, tag, wname, payload, oname):
        """The dispatch record the gate asserts on: (out BO, W BO, sha of W).

        Only built when the caller asks for it (record=[]): sha256 over the
        44 MB payloads is 0.26 s/layer, which the product must not pay.
        """
        if record is None:
            return
        record.append({"gemm": gemm, "layer": key, "tag": tag,
                       "out_bo": oname, "w_bo": wname,
                       "wsha": _sha16(payload),
                       "w_bytes": int(len(payload))})

    for d, (start, rows) in enumerate(pl["dispatches"]):
        if start >= seq:
            break
        rows = min(int(rows), seq - start)

        # ---- A (gu): the FFN input, zero-padded to Mblk rows --------------
        t0 = time.perf_counter()
        Ad = np.zeros((Mblk, int(g_gu["K"])), np.float32)
        Ad[:rows] = x[start:start + rows]
        a_gu = PK.pack_a_kk(Ad, m, kk, cols, cpc)
        del Ad
        if a_gu.size != int(g_gu["a_bytes"]):
            raise ValueError("ffn32_run: A(gu) is %d B, the plan declares %d B"
                             % (a_gu.size, g_gu["a_bytes"]))
        st["pack"] += 1e3 * (time.perf_counter() - t0)
        if _probe:
            # what the device is handed: the A payload AND the rows it came from
            print("[pffnprobe] disp=%d start=%d rows=%d Mblk=%d a_gu %s "
                  "xrows_absmax=%.6g a_gu_absmax_u16=%d"
                  % (d, start, rows, Mblk, _sha16(a_gu),
                     float(np.abs(x[start:start + rows]).max()),
                     int(a_gu.view(np.uint16).max())), flush=True)

        # ---- the two gu W payloads, then gate and up both in flight -------
        t0 = time.perf_counter()
        _sync_write(bo_x, a_gu)
        _sync_write(bo_w, b_gate)
        _sync_write(bo_wu, b_up)
        st["wgu"] += 1e3 * (time.perf_counter() - t0)
        t0 = time.perf_counter()
        run_g = im_gu["k"](3, im_gu["bo_i"], n_gu, bo_x, bo_w, bo_o)
        t_b = time.perf_counter()
        _barrier(dev)
        t_bar = 1e3 * (time.perf_counter() - t_b)
        st["barrier"] += t_bar
        run_u = im_gu["k"](3, im_gu["bo_i"], n_gu, bo_x, bo_wu, bo_ou)
        _wait(run_g, "gu(gate)")
        _wait(run_u, "gu(up)")
        st["gu"] += 1e3 * (time.perf_counter() - t0) - t_bar
        dev_ms += st["gu"]
        _rec("gu", t_gu, "b_%s_w" % t_gu, b_gate, "b_%s_o" % t_gu)
        _rec("gu", t_gu, "b_%s_w+up" % t_gu, b_up, "b_%s_o+up" % t_gu)

        # ---- drain both C tiles, silu(gate) * up --------------------------
        t0 = time.perf_counter()
        # BUGFIX (2026-11-03, DET): every C read here and below was made with NO
        # XCL_BO_SYNC_BO_FROM_DEVICE.  t18_run.Dev.dispatch syncs every out_key
        # before reading (that is why the delivered path is bit-reproducible);
        # this executor bypasses it, so the drain could see a C BO whose device
        # DMA had not landed -- a FEW elements stale, run to run, at ANY M.
        # That is the plan path's nondeterminism, and it is NOT M-dependent.
        dfrom = dev.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        bo_o.sync(dfrom)
        bo_ou.sync(dfrom)
        raw_g = bytes(bo_o.read(int(g_gu["c_bytes"]), 0))
        raw_u = bytes(bo_ou.read(int(g_gu["c_bytes"]), 0))
        st["read"] += 1e3 * (time.perf_counter() - t0)
        t0 = time.perf_counter()
        gate = drain_c(raw_g, pl, "gu")
        up = drain_c(raw_u, pl, "gu")
        del raw_g, raw_u
        st["dec"] += 1e3 * (time.perf_counter() - t0)
        t0 = time.perf_counter()
        if _probe:
            print("[pffnprobe] disp=%d gate absmax=%.6g mean=%.6g sha=%s | "
                  "up absmax=%.6g mean=%.6g sha=%s"
                  % (d, float(np.abs(gate).max()), float(gate.mean()),
                     _sha16(gate), float(np.abs(up).max()),
                     float(up.mean()), _sha16(up)), flush=True)
            global _PROBE_YREF
            _PROBE_YREF = _probe_reference(x[start:start + rows], key, pl,
                                           gate, up)
        # SAME BYTES as _bf16(_bf16_sigmoid_silu(gate) * up), 4.3x faster:
        # see swiglu_host() for the measured variants and why the all-numba
        # one (7.1 ms, 4.3x again) is NOT used.
        h = swiglu_host(gate, up)
        del gate, up
        st["swiglu"] += 1e3 * (time.perf_counter() - t0)
        if _probe:
            print("[pffnprobe] disp=%d h absmax=%.6g mean=%.6g"
                  % (d, float(np.abs(h).max()), float(h.mean())), flush=True)

        # ---- A (dn): the swiglu output, packed at the plan's kk -----------
        t0 = time.perf_counter()
        a_dn = PK.pack_a_kk(h, m, kk, cols, cpc)
        del h
        if a_dn.size != int(g_dn["a_bytes"]):
            raise ValueError("ffn32_run: A(dn) is %d B, the plan declares %d B"
                             % (a_dn.size, g_dn["a_bytes"]))
        st["pack"] += 1e3 * (time.perf_counter() - t0)

        # ---- dn ----------------------------------------------------------
        t0 = time.perf_counter()
        _sync_write(bo_dx, a_dn)
        _sync_write(bo_dw, b_dn)
        st["wdn"] += 1e3 * (time.perf_counter() - t0)
        t0 = time.perf_counter()
        run_d = im_dn["k"](3, im_dn["bo_i"], n_dn, bo_dx, bo_dw, bo_do)
        _wait(run_d, "dn")
        st["dn"] += 1e3 * (time.perf_counter() - t0)
        dev_ms += st["dn"]
        _rec("dn", t_dn, "b_%s_w" % t_dn, b_dn, "b_%s_o" % t_dn)
        if _probe:
            print("[pffnprobe] disp=%d dn-launch w=%s a_dn=%s"
                  % (d, _sha16(b_dn), _sha16(a_dn)), flush=True)

        t0 = time.perf_counter()
        # BUGFIX (2026-11-03, DET): no FROM_DEVICE sync before this C read
        bo_do.sync(dev.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        raw_d = bytes(bo_do.read(int(g_dn["c_bytes"]), 0))
        st["read"] += 1e3 * (time.perf_counter() - t0)
        t0 = time.perf_counter()
        yd = drain_c(raw_d, pl, "dn")
        st["dec"] += 1e3 * (time.perf_counter() - t0)
        y[start:start + rows] = yd[:rows]
        if _probe:
            print("[pffnprobe] disp=%d yd absmax=%.6g y_absmax=%.6g y_sha=%s "
                  "states=%r" % (d, float(np.abs(yd).max()),
                                 float(np.abs(y[start:start + rows]).max()),
                                 _sha16(y[start:start + rows]),
                                 st["states"]), flush=True)
            if _PROBE_YREF is not None:
                _probe_cmp("y", y[start:start + rows], _PROBE_YREF)

    st["total"] = 1e3 * (time.perf_counter() - t_all)
    st["device_ms"] = dev_ms
    if stats is not None:
        stats.update(st)
    if tim is not None:
        tim["dev"] = tim.get("dev", 0.0) + dev_ms
    return y


def _ffn32_run_fused(dev, x, key, pl, tim=None, b_cache=None, stats=None,
                     record=None, blocks=None):
    """THE FUSED FFN SEGMENT -- ONE gu dispatch does gate, up AND silu(gate)*up.

    Same contract as ffn32_run (y is the FFN output BEFORE ffn_norm2, row
    local, so a padded row changes no real row); only the dispatch list is
    shorter:

        A(gu) pack -> write A(gu), write the [gate|up] PAIR
        launch the fused gu ONCE; wait        (no barrier: not a same-image
                                               relaunch, this is one image)
        read C -> unpack_c_plan -> h (Mblk, N): the silu product, made on device
        A(dn) pack -> write; launch dn; wait; read; unpack -> y[start:...]

    What it removes is exactly what the plan path was paying for: the host
    elementwise pass (10.8 M exp over 43 MB at 512^2), the second gu C tile's
    read+decode, the second A pack (gate and up share one), and the
    same-image barrier.  What it adds is that the gu dispatch does 2x the
    FLOPs -- the same math, in one launch.
    """
    t_all = time.perf_counter()
    if pl["strategy"] != "xd":
        raise NotImplementedError(
            "ffn32_run(fused): strategy %r has no fused image (only xd does)"
            % (pl["strategy"],))
    tags = tags_for(pl)
    _require(dev, pl, tags)
    m, kk = int(pl["m"]), int(pl["kk"])
    cols, cpc = int(pl["cols"]), int(pl["cpc"])
    Mblk = int(pl["Mblk"])
    g_gu, g_dn = pl["gemms"]["gu"], pl["gemms"]["dn"]
    t_gu, t_dn = tags["gu"], tags["dn"]
    PK = pack_kk()

    if b_cache is None:
        b_cache = payloads_for(key, pl, blocks=blocks)
    elif isinstance(b_cache, dict):
        b_cache = b_cache[key]
    b_pair, _b_up, b_dn = (np.asarray(b, np.uint8) for b in b_cache)
    for nm, b, g in (("pair", b_pair, g_gu), ("dn", b_dn, g_dn)):
        if b.size != int(g["b_bytes"]):
            raise ValueError("ffn32_run(fused): %s payload is %d B, the plan's "
                             "image declares %d B"
                             % (nm, b.size, g["b_bytes"]))

    bo_x = dev.bo["b_%s_x" % t_gu]
    bo_w = dev.bo["b_%s_w" % t_gu]
    bo_o = dev.bo["b_%s_o" % t_gu]
    bo_dx = dev.bo["b_%s_x" % t_dn]
    bo_dw = dev.bo["b_%s_w" % t_dn]
    bo_do = dev.bo["b_%s_o" % t_dn]
    dto = dev.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    im_gu, im_dn = dev.imgs[t_gu], dev.imgs[t_dn]
    n_gu, n_dn = len(im_gu["inst"]) // 4, len(im_dn["inst"]) // 4

    seq = int(x.shape[0])
    st = {"rows": seq, "Mblk": Mblk, "ndisp": int(pl["ndisp"]),
          "plan": pl["plan_key"], "pack": 0.0, "wgu": 0.0, "gu": 0.0,
          "swiglu": 0.0, "wdn": 0.0, "dn": 0.0, "read": 0.0, "dec": 0.0,
          "barrier": 0.0, "states": [], "n_disp": 0, "fused": 1}
    dev_ms = 0.0
    y = np.zeros((seq, D), np.float32)

    def _sync_write(bo, payload):
        bo.write(payload, 0)
        bo.sync(dto)

    def _wait(run, tag):
        state = int(run.wait(300000))
        st["states"].append(state)
        st["n_disp"] += 1
        if state != 4:
            raise RuntimeError("ffn32_run(fused) %s %s: ERT state %d"
                               % (key, tag, state))

    def _rec(gemm, tag, payload):
        if record is None:
            return
        record.append({"gemm": gemm, "layer": key, "tag": tag,
                       "out_bo": "b_%s_o" % tag, "w_bo": "b_%s_w" % tag,
                       "wsha": _sha16(payload),
                       "w_bytes": int(len(payload))})

    for d, (start, rows) in enumerate(pl["dispatches"]):
        if start >= seq:
            break
        rows = min(int(rows), seq - start)

        # ---- A (gu): the FFN input, zero-padded to Mblk rows --------------
        t0 = time.perf_counter()
        Ad = np.zeros((Mblk, int(g_gu["K"])), np.float32)
        Ad[:rows] = x[start:start + rows]
        a_gu = PK.pack_a_kk(Ad, m, kk, cols, cpc)
        del Ad
        if a_gu.size != int(g_gu["a_bytes"]):
            raise ValueError("ffn32_run(fused): A(gu) is %d B, the plan "
                             "declares %d B" % (a_gu.size, g_gu["a_bytes"]))
        st["pack"] += 1e3 * (time.perf_counter() - t0)

        # ---- ONE fused dispatch: gate, up and the silu product ------------
        t0 = time.perf_counter()
        _sync_write(bo_x, a_gu)
        _sync_write(bo_w, b_pair)
        st["wgu"] += 1e3 * (time.perf_counter() - t0)
        t0 = time.perf_counter()
        run_g = im_gu["k"](3, im_gu["bo_i"], n_gu, bo_x, bo_w, bo_o)
        _wait(run_g, "gu(fused)")
        st["gu"] += 1e3 * (time.perf_counter() - t0)
        dev_ms += st["gu"]
        _rec("gu", t_gu, b_pair)

        # ---- drain h = silu(gate)*up: the image did it, no host pass ------
        t0 = time.perf_counter()
        # BUGFIX (2026-11-03, DET): this C read had no FROM_DEVICE sync either.
        bo_o.sync(dev.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        raw_g = bytes(bo_o.read(int(g_gu["c_bytes"]), 0))
        st["read"] += 1e3 * (time.perf_counter() - t0)
        t0 = time.perf_counter()
        h = drain_c(raw_g, pl, "gu")
        del raw_g
        st["dec"] += 1e3 * (time.perf_counter() - t0)

        # ---- A (dn): the fused h, packed at the plan's kk ------------------
        t0 = time.perf_counter()
        a_dn = PK.pack_a_kk(h, m, kk, cols, cpc)
        del h
        if a_dn.size != int(g_dn["a_bytes"]):
            raise ValueError("ffn32_run(fused): A(dn) is %d B, the plan "
                             "declares %d B" % (a_dn.size, g_dn["a_bytes"]))
        st["pack"] += 1e3 * (time.perf_counter() - t0)

        # ---- dn ----------------------------------------------------------
        t0 = time.perf_counter()
        _sync_write(bo_dx, a_dn)
        _sync_write(bo_dw, b_dn)
        st["wdn"] += 1e3 * (time.perf_counter() - t0)
        t0 = time.perf_counter()
        run_d = im_dn["k"](3, im_dn["bo_i"], n_dn, bo_dx, bo_dw, bo_do)
        _wait(run_d, "dn")
        st["dn"] += 1e3 * (time.perf_counter() - t0)
        dev_ms += st["dn"]
        _rec("dn", t_dn, b_dn)

        t0 = time.perf_counter()
        # BUGFIX (2026-11-03, DET): no FROM_DEVICE sync before this C read
        bo_do.sync(dev.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        raw_d = bytes(bo_do.read(int(g_dn["c_bytes"]), 0))
        st["read"] += 1e3 * (time.perf_counter() - t0)
        t0 = time.perf_counter()
        yd = drain_c(raw_d, pl, "dn")
        st["dec"] += 1e3 * (time.perf_counter() - t0)
        y[start:start + rows] = yd[:rows]

    st["total"] = 1e3 * (time.perf_counter() - t_all)
    st["device_ms"] = dev_ms
    if stats is not None:
        stats.update(st)
    if tim is not None:
        tim["dev"] = tim.get("dev", 0.0) + dev_ms
    return y


def _sha16(payload):
    """sha256 of a payload, first 16 hex chars -- the record's W identity."""
    import hashlib
    a = np.ascontiguousarray(payload)
    return hashlib.sha256(memoryview(a).cast("B")).hexdigest()[:16]


def _probe_cmp(nm, dev, ref):
    """Print the device C next to the host reference: magnitude AND correlation."""
    a = np.asarray(dev, np.float64).ravel()
    b = np.asarray(ref, np.float64).ravel()
    da, db = float(np.abs(a).max()), float(np.abs(b).max())
    corr = float(np.corrcoef(a, b)[0, 1]) if a.size > 1 else float("nan")
    print("[pffnprobe] REF %-4s dev_absmax=%.6g ref_absmax=%.6g ratio=%.4f "
          "corr=%.9f maxabs=%.6g"
          % (nm, da, db, (da / max(1e-30, db)), corr,
             float(np.abs(a - b).max())), flush=True)


def _probe_reference(x, key, pl, gate, up):
    """T22_PLAN_FFN_PROBE=1 (DET-BISECT): the host reference for the SAME x.

    Computed from the model's own BFP16 blocks through the same bf16 rounding
    the device applies (s2_ffn32.ffn32_reference), and printed NEXT TO the
    device's C so a magnitude defect is attributable to the device GEMM, to the
    host glue, or to neither.  Diagnostics only: never called unless the env
    flag is on, and it is slow (about 4 s of BLAS at 1056 rows).
    """
    t0 = time.perf_counter()
    blocks = load_blocks(key)
    xq = _bf16(np.asarray(x, np.float32))
    gu, dn = pl["gemms"]["gu"], pl["gemms"]["dn"]
    K, N = int(gu["K"]), int(gu["N"])

    w1 = _decode_blocks(blocks["w1"], K, N)
    g_ref = _bf16(xq @ w1)
    del w1
    w3 = _decode_blocks(blocks["w3"], K, N)
    u_ref = _bf16(xq @ w3)
    del w3, xq
    _probe_cmp("gate", gate, g_ref)
    _probe_cmp("up", up, u_ref)
    h_ref = _bf16(_bf16_sigmoid_silu(g_ref) * u_ref)
    del g_ref, u_ref
    w2 = _decode_blocks(blocks["w2"], int(dn["K"]), int(dn["N"]))
    y_ref = _bf16(h_ref @ w2)
    del w2
    print("[pffnprobe] REF h_absmax=%.6g y_ref_absmax=%.6g (%.1f s)"
          % (float(np.abs(h_ref).max()), float(np.abs(y_ref).max()),
             time.perf_counter() - t0), flush=True)
    return y_ref


def _barrier(dev):
    """Break the same-image relaunch (gate -> up) with one other launch.

    t18_run.Dev.barrier's measured law: two launches of the same xclbin
    back-to-back stall (state 8, seconds); ONE launch of a different xclbin
    between them removes it (~1.7-6.5 ms).  A second handle on the same xclbin
    is NOT a fix (it reports a false completion).
    """
    import os as _os
    if _os.environ.get("T22_PLAN_FFN_BARRIER", "1") == "0":
        return 0.0
    name = _os.environ.get("T22_PLAN_FFN_BARRIER_IMG", "att")
    fn = getattr(dev, "barrier", None)
    if fn is None:
        return 0.0
    try:
        return fn(name)
    except Exception as exc:                                    # noqa: BLE001
        raise RuntimeError(
            "ffn32_run: the gu image is dispatched twice per layer (gate, up) "
            "and a same-image relaunch stalls the ERT (t18_run.Dev.barrier).  "
            "The barrier launch of %r failed (%r); set T22_PLAN_FFN_BARRIER_IMG "
            "to a registered image of a different xclbin." % (name, exc))


def warm_b(keys, load_blocks, bench=False):
    """Repack the model's BFP16 blocks into the images' payloads, once.

    load_blocks(key) -> {"w1","w3","w2"} blocks (out, in//8, 9) uint8.
    Returns {key: (b_gate, b_up, b_dn)} and, with bench=True, the timings.
    """
    t0 = time.perf_counter()
    out, total = {}, 0
    for k in keys:
        w = load_blocks(k)
        b_gate = pack_b_from_blocks(w["w1"], 3840, 10240, 512)
        b_up = pack_b_from_blocks(w["w3"], 3840, 10240, 512)
        b_dn = pack_b_from_blocks(w["w2"], 10240, 3840, 384)
        out[k] = (b_gate, b_up, b_dn)
        total += b_gate.size + b_up.size + b_dn.size
    dt = time.perf_counter() - t0
    if bench:
        return out, {"keys": len(keys), "bytes": int(total), "seconds": dt,
                     "MB_per_s": total / 1e6 / dt if dt else None}
    return out


# ===========================================================================
# CLI
# ===========================================================================
def _load_model_layer(key="layers.0"):
    """The real weights: weights/blockN.npz, BFP16 blocks along the in-axis."""
    li = int(key.split(".")[1])
    p = HERE.parents[2] / "weights" / f"block{li}.npz"
    z = np.load(p, allow_pickle=False)
    out = {}
    for nm, shp in (("w1", (10240, 3840)), ("w3", (10240, 3840)),
                    ("w2", (3840, 10240))):
        b = z[f"{key}.feed_forward.{nm}.weight"]
        out[nm] = np.ascontiguousarray(b.reshape(shp[0], shp[1] // 8, 9))
    return out


def _selftest():
    ok = True
    rng = np.random.default_rng(7)
    print("[selftest] BFP16 codec vs the toolchain")
    try:
        sys.path.append(str(HERE.parents[3] / "_pylibs"))
        sys.path.append(str(HERE.parents[3] / "_pylibs" / "mlir_aie" / "python"))
        from aie.utils import bfp as _bfp
        x = rng.standard_normal((4, 8 * 37)).astype(np.float32)
        for r in ("floor", "conv_even"):
            a = encode_bfp(x, rounding=r)
            b = _bfp.encode(x, rounding=r)
            same = bool(np.array_equal(a, b))
            print(f"   encode({r}): {'IDENTICAL' if same else 'DIFFERS'} ({a.size} B)")
            ok &= same
        d = decode_bfp(_bfp.encode(x))
        same_d = bool(np.array_equal(d, _bfp.decode(_bfp.encode(x))))
        print(f"   decode: {'IDENTICAL' if same_d else 'DIFFERS'}")
        ok &= same_d
    except Exception as exc:  # noqa: BLE001
        print(f"   (toolchain not importable here: {exc}) -- skipped")

    print("[selftest] A round-trip")
    x = rng.standard_normal((1024, 3840)).astype(np.float32)
    a_buf = pack_a(x)
    same = bool(np.array_equal(unpack_a(a_buf, 1024, 3840), _bf16(x)))
    print(f"   main: {a_buf.size:,} B -> {'EXACT' if same else 'BROKEN'}")
    ok &= same
    at = pack_a_tail(x[:32])
    same_t = bool(np.array_equal(unpack_a_tail(at, 3840), _bf16(x[:32])))
    print(f"   tail: {at.size:,} B -> {'EXACT' if same_t else 'BROKEN'}")
    ok &= same_t

    print("[selftest] product A layout <-> this layout (byte level)")
    rows = 1056
    xp = _bf16(rng.standard_normal((rows, 3840)).astype(np.float32))
    prod = pack_product_a(xp)
    back = unpack_product_a(prod, rows, 3840)
    rt = bool(np.array_equal(back, xp))
    a_main, a_tail = convert_a(prod, rows, 3840)
    print(f"   product payload {len(prod):,} B; round-trip {'EXACT' if rt else 'BROKEN'}")
    print(f"   convert_a -> main {a_main.size:,} B + tail "
          f"{0 if a_tail is None else a_tail.size:,} B")
    rec_ok = bool(np.array_equal(unpack_a(a_main, 1024, 3840), xp[:1024]))
    tail_ok = bool(np.array_equal(unpack_a_tail(a_tail, 3840), xp[1024:]))
    print(f"   unpack_a(convert_a main) == x[:1024]: {'EXACT' if rec_ok else 'BROKEN'}")
    print(f"   unpack_a_tail(convert_a tail) == x[1024:]: {'EXACT' if tail_ok else 'BROKEN'}")
    ok &= rt and rec_ok and tail_ok
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--key", default="layers.0")
    ap.add_argument("--bench-b", action="store_true")
    ap.add_argument("--layers", type=int, default=30)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if args.selftest:
        return 0 if _selftest() else 1
    if args.bench_b:
        keys = [f"layers.{i}" for i in range(args.layers)]
        _, st = warm_b(keys, _load_model_layer, bench=True)
        print(json.dumps(st, indent=2))
        if args.json:
            pathlib.Path(args.json).write_text(json.dumps(st, indent=2))
        return 0
    if args.dry:
        w = _load_model_layer(args.key)
        rng = np.random.default_rng(3)
        xn = (rng.standard_normal((1056, 3840)).astype(np.float32) * 0.5)
        t0 = time.perf_counter()
        y, info = ffn32_dry(xn, w["w1"], w["w3"], w["w2"], key=args.key)
        t1 = time.perf_counter()
        yr = ffn32_reference(xn, w["w1"], w["w3"], w["w2"])
        t2 = time.perf_counter()
        d = (y - yr).astype(np.float64)
        exact = True
        for part in info["parts"].values():
            for k, v in part.items():
                exact &= bool(v["a_equal"] and v["b_equal"] and v["c_equal"])
        rec = dict(info)
        rec.update({"dry_s": round(t1 - t0, 2), "ref_s": round(t2 - t1, 2),
                    "corr": float(np.corrcoef(y.reshape(-1), yr.reshape(-1))[0, 1]),
                    "maxabs": float(np.abs(d).max()),
                    "y_absmax": float(np.abs(yr).max()),
                    "all_layouts_exact": bool(exact)})
        print(json.dumps(rec, indent=2)[:3000])
        if args.json:
            pathlib.Path(args.json).write_text(json.dumps(rec, indent=2))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
