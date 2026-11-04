# -*- coding: utf-8 -*-
"""bcore_tail.py -- numba kernels + timing/accuracy helpers for the bf16 study.

bf16 <-> f32 is done with true register bitcasts (numba intrinsic + llvmlite),
so the bf16 kernels are exact and branch-free.  cache=True on purpose: this
process may read the temp dir but not write it, so numba's disk cache is off.
"""
from __future__ import annotations

import time

import numpy as np
from llvmlite import ir
from numba import types
from numba.extending import intrinsic


@intrinsic
def _f32_as_u32(typingctx, x):
    sig = types.uint32(types.float32)

    def codegen(context, builder, signature, args):
        return builder.bitcast(args[0], ir.IntType(32))

    return sig, codegen


@intrinsic
def _u32_as_f32(typingctx, x):
    sig = types.float32(types.uint32)

    def codegen(context, builder, signature, args):
        return builder.bitcast(args[0], ir.FloatType())

    return sig, codegen


def _kernels():
    from numba import get_num_threads, get_thread_id, njit, prange

    @njit(cache=True, inline="always")
    def bf16_to_f32(u):
        return _u32_as_f32(np.uint32(u) << np.uint32(16))

    @njit(cache=True, inline="always")
    def f32_to_bf16(v):
        u = _f32_as_u32(np.float32(v))
        return np.uint16(
            (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1)))
            >> np.uint32(16)
        )

    @njit(cache=True, parallel=True, fastmath=True, nogil=True)
    def sm_f32_inplace(a):
        rows = a.shape[0]
        S = a.shape[1]
        for i in prange(rows):
            m = -3.4e38
            for j in range(S):
                v = a[i, j]
                if v > m:
                    m = v
            s = 0.0
            for j in range(S):
                e = np.exp(a[i, j] - m)
                a[i, j] = e
                s += e
            inv = 1.0 / s
            for j in range(S):
                a[i, j] *= inv

    @njit(cache=True, parallel=True, fastmath=True, nogil=True)
    def sm_f32_inplace_scaled(a, scale):
        rows = a.shape[0]
        S = a.shape[1]
        for i in prange(rows):
            m = -3.4e38
            for j in range(S):
                v = a[i, j] * scale
                if v > m:
                    m = v
            s = 0.0
            for j in range(S):
                e = np.exp(a[i, j] * scale - m)
                a[i, j] = e
                s += e
            inv = 1.0 / s
            for j in range(S):
                a[i, j] *= inv

    @njit(cache=True, parallel=True, fastmath=True, nogil=True)
    def sm_f32_out(a, out, scale):
        rows = a.shape[0]
        S = a.shape[1]
        for i in prange(rows):
            m = -3.4e38
            for j in range(S):
                v = a[i, j] * scale
                if v > m:
                    m = v
            s = 0.0
            for j in range(S):
                e = np.exp(a[i, j] * scale - m)
                out[i, j] = e
                s += e
            inv = 1.0 / s
            for j in range(S):
                out[i, j] *= inv

    @njit(cache=True, parallel=True, fastmath=True, nogil=True)
    def sm_bf16_bf16(u, out, scale):
        rows = u.shape[0]
        S = u.shape[1]
        bufs = np.zeros((get_num_threads(), S), np.float32)
        for i in prange(rows):
            buf = bufs[get_thread_id()]
            m = -3.4e38
            for j in range(S):
                v = bf16_to_f32(u[i, j]) * scale
                if v > m:
                    m = v
            s = 0.0
            for j in range(S):
                e = np.exp(bf16_to_f32(u[i, j]) * scale - m)
                buf[j] = e
                s += e
            inv = 1.0 / s
            for j in range(S):
                buf[j] = buf[j] * inv
            for j in range(S):
                out[i, j] = f32_to_bf16(buf[j])

    @njit(cache=True, parallel=True, fastmath=True, nogil=True)
    def sm_bf16_f32(u, out, scale):
        rows = u.shape[0]
        S = u.shape[1]
        bufs = np.zeros((get_num_threads(), S), np.float32)
        for i in prange(rows):
            buf = bufs[get_thread_id()]
            m = -3.4e38
            for j in range(S):
                v = bf16_to_f32(u[i, j]) * scale
                if v > m:
                    m = v
            s = 0.0
            for j in range(S):
                e = np.exp(bf16_to_f32(u[i, j]) * scale - m)
                out[i, j] = e
                s += e
            inv = 1.0 / s
            for j in range(S):
                out[i, j] *= inv

    return dict(bf16_to_f32=bf16_to_f32, f32_to_bf16=f32_to_bf16,
                sm_f32_inplace=sm_f32_inplace, sm_f32_out=sm_f32_out,
                sm_f32_inplace_scaled=sm_f32_inplace_scaled,
                sm_bf16_bf16=sm_bf16_bf16, sm_bf16_f32=sm_bf16_f32)


_NB = None


def K():
    global _NB
    if _NB is None:
        _NB = _kernels()
    return _NB


def bench(fn, n=7, warm=2):
    last = None
    for _ in range(warm):
        last = fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        last = fn()
        ts.append(1e3 * (time.perf_counter() - t0))
    ts.sort()
    return dict(med=float(np.median(ts)), mn=float(ts[0]), mx=float(ts[-1]),
                all=[round(t, 2) for t in ts]), last


def stats(ref, out):
    a = np.asarray(ref, np.float64).ravel()
    b = np.asarray(out, np.float64).ravel()
    ac = a - a.mean()
    bc = b - b.mean()
    den = np.sqrt((ac * ac).sum() * (bc * bc).sum())
    corr = float((ac * bc).sum() / den) if den > 0 else float("nan")
    d = b - a
    maxabs = float(np.abs(d).max())
    rref = float(np.abs(a).max())
    rms = float(np.sqrt((d * d).mean()))
    rrms = float(np.sqrt((a * a).mean()))
    return dict(corr=corr, maxabs=maxabs, rel_maxabs=maxabs / rref if rref else 0.0,
                rel_rms=rms / rrms if rrms else 0.0, ref_absmax=rref, ref_rms=rrms)