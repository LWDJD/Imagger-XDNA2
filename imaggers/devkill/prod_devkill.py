# SPDX-License-Identifier: MIT
"""prod_devkill.py -- t22_devkill projection path (plain X / row-major C, G=15).

Drop-in for prod_loop_bigm.bigm_attention.  Identical to prod_hostkill except
that the projection image is the G=15 (HALF=2) build registered as "plain15"
when T22_DEVKILL=1.

Gate: T22_DEVKILL=1 (default 0).  Product default is unchanged.
"""
from __future__ import annotations
import os, pathlib, time
import numpy as np

import prod_loop_bigm as PB
from qkv_image_g import qkv_to_image_g

HERE = pathlib.Path(__file__).resolve().parent
NPU = HERE.parents[1]

_PROJ_W = {}
_F110 = os.environ.get("T22_FINAL110", "0") == "1"
_YBUF = {}
_NAMES = {"q": "to_q", "k": "to_k", "v": "to_v", "o": "to_out0"}
_IMG = os.environ.get("T22_DEVKILL_PROJ_IMG", "plain15")


def _bn(tag):
    return "b_%s_%s" % (_IMG, tag)


def _w(key, name):
    li = int(str(key).split(".")[1])
    k2 = (li, name)
    if k2 not in _PROJ_W:
        li2 = int(str(key).split(".")[1])
        fn = {"q": "to_q", "k": "to_k", "v": "to_v", "o": "to_out0"}[name]
        raw = np.load(PB.QKV_DIR / ("L%d_%s.npy" % (li2, fn))).tobytes()
        _PROJ_W[k2] = qkv_to_image_g(raw, G=int(os.environ.get("T22_DEVKILL_G", "15"))).tobytes()
    return _PROJ_W[k2]


_WIDEN = None


def _widen(u16, y):
    # 2026-11-01 parent: bf16 bit pattern -> float32 by writing the shifted
    # uint32 straight into y's storage.  Bit-identical to
    # np.left_shift(u16, 16, out=y.view(np.uint32)); measured 3.52 -> 0.32 ms
    # for a 12-tile C (8 MB in, 17 MB out), i.e. 10.9x.
    global _WIDEN
    if _WIDEN is None:
        try:
            from numba import njit, prange

            @njit(parallel=True, cache=True)
            def _k(u, o):
                for i in prange(u.shape[0]):
                    for j in range(u.shape[1]):
                        o[i, j] = np.uint32(u[i, j]) << np.uint32(16)
            _WIDEN = _k
        except Exception:
            _WIDEN = False
    if _WIDEN is not False:
        try:
            _WIDEN(u16, y.view(np.uint32))
            return y
        except Exception:
            pass
    np.left_shift(u16, 16, out=y.view(np.uint32), dtype=np.uint32)
    return y


def _run(dev, xb, w, n_tiles, tim, tag):
    """One plain projection dispatch: X raw bf16 bytes in, row-major C out."""
    bo = dev.bo
    t0 = time.perf_counter()
    bo[_bn("x")].write(xb, 0)
    bo[_bn("x")].sync(dev.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo[_bn("w")].write(w, 0)
    bo[_bn("w")].sync(dev.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    t1 = time.perf_counter()
    im = dev.imgs[_IMG]
    run = im["k"](3, im["bo_i"], len(im["inst"]) // 4,
                  bo[_bn("x")], bo[_bn("w")], bo[_bn("o")])
    st = int(run.wait(300000))
    wall = 1e3 * (time.perf_counter() - t1)
    if st != 4:
        raise RuntimeError("devkill plain %s state %d" % (tag, st))
    t2 = time.perf_counter()
    if _F110:
        n = n_tiles * 96 * 3840
        key = (tag, n_tiles)
        y = _YBUF.get(key)
        if y is None:
            y = np.empty((n_tiles * 96, 3840), np.float32)
            _YBUF[key] = y
        ob = bo[_bn("o")]
        ob.sync(dev.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        try:
            u16 = np.frombuffer(ob.map(), np.uint16, count=n).reshape(n_tiles * 96, 3840)
        except Exception:
            u16 = np.frombuffer(bytes(ob.read(n * 2, 0)), np.uint16, count=n).reshape(n_tiles * 96, 3840)
        _widen(u16, y)
        t3 = time.perf_counter()
    else:
        # BUGFIX (2026-11-03, DET): this branch read the output BO with NO
        # XCL_BO_SYNC_BO_FROM_DEVICE, unlike the _F110 branch above and unlike
        # t18_run.Dev.dispatch (which syncs every out_key before reading).  A
        # host_only BO is read straight out of host memory, so without the sync
        # the CPU can see a line whose device DMA has not landed yet -- a FEW
        # elements stale, run to run.  Measured: two identical 1-step campaign
        # runs differ from layer 4 of step 0 on (a 2064 vs 2080, y 5600 vs
        # 5632), and the SAME pair with T22_FINAL110=1 (which syncs) is
        # bit-identical on all 30 layers.  This is the campaign env's
        # irreproducibility; the delivered path was never affected because it
        # goes through Dev.dispatch.
        ob = bo[_bn("o")]
        ob.sync(dev.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        raw = bytes(ob.read(n_tiles * 96 * 3840 * 2, 0))
        t3 = time.perf_counter()
        y = np.empty((n_tiles * 96, 3840), np.float32)
        _widen(np.frombuffer(raw, np.uint16).reshape(n_tiles * 96, 3840), y)
    t4 = time.perf_counter()
    if os.environ.get("T22_BIGM_DEBUG", "0") == "1":
        print("[plain15] %s tiles=%d write=%.1f run=%.1f read=%.1f cv=%.1f ms"
              % (tag, n_tiles, 1e3*(t1-t0), wall, 1e3*(t3-t2), 1e3*(t4-t3)),
              flush=True)
    if tim is not None:
        tim["dev"] = tim.get("dev", 0.0) + wall
        tim["write"] = tim.get("write", 0.0) + 1e3*(t1-t0)
        tim["pack"] = tim.get("pack", 0.0) + 1e3*(t4-t3)
    return y


def proj4(dev, x, key, names, tim=None, max_tiles=11):
    seq = x.shape[0]
    chunks = []
    for r0 in range(0, seq, max_tiles * 96):
        r1 = min(seq, r0 + max_tiles * 96)
        rows = r1 - r0
        nt = (rows + 95) // 96
        pad = np.zeros((nt * 96, x.shape[1]), np.float32)
        pad[:rows] = x[r0:r1]
        chunks.append((PB._bf16_bytes(pad).tobytes(), rows, nt))
    out = {}
    for nm in names:
        w = _w(key, nm)
        parts = [_run(dev, xb, w, nt, tim, "plain15/" + nm)
                 for xb, rows, nt in chunks]
        y = parts[0] if len(parts) == 1 else np.concatenate(parts, 0)
        out[nm] = y[:seq]
    return out


def bigm_attention(self, x1, key, cos, sin, tim):
    """Drop-in for Loop.attention: plain G=15 projection + host-f32 attention."""
    T = PB._t18()
    P = self.params(key)
    seq, D, HEADS, HD = T.SEQ, T.D, T.HEADS, T.HD
    t0 = time.perf_counter()
    Q = proj4(self.dev, x1, key, ("q", "k", "v"), tim)
    t_proj = time.perf_counter()

    def heads(a):
        return np.ascontiguousarray(a.reshape(seq, HEADS, HD).transpose(1, 0, 2))

    wq = T.bf16f32(P["norm_q"] * self.fold)
    kh = T.rope_rms_jit(heads(Q["k"]), P["norm_k"], cos, sin,
                        T.SCRATCH.get("kh", (HEADS, seq, HD), np.float32),
                        T.SCRATCH.get("rms0", (HEADS, seq, HD), np.float32))
    qh = T.rope_rms_jit(heads(Q["q"]), wq, cos, sin,
                        T.SCRATCH.get("qh", (HEADS, seq, HD), np.float32),
                        T.SCRATCH.get("rms1", (HEADS, seq, HD), np.float32))
    th = time.perf_counter()
    from webui.chunked import _attention
    vh = heads(Q["v"])
    o = _attention(qh, kh, vh)
    ta = time.perf_counter()
    if tim is not None:
        tim["host"] = tim.get("host", 0.0) + 1e3 * (time.perf_counter() - th)
    o_flat = np.ascontiguousarray(o.reshape(seq, D))
    y = proj4(self.dev, o_flat, key, ("o",), tim)["o"]
    t_end = time.perf_counter()
    if os.environ.get("T22_BIGM_DEBUG", "0") == "1":
        print("[devkill] %s proj_qkv %.3f rope %.3f attn %.3f proj_o %.3f"
              % (key, t_proj - t0, th - t_proj, ta - th, t_end - ta), flush=True)
    return y
