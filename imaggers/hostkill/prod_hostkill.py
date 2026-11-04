# SPDX-License-Identifier: MIT
"""prod_hostkill.py -- the t22_hostkill projection path.

Drop-in for prod_loop_bigm.bigm_attention.  The ONLY difference is the
projection image: bigm_proj_plain lets the shim DMA pack A and un-tile C, so
the host no longer runs a_pack() or _proj_decode().  The GEMM, the W stream and
the arithmetic are the same bytes as prod_loop_bigm.

Gate: T22_HOSTKILL=1 (default 0).  Product default is unchanged.
"""
from __future__ import annotations
import os, pathlib, time
import numpy as np

import prod_loop_bigm as PB

HERE = pathlib.Path(__file__).resolve().parent
NPU = HERE.parents[1]

_PROJ_W = {}
_NAMES = {"q": "to_q", "k": "to_k", "v": "to_v", "o": "to_out0"}


def _w(key, name):
    li = int(str(key).split(".")[1])
    k2 = (li, name)
    if k2 not in _PROJ_W:
        _PROJ_W[k2] = PB._proj_w(key, name)
    return _PROJ_W[k2]


def _run(dev, xb, w, n_tiles, tim, tag):
    """One plain projection dispatch: X raw bf16 bytes in, row-major C out."""
    bo = dev.bo
    t0 = time.perf_counter()
    bo["b_plain_x"].write(xb, 0)
    bo["b_plain_x"].sync(dev.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo["b_plain_w"].write(w, 0)
    bo["b_plain_w"].sync(dev.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    t1 = time.perf_counter()
    im = dev.imgs["plain"]
    run = im["k"](3, im["bo_i"], len(im["inst"]) // 4,
                  bo["b_plain_x"], bo["b_plain_w"], bo["b_plain_o"])
    st = int(run.wait(300000))
    wall = 1e3 * (time.perf_counter() - t1)
    if st != 4:
        raise RuntimeError("hostkill plain %s state %d" % (tag, st))
    t2 = time.perf_counter()
    raw = bytes(bo["b_plain_o"].read(n_tiles * 96 * 3840 * 2, 0))
    t3 = time.perf_counter()
    u = np.frombuffer(raw, np.uint16).reshape(n_tiles * 96, 3840).astype(np.uint32) << 16
    y = u.view(np.float32)
    t4 = time.perf_counter()
    if os.environ.get("T22_BIGM_DEBUG", "0") == "1":
        print("[plain] %s tiles=%d write=%.1f run=%.1f read=%.1f cv=%.1f ms"
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
        parts = [_run(dev, xb, w, nt, tim, "plain/" + nm)
                 for xb, rows, nt in chunks]
        y = parts[0] if len(parts) == 1 else np.concatenate(parts, 0)
        out[nm] = y[:seq]
    return out


def bigm_attention(self, x1, key, cos, sin, tim):
    """Drop-in for Loop.attention: plain projection + host-f32 attention."""
    import time as _time
    T = PB._t18()
    P = self.params(key)
    seq, D, HEADS, HD = T.SEQ, T.D, T.HEADS, T.HD
    t0 = _time.perf_counter()
    Q = proj4(self.dev, x1, key, ("q", "k", "v"), tim)
    t_proj = _time.perf_counter()

    def heads(a):
        return np.ascontiguousarray(a.reshape(seq, HEADS, HD).transpose(1, 0, 2))

    wq = T.bf16f32(P["norm_q"] * self.fold)
    kh = T.rope_rms_jit(heads(Q["k"]), P["norm_k"], cos, sin,
                        T.SCRATCH.get("kh", (HEADS, seq, HD), np.float32),
                        T.SCRATCH.get("rms0", (HEADS, seq, HD), np.float32))
    qh = T.rope_rms_jit(heads(Q["q"]), wq, cos, sin,
                        T.SCRATCH.get("qh", (HEADS, seq, HD), np.float32),
                        T.SCRATCH.get("rms1", (HEADS, seq, HD), np.float32))
    th = _time.perf_counter()
    from webui.chunked import _attention
    vh = heads(Q["v"])
    if os.environ.get("T22_AF9M", "0") == "1":
        import sys as _sys
        _af = NPU / "imaggers" / "openstack"
        if str(_af) not in _sys.path:
            _sys.path.insert(0, str(_af))
        import af9m_prod as _AF
        o = _AF.stream_attention(self.dev, qh, kh, vh)
    else:
        o = _attention(qh, kh, vh)
    ta = _time.perf_counter()
    if tim is not None:
        tim["host"] = tim.get("host", 0.0) + 1e3 * (_time.perf_counter() - th)
    o_flat = np.ascontiguousarray(o.reshape(seq, D))
    y = proj4(self.dev, o_flat, key, ("o",), tim)["o"]
    t_end = _time.perf_counter()
    if os.environ.get("T22_BIGM_DEBUG", "0") == "1":
        print("[hostkill] %s proj_qkv %.3f rope %.3f attn %.3f proj_o %.3f"
              % (key, t_proj - t0, th - t_proj, ta - th, t_end - ta), flush=True)
    return y
