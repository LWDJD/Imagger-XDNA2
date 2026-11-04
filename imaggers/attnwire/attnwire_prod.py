# SPDX-License-Identifier: MIT
"""Product wrapper: 6-core streaming attention; 1056-row query chunks for 1024^2."""
from __future__ import annotations
import os, pathlib, time
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
XCL = {"1056": ("awm6h_1056", 1056, 1056),
       "4128": ("awm6h_4128q1056", 1056, 4128)}
NH = 6


def _bf(x):
    u = np.asarray(x, np.float32).view(np.uint32)
    return ((u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))) >> 16).astype(np.uint16)


def _bff(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


class Engine:
    def __init__(self, dev, seq):
        self.pyxrt = dev.pyxrt
        self.tag, self.qlen, self.kvseq = XCL[str(seq)]
        p = HERE / (self.tag + ".xclbin")
        if not p.exists():
            raise RuntimeError("attnwire: missing %s" % p)
        self.dev = dev.dev
        xc = self.pyxrt.xclbin(str(p))
        uuid = self.dev.register_xclbin(xc)
        self.k = self.pyxrt.kernel(self.dev, uuid, "MLIR_AIE")
        self.inst = (HERE / (self.tag + ".bin")).read_bytes()
        self.ni = len(self.inst) // 4
        fl = self.pyxrt.bo.flags.host_only
        self.bi = self.pyxrt.bo(self.dev, len(self.inst), self.pyxrt.bo.flags.cacheable,
                                self.k.group_id(1))
        self.bi.write(self.inst, 0)
        self.bi.sync(self.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        hd = 128
        self.hd = hd
        self.qbytes = NH * self.qlen * hd * 2
        self.kvbytes = NH * (self.kvseq // 32) * (32 * hd * 2 + 32 * hd * 4)
        self.bq = self.pyxrt.bo(self.dev, self.qbytes, fl, self.k.group_id(3))
        self.bkv = self.pyxrt.bo(self.dev, self.kvbytes, fl, self.k.group_id(4))
        self.bo = self.pyxrt.bo(self.dev, self.qbytes, fl, self.k.group_id(5))
        self.walls = []

    def _kv(self, kh, vh):
        bk = 32; hd = self.hd; nk = self.kvseq // bk
        kvs = []
        for h in range(NH):
            if h < kh.shape[0]:
                k = _bf(kh[h]); vb = _bf(vh[h]); v = _bff(vb).astype(np.float32)
            else:
                k = np.zeros((self.kvseq, hd), np.uint16)
                vb = np.zeros((self.kvseq, hd), np.uint16)
                v = np.zeros((self.kvseq, hd), np.float32)
            kt = k.view(np.uint8).reshape(nk, bk * hd * 2)
            vt = v.view(np.uint8).reshape(nk, bk * hd * 4)
            kvs.append(np.concatenate([kt, vt], axis=1).reshape(-1))
        return np.concatenate(kvs)

    def group(self, qh, kh, vh):
        n, qlen, hd = qh.shape
        qs = []
        for h in range(NH):
            if h < n:
                q = _bf(qh[h] * np.sqrt(np.float32(hd)))
            else:
                q = np.zeros((self.qlen, hd), np.uint16)
            if q.shape[0] < self.qlen:
                pad = np.zeros((self.qlen - q.shape[0], hd), np.uint16)
                q = np.concatenate([q, pad], axis=0)
            qs.append(q.reshape(-1))
        q_all = np.concatenate(qs)
        kv_all = self._kv(kh, vh)
        self.bq.write(q_all.tobytes(), 0); self.bkv.write(kv_all.tobytes(), 0)
        self.bq.sync(self.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        self.bkv.sync(self.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        st = -1
        for attempt in range(3):
            t0 = time.perf_counter()
            run = self.k(3, self.bi, self.ni, self.bq, self.bkv, self.bo)
            st = int(run.wait(600000))
            self.walls.append(1e3 * (time.perf_counter() - t0))
            if st == 4:
                break
            time.sleep(1.0)
        if st != 4:
            raise RuntimeError("attnwire %s state %d" % (self.tag, st))
        self.bo.sync(self.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        out = _bff(np.frombuffer(bytes(self.bo.read(self.qbytes, 0)), np.uint16)).reshape(NH, self.qlen, hd)
        return out[:n].transpose(1, 0, 2)


_ENGINES = {}


def stream_attention(dev, qh, kh, vh):
    heads, seq, hd = qh.shape
    eng = _ENGINES.get(seq)
    if eng is None:
        eng = _ENGINES[seq] = Engine(dev, seq)
    out = np.empty((seq, heads, hd), np.float32)
    t0 = time.perf_counter()
    for q0 in range(0, seq, eng.qlen):
        q1 = min(seq, q0 + eng.qlen)
        for g in range(0, heads, NH):
            n = min(NH, heads - g)
            o = eng.group(np.ascontiguousarray(qh[g:g + n, q0:q1]),
                          np.ascontiguousarray(kh[g:g + n]),
                          np.ascontiguousarray(vh[g:g + n]))
            out[q0:q1, g:g + n, :] = o[:q1 - q0]
    dt = time.perf_counter() - t0
    if os.environ.get("T22_BIGM_DEBUG", "0") == "1":
        print("[attnwire] seq=%d heads=%d qchunk=%d wall=%.1f ms"
              % (seq, heads, eng.qlen, 1e3 * dt), flush=True)
    return np.ascontiguousarray(out.reshape(seq, heads * hd))
