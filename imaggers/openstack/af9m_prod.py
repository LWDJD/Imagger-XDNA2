# SPDX-License-Identifier: MIT
"""Product wrapper: 8-core v9 fused attention, 30 heads in 4 rounds of 8."""
import os, pathlib, time
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
TAG = "af9m8_1056"
NW = 8
SEQ = 1056
HD = 128
BK = 32
NK = SEQ // BK


def _bf(x):
    u = np.asarray(x, np.float32).view(np.uint32)
    return ((u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))) >> 16).astype(np.uint16)


def _bff(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


class Engine:
    def __init__(self, dev):
        self.pyxrt = dev.pyxrt
        self.dev = dev.dev
        p = HERE / (TAG + ".xclbin")
        xc = self.pyxrt.xclbin(str(p)); uuid = self.dev.register_xclbin(xc)
        self.k = self.pyxrt.kernel(self.dev, uuid, "MLIR_AIE")
        self.inst = (HERE / (TAG + ".bin")).read_bytes(); self.ni = len(self.inst) // 4
        fl = self.pyxrt.bo.flags.host_only
        self.bi = self.pyxrt.bo(self.dev, len(self.inst), self.pyxrt.bo.flags.cacheable, self.k.group_id(1))
        self.bi.write(self.inst, 0); self.bi.sync(self.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        self.bq = self.pyxrt.bo(self.dev, NW * SEQ * HD * 2, fl, self.k.group_id(3))
        self.bkv = self.pyxrt.bo(self.dev, NW * NK * BK * HD * 4, fl, self.k.group_id(4))
        self.bo = self.pyxrt.bo(self.dev, NW * SEQ * HD * 2, fl, self.k.group_id(5))
        self.walls = []

    def group(self, qh, kh, vh):
        n = qh.shape[0]
        qs = []; kvs = []
        for j in range(NW):
            if j < n:
                q = _bf(qh[j] * np.sqrt(np.float32(HD))); k = _bf(kh[j]); vb = _bf(vh[j])
            else:
                q = np.zeros((SEQ, HD), np.uint16); k = q.copy(); vb = q.copy()
            kt = np.concatenate([np.ascontiguousarray(k[i*BK:(i+1)*BK].T).reshape(-1) for i in range(NK)]).view(np.uint8).reshape(NK, BK*HD*2)
            vt = vb.view(np.uint8).reshape(NK, BK*HD*2)
            qs.append(q.reshape(-1)); kvs.append(np.concatenate([kt, vt], axis=1).reshape(-1))
        self.bq.write(np.concatenate(qs).tobytes(), 0)
        self.bkv.write(np.concatenate(kvs).tobytes(), 0)
        self.bq.sync(self.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        self.bkv.sync(self.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        t0 = time.perf_counter(); run = self.k(3, self.bi, self.ni, self.bq, self.bkv, self.bo)
        st = int(run.wait(600000)); self.walls.append(1e3 * (time.perf_counter() - t0))
        if st != 4: raise RuntimeError("af9m state %d" % st)
        self.bo.sync(self.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        raw = _bff(np.frombuffer(bytes(self.bo.read(NW * SEQ * HD * 2, 0)), np.uint16)).reshape(NW, SEQ, HD)
        return raw[:n]


_ENG = None

def stream_attention(dev, qh, kh, vh):
    global _ENG
    (HERE / "af9m_called.txt").write_text("pid=%d heads=%d\n" % (os.getpid(), qh.shape[0]))
    heads, seq, hd = qh.shape
    assert seq == SEQ and hd == HD
    if _ENG is None: _ENG = Engine(dev)
    out = np.empty((seq, heads, hd), np.float32)
    t0 = time.perf_counter()
    for g in range(0, heads, NW):
        n = min(NW, heads - g)
        out[:, g:g+n, :] = _ENG.group(np.ascontiguousarray(qh[g:g+n]), np.ascontiguousarray(kh[g:g+n]), np.ascontiguousarray(vh[g:g+n])).transpose(1, 0, 2)
    if os.environ.get("T22_BIGM_DEBUG", "0") == "1":
        print("[af9m] heads=%d groups=%d wall=%.1f ms per_group=%s" % (heads, (heads + NW - 1)//NW, 1e3*(time.perf_counter()-t0), [round(w,1) for w in _ENG.walls[-4:]]), flush=True)
    return np.ascontiguousarray(out.reshape(seq, heads * hd))
