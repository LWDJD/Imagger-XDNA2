# SPDX-License-Identifier: MIT
"""kk_prod.py -- the k-blocked (kk=32) GEMM wired into the product FFN gate/up.

    T22_FFN_KK32=1   (default 0)  on  -> the FFN gate/up runs on the k-blocked
                                         image out/kk32_m32_n512.xclbin
                                     off -> the shipped swiglu image, untouched

WHAT THE IMAGE IS (device-verified; KK-GEMM-REPORT.md + out/kk_prod_check.json):

    kk32_m32_n512   M=1024  K=3840  N=10240  m=32 kk=32 n=512 cols=8 cpc=4
    13.19 ms / 6.11 TFLOP/s  vs the S2 kk=8 image ffn_gu_m32 19.70 ms / 4.09
    corr 0.999915 against a numpy oracle built from the REAL layer-0 w1 blocks,
    worst of 32 M-tiles 0.999913, A-tile-swap control -0.0077.

WHAT THE PRODUCT WANTS (SEQ = 32 caption + 32x32 image tokens = 1056 rows):

    shipped  swiglu11   M=1056 K=3840 N=20480, silu(gate)*up fused ON DEVICE,
                        drains 1056x10240 straight into the down image's A BO
                        (t22_bigpipe3 device handoff), measured 48.0 ms

    1056 rows = 33 M-tiles of 32 and the image has 32 (column, core) slots, so
    one layer-step needs ceil(33/32) = 2 row chunks x 2 halves (gate, up) =
    FOUR dispatches.  That is the whole story of the measurement: 4 x 13.19 =
    52.8 ms of device time against the shipped 48.0 ms, and the fused swiglu
    plus the device handoff both move to the host.

B IS NEVER RE-ENCODED
---------------------
pack_bfp16.pack() scatters quantisation group i to slot[i]; the inverse is the
GATHER.  Doing only that gather and stopping leaves the checkpoint's 9-byte
units byte for byte in row-major group order -- ~0.3 % of them are
non-canonical and a re-encode silently rewrites them.  Every B payload below is
therefore a pure PERMUTATION of the stored bytes, and pack_b_kk's output is
asserted to be a permutation of its input on first use.

A IS PACKED FROM THE bf16 THE GLUE ALREADY HOLDS
------------------------------------------------
devkill_ffn.block() hands us (rows, 3840) uint16 bf16.  pack_a_kk() takes f32
and rounds; rounding an exact bf16 is the identity, so pack_a_kk(widen(u16)) ==
pack_a_kk_u16(u16) byte for byte (asserted in kk_prod_check.py --check-a), and
the u16 entry point skips a 16 MB widen per chunk.
"""
from __future__ import annotations

import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "out"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---- the verified image ----------------------------------------------------
KK = dict(tag="kk32_m32_n512", M=1024, K=3840, N=10240, m=32, kk=32, n=512,
          cols=8, cpc=4, ntiles=20)
A_BYTES = KK["M"] * KK["K"] * 2          # 7,864,320
B_BYTES = KK["N"] * KK["K"] * 9 // 8     # 44,236,800
C_BYTES = KK["M"] * KK["N"] * 2          # 20,971,520
IMG = "kk32"                             # dev.imgs / dev.bo key
XCLBIN = OUT / (KK["tag"] + ".xclbin")
INSTS = OUT / (KK["tag"] + ".bin")

# the product's FFN geometry (SEQ=1056 -> 11 row tiles of 96, 320 objects of 96x32)
SEQ = 1056
FF = 10240
DN_OBJ = 320
DN_A_BYTES = SEQ * FF * 2                # 21,626,880 -- b_<dn>_x

STATS = {"dispatch": 0, "rows": 0, "layers": set(), "calls": 0,
         "t_disp_ms": 0.0, "t_host_ms": 0.0, "t_write_ms": 0.0,
         "t_apack_ms": 0.0, "t_bpack_ms": 0.0, "walls": []}


def enabled():
    return os.environ.get("T22_FFN_KK32", "0") == "1"


def report():
    st = STATS
    w = st["walls"]
    print("[kk32] dispatched %d kk=32 images over %d layer-steps (%d layers: "
          "%s); device %.1f ms, host %.1f ms, A pack %.1f ms, B repack %.1f ms, "
          "writes %.1f ms%s"
          % (st["dispatch"], st["calls"], len(st["layers"]),
             ",".join(str(x) for x in sorted(st["layers"]))[:120],
             st["t_disp_ms"], st["t_host_ms"], st["t_apack_ms"], st["t_bpack_ms"],
             st["t_write_ms"],
             ("; dispatch ms/layer-step %.2f (4 x %.2f)"
              % (4 * float(np.median(w)), float(np.median(w)))) if w else ""),
          flush=True)


# ---------------------------------------------------------------------------
# the model's stored BFP16 blocks -- decoded, never re-encoded
# ---------------------------------------------------------------------------
def blocks(key, which):
    """'layers.N' + ('w1' = gate | 'w3' = up) -> (10240, 480, 9) uint8.

    Bytes are the checkpoint's own: pack_bfp16's slot GATHER puts the stored
    9-byte units back in row-major group order and nothing else touches them.
    """
    li = int(str(key).split(".")[1])
    import pack_bfp16 as PB
    z = np.load(ROOT / "weights" / ("block%d.npz" % li), allow_pickle=False)
    blob = z["layers.%d.feed_forward.%s.weight" % (li, which)]
    N, K = 10240, 3840
    ng = N * K // 8
    if blob.size != ng * 9:
        raise RuntimeError("kk_prod: %s.%s is %d B, expected %d"
                           % (key, which, blob.size, ng * 9))
    reg = np.ascontiguousarray(blob, np.uint8).reshape(ng, 9)
    units = reg[PB._slots(N, K, ng, PB.KNOWN_TILE_ROWS[(N, K)])]
    return np.ascontiguousarray(units).reshape(N, K // 8, 9)


_B = {}


def layer_by_prefix(w_gu):
    """Last-resort layer id: match a payload's first bytes against the cache.

    Only reached when the pointer test fails (i.e. the product is not holding
    its resident BG cache), and then we scan CACHE_BG's 30 rows' first 32 B.
    """
    T = __import__("prod_loop_bigm", fromlist=["_t18"])._t18()
    cb = getattr(T, "CACHE_BG", None)
    if cb is None:
        raise RuntimeError(
            "kk_prod: cannot identify the layer of the B payload -- run the "
            "product with --cache-all (CACHE_BG) or pass the key explicitly")
    probe = np.asarray(w_gu, np.uint8)[:32]
    for li in range(cb.shape[0]):
        if bytes(cb[li][:32]) == bytes(probe):
            return li
    raise RuntimeError("kk_prod: B payload matches none of the cached layers")


def packed_b(key):
    """(gate_payload, up_payload) for one layer, cached -- 2 x 44,236,800 B."""
    li = int(str(key).split(".")[1])
    if li not in _B:
        import pack_kk as PK
        t0 = time.perf_counter()
        out = []
        for which in ("w1", "w3"):
            blk = blocks(key, which)
            pay = PK.pack_b_kk(blk, KK["K"], KK["N"], KK["n"], KK["kk"])
            if pay.size != B_BYTES:
                raise RuntimeError("kk_prod: B payload %d B != %d"
                                   % (pay.size, B_BYTES))
            out.append(pay.tobytes())
            del blk
        dt = time.perf_counter() - t0
        STATS["t_bpack_ms"] += 1e3 * dt
        _B[li] = tuple(out)
        if os.environ.get("T22_FFN_KK32_PROF", "0") == "1":
            print("[kk32] layer %d B repack (2 x %.1f MB) %.1f ms"
                  % (li, B_BYTES / 1e6, 1e3 * dt), flush=True)
    return _B[li]


# ---------------------------------------------------------------------------
# numba kernels
# ---------------------------------------------------------------------------
_NB = None


def _kernels():
    global _NB
    if _NB is not None:
        return _NB
    from numba import njit, prange

    @njit(cache=True, parallel=True, nogil=True, boundscheck=False)
    def _pack_a_u16(xu, tm, out16, m, kk, cols, cpc, nkk, K):
        """(M,K) bf16 uint16 -> the image's A payload.

        Byte-identical to pack_kk.pack_a_kk on the same values widened to f32
        (its RNE on an exact bf16 is the identity).  Leaf order
        [col][k-step][core][leaf], leaf = [m/8][kk/8][8][8] bf16.
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
                            out16[dst + j] = xu[src + j]

    @njit(cache=True, parallel=True, nogil=True, boundscheck=False)
    def _swiglu_to_dn_a(gate, up, out, r_lo, r_hi, nrow_img):
        """silu(gate)*up from the two kk32 C BOs into the down image's A BO.

        gate/up are the image's C: [col][n-tile][core][m/8][n/8][8][8] bf16,
        where (col, core) owns M-tile c*cpc+p and every (col, core) covers
        every n-tile.  out is the down image's A: [rt][object 96x32][12][4]
        [8][8] with object q carrying output columns 32q.  One pass, one 8x8
        block per task; rows outside [r_lo, r_hi) are skipped (SEQ=1056 is two
        chunks of the same 1024-row image).
        """
        m, n = 32, 512
        cpc, ntiles = 4, 20
        leaf = m * n
        nrt = nrow_img // 96
        for task in prange(nrt * 320 * 12 * 4):
            cb4 = task & 3
            r1 = task >> 2
            rb9 = r1 % 12
            r2 = r1 // 12
            q = r2 % 320
            rt = r2 // 320
            grow = rt * 96 + rb9 * 8
            if grow < r_lo or grow >= r_hi:
                continue
            lrow = grow - r_lo
            tt = lrow // 32
            c = tt // cpc
            p = tt - c * cpc
            rb = (lrow - tt * 32) // 8
            obase = (((rt * 320 + q) * 12 + rb9) * 4 + cb4) * 64
            gb = np.empty(64, np.uint32)
            ub = np.empty(64, np.uint32)
            sb = np.empty(64, np.float32)
            for ci8 in range(8):
                col = q * 32 + cb4 * 8 + ci8
                t = col // n
                ci = col - t * n
                cb = ci // 8
                cj = ci - cb * 8
                src = ((c * ntiles + t) * cpc + p) * leaf + \
                      (rb * (n // 8) + cb) * 64 + cj
                for ri in range(8):
                    # the object's inner 64 words are [ri][ci] (untile_obj:
                    # [12][4][8][8] -> transpose -> [12][8][4][8])
                    k = ri * 8 + ci8
                    gb[k] = np.uint32(gate[src + ri * 8]) << np.uint32(16)
                    ub[k] = np.uint32(up[src + ri * 8]) << np.uint32(16)
            gv = gb.view(np.float32)
            uv = ub.view(np.float32)
            for k in range(64):
                g = gv[k]
                sb[k] = (g / (np.float32(1.0) + np.exp(-g))) * uv[k]
            su = sb.view(np.uint32)
            for k in range(64):
                u = su[k]
                out[obase + k] = np.uint16((u + np.uint32(0x7FFF)
                                            + ((u >> np.uint32(16))
                                               & np.uint32(1))) >> np.uint32(16))

    _NB = (_pack_a_u16, _swiglu_to_dn_a)
    return _NB


def pack_a_u16(xb_u16):
    """(rows, 3840) uint16 bf16, rows <= 1024 and a multiple of 32 -> A bytes."""
    k = _kernels()[0]
    import pack_kk as PK
    x = np.ascontiguousarray(xb_u16)
    rows = x.shape[0]
    if rows > KK["M"]:
        raise ValueError("kk_prod.pack_a_u16: %d rows > M=%d" % (rows, KK["M"]))
    if rows < KK["M"]:
        pad = np.zeros((KK["M"], x.shape[1]), np.uint16)
        pad[:rows] = x
        x = pad
    out = np.empty(A_BYTES, np.uint8)
    tm = np.array(PK.tile_map(KK["M"], KK["m"], KK["cols"], KK["cpc"]), np.int64)
    k(x.reshape(-1), tm, out.view(np.uint16), KK["m"], KK["kk"], KK["cols"],
      KK["cpc"], KK["K"] // KK["kk"], KK["K"])
    return out


# ---------------------------------------------------------------------------
# registration: exactly t18_run.Loop's own protocol
# ---------------------------------------------------------------------------
def install(dev):
    """Register the kk32 image on the product's device handle (idempotent)."""
    if IMG in dev.imgs:
        return dev.imgs[IMG]
    pyxrt = dev.pyxrt
    if not (XCLBIN.is_file() and INSTS.is_file()):
        raise RuntimeError("kk_prod: missing %s / %s" % (XCLBIN, INSTS))
    t0 = time.perf_counter()
    xc = pyxrt.xclbin(str(XCLBIN))
    uuid = dev.dev.register_xclbin(xc)
    kk = pyxrt.kernel(dev.dev, uuid, "MLIR_AIE")
    ib = INSTS.read_bytes()
    bo_i = pyxrt.bo(dev.dev, len(ib), pyxrt.bo.flags.cacheable, kk.group_id(1))
    bo_i.write(ib, 0)
    bo_i.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    im = {"k": kk, "inst": ib, "bo_i": bo_i, "xc": xc,
          "reg_s": time.perf_counter() - t0}
    dev.imgs[IMG] = im
    fl = pyxrt.bo.flags.host_only
    for nm, sz, gid in (("x", A_BYTES, 3), ("w", B_BYTES, 4),
                        ("o", C_BYTES, 5), ("o2", C_BYTES, 5)):
        dev.bo["b_%s_%s" % (IMG, nm)] = pyxrt.bo(dev.dev, sz, fl,
                                                 kk.group_id(gid))
    print("[kk32] %s registered (%s insts) in %.2f s -- A %s / B %s / C %s B"
          % (KK["tag"], format(len(ib), ","), im["reg_s"],
             format(A_BYTES, ","), format(B_BYTES, ","), format(C_BYTES, ",")),
          flush=True)
    return im


def _dispatch(dev, a_bo, b_bo, o_bo):
    im = dev.imgs[IMG]
    t0 = time.perf_counter()
    run = im["k"](3, im["bo_i"], len(im["inst"]) // 4, a_bo, b_bo, o_bo)
    st = int(run.wait(300000))
    wall = 1e3 * (time.perf_counter() - t0)
    if st != 4:
        raise RuntimeError("kk_prod %s state %d" % (KK["tag"], st))
    STATS["dispatch"] += 1
    STATS["t_disp_ms"] += wall
    STATS["walls"].append(wall)
    return wall


def _c_view(dev, bo):
    """uint16 view of a C BO -- mapped (zero copy) when the driver allows it."""
    pyxrt = dev.pyxrt
    bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    try:
        return np.frombuffer(bo.map(), np.uint16, count=C_BYTES // 2)
    except Exception:                                        # noqa: BLE001
        return np.frombuffer(bytes(bo.read(C_BYTES, 0)), np.uint16,
                             count=C_BYTES // 2)


def gu_kk32(dev, xb, key, dn_img, tim=None):
    """FFN gate/up for (rows, 3840) bf16 on the k-blocked image.

    Returns the down image's A payload (bytes, 21,626,880) -- the same contract
    the shipped swiglu image hands over through the device handoff, so the down
    dispatch is unchanged.

    Dispatch plan at SEQ=1056: 33 M-tiles > 32 slots, so the row axis is cut
    into 1024-row chunks ([1024, 32]) and each chunk runs gate then up.
    """
    import numpy as np
    t0 = time.perf_counter()
    install(dev)
    x = np.ascontiguousarray(xb)
    rows = x.shape[0]
    if rows % 32:
        raise RuntimeError("kk_prod: %d rows is not a multiple of the image's "
                           "m=32" % rows)
    gate_b, up_b = packed_b(key)
    bx = dev.bo["b_%s_x" % IMG]
    bw = dev.bo["b_%s_w" % IMG]
    bo_c = dev.bo["b_%s_o" % IMG]
    bo_c2 = dev.bo["b_%s_o2" % IMG]
    dto = dev.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    out = np.empty(SEQ * FF, np.uint16)          # 10,813,440 = 11x320x12x4x64
    kern = _kernels()[1]

    for r0 in range(0, rows, KK["M"]):
        seg = x[r0:r0 + KK["M"]]
        n = seg.shape[0]
        tw = time.perf_counter()
        ta = pack_a_u16(seg)
        mid = time.perf_counter()
        bx.write(ta.tobytes(), 0)
        bx.sync(dto)
        STATS["t_apack_ms"] += 1e3 * (mid - tw)
        STATS["t_write_ms"] += 1e3 * (time.perf_counter() - mid)
        for half, bpay, obo in ((0, gate_b, bo_c), (1, up_b, bo_c2)):
            if half == 0:
                tw = time.perf_counter()
                bw.write(bpay, 0)
                bw.sync(dto)
                STATS["t_write_ms"] += 1e3 * (time.perf_counter() - tw)
            else:
                tw = time.perf_counter()
                bw.write(bpay, 0)
                bw.sync(dto)
                STATS["t_write_ms"] += 1e3 * (time.perf_counter() - tw)
            w = _dispatch(dev, bx, bw, obo)
            if tim is not None:
                tim["dev"] = tim.get("dev", 0.0) + w
        th = time.perf_counter()
        g = _c_view(dev, bo_c)
        u = _c_view(dev, bo_c2)
        kern(g, u, out, r0, r0 + n, rows)
        STATS["t_host_ms"] += 1e3 * (time.perf_counter() - th)
    STATS["rows"] += rows
    STATS["calls"] += 1
    STATS["layers"].add(str(key))
    if os.environ.get("T22_FFN_KK32_PROF", "0") == "1":
        print("[kk32] %s rows=%d -> dn A in %.1f ms"
              % (key, rows, 1e3 * (time.perf_counter() - t0)), flush=True)
    return out.tobytes()
