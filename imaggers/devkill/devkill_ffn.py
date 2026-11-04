# SPDX-License-Identifier: MIT
"""devkill_ffn.py -- FFN segment with the dn W write overlapped with gu.

Two independent W BOs for the down image (W double buffer).  The write of the
NEXT payload starts the instant the gu dispatch is issued and is joined before
the dn dispatch, so the ~48 ms BO write that used to sit between gu and dn is
hidden under gu's ~70 ms of compute.

Flag: T22_DEVKILL_WDB=0 restores the serialised order (same process arm).
Nothing else changes: same images, same payloads, same arithmetic.
"""
from __future__ import annotations
import atexit, os, threading, time
import numpy as np

import prod_loop_bigm as L

_POOL = {}
#: DET-BISECT diagnostics (T22_PLAN_FFN_PROBE=1): one-shot markers, unused when off
_PROBED_H = [False]
_PROBED_Y = [False]

# ---------------------------------------------------------------------------
# T22_FFN_KK32=1 (default 0): the FFN gate/up runs on the k-blocked kk=32 GEMM
# (out/kk32_m32_n512.xclbin) instead of the shipped silu-fused swiglu image.
#
# Device-verified: 13.19 ms / 6.11 TFLOP/s vs the kk=8 ffn_gu_m32 19.70 ms /
# 4.09; corr 0.999915 against a numpy oracle built from the REAL layer-0 w1
# blocks (worst of 32 M-tiles 0.999913, A-tile-swap control -0.0077).  See
# imaggers/util/place/KK-GEMM-REPORT.md and out/kk_prod_check.json.
#
# The image is M=1024 (32 M-tiles of 32 rows) and SEQ is 1056 (= 32 caption +
# 32x32 image tokens = 33 M-tiles), so one layer-step needs 2 row chunks x 2
# halves (gate, up) = FOUR dispatches.  kk_prod.py owns that plan.
# ---------------------------------------------------------------------------
_KK32 = None


def _kk32():
    """kk_prod when T22_FFN_KK32=1 and it imports, else None (fail-open)."""
    global _KK32
    if _KK32 is None:
        if os.environ.get("T22_FFN_KK32", "0") != "1":
            _KK32 = False
        else:
            import sys, pathlib
            _p = pathlib.Path(__file__).resolve().parents[1] / "util" / "place"
            if str(_p) not in sys.path:
                sys.path.insert(0, str(_p))
            try:
                import kk_prod as _K
                atexit.register(_K.report)
                _KK32 = _K
                print("[devkill] T22_FFN_KK32=1: FFN gate/up -> the kk=32 "
                      "image %s (%d dispatches per layer-step)"
                      % (_K.KK["tag"], -(-_K.SEQ // _K.KK["M"]) * 2), flush=True)
            except Exception as _e:                          # noqa: BLE001
                print("[devkill] T22_FFN_KK32 unavailable, keeping the shipped "
                      "swiglu image: %r" % (_e,), flush=True)
                _KK32 = False
    return _KK32


def _layer_of(w_gu):
    """Which layer's BG payload is this?  Pointer into t18_run.CACHE_BG first.

    cached_payloads() hands out CACHE_BG[li] itself (a row view of the resident
    30-layer cache), so the byte offset from the cache base IS li -- exact, O(1)
    and with no product change.  The fallback scans the 30 stored w1 blobs'
    first 4 KiB, which is only paid if the resident cache is off.
    """
    T = L._t18()
    cb = getattr(T, "CACHE_BG", None)
    try:
        if cb is not None and isinstance(w_gu, np.ndarray):
            off = int(w_gu.ctypes.data) - int(cb.ctypes.data)
            row = int(cb.strides[0]) if cb.ndim > 1 else cb.size
            if 0 <= off < cb.nbytes and row > 0 and off % row == 0:
                return off // row
    except Exception:                                        # noqa: BLE001
        pass
    _mn = _kk32()
    if _mn:
        return _mn.layer_by_prefix(w_gu)
    raise RuntimeError("devkill: cannot identify the layer of w_gu")

_DEC = None


def _dec_kernel():
    # 2026-11-01 parent: fused transpose + bf16 widen in ONE pass.  The old
    # path materialised a 16 MB transposed uint16 buffer and then shifted it;
    # measured 9.70 -> 1.42 ms (6.81x) and byte-identical.
    global _DEC
    if _DEC is None:
        try:
            from numba import njit, prange

            @njit(parallel=True, cache=True)
            def _k(u, o, n_tiles):
                for r in prange(n_tiles * 96):
                    t = r // 96
                    rem = r % 96
                    d2 = rem // 8
                    d4 = rem % 8
                    for d1 in range(30):
                        for d3 in range(16):
                            c0 = (d1 * 16 + d3) * 8
                            for d5 in range(8):
                                o[r, c0 + d5] = np.uint32(u[t, d1, d2, d3, d4, d5]) << np.uint32(16)
            _DEC = _k
        except Exception:
            _DEC = False
    return _DEC or None


def decode_fast(raw, n_tiles, rows, _ref=None):
    """prod_loop_bigm._proj_decode with one allocation instead of two."""
    u = np.frombuffer(raw, np.uint16).reshape(n_tiles, 30, 12, 16, 8, 8)
    k = _dec_kernel()
    if k is not None:
        out = np.empty((n_tiles * 96, 3840), np.uint32)
        try:
            k(u, out, n_tiles)
            return out.view(np.float32)[:rows]
        except Exception:
            pass
    a = np.frombuffer(raw, np.uint16).reshape(n_tiles, 30, 12, 16, 8, 8)
    b = np.ascontiguousarray(a.transpose(0, 2, 4, 1, 3, 5)).reshape(n_tiles * 96, 3840)
    out = np.empty((n_tiles * 96, 3840), np.float32)
    np.left_shift(b, 16, out=out.view(np.uint32), dtype=np.uint32)
    return out[:rows]


def a_pack_fast(x):
    """Same bytes as prod_loop_bigm.a_pack, without the 6144-wide fancy index.

    a_order_96x64.npy is exactly [mb][ri][kb][ci] -> [mb][kb][ri][ci] (verified:
    np.arange(6144).reshape(12,8,8,8).transpose(0,2,1,3).reshape(-1) equals the
    saved order), so the permutation is a block transpose, not a gather.
    """
    x = np.asarray(x)
    rows, cols = x.shape
    assert rows % 96 == 0 and cols % 64 == 0, x.shape
    t = x.reshape(rows // 96, 12, 8, cols // 64, 8, 8).transpose(0, 3, 1, 4, 2, 5)
    return np.ascontiguousarray(t).tobytes()


def _alt_bo(dev, img, slot, nbytes):
    """A second BO for (image, arg-group 4) so a write never touches the BO the
    device is reading.  slot 0 is the product's own BO (dev.bo['b_<img>_w']).

    T22_CHAIN2B: the alternate BO is owned by the DEV (i.e. by the chain), not
    by this module.  The module-level _POOL is right for one chain and fatal for
    two: both chains would receive the SAME 44 MB down-W BO and the second
    chain's payload write would land between the first chain's write and the
    dispatch that reads it (STREAM2 sec.2.2 item 3 -- "one worker serialises two
    44 MB W writes" is the mild version; the fatal one is this cache).
    """
    if slot == 0:
        return dev.bo["b_%s_w" % img]
    pool = getattr(dev, "_alt_bo", None)
    if pool is None:
        pool = dev.__dict__["_alt_bo"] = {}
    key = (img, slot)
    if key not in pool:
        k = dev.imgs[img]["k"]
        pool[key] = dev.pyxrt.bo(dev.dev, nbytes,
                                 dev.pyxrt.bo.flags.host_only, k.group_id(4))
    return pool[key]


def _write_sync(dev, bo, payload, log=None):
    t0 = time.perf_counter()
    bo.write(payload, 0)
    bo.sync(dev.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    if log is not None:
        log.append(1e3 * (time.perf_counter() - t0))


# ---------------------------------------------------------------------------
# T22_FFN_TP=1: the dn-W write runs on ONE module-level ThreadPoolExecutor
# instead of a fresh threading.Thread per layer-step.
#
# block() below issues the gu dispatch, then starts a thread that writes the
# 44 MB dn W payload (hidden under gu's compute), then joins it before the dn
# dispatch.  The THREAD MACHINERY -- Thread() + start() + join() -- is the whole
# cost a persistent worker removes; the write itself and the ordering
# constraint are unchanged (the joiner is still called before the dn submit).
# Same pattern as t18_run's T22_PFILL (_PFILL pool).
#
# One behavioural difference, on purpose: Future.result() RE-RAISES an exception
# from the write, while thread.join() would swallow it and the run would go on
# with a half-written W.  A failed write must fail the run.
# Default 0 = the shipped per-layer Thread.
_TP = {"pool": None}


# ---------------------------------------------------------------------------
# T22_OVL_WGU (2026-11-02, default OFF; on whenever T22_OVL=1): write the NEXT
# layer's gu W payload into the gu W BO while THIS layer's dn dispatch runs.
#
# WHY IT IS SAFE, in one paragraph.  The gu W payload (88.5 MB) is a per-layer
# constant: it does not depend on any activation.  In the shipped order it is
# written on the main thread immediately before the gu dispatch, i.e. 4.4 ms of
# the layer's critical path that no dispatch covers.  The one interval in the
# layer where the device is busy and the host is idle AND the gu W BO is free is
# the dn dispatch: the gu dispatch that reads bw has already been waited for,
# and dn reads bwd (its own alt BO) and its own output BO.  So the write goes
# there, on the main thread, between the dn submit and the dn wait.  The next
# reader of bw is the next layer's gu dispatch, which happens after this call
# returns (same thread, program order).  The hit test at the top of block() is
# (address, nbytes) of the payload actually handed in, so a payload that is not
# the one we pre-wrote is still written normally -- the flag can only skip a
# write it can prove is already there.
# ---------------------------------------------------------------------------
_WGU = {"on": None, "ptr": 0, "nb": 0, "n": 0, "miss": 0}


def _ovl_wgu():
    if _WGU["on"] is None:
        _WGU["on"] = (os.environ.get("T22_OVL", "0") == "1"
                      and os.environ.get("T22_OVL_WGU", "1") != "0")
        if _WGU["on"]:
            print("[devkill] T22_OVL_WGU=1: layer i+1's gu W payload is "
                  "written into the gu W BO under layer i's dn dispatch",
                  flush=True)
            atexit.register(_wgu_report)
    return _WGU["on"]


def _wgu_report():
    print("[devkill] T22_OVL_WGU: %d hoist(s), %d not-hoistable (last layer "
          "of a step or no resident cache)" % (_WGU["n"], _WGU["miss"]),
          flush=True)


def _wgu_pre_hit(g_img, w_gu):
    """True iff THIS payload is already in THIS image's gu W BO."""
    return (bool(_ovl_wgu()) and bool(_WGU["ptr"])
            and isinstance(w_gu, np.ndarray)
            and int(w_gu.ctypes.data) == _WGU["ptr"]
            and int(w_gu.nbytes) == _WGU["nb"])


def _next_gu_payload(w_gu):
    """The next layer's gu W payload, or None (fail open).

    cached_payloads() hands out CACHE_BG[li] itself -- a row view of the
    resident 30-layer cache -- so the byte offset from the cache base IS li and
    the next row is the next layer's payload, already in image order.
    """
    T = L._t18()
    cb = getattr(T, "CACHE_BG", None)
    if (cb is None or not isinstance(cb, np.ndarray) or cb.ndim != 2
            or not isinstance(w_gu, np.ndarray)):
        return None
    row = int(cb.strides[0])
    off = int(w_gu.ctypes.data) - int(cb.ctypes.data)
    if row <= 0 or off < 0 or off % row or off >= cb.nbytes:
        return None
    li = off // row
    if li + 1 >= int(cb.shape[0]):
        return None
    return cb[li + 1]


def _wgu_hoist(dev, g_img, w_gu):
    nxt = _next_gu_payload(w_gu)
    if nxt is None:
        _WGU["miss"] += 1
        return 0
    bw = dev.bo["b_%s_w" % g_img]
    bw.write(nxt, 0)
    bw.sync(dev.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    _WGU["ptr"] = int(nxt.ctypes.data)
    _WGU["nb"] = int(nxt.nbytes)
    _WGU["n"] += 1
    return 1


def _start_write(db_on, dev, bwd, b32, wlog):
    """Start the dn-W write off-thread; return a zero-arg joiner, or None."""
    if not db_on:
        return None
    if os.environ.get("T22_FFN_TP", "0") == "1":
        # T22_CHAIN2B: one pool per DEV.  A shared single worker would serialise
        # the two chains' dn-W writes against each other; correctness survives
        # either way once _alt_bo is per-dev, so this is a throughput choice.
        pool = getattr(dev, "_ffn_pool", None)
        if pool is None:
            import concurrent.futures as _cf
            pool = _cf.ThreadPoolExecutor(max_workers=1,
                                          thread_name_prefix="ffn-wdn")
            dev.__dict__["_ffn_pool"] = pool
        return pool.submit(_write_sync, dev, bwd, b32, wlog).result
    th = threading.Thread(target=_write_sync, args=(dev, bwd, b32, wlog),
                          daemon=True)
    th.start()
    return th.join


def _dispatch(dev, name, *args):
    im = dev.imgs[name]
    t0 = time.perf_counter()
    run = im["k"](3, im["bo_i"], len(im["inst"]) // 4, *args)
    st = int(run.wait(300000))
    wall = 1e3 * (time.perf_counter() - t0)
    if st != 4:
        raise RuntimeError("devkill %s state %d" % (name, st))
    return wall


def block(dev, xb, w_gu, w_b3, pair, tim=None, slot=0, dbg=True, nreal=None):
    """One (gu+swiglu) -> down block.  xb is bf16 (rows, 3840), rows % 96 == 0.

    nreal is how many of those rows are REAL activations; the rest of the last
    row tile is the zero pad of a partial M (384^2: M=608 = 6*96 + 32, so a
    dispatch carries 672 rows of which 608 are real).  It is used for two
    things and nothing else: the rows= field of the [devkill] line reports the
    M the layer really ran at (postflight Q06 gates that field against the
    plan's M, so it must be 608 and not 672), and the pad count is printed next
    to it.  The arithmetic never reads the pad rows: the caller crops the
    returned y to the real rows.
    """
    t0 = time.perf_counter()
    g_img, d_img = pair
    rows = xb.shape[0]
    n_real = rows if nreal is None else int(nreal)
    kkp = _kk32()
    if kkp:
        # T22_FFN_KK32: the kk=32 image owns the A pack, the B payload and the
        # gate/up dispatches; the shipped swiglu image is not dispatched at all.
        x = None
        tp = time.perf_counter()
        tw = tp
        bx = bw = None
        dto = dev.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    else:
        x = a_pack_fast(xb) if os.environ.get("T22_DEVKILL_PACK", "1") != "0" else L.a_pack(xb)
        tp = time.perf_counter()
        bx = dev.bo["b_%s_x" % g_img]
        bw = dev.bo["b_%s_w" % g_img]
        bx.write(x, 0)
        # T22_OVL_WGU: the previous layer's dn dispatch already put THIS
        # layer's payload into bw and synced it; the hit test is on the payload
        # identity, so this only ever skips a write it can prove is there.
        if not _wgu_pre_hit(g_img, w_gu):
            bw.write(w_gu, 0)
        dto = dev.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        bx.sync(dto); bw.sync(dto)
        tw = time.perf_counter()
    if isinstance(w_b3, dict):
        if "n32" in w_b3:
            b32 = w_b3["n32"]
        elif "raw" in w_b3:
            from prod_bigborder import b3_to_image_k32
            b32 = b3_to_image_k32(w_b3["raw"]).tobytes()
        else:
            raise RuntimeError("devkill needs n32 B payload")
    else:
        b32 = w_b3
    bwd = _alt_bo(dev, d_img, slot, len(b32))
    db_on = os.environ.get("T22_DEVKILL_WDB", "1") != "0"
    wlog = []
    join_w = None
    tg0 = time.perf_counter()
    if kkp:
        # the kk=32 gate/up: 4 dispatches (2 row chunks x gate,up) + the fused
        # silu(g)*u pass that lands in the down image's A layout.  The dn W
        # write overlaps it exactly as it did with the shipped gu dispatch.
        join_w = _start_write(db_on, dev, bwd, b32, wlog)
        key = "layers.%d" % _layer_of(w_gu)
        _a = kkp.gu_kk32(dev, xb, key, d_img, tim)
        og = dev.bo["b_%s_o" % g_img]
        og.write(_a, 0)
        og.sync(dto)
        del _a
        wg = 1e3 * (time.perf_counter() - tg0)
        if join_w is not None:
            join_w()
        tbw = time.perf_counter()
        td = tbw
    else:
        # issue gu first, then write the dn W while it runs
        im = dev.imgs[g_img]
        run = im["k"](3, im["bo_i"], len(im["inst"]) // 4,
                      bx, bw, dev.bo["b_%s_o" % g_img])
        join_w = _start_write(db_on, dev, bwd, b32, wlog)
        st = int(run.wait(300000))
        wg = 1e3 * (time.perf_counter() - tg0)
        if st != 4:
            raise RuntimeError("devkill gu state %d" % st)
        if join_w is not None:
            join_w()
        tbw = time.perf_counter()
        if not db_on:
            _write_sync(dev, bwd, b32, wlog)
        td = time.perf_counter()
        og = dev.bo["b_%s_o" % g_img]
    # T22_OVL_WGU: submit dn FIRST, then write the NEXT layer's gu W payload
    # while dn runs, then wait.  dn reads bwd and its own output BO; the gu W
    # BO is free (this layer's gu was waited for above).  wgn is charged
    # separately so pack+wgu+gu+wdn+dn+read+dec+wgn still closes to total.
    im_d = dev.imgs[d_img]
    # DIAGNOSTIC (DET-BISECT, default OFF): the campaign arm chains the swiglu
    # image's C straight into the down image's A, so it never has a host copy of
    # h = silu(gate)*up.  Read it here (one shot, T22_PLAN_FFN_PROBE=1) so h can
    # be compared with the plan arm's h and the 36.7x y difference attributed.
    if (os.environ.get("T22_PLAN_FFN_PROBE", "0") == "1" and not _PROBED_H[0]):
        _PROBED_H[0] = True
        try:
            import hashlib as _hlp2
            og.sync(dev.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            _nb = rows * 10240 * 2
            _rawg = bytes(og.read(_nb, 0))
            _u16 = np.frombuffer(_rawg, np.uint16)
            _hf = (_u16.astype(np.uint32) << np.uint32(16)).view(np.float32)
            print("[pffnprobe] devkill-guC rows=%d bytes=%d u16_absmax=%d "
                  "h_absmax=%.6g h_sha=%s" % (
                      rows, _nb, int(_u16.max()),
                      float(np.abs(_hf).max()),
                      _hlp2.sha256(memoryview(_u16).cast("B")).hexdigest()[:16]),
                  flush=True)
            del _rawg, _u16, _hf
        except Exception as _e:                                  # noqa: BLE001
            print("[pffnprobe] devkill-guC probe failed: %r" % (_e,), flush=True)
    _rund = im_d["k"](3, im_d["bo_i"], len(im_d["inst"]) // 4,
                      og, bwd, dev.bo["b_%s_o" % d_img])
    _tw2 = time.perf_counter()
    wgn = 0.0
    if _ovl_wgu() and not kkp:
        _t1 = time.perf_counter()
        _wgu_hoist(dev, g_img, w_gu)
        wgn = 1e3 * (time.perf_counter() - _t1)
    _st = int(_rund.wait(300000))
    wd = 1e3 * (time.perf_counter() - _tw2) - wgn
    if _st != 4:
        raise RuntimeError("devkill dn state %d" % _st)
    tdn = time.perf_counter()
    # BUGFIX (2026-11-03, DET): no XCL_BO_SYNC_BO_FROM_DEVICE before this read
    # (t18_run.Dev.dispatch syncs every out_key; this fast path bypasses it).
    # Same defect class as prod_devkill._run -- see the note there.
    _ob = dev.bo["b_%s_o" % d_img]
    _ob.sync(dev.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    raw = bytes(_ob.read(rows * 3840 * 2, 0))
    tr = time.perf_counter()
    y = decode_fast(raw, rows // 96, rows)
    if os.environ.get("T22_PLAN_FFN_PROBE", "0") == "1" and _PROBED_H[0] \
            and not _PROBED_Y[0]:
        _PROBED_Y[0] = True
        import hashlib as _hlp3
        _yb = np.ascontiguousarray(np.asarray(y, np.float32))
        print("[pffnprobe] devkill-y rows=%d absmax=%.6g sha=%s" % (
            rows, float(np.abs(_yb).max()),
            _hlp3.sha256(memoryview(_yb).cast("B")).hexdigest()[:16]), flush=True)
        _pd = os.environ.get("T22_PLAN_FFN_PROBE_DUMP", "")
        if _pd:
            np.save(_pd + "_campaignty.npy", _yb)
            print("[pffnprobe] devkill-y dumped to %s_campaignty.npy" % _pd,
                  flush=True)
    if dbg:
        # rows = the REAL M of this layer (Q06: '[devkill] rows=<M> pack='),
        # pad = the zero rows of the last row tile (0 at 96 and 1056 rows).
        print("[devkill] rows=%d pack=%.1f wgu=%.1f gu=%.1f wdn=%.1f dn=%.1f read=%.1f dec=%.1f total=%.1f ms (wdb=%s wgn=%.1f pad=%d)"
              % (n_real, 1e3*(tp-t0), 1e3*(tw-tp), wg, 1e3*(td-tbw), wd,
                 1e3*(tr-tdn), 1e3*(time.perf_counter()-tr),
                 1e3*(time.perf_counter()-t0), "on" if db_on else "off",
                 wgn, rows - n_real), flush=True)
    if tim is not None:
        tim["dev"] = tim.get("dev", 0.0) + wg + wd
    return y


def bigm_ffn(dev, xm, swiglu_w, b3_w, tim=None, imgs=("swiglu", "down"),
             tail_imgs=("swiglu1", "down1")):
    """Drop-in for prod_loop_bigm.bigm_ffn with the W double buffer.

    ONE image pair for the WHOLE call, INCLUDING a partial last row tile.

    tail_imgs is accepted for call compatibility and NOT USED.  Measured at
    384^2 (M = 608 = 6*96 + 32, 8-step campaign arm tag p8camp8_384): the old
    code sent rows [576, 608) on their own through the RT=1 pair
    ("swiglu1", "down1"), and that dispatch is not the same contract:

      * A argument.  devkill chains the swiglu C drain straight into the down
        image (og -> dn).  That is only valid for a k=32 image, whose declared
        A layout equals mm_bfp's C layout byte for byte
        (prod_loop_bigm._ffn_block: "A(mm_bfp,k=32) == C(mm_bfp)").  down1 is
        ffn_dn_rt1, whose A is the PACKED A plane: the product untiles C at
        (96, n2g=32), transposes it object-major and a_packs it before that
        dispatch (_ffn_block's "the swiglu output must be A2-packed before the
        down image consumes it").  Feeding it the raw drain scrambles the tail.
        br_tail_gate.py gates exactly this pair and only reaches corr 0.999890
        WITH that untile+a_pack (pad32 arm), i.e. the pair is fine, the
        handoff into it was not.
      * B payload.  cached_payloads() hands out {"n32": ...} under
        T22_BIGM_DEVHANDOFF=1 (prod_loop_bigm.py:650-651), the k=32 order that
        dn11 consumes; ffn_dn_rt1 wants the n128 order (b3_to_image).

    So the 32 tail rows came out wrong, and attention carried them into every
    token over 30 layers: campaign vs delivered 0.560995 at 8 steps while the
    plan arm -- which dispatches [[0, M]] and pads INSIDE the tile -- measured
    0.992750 (DET-BISECT-REPORT-2 6.2, 5).

    The fix is the contract the other two working arms already use, verbatim:
    pad the last tile with ZEROS and run the SAME pair once, then crop the
    drain to the real rows -- exactly what the >_chunk branch below does, what
    prod_loop_bigm.bigm_ffn's own chunk loop does, what the delivered path does
    ("[t18] ... ffn M=96 x 7 block(s)") and what the plan does ([[0, 608]]).
    The pad rows are zeros, so the FFN is row-local and tile 6's real rows are
    computed by the same kernel, with the same A pack and the same B payload,
    as tiles 0..5.

    Nothing changes where seq is a multiple of 96 (128^2 M=96, 512^2 M=1056):
    rem == 0 takes the same single dispatch with the same bytes as before.
    """
    seq = xm.shape[0]
    # t22_time1024: chunk by the IMAGE's own row capacity.  The RT=11 pair
    # (swiglu11/dn11) holds 1056 rows per dispatch; the RT=17 pair holds 1632.
    # The old hard-coded 1632 wrote a 17-tile X into an 8.1 MB RT=11 BO, which
    # is the "1056-row chunking pit" of the 1024^2 (M=4128 = 43 tiles) shape.
    _rt = 11 if str(imgs[0]).endswith("11") else 17
    _chunk = _rt * 96
    if seq > _chunk:
        # Fill a preallocated drain chunk by chunk.  This used to keep a
        # .copy() of every chunk alive and then np.concatenate() them,
        # which needs a SECOND full (seq, 3840) f32 while all the parts
        # are still there: 120.00 MiB of transient at M=8192.  The
        # assignment into _out IS the copy the .copy() did (decode
        # returns a view into a reused scratch, so the previous chunk must
        # be out of it before the next call).  Same fix as install.py:827.
        _out = np.empty((seq, xm.shape[1]), np.float32)
        for r0 in range(0, seq, _chunk):
            seg = xm[r0:r0 + _chunk]
            rows = seg.shape[0]
            nt = (rows + 95) // 96
            pad = np.zeros((nt * 96, seg.shape[1]), np.float32)
            pad[:rows] = seg
            u = pad.view(np.uint32)
            xb = ((u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))) >> 16).astype(np.uint16)
            _out[r0:r0 + rows] = block(dev, xb, swiglu_w, b3_w, imgs,
                                       tim, slot=0, nreal=rows)[:rows]
        return _out
    rt, rem = divmod(seq, 96)
    u = np.asarray(xm, np.float32).view(np.uint32)
    xb = ((u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))) >> 16).astype(np.uint16)
    if not rem:
        return block(dev, xb, swiglu_w, b3_w, imgs, tim, slot=0)
    pad = np.zeros(((rt + 1) * 96, xb.shape[1]), np.uint16)
    pad[:seq] = xb
    return block(dev, pad, swiglu_w, b3_w, imgs, tim, slot=0, nreal=seq)[:seq]