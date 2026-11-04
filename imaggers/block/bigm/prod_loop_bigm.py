# SPDX-License-Identifier: MIT
"""prod_loop_bigm.py -- step 4 wiring, as a DROP-IN module (not an edit of t18_run).

WHY A MODULE AND NOT A PATCH: t18_run.py is 136 KB of product code and the
four-dispatch path cannot be exercised until blocker 1 (the gu+swiglu image) and
blocker 2 (the repacked weights) exist.  A module that t18_run.Loop imports keeps
the change reversible by ONE line and keeps the unverified path out of the product
until its gates pass -- the same rule this line has applied four times.

PLUG-IN POINT (the only edit t18_run.py would need):

    import prod_loop_bigm
    Loop.attention = ...                       # unchanged
    if os.environ.get("ZIMAGE_BIGM") == "1" and prod_loop_bigm.ready():
        Loop.block_step = prod_loop_bigm.block_step

Rollback: `ZIMAGE_BIGM=0` (the default), or delete the two lines.

THE FOUR DISPATCHES (prod_wire.py --dry-run prints the same table):
    1 nc_s3a            norm -> A-plane                     20.655 ms
    2 bigm_proj x4      q,k,v,o                             49.260 ms
    3 bigm_ffn_gu+swiglu gate/up + swiglu                   88.550 ms
    4 bigm_ffn_dn+residual down + residual                  39.251 ms
                                                           197.7 ms/block-step
"""
from __future__ import annotations
import os, pathlib, time
import pathlib

import numpy as np
_np = np

_T18 = None


def _t18():
    """The product module, imported lazily so this file stays CPU-importable."""
    global _T18
    if _T18 is None:
        import t18_run as _T
        _T18 = _T
    return _T18

HERE = pathlib.Path(__file__).resolve().parent
NPU = HERE.parents[2]

# ---- the layout contract, in one place (PRODASM-STEP345 9 and 10)
TILE = 576
B_OBJ = 9216
TAP_SIZES = (8, 4, 72)
TAP_STRIDES = (1152, 72, 1)


def tile_a(x: np.ndarray, m: int, k: int) -> np.ndarray:
    """A micro-tile layout -- MEASURED as A2 (corr 0.999975, atile gate)."""
    return np.ascontiguousarray(
        x.reshape(m // 8, 8, k // 8, 8).transpose(0, 2, 1, 3).reshape(m, k))


def untile_c(x: np.ndarray, m: int, n: int) -> np.ndarray:
    """C inverse -- the SAME transform as tile_a, applied on read-back."""
    return np.ascontiguousarray(
        x.reshape(m // 8, n // 8, 8, 8).transpose(0, 2, 1, 3).reshape(m, n))


def swiglu_slices(obj: np.ndarray, s: int, t: int, half: str) -> np.ndarray:
    """one 288 B piece of a 9,216 B object, by the LOCKED tap formula.

    gate: s*1152 + t*72        up: s*1152 + 576 + t*72
    s = k/8 block 0..7 (outer), t = n/8 block 0..3 (inner).
    A contiguous 2,304 B read does NOT give a gate/up pair -- the container is
    kk-major while the object is _block_layout, so the two differ by a sub-tile
    transpose.  Always address by this formula.
    """
    base = s * 1152 + (576 if half == "up" else 0) + t * 72
    return obj[base:base + 72]


# ---------------------------------------------------------------------------
# THE FOUR HARD FACTS this wiring must honour (all measured, not advisory)
# ---------------------------------------------------------------------------
# 1. ONE acquire, TWO views. A second acquire(1) on the same fifo returns the
#    SAME element (the contract pads to num_elem), so two objects per k-group
#    plus two acquires makes both matmuls read one B block. The gu+swiglu arm
#    therefore puts gate|up in ONE 4,608 B object and takes two constant-offset
#    views of it.
# 2. SACRIFICIAL ZERO. The first kernel call in a device loop body only takes
#    effect on the first iteration, so the C clear needs a throwaway zero() in
#    front of it. L1 accounting is unchanged.
# 3. C MUST BE UNTILED. The device writes C in the mmul tile layout; reading it
#    raw gives a perfect value multiset and corr 0.054. untile_c() is the
#    inverse, and A is tiled with the SAME transform (atile gate: A2, 0.999975).
# 4. B SLICES BY THE TAP FORMULA, NEVER BY CONTIGUITY. gate = s*1152 + t*72,
#    up = s*1152 + 576 + t*72, s = k/8 outer, t = n/8 inner. A contiguous
#    2,304 B read is NOT a gate/up pair: the container is kk-major while the
#    object is _block_layout, so the two differ by a sub-tile transpose.
#
# WHAT block_step() STILL NEEDS FROM t18_run (it cannot invent these):
#   - the per-block order: norm, q/k/v projections, attention, o, FFN
#   - the A-plane handoff (norm output) and the residual carries
#   - the attention itself (host f32 today, or the device path)
#   - the latents buffer and the step schedule
# Until those are supplied, raising is the honest behaviour.
REQUIRED = {
    "images": ["bigm_proj", "bigm_ffn_gu", "bigm_ffn_dn", "nc_s3a"],
    "weights": ["L0", "L1", "L2"],          # 30 in the product
    "reference": "455ad0e9",
}


def missing() -> dict:
    """what is absent right now -- the same list prod_wire.py refuses on."""
    cache = NPU / "webui" / "cache"
    have_img = [i for i in REQUIRED["images"]
                if any(cache.glob("*%s*" % i))] if cache.is_dir() else []
    work = HERE / "work"
    have_w = [w for w in REQUIRED["weights"] if (work / ("%s.npy" % w)).is_file()]
    return {"images_have": have_img,
            "images_missing": [i for i in REQUIRED["images"] if i not in have_img],
            "weights_have": have_w,
            "weights_missing_count": len(REQUIRED["weights"]) - len(have_w),
            "reference": REQUIRED["reference"]}


def ready() -> bool:
    m = missing()
    return not m["images_missing"] and m["weights_missing_count"] == 0


def block_step(self, *args, **kwargs):
    """Drop-in for Loop.block_step.  Refuses unless ready()."""
    if not ready():
        raise RuntimeError("prod_loop_bigm: not ready -- %s" % missing())
    raise NotImplementedError(
        "the four dispatches are specified and gated but the gu+swiglu image "
        "and the repacked weights do not exist yet; see PRODASM-STEP345 11")


if __name__ == "__main__":
    import json
    print(json.dumps(missing(), indent=1))
    print("ready:", ready())

# ---------------------------------------------------------------------------
# bigm_ffn: the FFN segment on device (gu+swiglu -> down), replacing L.ffn.
# ---------------------------------------------------------------------------
# contract: y = bigm_ffn(...) must be (SEQ, D) float32, because block_step does
#     p = rms_into(y, ffn_norm2);  out = xm + mod[3] * p
# and nothing else with it (t18_run.py:1858-1879).
def untile_obj(raw, m, n2):
    """undo the drain layout [ty][tx][r][c] -> [ty][r][tx][c] (ELEMENTS, not bytes)"""
    import numpy as _np
    a = _np.frombuffer(raw, _np.uint16)
    return a.reshape(-1, m // 8, n2 // 8, 8, 8).transpose(0, 1, 3, 2, 4).reshape(-1).tobytes()


def _dispatch_dev(dev, name, fills, out_name, tag=""):
    """Run one registered image and DO NOT read its output BO.

    t22_bigpipe3 device handoff: the gu image's output BO is handed
    straight to the k=32 down image as its A argument, so the 33.4 MB
    swiglu intermediate never goes through host memory.
    """
    im = dev.imgs[name]
    args = list(fills) + [dev.bo[out_name]]
    t0 = time.perf_counter()
    run = im["k"](3, im["bo_i"], len(im["inst"]) // 4, *args)
    st = int(run.wait(300000))
    wall = 1e3 * (time.perf_counter() - t0)
    if st != 4:
        raise RuntimeError("bigm devhandoff %s state %d" % (name, st))
    return wall


_A_PERM = None


def a_pack(x):
    """(rows, cols) bf16 -> the A payload the big-M images actually declare.

    NOT prod_aplane.a_plane.  That one ships PLAIN=True, which is the IDENTITY
    per (96,64) tile, and its comment justifies that with t18_run:1694-1755 --
    which is the DELIVERED FFN image's A, not this image's.  mm_bfp declares an
    A micro-tile (sw_pack.py builds X with a_layout.pack and asserts the round
    trip); measured here: pack is NOT the identity (9,786 of 12,288 bytes move,
    5,430 of 6,144 positions displaced) and a_plane is.

    No gate could see this: all-ones is permutation-invariant, and sw_pack's
    reference tiles ONE packed tile, so a tile-order error is invisible there
    too.  The permutation is frozen in ab480/evidence/a_perm_96x64.npy (py312
    only -- mlir_aie does not exist on the py310 device leg).
    """
    global _A_PERM
    import numpy as _np
    if _A_PERM is None:
        # t22_breadout: a_perm_96x64.npy (ab480) is NOT the declared layout --
        # prod_bigborder.selftest shows a_pack != a_layout.pack for all 60 tiles,
        # and the device agrees with the library (down A: plain 0.375111 ->
        # a_layout.pack 0.999914).  a_order_96x64.npy is derived from
        # a_layout.pack with 13 binary probes (a_order_gen.py) and re-verified.
        _A_PERM = _np.load(pathlib.Path(__file__).resolve().parent
                           / "a_order_96x64.npy").astype(_np.int64)
    x = _np.asarray(x)
    rows, cols = x.shape
    assert rows % 96 == 0 and cols % 64 == 0, x.shape
    t = x.reshape(rows // 96, 96, cols // 64, 64).transpose(0, 2, 1, 3)
    return _np.ascontiguousarray(t.reshape(rows // 96, cols // 64, -1)[..., _A_PERM]).tobytes()


def _ffn_block(dev, xb, w_gu, w_b3, pair, tim=None):
    """One 96*k-row block through (gu+swiglu) -> down.

    xb is bf16 (rows, 3840) with rows a multiple of 96; pair names the image
    pair whose row-tile count matches those rows.  Returns (rows, 3840) f32.
    """
    import numpy as _np
    from prod_aplane import a_plane

    g_img, d_img = pair
    rows = xb.shape[0]
    if os.environ.get("T22_BIGM_DEVHANDOFF", "0") == "1":
        # ---- t22_bigpipe3: device-resident swiglu handoff ----------------
        # gu's output BO is down's A argument.  A(mm_bfp,k=32) == C(mm_bfp)
        # byte for byte (laycmp), so no host untile/a_pack is needed.  The
        # down image is compiled at k=32 for exactly this reason.
        import time as _t
        _t0 = _t.perf_counter()
        gi, di = ("swiglu1", "dn32_1") if rows <= 96 else ("swiglu", "dn32")
        x = a_pack(xb)
        _tp = _t.perf_counter()
        bx = dev.bo["b_%s_x" % gi]
        bw = dev.bo["b_%s_w" % gi]
        bx.write(x, 0)
        bw.write(w_gu, 0)
        dir_to = dev.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        bx.sync(dir_to)
        bw.sync(dir_to)
        _tw = _t.perf_counter()
        wg = _dispatch_dev(dev, gi, [bx, bw], "b_%s_o" % gi, tag="gu")
        _tg = _t.perf_counter()
        if isinstance(w_b3, dict):
            if "n32" in w_b3:
                b32 = w_b3["n32"]
            elif "raw" in w_b3:
                from prod_bigborder import b3_to_image_k32
                b32 = b3_to_image_k32(w_b3["raw"]).tobytes()
            else:
                raise RuntimeError("devhandoff needs n32 B payload")
        else:
            b32 = w_b3
        _tb = _t.perf_counter()
        bwd = dev.bo["b_%s_w" % di]
        bwd.write(b32, 0)
        bwd.sync(dir_to)
        _tbw = _t.perf_counter()
        wd = _dispatch_dev(dev, di, [dev.bo["b_%s_o" % gi], bwd],
                           "b_%s_o" % di, tag="dn")
        _td = _t.perf_counter()
        if tim is not None:
            tim["dev"] = tim.get("dev", 0.0) + wg + wd
        # BUGFIX (2026-11-03, DET): no FROM_DEVICE sync before this read
        _bo = dev.bo["b_%s_o" % di]
        _bo.sync(dev.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        raw = bytes(_bo.read(rows * 3840 * 2, 0))
        _tr = _t.perf_counter()
        y = _proj_decode(raw, rows // 96, rows)
        if os.environ.get("T22_BIGM_DEBUG", "0") == "1":
            print("[devhand] rows=%d pack=%.1f wgu=%.1f gu=%.1f b3=%.1f wdn=%.1f dn=%.1f read=%.1f dec=%.1f ms"
                  % (rows, 1e3*(_tp-_t0), 1e3*(_tw-_tp), wg, 1e3*(_tbw-_tb),
                     1e3*(_td-_tbw), wd, 1e3*(_tr-_td),
                     1e3*(_t.perf_counter()-_tr)), flush=True)
        return y
    # T22_BIGM_KG: the k-group width the image was compiled for.  prod_aplane
    # defaults to 64 (the k=64 generation), but PRODASM-STEP345 110/123 record
    # that the PRODUCT instantiates every mm_bfp with dim_k=8 and that k=64
    # makes each call compute only 1/8 of the k axis.  The two choices have the
    # SAME byte count (ng*kg is constant), so only a numeric gate can tell them
    # apart -- hence the knob.
    # t22_breadout: the DECLARED A micro-tile, not plain rows.  ab480 4.13's
    # "a_pack tested, no change" compared it against a payload with the wrong B
    # order, so that reading was void; with the B order fixed the A convention is
    # worth 0.375111 -> 0.999914 on the down stage.
    x = a_pack(xb)
    r1 = dev.dispatch(g_img,
                      [(dev.bo['b_%s_x' % g_img], x),
                       (dev.bo['b_%s_w' % g_img], w_gu)],
                      [(dev.bo['b_%s_o' % g_img], rows * 10240 * 2)])
    # t22_breadout: per-hop dump (gu out / down in / y) so rt=17 can be compared
    # hop by hop against rt=1 at the same layer.  rt=1 is the 0.9999 baseline;
    # rt=17 is the only variable.  The gu IMAGE differs (sw64d2 vs sw_rt1) even
    # though the geometry is the same, so "0.999911 must not be carried across
    # images" applies here too.
    global _HOP_DONE
    _hop = os.environ.get("T22_DUMP_HOP", "")
    _hop_here = bool(_hop) and not _HOP_DONE
    if _hop_here:
        _HOP_DONE = True
        import pathlib as _pl
        _pl.Path(_hop + "_" + d_img + "_gu.bin").write_bytes(bytes(r1[0][0][:1966080]))
    # the swiglu output must be A2-packed before the down image consumes it --
    # the down image's x is an A-plane (one micro-tile per (rt, kg)), not the
    # logical (96, 10240) matrix.  Skipping this is a silent numeric error.
    # t22_breadout: untile_obj returns the drained objects OBJECT-MAJOR
    # (nrt, HALF*G, 96, 32), and the gu image drains object q = h*G + i carrying
    # output columns 32*q.  reshape(rows, 10240) treated that flat buffer as
    # row-major, which scrambles the gu output before it becomes the down A --
    # measured: gu stage 0.999875, whole FFN -0.000961, and with this transpose
    # the FFN goes to 0.9998.  This hop is the one AB480 4.15 listed as never
    # covered by any gate.
    s_obj = _np.frombuffer(untile_obj(r1[0][0], 96, 32), _np.uint16)
    s_logical = s_obj.reshape(rows // 96, 320, 96, 32).transpose(0, 2, 1, 3).reshape(
        rows, 10240)
    s = a_pack(s_logical)
    if _hop_here:
        _pl.Path(_hop + "_" + d_img + "_downin.bin").write_bytes(bytes(s[:737280]))
    if tim is not None:
        tim['dev'] += r1[1]
    # the two big-M down images do not share a B3 order (n2=128/HALF=3 vs
    # n2=64/HALF=6); pick by image, and convert on demand when dual residency
    # is off (T22_BIGM_B3_DUAL=0)
    if isinstance(w_b3, dict):
        if "raw" in w_b3:
            from prod_bigborder import b3_to_image, b3_to_image_n64
            b3_sel = (b3_to_image_n64(w_b3["raw"]) if d_img == "down"
                      else b3_to_image(w_b3["raw"])).tobytes()
        else:
            b3_sel = w_b3["n64" if d_img == "down" else "n128"]
    else:
        b3_sel = w_b3
    r2 = dev.dispatch(d_img,
                      [(dev.bo['b_%s_x' % d_img], s),
                       (dev.bo['b_%s_w' % d_img], b3_sel)],
                      [(dev.bo['b_%s_o' % d_img], rows * 3840 * 2)])
    # t22_genside: the object geometry is PER IMAGE and BOTH the object-major
    # transpose AND the untile width depend on it.  The untile used to be the
    # literal 128 here, which is ffn_dn_rt1's width: on the n64 image
    # (ffn_dn_n64, the 480p arm) the drain is 60 objects of 96x64, so an untile
    # at 96x128 re-groups the stream in pairs of objects and displaces 87.5 % of
    # the elements (measured: corr(shipped, truth) 0.1248 -- values right,
    # placement wrong, the exact signature AB480 chased through eight windows).
    # Both decodes have the same BYTE COUNT, so nothing crashes; only a
    # round-trip against the declared drain layout can see it
    # (imaggers/block/swiglu/br_dn_decode2.py is that ruler, and the assert
    # below is the cheap in-line version of it).
    #     ffn_dn_rt1 (RT=1,  128^2)  30 objects of 96x128
    #     ffn_dn_n64 (RT=17, 480p )  60 objects of 96x64
    nobj, n2d = (60, 64) if d_img == "down" else (30, 128)
    _objs = (rows // 96) * nobj
    assert len(r2[0][0]) == _objs * 96 * n2d * 2, (
        "down drain %d B != %d objects of 96x%d (%d B): the untile width must be "
        "the image's n2, not a constant" % (len(r2[0][0]), _objs, n2d,
                                            _objs * 96 * n2d * 2))
    d = untile_obj(r2[0][0], 96, n2d)
    if tim is not None:
        tim['dev'] += r2[1]
    y = _np.frombuffer(d, _np.uint16).astype(_np.uint32) << 16
    # t22_breadout: same object-major trap as the gu hop above.  The drain emits
    # (nrt, nobj, 96, n2d) with object q = h*G + i carrying output columns
    # n2d*q; reshape(rows, 3840) read that flat buffer row-major.  Measured on
    # layer 0: every object matches its reference at corr 0.9999 (object table =
    # identity) while the flat array correlates at 0.0049 -- the exact "values
    # right, placement wrong" signature AB480 chased through eight hypotheses.
    y = y.view(_np.float32).reshape(rows // 96, nobj, 96, n2d).transpose(
        0, 2, 1, 3).reshape(rows, 3840)
    if _hop_here:
        _pl.Path(_hop + "_" + d_img + "_y.bin").write_bytes(
            _np.ascontiguousarray(y[:96], _np.float32).tobytes())
    return y


def bigm_ffn(dev, xm, swiglu_w, b3_w, tim=None, imgs=("swiglu", "down"),
             tail_imgs=("swiglu1", "down1")):
    """xm (SEQ, 3840) float32 -> y (SEQ, 3840) float32, FFN on device.

    Up to 1632 rows (17 tiles) is one dispatch.  Above that the row axis is cut
    into 1632-row chunks (17+17+9 for 1024^2 = 4128 rows), each one RT=17
    dispatch; the last chunk is zero-padded to a whole tile.  FFN is row-local,
    so a padded chunk changes no real row.
    """
    import numpy as _np
    seq = xm.shape[0]
    if seq > 1632:
        D = xm.shape[1]
        parts = []
        for r0 in range(0, seq, 1632):
            seg = xm[r0:r0 + 1632]
            rows = seg.shape[0]
            nt = (rows + 95) // 96
            pad = _np.zeros((nt * 96, D), _np.float32)
            pad[:rows] = seg
            u = pad.view(_np.uint32)
            xb = ((u + _np.uint32(0x7FFF) + ((u >> 16) & _np.uint32(1))) >> 16).astype(_np.uint16)
            y = _ffn_block(dev, xb, swiglu_w, b3_w, imgs, tim)[:rows]
            parts.append(y)
        return parts[0] if len(parts) == 1 else _np.concatenate(parts, 0)
    rt, rem = divmod(seq, 96)
    u = _np.asarray(xm, _np.float32).view(_np.uint32)
    xb = ((u + _np.uint32(0x7FFF) + ((u >> 16) & _np.uint32(1))) >> 16).astype(_np.uint16)
    parts = []
    if rt:
        parts.append(_ffn_block(dev, xb[:rt * 96], swiglu_w, b3_w, imgs, tim))
    if rem:
        pad = _np.zeros((96, xb.shape[1]), _np.uint16)
        pad[:rem] = xb[rt * 96:]
        parts.append(_ffn_block(dev, pad, swiglu_w, b3_w, tail_imgs, tim)[:rem])
    return parts[0] if len(parts) == 1 else _np.concatenate(parts, axis=0)

_CACHE = {}


def to_product_order(container):
    """re-order my [g][w][gu][576] blocks into the product's [...][w][gu][576].

    The block multiset is identical on both sides -- only the order differs.
    CAUTION: the index formula below was fitted to the first three groups
    (mine 0,1,2 -> prod 0,4,8 groups) and OVERFLOWS for g > 66, because
    153,600/576 = 266.67 groups means the 576-block grid does not align with
    the product's group axis.  Do not enable until the mapping is re-derived
    from the full index range (PRODASM-STEP345 70).
    """
    import numpy as _np
    a = _np.frombuffer(container, _np.uint8).reshape(-1, 576)
    n = a.shape[0]
    out = _np.empty_like(a)
    for i in range(n):
        g, r = divmod(i, 576)
        out[4 * g * 576 + (r % 2) + 8 * (r // 2)] = a[i]
    return _np.ascontiguousarray(out).tobytes()


_MAP = None


def apply_bg_map(container):
    """re-order the container's 576 B blocks into the product's order.

    The map is STORED, not fitted (PRODASM-STEP345 71): the block multiset is
    identical on both sides, so bg_block_map.npy (int32 x 153,600) is derived
    once by make_bg_map.py, with a positive control (apply -> equals the
    product byte for byte) and a negative control (a rolled table must fail).
    """
    global _MAP
    import numpy as _np
    if _MAP is None:
        _MAP = _np.load(pathlib.Path(__file__).resolve().parent / 'work'
                        / 'bg_block_map.npy').astype(_np.int64)
    a = _np.frombuffer(container, _np.uint8).reshape(-1, 576)
    assert a.shape[0] == _MAP.size, (a.shape, _MAP.size)
    out = _np.empty_like(a)
    out[_MAP] = a
    return _np.ascontiguousarray(out).tobytes()


# ---------------------------------------------------------------------------
# The product's OWN resident cache, repacked in place (t22_480ab)
# ---------------------------------------------------------------------------
# load_payloads() keeps a SECOND copy of all 30 layers (3.98 GB) on top of the
# 3.98 GB the product already holds with --cache-all; the 480p arm died with
#     ArrayMemoryError: Unable to allocate 84.4 MiB (153600, 576) uint8
# inside apply_bg_map before the first layer finished.  The fix is the one the
# in-place repack was written for: permute the resident rows themselves and
# hand them out, so the big-M FFN costs 0 extra GB instead of 3.98 GB.
_BGM_PERM = None
_BGM_CACHE_READY = False
_B3_ALT = None      # the second down-generation B3 order (see repack_cache_inplace_bigm)
_HOP_DONE = False   # T22_DUMP_HOP: one per-hop dump per process (see _ffn_block)


def _bgm_perm():
    """old 576-B block id (product BG order) -> new id (what sw64d2 eats).

    The composite of prod_repack.to_bigm and bg_block_map, recovered ONCE by
    permuting an id-encoded probe: every 576-B block carries its own index in
    its first 8 bytes, so after the two permutations the destination position
    of block i is read straight out.  Cross-checked against the long path by
    repack_check.py (repack_cache_inplace_bigm == apply_bg_map(to_bigm(...))).
    """
    global _BGM_PERM
    if _BGM_PERM is not None:
        return _BGM_PERM
    import numpy as _np
    from prod_repack import shapes, to_bigm
    n = 88473600 // 576
    ids = _np.arange(n, dtype=_np.int64)
    probe = _np.zeros((n, 576), _np.uint8)
    probe[:, :8] = ids.view(_np.uint8).reshape(n, 8)
    cont = to_bigm(probe.reshape(-1), shapes(88473600, 4, 40)).tobytes()
    out = apply_bg_map(cont)
    got = _np.frombuffer(out, _np.uint8).reshape(n, 576)[:, :8].copy()
    got = got.view(_np.int64).reshape(-1)
    if sorted(got.tolist()) != list(range(n)):      # it must be a permutation
        raise RuntimeError("bgm map is not a permutation")
    _BGM_PERM = got.astype(_np.int64)
    return _BGM_PERM


def _bg_unpermute(buf, nh, ggu=4, kk=480, obj=1152):
    """Undo t11_permute.permute_bg (t11_permute.py:10-12):

        dst[w][p][k][j] = src[p*nh + j][k][w]
        dst_obj = ((w*passes + p)*kk + k)*nh + j
        src_obj = ((p*nh + j)*kk + k)*ggu + w

    The PRODUCT's resident cache is that permutation: webui/bridge.py:385-386
    sets drv.NH_BG = 5, drv.NH_BG_CORE = True, and t18_run's cache-all pass
    permutes every layer at startup.  B3 is NOT permuted.  Applying the image
    order to permuted bytes is silently wrong -- measured: the gu payload the
    product fed had sha 60be4265... where the correct order has 20dca39f...
    """
    import numpy as _np
    a = _np.asarray(buf, _np.uint8)
    n = a.size // obj
    passes = n // (kk * nh * ggu)
    assert passes * kk * nh * ggu == n, (n, passes, kk, nh, ggu)
    p = _np.arange(passes)[:, None, None, None]
    k = _np.arange(kk)[None, :, None, None]
    j = _np.arange(nh)[None, None, :, None]
    w = _np.arange(ggu)[None, None, None, :]
    s = (((p * nh + j) * kk + k) * ggu + w).reshape(-1)
    d = (((w * passes + p) * kk + k) * nh + j).reshape(-1)
    rows = a.reshape(n, obj)
    out = _np.empty_like(rows)
    out[s] = rows[d]
    return out.reshape(-1)


def repack_cache_inplace_bigm(cache_bg, cache_b3=None, layers=None, nh_bg=0):
    """Turn the product's resident CACHE_BG rows into the big-M order, in place.

    cache_bg[li] is (88,473,600,) uint8 in the delivered order.  B3 needs no
    reorder (the down image eats it as delivered), so only BG is touched.
    Measured on the product-shaped cache: 0.12 s/layer, private bytes +0.00 GiB.
    """
    import numpy as _np
    import pathlib as _pl
    import sys as _sys
    # THE 9-BYTE ORDER, not the 576-B table (t22_480ab 4.10).
    #   * bg_block_map.npy in bigm/work is to_bigm ITSELF (an involution), so
    #     apply_bg_map(to_bigm(x)) == x: that table can never change anything.
    #   * worse, it permutes 576-B tiles while the order the images consume is
    #     a permutation of 9-byte bfp16 units -- a 576-B table cannot express
    #     it by construction.
    # kgu_ffn.pack_bg is the 9-byte order for exactly this object geometry
    # (9,216 B = [half 2][n-group 8][k-tile 8][n-sub 8] x 9 B), and it is
    # cross-checked by a second, independent implementation: the int32 map
    # kgu64/out/bg_block_map.npy (9,830,400 entries) reproduces it byte for
    # byte under the out[argsort(m)] = src convention.
    # THE CONTAINER ORDER, not the delivered one (t22_480ab 4.16).
    # sw_pack.py --via-container is the construction whose reference PRODASM 27
    # gated at corr 0.999911 on THIS image, and it feeds the arm the DECIDED
    # prodasm container: [kk=8][gate 576 | up 576] objects, i.e. to_bigm(BG).
    # load_payloads() instead returns apply_bg_map(container), and since
    # bg_block_map IS to_bigm (an involution) that equals to_bigm(to_bigm(BG))
    # == BG -- the DELIVERED order.  The shipped path was feeding the image the
    # exact inverse of what it declares, which is why every order hypothesis
    # failed and why all-ones (permutation-invariant) kept passing.
    # SHIPPED BEHAVIOUR (reverted after testing): the resident rows go out
    # UNCHANGED.  The container order (to_bigm) was tested too -- corr 0.135 vs
    # 0.137, i.e. no change -- so with no culprit identified the product path
    # keeps exactly the bytes load_payloads() used to return.
    # t22_breadout: the shipped path went out UNCHANGED, and that is the defect.
    # The delivered payloads are ordered for the DELIVERED kg=8 image; the big-M
    # images are different compilations and the geometry is part of the payload
    # contract.  Measured against an independent float32 oracle (layer 0):
    #     gu    delivered -0.000692  ->  converted 0.999873
    #     down  delivered -0.000120  ->  converted 0.999914
    global _B3_ALT
    from prod_bigborder import (bg_to_image, b3_to_image, b3_to_image_n64,
                               b3_to_image_k32)
    devhand = os.environ.get("T22_BIGM_DEVHANDOFF", "0") == "1"
    n = cache_bg.shape[0] if layers is None else int(layers)
    dual = os.environ.get("T22_BIGM_B3_DUAL", "1") != "0"
    if cache_b3 is not None and dual and _B3_ALT is None:
        # +1.33 GiB (30 x 44.2 MB) buys BOTH down generations.  The two big-M
        # down images do not share a B3 order:
        #     ffn_dn_rt1 (RT=1,  128^2)  n2=128 HALF=3  B_OBJ 9,216
        #     ffn_dn_n64 (RT=17, 480p )  n2=64  HALF=6  B_OBJ 4,608
        # T22_BIGM_B3_DUAL=0 keeps the old footprint and converts per dispatch.
        _B3_ALT = _np.empty_like(cache_b3)
    for li in range(n):
        if nh_bg and nh_bg > 1:
            cache_bg[li] = _bg_unpermute(cache_bg[li], nh_bg)
        cache_bg[li] = bg_to_image(cache_bg[li])
        if cache_b3 is not None and dual:
            raw = _np.array(cache_b3[li], _np.uint8, copy=True)
            cache_b3[li] = b3_to_image(raw)          # n128 (ffn_dn_rt1)
            _B3_ALT[li] = (b3_to_image_k32(raw)      # k32 (dn32 handoff)
                           if devhand else b3_to_image_n64(raw))
            del raw
    return n


def cached_payloads(key, cache_bg, cache_b3, nh_bg=0):
    """(bg, b3) for one layer, taken from the product's resident cache.

    T22_BIGM_ORDER selects the B order this arm consumes:
      "map" (default)  the [n-group][k-tile][n-sub] order (bg_block_map)
      "raw"            the delivered order, untouched
    The two are one flag apart so the 480p A/B can attribute a numeric
    difference to the ORDER rather than to the resident-cache path itself.
    """
    global _BGM_CACHE_READY
    if not _BGM_CACHE_READY:
        _done = False
        # t22_brepack: the repack below is a PURE PERMUTATION of bytes that the
        # product already has on disk, and it is re-derived on every run
        # (measured 19.91 s of wall -- HOSTPROF-REPORT.md 8).  With
        # T22_BREPACK_CACHE=1 prod_loop_bigm asks brepack_cache for the bytes
        # instead: map them if the content-addressed file exists, otherwise
        # repack once and store them.  Same bytes, same order, same instant in
        # the run's life -- only the cost changes.  Off by default, and any
        # refusal falls through to the shipped path unchanged.
        if os.environ.get("T22_BREPACK_CACHE", "0") == "1":
            try:
                import brepack_cache as _BRC
                _hit = _BRC.join() if _BRC.started() else None
                if _hit is not None:
                    _BRC.arm(_hit)
                    print("[bigm] B payloads mapped from %s (%.2f GB, %.2f s): "
                          "the %d-layer repack is skipped"
                          % (_hit["name"], _hit["bytes"] / 2**30,
                             _hit["ident_s"] + _hit["header_s"]
                             + _hit["verify_s"], cache_bg.shape[0]), flush=True)
                    _done = True
                else:
                    _out = _BRC.build(cache_bg, cache_b3, None,
                                      cache_bg.shape[0], nh_bg)
                    if _out is not None:
                        print("[bigm] B payloads repacked once (%s) and stored "
                              "for the next run" % (_out.get("name"),),
                              flush=True)
                        _done = True
            except Exception as _e:                          # noqa: BLE001
                print("[bigm] brepack cache unavailable (%r); using the shipped "
                      "repack" % (_e,), flush=True)
        if not _done and os.environ.get("T22_BIGM_ORDER", "map") == "raw":
            print("[bigm] B order: delivered (T22_BIGM_ORDER=raw); no repack",
                  flush=True)
        elif not _done:
            repack_cache_inplace_bigm(cache_bg, cache_b3, nh_bg=nh_bg)
            print("[bigm] B order: mapped [n-group][k-tile][n-sub]; %d layers "
                  "repacked in place" % cache_bg.shape[0], flush=True)
        _BGM_CACHE_READY = True
    li = int(key.split('.')[1])
    dual = os.environ.get("T22_BIGM_B3_DUAL", "1") != "0"
    if not dual:
        # no extra resident memory: hand out the DELIVERED bytes and let
        # _ffn_block convert for the generation it is about to dispatch
        return cache_bg[li], {"raw": cache_b3[li]}
    if os.environ.get("T22_BIGM_DEVHANDOFF", "0") == "1":
        return cache_bg[li], {"n32": _B3_ALT[li]}
    if os.environ.get("T22_DEVKILL", "0") == "1":
        # t22_loud: the same refusal chunked.install() makes at startup, repeated
        # at the producer so a harness that arms this function directly (the
        # imaggers selftests do) cannot fall into the 60 s traceback either.
        raise SystemExit(
            "[bigm] REFUSAL: T22_DEVKILL=1 needs the n32 B payload; "
            "T22_BIGM_DEVHANDOFF!=1 builds {n128,n64} instead -- the sentence "
            "devkill_ffn.block would raise 60 s from now")
    return cache_bg[li], {"n128": cache_b3[li], "n64": _B3_ALT[li]}


def load_payloads(key):
    """key 'layers.N' -> (container_bytes, b3_bytes), cached per process."""
    import numpy as _np
    if key not in _CACHE:
        li = int(key.split('.')[1])
        w = pathlib.Path(__file__).resolve().parent / 'work'
        from prod_bigborder import (bg_to_image, b3_to_image, b3_to_image_n64,
                                   b3_to_image_k32)
        devhand = os.environ.get("T22_BIGM_DEVHANDOFF", "0") == "1"
        c = _np.load(w / ('L%d_container.npy' % li))
        b = _np.load(w / 'b3' / ('L%d_b3.npy' % li))
        _CACHE[key] = (bg_to_image(apply_bg_map(c.tobytes())).tobytes(),
                       ({"n32": b3_to_image_k32(b).tobytes()}
                        if devhand else
                        {"n128": b3_to_image(b).tobytes(),
                         "n64": b3_to_image_n64(b).tobytes()}))
    return _CACHE[key]


# ---------------------------------------------------------------------------
# t22_bigpipe2: whole-shape projection + the attention wrapper that uses it.
# The projection image (rt17) is the one already registered as "proj"; its
# BOs hold 17 row tiles.  A 512^2 step (11 tiles) feeds 11 real tiles and 6
# zero tiles, a 1024^2 step (43 tiles) is cut into 17+17+9.  Weights are read
# ONCE per dispatch (88.5 MB gu / 16.6 MB proj) instead of once per 96-row
# block, which is the whole point of the exercise.
# ---------------------------------------------------------------------------
QKV_DIR = HERE / "work" / "qkv"
_PROJ_W = {}


def _proj_w(key, name):
    """(layer, q|k|v|o) -> the 16,588,800 B B-stream of prod_qkv.pack_proj."""
    li = int(str(key).split('.')[1])
    fn = {"q": "to_q", "k": "to_k", "v": "to_v", "o": "to_out0"}[name]
    k2 = (li, fn)
    if k2 not in _PROJ_W:
        raw = _np.load(QKV_DIR / ("L%d_%s.npy" % (li, fn))).tobytes()
        # t22_bigpipe3: the shipped file is the [sweep][kk][w][576] container.
        # bigm_proj's B object is [n-block 16][k-tile 8][72 B]; the 72 B
        # sub-blocks must be scattered (prod_qkv.qkv_to_image).  T22_BIGM_PROJ_ORDER
        # keeps the old container order reachable for the negative control.
        if os.environ.get("T22_BIGM_PROJ_ORDER", "image") != "container":
            from prod_qkv import qkv_to_image
            raw = qkv_to_image(raw).tobytes()
        _PROJ_W[k2] = raw
    return _PROJ_W[k2]


def _bf16_bytes(x):
    """float32 (rows, cols) -> bf16 bytes, round-to-nearest-even."""
    u = _np.asarray(x, _np.float32).view(_np.uint32)
    return (((u + _np.uint32(0x7FFF) + ((u >> 16) & _np.uint32(1))) >> 16)
            .astype(_np.uint16))


def _proj_dispatch(dev, x, w, n_tiles, tim, tag="proj"):
    """One bigm_proj dispatch covering n_tiles row tiles (rows <= 96*n_tiles)."""
    rows = x.shape[0]
    pad = _np.zeros((n_tiles * 96, x.shape[1]), _np.float32)
    pad[:rows] = x
    xa = a_pack(_bf16_bytes(pad))
    return _proj_dispatch_xa(dev, xa, rows, w, n_tiles, tim, tag)


def _proj_decode(raw, n_tiles, rows):
    """Drain -> (rows, 3840) f32 in ONE transpose.

    object q = h*G+i carries columns 128q; within an object the mm C layout is
    [12 row-blocks][16 col-blocks][8][8].  The old path did untile_obj (a
    16 MB bytes copy) then a second transpose+reshape.  This is the same
    permutation composed once: [tile][obj][rb][cb][ri][ci] ->
    [tile][rb][ri][obj][cb][ci].
    """
    a = _np.frombuffer(raw, _np.uint16).reshape(n_tiles, 30, 12, 16, 8, 8)
    b = a.transpose(0, 2, 4, 1, 3, 5).reshape(n_tiles * 96, 3840)
    y = (b.astype(_np.uint32) << 16).view(_np.float32)[:rows]
    return _np.ascontiguousarray(y)


def _proj_dispatch_xa(dev, xa, rows, w, n_tiles, tim, tag="proj"):
    """Same dispatch, with the A payload already built (one pack serves q/k/v/o)."""
    t0 = time.perf_counter()
    _raws, wall, tw = dev.dispatch(
        "proj",
        [(dev.bo["b_proj_x"], xa), (dev.bo["b_proj_w"], w)],
        [(dev.bo["b_proj_o"], n_tiles * 3 * 10 * 96 * 128 * 2)],
        tag=tag)
    raw = _raws[0]
    pack_ms = 1e3 * (time.perf_counter() - t0) - wall
    if tim is not None:
        tim["dev"] = tim.get("dev", 0.0) + wall
        tim["write"] = tim.get("write", 0.0) + tw
        tim["pack"] = tim.get("pack", 0.0) + max(0.0, pack_ms)
    # drain: [rt][h*G+i] objects of 96 x 128, object q carries columns 128*q
    return _proj_decode(raw, n_tiles, rows)


def _proj_run_cached(dev, xa, rows, n_tiles, tim, tag):
    """One proj dispatch with W already resident in b_proj_w.

    The 1024^2 shape runs 3 row chunks per projection; the old path rewrote
    the same 16.6 MB W for every chunk (12 writes/layer instead of 4).  This
    writes only X, runs, and decodes the drain in one transpose.
    """
    im = dev.imgs["proj"]
    bo = dev.bo
    t0 = time.perf_counter()
    bo["b_proj_x"].write(xa, 0)
    bo["b_proj_x"].sync(dev.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    t1 = time.perf_counter()
    run = im["k"](3, im["bo_i"], len(im["inst"]) // 4,
                  bo["b_proj_x"], bo["b_proj_w"], bo["b_proj_o"])
    st = int(run.wait(300000))
    wall = 1e3 * (time.perf_counter() - t1)
    if st != 4:
        raise RuntimeError("bigm_proj cached-W %s state %d" % (tag, st))
    t2 = time.perf_counter()
    raw = bytes(bo["b_proj_o"].read(n_tiles * 30 * 96 * 128 * 2, 0))
    t3 = time.perf_counter()
    y = _proj_decode(raw, n_tiles, rows)
    t4 = time.perf_counter()
    if os.environ.get("T22_BIGM_DEBUG", "0") == "1":
        print("[proj] %s tiles=%d write=%.1f run=%.1f read=%.1f dec=%.1f ms"
              % (tag, n_tiles, 1e3*(t1-t0), wall, 1e3*(t3-t2), 1e3*(t4-t3)),
              flush=True)
    if tim is not None:
        tim["dev"] = tim.get("dev", 0.0) + wall
    return y


def bigm_proj4(dev, x, key, tim=None, max_tiles=17):
    """x (seq, 3840) f32 -> dict q/k/v/o (seq, 3840) f32, whole-shape dispatches.

    t22_bigpipe3: the A payload is packed ONCE per row chunk and reused by
    q/k/v/o.  The old shape packed the same x four times (a_pack is a fancy
    index over the whole tile; ~20 ms per 1,056 rows), which was pure host tax.
    """
    seq = x.shape[0]
    chunks = []
    for r0 in range(0, seq, max_tiles * 96):
        r1 = min(seq, r0 + max_tiles * 96)
        rows = r1 - r0
        nt = (rows + 95) // 96
        pad = _np.zeros((nt * 96, x.shape[1]), _np.float32)
        pad[:rows] = x[r0:r1]
        chunks.append((a_pack(_bf16_bytes(pad)), rows, nt))
    out = {}
    for name in ("q", "k", "v", "o"):
        w = _proj_w(key, name)
        dev.bo["b_proj_w"].write(w, 0)
        dev.bo["b_proj_w"].sync(
            dev.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        parts = [_proj_run_cached(dev, xa, rows, nt, tim, "proj/" + name)
                 for xa, rows, nt in chunks]
        out[name] = parts[0] if len(parts) == 1 else _np.concatenate(parts, 0)
    return out


def bigm_attention(self, x1, key, cos, sin, tim):
    """Drop-in for Loop.attention: whole-shape q/k/v/o + host-f32 attention."""
    import time as _time
    T = _t18()
    P = self.params(key)
    seq, D, HEADS, HD = T.SEQ, T.D, T.HEADS, T.HD
    t0 = _time.perf_counter()
    Q = bigm_proj4(self.dev, x1, key, tim)
    t_proj = _time.perf_counter()
    def heads(a):
        return _np.ascontiguousarray(a.reshape(seq, HEADS, HD).transpose(1, 0, 2))
    wq = T.bf16f32(P["norm_q"] * self.fold)
    kh = T.rope_rms_jit(heads(Q["k"]), P["norm_k"], cos, sin,
                        T.SCRATCH.get("kh", (HEADS, seq, HD), _np.float32),
                        T.SCRATCH.get("rms0", (HEADS, seq, HD), _np.float32))
    qh = T.rope_rms_jit(heads(Q["q"]), wq, cos, sin,
                        T.SCRATCH.get("qh", (HEADS, seq, HD), _np.float32),
                        T.SCRATCH.get("rms1", (HEADS, seq, HD), _np.float32))
    th = _time.perf_counter()
    from webui.chunked import _attention
    vh = heads(Q["v"])
    if os.environ.get("T22_AF9M", "0") == "1":
        # t22_openstack: 8-core v9 fused attention (30 heads in 4 rounds of 8).
        # Default 0 keeps the delivered host-f32 path byte-for-byte.
        import sys as _sys
        _af = pathlib.Path(__file__).resolve().parents[2] / "openstack"
        if str(_af) not in _sys.path:
            _sys.path.insert(0, str(_af))
        import af9m_prod as _AF
        o = _AF.stream_attention(self.dev, qh, kh, vh)
    elif os.environ.get("T22_ATTNWIRE", "0") == "1":
        # t22_attnwire: product streaming attention (6 cores, 30 heads in 5 rounds).
        # Default 0 keeps the delivered host-f32 path byte-for-byte.
        import sys as _sys
        _aw = pathlib.Path(__file__).resolve().parents[2] / "attnwire"
        if str(_aw) not in _sys.path:
            _sys.path.insert(0, str(_aw))
        import attnwire_prod as _AW
        o = _AW.stream_attention(self.dev, qh, kh, vh)
    else:
        o = _attention(qh, kh, vh)
    _dump = os.environ.get("T22_ATTN_DUMP", "")
    if _dump and str(key) == "layers.0":
        import pathlib as _pl
        _pl.Path(_dump + "_qh.npy").parent.mkdir(parents=True, exist_ok=True)
        _np.save(_dump + "_qh.npy", qh)
        _np.save(_dump + "_kh.npy", kh)
        _np.save(_dump + "_vh.npy", vh)
        _np.save(_dump + "_o.npy", o)
        print("[attn-dump] wrote %s_{qh,kh,vh,o}.npy" % _dump, flush=True)
    ta = _time.perf_counter()
    if tim is not None:
        tim["host"] = tim.get("host", 0.0) + 1e3 * (_time.perf_counter() - th)
    o_flat = _np.ascontiguousarray(o.reshape(seq, D))
    y = bigm_proj4(self.dev, o_flat, key, tim)["o"]
    t_end = _time.perf_counter()
    if os.environ.get("T22_BIGM_DEBUG", "0") == "1":
        print("[bigm2] %s proj_qkv %.3f rope %.3f attn %.3f proj_o %.3f"
              % (key, t_proj - t0, th - t_proj, ta - th, t_end - ta), flush=True)
    return y
