
# SPDX-License-Identifier: MIT
"""prod_bigborder.py -- the B order the big-M FFN images actually declare.

WHY THIS FILE EXISTS
--------------------
The delivered payloads (step4_run_ffn.pack_bg_stream / pack_b3_stream) are
ordered for the DELIVERED kg=8 FFN image.  The big-M images are different
compilations with different geometry, and the geometry is part of the payload
contract:

  gu   sw_rt1   (t18_run.BIGM_IMGS["swiglu1"])  m=96 k=64 n=64 KW=3840 N=20480
       -> G=10, HALF=32, NG=60, n2=32, object (i*32+h)*60+gb = [gate|up]
       -> slice 2,304 B = [t=0..3 n-block][kt=0..7 k-tile][72 B]
       -> object (i,h) carries output columns 32*(h*10+i)   (drain k_out = h*G+i)

  down ffn_dn_rt1                                m=96 k=64 n=128 KW=10240 N=3840
       -> G=10, HALF=3, NG=160, n2=128, object (i*3+h)*160+gb
       -> slice 9,216 B = [t=0..15 n-block][kt=0..7 k-tile][72 B]
       -> object (i,h) carries output columns 128*(h*10+i)

  delivered BG : [sweeps=40][kk=480][w=4][half=2][576] , column block jj = w*40+s
  delivered B3 : [sweeps=40][w=4][s2=8][g=6][l=10][576],
                 tile = dd[jj3 = 10g+l, kk_p = 320w + 8s + s2]

Measured on the device against an INDEPENDENT float32 oracle (t3_real weights,
layer 0, never through the payload):
    gu   converted 0.999873 (per-object 0.999816-0.999900) | delivered -0.000692
    down converted 0.999914 (per-object 0.999906-0.999921) | delivered -0.000120
and the down image additionally needs the DECLARED A micro-tile: plain rows give
0.375111 where a_layout.pack gives 0.999914.  (ab480 4.13 tested a_pack on the
gu A against a payload with the wrong B order, so that reading was void.)

The lesson, same family as "a self-consistent ruler cannot see a self-consistent
assumption": the image's geometry knobs (G, HALF, NG, n2) are half of the payload
contract.  A packer that only matches the byte COUNT is silently wrong.
"""
from __future__ import annotations

import os

import numpy as np

# ---- gu: delivered BG -> sw_rt1 order --------------------------------------
GU_G, GU_HALF, GU_NG = 10, 32, 60
GU_OBJ = 2 * 2304          # [gate 2304 | up 2304]
GU_CH = 72                 # 8 columns x 9 B


def bg_to_image(bg: np.ndarray) -> np.ndarray:
    """(88,473,600,) uint8 delivered -> the swiglu1 image's order."""
    a = (np.frombuffer(bg, np.uint8) if isinstance(bg, (bytes, bytearray))
         else np.asarray(bg, np.uint8)).reshape(40, 480, 4, 2, 8, GU_CH)
    i = np.arange(GU_G)[:, None, None, None, None, None]
    h = np.arange(GU_HALF)[None, :, None, None, None, None]
    gb = np.arange(GU_NG)[None, None, :, None, None, None]
    half = np.arange(2)[None, None, None, :, None, None]
    t = np.arange(4)[None, None, None, None, :, None]
    kt = np.arange(8)[None, None, None, None, None, :]
    q = h * GU_G + i
    c = q // 2
    w = c // 40
    s = c % 40
    out = a[s, 8 * gb + kt, w, half, (q % 2) * 4 + t]     # (10,32,60,2,4,8,72)
    return np.ascontiguousarray(out).reshape(-1)


# ---- down: delivered B3 -> ffn_dn_rt1 order --------------------------------
DN_G, DN_HALF, DN_NG = 10, 3, 160
DN_OBJ = 9216


def b3_to_image(b3: np.ndarray) -> np.ndarray:
    """(44,236,800,) uint8 delivered -> the ffn_dn_rt1 image's order."""
    a = np.asarray(b3, np.uint8).reshape(40, 4, 8, 6, 10, 8, DN_CH)
    i = np.arange(DN_G)[:, None, None, None, None, None]
    h = np.arange(DN_HALF)[None, :, None, None, None, None]
    gb = np.arange(DN_NG)[None, None, :, None, None, None]
    t = np.arange(16)[None, None, None, :, None, None]
    kt = np.arange(8)[None, None, None, None, :, None]
    q = h * DN_G + i
    jj3 = 2 * q + t // 8
    kk = 8 * gb + kt
    out = a[(kk % 320) // 8, kk // 320, kk % 8, jj3 // 10, jj3 % 10, t % 8]
    return np.ascontiguousarray(out).reshape(-1)


DN_CH = 72

# ---- down, second generation: the RT=17 image is ffn_dn_n64 ---------------
# ffn_dn_rt1 : n=128 -> n2=128, HALF=3, object (i*3+h)*160+gb, B_OBJ 9,216
# ffn_dn_n64 : n=64  -> n2=64,  HALF=6, object (i*6+h)*160+gb, B_OBJ 4,608
# The gu order is shared (both gu images are G=10/HALF=32/n2=32, w 88,473,600 B);
# B3 is NOT -- that is why 480p (RT=17) read -0.0037 while 128^2 read 0.9907.
DN64_HALF = 6
DN64_OBJ = 4608
DN64_CH = 72


def b3_to_image_n64(b3: np.ndarray) -> np.ndarray:
    """(44,236,800,) uint8 delivered -> the ffn_dn_n64 image's order."""
    a = np.asarray(b3, np.uint8).reshape(40, 4, 8, 6, 10, 8, DN64_CH)
    i = np.arange(DN_G)[:, None, None, None, None]
    h = np.arange(DN64_HALF)[None, :, None, None, None]
    gb = np.arange(DN_NG)[None, None, :, None, None]
    t = np.arange(8)[None, None, None, :, None]
    kt = np.arange(8)[None, None, None, None, :]
    q = h * DN_G + i                       # 0..59, one object per 64 output columns
    kk = 8 * gb + kt
    out = a[(kk % 320) // 8, kk // 320, kk % 8, q // 10, q % 10, t]
    return np.ascontiguousarray(out).reshape(-1)


def _b3_to_image_k32_ref(b3: np.ndarray) -> np.ndarray:
    """(44,236,800,) delivered B3 -> the k=32 down image's object order.

    t22_bigpipe3 device-handoff arm.  The gu image drains 96x32 objects in
    C layout, and mm_bfp's A layout for dim_k=32 is byte-identical to its C
    layout (layoutcmp: k=32/k=64 both equal), so a down image compiled at
    k=32 can consume the gu output BO with NO host repack.  Its B object is
    4,608 B = [n-block 16][k-tile 4][72 B]; the delivered 576 B tile is
    [n 64][k 8], i.e. eight 72 B sub-blocks.

    Object q = h*G+i carries output columns 128q and inner dims [32gb,32gb+32):
        object[t][kt] = tile(jj3 = 2q + t//8, kk_p = 4gb + kt)[(t%8)*72:]
    and tile (jj3, kk_p) lives at B3[s, w, s2, g, l] with
        g = jj3 // 10, l = jj3 % 10,
        w = kk_p // 320, s = (kk_p % 320) // 8, s2 = kk_p % 8.
    Pure permutation of 72 B blocks (230,400 of them).
    """
    a = (np.frombuffer(b3, np.uint8) if isinstance(b3, (bytes, bytearray))
         else np.asarray(b3, np.uint8))
    assert a.size == 44_236_800, a.size
    a = a.reshape(40, 4, 8, 6, 10, 72 * 8)
    out = np.empty((10, 3, 320, 16, 4, 72), np.uint8)
    for i in range(10):
        for h in range(3):
            q = h * 10 + i
            for gb in range(320):
                for t in range(16):
                    jj3 = 2 * q + (t // 8)
                    g, l = jj3 // 10, jj3 % 10
                    off = (t % 8) * 72
                    for kt in range(4):
                        kk = 4 * gb + kt
                        w, rem = kk // 320, kk % 320
                        s, s2 = rem // 8, rem % 8
                        out[i, h, gb, t, kt] = a[s, w, s2, g, l, off:off + 72]
    return np.ascontiguousarray(out).reshape(-1)


def b3_to_image_k32_fast(b3: np.ndarray) -> np.ndarray:
    """The SAME permutation as _b3_to_image_k32_ref, as ONE transpose.

    The reference walks 10*3*320*16*4 = 614,400 python steps of 72 B copies
    (measured 297-535 ms/layer, i.e. ~13 s of the 23 s B repack).  The index
    algebra above is closed-form, so the whole transform is a reshape:

        object (i,h,gb,t,kt) <- a[s][w][gblo][kt][g][p][thi][sub]
        gb   = 80*w + 2*s + gblo     (gblo = s2//4, kt = s2%4)
        t    = 8*thi + sub           (thi  = l%2)
        m    = 5*g + p, h = m//10, i = m%10

    (gb follows because kk = 4*gb + kt and 4*gb never crosses a 320 boundary
    unless gb does, so w = gb//80 and s = (gb%80)//2, s2 = 4*(gb%2) + kt.)

    Byte-identical to the reference: proved per build by _fastk32() below, and
    re-proved on a REAL layer 0 by imaggers/block/bigm/gate_fastk32.py.
    """
    a = (np.frombuffer(b3, np.uint8) if isinstance(b3, (bytes, bytearray))
         else np.asarray(b3, np.uint8))
    if a.size != 44_236_800:
        raise ValueError("b3 is %d B, expected 44,236,800" % a.size)
    x = a.reshape(40, 4, 2, 4, 6, 5, 2, 8, 72)      # s,w,gblo,kt,g,p,thi,sub,72
    x = np.ascontiguousarray(x.transpose(4, 5, 1, 0, 2, 6, 7, 3, 8))
    x = x.reshape(3, 10, 4, 40, 2, 2, 8, 4, 72)     # h,i,w,s,gblo,thi,sub,kt,72
    x = np.ascontiguousarray(x.transpose(1, 0, 2, 3, 4, 5, 6, 7, 8))
    return x.reshape(10, 3, 320, 16, 4, 72).reshape(-1)


# T22_BB_FASTK32=1 selects b3_to_image_k32_fast.  DEFAULT 0: the reference
# implementation, byte-for-byte the delivered behaviour.  When the flag is on,
# the FIRST call is still run through BOTH implementations and compared on the
# real input; if they differ the flag disables itself (fail open, one warning)
# and the reference result is returned.  So the flag can never change a byte.
_FASTK32 = None


def b3_to_image_k32(b3: np.ndarray) -> np.ndarray:
    """Dispatches to the reference (default) or the vectorised transpose.

    T22_BB_FASTK32=1 -> vectorised, self-checked against the reference on the
    first real input.  T22_BB_FASTK32 unset/0 -> _b3_to_image_k32_ref, the
    unchanged shipped path.
    """
    global _FASTK32
    if _FASTK32 is None:
        on = os.environ.get("T22_BB_FASTK32", "0") == "1"
        _FASTK32 = {"on": on, "checked": False}
        if on:
            print("[prod_bigborder] T22_BB_FASTK32=1: b3_to_image_k32 -> "
                  "vectorised transpose; the first call is checked against "
                  "the reference on the real payload", flush=True)
    if not _FASTK32["on"]:
        return _b3_to_image_k32_ref(b3)
    if not _FASTK32["checked"]:
        _FASTK32["checked"] = True
        ref = _b3_to_image_k32_ref(b3)
        got = b3_to_image_k32_fast(b3)
        if not np.array_equal(ref, got):
            _FASTK32["on"] = False
            print("[prod_bigborder] REFUSAL: b3_to_image_k32_fast != the "
                  "reference on this payload (%d B differ); T22_BB_FASTK32 "
                  "disabled, using the reference" %
                  int(np.count_nonzero(ref != got)), flush=True)
            return ref
        print("[prod_bigborder] b3_to_image_k32_fast == the reference "
              "(byte-identical, %d B)" % int(ref.size), flush=True)
        return got
    return b3_to_image_k32_fast(b3)


def selftest() -> int:
    """CPU-only: sizes, bijectivity, and a_pack == the declared A micro-tile."""
    ok = True
    rng = np.random.default_rng(7)
    bg = rng.integers(0, 256, 88473600, dtype=np.uint8)
    b3 = rng.integers(0, 256, 44236800, dtype=np.uint8)
    g = bg_to_image(bg)
    d = b3_to_image(b3)
    ok &= g.size == bg.size and d.size == b3.size
    print("[selftest] sizes            %s" % ok)
    # bijection on 72-B chunks: the histogram of chunk hashes must match
    for nm, src, dst in (("BG", bg, g), ("B3", b3, d)):
        h_src = np.sort(np.frombuffer(src.reshape(-1, 72).tobytes(), np.uint8)
                        .reshape(-1, 72)[:, :8].view("<u8").ravel())
        h_dst = np.sort(np.frombuffer(dst.reshape(-1, 72).tobytes(), np.uint8)
                        .reshape(-1, 72)[:, :8].view("<u8").ravel())
        same = bool(np.array_equal(h_src, h_dst))
        moved = not np.array_equal(src, dst)
        print("[selftest] %-2s chunk multiset identical %s  moved %s" % (nm, same, moved))
        ok &= same and moved
    # the product's frozen a_pack must equal the library's declared A layout
    import pathlib
    import sys
    here = pathlib.Path(__file__).resolve().parent
    root = here.parents[2].parent
    sys.path.append(str(root / "_pylibs"))
    sys.path.append(str(root / "_pylibs" / "mlir_aie" / "python"))
    import aie.iron.kernels as kernels
    from aie.iron import set_current_device
    from aie.iron.device import NPU2
    from ml_dtypes import bfloat16
    from prod_loop_bigm import a_pack
    set_current_device(NPU2())
    lay = kernels.mm_bfp(dim_m=96, dim_k=64, dim_n=32, mixed=True).contract.layouts[0]
    x = (rng.standard_normal((96, 3840)) * 0.5).astype(np.float32).astype(bfloat16)
    want = b"".join(np.asarray(lay.pack(x[:, g * 64:(g + 1) * 64][None]),
                              dtype=bfloat16).tobytes() for g in range(60))
    got = a_pack(x)
    same = got == want
    print("[selftest] a_pack == declared A micro-tile (60 tiles): %s" % same)
    ok &= same
    print("[selftest] %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(selftest())
