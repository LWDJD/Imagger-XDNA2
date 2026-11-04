# SPDX-License-Identifier: MIT
# prod_aplane.py -- pack xm (1632, 3840) into the gu image's A-plane.
#
# The gu image wants 12,533,760 B = 17 x 60 x 96 x 64 x 2: one A2 micro-tile
# per (row-tile, k-group), concatenated rt-major then kg.  sw_pack builds its x
# with np.tile of ONE tile because its reference only ever uses one A; for real
# data every (rt, kg) slice differs, and tiling would be silently wrong.
from __future__ import annotations

import numpy as np

M, K = 96, 64
RT, NG = 17, 60


PLAIN = True   # the FFN image wants PLAIN bf16 rows: t18_run:1694-1755 shows
# the payload is H.bf16_bytes_f32(xb, ...) over a zero-padded row block, with no
# A2 tiling.  A2 was validated only as the C INVERSE (atile gate), never as the
# gu image's input, so applying it here was an extra and wrong transform.


def a2_tile(t, kg=None):
    """(96, kg) -> A2 micro-tile layout.  NOT used for the FFN A-plane.

    kg must be threaded through: reading the module-level K here silently broke
    every call with kg != 64 (PRODASM-STEP345 121) -- the same 'parameter not
    passed all the way down' mistake as section 112.
    """
    if PLAIN:
        return t
    kg = K if kg is None else kg
    return t.reshape(M // 8, 8, kg // 8, 8).transpose(0, 2, 1, 3).reshape(M, kg)


def a_plane(xm, kg=K):
    """(RT*96, NG*kg) bf16 -> A_OBJ * NG * RT bytes, rt-major then kg.

    kg is the k-group width the image was compiled for: 64 for the original
    generation, 8 for the k=8 generation (PRODASM-STEP345 110 -- the product
    instantiates every mm_bfp with dim_k=8).  A_OBJ = 96*kg*2, so kg=64 gives
    12,288 B and kg=8 gives 1,536 B; NG = K_total // kg.
    """
    xm = np.asarray(xm)
    assert xm.shape[1] % kg == 0, xm.shape
    assert xm.shape[0] % M == 0, xm.shape
    rt_n = xm.shape[0] // M          # RT is derived, not fixed: SEQ//96
    ng_n = xm.shape[1] // kg         # NG is derived: 3840//64 = 60, 3840//8 = 480
    out = np.empty((rt_n, ng_n, M, kg), xm.dtype)
    for rt in range(rt_n):
        for gi in range(ng_n):
            out[rt, gi] = a2_tile(xm[rt * M:(rt + 1) * M, gi * kg:(gi + 1) * kg],
                                  kg)
    return np.ascontiguousarray(out).tobytes()


def a_plane_tiled(xm):
    """NEGATIVE CONTROL: the np.tile shortcut.  Must NOT match a_plane."""
    xm = np.asarray(xm)
    t = a2_tile(xm[0:M, 0:K])
    return np.tile(t.reshape(-1), (xm.shape[0] // M) * NG).tobytes()


def selftest() -> int:
    rng = np.random.default_rng(11)
    xm = rng.standard_normal((RT * M, NG * K)).astype(np.float32)
    a = np.frombuffer(a_plane(xm), np.float32)
    b = np.frombuffer(a_plane_tiled(xm), np.float32)
    print('[aplane] real  %d B  uniq %d' % (a.size, len(np.unique(a))))
    print('[aplane] tiled %d B  uniq %d' % (b.size, len(np.unique(b))))
    print('[aplane] negative control identical? %s (must be False)' % np.array_equal(a, b))
    # the real plane must reproduce xm after the inverse, the tiled one must not
    inv = a.reshape(RT, NG, M // 8, K // 8, 8, 8).transpose(0, 1, 2, 4, 3, 5)
    inv = inv.reshape(RT, NG, M, K)
    # reassemble tile by tile -- reshaping (RT,NG,M,K) straight to (RT*M,NG*K)
    # interleaves the axes and is NOT the inverse of the per-tile packing.
    back = np.empty((RT * M, NG * K), xm.dtype)
    for rt in range(RT):
        for kg in range(NG):
            back[rt * M:(rt + 1) * M, kg * K:(kg + 1) * K] = inv[rt, kg]
    ok = np.array_equal(back, xm)
    print('[aplane] round trip == xm? %s' % ok)
    print('[aplane] VERDICT:', 'PASS' if (ok and not np.array_equal(a, b)) else 'FAIL')
    return 0 if (ok and not np.array_equal(a, b)) else 1


if __name__ == '__main__':
    raise SystemExit(selftest())
