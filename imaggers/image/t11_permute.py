# SPDX-License-Identifier: MIT
"""t11_permute.py -- the BG object permutation for the T11 NH=5 per-core image.

The T11 image gives every gate/up core its OWN shim fill, so each core reads one
CONTIGUOUS slice of the BO, and its stream is ordered [pass][kk][j] (a core holds
NH n-tiles at once, so for one (pass, kk) it walks the NH B objects before the A
stream advances).  The payload on disk / in the driver is the DELIVERED
[sweep][kk][w][gate|up] order, so the host must write

    dst[w][pass][kk][j] == src[sweep = pass*NH + j][kk][w]

i.e. dst object index ((w*passes + p)*kk + k)*nh + j  <-  src ((p*nh + j)*kk + k)*ggu + w
at 1,152 B object granularity (kgu=8, G_GU=4).  Pure permutation: the device
reads exactly the bytes t11_pack.py produces, no re-encode.

One pass over 88.5 MB (76,800 objects x 144 uint64 words), destination written
linearly -- the same price as T9 route (a), ~11-25 ms per layer.
"""
from __future__ import annotations

import numpy as np
from numba import njit

OBJ_BYTES = 1152          # 2 * (8*64/8) * 9 = [gate|up] pair at kgu=8
WORDS = OBJ_BYTES // 8    # 144 x uint64


@njit(cache=True)
def _perm64c(src, dst, words, kk, nh, ggu):
    """dst[w][p][k][j] = src[p*nh+j][k][w].

    ORDER MATTERS AND IS MEASURED (29.3 -> 9.0 ms per layer): the group
    src[(p*nh+j)][k][w=0..ggu-1] is 4 x 1,152 B CONTIGUOUS, so it is read with
    one linear walk and scattered into the four per-core streams, which are
    each written sequentially.  The naive form (w outermost, one 1,152 B block
    per visit) reads the same 88.5 MB four times at 25 % density.
    """
    passes = (src.size // words) // (kk * nh * ggu)
    per_core = passes * kk * nh
    for p in range(passes):
        for k in range(kk):
            for j in range(nh):
                s = (((p * nh + j) * kk + k) * ggu) * words
                for w in range(ggu):
                    d = (w * per_core + (p * kk + k) * nh + j) * words
                    sw = s + w * words
                    for q in range(words):
                        dst[d + q] = src[sw + q]


def permute_bg(src, dst, nh, kk=480, ggu=4):
    """Permute one layer BG bytes (uint8, contiguous) into dst (same size)."""
    if nh <= 1:
        dst[:] = src
        return dst
    assert src.size % OBJ_BYTES == 0, src.size
    nobj = src.size // OBJ_BYTES
    assert nobj % (kk * nh * ggu) == 0, (nobj, kk, nh, ggu)
    s64 = np.frombuffer(src, np.uint64) if not isinstance(src, np.ndarray) \
        else np.frombuffer(src, np.uint64)
    d64 = np.frombuffer(dst, np.uint64)
    _perm64c(s64, d64, WORDS, kk, nh, ggu)
    return dst
