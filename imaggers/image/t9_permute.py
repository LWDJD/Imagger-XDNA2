# SPDX-License-Identifier: MIT
"""t9_permute.py -- the BG object permutation for the T9 n_hold FFN image.

The n_hold=2 image consumes the gate/up B stream as [pass][kk][j] with
sweep = pass*NH + j, because a core holds NH n-tiles' C tiles and, for one
(pass, kk), walks the NH B objects before the A stream advances.  The payload
on disk / in the driver is the DELIVERED [sweep][kk] order.

The permutation is at the OBJECT level (4,608 B = G_GU workers x [gate|up] x
one (k=8, n=64) bfp16ebs8 tile): no re-encode, no new codec, the device reads
the same bytes.  Destination is written linearly, so the kernel is one read +
one write of 88.5 MB (576 x 8 B words per object).

    measured: ~18-25 ms per layer, i.e. ~4-6 s over the 240 FFN dispatches of
    an 8-step run (the "route (a)" price in T9.md 9.3).

This is HOST payload plumbing, not arithmetic: the bytes the device sees are
identical to what t9_pack.py produces for the bench.
"""
from __future__ import annotations

import numpy as np
from numba import njit

OBJ_BYTES = 4608          # G_GU * 2 * (8*64/8) * 9 at kgu=8
WORDS = OBJ_BYTES // 8    # 576 x uint64


@njit(cache=True)
def _perm64(src, dst, words, kk, nh):
    """dst[i] = src[(pass*nh + j)*kk + k], i -> (pass, k, j) linear."""
    nobj = src.size // words
    per_pass = kk * nh
    for i in range(nobj):
        p = i // per_pass
        rem = i - p * per_pass
        k = rem // nh
        j = rem - k * nh
        s = (p * nh + j) * kk + k
        so = s * words
        do = i * words
        for w in range(words):
            dst[do + w] = src[so + w]


def permute_bg(bg, out, nh, kk=480):
    """Permute one layer's BG bytes (uint8, contiguous) into out (same size)."""
    if nh <= 1:
        out[:] = bg
        return out
    nobj = bg.size // OBJ_BYTES
    assert bg.size % OBJ_BYTES == 0, bg.size
    assert nobj % (kk * nh) == 0, (nobj, kk, nh)
    s64 = np.frombuffer(bg, np.uint64)
    d64 = np.frombuffer(out, np.uint64)
    _perm64(s64, d64, WORDS, kk, nh)
    return out
