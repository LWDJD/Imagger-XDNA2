# SPDX-License-Identifier: MIT
"""qkv_image_g.py -- qkv_to_image with the worker count EXPLICIT.

prod_qkv.qkv_to_image is frozen at G=10 (module constant).  Its object order is
[worker i][pass h] carrying output columns 128*(h*G+i), so changing G without
re-packing W silently computes the right columns with the wrong weights.
This is the same packing loop, parameterised on G.
"""
import numpy as np

N = K = 3840
TILE = 576


def qkv_to_image_g(container, G=15, N=N, K=K):
    a = np.frombuffer(container, np.uint8)
    assert a.size == N * K * 9 // 8, (a.size, N * K * 9 // 8)
    # the CONTAINER is laid out for the delivered G=10 image: sweeps is a
    # property of the file, NOT of the worker count being built.
    sweeps = (N // 64) // 10
    a = a.reshape(sweeps, K // 8, 10, 576)
    ng = K // 64
    half = N // (128 * G)
    out = np.empty((G, half, ng, 16, 8, 72), np.uint8)
    kk = np.arange(K // 8)
    for h in range(half):
        q = h * G + np.arange(G)
        for t8 in range(2):
            jj = 2 * q + t8
            s_, w_ = jj % sweeps, jj // sweeps
            for gb in range(ng):
                k0 = 8 * gb
                blk = a[s_[:, None], k0:k0 + 8, w_[:, None], :]   # (G,8,576)
                out[:, h, gb, t8 * 8:(t8 + 1) * 8] = blk.reshape(G, 8, 8, 72).transpose(0, 2, 1, 3)
    return np.ascontiguousarray(out).reshape(-1)
