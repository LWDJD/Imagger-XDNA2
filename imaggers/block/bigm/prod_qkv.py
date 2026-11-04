# SPDX-License-Identifier: MIT
"""prod_qkv.py -- q/k/v/o projection B streams, same shape discipline as the
FFN containers.

The rt17 image computes ONE 3840-wide projection per dispatch:
    kw RT=17, n=128, N=3840, G=10  ->  30 n-slices over 10 workers = HALF 3
    w_bytes 16,588,800 = 3840 x 3840 x 9/8   (ONE matrix, not three)
So each of to_q / to_k / to_v / to_out gets its OWN container, packed exactly
like pack_bg_stream but with a single stream instead of a gate|up pair:

    [sweep][kk][w][576]      sweeps = (3840/64)/10 = 6, kk = 480, w = 10
    gg = pack_b_blocks(W.T, K=3840, N=3840, k=8, n=64).reshape(60, 480, 576)
    out[s][kk][w] = gg[ jj = w*sweeps + s ][kk]        <- same jj as section 15.1
"""
from __future__ import annotations

import hashlib
import pathlib
import sys

import numpy as np

NPU = pathlib.Path(r"D:\Project\deepseek_harness\public_work\zimage-bfp16-npu")
sys.path.insert(0, str(NPU / 'imaggers' / 'block'))
sys.path.insert(0, str(NPU))

N = K = 3840
G = 10
TILE = 576
SWEEPS = (N // 64) // G          # 6


def sha16(b) -> str:
    return hashlib.sha256(np.asarray(b, np.uint8).tobytes()).hexdigest()[:16]


def pack_proj(w_bytes):
    import pack_bfp16 as P, step4_run_ffn as S
    W = np.asarray(P.unpack(w_bytes, N, K))
    gg = np.frombuffer(S.pack_b_blocks(W.T, K, N, 8, 64), np.uint8).reshape(
        N // 64, K // 8, TILE)
    out = np.empty((SWEEPS, K // 8, G, TILE), np.uint8)
    for s in range(SWEEPS):
        for w in range(G):
            out[s, :, w] = gg[w * SWEEPS + s]
    # the scatter is a pure permutation of the proven pack_b_blocks tiles:
    assert np.array_equal(np.sort(out.reshape(-1, TILE).view('V%d' % TILE).ravel()),
                          np.sort(gg.reshape(-1, TILE).view('V%d' % TILE).ravel())), \
        "container is not a permutation of the packed tiles"
    return np.ascontiguousarray(out.reshape(-1))


def qkv_to_image(container):
    """Container [sweep][kk][w][576] -> the bigm_proj B object order.

    THE MISSING CONTRACT (t22_bigpipe3).  pack_proj() emits 576 B tiles whose
    payload is [n 64][k 8] *for one k-tile* -- the delivered stream's unit.
    bigm_proj's B object is 9,216 B = [n-block 16][k-tile 8][72 B] (the
    mm_bfp layouts[1] block layout for dim_k=64/dim_n=128), exactly the same
    object geometry as ffn_dn_rt1 (see prod_bigborder.b3_to_image).  A whole
    tile permutation is therefore not enough: the 72 B sub-blocks inside each
    576 B tile must be scattered into [n-block][k-tile] order.

    For a down/ projection B tile at output columns [128q,128q+128) and
    k range [64gb,64gb+64):
        object[t][kt][:] = tile(jj = 2q + t//8, kk = 8*gb+kt)[(t%8)*72:]
    with jj = w*SWEEPS + s (s = jj % SWEEPS, w = jj // SWEEPS).
    The object order is [worker i][pass h][k-group gb] and it carries
    output columns 128*(h*G+i), which is the image's W tap.
    """
    import numpy as _np
    a = _np.frombuffer(container, _np.uint8)
    want = N * K * 9 // 8
    assert a.size == want, (a.size, want)
    a = a.reshape(SWEEPS, K // 8, G, TILE)
    ng = K // 64
    half = N // (128 * G)
    out = _np.empty((G, half, ng, 16, 8, 72), _np.uint8)
    for i in range(G):
        for h in range(half):
            q = h * G + i
            for gb in range(ng):
                kk0 = 8 * gb
                for t in range(16):
                    jj = 2 * q + (t // 8)
                    s, w = jj % SWEEPS, jj // SWEEPS
                    off = (t % 8) * 72
                    out[i, h, gb, t] = a[s, kk0:kk0 + 8, w, off:off + 72]
    return _np.ascontiguousarray(out).reshape(-1)


def main() -> int:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    import json
    out = NPU / 'imaggers' / 'block' / 'bigm' / 'work' / 'qkv'
    out.mkdir(parents=True, exist_ok=True)
    rec = []
    want = N * K * 9 // 8
    for li in range(30):
        p = NPU / 'weights' / ('block%d.npz' % li)
        if not p.is_file():
            print('[qkv] L%d: no npz' % li); continue
        z = np.load(p)
        for tail in ('to_q', 'to_k', 'to_v', 'to_out.0'):
            key = 'layers.%d.attention.%s.weight' % (li, tail)
            if key not in z:
                print('[qkv] L%d MISSING %s' % (li, key)); continue
            c = pack_proj(z[key])
            assert c.size == want, (c.size, want)
            nm = 'L%d_%s.npy' % (li, tail.replace('.', ''))
            np.save(out / nm, c)
            rec.append({'layer': li, 'tail': tail, 'bytes': int(c.size),
                        'sha16': sha16(c)})
        print('[qkv] L%-2d 4 streams %s B each' % (li, format(want, ',')), flush=True)
    (out.parent / 'qkv_manifest.json').write_text(json.dumps(rec, indent=1),
                                                  encoding='utf-8')
    print('[qkv] %d streams, all %s B, manifest written' % (len(rec), format(want, ',')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())