# SPDX-License-Identifier: MIT
"""s2_product.py -- the paste-in shim for the product (drop-in signatures).

The product keeps calling with its own device handle; this module owns the
S2Dev singleton, the one-time prewarm and the 1056-row chunking, so the
product change is ONE call each:

    y = s2_product.ffn32_run(_BIGM_DEV["d"], _xn, key, tim)          # (seq,3840) f32
    Q = s2_product.proj4(_BIGM_DEV["d"], x1, key, tim)               # {'q','k','v','o'}

Both accept ANY seq (the images are fixed at 1056 = 1024 + 32): rows are
chunked into 1056-row slabs and zero-padded inside a slab, then the junk rows
are dropped.  Rows are independent, so padding cannot change the answer.
"""
from __future__ import annotations

import atexit
import os
import pathlib
import sys
import time as _time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

M_FIXED = 1056
_DEV = None
_WARM = False


def _ensure():
    global _DEV, _WARM
    import s2_dev
    if not _WARM:
        # NO bulk prewarm: the payload cache is built per layer on first use
        # and LRU-bounded (T22_S2_CACHE_LAYERS, default 4 = 1.6 GB).  A miss
        # costs one repack of that layer (0.65 s).  Bulk prewarm is opt-in:
        #   s2_product.prewarm([f"layers.{i}" for i in range(30)])
        _WARM = True
    if _DEV is None:
        _DEV = s2_dev.S2Dev()
        atexit.register(shutdown)
    return _DEV


def shutdown():
    global _DEV
    if _DEV is not None:
        try:
            _DEV.close()
        finally:
            _DEV = None


def _slabs(seq):
    for r0 in range(0, seq, M_FIXED):
        yield r0, min(M_FIXED, seq - r0)


def _pad(x, rows):
    blk = np.zeros((M_FIXED, x.shape[1]), np.float32)
    blk[:rows] = x[:rows]
    return blk


def ffn32_run(dev_ignored=None, xn=None, key=None, tim=None):
    """(seq, 3840) f32 -> (seq, 3840) f32.  Same signature the product uses."""
    import s2_dev
    x = np.asarray(xn, np.float32)
    seq = x.shape[0]
    out = np.empty_like(x)
    dev = _ensure()
    for r0, rows in _slabs(seq):
        y = s2_dev.ffn32_run(dev, _pad(x[r0:r0 + rows], rows), key, tim)
        out[r0:r0 + rows] = y[:rows]
    return out


def prewarm(keys):
    """Bulk prewarm (opt-in): each layer is 398 MB on disk (8 payloads)."""
    import s2_dev
    return s2_dev.prewarm_ball(list(keys))


_XCHECKED = False


def _xcheck(dev, x, key, names, max_tiles):
    """T22_S2_XCHECK=1: run the OLD product proj4 on the same x, once.

    The old path needs the wrapped device the product passes (it has .bo and
    .imgs), so this runs inside the product, not in a standalone script.
    """
    global _XCHECKED
    if _XCHECKED or os.environ.get("T22_S2_XCHECK", "0") != "1":
        return None
    _XCHECKED = True
    try:
        sys.path.insert(0, str(HERE.parents[1] / "devkill"))
        import prod_devkill as PD
        t0 = _time.perf_counter()
        old = PD.proj4(dev, x, key, names or ("q", "k", "v", "o"), None,
                       max_tiles)
        t_old = _time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        print("[s2-xcheck] old path failed: %r" % (exc,), flush=True)
        return None
    new = _proj_core(dev, x, key, names, max_tiles)
    print("[s2-xcheck] x %s key %s" % (getattr(x, "shape", None), key),
          flush=True)
    for nm in ("q", "k", "v", "o"):
        if nm in old and nm in new:
            a = np.asarray(old[nm], np.float64).reshape(-1)
            b = np.asarray(new[nm], np.float64).reshape(-1)
            print("  %s corr %.6f maxabs %.4f  (old %.1f ms)"
                  % (nm, np.corrcoef(a, b)[0, 1],
                     float(np.abs(a - b).max()), t_old * 1e3), flush=True)
    # third opinion: the plain projection from the CHECKPOINT weights
    try:
        import s2_proj32 as P2
        import s2_ffn32 as S2
        w = P2.load_proj_weights(key)
        xq = S2._bf16(np.asarray(x, np.float32))
        for nm, ck in (("q", "to_q"), ("k", "to_k"), ("v", "to_v"),
                       ("o", "to_out.0")):
            ref = S2._bf16(xq @ S2._decode_blocks(w[ck], 3840, 3840))
            if nm in new and nm in old:
                for tag, arr in (("old", old[nm]), ("new", new[nm])):
                    a = np.asarray(arr, np.float64).reshape(-1)
                    b = ref.astype(np.float64).reshape(-1)
                    print("  %s vs checkpoint-math: %s corr %.6f"
                          % (nm, tag, np.corrcoef(a, b)[0, 1]), flush=True)
    except Exception as exc:  # noqa: BLE001
        print("[s2-xcheck] checkpoint ref failed: %r" % (exc,), flush=True)
    np.savez(str(HERE / "out" / "xcheck_proj.npz"),
             **{("old_" + k): v for k, v in old.items()},
             **{("new_" + k): v for k, v in new.items()})
    return old


def proj4(dev_ignored=None, x=None, key=None, names=None, tim=None,
          max_tiles=11):
    """(seq, 3840) f32 -> {'q','k','v','o'} each (seq, 3840) f32.

    REAL signature of the devkill path (prod_devkill.py:118/139):
        proj4(dev, x, key, names, tim=None, max_tiles=11)
    the FOURTH positional is 'names' (e.g. ('q','k','v') or ('o',)), NOT tim.
    'names' is accepted but the fused image always computes all four in one
    pass over N=15360, so we return the full dict: indexing by the requested
    subset works and the extra keys cost nothing.
    """
    old = _xcheck(dev_ignored, x, key, names, max_tiles)
    if old is not None:
        return old          # fail-open: the A/B must not change the product
    return _proj_core(dev_ignored, x, key, names, max_tiles)


def _proj_core(dev_ignored, x, key, names, max_tiles):
    import os as _os
    import time as _time
    import s2_dev
    _t0 = _time.perf_counter()
    arr = np.asarray(x, np.float32)
    seq = arr.shape[0]
    out = {n: np.empty_like(arr) for n in ("q", "k", "v", "o")}
    dev = _ensure()
    for r0, rows in _slabs(seq):
        # parent fix 2026-11-01: do NOT write into the product's `tim` dict --
        # it accumulates scalars (t18_run sums it); s2's own stats are recorded
        # inside proj32_run regardless.
        o = s2_dev.proj32_run(dev, _pad(arr[r0:r0 + rows], rows), key, None)
        for n in out:
            out[n][r0:r0 + rows] = o[n][:rows]
    if _os.environ.get("T22_S2_PROF", "0") == "1":
        print("[s2] proj4 %s %.1f ms (incl. any repack)"
              % (key, 1e3 * (_time.perf_counter() - _t0)), flush=True)
    return out
