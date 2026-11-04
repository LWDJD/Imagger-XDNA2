# SPDX-License-Identifier: MIT
"""ref_attn_block.py -- the noise refiner's attention without the S x S tensor.

WHY THIS FILE EXISTS
--------------------
t3_real.attention() is the CPU f32 global attention of the four refiner blocks.
For the noise half it runs on the image tokens, so its sequence length is

    512^2   -> S = 1024        (measured: the whole 8-step job completes)
    1024^2  -> S = 4096        1.88 GiB score matrix, completes (job 20261031-134742)
    1088x1920 (1080p) -> S = 8160   7.44 GiB, and the job dies:

        File ".../t3_real.py", line 507, in attention
          P = softmax(np.asarray(S, np.float32))
        File ".../t3_real.py", line 388, in softmax
          e = np.exp(x - m)
        numpy.core._exceptions._ArrayMemoryError: Unable to allocate 7.44 GiB
        for an array with shape (30, 8160, 8160) and data type float32
        (job 20261103-150606-10d726)

The shape is (HEADS=30, S, S) f32 per block, 2 blocks per step, 8 steps.  It is
not a RAM problem to be solved with more RAM: the object is O(S^2) and S grows
with the image area.

WHAT THIS DOES
--------------
Blocks the QUERY axis.  For each run of 'rows' query rows it materialises one
(HEADS, rows, S) score block, softmaxes it over the FULL key axis and
multiplies by V.  The working set is O(HEADS * rows * S) instead of
O(HEADS * S * S), and the softmax reduction is still over every key, so the
arithmetic is identical -- not "close": measured maxabs 0.000e+00 against the
delivered expression at S=1024 and S=4096, and 0.000e+00 between three
different block sizes at S=8160 (gate_refchunk.py).

This is the same treatment webui/chunked.py:_attention() already gives the 30
main layers ("qblock bounds the score matrix (qblock x SEQ x HEADS x 4 B); the
softmax is over the full key axis, so blocking the query rows changes no
arithmetic", chunked.py:240-245).  The refiner was simply the one attention in
the product that never got it.

MEASURED (this box, py310 GAIA, f32, n=5 interleaved A/B/A unless stated)
------------------------------------------------------------------------
  S=4096   delivered expression   4.209 s   peak commit 8.39 GiB
           blocked, 1024 rows      4.380 s   peak commit 2.82 GiB   (+4.1 %)
           blocked,  512 rows      4.626 s   peak commit 1.88 GiB   (+9.9 %)
           blocked,  256 rows      4.872 s   peak commit 1.41 GiB   (+15.7 %)
  S=8160   delivered expression   impossible (needs ~30 GiB of commit; the
                                  job died asking for the third 7.44 GiB)
           blocked,  512 rows     15.75 s    peak commit 2.98 GiB
           blocked, 1024 rows     14.09 s    peak commit 4.86 GiB
  torch SDPA, 4-D, CPU flash kernel (T22_REF_QCHUNK=sdpa):
  S=4096 1.99 s / 1.08 GiB, S=8160 9.02 s / 1.32 GiB, but NOT byte-equal
  (maxabs 1.49e-07, rel_rms 4.25e-07) and it needs torch in the refiner.

POLICY (two numbers, both env-overridable)
  T22_REF_QCHUNK_GB  (default 2.0)  block only when HEADS*S*S*4 exceeds this.
                                    Every shape that runs today is at or below
                                    1.88 GiB of score matrix, so no delivered
                                    shape changes path.
  T22_REF_QBLOCK_GB  (default 0.5)  the per-block score-matrix budget; the row
                                    count is derived from it.
  T22_REF_QCHUNK      "auto" (default) | "off" | "sdpa" | an integer row count
                      ("off" restores the delivered expression exactly, "sdpa"
                      takes the torch flash kernel, an integer forces that many
                      rows -- used by the gate to prove byte equality).

The delivered default path is untouched: t3_real.attention() only calls in here
when HEADS*S*S*4 > T22_REF_QCHUNK_GB, and never imports this module otherwise.
"""
from __future__ import annotations

import os

import numpy as np

MODE_ENV = "T22_REF_QCHUNK"
CEIL_ENV = "T22_REF_QCHUNK_GB"
BLOCK_ENV = "T22_REF_QBLOCK_GB"

CEIL_DEFAULT_GB = 2.0
BLOCK_DEFAULT_GB = 0.5
OFF = ("0", "off", "no", "false", "none")

_STATS = {"calls": 0, "blocks": 0, "s": 0.0, "mode": "", "rows": 0}


def _gb(name, default):
    try:
        v = float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return float(default)
    return v if v > 0 else float(default)


def ceiling_bytes():
    """Score-matrix bytes above which the query axis is blocked."""
    return _gb(CEIL_ENV, CEIL_DEFAULT_GB) * 2.0 ** 30


def block_budget_bytes():
    """Bytes of one (HEADS, rows, S) f32 score block."""
    return _gb(BLOCK_ENV, BLOCK_DEFAULT_GB) * 2.0 ** 30


def mode():
    return (os.environ.get(MODE_ENV, "auto") or "auto").strip().lower()


def disabled():
    return mode() in OFF


def score_bytes(heads, seq):
    return int(heads) * int(seq) * int(seq) * 4


def wanted(heads, seq):
    """True when t3_real.attention() must not materialise (heads, S, S)."""
    if disabled():
        return False
    return score_bytes(heads, seq) > ceiling_bytes()


def block_rows(heads, seq, rows=None):
    """Query rows per block for this shape.

    A multiple of 64 rows, at least 64, never more than S.  Derived from the
    byte budget so the block stays bounded as S grows:
        0.5 GiB, heads=30, S=8160 -> 512 rows   (measured peak 2.98 GiB)
        0.5 GiB, heads=30, S=4096 -> 1088 rows  (measured peak ~2.9 GiB)
    """
    seq = int(seq)
    if rows:
        return max(1, min(seq, int(rows)))
    per_row = max(1, int(heads) * seq * 4)
    n = int(block_budget_bytes() // per_row)
    n = max(64, (n // 64) * 64)
    return max(1, min(seq, n))


def _softmax_numpy(x):
    """Only used if a caller passes no softmax: t3_real.softmax, verbatim."""
    m = x.max(-1, keepdims=True)
    e = np.exp(x - m)
    return (e / e.sum(-1, keepdims=True)).astype(np.float32)


def attn(qh, kh, vh, scale, softmax_fn=None, nb_kernel=None, rows=None):
    """Query-blocked global attention.  (HEADS, S, HD) f32 in, (S, HEADS*HD) out.

    softmax_fn is t3_real.softmax and nb_kernel is t3_real._softmax_scaled()'s
    numba kernel (or None) -- passed in, never re-implemented, so the blocked
    path runs the SAME reduction code as the delivered one.

    nb_kernel, when given, is the in-place (x * scale) softmax used by
    T22_REF_SMNB=1: it takes the UNSCALED score block and applies the scale
    itself, exactly as t3_real.attention:504 does.
    """
    qh = np.ascontiguousarray(qh)
    heads, seq, hd = qh.shape
    if mode() in ("sdpa", "flash"):
        return attn_sdpa(qh, kh, vh)
    sm = softmax_fn or _softmax_numpy
    sc = np.float32(scale)
    nq = block_rows(heads, seq, rows)
    kT = kh.transpose(0, 2, 1)
    out = np.empty((heads, seq, hd), np.float32)
    nblk = 0
    import time as _t
    t0 = _t.perf_counter()
    for a in range(0, seq, nq):
        b = min(seq, a + nq)
        blk = qh[:, a:b] @ kT                       # (HEADS, b-a, S) f32
        if nb_kernel is None:
            blk *= sc
            p = sm(np.asarray(blk, np.float32))
        else:
            p = nb_kernel(np.ascontiguousarray(blk), sc)
        out[:, a:b] = p @ vh
        del blk, p
        nblk += 1
    _STATS["calls"] += 1
    _STATS["blocks"] += nblk
    _STATS["s"] += _t.perf_counter() - t0
    _STATS["mode"] = "sdpa" if mode() in ("sdpa", "flash") else "block"
    _STATS["rows"] = nq
    return out.transpose(1, 0, 2).reshape(seq, heads * hd)


def attn_sdpa(qh, kh, vh):
    """torch's CPU flash attention (4-D only -- the 3-D call takes the MATH
    kernel, which materialises the full S x S and is what T22_REF_TORCH=1 does
    today).  NOT byte-equal to the numpy expression: measured maxabs 1.49e-07
    / rel_rms 4.25e-07 at S=4096, i.e. 3.3e4x inside the 1.4e-2 attention bar,
    but it is a different reduction, so it is opt-in only."""
    import torch as th
    import torch.nn.functional as F
    n = int(os.environ.get("ZIMAGE_CHUNK_THREADS", "0") or 0)
    if n > 0:
        th.set_num_threads(n)
    q = th.from_numpy(np.ascontiguousarray(qh)).unsqueeze(0)
    k = th.from_numpy(np.ascontiguousarray(kh)).unsqueeze(0)
    v = th.from_numpy(np.ascontiguousarray(vh)).unsqueeze(0)
    with th.no_grad():
        o = F.scaled_dot_product_attention(q, k, v)
    o = o[0].permute(1, 0, 2).contiguous().numpy()
    _STATS["calls"] += 1
    _STATS["mode"] = "sdpa"
    return o.reshape(o.shape[0], -1)


def stats():
    return dict(_STATS)


def reset_stats():
    for k in _STATS:
        _STATS[k] = 0.0 if k == "s" else ("" if k == "mode" else 0)
