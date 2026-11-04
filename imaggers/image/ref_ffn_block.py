# SPDX-License-Identifier: MIT
"""ref_ffn_block.py -- the noise refiner's FFN without the (S, 10240) block.

WHY THIS FILE EXISTS
--------------------
t3_real.ffn_raw is the CPU f32 FFN of the four refiner blocks:

    lin(silu(lin(m, w1)) * lin(m, w3), w2)

m is (S, 3840) and the intermediate is (S, 10240) f32, so the call needs
S * 10240 * 4 bytes per live intermediate and holds three of them at once
(w1 out, w3 out, the product) plus silu's own output.  At the 1080p shape
S = 8160 that is

    8160 * 10240 * 4 = 334,233,600 B = 318.75 MiB  per intermediate

and the job died asking for exactly that (job 20261103-230421-e19b34,
rq_1080p_8f):

    File ".../t3_real.py", line 562, in ffn_raw
      return lin(silu(lin(m, Wb["w1"])) * lin(m, Wb["w3"]), Wb["w2"])
    numpy.core._exceptions._ArrayMemoryError: Unable to allocate 319. MiB for
    an array with shape (8160, 10240) and data type float32

This is the same shape of problem ref_attn_block.py solved for the refiner's
attention, and it gets the same treatment: block the QUERY axis.  Unlike
attention, the FFN is ROW-LOCAL -- row i of the output depends only on row i of
the input -- so blocking cannot change a single FLOP, let alone a reduction
order.  The only question is whether the BLAS call on a row SUB-block returns
bit-identical rows to the call on the whole block; the gate in
gate_ref_ffn.py measures that, and the gate is what makes this file's default
safe rather than optimistic.

WHY THE DELIVERED SHAPES DO NOT CHANGE
--------------------------------------
The policy is the same two-number shape as ref_attn_block.py:

  T22_REF_FFN_GB    (default 0.30) block only when rows*inner*4 exceeds this.
                    512^2   S = 1024 ->  40.0 MiB   delivered expression
                    1024^2  S = 4096 -> 160.0 MiB   delivered expression
                    1080p   S = 8160 -> 318.8 MiB   blocked
  T22_REF_FFN_BLOCK_GB (default 0.0625 = 64 MiB) the per-block intermediate
                    budget; the row count is derived from it.  Measured at
                    S=8160 on this box: 6528 rows -> +0.664 GiB over base,
                    1600 rows -> +0.19 GiB, 1024 rows -> +0.032 GiB.
  T22_REF_FFN_ROWS  (default 0) rows per block, overriding the budget.
  T22_REF_FFN        "auto" (default) | "off" | an integer row count.

"off" restores the delivered expression exactly, and is what the gate uses for
the reference side.
"""
from __future__ import annotations

import os

import numpy as np

MODE_ENV = "T22_REF_FFN"
CEIL_ENV = "T22_REF_FFN_GB"
BLOCK_ENV = "T22_REF_FFN_BLOCK_GB"
ROWS_ENV = "T22_REF_FFN_ROWS"

CEIL_DEFAULT_GB = 0.30
BLOCK_DEFAULT_GB = 0.0625
OFF = ("0", "off", "no", "false", "none")

_STATS = {"calls": 0, "blocked": 0, "rows": 0, "seq": 0}


def _gb(name, default):
    try:
        v = float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return float(default)
    return v if v > 0 else float(default)


def mode():
    return (os.environ.get(MODE_ENV, "auto") or "auto").strip().lower()


def disabled():
    return mode() in OFF


def ceiling_bytes():
    """Intermediate bytes above which the query axis is blocked."""
    return _gb(CEIL_ENV, CEIL_DEFAULT_GB) * 2.0 ** 30


def block_budget_bytes():
    """Bytes of ONE (rows, inner) f32 intermediate inside a block."""
    return _gb(BLOCK_ENV, BLOCK_DEFAULT_GB) * 2.0 ** 30


def block_bytes(rows, inner):
    """Bytes of ONE (rows, inner) f32 intermediate -- what a block costs."""
    return int(rows) * int(inner) * 4


def wanted(rows, inner):
    """True when t3_real.ffn_raw must not materialise (rows, inner) at once."""
    if disabled():
        return False
    if rows <= 0 or inner <= 0:
        return False
    return block_bytes(rows, inner) > ceiling_bytes()


def block_rows(inner, rows=None):
    """Rows per block.  A multiple of 64, at least 64, never more than S."""
    forced = rows
    if forced is None:
        try:
            forced = int(os.environ.get(ROWS_ENV, "0") or 0)
        except (TypeError, ValueError):
            forced = 0
    inner = max(1, int(inner))
    if forced and forced > 0:
        return int(forced)
    per_row = inner * 4
    n = int(block_budget_bytes() // per_row) if per_row else 0
    n = max(64, (n // 64) * 64)
    return max(1, n)


def ffn(m, w1, w3, w2, lin, silu, rows=None):
    """(rows, 3840) f32 -> (rows, 3840) f32, one query block at a time.

    lin/silu are t3_real's own functions, passed in and never re-implemented,
    so the blocked path runs the SAME arithmetic -- exactly the contract
    ref_attn_block.attn keeps with t3_real.softmax.
    """
    m = np.asarray(m, np.float32)
    inner = int(np.asarray(w1).shape[0])
    out_cols = int(np.asarray(w2).shape[0])
    nq = block_rows(inner, rows)
    out = np.empty((m.shape[0], out_cols), np.float32)
    nblk = 0
    for a in range(0, m.shape[0], nq):
        b = min(m.shape[0], a + nq)
        blk = m[a:b]
        out[a:b] = lin(silu(lin(blk, w1)) * lin(blk, w3), w2)
        nblk += 1
    _STATS["calls"] += 1
    _STATS["blocked"] += nblk
    _STATS["rows"] = nq
    _STATS["seq"] = int(m.shape[0])
    return out


def stats():
    return dict(_STATS)
