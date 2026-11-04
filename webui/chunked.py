# SPDX-License-Identifier: MIT
"""T22 chunk_path: arbitrary-M execution on the delivered M=96 kernels.

The product images are M=96 (projection/FFN row blocks) and a fused ip that
needs the whole key axis in one core's L1 (SEQ <= 304).  A shape above that
ceiling cannot use the ip image, but every *linear* row-wise operation can:
projection, FFN and the o projection are row-independent, so the delivered
M=96 images run them in 96-row blocks unchanged.

What is left is attention, which is O(SEQ^2) and needs the whole sequence.  This
module computes it on the host in f32 with torch (AVX-512/OpenMP), using the
SAME q/k/v bytes the device projections produced and the SAME rms+RoPE host
code as the delivered attention_fast path.  The device still executes every
weight-carrying GEMM; only the QK^T / softmax / PV product is host f32.

This is deliberately NOT a new device kernel: no image is compiled, no
arithmetic is changed in the delivered paths, and the M=96 cache profiles are
untouched.  The numerical tier is F-23 ("declared tier"): attention component
vs the device blocked reference measured corr 0.99997 / rel 0.72 % at 480p,
which is the BFP16 floor (T22-PROTOTYPE.md 162-197).  Chain latents therefore
are not byte-identical to the device blocked driver; the honest tier and the
measured drift are recorded by the gate script.

T22 ATTHEV (this campaign): the 480p loop spent 201.6 s outside the device, of
which the host attention is only ~58 s (measured 0.243 s/layer, t22_attdev
ad_bench1.py); the rest is BO staging (42 s), host packing (31 s) and misc
(70 s).  So the win is not "move attention to the device" (that caps at ~58 s),
it is OVERLAP: the ERT accepts a launch and returns, so the host can work under
an in-flight dispatch (t22_attdev ad_probe_pipe.py: 30 ms of host work hides
completely under a 36.2 ms FFN block, byte-identical output).

block_step_pipelined() below is that overlap, and nothing else:

  * q/k/v projections: unchanged dispatches; rms+RoPE is folded into the same
    row-block loop (row-local, bit-identical -- ad_probe_bits.py proves the
    attention itself is bit-identical for any query-block size);
  * then a 96-row software pipeline, two launches deep, double-buffered BOs:
        o-proj(j) -> ffn(j) -> o-proj(j+1) -> ffn(j+1) ...
    while the host computes attention(j+1), the norms, the bf16/bf16 packs and
    the drains.  The o-projection sits between two FFN launches, so the
    delivered state-8 barrier (Loop.ffn) is no longer needed -- the different
    xclbin it inserted is now real work;
  * BG/B3 (132.7 MB, identical on every one of the 18 blocks of a layer) and
    SBG are written once per layer instead of 18 times (write_ms ffn 23.3 s
    was 17/18 pure duplicate bytes).

Every value written to every BO is the same byte the delivered path writes, so
the chain output stays byte-identical (gate: latents 455ad0e947ef20c1).

The install() hook is called by t18_run.main() only when t18_run.configure()
set CHUNKED=True (SEQ > 304).  t13/t18 delivered paths never import this file.
ZIMAGE_CHUNK_PIPE=0 restores the T22-CHUNK-PATH schedule (host attention only,
no overlap) for A/B measurement.
"""
from __future__ import annotations

import os
import time

import numpy as np

_T = None
_ORIG_BLOCK_STEP = None
# diagnostics for the state-8 trigger hunt (all OFF = the delivered behaviour is
# "refill every block + barrier"; the pipeline's claim is that neither is needed)
_DIAG = {"ffn_refill": os.environ.get("AD_FFN_REFILL", "0") == "1",
         "ffn_barrier": os.environ.get("AD_FFN_BARRIER", "0") == "1"}
# the last completion instant, so the device time booked per dispatch is the
# BUSY interval and not the submit->completion window (which contains whatever
# the host did under it).  Joins happen in submission order, so this is exact
# while the queue is non-empty and an over-estimate only across a real idle gap.
_LAST_DONE = {"t": 0.0}


def _t18():
    global _T
    if _T is None:
        import t18_run as T
        _T = T
    return _T


def bg_order_guard(nh_bg, bg_preperm, where="submit_f"):
    """Refuse to write a BG payload the FFN kernel will read in the wrong order.

    NH_BG > 1 means the kernel's BG image was compiled for the NH-permuted
    layout, so whatever is written to f_BG must already be in that order.  In
    the delivered loop that is Loop.ffn's job (t18_run.py:1876 permutes into a
    double buffer when BG_PREPERM is False) -- but block_step_pipelined.
    submit_f() writes the loop's bg VERBATIM and never calls Loop.ffn, so it is
    the one place in the tree where the memory-mode difference reaches a BO
    with no compensating transform:

        --cache-all     hands submit_f CACHE_BG[li]   (pre-permuted, BG_PREPERM=True)
        --no-cache-all  hands submit_f the raw .npy   (delivered order, BG_PREPERM=False)

    On the plain 96-row pipeline NH_BG is 0 (t22_run.py:693), the kernel wants
    the delivered order, and both arms are correct -- which is why the 480x864
    low runs matched the fast reference byte for byte.  The moment NH_BG > 1
    this path is wrong unless the buffer really is pre-permuted, so fail loudly
    instead of dispatching bytes the kernel will read in the wrong order.  (The
    512^2 and 1080p runs do not use this seam: their logs carry [bigm]/[devkill]
    lines, i.e. they dispatch the big-M FFN whose payload prod_loop_bigm builds.)
    """
    nh = int(nh_bg or 0)
    if nh > 1 and not bg_preperm:
        raise AssertionError(
            "chunked.%s: NH_BG=%d (the FFN kernel expects the NH-permuted BG "
            "order) but BG_PREPERM=%r: the buffer handed to f_BG is in the "
            "delivered .npy order and this path never calls Loop.ffn, so "
            "nothing permutes it.  Run with --cache-all (BG_PREPERM=True) or "
            "dispatch this shape through the big-M/devkill FFN."
            % (where, nh, bg_preperm))


def _bg_fills(T, bo, bg, b3):
    """The per-layer BG/B3 BO fills, gated on the payload order."""
    bg_order_guard(getattr(T, "NH_BG", 0), getattr(T, "BG_PREPERM", False))
    return [(bo["f_BG"], bg), (bo["f_B3"], b3)]


def _chunk_threads() -> int:
    """ZIMAGE_CHUNK_THREADS through webui.config (one home for the rule).

    Falls back to the raw environment when this module is imported standalone
    (a few imaggers harnesses do "import chunked" with webui/ on sys.path),
    which is the same parse config.loop_threads() performs.
    """
    try:
        from . import config as _cfg
        return int(_cfg.loop_threads())
    except Exception:                                        # noqa: BLE001
        try:
            return max(0, int(os.environ.get("ZIMAGE_CHUNK_THREADS", "0") or 0))
        except Exception:                                    # noqa: BLE001
            return 0


_LOOP_THREADS = {"logged": False}


def _torch():
    import torch
    n = _chunk_threads()
    if n > 0:
        torch.set_num_threads(n)
    if not _LOOP_THREADS["logged"]:
        # t22_loud: the loop is the stage that costs 90 s, and until this line
        # existed its thread pool could not be proved from a run -- only the
        # CONFIGURED value was recorded (and None when it was unset, while torch
        # then used its own default).  The VAE stage has printed its pool since
        # 2026-11-03 (vae_local.py:89); this is the same fact for the loop.
        _LOOP_THREADS["logged"] = True
        _got = int(torch.get_num_threads())
        print("[chunk] torch intra-op threads: %d (ZIMAGE_CHUNK_THREADS=%r -> %d)"
              % (_got, os.environ.get("ZIMAGE_CHUNK_THREADS"), n), flush=True)
        if n > 0 and _got != n:
            raise SystemExit(
                "[chunk] REFUSAL: ZIMAGE_CHUNK_THREADS=%r asked for %d threads "
                "and torch reports %d, so the loop would run on a pool nobody "
                "asked for (the wall and the VAE decode inheritance both move)."
                % (os.environ.get("ZIMAGE_CHUNK_THREADS"), n, _got))
    return torch


_NB_SM = None
_NB_SM_BUF = {}


def _nb_softmax():
    # Lazily build the numba last-axis softmax (f32, parallel, cached).
    # Measured here for a (60, 512, 1056) f32 block: torch 32.1 ms,
    # numba 18.5 ms, maxabs 8.9e-08.  None => caller falls back to torch.
    global _NB_SM
    if _NB_SM is None:
        try:
            from numba import njit, prange

            @njit(parallel=True, fastmath=True, cache=True)
            def _kernel(a, out):
                rows = a.shape[0] * a.shape[1]
                S = a.shape[2]
                af = a.reshape(rows, S)
                of = out.reshape(rows, S)
                for i in prange(rows):
                    m = -1e30
                    for j in range(S):
                        v = af[i, j]
                        if v > m:
                            m = v
                    s = 0.0
                    for j in range(S):
                        e = np.exp(af[i, j] - m)
                        of[i, j] = e
                        s += e
                    inv = 1.0 / s
                    for j in range(S):
                        of[i, j] *= inv
            _NB_SM = _kernel
        except Exception:
            _NB_SM = False
    return _NB_SM or None


_BCORE = None


def _bcore():
    # fused numba softmax kernels (bf16/f32, register bitcast, in place)
    global _BCORE
    if _BCORE is None:
        try:
            import sys as _s
            import pathlib as _pl
            _d = str(_pl.Path(__file__).resolve().parents[1] / "imaggers" / "util" / "glue")
            if _d not in _s.path:
                _s.path.insert(0, _d)
            import bcore_tail as _b
            _BCORE = _b._kernels()
        except Exception:
            _BCORE = False
    return _BCORE or None


_ATT_BUF = {}
_ATT_DUMP = {"n": 0}
# T22_ATTN_OBUF: (seq, heads, hd) -> [numpy f32 buffer, torch view].  One
# persistent output buffer for the whole run (see _attention).
_ATT_OBUF = {}


# ---------------------------------------------------------------------------
# t22_variants -- the RESOLVED STATE of every attention-variant switch.
#
# A2, E and D are BYTE-GATED arms (ATTN-SEG-REPORT: D = f9537c176918), so a
# variant that could not be applied is invisible in the output: it publishes the
# CONTROL's bytes under the variant's name.  Until this block the drop was one
# line -- "if _use_d and _bc is None: _use_d = False" -- with no print and no
# record, and A2/E fell through the identical way because their branches test
# "_bc is not None".  The X3 census (630 job records, 2026-11-03) found 0
# cancellations and 0 records that could have shown one: the only arm that ever
# asked for D (20261102-115213-3fca3c, arm bt1) applied it, and the only
# evidence is a "[attnprew] sm_bf16_bf16" line that lives in no record.  Luck.
#
# The rules, in ONE home:
#   * the switch is read with the SAME test the branch uses ("==1"), so the
#     record cannot claim a variant the loop would not run; a truthy spelling
#     that test cannot honour ('true', 'on') gets its own status
#     (BAD-SPELLING) instead of being a silent no-op;
#   * a requested variant that cannot engage is a REFUSAL at install time;
#   * the ONLY way past the refusal is an explicit T22_GLUE_FAIL_OPEN=1 or
#     T22_GLUE_HALF_OK=1, and the record then says "operator chose fail-open";
#   * T22_ATTN_STRICT=1 refuses even when a hatch is open.
#
# The verdict is ONE line.  webui/runner.py parses it into job.json
# (attn_variants) and imaggers/repro/t22_repro.py Q19 gates the run on it, so a
# cancelled variant is visible in the record AND rejected by the postflight.
_ATTN_VARIANTS = (("D", "T22_ATTN_D"), ("A2", "T22_ATTN_A2"),
                  ("E", "T22_ATTN_E"))
_ATTN_TRUTHY = ("1", "true", "yes", "on", "y", "t")
_ATTN_STATE = {"done": False, "bcore": None, "variants": {}, "requested": [],
               "fail_open": False, "strict": False, "policy": "refuse",
               "line": "", "loud": False, "said_fail_open": False}


def _attn_truthy(name):
    """The tree's truthiness rule (config._TRUTHY), without importing config.

    chunked.py is imported standalone by harnesses with "webui/" on sys.path
    (:83-97 does the same for the thread knob), so this stays local.  It is
    used ONLY to decide that a switch was ASKED for; the branch selector below
    stays the literal "==1" it has always been.
    """
    return str(os.environ.get(name, "") or "").strip().lower() in _ATTN_TRUTHY


def attn_variant_state(refresh=False):
    """Resolve every attention-variant switch ONCE; return the state dict.

    bcore_tail is imported only when something was actually requested, so the
    delivered path pays nothing and imports nothing new.  The answer is what
    job.json and the postflight read -- never the raw environment.
    """
    if _ATTN_STATE["done"] and not refresh:
        return dict(_ATTN_STATE)
    bc, checked, out, req = None, False, {}, []
    for key, env in _ATTN_VARIANTS:
        raw = os.environ.get(env)
        spelled = raw is not None and str(raw).strip() != ""
        asked = spelled and _attn_truthy(env)
        exact = spelled and str(raw).strip() == "1"
        if not asked:
            status = "off"
        elif not exact:
            status = "bad-spelling"
            req.append(key)
        else:
            req.append(key)
            if not checked:
                bc, checked = _bcore(), True
            status = "engaged" if bc is not None else "cancelled"
        out[key] = {"env": env, "raw": raw, "requested": asked,
                    "engaged": status == "engaged", "status": status,
                    "spelling_ok": exact or not asked}
    fail_open = (_attn_truthy("T22_GLUE_FAIL_OPEN")
                 or _attn_truthy("T22_GLUE_HALF_OK"))
    strict = _attn_truthy("T22_ATTN_STRICT")
    policy = ("refuse" if (strict or not fail_open)
              else "operator chose fail-open")
    line = ("[attn] variants %s bcore=%s policy=%s"
            % (" ".join("%s=%s" % (k, v["status"].upper())
                        for k, v in out.items()),
               "?" if not checked else ("yes" if bc is not None else "no"),
               policy))
    _ATTN_STATE.update({"done": True, "bcore": None if not checked else
                        bc is not None,
                        "variants": out, "requested": req,
                        "fail_open": fail_open, "strict": strict,
                        "policy": policy, "line": line})
    return dict(_ATTN_STATE)


def _attn_gate(verbose=True):
    """Print the verdict, and REFUSE a requested variant that cannot engage.

    Called at install() time (startup, before a single layer runs) and, as a
    fallback, from _attention(): a harness that imports chunked and asks for a
    variant must not silently get the control's bytes either.
    """
    st = attn_variant_state()
    if not st["requested"]:
        return st                     # nothing asked: the shipped path, silent
    if verbose and not _ATTN_STATE["loud"]:
        _ATTN_STATE["loud"] = True
        print(st["line"], flush=True)
    bad = [k for k in st["requested"]
           if st["variants"][k]["status"] in ("cancelled", "bad-spelling")]
    if not bad:
        return st
    why = "; ".join(
        "%s=%s: %s" % (k, st["variants"][k]["status"].upper(),
                       "the switch needs exactly '1' (both the reader and the "
                       "branch selector test for it, and neither moved)"
                       if st["variants"][k]["status"] == "bad-spelling"
                       else "bcore_tail is not importable")
        for k in bad)
    if st["fail_open"] and not st["strict"]:
        if verbose and not _ATTN_STATE["said_fail_open"]:
            _ATTN_STATE["said_fail_open"] = True
            print("[attn] FAIL-OPEN (%s): %s.  The run continues on the "
                  "shipped attention, so its latents and its wall are the "
                  "CONTROL's -- do not publish them as the variant's."
                  % (st["policy"], why), flush=True)
        return st
    raise SystemExit(
        "[attn] REFUSAL: %s requested and CANNOT be applied (%s).  The "
        "variant arms are byte-gated (D = f9537c176918), so this run would "
        "produce the CONTROL's latents under the variant's name -- a wrong "
        "conclusion, not a slow run.  Fix the cause, unset the switch, or "
        "accept the control path knowingly with T22_GLUE_FAIL_OPEN=1 (the "
        "record then says 'operator chose fail-open').%s"
        % (", ".join(bad), why,
           "  T22_ATTN_STRICT=1 is set, so a fail-open hatch is ignored."
           if st["strict"] else ""))


def _attn_prewarm(verbose=True):
    """Compile the fused softmax kernels during install(), not in the loop.

    Measured first-call costs in a fresh py310 process: sm_bf16_bf16 1.14 s,
    sm_bf16_f32 0.45 s, sm_f32_inplace 0.28 s (plus 0.27 s to import
    bcore_tail).  numba refuses to disk-cache the two bf16 kernels because they
    use get_num_threads()/get_thread_id(), which are dynamic globals.  Paid
    lazily that lands 1-1.4 s on step 0, i.e. on loop_s; paid here it lands on
    startup_s.  Same total, honest loop.  No-op unless a variant flag is on.
    """
    # t22_variants: prewarm what was RESOLVED ENGAGED, not what the env says.
    # attn_variant_state() is the one home for that decision, and a cancelled
    # variant never reaches here: the gate refused, or the operator chose
    # fail-open and the recorded path is the control.
    _st = attn_variant_state()
    want = [n for n, e in (("sm_f32_inplace", "A2"),
                           ("sm_bf16_bf16", "D"),
                           ("sm_bf16_f32", "E"))
            if (_st["variants"].get(e) or {}).get("engaged")]
    if not want:
        return {}
    bc = _bcore()
    if not bc:                     # unreachable while bcore gates engagement
        print("[attnprew] bcore unavailable -- variants will fall back",
              flush=True)
        return {}
    res = {}
    try:
        import time as _t
        u = np.zeros((8, 64), np.uint16)
        f32 = np.zeros((8, 64), np.float64).astype(np.float32)
        o16 = np.empty((8, 64), np.uint16)
        of = np.empty((8, 64), np.float32)
        for nm in want:
            t0 = _t.perf_counter()
            if nm == "sm_f32_inplace":
                bc[nm](f32)
            elif nm == "sm_bf16_bf16":
                bc[nm](u, o16, 1.0)
            else:
                bc[nm](u, of, 1.0)
            res[nm] = _t.perf_counter() - t0
        if verbose:
            print("[attnprew] %s (%.2f s, paid at install not in the loop)"
                  % (", ".join("%s %.2fs" % (k, v) for k, v in res.items()),
                     sum(res.values())), flush=True)
    except Exception as e:                                   # noqa: BLE001
        print("[attnprew] skipped: %r" % (e,), flush=True)
    return res


def _att_buf(name, shape, dtype):
    """A reused working buffer -- reachable only with T22_ATTN_REUSE=1.

    Keyed by (name, shape, dtype) so two variants never share a buffer.  Nothing
    in the shipped path calls this, so the delivered arithmetic and its
    allocations are untouched.
    """
    shape = tuple(int(x) for x in shape)
    k = (name, shape, str(dtype))
    b = _ATT_BUF.get(k)
    if b is None:
        if getattr(dtype, "__module__", "").startswith("numpy"):
            b = np.empty(shape, dtype)
        else:
            import torch as _t
            b = _t.empty(shape, dtype=dtype)
        _ATT_BUF[k] = b
    return b


def _attention(qh, kh, v, qblock=0):
    """f32 exact global attention: (HEADS, SEQ, HD) in, (SEQ, D) out.

    qblock bounds the score matrix (qblock x SEQ x HEADS x 4 B); the softmax is
    over the full key axis, so blocking the query rows changes no arithmetic.
    """
    torch = _torch()
    heads, seq, hd = qh.shape
    # T22_ATTN_QB: override the query-block size.  The softmax is over the full
    # key axis, so the blocking changes NO arithmetic -- the gate shows identical
    # bytes at 256/352/1056 -- only how many rows share a score buffer (memory
    # is qblock * SEQ * HEADS * 4 B for f32, half that for bf16).  Measured at
    # 30x1056x128: the fused-softmax variants gain 7-19% going 512 -> 1056
    # because the qk matmul then sees one contiguous batch instead of a strided
    # slice.  Default 0 = the shipped 512/256 rule.
    _qb = int(os.environ.get("T22_ATTN_QB", "0") or 0)
    if not qblock:
        qblock = _qb or (512 if seq <= 8192 else 256)
    # 2026-11-01: opt-in bf16 matmul (f32 accumulate on AVX512-BF16).  Measured
    # on this machine, 60 heads x 1056 x 64: f32 55.9 ms -> bf16 27.6 ms
    # (306 -> 621 GFLOP/s).  Default 0 keeps the shipped arithmetic untouched.
    if os.environ.get("T22_ATTN_BF16", "0") == "1":
        dt = torch.bfloat16
    else:
        dt = torch.float32
    import time as _at
    _prof = os.environ.get("T22_ATTN_PROF", "0") == "1"
    _use_nb = os.environ.get("T22_ATTN_NB", "0") == "1"
    # 2026-11-02: variant E.  bf16 qk matmul + an f32 in-place numba softmax +
    # f32 pv.  Isolated medians per 512-row block (this box): qk f32 11.0 ->
    # bf16 4.56, softmax torch-f32 30.0 -> numba-f32 16.9, pv f32 9.89 (kept).
    # Measured block total 34.7 ms vs today's 50.9 (-32%); the two variants that
    # were tried in-product before (all-bf16 / numba softmax alone) lost because
    # torch's bf16 softmax is 1.6x SLOWER than f32 (47.8 vs 30.0) and because
    # .to(bf16) of the 130 MB score matrix costs 28.2 ms on its own.
    _use_e = os.environ.get("T22_ATTN_E", "0") == "1"
    # 2026-11-02: variant A2 -- f32 matmul, fused numba f32 softmax IN PLACE (no
    # 130 MB copy out).  Measured on this box: 92.9 ms/layer vs 107.9 for the
    # shipped torch path (-14%) and corr 1.000000000 / relrms 1.3e-7, i.e. exact.
    _use_a2 = os.environ.get("T22_ATTN_A2", "0") == "1"
    # 2026-11-02: variant D -- bf16 qk matmul + the FUSED bf16->bf16 numba
    # softmax (bcore_tail.sm_bf16_bf16, fed the score matrix through a
    # zero-copy uint16 view and writing bf16 bits into a reused buffer) + bf16
    # pv.  What it adds over E is that the probabilities never become f32, so
    # the pv matmul runs at the bf16 rate as well (measured per 512-row block
    # at 30x1056x128: pv f32 6.97 -> bf16 5.63 ms).  Default 0.
    _use_d = os.environ.get("T22_ATTN_D", "0") == "1"
    # 2026-11-02: the FIXED-COST eliminations, bit-exact.  Both matmuls write
    # into reused out= buffers, the (heads,q,hd)->(q,heads,hd) store goes
    # through torch instead of a numpy strided assign (measured 1.07 -> 0.29 ms
    # per block), and the (seq,heads,hd) result buffer is reused across calls.
    # Measured on this box: -6% of the segment alone, -10% with the numba
    # threads raised, with IDENTICAL output bytes (bench2.py arm B).  Default 0.
    _reuse = os.environ.get("T22_ATTN_REUSE", "0") == "1"
    # T22_ATTN_OBUF=1: the FIXED-COST half of REUSE, on its own switch.  ONE
    # persistent (seq,heads,hd) f32 output buffer for the whole run (today the
    # 16.2 MB buffer is np.empty'd once per layer) and the PV result is stored
    # into out[q0:q1] by a torch copy through a transposed VIEW instead of
    # numpy's strided assign.  Measured on this box (bench_three.py, 30x1056x128):
    # 0.643 -> 0.054 ms for one 512-row store, and -3.73 ms per _attention call
    # on the interleaved A/B, most of it the 16.2 MB first-touch it also avoids.
    # It is a pure copy of
    # the same float32 values into the same positions -- no arithmetic, no
    # reordering -- so the output bytes are unchanged.  The matmuls and the
    # softmax are untouched (REUSE's out= buffers are NOT part of this flag).
    # LIFETIME CONTRACT: the returned (seq, D) is the SAME buffer until the next
    # call with the same (seq, heads, hd), so a caller must consume or copy it
    # before calling again.  Both live callers do (install.py:420 -> bf16_of into
    # its own o16 buffer; prod_loop_bigm.py:851 -> the same shape).
    # Default 0 leaves the delivered allocations and the numpy store as they are.
    _obuf = (not _reuse) and os.environ.get("T22_ATTN_OBUF", "0") == "1"
    # numba threads for the fused softmax only (0 = leave the process setting,
    # which fastglue pins at T22_FASTGLUE_THREADS=8).  The kernel is bandwidth
    # bound: measured on this box 14.2 / 8.4 / 6.5 / 5.8 ms per block at
    # 8 / 12 / 16 / 20 threads, against 10.7 ms for torch.softmax f32 at 16.
    _nbthr = int(os.environ.get("T22_ATTN_NBTHREADS", "0") or 0)
    _bc = _bcore() if (_use_e or _use_a2 or _use_d) else None
    if _use_e or _use_a2 or _use_d:
        # t22_variants: the RESOLVED state decides, not a per-call env re-read.
        # A cancelled variant is a REFUSAL (raised at install(); repeated here
        # for a harness that never ran install) and NEVER a silent fall-through
        # to the shipped path -- that fall-through IS X3: the run then
        # publishes the control's bytes under the variant's name.
        _attn_gate()
        _vs = _ATTN_STATE["variants"]
        _use_e = _use_e and _vs["E"]["engaged"]
        _use_a2 = _use_a2 and _vs["A2"]["engaged"]
        _use_d = _use_d and _vs["D"]["engaged"]
    _nb_prev = None
    if _nbthr > 0 and (_use_a2 or _use_e or _use_d):
        try:
            from numba import get_num_threads as _gnt, set_num_threads as _snt
            _nb_prev = int(_gnt())
            if _nb_prev != _nbthr:
                _snt(_nbthr)
        except Exception:
            _nb_prev = None
    _t0 = _at.perf_counter()
    if _use_d and _bc is not None:
        qt = torch.from_numpy(np.ascontiguousarray(qh)).to(torch.bfloat16)
        kt = torch.from_numpy(np.ascontiguousarray(kh)).to(torch.bfloat16)
        vt = torch.from_numpy(np.ascontiguousarray(v)).to(torch.bfloat16)
    elif _use_e and _bc is not None:
        qt = torch.from_numpy(np.ascontiguousarray(qh)).to(torch.bfloat16)
        kt = torch.from_numpy(np.ascontiguousarray(kh)).to(torch.bfloat16)
        vt = torch.from_numpy(np.ascontiguousarray(v))
        _use_nb = True                      # E always uses the numba f32 softmax
    else:
        qt = torch.from_numpy(np.ascontiguousarray(qh)).to(dt)
        kt = torch.from_numpy(np.ascontiguousarray(kh)).to(dt)
        vt = torch.from_numpy(np.ascontiguousarray(v)).to(dt)
    _e_out = None
    _t1 = _at.perf_counter()
    if _reuse:
        out = _att_buf("out", (seq, heads, hd), np.float32)
        _otn = torch.from_numpy(out)
    elif _obuf:
        _e = _ATT_OBUF.get((seq, heads, hd))
        if _e is None:
            _e = [np.empty((seq, heads, hd), np.float32), None]
            _ATT_OBUF[(seq, heads, hd)] = _e
        out = _e[0]
        _otn = _e[1]
        if _otn is None:
            _otn = torch.from_numpy(out)      # same buffer, no copy
            _e[1] = _otn
    else:
        out = np.empty((seq, heads, hd), np.float32)
        _otn = None
    _tm = _ts = _tt = 0.0
    for q0 in range(0, seq, qblock):
        q1 = min(seq, q0 + qblock)
        _a0 = _at.perf_counter()
        if _reuse:
            _sb = _att_buf("s", (heads, q1 - q0, seq), qt.dtype)
            torch.bmm(qt[:, q0:q1], kt.transpose(1, 2), out=_sb)
            s = _sb
        else:
            s = torch.matmul(qt[:, q0:q1], kt.transpose(1, 2))
        _a1 = _at.perf_counter()
        if _use_e and _bc is not None:
            # zero-copy: view the bf16 score matrix as uint16, fused softmax
            # writing f32 straight into a reused buffer (no .float(), no 130 MB copy)
            _u = s.view(torch.uint16).numpy().reshape(int(s.shape[0] * s.shape[1]), int(s.shape[2]))
            if _e_out is None or _e_out.shape != _u.shape:
                _e_out = np.empty(_u.shape, np.float32)
            _bc["sm_bf16_f32"](_u, _e_out, 1.0)
            p = torch.from_numpy(_e_out).reshape(s.shape)
            if _reuse:
                _ob = _att_buf("o", (heads, q1 - q0, hd), vt.dtype)
                torch.bmm(p, vt, out=_ob)
                o = _ob
            else:
                o = torch.matmul(p, vt)
            if _otn is not None:
                _otn[q0:q1] = o.transpose(0, 1)
            else:
                out[q0:q1] = o.numpy().transpose(1, 0, 2)
            _a3 = _at.perf_counter()
            _a2 = _a3 - _a1
            _ts += _a2
            continue
        if _use_d and _bc is not None:
            # variant D.  s is bf16 and contiguous, so view(uint16) is a
            # zero-copy reinterpretation; the fused kernel reads bf16 bits,
            # accumulates in f32 registers and writes bf16 bits, so the
            # probabilities cost one 32 MB buffer instead of a 65 MB f32 one and
            # the pv matmul stays bf16.
            _u = s.view(torch.uint16).numpy().reshape(
                int(s.shape[0] * s.shape[1]), int(s.shape[2]))
            _po = _att_buf("p16", _u.shape, np.uint16)
            _bc["sm_bf16_bf16"](_u, _po, 1.0)
            p = torch.from_numpy(_po).view(torch.bfloat16).reshape(s.shape)
            if _reuse:
                _ob = _att_buf("o", (heads, q1 - q0, hd), vt.dtype)
                torch.bmm(p, vt, out=_ob)
                o = _ob
            else:
                o = torch.matmul(p, vt)
            if _otn is not None:
                _otn[q0:q1] = o.transpose(0, 1)
            else:
                out[q0:q1] = o.float().numpy().transpose(1, 0, 2)
            _ts += _at.perf_counter() - _a1
            continue
        if _use_a2 and _bc is not None:
            # the kernel is 2-D (rows, S); s is contiguous so this is a view and
            # the softmax really is in place (s is dead after this point)
            _bc["sm_f32_inplace"](s.numpy().reshape(-1, int(s.shape[2])))
            p = s
            _a2 = _at.perf_counter()
            if _reuse:
                _ob = _att_buf("o", (heads, q1 - q0, hd), vt.dtype)
                torch.bmm(p, vt, out=_ob)
                o = _ob
            else:
                o = torch.matmul(p, vt)
            if _otn is not None:
                _otn[q0:q1] = o.transpose(0, 1)
            else:
                out[q0:q1] = o.numpy().transpose(1, 0, 2)
            _a3 = _at.perf_counter()
            _tt += _a3 - _a2
            continue
        _k = _nb_softmax() if _use_nb else None
        if _k is not None:
            _sh = s.shape
            _pb = _NB_SM_BUF.get(_sh)
            if _pb is None:
                _pb = np.empty(_sh, np.float32)
                _NB_SM_BUF[_sh] = _pb
            _k(s.numpy(), _pb)
            p = torch.from_numpy(_pb)
        else:
            p = torch.softmax(s, dim=-1)
        _a2 = _at.perf_counter()
        if _reuse:
            _ob = _att_buf("o", (heads, q1 - q0, hd), vt.dtype)
            torch.bmm(p, vt, out=_ob)
            o = _ob
        else:
            o = torch.matmul(p, vt)
            if dt is torch.bfloat16:
                o = o.float()
        if _otn is not None:
            _otn[q0:q1] = o.transpose(0, 1)   # any dtype conversion is in here
        else:
            out[q0:q1] = o.numpy().transpose(1, 0, 2)
        _a3 = _at.perf_counter()
        _tm += _a1 - _a0; _ts += _a2 - _a1; _tt += _a3 - _a2
    if _nb_prev is not None:
        try:
            from numba import set_num_threads as _snt2
            _snt2(_nb_prev)
        except Exception:
            pass
    if _prof:
        _tot = _at.perf_counter() - _t0
        print("[attnprof] dt=%s convert %.2f  qk %.2f  softmax %.2f  pv+store %.2f  total %.2f ms"
              % (dt, 1e3*(_t1-_t0), 1e3*_tm, 1e3*_ts, 1e3*_tt, 1e3*_tot), flush=True)
    ret = out.reshape(seq, -1)
    # T22_ATTN_DUMP=<prefix> [T22_ATTN_DUMP_N=<count>]: save the real q/k/v an
    # in-product call actually saw, plus its own output, so a host-only bench
    # can replay every variant on product tensors.  Default off; it changes no
    # arithmetic (it only reads) and is off for every shipped path.
    _dump = os.environ.get("T22_ATTN_DUMP", "")
    if _dump and _ATT_DUMP["n"] < int(os.environ.get("T22_ATTN_DUMP_N", "1")):
        try:
            import pathlib as _pl
            _ATT_DUMP["n"] += 1
            _p = _pl.Path(_dump)
            _p.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(_p) + "_qh.npy", np.array(qh, copy=True))
            np.save(str(_p) + "_kh.npy", np.array(kh, copy=True))
            np.save(str(_p) + "_vh.npy", np.array(v, copy=True))
            np.save(str(_p) + "_o.npy", np.array(ret, copy=True))
            print("[attn-dump] wrote %s_{qh,kh,vh,o}.npy" % (_dump,), flush=True)
        except Exception as _e_d:
            print("[attn-dump] skipped: %r" % (_e_d,), flush=True)
    return ret


def attention_host(self, x1, key, cos, sin, tim):
    """Drop-in for Loop.attention_fast with the ip dispatch replaced by host f32.

    Identical projection code to t18_run.Loop.attention_fast's non-QKV branch:
    four 96-row projection blocks per matrix (q, k, v, o), the same x_image64
    pack, the same C permutation and the same rms+RoPE kernels.  Only the
    QP/KB/VB -> O ip dispatch is replaced by _attention().
    """
    T = _t18()
    bo = self.dev.bo
    sc = T.SCRATCH
    if sc is None:
        raise RuntimeError("chunk_path needs the fast host workspace "
                           "(bridge --fast-host / t18 FAST_HOST)")
    P = self.params(key)
    W = self.w_images(key)
    seq = T.SEQ
    n_proj = (seq + T.PROJ_M - 1) // T.PROJ_M
    Qm = {nm: np.empty((T.HEADS, seq, T.HD), np.float32)
          for nm in ("q", "k", "v")}
    for nm in ("q", "k", "v"):
        for bi in range(n_proj):
            r0 = bi * T.PROJ_M
            rows = min(T.PROJ_M, seq - r0)
            xb = T._xblock(self, x1, r0, rows)
            t0 = time.perf_counter()
            Xb = T.H.x_image64_bytes(xb, self.perm64, sc, "ximg")
            tim["pack"] += 1e3 * (time.perf_counter() - t0)
            raw = self.att_dispatch([(bo["a_x"], Xb), (bo["a_w"], W[nm])],
                                    "att/" + nm, tim)
            t0 = time.perf_counter()
            Qm[nm][:, r0:r0 + rows] = T.H.unpack_bytes(
                raw[0], self.inv, (T.HEADS, T.PROJ_M * T.HD), sc, "u" + nm
            ).reshape(T.HEADS, T.PROJ_M, T.HD)[:, :rows]
            tim["pack"] += 1e3 * (time.perf_counter() - t0)
    t0 = time.perf_counter()
    wq = T.bf16f32(P["norm_q"] * self.fold)
    kh = T.rope_rms_jit(Qm["k"], P["norm_k"], cos, sin,
                        sc.get("kh", (T.HEADS, seq, T.HD), np.float32),
                        sc.get("rms0", (T.HEADS, seq, T.HD), np.float32))
    qh = T.rope_rms_jit(Qm["q"], wq, cos, sin,
                        sc.get("qh", (T.HEADS, seq, T.HD), np.float32),
                        sc.get("rms1", (T.HEADS, seq, T.HD), np.float32))
    tim["pack"] += 1e3 * (time.perf_counter() - t0)
    t0 = time.perf_counter()
    o = _attention(qh, kh, Qm["v"])
    tim["host"] += 1e3 * (time.perf_counter() - t0)
    a_flat = np.empty((seq, T.D), np.float32)
    for bi in range(n_proj):
        r0 = bi * T.PROJ_M
        rows = min(T.PROJ_M, seq - r0)
        xb = T._xblock(self, o, r0, rows)
        t0 = time.perf_counter()
        X2 = T.H.x_image64_bytes(xb, self.perm64, sc, "ximg2")
        tim["pack"] += 1e3 * (time.perf_counter() - t0)
        raw_q, _ = self.att_dispatch([(bo["a_x"], X2), (bo["a_w"], W["o"])],
                                     "att/o", tim)
        t0 = time.perf_counter()
        a_flat[r0:r0 + rows] = T.H.unpack_bytes_t(
            raw_q, self.inv, (T.HEADS, T.PROJ_M * T.HD), T.HD, sc, "at")[:rows]
        tim["pack"] += 1e3 * (time.perf_counter() - t0)
    return a_flat


# ---------------------------------------------------------------------------
# T22 ATTHEV: submit / join, the two halves of Dev.dispatch
# ---------------------------------------------------------------------------
def _launch(dev, name, fills, args, outs):
    """Fill the BOs and hand the kernel to the ERT; do NOT wait for it.

    Same order as Dev.dispatch (fills -> sync -> kernel call), same BO list,
    same instruction count.  The only difference is that the wait is deferred,
    which is what lets the host work under an in-flight launch.
    """
    pyxrt = dev.pyxrt
    t0 = time.perf_counter()
    for bo, payload in fills:
        bo.write(payload, 0)
        bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    tw = 1e3 * (time.perf_counter() - t0)
    im = dev.imgs[name]
    t1 = time.perf_counter()
    run = im["k"](3, im["bo_i"], len(im["inst"]) // 4, *args)
    return {"name": name, "run": run, "args": args, "outs": outs,
            "tw": tw, "t1": t1}


def _join(dev, job, tim, tag):
    """Wait for a submitted job, book it, and read its outputs back."""
    T = _t18()
    st = dev.stats
    name = job["name"]
    run = job["run"]
    state = -1
    blocked = 0.0
    for i in range(len(T.RETRY_SLEEPS) + 1):
        _tw = time.perf_counter()
        try:
            state = int(run.wait(T.TIMEOUT_MS))
        except Exception:                                     # noqa: BLE001
            state = -1
        blocked += time.perf_counter() - _tw
        now = time.perf_counter()
        wall = 1e3 * (now - job["t1"])
        busy = 1e3 * max(0.0, now - max(job["t1"], _LAST_DONE["t"]))
        st["attempts"].append({"img": name, "state": state, "wall_ms": wall})
        if state == T.COMPLETED:
            _LAST_DONE["t"] = now
            dbt = st.setdefault("dev_by_tag", {})
            dbt[tag] = dbt.get(tag, 0.0) + busy
            st["dev_ms"][name] = st["dev_ms"].get(name, 0.0) + busy
            st["write_ms"][name] = st["write_ms"].get(name, 0.0) + job["tw"]
            st["calls"] += 1
            tim["dev"] += busy
            tim["write"] += job["tw"]
            im = dev.imgs[name]
            raws = []
            for bo, n in job["outs"]:
                bo.sync(dev.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
                raws.append(bytes(bo.read(n, 0)))
            return raws, blocked
        st["retries"] += 1
        if i < len(T.RETRY_SLEEPS):
            print("[t3d]   %s RETRY %d/%d after %.1f ms (state %s)"
                  % (tag, i + 1, len(T.RETRY_SLEEPS), wall, state), flush=True)
            time.sleep(T.RETRY_SLEEPS[i])
            im = dev.imgs[name]
            run = im["k"](3, im["bo_i"], len(im["inst"]) // 4, *job["args"])
            job["run"] = run
            job["t1"] = time.perf_counter()
    raise SystemExit("[t3d] REFUSAL: %s never completed (state %s); checkpoint "
                     "holds every dispatch already measured" % (tag, state))


def _ensure(L):
    """The extra BOs and workspaces the pipeline needs, built once."""
    st = getattr(L, "_adpipe", None)
    if st is not None:
        return st
    T = _t18()
    dev = L.dev
    if not dev.has_ip:
        raise RuntimeError("the T22 ATTHEV pipeline needs the split "
                           "projection+ip pair (--att-ip-image)")
    pyxrt = dev.pyxrt
    fl = pyxrt.bo.flags.host_only
    bo = dev.bo
    g = lambda im, i: dev.imgs[im]["k"].group_id(i)           # noqa: E731
    st = {}
    # double buffers: the o projection (X in, O out) and the FFN (A in, Y out)
    st["ox"] = [bo["a_x"], pyxrt.bo(dev.dev, T.PROJ_X_BYTES, fl, g("att", 3))]
    st["oq"] = [bo["a_outq"], pyxrt.bo(dev.dev, T.PROJ_OUT_BYTES, fl,
                                       g("att", 5))]
    st["fa"] = [bo["f_A"], pyxrt.bo(dev.dev, T.FFN_A_BYTES, fl, g("ffn", 3))]
    st["fy"] = [bo["f_Y"], pyxrt.bo(dev.dev, T.FFN_A_BYTES, fl, g("ffn", 8))]
    # the projection phase: two full sets of (W, X, OUT) so that block bi+1's
    # 16.6 MB weight fill happens while block bi's three launches are in flight
    st["pw"] = [pyxrt.bo(dev.dev, T.W_BYTES, fl, g("att", 4))
                for _ in range(6)]
    st["px"] = [bo["a_x"], pyxrt.bo(dev.dev, T.PROJ_X_BYTES, fl, g("att", 3))]
    st["pq"] = [pyxrt.bo(dev.dev, T.PROJ_OUT_BYTES, fl, g("att", 5))
                for _ in range(6)]
    st["out"] = [np.empty((T.SEQ, T.D), np.float32) for _ in range(2)]
    st["out_i"] = 0
    fm, pm = T.FFN_M, T.PROJ_M
    st["xf"] = np.zeros((fm, T.D), np.float32)
    st["xo"] = np.zeros((pm, T.D), np.float32)
    st["a_tmp"] = np.empty((fm, T.D), np.float32)
    st["o_rows"] = np.empty((pm, T.D), np.float32)
    st["xm"] = [np.empty((fm, T.D), np.float32) for _ in range(2)]
    st["n2"] = np.empty((fm, T.D), np.float32)
    st["n3"] = np.empty((fm, T.D), np.float32)
    st["a_flat"] = np.empty((T.SEQ, T.D), np.float32)
    st["y"] = np.empty((T.SEQ, T.D), np.float32)
    st["rms"] = np.empty((T.HEADS, pm, T.HD), np.float32)
    st["rope"] = np.empty((T.HEADS, pm, T.HD), np.float32)
    st["x_written"] = False
    L._adpipe = st
    return st


def _absmax(T, x):
    try:
        return T.absmax_fast(x)
    except Exception:                                         # noqa: BLE001
        return float(np.abs(x).max())


# ---------------------------------------------------------------------------
# T22 ATTHEV: the whole block step, row-block pipelined
# ---------------------------------------------------------------------------
def block_step_pipelined(L, u, key, mod, cos, sin, tim, bg, b3, host_ref=False,
                         sbg_bytes=None):
    """t18_run.block_step with the host attention overlapped with the device.

    Same arithmetic, same BO bytes, same drain order -- only WHEN each piece
    runs changes.  See the module docstring for the measured basis.
    """
    T = _t18()
    H = T.H
    T3 = L.T3
    dev = L.dev
    bo = dev.bo
    sc = T.SCRATCH
    P = L.params(key)
    W = L.w_images(key)
    seq, D, HEADS, HD = T.SEQ, T.D, T.HEADS, T.HD
    PM, FM = T.PROJ_M, T.FFN_M
    nb = (seq + PM - 1) // PM
    nf = T.FFN_BLOCKS
    assert nb == nf and nf > 1, (nb, nf)
    st = _ensure(L)
    torch = _torch()

    # ---- 1. attention_norm1 (+ adaln scale) ------------------------------
    t0 = time.perf_counter()
    x1 = T.rms_into(u, P["attention_norm1"], L.n1buf)
    if mod is not None:
        np.multiply(x1, mod[0], out=x1)
    tim["pack"] += 1e3 * (time.perf_counter() - t0)
    T.gmark("blk.norm1", t0)

    # ---- 2. q / k / v projections, rms+RoPE folded per row block ---------
    _tp = time.perf_counter()
    Qm = {nm: np.empty((HEADS, seq, HD), np.float32)
          for nm in ("q", "k", "v")}
    wq = T.bf16f32(P["norm_q"] * L.fold)
    kh = np.empty((HEADS, seq, HD), np.float32)
    qh = np.empty((HEADS, seq, HD), np.float32)
    rms_t, rope_t = st["rms"], st["rope"]
    jobs = [[], []]

    def proj_done(bi):
        """Join block bi's three launches and fold its rms+RoPE in place."""
        s = bi & 1
        r0 = bi * PM
        rows = min(PM, seq - r0)
        sl = slice(r0, r0 + rows)
        t0 = time.perf_counter()
        blk = 0.0
        for i, nm in enumerate(("q", "k", "v")):
            raw, b = _join(dev, jobs[s][i], tim, "att/" + nm)
            blk += b
            Qm[nm][:, sl] = H.unpack_bytes(
                raw[0], L.inv, (HEADS, PM * HD), sc, "u" + nm
            ).reshape(HEADS, PM, HD)[:, :rows]
        jobs[s] = []
        T.gadd("dev.wait", blk)
        # rms + RoPE for this row block: row-local, so the per-row arithmetic
        # is the same expression on the same values as the full-array call.
        T.rms_into(Qm["k"][:, sl], P["norm_k"], rms_t[:, :rows])
        H.rope_f32(rms_t[:, :rows], cos[sl], sin[sl], rope_t[:, :rows])
        kh[:, sl] = rope_t[:, :rows]
        T.rms_into(Qm["q"][:, sl], wq, rms_t[:, :rows])
        H.rope_f32(rms_t[:, :rows], cos[sl], sin[sl], rope_t[:, :rows])
        qh[:, sl] = rope_t[:, :rows]
        dt = time.perf_counter() - t0
        tim["pack"] += 1e3 * dt
        T.gadd("pipe.proj.drain", dt - blk)

    for bi in range(nb):
        if bi >= 2:
            proj_done(bi - 2)
        s = bi & 1
        r0 = bi * PM
        rows = min(PM, seq - r0)
        t0 = time.perf_counter()
        xb = T._xblock(L, x1, r0, rows)
        Xb = H.x_image64_bytes(xb, L.perm64, sc, "ximg")
        tim["pack"] += 1e3 * (time.perf_counter() - t0)
        T.gadd("pipe.proj.fill", time.perf_counter() - t0)
        for i, nm in enumerate(("q", "k", "v")):
            jobs[s].append(_launch(
                dev, "att", [(st["pw"][s * 3 + i], W[nm]), (st["px"][s], Xb)],
                [st["px"][s], st["pw"][s * 3 + i], st["pq"][s * 3 + i]],
                [(st["pq"][s * 3 + i], T.PROJ_OUT_BYTES)]))
    for bi in (nb - 2, nb - 1):
        proj_done(bi)

    T.gmark("blk.proj", _tp)
    # ---- 3. the pipeline: attention(j) | o-proj(j) | ffn(j) --------------
    kt = torch.from_numpy(kh)
    vt = torch.from_numpy(Qm["v"])
    qt = torch.from_numpy(qh)
    a_tmp, xo, xf = st["a_tmp"], st["xo"], st["xf"]
    a_flat, y_buf = st["a_flat"], st["y"]
    # the output escapes as the next layer's input, so it cannot be the buffer
    # being read; two of them alternate and the 25 MB per-layer allocation (and
    # its page faults) disappears
    out = st["out"][st["out_i"] & 1]
    st["out_i"] += 1
    sbg_m = (sbg_bytes[:T.FFN_SBG_BYTES // 2]
             if getattr(sbg_bytes, "dtype", None) == np.uint16
             else sbg_bytes[:T.FFN_SBG_BYTES])
    tim.setdefault("attn", 0.0)

    def att_block(j):
        r0 = j * FM
        rows = min(FM, seq - r0)
        t0 = time.perf_counter()
        s = torch.matmul(qt[:, r0:r0 + rows], kt.transpose(1, 2))
        p = torch.softmax(s, dim=-1)
        a_tmp[:rows] = torch.matmul(p, vt).numpy().transpose(1, 0, 2).reshape(
            rows, D)
        dt = time.perf_counter() - t0
        tim["attn"] += 1e3 * dt
        T.gadd("pipe.attn", dt)
        return r0, rows

    def submit_o(j, r0, rows):
        b = j & 1
        t0 = time.perf_counter()
        if rows < PM:
            xo[rows:].fill(0.0)
        np.copyto(xo[:rows], a_tmp[:rows])
        X2 = H.x_image64_bytes(xo, L.perm64, sc, "ximg2")
        tim["pack"] += 1e3 * (time.perf_counter() - t0)
        T.gadd("pipe.o.fill", time.perf_counter() - t0)
        return _launch(dev, "att", [(st["ox"][b], X2)],
                       [st["ox"][b], bo["a_w"], st["oq"][b]],
                       [(st["oq"][b], T.PROJ_OUT_BYTES)])

    def join_o(job, j, r0, rows):
        t0 = time.perf_counter()
        raw, blk = _join(dev, job, tim, "att/o")
        T.gadd("dev.wait", blk)
        T.unpack_bytes_t_into(raw[0], L.inv64, HD, st["o_rows"])
        a_flat[r0:r0 + rows] = st["o_rows"][:rows]
        dt = time.perf_counter() - t0
        tim["pack"] += 1e3 * dt
        T.gadd("pipe.o.unpack", dt - blk)

    def submit_f(j, r0, rows):
        b = j & 1
        xm = st["xm"][b]
        t0 = time.perf_counter()
        # attention_norm2 + residual  (xm = u + mod[1] * rms(a))
        a2 = T.rms_into(a_flat[r0:r0 + rows], P["attention_norm2"],
                        st["n2"][:rows])
        if mod is None:
            np.add(u[r0:r0 + rows], a2, out=xm[:rows])
        else:
            np.multiply(a2, mod[1], out=a2)
            np.add(u[r0:r0 + rows], a2, out=xm[:rows])
        if rows < FM:
            xf[rows:].fill(0.0)
        np.copyto(xf[:rows], xm[:rows])
        a_img = H.bf16_bytes_f32(xf, sc, "ffnA")
        fills = [(st["fa"][b], a_img)]
        if j == 0 or _DIAG["ffn_refill"]:
            # SBG and the 132.7 MB BG/B3 payload are IDENTICAL on all 18 blocks
            # of a layer; the delivered loop rewrites them every block (that is
            # the whole of write_ms ffn = 23.3 s).  Same bytes in the BO, so
            # the kernel reads the same values.
            fills.append((bo["f_SBG"], sbg_m))
            if bg is not None:
                # NH_BG > 1 => the kernel reads BG pre-permuted, and this path
                # never calls Loop.ffn, so nothing permutes it here.  See
                # bg_order_guard() for why this is the only un-gated seam.
                fills += _bg_fills(T, bo, bg, b3)
        if not st["x_written"]:
            fills.append((bo["f_X"], L.zero_x))
            st["x_written"] = True
        tim["pack"] += 1e3 * (time.perf_counter() - t0)
        T.gadd("pipe.f.fill", time.perf_counter() - t0)
        return _launch(dev, "ffn", fills,
                       [st["fa"][b], bo["f_SBG"], bo["f_BG"], bo["f_B3"],
                        bo["f_X"], st["fy"][b]],
                       [(st["fy"][b], T.FFN_A_BYTES)])

    def join_f(job, j, r0, rows):
        b = j & 1
        t0 = time.perf_counter()
        raw, blk = _join(dev, job, tim, "ffn[%d]" % j)
        T.gadd("dev.wait", blk)
        yb = H.bf16_to_f32_bits(np.frombuffer(raw[0], np.uint16), (FM, D),
                                sc, "ffnY")
        y_buf[r0:r0 + rows] = yb[:rows]
        # ffn_norm2 + residual
        p = T.rms_into(y_buf[r0:r0 + rows], P["ffn_norm2"], st["n3"][:rows])
        if mod is None:
            np.add(st["xm"][b][:rows], p, out=out[r0:r0 + rows])
        else:
            np.multiply(p, mod[3], out=p)
            np.add(st["xm"][b][:rows], p, out=out[r0:r0 + rows])
        dt = time.perf_counter() - t0
        tim["pack"] += 1e3 * dt
        T.gadd("pipe.f.tail", dt - blk)

    # the o projection's weight image is the same object for all 18 blocks
    # (the delivered loop re-fills 16.6 MB per block); write it once per layer,
    # before the first launch that reads it.
    t0 = time.perf_counter()
    bo["a_w"].write(W["o"], 0)
    bo["a_w"].sync(dev.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    tim["write"] += 1e3 * (time.perf_counter() - t0)

    r0, rows = att_block(0)
    job_o = submit_o(0, r0, rows)
    job_f = None
    for j in range(nf):
        r0j = j * FM
        rowsj = min(FM, seq - r0j)
        join_o(job_o, j, r0j, rowsj)
        # The delivered state-8 law (Dev.barrier): the FFN image never survives
        # a same-image relaunch, and one launch of a DIFFERENT xclbin in between
        # removes it.  In this pipeline the o projection of block j+1 is that
        # different launch for blocks 0..16 -- but the LAST block of a layer has
        # no o projection after it, so ffn[17] would follow ffn[16] directly
        # (measured: exactly one 7 s state-8 stall per layer, 30/30 layers).
        # One ip launch per layer restores the delivered spacing.
        if j + 1 == nf or _DIAG["ffn_barrier"]:
            if j:
                dev.barrier("attip" if dev.has_ip else "att")
        job_f = submit_f(j, r0j, rowsj)
        if j + 1 < nf:
            r0n, rowsn = att_block(j + 1)
            job_o = submit_o(j + 1, r0n, rowsn)
        # ffn[j] is joined IN THIS iteration, before ffn[j+1] can be submitted:
        # the delivered state-8 law (Dev.barrier) says the FFN image never
        # survives a same-image relaunch, and a deferred join lets ffn[j] still
        # be in flight when ffn[j+1] is submitted -- measured: 3 stalls over one
        # 8-step run at random blocks.  Joining here keeps the device saturated
        # (o[j+1] is already queued behind ffn[j]) and makes the relaunch clean.
        join_f(job_f, j, r0j, rowsj)

    # ---- 4. the per-layer record (same numbers the delivered path logs) --
    _tr = time.perf_counter()
    xm_max = 0.0
    for j in range(nf):
        xm_max = max(xm_max, float(np.abs(st["xm"][j & 1]).max()))
    rec = {"layer": key, "a_absmax": _absmax(T, a_flat),
           "xm_absmax": xm_max, "y_absmax": _absmax(T, y_buf)}
    T.gmark("blk.rec", _tr)
    if host_ref:
        # the numerical gate path: the delivered step (it computes the host
        # reference attention/FFN on step 0's first layer only)
        return _ORIG_BLOCK_STEP(L, u, key, mod, cos, sin, tim, bg, b3,
                                host_ref=host_ref, sbg_bytes=sbg_bytes)
    return out, rec


def _fg_line(st):
    """The ONE line that names which of the THREE host-glue states this run is.

    off        the switch was never on: 0 hooks is CORRECT, not a degradation
    installed  every hook that was wanted is in place
    incomplete the switch was ON and hooks failed to install: the loud one
    """
    if st.get("state") == "off":
        return ("[t18] fastglue hooks=0/8 OFF (%s -- no hooks were wanted, so "
                "none are missing; the delivered path is the one asked for)"
                % (st.get("why") or "T22_FASTGLUE is off"))
    if st.get("complete"):
        return "[t18] fastglue hooks=%d/%d installed" % (st["on"], st["total"])
    return ("[t18] fastglue hooks=%d/%d INCOMPLETE: %s"
            % (st["on"], st["total"], st.get("why") or "?"))


def install(T=None):
    """Install the chunk_path on the t18 Run loop (called once per chain).

    T22_CHAIN2B: T is the DRIVER MODULE this install belongs to.  A two-chain
    process has one t18_run module instance per chain, and everything below --
    Loop.attention, block_step, the image globals -- must land on THAT one.
    Default None keeps the single-chain path byte-identical (the process's own
    t18_run module).
    """
    global _ORIG_BLOCK_STEP
    T = T if T is not None else _t18()
    _torch()                       # fail at startup, not at the first layer
    T.Loop.attention = attention_host
    # t22_variants: the attention-variant verdict is a RECORDED FACT from here
    # on.  _attn_gate() prints the ONE line webui/runner.py parses into
    # job.json:attn_variants and t22_repro Q19 gates on, and REFUSES a
    # requested variant that cannot engage unless the operator opened the
    # hatch (and then the record says "operator chose fail-open").
    _att = _attn_gate()
    info = {"mode": "chunked", "attention": "host-f32-torch",
            "projection": "device M=96 row blocks",
            "ffn": "device M=96 row blocks",
            "ip": "not used (SEQ > delivered L1 ceiling)"}
    info["attn_variants"] = {"requested": list(_att.get("requested") or []),
                             "variants": _att.get("variants") or {},
                             "policy": _att.get("policy"),
                             "bcore": _att.get("bcore"),
                             "line": _att.get("line")}
    if os.environ.get("ZIMAGE_CHUNK_PIPE", "1") != "0":
        if os.environ.get("T22_PLAN_FFN", "0") == "1":
            # t22_adaptive: this pipeline runs the FFN as 96-row blocks of the
            # DELIVERED ffn image inside the loop, so it never reaches
            # t18_run.block_step's device-FFN branch -- which is the only place
            # the plan's images can be dispatched.  Refuse rather than silently
            # run the delivered FFN under a plan flag.
            raise SystemExit(
                "[planffn] REFUSAL: T22_PLAN_FFN=1 replaces the WHOLE-LAYER FFN "
                "(t18_run.block_step), but ZIMAGE_CHUNK_PIPE=1 installs the "
                "96-row pipeline, which dispatches the delivered ffn image "
                "itself.  Run with ZIMAGE_CHUNK_PIPE=0 ZIMAGE_BIGM=1 "
                "T22_PLAN_FFN=1 (add T22_DEVKILL=1 for the delivered "
                "whole-layer comparison).")
        if _ORIG_BLOCK_STEP is None:
            _ORIG_BLOCK_STEP = T.block_step
        T.block_step = block_step_pipelined
        info["schedule"] = ("96-row pipeline: host attention under the device "
                            "o-projection/FFN (double-buffered BOs)")
    else:
        info["schedule"] = "delivered (no overlap)"
    if (os.environ.get("ZIMAGE_BIGM", "0") == "1"
            and os.environ.get("ZIMAGE_BIGM_PROJ", "0") != "0"):
        # t22_bigpipe2: big-M replaces the four chunked projection passes with
        # whole-shape bigm_proj dispatches (one per q/k/v/o per 17-tile chunk).
        # ZIMAGE_CHUNK_PIPE=0 keeps the delivered block_step so the FFN BIGM
        # branch in t18_run is the one that runs.
        import sys as _sys
        from pathlib import Path as _P
        _bgm = _P(__file__).resolve().parents[1] / "imaggers" / "block" / "bigm"
        if str(_bgm) not in _sys.path:
            _sys.path.insert(0, str(_bgm))
        import prod_loop_bigm as _PB
        # t22_loud: T22_DEVKILL=1 consumes the n32 B payload, and
        # cached_payloads() builds it ONLY when T22_BIGM_DEVHANDOFF=1.  With the
        # handoff off it hands out {n128,n64} and devkill_ffn.block raises
        # "devkill needs n32 B payload" at the first FFN of step 0 -- i.e. ~60 s
        # in, after startup, as a traceback from inside the loop.  Both sides of
        # that contract are chosen here, so this is where it is checked.
        if (os.environ.get("T22_DEVKILL", "0") == "1"
                and os.environ.get("T22_BIGM_DEVHANDOFF", "0") != "1"
                and os.environ.get("T22_BIGM_B3_DUAL", "1") != "0"):
            raise SystemExit(
                "[bigm] REFUSAL: T22_DEVKILL=1 needs the n32 B payload, and "
                "prod_loop_bigm.cached_payloads() only builds it with "
                "T22_BIGM_DEVHANDOFF=1 (with T22_BIGM_B3_DUAL=1 it hands out "
                "{n128,n64}, so devkill_ffn.block raises 'devkill needs n32 B "
                "payload' at the first FFN of step 0).  Set "
                "T22_BIGM_DEVHANDOFF=1, or T22_BIGM_B3_DUAL=0 to convert per "
                "dispatch, or T22_DEVKILL=0.")
        if os.environ.get("T22_HOSTKILL", "0") == "1":
            # t22_hostkill: DMA-side A pack / C un-tile projection image.
            # Flag-gated; the default branch below is byte-for-byte unchanged.
            _hk = _P(__file__).resolve().parents[1] / "imaggers" / "hostkill"
            if str(_hk) not in _sys.path:
                _sys.path.insert(0, str(_hk))
            import prod_hostkill as _PH
            if os.environ.get("T22_DEVKILL", "0") == "1":
                # t22_devkill: G=15 plain projection (same plain-X/row-major-C contract).
                _dk = _P(__file__).resolve().parents[1] / "imaggers" / "devkill"
                if str(_dk) not in _sys.path:
                    _sys.path.insert(0, str(_dk))
                import prod_devkill as _PD
                # t22_util S1: byte-exact host-glue fast path.  install() is a
                # no-op unless T22_FASTGLUE=1, runs its own 6 byte gates first
                # and fails open.  Default 0 => everything below is untouched.
                _fg_state = {"on": 0, "total": 8, "complete": False,
                             "state": "unknown", "wanted": None,
                             "why": "install() did not run"}
                try:
                    _gl = _P(__file__).resolve().parents[1] / "imaggers" / "util" / "glue"
                    if str(_gl) not in _sys.path:
                        _sys.path.insert(0, str(_gl))
                    import install as _GLUE
                    _st = _GLUE.install(T=T) or {}
                    _fg_state = {"on": int(_st.get("n_on") or 0),
                                 "total": int(_st.get("n_total") or 8),
                                 "complete": bool(_st.get("complete")),
                                 "state": _st.get("state") or "unknown",
                                 "wanted": _st.get("wanted"),
                                 "why": _st.get("why") or ""}
                    info["fastglue"] = dict(_fg_state, hooks=_st.get("hooks") or {})
                    print(_fg_line(_fg_state), flush=True)
                except Exception as _e_fg:
                    # t22_loud: the switch is the user's statement of intent.  A
                    # failure to honour it is a failure, not a note -- but the
                    # delivered bytes are still correct, so the escape hatch is
                    # explicit and named in the sentence.
                    info["fastglue"] = dict(_fg_state, error=repr(_e_fg))
                    if (os.environ.get("T22_FASTGLUE", "0") == "1"
                            and os.environ.get("T22_GLUE_FAIL_OPEN", "0") != "1"):
                        raise SystemExit(
                            "[fastglue] REFUSAL: T22_FASTGLUE=1 was requested and "
                            "the hook install raised %r.  The run would continue "
                            "on the delivered path: same bytes, ~37 s slower at "
                            "512^2/8 steps (122.8 vs 88.6 s loop, "
                            "HOSTPROF-REPORT.md:264-266), and nothing in job.json "
                            "would say so.  Fix the install, unset T22_FASTGLUE, "
                            "or set T22_GLUE_FAIL_OPEN=1 to accept the slower "
                            "path knowingly." % (_e_fg,))
                    print("[fastglue] skipped: %r" % (_e_fg,), flush=True)
                if os.environ.get("T22_DEVKILL_PREWARM", "1") != "0":
                    # t22_devkill: pack the 120 G=15 W images at startup instead
                    # of on first use inside step 0 (measured +0.17 s/layer).
                    # Fail-open: any failure falls back to the lazy path.
                    try:
                        import devkill_prewarm as _DPW
                        _DPW.prewarm_all()
                    except Exception as _e_pw:
                        print("[devkill] prewarm skipped: %r" % (_e_pw,), flush=True)
                T.Loop.attention = _PD.bigm_attention
                info["projection"] = "big-M plain-X/C G=15 (HALF=2)"
                if os.environ.get("T22_S2_PROJ32", "0") == "1":
                    # t22_util S2: 32-core fused projection (opt-in).  Installed
                    # after fastglue, so it wins (last writer wins).
                    _s2p = _P(__file__).resolve().parents[1] / "imaggers" / "util" / "place"
                    if str(_s2p) not in _sys.path:
                        _sys.path.insert(0, str(_s2p))
                    import s2_product as _S2P
                    # the devkill path calls the module-level proj4 (not bigm_proj4)
                    _PD.proj4 = _S2P.proj4
                    try:
                        _PH.proj4 = _S2P.proj4
                    except (NameError, AttributeError):
                        pass
                    info["projection"] += " -> s2 proj32 (fused768)"
            else:
                T.Loop.attention = _PH.bigm_attention
                info["projection"] = "big-M plain-X/row-major-C (DMA pack/un-tile)"
        else:
            T.Loop.attention = _PB.bigm_attention
        info["attention"] = "big-M whole-shape proj (rt17) + host-f32-torch"
        info["projection"] = "big-M whole-shape (17-tile chunks)"
    # T22_ATTN_{D,A2,E}: pay the numba compile now instead of inside step 0.
    _attn_prewarm()
    return info
