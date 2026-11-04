# SPDX-License-Identifier: MIT
"""install.py -- S1: the whole host-glue design under the product, flag-gated.

WHAT IT DOES: function-level patches only (plus one mechanically generated
copy of t18_run.block_step, see HOOKS below).  No product file is edited.
Nothing happens unless T22_FASTGLUE=1.  Every hook is applied only if its own
byte gate passes AGAINST THE PRODUCT FUNCTION it replaces; a failed gate skips
that hook (fail-open) and leaves the reason in install.status().

    # in the bridge process, after the design modules are imported:
    import sys; sys.path.insert(0, r"imaggers/util/glue")
    import install as GLUE
    GLUE.install()              # no-op unless T22_FASTGLUE=1

KNOBS
    T22_FASTGLUE=1                 master switch
    T22_FASTGLUE_THREADS=8         numba threads for the fused kernels
    T22_FASTGLUE_SKIP=a,b,c        disable individual hooks (bisect list below)
    T22_FASTGLUE_HUNKS=1,2,3,4     subset of the four block_step hunks
    T22_FASTGLUE_STRICT=1          refuse to install if ANY hook gate fails

HOOKS (name -> product symbol -> what changes)
    bf16    prod_loop_bigm._bf16_bytes   f32->bf16 RNE, 5 temporaries -> 1 pass
    proj    prod_devkill.proj4           X prep (zeros+copy+chain+tobytes) -> 1 pass;
                                         drain without the bytes() copy
    pack    devkill_ffn.a_pack_fast      u16 -> packed image in one pass
    decode  devkill_ffn.decode_fast      blocked -> row-major f32 in one pass
    ffnchain devkill_ffn.bigm_ffn        the inline f32->bf16 chain (5 temps)
    rms     t18_run.rms_into             6 numpy passes -> 1 pass
    attn    prod_devkill.bigm_attention  q/k skip the f32 drain entirely;
                                         rope reads the bf16 drain; o->bf16+absmax
    block   t18_run.block_step           norm1/ffn_norm1 emit bf16, norm2 and
                                         ffnpost fused (generated copy, 4 hunks)

BISECT ORDER when the two arms differ (5 minutes, see GLUE-AUDIT.md 8.4):
    T22_FASTGLUE_SKIP=block,attn            -> back to the round-1 tier
    T22_FASTGLUE_SKIP=block                 -> attention only
    ... then add hooks back one at a time in the order block, attn, rms,
    decode, pack, ffnchain, proj, bf16.
"""
from __future__ import annotations

import inspect
import os
import sys
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import fast_glue as G                                        # noqa: E402

# names the generated block_step resolves through _fg_glue (this module)
norm_bf16 = G.norm_bf16
norm_residual = G.norm_residual
post_ffn = G.post_ffn

ALL_HOOKS = ["bf16", "proj", "pack", "decode", "ffnchain", "rms", "attn", "block"]
BISECT_ORDER = ["block", "attn", "rms", "decode", "pack", "ffnchain", "proj", "bf16"]

STATE = {"installed": False, "threads": 0, "why": "", "hooks": {}}

# ---------------------------------------------------------------------------
# T22_CHAIN2B (2026-11-04) -- PER-CHAIN OWNERSHIP OF EVERY MUTABLE HERE.
# DEFAULT OFF: with T22_CHAIN2B unset, chain_now() returns 0 for every thread,
# STATE is the only state dict, and this file behaves exactly as it did.
#
# WHY.  Two chains in one process (two Dev instances sharing one registered
# context set, two host threads) each need their OWN host scratch, because the
# buffers below are REUSED between calls and the two chains' calls interleave:
#
#   * _SCR[chain]    -- _make_attention's qh/kh/vh/o16 (the install.py:442 _scr
#                       dict of STREAM2 sec.2.2 item 4).  Chain B's q rope
#                       writing into chain A's qh is silent corruption, not a
#                       crash: the attention would read another image's q.
#   * _Y_SCR[chain]  -- fast_decode's (n_tiles,rows) buffer, which decode_y
#                       returns a VIEW into (install.py:776/784).  One dict for
#                       two chains means chain B's FFN drifts into chain A's
#                       decode buffer between the decode and the ffnpost read.
#   * _ATT[chain]    -- the attention closure itself (install.py:1051 installs
#                       ONE PD.bigm_attention; two chains need one per chain).
#   * STATE_CHAIN    -- install()'s re-entrancy latch.  It used to be a single
#                       bool, so a second install() for chain B returned "already
#                       installed" and chain B silently ran chain A's hook.
#
# The chain id is the CALLING THREAD's (one chain = one host thread, which is
# the only design the measured substrate allows -- see chain2b/REPORT.md).  It
# is ALSO resolved from the Loop where a Loop is available (PD.bigm_attention is
# called as self.attention(...)), so a per-Loop dispatch is exact rather than
# thread-dependent.
_CHAIN = 0
_CHAIN_TLS = threading.local()
_SCR = {}
_Y_SCR = {}
_ATT = {}
STATE_CHAIN = {}


def chain_now():
    """The chain the CALLING THREAD works for (0 when single-chain)."""
    return int(getattr(_CHAIN_TLS, "chain", _CHAIN))


def _own(chain):
    """Which scratch a chain gets.  T22_CHAIN2B_SHARE=1 is the NEGATIVE CONTROL:
    it hands both chains chain 0's buffers, i.e. exactly the pre-change code, so
    the bit-identity gate has something to bite on.  Never set it for a run
    whose numbers you intend to keep."""
    if os.environ.get("T22_CHAIN2B_SHARE", "0") == "1":
        return 0
    return int(chain)


def set_chain(chain):
    """Bind this THREAD to a chain.  Called by the two-chain driver."""
    _CHAIN_TLS.chain = int(chain)
    return int(chain)

# ---------------------------------------------------------------------------
# T22_OVL (2026-11-02) -- the in-loop HOST<->DEVICE overlap.  DEFAULT OFF.
#
# WHAT IT DOES: the block loop is a strict alternating chain -- every host
# segment's output is the next dispatch's input -- so "overlap the host with the
# device" can only mean "run the host segments that the NEXT dispatch does not
# depend on while that dispatch is in flight".  In the shipped loop that is
# exactly two segments, and both of them are hidden here:
#
#   * rope_fused(q)  -- the k projection does not read qh, so it is issued
#                       BEFORE q's drain is rope'd; same for rope_fused(k)
#                       under the v projection.
#   * the FFN's 88.5 MB gu W payload -- it is a per-layer CONSTANT that the
#                       current layer's dn dispatch does not read, so layer
#                       i+1's payload is written into the gu W BO while layer
#                       i's dn runs (devkill_ffn.block).
#
# Everything else in the loop is ON the chain (attention -> o dispatch,
# a -> norm2 -> ffn_norm1 -> A pack -> gu, gu drain -> dn, dn -> ffnpost) and
# is deliberately NOT touched: see OVL-REPORT.md section 2 for the map and the
# measurement that prices each row.
#
# NO BO IS DOUBLE BUFFERED.  _proj_join() copies the drain to host memory before
# the next _proj_post() rewrites the X/W BOs, and the gu W BO is only written
# after the gu dispatch that reads it has been waited for.
#
#   T22_OVL=1            master switch (default 0 = byte-for-byte the delivered
#                        path: _dispatch_proj is still post+join in one call)
#   T22_OVL_WGU=0        disable only the gu-W hoist (bisect: proj pipe alone)
#   T22_OVL_THREADS=N    host-pool cap for the OVERLAPPED kernels only (default
#                        0 = keep T22_FASTGLUE_THREADS).  CONTENTION-REPORT 2.3
#                        measured delta = +0.153 / eps = +0.165 at 18 threads
#                        against +0.578/+0.603 at 20, i.e. a co-tenant's cost is
#                        set by how long it holds many cores, so the overlapped
#                        kernel is the one place the cap has to be measured.
# ---------------------------------------------------------------------------
_OVL = False
_OVL_WGU = False
_OVL_TH = 0


# ---------------------------------------------------------------------------
# gates: every hook is checked against the PRODUCT function it replaces
# ---------------------------------------------------------------------------
def _refs(PB, DVF, T):
    """Verbatim numpy references, re-derived from the product SOURCE.

    They are used instead of the module symbols when a symbol is already ours
    (second install() / module reload): gating a patch against itself is
    circular, and the parent's device run showed why that matters.
    """
    def bf16(a):
        uu = np.ascontiguousarray(a, np.float32).view(np.uint32)
        return (((uu + np.uint32(0x7FFF) + ((uu >> 16) & np.uint32(1))) >> 16)
                .astype(np.uint16))

    def drain(u16, rows, cols):
        y = np.empty((rows, cols), np.float32)
        np.left_shift(u16, 16, out=y.view(np.uint32), dtype=np.uint32)
        return y

    def pack_img(u16):                      # devkill_ffn.a_pack_fast, verbatim
        x = np.asarray(u16)
        rows, cols = x.shape
        t = x.reshape(rows // 96, 12, 8, cols // 64, 8, 8).transpose(0, 3, 1, 4, 2, 5)
        return np.frombuffer(np.ascontiguousarray(t).tobytes(), np.uint16)

    def decode(raw, rt, rows, cols=3840):   # devkill_ffn.decode_fast, verbatim
        a = np.frombuffer(bytes(raw), np.uint16).reshape(rt, 30, 12, 16, 8, 8)
        b = np.ascontiguousarray(a.transpose(0, 2, 4, 1, 3, 5)).reshape(
            rt * 96, cols)
        out = np.empty((rt * 96, cols), np.float32)
        np.left_shift(b, 16, out=out.view(np.uint32), dtype=np.uint32)
        return out[:rows]

    def rms(x, w, out):                     # t18_run.rms_into, verbatim
        np.multiply(x, x, out=out)
        v = out.mean(-1, keepdims=True, dtype=np.float32)
        np.add(v, np.float32(1e-5), out=v)
        np.sqrt(v, out=v)
        np.divide(x, v, out=out)
        np.multiply(out, np.asarray(w, np.float32), out=out)
        return out
    return bf16, drain, pack_img, decode, rms


def _gate_all(PB, DVF, T, th):
    rng = np.random.default_rng(11)
    rows, D, HD = 192, 3840, 128
    x = (rng.standard_normal((rows, D)).astype(np.float32) * 40.0)
    w = (1.0 + 0.1 * rng.standard_normal(D)).astype(np.float32)
    mod = (1.0 + 0.05 * rng.standard_normal(D)).astype(np.float32)
    bf16, drain, pack_img, decode, rms = _refs(PB, DVF, T)
    out = {}

    def ours(f):
        return bool(getattr(f, "_fg_patched", False))

    # -- bf16: both product spellings --------------------------------------
    u16 = bf16(x)
    got = G.bf16_rne(x, threads=th)
    out["bf16"] = (got.tobytes() == u16.tobytes()
                   and got.tobytes() == bf16(x).tobytes())

    # -- proj: the drain half (the dispatch half is device code) -----------
    out["proj"] = (G.proj_read_cv(u16.tobytes(), rows, D, threads=th).tobytes()
                   == drain(u16, rows, D).tobytes())

    # -- pack: u16 -> image, and f32 -> (RNE + image) ----------------------
    refp = (np.frombuffer(DVF.a_pack_fast(u16), np.uint16)
            if not ours(DVF.a_pack_fast) else pack_img(u16))
    out["pack"] = (G.pack_a(u16, from_bf16=True, threads=th).tobytes()
                   == refp.tobytes()
                   and G.pack_a(x, threads=th).tobytes() == refp.tobytes())

    # -- decode -----------------------------------------------------------
    yd = np.frombuffer(u16.tobytes(), np.uint16)
    refd = (DVF.decode_fast(yd.tobytes(), 2, rows)
            if not ours(DVF.decode_fast) else decode(yd.tobytes(), 2, rows))
    out["decode"] = (G.decode_y(yd, 2, rows, threads=th).tobytes()
                     == refd.tobytes())

    # -- ffnchain: the inline expression devkill_ffn.py:156 spells ---------
    out["ffnchain"] = G.bf16_rne(x, threads=th).tobytes() == u16.tobytes()

    # -- rms: t18_run.rms_into (6 numpy passes) ----------------------------
    tmp = np.empty((rows, D), np.float32)
    ref_rms = (T.rms_into(x, w, tmp) if not ours(T.rms_into)
               else rms(x, w, tmp))
    out["rms"] = (G.norm_f32(x, w, None, threads=th).tobytes()
                  == ref_rms.tobytes())

    # -- attn: rope from the bf16 drain, v widen, o->bf16(+absmax) ---------
    seq, heads, hd = 96, 2, HD
    x2 = (rng.standard_normal((seq, heads * hd)).astype(np.float32) * 3.0)
    u2 = bf16(x2)
    w_hd = (1.0 + 0.1 * rng.standard_normal(hd)).astype(np.float32)
    cos = np.cos(rng.standard_normal((seq, hd // 2))).astype(np.float32)
    sin = np.sin(rng.standard_normal((seq, hd // 2))).astype(np.float32)
    hv = np.ascontiguousarray(
        drain(u2, seq, heads * hd).reshape(seq, heads, hd).transpose(1, 0, 2))
    tmp2 = np.empty((heads, seq, hd), np.float32)
    ref_rope = T.rope_rms_jit(hv, w_hd, cos, sin,
                              np.empty((heads, seq, hd), np.float32), tmp2)
    got_rope = G.rope_fused(u2, w_hd, cos, sin,
                            out=np.empty((heads, seq, hd), np.float32),
                            threads=th)
    got_v = G.widen_t(u2, seq, heads, hd, threads=th)
    got_o, got_am = G.bf16_of(x, want_absmax=True)
    out["attn"] = (got_rope.tobytes() == ref_rope.tobytes()
                   and got_v.tobytes() == hv.tobytes()
                   and got_o.tobytes() == u16.tobytes()
                   and got_am == T.absmax_fast(x))

    # -- block: the four fused spots, against the reference expressions ----
    ok_b1 = (G.norm_bf16(x, w, mod, threads=th).tobytes()
             == bf16(np.multiply(rms(x, w, np.empty((rows, D), np.float32)),
                                 mod, out=np.empty((rows, D), np.float32))
                     .reshape(rows, D)).tobytes()
             or True)
    n1 = np.empty((rows, D), np.float32)
    rms(x, w, n1)
    np.multiply(n1, mod, out=n1)
    ok_b1 = (G.norm_bf16(x, w, mod, threads=th).tobytes()
             == bf16(n1).tobytes())
    xm_like = (rng.standard_normal((rows, D)).astype(np.float32) * 20.0)
    n2 = np.empty((rows, D), np.float32)
    rms(x, w, n2)
    np.multiply(n2, mod, out=n2)
    ref_xm = np.add(xm_like, n2, out=np.empty((rows, D), np.float32))
    ok_b2 = (G.norm_residual(x, xm_like, w, mod, threads=th).tobytes()
             == ref_xm.tobytes())
    n3 = np.empty((rows, D), np.float32)
    rms(x, w, n3)
    np.multiply(n3, mod, out=n3)
    ref_post = np.add(xm_like, n3, out=np.empty((rows, D), np.float32))
    got_post, st = G.post_ffn(x, xm_like, w, mod, threads=th)
    ok_b4 = (got_post.tobytes() == ref_post.tobytes()
             and st[0] == T.absmax_fast(xm_like) and st[1] == T.absmax_fast(x))
    out["block_unit"] = ok_b1 and ok_b2 and ok_b4
    out["block"] = ok_b1 and ok_b2 and ok_b4
    return out


# ---------------------------------------------------------------------------
# the generated block_step: 4 named hunks, mechanical, refuses on drift
# ---------------------------------------------------------------------------
HUNKS = []

HUNKS.append(("h1-norm1-bf16", """    if T22_FAST_GLUE:
        # T22: same expressions, reused buffers, no temporaries (rms_into is
        # bit-equal to T3.rms -- t22_rmsbench.py asserts np.array_equal).
        x1 = rms_into(u, P["attention_norm1"], L.n1buf)
        if mod is not None:
            np.multiply(x1, mod[0], out=x1)
        n1 = x1""", """    if T22_FAST_GLUE and _fg_h1 and mod is not None and not host_ref:
        # S1 hook norm1: rms+scale+bf16 in ONE pass.  x1's only consumer is
        # L.attention -> proj4, which wants the bf16 image (t18_run.py:1889);
        # norm_bf16 is byte-equal to _bf16_bytes(rms_into(x, w) * mod).
        x1 = _fg_glue.norm_bf16(u, P["attention_norm1"], mod[0], threads=_fg_th)
        n1 = x1
    elif T22_FAST_GLUE:
        x1 = rms_into(u, P["attention_norm1"], L.n1buf)
        if mod is not None:
            np.multiply(x1, mod[0], out=x1)
        n1 = x1"""))

HUNKS.append(("h2-norm2-fused", """    if T22_FAST_GLUE:
        a2 = rms_into(a, P["attention_norm2"], L.n2buf)
        if mod is None:
            xm = np.add(u, a2, out=a2)
        else:
            np.multiply(a2, mod[1], out=a2)
            xm = np.add(u, a2, out=a2)""", """    if T22_FAST_GLUE and _fg_h2:
        # S1 hook norm2: rms+mul+residual in ONE pass, same operand order.
        xm = _fg_glue.norm_residual(a, u, P["attention_norm2"],
                                    None if mod is None else mod[1],
                                    out=L.n2buf, threads=_fg_th)
    elif T22_FAST_GLUE:
        a2 = rms_into(a, P["attention_norm2"], L.n2buf)
        if mod is None:
            xm = np.add(u, a2, out=a2)
        else:
            np.multiply(a2, mod[1], out=a2)
            xm = np.add(u, a2, out=a2)"""))

HUNKS.append(("h3-ffn-norm1-bf16", """        _xn = rms_into(xm, P["ffn_norm1"], L.n1buf)
        if mod is not None:
            np.multiply(_xn, mod[2], out=_xn)""", """        if _fg_h3 and mod is not None and not _DUMP:
            # S1 hook: ffn_norm1 emits the bf16 A image directly -- the FFN
            # contract is bf16 anyway (devkill_ffn.py:156 converted it next).
            _xn = _fg_glue.norm_bf16(xm, P["ffn_norm1"], mod[2], threads=_fg_th)
        else:
            _xn = rms_into(xm, P["ffn_norm1"], L.n1buf)
            if mod is not None:
                np.multiply(_xn, mod[2], out=_xn)"""))

HUNKS.append(("h4-ffnpost-fused", """    if T22_FAST_GLUE:
        p = rms_into(y, P["ffn_norm2"], L.n3buf)
        out = np.empty((SEQ, D), np.float32)   # escapes to the next layer
        if mod is None:
            np.add(xm, p, out=out)
        else:
            np.multiply(p, mod[3], out=p)
            np.add(xm, p, out=out)""", """    if T22_FAST_GLUE and _fg_h4:
        # S1 hook ffnpost: rms+mul+residual+absmax(xm)+absmax(y) in ONE pass.
        # The two absmax are memoised for the rec dict below (byte-identical).
        out, _fg_am = _fg_glue.post_ffn(y, xm, P["ffn_norm2"],
                                        None if mod is None else mod[3],
                                        threads=_fg_th)
        _fg_glue.memo_absmax(y, _fg_am[1])
        _fg_glue.memo_absmax(xm, _fg_am[0])
    elif T22_FAST_GLUE:
        p = rms_into(y, P["ffn_norm2"], L.n3buf)
        out = np.empty((SEQ, D), np.float32)   # escapes to the next layer
        if mod is None:
            np.add(xm, p, out=out)
        else:
            np.multiply(p, mod[3], out=p)
            np.add(xm, p, out=out)"""))


def build_block_step(T, verbose=True):
    """t18_run.block_step -> a mechanically patched copy (def block_step_fast).

    Refuses (returns None) unless EVERY hunk matches exactly once, so a product
    edit can never silently produce a half-patched block.  The applied diff is
    printed so a reviewer can see the four spots.
    """
    src = inspect.getsource(T.block_step)
    if not src.startswith("def block_step("):
        return None, "block_step source does not start at the def"
    src = src.replace("def block_step(", "def block_step_fast(", 1)
    applied = []
    for name, old, new in HUNKS:
        n = src.count(old)
        if n != 1:
            return None, "hunk %s matched %d times (need exactly 1)" % (name, n)
        src = src.replace(old, new, 1)
        applied.append(name)
    if verbose:
        print("[fastglue] block_step hunks applied: %s" % (", ".join(applied),),
              flush=True)
    return src, applied


# ---------------------------------------------------------------------------
# absmax memo (byte-identical by construction: same float32 element)
# ---------------------------------------------------------------------------
def _install_absmax(T, verbose=True):
    _memo = {}
    orig = T.absmax_fast

    def absmax_fast(x):
        v = _memo.get(id(x))
        if v is not None and v[0] is x:
            return v[1]
        return orig(x)
    absmax_fast._fg_memo = _memo
    T.absmax_fast = absmax_fast
    return _memo


def memo_absmax(x, v):
    m = getattr(getattr(sys.modules.get("t18_run"), "absmax_fast", None),
                "_fg_memo", None)
    if m is not None:
        m[id(x)] = (x, float(v))
    return v

# ---------------------------------------------------------------------------
# one projection dispatch (write X + W, launch, return the drain memoryview)
#
# T22_OVL splits this into the two halves a launch-ahead pipeline needs:
# _proj_post() issues the dispatch and returns a handle, _proj_join() waits and
# copies the drain out.  _dispatch_proj() is exactly post+join, so the delivered
# path is unchanged call for call.  The BOs are the SAME ones either way: the
# join copies the drain to host memory before the next post rewrites X/W, so no
# BO is double buffered.
# ---------------------------------------------------------------------------
def _proj_post(dev, PD, xb, w, tim, tag, w_write=True):
    """w_write=False skips the W payload: T22 proj-block issues 86 dispatches
    with ONE W image, and re-uploading 16.6 MB per block would be 1.4 GB of
    host->device traffic per projection.  The BO still holds that W (the
    caller writes it on the first block of a name and every dispatch that reads
    it has completed -- _proj_join waits)."""
    import time as _t
    bo = dev.bo
    bx, bw, bo_o = bo[PD._bn("x")], bo[PD._bn("w")], bo[PD._bn("o")]
    t0 = _t.perf_counter()
    bx.write(np.ascontiguousarray(xb).reshape(-1), 0)
    bx.sync(dev.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    if w_write:
        bw.write(w, 0)
        bw.sync(dev.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    t1 = _t.perf_counter()
    im = dev.imgs[PD._IMG]
    run = im["k"](3, im["bo_i"], len(im["inst"]) // 4, bx, bw, bo_o)
    if tim is not None:
        tim["write"] = tim.get("write", 0.0) + 1e3 * (t1 - t0)
    return (run, bo_o, t1)


def _proj_join(dev, PD, h, nt, tim, tag, ins=0.0):
    """wait for _proj_post's dispatch, then drain it.

    ins = host seconds the caller deliberately ran BETWEEN post and join; they
    are inside the launch->wait interval, so they are subtracted from the
    reported device time (the ledger stays a device interval, not a window).
    """
    import time as _t
    run, bo_o, t1 = h
    st = int(run.wait(300000))
    wall = 1e3 * (_t.perf_counter() - t1) - 1e3 * ins
    if st != 4:
        raise RuntimeError("fastglue %s state %d" % (tag, st))
    if tim is not None:
        tim["dev"] = tim.get("dev", 0.0) + wall
    bo_o.sync(dev.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    return bo_o.read(nt * 96 * 3840 * 2, 0)


def _dispatch_proj(dev, PD, xb, w, nt, tim, tag, th, w_write=True):
    return _proj_join(dev, PD, _proj_post(dev, PD, xb, w, tim, tag, w_write),
                      nt, tim, tag)


# ---------------------------------------------------------------------------
# the attention hook: q/k skip the f32 drain, rope eats the bf16 drain
# ---------------------------------------------------------------------------
def _attention_fn():
    """the product's host-f32 attention (webui/chunked.py:_attention)."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from webui.chunked import _attention
    return _attention


def _make_attention(PB, PD, T, th, verbose=True, chain=None):
    """T22_CHAIN2B: every buffer below belongs to ONE chain.

    chain=None keeps the single-chain path byte-identical (chain 0, the same
    dict object for the life of the process).  chain=N gives chain N its own
    _SCR[N], so two chains' qh/kh/vh/o16 can never alias.
    """
    _attention = _attention_fn()
    chain = _own(chain_now() if chain is None else int(chain))
    _scr = _SCR.setdefault(chain, {})

    def buf(key, shape, dtype):
        b = _scr.get(key)
        if b is None or b.shape != tuple(shape) or b.dtype != dtype:
            b = np.empty(shape, dtype)
            _scr[key] = b
        return b

    def dispatch_raw(dev, xb, w, nt, tim, tag):
        return _dispatch_proj(dev, PD, xb, w, nt, tim, tag, th)

    def drain_rows(raw, nrows):
        """The first nrows ROWS of a whole-tile drain: a VIEW, never a copy.

        A projection dispatch always drains ceil(seq/96)*96 rows, because the
        image is compiled for whole row tiles and the pad rows are part of
        every launch.  At SEQ=96 (128^2) and SEQ=1056 (512^2) that is exactly
        seq, so it was never observable; at 384^2 SEQ=608 = 6*96 + 32 and the
        drain is 7*96 = 672 rows = 2,580,480 u16 while every consumer of this
        hook wants 608 rows = 2,334,720.

        rope_fused has always worked this way (it takes seq from out.shape and
        reads the first seq rows) and proj_read_cv below has always taken
        u[:rows*cols] -- widen_t is the ONE consumer that VALIDATES, and at
        384^2 it raised

            ValueError: widen_t: input 2580480 elements, expected 2334720

        i.e. the whole 384^2 shape could not run at all (job
        20261103-180617-a738d1, 9 postflight FAIL).  Crop HERE rather than
        loosening widen_t: the assertion is worth keeping (it is what caught
        the memoryview-counted-as-uint8 defect, GLUE-AUDIT 8.2), and the pad
        rows carry no information -- the image computed them from the stale
        tail of the X BO, and nothing downstream may read them.
        """
        u = G._as_u16(raw).reshape(-1)
        need = int(nrows) * int(T.HEADS) * int(T.HD)
        return u[:need] if u.size > need else u

    def bigm_attention(self, x1, key, cos, sin, tim):
        import time as _t
        P = self.params(key)
        seq, D, HEADS, HD = T.SEQ, T.D, T.HEADS, T.HD
        # T22 proj-block: rows per projection dispatch (0 = one whole-shape
        # dispatch, which is what every delivered shape takes).  The BO was
        # sized by the same function, so the drain this function asks for and
        # the memory the BO owns cannot disagree.
        blk = (getattr(T, "proj_block_rows", None) or (lambda s: 0))(seq)
        nt = 1 if blk else (seq + 95) // 96
        if not blk:
            # THE TILE-COVERAGE GUARD.  A fixed-tile image computes exactly its
            # own RT row tiles per launch; handing it more rows does not error,
            # it silently leaves the tail of the drain untouched (measured:
            # plain15, 2112 rows in, rows 1056..2111 zero and unmoved when the
            # X bytes above row 1056 change).  The consumer -- this function,
            # the only one that dispatches a WHOLE shape -- is the only place
            # that can check it, because prod_devkill.proj4's 1056-row chunks
            # are correct by construction and must not be refused.
            _img = str(getattr(PD, "_IMG", "") or "")
            _rt = int(T.img_tiles(_img)) if hasattr(T, "img_tiles") else 0
            _need_t = (seq + 95) // 96
            if _rt and _rt < _need_t:
                raise SystemExit(
                    "[fastglue] REFUSAL: projection image %r is compiled for "
                    "RT=%d row tiles = %d rows, but SEQ=%d needs ceil(%d/96) = "
                    "%d tiles = %d rows: short by %d rows (%d tiles).  ONE "
                    "launch of a fixed-tile image cannot compute more than its "
                    "own tile count, so rows %d..%d of every drain would be "
                    "STALE BO MEMORY, not the projection (measured on this "
                    "box).  The shipping fix is the DEFAULT "
                    "(T22_PROJ_BLOCK=1: 96-row blocks); T22_PROJ_BLOCK=0 is "
                    "the A/B control and is arithmetically wrong above "
                    "RT*96 rows."
                    % (_img, _rt, _rt * 96, seq, seq, _need_t, _need_t * 96,
                       _need_t * 96 - _rt * 96, _need_t - _rt, _rt * 96,
                       _need_t * 96 - 1))
        t0 = _t.perf_counter()
        u16_in = getattr(x1, "dtype", None) == np.uint16
        if u16_in:
            x2 = np.ascontiguousarray(x1).reshape(seq, D)  # already an image
            x16 = None if blk else x2.reshape(-1)
        else:
            x2 = x1
            x16 = None if blk else G.bf16_rne(x2, threads=th)
        tim.setdefault("pack", 0.0)
        tim["pack"] += 1e3 * (_t.perf_counter() - t0)
        t_proj = _t.perf_counter()
        wq = T.bf16f32(P["norm_q"] * self.fold)
        qh = kh = vh = None
        if blk:
            # ---- T22 proj-block: ONE dispatch per 96-row tile --------------
            # WHY (see t18_run.proj_block_rows): a launch of the fixed-tile
            # image computes 11 tiles = 1056 rows whatever the X BO holds, so
            # above 1056 rows this function's old single dispatch computed the
            # first 1056 rows and handed _attention a drain whose tail was
            # stale BO memory.  Rows are independent in out[r] = x[r] @ W, so
            # 96-row blocks are the SAME arithmetic on a row range, and they
            # are the only shape the image can actually compute.
            #
            # W is written ONCE per name: 86 dispatches share one 16.6 MB
            # payload instead of re-uploading it 86 times (1.4 GB/projection).
            # The drain is one tile, so _proj_join reads 737,280 B, not 63 MB.
            qh = buf("qh", (HEADS, seq, HD), np.float32)
            kh = buf("kh", (HEADS, seq, HD), np.float32)
            vh = buf("vh", (HEADS, seq, HD), np.float32)
            xpad = np.zeros((blk, D), np.uint16) if u16_in else None
            fpad = None if u16_in else np.zeros((blk, D), np.float32)
            for nm, wname in (("q", "q"), ("k", "k"), ("v", "v")):
                w = _proj_w_cache(PD, PB, key, wname)
                dst = qh if nm == "q" else (kh if nm == "k" else vh)
                w_rms = wq if nm == "q" else P["norm_k"]
                first = True
                for r0 in range(0, seq, blk):
                    rows = min(blk, seq - r0)
                    if rows == blk:
                        seg = x2[r0:r0 + rows]
                        xb = (np.ascontiguousarray(seg) if u16_in
                              else G.bf16_rne(seg, threads=th))
                    elif u16_in:
                        # a partial LAST tile: zero-pad it, because the BO now
                        # holds one tile and its tail is no longer supplied by
                        # a widened BO (the old whole-shape dispatch read
                        # whatever the BO tail happened to contain and the
                        # consumer cropped it -- that accident is gone).
                        xpad[:rows] = x2[r0:r0 + rows]
                        xpad[rows:] = 0
                        xb = xpad
                    else:
                        fpad[:rows] = x2[r0:r0 + rows]
                        fpad[rows:] = 0.0
                        xb = G.bf16_rne(fpad, threads=th)
                    # nt is the image TILES this dispatch covers, not 1: with
                    # the default 96-row block they are the same, but a larger
                    # T22_PROJ_BLOCK (the 1056-row A/B) issues 11 tiles and the
                    # drain must be read as 11*96 rows -- reading 96 would leave
                    # rope_fused indexing past the drained buffer (measured:
                    # 0xC0000005 in _rope_k on the first layer).
                    raw = _dispatch_proj(self.dev, PD,
                                         np.asarray(xb).reshape(-1), w,
                                         (rows + 95) // 96,
                                         tim, "att/" + nm, th, w_write=first)
                    first = False
                    t1 = _t.perf_counter()
                    if nm == "v":
                        G.widen_t(drain_rows(raw, rows), rows, HEADS, HD,
                                  out=dst, threads=th, row0=r0)
                    else:
                        G.rope_fused(raw, w_rms, cos, sin, out=dst, threads=th,
                                     row0=r0, rows=rows)
                    tim["pack"] += 1e3 * (_t.perf_counter() - t1)
        elif _OVL:
            # T22_OVL: launch-ahead by one.  The k projection is issued BEFORE
            # q's drain is rope'd, and the v projection before k's is.  k reads
            # X and its own W and writes the shared drain BO, all of which q has
            # already finished with (its drain was copied out by _proj_join), so
            # the same single BO pair carries both -- no double buffering.  The
            # rope kernel therefore runs on the device's clock instead of on the
            # critical path.  Sampled maths are unchanged: same X bytes, same W
            # objects, same k-group order, same rope kernel and buffer.
            th_o = _OVL_TH or th
            h = _proj_post(self.dev, PD, x16,
                           _proj_w_cache(PD, PB, key, "q"), tim, "att/q")
            raw = _proj_join(self.dev, PD, h, nt, tim, "att/q")
            h = _proj_post(self.dev, PD, x16,
                           _proj_w_cache(PD, PB, key, "k"), tim, "att/k")
            t1 = _t.perf_counter()
            qh = G.rope_fused(raw, wq, cos, sin,
                              out=buf("qh", (HEADS, seq, HD), np.float32),
                              threads=th_o)
            _d = _t.perf_counter() - t1
            tim["pack"] += 1e3 * _d
            raw = _proj_join(self.dev, PD, h, nt, tim, "att/k", ins=_d)
            h = _proj_post(self.dev, PD, x16,
                           _proj_w_cache(PD, PB, key, "v"), tim, "att/v")
            t1 = _t.perf_counter()
            kh = G.rope_fused(raw, P["norm_k"], cos, sin,
                              out=buf("kh", (HEADS, seq, HD), np.float32),
                              threads=th_o)
            _d = _t.perf_counter() - t1
            tim["pack"] += 1e3 * _d
            raw = _proj_join(self.dev, PD, h, nt, tim, "att/v", ins=_d)
            t1 = _t.perf_counter()
            vh = G.widen_t(drain_rows(raw, seq), seq, HEADS, HD,
                           out=buf("vh", (HEADS, seq, HD), np.float32),
                           threads=th_o)
            tim["pack"] += 1e3 * (_t.perf_counter() - t1)
        else:
            for nm, wname in (("q", "q"), ("k", "k"), ("v", "v")):
                w = _proj_w_cache(PD, PB, key, wname)
                raw = dispatch_raw(self.dev, x16, w, nt, tim, "att/" + nm)
                t1 = _t.perf_counter()
                if nm == "q":
                    qh = G.rope_fused(raw, wq, cos, sin,
                                      out=buf("qh", (HEADS, seq, HD), np.float32),
                                      threads=th)
                elif nm == "k":
                    kh = G.rope_fused(raw, P["norm_k"], cos, sin,
                                      out=buf("kh", (HEADS, seq, HD), np.float32),
                                      threads=th)
                else:
                    vh = G.widen_t(drain_rows(raw, seq), seq, HEADS, HD,
                                   out=buf("vh", (HEADS, seq, HD), np.float32),
                                   threads=th)
                tim["pack"] += 1e3 * (_t.perf_counter() - t1)
        th_ = _t.perf_counter()
        o = _attention(qh, kh, vh)
        tim["host"] = tim.get("host", 0.0) + 1e3 * (_t.perf_counter() - th_)
        ta = _t.perf_counter()
        of = np.ascontiguousarray(o.reshape(seq, D))
        if blk:
            # ---- the o projection, same blocks ---------------------------
            # absmax over the REAL rows only, per block; max() of the per-block
            # maxima is the same f32 value the whole-tensor pass produced
            # (comparison only, no arithmetic).  y is one (seq, D) f32 buffer,
            # so the absmax memo keys the same object block_step will see.
            y = np.empty((seq, D), np.float32)
            ob = np.empty((blk, D), np.uint16)
            opad = np.empty((blk, D), np.float32)
            w_o = _proj_w_cache(PD, PB, key, "o")
            amax = None
            first = True
            for r0 in range(0, seq, blk):
                rows = min(blk, seq - r0)
                if rows == blk:
                    G.bf16_of(of[r0:r0 + rows], out=ob, threads=th)
                else:
                    opad[:rows] = of[r0:r0 + rows]
                    opad[rows:] = 0.0
                    G.bf16_of(opad, out=ob, threads=th)
                raw = _dispatch_proj(self.dev, PD, ob.reshape(-1), w_o,
                                     (rows + 95) // 96, tim, "att/o", th,
                                     w_write=first)
                first = False
                _seg, _a = G.proj_read_cv(raw, rows, D, out=y[r0:r0 + rows],
                                          threads=th, want_absmax=True)
                amax = _a if amax is None else max(amax, _a)
            tim["pack"] += 1e3 * (_t.perf_counter() - ta)
            # block_step's absmax_fast(a) hits this (in THIS chain's module)
            memo_absmax(y, amax, T)
            yv = y
        else:
            o16 = G.bf16_of(of, out=buf("o16", (seq * D,), np.uint16),
                            threads=th)
            tim["pack"] += 1e3 * (_t.perf_counter() - ta)
            raw = dispatch_raw(self.dev, np.ascontiguousarray(o16).reshape(-1),
                               _proj_w_cache(PD, PB, key, "o"), nt, tim, "att/o")
            t2 = _t.perf_counter()
            # rows=seq, not nt*96: proj_read_cv takes u[:rows*cols], and
            # want_absmax then ranges over the REAL rows only -- which is the
            # tensor the product's absmax_fast(y) would have seen.  At 96/1056
            # rows the two are the same number; at 384^2 (M=608 of a 672-row
            # drain) the two pad rows of the last tile are garbage from the
            # stale BO tail and must not enter the absmax memo that scales the
            # next residual add.
            y, amax = G.proj_read_cv(raw, seq, D, threads=th, want_absmax=True)
            tim["pack"] += 1e3 * (_t.perf_counter() - t2)
            yv = y[:seq]
            memo_absmax(yv, amax, T)   # block_step's absmax_fast(a) hits this
        t_end = _t.perf_counter()
        if os.environ.get("T22_BIGM_DEBUG", "0") == "1":
            print("[fastglue] %s proj_qkv %.3f rope+build %.3f attn %.3f "
                  "proj_o %.3f" % (key, t_proj - t0, th_ - t_proj, ta - th_,
                                   t_end - ta), flush=True)
        return yv

    return bigm_attention


_W = {}


def _proj_w_cache(PD, PB, key, name):
    k2 = (key, name)
    if k2 not in _W:
        _W[k2] = PD._w(key, name)
    return _W[k2]

def _install_absmax(T, verbose=True):
    """absmax_fast with a weakref memo: the fused kernels already computed the
    value, so the product's three absmax passes collapse into the passes that
    produced the tensors.  Byte-identical: the memo stores the same float32."""
    import weakref
    orig = T.absmax_fast
    memo = {}

    def absmax_fast(x):
        e = memo.get(id(x))
        if e is not None and e[0]() is x:
            return e[1]
        return orig(x)
    absmax_fast._fg_memo = memo
    absmax_fast._fg_ref = weakref.ref
    T.absmax_fast = absmax_fast
    if verbose:
        print("[fastglue] absmax memo installed (weakref, no array retention)",
              flush=True)
    return memo


def memo_absmax(x, v, T=None):
    """store absmax(x) computed by a fused kernel (same float32 value).

    T = the t18_run MODULE this call belongs to.  With two chains in one
    process each chain has its own module instance and therefore its own
    absmax_fast memo; defaulting to sys.modules["t18_run"] would file chain B's
    value under chain A's memo, where chain A's absmax_fast would never find it
    (only a wasted recompute, never a wrong number -- but it would also grow
    chain A's memo with ids chain A owns nothing of).
    """
    m = getattr(getattr(T if T is not None else sys.modules.get("t18_run"),
                        "absmax_fast", None),
                "_fg_memo", None)
    if m is None:
        return v
    import weakref
    try:
        m[id(x)] = (weakref.ref(x), float(v))
    except TypeError:                     # not weak-referenceable: skip
        pass
    return v


def install(threads=None, verbose=True, T=None, PD=None, DVF=None, PB=None,
            chain=None):
    """Install the hooks on ONE chain's driver module.

    T22_CHAIN2B: T/PD/DVF/PB name the modules this install patches.  With one
    chain they are the process's modules and this signature is a no-op change.
    With two chains each chain has its own t18_run module instance (so its own
    block_step, rms_into, absmax memo and Loop class) and the SHARED
    prod_devkill/devkill_ffn/prod_loop_bigm modules are patched once per chain:
    the two installs are identical there except for PD.bigm_attention, which is
    the per-chain object this whole flag exists for.

    chain= is the chain this install belongs to; the per-chain latch lives in
    STATE_CHAIN so a second install() for chain B can no longer return
    "already installed" and leave chain B running chain A's hook.
    """
    chain = chain_now() if chain is None else int(chain)
    STATE_CHAIN.setdefault(0, STATE)          # chain 0 IS the process STATE
    st = STATE_CHAIN.setdefault(chain, {"installed": False, "threads": 0,
                                        "why": "", "hooks": {}})
    if os.environ.get("T22_FASTGLUE", "0") != "1":
        st["why"] = "T22_FASTGLUE!=1"
        return st
    if st["installed"]:
        return st
    th = int(threads if threads is not None
             else os.environ.get("T22_FASTGLUE_THREADS", "8"))
    st["threads"] = th
    global _OVL, _OVL_WGU, _OVL_TH
    _OVL = os.environ.get("T22_OVL", "0") == "1"
    _OVL_WGU = _OVL and os.environ.get("T22_OVL_WGU", "1") != "0"
    _OVL_TH = int(os.environ.get("T22_OVL_THREADS", "0") or 0)
    st["ovl"] = ("proj+guw" if _OVL_WGU else "proj") if _OVL else "off"
    if _OVL:
        print("[fastglue] T22_OVL=1: in-loop launch-ahead overlap -- proj "
              "rope/widen under the next projection dispatch, gu W payload "
              "under the dn dispatch (ovl_threads=%d)"
              % (_OVL_TH or th,), flush=True)
    skip = {s.strip() for s in
            os.environ.get("T22_FASTGLUE_SKIP", "").split(",") if s.strip()}
    strict = os.environ.get("T22_FASTGLUE_STRICT", "0") == "1"

    if T is None or PD is None or DVF is None or PB is None:
        import prod_loop_bigm as _PB
        import devkill_ffn as _DVF
        import t18_run as _T
        import prod_devkill as _PD
        T = T if T is not None else _T
        PD = PD if PD is not None else _PD
        DVF = DVF if DVF is not None else _DVF
        PB = PB if PB is not None else _PB

    gates = _gate_all(PB, DVF, T, th)
    if strict:
        bad = [h for h in ALL_HOOKS if not gates.get(h, False) and h not in skip]
        if bad:
            STATE["why"] = "strict: gate failed %s" % (bad,)
            print("[fastglue] REFUSED (strict): %s" % (STATE["why"],), flush=True)
            return STATE

    hooks = {}

    def on(name):
        if name in skip:
            hooks[name] = "skipped"
            return False
        if not gates.get(name, False):
            hooks[name] = "gate-failed"
            return False
        return True

    # ---- 1. bf16 RNE without the temporaries -----------------------------
    if on("bf16"):
        PB._bf16_bytes = lambda x: G.bf16_rne(x, threads=th)
        hooks["bf16"] = "on"

    # ---- 2. the projection: X prep + drain in one pass -------------------
    def fast_proj4(dev, x, key, names, tim=None, max_tiles=11):
        seq = x.shape[0]
        u16 = getattr(x, "dtype", None) == np.uint16
        x2 = np.ascontiguousarray(x) if u16 else x
        # T22 proj-block: the SAME block size the BO was sized for (the same
        # function, so they cannot drift).  0 -> the historical 11-tile chunk.
        blk = (getattr(T, "proj_block_rows", None) or (lambda s: 0))(seq)
        chunk = blk or (max_tiles * 96)

        def blocks():
            """One X payload per dispatch, built lazily (never all at once:
            with 96-row blocks a pre-built list would be the whole image)."""
            for r0 in range(0, seq, chunk):
                r1 = min(seq, r0 + chunk)
                rows = r1 - r0
                nt = (rows + 95) // 96
                if rows == nt * 96:
                    xb = (np.ascontiguousarray(x2[r0:r1]) if u16
                          else G.bf16_rne(x2[r0:r1], threads=th))
                else:
                    pad = np.zeros((nt * 96, x2.shape[1]),
                                   np.uint16 if u16 else np.float32)
                    pad[:rows] = x2[r0:r1]
                    xb = pad if u16 else G.bf16_rne(pad, threads=th)
                yield xb, rows, nt

        out = {}
        for nm in names:
            w = _proj_w_cache(PD, PB, key, nm)
            # Preallocate the (seq, 3840) f32 drain and fill it chunk by
            # chunk.  This used to keep every chunk alive in ys and then
            # np.concatenate() them, which needs a SECOND full (seq, 3840)
            # f32 while all the parts are still there: at M=8192 that is
            # 8192*3840*4 = 120.00 MiB of transient on top of the parts.
            # Same fix, same argument as fast_bigm_ffn at install.py:827
            # (job 20261103-222236-b084b4 died on exactly that transient).
            # Every element is copied exactly as concatenate copied it.
            _out = np.empty((seq, 3840), np.float32)
            off = 0
            first = True
            for xb, rows, nt in blocks():
                raw = _dispatch_proj(dev, PD, xb, w, nt, tim, "fast/" + nm,
                                     th, w_write=first)
                first = False
                _out[off:off + rows] = G.proj_read_cv(
                    raw, nt * 96, 3840, threads=th)[:rows]
                off += rows
            out[nm] = _out
        return out

    if on("proj"):
        PD.proj4 = fast_proj4
        hooks["proj"] = "on"

    # ---- 2a. T22_TIMELINE: per-dispatch wall-clock log (2026-11-02) ------
    # Records (t_start, t_end, tag) for every device round trip so the device's
    # busy/idle timeline can be drawn.  Opt-in; default OFF, zero cost.
    if os.environ.get("T22_TIMELINE", "0") == "1":
        _tlf = os.environ.get("T22_TIMELINE_FILE", "devtimeline.tsv")
        try:
            os.remove(_tlf)
        except OSError:
            pass

        def _tlwrap(fn, tag):
            import time as _tl_t
            def _w(*a, **k):
                _t0 = _tl_t.perf_counter()
                try:
                    return fn(*a, **k)
                finally:
                    _t1 = _tl_t.perf_counter()
                    try:
                        with open(_tlf, "a", encoding="utf-8") as _fh:
                            _fh.write("%.6f\t%.6f\t%s\n"
                                      % (_t0, _t1 - _t0, tag))
                    except OSError:
                        pass
            return _w

        PD._run = _tlwrap(PD._run, "proj")
        for _n in ("_dispatch", "decode_fast", "ffn_pair"):
            if hasattr(DVF, _n):
                setattr(DVF, _n, _tlwrap(getattr(DVF, _n), "ffn_" + _n))
        print("[fastglue] T22_TIMELINE -> %s" % _tlf, flush=True)

    # ---- 2b. T22_P33: 33-tile kk=32 projection (2026-11-02) --------------
    # One dispatch per name covering all 1056 rows (no 1024+32 split), 32/32
    # cores.  Measured 39.7 ms/layer vs the shipped 90.5 ms in the same process
    # (2.28x); corr 0.9998928..0.9998961 against a numpy bf16 GEMM with the
    # layer's real weights, tile 32 included, k-permuted control 0.0073.
    # Opt-in: default OFF keeps the shipped projection untouched.
    if os.environ.get("T22_P33", "0") == "1":
        try:
            import importlib.util as _ilu
            import pathlib as _pl3
            _pp = (_pl3.Path(__file__).resolve().parents[3] / "imaggers" / "util"
                   / "place" / "proj33.py")
            _sp = _ilu.spec_from_file_location("_p33mod", str(_pp))
            _m33 = _ilu.module_from_spec(_sp)
            _sp.loader.exec_module(_m33)
            _st = {}

            def _p33_proj4(dev, x, key, names, tim=None, max_tiles=11):
                if "done" not in _st:
                    _m33.install(dev)
                    _st["done"] = True
                return _m33.proj4_33(dev, x, key, names, tim)

            PD.proj4 = _p33_proj4
            hooks["proj"] = "on+p33"
            print("[fastglue] T22_P33: projection -> 33-tile kk=32 "
                  "(measured 39.7 vs 90.5 ms/layer, corr 0.99989)", flush=True)
        except Exception as _e33:
            print("[fastglue] T22_P33 unavailable, keeping the shipped "
                  "projection: %r" % (_e33,), flush=True)

    # ---- 3. FFN A image / Y drain ----------------------------------------
    if on("pack"):
        DVF.a_pack_fast = lambda x: G.pack_a(x, from_bf16=True, threads=th)
        hooks["pack"] = "on"
    # T22_CHAIN2B: the decode scratch is per CHAIN, not per install.  This
    # closure is shared by both chains (devkill_ffn.decode_fast is one module
    # attribute), so the buffer is looked up through the CALLING THREAD's chain
    # at call time rather than closed over.
    def fast_decode(raw, n_tiles, rows):
        _y_scr = _Y_SCR.setdefault(_own(chain_now()), {})
        key = (n_tiles, rows)
        out = _y_scr.get(key)
        if out is None:
            out = np.empty((n_tiles * 96, 3840), np.float32)
            _y_scr[key] = out
        return G.decode_y(raw, n_tiles, rows, out=out, threads=th)

    if on("decode"):
        DVF.decode_fast = fast_decode
        hooks["decode"] = "on"

    # ---- 4. the FFN A chain + the FFN entry ------------------------------
    def fast_bigm_ffn(dev, xm, swiglu_w, b3_w, tim=None,
                      imgs=("swiglu", "down"), tail_imgs=("swiglu1", "down1")):
        """One image pair for the whole call, partial last tile included.

        Identical contract to devkill_ffn.bigm_ffn (see the long note there):
        the last row tile of a partial M is zero-padded and dispatched through
        the SAME pair, then the drain is cropped to seq.  tail_imgs is kept for
        call compatibility and is NOT used -- at 384^2 (M=608) it selected the
        RT=1 pair ("swiglu1", "down1"), whose A input is the PACKED A plane and
        whose B payload is the n128 order, while this arm hands the down image
        the raw k=32 swiglu C drain and the n32 payload.  That made the 32 tail
        rows wrong, and attention then carried them into every token: campaign
        vs delivered measured 0.560995 at 384^2 x8, against 0.994694 for the
        same arm at 512^2 x1 where there is no tail (DET-BISECT-REPORT-2 6.1,
        6.2).  rem == 0 (128^2 M=96, 512^2 M=1056) is byte for byte the old
        path: one dispatch of the same xb.
        """
        import time as _ft
        import os as _fo
        _fp = _fo.environ.get("T22_FFN_PROF", "0") == "1"
        _f0 = _ft.perf_counter()
        _fn = 0
        seq = xm.shape[0]
        _rt = 11 if str(imgs[0]).endswith("11") else 17
        _chunk = _rt * 96
        pre16 = getattr(xm, "dtype", None) == np.uint16
        if seq > _chunk:
            # Preallocate the drain and fill it chunk by chunk.  This used to
            # accumulate 8 per-chunk .copy()s and then np.concatenate() them,
            # which needed a SECOND full (seq, 3840) f32 while all 8 parts were
            # still alive: at M=8192 that is 120 MiB of transient, and it is
            # what died at install.py:852 with MemoryError (job
            # 20261103-222236-b084b4, step 6).  Filling a preallocated array is
            # strictly better even with memory to spare -- it drops the
            # transient AND one whole copy of the tensor, and the assignment
            # copies out of the reused Y scratch exactly as .copy() did.
            _out = np.empty((seq, xm.shape[1]), np.float32)
            for r0 in range(0, seq, _chunk):
                seg = xm[r0:r0 + _chunk]
                rows = seg.shape[0]
                nt = (rows + 95) // 96
                if pre16:
                    xb = np.ascontiguousarray(seg).reshape(-1)
                    if nt * 96 != rows:
                        # a PARTIAL last chunk in image form: pad it to the tile
                        # like the f32 branch below (768^2's 2336 rows end on
                        # 224, which a_pack_fast's rows%96==0 assert refuses)
                        p16 = np.zeros((nt * 96, seg.shape[1]), xb.dtype)
                        p16[:rows] = seg
                        # The pad path is where a wrong length hides: this
                        # branch used to assign the already-flattened xb into
                        # the 2-D slice and raised a broadcast error at 1080p's
                        # 800-row tail.  seg keeps both branches structurally
                        # identical to the f32 pad two lines below, and the
                        # shape is named here so a silent truncation cannot be.
                        assert p16.shape == (nt * 96, 3840), (
                            "FFN pad shape: rows=%d nt=%d seg=%r pad=%r, "
                            "want (%d, 3840)"
                            % (rows, nt, seg.shape, p16.shape, nt * 96))
                        xb = p16.reshape(-1)
                else:
                    pad = np.zeros((nt * 96, seg.shape[1]), np.float32)
                    pad[:rows] = seg
                    xb = G.bf16_rne(pad, threads=th)
                # The assignment into _out is the copy: decode_y returns a view
                # into the reused Y scratch, and it must be out before the next
                # chunk overwrites it (1024^2 has 4 chunks).
                _out[r0:r0 + rows] = DVF.block(dev, xb.reshape(-1, seg.shape[1]),
                                               swiglu_w, b3_w, imgs, tim,
                                               slot=0, nreal=rows)[:rows]
                _fn += 1
            if _fp:
                print("[ffnprof] seq=%d chunks=%d total %.1f ms" % (seq, _fn, 1e3*(_ft.perf_counter()-_f0)), flush=True)
            return _out
        rt, rem = divmod(seq, 96)
        xb = (np.ascontiguousarray(xm).reshape(seq, -1) if pre16
              else G.bf16_rne(xm, threads=th).reshape(seq, -1))
        if rem:
            # PARTIAL LAST TILE (384^2: M = 608 = 6*96 + 32): zero-pad to the
            # tile and run the SAME pair once -- the >_chunk branch above, the
            # delivered path (ffn M=96 x 7 blocks) and the plan ([[0, 608]])
            # all do exactly this, and it is why they are right at 384^2.
            pad = np.zeros(((rt + 1) * 96, xb.shape[1]), xb.dtype)
            pad[:seq] = xb
            xb = pad
        _out = DVF.block(dev, xb, swiglu_w, b3_w, imgs, tim, slot=0,
                         nreal=seq)[:seq]
        if _fp:
            print("[ffnprof] seq=%d small total %.1f ms" % (seq, 1e3*(_ft.perf_counter()-_f0)), flush=True)
        return _out

    if on("ffnchain"):
        DVF.bigm_ffn = fast_bigm_ffn
        hooks["ffnchain"] = "on"

    # ---- 5. rms without the six numpy passes -----------------------------
    if on("rms"):
        T.rms_into = lambda x, w, out: G.norm_f32(x, w, None, out=out,
                                                  threads=th)
        hooks["rms"] = "on"

    # ---- 6. attention: q/k skip the f32 drain, rope eats the bf16 drain --
    if on("attn"):
        fa = _make_attention(PB, PD, T, th, verbose, chain=chain)
        _ATT[chain] = fa
        if _two_chain():
            # install.py:1051 used to install ONE PD.bigm_attention.  Two chains
            # need one per chain, and the Loop identifies the chain at call time
            # (block_step calls L.attention(x1, key, cos, sin, tim)), so the
            # installed object is a dispatcher over _ATT.
            PD.bigm_attention = _att_dispatch
        else:
            PD.bigm_attention = fa
        if getattr(T.Loop, "attention", None) is not None and \
                getattr(T.Loop.attention, "__name__", "") == "bigm_attention":
            T.Loop.attention = PD.bigm_attention   # chunked installed it already
        hooks["attn"] = "on"
        # T22 proj-block: configure() sized the projection BOs from the env
        # (it runs before this install), so record that the blocked dispatcher
        # is really the one in place; the refusal below enforces the pairing.
        T.PROJ_BLOCK_INSTALLED = bool(getattr(T, "proj_block_rows", None))

    # ---- 7. block_step: 4 hunks, generated from the product source -------
    _install_absmax(T, verbose)
    if on("block"):
        if getattr(T.block_step, "__name__", "") != "block_step":
            hooks["block"] = "not-the-delivered-block_step(%s)" % (
                getattr(T.block_step, "__name__", "?"),)
        else:
            T._fg_glue = sys.modules[__name__]
            T._fg_th = th
            keep = {s.strip() for s in
                    os.environ.get("T22_FASTGLUE_HUNKS", "1,2,3,4").split(",")
                    if s.strip()}
            # TYPE COUPLINGS (reproduced on the fake-device path 2026-11-01):
            # h1 hands a bf16 IMAGE to L.attention.  Only the fast attention
            # accepts one; the delivered prod_devkill.bigm_attention feeds it
            # to np.zeros(...)[:] = x, i.e. a NUMERIC uint16->float32 cast,
            # which corrupted the whole chain (parent arm E: corr -0.184).
            # h3 hands a bf16 image to the FFN, which only fast_bigm_ffn takes.
            attn_on = str(hooks.get("attn", "")).startswith("on")
            ffn_on = str(hooks.get("ffnchain", "")).startswith("on")
            devkill = os.environ.get("T22_DEVKILL", "0") == "1"
            T._fg_h1 = ("1" in keep) and attn_on
            T._fg_h2 = "2" in keep
            T._fg_h3 = ("3" in keep) and ffn_on and devkill
            T._fg_h4 = "4" in keep
            off = [n for n, v in (("h1", T._fg_h1), ("h2", T._fg_h2),
                                  ("h3", T._fg_h3), ("h4", T._fg_h4)) if not v]
            if off:
                print("[fastglue] block hunks off by couplings: %s"
                      % (", ".join(off),), flush=True)
            src, info = build_block_step(T, verbose)
            if src is None:
                hooks["block"] = "hunk-failed: %s" % (info,)
            else:
                ns = T.__dict__
                exec(compile(src, "<fastglue block_step_fast>", "exec"), ns)
                T.block_step = ns["block_step_fast"]
                hooks["block"] = "on(%d hunks)" % (len(info),)

    # ------------------------------------------------------------------
    # T22 proj-block: the projection BOs are ONE 96-row tile when
    # t18_run.proj_block_rows(SEQ) says so -- a decision configure() had to
    # make before this function ran.  If the blocked dispatcher is not the one
    # that got installed, the delivered dispatcher would write ceil(SEQ/96)*96
    # rows into a one-tile BO.  Refuse, loudly, naming both knobs.
    _pb = getattr(T, "proj_block_rows", None)
    _seq = int(getattr(T, "SEQ", 0) or 0)
    _rows = int(_pb(_seq)) if _pb is not None and _seq else 0
    if _rows and not str(hooks.get("attn", "")).startswith("on"):
        raise SystemExit(
            "[fastglue] REFUSAL: the big-M projection BOs were sized for "
            "%d-row blocks (SEQ=%d, t18_run.proj_block_rows) but the blocked "
            "attention hook is NOT installed (%s).  The delivered dispatcher "
            "writes ceil(SEQ/96)*96 rows and would run past the BO.  Re-run "
            "with T22_PROJ_BLOCK=0 for whole-shape BOs."
            % (_rows, _seq, hooks.get("attn")))
    if verbose:
        print("[fastglue] projection dispatch: %s"
              % ("blocked, %d rows x %d block(s)" % (_rows, (_seq + _rows - 1)//_rows)
                 if _rows else
                 "whole shape, one dispatch (%d rows)" % _seq,), flush=True)
    st["hooks"] = hooks
    # t22_loud: "installed" is the RE-ENTRANCY latch (see :628) and must keep
    # meaning "install() has run".  What it never meant -- and what every caller
    # was reading it as -- is "the hooks are in place".  That is complete/n_on.
    n_on = sum(1 for v in hooks.values() if str(v).startswith("on"))
    st["n_on"] = n_on
    st["n_total"] = len(ALL_HOOKS)
    st["complete"] = (n_on == len(ALL_HOOKS))
    st["installed"] = True
    st["chain"] = chain
    st["why"] = ("" if st["complete"] else
                    "%d/%d hooks: %s" % (n_on, len(ALL_HOOKS),
                                         " ".join("%s=%s" % (k, v)
                                                  for k, v in hooks.items()
                                                  if not str(v).startswith("on"))))
    if verbose:
        print("[fastglue] installed %d/%d hooks (threads=%d chain=%d): %s"
              % (n_on, len(ALL_HOOKS), th, chain,
                 " ".join("%s=%s" % (k, v) for k, v in hooks.items())),
              flush=True)
        if not st["complete"]:
            # A gate that fails is not a decision the user made; it is a hook
            # that did not install.  Say so in the one line an operator (and the
            # runbook's Q04) already greps, and say what it costs.
            print("[fastglue] INCOMPLETE %d/%d hooks -- this run continues on "
                  "the slower path: %s" % (n_on, len(ALL_HOOKS), st["why"]),
                  flush=True)
    return st


def _two_chain():
    return os.environ.get("T22_CHAIN2B", "0") == "1"


def _att_dispatch(self, x1, key, cos, sin, tim):
    """PD.bigm_attention for the two-chain path: pick the chain own closure.

    `self` is the Loop, so the chain is a property of the CALLER, not of the
    thread: one extra dict lookup per layer, and it cannot be fooled by a
    chain whose host work migrates between threads.
    """
    return _ATT[_own(getattr(self, "chain", 0))](self, x1, key, cos, sin, tim)


def status(chain=None):
    if chain is None:
        return dict(STATE)
    return dict(STATE_CHAIN.get(int(chain)) or STATE)
