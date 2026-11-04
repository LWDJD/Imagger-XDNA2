# SPDX-License-Identifier: MIT
"""plan_ffn.py -- wire the ADAPTIVE PLAN's image set into the product loop.

THE GAP THIS CLOSES
    webui/npu_plan.py decides a legal 32-core placement for any M;
    imaggers/util/place/npu_synth.py builds its images and records a manifest.
    Neither is reachable from the product loop: t18_run.block_step's device-FFN
    branch calls devkill_ffn / prod_loop_bigm, whose images are the DELIVERED
    M=96 / RT=11 family.  So every shape still ran the delivered chunked path.

    This module is the missing host glue.  Three things, in this order:

      register(SEQ)   plan -> image tags -> t18_run.BIGM_IMGS / BIGM_SIZES, so
                      Dev.__init__ registers the xclbin and allocates
                      b_<tag>_{x,w,o} on the image's OWN group ids (3/4/5)
                      with the plan's a/b/c byte counts.  Must run BEFORE Dev().
      payloads(key)   the per-layer B payloads: pack_kk.pack_b_kk on the model's
                      own BFP16 blocks (a PERMUTATION, never a re-encode).
                      The product's resident CACHE_BG cannot be reused: it is
                      in the DELIVERED image's order, not the plan's.
      ffn(dev, ...)   s2_ffn32.ffn32_run + the section line the delivered
                      [devkill] line prints, so the columns are comparable.

WHY THE PRODUCT'S RESIDENT CACHE IS NOT REUSED
    prod_loop_bigm.repack_cache_inplace_bigm() rewrites the 30-layer resident
    cache into the DELIVERED swiglu/down image order.  The plan's B order is
    pack_kk's [n-tile][k-step][leaf] at the plan's kk (32, not 8), so the bytes
    are a different permutation.  Feeding CACHE_BG to the plan's images would
    be a silent wrong-B.  Repacking from weights/block<N>.npz is the honest
    route and costs 0.21 s/layer (measured, numba); T22_PLAN_BCACHE=1 makes it
    a one-off disk build (3.98 GB for 30 layers) that later runs mmap.

FLAGS (all default 0 => the delivered path is byte-for-byte unchanged)
    T22_PLAN_FFN=1            the product uses this path at all
    T22_PLAN_FFN_M=<int>      plan for this M instead of SEQ (diagnostics)
    T22_PLAN_B_LAYERS=<n>     in-memory payload LRU depth (default 3 = 398 MB)
    T22_PLAN_BCACHE=1         use/populate the on-disk payload cache
    T22_PLAN_FFN_BARRIER=0    disable the same-image-relaunch barrier (A/B)
    T22_PLAN_FFN_DEBUG=1      print the per-dispatch record
    T22_PLAN_FUSE_GU=1        plan+build the FUSED gu image (silu(gate)*up
                              inside the dispatch): ONE gu launch instead of
                              two, no barrier, no host elementwise pass.
                              Default 0 = the two-dispatch path above.

    py312 plan_ffn.py --m 1056 --check      # plan + images + sizes, no device
    py312 plan_ffn.py --m 1056 --prebuild   # build the disk payload cache
    py312 plan_ffn.py --selftest            # packers / perm / payload identity
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "out"
BCACHE = OUT / "plan_b"

for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pack_kk as PK                                          # noqa: E402
import s2_ffn32 as S2                                         # noqa: E402

PLAN = None
TAGS = None
_ENV = {
    # T22_PLAN_B_LAYERS = how many layers' B payloads stay RESIDENT.  0 (the
    # default) = as many as the byte budget allows, which for a 30-layer model
    # is all of them (30 x 132.7 MB = 3.98 GB).  It used to be 3, and that was
    # the plan arm's single largest cost: 30 layers > 3 slots means EVERY layer
    # MISSES EVERY STEP, so every step re-read 3.98 GB of weights/block<N>.npz
    # from disk and re-permuted it -- 0.21 s/layer measured on AC and up to
    # 1.1 s/layer on battery (a power-throttled disk), i.e. 6 s/step on AC and
    # ~33 s/step on battery, none of it visible in the [planffn] section line
    # (payloads() runs before ffn32_run starts its timer).  See
    # probe_plan_speed.py and section_table.py.
    "layers": int(os.environ.get("T22_PLAN_B_LAYERS", "0") or 0),
    # the byte budget for that cache; 0 = 6 GiB, enough for a 30-layer model
    # with room to spare.  T22_PLAN_B_BYTES overrides (bytes).
    "bytes": int(os.environ.get("T22_PLAN_B_BYTES", "0") or 0),
    "bcache": os.environ.get("T22_PLAN_BCACHE", "0") == "1",
    "debug": os.environ.get("T22_PLAN_FFN_DEBUG", "0") == "1",
    # the fused gu: a COMPILE-TIME knob of the gu image, so it must reach the
    # planner (and the plan_key) -- a different image, not a different launch
    "fuse_gu": os.environ.get("T22_PLAN_FUSE_GU", "0") == "1",
    "fuse_fill": float(os.environ.get("T22_PLAN_FUSE_FILL", "1.0") or 1.0),
    # DIAGNOSTIC ONLY (DET-BISECT test 1/2): hash the FFN input at the CALL SITE
    # and the payloads handed in, once per layer key, so they can be compared
    # with the executor's own record (s2_ffn32 prints its entry hash under the
    # same env flag).  Default 0: no hashing, no prints, no allocation.
    "probe": os.environ.get("T22_PLAN_FFN_PROBE", "0") == "1",
}
_LRU = collections.OrderedDict()
_LRU_BYTES = 0
_PROBED = set()
_BUDGET = 6 << 30          # 6 GiB unless T22_PLAN_B_BYTES says otherwise


def _payload_bytes(got):
    """Bytes a payload tuple holds, counting an aliased [gate|up] pair once."""
    seen, n = set(), 0
    for a in got:
        if a is None:
            continue
        i = id(a)
        if i in seen:
            continue
        seen.add(i)
        n += int(np.asarray(a).nbytes)
    return n


def _lru_clear():
    global _LRU_BYTES
    _LRU.clear()
    _LRU_BYTES = 0


def _lru_trim():
    """Evict oldest-first until the count AND the byte budget are satisfied.

    Both limits are optional: T22_PLAN_B_LAYERS=0 (default) means "no count
    limit", T22_PLAN_B_BYTES=0 (default) means 6 GiB.  The last entry is never
    evicted, so a single layer that exceeds the budget still runs (slowly).
    """
    global _LRU_BYTES
    budget = _ENV["bytes"] or _BUDGET
    depth = _ENV["layers"]
    while len(_LRU) > 1 and ((depth and len(_LRU) > depth)
                             or _LRU_BYTES > budget):
        _k, v = _LRU.popitem(last=False)
        _LRU_BYTES -= _payload_bytes(v)


# ---------------------------------------------------------------------------
# 1. the plan and its images
# ---------------------------------------------------------------------------
def image_paths(tag):
    """(xclbin, bin) for a plan image -- the paths npu_synth writes."""
    return OUT / (tag + ".xclbin"), OUT / (tag + ".bin")


def plan_m(seq):
    """The M this run plans for.  T22_PLAN_FFN_M overrides (diagnostics)."""
    v = os.environ.get("T22_PLAN_FFN_M", "")
    return int(v) if v else int(seq)


def resolve(seq, prefer_built=True):
    """plan for this SEQ, preferring an image set that is already built.

    Same affinity webui/shapes.npu_plan_for uses, so the product runs the plan
    the manifest names instead of a different equal-cost optimum.
    """
    global PLAN, TAGS
    M = plan_m(seq)
    if PLAN is not None and int(PLAN["M"]) == M:
        return PLAN
    if PLAN is not None:
        # a different M means a different plan, hence different payloads
        _lru_clear()
    prefer = None
    if prefer_built:
        import npu_synth as SYN
        prefer = SYN.built_plan_keys()
    PLAN = S2.plan_for(M, prefer=prefer, fuse_gu=_ENV["fuse_gu"],
                       fuse_fill=_ENV["fuse_fill"])
    TAGS = S2.tags_for(PLAN)
    if _ENV["fuse_gu"]:
        print("[planffn] T22_PLAN_FUSE_GU=1: the gu image is FUSED "
              "(silu(gate)*up in the dispatch) -- one gu launch per layer, no "
              "barrier, no host swiglu", flush=True)
    return PLAN


def missing_images(plan=None, tags=None):
    """[(gemm, tag, which)] for every artifact of the plan that is not on disk."""
    plan = plan or PLAN
    tags = tags or TAGS
    out = []
    for gemm in ("gu", "dn"):
        tag = tags[gemm]
        xcl, inst = image_paths(tag)
        for what, p in (("xclbin", xcl), ("bin", inst)):
            if not p.is_file():
                out.append((gemm, tag, what))
    return out


def register(seq, plan=None, tags=None, verbose=True):
    """Add the plan's images + BO sizes to t18_run's registry.  BEFORE Dev()."""
    import t18_run as T
    plan = plan or resolve(seq)
    tags = tags or TAGS
    miss = missing_images(plan, tags)
    if miss:
        raise RuntimeError(
            "plan_ffn.register: the plan's image set is incomplete (%s).  Build "
            "it with:  py312 imaggers/util/place/npu_synth.py --m %d"
            % (", ".join("%s/%s.%s" % m for m in miss[:4]), int(plan["M"])))
    added = []
    for gemm in ("gu", "dn"):
        tag = tags[gemm]
        g = plan["gemms"][gemm]
        T.BIGM_IMGS[tag] = image_paths(tag)
        T.BIGM_SIZES[tag] = (int(g["a_bytes"]), int(g["b_bytes"]),
                             int(g["c_bytes"]))
        added.append((tag, T.BIGM_SIZES[tag]))
    if verbose:
        print("[planffn] plan %s (%s m=%d kk=%d Mblk=%d ndisp=%d%s) -> %d images"
              % (plan["plan_key"], plan["strategy"], plan["m"], plan["kk"],
                 plan["Mblk"], plan["ndisp"],
                 " gu=FUSED" if plan.get("fuse_gu") else "", len(added)),
              flush=True)
        for tag, sz in added:
            print("[planffn]   %-44s x=%d w=%d o=%d B" % (tag, sz[0], sz[1], sz[2]),
                  flush=True)
    return plan


# ---------------------------------------------------------------------------
# 2. the per-layer B payloads
# ---------------------------------------------------------------------------
def _cache_paths(plan, li):
    """The on-disk payload files for one layer.

    A fused plan declares TWO payloads (the [gate|up] pair and dn), so its
    cache is two files of the same total size (132.7 MB/layer) -- the pair is
    stored once, not twice, and None marks the slot that does not exist.
    """
    d = BCACHE / plan["plan_key"]
    if plan.get("fuse_gu"):
        return (d / ("L%d_pair.npy" % li), None, d / ("L%d_dn.npy" % li))
    return (d / ("L%d_gate.npy" % li), d / ("L%d_up.npy" % li),
            d / ("L%d_dn.npy" % li))


def payloads(key, plan=None, use_cache=None):
    """{key: (b_gate, b_up, b_dn)} -- LRU, then disk cache, then repack."""
    plan = plan or PLAN
    if plan is None:
        raise RuntimeError("plan_ffn.payloads: no plan (call register(SEQ) first)")
    li = int(str(key).split(".")[1])
    k = (plan["plan_key"], li)
    hit = _LRU.get(k)
    if hit is not None:
        _LRU.move_to_end(k)
        return hit
    use_cache = _ENV["bcache"] if use_cache is None else bool(use_cache)
    got = None
    if use_cache:
        paths = _cache_paths(plan, li)
        have = [p for p in paths if p is not None]
        if have and all(p.is_file() for p in have):
            arrs = [np.load(p, mmap_mode="r") for p in have]
            got = ((arrs[0], arrs[0], arrs[1]) if plan.get("fuse_gu")
                   else tuple(arrs))
    if got is None:
        t0 = time.perf_counter()
        got = S2.payloads_for(key, plan)
        if use_cache:
            paths = _cache_paths(plan, li)
            paths[0].parent.mkdir(parents=True, exist_ok=True)
            for p, a in zip(paths, got):
                if p is not None:                 # a fused plan's pair once
                    np.save(p, a)
        if _ENV["debug"]:
            print("[planffn] packed %s in %.2f s" % (key, time.perf_counter() - t0),
                  flush=True)
    global _LRU_BYTES
    _LRU[k] = got
    _LRU_BYTES += _payload_bytes(got)
    _lru_trim()
    return got


def _probe_callsite(xn, key, plan, b):
    """T22_PLAN_FFN_PROBE=1: the bytes this call site is about to hand over.

    Test 2 of DET-BISECT is an ALIASING question: ffn() passes _xn, which IS
    L.n1buf (a REUSED buffer), and s2_ffn32.ffn32_run hands xn to as_a_f32,
    which returns a float32 input AS ITSELF (a REFERENCE, not a copy; a uint16
    bf16 image is decoded, and the executor records the image's dtype anyway).
    So the identity of the array is recorded here
    (id, data pointer, base, writeable) together with its sha, and the executor
    prints ITS OWN sha at entry: equal hashes mean the buffer survived the
    hand-off (test clean), unequal hashes name the aliasing bug.
    """
    import hashlib
    a = np.ascontiguousarray(xn)
    sha = hashlib.sha256(memoryview(a).cast("B")).hexdigest()[:16]
    print("[pffnprobe] call-site key=%s xn id=%d data=0x%x base=%r "
          "shape=%r dtype=%s writeable=%s absmax=%.6g sha=%s"
          % (key, id(xn), int(xn.ctypes.data),
             (type(xn.base).__name__ if xn.base is not None else None),
             tuple(np.asarray(xn).shape), np.asarray(xn).dtype,
             bool(np.asarray(xn).flags.writeable), float(np.abs(xn).max()), sha),
          flush=True)
    for nm, arr in zip(("b_gate", "b_up", "b_dn"), b):
        print("[pffnprobe] call-site key=%s %-6s %d B sha=%s"
              % (key, nm, int(np.asarray(arr).size),
                 hashlib.sha256(memoryview(np.ascontiguousarray(arr)).
                                cast("B")).hexdigest()[:16]), flush=True)


def prebuild(keys, plan=None, verbose=True):
    """Write the on-disk payload cache for these layers (one-off, 132.7 MB each)."""
    plan = plan or PLAN
    t0 = time.perf_counter()
    n = 0
    for key in keys:
        payloads(key, plan, use_cache=True)
        n += 1
        if verbose and n % 5 == 0:
            print("[planffn] prebuilt %d/%d layers (%.1f s)"
                  % (n, len(keys), time.perf_counter() - t0), flush=True)
    return {"layers": n, "seconds": round(time.perf_counter() - t0, 2),
            "dir": str(BCACHE / plan["plan_key"])}


def prewarm(keys, plan=None, verbose=True):
    """Fill the RESIDENT payload cache at startup, outside the loop's wall.

    Same work the first step would otherwise do lazily (30 x 0.21 s on AC), so
    it is weight preprocessing and belongs with [t22] wpack / [t13] cache-all /
    [devkill] prewarm.  Fail-open: on a MemoryError the cache keeps whatever it
    managed to hold and the loop falls back to the lazy repack for the rest --
    the payloads are identical either way, so nothing numeric depends on it.
    """
    plan = plan or PLAN
    if plan is None:
        return {}
    t0 = time.perf_counter()
    n = 0
    try:
        for key in keys:
            payloads(key, plan)
            n += 1
    except MemoryError:
        if verbose:
            print("[planffn] prewarm stopped at %d/%d layers (MemoryError); the "
                  "rest repacks lazily" % (n, len(keys)), flush=True)
        return {"layers": n, "seconds": round(time.perf_counter() - t0, 2),
                "resident": False}
    if verbose:
        print("[planffn] payload prewarm: %d layers resident (%.2f GB) in %.1f s "
              "(startup, outside the loop's wall); the loop no longer re-reads "
              "weights/block<N>.npz per step"
              % (n, _LRU_BYTES / 2**30, time.perf_counter() - t0), flush=True)
    return {"layers": n, "seconds": round(time.perf_counter() - t0, 2),
            "resident": True}


# ---------------------------------------------------------------------------
# 3. the product call
# ---------------------------------------------------------------------------
def ffn(dev, xn, key, tim=None, plan=None, stats=None):
    """(rows, 3840) f32 -> (rows, 3840) f32 through the plan's images.

    Drop-in for devkill_ffn.bigm_ffn at the t18_run.block_step dispatch
    boundary: same contract (the FFN output before ffn_norm2), same row
    locality, and the same section columns in the log line.
    """
    plan = plan or PLAN or resolve(np.asarray(xn).shape[0])
    b = payloads(key, plan)
    st = {}
    if _ENV["probe"] and key == "layers.0" and key not in _PROBED:
        _PROBED.add(key)
        _probe_callsite(xn, key, plan, b)
    y = S2.ffn32_run(dev, xn, key, tim=tim, b_cache=b, plan=plan, stats=st)
    if _ENV["debug"] or os.environ.get("T22_PLAN_FFN_LOG", "1") != "0":
        print("[planffn] rows=%d ndisp=%d pack=%.1f wgu=%.1f gu=%.1f "
              "swiglu=%.1f wdn=%.1f dn=%.1f read=%.1f dec=%.1f total=%.1f ms "
              "(barrier=%.1f plan=%s)"
              % (st["rows"], st["ndisp"], st["pack"], st["wgu"], st["gu"],
                 st["swiglu"], st["wdn"], st["dn"], st["read"], st["dec"],
                 st["total"], st["barrier"], st["plan"]), flush=True)
    if stats is not None:
        stats.update(st)
    return y


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _check(M):
    plan = S2.plan_for(M, prefer=None, fuse_gu=_ENV["fuse_gu"],
                       fuse_fill=_ENV["fuse_fill"])
    tags = S2.tags_for(plan)
    rec = {"M": M, "plan_key": plan["plan_key"], "strategy": plan["strategy"],
           "m": plan["m"], "kk": plan["kk"], "Mblk": plan["Mblk"],
           "ndisp": plan["ndisp"], "T": plan["T"], "nm": plan["nm"],
           "pad_rows": plan["pad_rows"], "l1_max": plan["l1_max"],
           "layer_ms": plan["layer_ms"], "per_col": plan["per_col"],
           "tags": tags, "images": {}, "gemms": {}}
    for gemm in ("gu", "dn", "proj"):
        g = plan["gemms"][gemm]
        rec["gemms"][gemm] = {k: g[k] for k in
                              ("K", "N", "n", "ntiles", "nkk", "l1",
                               "a_bytes", "b_bytes", "c_bytes")}
        xcl, inst = image_paths(tags[gemm])
        rec["images"][gemm] = {"xclbin": str(xcl) if xcl.is_file() else None,
                               "bin": str(inst) if inst.is_file() else None,
                               "xclbin_bytes": xcl.stat().st_size if xcl.is_file() else 0}
    rec["missing"] = ["%s/%s.%s" % m for m in missing_images(plan, tags)]
    rec["ok"] = not rec["missing"]
    print(json.dumps(rec, indent=1))
    return 0 if rec["ok"] else 2


def _selftest(M=1056, fuse=False):
    """Host-only checks: the perm, the packers, and B = a PERMUTATION.

    With fuse=True it also checks the FUSED gu plan: its own (smaller) n, the
    fused L1 model, the [gate|up] pair's round trip, and that the pair is still
    a permutation of the model's blocks (pair_b is a reshape, never a re-encode).
    """
    bad = []

    def chk(name, ok, detail=""):
        print("  %-52s %s %s" % (name, "OK" if ok else "FAIL", detail), flush=True)
        if not ok:
            bad.append(name)

    print("[planffn] selftest M=%d" % M, flush=True)
    plan = S2.plan_for(M, prefer=None)
    chk("plan strategy is xd", plan["strategy"] == "xd", plan["plan_key"])

    # the closed-form C perm must equal every baked file
    import glob
    for f in sorted(glob.glob(str(HERE / "c_perm_*x*.npy"))):
        mm, nn = pathlib.Path(f).stem.split("_")[-1].split("x")
        baked = np.load(f).astype(np.int64)
        chk("cperm(%s,%s) == %s" % (mm, nn, pathlib.Path(f).name),
            bool(np.array_equal(baked, S2.cperm(int(mm), int(nn)))))

    # A round trip at the plan's geometry
    rng = np.random.default_rng(7)
    A = S2._bf16(rng.standard_normal((plan["Mblk"], 3840)).astype(np.float32))
    a = PK.pack_a_kk(A, plan["m"], plan["kk"], plan["cols"], plan["cpc"])
    back = PK.unpack_a_kk(a, plan["Mblk"], 3840, plan["m"], plan["kk"],
                          plan["cols"], plan["cpc"])
    chk("pack_a_kk/unpack_a_kk round-trip (exact)",
        bool(np.array_equal(back, A)))
    chk("A payload is the plan's a_bytes",
        a.size == plan["gemms"]["gu"]["a_bytes"],
        "%d vs %d" % (a.size, plan["gemms"]["gu"]["a_bytes"]))

    # B: a permutation of the model's blocks, never a re-encode
    blocks = S2.load_blocks("layers.0")
    gu, dn = plan["gemms"]["gu"], plan["gemms"]["dn"]
    t0 = time.perf_counter()
    b_gate, b_up, b_dn = S2.pack_payloads(blocks, plan)
    dt = time.perf_counter() - t0
    chk("B payload sizes == the plan's b_bytes",
        b_gate.size == gu["b_bytes"] and b_dn.size == dn["b_bytes"],
        "%.2f s" % dt)
    if fuse:
        pf = S2.plan_for(M, prefer=None, fuse_gu=True)
        chk("the fused plan is xd and carries fuse_gu",
            pf["strategy"] == "xd" and bool(pf.get("fuse_gu")),
            pf["plan_key"])
        chk("the fused gu n is not the pure gu n (the epilogue needs 3 C "
            "tiles)", pf["gemms"]["gu"]["n"] != gu["n"],
            "%d vs %d" % (pf["gemms"]["gu"]["n"], gu["n"]))
        from webui import npu_plan as P
        chk("the fused gu fits the fused L1 model",
            P.l1_bytes_fused(pf["m"], pf["gemms"]["gu"]["n"], pf["kk"])
            <= P.L1_BYTES, "%d B" % P.l1_bytes_fused(
                pf["m"], pf["gemms"]["gu"]["n"], pf["kk"]))
        pay_f = S2.pack_payloads(blocks, pf)
        chk("the fused plan declares 3 payloads with b_gate IS b_up (one pair)",
            pay_f[0] is pay_f[1] and pay_f[0].size
            == pf["gemms"]["gu"]["b_bytes"],
            "%d B" % pay_f[0].size)
        g_f, u_f = PK.unpair_b(pay_f[0], 3840, 10240,
                               pf["gemms"]["gu"]["n"], pf["kk"])
        chk("unpair_b(pair) == the two pack_b_kk payloads (exact)",
            bool(np.array_equal(g_f, PK.pack_b_kk(
                blocks["w1"], 3840, 10240, pf["gemms"]["gu"]["n"], pf["kk"])))
            and bool(np.array_equal(u_f, PK.pack_b_kk(
                blocks["w3"], 3840, 10240, pf["gemms"]["gu"]["n"], pf["kk"]))))
        for nm, half, w in (("gate", g_f, blocks["w1"]),
                            ("up", u_f, blocks["w3"])):
            src = np.ascontiguousarray(w, np.uint8).reshape(-1, 9)
            dst = np.asarray(half, np.uint8).reshape(-1, 9)
            ok = False
            if src.shape == dst.shape:
                ok = np.array_equal(src[np.lexsort(src.T[::-1])],
                                    dst[np.lexsort(dst.T[::-1])])
            chk("the pair's %s half is a pure permutation of the model" % nm,
                ok)
    for nm, pay, w in (("w1", b_gate, blocks["w1"]), ("w3", b_up, blocks["w3"]),
                       ("w2", b_dn, blocks["w2"])):
        src = np.ascontiguousarray(w, np.uint8).reshape(-1, 9)
        dst = np.asarray(pay, np.uint8).reshape(-1, 9)
        same = False
        if src.shape == dst.shape:
            # lexsort: the LAST key is primary, so reverse the byte columns to
            # get a lexicographic order on the 9-byte records
            same = np.array_equal(src[np.lexsort(src.T[::-1])],
                                  dst[np.lexsort(dst.T[::-1])])
        chk("B(%s) is a pure permutation of the model's blocks" % nm, same)
    return bad


def register_check(M):
    """Prove the BO set t18_run.Dev will build for this plan, without a device.

    Dev.__init__ does, for every entry of BIGM_SIZES:
        b_<tag>_x = bo(a_bytes), b_<tag>_w = bo(b_bytes), b_<tag>_o = bo(c_bytes)
    on the image's own group ids 3/4/5.  That is the BO set the plan does NOT
    provide, so it is checked here against the plan's own byte counts.
    """
    import types
    fake = types.ModuleType("t18_run")
    fake.BIGM_IMGS, fake.BIGM_SIZES = {}, {}
    sys.modules["t18_run"] = fake
    try:
        plan = register(M, verbose=False)
        tags = TAGS
        bad = []
        for gemm in ("gu", "dn"):
            tag, g = tags[gemm], plan["gemms"][gemm]
            want = {"x": g["a_bytes"], "w": g["b_bytes"], "o": g["c_bytes"]}
            if tag not in fake.BIGM_IMGS:
                bad.append("BIGM_IMGS[%r] missing" % tag)
            got = fake.BIGM_SIZES.get(tag)
            if got != (want["x"], want["w"], want["o"]):
                bad.append("BIGM_SIZES[%r] = %r, want %r" % (tag, got, want))
            for s, n in want.items():
                name = "b_%s_%s" % (tag, s)
                print("  %-52s %10d B" % (name, n))
                if n <= 0:
                    bad.append("%s is %d B" % (name, n))
            xcl, inst = fake.BIGM_IMGS.get(tag, (None, None))
            if xcl is None or not pathlib.Path(xcl).is_file():
                bad.append("%s xclbin missing" % tag)
        if plan.get("fuse_gu"):
            # ONE gu dispatch and ONE W BO (the [gate|up] pair): no alt BOs
            print("  %-52s %10d B  (the [gate|up] PAIR, one BO)"
                  % ("b_%s_w" % tags["gu"], plan["gemms"]["gu"]["b_bytes"]))
        else:
            # the alt BOs s2_ffn32 allocates on the fly (gate/up double buffer)
            print("  %-52s %10d B (lazy, group 4)"
                  % ("b_%s_w+up" % tags["gu"], plan["gemms"]["gu"]["b_bytes"]))
            print("  %-52s %10d B (lazy, group 5)"
                  % ("b_%s_o+up" % tags["gu"], plan["gemms"]["gu"]["c_bytes"]))
        # the barrier image must exist on the product device
        print("  barrier image: %s (t18_run registers 'att' unconditionally)"
              % os.environ.get("T22_PLAN_FFN_BARRIER_IMG", "att"))
        for b in bad:
            print("  FAIL", b)
        print("[planffn] register-check:", "OK" if not bad else "%d FAIL" % len(bad))
        return bad
    finally:
        sys.modules.pop("t18_run", None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=None)
    ap.add_argument("--hw", type=int, nargs=2, default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--register-check", action="store_true")
    ap.add_argument("--prebuild", action="store_true")
    ap.add_argument("--layers", type=int, default=30)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fuse", action="store_true",
                    help="also check the FUSED gu plan and its [gate|up] pair")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if a.selftest:
        bad = _selftest(a.m or 1056, fuse=a.fuse)
        print("[planffn] selftest:", "OK" if not bad else "%d FAIL" % len(bad),
              flush=True)
        return 0 if not bad else 2
    M = a.m
    if M is None and a.hw:
        import npu_synth as SYN
        M = SYN.hw_to_M(a.hw[0], a.hw[1])
    if M is None:
        ap.print_help()
        return 1
    if a.check:
        return _check(M)
    if a.register_check:
        return 0 if not register_check(M) else 2
    resolve(M)
    if a.prebuild:
        _ENV["bcache"] = True
        keys = ["layers.%d" % i for i in range(a.layers)]
        print(json.dumps(prebuild(keys), indent=1), flush=True)
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
