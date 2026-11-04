# SPDX-License-Identifier: MIT
"""chain2b.py -- TWO CHAINS, ONE PROCESS, TWO HOST THREADS.  Default OFF.

THE SUBSTRATE, MEASURED (see imaggers/par/chain2b/logs/)
    * two processes: refused.  The second XRT client dies at the first kernel
      with "Failed to create context virtual (0xc01e0009)".
    * two Dev instances in one process: refused at the 27th hw context, for the
      same reason.  probe_ctx.py shows every pyxrt.kernel() call is its OWN
      hardware context with its own group-id space (the same uuid handed to two
      kernel() calls yields group_id 3538945 and 3604481), and the product
      registers EIGHTEEN per chain (att, attip, ffn + 15 big-M images).
    * TWO CHAINS ON ONE SHARED CONTEXT SET: works.  probe_shared_ctx.py builds a
      second BO table on the first chain kernels, drives both from two host
      threads, and every one of 48 drains is byte-identical to the solo arm.

SO THIS IS THE DESIGN: one registered context set (18 contexts, shared,
read-only), one Dev per chain with a COMPLETE BO table of its own, one host
thread per chain.  The device still serialises the two chains (one compute
unit); the win is that while chain A is doing its host work per layer, chain B
is dispatching, and vice versa.

HOW EACH CHAIN GETS ITS OWN MODULE
    A chain needs its own t18_run module instance: SEQ/D/tims/PROF/CACHE_BG/
    Loop/block_step/absmax memo are all module globals, and two chains sharing
    them would share the step ledger and the residency cache.  This file loads
    imaggers/image/t18_run.py a SECOND time under the name t18_run_c2 and runs
    it in its own thread.  Every module-level object is then per chain for free
    and the delivered run() is NOT refactored.

    The remaining shared state is the four objects the glue install owns, and
    those are per chain because install.py now keys them by chain
    (T22_CHAIN2B: _SCR/_Y_SCR/_ATT/STATE_CHAIN, and PD.bigm_attention dispatching
    on Loop.chain) and devkill_ffn keys its alternate W BO and its W-write pool
    by Dev.

    py310 launch_bridge.py <the usual argv>          with T22_CHAIN2B=1

TWO MODES (both behind T22_CHAIN2B=1; the product's argv decides which)
    PAIR  --batch 1 (the default): two INDEPENDENT images, chain A seed s and
          chain B seed s+1, tags <tag> and <tag>_b.  This is the measured arm of
          chain2b/REPORT.md (70.52 s/image against 87.9 = -19.8 %).

    BATCH --batch B >= 2: THE PRODUCT'S OWN BATCH PATH.  The T10 driver's batch
          loop (t18_run.py one_step: "for b in range(B)" with ONE payload/SBG)
          walks B images through the chain one after another.  With this file
          wired in, the same request runs as TWO CHAINS, each holding one image
          at a time: the batch is split into a first half (chain A, seeds
          s .. s+k_a-1) and a second half (chain B, seeds s+k_a .. s+B-1), so
          image index b still maps to seed s+b and the two halves are merged
          into the PRODUCT'S OWN artifacts, in image order:
              <tag>_latents.npy   (B, 1, 16, H/8, W/8)  == what the serial
                                  batch loop saves, image b at index b
              <tag>_ckpt.json     (the ledger the WebUI's resume/complete
                                  check reads: steps_rec, batch, lat0)
              <tag>_calls.json    (dev_ms_total, write_ms_total, dispatches,
                                  retries, steps_rec -- the job's device record)
              <tag>_traj.npz
          Each chain writes its own <tag>_a / <tag>_b artifacts FIRST (evidence,
          and the per-chain ledger), and the dispatcher merges them.  Nothing
          about the serial batch loop is refactored: B <= 1 and T22_CHAIN2B=0
          still run t18_run.run() exactly as delivered.

          WHY HALVES AND NOT WAVES OF TWO.  "Two at a time" is what the device
          admits (18 hw contexts per chain, ~26 per process, so a THIRD chain
          cannot exist).  Halves keep the two chains that DO exist busy for the
          whole window and reuse the delivered, already-byte-gated batch loop
          inside each chain instead of re-creating a Dev (and a 58-BO table) per
          wave; for B = 2, halves and waves are the same thing (one image each).

FLAGS
    T22_CHAIN2B=1        master switch (unset = the delivered single chain)
    T22_CHAIN2B_N=2      number of chains (2 only; 1 is the delivered path)
    T22_CHAIN2B_BATCH=1  0 = force the serial batch loop even for --batch >= 2
                         (the A/B knob for the batch wiring)
    T22_CHAIN2B_SEED_B=  PAIR mode only: chain B seed (default: seed + 1)
    T22_CHAIN2B_TAG_B=   PAIR mode only: chain B tag suffix (default: _b)
    T22_CHAIN2B_JSON=    where the per-chain ledger is written
    T22_CHAIN2B_THREADS= per-chain host pool
    T22_CHAIN2B_SHARE=   0; the NEGATIVE CONTROL (both chains share chain 0
                         scratch)
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import sys
import threading
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
_TLS = threading.local()
DONOR = {"dev": None}
DONOR_EVT = threading.Event()


def _chain():
    return int(getattr(_TLS, "chain", 0))


def _load_twin(name="t18_run_c2"):
    """A SECOND module instance of this directory t18_run.py."""
    import t18_run
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, t18_run.__file__)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# The two line shapes the PRODUCT parses out of this process's stdout, anchored
# at the start of the line: webui/runner.py:29 EVENT_RE (^@@WEBUI@@ {...}) and
# :30 STEP_RE (^\s*step\s+(\d+):).  Two chains share one stdout, so every other
# line is tagged [c0]/[c1] -- but a tagged event or step line would make the
# WebUI blind (no startup_done anchor, no step progress).  These two shapes are
# therefore passed through UNCHANGED, exactly as one chain prints them.
_PASSTHROUGH_RE = re.compile(r"^\s*(?:@@WEBUI@@|step\s+\d+:)")


class _ChainOut:
    """Tag every line with the chain that wrote it (one log, two chains).

    A plain attribute would be fatal here: print() goes through __getattr__ for
    everything it does not find, so a missing attribute recurses forever.  The
    real stream lives in __dict__ and is only touched through it.
    """

    def __init__(self, fh):
        self.__dict__["_fh"] = fh

    def write(self, s):
        fh = self.__dict__["_fh"]
        c = getattr(_TLS, "chain", None)
        if c is not None and s.strip() and not _PASSTHROUGH_RE.match(s):
            fh.write("[c%d] %s" % (c, s))
        else:
            fh.write(s)
        return len(s)

    def flush(self):
        self.__dict__["_fh"].flush()

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.__dict__["_fh"], name)


def _opt(argv, flag, cast=str, default=None):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def _put(argv, flag, val):
    if flag in argv:
        argv[argv.index(flag) + 1] = str(val)
    else:
        argv.extend([flag, str(val)])
    return argv


def _rewrite(argv, tag, seed, batch=None, drop_wpack_array=False):
    """ONE CHAIN's argv: same run, its own tag / seed / (batch) window.

    The driver rebuilds the initial latent from (seed + b) -- the same call the
    bridge makes when it injects lat0 -- so a chain's images are the same images
    the serial batch would have produced, in the same slots.
    """
    out = list(argv)
    _put(out, "--tag", tag)
    _put(out, "--seed", str(seed))
    if batch is not None:
        _put(out, "--batch", str(batch))
        if int(batch) >= 2:
            if "--share-weights" not in out:
                out.append("--share-weights")
        else:
            while "--share-weights" in out:
                out.remove("--share-weights")
    if "--meta" in out:
        m = pathlib.Path(out[out.index("--meta") + 1])
        _put(out, "--meta", str(m.with_name(m.stem + "_" + tag.split("_")[-1]
                                             + m.suffix)))
    if drop_wpack_array:
        # chain B never allocates the 1.99 GB W-image array: it reads chain A's
        # mapping through w_images_reuse (see _prepare).
        while "--wpack-array" in out:
            out.remove("--wpack-array")
    return out


def _ledger(tag):
    """The per-step ledger the driver already writes: <tag>_ckpt.json."""
    p = HERE / ("%s_ckpt.json" % tag)
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                          # noqa: BLE001
        return None
    steps = d.get("steps_rec") or []
    calls = d.get("calls") or []
    loop = sum(float(s.get("seconds") or 0.0) for s in steps)
    dev = sum(float((c.get("tim") or {}).get("dev") or 0.0) for c in calls)
    return {"tag": tag, "loop_s": round(loop, 3), "dev_s": round(dev / 1e3, 3),
            "dev_util": (round(dev / 1e3 / loop, 4) if loop else 0.0),
            "steps": [round(float(s.get("seconds") or 0.0), 3) for s in steps]}


def _apply_thread_cap():
    """T22_CHAIN2B_THREADS=N -- the per-chain host pool size.

    The two chains are two host threads sharing ONE 16-core box, and each pool
    is process-global (torch.set_num_threads is not per thread), so "N per
    chain" is N here.  Measured without it: two chains x ZIMAGE_CHUNK_THREADS=16
    = 32 threads on 16 cores, host seconds per image 134.5 -> 217 and the pack
    column 3.09 -> 11.46 s.  This knob exists to find the knee.

    It deliberately does NOT touch ZIMAGE_VAE_THREADS: the decode pool is
    decoupled from the loop pool (webui/config.py:93) and a loop knob must not
    move an image.
    """
    n = os.environ.get("T22_CHAIN2B_THREADS")
    if not n:
        return None
    n = int(n)
    os.environ["ZIMAGE_CHUNK_THREADS"] = str(n)
    os.environ["T22_FASTGLUE_THREADS"] = str(n)
    return n


# ---------------------------------------------------------------------------
# the shared, read-only setup: one payload, one W-image table, one twin module
# ---------------------------------------------------------------------------
def _prepare(A, argv):
    """Chain B's module instance on the SAME read-only caches.  Returns B."""
    # the glue install lives in imaggers/util/glue and is normally put on
    # sys.path by webui/chunked.install(); this driver needs it BEFORE that.
    _gl = ROOT / "imaggers" / "util" / "glue"
    if str(_gl) not in sys.path:
        sys.path.insert(0, str(_gl))
    h = _opt(argv, "--height", int, 0)
    w = _opt(argv, "--width", int, 0)
    layers = _opt(argv, "--layers", int, 30)

    # ---- ONE payload cache for both chains -------------------------------
    # The 3.71 GB B payload is the same weights for both chains, so it is
    # loaded ONCE here and both module instances adopt the same arrays.  This
    # is not an optimisation of the measurement: it is what lets two chains
    # fit at all, and the product has always treated that cache as read-only
    # and shareable (STREAM2 sec.2.1: "the 4 GB payload cache ... and every W
    # image are read-only and shared").
    # The payload's ORDER is a function of NH_BG (bridge.py:664 passes
    # --nh-bg 5 --nh-bg-core), and run() would only set it after main() parsed
    # the argv -- which happens after this function.  Resolving the cache with
    # the module default (0) builds/looks up a DIFFERENT 4.94 GB artifact whose
    # BG rows are in another order: measured, it missed d34e7a (nh_bg=5) and
    # built e156f6 (nh_bg=0).  So take the two globals from the argv first.
    A.NH_BG = _opt(argv, "--nh-bg", int, 0)
    A.NH_BG_CORE = "--nh-bg-core" in argv
    ns = type("ns", (), {})()
    ns.cache_all, ns.bo_resident, ns.layers = True, False, layers
    if h and w:
        A.configure(h, w)
    hit = None
    try:
        brc = A._brepack_prefetch(ns)
        if brc is not None:
            hit = brc.join()
    except Exception as e:                                     # noqa: BLE001
        print("[chain2b] brepack cache unavailable: %r" % (e,), flush=True)
    if hit is not None:
        A.CACHE_BG, A.CACHE_B3 = hit["bg"], hit["b3"]
        A.BG_PREPERM = True
        print("[chain2b] payload: %.2f GB mapped once, shared by both chains"
              % (hit["bytes"] / 2**30,), flush=True)
    if not A.BG_PREPERM:
        A._preload_payload(ns, at="cache")
        print("[chain2b] payload: %.2f GB read+permuted once, shared"
              % ((A.CACHE_BG.nbytes + A.CACHE_B3.nbytes) / 2**30,), flush=True)
    # prod_loop_bigm builds its per-layer table behind a module latch; warm it
    # here so neither chain can race the other into a second repack.
    try:
        import prod_loop_bigm as PB
        PB.cached_payloads("layers.0", A.CACHE_BG, A.CACHE_B3, 0)
    except Exception as e:                                     # noqa: BLE001
        print("[chain2b] cached_payloads warm failed: %r" % (e,), flush=True)

    # ---- chain B: its own module instance, its own Dev, shared contexts --
    B = _load_twin()
    B._CHAIN2B_CHILD = True
    B.CHAIN_ID = 1
    B.CACHE_BG, B.CACHE_B3, B.BG_PREPERM = A.CACHE_BG, A.CACHE_B3, A.BG_PREPERM
    A._brepack_prefetch = lambda _a: None
    B._brepack_prefetch = lambda _a: None
    A._BRC = B._BRC = None
    # The 1.99 GB packed projection images are a pure function of the weights,
    # so chain B reads chain A's mapping instead of repacking them.  bridge.py
    # (_load_wpack_file) installs w_images_reuse on the DRIVER module's Loop
    # class; the twin module has its own Loop class, so without this line chain
    # B repacks all 120 images inside its --warm (measured: it fell ~40 s behind
    # chain A before its first step, which is startup, not the loop).
    try:
        B.WPACK = A.WPACK
        if getattr(A, "WPACK_SLOTS", None):
            B.WPACK_SLOTS.clear()
            B.WPACK_SLOTS.update(A.WPACK_SLOTS)
        B.Loop.w_images = A.Loop.w_images
        # The refiner-block decode cache is a CLASS attribute, so each module's
        # Loop class has its own copy of ~2.8 GB of decoded f32 blocks.  The
        # weights are the same tensor for both chains and read-only, so chain B
        # adopts chain A's dicts instead of decoding them a second time -- this
        # is a memory decision as much as a startup one (two full copies do not
        # fit next to the 4.94 GB payload mapping).
        B.Loop._rcache = A.Loop._rcache
        B.Loop._pcache = A.Loop._pcache
        print("[chain2b] chain B reuses chain A's 1.99 GB packed W images "
              "(w_images_reuse, %d slots)" % (len(B.WPACK_SLOTS),), flush=True)
    except Exception as e:                                     # noqa: BLE001
        print("[chain2b] W-image share skipped: %r" % (e,), flush=True)
    print("[chain2b] chain B = %s (second module instance of %s)"
          % (B.__name__, A.__file__), flush=True)

    # Warm the read-only oracle module ONCE, single-threaded.  t3_real._oracle
    # used to publish dit_block before executing it, so whichever chain lost the
    # race got an empty module (AttributeError: dit_block has no attribute
    # 'Geometry', measured).  t3_real is fixed, and this makes the two chains
    # start from a loaded module regardless.
    try:
        import t3_real
        t3_real._oracle()
        print("[chain2b] oracle module warmed once (dit_block.Geometry=%s)"
              % (hasattr(__import__("sys").modules.get("dit_block"), "Geometry"),),
              flush=True)
    except Exception as e:                                     # noqa: BLE001
        print("[chain2b] oracle warm failed: %r" % (e,), flush=True)
    return B


def _launch(A, B, argv_a, argv_b):
    """Two chains, two host threads, ONE shared registered context set.

    Chain A builds the 18 hw contexts; chain B adopts them and builds only its
    own BO table.  Both threads run their own module's main(), so all of
    SEQ/D/tims/PROF/CACHE_BG/Loop/block_step are per chain by construction.
    """
    real_a, real_b = A.Dev, B.Dev

    def _capture(*a, **k):
        d = real_a(*a, **k)
        DONOR["dev"] = d
        DONOR_EVT.set()
        return d

    def _share(*a, **k):
        DONOR_EVT.wait(600)
        return real_b(*a, share=DONOR["dev"], **k)

    A.Dev, B.Dev = _capture, _share

    import install as GLUE
    res, err, t_chain = {}, {}, {}

    def work(mod, av, chain):
        _TLS.chain = chain
        GLUE.set_chain(chain)
        t_chain[chain] = [time.time(), None]
        try:
            res[chain] = mod.main(av)
        except BaseException as e:                             # noqa: BLE001
            import traceback
            traceback.print_exc()
            err[chain] = repr(e)
            res[chain] = 99
        finally:
            t_chain[chain][1] = time.time()

    old_out = sys.stdout
    sys.stdout = _ChainOut(old_out)
    tA = threading.Thread(target=work, args=(A, argv_a, 0), name="chainA")
    tA.start()
    if not DONOR_EVT.wait(600):
        print("[chain2b] REFUSAL: chain A never built a Dev", flush=True)
        sys.stdout = old_out
        return res, {"0": "no Dev from chain A"}, t_chain
    tB = threading.Thread(target=work, args=(B, argv_b, 1), name="chainB")
    tB.start()
    tA.join()
    tB.join()
    sys.stdout = old_out
    return res, err, t_chain


def _write_ledger(out, name="last_pair.json"):
    pj = os.environ.get("T22_CHAIN2B_JSON", "") or str(
        ROOT / "imaggers" / "par" / "chain2b" / "logs" / name)
    try:
        pathlib.Path(pj).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(pj).write_text(json.dumps(out, indent=1), encoding="utf-8")
    except Exception:                                          # noqa: BLE001
        pass
    return pj


# ---------------------------------------------------------------------------
# PAIR mode: two independent images (the measured arm of chain2b/REPORT.md)
# ---------------------------------------------------------------------------
def run_main(argv=None):
    cap = _apply_thread_cap()
    if cap:
        print("[chain2b] per-chain host pool capped at %d threads "
              "(ZIMAGE_CHUNK_THREADS=%s T22_FASTGLUE_THREADS=%s; "
              "ZIMAGE_VAE_THREADS left at %r)"
              % (cap, os.environ["ZIMAGE_CHUNK_THREADS"],
                 os.environ["T22_FASTGLUE_THREADS"],
                 os.environ.get("ZIMAGE_VAE_THREADS")), flush=True)
    n = int(os.environ.get("T22_CHAIN2B_N", "2") or 2)
    if argv is None:
        argv = sys.argv[1:]
    argv = [str(a) for a in argv]
    import t18_run as A
    if n < 2:
        A._CHAIN2B_CHILD = True
        return A.main(argv)
    batch = _opt(argv, "--batch", int, 1)
    if batch >= 2:
        if os.environ.get("T22_CHAIN2B_BATCH", "1") not in ("0", "false",
                                                            "False", "off"):
            return _run_batch(A, argv, batch)
        # the A/B knob is OFF: this process is the delivered single chain, and
        # the batch walks its images serially exactly as it does without the
        # flag.  It must NOT fall through to _run_pair (that would put the
        # whole batch on BOTH chains).
        A._CHAIN2B_CHILD = True
        return A.main(argv)
    return _run_pair(A, argv)


def _run_pair(A, argv):
    # THIS module is now a chain, not the dispatcher: without this the inner
    # A.main() would route straight back here.
    A._CHAIN2B_CHILD = True
    t_start = time.time()
    tag_a = _opt(argv, "--tag", str, "t4a")
    seed_a = _opt(argv, "--seed", int, 0)
    seed_b = int(os.environ.get("T22_CHAIN2B_SEED_B", seed_a + 1))
    tag_b = tag_a + os.environ.get("T22_CHAIN2B_TAG_B", "_b")
    argv_b = _rewrite(argv, tag_b, seed_b, drop_wpack_array=True)
    print("[chain2b] TWO CHAINS IN ONE PROCESS: A tag=%r seed=%d, B tag=%r"
          " seed=%d" % (tag_a, seed_a, tag_b, seed_b), flush=True)

    B = _prepare(A, argv)
    res, err, t_chain = _launch(A, B, argv, argv_b)

    la, lb = _ledger(tag_a), _ledger(tag_b)
    loops = [x["loop_s"] for x in (la, lb) if x]
    devs = [x["dev_s"] for x in (la, lb) if x]
    out = {"tags": [tag_a, tag_b], "seeds": [seed_a, seed_b],
           "rc": {str(k): res.get(k) for k in res}, "err": err,
           "pair_wall_s": round(time.time() - t_start, 2),
           "chain_wall_s": {str(k): [round(v[0], 2), round(v[1], 2)]
                            for k, v in t_chain.items()},
           "chains": {"a": la, "b": lb}}
    if len(loops) == 2:
        out["max_loop_s"] = round(max(loops), 3)
        out["s_per_image"] = round(max(loops) / 2.0, 3)
        out["dev_util"] = round(sum(devs) / max(loops), 4)
        out["loop_if_serial"] = round(sum(loops), 3)
    pj = _write_ledger(out)
    print("[chain2b] ===== TWO-CHAIN RESULT =====", flush=True)
    for k, lab in ((la, "A"), (lb, "B")):
        if k:
            print("[chain2b] chain %s tag=%-22s loop %7.2f s  dev %6.2f s  "
                  "dev %% %5.1f" % (lab, k["tag"], k["loop_s"], k["dev_s"],
                                    100.0 * k["dev_util"]), flush=True)
    if len(loops) == 2:
        print("[chain2b] 2 images in %.2f s = %.2f s/image; device busy "
              "%.1f%% of the pair window (serial single-chain would be %.1f s)"
              % (max(loops), max(loops) / 2.0, 100.0 * out["dev_util"],
                 sum(loops)), flush=True)
    print("[chain2b] ledger -> %s" % (pj,), flush=True)
    return 0 if all(v == 0 for v in res.values()) and not err else 5


# ---------------------------------------------------------------------------
# BATCH mode: the product's --batch B, two chains, one image each at a time
# ---------------------------------------------------------------------------
def _batch_split(batch):
    """First half -> chain A, second half -> chain B (never more than two)."""
    k_a = (int(batch) + 1) // 2
    return k_a, int(batch) - k_a


def _seed_chain_ckpt(src_tag, dst_tag, off, k, steps, layers):
    """Give a chain the EXACT initial latents the serial batch would use.

    webui/bridge.py injects the initial latent for shapes other than 128x128
    through the driver's --resume seam: <tag>_ckpt.json carries lat0 for every
    image of the batch and <tag>_ckpt_lat.npy the stacked array.  The serial
    driver then RESUMES from it.  A chain runs under its own tag, so it would
    find no ckpt and rebuild the same latents from np.random.default_rng(seed+b)
    -- the same call, the same bytes.  This makes it the same FILE as well: the
    chain's first k images are the product's lat0[off:off+k], byte for byte, so
    "image b is the image the serial batch produced" does not depend on two RNG
    call sites agreeing.  Missing/foreign ckpt (the 128x128 delivered path has
    none) simply means the driver's own default, which is the delivered path.
    """
    src_js = HERE / ("%s_ckpt.json" % src_tag)
    src_lat = HERE / ("%s_ckpt_lat.npy" % src_tag)
    dst_js = HERE / ("%s_ckpt.json" % dst_tag)
    dst_lat = HERE / ("%s_ckpt_lat.npy" % dst_tag)
    # a chain tag is scratch: never resume from a previous run of the same tag
    for p in (dst_js, dst_lat):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
    if not (src_js.is_file() and src_lat.is_file()):
        return None
    try:
        ck = json.loads(src_js.read_text(encoding="utf-8"))
        lat0 = ck.get("lat0") or []
        if int(ck.get("next_step") or 0) != 0 or len(lat0) < off + k:
            return None
        arr = np.load(src_lat)
        if arr.shape[0] < off + k:
            return None
        np.save(dst_lat, arr[off:off + k])
        dst_js.write_text(json.dumps(
            {"next_step": 0, "steps": int(steps), "layers": int(layers),
             "batch": int(k), "lat0": [np.asarray(z, np.float32).ravel().tolist()
                                       for z in lat0[off:off + k]],
             "calls": [], "steps_rec": []}), encoding="utf-8")
        return str(dst_lat)
    except Exception as e:                                     # noqa: BLE001
        print("[chain2b] per-chain ckpt seed failed for %r: %r"
              % (dst_tag, e), flush=True)
        return None


def _merge_steps(sa, sb, batch):
    """Per-step ledger of the batch from its two halves (concurrent => max)."""
    out = []
    for i in range(max(len(sa), len(sb))):
        a = sa[i] if i < len(sa) else {}
        b = sb[i] if i < len(sb) else {}
        rec = dict(a or b)
        w = round(max(float(a.get("seconds") or 0.0),
                      float(b.get("seconds") or 0.0)), 4)
        rec["step"] = int(a.get("step", b.get("step", i)))
        rec["seconds"] = w
        rec["seconds_per_image"] = round(w / batch, 4)
        rec["batch"] = int(batch)
        rec["pred_absmax"] = max(float(a.get("pred_absmax") or 0.0),
                                 float(b.get("pred_absmax") or 0.0))
        la = list(a.get("lat_absmax_by_image") or [])
        lb = list(b.get("lat_absmax_by_image") or [])
        rec["lat_absmax_by_image"] = [float(x) for x in la + lb]
        rec["lat_absmax"] = max(rec["lat_absmax_by_image"] or [0.0])
        ta = a.get("tim") or {}
        tb = b.get("tim") or {}
        rec["tim"] = {k: float(ta.get(k, 0.0)) + float(tb.get(k, 0.0))
                      for k in set(ta) | set(tb)}
        out.append(rec)
    return out


def _merge_calls(ca, cb, batch):
    """Per-step call records (the 'calls' list the driver writes each step)."""
    out = []
    for i in range(max(len(ca), len(cb))):
        a = ca[i] if i < len(ca) else {}
        b = cb[i] if i < len(cb) else {}
        rec = dict(a or b)
        w = round(max(float(a.get("wall_s") or 0.0), float(b.get("wall_s") or 0.0)), 4)
        rec["wall_s"] = w
        rec["wall_s_per_image"] = round(w / batch, 4)
        rec["pred_absmax"] = max(float(a.get("pred_absmax") or 0.0),
                                 float(b.get("pred_absmax") or 0.0))
        rec["pred_absmax_by_image"] = (
            list(a.get("pred_absmax_by_image") or [])
            + list(b.get("pred_absmax_by_image") or []))
        rec["tim_by_image"] = (list(a.get("tim_by_image") or [])
                               + list(b.get("tim_by_image") or []))
        ta = a.get("tim") or {}
        tb = b.get("tim") or {}
        rec["tim"] = {k: float(ta.get(k, 0.0)) + float(tb.get(k, 0.0))
                      for k in set(ta) | set(tb)}
        rec["layers"] = (list(a.get("layers") or [])
                         + list(b.get("layers") or []))
        out.append(rec)
    return out


def _read_json(p):
    try:
        return json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
    except Exception:                                          # noqa: BLE001
        return None


def _run_batch(A, argv, batch):
    """--batch B >= 2: two chains, halves of the batch, merged into <tag>."""
    A._CHAIN2B_CHILD = True
    t_start = time.time()
    tag = _opt(argv, "--tag", str, "t4a")
    seed = _opt(argv, "--seed", int, 0)
    steps = _opt(argv, "--steps", int, 8)
    layers = _opt(argv, "--layers", int, 30)
    k_a, k_b = _batch_split(batch)
    tag_a, tag_b = tag + "_a", tag + "_b"
    seed_b = seed + k_a
    argv_a = _rewrite(argv, tag_a, seed, batch=k_a)
    argv_b = _rewrite(argv, tag_b, seed_b, batch=k_b, drop_wpack_array=True)
    print("[chain2b] BATCH %d THROUGH TWO CHAINS (one process, two host "
          "threads): A tag=%r images 0..%d seed %d..%d (%d image(s)), "
          "B tag=%r images %d..%d seed %d..%d (%d image(s)); the product's "
          "serial loop would walk all %d through one chain"
          % (batch, tag_a, k_a - 1, seed, seed + k_a - 1, k_a,
             tag_b, k_a, batch - 1, seed_b, seed_b + k_b - 1, k_b, batch),
          flush=True)
    seeded = (_seed_chain_ckpt(tag, tag_a, 0, k_a, steps, layers),
              _seed_chain_ckpt(tag, tag_b, k_a, k_b, steps, layers))
    if any(seeded):
        print("[chain2b] per-chain initial latents = the product ckpt's "
              "lat0[0:%d] / lat0[%d:%d] (byte-identical start to the serial "
              "batch)" % (k_a, k_a, batch), flush=True)

    B = _prepare(A, argv)
    res, err, t_chain = _launch(A, B, argv_a, argv_b)

    la, lb = _ledger(tag_a), _ledger(tag_b)
    loops = [x["loop_s"] for x in (la, lb) if x]
    devs = [x["dev_s"] for x in (la, lb) if x]
    out = {"mode": "batch", "batch": int(batch), "tags": [tag_a, tag_b],
           "seeds": [seed, seed_b], "split": [k_a, k_b], "product_tag": tag,
           "rc": {str(k): res.get(k) for k in res}, "err": err,
           "pair_wall_s": round(time.time() - t_start, 2),
           "chain_wall_s": {str(k): [round(v[0], 2), round(v[1], 2)]
                            for k, v in t_chain.items()},
           "chains": {"a": la, "b": lb}}
    if len(loops) == 2:
        out["window_s"] = round(max(loops), 3)
        out["s_per_image"] = round(max(loops) / float(batch), 3)
        out["dev_util"] = round(sum(devs) / max(loops), 4)
        out["loop_if_serial"] = round(sum(loops), 3)

    failed = bool(err) or any(v != 0 for v in res.values())
    if not la or not lb:
        out["merged"] = False
        pj = _write_ledger(out, "%s.json" % tag)
        print("[chain2b] BATCH %d FAILED before both chains produced a ledger: "
              "rc=%s err=%r (no batch artifacts written; the product's <tag>_* "
              "files are absent, so the job fails loudly instead of shipping "
              "half a batch)" % (batch, res, err), flush=True)
        print("[chain2b] ledger -> %s" % (pj,), flush=True)
        return 5
    try:
        merged = _merge_artifacts(tag, tag_a, tag_b, batch, seed, steps, layers,
                                  k_a, k_b)
    except Exception as e:                                     # noqa: BLE001
        import traceback
        traceback.print_exc()
        out["merged"] = False
        out["merge_err"] = repr(e)
        _write_ledger(out, "%s.json" % tag)
        return 5
    out.update(merged["ledger"])
    out["merged"] = True
    pj = _write_ledger(out, "%s.json" % tag)

    dev_sum = (la["dev_s"] if la else 0.0) + (lb["dev_s"] if lb else 0.0)
    write_sum = merged["write_s"]
    total = merged["window_s"]
    print("[chain2b] ===== TWO-CHAIN BATCH RESULT =====", flush=True)
    for k, lab in ((la, "A"), (lb, "B")):
        print("[chain2b] chain %s tag=%-22s loop %7.2f s  dev %6.2f s  "
              "dev %% %5.1f" % (lab, k["tag"], k["loop_s"], k["dev_s"],
                                100.0 * k["dev_util"]), flush=True)
    print("[chain2b] batch %d (%d + %d images) in %.2f s = %.2f s/image; "
          "device service demand %.1f s = %.1f%% of the window; the two chains "
          "walked %.1f s of loop"
          % (batch, k_a, k_b, total, total / float(batch), dev_sum,
             100.0 * dev_sum / total, merged["ledger"]["loop_if_serial"]),
          flush=True)
    print("[chain2b] latents -> %s  (batch %d, image order 0..%d)"
          % (HERE / ("%s_latents.npy" % tag), batch, batch - 1), flush=True)
    print("[chain2b] ledger -> %s" % (pj,), flush=True)
    return 0 if not failed else 5


def _merge_artifacts(tag, tag_a, tag_b, batch, seed, steps, layers, k_a, k_b):
    """Write the PRODUCT'S batch artifacts from the two chains' own files.

    Same names, same place, same order as the serial batch loop:
        <tag>_latents.npy  (B,1,16,H/8,W/8)   image b at index b
        <tag>_ckpt.json    the ledger webui/runner.py:1473 checks
        <tag>_calls.json   the record runner.py:1596 reads its device numbers
        <tag>_traj.npz
    """
    # A chain that drove ONE image saved exactly what the driver saves for
    # --batch 1: lat[0], i.e. (1,16,H/8,W/8) with NO batch axis.  The serial
    # batch path saves np.stack(lat) -> (B,1,16,H/8,W/8).  Promote the single
    # image back to (1,1,16,H/8,W/8) so the merged FILE has the serial path's
    # shape, not just its bytes (the VAE accepts both, a byte comparison does
    # not, and "same output" means the same array).
    def _stacked(path):
        a = np.load(path)
        return a[None] if a.ndim == 4 else a

    lat_a = _stacked(HERE / ("%s_latents.npy" % tag_a))
    lat_b = _stacked(HERE / ("%s_latents.npy" % tag_b))
    lat = np.concatenate([lat_a, lat_b], axis=0)
    if int(lat.shape[0]) != int(batch):
        raise RuntimeError("merged latents have %d images, expected %d"
                           % (lat.shape[0], batch))
    np.save(HERE / ("%s_latents.npy" % tag), lat)

    za = np.load(HERE / ("%s_traj.npz" % tag_a))
    zb = np.load(HERE / ("%s_traj.npz" % tag_b))
    lat0 = np.concatenate([za["latents0"], zb["latents0"]], axis=0)
    np.savez(HERE / ("%s_traj.npz" % tag),
             latents0=lat0, latents=lat,
             timesteps=za["timesteps"], sigmas=za["sigmas"])

    ca = _read_json(HERE / ("%s_calls.json" % tag_a)) or {}
    cb = _read_json(HERE / ("%s_calls.json" % tag_b)) or {}
    ka = _read_json(HERE / ("%s_ckpt.json" % tag_a)) or {}
    kb = _read_json(HERE / ("%s_ckpt.json" % tag_b)) or {}
    steps_a = ca.get("steps_rec") or ka.get("steps_rec") or []
    steps_b = cb.get("steps_rec") or kb.get("steps_rec") or []
    calls_a = ca.get("calls") or ka.get("calls") or []
    calls_b = cb.get("calls") or kb.get("calls") or []
    merged_steps = _merge_steps(steps_a, steps_b, batch)
    merged_calls = _merge_calls(calls_a, calls_b, batch)

    window = max(sum(float(s.get("seconds") or 0.0) for s in steps_a),
                 sum(float(s.get("seconds") or 0.0) for s in steps_b))

    def _tim_sum(ss, key):
        # The per-step ledger is the PRODUCT'S OWN device column: at 512^2 the
        # fast/big-M path does not go through Dev.dispatch, so calls.json's
        # dev_ms_total is 0 there in every historical arm (c2b_base1 included)
        # while the step lines carry the real figure.  Sum the same numbers the
        # product prints.
        return sum(float((s.get("tim") or {}).get(key) or 0.0) for s in ss)

    dev_ms = _tim_sum(steps_a, "dev") + _tim_sum(steps_b, "dev")
    write_ms = _tim_sum(steps_a, "write") + _tim_sum(steps_b, "write")
    if not dev_ms:
        dev_ms = float(ca.get("dev_ms_total") or 0.0) + float(
            cb.get("dev_ms_total") or 0.0)
    if not write_ms:
        write_ms = float(ca.get("write_ms_total") or 0.0) + float(
            cb.get("write_ms_total") or 0.0)

    def _sum_int(k):
        return int((ca.get(k) or 0)) + int((cb.get(k) or 0))

    rep = {
        "seed": int(seed), "steps": int(steps), "layers": int(layers),
        "batch": int(batch),
        "images_per_minute": (60.0 * batch / window) if window > 0 else 0.0,
        "seconds_per_image": (window / float(batch)) if batch else 0.0,
        "dispatches": _sum_int("dispatches"), "retries": _sum_int("retries"),
        "barrier_calls": _sum_int("barrier_calls"),
        "barrier_ms": float(ca.get("barrier_ms") or 0.0)
                      + float(cb.get("barrier_ms") or 0.0),
        "images": ca.get("images") or cb.get("images") or [],
        "qkv": bool(ca.get("qkv", cb.get("qkv", False))),
        "p10": bool(ca.get("p10", cb.get("p10", False))),
        "dev_ms_total": dev_ms,
        "dev_ms_by_image": _sum_maps(ca.get("dev_ms_by_image"),
                                     cb.get("dev_ms_by_image")),
        "write_ms_total": write_ms,
        "write_ms_by_image": _sum_maps(ca.get("write_ms_by_image"),
                                       cb.get("write_ms_by_image")),
        "wall_s_total": round(window, 3),
        "npu_share_dispatch_only": (dev_ms / (1e3 * window)) if window else 0.0,
        "npu_share_dispatch_plus_staging": ((dev_ms + write_ms) / (1e3 * window)
                                            if window else 0.0),
        "steps_rec": merged_steps, "calls": merged_calls,
        "driver": "driver376 + chain2b batch: two chains on one shared context "
                  "set, halves of the batch, one image each at a time",
        "chain2b": {"mode": "batch", "batch": int(batch),
                    "split": [int(k_a), int(k_b)],
                    "tags": [tag_a, tag_b], "window_s": round(window, 3),
                    "dev_s": round(dev_ms / 1e3, 3),
                    "s_per_image": round(window / float(batch), 3)},
    }
    lat0_list = list(ka.get("lat0") or []) + list(kb.get("lat0") or [])
    if len(lat0_list) != int(batch):
        # fall back to the traj's own latents0 (always written by the driver)
        lat0_list = [np.asarray(z, np.float32).ravel().tolist()
                     for z in lat0]
    ck = {"next_step": int(steps), "steps": int(steps), "layers": int(layers),
          "batch": int(batch), "lat0": lat0_list, "calls": merged_calls,
          "steps_rec": merged_steps}
    (HERE / ("%s_ckpt.json" % tag)).write_text(json.dumps(ck), encoding="utf-8")
    (HERE / ("%s_calls.json" % tag)).write_text(json.dumps(rep, indent=1),
                                                encoding="utf-8")
    return {"ledger": {"window_s": round(window, 3),
                       "s_per_image": round(window / float(batch), 3),
                       "dev_s": round(dev_ms / 1e3, 3),
                       "write_s": round(write_ms / 1e3, 3),
                       "dev_util": (dev_ms / (1e3 * window)) if window else 0.0,
                       "loop_if_serial": round(
                           sum(float(s.get("seconds") or 0.0) for s in steps_a)
                           + sum(float(s.get("seconds") or 0.0)
                                 for s in steps_b), 3)},
            "latents": str(HERE / ("%s_latents.npy" % tag)),
            "dev_s": dev_ms / 1e3, "write_s": write_ms / 1e3,
            "window_s": window}


def _sum_maps(a, b):
    out = {}
    for m in (a or {}, b or {}):
        for k, v in m.items():
            out[k] = float(out.get(k, 0.0)) + float(v)
    return out
