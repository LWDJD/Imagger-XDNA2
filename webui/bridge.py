# SPDX-License-Identifier: MIT
"""py310 device bridge: t13_run.py + a real caption, without editing t13_run.py.

Run by the WebUI server.  It imports the delivered t13_run.py, patches exactly
two read-only seams in t3_real:

  cap_embedding(H)      -> the 32 x 3840 rows built from the encoded prompt
  load_schedule(args)   -> the schedule the WebUI derived/cached

and points t13_run.T4_OUT at the verified per-resolution cache.  The driver
itself (and every delivered xclbin) is untouched.  The child prints
@@WEBUI@@ {...} events so the HTTP server can show live stages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys
import time
import traceback

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
NPU = HERE.parent
IMAGE_DIR = NPU / "imaggers" / "image"
sys.path.insert(0, str(NPU))
sys.path.insert(0, str(NPU / "imaggers" / "image"))

import t13_run  # noqa: E402
import t3_real  # noqa: E402
from webui import shapes  # noqa: E402
from webui import config as _config  # noqa: E402
from webui.cap import build_cap_embedding, pe_sha256, validate_pe  # noqa: E402

_T0 = time.time()
_PE = None
_CAP = None
_STARTUP = False
_KEEP_DEVS = []
DRV = t13_run          # chosen per shape in main() (T18)

# ---------------------------------------------------------------------------
# T22 qkv (campaign t22_qkv_prod): the driver's FUSED q/k/v projection image.
# **ON by default since 2026-10-20** (owner decision; config.T22_QKV_DEFAULT
# carries the evidence and the rollback knobs -- commits 9e829a7 / 94501c6).
# Two doors, both forwarded to the driver through driver_argv():
#   --qkv / --no-qkv            one dispatch per 96-row block instead of three
#                               projection launches (always sent explicitly)
#   T22_QKV=0                   the env kill switch for the whole server
#   --xpack-once                the host-side X pack once per block (still opt-in)
# A missing fused image is NOT fatal: the bridge says so once and runs the
# delivered projection path, because *.xclbin is not in git and a fresh
# checkout must still produce images.
QKV_ENV = os.environ.get("T22_QKV", "1") not in ("0", "false", "False", "off",
                                                  "no", "")
XPACK_ENV = os.environ.get("T22_XPACK_ONCE", "0") not in ("0", "false",
                                                          "False", "off", "no", "")
QKV_DEFAULT_OUT = NPU / "imaggers" / "block" / "qkv" / "out"
QKV_DEFAULT_TAG = "qkv_par15m"

# ---------------------------------------------------------------------------
# T22 p10 (campaign t22_p15prod): the o PROJECTION on its own multi-stream
# image -- the one dispatch that is still on the 5-stream delivered t7_p5 once
# qkv is on.  **ON by default since 2026-10-20** (owner decision on
# PROD-REPORT.md; config.T22_P10_DEFAULT is the service copy of this switch),
# ORTHOGONAL to --qkv: they replace different dispatches.
#   --p10 / --p15 / --no-p10    forwarded to the driver, ALWAYS explicitly
#   T22_P10=0                   the env kill switch for a whole server
# A missing image is NOT fatal: one line and the delivered o path, because
# *.xclbin is not in git and a fresh checkout must still produce images.
P10_ENV = os.environ.get("T22_P10", "1") not in ("0", "false", "False", "off",
                                                 "no", "")
P10_DEFAULT_OUT = NPU / "imaggers" / "attn_impl" / "proj" / "out"
P10_DEFAULT_TAG = "p10"

# ---------------------------------------------------------------------------
# T22 glue (campaign t22_glue): the two HOST fast paths.  They change no image
# and no arithmetic -- only how the host builds/reads the same bytes -- so they
# are forwarded as explicit driver flags exactly like qkv/p10:
#   --t22-fast-glue   U1: rms_into + one-row SBG + rare SBG assert
#                     (driver env T22_FAST_GLUE=1; -1.68 s/图 in t22_util,
#                      latents byte-identical, unit gates 204/204)
#   --t22-glue2       glue-2: copy-free drain/pack path (T22_GLUE2=1)
# **DEFAULT OFF** until the product four-piece (image side-by-side, closure
# panel, implementation statement, batch gate) has passed on a real job; the
# one-line flip is FAST_GLUE_DEFAULT below.  A job never sets these, so a
# service that has not been taught them keeps the delivered path.
# 2026-10-21 four-piece gates all green (10/10: 8 E2E jobs latents 9f86aee9, four
# shapes A/B byte-identical incl. short-block fallback, batch img0 == single PNG
# 104ff564, closure +-0.11 s, run-to-run, t3d_baseline UNCHANGED, ledger residual
# +0.000); combined A/B/A gain -1.2~-1.7 s.  ROLLBACK: T22_FAST_GLUE=0 etc.
FAST_GLUE_DEFAULT = os.environ.get("T22_FAST_GLUE", "1") not in (
    "0", "false", "False", "off", "no", "")
GLUE2_DEFAULT = os.environ.get("T22_GLUE2", "1") not in (
    "0", "false", "False", "off", "no", "")



def _load_driver(height, width, force=""):
    """T18: t18_run is the product driver for EVERY shape.

    t18_run.py is t13_run.py with SEQ/perm/BO sizes parameterized and the
    attention run in row blocks (projection: 96-row blocks, ip: query blocks
    with ND = SEQ keys).  At SEQ=96 it is the delivered five dispatches and it
    is BYTE-IDENTICAL to the production 128x128 latents (sha16
    6004035d0cb97250, twice: --driver t18_run and through this bridge).

    Why it replaced t13_run for M=96 too: the delivered dispatch appends the
    output BO a second time (6 declared args -> 7 BOs); the M=128 FFN image
    aborts on that extra BO (0xC0000005 inside the ERT launch) and, in the
    current environment, the delivered t13_run path itself dies at the first
    dispatch even with a fresh numba cache, while t18_run (exactly the declared
    args) runs.  --driver t13_run is kept for A/B.
    """
    m = shapes.m_from_hw(int(height), int(width))
    name = force or "t18_run"
    if name == "t13_run":
        return t13_run
    if m < 96:
        raise SystemExit("[bridge] M=%d has no driver" % m)
    import t18_run  # noqa: WPS433 (deliberate lazy import)
    return t18_run


def _install_dev_keep(drv):
    """Keep pyxrt objects alive past run()'s return; bridge hard-exits after."""

    class _DevKeep(drv.Dev):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            _KEEP_DEVS.append(self)

    drv.Dev = _DevKeep


def emit(event, **kw):
    kw["event"] = event
    kw["t"] = time.time()
    # Same monotonic clock stream_process measures its wall with, so
    # startup + loop + tail == process wall without wall-clock drift (P0-4).
    kw["tp"] = time.perf_counter()
    print("@@WEBUI@@ " + json.dumps(kw, ensure_ascii=False), flush=True)


def _install_token_layout_fix():
    """T16 P0-1: the token-inner layout of a 2x2 patch.

    The checkpoint's `all_x_embedder.2-1` was trained on
    `ZImageTransformer2DModel.patchify_and_embed`'s flattening:

        x.view(C, F, pF, H, pH, W, pW).permute(1,3,5,2,4,6,0)
        -> token inner index ((pH*2 + pW) * C + c)      [PATCH-MAJOR]

    The delivered `t3_real.patchify` used

        lat.reshape(b,c,gh,2,gw,2).transpose(0,2,4,1,3,5)
        -> token inner index (c*4 + pH*2 + pW)          [CHANNEL-MAJOR]

    Same 64 numbers per token, different order into the same weight matrix.
    Every block still runs the verified diffusers arithmetic (the self-check
    compared NPU to this same reference, so it could never see it), but the
    conditioning is scrambled: one official-trace step moves from
    corr 0.432 / l2rel 109.4 % to corr 0.9986 / l2rel 5.3 % once the order is
    fixed.  `unpatchify` was already patch-major, so the model read a
    different layout on the way in than it wrote on the way out.

    Patch the seam instead of editing the delivered t3_real.py (read-only).
    `t3_real.x_embedding` -> `patchify` is a module-global lookup, and
    t13_run calls `t3_real.x_embedding`, so one patch covers host and device.
    """
    import t3_real

    def patchify_official(lat):
        b, c, h, w = lat.shape
        gh, gw = h // 2, w // 2
        x = lat.reshape(b, c, gh, 2, gw, 2)
        x = x.transpose(0, 2, 4, 3, 5, 1)      # (b, gh, gw, pH, pW, c)
        return x.reshape(b, gh * gw, c * 4)

    if getattr(t3_real, "_T16_TOKEN_LAYOUT_FIXED", False):
        return
    t3_real.patchify = patchify_official
    t3_real._T16_TOKEN_LAYOUT_FIXED = True
    emit("token_layout_fix", patchify="patch-major(pH,pW,C)",
         reason="checkpoint all_x_embedder expects zimage_pipeline.patchify order")


def _install(pe, timesteps, sigmas):
    global _PE
    _PE = validate_pe(pe)

    def cap_embedding(H):
        global _CAP, _STARTUP
        if _CAP is None:
            _CAP = build_cap_embedding(_PE, H)
            emit("cap_built", tokens=int(_PE.shape[1]),
                 shape=list(_CAP.shape),
                 sha256=hashlib.sha256(_CAP.tobytes()).hexdigest())
        if not _STARTUP:
            _STARTUP = True
            emit("startup_done")
        return _CAP

    def load_schedule(_args):
        return timesteps.copy(), sigmas.copy()

    t3_real.cap_embedding = cap_embedding
    t3_real.load_schedule = load_schedule


def _install_shape_geometry(height, width, tag, steps, seed, batch):
    """T16 P0-5 slice 1: same-M shapes (64x256 == 128x128's M=96).

    The delivered images (t7_p5 / t7_ip / t11_nh5) are compiled for M=96, i.e.
    32 caption rows + 64 image tokens.  64x256 has the same 64 image tokens
    (4x16 patch grid) but a different 2D geometry, so the only things that must
    change are host-side:

      * t3_real.IMG_GRID          -> the RoPE image ids (z=33, y, x)
      * t3_real.unpatchify        -> the 4x16 -> 8x32 latent grid
      * the initial latent shape  -> injected through t13_run's --resume ckpt
                                     (the delivered driver hardcodes 1x16x16x16)

    Nothing else: M, the row order, the permutations, SBG/walk and the weights
    are unchanged, which is exactly why this is the cheapest new shape.
    """
    m = shapes.m_from_hw(int(height), int(width))
    if (int(height), int(width)) == (128, 128):
        return None
    gh, gw = int(height) // 16, int(width) // 16
    tokens = gh * gw
    if tokens != 64 and DRV is t13_run:
        raise SystemExit(f"[bridge] shape {height}x{width} has {tokens} image "
                         f"tokens (M={m}); the delivered driver is M=96 only")
    t3_real.IMG_GRID = (1, gh, gw)
    orig_unpatch = t3_real.unpatchify

    def unpatchify_shape(tok, chans, _gh=None, _gw=None):
        return orig_unpatch(tok, chans, gh, gw)

    t3_real.unpatchify = unpatchify_shape

    # Inject the initial latent through the delivered --resume seam.
    lat = [np.random.default_rng(int(seed) + b).standard_normal(
        (1, 16, int(height) // 8, int(width) // 8)).astype(np.float32)
        for b in range(int(batch))]
    ck_lat = IMAGE_DIR / f"{tag}_ckpt_lat.npy"
    ck_js = IMAGE_DIR / f"{tag}_ckpt.json"
    # t13_run's --resume does `np.load(ck_lat)` and iterates the first axis, so
    # the file must be (B, 1, 16, H/8, W/8) even for B=1; saving lat[0] would
    # drop the batch axis and x_embedding would get a 3-D tensor.
    np.save(ck_lat, np.stack(lat))
    ck_js.write_text(json.dumps(
        {"next_step": 0, "steps": int(steps), "layers": 30, "batch": int(batch),
         "lat0": [np.asarray(x, np.float32).ravel().tolist() for x in lat],
         "calls": [], "steps_rec": []}), encoding="utf-8")
    emit("shape_geometry", height=int(height), width=int(width),
         patch_grid=[gh, gw], image_tokens=tokens, latent=[16, int(height) // 8,
                                                           int(width) // 8],
         m=m, resume=str(ck_lat))
    print(f"[bridge] shape {height}x{width}: patch grid {gh}x{gw}, "
          f"latent 16x{height // 8}x{width // 8}, M={m}, "
          f"initial latent via --resume", flush=True)
    return {"height": int(height), "width": int(width), "gh": gh, "gw": gw,
            "resume": True, "ckpt": str(ck_js), "ckpt_lat": str(ck_lat)}


def _install_refiner_mmap(refiner_cache, drv=None):
    """Keep the 2.67 GB refiner f32 cache file-backed, not private commit.

    The values are read-only and identical to the shipped cache; mmap changes
    no arithmetic, it only lets Windows evict clean pages under pressure.  If
    anything is missing, the original loader is used unchanged.
    """
    drv = drv or t13_run
    root = pathlib.Path(refiner_cache)
    orig = drv._refblock_cached

    def mmap_refblock(RW, key):
        d = root / key
        man = d / "manifest.json"
        if not man.is_file():
            return orig(RW, key)
        try:
            names = json.loads(man.read_text(encoding="utf-8"))
            out = {}
            for nm in names:
                if nm.startswith("!"):
                    out[nm[1:]] = None
                else:
                    out[nm] = np.load(d / (nm + ".npy"), mmap_mode="r")
            drv._RC_HIT[0] = True
            total = sum(v.nbytes for v in out.values() if v is not None)
            print("[webui] refiner mmap %-14s %.2f GB (file-backed)"
                  % (key, total / 2**30), flush=True)
            return out
        except Exception as exc:  # noqa: BLE001
            print("[webui] refiner mmap failed for %s (%s); loading"
                  % (key, exc), flush=True)
            return orig(RW, key)

    drv._refblock_cached = mmap_refblock
    return True


# ---------------------------------------------------------------------------
# T22 prewarm: split the startup's 25-27 s "W pack" out of the job
# ---------------------------------------------------------------------------
# The delivered startup spends 22-37 s in --warm: 30 layers x 4 projections
# (120 x 16,588,800 B = 1.85 GiB) are read out of weights.bfp16 (6.9 GB) and
# re-encoded into one uint8 array.  Measurement on this host (2026-10-20,
# imaggers/prewarm/t22_prewarm_ab.py): that work is
#
#   * PURE CPU.  Loop.w_images() reads weights through t3_real.Real and calls
#     pack_w64(); it never touches the device -- the _NoDevice guard below turns
#     any accidental attribute access into a hard error instead of an open.  So
#     the pack can run in a SIBLING process while this job is still text
#     encoding (or while another job's VAE runs): no GIL, no device context
#     (device contexts cannot coexist with the VAE-NPU chain, T21 2a2b2 4.3).
#   * a PURE FUNCTION of weights.bfp16 plus the 30 layer keys.  The sequence
#     length M does not enter the projection packing, so ONE artifact serves
#     every shape, seed, step and batch.
#
# Protocol (the job side is opt-in per launch, so a missing/bad artifact simply
# falls back to the delivered in-process pack):
#     py310 bridge.py --pack-prewarm --pack-out <file.bin> --cache-dir <dir> ...
#     py310 bridge.py <the delivered args> --wpack-file <file.bin>
#
# <file.bin> is 120 x W_BYTES raw bytes; <file.bin>.json records the slot
# order, the weights stamp, the format version and the sha256 of the bytes.
# The consumer maps the file as T.WPACK and turns Loop.w_images() into a cache
# hit -- the same seam the resident device worker already uses (dev_worker.py
# w_images_reuse), so the dispatch payload is the same bytes by construction.
WPACK_VERSION = 2          # bump when the FILE LAYOUT changes; a mismatch only
#                            means "ignored, pack in-process as delivered"
WPACK_CODE = "t22_prewarm_1"   # bump when the pack itself changes
WPACK_LAYERS = 30          # must match driver_argv()'s "--layers"
WPACK_NAMES = ("q", "k", "v", "o")


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _weights_stamp():
    """(path, size, mtime_ns) of weights.bfp16 -- the only input of the pack."""
    import t3_real as T3
    p = pathlib.Path(T3.WBF)
    st = p.stat()
    return [str(p), int(st.st_size), int(st.st_mtime_ns)]


class _NoDevice:
    """A device stand-in that refuses everything.

    The prewarm is pure CPU by design; if a future driver ever needs the device
    to pack, this fails loudly in a log line instead of opening a context that
    would then collide with the VAE chain.
    """

    def __getattr__(self, name):
        raise SystemExit("[bridge] T22 prewarm: packing touched the device "
                         "(%r); it must stay pure CPU" % name)


def _install_pack_runtime(drv):
    """The driver module globals the delivered --warm path sets before packing.

    Identical to dev_worker.py's prologue so the packer and the job cannot
    disagree about the codec: FAST_HOST + H.install() replace pack_w64/
    pack_b_nk with the nJIT kernels, which is what the delivered bridge runs.
    """
    drv.FAST_HOST = True
    drv.H.install()
    drv.SCRATCH = drv.H.Scratch()
    drv.NH_BG = 5
    drv.NH_BG_CORE = True
    drv.CACHE_ALL = True
    drv.ASSERT_SBG = True
    return drv



def pack_prewarm(args) -> int:
    """--pack-prewarm: build the 1.85 GiB packed projection array, pure CPU.

    Writes <pack-out> and only then <pack-out>.json, so a reader can never see
    a half-written file described by a valid manifest: the meta is the commit
    record.  Any earlier meta is removed first (a crashed pack leaves no meta).
    """
    out = pathlib.Path(args.pack_out).resolve()
    meta_path = pathlib.Path(str(out) + ".json")
    cache_dir = pathlib.Path(args.cache_dir).resolve()
    t_all = time.perf_counter()
    DRV = _load_driver(args.height, args.width, args.driver)
    layers = int(args.layers)
    if (int(args.height), int(args.width)) and hasattr(DRV, "configure"):
        DRV.configure(int(args.height), int(args.width))
    DRV.T4_OUT = cache_dir
    DRV.OUT_A = cache_dir
    _install_token_layout_fix()
    _install_pack_runtime(DRV)
    w_bytes = int(DRV.W_BYTES)
    n = layers * len(WPACK_NAMES)
    need = n * w_bytes
    out.parent.mkdir(parents=True, exist_ok=True)
    ok, why, meta0 = _wpack_check(out, DRV, layers)
    if ok:
        print("[t22] prewarm: %s is already valid (built %s, %.1f s pack); "
              "nothing to do" % (out.name, meta0.get("created"),
                                 float(meta0.get("pack_s") or 0.0)), flush=True)
        emit("prewarm_pack_done", path=str(out), reused=True,
             bytes=int(meta0.get("bytes") or 0), slots=int(meta0.get("count") or 0),
             pack_s=float(meta0.get("pack_s") or 0.0),
             sha256=str(meta0.get("sha256") or ""),
             process_s=round(time.perf_counter() - t_all, 3))
        return 0
    if why:
        print("[t22] prewarm: rebuilding (%s)" % why, flush=True)
    free = shutil.disk_usage(str(out.parent)).free
    if free < need + (1 << 30):
        print("[t22] prewarm: %.2f GiB free under %s, need %.2f GiB + 1 GiB "
              "headroom; not packing" % (free / 2**30, out.parent, need / 2**30),
              flush=True)
        return 3
    # the meta is the commit record: drop it before the bytes can change
    try:
        meta_path.unlink()
    except OSError:
        pass
    tmp = pathlib.Path(str(out) + ".part")
    try:
        tmp.unlink()
    except OSError:
        pass
    t0 = time.perf_counter()
    # ONE LAYER of scratch + a sequential write, not one 1.9 GB array.  The
    # delivered --warm needs 1.85 GiB of private commit for the whole array;
    # this host has 5-7 GB of commit headroom while other work runs, and the
    # first version of this packer died at layer 5 with "Unable to allocate
    # 56.2 MiB" (2026-10-20 11:07, commit 5.2 GB free).  Four slots are all the
    # packer ever needs: 4 x 16,588,800 B = 63 MiB, 1/30 of the array.
    buf = np.empty(len(WPACK_NAMES) * w_bytes, np.uint8)
    DRV.WPACK_SECONDS[0] = 0.0
    DRV.WPACK_SECONDS[1] = 0
    L = DRV.Loop(_NoDevice(), "t22pack", dimk64=True)
    slots = []
    print("[t22] prewarm: %d layers x %d projections, %.3f GB -> %s "
          "(device NOT opened, %.1f MiB of scratch)"
          % (layers, len(WPACK_NAMES), need / 1e9, out,
             buf.nbytes / 2**20), flush=True)
    fh = open(tmp, "wb")
    try:
        for li in range(layers):
            key = "layers.%d" % li
            # w_images() numbers slots by len(WPACK_SLOTS), so clearing it per
            # layer makes the layer's four images land in buf in q,k,v,o order;
            # the FILE slot is the global one (layer * 4 + that index), which is
            # exactly the order the delivered --warm writes its 1.85 GiB array.
            DRV.WPACK = buf
            DRV.WPACK_SLOTS.clear()
            L.w_images(key)
            L.params(key)
            base = li * len(WPACK_NAMES)
            for j, nm in enumerate(WPACK_NAMES):
                local = int(DRV.WPACK_SLOTS[(key, nm)])
                if local != j:
                    raise SystemExit("[bridge] prewarm: slot order %r -> %d, "
                                     "expected %d" % (nm, local, j))
                slots.append([key, nm, base + local])
            fh.write(buf)
            if (li + 1) % 5 == 0 or li + 1 == layers:
                print("[t22] prewarm pack: layer %2d/%d  %6.2f s  (%d slots, "
                      "%.2f GB written)"
                      % (li + 1, layers, time.perf_counter() - t0, len(slots),
                         fh.tell() / 1e9), flush=True)
        fh.flush()
        os.fsync(fh.fileno())
    finally:
        fh.close()
    t_pack = time.perf_counter() - t0
    L.wcache.clear()
    DRV.WPACK_SLOTS.clear()
    DRV.WPACK = None
    del L, buf
    import gc
    gc.collect()
    if tmp.stat().st_size != need:
        raise SystemExit("[bridge] prewarm: wrote %d B, expected %d B"
                         % (tmp.stat().st_size, need))
    sha = ""
    if args.pack_verify:
        t_v = time.perf_counter()
        sha = _sha256_file(tmp)
        print("[t22] prewarm: sha256 of the packed bytes %s (%.2f s)"
              % (sha[:16], time.perf_counter() - t_v), flush=True)
    os.replace(str(tmp), str(out))
    meta = {
        "version": WPACK_VERSION,
        "code": WPACK_CODE,
        "layers": layers,
        "names": list(WPACK_NAMES),
        "w_bytes": w_bytes,
        "slots": slots,
        "count": len(slots),
        "bytes": need,
        "file": out.name,
        "file_size": out.stat().st_size,
        "sha256": sha,
        "weights": _weights_stamp(),
        "images": _manifest_images(cache_dir),
        "shape": "%dx%d" % (int(args.height), int(args.width)),
        "pack_s": round(t_pack, 3),
        "process_s": round(time.perf_counter() - t_all, 3),
        "pid": os.getpid(),
        "python": sys.version.split()[0],
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": ("pure-CPU prewarm of the projection images; the device was "
                 "never opened. Consumed by --wpack-file."),
    }
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    emit("prewarm_pack_done", path=str(out), bytes=need, slots=len(slots),
         pack_s=round(t_pack, 3), sha256=sha,
         process_s=round(time.perf_counter() - t_all, 3))
    print("[t22] prewarm: packed %.3f GB in %.1f s (process %.1f s) -> %s"
          % (need / 1e9, t_pack, time.perf_counter() - t_all, out), flush=True)
    return 0


def _manifest_images(cache_dir):
    try:
        man = json.loads((pathlib.Path(cache_dir) / "manifest.json")
                         .read_text(encoding="utf-8"))
    except Exception:                                         # noqa: BLE001
        return {}
    return man.get("images") or {}


def _wpack_check(path, drv, layers=WPACK_LAYERS):
    """Cheap validity of a prewarm artifact: (ok, why, meta)."""
    p = pathlib.Path(path)
    mp = pathlib.Path(str(p) + ".json")
    if not p.is_file():
        return False, "missing %s" % p.name, {}
    if not mp.is_file():
        return False, "missing %s" % mp.name, {}
    try:
        meta = json.loads(mp.read_text(encoding="utf-8"))
    except Exception as exc:                                  # noqa: BLE001
        return False, "unreadable %s (%s)" % (mp.name, exc), {}
    if int(meta.get("version") or 0) != WPACK_VERSION:
        return False, ("format version %s, this bridge writes %d"
                       % (meta.get("version"), WPACK_VERSION)), meta
    if str(meta.get("code") or "") != WPACK_CODE:
        return False, ("written by %r, this bridge is %r"
                       % (meta.get("code"), WPACK_CODE)), meta
    if int(meta.get("layers") or 0) != int(layers):
        return False, "artifact has %s layers, job has %d" % (
            meta.get("layers"), layers), meta
    w_bytes = int(drv.W_BYTES)
    if int(meta.get("w_bytes") or 0) != w_bytes:
        return False, "artifact W image %s B, this driver %d B" % (
            meta.get("w_bytes"), w_bytes), meta
    n = int(layers) * len(WPACK_NAMES)
    slots = meta.get("slots") or []
    if len(slots) != n:
        return False, "artifact has %d slots, expected %d" % (len(slots), n), meta
    try:
        size = p.stat().st_size
    except OSError as exc:
        return False, "cannot stat %s (%s)" % (p.name, exc), meta
    if size != n * w_bytes:
        return False, "artifact is %d B, expected %d B" % (size, n * w_bytes), meta
    try:
        if list(meta.get("weights") or []) != list(_weights_stamp()):
            return False, "weights.bfp16 changed since the artifact was built", meta
    except OSError as exc:
        return False, "cannot stamp the weights (%s)" % exc, meta
    return True, "", meta


def _load_wpack_file(path, drv, layers=WPACK_LAYERS, verify=False):
    """Attach a prewarmed artifact to drv: WPACK is the mapped file, and
    Loop.w_images() returns the mapped slices instead of packing.

    The map is opened "r+" so numpy exposes a WRITABLE buffer: pyxrt's bo.write
    takes the buffer protocol and the delivered path hands it a writable
    ndarray view -- the driver only ever reads these slots, and nothing else in
    this process writes into T.WPACK (the --warm loop is a cache hit).
    """
    ok, why, meta = _wpack_check(path, drv, layers)
    if not ok:
        raise RuntimeError(why)
    p = pathlib.Path(path)
    w_bytes = int(drv.W_BYTES)
    n = int(layers) * len(WPACK_NAMES)
    if verify and meta.get("sha256"):
        t0 = time.perf_counter()
        got = _sha256_file(p)
        if got != meta["sha256"]:
            raise RuntimeError("artifact bytes do not match its manifest "
                               "(sha256 %s != %s)" % (got[:16], meta["sha256"][:16]))
        vs = time.perf_counter() - t0
    else:
        vs = 0.0
    t0 = time.perf_counter()
    mm = np.memmap(str(p), dtype=np.uint8, mode="r+", shape=(n * w_bytes,))
    slots = [(str(k), str(nm), int(s)) for k, nm, s in meta["slots"]]
    drv.WPACK = mm
    drv.WPACK_SLOTS.clear()
    drv.WPACK_SLOTS.update({(k, nm): s for k, nm, s in slots})
    wc = {}
    for key, nm, slot in slots:
        wc.setdefault(key, {})[nm] = mm[slot * w_bytes:(slot + 1) * w_bytes]
    if not getattr(drv.Loop, "_t22_prewarm_patched", False):
        orig = drv.Loop.w_images

        def w_images_reuse(self, key, _orig=orig, _wc=wc):
            got = _wc.get(key)
            if got is not None:
                return got
            return _orig(self, key)

        drv.Loop.w_images = w_images_reuse
        drv.Loop._t22_prewarm_patched = True
    map_s = time.perf_counter() - t0
    print("[t22] prewarm: %d slots of %d B mapped from %s (%.2f GB, %s); "
          "the driver's --warm loop is a cache hit"
          % (len(slots), w_bytes, p.name, n * w_bytes / 2**30,
             "sha256 verified in %.2f s" % vs if verify else "sha256 not checked"),
          flush=True)
    emit("wpack_prewarm", mode="file", path=str(p), slots=len(slots),
         bytes=n * w_bytes, map_s=round(map_s, 3), verify_s=round(vs, 3),
         pack_s=meta.get("pack_s"), built=meta.get("created"))
    return {"mode": "file", "path": str(p), "slots": len(slots),
            "bytes": n * w_bytes, "map_s": map_s, "verify_s": vs,
            "pack_s": meta.get("pack_s"), "sha256": meta.get("sha256") or "",
            "built": meta.get("created")}



def driver_argv(args, imgs, shape_info, ffn_img, cache_dir, wpack=True,
                t18=None):
    """The delivered driver's argv for one job (shared by main() and the
    resident device worker, so the two paths can never drift)."""
    driver = [
        "--tag", args.tag,
        "--steps", str(args.steps),
        "--layers", "30",
        "--seed", str(args.seed),
        "--batch", str(args.batch),
        "--warm",
        "--dimk64",
        "--ffn-image", ffn_img,
        "--nh-bg", "5",
        "--nh-bg-core",
        "--prefetch",
        "--refiner-cache", args.refiner_cache,
        "--att-image", imgs.get("proj", "t7_p5"),
        "--att-ip-image", imgs.get("ip", "t7_ip"),
        "--att-dir", str(cache_dir),
        "--fast-host",
        "--cache-all",
    ]
    if wpack:
        driver.append("--wpack-array")
    if DRV is not t13_run:
        driver += ["--height", str(int(args.height)),
                   "--width", str(int(args.width))]
    if args.host_ref:
        driver.append("--host-ref")
    if args.no_cache_all:
        driver.remove("--cache-all")
        driver.append("--no-cache-all")
    if args.fast_codec:
        driver.append("--fast-codec")
    if args.bo_resident:
        driver.append("--bo-resident")
    if args.batch > 1:
        driver.append("--share-weights")
    if shape_info:
        driver.append("--resume")
    # T22 qkv: ALWAYS explicit (the default is ON), so a page that unticks the
    # box really gets the delivered projection path.  getattr() because the
    # resident device worker builds its own namespace from a job request;
    # t18= is how that worker names its driver without touching bridge.DRV
    # (t13_run has neither flag).
    is_t18 = (DRV is not t13_run) if t18 is None else bool(t18)
    if getattr(args, "qkv", False):
        driver += ["--qkv", "--qkv-xclbin", args.qkv_xclbin,
                   "--qkv-bin", args.qkv_bin]
    elif is_t18:
        driver.append("--no-qkv")
    # T22 p10: same rule as qkv -- always explicit, so a caller that does not
    # know about the switch (a runner that has not been taught the field yet,
    # a resident worker's namespace) still runs the DELIVERED o path.
    if getattr(args, "p10", False):
        driver += ["--p10", "--p10-xclbin", args.p10_xclbin,
                   "--p10-bin", args.p10_bin]
    elif is_t18:
        driver.append("--no-p10")
    if getattr(args, "xpack_once", False):
        driver.append("--xpack-once")
    # T22 glue: ALWAYS explicit when on, and the negative form when the caller
    # knows the switch (so an opt-in job cannot be silently inherited).
    # The default is the SERVICE switch, not False: the resident device worker
    # builds its own namespace (dev_worker.py) and never passes these fields, so
    # a worker job must still see T22_FAST_GLUE/T22_GLUE2 from the server env.
    # t13_run has neither flag: only t18_run may receive them (same rule as
    # qkv/p10 above, or --driver t13_run would die in argparse).
    if is_t18 and getattr(args, "t22_fast_glue", FAST_GLUE_DEFAULT):
        driver.append("--t22-fast-glue")
    if is_t18 and getattr(args, "t22_glue2", GLUE2_DEFAULT):
        driver.append("--t22-glue2")
    return driver


def _pack_main(argv) -> int:
    """--pack-prewarm: its own parser, so the delivered job CLI cannot change.

    --tag/--pe/--schedule are meaningless here: the pack does not depend on the
    prompt, the schedule or the seed (see the section comment above).
    """
    ap = argparse.ArgumentParser(prog="bridge.py --pack-prewarm")
    ap.add_argument("--pack-prewarm", action="store_true", required=True)
    ap.add_argument("--pack-out", required=True,
                    help="the packed array file (its .json sibling is the "
                         "manifest a consumer validates)")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--layers", type=int, default=WPACK_LAYERS)
    ap.add_argument("--driver", default="")
    ap.add_argument("--pack-verify", action="store_true",
                    help="hash the packed bytes into the manifest (2 GB read)")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    return pack_prewarm(args)


def main() -> int:
    if "--pack-prewarm" in sys.argv[1:]:
        return _pack_main(sys.argv[1:])
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pe", required=True)
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--bo-resident", action="store_true")
    ap.add_argument("--no-cache-all", action="store_true")
    ap.add_argument("--fast-codec", action="store_true",
                    help="opt-in only: t6_codec crashed at first dispatch after "
                         "the 2026-10-18 reboot; safe default is off")
    ap.add_argument("--refiner-mmap", action="store_true",
                    help="opt-in: keep the refiner f32 cache file-backed; "
                         "post-reboot this path crashed at first dispatch")
    ap.add_argument("--refiner-cache", default=str(NPU / "imaggers" / "image" / "t6_refcache"))
    ap.add_argument("--meta", default="")
    ap.add_argument("--driver", default="",
                    help="T18: force t13_run | t18_run (default: by M)")
    ap.add_argument("--host-ref", dest="host_ref", action="store_true",
                    help="T18: the driver's per-layer host-f32 attention/FFN "
                         "reference on step 0 (numerical gate, slow)")
    ap.add_argument("--wpack-file", default="",
                    help="T22 prewarm: map the packed projection images from "
                         "this file instead of packing them in this process "
                         "(the delivered in-process pack is the fallback)")
    ap.add_argument("--wpack-verify", action="store_true",
                    help="T22 prewarm: check the file's sha256 before mapping "
                         "it (2 GB read; the manifest checks are free)")
    ap.add_argument("--layers", type=int, default=WPACK_LAYERS,
                    help="T22 prewarm: only used to validate the artifact; the "
                         "delivered argv pins --layers 30")
    ap.add_argument("--qkv", action="store_true", default=QKV_ENV,
                    help="T22: run the driver's FUSED q/k/v image (one "
                         "dispatch per 96-row block).  Default OFF; the "
                         "environment variable T22_QKV=1 is the same switch "
                         "for a WebUI-launched job")
    ap.add_argument("--no-qkv", dest="qkv", action="store_false",
                    help="T22 qkv: force the DELIVERED projection path even "
                         "though the fused image is the default")
    ap.add_argument("--qkv-xclbin", default="",
                    help="T22 qkv: the fused xclbin (default: the t22_qkv "
                         "delivery imaggers/block/qkv/out/qkv_par15m.xclbin)")
    ap.add_argument("--qkv-bin", default="",
                    help="T22 qkv: its instruction stream (default: next to "
                         "the xclbin)")
    ap.add_argument("--p10", "--p15", action="store_true", default=P10_ENV,
                    help="T22: run the o projection (att/o) on its own "
                         "multi-stream image instead of the delivered 5-stream "
                         "t7_p5.  Default OFF; T22_P10=1 is the same switch "
                         "for a WebUI-launched job")
    ap.add_argument("--no-p10", "--no-p15", dest="p10", action="store_false",
                    help="T22 p10: force the DELIVERED o projection")
    ap.add_argument("--p10-xclbin", default="",
                    help="T22 p10: the o-projection xclbin (default: the "
                         "t22_proj delivery imaggers/attn_impl/proj/out/p10.xclbin)")
    ap.add_argument("--p10-bin", default="",
                    help="T22 p10: its instruction stream (default: next to "
                         "the xclbin)")
    ap.add_argument("--xpack-once", action="store_true", default=XPACK_ENV,
                    help="T22: pack the projection X image once per row block "
                         "instead of once per projection (host-side only; "
                         "default OFF)")
    ap.add_argument("--t22-fast-glue", dest="t22_fast_glue",
                    action="store_true", default=FAST_GLUE_DEFAULT,
                    help="T22 U1 host fast path (T22_FAST_GLUE=1; default OFF "
                         "until the product four-piece passes)")
    ap.add_argument("--no-t22-fast-glue", dest="t22_fast_glue",
                    action="store_false")
    ap.add_argument("--t22-glue2", dest="t22_glue2", action="store_true",
                    default=GLUE2_DEFAULT,
                    help="T22 glue-2 copy-free drain/pack (T22_GLUE2=1; "
                         "default OFF)")
    ap.add_argument("--no-t22-glue2", dest="t22_glue2", action="store_false")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    z = np.load(args.pe)
    pe = np.asarray(z["pe"], np.float32)
    s = np.load(args.schedule)
    ts = np.asarray(s["timesteps"], np.float64)
    sg = np.asarray(s["sigmas"], np.float64)
    if len(ts) != args.steps or len(sg) != args.steps + 1:
        print(f"[bridge] schedule has {len(ts)} timesteps / {len(sg)} sigmas "
              f"for {args.steps} steps", flush=True)
        return 2
    if pe.shape[1] > 32:
        print(f"[bridge] prompt is {pe.shape[1]} tokens > cap_tokens=32; "
              f"the M=96 image cannot hold it", flush=True)
        return 2

    cache_dir = pathlib.Path(args.cache_dir).resolve()
    global DRV
    DRV = _load_driver(args.height, args.width, args.driver)
    # T22 qkv: resolve the fused image here, so a missing file is ONE bridge
    # line with the path in it instead of a driver traceback.
    args.qkv_fallback = False
    if args.qkv:
        if DRV is t13_run:
            # --driver t13_run stays usable: it has no fused-image support.
            print("[bridge] T22 qkv: t13_run has no fused-image registration;"
                  " running the delivered driver", flush=True)
            args.qkv = False
        else:
            _xc = (pathlib.Path(args.qkv_xclbin) if args.qkv_xclbin
                   else QKV_DEFAULT_OUT / (QKV_DEFAULT_TAG + ".xclbin"))
            _bn = (pathlib.Path(args.qkv_bin) if args.qkv_bin
                   else _xc.with_suffix(".bin"))
            if not (_xc.is_file() and _bn.is_file()):
                # *.xclbin is not in git: a fresh checkout must still run.
                print("[bridge] T22 qkv: fused image missing (%s); running "
                      "the delivered projection path.  Build it, or set "
                      "T22_QKV=0 to silence this." % _xc, flush=True)
                args.qkv = False
                args.qkv_fallback = True
            else:
                args.qkv_xclbin, args.qkv_bin = str(_xc), str(_bn)
                print("[bridge] T22 qkv: fused image %s (+ %s)"
                      % (_xc.name, _bn.name), flush=True)
    # T22 p10: resolve the o image here too (same reason as qkv -- a missing
    # file is ONE bridge line with the path in it, not a driver traceback).
    args.p10_fallback = False
    if args.p10:
        if DRV is t13_run:
            print("[bridge] T22 p10: t13_run has no o-image registration; "
                  "running the delivered o projection", flush=True)
            args.p10 = False
        else:
            _xc = (pathlib.Path(args.p10_xclbin) if args.p10_xclbin
                   else P10_DEFAULT_OUT / (P10_DEFAULT_TAG + ".xclbin"))
            _bn = (pathlib.Path(args.p10_bin) if args.p10_bin
                   else _xc.with_suffix(".bin"))
            if not (_xc.is_file() and _bn.is_file()):
                print("[bridge] T22 p10: o image missing (%s); running the "
                      "delivered o projection." % _xc, flush=True)
                args.p10 = False
                args.p10_fallback = True
            else:
                args.p10_xclbin, args.p10_bin = str(_xc), str(_bn)
                print("[bridge] T22 p10: o image %s (+ %s)"
                      % (_xc.name, _bn.name), flush=True)
    if args.xpack_once:
        print("[bridge] T22 xpack-once: the projection X image is packed "
              "once per row block", flush=True)
    man = {}
    try:
        man = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    except Exception:
        man = {}
    imgs = man.get("images") or {"proj": "t7_p5", "ip": "t7_ip",
                                 "ffn": _config.FFN_IMAGE}
    ffn_img = imgs.get("ffn", _config.FFN_IMAGE)
    if args.refiner_mmap:
        _install_refiner_mmap(args.refiner_cache, DRV)
    if not (cache_dir / (ffn_img + ".xclbin")).is_file():
        print(f"[bridge] cache dir has no {ffn_img}.xclbin: {cache_dir}",
              flush=True)
        return 2
    DRV.T4_OUT = cache_dir
    DRV.OUT_A = cache_dir
    _install_dev_keep(DRV)
    _install_token_layout_fix()
    shape_info = _install_shape_geometry(args.height, args.width, args.tag,
                                         args.steps, args.seed, args.batch)
    _install(pe, ts, sg)

    # T22 prewarm: the packed projection images are a pure function of
    # weights.bfp16, so a sibling process can build them once and every job can
    # map them.  When the artifact does not check out we say why ONCE and pack
    # in this process exactly as the delivered path does -- the attribution of
    # the startup stays the same either way.
    wpack_prewarm = None
    if args.wpack_file:
        try:
            wpack_prewarm = _load_wpack_file(args.wpack_file, DRV,
                                            layers=int(args.layers),
                                            verify=bool(args.wpack_verify))
        except Exception as exc:                              # noqa: BLE001
            print("[t22] prewarm: %s is unusable (%s); packing in this process "
                  "as delivered" % (args.wpack_file, exc), flush=True)
    driver = driver_argv(args, imgs, shape_info, ffn_img, cache_dir,
                         wpack=wpack_prewarm is None)
    sys.argv = [str(pathlib.Path(DRV.__file__))] + driver

    meta = {
        "tag": args.tag,
        "seed": args.seed,
        "batch": args.batch,
        "steps": args.steps,
        "bo_resident": bool(args.bo_resident),
        "low_memory": bool(args.no_cache_all),
        "fast_codec": bool(args.fast_codec),
        "refiner_mmap": bool(args.refiner_mmap),
        "cache_dir": str(cache_dir),
        "driver": pathlib.Path(DRV.__file__).name,
        "qkv": bool(args.qkv),
        "qkv_fallback": bool(getattr(args, "qkv_fallback", False)),
        "qkv_xclbin": str(getattr(args, "qkv_xclbin", "") or ""),
        "qkv_bin": str(getattr(args, "qkv_bin", "") or ""),
        "p10": bool(args.p10),
        "p10_fallback": bool(getattr(args, "p10_fallback", False)),
        "p10_xclbin": str(getattr(args, "p10_xclbin", "") or ""),
        "p10_bin": str(getattr(args, "p10_bin", "") or ""),
        "xpack_once": bool(args.xpack_once),
        "t22_fast_glue": bool(args.t22_fast_glue),
        "t22_glue2": bool(args.t22_glue2),
        "wpack_prewarm": wpack_prewarm,
        "images": imgs,
        "height": int(args.height), "width": int(args.width),
        "shape": f"{int(args.height)}x{int(args.width)}",
        "shape_geometry": shape_info,
        "pe_shape": list(pe.shape),
        "pe_sha256": pe_sha256(pe),
        "timesteps": [float(x) for x in ts],
        "sigmas": [float(x) for x in sg],
        "python": sys.version,
        "started": _T0,
    }
    emit("bridge_start", tag=args.tag, seed=args.seed, batch=args.batch,
         steps=args.steps, pe_tokens=int(pe.shape[1]),
         pe_sha256=meta["pe_sha256"])
    def hard_exit(code):
        # The delivered stack can crash in xclbin/BO destructors at interpreter
        # teardown (0xC0000005) AFTER calls.json/latents are on disk.  Skip
        # Python teardown entirely: the OS closes the handles, the next process
        # starts clean, and the artifact is the measurement (T13 R5).
        meta["rc"] = int(code)
        meta["ended"] = time.time()
        meta["elapsed_s"] = time.time() - _T0
        if args.meta:
            try:
                pathlib.Path(args.meta).write_text(
                    json.dumps(meta, indent=1), encoding="utf-8")
            except Exception:
                pass
        emit("driver_done", code=int(code), elapsed_s=time.time() - _T0)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(int(code))

    try:
        rc = int(DRV.main() or 0)
    except SystemExit as exc:
        code = exc.code
        rc = code if isinstance(code, int) else (0 if code is None else 2)
        if not isinstance(code, int) and code is not None:
            print(f"[bridge] SystemExit: {code}", flush=True)
        emit("driver_exit", code=rc)
        hard_exit(rc)
    except BaseException:
        traceback.print_exc()
        emit("driver_error", trace=traceback.format_exc()[-4000:])
        hard_exit(2)
    hard_exit(rc)


if __name__ == "__main__":
    raise SystemExit(main())
