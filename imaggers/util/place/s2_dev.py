# SPDX-License-Identifier: MIT
"""s2_dev.py -- product-facing 32-core FFN and projection segments.

Two calls, nothing else:

    import s2_dev
    dev = s2_dev.S2Dev()                      # takes imaggers/device.lock
    y   = s2_dev.ffn32_run(dev, xn, "layers.0")     # xn (1056,3840) f32 -> y
    o   = s2_dev.proj32_run(dev, x, "layers.0")     # x  (1056,3840) f32 -> q/k/v/o
    dev.close()

The BO / dispatch protocol is NOT re-implemented here: run_ffn.ImageRunner is
the code that ran the four FFN arms on the device (four gates PASS).  This
module only chains images and converts layouts.

Startup (once per process, before the loop):

    s2_dev.prewarm_ball()        # 30 layers, ~6.0 GB cache, ~8 s

Every B payload is a pure permutation of the model's own BFP16 blocks
(pack_b_from_blocks, 727 MB/s measured), so it is done ONCE and mmapped per
layer; doing it per layer would cost 274 ms/layer.

Smoke (one layer, one process, C on disk):

    py310 s2_dev.py --smoke ffn  --key layers.0
    py310 s2_dev.py --smoke proj --key layers.0
    py310 s2_dev.py --smoke ffn --dry-only     # CPU half only, no device
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "out"
CACHE = pathlib.Path(os.environ.get("T22_S2_CACHE_DIR", str(OUT / "bcache")))
LOCK = ROOT / "imaggers" / "device.lock"
sys.path.insert(0, str(HERE))

import s2_ffn32 as S2                      # layouts, codec, dry FFN
import s2_proj32 as P2                     # projection weights + dry
import run_ffn                             # ImageRunner: the tested BO path

D, FF, M_TILE = 3840, 10240, 32
COLS, CPC = 8, 4

# (shape, K, N, n, ntiles, cpc, C bytes) -- C bytes cross-checked against the
# xclbin's aie.runtime_sequence signature in the payload preps.
IMG = {
    "gu_main": ("ffn_gu_m32",   3840, 10240, 512, 20, 4, 20971520),
    "gu_tail": ("ffn_tail_m32", 3840, 10240, 320,  1, 4,   655360),
    "dn_main": ("ffn_dn2_m32", 10240,  3840, 384, 10, 4,  7864320),
    "dn_tail": ("ffn_tail_dn2", 10240, 3840, 240,  1, 2,   245760),
    "pj_main": ("proj32_fused768",      3840, 15360, 768, 20, 4, 31457280),
    "pj_tail": ("proj32_fused768_tail", 3840, 15360, 480,  1, 4,   983040),
}
# main images read [ntile][nkk][n*9]; the TAIL images read
# [cols][nkk][cpc][n*9] -- the same blocks in a different nest, and the byte
# COUNT is identical, so only the tag can tell them apart.
B_BYTES = {"gu_gate": 44236800, "gu_up": 44236800, "dn": 44236800,
           "proj": 66355200,
           "gu_gate_t": 44236800, "gu_up_t": 44236800, "dn_t": 44236800,
           "proj_t": 66355200}


# ---------------------------------------------------------------------------
# B: one-time repack, mmapped per layer
# ---------------------------------------------------------------------------
def _model_blocks(key, names):
    """Checkpoint -> the image's (N, K/8, 9) BFP16 blocks.

    THE CHECKPOINT BYTES ARE NOT (out, in) ROW-MAJOR BLOCKS.  pack_bfp16.unpack
    is the only decoder that knows the file's layout; decoding the raw bytes as
    reshape(N, K/8*9) gives a matrix with the right magnitude and corr 3e-4 to
    the truth (measured).  Everything here goes through pack.unpack, then
    re-encodes the (K, N) transpose into the block order the images read.
    """
    li = int(key.split(".")[1])
    sys.path.insert(0, str(ROOT))
    import pack_bfp16 as _pb
    z = np.load(ROOT / "weights" / f"block{li}.npz", allow_pickle=False)
    out = {}
    for nm, shp in names:
        W = np.asarray(_pb.unpack(z[f"{key}.{nm}.weight"], shp[0], shp[1]),
                       np.float32)                       # (out, in)
        # blocks = encode of the (N, K) matrix: rows are output columns, the
        # blocks run along K -- exactly what pack_b_from_blocks /
        # pack_b_tail_from_blocks expect
        enc = S2.encode_bfp(np.ascontiguousarray(W))     # (N, K/8 * 9)
        out[nm] = np.ascontiguousarray(
            enc.reshape(shp[0], shp[1] // 8, 9))
    return out


def _ffn_blocks(key):
    return _model_blocks(key, (("feed_forward.w1", (10240, 3840)),
                               ("feed_forward.w3", (10240, 3840)),
                               ("feed_forward.w2", (3840, 10240))))


def _proj_blocks(key):
    return _model_blocks(key, tuple(
        (f"attention.{n}", (3840, 3840))
        for n in ("to_q", "to_k", "to_v", "to_out.0")))


def _want_tags():
    """parent fix 2026-11-01: optionally restrict which payloads are cached.

    T22_S2_TAGS=proj,proj_t shrinks the per-layer cache from 398 MB to ~68 MB,
    which is what a projection-only A/B needs (30 layers = 2.0 GB instead of
    11.9 GB).  None means "all".
    """
    raw = os.environ.get("T22_S2_TAGS", "").strip()
    if not raw:
        return None
    return set(t.strip() for t in raw.split(",") if t.strip())


def prewarm_b(key):
    """Repack one layer's B payloads into CACHE.  Returns (bytes, seconds)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    w = _ffn_blocks(key)
    jobs = {
        "gu_gate": lambda: S2.pack_b_from_blocks(w["feed_forward.w1"], 3840, 10240, 512),
        "gu_up": lambda: S2.pack_b_from_blocks(w["feed_forward.w3"], 3840, 10240, 512),
        "dn": lambda: S2.pack_b_from_blocks(w["feed_forward.w2"], 10240, 3840, 384),
        "proj": None,
    }
    p = _proj_blocks(key)
    fused = np.concatenate([p[f"attention.{n}"] for n in
                            ("to_q", "to_k", "to_v", "to_out.0")], axis=0)
    jobs["proj"] = lambda: S2.pack_b_from_blocks(fused, 3840, 15360, 768)
    # the tails: same blocks, [cols][nkk][cpc][n*9] nest, n = N/(cols*cpc)
    jobs["gu_gate_t"] = lambda: S2.pack_b_tail_from_blocks(
        w["feed_forward.w1"], 3840, 10240, 10240 // 32, cols=8, cpc=4)
    jobs["gu_up_t"] = lambda: S2.pack_b_tail_from_blocks(
        w["feed_forward.w3"], 3840, 10240, 10240 // 32, cols=8, cpc=4)
    jobs["dn_t"] = lambda: S2.pack_b_tail_from_blocks(
        w["feed_forward.w2"], 10240, 3840, 3840 // 16, cols=8, cpc=2)
    jobs["proj_t"] = lambda: S2.pack_b_tail_from_blocks(
        fused, 3840, 15360, 15360 // 32, cols=8, cpc=4)
    total = 0
    want = _want_tags()
    for tag, fn in jobs.items():
        if want is not None and tag not in want:
            continue
        b = np.ascontiguousarray(fn(), np.uint8)
        assert b.size == B_BYTES[tag], (tag, b.size, B_BYTES[tag])
        np.save(CACHE / f"{key}_{tag}.npy", b)
        total += b.size
    return total, time.perf_counter() - t0


# 8 payloads = 398 MB per layer: 30 layers is 11.9 GB, 4 layers 1.6 GB,
# 1 layer 398 MB.  The default is the small one; a miss costs one repack
# (0.65 s at the measured 612 MB/s).
CACHE_MAX = int(os.environ.get("T22_S2_CACHE_LAYERS", "4"))


def _cache_tags():
    """The payload tags this process caches (all of them unless T22_S2_TAGS)."""
    want = _want_tags()
    return [t for t in B_BYTES if want is None or t in want]


def _cached_keys():
    if not CACHE.is_dir():
        return []
    want = set(_cache_tags())
    keys = set()
    for p in CACHE.glob("layers.*_*.npy"):
        k, _, rest = p.name.partition("_")      # "layers.N" has no underscore
        if rest.endswith(".npy") and rest[:-4] in want:
            keys.add(k)
    # every key here was found through a wanted tag, so min() is never empty
    return sorted(keys,
                  key=lambda k: min((CACHE / f"{k}_{t}.npy").stat().st_mtime
                                    for t in want
                                    if (CACHE / f"{k}_{t}.npy").is_file()))


def ensure_layer(key, verbose=False):
    """Prewarm ONE layer if missing, then evict the oldest beyond CACHE_MAX.

    T22_S2_CACHE_LAYERS (default 30) bounds the disk cache: 8 payloads =
    398 MB per layer, so 30 layers is 11.9 GB and 4 layers is 1.6 GB.  A miss
    costs one repack (0.65 s at the measured 612 MB/s).
    """
    _want = _want_tags()
    _tags = [t for t in B_BYTES if _want is None or t in _want]
    need = [t for t in _tags if not (CACHE / f"{key}_{t}.npy").is_file()]
    if need:
        t0 = time.perf_counter()
        prewarm_b(key)
        # parent 2026-11-01: always log repacks -- they dominate a small-cache
        # run and must be subtracted before quoting a projection cost.
        print("[s2] repacked %s: %.1f MB in %.2f s"
              % (key, 398.1, time.perf_counter() - t0), flush=True)
    else:
        for t in _tags:
            p = CACHE / f"{key}_{t}.npy"
            if p.is_file():
                os.utime(p, None)
    keep = max(1, CACHE_MAX)
    ks = _cached_keys()
    for old in ks[:-keep]:
        for t in _tags:
            p = CACHE / f"{old}_{t}.npy"
            if p.is_file():
                try:
                    p.unlink()
                except OSError:
                    # parent fix: still mmapped by this process; skip eviction
                    pass
    return key


def ensure_cache(keys=None, verbose=True):
    """Prewarm only what is missing (safe to call from product startup)."""
    keys = keys or [f"layers.{i}" for i in range(30)]
    missing = [k for k in keys
               if not all((CACHE / f"{k}_{t}.npy").is_file() for t in _cache_tags())]
    if not missing:
        return {"missing": 0, "keys": len(keys), "seconds": 0.0}
    st = prewarm_ball(missing, verbose=verbose)
    st["missing"] = len(missing)
    return st


def prewarm_ball(keys=None, verbose=True):
    keys = keys or [f"layers.{i}" for i in range(30)]
    t0 = time.perf_counter()
    total = 0
    for k in keys:
        n, _ = prewarm_b(k)
        total += n
        if verbose:
            print(f"[s2] prewarm {k}: {n/1e6:.1f} MB "
                  f"({time.perf_counter()-t0:.1f} s elapsed)", flush=True)
    dt = time.perf_counter() - t0
    st = {"layers": len(keys), "bytes": int(total), "seconds": dt,
          "MB_per_s": total / 1e6 / dt if dt else None,
          "cache_dir": str(CACHE)}
    (OUT / "s2_prewarm.json").write_text(json.dumps(st, indent=2), encoding="utf-8")
    if verbose:
        print(f"[s2] prewarm done: {total/1e9:.2f} GB in {dt:.1f} s "
              f"({st['MB_per_s']:.0f} MB/s)")
    return st


# ---------------------------------------------------------------------------
# device handle
# ---------------------------------------------------------------------------
class S2Dev:
    """Owns the device, the lock, the image runners and the B cache."""

    def __init__(self, take_lock=True, verbose=False):
        self.verbose = verbose
        self.runners = {}
        self.n_dispatch = 0
        self._lock_fd = None
        self._b = {}
        self.tim = {}
        if take_lock:
            try:
                self._lock_fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL |
                                        os.O_WRONLY)
                os.write(self._lock_fd, ("owner=s2_dev pid=%d\n"
                                         % os.getpid()).encode())
                os.close(self._lock_fd)
                self._lock_fd = "held"
            except FileExistsError:
                try:
                    _txt = LOCK.read_text()
                except Exception:
                    _txt = ""
                if (("pid=%d" % os.getpid()) in _txt
                        or ("pid=%d" % os.getppid()) in _txt):
                    # parent fix 2026-11-01: the product process already holds
                    # the device (devkill/bigpipe took the lock); sharing it is
                    # correct, and we must not unlink it on close.
                    self._lock_fd = "reentrant"
                else:
                    # parent fix 2026-11-01: take over a lock whose owner is
                    # dead (a crashed run leaves the file behind and blocks
                    # every later window).
                    _m = __import__("re").search(r"pid=(\d+)", _txt)
                    _alive = False
                    if _m:
                        try:
                            os.kill(int(_m.group(1)), 0)
                            _alive = True
                        except OSError:
                            _alive = False
                    if not _alive:
                        print("[s2] taking over stale device.lock (%r)"
                              % (_txt.strip(),), flush=True)
                        try:
                            LOCK.unlink()
                        except OSError:
                            pass
                        self._lock_fd = os.open(str(LOCK), os.O_CREAT |
                                                os.O_EXCL | os.O_WRONLY)
                        os.write(self._lock_fd, ("owner=s2_dev pid=%d\n"
                                                 % os.getpid()).encode())
                        os.close(self._lock_fd)
                        self._lock_fd = "held"
                    else:
                        raise RuntimeError(
                            f"device.lock is held: {_txt.strip()!r}")

    # -- images -----------------------------------------------------------
    def runner(self, name):
        if name not in self.runners:
            shape = IMG[name][0]
            self.runners[name] = run_ffn.ImageRunner(shape).open()
        return self.runners[name]

    def dispatch(self, name, a_bytes, b_arr):
        """One dispatch: A (small) + B (mmapped payload) -> C.

        B is staged with run_staged (persistent BOs + mapped memcpy); the B
        bytes are the dominant host->device traffic (44-66 MB per dispatch,
        ~199 MB per layer), so its bandwidth is reported separately.
        """
        c_len = IMG[name][6]
        a_arr = (np.frombuffer(a_bytes, np.uint8)
                 if isinstance(a_bytes, (bytes, bytearray)) else a_bytes)
        b_arr = np.ascontiguousarray(b_arr)
        assert b_arr.nbytes == B_BYTES[_b_tag(name)], (name, b_arr.nbytes)
        engine = os.environ.get("S2_ENGINE", "inline")
        t0 = time.perf_counter()
        if engine == "class":
            r = self.runner(name)
            use_map = os.environ.get("S2_STAGE", "write") == "map"
            craw, walls, states = r.run_staged(a_arr, b_arr, c_len, runs=1,
                                               warmup=0, use_map=use_map)
            stage_ms = float(getattr(r, "stage_ms", 0.0))
            stage_path = getattr(r, "stage_path", None)
        else:
            # the 08:06 path: one registration + the verbatim BO sequence
            craw, walls, states, n_bo = run_ffn.run_inline(
                IMG[name][0], bytes(a_arr), bytes(b_arr), c_len, runs=1,
                warmup=0)
            stage_ms = float("nan")
            stage_path = "inline"
        if states and states[0] != run_ffn.ERT_COMPLETED:
            raise RuntimeError(f"{name}: state {states[0]}")
        rec = self.tim.setdefault(name, {"wall_ms": 0.0, "stage_ms": 0.0,
                                         "n": 0})
        rec["wall_ms"] += float(walls[0]) if walls else 0.0
        if stage_ms == stage_ms:
            rec["stage_ms"] += stage_ms
        rec["n"] += 1
        rec["stage_path"] = stage_path
        rec["engine"] = engine
        self.n_dispatch += 1
        return craw

    # -- B cache ----------------------------------------------------------
    def b(self, key, tag):
        k = (key, tag)
        if k not in self._b:
            p = CACHE / f"{key}_{tag}.npy"
            if not p.is_file():
                # parent fix 2026-11-01: drop the previous layer's mmaps before
                # asking ensure_layer to evict, otherwise Windows refuses to
                # unlink a file this process still maps (WinError 32).
                cur = getattr(self, "_cur_key", None)
                if cur is not None and cur != key:
                    self._b.clear()
                ensure_layer(key)
                self._cur_key = key
            self._b[k] = np.load(p, mmap_mode="r")
        return self._b[k]

    def close(self):
        try:
            self._b.clear()
        except Exception:
            pass
        if self._lock_fd == "held":
            try:
                LOCK.unlink()
            except FileNotFoundError:
                pass
            self._lock_fd = None


def _b_tag(img):
    return {"gu_main": "gu_gate", "gu_tail": "gu_gate_t",
            "dn_main": "dn", "dn_tail": "dn_t",
            "pj_main": "proj", "pj_tail": "proj_t"}[img]


# ---------------------------------------------------------------------------
# the two product calls
# ---------------------------------------------------------------------------
def _a_from_xn(xn):
    """(1056,3840) f32 -> (main A, tail A) the way the product ships it.

    The product hands us f32 xn; the equivalent byte path is
    pack_product_a -> convert_a (proven EXACT on the real product layout),
    which we assert against the direct pack when S2_FFN32_CHECK=1.
    """
    # parent fix 2026-11-01: pack_a/pack_a_tail already apply the same RNE
    # rounding, so the _bf16() round trip (4 passes over 16 MB, 34.5 ms) was
    # pure waste.  Verified byte-identical for main and tail, incl. -0.0,
    # denormals, 1e30 and NaN/inf payloads.
    xb = np.ascontiguousarray(np.asarray(xn, np.float32))
    rows = xb.shape[0]
    # parent fix 2026-11-01: pack_product_a -> convert_a is a round trip
    # (pack to the product layout, immediately unpack back) and cost 452 ms of
    # the 530 ms per call.  Go straight to the image payload; the equivalence
    # check below keeps the two definitions pinned when explicitly enabled.
    a_main = S2.pack_a(xb[:min(rows, 1024)])
    a_tail = S2.pack_a_tail(xb[1024:]) if rows > 1024 else None
    if os.environ.get("S2_FFN32_CHECK", "0") == "1":
        prod = S2.pack_product_a(xb)
        m2, t2 = S2.convert_a(prod, rows, xb.shape[1])
        assert np.array_equal(np.frombuffer(a_main, np.uint8),
                              np.frombuffer(m2, np.uint8)), "direct main != convert_a"
        if a_tail is not None:
            assert np.array_equal(np.frombuffer(a_tail, np.uint8),
                                  np.frombuffer(t2, np.uint8)), "direct tail != convert_a"
    return a_main, a_tail


def _c_to_rows(raw, name):
    """device C -> (rows, N) f32, rows = 1024 (main) or 32 (tail)."""
    _, K, N, n, ntiles, cpc, _ = IMG[name]
    if ntiles == 1:
        return S2.unpack_c_tail(raw, COLS, cpc, M_TILE, n)
    return S2.unpack_c(raw, COLS, ntiles, cpc, M_TILE, n)


def ffn32_run(dev, xn, key, tim=None, stages=None):
    """xn (1056, 3840) f32 -> y (1056, 3840) f32.  SIX dispatches.

    gate = gu(x, w1)   main + tail
    up   = gu(x, w3)   main + tail          <- the same image, a second pass
    y    = dn(swiglu)  main + tail
    stages (a dict) receives gate/up/h for the stage-by-stage diagnostic.
    """
    xn = np.asarray(xn, np.float32)
    assert xn.shape == (1056, D), xn.shape
    t0 = time.perf_counter()
    a_main, a_tail = _a_from_xn(xn)
    t_a = time.perf_counter() - t0

    def gate_or_up(tag):
        c_main = _c_to_rows(dev.dispatch("gu_main", a_main,
                                         dev.b(key, tag)[:]), "gu_main")
        # the tail image reads its OWN payload nest, not the main's
        c_tail = _c_to_rows(dev.dispatch("gu_tail", a_tail,
                                         dev.b(key, tag + "_t")[:]), "gu_tail")
        return np.vstack([c_main, c_tail])            # (1056, 10240)

    gate = gate_or_up("gu_gate")
    up = gate_or_up("gu_up")
    h = S2._bf16(gate * (1.0 / (1.0 + np.exp(-gate))) * up)
    if stages is not None:
        stages.update(gate=gate, up=up, h=h)

    dn_main_a = S2.pack_a(h[:1024])                   # K=10240
    dn_tail_a = S2.pack_a_tail(h[1024:])
    c_main = _c_to_rows(dev.dispatch("dn_main", dn_main_a,
                                     dev.b(key, "dn")[:]), "dn_main")
    c_tail = _c_to_rows(dev.dispatch("dn_tail", dn_tail_a,
                                     dev.b(key, "dn_t")[:]), "dn_tail")
    y = np.vstack([c_main, c_tail])
    if tim is not None:
        tim.setdefault("s2_ffn32", []).append(
            {"total_ms": 1000.0 * (time.perf_counter() - t0),
             "a_convert_ms": 1000.0 * t_a, "dispatch_ms": dict(dev.tim)})
    return y


def proj32_run(dev, x, key, tim=None):
    """x (1056, 3840) f32 -> {'q','k','v','o'} each (1056, 3840) f32."""
    x = np.asarray(x, np.float32)
    assert x.shape == (1056, D), x.shape
    t0 = time.perf_counter()
    a_main, a_tail = _a_from_xn(x)
    b = dev.b(key, "proj")[:]
    c_main = _c_to_rows(dev.dispatch("pj_main", a_main, b), "pj_main")
    c_tail = _c_to_rows(dev.dispatch("pj_tail", a_tail,
                                     dev.b(key, "proj_t")[:]), "pj_tail")
    allq = np.vstack([c_main, c_tail])                # (1056, 15360)
    if tim is not None:
        tim.setdefault("s2_proj32", []).append(
            {"total_ms": 1000.0 * (time.perf_counter() - t0),
             "dispatch_ms": dict(dev.tim)})
    out = {nm: allq[:, i * D:(i + 1) * D] for i, nm in
           enumerate(("q", "k", "v", "o"))}
    assert set(out) == {"q", "k", "v", "o"}, sorted(out)   # product convention
    return out


# ---------------------------------------------------------------------------
# smoke: one layer, device vs the CPU dry path
# ---------------------------------------------------------------------------


def smoke_image(name, runs=1, repeat=1):
    """ONE image through the s2_dev BO path, against its own npz payload.

    This is the ladder the parent asked for: the npz C_ref is the reference
    that verify_ffn already passed at exact 1.0, so an exact match here proves
    the BO path, the A bytes and the un-tile for that image -- nothing else.
    """
    tag = IMG[name][0]
    io = np.load(OUT / f"{tag}_io.npz")
    cref = io["C_ref"].astype(np.uint8)
    res = {"image": name, "shape": tag}
    dev = S2Dev()
    try:
        runs_detail = []
        for i in range(max(1, repeat)):
            craw = dev.dispatch(name, io["A_bytes"].tobytes(), io["B_bytes"])
            got_i = _c_to_rows(craw, name)
            ref_i = _c_to_rows(cref, name)
            runs_detail.append({"i": i, "exact": bool(np.array_equal(got_i, ref_i)),
                                "sha16": hashlib.sha256(
                                    craw.tobytes()).hexdigest()[:16]})
            if i == 0:
                craw_first, got, ref = craw, got_i, ref_i
        res["repeat"] = runs_detail
        craw, got, ref = craw_first, got, ref
        res["stage_path"] = dev.tim.get(name, {}).get("stage_path")
        got = _c_to_rows(craw, name)
        ref = _c_to_rows(cref, name)
        d = np.abs(got.astype(np.float64) - ref.astype(np.float64))
        res.update({"rows": int(got.shape[0]), "cols": int(got.shape[1]),
                    "exact": bool(np.array_equal(got, ref)),
                    "exact_frac": float(np.mean(got == ref)),
                    "maxabs": float(d.max()),
                    "corr": _corr(got, ref),
                    "C_sha16": hashlib.sha256(craw.tobytes()).hexdigest()[:16],
                    "var": dev.tim.get(name)})
    finally:
        dev.close()
    (OUT / f"smoke_img_{name}.json").write_text(json.dumps(res, indent=2),
                                                encoding="utf-8")
    print(json.dumps(res, indent=2)[:2000])
    return res




def smoke_gureal(key="layers.0", twice=False):
    """ONE gu_main dispatch with the REAL w1 payload + the chain's A.

    Compares against out/ref_gate_percall_layers0.npy, the per-call bf16
    model of the same product (corr 0.999669 vs f32, maxabs 1.9).  If the
    device is far from THIS, the difference is not accumulation rounding.
    """
    xin = (np.random.default_rng(3).standard_normal((1056, D))
           .astype(np.float32) * 0.5)
    a_main, _ = _a_from_xn(xin)
    ref = np.load(OUT / "ref_gate_percall_layers0.npy")
    res = {"image": "gu_main", "weights": "real w1", "ref": "per-call bf16"}
    dev = S2Dev()
    try:
        if twice:
            # two dispatches, DIFFERENT B (w1 then w3), each un-tiled and
            # compared right after its own dispatch -- this is the case that
            # --only --repeat 2 cannot see (same A/B every time).
            seq = []
            prev_sha = None
            for tag in ("gate", "up"):
                ref_t = np.load(OUT / f"ref_{tag}_percall_layers0.npy")[:1024]
                craw = dev.dispatch("gu_main", a_main,
                                    dev.b(key, f"gu_{tag}")[:])
                got = _c_to_rows(craw, "gu_main")
                sha = hashlib.sha256(got.tobytes()).hexdigest()[:16]
                seq.append({
                    "tag": tag, "corr": _corr(got, ref_t),
                    "maxabs": float(np.abs(got - ref_t).max()),
                    "ref_absmax": float(np.abs(ref_t).max()),
                    "sha_dev": sha,
                    "sha_ref": hashlib.sha256(ref_t.tobytes()).hexdigest()[:16],
                    "same_as_previous": (sha == prev_sha)})
                prev_sha = sha
            res["twice"] = seq
            res["corr"] = min(s["corr"] for s in seq)
        else:
            craw = dev.dispatch("gu_main", a_main, dev.b(key, "gu_gate")[:])
            got = _c_to_rows(craw, "gu_main")
            res.update({
                "corr": _corr(got, ref[:1024]),
                "maxabs": float(np.abs(got - ref[:1024]).max()),
                "ref_absmax": float(np.abs(ref[:1024]).max()),
                "sha_dev": hashlib.sha256(got.tobytes()).hexdigest()[:16],
                "sha_ref": hashlib.sha256(ref[:1024].tobytes()).hexdigest()[:16]})
        res.update({"n_dispatch": dev.n_dispatch,
                    "stage_path": dev.tim.get("gu_main", {}).get("stage_path")})
    finally:
        dev.close()
    (OUT / "smoke_gureal.json").write_text(json.dumps(res, indent=2),
                                           encoding="utf-8")
    print(json.dumps(res, indent=2)[:1500])
    return res




def _src_W(key, tag):
    """The true (K, N) weight from pack_bfp16.unpack -- the only decoder that
    knows the checkpoint's layout (a raw-byte decode is corr 3e-4, measured)."""
    sys.path.insert(0, str(ROOT))
    import pack_bfp16 as _pb
    li = int(key.split(".")[1])
    z = np.load(ROOT / "weights" / f"block{li}.npz", allow_pickle=False)
    if tag.startswith("proj"):
        parts = [_pb.unpack(z[f"{key}.attention.{n}.weight"], 3840, 3840)
                 for n in ("to_q", "to_k", "to_v", "to_out.0")]
        return np.ascontiguousarray(np.concatenate(parts, axis=0).T)  # (3840,15360)
    nm = {"gu_gate": "feed_forward.w1", "gu_up": "feed_forward.w3",
          "dn": "feed_forward.w2"}[tag.rstrip("_t")]
    K, N = (10240, 3840) if nm.endswith("w2") else (3840, 10240)
    return np.ascontiguousarray(np.asarray(
        _pb.unpack(z[f"{key}.{nm}.weight"], N, K), np.float32).T)


def verify_b(key="layers.0", samples=6, seed=0):
    """Rebuild W from every payload and compare it with the source W.

    The byte COUNT cannot catch a different nest (the tail-B trap) and a raw
    byte decode cannot catch the checkpoint's own layout.  This asserts the
    SEMANTICS: payload -> unpack -> W must equal pack_bfp16.unpack -> W.
    """
    src = {"gu_gate": (3840, 10240, 512, 8, 4),
           "gu_up": (3840, 10240, 512, 8, 4),
           "dn": (10240, 3840, 384, 8, 4),
           "proj": (3840, 15360, 768, 8, 4),
           "gu_gate_t": (3840, 10240, 320, 8, 4),
           "gu_up_t": (3840, 10240, 320, 8, 4),
           "dn_t": (10240, 3840, 240, 8, 2),
           "proj_t": (3840, 15360, 480, 8, 4)}
    res = {}
    for tag, (K, N, n, cols, cpc) in src.items():
        pay = np.load(CACHE / f"{key}_{tag}.npy")
        W = _src_W(key, tag)
        if tag.endswith("_t"):
            got = S2.unpack_b_tail(pay, K, N, n, cols=cols, cpc=cpc)
        else:
            got = S2.unpack_b(pay, K, N, n)
        d = float(np.abs(got - W).max())
        # 1-2 bf16 ulp is the re-encode (decode -> encode is not bit-exact at
        # ties); anything above that is a layout error
        res[tag] = {"n": int(n), "tail": tag.endswith("_t"),
                    "maxabs": d, "ok": bool(d <= 0.02)}
    return res


def _corr(a, b):
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    return float(np.corrcoef(a, b)[0, 1])


def smoke(what, key="layers.0", dry_only=False, runs=1):
    xin = (np.random.default_rng(3).standard_normal((1056, D))
           .astype(np.float32) * 0.5)
    res = {"what": what, "key": key, "dry_only": dry_only}
    dev = None
    try:
        if what == "ffn":
            w = _ffn_blocks(key)
            st_dev = {}
            if not dry_only:
                dev = S2Dev()
                y_dev = ffn32_run(dev, xin, key, stages=st_dev)
                np.save(OUT / "smoke_ffn_y_dev.npy", y_dev)
                res["device_ms"] = dev.tim
                res["n_dispatch"] = dev.n_dispatch
                if dev.n_dispatch != 6:
                    res["FAIL_n_dispatch"] = dev.n_dispatch
            st_ref = {}
            y_ref, info = S2.ffn32_dry(
                xin, w["feed_forward.w1"], w["feed_forward.w3"],
                w["feed_forward.w2"], key=key, stages=st_ref)
            if not dry_only:
                d = (y_dev - y_ref).astype(np.float64)
                # stage-by-stage: the first stage that is not ~1.0 is the bug
                res["stages"] = {}
                if os.environ.get("S2_DUMP_STAGES", "0") == "1":
                    for nm in ("gate", "up", "h"):
                        if nm in st_dev and nm in st_ref:
                            np.save(OUT / f"stage_{nm}_dev.npy", st_dev[nm])
                            np.save(OUT / f"stage_{nm}_ref.npy", st_ref[nm])
                for nm in ("gate", "up", "h"):
                    a, b = st_dev.get(nm), st_ref.get(nm)
                    if a is None or b is None:
                        continue
                    res["stages"][nm] = {
                        "corr": _corr(a, b),
                        "maxabs": float(np.abs(a - b).max()),
                        "ref_absmax": float(np.abs(b).max()),
                        "sha_dev": hashlib.sha256(a.tobytes()).hexdigest()[:16],
                        "sha_ref": hashlib.sha256(b.tobytes()).hexdigest()[:16]}
                res.update({
                    "corr": _corr(y_dev, y_ref),
                    "exact_frac": float(np.mean(
                        y_dev.view(np.uint32) == y_ref.view(np.uint32))),
                    "maxabs": float(np.abs(d).max()),
                    "ref_absmax": float(np.abs(y_ref).max()),
                    "rel_max": float(np.abs(d).max() /
                                     (np.abs(y_ref).max() + 1e-30))})
                # negative: A's K blocks permuted -> must go red
                perm = np.random.default_rng(9).permutation(D // 8)
                xp = xin.reshape(1056, D // 8, 8)[:, perm, :].reshape(1056, D)
                y_neg, _ = S2.ffn32_dry(
                    xp, w["feed_forward.w1"], w["feed_forward.w3"],
                    w["feed_forward.w2"], key=key)
                res["neg_korder_corr"] = _corr(y_dev, y_neg)
            res["layouts_exact"] = all(
                v for part in info["parts"].values()
                for v in ([part] if isinstance(part, dict) and "main" in part
                          else [part]))
        elif what == "proj":
            w = P2.load_proj_weights(key)
            if not dry_only:
                dev = S2Dev()
                od = proj32_run(dev, xin, key)
                np.savez(OUT / "smoke_proj_qkvo_dev.npz",
                         **{k: v for k, v in od.items()})
                res["device_ms"] = dev.tim
            orf, info = P2.proj32_dry(xin, w, mode="fused", n_tile=768)
            if not dry_only:
                assert set(od) == {"q", "k", "v", "o"}, sorted(od)
                assert set(orf) == {"q", "k", "v", "o"}, sorted(orf)
                cs = {nm: _corr(od[nm], orf[nm]) for nm in ("q", "k", "v", "o")}
                mx = max(float(np.abs(od[nm] - orf[nm]).max())
                         for nm in ("q", "k", "v", "o"))
                res.update({"corr": cs, "corr_min": min(cs.values()),
                            "maxabs": mx,
                            "ref_absmax": max(float(np.abs(orf[nm]).max())
                                              for nm in ("q", "k", "v", "o"))})
                perm = np.random.default_rng(9).permutation(D // 8)
                xp = xin.reshape(1056, D // 8, 8)[:, perm, :].reshape(1056, D)
                oneg, _ = P2.proj32_dry(xp, w, mode="fused", n_tile=768)
                res["neg_korder_corr"] = min(_corr(od[nm], oneg[nm])
                                             for nm in ("q", "k", "v", "o"))
            res["layouts_exact"] = all(
                v for part in info["parts"].values()
                for v in ([part] if isinstance(part, dict) and "main" in part
                          else [part]))
        else:
            raise SystemExit(f"unknown --smoke {what}")
    finally:
        if dev is not None:
            dev.close()
    (OUT / f"smoke_{what}.json").write_text(json.dumps(res, indent=2),
                                            encoding="utf-8")
    print(json.dumps(res, indent=2)[:3000])
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", choices=("ffn", "proj"), default=None)
    ap.add_argument("--only", default=None, choices=sorted(IMG),
                    help="single-image ladder: npz payload -> un-tile vs C_ref")
    ap.add_argument("--gureal", action="store_true",
                    help="one gu_main dispatch with real w1 vs the per-call ref")
    ap.add_argument("--twice", action="store_true",
                    help="with --gureal: two dispatches, w1 then w3, each read "
                         "and compared right after its own dispatch")
    ap.add_argument("--repeat", type=int, default=1,
                    help="--only: dispatch N times in this process")
    ap.add_argument("--key", default="layers.0")
    ap.add_argument("--dry-only", action="store_true")
    ap.add_argument("--prewarm", action="store_true")
    ap.add_argument("--verify-b", dest="verify_b", action="store_true",
                    help="rebuild W from every payload and compare (semantic "
                         "B assertion: byte counts cannot catch a nest swap)")
    ap.add_argument("--layers", type=int, default=30)
    args = ap.parse_args()
    if args.verify_b:
        r = verify_b(args.key)
        (OUT / "s2_verify_b.json").write_text(json.dumps(r, indent=2),
                                              encoding="utf-8")
        print(json.dumps(r, indent=2)[:2000])
        return 0 if all(v["ok"] for v in r.values()) else 1
    if args.prewarm:
        prewarm_ball([f"layers.{i}" for i in range(args.layers)])
        return 0
    if args.gureal:
        r = smoke_gureal(args.key, twice=args.twice)
        return 0 if r["corr"] >= 0.999 else 1
    if args.only:
        r = smoke_image(args.only, repeat=args.repeat)
        return 0 if r["exact"] else 1
    if args.smoke:
        r = smoke(args.smoke, args.key, args.dry_only)
        if r.get("dry_only"):
            return 0
        ok = (r.get("corr_min", r.get("corr", 0)) >= 0.999) and \
             (r.get("neg_korder_corr", 1) <= 0.9)
        return 0 if ok else 1
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())