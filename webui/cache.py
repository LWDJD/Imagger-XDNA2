# SPDX-License-Identifier: MIT
"""Per-resolution build-product cache.

The delivered M=96 images are immutable, so the 128 profile's cache is a
verified copy of exactly the files the device loop registers.  First prepare
copies 0.85 MB and hashes it; later starts only verify the manifest and are
instant.  The cache is what the bridge uses (attention dir + t13_run.T4_OUT),
so a job cannot silently fall back to a different build.

256/512 are intentionally NOT half-built.  They need six new device images
(FFN, 4 projections, IP) plus new permutations and host constants; that is a
design unit, not a compile flag.  The UI disables them and this module raises
ResolutionNotReady with the concrete reason.
"""
from __future__ import annotations

import collections
import hashlib
import io
import json
import os
import pathlib
import re
import shutil
import threading
import time

import numpy as np

from . import config, shapes

CACHE_VERSION = 2
MAX_PROFILES = 10
DISK_BUDGET_BYTES = 1 << 30          # build products only; refiner cache is shared/read-only

# T22 prodlarge: only {H}x{W} directories are build-product profiles.  The rest
# of webui/cache is infrastructure and must NEVER be counted or evicted:
#   build_logs/     compile/audit logs (build_shape writes them here)
#   numba*/         NUMBA_CACHE_DIR of the device subprocesses
#   schedules/      schedule.npz cache
#   npu_t21/        root-owned shim dir (not even listable for this user)
#   "128"           the PRE-T16 profile dir; README §13 still passes
#                   --cache-dir webui\cache\128 for the hand-run bridge
PROFILE_RE = re.compile(r"^(\d+)x(\d+)$")


class ResolutionNotReady(RuntimeError):
    pass


# T18: the images a non-96 build needs.  The projection stays the delivered
# t7_p5 (m=96, run in 96-row blocks); the ip image is compiled with ND = SEQ
# keys and the query-block height the L1 budget allows; the FFN is row-
# independent and is dispatched in blocks of FFN_M rows:
#   128 -> t18_m128, one 128-row block (15,264 B of insts: completes)
#   288 -> the DELIVERED t11_nh5, three 96-row blocks.  The M=288 single
#          dispatch (36,144 B of insts) does NOT complete on this device
#          (ERT_CMD_STATE_TIMEOUT, reproduced standalone), so the row-block
#          decomposition is the supported form -- and it is exact, because
#          every row of the FFN is computed independently.
# Perms come from t18_perm.py.
T18_IMAGES = {
    128: {"proj": "t7_p5", "ip": "t18_ip_m64nd128", "ffn": "t18_m128",
          "ip_mq": 64, "perm_q": "t18_perm_64x128.json", "seq": 128,
          "ffn_m": 128, "ffn_blocks": 1},
    288: {"proj": "t7_p5", "ip": "t18_ip_m16nd288", "ffn": config.FFN_IMAGE,
          "ip_mq": 16, "perm_q": "t18_perm_16x128.json", "seq": 288,
          "ffn_m": 96, "ffn_blocks": 3},
    # T20: 128x256 -> 8x16 patch grid -> 128 image tokens -> SEQ = 32+128 = 160.
    # The ip image is keyed on the full sequence (ND=160) with the query-block
    # height the L1 budget allows: 32*128*2 + 128*160/8*9 + 32*160*2 + 6144
    # = 8,192 + 23,040 + 10,240 + 6,144 = 57,616 <= 65,536  -> Mq=32, 5 blocks.
    # The FFN is row-independent: 2 x 96-row blocks of the delivered t11_nh5
    # (the T20 relaunch barrier sits between them; see t18_run.Dev.barrier).
    160: {"proj": "t7_p5", "ip": "t18_ip_m32nd160", "ffn": config.FFN_IMAGE,
          "ip_mq": 32, "perm_q": "t18_perm_32x128.json", "seq": 160,
          "ffn_m": 96, "ffn_blocks": 2},
}


def _specs_128():
    return [
        ("att", "t7_p5.xclbin", config.T5_OUT / "t7_p5.xclbin"),
        ("att", "t7_p5.bin", config.T5_OUT / "t7_p5.bin"),
        ("attip", "t7_ip.xclbin", config.T5_OUT / "t7_ip.xclbin"),
        ("attip", "t7_ip.bin", config.T5_OUT / "t7_ip.bin"),
        ("ffn", config.FFN_IMAGE + ".xclbin",
         config.T4_OUT / (config.FFN_IMAGE + ".xclbin")),
        ("ffn", config.FFN_IMAGE + ".bin",
         config.T4_OUT / (config.FFN_IMAGE + ".bin")),
        ("perm", "t3d_perm_96x128.json", config.IMAGE_DIR / "t3d_perm_96x128.json"),
        ("perm", "t3d_perm_96x64.json", config.IMAGE_DIR / "t3d_perm_96x64.json"),
    ]


def _specs_for(plan):
    """(specs, images) for one plan.  Empty specs = no build in this tree."""
    if plan.get("exec_mode") == "chunked":
        # T22 chunk_path: no image is built.  The cache copies exactly the
        # delivered M=96 artifacts the chunked driver registers: the
        # projection image, a dummy ip image (registered but never dispatched;
        # t18's Dev binds the projection image's 3-arg layout through has_ip),
        # and the product FFN image.
        return [
            ("att", "t7_p5.xclbin", config.T5_OUT / "t7_p5.xclbin"),
            ("att", "t7_p5.bin", config.T5_OUT / "t7_p5.bin"),
            ("attip", "t7_ip.xclbin", config.T5_OUT / "t7_ip.xclbin"),
            ("attip", "t7_ip.bin", config.T5_OUT / "t7_ip.bin"),
            ("ffn", config.FFN_IMAGE + ".xclbin",
             config.T4_OUT / (config.FFN_IMAGE + ".xclbin")),
            ("ffn", config.FFN_IMAGE + ".bin",
             config.T4_OUT / (config.FFN_IMAGE + ".bin")),
        ], {"proj": "t7_p5", "ip": "t7_ip", "ffn": config.FFN_IMAGE,
            "exec_mode": "chunked", "attention": "host-f32-torch"}
    m = int(plan["m"])
    if m == 96:
        return _specs_128(), {"proj": "t7_p5", "ip": "t7_ip",
                              "ffn": config.FFN_IMAGE}
    spec = T18_IMAGES.get(m)
    if spec is None:
        return [], {}
    out = [
        ("att", "t7_p5.xclbin", config.T5_OUT / "t7_p5.xclbin"),
        ("att", "t7_p5.bin", config.T5_OUT / "t7_p5.bin"),
        ("attip", spec["ip"] + ".xclbin",
         config.T5_OUT / (spec["ip"] + ".xclbin")),
        ("attip", spec["ip"] + ".bin", config.T5_OUT / (spec["ip"] + ".bin")),
        ("ffn", spec["ffn"] + ".xclbin",
         config.T4_OUT / (spec["ffn"] + ".xclbin")),
        ("ffn", spec["ffn"] + ".bin", config.T4_OUT / (spec["ffn"] + ".bin")),
        ("perm", "t3d_perm_96x128.json",
         config.IMAGE_DIR / "t3d_perm_96x128.json"),
        ("perm", "t3d_perm_96x64.json",
         config.IMAGE_DIR / "t3d_perm_96x64.json"),
        ("perm", spec["perm_q"], config.IMAGE_DIR / spec["perm_q"]),
    ]
    return out, {"proj": spec["proj"], "ip": spec["ip"], "ffn": spec["ffn"],
                 "perm_q": spec["perm_q"], "ip_mq": spec["ip_mq"],
                 "seq": spec["seq"]}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _refiner_ok():
    d = config.REFINER_CACHE
    names = ("noise_refiner.0", "noise_refiner.1",
             "context_refiner.0", "context_refiner.1")
    if not d.is_dir():
        return False, f"missing refiner cache {d}"
    for n in names:
        if not (d / n / "manifest.json").is_file():
            return False, f"missing {d / n / 'manifest.json'}"
    return True, ""


def _cache_dir(h, w=None):
    """Per-shape directory: webui/cache/{H}x{W} (T16 P0-5)."""
    if w is None:
        h, w = int(h), int(h)
    return config.CACHE_DIR / f"{int(h)}x{int(w)}"


def _lru_path():
    return config.CACHE_DIR / "lru.json"


def _lru_load():
    try:
        return json.loads(_lru_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def _lru_save(d):
    """Write lru.json atomically (a half-written file must not lose the LRU)."""
    try:
        _lru_path().parent.mkdir(parents=True, exist_ok=True)
        tmp = _lru_path().with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d, indent=1), encoding="utf-8")
        tmp.replace(_lru_path())
    except Exception:  # noqa: BLE001
        pass


def _lru_touch(key):
    d = _lru_load()
    d[key] = time.time()
    _lru_save(d)
    return d


def profile_dirs():
    """{key: path} for every build-product profile directory in the cache.

    This is the LRU's inventory.  T22 prodlarge replaced "iterate lru.json"
    with "scan the cache directory": the spectrum 64..2048 can create profiles
    that the bookkeeping never saw (a hand copy, an interrupted build, a
    checkout from another machine), and a budget that only counts remembered
    profiles is not a budget.  Only PROFILE_RE names qualify -- build_logs,
    numba*, schedules, npu_t21 and the legacy "128" dir are infrastructure.
    """
    out = {}
    try:
        entries = list(config.CACHE_DIR.iterdir())
    except OSError:
        return out
    for p in entries:
        if not PROFILE_RE.match(p.name):
            continue
        try:
            if p.is_dir():
                out[p.name] = p
        except OSError:
            continue
    return out


def _dir_size(p):
    try:
        return sum(x.stat().st_size for x in p.rglob("*") if x.is_file())
    except OSError:
        return 0


_USAGE_MEMO = {"t": 0.0, "val": None}
_USAGE_TTL = 1.0        # /api/status polls every 5 s; 10 rows share one walk


def _usage_invalidate():
    _USAGE_MEMO["t"] = 0.0
    _USAGE_MEMO["val"] = None


def usage(refresh=False):
    """Measured cache footprint vs the LRU contract (T22 prodlarge).

    Reported per profile row so /api/status, /api/shapes and /api/cache all
    answer "how full is the cache" without a new endpoint.  The binding limit
    on this tree is the PROFILE COUNT, not the byte budget: one profile is
    ~1 MB, so 10 profiles use ~6 MiB of the 1 GiB budget -- a 2048x2048
    profile is the same order (its images are xclbins/permls, not weights).
    """
    now = time.monotonic()
    if not refresh and _USAGE_MEMO["val"] is not None \
            and (now - _USAGE_MEMO["t"]) < _USAGE_TTL:
        return _USAGE_MEMO["val"]
    inv = profile_dirs()
    sizes = {k: _dir_size(p) for k, p in inv.items()}
    lru = _lru_load()
    total = sum(sizes.values())
    non = []
    try:
        names = sorted(p.name for p in config.CACHE_DIR.iterdir())
    except OSError:
        names = []
    for n in names:
        if not PROFILE_RE.match(n):
            non.append(n)
    val = {
        "dir": config.rel_to_npu(config.CACHE_DIR),
        "profiles": len(inv),
        "max_profiles": MAX_PROFILES,
        "bytes": total,
        "disk_budget_bytes": DISK_BUDGET_BYTES,
        "budget_used_pct": round(100.0 * total / DISK_BUDGET_BYTES, 2),
        "lru_entries": len(lru),
        "lru_dead_entries": sum(1 for k in lru if k not in inv),
        "non_profile_dirs": non,
        "te": te_stats(),
        "largest": [{"key": k, "bytes": v} for k, v in
                    sorted(sizes.items(), key=lambda kv: -kv[1])[:3]],
    }
    _USAGE_MEMO.update({"t": now, "val": val})
    return val


def _lru_evict(keep_key, max_profiles=MAX_PROFILES,
               budget=DISK_BUDGET_BYTES, log=None):
    """Delete the oldest profile dirs beyond the count / disk budget.

    Inventory-driven (see profile_dirs): counted even when lru.json never saw
    the profile, and unknown recency sorts oldest.  Entries whose directory
    disappeared are pruned, so lru.json cannot grow without bound as the
    spectrum is explored.  Non-profile dirs are never touched.
    """
    d = _lru_touch(keep_key)
    inv = profile_dirs()
    for key in [k for k in d if k not in inv]:
        d.pop(key, None)
    rows = sorted((float(d.get(key, 0.0)), key, p, _dir_size(p))
                  for key, p in inv.items() if key != keep_key)
    keep_size = _dir_size(inv[keep_key]) if keep_key in inv else 0
    removed = []
    count = len(rows) + (1 if keep_key in inv else 0)
    total = sum(x[3] for x in rows) + keep_size
    while rows and (count > max_profiles or total > budget):
        _ts, key, p, size = rows.pop(0)
        try:
            shutil.rmtree(p)
            removed.append(key)
            d.pop(key, None)
            count -= 1
            total -= size
            if log:
                log(f"[cache] LRU evict {key} ({size/1024:.0f} KB)")
        except Exception:  # noqa: BLE001
            pass
    _lru_save(d)
    return removed


def _load_manifest(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _manifest_valid(d, man):
    if not man or int(man.get("version", 0)) != CACHE_VERSION:
        return False, "manifest version/missing"
    files = man.get("files") or []
    if not files:
        return False, "manifest has no files"
    for rec in files:
        p = d / rec["name"]
        if not p.is_file():
            return False, f"cached file missing: {p.name}"
        if p.stat().st_size != int(rec["size"]):
            return False, f"cached size changed: {p.name}"
        if _sha256(p) != rec["sha256"]:
            return False, f"cached hash changed: {p.name}"
    return True, ""


def _write_shape_package(d, plan):
    """Materialise the chunk_path shape package (host constants + recipe).

    The driver derives RoPE cos/sin from rope_ids at startup (the same
    t3_real.rope_cos_sin the delivered shapes use), and SBG rows from the
    per-layer modulation; the package persists the O(M) geometry inputs and
    the learned pad token so the shape is reproducible without re-deriving
    where its constants came from.  Nothing here is weights-dependent except
    x_pad_token (one 3,840-vector lifted verbatim from the delivered
    host-tensor npz).
    """
    b = shapes.budget(int(plan["height"]), int(plan["width"]))
    cap = np.zeros((shapes.CAP_TOKENS, 3), np.int64)
    cap[:, 0] = np.arange(1, 1 + shapes.CAP_TOKENS)
    gh, gw = b["patch_grid"]
    zz, yy, xx = np.meshgrid(np.array([33], np.int64),
                             np.arange(gh, dtype=np.int64),
                             np.arange(gw, dtype=np.int64), indexing="ij")
    img = np.stack([zz.reshape(-1), yy.reshape(-1), xx.reshape(-1)], 1)
    pad = max(0, int(b["padded_tokens"]) - int(b["image_tokens"]))
    img_pad = (np.concatenate([img, np.zeros((pad, 3), np.int64)], 0)
               if pad else img)
    rope = d / "rope_ids.npz"
    np.savez(rope, cap=cap, img=img, img_padded=img_pad)
    xtok = None
    src = config.IMAGE_DIR / "t3real_host_tensors.npz"
    try:
        z = np.load(src)
        if "x_pad_token" in z:
            xtok = np.asarray(z["x_pad_token"], np.float32).reshape(-1)
    except Exception:
        xtok = None
    files = [{"name": rope.name, "size": rope.stat().st_size,
              "sha256": _sha256(rope), "source": str(src),
              "group": "shape_package"}]
    if xtok is not None:
        p = d / "x_pad_token.npy"
        np.save(p, xtok)
        files.append({"name": p.name, "size": p.stat().st_size,
                      "sha256": _sha256(p), "source": str(src),
                      "group": "shape_package"})
    meta = {
        "version": 1,
        "shape": plan["key"],
        "requested": {"height": int(plan.get("requested_height", plan["height"])),
                      "width": int(plan.get("requested_width", plan["width"]))},
        "run": {"height": int(plan["height"]), "width": int(plan["width"])},
        "m": int(plan["m"]),
        "image_tokens": int(b["image_tokens"]),
        "padded_tokens": int(b["padded_tokens"]),
        "pad_rows": int(pad),
        "patch_grid": [int(gh), int(gw)],
        "latent": list(b["latent"]),
        "exec_mode": "chunked",
        "host_constants": {
            "rope_ids": {"file": rope.name, "cap_shape": list(cap.shape),
                         "img_shape": list(img.shape),
                         "img_padded_shape": list(img_pad.shape),
                         "cos_sin": "t3_real.rope_cos_sin(ids) at startup"},
            "sbg": "row=[mod[2]-1, 0, ffn_norm1] tiled to FFN_M (make_sbg)",
            "walk": "x_image64 / pack_blocked / unpack_bytes (t8_codec)",
            "x_pad_token": ({"file": "x_pad_token.npy",
                             "shape": list(xtok.shape),
                             "sha256": next((f["sha256"] for f in files
                                             if f["name"] == "x_pad_token.npy"), "")}
                            if xtok is not None else {"missing": str(src)}),
            "pe_alignment": ("cap 32 rows; image rows padded to padded_tokens "
                             "with x_pad_token at pos (0,0,0); caption follows"),
        },
        "images": plan.get("images", []),
        "estimate": plan.get("chunk_estimate"),
        "note": ("T22 chunk_path shape package; generated deterministically "
                 "from (H,W).  No device image is compiled for this shape."),
    }
    mp = d / "shape_package.json"
    mp.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    files.append({"name": mp.name, "size": mp.stat().st_size,
                  "sha256": _sha256(mp), "source": "webui/cache.py",
                  "group": "shape_package"})
    return files


def ensure_shape(h, w, log=None, allow_unbuilt=False):
    """Return cache info for one (H, W) shape, creating/verifying its copy.

    A shape that has no device build raises ResolutionNotReady with the precise
    build plan (six images + permutations + host constants).  It never falls
    back to another shape's images (no silent upscale / downscale).
    """
    plan = shapes.resolve(h, w)
    key = plan["key"]
    if not plan["ready"] and not allow_unbuilt \
            and not plan.get("chunked_runnable"):
        # T22 prodlarge: one sentence in the 512² spec's shape (why + exact
        # arithmetic), the itemised plan second -- never a generic refusal.
        raise ResolutionNotReady(
            f"{key} (M={plan['m']}) 未构建：{plan.get('reason', '')} "
            + "构建计划: " + "; ".join(plan.get("missing", [])))
    ok, why = _refiner_ok()
    if not ok:
        raise ResolutionNotReady(
            f"refiner cache is required for the delivered loop: {why}")

    d = _cache_dir(h, w)
    man_path = d / "manifest.json"
    man = _load_manifest(man_path)
    if man:
        valid, why = _manifest_valid(d, man)
        if valid:
            _lru_touch(key)
            if log:
                log(f"[cache] {key}: HIT {d} ({len(man['files'])} files, "
                    f"built {man.get('created','?')})")
            return {"shape": key, "height": h, "width": w, "m": plan["m"],
                    "resolution": (h if h == w else None), "dir": str(d),
                    "cached": True, "created": man.get("created"),
                    "files": man["files"], "seconds": 0.0}
        if log:
            log(f"[cache] {key}: stale ({why}); rebuilding copy")

    # M=96 profiles reuse the delivered images; T18's M=128/288 profiles have
    # their own FFN/ip builds.  Anything else is refused, never faked.
    specs, imgs = _specs_for(plan)
    if not specs:
        raise ResolutionNotReady(
            f"{key} has no source spec; refusing to fake it")
    t0 = time.perf_counter()
    d.mkdir(parents=True, exist_ok=True)
    files = []
    missing = []
    for _group, name, src in specs:
        src = pathlib.Path(src)
        if not src.is_file():
            missing.append(str(src))
            continue
        dst = d / name
        shutil.copy2(src, dst)
        files.append({"name": name, "size": dst.stat().st_size,
                      "sha256": _sha256(dst), "source": str(src),
                      "source_size": src.stat().st_size})
    if missing:
        raise ResolutionNotReady(
            "delivered M=96 build products are missing; the cache cannot be "
            "created without them: " + "; ".join(missing))
    if plan.get("exec_mode") == "chunked":
        pkg = _write_shape_package(d, plan)
        files.extend(pkg)
    man = {
        "version": CACHE_VERSION,
        "shape": key,
        "height": h,
        "width": w,
        "m": plan["m"],
        "pixels": h,
        "created": config.now_iso(),
        "note": ("Verified copy of the device images for this shape. "
                 "The driver's T4_OUT and --att-dir point here."),
        "images": imgs,
        "files": files,
        "refiner_cache": str(config.REFINER_CACHE),
    }
    tmp = man_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(man, indent=1), encoding="utf-8")
    tmp.replace(man_path)
    sec = time.perf_counter() - t0
    _usage_invalidate()
    evicted = _lru_evict(key, log=log)
    _usage_invalidate()
    if log:
        u = usage(refresh=True)
        log(f"[cache] {key}: PREPARED {d} ({len(files)} files, "
            f"{sum(f['size'] for f in files)/1024:.0f} KB) in {sec:.2f} s; "
            f"profiles {u['profiles']}/{u['max_profiles']}, "
            f"{u['bytes']/1048576:.2f} MiB / "
            f"{u['disk_budget_bytes']/1048576:.0f} MiB budget"
            + (f"; LRU evicted {', '.join(evicted)}" if evicted else ""))
    return {"shape": key, "height": h, "width": w, "m": plan["m"],
            "resolution": (h if h == w else None), "dir": str(d),
            "cached": False, "created": man["created"], "files": files,
            "seconds": sec, "evicted": evicted}


def ensure(resolution=128, log=None):
    """Backward-compatible square entry point."""
    return ensure_shape(int(resolution), int(resolution), log)


def profile_status(h, w):
    plan = shapes.resolve(h, w)
    d = _cache_dir(h, w)
    man = _load_manifest(d / "manifest.json")
    valid, why = (False, "not prepared")
    if man:
        valid, why = _manifest_valid(d, man)
    plan["cached"] = bool(valid)
    plan["cache_dir"] = config.rel_to_npu(d)
    plan["cache_detail"] = ("ready" if valid else why)
    plan["built"] = man.get("created") if valid else None
    plan["bytes"] = sum(int(x["size"]) for x in man["files"]) if valid else 0
    # T22 prodlarge: how full the cache is, on every row (one walk, memoized
    # for 1 s) so the UI can show "n/10 profiles, X MiB / 1 GiB" without a new
    # endpoint.
    plan["cache_usage"] = usage()
    return plan


def status():
    """The delivered profiles plus every preset (shape arithmetic, no build)."""
    out = []
    for value in sorted(config.RESOLUTIONS):
        meta = dict(config.RESOLUTIONS[value])
        meta["value"] = value
        plan = profile_status(value, value)
        if not plan.get("ready"):
            # shapes.resolve() only knows the generic "no gated build" sentence.
            # For an un-built fixed profile the specific reason in
            # config.RESOLUTIONS is the one the UI must show (512: M=1056 needs
            # a new ip kernel), so keep it instead of letting the generic win.
            plan.pop("reason", None)
        meta.update(plan)
        out.append(meta)
    return out


def shapes_status():
    out = []
    for p in shapes.list_shapes():
        out.append(profile_status(p["height"], p["width"]))
    return out


def clean(h=None, w=None):
    """Delete one shape's cache copy (or the whole cache).  Images untouched.

    T22 prodlarge: the LRU bookkeeping is pruned at the same time, so a deleted
    profile cannot leave a phantom entry behind (that is what made lru.json
    grow without bound once the spectrum started producing shapes).
    """
    removed = []
    if h is None:
        # T22 te cache: the whole-cache clean also drops the TE result cache
        # (its bytes live under CACHE_DIR/te) and the in-process copies.
        try:
            te_clear()
        except Exception:  # noqa: BLE001
            pass
        targets = [config.CACHE_DIR]
    else:
        targets = [_cache_dir(int(h), int(w if w is not None else h))]
    for d in targets:
        if d.exists():
            shutil.rmtree(d)
            removed.append(str(d))
    _usage_invalidate()
    lru = _lru_load()
    inv = profile_dirs()
    dead = [k for k in lru if k not in inv]
    if dead:
        for k in dead:
            lru.pop(k, None)
        _lru_save(lru)
    return removed


# --------------------------------------------------------------- T22 TE cache
# Why this exists
# ---------------
# A 128x128 job's textenc bucket measured 8.4-11.7 s over 108 jobs on this host
# (webui/jobs/*/job.json, 2026-10-18..21), of which 7.0-7.6 s is
# llama_load_model_from_file of the same 8.05 GB GGUF (T21 2a-a) and the rest is
# process/DLL start plus ~2 s of decode.  The page is used in runs of the same
# prompt, so most of that is pure repetition.  The resident encoder is NOT the
# answer on this host (6.74 GB of commit; T21 2a-a measured the NEXT job's
# device loop flipping to --no-cache-all, a 3x net loss), so the repetition is
# removed at the RESULT level and the model is never loaded on a hit.
#
# What is cacheable (and why the key is what it is)
# ------------------------------------------------
# pe is hidden_states[-2] of a fixed chat-templated prompt through a fixed
# model at a fixed layer tap.  It does NOT depend on height/width/steps/seed/
# batch -- the job history proves it: "a mountain lake at sunrise" has one pe
# sha256 (90514d08...) at 128x128, 144x128, 128x256, 256x256 and 64x256, while
# a different prompt has a different one.  It DOES depend on the backend: the
# SAME prompt on the CPU fallback is e6642367..., so the backend is in the key.
# The key is sha256 over canonical JSON:
#     prompt   the exact chat-templated string the encoder reads
#     backend  concrete gpu|cpu (never "auto")
#     lid      layer tap (35 -> hidden_states[-2]); fixed by te_worker
#     gguf     (path, size, mtime_ns) of the 8.05 GB blob
#     worker   (size, mtime_ns) of webui/te_worker.py    -- ctx/ABI constants
#     driver   (size, mtime_ns) of imaggers/textenc/encode_api.py -- N_CTX
#     v        TE_CACHE_VERSION
# A rebuilt GGUF, an edited worker/driver or a payload-format change therefore
# MISSES; no stale hit can survive an encoder change.
#
# Storage
# -------
# webui/cache/te/<key>.npz is the BYTE-EXACT npz the first run wrote (pe,
# tokens, prompt, layer), so a hit materialises a bit-identical per-job
# artifact.  <key>.json is the manifest (sha256 of the bytes + of the pe array,
# encoder stamps, prompt, tokens, last_used).  index.json is the LRU order.
# The in-process layer (TE_CACHE_MEM_ENTRIES, default 8) serves consecutive
# hits without touching the disk.  Capacity is capped by BOTH entry count and
# bytes; oldest last_used goes first.  Every entry is re-verified (size +
# sha256 + pe shape + pe hash) on load: a damaged file is deleted and counted
# as a miss, never returned.
TE_CACHE_VERSION = 1
TE_CACHE_DIR = pathlib.Path(
    os.environ.get("ZIMAGE_TE_CACHE_DIR") or (config.CACHE_DIR / "te"))
TE_CACHE_MAX_ENTRIES = max(1, int(os.environ.get("ZIMAGE_TE_CACHE_ENTRIES", "64")))
TE_CACHE_MAX_BYTES = max(1 << 20, int(
    float(os.environ.get("ZIMAGE_TE_CACHE_MB", "256")) * float(1 << 20)))
TE_CACHE_MEM_ENTRIES = max(1, int(
    os.environ.get("ZIMAGE_TE_CACHE_MEM_ENTRIES", "8")))
TE_CACHE_LID = 35            # te_worker tap: output of block 34 == hidden[-2]

_TE_LOCK = threading.RLock()
_TE_MEM = collections.OrderedDict()
_TE_STATS = {"hits": 0, "misses": 0, "stores": 0, "evicted": 0,
             "invalid": 0, "errors": 0, "last_hit": "", "last_miss": ""}


def _te_npz_path(key):
    return TE_CACHE_DIR / (key + ".npz")


def _te_meta_path(key):
    return TE_CACHE_DIR / (key + ".json")


def _te_index_path():
    return TE_CACHE_DIR / "index.json"


def _te_index_load():
    try:
        d = json.loads(_te_index_path().read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        d = None
    if not isinstance(d, dict) or int(d.get("version") or 0) != TE_CACHE_VERSION:
        return {"version": TE_CACHE_VERSION, "entries": {}}
    if not isinstance(d.get("entries"), dict):
        d["entries"] = {}
    return d


def _te_index_save(idx):
    try:
        TE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        p = _te_index_path()
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(idx, indent=1, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(p)
    except Exception:  # noqa: BLE001
        pass


def _te_index_touch(key, ts=None, hits_inc=0):
    idx = _te_index_load()
    rec = idx["entries"].setdefault(key, {})
    rec["last_used"] = float(ts if ts is not None else time.time())
    if hits_inc:
        rec["hits"] = int(rec.get("hits") or 0) + int(hits_inc)
    _te_index_save(idx)


def _stat_id(p):
    try:
        st = os.stat(str(p))
    except OSError:
        return None
    return {"path": str(pathlib.Path(p)), "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns)}


def te_key(prompt_text, backend, gguf=None):
    """Cache key for one encoded prompt.  None if the encoder is not present.

    prompt_text must be the EXACT string the encoder reads (the
    textenc.render_chat output), and backend the CONCRETE backend the bytes
    belong to -- "auto" is not a key, because gpu and cpu pe differ.
    """
    gguf = pathlib.Path(gguf) if gguf else config.find_te_gguf()
    if gguf is None or not pathlib.Path(gguf).is_file():
        return None
    gid = _stat_id(gguf)
    if gid is None:
        return None
    parts = {
        "v": TE_CACHE_VERSION,
        "prompt": prompt_text,
        "backend": str(backend),
        "lid": TE_CACHE_LID,
        "gguf": gid,
        "worker": _stat_id(config.TE_WORKER),
        "driver": _stat_id(config.NPU / "imaggers" / "textenc" / "encode_api.py"),
    }
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))
    return {"key": hashlib.sha256(blob.encode("utf-8")).hexdigest(),
            "parts": parts}


def _te_pe_of(data):
    with np.load(io.BytesIO(data)) as z:
        if "pe" not in z:
            raise ValueError("no pe array")
        pe = np.asarray(z["pe"], np.float32)
    if pe.ndim != 3 or pe.shape[0] != 1 or pe.shape[2] != 2560:
        raise ValueError("pe shape %r (expected (1,L,2560))" % (pe.shape,))
    return pe


def _te_read_entry(key):
    """(entry, why) from disk; entry is None when the file cannot be trusted."""
    try:
        meta = json.loads(_te_meta_path(key).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None, "no manifest"
    if int(meta.get("version") or 0) != TE_CACHE_VERSION:
        return None, "manifest version"
    if meta.get("key") != key:
        return None, "manifest key"
    try:
        data = _te_npz_path(key).read_bytes()
    except OSError:
        return None, "npz missing"
    if int(meta.get("bytes") or -1) != len(data):
        return None, "size changed"
    if meta.get("sha256") != hashlib.sha256(data).hexdigest():
        return None, "sha256 changed"
    try:
        pe = _te_pe_of(data)
    except Exception as exc:  # noqa: BLE001
        return None, "npz invalid: %s" % exc
    pe_sha = hashlib.sha256(
        np.ascontiguousarray(pe, np.float32).tobytes()).hexdigest()
    if meta.get("pe_sha256") != pe_sha:
        return None, "pe sha256 changed"
    meta["tokens"] = int(pe.shape[1])
    return {"key": key, "meta": meta, "bytes": data, "pe_sha256": pe_sha}, ""


def _te_invalidate(key, why=""):
    with _TE_LOCK:
        _TE_MEM.pop(key, None)
        _TE_STATS["invalid"] += 1
    for p in (_te_npz_path(key), _te_meta_path(key)):
        try:
            p.unlink()
        except OSError:
            pass
    idx = _te_index_load()
    if idx["entries"].pop(key, None) is not None:
        _te_index_save(idx)


def te_lookup(prompt_text, backend, gguf=None, log=None):
    """Return {hit, source, key, bytes, meta, pe_sha256} or None.

    Memory first, then the disk copy; a disk hit re-verifies every byte it
    returns and re-populates the memory layer.  Never raises for a damaged
    cache: it deletes the entry and reports a miss.
    """
    info = te_key(prompt_text, backend, gguf)
    if info is None:
        return None
    key = info["key"]
    now = time.time()
    with _TE_LOCK:
        ent = _TE_MEM.get(key)
        if ent is not None:
            _TE_MEM.move_to_end(key)
            _TE_STATS["hits"] += 1
            _TE_STATS["last_hit"] = key
            _te_index_touch(key, now, hits_inc=1)
            if log:
                log("[te-cache] mem hit %s (%d B)" % (key[:12], ent["bytes_len"]))
            return {"hit": True, "source": "mem", "key": key,
                    "bytes": ent["bytes"], "meta": ent["meta"],
                    "pe_sha256": ent["pe_sha256"]}
    ent, why = _te_read_entry(key)
    if ent is None:
        with _TE_LOCK:
            _TE_STATS["misses"] += 1
            _TE_STATS["last_miss"] = "%s:%s" % (key[:12], why)
        if why not in ("no manifest", "npz missing"):
            _te_invalidate(key, why)
        if log:
            log("[te-cache] miss %s (%s)" % (key[:12], why))
        return None
    with _TE_LOCK:
        _TE_MEM[key] = {"bytes": ent["bytes"], "meta": ent["meta"],
                        "pe_sha256": ent["pe_sha256"],
                        "bytes_len": len(ent["bytes"])}
        while len(_TE_MEM) > TE_CACHE_MEM_ENTRIES:
            _TE_MEM.popitem(last=False)
        _TE_STATS["hits"] += 1
        _TE_STATS["last_hit"] = key
        _te_index_touch(key, now, hits_inc=1)
    if log:
        log("[te-cache] disk hit %s (%d B)" % (key[:12], len(ent["bytes"])))
    return {"hit": True, "source": "disk", "key": key, "bytes": ent["bytes"],
            "meta": ent["meta"], "pe_sha256": ent["pe_sha256"]}


def te_store(prompt_text, backend, npz_path, gguf=None, log=None):
    """Store the npz a real encode just wrote; returns the entry or None."""
    info = te_key(prompt_text, backend, gguf)
    if info is None:
        return None
    key = info["key"]
    try:
        data = pathlib.Path(npz_path).read_bytes()
        pe = _te_pe_of(data)
    except Exception as exc:  # noqa: BLE001
        with _TE_LOCK:
            _TE_STATS["errors"] += 1
        if log:
            log("[te-cache] not stored: %s" % exc)
        return None
    pe_sha = hashlib.sha256(
        np.ascontiguousarray(pe, np.float32).tobytes()).hexdigest()
    now = time.time()
    meta = {
        "version": TE_CACHE_VERSION,
        "key": key,
        "created": config.now_iso(),
        "created_ts": now,
        "last_used": now,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "pe_sha256": pe_sha,
        "tokens": int(pe.shape[1]),
        "prompt": prompt_text,
        "prompt_sha256": hashlib.sha256(
            prompt_text.encode("utf-8")).hexdigest(),
        "backend": str(backend),
        "lid": TE_CACHE_LID,
        "gguf": info["parts"]["gguf"],
        "worker": info["parts"]["worker"],
        "driver": info["parts"]["driver"],
        "hits": 0,
    }
    try:
        TE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _te_npz_path(key).with_name(key + ".npz.tmp")
        tmp.write_bytes(data)
        tmp.replace(_te_npz_path(key))
        tmpm = _te_meta_path(key).with_name(key + ".json.tmp")
        tmpm.write_text(json.dumps(meta, indent=1, ensure_ascii=False),
                        encoding="utf-8")
        tmpm.replace(_te_meta_path(key))
    except Exception as exc:  # noqa: BLE001
        with _TE_LOCK:
            _TE_STATS["errors"] += 1
        if log:
            log("[te-cache] write failed: %s" % exc)
        return None
    idx = _te_index_load()
    idx["entries"][key] = {"last_used": now, "bytes": len(data),
                           "created": meta["created"],
                           "prompt_sha256": meta["prompt_sha256"],
                           "backend": meta["backend"], "tokens": meta["tokens"],
                           "hits": 0}
    _te_index_save(idx)
    with _TE_LOCK:
        _TE_MEM[key] = {"bytes": data, "meta": meta, "pe_sha256": pe_sha,
                        "bytes_len": len(data)}
        while len(_TE_MEM) > TE_CACHE_MEM_ENTRIES:
            _TE_MEM.popitem(last=False)
        _TE_STATS["stores"] += 1
    _te_prune_stale(log=log)
    _te_evict(keep_key=key, log=log)
    return {"key": key, "bytes": len(data), "tokens": int(pe.shape[1]),
            "pe_sha256": pe_sha, "meta": meta, "source": "store"}


def _te_prune_stale(log=None):
    """Delete entries whose encoder stamps no longer match the live files.

    A rebuilt GGUF / edited worker or driver changes the key, so such entries
    are unreachable anyway; pruning keeps the LRU budget for live entries.
    """
    gguf = config.find_te_gguf()
    if gguf is None or not pathlib.Path(gguf).is_file():
        return []
    cur = {"gguf": _stat_id(gguf),
           "worker": _stat_id(config.TE_WORKER),
           "driver": _stat_id(config.NPU / "imaggers" / "textenc"
                              / "encode_api.py")}
    removed = []
    try:
        metas = sorted(TE_CACHE_DIR.glob("*.json"))
    except OSError:
        return removed
    for mp in metas:
        if mp.name == "index.json":
            continue
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        key = str(meta.get("key") or mp.stem)
        if int(meta.get("version") or 0) != TE_CACHE_VERSION:
            _te_invalidate(key, "version")
            removed.append(key)
            continue
        if any(meta.get(bucket) != cur[bucket]
               for bucket in ("gguf", "worker", "driver")):
            _te_invalidate(key, "encoder changed")
            removed.append(key)
            if log:
                log("[te-cache] prune %s (encoder changed)" % key[:12])
    return removed


def _te_evict(keep_key=None, log=None):
    """Enforce the entry-count and byte budgets, oldest last_used first."""
    idx = _te_index_load()
    entries = idx["entries"]
    files = {}
    try:
        for p in TE_CACHE_DIR.glob("*.npz"):
            try:
                files[p.stem] = int(p.stat().st_size)
            except OSError:
                pass
    except OSError:
        files = {}
    for k in [k for k in entries if k not in files]:
        entries.pop(k, None)
    # npz files the index never saw (interrupted store / lost index): the
    # manifest is the truth, so rebuild the row instead of dropping live bytes.
    for k in [k for k in files if k not in entries]:
        meta = None
        try:
            meta = json.loads(_te_meta_path(k).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
        if isinstance(meta, dict) and meta.get("key") == k:
            entries[k] = {
                "last_used": float(meta.get("last_used")
                                   or meta.get("created_ts") or 0.0),
                "bytes": int(files[k]), "created": meta.get("created"),
                "prompt_sha256": meta.get("prompt_sha256"),
                "backend": meta.get("backend"), "tokens": meta.get("tokens"),
                "hits": int(meta.get("hits") or 0)}
        else:
            try:
                _te_npz_path(k).unlink()
            except OSError:
                pass
            files.pop(k, None)
    total = sum(int(v) for v in files.values())
    rows = sorted((float((entries.get(k) or {}).get("last_used") or 0.0), k)
                  for k in files if k != keep_key)
    removed = []
    while rows and (len(files) > TE_CACHE_MAX_ENTRIES
                    or total > TE_CACHE_MAX_BYTES):
        _ts, k = rows.pop(0)
        size = int(files.pop(k, 0))
        for pth in (_te_npz_path(k), _te_meta_path(k)):
            try:
                pth.unlink()
            except OSError:
                pass
        entries.pop(k, None)
        with _TE_LOCK:
            _TE_MEM.pop(k, None)
            _TE_STATS["evicted"] += 1
        total -= size
        removed.append(k)
        if log:
            log("[te-cache] evict %s (%d B)" % (k[:12], size))
    _te_index_save(idx)
    return removed


def te_materialize(entry, dest):
    """Write an entry's exact bytes to dest (atomic replace)."""
    dest = pathlib.Path(dest)
    data = entry.get("bytes")
    if data is None:
        data = _te_npz_path(entry["key"]).read_bytes()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return len(data)


def te_stats():
    idx = _te_index_load()
    entries = idx.get("entries") or {}
    files = []
    try:
        files = [p for p in TE_CACHE_DIR.glob("*.npz") if p.is_file()]
    except OSError:
        files = []
    total = 0
    for p in files:
        try:
            total += int(p.stat().st_size)
        except OSError:
            pass
    with _TE_LOCK:
        mem = len(_TE_MEM)
        st = dict(_TE_STATS)
    return {"version": TE_CACHE_VERSION,
            "dir": config.rel_to_npu(TE_CACHE_DIR),
            "entries": len(entries), "files": len(files), "bytes": total,
            "max_entries": TE_CACHE_MAX_ENTRIES,
            "max_bytes": TE_CACHE_MAX_BYTES,
            "memory_entries": mem, "memory_max": TE_CACHE_MEM_ENTRIES,
            "hits": st["hits"], "misses": st["misses"],
            "stores": st["stores"], "evicted": st["evicted"],
            "invalid": st["invalid"], "errors": st["errors"],
            "last_hit": st["last_hit"], "last_miss": st["last_miss"]}


def te_clear():
    """Delete every TE cache file and the in-process copies."""
    removed = 0
    with _TE_LOCK:
        _TE_MEM.clear()
    try:
        for p in TE_CACHE_DIR.iterdir():
            if p.is_file():
                try:
                    p.unlink()
                    removed += 1
                except OSError:
                    pass
    except OSError:
        pass
    return removed


def schedule_bytes(steps, sched=None):
    """Byte-stable schedule.npz for this step count.

    numpy's savez stamps a LOCAL TIME into the zip member, so two jobs that
    write the same timesteps/sigmas produced different files; the T22 gate asks
    for a bit-identical schedule artifact.  One copy lives in
    webui/cache/schedules/steps_N.npz and every job copies its exact bytes.
    When sched is given its arrays are compared with the cached ones (the
    delivered t13b trajectory is the steps==8 source), so a changed trajectory
    rewrites the cache instead of serving stale bytes.
    """
    steps = int(steps)
    p = config.CACHE_DIR / "schedules" / ("steps_%d.npz" % steps)
    if p.is_file():
        try:
            with np.load(p) as z:
                ts = np.asarray(z["timesteps"])
                sg = np.asarray(z["sigmas"])
            ok = len(ts) == steps and len(sg) == steps + 1
            if ok and sched is not None:
                ok = (np.array_equal(np.asarray(sched["timesteps"]), ts)
                      and np.array_equal(np.asarray(sched["sigmas"]), sg))
            if ok:
                return p.read_bytes()
        except Exception:  # noqa: BLE001
            pass
    if sched is None:
        return None
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, timesteps=np.asarray(sched["timesteps"], np.float64),
                 sigmas=np.asarray(sched["sigmas"], np.float64))
    tmp.replace(p)
    return p.read_bytes()
