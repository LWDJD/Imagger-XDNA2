# SPDX-License-Identifier: MIT
"""py312 worker: load the CPU VAE once, decode one batch's latents to PNGs."""
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
NPU = HERE.parent
PIPE = NPU / "imaggers" / "pipeline"
sys.path.insert(0, str(PIPE))
if str(NPU) not in sys.path:
    sys.path.insert(0, str(NPU))

import vae_decode as V  # noqa: E402  (shipped loader; diffusers inside load_vae)
from webui import config as CFG  # noqa: E402


def _fast_enabled() -> bool:
    """Same contract as webui/vae_local.py (default on, 0 restores shipped)."""
    return os.environ.get("ZIMAGE_VAE_FAST", "1") not in (
        "0", "false", "False", "off", "no", "")


def load_any():
    """(module, decoder, loader name).  lean first, shipped loader on any error."""
    if _fast_enabled():
        try:
            from webui import vae_lean
            return vae_lean, vae_lean.load_lean()[0], "lean"
        except Exception as exc:  # noqa: BLE001
            print(f"[vae_worker] ZIMAGE_VAE_FAST unavailable "
                  f"({type(exc).__name__}: {str(exc)[:160]}); shipped loader")
    return V, V.load_vae(), "diffusers"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--result", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lat = np.load(args.latents)
    if lat.ndim == 3:
        lat = lat[None]
    elif lat.ndim == 5:
        lat = lat[:, 0]          # batch saved as (B,1,16,16,16)
    if lat.ndim != 4:
        print(f"[vae_worker] latents shape {lat.shape}; expected (B,16,16,16)")
        return 2
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if len(seeds) != lat.shape[0]:
        print(f"[vae_worker] {len(seeds)} seeds vs {lat.shape[0]} latents")
        return 2
    t0 = time.perf_counter()
    try:
        V, vae, loader = load_any()
    except Exception as exc:
        print(f"[vae_worker] FAIL load_vae: {type(exc).__name__}: {exc}")
        return 3
    t_load = time.perf_counter() - t0
    # 2026-11-03: the decode threads are a property of the LATENT GEOMETRY
    # (config.vae_decode_threads), not of the box's core count and not of the
    # loop's ZIMAGE_CHUNK_THREADS.  Same rule as the resident decoder, so the
    # one-shot fallback cannot silently produce different bytes.
    lat_h, lat_w = int(lat.shape[-2]), int(lat.shape[-1])
    nthr, arm, why = CFG.vae_decode_threads(lat_h, lat_w)
    if nthr > 0:
        try:
            import torch
            if int(torch.get_num_threads()) != nthr:
                torch.set_num_threads(nthr)
        except Exception as exc:  # noqa: BLE001
            # do not claim an arm we did not get: the record says so
            print(f"[vae_worker] set_num_threads({nthr}) failed: {exc}")
            arm, why, nthr = "error", f"set_num_threads failed: {exc}", 0
    print(f"[vae_worker] decode threads: torch intra-op "
          f"{nthr if nthr > 0 else 'default'} [arm {arm}] "
          f"latent {lat_h}x{lat_w} ({why})")
    print(f"[vae_worker] VAE loaded in {t_load:.2f} s [{loader}]; decoding "
          f"{lat.shape[0]} image(s)")
    files = []
    decode_s = []
    save_s = []
    for b in range(lat.shape[0]):
        seed = seeds[b]
        t = time.perf_counter()
        try:
            img = V.decode(vae, lat[b:b + 1])
        except Exception as exc:
            print(f"[vae_worker] FAIL decode[{b}]: {type(exc).__name__}: {exc}")
            return 4
        dt = time.perf_counter() - t
        name = f"{args.prefix}_s{seed}.png"
        path = out_dir / name
        t_save = time.perf_counter()
        V.save_png(img, path)
        save_dt = time.perf_counter() - t_save
        decode_s.append(dt)
        save_s.append(save_dt)
        rec = {"seed": seed, "index": b, "file": name, "path": str(path),
               "bytes": path.stat().st_size, "sha256": _sha256(path),
               "width": int(img.shape[1]), "height": int(img.shape[0]),
               "decode_s": dt, "save_s": save_dt}
        files.append(rec)
        print(f"[vae_worker] {name}: {img.shape[1]}x{img.shape[0]} "
              f"{rec['bytes']} B in {dt:.3f} s sha256={rec['sha256'][:16]}")
    result = {"ok": True, "load_s": t_load, "decode_s": decode_s,
              "save_s": save_s, "total_s": time.perf_counter() - t0,
              "files": files,
              # the byte-changing knob, recorded (same keys as vae_local.py)
              "threads": int(nthr), "thread_arm": arm,
              "thread_class": CFG.vae_geometry_class(lat_h, lat_w),
              "thread_why": why, "thread_latent": [lat_h, lat_w],
              "thread_rule": "mod5" if CFG.vae_thread_rule() else "off"}
    pathlib.Path(args.result).write_text(json.dumps(result, indent=1),
                                         encoding="utf-8")
    print(f"[vae_worker] total {result['total_s']:.2f} s "
          f"(load {t_load:.2f} + decode {sum(decode_s):.2f} + save {sum(save_s):.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
