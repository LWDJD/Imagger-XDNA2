# SPDX-License-Identifier: MIT
"""Resident VAE decoder: the NPU chain, with the CPU decoder as the fallback.

T16 P0-2 history: the manager saw "VAE 20 s", which was the per-job torch load,
not the decode (measured: load 15.3 s, decode 0.455 s/image, PNG 0.018 s).  This
module keeps a decoder resident in the py312 HTTP server process, so only the
first job of a server pays the load.

T21-3B3 adds the NPU path.  imaggers/vae/t21_3b3_chain.py runs the real
AutoencoderKL decoder with conv / GroupNorm / upsample on the NPU (the 92 gated
3b2/3b2vf xclbins) and SiLU / add / attention on host fp32 -- the split is
declared per stage in webui/evidence/T21-3B3-VAE.md section 3.  The path is
used only for the six product sizes; every other shape, any device error and
ZIMAGE_VAE_NPU=0 fall back to the CPU decoder, whose behaviour is unchanged
(same load, same decode call, same records).

Crash isolation note: both paths are in-process, so a torch-level crash would
take the server down with it.  The runner falls back to the one-shot
webui/vae_worker.py if this module raises.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import sys
import threading
import time

import numpy as np

from . import config

HERE = pathlib.Path(__file__).resolve().parent
NPU = HERE.parent
PIPE = NPU / "imaggers" / "pipeline"
CHAIN_DIR = NPU / "imaggers" / "vae"
# T23 VAE-load (LANDED 2026-11-03): the diffusers-free decoder (torch +
# safetensors only, webui/vae_lean.py).  It is bit-identical to
# AutoencoderKL.decode on the same weights and loads in 0.21 s instead of
# 14.07 s, because "from diffusers import AutoencoderKL" pulls in
# transformers+onnx+onnxruntime+torchvision+scipy+sympy (10.5 s warm, 23.3 s
# cold) while from_pretrained itself costs 0.10 s.
# DEFAULT ON since the 512^2 8-step A/B produced byte-identical PNGs
# (webui/jobs/20261103-*-ab*, see VAE-THREADS-LANDED.md).
# ROLLBACK: ZIMAGE_VAE_FAST=0 restores the shipped diffusers loader line for
# line (the else branch below is that path, untouched).  A lean failure of any
# kind also falls back to it inside this same function, so a missing
# safetensors/checkpoint can never turn into a failed job.
# (the original research module lives in imaggers/vaeopt/; the product copy is
# webui/vae_lean.py, so nothing here needs that directory on sys.path)
VAEOPT = NPU / "imaggers" / "vaeopt"


def _fast_enabled() -> bool:
    return os.environ.get("ZIMAGE_VAE_FAST", "1") not in (
        "0", "false", "False", "off", "no", "")

_LOCK = threading.Lock()
_STATE = {"V": None, "vae": None, "loads": 0, "decodes": 0,
          "loader": None, "threads": 0,
          # 2026-11-03 thread rule: which arm this decode used, and torch's own
          # default (captured before we ever resize the pool, so a later
          # "legacy" job can be put back on the shipped path)
          "thread_arm": None, "thread_class": None, "thread_why": None,
          "thread_latent": None, "torch_default": None}
_LOAD_S = {"v": 0.0}
_NPU = {"mod": None, "decodes": 0, "last_err": None}


def _apply_threads(log=None) -> int:
    """Apply the thread count decode_batch picked; 0 = leave torch's default.

    The CHOICE is not made here.  It is made once, from the latent geometry, in
    config.vae_decode_threads(h, w) -- which is deliberately independent of
    ZIMAGE_CHUNK_THREADS, because the decode's bytes move with the thread count
    and a loop knob must not be able to move an image.

    This function only applies it, and it remembers torch's own default (the
    count before this process ever resized the pool) so the "legacy" arm can be
    restored after an earlier job in the same resident server widened it.
    """
    n = int(_STATE.get("threads") or 0)
    import torch
    if _STATE.get("torch_default") is None:
        _STATE["torch_default"] = int(torch.get_num_threads())
    want = n if n > 0 else int(_STATE["torch_default"])
    have = int(torch.get_num_threads())
    if have != want:
        torch.set_num_threads(want)
        if log:
            # NOT the canonical arm line (that is logged once per decode in
            # decode_batch, with the geometry and the reason) -- this one marks
            # a real pool transition, including the legacy arm's restore.
            log(f"[vae] torch intra-op pool: {have} -> {want}"
                f" [arm {_STATE.get('thread_arm') or 'default'}]")
    return n


def status():
    with _LOCK:
        return {"resident": _STATE["vae"] is not None,
                "loads": _STATE["loads"], "decodes": _STATE["decodes"],
                "npu_decodes": _NPU["decodes"],
                "npu_last_error": _NPU["last_err"],
                "npu_enabled": bool(config.VAE_NPU)}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_shipped(log):
    """The shipped loader, line for line: diffusers AutoencoderKL.from_pretrained."""
    import vae_decode as V  # noqa: E402  (torch/diffusers import here)
    _STATE["V"] = V
    _STATE["vae"] = V.load_vae()
    _STATE["loader"] = "diffusers"


def _cpu_module(log):
    """(vae_decode|vae_lean module, seconds of load paid by THIS call).

    ZIMAGE_VAE_FAST (default on) tries webui.vae_lean first; ANY failure there
    is logged and the shipped loader runs instead, so the flag can only ever
    make the load slower, never fail it.
    """
    if _STATE["vae"] is None:
        if str(PIPE) not in sys.path:
            sys.path.insert(0, str(PIPE))
        t0 = time.perf_counter()
        if _fast_enabled():
            try:
                from . import vae_lean as V   # torch + safetensors; no diffusers
                _STATE["V"] = V
                _STATE["vae"] = V.load_lean()[0]
                _STATE["loader"] = "lean"
            except Exception as exc:  # noqa: BLE001
                log(f"[vae] ZIMAGE_VAE_FAST unavailable "
                    f"({type(exc).__name__}: {str(exc)[:160]}); shipped loader")
                _STATE["vae"] = None
                _STATE["V"] = None
        if _STATE["vae"] is None:
            _load_shipped(log)
        load_s = time.perf_counter() - t0
        _STATE["loads"] += 1
        log(f"[vae] resident decoder loaded in {load_s:.2f} s "
            f"(one-time per server; later jobs pay 0)"
            f" [{_STATE['loader']}]")
        return _STATE["V"], load_s
    return _STATE["V"], 0.0


def _npu_module():
    if _NPU["mod"] is None:
        if str(CHAIN_DIR) not in sys.path:
            sys.path.insert(0, str(CHAIN_DIR))
        import t21_3b3_chain as C  # noqa: E402  (torch/aie paths inside)
        _NPU["mod"] = C
    return _NPU["mod"]


def _npu_shape_ok(C, lat) -> bool:
    key = (int(lat.shape[-2]), int(lat.shape[-1]))
    return key in C.LATENT_TO_SIZE


def _thread_record():
    """The thread arm that produced this batch's PNGs (lands in job.json).

    The 2026-11-03 audit found 0/627 job.json named a byte-changing knob; this
    is the one that moves the PNG, so it travels with the result and the runner
    copies it into the job record and every image record.
    """
    return {"threads": int(_STATE.get("threads") or 0),
            "thread_arm": _STATE.get("thread_arm"),
            "thread_class": _STATE.get("thread_class"),
            "thread_why": _STATE.get("thread_why"),
            "thread_latent": _STATE.get("thread_latent"),
            "thread_rule": "mod5" if config.vae_thread_rule() else "off"}


def decode_batch(latents_path, out_dir, prefix, seeds, on_line=None,
                 cancel_check=None):
    """Decode one batch's latents -> PNGs.  Returns the vae_worker result dict.

    impl = "npu" when the chain decoded every image, "cpu" when the resident
    CPU decoder did, "mixed" if the NPU path failed part-way through (the
    remaining images then take the CPU path; the batch record says so).
    """
    log = on_line or (lambda s: None)
    seeds = [int(s) for s in seeds]
    with _LOCK:
        t_all = time.perf_counter()
        load_s = 0.0
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        lat = np.load(latents_path)
        if lat.ndim == 3:
            lat = lat[None]
        elif lat.ndim == 5:
            lat = lat[:, 0]
        if lat.ndim != 4:
            raise ValueError(f"latents shape {lat.shape}; expected (B,16,16,16)")
        if len(seeds) != lat.shape[0]:
            raise ValueError(f"{len(seeds)} seeds vs {lat.shape[0]} latents")

        # ---- VAE decode threads (LANDED 2026-11-03) ------------------------
        # The arm is a property of the GEOMETRY, not of this box's core count
        # and not of the loop's pool: 20 threads where every SiLU numel is a
        # multiple of 320 (latent_H*latent_W % 5 == 0, bit-identical), 10
        # everywhere else (today's shipped bytes).
        # It is ANNOUNCED LAZILY, in the CPU branch below: a job the NPU chain
        # decodes has no torch arm, and must not be recorded as if it had one.
        lat_h, lat_w = int(lat.shape[-2]), int(lat.shape[-1])
        thr, arm, why = config.vae_decode_threads(lat_h, lat_w)
        arm_announced = False
        for _k in ("threads", "thread_arm", "thread_class", "thread_why",
                   "thread_latent"):
            _STATE[_k] = None if _k != "threads" else 0

        chain = None
        if config.VAE_NPU:
            try:
                chain = _npu_module()
                if not _npu_shape_ok(chain, lat):
                    log(f"[vae] NPU chain has no geometry for latent "
                        f"{lat.shape[-2]}x{lat.shape[-1]}; CPU decoder")
                    chain = None
                else:
                    log("[vae] NPU chain selected (conv/GroupNorm/upsample on "
                        "the NPU; SiLU/add/attention host fp32)")
            except Exception as exc:  # noqa: BLE001
                _NPU["last_err"] = f"{type(exc).__name__}: {exc}"
                log(f"[vae] NPU chain unavailable ({_NPU['last_err'][:160]}); "
                    f"CPU decoder")
                chain = None
        else:
            log("[vae] ZIMAGE_VAE_NPU=0; CPU decoder")

        V = None
        files, decode_s, save_s = [], [], []
        impls, npu_calls = [], []
        for b in range(lat.shape[0]):
            if cancel_check and cancel_check():
                return {"ok": False, "cancelled": True, "load_s": load_s,
                        "decode_s": decode_s, "save_s": save_s,
                        "total_s": time.perf_counter() - t_all, "files": files,
                        "impl": "npu" if chain else "cpu",
                        **_thread_record()}
            seed = seeds[b]
            img = None
            impl = "npu" if chain is not None else "cpu"
            ncalls = 0
            if chain is not None:
                try:
                    r = chain.decode_latent(lat[b:b + 1], scaled=False)
                    img = r["image"]
                    dt = float(r["decode_s"])
                    # the chain builds/patches its decoder once per process:
                    # that is LOAD, not DECODE (the closure gate caught this
                    # as 22 s of unaccounted time on the first job)
                    load_s += float(r.get("build_s", 0.0))
                    ncalls = int(r["n_calls"])
                    _NPU["decodes"] += 1
                    log(f"[vae] npu decode {r['size']} in {dt:.2f} s "
                        f"({ncalls} device calls, {r['conv_launches']} conv "
                        f"launches)")
                except Exception as exc:  # noqa: BLE001
                    _NPU["last_err"] = f"{type(exc).__name__}: {exc}"
                    log(f"[vae] NPU decode failed ({_NPU['last_err'][:200]}); "
                        f"falling back to the CPU decoder for the rest of "
                        f"this batch")
                    chain = None
                    img = None
            if img is None:
                impl = "cpu"
                V, dl = _cpu_module(log)
                load_s += dl
                if not arm_announced:
                    _STATE["threads"] = int(thr)
                    _STATE["thread_arm"] = arm
                    _STATE["thread_class"] = config.vae_geometry_class(lat_h, lat_w)
                    _STATE["thread_why"] = why
                    _STATE["thread_latent"] = [lat_h, lat_w]
                    # one line per CPU decode, so the job log always says which
                    # arm produced the PNGs
                    log(f"[vae] decode threads: torch intra-op "
                        f"{thr if thr > 0 else 'default'} [arm {arm}] "
                        f"latent {lat_h}x{lat_w} ({why})")
                    arm_announced = True
                _apply_threads(log)
                t0 = time.perf_counter()
                img = V.decode(_STATE["vae"], lat[b:b + 1])
                dt = time.perf_counter() - t0
            name = f"{prefix}_s{seed}.png"
            path = out_dir / name
            t_save = time.perf_counter()
            if V is None:
                if str(PIPE) not in sys.path:
                    sys.path.insert(0, str(PIPE))
                import vae_decode as V  # noqa: E402  (save_png only)
            V.save_png(img, path)
            save_dt = time.perf_counter() - t_save
            decode_s.append(dt)
            save_s.append(save_dt)
            impls.append(impl)
            npu_calls.append(ncalls)
            rec = {"seed": seed, "index": b, "file": name, "path": str(path),
                   "bytes": path.stat().st_size, "sha256": _sha256(path),
                   "width": int(img.shape[1]), "height": int(img.shape[0]),
                   "decode_s": dt, "save_s": save_dt, "impl": impl,
                   "npu_calls": ncalls}
            files.append(rec)
            log(f"[vae] {name}: {img.shape[1]}x{img.shape[0]} {rec['bytes']} B "
                f"in {dt:.3f} s [{impl}] sha256={rec['sha256'][:16]}")
        _STATE["decodes"] += 1
        impl = ("npu" if all(i == "npu" for i in impls)
                else "cpu" if all(i == "cpu" for i in impls) else "mixed")
        return {"ok": True, "resident": impl == "cpu", "impl": impl,
                "load_s": load_s, "decode_s": decode_s, "save_s": save_s,
                "total_s": time.perf_counter() - t_all, "files": files,
                "npu_calls": npu_calls,
                "npu_last_error": _NPU["last_err"],
                # T23: which loader produced these PNGs
                # 2026-11-03: and which THREAD ARM -- the knob that moves bytes
                "loader": _STATE["loader"], **_thread_record()}
