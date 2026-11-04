# SPDX-License-Identifier: MIT
"""vae_decode.py -- the CPU VAE decoder for the first image.

The scope decision in `KNOWN-ISSUES.md` puts the first image's decoder on the
CPU: the decoder runs once per image and is far cheaper than the 34-block DiT.
`zimage-port/ref_probe_env.py:63-73` loads it as

    AutoencoderKL.from_pretrained(ROOT, subfolder="vae")
    vae_scale_factor = 2 ** (len(block_out_channels) - 1) = 16

and `zimage-port/ref_native_128.py:580-582` shows the exact call:

    lat_v = latents / vae.config.scaling_factor + vae.config.shift_factor
    img = vae.decode(lat_v, return_dict=False)[0]

Match that arithmetic exactly -- the reference is the oracle.

Usage:
    python vae_decode.py --latents <npy> --out image.png
    python vae_decode.py --self-test
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
NPU_ROOT = HERE.parent.parent
WS = NPU_ROOT.parent
PORT = WS / "zimage-port"


def load_vae():
    sys.path.insert(0, str(PORT))
    import torch
    from diffusers import AutoencoderKL
    root = PORT / "_models" / "Z-Image-Turbo"
    vae = AutoencoderKL.from_pretrained(str(root), subfolder="vae")
    vae.eval()
    return vae


def decode(vae, latents: np.ndarray) -> np.ndarray:
    """latents (1, C, H, W) -> image (H_px, W_px, 3) in [0, 1]."""
    import torch
    lat = torch.from_numpy(np.ascontiguousarray(latents, dtype=np.float32))
    lat_v = lat / vae.config.scaling_factor + vae.config.shift_factor
    with torch.no_grad():
        img = vae.decode(lat_v, return_dict=False)[0]
    arr = img.float().cpu().numpy()
    arr = np.transpose(arr[0], (1, 2, 0))     # CHW -> HWC
    arr = (arr + 1.0) / 2.0
    return np.clip(arr, 0.0, 1.0)


def save_png(arr: np.ndarray, path: pathlib.Path) -> None:
    from PIL import Image
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8)).save(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", default=None)
    ap.add_argument("--out", default=str(HERE / "image.png"))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.self_test:
        rng = np.random.default_rng(0)
        lat = rng.standard_normal((1, 16, 8, 8)).astype(np.float32)
    elif args.latents:
        lat = np.load(args.latents)
        if lat.ndim == 3:
            lat = lat[None]
    else:
        print("[FAIL] give --latents or --self-test")
        return 1

    print(f"latents {lat.shape} absmax {float(np.abs(lat).max()):.4f}")
    try:
        vae = load_vae()
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] VAE load: {type(e).__name__}: {str(e)[:200]}")
        return 1
    print(f"vae {sum(p.numel() for p in vae.parameters()):,} params "
          f"scaling {vae.config.scaling_factor} "
          f"shift {vae.config.shift_factor}")

    img = decode(vae, lat)
    print(f"image {img.shape} min {img.min():.4f} max {img.max():.4f}")
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_png(img, out)
    print(f"wrote {out}  {out.stat().st_size:,} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
