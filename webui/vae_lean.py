# SPDX-License-Identifier: MIT
"""vae_lean.py -- the shipped CPU decoder WITHOUT importing diffusers.

LANDED 2026-11-03 from imaggers/vaeopt/vae_lean.py (the module the VAE-OPT
campaign measured and gated).  The class bodies below are that file verbatim;
only the checkpoint path derivation changed (it is resolved from this file's
own location instead of a hardcoded D:/ literal), so the product tree holds the
decoder and does not need imaggers on sys.path.

Why this file exists
--------------------
The per-server "VAE load" (16.07 s in job 20261103-063031) is not the
checkpoint: 167.7 MB reads in 0.02-0.10 s and AutoencoderKL.from_pretrained
costs 0.10 s once the package is imported.  Almost all of it is
"from diffusers import AutoencoderKL", which pulls in transformers + onnx +
onnxruntime + onnxscript + torchvision + scipy + sympy (3775 modules, 10.5 s
warm, 23.3 s cold on this box, 14.07 s in product).

This module rebuilds the *same* decoder out of plain torch ops with the same op
order as diffusers 0.36.0 Decoder / ResnetBlock2D / UNetMidBlock2D /
Upsample2D / AttnProcessor2_0, and loads the same bf16 safetensors checkpoint
directly into it.  It needs only torch + safetensors, both of which the server
process already holds.

Exactness contract: decode() must be BIT-IDENTICAL to
AutoencoderKL.from_pretrained(...).decode() on the same latent.

    py312 -m webui.vae_lean --gate <latents.npy>

re-derives that gate in the product tree (equal, maxabs, rel_rms, PNG sha,
param count); it is not assumed here.  Recorded result at 512^2 seed 7:
equal=True maxabs=0.0 rel_rms=0.0 PNG sha 571c96cc386285e7 both ways,
49,545,475 params.

ROLLBACK: ZIMAGE_VAE_FAST=0 in webui/vae_local.py restores the diffusers
loader line for line.
"""
from __future__ import annotations

import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LeanVAE", "load_lean", "CKPT", "SCALING", "SHIFT", "decode", "save_png"]

HERE = pathlib.Path(__file__).resolve().parent          # webui/
NPU = HERE.parent                                       # zimage-bfp16-npu/
WS = NPU.parent                                         # public_work/
PORT = WS / "zimage-port"
CKPT = PORT / "_models" / "Z-Image-Turbo" / "vae" / "diffusion_pytorch_model.safetensors"
SCALING = 0.3611      # vae.config.scaling_factor
SHIFT = 0.1159        # vae.config.shift_factor
GROUPS = 32
EPS = 1e-6


class Resnet(nn.Module):
    """diffusers.models.resnet.ResnetBlock2D, temb=None, default time scale/shift."""

    def __init__(self, cin: int, cout: int, shortcut: bool):
        super().__init__()
        self.norm1 = nn.GroupNorm(GROUPS, cin, eps=EPS, affine=True)
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(GROUPS, cout, eps=EPS, affine=True)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1)
        self.conv_shortcut = nn.Conv2d(cin, cout, 1, stride=1, padding=0) if shortcut else None
        self.output_scale_factor = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x.contiguous())
        return (x + h) / self.output_scale_factor


class MidAttention(nn.Module):
    """diffusers Attention + AttnProcessor2_0, heads=1, residual_connection=True."""

    def __init__(self, dim: int):
        super().__init__()
        self.heads = 1
        self.head_dim = dim
        self.group_norm = nn.GroupNorm(GROUPS, dim, eps=EPS, affine=True)
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=True), nn.Dropout(0.0)])
        self.residual_connection = True
        self.rescale_output_factor = 1

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        batch_size, channel, height, width = hidden_states.shape
        h = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        h = self.group_norm(h.transpose(1, 2)).transpose(1, 2)
        query = self.to_q(h)
        key = self.to_k(h)
        value = self.to_v(h)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        h = F.scaled_dot_product_attention(query, key, value, attn_mask=None,
                                           dropout_p=0.0, is_causal=False)
        h = h.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        h = h.to(query.dtype)
        h = self.to_out[0](h)
        h = self.to_out[1](h)                       # nn.Dropout(0.0) -> identity in eval
        h = h.transpose(-1, -2).reshape(batch_size, channel, height, width)
        h = h + residual
        return h / self.rescale_output_factor


class UpBlock(nn.Module):
    """diffusers UpDecoderBlock2D: 3 resnets, optional Upsample2D(use_conv=True)."""

    def __init__(self, cin: int, cout: int, upsample: bool):
        super().__init__()
        self.resnets = nn.ModuleList([
            Resnet(cin, cout, shortcut=(cin != cout)),
            Resnet(cout, cout, shortcut=False),
            Resnet(cout, cout, shortcut=False),
        ])
        self.upsamplers = None
        if upsample:
            self.upsamplers = nn.ModuleList([nn.ModuleDict({"conv": nn.Conv2d(cout, cout, 3, padding=1)})])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for r in self.resnets:
            x = r(x)
        if self.upsamplers is not None:
            for up in self.upsamplers:
                x = F.interpolate(x, scale_factor=2.0, mode="nearest")
                x = up["conv"](x)
        return x


class MidBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.resnets = nn.ModuleList([Resnet(dim, dim, False), Resnet(dim, dim, False)])
        self.attentions = nn.ModuleList([MidAttention(dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnets[0](x)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            x = attn(x)
            x = resnet(x)
        return x


class LeanDecoder(nn.Module):
    def __init__(self, block_out_channels=(128, 256, 512, 512), layers_per_block=2):
        super().__init__()
        rev = list(reversed(block_out_channels))
        self.conv_in = nn.Conv2d(16, rev[0], 3, padding=1)
        self.mid_block = MidBlock(rev[0])
        blocks = []
        out_ch = rev[0]
        for i in range(len(block_out_channels)):
            in_ch = out_ch
            out_ch = rev[i]
            blocks.append(UpBlock(in_ch, out_ch, upsample=(i != len(block_out_channels) - 1)))
        self.up_blocks = nn.ModuleList(blocks)
        self.conv_norm_out = nn.GroupNorm(GROUPS, block_out_channels[0], eps=EPS, affine=True)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], 3, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.conv_in(z)
        z = self.mid_block(z)
        for b in self.up_blocks:
            z = b(z)
        z = self.conv_norm_out(z)
        z = self.conv_act(z)
        return self.conv_out(z)


class LeanVAE(nn.Module):
    """Decoder only; the checkpoint's post_quant_conv is None (use_post_quant_conv=false)."""

    def __init__(self):
        super().__init__()
        self.decoder = LeanDecoder()
        self.scaling_factor = SCALING   # overwritten from config.json by load_lean()
        self.shift_factor = SHIFT

    @torch.no_grad()
    def decode_plain(self, lat_v: torch.Tensor) -> torch.Tensor:
        """latent already scaled: what AutoencoderKL._decode does (no post_quant_conv)."""
        return self.decoder(lat_v)

    @torch.no_grad()
    def decode(self, lat: torch.Tensor) -> torch.Tensor:
        return self.decoder(lat / self.scaling_factor + self.shift_factor)


def decode(model, latents):
    """Drop-in for vae_decode.decode(vae, latents) -- same arithmetic, same dtype.

    latents (1, C, H, W) float32 -> image (H_px, W_px, 3) in [0, 1].
    """
    import numpy as np

    lat = torch.from_numpy(np.ascontiguousarray(latents, dtype=np.float32))
    lat_v = lat / model.scaling_factor + model.shift_factor
    img = model.decode_plain(lat_v)
    arr = img.float().cpu().numpy()
    arr = np.transpose(arr[0], (1, 2, 0))
    arr = (arr + 1.0) / 2.0
    return np.clip(arr, 0.0, 1.0)


def save_png(arr, path) -> None:
    """Drop-in for vae_decode.save_png (byte-identical PNG)."""
    import numpy as np
    from PIL import Image

    Image.fromarray((np.asarray(arr) * 255.0 + 0.5).astype(np.uint8)).save(path)


def load_lean(ckpt=None, dtype=torch.float32):
    """Build + load.  Returns (model, timings dict)."""
    import time
    from safetensors import safe_open

    ckpt = pathlib.Path(ckpt or CKPT)
    t = time.perf_counter
    t0 = t()
    model = LeanVAE()
    t_build = t() - t0
    t0 = t()
    sd = {}
    with safe_open(str(ckpt), framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith("decoder."):
                sd[k[len("decoder."):]] = f.get_tensor(k).to(dtype)
    t_read = t() - t0
    t0 = t()
    model.decoder.load_state_dict(sd, strict=True)
    t_load = t() - t0
    model.eval()
    cfg_path = ckpt.parent / "config.json"
    if cfg_path.is_file():
        import json

        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        model.scaling_factor = float(cfg["scaling_factor"])
        model.shift_factor = float(cfg["shift_factor"])
        model.config_json = str(cfg_path)
    return model, {"build_s": t_build, "read_s": t_read, "load_s": t_load,
                   "total_s": t_build + t_read + t_load,
                   "scaling_factor": model.scaling_factor,
                   "shift_factor": model.shift_factor,
                   "params": sum(p.numel() for p in model.parameters())}

def _gate(latents_path) -> int:
    """Compare this decoder with diffusers' on one latent file (oracle gate).

    Prints the four numbers the VAE-OPT campaign used (equal / maxabs / rel_rms
    / PNG sha) and returns 0 when the PNG bytes are identical.
    """
    import hashlib
    import sys

    import numpy as np
    import torch as _t

    sys.path.insert(0, str(NPU / "imaggers" / "pipeline"))
    import vae_decode as VD  # noqa: E402  (the shipped path: diffusers)

    lat = np.load(latents_path)
    if lat.ndim == 3:
        lat = lat[None]
    if lat.ndim == 5:
        lat = lat[:, 0]
    lean, parts = load_lean()
    ref = VD.load_vae()
    a = decode(lean, lat[0:1])
    b = VD.decode(ref, lat[0:1])
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    denom = np.sqrt(np.mean(b.astype(np.float64) ** 2)) or 1.0
    # the scratch dir must live in the project tree: the DSH sandbox does not
    # let this process write the platform temp area
    # NOTE: not tempfile.mkdtemp -- under this host's file sandbox a mkdtemp
    # directory is not writable by the process that made it, while
    # pathlib.mkdir() is.  So the scratch dir is made explicitly.
    scratch = NPU / "imaggers" / "vaelean_gate"
    scratch.mkdir(parents=True, exist_ok=True)
    pa = scratch / "lean.png"
    pb = scratch / "ref.png"
    save_png(a, pa)
    VD.save_png(b, pb)
    ha = hashlib.sha256(pa.read_bytes()).hexdigest()
    hb = hashlib.sha256(pb.read_bytes()).hexdigest()
    print("GATE latent %s shape %s" % (latents_path, lat.shape))
    print("GATE params %d (lean) %d (diffusers)  equal=%s  maxabs=%.3g  "
          "rel_rms=%.3g" % (parts["params"],
                            sum(p.numel() for p in ref.parameters()),
                            bool(np.array_equal(a, b)), float(d.max()),
                            float(np.sqrt((d ** 2).mean()) / denom)))
    print("GATE png sha lean=%s ref=%s identical=%s"
          % (ha[:16], hb[:16], ha == hb))
    print("GATE load_s build=%.3f read=%.3f load=%.3f total=%.3f"
          % (parts["build_s"], parts["read_s"], parts["load_s"],
             parts["total_s"]))
    print("GATE threads torch=%d" % _t.get_num_threads())
    return 0 if ha == hb else 1


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate", metavar="LATENTS.npy")
    ap.add_argument("--threads", type=int, default=0,
                    help="torch.set_num_threads before the gate (0 = default)")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if not args.gate:
        ap.error("--gate LATENTS.npy is the only mode")
    raise SystemExit(_gate(args.gate))
