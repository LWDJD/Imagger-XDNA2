# t21_3b2_geom.py -*- Python -*-
#
# T21-3B2 U1: the frozen geometry table for the 92 second-tier xclbins.
#
# WHAT THIS FILE IS
# -----------------
# The 3b-2 build unit needs, per geometry, the exact parameters that get
# BURNED INTO the xclbin (they are CompileTime, not runtime):
#
#   conv  (Cin, Cout, kh, kw, M)  ->  (M, K, N) with
#           K = round_up(Cin*kh*kw + 8, 32)   (8 = the folded-bias k-block)
#           N = round_up(Cout, 16)            (mm_bfp needs N % 16 == 0)
#   GN    (C, S=H*W)              ->  single vs stream by the 64 KB L1 rule
#   UP    (C, H, W)               ->  plane_in = H*W, plane_out = 4*H*W
#
# The table is DERIVED (not hand-written): derive() runs the real CPU decoder
# at the six product sizes with forward hooks and dedups the tuples per size.
# FROZEN below is the result of one such run, checked in so that
# t21_3b2_plan.py --dry-run never needs torch/diffusers.
#
# WHAT THIS FILE DOES NOT DO
# --------------------------
# No compile, no device, no xclbin. Only a CPU forward with hooks.
#
# Usage:
#     py312 t21_3b2_geom.py --derive          # re-derive and print the table
#     py312 t21_3b2_geom.py --check           # derive and diff against FROZEN
#     py312 t21_3b2_geom.py --write-json FILE # dump the derived table
#     py312 t21_3b2_geom.py --frozen          # print FROZEN (no torch import)
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
PORT = ROOT / "zimage-port"
MODELS = PORT / "_models" / "Z-Image-Turbo"

SCALING = 0.3611
SHIFT = 0.1159
GROUPS = 32

# The six product sizes (image HxW) and the latent they decode from.
# 144x128 and 192x128 are non-square on purpose: their spatial counts are
# NOT reachable from the square sizes.
SIZES = [
    ("128x128", 16, 16),
    ("64x256", 8, 32),
    ("144x128", 18, 16),
    ("192x128", 24, 16),
    ("128x256", 16, 32),
    ("256x256", 32, 32),
]


# ---------------------------------------------------------------------------
# the parameter derivations (pure functions; the plan harness reuses them)
# ---------------------------------------------------------------------------

def round_up(v: int, m: int) -> int:
    return (int(v) + m - 1) // m * m


def conv_mkn(cin: int, cout: int, k: int, m: int) -> dict:
    """(Cin, Cout, k, M) -> the burned-in (M, K, N) and the real sizes."""
    k_real = cin * k * k
    return dict(
        M=int(m),
        K=round_up(k_real + 8, 32),      # +8 bias block, then 32-align
        N=round_up(cout, 16),
        K_real=int(k_real),
        N_real=int(cout),
        bias_col=int(k_real),
    )


def gn_mode(C: int, S: int) -> str:
    """single iff 2*L bytes in + 2*L out (+4 KB slack) fit in 64 KB L1."""
    L = (C // GROUPS) * S
    return "single" if 2 * 2 * L + 4096 <= 64 * 1024 else "stream"


def gn_chunk(C: int, S: int, cs_max: int = 4096):
    """vaeg_design.pick_chunk, but returning None instead of raising.

    pick_chunk needs Cs to be a whole number of spatial PLANES (Cs % S == 0),
    so a chunk is at least one plane: when S > cs_max there is NO valid Cs and
    the factory raises ValueError.  That is a design limit, not a typo: at
    S = 16384 one plane is already 32 KB in + 128 KB out, over the 64 KB L1,
    so those shapes need SUB-PLANE chunking (a design variant), not a
    recompile.  The plan reports them separately instead of pretending they
    build.
    """
    L = (C // GROUPS) * S
    cs = min(cs_max, L)
    while cs >= S and (L % cs or cs % S):
        cs -= S
    if cs < S:
        return None
    return cs


def gn_plan(C: int, S: int, cs_max: int = 4096) -> dict:
    L = (C // GROUPS) * S
    mode = gn_mode(C, S)
    chunk = None if mode == "stream" else None
    if mode == "stream":
        chunk = gn_chunk(C, S, cs_max)
    ok = (mode == "single") or (chunk is not None)
    note = ""
    if mode == "stream" and chunk is None:
        note = (f"design gap: S={S} > cs_max={cs_max}; pick_chunk has no "
                f"whole-plane chunk, needs sub-plane chunking")
    return dict(mode=mode, group_len=int(L), chunk=chunk, chunk_ok=bool(ok),
                note=note)


def up_plane(C: int, H: int, W: int) -> dict:
    S = H * W
    return dict(S=S, plane_in=S, plane_out=4 * S,
                in_bytes=C * S * 2, out_bytes=C * S * 8,
                # in + out object live together in the 64 KB L1 (depth-1 FIFOs)
                l1_bytes=10 * S,
                # in(2S) + out(8S) both live in the 64 KB L1, plus the
                # Worker's 4 KB stack (0x1000) and alignment slack: keep 8 KB.
                fits_existing=bool(10 * S + 8192 <= 64 * 1024))


# ---------------------------------------------------------------------------
# the derivation (needs torch + diffusers: py312 only)
# ---------------------------------------------------------------------------

def derive(verbose: bool = True) -> dict:
    import numpy as np
    import torch
    sys.path.insert(0, str(PORT))
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(str(MODELS), subfolder="vae").eval()
    convs = [m for _, m in vae.decoder.named_modules()
             if isinstance(m, torch.nn.Conv2d)]
    gns = [m for _, m in vae.decoder.named_modules()
           if isinstance(m, torch.nn.GroupNorm)]
    ups = [m for _, m in vae.decoder.named_modules()
           if m.__class__.__name__ == "Upsample2D"]
    if verbose:
        print(f"[arch] decoder: {len(convs)} Conv2d, {len(gns)} GroupNorm, "
              f"{len(ups)} Upsample2D")

    out = dict(sizes=[], conv=[], gn=[], up=[], per_size={})
    for size_name, lh, lw in SIZES:
        order = {"i": 0}
        seen_conv, seen_gn, seen_up = [], [], []
        conv_rows, gn_rows, up_rows = [], [], []
        handles = []

        def conv_hook(mod, name, inp, outp, _order=order, _rows=conv_rows):
            x = inp[0]
            m = int(x.shape[2] * x.shape[3])
            cout, cin, kh, kw = [int(v) for v in mod.weight.shape]
            _order["i"] += 1
            _rows.append(dict(order=_order["i"], module=name, Cin=cin,
                              Cout=cout, k=kh, M=m, H=int(x.shape[2]),
                              W=int(x.shape[3])))

        def gn_hook(mod, name, inp, outp, _order=order, _rows=gn_rows):
            x = inp[0]
            _order["i"] += 1
            # the mid-block ATTENTION group_norm sees (1, C, L) tokens, not
            # (1, C, H, W): S is the product of every axis after the channel
            # axis, which is exactly L = H*W for that instance.
            _rows.append(dict(order=_order["i"], module=name,
                              C=int(mod.num_channels),
                              S=int(math.prod(int(v) for v in x.shape[2:]))))

        def up_hook(mod, name, inp, outp, _order=order, _rows=up_rows):
            x = inp[0]
            _order["i"] += 1
            _rows.append(dict(order=_order["i"], module=name,
                              C=int(x.shape[1]), H=int(x.shape[2]),
                              W=int(x.shape[3])))

        for name, mod in vae.decoder.named_modules():
            if isinstance(mod, torch.nn.Conv2d):
                handles.append(mod.register_forward_hook(
                    lambda m, i, o, n=name: conv_hook(m, n, i, o)))
            elif isinstance(mod, torch.nn.GroupNorm):
                handles.append(mod.register_forward_hook(
                    lambda m, i, o, n=name: gn_hook(m, n, i, o)))
            elif mod.__class__.__name__ == "Upsample2D":
                handles.append(mod.register_forward_hook(
                    lambda m, i, o, n=name: up_hook(m, n, i, o)))

        rng = np.random.default_rng(0)
        lat = rng.standard_normal((1, 16, lh, lw)).astype(np.float32)
        lat_v = torch.from_numpy(lat / SCALING + SHIFT)
        with torch.no_grad():
            img = vae.decode(lat_v, return_dict=False)[0]
        for h in handles:
            h.remove()
        if verbose:
            print(f"[derive] {size_name}: latent {lh}x{lw} -> image "
                  f"{tuple(img.shape[2:])}  {len(conv_rows)} conv / "
                  f"{len(gn_rows)} GN / {len(up_rows)} UP instances")

        # dedup, first occurrence wins, order preserved
        def dedup(rows, keyf):
            seen, keep = set(), []
            for r in rows:
                k = keyf(r)
                if k in seen:
                    continue
                seen.add(k)
                keep.append(r)
            return keep

        dc = dedup(conv_rows, lambda r: (r["Cin"], r["Cout"], r["k"], r["M"]))
        dg = dedup(gn_rows, lambda r: (r["C"], r["S"]))
        # UP dedups on (C, S): the kernel is upsample_nearest2(x, y, n, w)
        # with n = S and w RUNTIME, so one xclbin serves every (H, W)
        # factorization of the same S.  H and W are kept as variants so the
        # gate can exercise each factorization (they are all real instances).
        du = dedup(up_rows, lambda r: (r["C"], r["H"] * r["W"]))
        for r in dc:
            r.update(size=size_name, kind="conv", **conv_mkn(
                r["Cin"], r["Cout"], r["k"], r["M"]))
            r.pop("order", None)
        for r in dg:
            r.update(size=size_name, kind="gn", **gn_plan(r["C"], r["S"]))
            r.pop("order", None)
        for r in du:
            r.update(size=size_name, kind="up", variants=[f"{r['H']}x{r['W']}"],
                     **up_plane(r["C"], r["H"], r["W"]))
            r.pop("order", None)
        out["per_size"][size_name] = dict(conv=dc, gn=dg, up=du,
                                          n_conv_inst=len(conv_rows),
                                          n_gn_inst=len(gn_rows),
                                          n_up_inst=len(up_rows))
        out["sizes"].append(size_name)

    # union across sizes (same dedup rule, first size wins)
    def union(kind, keyf):
        seen, keep = set(), []
        for size_name in out["sizes"]:
            for r in out["per_size"][size_name][kind]:
                k = keyf(r)
                if k in seen:
                    continue
                seen.add(k)
                keep.append(r)
        return keep

    out["conv"] = union("conv", lambda r: (r["Cin"], r["Cout"], r["k"], r["M"]))
    out["gn"] = union("gn", lambda r: (r["C"], r["S"]))
    def up_union():
        seen, keep = set(), []
        for size_name in out["sizes"]:
            for r in out["per_size"][size_name]["up"]:
                k = (r["C"], r["S"])
                if k in seen:
                    # same xclbin, second (H, W) factorization of the same S
                    for u in keep:
                        if (u["C"], u["S"]) == k:
                            for v in r.get("variants", []):
                                if v not in u["variants"]:
                                    u["variants"].append(v)
                            break
                    continue
                seen.add(k)
                keep.append(r)
        return keep

    out["up"] = up_union()
    return out


# ---------------------------------------------------------------------------
# THE FROZEN TABLE
# ---------------------------------------------------------------------------
# Baked verbatim from one derive() run (2026-10-19, py312 + diffusers,
# latent rng(0) like the chain).  Rows are the RAW keys only; every derived
# field (K, N, mode, chunk, L1 bytes) is recomputed by frozen_core(), so the
# literal cannot drift from the formulas above.
#   conv: [size, module, Cin, Cout, k, M]
#   gn:   [size, module, C, S]
#   up:   [size, module, C, H, W, [all (H,W) factorizations of the same S]]
# ---- generated begin ----
FROZEN = {
 "sizes": ["128x128", "64x256", "144x128", "192x128", "128x256", "256x256"],
 "conv": [
  ["128x128", "conv_in", 16, 512, 3, 256],
  ["128x128", "mid_block.resnets.0.conv1", 512, 512, 3, 256],
  ["128x128", "up_blocks.0.upsamplers.0.conv", 512, 512, 3, 1024],
  ["128x128", "up_blocks.1.upsamplers.0.conv", 512, 512, 3, 4096],
  ["128x128", "up_blocks.2.resnets.0.conv1", 512, 256, 3, 4096],
  ["128x128", "up_blocks.2.resnets.0.conv2", 256, 256, 3, 4096],
  ["128x128", "up_blocks.2.resnets.0.conv_shortcut", 512, 256, 1, 4096],
  ["128x128", "up_blocks.2.upsamplers.0.conv", 256, 256, 3, 16384],
  ["128x128", "up_blocks.3.resnets.0.conv1", 256, 128, 3, 16384],
  ["128x128", "up_blocks.3.resnets.0.conv2", 128, 128, 3, 16384],
  ["128x128", "up_blocks.3.resnets.0.conv_shortcut", 256, 128, 1, 16384],
  ["128x128", "conv_out", 128, 3, 3, 16384],
  ["144x128", "conv_in", 16, 512, 3, 288],
  ["144x128", "mid_block.resnets.0.conv1", 512, 512, 3, 288],
  ["144x128", "up_blocks.0.upsamplers.0.conv", 512, 512, 3, 1152],
  ["144x128", "up_blocks.1.upsamplers.0.conv", 512, 512, 3, 4608],
  ["144x128", "up_blocks.2.resnets.0.conv1", 512, 256, 3, 4608],
  ["144x128", "up_blocks.2.resnets.0.conv2", 256, 256, 3, 4608],
  ["144x128", "up_blocks.2.resnets.0.conv_shortcut", 512, 256, 1, 4608],
  ["144x128", "up_blocks.2.upsamplers.0.conv", 256, 256, 3, 18432],
  ["144x128", "up_blocks.3.resnets.0.conv1", 256, 128, 3, 18432],
  ["144x128", "up_blocks.3.resnets.0.conv2", 128, 128, 3, 18432],
  ["144x128", "up_blocks.3.resnets.0.conv_shortcut", 256, 128, 1, 18432],
  ["144x128", "conv_out", 128, 3, 3, 18432],
  ["192x128", "conv_in", 16, 512, 3, 384],
  ["192x128", "mid_block.resnets.0.conv1", 512, 512, 3, 384],
  ["192x128", "up_blocks.0.upsamplers.0.conv", 512, 512, 3, 1536],
  ["192x128", "up_blocks.1.upsamplers.0.conv", 512, 512, 3, 6144],
  ["192x128", "up_blocks.2.resnets.0.conv1", 512, 256, 3, 6144],
  ["192x128", "up_blocks.2.resnets.0.conv2", 256, 256, 3, 6144],
  ["192x128", "up_blocks.2.resnets.0.conv_shortcut", 512, 256, 1, 6144],
  ["192x128", "up_blocks.2.upsamplers.0.conv", 256, 256, 3, 24576],
  ["192x128", "up_blocks.3.resnets.0.conv1", 256, 128, 3, 24576],
  ["192x128", "up_blocks.3.resnets.0.conv2", 128, 128, 3, 24576],
  ["192x128", "up_blocks.3.resnets.0.conv_shortcut", 256, 128, 1, 24576],
  ["192x128", "conv_out", 128, 3, 3, 24576],
  ["128x256", "conv_in", 16, 512, 3, 512],
  ["128x256", "mid_block.resnets.0.conv1", 512, 512, 3, 512],
  ["128x256", "up_blocks.0.upsamplers.0.conv", 512, 512, 3, 2048],
  ["128x256", "up_blocks.1.upsamplers.0.conv", 512, 512, 3, 8192],
  ["128x256", "up_blocks.2.resnets.0.conv1", 512, 256, 3, 8192],
  ["128x256", "up_blocks.2.resnets.0.conv2", 256, 256, 3, 8192],
  ["128x256", "up_blocks.2.resnets.0.conv_shortcut", 512, 256, 1, 8192],
  ["128x256", "up_blocks.2.upsamplers.0.conv", 256, 256, 3, 32768],
  ["128x256", "up_blocks.3.resnets.0.conv1", 256, 128, 3, 32768],
  ["128x256", "up_blocks.3.resnets.0.conv2", 128, 128, 3, 32768],
  ["128x256", "up_blocks.3.resnets.0.conv_shortcut", 256, 128, 1, 32768],
  ["128x256", "conv_out", 128, 3, 3, 32768],
  ["256x256", "conv_in", 16, 512, 3, 1024],
  ["256x256", "up_blocks.1.upsamplers.0.conv", 512, 512, 3, 16384],
  ["256x256", "up_blocks.2.resnets.0.conv1", 512, 256, 3, 16384],
  ["256x256", "up_blocks.2.resnets.0.conv_shortcut", 512, 256, 1, 16384],
  ["256x256", "up_blocks.2.upsamplers.0.conv", 256, 256, 3, 65536],
  ["256x256", "up_blocks.3.resnets.0.conv1", 256, 128, 3, 65536],
  ["256x256", "up_blocks.3.resnets.0.conv2", 128, 128, 3, 65536],
  ["256x256", "up_blocks.3.resnets.0.conv_shortcut", 256, 128, 1, 65536],
  ["256x256", "conv_out", 128, 3, 3, 65536],
 ],
 "gn": [
  ["128x128", "mid_block.resnets.0.norm1", 512, 256],
  ["128x128", "up_blocks.1.resnets.0.norm1", 512, 1024],
  ["128x128", "up_blocks.2.resnets.0.norm1", 512, 4096],
  ["128x128", "up_blocks.2.resnets.0.norm2", 256, 4096],
  ["128x128", "up_blocks.3.resnets.0.norm1", 256, 16384],
  ["128x128", "up_blocks.3.resnets.0.norm2", 128, 16384],
  ["144x128", "mid_block.resnets.0.norm1", 512, 288],
  ["144x128", "up_blocks.1.resnets.0.norm1", 512, 1152],
  ["144x128", "up_blocks.2.resnets.0.norm1", 512, 4608],
  ["144x128", "up_blocks.2.resnets.0.norm2", 256, 4608],
  ["144x128", "up_blocks.3.resnets.0.norm1", 256, 18432],
  ["144x128", "up_blocks.3.resnets.0.norm2", 128, 18432],
  ["192x128", "mid_block.resnets.0.norm1", 512, 384],
  ["192x128", "up_blocks.1.resnets.0.norm1", 512, 1536],
  ["192x128", "up_blocks.2.resnets.0.norm1", 512, 6144],
  ["192x128", "up_blocks.2.resnets.0.norm2", 256, 6144],
  ["192x128", "up_blocks.3.resnets.0.norm1", 256, 24576],
  ["192x128", "up_blocks.3.resnets.0.norm2", 128, 24576],
  ["128x256", "mid_block.resnets.0.norm1", 512, 512],
  ["128x256", "up_blocks.1.resnets.0.norm1", 512, 2048],
  ["128x256", "up_blocks.2.resnets.0.norm1", 512, 8192],
  ["128x256", "up_blocks.2.resnets.0.norm2", 256, 8192],
  ["128x256", "up_blocks.3.resnets.0.norm1", 256, 32768],
  ["128x256", "up_blocks.3.resnets.0.norm2", 128, 32768],
  ["256x256", "up_blocks.2.resnets.0.norm1", 512, 16384],
  ["256x256", "up_blocks.3.resnets.0.norm1", 256, 65536],
  ["256x256", "up_blocks.3.resnets.0.norm2", 128, 65536],
 ],
 "up": [
  ["128x128", "up_blocks.0.upsamplers.0", 512, 16, 16, ["16x16", "8x32"]],
  ["128x128", "up_blocks.1.upsamplers.0", 512, 32, 32, ["32x32", "16x64"]],
  ["128x128", "up_blocks.2.upsamplers.0", 256, 64, 64, ["64x64", "32x128"]],
  ["144x128", "up_blocks.0.upsamplers.0", 512, 18, 16, ["18x16"]],
  ["144x128", "up_blocks.1.upsamplers.0", 512, 36, 32, ["36x32"]],
  ["144x128", "up_blocks.2.upsamplers.0", 256, 72, 64, ["72x64"]],
  ["192x128", "up_blocks.0.upsamplers.0", 512, 24, 16, ["24x16"]],
  ["192x128", "up_blocks.1.upsamplers.0", 512, 48, 32, ["48x32"]],
  ["192x128", "up_blocks.2.upsamplers.0", 256, 96, 64, ["96x64"]],
  ["128x256", "up_blocks.0.upsamplers.0", 512, 16, 32, ["16x32"]],
  ["128x256", "up_blocks.1.upsamplers.0", 512, 32, 64, ["32x64"]],
  ["128x256", "up_blocks.2.upsamplers.0", 256, 64, 128, ["64x128"]],
  ["256x256", "up_blocks.1.upsamplers.0", 512, 64, 64, ["64x64"]],
  ["256x256", "up_blocks.2.upsamplers.0", 256, 128, 128, ["128x128"]],
 ],
}
# ---- generated end ----


def frozen_core() -> dict:
    """FROZEN expanded into dicts with the derived fields filled in."""
    f = FROZEN
    conv = [dict(size=r[0], module=r[1], Cin=r[2], Cout=r[3], k=r[4],
                 kind="conv", **conv_mkn(r[2], r[3], r[4], r[5]))
            for r in f["conv"]]
    gn = [dict(size=r[0], module=r[1], C=r[2], S=r[3], kind="gn",
               **gn_plan(r[2], r[3]))
          for r in f["gn"]]
    up = [dict(size=r[0], module=r[1], C=r[2], H=r[3], W=r[4],
               variants=list(r[5]), kind="up",
               **up_plane(r[2], r[3], r[4]))
          for r in f["up"]]
    return dict(sizes=list(f["sizes"]), conv=conv, gn=gn, up=up)


def frozen_counts() -> dict:
    f = frozen_core()
    return dict(conv=len(f["conv"]), gn=len(f["gn"]), up=len(f["up"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--derive", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--frozen", action="store_true")
    ap.add_argument("--write-json", default=None)
    a = ap.parse_args()
    if a.frozen:
        print(json.dumps(frozen_core(), indent=1))
        return 0
    if not (a.derive or a.check or a.write_json):
        ap.print_help()
        return 2
    got = derive()
    print(f"[counts] conv {len(got['conv'])}  gn {len(got['gn'])}  "
          f"up {len(got['up'])}")
    for s in got["sizes"]:
        p = got["per_size"][s]
        print(f"[counts] {s:8s} conv {len(p['conv']):2d} gn {len(p['gn'])} "
              f"up {len(p['up'])}  (instances {p['n_conv_inst']}/"
              f"{p['n_gn_inst']}/{p['n_up_inst']})")
    if a.write_json:
        pathlib.Path(a.write_json).write_text(
            json.dumps(got, indent=1), encoding="utf-8")
        print(f"[json] -> {a.write_json}")
    if a.check:
        want = frozen_core()
        def keys(rows, fields):
            return [tuple(r[f] for f in fields) for r in rows]
        checks = [
            ("conv", ("Cin", "Cout", "k", "M")),
            ("gn", ("C", "S")),
            ("up", ("C", "S", "variants")),
        ]
        all_ok = True
        for kind, fields in checks:
            same = keys(want[kind], fields) == keys(got[kind], fields)
            all_ok &= same
            print(f"[check] FROZEN {kind} == derived: {same} "
                  f"({len(want[kind])} rows)")
        print(f"[check] {'PASS' if all_ok else 'FAIL'}")
        return 0 if all_ok else 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
