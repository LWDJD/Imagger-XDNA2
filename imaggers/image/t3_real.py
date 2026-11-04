# SPDX-License-Identifier: MIT
"""t3_real.py -- the REAL weights (zimage-port/weights.bfp16) end-to-end image
for task T3 / M2: 34-block DiT, 8 steps, the FFN of every unified-stream block
dispatched to the NPU.

Why this file exists: imaggers/image/image_run.py ran the loop with
dit_block.make_weights(geom, rng(0)) -- ONE synthetic block, applied as many
times as the caller liked.  This runs the checkpoint:

    noise_refiner.0/1 (S=64) -> context_refiner.0/1 (S=32) -> cat -> 96 tokens
    -> layers.0..29 (S=96) -> FinalLayer -> unpatchify -> VAE

Block math is diffusers 0.36 ZImageTransformerBlock verbatim:

    scale_msa, gate_msa, scale_mlp, gate_mlp = adaLN(adaln).chunk(4)
    gate = tanh(gate);  scale = 1 + scale
    x = x + gate_msa * attention_norm2(attn(attention_norm1(x) * scale_msa))
    x = x + gate_mlp * ffn_norm2(ff(w1/w3)(ffn_norm1(x) * scale_mlp))

--stage blockcheck proves the numpy block against the diffusers module itself,
on the real weights, before any image is believed.

Weights are decoded from weights.bfp16 with pack_bfp16.unpack (the repo codec),
so the host reference uses the SAME bfp16-grid weights the device reads: the
host-vs-device difference is arithmetic, not quantisation.

Stages:  blockcheck | host | prep | device | verify
Driver:  py3.12 for all but --stage device (py3.10 + pyxrt).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
NPU = HERE.parent.parent
WS = NPU.parent
PORT = WS / "zimage-port"
REPO = WS / "zimage-bfp16-repo"
BLOCK = NPU / "imaggers" / "block"
ORACLE_DIR = NPU / "imaggers" / "oracle"
PIPE = NPU / "imaggers" / "pipeline"
WBF = PORT / "weights.bfp16"
HOSTT = REPO / "_models" / "Z-Image-Turbo" / "host_tensors.safetensors"
MODELS = PORT / "_models" / "Z-Image-Turbo"
ORACLE96 = PIPE / "oracle96.png"

PY310 = pathlib.Path(r"D:\Users\15081\AppData\Local\GAIA\python\python.exe")
XRT_DIR = pathlib.Path(
    r"D:\Project\deepseek_harness\test7\WStmp\driver376\npu_mcdm_stack_prod")

D, HEADS, HD, FF, SEQ, CAP = 3840, 30, 128, 10240, 96, 32
IMG_GRID = (1, 8, 8)
EPS = 1e-5
NORM_FINAL_EPS = 1e-6
T_SCALE = 1000.0
F32 = np.float32

STAGE3 = BLOCK / "out" / "step3_M96_K3840_N110240_m16_k8_n64_G4x6L10"
KERNEL = "MLIR_AIE"
GROUPS = {"A": 3, "SBG": 4, "BG": 5, "B3": 6, "X": 7, "Y": 8}
PAYLOAD_ORDER = ("A", "SBG", "BG", "B3", "X", "Y")
ACT_BYTES = SEQ * D * 2
SBG_BYTES = SEQ * 3 * D * 2
BG_BYTES = 2 * FF * D * 9 // 8
B3_BYTES = D * FF * 9 // 8
TIMEOUT_MS = int(os.environ.get("T3_TIMEOUT_MS", "300000"))
COMPLETED = 4

PREPDIR = "t3real_prep"
REFINERS = ("noise_refiner.0", "noise_refiner.1",
            "context_refiner.0", "context_refiner.1")


# ---------------------------------------------------------------------------
# the record's own modules, imported read-only
# ---------------------------------------------------------------------------
_ORACLE_LOCK = None


def _oracle():
    """The record's dit_block module, loaded at most once.

    T22_CHAIN2B (2026-11-04): this used to publish the module into sys.modules
    BEFORE executing its body, so a second caller arriving in that window got a
    module with no attributes at all.  Measured on the two-chain path, twice:

        AttributeError: module 'dit_block' has no attribute 'Geometry'
        (t3_real.py:395 rope_cos_sin, chain B, ~1 s after chain A passed it)

    One process with one chain can never hit it, which is why it survived.  The
    fix is a lock plus publish-after-execute: the same module object, the same
    order for a single caller, and a second caller either waits or re-loads into
    a private module object -- never a half-built one.
    """
    global _ORACLE_LOCK
    import importlib.util
    if _ORACLE_LOCK is None:
        import threading
        _ORACLE_LOCK = threading.Lock()
    if str(ORACLE_DIR) not in sys.path:
        sys.path.insert(0, str(ORACLE_DIR))
    mod = sys.modules.get("dit_block")
    if mod is not None and getattr(mod, "_DB_READY", False):
        return mod
    with _ORACLE_LOCK:
        mod = sys.modules.get("dit_block")
        if mod is not None and hasattr(mod, "Geometry"):
            return mod
        spec = importlib.util.spec_from_file_location(
            "dit_block", ORACLE_DIR / "dit_block.py")
        mod = importlib.util.module_from_spec(spec)
        # The module MUST be in sys.modules while its body runs: dit_block.py
        # declares @dataclass classes and dataclasses._is_type resolves
        # sys.modules[cls.__module__], so publishing after exec is not an
        # option (measured: AttributeError "'NoneType' object has no attribute
        # '__dict__'" at dit_block.py:159 class Geometry).  So publish first --
        # but hold the lock and mark it ready, so a second caller waits for a
        # COMPLETE module instead of reading a half-built one.
        sys.modules["dit_block"] = mod
        spec.loader.exec_module(mod)
        mod._DB_READY = True
    return mod


def _ffn():
    if str(BLOCK) not in sys.path:
        sys.path.append(str(BLOCK))
    import step4_run_ffn as F
    return F


def _packbf():
    if str(NPU) not in sys.path:
        sys.path.insert(0, str(NPU))
    import pack_bfp16 as PB
    return PB


# ---------------------------------------------------------------------------
# the REAL weights
# ---------------------------------------------------------------------------
class Real:
    """weights.bfp16 -> per-block f32 tensors (the device's own byte grid)."""

    TAILS = ("attention.norm_q.weight", "attention.norm_k.weight",
             "attention.to_q.weight", "attention.to_k.weight",
             "attention.to_v.weight", "attention.to_out.0.weight",
             "feed_forward.w1.weight", "feed_forward.w2.weight",
             "feed_forward.w3.weight", "attention_norm1.weight",
             "ffn_norm1.weight", "attention_norm2.weight",
             "ffn_norm2.weight", "adaLN_modulation.0.weight",
             "adaLN_modulation.0.bias")

    def __init__(self):
        with open(WBF, "rb") as f:
            hlen = int.from_bytes(f.read(8), "little")
            hdr = json.loads(f.read(hlen))
        self.base = 8 + hlen
        self.rec = {r["name"]: r for r in hdr["tensors"]}
        self.fh = open(WBF, "rb")
        self.PB = _packbf()
        self.unpacked_bytes = 0

    def raw(self, name):
        r = self.rec[name]
        self.fh.seek(self.base + r["offset"])
        return self.fh.read(r["length"])

    def get(self, name):
        r = self.rec[name]
        blob = self.raw(name)
        self.unpacked_bytes += len(blob)
        if r["packed_bfp16"]:
            n, k = r["shape"]
            return self.PB.unpack(blob, n, k, r["tile_rows"])
        return np.frombuffer(blob, np.float32).reshape(r["shape"]).copy()

    def block(self, key):
        g = lambda t: self.get(key + "." + t)
        return {
            "norm_q": g("attention.norm_q.weight"),
            "norm_k": g("attention.norm_k.weight"),
            "to_q": g("attention.to_q.weight"),
            "to_k": g("attention.to_k.weight"),
            "to_v": g("attention.to_v.weight"),
            "to_out": g("attention.to_out.0.weight"),
            "w1": g("feed_forward.w1.weight"),
            "w2": g("feed_forward.w2.weight"),
            "w3": g("feed_forward.w3.weight"),
            "attention_norm1": g("attention_norm1.weight"),
            "ffn_norm1": g("ffn_norm1.weight"),
            "attention_norm2": g("attention_norm2.weight"),
            "ffn_norm2": g("ffn_norm2.weight"),
            "adaln_w": (g("adaLN_modulation.0.weight")
                        if key + ".adaLN_modulation.0.weight" in self.rec
                        else None),
            "adaln_b": (g("adaLN_modulation.0.bias")
                        if key + ".adaLN_modulation.0.bias" in self.rec
                        else None),
        }


def host_tensors(tag="t3real"):
    """The 15 host-side tensors, cached as .npz by --stage host.

    The device leg (py3.10) has no safetensors, and using the SAME bytes the
    host reference used is a stronger guarantee than re-reading the file.
    """
    npz = HERE / ("%s_host_tensors.npz" % tag)
    if npz.exists():
        z = np.load(npz)
        return {k: z[k] for k in z.files}
    from safetensors import safe_open
    out = {}
    with safe_open(str(HOSTT), framework="np") as f:
        for k in f.keys():
            out[k] = np.asarray(f.get_tensor(k), np.float32)
    return out


# ---------------------------------------------------------------------------
# real-block arithmetic (f32, the model's own dtype)
# ---------------------------------------------------------------------------
_RMS_NB = None


def _rms_kernels():
    # 2026-11-01 parent: one-pass numba rms.  numpy does (x*x).mean, x/sqrt,
    # *w and a final astype -- four 15 MB passes.  Measured 16.43 -> 0.56 ms
    # (29x) with maxabs 9.5e-07 (summation order differs, corr 1.00000000).
    global _RMS_NB
    if _RMS_NB is None:
        try:
            from numba import njit, prange

            @njit(parallel=True, cache=True)
            def _w(x, w, out, eps):
                for i in prange(x.shape[0]):
                    s = 0.0
                    for j in range(x.shape[1]):
                        v = x[i, j]
                        s += v * v
                    inv = 1.0 / np.sqrt(np.float32(s / x.shape[1] + eps))
                    for j in range(x.shape[1]):
                        out[i, j] = x[i, j] * inv * w[j]
                return out

            @njit(parallel=True, cache=True)
            def _p(x, out, eps):
                for i in prange(x.shape[0]):
                    s = 0.0
                    for j in range(x.shape[1]):
                        v = x[i, j]
                        s += v * v
                    inv = 1.0 / np.sqrt(np.float32(s / x.shape[1] + eps))
                    for j in range(x.shape[1]):
                        out[i, j] = x[i, j] * inv
                return out
            _RMS_NB = (_w, _p)
        except Exception:
            _RMS_NB = False
    return _RMS_NB or None


def _rms_fast(x, w, eps):
    import os as _o
    if _o.environ.get("T22_REF_RMS", "0") != "1":
        return None
    k = _rms_kernels()
    if k is None:
        return None
    arr = np.asarray(x, np.float32)
    shp = arr.shape
    xc = np.ascontiguousarray(arr).reshape(-1, shp[-1])
    out = np.empty_like(xc)
    wa = None if w is None else np.ascontiguousarray(np.asarray(w, np.float32))
    if wa is not None and wa.ndim != 1:
        return None
    try:
        if wa is None:
            k[1](xc, out, np.float32(eps))
        else:
            k[0](xc, out, wa, np.float32(eps))
        return out.reshape(shp)
    except Exception:
        return None


def rms(x, w=None, eps=EPS):
    _fast = _rms_fast(x, w, eps)
    if _fast is not None:
        return _fast
    x = np.asarray(x, np.float32)
    v = (x * x).mean(-1, keepdims=True, dtype=np.float32)
    y = (x / np.sqrt(v + F32(eps))).astype(np.float32)
    return y if w is None else (y * np.asarray(w, np.float32)).astype(np.float32)


# ---------------------------------------------------------------------------
# T22_REF_BF16=1: the refiner block's seven big GEMMs on the bf16 pipe
# ---------------------------------------------------------------------------
# Opt-in, default 0 (the delivered path below is untouched when it is off).
#
# WHY IT IS FREE ON THE WEIGHT SIDE: every refiner weight was decoded from
# weights.bfp16, i.e. it sits on an 8-bit-mantissa grid, and bf16 has the same
# 8-bit mantissa -- so f32->bf16 of these weights is EXACT (bf16->f32 round trip
# is bit-identical; verified over all 176.9 M values of noise_refiner.0 in
# imaggers/hostopt/bench_refiner_bf16.py).  The bf16 copy IS the delivered
# weight, and it is built once per weight object and kept.
#
# WHAT DOES CHANGE: the GEMM accumulates in bf16xbf16->f32 (torch's AVX512-BF16
# path) instead of f32, and the ACTIVATION is rounded to bf16 per call.  The
# measured cost of that on the real weights, per block, is in
# imaggers/hostopt/REFINER-BF16-REPORT.md.
#
# SCOPE: only the shapes the refiner block actually runs -- (S,3840)@(3840,3840)
# x4, (S,3840)@(3840,10240) x2 and (S,10240)@(10240,3840).  Everything else
# (the 1-row adaLN modulation, the 64-wide x_embedder, the final layer) keeps
# the delivered f32 path even with the flag on, because it is not the arithmetic
# this flag is about.
_BF16_W = {}                                  # id(f32 W) -> (weakref, bf16 t)
_BF16_LOG = {"casts": 0, "cast_s": 0.0, "hits": 0}
_TH = None


def _bf16_on():
    return os.environ.get("T22_REF_BF16", "0") == "1"


def _torch_bf16():
    """torch, once, with the product's own thread knob (ZIMAGE_CHUNK_THREADS).

    chunked.py:_torch() already owns that variable for the host attention; the
    refiner runs BEFORE the first attention call, so without this it would use
    torch's default pool on step 0 and the configured one afterwards.
    """
    global _TH
    if _TH is None:
        try:
            import torch as _th
        except ImportError as exc:                   # py3.12 AIE interpreter
            raise ImportError(
                "T22_REF_BF16=1 needs torch; this interpreter has none (%s). "
                "The product's host leg is py310 GAIA, which has it." % (exc,)
            ) from exc
        n = int(os.environ.get("ZIMAGE_CHUNK_THREADS", "0") or 0)
        if n > 0:
            _th.set_num_threads(n)
        _TH = _th
    return _TH


def _bf16_w(W):
    """The bf16 copy of one weight: built once per array object, then reused."""
    import time as _t
    import weakref as _wr
    key = id(W)
    ent = _BF16_W.get(key)
    if ent is not None and ent[0]() is W:
        _BF16_LOG["hits"] += 1
        return ent[1]
    th = _torch_bf16()
    t0 = _t.perf_counter()
    t = th.from_numpy(np.ascontiguousarray(W, np.float32)).to(th.bfloat16)
    _BF16_LOG["casts"] += 1
    _BF16_LOG["cast_s"] += _t.perf_counter() - t0
    if len(_BF16_W) > 64:                     # bounded: drop the dead entries
        for k in [k for k, v in _BF16_W.items() if v[0]() is None]:
            _BF16_W.pop(k, None)
    _BF16_W[key] = (_wr.ref(W), t)
    return t


def _bf16_ok(x, W):
    """True only for the seven big refiner GEMMs (see the header note)."""
    return (isinstance(W, np.ndarray) and W.dtype == np.float32 and W.ndim == 2
            and W.shape[0] >= 512 and W.shape[1] >= 512
            and x.ndim == 2 and x.shape[0] >= 8 and x.shape[-1] == W.shape[1])


def _lin_bf16(x, W):
    th = _torch_bf16()
    xa = np.ascontiguousarray(np.asarray(x, np.float32))
    yt = th.matmul(th.from_numpy(xa).to(th.bfloat16), _bf16_w(W).t())
    return yt.to(th.float32).numpy()


def bf16_stats():
    """(casts, cast seconds, cache hits) -- for the report's one-time cost."""
    return dict(_BF16_LOG)


def lin(x, W, b=None):
    if _bf16_on():
        xa = np.asarray(x, np.float32)
        if _bf16_ok(xa, W):
            y = _lin_bf16(xa, W)
            return (y if b is None
                    else (y + np.asarray(b, np.float32)).astype(np.float32))
    y = np.asarray(np.asarray(x, np.float32) @ np.asarray(W, np.float32).T,
                   np.float32)
    return y if b is None else (y + np.asarray(b, np.float32)).astype(np.float32)


def silu(x):
    x = np.asarray(x, np.float32)
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def softmax(x):
    m = x.max(-1, keepdims=True)
    e = np.exp(x - m)
    return (e / e.sum(-1, keepdims=True)).astype(np.float32)


def rope_cos_sin(ids):
    DB = _oracle()
    ids = np.asarray(ids)
    geom = DB.Geometry(hidden=D, heads=HEADS, head_dim=HD, ffn_inner=FF,
                       seq=int(ids.shape[0]))
    cfg = DB.DeviceCfg()
    c, s = DB.rope_table(geom, cfg, ids=ids)
    return c.astype(np.float32), s.astype(np.float32)


def apply_rope(x, cos, sin):
    c, s = cos[:, None, :], sin[:, None, :]
    xe, xo = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = xe * c - xo * s
    out[..., 1::2] = xe * s + xo * c
    return out


def real_ids():
    """zimage_pipeline.position_ids(32, (1,8,8)): cap (1..32,0,0), img
    (33+f, h, w) with h,w starting at ZERO (the diffusers patchify_and_embed
    convention -- NOT the zero-centred variant in imaggers/block/step4_oracle)."""
    ft, ht, wt = IMG_GRID
    cap = np.zeros((CAP, 3), np.int64)
    cap[:, 0] = np.arange(1, 1 + CAP)
    zz, yy, xx = np.meshgrid(np.arange(CAP + 1, CAP + 1 + ft),
                             np.arange(ht), np.arange(wt), indexing="ij")
    img = np.stack([zz.reshape(-1), yy.reshape(-1), xx.reshape(-1)], 1)
    return cap.astype(np.int64), img.astype(np.int64)


_SMS = None


def _softmax_scaled(x, scale):
    # 2026-11-01 parent: fused (x*scale) softmax, in place, numba.
    # The numpy path was: S = qh@kh.T ; S*scale (252 MB copy, 39.7 ms) ;
    # softmax(np.asarray(S,f32)) (235.5 ms).  softmax(x*c) == softmax over
    # (x*c) with c>0, so the scale folds into the max/exp with no extra
    # pass; measured 235.5+39.7 -> 27.5 ms for the (60,1024,1024) block.
    global _SMS
    if _SMS is None:
        try:
            from numba import njit, prange

            @njit(parallel=True, cache=True)
            def _k(x, sc):
                r = x.shape[0] * x.shape[1]
                S = x.shape[2]
                xf = x.reshape(r, S)
                for i in prange(r):
                    m = -1e30
                    for j in range(S):
                        if xf[i, j] > m:
                            m = xf[i, j]
                    ms = m * sc
                    s = 0.0
                    for j in range(S):
                        e = np.exp(xf[i, j] * sc - ms)
                        xf[i, j] = e
                        s += e
                    inv = 1.0 / s
                    for j in range(S):
                        xf[i, j] *= inv
                return x
            _SMS = _k
        except Exception:
            _SMS = False
    return _SMS or None


_REFCHUNK_NOTICE = [0]


def _ref_attn(qh, kh, vh, sc):
    """2026-11-03 refchunk: query-blocked attention, or None for the delivered code.

    The delivered expression below materialises (HEADS, S, S) f32.  At S=8160
    (1088x1920, the 1080p shape) that is 30 x 8160 x 8160 x 4 B = 7.44 GiB, and
    the job died in softmax()'s np.exp asking for it (job
    20261103-150606-10d726).  ref_attn_block.py blocks the QUERY axis: one
    (HEADS, rows, S) block at a time, the softmax still over the full key axis,
    so the arithmetic is unchanged (gate_refchunk.py: maxabs 0.000e+00 at S=1024
    and S=4096, and between three block sizes at S=8160).

    Returns None -- and imports nothing that matters -- unless the score matrix
    would exceed T22_REF_QCHUNK_GB (default 2.0 GiB).  Every shape that runs
    today is at or below 1.88 GiB (S=4096), so the delivered expression below is
    what executes for all of them.  See imaggers/util/refchunk/REFCHUNK-REPORT.md.
    """
    import os as _o
    try:
        import ref_attn_block as RB
    except Exception:                                       # noqa: BLE001
        if HEADS * qh.shape[1] * qh.shape[1] * 4 > 2.0 * 2.0 ** 30:
            print("[refchunk] ref_attn_block.py is missing; the refiner's "
                  "(heads, %d, %d) f32 score matrix will not fit"
                  % (qh.shape[1], qh.shape[1]), flush=True)
        return None
    if not RB.wanted(HEADS, qh.shape[1]):
        return None
    if not _REFCHUNK_NOTICE[0]:
        _REFCHUNK_NOTICE[0] = 1
        print("[refchunk] S=%d: (heads, S, S) f32 is %.2f GiB > %.2f GiB; "
              "attention runs in %d query row(s) per block (%s)"
              % (qh.shape[1], RB.score_bytes(HEADS, qh.shape[1]) / 2**30,
                 RB.ceiling_bytes() / 2**30, RB.block_rows(HEADS, qh.shape[1]),
                 RB.mode()), flush=True)
        for _v in ("T22_REF_TORCH", "T22_REF_FAST"):
            if _o.environ.get(_v, "0") == "1":
                print("[refchunk] %s=1 does not bound the score matrix; the "
                      "query-blocked f32 path runs instead" % _v, flush=True)
    _nb = (_softmax_scaled(qh, kh)
           if _o.environ.get("T22_REF_SMNB", "0") == "1" else None)
    return RB.attn(qh, kh, vh, sc, softmax, _nb)


def attention(x1, Wb, cos, sin):
    q = lin(x1, Wb["to_q"]).reshape(-1, HEADS, HD)
    k = lin(x1, Wb["to_k"]).reshape(-1, HEADS, HD)
    v = lin(x1, Wb["to_v"]).reshape(-1, HEADS, HD)
    q = apply_rope(rms(q, Wb["norm_q"]), cos, sin)
    k = apply_rope(rms(k, Wb["norm_k"]), cos, sin)
    qh = np.ascontiguousarray(q.transpose(1, 0, 2))
    kh = np.ascontiguousarray(k.transpose(1, 0, 2))
    vh = np.ascontiguousarray(v.transpose(1, 0, 2))
    _blk = _ref_attn(qh, kh, vh, float(1.0 / np.sqrt(HD)))
    if _blk is not None:
        return lin(_blk, Wb["to_out"])
    # 2026-11-01: opt-in torch SDPA for the refiner's global attention.
    # numpy on (60,1024,64) x (60,64,1024) runs at ~96 GFLOP/s here while
    # the same shapes in torch reach ~300; default 0 keeps numpy.
    import os as _os
    if _os.environ.get("T22_REF_TORCH", "0") == "1":
        import torch as _th
        import torch.nn.functional as _F
        _o = _F.scaled_dot_product_attention(
            _th.from_numpy(qh), _th.from_numpy(kh), _th.from_numpy(vh))
        o = _o.permute(1, 0, 2).contiguous().numpy().reshape(-1, D)
        return lin(o, Wb["to_out"])
    _sc = float(1.0 / np.sqrt(HD))
    if _os.environ.get("T22_REF_FAST", "0") == "1":
        # 2026-11-01 parent: torch bmm for the two matmuls + the fused numba
        # softmax.  Measured on (60,1024,64)x(60,64,1024): qk numpy 60.3 ->
        # torch 26.1 ms (bit-identical), pv 20.6 -> 14.6 ms (maxabs 2.7e-08),
        # and the numpy softmax+scale 275.2 -> 27.5 ms.
        import torch as _th
        _k = _softmax_scaled(qh, kh)
        _qt = _th.from_numpy(qh)
        _kt = _th.from_numpy(kh).transpose(1, 2)
        _S = _th.matmul(_qt, _kt).numpy()
        if _k is not None:
            P = _k(_S, np.float32(_sc))
        else:
            P = softmax(np.asarray(_S * F32(_sc), np.float32))
        _o = _th.matmul(_th.from_numpy(P), _th.from_numpy(vh)).numpy()
        o = _o.astype(np.float32).transpose(1, 0, 2).reshape(-1, D)
        return lin(o, Wb["to_out"])
    _k = _softmax_scaled(qh, kh) if _os.environ.get("T22_REF_SMNB", "0") == "1" else None
    if _k is not None:
        P = _k(np.ascontiguousarray(qh @ kh.transpose(0, 2, 1)), np.float32(_sc))
    else:
        S = (qh @ kh.transpose(0, 2, 1)) * F32(_sc)
        P = softmax(np.asarray(S, np.float32))
    o = (P @ vh).astype(np.float32).transpose(1, 0, 2).reshape(-1, D)
    return lin(o, Wb["to_out"])


REF_FFN = None      # ref_ffn_block, imported on the first blocking call
_REF_FFN_NOTICE = [0]


def ffn_raw(m, Wb):
    """The noise refiner's FFN.

    The delivered expression materialises (S, 10240) f32 three times
    over (w1 out, w3 out, the product) plus silu's own output.  At the
    1080p shape S=8160 one of those is 318.75 MiB, and the job died in
    lin asking for it (job 20261103-230421-e19b34, rq_1080p_8f:
    "Unable to allocate 319. MiB for an array with shape (8160, 10240)").

    ref_ffn_block.py blocks the QUERY axis -- the FFN is row-local, so
    blocking cannot change a single FLOP.  The policy is two numbers:
    T22_REF_FFN_GB (default 0.30 GiB) decides WHEN (512^2 is 40 MiB and
    1024^2 is 160 MiB, so both keep the delivered expression), and
    T22_REF_FFN_BLOCK_GB (default 64 MiB) decides how big a block is.
    Measured at S=8160 with gate_ref_ffn.py: peak commit above the
    weights +0.734 GiB delivered vs +0.099 GiB blocked, sha256 equal.
    """
    global REF_FFN
    w1 = Wb["w1"]
    try:
        rows = int(m.shape[0])
        inner = int(w1.shape[0])
    except Exception:                                     # noqa: BLE001
        return lin(silu(lin(m, w1)) * lin(m, Wb["w3"]), Wb["w2"])
    if REF_FFN is None:
        try:
            import ref_ffn_block as _RF
            REF_FFN = _RF
        except Exception:                                 # noqa: BLE001
            REF_FFN = False
    if REF_FFN is False or not REF_FFN.wanted(rows, inner):
        return lin(silu(lin(m, w1)) * lin(m, Wb["w3"]), Wb["w2"])
    if not _REF_FFN_NOTICE[0]:
        _REF_FFN_NOTICE[0] = 1
        print("[refchunk] FFN S=%d: (rows, %d) f32 would be %.2f GiB; the "
              "refiner FFN runs %d row(s) per block (%s)"
              % (rows, inner, rows * inner * 4 / 2**30,
                 REF_FFN.block_rows(inner), REF_FFN.mode()), flush=True)
    return REF_FFN.ffn(m, w1, Wb["w3"], Wb["w2"], lin, silu)


def modulation(adaln_vec, Wb):
    """(scale_msa, gate_msa, scale_mlp, gate_mlp), or None for refiner blocks."""
    if Wb["adaln_w"] is None:
        return None
    m = lin(adaln_vec[None, :], Wb["adaln_w"], Wb["adaln_b"])[0].reshape(4, D)
    return (F32(1.0) + m[0], np.tanh(m[1]), F32(1.0) + m[2], np.tanh(m[3]))


def block_attn(x, Wb, mod, cos, sin):
    x = np.asarray(x, np.float32)
    n1 = rms(x, Wb["attention_norm1"])
    x1 = n1 if mod is None else (n1 * mod[0]).astype(np.float32)
    a = attention(x1, Wb, cos, sin)
    a2 = rms(a, Wb["attention_norm2"])
    xm = (x + a2) if mod is None else (x + mod[1] * a2)
    return xm.astype(np.float32), mod


def block_ffn_host(x_mid, Wb, mod):
    x = np.asarray(x_mid, np.float32)
    n2 = rms(x, Wb["ffn_norm1"])
    if mod is not None:
        n2 = (n2 * mod[2]).astype(np.float32)
    y = ffn_raw(n2, Wb)
    p = rms(y, Wb["ffn_norm2"])
    return ((x + p) if mod is None else (x + mod[3] * p)).astype(np.float32)


def ffn_input(x_mid, Wb, mod):
    x = np.asarray(x_mid, np.float32)
    n2 = rms(x, Wb["ffn_norm1"])
    return n2 if mod is None else (n2 * mod[2]).astype(np.float32)


def ffn_post(x_mid, y, Wb, mod):
    x = np.asarray(x_mid, np.float32)
    p = rms(y, Wb["ffn_norm2"])
    return ((x + p) if mod is None else (x + mod[3] * p)).astype(np.float32)


def full_block_host(x, Wb, mod, cos, sin):
    # 2026-11-01: env-gated split of the block's two halves (default off).
    import os as _os
    import time as _t
    _p = _os.environ.get("T22_REF_PROF", "0") == "1"
    _t0 = _t.perf_counter()
    xm, mod = block_attn(x, Wb, mod, cos, sin)
    _t1 = _t.perf_counter()
    y = block_ffn_host(xm, Wb, mod)
    _t2 = _t.perf_counter()
    if _p:
        print("[refprof] attn %.1f ms  ffn %.1f ms  total %.1f ms"
              % (1e3*(_t1-_t0), 1e3*(_t2-_t1), 1e3*(_t2-_t0)), flush=True)
    return y


# ---------------------------------------------------------------------------
# host pre/post: patchify, embeds, timestep, final layer
# ---------------------------------------------------------------------------
def patchify(lat):
    b, c, h, w = lat.shape
    gh, gw = h // 2, w // 2
    x = lat.reshape(b, c, gh, 2, gw, 2)
    return x.transpose(0, 2, 4, 1, 3, 5).reshape(b, gh * gw, c * 4)


def unpatchify(tok, chans, gh, gw):
    y = np.asarray(tok, np.float32).reshape(gh, gw, 2, 2, chans)
    return y.transpose(4, 0, 2, 1, 3).reshape(1, chans, gh * 2, gw * 2)


def t_emb(step_t, H):
    """diffusers TimestepEmbedder: the model is fed (1000 - t) after t*1000."""
    h = float(T_SCALE - float(step_t))
    half = 128
    freqs = np.exp(-np.log(10000.0) * np.arange(half, dtype=np.float64) / half)
    a = (h * freqs).astype(np.float32)
    e = np.concatenate([np.cos(a), np.sin(a)]).astype(np.float32)[None, :]
    e = lin(e, H["t_embedder.mlp.0.weight"], H["t_embedder.mlp.0.bias"])
    e = silu(e[0])
    return lin(e[None, :], H["t_embedder.mlp.2.weight"],
               H["t_embedder.mlp.2.bias"])[0].astype(np.float32)


def final_layer(x, c, H):
    scale = F32(1.0) + lin(silu(c)[None, :],
                           H["all_final_layer.2-1.adaLN_modulation.1.weight"],
                           H["all_final_layer.2-1.adaLN_modulation.1.bias"])
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True, dtype=np.float32)
    xl = ((x - mu) / np.sqrt(var + F32(NORM_FINAL_EPS)) * scale).astype(
        np.float32)
    return lin(xl, H["all_final_layer.2-1.linear.weight"],
               H["all_final_layer.2-1.linear.bias"])


def cap_embedding(H):
    """A NULL caption: 32 rows of the checkpoint's own learned pad token.

    The real pipeline replaces every padded caption row with cap_pad_token, so
    this is exactly what the model sees for a caption shorter than the pad
    length.  No text encoder is run (declared).
    """
    pad = np.asarray(H["cap_pad_token"], np.float32).reshape(1, D)
    return np.repeat(pad, CAP, axis=0)


def x_embedding(lat, H):
    tok = patchify(lat)[0]
    return lin(tok, H["all_x_embedder.2-1.weight"],
               H["all_x_embedder.2-1.bias"]).astype(np.float32)


# ---------------------------------------------------------------------------
# one denoise step (shared by the host and the device stages)
# ---------------------------------------------------------------------------
def step_once(lat, t_step, RW, H, ffn_impl, layers=30):
    cap_ids, img_ids = real_ids()
    c_cos, c_sin = rope_cos_sin(cap_ids)
    x_cos, x_sin = rope_cos_sin(img_ids)
    adaln = t_emb(t_step, H)
    rec = {"t": float(t_step), "layers": []}
    xb = x_embedding(lat, H)
    for key in REFINERS[:2]:
        Wb = RW.block(key)
        xb = full_block_host(xb, Wb, modulation(adaln, Wb), x_cos, x_sin)
        del Wb
    cb = cap_embedding(H)
    for key in REFINERS[2:]:
        Wb = RW.block(key)
        cb = full_block_host(cb, Wb, modulation(adaln, Wb), c_cos, c_sin)
        del Wb
    u = np.concatenate([xb, cb], axis=0).astype(np.float32)
    u_cos = np.concatenate([x_cos, c_cos], axis=0)
    u_sin = np.concatenate([x_sin, c_sin], axis=0)
    for i in range(layers):
        key = "layers.%d" % i
        Wb = RW.block(key)
        mod = modulation(adaln, Wb)
        x_mid, mod = block_attn(u, Wb, mod, u_cos, u_sin)
        u, r = ffn_impl(x_mid, Wb, mod, key)
        r["layer"] = i
        r["xmid_absmax"] = float(np.abs(x_mid).max())
        rec["layers"].append(r)
        del Wb
    out = final_layer(u, adaln, H)
    pred = unpatchify(out[:64], 16, 8, 8)
    rec["pred_absmax"] = float(np.abs(pred).max())
    rec["u_absmax"] = float(np.abs(u).max())
    return pred, rec


def ffn_host_impl(x_mid, Wb, mod, key):
    return block_ffn_host(x_mid, Wb, mod), {"ffn": "host-f32"}


def ffn_host_bf16in_impl(x_mid, Wb, mod, key):
    """CONTROL: the host FFN fed bf16(x_mid) -- exactly what the DEVICE is fed.

    It differs from the device in one respect only: inside the FFN it computes
    in f32 instead of BFP16.  So PSNR(control vs reference) is the loop's own
    sensitivity to a one-ulp input rounding, and PSNR(device vs reference) is
    that plus the device arithmetic.
    """
    DB = _oracle()
    xb = DB.round_bf16(np.asarray(x_mid, np.float32))
    y = ffn_raw(ffn_input(xb, Wb, mod), Wb)
    p = rms(y, Wb["ffn_norm2"])
    out = (np.asarray(x_mid, np.float32)
           + (p if mod is None else mod[3] * p)).astype(np.float32)
    return out, {"ffn": "host-f32-bf16-input"}


# ---------------------------------------------------------------------------
# schedule + metrics
# ---------------------------------------------------------------------------
def schedule(steps):
    if str(PORT) not in sys.path:
        sys.path.insert(0, str(PORT))
    from diffusers import FlowMatchEulerDiscreteScheduler
    s = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(MODELS), subfolder="scheduler")
    s.set_timesteps(steps)
    return (np.asarray([float(t) for t in s.timesteps], np.float64),
            s.sigmas.numpy().astype(np.float64))


def load_schedule(args):
    """The reference schedule, taken from the host stage when it exists.

    py3.10 (the device leg) has no diffusers, and re-deriving the schedule
    would be exactly the divergence the record forbids: the sigmas must be the
    ones the host image was made with.
    """
    for cand in (args.tag, "t3real"):
        tr = HERE / ("%s_traj.npz" % cand)
        if tr.exists():
            z = np.load(tr)
            ts, sg = z["timesteps"], z["sigmas"]
            if len(ts) >= args.steps:
                # a prefix is the same schedule: a smoke run must use the real
                # first timesteps, never a re-derived ones
                return (ts[:args.steps].astype(np.float64),
                        sg[:args.steps + 1].astype(np.float64))
    return schedule(args.steps)


def psnr(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    mse = float(((a - b) ** 2).mean())
    return (float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse))), mse


def ssim(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.ndim == 3:
        a = a.mean(-1)
        b = b.mean(-1)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu1, mu2 = a.mean(), b.mean()
    v1, v2 = a.var(), b.var()
    cov = ((a - mu1) * (b - mu2)).mean()
    return float((2 * mu1 * mu2 + c1) * (2 * cov + c2)
                 / ((mu1 ** 2 + mu2 ** 2 + c1) * (v1 + v2 + c2)))


def load_png(p):
    from PIL import Image
    return np.asarray(Image.open(p).convert("RGB"), np.float64) / 255.0


def save_png(arr, p):
    from PIL import Image
    Image.fromarray((np.asarray(arr) * 255.0 + 0.5).astype(np.uint8)).save(p)


def vae_decode(lat):
    if str(PIPE) not in sys.path:
        sys.path.insert(0, str(PIPE))
    import vae_decode as V
    return V.decode(V.load_vae(), np.asarray(lat, np.float32))


def sha16(b):
    return hashlib.sha256(bytes(b)).hexdigest()[:16]


# ===========================================================================
# STAGE: blockcheck -- the numpy block vs the diffusers module, real weights
# ===========================================================================
def stage_blockcheck(args) -> int:
    import torch
    import torch.nn.functional as TF
    # The installed torch predates the enable_gqa kwarg this diffusers build
    # passes to scaled_dot_product_attention.  n_kv_heads == n_heads here, so
    # GQA is a no-op: shim the kwarg away rather than touching either package.
    _sdpa = TF.scaled_dot_product_attention

    def _sdpa_compat(*a, **kw):
        kw.pop('enable_gqa', None)
        return _sdpa(*a, **kw)

    TF.scaled_dot_product_attention = _sdpa_compat
    from diffusers.models.transformers.transformer_z_image import (
        ZImageTransformerBlock)
    RW = Real()
    ok_all = True
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((SEQ, D))).astype(np.float32)
    adaln = (rng.standard_normal(256) * 0.1).astype(np.float32)
    cap_ids, img_ids = real_ids()
    ids = np.concatenate([img_ids, cap_ids], axis=0)
    cos, sin = rope_cos_sin(ids)
    freqs = torch.polar(torch.ones(SEQ, HD // 2, dtype=torch.float32),
                        torch.from_numpy(np.arctan2(sin, cos)
                                         .astype(np.float32)))
    for key in ("layers.0", "noise_refiner.0", "context_refiner.0"):
        Wb = RW.block(key)
        modu = Wb["adaln_w"] is not None
        blk = ZImageTransformerBlock(layer_id=0, dim=D, n_heads=HEADS,
                                     n_kv_heads=HEADS, norm_eps=EPS,
                                     qk_norm=True, modulation=modu)
        sd = {
            "attention.to_q.weight": Wb["to_q"],
            "attention.to_k.weight": Wb["to_k"],
            "attention.to_v.weight": Wb["to_v"],
            "attention.to_out.0.weight": Wb["to_out"],
            "attention.norm_q.weight": Wb["norm_q"],
            "attention.norm_k.weight": Wb["norm_k"],
            "feed_forward.w1.weight": Wb["w1"],
            "feed_forward.w2.weight": Wb["w2"],
            "feed_forward.w3.weight": Wb["w3"],
            "attention_norm1.weight": Wb["attention_norm1"],
            "ffn_norm1.weight": Wb["ffn_norm1"],
            "attention_norm2.weight": Wb["attention_norm2"],
            "ffn_norm2.weight": Wb["ffn_norm2"],
        }
        if modu:
            sd["adaLN_modulation.0.weight"] = Wb["adaln_w"]
            sd["adaLN_modulation.0.bias"] = Wb["adaln_b"]
        missing = set(n for n, _ in blk.named_parameters()) - set(sd)
        if missing:
            print("[blockcheck] %s UNMAPPED %s" % (key, sorted(missing)))
            return 1
        blk.load_state_dict({k: torch.from_numpy(np.ascontiguousarray(v))
                             for k, v in sd.items()}, strict=True)
        blk.eval()
        seq = SEQ if key.startswith("layers") else (
            64 if key.startswith("noise") else CAP)
        xx = x[:seq]
        with torch.no_grad():
            ref = blk(torch.from_numpy(xx)[None], None, freqs[:seq][None],
                      torch.from_numpy(adaln)[None])[0].numpy()
        mine = full_block_host(xx, Wb, modulation(adaln, Wb), cos[:seq],
                               sin[:seq])
        d = np.abs(mine - ref)
        amax = float(np.abs(ref).max())
        rel = float(d.max()) / amax
        print("[blockcheck] %-18s seq %3d  max|d| %.4e  rel_to_absmax %.4e  "
              "mean|d| %.4e  ref absmax %.4f  %s"
              % (key, seq, float(d.max()), rel, float(d.mean()), amax,
                 "PASS" if rel < 5e-4 else "FAIL"))
        ok_all = ok_all and rel < 5e-4
        del Wb
    return 0 if ok_all else 2


# ===========================================================================
# STAGE: host -- the real-weight reference image
# ===========================================================================
def stage_host(args) -> int:
    RW = Real()
    H = host_tensors(args.tag)
    timesteps, sigmas = schedule(args.steps)
    np.savez(HERE / ("%s_host_tensors.npz" % args.tag), **H)
    lat = np.random.default_rng(args.seed).standard_normal(
        (1, 16, 16, 16)).astype(np.float32)
    lat0 = lat.copy()
    print("latents %s absmax %.4f  seed %d  steps %d  layers %d"
          % (lat.shape, float(np.abs(lat).max()), args.seed, args.steps,
             args.layers), flush=True)
    print("timesteps %s" % np.array2string(timesteps, precision=3), flush=True)
    impl = (ffn_host_bf16in_impl if args.ffn_mode == "bf16-input"
            else ffn_host_impl)
    print("[host] ffn mode %s" % args.ffn_mode, flush=True)
    calls, steps_rec = [], []
    t_loop = time.perf_counter()
    for i, t in enumerate(timesteps):
        t0 = time.perf_counter()
        pred, rec = step_once(lat, t, RW, H, impl, layers=args.layers)
        d = F32(sigmas[i + 1] - sigmas[i])
        lat = (lat + d * (-pred)).astype(np.float32)
        steps_rec.append({"step": i, "t": float(t), "sigma": float(sigmas[i]),
                          "sigma_next": float(sigmas[i + 1]),
                          "pred_absmax": float(np.abs(pred).max()),
                          "lat_absmax": float(np.abs(lat).max()),
                          "seconds": time.perf_counter() - t0})
        calls.append(rec)
        print("  step %d: t=%8.3f  pred absmax %8.4f  lat absmax %9.4f  "
              "%.1f s" % (i, t, float(np.abs(pred).max()),
                          float(np.abs(lat).max()),
                          time.perf_counter() - t0), flush=True)
    wall = time.perf_counter() - t_loop
    print("loop wall %.1f s" % wall)
    np.save(HERE / ("%s_latents.npy" % args.tag), lat)
    np.savez(HERE / ("%s_traj.npz" % args.tag), latents0=lat0, latents=lat,
             timesteps=timesteps, sigmas=sigmas)
    (HERE / ("%s_calls.json" % args.tag)).write_text(json.dumps(
        {"seed": args.seed, "steps": args.steps, "layers": args.layers,
         "timesteps": [float(x) for x in timesteps],
         "sigmas": [float(x) for x in sigmas], "steps_rec": steps_rec,
         "wall": wall, "calls": calls,
         "weights": "zimage-port/weights.bfp16 (pack_bfp16.unpack)",
         "conditioning": "null caption = 32 x cap_pad_token; no text encoder",
         "precision": "numpy float32 (the model's own dtype)"}, indent=1),
        encoding="utf-8")
    if args.no_vae:
        print("[host] --no-vae: stopping before the decode")
        return 0
    img = vae_decode(lat)
    save_png(img, HERE / ("%s.png" % args.tag))
    np.save(HERE / ("%s_image.npy" % args.tag), img)
    print("image %s -> %s.png" % (img.shape, args.tag))
    return 0


# ===========================================================================
# STAGE: prep -- per-block stage-3 device payloads (BG/B3)
# ===========================================================================
def stage_prep(args) -> int:
    F = _ffn()
    RW = Real()
    out = HERE / PREPDIR
    out.mkdir(exist_ok=True)
    meta = {}
    t_all = time.perf_counter()
    for i in range(args.layers):
        key = "layers.%d" % i
        t0 = time.perf_counter()
        Wb = RW.block(key)
        BG = F.pack_bg_stream(Wb["w1"], Wb["w3"])
        B3 = F.pack_b3_stream(Wb["w2"])
        assert BG.size == BG_BYTES, (BG.size, BG_BYTES)
        assert B3.size == B3_BYTES, (B3.size, B3_BYTES)
        np.save(out / ("%s_bg.npy" % key), BG)
        np.save(out / ("%s_b3.npy" % key), B3)
        meta[key] = {"seconds": time.perf_counter() - t0,
                     "bg_sha16": sha16(BG.tobytes()),
                     "b3_sha16": sha16(B3.tobytes())}
        del Wb, BG, B3
        print("[prep] %-10s BG %s B  B3 %s B  %.1f s"
              % (key, format(BG_BYTES, ","), format(B3_BYTES, ","),
                 meta[key]["seconds"]), flush=True)
    print("[prep] %d blocks in %.1f s -> %s"
          % (args.layers, time.perf_counter() - t_all, out), flush=True)
    (HERE / ("%s_prep.json" % args.tag)).write_text(json.dumps(
        {"blocks": meta, "layers": args.layers}, indent=1), encoding="utf-8")
    return 0


# ===========================================================================
# STAGE: device -- the loop with the FFN of every layer on the NPU (py3.10)
# ===========================================================================
def bf16_bytes_to_f32(b):
    u16 = np.frombuffer(bytes(b), dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32)


def stage_device(args) -> int:
    for d in (XRT_DIR, pathlib.Path(r"C:\Windows\System32\AMD")):
        if d.exists():
            sys.path.insert(0, str(d))
            try:
                os.add_dll_directory(str(d))
            except Exception:  # noqa: BLE001
                pass
    import pyxrt

    DB = _oracle()
    RW = Real()
    H = host_tensors(args.tag)
    xclbin = STAGE3.with_suffix(".xclbin")
    inst = STAGE3.with_suffix(".bin").read_bytes()
    ninstr = len(inst) // 4
    prep = HERE / PREPDIR
    timesteps, sigmas = load_schedule(args)

    print("[dev] python %s  numpy %s" % (sys.version.split()[0],
                                         np.__version__), flush=True)
    print("[dev] image %s (%s B)  insts %s B / %d words"
          % (xclbin.name, format(xclbin.stat().st_size, ","),
             format(len(inst), ","), ninstr))

    dev = pyxrt.device(0)
    xc = pyxrt.xclbin(str(xclbin))
    uuid = dev.register_xclbin(xc)
    k = pyxrt.kernel(dev, uuid, KERNEL)
    print("[dev] ONE image registered, kernel %s opened" % KERNEL, flush=True)
    fl = pyxrt.bo.flags.host_only
    bo_instr = pyxrt.bo(dev, len(inst), pyxrt.bo.flags.cacheable, k.group_id(1))
    bo_instr.write(inst, 0)
    bo_instr.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo = {}
    bo["SBG"] = pyxrt.bo(dev, SBG_BYTES, fl, k.group_id(GROUPS["SBG"]))
    bo["BG"] = pyxrt.bo(dev, BG_BYTES, fl, k.group_id(GROUPS["BG"]))
    bo["B3"] = pyxrt.bo(dev, B3_BYTES, fl, k.group_id(GROUPS["B3"]))
    for n in ("A", "X", "Y"):
        bo[n] = pyxrt.bo(dev, ACT_BYTES, fl, k.group_id(GROUPS[n]))
    bo["Y"].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    print("[dev] BOs bound %s (group ids %s)"
          % (sorted(bo), {n: GROUPS[n] for n in GROUPS}), flush=True)

    RETRY_SLEEPS = (10, 20, 30, 45, 60)

    def _one(a_bytes, x_bytes, sbg_bytes, bg, b3, write_w):
        if write_w:
            bo["BG"].write(bg, 0)
            bo["B3"].write(b3, 0)
            for n in ("BG", "B3"):
                bo[n].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        bo["SBG"].write(sbg_bytes, 0)
        bo["A"].write(a_bytes, 0)
        bo["X"].write(x_bytes, 0)
        for n in ("SBG", "A", "X"):
            bo[n].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        bos = [bo[n] for n in PAYLOAD_ORDER]
        t0 = time.perf_counter()
        run = k(3, bo_instr, ninstr, *bos)
        state = run.wait(TIMEOUT_MS)
        try:
            sname = str(pyxrt.ert.ERT_CMD_STATE(state))
        except Exception:  # noqa: BLE001
            sname = str(state)
        bo["Y"].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        raw = bytes(bo["Y"].read(ACT_BYTES, 0))
        return int(state), sname, raw, time.perf_counter() - t0

    def dispatch(a_bytes, x_bytes, sbg_bytes, bg, b3, write_w, tag=""):
        """One dispatch, with the contention protocol built in.

        A TIMEOUT/ABORT is retried with backoff (up to 5 times) because the
        measured cause is EXTERNAL: a parallel agent sharing the device makes a
        326 ms dispatch take 5-20 s, and the ERT kills the command at ~7 s.
        Every attempt is recorded verbatim in the call record.
        """
        attempts = []
        for i in range(len(RETRY_SLEEPS) + 1):
            try:
                state, sname, raw, wall = _one(a_bytes, x_bytes, sbg_bytes,
                                              bg, b3, write_w)
            except Exception as exc:  # noqa: BLE001  (0xc00002b6 and friends)
                state, sname, raw, wall = -1, "XRT %s" % type(exc).__name__,                     None, 0.0
            attempts.append({"state": sname, "wall_ms": 1e3 * wall})
            if state == COMPLETED:
                return state, sname, raw, wall, attempts
            if i < len(RETRY_SLEEPS):
                print("[dev]   %s RETRY %d/%d after %.1f ms (%s)"
                      % (tag, i + 1, len(RETRY_SLEEPS), 1e3 * wall, sname),
                      flush=True)
                time.sleep(RETRY_SLEEPS[i])
        return state, sname, raw, wall, attempts

    zeros = np.zeros(ACT_BYTES, np.uint8).tobytes()
    calls, steps_rec = [], []
    # ---- resume: the driver drops the device without warning (0xc00002b6,
    # "device has been removed"), so the loop checkpoints after every step and
    # --resume continues from the last one instead of redoing 17 minutes.
    ck_lat = HERE / ("%s_ckpt_lat.npy" % args.tag)
    ck_js = HERE / ("%s_ckpt.json" % args.tag)
    start_step = 0
    lat = np.random.default_rng(args.seed).standard_normal(
        (1, 16, 16, 16)).astype(np.float32)
    lat0 = lat.copy()
    if args.resume and ck_lat.exists() and ck_js.exists():
        ck = json.loads(ck_js.read_text())
        if ck.get("steps") == args.steps and ck.get("layers") == args.layers:
            lat = np.load(ck_lat).astype(np.float32)
            lat0 = np.asarray(ck["lat0"], np.float32).reshape(lat.shape)
            calls, steps_rec = ck["calls"], ck["steps_rec"]
            start_step = int(ck["next_step"])
            print("[dev] RESUME at step %d (%d calls already measured)"
                  % (start_step, sum(len(c["layers"]) for c in calls)),
                  flush=True)
    print("[dev] latents absmax %.4f" % float(np.abs(lat).max()), flush=True)

    def ffn_impl(x_mid, Wb, mod, key):
        t0 = time.perf_counter()
        SBG = np.zeros((SEQ, 3 * D), np.float32)
        if mod is not None:
            SBG[:, :D] = (mod[2] - F32(1.0))
        SBG[:, D:2 * D] = 0.0
        SBG[:, 2 * D:] = Wb["ffn_norm1"]
        sbg_bytes = DB._refs.bf16_bytes(SBG).tobytes()
        a_bytes = DB._refs.bf16_bytes(np.asarray(x_mid, np.float32)).tobytes()
        bg = np.load(prep / ("%s_bg.npy" % key))
        b3 = np.load(prep / ("%s_b3.npy" % key))
        state, sname, raw, wall, attempts = dispatch(
            a_bytes, zeros, sbg_bytes, bg.tobytes(), b3.tobytes(), True,
            tag="%s/step?" % key)
        del bg, b3
        if raw is None:
            raise SystemExit(
                "[dev] REFUSAL: %s never completed after %d attempts (%s); "
                "the dispatches already measured are in the checkpoint"
                % (key, len(attempts), attempts[-1]["state"]))
        y = bf16_bytes_to_f32(raw).reshape(SEQ, D)
        out = ffn_post(x_mid, y, Wb, mod)
        r = {"ffn": "npu", "state": sname, "completed": bool(state == COMPLETED),
             "wall_ms": 1e3 * wall, "attempts": attempts,
             "call_s": time.perf_counter() - t0,
             "y_dev_absmax": float(np.abs(y).max()),
             "y_dev_sha16": sha16(raw)}
        if args.host_ref:
            t1 = time.perf_counter()
            y_ref = ffn_raw(ffn_input(x_mid, Wb, mod), Wb)
            e = np.abs(y - y_ref)
            r["ffn_npu_vs_host_f32"] = {
                "max_abs": float(e.max()),
                "rel_to_absmax": float(e.max() / float(np.abs(y_ref).max())),
                "rms": float(np.sqrt((e ** 2).mean())),
                "ref_absmax": float(np.abs(y_ref).max())}
            r["host_ref_s"] = time.perf_counter() - t1
            del y_ref
        if args.dump_calls:
            li = int(key.split(".")[-1]) if "." in key else -1
            if li in args.dump_calls:
                np.save(HERE / ("%s_dump_l%d_A.npy" % (args.tag, li)),
                        np.frombuffer(a_bytes, np.uint8))
                np.save(HERE / ("%s_dump_l%d_SBG.npy" % (args.tag, li)),
                        np.frombuffer(sbg_bytes, np.uint8))
                np.save(HERE / ("%s_dump_l%d_Y.npy" % (args.tag, li)),
                        np.frombuffer(raw, np.uint8))
        return out, r

    for i, t in enumerate(timesteps):
        if i < start_step:
            continue
        t0 = time.perf_counter()
        pred, rec = step_once(lat, t, RW, H, ffn_impl, layers=args.layers)
        d = F32(sigmas[i + 1] - sigmas[i])
        lat = (lat + d * (-pred)).astype(np.float32)
        steps_rec.append({"step": i, "t": float(t),
                          "pred_absmax": float(np.abs(pred).max()),
                          "lat_absmax": float(np.abs(lat).max()),
                          "seconds": time.perf_counter() - t0})
        calls.append(rec)
        np.save(ck_lat, lat)
        ck_js.write_text(json.dumps(
            {"next_step": i + 1, "steps": args.steps, "layers": args.layers,
             "lat0": np.asarray(lat0, np.float32).ravel().tolist(),
             "calls": calls, "steps_rec": steps_rec}), encoding="utf-8")
        print("  step %d: t=%8.3f  pred absmax %8.4f  lat absmax %9.4f  "
              "%.1f s" % (i, t, float(np.abs(pred).max()),
                          float(np.abs(lat).max()),
                          time.perf_counter() - t0), flush=True)

    states = [4 if r.get("completed") else 0 for c in calls
              for r in c["layers"]]
    walls = [r["wall_ms"] for c in calls for r in c["layers"]]
    ok = all(s == COMPLETED for s in states)
    print("\n[dev] ERT_CMD_STATE_COMPLETED on every dispatch: %s (%d/%d)"
          % (ok, sum(1 for s in states if s == COMPLETED), len(states)),
          flush=True)
    print("[dev] dispatch wall: min %.1f  mean %.1f  max %.1f ms"
          % (min(walls), float(np.mean(walls)), max(walls)), flush=True)
    np.save(HERE / ("%s_latents.npy" % args.tag), lat)
    np.savez(HERE / ("%s_traj.npz" % args.tag), latents0=lat0, latents=lat,
             timesteps=timesteps, sigmas=sigmas)
    (HERE / ("%s_calls.json" % args.tag)).write_text(json.dumps(
        {"seed": args.seed, "steps": args.steps, "layers": args.layers,
         "dispatches": len(states), "completed_every": bool(ok),
         "wall_ms_min": float(min(walls)),
         "wall_ms_mean": float(np.mean(walls)),
         "wall_ms_max": float(max(walls)),
         "image": xclbin.name, "image_sha256": hashlib.sha256(
             xclbin.read_bytes()).hexdigest(),
         "calls": calls, "steps_rec": steps_rec}, indent=1), encoding="utf-8")
    print("[dev] latents -> %s_latents.npy (VAE decode is a py3.12 stage: "
          "this interpreter has no diffusers)" % args.tag, flush=True)
    return 0 if ok else 2


# ===========================================================================
# STAGE: model -- the exact BFP16 device model vs a dumped dispatch
# ===========================================================================
def stage_model(args) -> int:
    """The bitexact_own gate on a REAL-weight payload: recompute the record's

    exact device model from the dumped A/SBG bytes and compare with Y.  Every
    primitive is the record's own (device_rms_norm_weighted, device_adaln_
    modulate, device_gemm at gemm_k_chunk=8, device_swiglu), and X was zeros so
    Y is the raw FFN output.
    """
    DB = _oracle()
    cfg = DB.DeviceCfg()
    RW = Real()
    rep = {}
    for li in (args.dump_calls or []):
        key = "layers.%d" % li
        a = np.load(HERE / ("%s_dump_l%d_A.npy" % (args.tag, li)))
        s = np.load(HERE / ("%s_dump_l%d_SBG.npy" % (args.tag, li)))
        y_raw = np.load(HERE / ("%s_dump_l%d_Y.npy" % (args.tag, li)))
        x = bf16_bytes_to_f32(a.tobytes()).reshape(SEQ, D)
        sbg = bf16_bytes_to_f32(s.tobytes()).reshape(SEQ, 3 * D)
        Wb = RW.block(key)
        n = DB.device_rms_norm_weighted(x, sbg[:, 2 * D:], cfg)
        m = DB.device_adaln_modulate(n, sbg[:, :D], sbg[:, D:2 * D])
        gate = DB.device_gemm(m, Wb["w1"], cfg)
        up = DB.device_gemm(m, Wb["w3"], cfg)
        sg = DB.device_swiglu(gate, up)
        ff = DB.device_gemm(sg, Wb["w2"], cfg)
        y = bf16_bytes_to_f32(y_raw.tobytes()).reshape(SEQ, D)
        e = np.abs(y - ff)
        rep[key] = {"max_abs": float(e.max()),
                    "rel_to_absmax": float(e.max() / np.abs(ff).max()),
                    "bit_exact_frac": float((y == ff).mean()),
                    "model_absmax": float(np.abs(ff).max()),
                    "device_absmax": float(np.abs(y).max())}
        print("[model] %-10s device vs its OWN exact model: max|d| %.4e  "
              "rel_to_absmax %.4e  bit-exact %.2f%%  (model absmax %.4g)"
              % (key, rep[key]["max_abs"], rep[key]["rel_to_absmax"],
                 100.0 * rep[key]["bit_exact_frac"], rep[key]["model_absmax"]))
        del Wb
    (HERE / ("%s_model.json" % args.tag)).write_text(json.dumps(rep, indent=1),
                                                     encoding="utf-8")
    return 0


# ===========================================================================
# STAGE: vae -- decode a saved latent to a PNG (py3.12)
# ===========================================================================
def stage_vae(args) -> int:
    lat = np.load(HERE / ("%s_latents.npy" % args.tag))
    img = vae_decode(lat)
    out = HERE / ("%s_npu.png" % args.tag)
    save_png(img, out)
    np.save(HERE / ("%s_npu_image.npy" % args.tag), img)
    print("[vae] %s latents %s absmax %.4f -> %s"
          % (args.tag, lat.shape, float(np.abs(lat).max()), out.name))
    return 0


# ===========================================================================
# STAGE: verify -- PSNR / SSIM against the host reference and the record
# ===========================================================================
def stage_verify(args) -> int:
    rep = {}
    host_png = HERE / ("%s_host.png" % args.tag)
    npu_png = HERE / ("%s_npu.png" % args.tag)
    for name, p in (("host", host_png), ("npu", npu_png)):
        if not p.exists():
            print("[verify] MISSING %s" % p)
            return 1
    a = load_png(host_png)
    b = load_png(npu_png)
    d = np.abs(a - b) * 255.0
    p, mse = psnr(a, b)
    rep["npu_vs_host"] = {"psnr_db": p, "mse": mse, "ssim": ssim(a, b),
                          "max_abs_255": float(d.max()),
                          "mean_abs_255": float(d.mean()),
                          "frac_diff": float((d > 0).mean()),
                          "bytes_identical": bool(host_png.read_bytes()
                                                  == npu_png.read_bytes())}
    print("[verify] NPU image vs HOST (real weights) reference")
    print("   PSNR %.4f dB   SSIM %.6f   max|d| %.0f/255   mean|d| %.3f/255"
          "   differing %.2f%%"
          % (p, rep["npu_vs_host"]["ssim"], d.max(), d.mean(),
             100.0 * rep["npu_vs_host"]["frac_diff"]))
    for name in ("host", "npu"):
        q = load_png(host_png if name == "host" else npu_png)
        o = load_png(ORACLE96)
        p2, _ = psnr(q, o)
        rep["%s_vs_oracle96" % name] = {"psnr_db": p2, "ssim": ssim(q, o)}
        print("   %s image vs imaggers/pipeline/oracle96.png: PSNR %.4f dB  "
              "SSIM %.6f" % (name, p2, rep["%s_vs_oracle96" % name]["ssim"]))
    calls = json.loads((HERE / ("%s_npu_calls.json" % args.tag)).read_text())
    ent = [c for step in calls["calls"] for c in step["layers"]
           if "ffn_npu_vs_host_f32" in c]
    if ent:
        rel = [c["ffn_npu_vs_host_f32"]["rel_to_absmax"] for c in ent]
        mx = [c["ffn_npu_vs_host_f32"]["max_abs"] for c in ent]
        rep["per_call"] = {
            "n": len(ent), "rel_min": float(min(rel)),
            "rel_median": float(np.median(rel)), "rel_max": float(max(rel)),
            "maxabs_max": float(max(mx))}
        print("[verify] device FFN vs host f32 FFN, %d calls: "
              "rel_to_absmax min %.4e median %.4e max %.4e; max|d| %.4e"
              % (len(ent), min(rel), float(np.median(rel)), max(rel), max(mx)))
    rep["dispatches"] = calls["dispatches"]
    rep["completed_every"] = calls["completed_every"]
    rep["wall_ms_mean"] = calls["wall_ms_mean"]
    (HERE / ("%s_verify.json" % args.tag)).write_text(json.dumps(rep, indent=1),
                                                     encoding="utf-8")
    print("[verify] -> %s_verify.json" % args.tag)
    return 0


# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="blockcheck")
    ap.add_argument("--tag", default="t3real")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--layers", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-vae", action="store_true")
    ap.add_argument("--host-ref", action="store_true",
                    help="device stage: also compute the host f32 FFN per call")
    ap.add_argument("--dump-calls", type=int, nargs="*", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ffn-mode", default="f32",
                    choices=("f32", "bf16-input"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    fn = {"blockcheck": stage_blockcheck, "host": stage_host,
          "prep": stage_prep, "device": stage_device,
          "vae": stage_vae, "model": stage_model,
          "verify": stage_verify}[args.stage]
    return fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
