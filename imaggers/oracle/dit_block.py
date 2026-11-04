# SPDX-License-Identifier: MIT
"""dit_block.py -- ORACLE for one Z-Image DiT transformer block.

Pure numpy.  No NPU, no compile, no torch.  Two variants of the same block:

* ``dit_block_float``  -- the FLOAT ORACLE.  float64 throughout, no rounding
  anywhere.  This is the truth the NPU assembly is checked against.
* ``dit_block_device`` -- the DEVICE MODEL.  The precision/dtype behaviour the
  AIE assembly will actually produce.  ``dit_block_float`` minus
  ``dit_block_device`` is the gap the NPU work is allowed to lose.

The block (the 9 steps of ``zimage-bfp16-npu/ASSEMBLY.md:11-24``)::

    x = x + attn( adaln_modulate( rms_norm_weighted(x, gamma1), s1, b1 ) )
    x = x + ff(   adaln_modulate( rms_norm_weighted(x, gamma2), s2, b2 ) )

    rms_norm_weighted(x, g) = x / rms(x) * g            (rms over hidden dim)
    adaln_modulate(x, s, b) = x * (1 + s) + b           (per token s/b)
    attn = RoPE(Q,K) -> softmax(QK^T / sqrt(head_dim), NON-CAUSAL) @ V -> proj
    ff   = SwiGLU: silu(x @ W1) * (x @ W3) @ W2

Geometry is a parameter (``Geometry``); Z-Image-Turbo is
hidden=3840, heads=30, head_dim=128, ffn_inner=10240, seq=96.

WHAT THE DEVICE MODEL MODELS, AND ITS EVIDENCE
----------------------------------------------
The load-bearing fact (KNOWN-ISSUES.md "VERIFIED ON THE NPU" /
``imaggers/dit2/FINDINGS.md:80-95``): ``mm_bfp_mixed.cc`` loads C, MACs, and
stores C **re-rounded to bf16 after EVERY GEMM call**
(``imaggers/dit2/out/.../mm_bfp_mixed.cc:49-52,86-89``: `accC = load_v(C)`,
MAC loop, `store_v(pC, accC.to_vector<bfloat16>())`).  A numpy reference that
accumulates in f64 drifts up to 13 ulps at K=1792 and looks like a kernel bug
when it is not.  The device model therefore round-trips C through bf16 after
**every accumulation step** (``DeviceCfg.gemm_k_chunk``, default 8 = the
micro-tile s dim, the finest granularity dit2 measured bit-exactly; a larger
call K means fewer roundings, so 8 is the conservative choice).

Per stage:

* GEMMs (QKV, QK^T, P@V, out-proj, FFN x3): operands quantised to
  ``bfp16ebs8`` in groups of 8 along the reduction axis (the (r,s,t)=(8,8,8)
  micro-tile, ``mm_bfp_mixed.cc:15-22``), A converted on the core with
  conv_even (RNE) rounding (``mm_bfp_mixed.cc:130-135``), C accumulated in
  f32 (``accfloat``, ``mm_bfp_mixed.cc:49-52``) and stored bf16 per call.
  Host-side B codec rounding is NOT determinable -> conservative ``floor``
  (``aie.utils.bfp``'s default, ``bfp.py:76-88``); see FINDINGS.md.
* rms_norm_weighted: the fused kernel does not exist yet
  (``imaggers/adaln/refs.py`` WEIGHTED_RMSNORM_CONTRACT "MISSING in
  mlir-aie"), so the model is the WORST plausible build: the two-pass
  composition of existing kernels (norm store bf16, then gamma mul store
  bf16; ``refs.py:164-177``).  Sum-of-squares accumulated SEQUENTIALLY in
  f32 (``rms_norm_aie2p.h:123-138``), inv_rms in f32.
* adaln_modulate: also missing as a fused kernel -> conservative two-pass
  (``refs.py:209-232``): g=bf16(1+s), tmp=bf16(x*g), y=bf16(tmp+shift) --
  three bf16 stores instead of the proposed fused one.
* RoPE: ``rope.cc:29-39`` keeps each product in the accumulator with ONE
  rounding at the store; cos/sin arrive as a bf16 LUT.  Modelled exactly so.
  CONVENTION: Z-Image is the INTERLEAVED one (adjacent pairs), see
  ``zimage-bfp16-repo/rope_real.py:39-61`` (rotate-half proved WRONG there,
  error 6.288 vs 4.8e-7).  ASSEMBLY.md:18 names ``two_halves`` (= HF
  rotate-half, ``datamovement.py:322-323``) -- that is a discrepancy the
  assembly must resolve; both conventions are available here
  (``DeviceCfg.rope_convention``).
* attention softmax: ``softmax_rows_bf16`` computes exp2((x-m)*log2e) with
  log2e = 1.4453125 (``softmax_aie2p.h:15-16,57,88-98``), f32, one bf16
  store.  Modelled exactly so; the AIE exp2 polynomial's own error is NOT
  modelable (measured 2.94e-2 relative on npu2, ``activation.py:400-407``)
  -- see FINDINGS.md for the envelope.
* the 1/sqrt(head_dim) attention scale: the softmax kernel has no scale
  parameter (``attn/DESIGN.md:44-45,205``) and whether it is folded into Q
  or applied as a pre-scale pass over S is NOT determinable -> conservative
  extra bf16 store on S (``attn/DESIGN.md:71-73``).
* SwiGLU: ``activation/swiglu.cc`` narrows after EVERY multiply
  (``activation.py:907-941``: x*w1, x*w2, sigmoid factor, silu product,
  output each in bf16, with an x*w2 clamp at -8 and subnormal flush-to-zero).
  Modelled exactly so, except the sigmoid itself: which build the assembly
  uses (vtanh vs table) is NOT determinable -> exact sigmoid in the model,
  measured envelope in FINDINGS.md (``activation.py:144-148``).
* residual adds: ``add_sized`` (eltwise) -- f32 accumulate, one bf16 store.

WEIGHTS AND INPUTS
------------------
``make_weights`` produces torch-Linear-layout matrices ``(out, in)`` scaled by
1/sqrt(fan_in), stored float32 (ASSEMBLY.md:40 says the real comparison is
"the checkpoint's fp32 weights"); the float oracle upconverts to f64, the
device model quantises to bfp16ebs8 per reduction group.  ``make_inputs``
produces the residual stream x and the four per-token modulation tensors
s1,b1,s2,b2.  Producing s/b from the conditioning vector is a GEMM outside
this block's scope (ASSEMBLY.md:26-28).

Reuse: ``round_bf16``, ``rms_norm_weighted_ref``, ``adaln_modulate_ref`` and
``noncausal_softmax`` come from the existing verified references
``imaggers/adaln/refs.py`` and ``imaggers/attn/noncausal_softmax.py``.

Run the self-test with ``test_dit_block.py`` in this directory.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# never write bytecode caches into the read-only reference directories
sys.dont_write_bytecode = True

# ---------------------------------------------------------------------------
# Reuse the verified references already in the tree (read-only imports).
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_SCRATCH = _HERE.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_refs = _load("oracle_adaln_refs", _SCRATCH / "adaln" / "refs.py")
_softmax = _load("oracle_softmax_refs", _SCRATCH / "attn" / "noncausal_softmax.py")

round_bf16 = _refs.round_bf16  # float -> bf16 values (f32 arrays), RNE
rms_norm_weighted_ref = _refs.rms_norm_weighted_ref
adaln_modulate_ref = _refs.adaln_modulate_ref
noncausal_softmax = _softmax.noncausal_softmax

__all__ = [
    "Geometry",
    "DeviceCfg",
    "make_weights",
    "make_inputs",
    "rope_table",
    "dit_block_float",
    "dit_block_device",
    "run_both",
    "divergence",
    "round_bf16",
    "bfp16ebs8_quantize",
    "device_gemm",
    "device_rms_norm_weighted",
    "device_adaln_modulate",
    "device_rope",
    "device_softmax_rows",
    "device_swiglu",
]


# ---------------------------------------------------------------------------
# Geometry / configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Geometry:
    """Z-Image DiT block geometry.  Defaults are Z-Image-Turbo (ASSEMBLY.md:7-9)."""

    hidden: int = 3840
    heads: int = 30
    head_dim: int = 128
    ffn_inner: int = 10240
    seq: int = 96

    def validate(self) -> None:
        g = self
        if g.heads * g.head_dim != g.hidden:
            raise ValueError(
                f"Geometry: heads*head_dim must equal hidden, got "
                f"{g.heads}*{g.head_dim} != {g.hidden}"
            )
        for name in ("hidden", "head_dim", "ffn_inner", "seq"):
            v = getattr(g, name)
            if v <= 0 or v % 8:
                # bfp16ebs8 groups are 8 along every GEMM reduction axis:
                # hidden (QKV/FFN in), head_dim (QK^T), seq (P@V), ffn_inner
                # (FFN down) -- all must be multiples of 8.
                raise ValueError(f"Geometry: {name} must be a positive multiple of 8")
        if g.heads <= 0:
            raise ValueError("Geometry: heads must be positive")

    @property
    def attn_scale(self) -> float:
        return 1.0 / math.sqrt(self.head_dim)


@dataclass
class DeviceCfg:
    """Knobs of the device model.  Defaults are the CONSERVATIVE choices.

    "Conservative" = the choice that makes the device model look WORSE, so the
    NPU work is never judged by an optimistic oracle (task instruction).
    Every non-obvious default is justified in the module docstring and
    FINDINGS.md.
    """

    # bf16 re-rounding of C after every GEMM accumulation step.  8 = the
    # micro-tile s dim; dit2 measured its per-call model bit-exact against the
    # device at K=32..1792 with exactly this granularity (dit2/FINDINGS.md:84-95).
    gemm_k_chunk: int = 8
    # On-core A conversion rounding (mm_bfp_mixed.cc:130-135 -> conv_even).
    a_bfp_rounding: str = "conv_even"
    # Host-side B (weights / K / V) codec rounding.  NOT determinable; the
    # conservative choice is 'floor' -- aie.utils.bfp's DEFAULT (bfp.py:76-88),
    # truncating toward -inf and therefore biased.  Alternatives: 'conv_even',
    # 'half_away' (pack_bfp16's vendor-verified rule).
    b_bfp_rounding: str = "floor"
    # RoPE convention: Z-Image is 'interleaved' (rope_real.py:39-61);
    # 'two_halves' is what ASSEMBLY.md:18 currently names -- kept selectable.
    rope_convention: str = "interleaved"
    # rms norm epsilon.  Z-Image's own value is NOT determinable from the
    # sources read; refs.py uses rms_norm's default 1e-5.
    eps: float = 1e-5
    # RoPE frequency base (rope_real.py: precompute_freqs_cis_real theta=256).
    rope_theta: float = 256.0
    # Z-Image rope axes (rope_real.py:58: axes_dims=[32,48,48] at head_dim 128).
    # None -> (32, 48, 48) when head_dim == 128, else a single [head_dim] axis.
    rope_axes: tuple | None = None

    def validate(self, geom: Geometry) -> None:
        if self.gemm_k_chunk % 8 or self.gemm_k_chunk <= 0:
            raise ValueError("DeviceCfg.gemm_k_chunk must be a positive multiple of 8")
        if self.a_bfp_rounding not in ("conv_even", "floor", "half_away"):
            raise ValueError(f"bad a_bfp_rounding {self.a_bfp_rounding!r}")
        if self.b_bfp_rounding not in ("conv_even", "floor", "half_away"):
            raise ValueError(f"bad b_bfp_rounding {self.b_bfp_rounding!r}")
        if self.rope_convention not in ("interleaved", "two_halves"):
            raise ValueError(f"bad rope_convention {self.rope_convention!r}")
        if self.rope_axes is not None and sum(self.rope_axes) != geom.head_dim:
            raise ValueError("rope_axes must sum to head_dim")


def _rope_axes(geom: Geometry, cfg: DeviceCfg) -> list:
    if cfg.rope_axes is not None:
        return list(cfg.rope_axes)
    return [32, 48, 48] if geom.head_dim == 128 else [geom.head_dim]


# ---------------------------------------------------------------------------
# Inputs / weights (float32 checkpoint values, shared by both variants)
# ---------------------------------------------------------------------------
def make_weights(geom: Geometry, rng: np.random.Generator) -> dict:
    """fp32 checkpoint-like weights, torch-Linear layout (out, in).

    GEMM convention everywhere below: ``y = x @ W.T``.  Scaled by
    1/sqrt(fan_in) so one block keeps activations O(1).
    """
    h, ff = geom.hidden, geom.ffn_inner

    def w(out_fan: int, in_fan: int) -> np.ndarray:
        std = 1.0 / math.sqrt(in_fan)
        return (rng.standard_normal((out_fan, in_fan)) * std).astype(np.float32)

    return {
        "gamma1": (1.0 + 0.1 * rng.standard_normal(h)).astype(np.float32),
        "gamma2": (1.0 + 0.1 * rng.standard_normal(h)).astype(np.float32),
        "w_qkv": w(3 * h, h),
        "w_o": w(h, h),
        "w_gate": w(ff, h),  # W1 of silu(x @ W1) * (x @ W3) @ W2
        "w_up": w(ff, h),  # W3
        "w_down": w(h, ff),  # W2
    }


def make_inputs(geom: Geometry, rng: np.random.Generator) -> dict:
    """Residual stream x and the four per-token modulation tensors."""
    s, h = geom.seq, geom.hidden
    return {
        "x": rng.standard_normal((s, h)),
        "s1": 0.1 * rng.standard_normal((s, h)),
        "b1": 0.1 * rng.standard_normal((s, h)),
        "s2": 0.1 * rng.standard_normal((s, h)),
        "b2": 0.1 * rng.standard_normal((s, h)),
    }


# ---------------------------------------------------------------------------
# RoPE table (Z-Image convention: interleaved complex stream)
# ---------------------------------------------------------------------------
def rope_table(geom: Geometry, cfg: DeviceCfg, ids: np.ndarray | None = None):
    """cos, sin tables of shape (seq, head_dim//2), float64.

    Faithful to ``RopeEmbedder.precompute_freqs_cis`` in real form
    (``zimage-bfp16-repo/rope_real.py:264-289``): per axis
    ``freqs = 1/theta**(arange(0, d, 2)/d)``, ``angles = outer(t, freqs)``,
    axes concatenated along the last dim.

    Position ids are NOT determinable from the sources read (how the 96 tokens
    map to the three rope axes); the default is ``ids[t] = (t, t, t)``.  Both
    variants share this table, so the float-vs-device gap is unaffected.
    """
    geom.validate()
    cfg.validate(geom)
    axes = _rope_axes(geom, cfg)
    if sum(axes) != geom.head_dim:
        raise ValueError("rope axes must sum to head_dim")
    if ids is None:
        ids = np.tile(np.arange(geom.seq)[:, None], (1, len(axes)))
    ids = np.asarray(ids)
    if ids.shape != (geom.seq, len(axes)):
        raise ValueError(f"ids must be ({geom.seq}, {len(axes)})")
    cos_p, sin_p = [], []
    for a, d in enumerate(axes):
        freqs = 1.0 / (cfg.rope_theta ** (np.arange(0, d, 2, dtype=np.float64) / d))
        angles = np.outer(ids[:, a].astype(np.float64), freqs)
        cos_p.append(np.cos(angles))
        sin_p.append(np.sin(angles))
    return np.concatenate(cos_p, axis=-1), np.concatenate(sin_p, axis=-1)


# ---------------------------------------------------------------------------
# bfp16ebs8 quantisation model (aie.utils.bfp semantics, bfp.py:76-185)
# ---------------------------------------------------------------------------
_BFP_BLOCK = 8


def bfp16ebs8_quantize(x, rounding: str = "floor") -> np.ndarray:
    """What a bfp16ebs8 matmul kernel reads of ``x`` (``bfp.quantize``).

    Groups of 8 consecutive values along the LAST axis share one exponent
    (npdtypes block_size=8, bfp.py:6-27).  The shared grid is
    ``2**(max_exp_field - 127 - 6)`` (bfp.py:141,179 -- identical to
    pack_bfp16._quantise's ``e = e_unb - 6``).  Rounding modes:

    * ``floor``     -- the header's truncation toward -inf (bfp.py:15-16,82):
                      mantissas truncate toward negative infinity, and a value
                      more than 31 binades below its block maximum becomes 0
                      (positive) or -1 LSB (negative).
    * ``conv_even`` -- round to nearest, ties to even, with the one-sided
                      carry (+128 raises the block exponent and requantises,
                      -128 fits; bfp.py:153-165).  This is the on-core A
                      conversion (mm_bfp_mixed.cc:130-135).
    * ``half_away`` -- pack_bfp16._quantise's vendor-verified rule
                      (sign(q)*floor(|q|+0.5), same one-sided carry).

    Returns float32 dequantised values (exact on the block's grid).
    """
    xf = np.ascontiguousarray(x, dtype=np.float32)
    n = xf.shape[-1]
    if n % _BFP_BLOCK:
        raise ValueError(f"bfp16ebs8_quantize: last axis {n} not a multiple of 8")
    blocks = xf.reshape(*xf.shape[:-1], n // _BFP_BLOCK, _BFP_BLOCK)
    bits = blocks.view(np.uint32)
    sign = (bits >> 31).astype(bool)
    exp = (bits >> 23) & 0xFF
    max_exp = exp.max(axis=-1, keepdims=True).astype(np.int32)
    scale = np.ldexp(1.0, max_exp - 127 - 6)  # f64, exact power of two

    def requant(s):
        return np.rint(blocks.astype(np.float64) / s)

    if rounding == "conv_even":
        m = requant(scale)
        carry = m.max(axis=-1, keepdims=True) > 127
        if carry.any():
            scale = np.where(carry, scale * 2.0, scale)
            m = requant(scale)
        m = np.clip(m, -128, 127)
    elif rounding == "half_away":
        q = blocks.astype(np.float64) / scale
        m = np.sign(q) * np.floor(np.abs(q) + 0.5)
        carry = m.max(axis=-1, keepdims=True) > 127
        if carry.any():
            scale = np.where(carry, scale * 2.0, scale)
            q = blocks.astype(np.float64) / scale
            m = np.sign(q) * np.floor(np.abs(q) + 0.5)
        m = np.clip(m, -128, 127)
    elif rounding == "floor":
        q = blocks.astype(np.float64) / scale  # exact in f64 (24-bit mantissa)
        m = np.floor(q)  # toward -inf == the header's truncation
        far = (max_exp - exp.astype(np.int32)) >= 32
        m = np.where(far, np.where(sign, -1.0, 0.0), m)
        m = np.clip(m, -128, 127)
    else:
        raise ValueError(f"bfp16ebs8_quantize: unknown rounding {rounding!r}")
    return (m * scale).astype(np.float32).reshape(xf.shape)


def round_bf16_ftz(v) -> np.ndarray:
    """bf16 store with subnormal flush-to-zero (activation.py:835-842's model
    of an accumulator store): a subnormal product is flushed to zero first,
    keeping its sign."""
    v32 = np.asarray(v, dtype=np.float32)
    v32 = np.where(
        np.abs(v32) < np.float32(2.0**-126), np.copysign(np.float32(0.0), v32), v32
    )
    return round_bf16(v32)


# ---------------------------------------------------------------------------
# DEVICE MODEL primitives
# ---------------------------------------------------------------------------
_F32 = np.float32
# softmax_aie2p.h:15-16,57 -- exp2((x - m) * log2e), log2e = 1.4453125 exactly.
_LOG2E = _F32(1.4453125)


def device_gemm(A, W, cfg: DeviceCfg):
    """Device model of one ``mm_bfp(mixed=True)`` GEMM:  C = A @ W.T.

    A is (m, K) bf16-valued; W is (n, K) fp32 checkpoint weights (B operand).
    Faithful to ``mm_bfp_mixed.cc`` and dit2's measured reference model
    (dit2/FINDINGS.md:80-95):

      * operands quantised to bfp16ebs8 in groups of 8 along K -- A converted
        on the core (conv_even, mm_bfp_mixed.cc:130-135), B decoded from its
        packed blocks;
      * per GEMM call (``cfg.gemm_k_chunk`` reduction elements):
        ``C = bf16(C + A_chunk_q @ B_chunk_q.T)`` -- the accumulator is f32
        (accfloat) but is RE-STORED to bf16 at the end of every call
        (mm_bfp_mixed.cc:86-89), so every call contributes one bf16 rounding.
    """
    # (cfg is validated by the block entry points / Geometry.validate)
    A32 = np.ascontiguousarray(A, dtype=np.float32)
    W32 = np.ascontiguousarray(W, dtype=np.float32)
    m, K = A32.shape
    n, KW = W32.shape
    if KW != K:
        raise ValueError(f"device_gemm: A is (m,{K}) but W is (n,{KW})")
    kc = min(cfg.gemm_k_chunk, K)  # one call may span the whole K
    if K % 8 or kc % 8:
        raise ValueError(f"device_gemm: K={K} and k_chunk={kc} must be multiples of 8")
    C = np.zeros((m, n), dtype=np.float32)  # zero() kernel initialises C
    for kk in range(0, K, kc):
        aq = bfp16ebs8_quantize(A32[:, kk : kk + kc], cfg.a_bfp_rounding)
        wq = bfp16ebs8_quantize(W32[:, kk : kk + kc], cfg.b_bfp_rounding)
        # products of two <=8-bit mantissas are exact in f32; accumulate f32
        C = round_bf16(C + (aq @ wq.T))
    return C


def device_rms_norm_weighted(x, gamma, cfg: DeviceCfg):
    """Device model of weighted RMSNorm -- conservative TWO-PASS build.

    The fused ``rms_norm_weighted`` kernel does not exist yet
    (refs.py: WEIGHTED_RMSNORM_CONTRACT status "MISSING in mlir-aie"), so the
    model is the worst plausible composition (refs.py:164-177): unweighted
    rms_norm storing bf16, then a gamma multiply storing bf16.  Sum of squares
    accumulates sequentially in f32 (rms_norm_aie2p.h:123-138); inv_rms is an
    f32 scalar multiply (rms_norm_aie2p.h:140-142).
    """
    xf = np.ascontiguousarray(x, dtype=np.float32)
    g = round_bf16(gamma).astype(np.float32)
    sq = xf * xf
    ms = np.cumsum(sq, axis=-1, dtype=np.float32)[..., -1] / _F32(xf.shape[-1])
    rms = np.sqrt(ms + _F32(cfg.eps))
    inv = _F32(1.0) / rms
    tmp = round_bf16(xf * inv[..., None])  # pass 1 store: unweighted norm
    y = round_bf16(tmp * g[None, :])  # pass 2 store: gamma multiply
    return y


def device_adaln_modulate(x, scale, shift):
    """Device model of adaLN -- conservative TWO-PASS build (refs.py:209-232).

    The fused kernel is missing, so the model is the best composition the
    existing kernels give, which is also the worst plausible one:
    g = bf16(1+scale); tmp = bf16(x*g); y = bf16(tmp+shift) -- three bf16
    stores (the fused proposal would keep (1+s) unfused in f32 and store once).
    """
    xf = np.ascontiguousarray(x, dtype=np.float32)
    sf = np.ascontiguousarray(scale, dtype=np.float32)
    bf = np.ascontiguousarray(shift, dtype=np.float32)
    g = round_bf16(_F32(1.0) + sf)  # pass 0
    tmp = round_bf16(xf * g)  # pass 1
    y = round_bf16(tmp + bf)  # pass 2
    return y


def device_rope(x, cos, sin, convention: str = "interleaved"):
    """Device model of the rope kernel (rope.cc:29-39): one rounding at the
    store per output half, cos/sin read as a bf16 LUT.

    x is (seq, heads, head_dim) bf16-valued; cos/sin are (seq, head_dim//2).
    ``interleaved`` pairs (2k, 2k+1) -- Z-Image's convention
    (rope_real.py:39-61); ``two_halves`` pairs (k, k + d/2) -- HF rotate-half.
    """
    x32 = np.ascontiguousarray(x, dtype=np.float32)
    c = round_bf16(cos).astype(np.float32)[:, None, :]
    s = round_bf16(sin).astype(np.float32)[:, None, :]
    d = x32.shape[-1]
    if d % 2:
        raise ValueError("device_rope: head_dim must be even")
    if convention == "interleaved":
        xe, xo = x32[..., 0::2], x32[..., 1::2]
    elif convention == "two_halves":
        xe, xo = x32[..., : d // 2], x32[..., d // 2 :]
    else:
        raise ValueError(f"device_rope: unknown convention {convention!r}")
    # each half stays in the accumulator until one rounding at the end
    # (rope.cc:29-36)
    out_e = round_bf16(xe * c - xo * s)
    out_o = round_bf16(xe * s + xo * c)
    if convention == "interleaved":
        out = np.empty_like(x32)
        out[..., 0::2] = out_e
        out[..., 1::2] = out_o
    else:
        out = np.concatenate([out_e, out_o], axis=-1)
    return out


def device_softmax_rows(S):
    """Device model of ``softmax_rows_bf16`` (softmax_aie2p.h:222-229).

    exp2((x - m) * log2e) with the hard-coded log2e = 1.4453125
    (softmax_aie2p.h:15-16,57,88-98), f32, sequential row sum, one bf16
    store.  The AIE exp2 instruction's own error (measured 2.94e-2 relative,
    activation.py:400-407) is NOT modelled -- see FINDINGS.md.
    """
    S32 = np.ascontiguousarray(S, dtype=np.float32)
    m = S32.max(axis=-1, keepdims=True)
    z = (S32 - m) * _LOG2E
    e = np.exp2(z.astype(np.float32)).astype(np.float32)
    tot = np.cumsum(e, axis=-1, dtype=np.float32)[..., -1:]
    p = e / tot
    return round_bf16(p)


def device_swiglu(gate, up):
    """Device model of ``swiglu_bf16`` for already-projected tiles.

    Task formula: ``silu(x @ W1) * (x @ W3)``, gate = x@W1, up = x@W3 (both
    already bf16 from their GEMM stores).  Structure from
    activation.py:934-941 (_swiglu_model): the sigmoid factor, the silu
    product and the output each land in a bf16 register before the next step;
    x*w2 is clamped at -8 before the silu multiply; subnormal products are
    flushed to zero; where the silu product is 0 the output is 0.

    The sigmoid itself is computed exactly here: WHICH sigmoid the assembly
    gets (AIE2P vtanh vs the table build) is NOT determinable, and vtanh has
    no published spec.  FINDINGS.md carries the measured envelope
    (activation.py:144-148: swiglu 0.08 rel / 0.35 abs for the vtanh build).
    """
    g = np.ascontiguousarray(gate, dtype=np.float32)
    u = np.ascontiguousarray(up, dtype=np.float32)
    with np.errstate(over="ignore"):
        sig = 1.0 / (1.0 + np.exp(-g.astype(np.float64)))
        sig = round_bf16(sig.astype(np.float32))  # narrowed to bf16 (silu.cc)
        silu_out = round_bf16_ftz(np.maximum(g, _F32(-8.0)) * sig)
        out = np.where(silu_out == 0, _F32(0.0), round_bf16_ftz(u * silu_out))
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# FLOAT ORACLE helpers (f64, no rounding)
# ---------------------------------------------------------------------------
def float_gemm(A64, W32, k_block: int = 1024) -> np.ndarray:
    """y = A @ W.T in float64, chunked over K only to bound transient memory.

    Chunked f64 accumulation differs from one big dgemm only at ~1e-15
    relative -- far below anything the device model does.
    """
    A64 = np.ascontiguousarray(A64, dtype=np.float64)
    W32 = np.ascontiguousarray(W32, dtype=np.float32)
    m, K = A64.shape
    n, KW = W32.shape
    if KW != K:
        raise ValueError(f"float_gemm: A is (m,{K}) but W is (n,{KW})")
    C = np.zeros((m, n), dtype=np.float64)
    for kk in range(0, K, k_block):
        blk = min(k_block, K - kk)
        C += A64[:, kk : kk + blk] @ W32[:, kk : kk + blk].astype(np.float64).T
    return C


def float_rope(x, cos, sin, convention: str = "interleaved") -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    c = np.asarray(cos, dtype=np.float64)[:, None, :]
    s = np.asarray(sin, dtype=np.float64)[:, None, :]
    d = x64.shape[-1]
    if convention == "interleaved":
        xe, xo = x64[..., 0::2], x64[..., 1::2]
    elif convention == "two_halves":
        xe, xo = x64[..., : d // 2], x64[..., d // 2 :]
    else:
        raise ValueError(f"float_rope: unknown convention {convention!r}")
    out_e = xe * c - xo * s
    out_o = xe * s + xo * c
    if convention == "interleaved":
        out = np.empty_like(x64)
        out[..., 0::2] = out_e
        out[..., 1::2] = out_o
        return out
    return np.concatenate([out_e, out_o], axis=-1)


def float_silu(x64) -> np.ndarray:
    xf = np.maximum(x64, -np.finfo(np.float64).max)  # refs.py _no_neg_inf idiom
    with np.errstate(over="ignore"):
        return xf / (1.0 + np.exp(-xf))


# ---------------------------------------------------------------------------
# The two block variants
# ---------------------------------------------------------------------------
def _split_heads(t, geom: Geometry) -> np.ndarray:
    return t.reshape(geom.seq, geom.heads, geom.head_dim)


def _merge_heads(t, geom: Geometry) -> np.ndarray:
    return t.reshape(geom.seq, geom.hidden)


def dit_block_float(inputs: dict, weights: dict, geom: Geometry, cfg: DeviceCfg,
                    *, collect: dict | None = None, rope_cos_sin=None):
    """FLOAT ORACLE: one complete Z-Image DiT transformer block in float64."""
    geom.validate()
    cfg.validate(geom)
    x = np.asarray(inputs["x"], dtype=np.float64)
    s1 = np.asarray(inputs["s1"], dtype=np.float64)
    b1 = np.asarray(inputs["b1"], dtype=np.float64)
    s2 = np.asarray(inputs["s2"], dtype=np.float64)
    b2 = np.asarray(inputs["b2"], dtype=np.float64)
    cos, sin = rope_cos_sin if rope_cos_sin is not None else rope_table(geom, cfg)

    def stash(key, arr):
        if collect is not None:
            collect[key] = np.array(arr, dtype=np.float64, copy=True)

    stash("x_in", x)
    # ---- attention branch -------------------------------------------------
    n1 = rms_norm_weighted_ref(
        x, weights["gamma1"], eps=cfg.eps, out_dtype="f64"
    )
    m1 = adaln_modulate_ref(n1, s1, b1, out_dtype="f64")
    stash("adaln1", m1)
    qkv = float_gemm(m1, weights["w_qkv"])
    q, k, v = np.split(qkv, 3, axis=-1)
    q = float_rope(_split_heads(q, geom), cos, sin, cfg.rope_convention)
    k = float_rope(_split_heads(k, geom), cos, sin, cfg.rope_convention)
    v = _split_heads(v, geom)
    stash("q_rope", _merge_heads(q, geom))
    stash("k_rope", _merge_heads(k, geom))
    scale = geom.attn_scale
    o = np.empty_like(q)
    for hd in range(geom.heads):
        S = q[:, hd, :] @ k[:, hd, :].T  # (seq, seq)
        P = noncausal_softmax(S, scale=scale, out_dtype="f64")
        o[:, hd, :] = P @ v[:, hd, :]
    attn_o = _merge_heads(o, geom)
    stash("attn_o", attn_o)
    proj = float_gemm(attn_o, weights["w_o"])
    x = x + proj
    stash("x_mid", x)
    # ---- FFN branch -------------------------------------------------------
    n2 = rms_norm_weighted_ref(
        x, weights["gamma2"], eps=cfg.eps, out_dtype="f64"
    )
    m2 = adaln_modulate_ref(n2, s2, b2, out_dtype="f64")
    stash("adaln2", m2)
    gate = float_gemm(m2, weights["w_gate"])
    up = float_gemm(m2, weights["w_up"])
    sg = float_silu(gate) * up
    stash("swiglu", sg)
    ff = float_gemm(sg, weights["w_down"])
    x = x + ff
    stash("x_out", x)
    return x


def dit_block_device(inputs: dict, weights: dict, geom: Geometry, cfg: DeviceCfg,
                     *, collect: dict | None = None, rope_cos_sin=None):
    """DEVICE MODEL: one complete block at the precision the AIE assembly
    will produce.  Returns f32 arrays holding bf16 values."""
    geom.validate()
    cfg.validate(geom)
    # tensors live in bf16 on the device (the residual stream, the modulation
    # tiles, the GEMM accumulators) -- one rounding at entry.
    x = round_bf16(inputs["x"]).astype(np.float32)
    s1 = round_bf16(inputs["s1"]).astype(np.float32)
    b1 = round_bf16(inputs["b1"]).astype(np.float32)
    s2 = round_bf16(inputs["s2"]).astype(np.float32)
    b2 = round_bf16(inputs["b2"]).astype(np.float32)
    cos, sin = rope_cos_sin if rope_cos_sin is not None else rope_table(geom, cfg)

    def stash(key, arr):
        if collect is not None:
            collect[key] = np.array(arr, dtype=np.float64, copy=True)

    stash("x_in", x)
    # ---- attention branch -------------------------------------------------
    n1 = device_rms_norm_weighted(x, weights["gamma1"], cfg)
    m1 = device_adaln_modulate(n1, s1, b1)
    stash("adaln1", m1)
    qkv = device_gemm(m1, weights["w_qkv"], cfg)
    q, k, v = np.split(qkv, 3, axis=-1)
    q = device_rope(_split_heads(q, geom), cos, sin, cfg.rope_convention)
    k = device_rope(_split_heads(k, geom), cos, sin, cfg.rope_convention)
    v = _split_heads(v, geom)
    stash("q_rope", _merge_heads(q, geom))
    stash("k_rope", _merge_heads(k, geom))
    # the softmax kernel has no scale parameter (attn/DESIGN.md:44-45,205);
    # conservative pre-scale of the S tile: one extra bf16 store.
    scale16 = float(round_bf16(np.asarray(geom.attn_scale, dtype=np.float32)))
    o = np.empty_like(q)
    for hd in range(geom.heads):
        S = device_gemm(q[:, hd, :], k[:, hd, :], cfg)  # (seq, seq) bf16
        S = round_bf16(S * _F32(scale16))
        P = device_softmax_rows(S)
        # V enters as the kernel's (n, K) = (head_dim, seq) B operand
        v_h = np.ascontiguousarray(v[:, hd, :].T)
        o[:, hd, :] = device_gemm(P, v_h, cfg)
    attn_o = _merge_heads(o, geom)
    stash("attn_o", attn_o)
    proj = device_gemm(attn_o, weights["w_o"], cfg)
    x = round_bf16(x + proj)  # add_sized residual
    stash("x_mid", x)
    # ---- FFN branch -------------------------------------------------------
    n2 = device_rms_norm_weighted(x, weights["gamma2"], cfg)
    m2 = device_adaln_modulate(n2, s2, b2)
    stash("adaln2", m2)
    gate = device_gemm(m2, weights["w_gate"], cfg)
    up = device_gemm(m2, weights["w_up"], cfg)
    sg = device_swiglu(gate, up)
    stash("swiglu", sg)
    ff = device_gemm(sg, weights["w_down"], cfg)
    x = round_bf16(x + ff)  # add_sized residual
    stash("x_out", x)
    return x


# ---------------------------------------------------------------------------
# Divergence metrics
# ---------------------------------------------------------------------------
def divergence(ref, dev, *, rel_gate_frac: float = 0.01) -> dict:
    """Compare the float oracle (ref) against the device model (dev).

    * ``max_abs``        max |dev - ref|
    * ``max_rel``        max |dev-ref| / |ref| over elements with
                         |ref| >= rel_gate_frac * max|ref|  (the "max rel
                         error"; near-zero outputs make the ungated version
                         meaningless)
    * ``rel_to_absmax``  max_abs / max|ref|  (the repo's "rel" convention,
                         cf. KNOWN-ISSUES.md "rel 1.011607e-02")
    * ``max_rel_ungated`` max |dev-ref| / max(|ref|, tiny) -- reported for
                         completeness only
    """
    r = np.asarray(ref, dtype=np.float64)
    d = np.asarray(dev, dtype=np.float64)
    if r.shape != d.shape:
        raise ValueError(f"divergence: shape mismatch {r.shape} vs {d.shape}")
    err = np.abs(d - r)
    absmax = float(np.abs(r).max())
    floor = rel_gate_frac * absmax
    gate = np.abs(r) >= floor
    max_abs = float(err.max())
    max_rel = float((err[gate] / np.abs(r[gate])).max()) if gate.any() else 0.0
    tiny = np.finfo(np.float64).tiny
    max_rel_ungated = float((err / np.maximum(np.abs(r), tiny)).max())
    rms = float(np.sqrt(np.mean(err * err)))
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "rel_to_absmax": max_abs / absmax if absmax > 0 else 0.0,
        "max_rel_ungated": max_rel_ungated,
        "rms": rms,
        "ref_absmax": absmax,
        "n_elements": int(r.size),
    }


def run_both(inputs: dict, weights: dict, geom: Geometry, cfg: DeviceCfg,
             *, collect: bool = False):
    """Run both variants; returns (float_out, device_out, timings, collects)."""
    col_f, col_d = ({} if collect else None), ({} if collect else None)
    t0 = time.perf_counter()
    out_f = dit_block_float(inputs, weights, geom, cfg, collect=col_f)
    t1 = time.perf_counter()
    out_d = dit_block_device(inputs, weights, geom, cfg, collect=col_d)
    t2 = time.perf_counter()
    timings = {"float_s": t1 - t0, "device_s": t2 - t1}
    return out_f, out_d, timings, (col_f, col_d)


