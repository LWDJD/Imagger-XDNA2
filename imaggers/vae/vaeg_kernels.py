# vaeg_kernels.py -*- Python -*-
#
# IRON wrappers + numpy references for `group_norm.cc` -- the Z-Image VAE
# GroupNorm(32, eps=1e-6, affine=True) kernel for AIE2P.
#
# Factory style follows `imaggers/kernels/kernel_wrappers.py` (the proven
# rms_norm_weighted/adaln_modulate wrappers): one factory per entry symbol,
# built through `aie.iron.kernels._common._make_extern`, with a
# KernelContract carrying alignments and the host reference.
#
# OBJECT SHARING (hard-won, _common.py:633-654): several entry symbols of
# ONE translation unit must share one object_file_name, or each digest-named
# .o carries every symbol of the .cc and the link sees duplicates.  Every
# factory below therefore passes an explicit `object_file_name` derived from
# the ARGUMENT SIZES (the parameterization), never from the symbol.
#
# THE ABI (see group_norm.cc's header for the full derivation):
#     group_norm_stats(x, stats, n)                  pass 1 accumulate
#     group_norm_finalize(stats, total, epsilon)     mean/rstd + reset
#     group_norm_apply(x, y, gamma, beta, stats, n, spatial)
#     group_norm_group(x, y, gamma, beta, n, spatial, epsilon)   one call
# `epsilon` is a Param (1e-6) -- the exact thing `norm.layer_norm` cannot
# express (eps fixed at 1e-5, gamma=1/beta=0 only).
#
# ALIGNMENT: x/y chunks are whole ObjectFifo objects (bases are aligned);
# gamma/beta/stats are read and written scalar-only, so only x and y carry
# `contract.alignments` 64 B (AIE2P floors misaligned vector addresses to
# 64 B silently -- `imaggers/kernels/FINDINGS.md`).
from __future__ import annotations

import os
import sys
from pathlib import Path

# mlir_aie setup: package roots on sys.path, PEANO + cache BEFORE `import aie`
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent.parent                       # ...\public_work
_PYLIBS = _ROOT / "_pylibs"
sys.path.append(str(_ROOT / "zimage-bfp16-npu" / "imaggers" / "vae"))
sys.path.append(str(_PYLIBS))
sys.path.append(str(_PYLIBS / "mlir_aie" / "python"))
os.environ.setdefault("PEANO_INSTALL_DIR", str(_PYLIBS / "llvm-aie" / "llvm-aie"))
os.environ.setdefault("NPU_CACHE_HOME", str(_HERE / "_npucache_vaeg_default"))

import numpy as np
from aie.iron.kernel import ExternalFunction
from aie.iron.kernels._common import KernelContract, Param, Trace, _make_extern
from aie.utils.compile.jit.markers import In, InOut, Out
from ml_dtypes import bfloat16

SRC = _HERE / "group_norm.cc"


# ---------------------------------------------------------------------------
# wrappers
# ---------------------------------------------------------------------------

def _obj_name(*sizes: int) -> str:
    """One object per ARGUMENT-SIZE parameterization (see module docstring).

    Every entry symbol of one parameterization MUST share one name, or each
    digest-named .o carries all four symbols of the .cc and the core link
    sees duplicates (measured: ``ld.lld: error: duplicate symbol:
    group_norm_stats`` when the factories derived the name from the symbol).
    """
    return "group_norm_" + "_".join(str(s) for s in sizes) + ".o"


def group_norm_stats(n: int, obj: str) -> ExternalFunction:
    """Pass 1: x (n,) bf16 -> accumulate sum / sum-of-squares into stats[4]."""
    return _make_extern(
        "group_norm_stats",
        SRC,
        [np.ndarray[(n,), np.dtype[bfloat16]], np.ndarray[(4,), np.dtype[np.float32]],
         np.int32],
        object_file_name=obj,
        contract=KernelContract(
            trace=Trace.none("event markers bracket calls only on the "
                             "apply/group entries' inner loops"),
            roles=(In, InOut, Param),
            reference=None,
        ),
    )


def group_norm_finalize(obj: str) -> ExternalFunction:
    """stats[4] -> mean/rstd from the accumulators, then zero them."""
    return _make_extern(
        "group_norm_finalize",
        SRC,
        [np.ndarray[(4,), np.dtype[np.float32]], np.int32, np.float32],
        object_file_name=obj,
        contract=KernelContract(
            trace=Trace.none("no user-visible output"),
            roles=(InOut, Param, Param),
            reference=None,
        ),
    )


def group_norm_apply(n: int, nch: int, obj: str) -> ExternalFunction:
    """Pass 2: normalize one chunk of n = nch*spatial elements (whole runs)."""
    return _make_extern(
        "group_norm_apply",
        SRC,
        [
            np.ndarray[(n,), np.dtype[bfloat16]],
            np.ndarray[(n,), np.dtype[bfloat16]],
            np.ndarray[(nch,), np.dtype[bfloat16]],
            np.ndarray[(nch,), np.dtype[bfloat16]],
            np.ndarray[(4,), np.dtype[np.float32]],
            np.int32,
            np.int32,
        ],
        object_file_name=obj,
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, Out, In, In, In, Param, Param),
            reference=None,
            alignments=((0, 64), (1, 64)),
        ),
    )


def group_norm_group(n: int, cg: int, obj: str | None = None) -> ExternalFunction:
    """One call, whole group: n = cg*spatial elements, cg channels of affine."""
    return _make_extern(
        "group_norm_group",
        SRC,
        [
            np.ndarray[(n,), np.dtype[bfloat16]],
            np.ndarray[(n,), np.dtype[bfloat16]],
            np.ndarray[(cg,), np.dtype[bfloat16]],
            np.ndarray[(cg,), np.dtype[bfloat16]],
            np.int32,
            np.int32,
            np.float32,
        ],
        object_file_name=obj or _obj_name("group", n, cg),
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, Out, In, In, Param, Param, Param),
            reference=None,
            alignments=((0, 64), (1, 64)),
        ),
    )


# ---------------------------------------------------------------------------
# nearest-2x replicate shuffles (pure data movement; see group_norm.cc's
# NEAREST-2x REPLICATE section and VAEG.md section 4)
# ---------------------------------------------------------------------------

def replicate2(n: int, obj: str | None = None) -> ExternalFunction:
    """y[2i] = y[2i+1] = x[i]: n elements -> 2n, each twice (bit-exact)."""
    return _make_extern(
        "replicate2",
        SRC,
        [
            np.ndarray[(n,), np.dtype[bfloat16]],
            np.ndarray[(2 * n,), np.dtype[bfloat16]],
            np.int32,
        ],
        object_file_name=obj or _obj_name("rep2", n),
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, Out, Param),
            reference=lambda x: np.repeat(np.asarray(x), 2, axis=-1),
            alignments=((0, 64), (1, 64)),
        ),
    )


def upsample_nearest2(n: int, w: int, obj: str | None = None) -> ExternalFunction:
    """Full nearest-2x of one (h, w) plane: n = h*w -> 4n (bit-exact)."""
    return _make_extern(
        "upsample_nearest2",
        SRC,
        [
            np.ndarray[(n,), np.dtype[bfloat16]],
            np.ndarray[(4 * n,), np.dtype[bfloat16]],
            np.int32,
            np.int32,
        ],
        object_file_name=obj or _obj_name("up2", n, w),
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, Out, Param, Param),
            reference=None,
            alignments=((0, 64), (1, 64)),
        ),
    )


# ---------------------------------------------------------------------------
# references: THREE layers, so the error is attributable (the run.py style)
#
#   oracle        torch GroupNorm in fp32 with fp32 gamma/beta, on the SAME
#                 input the CPU decoder feeds the module -- `imaggers/
#                 pipeline/vae_decode.py`'s arithmetic (diffusers
#                 AutoencoderKL -> torch.nn.GroupNorm(32, C, eps=1e-6)).
#   model_dev     what the KERNEL computes: bf16 in/gamma/beta, f32 stats
#                 with the kernel's variance formula (E[x^2] - mean^2),
#                 f32 affine chain, ONE bf16 rounding at the store.
#   model_exact   the same formula in float64 on the bf16 inputs -- the
#                 arithmetic ideal the device is judged against at bf16
#                 ulp level.
# ---------------------------------------------------------------------------

def round_bf16(x) -> np.ndarray:
    """float -> bf16 values (as float32 arrays), RNE (conv_even)."""
    xf = np.ascontiguousarray(x, dtype=np.float32)
    u = xf.view(np.uint32)
    rounding_bias = ((u >> 16) & 1) + 0x7FFF
    u2 = (u + rounding_bias) & np.uint32(0xFFFF0000)
    return u2.view(np.float32)


def bf16_ulp(x) -> np.ndarray:
    """Per-element ulp of |x| on the bf16 grid: 2**(floor(log2|x|) - 7)."""
    xf = np.asarray(x, np.float64)
    xf = np.where(xf == 0.0, np.float64(2.0**-126), np.abs(xf))
    exp = np.floor(np.log2(xf)).astype(np.int64)
    return np.exp2(exp - 7)


def group_norm_exact(x, gamma, beta, *, groups=32, eps=1e-6):
    """float64 ideal of the kernel's formula on already-bf16 inputs.

    x is (C, S) flat [group][channel][spatial] -- any layout where each
    group is a contiguous cg*S segment with channels of S contiguous.
    """
    xf = np.asarray(x, np.float64)
    g = np.asarray(gamma, np.float64)
    b = np.asarray(beta, np.float64)
    C, S = xf.shape
    cg = C // groups
    out = np.empty_like(xf)
    for gi in range(groups):
        seg = xf[gi * cg:(gi + 1) * cg].reshape(-1)
        mu = seg.mean()
        var = ((seg - mu) ** 2).mean()          # biased, as torch GroupNorm
        rstd = 1.0 / np.sqrt(var + eps)
        out[gi * cg:(gi + 1) * cg] = (
            (xf[gi * cg:(gi + 1) * cg] - mu) * rstd * g[gi * cg:(gi + 1) * cg, None]
            + b[gi * cg:(gi + 1) * cg, None]
        )
    return out


def group_norm_device_formula(x, gamma, beta, *, groups=32, eps=1e-6):
    """The kernel's arithmetic, emulated in numpy float32 (per-call form).

    Matches group_norm.cc order: sum/sumsq in f32 per 64-lane block reduce,
    then scalar f32 mean/var (var = E[x^2] - mean^2 via scalar_mul_sub),
    scalar_invsqrt, then per channel s = rstd*gamma, c0 = beta - mean*s and
    y = x*s + c0 in f32, ONE bf16 rounding at the store.  The 64-lane tree
    reduce_add order is mirrored as an f32 pairwise sum per block.
    """
    xf = np.ascontiguousarray(x, np.float32)
    g = np.asarray(gamma, np.float32)
    b = np.asarray(beta, np.float32)
    C, S = xf.shape
    cg = C // groups
    out = np.empty_like(xf)
    f = np.float32
    for gi in range(groups):
        seg = xf[gi * cg:(gi + 1) * cg].reshape(-1)
        n = seg.size
        # per-64-lane block sums, f32 pairwise inside the block
        nb = (n + 63) // 64
        sb = np.zeros(nb, np.float32)
        qb = np.zeros(nb, np.float32)
        for k in range(nb):
            blk = seg[k * 64:(k + 1) * 64]
            sb[k] = blk.sum(dtype=f)
            qb[k] = (blk * blk).sum(dtype=f)
        ssum = sb.sum(dtype=f)
        ssq = qb.sum(dtype=f)
        inv_n = f(1.0) / f(n)
        mean = f(ssum * inv_n)
        var = f(f(ssq * inv_n) - f(mean * mean))
        rstd = f(1.0) / np.float32(np.sqrt(np.float64(var + np.float32(eps))))
        for c in range(cg):
            s = f(rstd * g[gi * cg + c])
            c0 = f(b[gi * cg + c] - f(mean * s))
            yv = (xf[gi * cg + c] * s + c0).astype(np.float32)
            out[gi * cg + c] = yv
    return out.astype(np.float64)


def group_norm_oracle(x, gamma, beta, *, groups=32, eps=1e-6):
    """torch.nn.functional.group_norm in fp32 -- the CPU decoder's op."""
    import torch
    import torch.nn.functional as F

    xf = torch.from_numpy(np.ascontiguousarray(x, np.float32))[None]
    gf = torch.from_numpy(np.ascontiguousarray(gamma, np.float32))
    bf = torch.from_numpy(np.ascontiguousarray(beta, np.float32))
    with torch.no_grad():
        y = F.group_norm(xf, groups, gf, bf, eps)
    return y[0].numpy().astype(np.float64)
