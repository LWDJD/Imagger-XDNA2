# SPDX-License-Identifier: MIT
"""Prompt -> the 32 x 3840 caption rows the DiT consumes.

This mirrors zimage-port/generate.py's cap_prelude exactly:

    pad the caption features to the fixed 32-token width by repeating the
    last token, RMSNorm with cap_embedder.0.weight, Linear 2560 -> 3840 with
    cap_embedder.1.{weight,bias}, then overwrite the padded rows with the
    checkpoint's cap_pad_token.

The null-caption path in t3_real.cap_embedding is the special case L=0
(all 32 rows are cap_pad_token).  The text encoder itself is not reimplemented
here: webui/te_worker.py drives the verified llama.cpp ctypes driver
(imaggers/textenc/encode_api.py) and returns hidden_states[-2].
"""
from __future__ import annotations

import hashlib

import numpy as np

CAP_TOKENS = 32
EPS = 1e-5


def _rms(x, w, eps=EPS):
    x = np.asarray(x, np.float32)
    v = (x * x).mean(-1, keepdims=True, dtype=np.float32)
    y = (x / np.sqrt(v + np.float32(eps))).astype(np.float32)
    return (y * np.asarray(w, np.float32)).astype(np.float32)


def _lin(x, w, b):
    y = np.asarray(np.asarray(x, np.float32) @ np.asarray(w, np.float32).T,
                   np.float32)
    return (y + np.asarray(b, np.float32)).astype(np.float32)


def validate_pe(pe, cap_tokens=CAP_TOKENS):
    pe = np.asarray(pe, np.float32)
    if pe.ndim != 3 or pe.shape[0] != 1 or pe.shape[2] != 2560:
        raise ValueError(f"pe must be (1, L, 2560), got {pe.shape}")
    L = int(pe.shape[1])
    if L < 1:
        raise ValueError("pe has zero tokens")
    if L > cap_tokens:
        raise ValueError(
            f"prompt is {L} tokens (chat template included), but the M=96 "
            f"image has cap_tokens={cap_tokens}. Shorten the prompt.")
    if not np.isfinite(pe).all():
        raise ValueError("pe contains non-finite values")
    return pe


def build_cap_embedding(pe, H, cap_tokens=CAP_TOKENS):
    """pe (1,L,2560) + host tensors -> (32,3840) float32 on the bf16 grid."""
    pe = validate_pe(pe, cap_tokens)
    L = int(pe.shape[1])
    last = pe[0, -1:, :]
    padded = (pe[0] if L == cap_tokens
              else np.concatenate([pe[0], np.repeat(last, cap_tokens - L, 0)], 0))
    padded = np.ascontiguousarray(padded, np.float32)
    emb = _rms(padded, H["cap_embedder.0.weight"])
    emb = _lin(emb, H["cap_embedder.1.weight"], H["cap_embedder.1.bias"])
    if L < cap_tokens:
        emb = emb.copy()
        emb[L:] = np.asarray(H["cap_pad_token"], np.float32).reshape(1, -1)
    return np.ascontiguousarray(emb, np.float32)


def pe_sha256(pe):
    return hashlib.sha256(np.ascontiguousarray(pe, np.float32).tobytes()).hexdigest()


def arrangement_info(pe):
    pe = np.asarray(pe, np.float32)
    return {
        "shape": list(pe.shape),
        "tokens": int(pe.shape[1]),
        "sha256": pe_sha256(pe),
        "absmax": float(np.abs(pe).max()),
        "mean_abs": float(np.abs(pe).mean()),
    }
