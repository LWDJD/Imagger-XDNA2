# SPDX-License-Identifier: MIT
"""encode_api.py — Z-Image text encoder through the llama.cpp LIBRARY.

Calls an upstream ``llama.dll`` (llama.cpp mainline >= 0.9.0, Vulkan build)
directly with ctypes: the 13 chat-prompt token ids go in as ONE sequence via
``llama_decode``, and the per-token ``hidden_states[-2]`` rows come out of the
layer-input tap, which is part of mainline (src/llama-ext.h):

    setter: ?llama_set_embeddings_layer_inp@@YAXPEAUllama_context@@I_N@Z
    getter: ?llama_get_embeddings_layer_inp@@YAPEAMPEAUllama_context@@I@Z

(tap semantics, from the fork's own docs: "input of layer k+1 is output of
layer k" — gguf-py/gguf/scripts/gguf_dspark_to_dflash.py:26 — so lid=35 is
the output of decoder block 34 == HF hidden_states[-2] for the 36-layer
Qwen3 tower).

Output npz:  pe (1, L, 2560) float32  +  tokens (L,) int32  +  prompt  +
layer (int32).  This is the payload `zimage-port/generate.py --embeds`
consumes unchanged.

No encoder is written here and no llama.cpp is built: this is the same
library the CLI and the llama-server wrap, driven with the token-id array.

Usage:
    python encode_api.py [--out OUT.npz] [--prompt-file chat_prompt.txt]
                         [--ids 151644,872,...] [--lid 35]
                         [--ref ab1b0a2d17949d71.npz] [--also-postnorm]
"""

from __future__ import annotations

import argparse
import ctypes
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent

# llama.cpp runtime: DLLs from an upstream Vulkan build (-DGGML_VULKAN=ON).
# Resolved in this order, first hit wins:
#   1. ZIMAGE_LLAMA_DIR                      -- the supported way to point elsewhere
#   2. <checkout>/third_party/llama.cpp/build/bin   -- where a fresh clone puts it
#   3. the legacy path below                 -- a previous agent's temp build; it is
#      NOT a dependency of this project, only a convenience so an existing machine
#      keeps working. Anything reproducible must use 1 or 2.
def _find_llama_dir() -> pathlib.Path:
    env = os.environ.get("ZIMAGE_LLAMA_DIR")
    if env and (pathlib.Path(env) / "llama.dll").is_file():
        return pathlib.Path(env)
    repo = HERE.parents[2] / "third_party" / "llama.cpp" / "build" / "bin"
    if (repo / "llama.dll").is_file():
        return repo
    legacy = pathlib.Path(r"D:\Project\openhanako\workbench\prism-vulkan")
    if (legacy / "llama.dll").is_file():
        return legacy
    return repo          # let the loader fail with the documented path


LLAMA_DIR = _find_llama_dir()
LLAMA_DLL = LLAMA_DIR / "llama.dll"


# The 8.05 GB Qwen3-4B captioner GGUF is third-party-derived and never committed.
# Search order, first hit wins:
#   1. ZIMAGE_TE_GGUF                                 -- the supported override
#   2. <checkout>/webui/models/<name>                 -- where a fresh clone puts it
#   3. <workspace>/webui/models/<name>                -- the source-tree convention
#   4. <workspace>/<name>                             -- flat drop next to the trees
#   5. the legacy session-temp path                   -- may no longer exist
NAME = "zimage-te-qwen3-4b-bf16.gguf"


def _find_gguf() -> pathlib.Path:
    env = os.environ.get("ZIMAGE_TE_GGUF")
    if env:
        return pathlib.Path(env)
    repo = HERE.parents[1]                    # the checkout root
    ws = repo.parent                          # the workspace it sits in
    for cand in (repo / "webui" / "models" / NAME,
                 ws / "webui" / "models" / NAME,
                 ws / NAME):
        if cand.is_file():
            return cand
    legacy = pathlib.Path(r"C:\Users\15081\AppData\Local\Temp\dsh-nPUWK7") / NAME
    if legacy.is_file():
        return legacy
    return repo / "webui" / "models" / NAME   # the documented place to put it


GGUF = _find_gguf()

# The known chat prompt for 'a mountain lake at sunrise' (decode_ids.py,
# round-trip exact).
EXPECTED_IDS = [151644, 872, 198, 64, 16301, 21800, 518, 63819, 151645, 198,
                151644, 77091, 198]

SET_LAYER_INP = "?llama_set_embeddings_layer_inp@@YAXPEAUllama_context@@I_N@Z"
GET_LAYER_INP = "?llama_get_embeddings_layer_inp@@YAPEAMPEAUllama_context@@I@Z"

N_CTX = 512          # --ctx-size 512 (device memory)
NO_KV_OFFLOAD = True  # --no-kv-offload (16.5 GB iGPU)
LLAMA_POOLING_TYPE_NONE = 0


# --------------------------------------------------------------------------
# struct layouts — include/llama.h of the prism fork (llama-ext.h/llama.h).
# The binary is an adjacent commit of the same fork; every write/read below is
# verified at runtime (n_ctx/n_ubatch/pooling read back after init), so a
# layout drift would be caught before any numbers are produced.
# --------------------------------------------------------------------------
class llama_model_params(ctypes.Structure):
    _fields_ = [
        ("dspark_head_source", ctypes.c_void_p),
        ("devices", ctypes.c_void_p),
        ("tensor_buft_overrides", ctypes.c_void_p),
        ("n_gpu_layers", ctypes.c_int32),
        ("split_mode", ctypes.c_int32),
        ("load_mode", ctypes.c_int32),
        ("main_gpu", ctypes.c_int32),
        ("tensor_split", ctypes.c_void_p),
        ("progress_callback", ctypes.c_void_p),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides", ctypes.c_void_p),
        ("vocab_only", ctypes.c_bool),
        ("check_tensors", ctypes.c_bool),
        ("use_extra_bufts", ctypes.c_bool),
        ("no_host", ctypes.c_bool),
        ("no_alloc", ctypes.c_bool),
        ("load_mtp", ctypes.c_bool),
    ]


class llama_context_params(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_rs_seq", ctypes.c_uint32),
        ("n_outputs_max", ctypes.c_uint32),
        ("n_outputs_max_per_seq", ctypes.c_uint32),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        ("ctx_type", ctypes.c_int32),
        ("rope_scaling_type", ctypes.c_int32),
        ("pooling_type", ctypes.c_int32),
        ("attention_type", ctypes.c_int32),
        ("flash_attn_type", ctypes.c_int32),
        ("rope_freq_base", ctypes.c_float),
        ("rope_freq_scale", ctypes.c_float),
        ("yarn_ext_factor", ctypes.c_float),
        ("yarn_attn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("yarn_orig_ctx", ctypes.c_uint32),
        ("defrag_thold", ctypes.c_float),
        ("cb_eval", ctypes.c_void_p),
        ("cb_eval_user_data", ctypes.c_void_p),
        ("type_k", ctypes.c_int32),
        ("type_v", ctypes.c_int32),
        ("path_kv_mean_center", ctypes.c_void_p),
        ("abort_callback", ctypes.c_void_p),
        ("abort_callback_data", ctypes.c_void_p),
        ("embeddings", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("no_perf", ctypes.c_bool),
        ("op_offload", ctypes.c_bool),
        ("swa_full", ctypes.c_bool),
        ("kv_unified", ctypes.c_bool),
        ("samplers", ctypes.c_void_p),
        ("n_samplers", ctypes.c_size_t),
        ("ctx_other", ctypes.c_void_p),
    ]


class llama_batch(ctypes.Structure):
    _fields_ = [
        ("n_tokens", ctypes.c_int32),
        ("token", ctypes.POINTER(ctypes.c_int32)),
        ("embd", ctypes.c_void_p),
        ("pos", ctypes.POINTER(ctypes.c_int32)),
        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
        ("seq_id", ctypes.POINTER(ctypes.POINTER(ctypes.c_int32))),
        ("logits", ctypes.POINTER(ctypes.c_int8)),
    ]


def load_llama() -> ctypes.CDLL:
    os.add_dll_directory(str(LLAMA_DIR))
    os.environ["PATH"] = str(LLAMA_DIR) + os.pathsep + os.environ.get("PATH", "")
    lib = ctypes.WinDLL(str(LLAMA_DLL))
    # this build does NOT auto-register backend DLLs:
    #   "llama_model_load_from_file_impl: no backends are loaded. hint: use
    #    ggml_backend_load() or ggml_backend_load_all() ..."
    ggml = ctypes.WinDLL(str(LLAMA_DIR / "ggml.dll"))
    try:
        load_all = ggml.ggml_backend_load_all_from_path
        load_all.restype = None
        load_all.argtypes = [ctypes.c_char_p]
        load_all(str(LLAMA_DIR).encode())
    except AttributeError:
        load_all = ggml.ggml_backend_load_all
        load_all.restype = None
        load_all.argtypes = []
        load_all()
    return lib


def bind(lib: ctypes.CDLL) -> None:
    lib.llama_backend_init.restype = ctypes.c_int32
    lib.llama_backend_init.argtypes = []
    lib.llama_model_default_params.restype = llama_model_params
    lib.llama_model_default_params.argtypes = []
    lib.llama_context_default_params.restype = llama_context_params
    lib.llama_context_default_params.argtypes = []
    lib.llama_load_model_from_file.restype = ctypes.c_void_p
    lib.llama_load_model_from_file.argtypes = [ctypes.c_char_p,
                                              llama_model_params]
    lib.llama_init_from_model.restype = ctypes.c_void_p
    lib.llama_init_from_model.argtypes = [ctypes.c_void_p,
                                          llama_context_params]
    lib.llama_free.restype = None
    lib.llama_free.argtypes = [ctypes.c_void_p]
    lib.llama_free_model.restype = None
    lib.llama_free_model.argtypes = [ctypes.c_void_p]
    lib.llama_model_n_embd.restype = ctypes.c_int32
    lib.llama_model_n_embd.argtypes = [ctypes.c_void_p]
    lib.llama_model_n_embd_out.restype = ctypes.c_int32
    lib.llama_model_n_embd_out.argtypes = [ctypes.c_void_p]
    lib.llama_model_n_layer.restype = ctypes.c_int32
    lib.llama_model_n_layer.argtypes = [ctypes.c_void_p]
    lib.llama_n_ctx.restype = ctypes.c_uint32
    lib.llama_n_ctx.argtypes = [ctypes.c_void_p]
    lib.llama_n_ubatch.restype = ctypes.c_uint32
    lib.llama_n_ubatch.argtypes = [ctypes.c_void_p]
    lib.llama_pooling_type.restype = ctypes.c_int32
    lib.llama_pooling_type.argtypes = [ctypes.c_void_p]
    lib.llama_set_embeddings.restype = None
    lib.llama_set_embeddings.argtypes = [ctypes.c_void_p, ctypes.c_bool]
    lib.llama_decode.restype = ctypes.c_int32
    lib.llama_decode.argtypes = [ctypes.c_void_p, llama_batch]
    lib.llama_get_embeddings_ith.restype = ctypes.POINTER(ctypes.c_float)
    lib.llama_get_embeddings_ith.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.llama_model_get_vocab.restype = ctypes.c_void_p
    lib.llama_model_get_vocab.argtypes = [ctypes.c_void_p]
    lib.llama_tokenize.restype = ctypes.c_int32
    lib.llama_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                  ctypes.c_int32,
                                  ctypes.POINTER(ctypes.c_int32),
                                  ctypes.c_int32, ctypes.c_bool,
                                  ctypes.c_bool]
    lib.llama_token_to_piece.restype = ctypes.c_int32
    lib.llama_token_to_piece.argtypes = [ctypes.c_void_p, ctypes.c_int32,
                                        ctypes.c_char_p, ctypes.c_int32,
                                        ctypes.c_int32, ctypes.c_bool]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "pe_api.npz"))
    ap.add_argument("--prompt-file", default=str(HERE / "chat_prompt.txt"))
    ap.add_argument("--ids", default=None,
                    help="comma-separated token ids; bypasses tokenization")
    ap.add_argument("--lid", type=int, default=35,
                    help="layer-input tap: output of decoder block lid-1 "
                         "(35 -> hidden_states[-2] on the 36-layer tower)")
    ap.add_argument("--ref", default=None,
                    help="reference npz (e.g. prompt_cache/ab1b0a2d17949d71.npz) "
                         "to compare against")
    ap.add_argument("--also-postnorm", action="store_true",
                    help="also emit llama_get_embeddings_ith (post-final-norm) "
                         "for cross-checking")
    args = ap.parse_args()

    if args.ids:
        ids = [int(x) for x in args.ids.split(",")]
    else:
        ids = None

    lib = load_llama()
    bind(lib)

    # resolve the two mangled extension exports (fail here, loudly, if absent)
    try:
        set_layer_inp = getattr(lib, SET_LAYER_INP)
        get_layer_inp = getattr(lib, GET_LAYER_INP)
    except AttributeError as e:
        print(f"FAILED to resolve mangled tap exports: {e!r}")
        return 2
    set_layer_inp.restype = None
    set_layer_inp.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_bool]
    get_layer_inp.restype = ctypes.POINTER(ctypes.c_float)
    get_layer_inp.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    print(f"resolved setter: {SET_LAYER_INP}")
    print(f"resolved getter: {GET_LAYER_INP}")

    lib.llama_backend_init()

    mparams = lib.llama_model_default_params()
    cparams = lib.llama_context_default_params()
    print(f"model_default_params: n_gpu_layers={mparams.n_gpu_layers} "
          f"split_mode={mparams.split_mode} load_mode={mparams.load_mode} "
          f"main_gpu={mparams.main_gpu}")
    print(f"context_default_params: n_ctx={cparams.n_ctx} "
          f"n_batch={cparams.n_batch} n_ubatch={cparams.n_ubatch} "
          f"n_seq_max={cparams.n_seq_max} pooling_type={cparams.pooling_type} "
          f"embeddings={cparams.embeddings} offload_kqv={cparams.offload_kqv}")

    mparams.n_gpu_layers = -1           # all layers on the Vulkan device
    cparams.n_ctx = N_CTX
    cparams.n_batch = N_CTX
    cparams.n_ubatch = N_CTX
    cparams.n_seq_max = 1
    cparams.pooling_type = LLAMA_POOLING_TYPE_NONE
    cparams.embeddings = True
    cparams.offload_kqv = not NO_KV_OFFLOAD

    t0 = time.perf_counter()
    model = lib.llama_load_model_from_file(str(GGUF).encode(), mparams)
    if not model:
        print("llama_load_model_from_file returned NULL")
        return 3
    t_load = time.perf_counter() - t0

    n_embd = lib.llama_model_n_embd(model)
    n_layer = lib.llama_model_n_layer(model)
    print(f"model loaded in {t_load:.1f}s: n_embd={n_embd} n_layer={n_layer}")
    assert n_embd == 2560 and n_layer == 36, "unexpected model geometry"

    ctx = lib.llama_init_from_model(model, cparams)
    if not ctx:
        print("llama_init_from_model returned NULL")
        return 3

    # ABI self-check: what we wrote must read back
    got_ctx, got_ub, got_pool = (lib.llama_n_ctx(ctx), lib.llama_n_ubatch(ctx),
                                 lib.llama_pooling_type(ctx))
    print(f"ctx readback: n_ctx={got_ctx} n_ubatch={got_ub} "
          f"pooling_type={got_pool}")
    assert got_ctx == N_CTX and got_ub == N_CTX, "context_params layout drift"
    assert got_pool == LLAMA_POOLING_TYPE_NONE, "pooling_type offset drift"

    vocab = lib.llama_model_get_vocab(model)

    if ids is None:
        text = pathlib.Path(args.prompt_file).read_text(encoding="utf-8")
        buf = (ctypes.c_int32 * (N_CTX))()
        n = lib.llama_tokenize(vocab, text.encode("utf-8"),
                               len(text.encode("utf-8")), buf, N_CTX,
                               False, True)   # add_special=False, parse_special=True
        if n < 0:
            print(f"llama_tokenize failed: {n}")
            return 4
        ids = list(buf[:n])
        print(f"tokenized prompt ({n} tokens): {ids}")
    else:
        text = f"<--ids {args.ids}>"

    if ids == EXPECTED_IDS:
        print("token ids match the known 13-id chat prompt exactly")
    else:
        print(f"NOTE: ids differ from the known chat prompt {EXPECTED_IDS}")

    L = len(ids)
    tokens_arr = (ctypes.c_int32 * L)(*ids)
    pos_arr = (ctypes.c_int32 * L)(*range(L))
    nseq_arr = (ctypes.c_int32 * L)(*[1] * L)
    seq0 = (ctypes.c_int32 * 1)(0)
    seq_ptr_arr = (ctypes.POINTER(ctypes.c_int32) * L)(
        *[ctypes.cast(seq0, ctypes.POINTER(ctypes.c_int32)) for _ in range(L)])
    logits_arr = (ctypes.c_int8 * L)(*[1] * L)

    batch = llama_batch()
    batch.n_tokens = L
    batch.token = tokens_arr
    batch.embd = None
    batch.pos = pos_arr
    batch.n_seq_id = nseq_arr
    batch.seq_id = seq_ptr_arr
    batch.logits = logits_arr

    lib.llama_set_embeddings(ctx, True)
    set_layer_inp(ctx, args.lid, True)
    print(f"capture lid={args.lid} enabled")

    t0 = time.perf_counter()
    rc = lib.llama_decode(ctx, batch)
    t_dec = time.perf_counter() - t0
    print(f"llama_decode: rc={rc} in {t_dec:.2f}s ({L} tokens, one sequence)")
    if rc != 0:
        print("llama_decode failed")
        return 5

    ptr = get_layer_inp(ctx, args.lid)
    if not ptr:
        print("llama_get_embeddings_layer_inp returned NULL")
        return 6
    pe = np.ctypeslib.as_array(ptr, shape=(L * 2560,)).copy().reshape(L, 2560)

    payload = {
        "pe": pe[np.newaxis].astype(np.float32),   # (1, L, 2560) float32
        "tokens": np.asarray(ids, dtype=np.int32),
        "prompt": np.asarray(text),
        "layer": np.asarray(args.lid, dtype=np.int32),
    }

    if args.also_postnorm:
        rows = []
        for i in range(L):
            p = lib.llama_get_embeddings_ith(ctx, i)
            rows.append(np.ctypeslib.as_array(p, shape=(2560,)).copy())
        payload["pe_postnorm"] = np.stack(rows)[np.newaxis].astype(np.float32)

    out = pathlib.Path(args.out)
    np.savez(out, **payload)
    print(f"saved {out}  pe {payload['pe'].shape} {payload['pe'].dtype}")

    if args.ref:
        ref = np.load(args.ref)
        rpe = ref["pe"].astype(np.float32)
        print(f"\nref {args.ref}: pe {rpe.shape} prompt={str(ref['prompt'])!r}")
        assert rpe.shape == payload["pe"].shape, (rpe.shape, payload["pe"].shape)
        d = payload["pe"] - rpe
        absd = np.abs(d)
        print(f"max abs err : {absd.max():.6g}")
        print(f"mean abs err: {absd.mean():.6g}")
        print(f"ref  |x| max: {np.abs(rpe).max():.6g}   "
              f"ours |x| max: {np.abs(payload['pe']).max():.6g}")
        a = payload["pe"][0]
        b = rpe[0]
        dot = (a * b).sum(axis=1)
        na = np.linalg.norm(a, axis=1)
        nb = np.linalg.norm(b, axis=1)
        cos = dot / (na * nb)
        for i, c in enumerate(cos):
            print(f"cos[{i:2d}] = {c:.8f}")
        print(f"cos per-row  : min={cos.min():.8f} mean={cos.mean():.8f}")
        flat = float((a.ravel() @ b.ravel()) /
                     (np.linalg.norm(a) * np.linalg.norm(b)))
        print(f"cos flattened: {flat:.10f}")
        if "pe_postnorm" in payload:
            q = payload["pe_postnorm"][0]
            d2 = np.abs(q - b).max()
            c2 = float((q.ravel() @ b.ravel()) /
                       (np.linalg.norm(q) * np.linalg.norm(b)))
            print(f"[cross-check] post-final-norm tap vs ref: "
                  f"max abs err {d2:.6g}  cos(flattened) {c2:.8f}")

    lib.llama_free(ctx)
    lib.llama_free_model(model)
    print(f"\nwall time: load {t_load:.1f}s + decode {t_dec:.2f}s")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
