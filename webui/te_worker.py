# SPDX-License-Identifier: MIT
"""py312 worker: run the read-only llama.cpp text-encoder driver.

We import imaggers/textenc/encode_api.py as a module and override only its
module-level GGUF path (the checked-in driver hard-codes a temp path from a
previous session) and, in CPU mode, the backend loader.  Nothing in
imaggers/textenc is edited.

Backends:
  gpu  the verified Vulkan path (ggml_backend_load_all_from_path)
  cpu  load ONLY ggml-cpu-*.dll and set n_gpu_layers=0.  This is a fallback
       for a machine whose GPU memory is held by other apps; the same llama.cpp
       build, same GGUF, same layer-input tap.  Measured PE vs the Vulkan tap:
       cosine 1.000000, max abs 0.37 on |x|max 13,730.

Two modes (T21 2a-a):
  one shot   the delivered mode: py312 te_worker.py --gguf ... --prompt-file
             ... --out ... --backend gpu|cpu.  Loads the model, encodes once,
             exits.  Costs the 8.05 GB GGUF load on every call (7.3 s measured).
  resident   --serve: load the model ONCE, print a ready event, then answer
             encode requests on stdin as one JSON object per line.  Each
             request gets a FRESH llama context (init + free per request) so
             the call sequence is exactly the one-shot sequence after the
             shared model load; that is what makes the output byte-identical
             (gate: pe sha256 equal, see webui/evidence/T21-2A2B-STARTUP.md).
             OPT-IN (ZIMAGE_TE_RESIDENT=1): the resident worker keeps the
             8.05 GB model in iGPU memory, which measures 6.74 GB of commit on
             this host and pushes the next job's auto memory mode to
             --no-cache-all (measured 25.8 s -> 59.6 s device loop).  The
             default WebUI path is one-shot; this mode is for a host with
             enough RAM to hold the encoder and the device path at once.

stdout carries both the driver's own log text and the protocol; protocol lines
are recognised by the "@@TE@@ " prefix, everything else is log text.
"""
from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
import pathlib
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
NPU = HERE.parent
ENCODE_API = NPU / "imaggers" / "textenc" / "encode_api.py"
TE_MARK = "@@TE@@ "


def _cpu_only(mod):
    """Monkeypatch encode_api to load the CPU backend and zero GPU layers."""
    d = mod.LLAMA_DIR

    def load_cpu_only():
        os.add_dll_directory(str(d))
        os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
        lib = ctypes.WinDLL(str(d / "llama.dll"))
        ggml = ctypes.WinDLL(str(d / "ggml.dll"))
        cpu = sorted(d.glob("ggml-cpu-*.dll"))
        if not cpu:
            raise RuntimeError(f"no ggml-cpu-*.dll in {d}")
        fn = ggml.ggml_backend_load
        fn.restype = ctypes.c_void_p
        fn.argtypes = [ctypes.c_char_p]
        handle = fn(str(cpu[0]).encode())
        print(f"[te_worker] CPU backend {cpu[0].name} -> {handle}", flush=True)
        if not handle:
            raise RuntimeError("ggml_backend_load returned NULL")
        return lib

    mod.load_llama = load_cpu_only
    orig_bind = mod.bind

    def bind_cpu(lib):
        orig_bind(lib)
        real = lib.llama_load_model_from_file

        def load(path, params):
            params.n_gpu_layers = 0
            print("[te_worker] n_gpu_layers=0 (CPU backend)", flush=True)
            return real(path, params)

        lib.llama_load_model_from_file = load

    mod.bind = bind_cpu


def _load_driver():
    """Import imaggers/textenc/encode_api.py under a private module name."""
    if not ENCODE_API.is_file():
        print(f"[te_worker] missing driver: {ENCODE_API}")
        return None
    spec = importlib.util.spec_from_file_location("encode_api_webui", ENCODE_API)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["encode_api_webui"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        traceback.print_exc()
        return None
    return mod


def _taps(lib):
    """Resolve the fork's two mangled layer-input exports."""
    set_layer_inp = getattr(lib, "?llama_set_embeddings_layer_inp@@YAXPEAUllama_context@@I_N@Z")
    get_layer_inp = getattr(lib, "?llama_get_embeddings_layer_inp@@YAPEAMPEAUllama_context@@I@Z")
    set_layer_inp.restype = None
    set_layer_inp.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_bool]
    get_layer_inp.restype = ctypes.POINTER(ctypes.c_float)
    get_layer_inp.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    print(f"resolved setter: ?llama_set_embeddings_layer_inp@@YAXPEAUllama_context@@I_N@Z")
    print(f"resolved getter: ?llama_get_embeddings_layer_inp@@YAPEAMPEAUllama_context@@I@Z")
    return set_layer_inp, get_layer_inp


def _ctypes_params(mod, lib):
    """The exact model/context params encode_api.main() uses."""
    mparams = lib.llama_model_default_params()
    cparams = lib.llama_context_default_params()
    mparams.n_gpu_layers = -1           # all layers on the Vulkan device
    cparams.n_ctx = mod.N_CTX
    cparams.n_batch = mod.N_CTX
    cparams.n_ubatch = mod.N_CTX
    cparams.n_seq_max = 1
    cparams.pooling_type = mod.LLAMA_POOLING_TYPE_NONE
    cparams.embeddings = True
    cparams.offload_kqv = not mod.NO_KV_OFFLOAD
    return mparams, cparams


def _encode_once(mod, lib, model, vocab, taps, prompt_file, out_path, lid,
                 abi_state):
    """One encode, with a FRESH context -- the one-shot call sequence."""
    set_layer_inp, get_layer_inp = taps
    _mp, cparams = _ctypes_params(mod, lib)
    t0 = time.perf_counter()
    ctx = lib.llama_init_from_model(model, cparams)
    ctx_s = time.perf_counter() - t0
    if not ctx:
        raise RuntimeError("llama_init_from_model returned NULL")
    try:
        got_ctx = lib.llama_n_ctx(ctx)
        got_ub = lib.llama_n_ubatch(ctx)
        got_pool = lib.llama_pooling_type(ctx)
        if not abi_state.get("checked"):
            print(f"ctx readback: n_ctx={got_ctx} n_ubatch={got_ub} "
                  f"pooling_type={got_pool}")
            if got_ctx != mod.N_CTX or got_ub != mod.N_CTX:
                raise RuntimeError("context_params layout drift")
            if got_pool != mod.LLAMA_POOLING_TYPE_NONE:
                raise RuntimeError("pooling_type offset drift")
            abi_state["checked"] = True

        text = pathlib.Path(prompt_file).read_text(encoding="utf-8")
        tb = text.encode("utf-8")
        buf = (ctypes.c_int32 * (mod.N_CTX))()
        n = lib.llama_tokenize(vocab, tb, len(tb), buf, mod.N_CTX, False, True)
        if n < 0:
            raise RuntimeError(f"llama_tokenize failed: {n}")
        ids = list(buf[:n])
        print(f"tokenized prompt ({n} tokens): {ids}")
        if ids == mod.EXPECTED_IDS:
            print("token ids match the known 13-id chat prompt exactly")
        else:
            print(f"NOTE: ids differ from the known chat prompt {mod.EXPECTED_IDS}")

        L = len(ids)
        tokens_arr = (ctypes.c_int32 * L)(*ids)
        pos_arr = (ctypes.c_int32 * L)(*range(L))
        nseq_arr = (ctypes.c_int32 * L)(*[1] * L)
        seq0 = (ctypes.c_int32 * 1)(0)
        seq_ptr_arr = (ctypes.POINTER(ctypes.c_int32) * L)(
            *[ctypes.cast(seq0, ctypes.POINTER(ctypes.c_int32)) for _ in range(L)])
        logits_arr = (ctypes.c_int8 * L)(*[1] * L)

        batch = mod.llama_batch()
        batch.n_tokens = L
        batch.token = tokens_arr
        batch.embd = None
        batch.pos = pos_arr
        batch.n_seq_id = nseq_arr
        batch.seq_id = seq_ptr_arr
        batch.logits = logits_arr

        lib.llama_set_embeddings(ctx, True)
        set_layer_inp(ctx, lid, True)
        print(f"capture lid={lid} enabled")

        t1 = time.perf_counter()
        rc = lib.llama_decode(ctx, batch)
        t_dec = time.perf_counter() - t1
        print(f"llama_decode: rc={rc} in {t_dec:.2f}s ({L} tokens, one sequence)")
        if rc != 0:
            raise RuntimeError(f"llama_decode failed: rc={rc}")

        ptr = get_layer_inp(ctx, lid)
        if not ptr:
            raise RuntimeError("llama_get_embeddings_layer_inp returned NULL")
        import numpy as np
        pe = np.ctypeslib.as_array(ptr, shape=(L * 2560,)).copy().reshape(L, 2560)
        payload = {
            "pe": pe[np.newaxis].astype(np.float32),
            "tokens": np.asarray(ids, dtype=np.int32),
            "prompt": np.asarray(text),
            "layer": np.asarray(lid, dtype=np.int32),
        }
        out = pathlib.Path(out_path)
        np.savez(out, **payload)
        print(f"saved {out}  pe {payload['pe'].shape} {payload['pe'].dtype}")
        return {"tokens": L, "decode_s": round(t_dec, 3), "ctx_s": round(ctx_s, 3),
                "out": str(out)}
    finally:
        lib.llama_free(ctx)


def _serve(args) -> int:
    """Resident mode: load the model once, then answer encode requests."""
    def emit(obj):
        sys.stdout.write(TE_MARK + json.dumps(obj) + "\n")
        sys.stdout.flush()

    mod = _load_driver()
    if mod is None:
        emit({"event": "ready", "ok": False, "error": f"missing driver {ENCODE_API}"})
        return 2
    if args.backend == "cpu":
        try:
            _cpu_only(mod)
        except Exception:
            traceback.print_exc()
            emit({"event": "ready", "ok": False, "error": traceback.format_exc()[-2000:]})
            return 2
    mod.GGUF = pathlib.Path(args.gguf)
    try:
        lib = mod.load_llama()
        mod.bind(lib)
        taps = _taps(lib)
        lib.llama_backend_init()
        mparams, _cparams = _ctypes_params(mod, lib)
        print(f"model_default_params: n_gpu_layers={mparams.n_gpu_layers} "
              f"split_mode={mparams.split_mode} load_mode={mparams.load_mode} "
              f"main_gpu={mparams.main_gpu}")
        t0 = time.perf_counter()
        model = lib.llama_load_model_from_file(str(mod.GGUF).encode(), mparams)
        t_load = time.perf_counter() - t0
        if not model:
            print("llama_load_model_from_file returned NULL")
            emit({"event": "ready", "ok": False,
                  "error": "llama_load_model_from_file returned NULL"})
            return 3
        n_embd = lib.llama_model_n_embd(model)
        n_layer = lib.llama_model_n_layer(model)
        print(f"model loaded in {t_load:.1f}s: n_embd={n_embd} n_layer={n_layer}")
        if not (n_embd == 2560 and n_layer == 36):
            emit({"event": "ready", "ok": False,
                  "error": f"unexpected model geometry {n_embd}/{n_layer}"})
            return 3
        vocab = lib.llama_model_get_vocab(model)
    except Exception:
        traceback.print_exc()
        emit({"event": "ready", "ok": False, "error": traceback.format_exc()[-2000:]})
        return 2
    emit({"event": "ready", "ok": True, "load_s": round(t_load, 3),
          "n_embd": n_embd, "n_layer": n_layer, "backend": args.backend,
          "gguf": str(mod.GGUF), "pid": os.getpid()})
    abi_state = {}
    served = 0
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except Exception:
            emit({"event": "error", "ok": False, "error": "unparsable request"})
            continue
        op = req.get("op")
        if op == "ping":
            emit({"event": "pong", "served": served})
            continue
        if op == "shutdown":
            emit({"event": "bye", "served": served})
            break
        if op != "encode":
            emit({"event": "error", "ok": False, "error": f"unknown op {op!r}"})
            continue
        rid = req.get("id")
        t_req = time.perf_counter()
        try:
            res = _encode_once(mod, lib, model, vocab, taps,
                               req["prompt_file"], req["out"],
                               int(req.get("lid", 35)), abi_state)
            served += 1
            out = dict(res)
            out.update({"event": "done", "id": rid, "ok": True,
                        "wall_s": round(time.perf_counter() - t_req, 3),
                        "served": served, "backend": args.backend})
            emit(out)
        except Exception:
            traceback.print_exc()
            emit({"event": "done", "id": rid, "ok": False,
                  "error": traceback.format_exc()[-4000:],
                  "wall_s": round(time.perf_counter() - t_req, 3),
                  "backend": args.backend})
    try:
        lib.llama_free_model(model)
    except Exception:
        pass
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--prompt-file")
    ap.add_argument("--out")
    ap.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--serve", action="store_true",
                    help="resident mode: load once, serve encode requests on stdin")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.serve:
        return _serve(args)
    if not args.prompt_file or not args.out:
        print("[te_worker] --prompt-file and --out are required in one-shot mode")
        return 2
    mod = _load_driver()
    if mod is None:
        return 2
    if args.backend == "cpu":
        try:
            _cpu_only(mod)
        except Exception:
            traceback.print_exc()
            return 2
    mod.GGUF = pathlib.Path(args.gguf)
    sys.argv = ["encode_api.py", "--prompt-file", args.prompt_file,
                "--out", args.out]
    try:
        rc = mod.main()
        return int(rc or 0)
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else (0 if code is None else 2)
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
