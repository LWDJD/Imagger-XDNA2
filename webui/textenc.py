# SPDX-License-Identifier: MIT
"""Real prompt encoding through the verified llama.cpp ctypes driver.

No text encoder is written here.  te_worker.py loads
imaggers/textenc/encode_api.py as a module, points its GGUF path at the local
blob, and runs it.  The returned pe is hidden_states[-2] (1, L, 2560), exactly
what zimage-port/generate.py --embeds consumes.

T21 2a-a: a RESIDENT encoder exists here and is OFF by default.  Measured on
this host, one encode cost 11.0 s, of which 7.3 s was llama_load_model_from_file
of the same 8.05 GB GGUF, ~1.7 s of process/DLL cold start and 1.96 s of decode.
Keeping ONE py312 te_worker.py --serve child alive drops a later encode to
0.62 s (pe byte-identical, gate in webui/evidence/T21-2A2B-STARTUP.md) -- but
that worker holds the model in Vulkan/iGPU memory: measured cost 6.74 GB of
commit / 8.32 GB of available physical RAM (load 70 % -> 97 %), which drops the
idle commit below config.MEMORY_FAST_THRESHOLD_GB.  The NEXT job then auto-
selects --no-cache-all, and the device loop went 25.8 s -> 59.6 s (+34 s), so
the resident path is a 3x NET LOSS on this host.  It is therefore opt-in only
(ZIMAGE_TE_RESIDENT=1) and the delivered default is the one-shot path.  The
resident path is complete and gated (crash self-heal, Vulkan->CPU fallback,
pe sha256 equal); it is simply not the default where the model does not fit
alongside the device path.

Semantics that must not change, and do not:
  * the Vulkan backend is still tried first, a Vulkan failure still falls back
    to the same llama.cpp CPU backend, and the failure is still remembered for
    the life of this server (_GPU_OK) so no later job re-tries it;
  * the pe bytes are identical to the one-shot path (each request gets a fresh
    llama context, i.e. exactly the one-shot call sequence after the shared
    model load).  Gate: same prompt/seed sha256 equal, see
    webui/evidence/T21-2A2B-STARTUP.md;
  * the textenc bucket is still the FULL wall of this step, load included --
    nothing is moved out of the buckets;
  * a dead worker is rebuilt on the next call (crash self-heal), and
    ZIMAGE_TE_RESIDENT=0 restores the delivered one-shot behaviour.
"""
from __future__ import annotations

import atexit
import json
import os
import pathlib
import queue
import re
import subprocess
import threading
import time

import numpy as np

from . import config
from .proc import CREATE_NO_WINDOW, stream_process


TOKEN_RE = re.compile(r"tokenized prompt \((\d+) tokens\)")
LOAD_RE = re.compile(r"model loaded in ([0-9.]+)s")
DECODE_RE = re.compile(r"llama_decode: rc=(-?\d+) in ([0-9.]+)s")
SAVED_RE = re.compile(r"saved .*? pe \(1, (\d+), 2560\)")

TE_MARK = "@@TE@@ "
# Default OFF: measured 2a-a is a net loss on this host (see the module
# docstring and T21-2A2B-STARTUP.md).  ZIMAGE_TE_RESIDENT=1 opts in.
_RESIDENT = os.environ.get("ZIMAGE_TE_RESIDENT", "0").strip().lower() \
    not in ("0", "false", "no", "off", "")

# Remember a GPU-backend failure for the life of this server; retrying a
# Vulkan OOM on every job costs ~10 s and cannot succeed while a browser holds
# the iGPU.  A restart (or --backend gpu) retries it.
_GPU_OK = None


def render_chat(prompt):
    """The Qwen3 chat template the delivered encoder was validated with.

    chat_prompt.txt in imaggers/textenc is exactly this shape, and the known
    13-token prompt tokenizes identically with llama_tokenize(parse_special).
    """
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("prompt must not be empty")
    if len(prompt) > config.MAX_PROMPT_CHARS:
        raise ValueError(f"prompt is longer than {config.MAX_PROMPT_CHARS} characters")
    return "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"


# --------------------------------------------------------------- resident TE
class _WorkerDead(RuntimeError):
    """The resident worker exited or could not come up."""


class _WorkerTimeout(RuntimeError):
    """The resident worker did not answer within the request budget."""


class _Cancelled(RuntimeError):
    """The caller asked to stop; the worker was killed and must be rebuilt."""


class _Worker:
    """One py312 te_worker.py --serve child (one llama model, one backend)."""

    def __init__(self, backend, gguf):
        self.backend = backend
        self.gguf = pathlib.Path(gguf)
        self.proc = None
        self.q = None
        self.boot_s = 0.0        # spawn -> ready (process + DLLs + model load)
        self.load_s = 0.0        # the driver's own "model loaded in Xs"
        self.info = {}
        self.requests = 0
        self._next_id = 0

    # -- lifecycle
    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, sink):
        cmd = [config.PY312, config.TE_WORKER, "--serve", "--gguf",
               str(self.gguf), "--backend", self.backend]
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(
            [str(x) for x in cmd], cwd=str(config.NPU),
            env=config.subprocess_env("base"), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", bufsize=1,
            creationflags=CREATE_NO_WINDOW)
        self.q = queue.Queue()
        threading.Thread(target=self._read_loop, args=(self.proc, self.q),
                         daemon=True).start()
        ev = self._pump("ready", sink, 300.0, None)
        self.boot_s = time.perf_counter() - t0
        if not ev.get("ok"):
            self.close()
            raise _WorkerDead(
                "resident text encoder could not load: "
                + (ev.get("error") or "no detail").strip()[-900:])
        self.load_s = float(ev.get("load_s") or 0.0)
        self.info = dict(ev)

    def close(self):
        p, self.proc = self.proc, None
        if p is None:
            return
        try:
            if p.poll() is None:
                try:
                    p.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
                    p.stdin.flush()
                    p.wait(timeout=2.0)
                except Exception:  # noqa: BLE001
                    pass
                if p.poll() is None:
                    p.terminate()
                    p.wait(timeout=5.0)
        except Exception:  # noqa: BLE001
            try:
                p.kill()
            except Exception:  # noqa: BLE001
                pass

    # -- io
    @staticmethod
    def _read_loop(proc, q):
        try:
            for line in proc.stdout:
                q.put(line)
        finally:
            q.put(None)

    def _pump(self, want, sink, timeout, cancel_check):
        t0 = time.perf_counter()
        while True:
            if cancel_check and cancel_check():
                self.close()
                raise _Cancelled("cancelled while waiting for the text encoder")
            left = timeout - (time.perf_counter() - t0)
            if left <= 0:
                self.close()
                raise _WorkerTimeout(
                    f"no '{want}' event within {timeout:.0f} s")
            try:
                line = self.q.get(timeout=min(0.2, max(0.01, left)))
            except queue.Empty:
                if self.proc.poll() is not None and self.q.empty():
                    rc = self.proc.returncode
                    self.close()
                    raise _WorkerDead(
                        f"resident text encoder exited rc={rc} before '{want}'")
                continue
            if line is None:
                rc = self.proc.poll()
                self.close()
                raise _WorkerDead(
                    f"resident text encoder closed its output (rc={rc}) before '{want}'")
            line = line.rstrip("\r\n")
            if line.startswith(TE_MARK):
                try:
                    ev = json.loads(line[len(TE_MARK):])
                except Exception:  # noqa: BLE001
                    continue
                if ev.get("event") == want:
                    return ev
                continue
            if sink:
                sink(line)

    def encode(self, prompt_file, out_npz, sink, timeout, cancel_check, lid=35):
        if not self.alive():
            raise _WorkerDead("resident text encoder is not running")
        self._next_id += 1
        rid = self._next_id
        try:
            self.proc.stdin.write(json.dumps(
                {"op": "encode", "id": rid, "prompt_file": str(prompt_file),
                 "out": str(out_npz), "lid": int(lid)}) + "\n")
            self.proc.stdin.flush()
        except Exception as exc:  # noqa: BLE001
            raise _WorkerDead(f"write to the resident text encoder failed: {exc}")
        ev = self._pump("done", sink, timeout, cancel_check)
        if ev.get("id") not in (None, rid):
            raise _WorkerDead("resident text encoder answered a stale request")
        self.requests += 1
        return ev


_LOCK = threading.RLock()
_STATE = {"worker": None}


def _get_worker(backend, gguf, sink):
    """Return (worker, spawned).  Reuses a live worker of the same backend."""
    with _LOCK:
        w = _STATE["worker"]
        if (w is not None and w.alive() and w.backend == backend
                and w.gguf == pathlib.Path(gguf)):
            return w, False
        if w is not None:
            w.close()
        w = _Worker(backend, gguf)
        _STATE["worker"] = w
        try:
            w.start(sink)
        except Exception:
            _STATE["worker"] = None
            raise
        return w, True


def _drop_worker():
    with _LOCK:
        w = _STATE["worker"]
        _STATE["worker"] = None
    if w is not None:
        w.close()


def status():
    """Resident-encoder state for /api/status and the evidence trail."""
    with _LOCK:
        w = _STATE["worker"]
        if w is None or not w.alive():
            return {"resident": False, "backend": None, "load_s": None,
                    "boot_s": None, "served": 0, "pid": None}
        return {"resident": True, "backend": w.backend,
                "load_s": round(w.load_s, 3), "boot_s": round(w.boot_s, 3),
                "served": w.requests,
                "pid": w.proc.pid if w.proc is not None else None}


def shutdown():
    _drop_worker()


atexit.register(shutdown)


# ------------------------------------------------------------------- results
def _parse(lines, res, gguf, text, used):
    tokens = None
    load_s = decode_s = None
    joined = "\n".join(lines[-250:])
    m = TOKEN_RE.search(joined)
    if m:
        tokens = int(m.group(1))
    m = LOAD_RE.search(joined)
    if m:
        load_s = float(m.group(1))
    m = DECODE_RE.search(joined)
    if m:
        decode_s = float(m.group(2))
    m = SAVED_RE.search(joined)
    if m:
        tokens = int(m.group(1))
    out = dict(res)
    out.update({"tokens": tokens, "load_s": load_s, "decode_s": decode_s,
                "gguf": str(gguf), "prompt_text": text, "backend": used})
    return out


def _hint(joined):
    low = joined.lower()
    if "erroroutofdevicememory" in low or "out of device memory" in low:
        return ("Both the Vulkan and CPU text-encoder backends failed. "
                "Close GPU-heavy apps and retry; the CPU fallback should "
                "not need the GPU.")
    if "0xc0000005" in low or "access violation" in low:
        return "Text encoder process crashed; a fresh retry usually passes."
    return ""


def run(prompt, out_npz, log_path, gguf=None, on_line=None,
        timeout=300.0, cancel_check=None, backend="auto", resident=None):
    """Encode a prompt.  backend: auto | gpu | cpu.

    auto tries the verified Vulkan backend first and falls back to the same
    llama.cpp build's CPU backend if the GPU cannot allocate its model/context
    (a browser holding the iGPU is enough).  The gate pins the backend after
    the first pass, so a gate never compares two different encoders.

    resident=None follows ZIMAGE_TE_RESIDENT (default on).  resident=False is
    the delivered one-shot path; the byte gate uses it as the reference.
    """
    global _GPU_OK
    gguf = pathlib.Path(gguf) if gguf else config.find_te_gguf()
    if not gguf or not pathlib.Path(gguf).is_file():
        raise config.WebUIError(
            "text encoder GGUF not found",
            hint=("Set ZIMAGE_TE_GGUF to the 8.05 GB GGUF, or rebuild it with "
                  "the command in README-WEBUI.md. The blob is not in git."))
    prompt_file = pathlib.Path(out_npz).with_suffix(".prompt.txt")
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    text = render_chat(prompt)
    prompt_file.write_text(text, encoding="utf-8", newline="\n")
    if backend == "auto":
        backends = ["cpu"] if _GPU_OK is False else ["gpu", "cpu"]
    else:
        backends = [backend]
    if resident is None:
        resident = _RESIDENT
    if resident:
        return _run_resident(prompt_file, out_npz, log_path, gguf, on_line,
                             timeout, cancel_check, backends, text)
    return _run_oneshot(prompt_file, out_npz, log_path, gguf, on_line,
                        timeout, cancel_check, backends, text)


def _run_resident(prompt_file, out_npz, log_path, gguf, on_line, timeout,
                  cancel_check, backends, text):
    """Resident path: one long-lived worker per backend, rebuilt if it dies."""
    global _GPU_OK
    t_all = time.perf_counter()
    ap = pathlib.Path(log_path)
    logf = ap.open("a", encoding="utf-8", errors="replace")
    lines = []
    errors = []
    used = None
    ev = None
    boot_s = 0.0
    spawned_any = False
    reused = False
    cancelled = False
    try:
        for b in backends:
            prefix = f"[te:{b}] "

            def sink(s, _p=prefix):
                lines.append(s)
                logf.write(s + "\n")
                logf.flush()
                if on_line:
                    on_line(_p + s)

            got = False
            for attempt in (1, 2):
                try:
                    w, spawned = _get_worker(b, gguf, sink)
                except _Cancelled:
                    cancelled = True
                    errors.append(f"--- backend {b}: cancelled while starting ---")
                    break
                except (_WorkerDead, _WorkerTimeout, OSError, ValueError) as exc:
                    errors.append(f"--- backend {b} could not start ---\n{exc}")
                    break
                if spawned:
                    boot_s += w.boot_s
                    spawned_any = True
                    msg = (f"[textenc] resident worker {b} up in {w.boot_s:.2f} s "
                           f"(llama model load {w.load_s:.2f} s)")
                    lines.append(msg)
                    logf.write(msg + "\n")
                    logf.flush()
                    if on_line:
                        on_line(msg)
                else:
                    reused = True
                try:
                    ev = w.encode(prompt_file, out_npz, sink, timeout,
                                  cancel_check)
                except _Cancelled:
                    cancelled = True
                    errors.append(f"--- backend {b}: cancelled ---")
                    break
                except (_WorkerDead, _WorkerTimeout) as exc:
                    errors.append(f"--- backend {b} died on attempt {attempt}: {exc} ---\n"
                                  + "\n".join(lines[-60:]))
                    if attempt == 1:
                        continue        # crash self-heal: rebuild once
                    ev = None
                    break
                if ev.get("ok") and pathlib.Path(out_npz).is_file():
                    used = b
                    got = True
                    break
                errors.append(f"--- backend {b} encode failed ---\n"
                              + str(ev.get("error") or "")[-4000:])
                ev = None
                break
            if got:
                break
            if b == "gpu" and len(backends) > 1 and not cancelled:
                _GPU_OK = False
                msg = ("[textenc] GPU backend failed; falling back to CPU backend "
                       "(this server will use CPU for later jobs; restart to retry GPU)")
                print(msg, flush=True)
                logf.write(msg + "\n")
                logf.flush()
                if on_line:
                    on_line(msg)
                continue
            break
    finally:
        logf.close()
    if used is None:
        raise config.WebUIError(
            "text encoder failed",
            raw="\n\n".join(errors) or "\n".join(lines[-120:]),
            hint=_hint("\n".join(lines)))
    if used == "gpu":
        _GPU_OK = True
    res = {"rc": 0, "timed_out": False, "cancelled": False,
           "seconds": time.perf_counter() - t_all, "lines": lines[-5000:],
           "resident": True, "boot_s": round(boot_s, 3), "reused": reused,
           "spawned": spawned_any,
           "worker": {"load_s": (ev or {}).get("load_s"),
                      "ctx_s": (ev or {}).get("ctx_s"),
                      "served": (ev or {}).get("served")}}
    out = _parse(lines, res, gguf, text, used)
    return out


def _run_oneshot(prompt_file, out_npz, log_path, gguf, on_line, timeout,
                 cancel_check, backends, text):
    """Delivered path: one py312 process per encode (loads the GGUF again)."""
    global _GPU_OK
    t_all = time.perf_counter()
    errors = []
    res = None
    joined = ""
    used = None
    for b in backends:
        cmd = [config.PY312, config.TE_WORKER, "--gguf", str(gguf),
               "--prompt-file", str(prompt_file), "--out", str(out_npz),
               "--backend", b]
        ap = pathlib.Path(log_path)
        blog = ap.with_name(ap.stem + f"_{b}" + ap.suffix)
        cb = (lambda s, bb=b: on_line(f"[te:{bb}] {s}")) if on_line else None
        res = stream_process(cmd, env=config.subprocess_env("base"),
                             cwd=config.NPU, log_path=blog, on_line=cb,
                             timeout=timeout, cancel_check=cancel_check)
        joined = "\n".join(res["lines"][-250:])
        if res["rc"] == 0 and pathlib.Path(out_npz).is_file():
            used = b
            if b == "gpu":
                _GPU_OK = True
            break
        errors.append(f"--- backend {b} rc={res['rc']} ---\n{joined}")
        if res["timed_out"] or res["cancelled"]:
            break
        if b == "gpu" and len(backends) > 1:
            _GPU_OK = False
            print("[textenc] GPU backend failed; falling back to CPU backend "
                  "(this server will use CPU for later jobs; restart to retry GPU)",
                  flush=True)
            continue
        break
    if used is None:
        low = joined.lower()
        hint = ""
        if "erroroutofdevicememory" in low or "out of device memory" in low:
            hint = ("Both the Vulkan and CPU text-encoder backends failed. "
                    "Close GPU-heavy apps and retry; the CPU fallback should "
                    "not need the GPU.")
        elif "0xc0000005" in low or "access violation" in low:
            hint = "Text encoder process crashed; a fresh retry usually passes."
        raise config.WebUIError(
            "text encoder failed", raw="\n\n".join(errors), hint=hint)
    res = dict(res)
    res["seconds"] = time.perf_counter() - t_all
    res.update({"resident": False, "boot_s": 0.0, "reused": False,
                "spawned": True, "worker": {}})
    return _parse(res["lines"], res, gguf, text, used)


def load_pe(path):
    z = np.load(path)
    if "pe" not in z:
        raise ValueError(f"{path}: no 'pe' array")
    pe = np.asarray(z["pe"], np.float32)
    if pe.ndim != 3 or pe.shape[0] != 1 or pe.shape[2] != 2560:
        raise ValueError(f"{path}: pe shape {pe.shape}, expected (1,L,2560)")
    return pe


def count_tokens(pe):
    return int(np.asarray(pe).shape[1])
