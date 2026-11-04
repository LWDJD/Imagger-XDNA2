# SPDX-License-Identifier: MIT
"""Z-Image-Turbo on XDNA2 NPU -- production WebUI (Python stdlib only).

    py312 webui.py --port 7860

The HTTP server is py312 (schedule + VAE workers need diffusers/PIL).  The
device loop is always a fresh py310 child (numba + pyxrt), through
webui/bridge.py, which imports the delivered t13_run.py unchanged.

Features: positive prompt (real llama.cpp text encoder), fixed 32-token
caption, 128x128 (M=96) native profile, batch 1..16 (T16), steps 1..16 (8 default),
seed, VAE on/off, output dir/prefix, serial queue, live stage/progress, gallery,
per-image timings, and a double-run byte-identical latents gate.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import secrets
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse, unquote

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from webui import __version__, cache, config, shapes, textenc  # noqa: E402
from webui.runner import Runner  # noqa: E402

APP = None

# T22 qkv (campaign t22_qkv_prod): the fused q/k/v projection image the page
# may offer.  This is only metadata -- the device sees it through bridge.py,
# which owns the same default path.  Availability is reported so the page can
# say "not built" instead of letting a job fail at startup.
QKV_IMAGE_TAG = "qkv_par15m"
QKV_OUT = config.NPU / "imaggers" / "block" / "qkv" / "out"
QKV_XCLBIN = QKV_OUT / (QKV_IMAGE_TAG + ".xclbin")
QKV_BIN = QKV_OUT / (QKV_IMAGE_TAG + ".bin")

# T22 p15prod: the o projection's own image (ON by default).  Same contract as
# the qkv block above -- metadata only, the device sees it through bridge.py.
P10_IMAGE_TAG = "p10"
P10_OUT = config.NPU / "imaggers" / "attn_impl" / "proj" / "out"
P10_XCLBIN = P10_OUT / (P10_IMAGE_TAG + ".xclbin")
P10_BIN = P10_OUT / (P10_IMAGE_TAG + ".bin")


def _log(msg):
    """Console log for request handlers: a GBK console must never fail a request.

    The prebuild/prepare log lines carry the same Chinese diagnostics the API
    returns; on a cp936 console a plain print() raises UnicodeEncodeError, which
    would turn a 200 response into a 400.  Fall back to ASCII on failure.
    """
    try:
        print(msg)
    except Exception:  # noqa: BLE001
        try:
            sys.stdout.write(str(msg).encode("ascii", "replace").decode("ascii") + "\n")
        except Exception:  # noqa: BLE001
            pass


def _int_side(value):
    """Strict integer for a requested H/W (130.5 is not 130)."""
    if isinstance(value, bool):
        raise ValueError("height/width must be integers")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError("height/width must be integers")
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s.lstrip("+-").isdigit():
            raise ValueError("height/width must be integers")
        return int(s)
    raise ValueError("height/width must be integers")


def shapes_limits():
    # T22 pad_crop: the REQUEST is any integer in 64..2048, not a multiple of
    # 16.  "pad_to" is the internal grid; "multiple_of" is 1 so old clients
    # stop rejecting 130x130 in the browser.
    return {"height_min": shapes.MIN_HW, "height_max": shapes.MAX_HW,
            "width_min": shapes.MIN_HW, "width_max": shapes.MAX_HW,
            "m_min": shapes.MIN_M, "m_max": shapes.MAX_M,
            "multiple_of": 1, "pad_to": shapes.PAD_TO,
            "max_profiles": cache.MAX_PROFILES,
            "disk_budget_bytes": cache.DISK_BUDGET_BYTES}


# ---------------------------------------------------------- T22 seed_res
# seed_mode contract (shared with t22_ui_trim):
#   fixed      keep the integer the client sent (the pre-T22 behaviour)
#   increment  current batch behaviour: image b uses seed+b (DEFAULT)
#   random     the SERVER draws the integer with secrets.randbelow(2**31)
#              and writes it into params.seed, so a rerun can reproduce it
# An omitted field means "increment", which is byte-for-byte the delivered
# path for batch=1.  random ignores a client seed on a fresh POST (the seed is
# server-owned); a retry re-submits the resolved params.seed and therefore
# reproduces the job instead of drawing a new seed.
SEED_MODES = ("fixed", "increment", "random")
DEFAULT_SEED_MODE = "increment"


class SeedModeRunner(Runner):
    """Runner whose params carry seed_mode and whose random seed is server-made.

    The device path is untouched: webui/runner.py and webui/bridge.py stay
    frozen, so fixed/increment submit exactly today's seed integer and batch
    (batch image b = seed+b).  Only the API contract gains the mode field and
    the random draw.
    """

    def _normalize(self, raw):
        if isinstance(raw, dict):
            raw = dict(raw)
            mode = raw.get("seed_mode", DEFAULT_SEED_MODE)
            mode = DEFAULT_SEED_MODE if mode is None else str(mode).strip().lower()
            if not mode:
                mode = DEFAULT_SEED_MODE
            if mode not in SEED_MODES:
                raise config.WebUIError(
                    "seed_mode must be one of fixed/increment/random")
            raw["seed_mode"] = mode
            if mode == "random" and "seed" not in raw:
                try:
                    batch = int(raw.get("batch", 1))
                except Exception:  # noqa: BLE001 - runner will reject batch
                    batch = 1
                batch = max(1, min(batch, config.MAX_BATCH))
                # keep seed+b (b = 0..batch-1) inside 0..2^31-1, the range the
                # runner already validates and the existing job contract names.
                raw["seed"] = secrets.randbelow(2**31 - batch + 1)
        params = super()._normalize(raw)
        params["seed_mode"] = raw.get("seed_mode", DEFAULT_SEED_MODE)
        return params

    def public(self, job):
        out = super().public(job)
        if out and isinstance(out.get("params"), dict):
            # jobs written before T22 seed_res have no field; the API still
            # reports the contract default instead of an undefined value.
            out["params"].setdefault("seed_mode", DEFAULT_SEED_MODE)
        return out


def _fresh_seed_body(body):
    """A fresh random POST must let the server own the seed.

    Retry does NOT pass through the HTTP handler; it re-submits the stored
    params (which already contain the resolved seed), so only this path pops
    a client seed.
    """
    if not isinstance(body, dict):
        return body
    mode = body.get("seed_mode", DEFAULT_SEED_MODE)
    mode = DEFAULT_SEED_MODE if mode is None else str(mode).strip().lower()
    if mode == "random" and "seed" in body:
        body = dict(body)
        body.pop("seed", None)
    return body


# ---------------------------------------------------------- T21 2b-a: memo
# The front end polls /api/status every 5 s, and cache.status() +
# cache.shapes_status() re-hash every cached build product on every poll
# (60 files / 6.79 MB measured on this host, 52 ms of the 77 ms response).
# That value can only change when a cache file does, so it is memoized on a
# stat stamp: (path, mtime_ns, size) of every profile's manifest.json AND of
# every file the manifest lists, plus the set of profile directories.  A
# touched/moved/deleted cache file, a new/removed profile dir, or a rebuilt
# copy (new manifest) invalidates on the next call.  The one blind spot is a
# rewrite that preserves both mtime and size -- the same (path, mtime, size)
# contract build systems use; POST /api/cache/prepare and clean both land in
# the invalidating set.
# The cached value is shared, not copied: every caller only json.dumps() it.
_STATUS_MEMO = {}
_STATUS_MEMO_LOCK = threading.Lock()
_STATUS_MEMO_STATS = {"hits": 0, "misses": 0}


def _cache_stamp():
    """Stat stamp over the whole build-product cache (no file is hashed)."""
    stamp = []
    try:
        dirs = sorted((p for p in config.CACHE_DIR.iterdir() if p.is_dir()),
                      key=lambda p: p.name)
    except OSError:
        return (("<no cache dir>",),)
    for d in dirs:
        man = d / "manifest.json"
        try:
            st = man.stat()
        except OSError:
            stamp.append((str(d), "manifest-missing"))
            continue
        stamp.append((str(d), st.st_mtime_ns, st.st_size))
        try:
            files = json.loads(man.read_text(encoding="utf-8")).get("files") or []
        except Exception:  # noqa: BLE001 - unreadable manifest -> let status() speak
            stamp.append((str(man), "manifest-unreadable"))
            continue
        for rec in files:
            p = d / str(rec.get("name"))
            try:
                s = p.stat()
            except OSError:
                stamp.append((str(p), None, None))
            else:
                stamp.append((str(p), s.st_mtime_ns, s.st_size))
    return tuple(stamp)


def _status_memo(key, stamp, value_fn):
    with _STATUS_MEMO_LOCK:
        rec = _STATUS_MEMO.get(key)
        if rec is not None and rec[0] == stamp:
            _STATUS_MEMO_STATS["hits"] += 1
            return rec[1]
    value = value_fn()
    with _STATUS_MEMO_LOCK:
        _STATUS_MEMO[key] = (stamp, value)
        _STATUS_MEMO_STATS["misses"] += 1
    return value


def _status_memo_stats(stamp_len=0):
    with _STATUS_MEMO_LOCK:
        return {"hits": _STATUS_MEMO_STATS["hits"],
                "misses": _STATUS_MEMO_STATS["misses"],
                "stamp_entries": stamp_len,
                "te_gguf": config.find_te_gguf_stats()}


class Handler(BaseHTTPRequestHandler):
    server_version = "ZImageWebUI/" + __version__
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("[http] %s - %s\n" % (self.address_string(), fmt % args))

    # ---------------------------------------------------------------- helpers
    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False, indent=1))

    def _err(self, code, message, raw="", hint="", detail=None):
        out = {"ok": False, "error": message, "raw": raw[-8000:],
               "hint": hint}
        if detail:
            # T22 pad_crop: a rejected padded request still echoes its
            # requested/padded pair so the client can show both numbers.
            out.update(detail)
        self._json(out, code)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 2 * 1024 * 1024:
            raise ValueError("request body too large")
        if n <= 0:
            return {}
        raw = self.rfile.read(n)
        return json.loads(raw.decode("utf-8"))

    def _job(self, parts):
        if len(parts) < 3 or parts[1] != "jobs":
            return None
        return APP.runner.get(parts[2])

    # -------------------------------------------------------------------- GET
    def do_GET(self):
        try:
            u = urlparse(self.path)
            path = unquote(u.path)
            parts = [p for p in path.split("/") if p]

            if path in ("/", "/index.html"):
                p = config.STATIC / "index.html"
                if not p.is_file():
                    return self._err(500, "static/index.html missing")
                return self._send(200, p.read_bytes(), "text/html; charset=utf-8")
            if path == "/favicon.ico":
                return self._send(204, b"", "image/x-icon")

            if path == "/api/status":
                return self._json(self._status())
            if path == "/api/presets":
                # Contract: a bare JSON array (3 groups x 4 tiers).  It is not
                # wrapped in {"ok": true} so t22_ui_trim can consume it as-is.
                return self._json(shapes.preset_list())
            if path == "/api/jobs":
                return self._json({"ok": True, "jobs": APP.runner.list()})
            if path == "/api/cache":
                return self._json({"ok": True, "resolutions": cache.status(),
                                   "shapes": cache.shapes_status()})
            if path == "/api/shapes":
                return self._json({"ok": True, "shapes": cache.shapes_status(),
                                   "presets": shapes.PRESETS,
                                   "limits": shapes_limits()})
            if path == "/api/capability":
                # ?memory=1 -> the measured host-memory budget, so the page
                # can grey the button out instead of starting a 20-minute
                # death (rq_1080p_8d/8e/8f).
                if "memory" in u.query or (parse_qs(u.query).get(
                        "table", ["0"])[0] or "0") not in ("0", "", "false"):
                    if "memory" in u.query:
                        return self._json({"ok": True,
                                           "memory": config.memory_table()})
                # The honest answer: what this size can actually do TODAY,
                # derived from shapes.resolve/chunk_gate/plan_manifest + the
                # recorded jobs (webui/capability.py).  ?height&width -> one
                # shape; ?table=1 -> every size the product can speak about.
                from webui import capability
                q = parse_qs(u.query)
                if q.get("height") or q.get("width"):
                    h = _int_side(q.get("height", q.get("width"))[0])
                    w = _int_side(q.get("width", q.get("height"))[0])
                    return self._json({"ok": True,
                                       "shape": capability.describe(h, w)})
                if (q.get("table", ["0"])[0] or "0") not in ("0", "", "false"):
                    return self._json(capability.table())
                return self._json({"ok": True, "envelope": capability.envelope()})

            if len(parts) >= 3 and parts[0] == "api" and parts[1] == "jobs":
                jid = parts[2]
                if len(parts) == 3:
                    job = APP.runner.get(jid)
                    if not job:
                        return self._err(404, "no such job")
                    return self._json({"ok": True, "job": job})
                if len(parts) == 4 and parts[3] == "log":
                    log = config.JOBS_DIR / jid / "job.log"
                    if not log.is_file():
                        return self._err(404, "no log for this job")
                    return self._send(200, log.read_bytes(), "text/plain; charset=utf-8")
                if len(parts) == 5 and parts[3] == "images":
                    job = APP.runner.get(jid)
                    if not job:
                        return self._err(404, "no such job")
                    try:
                        idx = int(parts[4])
                    except ValueError:
                        return self._err(400, "image index must be an integer")
                    img = next((i for i in job.get("images", []) if i.get("index") == idx), None)
                    if not img or not img.get("path"):
                        return self._err(404, "image not found (VAE may be off)")
                    p = pathlib.Path(img["path"]).resolve()
                    if not p.is_file() or not str(p).lower().endswith(".png"):
                        return self._err(404, "image file missing")
                    return self._send(200, p.read_bytes(), "image/png")
            return self._err(404, "not found", raw=path)

        except Exception as exc:
            return self._err(500, f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------- POST
    def do_POST(self):
        try:
            u = urlparse(self.path)
            parts = [p for p in u.path.split("/") if p]
            body = self._body()

            if parts == ["api", "jobs"]:
                job = APP.runner.submit(_fresh_seed_body(body))
                return self._json({"ok": True, "job": job}, 202)
            if len(parts) == 4 and parts[:2] == ["api", "jobs"]:
                jid, action = parts[2], parts[3]
                if action == "cancel":
                    ok, msg = APP.runner.cancel(jid)
                    return self._json({"ok": ok, "message": msg}, 200 if ok else 409)
                if action == "retry":
                    try:
                        job = APP.runner.retry(jid)
                    except config.WebUIError as exc:
                        return self._err(400, str(exc), exc.raw, exc.hint)
                    return self._json({"ok": True, "job": job}, 202)
            if parts == ["api", "shapes", "prebuild"]:
                from webui import prebuild
                want_build = bool(body.get("build", False))
                if body.get("presets"):
                    res = prebuild.prebuild_presets(_log, build=want_build)
                    echo = {}
                else:
                    h = _int_side(body.get("height", body.get("resolution", 128)))
                    w = _int_side(body.get("width", h))
                    plan = shapes.resolve(h, w)
                    rh, rw = int(plan["height"]), int(plan["width"])
                    # A real build always targets the padded run shape; the
                    # requested size is only a view over it.  A requested
                    # 130x130 must never create a cache directory named 130x130.
                    res = [prebuild.prebuild(rh, rw, _log, build=want_build)]
                    echo = {"requested": {"height": h, "width": w},
                            "padded": {"height": rh, "width": rw},
                            "output": {"height": h, "width": w},
                            "pad_applied": bool(plan.get("pad_applied")),
                            "crop_applied": bool(plan.get("crop_applied")),
                            "requested_key": plan.get("requested_key"),
                            "run_key": plan.get("run_key")}
                    if not plan.get("ready"):
                        # Speak about the REQUEST (incl. the pad + crop rule),
                        # not only about the padded run shape.
                        res[0]["reason"] = plan.get("reason") or res[0].get("reason", "")
                        res[0]["error"] = res[0]["reason"]
                if echo:
                    res = [dict(x, **echo) for x in res]
                return self._json({"ok": all(x.get("ok") for x in res),
                                   "results": res})
            if parts == ["api", "cache", "prepare"]:
                h = _int_side(body.get("height", body.get("resolution", 128)))
                w = _int_side(body.get("width", h))
                plan = shapes.resolve(h, w)
                try:
                    info = cache.ensure_shape(int(plan["height"]),
                                              int(plan["width"]), print)
                except Exception as exc:
                    return self._err(400, f"{type(exc).__name__}: {exc}")
                return self._json({"ok": True, "cache": info,
                                   "requested": {"height": h, "width": w},
                                   "padded": {"height": int(plan["height"]),
                                              "width": int(plan["width"])},
                                   "output": {"height": h, "width": w}})
            if parts == ["api", "cache", "clean"]:
                if body.get("height") is not None or body.get("resolution") is not None:
                    h = _int_side(body.get("height", body.get("resolution")))
                    w = _int_side(body.get("width", h))
                    plan = shapes.resolve(h, w)
                    removed = cache.clean(int(plan["height"]), int(plan["width"]))
                else:
                    removed = cache.clean()
                return self._json({"ok": True, "removed": removed})
            return self._err(404, "not found", raw=u.path)
        except config.WebUIError as exc:
            return self._err(400, str(exc), exc.raw, exc.hint,
                             detail=getattr(exc, "detail", None))
        except json.JSONDecodeError as exc:
            return self._err(400, f"invalid JSON: {exc}")
        except ValueError as exc:
            return self._err(400, str(exc))
        except Exception as exc:
            return self._err(500, f"{type(exc).__name__}: {exc}")

    # ----------------------------------------------------------------- status
    def _status(self):
        te = config.find_te_gguf()
        mode, snap, why = config.choose_memory_mode("auto")
        stamp = _cache_stamp()
        resolutions = _status_memo("resolutions", stamp, cache.status)
        shape_rows = _status_memo("shapes", stamp, cache.shapes_status)
        return {
            "ok": True,
            "version": __version__,
            "python": sys.version.split()[0],
            "py312": str(config.PY312),
            "py310": str(config.PY310),
            "te": {"gguf": str(te) if te else None,
                   "ready": bool(te),
                   "bytes": te.stat().st_size if te else 0,
                   "resident": textenc.status(),
                   "resident_enabled": textenc._RESIDENT},
            "model_present": config.model_present(),
            "qkv": {"available": bool(QKV_XCLBIN.is_file() and
                                     QKV_BIN.is_file()),
                    "image": QKV_XCLBIN.name,
                    "default": bool(config.T22_QKV_DEFAULT)},
            "p10": {"available": bool(P10_XCLBIN.is_file() and
                                     P10_BIN.is_file()),
                    "image": P10_XCLBIN.name,
                    "default": bool(config.T22_P10_DEFAULT)},
            "memory": snap,
            "memory_mode": mode,
            "memory_reason": why,
            "defaults": dict(config.DEFAULT_PARAMS,
                             seed_mode=DEFAULT_SEED_MODE),
            "seed_modes": list(SEED_MODES),
            "limits": {
                "batch": config.MAX_BATCH,
                "steps": config.MAX_STEPS,
                "cap_tokens": config.CAP_TOKENS,
                "prompt_chars": config.MAX_PROMPT_CHARS,
                **shapes_limits(),
            },
            "resolutions": resolutions,
            "shapes": shape_rows,
            "status_cache": _status_memo_stats(len(stamp)),
            "scheduler": "Euler (FlowMatchEulerDiscreteScheduler) — fixed, read-only",
            "runner": APP.runner.status(),
            "repo": str(config.NPU),
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--open", action="store_true", help="open the browser")
    ap.add_argument("--clean-cache", action="store_true",
                    help="delete webui/cache and exit")
    ap.add_argument("--prepare-cache", action="store_true",
                    help="verify/copy the 128 build cache and exit")
    args = ap.parse_args()

    # ---- reap layer (rq_1080p_8d/8e/8f) -------------------------------
    # Every FAILED attempt left a py310 device worker alive (1.8 GB low mode,
    # 7.3 GB fast mode) because the driver only ever held THIS process's pid
    # and terminate() is TerminateProcess: no atexit, no finally, no cleanup
    # below this line ever runs.  The job object is installed BEFORE the first
    # child exists (membership is inherited at CreateProcess time), so the
    # bridge / pack-prewarm / VAE / TE workers die with this process even when
    # it is hard-killed or dies with 0xC0000005.  See webui/reap.py.
    try:
        from webui import reap as _reap
        _reap.install_kill_on_close()
        _reap.install_exit_hook()
    except Exception as _exc:                                # noqa: BLE001
        print("[webui] reap layer unavailable (%r); a hard kill can still "
              "leak a device worker" % (_exc,))

    if args.clean_cache:
        print("removed:", cache.clean())
        return 0
    if args.prepare_cache:
        info = cache.ensure(128, print)
        print(json.dumps({k: info[k] for k in ("resolution", "dir", "cached",
                                               "created", "seconds")}, indent=1))
        return 0

    global APP
    APP = type("App", (), {})()
    APP.runner = SeedModeRunner()
    if not (config.STATIC / "index.html").is_file():
        print(f"[webui] missing {config.STATIC / 'index.html'}")
        return 2
    if sys.version_info[:2] != (3, 12):
        print(f"[webui] WARNING: running under Python {sys.version.split()[0]}; "
              f"the schedule/VAE workers are invoked with py312 explicitly, but "
              f"this server was validated on 3.12.")
    te = config.find_te_gguf()
    print("=" * 78)
    print(f" Z-Image-Turbo NPU WebUI {__version__}")
    print(f" repo      : {config.NPU}")
    print(f" python    : {sys.version.split()[0]}")
    print(f" py310     : {config.PY310}")
    print(f" te GGUF   : {te if te else 'MISSING (set ZIMAGE_TE_GGUF; see README-WEBUI.md)'}")
    print(f" cache     : {config.CACHE_DIR}")
    print(f" outputs   : {config.DEFAULT_OUTPUT}")
    print(f" listen    : http://{args.host}:{args.port}")
    print("=" * 78, flush=True)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    if args.open:
        threading.Thread(target=lambda: webbrowser.open(
            f"http://{args.host}:{args.port}"), daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[webui] stopping")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
