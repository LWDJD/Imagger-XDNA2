# SPDX-License-Identifier: MIT
"""py310 resident device worker (T21 2a-b): keep the device context, the three
registered xclbins and the packed projection images across jobs.

Why this exists
---------------
The delivered bridge is a fresh py310 process per job.  Measured breakdown of
its startup (128x128, M=96, 2026-10-19): process+imports ~0.7 s, device open +
3 xclbin registrations ~1.5 s, --warm packs 30 layers x 4 projections
(2.0 GB, reading ~7 GB of bf16 weights) 22-37 s, --cache-all reads+permutes
3.71 GB 2-7 s, the first two refiner blocks 1.4 s.  Only the per-job PE,
schedule and latents are actually per-job; everything else is the same bytes
for every run of the same shape.

This worker pays that once, then serves jobs over stdin.  It reuses the SAME
bridge seams (token layout, cap_embedding, load_schedule, driver_argv) and the
SAME t18_run.run() body, so the arithmetic cannot drift from the delivered
path: the loop is the delivered loop, called with the delivered argv.

Protocol (one JSON object per line on stdin; @@WEBUI@@ JSON lines on stdout):
    -> {"op":"run","id":7,"tag":"w...","pe":"...npz","schedule":"...npz",
        "steps":8,"seed":7,"batch":1}
    <- {"event":"startup_done"} ... (the driver's own event stream)
    <- {"event":"job_done","id":7,"ok":true,"rc":0,"seconds":6.1}
    -> {"op":"ping"} / {"op":"shutdown"}

T22 C-1, the VAE link
---------------------
This process also owns the VAE launches, on the device context it already
holds.  Two device contexts in two processes is what killed this worker with
0xC0000005 while a second py310 process decoded (2a2b2, 2/2), and dropping the
context in here is a teardown crash, so the only shape left is one context
serving both.  A loopback socket is opened at startup (127.0.0.1, ephemeral
port, advertised as "vae_addr" in the ready event and optionally written to
--vae-link-file); the host side (t22_vae_srv.DevSrv in connect mode) speaks
the vaeg_srv_proto frames over it.  The stdin/stdout JSON protocol below is
untouched.  The VAE Srv is built lazily on the first connection and handed a
shim whose device() returns THIS process's pyxrt device, so there is exactly
one device handle.  One lock orders the DiT loop and the VAE launches: they
never overlap (they share the GIL-bound bo.write anyway, F-03).

Client (py312, imported by webui.runner): class Worker.
"""
from __future__ import annotations

import json
import os
import pathlib
import queue
import socket
import subprocess
import sys
import threading
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
NPU = HERE.parent
# This interpreter (GAIA py310) does not put the script's directory on
# sys.path[0], so make the webui package importable explicitly.
for _p in (str(HERE), str(NPU), str(NPU / "imaggers" / "image")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
MARK = "@@WEBUI@@ "
VAE_DIR = NPU / "imaggers" / "vae"


# ---------------------------------------------------------------------------
# client (py312 server side; stdlib only)
# ---------------------------------------------------------------------------
class WorkerDead(RuntimeError):
    def __init__(self, rc, lines):
        super().__init__(f"device worker died (rc={rc})")
        self.rc = rc
        self.lines = lines


class Worker:
    """One resident py310 device worker, driven over stdin/stdout."""

    def __init__(self, python, script, cache_dir, height, width, refiner_cache,
                 log_path=None, env=None, keep_refiner="auto",
                 keep_payload="auto", cwd=None, vae_link="1",
                 wpack_file="", wpack_verify=False, reserve_gb=0.0):
        self.python = str(python)
        self.script = str(script)
        self.cache_dir = str(cache_dir)
        self.height = int(height)
        self.width = int(width)
        self.refiner_cache = str(refiner_cache)
        self.log_path = pathlib.Path(log_path) if log_path else None
        self.env = env
        self.keep_refiner = keep_refiner
        self.keep_payload = keep_payload
        self.cwd = str(cwd) if cwd else str(NPU)
        self.vae_link = str(vae_link)
        self.wpack_file = str(wpack_file or "")
        self.wpack_verify = bool(wpack_verify)
        self.reserve_gb = float(reserve_gb or 0.0)
        self.proc = None
        self.q = queue.Queue()
        self.info = {}
        self.lines = []
        self._logf = None

    # -- lifecycle ---------------------------------------------------------
    def start(self, on_line=None, timeout=240.0):
        if self.proc is not None and self.proc.poll() is None:
            return self.info
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._logf = open(self.log_path, "w", encoding="utf-8",
                              errors="replace")
        cmd = [self.python, self.script, "--serve",
               "--cache-dir", self.cache_dir,
               "--height", str(self.height), "--width", str(self.width),
               "--refiner-cache", self.refiner_cache,
               "--keep-refiner", self.keep_refiner,
               "--keep-payload", self.keep_payload,
               "--vae-link", self.vae_link,
               "--reserve-gb", "%.2f" % self.reserve_gb]
        if self.wpack_file:
            cmd += ["--wpack-file", self.wpack_file]
            if self.wpack_verify:
                cmd.append("--wpack-verify")
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", bufsize=1, cwd=self.cwd, env=self.env,
            creationflags=CREATE_NO_WINDOW)
        self.q = queue.Queue()
        self.lines = []
        threading.Thread(target=self._reader, daemon=True).start()
        t0 = time.perf_counter()
        while True:
            left = timeout - (time.perf_counter() - t0)
            if left <= 0:
                self.kill()
                raise WorkerDead(-9, self.lines[-200:])
            try:
                line = self.q.get(timeout=min(0.25, left))
            except queue.Empty:
                if self.proc.poll() is not None and self.q.empty():
                    raise WorkerDead(self.proc.returncode, self.lines[-200:])
                continue
            if line is None:
                raise WorkerDead(self.proc.returncode, self.lines[-200:])
            self._keep(line)
            if on_line:
                on_line(line)
            ev = self._event(line)
            if ev is not None and ev.get("event") == "ready":
                if not ev.get("ok"):
                    self.kill()
                    raise WorkerDead(self.proc.returncode or 2,
                                     self.lines[-200:])
                self.info = ev
                return ev

    def _reader(self):
        try:
            for line in self.proc.stdout:
                self.q.put(line.rstrip("\r\n"))
        finally:
            self.q.put(None)

    def _keep(self, line):
        self.lines.append(line)
        if len(self.lines) > 4000:
            self.lines = self.lines[-2000:]
        if self._logf:
            self._logf.write(line + "\n")
            self._logf.flush()

    @staticmethod
    def _event(line):
        if not line.startswith(MARK):
            return None
        try:
            return json.loads(line[len(MARK):])
        except Exception:                                     # noqa: BLE001
            return None

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def kill(self):
        p = self.proc
        if p is None:
            return
        try:
            if p.stdin:
                p.stdin.close()
        except Exception:                                     # noqa: BLE001
            pass
        if p.poll() is None:
            # The resident worker is a py310 python that can own a VAE
            # child; terminate() alone leaves them on the NPU.  reap()
            # walks the tree, kills children first and waits.  This is
            # the path a MemoryError in the fast pass takes, i.e. the
            # 7.3 GB worker that used to survive it.
            try:
                from . import reap as _reap
                _reap.reap(p, why="resident dev worker")
            except Exception:                             # noqa: BLE001
                try:
                    p.terminate()
                except Exception:                         # noqa: BLE001
                    pass
                try:
                    p.wait(timeout=15)
                except Exception:                         # noqa: BLE001
                    try:
                        p.kill()
                    except Exception:                     # noqa: BLE001
                        pass
        self.proc = None
        if self._logf:
            self._logf.close()
            self._logf = None

    def resident_bytes(self):
        return int(self.info.get("resident_bytes") or 0)

    @property
    def vae_addr(self):
        """The worker's VAE link address ("127.0.0.1:port"), or ''."""
        return str(self.info.get("vae_addr") or "") if self.alive() else ""

    def release(self, timeout=90.0):
        """Drop the device context in the worker (keeps the packed W images)."""
        if not self.alive():
            return False
        try:
            self.proc.stdin.write(json.dumps({"op": "release"}) + "\n")
            self.proc.stdin.flush()
        except Exception:                                     # noqa: BLE001
            return False
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            try:
                line = self.q.get(timeout=0.25)
            except queue.Empty:
                if not self.alive():
                    return False
                continue
            if line is None:
                return False
            self._keep(line)
            ev = self._event(line)
            if ev is not None:
                if ev.get("event") == "released":
                    return True
                if ev.get("event") == "error":
                    return False
        return False

    # -- one job -----------------------------------------------------------
    def run_job(self, req, on_line=None, timeout=900.0, cancel_check=None):
        """Send one job; return a stream_process-shaped result dict."""
        if not self.alive():
            raise WorkerDead(self.proc.returncode if self.proc else -9,
                             self.lines[-200:])
        t0 = time.perf_counter()
        lines0 = len(self.lines)
        try:
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
        except Exception:                                     # noqa: BLE001
            raise WorkerDead(self.proc.returncode, self.lines[-200:])
        rid = req.get("id")
        while True:
            if cancel_check and cancel_check():
                self.kill()
                return {"rc": -2, "timed_out": False, "cancelled": True,
                        "seconds": time.perf_counter() - t0,
                        "lines": self.lines[lines0:]}
            if time.perf_counter() - t0 > timeout:
                self.kill()
                return {"rc": -3, "timed_out": True, "cancelled": False,
                        "seconds": time.perf_counter() - t0,
                        "lines": self.lines[lines0:]}
            try:
                line = self.q.get(timeout=0.25)
            except queue.Empty:
                if self.proc.poll() is not None and self.q.empty():
                    return {"rc": self.proc.returncode, "timed_out": False,
                            "cancelled": False,
                            "seconds": time.perf_counter() - t0,
                            "lines": self.lines[lines0:], "dead": True}
                continue
            if line is None:
                return {"rc": self.proc.returncode, "timed_out": False,
                        "cancelled": False,
                        "seconds": time.perf_counter() - t0,
                        "lines": self.lines[lines0:], "dead": True}
            self._keep(line)
            if on_line:
                on_line(line)
            ev = self._event(line)
            if ev is not None and ev.get("event") == "job_done" \
                    and ev.get("id") == rid:
                return {"rc": int(ev.get("rc") or 0),
                        "timed_out": False, "cancelled": False,
                        "seconds": time.perf_counter() - t0,
                        "lines": self.lines[lines0:],
                        "ok": bool(ev.get("ok")), "worker_error": ev.get("error")}


# ---------------------------------------------------------------------------
# T22 C-1: the VAE link (one device context for DiT and VAE)
# ---------------------------------------------------------------------------
# 2a2b2 measured the two-context shape as fatal: a resident worker (device
# open) plus a second py310 process decoding the VAE killed this process with
# 0xC0000005, 2/2; a 2 GB host-only process survived the same decode (so it is
# the device context, not memory), and dropping the context in here is itself a
# teardown crash.  C-1 is the only shape left: the VAE dispatches on the
# context THIS process already holds.
#
# The host (t22_vae_srv.DevSrv "connect" mode) speaks the vaeg_srv_proto
# frames over a loopback socket instead of a pipe.  This listener is a daemon
# thread; it never writes to stdout (the @@WEBUI@@ line protocol stays clean)
# and it never exits the process: a failed launch is a frame plus a closed
# connection, and the host then raises (vae_local falls back to the CPU
# decoder; the runner's liveness check decides the worker's fate).
#
# Laws obeyed here:
#   * one pyxrt device handle: the Srv is handed a shim whose device() returns
#     the DiT device object -- no second context, no second xclbin handle on
#     any image (the VAE images are a disjoint set);
#   * one launch per request, each waited (Srv._launch -> run.wait);
#   * DiT loop and VAE launches never overlap: one lock orders them (they are
#     GIL-bound anyway, F-03);
#   * the VAE Srv is built lazily, so startup pays only a bind() and a thread.


class _OneDevice:
    """pyxrt proxy: Srv.device(0) returns the context the DiT loop owns."""

    def __init__(self, mod, dev):
        self._mod = mod
        self._dev = dev

    def device(self, index=0):
        return self._dev

    def __getattr__(self, name):
        return getattr(self._mod, name)


class VaeLink:
    """Loopback server that runs VAE launches in the worker's device context."""

    def __init__(self, cur, lock, boot_s=0.0, emit=None):
        self.cur = cur
        self.lock = lock
        self.boot_s = float(boot_s)
        self.emit = emit
        self.srv = None
        self.stats = dict(conns=0, requests=0, launches=0, errors=0,
                          last_err="")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(4)
        self.addr = "127.0.0.1:%d" % self.sock.getsockname()[1]
        threading.Thread(target=self._accept, name="vae-link",
                         daemon=True).start()

    # -- the device side ---------------------------------------------------
    def _srv(self):
        if self.srv is not None:
            return self.srv
        dev = self.cur.get("dev")
        if dev is None:
            raise RuntimeError("the DiT device context is not open")
        px = getattr(dev, "pyxrt", None)
        target = getattr(dev, "dev", None)
        if px is None or target is None:
            raise RuntimeError("no pyxrt device on the DiT Dev object")
        if str(VAE_DIR) not in sys.path:
            sys.path.insert(0, str(VAE_DIR))
        import vaeg_devsrv
        self.srv = vaeg_devsrv.Srv(_OneDevice(px, target))
        return self.srv

    # -- one connection ----------------------------------------------------
    def _accept(self):
        while True:
            try:
                conn, _peer = self.sock.accept()
            except OSError:
                # never let one accept error kill the listener: the host would
                # then fall back to spawning its own device server next to
                # this process, which is the shape C-1 exists to remove
                time.sleep(0.2)
                continue
            self.stats["conns"] += 1
            try:
                self._conn(conn)
            except BaseException:                              # noqa: BLE001
                self.stats["errors"] += 1
                self.stats["last_err"] = traceback.format_exc()[-1500:]
            finally:
                try:
                    conn.close()
                except Exception:                              # noqa: BLE001
                    pass

    def _conn(self, conn):
        if str(VAE_DIR) not in sys.path:
            sys.path.insert(0, str(VAE_DIR))
        import vaeg_srv_proto as PROTO
        conn.settimeout(None)
        rd = PROTO.reader(conn.makefile("rb", 0))
        wr = PROTO.writer(conn.makefile("wb", 0))
        PROTO.send_msg(wr, [json.dumps(
            {"ok": 1, "kind": "dev_worker", "pid": os.getpid(),
             "boot_s": round(self.boot_s, 3), "addr": self.addr}).encode()])
        while True:
            parts = PROTO.recv_msg(rd)
            if parts is None:
                return
            head = json.loads(parts[0])
            cmd = head.get("cmd")
            if cmd == "bye":
                PROTO.send_msg(wr, [b'{"ok":1}'])
                return
            if cmd == "ping":
                PROTO.send_msg(wr, [json.dumps(
                    dict(ok=1, kind="dev_worker", stats=self.stats)).encode()])
                continue
            if cmd == "init":
                with self.lock:
                    self._srv().set_barrier(head.get("barrier"))
                PROTO.send_msg(wr, [json.dumps(
                    dict(ok=1, kind="dev_worker",
                         barrier=bool(head.get("barrier")))).encode()])
                continue
            if cmd == "run":
                self.stats["requests"] += 1
                try:
                    with self.lock:
                        srv = self._srv()
                        n0 = len(srv.imgs)
                        self._note("vae_launch", stem=head.get("stem"),
                                   seq=head.get("seq"), images=n0)
                        hdr, raw = srv.run(head, parts[1:])
                        self.stats["launches"] = int(
                            srv.stats.get("launches") or 0)
                        n1 = len(srv.imgs)
                        if n1 != n0:
                            self._note("vae_reg", images=n1,
                                       last=head.get("stem"))
                except BaseException:                          # noqa: BLE001
                    self.stats["errors"] += 1
                    self.stats["last_err"] = traceback.format_exc()[-1500:]
                    self._send_error(wr, head)
                    return
                PROTO.send_msg(wr, [json.dumps(hdr).encode(), raw or b""])
                if not hdr.get("ok"):
                    return
                continue
            PROTO.send_msg(wr, [json.dumps(
                dict(ok=0, err="unknown cmd %r" % (cmd,))).encode()])
            return

    def _note(self, event, **kw):
        """A MARK line in the worker log (never on the wire, never in a job).

        The VAE link runs only while the DiT loop is idle (one lock), so this
        cannot interleave with a running job's event stream.  It is the only
        telemetry that survives a 0xC0000005: the last line says which image
        was being registered / which launch was in flight.
        """
        if self.emit is None:
            return
        try:
            self.emit(dict(event=event, **kw))
        except BaseException:                                  # noqa: BLE001
            pass

    @staticmethod
    def _send_error(wr, head):
        try:
            exc = sys.exc_info()[1]
            import vaeg_srv_proto as PROTO
            PROTO.send_msg(wr, [json.dumps(dict(
                ok=0, fatal=1, seq=head.get("seq"), stem=head.get("stem"),
                err="%s: %s" % (type(exc).__name__, exc),
                tb=traceback.format_exc()[-2500:])).encode()])
        except Exception:                                      # noqa: BLE001
            pass


def _serve(args) -> int:
    import numpy as np
    import bridge
    import t3_real as T3
    import t18_run as T

    link = None
    dev_lock = threading.Lock()

    def emit(obj):
        sys.stdout.write(MARK + json.dumps(obj) + "\n")
        sys.stdout.flush()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cache_dir = pathlib.Path(args.cache_dir).resolve()
    man = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    imgs = man.get("images") or {"proj": "t7_p5", "ip": "t7_ip",
                                 "ffn": "fs_ffn_j5"}
    if (int(args.height), int(args.width)) != (128, 128):
        emit({"event": "ready", "ok": False,
              "error": "device worker currently serves 128x128 (M=96) only"})
        return 2
    try:
        DRV = bridge._load_driver(args.height, args.width, args.driver)
        DRV.T4_OUT = cache_dir
        DRV.OUT_A = cache_dir
        bridge._install_token_layout_fix()
        shape_info = bridge._install_shape_geometry(
            args.height, args.width, "t21warm", 8, 0, 1)
        if shape_info is not None:
            emit({"event": "ready", "ok": False,
                  "error": "shape geometry is not supported by the worker"})
            return 2

        # -- the driver's own module globals, set once --------------------
        T.FAST_HOST = True
        T.H.install()
        T.SCRATCH = T.H.Scratch()
        T.NH_BG = 5
        T.NH_BG_CORE = True
        T.CACHE_ALL = True
        T.REFCACHE = args.refiner_cache or None
        T.ASSERT_SBG = True

        t_start = time.perf_counter()
        # T22 qkvprod: the warm Dev is built ONCE here and reused by every job,
        # so the fused q/k/v decision is made HERE, not at job parse time.  A
        # job that turns --qkv on against a Dev built without the image dies
        # with KeyError 'q_x' (measured 2026-10-20 15:19).  Use the same
        # default the bridge CLI uses and the same default image paths; a job's
        # --no-qkv just leaves the registered image unused.
        T.QKV = bool(getattr(bridge, "QKV_ENV", False))
        if T.QKV:
            T.QKV_XCLBIN = str(bridge.QKV_DEFAULT_OUT /
                               (bridge.QKV_DEFAULT_TAG + ".xclbin"))
            T.QKV_BIN = str(bridge.QKV_DEFAULT_OUT /
                            (bridge.QKV_DEFAULT_TAG + ".bin"))
        # T22 p15prod: the same trap applies to the o image.  The warm Dev is
        # the ONLY place that can register it, so the decision is made here too,
        # from the same default the bridge CLI uses (ON since 2026-10-20).  A
        # job's --no-p10 just leaves the registered image unused; a job that
        # asks for an image this Dev did not register would die with
        # KeyError 'o_x'.
        T.P10 = bool(getattr(bridge, "P10_ENV", False))
        if T.P10:
            T.P10_XCLBIN = str(bridge.P10_DEFAULT_OUT /
                               (bridge.P10_DEFAULT_TAG + ".xclbin"))
            T.P10_BIN = str(bridge.P10_DEFAULT_OUT /
                            (bridge.P10_DEFAULT_TAG + ".bin"))
        CUR = {"dev": None, "cls": DRV.Dev}
        CUR["dev"] = CUR["cls"]("t21warm", dimk64=True,
                                ffn_image=imgs["ffn"],
                                att_image=imgs["proj"],
                                att_dir=str(cache_dir),
                                att_ip_image=imgs["ip"])
        emit({"event": "warm", "stage": "device",
              "s": round(time.perf_counter() - t_start, 2)})

        # -- the packed projection images (the 22-37 s item) --------------
        # T22 prewarm: a sibling py310 process can build the 120 projection
        # images once (bridge.py --pack-prewarm).  This worker then MAPS that
        # artifact instead of packing -- the same seam the delivered bridge
        # uses for --wpack-file, so the dispatch payload is the same bytes by
        # construction.  An artifact that does not check out is one log line
        # and the delivered in-process pack, never a guess.
        t0 = time.perf_counter()
        wpack_src = "pack"
        if getattr(args, "wpack_file", ""):
            try:
                bridge._load_wpack_file(args.wpack_file, T, layers=30,
                                        verify=bool(args.wpack_verify))
                wpack_src = "file"
            except Exception as exc:                          # noqa: BLE001
                emit({"event": "warm", "stage": "wpack_file_ignored",
                      "why": ("%s: %s" % (type(exc).__name__, exc))[-400:]})
        if wpack_src == "pack":
            T.WPACK = np.empty(120 * T.W_BYTES, np.uint8)
            T.WPACK_SLOTS.clear()
            T.WPACK_SECONDS[0] = 0.0
            T.WPACK_SECONDS[1] = 0
        L = T.Loop(CUR["dev"], "t21warm", dimk64=True)
        if wpack_src == "pack":
            for li in range(30):
                L.w_images("layers.%d" % li)
                L.params("layers.%d" % li)
        t_wpack = time.perf_counter() - t0
        emit({"event": "warm", "stage": "wpack", "src": wpack_src,
              "s": round(t_wpack, 2)})

        # -- the payload (fused read+permute), optionally resident --------
        import types
        from webui import config as WCFG
        threshold = WCFG.MEMORY_FAST_THRESHOLD_GB
        snap = WCFG.memory_snapshot()

        # T22 workervae: how much commit headroom must stay free AFTER this
        # residency, besides the runner's own fast threshold.  The text-encoder
        # backend (GPU) loads in the host server after this worker warms, and
        # its footprint is not visible in the snapshot taken here; without the
        # reserve the NEXT job's memory decision sees 7.3 GB and flips to low
        # mode, which drops this worker (measured 2026-10-20 13:33).
        reserve_gb = float(getattr(args, "reserve_gb", 0.0) or 0.0)

        def _keep(flag, extra_gb, snap_):
            if flag == "1":
                return True
            if flag == "0":
                return False
            # Resident bytes are committed ONCE and reused by every job; the
            # runner's memory decision must still find the fast threshold
            # after they are charged.  (The W images are already charged in
            # snap; extra_gb is what THIS residency would add.)
            return (snap_["commit_gb"] - extra_gb - reserve_gb) >= threshold

        payload_gb = 3.71
        keep_payload = _keep(args.keep_payload, payload_gb, snap)
        t_payload = 0.0
        if keep_payload:
            t0 = time.perf_counter()
            T._PAYLOAD_PRE["err"] = None
            T._preload_payload(types.SimpleNamespace(cache_all=True,
                                                     bo_resident=False,
                                                     layers=30), at="cache")
            if T._PAYLOAD_PRE["err"] is not None:
                raise T._PAYLOAD_PRE["err"]
            t_payload = time.perf_counter() - t0
        emit({"event": "warm", "stage": "payload", "s": round(t_payload, 2)})

        snap2 = WCFG.memory_snapshot()
        keep_refiner = _keep(args.keep_refiner, 2.67, snap2)
        keep_refiner_resident = bool(keep_refiner)
        t_ref = 0.0
        if keep_refiner:
            t0 = time.perf_counter()
            for key in T3.REFINERS:
                L.refblock(key)
            t_ref = time.perf_counter() - t0
        else:
            T.Loop._rcache.clear()
        keep_payload_resident = bool(keep_payload)
        emit({"event": "warm", "stage": "refiner", "s": round(t_ref, 2)})

        # -- reuse patches ------------------------------------------------
        if wpack_src == "pack":
            # the delivered pack arm: every slot is already in this process,
            # so remember it and skip the per-job read+re-encode
            wc = {}
            orig_w = T.Loop.w_images

            def w_images_reuse(self, key):
                if key in wc:
                    return wc[key]
                out = orig_w(self, key)
                wc[key] = out
                return out

            wc.update(L.wcache)
            T.Loop.w_images = w_images_reuse
        # else: bridge._load_wpack_file already installed its own cache-hit
        # patch over Loop.w_images (the same seam, the same bytes)
        del L                        # the prewarm Loop holds the device; drop it

        def dev_factory(*a, **k):
            # T21 2a-b: the device is opened per job and RELEASED before the
            # VAE stage (the py312 VAE chain dispatches on the same NPU; a
            # device context held while another process dispatches killed this
            # process with 0xC0000005, measured 2026-10-19 22:5x).  The packed
            # W images and the payload stay in host RAM across the release.
            if CUR["dev"] is None:
                CUR["dev"] = CUR["cls"](*a, **k)
            return CUR["dev"]

        T.Dev = dev_factory
        host_memo = {}
        orig_ht = T3.host_tensors

        def host_tensors_once(tag="t3real"):
            if "v" not in host_memo:
                host_memo["v"] = orig_ht("t21warm")
            return host_memo["v"]

        T3.host_tensors = host_tensors_once

        # -- T22 C-1: the VAE link ----------------------------------------
        # Bind it AFTER the warm path so a cold start pays nothing but a
        # bind() + one daemon thread; the VAE Srv itself is built on the first
        # connection (lazy), so the 4-7 s hot-job startup is untouched.
        if str(getattr(args, "vae_link", "1")) != "0":
            try:
                link = VaeLink(CUR, dev_lock, emit=emit,
                               boot_s=time.perf_counter() - t_start)
                lf = str(getattr(args, "vae_link_file", "") or "")
                if lf:
                    pathlib.Path(lf).write_text(json.dumps(
                        {"pid": os.getpid(), "addr": link.addr,
                         "ts": time.time()}), encoding="utf-8")
            except BaseException:
                link = None
                emit({"event": "vae_link_failed",
                      "error": traceback.format_exc()[-2000:]})
        # a mapped artifact is file-backed, NOT private commit: report what
        # this process actually charges (the residency policy is about commit)
        wpack_bytes = 0 if wpack_src == "file" else int(T.WPACK.nbytes)
        resident_bytes = int(
            wpack_bytes
            + (T.CACHE_BG.nbytes + T.CACHE_B3.nbytes if keep_payload else 0)
            + (2.67 * 2**30 if keep_refiner else 0))
        bridge.emit("worker_ready", resident_gb=round(resident_bytes / 2**30, 2))
        emit({"event": "ready", "ok": True, "pid": os.getpid(),
              "vae_addr": link.addr if link is not None else "",
              "wpack_s": round(t_wpack, 2), "wpack_src": wpack_src,
              "payload_s": round(t_payload, 2),
              "refiner_s": round(t_ref, 2),
              "keep_payload": bool(keep_payload),
              "keep_refiner": bool(keep_refiner),
              "reserve_gb": reserve_gb,
              "mem": snap, "threshold_gb": threshold,
              "resident_bytes": resident_bytes})
    except BaseException:
        emit({"event": "ready", "ok": False,
              "error": traceback.format_exc()[-4000:]})
        return 2

    served = 0
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except Exception:                                     # noqa: BLE001
            emit({"event": "error", "ok": False, "error": "unparsable request"})
            continue
        op = req.get("op")
        if op == "release":
            # Drop the device context (BOs, registrations) but keep the packed
            # W images and the payload in host RAM.
            # T22 C-1: refuse while the VAE link owns a Srv on that context --
            # tearing it down under live VAE BOs is the teardown crash class.
            if link is not None and link.srv is not None:
                emit({"event": "released", "ok": False, "pid": os.getpid(),
                      "why": "the VAE link holds a Srv on this context"})
                continue
            with dev_lock:
                CUR["dev"] = None
            import gc
            gc.collect()
            emit({"event": "released", "pid": os.getpid(),
                  "wpack_kept": bool(T.WPACK is not None)})
            continue
        if op == "ping":
            emit({"event": "pong", "served": served})
            continue
        if op == "shutdown":
            emit({"event": "bye", "served": served})
            break
        if op != "run":
            emit({"event": "error", "ok": False, "error": f"unknown op {op!r}"})
            continue
        rid = req.get("id")
        t_job = time.perf_counter()
        try:
            import numpy as np
            z = np.load(req["pe"])
            pe = np.asarray(z["pe"], np.float32)
            if pe.shape[1] > 32:
                raise SystemExit("prompt is %d tokens > cap_tokens=32"
                                 % pe.shape[1])
            s = np.load(req["schedule"])
            ts = np.asarray(s["timesteps"], np.float64)
            sg = np.asarray(s["sigmas"], np.float64)
            # T22 qkv (campaign t22_qkv_prod): the SAME two doors the bridge
            # exposes, so a worker-served job and a fresh-bridge job cannot
            # disagree.  A request may name them; otherwise the worker's own
            # environment decides (default off).  driver_argv() reads them
            # with getattr(), so a request from a runner that does not know
            # about them still runs the delivered path.  The fused image is
            # registered per job by t18_run.Dev itself when --qkv is on (the
            # device context is rebuilt after every release).
            ns = types.SimpleNamespace(
                tag=req["tag"], steps=int(req["steps"]), seed=int(req["seed"]),
                batch=int(req.get("batch", 1)), height=int(args.height),
                width=int(args.width), host_ref=False,
                no_cache_all=False, fast_codec=False, bo_resident=False,
                refiner_cache=str(args.refiner_cache),
                qkv=bool(req.get("qkv", bridge.QKV_ENV)),
                xpack_once=bool(req.get("xpack_once", bridge.XPACK_ENV)),
                qkv_xclbin=str(req.get("qkv_xclbin") or
                               (bridge.QKV_DEFAULT_OUT /
                                (bridge.QKV_DEFAULT_TAG + ".xclbin"))),
                qkv_bin=str(req.get("qkv_bin") or
                            (bridge.QKV_DEFAULT_OUT /
                             (bridge.QKV_DEFAULT_TAG + ".bin"))),
                p10=bool(req.get("p10", bridge.P10_ENV)),
                p10_xclbin=str(req.get("p10_xclbin") or
                               (bridge.P10_DEFAULT_OUT /
                                (bridge.P10_DEFAULT_TAG + ".xclbin"))),
                p10_bin=str(req.get("p10_bin") or
                            (bridge.P10_DEFAULT_OUT /
                             (bridge.P10_DEFAULT_TAG + ".bin"))))
            bridge._CAP = None
            bridge._STARTUP = False
            bridge._install(pe, ts, sg)
            CUR["dev"].stats = {"dev_ms": {}, "write_ms": {}, "calls": 0,
                                "attempts": [], "retries": 0}
            sys.argv = [str(pathlib.Path(DRV.__file__))] + \
                bridge.driver_argv(ns, imgs, None, imgs["ffn"], cache_dir,
                                   wpack=False,
                                   t18=(DRV is not bridge.t13_run))
            # one device, one user at a time: the VAE link takes this same
            # lock around every launch (C-1)
            with dev_lock:
                rc = int(DRV.main() or 0)
            served += 1
            emit({"event": "job_done", "id": rid, "ok": True, "rc": rc,
                  "seconds": round(time.perf_counter() - t_job, 3),
                  "served": served})
        except BaseException:
            emit({"event": "job_done", "id": rid, "ok": False, "rc": 2,
                  "seconds": round(time.perf_counter() - t_job, 3),
                  "error": traceback.format_exc()[-4000:], "served": served})
            break
        finally:
            # Honesty of the residency policy: whatever was NOT kept must not
            # creep.  The job's cache-all allocated the 3.71 GB in this very
            # process, so free it here; the next job reloads it (2.2-6 s).
            if not keep_refiner_resident:
                T.Loop._rcache.clear()
            if not keep_payload_resident:
                T.CACHE_BG = None
                T.CACHE_B3 = None
                T.BG_PREPERM = False
                import gc
                gc.collect()
    return 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--refiner-cache", default="")
    ap.add_argument("--driver", default="")
    ap.add_argument("--keep-refiner", choices=["auto", "0", "1"], default="auto")
    ap.add_argument("--keep-payload", choices=["auto", "0", "1"], default="auto")
    ap.add_argument("--vae-link", choices=["0", "1"], default="1",
                    help="T22 C-1: serve the VAE on this process's device "
                         "context over a loopback socket (0 = off, A/B arm)")
    ap.add_argument("--vae-link-file", default="",
                    help="optional JSON file receiving {pid, addr} of the link")
    ap.add_argument("--wpack-file", default="",
                    help="T22 prewarm: map this projection-image artifact "
                         "instead of packing (same seam as bridge --wpack-file)")
    ap.add_argument("--wpack-verify", action="store_true",
                    help="verify the artifact sha256 before mapping (1.0-1.5 s)")
    ap.add_argument("--reserve-gb", type=float, default=0.0,
                    help="commit headroom (GB) the residency policy must leave "
                         "free for the host's text-encoder backend (T22 workervae: "
                         "without it the next job's memory decision flips to low "
                         "and drops this worker)")
    args = ap.parse_args()
    if not args.serve:
        print("this is the resident device worker; pass --serve")
        return 2
    return _serve(args)


if __name__ == "__main__":
    raise SystemExit(main())
