# t22_vae_srv.py -*- Python -*-
"""Host side (py312) of the VAE device server: spawn, request, retry, timing.

Two modes:

* **spawn** (the delivered default): this module owns a py310 child
  (vaeg_devsrv.py) and talks to it over a pipe.
* **connect** (T22 C-1): the VAE runs in the RESIDENT DEVICE WORKER
  (webui/dev_worker.py) -- the process that already holds the DiT device
  context -- over a loopback socket, same vaeg_srv_proto frames.  Two device
  contexts in two processes is what killed that worker with 0xC0000005 in
  2a2b2 (2/2), so the only stable shape is one context serving both.  The
  address comes from ZIMAGE_VAE_SRV_ADDR ("127.0.0.1:port", the worker's
  "vae_addr" ready field); the runner sets it per VAE call, and it is re-read
  on every (re)connect so a worker restart between jobs is followed by the
  same host process.  In connect mode there is NO respawn policy: killing or
  replacing the worker is the runner's decision, and a failed launch raises
  (webui/vae_local.py then falls back to the CPU decoder for that batch).

In spawn mode one session per host process.  That is the state-8 law taken
literally: the only py310 process that ever holds a VAE xclbin handle is this
child, and the host creates at most one child at a time (get() shuts the old
one down and WAITS for it before spawning the next -- two device contexts at
once is what crashes this stack: 2a2b2 saw the resident worker die
0xC0000005 while a second context decoded).

A request is idempotent: the child only writes into its own BOs and replies
with the output bytes, so re-sending the same request to a fresh child is
always safe.  That is the whole crash policy -- the 0xC0000005-on-first-
dispatch flake, a wedged ERT queue and a missing xclbin all end as "the child
is gone, retry the same launch once in a new process".
"""
from __future__ import annotations

import json
import os
import pathlib
import socket
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
SRV = HERE / "vaeg_devsrv.py"
# Third-party locations: resolved, never hard-coded (imaggers/paths.py).
_root = pathlib.Path(__file__).resolve().parent
while not (_root / 'imaggers' / 'paths.py').is_file():
    if _root.parent == _root:
        raise RuntimeError('cannot locate the checkout root')
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from imaggers import paths as _paths  # noqa: E402

PY310 = _paths.python310()
XRT_DIR = _paths.xrt_dir()

sys.path.insert(0, str(HERE))
import vaeg_srv_proto as PROTO  # noqa: E402


class DevSrvError(RuntimeError):
    pass


class DevSrv:
    """A py310 device server holding xclbins, kernel handles and BOs."""

    def __init__(self, tag: str = "srv", cache_root=None, log_dir=None,
                 barrier=None, spawn_timeout_s: float = 120.0,
                 max_retries: int = 2, verbose: bool = True, addr: str = "",
                 connect_timeout_s: float = 15.0):
        self.tag = tag
        self.addr = str(addr or "")    # explicit link override (tests)
        self.connect_timeout_s = float(connect_timeout_s)
        self.sock = None
        self.connect_mode = False
        self.cache_root = pathlib.Path(cache_root or (HERE / "out" / "t22_vae"
                                                      / "_npucache"))
        self.log_dir = pathlib.Path(log_dir or (HERE / "out" / "t22_vae"
                                                / "_devsrv"))
        self.barrier = barrier
        self.spawn_timeout_s = spawn_timeout_s
        self.max_retries = max_retries
        self.verbose = verbose
        self.proc = None
        self.rd = None
        self.wr = None
        self.attempt = 0
        self.hello = {}
        self.stats = dict(spawns=0, connects=0, requests=0, retries=0,
                          launches=0, send_s=0.0, wait_s=0.0, respawn_s=0.0)
        self.spawn_s = 0.0

    # -- process life ----------------------------------------------------
    def _logpath(self) -> pathlib.Path:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        return self.log_dir / ("%s_%02d.log" % (self.tag, self.attempt))

    def _link_addr(self) -> str:
        """The device worker's link address, or '' for spawn mode.

        Re-read on every use (not cached at construction): the runner sets
        ZIMAGE_VAE_SRV_ADDR from the live worker's ready event before each VAE
        call, and vae_local caches the chain across jobs, so a worker restart
        between jobs must be followed without rebuilding this object.
        """
        return self.addr or os.environ.get("ZIMAGE_VAE_SRV_ADDR", "").strip()

    def _connect(self, addr: str) -> None:
        host, _, port = addr.rpartition(":")
        try:
            port_i = int(port)
        except ValueError:
            raise DevSrvError("bad ZIMAGE_VAE_SRV_ADDR %r" % (addr,))
        t0 = time.perf_counter()
        try:
            s = socket.create_connection((host or "127.0.0.1", port_i),
                                         timeout=self.connect_timeout_s)
        except OSError as exc:
            raise DevSrvError("cannot reach the device worker at %s (%s)"
                              % (addr, exc)) from exc
        s.settimeout(None)
        self.sock = s
        self.rd = PROTO.reader(s.makefile("rb", 0))
        self.wr = PROTO.writer(s.makefile("wb", 0))
        parts = PROTO.recv_msg(self.rd)
        if parts is None:
            raise DevSrvError("device worker link closed during handshake (%s)"
                              % addr)
        self.hello = json.loads(parts[0])
        if not self.hello.get("ok") or self.hello.get("kind") != "dev_worker":
            raise DevSrvError("device worker handshake: %r" % (self.hello,))
        self.stats["connects"] = int(self.stats.get("connects") or 0) + 1
        self.spawn_s = time.perf_counter() - t0
        self.hello["host_connect_s"] = self.spawn_s
        self.hello["link"] = addr

    def start(self) -> dict:
        addr = self._link_addr()
        if addr:
            self.connect_mode = True
            self._connect(addr)
            if self.barrier:
                self._cmd(dict(cmd="init", barrier=self.barrier))
            if self.verbose:
                print("[srv] %s connected to the device worker at %s in "
                      "%.3f s (pid %s, boot %.2f s)"
                      % (self.tag, addr, self.spawn_s,
                         self.hello.get("pid"), self.hello.get("boot_s", 0.0)),
                      flush=True)
            return self.hello
        self.connect_mode = False
        t0 = time.perf_counter()
        self.attempt += 1
        cache = self.cache_root / ("%s_%02d" % (self.tag, self.attempt))
        env = dict(os.environ)
        env["NPU_CACHE_DIR"] = str(cache)
        env["NPU_CACHE_HOME"] = str(cache)
        p = str(XRT_DIR)
        if p not in env.get("PATH", ""):
            env["PATH"] = env.get("PATH", "") + os.pathsep + p
        errf = open(self._logpath(), "a", encoding="utf-8", buffering=1)
        self._errf = errf
        cmd = [str(PY310), str(SRV), "--cache", str(cache),
               "--ppid", str(os.getpid())]
        self.proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env,
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=errf,
                                     bufsize=0)
        self.rd = PROTO.reader(self.proc.stdout)
        self.wr = PROTO.writer(self.proc.stdin)
        self.stats["spawns"] += 1
        parts = PROTO.recv_msg(self.rd)
        if parts is None:
            raise DevSrvError("device server died during handshake (see %s)"
                              % self._logpath())
        self.hello = json.loads(parts[0])
        if not self.hello.get("ok"):
            raise DevSrvError("device server handshake: %r" % (self.hello,))
        self.spawn_s = time.perf_counter() - t0
        self.hello["host_spawn_s"] = self.spawn_s
        if self.barrier:
            self._cmd(dict(cmd="init", barrier=self.barrier))
        if self.verbose:
            print("[srv] %s up in %.2f s (child boot %.2f s, open %.3f s)"
                  % (self.tag, self.spawn_s, self.hello.get("boot_s", 0.0),
                     self.hello.get("open_s", 0.0)), flush=True)
        return self.hello

    def _cmd(self, head, parts=()) -> dict:
        PROTO.send_msg(self.wr, [json.dumps(head).encode()] + list(parts))
        rep = PROTO.recv_msg(self.rd)
        if rep is None:
            raise DevSrvError("device server closed on %r" % (head.get("cmd"),))
        return json.loads(rep[0])

    def kill(self) -> None:
        if self.connect_mode:
            # Close the socket only.  The device worker is NOT ours to kill:
            # it holds the DiT context and the runner decides its fate.
            for f in (self.rd, self.wr):
                try:
                    f.close()
                except Exception:  # noqa: BLE001
                    pass
            try:
                if self.sock is not None:
                    self.sock.close()
            except Exception:  # noqa: BLE001
                pass
            self.sock = None
            self.rd = self.wr = None
            return
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:
                try:
                    PROTO.send_msg(self.wr, [b'{"cmd": "bye"}'])
                except Exception:  # noqa: BLE001
                    pass
                try:
                    self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.proc.terminate()
                    try:
                        self.proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.proc.kill()
                        self.proc.wait(timeout=5)
        finally:
            for f in (self.rd, self.wr):
                try:
                    f.close()
                except Exception:  # noqa: BLE001
                    pass
            for f in (self.proc.stdin, self.proc.stdout):
                try:
                    f.close()
                except Exception:  # noqa: BLE001
                    pass
            try:
                self._errf.close()
            except Exception:  # noqa: BLE001
                pass
            self.proc = None
            self.rd = self.wr = None

    def close(self) -> None:
        self.kill()

    # -- one launch ------------------------------------------------------
    @staticmethod
    def _flat(p):
        """A 1-D uint8 view of a payload (no copy) + its true byte count."""
        if isinstance(p, (bytes, bytearray, memoryview)):
            return (p if isinstance(p, bytes) else bytes(p)), len(p)
        import numpy as np
        a = np.ascontiguousarray(p).view(np.uint8).reshape(-1)
        return a, int(a.size)

    def run(self, stem: str, d, payloads, out_len: int, seq: int = 0,
            barrier: bool = False) -> dict:
        # health: spawn mode = the child is alive; connect mode = the socket
        # is open (and the address may have changed if the worker restarted)
        addr = self._link_addr()
        if addr:
            if not self.connect_mode or self.sock is None or addr != \
                    str(self.hello.get("link") or ""):
                self.kill()
                self.start()
        elif self.proc is None or self.proc.poll() is not None:
            self.kill()
            self.start()
        payloads = [self._flat(p) for p in payloads]
        head = dict(cmd="run", stem=stem, dir=str(d),
                    ins=[n for _b, n in payloads], out=int(out_len),
                    seq=int(seq), barrier=int(bool(barrier)))
        payloads = [b for b, _n in payloads]
        last = None
        for try_i in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                PROTO.send_msg(self.wr, [json.dumps(head).encode()]
                               + list(payloads))
                send_s = time.perf_counter() - t0
                rep = PROTO.recv_msg(self.rd)
                wait_s = time.perf_counter() - t0 - send_s
                if rep is None:
                    if self.connect_mode:
                        raise DevSrvError(
                            "device worker link closed mid-request (stem %s)"
                            % (stem,))
                    raise DevSrvError("server died mid-request (exit %s)"
                                      % self.proc.poll())
                hdr = json.loads(rep[0])
            except (OSError, DevSrvError) as exc:
                last = "%s: %s" % (type(exc).__name__, exc)
                if self.connect_mode:
                    # no respawn policy on a worker we do not own: close the
                    # link so the next call reconnects, then report.  The
                    # caller (vae_local) falls back to the CPU decoder.
                    self.stats["retries"] += 1
                    self.kill()
                    raise DevSrvError("%s (stem %s, device worker link)"
                                      % (last, stem)) from exc
                self.stats["retries"] += 1
                t_r = time.perf_counter()
                self.kill()
                try:
                    self.start()
                except DevSrvError as exc2:
                    last += " | respawn: %s" % exc2
                    raise DevSrvError("%s (stem %s, after %d attempts)"
                                      % (last, stem, try_i + 1)) from exc2
                self.stats["respawn_s"] += time.perf_counter() - t_r
                continue
            if hdr.get("fatal"):
                # Device-context exhaustion ("Failed to create context
                # virtual", 0xc01e0009) is a PROCESS-level condition on this
                # driver stack: a fresh process gets a fresh context budget.
                # Retry the same launch there instead of failing the decode.
                # In connect mode there is no fresh process to retry in.
                if self.connect_mode:
                    self.kill()
                    raise DevSrvError("device worker: %s\n%s"
                                      % (hdr.get("err"), hdr.get("tb")))
                if any(t in str(hdr.get("err", "")) for t in
                       ("create context", "0xc01e0009", "context virtual",
                        "MemoryError", "out of memory")):
                    last = "context: %s" % hdr.get("err")
                    self.stats["retries"] += 1
                    t_r = time.perf_counter()
                    self.kill()
                    if try_i < self.max_retries:
                        self.start()
                        self.stats["respawn_s"] += time.perf_counter() - t_r
                        continue
                raise DevSrvError("device server: %s\n%s"
                                  % (hdr.get("err"), hdr.get("tb")))
            self.stats["requests"] += 1
            self.stats["send_s"] += send_s
            self.stats["wait_s"] += wait_s
            if hdr.get("ok") and len(rep) > 1:
                hdr["raw"] = rep[1]
            else:
                hdr["raw"] = None
            hdr["host_send_s"] = send_s
            hdr["host_wait_s"] = wait_s
            hdr["retries"] = try_i
            if not hdr.get("ok") and hdr.get("state") not in (4,):
                # a dispatch that did not complete: queue state is unknown,
                # retry this launch in a fresh process (the child exits 5)
                last = "dispatch state %s" % (hdr.get("state"),)
                if self.connect_mode:
                    self.stats["retries"] += 1
                    self.kill()
                    raise DevSrvError("%s (stem %s, device worker)"
                                      % (last, stem))
                self.stats["retries"] += 1
                t_r = time.perf_counter()
                self.kill()
                if try_i < self.max_retries:
                    self.start()
                    self.stats["respawn_s"] += time.perf_counter() - t_r
                    continue
                raise DevSrvError("%s (stem %s, %d attempts)"
                                  % (last, stem, try_i + 1))
            return hdr
        raise DevSrvError("unreachable")


# ---------------------------------------------------------------------------
# the process-wide instance (the platform allows exactly one device context)
# ---------------------------------------------------------------------------

_LIVE = {"srv": None}


def get(tag: str = "srv", restart: bool = False, **kw) -> DevSrv:
    s = _LIVE["srv"]
    if s is not None and (restart or s.tag != tag):
        s.close()
        s = None
    if s is None:
        s = DevSrv(tag=tag, **kw)
        s.start()
        _LIVE["srv"] = s
    return s


def shutdown() -> None:
    s = _LIVE["srv"]
    if s is not None:
        s.close()
        _LIVE["srv"] = None
