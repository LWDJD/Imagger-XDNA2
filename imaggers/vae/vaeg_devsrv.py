# vaeg_devsrv.py -*- Python 3.10 -*-
"""VAE device server: many launches inside ONE py310 process.

WHY THIS FILE EXISTS
--------------------
The 3b3 executor spawned one py310 process per launch (vaeg_run_xrt.py).  At
128x128 one decode is 137 launches (104 conv blocks + 30 GN + 3 UP) and the
measured per-launch tax is ~0.33-0.36 s, i.e. ~45 s of a 96-98 s decode.  The
kernel time itself is 57.6 s (from the 3b3 calls.json: the per-launch
kernel_ms sum).  This server keeps what the kernel needs resident -- device,
xclbin, kernel handle, instruction BO, data BOs -- and does one launch per
request, so the only per-launch costs left are the payload copy, the
dispatch/wait and the output read.

THE LAWS IT OBEYS (all measured elsewhere, not re-derived here)
---------------------------------------------------------------
* one pyxrt.kernel handle per xclbin, ever (T18/T20-TATE8: a second handle on
  the same xclbin is the false-completion trap);
* --runs 1 per dispatch (the ERT watchdog is ~7 s per dispatch);
* fresh NPU_CACHE_DIR/NPU_CACHE_HOME per process, with the 0o700->0o777 mkdir
  shim (the runtime creates its cache dir with mode 0o700 and the shim is what
  made it work on this host);
* the instruction stream is the .bin next to the .xclbin, unchanged.

PROTOCOL
--------
vaeg_srv_proto: MAGIC | u32 n_parts | (u32 len | bytes)* ; part 0 is JSON.

    -> {"cmd": "hello"}                     <- {"ok":1, "py", "pid", "open_s"}
    -> {"cmd": "init", "barrier": {...}|0}  <- {"ok":1}
    -> {"cmd": "run", "stem", "dir", "ins": [n...], "out": n, "seq",
        "barrier": 0|1} + [payloads...]
                                            <- {"ok":1,"state":4,"fill_ms",
                                                "wall_ms","read_ms",...} + out
    -> {"cmd": "bye"}                       <- {"ok":1} and exit 0

A device error (state != 4) or an exception is answered with {"ok":0, ...}
and the process then exits, so the host can retry the same launch in a fresh
process (that is the 0xC0000005-on-first-dispatch policy, judged per call).

    py310 vaeg_devsrv.py --cache DIR [--log JSONL]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
KERNEL = "MLIR_AIE"
TIMEOUT_MS = 600000
XRT_DIRS = [
    pathlib.Path(r"D:\Project\deepseek_harness\test7\WStmp\driver376"
                 r"\npu_mcdm_stack_prod"),
    pathlib.Path(r"C:\Windows\System32\AMD"),
]

T_START = time.perf_counter()


def prep_env(cache: pathlib.Path) -> dict:
    """Fresh NPU cache dir + the mode shim, before pyxrt is imported."""
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["NPU_CACHE_DIR"] = str(cache)
    os.environ["NPU_CACHE_HOME"] = str(cache)
    orig_mkdir = os.mkdir

    def mkdir_compat(path, mode=0o777, *, dir_fd=None):
        if mode == 0o700:
            mode = 0o777
        return orig_mkdir(path, mode, dir_fd=dir_fd)

    os.mkdir = mkdir_compat
    for d in XRT_DIRS:
        if d.exists():
            sys.path.insert(0, str(d))
            try:
                os.add_dll_directory(str(d))
            except Exception:  # noqa: BLE001
                pass
    return dict(cache=str(cache), xrt=[str(d) for d in XRT_DIRS if d.exists()])


class Srv:
    def __init__(self, pyxrt, log=None):
        self.pyxrt = pyxrt
        self.t_open = time.perf_counter()
        self.dev = pyxrt.device(0)
        self.open_s = time.perf_counter() - self.t_open
        self.imgs: dict = {}
        self.last_stem = None
        self.barrier = None          # dict(stem, dir, ins, out) or None
        self.barrier_im = None
        self.log = log
        self.stats = dict(launches=0, barrier=0, reg=0, reg_s=0.0,
                          bytes_in=0, bytes_out=0)

    # -- registration -----------------------------------------------------
    def image(self, stem: str, d: str, ins, out: int) -> dict:
        im = self.imgs.get(stem)
        if im is not None:
            if list(im["ins"]) != list(ins) or im["out"] != int(out):
                raise RuntimeError(
                    "stem %s reused with a different geometry (%s/%s vs %s/%s)"
                    % (stem, im["ins"], im["out"], list(ins), out))
            return im
        pyxrt = self.pyxrt
        dp = pathlib.Path(d)
        t0 = time.perf_counter()
        xc = pyxrt.xclbin(str(dp / (stem + ".xclbin")))
        uuid = self.dev.register_xclbin(xc)
        k = pyxrt.kernel(self.dev, uuid, KERNEL)
        inst = (dp / (stem + ".bin")).read_bytes()
        bo_i = pyxrt.bo(self.dev, len(inst), pyxrt.bo.flags.cacheable,
                        k.group_id(1))
        bo_i.write(inst, 0)
        bo_i.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        fl = pyxrt.bo.flags.host_only
        im = dict(k=k, inst=inst, ninstr=len(inst) // 4, bo_i=bo_i, stem=stem,
                  ins=[int(n) for n in ins], out=int(out),
                  bo_in=[pyxrt.bo(self.dev, max(int(n), 4), fl, k.group_id(3 + i))
                         for i, n in enumerate(ins)],
                  bo_out=None, reg_ms=0.0)
        im["bo_out"] = pyxrt.bo(self.dev, max(int(out), 4), fl,
                                k.group_id(3 + len(ins)))
        im["reg_ms"] = 1e3 * (time.perf_counter() - t0)
        self.imgs[stem] = im
        self.stats["reg"] += 1
        self.stats["reg_s"] += im["reg_ms"] / 1e3
        return im

    def set_barrier(self, spec) -> None:
        if not spec:
            self.barrier = None
            return
        self.barrier = spec
        self.barrier_im = self.image(spec["stem"], spec["dir"], spec["ins"],
                                     spec["out"])
        self._barrier_buf = [b"\0" * int(n) for n in self.barrier_im["ins"]]

    # -- one launch -------------------------------------------------------
    def _launch(self, im: dict, payloads, want_out: bool):
        pyxrt = self.pyxrt
        t = time.perf_counter()
        for bo, b in zip(im["bo_in"], payloads):
            if b:
                bo.write(b, 0)
            bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        fill = time.perf_counter() - t
        t0 = time.perf_counter()
        run = im["k"](3, im["bo_i"], im["ninstr"], *im["bo_in"], im["bo_out"])
        state = int(run.wait(TIMEOUT_MS))
        wall = time.perf_counter() - t0
        raw = None
        read = 0.0
        if want_out and state == 4:
            t_r = time.perf_counter()
            im["bo_out"].sync(
                pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            raw = bytes(im["bo_out"].read(im["out"], 0))
            read = time.perf_counter() - t_r
        return state, fill, wall, read, raw

    def run(self, head: dict, payloads) -> tuple:
        stem = head["stem"]
        im = self.image(stem, head["dir"], head["ins"], head["out"])
        if len(payloads) != len(im["ins"]):
            raise RuntimeError("%s: %d payloads for %d inputs"
                               % (stem, len(payloads), len(im["ins"])))
        for b, want in zip(payloads, im["ins"]):
            if len(b) != want:
                raise RuntimeError("%s: payload %d B, BO is %d B"
                                   % (stem, len(b), want))
        bstats = 0
        if (head.get("barrier") and self.barrier_im is not None
                and self.last_stem == stem and self.barrier_im is not im):
            st, _f, _w, _r, _raw = self._launch(self.barrier_im,
                                                self._barrier_buf, False)
            self.stats["barrier"] += 1
            bstats += 1
            if st != 4:
                raise RuntimeError("barrier launch state %d" % st)
        state, fill, wall, read, raw = self._launch(im, payloads, True)
        self.last_stem = stem
        self.stats["launches"] += 1
        self.stats["bytes_in"] += sum(len(p) for p in payloads)
        if raw is not None:
            self.stats["bytes_out"] += len(raw)
        hdr = dict(ok=int(state == 4), state=state, stem=stem,
                   fill_ms=1e3 * fill, wall_ms=1e3 * wall, read_ms=1e3 * read,
                   reg_ms=im["reg_ms"], barrier=bstats, seq=head.get("seq"))
        if self.log:
            self.log.write(json.dumps(dict(hdr, t=time.time())) + "\n")
            self.log.flush()
        return hdr, raw


def watch_parent(ppid: int) -> None:
    """Die with the host: an orphaned server holds the device context forever.

    The pipe EOF covers the normal case (the host closing stdin makes
    recv_msg return None); this thread covers the case where the host is
    KILLED while the child is inside a dispatch, where the child would
    otherwise sit in run.wait() for the whole timeout with the device held.
    """
    import threading

    def loop():
        import ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = ctypes.c_void_p
        while True:
            time.sleep(1.0)
            h = k32.OpenProcess(0x1000, False, ppid)   # QUERY_LIMITED_INFORMATION
            if not h:
                os._exit(7)
            k32.CloseHandle(ctypes.c_void_p(h))

    t = threading.Thread(target=loop, daemon=True)
    t.start()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--log", default=None)
    ap.add_argument("--ppid", type=int, default=0,
                    help="exit when this parent is gone (orphan guard: a "
                         "dead host must not keep the device context)")
    args = ap.parse_args()

    sys.path.insert(0, str(HERE))
    from vaeg_srv_proto import recv_msg, send_msg, reader, writer  # noqa: E402

    if args.ppid:
        watch_parent(args.ppid)
    info = prep_env(pathlib.Path(args.cache))
    inp = reader(sys.stdin.buffer.raw)
    out = writer(sys.stdout.buffer.raw)
    log = open(args.log, "a", encoding="utf-8") if args.log else None

    t_imp = time.perf_counter()
    import pyxrt  # noqa: E402
    import_s = time.perf_counter() - t_imp
    srv = Srv(pyxrt, log=log)
    hello = dict(ok=1, pid=os.getpid(), py=sys.version.split()[0],
                 import_s=import_s, open_s=srv.open_s,
                 boot_s=time.perf_counter() - T_START,
                 cache=info["cache"], xrt=info["xrt"])
    send_msg(out, [json.dumps(hello).encode()])

    while True:
        parts = recv_msg(inp)
        if parts is None:
            return 0
        head = json.loads(parts[0])
        cmd = head.get("cmd")
        if cmd == "bye":
            send_msg(out, [json.dumps(dict(ok=1)).encode()])
            return 0
        if cmd == "ping":
            send_msg(out, [json.dumps(dict(ok=1, stats=srv.stats)).encode()])
            continue
        if cmd == "init":
            srv.set_barrier(head.get("barrier"))
            send_msg(out, [json.dumps(dict(ok=1, images=len(srv.imgs),
                                           barrier=bool(srv.barrier))
                                      ).encode()])
            continue
        if cmd == "run":
            try:
                hdr, raw = srv.run(head, parts[1:])
            except Exception as exc:  # noqa: BLE001
                send_msg(out, [json.dumps(dict(
                    ok=0, fatal=1, seq=head.get("seq"), stem=head.get("stem"),
                    err="%s: %s" % (type(exc).__name__, exc),
                    tb=traceback.format_exc()[-3000:])).encode()])
                return 4
            send_msg(out, [json.dumps(hdr).encode(), raw or b""])
            if not hdr["ok"]:
                # a dispatch that did not complete wedges the queue: report
                # and die, the host retries this launch in a fresh process
                return 5
            continue
        send_msg(out, [json.dumps(dict(ok=0, err="unknown cmd %r" % cmd)
                                  ).encode()])
        return 6


if __name__ == "__main__":
    raise SystemExit(main())
