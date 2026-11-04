# SPDX-License-Identifier: MIT
"""Line-streaming subprocess helper (stdlib only, Windows-friendly)."""
from __future__ import annotations

import pathlib
import queue
import subprocess
import threading
import time

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _reap_tree(proc, why=""):
    """Kill the child AND everything it spawned (webui/reap.py).

    A cancelled/timed-out device child is the py310 bridge, which can
    itself own a VAE worker; proc.terminate() leaves those running and
    they keep their commit.  Never raises: proc.kill() is the floor.
    """
    try:
        from . import reap
        return reap.reap(proc, why=why)
    except Exception:                                    # noqa: BLE001
        try:
            proc.kill()
        except Exception:                                # noqa: BLE001
            pass
        return None


def stream_process(cmd, env=None, cwd=None, log_path=None, on_line=None,
                   timeout=900.0, cancel_check=None, tail_limit=5000,
                   on_proc=None):
    """Run cmd, stream merged stdout/stderr, tee to log_path."""
    log_path = pathlib.Path(log_path) if log_path else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        [str(x) for x in cmd], cwd=str(cwd) if cwd else None, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace", bufsize=1,
        creationflags=CREATE_NO_WINDOW)
    if on_proc:
        on_proc(proc)
    q = queue.Queue()

    def _reader():
        try:
            for line in proc.stdout:
                q.put(line)
        finally:
            q.put(None)

    th = threading.Thread(target=_reader, daemon=True)
    th.start()
    lines = []
    timed_out = False
    cancelled = False
    logf = open(log_path, "w", encoding="utf-8", errors="replace") if log_path else None
    try:
        while True:
            if cancel_check and cancel_check():
                cancelled = True
                _reap_tree(proc, why="cancelled device child")
            if time.perf_counter() - t0 > timeout:
                timed_out = True
                _reap_tree(proc, why="timed-out device child")
            try:
                line = q.get(timeout=0.25)
            except queue.Empty:
                if proc.poll() is not None and q.empty():
                    break
                continue
            if line is None:
                break
            line = line.rstrip("\r\n")
            lines.append(line)
            if len(lines) > tail_limit * 2:
                lines = lines[-tail_limit:]
            if logf:
                logf.write(line + "\n")
                logf.flush()
            if on_line:
                on_line(line)
        try:
            rc = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = proc.wait(timeout=30)
    finally:
        if logf:
            logf.close()
        if proc.poll() is None:
            # unconditional: this runs on success, on exception, on
            # MemoryError and on Cancelled alike
            _reap_tree(proc, why="device child left alive at exit")
    return {"rc": rc, "timed_out": timed_out, "cancelled": cancelled,
            "seconds": time.perf_counter() - t0, "lines": lines[-tail_limit:]}
