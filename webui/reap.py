# SPDX-License-Identifier: MIT
"""reap.py -- unconditional cleanup of the device worker tree.

WHY THIS FILE EXISTS
--------------------
Every failed 1080p attempt in rq_1080p_8d/8e/8f left a live py310 process
behind (1.8 GB low mode, 7.3 GB fast mode).  The mechanism is not a bug in one
cleanup path, it is the SHAPE of the process tree:

    driver            imaggers/repro/t22_repro.py  (py312)
      +- server       launch_webui_bigm.py -> webui.py  (py312, port 7874)
           +- bridge  webui/bridge.py -> t18_run.py     (py310, the NPU)
           +- wpack   bridge.py --pack-prewarm          (py310, 1.85 GB)
           +- te      webui/te_worker.py                (py312, GGUF)
           +- vae     webui/vae_worker.py               (py312)

The driver only ever holds the SERVER pid, and on Windows Popen.terminate() is
TerminateProcess(): no atexit, no signal, no finally -- the server dies without
running one line of its own cleanup, so everything below it is orphaned.
Measured on this box (imaggers/par/membudget/leak_probe.py, 300 MB stand-in):
the shipped path leaves the child alive with 0.305 GB RSS and the memory never
comes back.

The amplifier is the retry loop: one leak costs the headroom that makes the next
attempt fail, which leaks again.  The run that could not allocate 120 MiB had
its own previous attempt still resident.

WHAT THIS DOES (two independent layers, neither trusts the other)
----------------------------------------------------------------
1. A Windows JOB OBJECT with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, installed on the
   SERVER before it spawns anything.  Every descendant inherits membership, and
   the kernel closes the job when the last handle does -- which happens even for
   TerminateProcess, Task Manager "End task", or a 0xC0000005 crash.  This is
   the only mechanism that survives a kill the process never sees.
2. A psutil tree walk (kill_tree) for the case where the job could not be
   installed (a parent job that forbids nesting, a non-Windows host, or a
   process started before this file existed): kill children before the parent,
   verify, and fall back to "taskkill /F /T".

Both are unconditional: nothing here is on a success path only.  The delivered
512^2 device path is untouched -- no numeric code imports this module.
"""
from __future__ import annotations

import atexit
import os
import subprocess
import sys
import time

IS_WIN = os.name == "nt"

# Set ZIMAGE_NO_JOB=1 to turn the kernel layer off (debugging only: the tree
# walk still runs, and a run with it off can leak if it is hard-killed).
_ENV_OFF = "ZIMAGE_NO_JOB"

_JOB = {"handle": None, "tried": False, "why": "", "flags": 0}
_LOG = []


def log(msg):
    _LOG.append(str(msg))
    try:
        print("[reap] " + str(msg), flush=True)
    except Exception:                                        # noqa: BLE001
        try:
            sys.stdout.write(("[reap] " + str(msg)).encode(
                "ascii", "replace").decode("ascii") + "\n")
            sys.stdout.flush()
        except Exception:                                    # noqa: BLE001
            pass


def disabled():
    return os.environ.get(_ENV_OFF, "0").strip().lower() in (
        "1", "true", "yes", "on", "y", "t")


# ---------------------------------------------------------------------------
# 1. the kernel layer
# ---------------------------------------------------------------------------
KILL_ON_JOB_CLOSE = 0x2000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_TERMINATE = 0x0001


def _kernel32():
    import ctypes
    import ctypes.wintypes as wt
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.CreateJobObjectW.restype = ctypes.c_void_p
    k.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    k.OpenProcess.restype = ctypes.c_void_p
    k.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
    k.SetInformationJobObject.restype = wt.BOOL
    k.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                          ctypes.c_void_p, wt.DWORD]
    k.AssignProcessToJobObject.restype = wt.BOOL
    k.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    k.QueryInformationJobObject.restype = wt.BOOL
    k.QueryInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                            ctypes.c_void_p, wt.DWORD,
                                            ctypes.c_void_p]
    k.TerminateJobObject.restype = wt.BOOL
    k.TerminateJobObject.argtypes = [ctypes.c_void_p, wt.UINT]
    k.CloseHandle.restype = wt.BOOL
    k.CloseHandle.argtypes = [ctypes.c_void_p]
    return k


def _structs():
    import ctypes
    import ctypes.wintypes as wt

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in
                    ("ReadOperationCount", "WriteOperationCount",
                     "OtherOperationCount", "ReadTransferCount",
                     "WriteTransferCount", "OtherTransferCount")]

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wt.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wt.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wt.DWORD),
                    ("SchedulingClass", wt.DWORD)]

    class EXT(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC),
                    ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    return BASIC, EXT


def install_kill_on_close(pid=None):
    """Put this process (or pid) in a KILL_ON_JOB_CLOSE job.  Never raises.

    Call it BEFORE the first child is spawned: job membership is inherited at
    CreateProcess time, so a child created earlier is not covered.

    Returns the job handle, or None when the kernel layer is unavailable.
    """
    if _JOB["handle"] is not None:
        return _JOB["handle"]
    if not IS_WIN:
        _JOB["why"] = "not Windows"
        return None
    if disabled():
        _JOB["why"] = _ENV_OFF + "=1"
        log("kernel layer DISABLED by %s=1 (tree walk still runs)" % _ENV_OFF)
        return None
    import ctypes
    try:
        k = _kernel32()
        _, EXT = _structs()
        h = k.CreateJobObjectW(None, None)
        if not h:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW")
        info = EXT()
        info.BasicLimitInformation.LimitFlags = KILL_ON_JOB_CLOSE
        if not k.SetInformationJobObject(
                h, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(info), ctypes.sizeof(info)):
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject")
        target = os.getpid() if pid is None else int(pid)
        hp = k.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, target)
        if not hp:
            raise OSError(ctypes.get_last_error(), "OpenProcess")
        try:
            # NOT GetCurrentProcess(): its pseudo-handle is refused with
            # ERROR_INVALID_HANDLE and ctypes truncates it to 32 bits.
            if not k.AssignProcessToJobObject(h, hp):
                raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject")
        finally:
            k.CloseHandle(hp)
        _JOB["handle"] = h
        _JOB["flags"] = KILL_ON_JOB_CLOSE
        _JOB["why"] = ("pid %d joined a KILL_ON_JOB_CLOSE job; every descendant "
                       "dies with it, including on TerminateProcess" % target)
        log("kernel layer ON: " + _JOB["why"])
        return h
    except Exception as exc:                                 # noqa: BLE001
        _JOB["why"] = "unavailable: %r" % (exc,)
        log("kernel layer UNAVAILABLE (%r); the tree walk is the only "
            "protection" % (exc,))
        return None
    finally:
        _JOB["tried"] = True


def job_status():
    """(installed, why, active_processes) for a record or a preflight line."""
    h = _JOB["handle"]
    n = None
    if h and IS_WIN:
        import ctypes
        try:
            k = _kernel32()

            class ACC(ctypes.Structure):
                _fields_ = [("TotalUserTime", ctypes.c_int64),
                            ("TotalKernelTime", ctypes.c_int64),
                            ("ThisPeriodTotalUserTime", ctypes.c_int64),
                            ("ThisPeriodTotalKernelTime", ctypes.c_int64),
                            ("TotalPageFaultCount", ctypes.c_uint32),
                            ("TotalProcesses", ctypes.c_uint32),
                            ("ActiveProcesses", ctypes.c_uint32),
                            ("TotalTerminatedProcesses", ctypes.c_uint32)]

            acc = ACC()
            if k.QueryInformationJobObject(
                    h, _JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
                    ctypes.byref(acc), ctypes.sizeof(acc), None):
                n = int(acc.ActiveProcesses)
        except Exception:                                    # noqa: BLE001
            n = None
    return bool(h), _JOB["why"], n


def terminate_job():
    """Kill every process in the job (the job itself stays installed)."""
    h = _JOB["handle"]
    if not h or not IS_WIN:
        return False
    import ctypes
    try:
        return bool(_kernel32().TerminateJobObject(h, 1))
    except Exception:                                        # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# 2. the tree walk
# ---------------------------------------------------------------------------
def _psutil():
    try:
        import psutil
        return psutil
    except Exception:                                        # noqa: BLE001
        return None


def descendants(pid):
    """Every live descendant of pid ([] when psutil is absent)."""
    ps = _psutil()
    if ps is None:
        return []
    try:
        parent = ps.Process(int(pid))
        return [p for p in parent.children(recursive=True) if p.pid != os.getpid()]
    except Exception:                                        # noqa: BLE001
        return []


def alive(pid):
    ps = _psutil()
    if ps is None:
        return None
    try:
        return bool(ps.pid_exists(int(pid)) and ps.Process(int(pid)).is_running())
    except Exception:                                        # noqa: BLE001
        return False


def _taskkill(pid):
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(int(pid))],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                       timeout=30)
        return True
    except Exception:                                        # noqa: BLE001
        return False


def kill_tree(pid, timeout=15.0, include_root=True):
    """Kill pid and every descendant.  Returns {"killed": [...], "survivors": [...]}.

    Children first, so a dying parent cannot re-parent a child out of reach
    between the snapshot and the kill.  taskkill /F /T is the fallback when
    psutil is missing, and the second pass when a survivor is still there.
    """
    pid = int(pid)
    if pid == os.getpid():
        return {"killed": [], "survivors": [pid], "note": "refused to kill self"}
    order = [p.pid for p in descendants(pid)]
    if include_root:
        order.append(pid)
    killed, survivors = [], []
    ps = _psutil()
    if ps is not None:
        for k in order:
            try:
                ps.Process(k).kill()
                killed.append(k)
            except Exception:                                # noqa: BLE001
                pass
        t0 = time.time()
        while time.time() - t0 < timeout:
            if not [k for k in order if alive(k)]:
                break
            time.sleep(0.1)
        survivors = [k for k in order if alive(k)]
    else:
        for k in order:
            if _taskkill(k):
                killed.append(k)
        survivors = [k for k in order if alive(k)]
    if survivors:
        _taskkill(pid)
        survivors = [k for k in survivors if alive(k)]
    return {"killed": killed, "survivors": survivors}


def reap(proc, why="", timeout=15.0):
    """Kill a Popen's whole tree and reap the Popen object.  Never raises.

    This is the replacement for proc.terminate() at every site that owns a
    child which can itself own a child (the server, the resident worker, the
    pack prewarm).  It returns a dict that a test can assert on.
    """
    res = {"why": why, "pid": None, "killed": [], "survivors": []}
    if proc is None:
        return res
    pid = getattr(proc, "pid", None)
    res["pid"] = pid
    if pid is not None:
        k = kill_tree(pid, timeout=timeout)
        res["killed"] = k["killed"]
        res["survivors"] = k["survivors"]
    try:
        proc.wait(timeout=timeout)
    except Exception:                                        # noqa: BLE001
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:                                    # noqa: BLE001
            pass
    if why:
        log("reaped %s: pid %s, %d process(es) killed, %d survivor(s)"
            % (why, pid, len(res["killed"]), len(res["survivors"])))
    return res


# ---------------------------------------------------------------------------
# 3. exit hook
# ---------------------------------------------------------------------------
def _on_exit():
    """atexit: kill what is still ours.  A no-op when nothing is running."""
    me = os.getpid()
    kids = descendants(me)
    if not kids:
        return
    for p in kids:
        try:
            p.kill()
        except Exception:                                    # noqa: BLE001
            pass
    log("atexit: killed %d child process(es): %s"
        % (len(kids), ", ".join(str(p.pid) for p in kids)))


def install_exit_hook():
    atexit.register(_on_exit)


def selftest():
    ok, why, n = job_status()
    out = {"platform": sys.platform, "job_installed": ok, "job_why": why,
           "job_members": n, "psutil": _psutil() is not None,
           "children": [p.pid for p in descendants(os.getpid())]}
    log("selftest " + str(out))
    return out
