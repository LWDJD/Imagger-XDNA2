# SPDX-License-Identifier: MIT
"""Take the F-14.1 lock as this process, then run the product webui with
ZIMAGE_BIGM=1 / ZIMAGE_CHUNK_PIPE=0."""
import datetime, os, pathlib, runpy, sys
BASE = pathlib.Path(r"D:\Project\deepseek_harness\public_work\zimage-bfp16-npu")
LOCK = BASE / "_scratch" / "device.lock"
def _instrument():
    """T22_TIMELINE=1: log every device round trip (t_start, dur, tag) from
    inside pyxrt, so it works no matter which host path runs the dispatch."""
    if os.environ.get("T22_TIMELINE", "0") != "1":
        return
    import time as _t
    try:
        import pyxrt as _px
    except (ModuleNotFoundError, ImportError):
        _d = r"D:\Project\deepseek_harness\test7\WStmp\driver376" \
             r"\npu_mcdm_stack_prod"
        sys.path.insert(0, _d)
        try:
            os.add_dll_directory(_d)
        except (AttributeError, OSError):
            pass
        import pyxrt as _px
    path = os.environ.get("T22_TIMELINE_FILE", "devtimeline.tsv")
    try:
        os.remove(path)
    except OSError:
        pass
    st = {"last": None, "n": 0}

    def _log(t0, dur, tag):
        st["n"] += 1
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write("%.6f\t%.6f\t%s\n" % (t0, dur, tag))
        except OSError:
            pass

    _sync = _px.bo.sync

    def sync(self, *a, **k):
        t0 = _t.perf_counter()
        prev = st["last"]
        r = _sync(self, *a, **k)
        t1 = _t.perf_counter()
        if prev is not None:
            _log(prev, t0 - prev, "HOST")     # gap before this wait = host time
        _log(t0, t1 - t0, "DEV")              # the blocking wait = device tail
        st["last"] = t1
        return r

    _px.bo.sync = sync
    print("[tl] pyxrt.bo.sync instrumented -> %s" % path, flush=True)


def main():
    _instrument()
    try:
        fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print("LOCK_BUSY: " + LOCK.read_text(encoding="utf-8", errors="replace"))
        return 9
    os.write(fd, ("owner=t22_bigpipe2 pid=%d acquired=%s purpose=bigm2_512_1024\n"
                  % (os.getpid(), datetime.datetime.now().astimezone().isoformat(timespec="seconds"))).encode())
    os.close(fd)
    os.environ["ZIMAGE_BIGM"] = "1"
    os.environ["ZIMAGE_CHUNK_PIPE"] = "0"
    os.environ["T22_BIGM_DEBUG"] = "1"
    print("LOCK_ACQUIRED pid=%d" % os.getpid(), flush=True)
    sys.argv = ["webui.py", "--host", "127.0.0.1", "--port", "7874"]
    runpy.run_path(str(BASE / "webui.py"), run_name="__main__")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
