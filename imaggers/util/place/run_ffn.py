# SPDX-License-Identifier: MIT
"""run_ffn.py -- DEVICE side of the S2 real-FFN images (py310; the parent runs it).

    py310 run_ffn.py --shape ffn_gu_m48 --runs 7 --warmup 1

Reads out/<shape>_io.npz (prep_ffn.py), runs the image once per iteration,
writes out/<shape>_C.npy (raw device C, uint8) and out/<shape>_dev.json.
Does NOT verify -- verify_ffn.py does that.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

# Third-party locations: resolved, never hard-coded (imaggers/paths.py).
_root = pathlib.Path(__file__).resolve().parent
while not (_root / 'imaggers' / 'paths.py').is_file():
    if _root.parent == _root:
        raise RuntimeError('cannot locate the checkout root')
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from imaggers import paths as _paths  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
XRT_DIRS = [
    _paths.xrt_dir(),
    pathlib.Path(r"C:\Windows\System32\AMD"),
]
KERNEL = "MLIR_AIE"
ERT_COMPLETED = 4


def _stats(xs):
    a = np.asarray(xs, np.float64)
    if a.size == 0:
        return {}
    return {"n": int(a.size), "min": float(a.min()), "median": float(np.median(a)),
            "mean": float(a.mean()), "max": float(a.max()),
            "std": float(a.std(ddof=0))}



# ===========================================================================
# The 08:06 path -- VERBATIM.  This is the code that ran the four arms on the
# device (four gates PASS, 17.90/1.251/17.259/1.290 ms).  It is used by
# main() by default (--engine inline) and by s2_dev.run_inline().  DO NOT
# refactor it: a refactor of this exact sequence is what broke the only
# verification channel on 2026-11-01.  The class below is a SEPARATE branch
# (--engine class) that exists to be A/B'd, not to replace this.
# ===========================================================================
def _pyxrt(XRT_DIRS=XRT_DIRS):
    for d in XRT_DIRS:
        if d.exists():
            sys.path.insert(0, str(d))
            try:
                os.add_dll_directory(str(d))
            except Exception:  # noqa: BLE001
                pass
    import pyxrt
    return pyxrt


def run_inline(shape, a_bytes, b_bytes, c_len, runs=1, warmup=1,
               xclbin=None, insts=None, host_bos=None):
    """The 08:06 sequence, callable.  Returns (C, walls, states, n_bo).

    Every step below is copied line for line from the version that passed the
    four gates; the only addition is returning instead of printing.
    """
    xclbin = pathlib.Path(xclbin) if xclbin else OUT / f"{shape}.xclbin"
    insts = pathlib.Path(insts) if insts else OUT / f"{shape}.bin"
    n_bo = 3
    kj = xclbin.with_suffix(".prj") / "kernels_main.json"
    if host_bos is not None:
        n_bo = max(3, host_bos)
    elif kj.is_file():
        n_bo = max(3, len(__import__("re").findall(r'"name":\s*"bo\d+"',
                                                  kj.read_text(encoding="utf-8"))))
    pyxrt = _pyxrt()

    inst = insts.read_bytes()
    ninstr = len(inst) // 4
    dev = pyxrt.device(0)
    uuid = dev.register_xclbin(pyxrt.xclbin(str(xclbin)))
    kern = pyxrt.kernel(dev, uuid, KERNEL)

    fl = pyxrt.bo.flags.host_only
    bo_instr = pyxrt.bo(dev, len(inst), pyxrt.bo.flags.cacheable, kern.group_id(1))
    payloads = [bytes(a_bytes), bytes(b_bytes), bytes(int(c_len))] + \
               [b""] * (n_bo - 3)
    bos = []
    for i, payload in enumerate(payloads):
        bo = pyxrt.bo(dev, max(len(payload), 4), fl, kern.group_id(3 + i))
        if payload:
            bo.write(payload, 0)
        bos.append(bo)
    bo_c = bos[2]
    bo_instr.write(inst, 0)
    for b in [bo_instr] + bos:
        b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    walls, states = [], []
    for i in range(warmup + runs):
        t0 = time.perf_counter()
        run = kern(3, bo_instr, ninstr, *bos)
        state = run.wait()
        t1 = time.perf_counter()
        if i < warmup:
            continue
        walls.append(1000.0 * (t1 - t0))
        states.append(int(state))
    bo_c.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    craw = np.frombuffer(bytes(bo_c.read(int(c_len), 0)), np.uint8).copy()
    # keep the handles alive until the read is done (the class path dropped
    # them, which is the prime suspect for the 0xC0000005)
    del dev, uuid, kern
    return craw, walls, states, n_bo


# ===========================================================================
# The OPTIONAL branch (--engine class).  Same protocol, persistent BOs, plus
# a mapped B write.  A/B only -- never the default.
# ===========================================================================
class ImageRunner:
    """Register one xclbin, build its BOs, run it, read C back.

    NOT the default path.  It keeps the pyxrt handles on the instance
    (register_xclbin's uuid owns the xclbin's lifetime; dropping it is a
    plausible 0xC0000005).
    """

    def __init__(self, shape, xclbin=None, insts=None, host_bos=None):
        self.shape = shape
        self.xclbin = pathlib.Path(xclbin) if xclbin else OUT / f"{shape}.xclbin"
        self.insts = pathlib.Path(insts) if insts else OUT / f"{shape}.bin"
        self.host_bos = host_bos
        self.dev = None
        self.uuid = None
        self.xc = None
        self.kern = None
        self.bo_instr = None
        self.bos = None
        self.n_bo = 3

    def open(self):
        if self.kern is not None:
            return self
        if not (self.xclbin.is_file() and self.insts.is_file()):
            raise FileNotFoundError(f"{self.shape}: missing {self.xclbin} / "
                                    f"{self.insts}")
        n_bo = 3
        kj = self.xclbin.with_suffix(".prj") / "kernels_main.json"
        if self.host_bos is not None:
            n_bo = max(3, self.host_bos)
        elif kj.is_file():
            n_bo = max(3, len(__import__("re").findall(
                r'"name":\s*"bo\d+"', kj.read_text(encoding="utf-8"))))
        self.n_bo = n_bo
        pyxrt = _pyxrt()
        self._pyxrt = pyxrt
        if self.dev is None:
            self.dev = pyxrt.device(0)
        self.xc = pyxrt.xclbin(str(self.xclbin))
        self.uuid = self.dev.register_xclbin(self.xc)
        self.kern = pyxrt.kernel(self.dev, self.uuid, KERNEL)
        inst = self.insts.read_bytes()
        self.bo_instr = pyxrt.bo(self.dev, len(inst), pyxrt.bo.flags.cacheable,
                                 self.kern.group_id(1))
        self.bo_instr.write(inst, 0)
        self.bo_instr.sync(
            pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        self._n_instr = len(inst) // 4
        return self

    def alloc(self, a_len, b_len, c_len):
        self.open()
        pyxrt = self._pyxrt
        fl = pyxrt.bo.flags.host_only
        if self.bos is None:
            sizes = [int(a_len), int(b_len), int(c_len)] + [4] * (self.n_bo - 3)
            self.bos = [pyxrt.bo(self.dev, max(s, 4), fl,
                                 self.kern.group_id(3 + i))
                        for i, s in enumerate(sizes)]
        return self.bos

    def _stage(self, bo, arr, nbytes, use_map=True):
        if use_map:
            try:
                m = bo.map()
                try:
                    mv = memoryview(m).cast("B")
                    mv[:nbytes] = memoryview(arr).cast("B")[:nbytes]
                except TypeError:
                    import ctypes
                    ctypes.memmove(int(m), arr.ctypes.data, nbytes)
                self.stage_path = "map"
                return
            except Exception as exc:  # noqa: BLE001
                self.stage_err = repr(exc)
        bo.write(bytes(memoryview(arr).cast("B")[:nbytes]), 0)
        self.stage_path = "write"

    def run_staged(self, a_arr, b_arr, c_len, runs=1, warmup=0, use_map=True):
        pyxrt = self.open()._pyxrt
        self.alloc(a_arr.nbytes, b_arr.nbytes, int(c_len))
        bo_a, bo_b, bo_c = self.bos[:3]
        t0 = time.perf_counter()
        self._stage(bo_a, a_arr, a_arr.nbytes, use_map)
        self._stage(bo_b, b_arr, b_arr.nbytes, use_map)
        self.stage_ms = 1000.0 * (time.perf_counter() - t0)
        for b in self.bos:
            b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        walls, states = [], []
        for i in range(warmup + runs):
            t0 = time.perf_counter()
            run = self.kern(3, self.bo_instr, self._n_instr, *self.bos)
            state = int(run.wait())
            t1 = time.perf_counter()
            if i < warmup:
                continue
            walls.append(1000.0 * (t1 - t0))
            states.append(state)
        bo_c.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        craw = np.frombuffer(bytes(bo_c.read(int(c_len), 0)), np.uint8).copy()
        return craw, walls, states


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="ffn_gu_m48")
    ap.add_argument("--xclbin", default=None)
    ap.add_argument("--insts", default=None)
    ap.add_argument("--npz", default=None)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=7)
    ap.add_argument("--host-bos", type=int, default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--print-only", action="store_true")
    ap.add_argument("--engine", choices=("inline", "class"), default="inline",
                    help="inline = the 08:06 path (default, gates PASS); "
                         "class = the A/B branch")
    args = ap.parse_args()

    npz_p = pathlib.Path(args.npz) if args.npz else OUT / f"{args.shape}_io.npz"
    xclbin = pathlib.Path(args.xclbin) if args.xclbin else OUT / f"{args.shape}.xclbin"
    insts = pathlib.Path(args.insts) if args.insts else OUT / f"{args.shape}.bin"
    io = np.load(npz_p)
    M, K, N, m, kk, n, cols, cpc, ntiles = [int(x) for x in io["dims"]]
    a_bytes = io["A_bytes"].tobytes()
    b_bytes = io["B_bytes"].tobytes()
    c_len = int(io["C_ref"].size)
    nmac = M * K * N
    plan = {"shape": args.shape, "xclbin": str(xclbin),
            "xclbin_bytes": xclbin.stat().st_size if xclbin.is_file() else None,
            "insts_bytes": insts.stat().st_size if insts.is_file() else None,
            "M": M, "K": K, "N": N, "m": m, "kk": kk, "n": n,
            "cols": cols, "cpc": cpc, "ntiles": ntiles,
            "A_bytes": len(a_bytes), "B_bytes": len(b_bytes), "C_bytes": c_len,
            "GMAC": nmac / 1e9, "GFLOP": 2 * nmac / 1e9, "engine": args.engine}
    print(json.dumps(plan, indent=2), flush=True)
    if args.print_only:
        return 0
    if not (xclbin.is_file() and insts.is_file()):
        print("missing artifact", file=sys.stderr)
        return 2

    if args.engine == "class":
        runner = ImageRunner(args.shape, xclbin, insts, args.host_bos).open()
        a_arr = np.frombuffer(a_bytes, np.uint8)
        b_arr = np.frombuffer(b_bytes, np.uint8)
        craw, walls, states = runner.run_staged(a_arr, b_arr, c_len, args.runs,
                                                args.warmup, use_map=False)
        n_bo = runner.n_bo
    else:
        craw, walls, states, n_bo = run_inline(
            args.shape, a_bytes, b_bytes, c_len, args.runs, args.warmup,
            xclbin, insts, args.host_bos)

    np.save(OUT / f"{args.shape}_C.npy", craw)
    wall = _stats(walls)
    rec = dict(plan)
    rec.update({"host_bos": n_bo, "wall_ms": wall, "states": states,
                "all_completed": bool(states) and all(s == ERT_COMPLETED for s in states),
                "GFLOP_per_s": (2 * nmac / (wall["median"] * 1e-3) / 1e9
                                if wall else None),
                "C_sha16": __import__("hashlib").sha256(craw.tobytes()).hexdigest()[:16]})
    print(f"[s2] {args.shape} ({args.engine}): wall median {wall['median']:.3f} ms "
          f"(min {wall['min']:.3f} max {wall['max']:.3f})  "
          f"{rec['GFLOP_per_s']:.1f} GFLOP/s  states {states}", flush=True)
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return 0 if rec["all_completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
