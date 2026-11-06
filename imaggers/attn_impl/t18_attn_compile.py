# SPDX-License-Identifier: MIT
"""T18: compile a t7_ip variant for a (query-block M, key count ND) pair.

The delivered t7_ip is M=96 queries x ND=96 keys.  An arbitrary sequence length
SEQ needs the ip image keyed on the FULL sequence (attention is dense), so this
compiles the same design with M = the query-block height and ND = SEQ:
    192x128 -> SEQ=128: M=64, ND=128   (2 query blocks)
    256x256 -> SEQ=288: M=16, ND=288   (18 query blocks)
L1 (measured budget in the design): QP + KB + C + stack
    M=64,ND=128: 16,384 + 18,432 + 16,384 + 6,144 = 57,344 <= 65,536
    M=16,ND=288:  4,096 + 41,472 +  9,216 + 6,144 = 60,928 <= 65,536
"""
from __future__ import annotations

import argparse
import importlib
import os
import pathlib
import sys
import time

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
PYLIBS = HERE.parent.parent.parent / "_pylibs"
OUT = HERE / "t5_out"


def _install_mkdir_mode_shim() -> None:
    orig = os.mkdir

    def mkdir_compat(path, mode=0o777, *, dir_fd=None):
        if mode == 0o700:
            mode = 0o777
        return orig(path, mode, dir_fd=dir_fd)

    os.mkdir = mkdir_compat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--M", type=int, required=True, help="query-block height")
    ap.add_argument("--KD", type=int, default=128)
    ap.add_argument("--ND", type=int, required=True, help="keys = SEQ")
    ap.add_argument("--heads", type=int, default=30)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _install_mkdir_mode_shim()
    sys.path.append(str(HERE))
    sys.path.append(str(PYLIBS))
    sys.path.append(str(PYLIBS / "mlir_aie" / "python"))
    os.environ["PEANO_INSTALL_DIR"] = str(PYLIBS / "llvm-aie" / "llvm-aie")
    os.environ["NPU_CACHE_HOME"] = str(HERE / "t18_cache")
    # xclbinutil lives in the shipped XRT stack; aiecc looks it up on PATH.
    XRT_DIR = _paths.xrt_dir()
    if (XRT_DIR / "xclbinutil.exe").is_file():
        os.environ["PATH"] = str(XRT_DIR) + os.pathsep + os.environ.get("PATH", "")
    (HERE / "t18_cache").mkdir(parents=True, exist_ok=True)
    from aie.iron import set_current_device
    from aie.iron.device import NPU2

    D = importlib.import_module("t7_attn_split")
    set_current_device(NPU2())
    OUT.mkdir(exist_ok=True)
    xcl, ins = OUT / (a.tag + ".xclbin"), OUT / (a.tag + ".bin")
    t0 = time.perf_counter()
    spec = D.t7_ip.specialize(HEADS=a.heads, M=a.M, KD=a.KD, ND=a.ND,
                              use_cache=False)
    spec.compile(xclbin_path=str(xcl), inst_path=str(ins))
    print("[compile] %s (M=%d KD=%d ND=%d): %.1f s -> %s (%s B) + %s (%s B)"
          % (a.tag, a.M, a.KD, a.ND, time.perf_counter() - t0,
             xcl.name, format(xcl.stat().st_size, ","), ins.name,
             format(ins.stat().st_size, ",")), flush=True)
    prj = OUT / (a.tag + ".prj")
    print("[compile] prj %s exists=%s" % (prj.name, prj.is_dir()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
