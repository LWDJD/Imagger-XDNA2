# SPDX-License-Identifier: MIT
"""t4_compile.py -- py3.12 compile of a T4 FFN design variant into t4_out/.

The variant is a module in imaggers/block (t4_ffn_<unit>.py) exposing the
@iron.jit entry point and its CompileTime knobs; this script specializes it,
compiles xclbin+insts next to a <tag>.prj directory (which t4_build_win.py then
re-injects and audits per core).

    py312 t4_compile.py --design t4_ffn_bk8 --tag t4_bk8 [--KG 8]
"""
from __future__ import annotations

import argparse
import importlib
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "t4_out"
ROOT = HERE.parent.parent
PYLIBS = ROOT.parent / "_pylibs"
# Third-party locations: resolved, never hard-coded (imaggers/paths.py).
_root = pathlib.Path(__file__).resolve().parent
while not (_root / 'imaggers' / 'paths.py').is_file():
    if _root.parent == _root:
        raise RuntimeError('cannot locate the checkout root')
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from imaggers import paths as _paths  # noqa: E402

XRT_DIR = _paths.xrt_dir()
K = 3840
N1 = 10240
M_DEF = 96


def _install_mkdir_mode_shim() -> None:
    """Neutralize mode 0o700 for directories this process creates (proven in
    imaggers/dit_block_v1_run.py and step4_run_ffn.py): aiecc's
    prefix_symbols_in_object makes a TemporaryDirectory INSIDE the .prj with
    os.mkdir(0o700), which this token then cannot re-enter -- the compile dies
    with PermissionError on symbols.map.  Shim it before any aie import."""
    orig_mkdir = os.mkdir

    def mkdir_compat(path, mode=0o777, *, dir_fd=None):
        if mode == 0o700:
            mode = 0o777
        return orig_mkdir(path, mode, dir_fd=dir_fd)

    os.mkdir = mkdir_compat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="t4_ffn_bk8")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--M", type=int, default=M_DEF)
    ap.add_argument("--KG", type=int, default=8)
    ap.add_argument("--kgu", type=int, default=64)
    ap.add_argument("--G-gu", dest="G_GU", type=int, default=4)
    ap.add_argument("--G-g3", dest="G_G3", type=int, default=6)
    ap.add_argument("--L-g3", dest="L_G3", type=int, default=10)
    ap.add_argument("--level", type=int, default=3)
    # T9 U-4a: the norm-only probe exposes how many times each row is
    # re-normalized; designs that do not declare it in PARAMS never see it.
    ap.add_argument("--sweeps", type=int, default=40)
    ap.add_argument("--NH", type=int, default=5)
    # T11b: norm cores sharing a row-tile (designs that do not declare G_NORM
    # in PARAMS never see it).
    ap.add_argument("--G-norm", dest="G_NORM", type=int, default=1)
    # T11b: chunk width (0 = the design's own chunk_cols, 960).  n_chunks =
    # K/Kc is how often every row is re-read, so this is the replay knob.
    ap.add_argument("--Kc", type=int, default=960)
    # T11c: corner-turn fifo depth (the A-stream decoupling knob).
    ap.add_argument("--midb-depth", dest="midB_depth", type=int, default=4)
    # T12 U-1: cost attribution switches for the norm-core probe (only designs
    # that declare T12COST in PARAMS see them; TIMING ONLY images).
    ap.add_argument("--do-norm", dest="DO_NORM", type=int, default=1)
    ap.add_argument("--do-adaln", dest="DO_ADALN", type=int, default=1)
    ap.add_argument("--do-copy", dest="DO_COPY", type=int, default=1)
    ap.add_argument("--ina-depth", dest="inA_depth", type=int, default=1)
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _install_mkdir_mode_shim()
    sys.path.append(str(HERE))
    sys.path.append(str(PYLIBS))
    sys.path.append(str(PYLIBS / "mlir_aie" / "python"))
    sys.path.append(str(HERE.parent / "kernels"))
    os.environ["PEANO_INSTALL_DIR"] = str(PYLIBS / "llvm-aie" / "llvm-aie")
    os.environ["NPU_CACHE_HOME"] = str(HERE / "_npu_cache_t4")

    from aie.iron import set_current_device
    from aie.iron.device import NPU2

    if (XRT_DIR / "xclbinutil.exe").is_file():
        os.environ["PATH"] = str(XRT_DIR) + os.pathsep + os.environ.get("PATH", "")

    D = importlib.import_module(args.design)
    OUT.mkdir(parents=True, exist_ok=True)
    xcl = OUT / (args.tag + ".xclbin")
    ins = OUT / (args.tag + ".bin")
    kw = dict(M=args.M, K=K, N1=N1, m=16, k=8, n=64, n3=64,
              Kc=args.Kc, G_GU=args.G_GU, G_G3=args.G_G3, L_G3=args.L_G3,
              LEVEL=args.level, G3_MODE=2)
    if hasattr(D.step3_ffn, "specialize"):
        params = getattr(D, "PARAMS", {})
        if "KG" in params:
            kw["KG"] = args.KG
        if "kgu" in params:
            kw["kgu"] = args.kgu
        if "sweeps" in params:
            kw["sweeps"] = args.sweeps
        if "NH" in params:
            kw["NH"] = args.NH
        if "G_NORM" in params:
            kw["G_NORM"] = args.G_NORM
        if "midB_depth" in params:
            kw["midB_depth"] = args.midB_depth
        if "T12COST" in params:
            kw["DO_NORM"] = args.DO_NORM
            kw["DO_ADALN"] = args.DO_ADALN
            kw["DO_COPY"] = args.DO_COPY
            kw["inA_depth"] = args.inA_depth
    set_current_device(NPU2())
    t0 = time.perf_counter()
    spec = D.step3_ffn.specialize(use_cache=False, **kw)
    xclbin, insts = spec.compile(xclbin_path=str(xcl), inst_path=str(ins))
    print("[compile] %s in %.1f s -> %s (%s B) + %s (%s B)"
          % (args.tag, time.perf_counter() - t0, pathlib.Path(xclbin).name,
             format(pathlib.Path(xclbin).stat().st_size, ","),
             pathlib.Path(insts).name,
             format(pathlib.Path(insts).stat().st_size, ",")), flush=True)
    prj = OUT / (args.tag + ".prj")
    lls = sorted(prj.glob("opted_main_core_*.ll"))
    print("[compile] prj %s: %d core(s): %s"
          % (prj.name, len(lls), ", ".join(p.stem.replace("opted_main_", "")
                                           for p in lls)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
