# SPDX-License-Identifier: MIT
"""Where the non-Python pieces of this pipeline live.

Nothing in this repository hard-codes a path from the machine it was written
on.  Every external location is resolved in one order:

1. an environment variable, when the variable is set;
2. ``<checkout>/third_party/<name>``, the convention this repository documents;
3. a ``FileNotFoundError`` that names the variable and the directory it looked
   in, so the failure says what to do instead of dying somewhere downstream.

``third_party/`` is gitignored.  Put the real trees there, or link them there,
or point the variables somewhere else.  See README section 10.
"""
from __future__ import annotations

import os
import pathlib
import sys

# <checkout> is the directory that holds webui/ and imaggers/.
CHECKOUT = pathlib.Path(__file__).resolve().parent.parent
WORKSPACE = CHECKOUT.parent
THIRD_PARTY = CHECKOUT / "third_party"


def _first_file(env: str, candidates, what: str, hint: str) -> pathlib.Path:
    """First existing path wins; otherwise raise something readable."""
    override = os.environ.get(env)
    if override:
        p = pathlib.Path(override)
        if p.exists():
            return p
        raise FileNotFoundError(
            "%s is set to %s, which does not exist.  %s" % (env, p, hint))
    for c in candidates:
        if c and pathlib.Path(c).exists():
            return pathlib.Path(c)
    tried = "\n  ".join(str(c) for c in candidates if c)
    raise FileNotFoundError(
        "cannot find %s.\n"
        "Set %s, or put it at one of:\n  %s\n%s" % (what, env, tried, hint))


def _exe(root: pathlib.Path) -> pathlib.Path:
    """python.exe for a prefix, or the file itself if that is what was given.

    A variable may name either the executable or the directory that holds it;
    t22_run.ps1 makes the same assumption, so one value serves both.
    """
    if root.is_file():
        return root
    for cand in (root / "python.exe", root / "bin" / "python.exe",
                 root / "Scripts" / "python.exe"):
        if cand.is_file():
            return cand
    return root / "python.exe"


def python312() -> pathlib.Path:
    """The 3.12 interpreter: planning, packing, the schedule worker."""
    return _exe(_first_file(
        "ZIMAGE_PY312",
        [THIRD_PARTY / "python312",
         pathlib.Path(sys.executable).parent if sys.version_info[:2] == (3, 12) else None],
        "the Python 3.12 interpreter",
        "It needs numpy, torch (CPU), diffusers, ml_dtypes and onnx; see README 10.4."))


def python310() -> pathlib.Path:
    """The 3.10 interpreter: the device side.  pyxrt links python310.dll."""
    return _exe(_first_file(
        "ZIMAGE_PY310",
        [THIRD_PARTY / "python310",
         pathlib.Path(sys.executable).parent if sys.version_info[:2] == (3, 10) else None],
        "the Python 3.10 interpreter",
        "It must be 3.10.9: pyxrt.pyd links python310.dll.  See README 10.4."))


def xrt_dir() -> pathlib.Path:
    """XRT 2.21.0: pyxrt.pyd plus the DLLs it loads."""
    return _first_file(
        "ZIMAGE_XRT_DIR",
        [THIRD_PARTY / "xrt"],
        "the XRT runtime",
        "It ships in AMD's Ryzen AI 1.8.0 installer (RyzenAI_Deployment.1.8.0.nupkg)."
        "  See README 10.2.")


def aie_libs() -> tuple[pathlib.Path, pathlib.Path]:
    """(LLVM-AIE root, MLIR-AIE bin) -- only needed to COMPILE kernels.

    Running a fresh clone does not need these: the compiled .xclbin/.bin images
    are tracked in this repository.
    """
    llvm = _first_file(
        "ZIMAGE_LLVM_AIE",
        [THIRD_PARTY / "llvm-aie"],
        "the LLVM-AIE tree",
        "It ships in the mlir-aie 1.4.4 wheel; see README 10.2.")
    mlir = _first_file(
        "ZIMAGE_MLIR_AIE",
        [THIRD_PARTY / "mlir-aie" / "bin",
         THIRD_PARTY / "mlir-aie"],
        "the MLIR-AIE bin directory",
        "It ships in the mlir-aie 1.4.4 wheel; see README 10.2.")
    return llvm, mlir


def peano() -> pathlib.Path:
    """The Peano assembler, if a build wants it."""
    return _first_file(
        "ZIMAGE_PEANO",
        [THIRD_PARTY / "peano",
         THIRD_PARTY / "llvm-aie"],
        "the Peano assembler",
        "See README 10.2.")


def soft(fn):
    """Resolve without raising: for module-level constants.

    The product imports config.py before any job exists, so a missing third-party
    tree must not stop the import.  The value is still wrong, and the first job
    that tries to spawn it reports it; a preflight should call the strict
    functions above so the failure lands before the device is touched.
    """
    try:
        return fn()
    except FileNotFoundError:
        return None


def describe() -> dict:
    """What every resolver currently sees.  For a preflight to print."""
    out = {"checkout": str(CHECKOUT), "workspace": str(WORKSPACE)}
    for name, fn in (("python312", python312), ("python310", python310),
                     ("xrt", xrt_dir), ("peano", peano)):
        try:
            out[name] = str(fn())
        except FileNotFoundError as e:
            out[name] = "MISSING: " + str(e).splitlines()[0]
    return out


if __name__ == "__main__":
    for k, v in describe().items():
        print("%-10s %s" % (k, v))
