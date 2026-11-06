# SPDX-License-Identifier: MIT
# t4_build_win.py -*- Python -*-
#
# The WINNING-RECIPE build for the T4 FFN variants: a verbatim port of
# imaggers/attn_impl/t3c_build_win.py (the full-core injection version) to the
# FFN's own build directory.  Injects and audits EVERY compute core of the
# image -- the stage-3 FFN image has 17 (1 norm + 4 gate/up + 6 down + 6 add),
# so the gate is 17 audit lines plus "GATE PASS (17 core(s))".
#
# usage: t4_build_win.py --tag t4_bk8 [--level -O1]
import argparse
import os
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "t4_out"
# Third-party locations: resolved, never hard-coded (imaggers/paths.py).
_root = pathlib.Path(__file__).resolve().parent
while not (_root / 'imaggers' / 'paths.py').is_file():
    if _root.parent == _root:
        raise RuntimeError('cannot locate the checkout root')
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from imaggers import paths as _paths  # noqa: E402

PEANO = _paths.aie_libs()[0]
MLIR_AIE_BIN = _paths.aie_libs()[1]
XRT_DIR = _paths.xrt_dir()
PY312 = _paths.python312()


def run(cmd, timeout=2400, env=None, cwd=None):
    kw = {} if cwd is None else {"cwd": str(cwd)}
    r = subprocess.run(cmd, capture_output=True, timeout=timeout, env=env, **kw)
    if r.returncode != 0:
        sys.stderr.write(r.stdout.decode(errors="replace")[-3000:])
        sys.stderr.write(r.stderr.decode(errors="replace")[-3000:])
        raise SystemExit("command failed: %s ..." % (cmd[0],))
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="-O1")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gu-regalloc", dest="gu_regalloc", default="greedy",
                    help="register allocator for the 4 gate/up cores only.  The "
                         "shipped image exposes 3 acq-clobbers on each of them "
                         "(a -1 operand hoisted above a kernel call); a "
                         "different allocator is the minimal recipe-level fix "
                         "and changes no arithmetic (same .ll, same -O1).")
    ap.add_argument("--gu-cores", dest="gu_cores",
                    default="core_0_3,core_0_4,core_0_5,core_1_2")
    args = ap.parse_args()

    prj = OUT / (args.tag + ".prj")
    assert prj.is_dir(), "run t4_compile.py --tag %s first (%s)" % (
        args.tag, prj)

    env = dict(os.environ)
    env["PATH"] = str(XRT_DIR) + os.pathsep + env.get("PATH", "")

    ck = OUT / (args.tag + "_ckpt")
    if ck.exists():
        shutil.rmtree(ck)
    ck.mkdir(parents=True)

    # ---- 1. one winner object per compute core, aiecc's own llc flags -----
    lls = sorted(prj.glob("opted_main_core_*.ll"))
    assert lls, "no opted_main_core_*.ll in %s" % prj
    cores = [p.stem.replace("opted_main_", "") for p in lls]
    objs = {}
    gu_set = set(x for x in args.gu_cores.split(",") if x)
    for ll, core in zip(lls, cores):
        obj = OUT / ("%s_%s.o" % (args.tag, core))
        ra = args.gu_regalloc if core in gu_set else "greedy"
        run([str(PEANO / "bin" / "llc.exe"), str(ll), args.level,
             "--march=aie2p", "--function-sections", "-stack-size-section",
             "-aie-stack-addrspace=5", "--filetype=obj",
             "--regalloc=" + ra, "-o", str(obj)], cwd=prj)
        objs[core] = obj
        print("[win] core object %s (%s B) from %s at llc %s regalloc %s"
              % (obj.name, format(obj.stat().st_size, ","), ll.name, args.level,
                 ra))
    print("[win] %d compute core(s): %s" % (len(cores), ", ".join(cores)))

    # ---- 2. checkpoint the aiecc graph at the core-object edge ------------
    xcl = OUT / (args.tag + ".xclbin")
    insts = OUT / (args.tag + ".bin")
    run([str(MLIR_AIE_BIN / "aiecc.exe"), "--unified", "--peano=%s" % PEANO,
         "-O", "0", "--tmpdir=%s" % prj, "--output-dir=%s" % prj,
         "--get-input-with-addresses", "--get-xclbin",
         "--xclbin-name=%s" % xcl, "--get-npu-insts",
         "--npu-insts-name=%s" % insts, "--cut=objects_{0}.o",
         "--checkpoint=%s" % ck, str(prj / "aie.mlir")])
    print("[win] checkpoint -> %s\\manifest.json" % ck.name)

    # ---- 3. swap EVERY frontier object for ours --------------------------
    stage = ck / "objects__0_.o"
    for core, obj in objs.items():
        cands = [p for p in stage.iterdir()
                 if p.is_dir() and p.name.endswith("_" + core)]
        assert len(cands) == 1, (core, [p.name for p in stage.iterdir()])
        frontier = cands[0] / ("%s.o" % cands[0].name)
        assert frontier.is_file(), frontier
        shutil.copyfile(obj, frontier)
        print("[win] injected %s -> %s"
              % (obj.name, frontier.relative_to(OUT)))

    # ---- 4. resume to xclbin ---------------------------------------------
    run([str(MLIR_AIE_BIN / "aiecc.exe"),
         "--resume=%s\\manifest.json" % ck], env=env)
    print("[win] xclbin %s (%s B)  insts %s (%s B)"
          % (xcl.name, format(xcl.stat().st_size, ","), insts.name,
             format(insts.stat().st_size, ",")))

    # ---- 5. audit EVERY core ELF (the acceptance gate) -------------------
    rc = 0
    n_pass = 0
    for core in cores:
        ecands = [p for p in prj.iterdir()
                  if p.is_dir() and p.name.startswith("elfs_")
                  and p.name.endswith("_" + core)]
        elf = (ecands[0] / ("%s.elf" % ecands[0].name)) if ecands \
            else prj / ("elfs_main_" + core) / ("elfs_main_" + core + ".elf")
        if not elf.is_file():
            print("[win] NO ELF FOR %s (%s) -- treated as a FAIL" % (core, elf))
            rc = 2
            continue
        lcands = [p for p in prj.iterdir()
                  if p.is_file() and p.name.startswith("peano-linked_")
                  and p.name.endswith("_" + core + ".ll")]
        ll = lcands[0] if lcands else prj / ("peano-linked_main_" + core + ".ll")
        cmd = [str(PY312), str(HERE / "step2d_audit.py"), str(elf),
               "--json", str(OUT / ("%s_%s_win_audit.json" % (args.tag, core)))]
        if ll.is_file():
            cmd += ["--ll", str(ll)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        print(r.stdout)
        print("[win] audit exit code %d (%s)" % (r.returncode, core))
        if r.returncode != 0:
            rc = r.returncode
        else:
            n_pass += 1
    print("[win] AUDIT GATE: %s (%d core(s): %s)"
          % ("PASS" if rc == 0 else "FAIL", len(cores), ", ".join(cores)))
    print("[win] audit lines PASS: %d / %d" % (n_pass, len(cores)))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
