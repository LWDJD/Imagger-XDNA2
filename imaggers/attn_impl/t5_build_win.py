# SPDX-License-Identifier: MIT
# t5_build_win.py -*- Python -*- (T5: t3c_build_win.py with OUT = t5_out)
#
# The WINNING-RECIPE build (STEP2D 2b / STEP4 6.2 / step6_build_win.py), with
# ONE fix that step 1 did not have: it injects and audits EVERY compute core of
# the image, not just core 0_2.
#
# Step 1's build (t3b_fused_p_build.log) replaced exactly one frontier object
# and audited exactly one ELF -- while the image has two compute tiles.  Its
# log even says so: "elfs_main_core_0_2.elf: 3 acquires".  The softmax core on
# (0,3) therefore ran the -O0 allocator and was never gated.  For a three-tile
# image that is three cores, so this script walks the checkpoint's frontier,
# compiles each core's  <prj>/opted_main_core_<tx>_<ty>.ll  at llc -O1
# --regalloc=greedy, swaps all of them in, resumes once, and audits every ELF.
#
# Usage: t3c_build_win.py --tag t3c_fused_o [--level -O1]
import argparse
import os
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "t5_out"
BLOCK = HERE.parent / "block"
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
        raise SystemExit(f"command failed: {cmd[0]} ...")
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="-O1")
    ap.add_argument("--tag", default="t5_u1")
    args = ap.parse_args()

    prj = OUT / (args.tag + ".prj")
    assert prj.is_dir(), f"run t3c_fused_run.py --compile first ({prj})"

    env = dict(os.environ)
    env["PATH"] = str(XRT_DIR) + os.pathsep + env.get("PATH", "")

    ck = OUT / (args.tag + "_ckpt")
    if ck.exists():
        shutil.rmtree(ck)
    ck.mkdir(parents=True)

    # ---- 1. one winner object per compute core, aiecc's own llc flags -----
    lls = sorted(prj.glob("opted_main_core_*.ll"))
    assert lls, f"no opted_main_core_*.ll in {prj}"
    cores = [p.stem.replace("opted_main_", "") for p in lls]
    objs = {}
    for ll, core in zip(lls, cores):
        obj = OUT / f"{args.tag}_{core}.o"
        run([str(PEANO / "bin" / "llc.exe"), str(ll), args.level,
             "--march=aie2p", "--function-sections", "-stack-size-section",
             "-aie-stack-addrspace=5", "--filetype=obj",
             "--regalloc=greedy", "-o", str(obj)], cwd=prj)
        objs[core] = obj
        print(f"[win] core object {obj.name} ({obj.stat().st_size:,} B) "
              f"from {ll.name} at llc {args.level}")

    # ---- 2. checkpoint the aiecc graph at the core-object edge ------------
    xcl = OUT / (args.tag + ".xclbin")
    insts = OUT / (args.tag + ".bin")
    run([str(MLIR_AIE_BIN / "aiecc.exe"), "--unified", f"--peano={PEANO}",
         "-O", "0", f"--tmpdir={prj}", f"--output-dir={prj}",
         "--get-input-with-addresses", "--get-xclbin",
         f"--xclbin-name={xcl}", "--get-npu-insts",
         f"--npu-insts-name={insts}", "--cut=objects_{0}.o",
         f"--checkpoint={ck}", str(prj / "aie.mlir")])
    print(f"[win] checkpoint -> {ck.name}\\manifest.json")

    # ---- 3. swap EVERY frontier object for ours --------------------------
    # The frontier directory carries aiecc's own symbol prefix (objects_main_<core>
    # for a Program whose top function is @main), so it is FOUND, not assumed.
    stage = ck / "objects__0_.o"
    for core, obj in objs.items():
        cands = [p for p in stage.iterdir()
                 if p.is_dir() and p.name.endswith("_" + core)]
        assert len(cands) == 1, (core, [p.name for p in stage.iterdir()])
        frontier = cands[0] / f"{cands[0].name}.o"
        assert frontier.is_file(), frontier
        shutil.copyfile(obj, frontier)
        print(f"[win] injected {obj.name} -> "
              f"{frontier.relative_to(OUT)}")

    # ---- 4. resume to xclbin ---------------------------------------------
    run([str(MLIR_AIE_BIN / "aiecc.exe"), f"--resume={ck}\\manifest.json"],
        env=env)
    print(f"[win] xclbin {xcl.name} ({xcl.stat().st_size:,} B)  "
          f"insts {insts.name} ({insts.stat().st_size:,} B)")

    # ---- 5. audit EVERY core ELF (the acceptance gate) -------------------
    rc = 0
    for core in cores:
        ecands = [p for p in prj.iterdir()
                  if p.is_dir() and p.name.startswith("elfs_")
                  and p.name.endswith("_" + core)]
        elf = (ecands[0] / f"{ecands[0].name}.elf") if ecands \
            else prj / f"elfs_main_{core}" / f"elfs_main_{core}.elf"
        if not elf.is_file():
            print(f"[win] NO ELF FOR {core} ({elf}) -- treated as a FAIL")
            rc = 2
            continue
        lcands = [p for p in prj.iterdir()
                  if p.is_file() and p.name.startswith("peano-linked_")
                  and p.name.endswith("_" + core + ".ll")]
        ll = lcands[0] if lcands else prj / f"peano-linked_main_{core}.ll"
        cmd = [str(PY312), str(BLOCK / "step2d_audit.py"), str(elf),
               "--json", str(OUT / f"{args.tag}_{core}_win_audit.json")]
        if ll.is_file():
            cmd += ["--ll", str(ll)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        print(r.stdout)
        print(f"[win] audit exit code {r.returncode} ({core})")
        if r.returncode != 0:
            rc = r.returncode
    print(f"[win] AUDIT GATE: {'PASS' if rc == 0 else 'FAIL'} "
          f"({len(cores)} core(s): {', '.join(cores)})")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
