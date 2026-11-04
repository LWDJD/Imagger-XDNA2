# t21_3b2_plan.py -*- Python -*-
#
# T21-3B2: the table-driven build + per-geometry gate harness for the 92
# second-tier VAE xclbins (55 conv + 24 GroupNorm + 13 upsample).
#
# WHAT IT IS
# ----------
# One row per geometry, and per row exactly this pipeline:
#
#     disk check  ->  compile (py312/aie)  ->  prep (real activations, py312)
#                 ->  device run (py310/pyxrt)  ->  per-geometry gate
#                 ->  results JSON  ->  delete the .prj (and the bulky io/raw)
#
# The geometry table comes from t21_3b2_geom.py (derived from the real
# decoder, frozen).  Everything that is CompileTime in the design is in the
# row: conv (M, K, N), GN (C, S), UP (C, S).  Nothing here is hand-typed.
#
# THE FOUR MEASURED PITFALLS ARE ASSERTIONS, NOT COMMENTS
# -----------------------------------------------------
#   (1) W is quantized with bfp.encode's FLOOR default, A with the RNE bf16
#       cast (conv_even).  Mixing them silently *improves* PSNR (3b1: 37.41
#       -> 41.78 dB) = wrong.  assert_contracts() pins both sides and shows
#       the two modes differ on a crafted vector.
#   (2) The GroupNorm GB arena is a BF16 stream (2 B/value).  Writing float32
#       bytes overflows/shifts it.  build_gb() asserts the byte count and
#       refuses a 4-byte write; an existing artifact is cross-checked.
#   (3) The upsample output is C*H*W*8 bytes (4 replicated elements * 2 B),
#       NOT *4.  up_out_bytes() asserts it and the stored up artifacts are
#       cross-checked (their .npy payload == C*S*8).
#   (4) ulp is measured against the MAGNITUDE BOUND |x-mu|*rstd*|gamma|+|beta|,
#       never against the result itself: near-zero results report 512 ulp
#       against themselves while the arithmetic is 1 ulp.  ulp_pair() asserts
#       the bound on a synthetic near-zero case.
#
# DISK PROTOCOL
# -------------
# ~0.9 GiB free on this box.  Before EVERY build, prep and device run the free
# space is asserted >= --min-free-mb (default 800 MiB); below that the harness
# aborts, writes out/t21_3b2/ABORT_disk.json and leaves the scene intact (no
# cleanup, no retry).  Peak is one geometry at a time: the largest single
# conv (M=65536, K=1184) needs ~160 MB of A bytes plus ~34 MB of references,
# so the plan is per-geometry cleanup, not accumulation.
#
# USAGE
# -----
#   py312 t21_3b2_plan.py --dry-run            # 92 rows, no compile/device
#   py312 t21_3b2_plan.py --list               # compact table
#   py312 t21_3b2_plan.py --selftest           # 4 pitfalls + replay 18 artifacts
#   py312 t21_3b2_plan.py --calibrate          # gate-cost constants
#   py312 t21_3b2_plan.py --run --only 12      # one geometry, full pipeline
#   py312 t21_3b2_plan.py --run --from 56      # GN onwards
#   py312 t21_3b2_plan.py --dry-run --strict   # exit 3 if any row needs a
#                                              # design variant (not a recompile)
#
# What --run requires: py312 + mlir_aie (compile), py310 + pyxrt (device), a
# single-tenant NPU, and the driver376 xclbinutil (PATH is set here exactly as
# vaeg_run.py:186-191 does).
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np

import t21_3b2_geom as G

# ---------------------------------------------------------------------------
# paths and constants
# ---------------------------------------------------------------------------
HERE = pathlib.Path(__file__).resolve().parent          # ...\imaggers\vae
ROOT = HERE.parent.parent.parent                        # ...\public_work
PYLIBS = ROOT / "_pylibs"
PEANO = PYLIBS / "llvm-aie" / "llvm-aie"
PORT = ROOT / "zimage-port"
MODELS = PORT / "_models" / "Z-Image-Turbo"
OUT = HERE / "out"
ART = OUT / "t21_3b2"                                   # new artifacts only
XRT_RUNNER = HERE / "vaeg_run_xrt.py"
PY310 = pathlib.Path(r"D:\Users\15081\AppData\Local\GAIA\python\python.exe")
XRT_DIR = pathlib.Path(
    r"D:\Project\deepseek_harness\test7\WStmp\driver376\npu_mcdm_stack_prod")

SCALING = 0.3611
SHIFT = 0.1159
GROUPS = 32
EPS = 1e-6

TILE = dict(m=16, k=8, n=16)

# compile wallclock per geometry, the only measured samples we have
# (T21-3A-AUDIT.md section 3.1: 5 conv in 11.3 min, 6 gn_stream in 39 s,
# 3 up_kernel in 13 s)
COMPILE_S = {"conv": 138.0, "gn": 6.5, "up": 4.3}
COLD_FACTOR = 1.5
DEFAULT_MIN_FREE_MB = 800.0
MB = 1024.0 * 1024.0

CALIB_DEFAULT = dict(ms_per_tile=0.080, ms_per_pixel=0.0075)
DEFAULT_RESULTS = ART / "results.json"

# U0: the vectorized device model (t21_3b2_plan.bfp_gemm_model_v).  Enabled
# only after bit-identity against the 5 stored device-verified ref_model
# arrays and against the scalar path on the real geometries; the env var is
# the break-glass switch back to the per-tile scalar reference.
FAST_REF = os.environ.get("T21_3B2_FAST_REF", "1") != "0"

# pyxrt run.wait()'s default returns ERT_CMD_STATE_TIMEOUT long before a big
# conv dispatch finishes (row 3, a 3.5 s kernel, times out; with an explicit
# wait it completes in 3.52 s).  Ten minutes is ~170x the largest measured
# dispatch and still a real bound.
DEVICE_WAIT_MS = int(os.environ.get("T21_3B2_WAIT_MS", "600000"))


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _free_bytes() -> int:
    return shutil.disk_usage(str(OUT)).free


class DiskAbort(RuntimeError):
    """Free space fell under the protocol threshold: stop, leave the scene."""


class DispatchTimeout(RuntimeError):
    """The dispatch exceeded the ERT's fixed ~7 s watchdog (state 8).

    This is a platform limit, not an infrastructure hiccup: the command is
    aborted by the driver/firmware at ~7.0 s regardless of the wait timeout
    passed to pyxrt (measured: 7021/7022/7026/7041 ms across attempts, with
    run.wait(600000) and with state polling, and after inserting a foreign
    image launch -- the documented state-8 workaround -- which does not
    change it).  A row that hits this cannot be gated as a single dispatch.
    """


def require_disk(min_free_mb: float, item: dict | None, stage: str) -> int:
    free = _free_bytes()
    if free < min_free_mb * MB:
        ART.mkdir(parents=True, exist_ok=True)
        scene = dict(ts=_ts(), stage=stage, free_bytes=int(free),
                     free_mb=round(free / MB, 1),
                     min_free_mb=min_free_mb,
                     item=None if item is None else
                     dict(id=item["id"], kind=item["kind"], name=item["name"]),
                     note="abort by disk protocol; artifacts left in place")
        (ART / "ABORT_disk.json").write_text(json.dumps(scene, indent=1),
                                             encoding="utf-8")
        print(f"[disk] ABORT: free {free / MB:.1f} MiB < "
              f"{min_free_mb:.0f} MiB at stage {stage}"
              f"{'' if item is None else ' item ' + str(item['id'])}")
        print(f"[disk] scene written -> {ART / 'ABORT_disk.json'}")
        raise DiskAbort(stage)
    return free


# The 800 MiB line above is the ACCUMULATION guard: it is checked before a row
# starts.  A row's own peak is known (predict()'s peak_bytes: A stream + B +
# bf16 reference + raw), and once prep has written the io the free space is
# necessarily lower by that amount -- for the largest geometry, 359.6 MiB.  A
# single flat 800 MiB on every intra-row stage would therefore abort every
# M=65536 row by construction.  So intra-row stages use "this row's peak plus
# headroom", with a floor that still leaves the machine usable.
STAGE_FLOOR_MB = 350.0
STAGE_HEADROOM_MB = 40.0


def require_disk_stage(item: dict, stage: str,
                       floor_mb: float = STAGE_FLOOR_MB) -> int:
    """Intra-row check: free >= max(floor, this row's predicted peak + slack)."""
    pred = item.get("pred") or predict(item, load_calib())
    need = max(floor_mb, pred["peak_bytes"] / MB + STAGE_HEADROOM_MB)
    return require_disk(need, item, stage)


def _rm(path: pathlib.Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except OSError:
            pass


def _install_mkdir_mode_shim() -> None:
    """Neutralize mode 0o700 for directories this process creates."""
    orig = os.mkdir

    def mkdir_compat(path, mode=0o777, *, dir_fd=None):
        if mode == 0o700:
            mode = 0o777
        return orig(path, mode, dir_fd=dir_fd)

    os.mkdir = mkdir_compat


def _force_utf8() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# the plan: 92 rows from the frozen table minus what out/ already has
# ---------------------------------------------------------------------------

def scan_built(outdir: pathlib.Path = OUT) -> dict:
    """What out/ already contains, keyed the way the designs key it.

    conv : (M, K, N)                 -- the burned-in GEMM geometry
    gn   : (C, S)                    -- single and stream both count
    up   : (C, S) -> set(exp)        -- only kernel/full are the full
                                        nearest-2x op (rows/elems are other
                                        walks, not the decoder's op)
    """
    conv, gn, up = set(), set(), {}
    for p in sorted(outdir.glob("*.xclbin")):
        n = p.stem
        m = re.match(r"^cg_(.+)_M(\d+)_K(\d+)_N(\d+)$", n)
        if m:
            conv.add((int(m.group(2)), int(m.group(3)), int(m.group(4))))
            continue
        m = re.match(r"^vaeg_gn_(single|stream)_(.+)_C(\d+)_H(\d+)_W(\d+)$", n)
        if m:
            gn.add((int(m.group(3)), int(m.group(4)) * int(m.group(5))))
            continue
        m = re.match(r"^vaeg_up_(kernel|rows|full|elems)_(.+)_C(\d+)_H(\d+)_W(\d+)$", n)
        if m:
            up.setdefault((int(m.group(3)), int(m.group(4)) * int(m.group(5))),
                          set()).add(m.group(1))
    return dict(conv=conv, gn=gn, up=up)


def conv_stem(M: int, K: int, N: int) -> str:
    return f"cg_M{M}_K{K}_N{N}"


def gn_stem(C: int, S: int, mode: str) -> str:
    return f"gn_{mode}_C{C}_S{S}"


def up_stem(C: int, S: int) -> str:
    return f"up_C{C}_S{S}"


def _gn_hw(size: str, S: int) -> tuple:
    """(H, W) of a GroupNorm row: S == H*W and H/W follows the size's aspect.

    The frozen GN rows carry (C, S, mode) only, but both the design factory
    and the prep path need the spatial shape (the kernel tiles by H, W and the
    io meta stores them).  S and the layer's aspect pin it down uniquely across
    every row of this table: 128x128/S=4096 -> 64x64, 144x128/S=288 -> 18x16,
    192x128/S=384 -> 24x16, 128x256/S=512 -> 16x32, 128x256/S=2048 -> 32x64.
    """
    h, w = [int(v) for v in size.split("x")]
    ideal = (S * h / w) ** 0.5
    cands = sorted({max(1, int(ideal) + d) for d in (-2, -1, 0, 1, 2)})
    best = None
    for H in cands:
        if S % H:
            continue
        W = S // H
        err = abs(H / W - h / w)
        if best is None or err < best[0]:
            best = (err, H, W)
    if best is None:
        raise ValueError(f"cannot factor S={S} with the aspect of {size}")
    return best[1], best[2]


VB = OUT / "t21_3b2vf"


def variant_index() -> dict:
    """id -> the 3b2vf verdict for that row (U1: 92/92 as a gate).

    T21-3B2VF gated 40 of the 92 rows with a DESIGN VARIANT instead of the
    3b2 factories (18 M-block conv, 19 sub-plane GroupNorm, 3 row-strip
    upsample).  Its per-row json is the verdict; this reads it from disk so
    the plan never hand-copies a number.
    """
    out = {}
    if not VB.is_dir():
        return out
    for p in sorted(VB.glob("[0-9][0-9]_*.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if r.get("id") is None:
            continue
        out[int(r["id"])] = dict(name=r.get("name"), kind=r.get("kind"),
                                 stem=r.get("stem"),
                                 pass_=bool(r.get("pass_")),
                                 file=str(p.name))
    return out


def make_items() -> tuple:
    """(all_items, missing_items).

    all_items keeps the already-built rows so the reconciliation table can be
    produced from the same code; missing_items is what this unit builds (1..92).
    """
    f = G.frozen_core()
    built = scan_built()
    all_items = []
    for r in f["conv"]:
        key = (r["M"], r["K"], r["N"])
        item = dict(kind="conv", size=r["size"], module=r["module"],
                    name=conv_stem(*key), key=key, built=key in built["conv"],
                    params=r)
        all_items.append(item)
    for r in f["gn"]:
        key = (r["C"], r["S"])
        H, W = _gn_hw(r["size"], r["S"])
        item = dict(kind="gn", size=r["size"], module=r["module"],
                    name=gn_stem(r["C"], r["S"], r["mode"]), key=key,
                    built=key in built["gn"], params=dict(r, H=H, W=W))
        all_items.append(item)
    for r in f["up"]:
        key = (r["C"], r["S"])
        item = dict(kind="up", size=r["size"], module=r["module"],
                    name=up_stem(r["C"], r["S"]), key=key,
                    built=bool(built["up"].get(key, set()) & {"kernel", "full"}),
                    params=r)
        all_items.append(item)

    # buildability with the EXISTING design factories (3b2 is a recompile
    # unit; a design variant is a different, bigger unit)
    for it in all_items:
        r = it["params"]
        if it["kind"] == "gn" and not r["chunk_ok"]:
            it["buildable"] = "design_gap"
            it["gap_reason"] = r["note"]
        elif it["kind"] == "up" and not r["fits_existing"]:
            it["buildable"] = "design_gap"
            it["gap_reason"] = (
                f"L1 {r['l1_bytes'] / 1024:.1f} KB for in+out objects "
                f"(S={r['S']}); needs row-strip chunking")
        else:
            it["buildable"] = "existing"
            it["gap_reason"] = ""

    missing = [it for it in all_items if not it["built"]]
    # stable numbering: conv first, then gn, then up (kind-major)
    missing.sort(key=lambda it: (["conv", "gn", "up"].index(it["kind"]),
                                 all_items.index(it)))
    for i, it in enumerate(missing, 1):
        it["id"] = i
    # U1: a row blocked on the 3b2 factories is buildable if a VARIANT factory
    # already produced and gated it (T21-3B2VF).  The id -> verdict map comes
    # from the variant unit's own per-row json, never from this file.
    vv = variant_index()
    for it in missing:
        it["variant"] = None
        it["variant_pass"] = None
        v = vv.get(it["id"])
        if v and v["name"] == it["name"]:
            it["variant"] = v["stem"]
            it["variant_pass"] = v["pass_"]
            it["buildable"] = "variant"
            it["gap_reason"] = ""
    return all_items, missing


def closure_rows(missing: list) -> dict:
    """Every one of the 92 rows must have a PASS verdict on disk.

    52 rows were gated by the 3b2 harness (out/t21_3b2/<stem>_results.json),
    40 by the variant driver (out/t21_3b2vf/NN_<name>.json).  A row with no
    verdict, or with a non-PASS one, is a closure failure -- the 92/92 gate.
    """
    rows, bad, missing_files = [], [], []
    vv = variant_index()
    for it in missing:
        if it.get("variant"):
            p = VB / vv[it["id"]]["file"]
            src = "variant"
        else:
            p = ART / f"{it['name']}_results.json"
            src = "3b2"
        if not p.is_file():
            missing_files.append(it["id"])
            rows.append(dict(id=it["id"], name=it["name"], kind=it["kind"],
                             source=src, file=str(p), pass_=None))
            bad.append(it["id"])
            continue
        res = json.loads(p.read_text(encoding="utf-8"))
        ok = bool(res.get("pass_"))
        rows.append(dict(id=it["id"], name=it["name"], kind=it["kind"],
                         source=src, file=str(p.name), pass_=ok,
                         stem=res.get("stem")))
        if not ok:
            bad.append(it["id"])
    return dict(rows=rows, n=len(rows), n_pass=len(rows) - len(bad),
                bad=bad, missing_files=missing_files,
                green=bool(not bad))


def load_calib() -> dict:
    p = ART / "calib.json"
    if p.is_file():
        try:
            c = json.loads(p.read_text(encoding="utf-8"))
            return dict(CALIB_DEFAULT, **{k: float(c[k]) for k in
                                          ("ms_per_tile", "ms_per_pixel")
                                          if k in c})
        except Exception:  # noqa: BLE001
            pass
    return dict(CALIB_DEFAULT)


def predict(item: dict, calib: dict) -> dict:
    """Compile/gate/disk prediction for one row (modelled; treat as +/-50%)."""
    r = item["params"]
    kind = item["kind"]
    compile_s = COMPILE_S[kind]
    if kind == "conv":
        A = r["M"] * r["K"] * 2
        B = (r["K"] * r["N"] // 8) * 9
        C = r["M"] * r["N"] * 2
        # the gate needs ONE reference: the device-exact model rounded to
        # bf16 (uint16).  ref_quant / ref_f32w are attribution-only and are
        # opt-in (--full-refs), because at M=65536 they would add 200 MB.
        refs = r["M"] * r["N"] * 2 + 4096
        io_bytes = A + B + refs
        peak = io_bytes + C + 3 * MB
        tiles = ((r["M"] // TILE["m"]) * (r["N"] // TILE["n"])
                 * (r["K"] // TILE["k"]))
        gate_s = (tiles * calib["ms_per_tile"] / 1000.0
                  + r["M"] * calib["ms_per_pixel"] / 1000.0
                  + 2.5 + (A + B) / 80e6 + 1.5)
    elif kind == "gn":
        # X stream + the bf16-rounded model reference; the oracle and the
        # device-formula models are reduced to numbers at prep (see prep_gn).
        nbytes = r["C"] * r["S"] * 2
        io_bytes = 2 * nbytes + 4 * r["C"] + 4096
        peak = io_bytes + nbytes + 2 * MB
        gate_s = 1.0 + r["C"] * r["S"] * 4e-6 + 2.5
    else:
        nbytes = r["C"] * r["S"] * 2
        io_bytes = nbytes + 4 * nbytes
        peak = io_bytes + 4 * nbytes + 1 * MB
        gate_s = 0.7 + r["C"] * r["S"] * 2e-6 + 2.5 + 4 * nbytes / 80e6
    return dict(compile_s=compile_s, compile_cold_s=compile_s * COLD_FACTOR,
                gate_s=gate_s, io_bytes=io_bytes, peak_bytes=peak)


def plan_rows() -> tuple:
    all_items, missing = make_items()
    calib = load_calib()
    cum = 0.0
    cum_cold = 0.0
    for it in missing:
        p = predict(it, calib)
        it["pred"] = p
        cum += p["compile_s"]
        cum_cold += p["compile_cold_s"]
        p["cum_s"] = cum
        p["cum_cold_s"] = cum_cold
    return all_items, missing, calib


# ---------------------------------------------------------------------------
# the four measured pitfalls as executable contracts
# ---------------------------------------------------------------------------

def _src(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def assert_contracts(deep: bool = False) -> list:
    """The four 3b1 pitfalls.  Returns [(name, ok, detail), ...].

    deep=False: source-level + synthetic checks (no aie import), used by
    --dry-run.  deep=True: adds the aie-level assertions (bfp.encode's default
    rounding is floor and the two modes really differ); used by --selftest and
    inside --run, where aie is imported anyway.
    """
    out = []

    # (1) W floor / A conv_even ------------------------------------------
    ok = True
    detail = []
    vn = _src(HERE / "vae_npu.py")
    body = vn[vn.index("def pack_b_blocks"):]
    body = body[:body.index("\n\n\n")] if "\n\n\n" in body else body
    has_encode = "B.encode(" in body
    no_rounding = "rounding=" not in body
    mine = _src(HERE / "t21_3b2_plan.py")
    # the needles are assembled at runtime: a literal would match itself
    needle_pack = "_pack" + "_b("
    needle_q = "bfp." + "quantize"
    own_ok = (needle_pack in mine and "astype(bfloat16)" in mine
              and needle_q not in mine)
    ok &= has_encode and no_rounding and own_ok
    detail.append(f"pack_b_blocks->B.encode(floor default)={has_encode}, "
                  f"explicit rounding= absent={no_rounding}, "
                  f"harness uses _pack_b + bf16 cast={own_ok}")
    # the two modes must actually differ, or the whole pitfall is imaginary
    x = np.array([1.0 + 2 ** -8 + 2 ** -12, 1.0 + 2 ** -9, 0.1, 3.3],
                 np.float32)
    u = x.view(np.uint32)
    rne = ((u + (((u >> 16) & 1) + 0x7FFF)) & np.uint32(0xFFFF0000)).view(np.float32)
    floor = (u & np.uint32(0xFFFF0000)).view(np.float32)
    differ = bool((rne != floor).any())
    ok &= differ
    detail.append(f"RNE vs floor bf16 differ on the probe vector={differ}")
    if deep:
        # BOTH roots: mlir_aie's own deps (cloudpickle, ...) live in _pylibs,
        # exactly as vae_npu.py:79-82 arranges them.
        for _p in (str(PYLIBS), str(PYLIBS / "mlir_aie" / "python")):
            if _p not in sys.path:
                sys.path.append(_p)
        os.environ.setdefault("PEANO_INSTALL_DIR", str(PEANO))
        from aie.utils import bfp as B
        import inspect
        dflt = inspect.signature(B.encode).parameters["rounding"].default
        # bfp16ebs8 works in blocks of 8 along the last axis
        probe = np.array([1.0 + 2 ** -8 + 2 ** -12, 1.0 + 2 ** -9, 0.1, 3.3,
                          -2.7, 0.5, 1e-3, 7.9], np.float32)
        enc_floor = np.asarray(B.encode(probe)).view(np.uint8)
        enc_even = np.asarray(B.encode(probe, rounding="conv_even")).view(
            np.uint8)
        ok &= (dflt == "floor") and bool((enc_floor != enc_even).any())
        detail.append(f"bfp.encode rounding default={dflt!r}, "
                      f"floor != conv_even on probe="
                      f"{bool((enc_floor != enc_even).any())}")
    out.append(("P1 W=floor / A=conv_even", ok, "; ".join(detail)))

    # (2) GB arena is a bf16 stream --------------------------------------
    ok = True
    detail = []
    for C in (128, 256, 512):
        stat_off = (4 * C + 63) // 64 * 64
        gb_bytes, off = G_GB(C, True)
        ok &= (gb_bytes == stat_off + 16) and (off == stat_off)
        ok &= (4 * C <= gb_bytes)
    C = 512
    gb_bytes, stat_off = G_GB(C, True)
    ok &= (stat_off >= 4 * C)
    ok &= (gb_bytes - stat_off == 16)
    art = OUT / "vaeg_gn_stream_gn512_8_C512_H8_W8_io.npz"
    cross = None
    if art.is_file():
        cross = _npz_member_size(art, "in1_bytes")
        ok &= (cross == gb_bytes)
    detail.append(f"gb_layout(C=512)={gb_bytes}B (2C gamma + 2C beta + 16 "
                  f"stats); stored artifact in1_bytes={cross}; a float32 4C "
                  f"write would shift gamma/beta")
    out.append(("P2 GB arena = bf16 stream", ok, "; ".join(detail)))

    # (3) upsample output length = C*H*W*8 bytes -------------------------
    ok = True
    detail = []
    for C, H, W in ((512, 8, 8), (512, 16, 16), (256, 32, 32), (256, 64, 128)):
        n8 = up_out_bytes(C, H, W)
        ok &= (n8 == C * H * W * 8) and (n8 == 2 * (C * H * W * 4))
        ok &= (n8 != C * H * W * 4)
    for name, C, H, W in (("vaeg_up_kernel_up512_16_C512_H16_W16", 512, 16, 16),
                          ("vaeg_up_kernel_up256_32_C256_H32_W32", 256, 32, 32)):
        p = OUT / f"{name}_Y_raw.npy"
        if p.is_file():
            payload = npy_payload_bytes(p)
            ok &= (payload == C * H * W * 8)
            detail.append(f"stored {name} payload={payload}B == C*S*8")
    detail.append("C*H*W*8 = 4 replicated elements x 2 B, NOT C*H*W*4")
    out.append(("P3 UP out = C*H*W*8 B", ok, "; ".join(detail)))

    # (4) ulp against the magnitude bound --------------------------------
    ok = True
    detail = []
    ref = np.array([0.0, 1.0], np.float64)
    got = np.array([1e-3, 1.0], np.float64)
    mag = np.array([1.0, 1.0], np.float64)
    raw, bounded = ulp_pair(got, ref, mag)
    ok &= (raw > 1e6) and (bounded < 2.0)
    ok &= bool(gate_ulp_ok(bounded)) and not bool(gate_ulp_ok(raw))
    detail.append(f"synthetic near-zero: raw={raw:.3e} ulp, "
                  f"bounded={bounded:.3e} ulp; gate(bounded)="
                  f"{bool(gate_ulp_ok(bounded))}, gate(raw)="
                  f"{bool(gate_ulp_ok(raw))}")
    out.append(("P4 ulp vs magnitude bound", ok, "; ".join(detail)))
    return out


def G_GB(C: int, with_stats: bool = True) -> tuple:
    """Mirror of vaeg_design.gb_layout (no aie import in --dry-run)."""
    stats_off = (4 * C + 63) // 64 * 64
    return (stats_off + 16 if with_stats else 4 * C), stats_off


def up_out_bytes(C: int, H: int, W: int) -> int:
    """The upsample output BO: 4 replicated elements * 2 bytes per input."""
    return C * H * W * 4 * 2


def npy_payload_bytes(path: pathlib.Path) -> int:
    """Payload size of a .npy without loading it (header is 128 B here)."""
    with open(path, "rb") as f:
        head = f.read(128)
    return path.stat().st_size - len(head) if head[:6] == b"\x93NUMPY" else -1


def _npz_member_size(path: pathlib.Path, member: str) -> int:
    """Payload size of one member of a .npz (no full load)."""
    with zipfile.ZipFile(path) as z:
        info = z.getinfo(f"{member}.npy")
        with z.open(f"{member}.npy") as f:
            head = f.read(128)
        hdr = _npy_header_len(head)
    return info.file_size - hdr


def _npy_header_len(head: bytes) -> int:
    if head[:6] != b"\x93NUMPY":
        return 0
    return 128


def ulp_pair(got, ref, mag) -> tuple:
    """(max raw ulp, max ulp against the magnitude bound)."""
    ref = np.asarray(ref, np.float64)
    got = np.asarray(got, np.float64)
    mag = np.asarray(mag, np.float64)
    ulp_r = np.maximum(_bf16_ulp(ref), 2.0 ** -126)
    ulp_m = np.maximum(_bf16_ulp(mag), 2.0 ** -126)
    d = np.abs(got - ref)
    return float((d / ulp_r).max()), float((d / ulp_m).max())


def _bf16_ulp(x) -> np.ndarray:
    xf = np.asarray(x, np.float64)
    xf = np.where(xf == 0.0, np.float64(2.0 ** -126), np.abs(xf))
    exp = np.floor(np.log2(xf)).astype(np.int64)
    return np.exp2(exp - 7)


def gate_ulp_ok(u: float) -> bool:
    return float(u) <= 2.0


# ---------------------------------------------------------------------------
# reconciliation numbers for U1 (union / built / missing)
# ---------------------------------------------------------------------------

def reconciliation() -> dict:
    all_items, missing = make_items()
    union = dict(conv=0, gn=0, up=0)
    built = dict(conv=0, gn=0, up=0)
    for it in all_items:
        union[it["kind"]] += 1
        if it["built"]:
            built[it["kind"]] += 1
    miss = dict(conv=0, gn=0, up=0)
    for it in missing:
        miss[it["kind"]] += 1
    return dict(union=union, built=built, missing=miss,
                total_missing=len(missing))


# ---------------------------------------------------------------------------
# compile / prep / device
# ---------------------------------------------------------------------------

def _setup_aie() -> None:
    """Env + device before any aie import (same order as vaeg_run.py:47-58)."""
    if str(HERE) not in sys.path:
        sys.path.append(str(HERE))
    if str(PYLIBS) not in sys.path:
        sys.path.append(str(PYLIBS))
    ml = str(PYLIBS / "mlir_aie" / "python")
    if ml not in sys.path:
        sys.path.append(ml)
    os.environ["PEANO_INSTALL_DIR"] = str(PEANO)
    os.environ.setdefault(
        "NPU_CACHE_HOME",
        str(ART / f"_npucache_{os.getpid()}_{int(time.time())}"))
    if (XRT_DIR / "xclbinutil.exe").is_file():
        os.environ["PATH"] = str(XRT_DIR) + os.pathsep + os.environ.get(
            "PATH", "")
    from aie.iron.device import NPU2
    from aie.utils.hostruntime import set_current_device
    set_current_device(NPU2())


def paths_of(item: dict, stem: str | None = None) -> dict:
    stem = stem or item["name"]
    return dict(stem=stem,
                xclbin=ART / f"{stem}.xclbin",
                insts=ART / f"{stem}.bin",
                prj=ART / f"{stem}.prj",
                io=ART / f"{stem}_io.npz",
                raw=ART / f"{stem}_raw.npy",
                results=ART / f"{stem}_results.json")


def compile_item(item: dict, stem: str | None = None) -> float:
    _setup_aie()
    import vaeg_design as D
    p = paths_of(item, stem)
    r = item["params"]
    if item["kind"] == "conv":
        import vae_npu as N
        spec = N.conv_gemm.specialize(M=r["M"], K=r["K"], N=r["N"],
                                       m=TILE["m"], k=TILE["k"], n=TILE["n"],
                                       use_cache=False)
    elif item["kind"] == "gn":
        design = (D.make_gn_single if r["mode"] == "single"
                  else D.make_gn_stream)
        spec = design(r["C"], r["H"], r["W"], GROUPS).specialize(
            eps=EPS, use_cache=False)
    else:
        spec = D.make_up_kernel(r["C"], r["H"], r["W"]).specialize(
            use_cache=False)
    t0 = time.perf_counter()
    spec.compile(xclbin_path=str(p["xclbin"]), inst_path=str(p["insts"]))
    return time.perf_counter() - t0


_VAE = {}


def _load_vae():
    if "vae" not in _VAE:
        import torch
        sys.path.insert(0, str(PORT))
        from diffusers import AutoencoderKL
        _VAE["torch"] = torch
        _VAE["vae"] = AutoencoderKL.from_pretrained(
            str(MODELS), subfolder="vae").eval()
    return _VAE["vae"], _VAE["torch"]


def _size_latent(size_name: str) -> tuple:
    h, w = [int(v) for v in size_name.split("x")]
    return h // 8, w // 8


def capture_module(vae, torch, module_name: str, lh: int, lw: int,
                   want_output: bool = False) -> dict:
    """One decoder forward at (lh, lw) with a hook on one module."""
    mod = dict(vae.decoder.named_modules())[module_name]
    cap = {}

    def hook(m, i, o):
        cap["x"] = i[0].detach().cpu().numpy()
        if want_output:
            oo = o[0] if isinstance(o, (tuple, list)) else o
            cap["y"] = oo.detach().cpu().numpy()

    h = mod.register_forward_hook(hook)
    lat = np.random.default_rng(0).standard_normal(
        (1, 16, lh, lw)).astype(np.float32)
    with torch.no_grad():
        vae.decode(torch.from_numpy(lat / SCALING + SHIFT), return_dict=False)
    h.remove()
    return dict(mod=mod, cap=cap)


def _psnr(ref, got) -> float:
    d = np.asarray(got, np.float64) - np.asarray(ref, np.float64)
    mse = float(np.mean(d ** 2))
    peak = float(np.abs(ref).max()) or 1.0
    return float(10 * np.log10(peak ** 2 / (mse + 1e-30)))


from t21_3b2_fastref import bfp_gemm_model_v, unpack_b_bytes  # noqa: F401

def _ref_model_of(A_b, W_pad, B_bytes=None, full_refs: bool = False):
    """The gate reference: fast vectorized path, scalar fallback."""
    if not FAST_REF:
        return N_bfp_gemm(A_b, W_pad)
    bqT = (unpack_b_bytes(B_bytes, W_pad.shape[0], W_pad.shape[1])
           if B_bytes is not None else None)
    return bfp_gemm_model_v(A_b, W_pad, bqT=bqT)


def N_bfp_gemm(A_b, W_pad):
    """vae_npu.bfp_gemm_model (the per-tile scalar reference)."""
    import vae_npu as VPN
    return VPN.bfp_gemm_model(np.asarray(A_b, np.float32), W_pad,
                              m=TILE["m"], k=TILE["k"], n=TILE["n"])


def prep_conv(item: dict, full_refs: bool = False) -> dict:
    """Host im2col + BFP16 W + the device-exact gate reference.

    Stored per geometry (disk protocol): the A stream, the packed B, and ONE
    reference -- the device model rounded to bf16 (uint16), which is exactly
    what the device must reproduce bit for bit.  The oracle PSNR and the
    f32-weight attribution are reduced to NUMBERS here and not kept as arrays:
    at M=65536 they would cost 200+ MB each and the gate does not judge them.
    """
    import vae_npu as N
    from ml_dtypes import bfloat16

    vae, torch = _load_vae()
    r = item["params"]
    lh, lw = _size_latent(item["size"])
    got = capture_module(vae, torch, item["module"], lh, lw, want_output=True)
    mod, cap = got["mod"], got["cap"]
    x, oracle = cap["x"], cap["y"]
    W = mod.weight.detach().cpu().numpy().astype(np.float32)
    b = (mod.bias.detach().cpu().numpy().astype(np.float32)
         if mod.bias is not None else None)
    Cout, Cin, kh, kw = W.shape
    pad = (kh - 1) // 2
    assert (Cin, Cout, kh) == (r["Cin"], r["Cout"], r["k"]), (
        item["name"], (Cin, Cout, kh))

    A_ic = N.im2col(x, kh, kw, pad)
    Wmat = W.reshape(Cout, -1).T
    K_real, N_real = Wmat.shape
    W_pad, used = N.pad_kn(Wmat, r["K"], r["N"], bias_row=b)
    bias_col = K_real if b is not None else None
    A_pad = N.pad_a(A_ic, r["K"], bias_col)
    assert used == (K_real + 8 if b is not None else K_real)

    A_b = A_pad.astype(bfloat16)              # PITFALL 1: A is RNE bf16
    B_bytes = _pack_b(W_pad, r["K"], r["N"])  # PITFALL 1: W is floor/bfp16
    del A_ic, A_pad                           # ~1.2 GB back before the model
    _ref_q, ref_model = _ref_model_of(A_b, W_pad, B_bytes)
    ora = np.asarray(oracle, np.float64).reshape(Cout, -1).T
    ref_m_b = ref_model.astype(bfloat16)
    stats = dict(
        prep_psnr_model_vs_oracle=_psnr(ora[:, :N_real], ref_model[:, :N_real]),
        prep_max_abs_model_vs_oracle=float(
            np.abs(ref_model[:, :N_real] - ora[:, :N_real]).max()),
        oracle_absmax=float(np.abs(ora).max()),
        full_refs=bool(full_refs),
    )

    p = paths_of(item)
    # The io schema is vaeg_run_xrt.py's (n_in / in{i}_bytes / out_len), the
    # same one prep_gn and prep_up use.  The historical conv io used
    # A_bytes/B_bytes/C_len, which only run_xrt.py (not this harness's runner)
    # reads -- writing that schema made the device run die on KeyError n_in.
    payload = dict(
        n_in=np.int64(2),
        in0_bytes=np.frombuffer(np.ascontiguousarray(A_b).tobytes(), np.uint8),
        in1_bytes=np.frombuffer(B_bytes, np.uint8),
        out_len=np.int64(r["M"] * r["N"] * 2),
        dims=np.array([r["M"], r["K"], r["N"], TILE["m"], TILE["k"],
                       TILE["n"]], dtype=np.int64),
        geom=np.array([Cout, Cin, kh, kw, x.shape[2], x.shape[3], K_real,
                       N_real, bias_col if bias_col is not None else -1],
                      dtype=np.int64),
        ref_model_bf16=np.ascontiguousarray(ref_m_b).view(np.uint16),
        prep=np.array(json.dumps(stats)),
    )
    if full_refs:
        _, ref_f32w = N.bfp_gemm_model(
            np.asarray(A_b, np.float32), W_pad, m=TILE["m"], k=TILE["k"],
            n=TILE["n"], use_tile_ref=False)
        payload["ref_f32w"] = ref_f32w.astype(np.float32)
        stats["prep_psnr_f32w_vs_oracle"] = _psnr(
            ora[:, :N_real], ref_f32w[:, :N_real])
        payload["prep"] = np.array(json.dumps(stats))
    np.savez(p["io"], **payload)
    return dict(A_bytes=A_b.nbytes, B_bytes=len(B_bytes), M=r["M"])


def _pack_b(W_pad, K: int, N: int) -> bytes:
    """The ONLY W encoding path: bfp16ebs8 with bfp.encode's floor default."""
    import vae_npu as VPN   # NOT `as N`: that would rebind the N argument
    return VPN.pack_b_blocks(W_pad, K, N, TILE["k"], TILE["n"])


def prep_gn(item: dict, full_refs: bool = False) -> dict:
    """vaeg_run.py:236-297 form, lean: bf16 stream in, bf16 GB arena, one store.

    Stored: the X stream, the GB arena, the model reference ROUNDED TO BF16
    (uint16) and the prep numbers.  The fp64 model_exact/model_dev/oracle
    arrays are NOT kept: at (256, 65536) three float64 arrays are 400 MB and
    the gate only needs bf16(model_exact) plus the magnitude bound, which is
    recomputed from the stored X stream and gamma/beta.
    """
    import vaeg_kernels as K
    from ml_dtypes import bfloat16

    vae, torch = _load_vae()
    r = item["params"]
    C, S = r["C"], r["S"]
    lh, lw = _size_latent(item["size"])
    got = capture_module(vae, torch, item["module"], lh, lw, want_output=True)
    mod, cap = got["mod"], got["cap"]
    x = np.asarray(cap["x"], np.float32).reshape(C, -1)
    oracle = np.asarray(cap["y"], np.float32).reshape(C, -1)
    assert x.shape == (C, S), (x.shape, C, S)

    gamma = mod.weight.detach().cpu().numpy().astype(np.float32).reshape(-1)
    beta = mod.bias.detach().cpu().numpy().astype(np.float32).reshape(-1)
    x_b = x.astype(bfloat16)
    g_b = gamma.astype(bfloat16)
    b_b = beta.astype(bfloat16)

    gb = build_gb(C, g_b, b_b, with_stats=True)      # PITFALL 2
    model_ex = K.group_norm_exact(x_b.astype(np.float64),
                                  g_b.astype(np.float64),
                                  b_b.astype(np.float64), groups=GROUPS,
                                  eps=EPS)
    model_dev = K.group_norm_device_formula(
        x_b.astype(np.float32), g_b.astype(np.float32),
        b_b.astype(np.float32), groups=GROUPS, eps=EPS)
    ref_b = model_ex.astype(bfloat16)
    stats = dict(
        prep_psnr_model_vs_oracle=_psnr(oracle, model_ex),
        prep_psnr_model_dev_vs_model=_psnr(model_ex, model_dev),
        prep_max_abs_model_vs_oracle=float(np.abs(model_ex - oracle).max()),
        oracle_absmax=float(np.abs(oracle).max()),
        full_refs=bool(full_refs),
    )

    p = paths_of(item)
    payload = dict(
        in0_bytes=np.frombuffer(np.ascontiguousarray(x_b).tobytes(), np.uint8),
        in1_bytes=np.frombuffer(gb, np.uint8),
        n_in=np.int64(2),
        out_len=np.int64(C * S * 2),
        ref_bf16=np.ascontiguousarray(ref_b).view(np.uint16),
        gamma_bf16=np.ascontiguousarray(g_b).view(np.uint16),
        beta_bf16=np.ascontiguousarray(b_b).view(np.uint16),
        prep=np.array(json.dumps(stats)),
        meta=np.array([C, r["H"], r["W"], GROUPS, C // GROUPS, S,
                       r["group_len"], int(r["mode"] == "stream")]),
    )
    if full_refs:
        payload["model_exact"] = model_ex.astype(np.float32)
        payload["model_dev"] = model_dev.astype(np.float32)
        payload["oracle"] = oracle
    np.savez(p["io"], **payload)
    return dict(X_bytes=C * S * 2, GB_bytes=len(gb), Y_bytes=C * S * 2)


def _bf16_bytes(a) -> bytes:
    """A float32 array already on the bf16 grid -> the bf16 byte stream."""
    u = np.ascontiguousarray(np.asarray(a, np.float32)).view(np.uint32)
    return (u >> 16).astype(np.uint16).tobytes()


def build_gb(C: int, g_b, b_b, with_stats: bool = True) -> np.ndarray:
    """The GB arena: gamma 2C B, beta 2C B, [stats 16 B].  bf16 ONLY."""
    gb_bytes, stats_off = G_GB(C, with_stats)
    rg = _bf16_bytes(g_b)
    rb = _bf16_bytes(b_b)
    assert len(rg) == 2 * C, (len(rg), 2 * C)
    assert len(rb) == 2 * C, (len(rb), 2 * C)
    assert len(rg) + len(rb) <= gb_bytes, (len(rg) + len(rb), gb_bytes)
    gb = np.zeros(gb_bytes, np.uint8)
    gb[0:len(rg)] = np.frombuffer(rg, np.uint8)
    gb[2 * C:2 * C + len(rb)] = np.frombuffer(rb, np.uint8)
    return gb


def prep_up(item: dict) -> dict:
    """vaeg_up.py:189-212 form: bf16 stream in, exact replicate out."""
    from ml_dtypes import bfloat16

    vae, torch = _load_vae()
    r = item["params"]
    C, H, W = r["C"], r["H"], r["W"]
    lh, lw = _size_latent(item["size"])
    got = capture_module(vae, torch, item["module"], lh, lw)
    x = np.asarray(got["cap"]["x"], np.float32).reshape(C, H, W)
    x_b = x.astype(bfloat16)
    want = np.repeat(np.repeat(np.asarray(x_b), 2, axis=1), 2, axis=2)

    p = paths_of(item)
    np.savez(
        p["io"],
        n_in=np.int64(1),
        in0_bytes=np.frombuffer(
            np.ascontiguousarray(x_b.reshape(-1)).tobytes(), np.uint8),
        out_len=np.int64(up_out_bytes(C, H, W)),
        want=np.ascontiguousarray(want.reshape(-1)).view(np.uint16),
        meta=np.array([C, H, W]),
    )
    return dict(X_bytes=C * H * W * 2, Y_bytes=up_out_bytes(C, H, W))


def run_device(item: dict, runs: int = 2, stem: str | None = None,
               out: pathlib.Path | None = None) -> dict:
    p = paths_of(item, stem)
    cmd = [str(PY310), str(XRT_RUNNER), "--npz", str(p["io"]),
           "--xclbin", str(p["xclbin"]), "--insts", str(p["insts"]),
           "--out", str(out or p["raw"]), "--runs", str(runs),
           "--wait-ms", str(DEVICE_WAIT_MS)]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    dt = time.perf_counter() - t0
    tail = "\n".join((r.stdout or "").strip().splitlines()[-12:])
    if r.returncode != 0:
        # the child's stderr is where a KeyError/import error actually shows
        # up; the stdout-only tail hid the first two failures of this harness
        err = "\n".join((getattr(r, "stderr", "") or "").strip().splitlines()[-12:])
        print(tail)
        print(err)
        if "ERT_CMD_STATE_TIMEOUT" in (r.stdout or ""):
            raise DispatchTimeout(
                f"{item['name']}: ERT watchdog (state 8) on every run; "
                f"the dispatch does not fit in the platform's fixed ~7 s "
                f"budget")
        raise RuntimeError(f"device run failed for {item['name']} "
                           f"(exit {r.returncode}): {err.splitlines()[-1] if err else 'no stderr'}")
    if runs < 2:
        # NOTE: one launch per process on this platform.  KNOWN-ISSUES.md /
        # HANDOFF-2026-10-19 state the "state-8 law": a second launch of the
        # SAME xclbin in one session does not produce a completion signal and
        # the ERT watchdog kills it at ~7 s.  Determinism is therefore checked
        # across processes (the 1-in-10 rerun), not inside one.
        same = None
    else:
        same = "run-to-run byte-identical: True" in (r.stdout or "")
    print(f"[device] {item['name']}: exit 0, run-to-run identical={same}, "
          f"{dt:.2f} s")
    return dict(seconds=dt, run_to_run_identical=same, log_tail=tail)


# ---------------------------------------------------------------------------
# the per-geometry gates (shared by --run and the --selftest replay)
# ---------------------------------------------------------------------------

def _report(tag, got, ref, acc=None):
    d = np.asarray(got, np.float64) - np.asarray(ref, np.float64)
    mx = float(np.abs(d).max())
    rng = float(np.max(ref) - np.min(ref)) or 1.0
    mse = float(np.mean(d ** 2))
    peak = float(np.abs(ref).max()) or 1.0
    psnr = 10 * np.log10(peak ** 2 / (mse + 1e-30))
    row = dict(max_abs_err=mx, range_frac=mx / rng, mse=mse,
               psnr_db=float(psnr))
    if acc is not None:
        acc[tag] = row
    return row


def verify_conv(io_path, raw_path) -> dict:
    """run.py:325-398 gate: device vs bf16(device-exact model), 1 ulp.

    Two io schemas are accepted: the 3b2 lean one (ref_model_bf16 + prep
    numbers) and the historical run.py one (ref_model/ref_quant/ref_f32w/
    oracle), so --selftest can replay the stored artifacts unchanged.
    """
    from ml_dtypes import bfloat16
    io = np.load(io_path)
    raw = np.load(raw_path)
    geom = io["geom"]
    Cout, Cin, kh, kw, H, Wd, K_real, N_real = [int(v) for v in geom[:8]]
    M, K, N = [int(v) for v in io["dims"][:3]]
    N_real = min(N_real, N)
    C = raw.view(np.uint16).view(bfloat16).reshape(M, N).astype(np.float64)
    C = C[:, :N_real]

    res = {}

    def report(tag, g, r):
        return _report(tag, g, r, res)

    extra = {}
    if "ref_model_bf16" in io:
        ref_b = (io["ref_model_bf16"].view(np.uint16).view(bfloat16)
                 .astype(np.float64).reshape(M, N)[:, :N_real])
        stats = json.loads(str(io["prep"]))
        bit_exact = float((C == ref_b).mean())
        max_ulp = float((np.abs(C - ref_b)
                         / np.maximum(_bf16_ulp(ref_b), 2.0 ** -126)).max())
        report("device vs ref_model (bf16(model))", C, ref_b)
        oracle_psnr = stats["prep_psnr_model_vs_oracle"]
        extra = dict(prep=stats,
                     oracle_psnr_source=("prep model-vs-oracle; equals "
                                         "device-vs-oracle iff bit_exact"))
    else:
        ora = np.asarray(io["oracle"], np.float64)
        ref_m = np.asarray(io["ref_model"], np.float64)[:, :N_real]
        ref_b = ref_m.astype(bfloat16).astype(np.float64)
        report("device vs ref_model (bf16(model))", C, ref_b)
        report("device vs oracle (CPU fp32 conv)", C, ora)
        report("ref_model vs oracle", ref_m, ora)
        if "ref_quant" in io:
            ref_q = np.asarray(io["ref_quant"], np.float64)[:, :N_real]
            report("ref_quant vs oracle", ref_q, ora)
        if "ref_f32w" in io:
            ref_w = np.asarray(io["ref_f32w"], np.float64)[:, :N_real]
            report("ref_f32W vs oracle (bf16 cost only)", ref_w, ora)
            report("ref_quant vs ref_f32W (BFP16 W cost)", ref_q, ref_w)
        bit_exact = float((C == ref_b).mean())
        max_ulp = float((np.abs(C - ref_b)
                         / np.maximum(_bf16_ulp(ref_b), 2.0 ** -126)).max())
        oracle_psnr = res["device vs oracle (CPU fp32 conv)"]["psnr_db"]

    return dict(spec="conv", M=M, K=K, N=N,
                device_vs_model_max_ulp=max_ulp,
                bit_exact_frac=bit_exact,
                oracle_psnr_db=oracle_psnr,
                pass_=bool(max_ulp <= 1.0 and bit_exact == 1.0),
                results=res, **extra)


def verify_gn(io_path, raw_path) -> dict:
    """vaeg_run.py:320-412 gate: 2 bf16 ulp against the MAGNITUDE BOUND.

    Accepts the 3b2 lean schema (ref_bf16 + prep numbers) and the historical
    one (model_exact/model_dev/oracle), so --selftest replays both.
    """
    from ml_dtypes import bfloat16
    io = np.load(io_path)
    raw = np.load(raw_path)
    meta = [int(v) for v in io["meta"]]
    C, H, W, groups, cg, S, L = meta[:7]
    Y = raw.view(np.uint16).view(bfloat16).reshape(C, S).astype(np.float64)
    res = {}

    def report(tag, g, r):
        return _report(tag, g, r, res)

    extra = {}
    if "ref_bf16" in io:
        ref_b = (io["ref_bf16"].view(np.uint16).view(bfloat16)
                 .reshape(C, S).astype(np.float64))
        stats = json.loads(str(io["prep"]))
        xb = (io["in0_bytes"].view(np.uint16).view(bfloat16)
              .reshape(C, S).astype(np.float64))
        report("device vs bf16(model_exact)", Y, ref_b)
        oracle_psnr = stats["prep_psnr_model_vs_oracle"]
        extra = dict(prep=stats,
                     oracle_psnr_source=("prep model-vs-oracle; equals "
                                         "device-vs-oracle iff bit_exact"))
    else:
        oracle = np.asarray(io["oracle"], np.float64).reshape(C, S)
        model_ex = np.asarray(io["model_exact"], np.float64)
        model_dev = np.asarray(io["model_dev"], np.float64)
        ref_b = model_ex.astype(bfloat16).astype(np.float64)
        report("device vs bf16(model_exact)", Y, ref_b)
        report("device vs bf16(model_dev)", Y,
               model_dev.astype(bfloat16).astype(np.float64))
        report("device vs oracle (CPU torch GroupNorm)", Y, oracle)
        report("model_exact vs oracle", model_ex, oracle)
        report("model_dev vs model_exact", model_dev, model_ex)
        xb = (io["x_bf16"].view(np.uint16).view(bfloat16)
              .reshape(C, S).astype(np.float64))
        oracle_psnr = res["device vs oracle (CPU torch GroupNorm)"]["psnr_db"]
        extra = dict(device_vs_model_dev_max_ulp=float(
            (np.abs(Y - model_dev.astype(bfloat16).astype(np.float64))
             / np.maximum(_bf16_ulp(model_dev.astype(bfloat16).astype(
                 np.float64)), 2.0 ** -126)).max()))

    gbv = io["gamma_bf16"].view(np.uint16).view(bfloat16).astype(np.float64)
    bbv = io["beta_bf16"].view(np.uint16).view(bfloat16).astype(np.float64)
    mag = np.empty_like(ref_b)
    for g in range(groups):
        seg = xb[g * cg:(g + 1) * cg].reshape(-1)
        mu = seg.mean()
        var = ((seg - mu) ** 2).mean()
        rstd = 1.0 / np.sqrt(var + EPS)
        sl = slice(g * cg, (g + 1) * cg)
        mag[sl] = (np.abs(xb[sl] - mu) * rstd * np.abs(gbv[sl])[:, None]
                   + np.abs(bbv[sl])[:, None])
    max_ulp, max_ulp_mag = ulp_pair(Y, ref_b, mag)
    exact = float((Y == ref_b).mean())
    return dict(spec="gn", C=C, H=H, W=W, groups=groups, eps=EPS,
                device_vs_model_exact_max_ulp=max_ulp,
                device_vs_model_exact_max_ulp_mag=max_ulp_mag,
                bit_exact_frac=exact,
                oracle_psnr_db=oracle_psnr,
                pass_=bool(gate_ulp_ok(max_ulp_mag)), results=res, **extra)


def verify_up(io_path, raw_path) -> dict:
    """vaeg_up.py:227-241 gate: exact replicate, and the C*H*W*8 length."""
    from ml_dtypes import bfloat16
    io = np.load(io_path)
    raw = np.load(raw_path)
    meta = [int(v) for v in io["meta"]]
    C, H, W = meta[:3]
    want = io["want"].view(np.uint16).view(bfloat16).reshape(C, 2 * H, 2 * W)
    out_len = int(io["out_len"])
    Y = raw.view(np.uint16).view(bfloat16).reshape(-1)
    same = bool(np.array_equal(Y.view(np.uint16),
                               want.reshape(-1).view(np.uint16)))
    return dict(spec="up", C=C, H=H, W=W, bit_exact=same,
                elements=int(want.size), out_len=int(out_len),
                out_len_expected=int(up_out_bytes(C, H, W)),
                pass_=bool(same and out_len == up_out_bytes(C, H, W)))


VERIFIERS = dict(conv=verify_conv, gn=verify_gn, up=verify_up)


# ---------------------------------------------------------------------------
# one item, end to end
# ---------------------------------------------------------------------------

def run_item(item: dict, args) -> dict:
    p = paths_of(item)
    r = item["params"]
    ART.mkdir(parents=True, exist_ok=True)
    require_disk(args.min_free_mb, item, "compile")
    t0 = time.perf_counter()
    c_s = compile_item(item)
    print(f"[compile] {item['name']}: {c_s:.1f} s -> {p['xclbin'].name}")
    require_disk(args.min_free_mb, item, "prep")
    if item["kind"] == "conv":
        prep_conv(item, getattr(args, "full_refs", False))
    elif item["kind"] == "gn":
        prep_gn(item, getattr(args, "full_refs", False))
    else:
        prep_up(item)
    require_disk(args.min_free_mb, item, "device")
    dev = run_device(item, args.runs)
    res = VERIFIERS[item["kind"]](p["io"], p["raw"])
    res.update(id=item["id"], kind=item["kind"], size=item["size"],
               module=item["module"], name=item["name"],
               geom=dict((k, r[k]) for k in r
                         if isinstance(r[k], (int, float, str, bool))),
               compile_seconds=c_s, device_seconds=dev["seconds"],
               run_to_run_identical=dev["run_to_run_identical"],
               total_seconds=time.perf_counter() - t0,
               free_bytes_after=int(_free_bytes()), ts=_ts())
    p["results"].write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[gate] {item['name']}: pass_={res['pass_']}  -> "
          f"{p['results'].name}")
    if not args.keep_prj:
        _rm(p["prj"])
    if not args.keep_io:
        _rm(p["io"])
        _rm(p["raw"])
    return res


# ---------------------------------------------------------------------------
# CLI modes
# ---------------------------------------------------------------------------

def _fmt_row(it: dict) -> str:
    p = it["pred"]
    kind = it["kind"]
    if kind == "conv":
        geo = (f"M={it['params']['M']} K={it['params']['K']} "
               f"N={it['params']['N']}")
    elif kind == "gn":
        geo = (f"C={it['params']['C']} S={it['params']['S']} "
               f"[{it['params']['mode']}]")
    else:
        geo = (f"C={it['params']['C']} S={it['params']['S']} "
               f"W={'/'.join(it['params']['variants'])}")
    flag = {"existing": "ok", "variant": "var"}.get(it["buildable"], "GAP")
    return (f"{it['id']:3d} {kind:4s} {it['size']:8s} {geo:34s} "
            f"{p['compile_s']:7.1f} {p['cum_s']:8.1f} {p['gate_s']:7.1f} "
            f"{p['peak_bytes'] / MB:7.1f} {flag}")


def dry_run(args) -> int:
    all_items, missing, calib = plan_rows()
    rec = reconciliation()
    checks = assert_contracts(deep=False)
    gaps = [it for it in missing if it["buildable"] == "design_gap"]
    buildable = [it for it in missing if it["buildable"] == "existing"]
    variants = [it for it in missing if it["buildable"] == "variant"]
    clo = closure_rows(missing)
    free = _free_bytes()

    print("=" * 100)
    print("T21-3B2 PLAN (dry run: no compile, no device, no torch/aie)")
    print("=" * 100)
    print(f"[u1] union geometries      conv {rec['union']['conv']:3d}  "
          f"gn {rec['union']['gn']:3d}  up {rec['union']['up']:3d}")
    print(f"[u1] already built in out/ conv {rec['built']['conv']:3d}  "
          f"gn {rec['built']['gn']:3d}  up {rec['built']['up']:3d}")
    print(f"[u1] TO BUILD              conv {rec['missing']['conv']:3d}  "
          f"gn {rec['missing']['gn']:3d}  up {rec['missing']['up']:3d}"
          f"  = {rec['total_missing']}")
    if rec["total_missing"] != 92:
        print(f"[STOP-LINE 1] geometry count {rec['total_missing']} != 92 "
              f"(audit).  Deviation > 10%: stopping.")
        return 4
    print(f"[u1] design-variant needed {len(gaps):3d} ; buildable with the "
          f"existing designs {len(buildable)} ; gated by a 3b2vf VARIANT "
          f"{len(variants)}")
    print(f"[u1] closure (92/92 gate): {clo['n_pass']}/{clo['n']} rows have a "
          f"PASS verdict on disk "
          f"({'GREEN' if clo['green'] else 'RED: ' + str(clo['bad'])})")
    print()
    print("[contracts] the four measured pitfalls")
    ok_all = True
    for name, ok, detail in checks:
        ok_all &= ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name:28s} {detail}")
    print()
    print(f"[disk] free now {free / MB:.1f} MiB; threshold "
          f"{args.min_free_mb:.0f} MiB")
    worst = max(missing, key=lambda it: it["pred"]["peak_bytes"])
    print(f"[disk] predicted peak: one geometry at a time; worst = "
          f"{worst['name']} {worst['pred']['peak_bytes'] / MB:.1f} MiB "
          f"(io {worst['pred']['io_bytes'] / MB:.1f} MiB)")
    src = ("out/t21_3b2/calib.json" if (ART / "calib.json").is_file()
           else "defaults")
    print(f"[disk] gate constants: ms/tile {calib['ms_per_tile']:.4f}, "
          f"ms/pixel {calib['ms_per_pixel']:.4f} ({src})")
    print()
    print(f"{'idx':>3} {'kind':4s} {'size':8s} {'geometry':34s} "
          f"{'cmp_s':>7} {'cum_s':>8} {'gate_s':>7} {'diskMB':>7} flag")
    print("-" * 100)
    for it in missing:
        print(_fmt_row(it))
    print("-" * 100)
    tot_c = sum(it["pred"]["compile_s"] for it in missing)
    tot_cc = sum(it["pred"]["compile_cold_s"] for it in missing)
    tot_g = sum(it["pred"]["gate_s"] for it in missing)
    tot_c_ok = sum(it["pred"]["compile_s"] for it in buildable)
    tot_g_ok = sum(it["pred"]["gate_s"] for it in buildable)
    print(f"[totals] compile {tot_c / 60:.1f} min hot / {tot_cc / 60:.1f} "
          f"min cold (+50%) ; gate {tot_g / 60:.1f} min ; "
          f"sum {(tot_c + tot_g) / 3600:.2f} h")
    print(f"[totals] buildable now: compile {tot_c_ok / 60:.1f} min + gate "
          f"{tot_g_ok / 60:.1f} min = {(tot_c_ok + tot_g_ok) / 3600:.2f} h")
    if gaps:
        print(f"[gaps] {len(gaps)} rows are BLOCKED on a design variant "
              f"(not a recompile):")
        for it in gaps:
            print(f"  {it['id']:3d} {it['name']:28s} {it['gap_reason']}")
    if args.json:
        pathlib.Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.json).write_text(json.dumps(dict(
            ts=_ts(), reconciliation=rec,
            contracts=[dict(name=n, ok=o, detail=d) for n, o, d in checks],
            calib=calib, free_bytes=int(free),
            min_free_mb=args.min_free_mb,
            items=[dict(id=it["id"], kind=it["kind"], size=it["size"],
                        module=it["module"], name=it["name"],
                        buildable=it["buildable"],
                        gap_reason=it["gap_reason"],
                        params=it["params"], pred=it["pred"])
                   for it in missing],
            totals=dict(compile_s=tot_c, compile_cold_s=tot_cc, gate_s=tot_g),
        ), indent=1, default=str), encoding="utf-8")
        print(f"[json] -> {args.json}")
    if args.strict and (gaps or not ok_all or not clo["green"]):
        return 3
    return 0 if (ok_all and clo["green"]) else 3


def selftest(args) -> int:
    print("[selftest] contracts (deep: aie imported)")
    checks = assert_contracts(deep=True)
    ok_all = True
    for name, ok, detail in checks:
        ok_all &= ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name:28s} {detail}")
    print()
    print("[selftest] replay of the stored artifacts through the harness gates")
    replayed, mismatched, missing = 0, 0, 0
    rows = []
    for io in sorted(OUT.glob("*_io.npz")):
        stem = io.name[:-len("_io.npz")]
        if stem.startswith("_"):
            continue
        if stem.startswith("cg_"):
            raw, kind = OUT / f"{stem}_C_raw.npy", "conv"
        elif stem.startswith("vaeg_gn_"):
            raw, kind = OUT / f"{stem}_Y_raw.npy", "gn"
        elif stem.startswith("vaeg_up_"):
            # rows/elems are OTHER walks (one axis only), not the decoder's
            # nearest-2x: they are not part of this gate.
            if not (stem.startswith("vaeg_up_kernel_")
                    or stem.startswith("vaeg_up_full_")):
                print(f"  SKIP {stem}: different tap mode, not nearest-2x")
                missing += 1
                continue
            raw, kind = OUT / f"{stem}_Y_raw.npy", "up"
        else:
            missing += 1
            continue
        if not raw.is_file():
            missing += 1
            continue
        try:
            res = VERIFIERS[kind](io, raw)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR {stem}: {type(exc).__name__}: {exc}")
            mismatched += 1
            continue
        stored_p = OUT / f"{stem}_results.json"
        stored = (json.loads(stored_p.read_text(encoding="utf-8"))
                  if stored_p.is_file() else {})
        sp = stored.get("pass_", stored.get("bit_exact"))
        agree = (sp is None) or (bool(sp) == bool(res["pass_"]))
        replayed += 1
        mismatched += 0 if agree else 1
        if kind == "conv":
            key, val = "max_ulp", res.get("device_vs_model_max_ulp")
        elif kind == "gn":
            key, val = "max_ulp_mag", res.get(
                "device_vs_model_exact_max_ulp_mag")
        else:
            key, val = "bit_exact", res.get("bit_exact")
        rows.append((stem, kind, res["pass_"], sp, val, agree))
        print(f"  {stem[:52]:52s} {kind:4s} pass={str(res['pass_']):5s} "
              f"stored={str(sp):5s} {key}={val}")
    print()
    print(f"[selftest] replayed {replayed}, disagreements {mismatched}, "
          f"unpaired {missing}")
    print(f"[selftest] {'PASS' if ok_all and mismatched == 0 else 'FAIL'}")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(dict(
            ts=_ts(),
            contracts=[dict(name=n, ok=o, detail=d) for n, o, d in checks],
            replayed=replayed, disagreements=mismatched,
            rows=[dict(stem=s, kind=k, pass_=bool(p), stored=st, value=v,
                       agree=a) for s, k, p, st, v, a in rows],
        ), indent=1, default=str), encoding="utf-8")
        print(f"[json] -> {args.json}")
    return 0 if (ok_all and mismatched == 0) else 3


def calibrate(args) -> int:
    _setup_aie()
    from aie.iron.kernels.linalg import mm_bfp_tile_ref
    import vae_npu as N

    rng = np.random.default_rng(0)
    A = rng.standard_normal((TILE["m"], TILE["k"])).astype(np.float32)
    B = rng.standard_normal((TILE["k"], TILE["n"])).astype(np.float32)
    mm_bfp_tile_ref(A, B, dim_m=TILE["m"], dim_k=TILE["k"],
                    dim_n=TILE["n"], mixed=True)
    n = 300
    t0 = time.perf_counter()
    for _ in range(n):
        mm_bfp_tile_ref(A, B, dim_m=TILE["m"], dim_k=TILE["k"],
                        dim_n=TILE["n"], mixed=True)
    ms_tile = 1000.0 * (time.perf_counter() - t0) / n

    x = rng.standard_normal((1, 512, 8, 8)).astype(np.float32)
    N.im2col(x, 3, 3, 1)
    reps = 3
    t0 = time.perf_counter()
    for _ in range(reps):
        aic = N.im2col(x, 3, 3, 1)
    ms_px = 1000.0 * (time.perf_counter() - t0) / (reps * aic.shape[0])

    ART.mkdir(parents=True, exist_ok=True)
    calib = dict(ms_per_tile=ms_tile, ms_per_pixel=ms_px, measured_at=_ts(),
                 note="mm_bfp_tile_ref (16,8,16 mixed) and vae_npu.im2col "
                      "per output pixel, C=512 k=3")
    (ART / "calib.json").write_text(json.dumps(calib, indent=1),
                                    encoding="utf-8")
    print(f"[calib] ms/tile {ms_tile:.4f}  ms/pixel {ms_px:.4f} -> "
          f"{ART / 'calib.json'}")
    return 0


def main() -> int:
    _force_utf8()
    ap = argparse.ArgumentParser(
        description="T21-3B2: build+gate the 92 VAE second-tier xclbins "
                    "(dry-run by default).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the 92-row plan; no compile, no device")
    ap.add_argument("--list", action="store_true", help="compact table")
    ap.add_argument("--selftest", action="store_true",
                    help="4 pitfall contracts + replay the stored artifacts")
    ap.add_argument("--calibrate", action="store_true",
                    help="measure the gate-cost constants")
    ap.add_argument("--run", action="store_true",
                    help="build+gate for real (py312/aie + py310/pyxrt)")
    ap.add_argument("--only", type=int, default=None, help="1-based item id")
    ap.add_argument("--from", dest="from_id", type=int, default=None,
                    help="first item id to run")
    ap.add_argument("--runs", type=int, default=2, help="device repeats")
    ap.add_argument("--min-free-mb", type=float, default=DEFAULT_MIN_FREE_MB)
    ap.add_argument("--full-refs", action="store_true",
                    help="also store the f32-weight attribution arrays")
    ap.add_argument("--keep-prj", action="store_true")
    ap.add_argument("--keep-io", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="--dry-run exits 3 if a row needs a design variant")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    _install_mkdir_mode_shim()

    if args.calibrate:
        return calibrate(args)
    if args.selftest:
        return selftest(args)
    if args.list:
        _all, missing, _calib = plan_rows()
        for it in missing:
            print(_fmt_row(it))
        return 0
    if args.dry_run or not args.run:
        if not args.dry_run:
            ap.print_help()
            return 2
        return dry_run(args)

    all_items, missing, calib = plan_rows()
    todo = missing
    if args.only is not None:
        todo = [it for it in missing if it["id"] == args.only]
        if not todo:
            print(f"[FAIL] no item with id {args.only} (1..{len(missing)})")
            return 2
    elif args.from_id is not None:
        todo = [it for it in missing if it["id"] >= args.from_id]

    results = []
    rc = 0
    for it in todo:
        if it["buildable"] != "existing":
            print(f"[skip] {it['id']:3d} {it['name']}: {it['gap_reason']}")
            results.append(dict(id=it["id"], name=it["name"],
                                status="design_gap",
                                reason=it["gap_reason"]))
            continue
        try:
            res = run_item(it, args)
            res["status"] = "pass" if res["pass_"] else "fail"
            results.append(res)
            rc = rc or (0 if res["pass_"] else 3)
        except DiskAbort:
            results.append(dict(id=it["id"], name=it["name"],
                                status="disk_abort"))
            rc = rc or 5
            break
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results.append(dict(id=it["id"], name=it["name"],
                                status="error",
                                error=f"{type(exc).__name__}: {exc}"))
            rc = rc or 1
            break
    payload = dict(ts=_ts(), argv=sys.argv[1:], results=results)
    args.json = args.json or str(DEFAULT_RESULTS)
    pathlib.Path(args.json).write_text(
        json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print(f"[json] -> {args.json}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
