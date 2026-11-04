# SPDX-License-Identifier: MIT
"""npu_synth.py -- turn an npu_plan plan into build_sweep VARIANT entries and build it.

    py312 npu_synth.py --m 608              # plan -> variants -> build all 3
    py312 npu_synth.py --hw 384 384         # same, from a requested (H, W)
    py312 npu_synth.py --grid               # every size in the validation grid
    py312 npu_synth.py --list --m 608       # show the variants, build nothing

WHAT IT DOES, IN ORDER
  1. webui/npu_plan.plan(M)            the pure plan (or the refusal reason)
  2. variants_for(plan)                one build_sweep VARIANT per GEMM image
                                       (gu / dn / proj), same geometry the plan
                                       names -- nothing is re-derived here
  3. out/gen_<plan_key>.json           the generated VARIANT table that
                                       build_sweep merges with --gen (its own
                                       VARIANTS dict is untouched)
  4. build_sweep.py --one <tag>        one aiecc per image, PY312, serial
  5. out/<tag>.prj                     deleted after the build, kernels_main.json
                                       kept (run_ffn reads it for the BO count)
  6. webui/shape_manifests/plan_<key>.json
                                       the committed record: plan + images +
                                       sha256, keyed on the PLAN so two shapes
                                       with the same plan share one entry

THE PACKERS the plan implies are wired in npu_dev.py (the device leg):
    pack_kk.pack_a_kk / pack_b_kk      kk-parametric A/B (179 bit-verified checks)
    verify_t33.pack_a_t33              the multi-tile A form (ffn_t33)
B is a PERMUTATION of the model BFP16 blocks, never a re-encode (0.297 % of the
model's blocks are non-canonical, so a re-encode would change the numbers).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]                 # the repo
OUT = HERE / "out"
MANIFESTS = ROOT / "webui" / "shape_manifests"
PY312 = os.environ.get(
    "NPU_PY312", r"D:\Project\deepseek_harness\test7\.winml_venv\Scripts\python.exe")

sys.path.insert(0, str(ROOT))
from webui import npu_plan as P                                    # noqa: E402

DESIGN = {"xd": "ffn_xb", "t33": "ffn_t33"}
#: the FUSED gu (silu(gate)*up inside the image).  It is an xd-form image with a
#: different epilogue and a pair B object, so it is its own design, not a kw.
DESIGN_FUSED_GU = "ffn_xf"


def tag_for(plan, gemm):
    """A stable, readable tag for one image of a plan.

    The tag carries EVERY compile-time knob that changes the image, so two plans
    can never collide on a tag (and the tag is what the xclbin is named after).
    A FUSED gu gets an "f" on its n field: the fusion changes the image's B
    object (a [gate|up] pair), its C leaf width and its epilogue, so a fused
    gu and a pure gu of the same n must never share an xclbin name.
    """
    g = plan["gemms"][gemm]
    fused = bool(g.get("fused")) and bool(plan.get("fuse_gu"))
    return "ad_%s_m%dk%d_M%d_n%d%s_a%db%d%s_%s" % (
        plan["strategy"], plan["m"], plan["kk"], plan["Mblk"], g["n"],
        "f" if fused else "",
        plan["adepth"], plan["bdepth"],
        "cw" if plan.get("cwait") else "", gemm)


def variants_for(plan):
    """{tag: {"design", "kw", "flags"}} -- exactly what build_sweep needs."""
    out = {}
    for gemm, K, N in P.GEMMS:
        g = plan["gemms"][gemm]
        kw = dict(M=plan["Mblk"], K=K, N=N, m=plan["m"], kk=plan["kk"],
                  n=g["n"], ntiles=g["ntiles"], cols=plan["cols"],
                  cpc=plan["cpc"], mixed=bool(plan["mixed"]),
                  adepth=plan["adepth"], bdepth=plan["bdepth"])
        if plan["strategy"] == "xd":
            kw["bfeed"] = "shared"
        else:
            kw["rot"] = 7
            kw["cwait"] = bool(plan.get("cwait", True))
        kw["tag"] = tag_for(plan, gemm)
        design = DESIGN[plan["strategy"]]
        if bool(g.get("fused")) and bool(plan.get("fuse_gu")):
            # the fused gu: the same geometry the plan picked (its own, smaller
            # n -- the epilogue needs three C tiles), the pair B object and the
            # silu+mul epilogue.  ffn_xb's bfeed/rot knobs do not apply.
            design = DESIGN_FUSED_GU
            kw.pop("bfeed", None)
            kw.pop("rot", None)
            kw.pop("cwait", None)
            kw["EPI"] = 1
        out[kw["tag"]] = {"design": design, "kw": kw}
    return out


def _prj_clean(tag):
    """Delete out/<tag>.prj but keep kernels_main.json (run_ffn counts BOs)."""
    d = OUT / (tag + ".prj")
    if not d.is_dir():
        return None
    keep = d / "kernels_main.json"
    saved = keep.read_bytes() if keep.is_file() else None
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    if saved is not None:
        keep.write_bytes(saved)
    return str(d)


def _design_l1(tag):
    """The L1 the design ITSELF printed for this image (a cross-check on the plan).

    ffn_xb prints  "[ffn_xb] <tag>: L1/core = C .. + B .. + A .. + stack .. = NNNN"
    and ffn_t33 prints "... L1=NNNN B".  Reading it back means the plan's L1 and
    the compiler's L1 cannot silently disagree.
    """
    log = OUT / (tag + ".log")
    if not log.is_file():
        return None
    txt = log.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"L1/core = .*? = (\d+) of", txt)
    if m:
        return int(m.group(1))
    m = re.search(r"L1=([\d,]+)B", txt)
    if m:
        return int(m.group(1).replace(",", ""))
    return None


def write_gen(plan, variants=None):
    """Write the generated VARIANT table build_sweep --gen reads."""
    variants = variants or variants_for(plan)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / ("gen_%s.json" % plan["plan_key"])
    path.write_text(json.dumps({"plan_key": plan["plan_key"],
                                "variants": variants}, indent=1),
                    encoding="utf-8")
    return path


def build_one_tag(tag, gen_path, timeout=900):
    """Run build_sweep --one <tag> --gen <gen_path> in its own process."""
    t0 = time.perf_counter()
    cmd = [PY312, str(HERE / "build_sweep.py"), "--one", tag,
           "--gen", str(gen_path)]
    r = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True,
                       timeout=timeout)
    wall = time.perf_counter() - t0
    rec = {"tag": tag, "wall_s": round(wall, 2)}
    try:
        tail = (r.stdout or "")
        rec.update(json.loads(tail[tail.index("{"):]))
    except Exception:                                            # noqa: BLE001
        rec["status"] = "DRIVER_FAIL"
        rec["error"] = ((r.stdout or "")[-1500:] + (r.stderr or "")[-3000:])
    rec["wall_s"] = round(wall, 2)
    rec["l1_design"] = _design_l1(tag)
    rec["prj_cleaned"] = _prj_clean(tag)
    xclbin = OUT / (tag + ".xclbin")
    rec["xclbin"] = str(xclbin) if xclbin.is_file() else None
    rec["sha256"] = (hashlib.sha256(xclbin.read_bytes()).hexdigest()
                     if xclbin.is_file() else None)
    return rec


def build_plan(plan, timeout=900, force=False):
    """Build (or reuse) every image the plan needs.  Returns the record."""
    variants = variants_for(plan)
    gen_path = write_gen(plan, variants)
    recs = {}
    for tag in variants:
        xclbin = OUT / (tag + ".xclbin")
        if xclbin.is_file() and not force:
            recs[tag] = {"tag": tag, "status": "CACHED",
                         "xclbin": str(xclbin),
                         "sha256": hashlib.sha256(
                             xclbin.read_bytes()).hexdigest(),
                         "l1_design": _design_l1(tag)}
            continue
        recs[tag] = build_one_tag(tag, gen_path, timeout=timeout)
    return {"plan_key": plan["plan_key"], "M": plan["M"],
            "variants": list(variants), "gen": str(gen_path),
            "images": recs,
            "ok": all(r.get("status") in ("BUILT", "CACHED")
                      for r in recs.values())}


def write_manifest(plan, rec, note=""):
    """The committed per-PLAN record (webui/shape_manifests/plan_<key>.json).

    Keyed on the plan, not on (H, W): two shapes whose plan is identical share
    this one entry, and shapes.build_shape's SHAPE_MANIFESTS mechanism reads it
    back without knowing which shape asked first.
    """
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    # The key is the IMAGE SET, so a second size that lands on the same set
    # merges into the SAME entry (used_by / ndisp_by_M) instead of forking it.
    used_by, ndisp_by_m = [], {}
    prev = MANIFESTS / ("plan_%s.json" % plan["plan_key"])
    if prev.is_file():
        try:
            old = json.loads(prev.read_text(encoding="utf-8"))
            used_by = list(old.get("used_by") or [])
            ndisp_by_m = dict(old.get("ndisp_by_M") or {})
        except Exception:                                        # noqa: BLE001
            pass
    if plan["M"] not in used_by:
        used_by.append(plan["M"])
    ndisp_by_m[str(plan["M"])] = plan["ndisp"]
    man = {
        "plan_key": plan["plan_key"],
        "Mblk": plan["Mblk"],
        "used_by": sorted(used_by),
        "ndisp_by_M": {k: ndisp_by_m[k] for k in sorted(ndisp_by_m, key=int)},
        "strategy": plan["strategy"], "m": plan["m"], "kk": plan["kk"],
        "T": plan["T"], "cols": plan["cols"], "cpc": plan["cpc"],
        "mixed": plan["mixed"], "adepth": plan["adepth"],
        "bdepth": plan["bdepth"],
        # the fusion is a compile-time knob of the gu image: record it, and
        # record the gu's own n (it is NOT the pure gu's n)
        "fuse_gu": int(plan.get("fuse_gu") or 0),
        "gu_fused": int(plan["gemms"]["gu"].get("fused") or 0),
        "gu_l1": plan["gemms"]["gu"]["l1"],
        "pad_rows": plan["pad_rows"], "per_col": plan["per_col"],
        "l1_max": plan["l1_max"],
        "gemms": {k: {kk2: v2 for kk2, v2 in v.items()
                      if kk2 in ("K", "N", "n", "ntiles", "nkk", "l1",
                                 "a_bytes", "b_bytes", "c_bytes")}
                  for k, v in plan["gemms"].items()},
        "images": {t: {"status": r.get("status"),
                       "xclbin": r.get("xclbin"),
                       "sha256": r.get("sha256"),
                       "l1_design": r.get("l1_design")}
                   for t, r in rec["images"].items()},
        "built_ok": bool(rec.get("ok")),
        "note": note or ("adaptive 32-core placement image set; the plan is the "
                         "cache key, so an identical plan is built once"),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    p = MANIFESTS / ("plan_%s.json" % plan["plan_key"])
    p.write_text(json.dumps(man, indent=1), encoding="utf-8")
    return p


def manifest_for(plan):
    """The manifest of this plan, or None.  The cache hit test."""
    p = MANIFESTS / ("plan_%s.json" % plan["plan_key"])
    if not p.is_file():
        return None
    try:
        man = json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                            # noqa: BLE001
        return None
    if man.get("plan_key") != plan["plan_key"] or not man.get("built_ok"):
        return None
    if int(man.get("Mblk") or -1) != int(plan["Mblk"]):
        return None
    for img in (man.get("images") or {}).values():
        if img.get("status") not in ("BUILT", "CACHED") or not img.get("xclbin"):
            return None
        if not pathlib.Path(img["xclbin"]).is_file():
            return None
    return man


def built_plan_keys():
    """Every plan key with a usable manifest -- the planner's prefer set."""
    out = set()
    if not MANIFESTS.is_dir():
        return out
    for p in MANIFESTS.glob("plan_*.json"):
        try:
            man = json.loads(p.read_text(encoding="utf-8"))
        except Exception:                                        # noqa: BLE001
            continue
        if man.get("built_ok"):
            out.add(man.get("plan_key"))
    return out


def synth(M, force=False, prefer_built=True, note="", fuse_gu=False,
          fuse_fill=1.0):
    """plan -> build -> manifest.  The one entry point the product calls."""
    res = P.plan(M, prefer=(built_plan_keys() if prefer_built else None),
                 fuse_gu=fuse_gu, fuse_fill=fuse_fill)
    if not res["ok"]:
        return {"ok": False, "M": int(M), "reason": res["reason"],
                "rejected": res["rejected"], "plan": None}
    plan = res["plan"]
    rec = build_plan(plan, force=force)
    man = write_manifest(plan, rec, note=note)
    return {"ok": rec["ok"], "M": int(M), "plan": plan, "build": rec,
            "manifest": str(man), "rejected": res["rejected"],
            "reason": "" if rec["ok"] else "one or more images failed to build"}


def hw_to_M(h, w):
    """The product's own arithmetic (webui/shapes.py), duplicated here so the
    synthesizer can be driven from the CLI without importing the WebUI."""
    pad = lambda v: (int(v) + 15) // 16 * 16                   # noqa: E731
    hp, wp = pad(h), pad(w)
    tok = (hp // 16) * (wp // 16)
    tok = -(-tok // 32) * 32
    return 32 + tok


SQUARES = (128, 160, 192, 224, 256, 288, 320, 384, 448, 512, 640, 768, 896,
           1024)
RECTS = ((192, 384), (256, 512), (384, 512), (128, 512), (512, 768),
         (320, 448))


def grid_pairs():
    return [(s, s) for s in SQUARES] + list(RECTS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=None)
    ap.add_argument("--hw", type=int, nargs=2, default=None)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--fuse", action="store_true",
                    help="plan+build the FUSED gu (silu(gate)*up in the image)")
    ap.add_argument("--fuse-fill", type=float, default=1.0,
                    help="fraction of the 64 KiB L1 budget the fused n may use")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    jobs = []
    if a.grid:
        jobs = [(h, w) for h, w in grid_pairs()]
    elif a.hw:
        jobs = [(a.hw[0], a.hw[1])]
    elif a.m is not None:
        jobs = [(None, a.m)]
    else:
        ap.print_help()
        return 1
    rows = []
    for h, w in jobs:
        M = a.m if h is None else hw_to_M(h, w)
        if a.list:
            res = P.plan(M, fuse_gu=a.fuse, fuse_fill=a.fuse_fill)
            if not res["ok"]:
                print("%s -> M=%d REFUSED %s" % ((h, w), M, res["reason"]))
                continue
            print("%s -> M=%d  %s" % ((h, w), M, res["plan"]["plan_key"]))
            for tag, v in variants_for(res["plan"]).items():
                print("   ", tag, v["design"], json.dumps(v["kw"]))
            continue
        rec = synth(M, force=a.force, fuse_gu=a.fuse,
                    fuse_fill=a.fuse_fill)
        row = {"h": h, "w": w, "M": M, "ok": rec["ok"]}
        if rec.get("plan"):
            p = rec["plan"]
            row.update({"plan_key": p["plan_key"], "strategy": p["strategy"],
                        "fuse_gu": int(p.get("fuse_gu") or 0),
                        "m": p["m"], "kk": p["kk"], "Mblk": p["Mblk"],
                        "ndisp": p["ndisp"], "T": p["T"], "nm": p["nm"],
                        "per_col": p["per_col"], "l1_max": p["l1_max"],
                        "pad_rows": p["pad_rows"],
                        "n": {k: p["gemms"][k]["n"] for k in p["gemms"]},
                        "layer_ms": p["layer_ms"],
                        "built": {t: r.get("status")
                                  for t, r in rec["build"]["images"].items()},
                        "manifest": rec["manifest"]})
        else:
            row["reason"] = rec["reason"]
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(rows, indent=1),
                                        encoding="utf-8")
    return 0 if all(r["ok"] for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
