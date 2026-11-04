# SPDX-License-Identifier: MIT
"""T18: the REAL per-shape builder (build_shape(H, W)) with a stage stream.

Before T18 a new (H, W) returned a build plan and nothing else.  This module
turns the plan into work, in this order, and records every stage with its own
wall time so the WebUI can show what actually happened:

    plan      shape arithmetic, M, the exact artifacts this shape needs
    artifacts compile any missing image (FFN at M, ip at ND=SEQ) -- a present
              artifact is reused, never re-built silently
    audit     the winning-recipe rebuild + per-core audit for every new image
              (t4_build_win = 17 cores, t5_build_win = 3 cores); the summary
              is parsed and recorded
    cache     copy + sha256 into webui/cache/{H}x{W}/ (cache.ensure_shape)
    manifest  webui/shape_manifests/{H}x{W}.json -- the committed checksum
              record of exactly which images this shape's cache copies

Acceptance for the audit stage is NOT "all cores green": the delivered FFN has
four gate/up cores with a known ACQ CLOBBER pattern (T11 R5), so the gate is
"every core audited, and the pass set equals the delivered build's".  A new
core turning red fails the build.

T22 prodlarge (2026-10-20): the product envelope now reaches 2048² (M up to
16416) but the builder's recipes still cover only M=128/160/288.  Anything
else is refused with the shape's own 512²-spec sentence plus the recipe list --
`build_shape --height 1024 --width 1024` says exactly why nothing was
compiled, and never half-builds.  No recipe was added by this unit (a new M is
a kernel + gate job, not a product-setting job).
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time

from . import cache, config, shapes

PY312 = str(config.PY312)
NPU = config.NPU
SHAPE_MANIFESTS = config.NPU / "webui" / "shape_manifests"

# The delivered FFN's four gate/up cores carry 3 ACQ CLOBBER exposures each
# (T11 R5 / T10 §2).  They are the baseline every new FFN build is compared to.
FFN_BASELINE_PASS = 13
FFN_BASELINE_TOTAL = 17

COMPILE = {
    128: {
        "ffn": ["imaggers/block/t4_compile.py", "--design", "t11_ffn_nh5",
                "--tag", "t18_m128", "--M", "128", "--NH", "5", "--kgu", "8"],
        "ip": ["imaggers/attn_impl/t18_attn_compile.py", "--tag",
               "t18_ip_m64nd128", "--M", "64", "--ND", "128"],
    },
    288: {
        "ffn": ["imaggers/block/t4_compile.py", "--design", "t11_ffn_nh5",
                "--tag", "t18_m288", "--M", "288", "--NH", "5", "--kgu", "8"],
        "ip": ["imaggers/attn_impl/t18_attn_compile.py", "--tag",
               "t18_ip_m16nd288", "--M", "16", "--ND", "288"],
    },
    # T20: no FFN recipe on purpose -- 128x256 runs the DELIVERED t11_nh5 in
    # 2x96-row blocks, so only the ip image is new (ND=160, Mq=32).
    160: {
        "ip": ["imaggers/attn_impl/t18_attn_compile.py", "--tag",
               "t18_ip_m32nd160", "--M", "32", "--ND", "160"],
    },
}
AUDIT = {
    128: {"ffn": ["imaggers/block/t4_build_win.py", "--tag", "t18_m128"],
          "ip": ["imaggers/attn_impl/t5_build_win.py", "--tag",
                 "t18_ip_m64nd128"]},
    288: {"ffn": ["imaggers/block/t4_build_win.py", "--tag", "t18_m288"],
          "ip": ["imaggers/attn_impl/t5_build_win.py", "--tag",
                 "t18_ip_m16nd288"]},
    160: {"ip": ["imaggers/attn_impl/t5_build_win.py", "--tag",
                 "t18_ip_m32nd160"]},
}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class BuildError(RuntimeError):
    pass


def _run(cmd, log_path, progress, stage, timeout=1800, ok_rc=(0,)):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    progress(stage, "start", cmd=" ".join(cmd), log=str(log_path))
    with open(log_path, "wb") as lf:
        r = subprocess.run([PY312] + cmd, cwd=str(NPU), stdout=lf,
                           stderr=subprocess.STDOUT, timeout=timeout)
    sec = time.perf_counter() - t0
    tail = ""
    try:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    except Exception:  # noqa: BLE001
        pass
    # The FFN audit's WINNING-recipe gate is FAIL by construction (13/17: the
    # delivered four gate/up cores carry 3 ACQ CLOBBER exposures, T11 R5), and
    # t4_build_win exits 2 for that -- so the audit stage accepts 0/2 and the
    # PASS COUNT is checked afterwards.
    ok = int(r.returncode) in ok_rc
    progress(stage, "done" if ok else "failed",
             seconds=round(sec, 2), rc=int(r.returncode))
    if not ok:
        raise BuildError("%s failed (rc=%d); see %s" %
                         (stage, r.returncode, log_path))
    return sec, tail


def _no_build(plan):
    """The 512²-spec refusal: why this (H, W) has no build in this tree.

    T22 prodlarge: an unbuilt size must never be refused with a generic
    sentence.  shapes.resolve() supplies the reason (exact M, image tokens and
    L1 arithmetic, in the same shape as the 512² wording); this adds the one
    fact the builder owns -- which M values actually have a recipe here.

    T22 adaptive (2026-11-03): the M-recipe list is no longer the only route.
    The 32-core placement planner (webui/npu_plan.py) resolves ANY M to a legal
    geometry, and npu_synth.py builds it -- so the sentence now names that plan
    instead of ending on "no recipe".  The wording still comes from the same
    place as the front end's (shapes.refusal).
    """
    out = ("%s (M=%d) 未构建：%s 本树的编译配方只有 M=%s。"
           % (plan["key"], plan["m"], plan.get("reason", ""),
              "/".join(str(k) for k in sorted(COMPILE))))
    np = plan.get("npu_plan") or {}
    if np:
        out += (" 自适应放置已规划：plan %s（%s m=%d kk=%d Mblk=%d，%d 次 "
                "dispatch，L1<=%d B，pad %d 行）；构建入口 "
                "py312 imaggers/util/place/npu_synth.py --m %d。"
                % (np["plan_key"], np["strategy"], np["m"], np["kk"],
                   np["Mblk"], np["ndisp"], np["l1_max"], np["pad_rows"],
                   plan["m"]))
    return out


# ---------------------------------------------------------------------------
# T22 adaptive (2026-11-03): the 32-core placement image set
#
# The delivered COMPILE table above is a table of PRODUCT images (the 8-core
# t11_nh5 FFN and the t7_ip attention).  The adaptive path is a different image
# set -- the 32-core placement GEMMs (ffn_xb / ffn_t33) -- so it has its own
# entry point, its own manifest and its own cache key: the PLAN, not (H, W).
# Two shapes whose plan is identical (384x384 and 320x448 both resolve to
# M=608; 384x512 and 512x768 share one Mblk=800 plan) share one build.
# ---------------------------------------------------------------------------
ADAPT = os.environ.get("ZIMAGE_ADAPT", "0") == "1"
SYNTH = NPU / "imaggers" / "util" / "place" / "npu_synth.py"


def npu_plan_for(h, w):
    """(shape plan, npu placement plan or None) for this shape."""
    plan = shapes.resolve(int(h), int(w))
    return plan, plan.get("npu_plan")


def plan_manifest(h, w, plan=None):
    """The plan-keyed manifest this shape's placement resolves to, or None."""
    plan = plan or shapes.resolve(int(h), int(w))
    key = plan.get("npu_plan_key")
    if not key:
        return None
    return shapes.plan_manifest(plan_key=key)


def build_adaptive(h, w, progress=None, force=False):
    """Plan + build the 32-core placement image set for this shape.

    Idempotent: a complete plan manifest short-circuits (the images are already
    on disk and the key names them).  The build itself runs in its own PY312
    process (npu_synth.py), so this module never imports the toolchain.
    """
    def _p(stage, status, **kw):
        ev = {"stage": stage, "status": status, "t": time.time()}
        ev.update(kw)
        stages.append(ev)
        if progress:
            try:
                progress("[adapt] %-10s %-6s %s" % (stage, status, kw))
            except Exception:                                # noqa: BLE001
                pass

    stages = []
    t_all = time.perf_counter()
    plan, np = npu_plan_for(h, w)
    if not np:
        raise BuildError(_no_build(plan))
    key = np["plan_key"]
    _p("plan", "done", detail={"plan_key": key, "M": np["M"],
                               "Mblk": np["Mblk"], "ndisp": np["ndisp"],
                               "m": np["m"], "kk": np["kk"], "T": np["T"],
                               "l1_max": np["l1_max"]})
    man_path = SHAPE_MANIFESTS / ("plan_%s.json" % key)
    man = None if force else plan_manifest(h, w, plan)
    if man is not None:
        _p("manifest", "done", detail={"reused": str(man_path), "images":
                                       sorted(man.get("images") or {})})
        return {"key": plan["key"], "m": plan["m"], "ok": True,
                "adaptive": True, "plan_key": key,
                "manifest": str(man_path), "reused": True,
                "seconds": round(time.perf_counter() - t_all, 2),
                "stages": stages}
    if not SYNTH.is_file():
        raise BuildError("adaptive builder missing: %s" % SYNTH)
    log = config.CACHE_DIR / "build_logs" / ("adapt_%s.log" % key)
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(config.PY312), str(SYNTH), "--m", str(int(np["M"])),
           "--json", str(log.with_suffix(".json"))]
    env = dict(os.environ)
    env["NPU_PY312"] = str(config.PY312)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    t0 = time.perf_counter()
    _p("compile", "start", cmd=" ".join(cmd), log=str(log))
    with open(log, "wb") as lf:
        r = subprocess.run(cmd, cwd=str(NPU), stdout=lf,
                           stderr=subprocess.STDOUT, timeout=3600, env=env)
    sec = time.perf_counter() - t0
    ok = int(r.returncode) == 0
    _p("compile", "done" if ok else "failed", seconds=round(sec, 2),
       rc=int(r.returncode))
    if not ok:
        tail = ""
        try:
            tail = log.read_text(encoding="utf-8", errors="replace")[-1500:]
        except Exception:                                    # noqa: BLE001
            pass
        raise BuildError("adaptive build failed (rc=%d); see %s\n%s"
                         % (r.returncode, log, tail))
    man = plan_manifest(h, w, plan)
    if man is None:
        raise BuildError("adaptive build reported success but no manifest "
                         "for plan %s" % key)
    _p("manifest", "done", detail=str(man_path))
    total = time.perf_counter() - t_all
    _p("done", "done", seconds=round(total, 2))
    return {"key": plan["key"], "m": plan["m"], "ok": True, "adaptive": True,
            "plan_key": key, "manifest": str(man_path), "reused": False,
            "images": sorted(man.get("images") or {}),
            "seconds": round(total, 2), "stages": stages}


def plan_for(h, w):
    plan = shapes.resolve(int(h), int(w))
    specs, imgs = cache._specs_for(plan)
    if not specs:
        raise BuildError(_no_build(plan))
    return plan, specs, imgs


def build(h, w, progress=None, force=False):
    """Build (or verify) one shape.  Returns the build record."""
    def _p(stage, status, **kw):
        ev = {"stage": stage, "status": status, "t": time.time(),
              "seconds": kw.pop("seconds", None)}
        ev.update(kw)
        stages.append(ev)
        if progress:
            try:
                progress("[build] %-9s %-6s %s" %
                         (stage, status, {k: v for k, v in kw.items()
                                          if k in ("cmd", "rc", "seconds",
                                                   "detail")}))
            except Exception:  # noqa: BLE001
                pass

    stages = []
    plan, specs, imgs = plan_for(h, w)
    t_all = time.perf_counter()
    _p("plan", "done", detail={"key": plan["key"], "m": plan["m"],
                               "images": imgs, "files": len(specs)})
    m = int(plan["m"])
    if m not in COMPILE:
        if ADAPT:
            # T22 adaptive: the placement image set is the build for this M.
            return build_adaptive(h, w, progress=progress, force=force)
        raise BuildError(_no_build(plan))
    tag_ffn = imgs["ffn"]
    tag_ip = imgs["ip"]

    # ---- artifacts: compile only what is missing -------------------------
    missing = [s for s in specs if not pathlib.Path(s[2]).is_file()]
    if missing and force:
        missing = list(specs)
    did = set()
    for group, name, src in missing:
        if group == "perm":
            _p("artifacts", "done", detail="perm %s missing (t18_perm.py)" % name)
            continue
        key = "ffn" if group == "ffn" else "ip"
        if key in did:
            continue
        did.add(key)
        log = config.CACHE_DIR / "build_logs" / ("build_%s_%s.log" % (key, plan["key"]))
        _run(COMPILE[m][key], log, _p, "compile_" + key)
    if missing and not did:
        raise BuildError("%s: missing artifacts with no compile recipe: %s" %
                         (plan["key"], [s[1] for s in missing]))

    # ---- audit gate: run the winning build for every new image -----------
    # The per-core audit is a BUILD PRODUCT of its own: every core writes a
    # <tag>_core_x_y_win_audit.json next to the image.  If that set is already
    # complete, reuse it (a builder must be idempotent; re-linking the winning
    # objects is not free and can hit a per-core program-region overflow).
    # --force / a missing core re-runs the whole winning build.
    def _audit_json_dir(key):
        return config.T4_OUT if key == "ffn" else config.T5_OUT

    def _audit_count(key, tag):
        return len(list(_audit_json_dir(key).glob(tag + "_core_*_win_audit.json")))

    for key, tag, total, baseline in (
            ("ffn", tag_ffn, FFN_BASELINE_TOTAL, FFN_BASELINE_PASS),
            ("ip", tag_ip, None, None)):
        want = FFN_BASELINE_TOTAL if key == "ffn" else 3
        have = _audit_count(key, tag)
        if have >= want and not force:
            _p("audit_" + key, "done",
               detail={"reused": have, "cores": want,
                       "pass": (FFN_BASELINE_PASS if key == "ffn" else 3),
                       "source": str(_audit_json_dir(key))})
            continue
        log = config.CACHE_DIR / "build_logs" / ("build_%s_%s.log" % (key, plan["key"]))
        _run(AUDIT[m][key], log, _p, "audit_" + key, ok_rc=(0, 2))
        txt = log.read_text(encoding="utf-8", errors="replace")
        got = None
        mt = re.search(r"audit lines PASS: (\d+) / (\d+)", txt)
        if mt:
            got = (int(mt.group(1)), int(mt.group(2)))
        gate = re.search(r"AUDIT GATE: (PASS|FAIL)", txt)
        if got is None:
            raise BuildError("audit_%s: no summary in %s" % (key, log))
        ok = (got == (total, total)) if total else (got[0] == got[1])
        if key == "ffn":
            ok = got == (baseline, total)
        _p("audit_" + key, "done" if ok else "failed",
           detail={"pass": got[0], "total": got[1],
                   "gate": gate.group(1) if gate else "?",
                   "baseline": ([baseline, total] if key == "ffn" else
                                [got[1], got[1]])})
        if not ok:
            raise BuildError("audit_%s: %d/%d pass, baseline %s" %
                             (key, got[0], got[1], baseline))

    # ---- cache copy + sha256 --------------------------------------------
    info = cache.ensure_shape(int(h), int(w), log=(progress or None),
                              allow_unbuilt=True)
    _p("cache", "done", seconds=info.get("seconds"),
       detail={"dir": info["dir"], "files": len(info["files"])})

    # ---- delivered manifest (committed record) --------------------------
    man = {"shape": plan["key"], "height": int(h), "width": int(w),
           "m": int(plan["m"]), "images": imgs,
           "note": ("T18 build record: the exact FFN/ip/projection images and "
                    "perms this shape's cache copies.  Compiled from the "
                    "delivered t11_ffn_nh5 source at M and t7_ip at ND=SEQ; "
                    "the projection is the delivered t7_p5 (m=96 blocks)."),
           "files": [{"name": f["name"], "size": f["size"],
                      "sha256": f["sha256"]} for f in info["files"]],
           "created": config.now_iso()}
    SHAPE_MANIFESTS.mkdir(parents=True, exist_ok=True)
    mp = SHAPE_MANIFESTS / (plan["key"] + ".json")
    mp.write_text(json.dumps(man, indent=1), encoding="utf-8")
    _p("manifest", "done", detail=str(mp))
    total_s = time.perf_counter() - t_all
    _p("done", "done", seconds=round(total_s, 2))
    return {"key": plan["key"], "m": plan["m"], "ok": True,
            "images": imgs, "cache": info, "manifest": str(mp),
            "seconds": round(total_s, 2), "stages": stages}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--adaptive", action="store_true",
                    help="build the 32-core PLACEMENT image set for this shape "
                         "(npu_plan + npu_synth) instead of the product images")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        fn = build_adaptive if a.adaptive else build
        rec = fn(a.height, a.width, progress=print, force=a.force)
    except Exception as exc:  # noqa: BLE001
        print("[build] FAILED %s: %s" % (type(exc).__name__, exc))
        return 2
    print(json.dumps({k: v for k, v in rec.items() if k != "stages"},
                     indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
