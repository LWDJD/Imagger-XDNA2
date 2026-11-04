# SPDX-License-Identifier: MIT
"""Prebuild/warm the per-shape cache (T16 P0-5).

For a BUILT shape this prepares/verifies webui/cache/{H}x{W}/ (instant when the
manifest is valid).  For a shape with no device build it returns the exact build
plan instead of pretending to build: the six images + permutations + host
constants + the four gates are a design unit, and a "prebuild" that silently did
nothing would be the same half-built illusion the owner rejected.

CLI:
    py312 webui/prebuild.py                 # all presets (128 warms, others report)
    py312 webui/prebuild.py --height 128 --width 128
    py312 webui/prebuild.py --presets --json
"""
from __future__ import annotations

import argparse
import json
import sys

from . import cache, config, shapes


def _unbuilt_reason(h, w, plan):
    """The specific, user-facing reason this shape has no build.

    T22 prodlarge: every unbuilt size speaks in the 512² spec's shape -- one
    sentence with the exact M / L1 arithmetic and the verdict.  shapes.resolve
    generates it, and for the legacy fixed profiles it returns
    config.RESOLUTIONS' frozen text verbatim (that is the reference wording),
    so this function only has to prefer config when resolve has nothing.
    Display text only -- nothing here starts a compile.
    """
    h, w = int(h), int(w)
    meta = config.RESOLUTIONS.get(h) if h == w else None
    if meta and not meta.get("ready") and meta.get("reason"):
        return meta["reason"]
    return plan.get("reason") or ""


def _buildable(plan):
    """True when the builder has a compile recipe for this M (read-only)."""
    try:
        from . import build_shape
        return int(plan["m"]) in build_shape.COMPILE
    except Exception:  # noqa: BLE001
        return False


def prebuild(h, w, log=None, build=False, force=False):
    """Verify/prepare the cache.  build=True runs the REAL builder.

    T18: webui.build_shape.build() is now present, so an un-built shape can be
    built for real (compile missing artifacts -> audit gate -> cache -> shape
    manifest) and the returned record carries its stage stream.  Without
    build=True an un-built shape still returns the plan, never a fake success.
    """
    plan = shapes.resolve(h, w)
    out = {"key": plan["key"], "height": int(h), "width": int(w),
           "m": plan["m"], "ready": plan["ready"]}
    # T22 prodlarge: the geometry the size panel needs for a typed (non-preset)
    # size.  prebuild is the only endpoint that answers for an arbitrary (H, W)
    # without warming anything, so it carries them for both branches.
    for _k in ("image_tokens", "padded_tokens", "patch_grid", "latent",
               "m_scale_vs_128", "loop_estimate_s_per_image"):
        out[_k] = plan.get(_k)
    if not plan["ready"]:
        out["built"] = False
        out["exec_mode"] = plan.get("exec_mode", "none")
        out["chunked_runnable"] = bool(plan.get("chunked_runnable"))
        out["chunk_estimate"] = plan.get("chunk_estimate")
        out["buildable"] = _buildable(plan)
        if plan.get("chunked_runnable"):
            # T22 chunk_path: no compile.  The "prebuild" is the cache copy of
            # the delivered M=96 images + the generated shape package.
            out["reason"] = plan.get("chunk_reason", "")
            out["missing"] = []
            out["budget_t22"] = plan.get("budget_t22")
            try:
                info = cache.ensure_shape(int(h), int(w), log=log)
                out.update({"ok": True, "cache": info, "chunked": True})
                if log:
                    log(f"[prebuild] {plan['key']}: CHUNKED cache "
                        f"{'HIT' if info['cached'] else 'PREPARED'} "
                        f"{info['dir']} ({info['seconds']:.2f} s)")
            except Exception as exc:  # noqa: BLE001
                out["ok"] = False
                out["error"] = f"{type(exc).__name__}: {exc}"
                if log:
                    log(f"[prebuild] {plan['key']}: CHUNKED FAILED {out['error']}")
            return out
        out["reason"] = _unbuilt_reason(h, w, plan)
        out["missing"] = plan.get("missing", [])
        out["budget"] = {k: plan.get(k) for k in
                         ("a_bytes", "sbg_bytes", "y_bytes",
                          "m_scale_vs_128", "loop_estimate_s_per_image")}
        # T22: the campaign-1 size budget travels with the refusal (M, nq/nk,
        # dispatches, projected device time) -- same 口径 as T22-BUDGET.md.
        out["budget_t22"] = plan.get("budget_t22")
        out["ok"] = False
        # The reason IS the message: it already ends in "本树未构建，不可用。"
        out["error"] = out["reason"]
        if build and out["buildable"]:
            from . import build_shape
            try:
                rec = build_shape.build(int(h), int(w), progress=log,
                                        force=force)
                out.update({"ok": True, "built": True,
                            "stages": rec["stages"],
                            "cache": rec["cache"],
                            "manifest": rec["manifest"],
                            "seconds": rec["seconds"],
                            "images": rec["images"]})
                if log:
                    log(f"[prebuild] {plan['key']}: BUILT in "
                        f"{rec['seconds']:.2f} s ({len(rec['stages'])} stages)")
            except Exception as exc:  # noqa: BLE001
                out["ok"] = False
                out["error"] = f"{type(exc).__name__}: {exc}"
                if log:
                    log(f"[prebuild] {plan['key']}: BUILD FAILED {out['error']}")
            return out
        if not out["buildable"]:
            # No compile recipe / artifact spec for this M: say so instead of
            # "call build=true", and never start a compile for it.
            out["error"] += (" 本树没有 M=%d 的编译配方 / artifact spec："
                             "build=true 也不会构建（未触发任何编译）。"
                             % plan["m"])
            out["build_note"] = ("no compile recipe for M=%d (build_shape.COMPILE "
                                 "only has %s)" % (plan["m"], "M=128/160/288"))
        if log:
            log(f"[prebuild] {plan['key']}: BUILD REQUIRED (M={plan['m']}); "
                f"{out['error']}")
        return out
    try:
        info = cache.ensure_shape(int(h), int(w), log=log)
        out.update({"ok": True, "cache": info})
        if log:
            log(f"[prebuild] {plan['key']}: {'HIT' if info['cached'] else 'PREPARED'} "
                f"{info['dir']} ({info['seconds']:.2f} s)")
    except Exception as exc:  # noqa: BLE001
        out["ok"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"
        if log:
            log(f"[prebuild] {plan['key']}: FAILED {out['error']}")
    return out


def prebuild_presets(log=None, build=False):
    return [prebuild(p["height"], p["width"], log=log, build=build)
            for p in shapes.PRESETS]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int)
    ap.add_argument("--width", type=int)
    ap.add_argument("--presets", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if a.height is not None:
        res = [prebuild(a.height, a.width if a.width is not None else a.height)]
    else:
        res = prebuild_presets()
    if a.json:
        print(json.dumps(res, indent=1, ensure_ascii=False))
    else:
        for r in res:
            state = ("READY" if r.get("ok") else "BUILD REQUIRED")
            print(f"  {r['key']:>9s}  M={r['m']:>4d}  {state}"
                  + ("" if r.get("ok") else f"  ({len(r.get('missing', []))} items)"))
    return 0 if all(r.get("ok") for r in res) else 2


if __name__ == "__main__":
    raise SystemExit(main())
