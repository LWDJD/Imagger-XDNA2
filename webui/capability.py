# SPDX-License-Identifier: MIT
"""capability.py -- what a user can ACTUALLY get today, derived from the code.

The WebUI used to answer "can I render 1024x1024?" from one number
(shapes.chunk_gate) and a hand-written table.  This module is the answer the
product can stand behind, and it is not a table: every field comes from the
code that decides the question, or from a job record that measured it.

    shapes.resolve()          ready / exec_mode / chunk estimate / plan
    shapes.chunk_gate()       runnable + the binding constraint
    shapes.plan_manifest()    is the adaptive plan's image set BUILT
    webui/npu_plan.plan()     the geometry / dispatch count the NPU would run
    webui/jobs/*/job.json     the wall this exact run size actually took

Nothing here is a speed claim: an estimate is labelled "model", a wall is
labelled with the job id that measured it.

    py312 -m webui.capability --hw 1920 1080
    py312 -m webui.capability --table
"""
from __future__ import annotations

import json
import os
import pathlib

from . import config, shapes

# the three paths a request can take through the product today
PATHS = {
    "native": ("专用镜像", "a gated per-shape device image set exists (BUILT, "
                            "audited); the fastest and the only path with a "
                            "byte-identity gate per shape"),
    "chunked": ("分块·慢", "no dedicated kernel: the delivered M=96 projection/FFN/"
                            "o images run in 96-row blocks and attention runs on "
                            "the host in f32 torch"),
    "refused": ("不可用", "refused at submit; the reason names the binding constraint"),
}

# a run record older than this is still quoted, but labelled with its date
_HIST_CACHE = {"key": None, "idx": None}


def _job_files():
    d = config.JOBS_DIR
    try:
        return sorted(p for p in d.iterdir() if (p / "job.json").is_file())
    except Exception:                                          # noqa: BLE001
        return []


def history(refresh=False) -> dict:
    """{(run_h, run_w): [record, ...]} from the recorded jobs.

    A record is the measured wall of ONE job at that run size.  Only jobs that
    reached "done" are used for a wall; failures are kept so the answer can say
    "this exact size was tried and died".
    """
    files = _job_files()
    key = (len(files), max((p.stat().st_mtime for p in files), default=0))
    if not refresh and _HIST_CACHE["key"] == key:
        return _HIST_CACHE["idx"]
    idx = {}
    for d in files:
        try:
            j = json.loads((d / "job.json").read_text(encoding="utf-8"))
        except Exception:                                      # noqa: BLE001
            continue
        run = j.get("padded") or j.get("requested") or {}
        h, w = run.get("height"), run.get("width")
        if not h or not w:
            continue
        t = j.get("timing") or {}
        images = j.get("images") or [{}]
        rec = {
            "job": j.get("id") or d.name,
            "created": (j.get("created") or "")[:19],
            "status": j.get("status"),
            "steps": j.get("steps"),
            "memory_mode": j.get("memory_mode"),
            "exec_mode": j.get("exec_mode"),
            "total_s": round(float(t["total_s"]), 1) if t.get("total_s") else None,
            "loop_s": round(float(t["loop_s"]), 1) if t.get("loop_s") else None,
            "vae": bool(j.get("params", {}).get("vae")),
            "png_sha": (images[0].get("sha256") or "")[:16] if images else "",
            # the launcher env that selected this job's device path (recorded
            # from 2026-11-03 on; older jobs carry None)
            "env": j.get("env_flags"),
            "error": (str((j.get("error") or {}).get("message")
                          if isinstance(j.get("error"), dict) else
                          (j.get("error") or "")))[:160],
        }
        idx.setdefault((int(h), int(w)), []).append(rec)
    for v in idx.values():
        v.sort(key=lambda r: r["created"])
    _HIST_CACHE["key"], _HIST_CACHE["idx"] = key, idx
    return idx


def _measured(run_h, run_w, steps=8):
    """(best, best at this step count, worst, n_done, n_error, records).

    "best" is the fastest completed job at this run size of ANY step count;
    "best_n" is the fastest one that ran the requested number of steps, which
    is the only one comparable to an 8-step estimate.  A 1-step job quoted as
    the wall of an 8-step shape is exactly the kind of number this module is
    here to stop.
    """
    recs = history().get((int(run_h), int(run_w)), [])
    done = [r for r in recs if r["status"] == "done" and r["total_s"]]
    errs = [r for r in recs if r["status"] == "error"]
    # "server restarted while this job was running" is infrastructure, not a
    # property of the shape -- never quote it as the shape's failure.
    driver = [r for r in errs if "server restarted" not in (r["error"] or "")]
    at_n = [r for r in done if int(r.get("steps") or 0) == int(steps)]
    best = min(done, key=lambda r: r["total_s"]) if done else None
    best_n = min(at_n, key=lambda r: r["total_s"]) if at_n else None
    worst = max(done, key=lambda r: r["total_s"]) if done else None
    _measured.driver_errors = driver
    return best, best_n, worst, len(done), len(errs), recs


def _env_label(env):
    """The flags of a recorded job that actually pick a device path."""
    if not env:
        return ""
    keys = ("ZIMAGE_BIGM", "ZIMAGE_BIGM_PROJ", "ZIMAGE_CHUNK_PIPE",
            "T22_HOSTKILL", "T22_DEVKILL", "T22_FASTGLUE", "T22_PLAN_FFN")
    on = ["%s=%s" % (k, env.get(k)) for k in keys
          if env.get(k) not in (None, "", "0")]
    return " ".join(on)


def _env_now():
    return {k: os.environ.get(k) for k in config.LAUNCH_FLAGS}


def _plan_facts(m):
    from . import npu_plan
    res = shapes.npu_plan_for(m)
    if not res.get("ok"):
        return {"ok": False, "reason": res.get("reason"),
                "binding": res.get("binding")}
    pl = shapes.plan_summary(res["plan"])
    key = pl["plan_key"]
    man = shapes.plan_manifest(plan_key=key)
    return {"ok": True, "summary": pl, "built": man is not None,
            "built_at": (man or {}).get("created"),
            "manifest": "webui/shape_manifests/plan_%s.json" % key,
            "used_by": sorted((man or {}).get("used_by") or []),
            "strategy": pl["strategy"], "Mblk": pl["Mblk"], "ndisp": pl["ndisp"],
            "dispatches_ms": pl["layer_ms"]}


def envelope() -> dict:
    """The code-derived size envelope: what the gates admit at all."""
    ceiling = shapes.CHUNK_REFINER_MAX_BYTES
    try:
        ceiling = int(float(os.environ["ZIMAGE_CHUNK_REFINER_GB"]) * 2**30)
    except Exception:                                          # noqa: BLE001
        pass
    # largest padded image-token count whose refiner score matrix fits
    # UNBLOCKED -- kept because it is the number the 2026-11-03 refusal quoted,
    # and because T22_REF_QCHUNK=off restores exactly that behaviour.
    s_max = int((ceiling / (4.0 * shapes.T22_HEADS)) ** 0.5)
    side = 16
    best = (16, 0)
    while side <= shapes.MAX_HW:
        tok = (side // 16) ** 2
        if tok <= s_max:
            best = (side, tok)
        side += 16
    blocked = shapes.chunk_refiner_blocked()
    # with query blocking the refiner's tensor is bounded at any S, so the
    # binding limit becomes the product envelope itself
    s_now = int(((shapes.MAX_HW // 16) ** 2) if blocked else s_max)
    return {
        "refiner_max_bytes": ceiling,
        "refiner_max_gib": round(ceiling / 2**30, 2),
        "max_image_tokens": s_max,
        "max_square_side": best[0],
        "max_square_tokens": best[1],
        "refiner_query_blocked": blocked,
        "max_image_tokens_now": s_now,
        "refiner_block_rows": (shapes.chunk_refiner_rows(s_now) if blocked
                               else None),
        "refiner_block_bytes": (shapes.chunk_refiner_peak_bytes(
            32 + s_now, s_now) if blocked else None),
        "native_max_m": max(int(row["m"]) for row in shapes.BUILT.values()),
        "note": (
            ("the CPU noise refiner blocks its query axis above %.2f GiB, so "
             "its f32 score matrix is one %d-row block at any size and the "
             "refiner no longer limits the envelope (%d image tokens = 2048x2048)"
             % (shapes._env_gb(shapes.REF_QCEIL_ENV, shapes.REF_QCEIL_GB),
                shapes.chunk_refiner_rows(s_now), s_now))
            if blocked else
            ("padded image tokens (H/16)*(W/16) <= %d keeps the CPU "
             "refiner's f32 score matrix under %.2f GiB; every larger size "
             "is refused at submit with that arithmetic (T22_REF_QCHUNK=off)"
             % (s_max, ceiling / 2**30))),
    }


def describe(h, w) -> dict:
    """The honest answer for one request, in the product's current env."""
    h, w = int(h), int(w)
    plan = shapes.resolve(h, w)
    m = int(plan["m"])
    run_h, run_w = int(plan["height"]), int(plan["width"])
    est = plan.get("chunk_estimate") or {}
    best, best8, worst, n_done, n_err, recs = _measured(run_h, run_w)
    pf = _plan_facts(m) if m > shapes.CHUNK_MIN_SEQ else {"ok": False,
                                                          "reason": "M <= 304"}
    if plan.get("ready"):
        path = "native"
    elif plan.get("chunked_runnable"):
        path = "chunked"
    else:
        path = "refused"
    out = {
        "requested": {"height": h, "width": w},
        "run": {"height": run_h, "width": run_w},
        "output": {"height": h, "width": w},
        "pad_applied": bool(plan.get("pad_applied")),
        "crop": plan.get("crop"),
        "m": m,
        "image_tokens": int((plan.get("budget") or {}).get("image_tokens")
                            or (run_h // 16) * (run_w // 16)),
        "shape": plan.get("key"),
        "path": path,
        "path_label": PATHS[path][0],
        "renderable": path != "refused",
        "refusal": plan.get("chunk_gate_reason") or "",
        "native": {"ready": bool(plan.get("ready")),
                   "label": plan.get("label") if plan.get("ready") else "",
                   "images": plan.get("images") if plan.get("ready") else []},
        "exec_mode": plan.get("exec_mode"),
        "plan": pf,
        "estimate": {
            "source": "webui/shapes.chunk_estimate (model, not a promise)",
            "step_s": est.get("step_s"),
            "s_8step": est.get("s_8step"),
            "h_8step": est.get("h_8step"),
            "disp_8step": est.get("dispatches_8step"),
            "attention": est.get("attention"),
            "refiner_gib": est.get("refiner_gib"),
            "anchor": est.get("anchor"),
        } if est else None,
        "measured": {
            "n_done": n_done, "n_error": n_err,
            "n_error_driver": len(_measured.driver_errors),
            "best": best, "best_8step": best8, "worst": worst,
            "recent": recs[-3:],
            # did the measured wall come from the env we are in right now?
            "env_matches_now": bool(best and best.get("env")
                                    and _env_label(best["env"]) ==
                                    _env_label(_env_now())),
        },
        "flags": _flags(m, pf),
    }
    out["answer"] = _answer(out)
    return out


def _flags(m, pf):
    """What the env can change for THIS shape, with the cost of changing it."""
    fl = []
    fl.append({
        "env": "ZIMAGE_ADAPT=1",
        "effect": "require the adaptive plan's image set to be BUILT before the "
                  "job is accepted (the plan is %s)"
                  % ("BUILT: " + str(pf.get("summary", {}).get("plan_key"))
                     if pf.get("built") else "NOT built for this M"),
        "changes_wall": False,
    })
    fl.append({
        "env": "T22_PLAN_FFN=1 (+ZIMAGE_CHUNK_PIPE=0, ZIMAGE_BIGM=1, "
               "T22_DEVKILL=1)",
        "effect": "dispatch the plan's gu/dn images instead of the delivered "
                  "M=96 FFN.  It RAN end-to-end at a new size for the first "
                  "time on 2026-11-03 (384x384, 8 steps, job "
                  "20261103-152409-e8e997) and is SLOWER (loop 160.6 s vs "
                  "119.4 s on the delivered path) and it does not reproduce "
                  "the delivered image (latents corr 0.35, and 0.47 against "
                  "the same env without the flag).  Only 'xd' has an executor; "
                  "t33 raises.  NOT a product path.",
        "changes_wall": True,
    })
    fl.append({
        "env": "ZIMAGE_VAE_FAST=0",
        "effect": "use the shipped diffusers loader for the VAE instead of "
                  "webui/vae_lean.py (byte-identical PNGs, +15-18 s per job)",
        "changes_wall": True,
    })
    fl.append({
        "env": "ZIMAGE_VAE_THREADS=N",
        "effect": "pin the CPU VAE decode's torch intra-op count.  Unset = the "
                  "geometry rule: 20 threads when latent_H*latent_W % 5 == 0 "
                  "(bit-identical: 1088x1920, 320^2/640^2/800^2/960^2), else 10 "
                  "= today's bytes.  0 = torch's default",
        "changes_wall": True,
    })
    fl.append({
        "env": "ZIMAGE_VAE_THREAD_RULE=0",
        "effect": "disable the geometry rule: the decode follows "
                  "ZIMAGE_VAE_THREADS or torch's default (the pre-2026-11-03 "
                  "behaviour, whose bytes are this box's core count)",
        "changes_wall": True,
    })
    fl.append({
        "env": "ZIMAGE_CHUNK_THREADS=N",
        "effect": "the device loop's host-attention pool; unset = torch's own "
                  "default (10 intra-op threads on this 20-logical-CPU box).  "
                  "It does NOT move the VAE decode (decoupled 2026-11-03)",
        "changes_wall": True,
    })
    return fl


def _answer(d):
    rq = d["requested"]
    head = "%dx%d" % (rq["width"], rq["height"])
    run = d["run"]
    run_s = ("" if run == rq else
             (" -> runs %dx%d (16-px pad) and crops back"
              % (run["width"], run["height"])))
    if d["path"] == "refused":
        why = d["refusal"]
        # shapes.refusal() prefixes "<run_key> (M=<m>) 未构建：" -- the sentence
        # already says which shape and which M, so drop the prefix here.
        cut = why.find("未构建：")
        if cut >= 0:
            why = why[cut + len("未构建："):]
        return "%s%s (M=%d): 不可用 — %s" % (head, run_s, d["m"], why)
    if d["path"] == "native":
        best = d["measured"]["best"]
        wall = ("measured %.1f s for this exact run size (%s, %s steps, "
                "VAE %s)" % (best["total_s"], best["job"], best["steps"],
                             "on" if best["vae"] else "off") if best else
                "no job recorded at this exact run size in this tree")
        return ("%s%s (M=%d): 可跑 via its own device image set (%s); %s."
                % (head, run_s, d["m"], d["native"]["label"], wall))
    est, best = d["estimate"] or {}, d["measured"]["best_8step"]
    wall = ("model %.0f s/step, 8 steps %.2f h" % (est["step_s"], est["h_8step"])
            if est.get("step_s") else "no estimate")
    meas = ("; measured on this host: %.0f s total for 8 steps, VAE %s "
            "(%s, %s%s)"
            % (best["total_s"], "on" if best["vae"] else "off",
               best["job"], best["created"],
               ("; same launcher env as now"
                if d["measured"].get("env_matches_now") else
                ("; launcher env not recorded (pre-2026-11-03 job)"
                 if not best.get("env") else
                 "; launcher env " + _env_label(best.get("env")))))
               if best else
            "; no completed 8-step job at this exact size in this tree")
    bad = [r for r in d["measured"]["recent"]
           if r["status"] == "error"
           and "server restarted" not in (r["error"] or "")]
    if bad:
        meas += ("; %d failed run(s) recorded, last: %s (%s s)"
                 % (d["measured"].get("n_error_driver") or len(bad),
                    (bad[-1]["error"] or "?")[:80],
                    int(bad[-1]["total_s"] or 0)))
    return ("%s%s (M=%d): 可跑 but slow — 分块 path (delivered M=96 images + "
            "host f32 attention); %s%s." % (head, run_s, d["m"], wall, meas))


_TABLE_CACHE = {"key": None, "t": 0.0, "val": None}


def table(ttl=60.0, refresh=False) -> dict:
    """Every size the product can speak about, from the code + the history.

    Memoised for a minute on (env, history file count/mtime): the API is polled
    by the page and re-resolving every row is ~40 plan calls.
    """
    import time as _t
    files = _job_files()
    key = (tuple(sorted((k, os.environ.get(k)) for k in config.LAUNCH_FLAGS)),
           len(files), max((p.stat().st_mtime for p in files), default=0))
    c = _TABLE_CACHE
    if (not refresh and c["val"] is not None and c["key"] == key
            and (_t.time() - c["t"]) < ttl):
        return c["val"]
    val = _table_uncached()
    _TABLE_CACHE.update({"key": key, "t": _t.time(), "val": val})
    return val


def _table_uncached() -> dict:
    """Rows: the 12 UI presets, every native BUILT shape, every run size that
    has a recorded job, and every M whose adaptive plan is BUILT (named by M,
    since one M covers many HxW)."""
    rows, seen = [], set()

    def add(h, w, source):
        k = (int(h), int(w))
        if k in seen:
            return
        seen.add(k)
        try:
            d = describe(h, w)
        except Exception as exc:                               # noqa: BLE001
            rows.append({"requested": {"height": h, "width": w},
                         "source": source, "error": str(exc)[:160]})
            return
        d["source"] = source
        rows.append(d)

    for _group, _label, h, w in shapes.preset_pairs():
        add(h, w, "preset")
    for key in sorted(shapes.BUILT):
        b = shapes.BUILT[key]
        add(b["height"], b["width"], "native-built")
    for (rh, rw) in sorted(history()):
        add(rh, rw, "measured")
    built = sorted(shapes.built_plan_keys())
    out = {
        "ok": True,
        "generated_in_env": {k: os.environ.get(k) for k in (
            "ZIMAGE_ADAPT", "ZIMAGE_CHUNK_THREADS", "ZIMAGE_VAE_FAST",
            "ZIMAGE_VAE_THREADS", "ZIMAGE_VAE_THREAD_RULE", "ZIMAGE_VAE_NPU",
            "T22_PLAN_FFN",
            "T22_FASTGLUE", "T22_DEVKILL", "ZIMAGE_BIGM")},
        "envelope": envelope(),
        "paths": {k: v[0] for k, v in PATHS.items()},
        "plans_built": len(built),
        "plan_keys": built,
        "rows": rows,
        "native_m": sorted({int(v["m"]) for v in shapes.BUILT.values()}),
    }
    return out


def _main(argv=None) -> int:
    import argparse
    import sys
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hw", nargs=2, type=int, metavar=("H", "W"))
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if a.hw:
        print(json.dumps(describe(*a.hw), indent=1, ensure_ascii=False))
        return 0
    if a.table:
        t = table()
        print(json.dumps(t["envelope"], indent=1))
        for r in t["rows"]:
            if r.get("error"):
                print("%-11s ERROR %s" % ("%dx%d" % (r["requested"]["height"],
                                                     r["requested"]["width"]),
                                          r["error"]))
                continue
            b = r["measured"]["best_8step"] or r["measured"]["best"]
            print("%-11s M=%-6d %-9s %-24s %s" % (
                "%dx%d" % (r["requested"]["height"], r["requested"]["width"]),
                r["m"], r["path_label"],
                ("model %.0f s/step, 8 steps %.2f h" % (r["estimate"]["step_s"],
                                                        r["estimate"]["h_8step"]))
                if r.get("estimate") else "-",
                ("measured %.0f s %s (%s steps, VAE %s)"
                 % (b["total_s"], b["job"], b["steps"],
                    "on" if b["vae"] else "off")) if b else ""))
        return 0
    ap.error("give --hw H W or --table")
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
