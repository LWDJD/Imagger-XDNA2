# SPDX-License-Identifier: MIT
"""Serial job runner: one NPU job at a time, live stage parsing, gate mode.

The runner never edits the delivered drivers.  A job is:
  cache verify -> text encode (py312 worker) -> py310 bridge -> VAE (py312 worker)
and every child is a fresh process, so a known cold-start 0xC0000005 cannot
poison the next job.  The queue is strictly serial: the device is single-tenant.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import queue
import re
import secrets
import shutil
import threading
import time
import traceback

import numpy as np

from . import cache, config, dev_worker, schedule, shapes, textenc
from .proc import stream_process

EVENT_RE = re.compile(r"^@@WEBUI@@\s+(\{.*\})\s*$")
STEP_RE = re.compile(r"^\s*step\s+(\d+):")
# t22_loud: the hook count is a FACT about the path, so it is parsed out of the
# child's own line into the job record instead of being left in a log ring.
FASTGLUE_HOOKS_RE = re.compile(r"\[fastglue\] installed (\d+)/(\d+) hooks")
# T23 SKIP7.  A skipped step makes NO model call, so the arm's identity is not
# visible in any timing or any per-layer line -- it is visible ONLY here.  If
# this line is missing from a run whose env asked for a skip, the ask was
# silently ignored and the run is the BASE arm wearing the skip arm's name.
SKIP_RE = re.compile(r"\[skip\] (\S+) -> pattern \[([^\]]*)\] "
                     r"\(policy (\S+)\)(?:: (.+))?$")
CRASH_CODES = (-1073741819, 3221225477, 0xC0000005)
LOG_TAIL = 400

# T22 workervae: commit headroom the worker's residency policy must leave free
# for the host's text-encoder backend (GPU; measured ~6.7 GB on this host).  The
# worker warms BEFORE the TE loads in this same server, so the snapshot the
# policy takes does not contain it.  Without the reserve the NEXT job's memory
# decision reads ~7 GB, flips to low mode and drops the worker -- so the hot
# startup it was kept for never happens (measured 2026-10-20 13:33).
# 0 restores the old residency policy; it is an env knob so the A/B needs no
# edit here.
WORKER_TE_RESERVE_GB = float(os.environ.get("ZIMAGE_DEV_WORKER_RESERVE_GB",
                                            "7.0"))

# T22 workervae: EXPLICIT residency for the worker.  auto + reserve was measured
# to be snapshot-dependent (0 vs 6.38 GB private in two windows of the same
# day, T22-WORKERVAE.md 11.2), which is not a product setting.  The
# deterministic point is: payload resident (3.71 GB), refiner re-read inside
# each job (~2.6 s) -- steady startup is then a constant instead of a lottery.
# "auto" restores the old policy; both are env-overridable for the A/B.
WORKER_KEEP_PAYLOAD = os.environ.get("ZIMAGE_DEV_WORKER_KEEP_PAYLOAD", "1")
WORKER_KEEP_REFINER = os.environ.get("ZIMAGE_DEV_WORKER_KEEP_REFINER", "0")

# T22 te cache: the encoder's own commit footprint (8.05 GB GGUF; 6.74 GB of
# private commit measured on this host, T21 2a-a).  Used ONLY by the residency
# policy below: keeping the model loaded is allowed only if the next job's
# device loop can still take the fast path with it resident.  The job history's
# commit headroom here is 9.0-9.3 GB, so auto keeps the encoder one-shot --
# exactly the state that made 2a-a a 3x net loss when it was forced on.
# off/on/auto; auto is the product default, ZIMAGE_TE_RESIDENT still forces the
# textenc module itself for the A/B.
TE_MODEL_COMMIT_GB = float(os.environ.get("ZIMAGE_TE_MODEL_GB", "7.0"))
TE_MODEL_RESIDENT = os.environ.get("ZIMAGE_TE_MODEL_RESIDENT", "auto").strip().lower()

# T22 qkv (campaign t22_qkv_prod): the fused projection image is **ON by
# default since 2026-10-20** (owner decision; config.T22_QKV_DEFAULT carries
# the evidence and the rollback knobs).  A job's own "qkv" field overrides the
# service default, and the flag is ALWAYS sent explicitly (--qkv / --no-qkv)
# so the page toggle cannot be overridden by a default further down the chain.
# The bridge forwards it to the driver, so the page, the bridge, the resident
# worker and the driver agree.
_QKV_ENV = config.T22_QKV_DEFAULT
_XPACK_ENV = config.T22_XPACK_DEFAULT
# T22 p10 (campaign t22_p15prod): the o projection's own image -- **ON by
# default since 2026-10-20** (owner decision on PROD-REPORT.md;
# config.T22_P10_DEFAULT carries the evidence and the rollback knobs).  A job's
# own "p10" field overrides the service default and the flag is ALWAYS sent
# explicitly (--p10 / --no-p10), so the page, the resident worker and the
# driver cannot disagree.
_P10_ENV = config.T22_P10_DEFAULT

# P0-4: every second a job spends must land in exactly one of these buckets.
# queue_s starts at submit; the rest run inside the worker.  gate_second_s is
# the whole second pass of the reproducibility gate.  `unaccounted_s` is what
# is left; the acceptance gate is |unaccounted_s| <= 1.0 s and it is shown in
# the UI instead of being folded into `total_s`.
# T22 closurefix: three process-cost segments the closure could not see.  The
# reconstructed closure of the standalone bridge+py312 rig missed +4.58..+5.94 s
# (PROD-REPORT §3/§5); the real 16:06 job missed +1.12 s.  All of it is time a
# job really spent, so it gets a bucket instead of an explanation:
#   bridge_import_s  parent spawn mark -> the child's FIRST @@WEBUI@@ event
#                    (py310 interpreter + bridge imports).  On the delivered
#                    bridge this used to be swallowed by startup_s; startup_s is
#                    narrowed to first_event -> startup_done, so the two add up
#                    to the OLD startup_s exactly (attribution, not new time).
#   vae_spawn_s      parent-observed wall of a one-shot vae_worker.py MINUS the
#                    child's own total_s (py312 interpreter/imports/teardown).
#                    Zero on the resident VAE (in-process or worker link),
#                    which is why the product default path never pays it.
#   worker_drop_s    killing a LIVE resident device worker because this job
#                    cannot use it (4 contexts + ~4 GB resident).  Measured
#                    0.67 s on 2026-10-20 21:13 and 1.0 s at 16:06 -- the exact
#                    size that made the 16:06 job FAIL its closure.
#   prewarm_pack_s   seconds the job BLOCKED on the sibling packer (cold build).
#                    MOVED out of startup_s, never added: the packer's own wall
#                    stays out of every bucket because it overlaps TE/VAE
#                    (T22-PREWARM §6), and startup_s is again exactly
#                    startup_done - first_event, i.e. what a reconstruction can
#                    compute from a log.
TIMING_BUCKETS = (
    ("queue_s", "排队"),
    ("prepare_s", "构建缓存"),
    ("prewarm_pack_s", "冷建打包等待"),
    ("textenc_s", "文本编码"),
    ("schedule_s", "调度"),
    ("retry_s", "冷启动重试"),
    ("worker_drop_s", "常驻 worker 停用"),
    ("bridge_import_s", "桥进程启动/import"),
    ("startup_s", "启动/注册/载荷驻留"),
    ("loop_s", "扩散"),
    ("device_tail_s", "器件收尾/落盘"),
    ("vae_spawn_s", "VAE 子进程启动"),
    ("vae_load_s", "VAE 加载"),
    ("vae_decode_s", "VAE 解码 (CPU)"),
    ("vae_npu_s", "VAE 解码 (NPU)"),
    ("vae_save_s", "保存 PNG"),
    ("vae_other_s", "VAE 收尾(读 latents/sha256/日志)"),
    ("gate_second_s", "复现门第二遍"),
    ("post_s", "任务收尾"),
)
CLOSURE_TOL_S = 1.0

# T19: the device stage used to be capped at a flat 1800 s per attempt.  That
# is a 128x128 number (~97 s measured).  256x256 MEASURED 3551-4651 s (59-78
# min) for 8 steps back then, because its three 96-row FFN block dispatches hit
# the intermittent ERT state-8 (T18-P0-5-ARBITRARY-M.md §6, T19-256x256.md §4:
# 200-264 retries, 7 s wait + 10 s backoff each, 96-97 % of the wall).  A flat
# cap would have killed a *correct* 256x256 job, so the cap scaled with the
# measured worst per-layer-step wall; ZIMAGE_DEVICE_TIMEOUT_S overrides it.
#
# T20: the state-8 root cause is fixed (T20-STATE8.md) -- 256x256 is now 0
# retries, loop 89 s.  The per-layer-step table is the T20 measurement with a
# wide margin, and the 1800 s floor is again the binding cap for every shape
# (12.6x the measured 256x256 device time), so a *stuck* job can no longer hold
# the device for 3.4 h.
#   M= 96: 128x128 loop 27.7-31.2 s / 8 / 30 = 0.115-0.130 s per layer-step
#   M=128: 192x128 loop 47.1 s / 8 / 30      = 0.196 s
#   M=160: 128x256 loop 58.3 s / 8 / 30      = 0.243 s   (T20)
#   M=288: 256x256 loop 89.4 s / 8 / 30      = 0.373 s   (T20; was 34.6 s)
_LAYER_STEP_S = {96: 0.5, 128: 0.2, 160: 0.3, 288: 0.6}
DEVICE_TIMEOUT_MARGIN = 1.5


def device_timeout_s(steps: int, m: int, layers: int = 30,
                     exec_mode: str = "native", estimate=None) -> float:
    """Wall cap for ONE device attempt at a shape (m = sequence length).

    T22 chunk_path: the chunked path's estimate is per 8-step run; the cap is
    1.5x that estimate with the delivered 1800 s floor.  A shape whose estimate
    exceeds 6 h is refused up front by shapes.chunk_gate, so this cap cannot
    grow without a matching admission check.
    """
    if exec_mode == "chunked" and estimate:
        est = float(estimate.get("s_8step") or 0.0)
        if est > 0:
            return max(1800.0, est * float(steps) / 8.0 * 1.5)
    per = _LAYER_STEP_S.get(int(m), 4.0) * DEVICE_TIMEOUT_MARGIN
    return max(1800.0, per * float(layers) * float(steps))


def timing_closure(timing, total_s):
    """Sum the named buckets and report what is left over."""
    parts = {k: float(timing.get(k) or 0.0) for k, _ in TIMING_BUCKETS}
    s = sum(parts.values())
    delta = float(total_s) - s
    return {
        "buckets": [{"key": k, "label": lab, "seconds": parts[k]}
                    for k, lab in TIMING_BUCKETS],
        "sum_s": s,
        "total_s": float(total_s),
        "unaccounted_s": delta,
        "tolerance_s": CLOSURE_TOL_S,
        "pass": abs(delta) <= CLOSURE_TOL_S,
    }


class Cancelled(Exception):
    pass


class MemoryFailure(config.WebUIError):
    pass


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _vae_thread_record(vae_result):
    """The VAE decode-thread arm for job.json ("vae_decode_threads").

    The decode threads are the one knob that moves the PNG bytes without moving
    any other recorded number; the 2026-11-03 audit found 0/627 job.json in
    this tree naming such a knob.  Missing keys (an older vae_worker result)
    are recorded as None -- never invented, never defaulted to a guess.
    """
    vae_result = vae_result or {}
    return {"n": int(vae_result.get("threads") or 0),
            "arm": vae_result.get("thread_arm"),
            "class": vae_result.get("thread_class"),
            "latent": vae_result.get("thread_latent"),
            "rule": vae_result.get("thread_rule"),
            "why": vae_result.get("thread_why")}


def _crop_png_top_left(path, out_h, out_w):
    """Trim a decoded PNG to [0:out_h, 0:out_w] (right/bottom excess removed).

    The product rule for a padded run is the industry-standard one: generate
    the next legal frame, then keep the requested top-left H x W rectangle.
    "Bottom-right trim" names WHERE the excess is removed from; the kept box is
    [0:H, 0:W].  The 16-multiple path never calls this, so its PNG bytes and
    sha256 stay exactly as delivered.
    """
    from PIL import Image
    t0 = time.perf_counter()
    with Image.open(path) as im:
        if im.size == (int(out_w), int(out_h)):
            return 0.0, im.size
        im.load()
        im.crop((0, 0, int(out_w), int(out_h))).save(path, format="PNG")
    return time.perf_counter() - t0, (int(out_w), int(out_h))


def _rel(path):
    return config.rel_to_npu(path)


class Runner:
    # T22 prewarm: one sibling py310 process packs 120 projection images
    # (1.85 GiB, pure CPU).  It is killed if it has not finished in this long --
    # a hung packer must cost a log line, not the job.
    WPACK_TIMEOUT_S = 900.0

    def __init__(self):
        self.lock = threading.RLock()
        self.jobs = {}
        self.queue = queue.Queue()
        self.current = None
        self._stop = False
        self.worker = None            # T21 2a-b: resident py310 device worker
        self.worker_lock = threading.Lock()
        self.wpack = {}               # T22 prewarm: artifact path -> state
        self.wpack_lock = threading.Lock()
        self._load_history()
        self._wpack_boot()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    # ------------------------------------------------------- memory budget
    def _budget_or_refuse(self, params):
        """The host-memory gate at SUBMIT time (rq_1080p_8d/8e/8f).

        Eight attempts died at four different host allocations and the
        page let every one of them start.  The rule enforced here:

          * if NO mode fits -> refuse now, with the number, instead of
            spending 20 minutes and dying at step 6;
          * if only --no-cache-all fits -> say so and let the job run in
            low mode (the mode that actually completes);
          * if the caller EXPLICITLY asked for a mode that does not fit,
            refuse it when ZIMAGE_BUDGET_ENFORCE=1, and otherwise record
            the warning on the job (an operator measuring the cliff needs
            to be able to walk into it on purpose).
        """
        m = params.get("m")
        want = str(params.get("memory_mode", "auto")).lower()
        mode, snap, why = config.choose_memory_mode(want, m)
        v = config.memory_verdict(m, mode, snap)
        v["chosen"] = mode
        v["reason"] = why
        v["explicit"] = want in ("fast", "low")
        if v["ok"] or mode != "fast":
            return v
        low = config.memory_verdict(m, "low", snap)
        if not low["ok"]:
            raise config.WebUIError(
                low["message"],
                hint="the budget is the measured peak of the whole "
                     "process tree (webui/config.py MEMORY_MEASURED, "
                     "imaggers/par/membudget/peakwatch.py); close that "
                     "much or pick a smaller shape")
        if v["explicit"] and config.env_flag("ZIMAGE_BUDGET_ENFORCE", False):
            raise config.WebUIError(
                v["message"],
                hint="ZIMAGE_BUDGET_ENFORCE=1 makes the requested mode "
                     "binding; unset it or submit memory_mode=low")
        v["degrade_to"] = "low"
        v["message"] = (v["message"] + "  -> the job runs --no-cache-all")
        return v

    # ------------------------------------------------------------------ api
    def submit(self, raw_params):
        params = self._normalize(raw_params)
        budget = self._budget_or_refuse(params)
        jid = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
        job_dir = config.JOBS_DIR / jid
        job_dir.mkdir(parents=True, exist_ok=True)
        job = {
            "id": jid,
            "created": config.now_iso(),
            "created_ts": time.time(),
            # T22 pad_crop: the request, the internal run size and the final
            # output size are echoed as three separate records.  All three are
            # equal for a 16-multiple request; a padded request runs the middle
            # pair and is trimmed back to the requested pair.
            "requested": {"height": int(params["requested_height"]),
                          "width": int(params["requested_width"])},
            "padded": {"height": int(params["pad_height"]),
                       "width": int(params["pad_width"])},
            "output": {"height": int(params["output_height"]),
                       "width": int(params["output_width"])},
            "status": "queued",
            "stage": "queued",
            "stage_detail": "排队中(器件一次一个)",
            "step": 0,
            "steps": params["steps"],
            "exec_mode": params.get("exec_mode", "native"),
            "estimate": params.get("chunk_estimate"),
            "params": params,
            "timing": {},
            "images": [],
            "gate": {"mode": params["mode"], "status": "not_run",
                     "detail": "试跑 — 未过复现门" if params["mode"] == "trial"
                     else "等待双跑"},
            "memory_mode": None,
            "vae_resident": None,
            "memory_reason": "",
            # t22_loud: every fallback that survives gets one entry here (see
            # _degrade).  A consumer that reads only status can then still tell
            # a run that took the path it was asked for from one that did not.
            "degraded": [],
            "memory_snapshot": config.memory_snapshot(),
            # the budget this job was admitted under (required vs headroom)
            "memory_budget": budget,
            # which launcher env selected the device path (record only)
            "env_flags": config.launch_flags(),
            # t22_loud: the RESOLVED values, defaulted or explicit, for the
            # switches env_flags never carried (FFN image, qkv, p10, the glue
            # pair, REF_SMNB).  A None in env_flags is ambiguous; this is not.
            "env_resolved": config.resolved_flags(),
            "env_provenance": config.env_provenance(),
            "cache": None,
            "schedule": None,
            "cap": None,
            "tag": None,
            "log_tail": [],
            "log_lines": 0,
            "error": None,
            "hint": "",
            "cancel_requested": False,
            "started_ts": None,
            "finished_ts": None,
            "started": None,
            "finished": None,
            "attempts": [],
        }
        with self.lock:
            self.jobs[jid] = job
            self._save(job)
        self.queue.put(jid)
        return self.public(job)

    def get(self, jid):
        with self.lock:
            job = self.jobs.get(jid)
            return self.public(job) if job else None

    def list(self, limit=40):
        with self.lock:
            jobs = sorted(self.jobs.values(), key=lambda j: j["created_ts"],
                          reverse=True)[:limit]
            return [self.public(j) for j in jobs]

    def status(self):
        with self.lock:
            running = [j["id"] for j in self.jobs.values()
                       if j["status"] == "running"]
            queued = [j["id"] for j in self.jobs.values()
                      if j["status"] == "queued"]
            return {"queue": len(queued), "running": running,
                    "current": self.current, "jobs": len(self.jobs),
                    "max_batch": config.MAX_BATCH, "max_steps": config.MAX_STEPS,
                    "wpack": self.wpack_status()}

    def cancel(self, jid):
        with self.lock:
            job = self.jobs.get(jid)
            if not job:
                return False, "no such job"
            if job["status"] in ("done", "error", "cancelled"):
                return False, f"job already {job['status']}"
            job["cancel_requested"] = True
            proc = job.get("_proc")
        if job["status"] == "queued":
            job["status"] = "cancelled"
            job["finished"] = config.now_iso()
            job["finished_ts"] = time.time()
            self.append_log(job, "[webui] cancelled while queued")
            self._save(job)
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
        return True, "cancellation requested"

    def retry(self, jid):
        with self.lock:
            job = self.jobs.get(jid)
            if not job:
                raise config.WebUIError("no such job")
            params = copy.deepcopy(job["params"])
            # T22 pad_crop: params.height/width are the padded RUN size.  A
            # retry must submit the original REQUEST again, otherwise a
            # 130x130 job would silently become a 144x144 one.
            if params.get("requested_height") is not None:
                params["height"] = params["requested_height"]
            if params.get("requested_width") is not None:
                params["width"] = params["requested_width"]
            params["mode"] = "trial"
        return self.submit(params)

    # ------------------------------------------------------------------ queue
    def _loop(self):
        while not self._stop:
            try:
                jid = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            with self.lock:
                job = self.jobs.get(jid)
                self.current = jid
            if not job:
                continue
            if job.get("cancel_requested"):
                continue
            try:
                self._run(job)
            except Exception:
                # _run handles its own errors; this is a last resort.
                self.append_log(job, "[webui] WORKER CRASH\n" + traceback.format_exc())
                job["status"] = "error"
                job["error"] = {"message": "worker crash", "raw": traceback.format_exc()[-4000:], "hint": ""}
                job["finished_ts"] = time.time()
                job["finished"] = config.now_iso()
                self._save(job)
            finally:
                with self.lock:
                    self.current = None
                self.queue.task_done()

    def _run(self, job):
        t_start = time.perf_counter()
        job["status"] = "running"
        job["started_ts"] = time.time()
        job["started"] = config.now_iso()
        job["timing"]["queue_s"] = max(0.0, job["started_ts"] - job["created_ts"])
        self.append_log(job, "=" * 78)
        self.append_log(job, f"[webui] job {job['id']} started {job['started']}")
        self.append_log(job, f"[webui] prompt={job['params']['prompt']!r}")
        self.append_log(job, f"[webui] steps={job['params']['steps']} seed={job['params']['seed']} "
                              f"batch={job['params']['batch']} vae={job['params']['vae']} "
                              f"mode={job['params']['mode']}")
        self.append_log(job, "[webui] shape requested %sx%s -> run %sx%s -> "
                              "output %sx%s (pad_applied=%s, crop=%s, M=%s)"
                       % (job['params']['requested_height'],
                          job['params']['requested_width'],
                          job['params']['height'], job['params']['width'],
                          job['params']['output_height'],
                          job['params']['output_width'],
                          job['params']['pad_applied'],
                          job['params']['crop_applied'], job['params']['m']))
        self._save(job)
        try:
            self._execute(job, t_start)
            if job.get("status") == "running":
                job["status"] = "done"
            if job.get("degraded"):
                job["stage_detail"] = ("%s  [DEGRADED %d]"
                                       % (job.get("stage_detail") or "",
                                          len(job["degraded"])))
        except Cancelled:
            job["status"] = "cancelled"
            job["stage"] = "cancelled"
            job["stage_detail"] = "已取消"
            self.append_log(job, "[webui] CANCELLED")
        except config.WebUIError as exc:
            job["status"] = "error"
            job["stage"] = "error"
            job["stage_detail"] = str(exc)
            job["error"] = {"message": str(exc), "raw": exc.raw[-8000:],
                            "hint": exc.hint}
            self.append_log(job, f"[webui] ERROR: {exc}")
            if exc.raw:
                self.append_log(job, exc.raw[-8000:])
        except Exception as exc:
            job["status"] = "error"
            job["stage"] = "error"
            job["stage_detail"] = f"{type(exc).__name__}: {exc}"
            job["error"] = {"message": f"{type(exc).__name__}: {exc}",
                            "raw": traceback.format_exc()[-8000:], "hint": ""}
            self.append_log(job, "[webui] UNEXPECTED\n" + traceback.format_exc())
        finally:
            t_now = time.perf_counter()
            job["finished_ts"] = time.time()
            job["finished"] = config.now_iso()
            end = job.get("_passes_end") or t_now
            job["timing"]["post_s"] = max(0.0, t_now - end)
            queue_s = float(job["timing"].get("queue_s") or 0.0)
            exec_s = t_now - t_start
            total_s = queue_s + exec_s
            job["timing"]["exec_s"] = exec_s
            job["timing"]["total_s"] = total_s
            job["timing"]["closure"] = timing_closure(job["timing"], total_s)
            job["_timing_sealed"] = True
            job["elapsed_s"] = job["finished_ts"] - job["created_ts"]
            job["_proc"] = None
            self._save(job)
            c = job["timing"]["closure"]
            self.append_log(job, f"[webui] job {job['id']} {job['status']} "
                                  f"in {total_s:.1f} s; timing closure "
                                  f"{'PASS' if c['pass'] else 'FAIL'} "
                                  f"(sum {c['sum_s']:.2f} + unaccounted "
                                  f"{c['unaccounted_s']:+.2f} = total {c['total_s']:.2f})")

    def _execute(self, job, t_start):
        params = job["params"]
        ck = lambda: job.get("cancel_requested", False)
        if ck():
            raise Cancelled()
        job["stage"] = "prepare"
        job["stage_detail"] = f"校验/准备 {params['shape']} 构建缓存 (M={params['m']})"
        self.append_log(job, f"[webui] shape {params['shape']} (M={params['m']}): "
                              f"verify build cache")
        if params.get("exec_mode") == "chunked":
            est = params.get("chunk_estimate") or {}
            self.append_log(job, "[webui] exec_mode=chunked (no dedicated kernel): "
                                  "delivered M=96 projection/FFN/o images in 96-row "
                                  "blocks + host f32 attention; estimate %.1f s/step, "
                                  "8 steps %.2f h, %d dispatches"
                            % (float(est.get("step_s") or 0.0),
                               float(est.get("h_8step") or 0.0),
                               int(est.get("dispatches_8step") or 0)))
        cache_info = cache.ensure_shape(params["height"], params["width"],
                                        lambda s: self.append_log(job, s))
        job["cache"] = {k: cache_info[k] for k in
                        ("shape", "height", "width", "m", "dir", "cached",
                         "created", "seconds")}
        job["timing"]["prepare_s"] = time.perf_counter() - t_start
        self._save(job)

        mode, snap, why = config.choose_memory_mode(params.get("memory_mode", "auto"),
                                                    params.get("m"))
        job["memory_mode"] = mode
        job["memory_reason"] = why
        job["memory_snapshot"] = snap
        self.append_log(job, f"[webui] memory mode: {mode} ({why}); "
                              f"avail {snap['available_gb']} GB, commit {snap['commit_gb']} GB")
        # Re-checked HERE and not only at submit: the box can change between
        # the two, and this is the last moment before the work that dies.
        budget = config.memory_verdict(params.get("m"), mode, snap)
        job["memory_budget"] = budget
        self.append_log(job, "[webui] memory budget: " + budget["message"])
        if not budget["ok"] and mode == "fast":
            low = config.memory_verdict(params.get("m"), "low", snap)
            if low["ok"]:
                mode = "low"
                job["memory_mode"] = "low"
                job["memory_reason"] = budget["message"]
                self._degrade(job, "memory-budget", job["memory_reason"])
                self.append_log(job, "[webui] falling back to "
                                      "--no-cache-all: " + budget["message"])
            else:
                raise config.WebUIError(
                    low["message"],
                    hint="webui/config.py MEMORY_MEASURED; this is the "
                         "check that should have refused the submit")
        self._save(job)

        # T22 prewarm: start the pure-CPU pack of the 120 projection images now,
        # so it overlaps this job's text encode instead of sitting inside the
        # device stage.  It is a sibling py310 process with no device context
        # (bridge.py --pack-prewarm), so it cannot collide with the VAE chain;
        # the device stage joins it and the wait is booked to startup_s.
        if self._wpack_usable(params, mode):
            self._wpack_start(job, cache_info, params, reason="job")

        # T21 gatefix2: start the clock before the first pass.  When a fast
        # attempt dies on MemoryError, that attempt's wall is real time the job
        # spent; without booking it the fallback job's closure goes positive by
        # exactly that attempt (tens of seconds -> FAIL).  The except branch
        # measures it and carries it into the low-memory pass's retry_s.
        t_pass0 = time.perf_counter()
        try:
            pass0 = self._execute_pass(job, 0, mode, cache_info, params["vae"])
        except MemoryFailure as exc:
            if mode != "fast":
                raise
            fallback_s = max(0.0, time.perf_counter() - t_pass0)
            self.append_log(job, "[webui] fast cache-all hit a memory wall; "
                                  "falling back to --no-cache-all in a fresh process")
            self.append_log(job, (exc.raw or "")[-4000:])
            self.append_log(job, f"[webui] failed fast pass cost {fallback_s:.2f} s; "
                                  f"booked to retry_s so timing closure stays whole")
            mode = "low"
            job["memory_mode"] = "low"
            job["memory_reason"] = "fast cache-all failed with MemoryError; fallback --no-cache-all"
            self._degrade(
                job, "memory-fallback",
                "the fast cache-all pass hit a MemoryError after %.2f s; this "
                "job ran --no-cache-all, which is ~2.4x the loop AND a different "
                "latent baseline (229.26 s / 12eb3560 vs 94 s / 2abf9133 at "
                "512^2)" % (fallback_s,))
            self._save(job)
            pass0 = self._execute_pass(job, 0, mode, cache_info, params["vae"],
                                       tag_suffix="low", carry_retry_s=fallback_s)

        job["images"] = pass0["images"]
        job["timing"].update(pass0["timing"])
        if job.get("estimate") and pass0["timing"].get("loop_s"):
            est = job["estimate"]
            loop_s = float(pass0["timing"]["loop_s"])
            est["measured_loop_s"] = round(loop_s, 2)
            est["measured_step_s"] = round(loop_s / max(1, params["steps"]), 2)
            if est.get("s_8step"):
                est["measured_vs_estimate"] = round(
                    loop_s / (float(est["s_8step"]) * params["steps"] / 8.0), 3)
        job["pe"] = {"sha256": pass0["pe_sha"], "tokens": pass0["tokens"],
                    "backend": pass0.get("te_backend")}
        job["te_backend"] = pass0.get("te_backend")
        job["schedule"] = pass0["schedule"]
        job["cap"] = pass0.get("cap")
        job["latents_sha256"] = pass0["latents_sha"]
        job["crop"] = pass0.get("crop")
        job["dispatches"] = pass0["calls"].get("dispatches")
        job["retries"] = pass0["calls"].get("retries")
        job["teardown_rc"] = pass0.get("teardown_rc")
        if params["mode"] == "trial":
            job["gate"] = {"mode": "trial", "status": "not_run",
                           "detail": "试跑 — 单次运行,未过复现门"}
            for img in job["images"]:
                img["gate"] = "trial"
            job["stage"] = "done"
            job["stage_detail"] = "完成(试跑)"
            job["_passes_end"] = time.perf_counter()
            self._save(job)
            return

        # ---- gate: second full pass, compare latents byte-for-byte
        job["gate"] = {"mode": "gate", "status": "running",
                       "detail": "复现门:第二次运行中"}
        job["stage"] = "gate"
        job["stage_detail"] = "复现门:第二次运行(同参数)"
        self._save(job)
        self.append_log(job, "[gate] second pass: same prompt/steps/seed/batch; "
                              "compare latents bytes and pe sha256")
        # T21 gatefix2: take the second pass's start time BEFORE calling it.
        # If pass 1 raises, _execute_pass never reaches its own seal, so the time
        # it already burned would live only in total_s; without this the failed
        # job's closure can never pass.  The except branch books the whole
        # elapsed span as gate_second_s -- exactly once, because pass 1 never
        # merges its per-stage numbers into job["timing"] (pass 0 alone does).
        t_gate0 = time.perf_counter()
        try:
            pass1 = self._execute_pass(job, 1, mode, cache_info, vae=False,
                                       tag_suffix="g",
                                       te_backend=pass0.get("te_backend"))
        except config.WebUIError as exc:
            second_elapsed = max(0.0, time.perf_counter() - t_gate0)
            job["timing"]["gate_second_s"] = second_elapsed
            job["gate"] = {"mode": "gate", "status": "FAIL",
                           "detail": f"second pass failed: {exc}",
                           "pe_sha_first": pass0["pe_sha"],
                           "latents_sha_first": pass0["latents_sha"],
                           "timing": {
                               "first_total_s": pass0["timing"].get("pass_s"),
                               "second_elapsed_s": second_elapsed},
                           "error": (exc.raw or "")[-4000:]}
            for img in job["images"]:
                img["gate"] = "FAIL"
            job["stage"] = "done"
            job["stage_detail"] = "完成(复现门 FAIL — 第二次运行失败)"
            self.append_log(job, f"[gate] FAIL after {second_elapsed:.2f} s of the "
                                  f"second pass: {exc} (wall booked to gate_second_s)")
            job["_passes_end"] = time.perf_counter()
            self._save(job)
            return
        lat_eq = pass1["latents_sha"] == pass0["latents_sha"]
        pe_eq = pass1["pe_sha"] == pass0["pe_sha"]
        gate = {
            "mode": "gate",
            "status": "PASS" if lat_eq else "FAIL",
            "latents_equal": bool(lat_eq),
            "pe_equal": bool(pe_eq),
            "latents_sha_first": pass0["latents_sha"],
            "latents_sha_second": pass1["latents_sha"],
            "pe_sha_first": pass0["pe_sha"],
            "pe_sha_second": pass1["pe_sha"],
            "te_backend": pass0.get("te_backend"),
            "detail": ("latents 逐字节相同" if lat_eq else
                       "latents 不同 — 同参数两次运行不可复现"),
            "timing": {"first_total_s": pass0["timing"].get("pass_s"),
                       "second_total_s": pass1["timing"].get("pass_s")},
        }
        job["gate"] = gate
        for img in job["images"]:
            img["gate"] = gate["status"]
        job["timing"]["gate_second_s"] = pass1["timing"].get("pass_s", 0.0)
        job["timing"]["total_s"] = time.perf_counter() - t_start
        job["stage"] = "done"
        job["stage_detail"] = f"完成(复现门 {gate['status']})"
        self.append_log(job, f"[gate] {gate['status']}: latents {gate['latents_sha_first']} "
                              f"vs {gate['latents_sha_second']}; pe {gate['pe_sha_first']} "
                              f"vs {gate['pe_sha_second']}")
        job["_passes_end"] = time.perf_counter()
        self._save(job)

    # ------------------------------------------------------- resident worker
    def _get_worker(self, cache_info, params, log_path, wpack_file="",
                    reserve_gb=0.0, keep_payload="auto", keep_refiner="auto"):
        """The live device worker, started lazily and pre-warmed once.

        wpack_file / reserve_gb only apply when a NEW worker is started: the
        prewarm artifact is mapped at warm time (dev_worker --wpack-file, the
        same seam the delivered bridge uses) and the reserve is part of the
        residency decision taken there.
        """
        with self.worker_lock:
            w = self.worker
            if w is not None and w.alive():
                return w
            if w is not None:
                w.kill()
            w = dev_worker.Worker(
                config.PY310, config.WEBUI / "dev_worker.py",
                cache_info["dir"], params["height"], params["width"],
                config.REFINER_CACHE, log_path=log_path,
                env=config.subprocess_env("device"), cwd=config.NPU,
                wpack_file=wpack_file, wpack_verify=bool(config.WPACK_VERIFY),
                reserve_gb=reserve_gb,
                keep_payload=keep_payload, keep_refiner=keep_refiner)
            self.worker = w
            return w

    def _drop_worker(self, why=""):
        with self.worker_lock:
            w, self.worker = self.worker, None
        if w is not None:
            try:
                w.kill()
            except Exception:                                 # noqa: BLE001
                pass
        return why

    # --------------------------------------------------------- degraded
    def _degrade(self, job, kind, detail):
        """Record a fallback that still produces a 'done' job.

        The product has many fallbacks and each one is individually correct; the
        defect they shared was that none of them changed anything a consumer
        reads.  One entry here, one log line, and a suffix on stage_detail (which
        the page already renders) is the whole contract.
        """
        with self.lock:
            job.setdefault("degraded", []).append({"kind": kind,
                                                   "detail": detail})
        self.append_log(job, "[webui] DEGRADED %s: %s" % (kind, detail))

# ------------------------------------------------- T22 wpack prewarm
    @staticmethod
    def _wpack_on_disk(path):
        """Cheap truth check: the artifact and its manifest are still there.

        The state dict says what happened in THIS server process; a file can be
        deleted behind it (measured 2026-10-20: a job deleted the artifact while
        the runner still remembered ok=True, handed the bridge a missing file,
        and the bridge fell back to the delivered in-process pack -- correct,
        but the runner had booked it as a reuse).  Two stats, every hand-off.
        """
        p = pathlib.Path(path)
        try:
            st = p.stat()
            meta = json.loads(pathlib.Path(str(p) + ".json").read_text(
                encoding="utf-8"))
        except (OSError, ValueError):
            return False
        try:
            if int(meta.get("file_size") or 0) != int(st.st_size):
                return False
            if int(meta.get("bytes") or 0) != int(st.st_size):
                return False
        except (TypeError, ValueError):
            return False
        return True

    def _wpack_usable(self, params, mode):
        """Can this job consume the prewarmed projection images?

        T22 workervae: the resident device worker now MAPS an existing
        artifact too (dev_worker --wpack-file, the same seam the delivered
        bridge uses -- see _get_worker), but it must never START a packer: the
        worker takes the file at warm time, and a packer rewriting it under a
        live mapping is a sharing violation.  So a job that will run in the
        worker is not a packer consumer (usable = False), while the device
        stage still hands it the artifact if one is on disk.
        """
        if not config.WPACK_PREWARM:
            return False
        if (config.DEV_WORKER and mode == "fast"
                and not params.get("bo_resident")
                and shapes.m_from_hw(params["height"], params["width"]) == 96):
            return False
        return True

    def _wpack_cmd(self, cache_dir, params, out):
        cmd = [str(config.PY310), str(config.BRIDGE), "--pack-prewarm",
               "--pack-out", str(out), "--cache-dir", str(cache_dir),
               "--height", str(int(params["height"])),
               "--width", str(int(params["width"])),
               "--layers", str(int(config.WPACK_LAYERS))]
        if config.WPACK_VERIFY:
            cmd.append("--pack-verify")
        return cmd

    def _wpack_start(self, job, cache_dir, params, reason="job"):
        """Start (or reuse) the sibling packer.  Never blocks the caller."""
        out = str(config.WPACK_FILE)
        with self.wpack_lock:
            st = self.wpack.get(out)
            if st is not None and st.get("ok") is True:
                if self._wpack_on_disk(out):
                    return st
                st["ok"] = False
                st["err"] = "the artifact disappeared after it was built"
            th = st.get("thread") if st else None
            if th is not None and th.is_alive():
                if job is not None:
                    st.setdefault("jobs", []).append(job["id"])
                return st
            try:
                pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                if job is not None:
                    self.append_log(job, "[webui] T22 prewarm: cannot create %s "
                                          "(%s); this job packs in-process as "
                                          "delivered"
                                    % (pathlib.Path(out).parent, exc))
                return None
            cmd = self._wpack_cmd(cache_dir, params, out)
            st = {"path": out, "ok": None, "thread": None, "seconds": 0.0,
                  "pack_s": 0.0, "sha256": "", "reused": False, "err": "",
                  "reason": reason, "jobs": [job["id"]] if job is not None else [],
                  "started": time.time()}
            jid = job["id"] if job is not None else ""
            th = threading.Thread(target=self._wpack_body, args=(st, cmd, jid),
                                  name="t22-wpack", daemon=True)
            st["thread"] = th
            self.wpack[out] = st
        if job is not None:
            self.append_log(job, "[webui] T22 prewarm: packing the 120 projection "
                                  "images (1.85 GiB) in a sibling py310 process -- "
                                  "pure CPU, no device -- while this job text "
                                  "encodes; the device stage waits for it if it is "
                                  "not done yet")
        th.start()
        return st

    def _wpack_body(self, st, cmd, jid):
        log = (config.JOBS_DIR / jid / "wpack_prewarm.log") if jid else \
            (config.WPACK_DIR / "prewarm_boot.log")
        try:
            res = stream_process(
                cmd, env=config.subprocess_env("device"), cwd=config.NPU,
                log_path=log,
                on_line=(lambda s: self._wpack_line(jid, s)) if jid else None,
                timeout=self.WPACK_TIMEOUT_S)
            st["seconds"] = float(res.get("seconds") or 0.0)
            for line in res.get("lines") or []:
                m = EVENT_RE.match(line)
                if not m:
                    continue
                try:
                    ev = json.loads(m.group(1))
                except Exception:
                    continue
                if ev.get("event") == "prewarm_pack_done":
                    st["pack_s"] = float(ev.get("pack_s") or 0.0)
                    st["sha256"] = str(ev.get("sha256") or "")
                    st["reused"] = bool(ev.get("reused"))
            ok = (int(res.get("rc") or 0) == 0
                  and pathlib.Path(st["path"] + ".json").is_file())
            st["ok"] = bool(ok)
            if not ok:
                st["err"] = "\n".join((res.get("lines") or [])[-8:])
        except Exception as exc:                              # noqa: BLE001
            st["ok"] = False
            st["err"] = "%s: %s" % (type(exc).__name__, exc)
        if not st.get("ok"):
            for jid2 in list(st.get("jobs") or []):
                job = self.jobs.get(jid2)
                if job is not None:
                    self.append_log(job, "[webui] T22 prewarm FAILED (%s); this job "
                                          "packs in-process exactly as delivered"
                                    % ((st.get("err") or "")[-400:] or "no log"))

    def _wpack_line(self, jid, line):
        job = self.jobs.get(jid) if jid else None
        if job is not None:
            self.append_log(job, line)

    def _wpack_join(self, job, timing):
        """Wait for the packer and return the file to hand to the bridge.

        The wait is time THIS JOB really spends before the device stage, so it
        is booked to startup_s (the same rule T21 2a2b2 settled for a cold
        prewarm).  The packer's own wall is NOT booked anywhere: it overlaps the
        text encode / VAE and booking it too would count it twice.
        """
        out = str(config.WPACK_FILE)
        with self.wpack_lock:
            st = self.wpack.get(out)
        if st is None:
            return None
        th = st.get("thread")
        if th is not None and th.is_alive():
            t0 = time.perf_counter()
            th.join()
            timing["prewarm_wait_s"] = timing.get("prewarm_wait_s", 0.0) + (
                time.perf_counter() - t0)
        if st.get("ok") is not True:
            return None
        if not self._wpack_on_disk(st["path"]):
            st["ok"] = False
            st["err"] = "the artifact disappeared after it was built"
            job and self.append_log(job, "[webui] T22 prewarm: %s is gone; this "
                                        "job packs in-process as delivered"
                                   % pathlib.Path(st["path"]).name)
            return None
        timing["prewarm_src"] = ("reused" if st.get("reused")
                                 else str(st.get("reason") or "job"))
        # T22 closurefix: the packer's OWN wall (its json's pack_s) is not job
        # time -- it runs beside the TE/VAE.  It moves to prewarm_pack_wall_s
        # (informational, never booked); the bucket prewarm_pack_s is the
        # blocked-on-it time measured here and is filled at the seal.
        timing["prewarm_pack_wall_s"] = float(st.get("pack_s") or 0.0)
        timing["prewarm_proc_s"] = float(st.get("seconds") or 0.0)
        return st["path"]

    def _wpack_boot(self):
        """ZIMAGE_WPACK_PREWARM=auto|boot: have the artifact before job 1."""
        if not config.WPACK_PREWARM_BOOT or config.DEV_WORKER:
            return
        h, w = config.WPACK_PREWARM_BOOT_SHAPE
        cache_dir = config.CACHE_DIR / ("%dx%d" % (int(h), int(w)))
        if not (cache_dir / "manifest.json").is_file():
            return
        try:
            self._wpack_start(None, cache_dir,
                              {"height": int(h), "width": int(w)},
                              reason="boot")
        except Exception:                                     # noqa: BLE001
            pass

    def wpack_status(self):
        out = str(config.WPACK_FILE)
        with self.wpack_lock:
            st = dict(self.wpack.get(out) or {})
        st.pop("thread", None)
        p = pathlib.Path(out)
        return {"enabled": bool(config.WPACK_PREWARM),
                "boot": bool(config.WPACK_PREWARM_BOOT),
                "verify": bool(config.WPACK_VERIFY),
                "file": config.rel_to_npu(p),
                "present": p.is_file(),
                "bytes": p.stat().st_size if p.is_file() else 0,
                "state": st}

    # ------------------------------------------------------------- T22 te cache
    @staticmethod
    def _te_backend_target(requested):
        """The concrete backend textenc.run would use for "requested".

        textenc.run("auto") tries gpu first and only skips it after a GPU
        failure was seen in this server (its _GPU_OK is False).  The cache key
        needs the SAME resolution, because gpu and cpu produce different pe
        bytes -- the job history has 90514d08... (gpu) and e6642367... (cpu) for
        the SAME prompt, so a key on "auto" would mix two different results.
        """
        b = (requested or "auto").strip().lower()
        if b in ("gpu", "cpu"):
            return b
        return "cpu" if getattr(textenc, "_GPU_OK", None) is False else "gpu"

    def _te_model_policy(self, snap):
        """(resident, why) for the TE model on THIS job.

        T22 te cache: the result cache removes the load for a repeated prompt;
        this policy is what a MISS does.  "auto" keeps the 8.05 GB model loaded
        ONLY when the device loop can still take the fast path with it: the
        memory-mode chooser needs commit >= 9.0 GB (config), the encoder holds
        ~7.0 GB, so the bar is 16.0 GB of commit headroom.  On this host
        (9.0-9.3 GB) that is false, which is the measured 2a-a net loss avoided
        by construction; on a host with more commit the encoder stays resident
        and later MISSES drop to the 0.6 s served cost.
        """
        mode = TE_MODEL_RESIDENT
        if mode in ("0", "false", "no", "off"):
            return False, "off"
        if mode in ("1", "true", "yes", "on"):
            return True, "on"
        commit = float(snap.get("commit_gb") or 0.0)
        need = config.MEMORY_FAST_THRESHOLD_GB + TE_MODEL_COMMIT_GB
        if commit >= need:
            return True, ("auto: commit %.1f GB >= %.1f GB (resident encoder "
                          "leaves the fast device path intact)" % (commit, need))
        return False, ("auto: commit %.1f GB < %.1f GB (a resident encoder would "
                       "flip the device loop to --no-cache-all; T21 2a-a)"
                       % (commit, need))

    # --------------------------------------------------------------- one pass
    def _execute_pass(self, job, pass_index, mode, cache_info, vae,
                      tag_suffix="", te_backend=None, carry_retry_s=0.0):
        params = job["params"]
        ck = lambda: job.get("cancel_requested", False)
        t_pass = time.perf_counter()
        timing = {}
        jid_short = job["id"].replace("-", "")[-12:]
        tag = f"w{jid_short}p{pass_index}{tag_suffix}"
        job["tag"] = tag
        seeds = [params["seed"] + b for b in range(params["batch"])]
        prompt = params["prompt"]

        # ---- prompt encoding (real text encoder, hidden_states[-2])
        job["stage"] = "textenc"
        job["stage_detail"] = "文本编码 (llama.cpp Vulkan, cap_tokens=32)"
        self._save(job)
        te_log = config.JOBS_DIR / job["id"] / f"te_p{pass_index}{tag_suffix}.log"
        pe_path = config.JOBS_DIR / job["id"] / f"pe_p{pass_index}{tag_suffix}.npz"
        # T22 te cache: the key is the EXACT chat-templated string the encoder
        # reads + the concrete backend + the encoder identity (gguf/worker/
        # driver stamps).  H/W/steps/seed/batch are deliberately NOT inputs --
        # pe cannot depend on them (job-history proof in webui/cache.py).
        t_te0 = time.perf_counter()
        te_text = textenc.render_chat(prompt)
        backend_target = self._te_backend_target(te_backend)
        # Model-level residency policy: a HIT never touches the encoder at all;
        # a MISS runs one-shot unless the policy proves the model can stay
        # resident without flipping the next device loop to --no-cache-all.
        te_resident, te_resident_why = self._te_model_policy(
            config.memory_snapshot())
        if not te_resident and textenc.status().get("resident"):
            self.append_log(job, "[webui] te model: dropping the resident "
                                  "encoder (%s)" % te_resident_why)
            textenc.shutdown()
        te_entry = None
        try:
            te_entry = cache.te_lookup(te_text, backend_target)
        except Exception as exc:                              # noqa: BLE001
            self.append_log(job, "[webui] te-cache lookup failed (%s: %s); "
                                  "encoding as delivered"
                            % (type(exc).__name__, exc))
        if te_entry is not None:
            cache.te_materialize(te_entry, pe_path)
            pathlib.Path(pe_path).with_suffix(".prompt.txt").write_text(
                te_text, encoding="utf-8", newline="\n")
            te_backend = te_entry["meta"].get("backend") or backend_target
            sec = time.perf_counter() - t_te0
            te = {"seconds": sec, "resident": False, "boot_s": 0.0,
                  "load_s": 0.0, "decode_s": 0.0, "reused": False,
                  "cache": "hit", "cache_source": te_entry["source"],
                  "cache_key": te_entry["key"], "backend": te_backend,
                  "tokens": int(te_entry["meta"].get("tokens") or 0),
                  "pe_sha256": te_entry.get("pe_sha256")}
            with open(te_log, "a", encoding="utf-8", errors="replace") as f:
                f.write("[te-cache] HIT key=%s source=%s backend=%s tokens=%s "
                        "pe=%s in %.3f s (no model load, no decode)\n"
                        % (te["cache_key"][:12], te["cache_source"], te_backend,
                           te["tokens"], (te["pe_sha256"] or "")[:16], sec))
            self.append_log(job, "[webui] text enc: cache HIT %s (%s, %s, %s "
                                  "tokens) in %.3f s"
                            % (te["cache_key"][:12], te["cache_source"],
                               te_backend, te["tokens"], sec))
            job["te_cache"] = {"hit": True, "key": te["cache_key"],
                               "source": te["cache_source"],
                               "backend": te_backend, "tokens": te["tokens"],
                               "seconds": round(sec, 4),
                               "pe_sha256": te["pe_sha256"]}
        else:
            te = textenc.run(prompt, pe_path, te_log,
                             on_line=lambda s: self.append_log(job, s),
                             timeout=300.0, cancel_check=ck,
                             backend=(te_backend or "auto"),
                             resident=te_resident)
            te_backend = te.get("backend") or (te_backend or "auto")
            # The bucket is the FULL wall of this step, cache lookup included.
            te["seconds"] = time.perf_counter() - t_te0
            te["cache"] = "miss"
            self.append_log(job, "[webui] text enc: %s, resident=%s (policy: "
                                  "%s), boot %s s, load %s s, decode %s s"
                            % (te_backend, te.get("resident"), te_resident_why,
                               te.get("boot_s"), te.get("load_s"),
                               te.get("decode_s")))
        # T21 2a-a: textenc_s is still the FULL wall of this step (a first call
        # includes the resident worker's model load), so no second is moved out
        # of the buckets.  The new keys are informational only, like device_s.
        timing["textenc_s"] = te["seconds"]
        timing["textenc_load_s"] = te.get("load_s") or 0.0
        timing["textenc_decode_s"] = te.get("decode_s") or 0.0
        timing["textenc_boot_s"] = te.get("boot_s") or 0.0
        timing["textenc_reused"] = bool(te.get("reused"))
        timing["textenc_cache"] = te.get("cache") or "miss"
        timing["textenc_cache_source"] = te.get("cache_source") or ""
        timing["textenc_cache_key"] = te.get("cache_key") or ""
        job["te_resident"] = bool(te.get("resident"))
        pe = textenc.load_pe(pe_path)
        tokens = int(pe.shape[1])
        if tokens > config.CAP_TOKENS:
            raise config.WebUIError(
                f"prompt is {tokens} tokens (chat template included); the M=96 "
                f"image has cap_tokens={config.CAP_TOKENS}",
                hint=("Shorten the prompt. This is a fixed-shape M=96 image; "
                      "a longer caption needs a rebuilt image with a larger "
                      "cap_tokens, not a UI setting."))
        pe_sha = hashlib.sha256(np.ascontiguousarray(pe, np.float32).tobytes()).hexdigest()
        if te_entry is None:
            # Store only a pe this job is allowed to use (CAP_TOKENS checked
            # above), so a cached entry can never fail the cap on a later job.
            try:
                stored = cache.te_store(te_text, te_backend, pe_path)
                if stored is not None:
                    job["te_cache"] = {"hit": False, "key": stored["key"],
                                       "source": "store",
                                       "backend": te_backend,
                                       "tokens": stored["tokens"],
                                       "seconds": round(te["seconds"], 4),
                                       "pe_sha256": stored["pe_sha256"]}
                    self.append_log(job, "[webui] te-cache: stored %s (%d B, "
                                          "%d tokens)"
                                    % (stored["key"][:12], stored["bytes"],
                                       stored["tokens"]))
            except Exception as exc:                          # noqa: BLE001
                self.append_log(job, "[webui] te-cache store failed (%s: %s); "
                                      "the job continues"
                                % (type(exc).__name__, exc))
        st = cache.te_stats()
        self.append_log(job, "[webui] te-cache: %d/%d entries, %.1f/%.0f MiB, "
                              "hits %d, misses %d, evicted %d"
                        % (st["entries"], st["max_entries"],
                           st["bytes"] / 1048576.0,
                           st["max_bytes"] / 1048576.0,
                           st["hits"], st["misses"], st["evicted"]))
        job["pe"] = {"tokens": tokens, "sha256": pe_sha,
                    "backend": te_backend}
        job["stage_detail"] = f"文本编码完成 ({tokens} tokens, pe {pe_sha[:12]})"
        self.append_log(job, f"[webui] pe {tokens} tokens sha256={pe_sha}")
        self._save(job)

        # ---- schedule
        t_sched = time.perf_counter()
        sched = schedule.get_schedule(params["steps"])
        sched_path = config.JOBS_DIR / job["id"] / "schedule.npz"
        # T22 te cache: the schedule artifact is byte-stable.  numpy's savez
        # stamps a LOCAL TIME into the zip member, so two jobs writing the same
        # arrays produced different files; one copy lives in
        # webui/cache/schedules/steps_N.npz and every job copies its EXACT
        # bytes, which is what makes job 1 and job 2 bit-identical.
        sched_bytes = None
        try:
            sched_bytes = cache.schedule_bytes(params["steps"], sched)
        except Exception as exc:                              # noqa: BLE001
            self.append_log(job, "[webui] schedule cache failed (%s: %s); "
                                  "writing the arrays directly"
                            % (type(exc).__name__, exc))
        if sched_bytes:
            tmp_sched = sched_path.with_name(sched_path.name + ".tmp")
            tmp_sched.write_bytes(sched_bytes)
            tmp_sched.replace(sched_path)
        else:
            np.savez(sched_path, timesteps=sched["timesteps"],
                     sigmas=sched["sigmas"])
        timing["schedule_s"] = time.perf_counter() - t_sched
        timing["schedule_cache"] = "hit" if sched_bytes else "miss"
        job["schedule"] = {"source": sched["source"],
                           "out_of_distribution": sched["out_of_distribution"],
                           "steps": sched["steps"]}
        self.append_log(job, f"[webui] schedule {sched['source']} "
                              f"ood={sched['out_of_distribution']}")

        # ---- device loop (fresh py310 process; retry once on a cold dispatch)
        attempts = []
        # T21 gatefix2: carry_retry_s is the wall of a failed fast attempt that
        # is being redone in low mode.  retry_s is the only retry bucket, so
        # the failed attempt lands there exactly once instead of staying only in
        # total_s; the failed pass itself never merges timing (it never reaches
        # the seal at the bottom of this method).
        timing["retry_s"] = float(carry_retry_s)
        if carry_retry_s:
            timing["fallback_s"] = float(carry_retry_s)

        def drop_worker_timed(why=""):
            """Stop the resident worker and book the teardown to its bucket.

            T22 closurefix: killing a live worker (4 device contexts + ~4 GB
            resident) is time THIS job spends.  Measured 0.67 s (21:13) and
            ~1.0 s (16:06); unbooked it is exactly what pushed the 16:06 job's
            closure to +1.12 s FAIL.
            """
            t0 = time.perf_counter()
            self._drop_worker(why)
            timing["worker_drop_s"] = timing.get("worker_drop_s", 0.0) + (
                time.perf_counter() - t0)
        timing["retry_attempts"] = 0
        timing["device_wall_s"] = 0.0
        # T21 2a-b: the resident worker serves the delivered 128x128 fast path
        # only.  Low memory, other shapes and --bo-resident keep the delivered
        # one-process bridge, and a dead worker falls back to it mid-job.
        use_worker = bool(config.DEV_WORKER and mode == "fast"
                          and not params.get("bo_resident")
                          and shapes.m_from_hw(params["height"],
                                               params["width"]) == 96)
        if not use_worker and self.worker is not None:
            self.append_log(job, "[webui] stopping the resident device worker "
                                  "(this job does not use it)")
            drop_worker_timed()
        for attempt in range(4):
            t_attempt0 = time.perf_counter()
            atag = tag + ("" if attempt == 0 else f"r{attempt}")
            meta_path = config.JOBS_DIR / job["id"] / f"bridge_{atag}.json"
            cmd = [config.PY310, config.BRIDGE,
                   "--tag", atag, "--pe", str(pe_path),
                   "--schedule", str(sched_path),
                   "--cache-dir", cache_info["dir"],
                   "--steps", str(params["steps"]),
                   "--seed", str(params["seed"]),
                   "--batch", str(params["batch"]),
                   "--height", str(params["height"]),
                   "--width", str(params["width"]),
                   "--refiner-cache", str(config.REFINER_CACHE),
                   "--meta", str(meta_path)]
            if mode == "low":
                cmd.append("--no-cache-all")
            if params.get("bo_resident") and mode == "fast":
                cmd.append("--bo-resident")
            # T22 qkv: ALWAYS explicit (the service default is ON), so an
            # unchecked page box really means the delivered path.
            cmd.append("--qkv" if params.get("qkv") else "--no-qkv")
            # T22 p10: same rule -- always explicit, so an unticked box means
            # the delivered o projection even though the default is ON.
            cmd.append("--p10" if params.get("p10") else "--no-p10")
            if params.get("xpack_once"):
                cmd.append("--xpack-once")
            job["stage"] = "startup"
            job["stage_detail"] = ("载荷驻留: cache-all + warm + 30 layers"
                                   if mode == "fast" else
                                   "载荷驻留: --no-cache-all + prefetch + warm")
            job["step"] = 0
            self._save(job)
            proc_start = time.time()
            dev_perf0 = time.perf_counter()
            startup_t = {"v": None}
            startup_tp = {"v": None}      # T20: child perf_counter at startup_done
            # T22 closurefix: the child's FIRST @@WEBUI@@ event, i.e. the end of
            # the interpreter+import segment this bucket table used to book as
            # startup.  Recorded per attempt (a retry restarts the anchor).
            first_ev = {"tp": None}
            step_done = {"v": 0}
            last_step = {"t": None, "tp": None}

            def on_line(line):
                self.append_log(job, line)
                m_sk = SKIP_RE.search(line)
                if m_sk:
                    pat = [int(x) for x in m_sk.group(2).replace(" ", "").split(",")
                           if x.strip()]
                    job["skip"] = {"mode": m_sk.group(1), "pattern": pat,
                                   "policy": m_sk.group(3),
                                   "why": (m_sk.group(4) or "").strip(),
                                   "line": line.strip()}
                m_fg = FASTGLUE_HOOKS_RE.search(line)
                if m_fg:
                    n_on, n_tot = int(m_fg.group(1)), int(m_fg.group(2))
                    job["fastglue_hooks"] = {"on": n_on, "total": n_tot}
                    if n_on < n_tot:
                        self._degrade(
                            job, "fastglue-incomplete",
                            "%d of %d host-glue hooks installed; the run takes "
                            "the slower path (measured +37 s at 512^2/8 steps, "
                            "122.8 vs 88.6 s loop, HOSTPROF-REPORT.md:264-266)"
                            % (n_on, n_tot))
                m = EVENT_RE.match(line)
                if m:
                    try:
                        ev = json.loads(m.group(1))
                    except Exception:
                        return
                    if first_ev["tp"] is None and ev.get("tp") is not None:
                        try:
                            first_ev["tp"] = float(ev["tp"])
                        except (TypeError, ValueError):
                            pass
                    if ev.get("event") == "startup_done":
                        startup_t["v"] = ev.get("t")
                        startup_tp["v"] = float(ev.get("tp") or 0.0) \
                            or time.perf_counter()
                        if "tp" in ev:
                            job["startup_s"] = max(0.0, float(ev["tp"]) - dev_perf0)
                        else:
                            job["startup_s"] = max(0.0, float(ev["t"]) - proc_start)
                        job["stage"] = "diffusion"
                        job["stage_detail"] = "扩散 0/%d" % params["steps"]
                    elif ev.get("event") == "cap_built":
                        job["cap"] = {"tokens": ev.get("tokens"),
                                      "shape": ev.get("shape"),
                                      "sha256": ev.get("sha256")}
                    elif ev.get("event") == "driver_error":
                        job["driver_trace"] = ev.get("trace", "")[-6000:]
                    elif ev.get("event") == "wpack_prewarm":
                        # T22 prewarm: the child mapped the artifact (it prints
                        # the reason above when it cannot and packs in-process).
                        timing["prewarm_loaded"] = True
                        timing["prewarm_map_s"] = ev.get("map_s")
                        timing["prewarm_verify_s"] = ev.get("verify_s")
                    return
                sm = STEP_RE.match(line)
                if sm:
                    i = int(sm.group(1))
                    step_done["v"] = i + 1
                    last_step["t"] = time.time()
                    last_step["tp"] = time.perf_counter()
                    job["step"] = i + 1
                    job["stage"] = "diffusion"
                    detail = f"扩散 {i + 1}/{params['steps']}"
                    est = job.get("estimate")
                    if isinstance(est, dict):
                        est["done_steps"] = i + 1
                        est["elapsed_s"] = round(
                            max(0.0, time.time() - job["started_ts"]), 2)
                        if est.get("step_s"):
                            est["eta_s"] = round(
                                float(est["step_s"]) * (params["steps"] - i - 1), 1)
                            detail += f" · 预估 {est['step_s']:.0f} s/步"
                    job["stage_detail"] = detail
                    self._save(job)

            dev_timeout = float(os.environ.get("ZIMAGE_DEVICE_TIMEOUT_S") or 0) \
                or device_timeout_s(params["steps"],
                                    shapes.m_from_hw(params["height"],
                                                     params["width"]),
                                    exec_mode=params.get("exec_mode", "native"),
                                    estimate=params.get("chunk_estimate"))
            self.append_log(job, "[webui] device attempt cap %.0f s "
                                 "(shape %dx%d, steps %d)"
                            % (dev_timeout, params["height"],
                               params["width"], params["steps"]))
            res = None
            if use_worker:
                # T22 workervae: hand the worker the prewarm artifact when it
                # already exists (it maps it instead of packing 1.85 GiB) and
                # the TE reserve for its residency policy.  No artifact -> the
                # worker packs in-process exactly as delivered; no packer is
                # started here, so the mapped file is never rewritten under it.
                wpack_for_worker = ""
                if (config.WPACK_PREWARM and not params.get("bo_resident")
                        and config.WPACK_FILE.is_file()):
                    wpack_for_worker = str(config.WPACK_FILE)
                w = self._get_worker(
                    cache_info, params,
                    config.JOBS_DIR / job["id"] / f"worker_{atag}.log",
                    wpack_file=wpack_for_worker,
                    reserve_gb=WORKER_TE_RESERVE_GB,
                    keep_payload=WORKER_KEEP_PAYLOAD,
                    keep_refiner=WORKER_KEEP_REFINER)
                try:
                    # P0-4 / closure: the clock starts BEFORE a cold prewarm.
                    # The worker's device open + 2 GB W pack is time this job
                    # really spent, so it lands in startup_s (the 0.0 s "warm"
                    # line is only for an already-warm worker).
                    dev_perf0 = time.perf_counter()
                    if not w.alive():
                        self.append_log(job, "[webui] warming the resident device "
                                              "worker (device + 3 xclbins + packed "
                                              "projection images)")
                        w.start(on_line=on_line, timeout=300.0)
                    job["_proc"] = w.proc
                    res = w.run_job(
                        {"op": "run", "id": atag, "tag": atag,
                         "pe": str(pe_path), "schedule": str(sched_path),
                         "steps": params["steps"], "seed": params["seed"],
                         "batch": params["batch"],
                         "qkv": bool(params.get("qkv")),
                         "p10": bool(params.get("p10")),
                         "xpack_once": bool(params.get("xpack_once"))},
                        on_line=on_line, timeout=dev_timeout, cancel_check=ck)
                except dev_worker.WorkerDead as exc:
                    self.append_log(job, "[webui] resident device worker "
                                          "unavailable (%s); falling back to a "
                                          "fresh bridge process" % exc)
                    use_worker = False
                    self._drop_worker()
                    res = None
                finally:
                    job["_proc"] = None
            if res is None:
                # T22 prewarm: hand the bridge the packed projection images.
                # The join happens BEFORE the child's clock (dev_perf0), so the
                # child-derived startup_s is the child's own startup; the wait is
                # added to startup_s at the seal below, and device_tail_s is
                # computed from the CHILD's startup -- never from the sum, or the
                # wait would push the tail negative and break the closure.
                if not use_worker:
                    wpack_file = self._wpack_join(job, timing)
                    if wpack_file:
                        cmd += ["--wpack-file", str(wpack_file)]
                        if config.WPACK_VERIFY:
                            cmd.append("--wpack-verify")
                    elif config.WPACK_PREWARM and self.wpack:
                        self.append_log(job, "[webui] T22 prewarm: no usable "
                                              "artifact; packing in-process exactly "
                                              "as delivered")
                dev_perf0 = time.perf_counter()
                res = stream_process(
                    cmd, env=config.subprocess_env("device"), cwd=config.NPU,
                    log_path=config.JOBS_DIR / job["id"] / f"bridge_{atag}.log",
                    on_line=on_line, timeout=dev_timeout, cancel_check=ck,
                    on_proc=lambda p: job.__setitem__("_proc", p))
                job["_proc"] = None
            # T22 closurefix: which process served THIS attempt's result.  The
            # bridge import segment only exists when the delivered bridge ran;
            # a worker job's startup_s keeps the old meaning (dev_perf0 ->
            # startup_done, cold worker start included) and books 0 here.
            ran_in_worker = bool(res is not None and use_worker)
            launch_tp = dev_perf0
            if res.get("dead"):
                self.append_log(job, "[webui] resident device worker died mid-job; "
                                      "the retry (if any) uses a fresh bridge process")
                use_worker = False
                self._drop_worker()
            rc = res["rc"]
            tail = "\n".join(res["lines"][-300:])
            calls_path = config.IMAGE_DIR / f"{atag}_calls.json"
            latents_path = config.IMAGE_DIR / f"{atag}_latents.npy"
            calls = None
            complete = False
            if calls_path.is_file():
                try:
                    calls = json.loads(calls_path.read_text(encoding="utf-8"))
                except Exception:
                    calls = None
            if calls and len(calls.get("steps_rec", [])) >= params["steps"] and latents_path.is_file():
                complete = True
            attempts.append({"tag": atag, "rc": rc, "complete": complete,
                             "steps": step_done["v"], "seconds": res["seconds"]})
            job["attempts"] = list(job["attempts"]) + [attempts[-1]]
            if complete:
                timing["device_wall_s"] = res["seconds"]
                if rc != 0:
                    job["teardown_rc"] = rc
                    self.append_log(job, f"[webui] artifacts complete; process exited "
                                          f"rc={rc} after writing latents (known "
                                          f"teardown 0xC0000005 class)")
                break
            if res["cancelled"] or job.get("cancel_requested"):
                raise Cancelled()
            low = tail.lower()
            mem = ("memoryerror" in low or "unable to allocate" in low
                   or "failed to allocate host memory" in low
                   or "out of device memory" in low)
            crashed = (rc in CRASH_CODES or "0xc0000005" in low
                       or "access violation" in low)
            if mem:
                raise MemoryFailure(
                    "device loop ran out of memory", raw=tail,
                    hint=("The full-memory path needs ~6 GB of private commit; "
                          "close other apps or let the UI auto-select low-memory "
                          "mode (--no-cache-all)."))
            if crashed and step_done["v"] == 0 and attempt < 3:
                timing["retry_attempts"] += 1
                self.append_log(job, f"[webui] known first-dispatch 0xC0000005; "
                                      f"retry {attempt + 1}/3 in a fresh process "
                                      f"(this failed process cost {res['seconds']:.1f} s)")
                time.sleep(2.0)
                timing["retry_s"] += time.perf_counter() - t_attempt0
                continue
            hint = ""
            if crashed:
                hint = ("Known cold-start crash: the first NPU dispatch can raise "
                        "0xC0000005 (access violation). This is not a parameter "
                        "error; rerun in a new process usually passes. If it "
                        "persists, check the driver stack.")
            raise config.WebUIError(
                f"device loop failed (rc={rc})", raw=tail, hint=hint)

        if not complete:
            raise config.WebUIError("device loop ended without complete artifacts",
                                    raw=tail, hint="see the log")

        # T22 closurefix / T22 prewarm: the device stage's four anchors, all
        # child-measured against the parent's launch mark (perf_counter is
        # system-wide on Windows, so a child tp is comparable to a parent
        # perf_counter):
        #     launch -> first @@WEBUI@@ event   bridge_import_s   (bridge only)
        #     first event -> startup_done      startup_s         (child startup)
        #     startup_done -> last step        loop_s
        #     last step -> exit                device_tail_s
        # The tail must still subtract the WHOLE launch->startup_done span
        # (import + startup): device_wall starts inside stream_process, i.e.
        # AFTER the launch mark, so tail = device_wall - (import + startup) -
        # loop.  That is what makes the split pure attribution: the sum
        # startup_s + loop + tail is bit-identical to the old bookkeeping.
        #
        # The prewarm wait (cold build) is MOVED out of startup into its own
        # bucket -- it is time this job blocked, and putting it in startup is
        # why a log reconstruction (startup_done - first_event) could never
        # close a cold-build job.
        device_start = float(job.get("startup_s", 0.0))   # old startup_s
        startup_wait = float(timing.get("prewarm_wait_s") or 0.0)
        sd_tp = startup_tp["v"]
        ftp = first_ev["tp"]
        bridge_import = 0.0
        startup_child = device_start
        if ((not ran_in_worker) and sd_tp and ftp is not None
                and ftp >= launch_tp):
            # delivered bridge: split launch->startup_done at the child's first
            # event, so startup_s then equals exactly what a reconstruction can
            # read from the job's own log.
            bridge_import = max(0.0, ftp - launch_tp)
            device_start = max(0.0, sd_tp - launch_tp)
            startup_child = max(0.0, sd_tp - ftp)
        if timing.get("prewarm_src") and not timing.get("prewarm_loaded"):
            # we handed the child a file and it refused it: say so, do not
            # book the delivered in-process pack as a reuse
            timing["prewarm_src"] = "fallback"
        timing["bridge_import_s"] = bridge_import
        timing["startup_child_s"] = startup_child
        timing["startup_s"] = startup_child
        timing["prewarm_pack_s"] = startup_wait
        # raw anchors for an offline reconstruction (informational, never booked)
        timing["device_launch_tp"] = float(launch_tp)
        if ftp is not None:
            timing["first_event_tp"] = float(ftp)
        if sd_tp:
            timing["startup_done_tp"] = float(sd_tp)
        job["startup_s"] = startup_child
        if startup_wait or timing.get("prewarm_src") or bridge_import:
            self.append_log(job, "[webui] device stage: bridge import %.2f s + "
                                  "startup %.2f s (+ prewarm wait %.2f s, pack %s, "
                                  "%.2f s, src %s)"
                            % (bridge_import, startup_child, startup_wait,
                               "reused" if not timing.get("prewarm_pack_wall_s")
                               else "built",
                               float(timing.get("prewarm_pack_wall_s") or 0.0),
                               timing.get("prewarm_src") or "none"))
        steps_rec = calls.get("steps_rec") or []
        # T20: the diffusion bucket is the SPAN startup_done -> last step line,
        # not the sum of the driver's step walls.  Step 0's timer starts BEFORE
        # startup_done (the cap embedding is built inside step 0), so summing
        # the walls counts that interval twice -- measured +0.65 s at 128x128
        # (just inside the +-1 s gate) and +1.05 s at 256x256 (the only 256x256
        # job through the runner failed the closure gate on this alone, with
        # 0 retries).  Span arithmetic closes the partition exactly:
        #     launch -> startup_done (startup) -> last step (loop) -> exit (tail)
        steps_sum = sum(float(s.get("seconds", 0.0)) for s in steps_rec)
        if startup_tp["v"] and last_step["tp"]:
            loop_span = max(0.0, last_step["tp"] - startup_tp["v"])
        else:
            loop_span = steps_sum
        timing["loop_s"] = loop_span
        timing["loop_steps_sum_s"] = steps_sum
        timing["loop_overlap_s"] = max(0.0, steps_sum - loop_span)
        timing["device_tail_s"] = max(
            0.0, timing["device_wall_s"] - device_start - timing["loop_s"])
        timing["device_s"] = float(calls.get("dev_ms_total", 0.0)) / 1000.0
        timing["staging_s"] = float(calls.get("write_ms_total", 0.0)) / 1000.0
        timing["dispatches"] = calls.get("dispatches")
        timing["retries"] = calls.get("retries")
        job["steps_rec"] = [{"step": s.get("step"), "t": s.get("t"),
                             "seconds": s.get("seconds"),
                             "seconds_per_image": s.get("seconds_per_image"),
                             "pred_absmax": s.get("pred_absmax")}
                            for s in steps_rec]
        lat_sha = _sha256(latents_path)
        self.append_log(job, f"[webui] loop complete: {timing['loop_s']:.2f} s, "
                              f"device {timing['device_s']:.2f} s, latents {lat_sha[:16]}")
        self._save(job)

        # ---- VAE (resident decoder in this py312 server; P0-2)
        vae_result = None
        crop_meta = None
        images = []
        vae_link = ""          # T22 C-1: the worker's VAE link, if usable
        stem = f"{params['prefix']}_{jid_short}_p{pass_index}{tag_suffix}"
        # T22 closurefix: a deterministic door for the ONE-SHOT fallback
        # (default OFF; with it unset the product path is unchanged).  Without
        # it the py312 spawn segment can only be reached by making vae_local
        # raise, which is not a repeatable A/B.  The door does nothing else:
        # the fallback branch below is the delivered one, it just runs first.
        oneshot = os.environ.get("ZIMAGE_VAE_ONESHOT", "") not in (
            "", "0", "false", "False")
        if vae and oneshot:
            self.append_log(job, "[webui] ZIMAGE_VAE_ONESHOT: the delivered "
                                  "one-shot vae_worker.py (py312) is used instead of "
                                  "the resident decoder; the resident worker is "
                                  "dropped because the one-shot starts its own "
                                  "device server")
            drop_worker_timed()
        elif vae and self.worker is not None:
            # T22 C-1 (workervae): the VAE dispatches INSIDE the resident
            # device worker -- the process that already holds the DiT device
            # context -- over its loopback link (dev_worker.VaeLink <-> the
            # t22_vae_srv connect mode).  This replaces the T21 2a-b rule
            # "stop the worker before the VAE": two device contexts in two
            # processes killed that worker with 0xC0000005 in 2/2 runs, and
            # dropping the context inside the worker is a teardown crash, so
            # ONE context serving both is the only stable shape (C-1).  A
            # worker without a link (--vae-link 0, or an older one) keeps the
            # old, safe behaviour: no second context next to a live one.
            w = self.worker
            vae_link = w.vae_addr if w.alive() else ""
            if vae_link:
                self.append_log(job, "[webui] VAE runs on the resident device "
                                      "worker (T22 C-1: one device context "
                                      "for DiT and VAE)")
            else:
                self.append_log(job, "[webui] stopping the resident device "
                                      "worker before the VAE stage (no VAE "
                                      "link on this worker; two device "
                                      "contexts cannot coexist; T21 2a-b)")
                drop_worker_timed()
        if vae:
            job["stage"] = "vae"
            job["stage_detail"] = (f"VAE 解码 {params['batch']} 张 "
                                   f"({'NPU 链优先' if config.VAE_NPU else 'CPU, 常驻'})")
            self._save(job)
            out_dir = params["output_dir"]
            result_path = config.JOBS_DIR / job["id"] / f"vae_p{pass_index}{tag_suffix}.json"
            prev_link = os.environ.get("ZIMAGE_VAE_SRV_ADDR")
            env_set = bool(vae_link)
            if env_set:
                os.environ["ZIMAGE_VAE_SRV_ADDR"] = vae_link
            try:
                try:
                    if oneshot:
                        vae_result = None      # T22 closurefix test door
                    else:
                        from . import vae_local
                        vae_result = vae_local.decode_batch(
                            latents_path, out_dir, stem, seeds,
                            on_line=lambda s: self.append_log(job, s),
                            cancel_check=ck)
                        if vae_result.get("cancelled"):
                            raise Cancelled()
                        job["vae_resident"] = True
                        result_path.write_text(json.dumps(vae_result, indent=1),
                                               encoding="utf-8")
                except Cancelled:
                    raise
                except Exception as exc:  # noqa: BLE001
                    self.append_log(job, f"[webui] resident VAE failed "
                                          f"({type(exc).__name__}: {exc}); falling "
                                          f"back to a fresh vae_worker.py")
                    vae_result = None
                    if vae_link:
                        # the one-shot fallback below starts its own py310
                        # device server: it must never run next to a live
                        # worker (the 2a2b2 crash shape)
                        self.append_log(job, "[webui] dropping the resident "
                                              "device worker before the "
                                              "one-shot VAE fallback")
                        drop_worker_timed()
            finally:
                if env_set:
                    if prev_link is None:
                        os.environ.pop("ZIMAGE_VAE_SRV_ADDR", None)
                    else:
                        os.environ["ZIMAGE_VAE_SRV_ADDR"] = prev_link
            if env_set and self.worker is not None and not self.worker.alive():
                self.append_log(job, "[webui] the resident device worker died "
                                      "during the VAE stage; the next job "
                                      "restarts it (T22 C-1)")
                drop_worker_timed()
            if vae_result is None:
                job["vae_resident"] = False
                cmd = [config.PY312, config.VAE_WORKER,
                       "--latents", str(latents_path),
                       "--out-dir", out_dir,
                       "--prefix", stem,
                       "--seeds", ",".join(str(s) for s in seeds),
                       "--result", str(result_path)]
                # T22 closurefix: the child's own total_s starts inside
                # decode_batch, so the py312 interpreter/import/teardown of
                # THIS process is not in any vae_* bucket.  Parent wall minus
                # the child's self-report is exactly that segment.
                t_vae_child = time.perf_counter()
                res = stream_process(cmd, env=config.subprocess_env("base"),
                                     cwd=config.NPU,
                                     log_path=config.JOBS_DIR / job["id"] / f"vae_p{pass_index}{tag_suffix}.log",
                                     on_line=lambda s: self.append_log(job, s),
                                     timeout=600.0, cancel_check=ck)
                vae_child_wall = time.perf_counter() - t_vae_child
                if res["cancelled"] or job.get("cancel_requested"):
                    raise Cancelled()
                tail = "\n".join(res["lines"][-200:])
                if res["rc"] != 0 or not result_path.is_file():
                    raise config.WebUIError(f"VAE decode failed (rc={res['rc']})",
                                            raw=tail, hint="The latents are saved; see the log.")
                vae_result = json.loads(result_path.read_text(encoding="utf-8"))
                timing["vae_child_wall_s"] = vae_child_wall
                timing["vae_spawn_s"] = max(
                    0.0, vae_child_wall - float(vae_result.get("total_s") or 0.0))
            # T21-3B3: the decode bucket must say WHERE it ran.  "npu" is the
            # chain (conv/GroupNorm/upsample on the NPU, SiLU/add/attention
            # host fp32); "cpu" is the resident torch decoder; "mixed" means
            # the NPU path failed mid-batch and the rest fell back.
            if "vae_spawn_s" not in timing:
                # resident decoder / worker link: no child process, no spawn
                timing["vae_spawn_s"] = 0.0
            vae_impl = str(vae_result.get("impl") or "cpu")
            dec_s = float(sum(vae_result.get("decode_s", [])))
            timing["vae_load_s"] = float(vae_result.get("load_s", 0.0))
            timing["vae_npu_s"] = dec_s if vae_impl == "npu" else 0.0
            timing["vae_decode_s"] = dec_s if vae_impl != "npu" else 0.0
            timing["vae_save_s"] = float(sum(vae_result.get("save_s", [])))
            timing["vae_s"] = float(vae_result.get("total_s", 0.0))
            # whatever the VAE stage spent outside load/decode/save is booked
            # here instead of leaking into unaccounted (it is measured inside
            # the same stage: reading the latents file, the PNG sha256, the
            # per-image log round-trips)
            timing["vae_other_s"] = max(
                0.0, timing["vae_s"] - (timing["vae_load_s"] + dec_s
                                        + timing["vae_save_s"]))
            timing["vae_impl"] = vae_impl
            job["vae_impl"] = vae_impl
            # 2026-11-03: record the THREAD ARM next to the impl (see
            # _vae_thread_record).  It also rides on every image record below,
            # so a consumer that reads one image still knows which arm made it.
            vae_threads_rec = _vae_thread_record(vae_result)
            job["vae_decode_threads"] = vae_threads_rec
            timing["vae_thread_arm"] = vae_threads_rec["arm"]
            label = {"npu": "NPU 链", "cpu": "CPU, 常驻",
                     "mixed": "NPU 失败→CPU 回退"}[vae_impl]
            job["stage_detail"] = f"VAE 解码 {params['batch']} 张 ({label})"
            # T22 pad_crop: a non-16-multiple request runs the padded frame and
            # keeps [0:H, 0:W].  Do the trim AFTER the VAE has produced the
            # array/PNG, before the record is sealed; 16-multiple jobs skip this
            # whole block, so their PNG bytes/sha are unchanged.  The crop time
            # is booked into vae_save_s and vae_s so the closure stays whole.
            crop_meta = None
            if params.get("crop_applied"):
                req_h = int(params["output_height"])
                req_w = int(params["output_width"])
                run_h = int(params["height"])
                run_w = int(params["width"])
                crop_s = 0.0
                for _b, rec in enumerate(vae_result["files"]):
                    was_h, was_w = int(rec["height"]), int(rec["width"])
                    dt, _ = _crop_png_top_left(rec["path"], req_h, req_w)
                    crop_s += dt
                    rec["run_width"], rec["run_height"] = was_w, was_h
                    rec["width"], rec["height"] = req_w, req_h
                    rec["crop"] = {
                        "policy": "bottom-right-trim", "box": [0, 0, req_w, req_h],
                        "run_width": was_w, "run_height": was_h,
                        "output_width": req_w, "output_height": req_h,
                        "seconds": round(dt, 4),
                    }
                    rec["save_s"] = float(rec.get("save_s") or 0.0) + dt
                    if _b < len(vae_result.get("save_s") or []):
                        vae_result["save_s"][_b] = float(vae_result["save_s"][_b]) + dt
                    old_path = pathlib.Path(rec["path"])
                    new_path = old_path.with_name(
                        old_path.stem + "_crop%dx%d" % (req_w, req_h) + old_path.suffix)
                    try:
                        old_path.replace(new_path)
                        rec["file"], rec["path"] = new_path.name, str(new_path)
                    except OSError:
                        pass
                    rec["sha256"] = _sha256(rec["path"])
                    rec["bytes"] = pathlib.Path(rec["path"]).stat().st_size
                    self.append_log(job, "[webui] crop %s: %dx%d -> %dx%d "
                                          "([0:H,0:W], %.3f s) sha256=%s"
                                    % (rec["file"], was_w, was_h, req_w, req_h,
                                       dt, rec["sha256"][:16]))
                vae_result["total_s"] = float(vae_result.get("total_s") or 0.0) + crop_s
                vae_result["crop_s"] = crop_s
                crop_meta = {
                    "policy": "bottom-right-trim", "box": [0, 0, req_w, req_h],
                    "requested_width": req_w, "requested_height": req_h,
                    "run_width": run_w, "run_height": run_h,
                    "pad_width": run_w, "pad_height": run_h,
                    "seconds": round(crop_s, 4),
                    "semantics": ("generate the padded 16-px frame, then keep "
                                  "the requested top-left H x W rectangle; "
                                  "right/bottom excess is discarded"),
                }
                # The VAE stage's own result json is evidence too: keep it in
                # sync with the cropped records (the in-memory dict is already
                # what the job API reports).
                try:
                    result_path.write_text(json.dumps(vae_result, indent=1),
                                           encoding="utf-8")
                except OSError:
                    pass
                # Re-seal the VAE buckets AFTER the crop: the trim is real time
                # this stage spent, so vae_save_s/vae_s must include it and
                # vae_other_s must stay the residual.
                timing["vae_save_s"] = float(sum(vae_result.get("save_s", [])))
                timing["vae_s"] = float(vae_result.get("total_s", 0.0))
                timing["vae_other_s"] = max(
                    0.0, timing["vae_s"] - (timing["vae_load_s"] + dec_s
                                            + timing["vae_save_s"]))
            for rec in vae_result["files"]:
                images.append({
                    "index": rec["index"], "seed": rec["seed"],
                    "file": rec["file"], "path": rec["path"],
                    "url": f"/api/jobs/{job['id']}/images/{rec['index']}",
                    "width": rec["width"], "height": rec["height"],
                    "run_width": rec.get("run_width", rec["width"]),
                    "run_height": rec.get("run_height", rec["height"]),
                    "crop": rec.get("crop"),
                    "bytes": rec["bytes"], "sha256": rec["sha256"],
                    "decode_s": rec["decode_s"], "save_s": rec.get("save_s", 0.0),
                    # which thread arm produced THIS png (see job record)
                    "vae_decode_threads": vae_threads_rec,
                    "gate": "trial",
                    "tag": atag,
                })
        else:
            out_dir = pathlib.Path(params["output_dir"])
            latent_copy = out_dir / f"{stem}_latents.npy"
            shutil.copy2(latents_path, latent_copy)
            for b, seed in enumerate(seeds):
                images.append({
                    "index": b, "seed": seed, "file": None,
                    "latent_file": str(latent_copy),
                    "path": None, "url": None,
                    "width": params["width"], "height": params["height"],
                    "bytes": latent_copy.stat().st_size,
                    "sha256": lat_sha, "decode_s": 0.0, "gate": "trial",
                    "tag": atag,
                })

        B = params["batch"]
        te_share = timing.get("textenc_s", 0.0) / B
        startup_share = timing.get("startup_s", 0.0) / B
        loop_share = timing.get("loop_s", 0.0) / B
        for img in images:
            img["timing"] = {
                "textenc_share_s": te_share,
                "startup_share_s": startup_share,
                "loop_share_s": loop_share,
                "vae_s": float(img.get("decode_s", 0.0)),
                "vae_save_s": float(img.get("save_s", 0.0)),
                "total_share_s": (te_share + startup_share + loop_share
                                  + float(img.get("decode_s", 0.0))
                                  + float(img.get("save_s", 0.0))),
            }
            img["prompt"] = params["prompt"]
            img["negative_prompt"] = params["negative_prompt"]
            img["resolution"] = ("%dx%d" % (int(params["output_width"]),
                                            int(params["output_height"]))
                                 if params.get("crop_applied") else params["shape"])
            img["run_resolution"] = params["shape"]
            img["steps"] = params["steps"]
            img["mode"] = params["mode"]
            img["memory_mode"] = mode
            img["te_backend"] = te_backend
            img["created"] = config.now_iso()
        timing["pass_s"] = time.perf_counter() - t_pass
        # T21 gatefix: the 13 closure buckets describe ONE pass.  In gate mode
        # the second pass is accounted for wholesale by gate_second_s, so its
        # per-stage numbers must not be merged into job["timing"] here: that
        # overwrote the first pass's buckets and counted the second pass twice
        # (sum - one-time == 2 x pass1), which is how gate mode closed at
        # -3.54 s.  Pass 0 alone feeds the buckets; pass 1 stays in
        # job["gate"]["timing"]["second_total_s"].
        if pass_index == 0:
            job["timing"].update(timing)
        self._save(job)
        return {"tag": atag, "latents_path": str(latents_path),
                "latents_sha": lat_sha, "pe_sha": pe_sha, "tokens": tokens,
                "te": te, "te_backend": te_backend,
                "schedule": job["schedule"], "cap": job.get("cap"),
                "crop": crop_meta,
                "calls": calls, "vae": vae_result, "images": images,
                "timing": timing, "rc": rc,
                "teardown_rc": job.get("teardown_rc")}

    # ------------------------------------------------------------------ state
    def _normalize(self, raw):
        if not isinstance(raw, dict):
            raise config.WebUIError("POST body must be a JSON object")
        def s(key, default=""):
            v = raw.get(key, default)
            return "" if v is None else str(v)
        prompt = s("prompt")
        if not prompt.strip():
            raise config.WebUIError("prompt must not be empty")
        if len(prompt) > config.MAX_PROMPT_CHARS:
            raise config.WebUIError(f"prompt longer than {config.MAX_PROMPT_CHARS} chars")
        negative = s("negative_prompt")
        # T16 P0-5: arbitrary (H, W).  A legacy "resolution" is accepted as a
        # square shorthand so old clients keep working.  T22 pad_crop: JSON
        # floats and numeric strings are accepted only when integral; 130.5 is
        # a 400, not a silent truncation to 130.
        def _int_side(value):
            if isinstance(value, bool):
                raise ValueError("bool")
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                if not value.is_integer():
                    raise ValueError("fraction")
                return int(value)
            if isinstance(value, str):
                s = value.strip()
                if not re.fullmatch(r"[+-]?\d+", s):
                    raise ValueError("not an integer string")
                return int(s)
            raise ValueError("not a number")
        try:
            if "height" in raw or "width" in raw:
                height = _int_side(raw.get("height", raw.get("width", 128)))
                width = _int_side(raw.get("width", raw.get("height", 128)))
            else:
                height = width = _int_side(raw.get("resolution", 128))
        except Exception:
            raise config.WebUIError("height/width must be integers")
        try:
            plan = shapes.resolve(height, width)
        except shapes.ShapeError as exc:
            raise config.WebUIError(str(exc))
        if not plan["ready"] and not plan.get("chunked_runnable"):
            # 2026-11-03: the gate's own sentence is the binding constraint
            # (host BO / refiner / fastglue / attention), and it is the one
            # that actually stops this request.  reason_text() describes what
            # would have to be BUILT -- keep it as the fallback, and as the
            # hint, but never quote "needs a device build" for a shape whose
            # real blocker is a memory ceiling.
            why = plan.get("chunk_gate_reason") or plan["reason"]
            cut = why.find("未构建：")     # shapes.refusal() prefixes shape+M
            if cut >= 0:
                why = why[cut + len("未构建："):]
            exc = config.WebUIError(
                f"{plan['key']} (M={plan['m']}): {why}",
                hint=("binding constraint: " + why if plan.get("chunk_gate_reason")
                      else "build plan: " + "; ".join(plan.get("missing", []))))
            exc.detail = {
                "requested": {"height": height, "width": width},
                "padded": {"height": int(plan["height"]),
                           "width": int(plan["width"])},
                "output": {"height": height, "width": width},
                "pad_applied": bool(plan.get("pad_applied")),
                "crop_applied": bool(plan.get("crop_applied")),
                "needs_build": True,
                "chunk_gate_reason": plan.get("chunk_gate_reason") or "",
                "chunked_runnable": bool(plan.get("chunked_runnable")),
            }
            raise exc
        try:
            batch = int(raw.get("batch", 1))
            steps = int(raw.get("steps", 8))
            seed = int(raw.get("seed", 0))
        except Exception:
            raise config.WebUIError("batch/steps/seed must be integers")
        if not 1 <= batch <= config.MAX_BATCH:
            raise config.WebUIError(f"batch must be 1..{config.MAX_BATCH}")
        if not 1 <= steps <= config.MAX_STEPS:
            raise config.WebUIError(f"steps must be 1..{config.MAX_STEPS}")
        if not 0 <= seed <= 2**31 - 1:
            raise config.WebUIError("seed must be 0..2^31-1")
        vae = bool(raw.get("vae", True))
        mode = s("mode", "trial").lower()
        if mode not in ("trial", "gate"):
            raise config.WebUIError("mode must be 'trial' or 'gate'")
        mem = s("memory_mode", "auto").lower()
        if mem not in ("auto", "fast", "low"):
            raise config.WebUIError("memory_mode must be auto/fast/low")
        prefix = config.sanitize_prefix(s("prefix", "zimage"))
        out_dir = config.resolve_output_dir(s("output_dir", "webui/outputs"))
        bo = bool(raw.get("bo_resident", False))
        return {
            "prompt": prompt,
            "negative_prompt": negative,
            "resolution": height if height == width else None,
            "requested_height": height,
            "requested_width": width,
            "height": int(plan["height"]),
            "width": int(plan["width"]),
            "pad_height": int(plan["height"]),
            "pad_width": int(plan["width"]),
            "pad_applied": bool(plan.get("pad_applied")),
            "crop_applied": bool(plan.get("crop_applied")),
            "output_height": height,
            "output_width": width,
            "requested_shape": plan.get("requested_key", f"{height}x{width}"),
            "shape": plan["key"],
            "m": plan["m"],
            "exec_mode": plan.get("exec_mode", "native"),
            "chunk_estimate": plan.get("chunk_estimate"),
            "batch": batch,
            "steps": steps,
            "seed": seed,
            "vae": vae,
            "output_dir": str(out_dir),
            "prefix": prefix,
            "mode": mode,
            "memory_mode": mem,
            "bo_resident": bo,
            # chunk_path registers only the delivered projection/FFN image;
            # the fused qkv / p10 images are optional alternatives and must
            # not be requested for a shape whose cache does not carry them.
            "qkv": bool(raw.get("qkv", _QKV_ENV)) and plan.get("exec_mode") != "chunked",
            "p10": bool(raw.get("p10", _P10_ENV)) and plan.get("exec_mode") != "chunked",
            "xpack_once": bool(raw.get("xpack_once", _XPACK_ENV)),
        }

    def public(self, job):
        with self.lock:
            if not job:
                return None
            out = copy.deepcopy({k: v for k, v in job.items()
                                 if not k.startswith("_")})
            out["log_tail"] = list(job.get("log_tail", []))
            out["log_url"] = f"/api/jobs/{job['id']}/log"
            out["elapsed_s"] = ((time.time() - job["created_ts"])
                                if job["status"] in ("queued", "running")
                                else out.get("elapsed_s"))
            out["queue_position"] = None
            if job["status"] == "queued":
                out["queue_position"] = 1
            return out

    def append_log(self, job, line):
        line = (line or "").rstrip("\r\n")
        with self.lock:
            tail = job.setdefault("log_tail", [])
            tail.append(line)
            if len(tail) > LOG_TAIL:
                del tail[:len(tail) - LOG_TAIL]
            job["log_lines"] = job.get("log_lines", 0) + 1
        try:
            t_io = time.perf_counter()
            with open(config.JOBS_DIR / job["id"] / "job.log", "a",
                      encoding="utf-8", errors="replace") as f:
                f.write(line + "\n")
            t = job.get("timing")
            if isinstance(t, dict) and not job.get("_timing_sealed"):
                t["log_io_s"] = t.get("log_io_s", 0.0) + (time.perf_counter() - t_io)
        except Exception:
            pass

    def _save(self, job):
        try:
            t0 = time.perf_counter()
            d = config.JOBS_DIR / job["id"]
            d.mkdir(parents=True, exist_ok=True)
            pub = self.public(job)
            tmp = d / "job.json.tmp"
            tmp.write_text(json.dumps(pub, indent=1, ensure_ascii=False),
                           encoding="utf-8")
            tmp.replace(d / "job.json")
            t = job.get("timing")
            if isinstance(t, dict) and not job.get("_timing_sealed"):
                t["job_save_s"] = t.get("job_save_s", 0.0) + (time.perf_counter() - t0)
        except Exception:
            pass

    def _load_history(self):
        if not config.JOBS_DIR.is_dir():
            return
        dirs = sorted([d for d in config.JOBS_DIR.iterdir() if d.is_dir()],
                      key=lambda p: p.stat().st_mtime, reverse=True)[:40]
        for d in dirs:
            p = d / "job.json"
            if not p.is_file():
                continue
            try:
                job = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if job.get("status") in ("queued", "running"):
                job["status"] = "error"
                job["stage"] = "error"
                job["stage_detail"] = "server restarted while this job was running"
                job["error"] = {"message": "server restarted", "raw": "", "hint": "re-submit"}
                job["finished"] = config.now_iso()
                job["finished_ts"] = time.time()
            job.setdefault("log_tail", [])
            job.setdefault("params", {})
            job["id"] = d.name
            self.jobs[job["id"]] = job
