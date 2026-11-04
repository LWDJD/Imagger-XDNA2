# SPDX-License-Identifier: MIT
"""Shared paths, limits and subprocess environments for the WebUI.

The WebUI is a *front end*: it never changes the verified drivers.  It calls
the delivered t13_run.py through webui/bridge.py (py310) and the delivered
text/VAE paths through small py312 workers.
"""
from __future__ import annotations

import glob
import os
import pathlib
import re
import tempfile
import threading
import time

WEBUI = pathlib.Path(__file__).resolve().parent
NPU = WEBUI.parent
WS = NPU.parent

PY312 = pathlib.Path(r"D:\Users\15081\miniforge3\envs\ryzen-ai-1.8.0\python.exe")
PY310 = pathlib.Path(r"D:\Users\15081\AppData\Local\GAIA\python\python.exe")
XRT_DIR = pathlib.Path(r"D:\Project\deepseek_harness\test7\WStmp\driver376\npu_mcdm_stack_prod")
PEANO = WS / "_pylibs" / "llvm-aie" / "llvm-aie"

PORT = WS / "zimage-port"
MODELS = PORT / "_models" / "Z-Image-Turbo"
IMAGE_DIR = NPU / "imaggers" / "image"
ATT_DIR = NPU / "imaggers" / "attn_impl"
T4_OUT = NPU / "imaggers" / "block" / "t4_out"
# T22 kshape (2026-10-21): the product FFN image default flips to the
# B-object variant fs_ffn_j5 (one B fifo object = the NH j-pairs of one
# (pass, kk)).  Product gate: latents 9f86aee9 byte-identical to t11_nh5,
# 960/0 dispatches (bridge), closure 0; FFN device bucket 10.98 -> 8.21 s.
# ROLLBACK: ZIMAGE_FFN_IMAGE=t11_nh5 (or revert this one line).
FFN_IMAGE = os.environ.get("ZIMAGE_FFN_IMAGE", "fs_ffn_j5")
T5_OUT = ATT_DIR / "t5_out"
REFINER_CACHE = IMAGE_DIR / "t6_refcache"

CACHE_DIR = WEBUI / "cache"
JOBS_DIR = WEBUI / "jobs"
DEFAULT_OUTPUT = WEBUI / "outputs"
TE_WORKER = WEBUI / "te_worker.py"
VAE_WORKER = WEBUI / "vae_worker.py"

# T21-3B3: the NPU VAE chain (conv + GroupNorm + upsample on the NPU; SiLU /
# add / attention stay host fp32 -- see imaggers/vae/T21-3B3-VAE.md section 3).
# vae_local.py tries this path first and falls back to the resident CPU
# decoder on any failure; ZIMAGE_VAE_NPU=0 pins the CPU path.
VAE_CHAIN = NPU / "imaggers" / "vae" / "t21_3b3_chain.py"
# 2026-10-21 decision (owner): default to the CPU decoder.  Measured
# imaggers/t22_e2e_measure/vae_timing: CPU torch decode 128^2 ~1 s warm /
# 192x128 0.61 s / 256^2 1.40 s / 512^2 5.15 s vs NPU chain 61.9 / 119 /
# 362.7 s -- the chain is launch-tax bound (431 dispatches @ ~0.84 s at 256^2).
# PNG bytes follow the CPU decoder (104ff564 family), not the NPU-chain stream
# (e7aef10c family); each is internally stable and they are not comparable.
# ROLLBACK: ZIMAGE_VAE_NPU=1 restores the NPU chain (keep for much larger sizes
# or after conv/GN/SiLU fusion work).
VAE_NPU = os.environ.get("ZIMAGE_VAE_NPU", "0") not in ("0", "false", "False", "")

# T22-A (2026-10-20, commit a7f45e0's gates): the NPU VAE launch architecture.
# "srv" = ONE py310 device server per host process, one launch per request,
# xclbin/kernel/BOs resident.  Measured (T22-VAE-REPORT.md 2.2/2.3):
#   128^2 decode 115.64 -> 64.90 s (-43.9 %), same-window A/B 2.26x,
#   per-launch tax 350 -> 10-19 ms, and ALL SIX product sizes are BYTE
#   IDENTICAL to the 3b3 delivery (np.array_equal on <size>_npu.npy) with an
#   unchanged PSNR, closure <= 0.70 s.
# ROLLBACK: ZIMAGE_VAE_LAUNCH=proc restores the 3b3 one-process-per-launch
# executor (same xclbins, same bytes, ~2x slower).  A device server that cannot
# start falls back to proc by itself, and any NPU failure still falls back to
# the resident CPU decoder.
VAE_LAUNCH = os.environ.get("ZIMAGE_VAE_LAUNCH", "srv")
# T22-B (2026-10-20): the residual adds on the NPU.  "add" routes the 14 resnet
# residual adds through the f32-output add (vaddf).  It is byte-identical at
# 128^2 and 64x256 but NOT at 144x128 (38.0508 -> 37.9919, -0.059 dB): the AIE
# scalar f32 add differs from x86 SSE on exact ties (1 ulp on ~1e-6 of the
# elements, report 3.2c / platform fact F-21), and one flipped element is
# enough to move a pixel.  So the DEFAULT STAYS 0 -- the adds stay host fp32,
# the 3b3 path -- until the integer tie compensation (queue item 3a) makes it
# bit-exact.  ZIMAGE_VAE_OPNPU=add opts in for experiments.
VAE_OPNPU = os.environ.get("ZIMAGE_VAE_OPNPU", "0")
# the chain reads its own environment; this keeps ONE source of truth
os.environ.setdefault("ZIMAGE_VAE_LAUNCH", VAE_LAUNCH)
os.environ.setdefault("ZIMAGE_VAE_OPNPU", VAE_OPNPU)

# --------------------------------------------------------------- CPU threads
# Two places in this product run torch on the host: the device loop's f32
# attention (py310 child, webui/chunked.py:85) and the CPU VAE decoder (py312
# server, webui/vae_local.py).  The loop reads its pool through loop_threads();
# the decode reads its own, geometry-aware, through vae_decode_threads(h, w).
#
# The two are DELIBERATELY DECOUPLED (2026-11-03).  Until this change an unset
# ZIMAGE_VAE_THREADS followed ZIMAGE_CHUNK_THREADS, so a launcher that pinned
# the loop to 20 silently moved the decode -- and the decode's bytes move with
# the thread count.  A loop knob must not move an image.
#
# ZIMAGE_CHUNK_THREADS   0 / unset = leave torch's default (the shipped byte
#                        path; 10 intra-op threads on a 20-logical-CPU box).
#                        N > 0 sets the loop's pool.  It does NOT touch the VAE.
# ZIMAGE_VAE_THREADS     unset / auto = the geometry rule below.  N > 0 = pin N.
#                        0 = leave torch's default.  An explicit pin always wins
#                        (that is the escape hatch the 2026-11-03 A/B used).
# ZIMAGE_VAE_THREAD_RULE 0 = disable the geometry rule and fall back to
#                        ZIMAGE_VAE_THREADS / torch's default (the old
#                        behaviour).  Anything else, or unset, = the rule.
#
# THE RULE (LANDED 2026-11-03; evidence imaggers/par/vaethreads/REPORT.md and
# its 12-script predecessor imaggers/par/vae1lsb/REPORT.md):
#   decode at VAE_DECODE_THREADS (20) when latent_H * latent_W % 5 == 0,
#   otherwise at VAE_DECODE_THREADS_GUARD (10).
# Why 5: the decoder's 29 SiLU activations have numel 512*X, 2048*X, 4096*X,
# 8192*X and 16384*X with X = latent_H*latent_W.  torch's CPU SiLU runs a
# vectorised body and finishes each thread range (ceil(numel/T) elements) with
# a SCALAR remainder; the remainder positions -- and so the last 1-2 ulp of a
# handful of elements -- move with T.  All five numels are multiples of 320
# exactly when X % 5 == 0, and then the remainder is the same at T=10 and T=20.
# The other 88 arithmetic ops (oneDNN conv, GroupNorm, GEMM, SDPA, upsample,
# add) are bit-identical across thread counts: measured, not assumed.
#
# What each arm is, measured END TO END (product decode, PNG bytes):
#   20 threads, X % 5 == 0  -> 0 f32 values moved, identical PNG.  Verified at
#       320^2, 640^2, 800^2, 960^2 and 1088x1920 (the product's 1080p pad), and
#       here at 1280x720 + 480^2.  Decode 1.2-1.4x faster (1088x1920, five
#       samples: 10 threads 51.2-58.0 s, 20 threads 40.3-44.0 s).  The
#       campaign's 2.16x at this size came from a CONTENDED 10-thread sample;
#       its 20-thread 39.5 s agrees with these, which is the honest pair.
#   10 threads, any other geometry -> today's shipped bytes, bit for bit.  That
#       is the class where 20 threads moves 4-7 px of 786,432 by <=1 LSB, max
#       f32 1.76e-06 = 2.2e-04 of one 8-bit LSB, PSNR 143.5 dB.
# So the rule never spends the <=1 LSB budget outside the exact class, and a
# wrong prediction (a future torch changing the SiLU loop) degrades to the
# 1-LSB arm -- never to a crash and never to a new magnitude.
#
# "SHIPPED BYTES" ARE NOT PORTABLE, EITHER WAY: torch's default intra-op count
# is the box's physical core count (10 here, 16 on a 16-core box), and five
# thread counts give five different outputs.  Leaving the default pins nothing;
# pinning 10 pins the bytes to THIS machine.  The rule pins them by GEOMETRY
# instead, which is a property of the request.  On the amplitude question the
# project's own doctrine (docs/KNOWN-PLATFORM-FACTS.md F-21 v1.6) has already
# retired byte-identity as a gate for tie-sensitive ops in favour of an
# amplitude bound (ulp <= 1); the measured amplitudes here are 1-2 ulp per op,
# RMS 1.1 ulp end to end -- inside that bound.
#
# HISTORY: the first landing (VAE-THREADS-LANDED.md, 2026-11-03) made 20
# threads an env-only opt-in because the 512^2 PNG moved.  The 1-LSB report
# then showed the effect is a deterministic function of (numel, threads), which
# turns "20 is unsafe" into "20 is exact at these sizes and <=1 LSB at the
# others".  20-thread PNG 0f689f8b vs 10-thread a131cc55 (jobs ...-144746-ad9978
# / ...-144345-de7620) is the measurement that started it.
VAE_DECODE_THREADS = 20            # the exact-class arm (see above)
VAE_DECODE_THREADS_GUARD = 10      # the byte-preserving arm (today's bytes)
VAE_SILU_MODULUS = 5               # exact iff latent_H*latent_W % 5 == 0
# Flipped to True only because every check in imaggers/par/vaethreads/ passed:
# 1088x1920 end to end, 512^2's 1-LSB arm, 320^2/640^2/960^2 exact, 1280x720
# predicted-exact.  Set False to go back to "the decode follows an explicit
# env or torch's default".
VAE_THREAD_RULE_DEFAULT = True


def _int_env(name, default=0):
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = str(raw).strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# t22_loud: ONE expression of "this switch is on" for the whole tree.
# Before this, T22_FASTGLUE had four readers with four parsers and three
# defaults (install.py '!= "1"', t18_run '== "1"' on the OTHER spelling,
# bridge.py 'not in off-list' defaulting ON, shapes.py a third off-list), so a
# single value could install 0 hooks, forward the driver flag and still make
# capability.py compute its size table for the fastglue projection.  Every
# reader now calls this.  Unset/empty = the caller's default; anything else is
# read as a boolean, not as the string "1".
_TRUTHY = ("1", "true", "yes", "on", "y", "t")


def env_flag(name, default=False):
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return bool(default)
    return str(raw).strip().lower() in _TRUTHY


# The env that changes WHICH device path a job runs.  Recorded per job (the
# 2026-11-03 audit: two 512^2 8-step jobs in this tree differ 2.5x in the loop,
# 72.3 s vs 182.4 s, because one launcher set the devkill/bigm path and the job
# record did not say so).  Recording only -- nothing here selects anything.
LAUNCH_FLAGS = ("ZIMAGE_BIGM", "ZIMAGE_BIGM_PROJ", "ZIMAGE_CHUNK_PIPE",
                "ZIMAGE_ADAPT", "T22_HOSTKILL", "T22_DEVKILL",
                "T22_BIGM_DEVHANDOFF", "T22_FASTGLUE", "T22_PLAN_FFN",
                "T22_PLAN_FUSE_GU",
                "T22_ADAPT_M", "ZIMAGE_CHUNK_THREADS", "ZIMAGE_VAE_FAST",
                "ZIMAGE_VAE_THREADS", "ZIMAGE_VAE_THREAD_RULE",
                "ZIMAGE_VAE_NPU")


def launch_flags():
    """{env name: value or None} for the flags that pick a device path."""
    return {k: os.environ.get(k) for k in LAUNCH_FLAGS}


# t22_loud: launch_flags() answers "what did the shell say", which for an unset
# name is None -- and None is what three DEFAULT-ON paths report too, so the
# record cannot tell "the product chose" from "I chose the default".  These are
# the switches that pick a path or change bytes and were in NO record at all.
# (env_flags keeps its old shape on purpose: the runbook's Q11 compares it.)
def resolved_flags():
    """{name: {value, source, raw}} for the flags the record used to omit."""
    out = {}
    for name, value, default in (
            ("ZIMAGE_FFN_IMAGE", FFN_IMAGE, "fs_ffn_j5"),
            ("T22_QKV", bool(T22_QKV_DEFAULT), True),
            ("T22_P10", bool(T22_P10_DEFAULT), True),
            ("T22_FAST_GLUE", env_flag("T22_FAST_GLUE", True), True),
            ("T22_GLUE2", env_flag("T22_GLUE2", True), True),
            ("ZIMAGE_VAE_NPU", bool(VAE_NPU), False),
            ("T22_REF_SMNB", env_flag("T22_REF_SMNB", False), False)):
        raw = os.environ.get(name)
        explicit = raw is not None and str(raw).strip() != ""
        out[name] = {"value": value,
                     "source": "env" if explicit else "default",
                     "raw": raw,
                     "default_is": default}
    return out


def loop_threads():
    """ZIMAGE_CHUNK_THREADS as an int; 0 = leave torch's default alone."""
    return max(0, _int_env("ZIMAGE_CHUNK_THREADS", 0))


def vae_thread_rule():
    """Is the geometry rule on?  ZIMAGE_VAE_THREAD_RULE overrides the constant."""
    raw = str(os.environ.get("ZIMAGE_VAE_THREAD_RULE", "") or "").strip().lower()
    if raw in ("", "auto", "-1"):
        return bool(VAE_THREAD_RULE_DEFAULT)
    return raw not in ("0", "false", "off", "no", "none", "default", "torch")


def vae_threads_pinned():
    """ZIMAGE_VAE_THREADS as an int, or None when unset/"auto" (no pin).

    0 is a real answer: "leave torch's default alone", the pre-2026-11-03
    shipped path.
    """
    raw = str(os.environ.get("ZIMAGE_VAE_THREADS", "") or "").strip().lower()
    if raw in ("", "auto", "-1"):
        return None
    if raw in ("0", "off", "default", "torch"):
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return None


def vae_geometry_class(latent_h, latent_w):
    """'exact' iff the 20-thread decode is bit-identical at this geometry.

    The predicate is one line of arithmetic: every SiLU numel is 512*X,
    2048*X, 4096*X, 8192*X or 16384*X with X = latent_H*latent_W, and all of
    them are multiples of 320 (the vec-body/scalar-remainder period at 10 and
    20 threads) exactly when X is a multiple of 5.
    """
    h, w = int(latent_h), int(latent_w)
    if h <= 0 or w <= 0:
        return "none"
    return "exact" if (h * w) % VAE_SILU_MODULUS == 0 else "lsb1"


def vae_decode_threads(latent_h, latent_w):
    """(n, arm, why) -- torch intra-op threads for ONE decode; 0 = leave default.

    This is the one home for the decode's thread decision.  It is deliberately
    independent of ZIMAGE_CHUNK_THREADS: the decode's bytes move with the
    thread count, so a loop knob must not be able to move an image.

    arm  "exact"   20 threads on a geometry whose 20-thread decode is
                   bit-identical (latent_H * latent_W % 5 == 0)
         "lsb1"    10 threads on any other geometry: today's shipped bytes.
                   20 threads would move <=1 LSB there; this rule does not
                   spend that budget.
         "pinned"  ZIMAGE_VAE_THREADS was set explicitly -- always wins, so the
                   2026-11-03 A/B and any future one can still force an arm
         "legacy"  the rule is off: ZIMAGE_VAE_THREADS or torch's default
    """
    h, w = int(latent_h), int(latent_w)
    klass = vae_geometry_class(h, w)
    pinned = vae_threads_pinned()
    if pinned is not None:
        return (pinned, "pinned",
                "ZIMAGE_VAE_THREADS=%d explicit (geometry class %s)"
                % (pinned, klass))
    if klass == "none":
        return 0, "legacy", "no usable latent geometry; torch default"
    if not vae_thread_rule():
        return (0, "legacy",
                "ZIMAGE_VAE_THREAD_RULE=0 (geometry class %s); torch default "
                "= the shipped byte path" % klass)
    if klass == "exact":
        return (VAE_DECODE_THREADS, "exact",
                "latent %dx%d = %d = 0 mod %d: every SiLU numel = 0 mod 320, so "
                "20 threads is bit-identical"
                % (h, w, h * w, VAE_SILU_MODULUS))
    return (VAE_DECODE_THREADS_GUARD, "lsb1",
            "latent %dx%d = %d = %d mod %d: 20 threads would move <=1 LSB, so "
            "keep 10 (today's bytes)"
            % (h, w, h * w, (h * w) % VAE_SILU_MODULUS, VAE_SILU_MODULUS))


def vae_threads():
    """Legacy no-geometry answer: ZIMAGE_VAE_THREADS, else 0 (torch default).

    Kept for callers that have no latent shape.  Without a geometry the only
    safe answer is the shipped byte path (0), never the rule's 20 -- the decode
    itself calls vae_decode_threads(h, w), which is where the rule lives.
    """
    pinned = vae_threads_pinned()
    return 0 if pinned is None else int(pinned)


# T21 2a-b: resident py310 device worker.  It keeps the device context, the
# three registered xclbins and the packed projection images between jobs; the
# delivered one-process-per-job bridge remains the fallback for every other
# path (low memory, other shapes, --bo-resident, worker death).
# ZIMAGE_DEV_WORKER=0 pins the delivered path.
# DEFAULT OFF (opt-in): measured 2026-10-19 -- the worker takes startup from
# 35.2 s to 4.3-5.3 s and is byte-identical, but a resident device context
# cannot be held across the default VAE-NPU stage (0xC0000005), so the product
# default stays on the delivered one-process bridge.  Enable it for VAE-free
# runs (or with ZIMAGE_VAE_NPU=0): startup 35.2 -> ~5 s.
DEV_WORKER = os.environ.get("ZIMAGE_DEV_WORKER", "0") not in ("0", "false", "False", "")
BRIDGE = WEBUI / "bridge.py"
STATIC = WEBUI / "static"

# T22 prewarm: the first job's startup carries the --warm pack of the 120
# projection images (30 layers x 4 x 16,588,800 B = 1.85 GiB, 22-37 s measured).
# That pack is pure CPU (bridge.py --pack-prewarm).  This switch decides whether
# the runner builds that artifact AHEAD of the job, in a sibling py310 process,
# and hands the file to the job (--wpack-file) so the job's warm loop is a
# cache hit:
#   off    the delivered in-process pack, every job (startup 35.7 s scale)
#   job    build it during the job's prepare/TE window; the job waits for it at
#          the device stage (default: no work at all when no job runs)
#   boot   ALSO start it when the runner starts (the delivered 128x128 shape),
#          so a user who takes >25 s to submit the first job gets the whole pack
#          off the critical path; auto = boot + job
# The artifact lives in webui/cache/wpack (gitignored, 1.85 GiB) and is
# validated against weights.bfp16's stamp on every use: a stale or missing file
# is a log line plus the delivered pack, never a guess.
_WPACK_MODE = os.environ.get("ZIMAGE_WPACK_PREWARM", "job").strip().lower()
WPACK_PREWARM = _WPACK_MODE not in ("0", "false", "off", "no", "none", "")
WPACK_PREWARM_JOB = WPACK_PREWARM and _WPACK_MODE in ("auto", "job", "boot",
                                                      "1", "true", "on", "yes")
WPACK_PREWARM_BOOT = WPACK_PREWARM and _WPACK_MODE in ("auto", "boot")
# the shape the boot-time prewarm builds for: the delivered, always-built 128x128
WPACK_PREWARM_BOOT_SHAPE = (128, 128)
# The artifact is 120 x W_BYTES; the SIZE is validated by bridge.py against the
# driver (this constant only names the file).  Bump WPACK_FORMAT together with
# bridge.WPACK_VERSION: a mismatch is refused there and repacked.
WPACK_LAYERS = 30
WPACK_FORMAT = 2
WPACK_DIR = CACHE_DIR / "wpack"
WPACK_FILE = WPACK_DIR / ("wpack_l%d_v%d.bin" % (WPACK_LAYERS, WPACK_FORMAT))
# 0 = trust the manifest (version/code/layers/W size/weights stamp); 1 = also
# hash the 1.85 GiB (measured 1.0-1.5 s on this host).  On by default: it is the
# difference between "the manifest says so" and "the bytes are those bytes".
WPACK_VERIFY = os.environ.get("ZIMAGE_WPACK_VERIFY", "1") not in (
    "0", "false", "False", "off", "no", "")

# ---------------------------------------------------------------------------
# T22 qkv (campaign t22_qkv_prod) -- **DEFAULT ON since 2026-10-20**.
#
# The fused q/k/v projection image runs ONE dispatch per 96-row block instead
# of three projection launches.  Owner decision 2026-10-20, taken on the gate
# table in imaggers/block/qkv/QKV-PROD.md (commits 9e829a7 / 94501c6):
#   * latents byte-identical to the delivered path (array sha 1275a93a, file
#     sha 9f86aee9) and the PNG byte-identical (e7aef10c) -- the image gate;
#   * run-to-run identical (two processes), audit 15/15 on the on-disk image,
#     model gate 30/30 layers identical under --host-ref;
#   * device -1.79..-1.95 s per 128^2 image (-11 %), steady wall -0.195 s/step
#     in the fast state and -0.491 s/step in the slow state.
#
# ROLLBACK WITHOUT CODE: T22_QKV=0 in the server environment, or the page's
# "关闭(交付路径)", or the driver's --no-qkv.  Anything not in the off-list is
# read as on, so T22_QKV=1 also works explicitly.
T22_QKV_DEFAULT = os.environ.get("T22_QKV", "1") not in (
    "0", "false", "False", "off", "no", "")
# The host-side X-pack saving (--xpack-once) stays OPT-IN: it is worth ~0.3 s
# per image and changes nothing on the device, so it has no ruling of its own.
T22_XPACK_DEFAULT = os.environ.get("T22_XPACK_ONCE", "0") not in (
    "0", "false", "False", "off", "no", "")

# ---------------------------------------------------------------------------
# T22 p10 (campaign t22_p15prod) -- **DEFAULT ON since 2026-10-20**, owner
# decision taken on imaggers/attn_impl/proj/PROD-REPORT.md (commits 9ef65ee /
# c62735c / d2ed348 / 9bdfc05).
#
# The o projection -- the ONE dispatch still on the delivered 5-stream t7_p5
# once qkv is on -- runs on its own multi-stream image (p10: 10 W streams +
# X multicast, same three declared data args) as a separate launch.  The gate
# table it was flipped on:
#   * latents byte-identical, the product argv A vs B (0/4096 B, sha dd53e57e)
#     and the same sha as t22_proj's own A/B arms; run-to-run identical in
#     BOTH states; complete job (bridge + VAE) latents 9f86aee9 + PNG 104ff564
#     for all four runs;
#   * audit 10/10 on the shipped image (0 ACQ, 0 ARG), 960 dispatches / 0
#     retries, t3d_baseline --check unchanged;
#   * device -0.41..-0.45 s per 128^2 image (o bucket: 240 dispatches,
#     6.01-6.18 -> 4.22-4.35 ms each; drift corrected -0.434 s), i.e. -27..-31 %
#     of that bucket;
#   * orthogonal to qkv: with --no-qkv the same byte identity holds at 1440
#     dispatches (the two switches replace different dispatches).
#
# NOT claimed (recorded negative results, PROD-REPORT.md 9): the wall-clock
# delta (one window's run-to-run span swings +-4-7 s for a 0.43 s effect) and
# the runner's CLOSURE gate (its 15 buckets do not own the bridge import or the
# py312 VAE spawn; this flag does not change them either way).
#
# ROLLBACK WITHOUT CODE: T22_P10=0 in the server environment, or the page's
# "关闭(交付路径)", or the driver's --no-p10.  Anything not in the off-list is
# read as on, so T22_P10=1 also works explicitly.
T22_P10_DEFAULT = os.environ.get("T22_P10", "1") not in (
    "0", "false", "False", "off", "no", "")

# The three LEGACY fixed profiles.  They are what /api/status.resolutions
# reports, and 512's "reason" is the product's reference wording for an
# unbuilt size (the front end quotes it as SHAPE_512_REASON, and
# shapes.reason_text() reuses it verbatim through _legacy_reason()).
# T22 prodlarge (2026-10-20) left them as delivered; t22_factcards2 then
# corrected ONE number inside 512's sentence:
#   * every other size -- 64..2048 on each side, 16 px steps, M up to 16416 --
#     is resolved on demand by webui/shapes.py;
#   * 512's sentence used to read "L1 预算 83,456 B", which is the L1 of
#     SEQ=416 (256x384); M=1056 at Mq=16 is 196,096 B.  The number is now
#     corrected (verdict "> 64 KB, unusable" unchanged) so this table and the
#     on-demand path (shapes.reason_text -> t22_cur_ip_l1) agree; the precise
#     figure is also on the shape plan as budget_t22.t22_cur_ip_l1_bytes;
#   * 256's reason is historical (M=288 IS shipped as 256x256); it is only
#     reachable if that profile were ever un-built again.
RESOLUTIONS = {
    128: {
        "label": "128 x 128 (native, M=96)",
        "short": "128",
        "pixels": 128,
        "latent": 16,
        "image_tokens": 64,
        "seq": 96,
        "m": 96,
        "ready": True,
        "reason": "",
    },
    256: {
        "label": "256 x 256 (needs rebuilt images)",
        "short": "256",
        "pixels": 256,
        "latent": 32,
        "image_tokens": 256,
        "seq": 288,
        "m": 288,
        "ready": False,
        "reason": ("M=288 requires rebuilding the FFN, 4 projection and IP "
                   "images plus new permutations and host constants; this "
                   "delivery only ships M=96. Not half-built: see "
                   "README-WEBUI.md limits."),
    },
    512: {
        "label": "512 x 512 (needs rebuilt images)",
        "short": "512",
        "pixels": 512,
        "latent": 64,
        "image_tokens": 1024,
        "seq": 1056,          # 1024 image tokens + 32 caption (T16: was 1152, arithmetic was wrong)
        "m": 1056,
        "ready": False,
        "reason": ("512² 需新内核设计，暂不可用：M=1056（1024 image tokens + "
                   "32 caption）远超已编译的 M=96 镜像；ip 需按 key 分块 + 在线 "
                   "softmax 的新内核（M=1056 的 L1 预算 196,096 B > 64 KB），"
                   "本树未构建，不可用。"),
    },
}

DEFAULT_PARAMS = {
    "prompt": "a mountain lake at sunrise",
    "negative_prompt": "",
    "resolution": 128,
    "height": 128,
    "width": 128,
    "batch": 1,
    "steps": 8,
    "seed": 0,
    "vae": True,
    "output_dir": "webui/outputs",
    "prefix": "zimage",
    "mode": "trial",       # trial | gate
    "bo_resident": False,
}

PREFIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
MAX_BATCH = 16          # T16: the 4-image cap is gone; the driver already takes --batch N
MAX_STEPS = 16
MAX_PROMPT_CHARS = 4000
CAP_TOKENS = 32
VERSION = "1.0.0"


class WebUIError(RuntimeError):
    """A user-visible failure with an optional raw log tail and hint."""

    def __init__(self, message, raw="", hint=""):
        super().__init__(message)
        self.raw = raw or ""
        self.hint = hint or ""


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _newest(paths):
    paths = [p for p in paths if p and pathlib.Path(p).is_file()]
    if not paths:
        return None
    return pathlib.Path(max(paths, key=lambda p: pathlib.Path(p).stat().st_mtime))


# T21 2b-a: find_te_gguf() is on the /api/status path (the UI polls it every
# 5 s) and used to re-stat every candidate, and when no fixed candidate exists,
# re-glob the whole %TEMP% parent.  The result is memoized and revalidated with
# stats only:
#   * the resolved file and every candidate that was considered are watched by
#     (mtime_ns, size) -- a changed or deleted GGUF invalidates;
#   * every directory that was searched is watched by its own mtime, so a file
#     created or deleted in it (including webui/models) invalidates on the next
#     call without re-running the glob;
#   * ZIMAGE_TE_GGUF is re-read every call (it is one is_file() stat and it must
#     win immediately);
#   * a 30 s TTL is the safety net for the one case the directory stamps cannot
#     see: a new zimage-te/ subdirectory created inside an already-existing
#     dsh-* directory that was never watched.
_TE_MEMO = {"valid": False, "path": None, "files": (), "files_stamp": (),
            "dirs": (), "dirs_stamp": (), "t": 0.0, "hits": 0, "misses": 0}
_TE_MEMO_LOCK = threading.Lock()
_TE_MEMO_TTL = 30.0


def _stat_stamp(paths):
    """(path, mtime_ns, size) for each path; None marks a missing path."""
    out = []
    for p in paths:
        try:
            st = os.stat(p)
        except OSError:
            out.append((str(p), None, None))
        else:
            out.append((str(p), st.st_mtime_ns, st.st_size))
    return tuple(out)


def _te_search():
    """One full search.  Returns (result, files_watched, dirs_watched)."""
    fixed = [
        WEBUI / "models" / "zimage-te-qwen3-4b-bf16.gguf",
        pathlib.Path(tempfile.gettempdir()) / "zimage-te" / "zimage-te-qwen3-4b-bf16.gguf",
        pathlib.Path(r"C:\Users\15081\AppData\Local\Temp\dsh-nPUWK7\zimage-te\zimage-te-qwen3-4b-bf16.gguf"),
    ]
    files = list(fixed)
    dirs = [p.parent for p in fixed]
    hit = _newest(fixed)
    if hit:
        return hit, files, dirs
    parent = pathlib.Path(tempfile.gettempdir()).parent
    hits = []
    dirs.append(parent)
    for pat in ("dsh-*/zimage-te/*.gguf", "zimage-te/*.gguf"):
        try:
            hits.extend(glob.glob(str(parent / pat)))
        except OSError:
            pass
    files.extend(pathlib.Path(h) for h in hits)
    dirs.extend(pathlib.Path(h).parent for h in hits)
    return _newest(hits), files, dirs


def find_te_gguf(refresh=False):
    """Locate the 8.05 GB Qwen3-4B text-encoder GGUF built for this machine.

    The blob is deliberately outside the git tree (third-party weights).  It
    normally lives in the session temp area; ZIMAGE_TE_GGUF overrides.  The
    search is best-effort: the UI shows a clear "not found" state instead of
    silently producing a null-caption image.  The search result is memoized and
    stat-revalidated (see _TE_MEMO); pass refresh=True to force a full search.
    """
    env = os.environ.get("ZIMAGE_TE_GGUF")
    if env and pathlib.Path(env).is_file():
        return pathlib.Path(env)
    now = time.monotonic()
    memo = _TE_MEMO
    if not refresh:
        with _TE_MEMO_LOCK:
            if (memo["valid"] and _stat_stamp(memo["files"]) == memo["files_stamp"]
                    and _stat_stamp(memo["dirs"]) == memo["dirs_stamp"]
                    and (now - memo["t"]) < _TE_MEMO_TTL):
                memo["hits"] += 1
                return memo["path"]
    path, files, dirs = _te_search()
    with _TE_MEMO_LOCK:
        memo.update({"valid": True, "path": path, "files": tuple(files),
                     "files_stamp": _stat_stamp(files), "dirs": tuple(dirs),
                     "dirs_stamp": _stat_stamp(dirs), "t": now})
        memo["misses"] += 1
    return path


def find_te_gguf_stats():
    """Memo counters for /api/status (diagnostics; T21 2b-a)."""
    with _TE_MEMO_LOCK:
        return {"hits": _TE_MEMO["hits"], "misses": _TE_MEMO["misses"],
                "scanned_files": len(_TE_MEMO["files"]),
                "scanned_dirs": len(_TE_MEMO["dirs"])}


def model_present():
    return (PORT / "weights.bfp16").is_file() and (MODELS / "text_encoder" / "config.json").is_file()


def subprocess_env(kind="base"):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if kind == "device":
        env["PEANO_INSTALL_DIR"] = str(PEANO)
        env["PATH"] = str(XRT_DIR) + os.pathsep + env.get("PATH", "")
        env["T3_TIMEOUT_MS"] = env.get("T3_TIMEOUT_MS", "300000")
        nd = CACHE_DIR / "numba"
        nd.mkdir(parents=True, exist_ok=True)
        env["NUMBA_CACHE_DIR"] = str(nd)
    return env


def sanitize_prefix(value, default="zimage"):
    value = (value or "").strip()
    if not value:
        return default
    if not PREFIX_RE.match(value):
        raise ValueError(
            "prefix must start with a letter/digit and contain only "
            "letters, digits, '.', '_' or '-' (<=64 chars)")
    return value


def resolve_output_dir(value, create=True):
    value = (value or "").strip() or "webui/outputs"
    p = pathlib.Path(value)
    if not p.is_absolute():
        p = NPU / p
    p = p.resolve()
    try:
        p.relative_to(NPU)
    except ValueError:
        raise ValueError(f"output_dir must stay inside the project tree: {NPU}")
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


MEMORY_FAST_THRESHOLD_GB = 9.0
# ---------------------------------------------------------------------------
# HOST MEMORY BUDGET -- what a shape needs BEFORE anyone presses go
#
# rq_1080p_8d/8e/8f lost eight attempts to host memory and never named the
# number that would have refused them.  This is that number.  It is a PEAK,
# measured on the whole process tree (py312 server + py310 bridge + workers) by
# imaggers/par/membudget/peakwatch.py, not the sum of the allocations in a
# traceback: at 512^2/8 steps the tree peaks at 10.7 GB while the largest single
# allocation in it is 1.24 GiB.
#
# The two modes are not "the same job with a flag":
#   fast  --cache-all keeps the BG/B3 payload resident and, unless the B-repack
#         cache is on, needs another 1.24 GiB for the dual B3 order.  It is
#         ~2.5x the loop speed (measured 89.8 s vs 229.3 s at 512^2 x8) and it
#         is the mode that dies when it does not fit.
#   low   --no-cache-all re-derives each layer's payloads per dispatch: the same
#         bytes, ~2.4x the wall, and about a third of the footprint.
#
# required_gb = MEASURED PEAK + 15 % margin, rounded up.  "User checkable"
# means: Task Manager > Performance > Committed, subtract committed from the
# limit, compare with the number /api/capability?memory=1 prints.
# ---------------------------------------------------------------------------
MEMORY_MEASURED = {
    # m  : (peak tree in low mode, peak tree in fast mode, source)
    96:   (2.02, 4.55, "128x128 x8, delivered path"),
    1056: (3.90, 10.70, "512x512 x8 measured 2026-11-04 by peakwatch: device pid 9.51 GB private, job tree 10.70 GB with the driver excluded; the low number is an ESTIMATE, not sampled"),
    1632: (4.30, 11.30, "480x848 prototype + the measured 512^2 growth"),
    4128: (5.20, 12.60, "1024x1024 recorded job + the measured slope"),
    8192: (14.80, 14.80, "MEASURED FLOOR 2026-11-04: in --no-cache-all the device peaked at 14.71 GB private and then died asking for 120.00 MiB (M=8192 x 3840 f32) at install.py:861; fast was refused by this same budget at 14.40 GB of headroom, 0.43 GB short.  1080p does not fit this box in EITHER mode"),
}
BUDGET_MARGIN = 1.15
def memory_budget(m, mode="fast"):
    """Required host commit headroom for one job at this M and mode.
    Returns {"m", "mode", "required_gb", "peak_gb", "basis"}.  m is the run
    shape's padded token count (params["m"]), not the requested pixels: 1080p
    is 1088x1920 -> M=8192.
    """
    mode = "low" if str(mode).lower() == "low" else "fast"
    m = int(m or 0)
    idx = 0 if mode == "low" else 1
    keys = sorted(MEMORY_MEASURED)
    if m <= keys[0]:
        peak = MEMORY_MEASURED[keys[0]][idx]
        basis = MEMORY_MEASURED[keys[0]][2]
    elif m >= keys[-1]:
        # extrapolate with the last measured slope, never with a cliff
        k0, k1 = keys[-2], keys[-1]
        p0, p1 = MEMORY_MEASURED[k0][idx], MEMORY_MEASURED[k1][idx]
        peak = p1 + (p1 - p0) * (m - k1) / float(k1 - k0)
        basis = "%s (extrapolated from M=%d..%d)" % (
            MEMORY_MEASURED[k1][2], k0, k1)
    else:
        lo = max(k for k in keys if k <= m)
        hi = min(k for k in keys if k >= m)
        if hi == lo:
            peak, basis = MEMORY_MEASURED[lo][idx], MEMORY_MEASURED[lo][2]
        else:
            p0, p1 = MEMORY_MEASURED[lo][idx], MEMORY_MEASURED[hi][idx]
            peak = p0 + (p1 - p0) * (m - lo) / float(hi - lo)
            basis = "interpolated M=%d..%d" % (lo, hi)
    return {"m": m, "mode": mode, "required_gb": round(peak * BUDGET_MARGIN, 2),
            "peak_gb": round(peak, 2), "margin": BUDGET_MARGIN, "basis": basis}
def memory_verdict(m, mode="fast", snap=None):
    """Can this box run this shape in this mode?  A dict the UI can print.
    The comparison is against COMMIT headroom (ullAvailPageFile), because that
    is the counter that ran out (57.8 of 59.6 GB committed) and the one a
    MemoryError comes from.  Physical RAM alone under-reports: the pagefile is
    real until it is full.
    """
    snap = snap or memory_snapshot()
    b = memory_budget(m, mode)
    head = float(snap.get("commit_gb") or 0.0)
    short = round(max(0.0, b["required_gb"] - head), 2)
    ok = short <= 0.0
    low_gb = memory_budget(b["m"], "low")["required_gb"]
    if ok:
        msg = ("%s at M=%d needs %.2f GB of commit headroom (measured peak %.2f "
               "GB + %.0f %% margin); this box has %.2f GB"
               % (b["mode"], b["m"], b["required_gb"], b["peak_gb"],
                  (BUDGET_MARGIN - 1.0) * 100.0, head))
    else:
        msg = ("REFUSED: %s at M=%d needs %.2f GB of commit headroom (measured "
               "peak %.2f GB + %.0f %% margin) and this box has %.2f GB -- %.2f "
               "GB short.  Close that much, or run mode=low (%.2f GB, about "
               "2.4x the wall)."
               % (b["mode"], b["m"], b["required_gb"], b["peak_gb"],
                  (BUDGET_MARGIN - 1.0) * 100.0, head, short, low_gb))
    return {"ok": ok, "m": b["m"], "mode": b["mode"],
            "required_gb": b["required_gb"], "peak_gb": b["peak_gb"],
            "headroom_gb": round(head, 2), "short_gb": short,
            "low_required_gb": low_gb, "basis": b["basis"], "message": msg}
def memory_table():
    """Every measured row, for /api/capability?memory=1."""
    snap = memory_snapshot()
    out = []
    for m in sorted(MEMORY_MEASURED):
        row = {"m": m, "source": MEMORY_MEASURED[m][2],
               "peak_low_gb": MEMORY_MEASURED[m][0],
               "peak_fast_gb": MEMORY_MEASURED[m][1]}
        for mode in ("low", "fast"):
            v = memory_verdict(m, mode, snap)
            row[mode] = {"required_gb": v["required_gb"], "ok": v["ok"]}
        out.append(row)
    return {"headroom_gb": round(float(snap.get("commit_gb") or 0.0), 2),
            "margin": BUDGET_MARGIN, "rows": out}


def memory_snapshot():
    """Available physical RAM and commit (pagefile) headroom in GB.

    The full-memory path needs ~6 GB of private commit (1.85 GB W images +
    3.71 GB BG/B3 cache).  On this host the DSH session can run with less than
    that available; the WebUI then uses t13_run's --no-cache-all path and says
    so in the job metadata rather than dying with MemoryError.
    """
    import ctypes
    from ctypes import wintypes

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    m = MS()
    m.dwLength = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return {"available_gb": round(m.ullAvailPhys / 2**30, 2),
            "commit_gb": round(m.ullAvailPageFile / 2**30, 2),
            "total_gb": round(m.ullTotalPhys / 2**30, 2),
            "load_pct": int(m.dwMemoryLoad)}


def choose_memory_mode(requested="auto", m=None):
    """Pick fast or low.  With m, the choice is the BUDGET's, not a constant.

    The delivered rule was "commit >= 9.0 GB -> fast", which is why the 1080p
    run started in fast mode with 12.8 GB of headroom and died at step 6: at
    M=8192 fast needs ~14.7 GB and 9.0 is not a threshold for that shape.
    m=None reproduces the delivered behaviour EXACTLY, so every one-argument
    caller keeps the code it had.
    """
    requested = (requested or "auto").lower()
    snap = memory_snapshot()
    if requested in ("fast", "low"):
        return requested, snap, f"user selected {requested}"
    if m is None:
        if snap["commit_gb"] >= MEMORY_FAST_THRESHOLD_GB:
            return "fast", snap, (f"auto: commit {snap['commit_gb']:.1f} GB >= "
                                   f"{MEMORY_FAST_THRESHOLD_GB:.1f} GB")
        return "low", snap, (f"auto: commit {snap['commit_gb']:.1f} GB < "
                              f"{MEMORY_FAST_THRESHOLD_GB:.1f} GB; "
                              f"using --no-cache-all to avoid MemoryError")
    v = memory_verdict(m, "fast", snap)
    if v["ok"]:
        return "fast", snap, ("auto: M=%d needs %.2f GB fast and %.2f GB is "
                              "available" % (int(m), v["required_gb"],
                                             v["headroom_gb"]))
    return "low", snap, ("auto: M=%d needs %.2f GB fast but only %.2f GB is "
                          "available; --no-cache-all needs %.2f GB"
                          % (int(m), v["required_gb"], v["headroom_gb"],
                             v["low_required_gb"]))


def rel_to_npu(path):
    try:
        return str(pathlib.Path(path).resolve().relative_to(NPU)).replace("\\", "/")
    except Exception:
        return str(path)


def format_bytes(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024


# >>> provenance record (installed by imaggers/par/provenance/install.py)
# The complete knob set with (value, explicit, source) for EVERY environment
# variable the runtime reads.  The count is env_provenance()["counts"]["total"],
# never a literal -- env_audit.py derives the set from the runtime import closure
# and it grows every time the audit finds a new knob (171 when this block was
# first written, 177 knobs / 108 arm knobs at the 2026-11-04 install).  The text
# here used to read "171 ... 68-file closure" and was wrong within a day: a stale
# count inside the record that records everything is the same class of bug as a
# stale arm.  resolved_flags() is rebound to the superset, so the existing
# job.json:env_resolved line becomes complete with no other edit.
try:
    from .provenance import flags as provenance_flags
    from .provenance import record as env_provenance
except (ImportError, ValueError):          # loaded flat by tooling (by file path)
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from provenance import flags as provenance_flags
    from provenance import record as env_provenance
resolved_flags = provenance_flags
# <<< provenance record
