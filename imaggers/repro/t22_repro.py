# SPDX-License-Identifier: MIT
"""t22_repro.py -- the ONE entry point for a reproducible T22 device run.

Why this file exists: a fresh agent could not reproduce a previously successful
run from the prose that documented it.  Every precondition that is SILENT when
wrong is encoded here as a constant, a preflight check or a postflight
assertion, so the working path is structural instead of documentary.

    py312 imaggers/repro/t22_repro.py run --size 512 --steps 8 --tag t1
    py312 imaggers/repro/t22_repro.py preflight --size 512 --steps 8 --tag t1
    py312 imaggers/repro/t22_repro.py postflight --tag t1
    py312 imaggers/repro/t22_repro.py doc

Or, from PowerShell, the thin wrapper (same three arguments):

    imaggers\repro\t22_run.ps1 -Size 512 -Steps 8 -Tag t1

Nothing here changes product behaviour: no file under webui/ or imaggers/image/
is edited, every switch is a value the product already reads.

The three rules this file enforces:
  1. The job's environment is EXACTLY RECIPE.  Anything else that selects a
     device path is scrubbed, so an inherited shell cannot poison a run.
  2. The run does not start unless the preflight proves the device is idle,
     the size is renderable, the memory mode is 'fast' and the images exist.
  3. The run is not believed unless the postflight finds the *evidence lines*
     of the path that was supposed to run -- installed AND called.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import socket
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# 0. identity / paths
# ---------------------------------------------------------------------------
NAME = "t22_repro"
ROOT = pathlib.Path(__file__).resolve().parents[2]      # zimage-bfp16-npu/
HERE = pathlib.Path(__file__).resolve().parent          # imaggers/repro/
RECORDS = HERE / "last"
REFERENCE_FILE = HERE / "reference.json"
RUNBOOK = HERE / "RUNBOOK.md"

# the product modules are imported by absolute path, so the working directory
# does not matter (a fresh agent may be anywhere)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# the two interpreters the product itself hard-codes (webui/config.py:22-23)
# Third-party locations: resolved, never hard-coded (imaggers/paths.py).
_root = pathlib.Path(__file__).resolve().parent
while not (_root / 'imaggers' / 'paths.py').is_file():
    if _root.parent == _root:
        raise RuntimeError('cannot locate the checkout root')
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from imaggers import paths as _paths  # noqa: E402

# soft: a harness must import on a machine that has not been set up yet, and say
# what is missing in the preflight rather than dying on line 70.  The spawn sites
# below call _need() so the message still names the variable.
PY312 = _paths.soft(_paths.python312)
PY310 = _paths.soft(_paths.python310)
XRT_DIR = _paths.soft(_paths.xrt_dir)


def _need(value, what: str, env: str) -> pathlib.Path:
    if value is None:
        raise SystemExit(
            "%s not found.  Set %s, or put it under third_party/ (README 10.6)."
            % (what, env))
    return value

PORT = 7874
URL = "http://127.0.0.1:%d" % PORT
LOCK = ROOT / "imaggers" / "device.lock"
LAUNCHER = ROOT / "imaggers" / "bigpipe2" / "launch_webui_bigm.py"
JOBS = ROOT / "webui" / "jobs"
T18 = ROOT / "imaggers" / "image" / "t18_run.py"

# the one prompt/seed the reference arms were measured with
# (imaggers/bigpipe2/run_job_py.py:55, imaggers/devkill/run_arm*.ps1:25)
PROMPT = "a mountain lake at sunrise"
SEED = 7
LAYERS = 30                      # t18_run: blocks per step (webui/shapes.py:166)

# how long to wait for somebody else's device process before giving up
DEVICE_WAIT_S = 1800
QUIET_S = 10
# commit the fast path needs: 1.85 GB W images + 3.71 GB BG/B3 cache, ~6 GB
# (webui/config.py:515-522).  The counter is only a proxy; the probe allocates.
PROBE_GB = 6.0
PROBE_CHUNK_MB = 256
MEM_WAIT_S = 180      # give a finishing job a moment to release commit
# 0xc01e0009 = 'Failed to create context virtual': somebody else's process took
# the NPU between this run's preflight and its device stage.  The lock cannot
# prevent it (their launcher never checks it), so the tool retries on ITS OWN
# collision message and never on anything else.
COLLISION_TOKEN = "0xc01e0009"
RETRY_ON_COLLISION = 2

# A/B escape hatches.  Both are OFF by default and both are RECORDED in the
# preflight json, because a run that used either one is an ARM, not a reference:
# the recipe's env_id no longer describes it.
#   --extra NAME=VALUE   sets one variable AFTER the recipe is applied (and
#                        after the scrub), so an opt-in shipped switch such as
#                        T22_BREPACK_CACHE can be measured without hand-exporting
#                        an environment block.  It prints and it is stored.
#   --skip P04           drops ONE preflight check by id.  Its only legitimate
#                        use is a within-pair A/B of what the check itself does
#                        to the machine (P04 commits and touches 6 GiB before
#                        every run, which is exactly the page cache the job is
#                        about to read).
EXTRAS = []
SKIP_CHECKS = []

# ---------------------------------------------------------------------------
# 1. THE CONSTANTS BLOCK -- every required variable, one value, one reason.
#
#    RECIPE   : set to this value, always.
#    FORBID    : deleted before the child starts (a leftover switch is a
#                different device path, and the run would be a silent A/B mix).
#
#    Sources are file:line of the code that READS the variable, so this block
#    can be audited against the tree.  The generator emits it into RUNBOOK.md.
# ---------------------------------------------------------------------------
RECIPE = [
    # -- the big-M / devkill device path (the one the 94 s 512^2 arm used) ----
    ("ZIMAGE_BIGM", "1",
     "registers the big-M image set and the whole-shape projection",
     "imaggers/bigpipe2/launch_webui_bigm.py:66, imaggers/image/t18_run.py:317"),
    ("ZIMAGE_BIGM_PROJ", "1",
     "makes the projection a whole-shape big-M dispatch instead of 4 chunked passes",
     "webui/chunked.py:963-964"),
    ("ZIMAGE_CHUNK_PIPE", "0",
     "keeps the delivered block_step; with 1 the 96-row pipeline replaces it and "
     "T22_PLAN_FFN would refuse",
     "webui/chunked.py:942-958"),
    ("T22_HOSTKILL", "1",
     "DMA-side A pack / C un-tile projection image (the plain-X path)",
     "webui/chunked.py:975, imaggers/image/t18_run.py:358"),
    ("T22_DEVKILL", "1",
     "G=15 plain projection + RT=11 FFN pair (the 94 s FFN)",
     "webui/chunked.py:982, imaggers/image/t18_run.py:364"),
    # -- the attention segment (ATTN-SEG-REPORT.md). Variant D is REJECTED on
    #    quality (corr 0.9919453 < 0.9953); the accepted arm is A2 + REUSE +
    #    NBTHREADS 20 + QB 1056, which is exact to f32 rounding on the attention
    #    output and measured loop 82.896 -> 79.210 s at latents corr 0.9975444.
    #    All four default to the SHIPPED value (off) so an arm must opt in with
    #    -Extra; they are listed here so -Extra can actually reach the product
    #    (RECIPE_ENV is the only set of names the wrapper will pass through).
    ("T22_ATTN_A2", "0",
     "attention variant A2 (query-blocked, torch path)",
     "webui/chunked.py:477"),
    ("T22_ATTN_REUSE", "0",
     "reuse both matmul out= buffers and the result buffer; bit-exact",
     "webui/chunked.py:491"),
    ("T22_ATTN_NBTHREADS", "0",
     "numba thread count for the fused softmax; the accepted arm used 20",
     "webui/chunked.py:513"),
    ("T22_ATTN_QB", "0",
     "override the attention query-block size; the accepted arm used 1056",
     "webui/chunked.py:453"),
    ("T22_BIGM_DEVHANDOFF", "1",
     "REQUIRED by devkill_ffn: it demands the k=32 B payload, and without this "
     "prod_loop_bigm returns only {n128,n64} -> RuntimeError 'devkill needs n32 "
     "B payload' (an hour lost to this one)",
     "imaggers/block/bigm/prod_loop_bigm.py:647-649, "
     "imaggers/devkill/devkill_ffn.py:325-332"),
    ("T22_BIGM_DEBUG", "1",
     "per-layer [fastglue]/[plain15]/[devkill] timing lines -- THE evidence the "
     "postflight reads; the launcher forces it too",
     "imaggers/util/glue/install.py:536, imaggers/devkill/prod_devkill.py:111, "
     "imaggers/bigpipe2/launch_webui_bigm.py:68"),
    ("T22_DEVKILL_G", "15",
     "the G=15 projection image the devkill path registers",
     "imaggers/devkill/prod_devkill.py:38"),
    ("T22_DEVKILL_PREWARM", "1",
     "packs the 120 G=15 W images at startup instead of inside step 0 "
     "(+0.17 s/layer otherwise)",
     "webui/chunked.py:999"),
    ("T22_DEVKILL_BREPACK", "1",
     "repacks the B payload order at startup, outside the loop's wall",
     "imaggers/image/t18_run.py:2699-2700"),
    ("T22_BREPACK_CACHE", "1",
     "MAPS the 5.31 GB pre-permuted B payloads from a content-addressed file "
     "instead of re-deriving the permutation every run.  It is the largest "
     "single row of startup_s (27.02 s of 45.64 s on AC = 59%) and the cache "
     "was measured at -29.51 s of total (startup 42.45 -> 13.72 s) with "
     "BIT-IDENTICAL latents (2abf91331200d069, corr 1.000000000, maxdiff 0).  "
     "A cache that silently MISSES is indistinguishable from no cache, so "
     "postflight Q18 proves the run mapped it",
     "imaggers/block/bigm/prod_loop_bigm.py:613-635, "
     "imaggers/block/bigm/brepack_cache.py:675-772, "
     "imaggers/image/t18_run.py:2269,2681-2697"),
    # -- the fastglue host-glue hooks (TWO switches, they are NOT aliases) ----
    ("T22_FASTGLUE", "1",
     "the switch imaggers/util/glue/install.py checks.  WITHOUT THIS the 8 "
     "fused hooks do not install, the log says 'installed 0/8', the loop is "
     "~45% slower AND STILL 'WORKS'.  This is not T22_FAST_GLUE.",
     "imaggers/util/glue/install.py:591"),
    ("T22_FAST_GLUE", "1",
     "t18_run's own rms/RoPE in the drained layout -- the OTHER switch; the "
     "reference arm sets both",
     "imaggers/image/t18_run.py:196"),
    ("T22_GLUE2", "1",
     "copy-free drain/pack on every dispatch",
     "imaggers/image/t18_run.py:205"),
    ("T22_GLUE3", "1",
     "absmax without the 3 x 16 MB temporaries",
     "imaggers/image/t18_run.py:208"),
    ("T22_FASTGLUE_THREADS", "8",
     "numba threads for the fused kernels; 8 is what every >100 s 512^2 arm "
     "ran and what the log line 'threads=8' records",
     "imaggers/util/glue/install.py:597"),
    ("T22_FASTGLUE_STRICT", "0",
     "0 keeps the shipped fail-open install (a gate failure still INSTALLS the "
     "rest).  The postflight is the enforcement, not this flag -- see RUNBOOK "
     "'silent fallbacks'",
     "imaggers/util/glue/install.py:611-624"),
    ("T22_PLAN_FUSE_GU", "0",
     "RECORDED, not tuned: the FUSED gu image (silu(gate)*up inside the "
     "dispatch, ONE gu launch per layer) is a COMPILE-TIME knob of the plan "
     "arm -- it changes the plan_key, the image and the dispatch count -- and "
     "until now the only record of which arm ran was plan_ffn's log line, so "
     "the fused and unfused arms shared ONE env_id and a future A/B could not "
     "prove what it measured (same gap as T22_BREPACK_CACHE, see this file's "
     "doc section).  0 is the shipped value: the fused arm FAILS the "
     "correctness gate -- 0.995239938 vs the 0.9953 threshold, and worse than "
     "the unfused arm's 0.995965046 (T22-PLAN-FUSE-GU-ARM.md 3, 5).  Inert "
     "without T22_PLAN_FFN, which is on the FORBID list.",
     "imaggers/util/place/plan_ffn.py:94 (read), plan_ffn.py:40-43 (doc), "
     "webui/config.py:131-140 (recorded)"),
    ("T22_REF_SMNB", "1",
     "the refiner's fused in-place numba softmax.  It is NOT byte-neutral: "
     "without it the 512^2 8-step latents are 6b903914... instead of the "
     "2abf91331200... the reference arms produced (measured 2026-11-03, same "
     "proved path).  Every arm of the sweep that defined the reference sets it",
     "imaggers/devkill/run_arm_size.ps1:47, HANDOVER-2026-11-01.md:10,21,80, "
     "SWEEP-FIXED-VS-TOKEN.md:54"),
    ("T22_FFN_PROF", "1",
     "one [ffnprof] line per FFN call: the ONLY in-log proof that the ffnchain "
     "hook was actually CALLED, not merely installed (cost: a print)",
     "imaggers/util/glue/install.py:761,786-803"),
    # -- threads: the loop and the VAE must not be coupled implicitly ---------
    ("ZIMAGE_CHUNK_THREADS", "16",
     "the device loop's host-attention pool (gated at -13.8%, latents "
     "byte-identical).  Left at 0 it is torch's default 10",
     "webui/config.py:143-145, webui/chunked.py:100-105"),
    ("ZIMAGE_VAE_THREADS", "10",
     "PINNED.  config.vae_threads() follows ZIMAGE_CHUNK_THREADS when unset, so "
     "the loop's 16 would silently move the decode to 16 threads and change 4-7 "
     "pixels of 786,432.  Pinning 10 keeps the shipped byte path AND makes the "
     "count provable ([vae] decode threads: torch intra-op 10)",
     "webui/config.py:148-163, webui/vae_local.py:77-92"),
    # -- product defaults, pinned only so the record proves they were on ------
    ("ZIMAGE_VAE_NPU", "0", "CPU decoder (the NPU chain is 61.9 s at 128^2 vs 1 s)",
     "webui/config.py:60"),
    ("ZIMAGE_VAE_FAST", "1", "webui/vae_lean.py instead of the diffusers loader "
     "(byte-identical PNG, +15-18 s otherwise)", "webui/vae_local.py:56"),
    ("ZIMAGE_VAE_LAUNCH", "srv", "the delivered single device server for the VAE "
     "NPU chain (inert with ZIMAGE_VAE_NPU=0)", "webui/config.py:73"),
    ("ZIMAGE_DEV_WORKER", "0", "the delivered one-process bridge; the resident "
     "worker is an opt-in that dies with a VAE stage",
     "webui/config.py:176, webui/evidence/T22-WORKERVAE.md"),
    ("T22_QKV", "1", "fused q/k/v image (default on; pinned so the record says so)",
     "webui/config.py:231"),
    ("T22_P10", "1", "the o projection on its own 10-stream image (default on)",
     "webui/config.py:267"),
    ("T22_XPACK_ONCE", "0", "host X-pack reuse, opt-in, changes nothing on the device",
     "webui/config.py:235"),
    ("ZIMAGE_WPACK_PREWARM", "job", "build the 1.85 GiB wpack during the TE window",
     "webui/config.py:195-199"),
    ("ZIMAGE_WPACK_VERIFY", "1", "sha256 the 1.85 GiB wpack instead of trusting "
     "the manifest", "webui/config.py:212"),
    ("ZIMAGE_FFN_IMAGE", "fs_ffn_j5", "the shipped FFN image",
     "webui/config.py:37"),
    ("T20_FFN_BARRIER", "1",
     "keep the dummy ip launch between FFN row blocks.  WITHOUT IT a second "
     "back-to-back launch of the same xclbin never completes: ERT's ~7.0 s "
     "watchdog reports state 8, and a second kernel handle reports a FALSE "
     "completion (2-4 ms for a 132.7 MB stream)",
     "imaggers/image/t18_run.py:550, docs/KNOWN-PLATFORM-FACTS.md F-01/F-04"),
    ("ZIMAGE_DEVICE_TIMEOUT_S", "7200",
     "the device wait cap per dispatch; the product default is 1800 s",
     "webui/runner.py:1226"),
]

FORBID = [
    # name, why it must be off, source
    ("T22_PLAN_FFN",
     "dispatches the adaptive plan's gu/dn images -- SLOWER (loop 160.6 s vs "
     "119.4 s at 384^2), does not reproduce the delivered image (corr 0.35) and "
     "only 'xd' has an executor.  Not a product path",
     "webui/chunked.py:943-955, webui/capability.py:257-269"),
    ("ZIMAGE_ADAPT",
     "requires the adaptive image set to be BUILT before the job is accepted",
     "webui/shapes.py:626"),
    ("T22_ADAPT_M",
     "resizes the whole-shape host BO; a leftover value from a 1024^2 attempt "
     "changes the projection geometry",
     "imaggers/image/t18_run.py:396, webui/shapes.py:423"),
    ("T22_OVL",
     "launch-ahead overlap: a different dispatch order (measured worth 1.9 s, "
     "not 28.6 s)", "imaggers/util/glue/install.py:600"),
    ("T22_P33",
     "the 33-tile kk=32 projection: 2.28x isolated, NO-OP in the shipped config",
     "imaggers/util/glue/install.py:713, P33-E2E-REPORT.md"),
    ("T22_S2_PROJ32", "the 32-core fused projection, opt-in and unwired",
     "webui/chunked.py:1010"),
    ("T22_S2_FFN32", "opt-in FFN image", "imaggers/image/t18_run.py:225"),
    ("T22_FFN_KK32", "kk=32 FFN: +42..61 ms/layer-step in product",
     "imaggers/devkill/devkill_ffn.py:40"),
    ("T22_FINAL110", "the final110 arm changes the projection and the attention",
     "imaggers/devkill/prod_devkill.py:21"),
    ("T22_PREOVL", "pre-submits the refiner preload; only valid with cache-all",
     "imaggers/image/t18_run.py:222"),
    ("T22_PFILL", "payload prefill, opt-in", "imaggers/image/t18_run.py:236"),
    ("T22_TIMELINE", "pyxrt instrumentation: real overhead, different flags",
     "imaggers/util/glue/install.py:678"),
    ("T22_FASTGLUE_SKIP", "selectively uninstalls hooks -- an A/B tool, never a "
     "reference run", "imaggers/util/glue/install.py:610"),
    ("T22_GLUE_PROF", "the 32-bucket in-loop ledger: costs wall, changes nothing "
     "else; a reference arm runs with it OFF",
     "imaggers/image/t18_run.py:201"),
    ("ZIMAGE_VAE_OPNPU", "the residual adds on the NPU: 1-ulp ties move pixels "
     "at 144x128", "webui/config.py:82"),
    ("ZIMAGE_DEV_WORKER_KEEP_PAYLOAD", "worker-only knob",
     "webui/runner.py:51"),
    ("ZIMAGE_DEV_WORKER_KEEP_REFINER", "worker-only knob", "webui/runner.py:52"),
]

# every other env var the product/driver reads (the 2026-11-03 survey: 104
# distinct names in webui/*.py + imaggers/image/t18_run.py + the device path).
# None of these may reach the job by inheritance: they are deleted before the
# child starts, and anything left over is reported by preflight P06.
SURVEYED_ENV = """
AD_FFN_BARRIER AD_FFN_REFILL S2_DUMP_STAGES S2_ENGINE S2_FFN32_CHECK S2_STAGE
T18_CAP_X T18_DUMP T18_PAYLOAD_ORDER T18_SKIP_ATT T18_TRACE T20_FFN_BARRIER
T22_ADAPT_CHUNK T22_ADAPT_M T22_ATTN_A2 T22_ATTN_BF16 T22_ATTN_D T22_ATTN_DUMP
T22_ATTN_DUMP_N T22_ATTN_E T22_ATTN_NB T22_ATTN_NBTHREADS T22_ATTN_OBUF
T22_ATTN_PROF T22_ATTN_QB T22_ATTN_REUSE T22_BIGM_B3_DUAL T22_BIGM_DEBUG
T22_BIGM_DEVHANDOFF T22_BIGM_ORDER T22_BREPACK_CACHE T22_DEVKILL
T22_DEVKILL_BREPACK T22_DEVKILL_G T22_DEVKILL_PACK T22_DEVKILL_PREWARM
T22_DEVKILL_PROJ_IMG T22_DEVKILL_WDB T22_DUMP_FFN T22_FASTGLUE
T22_FASTGLUE_HUNKS T22_FASTGLUE_SKIP T22_FASTGLUE_STRICT T22_FASTGLUE_THREADS
T22_FAST_GLUE T22_FFN_KK32 T22_FFN_PROF T22_FFN_TP T22_FINAL110 T22_GLUE2
T22_GLUE3 T22_GLUE_PROF T22_HOSTKILL T22_HOSTPROF T22_HOSTPROF_DIR
T22_HOSTPROF_TAG T22_HOSTPROF_VERBOSE T22_OVL T22_OVL_THREADS T22_OVL_WGU
T22_P10 T22_P10_BIN T22_P10_XCLBIN T22_P33 T22_PFILL T22_PLAN_FFN T22_PREOVL
T22_QKV T22_QKV_BIN T22_QKV_XCLBIN T22_REF_SMNB T22_S2_CACHE_DIR
T22_S2_CACHE_LAYERS T22_S2_FFN32 T22_S2_FFN32_DIR T22_S2_PROJ32 T22_S2_TAGS
T22_SBG_ASSERT T22_TIMELINE T22_TIMELINE_FILE T22_XPACK_ONCE T3_TIMEOUT_MS
ZIMAGE_ADAPT ZIMAGE_BIGM ZIMAGE_BIGM_PROJ ZIMAGE_CHUNK_PIPE
ZIMAGE_CHUNK_REFINER_GB ZIMAGE_CHUNK_THREADS ZIMAGE_DEVICE_TIMEOUT_S
ZIMAGE_DEV_WORKER ZIMAGE_DEV_WORKER_KEEP_PAYLOAD ZIMAGE_DEV_WORKER_KEEP_REFINER
ZIMAGE_DEV_WORKER_RESERVE_GB ZIMAGE_FFN_IMAGE ZIMAGE_TE_CACHE_DIR
ZIMAGE_TE_CACHE_ENTRIES ZIMAGE_TE_CACHE_MB ZIMAGE_TE_CACHE_MEM_ENTRIES
ZIMAGE_TE_GGUF ZIMAGE_TE_MODEL_GB ZIMAGE_TE_MODEL_RESIDENT ZIMAGE_TE_RESIDENT
ZIMAGE_VAE_FAST ZIMAGE_VAE_LAUNCH ZIMAGE_VAE_NPU ZIMAGE_VAE_ONESHOT
ZIMAGE_VAE_OPNPU ZIMAGE_VAE_SRV_ADDR ZIMAGE_VAE_THREADS ZIMAGE_WPACK_PREWARM
ZIMAGE_WPACK_VERIFY
""".split()

# ZIMAGE_TE_GGUF is a PATH to a third-party blob, not a device switch: keep it.
SCRUB_KEEP = {"ZIMAGE_TE_GGUF"}

# the images the chunked+bigm+devkill path registers (t18_run.py:319-368,
# t18_run.py:458-459 qkv, t18_run.py:459-461 p10).  preflight P08 checks them.
NATIVE_IMAGES = [
    ("imaggers/block/qkv/out/qkv_par15m.xclbin",
     "fused q/k/v image (T22_QKV=1; only registered for a NATIVE shape: "
     "webui/runner.py:1885 forces qkv off on the chunked path)"),
    ("imaggers/attn_impl/proj/out/p10.xclbin",
     "o projection image (T22_P10=1; native shapes only, webui/runner.py:1886)"),
]

CHUNKED_IMAGES = [
    ("imaggers/block/normcore/out/nc_s3a.xclbin", "bigm norm image"),
    ("imaggers/block/bigm/out/rt17.xclbin", "big-M whole-shape projection (rt17)"),
    ("imaggers/devkill/out/plain_g15_rt11.xclbin",
     "devkill G=15 plain projection"),
    ("imaggers/devkill/out/sw_rt11.xclbin", "devkill RT=11 swiglu"),
    ("imaggers/devkill/out/dn_rt11.xclbin", "devkill RT=11 down"),
    ("imaggers/block/swiglu/out/sw_rt1.xclbin", "RT=1 swiglu (single-tile shapes)"),
    ("imaggers/block/bigm/out/ffn_dn_rt1.xclbin", "RT=1 down (single-tile shapes)"),
    ("imaggers/block/swiglu/out/dn32.xclbin", "k=32 down (the DEVHANDOFF consumer)"),
    ("imaggers/block/swiglu/out/dn32_1.xclbin", "k=32 down, single tile"),
    ("imaggers/block/bigm/out/plain_rt11.xclbin", "hostkill plain-X projection"),
    ("imaggers/block/swiglu/out/sw64d2.xclbin", "the delivered swiglu (RT=17)"),
    ("imaggers/block/bigm/out/ffn_dn_n64.xclbin", "the delivered down (RT=17)"),
    ("imaggers/block/bigm/out/sw8.xclbin", "k=8 swiglu (registered, unused)"),
    ("imaggers/block/bigm/out/dn8.xclbin", "k=8 down (registered, unused)"),
    ("imaggers/block/bigm/out/sw16.xclbin", "k=16 swiglu (registered, unused)"),
    ("imaggers/block/bigm/out/sw32.xclbin", "k=32 swiglu (registered, unused)"),
]

# ---------------------------------------------------------------------------
# 2. references: what a successful run of each (size, steps) measured.
#    Written by the postflight when absent, compared verbatim when present.
#    latents = webui/jobs/<id>/job.json:latents_sha256.  Two independent 512^2
#    8-step arms of this recipe (20261102-211411-109ea0, 20261102-213051-7c90ae)
#    produced THIS sha byte for byte; their totals differed by 10.9 s (the
#    14.5-19.4 s B repack lands inside total, outside loop).
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 2b. STARTUP_PHASES -- what startup_s is made of.
#
# Every row is a phase the CHILD bridge prints its own seconds for, except the
# last, which is what is left over.  The list exists so RUNBOOK section 13 is
# generated from it instead of drifting, and so a reader can attribute a
# startup_s without instrumenting anything: the number is already in job.log.
# `imaggers/startup/compose.py` re-derives the table from artefacts.
# ---------------------------------------------------------------------------
STARTUP_PHASES = [
    ("wpack_verify",
     "map + sha256 the 1.99 GB prewarm artifact (ZIMAGE_WPACK_VERIFY=1)",
     "[t22] prewarm: ... sha256 verified in N s (webui/bridge.py:591-650)"),
    ("devkill_prewarm",
     "pack the 120 G=15 projection W images (chunked path only)",
     "[devkill] prewarm: 120 G=15 ... in N s"),
    ("brepack_hit",
     "map the pre-permuted payload from the repack cache (T22_BREPACK_CACHE=1, "
     "in this recipe): 60 stat() + a header parse + a 5.31 GB sha256/multiset "
     "verify -- this row REPLACES brepack_pass and payload_load",
     "[brepack] HIT ... in N s"),
    ("warm_wpack",
     "pack the 30 layers' projection images from the mapped artifact",
     "[t3d] warm: 30 layers' projection images packed in N s"),
    ("payload_load",
     "read + permute the 3.71 GB BG/B3 payload (--cache-all)",
     "[t13] full-memory cache: 30 layers, 3.71 GB in N s"),
    ("brepack_pass",
     "the 5.31 GB in-place B-order permutation.  With the cache mapping it is "
     "only the join+arm (measured 0.0x s); it is the 27.02 s row when the cache "
     "is off or misses, which Q18 gates",
     "[devkill] B repack at startup: N s"),
    ("refiner_weights",
     "map the refiner blocks the first step needs (noise_refiner.0/1, "
     "2 x 0.67 GB); context_refiner.0/1 land AFTER startup_done and are not in "
     "this bucket",
     "[t3d] refiner weights loaded noise_refiner.N  N s"),
    ("bridge_jit",
     "the unprinted remainder: the child's torch/numba import, the fastglue byte "
     "gates, the 19 image registrations and the chunked projector install -- "
     "split per second in the AC ledger below",
     "no line -- startup_s minus the rows above"),
]

# MEASURED 2026-11-03 with this recipe.  The 06:30 arm's power state was never
# recorded (P33-E2E-REPORT.md section 7 only covers 08:20-10:10 battery and AC
# after 10:10), so it is labelled "not recorded" and MUST NOT be called AC.
STARTUP_MEASURED = {
    "512_0630": {"wpack_verify": 1.15, "devkill_prewarm": 2.41,
                 "brepack_hit": 3.02, "warm_wpack": 2.50, "payload_load": 0.00,
                 "brepack_pass": 0.00, "refiner_weights": 1.40,
                 "bridge_jit": 2.65, "startup_s": 13.13},
    "512_1711": {"wpack_verify": 2.10, "devkill_prewarm": 6.10,
                 "brepack_hit": 0.00, "warm_wpack": 3.50, "payload_load": 9.70,
                 "brepack_pass": 31.73, "refiner_weights": 2.50,
                 "bridge_jit": 12.76, "startup_s": 68.39},
    "128_1717": {"wpack_verify": 1.85, "devkill_prewarm": 0.00,
                 "brepack_hit": 0.00, "warm_wpack": 4.00, "payload_load": 9.10,
                 "brepack_pass": 31.58, "refiner_weights": 1.90,
                 "bridge_jit": 0.98, "startup_s": 49.41},
    # The AC arms, measured 2026-11-03 18:33 (512^2 x8) and 18:37 (128^2 x1) on
    # the SAME recipe and env_id as the 17:xx battery arms.  They are the other
    # half of the calibration: the pair (battery, AC) is what turns a startup_s
    # into a power-state share, and the two runs reproduce the battery latents
    # byte for byte, so the decomposition below has a byte-identical twin.
    "512_ac": {"wpack_verify": 1.61, "devkill_prewarm": 3.94,
               "brepack_hit": 0.00, "warm_wpack": 1.70, "payload_load": 3.20,
               "brepack_pass": 27.02, "refiner_weights": 1.20,
               "bridge_jit": 6.97, "startup_s": 45.64},
    "128_ac": {"wpack_verify": 1.43, "devkill_prewarm": 0.00,
               "brepack_hit": 0.00, "warm_wpack": 1.40, "payload_load": 4.30,
               "brepack_pass": 26.20, "refiner_weights": 1.00,
               "bridge_jit": 0.60, "startup_s": 34.93},
    # ---- ON BATTERY, 2026-11-03 19:23-19:42, recipe env_id 2b4fbff71396f728
    # (or 36628e0c59bcbdb6 for the recorded OFF arm).  These are the arms of
    # STARTUP_PAIR; they are NOT in STARTUP_ARMS, whose table is the older
    # calibration that section 12 explains.  Each entry also carries loop_s and
    # total_s, which the pair table prints and the phase table ignores.
    # Rows that overlap are marked: the cache's map+verify runs on a worker
    # thread, so here it can hide under another phase and bridge_jit can go
    # NEGATIVE -- that is overlap, not a missing bucket.
    "512_bt_off1": {"wpack_verify": 2.02, "devkill_prewarm": 5.41,
                    "brepack_hit": 0.00, "warm_wpack": 1.40,
                    "payload_load": 5.20, "brepack_pass": 22.89,
                    "refiner_weights": 2.10, "bridge_jit": 8.22,
                    "startup_s": 47.24, "loop_s": 95.09, "total_s": 156.29},
    "512_bt_on": {"wpack_verify": 1.88, "devkill_prewarm": 3.40,
                  "brepack_hit": 3.45, "warm_wpack": 2.90,
                  "payload_load": 0.00, "brepack_pass": 0.00,
                  "refiner_weights": 1.40, "bridge_jit": 4.12,
                  "startup_s": 17.15, "loop_s": 97.15, "total_s": 131.64},
    "512_bt_off2": {"wpack_verify": 1.88, "devkill_prewarm": 5.81,
                    "brepack_hit": 0.00, "warm_wpack": 3.70,
                    "payload_load": 7.80, "brepack_pass": 31.74,
                    "refiner_weights": 2.10, "bridge_jit": 10.94,
                    "startup_s": 63.97, "loop_s": 99.53, "total_s": 182.28},
    "128_bt_off": {"wpack_verify": 2.12, "devkill_prewarm": 0.00,
                   "brepack_hit": 0.00, "warm_wpack": 2.40,
                   "payload_load": 4.50, "brepack_pass": 32.50,
                   "refiner_weights": 2.00, "bridge_jit": 1.19,
                   "startup_s": 44.71, "loop_s": 9.55, "total_s": 64.56},
    "128_bt_on": {"wpack_verify": 2.10, "devkill_prewarm": 0.00,
                  "brepack_hit": 4.34, "warm_wpack": 3.90,
                  "payload_load": 0.00, "brepack_pass": 0.00,
                  "refiner_weights": 0.70, "bridge_jit": -2.13,
                  "startup_s": 8.91, "loop_s": 10.99, "total_s": 31.72},
}

# Which arms the section-13 table prints, in order.  An arm that is measured
# but not listed here stays out of the doc, so adding one is a single line.
STARTUP_ARMS = [
    ("512_0630", "512^2 x8 06:30 (state NOT recorded)"),
    ("512_1711", "512^2 x8 17:11 (battery)"),
    ("512_ac", "512^2 x8 18:33 (AC)"),
    ("128_1717", "128^2 x1 17:17 (battery)"),
    ("128_ac", "128^2 x1 18:37 (AC)"),
]

# ---------------------------------------------------------------------------
# 2c. STARTUP_PAIR -- the B repack removed, as a WITHIN-WINDOW A/B pair.
#
# A startup_s may only be compared with another startup_s taken in the SAME
# power state (section 12 legalises exactly one exception: a back-to-back pair,
# minutes apart).  So the cache was measured as a pair, not against the AC
# reference: same tree, same recipe, same shape, same power state, only
# T22_BREPACK_CACHE differing.  The difference between the two columns is the
# repack and nothing else -- no power coefficient, no second variable.
#
# Each arm is a dict: {"key": a STARTUP_MEASURED entry, "label", "job",
# "latents" (sha16), "cache" ("0"/"1"), "batt" (the probe's battery percent)}.
# Three arms where possible: OFF, ON, OFF -- the second OFF arm is the DRIFT
# CHECK, and it costs one run to know whether the box moved during the window.
# ---------------------------------------------------------------------------
STARTUP_PAIR = [
    {
        "shape": "512^2 x8",
        "state": "battery",
        "when": "2026-11-03 19:23:43 - 19:38:2x, back to back",
        "why": (
            "Same tree, same recipe, same shape, same power state; ONE value "
            "differs (T22_BREPACK_CACHE).  The second OFF arm is the drift "
            "check: it ran after the ON arm, so the pair shows how much the box "
            "moved while the measurement was being taken -- for free, because "
            "the run was needed anyway.  P02 recorded 'no device process' before "
            "each arm, and the P12 speed index is quoted per arm."),
        "arms": [
            {"key": "512_bt_off1", "label": "OFF #1", "cache": "0",
             "job": "20261103-192343-658254", "latents": "2abf91331200d069",
             "batt": 68, "mibs": 1127},
            {"key": "512_bt_on", "label": "ON", "cache": "1",
             "job": "20261103-192702-8edbf1", "latents": "2abf91331200d069",
             "batt": 67, "mibs": 1321},
            {"key": "512_bt_off2", "label": "OFF #2 (drift)", "cache": "0",
             "job": "20261103-193520-085610", "latents": "2abf91331200d069",
             "batt": 64, "mibs": 989},
        ],
    },
    {
        "shape": "128^2 x1",
        "state": "battery",
        "when": "2026-11-03 19:40:52 - 19:42:46, back to back",
        "why": (
            "The native path (M=96) pays exactly the same startup rows, so the "
            "same removal applies: a 1-step job is a STARTUP measurement, not a "
            "compute measurement (section 12), which is why the total nearly "
            "halves while the loop does not move."),
        "arms": [
            {"key": "128_bt_off", "label": "OFF", "cache": "0",
             "job": "20261103-194052-a03c20", "latents": "ba89b71cfd1dfaa0",
             "batt": 62, "mibs": 988},
            {"key": "128_bt_on", "label": "ON", "cache": "1",
             "job": "20261103-194246-47673d", "latents": "ba89b71cfd1dfaa0",
             "batt": 61, "mibs": 1136},
        ],
    },
]

# The 512_ac residual, split once.  job 20261103-183357-58b472, every second of
# startup_s = 45.64 accounted for.  Rows 1-8 are the product's own log lines
# (exact); rows 9-11 are the segments no product line prints and are measured
# by imaggers/startup/startup_probe.py in the SAME py310 interpreter (median of
# 3); the last row is what is left -- the chunked path's projector install.
STARTUP_AC_LEDGER = [
    ("shape geometry", "patch grid / M / resume-latent geometry", 0.03,
     "anchor delta first_event -> shape_geometry"),
    ("wpack artifact read", "map + sha256 the 1.85 GiB wpack", 1.61,
     "[t22] prewarm: ... sha256 verified in N s"),
    ("devkill prewarm", "pack the 120 G=15 projection W images", 3.94,
     "[devkill] prewarm: 120 G=15 ... in N s"),
    ("warm wpack", "pack 30 layers' projection images", 1.70,
     "[t3d] warm: ... in N s"),
    ("payload load", "read + permute the 3.71 GB BG/B3 payload", 3.20,
     "[t13] full-memory cache: ... in N s"),
    ("B repack", "the 5.31 GB in-place B-order permutation", 27.02,
     "[devkill] B repack at startup: N s"),
    ("refiner cache read", "map noise_refiner.0/1 (2 x 0.67 GB)", 1.20,
     "[t3d] refiner weights loaded noise_refiner.N  N s"),
    ("image registration", "19 images: 16 big-M + att/attip/ffn", 0.20,
     "[t3d] registered 19 images: sum of the per-image seconds"),
    ("torch import", "the chunked install's _torch() (chunked.py:936)", 1.93,
     "startup_probe.py, median of 3"),
    ("numba import", "first numba import inside the glue install", 0.24,
     "startup_probe.py, median of 3"),
    ("fastglue install", "the 8 byte gates + hook install (warms the JIT)", 0.64,
     "startup_probe.py, median of 3"),
    ("chunk_path install",
     "big-M whole-shape proj (rt17) + its BOs + [t18] parse + t8 + RESUME "
     "-- THE ONLY SEGMENT THE PRODUCT DOES NOT PRINT", 3.93,
     "startup_s minus the rows above"),
]

# What the word "startup" hides: these are real seconds of the same job, but
# they are NOT inside startup_s (webui/runner.py:1396-1424 defines it as
# first_event -> startup_done).  Quoted so nobody re-derives them from
# startup_s and calls them missing.
STARTUP_OUTSIDE = [
    ("device child spawn + interpreter + driver import", 0.55,
     "bridge_import_s: launch -> the child's first @@WEBUI@@ event"),
    ("TE init (text enc cache hit)", 0.01,
     "timing.textenc_s; the cache is on disk, so this is a read, not an encode"),
    ("wpack prewarm wait", 0.68,
     "timing.prewarm_pack_s; the packer runs while this job text encodes"),
    ("loop_s (8 steps)", 92.87,
     "startup_done -> last step; step 0's first layer carries torch's first-call "
     "warmup (host 5.54 s for that layer against ~0.3 s steady state)"),
    ("VAE loader", 3.94, "timing.vae_load_s: the lean decoder, after the loop"),
    ("VAE decode + save", 6.22, "timing.vae_decode_s + vae_save_s"),
]

# AC vs battery on THIS box, one recipe (env_id 0880daf30c579271), two shapes,
# measured back to back on 2026-11-03.  The battery column is the 17:11/17:17
# pair, the AC column the 18:33/18:37 pair; the latents sha is identical inside
# every pair, so nothing here is a code difference.
POWER_CAL = [
    ("loop_s", "512^2 x8", 99.87, 92.87),
    ("startup_s", "512^2 x8", 68.39, 45.64),
    ("total_s", "512^2 x8", 186.19, 150.94),
    ("loop_s", "128^2 x1", 8.73, 5.73),
    ("startup_s", "128^2 x1", 49.41, 34.93),
    ("total_s", "128^2 x1", 68.47, 48.38),
]

POWER_CAL_JOBS = {
    "battery": ["20261103-171112-09bfbe", "20261103-171717-97cdd7"],
    "ac": ["20261103-183357-58b472", "20261103-183756-2f06c3"],
}

# The recipe's env_id BEFORE T22_BREPACK_CACHE=1 joined it.  Every measurement
# dated before 2026-11-03 19:xx was taken under this id, i.e. with the 5.31 GB
# B repack PAID inside startup_s.  Kept as a name so the doc can say which
# recipe a historical number belongs to instead of implying today's.
PRE_CACHE_ENV_ID = "0880daf30c579271"

POWER_RULE = (
    "Quote, compare and gate AC numbers only; a battery number is real but is "
    "readable only against another battery number taken minutes apart -- never "
    "against an AC reference.")

REFERENCES = {
    "512x512:8": {
        "latents_sha256":
            "2abf91331200d0697d5582e16619db69cad9e387ca95137d18c234be38e84d6e",
        "path": "chunked", "m": 1056, "exec_mode": "chunked",
        "loop_s": 94.05, "loop_tol": 0.25,
        "total_s": [124.9, 135.7], "vae": False,
        "ref_jobs": ["20261102-211411-109ea0", "20261102-213051-7c90ae"],
        "note": "the campaign fast arm (T22_FASTGLUE=1 + devkill + bigm), "
                "VAE off; two runs, one sha",
    },
    "128x128:8": {
        "latents_sha256":
            "3c0065d704632b62433272f406cc29cdff521f3b92ab09247d74afc4bb78e0f4",
        "path": "native", "m": 96, "exec_mode": "native",
        "loop_s": 21.61, "loop_tol": 0.25,
        "total_s": [50.8], "vae": False,
        "ref_jobs": ["20261102-212947-1c18e0"],
        "note": "native M=96 path, devkill FFN at rows=96, fastglue NOT part of "
                "this path",
    },
}

OFF_TOKENS = ("0", "false", "False", "off", "no", "none", "")

# ---------------------------------------------------------------------------
# 2d. ACCEPTANCE, per-check SCOPE, the TAG rules, the TOTAL-wall window and the
#     post-run device states.  Five definitions that used to be prose-only (or
#     missing), each of which a cold reader had to infer from an output.
# ---------------------------------------------------------------------------
# The postflight registry is 18 checks (Q01-Q18).  Not every one of them can
# apply to every run: Q03-Q06 read evidence only the CHUNKED branch produces
# (webui/chunked.py:988-1012 installs fastglue/devkill inside the
# ZIMAGE_BIGM+DEVKILL arm of chunked.install(), which a native shape never
# reaches) and Q08/Q09 need a job that ran the VAE.  Those return SKIP.
#
# So "18/18" is not something a native run can print: 128^2 comes back
# 13 PASS + 1 WARN + 4 SKIP + 0 FAIL.  What this file asserts -- and what
# report() prints and RUNBOOK section 6 defines -- is
#
#     N evaluated (each PASS, WARN or a STRUCTURAL SKIP), 0 FAIL.
#
# SKIP is not a pass and never counts as one.  WARN is a number to read and act
# on, never a refusal.  FAIL is the only thing that rejects a run.
ACCEPTANCE_RULE = (
    "a run is ACCEPTED iff FAIL == 0 over every check in the registry; a SKIP "
    "is a check that is STRUCTURALLY N/A on this run's path and is NOT a pass, "
    "so the criterion is 'N evaluated, 0 FAIL' and never 'N/N PASS'")

# Which checks can apply, per id.  RUNBOOK section 6 prints this as its
# "applies to" column, and report() names the SKIPs it caused.
Q_SCOPE = {
    "Q01": "both paths",
    "Q02": "both paths",
    "Q03": "chunked path only",
    "Q04": "chunked path only (fastglue is installed by chunked.install())",
    "Q05": "chunked path only (same install)",
    "Q06": "chunked path only (its [ffnprof] half is a fastglue hook)",
    "Q07": "both paths",
    "Q08": "vae=1 jobs only",
    "Q09": "vae=1 jobs only",
    "Q10": "both paths",
    "Q11": "both paths",
    "Q12": "both paths",
    "Q13": "both paths, in two parts (each needs its own reference number)",
    "Q14": "both paths",
    "Q15": "both paths",
    "Q16": "both paths",
    "Q17": "both paths",
    "Q18": "both paths (the --cache-all site maps on native too)",
}

# -Tag.  It is a FILENAME KEY and a job prefix, so its rules are enforced by
# safe_tag() below, announced by announce_tag(), and written down in RUNBOOK
# section 1.  Before this block they existed only inside the regex.
TAG_RE = re.compile(r"[^A-Za-z0-9._-]+")
TAG_MAX = 40
TAG_PREFIX = "repro_"

# The total-wall window (Q13's second part).  The reference totals (124.9 /
# 135.7 s at 512^2 x8) are ~25 % below what this engine's own runs measure for
# the SAME latents (150.9 s), because the reference arms ran vae=False while
# this engine submits vae=True (+~10 s at 512^2) and because startup_s is the
# 15-30 s row that moves with the power state and with the B-repack arm.  So the
# gate is on the DIFFUSION CORE = startup_s + loop_s (for a vae-off arm that IS
# its total minus the child's import/TE, ~1 s) against the reference total.
CORE_TOL = 0.20

# How a run can END, device-wise.  Printed by run_job(), stored in
# last/<tag>.device.json, re-printed by the postflight, and listed in RUNBOOK
# section 9 (generated from this list).
POSTRUN_STATES = [
    ("device.lock: released (pid N, owner provably dead)",
     "the clean case: this run took the F-14.1 lock, its server is dead, and "
     "the engine removed it",
     "nothing.  The device is not held by this run any more"),
    ("device.lock: absent",
     "no lock existed when the server stopped: this launcher never took one, or "
     "another script unlinked it (run_job_py.py:31-33)",
     "nothing.  P02 -- processes, not the lock -- is the device gate"),
    ("device.lock: held by another pid N -- NOT touched",
     "SOMEBODY ELSE owns the device: another agent's run took the lock while "
     "this job was finishing.  The file is THEIRS and this engine refuses to "
     "delete another owner's lock (that is what P03 is for)",
     "do NOT delete the lock and do NOT start a second job.  Your job has "
     "already finished, and the postflight reads FILES only, so run it now, "
     "even with the NPU busy; the next run's P02 waits for the other pid"),
    ("device.lock: owner pid N still alive -- NOT touched",
     "the lock is THIS run's, but its server process has not exited yet (the "
     "20 s terminate window elapsed)",
     "wait ~20 s and run -Action status; a pid that never dies is a hung "
     "server, not a lock problem"),
    ("[run] WARNING device still busy Ns after the server stopped: [...]",
     "an orphaned bridge or another agent's process was still on the NPU when "
     "this run ended",
     "not a failure of your job.  The next run's P02 waits for it; do not kill "
     "another session's bridge"),
]

# ---------------------------------------------------------------------------
# 3. small helpers
# ---------------------------------------------------------------------------
def out(msg=""):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(str(msg).encode("ascii", "replace").decode(), flush=True)


def hr(ch="-", n=78):
    out(ch * n)


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def sha16(s):
    return hashlib.sha256(s.encode("utf-8", "replace")).hexdigest()[:16]


def read_text(p):
    b = pathlib.Path(p).read_bytes()
    if b[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return b.decode("utf-16", "replace")
    return b.decode("utf-8", "replace")


def launch_flags_from_source():
    """The env names webui/config.py records per job (LAUNCH_FLAGS), as TEXT.

    Read, not imported: doc --check must be byte-stable and must not depend on
    whichever environment happens to be exported when it runs.  This is the
    tuple job.json:env_flags can ever show, which is why it is the ceiling on
    what Q11's first level can prove.
    """
    try:
        txt = (ROOT / "webui" / "config.py").read_text(encoding="utf-8",
                                                       errors="replace")
    except OSError:
        return ()
    m = re.search(r"LAUNCH_FLAGS\s*=\s*\(([^)]*)\)", txt)
    if not m:
        return ()
    return tuple(re.findall(r'"([A-Za-z0-9_]+)"', m.group(1)))


def recipe_env():
    return {k: val for k, val, _, _ in RECIPE}


def env_id(env=None):
    env = env or recipe_env()
    body = "\n".join("%s=%s" % (k, env[k]) for k in sorted(env))
    return sha16(body)


def apply_env(verbose=True):
    """Set RECIPE, delete everything else that selects a device path.

    Returns (env, scrubbed, conflicts).  This is the single place the job's
    environment is decided: the engine and every child see the same dict.
    """
    scrubbed, conflicts = [], []
    have = {k for k, _, _, _ in RECIPE}
    names = set(SURVEYED_ENV) | have | {n for n, _w, _s in FORBID}
    for n in sorted(names):
        if n in SCRUB_KEEP or n in have:
            continue
        if n in os.environ:
            scrubbed.append((n, os.environ.pop(n)))
    for k, v, _why, _src in RECIPE:
        if k in os.environ and os.environ[k] != v:
            conflicts.append((k, os.environ[k], v))
        os.environ[k] = v
    env = dict(os.environ)
    if verbose:
        for k, old in scrubbed:
            out("[env] scrubbed inherited %s=%r (not in the recipe)" % (k, old))
        for k, old, new in conflicts:
            out("[env] overrode %s=%r -> %r" % (k, old, new))
    return env, scrubbed, conflicts


def _psutil():
    import psutil
    return psutil


DEVICE_HINTS = ("t18_run", "launch_webui", "webui.py", "bridge.py",
                "vae_worker", "run_job_py", "webui/bridge", "webui\\bridge")


def my_tree():
    ps = _psutil()
    me = ps.Process()
    pids = {me.pid}
    try:
        for c in me.children(recursive=True):
            pids.add(c.pid)
    except Exception:
        pass
    return pids


def device_users():
    """Processes that can own / are using the NPU right now.

    'device.lock is absent' proves nothing: run_job_py.py:31-33 unlinks it at
    startup, and a launcher that bypasses run_job_py never writes one.  This
    looks at processes instead.
    """
    ps = _psutil()
    ignore = my_tree()
    hits = []
    root_s = str(ROOT).lower()
    for p in ps.process_iter(["pid", "name", "cmdline", "cpu_times", "memory_info"]):
        try:
            if p.info["pid"] in ignore:
                continue
            cl = p.info.get("cmdline") or []
            joined = " ".join(cl)
            low = joined.lower()
            if not (root_s in low or any(h in low for h in DEVICE_HINTS)):
                continue
            if len(cl) < 2:
                continue
            script = cl[1].lower()
            if not any(h in script for h in DEVICE_HINTS):
                continue
            cpu = p.info.get("cpu_times")
            rss = p.info.get("memory_info")
            hits.append({
                "pid": p.info["pid"], "name": p.info["name"],
                "rss_gb": round(rss.rss / 2**30, 2) if rss else None,
                "cpu_s": round((cpu.user + cpu.system), 1) if cpu else None,
                "cmd": joined[:220],
            })
        except Exception:
            continue
    return hits


def port_busy(port=None, host="127.0.0.1"):
    port = PORT if port is None else port
    s = socket.socket()
    s.settimeout(0.4)
    try:
        return s.connect_ex((host, port)) == 0
    finally:
        s.close()


def wait_quiet(deadline_s=None, quiet_s=None, say=True):
    """Block until no device user and the port is free, sustained quiet_s."""
    deadline_s = DEVICE_WAIT_S if deadline_s is None else deadline_s
    quiet_s = QUIET_S if quiet_s is None else quiet_s
    t0 = time.time()
    quiet_from = None
    last = None
    while True:
        users = device_users()
        busy = port_busy()
        cur = (tuple(sorted(u["pid"] for u in users)), busy)
        if users or busy:
            quiet_from = None
            if say:
                who = ", ".join("pid %s %s rss %s GB" %
                                (u["pid"], u["name"], u["rss_gb"]) for u in users)
                out("[wait] %5.0fs device busy: %s%s" %
                    (time.time() - t0, who or "port %d open" % PORT,
                     " (port open)" if busy and who else ""))
                for u in users:
                    out("        %s" % u["cmd"])
            last = cur
        else:
            if quiet_from is None:
                quiet_from = time.time()
                if say:
                    out("[wait] %5.0fs no device process, port free -- holding "
                        "%.0fs quiet" % (time.time() - t0, quiet_s))
            if time.time() - quiet_from >= quiet_s:
                return True, time.time() - t0, []
        if time.time() - t0 > deadline_s:
            return False, time.time() - t0, users
        time.sleep(2 if cur != last else 3)


def lock_state():
    """(present, text, owner, pid, pid_alive)."""
    if not LOCK.exists():
        return False, "", None, None, None
    try:
        txt = LOCK.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        txt = ""
    owner = None
    pid = None
    m = re.search(r"owner=(\S+)", txt)
    if m:
        owner = m.group(1)
    m = re.search(r"pid=(\d+)", txt)
    if m:
        pid = int(m.group(1))
    alive = None
    if pid:
        try:
            ps = _psutil()
            alive = bool(ps.pid_exists(pid) and ps.Process(pid).is_running())
        except Exception:
            alive = None
    return True, txt, owner, pid, alive


def memory_snapshot():
    from webui import config
    return config.memory_snapshot()


# ---------------------------------------------------------------------------
# 5b. POWER STATE -- the covariate that made two identical runs differ 15-20 %
#
# Measured on this box (2026-11-03): the SAME switch set, the same shape and the
# same latents sha produced loop 94.90 s on AC and 82.90 / 79.21 s on battery.
# Every number in this file is only comparable to another number taken in the
# SAME power state, so the state is probed and RECORDED with every run instead
# of being remembered.
#
# The probe is GetSystemPowerStatus (kernel32) -- the same call Windows itself
# uses, and the only one that worked here: Get-CimInstance Win32_Battery,
# root\\wmi BatteryStatus, Win32_ComputerSystem and powercfg /batteryreport are
# all access-denied inside this sandbox, and Get-CimInstance returning nothing
# (the "no battery object" reading) is a PERMISSION symptom, not a desktop.
# ---------------------------------------------------------------------------
POWER_PROBE = ("ctypes.windll.kernel32.GetSystemPowerStatus")
_AC = {0: "battery", 1: "ac", 255: "unknown"}


def power_state():
    """(state, dict) -- 'ac' / 'battery' / 'unknown' plus the raw fields.

    Also takes a fixed-size speed index (sha256 over 64 MiB, C-speed, no numpy
    import) so a run carries its own magnitude even when the power state was
    not captured.  Cost: ~0.2 s, once per preflight.
    """
    d = {"probe": POWER_PROBE, "ac_line": None, "state": "unknown",
         "battery_flag": None, "battery_pct": None, "err": ""}
    try:
        import ctypes

        class _SPS(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_ubyte),
                        ("BatteryFlag", ctypes.c_ubyte),
                        ("BatteryLifePercent", ctypes.c_ubyte),
                        ("SystemStatusFlag", ctypes.c_ubyte),
                        ("BatteryLifeTime", ctypes.c_int),
                        ("BatteryFullLifeTime", ctypes.c_int)]

        s = _SPS()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
            d["err"] = "GetSystemPowerStatus returned 0"
        else:
            d["ac_line"] = int(s.ACLineStatus)
            d["battery_flag"] = int(s.BatteryFlag)
            d["battery_pct"] = int(s.BatteryLifePercent)
            d["state"] = _AC.get(int(s.ACLineStatus), "unknown")
            if int(s.BatteryFlag) & 128:
                d["battery_flag_meaning"] = "no system battery"
    except Exception as exc:                                  # noqa: BLE001
        d["err"] = "%s: %s" % (type(exc).__name__, exc)
    try:
        # utf-8 with errors=replace, not the default: powercfg answers in
        # UTF-8 here and a text=True capture decoded as GBK raised inside the
        # reader thread, which left the plan EMPTY in every record (found
        # 2026-11-03 18:52).
        d["scheme"] = (subprocess.run(["powercfg", "/getactivescheme"],
                                      capture_output=True, text=True,
                                      encoding="utf-8", errors="replace",
                                      timeout=20).stdout or "").strip()[:120]
    except Exception as exc:                                  # noqa: BLE001
        d["scheme"] = "ERROR %s" % exc
    t0 = time.perf_counter()
    try:
        buf = bytes(4 * 1024 * 1024)
        hh = hashlib.sha256()
        for _ in range(16):
            hh.update(buf)
        d["sha256_mib_s"] = round(64.0 / max(1e-6, time.perf_counter() - t0), 1)
    except Exception as exc:                                  # noqa: BLE001
        d["sha256_mib_s"] = None
        d["speed_err"] = str(exc)
    return d["state"], d


def power_line(d):
    """One line, printed by every action, so no record is power-blind."""
    return ("power: %s (ACLineStatus=%s scheme=%s battery=%s%% "
            "sha256=%.0f MiB/s)" %
            (d.get("state"), d.get("ac_line"),
             (d.get("scheme") or "").split("(")[0].strip()[:36],
             d.get("battery_pct"),
             d.get("sha256_mib_s") or 0.0))


def reserve_probe(gb=None, chunk_mb=None):
    """Actually commit the fast path's working set, then give it back.

    The counter (ullAvailPageFile) is a proxy that other sessions move; this
    allocates and touches the pages, which is what the cache-all pass needs.
    Returns (ok, achieved_gb, err, seconds).
    """
    gb = PROBE_GB if gb is None else gb
    chunk_mb = PROBE_CHUNK_MB if chunk_mb is None else chunk_mb
    t0 = time.time()
    got = 0.0
    held = []
    try:
        n_chunks = int(round(gb * 1024.0 / chunk_mb))
        for _ in range(n_chunks):
            b = bytearray(chunk_mb * 1024 * 1024)
            for off in range(0, len(b), 4096):
                b[off] = 1
            held.append(b)
            got += chunk_mb / 1024.0
        return True, got, "", time.time() - t0
    except MemoryError:
        return False, got, "MemoryError after %.2f GiB" % got, time.time() - t0
    except Exception as exc:
        return False, got, "%s: %s" % (type(exc).__name__, exc), time.time() - t0
    finally:
        held.clear()


def plan_for(h, w):
    """What webui/capability.py says this request will do, in THIS env."""
    from webui import capability, shapes
    d = capability.describe(h, w)
    r = shapes.resolve(h, w)
    whole_rows = None
    try:
        whole_rows = shapes._whole_shape_rows()
    except Exception:
        pass
    planner = {
        "h": int(h), "w": int(w), "m": int(d["m"]), "path": d["path"],
        "renderable": bool(d["renderable"]),
        "refusal": d.get("refusal") or "",
        "exec_mode": d.get("exec_mode"),
        "run": d.get("run"), "pad_applied": d.get("pad_applied"),
        "ready": bool(d.get("ready")),
        "chunked_runnable": bool(r.get("chunked_runnable")),
        "chunk_estimate": (r.get("chunk_estimate") or {}),
        "chunk_reason": r.get("chunk_reason") or "",
        "whole_shape_rows": whole_rows,
        "rows": int(d["m"]),
        "capability": d,
    }
    return planner


def thread_plan():
    """The thread counts the product will apply, plus a live probe of them."""
    from webui import config
    loop_thr = int(config.loop_threads())
    vae_thr = int(config.vae_threads())
    probe = {"ok": False, "loop": None, "err": ""}
    code = "import sys\nsys.path.insert(0, {root!r})\nimport torch\n" \
           "from webui import chunked\nchunked._torch()\n" \
           "print('TORCH_THREADS', torch.get_num_threads())\n".format(
               root=str(ROOT))
    try:
        r = subprocess.run([str(_need(PY312, "the Python 3.12 interpreter", "ZIMAGE_PY312")), "-c", code], cwd=str(ROOT),
                           capture_output=True, text=True, timeout=300,
                           env=dict(os.environ))
        for line in (r.stdout or "").splitlines():
            if line.startswith("TORCH_THREADS"):
                probe = {"ok": True, "loop": int(line.split()[1]), "err": ""}
        if not probe["ok"]:
            probe["err"] = ((r.stdout or "")[-200:] +
                            (r.stderr or "")[-300:]).replace("\n", " ")
    except Exception as exc:
        probe["err"] = "%s: %s" % (type(exc).__name__, exc)
    return loop_thr, vae_thr, probe


class R(dict):
    """One check result: id / title / status / detail / why / src."""

    def __init__(self, cid, title, status, detail, why="", src=""):
        super().__init__(cid=cid, title=title, status=status, detail=detail,
                         why=why, src=src)


# (accept-sentence, reject-sentence) per report kind
VERDICT = {
    "preflight": ("the run may start",
                  "the run must NOT start; fix the FAILs and re-run the "
                  "preflight"),
    "postflight": ("the path proof holds and the job's number is usable",
                   "the job's number is NOT usable; read the FAIL lines above "
                   "and RUNBOOK section 9"),
}


def report(results, heading, kind="postflight"):
    """Print the checks AND the definition of acceptance, not just a count.

    A bare "N checks, F FAIL" was ambiguous in exactly the case that mattered: a
    native shape has four STRUCTURAL SKIPs, so "18/18" was never available, and a
    reader had to decide on their own whether 14 PASS + 4 SKIP was a pass.  The
    rule is printed with the numbers now: FAIL == 0 over the registry, SKIP is
    N/A and not a pass.
    """
    hr("=")
    out(heading)
    hr("=")
    n = {"PASS": 0, "WARN": 0, "SKIP": 0, "FAIL": 0, "RECORDED": 0}
    for r in results:
        st = r["status"]
        n[st] = n.get(st, 0) + 1
        out("%-4s %-6s %-46s %s" % (st, r["cid"], r["title"], r["detail"]))
        if st != "PASS" and r.get("why"):
            out("             why it matters: %s" % r["why"])
        if st == "FAIL" and r.get("src"):
            out("             decides it:     %s" % r["src"])
    total = len(results)
    hr("-")
    out("%d checks evaluated: %d PASS, %d WARN, %d SKIP, %d FAIL%s"
        % (total, n["PASS"], n["WARN"], n["SKIP"], n["FAIL"],
           (", %d RECORDED" % n["RECORDED"]) if n["RECORDED"] else ""))
    skips = [r["cid"] for r in results if r["status"] == "SKIP"]
    if skips:
        out("  structural SKIP (N/A on this run's path, NOT a pass): %s"
            % " ".join(skips))
    out("  ACCEPTANCE: " + ACCEPTANCE_RULE)
    ok_say, bad_say = VERDICT.get(kind, ("", ""))
    if n["FAIL"]:
        out("REJECTED: %d FAIL of %d evaluated -- %s"
            % (n["FAIL"], total, bad_say))
        return n["FAIL"]
    out("ACCEPTED: %d evaluated, 0 FAIL -- %s" % (total, ok_say))
    return 0


def pf_repo(ctx):
    missing = [str(p) for p in (ROOT / "webui" / "config.py", T18, LAUNCHER,
                                ROOT / "webui.py") if not p.is_file()]
    for py, env in ((PY312, "ZIMAGE_PY312"), (PY310, "ZIMAGE_PY310")):
        if py is None:
            missing.append("%s (set %s)" % (env, env))
        elif not py.is_file():
            missing.append(str(py))
    if missing:
        return R("P01", "repo tree + both interpreters", "FAIL",
                 "%d missing, first: %s" % (len(missing), missing[:2]),
                 "every path in this file is relative to it",
                 "webui/config.py:22-23")
    return R("P01", "repo tree + both interpreters", "PASS",
             "py312 + py310 + t18_run + launcher present", "",
             "webui/config.py:22-23")


def pf_idle(ctx):
    ok, waited, users = wait_quiet(say=True)
    if not ok:
        return R("P02", "device actually idle (processes, not the lock)", "FAIL",
                 "still busy after %.0fs: %s" %
                 (waited, [(u["pid"], u["name"]) for u in users]),
                 "'lock absent' means nothing: run_job_py.py:31-33 unlinks it at "
                 "startup, so a job can hold the device with no lock present. "
                 "Two device users = 0xc01e0009 and a corrupt timing",
                 "imaggers/contend/wait_and_run.py:34-40, "
                 "imaggers/util/adaptive/npu_dev.py:275-285")
    return R("P02", "device actually idle (processes, not the lock)", "PASS",
             "no device process for this repo, port %d free (waited %.0fs)" %
             (PORT, waited), "", "imaggers/contend/wait_and_run.py:34-40")


def pf_lock(ctx):
    present, txt, owner, pid, alive = lock_state()
    if not present:
        return R("P03", "lock file: absent, or stale-with-dead-owner", "PASS",
                 "no imaggers/device.lock (P02 is the real gate)", "", "")
    if alive:
        return R("P03", "lock file: absent, or stale-with-dead-owner", "FAIL",
                 "held by LIVE pid %s (%s): %r" % (pid, owner, txt[:80]),
                 "removing a live owner's lock is how two jobs end up on one "
                 "device", "imaggers/bigpipe2/launch_webui_bigm.py:59-62")
    if alive is None:
        return R("P03", "lock file: absent, or stale-with-dead-owner", "FAIL",
                 "lock present, owner pid unknown/unprobeable: %r" % (txt[:80],),
                 "a stale lock blocks the next launcher until someone deletes it",
                 "imaggers/bigpipe2/launch_webui_bigm.py:59-62")
    dst = LOCK.with_name("device.lock.stale-%s-%s" %
                         (pid, time.strftime("%H%M%S")))
    try:
        LOCK.replace(dst)
    except OSError as exc:
        return R("P03", "lock file: absent, or stale-with-dead-owner", "FAIL",
                 "stale lock (pid %s dead) could not be moved: %s" % (pid, exc),
                 "", "")
    return R("P03", "lock file: absent, or stale-with-dead-owner", "PASS",
             "stale lock, owner pid %s provably dead -> moved to %s" %
             (pid, dst.name), "", "imaggers/util/place/s2_dev.py:300-310")


def pf_memory(ctx):
    from webui import config
    thr = float(config.MEMORY_FAST_THRESHOLD_GB)
    # The commit headroom is a shared resource: the previous job's teardown can
    # take a few seconds to give it back.  Wait, but never lie about the number.
    t0 = time.time()
    tries = 0
    while True:
        tries += 1
        snap = memory_snapshot()
        auto_mode, _, auto_why = config.choose_memory_mode("auto")
        ok, got, err, secs = reserve_probe()
        if ok or time.time() - t0 > MEM_WAIT_S or tries >= 8:
            break
        out("[mem] %.2f GB commit / %.2f GB avail, probe needs %.1f GiB (%s) -- "
            "retrying in 10 s (%.0fs of %ds)" %
            (snap["commit_gb"], snap["available_gb"], PROBE_GB, err,
             time.time() - t0, MEM_WAIT_S))
        time.sleep(10)
    ctx["memory"] = {"snapshot": snap, "auto_mode": auto_mode,
                     "auto_why": auto_why, "threshold_gb": thr,
                     "probe_gb": got, "probe_ok": ok,
                     "probe_s": round(secs, 1), "tries": tries}
    detail = ("commit %.2f GB (threshold %.1f), avail %.2f GB; auto would pick "
              "%r; reserve probe %.2f GiB %s in %.1fs" %
              (snap["commit_gb"], thr, snap["available_gb"], auto_mode, got,
               "OK" if ok else "FAILED (" + err + ")", secs))
    if not ok:
        return R("P04", "memory mode will be 'fast' (proved, not read)", "FAIL",
                 detail,
                 "the cache-all path needs ~6 GB of private commit.  Under it "
                 "the runner flips to --no-cache-all (2x slower loop AND a "
                 "different latent baseline), or falls back mid-run with "
                 "'fast cache-all failed with MemoryError'",
                 "webui/config.py:512,546-556, webui/runner.py:492-527")
    if auto_mode == "fast":
        return R("P04", "memory mode will be 'fast' (proved, not read)", "PASS",
                 detail, "", "webui/config.py:546-556")
    ctx["forced_fast"] = True
    return R("P04", "memory mode will be 'fast' (proved, not read)", "PASS",
             detail + " -> the job will ask for memory_mode=fast EXPLICITLY",
             "auto would silently choose the slow path AND a different latent "
             "baseline; a run in that mode is not comparable to the reference",
             "webui/config.py:546-556")


def pf_renderable(ctx):
    p = ctx["plan"]
    if not p["renderable"]:
        return R("P05", "size is renderable per webui/capability.py", "FAIL",
                 "%dx%d refused: %s" % (p["h"], p["w"], p["refusal"][:300]),
                 "a refused size dies at submit with an HTTP 400 and a Chinese "
                 "sentence; this is the same sentence, one step earlier",
                 "webui/capability.py:185-243, webui/shapes.py:570-677")
    note = ""
    if p["refusal"]:
        note = (" -- NOTE: describe() also returns a non-empty 'refusal' (the "
                "chunk_gate M<=304 sentence); ignore it, path=%r is what runs"
                % p["path"])
    return R("P05", "size is renderable per webui/capability.py", "PASS",
             "%dx%d -> path=%s m=%d exec=%s%s" %
             (p["h"], p["w"], p["path"], p["m"], p["exec_mode"], note),
             "", "webui/capability.py:185-243")


def pf_env(ctx):
    # apply_env already ran once, at the top of preflight(); a second pass here
    # would report only what webui.config's own os.environ.setdefault() lines
    # re-added (ZIMAGE_VAE_OPNPU/ZIMAGE_VAE_LAUNCH) and hide the real list.
    scrubbed = ctx.get("scrubbed0") or []
    conflicts = ctx.get("conflicts0") or []
    expected = ctx.get("expected") or dict(RECIPE_ENV)
    arms = [(k, v) for k, v in (ctx.get("extras") or []) if k in RECIPE_ENV
            and str(RECIPE_ENV[k]) != str(v)]
    bad = ["%s=%r (expected %r)" % (k, os.environ.get(k), v)
           for k, v in sorted(expected.items()) if os.environ.get(k) != v]
    ctx["env"] = dict(expected)
    ctx["env_id"] = env_id(ctx["env"])
    ctx["scrubbed"] = scrubbed
    ctx["arms"] = arms
    if bad:
        return R("P06", "the job env is EXACTLY the recipe", "FAIL",
                 "not applied: %s" % bad, "", "see the constants block")
    for n, why, src in FORBID:
        v = os.environ.get(n)
        if v is not None and str(v).strip() not in OFF_TOKENS:
            return R("P06", "the job env is EXACTLY the recipe", "FAIL",
                     "%s=%r must be off (%s)" % (n, v, why), why, src)
    detail = ("%d vars set, env_id=%s; %d inherited path switches scrubbed%s" %
              (len(expected), ctx["env_id"], len(scrubbed),
               (": " + ", ".join("%s=%s" % (k, v) for k, v in scrubbed[:6]))
               if scrubbed else ""))
    if arms:
        return R("P06", "the job env is EXACTLY the recipe", "WARN",
                 "ARM (not a reference run): %s -- every other one of the %d "
                 "recipe variables is exactly as specified; %s" %
                 (", ".join("%s=%s (recipe %s)" % (k, v, RECIPE_ENV[k])
                            for k, v in arms), len(expected), detail),
                 "the two arms differ in ONE value on purpose; that is what "
                 "makes the difference between them attributable to it",
                 "this file's --extra")
    return R("P06", "the job env is EXACTLY the recipe", "PASS", detail, "",
             "webui/config.py:131-140")


def pf_threads(ctx):
    loop_thr, vae_thr, probe = thread_plan()
    ctx["threads"] = {"loop": loop_thr, "vae": vae_thr, "probe": probe}
    want = recipe_env()
    want_loop = int(want["ZIMAGE_CHUNK_THREADS"])
    want_vae = int(want["ZIMAGE_VAE_THREADS"])
    detail = ("loop=%d vae=%d (pinned); torch probe=%s" %
              (loop_thr, vae_thr,
               ("%d threads" % probe["loop"]) if probe["ok"]
               else "ERR " + probe["err"][:90]))
    if loop_thr != want_loop or vae_thr != want_vae:
        return R("P07", "thread counts the product will apply", "FAIL", detail,
                 "config.vae_threads() follows ZIMAGE_CHUNK_THREADS when unset, "
                 "so a loop setting can silently move the VAE decode and change "
                 "pixels", "webui/config.py:143-163")
    if probe["ok"] and probe["loop"] != want_loop:
        return R("P07", "thread counts the product will apply", "FAIL", detail,
                 "the product code did not end up on %d threads" % want_loop,
                 "webui/chunked.py:100-105")
    if not probe["ok"]:
        return R("P07", "thread counts the product will apply", "WARN", detail,
                 "the torch probe failed; the loop's effective pool is "
                 "unverified (the product prints no line for it)",
                 "webui/chunked.py:100-105")
    return R("P07", "thread counts the product will apply", "PASS", detail, "",
             "webui/config.py:143-163, webui/chunked.py:100-105")


def pf_images(ctx):
    p = ctx["plan"]
    src = T18.read_text(encoding="utf-8", errors="replace")
    need = NATIVE_IMAGES if p["path"] == "native" else CHUNKED_IMAGES
    missing, noref = [], []
    for rel, _why in need:
        f = ROOT / rel
        if not f.is_file():
            missing.append(rel)
            continue
        base = pathlib.Path(rel).stem
        if base not in src:
            noref.append(base)
    if missing:
        return R("P08", "every device image the path registers exists", "FAIL",
                 "missing %d/%d: %s" % (len(missing), len(need), missing[:4]),
                 "the job dies 20-60 s in, after paying startup, with a "
                 "'No such file' that looks like a code bug",
                 "imaggers/image/t18_run.py:319-368")
    if noref:
        return R("P08", "every device image the path registers exists", "FAIL",
                 "on disk but no longer referenced by t18_run.py: %s" % noref,
                 "the image table moved -- update IMAGES in this file",
                 "imaggers/image/t18_run.py:319-368")
    return R("P08", "every device image the path registers exists", "PASS",
             "%d/%d image pairs present and still referenced" %
             (len(need), len(need)), "", "imaggers/image/t18_run.py:319-368")


# ---------------------------------------------------------------------------
# 7b. the B-repack cache: a PREFLIGHT fact (P09b) and a POSTFLIGHT proof (Q18).
#
# The file is content-addressed: its NAME carries a 16-hex key over
#   (a) the sha256 of all 60 source .npy payloads,
#   (b) the source text of the five permutation functions (bg_to_image,
#       b3_to_image, b3_to_image_n64, b3_to_image_k32, _bg_unpermute),
#   (c) the geometry in force (GU/DN/DN64, layers, sizes, nh_bg, dual, alt).
# So "the file exists" is NOT the question -- "the file THIS recipe resolves to
# exists" is.  A stale file is never even opened (brepack_cache.file_for).
#
# Two separate proofs, and neither is sufficient alone, exactly like Q04/Q05:
#   P09b  the file the identity resolves to is THERE, is the size its header
#         says, and its header names THIS identity  -> the run can map it.
#   Q18   the run's own log says it MAPPED it      -> the run did.
# ---------------------------------------------------------------------------
BREPACK_LAYERS = 30
BREPACK_BG_BYTES = 88_473_600        # t18_run.py:514  BG_BYTES
BREPACK_B3_BYTES = 44_236_800        # t18_run.py:515  B3_BYTES
BREPACK_HEADER_BYTES = 64 * 1024     # brepack_cache.py:105
# nh_bg=5 is what the PRODUCT sets on the driver (webui/bridge.py:385-386
# "drv.NH_BG = 5"), which is the value t18_run's _brepack_prefetch hands the
# cache.  It is part of the identity, so a wrong value here is visible as a
# MISS, not as a silent wrong answer.
BREPACK_NH_BG = 5

# The ONE line that proves the consumer armed the cache (prod_loop_bigm.py:619).
BREPACK_MAPPED = "[bigm] B payloads mapped from"
# Every way a run can PAY the repack instead of mapping it.  Each one is a
# silent fallback: the job succeeds, the latents are identical, and the only
# difference is 27 s of wall nobody attributes.
BREPACK_MISS_TOKENS = (
    ("[bigm] B payloads repacked once",
     "the identity MISSED: the 5.31 GB permutation was re-derived and written"),
    ("[bigm] brepack cache unavailable",
     "the consumer raised; the shipped in-place repack ran instead"),
    ("[bigm] B order: mapped",
     "repack_cache_inplace_bigm ran: the cache was never consulted"),
    ("[brepack] cache MISS",
     "no file matched this recipe's identity"),
    ("[brepack] cache rejected",
     "a file existed but its header named a different identity"),
    ("[brepack] cache is ",
     "a file existed at the resolved name but was the wrong SIZE"),
    ("[brepack] CONTENT HASH MISMATCH",
     "the file's bytes are not what its header says"),
    ("[brepack] MULTISET MISMATCH",
     "the file is not a permutation of its sources"),
    ("[brepack] cache unavailable",
     "the prefetch could not import brepack_cache at all"),
    ("[brepack] cannot map",
     "the 5.31 GB file could not be mmapped"),
)
# the second, independent place the mapping shows up: the 3.71 GB read+permute
# (payload_load) is skipped by the same map, at t18_run.py:2692.
BREPACK_MAPPED_T13 = "[t13] cache-all: B payloads mapped from"
DK_BREPACK_ROW = re.compile(r"\[devkill\] B repack at startup: ([0-9.]+) s")


def brepack_modules():
    """Import brepack_cache with the repo's two extra sys.path roots."""
    for p in (ROOT / "imaggers" / "block" / "bigm", ROOT / "imaggers" / "image"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import brepack_cache                       # noqa: E402
    return brepack_cache


def brepack_state():
    """What THIS recipe's identity resolves to, without touching the device.

    Cheap by construction: 60 stat() calls (each source's sha256 is memoised in
    src_idmap.json by name|size|mtime_ns), a 64 KiB header read, and the source
    text of five functions.  It does NOT re-verify the 5.31 GB -- the job's own
    load does that with sha256 + word multiset, and P09 has to stay a check.
    """
    d = {"enabled": os.environ.get("T22_BREPACK_CACHE", "0") == "1",
         "dir": "", "path": None, "name": "", "exists": False, "bytes": 0,
         "want_bytes": 0, "ok": False, "why": "", "alt_kind": "", "key16": "",
         "built": "", "hdr": None, "n_files": 0, "reason": ""}
    try:
        BRC = brepack_modules()
    except Exception as exc:                              # noqa: BLE001
        d["reason"] = "import failed: %r" % (exc,)
        return d
    prep = BRC.prep_default()
    cache_dir = BRC.cache_dir()
    d["dir"] = str(cache_dir)
    dual = os.environ.get("T22_BIGM_B3_DUAL", "1") != "0"
    devhand = os.environ.get("T22_BIGM_DEVHANDOFF", "0") == "1"
    alt = "none" if not dual else ("k32" if devhand else "n64")
    d["alt_kind"] = alt
    try:
        cand = sorted(cache_dir.glob("brepack_*.bin"))
        d["n_files"] = len(cand)
        ident = BRC.identity(prep, BREPACK_LAYERS, BREPACK_BG_BYTES,
                             BREPACK_B3_BYTES, BREPACK_NH_BG, dual, alt)
        key = BRC.identity_key(ident)
        d["key16"] = key[1]
        path = BRC.file_for(key[1], BREPACK_LAYERS, alt)
        d["want_bytes"] = int(BRC._sections(BREPACK_LAYERS, BREPACK_BG_BYTES,
                                            BREPACK_B3_BYTES, alt)[1])
    except Exception as exc:                              # noqa: BLE001
        d["reason"] = "identity failed: %r" % (exc,)
        return d
    d["path"], d["name"] = str(path), path.name
    if not path.is_file():
        others = [c.name for c in cand]
        d["reason"] = ("no file at the name this recipe's identity resolves to"
                       + ("; %d other file(s) present: %s"
                          % (len(others), ", ".join(others[:3])) if others else ""))
        return d
    d["exists"] = True
    try:
        d["bytes"] = path.stat().st_size
    except OSError as exc:
        d["reason"] = "stat failed: %r" % (exc,)
        return d
    try:
        hdr = BRC.read_header(path)
    except Exception as exc:                              # noqa: BLE001
        d["reason"] = "header unreadable: %r" % (exc,)
        return d
    d["hdr"] = hdr
    d["built"] = str(hdr.get("built") or "")
    bad = [k for k, v in {"magic": BRC.MAGIC, "version": BRC.VERSION,
                          "layers": BREPACK_LAYERS, "nh_bg": BREPACK_NH_BG,
                          "dual": bool(dual), "alt_kind": alt,
                          "key16": key[1]}.items() if hdr.get(k) != v]
    if bad:
        d["reason"] = "header names a different identity: %s differ" % (bad,)
        return d
    total = int(hdr.get("file_bytes") or 0)
    if total != d["bytes"]:
        d["reason"] = ("the file is %d B, its header says %d (truncated or "
                       "torn?): the load would treat it as a miss"
                       % (d["bytes"], total))
        return d
    if total != d["want_bytes"]:
        d["reason"] = ("the header's size %d B is not the %d B this geometry "
                       "implies" % (total, d["want_bytes"]))
        return d
    d["ok"] = True
    return d


def brepack_line(d):
    """One sentence: the file, its size, when it was built, and its key."""
    return ("%s, %.2f GiB (%d B), built %s, alt=%s, key16=%s, reap_s=%s" %
            (d["name"], d["bytes"] / 2**30, d["bytes"], d["built"] or "?",
             d["alt_kind"], d["key16"],
             (d.get("hdr") or {}).get("repack_s", "?")))


def pf_prewarm(ctx):
    """P09 -- the artifact EXISTS and is REUSED.  Two different numbers.

    The 33.0 s that used to be quoted as "33 s of the job's wall" is the
    artifact's OWN build time recorded in its manifest (\"pack_s\": 33.035 for
    wpack_l30_v2.bin, built 2026-11-02T15:22:06) -- a HISTORIC constant that a
    reusing job never pays.  Measured 2026-11-03: the job's own cost of a
    present artifact is map_s 0.001 s + verify_s 1.1-2.1 s (webui/runner.py
    838-845 books the pack WALL to prewarm_pack_wall_s, which is not a bucket).
    Only an ABSENT artifact costs the job the pack (25-32 s, booked to
    prewarm_pack_s because the job blocks on it).
    """
    from webui import config
    f = config.WPACK_FILE
    parts, why, status = [], "", "PASS"
    src = ["webui/config.py:195-213, webui/runner.py:838-845"]
    if f.is_file():
        st = f.stat()
        meta = {}
        try:
            mp = f.with_suffix(f.suffix + ".json")
            if mp.is_file():
                meta = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        parts.append("wpack artifact %s, %.2f GiB, built %s, pack_s=%s (the BUILD time, "
                     "not this job's) -> map+verify only (~1-2 s)"
                     % (f.name, st.st_size / 2**30,
                        time.strftime("%Y-%m-%d %H:%M",
                                      time.localtime(st.st_mtime)),
                        meta.get("pack_s", "?")))
    else:
        status = "WARN"
        parts.append("wpack artifact %s ABSENT: THIS job packs it (25-32 s, booked to "
                     "prewarm_pack_s, outside the loop's wall)" % f.name)
        why = "not fatal, but it moves ~30 s into the job's wall"
    # ---- the second artifact: the B repack, as a map --------------------
    d = brepack_state()
    ctx["brepack"] = d
    src.append("brepack_cache.py:675-772")
    arms = [a for a in (ctx.get("arms") or []) if a[0] == "T22_BREPACK_CACHE"]
    if not d["enabled"]:
        parts.append("B-repack cache DISABLED (T22_BREPACK_CACHE=%r)"
                     % os.environ.get("T22_BREPACK_CACHE"))
        if arms:
            # the recorded negative control: this run is the un-mapped arm of
            # the pair, and Q18 is expected to say so.
            status = "WARN"
            why = ("recorded ARM: the cache is deliberately off, so THIS run "
                   "pays the 5.31 GB permutation -- that is the measurement, "
                   "not a fault.  Q18 will FAIL on it by design (that is the "
                   "check proving it can see a miss); the number is a BASELINE "
                   "for the mapped arm, never a reference")
        else:
            status = "FAIL"
            why = ("the recipe sets it to 1: without it the run pays the 5.31 GB "
                   "permutation again.  There is no recorded arm that explains "
                   "this, so it is a recipe error -- P06 should have caught it")
    elif d["ok"]:
        parts.append("B-repack cache %s -> the run can MAP it"
                     % brepack_line(d))
    else:
        status = "FAIL"
        parts.append("B-repack cache NOT REUSABLE: %s [wanted %s, %d B]"
                     % (d["reason"], d["name"] or "-", d["want_bytes"]))
        why = ("a cache that silently misses is indistinguishable from no "
               "cache: the run would re-derive the 5.31 GB permutation "
               "(27.02 s of the 45.64 s AC startup, 59% of it) and Q18 would "
               "then refuse the number.  Build it ONCE, on purpose, and time "
               "it:  py312 imaggers\\repro\\t22_repro.py cache --build   "
               "(-Skip P09 is the only legitimate way past this check: it "
               "measures the un-mapped arm)")
    return R("P09", "wpack artifact + B-repack cache (both present and reusable)",
             status, " | ".join(parts), why, "; ".join(src))


def pf_power(ctx):
    """P12 -- the power state, recorded, and never a reason to refuse a run.

    A WARN on battery is the whole point: the number is still real, it is just
    not comparable to an AC number.  Refusing to run would hide that.
    """
    state, d = power_state()
    ctx["power"] = d
    out("[power] " + power_line(d))
    if state == "ac":
        return R("P12", "power state is AC (numbers comparable to references)",
                 "PASS", power_line(d), "", POWER_PROBE)
    if state == "battery":
        return R("P12", "power state is AC (numbers comparable to references)",
                 "WARN", power_line(d),
                 "on battery the SAME recipe with the SAME latents measured "
                 "startup_s 68.39 vs 45.64 s on AC (x1.50), loop_s 99.87 vs "
                 "92.87 (x1.075), total_s 186.19 vs 150.94 (x1.23) on "
                 "2026-11-03.  A number taken now is real but is NOT comparable "
                 "to an AC reference: compare WITHIN a back-to-back pair only",
                 POWER_PROBE)
    return R("P12", "power state is AC (numbers comparable to references)",
             "WARN", power_line(d) + " (%s)" % (d.get("err") or "unprobeable"),
             "the power state could not be read, so the magnitude of every "
             "number in this record is unknown -- say so wherever you quote it",
             POWER_PROBE)


def pf_disk(ctx):
    u = shutil.disk_usage(str(ROOT))
    free_gb = u.free / 2**30
    if free_gb < 8:
        return R("P10", "free disk for images + PNGs", "WARN",
                 "%.1f GiB free on the repo volume (< 8 GiB)" % free_gb, "", "")
    return R("P10", "free disk for images + PNGs", "PASS",
             "%.1f GiB free" % free_gb, "", "")


def pf_prior(ctx):
    """P11 -- the last run of THIS recipe+shape, not the last run of anything.

    The old title said "previous run of this recipe was trusted" while the code
    read runs[-1] of last/index.json, i.e. whatever ANY agent recorded last --
    another shape, another recipe, another tag.  A cold reader cannot tell those
    apart from the PASS line, so the selection is now the recipe's env_id AND the
    requested shape+steps, and the run it picked (or the fact that there is none)
    is printed.  The globally-last run is still quoted, as a courtesy, but it can
    no longer be mistaken for this recipe's predecessor.
    """
    title = "last run of this recipe+shape proved its path"
    idx = RECORDS / "index.json"
    if not idx.is_file():
        return R("P11", title, "SKIP",
                 "no run has ever been recorded in imaggers/repro/last/", "", "")
    try:
        d = json.loads(idx.read_text(encoding="utf-8"))
    except Exception:
        return R("P11", title, "SKIP", "last/index.json is unreadable", "", "")
    runs = d.get("runs") or []
    if not runs:
        return R("P11", title, "SKIP", "last/index.json has no rows", "", "")
    want_env = ctx.get("env_id") or env_id()
    want = (int(ctx.get("h") or 0), int(ctx.get("w") or 0),
            int(ctx.get("steps") or 0))
    same = [r for r in runs
            if r.get("env_id") == want_env
            and (r.get("h"), r.get("w"), r.get("steps")) == want]
    if same:
        prev = same[-1]
        if prev.get("postflight_fail", 0) > 0:
            return R("P11", title, "WARN",
                     "the last run of THIS recipe+shape, %s (%s), FAILED %d "
                     "check(s): %s" %
                     (prev.get("tag"), prev.get("when"),
                      prev.get("postflight_fail"), prev.get("failed_ids")),
                     "a number from a run that did not prove its path is not a "
                     "reference; Q12's comparison is against the reference "
                     "TABLE, not against that run", "")
        detail = ("%s (%s) proved its path: 0 FAIL at env_id %s, %dx%d x%d" %
                  (prev.get("tag"), prev.get("when"), want_env, want[1],
                   want[0], want[2]))
        if len(same) > 1:
            detail += " (%d runs of this recipe+shape recorded)" % len(same)
        return R("P11", title, "PASS", detail, "", "")
    last = runs[-1]
    detail = ("no run of THIS recipe+shape has been recorded yet (env_id %s, "
              "%dx%d x%d).  The last run of anything was %s (%s, %dx%d x%d, "
              "env_id %s)" %
              (want_env, want[1], want[0], want[2], last.get("tag"),
               last.get("when"), last.get("w"), last.get("h"),
               last.get("steps"), last.get("env_id")))
    if last.get("postflight_fail", 0) > 0:
        detail += " and it FAILED %d check(s): %s" % (
            last.get("postflight_fail"), last.get("failed_ids"))
    return R("P11", title, "SKIP", detail,
             "nothing to compare with is not a failure -- Q12 RECORDS the "
             "reference on this run's first use, and compares from then on", "")


PF_CHECKS = [
    ("P01", "repo tree + both interpreters", pf_repo),
    ("P02", "device actually idle (processes, not the lock)", pf_idle),
    ("P03", "lock file: absent, or stale-with-dead-owner", pf_lock),
    ("P04", "memory mode will be 'fast' (proved, not read)", pf_memory),
    ("P05", "size is renderable per webui/capability.py", pf_renderable),
    ("P06", "the job env is EXACTLY the recipe", pf_env),
    ("P07", "thread counts the product will apply", pf_threads),
    ("P08", "every device image the path registers exists", pf_images),
    ("P09", "wpack artifact + B-repack cache (both present and reusable)",
     pf_prewarm),
    ("P10", "free disk for images + PNGs", pf_disk),
    ("P11", "last run of this recipe+shape proved its path", pf_prior),
    ("P12", "power state is AC (numbers comparable to references)", pf_power),
]

PF_WHY = {
    "P01": "a wrong path fails before anything else can be trusted",
    "P02": "the device is shared and has no real mutex; two users = 0xc01e0009",
    "P03": "a crashed run leaves a lock that blocks the next launcher",
    "P04": "low mode is 2x slower AND a different latent baseline",
    "P05": "a refused size dies at submit with an unhelpful HTTP 400",
    "P06": "an inherited switch is a different device path",
    "P07": "loop and VAE threads are coupled unless both are pinned",
    "P08": "a missing image looks like a code bug 60 s into the run",
    "P09": "an ABSENT wpack costs the job ~30 s (a present one is map+verify, "
           "~1-2 s); an UNREUSABLE B-repack cache costs it 27 s -- the largest "
           "single row of startup_s -- and looks exactly like a cache that "
           "works, so it is a refusal and Q18 is the in-log proof",
    "P10": "a rebuild of the 5.31 GB B-repack cache and the PNGs need room",
    "P11": "a number from an unproven run is not a reference -- and the run "
           "it reads must be THIS recipe+shape (env_id, size, steps), not "
           "whatever happened to be recorded last by any agent",
    "P12": "on battery the same recipe measured startup_s x1.50 and total_s "
           "x1.23 of AC (loop only x1.075); a WARN, never a refusal -- the "
           "number is real, just not comparable",
}


# ---------------------------------------------------------------------------
# 8. the run: one server, one job, no manual env editing
# ---------------------------------------------------------------------------
def http_json(url, body=None, timeout=30):
    import urllib.error
    import urllib.request
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            payload = exc.read().decode("utf-8", "replace")
        except Exception:
            payload = ""
        raise RuntimeError("HTTP %s from %s: %s" %
                           (exc.code, url, payload[:600])) from None


def safe_tag(tag):
    """-Tag -> the string that names every record file and the job prefix.

    The rules (RUNBOOK section 1): charset A-Za-z0-9._-, every other character
    becomes '-', leading/trailing '-' are stripped, the result is cut to
    TAG_MAX; an empty tag becomes 'run' (the caller substitutes t<HHMMSS>
    BEFORE this call, so the normal case never reaches that branch).
    """
    t = TAG_RE.sub("-", str(tag or "run")).strip("-")
    return (t or "run")[:TAG_MAX]


def tag_records(tag):
    """The record files this tag already owns ([] = a fresh tag)."""
    RECORDS.mkdir(parents=True, exist_ok=True)
    return sorted(f.name for f in RECORDS.glob(tag + ".*") if f.is_file())


def announce_tag(requested, tag):
    """Say what the tag BECAME, and whether reusing it overwrites records.

    Silent normalisation is how a run gets filed under a name its author did not
    choose; a silent overwrite is how two runs end up sharing one row of
    last/index.json (save_run keeps one row per tag).  Both were things a cold
    reader had to infer, so both are printed.
    """
    if str(requested or "") != tag:
        out("[tag] %r -> %r (charset A-Za-z0-9._-, every other char -> '-', "
            "max %d)" % (requested, tag, TAG_MAX))
    else:
        out("[tag] %s  (records: imaggers/repro/last/%s.*.json, job prefix %s%s)"
            % (tag, tag, TAG_PREFIX, tag))
    old = tag_records(tag)
    if old:
        out("[tag] WARNING %d record file(s) already exist for this tag and "
            "will be OVERWRITTEN: %s" % (len(old), ", ".join(old[:6])))
        out("[tag]         use a FRESH -Tag per run: last/index.json keeps one "
            "row per tag, so a reused tag replaces the older run's history")
    return tag


def start_server(tag):
    RECORDS.mkdir(parents=True, exist_ok=True)
    so = open(RECORDS / (tag + ".server.out.log"), "wb")
    se = open(RECORDS / (tag + ".server.err.log"), "wb")
    p = subprocess.Popen([str(_need(PY312, "the Python 3.12 interpreter", "ZIMAGE_PY312")), str(LAUNCHER)], cwd=str(ROOT),
                         stdout=so, stderr=se, stdin=subprocess.DEVNULL,
                         env=dict(os.environ))
    return p, so, se


def wait_server(p, timeout_s=180):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if p.poll() is not None:
            return False, "server exited rc=%s before listening" % p.returncode
        try:
            http_json(URL + "/api/status", timeout=5)
            return True, ""
        except Exception:
            time.sleep(0.5)
    return False, "server did not answer /api/status in %ss" % timeout_s


def release_lock(pid):
    """Delete the F-14.1 lock THIS run took, and only after its owner is dead.

    launch_webui_bigm.py takes the lock and never releases it (run_job_py.py
    deletes it in its own finally; this engine does not use run_job_py).  Left
    alone, every run would leave a stale lock behind -- the exact pathology
    this file exists to remove.
    """
    present, txt, owner, lpid, alive = lock_state()
    if not present:
        return "absent"
    if lpid != pid:
        return "held by another pid %s -- NOT touched" % lpid
    if alive:
        return "owner pid %s still alive -- NOT touched" % lpid
    try:
        LOCK.unlink()
        return "released (pid %s, owner provably dead)" % pid
    except OSError as exc:
        return "could not release: %s" % exc


def device_line(d):
    """One sentence: how the device looked when a run ended."""
    bits = ["lock " + (d.get("lock") or "not recorded")]
    users = d.get("users") or []
    if d.get("busy"):
        bits.append("%d other device process(es) still on the NPU%s"
                    % (len(users), (": " + ", ".join(
                        "%s %s" % (u.get("pid"), u.get("name")) for u in users[:3]))
                       if users else ""))
    else:
        bits.append("no other device process")
    bits.append("port %d %s" % (PORT, "OPEN" if d.get("port_open") else "free"))
    return "; ".join(bits)


def write_device_state(tag, dev):
    """last/<tag>.device.json -- the post-run device state, readable later."""
    try:
        RECORDS.mkdir(parents=True, exist_ok=True)
        (RECORDS / (tag + ".device.json")).write_text(
            json.dumps(dev, indent=1, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def _reap():
    """webui/reap.py, or None when this tree predates it."""
    try:
        from webui import reap
        return reap
    except Exception:                                    # noqa: BLE001
        return None


def stop_server(p, so, se):
    """Stop the server AND the tree below it.

    p.terminate() is TerminateProcess: the py310 bridge (1.8-7.3 GB)
    that the server owns is orphaned and keeps the device.  That is
    what made rq_1080p_8e fail twice on the same 1.24 GiB allocate and
    what turned clean MemoryErrors into 0xC0000005 in the retry
    spiral.  reap.reap() walks the tree first; the server also owns a
    KILL_ON_JOB_CLOSE job (webui/reap.py), so a server that was already
    hard-killed still takes its children with it.
    """
    for stream in (so, se):
        try:
            stream.close()
        except Exception:
            pass
    if p.poll() is not None:
        return
    r = _reap()
    if r is not None:
        out("[run] stopping the server tree: %d descendant(s) at kill "
            "time" % len(r.descendants(p.pid)))
        res = r.reap(p, why="server", timeout=20)
        if res["survivors"]:
            out("[run] WARNING %d process(es) survived the reap: %s"
                % (len(res["survivors"]), res["survivors"]))
        return
    p.terminate()
    try:
        p.wait(timeout=20)
    except Exception:
        p.kill()
        try:
            p.wait(timeout=10)
        except Exception:
            pass


def run_job(ctx):
    """Spawn the known-good server, submit one job, follow it to the end."""
    tag = ctx["tag"]
    plan = ctx["plan"]
    steps = ctx["steps"]
    prefix = safe_tag("repro_" + tag)
    env_body = {"height": plan["h"], "width": plan["w"], "steps": steps,
                "seed": SEED, "vae": True, "mode": "trial", "prompt": PROMPT,
                "prefix": prefix, "memory_mode": "fast"}
    hr("=")
    out("RUN  %s  %dx%d  steps=%d  env_id=%s" %
        (tag, plan["h"], plan["w"], steps, ctx["env_id"]))
    out("     body: %s" % json.dumps(env_body, ensure_ascii=False))
    hr("=")
    p, so, se = start_server(tag)
    out("[run] server pid %d -> %s" % (p.pid, RECORDS / (tag + ".server.err.log")))
    ok, why = wait_server(p)
    if not ok:
        stop_server(p, so, se)
        raise SystemExit("[run] %s (see %s)" % (why, RECORDS / (tag + ".server.err.log")))
    present, txt, owner, lpid, alive = lock_state()
    if present and lpid == p.pid:
        out("[run] F-14.1 lock owner=pid %d (this server) purpose=%r" %
            (p.pid, (re.search(r"purpose=(\S+)", txt) or [None, None])[1]))
    elif present:
        out("[run] WARNING lock says pid %s but the server is %d" % (lpid, p.pid))
    else:
        out("[run] WARNING no device.lock after start (the launcher did not take it)")
    dev = {"lock": "", "busy": None, "waited_s": 0.0, "users": [],
           "port_open": None}
    try:
        r = http_json(URL + "/api/jobs", env_body, timeout=60)
        job = r.get("job") or {}
        jid = job.get("id")
        if not jid:
            raise RuntimeError("no job id in %r" % (r,))
        out("[run] job %s accepted (status=%s)" % (jid, job.get("status")))
        t0 = time.time()
        limit = max(900.0, steps * 150.0 + 600.0)
        last = None
        while True:
            time.sleep(2)
            j = http_json(URL + "/api/jobs/" + jid, timeout=30)["job"]
            st = (j.get("status"), j.get("stage"), j.get("step"))
            if st != last:
                out("[run] t=%4.0fs status=%-8s stage=%-18s step=%s" %
                    (time.time() - t0, j.get("status"), j.get("stage"),
                     j.get("step")))
                last = st
            if j.get("status") in ("done", "error", "cancelled", "failed"):
                break
            if time.time() - t0 > limit:
                out("[run] TIMEOUT after %.0fs (limit %.0fs)" % (time.time() - t0, limit))
                break
        (RECORDS / (tag + ".job.json")).write_text(
            json.dumps(j, indent=1, ensure_ascii=False), encoding="utf-8")
        t = j.get("timing") or {}
        out("[run] FINAL %s total_s=%s loop_s=%s steps=%s latents=%s" %
            (j.get("status"), t.get("total_s"), t.get("loop_s"),
             len(j.get("steps_rec") or []), (j.get("latents_sha256") or "")[:16]))
    finally:
        stop_server(p, so, se)
        dev["lock"] = release_lock(p.pid)
        out("[run] server stopped; device.lock: %s" % dev["lock"])
        ok, waited, users = wait_quiet(deadline_s=120, quiet_s=3, say=False)
        dev["waited_s"] = round(waited, 1)
        dev["busy"] = not ok
        dev["users"] = [{"pid": u["pid"], "name": u["name"]} for u in users]
        # A device process still alive here can only be OURS: the
        # preflight proved the box was quiet before this run started.
        # Kill it instead of reporting it, because the next attempt is
        # the thing that pays for it.
        dev["leftover_killed"] = []
        if not ok:
            r = _reap()
            for u in users:
                out("[run] leftover device process pid %s (%s, rss %s GB) "
                    "-- killing it rather than letting the next attempt "
                    "pay for it" % (u["pid"], u["name"], u["rss_gb"]))
                try:
                    if r is not None:
                        r.kill_tree(u["pid"])
                    else:
                        import psutil
                        psutil.Process(u["pid"]).kill()
                    dev["leftover_killed"].append(u["pid"])
                except Exception as exc:                 # noqa: BLE001
                    out("[run] could not kill pid %s: %r" % (u["pid"], exc))
            ok2, waited2, users2 = wait_quiet(deadline_s=60, quiet_s=3,
                                              say=False)
            dev["busy"] = not ok2
            dev["users"] = [{"pid": u["pid"], "name": u["name"]}
                            for u in users2]
            dev["waited_s"] = round(waited + waited2, 1)
            out("[run] after the reap: %s" % device_line(dev))
        try:
            dev["port_open"] = bool(port_busy())
        except Exception:
            dev["port_open"] = None
        if not ok:
            out("[run] WARNING device still busy %.0fs after the server stopped: %s"
                % (waited, [(u["pid"], u["name"]) for u in users]))
        ctx["device_after"] = dev
        write_device_state(tag, dev)
        out("[run] post-run device state: %s" % device_line(dev))
        if dev["busy"] or "another pid" in str(dev["lock"]) or "still alive" in str(dev["lock"]):
            out("[run] NOTE that state is NOT a failure of this job: the "
                "postflight reads files only, so run it now; section 9 of the "
                "RUNBOOK lists every post-run device state")
    ctx["job_id"] = jid
    return jid


# ---------------------------------------------------------------------------
# 9. POSTFLIGHT -- which path actually ran, from the finished job's own log
# ---------------------------------------------------------------------------
# chain=<n> was added by the two-chain work (T22_CHAIN2B); the suffix is
# optional so a single-chain line still parses, but its ABSENCE must not be
# mistaken for "install() was never reached", which is what this check exists
# to catch.  Getting this regex wrong makes the check cry wolf on every run.
CG_INSTALLED = re.compile(
    r"\[fastglue\] installed (\d+)/(\d+) hooks \(threads=(\d+)"
    r"(?: chain=(\d+))?\)")
CG_CALL = re.compile(r"\[fastglue\] (layers\.\d+) proj_qkv")
DK_ROW = re.compile(r"\[devkill\] rows=(\d+) pack=")
FFNPROF = re.compile(r"\[ffnprof\] seq=(\d+)")
MEMLINE = re.compile(r"\[webui\] memory mode: (\w+) \(([^)]*)\); avail ([\d.]+) GB, commit ([\d.]+) GB")
LOOPLINE = re.compile(r"\[webui\] loop complete: ([\d.]+) s")
# The --cache-all site has FOUR shapes, and only the first allocates anonymous
# RAM: mapped (B repack cache), already resident (resident worker / preload
# "first"), worker thread, and the inline allocation.  A check that knew only
# the last one FAILED a run that was MORE resident than the baseline (measured:
# the first mapped 512^2 arm), which is the kind of false negative this file
# exists to prevent.
CACHED = re.compile(
    r"\[t13\] cache-all: (?:([\d.]+) GB free before allocation"
    r"|B payloads mapped from (\S+)"
    r"|payload already resident \(([^)]*)\)"
    r"|payload preload is on a worker thread)")
REGISTERED = re.compile(r"\[t3d\] registered (\d+) images: (.+)")
VAELOADER = re.compile(r"\[vae\] resident decoder loaded in [\d.]+ s .*\[(\w+)\]")
VAETHREADS = re.compile(r"\[vae\] decode threads: torch intra-op (\d+)")
CHUNKPATH = re.compile(r"\[t18\] chunk_path installed: (.+?) \(proj/ffn/o")
FAULTS = [
    ("state 8", "ERT watchdog fired: a second back-to-back launch of the same "
                "xclbin never completes (F-01).  The barrier launch is missing"),
    ("ERT_CMD_STATE_TIMEOUT", "same as 'state 8' (F-01/F-04)"),
    ("attempting to write past buffer size",
     "the fastglue projection BO holds 1632 rows (T22_ADAPT_CHUNK); M is larger"),
    ("ran out of memory", "the device loop ran out of memory"),
    ("MemoryError", "a host allocation failed -- the memory mode is not what "
                    "the preflight proved"),
    ("TimeoutError", "a dispatch or a wait timed out (F-04: ~7.0 s watchdog)"),
    ("Traceback (most recent call last)", "a python exception reached the log"),
]


def _first(rx, text, group=1):
    m = rx.search(text)
    return m.group(group) if m else None


def load_run_record(tag):
    f = RECORDS / (tag + ".job.json")
    if not f.is_file():
        raise SystemExit("no record %s -- run the job first (action run)" % f)
    return json.loads(f.read_text(encoding="utf-8"))


def preflight_record(tag):
    """The preflight record of THIS tag, or {} (a postflight may run later)."""
    f = RECORDS / (tag + ".preflight.json")
    if not f.is_file():
        return {}
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return d if isinstance(d, dict) else {}


def pf_ctx(tag):
    j = load_run_record(tag)
    jid = j.get("id")
    pjf = JOBS / str(jid) / "job.json"
    logf = JOBS / str(jid) / "job.log"
    if not pjf.is_file() or not logf.is_file():
        raise SystemExit("product records missing for job %s (%s)" % (jid, pjf))
    pj = json.loads(read_text(pjf))
    log = read_text(logf)
    req = pj.get("requested") or {}
    h, w = int(req.get("height") or 0), int(req.get("width") or 0)
    # the env the run ACTUALLY had: an --extra arm changes it, and the record
    # must say so rather than stamping the recipe's id onto an arm
    pre = {}
    pf = RECORDS / (tag + ".preflight.json")
    if pf.is_file():
        try:
            pre = json.loads(pf.read_text(encoding="utf-8")) or {}
        except Exception:
            pre = {}
    dev = {}
    df = RECORDS / (tag + ".device.json")
    if df.is_file():
        try:
            dev = json.loads(df.read_text(encoding="utf-8")) or {}
        except Exception:
            dev = {}
    ctx = {"tag": tag, "job_id": jid, "driver": j, "product": pj, "log": log,
           "h": h, "w": w, "steps": int(pj.get("steps") or 0),
           "env_id": pre.get("env_id") or env_id(),   # the RECIPE (or the arm)
           "extras": pre.get("extras") or [],
           "device_after": dev if isinstance(dev, dict) else {},
           "job_dir": JOBS / str(jid)}
    ctx["plan"] = plan_for(h, w) if h and w else {"path": "?", "m": 0}
    return ctx


def q_status(ctx):
    j, pj = ctx["driver"], ctx["product"]
    if pj.get("status") == "done":
        t = pj.get("timing") or {}
        return R("Q01", "job finished 'done'", "PASS",
                 "total %.1f s (loop %.1f s), %d steps" %
                 (t.get("total_s") or 0, t.get("loop_s") or 0, ctx["steps"]),
                 "", "webui/jobs/<id>/job.json:status")
    err = pj.get("error")
    msg = (err.get("message") if isinstance(err, dict) else str(err or ""))[:400]
    hint = pj.get("hint") or ""
    first = ""
    for line in ctx["log"].splitlines():
        if line.startswith(("RuntimeError:", "ValueError:", "AssertionError:",
                            "ShapeError:", "[planffn] REFUSAL")):
            first = line.strip()
            break
    return R("Q01", "job finished 'done'", "FAIL",
             "status=%r error=%r hint=%r %s" %
             (pj.get("status"), msg, hint, first), "", "job.json:status/error")


def q_path(ctx):
    p, log = ctx["plan"], ctx["log"]
    m_chunk = _first(re.compile(r"\[t18\] CHUNKED SEQ=(\d+)"), log)
    m_nat = _first(re.compile(r"\[t18\] SEQ=(\d+)"), log)
    seen = "chunked" if m_chunk else ("native" if m_nat else "none")
    got_m = int(m_chunk or m_nat or 0)
    detail = "log says path=%s SEQ=%s; capability says path=%s m=%d" % (
        seen, got_m or "-", p["path"], p["m"])
    if seen == "none":
        return R("Q02", "the shape took the path the plan named", "FAIL",
                 detail + " (no [t18] shape line at all)",
                 "every number below belongs to a path; without this anchor the "
                 "run is unattributable", "imaggers/image/t18_run.py:3256-3261")
    if seen != p["path"]:
        return R("Q02", "the shape took the path the plan named", "FAIL", detail,
                 "a chunked/native switch changes the latency and the image set",
                 "webui/shapes.py:937+, webui/capability.py:195-200")
    if got_m != p["m"]:
        return R("Q02", "the shape took the path the plan named", "FAIL",
                 detail + " (M mismatch)",
                 "M is the shape contract: projection blocks, FFN rows, the "
                 "refiner's score matrix", "webui/shapes.py:570-677")
    return R("Q02", "the shape took the path the plan named", "PASS", detail,
             "", "imaggers/image/t18_run.py:3256-3261")


def q_chunkpath(ctx):
    if ctx["plan"]["path"] != "chunked":
        return R("Q03", "chunk_path description names the big-M projector",
                 "SKIP", "STRUCTURALLY N/A: a native shape installs no "
                 "chunk_path (webui/chunked.py:932 install() is the chunked "
                 "branch only)", "", "")
    got = _first(CHUNKPATH, ctx["log"])
    want = "big-M whole-shape proj (rt17) + host-f32-torch"
    if got is None:
        return R("Q03", "chunk_path description names the big-M projector",
                 "FAIL", "no '[t18] chunk_path installed:' line",
                 "without it the loop ran the delivered 96-row projection, not "
                 "the whole-shape one", "imaggers/image/t18_run.py:3259")
    if got != want:
        return R("Q03", "chunk_path description names the big-M projector",
                 "FAIL", "installed %r, expected %r" % (got, want),
                 "a different install means a different dispatch order",
                 "webui/chunked.py:963-1029")
    return R("Q03", "chunk_path description names the big-M projector", "PASS",
             "installed: %s" % got, "", "imaggers/image/t18_run.py:3259")


def q_cg_installed(ctx):
    if ctx["plan"]["path"] != "chunked":
        return R("Q04", "fastglue 8/8 hooks installed", "SKIP",
                 "STRUCTURALLY N/A: fastglue is installed only inside the "
                 "chunked branch (webui/chunked.py:988-1012), so a native run "
                 "has no [fastglue] line to read", "", "")
    log = ctx["log"]
    if "[fastglue] skipped:" in log:
        m = re.search(r"\[fastglue\] skipped: (.+)", log)
        return R("Q04", "fastglue 8/8 hooks installed", "FAIL",
                 "install raised: %s" % (m.group(1)[:200] if m else "?"),
                 "the run then proceeds on the SLOWER delivered host path and "
                 "still 'works'", "webui/chunked.py:995-998")
    if "REFUSED (strict)" in log:
        return R("Q04", "fastglue 8/8 hooks installed", "FAIL",
                 "install refused itself (T22_FASTGLUE_STRICT): %s" %
                 (_first(re.compile(r"\[fastglue\] REFUSED \(strict\): (.+)"), log) or "")[:200],
                 "a gate failed; the hooks are NOT installed",
                 "imaggers/util/glue/install.py:611-624")
    m = CG_INSTALLED.search(log)
    if not m:
        return R("Q04", "fastglue 8/8 hooks installed", "FAIL",
                 "no '[fastglue] installed N/8' line",
                 "T22_FASTGLUE (no underscore) is not set, or install() was "
                 "never reached.  'installed 0/8' costs +37 s and looks like a "
                 "working run", "imaggers/util/glue/install.py:591,869")
    n, tot, thr = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if n != tot or tot != 8:
        return R("Q04", "fastglue 8/8 hooks installed", "FAIL",
                 "installed %d/%d hooks (threads=%d): %s" %
                 (n, tot, thr,
                  _first(re.compile(r"\[fastglue\] installed .*?: (.+)"), log) or ""),
                 "a partial install is the silent-slower case this file exists "
                 "to catch", "imaggers/util/glue/install.py:869")
    if thr != int(recipe_env()["T22_FASTGLUE_THREADS"]):
        return R("Q04", "fastglue 8/8 hooks installed", "FAIL",
                 "installed %d/%d but threads=%d, recipe says %s" %
                 (n, tot, thr, recipe_env()["T22_FASTGLUE_THREADS"]),
                 "the fused kernels' thread count is a byte-path input",
                 "imaggers/util/glue/install.py:597")
    return R("Q04", "fastglue 8/8 hooks installed", "PASS",
             "installed %d/%d hooks (threads=%d)" % (n, tot, thr), "",
             "imaggers/util/glue/install.py:869")


def q_cg_called(ctx):
    if ctx["plan"]["path"] != "chunked":
        return R("Q05", "fastglue hooks WERE CALLED (per-layer lines)", "SKIP",
                 "STRUCTURALLY N/A: no fastglue install on the native path, so "
                 "there are no per-layer call lines -- this is NOT evidence that "
                 "the loop is slower, and it is not a pass either", "", "")
    calls = CG_CALL.findall(ctx["log"])
    layers = sorted({int(x.split(".")[1]) for x in calls})
    want = ctx["steps"] * LAYERS
    detail = "%d call lines, %d distinct layers; expected %d (%d steps x %d)" % (
        len(calls), len(layers), want, ctx["steps"], LAYERS)
    if not calls:
        return R("Q05", "fastglue hooks WERE CALLED (per-layer lines)", "FAIL",
                 detail + " -- installed but never called",
                 "'installed 8/8' is not the same as 'the fused path ran'; this "
                 "is the exact failure the whole file exists for",
                 "imaggers/util/glue/install.py:536-539")
    if len(calls) != want:
        return R("Q05", "fastglue hooks WERE CALLED (per-layer lines)", "FAIL",
                 detail, "the call count is the layer x step contract",
                 "imaggers/util/glue/install.py:536-539")
    if len(layers) != LAYERS:
        return R("Q05", "fastglue hooks WERE CALLED (per-layer lines)", "FAIL",
                 detail + " (layer ids %d..%d)" % (layers[0], layers[-1]),
                 "a missing layer id means a partial hook set",
                 "imaggers/util/glue/install.py:536-539")
    return R("Q05", "fastglue hooks WERE CALLED (per-layer lines)", "PASS",
             detail, "", "imaggers/util/glue/install.py:536-539")


def q_ffn_called(ctx):
    if ctx["plan"]["path"] != "chunked":
        return R("Q06", "FFN: devkill rows + ffnchain calls", "SKIP",
                 "STRUCTURALLY N/A: the [devkill] rows lines DO appear on the "
                 "native path (30 per step), but the [ffnprof] half of this "
                 "check is the ffnchain hook, which only the fastglue install "
                 "adds -- and that install is chunked-only", "", "")
    log = ctx["log"]
    rows = [int(x) for x in DK_ROW.findall(log)]
    prof = [int(x) for x in FFNPROF.findall(log)]
    want = ctx["steps"] * LAYERS
    m = ctx["plan"]["m"]
    # The FFN chunks at its own block size (1056 = RT=11 tiles) for every shape
    # above it, so the old invariant "every line says rows=M, and there are
    # exactly steps*layers of them" was true only while M <= the block.  The
    # invariant that still catches a missing layer or a wrong row count is:
    # the lines PARTITION M per layer -- len(rows) is (steps*layers*blocks) and
    # the block count equals ceil(M / max(rows)).
    max_rows = max(rows) if rows else 0
    nblk = (m + max_rows - 1) // max_rows if max_rows else 0
    bad_rows = sorted({r for r in rows if r > m or r <= 0})
    detail = ("%d [devkill] rows lines (rows=%s, blocks/layer=%d), %d "
              "[ffnprof] lines; expected %d lines at %d block(s) of <=%d "
              "covering M=%d" %
              (len(rows), sorted(set(rows)) or "-", nblk, len(prof),
               want * nblk, nblk, max_rows or 0, m))
    if len(rows) != want * nblk or nblk < 1:
        return R("Q06", "FFN: devkill rows + ffnchain calls", "FAIL",
                 detail + " -- the devkill FFN did not run every layer",
                 "T22_BIGM_DEVHANDOFF=1 is required for the k=32 B payload; "
                 "without it prod_loop_bigm hands out {n128,n64} and "
                 "devkill_ffn raises 'devkill needs n32 B payload'",
                 "imaggers/devkill/devkill_ffn.py:325-332, "
                 "imaggers/block/bigm/prod_loop_bigm.py:647-649")
    if bad_rows:
        return R("Q06", "FFN: devkill rows + ffnchain calls", "FAIL",
                 detail + " (rows %s != M %d)" % (bad_rows, ctx["plan"]["m"]),
                 "the FFN ran at the wrong row count",
                 "imaggers/devkill/devkill_ffn.py:300")
    if len(prof) != want:
        return R("Q06", "FFN: devkill rows + ffnchain calls", "FAIL",
                 detail + " -- no ffnchain call evidence",
                 "T22_FFN_PROF=1 is what makes this hook provable.  With the "
                 "recipe applied, 0 lines means fast_bigm_ffn did not run; on "
                 "an older job it means the flag was never set (Q11 says which)",
                 "imaggers/util/glue/install.py:757-808")
    if "[devkill] prewarm: 120 G=15 projection images packed" not in log:
        return R("Q06", "FFN: devkill rows + ffnchain calls", "WARN",
                 detail + "; no prewarm line (T22_DEVKILL_PREWARM=0?)",
                 "without the prewarm step 0 pays the packing inside the loop",
                 "webui/chunked.py:999-1007")
    return R("Q06", "FFN: devkill rows + ffnchain calls", "PASS", detail, "",
             "imaggers/devkill/devkill_ffn.py:325-332 + install.py:757-808")


def q_memory(ctx):
    pj, log = ctx["product"], ctx["log"]
    m = MEMLINE.search(log)
    mm = pj.get("memory_mode")
    cache = CACHED.search(log)
    detail = ("job.json memory_mode=%r; log line=%s; [t13] cache-all=%s" %
              (mm, ("%r (%s), commit %s GB" % (m.group(1), m.group(2), m.group(4)))
               if m else "MISSING",
               (cache.group(0).strip()[len("[t13] cache-all: "):]
                if cache else "MISSING")))
    if mm != "fast" or (m and m.group(1) != "fast"):
        return R("Q07", "memory mode actually used is 'fast'", "FAIL", detail,
                 "low mode (--no-cache-all) is ~2x slower AND a different latent "
                 "baseline: an A/B that mixes modes is worthless",
                 "webui/config.py:546-556, webui/runner.py:492-527")
    if not m:
        return R("Q07", "memory mode actually used is 'fast'", "FAIL", detail,
                 "the product prints this line per job; its absence means the "
                 "log is not from the run you think", "webui/runner.py:496")
    if not cache:
        return R("Q07", "memory mode actually used is 'fast'", "FAIL", detail,
                 "fast mode IS the cache-all pass: no '[t13] cache-all:' line "
                 "of ANY shape (allocated, mapped, already resident, threaded) "
                 "means the 3.71 GB resident set was never built",
                 "imaggers/image/t18_run.py:2674-2680")
    if "falling back to --no-cache-all" in log:
        return R("Q07", "memory mode actually used is 'fast'", "FAIL", detail +
                 " + 'falling back to --no-cache-all'",
                 "the fast attempt died with MemoryError and the run silently "
                 "continued on the slow path", "webui/runner.py:526-527")
    return R("Q07", "memory mode actually used is 'fast'", "PASS", detail, "",
             "webui/runner.py:496-527")


def q_loader(ctx):
    pj, log = ctx["product"], ctx["log"]
    if not (pj.get("params") or {}).get("vae"):
        return R("Q08", "VAE loader actually used (lean vs diffusers)", "SKIP",
                 "STRUCTURALLY N/A: vae=False in this job's params, so no VAE "
                 "stage ran (this engine submits vae=True)", "", "")
    m = VAELOADER.search(log)
    if "ZIMAGE_VAE_FAST unavailable" in log:
        return R("Q08", "VAE loader actually used (lean vs diffusers)", "FAIL",
                 "the lean loader raised: %s" %
                 (_first(re.compile(r"\[vae\] ZIMAGE_VAE_FAST unavailable \((.+?)\);"),
                         log) or "")[:200],
                 "the shipped diffusers loader is byte-identical but +15-18 s",
                 "webui/vae_local.py:132-144")
    if not m:
        return R("Q08", "VAE loader actually used (lean vs diffusers)", "FAIL",
                 "no '[vae] resident decoder loaded ... [loader]' line",
                 "the loader is a path, and a path with no log line is a guess",
                 "webui/vae_local.py:147-149")
    if m.group(1) != "lean":
        return R("Q08", "VAE loader actually used (lean vs diffusers)", "FAIL",
                 "loader=%r (expected 'lean')" % m.group(1),
                 "diffusers is the shipped loader: same PNG, +15-18 s per job",
                 "webui/vae_local.py:118,137")
    return R("Q08", "VAE loader actually used (lean vs diffusers)", "PASS",
             "loader=lean", "", "webui/vae_local.py:147-149")


def q_vae_threads(ctx):
    pj, log = ctx["product"], ctx["log"]
    if not (pj.get("params") or {}).get("vae"):
        return R("Q09", "VAE decode thread count actually used", "SKIP",
                 "STRUCTURALLY N/A: vae=False in this job's params", "", "")
    want = int(recipe_env()["ZIMAGE_VAE_THREADS"])
    m = VAETHREADS.search(log)
    if not m:
        return R("Q09", "VAE decode thread count actually used", "FAIL",
                 "no '[vae] decode threads: torch intra-op N' line",
                 "unpinned, the decode runs on torch's default (10 here) and "
                 "the count is not in the log at all; the loop's "
                 "ZIMAGE_CHUNK_THREADS would silently move it",
                 "webui/vae_local.py:81-92")
    if int(m.group(1)) != want:
        return R("Q09", "VAE decode thread count actually used", "FAIL",
                 "decode used %s threads, recipe pins %d" % (m.group(1), want),
                 "20 threads is 4.04 s instead of 6.33 s at 512^2 but is NOT "
                 "byte-identical (4-7 of 786,432 pixels move 1 LSB)",
                 "webui/config.py:101-111")
    return R("Q09", "VAE decode thread count actually used", "PASS",
             "torch intra-op %s threads" % m.group(1), "",
             "webui/vae_local.py:89-91")


def q_loop_threads(ctx):
    pj = ctx["product"]
    flags = pj.get("env_flags") or {}
    want = recipe_env()["ZIMAGE_CHUNK_THREADS"]
    got = flags.get("ZIMAGE_CHUNK_THREADS")
    detail = "job.json env_flags ZIMAGE_CHUNK_THREADS=%r (recipe %r)" % (got, want)
    if got is None:
        return R("Q10", "loop thread count the server recorded", "FAIL",
                 detail + " -- null means the server never saw it",
                 "ZIMAGE_CHUNK_THREADS=0/unset is torch's default 10, a "
                 "different host-attention pool, i.e. a different wall",
                 "webui/config.py:131-145")
    if str(got) != str(want):
        return R("Q10", "loop thread count the server recorded", "FAIL", detail,
                 "the run used a different pool than the reference",
                 "webui/config.py:143-145")
    return R("Q10", "loop thread count the server recorded", "PASS", detail +
             " (the product prints no line for the effective value: FIX-7)",
             "", "webui/config.py:131-145")


def q_env_fingerprint(ctx):
    """Q11 -- the run's env is the recipe, at the TWO levels that exist.

    LEVEL 1, the product's own record: job.json env_flags, i.e. LAUNCH_FLAGS
    (webui/config.py:131-136).  That tuple is the only env a job record can ever
    show, so a check that read only it can never be a whole-recipe proof: of the
    recipe variables in this file, most are not in it -- including T22_REF_SMNB,
    the one variable section 3 says moves the 512^2 latents.

    LEVEL 2, the whole recipe: the env P06 compared and the 16-hex env_id it
    computed over ALL of it.  That label is stored with the env in the preflight
    record, so this check RE-DERIVES it here instead of trusting a string.

    Why both: the old PASS line said "9 path flags recorded and equal" under a
    title that claimed the whole recipe, and only Q12 could ever catch a wrong
    T22_REF_SMNB.  The two levels are now reported separately, with the number of
    recipe variables each one can see.
    """
    title = "the run's env == the recipe (flags + whole-recipe env_id)"
    src = "webui/config.py:131-140, this file's RECIPE, P06"
    pj = ctx["product"]
    flags = pj.get("env_flags")
    rec = RECIPE_ENV
    arms = [(str(k), str(v)) for k, v in (ctx.get("extras") or [])]
    if not isinstance(flags, dict):
        return R("Q11", title, "FAIL", "job.json has no env_flags",
                 "no fingerprint, no comparison", "webui/config.py:131-140")
    bad = []
    for k, v in sorted(flags.items()):
        if k in rec:
            if str(v) != str(rec[k]):
                bad.append("%s=%r (recipe %r)" % (k, v, rec[k]))
        elif v not in (None, "", "0"):
            bad.append("%s=%r (not in the recipe, must be off)" % (k, v))
    seen = [k for k in rec if k in flags]
    blind = [k for k in rec if k not in flags]
    if bad:
        return R("Q11", title, "FAIL",
                 "LEVEL 1 job.json env_flags (%d keys): %d mismatch(es): %s"
                 % (len(flags), len(bad), "; ".join(bad[:6])),
                 "this is the anti-mix gate: two runs with different flags must "
                 "never be compared, and the product's own record is the proof",
                 src)
    lvl1 = ("LEVEL 1 job.json env_flags: all %d keys equal, but that tuple is "
            "LAUNCH_FLAGS -- %d of the %d recipe variables -- so %d are "
            "INVISIBLE to it (%s)"
            % (len(flags), len(seen), len(rec), len(blind),
               ", ".join(blind[:5]) + (" ..." if len(blind) > 5 else "")))
    # ---- level 2: the whole recipe, re-derived from the recorded env -----
    pre = preflight_record(ctx["tag"])
    pre_env = pre.get("env")
    pre_id = pre.get("env_id")
    recipe = env_id()
    status, why = "PASS", ""
    if isinstance(pre_env, dict) and pre_env:
        got_id = env_id({str(k): str(v) for k, v in pre_env.items()})
        diff, arm_diff = [], []
        for k in sorted(rec):
            got = pre_env.get(k)
            if str(got) != str(rec[k]):
                if any(k == ak and str(got) == av for ak, av in arms):
                    arm_diff.append("%s=%r (recipe %r)" % (k, got, rec[k]))
                else:
                    diff.append("%s=%r (recipe %r)" % (k, got, rec[k]))
        if diff:
            return R("Q11", title, "FAIL",
                     lvl1 + ".  LEVEL 2 the env P06 recorded DIFFERS from the "
                     "recipe this file defines now in %d place(s): %s"
                     % (len(diff), "; ".join(diff[:6])),
                     "env_id is the whole-recipe label; a difference here means "
                     "the recipe changed after the run (or the record is not "
                     "this run's), so the number belongs to another recipe", src)
        extra = [str(e[0]) for e in (pre.get("extras") or [])
                 if isinstance(e, (list, tuple)) and len(e) == 2
                 and str(e[0]) not in rec]
        if got_id == recipe:
            lvl2 = ("LEVEL 2 env_id: the %d values P06 recorded re-derive to "
                    "%s, which IS this file's recipe id" % (len(rec), got_id))
        elif arm_diff:
            lvl2 = ("LEVEL 2 env_id: the %d values P06 recorded re-derive to "
                    "%s, the ARM id -- not the recipe's %s" % (len(rec), got_id, recipe))
            status = "WARN"
            why = ("recorded ARM: %s.  The value really changed, the record "
                   "keeps the arm's env_id, and this run is NOT a reference run"
                   % ", ".join(arm_diff[:4]))
        else:
            lvl2 = ("LEVEL 2 env_id: the recorded values re-derive to %s, but "
                    "this file's recipe is %s" % (got_id, recipe))
            status = "WARN"
            why = ("the recorded env does not re-derive to this recipe and is not "
                   "an --extra arm either: the recipe changed after the run")
        if pre_id and str(pre_id) != got_id:
            lvl2 += " (the record's stored label %r is stale -- re-derived here)" % (pre_id,)
        if extra:
            lvl2 += ("; plus %d name(s) OUTSIDE the recipe (%s), which env_id "
                     "cannot cover because it is defined over the recipe only"
                     % (len(extra), ", ".join(extra[:4])))
            if status == "PASS":
                status = "WARN"
                why = ("these names are not recipe variables, so they can change "
                       "the run without changing its env_id; the record names "
                       "them and the doc says so")
    elif pre_id:
        lvl2 = "LEVEL 2 env_id: the record says %s, this file's recipe is %s" % (
            pre_id, recipe)
        if str(pre_id) != recipe:
            status = "WARN"
            why = ("the preflight that produced this run computed a different "
                   "env_id: the recipe CHANGED between the run and this "
                   "postflight, so the number belongs to the older recipe")
    else:
        lvl2 = ("LEVEL 2 env_id: NOT PROVABLE from this record -- there is no "
                "preflight record for tag %r, so the whole-recipe comparison "
                "was never stored (run preflight and postflight in one "
                "invocation, which the run action does)" % ctx["tag"])
        status = "WARN"
        why = ("without the recorded env there is no whole-recipe proof: LEVEL 1 "
               "can only ever see %d of the %d recipe variables, which is where "
               "T22_REF_SMNB hides" % (len(seen), len(rec)))
    key = "%dx%d:%d" % (ctx["h"], ctx["w"], ctx["steps"])
    ref = references().get(key) or {}
    ref_env = str(ref.get("env_id") or "")
    note = ""
    if ref_env and ref_env == recipe:
        note = ("  The reference for %s carries the same env_id (%s)."
                % (key, ref_env))
    elif ref_env:
        note = ("  The reference for %s was measured under env_id %s, not %s: a "
                "byte-neutral variable joined or left the recipe since that arm. "
                "That difference is not a failure and cannot be one here -- Q11 "
                "compares the ENV of THIS run, Q12 compares the BYTES against "
                "the reference." % (key, ref_env, recipe))
    return R("Q11", title, status, lvl1 + ".  " + lvl2 + "." + note, why, src)


def q_latents(ctx):
    pj = ctx["product"]
    sha = pj.get("latents_sha256") or ""
    key = "%dx%d:%d" % (ctx["h"], ctx["w"], ctx["steps"])
    ref = references().get(key)
    if not sha:
        return R("Q12", "latents reproduce the reference sha256", "FAIL",
                 "job.json has no latents_sha256", "", "job.json:latents_sha256")
    if not ref or not ref.get("latents_sha256"):
        _write_reference(key, {"latents_sha256": sha,
                               "path": ctx["plan"]["path"],
                               "m": ctx["plan"]["m"],
                               "loop_s": round(float((pj.get("timing") or {}).get("loop_s") or 0), 2),
                               "total_s": round(float((pj.get("timing") or {}).get("total_s") or 0), 1),
                               "vae": bool((pj.get("params") or {}).get("vae")),
                               "env_id": ctx["env_id"],
                               "ref_jobs": [ctx["job_id"]],
                               "note": "recorded by t22_repro on first use"})
        return R("Q12", "latents reproduce the reference sha256", "RECORDED",
                 "no reference for %s yet; recorded %s at env_id %s" %
                 (key, sha[:16], ctx["env_id"]),
                 "the next run of this recipe must reproduce this sha", "")
    ref_env = ref.get("env_id")
    # the full 64 hex on BOTH sides: the gate compares all 64, and a 16-hex
    # prefix in a record is what made a cold reader ask which value is gated
    detail = ("got %s, reference %s (%s); this run env_id %s, reference env_id %s"
              % (sha, ref["latents_sha256"], key, ctx["env_id"],
                 ref_env or "not recorded"))
    if not ref_env:
        _write_reference(key, {"env_id": ctx["env_id"]}, merge=True)
    if sha != ref["latents_sha256"]:
        hint = ""
        if ref_env and ref_env != ctx["env_id"]:
            hint = (" The two env_ids DIFFER: this is not the same recipe the "
                    "reference was measured with (see the RECIPE table).")
        elif not ref_env:
            hint = (" The reference's env was never recorded, so the difference "
                    "has to be found by bisection.")
        ctx["latents_ok"] = False
        return R("Q12", "latents reproduce the reference sha256", "FAIL",
                 detail + "." + hint,
                 "the bytes are the product; a different sha means a different "
                 "env, a different path, or a non-deterministic path",
                 "references in this file / reference.json")
    ctx["latents_ok"] = True
    return R("Q12", "latents reproduce the reference sha256", "PASS",
             detail + " -- from %s" % (", ".join(ref.get("ref_jobs") or [])), "",
             "job.json:latents_sha256")


def _ref_total_range(ref):
    """(lo, hi) of a reference's total_s, or (None, None) when it has none."""
    tot = ref.get("total_s")
    if isinstance(tot, (list, tuple)):
        vals = [float(x) for x in tot if x]
        if not vals:
            return None, None
        return min(vals), max(vals)
    if isinstance(tot, (int, float)) and tot:
        return float(tot), float(tot)
    return None, None


def q_loop_wall(ctx):
    """Q13 -- the wall, in the TWO pieces that can be compared, with a VERDICT.

    PART 1, loop_s: a WARN window of +-loop_tol around the reference.  It is not
    a byte gate (two identical-sha 512^2 arms differed 0.2 %) but a swing this
    large usually means a different path.

    PART 2, the total wall -- which had NO gate at all, so a reader had to decide
    on their own whether a total of 186.2 s against a 124.9/135.7 s reference
    (+37..48 %) was a failure.  Comparing total_s directly is wrong twice over:
    the reference arms ran vae=False while this engine submits vae=True (+~10 s
    at 512^2), and startup_s is the 15-30 s row that moves with the power state
    and with the B-repack arm (sections 12 and 13).  So the gate is on the
    DIFFUSION CORE = startup_s + loop_s -- for a vae-off arm that IS its total
    minus the child's import/TE, ~1 s -- against the reference total, with
    CORE_TOL.

    When the core is outside the band the check DECIDES instead of reporting:
      * the excess is in startup -> WARN: the number is real, it is not the loop,
        and sections 12/13 plus Q18 name what moves startup;
      * the excess is in the loop  -> FAIL: the diffusion work itself is outside
        its window, so this number is not comparable to the reference.

    The 186.2 s case (loop +6.2 % inside its window, startup 68 s instead of
    ~40 s) therefore lands on WARN with the decomposition printed, and nobody has
    to guess whether it was a failure.
    """
    title = "wall inside the reference windows (loop, and the total decomposed)"
    pj = ctx["product"]
    t = pj.get("timing") or {}
    loop = float(t.get("loop_s") or 0)
    total = float(t.get("total_s") or 0)
    startup = float(t.get("startup_s") or 0)
    key = "%dx%d:%d" % (ctx["h"], ctx["w"], ctx["steps"])
    ref = references().get(key) or {}
    rl, tol = ref.get("loop_s"), float(ref.get("loop_tol") or 0.25)
    rlo, rhi = _ref_total_range(ref)
    if not rl or not loop:
        return R("Q13", title, "SKIP", "no loop reference for %s" % key, "", "")
    dev = abs(loop - rl) / rl
    loop_out = dev > tol
    parts = ["loop %.2f s vs reference %.2f s (%+.1f%%, window +-%.0f%%)"
             % (loop, rl, 100.0 * (loop - rl) / rl, 100 * tol)]
    core = (startup + loop) if startup else 0.0
    core_state = "n/a"
    if core and rhi:
        band = ("%.1f / %.1f s" % (rlo, rhi)) if rlo != rhi else ("%.1f s" % rhi)
        parts.append("core startup %+.2f + loop %.2f = %.2f s vs reference total "
                     "%s (%+.1f%%, window +-%.0f%%); job total_s %.2f s"
                     % (startup, loop, core, band,
                        100.0 * (core - rhi) / rhi, 100 * CORE_TOL, total))
        if core <= rhi * (1.0 + CORE_TOL):
            core_state = "in"
        else:
            excess = core - rhi
            loop_excess = max(0.0, loop - rl)
            share = (loop_excess / excess) if excess > 0 else 0.0
            core_state = "loop" if share > 0.5 else "startup"
            start_excess = max(0.0, excess - loop_excess)
            parts.append("excess %+.2f s = %.2f s loop + %.2f s startup, so "
                         "%.0f%% of it is startup"
                         % (excess, loop_excess, start_excess,
                            100.0 * start_excess / excess))
    elif not core:
        parts.append("job.json has no timing.startup_s, so the total wall cannot "
                     "be decomposed for this job (only the loop part could run)")
    elif core:
        parts.append("the reference carries no total_s for %s, so the total wall "
                     "cannot be gated on this key" % key)
    detail = "; ".join(parts)
    if core_state == "loop":
        return R("Q13", title, "FAIL", detail,
                 "the DIFFUSION work itself is out of window: the excess is made "
                 "of loop seconds, not of startup, so this number is not "
                 "comparable to the reference (Q12 still decides whether the "
                 "bytes are right)", "")
    pw = (preflight_record(ctx["tag"]).get("power") or {})
    pw = str(pw.get("state") or "not recorded")
    if loop_out or core_state == "startup":
        why = ""
        if core_state == "startup":
            why = ("the excess is in STARTUP_S, not in the loop.  startup_s is a "
                   "wall: section 12 measured it x1.50 on battery vs AC while the "
                   "8-step loop moved x1.075, and section 13 splits it per phase "
                   "(the B-repack row is the one Q18 gates).  The power state "
                   "recorded with this run was %r.  A startup excess is NOT a "
                   "path failure and NOT a slower step -- quote it as startup"
                   % pw)
        elif loop_out:
            why = ("the wall is not a byte gate (two identical-sha runs differed "
                   "0.2% here) but a loop swing this large usually means a "
                   "different path; the printed core shows whether the total "
                   "moved with it.  ONE exception: a 1-step job's loop carries "
                   "torch's first call (section 12), so it is a warmup number, "
                   "not a compute number")
        return R("Q13", title, "WARN", detail, why, "")
    return R("Q13", title, "PASS", detail, "", "")


def q_faults(ctx):
    log = ctx["log"]
    hits = []
    for pat, why in FAULTS:
        if pat in log:
            line = ""
            for ln in log.splitlines():
                if pat in ln:
                    line = ln.strip()[:160]
                    break
            hits.append((pat, why, line))
    if hits:
        pat, why, line = hits[0]
        return R("Q14", "no device/host fault strings in the log", "FAIL",
                 "%d fault(s); first %r: %s" % (len(hits), pat, line), why,
                 "docs/KNOWN-PLATFORM-FACTS.md F-01/F-04")
    return R("Q14", "no device/host fault strings in the log", "PASS",
             "none of the %d fault patterns appear" % len(FAULTS), "", "")


def q_closure(ctx):
    pj = ctx["product"]
    t = pj.get("timing") or {}
    cl = t.get("closure") or {}
    if not cl:
        return R("Q15", "product timing closure gate", "FAIL",
                 "no timing.closure in job.json", "the product's own gate",
                 "webui/runner.py (closure)")
    if not cl.get("pass"):
        return R("Q15", "product timing closure gate", "FAIL",
                 "closure FAIL: sum %.2f s vs total %.2f s (unaccounted %+.2f s)" %
                 (cl.get("sum_s") or 0, cl.get("total_s") or 0,
                  cl.get("unaccounted_s") or 0),
                 "an unclosed ledger means a bucket is missing, i.e. a path the "
                 "record does not name", "webui/runner.py (closure)")
    return R("Q15", "product timing closure gate", "PASS",
             "sum %.2f + unaccounted %+.2f = total %.2f s (tolerance +-%.0f s)" %
             (cl.get("sum_s") or 0, cl.get("unaccounted_s") or 0,
              cl.get("total_s") or 0, cl.get("tolerance_s") or 0), "", "")


def q_image(ctx):
    pj = ctx["product"]
    req = pj.get("requested") or {}
    imgs = pj.get("images") or []
    vae = bool((pj.get("params") or {}).get("vae"))
    if not imgs:
        return R("Q16", "output artefact exists at the requested size", "FAIL",
                 "job.json lists no images", "", "job.json:images")
    im = imgs[0]
    h, w = int(im.get("height") or 0), int(im.get("width") or 0)
    if not vae:
        lf = pathlib.Path(im.get("latent_file") or "")
        detail = ("vae=False: latents artefact %s, %s B, sha %s" %
                  (lf.name or "<none>", im.get("bytes"),
                   (im.get("sha256") or "")[:16]))
        if lf.is_file():
            return R("Q16", "output artefact exists at the requested size",
                     "PASS", detail, "", "job.json:images[0].latent_file")
        return R("Q16", "output artefact exists at the requested size", "WARN",
                 detail + " -- the .npy is no longer on disk (the sha in "
                 "job.json is the record)", "", "job.json:images[0]")
    p = pathlib.Path(im.get("path") or "")
    detail = ("%s %dx%d, %s B, sha %s" %
              (p.name or "<none>", w, h, im.get("bytes"),
               (im.get("sha256") or "")[:16]))
    if not p.is_file():
        return R("Q16", "output image exists at the requested size", "FAIL",
                 detail + " -- file missing on disk", "", "job.json:images")
    if (h, w) != (int(req.get("height") or 0), int(req.get("width") or 0)):
        return R("Q16", "output image exists at the requested size", "FAIL",
                 detail + " != requested %sx%s" % (req.get("width"),
                                                   req.get("height")),
                 "the pad/crop contract: a 16-multiple runs padded and must be "
                 "cropped back", "T22-PAD-CROP-REPORT.md")
    key = "%dx%d:%d" % (ctx["h"], ctx["w"], ctx["steps"])
    ref = references().get(key) or {}
    if ref.get("png_sha256"):
        if im.get("sha256") != ref["png_sha256"]:
            return R("Q16", "output image exists at the requested size", "FAIL",
                     detail + " != reference PNG %s" % ref["png_sha256"][:16],
                     "the PNG is byte-stable at a fixed thread count (measured "
                     "2/2 on 2026-11-03 at 512^2, 10 threads) and 20 threads "
                     "moves 4-7 pixels of 786,432 -- so a mismatch is a "
                     "different decode, not noise",
                     "webui/config.py:101-111")
        return R("Q16", "output image exists at the requested size", "PASS",
                 detail + " (== reference PNG)", "", "job.json:images")
    if ctx.get("latents_ok") is False:
        return R("Q16", "output image exists at the requested size", "WARN",
                 detail + " (NOT recorded: the latents of this run did not match "
                 "the reference, so this PNG belongs to a different baseline)",
                 "recording it would poison the next run's comparison", "")
    _write_reference(key, {"png_sha256": im.get("sha256")}, merge=True)
    return R("Q16", "output image exists at the requested size", "PASS",
             detail + " (PNG sha recorded for next time)", "", "job.json:images")


def q_registered(ctx):
    log = ctx["log"]
    m = REGISTERED.search(log)
    # att + attip + ffn are always registered; qkv + attout only on the native
    # path (webui/runner.py:1885-1886 forces them off for chunked); the 16
    # BIGM_IMGS are registered whenever ZIMAGE_BIGM=1, which is why a 128^2 run
    # registers 21 and a 512^2 run registers 19 (measured 2026-11-03).
    want = (3 + len(CHUNKED_IMAGES)
            + (len(NATIVE_IMAGES) if ctx["plan"]["path"] == "native" else 0))
    if not m:
        return R("Q17", "image set registered at startup", "WARN",
                 "no '[t3d] registered N images' line", "", "")
    n = int(m.group(1))
    detail = "%d images registered (%s); expected %d for path=%s" % (
        n, m.group(2)[:110], want, ctx["plan"]["path"])
    if n != want:
        return R("Q17", "image set registered at startup", "FAIL", detail,
                 "the registered set IS the path: a missing image means the "
                 "dispatch went somewhere else", "imaggers/image/t18_run.py:319-368")
    return R("Q17", "image set registered at startup", "PASS", detail, "",
             "imaggers/image/t18_run.py:319-368")


def q_brepack(ctx):
    """Q18 -- the run MAPPED the repack; it did not re-derive it.

    The same class of check as Q05: a cache that is present, verified and then
    silently NOT used produces a job that succeeds, emits the reference latents
    and costs 27 s more -- there is no other symptom.  So the proof is the
    consumer's own line, plus the absence of every fallback line, plus the
    number the fallback would have printed.
    """
    log = ctx["log"]
    m = DK_BREPACK_ROW.search(log)
    row = float(m.group(1)) if m else None
    hit_s = _first(re.compile(r"\[brepack\] HIT .* in ([0-9.]+) s"), log)
    # TWO arming sites, and which one fires depends on the order the two
    # consumers are reached in.  Measured on the first mapped 512^2 arm:
    # t18_run's --cache-all site calls brepack_cache.arm() BEFORE the FFN's
    # lazy cached_payloads() runs, and arm() sets prod_loop_bigm's
    # _BGM_CACHE_READY, so the [bigm] line NEVER prints on the chunked path --
    # the [t13] one does.  Requiring only the [bigm] line failed a run that had
    # mapped perfectly.  Either line proves the same thing (the consumer was
    # armed); the ~0 [devkill] row proves the permutation did not run.
    sites = []
    if BREPACK_MAPPED in log:
        sites.append("prod_loop_bigm.cached_payloads")
    if BREPACK_MAPPED_T13 in log:
        sites.append("t18_run --cache-all site")
    misses = [(tok, why) for tok, why in BREPACK_MISS_TOKENS if tok in log]
    detail = ("armed by: %s; [brepack] HIT in %s s; "
              "[devkill] B repack at startup=%s s"
              % (", ".join(sites) or "NOBODY", hit_s or "-",
                 ("%.3f" % row) if row is not None else "-"))
    if misses:
        tok, why = misses[0]
        return R("Q18", "the B repack was MAPPED, not re-derived", "FAIL",
                 detail + "; found %r: %s" % (tok, why),
                 "the cache exists and the run paid the permutation instead: "
                 "27.02 s of a 45.64 s AC startup (59%) that this recipe is "
                 "supposed to remove, with byte-identical latents so nothing "
                 "else moves", "imaggers/block/bigm/prod_loop_bigm.py:613-642")
    if not sites:
        return R("Q18", "the B repack was MAPPED, not re-derived", "FAIL",
                 detail + "; no arming line at all (%r or %r)"
                 % (BREPACK_MAPPED, BREPACK_MAPPED_T13),
                 "without it there is no evidence the consumer armed the "
                 "cache: 'the file exists' is not 'the run used it' -- the "
                 "same failure Q05 catches for the fastglue hooks",
                 "imaggers/block/bigm/prod_loop_bigm.py:619, "
                 "imaggers/image/t18_run.py:2692")
    if row is None:
        return R("Q18", "the B repack was MAPPED, not re-derived", "WARN",
                 detail + "; the [devkill] B repack row is missing",
                 "T22_DEVKILL_BREPACK=0?  An arming line is present, so the "
                 "cache was armed; there is simply no second number to read",
                 "imaggers/image/t18_run.py:2737-2746")
    if row > 2.0:
        return R("Q18", "the B repack was MAPPED, not re-derived", "FAIL",
                 detail + "; the row is %.2f s, not ~0" % row,
                 "the row the map removes is still being paid: armed but not "
                 "used is exactly the silent miss this check exists for",
                 "imaggers/image/t18_run.py:2739-2746")
    if BREPACK_MAPPED_T13 not in log:
        return R("Q18", "the B repack was MAPPED, not re-derived", "WARN",
                 detail,
                 "the B payloads were mapped for the FFN, but the 3.71 GB "
                 "read+permute at the --cache-all site was NOT skipped: "
                 "T18_PAYLOAD_ORDER is not 'last', or a preload thread won "
                 "the race -- the 27 s is gone, a few seconds of it are not",
                 "imaggers/image/t18_run.py:2681-2703")
    return R("Q18", "the B repack was MAPPED, not re-derived", "PASS",
             detail + "; the 5.31 GB permutation AND the 3.71 GB read+permute "
             "were both skipped", "", "imaggers/block/bigm/prod_loop_bigm.py:619, "
             "imaggers/image/t18_run.py:2692")


QF_CHECKS = [
    ("Q01", "job finished 'done'", q_status),
    ("Q02", "the shape took the path the plan named", q_path),
    ("Q03", "chunk_path description names the big-M projector", q_chunkpath),
    ("Q04", "fastglue 8/8 hooks installed", q_cg_installed),
    ("Q05", "fastglue hooks WERE CALLED (per-layer lines)", q_cg_called),
    ("Q06", "FFN: devkill rows + ffnchain calls", q_ffn_called),
    ("Q07", "memory mode actually used is 'fast'", q_memory),
    ("Q08", "VAE loader actually used (lean vs diffusers)", q_loader),
    ("Q09", "VAE decode thread count actually used", q_vae_threads),
    ("Q10", "loop thread count the server recorded", q_loop_threads),
    ("Q11", "the run's env == the recipe (flags + whole-recipe env_id)",
     q_env_fingerprint),
    ("Q12", "latents reproduce the reference sha256", q_latents),
    ("Q13", "wall inside the reference windows (loop, and the total decomposed)",
     q_loop_wall),
    ("Q14", "no device/host fault strings in the log", q_faults),
    ("Q15", "product timing closure gate", q_closure),
    ("Q16", "output image exists at the requested size", q_image),
    ("Q17", "image set registered at startup", q_registered),
    ("Q18", "the B repack was MAPPED, not re-derived", q_brepack),
]

QF_EVIDENCE = {
    "Q01": "job.json status + the first exception line of job.log",
    "Q02": "'[t18] CHUNKED SEQ=<M>' vs '[t18] SEQ=<M>'",
    "Q03": "'[t18] chunk_path installed: <what> (proj/ffn/o = ...)'",
    "Q04": "'[fastglue] installed N/8 hooks (threads=T)'",
    "Q05": "'[fastglue] layers.N proj_qkv ...' x layers x steps",
    "Q06": "'[devkill] rows=M pack=...' and '[ffnprof] seq=M ...' x layers x steps",
    "Q07": "'[webui] memory mode: fast (...)' + '[t13] cache-all: ...' in any "
           "of its four shapes (N GB free before allocation / B payloads mapped "
           "from / payload already resident / preload is on a worker thread)",
    "Q08": "'[vae] resident decoder loaded ... [lean]'",
    "Q09": "'[vae] decode threads: torch intra-op N (ZIMAGE_VAE_THREADS=...)'",
    "Q10": "job.json env_flags.ZIMAGE_CHUNK_THREADS",
    "Q11": "TWO levels: job.json env_flags (the LAUNCH_FLAGS the product "
           "records -- a subset of the recipe) AND the env the preflight recorded, "
           "whose env_id is re-derived over EVERY recipe variable",
    "Q12": "job.json latents_sha256 vs the reference table",
    "Q13": "job.json timing.loop_s vs the reference loop_s, AND startup_s + "
           "loop_s (the diffusion core) vs the reference total_s, with the excess "
           "split between loop seconds and startup seconds",
    "Q14": "the fault strings themselves",
    "Q15": "job.json timing.closure.pass",
    "Q16": "job.json images[0] path/width/height/sha256",
    "Q17": "'[t3d] registered N images: <names>'",
    "Q18": "an arming line -- '[bigm] B payloads mapped from ...' or "
           "'[t13] cache-all: B payloads mapped from ...' (whichever consumer "
           "fires first) -- plus '[brepack] HIT ... in N s' and "
           "'[devkill] B repack at startup: ~0.0 s', and NO 'repacked once', "
           "'cache unavailable', 'cache MISS' or 'B order: mapped'",
}


# ---------------------------------------------------------------------------
# 10. records + references
# ---------------------------------------------------------------------------
RECIPE_ENV = {k: val for k, val, _, _ in RECIPE}
BT = "@BT@"          # rendered to a backtick, so this source needs no literal one


def references():
    refs = json.loads(json.dumps(REFERENCES))
    if REFERENCE_FILE.is_file():
        try:
            extra = json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))
        except Exception:
            extra = {}
        for k, v in (extra.get("refs") or {}).items():
            refs.setdefault(k, {}).update(v)
    return refs


def _write_reference(key, vals, merge=False):
    cur = {}
    if REFERENCE_FILE.is_file():
        try:
            cur = json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))
        except Exception:
            cur = {}
    refs = cur.setdefault("refs", {})
    if merge:
        refs.setdefault(key, {}).update(vals)
    else:
        refs[key] = dict(vals)
    cur["updated"] = now_iso()
    cur["note"] = ("written by imaggers/repro/t22_repro.py; the built-in "
                   "REFERENCES in that file win for the keys they define")
    REFERENCE_FILE.write_text(json.dumps(cur, indent=1, ensure_ascii=False),
                              encoding="utf-8")


def save_run(kind, tag, ctx, results):
    RECORDS.mkdir(parents=True, exist_ok=True)
    fails = [r["cid"] for r in results if r["status"] == "FAIL"]
    rec = {"kind": kind, "tag": tag, "when": now_iso(),
           "job_id": ctx.get("job_id"), "h": ctx.get("h"), "w": ctx.get("w"),
           "steps": ctx.get("steps"), "env_id": ctx.get("env_id"),
           "path": (ctx.get("plan") or {}).get("path"),
           "power": ctx.get("power"), "extras": ctx.get("extras") or [],
           "skipped": ctx.get("skipped") or [],
           "env": ctx.get("env"),
           "device_after": ctx.get("device_after") or {},
           "results": results, "fail": len(fails), "failed_ids": fails}
    (RECORDS / (tag + "." + kind + ".json")).write_text(
        json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    idx = RECORDS / "index.json"
    d = {"runs": []}
    if idx.is_file():
        try:
            d = json.loads(idx.read_text(encoding="utf-8"))
        except Exception:
            d = {"runs": []}
    rows = [r for r in d.get("runs", []) if r.get("tag") != tag]
    rows.append({"tag": tag, "when": rec["when"], "job_id": rec["job_id"],
                 "h": rec["h"], "w": rec["w"], "steps": rec["steps"],
                 "env_id": rec["env_id"], "path": rec["path"],
                 "power": (rec.get("power") or {}).get("state"),
                 "extras": [list(e) for e in (rec.get("extras") or [])],
                 "skipped": rec.get("skipped") or [],
                 "postflight_fail": len(fails), "failed_ids": fails})
    d["runs"] = rows[-60:]
    idx.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    return rec


# ---------------------------------------------------------------------------
# 11. the two orchestrators
# ---------------------------------------------------------------------------
def preflight(tag, h, w, steps, apply=True):
    scrubbed0, conflicts0 = [], []
    if apply:
        out("[env] applying the recipe: %d variables" % len(RECIPE))
        _env, scrubbed0, conflicts0 = apply_env(verbose=True)
    extras = []
    for item in EXTRAS:
        if "=" not in item:
            out("[extra] IGNORED %r (needs NAME=VALUE)" % (item,))
            continue
        k, v = item.split("=", 1)
        k = k.strip()
        os.environ[k] = v
        extras.append((k, v))
        out("[extra] ARM OVERRIDE %s=%s -- this run is an A/B arm, NOT a "
            "reference run of the recipe" % (k, v))
    # The env this run is SUPPOSED to have.  An --extra that names a RECIPE
    # variable is an ARM: the value really changes, it is printed, and it is
    # stored -- so the expected env is the recipe WITH the arm applied, and P06
    # reports the deviation instead of refusing a measurement the tool was
    # asked to make.  Anything else still fails P06.
    expected = dict(RECIPE_ENV)
    for k, v in extras:
        if k in expected:
            expected[k] = v
    ctx = {"tag": tag, "h": int(h), "w": int(w), "steps": int(steps),
           "scrubbed0": scrubbed0, "conflicts0": conflicts0,
           "expected": expected,
           "extras": extras, "skipped": list(SKIP_CHECKS)}
    # P06 publishes env/env_id/scrubbed/arms, and run_job() needs them.  A
    # SKIPPED check must not break the run it was skipped for (measured: the
    # first plan-arm attempt died here with KeyError 'env_id'), so the same
    # values are published here and P06 overwrites them identically when it
    # does run.
    ctx["env"] = dict(expected)
    ctx["env_id"] = env_id(ctx["env"])
    ctx["scrubbed"] = scrubbed0
    ctx["arms"] = [(k, v) for k, v in extras
                   if k in RECIPE_ENV and str(RECIPE_ENV[k]) != str(v)]
    from webui import config      # noqa: F401  (env-sensitive module constants)
    try:
        ctx["plan"] = plan_for(h, w)
    except Exception as exc:
        ctx["plan"] = {"h": h, "w": w, "m": 0, "path": "?", "renderable": False,
                       "refusal": "%s: %s" % (type(exc).__name__, exc),
                       "exec_mode": None}
    results = []
    for cid, title, fn in PF_CHECKS:
        if cid in SKIP_CHECKS:
            out("[skip] %s not run (--skip %s): its effect on this machine is "
                "part of what the arm is measuring" % (cid, cid))
            results.append(R(cid, title, "SKIP",
                             "not run: --skip %s on this arm" % cid, "", ""))
            continue
        try:
            results.append(fn(ctx))
        except Exception as exc:
            results.append(R(cid, title, "FAIL",
                             "%s: %s" % (type(exc).__name__, exc),
                             "the check itself raised", ""))
    return results, ctx


def postflight(tag):
    ctx = pf_ctx(tag)
    results = []
    for cid, title, fn in QF_CHECKS:
        try:
            results.append(fn(ctx))
        except Exception as exc:
            results.append(R(cid, title, "FAIL",
                             "%s: %s" % (type(exc).__name__, exc),
                             "the check itself raised", ""))
    save_run("postflight", tag, ctx, results)
    return results, ctx


def device_footer(ctx):
    """Printed after the checks: how the device looked then, and how it looks now.

    NOT a check, and deliberately outside the registry: another agent's job may
    legitimately own the NPU when this one ends, and the path proof does not
    depend on it.  But a cold reader must not have to infer what "lock held by
    another pid -- NOT touched" means, so the state is printed, stored in
    last/<tag>.device.json, and listed in RUNBOOK section 9.
    """
    d = ctx.get("device_after") or {}
    if d:
        out("[device] when this job ended: %s" % device_line(d))
    else:
        out("[device] when this job ended: NOT RECORDED (no last/%s.device.json)"
            % ctx.get("tag"))
    present, owner, pid, alive = None, None, None, None
    try:
        present, _txt, owner, pid, alive = lock_state()
    except Exception as exc:                              # noqa: BLE001
        out("[device] lock probe failed: %s" % exc)
        present = None
    try:
        users = device_users()
    except Exception:                                     # noqa: BLE001
        users = []
    out("[device] now: lock %s; %d device process(es)%s; port %d %s"
        % (("present owner=%s pid=%s alive=%s" % (owner, pid, alive))
           if present else ("absent" if present is False else "unprobeable"),
           len(users),
           (": " + ", ".join("pid %s %s" % (u["pid"], u["name"])
                             for u in users[:4])) if users else "",
           PORT, "OPEN" if port_busy() else "free"))
    out("[device] the postflight reads FILES ONLY -- it never touches the NPU, "
        "so it is valid even while somebody else's job owns the device")


# ---------------------------------------------------------------------------
# 12. RUNBOOK.md is GENERATED from the constants and the check registry above
# ---------------------------------------------------------------------------
DIAGNOSE = [
    ("P02 keeps printing a pid",
     "somebody else's job -- or a killed server's orphaned bridge -- owns the "
     "device",
     "wait.  Do NOT delete the lock.  If the pid is gone but P02 still sees it, "
     "re-run.  Killing another session's bridge is how a run dies mid-dispatch"),
    ("P02 sees a bridge.py and no lock file",
     "a launcher bypassed run_job_py, so no lock was ever written",
     "nothing to clean: P02 is the gate, the lock is only a hint"),
    ("P03 says stale lock, owner alive",
     "a live process owns it", "wait for it; the check refuses to steal"),
    ("P04 reserve probe FAILED",
     "the fast path's ~6 GB of commit is not there",
     "close memory and re-run.  Starting anyway means --no-cache-all (2x "
     "slower, different latent baseline) or a mid-run MemoryError fallback"),
    ("P05 refused",
     "webui/capability.py refuses this size in THIS env (the fastglue "
     "projection BO covers 1632 rows, or the whole-shape BO covers M rows)",
     "read the refusal: it names the exact bytes/rows.  A big-M shape needs "
     "T22_ADAPT_M at startup, which is a DIFFERENT recipe and not this one"),
    ("P07 FAIL",
     "ZIMAGE_CHUNK_THREADS and ZIMAGE_VAE_THREADS are unbalanced",
     "they are pinned here; a failure means the product's own derivation moved"),
    ("Q01 fails with 0xc01e0009 ('Failed to create context virtual')",
     "another process took the NPU between this run's preflight and its device "
     "stage.  The F-14.1 lock cannot stop it: that launcher never checks the "
     "lock (see P02's note)",
     "the tool retries by itself under a new tag suffix (RETRY_ON_COLLISION).  "
     "If it still fails, wait for the other session; deleting the lock does "
     "nothing"),
    ("Q04 installed 0/8 or 4/8",
     "T22_FASTGLUE (no underscore) is missing, or a gate failed",
     "the loop is ~40% slower and the run still works -- this is the most "
     "expensive silent fallback in the repo"),
    ("Q06 RuntimeError devkill needs n32 B payload",
     "T22_BIGM_DEVHANDOFF is not 1, so prod_loop_bigm hands out {n128,n64}",
     "set it (RECIPE does).  It costs an hour if you believe the "
     "lock/payload story instead"),
    ("P09 B-repack cache NOT REUSABLE",
     "the file the recipe's identity resolves to is absent, the wrong size, or "
     "built for a different identity (the key covers the 60 source shas, the "
     "five permutation functions and the geometry)",
     "build it once, on purpose, and time it: -Action cache with -Extra BUILD.  "
     "Do NOT start the job 'to let it build itself': that is the 27 s "
     "permutation paid inside startup_s, which is what the recipe removes"),
    ("Q18 found 'repacked once' / 'cache unavailable' / 'cache MISS'",
     "the cache was present (or not) and the run paid the permutation anyway: "
     "27.02 s of a 45.64 s AC startup, with byte-identical latents, i.e. no "
     "symptom anywhere else",
     "read the token Q18 names.  'repacked once' means the identity MISSED (a "
     "stale file); 'unavailable' means the consumer raised; 'cache MISS' means "
     "no file matched.  None of them is a successful run of this recipe.  ONE "
     "failure is EXPECTED and is not a fault: an --extra arm with "
     "T22_BREPACK_CACHE=0 is the negative control of section 13's pair, and Q18 "
     "failing on it is the proof that Q18 can see a miss at all"),
    ("Q07 memory_mode low",
     "the job ran --no-cache-all",
     "its number is not comparable to a fast-mode reference; delete it"),
    ("Q11 FAIL, LEVEL 1 (the env_flags line)",
     "the env the PRODUCT recorded is not this file's RECIPE",
     "the engine scrubs inheritance, so something else launched the server -- "
     "check how the run was started"),
    ("Q11 FAIL, LEVEL 2 (the env_id line)",
     "the env P06 recorded does not match the recipe this file defines NOW",
     "either the recipe changed after the run (the number belongs to the old "
     "recipe: re-run) or the preflight record is another run's.  LEVEL 2 "
     "re-derives the id from the stored values, so a stale label is reported "
     "separately from a real difference"),
    ("Q12 sha mismatch while Q11 passes",
     "same env, different bytes",
     "run it twice: same sha twice means the reference is wrong; two different "
     "shas mean the path is not deterministic"),
    ("Q14 state 8",
     "the ERT watchdog fired: two back-to-back launches of the SAME xclbin with "
     "no barrier between them (F-01)",
     "keep T20_FFN_BARRIER=1; a completion right after a relaunch may be FALSE "
     "(a second kernel handle reports 2-4 ms for a 132.7 MB stream)"),
    ("Q13 says the total wall is outside its band",
     "the diffusion core (startup_s + loop_s) moved, and Q13 prints how much of "
     "the excess is startup and how much is loop",
     "read the split -- do not re-derive the decision by hand: excess in "
     "startup_s is a startup number (sections 12/13: power state, B-repack arm) "
     "and NOT a path failure; excess in the loop is a FAIL and the number is "
     "then not comparable to the reference"),
    ("Q11 WARN 'LEVEL 2 env_id: NOT PROVABLE from this record'",
     "there is no preflight record for this tag, so the env_id over the WHOLE "
     "recipe could not be re-derived -- only the subset the product records",
     "run the preflight and the postflight in one invocation (the run action "
     "does).  Never read level 1 (env_flags) as the whole recipe: see 6.2"),
    ("P11 SKIP 'no run of THIS recipe+shape has been recorded yet'",
     "nothing has been recorded for this shape+steps under this env_id",
     "not a failure: Q12 RECORDS the reference on this run's first use and "
     "compares against it from the next run on"),
    ("a run ends with 'device.lock: held by another pid N -- NOT touched'",
     "another agent's run owns the device; this engine refuses to delete a live "
     "owner's lock, which is exactly what P03 is for",
     "do nothing to the lock and do not start a second job.  The postflight reads "
     "files only, so finish it now; the next run's P02 waits for that pid "
     "(section 9.1)"),
    ("'[run] WARNING device still busy Ns after the server stopped'",
     "an orphaned bridge, or another session's process, was still on the NPU "
     "when this run ended",
     "not a failure of your job: P02 waits for it on the next run, and killing "
     "another session's bridge is how a run dies mid-dispatch (section 9.1)"),
]

LAWS = [
    ("F-01", "state-8: a second back-to-back launch of the same xclbin never "
             "completes; the ERT watchdog (~7.0 s) reports state 8.  Any other "
             "xclbin's launch clears it -- that is what T20_FFN_BARRIER=1 does. "
             "A second kernel handle reports a FALSE completion",
     "docs/KNOWN-PLATFORM-FACTS.md F-01/F-04, webui/evidence/T20-STATE8.md"),
    ("F-08", "at most 7 declared data arguments per kernel; 8/9/10 all ABORT in "
             "under 2 ms (add an argument and the kernel dies instantly)",
     "docs/KNOWN-PLATFORM-FACTS.md F-08, HANDOFF.md:168-170"),
    ("F-14", "the device is a shared resource; a second user gets 0xc01e0009.  "
             "One process, one xclbin, one context.  device.lock is a protocol, "
             "not a mutex",
     "docs/KNOWN-PLATFORM-FACTS.md F-14, this file's P02"),
    ("F-09", "teardown crashes are normal and the exit code is not evidence; the "
             "judge is the artefact plus the log line",
     "docs/KNOWN-PLATFORM-FACTS.md F-09"),
]

FIXES = [
    ("webui/chunked.py:100-105",
     "torch.set_num_threads(n) has NO log line, so the loop's effective thread "
     "pool cannot be proved from a run",
     "print once after the first _torch(): [t18] loop threads: torch intra-op "
     "N (one line, no behaviour change)"),
    ("imaggers/util/glue/install.py:869",
     "install() prints installed N/8 and returns success even when hooks are "
     "missing; the caller ignores the returned state",
     "chunked.install() should record STATE in the job (or print one "
     "[t18] fastglue hooks=N/8 line) so a partial install is a number, not a "
     "log line somebody has to notice"),
    ("webui/chunked.py:995-998",
     "an exception inside glue install is printed as [fastglue] skipped and the "
     "run continues on the slower path",
     "raise in trial mode, or set a job-level warning the UI shows"),
    ("imaggers/block/bigm/prod_loop_bigm.py:647-649",
     "T22_DEVKILL=1 with T22_BIGM_DEVHANDOFF!=1 returns {n128,n64}, and the "
     "mismatch only surfaces 60 s later inside devkill_ffn.py:332",
     "raise at startup in cached_payloads() when devkill is on and the handoff "
     "payload is absent, with the sentence at devkill_ffn.py:332 instead of a "
     "traceback from inside the loop"),
    ("imaggers/bigpipe2/run_job_py.py:31-33",
     "the lock is unlinked at startup WITHOUT checking the owner, so a live "
     "job's lock disappears and a crashed job's lock is never diagnosed",
     "refuse when the owner pid is alive: the same 8 lines as this file's P03"),
    ("imaggers/bigpipe2/launch_webui_bigm.py:63",
     "the lock's purpose string is the literal bigm2_512_1024 whatever the run "
     "is",
     "take the tag/shape from argv so the lock names the run it blocks"),
    ("webui/config.py:148-163 + webui/vae_local.py:81-92",
     "ZIMAGE_VAE_THREADS unset silently inherits the LOOP's thread count, which "
     "moves 4-7 pixels of 786,432 at 512^2 -- and when both are 0 nothing is "
     "logged at all",
     "always log the effective count, including the torch-default case, and "
     "make the inheritance explicit (auto -> N, follows ZIMAGE_CHUNK_THREADS)"),
    ("webui/shapes.py:580-583 + webui/capability.py:214",
     "chunk_gate() returns a refusal sentence for a shape that is renderable on "
     "the NATIVE path, and capability.describe() passes it through, so "
     "describe(128,128)['refusal'] is non-empty on a working size",
     "blank refusal when path != refused (one condition)"),
    ("webui/runner.py:524-527",
     "the MemoryError fallback to --no-cache-all is a log line plus a "
     "memory_reason; the job still reports done",
     "add job['degraded']=True and a [webui] DEGRADED banner: a consumer that "
     "reads only status cannot tell"),
    ("imaggers/image/t18_run.py:196 + imaggers/util/glue/install.py:591",
     "T22_FAST_GLUE and T22_FASTGLUE are two different switches one character "
     "apart, with opposite defaults (0 vs 1) and different effects",
     "let install() treat T22_FAST_GLUE=1 as consent when T22_FASTGLUE is "
     "unset, or rename one of them"),
    ("webui/config.py:131-135",
     "T22_BREPACK_CACHE is not in LAUNCH_FLAGS, so the JOB's own record cannot "
     "say which of the two startup arms it was.  The recipe label (env_id) does "
     "cover it, and Q18 reads the consumer's own line -- but neither is in "
     "job.json, so a consumer of the product record alone still cannot tell",
     "add it to LAUNCH_FLAGS (recording only, it selects nothing), so the job "
     "record itself distinguishes a mapped run from a repacked one"),
    ("webui/config.py:37,231,267",
     "ZIMAGE_FFN_IMAGE / T22_QKV / T22_P10 default ON while every other device "
     "path defaults OFF, and env_flags cannot tell 'default' from 'explicit'",
     "record the resolved value plus whether it came from the env or the "
     "default, so a record cannot be misread as a deliberate choice"),
]


def doc(check=False):
    """Render RUNBOOK.md from the structures above.  Drift is impossible."""
    refs = references()
    Q = BT
    # what job.json:env_flags can and cannot see (read from webui/config.py)
    flags = launch_flags_from_source()
    flag_recipe = [k for k, _v, _w, _s in RECIPE if k in flags]
    blind = [k for k, _v, _w, _s in RECIPE if k not in flags]
    L = []
    add = L.append
    add("# RUNBOOK -- one reproducible T22 device run")
    add("")
    add("<!-- GENERATED FILE.  Do not edit by hand: " + Q + "py312 "
        "imaggers/repro/t22_repro.py doc" + Q + " rewrites it from the constants")
    add("     and the check registry of that file, so this document cannot drift")
    add("     from the code that enforces it. -->")
    add("")
    add("Generated by " + Q + "imaggers/repro/t22_repro.py" + Q + ".  No timestamp on")
    add("purpose: this file is byte-stable, so " + Q + "doc --check" + Q + " can prove it is current.")
    add("env_id = first 16 hex chars of sha256 over the sorted RECIPE name=value lines.")
    add("")
    add("## 1. The one command")
    add("")
    add(Q*3 + "powershell")
    add("cd <checkout>")
    add("imaggers\\repro\\t22_run.ps1 -Size 512 -Steps 8 -Tag mytag")
    add("imaggers\\repro\\t22_run.ps1 -Size 128 -Steps 1 -Tag small")
    add(Q*3)
    add("")
    add("That is the whole interface: a size, a step count, a tag.  Do not export")
    add("anything, do not copy an env block out of a report.  The script sets every")
    add("variable itself (section 3), deletes every other device switch it")
    add("inherited (section 4), refuses to start if the device is not provably idle")
    add("or the size is not renderable (section 5), runs the job, and then proves")
    add("which path actually ran (section 6).")
    add("")
    add("### 1.1 the three arguments, including the tag rules")
    add("")
    add("* **-Size** -- a shape (" + Q + "512" + Q + " or " + Q + "512x384" + Q + ").  512 takes the chunked")
    add("  path, 128 the native one.  Any integer per side is accepted, the internal")
    add("  grid is 16 px, so a size off that grid is PADDED and cropped back by the")
    add("  product (T22-PAD-CROP-REPORT.md).  P05 refuses -- before the device is")
    add("  touched -- whatever webui/capability.py cannot render in this env.")
    add("* **-Steps** -- the reference table keys on shape+steps, so a new step")
    add("  count RECORDS a reference on its first run instead of comparing (Q12).")
    add("* **-Tag** -- the name of THIS run, and it is a FILENAME KEY, so its rules")
    add("  are enforced, not suggested:")
    add("    * charset " + Q + "A-Za-z0-9._-" + Q + ".  Every other character (a space, a")
    add("      slash, a colon) becomes " + Q + "-" + Q + ", leading/trailing " + Q + "-" + Q + " are stripped and")
    add("      the result is cut to " + str(TAG_MAX) + " characters; empty becomes " + Q + "t<HHMMSS>" + Q + ".")
    add("      When anything changed, the engine prints " + Q + "[tag] <given> -> <used>" + Q + ".")
    add("    * it names " + Q + "imaggers/repro/last/<tag>.preflight.json" + Q + ", " + Q + "...job.json" + Q + ",")
    add("      " + Q + "...postflight.json" + Q + ", " + Q + "...device.json" + Q + ", " + Q + "...server.out.log" + Q + ",")
    add("      " + Q + "...server.err.log" + Q + ", and ONE row of " + Q + "last/index.json" + Q + ".")
    add("    * **reusing a tag overwrites that history**: save_run() keeps one row")
    add("      per tag, so a second run under the same tag REPLACES the first one's")
    add("      row.  Use a fresh tag per run; when you do not, the engine prints")
    add("      " + Q + "[tag] WARNING ... will be OVERWRITTEN" + Q + " and names the files.")
    add("    * it is also the JOB PREFIX, " + Q + TAG_PREFIX + "<tag>" + Q + ", so the artefact is")
    add("      " + Q + TAG_PREFIX + "<tag>_<job-id>_p0_s<seed>.png" + Q + ": a tag you can read in a")
    add("      file name is a tag you can find the record of.")
    add("    * a device-collision retry appends " + Q + "-r2" + Q + " (RETRY_ON_COLLISION), and the")
    add("      retried run writes its records under THAT tag.")
    add("")
    add('What "it worked" means is defined once -- section 6 -- and printed by every')
    add("run: **N evaluated, 0 FAIL**, never \"N/N PASS\".  A native shape has four")
    add("checks that are structurally N/A, so no native run can ever print \"18/18\".")
    add("")
    add("**Read section 12 before comparing any two numbers from this box**, and")
    add("section 13 before quoting " + Q + "startup_s" + Q + ": the first is the")
    add("power-state rule (measured: startup_s x1.50, an 8-step loop x1.075), the")
    add("second is the per-second decomposition of a number nobody could")
    add("previously attribute.  Only AC numbers are comparable to the references.")
    add("")
    add("Maintenance verbs, not needed to run a job:")
    add("")
    add(Q*3 + "powershell")
    add("imaggers\\repro\\t22_run.ps1 -Action doc                 # regenerate this file")
    add("imaggers\\repro\\t22_run.ps1 -Action preflight -Size 512 -Steps 8 -Tag t")
    add("imaggers\\repro\\t22_run.ps1 -Action postflight -Tag t")
    add("imaggers\\repro\\t22_run.ps1 -Action status")
    add("imaggers\\repro\\t22_run.ps1 -Action cache                 # resolve the B-repack cache")
    add("imaggers\\repro\\t22_run.ps1 -Action cache -Build           # build it once, timed")
    add("imaggers\\repro\\t22_run.ps1 -Size 512 -Steps 8 -Tag off -Extra T22_BREPACK_CACHE=0")
    add(Q*3)
    add("")
    add("## 2. What this is for")
    add("")
    add("A 512^2 8-step run of this recipe measured loop 94.15 s and 93.94 s in two")
    add("independent arms and produced the SAME latents sha256 (2abf91331200d069...).")
    add("The same shape on the product-default env measured 229.26 s and a DIFFERENT")
    add("sha (12eb356024a95971...).  Both are real runs of real code; only one of")
    add("them is the reference.  Every precondition that separates them is silent")
    add("when wrong, which is what this file removes.")
    add("")
    add("Two conventions to read those shas correctly, because both were inferred the")
    add("hard way:")
    add("")
    add("* A latents sha256 is **64 hex characters**.  The 16-character prefixes in")
    add("  this prose (" + Q + "2abf91331200d069" + Q + ", " + Q + "12eb356024a95971" + Q + ") are abbreviations for")
    add("  reading; section 7 prints the full value and Q12 compares all 64.  The")
    add("  16-hex " + Q + "env_id" + Q + " is a DIFFERENT thing -- the recipe's label, not the bytes.")
    add("* The acceptance sentence is " + Q + "N evaluated, 0 FAIL" + Q + " (section 6), not")
    add("  " + Q + "N/N PASS" + Q + ": a SKIP is a structural N/A and never a pass.")
    add("")
    add("## 3. The recipe (every variable, one value, one reason)")
    add("")
    add("| variable | value | why it is required | reads it |")
    add("|---|---|---|---|")
    for k, v, why, src in RECIPE:
        add("| " + Q + k + Q + " | " + Q + v + Q + " | " +
            why.replace("|", "/") + " | " + src + " |")
    add("")
    add("env_id of this recipe: **" + env_id() + "**")
    add("")
    add("That id is the sha256 of the sorted " + str(len(RECIPE)) + " " + Q + "name=value" + Q + " lines, cut to 16")
    add("hex.  It is the ONLY label that covers the whole recipe: P06 computes it")
    add("over the environment the child inherits, prints it, and stores it with the")
    add("env in " + Q + "imaggers/repro/last/<tag>.preflight.json" + Q + "; Q11 re-derives it.")
    add("")
    add("It is NOT the same thing as " + Q + "job.json:env_flags" + Q + ".  That records")
    add(Q + "LAUNCH_FLAGS" + Q + " (webui/config.py:131-136) -- " + str(len(flags)) + " names, of which " +
        str(len(flag_recipe)) + " are recipe")
    add("variables -- so **" + str(len(blind)) + " of the " + str(len(RECIPE)) + " variables are INVISIBLE to the job")
    add("record**, including " + Q + "T22_REF_SMNB" + Q + ", the one variable the table above says")
    add("moves the 512^2 latents.  A check that read only " + Q + "env_flags" + Q + " could")
    add("therefore never be the whole-recipe proof this runbook needs (section 6.2):")
    add("the whole-recipe proof is the env_id, and the byte proof is Q12.")
    add("")
    add("Pinned constants that are not env: prompt " + Q + PROMPT + Q + ", seed " +
        str(SEED) + ", vae on, mode trial, memory_mode=fast, job prefix " +
        Q + TAG_PREFIX + "<tag>" + Q + ".")
    add("")
    add("## 4. What the entry point DELETES before the job starts")
    add("")
    add("An inherited switch is a different device path.  These are removed from the")
    add("child environment and reported as " + Q + "[env] scrubbed ..." + Q + ":")
    add("")
    add("| variable | must be off because | reads it |")
    add("|---|---|---|")
    for k, why, src in FORBID:
        add("| " + Q + k + Q + " | " + why.replace("|", "/") + " | " + src + " |")
    add("")
    add("Everything else the product reads is scrubbed too: " + str(len(SURVEYED_ENV)) +
        " names were found in the 2026-11-03 survey of webui/*.py,")
    add("imaggers/image/t18_run.py and the device path.  ZIMAGE_TE_GGUF is kept (it")
    add("is a path to a blob, not a device switch).")
    add("")
    add("## 5. Preflight -- it refuses to start unless all of these pass")
    add("")
    add("| id | check | why it matters |")
    add("|---|---|---|")
    for cid, title, _fn in PF_CHECKS:
        add("| " + cid + " | " + title + " | " + PF_WHY.get(cid, "") + " |")
    add("")
    add("P02 is the real device gate: **a device.lock that is absent means nothing**")
    add("(run_job_py.py:31-33 unlinks it at startup, and a launcher that bypasses")
    add("that script never writes one).  Measured on 2026-11-03 at 16:05: a")
    add("webui\\bridge.py dispatch with 9.72 GB resident, actively on the NPU, with")
    add("**no lock file present**.  P02 waits up to " + str(DEVICE_WAIT_S) +
        " s for the device to go quiet,")
    add("then FAILS instead of racing it.")
    add("")
    add("P04 does not read a counter and hope: it **allocates " + ("%.0f" % PROBE_GB) +
        " GiB and touches every")
    add("page**, because that is what the --cache-all pass needs.  If the product's")
    add("own auto rule would pick low (commit < 9.0 GB), the job is submitted with")
    add("memory_mode=fast explicitly and the preflight says so in its PASS detail.")
    add("")
    add("P04 has a cost the check does not report about itself: the " +
        ("%.0f" % PROBE_GB) + " GiB it touches is anonymous memory, so it evicts")
    add("exactly the page cache the job is about to read (the 1.99 GB wpack")
    add("artifact, the 3.71 GB payload shards).  Measure it as a WITHIN-PAIR")
    add("comparison, where the power state cancels:")
    add("")
    add(Q*3 + "powershell")
    add("imaggers\\repro\\t22_run.ps1 -Size 128 -Steps 1 -Tag p04on")
    add("imaggers\\repro\\t22_run.ps1 -Size 128 -Steps 1 -Tag p04off -Skip P04")
    add(Q*3)
    add("")
    add("P06 has one WARN: an " + Q + "--extra" + Q + " arm that names a RECIPE variable.  The")
    add("value really changed, so the expected env is the recipe WITH the arm, the")
    add("arm is printed and stored, and the record's " + Q + "env_id" + Q + " is the arm's --")
    add("not the recipe's.  That is how the two startup arms of section 13 are")
    add("measured without a second recipe file, and why neither is mistaken for a")
    add("reference run.")
    add("")
    add("P06 is the ONLY place the ENTIRE recipe is compared BEFORE the run: it")
    add("checks all " + str(len(RECIPE)) + " variables in the environment the child is about")
    add("to inherit, and its " + Q + "env_id" + Q + " is the label of that comparison.  The")
    add("postflight does not trust the label on its own -- Q11 level 2 re-derives")
    add("the id from the env P06 stored (section 6.2).  That label -- not")
    add(Q + "job.json:env_flags" + Q + ", which can see only " + str(len(flag_recipe)) + " of the " +
        str(len(RECIPE)) + " -- is what makes a run's")
    add("configuration provable, and it is why section 6.2 has two levels.")
    add("")
    add("P11 reads the last run recorded for THIS recipe and shape -- same env_id,")
    add("same size, same steps -- and says whether it proved its path.  It is NOT")
    add('"the last run of anything": another agent\'s 384^2 arm is not your')
    add("predecessor.  When there is no such run it SKIPs (not a failure: Q12")
    add("records the reference on first use) and quotes the last run of anything")
    add("only as context.")
    add("")
    add("P12 records the power state and WARNS on battery; it never refuses, because")
    add("a battery number is a real number -- it is just not comparable.  Section 12")
    add("is the rule; section 13 is what " + Q + "startup_s" + Q + " is made of.")
    add("")
    add("## 6. Postflight -- it proves WHICH PATH RAN, not that the job exited 0")
    add("")
    add("### 6.1 what \"it passed\" means")
    add("")
    add("The registry below has **" + str(len(QF_CHECKS)) + " checks**.  A run is ACCEPTED iff")
    add("**FAIL == 0**, over every one of them.  That is the whole rule, and every")
    add("run prints it right after its table:")
    add("")
    add(Q*3 + "text")
    add("ACCEPTED: " + str(len(QF_CHECKS)) + " evaluated, 0 FAIL -- the path proof holds and the "
        "job's number is usable")
    add(Q*3)
    add("")
    add("* **SKIP is not a pass.**  It marks a check that is STRUCTURALLY N/A on")
    add("  this run's path (the " + Q + "applies to" + Q + " column below), and it is printed with")
    add("  the ids it skipped so nobody has to guess which one, or why.")
    add("* **WARN is not a refusal.**  It is a number to read and act on (the")
    add("  total-wall decomposition, the power state, an --extra arm).")
    add("* **\"18/18\" is not a thing.**  A 128^2 native run comes back 13 PASS +")
    add("  1 WARN + 4 SKIP + 0 FAIL: " + str(len(QF_CHECKS)) + " evaluated, 0 FAIL, and that is a pass.")
    add("  \"N/N PASS\" was never available on the native path, and a reader who")
    add("  expected it had to decide for themselves whether the run had passed.")
    add("  The engine prints this definition (ACCEPTANCE_RULE in the generator)")
    add("  after the table, so a run's console carries its own acceptance rule.")
    add("")
    add("| id | check | applies to | evidence it reads |")
    add("|---|---|---|---|")
    for cid, title, _fn in QF_CHECKS:
        add("| " + cid + " | " + title + " | " + Q_SCOPE.get(cid, "both paths") +
            " | " + QF_EVIDENCE.get(cid, "") + " |")
    add("")
    add("The four that matter most:")
    add("")
    add("* **Q04 vs Q05** -- installed 8/8 is not called.  Q05 counts the per-layer")
    add("  [fastglue] layers.N proj_qkv lines and requires exactly layers x steps.")
    add("  A run that installs and then never calls the hook FAILS here.  Both are")
    add("  CHUNKED ONLY: webui/chunked.py:988-1012 installs the hooks inside the")
    add("  ZIMAGE_BIGM + T22_DEVKILL arm of chunked.install(), which a native shape")
    add("  never reaches.")
    add("* **Q06** -- the same for the FFN chain: [devkill] rows=M lines AND")
    add("  [ffnprof] seq=M lines, each layers x steps.  That is why T22_FFN_PROF=1")
    add("  is in the recipe: without it the ffnchain hook has no call evidence at")
    add("  all, only installation evidence.  Q06 is SKIPped on the native path for a")
    add("  PRECISE reason, not a vague one: the [devkill] lines DO run there (30")
    add("  \"[devkill] rows=96\" lines per step, measured on 20261103-194246-47673d),")
    add("  but [ffnprof] is emitted by the fastglue install, which is chunked-only --")
    add("  so the check could only ever FAIL on native, and a structural N/A is the")
    add("  honest answer.")
    add("* **Q11** -- section 6.2: the env is proved at two levels, and only the")
    add("  second one is the whole recipe.")
    add("* **Q13** -- section 6.3: the wall is gated, and the total is DECOMPOSED")
    add("  into the loop and the startup it is made of.")
    add("")
    add("### 6.2 Q11 -- the env check has TWO levels, and only one is the whole recipe")
    add("")
    add("Q11 reads " + Q + "job.json:env_flags" + Q + " first.  That is " + Q + "LAUNCH_FLAGS" + Q + " (webui/config.py:")
    add("131-136): " + str(len(flags)) + " names, " + str(len(flag_recipe)) + " of them recipe variables.  It is a SUBSET, and no")
    add("amount of reading it can prove the other " + str(len(blind)) + " -- " + Q + "T22_REF_SMNB" + Q + " among")
    add("them, the one variable section 3 says moves the 512^2 latents.  A PASS")
    add("line that says \"the whole recipe\" while comparing a subset is exactly the")
    add("trap a cold reader fell into, so both levels are printed now:")
    add("")
    add("(the two lines below are one printed line each, wrapped for width)")
    add("")
    add(Q*3 + "text")
    add("LEVEL 1 job.json env_flags: all " + str(len(flags)) + " keys equal, but that tuple is")
    add("  LAUNCH_FLAGS -- " + str(len(flag_recipe)) + " of the " + str(len(RECIPE)) + " recipe variables -- so " + str(len(blind)) + " are")
    add("  INVISIBLE to it (" + ", ".join(blind[:4]) + (", ..." if len(blind) > 4 else "") + ")")
    add("LEVEL 2 env_id: the " + str(len(RECIPE)) + " values P06 recorded re-derive to " + env_id() + ",")
    add("  which IS this file's recipe id")
    add(Q*3)
    add("")
    add("* Level 2 is RE-DERIVED from the env stored in " + Q + "last/<tag>.preflight.json" + Q +
        ",")
    add("  not read from a string, so a stale label cannot fake it (a mismatch")
    add("  between the stored label and the re-derivation is reported).")
    add("* If that record is missing -- a postflight re-run in a later session --")
    add("  level 2 is a WARN and the line says the whole-recipe claim was NOT made:")
    add("  only level 1's subset was compared.  Run the preflight and the")
    add("  postflight from the same invocation (the " + Q + "run" + Q + " action does) to keep it.")
    add("* A reference whose " + Q + "env_id" + Q + " differs from today's recipe is **not** a")
    add("  failure: the 512^2 x8 reference arms were measured under the pre-cache")
    add("  recipe id (section 12), and a byte-neutral variable joining the recipe")
    add("  cannot move bytes.  Q11 compares the ENV of this run; Q12 compares the")
    add("  BYTES -- and Q12 is the only one of the two that can catch a wrong")
    add("  " + Q + "T22_REF_SMNB" + Q + ".")
    add("")
    add("### 6.3 Q13 -- the wall has a gate, and the total is decomposed")
    add("")
    add("Q13 has two parts and one verdict.")
    add("")
    add("* **loop_s** -- a WARN window of +-" + ("%.0f" % (100 * 0.25)) + "% around the reference loop.  It")
    add("  is not a byte gate (two identical-sha 512^2 arms differed 0.2%), but a")
    add("  swing that large usually means a different path.")
    add("* **the total wall** -- which had NO gate at all, so a reader had to decide")
    add("  alone whether a 186.2 s total against a 124.9/135.7 s reference (i.e.")
    add("  +37..48%) was a failure.  Comparing " + Q + "total_s" + Q + " directly is wrong twice")
    add("  over: the reference arms ran " + Q + "vae=False" + Q + " while this engine submits")
    add("  " + Q + "vae=True" + Q + " (+~10 s at 512^2), and " + Q + "startup_s" + Q + " is the 15-30 s row that")
    add("  moves with the power state and with the B-repack arm (sections 12, 13).")
    add("  So the gate is on the **diffusion core = startup_s + loop_s** -- for a")
    add("  vae-off arm that IS its total minus the child's import/TE, ~1 s --")
    add("  against the reference total, with a +-" + ("%.0f" % (100 * CORE_TOL)) + "% window")
    add("  (CORE_TOL in the generator).  For a " + Q + "vae=yes" + Q + " reference row the band")
    add("  is ~10 s looser (its total carries the decode the core does not), which")
    add("  is the safe direction: it can miss a slow run by that much, it cannot")
    add("  fail a good one by it.")
    add("")
    add("When the core is outside that window Q13 DECIDES instead of reporting:")
    add("")
    add("* **the excess is in startup_s -> WARN.**  The number is real and it is")
    add("  not the loop; sections 12/13 and Q18 name what moves startup.  The")
    add("  186.2 s case is exactly this: loop +6.2% (inside its window) and startup")
    add("  68 s instead of ~40 s -> WARN, not a failure, and no re-run needed.")
    add("* **the excess is in the loop -> FAIL.**  The diffusion work itself is")
    add("  outside its window, so the number is not comparable to the reference")
    add("  (Q12 still decides whether the bytes are right).")
    add("")
    add("The FAIL branch cannot be reached by the power state alone: the measured")
    add("cross-state drift of a multi-step " + Q + "loop_s" + Q + " is x1.075 (section 12)")
    add("against a +-25% window, while what the power state does move is " + Q + "startup_s" + Q)
    add("(x1.50) -- which is the WARN branch.  A FAIL here therefore means the")
    add("diffusion work, not the box.")
    add("")
    add("The split is printed, so \"the total is +40%, but the loop is +6% and")
    add("startup is 68 s\" is in the record instead of in somebody's head:")
    add("")
    add(Q*3 + "text")
    add("core startup +68.39 + loop 99.87 = 168.26 s vs reference total 124.9 /")
    add("  135.7 s (+24.0%, window +-20%); job total_s 186.19 s; excess +32.56 s =")
    add("  5.82 s loop + 26.74 s startup, so 82% of it is startup")
    add(Q*3)
    add("")
    add("## 7. References (what a successful run measured)")
    add("")
    add("| shape:steps | path | M | vae | latents sha256 (all 64 hex) | loop s | total s | provenance |")
    add("|---|---|---|---|---|---|---|---|")
    for k in sorted(refs):
        r = refs[k]
        tot = r.get("total_s")
        if isinstance(tot, list):
            tot_s = " / ".join("%.1f" % float(x) for x in tot)
        elif isinstance(tot, (int, float)) and tot:
            tot_s = "%.1f" % float(tot)
        else:
            tot_s = "-"
        loop_s = r.get("loop_s")
        vae_s = ("yes" if r.get("vae") else "no") if "vae" in r else "?"
        add("| " + k + " | " + str(r.get("path")) + " | " + str(r.get("m")) + " | " +
            vae_s + " | " + (r.get("latents_sha256") or "-") + " | " +
            (("%.2f" % float(loop_s)) if loop_s else "-") + " | " + tot_s + " | " +
            ", ".join(r.get("ref_jobs") or []) + " |")
    add("")
    add("A key absent from this table is RECORDED on the first run of that")
    add("shape+steps and compared from then on (imaggers/repro/reference.json).")
    add("")
    add("**The sha256 column is the whole value, all 64 hex characters, and that is")
    add("what Q12 compares.**  The 16-character prefixes used in this document's")
    add("prose (" + Q + "2abf91331200d069" + Q + ", " + Q + "ba89b71cfd1dfaa0" + Q + ") are abbreviations for")
    add("reading only.  Do not confuse the two 16-hex strings in this file: the")
    add("latents sha256 is 64 hex, the " + Q + "env_id" + Q + " is 16 hex, and neither is a")
    add("prefix of the other.")
    add("")
    add("**The " + Q + "total" + Q + " column is not one quantity either**, which is why its rows")
    add("are not directly comparable:")
    add("")
    add("* " + Q + "vae=no" + Q + " -- a reference arm measured with the VAE stage OFF: its")
    add("  total is startup + import/TE + loop.")
    add("* " + Q + "vae=yes" + Q + " -- a run of this engine (which submits vae=True): its")
    add("  total includes the decode, +~10 s at 512^2.")
    add("")
    add("That is exactly why Q13 gates the DIFFUSION CORE (startup_s + loop_s) against")
    add("the reference total instead of total_s against total_s (section 6.3), and it")
    add("is why the loop wall is only a WARN: two runs with identical latents differed")
    add("0.2% in loop and 8.7% in total, because the 14.5-19.4 s B repack lands inside")
    add("total and outside loop.")
    add("")
    add("## 8. Reading a result")
    add("")
    add("| file | what is in it |")
    add("|---|---|")
    for f, what in [
            ("imaggers/repro/last/<tag>.preflight.json",
             "every check, its PASS/FAIL and the reason, plus the ENV this run "
             "had and its env_id (Q11 level 2 re-derives that id from this "
             "file)"),
            ("imaggers/repro/last/<tag>.device.json",
             "what the device looked like when the job ended: the lock line, any "
             "other device process, the port (section 9.1)"),
            ("imaggers/repro/last/<tag>.job.json",
             "the driver-side copy of the job record"),
            ("imaggers/repro/last/<tag>.postflight.json",
             "the path proof, per item"),
            ("imaggers/repro/last/<tag>.server.err.log",
             "the server's own stderr"),
            ("webui/jobs/<id>/job.log",
             "the authoritative product log the postflight reads"),
            ("webui/jobs/<id>/job.json",
             "memory_mode, env_flags, timing.closure, latents_sha256"),
            ("imaggers/repro/reference.json",
             "references recorded by earlier runs")]:
        add("| " + f + " | " + what + " |")
    add("")
    add("The exit code is 0 only if no check FAILed.  That is section 6.1 in one")
    add("sentence: **N evaluated, 0 FAIL**; WARN and SKIP are printed and never fail")
    add("a run.  A run whose checks all pass still prints a " + Q + "[device] ..." + Q + " line at")
    add("the end -- the device state AFTER the job is deliberately not part of the")
    add("proof (section 9.1), because another agent's job may legitimately own the")
    add("NPU by then.")
    add("")
    add("## 9. When it fails")
    add("")
    add("| symptom | meaning | what to do |")
    add("|---|---|---|")
    for sym, mean, act in DIAGNOSE:
        add("| " + sym + " | " + mean.replace("|", "/") + " | " +
            act.replace("|", "/") + " |")
    add("")
    add("### 9.1 after the run: the device states, and which one means trouble")
    add("")
    add("A finished job leaves the device in one of the states below.  The engine")
    add("prints it when the server stops (" + Q + "[run] post-run device state: ..." + Q + ") and")
    add("again at the end of the postflight (" + Q + "[device] when this job ended /" + Q + "")
    add(Q + "[device] now" + Q + "), and stores it in " + Q + "last/<tag>.device.json" + Q + ".")
    add("**None of these is part of the path proof.**  The postflight reads FILES")
    add("only and never touches the NPU, so it is valid even while somebody else's")
    add("job owns the device -- including the \"held by another pid\" case, which is")
    add("the one a reader is most likely to mistake for a failure.")
    add("")
    add("| what it says | what it means | what to do |")
    add("|---|---|---|")
    for _state, _mean, _act in POSTRUN_STATES:
        add("| " + Q + _state + Q + " | " + _mean.replace("|", "/") + " | " +
            _act.replace("|", "/") + " |")
    add("")
    add("## 10. Platform laws a fresh agent must not re-learn")
    add("")
    for fid, law, src in LAWS:
        add("* **" + fid + "** -- " + law + "  (" + src + ")")
    add("")
    add("## 11. What this runbook does NOT prove, and the code fixes that would")
    add("")
    add("Each item below is a silent fallback this harness works AROUND.  Fixing it")
    add("in code would make the workaround unnecessary:")
    add("")
    for i, (where, what, fix) in enumerate(FIXES, 1):
        add(str(i) + ". " + Q + where + Q + " -- " + what)
        add("   fix: " + fix)
    add("")
    add("Known blind spots of the postflight itself:")
    add("")
    add("* The loop's EFFECTIVE torch thread count has no product log line.  Q10")
    add("  reads the value the server recorded plus a runtime probe of the same")
    add("  code path, which is the strongest evidence available today (fix 1).")
    add("* Of the 8 fastglue hooks, 2 (attn, ffnchain) have per-call log lines and 6")
    add("  do not.  Q04 proves all 8 installed; Q05 and Q06 prove 2 of them ran.  A")
    add("  call count for the other 6 needs fix 2.")
    add("* The PNG sha is recorded, and gated only once a reference exists for that")
    add("  exact shape+steps; it is byte-stable only at a fixed thread count.")
    add("* Which startup arm a run was in is provable from the log (Q18) and")
    add("  from this run's " + Q + "env_id" + Q + " (Q11 level 2, which covers")
    add("  " + Q + "T22_BREPACK_CACHE" + Q + "), but NOT from " + Q + "job.json:env_flags" + Q + ": that is")
    add("  not in " + Q + "LAUNCH_FLAGS" + Q + ", so the job record alone cannot distinguish a")
    add("  mapped run from a repacked one (fix 11).")
    add("* " + Q + "job.json" + Q + " itself carries no env dump, so the whole-recipe proof")
    add("  (Q11 level 2) rests on the preflight record this tool wrote.  If that")
    add("  record is gone, the claim downgrades to a WARN instead of being faked.")
    add("")
    add("## 12. Power state -- what is comparable to what")
    add("")
    add("This box is a laptop and its wall time moves with the power state.  The")
    add("project already recorded that, twice:")
    add("")
    add("* P33-E2E-REPORT.md section 7: the machine was on battery ~08:20-10:10 on")
    add("  2026-11-03 and on AC after that; the SAME configuration measured")
    add("  89.98-94.45 s on AC and 96.5-135.4 s on battery, and the battery-era")
    add("  numbers were DISCARDED rather than compared.")
    add("* imaggers/bigpipe2/cyc/CYC-REPORT.md: hardware cycle counts do not move")
    add("  with the power state; every WALL time does.")
    add("")
    add("Those two anchors are directional only: neither was a same-recipe pair.")
    add("On 2026-11-03 the pair WAS measured -- same tree, same recipe (env_id")
    add(PRE_CACHE_ENV_ID + ", i.e. the recipe BEFORE " + Q + "T22_BREPACK_CACHE=1" +
        Q + " joined it),")
    add("same shape, ONE latents sha per pair, once on battery and once on AC,")
    add("minutes apart:")
    add("")
    add("| metric | shape | battery s | AC s | battery / AC | battery - AC |")
    add("|---|---|---|---|---|---|")
    for _metric, _shape, _bat, _ac in POWER_CAL:
        add("| " + Q + _metric + Q + " | " + _shape + " | " + ("%.2f" % _bat) + " | " +
            ("%.2f" % _ac) + " | " + ("%.3f" % (_bat / _ac)) + " | " +
            ("%+.2f" % (_bat - _ac)) + " |")
    add("")
    add("Jobs: battery " + " + ".join(POWER_CAL_JOBS["battery"]) + ", AC " +
        " + ".join(POWER_CAL_JOBS["ac"]) + ".")
    add("")
    add("**The recipe id moved.**  " + PRE_CACHE_ENV_ID + " was the recipe with the")
    add("B-repack cache SCRUBBED; adding " + Q + "T22_BREPACK_CACHE=1" + Q + " to the")
    add("recipe (2026-11-03 ~19:20) made the id **" + env_id() + "**.  Every number in")
    add("this section belongs to the OLD id -- it was measured with the 27 s")
    add("permutation PAID -- so it is a calibration constant for the power state,")
    add("NOT a baseline for a run of today's recipe.  Section 13's pair is the")
    add("comparable measurement, taken inside one power state.")
    add("")
    add("That is the coefficient this project never had, and its SHAPE is the")
    add("finding: " + Q + "startup_s" + Q + " and " + Q + "total_s" + Q + " move x1.42-1.50 with")
    add("the power state while " + Q + "loop_s" + Q + " at 512^2 x8 moves x1.075.  A 68 s")
    add("startup in a battery record is therefore NOT a 68 s startup -- and it is not")
    add("a code regression either, because the latents sha and the loop did not move.")
    add("At 128^2 x1 the LOOP moved x1.52, because a 1-step loop carries torch's")
    add("first call (section 13).  A 1-step job is a startup measurement, not a")
    add("compute measurement.")
    add("")
    add("So the power state is PROBED and RECORDED, never assumed, and it is the")
    add("first thing to check before calling a difference a regression.")
    add("")
    add("| what | how | what it gives you |")
    add("|---|---|---|")
    add("| AC / battery | " + Q + POWER_PROBE + Q + " | " + Q + "ACLineStatus" +
        Q + ": 1 = AC, 0 = BATTERY, 255 = unknown |")
    add("| the plan | " + Q + "powercfg /getactivescheme" + Q + " | " + Q +
        "381b4222-f694-41f0-9685-ff5bb260df2e" + Q + " = Balanced (shipped plan) |")
    add("| magnitude | sha256 over 64 MiB, ~0.2 s | MiB/s travels with every record")
    add("  as " + Q + "power.sha256_mib_s" + Q + ", so a run carries its own speed index |")
    add("")
    add("Probes that DO NOT work in this session -- all access-denied, written down")
    add("so that nobody reads \"no battery object\" as \"a desktop\":")
    add(Q + "Get-CimInstance Win32_Battery" + Q + ", " + Q +
        "Get-CimInstance Win32_ComputerSystem" + Q + ",")
    add(Q + "Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus" + Q + ",")
    add(Q + "Get-CimInstance -Namespace root/cimv2/power -ClassName Win32_PowerPlan" + Q + ",")
    add(Q + "Get-CimInstance Win32_Processor" + Q + ", " + Q + "powercfg /batteryreport" + Q + ".")
    add("")
    add("The ones that DO work: " + Q + "powercfg /getactivescheme" + Q + " (the plan),")
    add(Q + "powercfg /query SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX" + Q + ",")
    add(Q + "Get-Counter" + Q + " (works, e.g. " + Q + "% Processor Performance" + Q +
        "), and the ctypes")
    add("kernel32 probe above -- which is the one this file records.")
    add("")
    add("THE RULE:")
    add("")
    add("* **" + POWER_RULE + "**")
    add("* Two numbers may be compared only if both records carry the SAME " + Q +
        "power.state" + Q + ",")
    add("  the same recipe " + Q + "env_id" + Q + ", and neither is an " + Q + "--extra" + Q + " arm.")
    add("* Numbers that MAY cross states: " + Q + "loop_s" + Q + " of a multi-step job, and")
    add("  then only as a magnitude (x1.075 measured at 512^2 x8) -- never as a gate.")
    add("* Numbers that may NOT cross states: " + Q + "startup_s" + Q + " (x1.50), " + Q +
        "total_s" + Q + " (x1.23),")
    add("  every startup phase (x1.3-3.0), " + Q + "vae_load_s" + Q + ", " + Q +
        "bridge_import_s" + Q + ", and the")
    add("  " + Q + "loop_s" + Q + " of a 1-step job (x1.52, because it carries torch's first call).")
    add("* On battery: compare WITHIN a back-to-back pair, minutes apart.  Do NOT")
    add("  compare a battery number to an AC reference, and do NOT call the")
    add("  difference a code regression.")
    add("* If the state was not recorded, the honest answer is \"unknown, power state")
    add("  differs\" -- not a percentage.")
    add("* " + Q + "startup_s" + Q + " is a WALL, so it moves with the power state.  Its")
    add("  COMPOSITION (section 13) is power-independent; its magnitude is not:")
    add("  the same 45.64 s of AC work is 68.39 s on battery.")
    add("")
    add("## 13. What " + Q + "startup_s" + Q + " is made of, and how to re-derive it")
    add("")
    add(Q + "startup_s" + Q + " is ONE span: the child bridge's first " + Q + "@@WEBUI@@" + Q + " event")
    add("-> the " + Q + "startup_done" + Q + " event (webui/runner.py:1400-1416).  It is big")
    add("because it is the payload-residency pass, and none of it is the loop:")
    add("")
    add("| phase | what it costs | where it is printed |")
    add("|---|---|---|")
    for _key, _what, _src in STARTUP_PHASES:
        add("| " + Q + _key + Q + " | " + _what + " | " + _src + " |")
    add("")
    add("A job's own log already carries each of those seconds, so the decomposition")
    add("is re-derivable from artefacts alone -- no instrumentation and no guess:")
    add("")
    add(Q*3 + "powershell")
    add("py312 imaggers\\startup\\compose.py <job-id> [<job-id> ...]")
    add(Q*3)
    add("")
    add("MEASURED 2026-11-03, same tree, same recipe, seconds:")
    add("")
    add("| phase | " + " | ".join(_label for _arm, _label in STARTUP_ARMS) + " |")
    add("|" + "---|" * (len(STARTUP_ARMS) + 1))
    for _key, _what, _src in STARTUP_PHASES:
        _row = [("%.2f" % STARTUP_MEASURED[_arm][_key]) for _arm, _l in STARTUP_ARMS]
        add("| " + _key + " | " + " | ".join(_row) + " |")
    add("| **" + Q + "startup_s" + Q + " total** | **" +
        "** | **".join("%.2f" % STARTUP_MEASURED[_arm]["startup_s"]
                       for _arm, _l in STARTUP_ARMS) + "** |")
    add("")
    add("Three things follow, and together they answer \"is 68 s a regression\":")
    add("")
    add("1. The COMPOSITION changed.  The 06:30 arm had " + Q + "T22_BREPACK_CACHE=1" +
        Q + ",")
    add("   inherited from its shell, so it MAPPED a pre-permuted payload and paid")
    add("   3.02 s; the 17:xx and 18:xx arms ran the recipe AS IT WAS THEN, which")
    add("   SCRUBBED that switch, so they paid the real 5.31 GB permutation")
    add("   (brepack_pass) AND the 3.71 GB payload read (payload_load) that a")
    add("   mapped cache makes free.  That is ~38-41 s of the difference and it is")
    add("   work the Nov-2 reference did not do either.  It is not a slow machine")
    add("   and it is not a re-defined bucket: the bucket boundaries are unchanged")
    add("   since 06:30 (both arms report the same " + Q + "startup_child_s" + Q +
        " == " + Q + "startup_s" + Q + " split), only the WORK inside it changed.")
    add("   AS OF 2026-11-03 ~19:20 the recipe SETS the switch (section 3), so this")
    add("   table is the LAST measurement of the un-mapped arm -- kept because it")
    add("   is the baseline the pair below is measured against.")
    add("2. The MAGNITUDE changed too.  Every phase that BOTH arms pay is slower on")
    add("   battery, and the two AC arms (18:33/18:37, same recipe, byte-identical")
    add("   latents) size it: wpack verify 1.61 -> 2.10, devkill prewarm 3.94 ->")
    add("   6.10, warm wpack 1.70 -> 3.50, payload load 3.20 -> 9.70, refiner 1.20")
    add("   -> 2.50, i.e. 1.3-3.0x per phase.  That is the power-state drift of")
    add("   section 12, not code.")
    add("3. THE POWER SHARE IS NOW A MEASURED NUMBER, not a range: " + Q + "startup_s" +
        Q + " 45.64")
    add("   on AC -> 68.39 on battery (x1.50), " + Q + "loop_s" + Q + " 92.87 -> 99.87")
    add("   (x1.075), " + Q + "total_s" + Q + " 150.94 -> 186.19 (x1.23).  So of the 68.39 s,")
    add("   **22.75 s was the power state and 45.64 s was work the recipe really")
    add("   did** (27.02 s of it the B repack, which the recipe now MAPS instead")
    add("   -- the pair below).  Neither is a regression: a regression moves the")
    add("   latents and the loop, and neither moved.")
    add("")
    add("### the B repack removed -- WITHIN-WINDOW pairs")
    add("")
    if not STARTUP_PAIR:
        add("Not measured yet (STARTUP_PAIR is empty in the generator).")
    for _pair in STARTUP_PAIR:
        _arms = _pair["arms"]
        _offs = [_a for _a in _arms if _a["cache"] == "0"]
        _ons = [_a for _a in _arms if _a["cache"] == "1"]
        add("")
        add("#### " + _pair["shape"] + " -- " + _pair["state"] + ", " + _pair["when"])
        add("")
        add(_pair["why"].replace("|", "/"))
        add("")
        add("| arm | " + Q + "T22_BREPACK_CACHE" + Q + " | battery | P12 speed | job | "
            "latents sha16 | " + Q + "startup_s" + Q + " | " + Q + "loop_s" + Q +
            " | " + Q + "total_s" + Q + " |")
        add("|---|---|---|---|---|---|---|---|---|")
        for _a in _arms:
            _m = STARTUP_MEASURED[_a["key"]]
            add("| " + _a["label"] + " | " + _a["cache"] + " | " +
                str(_a["batt"]) + "% | " + str(_a["mibs"]) + " MiB/s | " +
                _a["job"] + " | " + _a["latents"] + " | " +
                ("%.2f" % _m["startup_s"]) + " | " +
                ("%.2f" % _m["loop_s"]) + " | " +
                ("%.2f" % _m["total_s"]) + " |")
        add("")

        def _delta(_k, _offs=_offs, _ons=_ons):
            if not _offs or not _ons:
                return "-"
            _o = sum(STARTUP_MEASURED[_x["key"]][_k] for _x in _offs) / len(_offs)
            _n = sum(STARTUP_MEASURED[_x["key"]][_k] for _x in _ons) / len(_ons)
            return "%+.2f" % (_n - _o)

        add("| phase | " + " | ".join(_a["label"] for _a in _arms) +
            " | ON - mean(OFF) |")
        add("|" + "---|" * (len(_arms) + 1))
        for _key, _what, _src in STARTUP_PHASES:
            add("| " + _key + " | " +
                " | ".join("%.2f" % STARTUP_MEASURED[_a["key"]][_key]
                           for _a in _arms) + " | " + _delta(_key) + " |")
        add("| **" + Q + "startup_s" + Q + " total** | **" +
            "** | **".join("%.2f" % STARTUP_MEASURED[_a["key"]]["startup_s"]
                           for _a in _arms) + "** | **" + _delta("startup_s") +
            "** |")
        add("")
        add("Every arm carries the same latents sha16 (Q12 is the gate), so the")
        add("table is a cost difference and nothing else.  " + Q + "loop_s" + Q + " is")
        add("printed because it must not move either: the repack was weight")
        add("preprocessing inside startup, never a step.")
        if any(STARTUP_MEASURED[_a["key"]]["bridge_jit"] < 0 for _a in _arms):
            add("")
            add("A NEGATIVE " + Q + "bridge_jit" + Q + " is overlap, not a missing bucket: the")
            add("cache's map+verify (brepack_hit) runs on a worker thread while the")
            add("warm pack runs on the main one, so two rows cover the same seconds.")
            add("The bucket is still defined as startup_s minus the rows above; it is")
            add("the rows that are not disjoint.")
        if len(_offs) > 1:
            _spread = (max(STARTUP_MEASURED[_a["key"]]["startup_s"] for _a in _offs)
                       - min(STARTUP_MEASURED[_a["key"]]["startup_s"] for _a in _offs))
            add("")
            add("The " + str(len(_offs)) + " OFF arms differ by **%.2f s** of startup_s"
                % _spread)
            add("-- that is this box's run-to-run spread inside one window (the P12")
            add("speed index moves with it: " +
                ", ".join(str(_a["mibs"]) for _a in _arms) + " MiB/s).  The saving")
            add("is therefore quoted against BOTH OFF arms, not against one.")
    add("")
    add("### every second of " + Q + "startup_s" + Q + " on AC")
    add("")
    add("job 20261103-183357-58b472, 512^2 x8, AC, " + Q + "startup_s" + Q + " = 45.64 s.")
    add("Row 1 is an anchor delta; rows 2-8 are the product's own log lines; rows")
    add("9-11 are the segments no product line prints, measured in the same py310")
    add("interpreter by " + Q + "imaggers/startup/startup_probe.py" + Q + "; the last row")
    add("closes the table.")
    add("")
    add("| bucket | what it is | seconds | where the number comes from |")
    add("|---|---|---|---|")
    for _bucket, _what, _sec, _src in STARTUP_AC_LEDGER:
        add("| " + _bucket + " | " + _what.replace("|", "/") + " | " +
            ("%.2f" % _sec) + " | " + _src.replace("|", "/") + " |")
    add("| **" + Q + "startup_s" + Q + " total** | **the whole span** | **" +
        ("%.2f" % sum(_s for _b, _w, _s, _r in STARTUP_AC_LEDGER)) + "** | " +
        "first_event -> startup_done |")
    add("")
    add("The " + Q + "chunk_path install" + Q + " row is the only part no artefact names.")
    add("It is chunked-path work, not device time: the native 128^2 x1 arm's whole")
    add("residual is 0.60 s and already contains " + Q + "[t18]" + Q + " parse, " + Q +
        "t8" + Q + ", the device")
    add("context and 21 image registrations.  On battery this row is 12.76 s")
    add("(x1.83) -- the most power-sensitive row in the table, which is the shape")
    add("of a memory/CPU-bound install, not a device one.")
    add("")
    add("### the seconds of the same job that are NOT inside " + Q + "startup_s" + Q)
    add("")
    add("Quoted because they are the buckets people expect to find here (TE init,")
    add("the VAE loader, the child's spawn) and they are real -- they are simply")
    add("outside the span " + Q + "webui/runner.py:1396-1424" + Q + " books.")
    add("")
    add("| segment | seconds | where it is booked |")
    add("|---|---|---|")
    for _seg, _sec, _src in STARTUP_OUTSIDE:
        add("| " + _seg + " | " + ("%.2f" % _sec) + " | " + _src.replace("|", "/") + " |")
    add("")
    add("Step 0's first layer is where the loop's own warmup lands: " + Q +
        "[tim] img 1/1 first")
    add("layer: ... host 5.54 s" + Q + " on AC (9.80 s on battery) against ~0.3 s of")
    add("steady-state per-layer host work in the " + Q + "[fastglue]" + Q + " and " +
        Q + "[devkill]" + Q + " lines --")
    add("torch's first call, not a diffusion cost.  That is why")
    add("a 1-step job's " + Q + "loop_s" + Q + " is NOT a compute number (section 12: it moved")
    add("x1.52 with the power state, while the 8-step loop moved x1.075).")
    add("")
    add("Re-derive it from the artefacts:")
    add("")
    add(Q*3 + "powershell")
    add("py312 imaggers\\startup\\compose.py <job-id>    # the per-phase rows")
    add("py312 imaggers\\startup\\startup_probe.py 3     # torch/numba/fastglue")
    add(Q*3)
    add("")
    add("What " + Q + "startup_s" + Q + " does NOT contain: the wpack pack itself.  A present")
    add("artifact costs map+verify only; the 33.035 s in its manifest is the BUILD")
    add("time from 2026-11-02T15:22:06 and is booked to " + Q + "prewarm_pack_wall_s" + Q + ",")
    add("which is not a bucket.  An earlier draft of P09 said \"33 s of the job's")
    add("wall\" -- that sentence was the mis-measurement, and P09 now says so.")
    add("")
    text = ("\n".join(L) + "\n").replace(BT, chr(96))

    if check:
        old = RUNBOOK.read_text(encoding="utf-8") if RUNBOOK.is_file() else ""
        if old == text:
            out("[doc] RUNBOOK.md is up to date (%d chars)" % len(text))
            return 0
        out("[doc] RUNBOOK.md DIFFERED from the generator -- rewritten")
        RUNBOOK.write_text(text, encoding="utf-8")
        return 1
    RUNBOOK.write_text(text, encoding="utf-8")
    out("[doc] wrote %s (%d chars, %d recipe vars, %d preflight + %d "
        "postflight checks)"
        % (RUNBOOK, len(text), len(RECIPE), len(PF_CHECKS), len(QF_CHECKS)))
    return 0


# ---------------------------------------------------------------------------
# 13. CLI
# ---------------------------------------------------------------------------
def cache_cmd(build=False):
    """Maintenance verb: resolve (and optionally BUILD) the B-repack cache.

    CPU only -- no device, no lock, no job.  It exists so that the one-off
    5.31 GB build is a MEASURED, REPORTED step instead of the 27 s a job pays
    invisibly inside startup_s when the identity misses.  P09 refuses to start
    a run that would pay it, so this is the documented way to fix that.

    The build is the same code path the CLI (brepack_cache.py --build) uses:
    read the 60 sources into the product's resident layout, repack them in
    place, write the file, re-verify it, and print the seconds per stage.
    """
    env, scrubbed, conflicts = apply_env(verbose=False)
    out("[cache] recipe applied: %d variables, env_id %s; %d inherited name(s) "
        "scrubbed" % (len(RECIPE), env_id(), len(scrubbed)))
    try:
        out("[cache] power " + power_line(power_state()[1]))
    except Exception as exc:                              # noqa: BLE001
        out("[cache] power UNPROBEABLE (%r) -- the build seconds have no power "
            "state attached" % (exc,))
    d = brepack_state()
    out("[cache] enabled=%s  dir=%s  alt=%s"
        % (d["enabled"], d["dir"], d["alt_kind"] or "?"))
    if d["ok"]:
        out("[cache] REUSABLE: %s" % brepack_line(d))
        out("[cache] the next job maps it instead of paying the 5.31 GB "
            "permutation (27.02 s measured on AC, 31.7 s on battery)")
        return 0
    out("[cache] NOT REUSABLE: %s" % (d["reason"]))
    if not build:
        out("[cache] re-run with --build to derive it once, here, and time it "
            "(the same work a job would otherwise pay inside startup_s)")
        return 1
    try:
        BRC = brepack_modules()
    except Exception as exc:                              # noqa: BLE001
        out("[cache] cannot import brepack_cache: %r" % (exc,))
        return 2
    prep = BRC.prep_default()
    t0 = time.perf_counter()
    try:
        bg, b3 = BRC._fill_sources(prep, BREPACK_LAYERS, BREPACK_NH_BG)
    except Exception as exc:                              # noqa: BLE001
        out("[cache] BUILD FAILED reading %s: %r" % (prep, exc))
        return 2
    t_read = time.perf_counter() - t0
    out("[cache] sources resident in %.2f s (%.2f + %.2f GB)"
        % (t_read, bg.nbytes / 2**30, b3.nbytes / 2**30))
    old = os.environ.get("T22_BREPACK_WRITE_SYNC")
    os.environ["T22_BREPACK_WRITE_SYNC"] = "1"   # this process only: time the write
    try:
        got = BRC.build(bg, b3, prep, BREPACK_LAYERS, BREPACK_NH_BG,
                        log=lambda m: out("  " + str(m)))
    finally:
        if old is None:
            os.environ.pop("T22_BREPACK_WRITE_SYNC", None)
        else:
            os.environ["T22_BREPACK_WRITE_SYNC"] = old
    t_all = time.perf_counter() - t0
    hdr = (got or {}).get("hdr") or {}
    out("[cache] BUILD WALL %.2f s total = %.2f s read + %s s repack + %s s "
        "write + %s s verify"
        % (t_all, t_read, (got or {}).get("repack_s", "?"),
           hdr.get("write_s", "?"), hdr.get("verify_s", "?")))
    d2 = brepack_state()
    if not d2["ok"]:
        out("[cache] the build finished but the identity still does not "
            "resolve: %s" % d2["reason"])
        return 2
    out("[cache] FILE %s  %.2f GiB (%d B)  built %s  key16=%s"
        % (d2["path"], d2["bytes"] / 2**30, d2["bytes"], d2["built"],
           d2["key16"]))
    out("[cache] re-verified by this process: header identity + size match "
        "(the 5.31 GB content is re-hashed by every job's own load)")
    return 0


def _size(s):
    s = str(s).lower().replace(" ", "")
    if "x" in s:
        a, b = s.split("x", 1)
        return int(a), int(b)
    return int(s), int(s)


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog=NAME,
                                 description="one reproducible T22 device run")
    ap.add_argument("action", nargs="?", default="run",
                    choices=["run", "preflight", "postflight", "doc", "status",
                             "cache"])
    ap.add_argument("--build", action="store_true",
                    help="with cache: build the B-repack file (CPU only, no "
                         "device, no lock) and report the seconds per stage")
    ap.add_argument("--size", default="512", help="512 or 512x384")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--tag", default="")
    ap.add_argument("--check", action="store_true",
                    help="with doc: only report whether RUNBOOK.md is current")
    ap.add_argument("--extra", action="append", default=[],
                    metavar="NAME=VALUE",
                    help="A/B arm only: set one variable AFTER the recipe and "
                         "the scrub (recorded; the run is then an ARM, not a "
                         "reference).  Naming a RECIPE variable is allowed and "
                         "is exactly how the two startup arms are measured: "
                         "P06 warns, the record keeps the arm's env_id")
    ap.add_argument("--skip", action="append", default=[], metavar="CHECK",
                    help="A/B arm only: drop one preflight check by id (e.g. "
                         "P04) so its effect on the machine can be measured "
                         "against the same job with it on")
    a = ap.parse_args(argv)
    global EXTRAS, SKIP_CHECKS
    EXTRAS = list(a.extra or [])
    SKIP_CHECKS = [s.strip().upper() for s in (a.skip or []) if s.strip()]
    if a.action == "doc":
        return doc(check=a.check)
    if a.action == "cache":
        return cache_cmd(build=bool(a.build))
    if a.action == "status":
        out("repo          %s" % ROOT)
        out("env_id        %s" % env_id())
        try:
            out("power         " + power_line(power_state()[1]))
        except Exception as exc:                              # noqa: BLE001
            out("power         ERROR %s" % exc)
        present, txt, owner, pid, alive = lock_state()
        out("lock          %s" % (("present owner=%s pid=%s alive=%s" %
                                  (owner, pid, alive)) if present else "absent"))
        out("port %d       %s" % (PORT, "open" if port_busy() else "free"))
        for u in device_users():
            out("device user   pid %s %s rss %s GB" %
                (u["pid"], u["name"], u["rss_gb"]))
        try:
            out("memory        %s" % json.dumps(memory_snapshot()))
        except Exception as exc:
            out("memory        ERROR %s" % exc)
        idx = RECORDS / "index.json"
        if idx.is_file():
            d = json.loads(idx.read_text(encoding="utf-8"))
            for r in (d.get("runs") or [])[-5:]:
                out("last run      %-12s %s %sx%s:%s fails=%s" %
                    (r.get("tag"), r.get("when"), r.get("w"), r.get("h"),
                     r.get("steps"), r.get("postflight_fail")))
        return 0
    tag = announce_tag(a.tag, safe_tag(a.tag or ("t" + time.strftime("%H%M%S"))))
    h, w = _size(a.size)
    if a.action == "postflight":
        results, ctx = postflight(tag)
        n = report(results, "POSTFLIGHT %s  job %s" % (tag, ctx["job_id"]),
                   "postflight")
        device_footer(ctx)
        out("[record] %s" % (RECORDS / (tag + ".postflight.json")))
        return 0 if n == 0 else 2
    hr("=")
    out("POWER STATE (recorded with this run; a number is only comparable to "
        "one taken in the SAME state)")
    hr("=")
    pf, ctx = preflight(tag, h, w, a.steps)
    n_pf = report(pf, "PREFLIGHT %s  %dx%d  steps=%d" % (tag, h, w, a.steps),
                  "preflight")
    ctx["steps"] = a.steps
    save_run("preflight", tag, ctx, pf)
    if n_pf:
        out("REFUSING TO START: %d preflight check(s) failed." % n_pf)
        return 2
    if a.action == "preflight":
        return 0
    attempt = 1
    while True:
        atag = tag if attempt == 1 else "%s-r%d" % (tag, attempt)
        if attempt > 1:
            out("[retry] attempt %d (tag %s) after another process took the "
                "device" % (attempt, atag))
            ok, waited, users = wait_quiet(deadline_s=DEVICE_WAIT_S, quiet_s=QUIET_S)
            if not ok:
                out("[retry] device never went quiet; giving up")
                return 2
            ctx["tag"] = atag
            # The retry writes its records under a NEW tag.  Carry the
            # preflight proof over, or Q11 level 2 would report "no preflight
            # record" for a run that HAD one -- the retry re-runs the SAME job
            # in the SAME env, which is why the preflight is not repeated.
            _src = RECORDS / (tag + ".preflight.json")
            if _src.is_file():
                try:
                    _d = json.loads(_src.read_text(encoding="utf-8"))
                    _d["tag"] = atag
                    _d["carried_from"] = tag
                    (RECORDS / (atag + ".preflight.json")).write_text(
                        json.dumps(_d, indent=1, ensure_ascii=False),
                        encoding="utf-8")
                    out("[retry] preflight record carried to tag %s (same env, "
                        "same recipe)" % atag)
                except Exception as _exc:                 # noqa: BLE001
                    out("[retry] could not carry the preflight record: %r" % (_exc,))
        try:
            run_job(ctx)
        except SystemExit as exc:
            out(str(exc))
            return 2
        results, ctx2 = postflight(atag)
        n_qf = report(results, "POSTFLIGHT %s  job %s" % (atag, ctx2["job_id"]),
                      "postflight")
        device_footer(ctx2)
        out("[record] %s" % (RECORDS / (atag + ".postflight.json")))
        if n_qf == 0:
            out("OK: %d evaluated, 0 FAIL -- the run took the path the plan named "
                "and reproduced the reference." % len(results))
            return 0
        product = ctx2.get("product") or {}
        log = ctx2.get("log") or ""
        collision = (COLLISION_TOKEN in log
                     and product.get("status") != "done")
        if collision and attempt < RETRY_ON_COLLISION:
            out("[retry] this run lost the device to another process "
                "(0xc01e0009) -- not a path failure.  Retrying.")
            attempt += 1
            continue
        out("RUN FINISHED BUT THE PATH PROOF FAILED: %d check(s)." % n_qf)
        out("The job's NUMBER is not usable.  Read the FAIL lines above.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
