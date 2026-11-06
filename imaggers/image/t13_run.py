# SPDX-License-Identifier: MIT
"""t13_run.py -- T11 + T13 U-4: the full-memory (--cache-all) driver.

T13 adds exactly one mode to t11_run.py, and it is the DEFAULT.  Everything
else is verbatim.

  --cache-all   DEFAULT.  Read all 30 layers' BG (88.5 MB) + B3 (44.2 MB)
                payloads ONCE,
                permute BG into the NH=5 per-core order ONCE, and keep them in
                two flat uint8 arrays (3.98 GB).  The loop takes them by
                reference: the per-step np.load (50-70 ms/layer) and the
                per-step permutation (11-22 ms/layer) both disappear.  The
                arrays are STABLE for the whole run, so the T9 double-buffer
                hazard (an async BO write over a reused scratch buffer) cannot
                occur here by construction -- the double buffer is kept on the
                non-cached path.
                Gate: latents and PNG byte-identical to t11b, PSNR unchanged.
  --no-cache-all
                The T11 path, unchanged: np.load + permutation per layer with
                the prefetch thread.  This is the A arm of every T13 A/B.

--- t11_run.py header follows ---
t11_run.py -- T11: the T10 batch driver on the T11 FFN image with a BATCH dimension.

T10 unit U-1's measurement arm.  Added to t8_run.py (everything else verbatim):

  --batch B   run B independent images through the same 8 x 30 block-step loop.
              Image b uses seed+b (so batch=1 is the delivered run, byte for
              byte) and the images share EXACTLY three things, all of them
              values that do not depend on the image:
                * the layer's BG/B3 payload bytes (132.7 MB/layer, read once)
                * the layer's SBG bytes (one build per layer, not per image)
                * the loop's structure and its 1440 dispatches/layer-set
              Everything numerical stays per image: latents, u, rms, RoPE, the
              FFN A image, the attention drains.  The red line is that each
              image's latents are BYTE-IDENTICAL to its own batch=1 run, which
              t10_cmp.py checks (and that implies its PSNR is unchanged, since
              the VAE decode is a function of the latents alone).

This is the CONTROL arm the task asks for: with the M=96 image a batch of B
still costs B dispatches per layer, so the device side must be exactly linear
and the only gain is the host-side sharing above.  The large-M image (one
dispatch eating M=96*B) is priced separately by t10_mprobe.

--- t8_run.py's own header follows ---
t8_run.py -- T7: the T6 product loop with the attention image SPLIT IN TWO.

T7 U-1: --att-image is now the PROJECTION-ONLY image and --att-ip-image the
IP-ONLY one.  The five attention dispatches keep their identity and their fills
(d1/d2/d3 = to_q/to_k/to_v with X+W, d4 = the ip with QP/KB/VB, d5 = to_out with
X+W), so the HOST side of the loop is the T6 one; what changes is which image
each dispatch runs, and the fact that the projection's redundant second run (d4)
and the ip's four redundant runs (d1,d2,d3,d5) are gone.
Empty --att-ip-image = the T6 fused behaviour, which is the A arm.

--- t5_run.py's own header follows ---
t5_run.py -- T5: the T3d product loop with a T5 attention image.

T5 unit U1: identical to t4_run.py except that the ATTENTION image can be
pointed at another build (--att-image/--att-dir), so the wall-clock delta of a
new attention image is attributable to that image alone (same FFN image, same
payload, same session).  Everything else -- the host glue, the accounting
buckets, the checkpoint protocol, the SBG assertion -- is unchanged.

    py310 t5_run.py --steps 8 --layers 30 --warm --dimk64 --tag t5a \
        --att-image t5_pipe3 --att-dir ../attn_impl/t5_out --prefetch

--- t4_run.py's own header follows ---
T4: the T3d product loop with a T4 FFN image.

Identical to t3d_run.py (the t3d5 delivery driver) except for ONE thing: the
stage-3 FFN image is taken from imaggers/block/t4_out/<--ffn-image>.xclbin
instead of the shipped block/out/step3_M96...xclbin.  Everything else -- the
attention image (t3d_pipe2, dim_k=64), the host glue, the accounting buckets,
the checkpoint protocol -- is unchanged, so a wall-clock delta is attributable
to the FFN image alone.

    py310 t4_run.py --steps 8 --layers 30 --warm --dimk64 --tag t4a --host-ref

The architecture this runs (measured, not assumed -- see _t3d_twoimage_probe.py):
a single py3.10 process registers THREE xclbins and dispatches them alternately:

    proj   t3c_chain_q.xclbin   mode q of the attention chain: one 1.415 GMAC
                                projection per dispatch (30 heads x 60 k-groups),
                                Q bit-exact gate in t3c_chain_q_verify.json
    ip     t3c_fused_o.xclbin   qk -> HOP1 -> sm -> HOP2 -> pv, O bit-identical
                                gate in t3c_fused_o_verify.json
    ffn    step3_M96_K3840_N110240_m16_k8_m64_G4x6L10.xclbin   the stage-3 FFN

Per block-step: 4 projection dispatches (to_q, to_k, to_v, to_out) + 1 ip
dispatch + 1 FFN dispatch = 6.  The HOST between dispatches does exactly what it
did in the f32 chain -- rms(qk_norm) + RoPE + the bf16/bfp16ebs8 packaging -- and
nothing else: every matmul in attention and in the FFN is on the NPU.

The packaging is done with t3d_codec.py (pure numpy), whose every codec is gated
byte-for-byte against the library object it replaces (t3d_codec_gate.py,
ALL_PASS True).  That gate is the precondition for believing any number here.

Wall-clock accounting is kept in four buckets per block-step so the NPU share is
a measured ratio, not a claim:
    t_dev   sum of run.wait() walls (the NPU actually working)
    t_write host-side BO writes (payload staging -- the price of bfp16 weights)
    t_host  host arithmetic (norms, RoPE, pack/unpack)
    t_loop  wall clock of the block-step

Driver 376 / XRT 2.21.0: two (here three) registered xclbins in one process work
and produce byte-identical output to the same image dispatched alone; the old
one-process-one-xclbin rule is version-scoped and this file records the version.

Run (py3.10, GAIA):
    py310 t3d_run.py --gate            # one block-step vs the host f32 block
    py310 t3d_run.py --steps 8 --resume
"""
from __future__ import annotations

import argparse
import concurrent.futures as _fut
import json
import os
import pathlib
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
NPU = HERE.parent.parent
ATT = NPU / "imaggers" / "attn_impl"
OUT_A = ATT / "t3b_out"
BLOCK = NPU / "imaggers" / "block"
STAGE3 = BLOCK / "out" / "step3_M96_K3840_N110240_m16_k8_n64_G4x6L10"
T4_OUT = BLOCK / "t4_out"
# T4-F: the t4_vs FFN image takes ONE [scale|shift|gamma] per row-tile, which is
# exact only when all 96 SBG rows are equal.  Checked in block_step (module-level,
# hence the global) so the image can never be silently wrong.
ASSERT_SBG = True
NH_BG_DEFAULT = 0
# ---- T6 (this file's three deltas from t5_run.py) -------------------------
#   --fast-codec      t6_codec.install(): pack_b_nk 1.78x, unpack_blocked 1.06x,
#                     byte-identical (t6_codec_gate.py: 40/40 pack cases + all
#                     unpack cases + pack_w64; end-to-end gate = the latents)
#   --refiner-cache D the four refiner blocks' decoded f32 (2.83 GB) persisted;
#                     step 0 pays a 2.83 GB read instead of a 5.5 s decode
#   --wpack-array     hold the 30 layers' projection images (2.0 GB) in ONE
#                     preallocated uint8 array instead of 120 bytes objects
REFCACHE = None
# T7 U-2 knobs (one variable each; both gated by BYTES):
#   --fast-codec2  t7_codec.install(): pack_b_nk 1.48x offline (K), 1.10x (V)
#   --fast-glue    the q/k path stays in the drain's own (HEADS, M, HD) layout,
#                  so the 4 ascontiguousarray transposes around rms+RoPE go away
GLUE_FAST = False
WPACK = None            # None | the one big uint8 array
WPACK_SLOTS = {}        # (key, nm) -> index into WPACK
WPACK_SECONDS = [0.0, 0]
# T13 U-4: full-memory payload cache.  CACHE_BG is ALREADY permuted into the
# per-core NH order when NH_BG > 1, so ffn() must not permute it again.
CACHE_ALL = False
CACHE_BG = None
CACHE_B3 = None
BG_PREPERM = False
# T13 U-4b: 30 BG + 30 B3 BOs, written once.  Implies no host cache (the same
# 3.98 GB lives in BOs instead of RAM).
BO_RESIDENT = False
BG_BOS = None
B3_BOS = None


def _mem_gb():
    """Available physical GB (ctypes; no psutil dependency in py310)."""
    import ctypes
    from ctypes import wintypes

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
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
    return m.ullAvailPhys / 2**30
PREP = HERE / "t3real_prep"
# Third-party locations: resolved, never hard-coded (imaggers/paths.py).
_root = pathlib.Path(__file__).resolve().parent
while not (_root / 'imaggers' / 'paths.py').is_file():
    if _root.parent == _root:
        raise RuntimeError('cannot locate the checkout root')
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from imaggers import paths as _paths  # noqa: E402

XRT_DIR = _paths.xrt_dir()

sys.path.insert(0, str(HERE))
sys.path.append(str(ATT))

import t3d_codec as K                                             # noqa: E402
import t8_codec as H                                              # noqa: E402
import t9_permute as _P9                                          # noqa: E402  (T9 U-4b)
import t11_permute as _P11                                        # noqa: E402  (T11 U-1)

# T9 U-4b: the n_hold FFN image consumes BG as [pass][kk][j]; the payload on
# disk is the delivered [sweep][kk].  0 = off (the delivered path, byte-for-byte
# what t8b runs); 2 = the n_hold=2 image (t9_nh2b).  The permutation is host
# payload plumbing -- it changes no arithmetic, and it is charged to the pack
# bucket so the ledger stays honest.
NH_BG = 0
# T11 U-1: with --nh-bg N, --nh-bg-core writes the PER-CORE contiguous order
# [w][pass][kk][j] that the per-core-fill NH image reads (each gate/up core owns
# its own shim fill, so its bytes must be one contiguous slice of the BO).
NH_BG_CORE = False
_BG_SCRATCH = None
_BG_FLIP = 0

D, HEADS, HD, FF, SEQ, CAP = 3840, 30, 128, 10240, 96, 32
M = SEQ
KD, ND = HD, M
HEADS_ = HEADS
ACT_BYTES = SEQ * D * 2                      # 737,280
SBG_BYTES = SEQ * 3 * D * 2
BG_BYTES = 2 * FF * D * 9 // 8               # 88,473,600
B3_BYTES = D * FF * 9 // 8                   # 44,236,800
KB_BYTES = KD * ND // 8 * 9                  # 13,824
W_BYTES = HEADS * (D * HD // 8 * 9)          # one projection's bfp image
assert W_BYTES == 16_588_800, W_BYTES
COMPLETED = 4
TIMEOUT_MS = int(os.environ.get("T3_TIMEOUT_MS", "300000"))
RETRY_SLEEPS = (10, 20, 30, 45, 60)
PROJ_ARGS = ("x", "w", "kb", "vb")
FFN_ORDER = ("A", "SBG", "BG", "B3", "X", "Y")
FFN_GROUPS = {"A": 3, "SBG": 4, "BG": 5, "B3": 6, "X": 7, "Y": 8}


def act(x):
    return np.asarray(x, np.float32)


def bf16f32(x):
    """bf16(x) back in float32 -- the grid the device's operands live on."""
    return (K.bf16_rne(x).astype(np.uint32) << 16).view(np.float32)


def rope_rms_hmd(x, w, cos, sin, T3):
    """rms + RoPE in the (HEADS, M, HD) layout the drain already produces.

    t3_real.rms reduces over the LAST axis and t3_real.apply_rope rotates the
    0::2 / 1::2 pairs of the last axis with cos/sin indexed by the SEQUENCE
    POSITION (axis -2 of (M, HEADS, HD)).  In the drained layout the sequence is
    axis 1, so the very same element-wise expressions apply with cos[None] --
    same operands, same order, same rounding.  The only thing that changes is
    that no (M, HEADS, HD) copy is ever materialised: 4 x 1.47 MB per layer.
    """
    y = T3.rms(x, w)                       # (HEADS, M, HD), reduction over HD
    c, s = cos[None, :, :], sin[None, :, :]
    xe, xo = y[..., 0::2], y[..., 1::2]
    out = np.empty_like(y)
    out[..., 0::2] = xe * c - xo * s
    out[..., 1::2] = xe * s + xo * c
    return out


# ---------------------------------------------------------------------------
# weight packing: to_q/to_k/to_v/to_out -> the proj image's W payload
# ---------------------------------------------------------------------------
FAST_HOST = False
SCRATCH = None


PROF = None


def pmark(name, t0):
    if PROF is not None:
        PROF[name] = PROF.get(name, 0.0) + 1e3 * (time.perf_counter() - t0)


def rms_into(x, w, out):
    """T3.rms into a caller-supplied buffer: same ops, same order, no new page.

    t3_real.rms is  y = (x / sqrt(mean(x*x) + eps)).astype(f32); * w, all in
    float32.  Writing through np.divide/np.multiply with out= computes exactly
    the same float32 expressions (the .astype is a no-op on float32 input); what
    disappears is two 1.47 MB allocations per call.
    """
    import t3_real as T3
    np.multiply(x, x, out=out)
    v = out.mean(-1, keepdims=True, dtype=np.float32)
    np.add(v, np.float32(T3.EPS), out=v)
    np.sqrt(v, out=v)
    np.divide(x, v, out=out)
    np.multiply(out, np.asarray(w, np.float32), out=out)
    return out


def rope_rms_jit(x, w, cos, sin, out, tmp):
    """rms -> rope, both into reusable buffers (no copy between them)."""
    rms_into(x, w, tmp)
    H.rope_f32(tmp, cos, sin, out)
    return out


def pack_w(Wh: np.ndarray) -> np.ndarray:
    """W (out=D, in=D) -> the 16,588,800 B image the proj core reads.

    Block order = t3c_chain_run.pack_w (the packer behind the gated mode-q
    image), vectorized: the caller's 14,400 python-level bfp.encode calls of one
    (16,8,8) tile each become one encode of a (8*16, 8, 8) stack per (head,
    k-group).  Byte-equality with the slow packer is proven by t3d_wgate.py; the
    slow one must not be trusted either, so t3d_wgate also re-derives the whole
    projection against the k-slice reference.
    """
    K_ = Wh.shape[1]
    NG = K_ // 64
    out = np.empty((HEADS, NG, 8 * 16 * 72), np.uint8)
    for h in range(HEADS):
        Wn = np.ascontiguousarray(Wh[h * HD:(h + 1) * HD, :], np.float32)
        for g in range(NG):
            # (nb=16, n8=8, s=8, k8=8) -> (s, nb, n8, k8): the packer's order is
            # [s][nb] with each tile an (8 n, 8 k) block of 8 groups of 9 B.
            sl = Wn[:, g * 64:(g + 1) * 64].reshape(16, 8, 8, 8)
            blk = np.ascontiguousarray(sl.transpose(2, 0, 1, 3)).reshape(128, 8, 8)
            out[h, g] = K.bfp_encode(blk).reshape(-1)
    return out


def pack_w64(Wh: np.ndarray) -> np.ndarray:
    """W (out=D, in=D) -> the payload of the dim_k=64 image: per (head, k-group)
    the (n=128, k=64) B image, in pack_b_nk's [nb][kb] order.  Same 16,588,800 B
    total as the k=8 packer, different internal grouping."""
    out = np.empty((HEADS, D // 64, 128 * 64 // 8 * 9), np.uint8)
    for h in range(HEADS):
        for g in range(D // 64):
            blk = np.ascontiguousarray(Wh[h * HD:(h + 1) * HD, g * 64:(g + 1) * 64])
            assert blk.shape == (HD, 64), blk.shape
            out[h, g] = K.pack_b_nk(blk[None])[0]
    return out


def x_image64(x_mk: np.ndarray, perm64: np.ndarray) -> bytes:
    """(96, K) float32 -> the dim_k=64 X object: 60 groups of the blocked (96,64)."""
    m, Kd = x_mk.shape
    parts = [K.pack_blocked(np.ascontiguousarray(x_mk[:, g * 64:(g + 1) * 64])
                            .reshape(1, -1), perm64) for g in range(Kd // 64)]
    return b"".join(parts)


def make_sbg(P, mod):
    """The FFN image's [scale|shift|gamma] stream, bf16, one row per token.

    Verbatim the block that used to live in block_step: it depends only on the
    (step, layer) modulation and on ffn_norm1 (a weight), NEVER on the image --
    which is why T10 builds it once per layer and hands the same bytes to every
    image of the batch.  batch=1 gets the identical bytes from the identical
    expressions, so nothing about the delivered run changes.
    """
    SBG = np.zeros((SEQ, 3 * D), np.float32)
    if mod is not None:
        SBG[:, :D] = (mod[2] - np.float32(1.0))
    SBG[:, D:2 * D] = 0.0
    SBG[:, 2 * D:] = P["ffn_norm1"]
    if FAST_HOST:
        return H.bf16_bytes_f32(SBG, SCRATCH, "sbg")
    return np.frombuffer(K.bf16_bytes(SBG), np.uint8)


# T10 U-1b: with a batch, layer L's BG/B3 bytes are already staged in the BO
# after image 0's dispatch; rewriting them for images 1..B-1 moves the same
# bytes to the same place.  --share-weights skips those writes (the red line is
# the latents: byte-identical to the batch=1 run).
SHARE_W = False


# ---------------------------------------------------------------------------
class Dev:
    """The three images, their BOs, and the retry/measure protocol."""

    def __init__(self, tag="t3d", dimk64=False, ffn_image="t4_sbg",
                 att_image="", att_dir=None, att_ip_image=""):
        for d in (XRT_DIR, pathlib.Path(r"C:\Windows\System32\AMD")):
            if d.exists():
                sys.path.insert(0, str(d))
                try:
                    os.add_dll_directory(str(d))
                except Exception:  # noqa: BLE001
                    pass
        global pyxrt
        import pyxrt
        self.pyxrt = pyxrt
        self.dev = pyxrt.device(0)
        self.tag = tag
        self.imgs = {}
        att_x = (att_image, att_image) if att_image else (
            ("t3d_pipe2", "t3d_pipe2") if dimk64 else ("t3d_pipe", "t3d_pipe"))
        att_base = pathlib.Path(att_dir) if att_dir else OUT_A
        if att_dir and not att_base.is_absolute():
            att_base = (HERE / att_base).resolve()
        # T7: the split pair.  "att" is the projection-only image when an ip
        # image is named, and the T6 fused image otherwise (the A arm).
        self.has_ip = bool(att_ip_image)
        regs = [("att", att_base / (att_x[0] + ".xclbin"),
                 att_base / (att_x[1] + ".bin"))]
        if self.has_ip:
            regs.append(("attip", att_base / (att_ip_image + ".xclbin"),
                         att_base / (att_ip_image + ".bin")))
        regs.append(("ffn", T4_OUT / (ffn_image + ".xclbin"),
                     T4_OUT / (ffn_image + ".bin")))
        for name, xcl, inst in regs:
            t0 = time.perf_counter()
            xc = pyxrt.xclbin(str(xcl))
            uuid = self.dev.register_xclbin(xc)
            kk = pyxrt.kernel(self.dev, uuid, "MLIR_AIE")
            ib = inst.read_bytes()
            bo_i = pyxrt.bo(self.dev, len(ib), pyxrt.bo.flags.cacheable,
                            kk.group_id(1))
            bo_i.write(ib, 0)
            bo_i.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self.imgs[name] = {"k": kk, "inst": ib, "bo_i": bo_i,
                               "reg_s": time.perf_counter() - t0}
        fl = pyxrt.bo.flags.host_only
        g = lambda im, i: self.imgs[im]["k"].group_id(i)
        self.bo = {}
        # the attention image's 7 data args, in declaration order
        #   x(3) w(4) qp(5) kb(6) vb(7) -> outq(8) outo(9)
        if self.has_ip:
            # T7 split pair: the projection image declares (X, W -> OUTQ) and the
            # ip image (QP, KB, VB -> OUTO); each argument needs a BO carrying
            # THAT kernel's group id.
            for i, n in enumerate(("x", "w")):
                sz = {"x": ACT_BYTES, "w": W_BYTES}[n]
                self.bo["a_" + n] = pyxrt.bo(self.dev, sz, fl, g("att", 3 + i))
            self.bo["a_outq"] = pyxrt.bo(self.dev, ACT_BYTES, fl, g("att", 5))
            for i, n in enumerate(("qp", "kb", "vb")):
                sz = {"qp": ACT_BYTES, "kb": KB_BYTES * HEADS,
                      "vb": KB_BYTES * HEADS}[n]
                self.bo["a_" + n] = pyxrt.bo(self.dev, sz, fl, g("attip", 3 + i))
            self.bo["a_outo"] = pyxrt.bo(self.dev, ACT_BYTES, fl, g("attip", 6))
        else:
            for i, n in enumerate(("x", "w", "qp", "kb", "vb")):
                sz = {"x": ACT_BYTES, "w": W_BYTES, "qp": ACT_BYTES,
                      "kb": KB_BYTES * HEADS, "vb": KB_BYTES * HEADS}[n]
                self.bo["a_" + n] = pyxrt.bo(self.dev, sz, fl, g("att", 3 + i))
            self.bo["a_outq"] = pyxrt.bo(self.dev, ACT_BYTES, fl, g("att", 8))
            self.bo["a_outo"] = pyxrt.bo(self.dev, ACT_BYTES, fl, g("att", 9))
        for n in ("SBG", "BG", "B3"):
            sz = {"SBG": SBG_BYTES, "BG": BG_BYTES, "B3": B3_BYTES}[n]
            self.bo["f_" + n] = pyxrt.bo(self.dev, sz, fl, g("ffn", FFN_GROUPS[n]))
        for n in ("A", "X", "Y"):
            self.bo["f_" + n] = pyxrt.bo(self.dev, ACT_BYTES, fl,
                                        g("ffn", FFN_GROUPS[n]))
        self.stats = {"dev_ms": {}, "write_ms": {}, "calls": 0,
                      "attempts": [], "retries": 0}

    # -- one dispatch with the contention protocol -------------------------
    def _run(self, name, args, out_keys, tag):
        im = self.imgs[name]
        t0 = time.perf_counter()
        try:
            run = im["k"](3, im["bo_i"], len(im["inst"]) // 4, *args)
            state = int(run.wait(TIMEOUT_MS))
        except Exception as exc:  # noqa: BLE001 (0xc00002b6 and friends)
            state = -1
            tag = "%s (XRT %s)" % (tag, type(exc).__name__)
        wall = 1e3 * (time.perf_counter() - t0)
        t0 = time.perf_counter()
        for b in out_keys:
            b.sync(self.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        pmark("host.osync", t0)
        if PROF is not None:
            PROF["host.wait"] = PROF.get("host.wait", 0.0) + wall
        return state, wall, tag

    def dispatch(self, name, fills, outs, tag=""):
        """fills: [(bo, bytes)] written + synced; outs: [(bo, nbytes)] read."""
        pyxrt = self.pyxrt
        t0 = time.perf_counter()
        for bo, payload in fills:
            bo.write(payload, 0)
            bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        tw = 1e3 * (time.perf_counter() - t0)
        pmark("host.fill", t0)
        if name == "att":
            if self.has_ip:
                args = [self.bo["a_" + n] for n in ("x", "w", "outq")]
                out_keys = [self.bo["a_outq"]]
            else:
                args = [self.bo["a_" + n] for n in ("x", "w", "qp", "kb", "vb")]
                out_keys = [self.bo["a_outq"], self.bo["a_outo"]]
        elif name == "attip":
            args = [self.bo["a_" + n] for n in ("qp", "kb", "vb", "outo")]
            out_keys = [self.bo["a_outo"]]
        else:
            args = [self.bo["f_" + n] for n in FFN_ORDER]
            out_keys = [self.bo["f_Y"]]
        # EVERY declared data argument must be bound: the ERT maps the BOs to
        # the runtime sequence's arguments and a short list leaves the tail
        # unbound (measured: att hung for the whole 300 s timeout when the two
        # outputs were omitted, while the outputs' group ids were still
        # allocated -- so "it completed anyway" is not evidence).
        args = args + [bo for bo, _ in outs] if outs else args + out_keys
        if outs:
            pass
        elif name == "att":
            outs = ([(self.bo["a_outq"], ACT_BYTES),
                     (self.bo["a_outo"], ACT_BYTES)] if not self.has_ip
                    else [(self.bo["a_outq"], ACT_BYTES)])
        elif name == "attip":
            outs = [(self.bo["a_outo"], ACT_BYTES)]
        else:
            outs = [(self.bo["f_Y"], ACT_BYTES)]
        for i in range(len(RETRY_SLEEPS) + 1):
            state, wall, tag2 = self._run(name, args, out_keys, tag)
            self.stats["attempts"].append({"img": name, "state": state,
                                           "wall_ms": wall})
            if state == COMPLETED:
                # T4-G: per-DISPATCH device time (att/q, att/k, att/v, att/ip,
                # att/o, ffn), so the attention's 28 s can be attributed to its
                # five dispatches instead of being one number.
                dbt = self.stats.setdefault("dev_by_tag", {})
                dbt[tag] = dbt.get(tag, 0.0) + wall
                self.stats["dev_ms"][name] = self.stats["dev_ms"].get(
                    name, 0.0) + wall
                self.stats["write_ms"][name] = self.stats["write_ms"].get(
                    name, 0.0) + tw
                self.stats["calls"] += 1
                t0 = time.perf_counter()
                raws = [bytes(bo.read(n, 0)) for bo, n in outs]
                pmark("host.read", t0)
                return raws, wall, tw
            self.stats["retries"] += 1
            if i < len(RETRY_SLEEPS):
                print("[t3d]   %s RETRY %d/%d after %.1f ms (%s)"
                      % (tag2, i + 1, len(RETRY_SLEEPS), wall, state),
                      flush=True)
                time.sleep(RETRY_SLEEPS[i])
        raise SystemExit("[t3d] REFUSAL: %s never completed (%s); checkpoint "
                         "holds every dispatch already measured" % (tag, state))


# ---------------------------------------------------------------------------
# T6: the four refiner blocks' decoded f32 weights, persisted
# ---------------------------------------------------------------------------
_RC_HIT = [False]


def _refblock_cached(RW, key):
    """RW.block(key), but the decoded f32 tensors survive the process.

    The decode is 5.5 s per process (4 x ~1.4 s, 2.83 GB) and the values are
    identical on every run, so they are saved as .npy (exact for f32) and read
    back.  The gate is BYTES: t6_refcache_gate.py compares every tensor of all
    four blocks against a fresh decode, and the delivered latents are compared
    byte-for-byte with t3d5's.
    """
    _RC_HIT[0] = False
    if REFCACHE is None:
        return RW.block(key)
    d = pathlib.Path(REFCACHE) / key
    man = d / "manifest.json"
    if man.exists():
        try:
            names = json.loads(man.read_text())
            out = {}
            for nm in names:
                if nm.startswith("!"):
                    out[nm[1:]] = None
                else:
                    out[nm] = np.load(d / (nm + ".npy"))
            _RC_HIT[0] = True
            return out
        except Exception as exc:                              # noqa: BLE001
            print("[t6] refiner cache unreadable (%s); decoding" % exc,
                  flush=True)
    out = RW.block(key)
    try:
        d.mkdir(parents=True, exist_ok=True)
        names = []
        for nm, v in out.items():
            if v is None:
                names.append("!" + nm)
            else:
                np.save(d / (nm + ".npy"), v)
                names.append(nm)
        (d / "manifest.json").write_text(json.dumps(names))
    except Exception as exc:                                  # noqa: BLE001
        print("[t6] refiner cache write failed (%s); running uncached" % exc,
              flush=True)
    return out


# ---------------------------------------------------------------------------
# the block on the NPU
# ---------------------------------------------------------------------------
class Loop:
    def __init__(self, dev: Dev, tag="t3d", dimk64=False):
        import t3_real as T3
        self.T3 = T3
        self.dev = dev
        self.RW = T3.Real()
        self.dimk64 = dimk64
        self.perm = K.load_perm(HERE / "t3d_perm_96x128.json")
        self.inv = K.load_perm_inv(HERE / "t3d_perm_96x128.json")
        self.perm64 = K.load_perm(HERE / "t3d_perm_96x64.json") if dimk64 else None
        self.fold = np.float32(1.0 / np.sqrt(HD))
        self.wcache = {}
        self.zero_x = bytes(ACT_BYTES)
        self.written = set()
        self.tag = tag

    # -- weight images (packed once per layer, kept in RAM) ---------------
    def ximg(self, x2d):
        """The projection's X payload, in whichever k-grouping this run uses."""
        if self.dimk64:
            return x_image64(np.asarray(x2d, np.float32), self.perm64)
        return K.x_image(np.asarray(x2d, np.float32))

    def w_images(self, key):
        if key in self.wcache:
            return self.wcache[key]
        t0 = time.perf_counter()
        out = {}
        for nm, tail in (("q", "attention.to_q.weight"),
                         ("k", "attention.to_k.weight"),
                         ("v", "attention.to_v.weight"),
                         ("o", "attention.to_out.0.weight")):
            W = self.RW.get(key + "." + tail)
            img = np.ascontiguousarray(
                pack_w64(W) if self.dimk64 else pack_w(W))
            del W
            if WPACK is None:
                out[nm] = img.tobytes()
            else:
                # T6: one 2.0 GB array instead of 120 x 16.6 MB bytes objects
                slot = (len(WPACK_SLOTS))
                WPACK_SLOTS[(key, nm)] = slot
                buf = WPACK[slot * W_BYTES:(slot + 1) * W_BYTES]
                buf[:] = img.reshape(-1)
                out[nm] = buf
        out["_s"] = time.perf_counter() - t0
        WPACK_SECONDS[0] += out["_s"]
        WPACK_SECONDS[1] += 1
        self.wcache[key] = out
        return out

    def params(self, key):
        if key not in self._pcache:
            g = lambda t: self.RW.get(key + "." + t)
            self._pcache[key] = {
                "norm_q": g("attention.norm_q.weight"),
                "norm_k": g("attention.norm_k.weight"),
                "attention_norm1": g("attention_norm1.weight"),
                "attention_norm2": g("attention_norm2.weight"),
                "ffn_norm1": g("ffn_norm1.weight"),
                "ffn_norm2": g("ffn_norm2.weight"),
                "adaln_w": (g("adaLN_modulation.0.weight")
                            if key + ".adaLN_modulation.0.weight" in self.RW.rec
                            else None),
                "adaln_b": (g("adaLN_modulation.0.bias")
                            if key + ".adaLN_modulation.0.bias" in self.RW.rec
                            else None),
            }
        return self._pcache[key]

    _pcache = {}
    _rcache = {}

    def refblock(self, key):
        """The four refiner blocks' weights, decoded ONCE per process.

        They are identical on every one of the 8 steps, and RW.block() decodes
        ~708 MB of f32 (w1|w3|w2 157 MB each + to_q/k/v/out 59 MB each) every
        time it is called -- which is 16.2 s of the 30 s a step used to spend
        outside the NPU.  Caching 4 blocks costs ~2.8 GB of RAM; the gate is
        that the whole run stays BIT-IDENTICAL to the uncached one.
        """
        if key not in self._rcache:
            t0 = time.perf_counter()
            self._rcache[key] = _refblock_cached(self.RW, key)
            print("[t3d] refiner weights %-14s %-18s %5.1f s (%.2f GB)"
                  % ("loaded" if _RC_HIT[0] else "decoded", key,
                     time.perf_counter() - t0,
                     sum(v.nbytes for v in self._rcache[key].values()
                         if v is not None) / 2**30), flush=True)
        return self._rcache[key]

    # -- the attention branch: 5 dispatches of ONE image ------------------
    def att_dispatch(self, fills, tag, tim, img="att"):
        raw, wall, tw = self.dev.dispatch(img, fills, [], tag=tag)
        tim["dev"] += wall
        tim["write"] += tw
        if img == "attip":
            return None, raw[0]
        return raw[0], (raw[1] if len(raw) > 1 else None)

    def attention(self, x1, key, cos, sin, tim):
        if FAST_HOST:
            return self.attention_fast(x1, key, cos, sin, tim)
        return self.attention_ref(x1, key, cos, sin, tim)

    def attention_fast(self, x1, key, cos, sin, tim):
        """The same five dispatches, the same fill/ledger order; only the host
        codecs and their buffers changed."""
        T3 = self.T3
        bo = self.dev.bo
        sc = SCRATCH
        P = self.params(key)
        W = self.w_images(key)
        t0 = time.perf_counter()
        Xb = H.x_image64_bytes(x1, self.perm64, sc, "ximg")
        pmark("att.ximg", t0)
        tim["pack"] += 1e3 * (time.perf_counter() - t0)
        raw = {}
        for i, nm in enumerate(("q", "k", "v")):
            fills = []
            if i == 0:
                fills.append((bo["a_x"], Xb))
            fills.append((bo["a_w"], W[nm]))
            raw[nm] = self.att_dispatch(fills, "att/" + nm, tim)
        t0 = time.perf_counter()
        Qm = {nm: H.unpack_bytes(raw[nm][0], self.inv, (HEADS, M * HD), sc,
                                 "u" + nm).reshape(HEADS, M, HD)
              for nm in ("q", "k", "v")}
        pmark("att.unpack3", t0)
        t0 = time.perf_counter()
        wq = bf16f32(P["norm_q"] * self.fold)
        kh = rope_rms_jit(Qm["k"], P["norm_k"], cos, sin,
                          sc.get("kh", (HEADS, M, HD), np.float32),
                          sc.get("rms0", (HEADS, M, HD), np.float32))
        qh = rope_rms_jit(Qm["q"], wq, cos, sin,
                          sc.get("qh", (HEADS, M, HD), np.float32),
                          sc.get("rms1", (HEADS, M, HD), np.float32))
        pmark("att.rms_rope", t0)
        t0 = time.perf_counter()
        a_bytes = H.pack_blocked_bytes(qh.reshape(HEADS, M * HD), self.perm,
                                       sc, "qp")
        pmark("att.packq", t0)
        t0 = time.perf_counter()
        kb = H.pack_b_nk_f32(kh, sc, "kb")
        pmark("att.packkb", t0)
        t0 = time.perf_counter()
        vb = H.pack_b_nk_t_f32(Qm["v"], sc, "vb")
        pmark("att.packvb", t0)
        tim["pack"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        _, raw_o = self.att_dispatch(
            [(bo["a_qp"], a_bytes), (bo["a_kb"], kb), (bo["a_vb"], vb)],
            "att/ip", tim, img=("attip" if self.dev.has_ip else "att"))
        o_flat = H.unpack_bytes_t(raw_o, self.inv, (HEADS, M * HD), HD, sc, "ot")
        pmark("att.unpack_o", t0)
        t0 = time.perf_counter()
        X2 = H.x_image64_bytes(o_flat, self.perm64, sc, "ximg2")
        tim["pack"] += time.perf_counter() - t0
        raw_q, _ = self.att_dispatch([(bo["a_x"], X2), (bo["a_w"], W["o"])],
                                     "att/o", tim)
        t0 = time.perf_counter()
        a = H.unpack_bytes_t(raw_q, self.inv, (HEADS, M * HD), HD, sc, "at")
        pmark("att.unpack_a", t0)
        tim["pack"] += time.perf_counter() - t0
        return a

    def attention_ref(self, x1, key, cos, sin, tim):
        """4 projections + the ip, all on the NPU, in one image.

        d1..d3  W = to_q / to_k / to_v, X filled once  -> Q, K, V drains
        host    rms(qk_norm) + RoPE + bf16/bfp16ebs8 packaging (t3d_codec)
        d4      QP/KB/VB filled -> O drain (its proj drain is discarded)
        d5      X = the bf16 image of o, W = to_out -> the attention output
        """
        T3 = self.T3
        bo = self.dev.bo
        P = self.params(key)
        W = self.w_images(key)
        t0 = time.perf_counter()
        Xb = self.ximg(x1)
        tim["pack"] += 1e3 * (time.perf_counter() - t0)
        raw = {}
        for i, nm in enumerate(("q", "k", "v")):
            fills = []
            if i == 0:
                fills.append((bo["a_x"], Xb))
            fills.append((bo["a_w"], W[nm]))
            t0 = time.perf_counter()
            raw[nm] = self.att_dispatch(fills, "att/" + nm, tim)
            pass
        t0 = time.perf_counter()
        # T3.rms / T3.apply_rope are (seq, heads, hd) -- t3_real.attention's own
        # layout -- so the device drains are transposed into it and back.
        Qm = {nm: K.unpack_blocked(raw[nm][0], self.inv,
                                   (HEADS, M * HD)).reshape(HEADS, M, HD)
              for nm in ("q", "k", "v")}
        wq = bf16f32(P["norm_q"] * self.fold)
        if GLUE_FAST:
            kh = rope_rms_hmd(Qm["k"], P["norm_k"], cos, sin, T3)
            qh = rope_rms_hmd(Qm["q"], wq, cos, sin, T3)
        else:
            qm = np.ascontiguousarray(Qm["q"].transpose(1, 0, 2))
            km = np.ascontiguousarray(Qm["k"].transpose(1, 0, 2))
            q = np.asarray(T3.apply_rope(T3.rms(qm, wq), cos, sin), np.float32)
            k = np.asarray(T3.apply_rope(T3.rms(km, P["norm_k"]), cos, sin),
                           np.float32)
            qh = np.ascontiguousarray(q.transpose(1, 0, 2))
            kh = np.ascontiguousarray(k.transpose(1, 0, 2))
        a_bytes = K.pack_blocked(qh.reshape(HEADS, M * HD), self.perm)
        kb = K.pack_b_nk(kh)
        vb = K.pack_b_nk(np.ascontiguousarray(
            Qm["v"].transpose(0, 2, 1), np.float32))
        tim["pack"] += time.perf_counter() - t0
        # d4: the ip.  w still holds to_v and x the attention input: no rewrite.
        t0 = time.perf_counter()
        # att_dispatch returns (OUTQ, OUTO).  The ip's answer is OUTO, and OUTQ
        # of this dispatch is the *projection* (X @ to_v) -- reading raw[0] here
        # is the bug that made the whole image 11.65 dB: every block-step fed
        # x1 @ to_v into to_out as if it were o.  Measured cost of the slip:
        # attention output 5.5 % rmse, image 11.65 dB, and a false attribution to
        # the projection's C cadence (t3d_acc.json's first version).
        # T7 split: the ip-only image takes QP/KB/VB from ITS OWN BOs; the fused
        # image (A arm) takes them from the projection image's BOs.
        _, raw_o = self.att_dispatch(
            [(bo["a_qp"], a_bytes), (bo["a_kb"], kb.tobytes()),
             (bo["a_vb"], vb.tobytes())], "att/ip", tim,
            img=("attip" if self.dev.has_ip else "att"))
        o = K.unpack_blocked(raw_o, self.inv,
                             (HEADS, M * HD)).reshape(HEADS, M, HD)
        o_flat = np.ascontiguousarray(o.transpose(1, 0, 2)).reshape(M, D)
        X2 = self.ximg(o_flat)
        tim["pack"] += time.perf_counter() - t0
        # d5: to_out.  qp/kb/vb keep d4's bytes, so the ip re-runs on them and
        # its O is discarded (2.3 ms of the 25 ms this dispatch costs anyway).
        raw_q, _ = self.att_dispatch([(bo["a_x"], X2), (bo["a_w"], W["o"])],
                                     "att/o", tim)
        t0 = time.perf_counter()
        A = K.unpack_blocked(raw_q, self.inv,
                             (HEADS, M * HD)).reshape(HEADS, M, HD)
        a = np.ascontiguousarray(A.transpose(1, 0, 2)).reshape(M, D)
        tim["pack"] += time.perf_counter() - t0
        return a

    # -- the FFN, unchanged from the M2 image ----------------------------
    def ffn(self, x_mid, key, mod, tim, bg, b3, sbg_bytes):
        dev = self.dev
        t0 = time.perf_counter()
        if NH_BG > 1 and not BG_PREPERM and not BO_RESIDENT:
            # T9 U-4b route (a): one pass over 88.5 MB, charged to `pack`.
            # T13: skipped when --cache-all pre-permuted it at startup, or when
            # --bo-resident already wrote the permuted payload into the BO.
            global _BG_SCRATCH, _BG_FLIP
            n_bg = len(bg)
            if _BG_SCRATCH is None or _BG_SCRATCH.size != n_bg:
                # DOUBLE buffered on purpose: the BO write is asynchronous, so
                # the layer-i+1 permutation must not overwrite the bytes the
                # layer-i dispatch may still be streaming (that cost two runs).
                _BG_SCRATCH = np.empty((2, n_bg), np.uint8)
                _BG_FLIP = 0
            _BG_FLIP ^= 1
            dst = _BG_SCRATCH[_BG_FLIP]
            src = (np.frombuffer(bg, np.uint8)
                   if isinstance(bg, (bytes, bytearray))
                   else np.ascontiguousarray(bg, np.uint8))
            (_P11.permute_bg if NH_BG_CORE else _P9.permute_bg)(
                src, dst, NH_BG)
            bg = dst
        a_img = (H.bf16_bytes_f32(np.asarray(x_mid, np.float32), SCRATCH, "ffnA")
                 if FAST_HOST else
                 np.frombuffer(K.bf16_bytes(np.asarray(x_mid, np.float32)),
                               np.uint8))
        fills = [(dev.bo["f_A"], a_img), (dev.bo["f_SBG"], sbg_bytes)]
        if not BO_RESIDENT:
            # T13 U-4b: the layer's BG/B3 BOs already hold the payload.
            fills += [(dev.bo["f_BG"], bg), (dev.bo["f_B3"], b3)]
        if "fx" not in self.written:
            fills.append((dev.bo["f_X"], self.zero_x))
            self.written.add("fx")
        tim["pack"] += time.perf_counter() - t0
        raw, wall, tw = dev.dispatch("ffn", fills,
                                     [(dev.bo["f_Y"], ACT_BYTES)], tag="ffn")
        tim["dev"] += wall
        tim["write"] += tw
        t0 = time.perf_counter()
        y = (H.bf16_to_f32_bits(np.frombuffer(raw[0], np.uint16), (SEQ, D),
                                SCRATCH, "ffnY")
             if FAST_HOST else K.bf16_to_f32(raw[0]).reshape(SEQ, D))
        tim["pack"] += time.perf_counter() - t0
        return y


# ---------------------------------------------------------------------------
# one block-step: attention on the NPU + FFN on the NPU
# ---------------------------------------------------------------------------
def block_step(L: Loop, u, key, mod, cos, sin, tim, bg, b3, host_ref=False,
               sbg_bytes=None):
    T3 = L.T3
    P = L.params(key)
    t0 = time.perf_counter()
    n1 = T3.rms(u, P["attention_norm1"])
    x1 = n1 if mod is None else (n1 * mod[0]).astype(np.float32)
    tim["host"] += 1e3 * (time.perf_counter() - t0)
    pmark("post.norm1", t0)
    a = L.attention(x1, key, cos, sin, tim)
    t0 = time.perf_counter()
    a2 = T3.rms(a, P["attention_norm2"])
    xm = (u + a2) if mod is None else (u + mod[1] * a2)
    xm = xm.astype(np.float32)
    pmark("post.norm2", t0)
    # ---- FFN: SBG exactly as the M2 driver built it --------------------
    # T10: the batch driver builds it ONCE per layer and passes it in (it does
    # not depend on the image); batch=1 calls make_sbg(P, mod), the same
    # expressions on the same buffers as the delivered run.
    if sbg_bytes is None:
        sbg_bytes = make_sbg(P, mod)
    # T4-F: the FFN image's norm core takes the [scale|shift|gamma] element once
    # per row-tile, which is exact ONLY if all 96 rows of SBG are equal -- true
    # by construction here (scale/shift are per-block constants, gamma a weight),
    # and ASSERTED rather than assumed: 96 x 23,040 B compare, ~1 ms/layer.
    if ASSERT_SBG:
        _rows = (np.ascontiguousarray(sbg_bytes).view(np.uint16)
                 if not FAST_HOST else sbg_bytes).reshape(SEQ, -1)
        if not bool((_rows == _rows[0]).all()):
            raise SystemExit(
                "[t4] REFUSAL: SBG rows differ -- the t4_vs FFN image assumes "
                "one [scale|shift|gamma] per block; rebuild/choose the C-only "
                "image (t4_vcopy2) for this payload")
    pmark("post.sbg", t0)
    t0 = time.perf_counter()
    y = L.ffn(xm, key, mod, tim, bg, b3, sbg_bytes)
    p = T3.rms(y, P["ffn_norm2"])
    out = (xm + (p if mod is None else mod[3] * p)).astype(np.float32)
    pmark("post.ffnpost", t0)
    tim["host"] += 1e3 * (time.perf_counter() - t0)
    rec = {"layer": key, "a_absmax": float(np.abs(a).max()),
           "xm_absmax": float(np.abs(xm).max()),
           "y_absmax": float(np.abs(y).max())}
    if host_ref:
        t0 = time.perf_counter()
        Wb = L.RW.block(key)
        a_ref = T3.attention(bf16f32(x1), Wb, cos, sin)
        y_ref = T3.ffn_raw(T3.ffn_input(xm, Wb, mod), Wb)
        e = np.abs(a - a_ref)
        rec["attn_dev_vs_host"] = {
            "max_abs": float(e.max()),
            "rel_to_absmax": float(e.max() / max(np.abs(a_ref).max(), 1e-30)),
            "rms": float(np.sqrt((e ** 2).mean())),
            "ref_absmax": float(np.abs(a_ref).max())}
        e2 = np.abs(y - y_ref)
        rec["ffn_dev_vs_host"] = {
            "max_abs": float(e2.max()),
            "rel_to_absmax": float(e2.max() / max(np.abs(y_ref).max(), 1e-30)),
            "rms": float(np.sqrt((e2 ** 2).mean()))}
        del Wb, a_ref, y_ref
        rec["host_ref_s"] = time.perf_counter() - t0
    return out, rec


def run(args) -> int:
    import t3_real as T3
    dev = Dev(args.tag, dimk64=args.dimk64, ffn_image=args.ffn_image,
              att_image=args.att_image, att_dir=args.att_dir,
              att_ip_image=args.att_ip_image)
    print("[t3d] registered %d images: %s" % (len(dev.imgs), "  ".join(
        "%s %.2fs" % (n, v["reg_s"]) for n, v in dev.imgs.items())), flush=True)
    print("[t3d] BOs: %s  ffn(A,SBG,BG,B3,X)->Y"
          % ("att(x,w)->Q  attip(qp,kb,vb)->O" if dev.has_ip
             else "att(x,w,qp,kb,vb)->Q,O"), flush=True)
    for n in (("att", "attip", "ffn") if dev.has_ip else ("att", "ffn")):
        k = dev.imgs[n]["k"]
        print("[t3d]   %-4s group ids %s" % (n, [k.group_id(i)
                                                 for i in range(4)]), flush=True)
    L = Loop(dev, args.tag, dimk64=args.dimk64)
    H = T3.host_tensors(args.tag)
    RW = L.RW
    timesteps, sigmas = T3.load_schedule(args)
    # T10: one (1,16,16,16) latent PER IMAGE (seed+b).  Kept as a list of the
    # delivered shape so every downstream op sees exactly the tensor it saw in
    # the batch=1 run -- the batch never concatenates the images' activations.
    B = int(args.batch)
    lat = [np.random.default_rng(args.seed + b).standard_normal(
        (1, 16, 16, 16)).astype(np.float32) for b in range(B)]
    lat0 = [x.copy() for x in lat]

    ck_lat = HERE / ("%s_ckpt_lat.npy" % args.tag)
    ck_js = HERE / ("%s_ckpt.json" % args.tag)
    calls, steps_rec = [], []
    start = 0
    if args.resume and ck_lat.exists() and ck_js.exists():
        ck = json.loads(ck_js.read_text())
        if (ck.get("steps") == args.steps and ck.get("layers") == args.layers
                and int(ck.get("batch", 1)) == B):
            lat = [np.asarray(z, np.float32) for z in np.load(ck_lat)]
            lat0 = [np.asarray(z, np.float32).reshape(lat[b].shape)
                    for b, z in enumerate(ck["lat0"])]
            calls, steps_rec = ck["calls"], ck["steps_rec"]
            start = int(ck["next_step"])
            print("[t3d] RESUME at step %d" % start, flush=True)

    if args.warm:
        t0 = time.perf_counter()
        for li in range(args.layers):
            L.w_images("layers.%d" % li)
            L.params("layers.%d" % li)
        print("[t3d] warm: %d layers' projection images packed in %.1f s "
              "(startup, outside the loop's wall)"
              % (args.layers, time.perf_counter() - t0), flush=True)

    # ---- the host-only preamble of step_once, verbatim -------------------
    cap_ids, img_ids = T3.real_ids()
    c_cos, c_sin = T3.rope_cos_sin(cap_ids)
    x_cos, x_sin = T3.rope_cos_sin(img_ids)

    def pre(step_t, lat_):
        """(adaln, [xb per image], cb): B=1 reproduces the delivered order."""
        adaln = T3.t_emb(step_t, H)
        xbs = []
        for b in range(B):
            xb = T3.x_embedding(lat_[b], H)
            for key in T3.REFINERS[:2]:
                Wb = L.refblock(key)
                xb = T3.full_block_host(xb, Wb, T3.modulation(adaln, Wb),
                                        x_cos, x_sin)
            xbs.append(xb)
        cb = T3.cap_embedding(H)
        for key in T3.REFINERS[2:]:
            Wb = L.refblock(key)
            cb = T3.full_block_host(cb, Wb, T3.modulation(adaln, Wb),
                                    c_cos, c_sin)
        return adaln, xbs, cb

    # ---- T4-D: the per-layer weight payload (132.7 MB on disk) -----------
    # MEASURED (t4_hostprof.py, py3.12, no device): reading layer L's BG (88.5 MB)
    # + B3 (44.2 MB) costs 1.59-3.82 s per STEP (30 layers), i.e. 13-30 s of the
    # 60 s host bucket -- the per-layer refiner f32 forward that the brief
    # suspected is only 0.26 s/STEP.  Both fixes here move the SAME bytes into
    # the SAME BOs, so they are bit-identical by construction:
    #   --prefetch       read layer i+1 on a worker thread during layer i's
    #                    dispatch (zero extra RAM; the 269 ms dispatch is a
    #                    wide enough window for a 53 ms read)
    #   --cache-weights  read all 30 layers once before the loop (3.98 GB RAM;
    #                    only if the machine has the headroom)
    _pool = (_fut.ThreadPoolExecutor(max_workers=1)
             if (args.prefetch and not args.cache_all
                 and not args.bo_resident) else None)
    _wcache = {}

    def read_payload(key):
        return (np.load(PREP / ("%s_bg.npy" % key)).tobytes(),
                np.load(PREP / ("%s_b3.npy" % key)).tobytes())

    if args.cache_weights:
        t0 = time.perf_counter()
        nb = 0
        for li in range(args.layers):
            key = "layers.%d" % li
            _wcache[key] = read_payload(key)
            nb += len(_wcache[key][0]) + len(_wcache[key][1])
        print("[t3d] weight payloads cached: %d layers, %.2f GB in %.1f s "
              "(startup, outside the loop's wall)"
              % (args.layers, nb / 2**30, time.perf_counter() - t0), flush=True)

    if args.cache_all and not args.bo_resident:
        # T13 U-4: ONE resident pass, allocated after the W-image array and the
        # refiner cache (the device is already open; measured to fit: 5.1-7.9 GB
        # free after the allocation).  Fill sequentially, hand the loop
        # references.  BG is permuted ONCE here; ffn() sees BG_PREPERM and
        # skips its per-step permutation.
        global CACHE_BG, CACHE_B3, BG_PREPERM
        t0 = time.perf_counter()
        print("[t13] cache-all: %.2f GB free before allocation"
              % _mem_gb(), flush=True)
        cache_bg = np.empty((args.layers, BG_BYTES), np.uint8)
        cache_b3 = np.empty((args.layers, B3_BYTES), np.uint8)
        for li in range(args.layers):
            key = "layers.%d" % li
            bg = np.load(PREP / (key + "_bg.npy")).tobytes()
            b3 = np.load(PREP / (key + "_b3.npy")).tobytes()
            src = np.frombuffer(bg, np.uint8)
            if NH_BG > 1:
                (_P11.permute_bg if NH_BG_CORE else _P9.permute_bg)(
                    src, cache_bg[li], NH_BG)
            else:
                cache_bg[li][:] = src
            cache_b3[li][:] = np.frombuffer(b3, np.uint8)
            del bg, b3, src
        CACHE_BG, CACHE_B3 = cache_bg, cache_b3
        BG_PREPERM = True
        nb = cache_bg.nbytes + cache_b3.nbytes
        print("[t13] full-memory cache: %d layers, %.2f GB in %.1f s "
              "(startup, outside the loop's wall); %.2f GB free after"
              % (args.layers, nb / 2**30, time.perf_counter() - t0, _mem_gb()),
              flush=True)

    if args.bo_resident:
        # T13 U-4b: one BO per (layer, BG/B3), written ONCE.  The loop selects
        # the layer's BOs and writes only A/SBG/X -- staging drops from 49.4 GB
        # to ~17.3 GB per image.  No host cache is kept (the same 3.98 GB lives
        # in BOs), so memory is not doubled versus --cache-all.
        global BO_RESIDENT, BG_BOS, B3_BOS
        t0 = time.perf_counter()
        kf = dev.imgs["ffn"]["k"]
        print("[t13] BO-resident: %d BOs, %.2f GB, %.2f GB free before"
              % (2 * args.layers,
                 (BG_BYTES + B3_BYTES) * args.layers / 2**30, _mem_gb()),
              flush=True)
        BG_BOS = [dev.pyxrt.bo(dev.dev, BG_BYTES,
                               dev.pyxrt.bo.flags.host_only, kf.group_id(5))
                  for _ in range(args.layers)]
        B3_BOS = [dev.pyxrt.bo(dev.dev, B3_BYTES,
                               dev.pyxrt.bo.flags.host_only, kf.group_id(6))
                  for _ in range(args.layers)]
        scratch_bg = np.empty(BG_BYTES, np.uint8)
        for li in range(args.layers):
            key = "layers.%d" % li
            bg = np.load(PREP / (key + "_bg.npy")).tobytes()
            b3 = np.load(PREP / (key + "_b3.npy")).tobytes()
            src = np.frombuffer(bg, np.uint8)
            if NH_BG > 1:
                (_P11.permute_bg if NH_BG_CORE else _P9.permute_bg)(
                    src, scratch_bg, NH_BG)
            else:
                scratch_bg[:] = src
            BG_BOS[li].write(scratch_bg, 0)
            BG_BOS[li].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            B3_BOS[li].write(b3, 0)
            B3_BOS[li].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            del bg, b3, src
        dev.bo["f_BG"] = BG_BOS[0]
        dev.bo["f_B3"] = B3_BOS[0]
        BO_RESIDENT = True
        print("[t13] BO-resident: %d layers written in %.1f s (startup, "
              "outside the loop's wall); %.2f GB free after"
              % (args.layers, time.perf_counter() - t0, _mem_gb()), flush=True)
    if args.prefetch:
        print("[t3d] weight payload prefetch: layer i+1 is read on a worker "
              "thread during layer i's dispatch", flush=True)

    def one_step(i, t_step, lat_, host_ref):
        """B images through one layer loop: one payload read, one SBG, B blocks."""
        t_step0 = time.perf_counter()
        tims = [{"dev": 0.0, "write": 0.0, "pack": 0.0, "host": 0.0}
                for _ in range(B)]
        tpre = time.perf_counter()
        adaln, xbs, cb = pre(t_step, lat_)
        us = [np.concatenate([xbs[b], cb], axis=0).astype(np.float32)
              for b in range(B)]
        u_cos = np.concatenate([x_cos, c_cos], axis=0)
        u_sin = np.concatenate([x_sin, c_sin], axis=0)
        tims[0]["host"] += 1e3 * (time.perf_counter() - tpre)
        rec = {"t": float(t_step), "layers": []}
        if _pool is not None:
            _pending = _pool.submit(read_payload, "layers.0")
        for li in range(args.layers):
            key = "layers.%d" % li
            P = L.params(key)
            tp = time.perf_counter()
            mod = T3.modulation(adaln, P)
            pmark("loop.mod", tp)
            tp = time.perf_counter()
            if BO_RESIDENT:
                dev.bo["f_BG"] = BG_BOS[li]
                dev.bo["f_B3"] = B3_BOS[li]
                bg, b3 = None, None
            elif CACHE_ALL:
                bg, b3 = CACHE_BG[li], CACHE_B3[li]
            elif key in _wcache:
                bg, b3 = _wcache[key]
            elif _pending is not None:
                bg, b3 = _pending.result()
                nxt = li + 1
                _pending = (_pool.submit(read_payload, "layers.%d" % nxt)
                            if nxt < args.layers else None)
            else:
                bg, b3 = read_payload(key)
            pmark("loop.payload", tp)
            tp = time.perf_counter()
            # T10: ONE SBG for the whole batch (per (step, layer), not per image)
            sbg_b = make_sbg(P, mod)
            pmark("loop.sbg", tp)
            first = []
            for b in range(B):
                t0 = time.perf_counter()
                us[b], r = block_step(L, us[b], key, mod, u_cos, u_sin,
                                      tims[b], bg, b3,
                                      host_ref=(host_ref and b == 0),
                                      sbg_bytes=sbg_b)
                tims[b]["host"] += 1e3 * (time.perf_counter() - t0)
                rec["layers"].append(r)
                if li == 0:
                    first.append(dict(tims[b]))
                print("      %-10s img %d/%d  a %.3f xm %.3f y %.1f%s"
                      % (key, b + 1, B, r["a_absmax"], r["xm_absmax"],
                         r["y_absmax"],
                         ("  attn rel %.2e ffn rel %.2e"
                          % (r["attn_dev_vs_host"]["rel_to_absmax"],
                             r["ffn_dev_vs_host"]["rel_to_absmax"]))
                         if (host_ref and b == 0) else ""), flush=True)
            del bg, b3
            if li == 0:
                for b in range(B):
                    print("      [tim] img %d/%d first layer: dev %.0f ms  "
                          "write %.0f ms  pack %.0f ms  host %.0f ms"
                          % (b + 1, B, first[b]["dev"], first[b]["write"],
                             first[b]["pack"], first[b]["host"]), flush=True)
        tfin = time.perf_counter()
        preds = []
        for b in range(B):
            out = T3.final_layer(us[b], adaln, H)
            preds.append(T3.unpatchify(out[:64], 16, 8, 8))
        tims[0]["host"] += 1e3 * (time.perf_counter() - tfin)
        rec["pred_absmax"] = float(max(np.abs(p).max() for p in preds))
        rec["pred_absmax_by_image"] = [float(np.abs(p).max()) for p in preds]
        rec["tim"] = {k: sum(t[k] for t in tims) for k in tims[0]}
        rec["tim_by_image"] = [dict(t) for t in tims]
        rec["wall_s"] = time.perf_counter() - t_step0
        rec["wall_s_per_image"] = rec["wall_s"] / B
        return preds, rec

    for i, t in enumerate(timesteps):
        if i < start:
            continue
        preds, rec = one_step(i, t, lat, args.host_ref and i == 0)
        d = np.float32(sigmas[i + 1] - sigmas[i])
        for b in range(B):
            lat[b] = (lat[b] + d * (-preds[b])).astype(np.float32)
        steps_rec.append({"step": i, "t": float(t),
                          "pred_absmax": rec["pred_absmax"],
                          "pred_absmax_by_image": rec["pred_absmax_by_image"],
                          "lat_absmax_by_image": [float(np.abs(x).max())
                                                  for x in lat],
                          "lat_absmax": float(max(np.abs(x).max()
                                                  for x in lat)),
                          "seconds": rec["wall_s"],
                          "seconds_per_image": rec["wall_s_per_image"],
                          "batch": B,
                          "tim": rec["tim"]})
        calls.append(rec)
        np.save(ck_lat, lat[0] if B == 1 else np.stack(lat))
        ck_js.write_text(json.dumps(
            {"next_step": i + 1, "steps": args.steps, "layers": args.layers,
             "batch": B,
             "lat0": [np.asarray(x, np.float32).ravel().tolist() for x in lat0],
             "calls": calls, "steps_rec": steps_rec}), encoding="utf-8")
        print("  step %d: t=%8.3f  pred absmax %8.4f  lat absmax %9.4f  "
              "%.1f s (%.2f s/image)  "
              "(dev %.1f / write %.1f / pack %.1f / host %.1f ms)"
              % (i, t, rec["pred_absmax"],
                 float(max(np.abs(x).max() for x in lat)),
                 rec["wall_s"], rec["wall_s_per_image"],
                 rec["tim"]["dev"], rec["tim"]["write"],
                 rec["tim"]["pack"], rec["tim"]["host"]), flush=True)

    st = dev.stats
    walls = [a["wall_ms"] for a in st["attempts"] if a["state"] == COMPLETED]
    dev_sum = 1e3 * sum(st["dev_ms"].values()) if False else sum(st["dev_ms"].values())
    write_sum = sum(st["write_ms"].values())
    total = sum(s["seconds"] for s in steps_rec)
    lat_saved = lat[0] if B == 1 else np.stack(lat)
    np.save(HERE / ("%s_latents.npy" % args.tag), lat_saved)
    np.savez(HERE / ("%s_traj.npz" % args.tag),
             latents0=np.stack(lat0), latents=np.stack(lat),
             timesteps=timesteps, sigmas=sigmas)
    rep = {"seed": args.seed, "steps": args.steps, "layers": args.layers,
           "batch": B,
           "images_per_minute": 60.0 * B / total if total > 0 else 0.0,
           "seconds_per_image": total / B if B > 0 else 0.0,
           "dispatches": st["calls"], "retries": st["retries"],
           "images": [args.att_image or (("t3d_pipe2" if args.dimk64
                                          else "t3d_pipe") + ".xclbin"),
                      (args.att_ip_image + ".xclbin") if args.att_ip_image else "",
                      args.ffn_image + ".xclbin"],
           "projection_dim_k": 64 if args.dimk64 else 8,
           "dev_ms_total": float(sum(st["dev_ms"].values())),
           "dev_ms_by_image": st["dev_ms"],
           "write_ms_total": float(write_sum),
           "write_ms_by_image": st["write_ms"],
           "wall_s_total": float(total),
           "npu_share_dispatch_only": float(
               sum(st["dev_ms"].values()) / (1e3 * total)),
           "npu_share_dispatch_plus_staging": float(
               (sum(st["dev_ms"].values()) + write_sum) / (1e3 * total)),
           "steps_rec": steps_rec, "calls": calls,
           "driver": "driver376 npu_mcdm_stack_prod; three xclbins, one process"}
    (HERE / ("%s_calls.json" % args.tag)).write_text(
        json.dumps(rep, indent=1), encoding="utf-8")
    rep["dev_ms_by_tag"] = st.get("dev_by_tag", {})
    for k in ("dev_ms_by_image", "write_ms_by_image"):
        print("[t3d] %s: %s" % (k, {n: round(v, 0) for n, v in rep[k].items()}))
    print("[t3d] dispatches %d (%d retries)  device %.1f s  staging %.1f s  "
          "wall %.1f s" % (st["calls"], st["retries"],
                           rep["dev_ms_total"] / 1e3, write_sum / 1e3, total))
    print("[t10] share_weights %s" % SHARE_W)
    print("[t10] batch %d: wall %.2f s  ->  %.3f s/image  %.4f images/min  "
          "(device %.2f s = %.3f s/image, staging %.2f s = %.3f s/image)"
          % (B, total, total / B, rep["images_per_minute"],
             rep["dev_ms_total"] / 1e3, rep["dev_ms_total"] / 1e3 / B,
             write_sum / 1e3, write_sum / 1e3 / B))
    print("[t3d] dev by dispatch tag: %s"
          % {n: round(v, 1) for n, v in rep["dev_ms_by_tag"].items()})
    print("[t3d] NPU share: %.1f %% of the wall is the NPU computing; "
          "%.1f %% counting the host-side staging it forces"
          % (100 * rep["npu_share_dispatch_only"],
             100 * rep["npu_share_dispatch_plus_staging"]))
    if PROF is not None:
        nlayer = float(args.steps * args.layers)
        print("[prof] %-16s %9s %9s %8s" % ("phase", "total s", "ms/layer", "%wall"))
        for kk in sorted(PROF, key=lambda z: -PROF[z]):
            print("[prof] %-16s %9.3f %9.3f %7.1f%%"
                  % (kk, PROF[kk] / 1e3, PROF[kk] / nlayer,
                     100 * PROF[kk] / (1e3 * total)))
        print("[prof] %-16s %9.3f %9.3f %7.1f%%"
              % ("WALL", total, total * 1e3 / nlayer, 100.0))
    print("[t3d] latents -> %s_latents.npy (VAE decode is the py3.12 stage)"
          % args.tag, flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="t4a")
    ap.add_argument("--ffn-image", dest="ffn_image", default="t4_sbg",
                    help="FFN xclbin stem under imaggers/block/t4_out")
    ap.add_argument("--att-image", dest="att_image", default="",
                    help="ATTENTION xclbin stem (T5 U1); empty = the T3d image "
                         "selected by --dimk64")
    ap.add_argument("--att-dir", dest="att_dir", default="",
                    help="directory holding the attention image (default the "
                         "T3d t3b_out)")
    ap.add_argument("--att-ip-image", dest="att_ip_image", default="",
                    help="T7 U-1: the IP-ONLY xclbin stem; when given, "
                         "--att-image must be the PROJECTION-ONLY image and the "
                         "loop runs 4 projections + 1 ip instead of 5 fused "
                         "dispatches.  Empty = the T6 fused behaviour (A arm)")
    ap.add_argument("--share-weights", dest="share_weights",
                    action="store_true",
                    help="T10 U-1b: with --batch, stage the layer's BG/B3 into "
                         "the BO once instead of once per image (same bytes, "
                         "gated by the per-image latents)")
    ap.add_argument("--batch", type=int, default=1,
                    help="T10: B independent images (seeds seed..seed+B-1) per "
                         "pass; they share the layer's BG/B3 and SBG bytes and "
                         "nothing else (each image's latents must stay "
                         "byte-identical to its batch=1 run)")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--layers", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dimk64", action="store_true",
                    help="use the dim_k=64 projection image (t3d_pipe2): 60 C "
                         "roundings per projection instead of 480")
    ap.add_argument("--warm", action="store_true",
                    help="pack every layer's projection weights before the loop "
                         "(reported as startup, never inside the loop's wall)")
    ap.add_argument("--assert-sbg", dest="assert_sbg", action="store_true",
                    default=True,
                    help="check that all 96 SBG rows are equal (required by the "
                         "t4_vs FFN image); on by default")
    ap.add_argument("--no-assert-sbg", dest="assert_sbg", action="store_false")
    ap.add_argument("--nh-bg", dest="nh_bg", type=int, default=0,
                    help="T9 U-4b route (a): permute the per-layer BG payload to "
                         "the n_hold image's [pass][kk][j] order (0 = off = the "
                         "delivered path; 2 = for --ffn-image t9_nh2b)")
    ap.add_argument("--nh-bg-core", dest="nh_bg_core", action="store_true",
                    help="T11 U-1: with --nh-bg N, write the PER-CORE order "
                         "[w][pass][kk][j] (t11_permute.py) that the "
                         "per-core-fill image reads; off = the T9 [pass][kk][j] "
                         "order, i.e. the delivered path")
    ap.add_argument("--prefetch", action="store_true",
                    help="read layer i+1's BG/B3 on a worker thread during layer "
                         "i's dispatch (T4-D; zero extra RAM)")
    ap.add_argument("--cache-weights", dest="cache_weights", action="store_true",
                    help="read all 30 layers' BG/B3 once before the loop (T4-D; "
                         "3.98 GB RAM)")
    ap.add_argument("--host-ref", action="store_true",
                    help="compute the host f32 attention+FFN on step 0's first "
                         "layer only (the numerical gate)")
    ap.add_argument("--fast-codec", dest="fast_codec", action="store_true",
                    help="T6: t6_codec's byte-identical, lower-traffic codecs")
    ap.add_argument("--refiner-cache", dest="refiner_cache", default="",
                    help="T6: directory for the decoded refiner f32 weights "
                         "(2.83 GB); empty = decode every run")
    ap.add_argument("--fast-codec2", dest="fast_codec2", action="store_true",
                    help="T7 U-2: t7_codec's pack_b_nk (fewer full-size "
                         "temporaries; 40-case byte gate in t7_codec_gate.py)")
    ap.add_argument("--prof", dest="prof", action="store_true",
                    help="T8: per-phase in-loop attribution (prints s and "
                         "ms/layer at exit; no effect on the arithmetic)")
    ap.add_argument("--fast-host", dest="fast_host", action="store_true",
                    help="T8 U-1: the attention/FFN host glue on numba nJIT "
                         "kernels with one persistent workspace; byte-equal, "
                         "gated by t8_codec_gate.py")
    ap.add_argument("--fast-glue", dest="fast_glue", action="store_true",
                    help="T7 U-2: rms+RoPE in the drain's own layout, so the "
                         "four q/k transposes disappear (latents are the gate)")
    ap.add_argument("--cache-all", dest="cache_all", action="store_true",
                    default=True,
                    help="T13 U-4 (DEFAULT since T13b): read+permute all 30 "
                         "layers' BG/B3 ONCE into 3.98 GB of RAM; the loop then "
                         "does no np.load and no permutation (gate: latents "
                         "byte-identical)")
    ap.add_argument("--no-cache-all", dest="cache_all", action="store_false",
                    help="T13 U-4 A arm: the T11 path (np.load + permutation per "
                         "layer, prefetch thread)")
    ap.add_argument("--bo-resident", dest="bo_resident", action="store_true",
                    help="T13 U-4b: 30 BG + 30 B3 BOs written once (3.98 GB of "
                         "BO memory); the loop writes only A/SBG/X.  Implies "
                         "--no-cache-all (same 3.98 GB, in BOs not RAM).")
    ap.add_argument("--wpack-array", dest="wpack_array", action="store_true",
                    help="T6: one preallocated 2.0 GB array for the projection "
                         "images instead of 120 x 16.6 MB bytes objects, "
                         "allocated BEFORE the device is opened")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    global ASSERT_SBG, REFCACHE, WPACK, NH_BG, NH_BG_CORE
    ASSERT_SBG = bool(a.assert_sbg)
    global CACHE_ALL
    CACHE_ALL = bool(getattr(a, "cache_all", True))
    if getattr(a, "bo_resident", False):
        CACHE_ALL = False
    NH_BG = int(getattr(a, "nh_bg", 0))
    NH_BG_CORE = bool(getattr(a, "nh_bg_core", False))
    if a.fast_codec:
        import t6_codec
        t6_codec.install()
        print("[t6] fast codec installed: pack_b_nk 1.78x, unpack_blocked "
              "1.06x (byte-identical, t6_codec_gate.py ALL_PASS)", flush=True)
    if a.fast_codec2:
        import t7_codec
        t7_codec.install()
        print("[t7] fast codec 2 installed: pack_b_nk K 1.48x / V 1.10x, "
              "unpack from t6 (t7_codec_gate.py ALL_PASS)", flush=True)
    global FAST_HOST, SCRATCH, PROF
    FAST_HOST = bool(a.fast_host)
    if a.prof:
        PROF = {}
    if FAST_HOST:
        # Every codec this loop calls becomes the nJIT kernel, and every
        # in-loop buffer comes from one persistent workspace.  The gate is
        # byte equality against t3d_codec on the 40 T6/T7 adversarial cases
        # (t8_codec_gate.py), plus the delivered latents.
        H.install()
        SCRATCH = H.Scratch()
        print("[t8] fast host installed: pack_b_nk %.2fx, unpack %.2fx, "
              "pack_blocked %.2fx, x_image64 %.2fx, bf16 %.2fx, rope %.2fx "
              "(byte-equal, t8_codec_gate.py ALL_PASS)"
              % (9.6, 4.3, 16.7, 4.2, 40.9, 4.4), flush=True)
    global GLUE_FAST
    GLUE_FAST = bool(a.fast_glue)
    if GLUE_FAST:
        print("[t7] fast glue: rms+RoPE stay in the drained (HEADS, M, HD) "
              "layout (no q/k transpose)", flush=True)
    REFCACHE = a.refiner_cache or None
    if REFCACHE:
        pathlib.Path(REFCACHE).mkdir(parents=True, exist_ok=True)
        print("[t6] refiner cache dir: %s" % pathlib.Path(REFCACHE).resolve(),
              flush=True)
    if a.wpack_array:
        n = a.layers * 4
        print("[t6] allocating %.2f GB for the projection images BEFORE the "
              "device is opened" % (n * W_BYTES / 2**30), flush=True)
        WPACK = np.empty(n * W_BYTES, np.uint8)
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
