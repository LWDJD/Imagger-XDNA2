# t21_3b3_chain.py -*- Python -*-
"""T21-3B3 U1/U2: the six-size VAE decoder chain with conv/GN/UP on the NPU.

WHAT THIS FILE IS
-----------------
The 3b1 driver (vaeg_chain_single.py) projected the decoder with
device-exact conv *models on the host* and real NPU GN/UP kernels.  3b2 and
3b2vf then built and gated 92 (geometry) xclbins -- every (M,K,N) conv row,
every (C,S) GroupNorm, every upsample plane of the six product sizes.  This
driver is the missing executor: it runs the real decoder with

    conv      -> the 3b2/3b2vf conv xclbins   (NPU)
    GroupNorm -> the 3b2/3b2vf gn xclbins     (NPU)
    upsample  -> the 3b2/3b2vf up xclbins     (NPU)
    SiLU / add / attention / conv concatenation -> host fp32 torch (declared)

and nothing else.  Every device call is one fresh py310 process
(vaeg_run_xrt.py), because that is the platform's measured law: one launch
per xclbin per session (state-8) and the ERT watchdog at ~7 s.

THE EXECUTOR RULES (from T21-3B2VF-REPORT.md section 7)
-------------------------------------------------------
conv  a row is dispatched either as ONE launch of the exact (M,K,N) image or
      as ceil(M/1024) blocks of the M'=1024 variant image.  Because C's rows
      are independent and K/Kc/k-order are M-blind, the block concatenation
      is bit-identical to a full-M dispatch (18/18 rows, max_ulp 0.0).
      Two families are forced:
        * (K=2336, N=256) MUST use the N=128 half-width image, twice per
          block.  The full-width image stalls (zero output, no completion),
          and column independence makes the halves bit-exact (9/9 rows).
        * any exact image whose M*K*N exceeds EXACT_CAP MAC is replaced by
          the M-block image: those are exactly the 3b2 DISPATCH-TIMEOUT
          rows (a >~7 s dispatch is killed by the ERT watchdog).
GN    (C,S) keyed; the xclbin's chunking is internal (whole plane or
      sub-plane, gamma base is a memref offset), the host tap is one
      (C,S) stream -- identical for both designs.
UP    (C,H,W) keyed, NOT (C,S): the kernel is upsample_nearest2(n, w) and
      w is an immediate in the xclbin's instruction stream (group_norm.cc:
      y[(2r+j)(2w)+2c] from x[r*w+c]), so a second (H,W) factorization of the
      same S needs its own image.

HOST / DEVICE SPLIT, HONESTLY
-----------------------------
Everything the device touches is listed in the calls log with its stem, its
block decomposition and its per-call check.  The host still does: im2col,
bf16 rounding of the activation stream, the BFP16 packing of the weights
(the xclbin's input stream is host-produced -- that is the same contract the
gated rows used), the concat of block outputs, and the fp32 SiLU/add/attention
of the decoder graph.  No host float32 GEMM stands in for a conv.

Usage (py312; the device side is spawned as py310):
    py312 t21_3b3_chain.py --index              # what xclbins exist, per geometry
    py312 t21_3b3_chain.py --plan               # paper prediction, six sizes
    py312 t21_3b3_chain.py --size 128x128
    py312 t21_3b3_chain.py --all --repeat 2
    py312 t21_3b3_chain.py --selfref            # 64x64 regression vs 3b1 (37.3189 dB)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent          # ...\imaggers\vae
ROOT = HERE.parent.parent.parent                        # ...\<workspace>
OUT = HERE / "out"
ART = OUT / "t21_3b2"                                   # 3b2 + 3b2vf artifacts
T3 = OUT / "t21_3b3"                                    # this unit's artifacts
OPS = OUT / "t22_vae" / "ops"                          # T22-B element-wise images
PORT = ROOT / "zimage-port"
MODELS = PORT / "_models" / "Z-Image-Turbo"
XRT_RUNNER = HERE / "vaeg_run_xrt.py"
# Third-party locations: resolved, never hard-coded (imaggers/paths.py).
_root = pathlib.Path(__file__).resolve().parent
while not (_root / 'imaggers' / 'paths.py').is_file():
    if _root.parent == _root:
        raise RuntimeError('cannot locate the checkout root')
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from imaggers import paths as _paths  # noqa: E402

PY310 = _paths.python310()
XRT_DIR = _paths.xrt_dir()

sys.path.insert(0, str(HERE))
sys.path.append(str(ROOT / "_pylibs"))
sys.path.append(str(ROOT / "_pylibs" / "mlir_aie" / "python"))
sys.path.insert(0, str(PORT))

SCALING, SHIFT = 0.3611, 0.1159
GROUPS, EPS = 32, 1e-6
TILE = dict(m=16, k=8, n=16)
# T22-B: the largest element-wise image that is still dispatchable inside the
# ~7 s ERT watchdog (measured: a 4194304-element vaddf image completes run 0 in
# 51.9 ms and times out on run 1; 8388608 times out on run 0).  Bigger tensors
# are split over this size -- exact, because the op is per element.
OPS_MAX_ELEMS = 2097152
MC_M_BLOCK = 1024
# a dispatch must stay well inside the ERT watchdog (~7 s measured).  6.0 GMAC
# is ~4 s at the measured 1.3-1.9 GMAC/s; anything larger goes to blocks.
EXACT_CAP = 6.0e9
DEVICE_WAIT_MS = 600000
MB = 1024.0 * 1024.0

SIZES = [("128x128", 16, 16), ("64x256", 8, 32), ("144x128", 18, 16),
         ("192x128", 24, 16), ("128x256", 16, 32), ("256x256", 32, 32)]
SIZE_LATENT = {s: (lh, lw) for s, lh, lw in SIZES}

TIMERS: dict = {}


def tick(name: str, dt: float) -> None:
    TIMERS[name] = TIMERS.get(name, 0.0) + dt


def _force_utf8() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def sha16(b) -> str:
    return hashlib.sha256(bytes(b)).hexdigest()[:16]


def _write_json(path: pathlib.Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _rm(path: pathlib.Path) -> None:
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# the xclbin index: what the 92 gated rows + the 3b1 images actually cover
# ---------------------------------------------------------------------------

RE_CG = re.compile(r"^cg_M(\d+)_K(\d+)_N(\d+)$")
RE_CG_OLD = re.compile(r"^cg_[a-z0-9_]*?_M(\d+)_K(\d+)_N(\d+)$")
RE_CGV = re.compile(r"^cgv_M(\d+)_K(\d+)_N(\d+)$")
RE_GN = re.compile(r"^gn_(?:single|stream)_C(\d+)_S(\d+)$")
RE_GNV = re.compile(r"^gnv_C(\d+)_S(\d+)$")
RE_GN31 = re.compile(r"^vaeg_gn_(?:single|stream)_.*_C(\d+)_H(\d+)_W(\d+)$")
RE_UP = re.compile(r"^up_C(\d+)_S(\d+)$")
RE_UPV = re.compile(r"^upv_C(\d+)_S(\d+)_R(\d+)$")
RE_UP31 = re.compile(r"^vaeg_up_kernel_.*_C(\d+)_H(\d+)_W(\d+)$")
RE_UPW = re.compile(r"^upw_C(\d+)_H(\d+)_W(\d+)$")
# T22-B: the element-wise ops (silu / residual add), one image per element count
RE_OPS = re.compile(r"^(silu|vadd|vaddf)_N(\d+)$")

IDX: dict = {}


def scan_images(dirs=None) -> dict:
    """Index every xclbin by the geometry it was BURNED-IN with."""
    dirs = dirs or [OUT, ART, T3, OPS]
    conv_exact, conv_mb = {}, {}
    gn, up, paths = {}, {}, {}
    ops = {}
    for d in dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.xclbin")):
            stem = p.stem
            paths[stem] = str(d)
            m = RE_CG.match(stem) or RE_CG_OLD.match(stem)
            if m:
                # 3b1's five images carry a tag between cg_ and _M
                conv_exact[(int(m.group(1)), int(m.group(2)), int(m.group(3)))] = stem
                continue
            m = RE_CGV.match(stem)
            if m:
                mb, K, N = int(m.group(1)), int(m.group(2)), int(m.group(3))
                conv_mb[(K, N)] = (stem, mb)
                continue
            m = RE_GN.match(stem) or RE_GNV.match(stem)
            if m:
                gn.setdefault((int(m.group(1)), int(m.group(2))), []).append(
                    dict(stem=stem, hw=None))
                continue
            m = RE_GN31.match(stem)
            if m:
                C, H, W = int(m.group(1)), int(m.group(2)), int(m.group(3))
                gn.setdefault((C, H * W), []).append(
                    dict(stem=stem, hw=(H, W)))
                continue
            m = RE_OPS.match(stem)
            if m:
                ops[(m.group(1), int(m.group(2)))] = stem
                continue
            m = RE_UP31.match(stem) or RE_UPW.match(stem)
            if m:
                # (C,H,W) in the name: the only unambiguous UP key
                up[(int(m.group(1)), int(m.group(2)), int(m.group(3)))] = stem
                continue
            m = RE_UP.match(stem) or RE_UPV.match(stem)
            if m:
                # S-only name: the (H,W) it was built with is recorded in the
                # build json; resolved (and checked) at call time.
                up.setdefault(("S", int(m.group(1)), int(m.group(2))), []).append(
                    stem)
                continue
    out = dict(conv_exact=conv_exact, conv_mb=conv_mb, gn=gn, up=up,
               ops=ops, paths=paths, dirs=[str(d) for d in dirs])
    return out


def image_index() -> dict:
    global IDX
    if not IDX:
        IDX = scan_images()
    return IDX


def rup(v: int, m: int = 32) -> int:
    return (int(v) + m - 1) // m * m


def conv_mkn(cin: int, cout: int, k: int, bias: bool = True) -> tuple:
    k_real = cin * k * k
    K = rup(k_real + 8, 32) if bias else rup(k_real, 32)
    N = rup(cout, 16)
    return K, N, k_real


def select_conv_image(M: int, K: int, N: int) -> dict:
    """The executor's conv routing rule (see the module docstring)."""
    idx = image_index()
    exact, mb = idx["conv_exact"], idx["conv_mb"]
    mac = float(M) * K * N
    # the full-width (2336,256) image is the measured STALL (zero output); it
    # exists only as the evidence artifact and must never be dispatched.
    STALL = "cgv_M1024_K2336_N256"
    if K == 2336 and N == 256:
        stem, mblk = mb[(2336, 128)]
        assert stem != STALL
        return dict(stem=stem, m_block=mblk, n_xcl=128, halves=2,
                    why="family (K=2336,N=256) needs the N=128 half")
    if (K, N) in mb and ((M, K, N) not in exact or mac > EXACT_CAP):
        stem, mblk = mb[(K, N)]
        if stem == STALL:
            raise RuntimeError("routing picked the stall image for %s"
                               % ((M, K, N),))
        return dict(stem=stem, m_block=mblk, n_xcl=N, halves=1,
                    why=("M-block (%.1f GMAC dispatch over the %.0f GMAC cap)"
                         % (mac / 1e9, EXACT_CAP / 1e9)) if mac > EXACT_CAP
                    else "M-block")
    if (M, K, N) in exact:
        return dict(stem=exact[(M, K, N)], m_block=None, n_xcl=N, halves=1,
                    why="exact geometry")
    if (K, N) in mb:
        stem, mblk = mb[(K, N)]
        if stem == STALL:
            raise RuntimeError("routing picked the stall image for %s"
                               % ((M, K, N),))
        return dict(stem=stem, m_block=mblk, n_xcl=N, halves=1, why="M-block")
    raise KeyError("no conv xclbin for (M=%d, K=%d, N=%d)" % (M, K, N))


def gn_stem_for(C: int, S: int, hw=None) -> tuple:
    idx = image_index()["gn"]
    cands = idx.get((C, S))
    if not cands:
        raise KeyError("no GroupNorm xclbin for (C=%d, S=%d)" % (C, S))
    if hw is not None:
        for c in cands:
            if c["hw"] == tuple(hw):
                return c["stem"], c["hw"]
    return cands[0]["stem"], cands[0]["hw"]


def _up_hw_recorded(stem: str):
    """The (H,W) an S-keyed UP image was built with, from its artifacts."""
    b = ART / (stem + ".build.json")
    if b.is_file():
        sp = (json.loads(b.read_text(encoding="utf-8")).get("spec") or {})
        if sp.get("H") is not None:
            return (int(sp["H"]), int(sp["W"]))
    for cand in (ART / (stem + "_results.json"), T3 / (stem + ".json")):
        if cand.is_file():
            g = (json.loads(cand.read_text(encoding="utf-8")).get("geom") or {})
            if g.get("H") is not None:
                return (int(g["H"]), int(g["W"]))
    return None


def up_stem_for(C: int, H: int, W: int) -> tuple:
    """The UP image for exactly this (C,H,W).

    w is an immediate in the instruction stream (group_norm.cc:
    upsample_nearest2(x, y, n, w) writes y[(2r+j)*(2w)+2c] from x[r*w+c]), so
    an S-keyed stem is only usable when its build json says the same (H,W).
    """
    idx = image_index()["up"]
    stem = idx.get((C, H, W))
    if stem:
        return stem, "exact (C,H,W)"
    cands = idx.get(("S", C, H * W)) or []
    for s in cands:
        hw = _up_hw_recorded(s)
        if hw == (H, W):
            return s, "exact (C,S), recorded (H,W) matches"
    if cands and any(_up_hw_recorded(s) is None for s in cands):
        return cands[0], "S-keyed, (H,W) not recorded (row's first variant)"
    raise KeyError("no upsample xclbin for (C=%d, H=%d, W=%d); have %s"
                   % (C, H, W, [(_up_hw_recorded(s), s) for s in cands]))


# ---------------------------------------------------------------------------
# the device: one py310 process per launch (the platform's law)
# ---------------------------------------------------------------------------

class Dev:
    def __init__(self, tag: str = "", keep_npz: bool = False,
                 mode: str | None = None):
        self.tag = tag
        self.tmp = T3 / "_tmp" / (tag or "run")
        _rm(self.tmp)
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.seq = 0
        self.calls: list = []
        self.keep_npz = keep_npz
        self.env = dict(os.environ)
        cache = T3 / "_npucache" / (tag or "run")
        _rm(cache)
        cache.mkdir(parents=True, exist_ok=True)
        self.env["NPU_CACHE_DIR"] = str(cache)
        self.env["NPU_CACHE_HOME"] = str(cache)
        # T22-A: launch architecture.  "proc" = one py310 process per launch
        # (the 3b3 delivery path, kept as the A/B arm and as the fallback);
        # "srv" = one py310 device server per host process, one launch per
        # request (see vaeg_devsrv.py).  ZIMAGE_VAE_LAUNCH overrides.
        self.mode = mode or os.environ.get("ZIMAGE_VAE_LAUNCH", "proc")
        self.srv = None
        self.last_srv_meta = None
        self.barrier_spec = None
        if self.mode == "srv":
            import t22_vae_srv as S
            self.srvmod = S
            self.barrier_spec = self._barrier_spec()
            try:
                self.srv = S.get(tag=(tag or "run"),
                                 barrier=self.barrier_spec,
                                 cache_root=T3 / "_npucache",
                                 log_dir=T3 / "_devsrv")
            except Exception as exc:            # noqa: BLE001
                # a device server that cannot start must not take the NPU
                # path down with it: fall back to the per-launch executor
                # (same xclbins, same bytes, the slow door)
                self.srv_error = "%s: %s" % (type(exc).__name__, exc)
                print("[dev] srv unavailable (%s); per-launch fallback"
                      % self.srv_error[:200], flush=True)
                self.mode = "proc"
                self.srv = None

    def _barrier_spec(self):
        """The smallest GN image, used as the between-same-image barrier.

        T21-3B1 measured 24 barriers at 1.1 ms each on the same 9-13 KB
        images and byte-identical output with them on.  In a long-lived
        process the back-to-back count per image is much larger than 3b1's 3
        (16 for M=16384 at 128^2), so the barrier stays on by default here;
        --no-barrier / ZIMAGE_VAE_BARRIER=0 turns it off for the A/B.
        """
        if os.environ.get("ZIMAGE_VAE_BARRIER", "1") not in ("1", "on", "yes"):
            return None
        idx = image_index()
        for key in ((512, 64), (512, 256), (128, 4096), (256, 1024)):
            cands = idx["gn"].get(key)
            if not cands:
                continue
            stem = cands[0]["stem"]
            C, S = key
            gb = (4 * C + 63) // 64 * 64 + 16
            return dict(stem=stem, dir=idx["paths"].get(stem, str(ART)),
                        ins=[C * S * 2, gb], out=C * S * 2)
        return None

    def _dir_of(self, stem: str) -> pathlib.Path:
        """The directory the xclbin lives in (3b1 images are in out/, not
        out/t21_3b2/; the stem alone does not say which)."""
        return pathlib.Path(image_index()["paths"].get(stem, str(ART)))

    # -- raw launch ------------------------------------------------------
    def _launch(self, stem: str, payload: dict, out_len: int) -> bytes:
        self.seq += 1
        if self.mode == "srv":
            return self._launch_srv(stem, payload, out_len)
        tag = "%06d" % self.seq
        npz = self.tmp / ("i" + tag + ".npz")
        raw = self.tmp / ("o" + tag + ".npy")
        np.savez(npz, out_len=np.int64(out_len), **payload)
        d = self._dir_of(stem)
        cmd = [str(PY310), str(XRT_RUNNER), "--npz", str(npz),
               "--xclbin", str(d / (stem + ".xclbin")),
               "--insts", str(d / (stem + ".bin")),
               "--out", str(raw), "--runs", "1",
               "--wait-ms", str(DEVICE_WAIT_MS)]
        t0 = time.perf_counter()
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True,
                           encoding="utf-8", errors="replace", env=self.env)
        dt = time.perf_counter() - t0
        out = r.stdout or ""
        if r.returncode != 0:
            err = (r.stderr or "").strip().splitlines()[-6:]
            _write_json(T3 / ("FAIL_%s_%s.json" % (stem, tag)),
                        dict(stem=stem, rc=r.returncode, stdout=out[-4000:],
                             stderr="\n".join(err), cmd=cmd, ts=_ts()))
            raise RuntimeError("device run failed (%s, exit %s): %s"
                               % (stem, r.returncode, " | ".join(err)))
        m = re.search(r"wall ([\d.]+) ms", out)
        self.last_kernel_ms = float(m.group(1)) if m else None
        data = np.load(raw).view(np.uint8).reshape(-1).tobytes()
        if not self.keep_npz:
            npz.unlink(missing_ok=True)
        _rm(raw)
        return data

    def _launch_srv(self, stem: str, payload: dict, out_len: int) -> bytes:
        order = [payload["in%d_bytes" % i]
                 for i in range(int(payload["n_in"]))]
        t0 = time.perf_counter()
        r = self.srv.run(stem, self._dir_of(stem), order, out_len,
                         seq=self.seq, barrier=True)
        dt = time.perf_counter() - t0
        self.last_kernel_ms = float(r["wall_ms"])
        self.last_srv_meta = dict(fill_ms=r["fill_ms"], read_ms=r["read_ms"],
                                  wall_ms=r["wall_ms"], state=r["state"],
                                  retries=int(r["retries"]),
                                  send_ms=1e3 * r["host_send_s"],
                                  wait_ms=1e3 * r["host_wait_s"],
                                  host_ms=1e3 * dt,
                                  transfer_ms=1e3 * dt - float(r["wall_ms"]))
        if r["raw"] is None:
            raise RuntimeError("device launch did not complete (%s, state %s)"
                               % (stem, r["state"]))
        return r["raw"]

    # -- the three ops ---------------------------------------------------
    def conv(self, A_b, B_bytes: bytes, img: dict, M: int, K: int, N: int,
             check: dict | None = None) -> tuple:
        import vae_npu as N_
        m_block = int(img["m_block"] or M)
        halves = int(img["halves"])
        n_xcl = int(img["n_xcl"])
        assert halves * n_xcl == N, (N, n_xcl, halves)
        row_bytes = K * 2
        b_per_half = (K * n_xcl // 8) * 9
        assert len(B_bytes) == b_per_half * halves, (len(B_bytes), b_per_half,
                                                     halves)
        A = np.ascontiguousarray(A_b).view(np.uint8)
        assert A.shape == (M, row_bytes), (A.shape, M, row_bytes)
        out = np.zeros((M, N), np.uint16)
        metas = []
        buf = np.zeros(m_block * row_bytes, np.uint8)
        blocks = (M + m_block - 1) // m_block
        t_host = 0.0
        t_dev = 0.0
        walls = []
        for b in range(blocks):
            r0 = b * m_block
            r1 = min(M, r0 + m_block)
            nb = r1 - r0
            if blocks == 1 and nb == m_block:
                in0 = A
            else:
                buf[:nb * row_bytes] = A[r0:r1].reshape(-1)
                if nb < m_block:
                    buf[nb * row_bytes:] = 0
                in0 = buf.copy()
            for h in range(halves):
                payload = dict(n_in=np.int64(2), in0_bytes=in0,
                               in1_bytes=B_bytes[h * b_per_half:
                                                (h + 1) * b_per_half])
                t0 = time.perf_counter()
                raw = self._launch(img["stem"], payload,
                                   m_block * n_xcl * 2)
                t_dev += time.perf_counter() - t0
                walls.append(self.last_kernel_ms)
                metas.append(self.last_srv_meta)
                out[r0:r1, h * n_xcl:(h + 1) * n_xcl] = (
                    np.frombuffer(raw, np.uint16).reshape(m_block, n_xcl)[:nb])
        res = np.empty((M, N), np.float32)
        np.copyto(res.view(np.uint32), out.astype(np.uint32) << 16)
        t_check = 0.0
        if check is not None:
            t_c = time.perf_counter()
            rows = int(check.get("rows", 64))
            rows = min(rows, M)
            from t21_3b2_fastref import bfp_gemm_model_v
            A_s = A_b[:rows]
            _q, ref = bfp_gemm_model_v(A_s, check["W_pad"],
                                       m=TILE["m"], k=TILE["k"],
                                       n=TILE["n"], bqT=check["bqT"])
            got = res[:rows]
            from ml_dtypes import bfloat16
            gb = got.astype(bfloat16).astype(np.float64)
            rb = ref.astype(bfloat16).astype(np.float64)
            eq = float((gb == rb).mean())
            max_ulp = float((np.abs(gb - rb) / np.maximum(
                _bf16_ulp(rb), 2.0 ** -126)).max()) if gb.size else 0.0
            check.update(sample_rows=rows, sample_bit_exact=eq,
                         sample_max_ulp=max_ulp,
                         sample_pass=bool(max_ulp <= 1.0))
            t_check = time.perf_counter() - t_c
        self.calls.append(dict(kind="conv", stem=img["stem"], M=M, K=K, N=N,
                               blocks=blocks, m_block=m_block, halves=halves,
                               n_xcl=n_xcl, why=img["why"],
                               gmac=blocks * m_block * K * n_xcl / 1e9 * halves,
                               dev_s=t_dev, check_s=t_check, kernel_ms=walls,
                               srv=metas, check=check))
        return res, t_dev, t_check

    def groupnorm(self, x_b, g_b, b_b, C: int, S: int,
                  grade=None) -> tuple:
        import t21_3b2_plan as P
        stem, hw = gn_stem_for(C, S, grade)
        gb = P.build_gb(C, g_b, b_b, with_stats=True)
        payload = dict(n_in=np.int64(2),
                       in0_bytes=np.ascontiguousarray(x_b).view(np.uint8),
                       in1_bytes=gb)
        t0 = time.perf_counter()
        raw = self._launch(stem, payload, C * S * 2)
        dt = time.perf_counter() - t0
        y = np.frombuffer(raw, np.uint16).astype(np.uint32)
        y = (y << 16).view(np.float32).reshape(C, S)
        self.calls.append(dict(kind="gn", stem=stem, C=C, S=S, hw=hw,
                               dev_s=dt, kernel_ms=self.last_kernel_ms,
                               srv=self.last_srv_meta))
        return y, dt

    def upsample(self, x_b, C: int, H: int, W: int) -> tuple:
        stem, why = up_stem_for(C, H, W)
        payload = dict(n_in=np.int64(1),
                       in0_bytes=np.ascontiguousarray(x_b).view(np.uint8))
        t0 = time.perf_counter()
        raw = self._launch(stem, payload, C * H * W * 8)
        dt = time.perf_counter() - t0
        y = np.frombuffer(raw, np.uint16).astype(np.uint32)
        y = (y << 16).view(np.float32).reshape(C, 2 * H, 2 * W)
        self.calls.append(dict(kind="up", stem=stem, C=C, H=H, W=W, why=why,
                               dev_s=dt, kernel_ms=self.last_kernel_ms,
                               srv=self.last_srv_meta))
        return y, dt

    def ops(self, kind: str, total: int, payloads: list,
            out_len: int, f32: bool = False) -> tuple:
        """One element-wise launch (T22-B): silu, the residual add, or vaddf.

        f32=True means the image's output BO is float32 (vaddf): the residual
        add that keeps the host's f32 sum instead of storing BF16.
        """
        stem = image_index()["ops"].get((kind, total))
        if stem is None:
            raise KeyError("no %s xclbin for N=%d" % (kind, total))
        payload = dict(n_in=np.int64(len(payloads)))
        for i, p in enumerate(payloads):
            payload["in%d_bytes" % i] = p
        t0 = time.perf_counter()
        raw = self._launch(stem, payload, out_len)
        dt = time.perf_counter() - t0
        if f32:
            y = np.frombuffer(raw, np.float32)
        else:
            y = np.frombuffer(raw, np.uint16).astype(np.uint32)
            y = (y << 16).view(np.float32)
        self.calls.append(dict(kind=kind, stem=stem, N=int(total), dev_s=dt,
                               kernel_ms=self.last_kernel_ms,
                               srv=self.last_srv_meta))
        return y, dt

    def close(self) -> None:
        _rm(self.tmp)
        if self.mode == "srv" and self.srv is not None:
            self.srv.close()          # one context at a time: waited, not
            self.srv = None           # just signalled


def _round_bf16_np(x) -> np.ndarray:
    """BF16 round-to-nearest-even, the integer form (same as vaeg_kernels)."""
    xf = np.ascontiguousarray(x, dtype=np.float32)
    u = xf.view(np.uint32)
    bias = ((u >> 16) & 1) + 0x7FFF
    return ((u + bias) & np.uint32(0xFFFF0000)).view(np.float32)


def _bf16_ulp(x) -> np.ndarray:
    xf = np.asarray(x, np.float64)
    xf = np.where(xf == 0.0, np.float64(2.0 ** -126), np.abs(xf))
    return np.exp2(np.floor(np.log2(xf)).astype(np.int64) - 7)


# ---------------------------------------------------------------------------
# patching the real decoder: conv / GroupNorm / Upsample2D -> the NPU ops
# ---------------------------------------------------------------------------

class Chain:
    def __init__(self, dec, dev: Dev, checks: bool = True,
                 verbose: bool = True, full_check: bool = False,
                 conv_impl: str = "npu", ops_npu: bool | None = None):
        self.dec = dec
        self.dev = dev
        self.checks = checks
        self.verbose = verbose
        self.full_check = full_check
        self.conv_impl = conv_impl
        # T22-B: put SiLU (x31) and the residual add (x40) on the board too.
        # ZIMAGE_VAE_OPNPU=1 selects it; the default keeps the 3b3 path.
        v = os.environ.get("ZIMAGE_VAE_OPNPU", "0").lower()
        if ops_npu is not None:
            self.ops_silu = self.ops_add = bool(ops_npu)
        else:
            # "1"/all => both; "silu" / "add" => that one only (they have
            # very different costs on the board: see T22-VAE-REPORT.md 3.5)
            self.ops_silu = v in ("1", "on", "yes", "all", "silu")
            self.ops_add = v in ("1", "on", "yes", "all", "add")
        self.ops_npu = self.ops_silu or self.ops_add
        self.trace = {}
        # A/B arm for the add: install the re-written ResnetBlock2D.forward but
        # keep the host add in it.  It must be BIT-IDENTICAL to diffusers'.
        self.resnet_rewrite = (self.ops_add
                               or os.environ.get("ZIMAGE_VAE_RESNET_REWRITE",
                                                 "0") in ("1", "on", "yes"))
        self.mods: dict = {}

    def unpatch(self) -> None:
        """Restore the modules' class-level forward methods.

        The decoder is shared across sizes (one load per process); leaving the
        device forwards installed would make the NEXT size's fp32 reference
        pass dispatch the PREVIOUS size's ops.
        """
        for mod in getattr(self, "_patched", []):
            try:
                del mod.forward
            except AttributeError:
                pass
        self._patched = []

    def patch(self, gnup: bool = True) -> dict:
        import torch
        import torch.nn as nn
        self._patched = []
        # the resnets that feed an attention block's residual (see below)
        self._add_skip = set()
        for pname, pmod in self.dec.named_modules():
            atts = getattr(pmod, "attentions", None)
            ress = getattr(pmod, "resnets", None)
            if atts and ress:
                for i in range(min(len(atts), len(ress))):
                    self._add_skip.add("%s.resnets.%d" % (pname, i))
        n_c = n_g = n_u = 0
        for name, mod in self.dec.named_modules():
            if isinstance(mod, nn.Conv2d):
                self._patch_conv(name, mod)
                self._patched.append(mod)
                n_c += 1
            elif isinstance(mod, nn.GroupNorm) and gnup:
                self._patch_gn(name, mod)
                self._patched.append(mod)
                n_g += 1
        if gnup:
            for name, mod in self.dec.named_modules():
                if mod.__class__.__name__ == "Upsample2D":
                    self._patch_up(name, mod)
                    self._patched.append(mod)
                    n_u += 1
        # the HOST fp32 stages are bucketed too (product gate 2 needs every
        # second attributed): SiLU, the residual add / module glue, and the
        # mid-block attention.  The two "container" wrappers book only their
        # OWN time (whatever their children did not account for).
        for name, mod in self.dec.named_modules():
            cls = mod.__class__.__name__
            if cls in ("SiLU", "Silu"):
                self._patch_silu(name, mod)
                self._patched.append(mod)
            elif cls == "ResnetBlock2D":
                # A resnet whose output is the RESIDUAL of an attention block
                # must keep its unrounded f32 sum: AttentionBlock adds the
                # block INPUT to the attention output and only the sum is
                # later rounded, so rounding it earlier changes the result
                # (measured: max|d| 0.081 on the 128^2 image, 2026-10-20).
                if not (self.resnet_rewrite
                        and self._patch_resnet(name, mod)):
                    self._patch_container(name, mod, "add_s")
                self._patched.append(mod)
            elif cls in ("AttentionBlock", "Attention"):
                self._patch_container(name, mod, "attn_s")
                self._patched.append(mod)
        return dict(convs=n_c, gns=n_g, ups=n_u)

    # -- conv -------------------------------------------------------------
    def _patch_conv(self, name: str, mod) -> None:
        import torch
        import vae_npu as N_
        from ml_dtypes import bfloat16
        from t21_3b2_fastref import unpack_b_bytes
        Cout, Cin, kh, kw = [int(v) for v in mod.weight.shape]
        pad = (kh - 1) // 2
        W = mod.weight.detach().cpu().numpy().astype(np.float32)
        b = (mod.bias.detach().cpu().numpy().astype(np.float32)
             if mod.bias is not None else None)
        Wmat = W.reshape(Cout, -1).T
        K_real, N_real = Wmat.shape
        K, N, _ = conv_mkn(Cin, Cout, kh, b is not None)
        W_pad, _used = N_.pad_kn(Wmat, K, N, bias_row=b)
        B_bytes = N_.pack_b_blocks(W_pad, K, N, TILE["k"], TILE["n"])
        bias_col = K_real if b is not None else None
        bqT = unpack_b_bytes(B_bytes, K, N) if self.checks else None
        state = dict(mod=name, Cout=Cout, Cin=Cin, k=kh, K=K, N=N,
                     K_real=K_real, N_real=N_real, first=True)

        def forward(x, _N=N_, _state=state, _W=W_pad, _B=B_bytes,
                    _kh=kh, _kw=kw, _pad=pad, _bc=bias_col, _bq=bqT):
            t0 = time.perf_counter()
            xn = x.detach().cpu().numpy()
            if xn.shape[0] != 1:
                raise ValueError("chain decodes one image at a time")
            xin = np.ascontiguousarray(xn[0], np.float32)
            H, Wd = int(xin.shape[1]), int(xin.shape[2])
            M = H * Wd
            A_ic = _N.im2col(xin, _kh, _kw, _pad)
            A_pad = _N.pad_a(A_ic, _state["K"], _bc)
            A_b = A_pad.astype(bfloat16)
            del A_ic, A_pad
            t1 = time.perf_counter()
            img = select_conv_image(M, _state["K"], _state["N"])
            chk = None
            if self.checks and _state["first"]:
                chk = dict(W_pad=_W, bqT=_bq,
                           rows=(M if (self.full_check and M <= 4096) else 64))
            if self.conv_impl == "model":
                # the 3b1 tier's conv: the device-exact model ON THE HOST.
                # Only for the regression experiment (it is NOT the product).
                from t21_3b2_fastref import bfp_gemm_model_v
                _q, C = bfp_gemm_model_v(A_b, _W, m=16, k=8, n=16, bqT=_bq)
                C = np.asarray(C, np.float32)
                dev_s, check_s = 0.0, 0.0
                self.dev.calls.append(dict(kind="conv_host_model", M=M,
                                           K=_state["K"], N=_state["N"],
                                           stem="bfp_gemm_model_v"))
            else:
                C, dev_s, check_s = self.dev.conv(A_b, _B, img, M,
                                                  _state["K"], _state["N"],
                                                  check=chk)
            t2 = time.perf_counter()
            if chk is not None:
                _state["first"] = False
                _state["sample"] = {k: v for k, v in chk.items()
                                    if k.startswith("sample")}
                tick("chain_check_s", check_s)
            Y = C[:, :_state["N_real"]].T.reshape(
                _state["Cout"], H, Wd).astype(np.float32)
            self.dev.calls[-1]["module"] = name
            tick("conv_npu_s", dev_s)
            tick("conv_host_s", (t1 - t0) + (time.perf_counter() - t2))
            return torch.from_numpy(Y[None])

        mod.forward = forward
        self.mods[name] = state

    # -- the two element-wise ops (T22-B) --------------------------------
    def _ops_plan(self, kind: str, total: int):
        """[chunk_n, ...] or None -- an element-wise op is chunk-exact.

        The single-launch image stops being dispatchable past ~2 M elements on
        the f32-out variant: measured 2026-10-20, N=4194304 run 0 completed in
        51.9 ms and run 1 came back ERT_CMD_STATE_TIMEOUT at 7019 ms, and
        N=8388608 timed out on run 0 -- that is the ERT watchdog, not the
        kernel.  A chunked launch set is bit-identical because the op is per
        element, so the big tensors are split over a supported image.
        """
        ops = image_index()["ops"]
        if total <= OPS_MAX_ELEMS and (kind, total) in ops:
            return [total]
        for n in sorted((n for (k, n) in ops
                         if k == kind and n <= OPS_MAX_ELEMS), reverse=True):
            if total % n == 0:
                return [n] * (total // n)
        return None

    def _npu_silu(self, x):
        """SiLU on the board; None means "no image for this length"."""
        import torch
        from ml_dtypes import bfloat16
        xn = x.detach().cpu().numpy()
        if xn.ndim != 4 or xn.shape[0] != 1:
            return None
        total = int(xn[0].size)
        plan = self._ops_plan("silu", total)
        if plan is None:
            return None
        xb = np.ascontiguousarray(xn[0].reshape(-1)).astype(bfloat16)
        out = []
        t_all = time.perf_counter()
        o = 0
        for n in plan:
            piece = np.ascontiguousarray(xb[o:o + n])
            y, _dt = self.dev.ops("silu", n,
                                  [piece.view(np.uint8)], n * 2)
            out.append(y)
            o += n
        tick("silu_npu_s", time.perf_counter() - t_all)
        y = out[0] if len(out) == 1 else np.concatenate(out)
        # writable copy: torch warns on read-only numpy (and an in-place op
        # downstream would corrupt it silently)
        return torch.from_numpy(np.array(y, copy=True).reshape(xn.shape))

    def _npu_add(self, x, h, allow_bf16: bool = True):
        """h + x on the board; the f32-output variant is the byte-exact one."""
        import torch
        from ml_dtypes import bfloat16
        xn = x.detach().cpu().numpy()
        hn = h.detach().cpu().numpy()
        if xn.ndim != 4 or xn.shape[0] != 1 or hn.shape != xn.shape:
            return None
        total = int(xn[0].size)
        ops = image_index()["ops"]
        a = np.ascontiguousarray(xn[0].reshape(-1)).astype(bfloat16)
        b = np.ascontiguousarray(hn[0].reshape(-1), dtype=np.float32)
        kind, scale = "vaddf", 4          # bytes per element of the OUTPUT
        plan = self._ops_plan("vaddf", total)
        if plan is None:
            if not allow_bf16:
                return None
            plan = self._ops_plan("vadd", total)
            kind = "vadd"
            if plan is None:
                return None
        out = []
        t_all = time.perf_counter()
        o = 0
        for n in plan:
            pa = np.ascontiguousarray(a[o:o + n])
            pb = np.ascontiguousarray(b[o:o + n])
            y, _dt = self.dev.ops(kind, n,
                                  [pa.view(np.uint8), pb.view(np.uint8)],
                                  n * scale, f32=(kind == "vaddf"))
            out.append(y)
            o += n
        tick("add_npu_s", time.perf_counter() - t_all)
        y = out[0] if len(out) == 1 else np.concatenate(out)
        if y.size != total:
            return None
        return torch.from_numpy(np.array(y, copy=True).reshape(xn.shape))

    # -- host stages (buckets only; the arithmetic stays torch fp32) ------
    def _patch_silu(self, name: str, mod) -> None:
        import torch

        def forward(x):
            t0 = time.perf_counter()
            if self.ops_silu:
                y = self._npu_silu(x)
                if y is not None:
                    self.dev.calls[-1].update(module=name)
                    return y
            y = torch.nn.functional.silu(x)
            tick("silu_s", time.perf_counter() - t0)
            return y

        mod.forward = forward

    def _trace(self, name: str, t) -> None:
        """Output hash per module, for the 'which op first diverges' bisect."""
        if os.environ.get("ZIMAGE_VAE_TRACE", "0") in ("1", "on", "yes"):
            arr = np.ascontiguousarray(t.detach().cpu().numpy(), np.float32)
            self.trace[name] = sha16(arr.tobytes())

    def _patch_resnet(self, name: str, mod) -> bool:
        """Route the resnet's final (h + x)/scale add through the board.

        diffusers' ResnetBlock2D.forward (resnet.py:320-373) for THIS VAE is
        the temb-free, upsample-free branch; anything else returns False and
        the caller keeps the bookkeeping-only wrapper.  With ops_npu off the
        replacement still has to be bit-identical to the original -- the
        gate runs that arm too.
        """
        import torch
        # time_embedding_norm is "group" for 12 of the 14 decoder resnets
        # (diffusers falls through to the same norm2 for it), "default" for the
        # mid-block pair; both take the SAME arithmetic once temb is None.
        if (getattr(mod, "upsample", None) is not None
                or getattr(mod, "downsample", None) is not None
                or getattr(mod, "time_emb_proj", None) is not None
                or getattr(mod, "time_embedding_norm", "default")
                not in ("default", "group")):
            return False
        if float(mod.output_scale_factor) != 1.0:
            return False
        scale = mod.output_scale_factor
        has_short = mod.conv_shortcut is not None
        skip_bf16 = name in getattr(self, "_add_skip", set())

        def forward(input_tensor, temb=None, *a, **kw):
            h = input_tensor
            xs = mod.norm1(h)
            xs = mod.nonlinearity(xs)
            xs = mod.conv1(xs)
            xs = mod.norm2(xs)
            xs = mod.nonlinearity(xs)
            xs = mod.dropout(xs)
            xs = mod.conv2(xs)
            if has_short:
                h = mod.conv_shortcut(h.contiguous())
            if self.ops_add:
                # allow_bf16=False for the resnet that feeds an attention
                # residual: it may only be routed when the f32-output image
                # exists (the BF16-storing one would round the residual).
                y = self._npu_add(xs, h, allow_bf16=not skip_bf16)
                if y is not None:
                    self.dev.calls[-1].update(module=name + ".add")
                    self._trace(name, y)
                    return y
            out = (h + xs) / scale
            self._trace(name, out)
            return out

        mod.forward = forward
        return True

    def _patch_container(self, name: str, mod, bucket: str) -> None:
        """Book ONLY the time the children did not account for."""
        orig = mod.forward

        def forward(x, *a, **kw):
            before = sum(TIMERS.values())
            t0 = time.perf_counter()
            y = orig(x, *a, **kw)
            dt = time.perf_counter() - t0
            child = sum(TIMERS.values()) - before
            tick(bucket, max(0.0, dt - child))
            return y

        mod.forward = forward

    # -- GroupNorm --------------------------------------------------------
    def _patch_gn(self, name: str, mod) -> None:
        import torch
        import vaeg_kernels as K_
        from ml_dtypes import bfloat16
        C, G = int(mod.num_channels), int(mod.num_groups)
        g_b = mod.weight.detach().cpu().numpy().astype(np.float32) \
            .astype(bfloat16)
        b_b = mod.bias.detach().cpu().numpy().astype(np.float32) \
            .astype(bfloat16)

        def forward(x, _C=C, _G=G, _g=g_b, _b=b_b):
            t0 = time.perf_counter()
            t = x.detach().cpu().numpy().astype(np.float32)
            shape = t.shape
            cs = np.ascontiguousarray(t.reshape(_C, -1))
            xb = cs.astype(bfloat16)
            S = cs.shape[1]
            hw = None
            if t.ndim == 4:
                hw = (int(t.shape[2]), int(t.shape[3]))
            t1 = time.perf_counter()
            y, dev_s = self.dev.groupnorm(np.ascontiguousarray(xb), _g, _b,
                                          _C, S, grade=hw)
            t2 = time.perf_counter()
            # per-call check, SAME 口径 as the 3b2 gate (verify_gn): the f64
            # ideal of the kernel's formula on the bf16 input, judged against
            # the MAGNITUDE BOUND (never against the result itself).
            t_c = time.perf_counter()
            ref = K_.group_norm_exact(
                np.asarray(xb, np.float64), np.asarray(_g, np.float64),
                np.asarray(_b, np.float64), groups=_G, eps=float(mod.eps))
            ref_b = ref.astype(bfloat16).astype(np.float64)
            mag = _gn_mag(xb, _g, _b, _G, float(mod.eps))
            ulp = float((np.abs(np.asarray(y, np.float64) - ref_b)
                         / np.maximum(_bf16_ulp(mag), 2.0 ** -126)).max())
            exact = float((np.asarray(y, np.float64) == ref_b).mean())
            self.dev.calls[-1].update(module=name, max_ulp_mag=ulp,
                                      bit_exact=exact,
                                      pass_=bool(ulp <= 2.0),
                                      ref_kind="f64 ideal of the kernel formula",
                                      check_s=time.perf_counter() - t_c)
            t_end = time.perf_counter()
            tick("chain_check_s", self.dev.calls[-1]["check_s"])
            tick("gn_npu_s", dev_s)
            tick("gn_host_s", (t1 - t0) + (t_end - t2)
                 - self.dev.calls[-1]["check_s"])
            return torch.from_numpy(np.asarray(y, np.float32).reshape(shape))

        mod.forward = forward

    # -- Upsample2D -------------------------------------------------------
    def _patch_up(self, name: str, mod) -> None:
        import torch
        from ml_dtypes import bfloat16
        conv = mod.conv

        def forward(x, output_size=None, _c=conv):
            t0 = time.perf_counter()
            t = x.detach().cpu().numpy().astype(np.float32)
            if t.ndim != 4 or t.shape[0] != 1:
                raise ValueError("chain decodes one image at a time")
            xb = t[0].astype(bfloat16)
            C, H, W = [int(v) for v in xb.shape]
            t1 = time.perf_counter()
            y, dev_s = self.dev.upsample(xb, C, H, W)
            t2 = time.perf_counter()
            t_c = time.perf_counter()
            ref = np.repeat(np.repeat(np.asarray(xb), 2, axis=1), 2, axis=2)
            same = bool(np.array_equal(np.asarray(y), ref))
            self.dev.calls[-1].update(module=name, bit_exact=bool(same),
                                      pass_=bool(same),
                                      check_s=time.perf_counter() - t_c)
            t_end = time.perf_counter()
            tick("chain_check_s", self.dev.calls[-1]["check_s"])
            tick("up_npu_s", dev_s)
            tick("up_host_s", (t1 - t0) + (t_end - t2)
                 - self.dev.calls[-1]["check_s"])
            return _c(torch.from_numpy(np.asarray(y, np.float32)[None]))

        mod.forward = forward


def _gn_mag(xb, g, b, groups, eps):
    """The magnitude bound of verify_gn, in the SAME dtype (float64).

    verify_gn computes it from the bf16 input cast to float64 and the f64
    gamma/beta; a float32 mean here would move |x-mu| near a cancellation and
    report a different ulp on elements whose arithmetic is identical.
    """
    xb = np.asarray(xb, np.float64)
    g = np.asarray(g, np.float64)
    b = np.asarray(b, np.float64)
    C, S = xb.shape
    cg = C // groups
    mag = np.empty_like(xb)
    for gi in range(groups):
        seg = xb[gi * cg:(gi + 1) * cg].reshape(-1)
        mu = float(seg.mean())
        var = float(((seg - mu) ** 2).mean())
        rstd = 1.0 / math.sqrt(var + eps)
        sl = slice(gi * cg, (gi + 1) * cg)
        mag[sl] = (np.abs(xb[sl] - mu) * rstd * np.abs(g[sl])[:, None]
                   + np.abs(b[sl])[:, None])
    return mag


# ---------------------------------------------------------------------------
# the reference (CPU VAE) and the NPU decode
# ---------------------------------------------------------------------------

_VAE = {}


def load_vae():
    import torch
    if "vae" not in _VAE:
        from diffusers import AutoencoderKL
        t0 = time.perf_counter()
        _VAE["vae"] = AutoencoderKL.from_pretrained(
            str(MODELS), subfolder="vae").eval()
        _VAE["load_s"] = time.perf_counter() - t0
        _VAE["torch"] = torch
    return _VAE["vae"], _VAE["torch"]


def latents_for(size_name: str, which: str = "rand0"):
    lh, lw = SIZE_LATENT[size_name]
    if which == "rand0":
        lat = np.random.default_rng(0).standard_normal(
            (1, 16, lh, lw)).astype(np.float32)
    else:
        raise ValueError(which)
    return lat / SCALING + SHIFT


def to_image(out) -> np.ndarray:
    img = ((np.asarray(out[0].numpy()).transpose(1, 2, 0) + 1.0) / 2.0)
    return np.clip(img, 0, 1).astype(np.float32)


def metrics(img, ref) -> dict:
    d = np.asarray(img, np.float64) - np.asarray(ref, np.float64)
    mse = float(np.mean(d ** 2))
    return dict(max_abs_err=float(np.abs(d).max()),
                max_abs_err_8bit=float(np.abs(d).max()) * 255,
                mean_abs_err=float(np.abs(d).mean()), mse=mse,
                psnr_db=float(10 * np.log10(1.0 / (mse + 1e-30))))


def capture_instances(dec, lat_v, torch):
    """Reference pass: fp32 CPU decode + the op inventory (hooks)."""
    inst, hs = [], []

    def ch(m, n, i, o):
        x = i[0]
        inst.append(dict(kind="conv", module=n,
                         Cin=int(m.weight.shape[1]),
                         Cout=int(m.weight.shape[0]),
                         k=int(m.weight.shape[2]),
                         M=int(x.shape[2] * x.shape[3])))

    def gh(m, n, i, o):
        x = i[0]
        inst.append(dict(kind="gn", module=n, C=int(m.num_channels),
                         S=int(math.prod(int(v) for v in x.shape[2:]))))

    def uh(m, n, i, o):
        x = i[0]
        inst.append(dict(kind="up", module=n, C=int(x.shape[1]),
                         H=int(x.shape[2]), W=int(x.shape[3])))

    for n, m in dec.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            hs.append(m.register_forward_hook(
                lambda mm, ii, oo, n=n: ch(mm, n, ii, oo)))
        elif isinstance(m, torch.nn.GroupNorm):
            hs.append(m.register_forward_hook(
                lambda mm, ii, oo, n=n: gh(mm, n, ii, oo)))
        elif m.__class__.__name__ == "Upsample2D":
            hs.append(m.register_forward_hook(
                lambda mm, ii, oo, n=n: uh(mm, n, ii, oo)))
    t0 = time.perf_counter()
    with torch.no_grad():
        out = dec(torch.from_numpy(lat_v))
    ref_s = time.perf_counter() - t0
    for h in hs:
        h.remove()
    return to_image(out), inst, ref_s


def decode_npu(dec, lat_v, chain: Chain, torch) -> tuple:
    t0 = time.perf_counter()
    with torch.no_grad():
        out = dec(torch.from_numpy(lat_v))
    dt = time.perf_counter() - t0
    return to_image(out), dt


# ---------------------------------------------------------------------------
# one size, end to end
# ---------------------------------------------------------------------------

def run_size(size: str, tag: str = "", repeat: int = 1, checks: bool = True,
             keep_calls: bool = True, seed_ref=True) -> dict:
    import torch
    vae, torch = load_vae()
    T3.mkdir(parents=True, exist_ok=True)
    lat_v = latents_for(size)
    dec = vae.decoder
    ref_img, inst, ref_s = capture_instances(dec, lat_v, torch)
    plan = plan_from_instances(inst)
    print("[%s] reference (CPU VAE fp32) %.2f s  image %s  %d ops"
          % (size, ref_s, ref_img.shape, len(inst)), flush=True)
    print("[%s] plan: conv %d (%d launches), gn %d, up %d"
          % (size, plan["conv"]["instances"], plan["conv"]["launches"],
             plan["gn"]["instances"], plan["up"]["instances"]), flush=True)

    results = dict(size=size, ts=_ts(), latent_sha16=sha16(lat_v.tobytes()),
                   reference=dict(seconds=ref_s, image=ref_img.shape,
                                  sha16=sha16(ref_img.tobytes())),
                   ops=dict(conv=sum(1 for i in inst if i["kind"] == "conv"),
                            gn=sum(1 for i in inst if i["kind"] == "gn"),
                            up=sum(1 for i in inst if i["kind"] == "up")),
                   plan=plan)
    runs = []
    for ri in range(repeat):
        dev = Dev(tag="%s_r%d" % (size, ri + 1))
        chain = Chain(dec, dev, checks=checks and ri == 0)
        t_patch = time.perf_counter()
        patch_info = chain.patch()
        patch_s = time.perf_counter() - t_patch
        TIMERS.clear()
        try:
            img, dec_s = decode_npu(dec, lat_v, chain, torch)
        finally:
            chain.unpatch()
        if getattr(chain, "trace", None):
            _write_json(T3 / ("%s_trace.json" % size), chain.trace)
        buckets = {k: float(v) for k, v in TIMERS.items()}
        dev.close()
        rec = dict(run=ri + 1, patch_s=patch_s, decode_s=dec_s,
                   image_sha16=sha16(img.tobytes()),
                   image=img, buckets=buckets,
                   bucket_sum=float(sum(buckets.values())),
                   unaccounted=float(dec_s - sum(buckets.values())),
                   patch_info=patch_info,
                   calls=[dict(c) for c in dev.calls])
        runs.append(rec)
        print("[%s] run %d: %.2f s  buckets %s  unaccounted %.2f s"
              % (size, ri + 1, dec_s,
                 " ".join("%s=%.1f" % (k, v) for k, v in sorted(buckets.items())),
                 rec["unaccounted"]), flush=True)
        _write_json(T3 / ("%s_run%d.json" % (size, ri + 1)),
                    {k: v for k, v in rec.items() if k != "image"})
    r0 = runs[0]
    m = metrics(r0["image"], ref_img)
    results["decode"] = dict(
        seconds=r0["decode_s"], buckets=r0["buckets"],
        bucket_sum=r0["bucket_sum"], unaccounted_s=r0["unaccounted"],
        calls=len(r0["calls"]), patch=r0["patch_info"],
        device_s=float(sum(c["dev_s"] for c in r0["calls"])),
        host_s=float(sum(c.get("host_s", 0.0) for c in r0["calls"])))
    results["metrics"] = m
    results["repeat"] = [dict(run=r["run"], image_sha16=r["image_sha16"],
                              decode_s=r["decode_s"]) for r in runs]
    if len(runs) > 1:
        same = all(r["image_sha16"] == r0["image_sha16"] for r in runs[1:])
        same_b = all(np.array_equal(r["image"].view(np.uint32),
                                    r0["image"].view(np.uint32))
                     for r in runs[1:])
        results["run_to_run"] = dict(runs=len(runs), sha_equal=bool(same),
                                     bytes_equal=bool(same_b))
    # per-call verdicts
    conv_calls = [c for c in r0["calls"] if c["kind"] == "conv"]
    gn_calls = [c for c in r0["calls"] if c["kind"] == "gn"]
    up_calls = [c for c in r0["calls"] if c["kind"] == "up"]
    results["gates"] = dict(
        conv_calls=len(conv_calls),
        conv_launches=sum(c["blocks"] * c["halves"] for c in conv_calls),
        conv_gmac=float(sum(c["gmac"] for c in conv_calls)),
        conv_samples=sum(1 for c in conv_calls if c.get("check")),
        conv_sample_pass=all(c["check"]["sample_pass"]
                             for c in conv_calls if c.get("check")),
        conv_sample_bit_exact=min(
            [c["check"]["sample_bit_exact"] for c in conv_calls
             if c.get("check")] or [1.0]),
        gn_calls=len(gn_calls),
        gn_pass=all(c.get("pass_") for c in gn_calls),
        gn_max_ulp=max([c.get("max_ulp_mag", 0.0) for c in gn_calls] or [0.0]),
        gn_over2=sum(1 for c in gn_calls if not c.get("pass_")),
        gn_worst=(max([c for c in gn_calls] or [{}],
                      key=lambda c: c.get("max_ulp_mag") or 0.0).get("stem"),
                  max([c for c in gn_calls] or [{}],
                      key=lambda c: c.get("max_ulp_mag") or 0.0).get("module")),
        gn_bit_exact_min=min([c.get("bit_exact", 1.0) for c in gn_calls]
                             or [1.0]),
        up_calls=len(up_calls),
        up_pass=all(c.get("pass_") for c in up_calls),
    )
    np.save(T3 / ("%s_npu.npy" % size), r0["image"])
    np.save(T3 / ("%s_ref.npy" % size), ref_img)
    try:
        from PIL import Image
        for nm, arr in (("npu", r0["image"]), ("ref", ref_img)):
            Image.fromarray((arr * 255 + 0.5).astype(np.uint8)).save(
                T3 / ("%s_%s.png" % (size, nm)))
    except Exception as exc:  # noqa: BLE001
        print("[%s] PNG skipped: %s" % (size, exc))
    if keep_calls:
        _write_json(T3 / ("%s_calls.json" % size),
                    dict(size=size, ts=_ts(), calls=r0["calls"]))
    _write_json(T3 / ("%s_report.json" % size),
                {k: v for k, v in results.items()})
    worst = max([c for c in gn_calls] or [dict(max_ulp_mag=0.0)],
                key=lambda c: c.get("max_ulp_mag") or 0.0)
    print("[%s] PSNR %.4f dB  max|d| %.6f  calls %d  (conv launches %d)  "
          "worst GN %s %s ulp=%.3f"
          % (size, m["psnr_db"], m["max_abs_err"], len(r0["calls"]),
             results["gates"]["conv_launches"], worst.get("stem"),
             worst.get("module"), worst.get("max_ulp_mag") or 0.0),
          flush=True)
    return results


def plan_from_instances(inst: list) -> dict:
    """The executor's paper bill for one size (from the op inventory)."""
    conv = dict(instances=0, launches=0, gmac=0.0, exact=0, blocks=0,
                halved=0, stems={})
    gn = dict(instances=0, stems={})
    up = dict(instances=0, stems={})
    for it in inst:
        if it["kind"] == "conv":
            K, N, _ = conv_mkn(it["Cin"], it["Cout"], it["k"])
            img = select_conv_image(it["M"], K, N)
            blocks = (it["M"] + (img["m_block"] or it["M"]) - 1) // \
                (img["m_block"] or it["M"])
            conv["instances"] += 1
            conv["launches"] += blocks * img["halves"]
            conv["gmac"] += float(blocks * (img["m_block"] or it["M"]) * K
                                  * img["n_xcl"] / 1e9 * img["halves"])
            conv["exact"] += int(img["m_block"] is None)
            conv["blocks"] += blocks
            conv["halved"] += int(img["halves"] > 1)
            conv["stems"][img["stem"]] = conv["stems"].get(img["stem"], 0) + 1
        elif it["kind"] == "gn":
            stem, hw = gn_stem_for(it["C"], it["S"])
            gn["instances"] += 1
            gn["stems"][stem] = gn["stems"].get(stem, 0) + 1
        else:
            stem, why = up_stem_for(it["C"], it["H"], it["W"])
            up["instances"] += 1
            up["stems"][stem] = up["stems"].get(stem, 0) + 1
    conv["est_device_s"] = conv["gmac"] / 1.35 + conv["launches"] * 0.30
    gn["est_device_s"] = gn["instances"] * 0.32
    up["est_device_s"] = up["instances"] * 0.32
    return dict(conv=conv, gn=gn, up=up)


# ---------------------------------------------------------------------------
# the 64x64 self-test (3b1 regression)
# ---------------------------------------------------------------------------

def run_selfref(tag: str = "", conv_impl: str = "npu",
                gnup: bool = True) -> dict:
    """The 3b1 64x64 chain, now with the conv on the NPU.

    The 3b1 device2_measured tier was: device-exact conv ON THE HOST + real
    NPU GN/UP -> 37.3189 dB.  Its conv model is bit-identical to the gated
    NPU conv rows (max_ulp 0.0 / bit_exact 1.0 on all 92 rows), so running
    the same decode with the conv on the NPU must reproduce that image
    BIT FOR BIT.  That is the regression this mode checks.
    """
    import torch
    vae, torch = load_vae()
    T21 = OUT / "t21_3b1"
    lat_v = np.load(T21 / "latent.npy")
    ref_img = np.load(T21 / "ref_img.npy")
    want = np.load(T21 / "img_device2_measured_run1.npy")
    dec = vae.decoder
    _, inst, _ = capture_instances(dec, lat_v, torch)
    plan = plan_from_instances(inst)
    print("[selfref] 64x64: %d conv instances, %d launches, %.1f GMAC; "
          "gn %d, up %d" % (plan["conv"]["instances"],
                            plan["conv"]["launches"], plan["conv"]["gmac"],
                            plan["gn"]["instances"], plan["up"]["instances"]),
          flush=True)
    dev = Dev(tag="selfref")
    chain = Chain(dec, dev, checks=(conv_impl == "npu"), full_check=True,
                  conv_impl=conv_impl)
    chain.patch(gnup=gnup)
    TIMERS.clear()
    try:
        img, dec_s = decode_npu(dec, lat_v, chain, torch)
    finally:
        chain.unpatch()
    buckets = {k: float(v) for k, v in TIMERS.items()}
    calls = [dict(c) for c in dev.calls]
    dev.close()
    m = metrics(img, ref_img)
    np.save(T3 / "selfref64_npu.npy", img)
    eq = bool(np.array_equal(img.view(np.uint32), want.view(np.uint32)))
    against = {}
    for nm in ("img_fp32_fp32", "img_baseline_baseline", "img_baseline_ba",
               "img_device2_measured_run1", "img_device2_measured_run2",
               "img_device2_measured_run3"):
        p = T21 / (nm + ".npy")
        if not p.is_file():
            continue
        ref = np.load(p)
        mm = metrics(img, ref)
        against[nm] = dict(
            psnr_db=mm["psnr_db"], max_abs_err=mm["max_abs_err"],
            n_diff=int((img.view(np.uint32) !=
                        np.asarray(ref, np.float32).view(np.uint32)).sum()),
            bit_identical=bool(np.array_equal(
                img.view(np.uint32),
                np.asarray(ref, np.float32).view(np.uint32))))
    print("[selfref] vs 3b1 tiers: " + "  ".join(
        "%s=%.4fdB/diff%d" % (k.replace("img_", ""), v["psnr_db"], v["n_diff"])
        for k, v in against.items()), flush=True)
    res = dict(selfref="64x64", conv_impl=conv_impl, gnup=gnup, ts=_ts(),
               decode_s=dec_s, metrics=m,
               psnr_3b1=37.31888633548063,
               image_sha16=sha16(img.tobytes()),
               want_sha16=sha16(np.asarray(want, np.float32).tobytes()),
               bit_identical_to_3b1=eq, against_3b1_tiers=against,
               buckets=buckets,
               bucket_sum=float(sum(buckets.values())),
               unaccounted=float(dec_s - sum(buckets.values())),
               calls=len(calls),
               conv_launches=sum(c["blocks"] * c["halves"]
                                 for c in calls if c["kind"] == "conv"),
               plan=plan,
               gates=dict(
                   conv_samples=sum(1 for c in calls if c.get("check")),
                   conv_sample_pass=all(c["check"]["sample_pass"]
                                        for c in calls if c.get("check")),
                   gn_pass=all(c.get("pass_") for c in calls
                               if c["kind"] == "gn"),
                   gn_max_ulp=max([c.get("max_ulp_mag", 0.0) for c in calls
                                   if c["kind"] == "gn"] or [0.0]),
                   up_pass=all(c.get("pass_") for c in calls
                               if c["kind"] == "up")),
               )
    suffix = ("_npu" if conv_impl == "npu" else "_model") + (
        "" if gnup else "_hostgnup")
    _write_json(T3 / ("selfref64%s.json" % suffix), res)
    _write_json(T3 / ("selfref64%s_calls.json" % suffix), dict(calls=calls))
    print("[selfref] PSNR %.4f dB (3b1 reports 37.3189)  bit-identical to the "
          "3b1 image: %s  calls %d  unaccounted %.2f s"
          % (m["psnr_db"], eq, len(calls), res["unaccounted"]), flush=True)
    return res


# ---------------------------------------------------------------------------
# the 3 extra UP images 64x256 needs (second (H,W) factorization of (C,S))
# ---------------------------------------------------------------------------

UP_EXTRA = [(512, 8, 32), (512, 16, 64), (256, 32, 128)]


def build_up_extra(size: str = "64x256") -> dict:
    import t21_3b2_plan as P
    # fresh compile cache + the 0o700 -> 0o777 mkdir shim (the poisoned .prj
    # lesson from T21-3B2VF section 7.7), PEANO before any aie import.
    P._install_mkdir_mode_shim()
    os.environ["PEANO_INSTALL_DIR"] = str(ROOT / "_pylibs" / "llvm-aie"
                                          / "llvm-aie")
    os.environ["NPU_CACHE_HOME"] = str(T3 / "_npucache_buildup")
    os.environ["NPU_CACHE_DIR"] = os.environ["NPU_CACHE_HOME"]
    P._setup_aie()
    import vaeg_design as D
    out = {}
    vae, torch = load_vae()
    for (C, H, W) in UP_EXTRA:
        stem = "upw_C%d_H%d_W%d" % (C, H, W)
        xclbin = ART / (stem + ".xclbin")
        insts = ART / (stem + ".bin")
        rec = dict(stem=stem, C=C, H=H, W=W)
        if not xclbin.is_file():
            spec = D.make_up_kernel(C, H, W).specialize(use_cache=False)
            t0 = time.perf_counter()
            spec.compile(xclbin_path=str(xclbin), inst_path=str(insts))
            rec["compile_s"] = time.perf_counter() - t0
        # gate it with the 3b2 harness: prep from the real activation, one
        # device run, verify_up (bit-exact replicate + C*H*W*8 length)
        want = _up_module_for(vae.decoder, latents_for(size), torch, C, H, W)
        item = dict(kind="up", size=size, module=want, name=stem,
                    params=dict(C=C, H=H, W=W, S=H * W), id=None)
        P.prep_up(item)
        t0 = time.perf_counter()
        dev = P.run_device(item, 1, stem)
        rec["device_s"] = time.perf_counter() - t0
        rec["device"] = dev
        res = P.VERIFIERS["up"](P.paths_of(item)["io"], P.paths_of(item)["raw"])
        rec.update(up_gate=res)
        _write_json(T3 / (stem + ".json"), rec)
        _write_json(ART / (stem + ".build.json"),
                    dict(stem=stem, kind="up",
                         spec=dict(C=C, H=H, W=W, why="second (H,W) of (C,S)"),
                         ts=_ts()))
        P._rm(P.paths_of(item)["io"])
        P._rm(P.paths_of(item)["raw"])
        print("[up-extra] %s: pass=%s bit_exact=%s device %.2f s"
              % (stem, res["pass_"], res["bit_exact"], rec["device_s"]),
              flush=True)
        out[stem] = rec
    return out


# The 64x64 self-test needs two conv geometries that no product size owns
# (the six sizes' M values are 288..65536; 64x64's last block is M=4096 at
# Cin=128/Cout=128 and the 256->128 1x1 shortcut).  Everything else the 64x64
# chain needs is already covered by the 92 rows or by 3b1's five images.
CONV_EXTRA = [(4096, 128, 128, 3), (4096, 256, 128, 1)]


def build_conv_extra() -> dict:
    import t21_3b2_plan as P
    P._install_mkdir_mode_shim()
    os.environ["PEANO_INSTALL_DIR"] = str(ROOT / "_pylibs" / "llvm-aie"
                                          / "llvm-aie")
    os.environ["NPU_CACHE_HOME"] = str(T3 / "_npucache_buildconv")
    os.environ["NPU_CACHE_DIR"] = os.environ["NPU_CACHE_HOME"]
    P._setup_aie()
    import vae_npu as N_
    out = {}
    for (M, Cin, Cout, k) in CONV_EXTRA:
        K, N, k_real = conv_mkn(Cin, Cout, k)
        stem = "cg_M%d_K%d_N%d" % (M, K, N)
        xclbin, insts = ART / (stem + ".xclbin"), ART / (stem + ".bin")
        rec = dict(stem=stem, M=M, Cin=Cin, Cout=Cout, k=k, K=K, N=N)
        if not xclbin.is_file():
            spec = N_.conv_gemm.specialize(
                M=M, K=K, N=N, m=TILE["m"], k=TILE["k"], n=TILE["n"],
                use_cache=False)
            t0 = time.perf_counter()
            spec.compile(xclbin_path=str(xclbin), inst_path=str(insts))
            rec["compile_s"] = time.perf_counter() - t0
        _write_json(ART / (stem + ".build.json"),
                    dict(stem=stem, kind="conv", ts=_ts(),
                         spec=dict(M=M, K=K, N=N, Cin=Cin, Cout=Cout, k=k,
                                   why="64x64 self-test geometry")))
        _write_json(T3 / (stem + ".json"), rec)
        print("[conv-extra] %s compiled %s s" % (stem, rec.get("compile_s")),
              flush=True)
        out[stem] = rec
    return out


def _up_module_for(dec, lat_v, torch, C: int, H: int, W: int) -> str:
    """Which Upsample2D instance has an input plane (C,H,W)?

    Channel count alone is ambiguous (up_blocks.0 and .1 are both C=512), so
    the module is identified by running the size's decoder once with hooks on
    every Upsample2D and matching the recorded input shape.
    """
    seen, hs = [], []
    for n, m in dec.named_modules():
        if m.__class__.__name__ == "Upsample2D":
            hs.append(m.register_forward_hook(
                lambda mm, ii, oo, n=n: seen.append(
                    (n, tuple(int(v) for v in ii[0].shape[1:])))))
    with torch.no_grad():
        dec(torch.from_numpy(lat_v))
    for h in hs:
        h.remove()
    for n, shp in seen:
        if shp == (C, H, W):
            return n
    raise KeyError("no Upsample2D with input %s; saw %s" % ((C, H, W), seen))


# ---------------------------------------------------------------------------
# the WebUI entry point: a resident patched decoder, one image per call
# ---------------------------------------------------------------------------

LATENT_TO_SIZE = {(lh, lw): s for s, lh, lw in SIZES}
_WEB = {}


def npu_decoder(size: str) -> dict:
    """(build once) the patched decoder + its device handle for one size."""
    if size not in _WEB:
        vae, torch = load_vae()
        dec = vae.decoder
        dev = Dev(tag="webui_" + size)
        chain = Chain(dec, dev, checks=False)
        info = chain.patch()
        _WEB[size] = dict(dec=dec, dev=dev, chain=chain, patch=info,
                          size=size, vae=vae,
                          scaling=float(vae.config.scaling_factor),
                          shift=float(vae.config.shift_factor))
    return _WEB[size]


def decode_latent(lat_v, scaled: bool = False, checks: bool = False) -> dict:
    """One latent (1,16,lh,lw) -> HWC float32 image in [0,1], via the NPU.

    Raises for any shape that is not one of the six product sizes: the caller
    (webui/vae_local.py) then keeps its CPU path.  No fallback is hidden here
    -- a device failure is an exception, and the exception is what triggers
    the CPU path upstream.
    """
    import torch
    lat_v = np.asarray(lat_v, np.float32)
    if lat_v.ndim == 3:
        lat_v = lat_v[None]
    if lat_v.ndim != 4:
        raise ValueError("latent shape %s" % (lat_v.shape,))
    key = (int(lat_v.shape[-2]), int(lat_v.shape[-1]))
    size = LATENT_TO_SIZE.get(key)
    if size is None:
        raise ValueError("no NPU VAE geometry for latent %dx%d" % key)
    t_build = time.perf_counter()
    d = npu_decoder(size)
    build_s = time.perf_counter() - t_build
    if not scaled:
        # the pipeline's saved latents are RAW: decode() divides by the
        # config scaling factor and adds the shift (vae_decode.decode)
        lat_v = np.ascontiguousarray(lat_v / d["scaling"] + d["shift"],
                                     dtype=np.float32)
    dev, dec, chain = d["dev"], d["dec"], d["chain"]
    n0 = len(dev.calls)
    TIMERS.clear()
    img, dt = decode_npu(dec, lat_v, chain, torch)
    calls = [dict(c) for c in dev.calls[n0:]]
    return dict(image=img, decode_s=dt, size=size, build_s=build_s,
                buckets={k: float(v) for k, v in TIMERS.items()},
                calls=calls, n_calls=len(calls),
                conv_launches=sum(c["blocks"] * c["halves"] for c in calls
                                  if c["kind"] == "conv"),
                device_s=float(sum(c["dev_s"] for c in calls)),
                impl="npu")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_index() -> int:
    idx = image_index()
    print("conv exact  : %d" % len(idx["conv_exact"]))
    print("conv M-block: %d  %s" % (len(idx["conv_mb"]),
                                    sorted("%s=%s" % (k, v[0]) for k, v in
                                           idx["conv_mb"].items())))
    print("gn          : %d  %s" % (len(idx["gn"]), sorted(idx["gn"])))
    print("up (C,H,W)  : %s" % sorted(k for k in idx["up"] if k[0] != "S"))
    print("up (C,S)    : %s" % sorted(k for k in idx["up"] if k[0] == "S"))
    return 0


def cmd_plan(sizes) -> int:
    import torch
    vae, torch = load_vae()
    tot = dict(launches=0, gmac=0.0, conv=0, gn=0, up=0, est=0.0)
    rows = []
    for size in sizes:
        lat_v = latents_for(size)
        _, inst, ref_s = capture_instances(vae.decoder, lat_v, torch)
        plan = plan_from_instances(inst)
        est = (plan["conv"]["est_device_s"] + plan["gn"]["est_device_s"]
               + plan["up"]["est_device_s"])
        rows.append(dict(size=size, ref_s=ref_s, **plan, est_device_s=est))
        print("%-8s conv %2d inst / %4d launches / %6.1f GMAC ; gn %2d ; up %d "
              "; est device %5.1f min"
              % (size, plan["conv"]["instances"], plan["conv"]["launches"],
                 plan["conv"]["gmac"], plan["gn"]["instances"],
                 plan["up"]["instances"], est / 60.0), flush=True)
        tot["launches"] += plan["conv"]["launches"]
        tot["gmac"] += plan["conv"]["gmac"]
        tot["conv"] += plan["conv"]["instances"]
        tot["gn"] += plan["gn"]["instances"]
        tot["up"] += plan["up"]["instances"]
        tot["est"] += est
    print("TOTAL conv %d / launches %d / %.1f GMAC / gn %d / up %d / est %.1f min"
          % (tot["conv"], tot["launches"], tot["gmac"], tot["gn"], tot["up"],
             tot["est"] / 60.0))
    _write_json(T3 / "plan.json", dict(ts=_ts(), rows=rows, total=tot))
    return 0


def main() -> int:
    _force_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--size", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--selfref", action="store_true")
    ap.add_argument("--build-up", action="store_true")
    ap.add_argument("--build-conv", action="store_true")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--conv-impl", default="npu", choices=["npu", "model"])
    ap.add_argument("--no-gnup", action="store_true",
                    help="leave GroupNorm/upsample on the host (the 3b1 "
                         "baseline tier)")
    ap.add_argument("--no-checks", action="store_true")
    ap.add_argument("--tag", default="")
    # T22-A: launch architecture (see Dev.__init__).  proc = the 3b3 delivery
    # path (one py310 process per launch), srv = one device server per host
    # process.  The env var is what webui/vae_local picks up.
    ap.add_argument("--launch", default=None, choices=["srv", "proc"])
    ap.add_argument("--opnpu", default=None,
                    choices=["1", "silu", "add"],
                    help="T22-B: SiLU and/or the residual add on the NPU "
                         "(default off: host fp32, the 3b3 path)")
    ap.add_argument("--out", default=None,
                    help="artifact dir (default out/t21_3b3; T22 runs use "
                         "out/t22_vae so the 3b3 evidence stays frozen)")
    args = ap.parse_args()
    if args.launch:
        os.environ["ZIMAGE_VAE_LAUNCH"] = args.launch
    if args.opnpu:
        os.environ["ZIMAGE_VAE_OPNPU"] = args.opnpu
    if args.out:
        global T3
        T3 = pathlib.Path(args.out).resolve()
    T3.mkdir(parents=True, exist_ok=True)
    if args.index:
        return cmd_index()
    if args.build_up:
        build_up_extra()
        return 0
    if args.build_conv:
        build_conv_extra()
        return 0
    if args.plan:
        return cmd_plan([s for s, _, _ in SIZES])
    if args.selfref:
        run_selfref(args.tag, conv_impl=args.conv_impl, gnup=not args.no_gnup)
        return 0
    if args.size:
        run_size(args.size, tag=args.tag, repeat=args.repeat,
                 checks=not args.no_checks)
        return 0
    if args.all:
        out = {}
        for s, _, _ in SIZES:
            out[s] = run_size(s, tag=args.tag, repeat=args.repeat,
                              checks=not args.no_checks)
        _write_json(T3 / "six_sizes.json", out)
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
