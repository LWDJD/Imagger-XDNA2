# SPDX-License-Identifier: MIT
# step3_run.py -*- Python -*-
#
# STEP4: working copy of step3_run.py (stage 3, FFN + second residual).
# Only the file name differs from the record; OUT stays out/ so the
# already-built step3 artifacts are reused rather than rebuilt.
#
# STEP3 harness: compile + run + verify imaggers/block/step3_ffn.py (the FFN
# branch + second residual at K=3840 / ffn_inner 10240 / M=96) against
# imaggers/oracle/dit_block_float -- "the truth" per STAGE3.md judging --
# and against an exact device model.
#
# The device computes per token row t (x = the post-attention residual
# stream, i.e. the oracle's x_mid rounded to bf16):
#
#     N2[t] = rms_norm_weighted(x[t], gamma2)     y = x/rms(x)*g
#     Y2[t] = adaln_modulate(N2[t], s2[t], b2[t]) y = x*(1+s)+b
#     gate  = Y2 @ W1.T   up = Y2 @ W3.T          mm_bfp(mixed), k=8 calls
#     sg    = up * silu(gate)                     swiglu(use_lut=True)
#     ff    = sg @ W2.T                           mm_bfp(mixed), k=8 calls
#     out   = x + ff                              add_sized
#
# JUDGING (STAGE3.md): rel_to_absmax <= 5.4e-2 of imaggers/oracle/
# dit_block_float's x_out, raised by +0.08 unless swiglu uses the LUT
# build.  This design builds swiglu(use_lut=True), so the budget is the
# plain 5.4e-2 and the vtanh envelope is not spent.
#
# Interpreter split (forced by pyxrt.pyd linking python310.dll):
#   static checks + compile + data prep + verify : Python 3.12 (mlir_aie)
#   device run                                   : Python 3.10 (pyxrt)
# The SAME file serves both (step1_run.py's proven plumbing).
#
# Usage (Python 3.12):
#   python step3_run.py [-M 96] [--seed 0] [--runs 2] [--skip-compile]
#                       [--skip-run] [--verify-only] [--smoke]
from __future__ import annotations

import argparse
import importlib.util
import os
import pathlib
import re
import subprocess
import sys
import time

import numpy as np

# --- environment MUST be set before any `aie` import -------------------------
HERE = pathlib.Path(__file__).resolve().parent          # ...\imaggers\block
NPU_ROOT = HERE.parent.parent                           # ...\zimage-bfp16-npu
ROOT = NPU_ROOT.parent                                  # ...\public_work
PYLIBS = ROOT / "_pylibs"
SCRATCH = NPU_ROOT / "imaggers"
sys.path.append(str(HERE))
sys.path.append(str(PYLIBS))
sys.path.append(str(PYLIBS / "mlir_aie" / "python"))
sys.path.append(str(SCRATCH / "kernels"))   # LAST: kernels/refs.py must win
os.environ["PEANO_INSTALL_DIR"] = str(PYLIBS / "llvm-aie" / "llvm-aie")
os.environ["NPU_CACHE_HOME"] = str(HERE / "_npu_cache_s3")

OUT = HERE / "out"   # STEP4: block step3 artifacts reuse out/ (the record dir)
PY310 = pathlib.Path(r"D:\Users\15081\AppData\Local\GAIA\python\python.exe")
XRT_DIR = pathlib.Path(
    r"D:\Project\deepseek_harness\test7\WStmp\driver376\npu_mcdm_stack_prod"
)
KERNEL = "MLIR_AIE"
EPS = 1e-5

# fixed geometry of the stage-3 chain (STAGE3.md census)
K = 3840          # hidden
N1 = 10240        # ffn inner
M_DEF = 96        # 32 context + 64 image tokens (zimage_pipeline.py:95-114)
M_TILE = 16
K_MICRO = 8
N_TILE = 64       # gate/up and down C tiles (m*n == 1024 == swiglu tile)
G_GU = 4          # gate/up workers
G_G3 = 6          # down-GEMM workers (and add workers)
L_G3 = 10         # C3 tiles held per down worker (6*10 == K/n3 == 60)
LEVEL = 3         # 1 = through join, 2 = + down-GEMM, 3 = full (bisect knob)


def _force_utf8() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass


def _install_mkdir_mode_shim() -> None:
    """Neutralize mode 0o700 for directories this process creates (proven in
    imaggers/dit_block_v1_run.py): `os.mkdir(path, 0o700)` creates a
    directory this token cannot re-enter and poisons the object cache."""
    orig_mkdir = os.mkdir

    def mkdir_compat(path, mode=0o777, *, dir_fd=None):
        if mode == 0o700:
            mode = 0o777
        return orig_mkdir(path, mode, dir_fd=dir_fd)

    os.mkdir = mkdir_compat


def _paths(M, level=None):
    level = LEVEL if level is None else level
    stem = f"step3_M{M}_K{K}_N1{N1}_m{M_TILE}_k{K_MICRO}_n{N_TILE}_G{G_GU}x{G_G3}L{L_G3}" + (
        "" if level == 3 else f"_L{level}"
    )
    return (
        OUT / f"{stem}.xclbin",
        OUT / f"{stem}.bin",
        OUT / f"{stem}_io",
        OUT / f"{stem}_Y_raw.npy",
        stem,
    )


# --------------------------------------------------------------------------
# host packing helpers (the BG/B3/X host buffers are PREPACKED in device
# stream order so every fill tap is a trivial object tiling)
# --------------------------------------------------------------------------


def pack_b_blocks(W, K, N, k, n) -> bytes:
    """Encode W (K, N) as bfp16ebs8 blocks packed in (jj, kk, row) order:
    flat block (jj*K_div_k + kk)*n + r holds W^T row jj*n+r, hidden block
    kk.  Copied verbatim from memtile2/run.py via step1_run.py (proven);
    the codec is aie.utils.bfp.encode's default `floor`, which is what the
    oracle's DeviceCfg.b_bfp_rounding models conservatively."""
    from aie.utils import bfp as B

    assert k == 8, f"pack_b_blocks assumes k==8, got {k}"
    WT = np.ascontiguousarray(np.asarray(W, np.float32).T.copy())
    enc = B.encode(WT.reshape(-1))
    enc = (
        np.asarray(enc, dtype=np.uint8)
        if isinstance(enc, np.ndarray)
        else np.frombuffer(bytes(enc), np.uint8)
    )
    blocks = enc.reshape(N, K // 8, 9)
    packed = (
        blocks.reshape(N // n, n, K // k, k // 8, 9)
        .transpose(0, 2, 1, 3, 4)
        .reshape(-1)
    )
    return np.ascontiguousarray(packed).tobytes()


def pack_bg_stream(W_gate, W_up) -> np.ndarray:
    """Combined gate/up B stream: per (sweep s, kk, gate/up worker w) one
    [B_gate | B_up] object (2 * n*9 bytes), worker w's n-tile jj = w*sweeps+s.
    Returns uint8 blocks (one v8bfp16ebs8 == 9 bytes flattened to u8)."""
    k, n = K_MICRO, N_TILE
    sweeps = (N1 // n) // G_GU
    # GEMM convention: gate = Y2 @ W_gate.T with W_gate torch (N1, K), so
    # pack_b_blocks wants W (K, N) = W_gate.T.
    gg = np.frombuffer(pack_b_blocks(W_gate.T, K, N1, k, n), np.uint8).reshape(
        N1 // n, K // k, n * 9
    )
    uu = np.frombuffer(pack_b_blocks(W_up.T, K, N1, k, n), np.uint8).reshape(
        N1 // n, K // k, n * 9
    )
    s = np.arange(sweeps)[:, None, None]
    kk = np.arange(K // k)[None, :, None]
    w = np.arange(G_GU)[None, None, :]
    jj = w * sweeps + s
    out = np.empty((sweeps, K // k, G_GU, 2, n * 9), np.uint8)
    out[..., 0, :] = gg[jj, kk]
    out[..., 1, :] = uu[jj, kk]
    return np.ascontiguousarray(out.reshape(-1))


def pack_b3_stream(W_down) -> np.ndarray:
    """Combined down-GEMM B stream: per (sweep s, gate/up worker w, slice s2
    of the chunk, down worker g, held tile l) one B block of n3*9 bytes.
    The A slice (s, w, s2) is inner block kk' = 320*w + 8*s + s2 of the
    10240 inner dim; down worker g holds n3-tiles {L_G3*g + l}."""
    k, n3 = K_MICRO, N_TILE
    sweeps = (N1 // N_TILE) // G_GU
    Wt = np.ascontiguousarray(np.asarray(W_down, np.float32))  # (K_out, N1)
    dd = np.frombuffer(pack_b_blocks(Wt.T, N1, K, k, n3), np.uint8).reshape(
        K // n3, N1 // k, n3 * 9
    )   # (jj3, kk')
    s = np.arange(sweeps)[:, None, None, None, None]
    w = np.arange(G_GU)[None, :, None, None, None]
    s2 = np.arange(n3 // k)[None, None, :, None, None]
    g = np.arange(G_G3)[None, None, None, :, None]
    l = np.arange(L_G3)[None, None, None, None, :]
    kk_p = 320 * w + 8 * s + s2
    jj3 = L_G3 * g + l
    out = np.empty((sweeps, G_GU, n3 // k, G_G3, L_G3, n3 * 9), np.uint8)
    out[:] = dd[jj3, kk_p]
    return np.ascontiguousarray(out.reshape(-1))


def pack_x_stream(x_bf16) -> np.ndarray:
    """Combined x stream for the add workers: per (row-tile i, held tile l)
    one [g0|g1|...|g5] object; region g is x's (m, n3) tile at output
    column (L_G3*g + l)*n3, row-major (the layout the C3 hop emits)."""
    m, n3 = M_TILE, N_TILE
    row_tiles = x_bf16.shape[0] // m
    Xt = np.ascontiguousarray(x_bf16, np.float32).reshape(
        row_tiles, m, K // n3, n3
    )
    idx = (np.arange(G_G3)[None, :] * L_G3 + np.arange(L_G3)[:, None]).reshape(-1)
    sel = Xt[:, :, idx, :]                       # (row_tiles, m, 60, n3) in (l, g)
    out = sel.transpose(0, 2, 1, 3).reshape(row_tiles, L_G3, G_G3, m * n3)
    return np.ascontiguousarray(out.reshape(-1))


# --------------------------------------------------------------------------
# static checks (no compile, no device)
# --------------------------------------------------------------------------


def static_checks(M) -> int:
    import step3_ffn as D

    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"[static] {name}: {'OK' if cond else 'FAIL'}")
        ok = ok and bool(cond)

    m, k, n = M_TILE, K_MICRO, N_TILE
    check("m*n == 1024 (swiglu fixed tile)", m * n == 1024)
    check("K % 32 == 0 (64 B row views)", K % 32 == 0)
    check("n % 16 == 0 and n3 % 16 == 0 (mm_bfp_mixed)",
          n % 16 == 0 and N_TILE % 16 == 0)
    s_off, b_off, g_off, elem = D.sbg_row_layout(K)
    check("sbg splits 64 B aligned",
          s_off % 64 == 0 and b_off % 64 == 0 and g_off % 64 == 0)
    Kc = D.chunk_cols(K)
    check("Kc divides K, 32-aligned, BD under cap",
          K % Kc == 0 and Kc % 32 == 0 and 8 * Kc <= 16383)

    # swiglu-out walk: blocked (m,n) -> (m,8) row-major slices (byte units)
    sw = D.swiglu_out_tap(m, n)
    obj = np.arange(m * n).reshape(m, n)
    bobj = np.empty(m * n, np.int64)
    for rr in range(m):
        for cc in range(n):
            bobj[((rr // 8) * (n // 8) + (cc // 8)) * 64 + (rr % 8) * 8 + (cc % 8)] = (
                obj[rr, cc]
            )
    exp = np.concatenate([obj[:, j * 8 : (j + 1) * 8].ravel() for j in range(n // 8)])
    got = sw.gather(bobj.astype(np.uint16).view(np.uint8))
    check(
        "swiglu walk emits GEMM3 A slices (m,8) row-major",
        np.array_equal(got, exp.astype(np.uint16).view(np.uint8)),
    )
    check(
        "swiglu walk 3 dims, BD strides 4-byte multiples (inner run exempt)",
        len(sw.sizes) == 3
        and sw.numel == m * n * 2
        and all(s % 4 == 0 for s in sw.strides[:-1])
        and sw.strides[-1] == 1,
    )

    # memtile2 corner turn, unchanged
    ct = D.corner_turn_tap(m, Kc)
    got_b = ct.gather(np.arange(m * Kc * 2).reshape(m, 2 * Kc))
    exp_b = np.concatenate(
        [
            np.arange(m * Kc * 2).reshape(m, 2 * Kc)[:, kk * 16 : (kk + 1) * 16].ravel()
            for kk in range(Kc // 8)
        ]
    )
    check("corner turn byte tap (memtile2, unchanged)", np.array_equal(got_b, exp_b))

    # row / param tap lockstep (step1's proof)
    A_rows = D.a_host_tap(M, K, m)
    SBG_rows = D.sbg_host_tap(M, 3 * K)
    a_idx = np.arange(M * K).reshape(M, K)
    sbg_idx = np.arange(M * 3 * K).reshape(M, 3 * K)
    r0 = m
    got_a = A_rows[r0 : r0 + m].repeat(2).gather(a_idx).reshape(-1, K) // K
    got_sbg = (
        SBG_rows[r0 : r0 + m].repeat(2).gather(sbg_idx).reshape(-1, 3 * K) // (3 * K)
    )
    check("inA/inSBG taps pair the same tokens in the same order",
          np.array_equal(got_a[:, 0], got_sbg[:, 0]))

    # prepack stream sizes line up with the design's tap object counts
    sweeps = (N1 // N_TILE) // G_GU
    check("swipes * G_GU * n == N1 (inner dim coverage)",
          sweeps * G_GU * N_TILE == N1)
    check("G_G3 * L_G3 * n3 == K (down output coverage)",
          G_G3 * L_G3 * N_TILE == K)

    print(f"[static] ALL PASS: {ok}")
    return 0 if ok else 1


# --------------------------------------------------------------------------
# compile
# --------------------------------------------------------------------------


def inspect_artifacts(M) -> None:
    xclbin, _, _, _, stem = _paths(M)
    prj = OUT / (stem + ".prj")
    mlir = prj / "input_with_addresses.mlir"
    if not mlir.is_file():
        print(f"[inspect] no {mlir.name}; skipping placement report")
        return
    text = mlir.read_text(encoding="utf-8", errors="replace")
    tile_of = {}
    for mm in re.finditer(r"%(\w+) = aie\.tile\((\d+), (\d+)\)", text):
        tile_of[mm.group(1)] = f"({mm.group(2)},{mm.group(3)})"
    print("[inspect] buffer placement (address, 64B-aligned?):")

    def _elsize(dt: str) -> int:
        return {"i8": 1, "bf16": 2, "i32": 4, "i64": 8}.get(
            dt.split("<")[0], 9 if "bfp" in dt else 2
        )

    l1_used = {}
    n_bad = 0
    for mm in re.finditer(
        r"%(\w+) = aie\.buffer\(%(\w+)\) \{address = (\d+) : i32, mem_bank = (\d+) : i32,"
        r' sym_name = "([^"]+)"\} : memref<(\d+)x([^>]+)>',
        text,
    ):
        _val, tile_val, addr, bank, name, nelem, dt = mm.groups()
        nbytes = int(nelem) * _elsize(dt)
        aligned = (int(addr) % 64) == 0
        if not aligned:
            n_bad += 1
        tile = tile_of.get(tile_val, "?")
        l1_used[tile] = l1_used.get(tile, 0) + nbytes
        if not aligned or "ones" in name or "Cg" in name or "Cu" in name:
            print(
                f"[inspect]   {name:<24} tile {tile:<7} "
                f"bank {bank} addr {addr:>8}  {nbytes:>7} B  64B:{aligned}"
            )
    for tile, used in sorted(l1_used.items()):
        print(f"[inspect]   L1 data total at {tile}: {used:,} B")
    print(f"[inspect] buffers NOT 64 B aligned: {n_bad}")


def compile_design(M, level=3, g3_mode=2) -> int:
    from aie.iron import set_current_device
    from aie.iron.device import NPU2

    import step3_ffn as D

    xclbin, insts, _, _, stem = _paths(M, level)

    if (XRT_DIR / "xclbinutil.exe").is_file():
        os.environ["PATH"] = str(XRT_DIR) + os.pathsep + os.environ.get("PATH", "")

    # NPU2() (NOT from_name("npu2")): 5 shim fills + 6 drains need the
    # per-column shim tiles (step1's measured refusal/fix).
    set_current_device(NPU2())
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    spec = D.step3_ffn.specialize(
        M=M, K=K, N1=N1, m=M_TILE, k=K_MICRO, n=N_TILE, n3=N_TILE,
        G_GU=G_GU, G_G3=G_G3, L_G3=L_G3, LEVEL=level, G3_MODE=g3_mode,
        use_cache=False,
    )
    xcl, ins = spec.compile(xclbin_path=str(xclbin), inst_path=str(insts))
    t1 = time.perf_counter()
    print(f"[compile] done in {t1 - t0:.1f} s -> {xcl} + {ins}")
    for p in (xclbin, insts):
        print(f"[compile] {p.name}: {p.stat().st_size:,} B")
    prj = OUT / (stem + ".prj")
    if prj.is_dir():
        total = sum(f.stat().st_size for f in prj.rglob("*") if f.is_file())
        nfiles = sum(1 for f in prj.rglob("*") if f.is_file())
        print(f"[compile] {prj.name}\\: {nfiles} files, {total:,} B total")
    inspect_artifacts(M)
    return 0


# --------------------------------------------------------------------------
# data prep: oracle truth + device inputs + exact device model
# --------------------------------------------------------------------------


def _load_oracle():
    path = SCRATCH / "oracle" / "dit_block.py"
    spec = importlib.util.spec_from_file_location("dit_block_oracle_s3", path)
    mod = importlib.util.module_from_spec(spec)
    # dit_block.py has `from __future__ import annotations` + @dataclass;
    # dataclasses._is_type looks the module up in sys.modules, so register
    # it before exec.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def gemm_model(A_b, Wq, kk_order):
    """Exact model of the device's mm_bfp(mixed) accumulation: operands are
    bfp16ebs8-quantized (A conv_even on-core, B floor host codec -- both done
    by the callers on the full arrays, which equals the per-call blocks),
    products accumulated in f32 and C re-rounded to bf16 after EVERY call,
    in the exact call order `kk_order` (kk indices along the reduction)."""
    from ml_dtypes import bfloat16

    M, K_ = A_b.shape
    N = Wq.shape[0]
    C = np.zeros((M, N), np.float32)
    for kk in kk_order:
        P = A_b[:, kk * 8 : (kk + 1) * 8] @ Wq[:, kk * 8 : (kk + 1) * 8].T
        C = (C + P.astype(np.float32)).astype(bfloat16).astype(np.float32)
    return C


def prep_data(M, seed: int) -> int:
    from ml_dtypes import bfloat16

    from aie.iron.kernels.linalg import mm_bfp_tile_ref

    import refs

    import step3_ffn as D

    _, _, io_dir, _, _ = _paths(M)
    io_dir.mkdir(parents=True, exist_ok=True)

    db = _load_oracle()
    geom = db.Geometry(hidden=K, heads=30, head_dim=128, ffn_inner=N1, seq=M)
    cfg = db.DeviceCfg()
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    weights = db.make_weights(geom, rng)
    inputs = db.make_inputs(geom, rng)
    print(f"[prep] oracle weights/inputs generated in "
          f"{time.perf_counter() - t0:.1f} s")

    # ---- THE TRUTH: dit_block_float, one complete block ------------------
    col = {}
    t0 = time.perf_counter()
    x_out_truth = db.dit_block_float(inputs, weights, geom, cfg, collect=col)
    print(f"[prep] dit_block_float (full block) in "
          f"{time.perf_counter() - t0:.1f} s")
    x_mid = col["x_mid"]                 # the residual stream entering FFN
    # The device receives the post-attention residual rounded to bf16 --
    # exactly what dit_block_device does with any block input.
    x_dev = db.round_bf16(x_mid).astype(np.float32)

    gamma2 = np.asarray(weights["gamma2"], np.float32)
    s2 = np.asarray(inputs["s2"], np.float64)
    b2 = np.asarray(inputs["b2"], np.float64)
    W_gate = np.asarray(weights["w_gate"], np.float32)   # (N1, K) torch
    W_up = np.asarray(weights["w_up"], np.float32)       # (N1, K) torch
    W_down = np.asarray(weights["w_down"], np.float32)   # (K, N1) torch

    # ---- device-exact host buffers --------------------------------------
    A_b = x_dev.astype(bfloat16)
    A_bytes = np.ascontiguousarray(A_b).tobytes()

    s_off, b_off, g_off, elem_bytes = D.sbg_row_layout(K)
    SBG = np.zeros((M, 3 * K), np.float32)
    SBG[:, :K] = s2
    SBG[:, K : 2 * K] = b2
    SBG[:, 2 * K :] = gamma2
    sbg_bytes = np.ascontiguousarray(SBG.astype(bfloat16)).tobytes()

    t0 = time.perf_counter()
    BG = pack_bg_stream(W_gate, W_up)
    B3 = pack_b3_stream(W_down)
    X = pack_x_stream(A_b)
    print(f"[prep] prepacked BG {BG.size:,} B  B3 {B3.size:,} B  "
          f"X {X.size:,} B in {time.perf_counter() - t0:.1f} s")
    C_len = M * K * 2

    # ---- exact device model (per-call bf16 C re-rounding) ----------------
    x_b = A_b.astype(np.float32)
    g_b = gamma2.astype(bfloat16).astype(np.float32)
    s_b = np.asarray(s2, np.float32).astype(bfloat16).astype(np.float32)
    b_b = np.asarray(b2, np.float32).astype(bfloat16).astype(np.float32)
    N2 = refs.rms_norm_weighted_ref(x_b, g_b, eps=EPS, out_dtype="bf16").astype(
        np.float32
    )
    Y2 = refs.adaln_modulate_ref(N2, s_b, b_b, out_dtype="bf16").astype(np.float32)

    from aie.utils import bfp as _B  # noqa: F401  (codec: pack_b_blocks docs)
    oracle = db
    Aq = oracle.bfp16ebs8_quantize(Y2, "conv_even")        # on-core A codec
    Wq_gate = oracle.bfp16ebs8_quantize(W_gate, "floor")   # host B codec
    Wq_up = oracle.bfp16ebs8_quantize(W_up, "floor")
    Wq_down = oracle.bfp16ebs8_quantize(W_down, "floor")   # (K, N1)

    t0 = time.perf_counter()
    gate = gemm_model(Aq, Wq_gate, range(K // K_MICRO))
    up = gemm_model(Aq, Wq_up, range(K // K_MICRO))
    print(f"[prep] gate/up GEMM model in {time.perf_counter() - t0:.1f} s")

    # cross-check the vectorized model against mm_bfp_tile_ref on sample tiles
    i0, j0 = 0, 0
    acc = np.zeros((M_TILE, N_TILE), np.float32)
    for kk in range(K // K_MICRO):
        r = mm_bfp_tile_ref(
            Y2[i0 : i0 + M_TILE, kk * 8 : (kk + 1) * 8],
            W_gate[j0 : j0 + N_TILE, kk * 8 : (kk + 1) * 8].T,
            dim_m=M_TILE, dim_k=K_MICRO, dim_n=N_TILE, mixed=True,
        ).reshape(M_TILE, N_TILE)
        acc = (acc + r.astype(np.float32)).astype(bfloat16).astype(np.float32)
    tile_ok = np.array_equal(
        acc, gate[i0 : i0 + M_TILE, j0 : j0 + N_TILE]
    )
    print(f"[prep] gemm_model == mm_bfp_tile_ref loop on sample tile: {tile_ok}")

    # swiglu: the kernel computes (x*w1)*silu(x*w2) with x=1.0, w1=up,
    # w2=gate -> exactly up*silu(gate); use_lut=True build = table model.
    ones = np.ones_like(up)
    from aie.iron.kernels.activation import swiglu_table_ref, _swiglu_model

    sg = np.asarray(swiglu_table_ref(ones, up, gate), np.float32)
    # structural model with an EXACT sigmoid = the oracle's device_swiglu
    sg_exact = db.device_swiglu(gate, up)
    lut_gap = float(np.abs(sg - sg_exact).max())

    t0 = time.perf_counter()
    # down GEMM call order: (sweep s, gate/up worker w, slice s2) visiting
    # inner block kk' = 320*w + 8*s + s2 -- the join/broadcast order the
    # device really accumulates in.
    order = [
        320 * w + 8 * s + s2
        for s in range((N1 // N_TILE) // G_GU)
        for w in range(G_GU)
        for s2 in range(N_TILE // K_MICRO)
    ]
    Aq2 = oracle.bfp16ebs8_quantize(sg, "conv_even")
    ff = gemm_model(Aq2, Wq_down, order)
    print(f"[prep] down GEMM model (permuted {len(order)} calls) in "
          f"{time.perf_counter() - t0:.1f} s")

    y_model = db.round_bf16(x_b + ff).astype(np.float32)   # add_sized

    np.save(io_dir / "A.npy", np.frombuffer(A_bytes, np.uint8))
    np.save(io_dir / "SBG.npy", np.frombuffer(sbg_bytes, np.uint8))
    np.save(io_dir / "BG.npy", BG)
    np.save(io_dir / "B3.npy", B3)
    np.save(io_dir / "X.npy", np.frombuffer(
        np.ascontiguousarray(X.astype(bfloat16)).tobytes(), np.uint8))
    np.savez(
        io_dir / "ref.npz",
        x_mid=x_mid, x_out_truth=x_out_truth,
        N2=N2, Y2=Y2, gate=gate, up=up, sg=sg, sg_exact=sg_exact, ff=ff,
        y_model=y_model, lut_gap=np.float64(lut_gap),
        x_dev=x_b, dims=np.array([M, K, N1], np.int64), seed=np.int64(seed),
    )
    print(f"[prep] lut-vs-exact-sigmoid swiglu gap: {lut_gap:.4e}")
    print(f"[prep] Y model absmax {float(np.abs(y_model).max()):.4f}   "
          f"truth absmax {float(np.abs(x_out_truth).max()):.4f}")
    print(f"[prep] saved -> {io_dir}")
    return 0


# --------------------------------------------------------------------------
# device run (Python 3.10, --xrt-only)
# --------------------------------------------------------------------------


def xrt_main(args) -> int:
    for d in (XRT_DIR, pathlib.Path(r"C:\Windows\System32\AMD")):
        if d.exists():
            sys.path.insert(0, str(d))
            try:
                os.add_dll_directory(str(d))
            except Exception:  # noqa: BLE001
                pass
    import pyxrt

    io = pathlib.Path(args.io)
    a_bytes = np.load(io / "A.npy").tobytes()
    sbg_bytes = np.load(io / "SBG.npy").tobytes()
    bg = np.load(io / "BG.npy")
    b3 = np.load(io / "B3.npy")
    x_bytes = np.load(io / "X.npy").tobytes()
    bg_bytes = np.ascontiguousarray(bg).tobytes()
    b3_bytes = np.ascontiguousarray(b3).tobytes()
    c_len = int(args.c_len)
    print(f"A {len(a_bytes):,} B   SBG {len(sbg_bytes):,} B   "
          f"BG {len(bg_bytes):,} B   B3 {len(b3_bytes):,} B   "
          f"X {len(x_bytes):,} B   Y {c_len:,} B")

    dev = pyxrt.device(0)
    x = pyxrt.xclbin(args.xclbin)
    uuid = dev.register_xclbin(x)
    k = pyxrt.kernel(dev, uuid, KERNEL)
    print(f"kernel {KERNEL} opened")

    inst = pathlib.Path(args.insts).read_bytes()
    ninstr = len(inst) // 4
    print(f"insts {len(inst):,} B  ({ninstr} words)")

    fl = pyxrt.bo.flags.host_only
    bo_instr = pyxrt.bo(dev, len(inst), pyxrt.bo.flags.cacheable, k.group_id(1))
    payloads = [a_bytes, sbg_bytes, bg_bytes, b3_bytes, x_bytes, bytes(c_len)]
    bos = []
    for i, payload in enumerate(payloads):
        size = max(len(payload), 4)
        bo = pyxrt.bo(dev, size, fl, k.group_id(3 + i))
        if payload:
            bo.write(payload, 0)
        bos.append(bo)
    bo_y = bos[-1]

    bo_instr.write(inst, 0)
    for b in [bo_instr] + bos:
        b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    states = []
    raws = []
    for run_i in range(args.runs):
        t0 = time.perf_counter()
        run = k(3, bo_instr, ninstr, *bos)
        state = run.wait(args.timeout_ms)
        t1 = time.perf_counter()
        states.append(int(state))
        try:
            state_name = str(pyxrt.ert.ERT_CMD_STATE(state))
        except Exception:  # noqa: BLE001
            state_name = str(state)
        print(f"run {run_i}: wait() -> {state} ({state_name})  "
              f"wall {1000 * (t1 - t0):.2f} ms")
        bo_y.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        raw = np.frombuffer(bytes(bo_y.read(c_len, 0)), dtype=np.uint8)
        raws.append(raw)

    if len(raws) > 1:
        same = all(np.array_equal(raws[0], r) for r in raws[1:])
        print(f"run-to-run byte-identical: {same}")

    raw = raws[0]
    np.save(args.out, raw)
    print(f"raw Y saved -> {args.out}  ({raw.size:,} B)")
    print(f"first 32 bytes: {raw[:32].tobytes().hex()}")

    completed = 4  # ERT_CMD_STATE_COMPLETED
    ok = all(s == completed for s in states)
    print(f"ERT_CMD_STATE_COMPLETED (4) on every run: {ok}")
    return 0 if ok else 2


def run_on_npu(M, runs: int) -> int:
    xclbin, insts, io_dir, c_raw, _ = _paths(M)
    cmd = [
        str(PY310), str(pathlib.Path(__file__).resolve()),
        "--xrt-only",
        "--io", str(io_dir),
        "--xclbin", str(xclbin),
        "--insts", str(insts),
        "--out", str(c_raw),
        "--runs", str(runs),
        "--c-len", str(M * K * 2),
        "--timeout-ms", str(int(os.environ.get("STEP3_TIMEOUT_MS", "300000"))),
    ]
    print(f"[run] {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(ROOT))
    print(f"[run] xrt runner exit code: {r.returncode}")
    return r.returncode


# --------------------------------------------------------------------------
# verify: the JUDGED metric (vs dit_block_float) + the model ulps
# --------------------------------------------------------------------------


def verify(M) -> int:
    from ml_dtypes import bfloat16

    import refs

    _, _, io_dir, c_raw, _ = _paths(M)
    ref = np.load(io_dir / "ref.npz")
    raw = np.load(c_raw)
    Y = raw.view(np.uint16).view(bfloat16).reshape(M, K).astype(np.float64)

    truth = np.asarray(ref["x_out_truth"], np.float64)
    model = np.asarray(ref["y_model"], np.float64)
    sg = np.asarray(ref["sg"], np.float32)
    sg_exact = np.asarray(ref["sg_exact"], np.float32)

    # ---- THE JUDGED METRIC (STAGE3.md): rel_to_absmax vs dit_block_float --
    db = _load_oracle()
    div = db.divergence(truth, Y)
    u_t = refs.bf16_ulp(np.maximum(np.abs(truth), np.abs(Y)))
    live_t = (truth != 0.0) | (Y != 0.0)
    ulp_t = np.zeros_like(Y)
    ulp_t[live_t] = np.abs(Y - truth)[live_t] / np.maximum(u_t[live_t], 2.0**-126)
    print("[verify] === vs imaggers/oracle/dit_block_float x_out (THE TRUTH) ===")
    print(f"[verify] max abs err            : {div['max_abs']:.6e}")
    print(f"[verify] max ulp (bf16, floored): {float(ulp_t.max()):.3f}")
    print(f"[verify] rel_to_absmax (JUDGED) : {div['rel_to_absmax']:.6e} "
          f"(budget 5.4e-02; swiglu use_lut=True so no +0.08)")
    print(f"[verify] max rel err (gated 1%) : {div['max_rel']:.6e}")
    print(f"[verify] rms err                : {div['rms']:.6e}")
    print(f"[verify] truth absmax           : {div['ref_absmax']:.6e}")
    judged_ok = div["rel_to_absmax"] <= 5.4e-2

    # ---- exact device model (per-call bf16 re-rounding chain) ------------
    model_b = model.astype(bfloat16).astype(np.float64)
    ok_ulp, max_abs_m, max_ulp = refs.check_within_ulps(Y, model_b)
    exact = float((Y == model_b).mean())
    # Robust ulp distance: refs.check_within_ulps divides by the ref's ulp,
    # which is 0 at exact zeros (the swiglu ftz gate writes exact 0s), so
    # the max can come out inf.  Floor the ulp at the smallest normal.
    u = refs.bf16_ulp(np.maximum(np.abs(model_b), np.abs(Y)))
    live = (model_b != 0.0) | (Y != 0.0)
    ulp_dist = np.zeros_like(Y)
    ulp_dist[live] = np.abs(Y - model_b)[live] / np.maximum(
        u[live], 2.0**-126
    )
    max_ulp_r = float(ulp_dist.max()) if live.any() else 0.0
    n_gt2 = int((ulp_dist > 2.0).sum())
    print("[verify] === vs exact device model (bf16 per call) ===")
    print(f"[verify] max abs err {max_abs_m:.4e}   max ulp {max_ulp_r:.3f}   "
          f"bit-exact {100.0 * exact:.2f}% of {Y.size} elements   "
          f"elements > 2 ulp: {n_gt2}")
    print(f"[verify] refs.check_within_ulps(<= 2): {ok_ulp} "
          f"(raw max ulp {max_ulp:.3f} includes zero-vs-subnormal pairs)")

    # ---- swiglu build gap (table vs exact sigmoid) -----------------------
    print("[verify] === swiglu build ===")
    print(f"[verify] use_lut=True table model vs oracle's exact-sigmoid "
          f"device_swiglu: max abs {float(np.abs(sg - sg_exact).max()):.4e}")

    print(f"[verify] Y[0,:6]      = {np.round(Y[0, :6], 4)}")
    print(f"[verify] truth[0,:6]  = {np.round(truth[0, :6], 4)}")
    print(f"[verify] JUDGED BUDGET PASS (rel_to_absmax <= 5.4e-2): {judged_ok}")
    return 0 if judged_ok else 3


# --------------------------------------------------------------------------


def main() -> int:
    global LEVEL, G_GU, G_G3, L_G3
    _force_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("-M", type=int, default=M_DEF)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--level", type=int, default=3, choices=(1, 2, 3),
                    help="bisect cut: 1 = through join, 2 = + down-GEMM, 3 = full")
    ap.add_argument("--g-gu", dest="g_gu", type=int, default=G_GU,
                    help="gate/up worker count (bisect knob)")
    ap.add_argument("--g-g3", dest="g_g3", type=int, default=G_G3,
                    help="down-GEMM worker count (bisect knob)")
    ap.add_argument("--l-g3", dest="l_g3", type=int, default=L_G3,
                    help="C3 tiles held per down worker (bisect knob)")
    ap.add_argument("--g3-mode", dest="g3_mode", type=int, default=2,
                    choices=(0, 1, 2),
                    help="down-core probe: 0 = relay, 1 = batch+zero, 2 = full")
    ap.add_argument("--skip-compile", action="store_true")
    ap.add_argument("--skip-run", action="store_true")
    ap.add_argument("--skip-static", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--prep-only", action="store_true")
    # --xrt-only mode: the Python 3.10 device half.
    ap.add_argument("--xrt-only", action="store_true")
    ap.add_argument("--io", type=str, default=None)
    ap.add_argument("--xclbin", type=str, default=None)
    ap.add_argument("--insts", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--c-len", type=int, default=0)
    ap.add_argument("--timeout-ms", type=int, default=300000)
    args = ap.parse_args()

    if args.xrt_only:
        return xrt_main(args)

    _install_mkdir_mode_shim()
    M = args.M
    LEVEL = args.level
    G_GU = args.g_gu
    G_G3 = args.g_g3
    L_G3 = args.l_g3

    if args.verify_only:
        return verify(M)

    print("=" * 78)
    print(f"step3: FFN branch + second residual   M={M} K={K} N1={N1} "
          f"tile ({M_TILE},{K_MICRO},{N_TILE})  G_GU={G_GU} G_G3={G_G3} "
          f"L_G3={L_G3}")
    print("=" * 78)

    if not args.skip_static:
        if static_checks(M) != 0:
            print("[step3] static FAILED")
            return 2

    if not args.skip_compile:
        try:
            compile_design(M, args.level, args.g3_mode)
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"[compile] REFUSED: {type(exc).__name__}: {exc}")
            return 1

    prep_data(M, args.seed)
    if args.prep_only or args.skip_run:
        return 0
    rc = run_on_npu(M, args.runs)
    if rc != 0:
        print(f"[run] NPU run did not complete cleanly (exit {rc})")
    vrc = verify(M)
    return vrc


if __name__ == "__main__":
    raise SystemExit(main())
