# SPDX-License-Identifier: MIT
"""s2_proj32.py -- the projections (q/k/v/o) at 32 cores: executor + CPU dry.

Same placement skeleton as the FFN images (8 columns x 4 cores, A split 1->4,
B direct fan-out, C joined 4->1 at the memtile), same layout functions from
s2_ffn32 (so a layout proof carries over), K = 3840 and N = 4 x 3840.

    py312 python s2_proj32.py --selftest
    py312 python s2_proj32.py --dry --key layers.0 --mode fused
    py312 python s2_proj32.py --dry --key layers.0 --mode four

Two modes:
  fused  one image,  N = 15360, n = 384, 40 n-tiles   (proj32_fused)
  four   four images, N = 3840,  n = 384, 10 n-tiles each (reuses ffn_dn_m32)

The DRAM byte account is printed either way; it is the same number, which is
the whole point of the fused-vs-four answer (see --account).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s2_ffn32 as S2                     # layout primitives + codec

PROJ = ("to_q", "to_k", "to_v", "to_out.0")
K = 3840
N1 = 3840                                  # one projection
NF = N1 * 4                                # fused
N_TILE = 384


def _bf16(x):
    return S2._bf16(x)


def load_proj_weights(key="layers.0"):
    """The model's four projection matrices as BFP16 blocks (out, in//8, 9)."""
    li = int(key.split(".")[1])
    p = HERE.parents[2] / "weights" / f"block{li}.npz"
    z = np.load(p, allow_pickle=False)
    out = {}
    for name in PROJ:
        b = z[f"{key}.attention.{name}.weight"]
        out[name] = np.ascontiguousarray(b.reshape(3840, 3840 // 8, 9))
    return out


def _blocks_concat(w):
    """[wq|wk|wv|wo] along N -> (15360, 480, 9)."""
    return np.concatenate([w[n] for n in PROJ], axis=0)


def proj32_dry(x, w, mode="fused", n_tile=N_TILE):
    """Run q/k/v/o through the image layouts; returns ({name: (M,3840)}, info)."""
    M = x.shape[0]
    info = {"mode": mode, "M": int(M), "parts": {}}
    if mode == "fused":
        blocks = _blocks_concat(w)
        gate_all = np.zeros((M, NF), np.float32)
        if M > 1024:
            c, i1 = S2.gemm_dry(x[:1024], blocks, K, NF, n_tile)
            ct, i2 = S2.gemm_dry(x[1024:], blocks, K, NF, NF // (S2.COLS * S2.CPC),
                                 tail=True)
            gate_all[:1024], gate_all[1024:] = c, ct
            info["parts"]["main"], info["parts"]["tail"] = i1, i2
        else:
            c, i1 = S2.gemm_dry(x, blocks, K, NF, n_tile)
            gate_all[:] = c
            info["parts"]["main"] = i1
        out = {n: gate_all[:, i * N1:(i + 1) * N1] for i, n in enumerate(PROJ)}
    else:
        blocks = {n: w[n] for n in PROJ}
        out = {}
        for n in PROJ:
            tgt = np.zeros((M, N1), np.float32)
            if M > 1024:
                c, i1 = S2.gemm_dry(x[:1024], blocks[n], K, N1, N_TILE)
                ct, i2 = S2.gemm_dry(x[1024:], blocks[n], K, N1,
                                     N1 // (S2.COLS * S2.CPC), tail=True)
                tgt[:1024], tgt[1024:] = c, ct
                info["parts"][n] = {"main": i1, "tail": i2}
            else:
                c, i1 = S2.gemm_dry(x, blocks[n], K, N1, N_TILE)
                tgt[:] = c
                info["parts"][n] = {"main": i1}
            out[n] = tgt
    # the PRODUCT convention: q/k/v/o, not the checkpoint's to_q/to_k/...
    QKVO = {"q": "to_q", "k": "to_k", "v": "to_v", "o": "to_out.0"}
    return {k: out[v] for k, v in QKVO.items()}, info


def proj32_reference(x, w):
    xq = _bf16(np.asarray(x, np.float32))
    QKVO = {"q": "to_q", "k": "to_k", "v": "to_v", "o": "to_out.0"}
    return {k: _bf16(xq @ S2._decode_blocks(w[v], K, N1)) for k, v in QKVO.items()}


def account(M=1056, n_tile=None):
    """The DRAM ledger for one layer's four projections, both modes.

    n_tile must be threaded through: A is re-read NF/n_tile times, so the
    ledger changes with the tile width (n=768 halves A versus n=384).
    Parent fix 2026-11-01: the old body used the module constant N_TILE, so
    --n had no effect on the account (the 413.6 MB row was n=384 twice).
    """
    n_tile = N_TILE if n_tile is None else int(n_tile)
    rows_main, rows_tail = 1024, 32
    a_main = rows_main * K * 2
    a_tail = rows_tail * K * 2
    b_one = N1 * K // 8 * 9
    res = {}
    for mode, nt in (("fused", NF // n_tile), ("four", N1 // n_tile)):
        nimg = 1 if mode == "fused" else 4
        a = nimg * (nt * a_main + a_tail)
        b = nimg * b_one * (4 if mode == "fused" else 1)
        c = M * N1 * 2 * 4
        res[mode] = {"a_bytes": int(a), "b_bytes": int(b), "c_bytes": int(c),
                     "total_bytes": int(a + b + c),
                     "dispatches": int(nimg * 2),
                     "ms_at_49GBs": (a + b + c) / 49e9 * 1e3}
    return res


def _selftest():
    rng = np.random.default_rng(11)
    x = _bf16(rng.standard_normal((1024, K)).astype(np.float32))
    a = S2.pack_a(x)
    ok = bool(np.array_equal(S2.unpack_a(a, 1024, K), x))
    print(f"[selftest] A round-trip {a.size:,} B -> {'EXACT' if ok else 'BROKEN'}")
    w = load_proj_weights()
    b1 = S2.pack_b_from_blocks(w["to_q"], K, N1, N_TILE)
    back = S2.unpack_b(b1, K, N1, N_TILE)
    ok2 = bool(np.array_equal(back, S2._decode_blocks(w["to_q"], K, N1)))
    print(f"[selftest] B round-trip {b1.size:,} B -> {'EXACT' if ok2 else 'BROKEN'}")
    return ok and ok2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--mode", default="fused", choices=("fused", "four"))
    ap.add_argument("--n", type=int, default=384, help="n tile (384 or 768)")
    ap.add_argument("--key", default="layers.0")
    ap.add_argument("--account", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if args.account:
        print(json.dumps(account(), indent=2))
        return 0
    if args.selftest:
        return 0 if _selftest() else 1
    if args.dry:
        w = load_proj_weights(args.key)
        rng = np.random.default_rng(5)
        x = (rng.standard_normal((1056, K)).astype(np.float32) * 0.5)
        t0 = __import__("time").perf_counter()
        out, info = proj32_dry(x, w, mode=args.mode, n_tile=args.n)
        t1 = __import__("time").perf_counter()
        ref = proj32_reference(x, w)
        t2 = __import__("time").perf_counter()
        rec = {"mode": args.mode, "n": args.n, "key": args.key, "dry_s": round(t1 - t0, 2),
               "ref_s": round(t2 - t1, 2), "proj": {}}
        worst_corr, worst_max = 1.0, 0.0
        for n in PROJ:
            d = (out[n] - ref[n]).astype(np.float64)
            c = float(np.corrcoef(out[n].reshape(-1), ref[n].reshape(-1))[0, 1])
            m = float(np.abs(d).max())
            rec["proj"][n] = {"corr": c, "maxabs": m,
                              "absmax": float(np.abs(ref[n]).max())}
            worst_corr, worst_max = min(worst_corr, c), max(worst_max, m)
        exact = True
        for v in info["parts"].values():
            group = (list(v.values()) if isinstance(v, dict) and "main" in v
                     else [v])
            for one in group:
                exact &= bool(one["a_equal"] and one["b_equal"] and one["c_equal"])
        rec.update({"worst_corr": worst_corr, "worst_maxabs": worst_max,
                    "all_layouts_exact": bool(exact),
                    "account": account(n_tile=args.n)})
        print(json.dumps(rec, indent=2)[:3500])
        if args.json:
            pathlib.Path(args.json).write_text(json.dumps(rec, indent=2))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
