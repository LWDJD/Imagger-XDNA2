# SPDX-License-Identifier: MIT
"""prod_repack.py -- step 3: fs_ffn_j5 BG/B3 -> big-M [worker][sweep][k-group].

DELIVERED (t4_out/payload_nh5c_k8/BG.npy, 88,473,600 B):
    [sweep][kk][w][gate|up][576 B tile]      SWEEPS=40, KK=480, G_GU=4, 2=gate|up
BIG-M target:
    [worker][sweep][k-group][8 kk][gate|up][576 B tile]
    one k-group = 8 delivered kk tiles = 64 k = the 4,608 B object bigm_proj
    consumes at n=64.

KEY FINDING: the delivered [gate|up] axis IS the swiglu pairing -- each 64-column
group already carries its gate tile and its up tile adjacently.  Step 3 is
therefore a pure permutation of whole 576 B tiles; it does not have to invent the
pairing, only preserve it.  That is the interface t22_swiglu needs.

    py312 prod_repack.py --selftest
    py312 prod_repack.py --which B3 --out io_repacked
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np

TILE = 576
KG_TILES = 8          # 8 x 8k = 64 k per big-M object (the k=64 generation)
# PRODASM-STEP345 110: the product instantiates every mm_bfp with dim_k=8, and
# the k=64 build made each call compute only 1/8 of the k axis.  For the k=8
# generation one object must carry 1 x 8k = 8 k, i.e. KG_TILES = 1.  The total
# byte count is unchanged either way; only the block grouping differs.


def set_kg_tiles(n: int) -> None:
    """Select the k-grouping the target image was compiled for (8 -> 1)."""
    global KG_TILES
    if n not in (1, 8):
        raise SystemExit("KG_TILES must be 1 (k=8) or 8 (k=64), got %r" % (n,))
    KG_TILES = n


def shapes(nbytes: int, G_GU: int, sweeps: int) -> dict:
    per_sweep = nbytes // sweeps
    kk = per_sweep // (G_GU * 2 * TILE)
    if kk * G_GU * 2 * TILE * sweeps != nbytes:
        raise SystemExit("size does not factor: %d" % nbytes)
    if kk % KG_TILES:
        raise SystemExit("kk=%d not a multiple of %d" % (kk, KG_TILES))
    return {"bytes": nbytes, "sweeps": sweeps, "kk": kk, "G_GU": G_GU,
            "k_groups": kk // KG_TILES, "tile": TILE,
            "objects": sweeps * (kk // KG_TILES) * G_GU,
            "object_bytes": KG_TILES * 2 * TILE}


def to_bigm(a: np.ndarray, s: dict) -> np.ndarray:
    src = a.reshape(s["sweeps"], s["kk"], s["G_GU"], 2, TILE)
    mid = src.reshape(s["sweeps"], s["k_groups"], KG_TILES, s["G_GU"], 2, TILE)
    return np.ascontiguousarray(mid.transpose(3, 0, 1, 2, 4, 5)).reshape(-1)


def from_bigm(b: np.ndarray, s: dict) -> np.ndarray:
    mid = b.reshape(s["G_GU"], s["sweeps"], s["k_groups"], KG_TILES, 2, TILE)
    # back to [sweep][k-group][8 kk][worker][gate|up][tile], which is the same
    # memory as the source's [sweep][kk=60*8][worker][gate|up][tile]
    src = mid.transpose(1, 2, 3, 0, 4, 5)
    return np.ascontiguousarray(src).reshape(-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"D:\Project\deepseek_harness\public_work"
                    r"\zimage-bfp16-npu\imaggers\block\t4_out\payload_nh5c_k8")
    ap.add_argument("--out", default="")
    ap.add_argument("--which", default="BG", choices=["BG", "B3"])
    ap.add_argument("--sweeps", type=int, default=0)
    ap.add_argument("--G_GU", type=int, default=4)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    src_path = pathlib.Path(a.src) / (a.which + ".npy")
    if not src_path.is_file():
        raise SystemExit("missing %s" % src_path)
    a_src = np.load(src_path).reshape(-1)
    sweeps = a.sweeps
    if not sweeps:
        for cand in (40, 20, 10, 80):
            try:
                shapes(a_src.size, a.G_GU, cand); sweeps = cand; break
            except SystemExit:
                continue
        if not sweeps:
            raise SystemExit("cannot infer sweeps for %d B" % a_src.size)
    s = shapes(a_src.size, a.G_GU, sweeps)
    s["src"] = str(src_path)
    b = to_bigm(a_src, s)
    assert b.size == a_src.size, (b.size, a_src.size)
    print(json.dumps(s, indent=1))
    print("[repack] %s: %s B -> %s objects x %s B"
          % (a.which, format(a_src.size, ","), format(s["objects"], ","),
             format(s["object_bytes"], ",")))
    ok = True
    if a.selftest:
        back = from_bigm(b, s)
        rt = np.array_equal(back, a_src)
        moved = not np.array_equal(b, a_src)
        ob = s["object_bytes"]
        # object 0 is (worker 0, sweep 0, k-group 0) = source [0][kk 0..7][w 0]
        want = a_src.reshape(s["sweeps"], s["kk"], s["G_GU"], 2, TILE)[0, 0:KG_TILES, 0]
        got = b[:ob].reshape(KG_TILES, 2, TILE)
        pt = np.array_equal(got, want)
        print("[selftest] round-trip %s | permutation non-trivial %s | point-check %s"
              % ("OK" if rt else "FAIL", "yes" if moved else "NO",
                 "OK" if pt else "FAIL"))
        ok = rt and pt
    if a.out and ok:
        d = pathlib.Path(a.out); d.mkdir(parents=True, exist_ok=True)
        np.save(d / (a.which + ".npy"), b)
        print("[repack] wrote %s" % (d / (a.which + ".npy")))
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(s, indent=1), encoding="utf-8")
    print("[repack] %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
