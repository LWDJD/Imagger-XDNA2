"""build_skeleton_manifest.py -- describe the exported skeletons for the runner.

`generate.py` drives the DiT through `ShardRunner`, which expects a manifest in
this shape:

    {
      "cap_tokens": int,
      "plan":      [shard names in execution order],
      "shards":    [{"name", "onnx", "inputs", "output", ...}]
    }

The exported skeletons live in a flatter layout and carry their weights as graph
INPUTS rather than initialisers, so this script reads each `skeleton.onnx` and
produces a manifest with two additions:

    "activation_inputs": the graph inputs that are NOT weights
                         (hidden / cos / sin / adaln)
    "weight_inputs":     the graph inputs that ARE weights, in graph order

Those two lists are what `SkeletonRunner` needs to build the feed dict.

Nothing is overwritten unless you point --out at an existing file on purpose.

Usage:
    python build_skeleton_manifest.py
    python build_skeleton_manifest.py --skel-dir _probe_out/skeleton_all
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))

DEFAULT_SKEL = HERE / "_probe_out" / "skeleton_all"

# Execution order of the Z-Image DiT shards.  `zimage_pipeline.pure_forward`
# runs noise_refiner, then context_refiner, then the layer shards in manifest
# order.  A glob would give lexicographic order, which is wrong for the layers.
EXEC_ORDER = (
    ["noise_refiner", "context_refiner"]
    + [f"layers_{i:02d}_{i + 2:02d}" for i in range(0, 30, 2)]
)


def _force_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass


def inspect(skel: pathlib.Path) -> dict:
    import onnx

    m = onnx.load(str(skel), load_external_data=False)
    g = m.graph
    node_types: dict[str, int] = {}
    for n in g.node:
        node_types[n.op_type] = node_types.get(n.op_type, 0) + 1

    outs = [o.name for o in g.output]
    ins = [i.name for i in g.input]
    # weights arrive as inputs whose name looks like a parameter path
    weightish = []
    activations = []
    seq = None
    for i in g.input:
        name = i.name
        base = name.split("__", 1)[0]
        if base in ("hidden", "cos", "sin", "adaln"):
            activations.append(name)
            if base == "hidden":
                dims = [d.dim_value for d in
                        i.type.tensor_type.shape.dim]
                # hidden is (1, S, DIM); S is what the shard was exported for
                if len(dims) == 3 and dims[1]:
                    seq = int(dims[1])
        else:
            weightish.append(name)

    return {
        "inputs": ins,
        "activation_inputs": activations,
        "weight_inputs": weightish,
        "output": outs[0] if outs else "",
        "seq": seq,
        "n_nodes": len(g.node),
        "op_histogram": dict(sorted(node_types.items(),
                                    key=lambda kv: -kv[1])),
    }


def main() -> int:
    _force_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--skel-dir", default=str(DEFAULT_SKEL))
    ap.add_argument("--out", default=None)
    ap.add_argument("--cap-tokens", type=int, default=32,
                    help="caption tokens the context_refiner shard expects")
    args = ap.parse_args()

    sdir = pathlib.Path(args.skel_dir).resolve()
    if not sdir.is_dir():
        print(f"[FAIL] {sdir} is not a directory")
        return 1

    found = {p.parent.name: p for p in sdir.glob("*/skeleton.onnx")}
    print("=" * 78)
    print("build_skeleton_manifest")
    print("=" * 78)
    print(f"  skeletons in {sdir}: {len(found)}")

    order = [n for n in EXEC_ORDER if n in found]
    extra = sorted(set(found) - set(order))
    if extra:
        print(f"  note: {extra} are not in EXEC_ORDER and will be appended")
        order += extra
    missing = [n for n in EXEC_ORDER if n not in found]
    if missing:
        print(f"  [WARN] missing from EXEC_ORDER: {missing}")

    shards = []
    for name in order:
        info = inspect(found[name])
        io_path = found[name].parent / "io.npz"
        rec = {
            "name": name,
            "onnx": str(found[name]),
            "inputs": info["inputs"],
            "activation_inputs": info["activation_inputs"],
            "weight_inputs": info["weight_inputs"],
            "output": info["output"],
            "n_nodes": info["n_nodes"],
            "seq": info["seq"],
            "io": str(io_path) if io_path.exists() else "",
            "group": ("layers" if name.startswith("layers_")
                      else name),
        }
        shards.append(rec)
        print(f"  {name:16s} seq {str(info['seq']):>5}  "
              f"nodes {info['n_nodes']:4d}   "
              f"activations {len(info['activation_inputs'])}   "
              f"weights {len(info['weight_inputs'])}")

    # `generate.py` guards the geometry with these: the token counts the shards
    # were exported for.  img_tokens comes from noise_refiner's hidden, which is
    # the image stream; cap_tokens from context_refiner's, the caption stream.
    seq_of = {s["name"]: s["seq"] for s in shards}
    img_tokens = seq_of.get("noise_refiner")
    cap_tokens = seq_of.get("context_refiner")
    if img_tokens is None or cap_tokens is None:
        print(f"  [WARN] could not derive img_tokens/cap_tokens: {seq_of}")

    manifest = {
        "img_tokens": img_tokens,
        "cap_tokens": cap_tokens,
        "plan": [s["name"] for s in shards],
        "shards": shards,
        "skeleton_dir": str(sdir),
    }
    out = pathlib.Path(args.out) if args.out else sdir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}  ({out.stat().st_size:,} B)")
    print(f"  {len(shards)} shards, "
          f"{sum(s['n_nodes'] for s in shards)} nodes total, "
          f"{sum(len(s['weight_inputs']) for s in shards)} weight inputs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
