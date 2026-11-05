# weights — how `weights.bfp16` is made

One 6.91 GB file holds the whole Z-Image-Turbo DiT in AMD's kernel-ready
`bfp16ebs8` layout: **1.125 bytes per parameter**, 6,143,056,384 parameters,
22.93 GiB of fp32 source compressed to 6.44 GiB.

## The chain

```
Tongyi-MAI/Z-Image-Turbo/transformer/
    diffusion_pytorch_model-0000{1,2,3}-of-00003.safetensors     (22.93 GiB fp32)
    diffusion_pytorch_model.safetensors.index.json               name -> shard
            |
            |  build_bfp16_weights.py --index ... --shards ... --manifest skeleton_manifest.json
            v
    weights.bfp16                                                 6,913,965,970 B
            |
            |  verify_bfp16_weights.py --file weights.bfp16 --index ... --shards ...
            v
    pass: every packed tensor round-trips within 2^-6 relative error
```

`weights.bfp16` is then fronted by two derived caches that the product actually
loads at run time — neither is in this repository, both are rebuilt by the
product itself:

| cache | size | built by |
|---|---|---|
| `webui/cache/wpack/wpack_l30_v2.bin` | 1.99 GB | `webui/bridge.py --pack-prewarm` (started automatically by `webui/runner.py`) |
| `brepack_30_k32_*.bin` | 10.6 GB | `imaggers/block/bigm/brepack_cache.py`, `prod_repack.py` |

## Files

| file | role | re-runnable from this repo? |
|---|---|---|
| `build_bfp16_weights.py` | **the converter.** Lays the file out arithmetically (no reads), writes the header, then streams each shard's tensors through `pack_bfp16.pack` | yes, given the upstream model |
| `verify_bfp16_weights.py` | **the verifier.** Decodes every tensor back and compares against the fp32 source | yes, given both |
| `skeleton_manifest.json` | the plan that produced the shipped file: `img_tokens`, `cap_tokens`, `shards`, `plan` | it *is* the input |
| `weight_map.json` | per-shard tensor pairs (`checkpoint` name, `shape`) — what the converter iterates | it *is* the input |
| `skeleton_verify_all.json` | the plan's own verification record | — |
| `build_skeleton_manifest.py` | derives `skeleton_manifest.json` from a compiled skeleton dir | **no** — needs `onnx` and the 49 MB `_probe_out/skeleton_all` tree |
| `extract.py` | a different artifact: one block's tensors as `block*.npz` | yes, given the upstream model |
| `manifest.json` | `extract.py`'s output manifest for one block (15 tensors) | — |

## The acceptance criteria

| value | expected |
|---|---|
| file bytes | 6,913,965,970 |
| **whole-file sha256** | **`4ff5b6a66ea3b3ae5a29ecc601097d50eade61979c222f8e0c58d3a272cb8042`** |
| header bytes | 87,946 (after the 8-byte little-endian length) |
| payload bytes | 6,913,878,016 |
| payload sha256 | `e3fc7e5b2b46744374c2df25d317561dd835243e5c872252ec8014b59a2c55ca` |
| tensors | 506 — 270 packed, 236 raw |

Re-run and compare with `sha256sum`; anything else means the upstream shards or the
codec changed.

## Three things that are easy to get wrong

* **The header goes first.** An earlier version wrote it after the payloads and
  overwrote the first 88 KB of payload — the header is larger than the 8 bytes
  reserved for its length prefix. It surfaced much later as a short read on a late
  tensor plus garbage exponents on unpack.
* **`tile_rows` is not a free parameter.** It comes from `pack_bfp16.KNOWN_TILE_ROWS`
  keyed by `(N, K)`, and it was *measured* against a vendor `.rai`. Getting it wrong
  changes the byte order inside every tile and the file still decodes — just wrongly.
* **Only 2-D tensors are packed.** Norms, biases and adaLN vectors stay fp32 and are
  copied verbatim; that is why 236 of the 506 records have `packed_bfp16: false`.

## Provenance

`pack_bfp16.py` at the repository root reproduces a vendor-produced `.rai` byte for
byte over 4,915,200 / 4,915,200 groups, which is what makes this conversion
trustworthy rather than merely plausible. See `docs/guides/BFP16.md`.