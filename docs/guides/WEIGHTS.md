# Weights: where they come from and the two chains that consume them

Everything here is measured; the machine-readable form with full SHA-256 per file
lives in `imaggers/par/provenance/artifacts.json` (15 records, 30.6 GB hashed in
155 s at ~1.2 GB/s). The hashes were validated two ways: the wpack hash reproduces
what the packer itself logged, and the 30 block hashes reproduce
`extract_manifest.json` 28/28.

## 1. The two chains

The checkpoint feeds **two independent** consumers. They diverge at the upstream
model and never meet.

```
Z-Image-Turbo transformer shards (~23.5 GB, 3 safetensors)
        |
        |  build_bfp16_weights.py
        v
   weights.bfp16                     6,913,965,970 B
        |
        |  bridge.py --pack-prewarm        (PURE FUNCTION of weights.bfp16)
        v
   webui/cache/wpack/wpack_l30_v2.bin 1,990,656,000 B   ~33 s to build, 1-2 s to map

        ALSO, from the same shards:
        |
        |  weights/extract.py + extract_all.py
        v
   weights/block0.npz .. block29.npz   34 files, 6,586,939,269 B
        |
        |  s2_ffn32.load_blocks
        v
   brepack_cache/brepack_30_k32_<key>.bin   2 files x 5,308,481,536 B
```

**The wpack is not derived from `block*.npz`.** `bridge.py:354-355` calls it "the
only input of the pack" and `bridge.py:584-585` refuses the artifact if
`weights.bfp16`'s (path, size, mtime) stamp changed. Do not "fix" a wpack by
rebuilding the npz files — they are on different branches.

The text encoder is a separate chain entirely:

```
text_encoder shards (8.05 GB, present on the reference box)
        |  te_npu_pack.py              -> _out/textenc/te_npu_weights.bin   4.09 GB
        |  convert_hf_to_gguf.py       -> webui/models/*.gguf               8,051,284,800 B
```

## 2. Two constraints that must not be relaxed

### 2.1 Never re-encode the BFP16 payload

0.297 % of the stored bytes differ from what any re-encode would produce. The code
guards this three ways in `brepack_cache.py`:

* a 64-bit word multiset comparison (:326-343)
* a layer-0 256-bin value histogram (:378)
* a Merkle-1 tree over 16 MiB pieces (:348)

A re-encode still *looks* right — it decodes to nearly the same floats — and it
silently changes the last bits of every layer. Do not do it.

### 2.2 The B-repack cache key is over the payload ORDER

`brepack_cache.py:230-318`, `identity_key()` = SHA-256 of 60 source `.npy` files
plus the source digests of 5 permutation functions plus the geometry. A cache built
in a different order gets a different key and different bytes.

The consequence is not cosmetic: layer 0 moves by 1 f32 ULP, that amplifies ~1.55x
per layer, and by step 8 the latents are **4.19e-2** apart — an order of magnitude
more than the project's provenance floor. The shipped key's latents are
`2abf91331200d069…`.

## 3. Sizes and file counts (measured, not quoted)

| artifact | files | bytes | rebuilt by | in git? |
|---|---:|---:|---|---|
| `weights/block*.npz` | 30 | 6,378,809,xxx | `weights/extract_all.py` | no |
| `weights/` total | 34 | 6,586,939,269 | — | no |
| `weights.bfp16` | 1 | 6,913,965,970 | `build_bfp16_weights.py` | no |
| `wpack_l30_v2.bin` | 1 | 1,990,656,000 | `bridge.py --pack-prewarm` | no |
| `brepack_30_k32_*.bin` | 2 | 10,616,963,072 | `brepack_cache.py --build` | no |
| `webui/models/*.gguf` | 1 | 8,051,284,800 | `convert_hf_to_gguf.py` | no |
| **device images** | **66** | **9,215,xxx** | MLIR-AIE + Peano | **yes** |

`block0..9` are 212,626,930 B and `block10..29` are 212,626,988 B. The 58-byte
difference is only the longer layer-index string in the zip member names; the
uncompressed payload is identical at 212,621,984 B.

Note that **only one** of the two brepack files is ever resolved — the
`...d34e7a7a5fcb02ac.bin` sibling. The other is a second `alt_kind`. Budget for both
if you rebuild, but a run needs one.

## 4. THE BLOCKER: the upstream transformer shards are gone

On the reference box, `zimage-port/_models/Z-Image-Turbo/transformer/` now holds only
`config.json` (473 B) and the index JSON. **No `*.safetensors` over 1 GB exists
anywhere under `public_work`.**

Therefore, on that box:

* `weights/block*.npz` (6.38 GB) **cannot be rebuilt**
* `weights.bfp16` (6.91 GB) **cannot be rebuilt**
* the on-disk copies are **the only copies**

The text-encoder shards *are* present, so the GGUF and the 4.09 GB text pack remain
rebuildable there.

**Action required before anything is deleted:** either archive those two artifacts
elsewhere, or re-download the upstream Z-Image-Turbo transformer shards from
Hugging Face. Everything downstream of them can then be regenerated.

## 5. What a fresh clone must do

Ordered, tagged by what it needs:

1. **[CPU]** install the interpreter pair and the Python packages — see
   [BUILD.md](BUILD.md)
2. **[NET]** download the text encoder GGUF to `webui/models/`
   (8,051,284,800 B). The WebUI refuses to start without it.
3. **[NET]** obtain `weights.bfp16` (6,913,965,970 B) — the wpack's only input
4. **[CPU]** build the wpack: `python webui.py` warms it, or
   `bridge.py --pack-prewarm`; ~33 s to build, 1-2 s to map afterwards
5. **[NET]** obtain `weights/block*.npz` (34 files, 6,586,939,269 B)
6. **[CPU]** build the B-repack cache once: `python imaggers/block/bigm/brepack_cache.py --build`
7. **[NPU]** run the preflight, then generate

A **CPU-only smoke test** can skip steps 3, 5 and 7 and still exercise the planner,
the capability endpoint, the shape gates and the WebUI surface — but it cannot
produce an image, because the device images need the wpack and the B payloads.

## 6. Open questions

Four things this document does **not** settle, each with the experiment that would:

| question | how to settle it |
|---|---|
| Which XRT revision the driver needs — `ENVIRONMENT.md` and `webui/config.py:24` tell different stories | `xrt-smi examine --report` plus hashes of the three `xrt_coreutil.dll` copies |
| Whether `rebuild_te_gguf.ps1` really reproduces the 8.05 GB GGUF | run it; script and inputs are both present |
| Which chain produced `t3real_prep` (its docstring says `weights.bfp16`; several harnesses reuse the dir) | instrument the writer |
| Whether the two brepack files differ only by `alt_kind` | compare their headers and a sampled payload window |
