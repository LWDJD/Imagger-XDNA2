# BREPACK-CACHE-REPORT

**What was removed:** the 5.31 GB big-M B repack that every run re-derived from the
weights. It is now a content-addressed file that is written once and mmapped
afterwards -- the same treatment *wpack* already gives the projection images.

**Result, 512x512, 8 steps, delivered harness, measured back-to-back on
2026-11-02 between 15:55 and 16:04 (one launcher at a time, device lock and
foreign-python checked before each launch):**

| run | tag | window | cache | total_s | loop_s | startup_s | B repack | latents sha12 |
|---|---|---|---:|---:|---:|---:|---|
| baseline | bt_r1_base | 15:55:10-15:57:32 | off | **139.97** | 94.90 | 42.45 | 23.02 | 2abf9133 |
| cold / miss | bt_r2_miss | 15:59:13-16:01:27 | on | **132.62** | 100.22 | 30.06 | 10.58 | 2abf9133 |
| warm / hit | bt_r3_hit | 16:02:25-16:04:17 | on | **110.46** | 94.26 | 13.72 | 0.00 | 2abf9133 |

* warm hit vs baseline: **-29.51 s of total_s** (startup -28.73 s, loop -0.64 s)
* cold miss vs baseline: **-7.35 s** even though it did the whole repack *and*
  wrote 4.94 GiB, because the build path now uses a vectorised
  b3_to_image_k32 (10.58 s instead of 23.02 s)
* the box is ~15-20 % slower today than at 12:00 (baseline loop 94.90 vs 82.90
  and 79.21 for the two 12:0x arms; repack 23.02 vs 18.61 / 18.95), so compare
  the deltas in this table, not the absolute totals

## 1. What was actually being recomputed

prod_loop_bigm.cached_payloads -> repack_cache_inplace_bigm, on the first FFN
call of the run, per layer, over the product's resident --cache-all rows:

| step | in | out | measured, 1 layer |
|---|---|---|---:|
| _bg_unpermute(BG, nh=5) | 88,473,600 B | 88,473,600 B | 62.2 ms |
| bg_to_image(BG) | 88,473,600 B | 88,473,600 B | 39.8 ms |
| b3_to_image(B3) | 44,236,800 B | 44,236,800 B | 16.9 ms |
| b3_to_image_k32(B3) | 44,236,800 B | 44,236,800 B | **297-535 ms** |
| x 30 layers | 3.98 GB of sources | **5.31 GB of new bytes** | 5.35 s CPU, **23.02 s in-product** |

Two facts fall out of that table:

1. b3_to_image_k32 is a 614,400-step Python loop over a pure 72-byte-unit
   permutation. It alone was ~13 s of the 23 s. The vectorised transpose in
   brepack_cache.py is byte-identical to it (probe-checked on every build, and
   the full-cache word multiset + layer-0 byte histograms agree).
2. _bg_unpermute(permute_bg(x, 5), 5) == x **exactly** (selftest). The NH-5
   round trip in the cache-all pass is therefore a no-op for this consumer --
   1.5 s of permute plus 1.9 s of unpermute of pure waste. Nothing was changed
   for this; it is reported only.

## 2. Cache design

    +0            64 KiB   header: UTF-8 JSON (identity, content sha256, multiset)
    +64 KiB       BG   : 30 x 88,473,600   image order (gu / swiglu1)
                  B3   : 30 x 44,236,800   image order (ffn_dn_rt1)
                  ALT  : 30 x 44,236,800   second down generation (k32 here)
    total         5,308,481,536 B = 4.94 GiB

Every section offset is 64 KiB aligned (88,473,600 = 1350 x 65536, 44,236,800 =
675 x 65536), so each one is a legal np.memmap offset on Windows. The maps are
opened mode="r+" because pyxrt's bo.write takes the buffer protocol and the
delivered path hands it a writable ndarray view -- the same reason
webui/bridge.py:595 opens the wpack file r+.

Location: **imaggers/brepack_cache/brepack_30_k32_0822d4717f8d1887.bin**
(+ src_idmap.json, 7 KB). D: free space 15.53 GB before the hit run, 15.52 GB
after: a hit writes nothing.

A hit costs identify 0.14 s + header 0.000 s + verify 2.49 s = 2.63 s, and it
also removes the 4.4 s --cache-all read+permute, because the mapped rows are
already in image order and 3.71 GB of anonymous RAM is never allocated.
>
> Terminology: "mapped" in this report always means the payload is *mmapped from the
> cache file* -- `T22_BREPACK_CACHE=1`, the side-car's `cache-map`. The driver's log
> word "B order: mapped ... repacked in place" means the reverse arm, `inplace`
> (`T22_BREPACK_CACHE=0`); the two must never be read as the same state.

### Why it cannot silently re-encode

The bytes are BFP16 and the stored blocks are NOT canonical (0.297 % of bytes
differ from any re-encode), so the cache must be a byte-for-byte PERMUTATION.

1. **Identity.** The filename is a sha256 over the sha256 of all 60 source .npy
   files, the source digests of the five permutation functions
   (bg_to_image / b3_to_image / b3_to_image_n64 / b3_to_image_k32 /
   _bg_unpermute), and the geometry in force (layers, sizes, nh_bg, dual, alt
   kind). Change any of them and the key changes, so a stale file is never even
   opened. The full key is repeated in the header and re-derived on every load.
   The source digests are memoised by (name, size, mtime_ns), so a warm load
   pays 60 stat() calls and no hashing; a changed source re-hashes 3.98 GB at
   ~14 GB/s and the key changes.
2. **Multiset.** At build, every layer's output must have the SAME 64-bit word
   multiset (count, sum, and sum of w*K mod 2^64) as its input -- order
   invariant statistics that a permutation of 72-byte units preserves exactly
   and that a re-encode destroys -- and layer 0 must additionally reproduce the
   EXACT 256-bin byte histogram. On load the same statistics are recomputed
   from the mapped bytes and compared against the values recorded for the
   SOURCE. The whole-cache assertion runs over 663,552,000 words.
3. **Content.** A Merkle-1 sha256 over 16 MiB pieces of the content region
   (sha256 runs at 14 GB/s here thanks to SHA-NI) is stored in the header and
   recomputed on load, so any torn or edited byte is caught.

Build is atomic (.tmp + fsync + os.replace), the header is stamped last, and
any refusal on load is a MISS that falls back to the shipped path -- the cache
can never fail a run.

## 3. Env switches (all default off / harmless)

    T22_BREPACK_CACHE=1       enable                     (default 0)
    T22_BREPACK_DIR=<path>    where the file lives        (default imaggers/brepack_cache)
    T22_BREPACK_VERIFY=0      skip guards 2+3 on load     (default 1)
    T22_BREPACK_THREADS=N     verification threads        (default min(16, cpu))
    T22_BREPACK_BUILD=0       never build on a miss       (default 1)
    T22_BREPACK_WRITE_SYNC=1  write on the critical path  (default 0: daemon thread)
    T22_BREPACK_FASTK32=0     reference b3_to_image_k32   (default 1, self-checked)
    T22_BREPACK_PIECE=<bytes> verification piece size     (default 16 MiB)

CLI, CPU only, no device, no lock:

    python imaggers/block/bigm/brepack_cache.py --build     # prewarm out of band (26.8 s)
    python imaggers/block/bigm/brepack_cache.py --verify
    python imaggers/block/bigm/brepack_cache.py --stat
    python imaggers/block/bigm/brepack_cache.py --selftest  # PASS

## 4. Numerics

All three runs produced the SAME latents: sha12 **2abf9133 1200d069**, and they
are bit-identical (corr 1.000000000, max |diff| 0).

The pre-change reference for this exact config (T22_FASTGLUE=1 T22_REF_SMNB=1
ZIMAGE_CHUNK_THREADS=16) is 2abf9133, produced by the 12:01 arm
(webui/jobs/20261102-120108-3215d9) and reproduced bit-for-bit by my baseline
run with the cache OFF.

The 1c3beea9 value quoted in the brief belongs to the OTHER arm of that
12:01/12:03 pair: bt3 differs from at3 at corr 0.997544 (max |diff| 1.76), which
is the T22_REF_SMNB refiner-fused-softmax A/B (documented as the corr 0.9953
tier). So 1c3beea9 is the SMNB-off arm, not a regression, and not this config.

## 5. Costs and how to avoid them

* A MISS writes 4.94 GiB on a daemon thread while the loop runs: measured
  9.70 s of write and +5.3 s of loop (step 0 22.9 s vs 14.7 s). A miss happens
  once per weight set. To take it off the critical path entirely, prewarm with
  --build (26.8 s standalone, 2.2 sources + 6.7 repack + 12.5 write + 1.9
  re-verify) or set T22_BREPACK_BUILD=0 and build out of band.
* The hit's verify is 2.49 s in-product vs 1.31 s standalone (it overlaps the
  refiner weight loads). It is a prefetch thread started right after the device
  opens, so most of it is hidden already. T22_BREPACK_VERIFY=0 removes the rest
  and the integrity check with it.
* Disk: one file per (weight set, alt kind). 4.94 GiB each. The n64 build made
  earlier today was deleted once the k32 build (the generation this config
  dispatches) existed.

## 6. Files

**New**

| file | what |
|---|---|
| imaggers/block/bigm/brepack_cache.py | the cache module (build / verify / map / CLI) |
| imaggers/devkill/bench_brepack.py, bench_brepack2.py | what the repack costs, per stage and whole |
| imaggers/devkill/bench_hash.py, bench_hist.py, bench_fp.py | verification throughput measurements |
| imaggers/devkill/cmp_latents.py | the latents A/B against both historical arms |
| imaggers/brepack_cache/brepack_30_k32_0822d4717f8d1887.bin | the cache, 4.94 GiB |
| imaggers/brepack_cache/src_idmap.json | source digest memo, 7 KB |

**Modified, env-gated, default off**

| file | change |
|---|---|
| imaggers/block/bigm/prod_loop_bigm.py | +33 lines in cached_payloads(): ask brepack_cache first, fall through to repack_cache_inplace_bigm |
| imaggers/image/t18_run.py | +52 lines: _brepack_prefetch() helper, one call after the device opens, one adopt block at the --cache-all site |

Neither changes a byte the device sees when T22_BREPACK_CACHE is unset: the
helper returns None and cached_payloads takes the shipped branch. block_step()
and its source are untouched, so the fastglue inspect.getsource gate is
unaffected. webui/** untouched.
