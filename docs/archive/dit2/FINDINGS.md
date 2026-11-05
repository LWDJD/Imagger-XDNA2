# FINDINGS — dit2: two-Worker RMSNorm -> DMA corner turn -> BFP16 GEMM

Working directory: `zimage-bfp16-npu\_scratch\dit2\`.  Files:
`two_worker.py` (the design), `run.py` (static proofs + compile + prep +
verify, Python 3.12), `run_xrt.py` (device run, Python 3.10/pyxrt),
`log_*.txt` (full transcripts), `out\` (xclbins, insts, io npz, raw C).

## RESULT

| criterion | status | evidence |
|---|---|---|
| MINIMUM: two-Worker design compiles to an xclbin | **PASS** | `out\tw_M32_K32_N32_m16_k8_n16_d22.xclbin` 14-17 KB + 660 B insts; every K from 32 to 1792 compiles (~9-13 s warm aiecc) |
| TARGET: runs with `ERT_CMD_STATE_COMPLETED`, matches a numpy reference | **PASS** | 2/2 runs `ERT_CMD_STATE_COMPLETED` at every K tried (32, 64, 128, 256, 512, 1024, 1536, 1728, 1792), run-to-run byte-identical, and **100.00% of output elements bit-exact** against the numpy RMSNorm-then-matmul reference (per-call-bf16 model, see section 3) |
| STRETCH: largest hidden dim that compiles and runs | **K = 1792** (with M=32 tokens, N=32, tile (m,k,n)=(16,8,16), fifo depths 1/1), bit-exact.  K=1856 is the first L1 refusal; K=2048 and K=3840 refuse on the DMA BD length cap (exact texts in section 4) |

## 1. The architecture

```
host A (M tokens x K hidden, bf16 row-major, one ROW per transfer)
   |
   v  inA fifo (row objects, K*2 bytes)
Worker A "norm core"  @ (0,2):  rms_norm(tile_size=K), one contiguous row
   |                            per call, rows written into the midA object
   v  midA fifo (object = m*K*2 i8), to_stream = corner_turn_tap(m, K)
   |       *** THE WALK IS THE CORNER TURN ***
   v
Worker B "GEMM core"  @ (0,3):  zero(C); for kk in K/8:
   |                            mm_bfp(m, 8, n, mixed=True) on slice kk
   v  memC fifo -> memtile forward hop (to_stream = stream_dims.C) -> host
host C (M x N, bf16 row-major)
```

* **Why per-GEMM k == 8.**  `mm_bfp(mixed=True)` reads its A operand in the
  (r,s,t)=(8,8,8) micro-tile layout; at k == s == 8 one (m, 8) A tile's
  block layout is *exactly row-major* (verified: `full((m,8)).tile((8,8))`
  gathers `arange(m*8)` unchanged), so each hidden slice of the normed tile
  is contiguous and needs no intra-tile shuffle.  The whole corner turn
  then collapses to permuting **(m, K/8, 8) -> (K/8, m, 8)**.
* **The corner turn tap** (byte space of the i8 object):
  `TensorAccessPattern((m, 2K), 0, [K//8, m, 16], [16, 2K, 1])` — stream
  order is `for kk: for r: row r's 16 bytes of slice kk`, i.e. Worker B
  finds slice kk contiguous at byte offset `kk*m*k*2` and that slice is in
  mm_bfp's A layout.  For k > 8 the same walk needs 5 dims (kk, br, bs, i,
  j) and slices stop being contiguous — k == 8 is asserted in the design.
* Worker B computes slice offsets dynamically with `arith.muli` on the
  `range_` loop index (a Python-unrolled 480-iteration loop at K=3840 would
  overflow core instruction memory; the emitted instruction stream is
  660 B / 165 words at every K — the loops are device loops).
* Host B is **packed (jj, kk, row) block order by `pack_b_blocks`** (NOT
  `bfp.shuffle`, see refusal 2) and tapped with the flat
  `full((K*N//8,)).tile((n,))` tap dit_block_v1 proved.
* The 4-dim C inverse walk runs on a **memtile forward hop**
  (`memC.cons().forward(to_stream=matmul_kernel.stream_dims.C)`), exactly
  as dit_block_v1, because compute-tile DMA BDs take at most 3 dims
  (refusal 1).
* `inA` carries ONE ROW per object so the norm core's L1 holds only the
  midA object plus a couple of `K*2`-byte rows (this is what extends the
  run from K=512 to K=1792 at the same fifo depths).

## 2. Static proofs of the walks (host, before compiling)

`run.py` refuses to compile unless all of these hold (`[static]` lines in
every log):

1. `full((m,8)).tile((8,8))` gathers the identity — mm_bfp's A layout is
   row-major at k == 8.
2. The corner tap gathers exactly the (m, K/8, 8) -> (K/8, m, 8)
   permutation, in both element and byte space, and every stream slice is
   row-major (m, 8) = the kernel's A layout.
3. DMA granule rules: corner-tap innermost 16 B, A host tap 2K B, B host
   tap 9n B — all multiples of 4.
4. `pack_b_blocks`: flat block (jj*K_div_k + kk)*n + r decodes to
   W^T[jj*n+r, kk*8:(kk+1)*8] (bfp decode of individual 9-byte blocks).
5. `bfp.shuffle` (with per-core tile (k,n)) puts (row r, block kk) at flat
   `r*K//8 + kk` — i.e. the kk step is one 9-byte block, which is why the
   design does not use it (refusal 2).
6. The B tap delivers transfers ((jj, kk) kk-fastest, then rows) with the
   n contiguous blocks of each GEMM tile.

## 3. Numerics: the reference must round C to bf16 after EVERY call

`mm_bfp_mixed.cc` does per call: `accC = aie::load_v<bfloat16>(pC)` ->
MAC over colA micro-tiles -> `aie::store_v(pC, accC.to_vector<bfloat16>())`.
C is **re-rounded to bf16 after every GEMM call**, so a K/k-call
accumulation carries K/k intermediate bf16 roundings.  The numpy reference
(`run.py: prep_data`) reproduces that exactly (`ref_model`: f32 accumulate
of the per-call quantized products, then a bf16 round-trip per call).

Measured against `ref_model`, the device is **bit-exact (100.00% of
elements, max abs err 0.0) at every K from 32 to 1792**.  Against the f64
sum of the same quantized products (`ref_quant`, which ignores the
intermediate stores) the distance grows with K/k exactly as the model
predicts (max ulp distance: 1 @K=32, 2 @K=64, 3 @K=128/256, 7 @K=1024,
10 @K=1536, 13 @K=1792) — that spread is the algorithm's own quantization,
not device error.

NOTE for tolerance users: dit_block_v1's `_BFP_MIXED_TOLERANCE` check
`range_frac <= 2 * 2^-8` is magnitude-blind and fails here at every K
(measured range_frac 1.19e-2 @K=32 to ~1e-1 @K=1792): the residual is the
bf16 *store* step, which is 2^-8 of the value's magnitude (ulp = 2.0 at
output 290).  The magnitude-aware statement is the ulp-vs-model one above.

## 4. Every refusal, with exact error text

1. **Compute-tile DMA BDs take at most 3 dimensions** (the C inverse walk is
   4).  First compile of the direct worker->shim outC fifo:
   ```
   ...\tw_M32_K32_N32_m16_k8_n16.prj\aie.mlir:11:5: error: 'aie.dma_bd' op
   Cannot give more than 3 dimensions for step sizes and wraps on this tile
   (got 4 dimensions).
       aie.objectfifo @outC(%logical_core_0 dimensionsToStream [<size = 2,
       stride = 128>, <size = 8, stride = 8>, <size = 2, stride = 64>,
       <size = 8, stride = 1>], {%logical_shim_noc_2}, 2 : i32) : ...
   ```
   Fixed by moving the 4-dim walk to a memtile forward hop (dit_block_v1's
   structure).  The corner-tap itself is 3 dims and is accepted on the
   compute tiles.  (Shim/memtile hops accept 4 dims: dit_block_v1's taps
   `[2, 1, 16, 1]` and the C walk ran there.)

2. **A bfp16ebs8 stride must be a multiple of 4 bytes** — the shuffled B
   layout steps one 9-byte block between hidden slices:
   ```
   ...\tw_M32_K32_N32_m16_k8_n16.prj\aie.mlir:170:9: error: 'aie.dma_bd' op
   Stride 1 is 1 elements * 9 bytes = 9 bytes, which is not divisible by 4.
       aie.dma_bd(%arg1 : memref<128x!aiex.bfp<"v8bfp16ebs8">> offset = 0
       len = 128 sizes = [1, 2, 4, 16] strides = [0, 64, 1, 4])
   ```
   Fixed by packing host B in (jj, kk, row) block order so every transfer
   is n contiguous blocks (9n bytes; n % 16 == 0 -> divisible by 4) with
   the flat tap.  (This is the same 9-byte rule as dit_block_v1's finding 4,
   "Transfer sizes must be multiples of 4 bytes. 1 elements at 9 bytes each
   equal 9 bytes, which is not divisible by 4", but on a *stride*.)

3. **L1 is 64 KB per AIE2P compute tile (4 banks x 16 KB)** — buffer
   placement refusals, exact texts:
   ```
   (K=512, depths 2/2, (m,K) inA objects -- before the per-row change)
   ...\tw_M32_K512_..._d22...:14:5: error: 'aie.buffer' op could not be
   placed: buffer "midA_buff_1" needs 16384 bytes and this tile has no
   room left for it
   note: Current configuration of buffers in bank(s) : MemoryMap:
       bank : 0  0x0-0x3FFF
           (stack)          : 0x0-0xEFF  (3840 bytes)
       bank : 1  0x4000-0x7FFF
           midA_buff_0      : 0x4000-0x7FFF  (16384 bytes)
       bank : 2  0x8000-0xBFFF
           inA_cons_buff_1  : 0x8000-0xBFFF  (16384 bytes)
       bank : 3  0xC000-0xFFFF
           inA_cons_buff_0  : 0xC000-0xFFFF  (16384 bytes)

   (K=1856, depths 1/1, per-row inA -- first refusal of the final design)
   error: 'aie.buffer' op could not be placed: buffer "inA_cons_buff_0"
   needs 3712 bytes and this tile has no room left for it

   (K=1920, depths 1/1)
   error: 'aie.buffer' op could not be placed: buffer "memC_buff_0" needs
   512 bytes and this tile has no room left for it
   ```
   K=1792 is the largest that places.  Per-tile accounting at depths 1/1:
   worker A holds midA `m*K*2` + one `2K`-byte inA row + the 3840 B stack =
   64,768 B of 65,536 B at K=1792 (768 B slack); worker B holds midA +
   stack + inB + memC ~ 62.5 KB.  At K=1856 worker A alone needs
   59,392 + 3,712 + 3,840 = 66,944 B -> refused (the error names the
   inA row buffer because it is placed last).

4. **A single DMA BD transfer is capped at 16383 32-bit words (65,532 B)**
   — this is what refuses hidden 2048 and 3840 outright (the (m,K) midA
   object is one transfer):
   ```
   (K=2048)  ...\tw_M32_K2048_...:14:5: error: 'aie.dma_bd' op buffer
   descriptor length (16384 32-bit words) exceeds the maximum of 16383
   words supported by this tile type

   (K=3840)  error: 'aie.dma_bd' op buffer descriptor length (30720 32-bit
   words) exceeds the maximum of 16383 words supported by this tile type
   ```
   Note the Z-Image hidden dim 3840 would need `m*K*2` = 122,880 B in ONE
   transfer — over both this cap and the whole L1.

Other API refusals inherited from dit_block_v1 (unchanged, see its
FINDINGS): `memref.view` needs a 1-D i8 source (rows/slices are views of
raw i8 objects); MLIR types must be built inside `core_fn`; the
`os.mkdir(0o700)` machine bug (mitigated by the process-local mkdir shim +
`use_cache=False`, copied from `dit_block_v1_run.py`); `xclbinutil` is
off-PATH (driver prepends
`D:\Project\deepseek_harness\test7\WStmp\driver376\npu_mcdm_stack_prod`).

## 5. Sweep data (M=32 tokens, N=32, tile (16,8,16), seed 0)

| hidden K | depths | compile | run | vs ref_model | note |
|---|---|---|---|---|---|
| 32  | 2/2 | 7.4 s | 2/2 COMPLETED | **100% bit-exact** | step 1 proof point |
| 64  | 2/2 | 9.1 s | 2/2 COMPLETED | **100% bit-exact** | |
| 128 | 2/2 | 9.5 s | 2/2 COMPLETED | **100% bit-exact** | |
| 256 | 2/2 | 9.3 s | 2/2 COMPLETED | **100% bit-exact** | |
| 512 | 2/2 | 9.3 s | 2/2 COMPLETED | **100% bit-exact** | per-row inA needed (refusal 3a) |
| 1024 | 1/1 | 9.4 s | 2/2 COMPLETED | **100% bit-exact** | depths 1/1 needed |
| 1536 | 1/1 | 12.9 s | 2/2 COMPLETED | **100% bit-exact** | |
| 1728 | 1/1 | 10.2 s | 2/2 COMPLETED | **100% bit-exact** | |
| **1792** | 1/1 | 9.4 s | 2/2 COMPLETED | **100% bit-exact** | **largest that runs** |
| 1856 | 1/1 | FAIL | — | — | L1 (refusal 3b) |
| 1920 | 1/1 | FAIL | — | — | L1 (refusal 3c) |
| 2048 | 1/1 | FAIL | — | — | BD length (refusal 4) |
| 3840 | 1/1 | FAIL | — | — | BD length (refusal 4) |

Representative device numbers (K=1792, depths 1/1):
`A 114,688 B  B 64,512 B  C 2,048 B`; insts 660 B (165 words); run walls
0.68 ms / 0.34 ms; xclbin 13,392 B; run-to-run byte-identical; C absmax
12458.0 matching ref absmax 12458.03.

## 6. What this proves, and the path to hidden 3840

* **Proven**: the two-Worker shape works end-to-end; the DMA
  `to_stream` walk performs the corner turn exactly as hypothesized
  ("that walk IS the corner turn" — measured, section 2); the fused
  RMSNorm -> mixed BFP16 GEMM is bit-exact against the numpy model at
  hidden up to 1792 on one worker pair.
* **Not proven / out of reach of the direct two-worker hop**: hidden
  3840.  The midA object is (m, K) and must live in L1 at *both* workers;
  at K=3840, m=16 that is 120 KB per side, over the 64 KB tile L1 *and*
  over the 65,532 B BD transfer cap.
* **Recommended architecture for 3840** (uses only existing API, none of it
  verified here): put the (m, K) tile in a **memtile** (512 KB) — norm core
  streams ROW objects to a `join` at the memtile with row offsets, the
  memtile's DMA applies `corner_turn_tap` and the fifo uses
  `consumer_obj_type = (m*k*2,) i8` (the ObjectFifo's asymmetric transfer
  granularity) so the GEMM core receives ready-made (m, k) slice objects
  and never holds the (m, K).  The norm core then needs only K*2-byte rows
  and the GEMM core only 256-byte slices.  Two caveats to check next: the
  join's per-input transfer sizes against the corner walk
  (`_check_walk_size` runs `link._transfer_sizes` on it), and the memtile's
  own 16383-word BD cap on the join buffer (2 x 122,880 B objects at depth
  2 fit its 512 KB).

## 7. Reproduction

```
"..." = "D:\Users\15081\miniforge3\envs\ryzen-ai-1.8.0\python.exe"   (3.12)

"..." zimage-bfp16-npu\_scratch\dit2\run.py -K 32                (all phases)
"..." zimage-bfp16-npu\_scratch\dit2\run.py -K 1792 --inA-depth 1 --mid-depth 1
"..." zimage-bfp16-npu\_scratch\dit2\run.py -K 32 --skip-compile --skip-run
                                                                (static proofs)
"..." zimage-bfp16-npu\_scratch\dit2\run.py -K 32 --verify-only (re-verify npz)
```

The device half is `run_xrt.py` under Python 3.10
(`D:\Users\15081\AppData\Local\GAIA\python\python.exe`), invoked
automatically by run.py.  Artifacts per config:
`out\tw_M32_K<K>_N32_m16_k8_n16_d<in><mid>.{xclbin,bin}` plus `_io.npz`
(inputs + references) and `_C_raw.npy` (raw device C).
