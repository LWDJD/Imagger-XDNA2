# Known issues and ruled-out paths

Recorded so nobody repeats work that has already been done.

## Open

### The memtile corner turn is PROVEN; the producer side is not

`imaggers/memtile/probe_cap.py` proved the load-bearing piece: host ->
`(m,K)=(16,3840)` i8 object (122,880 B = 30,720 32-bit words) -> 1:1 link at
memtile (0,1) -> `to_stream = corner_turn_tap` -> `consumer_obj_type=(m*k*2,) i8`
-> worker drains slices.  xclbin compiles (5.3 s), 2/2 `ERT_CMD_STATE_COMPLETED`,
byte-identical reruns, **100% bit-exact vs numpy at K=3840**.

Two assumptions that were wrong and are now settled:
* the memtile DMA BD cap is **NOT** 16383 words; 30,720-word BDs run.  The
  16383 cap is a **compute-tile** BD limit
* `'aie.objectfifo' op objectfifo cannot be in more than one ObjectFifoLinkOp`
  — a fifo cannot join AND feed a forward hop, so the walk cannot ride a hop
  after a join; and a join output's transfers are per-input-segment, so the
  walk can never live on a join output

**What is still open:** the *producer* side.  The agent failed mid-work on the
norm core's `outA` BD — the compute-tile BD cap — because at K=3840, m=16 the
`(m,K)` object is 122,880 B and the norm core's L1 is 64 KB, so the producer
cannot hold the object it must emit.  Candidate fixes, in order:
  1. a `delegate_tile` / memtile-resident producer pool so the norm core never
     holds `(m,K)`
  2. four norm cores each producing an `(m/4, K)` quarter object, joined later
  3. segment the producer BD into per-row-block transfers

Its `TensorAccessPattern` question (stride semantics vs `tensor_dims` for a
join-output walk) was never answered and should be read from
`objectfifo.py`'s `_check_walk_sizes` / `ObjectFifoLink._transfer_sizes`
before redesigning.

`imaggers/memtile/` holds `probe_cap.py` and the log; `memtile_worker.py` and
`run.py` were NOT produced.

### The DiT block weight layout is now KNOWN AND VERIFIED

`imaggers/weights/` extracted `layers.0` (15 tensors, 180,910,336 fp32 params)
and packed it to `bfp16ebs8`, with the **strongest verification available**: the
same pipeline reproduces the surviving vendor `.rai` payload
**416,071,680 / 416,071,680 bytes, 0 differing**, across all 28 sub-buffers.

Per block, in the vendor buffer the order is

    adaLN, to_q, to_k, to_v, to_out, w1, w3, w2   then the 6 consts

-- note **adaLN FIRST** and **w3 BEFORE w2**.  A block is **not contiguous** in
the buffer: the GEMM part sits at payload 0..203,612,160 and the 6 consts at
407,224,320..411,648,000.

Sizes, `tile_rows` verified at both seq=64 and seq=96:

    to_q/to_k/to_v/to_out.0   [3840,3840]   tr=48   16,588,800 B each
    feed_forward.w1/w3        [10240,3840]  tr=32   44,236,800 B each
    feed_forward.w2           [3840,10240]  tr=32   44,236,800 B
    adaLN_modulation.0.weight [15360,256]   tr=64    4,423,680 B
    adaLN_modulation.0.bias   [15360]                 17,280 B
    attention_norm1/2, ffn_norm1/2  [3840]             4,320 B each
    attention.norm_q / norm_k       [128]                144 B each

`tile_rows` is f(N,K,M) -- M-dependence is proven for (3840,256) (64 -> 48) and
for 256 -> 32.  **Re-measure with `pack_bfp16.detect_tile_rows` for any M other
than 64 or 96.**

Two corrections to assumed naming:
* there is no fused `ff.net.0.proj`; `pack(vstack(w1,w3), 32) == pack(w1) ++ pack(w3)`
  was MEASURED, so a fused [20480,3840] is just the two regions concatenated
* in blocks the adaLN Linear is index `.0`, but in the host `FinalLayer` it is
  `.1` -- do not derive one from the other

Also: `get_params.py`, `shard_inspect.py`, `skeleton_names_all()` and
`verify_rai_rebuild.py` DO NOT EXIST in this workspace despite being referenced
elsewhere; the extraction used `docs/WEIGHT-MAP.md`'s 506-name registry plus
safetensors header offsets.

### The only surviving vendor `.rai` has been backed up

`zimage-port\_probe_out\route3\` (which held `CLEAN_baseline.rai` and the other
controlled baselines) was deleted during the space reclaim.  The `fresh_cache`
artifact became the only surviving vendor `.rai`, so it is now copied to

    zimage-bfp16-npu\vendor_artifacts\39e41383350b8c53e90996269478cfb7.rai
    418,881,856 B
    sha256 dce407a3c5c5ffcf9a369c31309e29aa8af182711b814f0cbdb1936dcd412a10

Treat it as irreplaceable.  Provenance caveat is in `imaggers/weights/FINDINGS.md`
§3c (the `docs/WORKLOG` overwrite episode).

### `_probe_out` / `_models` are directory JUNCTIONS, not per-file hardlinks

Measured by the weight-regeneration agent with `Get-Item .Target`:

    zimage-port\npu-conversion\_probe_out  -> Global\D:\...\zimage-port\_probe_out
    zimage-bfp16-repo\_probe_out           -> same
    (_models is junctioned the same way)

so the three "copies" are ONE physical tree reached three ways; every file has
`st_nlink == 1`.  `hardlink_check.py`'s "inodes with >1 link" count was reading
the SAME inode through different paths and reporting it as sharing.

**Consequences:**
* deleting one path removes the content for all three; a second `rmtree` raises
  `FileNotFoundError [WinError 3]` because it is already gone
* the earlier "you must delete all three roots" note in this file was WRONG and
  has been withdrawn
* `_models` is junctioned the same way — never delete it

Reclaim actually measured: deleting `{route3, single, fresh_export, merge_probe,
p2_fold}` under `zimage-port\_probe_out` removed 111 files / 9,068,594,024 unique
bytes and freed 9,068,683,264 B (D: 5.155 -> 13.601 GiB).

### Disk reclaim requires deleting ALL THREE roots -- WITHDRAWN, see above

### `pack_bfp16`'s real API -- corrected after two subagents were given a wrong one

The signatures actually exported by `pack_bfp16.py` are

    pack(W: np.ndarray, tile_rows: int | None = None) -> bytes   # ONE array
    unpack(blob: bytes, N: int, K: int, tile_rows) -> ...
    KNOWN_TILE_ROWS = {...}            # measured tile_rows per shape
    detect_tile_rows(W, blob) -> int
    _quantise(v) -> (mantissa, exponent)
    _slots(N, K, ng, tile_rows=32) -> np.ndarray

There is **no** `quantize(x, *, dim=-1, dtype)`, **no** `unpack_and_quantize`,
and **no** `pack(arrays, *, dim, m, k, n, tile_rows)`.  Two subagents were
briefed with those invented names and had to be corrected mid-run.  Read the
module; do not trust a remembered API.

### `weights.bfp16` NO LONGER EXISTS -- it was deleted in the disk cleanup **(WITHDRAWN 2026-10-19: it EXISTS, 6,913,965,970 B -- see ERRATA E1 at the end of this file)**

`weights_audit.py` found `zimage-port/weights.bfp16` absent.  `docs/WORKLOG.md`
records "8.88 GiB of regenerable artifacts deleted" during the disk-pressure
fix, and the 6.4 GiB packed weight file was among them.

Consequences:
  * any work that assumed the file is present must regenerate it first
  * it must be regenerated with the CURRENT `pack_bfp16.pack` anyway, because
    the copy that was deleted predated the `_quantise` power-of-two fix
    (`e_unb - 6` unconditionally) and held wrong bytes for some groups
  * `imaggers/kernels/` and the block-weight extraction
    (`imaggers/weights/`) produce their own packed bytes and are unaffected

`weights_audit.py` also shows the shard discovery currently lands on the
`text_encoder` shard; when regenerating, target the `transformer` shards
(`layers_*`), which is where the DiT weights are.

### Disk: D: has 5.2 GiB free; 35.5 GiB is reclaimable if it runs out

`hardlink_check.py` measured the three apparent copies of `_models` and
`_probe_out`:

    _models      138 files   91.91 GiB summed   30.64 GiB unique   46 inodes, all 3-linked
    _probe_out   37,167      118.88 GiB summed  35.49 GiB unique   10,230 inodes, all 3-linked

So the trees ARE fully hard-linked (the claim in `docs/WORKLOG.md` checks out) and
the disk report's per-path summing over-counts by 3x.  Real unique use is
66.1 GiB, of which **`_probe_out`'s 35.49 GiB is regenerable probe output from
the `.rai` research**.

Do NOT delete `_models` -- it is the checkpoint.  If D: fills up, delete
`_probe_out` subdirectories (start with `route3`, `single`, `fresh_export`,
`merge_probe`, `p2_fold`, `packverify`) and record what was removed.  As of
this entry nothing has been deleted.

### VAE is a stock `AutoencoderKL` — and a scope decision

`zimage-port/ref_probe_env.py:63-73` loads it as

    AutoencoderKL.from_pretrained(ROOT, subfolder="vae")
    vae_scale_factor = 2 ** (len(block_out_channels) - 1) = 16

so it is a stock diffusers KL autoencoder with 5 resolution levels.  Nothing
model-specific.

**Scope decision, recorded so it is deliberate:** the decoder runs ONCE per
image and is far cheaper than the 34-block DiT.  For the first end-to-end image
the VAE can stay on the CPU while the DiT runs on the NPU with BlockFP16
weights.  That still puts the NPU on the part that matters and avoids blocking
the project on a decoder port.  Porting the VAE is a later step, and by then
the same kernel set (conv, GELU, group-norm, upsample) applies.

The sampler is `FlowMatchEulerDiscreteScheduler` (`ref_probe_env.py:68`) — a
flow-matching Euler step, short and cheap on any device.

### The core codegen recipe: `llc -O1 --regalloc=greedy`, injected — and the `-O0` build that LOOKS GREEN

**Explicitly authorised write** (parent agent, stage 4).  Source of every number
below: `imaggers/block/STEP2D.md` sections 0-5 and `imaggers/block/STEP4.md`
sections 6.2 and 8.

**A plain `aiecc -O0` build of the stage-2 attention design COMPLETES 2/2
`ERT_CMD_STATE_COMPLETED` and is WRONG and NOT run-to-run byte-identical.**
Measured on the real NPU (M=96, K=96, heads=3, head_dim=32, m=8):

    run 0: wait() -> ERT_CMD_STATE_COMPLETED   wall 110.35 ms
    run 1: wait() -> ERT_CMD_STATE_COMPLETED   wall 110.22 ms
    run-to-run byte-identical (Z and x_out): False
    [verify] x_out vs composed device-arithmetic model: max abs err 1.087891
    [verify] PASS (<= 2 bf16 ulps ...): False

This is the worst kind of failure: it looks green.  A completion check alone
cannot certify a build; the acceptance triple is **audit clean + 2/2
`ERT_CMD_STATE_COMPLETED` + run-to-run byte-identical**, and the plain `-O0`
build fails the third and the arithmetic.

**The correct codegen is `llc -O1 --regalloc=greedy`, injected.**  One command
sequence (the same recipe at stage 2 and stage 4):

    PY=D:\Users\15081\miniforge3\envs\ryzen-ai-1.8.0\python.exe
    $PY step2d_run.py --probe 0 --skip-static        # build (aiecc -O0)
    $PY step2d_build_win.py --level=-O1 --out step2d_win   # inject llc -O1
    $PY step2d_run.py --probe 0 --skip-compile --runs 2    # device acceptance

`step2d_build_win.py` (and its stage-4 copy `step4_build_win.py`, identical
recipe, different output-dir constants) does: `aiecc --cut="objects_{0}.o"
--checkpoint=<dir>` -> swap
`<dir>/objects__0_.o/objects_main_core_0_2/objects_main_core_0_2.o` with
`llc <prj>/opted_main_core_0_2.ll -O1 --march=aie2p --function-sections
-stack-size-section -aie-stack-addrspace=5 --filetype=obj --regalloc=greedy`
-> `aiecc --resume=<dir>/manifest.json`, with
`test7\WStmp\driver376\npu_mcdm_stack_prod` on PATH.

Measured with the injection (stage 4, M=96 K=96 h=3 hd=32): 2/2
`ERT_CMD_STATE_COMPLETED`, 152.04/152.32 ms, **run-to-run byte-identical
TRUE**, core object 5,728 B, x_out vs model max abs err 3.125e-02, Z 2.441e-02
— and the final-ELF audit is `6 acquires, 0 ACQ clobber` (**PASS**).

**The "-O1 wall" was NOT llc.**  aiecc's IR-level `opt -O1` blows the IR up
(measured cell I1: 17 acquires after loop unrolling, 8,240 B core, 15 acquire
exposures; cell I2 `opt -O1` + `llc -O1` grinds > 900 s — the record's "llc
grinds ~40 min").  Direct `llc -O1` on the merged core IR emits a **3,024 B**
core `.text` versus `-O0`'s **4,816 B**: it is *smaller*, and it is
value-clean (0 acquire exposures) in all 12 measured `-O1`/`-O2` cells.

**`llc -O0` hard-refuses a non-fast allocator** — so the plan's literal first
command is impossible:

    Must use fast (default) register allocator for unoptimized regalloc

**AIE2P `jl`/`j` HAVE DELAY SLOTS.**  Instructions printed *after* a call or
branch execute *before* the transfer takes effect.  Argument setups printed
"after" a `jl` are its delay-slot instructions, so `acq` printed after a `jl`
may execute before the callee.  **Disassembly order is not execution order on
this ISA**, and a value audit alone is therefore not sufficient: judge with
`step2d_audit.py` AND a device run.  This is why the audit's single reported
ARG line at `adaln` is a **false positive** on a build that runs.

**The `step2c` trim shim is UNSOUND as written — any build using it hangs.**
It `/DISCARD/ed` `.text._ZL9load_tail*`, `.text._ZL10store_tail*`, which live
code CALLS; `llvm-readelf -r rms_norm_weighted_d52c1240.o` proves the
relocations in `.rela.text.rms_norm_weighted`.  The result is dangling calls
(jumps into garbage).  Only the **unreferenced**
`vector_elem_ref<float,32>` variant (80 B, zero relocation entries) is safe to
drop.  Do not re-add the tail trims.

### RMSNorm scale-invariance: fold `1/sqrt(head_dim)` into the Q weight ONLY

The Z-Image attention scale is applied by folding it into the QK-norm weight,
because RMSNorm is scale-invariant:

    norm_q(x * s) == norm_q(x)        for any s > 0

**Therefore folding the scale into BOTH the Q and the K norm weight is a NO-OP:
the scale cancels out of the normalized vectors, the QK^T pre-scale is skipped,
and S comes out `1/scale` too large.**  Fold it into the **Q weight only**
(equivalently: one bf16 pre-scale of S, which is what
`imaggers/oracle/dit_block.py`'s device model does).

Measured cost of getting it wrong (stage 4, real geometry, `rel_to_absmax` vs
the corrected truth): **1.5852e-01**, versus **5.0760e-02** with the fold on the
Q weight only — a 3.1x error, and the difference between OVER and PASS.  The
symptom is diagnostic: `q_rope` absmax 6.25 against the model's 0.5508, i.e.
exactly `1/scale` = `sqrt(128)` ≈ 11.3 on one side.

`imaggers/block/STEP4.md` section 10.2 carries the full derivation and the
measurement; `step4_compose.py` documents the correct placement at the point of
use.

### `oracle.make_weights` produces NO QK-norm gamma — a synthetic block silently runs `w = 1`

`imaggers/oracle/dit_block.py`'s `make_weights` returns exactly
`{gamma1, gamma2, w_qkv, w_o, w_gate, w_up, w_down}` (verified by enumerating
the keys).  It produces **no `attention.norm_q.weight` / `attention.norm_k.weight`
at all**.  The block DOES have that stage — the real checkpoint carries the
tensors and `zimage-port/export_dit.py:65` sets `qk_norm=True` — so a
synthetic-weight block has **no QK-norm gamma of its own**, and the obvious
fallback (`np.ones`) is a configuration the record never accepted.

`step4_compose.weights_inputs` fills the hole with `np.ones(head_dim)`:

    w["wq_n"] = np.ones(HD) if "wq_n" not in w else w["wq_n"]

That is why `step4_accept.py` passes `real_qk_norm=True` for the SYNTHETIC
column: the accepted configuration pairs synthetic block weights with the
CHECKPOINT's own norm_q / norm_k gammas.  Measured, same seed, same everything
else (stage 5, two-block chain harness, `rel_to_absmax` vs the corrected truth,
budget 5.4e-02):

| QK-norm gammas | matched truth (both sides) | verdict |
|---|---|---|
| the checkpoint's own gammas | **5.0760e-02** | **PASS** |
| `w = 1`, matched (both sides unit) | 5.8340e-02 | OVER |
| `w = 1` assembled vs *checkpoint-gamma* truth (mismatched) | 1.2583e-01 | OVER |

Two lessons, both paid for:

1. The real, learned gammas are worth **0.76e-02** of block error at this seed.
2. A **mismatched** gamma pairing costs 7.5e-02 of *apparent* error and looks
   exactly like an assembly defect.  If the truth and the composition disagree
   about which gammas are in play, the comparison prices the disagreement, not
   the block.  When comparing a synthetic-weight block, Either patch BOTH sides
   or use the checkpoint's gammas on BOTH sides.

For a multi-block chain each block needs its OWN gammas
(`layers.<i>.attention.norm_q/k.weight`).  `step4_oracle.real_qk_norm_weights`
has no block argument and caches `layers.0`, so `step5_chain.py` patches the
module attribute in memory (`use_qk_norm`) and **`step4_oracle.py` is not
modified** — its single-block semantics stay intact for the earlier records.

Measurements and the harness calibration that found this (`step5_chain.py
--calib` reproduces STEP4.md's 5.0760e-02 / 6.4494e-02 / 4.8387e-02 to the
digit) are in `imaggers/block/STEP5.md` sections 2.2 and 3.

## Resolved

### VERIFIED ON THE NPU: bfp16ebs8 matmul at the quantisation floor

    A/B/C 18,432 B each   reference absmax 9391   decoded absmax 9344
    tile (n,m)=(64,64)    rel 1.011607e-02        <- BFP16 quantisation floor
    ref-vs-plain rel 0.000000e+00                 <- both references agree

**Root cause was NOT on the candidate list**: the host fed the compiled design
the WRONG PROBLEM SIZE.  `_out/sc.xclbin` was built from
`single_core/single_core.py` with its DEFAULTS -- problem `M=K=N=128`, per-core
tile `m=k=n=64`.  Proof is in the compiled artifact
`_out/sc.prj/aie.mlir`: the core loop is

    for arg1 in 0..4 { zero(C); for arg2 in 0..2 { matmul } }

i.e. `tiles=4`, `K_div_k=2` -- a 128x128x128 GEMM accumulating two K-chunks per
tile -- and `runtime_sequence` declares `memref<2048 blocks>` per operand
(18,432 B).  The host was generating a 64x64x64 problem in 4,608 B buffers, so
the DMA descriptors read 3/4 of every A/B object OUT OF BOUNDS and wrote
18,432 B of C into a 4,608 B buffer.

**Every earlier symptom was out-of-bounds DMA garbage**: `absmax 9344 vs 5351`,
`rel 1.0`, and the earlier `0.108` / `5e29` chaos.

**CORRECTIONS to earlier notes in this file** (kept because they mislead):
* "Ruled out: buffer sizes ... ours matches" was WRONG.  `bfp_test.cpp`'s
  `M*K*1.125` is 18,432 B for the design's M=K=128; ours was 4,608 B.  The
  comparison was made against the wrong problem size.
* "compile used -DDIM_M=64 -> tiles=1, K_div_k=1" conflated the kernel's MICRO
  dims (`-DDIM_M/K/N=64`) with the design's PROBLEM dims (`M=K=N=128`).

### The other two candidates are excluded, with proof

* **`floatToBfp16`'s 4th argument** — it is `int rounding` ("0 zero, 1
  nearest-even") but `helper.h`'s body NEVER READS IT.  A verbatim C++ port vs
  `bfp.encode` gives **0 differing bytes of 18,432** on both A and Bm.T
  (`imaggers/probe_codec.py`); `bfp.shuffle` is byte-identical to a port of
  `shuffleMatrixForBfp16ebs8`.
* **accumulator state** — excluded: `aie.mlir` shows `zero(C)` runs before each
  tile's K loop and `C = A(t,0)B(0,t) + A(t,1)B(1,t)` is the full K=128
  reduction.  The earlier "host zeroing made it worse" was the same OOB chaos.

### Output unshuffle: the TILE SIZE was the issue

    (n,m)=(64,64) per-core tile   ->  rel 1.01e-02   CORRECT
    whole-matrix tile             ->  rel 2.39e-01
    transposed                    ->  rel 3.06e-01

The earlier "whole-matrix beats 8x8" was measured on a 64x64 problem, where the
whole matrix IS the 64 tile, so it never tested the difference.  Derived push
order confirms the layout: `accC00/01/10/11 -> sub-tiles at z*colB+j raster`,
each `to_vector` = 8 rows of one block = exactly `bfp.shuffle`'s order.

Residual `1.01e-02` is C's `to_vector<bfp16ebs8>` truncation (max abs err 95 vs
a block step of 128 at absmax 9391; 9391 -> 9344 = 73*128).  Inside
`_BFP_TOLERANCE` (4 half-steps = 256).

---

## Open

`bfp_test.cpp` verifies by dequantising C and comparing against a **plain fp32
A @ B of the raw operands**; `mm_bfp_ref` instead quantises A and B first.
Measured on the same data:

    plain A@B absmax        5351.000000
    mm_bfp_ref absmax       5351.000000
    ref-vs-plain rel        0.000000e+00      <- the two references are IDENTICAL

So `mm_bfp_ref` is not the problem; the NPU output really is wrong
(`decoded C absmax 9344`, rel 1.00).

That closes the question "is the reference wrong?" and leaves only the
kernel/run side.  Remaining candidates, smallest first:

  1. **the output unshuffle geometry** -- the encoder's round-trip was exact,
     but that tested the ENCODER's inverse, not the device's output layout.
     `_block_layout.unpack` uses `bfp.shuffle(value, w, h, w, h, unshuffle=True)`
     with `w,h` from the stored shape; for C that is (M, N).  Try also
     `(n//8, m)`-style block geometry.
  2. **`floatToBfp16(8, A_SIZE, AVec.data(), 0)`'s fourth argument** -- an
     unknown flag/rounding parameter; the Python side uses `bfp.encode(x)` with
     the default rounding.  `helper.h` in the example tree has the C++ codec
     and says what the argument means.
  3. **the accumulator's initial state** -- `zero` runs on-device per tile, but
     `bo_c` is read back directly; if the accumulator holds something the
     `to_vector<bfp16ebs8>()` conversion interprets differently, that would
     inflate absmax (9344 vs 5351 is ~1.75x).

Everything else is ruled out; see the table below.

### Ruled out, with the measurements that did it

    defaults  M=K=N=128, m=k=n=64   ->   K_div_k=2, tiles=4
    but the compile used -DDIM_M=64 -DDIM_K=64 -DDIM_N=64
                                     ->   K_div_k=1, tiles=1  (one zero + one matmul)

    A_ty = np.ndarray[(M*K//8), v8bfp16ebs8]      host A is (M, K)
    B_ty = np.ndarray[(K*N//8), v8bfp16ebs8]      host B is (N, K) -- transposed
    C_ty = np.ndarray[(M*N//8), v8bfp16ebs8]      host C is (M, N)

    A_tiles = TensorAccessPattern.full((M, K//8)).tile((m, k//8))
    b_tap   = TensorAccessPattern.full((N, K//8)).tile((n, k//8))
    C_tiles = TensorAccessPattern.full((M, N//8)).tile((m, n//8))

    core_fn:  for tile: zero(elem_out); for ksplit: matmul(...)

All of that matches `prep_bfp16_io.py` (A `(M,K)`, B transposed `(N,K)`, C
`(M,N)`, one host call, whole-matrix tile when m=M).  So the host contract is
right and the residual is between `mm_bfp_ref` and what the kernel computes.

**Highest-value next step:** the example's own verifier
(`matmul_common::verify` in `bfp_test.cpp` line 286) is a THIRD opinion.  Feed
it the same A/B and see whether it agrees with `mm_bfp_ref`.  If the C++
reference and the python reference disagree, the fault is in `mm_bfp_ref`, not
in the run — and that would end this cleanly either way.

### Residual: rel 1.00 with integer data, rel 0.925 with gaussian

Using the reference's own data distribution (`rand() % 16` integers, as
`bfp_test.cpp` generates) instead of gaussian noise:

    encode/decode round-trip error on A:  0.000000   <- exact, as integers are
    reference absmax   5351
    decoded C absmax   9344
    rel                1.000

With gaussian noise it was rel 0.925.  So the failure is not data-distribution
dependent, and the round-trip of the encoder is exact, which means **the
encoder/decoder pair is self-consistent** and the disagreement is between what
the kernel computes and what `mm_bfp_ref` computes.

That reframes the problem usefully: it is no longer "is my byte layout right?"
(the round-trip is exact and the layout variants are all worse), it is
"does `mm_bfp`'s accumulation semantics match a single matmul?".  `mm_bfp`
implements `C += A @ B` over a 2x2 outer tiling with four accumulators; the
single host call drives all of it through `insts.bin`.  If the design's
`zero` kernel runs only once but `matmul` runs per tile, the result accumulates
across tiles -- which is what a mismatched K-split would look like.

Next, in order:
  1. run the example's OWN verification path (`bfp_test.cpp`'s
     `matmul_common::verify`) on a design built with `mm_bfp(dim_m=64, dim_k=64,
     dim_n=64)` and compare its expectation with `mm_bfp_ref` -- if the two
     references disagree with each other, the fault is in the reference, not
     the run
  2. check `single_core.py`'s `tiles`, `K_div_k`, `M_div_m` variables: the
     loop nests `zero -> matmul x K_div_k` per tile, and a value > 1 would
     change what one host call computes

Everything else is settled (see the ruled-out table below).

### Ruled out, with the measurements that did it

`bfp_test.cpp` lines 184-191 showed the shuffle takes
`(width, height, tile_width, tile_height)`, where the tile comes from the
kernel's `--w/--y/--z` options.  Measured both ways:

    whole-matrix tile (w,h,w,h)   ->  rel 0.925   (best so far)
    micro-tile 8x8                ->  rel 1.793   worse

`single_core.py` passes `dim_m=dim_k=dim_n=64` and the compile used
`-DDIM_M=64 -DDIM_K=64 -DDIM_N=64`, so the tile really is the whole matrix
and the 8x8 variant is simply the wrong geometry.  **Both are now ruled out as
the source of the residual.**

What is also ruled out, with the measurements that did it:

| candidate | measurement |
|---|---|
| operand layout (A/B/C orientation) | `as-is` beats transposed: 0.925 vs 1.80 |
| buffer sizes | `bfp_test.cpp` `A_SIZE=M*K, B_SIZE=N*K, C_SIZE=M*N` x1.125 matches exactly |
| tile geometry | whole-matrix 0.925, micro-tile 1.79 |
| host zeroing of the accumulator | makes it worse (absmax 5e29) |
| missing `zero` kernel | `sc.xclbin` already has it |
| number of kernel calls | `bfp_test.cpp` issues exactly ONE: `kernel(opcode=3, bo_instr, instr_v.size(), bo_a, bo_b, bo_out)` -- matches |

**The invocation is confirmed identical to the reference.**  So what is left is
subtler: probably `floatToBfp16(8, A_SIZE, AVec.data(), 0)`'s fourth argument
(a rounding/flag parameter, currently 0), or the fact that `bfp_test.cpp`
generates its data as `rand() % 16` (small integers) while `prep_bfp16_io.py`
uses gaussian noise — the shuffle/quantise may behave differently at the edges
of the range.  Try the reference's data generation first: it is one line and
rules out a whole class of explanation.

### `sc.xclbin` already contains the `zero` kernel -- so that is NOT the gap

`_examples/.../matrix_multiplication/bfp_test.cpp` lines 73-79:

    int A_SIZE = M * K;
    int B_SIZE = N * K;      // B is stored transposed -- matches axes=(1,0)
    int C_SIZE = M * N;
    size_t A_VOLUME = (A_SIZE * sizeof(uint8_t)) * 1.125;
    size_t B_VOLUME = (B_SIZE * sizeof(uint8_t)) * 1.125;
    size_t C_VOLUME = (C_SIZE * sizeof(uint8_t)) * 1.125;

Those agree with `prep_bfp16_io.py` (A 4,608 B, B 4,608 B, C 4,608 B for
64x64x64), so **the buffer geometry is not the problem either**.

What is left is the **invocation pattern**.  `matmul_vectorized_2x2_bfp16` does
a 2x2 outer tiling with four accumulators (accC00/01/10/11) and the design's
`core_fn` runs `zero` then several `matmul` calls per tile.  `run_bfp16_on_npu.py`
issues ONE kernel call.  Read `bfp_test.cpp` from line 135 to the end for the
exact `kernel(...)` arguments and the number of dispatches per host call; the
Python side mirrors whatever that does.

Also worth checking: `xrt::hw_context context(device, xclbin.get_uuid())` then
`xrt::kernel(context, name)` -- the reference uses a hardware context rather
than `pyxrt.kernel(dev, uuid, name)`.

### `sc.xclbin` already contains the `zero` kernel -- so that is NOT the gap

**Correction, recorded because it nearly sent the work the wrong way:**
`_out/sc.xclbin` was compiled from `_examples/.../single_core/single_core.py`,
which already has the `zero` kernel (lines 65-81):

    matmul_kernel = kernels.mm_bfp(dim_m=m, dim_k=k, dim_n=n)
    zero_kernel   = kernels.zero(m * n // 8, v8bfp16ebs8)

So the compiled artifact initialises the accumulator correctly.  The "missing
zero kernel" applies only to the hand-written design in `bfp16_gemm_jit.py`,
which is not what produced the artifact being run.

What IS still true:
* `mm_bfp` accumulates, so writing zero BYTES from the host is not equivalent
  and made the result worse (absmax 5e29) -- the all-zero byte pattern is not
  a valid bfp16ebs8 accumulator state.
* the operand LAYOUT is correct: `as-is` decodes better than either transposed
  variant (rel 0.925 vs 1.80).

So the remaining gap is between `single_core.py`'s expected host buffers and
what `run_bfp16_on_npu.py` passes.  `single_core.py` wraps the design with
`run_design_cli(...)` and refuses to run without a `run_and_verify` callback,
so its exact host sequence is in `test.cpp` / `bfp_test.cpp` rather than in
the Python.  That is the next thing to read: how many buffers, what sizes, and
how many times the kernel is invoked per host call (the 2x2 outer tiling in
`matmul_vectorized_2x2_bfp16` suggests several).

### Layout is RIGHT, ruled out

`_examples/block_datatypes/matrix_multiplication/single_core/single_core.py`
lines 65-81 show the required structure:

    matmul_kernel = kernels.mm_bfp(dim_m=m, dim_k=k, dim_n=n)
    zero_kernel   = kernels.zero(m * n // 8, v8bfp16ebs8)      # <-- missing here

    def core_fn(of_a, of_b, of_c, zero, matmul):
        for _ in tiles:
            zero(elem_out)                 # clears the accumulator ON DEVICE
            for _ in K_div_k:
                matmul(...)

`mm_bfp`'s contract is `C += A @ B`, so the accumulator must be initialised by
the `zero` kernel before every tile.  Writing zero BYTES from the host is not
equivalent and made things worse (absmax 5e29), because the all-zero byte
pattern is not a valid bfp16ebs8 accumulator state.

Current symptom: `decoded C absmax 0.108` vs `ref 0.0725`, rel 0.925 -- the
right magnitude, accumulated over uninitialised garbage.

**Fix**: add `kernels.zero(m * n // 8, v8bfp16ebs8)` to the design and call it
before `mm_bfp`, exactly as the example does.  The layout is already correct
(the `as-is` decode beats the transposed variants, 0.925 vs 1.80).

### Layout is RIGHT, ruled out

`aie/iron/kernels/linalg.py:975-983` gives the authoritative operand layouts
and `prep_bfp16_io.py` matches them exactly:

    A: _block_layout((dim_m, dim_k))               h=M, w=K
    B: _block_layout((dim_k, dim_n), axes=(1,0))   transposed
    C: _block_layout((dim_m, dim_n))               h=M, w=N
    pack  = bfp.shuffle(bfp.encode(v), w, h, w, h)
    unpack= bfp.decode(bfp.shuffle(v, w, h, w, h, unshuffle=True))

Decoding C as transposed is worse (rel 1.80), confirming the orientation.

### The whole pipeline is proven; only the numerics are left

The authoritative layout is `aie/iron/kernels/linalg.py:975-983`:

    A: _block_layout((dim_m, dim_k))                  h=M, w=K
    B: _block_layout((dim_k, dim_n), axes=(1,0))      transposed: h=N, w=K
    C: _block_layout((dim_m, dim_n))                  h=M, w=N

and `_block_layout.pack` does `bfp.shuffle(bfp.encode(value), w, h, w, h)`,
`unpack` does `bfp.decode(bfp.shuffle(value, w, h, w, h, unshuffle=True))`.

**`prep_bfp16_io.py` already matches that exactly** (A as `(M,K)`, B as
`Bm.T` with `(N,K)` geometry, C un-shuffled with `(N,M)`).  So the operand
encoding is not what is wrong.  Remaining suspects, in order:

  1. **`mm_bfp` ACCUMULATES**: its contract is `C += A @ B` (`_ZeroInitializedKernel`).
     If the C buffer is not zeroed on the device before the first call, the
     result is a sum of garbage.  `pyxrt.bo` may not guarantee zeros.
     Try explicit zeroing of `bo_c`.
  2. **Arg slots**: the design declares 3 ObjectFIFOs, and the reference
     wrapper places the OUTPUT at `out_arg` while inputs are `3+i`.  With two
     inputs that is 3, 4 and the output 5 -- which is what is used -- but
     `num_trailing_params` can shift the output slot.  Dump
     `group_id(0..7)` and cross-check against `xrt_test_wrapper.h`'s
     `data_arg_slot`.
  3. **Instruction buffer size**: `instr_v.size()` is the word count; the
     buffer is `size * sizeof(int)`.  Currently 165 words / 660 B, which
     matches.

Measured progress, so the direction is confirmed:

    inline row-major encode   ->  rel 9.476e+03
    aie.utils.bfp + shuffle   ->  rel 9.253e-01     (absmax 0.108 vs ref 0.0725)

### The whole pipeline is proven; only the numerics are left

Everything else works end to end:

    IRON design -> MLIR -> aiecc -> Peano(22.0.0) -> xclbinutil -> sc.xclbin
    pyxrt -> device(0) -> MLIR_AIE -> dispatch -> ERT_CMD_STATE_COMPLETED
    -> 4,608 B of BlockFP16 come back

No fp32 expansion anywhere: packed `bfp16ebs8` bytes go in and come out.

### Dispatch ABI -- FIXED, recorded so it is not rediscovered

Progress measured across the encoding fixes:

    inline row-major encode   ->  rel 9.476e+03   (decoded absmax 688)
    aie.utils.bfp + shuffle   ->  rel 9.253e-01   (decoded absmax 0.108,
                                                    reference absmax 0.0725)

So the encoding is now in the right family and the kernel is computing with the
right magnitudes; what remains is the exact shuffle geometry.  Likely causes:

  * `bfp.shuffle(x, w, h, w, h)` uses the WHOLE operand as one tile.  The
    reference `mm_bfp_tile_ref` does exactly that for one call, but `mm_bfp`
    in the program is called with `dim_m=dim_k=dim_n=64` inside a 2x2 outer
    tiling (`matmul_vectorized_2x2_bfp16`), so the host-side tile may need to
    match the kernel's `r,s,t` (8,8,8) rather than the full matrix.
  * B is documented as "read transposed"; the prep script encodes `Bm.T` with
    `(K, N)` geometry, which may not be the layout the stream expects.
  * C's readback uses `unshuffle=True` with the same geometry, which inherits
    whatever the input side gets wrong.

Where to look: `mm_bfp_tile_ref` (line ~263 of
`aie/iron/kernels/linalg.py`) is the authoritative encoder; `mm_bfp`'s
`stream_dims` is `None` for every operand, meaning the DMA does NOT transform
and the host must supply the exact bytes.  Compare the shapes it passes to
`bfp.shuffle` against `prep_bfp16_io.py`.

### The whole pipeline is proven; only the operand geometry is left

Everything else works end to end:

    IRON design -> MLIR -> aiecc -> Peano(22.0.0) -> xclbinutil -> sc.xclbin
    pyxrt -> device(0) -> MLIR_AIE -> dispatch -> ERT_CMD_STATE_COMPLETED
    -> 4,608 B of BlockFP16 come back

No fp32 expansion anywhere: packed `bfp16ebs8` bytes go in and come out.

### Dispatch ABI -- FIXED, recorded so it is not rediscovered

The kernel now **executes on the NPU**:

    run state: ert_cmd_state.ERT_CMD_STATE_COMPLETED
    result bytes: 4,608   first 18: 87e4ed93ee3af72bde87189e3e20cdba15eb
    result absmax 688   vs   float64 reference absmax 0.0726
    rel vs float64 matmul: 9.476e+03

So the execution path is done and only the data layout is wrong.  `mm_bfp`'s
docstring is explicit about it:

    A, B and C are `v8bfp16ebs8` blocks, **all pre-shuffled into the mmul
    layout**, so no DMA transform applies.

The inline `encode_bfp16` in `run_bfp16_on_npu.py` only does a plain row-major
block encode.  It needs the mmul shuffle on top.  The pieces already exist:

    aie.utils.bfp.shuffle(b, width, height, tile_width, tile_height, unshuffle)
    aie.iron.kernels.linalg.mm_bfp_shuffle_ref(tile, tile_width, tile_height,
                                               unshuffle)

`pack_bfp16.pack` implements that permutation and is byte-exact against a
vendor `.rai`, but it lives in the 3.12 half of the toolchain; the run side is
3.10 because `pyxrt.pyd` links `python310.dll`.  Either precompute the packed
bytes on 3.12 and hand them to 3.10, or port the ~30 lines of `pack_bfp16`'s
slot arithmetic.

The output must be un-shuffled the same way before it can be compared.

### Dispatch ABI -- FIXED, recorded so it is not rediscovered

`runtime_lib/x86_64/test_lib/include/xrt_test_wrapper.h` lines 288-302:

    unsigned int opcode = 3;                 // NOT 0
    run.set_arg(0, opcode);
    run.set_arg(1, bo_instr);                // CACHEABLE buffer
    run.set_arg(2, instr_v.size());          // instruction count in WORDS
    run.set_arg(3 + i, in_bos[i]);           // data buffers
    run.set_arg(out_arg, bo_out);

`pyxrt` facts, all established here:
  * `pyxrt.bo(dev, size, flags, group_id)` -- `flags` is the `pyxrt.bo.flags`
    ENUM (`host_only`, `normal`, `cacheable`, ...), not `XCL_BO_FLAGS_NONE`
  * `dev.register_xclbin(x)` before `pyxrt.kernel(dev, uuid, name)`
  * the kernel is `MLIR_AIE`, not `main`
  * `group_id`: 0->2883583, 1->2818049, 2->2883583, 3..7->2818048

## Resolved

`run_bfp16_on_npu.py` opens `MLIR_AIE`, allocates the three BFP16 buffers
(4,608 B each for 64x64), writes them and dispatches.  The run completes but
reports `ert_cmd_state.ERT_CMD_STATE_ERROR` and the output buffer is all
zeros, so the argument layout is wrong.

Tried so far, following `runtime_lib/x86_64/test_lib/include/xrt_test_wrapper.h`:

    "The fixed scalar args occupy kernel arg slots 0..2 (opcode, instr, ninstr).
     Data buffers start at slot 3, i.e. group_id(3 + data_index)."
    k(0, bo_instr, ninstr, bo_a, bo_b, bo_c)

`pyxrt` facts established while getting here (recorded so they are not
rediscovered):
  * `pyxrt.bo(dev, size, flags, group_id)` -- `flags` must be the
    `pyxrt.bo.flags` ENUM (`host_only`, `normal`, ...), not the
    `XCL_BO_FLAGS_NONE` int
  * `dev.register_xclbin(x)` must run before `pyxrt.kernel(dev, uuid, name)`,
    otherwise the uuid is rejected
  * the kernel is named `MLIR_AIE`, not `main`
  * `group_id` values: 0->2883583, 1->2818049, 2->2883583, 3..7->2818048

Next: read `runtime_lib/x86_64/test_lib/include/xrt_test_wrapper.h` and the
example's `test.cpp` for the exact `kernel(...)` call and buffer sizes; the
instruction buffer may need padding, and the opcode slot may not be 0.

**Context:** everything upstream of dispatch is DONE.  The toolchain compiles
BFP16 to a real xclbin, `pyxrt` reaches the device, the kernel opens.  This is
the last step.

## Resolved

### THE WHOLE TOOLCHAIN NOW COMPILES BFP16 -- it was a Peano version mismatch

AMD's own BFP16 matmul (`_examples/block_datatypes/matrix_multiplication/
single_core/`) now compiles to a real xclbin:

    sc.bin     660 B      instruction stream
    sc.xclbin  10,816 B   the compiled design

**The fix was exactly the pin from `utils/peano-requirements.txt` at tag v1.4.4:**

    llvm-aie==22.0.0.2026092101+0006955e

We had been on `21.0.0.*` nightlies -- a major version away.  The mlir_aie
README is explicit about it and even warns that mixing versions makes the
programming examples fail to compile:

    Install Peano from the llvm-aie wheel, pinned to the tested nightly in
    utils/peano-requirements.txt (the same pin CI uses)

`get_peano_pin.py` reads that pin; `get_pinned_peano.py` fetches that exact
wheel.  Two more tools were needed and are in the same XRT bundle as `pyxrt`
(`D:\Project\deepseek_harness\test7\WStmp\driver376\npu_mcdm_stack_prod`):

    xclbinutil.exe    -- aiecc's last step; add its directory to PATH
    xrt-smi.exe       -- device enumeration

Full working environment:

    PYTHONPATH        = _pylibs\mlir_aie\python;_pylibs
    PEANO_INSTALL_DIR = _pylibs\llvm-aie\llvm-aie
    NPU_CACHE_HOME    = zimage-bfp16-npu\_npu_cache
    PATH             += <npu_mcdm_stack_prod>

**Lesson recorded because it cost several rounds:** do not patch kernel source
to work around a compiler error until you have compiled the vendor's own
example.  It failed identically, which proved the problem was the toolchain
and not our code.

## Open

### DECISIVE: upstream's own example fails identically -- it is a toolchain
### version mismatch, not a bug in our code

`_examples/block_datatypes/matrix_multiplication/single_core/single_core.py`
is AMD's own BFP16 matmul example, downloaded from `Xilinx/mlir-aie`.  Compiled
against this toolchain it fails with **exactly the same error** as our design:

    RuntimeError: [Peano] compilation failed:
      fatal error: error in backend: unable to legalize instruction:
        %22:_(<8 x s8>) = G_BUILD_VECTOR %23:_(s8), ... x8
        (in function: matmul_vectorized_2x2_bfp16<8,8,8,8,8,8>)
      target: --target=aie2p-none-unknown-elf
      source: mm_bfp.cc   -DDIM_M=64 -DDIM_K=64 -DDIM_N=64 -DMATMUL_ONLY

**Consequences — read this before touching kernel code:**

* Do NOT patch `mm_bfp.cc` or add `aie_api/detail/aie2p/broadcast.hpp`.  The
  earlier "aie2p has no broadcast.hpp" hypothesis is at best not the whole
  story; the reference implementation cannot compile either.
* The toolchain pieces are mutually incoherent:

      mlir_aie 1.4.4 wheel        (bundles aie_api with `AIE_GIT_COMMIT ""`
                                   and `AIE_PEANO_VERSION "not configured"`)
      llvm-aie nightly 21.0.0.*   (two tried: 2026061101, 2026061601)

  AMD's own VitisAI build uses `-I C:/embedded/include/aie_api` -- a **separate,
  matched** aie_api checkout, and a matched Peano.  There is no published
  statement of which llvm-aie goes with mlir_aie 1.4.4.

**What would resolve it** (in order of expected yield):
  1. find the llvm-aie build mlir_aie 1.4.4 was released against (release
     notes, or the `pyproject`/`README` of that tag)
  2. try an older llvm-aie nightly — the wheel is from 2026-10-06, so a Peano
     from around then is the likeliest match
  3. take aie_api from the same tag as mlir_aie rather than the bundled copy

The whole pipeline up to this point is proven to work: the IRON design
resolves to MLIR, `@jit` reaches `_compile_and_build_kernel`, `mm_bf16`
gets through Peano, and `xrt-smi` sees the device.  Only the kernel compile
is blocked, and it is blocked for upstream's own code too.

### Flag sweep: no compiler flag works around it

    aie_api/detail/broadcast.hpp   ->  #include "aie1/broadcast.hpp"
                                       #include "aie2/broadcast.hpp"
                                       (no aie2p branch)
    aie_api/detail/aie2p/          ->  mmul_bfp16_bfp16.hpp, block_vector.hpp,
                                       emulated_mmul_intrinsics.hpp, mmul_*.hpp
                                       ... but NO broadcast.hpp

So an AIE2P build falls back to `aie2/broadcast.hpp` for splats, and the AIE2P
backend cannot legalise the `<8 x s8>` that produces:

    %22:_(<8 x s8>) = G_BUILD_VECTOR %23:_(s8), ... x8

This matches the failure exactly, and it explains why no compiler flag helped.

Other observations that fit:
* `include/aie/version.h` reads `AIE_GIT_COMMIT ""` and
  `AIE_PEANO_VERSION "not configured"` -- the bundled aie_api is unversioned,
  so there is no way to tell which Peano it was written against.
* AMD's own VitisAI build passes `-I C:/embedded/include/aie_api`, i.e. a
  **separate** aie_api checkout, not the wheel's copy.

Next step: write `aie_api/detail/aie2p/broadcast.hpp` (a 128-bit-vector
broadcast using AIE2P element-wise `insert`, or whatever `aie2p/block_vector.hpp`
already provides), or copy `aie2/broadcast.hpp` and replace the splat with an
explicit insert loop.  Re-run `probe_mm_bfp_flags.py` afterwards.

### Flag sweep: no compiler flag works around it

`probe_mm_bfp_flags.py` tried seven combinations against `mm_bfp.cc`:

    baseline / -O0 / -O1 / -O2 / -O3
    -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16=1   (the define AMD's own
                                                    VitisAI aiecompiler line uses)
    -fno-vectorize -fno-slp-vectorize
    -O1 + the emulate define

**All seven fail with the identical `G_BUILD_VECTOR <8 x s8>` error.**  So this
is not an optimisation-level artefact and no flag works around it; the fix has
to be in the kernel source or in `aie_api`.

Where that leaves it: the construct that cannot be legalised is a splat of one
`s8` into `<8 x s8>`, which is what unpacking a `v8bfp16ebs8` block into its
eight int8 mantissas produces.  Look at
`aie_kernels/linalg/mm_bfp.cc::matmul_vectorized_2x2_bfp16` and
`aie_api/detail/aie2p/mmul_bfp16_bfp16.hpp`.  Rewriting the unpack with an
explicit element-wise `insert` loop, or with `aie::vector` construction, is the
usual way past a missing splat rule.

Note also that the VitisAI build passes `-I C:/embedded/include/aie_api` -- a
separate `aie_api` checkout, not the one bundled in the mlir_aie wheel.  A
version skew between `aie_api` and Peano is a plausible root cause worth
checking first.

### Isolation result: the gap is specific to the BFP16 path

`probe_kernel_compile.py` compiles each kernel in its own process, calling
`CompilableDesign.compile()` so a success really means Peano emitted an object:

    mm            (script error -- the float32 annotation form)
    mm_bf16       PermissionError on _npu_cache/objects/.../aie-symbol-map-*
                  -> gets PAST Peano; blocked only by the sandbox's temp-file rule
    mm_bfp        [Peano] compilation failed: unable to legalize G_BUILD_VECTOR
                  -> the real backend gap
    mm_bfp_mixed  MLIRError: Verification failed (a different problem)

So `mm_bf16` proves the compile machinery works and the failure is specific to
`mm_bfp.cc`.

**Caveat recorded because it wasted a round:** `jit(build)` only WRAPS the
design; the compile happens on the first call.  A test that does
`d = jit(build)` and prints success is a false positive.  Call
`CompilableDesign(build).compile()` instead.

### Peano cannot legalise `mm_bfp.cc`'s `G_BUILD_VECTOR <8 x s8>`  <-- current blocker

The pipeline now reaches the kernel compile and dies there:

    [Peano] compilation failed:
    fatal error: error in backend: unable to legalize instruction:
      %22:_(<8 x s8>) = G_BUILD_VECTOR %23:_(s8), ... x8
      (in function: matmul_vectorized_2x2_bfp16<8,8,8,8,8,8>(bfp16ebs8 const*, ...))
    target: --target=aie2p-none-unknown-elf
    source: _pylibs/mlir_aie/include/aie_kernels/linalg/mm_bfp.cc
    flags : -DDIM_M=64 -DDIM_K=64 -DDIM_N=64 -DMATMUL_ONLY

Reproduced on **two** llvm-aie builds
(`21.0.0.2026061101+50bd3557` and `21.0.0.2026061601+dc0ebf29`, both
`py3-none-win_amd64`), so it is not a stale-version problem.  The AIE2P
backend is handed a splat to `<8 x s8>` its Legalizer has no rule for.

Where to look next:
  * `mm_bfp.cc`'s `matmul_vectorized_2x2_bfp16`, around the scalar_shuffle /
    mantissa path — a `v8bfp16ebs8` block unpacks to 8 int8 mantissas, which is
    exactly this vector
  * `aie_api/detail/aie2p/mmul_bfp16_bfp16.hpp` — whether the intrinsic has a
    newer spelling than the one the kernel uses
  * `mm_bfp_mixed.cc` as an alternative entry
  * `mm()` (plain bf16) as a control: if that compiles, the gap is specific to
    the BFP16 path

### Peano (llvm-aie) must be installed manually

`aie/utils/configure.py` resolves Peano in this order:

    $PEANO_INSTALL_DIR
    <aie_dir>/peano
    <aie_dir>/../peano
    <sysconfig platlib>/llvm-aie
    -> "peano_not_found"

None of the four exist on this machine, so `compile_external_kernels` raises
`RuntimeError: Invalid Peano install directory: peano_not_found`.  The fix is
the `llvm-aie` wheel (which ships `clang++`/`llvm-link`/`opt`/`llc`) or any
Peano build pointed at by `PEANO_INSTALL_DIR`.  `Xilinx/llvm-aie` publishes
releases; the PyPI names `llvm-aie` / `llvm_aie` were not resolvable from here.
See `find_peano.py` for the exact probe.

Everything upstream of that works: the design resolves to MLIR, `@jit` reaches
`_compile_and_build_kernel`, and the compile cache is redirected.

### The compile cache must be redirected before importing `aie`

`aie.utils.compile.NPU_CACHE_HOME` is a module-level constant read at import
time, defaulting to `~/.npu`, which a confined process cannot write to.  Set
`NPU_CACHE_HOME` before anything under `aie` is imported — see
`bfp16_gemm_jit.py`, which does it at module scope.

### The dtype annotation must be `np.dtype[marker]`, not the bare marker

`aie.helpers.npdtypes.np_ndarray_type_get_dtype` is
`get_args(get_args(t)[1])[0]`.  With a bare `v8bfp16ebs8` the inner `get_args`
is empty and the resolver dies with `IndexError: tuple index out of range` --
the same error that looks like a `program.py` bug.  Write

    np.ndarray[(M, K // 8), np.dtype[v8bfp16ebs8]]

### `pack_bfp16` vs `aie.utils.bfp`: same length, different content

`verify_packer.py` measures it directly:

    shape      tile     bytes   length    diff
    (8,  32)      8       288      288      124
    (16, 32)      8       576      576      260
    (16, 64)     16     1,152    1,152    1,036
    (32, 64)     32     2,304    2,304    2,100

**The byte LENGTHS agree exactly** -- both are 1.125 B/param, so the block
layout is the same.  The CONTENT differs on 43-91% of bytes, which is too much
to be rounding alone; the shuffle parameterisation is probably wrong too.

Two things to keep apart:

* `pack_bfp16` was verified **byte-for-byte against a vendor `.rai`** (12
  synthetic shapes, then a real 16-tensor shard: 416,071,680 B region, 0
  differing bytes).  For anything that goes through the RyzenAI EP, that is the
  ground truth.
* `aie.utils.bfp` is the **open toolchain's** codec and only offers
  `rounding='floor'` (default) or `'conv_even'`; its `quantize` rejects anything
  else.  The C++ reference (`bfp_kernel.cc`) has three modes:
  `STD_ROUND` (= `std::round`, half away from zero), `DPU_ROUND`, `PY3_ROUND`.

So the two encoders are aimed at different consumers.  Before feeding the open
route, the encoding must be pinned to whatever `mm_bfp` expects -- use
`aie.utils.bfp.encode` + `shuffle` directly, and compare against
`mm_bfp_tile_ref`, rather than reusing `pack_bfp16` blindly.

### `program.py:200` — `IndexError: tuple index out of range`

The IRON design reaches MLIR verification with the block-float operand
annotation `np.ndarray[(M, K//8), v8bfp16ebs8]`, then fails inside
`aie/iron/program.py::device_body`.  Next step is to read
`_pylibs/mlir_aie/python/aie/iron/program.py` around line 200 and see what
shape/dtype form it expects.  `aie.helpers.npdtypes` documents `v8bfp16ebs8`
as "a marker for the type an ndarray annotation carries", so the annotation
form is what needs fixing, not the kernel.

### `tile_rows` depends on the token count M

`pack_bfp16.KNOWN_TILE_ROWS` was measured at M=64.  At seq=96 the four shapes
of the Z-Image transformer take 48 / 32 / 32 / 64.  Any new geometry must have
its `tile_rows` measured, not assumed.

### `weights.bfp16` holds some wrong bytes

It was packed before the `_quantise` power-of-two fix (see below), so its GEMM
and adaLN regions differ from the corrected encoder on ~0.4% of bytes.  The
raw-fp32 const regions are fine.  Regenerate it before using it as the source
of truth.

---

## Fixed — do not re-investigate

### `_quantise` power-of-two edge case

`pack_bfp16` used `e = e_unb - 7` when `amax` was an exact power of two.  That
is wrong: AMD always uses `e_unb - 6`.  On a real 16-tensor shard ~0.4% of
weight groups hit the case, and on every one of the 7,198 mismatching groups
the vendor's exponent byte was exactly 1 higher with the mantissas halved.
Synthetic fixtures never exposed it because none of them had a power-of-two
amax.  Fixed; now unconditional.

### `safetensors.get_tensor()` access violation

It raises `0xC0000005` in `torch/storage.py:354` on this machine.  The files
are not corrupt.  Read tensors by parsing the safetensors header and using
`np.frombuffer` on the recorded offsets.

### RoPE silently missing from exported graphs

`build_meta_model()` never calls `patch_forward_real_rope` / `patch_processors`,
so `wrap_shard` received `freqs_cis=None` and skipped the rotation entirely.
Verified by swapping cos/sin and observing a bit-identical output.  Any graph
exported through that path must install both patches and assert they did
something.

### `do_constant_folding=False` breaks VAIP

Leaving the optimiser off makes VAIP see unranked types and die on
`cannot query rank of unranked shaped type`.  Session creation must use
`ORT_ENABLE_ALL`.

---

## Ruled out — do not repeat

### Guessing the `.rai` digest algorithm

12 contiguous regions x 6 algorithms (md5/sha1/sha256/sha512/blake2s/blake2b),
plus per-section (12 sections), arbitrary prefix concatenations, every section
boundary prefix/suffix, 5 `XCLBIN_MIRROR_DATA` block variants, the ONNX file,
every cache sidecar.  **Zero matches.**  The digests are named in the artifact's
plaintext JSON (`config.onnxMd5File`, `onnxMd5A`, `onnxMd5B`, `cacheKey`) but
only `onnxMd5File` has a known formula (`md5(ModelProto)`).

### A patched `.rai` being rejected because of a checksum

`route3_bit.py` flipped one bit at four different offsets inside the
`coalesed_weights` payload — GEMM body, adaLN bias prefix, adaLN weights, const
region.  **All four were accepted.**  The payload is not covered by an integrity
check.  Earlier "corrupted" results came from a contaminated baseline (a copy
saved from an already-patched file) and from a patch that grew the file and
destroyed the trailer.  Always run the `--offset none` control first.

### BFP16 as a documented developer format

Ryzen AI's own docs list supported quantization types as **BF16, A16W8, A8W8,
XINT8**.  BFP16 appears exactly once, as `MatMulNBits (W4ABF16/W4ABFP16)` — 4-bit
weights with BFP16 *activations*.  BFP16 **weights** are not a supported
developer-facing path, which is why this route uses the open toolchain.

### Parallel session creation

Measured 62.91 s vs 63.31 s serial — the EP holds a global lock.  Zero gain.

### The VitisAI EP weight-feed path

Weights fed as fp32 graph inputs cost ~58 ns/element of single-threaded
preparation; that, not operator fallback, is why CPU time exceeded 95%.
`declined.json` shows no `CPUExecutionProvider` at all.

---

### Reclaim inventory: 24.68 GiB of abandoned probe output still on D:

`zimage-port\_probe_out\` still holds 24.68 GiB of probe artifacts from
abandoned research.  D: has ~5.7 GiB free, enough for the remaining compiles,
so nothing has been deleted.  If space runs short, work down this list:

    zimage_loop        17,204 MiB     <- the single biggest item
    packverify          1,150 MiB
    p2_wholegraph         777 MiB
    ryzenai_nuget         730 MiB
    bfp16_deadlock        525 MiB
    gl_work               421 MiB
    fresh_cache           400 MiB     <- KEEP: only surviving vendor .rai
    noinit, dd85, feedcost, dd81, dd_swap, subv_search, waic_gate2, bfp16order, ...

**Do not delete `fresh_cache`** -- it holds the only surviving vendor `.rai`
(already copied to `vendor_artifacts/`, sha256 verified).  Everything else in
that tree is regenerable.  `_probe_out` is a directory junction, so deleting
through `zimage-port\_probe_out` removes the content for all three roots.

### CORRECTION: the SHIM refusal is the DEVICE CONSTRUCTOR, not the design

**Withdraw the "SHIM tile has only 2 input DMA channels" conclusion below.**
A 4-stream design (3 shim-fed fills + 1 shim-drained drain) compiles with one
device constructor and refuses with the other:

    set_current_device(NPU2())             -> COMPILES
    set_current_device(from_name("npu2"))  -> refuses: "all 1 ShimNOCTile(s)
                                               are at 0/2 input, 2/2 output"

Evidence is the placer output (`imaggers/block/out/step1.prj/
input_with_addresses.mlir`), which assigns DIFFERENT shim tiles per column:

    @inA  MM2S @shim(0,0) ch0
    @inB  MM2S @shim(0,0) ch1
    @inSB MM2S @shim(1,0) ch0     <- a second shim tile, column 1
    @outC S2MM @shim(0,0) ch0

The device has per-column shim tiles; `from_name("npu2")` exposes only one to
the placer, so the refusal text is describing the reduced model, not the
silicon.

**Rule: use `set_current_device(NPU2())` for any design with more than two
shim-fed streams.**  `memtile2/run.py` uses `from_name("npu2")` and is fine
only because its design has two.  The compute-tile `core_dma_channels_in == 2`
limit is real and still applies per Worker.

### The SHIM tile has 2 input DMA channels — the host interface budget

    error: no ShimNOCTile on the device has 0 input/1 output DMA channel(s)
           free: all 1 ShimNOCTile(s) are at 0/2 input, 2/2 output channels used
    note: this is the device's total ShimNOCTile DMA budget, not a placement
          choice; fan traffic through a MemTile, or reduce the number of
          DMA-attached ObjectFIFOs feeding this tile

A compute tile separately has `core_dma_channels_in == 2`.  **Every stage must
be planned as at most two host-fed input streams plus one output.**  Per-token
parameters (adaLN scale/shift/gamma) must ride inside one packed stream element
(`sb_row_layout` in `imaggers/kernels/designs.py`), not get their own channel.

This bit when the adaLN stream was added as a third host input, and it will
bite every later stage unless the interface is planned around it.

### Two modules are both named `refs` — the import order decides the winner

`imaggers/kernels/refs.py` and `imaggers/attn_impl/refs.py` share a module
name.  When both directories are on `sys.path`, whichever is inserted LAST
(at position 0) wins, and `kernel_wrappers` then fails with

    AttributeError: module 'refs' has no attribute 'F32_RTOL'

because it got the attention one.  Fix: put `kernels` last in the insertion
order so it wins, or load the other module by explicit path.  Any design that
composes both halves must handle this explicitly.

## A runtime-sequence command with more than 7 declared DATA ARGUMENTS is rejected: the ERT ABORTS it in under 2 ms

**Status: HARD LIMIT, measured on silicon.  Keep the declared data-argument count at 7 or below, and concatenate same-shape streams instead of adding arguments.**

Stage STEP6 (`imaggers/block/step6_attention.py`, the Z-Image attention branch
at `hidden 3840, heads 30, head_dim 128, seq 96`) declared **ten** data
arguments and the device refused every dispatch with

```
run 0: wait() -> ert_cmd_state.ERT_CMD_STATE_ABORT (ert_cmd_state.ERT_CMD_STATE_ABORT)  wall 0.31 ms
run 1: wait() -> ert_cmd_state.ERT_CMD_STATE_ABORT (ert_cmd_state.ERT_CMD_STATE_ABORT)  wall 0.12 ms
```

`ERT_CMD_STATE_ABORT` in under 2 ms with **zero-filled outputs is a bad command
descriptor**, not a deadlock (`TIMEOUT` is what a hang looks like, at ~7.0 s).
The codegen was clean throughout: the final-ELF audit passed (14 acquires, 0 ACQ
CLOBBER, 0 ARG), `.text` was 10,160 B of the 16,384 B program region, and
`.stack_sizes` was 384 B against 6,144 B reserved — identical in shape to the
working stage-2 object.

### The evidence — it is the xclbin's declared argument list, not the BOs supplied

| xclbin | declared data args | BOs supplied | result |
|---|---|---|---|
| 10-argument design | 10 | 10 | `ERT_CMD_STATE_ABORT` 0.39 ms |
| 10-argument design | 10 | **7** | `ERT_CMD_STATE_ABORT` 0.43 ms |
| 7-argument design | 7 | 7 | **`ERT_CMD_STATE_COMPLETED` 1.38 ms** |

The number of BOs the harness passes is irrelevant — the 10-argument xclbin
aborts even when handed seven.  `k.group_id(13)` raises
`IndexError: invalid vector subscript` while `group_id(3..12)` do not, which
confirms the kernel really does declare `opcode, instr, ninstr` plus 10 data
args, so the 10-BO call was the correct ABI for that xclbin and the xclbin
itself is what the runtime rejects.

Ruled out first, each by measurement: the **instruction stream** (an empty
runtime sequence, 16 B / 4 words, still aborts), the **BD content and count**
(a 4-fill sequence with clean `len`s still aborts), the **BO sizes** (replacing
four 29,491,200-B BOs with 4-byte pads still aborts), the **codegen level**
(plain `aiecc -O0` aborts identically), the **stack**, and the **tile
placement** (identical to the working stage-2 image).

### The boundary, swept

`WSPLIT` partitions the four same-shape weight matrices `Wk|Wq|Wv|Wo` into 1..4
runtime buffers, changing nothing else — same core program, same BD shapes, same
payload semantics.  Each point is a real build and a real device run:

| WSPLIT | weight buffers | declared data args | result | wall |
|---|---|---|---|---|
| 4 | `wk`, `wq`, `wv`, `wo` | 10 | `ERT_CMD_STATE_ABORT` | 1.27 ms |
| 3 | `wk|wq`, `wv`, `wo` | 9 | `ERT_CMD_STATE_ABORT` | 0.43 ms |
| 2 | `wk|wq`, `wv|wo` | 8 | `ERT_CMD_STATE_ABORT` | 0.38 ms |
| **1** | **`wk|wq|wv|wo`** | **7** | **`ERT_CMD_STATE_COMPLETED`** | 0.71 ms |

**7 works.  8 aborts.**

### The argument-count census of every previously verified stage

| image | declared data args |
|---|---|
| `imaggers/block/out/step1.prj` (stage 1) | 4 |
| `imaggers/block/out/s1w_M96_K3840_N32_m16_k8_n16_d111.prj` (stage 1 at M=96) | 4 |
| `imaggers/block/step4_out/s4a_M96_K96_h3_d32_m8_d111.prj` (stage 2) | 4 |
| `imaggers/block/out/step3_M96_K3840_N110240_m16_k8_n64_G4x6L10.prj` (stage 3) | **6 <- largest previously proven** |

So STEP6's ten was two past the edge and **no earlier stage had ever exercised
more than six**.

### The fix, and the rule

Concatenate same-shape streams into ONE runtime buffer and index it by explicit
object offsets.  `Wk`, `Wq`, `Wv` and `Wo` are four bf16 streams of *identical*
element count (14,745,600) already host-blocked into the core's consumption
order, so this is a pure HOST-LAYOUT change: **the core program is byte-identical
(same `.text`, same 14 acquires, same clean audit)**, and the command goes from
10 declared data arguments to 7.

> **Rule for every later stage (VAE, text encoder): keep the declared
> data-argument count at or below 7 — the largest previously proven value is 6 —
> and prefer concatenating same-shape streams into one buffer over adding
> arguments.**  Learning this cost two 36-minute compiles; applying it costs one
> `np.concatenate`.

The mechanism is **not** isolated and no claim is made about it: XRT's command
payload is nowhere near full at 10 arguments, so it does not look like a size
limit.  What is measured, repeatable, and sufficient to act on is the boundary.

### Companion finding, same stage: an un-awaited `fill` frees a BD id while its transfer is in flight

The same design emits **30,840 `dma_start_task`s / 1,080 `dma_await_task`s /
30,840 `dma_free_task`s** (the 1,080 awaits are exactly its 1,080 drains), while
the shim tile has only **16 BD ids** — each recycled ~1,900 times.  aiecc states
the hazard verbatim in its own refusal:

> "`aiex.dma_free_task` also recycles ids but does NOT wait for completion --
> using it before the task has finished is a race -- so reach for it only when
> some other synchronization already guarantees completion."

The verified stage-2 image never hit it because 7 tasks in a 16-id pool means the
pool is never recycled.

**Rule: pass `wait=True` on every fill, not only on the drains.**  The lowering
coalesces this to one `dma_await_task` per channel per `TaskGroup` (a 4-fill
group goes from `4 starts / 0 awaits / 4 frees` to `4 starts / 2 awaits / 4
frees`), and because a DMA channel is FIFO, awaiting the channel's last task
proves every earlier task on it has completed.  That is the "other
synchronization" the refusal asks for.

### THE STALL SIGNATURE: compare a build against a build that does LESS work

`ERT_CMD_STATE_TIMEOUT` at ~7.0 s is the ERT's *fixed* timeout, so the wall clock
alone can never tell "hung" from "still working".  The discriminator that does:

> **Cut the work by a large factor and compare the partial output.  If the
> smaller build produces LESS than the larger one, it is a STALL; if it produces
> the same or more, it is throughput.**

STEP6 (`imaggers/block/step6_attention.py`) is the worked example.  Its
`probe_heads` diagnostic truncates the core and the sequence together at the same
head boundary, changing nothing else.  The witness is `O`, drained per
(head, row-block) at the end of P5, so a non-zero `(96, 128)` head-0 block proves
P1..P5 all ran — projections, QK-norm, RoPE, QK^T, softmax and P*V:

| heads | projections | expected at 35 MMAC/s | `O` actually produced |
|---|---|---|---|
| 30 | 4.25 GMAC | ~123 s | **head 0 complete** (12,288/12,288, absmax 1.05469); heads 1-29 zero |
| 2 | 283 MMAC | ~8 s | head 0 half done (6,144/12,288) |
| 1 | 141 MMAC | ~4 s | **exactly all-zero** |

**The 1-head build does one thirtieth of the work and produces LESS than the
30-head build.**  That is a stall, not slowness.  Each build was run-to-run
byte-identical, so the amount of progress is a property of the image and not of
scheduling noise.  Note also that the raw `TIMEOUT` value is not a verdict in
either direction — a hang and an over-long kernel present identically.

### The concrete evidence that `dma_free_task` does not wait: the emitted counts

STEP6's full image, read off its own `aie.mlir`:

```
step6 (full, 7 args): dma_start_task = 30,840   dma_await_task = 1,080   dma_free_task = 30,840
step4 (verified, 4 args): dma_start_task =  7   dma_await_task =     2   dma_free_task =  7
```

**1,080 awaits is exactly the drain count** (360 `outO` + 720 `outX`), so all
30,840 fills were freed without ever being awaited — against a shim tile with
only **16 BD ids**, i.e. each id recycled ~1,900 times.  The verified stage-2
image never hit this because 7 tasks in a 16-id pool means the pool is never
recycled: **heavy BD recycling is new with STEP6.**

With `wait=True` on every fill the same design emits, for a 4-fill group,
`4 starts / 2 awaits / 4 frees` (one await per channel per `TaskGroup`, because
the lowering coalesces and a DMA channel is FIFO).  That is the fix, and it is
what the record's own note already prescribed.

### Also measured, and worth knowing before designing this branch again

At the record's own measured scalar-`mm` throughput (STEP2D.md §5: 5.3 MMAC in
152 ms = **35 MMAC/s** on this same kernel family), the real-geometry attention
branch's QKV+to_out projections are 4.25 GMAC = **~123 s of pure arithmetic**.
The ERT command timeout is ~7.0 s.  **A correct full-geometry scalar-`mm`
image therefore cannot return `ERT_CMD_STATE_COMPLETED` at all.**  Any
real-geometry attention stage must either use a vectorized GEMM family
(`mm_bfp`/`mmul`) or a longer command timeout; this is a design constraint, not
a defect, and it is measured rather than assumed.

## Resolved — `step2d_audit.py` ACQ CLOBBER false positive on a CONST operand

**Class: static-analysis false positive. Not a hazard. Safe to proceed on.**

### The symptom

`step2d_audit.py` reports exposures that the design does not have:

```
[audit] elfs_main_core_0_3.elf: 9 acquires, 7 ACQ CLOBBER exposure(s), 0 ARG exposure(s)
[audit] ACQ CLOBBER: acq line 54: 10a:  acq #0x35, r15
             def  line 49: f6:  mova r15, #-0x1
             call line 52: 100: jl  #0x380
   ... 7 sites, all the same shape ...
[audit] FAIL: 7 acquire executes with a stale value (build gate)
```

Measured on the Z-Image text-encoder GEMM
(`imaggers/textenc`, `n=256`, `Kc=960`, `m=16 k=8`). It reproduces at **both**
codegen levels: `spec.compile()`'s default `llc -O0`, **and** the STEP2D
`-O1 --regalloc=greedy` injection (which does cut the acquire *site* count
10 -> 3, so the injection works — it simply does not remove the report).

### The cause

The rule fires as: *"a `jl` (call, which clobbers caller-saved registers) sits
between the def of an `acq`'s value register and the `acq` itself."*

Here the value operand is **the IR constant `-1`**, not a live value. The
audit's own output says so, verbatim, at every one of the 7 sites:

```
[audit]   .ll line 24: CONST (IR-constant operand; -O0 rematerializes it
             through a stack slot -- mem2reg cannot help)
             acq : call void @llvm.aie2p.acquire(i32 50, i32 -1)
             val : -1
```

`mova r15, #-0x1` is a **rematerialisable constant**. The check tracks a
register DEF and a later clobbering CALL, but it does not model
rematerialisation — so a constant that is re-materialised *after* the call is
indistinguishable, to a static reader, from a live value that was clobbered.
That is the false-positive class. **The check is correct for the case it was
written for** (the attention design's live register across a call); it has no
way to represent this one.

### The independent evidence, which is what settles it

1. **The IR order is preserved.** Read directly out of
   `peano-linked_main_core_0_3.ll`, the event order is
   `ACQ 50 -> ACQ 49 -> ACQ 53 -> CALL matmul -> REL 52 -> REL 48 -> REL 51`,
   with the kernel call strictly between the acquire and the release. No call
   is slid across an acquire.
2. **`--order` returns no usable signal either way.** It reports
   `IR[0:0]=[] vs ASM[0:19]` and "1 divergence" — i.e. its IR event extractor
   returned an EMPTY list against a file that provably contains acquires,
   releases and kernel calls. **The `--order` check is not reporting a
   reorder; its IR side is not extracting.** That is a second, separate
   tooling defect and should be fixed before `--order` is trusted again.

   > **Update 2026-10-20 (t22_factcards2).** The extractor defects were fixed
   > (five of them) and `--order` was retired as an acceptance leg.  After the
   > fix the IR side reports **36 events vs ASM 36, multiset identical, added 0 /
   > dropped 0**, while the linear order still differs in 8 places — i.e. the
   > **sequence comparison itself is not a valid gate** (basic-block layout, not
   > codegen reordering).  `--order` now prints `DIAGNOSTIC ONLY` and no longer
   > changes the exit code; the only gate-worthy signal kept is the multiset
   > invariant.  The durable fix (per-lock CFG check) is a separate 0.5-1
   > person-day unit.  Source: `docs/T22-QUICKFIX-REPORT.md` §2.
3. **A genuinely stale acquire does not behave like this.** The failure family
   is hangs, `ERT_CMD_STATE_TIMEOUT`, or stale objects. This design returns
   `ERT_CMD_STATE_COMPLETED` on every run, is **run-to-run byte-identical**,
   and is **100.00 % bit-exact against `mm_bfp_tile_ref`'s per-call model over
   640 GEMM calls**. A stale lock value cannot produce that.

### What to do

* Treat an `ACQ CLOBBER` whose site is annotated `CONST` as **informational,
  not a gate failure**, and require the three independent checks above (IR
  order read directly, COMPLETED, byte-identical + bit-exact) to carry the
  decision.
* **Do not** silence the rule for live-register sites — the attention design's
  exposures were real.
* The durable fix is for the audit to resolve whether the flagged def is a
  rematerialisable constant; until then, classify per site by the `CONST`
  annotation it already prints.

### Also worth fixing here

`check_order`'s IR extractor returned `[]` on
`peano-linked_main_core_0_3.ll` (127 lines, 4 `llvm.aie2p.acquire`, 3
`llvm.aie2p.release`, 2 kernel-call sites). An empty IR event list can only
produce a spurious divergence, so the `--order` gate currently cannot pass on
any design until that is repaired.

> **Update 2026-10-20 (t22_factcards2).** Repaired: on
> `peano-linked_main_core_0_3.ll` the extractor went from **[] to 7 events**.
> The empty list was one of five defects (entry-name regex, call regex quote
> anchoring, body scan starting at depth 0, ASM `JL_RE` line anchoring,
> hard-coded entry names).  The gate as a whole stays retired — fixing the
> extractor only made the *comparison's* invalidity visible.  Source:
> `docs/T22-QUICKFIX-REPORT.md` §2.1.
## Resolved — two AIE2P build traps that masquerade as unrelated errors

Both were hit building the Z-Image text-encoder norm stage
(`imaggers/textenc`, `K=2560`, `n=256`). Both cost multiple failed attempts
because the error message does not name the cause.

### 1. `memref.view` needs a 1-D i8 source, and a malformed one aborts the tracing context mid-region

`memref.view` accepts **only** a 1-D memref of 8-bit signless integers. Give it
anything else and the error is explicit:

```
error: 'memref.view' op operand #0 must be 1D memref of 8-bit signless
integer values, but got 'memref<2560xbf16>'
note: see current operation: %39 = "memref.view"(%35, %38)
      : (memref<2560xbf16>, index) -> memref<2560xbf16>
```

**But if that view is built inside a worker's nested `scf.for` region, the
failure surfaces as something else entirely:**

```
RuntimeError: Unbalanced Location enter/exit
Exception ignored in: <generator object _for at
  _pylibs/mlir_aie/python/aie/helpers/dialects/scf.py", line 76, in _for>
RuntimeError: Unbalanced InsertionPoint enter/exit
```

**The malformed `memref.view` aborts the tracing context mid-region**, so the
region's `InsertionPoint`/`Location` are never exited and the tracer reports an
unbalanced context rather than the real type error.

**Why this is worth knowing:** it makes the failure look like a *structural*
problem with the loop nesting, so the natural response is to vary the loop
shape — which reproduces the identical error every time and teaches nothing.
Four parameter/region variants of the same worker all produced the byte-identical
`Unbalanced Location enter/exit`. The fix had nothing to do with the regions.

**Rule:** on `Unbalanced Location enter/exit` (or the matching
`InsertionPoint` message) from `scf.py:_for`, **check every `memref.view`
source type in that region body FIRST** — the region is a symptom, the view is
the cause. Typed objects must be carried as raw i8 and `memref.view`'d out:
that is why the proven designs keep their rows/slices as `int8` objects and
`view` typed memrefs from them.

### 2. A ShimNOCTile has only FOUR DMA channels, and that is the DEVICE TOTAL

```
error: no ShimNOCTile on the device has 0 input/1 output DMA channel(s) free:
all 1 ShimNOCTile(s) are at 0/2 input, 2/2 output channels used
note: this is the device's total ShimNOCTile DMA budget, not a placement
choice -- no column has spare capacity to pin to; fan traffic through a
MemTile, or reduce the number of DMA-attached ObjectFIFOs feeding this tile
```

The note says it outright: **it is a budget, not a placement decision.** There
is no column to move to and no `PlaceTiles` hint that helps. A design gets
**2 input and 2 output** host-facing channels, total, and a compute tile gets
**2 inputs**. Adding a third input fifo for one more per-token/per-channel
parameter therefore cannot be made to fit.

**The consequence, and the reason it matters:** any extra per-token or
per-channel parameter must **ride a packed stream element** rather than a
second input fifo. `imaggers/block/step1_norm_adaln_gemm.py` packs
`[scale | shift | gamma]` into ONE object for exactly this reason (its
`inSBG` element is `3*K*2` B, `sbg_row_layout()` gives the byte offsets), and
its comment says so. The Z-Image text encoder needs the same treatment for
`rms_norm_weighted`'s gamma: `[x_row | gamma]`, gamma at byte offset `K*2` of
the same object, one fifo.

**Rule:** count `2 in + 2 out` **before** designing a worker's inputs. A
parameter that arrives per token is a packed-stream-element decision, not a
fifo decision. This is a **channel budget**, not a code shape — and it is the
kind of thing only building it surfaces.
## Terminology

* **BFP16 / `bfp16ebs8`** — block float, 8 elements sharing one 8-bit exponent,
  9 bytes per group of 8.  This is what this repo produces.
* **BF16 / bfloat16** — a plain 16-bit float with 8 mantissa bits.  NOT what we
  want; using it doubles the residency and loses the advantage.
* The frequently quoted "32 elements share an exponent" is wrong for
  `bfp16ebs8`.  It comes from `q4nx`'s group of 32 (`Q4NX_GROUP = 32`), a
  different format.

---

## ERRATA E1 (2026-10-19, t21_qfix / C1) -- `weights.bfp16` EXISTS; the note at :134 is withdrawn

**The file was never lost.**  Measured 2026-10-19:

    zimage-port\weights.bfp16              6,913,965,970 B      present
    zimage-port\build_bfp16_weights.py          9,327 B          present

`final_audit.py` reads exactly that path and prints
`OK weights.bfp16 6,913,965,970 B`.

* **The correct location is `zimage-port\weights.bfp16`** -- the sibling
  `zimage-port` tree.  There is no in-tree copy, and that is what the earlier
  "existence contradiction" really was: a **path statement**, not a missing file.
* Rebuild ONLY if it is ever lost: from `public_work\zimage-port\`, run
  `python build_bfp16_weights.py --out weights.bfp16` (it reads the transformer
  shards; see its `--manifest/--out/--m`).
* The "Consequences" bullets and the `weights_audit.py` shard paragraph under
  :134 are the superseded incident report.  **Keep them as history; do NOT act
  on them** -- in particular, do not "regenerate the file first".

Why here and not inline at :134: inserting a correction there would have shifted
every later line anchor, and this file is cited BY LINE NUMBER from
`verify_packer.py`, `imaggers/perfc/perfc_probe.py` and the questions document.
The heading at :134 therefore carries the WITHDRAWN tag and points here instead.

The same note exists in the sibling `zimage-port\KNOWN-ISSUES.md` (fixed
2026-10-19; that file is untracked in the `zimage-port` repository, so the fix
there is a working-tree edit only).

## ERRATA E2 (2026-10-19, t21_mvdocs2) -- "the questions document" now lives at `docs/zimage-bfp16-npu-questions.md`

The paragraph above at :1549 names three callers that cite this file BY LINE
NUMBER: `verify_packer.py`, `imaggers/perfc/perfc_probe.py`, and "the questions
document".

That third document has moved.  It now lives at
**`docs/zimage-bfp16-npu-questions.md`** -- inside this repository's `docs/`,
relocated out of the workspace root by t21_mvdocs (commit `8ea227d`).  The
follow-up batch t21_mvdocs2 moved ten further root documents into the same
directory: CLEANUP-INVENTORY.md, CLEANUP-INVENTORY-2.md, HANDOFF-2026-10-12.md,
HANDOFF-2026-10-19.md, MEM-PROFILE.md, TIMING-RECONCILIATION.md, WEIGHT-MAP.md,
WORKLOG.md, zimage-npu-port-notes.md, ZIMAGE-TENSOR-INVENTORY.md.

Nothing else in that paragraph changes.  The anchors it relies on -- :988-:1000,
:134, and :1549 itself -- are untouched: this note is appended at EOF rather
than inserted inline, so every existing line number still means what it did.
