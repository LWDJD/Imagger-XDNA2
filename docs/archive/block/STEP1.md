# STEP1 — `rms_norm_weighted -> adaln_modulate -> mm_bfp(mixed=True)` at hidden 3840

Status: **TARGET criterion met.**  The chain compiles to an xclbin, runs on the
XDNA2 NPU with `ERT_CMD_STATE_COMPLETED` on every run, and matches the numpy
reference within **1.0 bf16 ulp** at both tested seeds (the budget is 2 bf16
uls, `refs.TOL_ULPS`, which is what `_scratch/kernels/refs.py` already accepts).

Deliverables (this directory):

| file | what |
|---|---|
| `step1_norm_adaln_gemm.py` | the IRON design (memtile2 architecture + inserted adaLN stage + weighted norm) |
| `step1_run.py` | compile + run + verify harness (Python 3.12 side + `--xrt-only` Python 3.10 side) |
| `STEP1.md` | this report |
| `out/s1w_M32_K3840_N32_m16_k8_n16_d111.{xclbin,bin}` | the compiled design |
| `out/s1w_*_io.npz`, `out/s1w_*_C_raw.npy` | test vectors, references, raw device output |

## What the device computes

Per token row `t` of `X` (M=32 tokens x K=3840 hidden, bf16):

```
N[t] = X[t] / sqrt(mean(X[t]**2) + 1e-5) * gamma      # rms_norm_weighted, per-channel gamma
Y[t] = N[t] * (1 + scale[t]) + shift[t]                # adaln_modulate, per-token scale/shift
C    = Y @ W                                           # mm_bfp(mixed=True), W bfp16ebs8, C bf16
```

Each kernel rounds to bf16 exactly once, at its store (conv_even inside each
entry), and `mm_bfp_mixed` re-rounds C to bf16 after **every** `(m,k)x(k,n)`
GEMM call — the reference models all of that (`kernels/refs.py` formulas +
memtile2's per-call bf16 `ref_model` loop).

## Measured results (M=32 K=3840 N=32, tile (m,k,n)=(16,8,16), Kc=960)

| stage | seed 0 | seed 1 |
|---|---|---|
| static checks (taps, packing, alignment, corner walk) | ALL PASS | ALL PASS (same) |
| compile (aiecc + xclbinutil) | 12.6 s -> 21,304 B xclbin + 900 B insts | (reused) |
| device runs | 2/2 `ERT_CMD_STATE_COMPLETED`, 6.45 / 6.01 ms | 2/2 `ERT_CMD_STATE_COMPLETED`, 6.23 / 5.82 ms |
| run-to-run byte-identical | True | True |
| **max abs err vs ref_model** | **128.0 (= 1 ulp of C)** | **128.0 (= 1 ulp of C)** |
| **max ulp distance vs ref_model** | **1.000** | **1.000** |
| bit-exact vs ref_model | 99.90% (1023/1024) | 99.90% (1023/1024) |
| `refs.check_within_ulps(<= 2)` | **PASS** | **PASS** |

The single 1-ulp element per seed is a bf16 tie flip between the reference's
f64 formula and the device's f32 chain inside `rms_norm_weighted` — exactly
the effect `refs.py`'s 2-ulp budget covers.  The GEMM stage itself is
bit-exact against `mm_bfp_tile_ref` (as memtile2 measured 100.00%); the whole
residual is the norm stage's sub-ulp tie.

(For context: the error vs the *unquantized* float chain is large
~1 ulp of the output range — that is the bfp16ebs8 weight quantization plus
the per-call bf16 C re-rounding, both modeled in `ref_model`; the unquantized
chain is reported for reference only.)

## Architecture — memtile2, reused, with the adaLN stage inserted

Everything that made `_scratch/memtile2/memtile_worker.py` verify 100.00%
bit-exact at K=3840 is unchanged:

* column-chunked producer (Kc=960 at K=3840): Worker A `(0,2)` emits one
  `(m, Kc)` chunk object per outA transfer (producer BD = 8*Kc = 7,680 words,
  compute cap 16,383); rows are re-sent per chunk pass so the norm core never
  holds `(m, K)`.
* the memtile corner turn: 1:1 `ObjectFifoLink` at `(0,1)`, `midB.to_stream =
  corner_turn_tap(m, Kc)`, `consumer_obj_type = (m*k*2,) i8` -> Worker B gets
  ready `(m,k)` slice objects.
* Worker B `(0,3)`: `zero(C); for kk in K/8: mm_bfp(mixed)` verbatim.
* C drain via the memtile forward hop (the inverse walk is 4-dim; compute-tile
  BDs take at most 3 dims), and memtile2's TaskGroup choreography verbatim.

The inserted stage sits in Worker A between the norm and the chunk copy:

```
for c in chunks: for r in rows:
    acquire row r;  acquire this token's [scale|shift|gamma] element
    norm(in_row, g_row, in_row, K, 1e-5)     # rms_norm_weighted, in place
    adaln(in_row, s_row, b_row, in_row, K)   # modulate, in place
    copy row columns [c*Kc,(c+1)*Kc) -> (m,Kc) chunk object   # memtile2 verbatim
```

Both calls are alias-safe at `x == y` (verified in the kernel sources:
`rms_norm_weighted.cc` pass 1 reads the whole row before pass 2's
load-chunk/store-chunk; `adaln_modulate.cc` is elementwise load->acc->store).

### The one real constraint it forces: `core_dma_channels_in == 2`

The weighted norm + per-token modulation need THREE parameter rows (gamma,
scale, shift) per token in addition to the x row — three input streams, but a
Worker takes at most TWO input ObjectFifos.  The fix is the
`kernels/designs.py` idiom extended: the parameter rows ride on ONE packed
stream element, one per token row:

```
inSBG element, 23,040 B at K=3840:  [ scale (K bf16) | shift (K bf16) | gamma (K bf16) ]
```

All three splits are 64 B multiples at `K % 32 == 0` (offsets 0, 2K, 4K), so
every `memref.view` satisfies the kernels' 64 B alignment contracts with zero
padding.  gamma is shared but rides per element (only read by that element's
norm call), so no held buffer and no third fifo.  inA (x rows) is memtile2's
row stream unchanged, and both taps use the identical
`A_rows[r0:r0+m].repeat(row_repeats)` schedule, so the r-th parameter element
always pairs with the r-th row (proved by the static tap-lockstep check).

### L1 and placement (measured in `out/s1w_*.prj/input_with_addresses.mlir`)

```
(0,2) inA 7,680 B + inSBG 23,040 B + outA 30,720 B = 61,440 B data
      + 0xF00 stack (3,840) = 65,280 of 65,536 B   (256 B spare)
(0,1) 31,744 B    (0,3) 1,568 B
every buffer 64 B aligned: True (all tiles)
```

Depth 1 on all fifos; depth 2 on inA does not fit (69,120 B).  This matches
memtile2's own sweep experience (depth-1 input fifos are what fit at large K).

## Refusals and errors encountered (exact text)

1. **Shim DMA exhaustion with `from_name("npu2")`** — the first compile of the
   4-stream design (3 shim-fed fills + 1 shim-drained drain) refused:

   ```
   [0/1] placed.mlir ...aie.mlir:7:27: error: no ShimNOCTile on the device
   has 0 input/1 output DMA channel(s) free: all 1 ShimNOCTile(s) are at
   0/2 input, 2/2 output channels used
   note: this is the device's total ShimNOCTile DMA budget, not a placement
   choice -- no column has spare capacity to pin to; fan traffic through a
   MemTile, or reduce the number of DMA-attached ObjectFIFOs feeding this tile
   ```

   Root cause: the **device constructor**, not the design.  With
   `set_current_device(from_name("npu2"))` (as memtile2/run.py uses) the
   placer sees ONE ShimNOCTile (2 MM2S + 2 S2MM) and cannot take a third fill.
   With `set_current_device(NPU2())` the placer uses the per-column shim
   tiles: the third fill lands on `aie.shim_dma_allocation @inSBG_shim_alloc
   (%shim_noc_tile_1_0, MM2S, 0)` and the build compiles + runs.  (This same
   refusal also stopped a parallel workstream in this directory; the fix is
   `NPU2()` for any 3-fill design.)  No design change was needed.

2. No other refusals.  The kernels (`_scratch/kernels/kernel_wrappers.py`
   `rms_norm_weighted`, `adaln_modulate`), the memtile2 producer/corner-turn,
   and the GEMM core all accepted the insertion without modification.

## Success criteria

* **MINIMUM (compiles to an xclbin): MET** — `out/s1w_M32_K3840_N32_m16_k8_n16_d111.xclbin`.
* **TARGET (runs `ERT_CMD_STATE_COMPLETED` and matches numpy within 2 bf16
  ulps): MET** — 4/4 device runs completed; max 1.000 bf16 ulp vs the refs.py
  reference chain at seeds 0 and 1 (budget 2.0, `refs.check_within_ulps`).

## Reproduce

```
D:\Users\15081\miniforge3\envs\ryzen-ai-1.8.0\python.exe step1_run.py -M 32 -K 3840 -N 32 --runs 2
```

(Static checks -> compile -> prep -> NPU run via the self-re-executed
`--xrt-only` Python 3.10 half -> verify.  `--skip-compile --seed N` re-runs an
already-built xclbin.)

## Notes

* `os.mkdir(path, 0o700)` sandbox shim + `use_cache=False` are copied from
  `_scratch/dit_block_v1_run.py` (required: `tempfile`'s 0o700 staging dirs
  cannot be re-entered by this token).
* Environment set before `import aie`: `PEANO_INSTALL_DIR` and
  `NPU_CACHE_HOME = _npu_cache7`; `xclbinutil` comes from
  `test7\WStmp\driver376\npu_mcdm_stack_prod` on PATH.
* A parallel workstream in this directory (`step1_design.py`,
  `step1_build.py`, `step1_prep.py`, `step1_verify.py`, `step1_run_xrt.py`,
  `out/step1.*`) was left untouched.  That variant chains the **unweighted**
  `kernels.rms_norm` with `scale == shift` and does not meet this task's
  kernel chain; the deliverables above are self-contained and use
  `rms_norm_weighted` with `gamma != 1` and independent per-token
  scale/shift (enforced in `prep_data`'s asserts).
