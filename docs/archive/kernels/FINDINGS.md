# FINDINGS — `rms_norm_weighted` + `adaln_modulate` kernels for AIE2P (XDNA2)

Scope: new files only under `zimage-bfp16-npu\_scratch\kernels\`.  Nothing
outside that directory was created, modified or deleted.  The only writes
outside it are the task-mandated toolchain caches
(`NPU_CACHE_HOME = zimage-bfp16-npu\_npu_cache3`, and `%TEMP%` staging by
aiecc/Peano).  `_out\`, `_scratch\dit2\`, `_scratch\dit_block_v1*` were
read-only.

Deliverables (all under `_scratch\kernels\`):

| file | what |
|---|---|
| `rms_norm_weighted.cc` | weighted RMSNorm C++ kernel (aie2p tuned + generic + f32 entry) |
| `adaln_modulate.cc` | fused adaLN modulation C++ kernel (bf16 + f32 entry) |
| `kernel_wrappers.py` | IRON wrappers (4 factories, `KernelContract` per variant) |
| `designs.py` | the two IRON programs (one Worker, one row per kernel call) |
| `refs.py` | numpy references, self-testing (f32 semantics FIRST, then bf16) |
| `run.py` | compile + run + verify, Py3.12/Py3.10 split |
| `xrt_run.py` | Py3.10 (pyxrt) device runner |
| `diag_adaln.py`, `diag_const.py` | structured-input probes (alignment forensics) |
| `out/` | xclbins, insts, prj dirs, npz/npy IO, measurements' inputs |

---

## 1. SUCCESS CRITERIA

See section 6 for the measured numbers (filled from `run.py` output).

* **MINIMUM: both kernels compile to an xclbin — MET.**  Four xclbins built
  (bf16 and f32 variant of each kernel); `out/*.{xclbin,bin}`.
* **TARGET: both run with `ERT_CMD_STATE_COMPLETED` and match numpy within a
  bf16-level tolerance at hidden 3840 — see section 6.**  The device run
  prints `wait() -> ERT_CMD_STATE_COMPLETED` per run (2 runs each,
  run-to-run byte-identical).

---

## 2. KERNELS

### 2.1 `rms_norm_weighted` — `y = x / rms(x) * gamma`, one row per call

* Template: `norm/rms_norm_aie2p.h:114-170` (`rms_norm_aie2p`) — the
  two-pass shape (f32 sum-of-squares `:123-138`, f32 scalar inv_rms
  `:140-142`, 64-lane chunks `:77-79`, 16-lane-part/scalar tails `:84-110`)
  is kept verbatim; the normalize pass gains ONE multiply per 64-lane chunk
  and per tail: `xg = x * gamma` in f32 (the task's delta over the shipped
  loop at `:156-162` / `:164-168`).  The weighted arithmetic order follows
  `flm_gemma4/rms_norm.h:53-68` (`rms_norm<D>(y, x, w)`: product first,
  then scale).
* Entry: `void rms_norm_weighted(bfloat16 *x, bfloat16 *gamma,
  bfloat16 *y, int32_t cols, float epsilon)` (the DESIGN.md §c contract;
  sibling of `rms_norm_aie2p.h:174-191` + a gamma pointer).  Plus
  `rms_norm_weighted_f32` for the np.float32 sanity stage and a portable
  generic template (any positive row length, scalar tail).
* Rounding model — ONE bf16 rounding, at the store (conv_even set inside the
  entry, `layer_norm_aie2p.h:198-205` idiom):
  * `x * gamma` is exact in f32 (two 8-significand-bit operands).
  * `inv_rms = s_hi + s_lo` ships verbatim (`rms_norm_aie2p.h:143-148`);
    the limbs are widened to exact f32 vectors and the shipped mac pair
    applies them to `xg`: `y = xg*s_hi + xg*s_lo`.  Both products are exact
    (16+8 <= 24 significand bits) and the f32 add rounds at 2^-24.
    The limb pair recovers inv_rms to ~2^-16 relative (the low limb carries
    the rounding residual), so the visible rounding is the bf16 store alone.
  * The aie_api does NOT allow mixed operand classes
    (`aie_api/detail/mul.hpp:50-91 is_valid_mul_op`: bf16xbf16 or
    floatxfloat, never bf16xf32) — this is why the limbs widen to f32
    instead of scaling the f32 `xg` by bf16 limbs directly.
* At hidden 3840: 60 x 64-lane chunks, no tail
  (`rms_norm_aie2p.h:77-79,118-121`), 96 calls per (96, 3840) block.

### 2.2 `adaln_modulate` — `y = x * (1 + scale) + shift`, one row per call

* Template: `eltwise/scale_shift.cc` — that file is the scale->shift
  two-phase building block, but its `mul_add` computes `a*b` **or** `a+b`
  per call (`is_mul` branch, `scale_shift.cc:17-27`), never `a*b+c`; the
  fused form is what this kernel adds.  Body is the acc-from-vector + mac
  idiom (`scale_shift.cc:43-45`, `add.cc:39-51`, `layer_norm_aie2p.h:39`):

  ```c++
  acc.from_vector(X);          // + x   (the implicit "1 + scale")
  acc = aie::mac(acc, X, S);   // + x*scale   (exact in f32)
  acc = aie::add(acc, B);      // + shift
  *pY++ = acc.to_vector<bfloat16>();   // ONE bf16 rounding (conv_even)
  ```

* Entry: `void adaln_modulate(bfloat16 *x, bfloat16 *scale, bfloat16 *shift,
  bfloat16 *y, int32_t cols)` (the DESIGN.md §c contract) + `..._f32`.
  Any positive row length: whole `AIE_BF16_LANES` vectors (32 on aie2p,
  `aie_arch.h:30`; `mul.cc:44` idiom) + scalar tail (`mul.cc:121-126`).
* The `1 + scale` never lands in bf16; f32 adds contribute <= 2^-24 of
  `mag = |x(1+s)| + |shift|`; the single store rounding dominates.

---

## 3. HARD-WON CONSTRAINTS (new, measured here)

### 3.1 A compute tile has only 2 input DMA channels

`from_name("npu2").core_dma_channels_in == 2` (and shim in/out = 2).  A
Worker can therefore be fed through at most TWO input ObjectFifos.  The
adaLN design packs each token's scale row and shift row into ONE stream
element (`[scale | pad | shift]`, a 1-D i8 arena carved with `memref.view`
in `core_fn` — the proven `dit_block_v1` idiom) to stay at 2-in/1-out.
Any future DiT-block design with >2 distinct input streams must pack or
time-multiplex.

### 3.2 AIE2P vector loads/store SILENTLY FLOOR misaligned addresses to 64 B

**The bug this caused, and its forensic trail:**

* First f32 adaLN run (`M=8, H=1027`, shift view at byte offset 4108) ran
  `ERT_CMD_STATE_COMPLETED` twice but computed `y[3+j] = x[3+j]*(1+s[3+j])
  + b[j]` — the shift operand read 3 f32 (12 bytes) early over the whole
  vector loop, while the scalar tail read correctly.
* Structured probes (`diag_adaln.py`) pinned it exactly: with the packed
  element a float ramp, the vector loop read the shift half at float index
  `1024+j` instead of `1027+j` (tail: correct `1027+j`).
* Mechanism: `aie_api/detail/ld_st.hpp:33-41` — `vector_ldst_align_v` on
  aie2p (`__AIE_ARCH__ == 21`) is **64 bytes** for 512/1024-bit vectors;
  `ld_st.hpp:66-77` (`floor_ptr`) documents "mimic hardware pointer
  semantics ... truncate the pointer to perform an aligned load" and
  "Rely on automatic HW truncation for enhanced performance".  8204 & ~63
  = 8192 = the observed read address.  The `restrict_vector_iterator`
  deref (`aie_api/detail/array_helpers.hpp:1554-1558`) carries the same
  requirement ("Insufficient alignment" assert).
* **There is no error**: `RUNTIME_ASSERT` is compiled out and the hardware
  ignores the low address bits for vector ldst — wrong numbers, silently.
  (The project's own warning class: "a plain row-major stream yields wrong
  numbers, not an error", `linalg.py:391-393`.)
* Fix (both layers):
  1. `designs.sb_row_layout` pads the packed element so the shift half
     starts at a multiple of 64 B (at H=3840 bf16 the split is already
     64-aligned — 7680 = 60x128 — so the pad is 0; at H=1027 f32 it is 13
     f32 columns).
  2. All four kernel contracts declare `alignments = ((i, 64), ...)`, so a
     misaligned `memref.view` now raises at DESIGN TIME
     (`kernel.py:534-545`) instead of computing wrong numbers.
* Corollary for every future kernel call: a buffer/row handed to a vector
  loop must start at a multiple of 64 B on aie2p.  Library kernels assume
  this too (their loads go through the same `load_v`); fifo element bases
  are bank-aligned (measured: 4096/16384/32768) so plain elements are safe,
  but views into packed arenas must be aligned explicitly.

### 3.3 Compile-side Python 3.12 is the ryzen-ai conda env, not `py -3.12`

`py -3.12` (Programs\Python\Python312) has numpy but NOT ml_dtypes;
`import aie.iron` fails there at
`aie/utils/hostruntime/__init__.py:11: from ml_dtypes import bfloat16`.
The working interpreter is
`<python312>\python.exe` (3.12.11,
numpy 1.26.4, ml_dtypes present) — `zimage-bfp16-npu\ENVIRONMENT.md:27`.
(This is the "refusal" recorded verbatim in section 7.)

### 3.4 `aie::mul` pairs operands of ONE class

`aie_api/detail/mul.hpp:50-91` (`is_valid_mul_op`): bf16xbf16 or
floatxfloat; a mixed bf16xf32 multiply does not exist.  Verified in the
headers before writing the kernels (it shaped `rms_norm_weighted`'s limb
widening).

---

## 4. CONSTRAINTS HONORED (from the task; not re-discovered)

* Environment set before `import aie`: `PEANO_INSTALL_DIR =
  _pylibs\llvm-aie\llvm-aie`, `NPU_CACHE_HOME =
  zimage-bfp16-npu\_npu_cache3`; `_pylibs` and `_pylibs\mlir_aie\python` on
  `sys.path`.
* PATH gains `test7\WStmp\driver376\npu_mcdm_stack_prod` for `xclbinutil`.
* The proven process-local `os.mkdir` shim (0o700 -> 0o777,
  `dit_block_v1_run.py:58-81`) is installed at `run.py` start; compiles use
  `use_cache=False`.  `prefix_symbols_in_object`'s
  `tempfile.TemporaryDirectory` staging worked every time (4 compiles).
* Dtype annotations use `np.ndarray[(shape), np.dtype[dtype]]` (never a bare
  marker class).
* `memref.view` sources are 1-D i8; MLIR types are built inside `core_fn`.
* DMA descriptors are <= 3 dims (mine are 1-2).
* No bfp16ebs8 involved (pure bf16/f32 kernels).
* L1 budget: depth-1 fifos; at H=3840 bf16 the adaLN design holds
  7680 + 15360 + 7680 = 30 720 B of buffers + 4096 B stack in the 64 KB.
* `np.float32` sanity stage ran FIRST (both kernels, H=1027 so the scalar
  tails execute too), then bf16.

---

## 5. VERIFICATION METHOD

* `refs.py` computes in float64 and rounds to bf16 once (RNE bit-twiddle,
  matching the kernels' `conv_even` stores).  Self-test order: f32
  definitional identities, then the bf16 model, byte contract, and a
  rounding-bound sanity check.
* Device outputs are compared elementwise:
  * rms_norm_weighted: ulp distance from `ref` (`ulp = 2**(exp-7)`);
  * adaln_modulate: ulp distance from `mag = |x(1+s)| + |shift|` (the
    single-rounded sum can cancel near zero while its terms do not).
* Data: `np.random.default_rng(seed).standard_normal`, x*3, gamma*0.5+1,
  scale*0.2, shift*0.1; inputs fed as the exact bf16 values the reference
  uses (round-tripped through `refs.round_bf16`).

---

## 6. MEASUREMENTS (from `run.py`; exact numbers)

(filled from the run logs — see the table below)

| kernel | stage | M | H | max abs err | max ulp err | tolerance used | result |
|---|---|---|---|---|---|---|---|
| rms_norm_weighted | f32 | 8 | 1027 | 4.768372e-07 | 0.0000 | f32: 2^-18*abs(ref) + 2^-20 | PASS |
| adaln_modulate | f32 | 8 | 1027 | 1.430511e-06 | 0.0000 | f32: 2^-18*mag + 2^-20 | PASS |
| rms_norm_weighted | bf16 | 96 | 3840 | 3.125000e-02 | 1.0000 | bf16: 2 ulp of abs(ref) | PASS |
| adaln_modulate | bf16 | 96 | 3840 | 0.000000e+00 | 0.0000 | bf16: 2 ulp of mag | PASS |

(Verbatim from the `run.py` summary table.  bf16 ulp = 2**(exp-7); declared
budget 2 ulps + 2^-126 floor — `refs.py TOL_ULPS`.)

* **Hidden 3840 FITS** — the target geometry ran as 96 tokens x 3840 hidden
  (the Z-Image block size), 1 row per kernel call, both kernels.
* `rms_norm_weighted` (bf16): worst element 1.0 bf16 ulp from the f64
  reference (3.125e-2 absolute, at an element whose ulp is 3.125e-2) —
  within the one-rounding theory of section 2.1.
* `adaln_modulate` (bf16): the device output is **bit-identical** to
  `refs.round_bf16(exact)` at all 368 640 elements (max abs err 0.0): the
  f32 accumulator chain never flipped a single rounding.
* Both kernels ran 2x per xclbin, `ERT_CMD_STATE_COMPLETED` on every run,
  run-to-run byte-identical.
* The f32 sanity geometry H=1027 exercises the scalar tail path
  (1027 = 32x32 + 3) in both kernels; max abs err at f32 rounding level.

---

## 7. REFUSALS / FAILURES, VERBATIM

No sandbox refusal occurred in this task: every file operation and command
was permitted (nothing needed `sandbox_permissions`).  The messages that
gated the work, verbatim:

1. Wrong compile interpreter (`py -3.12`):

   ```
   Traceback (most recent call last):
     File "<string>", line 1, in <module>
     File "<workspace>\_pylibs\mlir_aie\python\aie\iron\__init__.py", line 41, in <module>
       from aie.utils import (
     File "<workspace>\_pylibs\mlir_aie\python\aie\utils\__init__.py", line 10, in <module>
       from . import (
     File "<workspace>\_pylibs\mlir_aie\python\aie\utils\hostruntime\__init__.py", line 11, in <module>
       from ml_dtypes import bfloat16
   ModuleNotFoundError: No module named 'ml_dtypes'
   [exit code: 1]
   ```

2. First f32 adaLN verification (unshifted packed layout, H=1027) — a real
   numeric failure, fixed in section 3.2:

   ```
   [verify] adaln_modulate (f32) M=8 H=1027
   [verify]   max abs err   : 8.842750e-01
   [verify]   max ulp err   : 3419.1436
   [verify]   tolerance     : f32: |err| <= 3.815e-06*mag + 9.537e-07, mag = |x(1+s)|+|shift|
   [verify]   PASS: False
   ```

3. aiecc warning, benign (the f32 scalar tail lowers to soft-float; it
   links and the tail numbers are correct):

   ```
   [aiecc] (35/48) measured_stack_sizes.mlir...aie.mlir:11:10: warning: no stack size information for 2 function(s) this core reaches, so its requirement is at least 320 bytes and may be higher; compile the affected source with -fstack-size-section, or set stack_size_override on the kernel's external_func()/func.func declaration (Kernel(...)/ExternalFunction(...) in IRON): __muldi3, __mulsf3
   ```

4. argparse quirk in run.py's own CLI (a positional `nargs="*"` with
   `choices` and a list `default` type-checks the whole default list as one
   choice when no stage argument is given; fixed by dropping `choices` and
   validating in code):

   ```
   usage: run.py [-h] [--rows-f32 ROWS_F32] [--hidden-f32 HIDDEN_F32]
                 [--rows ROWS] [--hidden HIDDEN] [--seed SEED] [--runs RUNS]
                 [--skip-compile] [--skip-run]
                 [{refs,f32,bf16} ...]
   run.py: error: argument stages: invalid choice: "['refs', 'f32', 'bf16']" (choose from refs, f32, bf16)
   ```

5. My own refs.py self-test bug (an absolute epsilon perturbation against
   near-zero elements tripped its own bound; the tolerance model was fine,
   the test perturbation was wrong — changed to a relative 2^-9
   perturbation):

   ```
   Traceback (most recent call last):
     File "..._scratch\kernels\refs.py", line 281, in <module>
       _self_test()
     File "..._scratch\kernels\refs.py", line 272, in _self_test
       assert ok and mu <= 2.0, (mx, mu)
   AssertionError: (0.0078125, 4.0)
   [exit code: 1]
   ```

---

## 8. NOTES / OPEN POINTS

* The kernel entries set `conv_even` themselves (`layer_norm_aie2p.h:198`
  idiom), so the designs need no `conv_even` setup kernel (contract
  `setup=None`).
* `rms_norm_weighted.cc` includes `aie_kernels/norm/rms_norm_aie2p.h` (for
  kLanes/load_tail/store_tail/scalar_f32 and to keep the sum-of-squares
  arithmetic byte-identical to the shipped kernel); the header's `rms_norm`
  / `rms_norm_eps` externs therefore also end up in the object, unused and
  renamed by the digest symbol prefix — harmless.
* Z-Image's own RMSNorm epsilon is a model property (library default 1e-5,
  Gemma4 1e-6); the kernel takes `epsilon` at runtime, tests use 1e-5.
* The adaLN packed-element alignment fix is a LAYOUT fix, not a kernel
  fix: the kernel ABI `(x, scale, shift, y, cols)` is unchanged and any
  future design may feed it separate aligned fifos if channels allow.
* `diag_adaln.py` / `diag_const.py` are the repro probes for section 3.2.
  Against the fixed `out/f32_adaln.xclbin` they now measure PASS, exactly:
  `sb_ramp: max|y - expected| = 0.0`, `x_ramp: 0.0`, `s_ramp: 0.0`, and the
  constant probes read 5/2/8/13 exactly at every position.
