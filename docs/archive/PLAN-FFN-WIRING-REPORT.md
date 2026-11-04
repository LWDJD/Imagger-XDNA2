# The plan's images are wired into the product loop

**2026-11-03 · T22 adaptive, part 2 · new flag `T22_PLAN_FFN` (default 0) · the
delivered default path is untouched (proven, section 4)**

~~~
THE ONE SENTENCE: the per-shape plan that npu_synth builds can now be dispatched
by the product loop -- gu gate, gu up, host silu*up, dn -- bit-exactly against
the numpy model on the model's own BFP16 blocks, for any M the planner covers
(xd), with the delivered path still byte-for-byte what it was when the flag is 0.
~~~

---

## 1. The dispatch boundary, precisely

The product loop reaches the device FFN in exactly one place:

~~~
_scratch/image/t18_run.py :: block_step()          (line ~2090)
    if BIGM:                                       # ZIMAGE_BIGM=1
        _xn = rms_into(xm, P["ffn_norm1"], L.n1buf); _xn *= mod[2]
        _c, _b = cached_payloads(...) | load_payloads(key)
        _pair  = ("swiglu1","down1") if _xn.shape[0] == 96 else ("swiglu","down")
        if T22_S2_FFN32:      y = s2_product.ffn32_run(dev, _xn, key, tim)
        elif T22_DEVKILL:     y = devkill_ffn.bigm_ffn(dev, _xn, _c, _b, tim, imgs=...)
        else:                 y = prod_loop_bigm.bigm_ffn(dev, _xn, _c, _b, tim, imgs=_pair)
~~~

Contract: y is (SEQ, 3840) float32, the FFN output **before** ffn_norm2,
row-local (a padded row changes no real row). Everything after it (norm2, the
residual add, the per-layer record) is shared, so an FFN implementation is a
drop-in iff it honours that.

Two facts about the boundary that decide the wiring:

* it is **whole-layer**, not per row block. The delivered default product loop
  (webui/chunked.py :: block_step_pipelined) does not call it at all: it
  dispatches the delivered M=96 ffn image 11 times per layer inside its own
  pipeline. So a whole-layer plan needs ZIMAGE_CHUNK_PIPE=0; chunked.install()
  now **refuses** T22_PLAN_FFN=1 with the pipeline on instead of silently
  running the delivered FFN.
* the images are chosen **per layer** by tag. devkill_ffn.block reads
  dev.imgs["swiglu11"|"dn11"] and the payloads handed to it (_c, _b) are
  per-layer resident-cache rows. t18_run.Dev.__init__ is what turns
  BIGM_IMGS/BIGM_SIZES into registered xclbins and b_<tag>_{x,w,o} BOs on the
  image's own group ids 3/4/5. That registry is the hook: anything in it is
  registered and BO'd before the loop starts.

## 2. What the loop needs that the shipped plan does NOT provide

The plan (webui/npu_plan.plan) carries geometry and byte counts. It carries
none of the following, and each one is a place a wiring can be silently wrong:

| # | what the loop needs | what the plan gives | what had to be built |
|---|---|---|---|
| 1 | **BO set**: names, group ids, sizes | a_bytes / b_bytes / c_bytes per GEMM | plan_ffn.register(SEQ) puts (a,b,c) into t18_run.BIGM_SIZES[tag], so Dev allocates b_<tag>_x/w/o on groups 3/4/5. Plus **two lazy BOs** (_alt_bo, groups 4/5) for the up projection's W and C -- one W BO would be overwritten under the other dispatch's read |
| 2 | **packers** | nothing | pack_kk.pack_a_kk (A, bf16, kk-parametric) and pack_kk.pack_b_kk (B, a **permutation** of the model's BFP16 blocks -- never a re-encode) |
| 3 | **per-layer payload** | b_bytes only | s2_ffn32.load_blocks / pack_payloads + plan_ffn.payloads (LRU, optional disk cache). The product's resident CACHE_BG **cannot** be reused: it is in the delivered image's order, not the plan's kk order. block_step now skips cached_payloads/load_payloads entirely when the flag is on (that also skips the 30-layer in-place repack) |
| 4 | **drain** | c_bytes + tile_map | s2_ffn32.unpack_c_plan. **The shipped _cperm is a trap**: it loads a baked c_perm_<m>x<n>.npy and returns None (identity!) when the file is missing -- fine for m=32, silently wrong for m=16/48. The closed form (verified equal to all seven baked files) is used instead |
| 5 | **host glue** | nothing | the silu(gate)*up pass (10.8 M exp on 43 MB), the Mblk row padding per dispatch, the ndisp loop, the pad-row drop, and the **same-image-relaunch barrier** (3.4) |

## 3. What was implemented

### 3.1 s2_ffn32.ffn32_run(dev, xn, key, tim, b_cache, plan, blocks, stats, record)

The stub that raised NotImplementedError is now the executor for the **xd**
strategy (one M-tile per core, ndisp dispatches). Per dispatch block:

~~~
A(gu) = pack_a_kk(rows zero-padded to Mblk, m, kk, cols, cpc)   11.8 MB at 512^2
write b_gate -> gu W BO, write b_up -> alt W BO
launch gu(gate);  barrier(<another xclbin>);  launch gu(up);  wait both
read both C BOs -> unpack_c_plan (8x8 C order, plan tile map)  2 x 31.4 MB
h = bf16(silu(gate) * up)
A(dn) = pack_a_kk(h, ...)                                       31.4 MB
write b_dn -> dn W BO;  launch dn;  wait;  read;  unpack -> y[start:start+rows]
~~~

stats carries the same column names the delivered [devkill] line prints
(pack, wgu, gu, wdn, dn, read, dec, total) plus swiglu and barrier, so the
two arms' section columns are directly comparable. record is the
(out BO, W BO, sha-of-W) dispatch record the gate asserts on (off by default:
the shas cost 0.26 s/layer).

### 3.2 plan_ffn.py -- the product glue

register(SEQ) (called from configure(), before Dev()) resolves the plan with
prefer=built_plan_keys() -- the same affinity webui/shapes.npu_plan_for uses, so
the product runs the plan the manifest names -- checks the images are on disk,
and adds them to the registry. payloads(key) is the per-layer B provider (LRU
depth 3 = 398 MB, or T22_PLAN_BCACHE=1 + --prebuild for a 3.98 GB mmap cache).
ffn() is the drop-in call plus the section line.

### 3.3 t18_run.py -- three lines and one branch

The PLAN_FFN flag; plan_ffn.register(SEQ) after refresh_bigm_sizes; if PLAN_FFN:
as the first arm of the FFN dispatch; and the cached_payloads skip. Nothing
else. webui/chunked.py gained the refusal.

### 3.4 The barrier is not optional

gate and up are the **same image** (K=3840, N=10240), so the second launch is a
same-image relaunch. t18_run.Dev.barrier documents the measured law: the FFN
image never survives a same-image relaunch (a ~7 s state-8 stall, 30/30
layers); one launch of a **different xclbin** in between removes it completely
(~1.7-6.5 ms); and a second kernel handle on the same xclbin is **not** a fix
(it reports a false completion without doing the work). So ffn32_run issues
dev.barrier("att") between the two gu launches -- one per layer,
T22_PLAN_FFN_BARRIER=0 disables it for an A/B.

### 3.5 Two bugs found on the way

* **pack_kk._numba_kernels() returned False, not None, on the second call when
  numba is absent** -- pack_a_kk then did nb[0] on a bool and raised TypeError.
  Invisible until now because every existing caller runs on py310 (numba
  present); the host-only py312 leg hits it on the second pack. Fixed to
  return _NB or None.
* **s2_ffn32._cperm silently returns the identity for any (m, n) without a
  baked file.** The plan's geometries are m=16/48, so a drain built on it would
  have been wrong for every adaptive shape and right for the delivered one.
  The closed-form permutation is used and checked (4.1).

## 4. The host gate (no device)

All four run on **py312, host only**. Every one is a pass/fail script, not a
notebook.

### 4.1 gate_ffn32.py --all -- the fake-device gate

The "device" is **functional**, not a stub: it decodes the A and B bytes the
BOs actually hold (unpack_a_kk / unpack_b_kk), runs the GEMM with the kernel's
own accumulation model (per-k-step bf16 rounding of C, A quantised to bfp16 by
conv_even exactly as mm_bfp_mixed.cc does), and writes the C BO in the device's
8x8 C order. A layout bug therefore shows up as a number, not a crash. The
reference is the numpy FFN on the model's own blocks -- it uses **neither** the
packers **nor** the tile map, so an error in either cannot cancel.

~~~
[gate] new      M=96    layers.0 corr 1.000000000 maxabs 0 exact 1.0000 record(rep/dev) True/True
[gate] new      M=96    layers.1 corr 1.000000000 maxabs 0 exact 1.0000 record(rep/dev) True/True
[gate] glue     M=96    layers.0 corr 1.000000000 maxabs 0 exact 1.0000   <- plan_ffn.ffn()
[gate] new      M=1056  layers.0 corr 1.000000000 maxabs 0 exact 1.0000 record(rep/dev) True/True
[gate] w        M=1056  layers.0 corr -0.001311850 maxabs 3224
[gate] a        M=1056  layers.0 corr  0.007712145 maxabs 2288
[gate] ALL PASS -- layouts exact, the record is the plan's, and the gate bites   (215.1 s)
~~~

What it asserts:

1. **y is bit-identical to the numpy FFN** at M=96 (2 layers, 6 of 32 slots
   real, so the dummy-tile path runs) and at M=1056 = 512^2 (1 layer):
   corr 1.0, maxabs 0, exact_frac 1.0. Not "close" -- the same bits.
2. **the (out BO, W BO, sha-of-W) record equals the plan's**, twice: once from
   what ffn32_run reports, and once from what the *device observed* -- the sha
   is taken from the W BO's contents at the moment the kernel is told to run,
   so a write-order bug (a dispatch reading a BO still being written) fails.
   At 512^2: gu_o / gu_w / b1f2a402bad8f052, gu_o+up / gu_w+up /
   89704365e2978426, dn_o / dn_w / 5926e16f02b41650.
3. **B is a permutation of the model's blocks** (multiset identity, 4,915,200
   9-byte blocks per matrix).
4. **one barrier per layer** (the same-image relaunch).
5. **the gate bites**: negctl_w (up fed the gate payload) collapses to
   corr -0.0013, negctl_a (the dn A's k-steps permuted inside the packed
   payload -- a real k-order error) collapses to corr 0.0077.
6. **the product entry point, not just the executor**: the `glue` arm calls
   `plan_ffn.ffn()` -- the function `t18_run.block_step` actually invokes, with
   its plan cache, its payload LRU (a real pack from `weights/block0.npz`) and
   its section line -- and is also bit-identical. 26 assertions, 0 failures.

**Why there is no delivered-path record to diff against.** devkill_ffn runs a
*fused* gate+up+silu+mul image (N=20480, kk=8) plus a down image whose A is the
gu image's C byte for byte. The plan runs two separate gu dispatches at kk=32
and a down image whose A is a freshly packed (Mblk, 10240) tile. Different
images, different kk, different N split -- the two records cannot be
element-wise equal, and a gate that pretended otherwise would measure nothing.
What the two paths *do* share is the model's BFP16 blocks, and 4.2 shows they
do not even share those.

### 4.2 check_payload_identity.py -- "B is permuted, never re-encoded", and the finding that decides the 512^2 comparison

~~~
the plan's B(w1), B(w3), B(w2)   9-byte block multiset IDENTICAL to the model's
the DELIVERED payload            NOT a permutation:
  work/L0_container.npy vs model(w1+w3)  99.754 % identical, max|d| = 0.0078125
  work/b3/L0_b3.npy     vs model(w2)     99.667 % identical, max|d| = 0.0078125
~~~

0.0078125 is exactly **one bfp16 step**. So the delivered image family is fed a
**re-encode** of the weights (the small fraction of non-canonical blocks come
back one step different: 0.25 % of the VALUES differ, by exactly one bfp16
step), while the plan path is fed the model's own bytes in a different order. The two paths therefore multiply by slightly different
matrices *before* any geometry difference -- which is why **byte-identical
latents between the delivered path and the plan path are impossible by
construction**, and why the byte-identity gate has to live on the delivered
path with the flag **off**.

### 4.3 plan_ffn.py --selftest / --register-check / check_additive.py

~~~
plan strategy is xd                                   OK  xd_m48_kk32_Mblk1056_n384-320-384_4a211e7e
cperm(m,n) == c_perm_<m>x<n>.npy  (all 7 baked files) OK
pack_a_kk/unpack_a_kk round-trip (exact)              OK
A payload is the plan's a_bytes                       OK  11,796,480
B payload sizes == the plan's b_bytes                 OK
B(w1)/B(w3)/B(w2) pure permutation of the model       OK
[planffn] selftest: OK

b_<tag>_gu_x  11,796,480 B   b_<tag>_gu_w  44,236,800 B   b_<tag>_gu_o  31,457,280 B
b_<tag>_dn_x  31,457,280 B   b_<tag>_dn_w  44,236,800 B   b_<tag>_dn_o  11,796,480 B
b_<tag>_gu_w+up / _o+up (lazy, groups 4/5)  44,236,800 / 31,457,280 B
[planffn] register-check: OK

PASS _scratch/image/t18_run.py       3 delivered line(s) rewritten, 0 unannounced
PASS webui/chunked.py                0 delivered line(s) rewritten, 0 unannounced
PASS _scratch/util/place/pack_kk.py  1 delivered line(s) rewritten, 0 unannounced
PASS _scratch/util/place/s2_ffn32.py 12 delivered line(s) rewritten, 0 unannounced
[additive] PASS -- the delivered path is intact; the new flag is 0 by default
~~~

check_additive.py is a **subsequence check**: every line of the pristine copy
must still appear, in order, in the modified file, and any line that does not
must be on a named allowlist. The allowlist is 3 control-flow lines in t18_run
(if T22_S2_FFN32: -> elif, if CACHE_ALL ... -> elif, and a _c is not None guard
on a diagnostic dump), 1 in pack_kk (return _NB -> return _NB or None), and the
NotImplementedError stub in s2_ffn32 that this change was asked to replace.
**0 unannounced changes.**

## 5. The 512^2 device comparison -- NOT RUN, and why

The device is owned by another agent for landscape validation this session, so
the 8-step run was **not** executed. What is in the tree instead:

* _scratch/util/place/PLAN-FFN-512-AB.md -- the exact procedure: both arms
  ZIMAGE_BIGM=1 T22_DEVKILL=1 ZIMAGE_CHUNK_PIPE=0, the only difference
  T22_PLAN_FFN=0 -> 1, memory mode pinned (the config.py:433 trap), and the
  --expect-baseline 4cbf4f4a1b98 check.
* _scratch/util/place/compare_512_ab.py -- reads the two job records and
  asserts memory-mode equality, the delivered baseline, latents byte-identity,
  loop_s within +-2 s, and prints the median of every section column. Smoke-
  tested on the recorded 512_devkill_8a job, whose delivered columns are:

~~~
pack 8.5  wgu 3.6  gu 45.9  wdn 0.0  dn 31.2  read 2.4  dec 10.7  total 102.4 ms
~~~

**And the two answers the task asked for are, on the evidence, already known:**

1. **latents byte-identical: NO.** Two independent causes, both measured here:
   the delivered W is a re-encode (4.2) and the GEMM geometry differs (kk=8
   fused + kk=32 dn vs kk=32 gu x2 + kk=32 dn). Expect corr 0.9999+, not
   equality. The byte-identity requirement belongs on the flag-off path, and
   4.3 proves that path is untouched.
2. **loop_s within +-2 s: NO, and predictably so.** Per layer the plan wins on
   the device (2 x ~19 ms gu + ~19 ms dn + ~6 ms barrier = ~63 ms vs the
   delivered 77 ms of dispatch time at 512^2, because kk=32 is the better rate)
   and loses on the host: the delivered image **fuses** silu(g)*u into the gu
   dispatch, the plan's images are pure GEMMs, so the elementwise pass
   (10.8 M exp over 43 MB) lands on the host at ~50 ms, plus ~7 ms of extra
   pack/read/dec and ~6 ms of extra W write. Net **~+64 ms/layer = +1.9 s/step =
   +15 s per 8-step run**.

So the honest verdict on this wiring is: **it makes the plan reachable, not
faster.** The plan's advantage is the device rate; the delivered path's
advantage is that its image does the elementwise work. Flipping the sign needs
one of the two follow-ups in section 6.

## 6. What still cannot be chained, and why

| gap | why it cannot be chained today | what it would take |
|---|---|---|
| **attention / ip above SEQ=304** | there is no device image for dense attention above SEQ=304 (the key axis must fit L1; the key-blocked online-softmax kernel is designed but ungated). The chunked path runs attention on the host in f32, and this change does not touch it -- so a large shape keeps host attention no matter what the FFN does | the T22 campaign-1 kernel + its four gates. The plan covers only the linear half |
| **the t33 strategy** | ffn32_run raises for strategy != "xd". t33 (768^2, 896^2, 1024^2 -- built and device-validated) is a different executor: per-column A elements of T*a_leaf, verify_t33.pack_a_t33, and a per-n-tile C drain with the cwait await (the delivered t33 form corrupts the last n-tile, section 7.2 of ADAPTIVE-NPU-REPORT) | a second executor behind the same contract: pack_a_t33 + a column-element A BO + the per-n-tile drain loop |
| **the projection (proj image)** | only gu/dn are registered. The projection still runs prod_devkill.proj4 (plain15, 4 dispatches/layer) or bigm_proj4. Wiring it is a different call site (Loop.attention), not the FFN branch | s2_product.proj4 already exists for the *delivered* 32-core images; it needs the same plan-tag treatment |
| **the host swiglu** | the plan's images are pure GEMMs. The elementwise pass is ~50 ms/layer at 512^2, which is the whole of the regression in section 5 | a fused gu+swiglu image at the plan's kk (one dispatch instead of two, no host pass), **or** the device-side handoff of gu's C BO into dn's A BO -- which needs A(mm_bfp, kk) == C(mm_bfp) at that kk. That equality holds for the delivered dn32 image (k=32) but has **not** been checked for the plan's m=48/kk=32 geometry |
| **byte-identical latents vs the delivered path** | impossible by construction: the delivered payload is a re-encode (4.2) and the geometry differs | nothing -- this is a property of the two image families, not a defect to fix |
| **the delivered path's device byte-identity** | proven at the source level (4.3) and by construction (flag 0), not by a device run | arm A of section 5: the flag-off 8-step run must reproduce 4cbf4f4a1b98 (fast) / 2abf91331200 (low) |

## 7. Files

**New**

| file | what |
|---|---|
| _scratch/util/place/plan_ffn.py | the product glue: register / payloads / ffn, LRU + disk payload cache, --check / --register-check / --selftest / --prebuild |
| _scratch/util/place/gate_ffn32.py | the fake-device gate (functional kernel, dispatch record, two negative controls) |
| _scratch/util/place/check_additive.py | the subsequence proof that the delivered path is intact |
| _scratch/util/place/check_payload_identity.py | permutation vs re-encode, and the closed-form C perm |
| _scratch/util/place/compare_512_ab.py | the 512^2 A/B report (latents sha, loop_s, section columns) |
| _scratch/util/place/PLAN-FFN-512-AB.md | the device procedure + the predicted numbers |

**Modified**

| file | change |
|---|---|
| _scratch/util/place/s2_ffn32.py | ffn32_run implemented for xd (+ cperm, unpack_c_plan, load_blocks, pack_payloads, _alt_bo, _barrier); the NotImplementedError stub is gone |
| _scratch/image/t18_run.py | PLAN_FFN flag, plan_ffn.register(SEQ) in configure(), one if PLAN_FFN: arm in block_step, and the cached_payloads skip |
| webui/chunked.py | refuses T22_PLAN_FFN=1 with the 96-row pipeline (the branch it needs is never reached there) |
| _scratch/util/place/pack_kk.py | return _NB or None -- the numba-absent second call raised TypeError |

**Reproduce (host only, py312)**

~~~
py312 _scratch/util/place/plan_ffn.py --selftest
py312 _scratch/util/place/plan_ffn.py --m 1056 --register-check
py312 _scratch/util/place/check_payload_identity.py
py312 _scratch/util/place/check_additive.py
py312 _scratch/util/place/gate_ffn32.py --all          # ~4 min
~~~
