# The recorded set -- every environment variable the runtime reads

Generated from knobs.py, registry sha `fe1c0a91f013dada42e765687334fbc0777118e54fad12438b055423beb47a96`.

177 variables, derived by env_audit.py from the 68-file import closure of 21
entrypoints (349 read sites).  The effect class decides whether an A/B has to
match it: **bytes, bytes-risk, path** enter the ARM IDENTITY; the rest are recorded but cannot
move a byte.

| class | count | meaning |
|---|---:|---|
| bytes | 4 | proven to change the latents bytes (evidence is a measurement) |
| bytes-risk | 70 | selects arithmetic/dtype/kernel/order; byte-neutrality unproven |
| path | 34 | selects a dispatch path whose byte-neutrality was measured |
| timing | 12 | wall clock only |
| diag | 32 | logging/dumping/probing only |
| infra | 25 | paths, caches, memory, interpreters, addresses |

## The two knobs that caused the finding

| knob | class | default | recorded before? | evidence |
|---|---|---|---|---|
| `T22_BREPACK_CACHE` | **bytes** | '0' | NO (0 of 629 job.json) | MEASURED: imaggers/par/gate/REPORT.md A.4/A.6 -- the two B states produce two byte-stable arrays 8.91e-4 apart at 512^2x1 (corr 0.999108509136); 4/4 runs each |
| `T22_PLAN_FUSE_GU` | **bytes** | '0' | NO (0 of 629 job.json) | MEASURED: imaggers/par/gate/REPORT.md A.6 -- four jobs, identical env_flags, two alternating arrays, split exactly by the 'gu=FUSED' log line |

## Every knob, by class

### bytes -- proven to change the latents bytes (evidence is a measurement)

| knob | default | conflict | readers | evidence |
|---|---|---|---|---|
| `T22_BIGM_ORDER` | 'map' |  | imaggers/block/bigm/brepack_cache.py:666, imaggers/block/bigm/prod_loop_bigm.py:636 | CODE: imaggers/block/bigm/prod_loop_bigm.py:636 selects the B order; the log line '[bigm] B order: delivered (T22_BIGM_ORDER=raw); no repack' is in 2 jobs |
| `T22_BREPACK_CACHE` | '0' |  | imaggers/block/bigm/brepack_cache.py:124, imaggers/block/bigm/prod_loop_bigm.py:613 (+1) | MEASURED: imaggers/par/gate/REPORT.md A.4/A.6 -- the two B states produce two byte-stable arrays 8.91e-4 apart at 512^2x1 (corr 0.999108509136); 4/4 runs each |
| `T22_PLAN_FUSE_GU` | '0' |  | imaggers/util/place/plan_ffn.py:94, webui/capability.py:135 (+2) | MEASURED: imaggers/par/gate/REPORT.md A.6 -- four jobs, identical env_flags, two alternating arrays, split exactly by the 'gu=FUSED' log line |
| `T22_REF_SMNB` | False |  | imaggers/image/t3_real.py:506, imaggers/image/t3_real.py:551 (+3) | MEASURED: imaggers/repro/REPORT.md:35 -- 512^2x8 latents 6b90391431797621 without it vs 2abf91331200d069 with it, same proved path |

### bytes-risk -- selects arithmetic/dtype/kernel/order; byte-neutrality unproven

| knob | default | conflict | readers | evidence |
|---|---|---|---|---|
| `PATH` | '' |  | imaggers/block/step4_run_ffn.py:353, imaggers/block/step4_run_ffn.py:353 (+4) | AUTO: unknown family -- conservatively in the arm identity |
| `T18_CAP_X` | '' |  | imaggers/image/t18_run.py:1231 | AUTO: name-family heuristic -- not verified against the code |
| `T18_SKIP_ATT` | '' |  | imaggers/image/t18_run.py:2011, imaggers/image/t22_t18_base.py:1187 | AUTO: name-family heuristic -- not verified against the code |
| `T22_ADAPT_M` | '1056' |  | imaggers/image/t18_run.py:396, webui/capability.py:135 (+3) | CODE: imaggers/image/t18_run.py:396 -- resolves the shape's M |
| `T22_AF9M` | '0' |  | imaggers/block/bigm/prod_loop_bigm.py:843 | CODE: imaggers/block/bigm/prod_loop_bigm.py:835 |
| `T22_ATTNWIRE` | '0' |  | imaggers/block/bigm/prod_loop_bigm.py:852 | CODE: imaggers/block/bigm/prod_loop_bigm.py:844 -- attention wiring |
| `T22_ATTN_A2` | False |  | webui/chunked.py:288, webui/chunked.py:477 (+3) | CODE: webui/chunked.py:279 -- f32 matmul + fused numba f32 softmax in place |
| `T22_ATTN_BF16` | '0' |  | webui/chunked.py:459 | CODE: webui/chunked.py:261 -- dt = torch.bfloat16 for the qk matmul (f32 accumulate) |
| `T22_ATTN_D` | False |  | webui/chunked.py:288, webui/chunked.py:484 (+3) | CODE: webui/chunked.py:286 -- bf16 qk + bf16 softmax + bf16 pv |
| `T22_ATTN_E` | False |  | webui/chunked.py:288, webui/chunked.py:473 (+3) | CODE: webui/chunked.py:275 -- bf16 qk + numba f32 in-place softmax + f32 pv |
| `T22_ATTN_NB` | '0' |  | webui/chunked.py:465 | CODE: webui/chunked.py:267 -- numba in-place softmax variant |
| `T22_ATTN_NBTHREADS` | '0' |  | webui/chunked.py:513 | CODE: webui/chunked.py:315 -- numba threads for the fused softmax; a parallel row sum can reorder |
| `T22_ATTN_OBUF` | '0' |  | webui/chunked.py:508 | CODE: webui/chunked.py:310 -- one persistent output buffer |
| `T22_ATTN_QB` | '0' |  | webui/chunked.py:453 | CODE: webui/chunked.py:255 -- rows per score block; changes the matmul batch shape |
| `T22_ATTN_REUSE` | '0' |  | webui/chunked.py:491 | CODE: webui/chunked.py:293 -- reused out= buffers |
| `T22_ATTN_STRICT` | False |  | webui/chunked.py:307, webui/config.py:238 (+2) | CODE: webui/chunked.py:307 + config.py:235 -- refuses a variant even when a hatch is open, so it also decides which kernels execute |
| `T22_BB_FASTK32` | '0' |  | imaggers/block/bigm/prod_bigborder.py:198 | CODE: imaggers/block/bigm/prod_bigborder.py:198 -- k32 border path |
| `T22_BIGM_B3_DUAL` | '1' |  | imaggers/block/bigm/brepack_cache.py:662, imaggers/block/bigm/prod_loop_bigm.py:572 (+2) | CODE: imaggers/block/bigm/prod_loop_bigm.py:572 -- B3 payload layout |
| `T22_BIGM_DEVHANDOFF` | '0' |  | imaggers/block/bigm/brepack_cache.py:658, imaggers/block/bigm/prod_loop_bigm.py:221 (+5) | CODE: imaggers/block/bigm/prod_loop_bigm.py:647-649 -- without it the loop only offers {n128,n64} B payloads and devkill_ffn raises |
| `T22_BIGM_PROJ_ORDER` | 'image' |  | imaggers/block/bigm/prod_loop_bigm.py:705 | CODE: imaggers/block/bigm/prod_loop_bigm.py:697 -- projection order |
| `T22_BREPACK_BUILD` | '1' |  | imaggers/block/bigm/brepack_cache.py:890 | CODE: imaggers/block/bigm/brepack_cache.py:890 |
| `T22_BREPACK_FASTK32` | '1' |  | imaggers/block/bigm/brepack_cache.py:438 | CODE: imaggers/block/bigm/brepack_cache.py:438 -- fast k32 repack |
| `T22_BREPACK_PREP` | '' |  | imaggers/block/bigm/brepack_cache.py:314 | CODE: imaggers/block/bigm/brepack_cache.py:314 -- pre-permutation payload |
| `T22_DEVKILL_G` | '15' |  | imaggers/devkill/prod_devkill.py:38 | CODE: imaggers/devkill/prod_devkill.py:38 -- the G=15 projection geometry |
| `T22_DEVKILL_PACK` | '1' |  | imaggers/devkill/devkill_ffn.py:326 | CODE: imaggers/devkill/devkill_ffn.py:326 -- which B payload the FFN consumes |
| `T22_DEVKILL_PROJ_IMG` | 'plain15' |  | imaggers/devkill/prod_devkill.py:24 | CODE: imaggers/devkill/prod_devkill.py:24 -- projection image (plain15) |
| `T22_DEVKILL_WDB` | '1' |  | imaggers/devkill/devkill_ffn.py:350 | CODE: imaggers/devkill/devkill_ffn.py:350 -- double-buffer mode |
| `T22_FASTGLUE` | '0' |  | imaggers/util/glue/install.py:628, webui/capability.py:135 (+5) | CODE: imaggers/util/glue/install.py:625 + webui/chunked.py:1014 -- installs the 8 fused host-glue hooks |
| `T22_FASTGLUE_HUNKS` | '1,2,3,4' |  | imaggers/util/glue/install.py:945 | CODE: imaggers/util/glue/install.py:911 -- which hunks install |
| `T22_FASTGLUE_SKIP` | '' |  | imaggers/util/glue/install.py:662 | CODE: imaggers/util/glue/install.py:644 -- hunks to skip |
| `T22_FASTGLUE_THREADS` | '8' |  | imaggers/util/glue/install.py:649 | AUTO: name-family heuristic -- not verified against the code |
| `T22_FAST_GLUE` | True | yes | imaggers/image/t18_run.py:196, webui/bridge.py:97 (+3) | CODE: imaggers/image/t18_run.py:196 -- t18's own rms/RoPE in the drained layout |
| `T22_FFN_KK32` | '0' |  | imaggers/devkill/devkill_ffn.py:43, imaggers/util/place/kk_prod.py:82 | CODE: imaggers/devkill/devkill_ffn.py:43 -- kk32 FFN segment |
| `T22_FFN_PROBE` | '' |  | imaggers/image/t22_run.py:249 | AUTO: name-family heuristic -- not verified against the code |
| `T22_FFN_PROBE_MODE` | 'kcdiff' |  | imaggers/image/t22_run.py:252, imaggers/image/t22_run.py:279 | AUTO: name-family heuristic -- not verified against the code |
| `T22_FFN_TP` | '0' |  | imaggers/devkill/devkill_ffn.py:275 | CODE: imaggers/devkill/devkill_ffn.py:275 -- FFN thread partition |
| `T22_GLUE2` | True | yes | imaggers/image/t18_run.py:205, webui/bridge.py:99 (+3) | CODE: imaggers/image/t18_run.py:205 -- copy-free drain/pack |
| `T22_GLUE3` | '0' |  | imaggers/image/t18_run.py:208 | CODE: imaggers/image/t18_run.py:208 -- absmax without the 3 temporaries |
| `T22_GLUE_ABSMAXP` | '0' |  | imaggers/util/glue/fast_glue.py:274 | CODE: imaggers/util/glue/fast_glue.py:274 -- parallel absmax |
| `T22_GLUE_HALF_OK` | False |  | webui/chunked.py:306, webui/config.py:238 (+2) | CODE: webui/chunked.py:306 + config.py:237 -- a hatch that lets a partially installed glue variant RUN instead of being refused, so it changes which kernels exe |
| `T22_LOAD_BLOCKS_ROWMAJOR` | '0' |  | imaggers/util/place/s2_ffn32.py:937 | CODE: imaggers/util/place/s2_ffn32.py:937 -- block load order |
| `T22_OVL` | '0' |  | imaggers/devkill/devkill_ffn.py:211, imaggers/util/glue/install.py:652 | CODE: imaggers/util/glue/install.py:634 -- host/device overlap segmentation |
| `T22_OVL_THREADS` | '0' |  | imaggers/util/glue/install.py:654 | CODE: imaggers/util/glue/install.py:636 |
| `T22_OVL_WGU` | '1' |  | imaggers/devkill/devkill_ffn.py:212, imaggers/util/glue/install.py:653 | CODE: imaggers/util/glue/install.py:635 |
| `T22_P33` | '0' |  | imaggers/util/glue/install.py:781 | CODE: imaggers/util/glue/install.py:747 -- the +p33 projection variant |
| `T22_PFILL` | '0' |  | imaggers/image/t18_run.py:236 | AUTO: name-family heuristic -- not verified against the code |
| `T22_PLAN_BCACHE` | '0' |  | imaggers/util/place/plan_ffn.py:90 | CODE: imaggers/util/place/plan_ffn.py:90 -- plan B cache |
| `T22_PLAN_B_BYTES` | '0' |  | imaggers/util/place/plan_ffn.py:89 | CODE: imaggers/util/place/plan_ffn.py:89 |
| `T22_PLAN_B_LAYERS` | '0' |  | imaggers/util/place/plan_ffn.py:86 | CODE: imaggers/util/place/plan_ffn.py:86 |
| `T22_PLAN_FAST_DRAIN` | '1' |  | imaggers/util/place/s2_ffn32.py:810 | CODE: imaggers/util/place/s2_ffn32.py:810 |
| `T22_PLAN_FFN_BARRIER` | '1' |  | imaggers/util/place/s2_ffn32.py:1691 | CODE: imaggers/util/place/s2_ffn32.py:1691 -- inter-launch barrier |
| `T22_PLAN_FFN_BARRIER_IMG` | 'att' |  | imaggers/util/place/plan_ffn.py:560, imaggers/util/place/s2_ffn32.py:1693 | CODE: imaggers/util/place/plan_ffn.py:560 |
| `T22_PLAN_FFN_M` | '' |  | imaggers/util/place/plan_ffn.py:154 | CODE: imaggers/util/place/plan_ffn.py:154 -- plan M override |
| `T22_PLAN_FFN_PROBE` | '0' |  | imaggers/devkill/devkill_ffn.py:396, imaggers/devkill/devkill_ffn.py:435 (+3) | AUTO: name-family heuristic -- not verified against the code |
| `T22_PLAN_FFN_PROBE_ALT` | '' |  | imaggers/image/t18_run.py:2152 | AUTO: name-family heuristic -- not verified against the code |
| `T22_PLAN_FUSE_FILL` | '1.0' |  | imaggers/util/place/plan_ffn.py:95 | CODE: imaggers/util/place/plan_ffn.py:95 -- fused fill factor |
| `T22_REF_BF16` | '0' |  | imaggers/image/t3_real.py:303 | CODE: imaggers/image/t3_real.py:303 -- bf16 in the refiner reference path |
| `T22_REF_FAST` | '0' |  | imaggers/image/t3_real.py:534, imaggers/image/t3_real.py:502 | CODE: imaggers/image/t3_real.py:534 -- torch bmm for the two refiner matmuls |
| `T22_REF_FFN` | 'auto' |  | imaggers/image/ref_ffn_block.py:79 | CODE: imaggers/image/ref_ffn_block.py:79 -- selects the refiner FFN block path |
| `T22_REF_FFN_ROWS` | '0' |  | imaggers/image/ref_ffn_block.py:115 | CODE: imaggers/image/ref_ffn_block.py:115 -- rows per refiner FFN block |
| `T22_REF_QCHUNK` | 'auto' |  | imaggers/image/ref_attn_block.py:106, webui/shapes.py:517 | CODE: imaggers/image/ref_attn_block.py:106 -- refiner attention chunking |
| `T22_REF_RMS` | '0' |  | imaggers/image/t3_real.py:243 | CODE: imaggers/image/t3_real.py:243 -- numba rmsnorm kernel |
| `T22_REF_TORCH` | '0' |  | imaggers/image/t3_real.py:526, imaggers/image/t3_real.py:502 | CODE: imaggers/image/t3_real.py:526 -- torch SDPA for the refiner attention |
| `T22_S2_FFN32` | '0' |  | imaggers/image/t18_run.py:225 | CODE: imaggers/image/t18_run.py:225 -- 32-core FFN GEMM segment |
| `T22_S2_FFN32_DIR` | '' |  | imaggers/image/t18_run.py:226 | CODE: imaggers/image/t18_run.py:226 -- image directory for the S2 FFN set |
| `T22_S2_PROJ32` | '0' |  | webui/chunked.py:1292 | CODE: webui/chunked.py:1037 -- 32-core fused projection |
| `T22_SKIP_FFN` | '' |  | imaggers/image/t22_run.py:737 | AUTO: name-family heuristic -- not verified against the code |
| `ZIMAGE_CHUNK_THREADS` | 0 |  | imaggers/image/ref_attn_block.py:198, imaggers/image/t3_real.py:322 (+5) | CODE: webui/chunked.py:100-105 + webui/config.py:143-145 -- the host attention pool |
| `ZIMAGE_VAE_BARRIER` | '1' |  | imaggers/vae/t21_3b3_chain.py:382 | AUTO: name-family heuristic -- not verified against the code |
| `ZIMAGE_VAE_RESNET_REWRITE` | '0' |  | imaggers/vae/t21_3b3_chain.py:639 | AUTO: name-family heuristic -- not verified against the code |

### path -- selects a dispatch path whose byte-neutrality was measured

| knob | default | conflict | readers | evidence |
|---|---|---|---|---|
| `AD_FFN_BARRIER` | '0' |  | webui/chunked.py:67 | CODE: webui/chunked.py:67 -- inter-launch barrier in the pipeline |
| `AD_FFN_REFILL` | '0' |  | webui/chunked.py:66 | CODE: webui/chunked.py:66 -- refill the FFN buffer per block |
| `T18_PAYLOAD_ORDER` | 'last' |  | imaggers/image/t18_run.py:2331, imaggers/image/t22_t18_base.py:1298 | CODE: imaggers/image/t18_run.py:2331 -- payload fill order |
| `T20_FFN_BARRIER` | '1' |  | imaggers/image/t18_run.py:568, imaggers/image/t22_t18_base.py:256 | CODE: imaggers/image/t18_run.py:568 -- the dummy ip launch between FFN row blocks |
| `T21_3B2_FAST_REF` | '1' |  | imaggers/vae/t21_3b2_plan.py:113 | CODE: imaggers/vae/t21_3b2_plan.py:113 |
| `T22_DEVKILL` | '0' |  | imaggers/block/bigm/prod_loop_bigm.py:652, imaggers/image/t18_run.py:364 (+5) | CODE: webui/chunked.py:982 + imaggers/image/t18_run.py:364 -- G=15 projection + RT=11 FFN |
| `T22_DEVKILL_BREPACK` | '1' |  | imaggers/image/t18_run.py:2780 | CODE: imaggers/image/t18_run.py:2780 |
| `T22_DEVKILL_PREWARM` | '1' |  | webui/chunked.py:1281 | CODE: webui/chunked.py:1026 |
| `T22_FASTGLUE_STRICT` | '0' |  | imaggers/util/glue/install.py:663 | CODE: imaggers/util/glue/install.py:645 -- fail-open vs fail-closed install |
| `T22_FINAL110` | '0' |  | imaggers/devkill/prod_devkill.py:21 | CODE: imaggers/devkill/prod_devkill.py:21 |
| `T22_GLUE_FAIL_OPEN` | False |  | webui/chunked.py:305, webui/chunked.py:1270 (+3) | CODE: webui/chunked.py:1015 |
| `T22_HOSTKILL` | '0' |  | imaggers/image/t18_run.py:358, imaggers/image/t18_run.py:425 (+5) | CODE: webui/chunked.py:975 + imaggers/image/t18_run.py:358 -- DMA-side A pack |
| `T22_P10` | '1' | yes | imaggers/image/t18_run.py:312, webui/bridge.py:76 (+2) | MEASURED IDENTICAL: webui/config.py:282-311 -- 0/4096 B, sha dd53e57e, 4/4 runs |
| `T22_PLAN_FFN` | '0' |  | imaggers/image/t18_run.py:235, webui/capability.py:135 (+4) | CODE: imaggers/image/t18_run.py:235 + webui/chunked.py:943 -- the plan FFN arm |
| `T22_PLAN_PREWARM` | '1' |  | imaggers/image/t18_run.py:2701 | CODE: imaggers/image/t18_run.py:2701 -- payload prewarm before the loop |
| `T22_PREOVL` | '0' |  | imaggers/image/t18_run.py:222 | MEASURED IDENTICAL: imaggers/image/t18_run.py:217-222 (gate_preovl.py) |
| `T22_QKV` | '1' | yes | imaggers/image/t18_run.py:271, webui/bridge.py:59 (+2) | MEASURED IDENTICAL: webui/config.py:259-278 -- latents sha 1275a93a, PNG e7aef10c, 15/15 audit |
| `T22_XPACK_ONCE` | '0' |  | imaggers/image/t18_run.py:274, webui/bridge.py:61 (+1) | CODE: imaggers/image/t18_run.py:274 -- host X-pack reuse |
| `ZIMAGE_ADAPT` | '0' |  | webui/build_shape.py:160, webui/capability.py:135 (+4) | CODE: webui/build_shape.py:160 + webui/shapes.py:706 |
| `ZIMAGE_BIGM` | '0' |  | imaggers/image/t18_run.py:317, webui/capability.py:135 (+5) | CODE: imaggers/image/t18_run.py:317 + webui/chunked.py:963 -- registers the big-M image set |
| `ZIMAGE_BIGM_PROJ` | '0' |  | webui/capability.py:135, webui/capability.py:380 (+2) | CODE: webui/chunked.py:964 -- whole-shape big-M projection |
| `ZIMAGE_CHUNK_PIPE` | '1' |  | webui/capability.py:135, webui/capability.py:380 (+2) | CODE: webui/chunked.py:942-958 -- 96-row pipeline instead of block_step |
| `ZIMAGE_DEV_WORKER` | '0' |  | webui/config.py:352 | CODE: webui/config.py:220 -- resident py310 worker vs one process per job |
| `ZIMAGE_DEV_WORKER_KEEP_PAYLOAD` | '1' |  | webui/runner.py:59 | CODE: webui/runner.py:54 |
| `ZIMAGE_DEV_WORKER_KEEP_REFINER` | '0' |  | webui/runner.py:60 | CODE: webui/runner.py:55 |
| `ZIMAGE_FFN_IMAGE` | 'fs_ffn_j5' |  | webui/config.py:37, webui/config.py:238 | MEASURED IDENTICAL: webui/config.py:32-36 -- fs_ffn_j5 vs t11_nh5 latents 9f86aee9, 960/0 dispatches |
| `ZIMAGE_NO_JOB` | '0' |  | webui/reap.py:75 | CODE: webui/reap.py:75 -- disables the job-object reaper (child lifetime, not arithmetic) |
| `ZIMAGE_TE_MODEL_RESIDENT` | 'auto' |  | webui/runner.py:71 | CODE: webui/runner.py:66 |
| `ZIMAGE_TE_RESIDENT` | '0' |  | webui/textenc.py:63 | CODE: webui/textenc.py:63 -- resident TE model |
| `ZIMAGE_VAE_FAST` | '1' |  | webui/capability.py:135, webui/capability.py:421 (+4) | CODE: webui/vae_local.py:56 -- vae_lean loader |
| `ZIMAGE_VAE_LAUNCH` | 'srv' | yes | imaggers/vae/t21_3b3_chain.py:1667, imaggers/vae/t21_3b3_chain.py:350 (+1) | CODE: webui/config.py:73 -- srv vs proc |
| `ZIMAGE_VAE_NPU` | '0' |  | webui/capability.py:135, webui/capability.py:421 (+4) | CODE: webui/config.py:52-60 -- NPU chain vs CPU decoder |
| `ZIMAGE_VAE_OPNPU` | '0' |  | imaggers/vae/t21_3b3_chain.py:1669, imaggers/vae/t21_3b3_chain.py:626 (+1) | CODE: webui/config.py:82 -- residual adds on the NPU |
| `ZIMAGE_WPACK_PREWARM` | 'job' |  | webui/config.py:371 | CODE: webui/config.py:224-243 -- prewarm the 1.85 GiB pack off the critical path |

### timing -- wall clock only

| knob | default | conflict | readers | evidence |
|---|---|---|---|---|
| `STEP3_TIMEOUT_MS` | '300000' |  | imaggers/block/step4_run_ffn.py:654 | AUTO: name-family heuristic -- not verified against the code |
| `T21_3B2_WAIT_MS` | '600000' |  | imaggers/vae/t21_3b2_plan.py:119 | CODE: imaggers/vae/t21_3b2_plan.py:119 |
| `T22_ADAPT_CHUNK` | '1632' |  | imaggers/image/t18_run.py:397, webui/shapes.py:567 | CODE: imaggers/image/t18_run.py:397 -- chunk size for the host attention |
| `T22_BREPACK_PIECE` | '' |  | imaggers/block/bigm/brepack_cache.py:144 | CODE: imaggers/block/bigm/brepack_cache.py:144 -- I/O piece size for the cache file |
| `T22_BREPACK_THREADS` | 0 |  | imaggers/block/bigm/brepack_cache.py:137 | CODE: imaggers/block/bigm/brepack_cache.py:137 -- thread count for the repack; the permutation it builds does not depend on the count (not measured) |
| `T22_BREPACK_WRITE_SYNC` | '0' |  | imaggers/block/bigm/brepack_cache.py:904 | CODE: imaggers/block/bigm/brepack_cache.py:904 |
| `T22_DEVKILL_PREWARM_LAYERS` | '30' |  | imaggers/devkill/devkill_prewarm.py:24 | CODE: imaggers/devkill/devkill_prewarm.py:24 |
| `T22_FILL_DELAY_MS` | '0' |  | imaggers/image/t22_run.py:227, imaggers/image/t22_run.py:239 (+1) | CODE: imaggers/image/t22_run.py:227 -- time.sleep after the BO fill |
| `T3_TIMEOUT_MS` | '300000' |  | imaggers/image/t13_run.py:225, imaggers/image/t18_run.py:731 (+2) | CODE: imaggers/image/t18_run.py:731 -- device wait cap |
| `ZIMAGE_DEVICE_TIMEOUT_S` | '' |  | webui/runner.py:1382 | CODE: webui/runner.py:1272 |
| `ZIMAGE_VAE_THREADS` | '' |  | webui/capability.py:135, webui/capability.py:421 (+3) | CODE: webui/config.py:192-207 -- the CPU decode pool |
| `ZIMAGE_VAE_THREAD_RULE` | '' |  | webui/capability.py:135, webui/capability.py:421 (+3) | CODE: webui/config.py -- whether the VAE decode follows the loop thread pool |

### diag -- logging/dumping/probing only

| knob | default | conflict | readers | evidence |
|---|---|---|---|---|
| `S2_DUMP_STAGES` | '0' |  | imaggers/util/place/s2_dev.py:699 | CODE: imaggers/util/place/s2_dev.py:699 |
| `S2_FFN32_CHECK` | '0' |  | imaggers/util/place/s2_dev.py:429 | CODE: imaggers/util/place/s2_dev.py:429 |
| `S2_STAGE` | 'write' |  | imaggers/util/place/s2_dev.py:346 | CODE: imaggers/util/place/s2_dev.py:346 |
| `T18_DUMP` | '' |  | imaggers/image/t18_run.py:1752, imaggers/image/t18_run.py:2232 (+2) | AUTO: name-family heuristic -- not verified against the code |
| `T18_TRACE` | '' |  | imaggers/image/t18_run.py:744, imaggers/image/t22_t18_base.py:378 | AUTO: name-family heuristic -- not verified against the code |
| `T22_ATTN_DUMP` | '' |  | imaggers/block/bigm/prod_loop_bigm.py:863, webui/chunked.py:683 | CODE: webui/chunked.py:476 |
| `T22_ATTN_DUMP_N` | '1' |  | webui/chunked.py:684 | CODE: webui/chunked.py:477 |
| `T22_ATTN_PROF` | '0' |  | webui/chunked.py:464 | CODE: webui/chunked.py:266 |
| `T22_BIGM_DEBUG` | '0' |  | imaggers/block/bigm/prod_loop_bigm.py:785, imaggers/block/bigm/prod_loop_bigm.py:878 (+4) | CODE: imaggers/block/bigm/prod_loop_bigm.py:777 -- timing lines |
| `T22_DBG` | '' |  | imaggers/image/t22_run.py:524, imaggers/image/t22_run.py:597 | AUTO: name-family heuristic -- not verified against the code |
| `T22_DBG2` | '' |  | imaggers/image/t22_run.py:473 | AUTO: name-family heuristic -- not verified against the code |
| `T22_DUMP_A` | '' |  | imaggers/image/t22_run.py:637 | AUTO: name-family heuristic -- not verified against the code |
| `T22_DUMP_BG` | '' |  | imaggers/image/t22_run.py:283 | AUTO: name-family heuristic -- not verified against the code |
| `T22_DUMP_FFN` | '' |  | imaggers/image/t18_run.py:2059, imaggers/image/t22_run.py:727 | CODE: imaggers/image/t18_run.py:2059 |
| `T22_DUMP_FFNBLOCK` | '' |  | imaggers/image/t22_run.py:293 | AUTO: name-family heuristic -- not verified against the code |
| `T22_DUMP_HOP` | '' |  | imaggers/block/bigm/prod_loop_bigm.py:294 | CODE: imaggers/block/bigm/prod_loop_bigm.py:294 |
| `T22_DUMP_QKV` | '' |  | imaggers/image/t22_run.py:455 | AUTO: name-family heuristic -- not verified against the code |
| `T22_FFN_KK32_PROF` | '0' |  | imaggers/util/place/kk_prod.py:413, imaggers/util/place/kk_prod.py:163 | CODE: imaggers/util/place/kk_prod.py:413 |
| `T22_FFN_PROF` | '0' |  | imaggers/util/glue/install.py:844 | CODE: imaggers/util/glue/install.py:810 |
| `T22_GLUE_PROF` | '0' |  | imaggers/image/t18_run.py:201 | CODE: imaggers/image/t18_run.py:201 |
| `T22_PLAN_FFN_DEBUG` | '0' |  | imaggers/util/place/plan_ffn.py:91 | CODE: imaggers/util/place/plan_ffn.py:91 |
| `T22_PLAN_FFN_LOG` | '1' |  | imaggers/util/place/plan_ffn.py:380 | CODE: imaggers/util/place/plan_ffn.py:380 |
| `T22_PLAN_FFN_PROBE_DUMP` | '' |  | imaggers/devkill/devkill_ffn.py:443, imaggers/image/t18_run.py:2102 | AUTO: name-family heuristic -- not verified against the code |
| `T22_REF_PROF` | '0' |  | imaggers/image/t3_real.py:650 | CODE: imaggers/image/t3_real.py:609 |
| `T22_S2_PROF` | '0' |  | imaggers/util/place/s2_product.py:177 | CODE: imaggers/util/place/s2_product.py:177 |
| `T22_S2_TAGS` | '' |  | imaggers/util/place/s2_dev.py:123 | CODE: imaggers/util/place/s2_dev.py:123 |
| `T22_S2_XCHECK` | '0' |  | imaggers/util/place/s2_product.py:98 | CODE: imaggers/util/place/s2_product.py:98 |
| `T22_SBG_ASSERT` | 'step' |  | imaggers/image/t18_run.py:237 | CODE: imaggers/image/t18_run.py:237 -- assert mode string ('step') |
| `T22_TIMELINE` | '0' |  | imaggers/util/glue/install.py:746 | CODE: imaggers/util/glue/install.py:712 |
| `T22_TIMELINE_FILE` | 'devtimeline.tsv' |  | imaggers/util/glue/install.py:747 | CODE: imaggers/util/glue/install.py:713 |
| `ZIMAGE_VAE_ONESHOT` | '' |  | webui/runner.py:1634 | CODE: webui/runner.py:1524 |
| `ZIMAGE_VAE_TRACE` | '0' |  | imaggers/vae/t21_3b3_chain.py:883 | CODE: imaggers/vae/t21_3b3_chain.py:883 |

### infra -- paths, caches, memory, interpreters, addresses

| knob | default | conflict | readers | evidence |
|---|---|---|---|---|
| `NH_BG` | '5' |  | imaggers/block/bigm/brepack_cache.py:1043 | CODE: imaggers/block/bigm/brepack_cache.py:1043 |
| `NPU_CACHE_DIR` | '' |  | imaggers/vae/t21_3b3_chain.py:1424, imaggers/vae/t21_3b3_chain.py:1478 (+1) | AUTO: name-family heuristic -- not verified against the code |
| `NPU_CACHE_HOME` | '' |  | imaggers/block/step3_ffn.py:122, imaggers/block/step4_run_ffn.py:60 (+5) | AUTO: name-family heuristic -- not verified against the code |
| `NPU_PY312` | 'D:\\Project\\deepseek_harness\\test7\\.winml_venv\\Scripts\\python.exe' |  | imaggers/util/place/npu_synth.py:48 | AUTO: name-family heuristic -- not verified against the code |
| `PEANO_INSTALL_DIR` | '' |  | imaggers/block/step3_ffn.py:121, imaggers/block/step4_run_ffn.py:59 (+4) | AUTO: name-family heuristic -- not verified against the code |
| `S2_ENGINE` | 'inline' |  | imaggers/util/place/s2_dev.py:342 | CODE: imaggers/util/place/s2_dev.py:342 |
| `T22_BREPACK_DIR` | '' |  | imaggers/block/bigm/brepack_cache.py:128 | CODE: imaggers/block/bigm/brepack_cache.py:128 -- where the B cache lives |
| `T22_BREPACK_VERIFY` | '1' |  | imaggers/block/bigm/brepack_cache.py:133 | CODE: imaggers/block/bigm/brepack_cache.py:133 -- sha256 + word multiset check |
| `T22_P10_BIN` | '' |  | imaggers/image/t18_run.py:477 | AUTO: name-family heuristic -- not verified against the code |
| `T22_P10_XCLBIN` | '' |  | imaggers/image/t18_run.py:476 | AUTO: name-family heuristic -- not verified against the code |
| `T22_QKV_BIN` | '' |  | imaggers/image/t18_run.py:273 | AUTO: name-family heuristic -- not verified against the code |
| `T22_QKV_XCLBIN` | '' |  | imaggers/image/t18_run.py:272 | AUTO: name-family heuristic -- not verified against the code |
| `T22_S2_CACHE_DIR` | '' |  | imaggers/util/place/s2_dev.py:45 | CODE: imaggers/util/place/s2_dev.py:45 |
| `T22_S2_CACHE_LAYERS` | '4' |  | imaggers/util/place/s2_dev.py:168 | CODE: imaggers/util/place/s2_dev.py:168 |
| `ZIMAGE_BUDGET_ENFORCE` | False |  | webui/runner.py:305 | CODE: webui/runner.py:300 -- admission gate: an EXPLICIT memory_mode that does not fit the measured budget is REFUSED instead of being degraded onto --no-cache- |
| `ZIMAGE_CHUNK_REFINER_GB` | '0' |  | webui/capability.py:159, webui/shapes.py:727 (+1) | CODE: webui/shapes.py:726 -- refiner memory budget |
| `ZIMAGE_DEV_WORKER_RESERVE_GB` | '7.0' |  | webui/runner.py:50 | CODE: webui/runner.py:45 -- memory reserve before a worker launch |
| `ZIMAGE_TE_CACHE_DIR` | '' |  | webui/cache.py:650 | CODE: webui/cache.py:650 |
| `ZIMAGE_TE_CACHE_ENTRIES` | '64' |  | webui/cache.py:651 | CODE: webui/cache.py:651 |
| `ZIMAGE_TE_CACHE_MB` | '256' |  | webui/cache.py:653 | CODE: webui/cache.py:653 |
| `ZIMAGE_TE_CACHE_MEM_ENTRIES` | '8' |  | webui/cache.py:655 | CODE: webui/cache.py:655 |
| `ZIMAGE_TE_GGUF` | '' |  | webui/config.py:614 | CODE: webui/config.py:482 -- the text-encoder weights |
| `ZIMAGE_TE_MODEL_GB` | '7.0' |  | webui/runner.py:70 | CODE: webui/runner.py:65 |
| `ZIMAGE_VAE_SRV_ADDR` | '' |  | imaggers/vae/t22_vae_srv.py:100, webui/runner.py:1673 (+2) | AUTO: name-family heuristic -- not verified against the code |
| `ZIMAGE_WPACK_VERIFY` | '1' |  | webui/config.py:388 | CODE: webui/config.py:253-257 -- hash the wpack instead of trusting the manifest |

## Defaults the readers disagree about

A resolved value is only as good as the default it was resolved FROM.  These
knobs have more than one literal default in the runtime closure, so an unset
value means different things to different readers:

| knob | defaults seen | where |
|---|---|---|
| `T22_FAST_GLUE` | '0' vs '1' vs True | imaggers/image/t18_run.py, webui/bridge.py, webui/config.py |
| `T22_GLUE2` | '0' vs '1' vs True | imaggers/image/t18_run.py, webui/bridge.py, webui/config.py |
| `T22_P10` | '0' vs '1' | imaggers/image/t18_run.py, webui/bridge.py, webui/config.py |
| `T22_QKV` | '0' vs '1' | imaggers/image/t18_run.py, webui/bridge.py, webui/config.py |
| `ZIMAGE_VAE_LAUNCH` | 'proc' vs 'srv' | imaggers/vae/t21_3b3_chain.py, webui/config.py |

## The product writes these itself

They are in os.environ of every child because the product put them there, so a
naive record calls them inherited.  provenance.py marks them instead.

| knob | written by |
|---|---|
| `ZIMAGE_VAE_LAUNCH` | webui/config.py:84 os.environ.setdefault |
| `ZIMAGE_VAE_OPNPU` | webui/config.py:85 os.environ.setdefault |
| `ZIMAGE_VAE_SRV_ADDR` | webui/runner.py:1583 os.environ[...] per job |

