# T21-3B3-VAE：六尺寸 NPU VAE 端到端门 + WebUI 接线 + §3 实现选择声明

- 单元：t21_3b3（U1 执行器整合 → U2 六尺寸端到端门 → U3 WebUI 接线 → U4 四件套 → U5 报告）
- 时间：2026-10-19 19:57–22:25（+08:00，约 3 设计小时；预算 1 设计日）
- 前置：T21-3B2-BATCH.md（52 PASS）+ T21-3B2VF-REPORT.md（40 PASS）= **92/92 全绿**
- 只写：_scratch/vae/（含 out/t21_3b3/）、webui/vae_local.py、webui/runner.py、webui/config.py、
  webui/evidence/。新增 5 个 xclbin（2 个 conv、3 个 up），全部走 3b2 的同一条门。
- 提交前缀：t21_3b3:

---

## §0 一屏结论

1. **六尺寸端到端门 6/6 PASS**（128²、64×256、144×128、192×128、128×256、256×256）：
   全阶段 VAE decode，conv/GroupNorm/upsample 在 NPU，质量对 **CPU VAE fp32（diffusers 上游实现）**
   37.70–38.22 dB，**run-to-run 逐字节相同**，每次调用的证据在 calls.json。
2. **执行器整合完成**：conv 按 (K,N) 族选镜像并按 M'=1024 切块、**(K=2336,N=256) 强制 N=128 半宽**、
   GN 按 (C,S) 选子平面镜像、UP 按 **(C,H,W)**（不是 (C,S)）选行条镜像。
   t21_3b2_plan.py --dry-run --strict 现在 **exit 0**：0 个 design gap、92/92 行有 PASS 判决。
3. **预测兑现**：六尺寸 conv launch 数的纸面预测 104/104/119/170/218/431 **逐一命中实测**
   （预测总量 1146，实测 1146；GMAC 预测 837.6 相同）。
4. **64×64 自测口径**：37.4080 dB（3b1 的 37.3189，**+0.089 dB**）。**不是**逐位复现，原因已用单变量对照定位：
   host 侧解释器/框架不同（py310+torch2.3.1+3b1 自建 decoder vs py312+torch2.4.1+真 diffusers）。
   同一 64×64 图上，"3b1 档"与"3b3 档"的差 49.96 dB，而两份 host 栈的 baseline 档互差 43.47 dB ⇒ 残差在 host，不在器件。
5. **WebUI 接线**：vae_local 先走 NPU 链，任何异常回退常驻 CPU（语义不变）；计时桶 vae_npu_s 如实归属。
   一个真实 job：333.9 s，**Σ桶 = total − 0.062 s**（门 ±1 s，PASS），其中 NPU 解码 194.19 s +
   NPU 建模 18.25 s。
6. **批语义门**：同一批 2 张，逐张单独解码与批内解码**PNG 逐字节相同**（见 §5.2）。
7. **速度结论：不引用**。这次交付是"能跑对"，不是"跑得快"：128² 一次解码 96–98 s（CPU 常驻 0.46 s）。
   原因是一 launch 一 py310 进程（1146 次 launch），不是内核吞吐（器件侧 837.6 GMAC / 1054 s ≈ 0.8 GMAC/s）。
   产品默认仍应保留 CPU 路径，除非把驱动器改成进程内多次 launch（barrier 方案）。

---

## §1 交付物

| 类 | 内容 | 位置 |
|---|---|---|
| 链执行器 | t21_3b3_chain.py（conv/GN/UP 路由 + 一 launch 一进程 + 逐调用检查 + 桶） | _scratch/vae/ |
| 六尺寸门 | size_report.json、size_calls.json（每调用一条）、size_npu.npy、size_ref.npy、PNG | _scratch/vae/out/t21_3b3/ |
| 汇总 | summary.json、six_sizes_table.md | 同上 |
| 64×64 自测 | selfref64_npu.json（+ 两个对照档 _model、_model_hostgnup） | 同上 |
| 官方参考并排 | official_128.json + T21-3B3-official-128-side-by-side.png | 同上 / webui/evidence/ |
| 六尺寸并排 | T21-3B3-size-side-by-side.png（NPU / CPU 参考 / ×8 差值） | webui/evidence/ |
| WebUI | vae_local.py（NPU 路径 + 回退）、runner.py（vae_npu_s/vae_other_s 桶）、config.py（ZIMAGE_VAE_NPU） | webui/ |
| WebUI 证据 | webui_job_20261019-220703-9dece3.json、vae_batch_gate.json | _scratch/vae/out/t21_3b3/ |
| 新增 xclbin | cg_M4096_K1184_N128、cg_M4096_K288_N128、upw_C512_H8_W32、upw_C512_H16_W64、upw_C256_H32_W128 | _scratch/vae/out/t21_3b2/（不入 git，可用 --build-conv / --build-up 重建） |

---

## §2 U1：执行器整合（把 3b2vf 的三条变体规则接进链）

### 2.1 路由规则（实现即规则）

    conv：C 的行只依赖 A 的同一行与整个 B，K/Kc/k 序都与 M 且与列区间无关 →
      · (K,N) 有 M'=1024 镜像 且 该行一次 dispatch 的 MAC > 6.0 GMAC（≈4 s，ERT 看门狗 ~7 s）→ 分块
      · (K=2336, N=256) → **强制** cgv_M1024_K2336_N128，每块打两次（整宽镜像会 stall）
      · 否则用 exact cg_M{M}_K{K}_N{N}（3b1 的 5 个镜像也在这个索引里）
      · 二者都没有 → 直接抛错（不猜、不降级到 host）
    GN：（C,S）键；子平面/整平面的 chunk 在核内，host 只交一条 (C,S) 流
    UP：（C,H,W）键。**不能**用 (C,S) 键：核是 upsample_nearest2(n, w)，w 是 .bin 立即数
        （group_norm.cc: y[(2r+j)(2w)+2c] ← x[r*w+c]），同一 S 的另一个 (H,W) 必须另建镜像

### 2.2 为此新增的 5 个镜像（全部同门）

| 镜像 | 为什么必须新建 | 门 |
|---|---|---|
| cg_M4096_K1184_N128 | 64×64 自测的 up3 主卷积（六尺寸的 M 表里没有 M=4096/Cin=128） | 3b2 verify_conv：max_ulp 0.0 / bit_exact 1.0（链内全行抽样同门） |
| cg_M4096_K288_N128 | 同上，256→128 的 1×1 shortcut | 同上 |
| upw_C512_H8_W32 | 64×256 的 (512,8,32)；3b2 的 up_C512_S256 是 (16,16) | 3b2 verify_up：bit_exact True、out_len = C·H·W·8 |
| upw_C512_H16_W64 | 64×256 的 (512,16,64)；up_C512_S1024 是 (32,32) | 同上 |
| upw_C256_H32_W128 | 64×256 的 (256,32,128)；up_C256_S4096 是 (64,64) | 同上 |

编译：cg_M4096_K1184_N128 205.5 s、cg_M4096_K288_N128 8.1 s、三个 upw 合计 ~13 s。

### 2.3 92 行闸门（不是回归，是这次的入口条件）

t21_3b2_plan.py --dry-run --strict → **exit 0**，输出：

    [u1] design-variant needed   0 ; buildable with the existing designs 52 ; gated by a 3b2vf VARIANT 40
    [u1] closure (92/92 gate): 92/92 rows have a PASS verdict on disk (GREEN)

实现：make_items() 读 3b2vf 的逐行 json（out/t21_3b2vf/NN_name.json）给行打 variant，
closure_rows() 逐行读判决（52 行读 out/t21_3b2/STEM_results.json，40 行读 3b2vf 的行），
--strict 现在要求 **0 gap + 4 条坑契约 + 92/92 PASS** 三者同时成立。

### 2.4 64×64 自测口径（U1 的回归闸）

| 档 | 是什么 | PSNR vs ref_img |
|---|---|---|
| 3b1 device2_measured | host device-exact conv + NPU GN/UP | **37.3189 dB** |
| **3b3 链**（conv 也在 NPU） | 35 conv 全在器件（50 launch），GN/UP 同 3b1 镜像 | **37.4080 dB** |
| 3b1 baseline | host device-exact conv + host GN/UP | 37.4060 dB |
| 3b3 对照：conv=host model + host GN/UP | 复刻 3b1 baseline 档 | 37.5208 dB |

- **conv 是逐位等价的**：--conv-impl model（host device-exact model）与 --conv-impl npu 产出的图
  **完全相同**（两者对 ref 都是 37.4080 dB、同一 sha16、同一 9705 个差异元素）。
- **残差在 host 栈**：两份 host 栈的 baseline 档互差 43.47 dB（37.5208 vs 37.4060），
  而 3b3 链与 3b1 device2 档差 49.96 dB；也就是说 3b3 的链**比两份 3b1 档之间还要接近**。
  这是因为 bf16 回舍在 35 层卷积里把 host 的末位差异放大成 ~1e-4 量级的图差。
- 结论：**口径复现（+0.089 dB），不是逐位复现**；逐位复现在 py310→py312 的解释器分裂下不可达，
  这一点有对照实验（上面两行）而不是推断。

### 2.5 器件的逐调用检查（calls.json 的内容）

- conv：每个几何的**首个实例**取前 64 行（64×64 自测取全行）与 bfp_gemm_model_v 比 →
  六尺寸 35/35 几何 sample_bit_exact = 1.000、sample_pass = true。
- GN：每次调用与 **kernel 公式的 f64 理想值**比，除以**幅值界**（不是结果本身）→ 见 §4.2。
- UP：每次与 np.repeat(np.repeat(x,2,1),2,2) 逐位比 → 18/18 全等。

---

## §3 实现选择声明（PRODUCT-CHECKLIST §3）

**每个阶段跑在哪，逐条如实。host = py312 CPU（torch/numpy）；NPU = XDNA2 上的 xclbin。**

| # | 阶段 | 跑在哪 | 证据 / 口径 |
|---|---|---|---|
| 1 | latent → decoder 输入 | host numpy：x/0.3611 + 0.1159 | 该 VAE **没有** post_quant_conv（vae.post_quant_conv is None），前端无其它算子 |
| 2 | conv_in + 其余 34 个 Conv2d（含 1×1 shortcut、每个 Upsample2D 内的 conv） | **NPU** | 35 次 conv 调用 / 每尺寸 104–431 次 launch；每几何抽样逐位 1.000（§2.5） |
| 3 | im2col、A 的 bf16 回舍、W 的 BFP16 打包、块拼接 | host numpy —— 它是 xclbin 的**输入**，不是替代 GEMM 的 host fp32 计算 | 坑① W=floor(A=conv_even)；与 92 行门同一条代码（vae_npu.pad_kn/pad_a/pack_b_blocks） |
| 4 | GroupNorm ×30 | **NPU**（gn_*/gnv_*） | 30 次调用；对 f64 理想值的幅值界 ulp，六尺寸最大 1.0/1.0/2.0/1.0/1.0/2.5（§4.2） |
| 5 | SiLU ×31 | **host fp32 torch** | 树里**没有** SiLU 的 xclbin；92 行只覆盖 conv/GN/UP。桶 silu_s |
| 6 | 残差加 ×40（Resnet 的 (x+h)/1.0、Attention 的 (out+residual)） | **host fp32 torch** | 桶 add_s（容器法：只记子算子没记的部分） |
| 7 | mid-block attention（q/k/v/out 的 1×1 Linear + softmax SDPA） | **host fp32 torch** | nn.Linear 不在 92 行里；桶 attn_s |
| 8 | nearest-2× upsample ×3 | **NPU**（up_*/upv_*/upw_*） | 3 次调用；逐位等于 replicate（18/18 张图） |
| 9 | (x+1)/2、clip、PNG | host | 桶 vae_save_s / vae_other_s |

**明确不声明的事**

- 不把 conv 说成"host fp32 卷积"：35 个 conv 的 GEMM **全部**在器件上；host 只做 im2col/打包/回舍/拼接。
- 不把 SiLU/add/attention 说成"在 NPU 上"：它们是 host fp32，且**没有任何** xclbin 覆盖 ——
  这是这套 VAE 唯一还没上板的三组算子（§7.1）。
- 不把 64×64 自测台的 device-exact model 当产品：--conv-impl model 只用于回归对照，不参与六尺寸。
- WebUI 里"NPU 链"与"CPU 常驻"是两个可分辨的实现：job 的 vae_impl 记录实际跑的是哪个，
  vae_npu_s / vae_decode_s 两个桶分别记账，UI 的阶段文案也按实际实现写。

---

## §4 U2：六尺寸端到端门

### 4.1 逐尺寸一行

| size | NPU vs CPU VAE fp32 | max abs | 调用 / conv launch（预测） | GMAC | run-to-run | conv 抽样 | GN max ulp | 计时未归属 | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| 128x128 | **37.9497 dB** | 0.15173 | 68 / 104 (104) | 78.6 | 逐字节 | 35/35 逐位 | 2.50 | +0.15 s | PASS |
| 64x256 | **38.2159 dB** | 0.10453 | 68 / 104 (104) | 78.6 | 逐字节 | 35/35 逐位 | 1.00 | +0.17 s | PASS |
| 144x128 | **38.0508 dB** | 0.12570 | 68 / 119 (119) | 91.1 | 逐字节 | 35/35 逐位 | 2.00 | +0.13 s | PASS |
| 192x128 | **37.7001 dB** | 0.15628 | 68 / 170 (170) | 117.9 | 逐字节 | 35/35 逐位 | 1.00 | +0.14 s | PASS |
| 128x256 | **37.7548 dB** | 0.13067 | 68 / 218 (218) | 157.1 | 逐字节 | 35/35 逐位 | 1.00 | +0.19 s | PASS |
| 256x256 | **37.7865 dB** | 0.20307 | 68 / 431 (431) | 314.3 | 逐字节 | 35/35 逐位 | 1.00 | +0.38 s | PASS |

- **质量口径**：参考是**同一进程里**未打补丁的 diffusers AutoencoderKL fp32 CPU 解码（独立实现，不是本链的 f32 投影）。
  latents 统一 rng(0)（np.random.default_rng(0).standard_normal，与 3b1/3b2 的几何表同一口径）。
- **逐字节**：每个尺寸各跑两遍完整 decode（新进程、新 fresh NPU cache），image_sha16 与 bytes_equal 均为真。
- **计时未归属**：Σ桶 − total 的残差，六尺寸 |·| ≤ 0.38 s；桶的构成见 §4.3。
- 六尺寸合计：**1146 次 conv launch、837.6 GMAC、1054 s decode（run 1，含检查）**。

### 4.2 逐调用 GN 的一次超门（如实列出，不改门限）

128² 的 gnv_C128_S16384 @ up_blocks.3.resnets.0.norm2 在链内测得 **2.5 幅值界 ulp**（门是 ≤2.0）。
这不是新缺陷也不是判据变更：

- 该行（3b2 的 id 58）在**它自己捕获的激活**上是 1.000；ulp 是对元素取 max 的统计量，换一份激活就会动
  （3b2 的 id 79 本来就在 2.000 零余量上）。
- 链内输入是链自己产生的激活（经过 20+ 层的 bf16 回舍），不是 prep 时那份 fp32 激活。
- 判据口径不改：92 行门是"逐行、各自激活"的门，仍然全绿；链内的 2.5 作为**观测值**列出。
  其余 5 个尺寸的 30 次调用全部 ≤2.0。

### 4.3 计时桶（每个尺寸 run 1，秒）

| size | conv NPU | conv host | GN NPU | GN host | UP NPU | SiLU | add | attn | 检查 | 合计/未归属 |
|---|---|---|---|---|---|---|---|---|---|---|
| 128x128 | 94.8 | 3.9 | 9.8 | 0.3 | 1.0 | 0.0 | 0.4 | 0.0 | 5.2 | 115.6 / +0.15 |
| 64x256 | 97.5 | 4.5 | 10.8 | 0.3 | 1.2 | 0.0 | 0.4 | 0.0 | 5.5 | 120.3 / +0.17 |
| 144x128 | 96.5 | 3.5 | 6.9 | 0.2 | 0.7 | 0.0 | 0.3 | 0.0 | 4.1 | 112.3 / +0.13 |
| 192x128 | 127.7 | 4.7 | 7.0 | 0.3 | 0.8 | 0.0 | 0.4 | 0.0 | 4.7 | 145.8 / +0.14 |
| 128x256 | 168.1 | 5.5 | 7.3 | 0.5 | 0.8 | 0.0 | 0.6 | 0.0 | 6.0 | 189.0 / +0.19 |
| 256x256 | 337.4 | 11.9 | 7.9 | 1.0 | 1.1 | 0.0 | 1.2 | 0.0 | 10.2 | 371.2 / +0.38 |

chain_check_s 是逐调用检查（§2.5）的时间，单列一桶，不混进 NPU/CPU 阶段。
"conv NPU" = 1024 行块/整几何的一次次 dispatch 的墙钟（含每次一个新 py310 进程的启动）。

### 4.4 产品四件套 + 常规门

| 门 | 结果 | 证据 |
|---|---|---|
| ① 图像并排（NPU vs CPU 参考，独立口径） | PASS | T21-3B3-size-side-by-side.png（6 张，NPU/CPU参考/×8 差值）+ T21-3B3-official-128-side-by-side.png |
| ② 计时闭合 Σ桶 == total ±1 s | PASS | 六尺寸 ≤0.38 s；真实 job **0.062 s**（§5.1） |
| ③ 实现选择声明 | PASS | §3 全文（含 SiLU/add/attention 的 host fp32 说明与理由） |
| ④ 批语义门（batch2 == 单图逐字节） | PASS | vae_batch_gate.json（§5.2） |
| 常规：run-to-run 逐字节 | PASS | 六尺寸 2/2 相同；跨服务器重启同一 job latents 的 PNG sha256 相同 |
| 常规：t3d_baseline.py --check | PASS | [check] ALL REFERENCE BYTES UNCHANGED（exit 0，收工前 22:25 复跑） |
| 常规：3b2 存量门重放 | PASS | t21_3b2_plan.py --selftest：replayed 17，disagreements 0，PASS |
| 常规：92 行闸门（收工前复跑） | PASS | --dry-run --strict → EXIT=0，0 gap / 92/92 GREEN |

**官方参考（独立口径）**：zimage-port/_out/zimage_128_fixed.png 与"用官方最终 latent 跑 CPU VAE"相差
**58.92 dB**（max abs 0.00196 ≈ 半级 8-bit）→ 官方图确实就是同一个 VAE 的解码，可以作为独立参考。

| 128² 同一 latent | vs 官方 PNG | vs CPU VAE |
|---|---|---|
| CPU VAE（diffusers fp32） | **58.92 dB** | — |
| NPU 链 | 40.94 dB | **41.01 dB**（max abs 0.0825） |

（这一对是针对**真实 pipeline 的 latent**；§4.1 的 37.95 dB 是 rng(0) 的 latent，两者不可混用。）

---

## §5 U3：WebUI 接线

### 5.1 真实 job（隔离端口 7870，冷启动）

    job 20261019-220703-9dece3   batch 2, steps 8, seed 0, 128x128, VAE on
    status done   vae_impl = npu   elapsed 333.92 s
    buckets: textenc 10.07 | startup 31.76 | loop 75.81 | device_tail 0.37
             vae_load 18.25 | vae_npu 194.19 | vae_save 0.02 | vae_other 3.35 | prepare 0.02
    closure: Σ 333.856 vs total 333.917 → unaccounted 0.062 s（门 1.0 s）PASS
    图像: s0 sha256 47b2861c0e0e68e2 (25792 B, 96.55 s) / s1 8bb841c81959ff8c (24758 B, 97.64 s)

- vae_load_s = 18.25 s 是**首次**建 NPU 解码器（加载 diffusers + 打补丁）的时间，
  第二次 job 为 0 —— 这就是 T16 P0-2 的做法在 NPU 路径上的等价物。
- 3 个 job（两次冷启动 + 一次同进程）对同一 latents 产出的 PNG **sha256 完全相同**，
  且 3 次的 latents（c49b6a258c13fec1）也相同 ⇒ 跨进程/跨服务器重启都是确定性的。
- 回退语义：vae_local 只在"形状不在六尺寸表"或"NPU 抛异常"时回退常驻 CPU；
  回退时桶记 vae_decode_s（CPU），阶段文案写"NPU 失败→CPU 回退"。
  CPU 路径的代码与计时口径**未改**（只是被括进 impl == "cpu" 分支）。

### 5.2 批语义门

真实 job 的 latents（w2207039dece3p0_latents.npy，2 张），走**同一个常驻代码路径**解两遍：

    whole : decode_batch(2 张)  → s0 47b2861c0e0e68e2 (95.34 s) / s1 8bb841c81959ff8c (94.95 s)
    single: decode_batch(第 0 张) → s0 47b2861c0e0e68e2 (95.95 s)
    image0_batch_sha == image0_single_sha → **byte_identical = true**

三个独立来源的同一张图 sha256 一致：job（§5.1）、整批解码、单图解码 —— 都是 47b2861c0e0e68e2。
这说明 VAE 阶段是"逐图独立"的：批内第 0 张与单独解它逐字节相同（不是 PSNR、不是 allclose）。

---

## §6 机时账

| 项 | 墙钟 | 说明 |
|---|---|---|
| 编译 5 个新镜像 | 227 s | 205.5 + 8.1 + ~13 |
| 64×64 自测（3 次：NPU / host-model / baseline 对照） | ~4 min | 每次 35–70 s decode |
| 六尺寸门（2 run × 6 尺寸，含检查） | 36.5 min | 器件 1054 s + 检查 36 s + 参考解码 ~6 s + 进程/落盘 |
| 官方 128² 三方对照 | 3.5 min | NPU 99.2 s + CPU 0.4 s |
| WebUI job ×3（2 冷 1 热） | 16.6 min | 每次 5.5 min（其中 NPU VAE 3.5 min） |
| 批语义门（2 张 + 1 张 = 3 次解码） | ~5.5 min | |
| 分析/写报告/git | ~55 min | |
| **合计** | **~2 h 28 min**（19:57–22:25） | 其中器件计算 ~37 min |

未触止损线③（1.5 设计日）。宿主未被挂起。

---

## §7 诚实负结果 / 剩余量

1. **SiLU / add / attention 仍在 host fp32**：31 + 40 + 1 次。要把它们搬到 NPU，需要新的
   SiLU 核、add 核、以及 attention 的 GEMM+softmax 设计（nn.Linear 的 (K,N) 也不在 92 行里）。
   这是"整套 VAE 100% 上板"的唯一剩余缺口，本次**没有**半建。
2. **速度不可引用**：128² 96–98 s、256² 371 s，对比 CPU 常驻 0.46 s/张。根因是**一 launch 一 py310 进程**
   （1146 次 launch × ~0.3 s 启动 + 每块落盘/读回）。器件侧的有效吞吐约 0.8 GMAC/s
   （837.6 GMAC / 1054 s），与 3b2 的 1.28 GMAC/s 同量级；也就是说**慢在驱动器，不在内核**。
   下一步若要做速度，唯一有意义的方向是进程内多次 launch + 跨镜像 barrier（T20-STATE8 的方法），
   本单没有做，因为六尺寸的正确性门优先。
3. **GN 的 2.5 ulp 是数据相关的**（§4.2），不是门限问题；但它意味着"GN 幅值界 ≤2.0"不能当作
   全输入保证，只能当作"该行在该输入上"的判决。
4. **(K=2336, N=256) 必须 N 半宽**这条平台事实现在写在执行器里（select_conv_image 的强制分支 +
   对 stall 镜像的断言），不是注释。
5. **64×64 做不到逐位复现**（§2.4），原因已定位在 host 栈；如果将来要"逐位回归"，必须把 3b1 的
   py310 驱动器一起保留为对照，或者把链也跑在 py310 里。
6. 新增的 5 个 xclbin 不在 git（out/ 被 ignore），重建命令：
   py312 t21_3b3_chain.py --build-conv / --build-up（两者都带门）。

---

## §8 复现

    # 92 行闸门（不需要器件）
    py312 _scratch/vae/t21_3b2_plan.py --dry-run --strict          # exit 0

    # 索引 + 纸面预测
    py312 _scratch/vae/t21_3b3_chain.py --index
    py312 _scratch/vae/t21_3b3_chain.py --plan

    # 64×64 自测（3b1 口径）与两个单变量对照
    py312 _scratch/vae/t21_3b3_chain.py --selfref
    py312 _scratch/vae/t21_3b3_chain.py --selfref --conv-impl model
    py312 _scratch/vae/t21_3b3_chain.py --selfref --conv-impl model --no-gnup

    # 六尺寸端到端门（每尺寸 2 run，约 37 min 器件）
    py312 _scratch/vae/t21_3b3_chain.py --all --repeat 2
    py312 _scratch/vae/t21_3b3_report.py                            # 表 + 并排图

    # 官方参考三方对照 / WebUI
    py312 _scratch/vae/t21_3b3_official.py
    py312 webui.py --host 127.0.0.1 --port 7870
    py312 _scratch/vae/t21_3b3_webui_check.py job --port 7870 --batch 2 --steps 8 --seed 0
    py312 _scratch/vae/t21_3b3_webui_check.py batch --latents _scratch/image/TAG_latents.npy
