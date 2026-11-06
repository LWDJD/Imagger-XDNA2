# T16 — 产品返工报告(P0 四项)

日期:2026-10-18 · 仓库:<checkout>
负责人否决理由(全部成立):①解码图像混乱;②VAE 有 NPU 版却用 CPU(20 s 加载);
③批量上限 4、4×128² 263 s;④计时 33.3+99.8+21.8=154.9 ≠ 263.0,108 s 未归账。

## 0. 验收总表

| # | 验收项 | 结果 | 证据 |
|---|---|---|---|
| ① | 计时分项之和 == 总计 ±1 s | **PASS** | §4,job 20261018-170459-ed2a32 closure P0.77 s(旧口径);新口径见 §4 |
| ② | 图像肉眼可用(并排对照) | **PASS** | §1.2 `T16-P0-1-side-by-side.png` |
| ③ | VAE 在 NPU(计时为证) | **未达成(如实报)** | §2:工作区里没有 t3_real_vae.onnx;NPU 解码器只有 kernel 级物料,整链未建。已交付常驻 CPU VAE 把 20 s 加载降为 0 |
| ④ | batch N>4 且每图与单图逐字节一致 | **PASS** | §3.3:batch 5 五图、batch 8 八图全部与各自单图 latents 逐字节相同(`gate.pass=true`);设备恢复后补跑 |

**交付状态:4 项里 3 项 PASS(① 计时闭合、② 图像、④ batch>4 逐字节门),
③ 未达成(无 NPU 整链,已如实报)。** 下面逐项给证据与剩余的精确工作量。

**P0-1 的结论推翻了任务书的分支前提**:host f32 确实是抽象纹理,但这不是
"128² 降尺度的模型分布问题"——官方 ZImagePipeline 在 128²、同 prompt、同 seed、
同 8 步下出的是一张清晰的湖景(§1.1)。真正的根因是 host/device 共用的
**patch token 内部布局**错了,官方 128² 参考把它当场证伪。修复后 128² 图像肉眼可用,
因此 256/512 不因图像质量而"必选",T15 的镜像重建没有半建(§1.4)。

---

## 1. P0-1 图像混乱:诊断与修复

### 1.1 先证伪"128² 分布外"

工作区里已有官方 diffusers `ZImagePipeline` 在 **128×128、prompt "a mountain lake
at sunrise"、seed 0、8 steps、guidance 1.0** 下的产物和逐步轨迹:

* `../zimage-port/_out/zimage_128_fixed.png` — 官方整链产物(湖景)
* `../zimage-port/_out/zimage_128_fixed.latents.npz` — 官方 8 步的 `v`(pre-negation)、
  `latents`(每步后)、`rms`(19 个检查点)
* `../zimage-port/_out/zimage_128_s8_seed0.png` — 早期官方 native 产物(同参数,湖景)

把官方**最终 latent** 用本仓库交付的 CPU VAE(`_scratch/pipeline/vae_decode.py`)
解码,得到的 PNG 与 `zimage_128_fixed.png` **逐字节相同**(sha256 `01156add...`,
24,010 B)。=> VAE 解码路径本身没有错;错在 latent。

所以判据不是"host f32 是否抽象",而是"官方 128² 是否可用"。官方 128² 可用,
**128² 不是分布外**;host f32 与 WebUI 同时抽象,说明两条路径共享同一个 bug。

### 1.2 并排对照图(肉眼可判)

`webui/evidence/T16-P0-1-side-by-side.png`:五格,同 prompt、同 seed 0、同 8 步、同 128²:

1. 官方 ZImagePipeline —— 湖景;
2. host f32 `t3_real`(BFP16 权重,channel-major patchify)—— 抽象纹理;
3. host f32 `t3_real`(只把 patchify 换成官方顺序)—— 湖景;
4. WebUI 器件产物(channel-major)—— 抽象纹理;
5. WebUI 器件产物(patch-major)—— 湖景。

第 3 格是独立的 host f32 对照,用同一批 BFP16 权重、同一 schedule、同一 seed,
只改 patchify 顺序:latent rms 1.439 -> 0.946(官方 0.821),absmax 6.29 -> 3.51
(官方 2.87),corr vs 官方 0.104 -> 0.538。host 与器件两条路径同向变化,把根因
锁死在 patchify,而不是器件算术或分辨率。

### 1.3 根因:2×2 patch 的 token 内部布局(T16-P0-1-FIX)

官方 `ZImageTransformer2DModel.patchify_and_embed`:

    x.view(C, F, pF, H, pH, W, pW).permute(1,3,5,2,4,6,0)
    -> 每个 token 内部 index = ((pH*2 + pW) * C + c)        # PATCH-MAJOR

交付的 `_scratch/image/t3_real.py::patchify`(`t13_run.py` 也经
`t3_real.x_embedding` 调用它):

    lat.reshape(b,c,gh,2,gw,2).transpose(0,2,4,1,3,5)
    -> 每个 token 内部 index = (c*4 + pH*2 + pW)              # CHANNEL-MAJOR

同样的 64 个数,换了个顺序喂进 checkpoint 的 `all_x_embedder.2-1`。更刺眼的是:
`t3_real.unpatchify` 本来就是 patch-major(和官方一致)——模型入口和出口用了两套
布局。自查为什么没看见:blockcheck 比的是"NPU vs 同一套 host 参考",patchify 在
block 之外;而逐层检查只看了 RMS,RMS 对 token 内置换不敏感。

**单变量 A/B(同一个官方轨迹步)**:取官方第 2 步输入 `L_2`(即 npz `latents[1]`)
和官方 timestep 947.368,只改 patchify 顺序:

| patchify | pred vs 官方 v 的 corr | l2rel | max abs |
|---|---|---|---|
| 交付的 channel-major | +0.43233 | 109.45 % | 6.151 |
| 官方 patch-major | **+0.99862** | **5.27 %** | 0.536 |

5.27 % 的残差就是 BFP16 权重相对官方 f32 权重的量化差,量级正常。

**整链 8 步的同口径数字**(同 prompt/seed/步数):

| | latent rms | absmax | corr vs 官方最终 latent | PNG |
|---|---|---|---|---|
| 官方 | 0.8213 | 2.8704 | 1.0000 | 24,010 B |
| WebUI 修复前 | 1.4340 | 6.2232 | +0.1047 | 39,282 B(纹理) |
| WebUI 修复后 | **0.9392** | **3.3540** | **+0.5702** | **22,926 B(湖景)** |

修复后与官方不是同一张图(初始噪声不同:官方 torch RNG,器件 numpy RNG),
但落在同一分布;corr 0.57 是"同构图不同采样"的水平。

**修复位置**:`webui/bridge.py::_install_token_layout_fix()`,在 `t13_run.main()`
之前把 `t3_real.patchify` 换成官方顺序(与 cap_embedding / load_schedule 同一
"只 patch 交付文件接缝、不改交付文件"的方式)。`unpatchify` 不动(本来就对)。
`t13_run.py` 通过 `t3_real.x_embedding` 取 patchify,所以一处 patch 同时覆盖 host
与 device。host f32 参考驱动 `_scratch/image/t16_hostref.py` 装同一补丁,用于
§1.2 的第 2 格(修复前)与复跑(修复后)。

### 1.4 分辨率结论(为什么没有半建 256/512)

任务书的分支是"host f32 也抽象 ⇒ 128² 分布问题 ⇒ 256/512 必选"。实测:
host f32 抽象、官方 128² 好、修 token 布局后 WebUI 128² 好。**分支前提不成立**,
所以不启动 T15 的 6 镜像重建(M=288 的 FFN/4 投影/IP + 新 perm + host 常量),
也不生成半套镜像。`webui/config.py` 的 512 序列长度从错误的 1152 修正为
**1056**(1024 image + 32 cap);256/512 仍标记未实现,UI 禁用。

如果将来要做 256(M=288):需要重建 FFN、4 投影、IP 共 6 个镜像 + 新 perm + host
常量;权重字节量按 K 不变、M×3 增长,器件每图读权重次数不变(逐图计),所以
256² 的每图 loop 成本 ≈ 128² 的 3 倍而不是 1 倍——这是预算过账,不是本轮交付。

---

## 2. P0-2 VAE:实测、缺口、以及已交付的部分

### 2.1 任务书里的 t3_real_vae.onnx 不存在

对 `<workspace>` 全树做 `*.onnx` 与文件名含 `vae` 的检索:

* 没有任何 VAE 的 ONNX;没有 `t3_real_vae.onnx`;没有"验证台架 1.8 s/图"的脚本或日志。
* 仓库里真实的 NPU VAE 物料是 `_scratch/vae/`:
  * `vae_npu.py` / `run.py`:35 个 conv 的 im2col+GEMM 形式,conv_in 实测
    **4.63 ms / 4.23 ms**(两次,逐字节相同),单 conv 与 device-exact 模型一致;
  * `group_norm.cc` + `vaeg_*`:GroupNorm(32) 6 个真实形状、30 个实例器件实测
    max **1.0–1.5 bf16 ulp**,99.83–99.95 % 逐位相同;nearest-2x upsample 3 个
    实例**逐位相同**;整链投影(both ops on device)PSNR 37.32 dB;
  * 但"整解码器上器件"从未串起来:VAEG 的 measured tier 是 **33 次 GN/upsample
    器件调用 + 35 个 conv 的 device-exact 模型(host 仿真)**,27.2 s 总;**不是**
    35 个 conv 都在器件上,也没有 1.8 s/图的整链计时。

器件实测(来自 `_scratch/vae/out/*_results.json`,不是本轮重跑):

| 算子 | 形状/实例 | 实测 |
|---|---|---|
| conv_in(GEMM) | M=64,K=144,N=64 | 4.63 ms / 4.23 ms 两次,逐字节相同 |
| GroupNorm(32) | (512,8,8) … (128,64,64),30 个实例 | max 1.0–1.5 bf16 ulp(以 bf16(model) 幅度为界),99.83–99.95 % 逐位 |
| nearest-2x upsample | 3 个真实形状 | 逐位相同(bit_exact=True) |
| 整链投影(35 conv 仿真 + 两个真实算子器件) | latent 8×8 | PSNR 37.32 dB vs CPU 参考 |

=> 结论:③ **未达成**。不是"有 NPU 版不用",而是 NPU 版只有 kernel 级物料,
整链还没建;任务书对现状的判断需要修正。

### 2.2 真正的 20 s:是加载,不是解码

同一 job 的 VAE 分项:**load 15.3–20.2 s,decode 0.43–0.46 s/图,
PNG save 0.018 s**。20 s 是每个 job 重新 import torch/diffusers 并加载
83.8 M 参数解码器的钱,不是解码本身。

已交付:**常驻 VAE**(`webui/vae_local.py`),py312 server 进程内缓存解码器,
第一次 decode 付一次加载,之后每个 job 的 VAE 只剩 decode+save。实测同一进程两次:

    call 0: load 19.630 s  decode 0.453 s  save 0.023 s  total 20.109 s
    call 1: load  0.000 s  decode 0.342 s  save 0.006 s  total  0.350 s   (57x)

PNG 逐字节相同(22,926 B)。失败自动回退到一次性的 `vae_worker.py` 子进程。
注意:这是 **CPU** 解码器;没有把它冒充成 NPU。

### 2.3 latent 布局/缩放核对(任务书要求"顺带查")

官方最终 latent -> 交付 CPU VAE -> 与官方 PNG 逐字节相同(§1.1)。说明
`latents / scaling_factor + shift_factor`(0.3611 / 0.1159)、通道顺序、
NCHW、`(x+1)/2` 全部与官方一致;器件输出的 latent 也是同一布局(修复后整图正确
即为证)。"混乱"不是 VAE 布局造成的,是 patch 布局造成的。

### 2.4 未达成的精确剩余量(不半建)

要把 VAE 真正放上 NPU,还差:

1. 35 个 conv 的**器件**执行链:每个 conv 一次 im2col(host)-> A 流 -> `mm_bfp`
   -> C(host),按 (M,K,N) 编译对应设计。现有 `run.py` 已验证单 conv 的
   正确性与 4.6 ms 级耗时,但只编译了 5 个 spec,整链驱动不存在。
2. 30 个 GroupNorm + 3 个 upsample 已有器件 kernel(见上),需要接进同一条
   driver;SiLU/add 已有 `silu_sized` / `add_sized`。
3. 整链计时与容差门:预计 conv 35 × 4–50 ms + GN/up 33 × ~0.8 s(VAEG 实测
   每次器件调用 ~0.8 s,含 host 往返)≈ 1.8 s/图**仅在 GN/up 的 host 往返
   被摊薄/合并后才成立**;当前 27.2 s 的 measured tier 说明往返是主成本。
4. 与 WebUI 的集成:py310 常驻 VAE 进程 + 器件单租户互斥。

---

## 3. P0-3 批量

### 3.1 上限

`config.MAX_BATCH = 16`(原 4);UI 批量输入与后端校验都读这个值;驱动本来就
支持 `--batch N --share-weights`。T10 验证过 8。

### 3.2 263 s 归账(4×128²,job 20261018-163636-f54297)

该 job 的真实分项与旧口径缺失项:

| 项 | 秒 | 旧 UI 是否显示 |
|---|---|---|
| 排队 | 0.00 | 否 |
| 构建缓存 | 0.03 | 否 |
| 文本编码 TE | 12.30 | 否(旧表只显示 3 项) |
| 启动/注册/载荷驻留 | 33.31 | 是(33.3) |
| 扩散 loop(4 图 × 8 步) | 99.80 | 是(99.8) |
| VAE load | 20.21 | 否(只显示 VAE 合计 21.8) |
| VAE decode(4 图) | 1.49 | 合计为 21.8 |
| **冷启动重试 2 次** | **91.45**(48.44 + 43.01) | **否** |
| 重试退避 + 编排/落盘 | ~4.3 | 否 |
| **总计** | **262.96** | |

旧口径 33.3 + 99.8 + 21.8 = 154.9;差额 108.1 = TE 12.3 + 重试 91.45 +
编排 ~4.3。**"108 s 未归账"就是这两次没被显示的首 dispatch 0xC0000005 冷启动
+ TE**。新口径把 retry_s、retry_attempts、TE、queue、save 全部显式列出,见 §4。

### 3.3 批语义门(逐字节):PASS

**判据**(逐字节,不是 PSNR/allclose):batch N 的第 b 张图 == batch 1、seed=b 的
单图 latents。8 个单图(seed 0..7)与 batch 5 / batch 8 都在同一 cache
(`cache/128x128`)、同一 schedule、同一 caption 下跑。结果:

| 批次 | 每图结果 | 每图 sha256 与单图 |
|---|---|---|
| batch 5 | img0..img4 **全部 ==** | 相同 |
| batch 8 | img0..img7 **全部 ==** | 相同 |

`t16_batch_gate.json`:`gate.pass=true`,N=5/8 每图 `equal=true`。
这是 T13 只比了 batch-4 的 img0 之后,第一次把 **N>4 的每一张**都对到自己的单图。

**摊薄曲线**(同 session、同设备状态;N=1 的 wall 含 4 次冷启动重试,不代表稳态):

| N | wall s | loop s | wall/图 | loop/图 |
|---:|---:|---:|---:|---:|
| 1 | 50.45 | 26.37 | 50.45 | 26.37 |
| 5 | 143.69 | 116.31 | 28.74 | 23.26 |
| 8 | 215.17 | 182.62 | 26.90 | 22.83 |

读法:每图 loop 从 26.4 s 降到 22.8 s(权重按图重读,摊薄来自每图的宿主
pack/传输与 dispatch 重叠);启动/注册/TE/VAE 是一次性成本,在 wall/图上摊得
更明显(50.5 → 26.9)。batch 4 的另一条实测:job 20261018-163636-f54297,
loop 99.80 s = 24.95 s/图,总 262.96 s(§3.2)。

**执行记录(为什么中间停了)**:设备栈在约 20 个器件进程后进入确定性
0xC0000005(`xrt_coreutil.dll+0x1228ce`,首 dispatch;cache-all 与
`--no-cache-all` 都复现,换 `NUMBA_CACHE_DIR` 无效,Windows Event 1000
同一偏移)。设备恢复后补跑:seed 4(2 次)、seed 6(3 次)、seed 7(5 次)、
batch 5(1 次)、batch 8(2 次)全部通过。这一类卡死仍是环境行为,不是批算术错;
README §9 #14 的"重试/重启"处置继续有效。重跑命令:

    py312 _scratch\image\t16_batch_suite.py      # 8 单图 + batch 5 + batch 8
    py312 _scratch\image\t16_gate_fix.py         # 补跑失败项 + 重算逐字节门

---

## 4. P0-4 计时全闭合

### 4.1 桶(每个 job 的每一秒只落一个桶)

queue_s / prepare_s / textenc_s / schedule_s / retry_s / startup_s / loop_s /
device_tail_s / vae_load_s / vae_decode_s / vae_save_s / gate_second_s / post_s

* startup 用 bridge 事件里的 **perf_counter**(`tp`)而不是 wall clock,和
  `stream_process` 的 wall 同一时钟;startup + loop + tail == 器件进程 wall。
* retry_s 计整个失败 attempt(进程 wall + 2 s 退避 + 记账),不再只计进程。
* 复现门第二遍整段进 gate_second_s,不和第一遍混算。
* unaccounted_s = total_s − Σ桶,UI 显示"未归账 X s",closure 门限 ±1 s。

### 4.2 实测

job 20261018-170459-ed2a32(8 步、batch 1、修复后):

    textenc 14.90 (load 11.10 + decode 1.99)   schedule 0.004
    startup 37.68   loop 27.73   device_tail 0.00
    vae load 15.28  decode 0.455  save 0.018
    post 0.003      total 96.86
    Σ桶 96.09 + 未归账 0.77 = 96.86  -> PASS

新口径复跑(含 perf 时钟 startup、重试含退避、常驻 VAE;job
20261018-191917-4e3451,1 次冷启动重试):

    queue 0.002  prepare 0.013  TE 12.645  schedule 0.003
    retry 48.559 (1 次失败进程 + 2 s 退避)
    startup 34.453  loop 27.569  tail 0.000
    vae load 20.474 (常驻第一次)  decode 0.494  save 0.021
    post 0.004
    Σ桶 144.236 + 未归账 -0.280 = 143.956  -> PASS

该 job 同时验证了 P0-5 的 cache/128x128 与新 UI 路径;PNG 22,926 B(湖景)。

### 4.3 UI

"当前任务"面板新增完整计时表 + "计时闭合:PASS/FAIL — 分项之和 vs 总计 vs
未归账(门限 ±1 s)"徽标;冷启动重试单独显示"N 次 / X s";不再只显示 4 个格子。
---

## 5. 证据索引

| 证据 | 路径 | 说明 |
|---|---|---|
| 并排对照图 | webui/evidence/T16-P0-1-side-by-side.png | 官方 128² / host f32 修复前 / WebUI 修复前 / WebUI 修复后 |
| 修复后产品 PNG | webui/outputs/t16fix_170459ed2a32_p0_s0.png | 22,926 B, sha256 fd7722e1... |
| host f32 修复后 PNG | _scratch/image/t16hostref_fix2.png | 同 prompt/seed/8 步,只改 patchify;湖景 |
| host f32 修复后 latent | _scratch/image/t16hostref_fix2_latents.npy | rms 0.946 / absmax 3.51 / corr 0.538 |
| 修复前产品 PNG | webui/evidence/E2E-prompt-8step.png | 39,282 B, sha256 90f0e2d6... |
| 计时闭合(旧口径) | webui/jobs/20261018-170459-ed2a32/job.json | sum 96.086 / total 96.857 / unaccounted 0.771 / pass true |
| 计时闭合(新口径+常驻VAE) | webui/jobs/20261018-191917-4e3451/job.json | sum 144.236 / total 143.956 / unaccounted **-0.280** / pass true |
| 官方 128² 参考 | ../zimage-port/_out/zimage_128_fixed.png | 同 prompt/seed/8 步;官方 latent 用交付 VAE 解码逐字节相同 |
| 官方 latent 轨迹 | ../zimage-port/_out/zimage_128_fixed.latents.npz | v / latents / rms,用于单变量 A/B |
| 单步 A/B 脚本 | _scratch/image/t16_fixprobe.py | corr 0.432 -> 0.9986 |
| host f32 参考驱动 | _scratch/image/t16_hostref.py | 同 prompt 的 t3_real host 参考(含 layout patch 开关) |
| 批语义门驱动 | _scratch/image/t16_batch_suite.py | 8 单图 + batch 5/8 + 逐字节比较 |
| 批语义门补跑 | _scratch/image/t16_gate_fix.py | 崩溃补跑 + 重算 gate |
| 批语义门结果 | _scratch/image/t16_batch_gate.json | **PASS**:N=5 五图、N=8 八图逐字节 == 各自单图;曲线 26.4->22.8 s/图 |
| 预构建入口 | webui/prebuild.py;POST /api/shapes/prebuild | 已构建=暖缓存;未构建=6 镜像计划 + 预算 |
| 263 s 原始 job | webui/jobs/20261018-163636-f54297/job.json | 4×128²,总 262.96 s,重试 91.45 s |
| 常驻 VAE 自测 | 本轮 stdout:_scratch/image/t16_vaelocal_test.py | 第二次 load 0.000 s / total 0.350 s |
| NPU VAE kernel 结果 | _scratch/vae/out/*_results.json | GN 1.0–1.5 ulp、upsample bit-exact、chain 37.32 dB |
| 冻结参考字节 | t3d_baseline.py --check | ALL REFERENCE BYTES UNCHANGED(收尾跑) |

## 6. 复现(最小集)

    # 1) 产品端到端(修复后,设备健康时)
    py312 webui.py --host 127.0.0.1 --port 7860
    # POST /api/jobs: prompt=a mountain lake at sunrise, steps=8, seed=0, batch=1
    # -> job.json 的 timing.closure.pass == true;PNG 是湖景

    # 2) 单步 A/B(patchify 顺序)
    py312 _scratch\image\t16_fixprobe.py

    # 3) 批语义门(需要设备未卡死)
    py312 _scratch\image\t16_batch_suite.py
    py312 _scratch\image\t16_gate_fix.py

    # 4) host f32 参考(同 prompt;约 9 分钟)
    py312 _scratch\image\t16_hostref.py --pe webui\jobs\20261018-170459-ed2a32\pe_p0.npz ^
          --tag t16hostref_fix --steps 8 --seed 0 --schedule-source official

    # 5) 参考字节
    py312 _scratch\image\t3d_baseline.py --check
