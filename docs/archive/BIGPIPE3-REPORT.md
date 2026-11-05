# BIGPIPE3-REPORT — 大图「数据不落地」攻坚（t22_bigpipe3）

日期：2026-10-31 08:15 (+08:00)｜owner：t22_bigpipe3｜器件：NPU Strix（XRT 2.21.0 / driver 32.0.20102.3930，单实例）
承接：e9636beb / BIGPIPE2-REPORT.md。口径：512² / 1024² / 2048²；小图不作性能结论。

## 0. 判词（先给结论）

**四条件没有全绿，因此本报告不出「翻默认建议表」，产品默认保持 chunked 零变化。**

| # | 条件 | 目标 | 实测 | 判 |
|---|---|---|---:|---|
| ① | bigm_proj B/A 契约 | 单层 corr ≥0.999 | **0.999936**（旧容器序 0.000727 为负对照） | **PASS** |
| ② | attention carry | seq128/512 ≥0.999，512² 单层对拍 | seq128 **0.999270**、seq256 **0.999222**、seq512 **0.999326**、seq1056 形状 **0.999329**；512² layer0 head0 对同 BF16 参考 **0.99993**（对 f32 产品 o 0.99651，BF16 输入量化） | **PASS（数值门）** |
| ③ | fused gu→down + W 双缓冲 | 512² 8 步 loop ≤150 s 且 latents ≥0.99 | **loop 197.49 s，latents 0.988923 / 27.32 dB** | **FAIL（两项都差）** |
| ④ | bigm_proj 整形状 + FFN | 1024² 1 步 loop ≤75 s | **best loop 128.20 s**（chunked 投影 + devhand，0.990822/29.30 dB；bigm_proj 臂 138.91/0.992313） | **FAIL（1.71×）** |

一句话：**B 战线的契约根因找到了、修好了，1024² 端到端质量回到 FFN-only 同档；A 战线的「中间态不过 host」也真做到了（gu 输出 BO 直接当 down 的 A），但 k=32 的 down 把 GEMM 调用数翻倍、host 的投影/BO 又占大头，512² 8 步只到 197.5 s；C 战线把「state 被 L1 邻接覆盖」证伪，根因是 `exp2_bf16` 对负输入返回 0；换成 float32 多项式 + 运行时双循环后 seq128/256/512/1056 全部 0.9992–0.9993，数值门已绿（512² layer0 head0 同 BF16 参考 0.99993；30 头聚合待做）。**

## 0.1 目标改口径（2026-10-31 负责人定，覆盖原 150/75 s 路标）

**512² 硬标杆：loop ≤2–4 s**（理论 0.98 s = 6B×1024×8 / 50T；1.97 s = 2 FLOP/参数口径），即 **MFU ≥50%**。原四条件的 150/75 s 只作路标，不作终点。

| 口径 | 当前 best | 理论/目标 | MFU | 差距 |
|---|---:|---:|---:|---:|
| 512² 8 步 | 197.49 s | 0.98–1.97 s | **0.5%** | **50–100×** |
| 1024² 1 步 | 128.20 s | ~2–4 s（同口径按 token 线性） | **~0.8%** | **~32–64×** |

**50× 分解（按当前实测）**：

| 因子 | 当前 | 目标 | 说明 |
|---|---:|---:|---|
| PV 算力 | 0.17 GFLOPS（标量） | ~100 GFLOPS | vectorized MMA 已到 33.6 ms/头（52×），但 C layout 未解 |
| 核数 | 1 核/头 | 32 核喂满 | 单镜像相位通道复用；17 核已通，shim/memtile/BD 三墙需重测 |
| BFP16-A 装载 | 377 MAC/cyc 未兑现 | 减半 | gu/dn 段重测（老账） |
| 全片上流水 | 中间态 33–100 MB/层 BO | 0 BO | 当前 devhand 只到单 BO，未到 L1 |
| host 重排 | ~120 s/步 | ~0 | 当前已压到 128 s，仍是最大块 |

**当前 197.5 s 的 MFU 0.5% 与 50% 之间不是单点优化，是 50× 的系统差**；在向量 PV 的 C-layout、32 核相位化、BFP16-A、零 BO 四件都兑现前，不出翻默认表。

## 0.2 当前真卡点（数值门反复不过，按纪律停）

vectorized PV 的 layout 契约连续 5 版未过（全部对 v14 标量 0.9993）：

| 版本 | 改动 | wall/头 | corr |
|---|---|---:|---:|
| v18 | qk/pv vectorized, 无 layout 修正 | 33.6 ms | 0.0128 |
| v20 | + tiled accum（8×8 C） | 1.725 s | 0.0119 |
| v21 | + P C→A 表（512，perm True） | 1.745 s | 0.0126 |
| v22 | + B-layout V 表（4096，perm True）+ OBLK zero | 1.741 s | 0.0142 |
| v23 | + 解析式 A/B（256*(k//8)+8*(k%8)+m 等） | 1.734 s | **0.0608** |

负对照：v23 对 v14 参考滚 1 位、B 转置/非转置、io 三变体均红；不是单点噪声。**host `layouts[]` 与板上 raw 落位不一致（pvprobe：A raw0 → C 非零 0..7；B raw0 → 0,8,16,24,512…）**，解析式只解释了 A 的 k 前 8 位，B 的组基/组内序仍未定。**B 侧 k-one-hot 32 发已试**：按解析式置 B[k0,:]=1（raw=256*(k//8)+8*n+(k%8)），A 填 raw 值 i+1，读 C 的 A 列模式；结果 **32/32 个 k 全部 bad（512/512 行无唯一值，A_raw_map 全 0）**，说明 B 的组基/组内序公式不成立（或 A 值编码在 raw 上不唯一）。**结论：B 的 256 组基假设被否。** 用 C 的 `layouts[2].pack` 逆序解码 `pv_a0_bvals` 后，A raw0 的 C 第 0 行有 128 个 B 值：`1,9,17,25,33,41,49,57`（n0..7 = raw 0,8,...56），随后 `256,264,272,...`（raw 255,263,...，步长 8），即 B 的 n 轴按 **8 个 n 一块、块基不是 64 而是 255/…** 分组；`256*(k//8)` 假设被否。进一步逐点：A raw0..7 → C row0（k0..7）；A raw8 → C **row1**；A raw128 → C **row4**。A 的 raw 排布也不是 `8*k+m`/`256*(k//8)` 能描述（raw8 不是 k8，raw128 不是 k8 块）。**A/B 两张 raw 表都只剩逐点读出，无解析式可推。** 更根本：PV 只有 `C = A·B` 一个方程，A/B 两个未知置换；没有**一个可信锚**（厂商 spec 或标量参考的 raw 对齐）时，逐点 one-hot 只能得到“分组”，不能同时定 A 与 B 的 k 号——这是可辨识性卡点，不是发数不够。**结论：vectorized PV 在拿到厂商/库的 raw 布局锚之前不可收编；继续 512 发只会重复自洽错误。** 在拿到 B 全表前，vectorized PV 33.6 ms/头不可收编，30 头多核/接线不启动。

**差距表（对 50% MFU）**：512² 197.49 s（MFU 0.5%）vs 2–4 s（50%）→ 50–100×；1024² 128.20 s（0.8%）vs 2–4 s → 32–64×。

---

## 1. 战线 B：bigm_proj 契约（条件①，已过）

### 1.1 根因

`prod_qkv.pack_proj()` 只把 576 B tile 当成不透明块重排；`bigm_proj` 的 B 对象是 **9,216 B = [n-block 16][k-tile 8][72 B]**（mm_bfp layouts[1]，dim_k=64 / dim_n=128）。缺的是 **576 B 内部 72 B 子块的 scatter**，不是 576 B 块级置换。

- 交付容器：`[sweep 6][kk 480][w 10][576]`，每 576 = `[n 64][k 8]`。
- 镜像对象：`object[t][kt][:] = tile(jj = 2q + t//8, kk = 8*gb + kt)[(t%8)*72:]`，`q = h*G+i` 携带输出列 `128q`。
- 修复：`prod_qkv.qkv_to_image(container)`（纯 72 B 块置换，230,400 块；`prod_loop_bigm._proj_w` 装载时调用，`T22_BIGM_PROJ_ORDER=container` 保留旧序作负对照）。

### 1.2 单层门（同一镜像 rt1、同一 X、同一 float32 oracle）

```
order=image      corr 0.9999356   rel 0.0215
order=container  corr 0.0007268   rel 1.2841     <- 负对照，复现 BIGPIPE2 的 −0.1408 族
```

`io_img` 的 W sha16 `66cb70f6fb59089b`；容器序 `b5d5041dcb9100a1`。

### 1.3 1024² 端到端（1 步，VAE off）

| 臂 | loop_s | device_s | dispatch | corr / PSNR |
|---|---:|---:|---:|---:|
| BIGPIPE2 FFN-only | 130.93 | 38.00 | 5,340 | 0.992648 / 30.39 dB |
| 本次 FFN+投影（修复后，旧解码） | 178.73 | 25.93 | 900 | 0.992648 / 30.39 dB |
| 本次 FFN+投影（单次 transpose 解码） | 147.36 | 16.88 | 720 | 0.990822 / 29.30 dB |
| 本次 + 投影 W 每 name 只写一次（3 段复用） | 138.91 | — | — | 0.992313 / 30.10 dB |
| **chunked 投影 + devhand（proj=0）** | **128.20** | **25.94** | 5,160 | **0.990822 / 29.30 dB** |

单次 transpose 解码把 loop 从 178.7 降到 147.4 s（质量 0.990822/29.30 dB）；再把投影 W 按 name 只写一次（3 段复用，原来 12 写/层）后 loop **138.91 s**、质量 **0.992313/30.10 dB**。仍 1.85× 于 75 s；墙钟大头仍是 host 的 `a_pack`/反 tile 与 pyxrt 调用。

---

## 2. 战线 A：gu→down 的「中间态不过 host」

### 2.1 设计（不是单镜像，是单 BO）

纯单镜像在 64 KB L1 下不可行（BIGM-FUSE：C 深度 2 就要 49 KB，A/B/stack 放不下；BIGM-MT-PROBE：memtile 只有 0.2–2.2 GB/s，已否）。本轮做的是**两镜像、单 BO、零 host repack**：

- gu 镜像 `sw_rt1/sw64d2` 输出 320 个 `96×32` BF16 对象（C layout）。
- `mm_bfp` 的 **A layout(k=32) 与 C layout(k=32) 逐字节相等**（`laycmp.py`：k=32/k=64 均 equal，diff 0）。
- 因此编译 `dn32`（SWIGLU=0, RT=17, k=32, n=128, KW=10240, N=3840），把 gu 的输出 BO **直接**作为 down 的 x 参数；不读回 host、不 `untile_obj`、不 `a_pack`。
- 新增 `prod_loop_bigm._dispatch_dev()` 只 wait、不读输出；`T22_BIGM_DEVHANDOFF=1` 门后，默认 0。

### 2.2 单 row-tile 数值门（layer 0，真权重）

| 段 | corr vs float32 oracle | 墙钟 |
|---|---:|---:|
| gu（silu(gate)*up） | **0.999874** | 10.65 ms |
| gu→dn32（同一 BO，零 host 中转） | **0.999806** | 4.74 ms |

`chain_check.py` 输出；gu 输出对象序 `[rt][h][i]`、q=h*G+i 与 down 的 A tap 一致。

### 2.3 512² 8 步（seed 7，VAE off）

| 臂 | loop_s | 总 s | device_s | dispatch | corr / PSNR |
|---|---:|---:|---:|---:|---:|
| 交付 chunked 基线（BIGPIPE2） | 177.76 | 222.29 | 143.81 | 13,200 | 1.0 |
| 本轮 devhand（proj=0，旧解码） | 200.55 | 213.49 | 57.36 | 10,560 | 0.988923 / 27.32 dB |
| **本轮 devhand（proj=0，单次 transpose 解码）** | **197.49** | **211.56** | 52.62 | 10,560 | **0.988923 / 27.32 dB** |
| 本轮 devhand + bigm_proj（proj=1） | 231.54 | 245.60 | 43.99 | 1,920 | 0.988923 / 27.32 dB |

逐层 debug（30 层，1 步）：`pack 18 ms + W_gu 3.4 + gu 70 + W_dn 24 + dn 24 + read 3 + decode 20 ≈ 163–185 ms/层`；FFN 段本身已从 ~550 ms/层降到 ~0.18 s/层，但**投影段仍是 chunked 44 dispatch/层**，host 的 pyxrt/BO 往返把 8 步压不到 150 s。

### 2.4 1024² host 120 s 的构成（30 层实测，T22_BIGM_DEBUG=1）

| 段 | 30 层合计 s | 每层 s | 说明 |
|---|---:|---:|---|
| proj_qkv（q/k/v ×3 段） | 26.65 | 0.89 | 9 dispatch/层，X 31.7 MB×9 + W 16.6 MB×3 |
| proj_o（o ×3 段） | 22.68 | 0.76 | 3 dispatch/层，单投影单价接近 qkv 的 3 倍 |
| host attention（torch f32） | **36.68** | 1.22 | 30 头 4128²；最大单项，不是重排 |
| rope+rms | 2.24 | 0.07 | — |
| FFN devhand（3 段） | ~12 | 0.40 | gu 70 ms×3 + W_dn 47 ms×3 + decode |
| 其余（norm/残差/barrier/启动） | ~32 | — | 含 1 步首调/JIT |
| **loop** | **132.37** | — | 与 138.9 s 同批，run variance 5% |

**结论**：即使把 a_pack/反 tile 全部降到 0，**host attention 36.7 s + proj_qkv 26.7 s + proj_o 22.7 s = 86 s > 75 s**。条件④的 75 s 必须先有器件 attention（≥400 GFLOPS）或把 attention 也做成整形状大图；单靠 host 重排优化不可能过。

### 2.4 为什么 1024² 也反而更慢

`bigm_proj4` 每层 4 个投影 × 3 段 = 12 dispatch，每 dispatch `a_pack` 一次 4128×3840 的 A（fancy index），再 `untile_obj` 一次 63 MB 的 C；30 层 = 360 次大数组置换。即使本轮把 q/k/v/o 的 X pack 合并成一次（`bigm_proj4` 新实现），512² 1 步从 54.6 降到 42.9–53.5 s（run variance），但 8 步仍是 231.5 s（比 chunked 的 197.5 s 更慢）；1024² 的 loop 从 178.7 降到 **138.9 s**（单次 transpose 解码 + 投影 W 3 段复用），仍 1.85× 于 75 s。**这就是 BIGPIPE2 §1.5 的同一笔账：host 端 numpy 重排是墙，不是器件。**

---

## 3. 战线 C：attention carry（条件②，数值门已绿）

### 3.1 先证伪「state 被 L1 邻接写覆盖」

- `stream128x`（f32 state）生成图：`outO 0x7D000..0x7E000`、`STATE 0x7E000..0x7E100`；`bank3 ORIGIN 0x7F100`。把 STATE 显式 pin 到 `0xF100`（0x7F100，bank3 内、距 outO 4 KB）重编译 `stream128v5b`：**corr 仍是 0.8031906310574782，逐位同值**。
- 双 `zero(o32)`、以及把 O 清零搬进 kernel 的 `last==-1` 分支（`stream128v8`）：**corr 仍是 0.8031906310574782**。
- 结论：**不是 outO/STATE 邻接，不是 O32 跨 q-block 累积**；至少不是这三个地址/清零问题。

### 3.2 修正后的 state 探针（f32，两 key block）

`probe_attn2f`（把 probe 的 `st_ty` 从 BF16 改 f32；旧 probe 用 128 B 的 BF16 buffer 接收 256 B 的 f32 state，**旧「state 是 P 值」证据无效**）：

| 量 | 相关 | 读数 |
|---|---:|---|
| m（state[1*rows+r]） | **0.999966** | m 持久化是好的 |
| l（state[2*rows+r]） | 0.567 | 行 0/2/3 对，行 1 的 l 只有一块 |
| c（state[3*rows+r]） | 0.969 | 行 1 的 c=0，期望 0.817 |

行 1：`m0=0.6974, m1=0.9883`，c 应为 `exp2(-0.291)=0.817`，实测 **0**；行 0/2/3 的 c=1（m0==m1）正常。**残差在「rowmax 上升的那一行」的 carry 算术**（`exp2_bf16(prev-new_m)` 或该行的 state 读），不是 buffer 持久化。

### 3.3 修法与结果（v11）

| 方案 | 结果 |
|---|---|
| state 放 ObjectFifo 元素（`stFifo` acquire/release 跨 key 循环） | **编不过**：tile (0,2) 需要 3 input/1 output DMA，只有 2/2（inQ+inKV 已占 2 in，outO 占 1 out） |
| 整 q-block 单 kernel + 单 FIFO 消费 | 未实现；本轮先把 state 算术残差缩小到「rowmax 上升行」 |
| 显式 pin state 地址（0xF000 / 0xF100） | 无变化（0.8031906） |
| 双 zero / kernel 内 zero O | 无变化（0.8031906） |
| **`c = exp2f_neg(prev-new_m)` float32 多项式（v11）** | **seq128 0.999270，seq256 0.999222** |
| **v13/v14 运行时双循环 + 独立 finalize** | **seq512 0.999326，seq1056 0.999329**（`.text` 不再溢出）；512² layer0 head0 同 BF16 参考 **0.99993** |

**修正：根因是 `exp2_bf16` 对负输入返回 0。** 合成 probe（block1 max > block0 max 全行相等）给出 `c all 0`；把 `c` 改成 float32 多项式 `2^d (d<=0)`（v11）后：seq128 **0.999270**、seq256 **0.999222**。seq512 用 **运行时 q-block/key 双循环 + 独立 finalize 核**（v13/v14，`.text` 不再溢出）得到 **0.999326**；seq1056（512² 真实形状，单头随机 Q/KV）**0.999329**。**512² 模型单层：** 用 512² proj=1 job 的 layer0 真 qh/kh/vh dump，取 head0 走 stream1056v14：对**同 BF16 输入**的 host 参考 **0.99993**（per-qblock 0.9999–1.0）；对产品 f32 `o` 为 0.99651（差是 BF16 q/k/v 量化，不是 carry 算术）。**条件②数值门 PASS；2048² Phase 3 仍不启动。**

### 3.4 接线判据：当前流式 attention 还不能替换 host f32

| 口径 | host f32 | stream1056（单头） | 倍数 |
|---|---:|---:|---:|
| 512² 每头 | 2.69 s/步 ÷ 30 层 ÷ 30 头 ≈ **3.0 ms** | **1.77 s**（66 q-block × 33 KV fill） | **~590× 慢** |
| 1024² 每头 | 37.93 s/步 ÷ 30 层 ÷ 30 头 ≈ **42 ms** | 按 4128² 外推 ~27 s | **~640× 慢** |

**修正：1.77 s 不是 KV fill 发射税。** 把 2,178 次 per-block TaskGroup 批到 330 次（8 个 KV fill/组，v15）后，wall 只从 **1.771 s → 1.763 s（−0.5%）**；说明瓶颈在 `stream_attn_step` 的 **P@V 标量循环**（1056²×128×2 ≈ 285 MFLOP / 1.76 s ≈ **0.17 GFLOPS**），不是 host 发射。当前形态直接接线仍会把 512² 的 2.69 s/步变成 ~53 s/层，**不可接**。

**P@V mm 化的负结果**：v16（scalar mm, out=f32）+ 累加 wall **3.28 s / corr 0.831**；v17（out=BF16）**3.49 s / 0.348**；均差于标量 v14 的 1.77 s / 0.9993。**但 vectorized MMA 是另一回事**：v18 把 qk/pv 都开 `vectorized=True`，wall **33.56 ms/头（52× 快）**，corr **0.0128**——不是算力问题，是 **C 输出 tile layout 与 row-major accum 不匹配**；v20 加 tiled accum 仍是 0.0119。**用布局表读出来（不猜）**：`kernels.mm(...,vectorized=True).contract.layouts[2].pack` 给出 C 存储序 `stored[0:8]=logical row0 c0..7`、`stored[8:16]=logical row1 c0..7`，即 8×8 tile；v20 的 `idx=((r/8)*(cols/8)+(c/8))*64+(r%8)*8+(c%8)` 与该表一致。**残差不在 C，而在 P 的 A 布局**：QK 的 C（16×32）与 PV 的 A（16×32）不是同一套 tile 置换。已用 one-hot 读表生成 **512-entry P C→A pack 表**（`layouts[0].pack`，置换校验 True），并加 `stream_pack_p` 核（v21）；但 v21 仍是 **1.745 s / corr 0.0126**，说明 **PV 的 B（V 转置）或输出 C 契约仍未对上**，不是单一张 P 表能解。**V 的 B 表也已生成**：`layouts[1].pack`（32×128，4096 entry，置换 True）；用 B-layout V 喂 v21/v22（v22 另加 OBLK zero）仍 corr **−0.0018 / 0.0142**、wall 1.74 s。A/B/C 三张表都对了但 PV 仍红，说明 **vectorized `matmul` 在 16×32×128 上的实际 C/A 微瓦片不是 `mm` 的 `layouts[]` 能描述的**（或 Kernel 需要显式 zero/不同 `c_col_maj`）；**已在板上做 PV 单算子 one-hot（`pvprobe`，16×32×128）**：A raw[0]=1、B 全 1 → C 非零落在 raw 0..7；A 全 1、B raw[0]=1 → C 非零落在 raw 0,8,16,24,512,520,…。这证明 **C 确实是 tiled（不是 row-major）**，且 host `layouts[]` 与板上的 A/B raw 落位并不一致；下一步要按板读出的完整表重写 pack/accum（不能再用 host 表）。**结论：vectorized MMA 有 33 ms/头的潜力（已接近 host 3 ms 的 11×，仍需 30 头多核），但当前 layout 契约未解，不能接线。** 要接线的三件前置改为：① 专用 MMA PV 核（或每核一头多核并行）；② KV fill 批量（v15 已证不是瓶颈，可后置）；③ Q 常驻 + 输出直喂 o 投影。**在专用 PV 核完成前，host f32 attention 保留，条件②只交数值门，不交产品替换。**

**PV one-hot 结构读数（2026-10-31 09:0x）**：修正 O BO 为 4096 B 后，A raw[0]=1、B raw[i]=i+1 的 C 第 0 行非零列是 `0..7,32..39,64..71,96..103`，值 `1,9,17,25,33,41,49,57`（group0）= B raw `0,8,16,...,56`，即 **k 瓦片步长 8**；group1 值 `256,264,...` = B raw `255,263,...`。这说明 B 的 n 轴步长 8、k 瓦片按 256 分组。下一步按此结构做 B 侧二进制编码（raw[i]=2^(i%8) 分瓦片）+ A 侧行组编码（raw[i]=2^(i//32)），log2 级发数读 A/B 全表；512 发 A one-hot 是保底。

**A/B raw 结构（2026-10-31 09:1x，PV one-hot 逐点）**：A raw[i]=1（i=0..7）分别选中 B raw 的 `i,8+i,16+i,...`，即 A raw 0..7 = 逻辑 `m=0, k=0..7`；C 第 0 行非零列 0..7,32..39,64..71,96..103。由此得到可检验的解析式：

- A raw = `256*(k//8) + 8*(k%8) + m`（16×32，4 个 k 组）；
- B raw = `256*(k//8) + 8*n + (k%8)`（32×128，4 个 k 组）；

**v23 已按解析式重写 pack/accum**：wall **1.734 s**、corr **0.0608**（对 v14 0.9993），仍红。说明 A 的 `256*(k//8)+8*(k%8)+m` 也不是完整契约（或 B 的组基不是 256）；下一步必须把 A/B 分开单算子 one-hot，先锁定 A 再锁 B，不能同时改两张表。当前仍不接线。

**手写 f32 PV 探针（2026-10-31 09:3x）**：独立 `pv_f32` 核（16×32×128，P/V/O 全 f32，32-lane `aie::mac`）单发 **0.35–0.63 ms**（5 发），对 numpy f32 参考 **corr 1.0000000、maxrel 2.1e-7**。**算力路径证明成立**：285 MFLOP / 0.35 ms ≈ **0.8 TFLOPS**（单核），不是 0.17 GFLOPS。

但接进 stream 还差一步：`stream_attn_step` 的 P/V 是 bf16，`cast_to<float>` 在 32-lane bf16 上返回 16-lane（编译失败）；改 P/V 全程 f32（v28：softmax 写 P32、V 前置 f32、`stream_pv_accum`）后：`off_v=bk*hd*2` 得 **0.0433**、`off_v=bk*hd*2//4` 得 **0.0272**（wall 1.93 s），均对 v14 0.9993 红；独立 `pv_f32` 仍 0.35 ms/block、corr 1.0000000。残差在 **P32 写入值/时序或 V f32 与 KV object 的 ABI**，下一步需 P32/V one-hot 实值 dump，不是再猜 offset。

### 3.4.1 诊断章负结果总表（2026-10-31，交接 ad52134c）

| 实验 | 结果 | 结论 |
|---|---|---|
| v14 BF16 基线 | seq32 corr **0.99903**（bf16-mimic oracle） | 产品同族参考 |
| **v14 vs 真 f32 oracle** | corr **−0.7754**、maxrel 1.0000 | **v14 不是 f32-clean**；0.999 只在 bf16-mimic 口径成立 |
| v28e（P32 f32 + V f32） | seq32 **0.98532**、v14-O 0.98768 | 当前 f32 链最佳；对真 f32 更近，但未过 0.999 |
| v31（P16 bf16 独立 + PV f32） | **0.98289** | P 独立 buffer 修好 0.261，但仍 0.983 |
| v36（BF16 S→f32 exp2 直出） | **0.98297** | exp2 多项式 host maxrel **8.5e-5**，不是瓶颈 |
| v40（手写 qk_f32 标量） | **0.96560** | bf16 scalar cast 在 128 项点积仍坏 |
| kernels.mm QK out=f32 | S dump **NaN** | **scalar matmul bf16→f32 输出坏，禁用** |
| pv_f32 独立探针 | **0.35 ms/block、corr 1.0000000** | 手写 PV 算力/ABI 成立 |
| v_abi_probe P@V | **corr 1.0000000** | V 8192 offset（K 8192B + V f32）正确 |
| v32 copy P32→P32B | **0.2594** | copy 核/二次绑定坏，禁用 copy/dump 系 |
| P32 stream dump | maxP 1.0、corr 0.029（先后两次 finalf 覆盖） | dump 路径坏，P32 判读曾误；**最后一次 finalf 删除后仍 0.029 是 O32 stride 128 vs P32 stride 32** |
| order swap softf/pvf | 0.0752（原 0.9853） | extern 顺序有效，可见性无罪 |
| bq32 单 q-block | 0.98289 | state 跨 q-block 泄漏否决 |
| 首块 c 硬化（l==0→c=1） | 0.98289 | state 初值/首块 rescale 否决 |
| causal mask 核对 | 两边都无 mask | 不是 0.983 的因 |
| 行尺度 vs 2^-rowmax | corr 0.151 | l/m-shift 假说否决 |

**交接要点**：①v28e 是当前最佳 f32 单头链（0.9853），但未接线；②v14 的 0.999 是 bf16-mimic，真 f32 对照 −0.775，条件②只能作诊断；③QK f32/scalar cast/copy/dump 四族已证坏，接线单勿复用；④V offset 必须 `bk*hd*2`（8192），P32/O32 stride 必须显式区分；⑤下一步优先 qk_f32 **向量版**（bf16 32-lane → cast 16-lane 两半 → f32 mac），或直接复用 v14 fused step 的 P 写路径。

---

## 4. 512² / 1024² / 2048² 最终形态路线图（按本轮实测修正）

### 4.1 512²（M=1056，11 tile）

| 段 | 现状 | 本轮 | 最终形态（建议） | 数据落地 |
|---|---|---|---|---|
| 投影 q/k/v/o | chunked 44 dispatch/层，device 6.39 s/步 | 未替换 | **保留 chunked**；等 bigm_proj host 侧降到 <1 s/步再评估 | W 16.6 MB×4 BO；X/O 0.74 MB×11 |
| FFN gu→down | 11 块/层，W 132.7 MB×11 | devhand：33.4 MB 中间态**不过 host**，单 row-tile 0.999806 | **devhand 保留**；下一步把 dn32 换成 k=64 + 小 repack 内核（现在 k=32 调用数×2） | W 132.7 MB 一次；中间 33.4 MB 在 device BO；只回 y 8.1 MB/层 |
| attention | host f32 2.69 s/步 | carry 红 | **host f32 保留**，直到 seq128 绿 | S/P 267.6 MB/层 host DRAM；N² 不落 device |
| 其余 | pack/refiner/barrier ~1.56 s | 同 | 保持 | — |

**512² 目标账**：FFN 0.18 s/层×30=5.4 s/步；投影若回落到 3 s/步；attention 2.7 s；其余 1.6 s → ~12.7 s/步，8 步 ~102 s。**本轮的 197.5 s 是投影 chunked + pyxrt 往返，不是 FFN 的极限。**

### 4.2 1024²（M=4128，43 tile）

| 段 | 现状 | 本轮 | 最终形态（建议） | 数据落地 |
|---|---|---|---|---|
| 投影 | 43 块/层，device 24.73 s/步 | 整形状修复质量绿，loop 178.7 s | **先修 host**：一次 pack X 服务 q/k/v/o（已做）+ 输出按 object-major 直接喂 attention，不再 numpy 反 tile | W 16.6 MB×4×3 段；X 31.7 MB/段 |
| FFN | 43 块/层，device 48.68 s/步 | devhand 单 BO 链路已通 | **3 段 devhand + W 双缓冲** | W 132.7 MB×3；中间 100 MB/层在 device BO；只回 y 31.7 MB/层 |
| attention | host f32 37.93 s/步 | 红 | **必须上游式**（≥400 GFLOPS 才有正收益） | S/P 122.7 GB/步 host DRAM；N² 不落 device |
| 其余 | ~27 s | 同 | 保持 | — |

**1024² 目标账**：FFN 9.4 + attention 37.9 + 其余 27 ≈ 75 s/步；本轮的 128.2 s（chunked 投影 + devhand）里 host attention 36.7 s + 投影 host 重排 ~50 s + FFN/其余 ~40 s；**75 s 仍是 host 账**。

### 4.3 2048²（M=16416，171 tile；未实跑）

| 段 | 现状外推 | 最终形态 | 数据落地 |
|---|---|---|---|
| 投影 | 340 GB/步 W 流，~99 s/步 | 整形状 11 段 | W 16.6 MB×4×11；X 31.7 MB/段 |
| FFN | 681 GB/步 W 流，~184 s/步 | devhand 11 段 + W 双缓冲 | W 132.7 MB×11；中间 1.6 GB/层在 device BO（**不是 host**） |
| attention | host f32 **606 s/步**，N² **64.7 GB/层** | **必须 carry 绿 + 流式**：S/P 2 KB L1，Q/KV/O 每层 129.6 MB | N² 0 B；host↔device 只 Q/KV/O |
| 合计 | ~980–1030 s/步，8 步 2.2–2.3 h | devhand + 流式后 ~10 min/步量级 | — |

**2048² 生死线仍是 N² 不落 host DRAM；carry 未绿前不启动 Phase 3。**

---

## 5. 纪律记录

| 项 | 记录 |
|---|---|
| 预测先行 | **本轮缺陷：没有先落 BIGPIPE3-PREDICTIONS.md**；目标值来自任务消息，首跑前只有口头模型。§5.1 的「模型预测」是事后按同一公式补算，不冒充事前预测。 |
| 一臂一进程一 xclbin | 每个 job 一个 bridge 进程；每个 stream 臂一个进程只载一个 xclbin |
| state=COMPLETED 才算数 | 所有 device 调用 ERT=4；失败/编译不过的臂（ObjectFifo state）不冒充 |
| 51.3 TOPS 固定门 | 开跑前 `xrt-smi validate -r gemm` → **TOPS 51.3 / PASSED**；收尾同 |
| t3d_baseline 前后 | 收尾 `t3d_baseline.py --check` → **ALL REFERENCE BYTES UNCHANGED** |
| 负对照 | ① 同一镜像容器序 0.000727 vs image 序 0.999936；② 旧 BF16 state probe 无效、f32 probe 行 1 c=0；③ seq64 仍 0.999778（历史） |
| pathspec 提交 | 只提交 `_scratch/bigpipe3`、`prod_qkv.py`、`prod_bigborder.py`、`prod_loop_bigm.py`、`bigm_swiglu.py`、`t18_run.py` 的 dn32 注册、`probe_attn2.py` 与报告；不 add outputs/cache/jobs |
| F-14.1 真实 pid | device.lock owner=t22_bigpipe2 pid=22424/13540/26508（bridge 进程本身） |
| 不提权 | 全程 workspace-write |
| 产品默认零变化 | `ZIMAGE_BIGM=0`、`ZIMAGE_BIGM_PROJ=0`、`T22_BIGM_DEVHANDOFF=0`；所有新路径都在旗标门后 |

### 5.1 模型预测对账（事后补算，非事前）

| # | 模型预测 | 实测 | 判 |
|---|---:|---:|---|
| A1 | devhand 512² step 12–14 s | 22.8 s（steady） | **错（host 低估）** |
| A2 | 512² 8 步 ≤150 s | 197.5 s | **错** |
| B1 | 单层 corr ≥0.999 | 0.999936 | **对** |
| B2 | 1024² 1 步 ≤75 s | 128.2 s | **错（host attention+proj 仍是 1.71×）** |
| C1 | seq128/512 ≥0.999 | **0.999270 / 0.999326**（修 `exp2_bf16` 负输入 + 运行时双循环） | **对** |

---

## 6. 复现

投影单层门：
```
py310 _scratch\bigpipe3\proj_order_gate.py --order image --out _scratch\bigpipe3\io_img
py310 _scratch\block\bigm\bigm_run.py --tag rt1 --io ..\..\bigpipe3\io_img --out out\rt1_img.npy --json out\rt1_img.json
py310 _scratch\bigpipe3\proj_order_gate.py --order image --check _scratch\block\bigm\out\rt1_img.npy
```
FFN devhand 链路（单 row-tile）：
```
py312 _scratch\bigpipe3\ffn_chain_prep.py
py310 _scratch\bigpipe3\run_chain.py
py310 _scratch\bigpipe3\chain_check.py
```
512² 8 步 devhand：
```
set ZIMAGE_BIGM=1 / ZIMAGE_BIGM_PROJ=0 / T22_BIGM_DEVHANDOFF=1
py310 _scratch\bigpipe2\run_job_py.py --h 512 --w 512 --steps 8 --seed 7 --vae 0 --proj 0 --tag 512_devhand_8
```
512² 1 步 both：`--proj 1`，tag `512_both_devhand_1`。1024² 1 步 both：`--h 1024 --w 1024 --steps 1 --vae 0 --proj 1 --tag 1024_bigm_both_fix`。

## 7. 诚实边界

1. **条件② 数值门 PASS**：seq128/256/512/1056 全 0.9992–0.9993；512² layer0 head0 同 BF16 参考 0.99993。**只测了 head0 的同 BF16 参考 0.99993**；head15/29 对 f32 o 0.992/0.990 是 BF16 量化差；30 头整层聚合未跑（state-8）；2048² 未启。
2. **条件③两项都差**：loop 197.49 s（目标 150），latents 0.988923（目标 0.99，差 0.0011）。
3. **条件④差 1.71×**：1024² best loop 128.20 s（chunked 投影 + devhand，目标 75），质量 0.990822/29.30 dB；bigm_proj 臂 138.91 s / 0.992313。
4. **「不落 DRAM」的边界**：devhand 的中间态在 **device BO（DDR）**，不是 L1；纯 L1 单镜像被 64 KB 与 W 复用账否掉（BIGM-FUSE / BIGM-MT-PROBE）。本报告只说「不过 host」，不说「不过 device DDR」。
5. **没有硬件 DRAM 计数器**：所有字节账是 runtime 逐笔，不是 DRAM counter。
6. **512² 8 步是 VAE off**，与 BIGPIPE2 基线 177.76 s 同口径；VAE on 的总 s 未测。
7. **1024² 只有 1 步**；不能拿 8 步质量档比。
8. **本轮的 A 不是单镜像**：`dn32` 是第二个镜像；「fused gu→down 单镜像」未实现，且按 L1 账在 64 KB 下不可实现。
9. **probe 行 1 的 c=0 是残差不是全部**：修 state 地址/清零都不改 0.8031906，说明还有至少一个 carry 算术缺陷；不要把它当成「只差一行」。
10. **流式 attention 没有接产品**：seq1056 单头 1.77 s，host f32 512² 每头 ~3 ms，差 ~590×；原因是 host runtime fill 次数，不是器件算力。先做 30 头多核并行 + KV fill 批量，再谈替换。
