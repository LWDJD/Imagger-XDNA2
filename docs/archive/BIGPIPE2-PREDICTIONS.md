# BIGPIPE2-PREDICTIONS — 开跑前写的预测（两战线）

日期：2026-10-30 23:0x (+08:00)｜owner：t22_bigpipe2｜器件：NPU Strix（xrt-smi 2.21.0 / driver 32.0.20102.3930，单实例）

> 纪律：本文件写在任何 512²/1024² big-M 任务与任何 stream-attn 多块重编之前。
> 所有预测均为**可证伪的数字**；实测后逐条对账，错了就写错在哪。

## 0. 口径

- 一切以 512²/1024²/2048² 计；小图（128²/480p）只作为**既有质量档的参照**，不作为本战役的性能结论。
- 墙钟 = loop_s / step wall；ERT state、看门狗、device_s 单列。state=COMPLETED 才算数。
- 51.3 TOPS 固定门在每次改形态前后各跑一次（xrt-smi validate -r gemm）。
- 产品默认零变化：所有新路径在 ZIMAGE_BIGM=1 / 新模块门后；ZIMAGE_BIGM 默认 0。

## 1. 战线 A：big-M 整形状 @ 大图

### 1.1 形态假设

- 512²：SEQ=1056=11×96。整形状 FFN/投影各用**一次 dispatch**覆盖全部 11 个 row-tile；现成 RT=17 镜像装 11 个真 tile + 6 个零 tile（权重只读一遍；零 tile 的行是行局部零，不影响真行）。
- 1024²：SEQ=4128=43×96。按 17 tile 一段切成 3 段（17+17+9，最后一段 8 个零 tile），每段一次 dispatch；权重每段读一遍（3 遍/层，仍比 43 遍少 14×）。
- attention 仍是 host f32 torch（本战线不动）；ZIMAGE_CHUNK_PIPE=0 走原 block_step，FFN 由 BIGM 分支替换，投影由新 bigm_attention 替换。

### 1.2 512² 预测（基线 job 20261030-221051-99ea8a：222.29 s 总，22.22 s/步，8 步）

| 段 | 基线 s/步 | big-M 预测 s/步 | 依据 |
|---|---:|---:|---|
| FFN device | 11.58 | **3.8–4.3** | RT=17 实测 88.55+39.25=127.75 ms/层 ×30；a_pack/host 另计 |
| 投影 device | 6.39 | **2.4–3.0** | bigm_proj RT=17 实测 19.8 ms/次 ×4×30；a_pack 另计 |
| host attention | 2.69 | 2.69 | 不变 |
| 其余 | 1.56 | 1.56 | 不变 |
| **step wall** | **22.22** | **11.5–14.0** | 加 host pack 余量 |
| **8 步 loop** | 177.76 | **92–112** | |
| **整任务（含 startup+VAE ≈45 s）** | 222.29 | **137–157** | |

任务书给的目标预测是 **FFN 92.6→~18 s、proj 51→~13 s、总 222→~110 s**（即 step≈8 s）。
本预测**比任务书保守**：按已实测的 RT=17 dispatch 时间，FFN 30.6 s / proj 19 s 已是下界，
除非 big-M 在大 M 上的权重流比 480p 更快，否则「总 110 s」需要 step 8 s，缺口在 host pack/a_pack。
**先记在这里，实测后按实测报。**

### 1.3 1024² 预测（基线 job 20261030-221644-f60a3b：100.78 s/步，1 步，VAE off）

| 段 | 基线 s/步 | big-M 预测 s/步 |
|---|---:|---:|
| FFN device | 48.68 | **11.5–13.5**（3 段×127.75 ms×30） |
| 投影 device | 24.73 | **7.1–9.0**（3 段×79.2 ms×30） |
| host attention | 37.93 | 37.93 |
| 其余（staging/pack/refiner） | ~27.4 | ~27.4（big-M 只换 FFN/投影） |
| **step wall** | **100.78** | **84–88** |

→ 1024² 的 big-M 只值 ~13–17 %：attention 已是 38 %，FFN/投影的权重流被 3 段摊薄后仍不是唯一墙。
这与 BOTTLENECK-MAP 的「1024² 三者同量级」一致。

### 1.4 2048² 预测（不实跑，仅路线图）

- 171 tile；FFN 11 段×127.75 ms×30 = 42 s/步（vs 外推 184 s），投影 11 段×79.2×30 = 26 s/步（vs 99 s）。
- attention host f32 外推 606 s/步仍是绝对主项；big-M 把 2048² 从 ~980–1030 s/步压到 ~700 s/步，**不改变量级**。
- 2048² 的生死线仍是流式 attention（战线 B）。

### 1.5 质量预测（F-23 声明档）

- 512² big-M latents vs 分块基线（同 seed 7、同 prompt、8 步、VAE on）：
  **corr 0.93–0.97、PSNR 21–25 dB、sorted corr ≥0.9999**。
  依据：480p（1664 token）0.948449/22.66 dB、128²（96 token）0.990743/28.77 dB；512² 在两者之间偏 480p 侧。
- 1024² 1 步：**corr 0.95–0.98、PSNR 23–27 dB**（单步漂移小于 8 步累积）。
- **负对照**：把 big-M 投影的 B 序故意换成 delivered p5 序（或把 a_pack 换成平铺），corr 必须掉到 <0.5；
  若不掉，说明门看不见布局，质量数不作数。

## 2. 战线 B：流式 attention 多块 carry

### 2.1 根因假设（按 STREAM-ATTN §2.4 缩小后的两条）

1. **H1（首选）**：`scf.for` 的 key-block 回边不携带 L1 Buffer 的写——`state` 在第二次迭代读到循环前的值
   （final l=单块和、c=0 的签名）。修法：把 key 块循环在 trace 期展开（Python `range(nk)`），
   即 2/4/8 次静态调用，彻底去掉回边。
2. **H2（后备）**：两个 extern kernel 之间的 L1 Buffer 写可见性本身有问题。修法：把
   softmax step + PV accumulate + finalize 合成**一个** C kernel（每次 key 块一次调用），
   所有 state 读写都在同一个函数体内。

### 2.2 预测（数值 / 时间）

- seq128（2 块，bk64）：修后 **corr ≥0.9995、rel ≤4 %**（单块 seq64 是 0.999778/3.58 %；两块应同档）。
- seq512（8 块，bk64）：**corr ≥0.9990、rel ≤5 %**；run-to-run 逐字节相同；ERT 6/6 COMPLETED。
- 512² 单层（SEQ=1056）对拍 host f32：**corr ≥0.999**。
- 时间：若器件有效吞吐 50 GFLOPS，512² 单层 1.71e10 FLOP → 0.34 s，host 0.09 s ⇒ **慢 3.8×**；
  若 100 GFLOPS → 0.17 s，慢 1.9×；盈亏线仍是 **191 GFLOPS**（BOTTLENECK-MAP）。
  预测：本 spike 的标量/非向量化实现 **< 50 GFLOPS**，512² 上必然慢于 host；它的价值在 2048²。
- DRAM：流式每层 host↔device 只 Q/K/V/O ≈ **32.4 MB**（1056×3840×2×4），
  vs host f32 路径 332.5 MB/层 ⇒ **10.3× 少**；S/P 的 N² 对象为 **0 B**（结构事实）。
  预测实测传输字节 = 32.4 MB/层（±5 %），DRAM 流量（若可得）同量级。

## 3. 对账表（实测后填）

| # | 预测 | 实测 | 判 |
|---|---|---|---|
| A1 | 512² step 11.5–14.0 s | | |
| A2 | 512² 总 137–157 s | | |
| A3 | 512² latents corr 0.93–0.97 | | |
| A4 | 1024² step 84–88 s | | |
| B1 | seq128 corr ≥0.9995 | | |
| B2 | seq512 corr ≥0.9990 | | |
| B3 | 512² 单层 corr ≥0.999 | | |
| B4 | 流式 <50 GFLOPS（512² 慢于 host） | | |
