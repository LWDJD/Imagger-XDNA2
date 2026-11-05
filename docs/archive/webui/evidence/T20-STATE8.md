# T20 — 256×256 的 state-8:根因、修复与实测(8 步 142.7 s)

本文件是 T20 U-1 的交付证据。一句话:**state-8 不是"偶发的环境行为",是一条可复现的设备
栈定律——同一 xclbin 连续 launch 时第二次不产生完成信号;中间插一次别的 xclbin 的 launch
就完全消失。** 256×256 的 3 个 FFN 行块正好构成 2 条这样的边界,所以"每层 2 次重试"。
加上这条 barrier 之后 8 步 **0 次 state-8**,latents 与 T19 **逐字节相同**。

## 0. 结论

| 项 | T19(修复前) | **T20(修复后)** |
|---|---|---|
| 256² 8 步 loop | 3501.3 s / 4593.3 s | **89.44 s / 90.75 s** |
| 256² 8 步 total(含启动/VAE) | 3550.6 s / 4650.8 s | **142.73 s / 124.06 s** |
| state-8 重试 | 200 / 264 | **0 / 0** |
| 每次 FFN 块 launch | 7.0 s 超时 + 10 s 退避 | **40 ms(正常)** |
| latents sha16 | `683050a95f4bd357` | **`683050a95f4bd357`**(逐字节相同) |
| 一步数值门 vs host f32 | corr 0.999964 | **corr 0.999964**(device 字节也相同 `3136c88dc3297d44`) |
| audit | 13/17 + 3/3 | 13/17 + 3/3(未改任何镜像) |

**修成判据(任务书)**:"256² 8 步 **0 次 state-8** 或总墙钟 <10 min" —— 两条都满足
(0 次;142.7 s ≈ 2.4 min)。

## 1. 先纠正一个读法:T19 的"间歇/成簇"其实是"每个块边界一次"

T19 §4 记录"重试是间歇且成簇的:一步 60 次、下一步 0 次",但同一段里还有两条更硬的数字:

* **重试只发生在 ffn 块 dispatch**;
* 每层 = 3 个 FFN 块,**每层正好 2 次重试**(`layers.0` 之前正好 2 行 `ffn RETRY`)。

60 = 30 层 × 2,不是随机数。**3 个块、2 次重试 = 每条"块→块"边界一次**:块 0 前面是
attention(另一个 xclbin),块 1/2 前面是同一个 FFN 镜像。T18 的 `T18_SKIP_ATT=1`
(不做 attention)仍复现,也正好符合:它没有去掉"3 个连续 FFN 块"。

## 2. 决定性实验(standalone;零 build)

工具 `_scratch/image/t20_bench.py`:一个进程、一个镜像、一段 (间隔, 次数) 计划,
逐次记录 ERT 状态、墙钟、**以及超时后命令自己的 `run.state()`**。载荷用
`_scratch/block/t4_out/payload_nh5c_k8`,镜像用交付的 `t11_nh5`(11,088 B insts)。

| 臂 | 形态 | 结果(每次 dispatch 的 state / ms) |
|---|---|---|
| A 基线 | 同一镜像背靠背 4 次 | `4:40 / 8:7017 / 4:1071 / 8:7017` |
| B 间隔 | 0 / 10 / 30 / 100 / 500 / 2000 ms | **全部一样挂**(1/2 次),间隔完全无效 |
| C 中间插 **ip** 镜像 | 40 ms 后插 1.6–1.8 ms 的 ip launch | **0/4 挂**,每次 **39 ms**,Y 字节相同 |
| D 中间插 **proj** 镜像 | 40 ms 后插 5.3–6.5 ms 的投影 launch | **0/3 挂**,每次 39–40 ms |
| E 同一 xclbin 开**第二个 kernel handle** | 交替两个句柄 | "COMPLETED 2–4 ms" —— **假完成** |
| F 超时后的状态 | 轮询 `run.state()` 6×200 ms | **一直是 8**(ERT 自己的看门狗判定,不是我们的 wait 超时) |

三条关键读数:

1. **间隔 0–2000 ms 全部无效** ⇒ 不是主机/时序竞态,是"上下文复用"本身。
2. **插一次别的 xclbin 就干净**(1.6 ms 的 ip 也够)⇒ 换 PDI/上下文会把那个状态清掉。
3. **E 是陷阱**:第二个 kernel handle 让命令"4 ms 完成"——132.7 MB 权重流不可能 4 ms,
   它是**报了一个没做事的完成**。任何"用两个 kernel 句柄绕过去"的方案都是错的。

## 3. 根因(可操作的定律)

> **交付 FFN 镜像 `t11_nh5` 在 kernel 结束时没有把 AIE 阵列留在"可被同一镜像再启动"
> 的状态。紧接其后的第二次 launch 不产生完成信号,ERT 看门狗在 ~7.0 s 把它判成
> `ERT_CMD_STATE_TIMEOUT`(state 8),命令自身状态保持 8。**
> **一次别的 xclbin 的 launch(哪怕 1.6 ms 的 ip)会清掉它;主机侧等待(0–2000 ms)不会。**

对上位事实的机械解释(证据链,不是猜测):

* 为什么"每层正好 2 次":3 个连续 FFN 块 = 2 条同镜像边界。
* 为什么 128×128 / 192×128 从来没中:它们每层只有 **1** 个 FFN 块,块前是 attention
  的 proj/ip dispatch(别的 xclbin),**从来没有同镜像连续 launch**。T11 R6 的
  "1440 dispatch / 0 重试"因此与这条定律完全一致,不是反例。
* 为什么超时后的下一次 launch 要 ~1.0–1.5 s:看门狗超时会复位阵列,复位成本由下一次
  launch 付(这也解释了 T19 的 10 s 退避为什么"看起来"是必要的——它只是把复位等完了)。
* 候选物理机制(未证明,但审计早就记下了):交付 FFN 的 4 个 gate/up 核带 **ACQ CLOBBER**
  (`core_0_3/0_4/0_5/1_2`,每核 3 处,audit 13/17 的基线),即 acquire 的"值"寄存器
  在定义与使用之间可能被 `jl` 覆盖。若某个 lock 因此在核退出时没有被释放,下一次
  同镜像 launch 的 acquire 就会永远等下去——这正是 state-8 的形状;而换 PDI 会复位锁。
  **本轮没有做"去掉 clobber 重建镜像"的实验**,所以这条只作为机制假设记录,修复不依赖它。

**T11 R1 早就记过这条签名**(`t4_bench.py --runs 3` 的第 2 次 dispatch 会挂,
`states [4, 8, 4]`),当时的处置是"单 dispatch 数一律用 3 个独立进程",没有继续
追——因为它只在"同镜像连续 launch"时出现,而当时的产品循环里不存在这个形态。T18 把
FFN 拆成 3 块之后,它第一次进入了产品路径。

## 4. 修复:一条 1.7 ms 的 relaunch barrier

`_scratch/image/t18_run.py`:

* `Dev.barrier(name)`:在两个 FFN 块之间 launch 一次**别的**镜像(默认 `attip`),
  用它在 BO 里已有的操作数、结果丢弃。**不改任何算术**(ip 的输出早已被 drain 走;
  它重算同一份输入得到同一份输出)。代价实测 **1254 ms / 480 次 = 2.6 ms/次**,
  占 8 步 loop 的 **1.4 %**。
* `Loop.ffn`:块 `bi > 0` 之前调一次(FFN_BLOCKS==1 的形状永不调用,
  因此 128×128 / 64×256 / 192×128 / 144×128 的行为逐字节不变)。
  `T20_FFN_BARRIER=0` 是 A/B 开关。
* 首次重试退避 **10 s → 2 s**(`RETRY_SLEEPS = (2, 5, 10, 20, 30)`):既然超时后的
  立即 relaunch 只要 1.0–1.5 s 就能完成,10 s 的等只是在替设备付复位成本。这条只影响
  "万一还挂"时的成本,不影响正确性。

## 5. 256×256 四件套(全部重跑)

| 门 | 结果 | 证据 |
|---|---|---|
| 1 run-to-run 逐字节 | **PASS** `683050a95f4bd357` ×2,`np.array_equal` True,max\|d\| **0.0** | `t20_256x256_latents.npy` / `t20_256x256_b_latents.npy` |
| 2 模型门(一步) | **PASS** vs host f32 **corr 0.999964**,max\|d\| 0.0802 | `t20_256x256_s1_latents.npy` vs `t19host256s1_latents.npy` |
| 3 audit | **PASS** FFN 13/17(交付基线)+ ip 3/3(零镜像改动,缓存 sha256 校验) | `t11_nh5_core_*_win_audit.json` 等 |
| 4 并排图 | **PASS**(沿用 T19 的 device 图;NPU 侧字节与 T19 相同) | `webui/evidence/T19-256x256-host-vs-npu.png` |
| 计时闭合 | **PASS** +0.04 s / +0.04 s | `t20_256x256_timing.json` / `_b_timing.json` |

计时(跑 #1,含 VAE):startup 33.70 + loop 89.44 + tail 0.49 + VAE 17.53+1.44+0.03
+ prepare 0.04 = **142.69 / 142.73 s,未归账 +0.04 s**。
跑 #2(no-vae):32.77 + 90.75 + 0.49 + 0.01 = 124.02 / 124.06 s。

**设备时间账**:dev 52.57 s = ffn 28.99 + att 15.90 + attip 7.68;staging 写 7.0 s;
barrier 1.25 s。8 步 7920 dispatch + 480 barrier。

### 5.1 产品路径(WebUI runner)端到端

同一形状从 **产品入口** 跑一遍(job `20261019-023928-614999`,TE 用 GPU、自动 fast 内存模式、
VAE 常驻):**status done、8 步、7920 dispatch、0 次 state-8**,总计 **155.46 s**
(排队 0 + 文本编码 9.19 + 启动 36.11 + 扩散 90.77 + 器件尾 0.61 + VAE 17.20+1.44+0.03);
PNG sha256 `7178823d93b846d0…` 与 T19 那张**相同**;latents `683050a95f4bd357` 与 T19 逐字节相同。

**同一个 job 在 T19 时点是 59–78 min**,现在是 **2 min 35 s**。

这轮还揪出一个**产品侧计时缺陷**(与 state-8 无关,但被"变快"暴露出来):runner 的
`loop_s` 原来取"driver 各步墙钟之和",而**第 0 步的计时起点在 `startup_done` 之前**
(`cap_build` 发生在第 0 步内部),于是这段被 startup 与 loop **重复计入**。
实测重复量:**128×128 2.24 s、256×256 1.74 s**;T19 的 128² job 恰好以 −0.65 s 擦过 ±1 s 门,
而 256 的第一个 job 直接 **−1.05 s → FAIL**(0 重试、结果完全正确)。
修法(**不改门限**):扩散桶改成 **区间** `startup_done → 最后一行 step`,driver 自己的
步墙钟和与重复量另行记录(`loop_steps_sum_s` / `loop_overlap_s`),器件尾取残差。
复测:256×256 `+0.08 s PASS`,128×128 `+0.08 s PASS`(同一修法,job
`20261019-024206-00bac8` latents 仍为 `6004035d0cb97250`)。

## 6. 成本表

| 项 | 每次成本 | 出现次数 | 合计 |
|---|---:|---:|---:|
| FFN 块 dispatch(修复后) | 40 ms | 720 | 28.99 s |
| barrier(ip relaunch) | 2.6 ms | 480 | 1.25 s |
| 旧的 state-8 重试 | 17 s(7+10) | 200–264 | **3400–4490 s** |

## 7. 逐字拒绝 / 环境事实

* R1 **第二 kernel handle 是假完成**,不是修复(4 ms 完成 132.7 MB 的权重流)。已在
  `t20_bench.py --rotate-kernel` 里留档。
* R2 **`run.state()` 在超时后恒为 8** ⇒ state-8 是 ERT 自己的固定看门狗(≈7.0 s),
  不是我们的 `run.wait(TIMEOUT_MS)` 超时;把 T3_TIMEOUT_MS 调大没有用。
* R3 **超时后的下一次 launch 变慢到 1.0–1.5 s**(复位成本),不是"随机变慢"。
* R4 **M=288 单 dispatch 仍然不完成**(T18 的确定性 TIMEOUT),与本修复无关:256² 本来
  就用 96 行块,现在块间有 barrier。
* R5 barrier 用 ip 镜像;若某形状没有 ip 镜像(`has_ip=False`),barrier 自动改用 att。
  当前只有 FFN_BLOCKS>1 的形状会调用,而它们都有 ip。

## 8. 复现

    # 1 定律(standalone,~30 s/臂)
    py310 _scratch/image/t20_bench.py --xclbin _scratch/block/t4_out/t11_nh5.xclbin \
        --insts _scratch/block/t4_out/t11_nh5.bin --io _scratch/block/t4_out/payload_nh5c_k8 \
        --plan 0:4 --json _scratch/block/t4_out/t20_v_base.json --tag v_base
    #   中间插 ip:再加
    #   --alt-xclbin webui/cache/256x256/t18_ip_m16nd288.xclbin \
    #   --alt-insts webui/cache/256x256/t18_ip_m16nd288.bin \
    #   --alt-bo qp=122880,kb=1244160,vb=1244160,outo=122880
    # 2 产品(8 步)
    py312 _scratch/image/t19_run.py --tag t20_256x256 --height 256 --width 256 \
        --steps 8 --seed 0 --pe webui/jobs/20261018-170459-ed2a32/pe_p0.npz \
        --schedule webui/jobs/20261018-170459-ed2a32/schedule.npz \
        --cache-dir webui/cache/256x256 --timeout-ms 7000