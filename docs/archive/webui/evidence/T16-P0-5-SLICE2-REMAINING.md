# T16-P0-5 切片 2/3:精确剩余量(未构建,不假装)

> **【更正 2026-10-19】本文件 §1、§6、§7、§8.1 的"M=128 L1 放置墙 / 必须做 wave 设计"结论全部作废,一律以 §9 为准。**
>
> 失败对象 @@BT@@midB_3_cons_buff_0, 2048 B = m*kgu*2 = 16*64*2@@BT@@ 是 **kgu=64** 造成的假象:
> T16 的复现命令漏了 @@BT@@--kgu 8@@BT@@(T11 的交付命令与 @@BT@@t11_ffn_nh5@@BT@@ 的 PARAMS 都是 kgu=8)。
> T18 用交付旋钮重编,源码零改动即过(@@BT@@t18_m128@@BT@@ 18.5 s / 177,179 B、17 核 audit 13/17;
> @@BT@@t18_m288@@BT@@ 21.1 s / bin 36,144 B),**不需要 wave**。192×128 已过四件套;
> T20 补建 128×256、T19/T20 收口 256×256,三者也已 ready。
>
> 下文原文**保留**(逐字不改),仅作历史记录:**不得据此安排工作**。

## 0. 状态

| 切片 | 尺寸 | 结果 |
|---|---|---|
| 1 | 64×256(M=96,host geometry) | **PASS**(四件套,见 T16-P0-5-BUDGET.md §8) |
| 2 | 192×128(M=128) | **未构建**:第一个动作(FFN 镜像)在 aiecc 放置阶段撞 L1 墙(§1) |
| 3 | 256×256(M=288)+ build_shape 收口 | 未开始,依赖切片 2 的镜像与 host 驱动 |

`/api/shapes` 对 192×128 / 256×256 仍是 `ready=false` + 精确 missing 列表;
`POST /api/jobs` 返回 400。**没有假进度条、没有复用 128 的镜像。**

## 1. 实测:FFN M=128 不是"--M 重编译",是 L1 放置墙

> 【更正 2026-10-19】本节的"L1 墙"**作废**(见文首横幅与 §9):2048 B 是 kgu=64 的 midB
> 消费者对象,T16 复现时漏了 @@BT@@--kgu 8@@BT@@。§1 末"FFN 需要一个设计变体"的推论同样作废。

命令(交付的 FFN 设计 + t4_compile 的 M/Kc 旋钮):

    py312 t4_compile.py --design t11_ffn_nh5 --tag t16m128_ffn --M 128 --NH 5

结果(默认 Kc=960):

    error: 'aie.buffer' op could not be placed: buffer "midB_3_cons_buff_0"
    needs 2048 bytes and this tile has no room left for it
    aiecc: edge 'input_with_addresses.mlir' failed

变体 Kc=384(减小 midB 列块)同一错误;Kc=480 先在设计断言
`Kc % kgu == 0` 处拒绝(kgu=64)。每次编译 ~2-3 分钟,无产物。

**读法**:交付 FFN 设计的 compute tile 在 M=96(6 个 row tile)时已经用满
(inBG3 5×9216 B + outS3 2×2048 B + stack 4096 B + static 512 B ≈ 55 KB / 64 KB);
M=128 变成 8 个 row tile,midB 的 2048 B 消费者 buffer 放不下。改 Kc 只动
producer 侧,救不了 consumer 侧。**--M 本身不够;FFN 需要一个设计变体**
(行 tile / double-buffer 方案),attention 两个镜像(M 行)同理,然后才是
perm/packer/SBG/walk 与 host 常量。

### 预算偏离(>30%,按要求带数据)

* 纸面预算(§2)说 M 线性放大对象:128/96 = **1.33×**,loop ≈ 36.9 s/图;
  这部分与实测一致(slice 1 的 M 相同,loop 26.9 s 同量级)。
* 但"6 镜像 = 按 M 重编译"的假设被实测推翻:FFN 在 M=128 **放置失败**。
  构建成本不是 §2 的"对象线性放大",而是**设计变体 + 重新 audit**。
  这是本轮最重要的预算修正。

## 2. 每个新 M 的构建清单(M=128 为例,按依赖顺序)

1. **FFN 镜像**:t11_ffn_nh5 的 M=128 变体。先解决 §1 的 L1 放置(行 tile 数
   8 → 重新分配 midB/outS3/inBG3 的 buffer),再 audit 全 core;
2. **投影镜像 t7_p5**:X,W→Q(K/V/to_out 在同一镜像),M=128 的 tile/walk;
3. **IP 镜像 t7_ip**:QP,KB,VB→O,M=128;
4. **其余投影**(按任务书 6 镜像口径)与 2/3 一起构成 6 个;
5. **perm/packer**:交付的是 96×128 / 96×64;M=128 需要 128×128 / 128×64,
   由 codec 的 perm 生成脚本重算并过 round-trip 门;
6. **host 常量**:t13_run/t3_real 是 M=96 专用(行数 96、image tokens 64、
   cap 32、SBG 偏移、pack/walk、`out[:64]`、unpatchify 尺寸)。切片 1 能只
   patch 接缝,是因为 M 没变;M=128 必须新写一个按 M 参数化的 host 驱动
   (或把这两处变成 seam),这是切片 2 的主要工作量;
7. **每尺寸四件套**:模型门(M=128 vs host f32)、run-to-run 逐字节、audit 全 core、
   首图并排。

预计:1 个 FFN 设计变体(1-2 轮编译 + audit,每轮 3-5 分钟)+ 2 个 attention 变体
+ perm + host 驱动(这四项是大头,按 T13 §13.3 的设计单元口径,不是小时级)。

## 3. 切片 3(M=288)与 build_shape 收口

M=288 的对象是 M=96 的 3×(A 2.21 MB / SBG 6.64 MB / Y 2.21 MB),权重流不变;
纸面 loop ≈ 83 s/图。它依赖切片 2 的 FFN/attention 设计变体与 host 驱动,
之后 `webui.build_shape(H,W)` 才能把"计划 → 逐镜像 build+audit → 产物+manifest
入 cache → 进度流"串起来。本轮的 prebuild 端点已经是对接点:已构建=暖缓存,
未构建=plan;接上 builder 后,进度流(SSE/轮询)才有真实的阶段可报。

## 4. 本轮为什么不硬做

* 交付的 host 驱动是 72 KB 的 M=96 专用程序,M=128 要重写参数化版本;
* FFN 在 M=128 放置失败,必须先做设计变体,不是调参;
* 设备间歇性卡死(0xC0000005),编译后还要跑四件套,时间不在本轮预算内;
* 按纪律,"不半建":没有把 192×128 显示成可用,也没有生成半套镜像。

## 5. 复现(撞墙证据)

    py312 t4_compile.py --design t11_ffn_nh5 --tag t16m128_ffn --M 128 --NH 5
    # -> aie.buffer could not be placed: midB_3_cons_buff_0 needs 2048 bytes
    py312 t4_compile.py --design t11_ffn_nh5 --tag t16m128_ffn_kc384 --M 128 --NH 5 --Kc 384
    # -> same
## 6. NH/Kc 扫描(2026-10-18 追加,给"波次"方向的数据)

在"先试便宜的旋钮"原则下扫了交付设计的 NH/Kc,全部在 aiecc 放置阶段失败:

| 变体 | 结果 | 放不下的 buffer |
|---|---|---|
| M=128, NH=5, Kc=960(交付默认) | 失败 | @@BT@@midB_3_cons_buff_0@@BT@@,2048 B |
| M=128, NH=5, Kc=384 | 失败(同一处) | @@BT@@midB_3_cons_buff_0@@BT@@,2048 B |
| M=128, NH=4, Kc=960 | 失败(换了一处) | @@BT@@C3_4@@BT@@,2048 B |
| M=128, NH=5, Kc=480 | 设计断言拒绝 | @@BT@@Kc % kgu == 0@@BT@@(kgu=64) |

读法:降 NH 只把压力从一个 2048 B buffer 挪到另一个;compute tile 在 M=96 已经
~55/64 KB,M=128 的 C 态/中间 buffer 无论怎么调 NH/Kc 都放不下。**这支持"波次"
方向:把"同时持有的 row-tile 数"压到 ≤4,而不是继续调 NH。**

### 波次方向的成本对账(先算账)

* 权重流:交付设计**本来就按 row-tile 重读权重**(T10:M=96 每图 6×132.7 MB,
  6 = M/16 个 row tile)。M=128 是 8 个 row tile(1.06 GB/图,+33%);
  M=288 是 18 个(2.39 GB/图,3×)。**波次分组不增加权重流量**——每个 row tile
  仍只读一遍权重,只要波内不复读;它只改变 C 态的同时驻留数。
* 真正随波数涨的是 norm 核的重放/visits 和 host 的 A 流分波(每波一个 dispatch
  的边界),这部分要在设计里按 (pass, chunk) 序记账;换序就是换算术,逐字节门会抓。
* 结论:波次是"串行"路线,绕开 T11b 的行并行三条结构墙;它的代价是 norm 重放,
  不是权重。预算上 M=128 的 loop 仍在 ~1.33× M=96 的量级,但**前提是波内权重
  只读一遍**;如果实现成每波重读权重,成本会按波数(2 波/5 波)放大,必须在设计时
  用 B 流的 liveness 明确守住。
## 7. wave 循环的设计读(下一次设计会话的入口)

按批准顺序读了 t11_ffn_nh5.py 的 row-tile 结构与 C 态,并补扫了两个"更便宜的"
旋钮;结论:**wave 是真正的设计迭代,不是几行循环**。

### 7.1 结构(实测代码位置)

* @@BT@@row_tiles = M // m@@BT@@(line 487);C 态按 (rt, n-tile) 组织;
* 设备侧 @@BT@@range_(row_tiles)@@BT@@ 出现在三个 stage:GU(~743)、down(~828/856)、
  X/Y(~902);host 侧按 row tile 分配 tap(981/996/1006/1021);
* 每个 worker 的 C buffer 是 **NH 对**(C3_0..C3_NH),@@BT@@outC3@@BT@@ 深度写死 12,
  @@BT@@inBG@@BT@@ 深度 NH+1;@@BT@@L_G3=15@@BT@@ 时 @@BT@@outC3@@BT@@ 的 12 就不够(见 7.2);
* 权重 B 流按 row tile 重读(T10 口径),wave 分组本身不加权重流量,只要波内
  每个 row tile 只读一遍。

### 7.2 补扫的两个旋钮(同样失败)

| 变体(M=128) | 结果 | 原因 |
|---|---|---|
| G_GU=2, G_G3=4, L_G3=15 | 生成期失败 | @@BT@@outC3@@BT@@ 深度写死 12 < L_G3=15(ValueError)+ IRON Unbalanced Location |
| G_GU=4(交付),NH=4 | aiecc 失败 | @@BT@@C3_4@@BT@@ 2048 B 放不下 |

=> 改 worker 数要连 @@BT@@outC3@@BT@@ 深度一起改,且 IRON 的 join/worker 生成对
G_GU 有结构假设;这不是调参能过的。

### 7.3 建议的最小 wave 改法(下一次)

1. 加 @@BT@@WAVE = 4@@BT@@(每波 row-tile 数,<=4);
2. 三处 @@BT@@range_(row_tiles)@@BT@@ 改成 @@BT@@for wave in range_(ceil(row_tiles/WAVE))@@BT@@ +
   内层 @@BT@@range_(WAVE)@@BT@@(末波越界保护);host tap 同步按 wave 分块;
3. C buffer 仍按 worker 的 NH 对分配(不按 row tile),但要把 **深度/对象数**与
   wave 内的在飞对象数对齐;权重 B 流按 (n-tile, wave) 填一次,波内每个 row tile
   只读一遍(红线);
4. 重算 (pass, chunk) 序;换序必须由逐字节门判定;
5. M=128 编译 + audit;若 L1 仍紧,再按 7.2 的教训同步改 @@BT@@outC3@@BT@@ 深度后
   再降 worker 数。

每次编译 ~2-3 分钟;本轮四个变体(NH5/Kc960、NH5/Kc384、NH4、G_GU2)均无产物,
.prj 已清。下一步是设计会话,不是本轮能收口的量。
## 8. T18 交接:墙表、失败变体与审查清单

T18 按 §7.3 执行,不重查。设计输入 = 本文件 §1(放置墙)、§6(NH/Kc 扫描)、
§7(设计读 + 最小改法)。**不要**再把 NH/Kc/G_GU 当旋钮重扫;四组数据已经证明
它们过不了(或触发 IRON 结构假设)。

### 8.1 墙表(直接引用)

| 变体(M=128) | 结果 | 限制 buffer |
|---|---|---|
| NH=5, Kc=960 | aiecc 放置失败 | @@BT@@midB_3_cons_buff_0@@BT@@ 2048 B |
| NH=5, Kc=384 | 同处失败 | @@BT@@midB_3_cons_buff_0@@BT@@ 2048 B |
| NH=4, Kc=960 | 换一处失败 | @@BT@@C3_4@@BT@@ 2048 B |
| G_GU=2, G_G3=4, L_G3=15 | 生成期失败 | @@BT@@outC3@@BT@@ depth 12 < L_G3 15 + IRON Unbalanced Location |
| Kc=480 | 设计断言 | @@BT@@Kc % kgu == 0@@BT@@ |

### 8.2 审查清单(对应负责人的四点)

1. **B 流红线**:看新增的 wave 结构里 B(inBG/inB3)的 fill/acquire 是否按
   (n-tile, wave) 各一次;每个 row tile 的 A 侧可以重放,但 B 侧不能按 row tile
   再读。核对点:wave 循环内 B 的 prod/cons 计数、host tap 的 per-wave 分块;
   流量账 = row_tiles × 132.7 MB/图(不应乘 wave 数)。
2. **(pass, chunk) 序**:wave 化后 GU 的 pass/chunk 序、down 的 kk' 序必须重算;
   判据是逐字节门(同 seed 两次 + 与 host f32 并排),不是"看起来对"。
   任何换序都会被 @@BT@@gate.pass@@BT@@ 抓住;若差异只在 BFP16 舍入,应落在既有容差内。
3. **outC3 深度 / worker 数**:必须同步改(§7.2 的 ValueError 就是深度 12 < L_G3);
   IRON 对 G_GU 有 join/worker 结构假设,G_GU=2 会 Unbalanced Location。改 worker
   数时先改深度常量,再跑生成期。
4. **四件套 / BUILD REQUIRED**:192×128 在真构建+四件套过之前,@@BT@@/api/shapes@@BT@@
   必须继续返回 @@BT@@ready=false@@BT@@ + missing 列表;不要提前把 192×128 标 ready。
   四件套逐尺寸:模型门(host f32 并排)、run-to-run 逐字节、audit 全 core、首图并排;
   产物 manifest 入 @@BT@@webui/cache/192x128/@@BT@@,manifest 入库到
   @@BT@@webui/shape_manifests/@@BT@@;.prj 用后清。

### 8.3 我这边保持不动的东西

* 切片 1(64×256)已 PASS,不要回退;它的 bridge 接缝(IMG_GRID/unpatchify/--resume)
  是 M 相同情况下的参考实现。
* 128×128 与批量门/计时/常驻 VAE 的现状不动。
* 失败 .prj 已清,磁盘留给 T18;BUILD REQUIRED 应答保留。

## 9. T18 更正(2026-10-18):§1/§6/§7/§8.1 的 "M=128 L1 墙" 是 kgu=64 假象

本文件 §1 的失败对象 **@@BT@@midB_3_cons_buff_0, 2048 B@@BT@@ = m*kgu*2 = 16*64*2**,
即 kgu=64 的 midB 消费者对象。T16 的复现命令
@@BT@@t4_compile.py --design t11_ffn_nh5 --M 128 --NH 5@@BT@@ **没有带 --kgu**,
而 @@BT@@t4_compile.py@@BT@@ 的默认值是 kgu=64;T11 的交付命令是
@@BT@@--kgu 8@@BT@@(t11_ffn_nh5 的 PARAMS 也写 kgu=8)。

T18 用交付旋钮重编:@@BT@@t18_m128@@BT@@ 18.5 s / 177,179 B / bin 15,264 B,
17 核 audit 13/17(与交付同一批 4 个 gate/up 核);@@BT@@t18_m288@@BT@@ 21.1 s,
bin 36,144 B,同样 13/17。**源码零改动,不需要 wave**;因此 §6/§7 的波次设计
不再需要(B 流按 row-tile 一读的红线天然满足)。M=288 的单 dispatch 在设备上
**不完成**(36,144 B insts,standalone 复现 ERT state 8),所以 FFN 改成 96 行块
(t11_nh5 ×3)运行。192×128 已过四件套;细节、证据与 256×256 的卡点见
@@BT@@T18-P0-5-ARBITRARY-M.md@@BT@@。

