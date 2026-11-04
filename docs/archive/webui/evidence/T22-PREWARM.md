# T22 prewarm 拆分单：把 25 s 的 --warm 纯打包预热从器件段里拿出来

**一句话结论**：首 job startup 里 25-27 s 的 `--warm`（30 层 × 4 投影 = 120 × 16,588,800 B =
1.85 GiB）是**纯 CPU** 工作（`Loop.w_images` 只读 weights.bfp16、调 pack_w64，从不碰器件），
本次把它拆成**兄弟进程预打包**（`bridge.py --pack-prewarm`）+ **消费端映射**（`--wpack-file`），
与文本编码/VAE 阶段重叠。同 session 器件 E2E（128²，8 步，seed 7，forced fast，vae off）：

* **首 job（boot 模式，产物就绪）：startup 43.43 → 10.96 s（−32.5 s，−75 %）**，total 95.56 → 52.34 s；
* 产物在盘的第 2 个 job：startup 11.39 s（−32.0 s），wait 0；
* 冷建（job 模式、盘上无产物）：startup 37.83 s = 子进程 10.80 + **等待 27.04**（打包 36.4 s 只被
  TE 的 12.3 s 窗口遮住一部分）；
* 五个 job 的 latents **9f86aee9…**、pe **90514d08…** 全同，1440 dispatch / 0 重试，
  closure **+0.037…+0.104 s**（全 PASS）；字节门 **120/120 slot 逐字节 + 拼接 sha == 文件 sha
  == da2f45a6…**。

收益落在**交付默认路径**（one-process bridge），不需要开 2a2b2 那个默认关、且被 VAE-NPU 共存
挡住的常驻器件 worker。

锚点 2026-10-20T10:52 起（含一次机器重启、一次误杀事故）；HEAD 起点 `2ba9c48`；py310 GAIA 3.10.9
做器件侧、py312 做服务侧；本 session 全程有其他单抢器件，所有器件窗口按 F-14.1 取锁。

---

## 1. 预测 vs 实测（预测写在 T22-PREWARM-PREDICTIONS.md，先于任何实现/测量）

| # | 预测 | 实测 | 判 |
|---|---|---|---|
| P1 | 打包进程：30 层 24-30 s | **32.9 / 36.4 / 36.8 / 41.6 / 46.4 s**（五次；本 session 慢态 + 邻居） | ❌ 高估：环境态，见下 |
| P2 | 消费端 warm 段 1.0-2.5 s | **1.34 / 1.88 s** | ✅ |
| P3 | 文件 1,990,656,000 B；120 槽逐字节；文件 sha == 打包 sha | **120/120 相等**，两个 sha 都是 `da2f45a6…` | ✅ |
| P4 | 映射不增加 1.85 GiB 私有 commit | 映射 1.4 ms；打包端峰值从 1.85 GiB → **63 MiB scratch**（更强） | ✅ |
| P5 | 打包与 TE 并发：TE 12.9 → 13.5-16.5 s（+0.5-3.5 s） | E（冷建）TE 12.28 s vs D 10.22 / C1 11.44 → **+0.8~+2.1 s** | ✅ |
| P6 | 首 job：job 模式 21-24 s / boot 模式 9-11 s | **boot 10.96 s** ✅；**job 冷建 37.83 s** ❌（打包 36.4 s 远超预测的 25 s，TE 窗口只遮住 12 s） | 半中 |
| P7 | 同 server 第二 job：startup 9-11 s、wait=0 | **11.39 s，wait=0** | ✅ |
| P8 | latents 9f86aee9…/pe 90514d08…、closure ±1 s | 五个 job 全中；closure +0.037…+0.104 | ✅ |

**预测失败点（诚实记录）**：P1/P6 的 job 模式。我把 2a2b2 的 25.4 s 当成"这台机器也 25 s"，
本 session 同代码同层数要 32.9-46.4 s（邻居占 CPU/IO + 慢态）。结论不变（收益来自"挪出关键
路径"），但**冷建窗口收益依赖打包时长与 TE 窗口的比值**：打包 36 s / TE 12 s → 只能省 5.6 s；
打包 ≤ TE 或产物已就绪 → 省 32 s。这也是为什么默认给了 `auto`（boot + job）。

## 2. 拆分口径：哪一段是"纯打包"，哪一段留在 job 里

| 段 | 内容 | 依赖器件 | 本次处理 |
|---|---|---|---|
| **纯打包** | 读 ~6.9 GB weights.bfp16 → pack_w64 重编码 → 120 张投影图（1.85 GiB） | **否** | 兄弟 py310 进程提前做，写文件；零器件上下文 |
| **器件启动** | `Dev()`：device open + 3 个 xclbin 注册（U1 实测 ~1.5 s） | 是 | 留在 job 内不动 |
| 载荷/refiner | `--cache-all` 3.71 GB + 前 2 个 refiner 块 | 否（但吃内存） | 不动（2a2b2 已定案） |

三点让这件事成立：

1. **纯 CPU 可证伪**：`Loop.w_images` 走 `t3_real.Real`（读文件）+ `pack_w64`（numba 核），
   本次给它一个 `_NoDevice` 守卫——任何属性访问直接 `SystemExit`，所以"打包偷偷开了器件"
   会变成一行日志而不是一个上下文。30 层在守卫下跑通（§4），器件 E2E 五次全部 0 重试也从另一侧
   证明打包进程没有抢器件。
2. **打包是 weights.bfp16 的纯函数**：M（序列长度）不进投影打包，所以**一个 artifact 服务所有
   形状/seed/step/batch**；键只需要 layers/dimk64/W_BYTES + 权重戳。
3. **消费端不重新打包**：把文件 mmap 成 `T.WPACK`，用槽位表把 `Loop.w_images` 变成 cache hit
   ——与 2a2b2 常驻 worker 的 `w_images_reuse` 是同一个 seam（那条路径已证逐字节）。

## 3. 字节等价性：构造性论证 + 实测

**构造性**：artifact 的第 s 槽 = 交付路径 `WPACK[s*W_BYTES:(s+1)*W_BYTES]`，槽序 (layer 0
q,k,v,o, layer 1 …, layer 29) 与 `Loop.w_images` 里 `slot = len(WPACK_SLOTS)` 的递增顺序**逐槽相同**；
打包函数、数值、输入字节都没变。消费端只做两件事：`T.WPACK = memmap`、`Loop.w_images` 返回该槽视图。

**实测（arm=gate，逐层流式，63 MiB scratch）**：每层用**交付代码**打进 scratch，与该层 4 个 slot
的文件区间逐字节比较，再按槽序哈希：

| 项 | 结果 |
|---|---|
| slot 逐字节 | **120 相等 / 0 不等** |
| 拼接 sha256（槽序） | `da2f45a62b0e7e4897ab1f6ef2db16120332f81305c78c9aa3f86696450167c6` |
| 文件 sha256 | 同上（相等） |
| 文件大小 | 1,990,656,000 B = 120 × 16,588,800 B |
| 槽序抽查 | 首 `['layers.0','q',0]`，末 `['layers.29','o',119]` |
| 跨进程确定性 | 独立打包（11:14:33，pack 46.38 s）与 **job 内重建**（12:46:08，pack 36.36 s）的 sha256 都是 `da2f45a6…` —— 打包是确定性的，与谁跑、跑多久无关 |

器件侧的另一半证据：五个 E2E job 的 latents 与 pe 与交付参考逐字节相同（§5），即"器件真的收到
了同一批字节"。

## 4. CPU A/B（同 session、交替、单变量；不占器件）

`_scratch/prewarm/t22_prewarm_ab.py`，两臂只差"W 从哪来"：pack = 交付的进程内打包（`--warm` 做的事），
file = 映射 artifact + 同一个 warm 循环（cache hit）。

| 轮 | 臂 | import | **warm 段** | 映射(+校验) | 进程总 | commit 余量 |
|---|---|---|---|---|---|---|
| 1 | pack（交付） | 1.19 s | **41.61 s** | — | 42.85 s | 20.95 GB |
| 1 | file（+sha256 校验） | 1.55 s | **1.88 s** | 2.46 s（校验 2.45） | 8.70 s | 23.45 GB |
| 2 | file（不校验） | 0.62 s | **1.34 s** | 0.0014 s | 2.00 s | 23.35 GB |
| 2 | pack（交付） | 0.72 s | **36.79 s** | — | 37.54 s | 19.68 GB |

**读法**：warm 段 −37.6 s（96 %）；映射本体免费（1.4 ms），**校验 2.45-2.69 s 是可选成本**
（`ZIMAGE_WPACK_VERIFY`，默认开：它把 1.99 GB 页调入从 loop 挪到 startup，总量不增，只是桶不同；
关掉它 startup 可再少 2.45 s，见 §5 的 10.96 s 里已含这笔）。独立跑一次真打包进程
（`--pack-prewarm`）：pack 46.38 s / 进程 49.03 s（含写 1.99 GB + sha256 2.57 s），与 pack 臂同量级。

### 4.1 三个 MemoryError / 一次失败（诚实记录，已修）

* 第一版打包器把 1.85 GiB 映射成 `w+` 数组 → **第 5 层 "Unable to allocate 56.2 MiB"**
  （当时 commit 余量 5.2 GB，兄弟单在跑器件）。改成一层的 63 MiB scratch + 顺序写文件。
* 第一版 gate 臂每层 new 一个 scratch，`Loop.wcache` 把 30 个视图全钉住（1.9 GiB）→ 同样的
  MemoryError。改成每层 `L.wcache.clear()`。
* 本机 commit 余量在 11:0x-11:1x 摆到 **0.1-5 GB**，打包前两次尝试分别死在第 1/5 层；第三次（11:14）成功。
* **12:41 一次真实缺陷**：B 臂在 job 外删掉了产物，而 server 进程仍记得 `ok=True` → runner 把
  一个不存在的路径交给 bridge，bridge 打印 "unusable … packing in this process as delivered" 并在
  子进程里打包（startup 39.19 s）。job **结果正确**（latents/pe/closure 全对），但 timing 里
  `prewarm_src=reused` 在说谎。修复（commit 931f877）：交接前两次 stat 校验（`_wpack_on_disk`），
  并且"给了文件但子进程没发 `wpack_prewarm`"一律记 `fallback`。修完后 E 臂才拿到干净的冷建数字。

## 5. 器件 E2E（五个 job，128²，8 步，seed 7，batch 1，forced fast，vae=false；同 session 12:35-12:47）

| job | 臂 | startup | 其中子进程 | 其中 wait | src | TE | loop | total | closure | latents | pe | dispatch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 123526 | **A 交付（prewarm=0）** | **43.43** | 43.43 | — | — | 20.05 | 31.25 | 95.56 | +0.104 | 9f86aee9 | 90514d08 | 1440/0 |
| 124051 | C1 产物在盘 | **11.39** | 11.39 | 0 | reused | 11.44 | 30.14 | 53.73 | +0.041 | 9f86aee9 | 90514d08 | 1440/0 |
| 124435 | **D boot 首 job** | **10.96** | 10.96 | 0 | boot | 10.22 | 30.28 | 52.34 | +0.042 | 9f86aee9 | 90514d08 | 1440/0 |
| 124529 | E 冷建（job 模式） | 37.83 | 10.80 | **27.04** | job | 12.28 | 33.73 | 85.01 | +0.049 | 9f86aee9 | 90514d08 | 1440/0 |
| 124145 | B 产物被删（旧代码，实为 fallback） | 39.19 | 39.19 | — | reused(误标) | 10.04 | 27.77 | 77.73 | +0.037 | 9f86aee9 | 90514d08 | 1440/0 |

**首 job**：boot 模式（server 起时打包，产物 36.1 s 后就绪）→ **43.43 → 10.96 s（−32.5 s，−75 %）**；
同一台机器、同一 fixture、同一天。**稳态**：产物在盘 → 11.39 s。**冷建**：37.83 s（−5.6 s，因为
打包 36.4 s 只被 TE 的 12.3 s 遮住）。TE 在冷建 job 上 +0.8~+2.1 s = 并发的真实代价（P5 命中）。

E2E 记录：`_scratch/prewarm/t22_prewarm_e2e.json`；复现：`_scratch/prewarm/_window2.ps1` 与
`_part2b/_part3` 的等价内联命令（见 §10）。

## 6. 计时账：哪个桶拿这笔时间（重叠段不重复记账）

* 打包是**兄弟进程**，它的 wall 只有在 job **必须等它**的时候才是 job 的时间。
* `prewarm_wait_s`（器件段之前阻塞的秒数）→ **记进 startup_s**（与 T21 2a2b2"冷 prewarm 进
  startup"同口径，是真花掉的时间）。E 臂实测：startup 37.83 = 子进程 10.80 + wait 27.04。
* `prewarm_pack_s`（打包进程自身 wall）→ **不进任何闭包桶**（与 TE/VAE 重叠，记进去就是重复
  记账），只作信息位；`prewarm_src` = boot/job/reused/fallback/none。
* **陷阱（实现前就写进预测，实现时避开）**：`device_tail_s` 原 = `device_wall - startup_s - loop`。
  一旦 startup_s 含 wait W 而 device_wall 不含它，tail 被压成负数并 clamp 到 0，闭包变成 **−W**
  （E 臂会是 −27 s FAIL）。改成 `device_wall - startup_child_s - loop`，
  `startup_s = startup_child_s + prewarm_wait_s`。E 臂实测 tail 1.04 s（不是 0）、closure +0.049，
  这条修复被数据验证。

## 7. 门禁表

| # | 门 | 结果 | 证据 |
|---|---|---|---|
| ① | pe/latents 逐字节 | **PASS** | 五个 job 全为 latents `9f86aee91de3…`、pe `90514d089ec4…`（与交付参考/T21 gatefix 相同） |
| ② | seed 复现 | **PASS** | 同一 fixture 五次运行（含两条不同代码路径 A/E）latents 全等 |
| ③ | 批语义 | **未改**（本单不碰 batch 代码）；交付证据沿用 T21 2a2b2 §5③ | — |
| ④ | 计时闭合 ±1 s | **PASS** | +0.104 / +0.041 / +0.042 / +0.049 / +0.037（含 E 的 27 s wait 那条） |
| ⑤ | t3d_baseline --check 前后全绿 | **PASS** | 改动前、改动后各一次 "ALL REFERENCE BYTES UNCHANGED" |
| ⑥ | 打包不碰器件 | **PASS** | `_NoDevice` 守卫下 30 层跑通；五次器件 job 0 重试、无 0xc01e0009 |
| ⑦ | 桶归属如实 | **PASS** | §6；wait 进 startup、pack 不进桶、fallback 不再冒充 reuse |

## 8. 改动文件与 commit

| commit | 内容 | 文件 |
|---|---|---|
| 63f3b96 | `--pack-prewarm`（纯 CPU 打包，63 MiB scratch，meta = 提交记录）+ `--wpack-file`（映射消费，坏文件则交付内联打包） | webui/bridge.py |
| 1d5f1c5 | runner：prepare 后起打包进程（与 TE 重叠）、器件段前 join、wait 只进 startup_s、device_tail 用子进程 startup | webui/runner.py |
| 931f877 | 交接前 stat 校验 + "给了文件没加载" 记 fallback（12:41 的误标缺陷） | webui/runner.py |
| b4cba96 | 预测、A/B 证据、报告、复现脚本 | webui/evidence/T22-PREWARM*.md, _scratch/prewarm/ |
| a038a30 | 器件 E2E 五 job 记录 + 报告 §5 | _scratch/prewarm/t22_prewarm_e2e.json |
| 0d26547 / 5210c80 | 确定性记录（两次打包同 sha）+ 文档转义 | _scratch, evidence |
| （已在 HEAD） | `ZIMAGE_WPACK_PREWARM=job|boot|auto`、`ZIMAGE_WPACK_VERIFY`、artifact 路径 | webui/config.py |

未改：webui/dev_worker.py（禁动，常驻 worker 保持自己的打包）、vae_*、t18_run/t3_real/t8_codec、
任何 xclbin/bin、任何 reference 字节。

## 9. 精确剩余量

| 项 | 状态 |
|---|---|
| 纯打包预热（兄弟进程） | 完成；字节门 120/120 + sha |
| 消费端映射（交付默认路径） | 完成；warm 1.34-1.88 s；startup 10.96-11.39 s |
| 与 TE 重叠 / 冷建等待 | 完成（+0.8~+2.1 s 的并发代价已量化） |
| 器件 E2E 四项判据 | **完成**（§5，五个 job） |
| 常驻 worker 复用同一 artifact | **未做**（dev_worker.py 禁动）：给 worker 加一个 `--wpack-file` 直通即可，约 0.1 设计日；收益是 opt-in 路径再省 ~2 s |
| boot 模式的资源账 | 未量化"server 起后无 job"时那 33-46 s CPU 的代价；默认 `job` 是保守选择，`auto` 需产品侧决定 |
| 多形状 | artifact 与形状无关（只看权重），已按此实现；未在 256²/M=288 实测（本树那条路径还要器件镜像） |
| gate 模式（两遍） | 未跑两遍 gate job；第二遍与第一遍共用产物，预期相同（未测，不计入判据） |

## 10. 复现

```
# 1) 打包（纯 CPU，无器件；~33-46 s，写 1.99 GB）
py310 webui\bridge.py --pack-prewarm --pack-out webui\cache\wpack\wpack_l30_v2.bin \
      --cache-dir webui\cache\128x128 --pack-verify
# 2) 字节门（每层用交付代码打、逐槽比）
py310 _scratch\prewarm\t22_prewarm_ab.py --arm gate      # 120/120, sha da2f45a6…
# 3) CPU A/B
py310 _scratch\prewarm\t22_prewarm_ab.py --arm pack      # 交付的进程内打包
py310 _scratch\prewarm\t22_prewarm_ab.py --arm file --verify
# 4) 器件 E2E（需 F-14.1 锁）：起 server（ZIMAGE_WPACK_PREWARM=job|auto）+ 提交 128²/8 步/seed 7
```
