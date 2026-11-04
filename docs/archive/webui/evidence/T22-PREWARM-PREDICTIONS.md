# T22 prewarm 拆分单 — 预测（先于实现与任何测量）

锚点：2026-10-20T10:52 起；HEAD 05bd0c9 之后（工作树有他单未提交文件，本单只动
webui/bridge.py、webui/runner.py、webui/config.py）；器件锁 10:56 起由 t22_t2 持有 →
本轮先做纯 CPU 侧，器件 E2E 待锁。

## 0. 拆分口径（先写清楚"什么算纯打包"）

交付 startup 的 25-27 s 是 `--warm`：30 层 × 4 投影 = 120 × 16,588,800 B（1.85 GiB）
从 weights.bfp16（6,913,965,970 B）读 6.9 GB、重编码 2.0 GB，写进一个 uint8 数组。
本次拆成两段：

| 段 | 内容 | 依赖器件？ | 本次处理 |
|---|---|---|---|
| **纯打包** | 读权重 + pack_w64 重编码 → WPACK（1.85 GiB） | **否**（`Loop.w_images` 只用 `Real` 和 pack_w64） | 兄弟进程（无器件上下文）提前做，写文件 |
| **器件启动** | Dev()：device open + 3 xclbin 注册（U1 实测 ~1.5 s；worker 内 0.04 s） | 是 | 留在 job 内，不动 |

消费侧不重新打包：把文件 mmap 成 `T.WPACK`，用槽位表把 `Loop.w_images` 变成 cache hit
（与 2a2b2 器件 worker 的 `w_images_reuse` 同一手法，那条路径已证逐字节）。

## 1. 预测（可证伪）

| # | 预测 | 证伪条件 |
|---|---|---|
| P1 | 纯打包兄弟进程：import+driver 2.7-3.0 s，30 层打包 24-30 s，总 27-33 s | 打包总时长 <20 s 或 >40 s |
| P2 | 消费侧 warm 段（mmap + 30 层 params + cache hit）：**1.0-2.5 s**（即 25.4 s → 1-2.5 s） | >5 s（则收益 <20 s，须查是否真的没打包） |
| P3 | 文件 1,990,656,000 B = 120 × 16,588,800 B；文件 sha256 == 同进程打包 WPACK 的 sha256；120 槽逐字节相等 | 任一槽不等 |
| P4 | mmap 装载不增加私有 commit（job 内 commit 余量不变，±0.1 GB）；同进程打包则 +1.85 GB | mmap 后 commit 下降 >0.2 GB |
| P5 | 打包与 TE 并发：TE 12.9 s → 13.5-16.5 s（**不免费**，+0.5-3.5 s） | TE 不变（则并发确实免费）或 >20 s（则重叠净亏） |
| P6 | 首 job startup（vae=false, forced fast, 器件）：35.7 → **job 模式 21-24 s**（wait≈12 s + 子进程 startup≈9-10 s）；**boot 模式（文件已就绪）9-11 s** | 首 job startup 下降 <10 s |
| P7 | 同 server 第二个 job：startup 9-11 s，prewarm_wait=0 | 第二个 job 仍 wait |
| P8 | latents 9f86aee9…、pe 90514d08… 与交付逐字节；closure ±1 s（wait 记进 startup，device_tail 用子进程自身 startup 算） | 任一破 |
| P9 | 打包进程不打开器件：`_NoDevice` 守卫下 30 层跑通，无 pyxrt.device() | 需要 Dev() 才能打包 |

## 2. 计时账（先说清楚怎么记，避免重叠段重复记账）

* 打包是**兄弟进程**：它的 wall 只在 job 必须**等**它的时候才是 job 的时间。
  * `prewarm_wait_s`：job 在器件段之前阻塞等待的秒数 → **记进 startup_s**（与
    T21 2a2b2 的"冷 prewarm 进 startup"同口径，是真花掉的时间）；
  * `prewarm_pack_s`：打包进程自身 wall → **不进任何闭包桶**（它与 TE/VAE 重叠，
    记进去就重复记账）；只在 job 的 timing 里做信息位；
  * `prewarm_src`：`reused`（本 server 早先建的）/`built`（本 job 触发）/`boot` /`none`。
* `device_tail_s` 必须用**子进程自身**的 startup 算（`device_wall - startup_child - loop`），
  否则 wait 会把 tail 压成负数、闭包变成负偏差（预计 −11 s FAIL）——这是本单的
  第一个已知算术陷阱，先记在这里。

## 3. 止损线（照任务书）

① 逐字节破 → 回退；② 重叠引入竞态/GIL 死结 → 停报；③ >0.5 设计日 → 带数据回来。
