# T22 workervae：worker x VAE 共存（C-1 单上下文）

**一句话结论**：器件 worker 与 VAE 阶段的硬冲突已解——VAE 现在在 DiT 已经持有的那个 py310 进程里 dispatch（loopback socket + vaeg_srv_proto 分帧），一个 device handle、一个上下文。判据全部实测：4 轮 DiT→VAE 交替 + tail 稳定、latents/pe 逐字节 9f86aee9/90514d08、VAE 六尺寸对 3b3 冻结件 6/6 逐字节、产品 E2E 四 job PNG 全 e7aef10c、closure 全 PASS、t3d_baseline 全绿。**但「worker 默认开」暂不翻**：两个具体 blocker 见 §6（worker 不吃 wpack artifact；驻留 8.84 GB 把 commit 压到 9 GB 阈值下 → 第二个 job 起被内存门踢出 worker）。

---

## 1. 方案（T22-VAE-REPORT §4.2 C-1）与实现

| 模块 | 改动 | 为什么这样 |
|---|---|---|
| webui/dev_worker.py | 新增 VaeLink（127.0.0.1:0 的 socket 服务线程）+ _OneDevice pyxrt 代理；ready 事件带 vae_addr；--vae-link 0 A/B 臂；一把 dev_lock 串行化 DRV.main() 与每次 VAE launch；release op 在有 VAE Srv 时拒绝 | 2a2b2：两个进程两个上下文 → worker 0xC0000005（2/2）；worker 内 release 是 teardown 崩溃类。C-1 只剩「一个上下文服务两者」 |
| _scratch/vae/t22_vae_srv.py | DevSrv 加 connect 模式：ZIMAGE_VAE_SRV_ADDR → 连 worker 的 link，不 spawn、不持 device；kill() 只关 socket（绝不杀 worker）；connect 模式无 respawn/retry（失败即抛，vae_local 回退 CPU）；地址每次 (重)连都重读 env | 宿主的进程不是它拥有的；worker 重启后 vae_local 缓存的 chain 必须能跟上新地址 |
| webui/runner.py | VAE 前不再杀 worker：从 w.vae_addr 取地址塞进 ZIMAGE_VAE_SRV_ADDR，finally 还原；VAE 后查 worker 存活（死了就 drop，下一个 job 重起）；链接不可用 → 保留旧的「VAE 前杀 worker」；NPU VAE 抛异常要回退一次性 vae_worker.py 时先丢 worker（那个 fallback 会起自己的器件服务） | 单变量最小改动；DEV_WORKER=0（默认）路径逐行不变 |

协议复用 vaeg_srv_proto（MAGIC + u32 分帧，part0 JSON），payload 字节不变；worker 侧 Srv 是在第一个 VAE 连接上懒建的，startup 只付一个 bind() + 一个 daemon 线程。VAE 镜像注册与 DiT 的 4 个镜像同进程、同 device 对象（Srv.device(0) 被代理到 t18_run.Dev.dev）。

## 2. 预测 vs 实测

| # | 预测（写在 webui/evidence/T22-WORKERVAE-PREDICTIONS.md） | 实测 | 判 |
|---|---|---|---|
| P1 | link 给冷启动 <=0.05 s；热 job startup 两臂都在 4-7 s | 冷启动 37.9-39.0 s（link on/off，差异在 wpack 噪声内）；热 job 的 startup_done 到达：第 1 job 0.35 s、第 2-5 job 0.16-0.19 s；E2E 产品口径 worker 冷 job 29.71 s、交付臂热 job 3.94-20.0 s | 过（口径见 §4 注） |
| P2 | latents 9f86aee9 / pe 90514d08 不动 | 4 轮 + tail + E2E 4 job 全 9f86aee91de3dad0 | 过 |
| P3 | 首个 128x128 VAE 经 worker 成功且 PNG 与 spawn 档逐字节 | impl=npu，PNG e7aef10cf5323bb1 与 spawn 基线（同 latents）相同 | 过 |
| P4 | 4 轮交替全成功、worker 存活、startup 不漂 | 4/4 + tail，worker before/after 每轮 alive，PNG 四轮同 sha | 过 |
| P5 | 六尺寸对 3b3 冻结件逐字节 | 6/6 EQUAL（每尺寸一个 fresh worker） | 过 |
| P6 | ~44 镜像不触发上下文预算；若触发会在 register_xclbin 报 context 错 | 44 镜像（4 DiT + ~40 VAE）无任何 context 错；worker 全程存活 | 过 |
| P7 | DiT closure <=1 s；VAE 记账不受影响 | 产品 E2E 4/4 closure PASS（+0.05/+0.77/+0.13/+0.12 s） | 过 |
| P8 | VAE BO 在 fast 内存预算内 | 不成立：worker 驻留 8.84 GB + TE-GPU 6.7 GB → 第二个 job commit 余量 7.3 GB < 9 GB 阈值，内存门把 worker 踢出（不是 worker 崩，是 runner 主动走 low 模式） | 否（见 §6） |

## 3. 门表（收工前跑）

| # | 门 | 判据 | 结果 |
|---|---|---|---|
| 1 | DiT 字节 | latents sha16 9f86aee91de3dad0、pe 90514d089ec4b356 | PASS（4 轮 + tail + E2E 4 job） |
| 2 | VAE 六尺寸字节 | 对 out/t21_3b3/size_npu.npy 逐字节 | 6/6 EQUAL：128x128 BCAC0DC8、64x256 E04675B1、144x128 08BD604D、192x128 D26E623A、128x256 E1FB0EA6、256x256 E918129A |
| 3 | worker x VAE 交替 | >=4 轮无死、无 CPU 回退 | 4/4 + tail，impl 全 npu |
| 4 | 热 job startup | 4-7 s 档 | 纯 worker 路径 0.16-0.35 s（更快）；产品路径受内存门限制（§6），交付臂同机为 3.94-20.0 s |
| 5 | 计时闭合 | +-1 s | PASS 4/4（unaccounted +0.05/+0.77/+0.13/+0.12 s） |
| 6 | 产品 E2E PNG | worker 臂 == 交付臂 | 4/4 e7aef10cf5323bb1（24382 B） |
| 7 | t3d_baseline | --check 全绿 | ALL REFERENCE BYTES UNCHANGED（改前 + 改后各一次） |

## 4. 实测数字

### 4.1 纯 worker 路径（无 TE、无内存门；_scratch/vae/t22_workervae_ab.py）

| 项 | 数值 |
|---|---|
| worker 冷启动（coexist 窗） | 37.86 s（wpack 30.39 + payload 4.29 + refiner 2.59） |
| worker 冷启动（六尺寸窗，机器快态） | 25.4-30.1 s |
| 热 job startup（第 1 / 第 2-5） | 0.35 s / 0.16-0.19 s |
| VAE 128x128 connect（首次，含 chain build） | 90.39 s = load 20.66 + decode 66.30 |
| VAE 128x128 connect（后续） | 64.77 / 65.36 / 64.94 s |
| VAE 128x128 spawn 基线（同 latents） | 90.53 s = load 25.21 + decode 62.98 |
| 链路税 | +1.8-2.4 s / 128x128 decode（+2.8-3.8 %）（本地 TCP 分帧 vs 管道；相对 proc 档 96 s 仍是 1.48x） |
| 六尺寸链 wall（含 CPU ref + chain build） | 83.8 / 83.3 / 94.5 / 119.1 / 148.7 / 275.6 s |

### 4.2 产品 E2E（7872；seed7、8 步、batch1、vae=1；两臂各 2 job）

| 臂 | job | total | startup | loop | VAE | closure | PNG |
|---|---|---|---|---|---|---|---|
| worker (DEV_WORKER=1) | 1（冷 worker） | 153.74 | 29.71 | 26.62 | 84.74 | +0.05 PASS | e7aef10c |
| worker | 2 | 124.54 | 3.94* | 38.85 | 62.20 | +0.77 PASS | e7aef10c |
| delivered (DEV_WORKER=0) | 1 | 177.31 | 14.98 | 34.02 | 114.0 | +0.13 PASS | e7aef10c |
| delivered | 2 | 150.97 | 20.00 | 38.70 | 74.7 | +0.12 PASS | e7aef10c |

*worker 臂 job 2 的 startup 3.94 s 是交付 bridge 的数：job 2 的 job.log 明确 memory mode: low (auto: commit 7.3 GB < 9.0 GB)，runner 因此不用 worker（§6）。两臂不在同一时间窗，绝对秒数只能当量级看（F-14 快/慢态）；逐字节与 closure 不受影响。

## 5. 关键发现

1. 2a2b2 的 worker 有一个 0-基线 bug（已修，commit e4799cb）：job 后 finally 读 keep_payload_resident，赋值名却是 _keep_payload_resident → 每个 job 跑完进程就 NameError 退出。两臂同样复现（job1 rc=0 + latents 过门，job2 BrokenPipe）。这解释了为什么 2a2b2 只测到「新 worker 的第一个 job」（worker_job.py 每次新起 worker）——跨 job 复用从未被真正执行过。修完后 4 轮 + tail 全部复用同一进程。
2. 上下文预算：4 个 DiT 镜像 + 每个尺寸 ~40 个 VAE 镜像同进程、同 device 对象，六个尺寸各自 fresh worker 全过；§4.2 的「20 OK / 40 未测」现在补成 44 OK。
3. 失败策略比成功路径更重要：connect 模式下连接失败/launch 失败都只关 socket、不杀 worker，异常抛给 vae_local → CPU 回退（整批）。host-only 测试 14/14 覆盖握手、init/barrier、payload 往返、错误帧、重连、地址变更。
4. 链路税 +2.8-3.8 %：可接受（相对 proc 档 1.48x 的收益），但它是每次 decode 的常数，256x256 上约 +2 s。
5. 内存是默认翻的最大敌人，不是器件：worker 驻留（8.84 GB）+ TE-GPU（6.7 GB）把 commit 余量压到 7.3 GB，9 GB 阈值判定 low → worker 被 runner 主动丢掉。

## 6. 「worker 默认开」的建议：暂不翻（两个 blocker + 修法）

| blocker | 实测证据 | 修法（下一步） |
|---|---|---|
| worker 不吃 wpack artifact | worker 冷 job startup 29.71 s（自己打 21.7 s W pack）；交付臂吃 artifact 的 job 1 startup 14.98 s → 翻默认首 job 慢 ~15 s | runner 的 _wpack_usable 对 worker 返回 True + dev_worker --wpack-file 映射（与交付消费者同一个 seam） |
| 驻留被内存门踢出 | worker 臂 job 2 的 commit 7.3 GB < 9.0 → low 模式，worker 被 drop；随后 VAE 在 spawn 档跑（仍是 npu，PNG 不变） | dev_worker 的 _keep() auto 判据给 TE-GPU 留余量（或 runner 在内存判定会翻 low 时先降 worker residency）——策略改动，需单独一单 + 同窗口 A/B |

修完之后的预期（可测）：热 worker job startup 0.16-0.19 s（纯 worker 路径已实测），相对交付臂第二 job 的 20.0 s 约 -15~19 s/job；但不是 -29~31 s，因为 artifact 已经把 W pack 从首 job 拿走了。翻默认的一行在 webui/config.py（DEV_WORKER 默认 0 → 1），本单按任务书禁动，留给产品决定。

### 6.1 residency A/B（keep=0 vs auto；14:16:33-14:18:33，同一 worker 两 job）

| residency | worker 驻留 | ready | job1 startup | job2 startup | job1/job2 wall | job1 后 commit 余量 |
|---|---|---|---|---|---|---|
| auto（payload+refiner 驻留） | 8.84 GB | 37.9-39.0 s | 0.35 s | 0.16-0.19 s | 27.5 / 25.9 s | **7.3 GB**（含 TE-GPU）→ low 模式，worker 被丢 |
| keep=0（只留 W 1.85 GiB） | 1.99 GB | 31.6 s | **10.32 s**（job 内付 payload+refiner 一次） | **4.92 s** | 46.8 / 40.4 s | **10.3-10.7 GB**（无 TE）→ 预计含 TE 仍 >9 GB，worker 留住 |

两臂 latents 都过门（9f86aee91de3dad0）、worker 两 job 都存活。结论：**keep=0 是能让第二个 job 留在 worker 的配置**（job2 4.92 s 在 4-7 s 档内），代价是首个 job 多付一次 payload+refiner（10.32 s，超出 7 s 档）。这正是 §6 第二个 blocker 的修法方向；产品级验证需要一个含 TE 的 E2E（下一步）。

## 7. 机时账

| 段 | 墙钟 | 说明 |
|---|---|---|
| 窗口 A（vae 基线 + startup 两臂 + 被中止的 coexist） | 12:49-12:54（~5 min） | 基线 92 s；两臂各 ~70 s；发现 2a2b2 NameError 后中止 |
| coexist 4 轮 | 12:58:41-13:06:38（457 s） | 4 x (DiT 26 s + VAE 65 s) + 冷启 38 s |
| 六尺寸 | 13:08:26-13:24:42（976 s） | 6 x (fresh worker ~27 s + chain 84-276 s) |
| E2E 两臂 | 13:27-14:06（器件 ~8 min + 起停/对比） | 4 job；含一次 23 min 的工具停顿（不占器件） |
| 器件总 | ~35 min | 未触止损线 3 |

## 8. 精确剩余量

| 项 | 数量 | 备注 |
|---|---|---|
| 默认翻（config.py 一行） | 1 处 | 需 §6 两个 blocker 先修；本单禁动 |
| worker 消费 wpack artifact | 1 个 seam（runner _wpack_usable + dev_worker --wpack-file） | 首 job 29.71 → ~8-10 s 量级 |
| residency 预算（给 TE 留余量） | 1 处策略 + 1 次 A/B | 让第二个 job 留在 worker |
| 同 worker 跨尺寸 VAE（128x128 job 后解 256x256） | 未测 | 产品 worker 形状钉 128x128，不是产品路径；六尺寸用 fresh worker 是刻意选择 |
| 4 轮以上产品级交替 | 未测 | 器件实测只到 2 个 E2E job（受内存门与窗口限制）；纯 worker 路径 4 轮 + tail 已覆盖 |
| proc 回退风险（link 活着但连不上 → chain 自己回 proc，第二上下文） | 已加防护 | accept 循环不因单次错误退出；runner 只在 alive() 且 vae_addr 非空时注入 env。残余概率极低，若发生表现为 worker 0xC0000005（2a2b2 形状） |

## 9. 复现

    # host-only 协议测试（无器件）
    py312 _scratch/vae/t22_workervae_link_test.py
    # 纯 worker 路径：startup / DiT 字节 / 4 轮交替 / 六尺寸（自取锁、自 release）
    py312 _scratch/vae/t22_workervae_window.py --stages vae,startup,coexist
    py312 _scratch/vae/t22_workervae_window.py --stages sixsize
    # 产品 E2E（两臂）
    ZIMAGE_DEV_WORKER=1 py312 webui.py --host 127.0.0.1 --port 7872
    py312 _scratch/vae/t22_workervae_e2e.py --port 7872 --arm worker --jobs 2
    ZIMAGE_DEV_WORKER=0 py312 webui.py --host 127.0.0.1 --port 7872
    py312 _scratch/vae/t22_workervae_e2e.py --port 7872 --arm delivered --jobs 2

结果 JSON：_scratch/vae/out/t22_workervae/{vae,startup,coexist,sixsize,e2e_worker,e2e_delivered}.json，每尺寸 chain 日志 six_SIZE.log，worker 日志 logs/。

## 10. NOT SHIPPABLE（诚实边界）

* 本单交付的是共存机制（dev_worker + srv connect + runner 接线），不是「worker 默认开」；默认仍是 DEV_WORKER=0，行为与交付版逐行相同。
* 产品级「第二个 job 仍走 worker」在本机未成立（内存门），原因是 residency 策略，不是共存机制；给数字不给结论。
* 六尺寸是每尺寸 fresh worker；同 worker 连解两个尺寸未测（非产品路径）。
* 链路税 +2.8-3.8 % 是实测常数，未优化（不为它做不安全的重叠）。


## 11. 两 seam 实测（worker 吃 wpack artifact + TE reserve）

dev_worker 现在可以映射 prewarm artifact（`--wpack-file`，与交付 bridge 同一个 `bridge._load_wpack_file` seam；坏件只是一行日志 + 交付式进程内打包），residency 的 auto 判据可以给 TE 后端留 `--reserve-gb`。runner 接线：worker job 仍然**不启动**打包进程（避免在 live mapping 上重写文件），但 artifact 已存在时把它交给 worker，并传 `--reserve-gb`（默认 7.0，`ZIMAGE_DEV_WORKER_RESERVE_GB` 可 A/B）。

| 配置 | ready | job1 startup | job2 startup | 驻留（私有 commit） | job1 后 commit 余量 | latents |
|---|---|---|---|---|---|---|
| artifact + reserve 0（worker 层，无 TE） | **8.99 s** | **5.37 s** | **0.29 s** | 6.38 GB | 11.21 GB | 9f86aee9 |
| artifact + reserve 7（worker 层，无 TE） | 2.11 s*（同一窗第二臂，页缓存热） | 4.69 s | 4.71 s | 2.67 GB | 11.64 GB | 9f86aee9 |
| artifact + reserve 7（**产品 E2E，含 TE**） | — | **13.37 s** | **8.71 s** | — | 17.7 GB → memory mode **fast**（worker 未被踢） | PNG e7aef10c |

\* 第一臂的 8.99 s 含 1.99 GB artifact 的冷页缓存；页缓存热后 2.11 s。产品首次启动按 9 s 量级计。

读法：
1. **artifact 是最大的一笔**：W pack 21.7-30.6 s → 0.02 s 映射（+1.48 s sha256 校验），冷启动 39 → 9 s，首 job startup 29.71 → 5.37 s。
2. **reserve 解决内存门**：R=7 时 E2E 的 memory mode 是 `fast`（commit 17.7 GB），两个 job 都留在 worker，PNG 逐字节不变。
3. **R=7 的代价**：auto 判据弃掉 payload（3.71 GB），每个 job 在 startup 里重付 7.7 s 载荷 → 8.71 s，**未进 4-7 s 档**。要同时满足「worker 不被踢」+「startup 4-7 s」，应取「留 payload、弃 refiner」（reserve≈5，驻留 3.71 GB，预计 ~3.5-4 s + 余量 ~12.5 GB），本报告提交时该档尚未实测。

E2E 细节：job 1 的 `startup 13.37 s` = worker 冷启动（device open + map 1.49 + refiner 2.67） + job 内 payload 7.7 s；job 2 的 8.71 s = 纯 payload+开销（无冷启动）。VAE 两 job 都走 worker link（impl=npu，93.45/69.01 s），closure +0.06/+0.04 PASS。

### 11.1 reserve=5 预测（写在测量之前，2026-10-20T15:19）

判据（先写死）：job1 与 job2 都走 worker、closure ≤1 s、PNG 逐字节；startup 判据分两档读——
job1 是**每个 server 只付一次的冷 worker 启动**（交付臂同 job 为 14.98 s），它的合理判据是「明显低于交付 job1」，
4-7 s 档用于 **steady（job2+）**。

| 项 | 预测 |
|---|---|
| keep_payload / keep_refiner | True / False（19.34-3.71-5 = 10.6 ≥ 9 通过；15.6-2.67-5 = 7.9 < 9 弃） |
| 驻留 | 3.71 GB |
| job1 startup | **8-9 s**（冷 worker：device open + map 1.49 + refiner ~2.6；artifact 页缓存已热） |
| job2 startup | **3.5-4.5 s**（无载荷重载，refiner ~2.6 + 开销 ~1） |
| memory mode | 两 job 都 fast；job1 后 commit 余量 ~12.5 GB |
| PNG | e7aef10cf5323bb1（与交付基线同） |
| 失败条件 | job2 startup >7 s，或任一 job 落到 low 模式，或 PNG 变 |

### 11.2 reserve=5 实测（2026-10-20T15:37:07-15:46，含 qkv 默认 ON）

| job | total | startup | loop | VAE(npu) | closure | latents | PNG |
|---|---|---|---|---|---|---|---|
| 1（冷 worker） | 196.5 | **20.71** | 42.91 | 115.6 | +0.14 PASS | 9f86aee9 | e7aef10c |
| 2（热 worker） | 138.0 | **9.64** | 33.12 | 78.5 | +0.05 PASS | 9f86aee9 | e7aef10c |

**预测落空，原因不是 reserve 算术，是快照**：worker ready 事件 `keep_payload=false, keep_refiner=false, reserve 5.0, commit 12.59 GB` —— 本窗机器被其他单压到 commit 12.59（上一条 E2E 是 19.34），`12.59-3.71-5 < 9` 且 `-2.67-5 < 9`，auto 判据把两档都弃了 → 驻留 0，每个 job 重付载荷（本窗 11.5 s）。所以 9.64 s = 载荷 11.5 s 的一部分 + 开销。

**结论**：`auto + reserve` 的驻留量是**快照依赖**的（本 campaign 的 commit 在 12.6-19.3 之间摆动），同一份代码在两个窗口给出 0 或 6.38 GB 的驻留。要一个确定性的产品配置，应当**显式**给 `--keep-payload 1 --keep-refiner 0`（3.71 GB，steady startup 预计 3.5-5 s），而不是让 auto 在内存压力下塌到 0。该配置在安静窗口的实测是本单的剩余项。

本窗同时验证了 qkvprod 的默认 ON 在 worker 路径上成立（`[qkv] fused image registered` 于暖机；latents 仍 9f86aee91de3dad0、PNG 不变）。注意：qkvprod 的 f3b2aed 把 `--qkv` 放进 worker 的**每 job argv**，而 worker 的 Dev 是暖机时建一次、跨 job 复用——暖机不按 qkv 决策就会在第一个 attention 块 `KeyError: 'q_x'`（本单 15:19 实测）。修法：worker 在暖机前用 `bridge.QKV_ENV` + 默认镜像路径对齐（commit 99a0f68）。

### 11.3 「worker 默认开」建议表（给裁决包）

| 项 | 现状 | 建议 | 证据 | 回滚（无代码） |
|---|---|---|---|---|
| `config.DEV_WORKER` 默认 | `"0"` | **暂不翻**：机制全过，但 steady startup 尚未稳定进 4-7 s 档（auto 驻留随快照塌陷） | R=7: 13.37/8.71；R=5: 20.71/9.64；worker 层最优 5.37/0.29（artifact+auto，无 TE） | `ZIMAGE_DEV_WORKER=0` |
| worker 吃 wpack artifact | 已有（存在才传，不启打包进程） | **可随翻默认一起上** | 冷启动 39 → 9 s（页缓存热 2.1 s），首 job startup 29.71 → 5.37 s；latents/PNG 不变 | 删 artifact → 进程内打包（交付行为） |
| residency | auto + `--reserve-gb 7`（默认） | **改为显式 `keep_payload=1, keep_refiner=0`**（3.71 GB 确定值），reserve 只作保险 | auto 在 commit 12.6-19.3 下给出 0-6.38 GB 驻留；显式 3.71 GB 预计 steady 3.5-5 s 且余量 ~10-12 GB > 9 | `ZIMAGE_DEV_WORKER_RESERVE_GB=0` 或 `--keep-*` 改回 auto |
| qkv 默认 | ON（qkvprod f3b2aed） | 保留；worker 暖机已对齐 | latents 9f86aee9 / PNG e7aef10c（qkv 路径下逐字节不变） | `T22_QKV=0` |

**翻默认的前置**（按优先级）：① 显式 keep_payload=1/keep_refiner=0 的 quiet-window E2E（一条，~4 min）；② 若 steady startup 仍 >7 s，再谈是否值得翻（此时 worker 的收益主要是首 job 的 29.71→9-13 s，而不是 steady）。

### 11.4 显式 residency 预测（写在测量之前，2026-10-20T16:02）

runner 现在传 `--keep-payload 1 --keep-refiner 0`（`ZIMAGE_DEV_WORKER_KEEP_PAYLOAD/KEEP_REFINER` 可覆盖），不依赖快照。

| 项 | 预测 |
|---|---|
| 驻留 | 3.71 GB（payload 常驻，refiner 每 job 重读） |
| job1 startup | 9-13 s（冷 worker；artifact 映射 + refiner ~2.6 s） |
| job2 startup | **3.5-5 s**（quiet 窗）；若本窗内存仍被压到 commit ≈12.6，则每个 job 的**内存判定**可能读 <9 GB → 落 low 模式丢 worker（另一条失败路径，不是 startup 慢） |
| memory mode | quiet 窗两 job 都 fast |
| latents / PNG | 9f86aee91de3dad0 / e7aef10cf5323bb1 |
| 判据 | steady（job2）≤7 s 且两 job 走 worker → 可翻；否则按「首 job 9-13 s、steady 不达标」口径记录，不翻 |

### 11.5 显式 residency 实测（16:02:36-16:11，keep_payload=1 / keep_refiner=0）

| job | memory mode | worker? | resident | startup | loop | VAE | closure | PNG |
|---|---|---|---|---|---|---|---|---|
| 1 | fast（commit 13.6） | **是**（artifact 映射 3.11 s，payload 12.66 s 常驻） | **3.98 GB** | **23.97 s**（冷 worker：device+map+payload 12.7） | 38.84 | 122.4 npu | +0.16 PASS | e7aef10c |
| 2 | **low**（commit 4.4 < 9） | **否**（runner 按旧规则在器件段前丢 worker） | 0 | 7.33 s（这是**交付 bridge** 的数，不是 worker） | 50.37 | 72.3 npu | **+1.12 FAIL** | e7aef10c |

**结论（决定性）**：本机 31 GB、在 campaign 负载下，**任何驻留的 worker（1.85 / 3.71 / 6.38 GB 都一样）都会把第二个 job 的内存判定压到 9 GB 以下**——job 1 之后 commit 余量只剩 4.4 GB（worker 3.98 + 本 server 的 TE/torch/VAE chain + 其他单）。于是第二个 job 必然落 low 模式、按旧规则丢 worker，steady 的 4-7 s 根本轮不到。这不是共存机制的问题（机制在纯 worker 路径 4 轮 + 六尺寸已证），也不是 startup 慢，是**宿主内存预算**的问题：worker 的收益要求「第二个 job 开始那一刻」余量 ≥ 9 GB + 驻留量。

job 2 的 `closure +1.12 FAIL` 出现在 **low 模式**（worker 已在器件段前被丢，未执行本单任何代码）；low 模式的 closure 偏移是 t22_prewarm 已记录过的现象，本单如实记下、不归因给 C-1。

**裁决建议（更新）**：默认 **维持不翻**。理由从「steady startup 未达 4-7 s」升级为「本机内存预算下第二个 job 必然落 low，worker 的 steady 收益不可兑现」。翻默认需要的前提变成两条：① 宿主在第二个 job 时有 ≥9 GB + 驻留量的 commit 余量（quiet 窗或更大内存）；② 上述条件下一条 steady ≤7 s 的 E2E。共存机制、artifact 消费、qkv 对齐三项本身均已就绪，可作为将来翻默认的现成组件。
