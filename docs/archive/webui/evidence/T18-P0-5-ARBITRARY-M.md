# T18 — 任意宽高:M 参数化镜像 + 行块 host 驱动(192×128 PASS)

本文件是 T18 的交付证据。T18 的任务书按 T16 §7.3 的 **wave 设计**派发,但第一步复核
就把那个前提推翻了(§1)。实际做出来的是一条更小的路:**交付设计源码零改动,按 M 编译/
分块运行**;192×128 已过四件套,256×256 的镜像全部构建并审计,但设备运行被一个
偶发 ERT state-8 卡住(§6),按纪律标 BUILD REQUIRED,不冒充可用。

## 0. 结论

| 尺寸 | M | 状态 | 证据 |
|---|---:|---|---|
| 128×128 | 96 | A/B 字节门 PASS | t18_run 与交付驱动同 seed/prompt/8 步 → latents **逐字节相同** sha16 `6004035d0cb97250` |
| **192×128** | **128** | **PASS(四件套)** | run-to-run sha16 `1e8a3ddc1b65de55`;vs host f32 **corr 0.99422 / PSNR 29.86 dB**;image 审计(17 核 + 3 核);并排图 `T18-192x128-host-vs-npu.png` |
| **256×256** | 288 | **PASS(T19 收口)** | run-to-run `683050a95f4bd357`;一步 vs host f32 corr **0.999964**;audit 13/17+3/3;并排图;state-8 重试透明(§6 的"未过"已被 T19 关闭,见 `T19-256x256.md`) |

## 1. T16 的 "M=128 L1 墙" 是 kgu=64 假象(实测更正)

T16 §1/§6/§7/§8.1 记录 M=128 在 aiecc 放置阶段失败:

    aie.buffer could not be placed: buffer "midB_3_cons_buff_0" needs 2048 bytes

**2048 B = m*kgu*2 = 16*64*2,即 kgu=64 的 midB 消费者对象。** T16 的复现命令
`t4_compile.py --design t11_ffn_nh5 --M 128 --NH 5` 没有带 `--kgu`,而
`t4_compile.py` 的默认值是 **kgu=64**;T11 的交付命令是 `--kgu 8`(t11_ffn_nh5 的
PARAMS 也写着 kgu=8)。用交付旋钮重编:

| 变体 | 结果 | xclbin | bin | 核 | audit |
|---|---|---:|---:|---:|---|
| `t18_m128`(--M 128 --NH 5 --kgu 8) | **成功 18.5 s** | 177,179 B | 15,264 B | 17 | 13/17 PASS |
| `t18_m288`(--M 288 --NH 5 --kgu 8) | **成功 21.1 s** | 177,179 B | 36,144 B | 17 | 13/17 PASS |

13/17 与交付件**同一批 4 个 gate/up 核**(core_0_3/0_4/0_5/1_2 的 3 个 ACQ CLOBBER,
T11 R5 既有),没有新红核。**结论:不需要 wave,不需要改 t11_ffn_nh5.py 一行;
源码不动,所以"波内每 row-tile 权重只读一遍"的红线天然满足**(row_tiles 只改运行时
循环次数;每次 dispatch 每 row-tile 仍是 88.5 MB BG + 44.2 MB B3 各一遍)。

## 2. 设计:三类行块,每个都按"行独立"证明

| 阶段 | 镜像 | 行块 | 为什么精确 |
|---|---|---|---|
| 投影 q/k/v/o | 交付 **t7_p5**(m=96) | 96 行/块 | 投影每次调用只算自己的 96 行;k-group 序、C 舍入链都不跨行 |
| ip(QK^T/softmax/PV) | 新编 `t18_ip_m64nd128` / `t18_ip_m16nd288` | 查询块 M=64 / 16;**键 = 全 SEQ** | attention 是稠密的,所以 ND=SEQ;查询行彼此独立,块拼接精确 |
| FFN | `t18_m128`(M=128) / 交付 **t11_nh5**(M=96 块) | 1 块 / 3 块 | FFN 每行独立;第一张图证明 128 单 dispatch 可跑,288 单 dispatch 跑不了(§6) |

ip 镜像的 L1 账(设计与 `t18_attn_compile.py` 都断言):

    M=64,  ND=128: QP 16,384 + KB 18,432 + C 16,384 + stack 6,144 = 57,344 <= 65,536
    M=16,  ND=288: QP  4,096 + KB 41,472 + C  9,216 + stack 6,144 = 60,928 <= 65,536

blocked-layout perm 是 (rows, K) 的函数,由 `t18_perm.py` 从 kernel 自己的
`mm_bfp(...).contract.layouts[2].pack(arange)` 提取;**自门:96×128 / 96×64 与已提交
表逐字节一致**(否则拒绝写新表)。新增 `t18_perm_64x128.json`、`t18_perm_16x128.json`。

## 3. 构建表(全部实测;.prj 已清)

| 镜像 | 编译 | xclbin | insts | 核 | audit |
|---|---|---:|---:|---:|---|
| t18_m128 | 18.5 s | 177,179 B | 15,264 B | 17 | 13/17(同交付基线) |
| t18_m288 | 21.1 s | 177,179 B | 36,144 B | 17 | 13/17(同交付基线);**单 dispatch 设备不完成** |
| t18_ip_m64nd128 | 8.8 s | 13,864 B(win 后) | 19,696 B | 3 | **3/3 PASS** |
| t18_ip_m16nd288 | 13.7 s | 13,896 B(win 后) | 19,696 B | 3 | **3/3 PASS** |
| t7_p5 / t11_nh5 | — | 交付件,字节未变 | | | 交付 audit 不变 |

ip 镜像的数值自门(`t18_ipgate_*.py`,合成 Q/K/V + 量化后 f32 参考):
**corr 0.998995,mean|Δ| 0.0238(参考 absmax 9.87),两个查询块都 ~0.999**。

## 4. host 驱动 `t18_run.py` 与桥

* `t18_run.py` = t13_run 的 M 参数化:**SEQ/BO 尺寸/perm 由 `configure(H,W)` 设**;
  投影 96 行块;ip 查询块;FFN 行块(§2);unpatchify 用 `out[:tokens]`。
* `webui/bridge.py`:按 M 选驱动(M=96 用交付 t13_run,新 M 用 t18_run),
  从缓存 manifest 的 `images` 选 xclbin;新增 `--host-ref` 透传。
* **A/B 字节门(M=96)**:同 seed/prompt/8 步,与生产 128×128 的 latents
  **逐字节相同** sha16 `6004035d0cb97250`——参数化没有动算术。

## 5. 192×128 四件套(真实出图)

| 门 | 结果 | 证据 |
|---|---|---|
| run-to-run 逐字节 | **PASS** sha16 `1e8a3ddc1b65de55`(两次独立进程) | `t18_192x128_latents.npy` / `t18_192x128_b_latents.npy` |
| 模型门 vs host f32 | **PASS**:latent corr **0.99422**;图像 PSNR **29.86 dB**,mean abs 0.0126 | `t18host192_latents.npy` / `t18host192.png` |
| audit 全 core | 新镜像全审(FFN 17 核 13/17 = 交付基线;ip 3/3 PASS) | `*_win_audit.json` |
| 首图并排 | **PASS** | `webui/evidence/T18-192x128-host-vs-npu.png` |

运行数据:2640 dispatches(30 层×8 步×11),**0 重试**,wall **47.07 s**,device 23.44 s
(att 9.90 + attip 1.16 + ffn 12.38),staging 3.11 s。loop 47.1 s vs 纸面 128/96×27.7=36.9 s
的偏差在 host 侧(3× cache-all 写 88.5 MB/层 + 多出来的块)。

## 5.1 驱动默认切换(2026-10-18):所有尺寸都用 t18_run

交付 t13_run 的 `dispatch` 把输出 BO 又追加一次(6 个声明参数 → 7 个 BO)。M=128 的
FFN 镜像在这第 7 个 BO 上**直接 0xC0000005**(standalone 复现:6 args COMPLETED /
7 args 进程死);而且在当前环境里**交付 t13_run 路径本身**也在首 dispatch 硬崩
(fresh NUMBA_CACHE_DIR 也一样),t18_run(只传声明参数)同图同参数正常。因此桥
**默认对所有尺寸用 t18_run**,`--driver t13_run` 只留 A/B。

切换的字节门(全部与既有生产 latents 逐字节相同):

| 尺寸 | t18_run latents sha16 | 参照 |
|---|---|---|
| 128×128 | `6004035d0cb97250` | 生产 job 20261018-170459 的 latents |
| 64×256 | `0df40ca3c18baa0a` | T16 切片 1 的 latents(`w200806a8ac5cp0r1`) |
| 192×128 | `1e8a3ddc1b65de55` | 两次独立运行互比 |



## 5.2 image-token padding 分支(任意尺寸的最后一块)

`patchify_and_embed`(diffusers `transformer_z_image.py` 431-512,本机路径 …/diffusers/
models/transformers/transformer_z_image.py)的规则,已在驱动和 host 参考里逐条实现:

    image_padding_len = (-image_tokens) % 32
    填充行 = 学习到的 x_pad_token(在 all_x_embedder 之后覆盖),不是 patch 投影
    填充行的 pos id = (0,0,0);真实图 row 的 z = cap_ori_len + cap_padding_len + 1
    序列 = [image 真实行..., image 填充行..., caption 32 行],M = 32 + padded
    unpatchify 只取前 image_tokens 行

驱动 `configure()` 现在设 `REAL_TOKENS/PAD_ROWS`,并替换 `t3_real.real_ids`/
`t3_real.x_embedding`(填充行、pos id、x_pad_token);host f32 参考 `t18_hostref.py`
用同一组替换。这样 **tokens % 32 != 0 的尺寸不再被拒绝**。

验证尺寸 **144×128**:grid 9×8 → 72 真实行 + 24 pad = 96 → SEQ=128(复用 192×128 的
M=128 镜像与 perm,缓存 `webui/cache/144x128/`)。门见 §5.3。
### 5.3 144×128 四件套(padding 分支的首个验证;72 真实 + 24 pad → SEQ=128)

| 门 | 结果 | 证据 |
|---|---|---|
| run-to-run 逐字节 | **PASS** sha16 `d1a5f5fa92faaff1`(两次独立进程,maxdiff 0) | `t18_144x128_latents.npy` / `t18_144x128_b_latents.npy` |
| 模型门 vs host f32 | **PASS**:latent corr **0.98241**;图像 **PSNR 25.00 dB**,mean abs 0.0210 | `t18host144_latents.npy` / `t18host144.png` |
| audit 全 core | N/A(复用 M=128 镜像;192×128 的 17+3 核 audit 记录适用) | `*_win_audit.json` |
| 首图并排 | **PASS** | `webui/evidence/T18-144x128-host-vs-npu.png` |

运行:2640 dispatches(30 层×8 步×11),**0 重试**,wall **48.24 s**,device 25.01 s
(att 11.86 + ip 1.50 + ffn 12.65),staging 2.93 s。cache `webui/cache/144x128/`(984 KB,
manifest images = M=128 那套),交付 manifest `webui/shape_manifests/144x128.json`。
**padding 分支结论**:x_pad_token + pos (0,0,0) 的 24 行不改变真实行的注意力语义
(corr 与 192×128 的 0.99422 同一量级),unpatchify 只取 72 行后图像正常(日出山湖)。

## 6. 256×256 的精确剩余量(T18 时点;已由 T19 关闭)

> **T19 更新**:本节描述的 state-8 仍然存在,但它**不改变结果**——T19 用同一路径跑完
> 两次 8 步(run-to-run 逐字节 `683050a95f4bd357`,重试 200/264 次分布完全不同),
> 一步数值门 corr 0.999964,四件套全过,ready=true。实测时长 59–78 min/8 步
> (不是下面估的 2.3–4 h)。证据:`T19-256x256.md`。以下保留 T18 原始记录。

已经做完:ip 镜像(ND=288)+ audit、FFN 行块路径、perm、缓存 + manifest
(`webui/cache/256x256`,`images.ffn=t11_nh5`,3×96 行块)。

**卡点(实测,不是估计)**:M=288 单 dispatch 的 FFN 镜像(36,144 B insts)在设备上
**不完成**(standalone `t4_bench` → `ERT_CMD_STATE_TIMEOUT` 7,025 ms;M=128 的
15,264 B 镜像同 bench 51.8 ms COMPLETED)。改成 t11_nh5 的 96 行块后能跑,但
**每个 FFN 块 dispatch 间歇性返回 ERT state 8**,重试(10 s 退避)后成功:

    192x128: 240 次 FFN dispatch,0 重试
    256x256: 每层约 2 次重试(3 个块),即 ~34 s/层 → 8 步约 2.3 h

**bounded probe(本轮新增,三条)**:
1. `T18_SKIP_ATT=1`(完全不做 attention)→ 仍重试 ⇒ 不是 attention 交互;
2. `--no-cache-all`(省 3.7 GB cache-all)→ 仍重试 ⇒ 不是 cache-all 内存压力;
3. 计时:20 min 内 15 次重试、step 0 未完成 ⇒ **按 ~4 h/8 步**,比 §6 的 2.3 h 估计更差,
   已停跑(不值得占设备 4 h 去赌一个门)。
**结论(记档,不硬撞)**:state-8 只在"M=288 + 96 行块 ×3"这一形态出现;同镜像在
SEQ=96/128 上 240 次 dispatch 0 重试,standalone M=288 单 dispatch 是确定性 TIMEOUT。
一个 probe 内能做的都做了(参数数、cache-all、attention 交互、退避),根因未定位。
**剩余工作(按小时记)**:根治 state-8 或接受 ~4 h 重试重跑 → host f32 参考 27 min →
run-to-run 第二次 → 并排+manifest。

host f32 参考(256×256)预计 288/96×535.9 ≈ 27 min;run-to-run 需要第二次设备跑。
**在这些之前,/api/shapes 对 256×256 保持 ready=false,不显示假进度。**

## 7. 逐字拒绝 / 环境事实

* R1 **ERT 参数必须正好是声明的 BO 数**:交付驱动在 `dispatch` 里把输出 BO 又追加了
  一次(6 数据参数 → 7 个 BO)。M=96 镜像容忍;M=128 镜像**在 kernel call 里直接
  0xC0000005**。standalone 复现:t18_m128 6 args COMPLETED / 7 args 进程死。
  修法:t18_run 只传声明参数(A/B 门仍逐字节相同)。
* R2 **投影行块的结果不能 append 成 Scratch 视图**:`H.unpack_bytes` 返回共享
  Scratch 的视图,块 2 会覆盖块 1 → rows 0..95 变成块 2 的填充行。host-ref 逐 16 行
  RMS 显示 rows 0..95 rel 1.00、96..127 rel 0.008,正好是这个别名。修法:预分配
  (HEADS, SEQ, HD) 直接写切片。
* R3 **手跑 bridge 必须设 NUMBA_CACHE_DIR**:生产 runner 设成
  `webui/cache/numba`;手跑不设时用了源码旁的缓存,首 dispatch 直接硬崩(0xC0000005,
  无 traceback)。用 fresh `NUMBA_CACHE_DIR` 后同命令通过。**这是环境/缓存问题,不是镜像。**
* R4 teardown 的 0xC0000005(-1073741819)在 calls.json/latents 落盘之后,按 T13 R5
  判据不算失败。
* R5 256 单 dispatch 的 36,144 B insts 上限是**设备行为**(standalone 复现),不是
  host 代码;行块是可行形态,但 state-8 频率待解(§6)。
* R6 交付 t13_run 路径在当前环境首 dispatch 硬崩(无 traceback,fresh numba cache
  同样);t18_run 同图同参数通过且逐字节相同 → 产品默认改 t18_run(§5.1)。


## 9. build_shape(H,W) 接真构建 + prebuild 进度流

`webui/build_shape.py` 把"计划→编译→审计→缓存→交付 manifest"做成一条**带阶段的
记录流**,每个阶段有自己的墙钟与结果;CLI 与 API 用的是同一个函数:

    py312 -m webui.build_shape --height 192 --width 128
    [build] plan      done   {'detail': {'key': '192x128', 'm': 128, 'images': {...}, 'files': 9}}
    [build] audit_ffn done   {'detail': {'reused': 17, 'cores': 17, 'pass': 13, ...}}
    [build] audit_ip  done   {'detail': {'reused': 3, 'cores': 3, 'pass': 3, ...}}
    [cache] 192x128: HIT ... (9 files, built 2026-10-18T21:08:59)
    [build] cache     done   {'detail': {'dir': ..., 'files': 9}}
    [build] manifest  done   {'detail': '...\\webui\\shape_manifests\\192x128.json'}
    [build] done      done   {}

* 缺产物才编译(COMPILE 表:FFN = t4_compile --M/--kgu 8;ip = t18_attn_compile
  --M/--ND);**已有审计就复用**(每个核一个 `*_core_x_y_win_audit.json`),不重复
  relink。**实测**:重跑 winning build 会在 core_2_5 触发
  `ld.lld: section '.text' will not fit in region 'program': overflowed by 880 bytes`
  ——审计是构建产物,应复用而不是每次重建;这条写进 AUDIT 的复用判据。
* 审计判据不是"全绿":FFN 基线是 **13/17**(交付件 4 个 gate/up 核的 ACQ CLOBBER),
  ip 是 3/3;新核变红才算失败。
* `POST /api/shapes/prebuild {height,width,build:true}` → `prebuild.prebuild(...,
  build=True)` → `build_shape.build()`,返回 `stages`;不带 build=true 时,未构建
  尺寸仍只返回构建计划(BUILD REQUIRED 保留,不假装)。
* 交付 manifest:`webui/shape_manifests/192x128.json`(入库),逐文件 sha256;
  `/api/shapes` 对 192×128 现在 ready=true,256×256 仍 ready=false。


## 11. 任意宽高能力总表(T18 收口)

| 尺寸 H×W | image tokens | M | 状态 | 门证据 | 构建耗时 / 产物 |
|---|---|---:|---|---|---|
| 128×128 | 64 | 96 | **ready**(交付) | A/B 与生产 latents 逐字节 `6004035d0cb97250` | 交付镜像,0 s;cache 复用 |
| 64×256 | 64 | 96 | **ready**(T16 切片 1) | latents 逐字节 `0df40ca3c18baa0a`;host f32 PSNR 20.70 dB | 交付镜像,0 s;host geometry |
| **192×128** | 96 | 128 | **ready(T18)** | run-to-run `1e8a3ddc1b65de55`;vs host f32 **corr 0.99422 / 29.86 dB**;audit 17+3;并排图 | FFN 18.5 s + ip 8.8 s + perm ~1 s;cache 984 KB |
| **144×128**(padding) | 72(+24 pad) | 128 | **ready(T18)** | run-to-run `d1a5f5fa92faaff1`;vs host f32 **corr 0.98241 / 25.00 dB**;并排图 | 复用 M=128 镜像;cache 984 KB |
| **256×256** | 256 | 288 | **ready(T19)** | run-to-run `683050a95f4bd357`;一步 vs host f32 **0.999964**(8 步 0.90531 / 17.19 dB 为混沌放大,T19 §3.3 对照);audit 13/17+3/3;并排图 | ip 13.7 s(T18)+ FFN 行块复用 t11_nh5;cache **938 KiB** |
| **128×256** | 128 | 160 | **ready(T20)** | run-to-run `7fa580b136b65830`;一步 vs host f32 **0.999975**;audit 3/3+13/17;并排图 | ip ND=160/Mq=32 编译 13.6 s + perm 32×128;FFN 2×96 行块;cache 952 KiB(见 `T20-128x256.md`) |
| 256×384 | 384 | 416 | 构建计划 | — | 需 ip ND=416(M=16 时 L1 超?)+ FFN 行块 |
| 512×512 | 1024 | 1056 | 构建计划(纸面) | — | 需 ip ND=1056 的键分块设计(超出本拓扑 L1) |
| 1024×1024 | 4096 | 4128 | 纸面上限,不宣称 | — | 同上;SBG 95 MB,host BO 压力 |

规则(实测得出的可建性边界):
* **FFN**:行独立,任何 M 都可用"按 M 编译一次(≤~128 行 insts 安全)"或 96 行块拼;
  M≥288 单 dispatch 在本设备上 ERT TIMEOUT(§6),必须分块。
* **T20 增补(实测)**:分块后**不要连续 launch 同一个镜像**——同镜像背靠背的第二次
  launch 不产生完成信号(ERT 看门狗 7.0 s 报 state 8,主机间隔 0–2000 ms 无效);
  块间插一次别的 xclbin 的 launch(1.6 ms 的 ip 即够)就完全消失。256×256 的
  "每层 2 次 state-8"就是这个(3 块 = 2 条边界),由 `t18_run.Dev.barrier` 处理,
  8 步 0 次重试、total 142.7 s。根因见 `T20-STATE8.md`。
* **投影**(t7_p5,m=96):行独立,任意 M 用 96 行块,无需新镜像。
* **ip**:ND=SEQ 键是硬约束,L1 账 QP+KB+C+stack ≤ 64 KB 决定查询块 M:
  `Mq*(KD+SEQ)*2 + KD*SEQ/8*9 + 6144 ≤ 65536`。SEQ=96→Mq=96;128→64;288→16;
  416→16? (16*128*2+128*416/8*9+16*416*2+6144 = 4096+59,904+13,312+6,144 = 83,456 > 64K
  ⇒ 需要键分块/在线 softmax,那是新设计,不是本轮的 M 参数化)。
* **padding**:tokens % 32 ≠ 0 时补 x_pad_token + pos (0,0,0),§5.2;144×128 是首个验证。

## 12. 复现

    py312 _scratch/block/t4_compile.py --design t11_ffn_nh5 --tag t18_m128 --M 128 --NH 5 --kgu 8
    py312 _scratch/attn_impl/t18_attn_compile.py --tag t18_ip_m64nd128 --M 64 --ND 128
    py312 _scratch/block/t4_build_win.py --tag t18_m128
    py312 _scratch/attn_impl/t5_build_win.py --tag t18_ip_m64nd128
    py312 _scratch/image/t18_perm.py --check --emit 64x128 16x128
    py312 -c "from webui import cache; cache.ensure_shape(192,128,print)"
    # 设备(注意 NUMBA_CACHE_DIR 必须是新目录)
    py310 webui/bridge.py --tag t18_192x128 --pe <pe> --schedule <sched> \
          --cache-dir webui/cache/192x128 --steps 8 --seed 0 --height 192 --width 128 \
          --refiner-cache _scratch/image/t6_refcache --driver t18_run
    py312 _scratch/image/t3_real.py --stage vae --tag t18_192x128
    py312 _scratch/image/t18_hostref.py --pe <pe> --schedule-npz <sched> \
          --tag t18host192 --height 192 --width 128 --steps 8 --seed 0
    py312 _scratch/image/t18_cmp_shape.py --dev t18_192x128 --host t18host192 \
          --height 192 --width 128 --out webui/evidence/T18-192x128-host-vs-npu.png
