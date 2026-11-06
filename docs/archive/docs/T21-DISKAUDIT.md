# T21 磁盘清理审计单（只读，零删除）

**任务号** t21_diskaudit ｜ **模式** 只读审计，本文件是本会话唯一新建文件 ｜ **审计窗口** 2026-10-19 09:30:39–09:53（Asia/Shanghai）
**工作根** <workspace> ｜ **git 仓库** zimage-bfp16-npu（HEAD 6be3320，git status 894 条未提交/未跟踪）
**盘况锚点** D: total 652.9 GiB，**free = 953,962,496 B = 0.89 GiB**（[System.IO.DriveInfo] 于 **09:35:07** 读数；Get-PSDrive/Get-CimInstance/fsutil 在本沙箱被拒）
**单位口径** 除标注 GB(10^9) 外，全部为 **MiB/GiB**（PowerShell 1MB=2^20）。历史台账混用二进制与十进制，凡引用处均注明。

> **读数漂移警示**：扫描期间 `_scratch/vae` 在 **09:33:46 / 09:37:46** 仍被 3b1 写入（t21_3b1 子目录），`_scratch/image` 最新写入 **09:14:44**，`_scratch/block` 最新 **09:06:11**。凡涉及这三个区域的数字都是 09:34–09:52 的快照，随时会变。

---

## 0. 结论先行

1. **A 档（纯可再生产物）实测 724.93 MiB**（.prj 717.73 + 仓库 __pycache__ 7.2），其中**可立即执行 591.09 MiB**（A3 的 133.84 待 t5 定稿）。删除后 D: 余量 ≈ **1.47 GiB（立即口径）/ 1.60 GiB（全 A 档）**。
2. **A 档不足以支撑 3b 的 92 个 xclbin 编译**：按本仓库实测单几何足迹 7.9 MiB 估算需 ~727 MiB，全 A 档 724.93 MiB 与需求 727 MiB **几乎相等**，等于把安全余量归零；立即执行版 591 MiB 则连需求本身都不够；若按任务书提到的“.prj 单个大小 × 92”纸面口径（1.95 MiB × 92 = 179 MiB）算则会低估 4 倍。
3. **重大负面发现**：`zimage-port\_probe_out` 下 **4 个目录已消失且不在清理台账内**，合计 **9,441,002,736 B（8.79 GiB）**，其中 `single` 在作者 **NEVER 名单**上且是 10 个 p2_probe 脚本的 live fixture。这与 bfp16_weights_v2 的“未记账删除”是同一模式，建议在负责人确认前暂停一切删除。
4. **E 项（r3v_/r3img_）实际仍在**（93 文件 / 326.19 MiB，mtime 2026-10-17 08:48–08:52），但 `docs/zimage-bfp16-npu-questions.md:95` 已写“E 已落实（目录已不在）”——文档与文件系统直接冲突。
5. **硬链接去重后**：weights.bfp16 是 **2 份物理副本**（不是 3 份），TE GGUF 是 **1 份物理**（3 个名字）。任何按文件名累加的释放预估都会虚高约 13.9 GiB。

---

## 1. 体量分布（U1）

### 1.1 <workspace> 顶层（files = 文件数，MiB = MiB）

| 顶层区域 | 文件数 | MiB | GiB | 备注 |
|---|---:|---:|---:|---|
| zimage-port | 12,104 | 53,157.2 | 51.91 | _models 30.64 GiB + _probe_out 14.58 GiB + 其余 |
| zimage-bfp16-npu | 37,741 | 32,201.0 | 31.45 | 本仓库（含 _scratch 19.2 GiB） |
| _backup_staging | 7 | 24,984.5 | 24.40 | 7 个载荷，与 MANIFEST 24.399 GiB 吻合 |
| zimage-te-qwen3-4b-bf16.gguf | 1 | 7,678.3 | 7.50 | 与 webui\models、01 备份**同一 inode**（nlink=3） |
| _pylibs | 4,431 | 1,573.2 | 1.54 | AIE 工具链 |
| _wheels | 12 | 1,314.1 | 1.28 | 轮子缓存 |
| llama.cpp-0.5.0 | 6,209 | 837.6 | 0.82 | 其它主题 |
| _git_backup_pre_purge | 1,959 | 811.9 | 0.79 | 隐藏目录，purge 前 git 备份 |
| k2h-llama.cpp | 7,605 | 702.4 | 0.69 | 其它主题 |
| models | 1 | 635.3 | 0.62 | K2-Horizon GGUF（权重） |
| ryzen_ai_nuget_1.8.0 | 3 | 262.7 | 0.26 | 供应商包 |
| research | 2,223 | 105.1 | 0.10 | 其它主题 |
| deepseek-harness | 10,351 | 91.5 | 0.09 | 其它主题 |
| _git_backup_pre_purge2 | 434 | 87.1 | 0.09 | 同上 |
| 其余（aie-rt、wallets、.npm-cache、_faucet、_net、_sn、_ln 等） | ~4,700 | ~75 | 0.07 | — |
| **全树合计** | **86,089** | **121.62 GiB** | 130.57 GB(10^9) | 跳过 reparse point，见 §5.4 |

### 1.2 仓库 zimage-bfp16-npu 二级

| 目录 | 文件数 | MiB | 说明 |
|---|---:|---:|---|
| _scratch | 32,369 | 19,625.9 | 实验/编译总区，见 1.3 |
| webui | 642 | 7,703.8 | 其中 models\*.gguf 7,678.3（1 个物理文件） |
| _out | 27 | 3,899.8 | textenc\te_npu_weights.bin 3,898.9（权重） |
| weights | 6 | 401.3 | block0.npz 202.8 + block0.coalesced.bin 198.4 |
| vendor_artifacts | 1 | 399.5 | 供应商 .rai 唯一存档 |
| attn_impl | 335 | 62.9 | 顶层旧副本 |
| memtile2 | 762 | 48.0 | — |
| dit2 | 631 | 14.0 | — |
| .git | 2,401 | 13.8 | — |
| kernels | 145 | 13.2 | — |
| block | 112 | 9.2 | 顶层旧副本 |
| _npu_cache / 6 / 7 / 8 | 75 / 1 / 19 / 63 | 2.8 / 0.0 / 0.7 / 2.7 | VAIP 缓存 |
| docs | 13 | 0.2 | 含两份 CLEANUP-INVENTORY |

### 1.3 _scratch 三级（19,625.9 MiB / 32,369 文件）

| 子目录 | 文件数 | MiB | 内部大头 |
|---|---:|---:|---|
| image | 2,483 | 7,777.0 | t3real_prep 3,796.88；t6_refcache 2,730.36；根散落 1,234.2 |
| block | 5,552 | 4,496.3 | step6_out 2,036.29；out 1,032.51；step5_out 628.16；t10_out 396.56；t4_out 322.82 |
| perfc | 12,691 | 3,849.2 | out 3,848.44（12,671 文件；其中 .prj 10,522 文件 / 227.61 MiB） |
| util | 2,978 | 1,508.9 | out 1,441.74；out_r5e128b 50.54 |
| attn_impl | 3,387 | 936.9 | t3b_out 645.51；t5_out 227.33；out 62.78 |
| weights | 6 | 401.3 | block0.npz + block0.coalesced.bin（与仓库 weights\ 重复） |
| textenc | 2,198 | 255.6 | out_xn 159.65；out_layer1 75.53 |
| vae | 680 | 233.1 | out 142.34（**审计中仍在增长**）；vae_weights.bfp16 90.17 |
| perf | 689 | 89.6 | out 89.41 |
| memtile2 / dit2 / kernels | 762 / 631 / 151 | 48.0 / 14.0 / 13.3 | — |
| 其余（memtile/out/pipeline/oracle/adaln/attn/qwenimg/residency/wregen） | ~140 | ~2.3 | — |

### 1.4 webui（7,703.8 MiB / 642 文件）

| 子项 | 文件数 | MiB | 属性 |
|---|---:|---:|---|
| models\zimage-te-qwen3-4b-bf16.gguf | 1 | 7,678.30 | 权重（nlink=3，与根目录/备份同 inode） |
| jobs | 320 | 13.05 | 产品作业状态，最新 09:16:26 |
| cache | 113 | 7.35 | 产品缓存；**cache\npu_t21 拒绝访问，未计入** |
| _selftest | 115 | 1.81 | — |
| evidence | 24 | 1.76 | 验收证据（T21-2A2B-STARTUP.md 等） |
| outputs | 30 | 1.15 | 产品输出 |
| shape_manifests | 5 | 0.01 | 128x128 / 128x256 / 144x128 / 192x128 / 256x256 .json |

### 1.5 >50 MiB 文件

全树 **190 个**文件 >50 MiB，合计 **107,081 MiB（104.6 GiB）**逻辑大小。全部清单见**附录 A**（按大小降序；含 43 个 C 档模型/权重/工具链文件与 147 个实验产物）。

### 1.6 .prj / __pycache__ / 根日志 / 空临时目录

| 项 | 实测 | 备注 |
|---|---:|---|
| `.prj` 目录 | **582 个**（仓库 559 + zimage-port 23），内部 **1,130.6 MiB** | 平均 1.95 MiB/个；最大单目录 66.83 MiB（t3b_out\t3b_proj_n3840.prj） |
| `__pycache__` / `.pyc` | 152 目录 / 834 文件 / **16.60 MiB**（.pyc 15.68） | 其中 _pylibs 工具链 ~7.3、_probe_out 2.1、仓库 ~7.2 |
| `*.lock` | 599 个 / 0.00 MiB | 全部 <1 KiB |
| 根目录 `*.log` + `*.txt` | **16 个 / 10.86 MiB** | build-vulkan.log 10.22 为主 |
| 空 `tmp*` 目录 | <workspace> 根 **37**、仓库 **9**、zimage-port **45**，**全部 0 字节** | 另有 pip-install / pip-ephem / quark 等 12 个空目录 |

---

## 2. 三档分类（U2：预测 → 实测）

### 2.0 预测 vs 实测

| 项 | 预测（依 10/11–10/12 台账外推） | 实测 | 判定 |
|---|---|---|---|
| A 档体量 | 0.5–1 GiB（.prj + pycache） | **0.577 GiB** | 命中 |
| B 档体量 | 12–15 GB | **13.68 GiB** | 命中 |
| C 档体量 | ~100 GB | **95.88 GiB**（逻辑，已按硬链接去重） | 略低（因 weights 只算 2 份） |
| _probe_out 残留 | 15.66 GB，成分 = zimage_loop + weights + 探针 | 总量命中 14.58 GiB，但**成分不符**：single / dd_iter / merge_probe / p2_fold 已消失 | **预测错了** |
| E 项 r3v/r3img | 存在、待批准 | **存在**（48 + 45 文件） | 存在，但文档称已删 |

### 2.1 A 档 — 安全可删（纯可再生产物，已过 §3 交叉校验）

| # | 路径 | MiB | 依据 | 可恢复性 | 风险 | 建议时机 |
|---|---|---:|---|---|---|---|
| A1 | `_scratch\perfc\out\**\*.prj` | 227.61 | 最后写入 10-17 18:01（休眠 2 天） | 重跑 perfc 脚本重编 | 低（不涉门） | 立即 |
| A2 | `_scratch\block\step6_out\**\*.prj` | 217.62 | 休眠至 10-17 17:20 | 重编 step6 变体 | 低 | 立即 |
| A3 | `_scratch\attn_impl\t5_out\**\*.prj` | 133.84 | .prj 为 10-15 产物；但同目录 10-19 02:17 有写 | 重编 | **中：t5 昨晨仍在动** | 待 t5 定稿 |
| A4 | `_scratch\util\out\**\*.prj` | 53.74 | 休眠至 10-17 17:29 | 重编 | 低 | 立即 |
| A5 | `_scratch\textenc\out_layer1\**\*.prj` | 19.52 | 休眠至 10-16 19:57 | 重编 | 低 | 立即 |
| A6 | `_scratch\textenc\out_xn\**\*.prj` | 16.79 | 休眠至 10-16 20:54 | 重编 | 低 | 立即 |
| A7 | `_scratch\memtile2\out\**\*.prj` | 16.29 | 无近期写入 | 重编 | 低 | 立即 |
| A8 | `_scratch\perf\out\**\*.prj` | 12.15 | 休眠至 10-16 21:40 | 重编 | 低 | 立即 |
| A9 | `_scratch\dit2\out\**\*.prj` | 10.54 | 无近期写入 | 重编 | 低 | 立即 |
| A10 | `_scratch\attn_impl\out\**\*.prj` | 9.63 | 顶层旧副本（与 attn_impl\out 同尺寸两份） | 重编 | 低 | 立即 |
| A11 | 仓库内 `__pycache__\*.pyc`（排除 _pylibs / _probe_out） | ~7.2 | 全树 152 目录 16.60，扣工具链 7.3 与探针 2.1 | Python 自动再生 | 无 | 立即 |
| A12 | `*.lock`（599 个） | ~0.0 | 空锁文件 | 进程自动重建 | 无 | 立即 |
| A13 | 空 `tmp*` 目录 91 个 + pip/ephem 空目录 12 个 | 0.0 | 0 字节 | 无内容 | 无（收益为 0） | 随时 |
| | **A 档合计（全部）** | **724.93** | 其中可立即执行（不含 A3）= **591.09** | | | |

**A 档未纳入（虽同为 .prj）**：`attn_impl\t3b_out`（258.06，门①）、`block\out`（38.13，含门①的 ffn xclbin 与交付 .ll）、`vae\out`（11.18，3b1 正在写 + 门③）、`block\t4_out`（36.25，10-19 02:10 活跃，且被 `webui/config.py:31` 硬编码 → 转 C19）。

### 2.2 B 档 — 需负责人批准

| # | 路径 | MiB | 依据 | 可恢复性 | 风险 | 建议时机 |
|---|---|---:|---|---|---|---|
| B1 | `zimage-port\_probe_out` 探针残留（已剔除 C 部分：zimage_loop / weights / nuget / NEVER / shards） | 3,720.69 | 10-09→10-12 的结论型探针，多数目录自带 verdict json | 重跑对应探针（分钟级）；部分 vendor zip 不可再下载 | 中：含 17 个 87 MiB 的 plugins 目录 | 逐目录签字后 |
| B2 | `_scratch\perfc\out` 非 .prj（_io.npz / _C_raw.npy / _events.json） | 3,621.57 | perfc 12,691 文件，休眠 2 天 | 重跑 perf 实验（耗器件时间） | 中：含 187.7 MiB events.json | 负责人批准后 |
| B3 | `_scratch\block\step6_out` 非 .prj（15 × 113.9 MiB _io.npz） | 1,818.67 | 休眠至 10-17 17:20 | 重跑 step6 | 中：step6 可能是 T4 上游 | 确认 step6 已归档 |
| B4 | `_scratch\util\out` 非 .prj | 1,388.00 | 休眠至 10-17 17:29 | 重跑 util 实验 | 中 | 负责人批准后 |
| B5 | `_scratch\image` 根散落大 .npy/.npz（非门引用） | ~1,230 | 含 7 × 45.22 host_tensors.npz、5 × 84.38 *_bg.npy、4 × 42.19 *_b3.npy、image_b34_npu_Y 191.25 | 可由 t3real_prep + 脚本重算 | **高：t3real_* 系是 PSNR 参考上游，须逐文件核** | 仅删明确非 t3real*/t3d* 项 |
| B6 | `_git_backup_pre_purge` + `_git_backup_pre_purge2` | 899.0 | 隐藏的 purge 前 git 备份 | 不可从现仓库恢复 | 高：可能是历史保险 | 负责人裁决 |
| B7 | `_scratch\weights\block0.npz` + `block0.coalesced.bin`（与仓库 weights\ 同尺寸两份，**非硬链接**） | 401.18 | samefile=False，确认 2 个物理副本 | 由 block0 重新导出 | 中：若脚本指向 _scratch 路径 | 核对引用后 |
| B8 | `_scratch\image\r3v_*` + `r3img_*`（E 项） | 326.19 | HANDOFF-SUPPLEMENT §E 明确建议删除（完整复现 / 残缺跑） | 一条命令重跑（image_run.py --stage all --tag r3v） | 低 | **负责人显式批准**（任务书前提：未批） |
| B10 | `_scratch\textenc\out_*` 非 .prj | 198.87 | 休眠至 10-16 | 重跑 textenc | 中 | 负责人批准后 |
| B11 | `_scratch\attn_impl\t5_out` 非 .prj | 93.49 | 10-19 02:17 有写 | 重跑 t5 | 高：在动 | 不建议现在 |
| B12 | `_scratch\vae\out\*.prj`（21 个，10-15 编译记录） | 11.18 | 3b1 正在该树工作 | 重编（2.3 min/conv） | 中：毁灭 .ll 交付源 | 3b 收工后 |
| B13 | 根目录 `*.log` / `*.txt`（16 个） | 10.86 | 其它主题（llama.cpp / vulkan）构建日志 | 不可再生但无信息价值 | 低 | 负责人点头即删 |
| | **B 档合计** | **13,719.70（13.40 GiB）** | 原 B9（t4_out 286.57）已在 §8 升级为 C19 | | | |

### 2.3 C 档 — 永不可删（含“最后手段须显式批准”）

| # | 项 | MiB | 归类依据 |
|---|---|---:|---|
| C1 | `_backup_staging\` 7 个载荷 | 24,984.53 | 唯一异地备份集（24.40 GiB，与 MANIFEST 一致）——**最后手段须显式批准** |
| C2 | `zimage-port\_models\`（transformer 3 片 24.62 GB 等） | 31,371.82 | 一切权重的源；不可再生 |
| C3 | `weights.bfp16` **2 个物理副本**：`zimage-port\weights.bfp16`（= `_backup_staging\02` 同 inode）+ `zimage-port\_probe_out\weights.bfp16`（独立 inode） | 13,187.34 | 权重；后者是 check_handoff.py 期望位（CLEANUP-EXECUTION:24） |
| C4 | TE GGUF（1 个物理文件，3 个名字） | 7,678.30 | 权重；WebUI 运行必需 |
| C5 | `_scratch\image\t3real_prep`（30 层载荷） | 3,796.88 | 门① PSNR 参考的上游；且是备份载荷 05 的成员（MANIFEST:13） |
| C6 | `_scratch\image\t6_refcache` | 2,730.36 | webui/config.py:33 REFINER_CACHE —— 产品缓存 |
| C7 | `_scratch\attn_impl\t3b_out` | 645.51 | **门①冻结参考**（10 个 references 全在此）；清一次 = 门重置 |
| C8 | `_scratch\vae`（除 out\*.prj 11.18） | 221.88 | 含 vae_weights.bfp16（权重）+ 3b1 活跃区 + 门③证据 |
| C9 | `_out\`（textenc\te_npu_weights.bin） | 3,899.80 | 权重 |
| C10 | webui\cache + jobs + evidence + outputs + _selftest + shape_manifests + static | ~25.17 | 产品缓存/验收证据；cache 内 manifest 带 sha256 + source（门④） |
| C11 | `vendor_artifacts\` | 399.50 | 供应商 .rai 唯一存档 |
| C12 | `ryzen_ai_nuget_1.8.0` + `_probe_out\ryzenai_nuget` | 992.59 | 无下载政策的工具链缓存 |
| C13 | `_pylibs` + `_wheels` | 2,887.30 | 编译工具链 |
| C14 | `_probe_out\zimage_loop`（8 个 KEEP 基线） | 3,476.15 | 作者 NEVER 名单 |
| C15 | `_scratch\block\out` | 1,032.51 | 含门①的 ffn_image_current xclbin 与交付 .ll（HANDOFF C3） |
| C16 | `_probe_out` NEVER 项：shards_i1024_c32 31.46 + aiert_build 7.81 + cast_nb3 1.03 + dd103t 87.00 + plugins_clean 87.00 | 214.30 | CLEANUP-INVENTORY §4 NEVER 名单 |
| C17 | `models\K2-Horizon-1B-Q4_K_M.gguf` | 635.30 | 下载得到的权重 |
| C18 | `_scratch\pipeline\oracle96.png` + `_scratch\image` 门①文件（t3d* / t3real_host|npu.png） | ~0.05 | 门① psnr_references / evidence |
| C19 | `_scratch\block\t4_out`（原 B9） | 286.57 | **`webui/config.py:31 T4_OUT` 硬编码引用** → 产品路径，非纯 scratch；见 §8.1 |
| | **C 档合计** | **98,465.86（96.16 GiB）** | 逻辑大小，已按硬链接去重（GGUF 记 1 份、备份 02 不另计） |

### 2.4 合账

A 724.93 MiB + B 13,719.70 MiB + C 98,465.86 MiB = **112,910.49 MiB = 110.26 GiB**，占 121.62 GiB 扫描量的 **90.7%**。余下 ~11.5 GiB 是其它主题区（llama.cpp 837.6、k2h 702.4、deepseek-harness 91.5、research 105.1 等）与各实验区 <20 MiB 的长尾。

---

## 3. 交叉校验（U3，关键门）

### 3.1 门①：t3d_baseline.json 三桶 + ffn xclbin —— 全部剔除

`_scratch/image/t3d_baseline.py` 定义 NPU = 仓库根、ATT = `_scratch/attn_impl`、OUT_A = `ATT/t3b_out`、HERE = `_scratch/image`，写死 25 条记录 + 1 个 xclbin：

| 桶（基准目录） | 文件 | 命中候选 | 处置 |
|---|---|---|---|
| references（`_scratch\attn_impl\t3b_out\`） | t3c_chain_q_raw.npy、t3c_chain_o_raw.npy、t3c_fused_o_raw.npy、t3c_fused_h_raw.npy、t3b_ipbfp_qkv.npz、t3b_ipbfp_O/P/S_f32.npy、t3b_ipbfp_pv_raw.npy、t3b_ipbfp_qk_raw.npy（10 个） | A 档原拟包含 `t3b_out\**\*.prj`（258.06 MiB） | **剔除**，整树转 C7 |
| evidence（`_scratch\image\`） | t3d5_calls/verify/npu.png/latents.npy、t3d3_calls/verify、t3d4_calls/verify、t3d_precision.json、t3d_perm_96x128/96x64.json、t3d_twoimage_probe.json（12 个） | B5 的 image 根散落文件 | **剔除**，归 C18 |
| psnr_references（`_scratch\image\` + `_scratch\pipeline\`） | t3real_host.png、t3real_npu.png、oracle96.png（3 个） | B5 / pipeline | **剔除**，归 C18 |
| ffn_image_current | `_scratch\block\out\step3_M96_K3840_N110240_m16_k8_n64_G4x6L10.xclbin` | 同目录的 .prj（A 档曾候选） | **剔除**，整目录转 C15 |

结论：门①对 A/B 档的净削减 = **-258.06（t3b_out prj）-38.13（block/out prj）**，并把 `image\t3real_*`、`pipeline\oracle96.png` 钉死。

### 3.2 门②：CLEANUP-INVENTORY NEVER 名单

命中并转 C：`zimage_loop` 的 8 个 KEEP 基线（out_op1/out_fix2/out_r1/t_a4/t_w1/g_ca1/out_nomul2/out_nomul2_keep）、`shards*`（shards_i1024_c32）、`aiert_build`、`cast_nb3`、`dd103t`、`plugins_clean`、全部 .md 与实现代码、export_shards_fp32.py / _bfp16_pack.py；另 `single` 本应在此列，**但已不存在**（见 §5.1）。`_out_512\` 在 `_probe_out` 下不存在（ABSENT，与名单记载不符，无文件可保护）。

### 3.3 门③：`_scratch\vae\out\*_results.json`

实测 vae/out 内 **20 个 `*_results.json`，仅 0.02 MiB**，是 T21-3A-AUDIT.md §1.2 / §3.2 逐条引用的“0 ulp / 100% 逐位”证据。它们与 18 个 xclbin、21 个 .prj 同目录；任何“整目录删除 vae/out”都会连带毁灭证据链 → 整个 `_scratch\vae` 归 C8，仅允许在 3b 收工后单独评审 `out\*.prj`（B12，11.18 MiB）。

### 3.4 门④：webui 缓存与 sha256 清单

`webui/build_shape.py:229` 写 {name, sha256} 清单、`webui/cache.py:213` 用 `if _sha256(p) != rec["sha256"]` 做校验、`cache.py:270` 记录 {sha256, source}。即：cache 条目自带 sha256 并回指源 xclbin。因此 ①`webui\cache`、`webui\shape_manifests`、`webui\evidence` 全部归 C10；②cache 引用的源 xclbin 位于 `_scratch\block\out` 等编译目录 → 与门①重复保护。

**盲区**：`webui\cache\npu_t21` 拒绝访问（UnauthorizedAccess），113 个文件 / 7.35 MiB 不含它，其 sha256 清单也未读取。

### 3.5 门⑤：.gitignore 不作为可删依据

`.gitignore` 含 `*.prj/`、`**/out/`、`webui/cache/`、`*.npy`、`*.npz`、`*.xclbin`、`*.bfp16`、`*.gguf`、`*.lock`、`__pycache__/` 等。**本报告未用任何一条 .gitignore 规则论证可删性**；所有 A/B 判定均基于“目录能否重生成 + 是否被门引用 + 最后写入时间”。反例：weights.bfp16、webui\cache、_probe_out\zimage_loop 都在 .gitignore 里，但全部是 C 档。

### 3.6 冲突单列（止损线①）

| # | 冲突 | 证据 | 处置 |
|---|---|---|---|
| K1 | 文档称 E 项已删除，实测仍在 | docs/zimage-bfp16-npu-questions.md:95 写“E 已落实（目录已不在），_scratch\image 下 r3* 通配 0 命中”；实测 93 文件 / 326.19 MiB，mtime **2026-10-17 08:48:19–08:52:03** | 以文件系统为准（仍在），**不改文档**；B8 保持“待批准”并上报 |
| K2 | single / dd_iter / merge_probe / p2_fold 消失但无台账 | Test-Path 全部 False；台账只记 6 项 + bfp16_weights_v2 补记；single 在 NEVER 名单（INVENTORY §5.4/§7），是 10 个 p2_probe_*.py 的 live fixture | **上报，不追溯**；合计 9,441,002,736 B（8.79 GiB） |
| K3 | `_probe_out` 下 4 个 44.58 MB `.rai` 缓存近似同源 | INVENTORY-2 §6 已记 | 已在 B1 内，未单列 |
| K4 | `_out_512` 在 NEVER 名单却不存在 | Test-Path False | 记入 §5.1 |

---

## 4. 释放预估与建议时机（U4）

| 方案 | 释放 | 删后 D: 余量 | 前提 |
|---|---:|---:|---|
| 现状 | — | **0.89 GiB** | — |
| 只删 A 档（可立即执行部分） | 591.09 MiB | **1.47 GiB** | 无（A 档均过门） |
| 只删 A 档（全部，含 A3） | 724.93 MiB | **1.60 GiB** | 待 t5 定稿 |
| A（全）+ B8（r3v/r3img） | 1,051.12 MiB | 1.92 GiB | 负责人批准 |
| A（全）+ B1（_probe_out 探针） | 4,445.62 MiB | 5.23 GiB | 逐目录签字 |
| A（全）+ B1 + B2 + B3 + B4 | 11,273.86 MiB | 11.90 GiB | 逐目录签字 |
| A（全）+ 全 B 档 | 14,444.63 MiB | 15.00 GiB | 全部批准 |
| C1（备份集） | 24,984 MiB | 25.30 GiB | **最后手段须显式批准** |

### 4.1 3b 编译的纸面账（对照历史 .prj 单个体积 × 92）

- 3b 需新编 **92 个 xclbin**（T21-3A-AUDIT §4：conv 55 + GN 24 + UP 13；门次 113 实例 → 92 去重）。
- **纸面 A（任务书口径）**：.prj 平均 **1.95 MiB**（1,130.6 ÷ 581）× 92 = **179 MiB**。
- **纸面 B（实测单几何足迹）**：`_scratch/vae/out` 142.34 MiB ÷ 18 个 xclbin = **7.9 MiB/几何**（.prj + xclbin + ELF/.bin + _io.npz + _C_raw.npy + _results.json）× 92 = **727 MiB**。
- 二者差 **4 倍**——只按 .prj 估算会严重低估；口径 A 不可用作容量规划。
- 另需计入：VAIP/RAI 缓存（历史单次编译 24–52 s，产出 ~44.58 MB/份）、失败重试、日志。

**最小安全余量建议：≥2.5 GiB 空闲再开编**（= 727 MiB 的 3.4 倍，覆盖 2 次全量重试 + 缓存 + 日志）。若只求“不中途写满”，硬下限为 **1.0 GiB**（727 MiB + 30% 缓冲）。

### 4.2 建议时机

| 顺序 | 动作 | 时机 | 理由 |
|---:|---|---|---|
| 1 | 删 A1–A12（591 MiB） | **现在**，但先把 §5.1 的 9.44 GB 去向问清楚 | 零门风险、零恢复成本 |
| 2 | 删 B8（r3v/r3img 326 MiB） | 负责人一句批准 | 复现只需一条命令；E 项本就待批 |
| 3 | 删 B13（根日志 10.86 MiB） | 负责人一句批准 | 其它主题，无信息价值 |
| 4 | 删 B1（_probe_out 3.7 GiB） | 逐目录签字；避开 dd103t / plugins_clean / aiert_build / cast_nb3 / zimage_loop | 结论已在 json/文档 |
| 5 | B2/B3/B4（7.5 GiB） | 确认 perfc / step6 / util 已归档后 | 重跑需器件时间 |
| 6 | B12（vae\out\*.prj） | 3b 收工后 | 3b1 正在该树写 |
| — | B5 / B9 / B11 | **不建议现在** | t4 / t5 / image 仍在写（09:06–09:14） |
| — | C1（备份集 24.4 GiB） | 仅当负责人判定“已有异地副本” | 最后手段 |

---

## 5. 诚实负结果 / 读数漂移声明（U5⑤）

### 5.1 未记账删除（最重要）

`zimage-port\_probe_out` 下 4 个目录 **Test-Path = False**，但 CLEANUP-EXECUTION.md 的 6 项清单与 bfp16_weights_v2 补记里都没有它们：

| 目录 | 10/12 记录大小 | 状态 |
|---|---:|---|
| `single` | 2,204,054,023 B（2.05 GiB） | 不存在。**作者 NEVER 名单成员**（INVENTORY §5.4/§7：live fixture，10 个 p2_probe_*.py 引用） |
| `dd_iter` | 4,342,209,036 B（4.04 GiB） | 不存在 |
| `merge_probe` | 1,447,418,023 B（1.35 GiB） | 不存在 |
| `p2_fold` | 1,447,321,654 B（1.35 GiB） | 不存在 |
| **合计** | **9,441,002,736 B（8.79 GiB）** | 无任何台账条目 |

按台账口径：_probe_out 由 10/12 的 42,726,873,534 B（39.79 GiB）降到今日 **14.58 GiB（15.66 GB 十进制）**，共减 ~27.1 GB；已记账 17.83 + 6.92 = 24.75 GB，缺口 ~2.3 GB。但 4 个消失目录合计 9.44 GB，说明 10/12 之后 `_probe_out` 又**新增**了约 7 GB（gate3、dd_partition、bfp16order、feedcost、noinit、rootcure、skeleton_all_v2、chain、official_four、warm_1024 等新目录）。**只读审计无法闭合这本账**，只能指出：至少 9.44 GB 的删除没有记录，且其中一个在 NEVER 名单上。

这正是任务书点名的那条教训（bfp16_weights_v2 6.9 GB / 521 张量曾未记账删除、不可从备份恢复）的复发。建议：在负责人确认前，**任何新删除动作暂停**。

### 5.2 硬链接去重（防止虚账）

| 组 | 名字数 | 物理份数 | 证据 |
|---|---:|---:|---|
| weights.bfp16 | 3 | **2** | `zimage-port\weights.bfp16` 与 `_backup_staging\02_weights.bfp16` samefile=True（nlink=2）；`_probe_out\weights.bfp16` nlink=1，samefile=False → **独立第二份** |
| TE GGUF | 3 | **1** | 根目录 / webui\models / _backup_staging\01 三者 nlink=3，两两 samefile=True |
| block0.npz | 2 | **2** | weights\block0.npz 与 _scratch\weights\block0.npz 均 nlink=1，samefile=False |

若按文件名累加，权重会虚高 **7,678.30 MiB（GGUF 重复）+ 6,593.67 MiB（02 备份重复）= 14,271.97 MiB ≈ 13.9 GiB**。本报告已全部去重。

### 5.3 读数漂移（并发代理）

| 区域 | 最新写入 | 期间变化 | 影响 |
|---|---|---|---|
| `_scratch\vae` | **09:37:46**（out\t21_3b1\ALL13_r1.log） | 08:40 的 T21-3A-AUDIT 记 616 文件 / 244,217,795 B；本审计 09:34 记 **680 文件 / 233.06 MiB**，09:37 记 out 656 文件 / 142.40 MiB | 3b1 正在写，C8 / B12 数字仅代表本快照 |
| `_scratch\image` | **09:14:44**（w091323ed0f82p0_*.json/npz/npy） | 扫描期间仍在落新作业产物 | B5 的散落文件清单会变 |
| `_scratch\block` | **09:06:11**（step5_run.py） | t4_out 02:10 仍有写 | B9 不建议动 |
| 根目录 md | 并发代理 mvdocs2 在搬 | 本报告引用的根 md 路径可能已变 | 文档引用以 zimage-bfp16-npu\docs\ 版本为准 |

### 5.4 覆盖盲区

- `webui\cache\npu_t21`：UnauthorizedAccess，未计入任何统计（可能含数 MiB–数十 MiB）。
- 全部 reparse point（junction / symlink）被跳过：`_ln\` 10 项、`_pylibs` 内若干链接。故“全树 121.62 GiB”是不含链接目标的口径；指向仓库内部的链接不会重复计数，指向外部的则未计。
- `Get-CimInstance Win32_LogicalDisk` 与 `fsutil` 被拒；盘况取自 `[System.IO.DriveInfo]`。
- 未逐文件校验 190 个 >50 MiB 文件的 sha256（只做路径 / 大小 / 引用交叉），故“重复副本”结论只对已明确 samefile 的三组成立。

### 5.5 本审计自身的写入

唯一应新建的文件是本报告。**如实披露**：扫描过程中我误把一次 stdout（64,000 B 的截断扫描文本）写成了 `<workspace>\_scan_v3.txt.tmp`，随即在同一会话内 Remove-Item 删除并 Test-Path 复核为 False。除此之外零写入、零移动、零改名；未删除任何审计对象。

### 5.6 其它诚实负结果

- **E 项收益被高估**：任务书说 171 MB × 2，实测 **163.21 + 162.98 = 326.19 MiB**（= 342.1 MB 十进制）。171 MB 是十进制口径，MiB 口径只有 163.2。
- **`_probe_out` 残留 15.66 GB 里 6.45 GiB 是权重**，且是独立物理副本；把它计入“可释放”必须由负责人对“权重”身份松口，本报告按规则归 C。
- **git 工作树脏**：HEAD 6be3320，git status --porcelain **894 条**，说明仓库正被多个代理改动；本报告未做任何 git 操作。
- **A3（t5_out .prj，133.84 MiB）可能不该算 A**：同目录 10-19 02:17 有写入。若按“目录活跃即 B”的严口径，A 档只剩 **591.09 MiB**，结论“A 档不足”更强。

---

## 6. 9.44 GB 消失追查（控制器指定①）

### 6.1 时间锚点：唯一的痕迹是父目录 mtime

| 对象 | mtime | 说明 |
|---|---|---|
| `zimage-port\_probe_out` | **2026-10-17 12:24:15** | 目录项增删才会改 mtime；4 个目录被移除正好改它 |
| `_probe_out\zimage_loop` | **2026-10-17 12:24:14** | 其内容文件最新仍为 10-10 20:38，说明改的是**目录项**（少了一个子项），不是文件内容 |
| `_probe_out` 其它子目录 | 最新 2026-10-13 18:59（route3_diff） | 10-17 12:24 那一刻**只有** _probe_out 与 zimage_loop 被动过 |

两条 mtime 相差 1 秒，指向**同一次操作**。除此之外，全 <workspace> 与 zimage-port 根目录在 **10-17 12:10–12:40 之间没有任何文件被写入**（含日志）——即这次删除**没有留下任何日志文件**。

### 6.2 回收站：没有

`D:\$Recycle.Bin` 共 **1,538 文件 / 3,752.7 MiB**，但 2026-10-01 之后新增的条目**只有 2 个、合计 0.0 MiB**（`$R33LBS7.py` / `$I33LBS7.py`，10-03）。9.44 GB 远超回收站总量，且无 10-17 条目 → **永久删除（Remove-Item 或 shutil.rmtree），不可经回收站还原**。

### 6.3 脚本痕迹：树里没有一支脚本指向这 4 个名字（反而都在保护它们）

在 `zimage-port\` 根枚举到 15 个 cleanup 相关脚本/清单，逐个查目标集：

| 文件 | 时间 | 目标集 | 是否涉及本 4 项 |
|---|---|---|---|
| `_scan_clean.py` | 10-11 09:39 | 只读扫描；`PROTECT = {shards, shards_i1024_c32, vaip_cache, **single**, aiert_build, cast_nb3, dd103t, plugins_clean}` | **相反：single 被显式保护** |
| `_clean_exp.py` | 10-11 05:52 | 按前缀删 zimage_loop 实验；`KEEP` 含 **dd_iter** | **相反：dd_iter 被显式保留** |
| `_gen_cleanlist.py` / `_CLEANUP-LIST.txt` | 10-11 09:43 | 生成【待清理】清单 | 只列清单，不删 |
| `_cleanup.py` | 10-10 19:07 | TARGETS = vaip_cache / dd_zip / dd_ab_final / dd_ab / dd_final | 否 |
| `cleanup_large_files.py` | 10-06 19:47 | npu_full / rms_sum_mul / bf16_realgeom / fp16_realgeom / prof* / vaip_tmp | 否 |
| 其余（`cleanup_round18.py`、`_dd31_clean.py`、`_dd76_mt_clean.py`、`_dd80_check_clean.py`、`route3_verify_clean.py`、`_commit_cleanup.log` 等） | 10-06…10-13 | 各自实验目录 | 否 |

**结论**：仓库里的清理工具**没有一支**会把 `single / dd_iter / merge_probe / p2_fold` 列为目标；`single` 与 `dd_iter` 甚至被写进 PROTECT / KEEP。即**不是原作者脚本所为**。

### 6.4 命令历史：没有

`%APPDATA%\Microsoft\Windows\PowerShell\PSReadLine\ConsoleHost_history.txt` 存在（20,809 B，mtime 10-19 07:53），对 `dd_iter|merge_probe|p2_fold|single|Remove-Item` **零命中**。这与“由代理的子进程执行”一致（子进程不写 PSReadLine 历史），但也**不能据此排除**人工删除（可能用了别的 shell 或历史被清过）。

### 6.5 假设检验：是否“上一会话按 CLEANUP-INVENTORY 的可恢复分类执行了未记账删除”

**支持（强间接证据）**

1. 消失的 4 个**恰好是 INVENTORY-2 §7 排名表的第 3、4、5 行**（dd_iter 4.04 GiB / merge_probe 1.35 / p2_fold 1.35）+ 排名表之外的 `single`；已记账的第 1、2、6、7、8 行（zimage_loop 非 KEEP、bfp16_weights_v2、p2_wholegraph、packverify、bfp16_deadlock、gl_work）全部在台账里。**也就是说：台账记到第 2 行就停了，而第 3–5 行也消失了。**
2. 4 个都是 `_probe_out` 内“除保留名单外仅存的 >1 GiB 目录”（保留者为 zimage_loop 3.48 GB、weights.bfp16 6.45 GB、ryzenai_nuget 0.73 GB，三者都在）、与“按大小阈值继续扫”的删除模式吻合。
3. 时间集中：两条 mtime 相差 1 秒，同一次操作。
4. 无日志、无回收站、无历史 → 典型的“顺手删、不记账”。

**反对（不可忽视）**

1. 没有任何命令、脚本、日志直接指向这 4 个名字（§6.3/§6.4 均为零命中）。
2. `single` 在作者 PROTECT 名单与 INVENTORY §7 KEEP 表里，按“可恢复分类”执行的人**不该**删它——除非用了“非保留的大目录全删”这类粗口径。
3. 删除发生在 **10-17 12:24**，与 10-12 的台账会话相隔 5 天；这 5 天里 `_probe_out` 还在长新目录（gate3、dd_partition、bfp16order…）。

**裁定**：**有强间接证据、无直接证据**。可以确认的是“2026-10-17 12:24 有一次针对 `_probe_out` 的多目录删除，目标集与 INVENTORY 排名表第 3–5 行高度一致，且波及 NEVER 名单上的 `single`，全程未记账”；**不能**确认执行者与执行口径。按任务书的记账原则，这 4 项应视为“**不可恢复的未记账删除**”（回收站与备份均无）。

### 6.6 影响与补救

| 目录 | 能否重建 | 代价 |
|---|---|---|
| `dd_iter` | 能：`_dd4_iter.py` / `_dd5_vi.py`（仍在） | REGENERABLE-SLOW（作者原话“耗时较长”），未实测 |
| `merge_probe` | 能：`_probe_merge_shards.py`（仍在），但需先有 shards | 依赖 22.93 GB shard 重导出 |
| `p2_fold` | 能：重跑 ORT 优化（分钟级） | 低 |
| **`single`** | 能：`export_shards_fp32.py` 从 `_models` 重导出（分钟级）；**但 `single\layers_00_03\io.npz` 里那份 CPU 参考输出是当时用真实权重算的**，重算需另跑 host 前向 | 中；**10 个 `p2_probe_*.py` 的 live fixture 已失效**，需重导后逐个复跑 |

**制度性建议**（不改任何文件，仅建议）：在 `zimage-port\` 下建 append-only 的 `_cleanup_log\<UTC时间戳>.json`，任何 >100 MiB 的删除前先写 {路径, 字节, sha256 或 文件数, 授权来源, 执行者}；并在 CI/脚本层拦截 `rmtree` 直调。

---

## 7. E 项冲突对账（控制器指定②）

### 7.1 实际路径：93 个文件全在 `_scratch\image` 根，不在子目录

命令：`Get-ChildItem <repo>\_scratch\image -File -Filter 'r3*'` → **93**，`Group-Object DirectoryName` 只有一个桶（`_scratch\image`）。

| 组 | 文件数 | MiB | 构成 |
|---|---:|---:|---|
| `r3v_*` | **48** | **163.21** | 3 个大数组 + 32 个 dump + 13 个辅助（json / npz / png） |
| `r3img_*` | **45** | **162.98** | 3 个大数组 + 32 个 dump + 10 个辅助 |
| 合计 | **93** | **326.19**（= 342.1 MB 十进制） | — |

大数组（两份各一套，合计 264.4 MiB，占总量的 81%）：
`*_bg.npy` 84.38 MiB ×2、`*_b3.npy` 42.19 MiB ×2、`*_npu_Y.npy` 5.63 MiB ×2。
dump：`*_dumpNNNNN_{A,X,xmid,Y}.npy`，每 tag 32 个（A/X/Y 各 737,408 B，xmid 1,474,688 B），两 tag 共 64 个 / 56.26 MiB。
辅助 23 个 / ~5.5 MiB：`host.png`、`npu.png`、`host_image*`、`npu_image`、`*_calls.json`、`*_traj.npz`、`*_schedule.npz`、`*_split.json`、`*_prep.json`、`verify.json`、`sbg*`。

时间戳：**全部 2026-10-17 08:48:19–08:52:03**（原始写入时间，未被打断/重建）。

### 7.2 为什么 qfix 单的 glob 报 0 命中 —— 已复现，是**基点错误**

| 命令（同一台机、同一分钟） | 结果 |
|---|---|
| cwd = `<workspace>`，`Get-ChildItem '_scratch\image' -Filter 'r3*' -ErrorAction SilentlyContinue` | **0** |
| cwd = `zimage-bfp16-npu`，同一条命令 | **93** |
| cwd = `zimage-bfp16-npu`，`Get-ChildItem '_scratch/image/r3*'` | **93** |

原因：`<workspace>\_scratch` **存在但只有 `wregen\`**，没有 `image\` 子目录。从 <workspace> 基点写相对路径 `_scratch\image` 会命中一个不存在的路径；配上 `-ErrorAction SilentlyContinue`（或 `Test-Path` 直接 False），错误被吞掉 → 得到“0 命中”的**假阴性**。

其它可能但**未成立**的解释：①文件被删后又重建（不成立——mtime 仍是 10-17 08:48–08:52，重建会刷新）；②东西被搬到子目录（不成立——`Group-Object DirectoryName` 只有一个桶）；③在 `zimage-bfp16-repo` 里查的（该树没有 `_scratch`，只有 `_models`/`_probe_out` 两个 junction）。

### 7.3 裁定

**以文件系统为准：E 项未执行，r3v_*/r3img_* 仍在。** `docs/zimage-bfp16-npu-questions.md:95` 那句“E 已落实（目录已不在，r3* 通配 0 命中）”是**假阴性**，不能作为“已删除”的证据；同理，`_CLEANUP-LIST.txt`/INVENTORY 的“待清理”字样也不能当作“已清理”。

本单**不改**那份文档（只读纪律）；建议由负责人在 qfix 单上追加更正注：E 仍未执行，且 glob 需在仓库根执行或改用绝对路径。

---

## 8. B 档大件逐项量化（控制器指定③）

### 8.1 逐项：释放量 / 可恢复性 / 被引用检查

引用检查三件套 = ①`t3d_baseline.json` 三桶 + ffn xclbin；②CLEANUP-INVENTORY NEVER 名单；③脚本硬编码路径（grep `*.py`）。

| 项 | 逻辑 MiB | **物理释放** | 可恢复性 | ①门 | ②NEVER | ③脚本硬编码 | 结论 |
|---|---:|---:|---|---|---|---|---|
| `_probe_out\weights.bfp16` | 6,593.67 | **6,593.67**（nlink=1，独立 inode，删了真释放） | 与 `zimage-port\weights.bfp16` **同尺寸同内容推定**（未做 sha256），可重编（REGENERABLE-SLOW） | 不引用 | 不在名单，但属“一切权重” | `check_handoff.py` 期望该位，但**该脚本在本树不存在**（grep 只命中 CLEANUP-EXECUTION:24） | **B 档最大单项**；须先 sha256 对比再报批。【更正 2026-10-19 t21_wcensus】"同尺寸同内容推定"已被证伪:实测 SHA256 不同(10-12 旧版 bfac45f8… vs 10-14 交付版 4ff5b6a6…),详见 CLEANUP-EXECUTION t21_clean_b8 补记与 T21-WCENSUS 记录;该文件是唯一旧版,删除即销毁旧构建。 |
| `_probe_out` 探针残留（B1 合计） | 3,720.69 | 3,720.69 | 逐目录重跑探针 | 不引用 | 无 | 无 | B（逐目录签字） |
| `_scratch\perfc\out` 非 .prj | 3,621.57 | 3,621.57 | 重跑 perf 实验（耗器件时间） | 不引用 | 无 | **有**：`perfc/t2b_slots.py`、`t2c_read.py`、`t2c_analysis.py` 硬编码 glob `perfc_*_C_raw.npy` | **B，且须保留 `_C_raw.npy`/`_io.npz` 或同步改脚本** |
| `_scratch\block\step6_out` 非 .prj | 1,818.67 | 1,818.67 | 重跑 step6 | 不引用 | 无 | **有**：`attn_impl/t3_h1_slice.py` 硬编码读 `s6_h1_ref.npz`(7.03) / `s6_h1_O.npy`(0.70) / `s6_h1_X.npy`(0.70) | B；**这 3 个 8.43 MiB 必须保留** |
| `_scratch\util\out` 非 .prj | 1,388.00 | 1,388.00 | 重跑 util 实验 | 不引用 | 无 | 无命中 | B |
| `_scratch\image` 根散落（非门） | ~1,230 | ~1,230 | t3real_prep + 脚本重算 | **部分引用**（t3real_host/npu.png、t3d*.json/png 在桶内） | 无 | `r3v_/r3img_` 在 py 中 **0 命中** | B，逐文件 |
| `_git_backup_pre_purge` + `2` | 899.0 | 899.0 | **不可**从现仓库恢复 | 不引用 | 无 | 无 | B（负责人裁决） |
| `_scratch\weights` 重复副本 | 401.18 | 401.18 | `extract.py` 自带 `--out` 默认本目录，可重导 | 不引用 | 无 | **仅自引用**（该目录内的 `extract.py`） | B |
| `r3v_*` + `r3img_*` | 326.19 | 326.19 | 一条命令重跑（`--tag r3v`） | 不引用 | 无 | **0 命中**（只有文档提及） | B（E 项，待批） |
| `_scratch\block\t4_out` 非 .prj | 286.57 | 286.57 | 重跑 T4 | 不引用 | 无 | **有：`webui/config.py:31 T4_OUT`** | **升级 C19**（产品路径） |
| `_scratch\textenc\out_*` 非 .prj | 198.87 | 198.87 | 重跑 textenc | 不引用 | 无 | 无命中 | B |
| `_scratch\attn_impl\t5_out` 非 .prj | 93.49 | 93.49 | 重跑 t5 | 不引用 | 无 | 无命中 | B（活跃） |
| `_scratch\vae\out\*.prj`（21 个） | 11.18 | 11.18 | 重编 2.3 min/conv | 不引用 | 无 | 无直接引用；但同目录 18 个 xclbin 与 20 个 `*_results.json` 是门③证据 | B（3b 收工后） |
| 根 `*.log`/`*.txt`（16 个） | 10.86 | 10.86 | 不可再生、无信息价值 | 不引用 | 无 | 无 | B |

### 8.2 由本表触发的重分类（对 §2 的更正）

- **B9（`t4_out` 286.57 MiB）→ C19**：`webui/config.py:31` 把它定义为 `T4_OUT`，是产品代码路径，不是纯 scratch。B 档总量由 14,006.27 降为 **13,719.70 MiB（13.40 GiB）**，C 档升为 **98,465.86 MiB（96.16 GiB）**。
- **B2/B3 增加“脚本引用”约束**：不是整目录可删，必须先保住 `perfc_*_C_raw.npy`（若脚本要复算）与 `s6_h1_*.{npz,npy}`（3 个 8.43 MiB）。

---

## 9. 预计释放总量与 3b 编译空间缺口对策

### 9.1 总量

| 档 | MiB | GiB | 删后 D: 余量（自 0.89 GiB 起） |
|---|---:|---:|---:|
| A（可立即执行部分，不含 A3） | 591.09 | 0.577 | **1.47 GiB** |
| A（全部，含 A3） | 724.93 | 0.708 | **1.60 GiB** |
| **A + B（全部）** | **14,444.63** | **14.11** | **15.00 GiB** |
| 参考：C 档（仅作上限，需最后手段批准） | 98,465.86 | 96.16 | 96.9 GiB |

### 9.2 3b 的需求

- 92 个新 xclbin；实测单几何足迹 7.9 MiB → **727 MiB**；若只算 .prj（1.95 MiB/个）= 179 MiB，**低估 4 倍**。
- 加 VAIP/RAI 缓存（历史 ~44.58 MB/份）、失败重试、日志 → **建议开编前 ≥2.5 GiB 空闲**，硬下限 1.0 GiB。

### 9.3 缺口与对策（按风险从低到高）

| # | 对策 | 可释放 | 风险 | 前置条件 |
|---|---:|---|---|---|
| 0 | **清空 D: 回收站**（3,752.7 MiB，含 7 月的 vendor dll；与本项目无关） | 3.67 GiB | 极低 | 不在本审计范围，需负责人执行；本单不碰 |
| 1 | 删 A 档（591 MiB 立即 / 725 MiB 全量） | 0.58–0.71 GiB | 极低 | 通过门①②③④（已过） |
| 2 | + B8（r3v/r3img 326 MiB） | +0.32 GiB | 低 | 负责人一句批准；E 项本就待批 |
| 3 | + B1（_probe_out 探针 3.72 GiB） | +3.63 GiB | 中 | 逐目录签字，避开 NEVER 5 项 |
| 4 | 单独处置 `_probe_out\weights.bfp16`（6.45 GiB） | +6.44 GiB | 中高 | 先 sha256 与 `zimage-port\weights.bfp16` 比对一致 + 负责人对“权重”松口 |
| 5 | 就地压缩：`_backup_staging\05_runtime-payloads-gates.tar`(7.01 GiB) 已含 t3real_prep + t6_refcache + t3b_out；若确认 tar 可解，可删 t3real_prep(3.71 GiB) | 3.71 GiB | **高** | C 档“最后手段”；**tar 完整性未验证**（本单只读，未解开） |
| 6 | 同上，t6_refcache(2.67 GiB) 若确认可由 webui 重建 | 2.67 GiB | **高** | 同上 + 重建成本未实测 |

### 9.4 一句话对策

**A 档单独不够（591 MiB 立即口径 < 727 MiB 需求；全量 725 MiB ≈ 需求本身，余量为零）。** 若要 3b 一次跑完 92 个 xclbin 并留安全余量，最小组合是 **A + B8 + B1（≈ 4.44 GiB，删后 5.23 GiB 空闲）**；零风险捷径是**先清回收站（3.67 GiB）**。`_probe_out\weights.bfp16`（6.44 GiB）是单笔最大可释放项，但必须先做哈希复核。

---

## 附录 A — 全树 >50 MiB 文件（190 个，降序；单位 MiB）

路径相对 `<workspace>\`；C = 永不可删，B = 需批准。档位按 §1.5 的关键词口径粗分；`t4_out` / `step5_out` 的个别大文件以 §2.3（C19）与 §8.2 的更正为准，不以本表字母为准。

| # | 文件 | MiB | 档 |
|---:|---|---:|---|
| 1 | zimage-port\_models\Z-Image-Turbo\transformer\diffusion_pytorch_model-00002-of-00003.safetensors | 9,511.7 | C |
| 2 | zimage-port\_models\Z-Image-Turbo\transformer\diffusion_pytorch_model-00001-of-00003.safetensors | 9,511.7 | C |
| 3 | zimage-bfp16-npu\webui\models\zimage-te-qwen3-4b-bf16.gguf | 7,678.3 | C |
| 4 | _backup_staging\01_te-gguf-zimage-te-qwen3-4b-bf16.gguf | 7,678.3 | C |
| 5 | zimage-te-qwen3-4b-bf16.gguf | 7,678.3 | C |
| 6 | _backup_staging\05_runtime-payloads-gates.tar | 7,181.1 | C |
| 7 | _backup_staging\02_weights.bfp16 | 6,593.7 | C |
| 8 | zimage-port\_probe_out\weights.bfp16 | 6,593.7 | C |
| 9 | zimage-port\weights.bfp16 | 6,593.7 | C |
| 10 | zimage-port\_models\Z-Image-Turbo\transformer\diffusion_pytorch_model-00003-of-00003.safetensors | 4,455.8 | C |
| 11 | zimage-bfp16-npu\_out\textenc\te_npu_weights.bin | 3,898.9 | C |
| 12 | zimage-port\_models\Z-Image-Turbo\text_encoder\model-00002-of-00003.safetensors | 3,802.7 | C |
| 13 | zimage-port\_models\Z-Image-Turbo\text_encoder\model-00001-of-00003.safetensors | 3,774.5 | C |
| 14 | _backup_staging\04_toolchain-pylibs-llama.tar | 3,127.5 | C |
| 15 | models\K2-Horizon-1B-Q4_K_M.gguf | 635.3 | C |
| 16 | _backup_staging\03_vendor_39e41383350b8c53e90996269478cfb7.rai | 399.5 | C |
| 17 | zimage-bfp16-npu\vendor_artifacts\39e41383350b8c53e90996269478cfb7.rai | 399.5 | C |
| 18 | _wheels\llvm_aie-22.0.0.2026092101+0006955e-py3-none-win_amd64.whl | 382.3 | C |
| 19 | _wheels\llvm_aie-21.0.0.2026061601+dc0ebf29-py3-none-win_amd64.whl | 371.7 | C |
| 20 | _wheels\llvm_aie-21.0.0.2026061101+50bd3557-py3-none-win_amd64.whl | 371.7 | C |
| 21 | _git_backup_pre_purge\objects\64\33c885c74c15bff8a48e550b2bded896794f48 | 369.2 | C |
| 22 | zimage-port\_probe_out\ryzenai_nuget\RyzenAI_Deployment.1.8.0\runtimes\win-x64\native\vaiml.dll | 265.4 | B |
| 23 | ryzen_ai_nuget_1.8.0\RyzenAI_Deployment.1.8.0.nupkg | 262.1 | C |
| 24 | zimage-bfp16-npu\_scratch\weights\block0.npz | 202.8 | B |
| 25 | zimage-bfp16-npu\weights\block0.npz | 202.8 | B |
| 26 | zimage-bfp16-npu\weights\block0.coalesced.bin | 198.4 | B |
| 27 | zimage-bfp16-npu\_scratch\weights\block0.coalesced.bin | 198.4 | B |
| 28 | zimage-bfp16-npu\_scratch\image\image_b34_npu_Y.npy | 191.3 | B |
| 29 | _git_backup_pre_purge\objects\a6\7c6e6185a8089abd1985121d2d13b29a4f79cb | 188.0 | C |
| 30 | zimage-bfp16-npu\_scratch\perfc\out\perfc_r15mix_M1024_K32_N512_m32_kk8_n512_Kc32_d21b2_full_events.json | 187.7 | B |
| 31 | _wheels\mlir_aie-1.4.4-cp312-cp312-win_amd64.whl | 185.0 | C |
| 32 | _git_backup_pre_purge\objects\2f\aaee4d6d3c9cfd1fab62a10eacb9bf1280747b | 183.9 | C |
| 33 | zimage-port\_models\Z-Image-Turbo\vae\diffusion_pytorch_model.safetensors | 159.9 | C |
| 34 | zimage-port\_probe_out\ryzenai_nuget\RyzenAI_Deployment.1.8.0\runtimes\win-x64\native\dyn_dispatch_core.dll | 151.1 | B |
| 35 | zimage-port\_probe_out\bfp16order\gemm_probe.onnx | 150.0 | B |
| 36 | zimage-port\_probe_out\noinit\gemm_inline.onnx | 150.0 | B |
| 37 | zimage-port\_probe_out\feedcost\gemm_inline.onnx | 150.0 | B |
| 38 | zimage-bfp16-npu\_scratch\image\t6_refcache\noise_refiner.0\w3.npy | 150.0 | B |
| 39 | zimage-bfp16-npu\_scratch\image\t6_refcache\context_refiner.1\w3.npy | 150.0 | B |
| 40 | zimage-bfp16-npu\_scratch\image\t6_refcache\context_refiner.1\w1.npy | 150.0 | B |
| 41 | zimage-bfp16-npu\_scratch\image\t6_refcache\context_refiner.1\w2.npy | 150.0 | B |
| 42 | zimage-bfp16-npu\_scratch\image\t6_refcache\noise_refiner.0\w2.npy | 150.0 | B |
| 43 | zimage-bfp16-npu\_scratch\image\t6_refcache\noise_refiner.1\w3.npy | 150.0 | B |
| 44 | zimage-bfp16-npu\_scratch\image\t6_refcache\context_refiner.0\w3.npy | 150.0 | B |
| 45 | zimage-bfp16-npu\_scratch\image\t6_refcache\noise_refiner.1\w1.npy | 150.0 | B |
| 46 | zimage-bfp16-npu\_scratch\image\t6_refcache\noise_refiner.1\w2.npy | 150.0 | B |
| 47 | zimage-bfp16-npu\_scratch\image\t6_refcache\context_refiner.0\w1.npy | 150.0 | B |
| 48 | zimage-bfp16-npu\_scratch\image\t6_refcache\noise_refiner.0\w1.npy | 150.0 | B |
| 49 | zimage-bfp16-npu\_scratch\image\t6_refcache\context_refiner.0\w2.npy | 150.0 | B |
| 50 | zimage-port\_probe_out\noinit\w.bin | 150.0 | B |
| 51 | zimage-port\_probe_out\ryzenai_nuget\RyzenAI_Deployment.1.8.0\runtimes\win-x64\native\aiecompiler_client.dll | 142.6 | B |
| 52 | k2h-llama.cpp\build-vulkan\ggml\src\ggml-vulkan\mul_mm.comp.cpp | 123.3 | B |
| 53 | _pylibs\mlir_aie\bin\aiecc.exe | 121.9 | C |
| 54 | _pylibs\mlir_aie\python\aie\_mlir_libs\AIEAggregateCAPI.dll | 121.4 | C |
| 55 | zimage-bfp16-npu\_scratch\block\step6_out\s6_M96_H30_D128_m16_8_kc32_nc32_d111_ws4_cap4_io.npz | 113.9 | B |
| 56 | zimage-bfp16-npu\_scratch\block\step6_out\s6_M96_H30_D128_m16_8_kc32_nc32_d111_io.npz | 113.9 | B |
| 57 | zimage-bfp16-npu\_scratch\block\step6_out\s6_M96_H30_D128_m16_8_kc32_nc32_d111_noop_io.npz | 113.9 | B |
| 58 | zimage-bfp16-npu\_scratch\block\step6_out\s6_M96_H30_D128_m16_8_kc32_nc32_d111_cap1_io.npz | 113.9 | B |
| 59 | zimage-bfp16-npu\_scratch\block\step6_out\s6_M96_H30_D128_m16_8_kc32_nc32_d111_cap4_io.npz | 113.9 | B |
| 60 | zimage-bfp16-npu\_scratch\block\step6_out\s6_M96_H30_D128_m16_8_kc32_nc32_d111_ws3_cap4_io.npz | 113.9 | B |
| 61 | zimage-bfp16-npu\_scratch\block\step6_out\s6_M96_H30_D128_m16_8_kc32_nc32_d111_ws2_cap4_io.npz | 113.9 | B |
| 62 | zimage-bfp16-npu\_scratch\block\step6_out\s6_M96_H30_D128_m16_8_kc32_nc32_d111_w1_io.npz | 113.9 | B |
| 63 | zimage-bfp16-npu\_scratch\block\step6_out\s6_M96_H30_D128_m16_8_kc32_nc32_d111_w1_cap4_io.npz | 113.9 | B |
| 64 | zimage-bfp16-npu\_scratch\block\step6_out\s6_M96_H30_D128_m16_8_kc32_nc32_d111_w2_io.npz | 113.9 | B |
| 65 | zimage-bfp16-npu\_scratch\block\step6_out\s6_M96_H30_D128_m16_8_kc32_nc32_d111_ws1_cap4_io.npz | 113.9 | B |
| 66 | zimage-bfp16-npu\_scratch\block\step6_out\s6_t3full_io.npz | 113.9 | B |
| 67 | zimage-bfp16-npu\_scratch\block\step6_out\s6_M96_H30_D128_m16_8_kc32_nc32_d111_h1_io.npz | 113.9 | B |
| 68 | zimage-bfp16-npu\_scratch\block\step6_out\s6_M96_H30_D128_m16_8_kc32_nc32_d111_h2_io.npz | 113.9 | B |
| 69 | zimage-bfp16-npu\_scratch\block\step6_out\s6_h1_io.npz | 113.9 | B |
| 70 | _pylibs\mlir_aie\bin\aie-opt.exe | 113.7 | C |
| 71 | zimage-port\_probe_out\ryzenai_nuget\RyzenAI_Deployment.1.8.0\runtimes\win-x64\native\onnxruntime_vitisai_ep.dll | 113.6 | B |
| 72 | _pylibs\llvm-aie\llvm-aie\bin\clang.exe | 104.0 | C |
| 73 | _pylibs\llvm-aie\llvm-aie\bin\clang-cpp.exe | 104.0 | C |
| 74 | _pylibs\llvm-aie\llvm-aie\bin\clang++.exe | 104.0 | C |
| 75 | _pylibs\llvm-aie\llvm-aie\bin\clang-cl.exe | 104.0 | C |
| 76 | zimage-port\_models\Z-Image-Turbo\text_encoder\model-00003-of-00003.safetensors | 95.0 | C |
| 77 | zimage-bfp16-npu\_scratch\vae\vae_weights.bfp16 | 90.2 | C |
| 78 | llama.cpp-0.5.0\build-vulkan\ggml\src\ggml-vulkan\mul_mm.comp.cpp | 85.9 | B |
| 79 | _git_backup_pre_purge2\objects\ea\f195610efc0775f1c375588b53c1173e66223c | 85.6 | C |
| 80 | zimage-bfp16-npu\_scratch\image\t3real_prep\layers.10_bg.npy | 84.4 | B |
| 81 | zimage-bfp16-npu\_scratch\image\t3real_prep\layers.13_bg.npy | 84.4 | B |
| 82 | zimage-bfp16-npu\_scratch\image\t3real_prep\layers.15_bg.npy | 84.4 | B |
| 83 | zimage-bfp16-npu\_scratch\image\t3real_prep\layers.14_bg.npy | 84.4 | B |
| 84 | zimage-bfp16-npu\_scratch\image\t3real_prep\layers.16_bg.npy | 84.4 | B |
| 85 | zimage-bfp16-npu\_scratch\image\t3real_prep\layers.18_bg.npy | 84.4 | B |
| 86 | zimage-bfp16-npu\_scratch\image\t3real_prep\layers.19_bg.npy | 84.4 | B |
| 87 | zimage-bfp16-npu\_scratch\image\t3real_prep\layers.1_bg.npy | 84.4 | B |
| 88 | zimage-bfp16-npu\_scratch\image\t3real_prep\layers.11_bg.npy | 84.4 | B |
| 89 | zimage-bfp16-npu\_scratch\image\t3real_prep\layers.17_bg.npy | 84.4 | B |
| 90 | zimage-bfp16-npu\_scratch\image\t3real_prep\layers.12_bg.npy | 84.4 | B |
| 91 | zimage-bfp16-npu\_scratch\image\t3real_prep\layers.8_bg.npy | 84.4 | B |
| 92 | zimage-bfp16-npu\_scratch\image\t3real_prep\layers.9_bg.npy | 84.4 | B |
| 93 | zimage-bfp16-npu\_scratch\block\t4_out\payload_sbgrow0\BG.npy | 84.4 | B |
| 94 | zimage-bfp16-npu\_scratch\image\t3real_prep\layers.5_bg.npy | 84.4 | B |
| 95 | zimage-bfp16-npu\_scratch\image\t3real_prep\layers.6_bg.npy | 84.4 | B |
| 96 | zimage-bfp16-npu\_scratch\image\t3real_prep\layers.7_bg.npy | 84.4 | B |
| 97 | zimage-bfp16-npu\_scratch\block\t4_out\payload_nh5c_k8\BG.npy | 84.4 | B |
| 98 | zimage-bfp16-npu\_scratch\block\step5_out\synth_block1\BG.npy | 84.4 | B |
| 99 | zimage-bfp16-npu\_scratch\block\step5_out\synth_block0\BG.npy | 84.4 | B |
| 100 | zimage-bfp16-npu\_scratch\block\step5_out\real_block1\BG.npy | 84.4 | B |
| 101 | zimage-bfp16-npu\_scratch\block\t10_out\m96_lo_io\BG.npy | 84.4 | B |
| 102 | zimage-bfp16-npu\_scratch\block\t10_out\m96_hi_io\BG.npy | 84.4 | B |
| 103 | zimage-bfp16-npu\_scratch\block\t10_out\m192_io\BG.npy | 84.4 | B |
| 104 | zimage-bfp16-npu\_scratch\image\t3real_prep\layers.23_bg.npy | 84.4 | B |
| 105 | zimage-bfp16-npu\_scratch\image\t3real_prep\layers.24_bg.npy | 84.4 | B |
| 106 | zimage-bfp16-npu\_scratch\image\t3real_prep\layers.25_bg.npy | 84.4 | B |
| 107 | zimage-bfp16-npu\_scratch\image\t3real_prep\layers.20_bg.npy | 84.4 | B |
| 108 | zimage-bfp16-npu\_scratch\image\t3real_prep\layers.21_bg.npy | 84.4 | B |
| 109 | zimage-bfp16-npu\_scratch\image\t3real_prep\layers.22_bg.npy | 84.4 | B |
| 110 | zimage-bfp16-npu\_scratch\image\t3real_prep\layers.26_bg.npy | 84.4 | B |
| 111 | zimage-bfp16-npu\_scratch\image\t3real_prep\layers.2_bg.npy | 84.4 | B |
| 112 | zimage-bfp16-npu\_scratch\image\t3real_prep\layers.3_bg.npy | 84.4 | B |
| 113 | zimage-bfp16-npu\_scratch\image\t3real_prep\layers.4_bg.npy | 84.4 | B |
| 114 | zimage-bfp16-npu\_scratch\image\t3real_prep\layers.27_bg.npy | 84.4 | B |
| 115 | zimage-bfp16-npu\_scratch\image\t3real_prep\layers.28_bg.npy | 84.4 | B |
| 116 | zimage-bfp16-npu\_scratch\image\t3real_prep\layers.29_bg.npy | 84.4 | B |
| 117 | zimage-bfp16-npu\_scratch\block\out\step3_M16_K3840_N110240_m16_k8_n64_G1x1L10_L2_io\BG.npy | 84.4 | B |
| 118 | zimage-bfp16-npu\_scratch\block\out\step3_M16_K3840_N110240_m16_k8_n64_G1x1L1_L2_io\BG.npy | 84.4 | B |
| 119 | zimage-bfp16-npu\_scratch\block\out\step3_M16_K3840_N110240_m16_k8_n64_G1x6L10_L1_io\BG.npy | 84.4 | B |
| 120 | zimage-bfp16-npu\_scratch\image\r3v_bg.npy | 84.4 | B |
| 121 | zimage-bfp16-npu\_scratch\image\image_b34_bg.npy | 84.4 | B |
| 122 | zimage-bfp16-npu\_scratch\image\image_bg.npy | 84.4 | B |
| 123 | zimage-bfp16-npu\_scratch\image\r3img_bg.npy | 84.4 | B |
| 124 | zimage-bfp16-npu\_scratch\block\out\step3_M16_K3840_N110240_m16_k8_n64_G4x6L10_L2_io\BG.npy | 84.4 | B |
| 125 | zimage-bfp16-npu\_scratch\block\out\step3_M96_K3840_N110240_m16_k8_n64_G4x6L10_io\BG.npy | 84.4 | B |
| 126 | zimage-bfp16-npu\_scratch\block\step5_out\real_block0\BG.npy | 84.4 | B |
| 127 | zimage-bfp16-npu\_scratch\image\t3real_prep\layers.0_bg.npy | 84.4 | B |
| 128 | zimage-bfp16-npu\_scratch\block\out\step3_M16_K3840_N110240_m16_k8_n64_G4x1L10_L2_io\BG.npy | 84.4 | B |
| 129 | zimage-bfp16-npu\_scratch\block\out\step3_M16_K3840_N110240_m16_k8_n64_G4x6L10_io\BG.npy | 84.4 | B |
| 130 | zimage-bfp16-npu\_scratch\block\out\step3_M16_K3840_N110240_m16_k8_n64_G4x6L10_L1_io\BG.npy | 84.4 | B |
| 131 | zimage-bfp16-npu\_scratch\perfc\out\perfc_r6g10_8_M4096_K3840_N10240_m32_kk8_n512_Kc640_d21b2_full_stock_C_raw.npy | 80.0 | B |
| 132 | zimage-bfp16-npu\_scratch\perfc\out\perfc_r6g10_8_M4096_K3840_N10240_m32_kk8_n512_Kc640_d21b2_full_stock_io.npz | 72.2 | B |
| 133 | zimage-port\_probe_out\dd_partition\control_sd3_dims.onnx | 72.0 | B |
| 134 | zimage-port\_probe_out\dd_live\ctrl2.onnx | 72.0 | B |
| 135 | zimage-bfp16-npu\_scratch\perfc\out\perfc_vcopy_M1024m32n256d1_M1024_K2560_N1024_m32_kk8_n256_Kc640_d21b2_full_stock_io.npz | 67.8 | B |
| 136 | _pylibs\llvm-aie\llvm-aie\bin\lld.exe | 62.7 | C |
| 137 | _pylibs\llvm-aie\llvm-aie\bin\wasm-ld.exe | 62.7 | C |
| 138 | _pylibs\llvm-aie\llvm-aie\bin\ld.lld.exe | 62.7 | C |
| 139 | _pylibs\llvm-aie\llvm-aie\bin\ld64.lld.exe | 62.7 | C |
| 140 | _pylibs\llvm-aie\llvm-aie\bin\lld-link.exe | 62.7 | C |
| 141 | zimage-port\_probe_out\gx_plugins1\sd_fastpm_DiT.zip | 59.8 | B |
| 142 | zimage-port\_probe_out\only_vendor\plugins\sd_fastpm_DiT.zip | 59.8 | B |
| 143 | zimage-port\_probe_out\gl_plug\sd_fastpm_DiT.zip | 59.8 | B |
| 144 | zimage-port\_probe_out\dd85\nb6\sd_fastpm_DiT.zip | 59.8 | B |
| 145 | zimage-port\_probe_out\gl17_plugins\sd_fastpm_DiT.zip | 59.8 | B |
| 146 | zimage-port\_probe_out\plugins_nb5\sd_fastpm_DiT.zip | 59.8 | B |
| 147 | zimage-port\_probe_out\plugins_nb6\sd_fastpm_DiT.zip | 59.8 | B |
| 148 | zimage-port\_probe_out\zimage_loop\plugins\sd_fastpm_DiT.zip | 59.8 | B |
| 149 | zimage-port\_probe_out\plugins_mark\sd_fastpm_DiT.zip | 59.8 | B |
| 150 | zimage-port\_probe_out\plugins_clean\sd_fastpm_DiT.zip | 59.8 | B |
| 151 | zimage-port\_probe_out\plugins_clean2\sd_fastpm_DiT.zip | 59.8 | B |
| 152 | zimage-port\_probe_out\plugins_fix\sd_fastpm_DiT.zip | 59.8 | B |
| 153 | zimage-port\_probe_out\ca_plugins_g4\sd_fastpm_DiT.zip | 59.8 | B |
| 154 | zimage-port\_probe_out\ctrl_vendor\plugins\sd_fastpm_DiT.zip | 59.8 | B |
| 155 | zimage-port\_probe_out\dd100\plugins\sd_fastpm_DiT.zip | 59.8 | B |
| 156 | zimage-port\_probe_out\ca_plugins_g3\sd_fastpm_DiT.zip | 59.8 | B |
| 157 | zimage-port\_probe_out\ca_plugins\sd_fastpm_DiT.zip | 59.8 | B |
| 158 | zimage-port\_probe_out\ca_plugins_g1\sd_fastpm_DiT.zip | 59.8 | B |
| 159 | zimage-port\_probe_out\ca_plugins_g2\sd_fastpm_DiT.zip | 59.8 | B |
| 160 | zimage-port\_probe_out\dd103t\plugins\sd_fastpm_DiT.zip | 59.8 | B |
| 161 | zimage-port\_probe_out\dd85\nb4\sd_fastpm_DiT.zip | 59.8 | B |
| 162 | zimage-port\_probe_out\dd81\B_vendor_cast\sd_fastpm_DiT.zip | 59.8 | B |
| 163 | zimage-port\_probe_out\dd85\nb5\sd_fastpm_DiT.zip | 59.8 | B |
| 164 | zimage-port\_probe_out\dd112p\plugins\sd_fastpm_DiT.zip | 59.8 | B |
| 165 | zimage-port\_probe_out\dd81\A_only_vendor\sd_fastpm_DiT.zip | 59.8 | B |
| 166 | zimage-port\_probe_out\dd_swap\sd_fastpm_DiT_B.zip | 59.8 | B |
| 167 | zimage-port\_probe_out\dd_swap\broot\sd_fastpm_DiT.zip | 59.8 | B |
| 168 | zimage-bfp16-npu\_scratch\attn_impl\t3b_out\t3b_proj_n3840_ref.npz | 57.7 | B |
| 169 | zimage-bfp16-npu\_scratch\image\t6_refcache\noise_refiner.0\to_out.npy | 56.3 | B |
| 170 | zimage-bfp16-npu\_scratch\image\t6_refcache\noise_refiner.0\to_k.npy | 56.3 | B |
| 171 | zimage-bfp16-npu\_scratch\image\t6_refcache\noise_refiner.0\to_v.npy | 56.3 | B |
| 172 | zimage-bfp16-npu\_scratch\image\t6_refcache\noise_refiner.0\to_q.npy | 56.3 | B |
| 173 | zimage-bfp16-npu\_scratch\image\t6_refcache\noise_refiner.1\to_out.npy | 56.3 | B |
| 174 | zimage-bfp16-npu\_scratch\image\t6_refcache\noise_refiner.1\to_k.npy | 56.3 | B |
| 175 | zimage-bfp16-npu\_scratch\image\t6_refcache\noise_refiner.1\to_v.npy | 56.3 | B |
| 176 | zimage-bfp16-npu\_scratch\image\t6_refcache\noise_refiner.1\to_q.npy | 56.3 | B |
| 177 | zimage-bfp16-npu\_scratch\image\t6_refcache\context_refiner.0\to_k.npy | 56.3 | B |
| 178 | zimage-bfp16-npu\_scratch\image\t6_refcache\context_refiner.1\to_v.npy | 56.3 | B |
| 179 | zimage-bfp16-npu\_scratch\image\t6_refcache\context_refiner.0\to_out.npy | 56.3 | B |
| 180 | zimage-bfp16-npu\_scratch\image\t6_refcache\context_refiner.0\to_v.npy | 56.3 | B |
| 181 | zimage-bfp16-npu\_scratch\image\t6_refcache\context_refiner.0\to_q.npy | 56.3 | B |
| 182 | zimage-bfp16-npu\_scratch\image\t6_refcache\context_refiner.1\to_out.npy | 56.3 | B |
| 183 | zimage-bfp16-npu\_scratch\image\t6_refcache\context_refiner.1\to_k.npy | 56.3 | B |
| 184 | zimage-bfp16-npu\_scratch\image\t6_refcache\context_refiner.1\to_q.npy | 56.3 | B |
| 185 | _pylibs\llvm-aie\llvm-aie\bin\llc.exe | 55.9 | C |
| 186 | _pylibs\llvm-aie\llvm-aie\bin\opt.exe | 55.8 | C |
| 187 | zimage-bfp16-npu\_scratch\perfc\out\perfc_r10t_M1024_K3840_N10240_m32_kk8_n512_Kc640_d21b2_full_events.json | 55.6 | B |
| 188 | zimage-bfp16-npu\_scratch\attn_impl\t3b_out\t3b_proj_n3840.prj\input_with_addresses.mlir | 52.8 | B |
| 189 | zimage-bfp16-npu\_scratch\attn_impl\t3b_out\t3b_proj_w4.prj\input_with_addresses.mlir | 52.1 | B |
| 190 | k2h-llama.cpp\build-vulkan\bin\Release\ggml-vulkan.dll | 51.9 | B |

统计：190 个文件，合计 107,081 MiB（104.6 GiB）逻辑大小；其中 C 档 43 个（模型/权重/工具链/备份/门引用），B 档 147 个（实验产物）。

---

## 附录 B — 方法学与复算方式

1. 遍历：PowerShell 5.1 无 EnumerationOptions，改用自建栈 + System.IO.Directory::EnumerateFileSystemEntries + System.IO.File::GetAttributes；显式跳过 ReparsePoint。全树 86,089 文件，单次遍历 19.6–34.0 s。
2. 聚合：一次遍历同时产出顶层/二级/三级/.prj/__pycache__/根日志/>50MiB 六组聚合，避免多次 Get-ChildItem -Recurse。
3. 盘况：System.IO.DriveInfo::new("D")（Get-PSDrive / Get-CimInstance / fsutil 均被沙箱拒绝）。
4. 硬链接：py312 的 os.stat().st_nlink + os.path.samefile 判定；结论见 §5.2。
5. 门校验：t3d_baseline.py 源码解析（NPU/ATT/OUT_A/HERE 四个基）+ t3d_baseline.json 三个桶逐条比对；NEVER 名单来自 CLEANUP-INVENTORY.md §4/§7 与 CLEANUP-INVENTORY-2.md §7；脚本引用用 ripgrep 在 *.py 内检索。
6. 零删除：本单未删除/移动/改名任何审计对象；唯一写入是本文件（及一次误建临时文件的即时删除，见 §5.5）。

## 附录 C — 本单引用的一手材料

- docs/CLEANUP-EXECUTION.md（6 项 + bfp16_weights_v2 补记）
- docs/CLEANUP-INVENTORY.md、docs/CLEANUP-INVENTORY-2.md（可恢复性分类与 NEVER 名单）
- docs/MANIFEST.md（备份集 7 载荷 / 24.399 GiB / 恢复步骤）
- _scratch/image/t3d_baseline.py + t3d_baseline.json（门①）
- _scratch/vae/T21-3A-AUDIT.md（92 个 xclbin 的纸面账：55 conv + 24 GN + 13 UP）
- HANDOFF-SUPPLEMENT.md §E（r3v_/r3img_ 的删除建议）
- docs/zimage-bfp16-npu-questions.md:95（E 项"已落实"的假阴性）
- webui/config.py、webui/cache.py、webui/build_shape.py（产品路径与 sha256 清单）
- zimage-port/_scan_clean.py、_clean_exp.py、_cleanup.py、cleanup_large_files.py、_CLEANUP-LIST.txt（清理工具考古）


---

## 附录 D — 断点续跑复核（2026-10-19 10:19，崩溃恢复）

本文件已于 **09:47:06** 落盘完成（60,448 B / 717 行），随后会话因断网中断，但**文件本身完整**：13 个一级标题齐全、附录 A 的 190 行齐全、末行完整、UTF-8 无乱码。

- **断点期间无数据丢失**：本单的全部产物就是这一个文件；本单未运行任何器件/编译，不存在 calls.json 一类运行态产物。
- **10:19:34 复核盘况**：D: free = 938,491,904 B = **0.874 GiB**（较 09:35 的 0.89 GiB 再降 0.016 GiB，仍在 <1 GB 区间）。
- **10:19 漂移复核**：`_scratch\vae` 由 680 文件 / 233.06 MiB → **704 文件 / 242.87 MiB**（3b1 仍在写）；`_scratch\image` 与 `_scratch\block` 无变化（7,777.00 / 4,496.33 MiB）。§5.3 的漂移声明继续有效。
- **git**：09:47 落盘时 HEAD=6be3320；10:19 复核时工作树 899 条变更（其它代理陆续提交 t21_mvdocs2、t21_3b1 585672c 等）。本单只新增 `docs/T21-DISKAUDIT.md` 一个文件，并以独立提交 7639eda 落库（第二次提交仅修正本条 HEAD 记述）。
- **结论不变**：§9.4 的一句话对策与“删除冻结、先查 9.44 GB 去向”的建议继续有效。
