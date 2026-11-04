# T21 全域磁盘普查（public_work / 只读，零删除）

**任务号** t21_diskmap ｜ **模式** 只读全域普查 ｜ **本文件是本会话唯一新建文件**

> ## ⚠ 范围声明
> **本报告的范围 = 工作区 `D:\Project\deepseek_harness\public_work`。**
> **D 盘其它区域（D:\ 下 public_work 之外的目录、其它盘符）不在本单范围：不测、不归因、不写入本报告。**
> 任何"D 盘 651 GiB 都去哪了"的问题，本报告**不作回答**；本报告只回答"public_work 这 98.55 GiB 物理占用由什么构成"。

**时间锚点**
- 首轮全量扫描：**2026-10-19T11:35:38 – 11:35:41**（聚合口径）＋ 11:36–11:37（inode 去重 / 扩展名 / 全深度目录树三遍）
- 复核扫描：**2026-10-19T12:35:23 – 12:35:40**
- 两轮结果**完全一致**：`130,601,117,006` 字节 / `86,185` 文件；树内最新文件 mtime = **2026-10-19 11:22:39** → 两次扫描之间树是静止的，本报告是一份**自洽快照**，不是两次读数拼起来的。
- 盘况锚点（仅作余量参考，不作归因）：12:34:53 `free = 981,938,176 B（0.914 GiB）`；12:35:40 `free = 981,921,792 B（0.9145 GiB）`。11:35:41 曾读到 `869,138,432 B（0.810 GiB）` → 期间有并发代理释放了约 **113 MiB**。

**口径**
- GiB = 2^30、MiB = 2^20；所有数字均为**字节实测**，不是估算，除非显式标注"≈ / 估"。
- **路径合计**＝沿目录树累加（同一物理文件有多个名字时**重复计入**）。
- **物理**＝按 NTFS 文件 ID（VolumeSerialNumber + FileIndex）去重后的真实占用。
- 不跟随 reparse point（Junction / Symlink），遇到即跳过并单列。

---

## 0. 结论先行

1. **public_work 路径合计 121.632 GiB（130,601,117,006 B / 86,185 文件），inode 去重后物理占用 98.548 GiB（105,815,386,676 B）。** 两者相差 **23.084 GiB**，全部来自硬链接（263 个 inode 的 nlink>1），**不是"文件消失了"，是"同一份数据有多个名字"**。
2. **"工作区已占 100 GB+"成立，并且账是闭合的。** 98.548 GiB 物理占用逐块归位：权重/模型 **55.92 GiB（56.7%）**＋zimage 实验产物区 **26.10 GiB（26.5%）**＋备份集独占载荷 **10.07 GiB（10.2%）**＋工具链 **3.79 GiB（3.8%）**＋其它项目 **2.67 GiB（2.7%）**，残差 0.00 GiB。
3. **public_work 内没有"隐藏的巨块"。** 全部非 zimage 主题（llama.cpp / k2h / deepseek-harness / research / 加密研究…）合计只有 **6.37 GiB（占路径 5.2% / 占物理 6.5%）**。所谓"大头去向不明"在 public_work 范围内**不成立**——如果 D 盘确实有 550+ GiB 说不清，那部分**在 public_work 之外**，按本单范围不予追查。
4. **旧审计 T21-DISKAUDIT 的一级目录表与本单逐行吻合（最大偏差 0.01 GiB），但它的 A/B/C 分档不闭合。** A+B+C = 110.26 GiB，对不上它自己的路径合计 121.62 GiB（差 11.4），也对不上物理 98.55 GiB（多 11.7）。原因有三：①C1 备份集与其 3 个硬链接成员**重复计 14.67 GiB**；②`_probe_out\weights.bfp16`（6.44 GiB）同时被算进"B 档最大单项"和"C3"，双计；③硬链接虚增被低估 **9.15 GiB**（旧文估 13.9，实测 23.08）。旧文把这 11.4 GiB 差额甩给了"其它主题区长尾"，但真正**完全没分档**的非 zimage 只有 **1.79 GiB**（详见 §5.2）。
5. **可立即释放的仍然只有 A 档 743.1 MiB**；B 档（需批准）**17,405.0 MiB = 17.0 GiB**。C 档 103.909 GiB（路径）/ 80.910 GiB（物理）不可删。
6. **144 个 ACL 拒绝目录，144/144 连 `icacls` 都返回 `Access is denied`（rc=5）**，只能计数、无法测容，下界 0。**而旧审计把这批目录写成了"空目录、全部 0 字节"**——数量高度重合（旧文 37+9+45+12，本单 48+9+45+6+…），这几乎可以确定旧文的"0 字节"是**把"测不到"当成了"空"**。
7. **4 个 Junction 别名不产生任何虚增**：`zimage-port\npu-conversion\{_models,_probe_out}`、`zimage-bfp16-repo\{_models,_probe_out}` 都指向 `zimage-port\_models` / `_probe_out`，本单按 reparse point 跳过，计 0 字节。
8. **零删除、零修改。** 唯一新建文件 = 本报告；另在 `%TEMP%`（**不在工作区**）落了 4 个中间 JSON，已在下文 §6.5 披露。

---

## 1. 总量对账（U1）

### 1.1 一级目录逐个实占

按物理占用降序；`路径 GiB` 与 `物理 GiB` 的差即该区持有的"别名"体积。

| # | 一级项 | 文件数 | 路径 GiB | 物理 GiB | 最后写入 |
|---:|---|---:|---:|---:|---|
| 1 | zimage-port | 12,104 | 51.911 | 44.215 | 10-19 09:14 |
| 2 | zimage-bfp16-npu | 37,837 | 31.457 | 23.568 | 10-19 11:22 |
| 3 | _backup_staging | 7 | 24.399 | 24.399 | 10-19 08:16 |
| 4 | zimage-te-qwen3-4b-bf16.gguf（根文件） | 1 | 7.498 | 0.000 \* | 10-18 10:36 |
| 5 | _pylibs | 4,431 | 1.536 | 1.536 | 10-14 15:07 |
| 6 | _wheels | 12 | 1.283 | 1.283 | 10-14 14:21 |
| 7 | llama.cpp-0.5.0 | 6,209 | 0.818 | 0.818 | 10-05 11:24 |
| 8 | _git_backup_pre_purge | 1,959 | 0.793 | 0.793 | 10-15 08:19 |
| 9 | k2h-llama.cpp | 7,605 | 0.686 | 0.686 | 10-05 19:41 |
| 10 | models | 1 | 0.620 | 0.620 | 10-05 18:51 |
| 11 | ryzen_ai_nuget_1.8.0 | 3 | 0.257 | 0.257 | 07-21 06:40 |
| 12 | research | 2,223 | 0.103 | 0.103 | 09-11 09:24 |
| 13 | deepseek-harness | 10,351 | 0.089 | 0.089 | 09-11 11:22 |
| 14 | _git_backup_pre_purge2 | 434 | 0.085 | 0.085 | 10-15 09:02 |
| 15 | aie-rt | 334 | 0.019 | 0.019 | 10-10 20:59 |
| 16 | wallets | 1,861 | 0.018 | 0.018 | 09-11 09:04 |
| 17 | .npm-cache | 42 | 0.017 | 0.017 | 09-11 09:02 |
| 18 | arweave_research | 83 | 0.014 | 0.014 | 09-10 08:48 |
| 19 | build-vulkan.log（根文件） | 1 | 0.010 | 0.010 | 10-05 11:24 |
| 20 | _faucet | 264 | 0.009 | 0.009 | 09-11 09:13 |
| 21 | _net | 115 | 0.004 | 0.004 | 09-11 09:15 |
| 22 | _sn | 9 | 0.002 | 0.002 | 09-11 09:05 |
| 23 | 其余 60 项（_ln / _urls / _earn / _plat / ops / dsh-subagent-control / zimage-bfp16-repo / _acl_recovery / .jspace / _piptmp / 48 个 pip-tmp-quark 目录 / 根散件 .mjs .md .py） | ~1,200 | 0.186 | 0.186 | 09-11 – 10-19 |
| | **合计** | **86,185** | **121.632** | **98.548** | — |

\* 根目录那个 GGUF 的物理归属记在 `_backup_staging\01_…`（同一 inode 的三个名字里，字典序最小的是备份集那份），故此处为 0.000，避免重复计。

### 1.2 口径桥接

```
路径合计                 121.632 GiB   100.0%
  − 硬链接重复计量       −23.084 GiB    19.0%   （263 个 inode，nlink>1）
= 物理占用                98.548 GiB    81.0%
  − junction 别名          −0.000 GiB     0.0%   （4 个，全部指向树内，已跳过）
```

- 目录数：**13,285**（含根）；真实空目录 **308**；ACL 拒绝目录 **144**（见 §6.1）。
- 路径合计 / 物理的差 **23.084 GiB**，逐项见 §3。

### 1.3 每 GiB 的归属（物理口径，残差 0.00）

| 归属块 | 物理 GiB | 占物理 | 构成（括号内为归属明细） |
|---|---:|---:|---|
| **权重与模型** | **55.92** | 56.7% | `_models\Z-Image-Turbo` 30.64 ＋ `weights.bfp16` 两个独立 inode 12.88（6.44×2） ＋ TE GGUF 7.50（3 个名字 1 份） ＋ `te_npu_weights.bin` 3.81 ＋ K2-Horizon GGUF 0.62 ＋ `vae_weights.bfp16` 0.09 ＋ vendor `.rai` 0.39 |
| **zimage 实验/产物区** | **26.10** | 26.5% | `zimage-bfp16-npu\_scratch` 19.09 ＋ `_probe_out` 非权重部分 6.43 ＋ 仓库杂项（webui 非模型 / .git / kernel 目录）0.33 ＋ `zimage-port` 杂项 0.25 |
| **备份集独占载荷** | **10.07** | 10.2% | `05_runtime-payloads-gates.tar` 7.01 ＋ `04_toolchain-pylibs-llama.tar` 3.05 ＋ `zimage-code-docs.bundle` 0.004（**注**：备份集另外 14.33 GiB 是上面"权重与模型"那几份的同 inode 别名，已计入权重块，不重复计） |
| **工具链** | **3.79** | 3.8% | `_pylibs` 1.54 ＋ `_wheels` 1.28 ＋ `_probe_out\ryzenai_nuget` 0.71 ＋ `ryzen_ai_nuget_1.8.0` 0.26 |
| **其它项目/杂项** | **2.67** | 2.7% | llama.cpp-0.5.0 0.82 ＋ k2h-llama.cpp 0.69 ＋ `_git_backup_pre_purge(2)` 0.88 ＋ deepseek-harness 0.09 ＋ research 0.10 ＋ 其余 0.09 |
| **合计** | **98.55** | 100.0% | — |

### 1.4 zimage 域 vs 非 zimage 域

| 域 | 路径 GiB | 物理 GiB | 占物理 |
|---|---:|---:|---:|
| zimage 域（zimage-port ＋ zimage-bfp16-npu ＋ _backup_staging ＋ 根 GGUF） | 115.265 | 92.182 | 93.5% |
| **非 zimage 域**（工具链 _pylibs 1.54 / _wheels 1.28 / ryzen_ai_nuget 0.26 ＋ K2 权重 0.62 ＋ llama.cpp-0.5.0 0.82 / k2h-llama.cpp 0.69 / _git_backup_pre_purge(2) 0.88 / deepseek-harness 0.09 / research 0.10 ＋ wallets / arweave_research / aie-rt / _faucet / _net / _sn / _ln / _urls / _earn / _plat / ops / dsh-subagent-control / .npm-cache / _piptmp / 48 个 pip-tmp-quark 目录 / 根散件 ≈0.19） | **6.367** | **6.366** | **6.5%** |

> 结论：**"其它项目"不是本次的隐藏大头，它只有 6.37 GiB。** 若要腾出 10 GiB 以上，动不到"其它项目"，只能动 B 档（实验产物 17.0 GiB）或 C 档。旧审计 §1.1 表里那句"其余（aie-rt、wallets…）~75 MiB"也是对的，本单实测 ~95 MiB。

---

## 2. 大件清单（U2）

### 2.1 > 1 GiB 文件全清单（14 个）

路径相对 `public_work\`；`nlink` 由 `GetFileInformationByHandle` 实测。

| # | 路径 | GiB | inode 组 | nlink | 别名 / Junction | 档 |
|---:|---|---:|---|---:|---|---|
| 1 | zimage-port\_models\Z-Image-Turbo\transformer\diffusion_pytorch_model-00002-of-00003.safetensors | 9.289 | 独立 | 1 | — | C |
| 2 | zimage-port\_models\Z-Image-Turbo\transformer\diffusion_pytorch_model-00001-of-00003.safetensors | 9.289 | 独立 | 1 | — | C |
| 3 | zimage-te-qwen3-4b-bf16.gguf | 7.498 | G-A | **3** | 本表 #4 #5 | C |
| 4 | _backup_staging\01_te-gguf-zimage-te-qwen3-4b-bf16.gguf | 7.498 | G-A | **3** | 本表 #3 #5 | C |
| 5 | zimage-bfp16-npu\webui\models\zimage-te-qwen3-4b-bf16.gguf | 7.498 | G-A | **3** | 本表 #3 #4 | C |
| 6 | _backup_staging\05_runtime-payloads-gates.tar | 7.013 | 独立 | 1 | — | C |
| 7 | _backup_staging\02_weights.bfp16 | 6.439 | G-B | **2** | 本表 #8 | C |
| 8 | zimage-port\weights.bfp16 | 6.439 | G-B | **2** | 本表 #7 | C |
| 9 | zimage-port\_probe_out\weights.bfp16 | 6.439 | **独立** | 1 | — | C |
| 10 | zimage-port\_models\Z-Image-Turbo\transformer\diffusion_pytorch_model-00003-of-00003.safetensors | 4.351 | 独立 | 1 | — | C |
| 11 | zimage-bfp16-npu\_out\textenc\te_npu_weights.bin | 3.807 | 独立 | 1 | — | C |
| 12 | zimage-port\_models\Z-Image-Turbo\text_encoder\model-00002-of-00003.safetensors | 3.714 | 独立 | 1 | — | C |
| 13 | zimage-port\_models\Z-Image-Turbo\text_encoder\model-00001-of-00003.safetensors | 3.686 | 独立 | 1 | — | C |
| 14 | _backup_staging\04_toolchain-pylibs-llama.tar | 3.054 | 独立 | 1 | — | C |

- **路径合计 86.014 GiB；inode 去重后 64.579 GiB**（差 21.435 GiB 全来自 G-A 与 G-B 两组）。
- **没有任何 >1 GiB 文件位于 Junction 路径下**：4 个 Junction 全部指向 `_models` / `_probe_out`，本单跳过，天然不产生别名。
- 14 个里有 **13 个是 C 档**；唯一可能翻盘的是 #9 `_probe_out\weights.bfp16`，但据 t21_wcensus 已证伪"同内容推定"（SHA256 不同），它是**唯一旧版**，删即销毁旧构建 → 仍判 C。

### 2.2 > 100 MiB 目录 top40（depth ≤ 3，逐层展开）

全树 depth≤3 且 >100 MiB 的目录共 **59 个**，合计 263.95 GiB（含重复计名）。按体积降序取前 40：

| # | 目录（相对 public_work） | MiB | 深度 |
|---:|---|---:|---:|
| 1 | zimage-port | 53,157.18 | 1 |
| 2 | zimage-bfp16-npu | 32,211.52 | 1 |
| 3 | zimage-port\_models | 31,371.82 | 2 |
| 4 | zimage-port\_models\Z-Image-Turbo | 31,371.80 | 3 |
| 5 | _backup_staging | 24,984.53 | 1 |
| 6 | zimage-bfp16-npu\_scratch | 19,636.15 | 2 |
| 7 | zimage-port\_probe_out | 14,930.73 | 2 |
| 8 | zimage-bfp16-npu\_scratch\image | 7,777.00 | 3 |
| 9 | zimage-bfp16-npu\webui | 7,703.82 | 2 |
| 10 | zimage-bfp16-npu\webui\models | 7,678.30 | 3 |
| 11 | zimage-bfp16-npu\_scratch\block | 4,496.33 | 3 |
| 12 | zimage-bfp16-npu\_out | 3,899.85 | 2 |
| 13 | zimage-bfp16-npu\_out\textenc | 3,898.94 | 3 |
| 14 | zimage-bfp16-npu\_scratch\perfc | 3,849.18 | 3 |
| 15 | zimage-port\_probe_out\zimage_loop | 3,476.15 | 3 |
| 16 | _pylibs | 1,573.22 | 1 |
| 17 | zimage-bfp16-npu\_scratch\util | 1,508.86 | 3 |
| 18 | _wheels | 1,314.13 | 1 |
| 19 | _pylibs\llvm-aie | 948.51 | 2 |
| 20 | _pylibs\llvm-aie\llvm-aie | 948.33 | 3 |
| 21 | zimage-bfp16-npu\_scratch\attn_impl | 936.89 | 3 |
| 22 | llama.cpp-0.5.0 | 837.58 | 1 |
| 23 | _git_backup_pre_purge | 811.90 | 1 |
| 24 | _git_backup_pre_purge\objects | 811.55 | 2 |
| 25 | zimage-port\_probe_out\ryzenai_nuget | 729.89 | 3 |
| 26 | k2h-llama.cpp | 702.39 | 1 |
| 27 | llama.cpp-0.5.0\build-vulkan | 672.91 | 2 |
| 28 | models | 635.32 | 1 |
| 29 | _pylibs\mlir_aie | 609.71 | 2 |
| 30 | k2h-llama.cpp\build-vulkan | 505.73 | 2 |
| 31 | zimage-bfp16-npu\weights | 401.31 | 2 |
| 32 | zimage-bfp16-npu\_scratch\weights | 401.31 | 3 |
| 33 | zimage-bfp16-npu\vendor_artifacts | 399.48 | 2 |
| 34 | _git_backup_pre_purge\objects\64 | 369.26 | 3 |
| 35 | k2h-llama.cpp\build-vulkan\ggml | 332.54 | 3 |
| 36 | _pylibs\mlir_aie\bin | 302.53 | 3 |
| 37 | zimage-port\_probe_out\noinit | 300.08 | 3 |
| 38 | llama.cpp-0.5.0\build-vulkan\ggml | 279.15 | 3 |
| 39 | ryzen_ai_nuget_1.8.0 | 262.69 | 1 |
| 40 | zimage-port\_probe_out\dd85 | 260.99 | 3 |

**深一层展开（>500 MiB 的实质载体，任意深度）**

| 目录 | GiB |
|---|---:|
| zimage-port\_models\Z-Image-Turbo\transformer | 22.93 |
| zimage-bfp16-npu\_scratch\image\t3real_prep（60 文件） | 3.71 |
| zimage-bfp16-npu\_scratch\perfc\out（12,671 文件） | 3.76 |
| zimage-bfp16-npu\_scratch\block\step6_out（1,231 文件） | 1.99 |
| zimage-bfp16-npu\_scratch\util\out（2,677 文件） | 1.41 |
| zimage-bfp16-npu\_scratch\block\out（1,442 文件） | 1.01 |
| _pylibs\llvm-aie\llvm-aie\bin | 0.91 |
| zimage-bfp16-npu\_scratch\image\t6_refcache（60 文件） | 2.67 |

### 2.3 100 MiB – 1 GiB 文件

**61 个，合计 10.52 GiB**。分布：`_scratch\image\t6_refcache` 的 12 个 150 MiB `w*.npy`、`_scratch\block\step6_out` 的 15 个 113.9 MiB `*_io.npz`、`_scratch\image\t3real_prep` 的 20 个 84.4 MiB `layers.*_bg.npy`、`_probe_out\*` 的 `.onnx`/`w.bin`（150 MiB ×4）、`_pylibs`/`_wheels` 的工具链 exe/dll、`k2h-llama.cpp\build-vulkan\ggml\src\ggml-vulkan\mul_mm.comp.cpp`（123.3 MiB，编译器生成的巨型 switch）。

### 2.4 扩展名聚合（全树，路径口径）

| 扩展名 | 文件数 | GiB | 扩展名 | 文件数 | GiB |
|---|---:|---:|---|---:|---:|
| .safetensors | 8 | 30.62 | .whl | 12 | 1.28 |
| .gguf | 42 | 23.26 | .exe | 103 | 1.22 |
| .bfp16 | 4 | 19.41 | .dll | 54 | 0.96 |
| .npy | 1,346 | 10.35 | 无扩展名 | 8,710 | 0.94 |
| .tar | 3 | 10.07 | .onnx | 88 | 0.88 |
| .npz | 816 | 7.54 | .rai | 8 | 0.83 |
| .bin | 2,148 | 4.51 | .mlir | 965 | 0.66 |
| .const | 798 | 3.23 | .json | 8,539 | 0.46 |
| .zip | 114 | 2.27 | .cpp | 1,808 | 0.43 |

> `.gguf` 的 23.26 GiB 是**路径口径**（TE GGUF 的 3 个名字占了 22.49），物理只有 7.50 + 0.62。

---

## 3. inode 去重：路径合计 vs 真实物理（U3③）

**263 个 inode 的 nlink > 1，虚增 24,785,730,330 B = 23.084 GiB。**

| 组 | 名字数 | 单份 GiB | 虚增 GiB | 位置 |
|---|---:|---:|---:|---|
| G-A：TE GGUF | 3 | 7.498 | **14.997** | 根 ＋ `webui\models` ＋ `_backup_staging\01` |
| G-B：weights.bfp16（交付版） | 2 | 6.439 | **6.439** | `zimage-port\weights.bfp16` ＋ `_backup_staging\02` |
| G-C：vendor .rai | 2 | 0.390 | **0.390** | `zimage-bfp16-npu\vendor_artifacts` ＋ `_backup_staging\03` |
| G-D：`_probe_out` 内部同一化 | 260 组 | — | **1.258** | 见下 |
| **合计** | 263 组 | — | **23.084** | |

**G-D 是什么**：`_probe_out` 内部有 260 组同 inode 文件，最大的一类是 `zimage_loop\{out_op1,t_a4,t_w1,…}\cache\blocks.*{w1,w2,w3}MatMul_*.const` —— 同一份 43.36 MiB 的权重常量被 4 个 run 目录各自"复制"（实为硬链接），另有 `gx_plugins1\sd_fastpm_DiT.zip` 与 `gl_plug\sd_fastpm_DiT.zip`（59.79 MiB）同 inode。合计虚增 1.258 GiB。

**Junction 别名（不计入虚增，因为根本没走进去）**

| 路径 | 目标 |
|---|---|
| zimage-port\npu-conversion\_models | `…\public_work\zimage-port\_models` |
| zimage-port\npu-conversion\_probe_out | `…\public_work\zimage-port\_probe_out` |
| zimage-bfp16-repo\_models | `…\public_work\zimage-port\_models` |
| zimage-bfp16-repo\_probe_out | `…\public_work\zimage-port\_probe_out` |

→ 4 个 Junction、0 字节。**若跟随它们遍历，会凭空多出 45.2 GiB 的双计**（_models 30.64 + _probe_out 14.58，各多算一遍，npu-conversion 与 bfp16-repo 两处共 4 个入口）。这正是"路径合计"最容易虚增的地方，本单已排除。

**释放量口径提醒**：G-A / G-B / G-C 这三组里，**删掉任意一个名字都不释放 1 字节**，只有把同 inode 的所有名字一起删才释放。旧审计 §8.1 把 `_probe_out\weights.bfp16` 标成"B 档最大单项 6.44 GiB 可释放"——那一条是**对的**（它是 nlink=1 的独立 inode），但同一份报告在 C3 里又把它算进 C，口径打架。

---

## 4. 分档候选（U3，按释放量降序）

分档口径与旧审计一致：**A 纯可再生 / B 需批准 / C 永不可删**。所有数字为**路径口径实测**；A/B 两项几乎全为 nlink=1，故释放量≈路径体积。三者合计正好 **121.632 GiB**（路径）／**98.548 GiB**（物理）。

### 4.1 汇总

| 档 | 文件数 | 路径 MiB | 路径 GiB | 物理 GiB | 可释放（物理） |
|---|---:|---:|---:|---:|---:|
| **A 纯可再生** | 20,936 | **743.1** | 0.726 | 0.726 | **0.726 GiB** |
| **B 需批准** | 49,938 | **17,405.0** | 16.997 | 16.912 | **16.912 GiB** |
| **C 永不可删** | 15,311 | **106,402.8** | 103.909 | 80.910 | 0 |
| 合计 | 86,185 | 124,550.9 | 121.632 | 98.548 | — |

> A 档文件数很多（20,936）是因为 `.prj` 编译目录里全是几千个小文件；B 档文件数多是因为构建树（llama/k2h）与实验输出；C 档文件数最少但体积最大。

### 4.2 A 档 — 纯可再生（743.1 MiB）

| # | 路径 | MiB | 依据 | 可恢复性 | 风险 | 时机 |
|---:|---|---:|---|---|---|---|
| A1 | `_scratch\perfc\out\**\*.prj`（10,522 文件） | 227.61 | 休眠至 10-17 18:01 | 重跑 perfc 重编 | 低 | 立即 |
| A2 | `_scratch\block\step6_out\**\*.prj` | 217.62 | 休眠至 10-17 17:20 | 重编 step6 变体 | 低 | 立即 |
| A3 | `_scratch\attn_impl\t5_out\**\*.prj` | 133.84 | .prj 为 10-15 产物；同目录 10-19 02:17 仍有写 | 重编 | **中：t5 昨夜仍在动** | 待 t5 定稿 |
| A4 | `_scratch\util\out\**\*.prj` | 53.74 | 休眠至 10-17 17:29 | 重编 | 低 | 立即 |
| A5 | `_scratch\textenc\out_layer1\**\*.prj` | 19.52 | 休眠至 10-16 19:57 | 重编 | 低 | 立即 |
| A6 | `_scratch\textenc\out_xn\**\*.prj` | 16.79 | 休眠至 10-16 20:54 | 重编 | 低 | 立即 |
| A7 | `_scratch\perf\out\**\*.prj` | 12.15 | 休眠至 10-16 21:40 | 重编 | 低 | 立即 |
| A8 | 仓库 `memtile2\out` / `dit2\out` ＋ `_scratch` 同名副本 里的 .prj（旧审计只算了 _scratch 那两份） | 45.01 | 顶层旧副本，无近期写入 | 重编 | 低 | 立即 |
| A9 | `__pycache__\*.pyc`（847 文件，152 目录） | 16.76 | Python 自动再生 | 无成本 | 无 | 立即 |
| A10 | `*.lock`（**4 个**，旧文记 599 —— 现已几乎清空） | 0.06 | 空锁文件 | 进程自建 | 无 | 立即 |
| A11 | 空目录 308 个（pip-install / tmp* / quark 等，**0 字节**） | 0.00 | 无内容 | — | 无 | 随时（收益 0） |
| | **合计** | **743.1** | 其中**可立即执行（不含 A3）= 609.3 MiB** | | | |

**A 档未纳入的 .prj**（虽是 .prj 但属 C）：`attn_impl\t3b_out` 258.06（门①冻结参考）、`block\out` 38.13（含门① ffn xclbin 与交付 .ll）、`block\t4_out` 36.25（`webui/config.py:31` 硬编码 T4_OUT）、`vae\out` 11.18（3b1 活跃区＋门③证据）。

### 4.3 B 档 — 需批准（17,405.0 MiB / 16.997 GiB）

按释放量降序。**"释放"一栏是物理释放量**（B 档的 nlink 虚增只有 0.085 GiB，几乎全可释放）。

| # | 路径 / 项 | MiB | 依据 | 可恢复性 | 风险 | 建议 |
|---:|---|---:|---|---|---|---|
| B1 | `zimage-port\_probe_out` 探针残留（14,930.7 − zimage_loop 3,476.1 − ryzenai_nuget 729.9 − weights.bfp16 6,593.7） | **4,131.0** | 10-09→10-12 的结论型探针，多数目录自带 verdict json | 重跑探针（分钟级）；部分 vendor zip 不可再下载 | 中：含 30+ 个 87 MiB 的 plugins 目录 | 逐目录签字 |
| B2 | `_scratch\perfc` 非 .prj（_io.npz / _C_raw.npy / _events.json） | **3,621.4** | perfc 12,671 文件，休眠 2 天 | 重跑 perf（耗器件时间） | 中：脚本 `t2b_slots.py`/`t2c_read.py` 硬编码 glob `perfc_*_C_raw.npy` | 保留 _C_raw.npy 或同步改脚本 |
| B3 | `_scratch\block` 非 .prj（step6_out 1,818.7 ＋ step5_out 613.0 ＋ t10_out 387.0 ＋ 长尾） | **2,922.0** | 休眠至 10-17 17:20 | 重跑 step6/step5 | 中：`t3_h1_slice.py` 硬编码读 `s6_h1_*.{npz,npy}`（3 个 8.4 MiB 必须保留） | 确认归档后 |
| B4 | `_scratch\util` 非 .prj（out 1,388 ＋ out_r5e* 67） | **1,455.0** | 休眠至 10-17 17:29 | 重跑 util 实验 | 中 | 负责人批准后 |
| B5 | `_scratch\image` 散落（非 t3real_prep / t6_refcache），含 r3v_*/r3img_* 326.19 | **1,247.9** | 混合；r3v/r3img 全部 mtime 10-17 08:48–08:52 | t3real_prep 可重算；r3v 一条命令重跑 | **高：t3real_* 系是 PSNR 参考上游，必须逐文件核** | 仅删明确非 t3real*/t3d* 项 |
| B6 | `_git_backup_pre_purge` ＋ `_git_backup_pre_purge2` | **899.0** | 隐藏的 purge 前 git 备份 | **不可**从现仓库恢复 | 高：可能是历史保险 | 负责人裁决 |
| B7 | `llama.cpp-0.5.0`（构建树 837.6） | **837.6** | 其它主题；10-05 后无写入 | 重新 cmake 构建（需 Vulkan SDK） | 中：与 zimage 无关，但不属本仓库 | 负责人裁决 |
| B8 | `k2h-llama.cpp`（构建树 702.4） | **702.4** | 其它主题；10-05 后无写入 | 同上 | 中 | 负责人裁决 |
| B9 | `_scratch\weights\block0.{npz,coalesced.bin}`（与仓库 `weights\` 同尺寸**两份物理副本**，samefile=False） | **401.3** | 可重导出 | `extract.py --out` 重导 | 中：若脚本指向 _scratch 路径 | 核对引用后 |
| B10 | `_scratch\textenc` 非 .prj | **219.2** | 休眠至 10-16 | 重跑 textenc | 中 | 批准后 |
| B11 | `_scratch\attn_impl` 非 .prj（t5_out 93.5 ＋ out 61.0） | **157.0** | t5_out 10-19 02:17 有写 | 重跑 t5 | 高：在动 | 不建议现在 |
| B12 | `_scratch` 其它（memtile2 / dit2 / kernels / perf 非 .prj / pipeline / oracle 等） | **130.0** | 无近期写入 | 重编译 | 低 | 批准后 |
| B13 | `research` ＋ `deepseek-harness`（其它主题源码/构建） | **196.6** | 09-11 后无写入 | 不可再生（源码）但非本仓库 | 中 | 负责人裁决 |
| B14 | `aie-rt` ＋ `wallets` ＋ `.npm-cache` ＋ `arweave_research` ＋ `_faucet` ＋ `_net` ＋ 根 `*.log/*.txt`（16 个）＋ 根散件 | **100.0** | 其它主题/日志 | 日志不可再生但无信息价值 | 低 | 负责人点头即删 |
| B15 | 其它 <50 MiB 目录群 | **384.6** | 长尾 | — | 低 | 随批 |
| | **合计** | **17,405.0** | | | | |

### 4.4 C 档 — 永不可删（103.909 GiB 路径 / 80.910 GiB 物理）

| # | 项 | MiB | 归类依据 |
|---:|---|---:|---|
| C1 | `zimage-port\_models\Z-Image-Turbo`（transformer 3 片 22.93 ＋ text_encoder 3 片 7.49 ＋ vae 0.16） | 31,371.8 | 一切权重的源；不可再生 |
| C2 | `_backup_staging`（7 个载荷，24,984.5） | 24,984.5 | 唯一异地备份集；**其中 14,671.5 MiB 是 C4/C5/C11 的同 inode 别名，独占 10,312.6**——最后手段须显式批准 |
| C3 | `webui\models\zimage-te-qwen3-4b-bf16.gguf` | 7,678.3 | 权重；WebUI 运行必需（G-A） |
| C4 | 根 `zimage-te-qwen3-4b-bf16.gguf` | 7,678.3 | 同上（G-A 的第二名字） |
| C5 | `zimage-port\weights.bfp16` | 6,593.7 | 权重（G-B，与 C2 的 02 同 inode） |
| C6 | `zimage-port\_probe_out\weights.bfp16` | 6,593.7 | 权重；**独立 inode，唯一旧版**（SHA256 与交付版不同，t21_wcensus 已证伪"同内容"） |
| C7 | `_scratch\image\t3real_prep`（30 层载荷 ×2） | 3,796.9 | 门① PSNR 参考上游；且是备份载荷 05 的成员 |
| C8 | `_scratch\image\t6_refcache` | 2,730.4 | `webui/config.py:33 REFINER_CACHE` 产品缓存 |
| C9 | `_out\textenc\te_npu_weights.bin`（＋ sched bin） | 3,898.9 | 权重 |
| C10 | `_probe_out\zimage_loop`（8 个 KEEP 基线） | 3,476.1 | 作者 NEVER 名单 |
| C11 | `_pylibs` | 1,573.2 | AIE 编译工具链 |
| C12 | `_scratch\block\out` | 1,032.5 | 含门①的 ffn xclbin 与交付 .ll（HANDOFF C3） |
| C13 | `_wheels` | 1,314.1 | 轮子缓存（工具链） |
| C14 | `_probe_out\ryzenai_nuget` | 729.9 | 无下载政策的工具链缓存 |
| C15 | `_scratch\attn_impl\t3b_out` | 645.5 | **门①冻结参考**（10 个 references 全在此） |
| C16 | `models\K2-Horizon-1B-Q4_K_M.gguf` | 635.3 | 下载得到的权重 |
| C17 | `weights\block0.npz` ＋ `block0.coalesced.bin` | 401.3 | 权重中间件（交付件） |
| C18 | `vendor_artifacts\*.rai` | 399.5 | 供应商唯一存档（G-C） |
| C19 | `ryzen_ai_nuget_1.8.0` | 262.7 | 供应商包 |
| C20 | `_scratch\block\t4_out` | 322.8 | `webui/config.py:31 T4_OUT` 硬编码 → 产品路径 |
| C21 | `_scratch\vae`（含 20 个 `*_results.json` 门③证据 ＋ vae_weights.bfp16） | 243.4 | 门③证据链 ＋ 权重 ＋ 3b1 活跃区 |
| C22 | `webui\{jobs,cache,evidence,outputs,_selftest,shape_manifests,static}` | 25.0 | 产品缓存/验收证据（cache 内 manifest 带 sha256，门④） |
| C23 | `zimage-bfp16-npu\.git` | 13.6 | 版本历史 |
| C24 | `_scratch\pipeline\oracle96.png` ＋ `_scratch\image` 门①文件（t3d*） | ≈0.1 | 门① evidence / psnr_references |
| | **合计** | **106,402.8** | 路径口径；物理 **82,852 MiB = 80.910 GiB** |

---

## 5. 与旧审计 T21-DISKAUDIT 的对账（U4）

### 5.1 逐项对账

| 指标 | 旧审计（09:30–09:53 快照） | 本单（11:35 / 12:35 双读一致） | 差 | 解释 |
|---|---:|---:|---:|---|
| 全树路径合计 | 121.62 GiB / 86,089 文件 | **121.632 GiB / 86,185 文件** | +0.01 GiB / +96 | 一致。96 个新文件是 09:53→11:35 的正常增长，**一级目录表逐行吻合**（最大偏差 0.01 GiB） |
| 全树物理（inode 去重） | **未给出**（只有定性"虚高约 13.9 GiB"） | **98.548 GiB** | — | 旧文没有做过全树 inode 去重，这是本单新增的核心数字 |
| 硬链接虚增 | 13.9 GiB（估） | **23.084 GiB**（263 组实测） | **+9.15 GiB** | 旧文只扣了 GGUF 的 1 份重复（记 7,678.3），实际 3 个名字要扣 2 份（15,356.6）；另漏 vendor .rai（399.5）与 `_probe_out` 内部 260 组（1,258.3） |
| A 档 | 724.93 MiB | **743.1 MiB** | +18.2 | 本单把仓库 `memtile2\out`、`dit2\out` 里的 .prj 也计入（旧文只算 _scratch 的副本）；另 `*.lock` 由 599 个降到 4 个（−0.06） |
| B 档 | 13,719.70 MiB | **17,405.0 MiB** | **+3,685.3** | 本单把非 zimage 主题也纳入 B（llama.cpp-0.5.0 837.6、k2h-llama.cpp 702.4、deepseek-harness 91.5、research 105.1、aie-rt/wallets/npm/arweave 等 60.0）；旧文只在 §1.1 表里列出、**未分档**。另 `_scratch\image` 散落区因新作业增长 |
| C 档 | 98,465.86 MiB（自称"已按硬链接去重"） | 106,402.8 MiB 路径 / **82,852.0 MiB 物理（80.910 GiB）** | — | **旧 C 同时包含 C1 备份集全量与其 3 个硬链接成员，重复计 14,671.5 MiB**（旧文嘴上说"备份 02 不另计"，但 C1 的 24,984.53 是 MANIFEST 全量，并未扣） |
| **A+B+C 合计** | **112,910.49 MiB = 110.26 GiB**，自称"占 121.62 的 90.7%，余 11.4 GiB 是其它主题区长尾" | **124,550.9 MiB = 121.632 GiB**（路径），**100,913.2 MiB = 98.548 GiB**（物理） | — | **旧账面不闭合** |

### 5.2 那 11.4 GiB 差额到底是什么

旧文把 11.4 GiB 归给"其它主题区（llama.cpp 837.6 ＋ k2h 702.4 ＋ deepseek-harness 91.5 ＋ research 105.1）与各实验区 <20 MiB 的长尾"——但这四项加起来只有 **1.70 GiB**，且旧文 C13 已含 _pylibs/_wheels（2,887.3）、C17 已含 K2 权重（635.3）、C12 已含 ryzen nuget（262.7）、B6 已含 _git_backup（899.0）。真正**完全没分档**的非 zimage 只有 **1,834.5 MiB（1.79 GiB）**。所以"11.4 GiB 长尾"**不存在**。

真实是两笔方向相反的错误互相掩盖。可核的桥只有四行：

```
旧 A+B+C                                    112,910.5 MiB = 110.26 GiB
  − 备份集重复计（C1 与 C3/C4/C11 同 inode）  −14,671.5 MiB = −14.33 GiB
= 旧文"有效"账面                              98,239.0 MiB =  95.94 GiB
  ＋ 旧文完全没分档的非 zimage 主题             + 1,834.5 MiB = + 1.79 GiB
= 可对账部分                                  100,073.5 MiB =  97.73 GiB

  实测物理                                   100,913.2 MiB =  98.55 GiB
  未解释残差                                  +   839.7 MiB = + 0.82 GiB（0.8%）
```

- **残差 0.82 GiB（0.8%）如实保留**：来自旧文 B2/B3/B4/B5 等条目本身是"估"（如 "~1,230"、"~75"），以及分档边界（`_scratch\image` 的散落文件在旧文里一半算 B5、一半算 C18）。本单不再强行追平——任何"精确到 MiB 的桥"都会是编的。
- 那 1,834.5 MiB 的明细：llama 837.6 ＋ k2h 702.4 ＋ _git_backup2 87.1 ＋ deepseek-harness 91.5 ＋ research 105.1 ＋ 根日志/散件 ~37 ＋ 其余 ~24。

即：**旧文一边用 14.33 GiB 的备份重复计数把账吹高，一边用"少扣 9.15 GiB 硬链接"和"未分档的 1.79 GiB 非 zimage"把账压低**，两股力把总量拉回 110.26，于是它看起来"解释了 90.7%"，剩下的 11.4 GiB 就被一句"长尾"打发了。**而 11.4 GiB 这个数本身，是"14.33 虚增 − 2.9 少计"的残差，不是任何一块真实资产。**

### 5.3 "此前 zimage 域审计漏了哪些大头"

**结论：在 public_work 范围内，没有漏掉任何网点级大头。** 逐条核过：

| 旧文可能漏的 | 本单实测 | 判定 |
|---|---:|---|
| llama.cpp-0.5.0 构建树 | 837.6 MiB | 列在 §1.1，但**未进 A/B/C**；本单已归 B7 |
| k2h-llama.cpp 构建树 | 702.4 MiB | 同上 → B8 |
| deepseek-harness / research / wallets / arweave / npm-cache / aie-rt / _faucet / _net | 302.0 MiB | 同上 → B13/B14 |
| 48 个根 pip-*/tmp*/quark 目录 | **不可测**（ACL 拒绝） | 旧文写成"0 字节空目录"，本单判为**测不到** |
| `_probe_out` 内部 260 组硬链接 | 1,258.3 MiB 虚增 | 旧文完全未发现 → 已计入 §3 |
| vendor .rai 硬链接 | 399.5 MiB 虚增 | 旧文完全未发现 → 已计入 §3 |

**真正"漏"的不是目录，是记账口径。** 如果要给负责人一句可执行的话：_旧审计的一级表可以直接用，把 §2.4 的 A/B/C 换成本报告 §4 的三张表即可。_

---

## 6. 诚实负结果 / 测不到项（U5⑤）

### 6.1 ACL 拒绝目录：144 个，全部"不可读"

- `os.scandir` 拒绝 **144** 个目录；对同样 144 个路径逐个跑 `icacls`，**144/144 返回 rc=5 `Access is denied`**。
- 按任务书口径：**全部记为"不可读"**（不是"空"）。可测容积**下界 = 0 B**，上限**无法估计**——没有可用的枚举通道。
- 分布（按父目录）：

| 父目录 | 个数 |
|---|---:|
| `public_work\`（48 个：Work / _condaenvs / _condapkgs / pip-install-* / pip-ephem-* / pip-build-tracker-* / pip-unpack-* / quark_onnx.quant.* / tmp*） | 48 |
| `zimage-port\`（tmp*/…） | 45 |
| `zimage-bfp16-npu\` | 9 |
| `_probe_out\utils_probe\tmp` | 6 |
| `_scratch\block\t4_out\t4_base.prj` | 6 |
| `_piptmp\` | 5 |
| `zimage-port\npu-conversion\` | 5 |
| `_npu_cache\objects` | 5 |
| `_scratch\util\out\…prj` | 3 |
| `_probe_out\quark_tmp` / `_probe_out\aie-rt` / `_scratch\vae` / `webui\cache\npu_t21` / `_npu_cache7\objects\*` | 1 各 |

- **重要更正**：旧审计 §1.6 写"空 `tmp*` 目录 37 ＋ 9 ＋ 45 ＝ 91 个，**全部 0 字节**"。这与本单的 144 个拒绝目录高度重合（48+9+45=102 就在同一批名字上）。几乎可以确定：**旧文的"0 字节"是把"拿不到大小"当成了"空"**。本单不采信该结论，改记"不可读、容积未知"。
- 唯一能看到 ACL 的是 `webui\cache`（其子目录 `npu_t21` 被拒）：`Everyone:(I)(CI)(DENY)(DC)`、`BUILTIN\Administrators (F)`、`Authenticated Users (M)`。即使如此，Python 仍无法枚举 `npu_t21`。
- 未做任何提权、未改 ACL、未硬闯。**建议**：若必须测容，需由具备 `SeBackupPrivilege` 的账户执行 `takeown`/`icacls` 修复，这超出本单"只读零修改"的边界。

### 6.2 读数漂移（并发代理）

| 时刻 | D: free | 说明 |
|---|---:|---|
| 11:35:41（首轮扫描末） | 869,138,432 B（0.810 GiB） | — |
| 12:34:53 | 981,938,176 B（0.914 GiB） | 期间 +112,799,744 B |
| 12:35:40（复核末） | 981,921,792 B（0.9145 GiB） | 与 12:34:53 基本持平 |

但**树本身是静止的**：两轮扫描的字节数与文件数完全一致，树内最新 mtime = 11:22:39（两轮都在其后）。所以 11:35→12:35 的 +113 MiB 是**工作区之外**的释放，与本报告数字无关。

受影响的历史数字：`_scratch\vae` 在旧审计期间（09:37）仍在被 3b1 写入；本单快照 `vae` = 243.4 MiB（旧文 09:34 记 233.06 MiB），**+10 MiB 是那之后的增长**。

### 6.3 未做的事（如实声明）

- **没有做 sha256 校验**（除引用 t21_wcensus 已有的 weights.bfp16 结论外）。所有"同 inode / 不同 inode"的判定基于 `VolumeSerialNumber + FileIndex`，这是 NTFS 的权威标识，比路径比较可靠；但"两个不同 inode 内容相同"本单**没有验证**（例如 `weights\block0.npz` 与 `_scratch\weights\block0.npz` 只验了 samefile=False）。
- **没有解开任何 tar/bundle** 去看内部（`_backup_staging\04/05` 的内容归属来自旧审计与 MANIFEST，本单未复核）。
- **没有测硬盘的"已分配簇"**：本报告所有体积都是**文件逻辑大小**（`st_size`），不是 NTFS 实际占用的簇。若存在大量小文件或压缩/稀疏文件，物理簇占用会与本报告有偏差。全树 86,185 文件、13,285 目录，按 4 KiB 簇估算的碎片开销在数百 MiB 量级，本单**未计入也未测量**。
- **没有枚举备用数据流（ADS）**。
- **没有读取 `$MFT` / USN 日志**（`fsutil` 在本沙箱被拒）。
- 上述任一限制若被要求突破，都需要额外权限或改变"只读"边界，**未执行**。

### 6.4 方法学自检（防止本次也犯错）

| 陷阱 | 本单如何规避 |
|---|---|
| 跟随 Junction 导致重复计数 | 用 `st_file_attributes & 0x400` 判定 reparse point，**一律跳过并单列**（4 个，0 字节） |
| 用文件名累加当物理占用 | 逐文件 `GetFileInformationByHandle` 取 VolumeSerial+FileIndex，**263 组同 inode 只计一次**（差 23.084 GiB） |
| 把 ACL 拒绝当空目录 | 单独记录拒绝清单，**不并入"空目录"**；空目录从 452 修正为 308 |
| Python `st_ino` 在本机全为 0 | 首次尝试即发现（86,185 文件 → 1 个 inode），改用 Win32 API；**若照抄 `os.stat().st_ino` 会得出"零重复"的错误结论** |
| 逐目录多次调用造成漂移 | 全部用 **单遍 `os.walk` 聚合**（首轮 3.4 s，含 inode 19.8 s），并在报告头部给出两次独立读数 |

### 6.5 本单自身的写入

- **工作区内唯一新建文件 = 本报告**（`zimage-bfp16-npu\docs\T21-DISKMAP-FULL.md`）。
- 中间数据落在 **`%TEMP%\dsh-kxZHGu\`**（`t21_diskmap.json` / `t21_diskmap2.json` / `t21_dir`… 共 4 个），**不在工作区、不在任何被审计路径下**，可随时删除，不参与任何统计。
- **零删除、零移动、零改名、零权限修改**；未 `git` 操作；未执行任何 `rmtree`/`Remove-Item`。

---

## 7. 方法与复现

### 7.1 三次单遍扫描

| 趟 | 命令要点 | 产出 | 耗时 |
|---|---|---|---:|
| P1 | `os.walk` 聚合 per-dir（depth≤3 累加） | 一级目录表、目录树、>1 GiB 清单、拒绝清单、空目录 | 3.4 s |
| P2 | P1 ＋ `CreateFileW`/`GetFileInformationByHandle` 取 inode/nlink | 物理去重、263 组别名、物理归属 | 19.8 s |
| P3 | 全深度目录树 ＋ 扩展名聚合 ＋ `.prj` 归属 | 2.2/2.4 的表、A 档 .prj 明细 | 1.5–17 s |

关键片段（可直接复跑，只读）：

```python
# 单遍聚合（不跟随 reparse point）
import os, stat
REPARSE = 0x400
def walk(path, rel):
    with os.scandir(path) as it:
        for e in it:
            st = e.stat(follow_symlinks=False)
            if st.st_file_attributes & REPARSE:      # junction / symlink：跳过
                continue
            if stat.S_ISDIR(st.st_mode): walk(e.path, nm); continue
            yield nm, st.st_size

# NTFS 真 inode（Python 的 st_ino 在本机恒为 0，不可用）
k32.CreateFileW.restype = ctypes.c_void_p
h = k32.CreateFileW(u'\\\\?\\' + path, 0, 7, None, 3, 0x02000000, None)
k32.GetFileInformationByHandle(h, ctypes.byref(bhfi))
key = (bhfi.dwVolumeSerialNumber, (bhfi.nFileIndexHigh << 32) | bhfi.nFileIndexLow)
nlink = bhfi.nNumberOfLinks
```

### 7.2 复核清单（任何人可在一分钟内验证）

| 断言 | 一条命令 |
|---|---|
| 全树 121.632 GiB / 86,185 文件 | 本报告 §1.1 求和 |
| 物理 98.548 GiB | 同 §3 的 inode 去重 |
| GGUF 3 个名字 1 个 inode | `fsutil hardlink list zimage-te-qwen3-4b-bf16.gguf`（若 fsutil 可用） |
| 4 个 Junction | `Get-ChildItem -Recurse -Force -Attributes ReparsePoint` |
| 144 个拒绝目录 | `(Get-ChildItem -Recurse -Force -ErrorAction SilentlyContinue \| ? {$_.PSIsContainer}).Count` 与总数 13,285 对比 |

---

## 8. 一句话结论

**public_work 的 98.55 GiB 物理占用已经全部对上账（权重 55.92 ＋ 实验 26.10 ＋ 备份独占 10.07 ＋ 工具链 3.79 ＋ 其它 2.67，残差 0.00）；路径合计 121.63 GiB 与物理的 23.08 GiB 差额是 263 组硬链接，不是失踪文件。旧审计不是"漏了大头"，而是备份集重复计 14.33 GiB 与硬链接少扣 9.15 GiB 互相抵消，把 11.4 GiB 的缺口伪装成了"长尾"。可立即释放仍然只有 A 档 743 MiB；要动 10 GiB 以上，只能批 B 档（17.0 GiB，最大三项是 _probe_out 4.13 GiB、perfc 3.62 GiB、block 2.92 GiB）。**

---

*本报告由 t21_diskmap 子代理生成，只读产出；扫描窗口 2026-10-19T11:35:38–12:35:40（Asia/Shanghai），双轮读数一致。*
