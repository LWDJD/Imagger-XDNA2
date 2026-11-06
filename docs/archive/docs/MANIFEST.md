# _backup_staging — 精简备份集(8 个文件,≈24.4 GB,无碎片)

> **迁移 2026-10-19**:本文件已移至 `zimage-bfp16-npu\docs\MANIFEST.md`(不再位于 `_backup_staging\`);**文件已移,恢复步骤照旧**——下表与「最小恢复步骤」1-4 不变,`_backup_staging\` 仍放 7 个载荷文件(01-06 与 bundle);06 包内是迁移前根目录副本,解包重现的是旧位置,以 `zimage-bfp16-npu\docs\` 下版本为准。

上传本目录的全部文件即可。每个文件可独立恢复;硬链接文件与源同数据,上传工具读取的是同一份内容。

| 文件 | 大小 | 内容 | 恢复到 |
|---|---|---|---|
| 01_te-gguf-zimage-te-qwen3-4b-bf16.gguf | 7.68 GB | TE 权重(webui 运行必需) | 任意位置;设 env ZIMAGE_TE_GGUF,或放 zimage-bfp16-npu\webui\models\ 同名 |
| 02_weights.bfp16 | 6.59 GB | BFP16 权重(验证/重建载荷用) | zimage-port\weights.bfp16 |
| 03_vendor_39e41383….rai | 0.39 GB | vendor .rai 唯一存档 | zimage-bfp16-npu\vendor_artifacts\ |
| 04_toolchain-pylibs-llama.tar | 3.05 GB | _pylibs(编译工具链)+ llama.cpp-0.5.0 + k2h-llama.cpp(TE 运行时)——18k 碎片文件打包 | 解到 <workspace>\ 根 |
| 05_runtime-payloads-gates.tar | 7.01 GB | t3real_prep(30 层载荷)+ t6_refcache(refiner 缓存)+ t3b_out(逐位门基线)+ webui(代码/缓存/验收证据) | 解到 zimage-bfp16-npu\ 对应路径(t3real_prep/t6_refcache 在 _scratch\image\,t3b_out 在 _scratch\attn_impl\) |
| 06_workspace-key-docs.tar | <10 MB | 工作区根全部关键文档(HANDOFF、WEIGHT-MAP、TIMING-RECONCILIATION、questions、CLEANUP 台账等) | 解到 <workspace>\ 根 |
| zimage-code-docs.bundle | 4 MB | **git 全历史 + 961 跟踪文件**(全部复现代码与报告,含 webui.py、README-WEBUI.md) | git clone zimage-code-docs.bundle zimage-bfp16-npu |
| MANIFEST.md | 本文件 | — | — |

## 最小恢复步骤
1. git clone zimage-code-docs.bundle zimage-bfp16-npu
2. 解开 04/05/06 三个 tar 到上表路径
3. 权重按表放置(01 可只放 webui\models\,自动命中)
4. 启动:cd zimage-bfp16-npu ; & py312 webui.py --host 127.0.0.1 --port 7860

## 有意未纳入(按"运行需要 + 关键复现"原则)
- _models 主权重库(≈30.6 GB):仅重建 t3real_prep/weights 时需要;载荷已打包,运行不再依赖
- models\K2 GGUF、llama.cpp 源码树、_wheels、_probe_out 探针残留:非本产品运行所需
- TE 重建脚本 convert_hf_to_gguf.py(<retired converter>):ACL 拒绝链接;GGUF 本体已备,通常无需重建
---

## 【更正 2026-10-19】实测字节数(上表 01/02 两行是旧值,且全表单位混用)

以文件系统实测 `Length` 为准(`Get-ChildItem`),单位同时给 GiB(2^30)与 GB(10^9):

| 文件 | 上表值 | 实测 Length(B) | 实测 GiB | 实测 GB |
|---|---|---|---|---|
| 01_te-gguf-…gguf | 7.68 GB | 8,051,284,800 | **7.498** | **8.051** |
| 02_weights.bfp16 | 6.59 GB | 6,913,965,970 | **6.439** | **6.914** |
| 03_vendor_….rai | 0.39 GB | 418,881,856 | 0.390 | 0.419 |
| 04_toolchain-pylibs-llama.tar | 3.05 GB | 3,279,426,560 | 3.054 | 3.279 |
| 05_runtime-payloads-gates.tar | 7.01 GB | 7,529,954,816 | 7.013 | 7.530 |
| 06_workspace-key-docs.tar | <10 MB | 422,912 | 0.0004 | 0.0004 |
| zimage-code-docs.bundle | 4 MB | 4,239,520 | 0.004 | 0.004 |

- **01 与 02 是旧值**:01 差 3,785,884,800 B(7.68 → 8.051 GB),02 差 334,034,030 B(6.59 → 6.914 GB)。
  两个数都既不是 GiB 也不是 GB,**来源未核**,按实测值执行。
- **单位口径**:上表其余行(03/04/05)按 GiB 写、却标成 "GB";本表起统一注明。
- **合计实测**(8 个文件,含本 MANIFEST):26,198,178,605 B = **24.399 GiB** = **26.198 GB**。
  标题"≈24.4 GB"**按 GiB 读是对的**,按 GB 读则偏低约 1.8 GB;标题原文保留不改。
- **02 的性质**:6,913,965,970 B / 506 张量,与 `zimage-port\weights.bfp16` 同大小(硬链接副本)。
  它**不是** `_probe_out\bfp16_weights_v2` 的 521 张量产物(6,924,272,328 B);后者已删除且不能由
  02 恢复,见 `CLEANUP-EXECUTION.md` 补记段。

