# SESS-PACK — 主线验证被两处硬阻塞挡住（诚实报告）

> t22_sessA 第六轮（2026-10-24）。负责人改判：主线=代码生成精确 shape，padding 只是备胎。
> 本轮试图跑 ②（重导出 1 片 M=96 冷编译）。**没跑成**，原因是两个硬阻塞，均已定位到可复现的证据。

---

## 0. 三条判读

1. **③ 的坏件是系统性的，不是一个文件。** 项目自己打的 `sd_fastpm_ours*.zip⟧ 里，
   **每一个** txn 的 `*_ctrl_meta.json⟧ 都是 65 B 的坏 JSON（`ctrl_pkt_patch_info⟧ 空数组写坏）。
   覆盖的正是 M=1056 的全部 Z-Image 键。⇒ **DD 路径从来没有真正加载过我们的 Z-Image 分片**；
   0.23 s 那个数字是在**官方 shape** 的图上测的（真，但不是我们的几何）。
2. **② 被沙箱挡住，不是被模型挡住。** VitisAI EP 的编译缓存根是 `C:\temp\<USER>\vaip\.cache⟧，
   本会话 workspace-write 下**写拒绝**；`XLNX_CACHE_DIR⟧ + `XLNX_ENABLE_CACHE⟧ + `KEEP_CACHE⟧ + `XLNX_ENABLE_FILE_BASED_CACHE_KEY⟧ 全设上，
   仍然 **0 个缓存文件、0 个 .rai**，三次会话 24.6 / 24.8 / 24.7 s。**冷编译验证需要放开沙箱**——这是负责人级决定，不是代码改动。
3. **"M=96 可生成、无需 padding"在原理上成立，但生成器正是那个坏掉的打包器。**
   M=1056 的 txn 就是它生成的，全是坏件。所以这条主线的前提不是"旧管线跑通过"，
   而是"**先修好打包器**"——旧管线跑通的是 **VAIP** 路径，不是 DD 路径。

---

## 1. 坏件取证（③）

### 1.1 坏件长什么样（逐字）

```json
// _probe_out/pack/cast/cast_a16bfacc16bfp_1056_3840_ctrl_meta.json   65 B
{
    "version": "1.0",
    "ctrl_pkt_patch_info": 
    ]
}
```

**空数组被写成了孤立的 `]⟧**（缺 `[⟧）——所以报 `parse error at line 4, column 5: unexpected ']'⟧。
这是写盘逻辑的 bug（空列表分支），不是数据损坏。

### 1.2 好件长什么样

```json
// _probe_out/gl16_pack/cast_a16bfacc16bfp_1056_3840_ctrl_meta.json   2,820 B
{
    "version": "1.0",
    "ctrl_pkt_patch_info": [
        { "offset": 7260, "size": 6, "xrt_arg_idx": 3, "bo_offset": 0 },
        { "offset": 7308, "size": 6, "xrt_arg_idx": 2, "bo_offset": 0 },
        ...
    ]
}
```

`ctrl_pkt_patch_info⟧ 是运行时按 BO 地址打补丁的偏移表。空表 = 补丁位置全丢 = 跑起来必错。
**所以坏件不是"能跑但慢"，是"跑不了"。**

### 1.3 覆盖面（插件包逐个验 JSON）

| txn key | 出现在 | ctrl_meta 状态 |
|---|---|---|
| `gemm_..._1_1056_3840_10240⟧ | ours / ours2 / ours3 / ours4 | **全 65 B BAD** |
| `gemm_..._1_1056_10240_3840⟧ | ours / ours2 / ours3 / ours4 | **全 65 B BAD** |
| `gemm_a16bfw16bfpacc16bf_1_1056_3840_3840⟧ | ours / ours2 / ours3 / ours4 | **全 65 B BAD** |
| `layernorm_..._1_1056_3840⟧ | ours2 / ours3 / ours4 | **全 65 B BAD** |
| `elwmul_..._1056_3840⟧ | enum | **65 B BAD** |
| `cast_a16bfacc16bfp_1056_3840⟧ | exp(14,761 B ok) / mine(1,426 B ok) / ours3,4(BAD) | **混装，谁赢看加载顺序** |
| 官方 DiT / Unet 里的 elwadd / silu / gemm | DiT / Unet | **5–300 KB，全 ok** |

**判读**：官方件是好的，**项目自己打的全是坏的**。M=1056（我们的真实几何）只存在于自打件里。

---

## 2. ② 的阻塞（沙箱）

### 2.1 实测

```
XLNX_CACHE_DIR=<workspace>  XLNX_ENABLE_CACHE=1  KEEP_CACHE=1  XLNX_ENABLE_FILE_BASED_CACHE_KEY=1
session 0: 24.622 s   cache files=0
session 1: 24.832 s   cache files=0
session 2: 24.723 s   cache files=0
rai files: []
```

三次会话全价，缓存目录**一个文件都没有**。

### 2.2 为什么

- EP 的缓存根按 `C:\temp\<USERNAME>\vaip\.cache⟧ 硬编码推导（`vaip_env.py⟧ 的字符串取证 + `test7\cache\clear⟧ 同源）；
- 本会话 workspace-write，该路径 **WRITE DENIED / MKDIR DENIED**（第一轮已实测）；
- `XLNX_CACHE_DIR⟧ 在 DLL 里确实存在（连同 `XLNX_ENABLE_CACHE⟧ / `KEEP_CACHE⟧ / `FLEXMLRT_FORCE_CACHE⟧），
  但**设了也不落盘**——所以它至少不是"产物输出目录"这一个语义。

**⇒ 冷编译（350–380 s/片）无法在本会话内验证。** 需要负责人放开沙箱到该路径，或换一台不受限的机器。

---

## 3. 对负责人计划的修正建议

| 原计划 | 现状 | 建议 |
|---|---|---|
| ② 重导出 1 片 M=96 冷编译 | **沙箱挡住**（VAIP 缓存不可写） | 需放开 `C:\temp\15081\vaip\⟧ 写权限；否则此项无法在本环境完成 |
| ③ 修 80 子图坏件（重打包/重跑 pack） | **坏的是打包器本身**，不是单个文件 | 先定位 pack 脚本里写 `ctrl_pkt_patch_info⟧ 的分支；空表分支写坏了 |
| ④ 17 片组装 | 前置依赖 ③ | 打包器修好前，M=1056/M=96 的 DD 路线都跑不起来 |

**关键提醒**：负责人说"旧管线本就跑通过，预期过"——**旧管线跑通的是 VAIP 路径**（`.rai` + 24.6 s 会话）。
**DD 路径从来没有在我们的几何上跑通过**，因为它的 txn 全是坏件。这两条路线的"跑通过"不能互相背书。

---

## 4. 本轮仍然成立的东西（不受影响）

- **C3 的启动曲线**：官方 shape 上 warm ≈ 0.24 s + 0.010 s/子图 ⇒ 442 子图 ≈ 4.7 s。
- **B2 的 worker 模式**：一进程一会话 + state sha256 前置 + 产物判据；4/4 成功、跨进程输出逐字节一致。
- **C1/C2 的精度结论**：bfp16 量化地板，LN mean 0.55% / 全链 mean 1.4%。
- **B3 的 M 集合**：sd_fastpm 无 96/64/128，最近 160（仅在"复用预编译表"这条路成立）。

---

## 5. 复现

```powershell
$py = 'D:\Users\15081\miniforge3\envs\ryzen-ai-1.8.0\python.exe'
$d  = 'D:\Project\deepseek_harness\public_work\zimage-bfp16-npu'
& $py -u "$d\_scratch\sessA_packchk.py"    # 逐包验 ctrl_meta JSON
& $py -u "$d\_scratch\sessA_packidx2.py"   # 逐 key 覆盖面 + 好坏标记
& $py -u "$d\_scratch\sessA_cache_en.py"   # XLNX_ENABLE_CACHE 全开仍 0 文件
& $py -u "$d\_scratch\sessA_cachedir.py"   # EP DLL 里的缓存相关环境变量
```

---

## 6. 纪律与边界

- **预测先行**：本轮预判"② 能跑通"——**未达成**（沙箱）；预判"③ 是单个坏文件"——**否证**（系统性）。
- **负对照必配**：官方 DiT/Unet 的 ctrl_meta 全 ok（证明不是解析器问题）；好件/坏件同 key 对比（证明是写入 bug）。
- **诚实负结果**：② 未跑、③ 未修、④ 未开始。本轮产出是**阻塞定位**，不是主线推进。
- **产品零变化**：只新增 `_scratch/sessA_pack*.py⟧、`sessA_cache_en.py⟧、`sessA_cachedir.py⟧ 与本文。
- **51.3 TOPS / F-14.1**：本轮无吞吐测量、无器件独占，未触碰。
