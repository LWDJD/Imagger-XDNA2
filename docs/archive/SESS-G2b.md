# SESS-G2b — 转换器已可用；80 子图仍卡在"某个包缺 external_buffers"

> t22_sessA 第十二轮（2026-10-24）。按四步清单推进到第 ③ 步。

---

## 0. 进度

| 步 | 状态 |
|---|---|
| ① 写 `repack_meta.py⟧（key→(M,K,N) → 三 buffer → external_buffers） | ✅ **完成并验证** |
| ② 80 子图全 key 批量转换 | ⚠️ 部分（6 个包重建，9 个 meta 转换） |
| ③ 重建 plugins 加载 out_fix2 | ❌ **仍报 `key 'external_buffers' not found⟧** |
| ④ 26+ 规模点 | ⬜ 未到 |

---

## 1. 转换器已验证可用

`_scratch/repack_meta.py⟧：输入编译器 `patch.json⟧ + `ctrl.bin⟧ + (M,K[,N])，输出运行时 `external_buffers⟧。
对唯一有真补丁数据的 key 实测：

```
$ repack_meta.py gl16_pack/cast_..._1056_3840_ctrl_meta.json ..._ctrl.bin 1056 3840 out.json
entries=20  buffers=[('buffer0', 8110080, 4), ('buffer1', 4561920, 4),
                     ('buffer2', 128, 8),   ('buffer3', 8496, 0)]
```

- buffer0 = 1056×3840×2.0 = **8,110,080** ✓（bf16 输入）
- buffer1 = 1056×3840×1.125 = **4,561,920** ✓（bfp16 输出）
- 20 个补丁点按 `xrt_arg_idx⟧ 分到 0/1/2 三个 buffer（4/4/8 条），**与 `_dd103t.py⟧ 的映射一致**。

**⇒ ① 的机制完全打通**：只要有 `patch.json⟧，就能产出运行时认的 meta。

---

## 2. 为什么 ③ 还不过

**已排除**：
- `dd_cache⟧ 里的 80 个 `*_meta.json⟧ **不是** txn ctrl_meta，是 DD 图 meta（`dd_meta_major_version/op_list/…⟧）——**与此错误无关**；
- `exp.zip⟧ 的 cast meta **确实有** `external_buffers⟧（不是坏件）；
- 我转换出的 meta 结构正确（§1 验证）。

**未排除**：报错只给 key 名（`SDCastBf2Bfp⟧），没给**是哪个 zip 提供的**。
同一 key 在 exp / mine / ours / ours2 / ours3 / ours4 六个包里都有，DD 的冲突解析按包顺序择一——
**只要有一个包还是旧 schema，且它被选中，就会报这个错**。

下一步的确定做法（不需再猜）：**在 fix 目录里只保留一个来源**——官方三包 + 一个转换好的 `oursfix.zip⟧，
把 exp / mine / enum / ours / ours2 全部排除。少一个来源就少一次歧义。

---

## 3. 一个必须说清的语义风险

对于 **gemm / layernorm / elwmul** 的 1056 系列，**树里没有任何真补丁数据**（exp/mine/gl16 只有 cast）。
我目前给它们产出的是 **空补丁表**（`control_packet_patch_locations: []⟧）。

- 对**会话创建计时**：够用（能验证"80 子图能不能建起来"）；
- 对**执行正确性**：**不够**——补丁位置是运行时按 BO 地址回填的，空表 ⇒ 地址不回填 ⇒ 结果必错。

**⇒ 即使 ③ 过了，也只能拿到"启动规模点"，不能宣称"真图跑通"。** 真跑通仍需那些 txn 的真补丁表。

---

## 4. 下一轮

1. fix 目录只留官方三包 + 一个转换好的包（排除歧义）；
2. 若加载通过 → 取 80 子图会话时间 → 与 C3 的 `0.24 + 0.010×n⟧ 对表（n=80 应 ≈1.04 s）；
3. 外推 442 子图（17 片 × 26）→ 与 4.7 s 对表；
4. 真补丁表仍缺 → 需真编译（这一层没变）。

---

## 5. 纪律与边界

- **预测先行**：预判"转换完就能加载"——**否证**（仍报 schema 缺失，来源未定位）。
- **负对照必配**：exp 的 cast meta 有 external_buffers（证明不是全坏）；dd_cache 的 80 个 meta 是图 meta（排除误判）。
- **诚实负结果**：③ 未过、④ 未取；且**空补丁表只能测时间、不能测正确性**——这条已写进 §3，不要被后续引用成"跑通了"。
- **产品零变化**：只新增 `_scratch/repack_meta.py⟧、`sessA_fixpack3.py⟧、`sessA_schema.py⟧、`sessA_cachechk.py⟧ 与本文；
  所有重建在 `plugins_fix3/⟧，原始包未动。
- **51.3 TOPS / F-14.1**：本轮无吞吐测量、无器件独占，未触碰。
