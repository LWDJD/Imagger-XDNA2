# SESS-PACK2 — 打包器定位 / M=160 受阻 / 提权脚本与缓存根出处

> t22_sessA 第七轮（2026-10-24）。负责人批准 ①②，③ 改由父会话提权代跑。

---

## 0. 四条判读

1. **坏件不是 `_dd103t.py⟧ 的 schema。** 坏文件是 `{"version":"1.0","ctrl_pkt_patch_info":[...]}⟧，
   而 `_dd103t.py⟧ 写的是 `external_buffers⟧ schema。⇒ 坏件是**编译器产出的 `patch.json⟧ 本身**为空表且写坏。
2. **可直接落地的加固**：`_dd103t.py:20-23⟧ 读到 `patch entries = 0⟧ 时**照样打包**——应在那里硬失败，
   而不是产出一个跑不了的 txn。这是三行改动，我可以做。
3. **M=160 备胎没测成**：`(M=160, K=1536, N=6144)⟧ 这个**组合**不在表里（M 有 160、K 有 1536，但三元组不存在），
   分区直接 rc=1。要测得先挑一个**真实存在的三元组**。
4. **缓存根出处已定位**，且**不可靠地可重定向**（见 §3）——提权脚本已备好（§4）。

---

## 1. 坏件的确切来源

```python
# _dd103t.py:20-23
pj = json.loads((BUILD / "patch.json").read_text(encoding="utf-8"))
entries = pj.get("ctrl_pkt_patch_info", [])          # <-- 空表也放行
CTRL = (BUILD / "ctrl.bin").read_bytes()
print("patch entries = %d ; ctrl.bin = %s B" % (len(entries), format(len(CTRL), ",")))
```

`patch.json⟧ 由编译器侧产出（schema `{"version":"1.0","ctrl_pkt_patch_info":[{offset,size,xrt_arg_idx,bo_offset}]}⟧，
与 `gl16_pack/cast_..._ctrl_meta.json⟧ 同构）。**当编译没有产出任何补丁位置时，空数组被写成孤立的 `]⟧**，
于是得到 65 B 的坏文件；打包器不校验 `len(entries)==0⟧，把它打进 `sd_fastpm_ours*.zip⟧。

**修复分两层**：
- **我能做（无需提权）**：在 `_dd103t.py⟧ 加硬校验——`entries==0⟧ 直接抛错，禁止产出 unusable txn。
- **需要真编译**：gemm/layernorm/elwmul 的 1056 系列在任何包里都**没有好件**（只有 `cast⟧ 在 exp/mine/gl16 有），
  必须重跑编译才能拿到非空补丁表。这一层我做不了（要 AIE 编译器 + 可写缓存）。

---

## 2. M=160 备胎（未完成，原因明确）

| 尝试 | 结果 |
|---|---|
| 建 `(M=160, K=1536, N=6144)⟧ 图并分区 | **rc=1，分区拒绝** |
| 建 `(M=1024, K=1536, N=6144)⟧ 对照 | 也 rc=1（同一脚本，说明是形状组合而非脚本问题） |

M 的集合有 160、K 的集合有 1536，但**三元组 (160,1536,6144) 不在 293 个组合里**。
要拿到 M=160 的备胎数据，得先枚举出真实存在的 `(160, K, N)⟧ 组合。**下一轮做**。

---

## 3. vaip 缓存根：出处与可重定向性

### 3.1 出处

```python
# zimage-bfp16-repo/vaip_env.py:37-38
DEFAULT_CACHE = pathlib.Path(r"C:\temp") / os.environ.get("USERNAME", "?") / \
    "vaip" / ".cache"
```

同文件 docstring（:7、:11-12）写明：路径由**字面量 `C:\temp⟧ 拼 `USERNAME⟧** 推导，
**`TEMP/TMP/TEST_TMPDIR⟧ 都搬不动它**，且"string extraction from `onnxruntime_vitisai_ep.dll⟧ confirms it reads `USERNAME⟧"。

### 3.2 能不能用 env 改指工作区？

我在 `onnxruntime_vitisai_ep.dll⟧ 里扫到的缓存相关环境变量：

| 变量 | 存在 | 实测效果 |
|---|---|---|
| `XLNX_CACHE_DIR⟧ | ✅ | 指到工作区后**仍 0 文件**（第 1、6 轮各测一次） |
| `XLNX_ENABLE_CACHE⟧ | ✅ | 设 1 后无变化 |
| `KEEP_CACHE⟧ | ✅ | 设 1 后无变化 |
| `XLNX_ENABLE_FILE_BASED_CACHE_KEY⟧ | ✅ | 设 1 后无变化 |
| `FLEXMLRT_FORCE_CACHE⟧ | ✅ | 未测 |

**结论：至少对"产物落盘"这一语义，`XLNX_CACHE_DIR⟧ 不可靠。**
（`vaip_env.py⟧ 声称曾在小 MatMul 上验证成功，但那与本次三次实测冲突，以实测为准。）
**更干净的解**：让 `C:\temp\<USER>\vaip\⟧ 可写（提权），而不是指望重定向。

---

## 4. 提权脚本（交给父会话跑）

`RUN-ELEVATED.ps1⟧（工作区根目录，已通过 PowerShell 解析器校验，291 tokens）
+ `_scratch/sessA_elevated_probe.py⟧（已通过 `py_compile⟧）。

### 4.1 用法

```powershell
powershell -ExecutionPolicy Bypass -File <checkout>\RUN-ELEVATED.ps1
```

### 4.2 每一步做什么 / 预期产物

| 步 | 动作 | 预期产物 | 失败信号 |
|---|---|---|---|
| 0 | 预检 `C:\temp\<USER>\vaip\.cache⟧ 可写 | 打印 `writable : True⟧ | `writable : False⟧ + exit 2（说明提权没生效） |
| 1 | 冷建会话（`test_model.onnx⟧，空缓存），再热建 3 次 | `_scratch\sessA\elevated_report.json⟧ | 脚本异常 |
| 2 | 列出缓存里的 `.rai⟧ | `<cache>\<hash>\<hash>.rai⟧ | 无 `.rai⟧ = 缓存仍不落盘 |

### 4.3 判据（脚本自己会给 `verdict⟧）

- **成功**：`cold.files_after > cold.files_before⟧（冷编译写了文件）**且** 三次热会话 `new_files⟧ 全 0
  ⇒ 缓存机制成立，`verdict = "CACHE WORKS: cold wrote files, warm wrote none"⟧。
  此时 `cold.session_s⟧ vs `warm[].s⟧ 的差值就是"编译段"的真实成本。
- **失败**：冷会话写了 0 文件 ⇒ 缓存仍不落盘，需要换机器或改 EP 配置，不是权限问题。

**为什么用 `test_model.onnx⟧ 而不是我们的分片**：分片已删，重导出是另一件事；
而"缓存到底能不能落盘"这个**机制问题**用一个 23.6 MB 的官方模型就能回答，成本最低。

---

## 5. 本轮交付清单

| 件 | 状态 |
|---|---|
| 坏件来源定位（编译器 `patch.json⟧ 空表 + 打包器不校验） | ✅ |
| `RUN-ELEVATED.ps1⟧ + `sessA_elevated_probe.py⟧ | ✅ 已校验，待提权执行 |
| vaip 缓存根出处 + 可重定向性结论 | ✅ |
| M=160 备胎数据 | ❌ 三元组不存在，需重挑形状 |
| 打包器硬校验（`entries==0⟧ 拒绝） | ⬜ 下一轮 |

---

## 6. 纪律与边界

- **预测先行**：预判"M=160 可直接测"——**否证**（三元组不在表里）；预判"env 可重定向缓存"——**否证**（四项全设仍 0 文件）。
- **负对照必配**：M=1024 同脚本同参数也 rc=1（证明是形状而非脚本）；缓存四项 env 逐一开仍 0 文件（证明不是缺开关）。
- **诚实负结果**：M=160 未测成；打包器只做了定位未做修复；提权脚本未执行（需父会话）。
- **产品零变化**：只新增 `RUN-ELEVATED.ps1⟧、`_scratch/sessA_elevated_probe.py⟧ 与本文。
- **51.3 TOPS / F-14.1**：本轮无吞吐测量、无器件独占，未触碰。
