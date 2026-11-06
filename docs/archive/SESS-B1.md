# SESS-B1 — xclbin/PDI 配对的真解：换入口，不是补文件

> t22_sessA 第三轮（2026-10-24）。上一轮 E1 卡在 `FusionRuntime.compile` 的 `op_pdi_map`/`DPU_1`。
> 本轮沿"旧实验脚本里必有配方"这条线索找到了 `_dd22_live.py`，结论是**入口选错了**。

---

## 0. 结论先行

1. **不需要那个 xclbin。** `FusionRuntime` 是 DD 的**独立 API 入口**，它要调用方自己给 xclbin；
   而官方真正在跑的是 **EP 入口**（`RyzenAILightExecutionProvider`），PDI/kernel 由 EP 内部解析，**根本不经过 `op_pdi_map`**。
   我上一轮在找一个不存在的文件——方向错了。
2. **EP + 分区缓存的热会话 = 0.226–0.258 s**，且 **3/3 节点全在 NPU 上**（`RyzenAILightExecutionProvider`）。
   对照官方 VitisAI EP 路径的 15–26 s/分片 ⇒ **快约 100×**。这就是 B1 要的东西。
3. **冷编译不在运行时**：`compile_fusion_rt=1` 配空目录直接报 `No valid cache directory found`。
   真正的"编译"是离线 **partitioner**：`onnx_utils partition` 实测 **8.1–12.0 s**（6 节点图），
   产出 `replaced.onnx`（1,492 B）+ `cache/`（meta + 5 个 const，21.5 MB）。
4. **但数值未验**：对 fp64 参考 **max abs 0.133 / 相对 rms 17.4%**。CPU 对照做不了（`CastAvx` 是 EP 专有算子），
   也没有 dd_live 的存档输出可比。**"字节一致"判据本轮未达成**。

---

## 1. 线索是怎么合流的

上一轮我在找 `stx_SD_unified_fastpm_2x4x4.xclbin`。本轮把三条线索并起来：

| 线索 | 事实 | 含义 |
|---|---|---|
| `_dd22_live.py`（旧 DD 实验驱动） | `DD_PLUGINS_ROOT = C:\RyzenAI\1.8.0\GenAI-SD\lib\transaction\stx`、`DD_ROOT = deployment`、EP 名 `RyzenAILightExecutionProvider` | **成功路径是 EP，不是 FusionRuntime** |
| `stx/` 目录 | 只有 `sd_fastpm_DiT.zip`(62.7 MB) / `sd_fastpm_Unet.zip` / `sd_preemption.zip` | 10,338 件全是 `.bin/.json`，**0 个 xclbin** |
| 全盘 `C:\RyzenAI` 搜 `*.xclbin` | **0 个** | 安装树里根本没有 xclbin 文件 |
| `onnxruntime_providers_ryzenai.dll`（RyzenAILight EP） | `xclbin2` magic **0 个** | EP 也不内嵌 xclbin |
| `onnxruntime_vitisai_ep.dll` | `xclbin2` magic **10 个** | 那是 VitisAI EP 自己的，与 DD 无关 |

**⇒ "补一个 xclbin 文件"这条路是死的**：DD 的 EP 入口不需要它，FusionRuntime 入口要的那份从来不在盘上。

### 1.1 旧实验的完整配方（`_dd22_live.py`）

```python
os.environ['DD_PLUGINS_ROOT'] = r'C:\RyzenAI\1.8.0\GenAI-SD\lib\transaction\stx'
os.environ['DD_ROOT']         = r'C:\RyzenAI\1.8.0\deployment'
for k in ('TMP','TEMP','TMPDIR'): os.environ[k] = <可写目录>   # fusion_rt 用 tmpfile_s，必须进程内设
ort.register_execution_provider_library('RyzenAILightExecutionProvider', <onnxruntime_providers_ryzenai.dll>)
so.add_provider_for_devices(devs, {'dd_cache': <cache>, 'onnx_custom_ops_const_key': '', 'compile_fusion_rt': '0'})
so.register_custom_ops_library(<同一个 dll>)
```

三个易踩点：①`TMP/TEMP/TMPDIR` 必须在**当前进程**里设（只设 subprocess env 不够）；②必须 `register_custom_ops_library`；
③必须加载 **`replaced.onnx`**（分区产物），不是原图——我第一版喂原图，EP 一个节点都不接，全落 CPU，
那时测到的 0.466 s 是 **CPU 回退**的假阳性。

---

## 2. 实测

图：`dd_live/ctrl2.onnx`（6 节点，SD3-medium 形状 S=1024 / H=1536 / F=6144，75.5 MB）
→ partition → `out/replaced.onnx`（3 节点：`CastAvx` → `DynamicDispatch(ln0)` → `CastAvx`）。

| 量 | 实测 | 备注 |
|---|---:|---|
| **离线 partition** | **8.1 / 12.0 s** | 两次历史日志；产 `replaced.onnx` 1,492 B + `cache/` 21.5 MB |
| **热会话（EP + 分区缓存）** | **0.226 / 0.258 s** | 0 新缓存文件 |
| 冷会话（空缓存 + `compile_fusion_rt=1`） | **失败** | `No valid cache directory found` |
| 单次 run | 0.089–0.131 s | S=1024 的两级 GEMM |
| 节点归属 | **3/3 RyzenAILightExecutionProvider** | 真在 NPU 上 |
| 对 fp64 参考的误差 | max 0.133 / 相对 17.4% | **未验，见 §3** |

**对照官方 VitisAI EP 路径**（`_prof128.log`）：单分片热会话 15.1 s（安静）/ 25.0 s（负载）。
**DD/EP 路径 0.23 s ⇒ 约 65–108×。**

---

## 3. 数值：未验（诚实负结果）

对同一输入，用 `ctrl2.onnx` 的 fp32 权重在 fp64 下重算整链（LN → W_up → +b → Gelu → W_dn → +b）：

| 量 | 值 |
|---|---:|
| ref rms | 0.767 |
| max abs err | **0.133** |
| mean abs err | 0.0142 |
| 相对 rms | **17.4%** |

17.4% 远大于任何单纯量化模型的预期（bfp16 权重 ≈ 2e-3 量级、fp16 累加 ≈ 1e-3 量级），所以有三种可能：
①DD 内核用的是 16 位累加（txn 名 `abfp16wbfp16acc16f` 里的 `acc16f`）且 K=6144 下有严重抵消；
②Gelu/epsilon 的实现与 ONNX 参考不同；③我的参考本身不是这张图的语义。**现有证据分不开。**

**为什么没做成对照**：
- CPU 跑不了 `replaced.onnx`：`CastAvx` 是 EP 专有算子，CPU EP 报 `NOT_IMPLEMENTED`；
- `dd_live` 没有存档输出（只有 meta/const/state），没有 golden 可比；
- 要判定，需要一次**受控的**单算子对拍（比如只用 `ln0` 的 LayerNorm，去掉两级 GEMM），或拿到当年的输出存档。

**所以父任务的判据"execute 打通（同输入字节一致）"：execute 通了，字节一致**没有**达成。**

---

## 4. 这对四段解剖意味着什么

| 段 | 官方 VitisAI EP 路径 | DD/GenAI-SD EP 路径（本轮） |
|---|---|---|
| ① 编译 | 350–380 s/片（VAIP+AIE） | **partition 8.1–12.0 s/图**（离线，可缓存） |
| ② 会话 | 15–26 s/片 | **0.23 s** |
| ③ 权重装载 | O(参数)，~50–70 ns/param | **未测**（const 21.5 MB，EP 内部装载） |
| ④ 首跑 | ≈0 | ≈0 |

**关键判断：上一轮"几百秒是 O(参数) 权重装载"的结论，只对 VitisAI EP 路径成立。**
DD 路径把同一件事做到了 0.23 s——不是因为权重变少，而是因为**它不走 VAIP 的逐元素准备**。
这恰好印证了 E1 的猜想方向（状态层可缓存、真正的开销是 EP 侧处理），只是实现路径不是 `FusionRuntime.load_state`，而是 **EP + 分区缓存**。

---

## 5. 组装方案（修订）

```
[离线]                                    [服务启动]              [每张图]
onnx_utils partition (8-12 s/图)           EP 建会话 0.23 s/图      execute
  → replaced.onnx + cache/(meta+const)      17 片 ≈ 4 s             8 步
  → 落盘复用（0 新文件）                     const 按需 mmap        VAE 0.57 s
```

**启动从 425.5 s → 约 4 s（17 片 × 0.23 s）**，这是本轮最大的确定收益。

但**先要过数值关**：在 17.4% 的误差被解释清楚之前，这条路径不能作为产品路线，只能作为"启动开销可压到 0.23 s"的存在性证明。

### 5.1 待办（按阻塞度）

| # | 事 | 为什么 |
|---|---|---|
| **C1** | 数值对拍：单算子（纯 LayerNorm）先对，再逐级加 GEMM | 17.4% 不解释清楚，一切免谈 |
| **C2** | 确认 `acc16f` 假设 | txn 名里写着 16 位累加，若属实这是**官方路线的精度上限**，不是 bug |
| **C3** | 17 片规模验证 | 本轮只有 1 个融合 op；Z-Image 分片约 26 个子图，0.23 s 能否保持未知 |
| **C4** | B3：M=96 | 仍无预编译 key；但 DD 路径的 txn 是 `sd_fastpm` 里的 shape 表，与 `llm_txn_query` 不是同一套，**需要重查** |

---

## 6. 复现

```powershell
$py = '<python312>\python.exe'
$d  = '<checkout>'
& $py -u "$d\_scratch\sessA_b1_ep.py"      # EP 会话（原图，CPU 回退假阳性）
& $py -u "$d\_scratch\sessA_b1_repl.py"    # ★ EP 会话（replaced.onnx，3/3 在 NPU）
& $py -u "$d\_scratch\sessA_b1_ab.py"      # 冷/热 A/B + 数值
& $py -u "$d\_scratch\sessA_b1_num.py"     # CPU vs NPU 数值对照（CPU 不可用）
& $py -u "$d\_scratch\sessA_zipls.py"      # stx 事务包清单（0 xclbin）
& $py -u "$d\_scratch\sessA_xcldump.py"    # 从 EP DLL 抽 xclbin（10 个，与 DD 无关）
```

---

## 7. 纪律与边界

- **预测先行**：本轮预判"xclbin 藏在某处"——**否证**（全盘 0 个）；预判"旧脚本里有配方"——**成立**。
- **负对照必配**：①原图 vs `replaced.onnx`（原图全落 CPU，暴露了 0.466 s 假阳性）；②空缓存冷会话（失败，证明运行时不做编译）；③CPU 对拍（不可用，已记录）。
- **诚实负结果**：数值 17.4% 未解释；17 片规模未验；B3 未测。
- **自我更正**：上一轮 §1.5-N2 说"DD 自己的 xclbin 不在磁盘"——**这句话本身没错，但结论错了**：
  DD 的 EP 入口根本不需要 xclbin。把它当成缺口去补，是找错了门。
- **产品零变化**：只新增 `_scratch/sessA_b1_*`、`sessA_zipls.py`、`sessA_xcldump.py` 与本文。
- **看门狗 ≠ 墙钟**：所有秒均为 `time.perf_counter()`。
- **51.3 TOPS / F-14.1**：本轮无吞吐测量、无器件独占需求，未触碰。
