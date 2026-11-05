# SESS-G2 — 坏件是"未转换的编译器 patch.json"，运行时要的是 external_buffers

> t22_sessA 第十一轮（2026-10-24）。裁决走 C；本轮验 G2 的定性。
> **结论：G2 的修法确定了，但方向和父会话的描述相反——好件不是目标 schema，坏件也不是。**

---

## 0. 一句话

**运行时（`ctrl_pkt_utils.cpp:106`）要的是 `external_buffers` 这个 key；**
而 65 B 坏件与 `gl16_pack` 好件**都是编译器 patch.json 的 `ctrl_pkt_patch_info` schema**。
**两者都不是运行时想要的形状。** 真正的正确产物是 `_dd103t.py` 转换出来的那一份。

---

## 1. 证据链（三轮实测，逐字）

| 轮次 | 输入 | 报错 | 含义 |
|---|---|---|---|
| 原状 | `ours*.zip` 原始坏件 | `parse error at line 4, column 5: unexpected ']'` | **JSON 本身非法**（空数组写成孤立 `]`） |
| 本轮第 1 步 | 我修成合法 JSON `{"version":"1.0","ctrl_pkt_patch_info":[]}` | `Transaction not found: sd_fastpm_elwmul_..._ctrl` | JSON 过了，但 `elwmul` 事务缺失 |
| 本轮第 2 步 | 补齐所有包 + 全部修成合法 JSON | **`key 'external_buffers' not found`** | **schema 不对**——运行时根本不读 `ctrl_pkt_patch_info` |

**⇒ 空数组不是"缺补丁"，而是"**根本没做 patch.json → external_buffers 的转换**"。**

---

## 2. 正确的生产者长什么样

`_dd103t.py⟧ 的流程才是对的：

```python
pj = json.loads((BUILD / "patch.json").read_text())        # 编译器产物：ctrl_pkt_patch_info
entries = pj.get("ctrl_pkt_patch_info", [])
CTRL = (BUILD / "ctrl.bin").read_bytes()
def cb(lst, logical_id, name):                              # 转成运行时 schema
    return [{"logical_id": logical_id, "offset_in_bytes": 0, "name": name,
             "control_packet_patch_locations": [
                 {"offset": e["offset"], "size": e["size"],
                  "operation": "read_add_write"} for e in lst]}]
meta = {"external_buffers": {
    "buffer0": {"xrt_id": 0, "size_in_bytes": int(M*K*2.0),  "coalesed_buffers": cb(g[0], 0, "compute_graph.ifm_ddr")},
    "buffer1": {"xrt_id": 2, "size_in_bytes": int(M*K*1.125),"coalesed_buffers": cb(g[2], 2, "compute_graph.ofm_ddr")},
    "buffer2": {"xrt_id": 1, "size_in_bytes": 128,          "coalesed_buffers": cb(g[1], 1, "compute_graph.wts_ddr")},
    "buffer3": {"xrt_id": 3, "size_in_bytes": len(CTRL),    "coalesed_buffers": []}}}
```

**三个 buffer 的 `size_in_bytes` 需要该 txn 的 (M,K) 与 dtype**：输入 bf16 = M·K·2.0、输出 bfp16 = M·K·1.125、
权重 buffer 在 cast 用例里是 128 B（固定）。这些都能从 txn key 里解析出来（`<op>_..._<B>_<M>_<K>_<N>`）。

---

## 3. 我这一轮做到哪一步

| 步 | 状态 |
|---|---|
| 复现坏件（非法 JSON） | ✅ |
| 修成合法 JSON | ✅ 6 个文件，0 残留非法 |
| 补齐缺失的 elwmul 事务（全包 rebuilt） | ✅ 10 个 zip，0 非法 meta |
| **暴露出真正缺口：schema 必须是 external_buffers** | ✅ **这是本轮的关键发现** |
| 写通用 patch.json → external_buffers 转换器 | ⬜ 未做（时间） |
| 80 子图真图加载 | ⬜ 卡在上一行 |

**所以父会话说的"G2 是 schema 映射错、成本远低于估计"——方向对，但对象要改**：
不是"反推 gl16_pack 好件的 schema"（好件也不是运行时 schema），
而是"**把编译器 patch.json 按 `_dd103t.py` 的映射转成 external_buffers**"。
这个转换器是纯 Python、无需编译器、无需提权——**成本确实很低**，但工作量在"为每个 txn key 解析出 (M,K,dtype)"。

---

## 4. 下一轮的具体动作（可直接执行）

1. 写 `repack_meta.py⟧：输入 txn key + `patch.json` + `ctrl.bin`，输出 `external_buffers` 形态的 `_ctrl_meta.json`；
2. 对 80 子图图需要的**全部 key** 批量转换（key 从 `replaced.onnx` 的节点属性里读）；
3. 重建 plugins，加载 `out_fix2`（80 子图），拿到 **26+ 规模点**；
4. 通过后按 C3 的曲线外推 442 子图的启动，并与 4.7 s 对表。

---

## 5. 纪律与边界

- **预测先行**：预判"修好 JSON 就能加载"——**否证**（缺 elwmul → 再否证于 schema）。
- **负对照必配**：三步报错各不相同（parse → missing txn → missing key），逐步排除了"只是 JSON 坏"这个过窄假设。
- **诚实负结果**：转换器未写、80 子图未加载、26+ 规模点未取。本轮产出是**把 G2 从"需要真编译器"精确化为"需要一次纯 Python 的 schema 转换"**。
- **产品零变化**：只新增 `_scratch/sessA_fixpack.py⟧、`sessA_fixpack2.py⟧、`sessA_load80.py⟧ 与本文；
  所有重建都落在 `_probe_out/zimage_loop/plugins_fix2/⟧，未改动原始插件包。
- **51.3 TOPS / F-14.1**：本轮无吞吐测量、无器件独占，未触碰。
