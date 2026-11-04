# OBS-id79 — 幅值界 ulp 恰好 2.000（零余量）的观测归档

**状态：归档。门限不变。** 本条只把既有观测固定成可引用的记录；`gate_ulp_ok(u) = (u <= 2.0)`
（`_scratch/vae/t21_3b2_plan.py:658-659`）**一字未改**，本单也没有为该行调任何参数。

| 项 | 值 |
|---|---|
| 归档日期 | 2026-10-20 |
| 原观测 | 2026-10-19 19:51（`t21_3b2vf` 单，设备测量） |
| 行 id | **79** |
| 算子 | GroupNorm（`gn`，stream 形态） |
| 名字 | `gn_stream_C128_S65536`，variant `gnv_C128_S65536` |
| 位置 | `up_blocks.3.resnets.0.norm2` |
| 几何 | C=128, S=65536, H×W=256×256, groups=32, eps=1e-6, L=262144 |
| 分块 | 子平面，Cs=4096，64 chunk/group，fill 4096，drain 2048 |
| **raw ulp** | **271.0** |
| **幅值界 ulp（门的量）** | **2.000** |
| 门 | `<= 2.0` → **PASS，零余量** |
| bit-exact | **0.9988372325897217** |
| oracle PSNR | 77.87106749028561 dB |
| device wall | 11.72 ms（整行 gate 20.54 s，其中 prep 19.35 s） |
| 跨进程确定性 | raw sha256 两次相同：`9b9fb6a5bff41ad3af5432b7bfa119a25caaaba64459665db8e1f472e6a4c25f` |

证据文件（全部从磁盘读出，无手抄）：

* `_scratch/vae/out/t21_3b2vf/79_gn_stream_C128_S65536.json` —— 本行的 verify 原始输出
* `_scratch/vae/out/t21_3b2vf/closure.json` —— 92 行收口表（本行 `max_ulp: 2.0`）
* `_scratch/vae/out/t21_3b2vf/detpass.json` —— 跨进程重放（id 79，`byte_identical: true`）
* 正文记录：`_scratch/vae/T21-3B2VF-REPORT.md` §4.4（逐行表）、§7.3（零余量）、§8.5（3b3 余量）

---

## 1. 这个数是怎么算的（逐行引自源码，`_scratch/vae/t21_3b2_plan.py`）

```python
# :640-648
def ulp_pair(got, ref, mag) -> tuple:
    """(max raw ulp, max ulp against the magnitude bound)."""
    ref = np.asarray(ref, np.float64); got = np.asarray(got, np.float64)
    mag = np.asarray(mag, np.float64)
    ulp_r = np.maximum(_bf16_ulp(ref), 2.0 ** -126)
    ulp_m = np.maximum(_bf16_ulp(mag), 2.0 ** -126)
    d = np.abs(got - ref)
    return float((d / ulp_r).max()), float((d / ulp_m).max())

# :651-655
def _bf16_ulp(x):                      # bf16 ulp = 2**(exp-7)
    xf = np.where(x == 0.0, np.float64(2.0 ** -126), np.abs(x))
    exp = np.floor(np.log2(xf)).astype(np.int64)
    return np.exp2(exp - 7)

# :1169-1177  (GN 的幅值界, 与设备分块无关, f64 从 bf16 输入重算)
for g in range(groups):
    seg = xb[g * cg:(g + 1) * cg].reshape(-1)
    mu = seg.mean(); var = ((seg - mu) ** 2).mean(); rstd = 1.0 / np.sqrt(var + EPS)
    mag[sl] = (np.abs(xb[sl] - mu) * rstd * np.abs(gbv[sl])[:, None]
               + np.abs(bbv[sl])[:, None])
max_ulp, max_ulp_mag = ulp_pair(Y, ref_b, mag)   # :1178
pass_ = bool(gate_ulp_ok(max_ulp_mag))           # :1185 -> <= 2.0
```

即：`幅值界 ulp = max |device_Y - bf16(model_exact)| / bf16_ulp(mag)`，
`mag = |x-mu| * rstd * |gamma| + |beta|` 是 GroupNorm 输出的**逐元素理论上界**。

---

## 2. 为什么门用的是幅值界，而不是 raw ulp：271.0 vs 2.000

同一行、同一份 raw，两个数差 **135 倍**：`raw ulp = 271.0`，`幅值界 ulp = 2.000`。
原因是 GN 的输出在通道内做的是 `(x-mu)*rstd*gamma + beta`，**接近 0 是正常抵消**，
拿 |y| 当 ulp 分母会重演 `step4_run --verify-only` 的 ulp 爆炸（见
`docs/T22-QUICKFIX-REPORT.md` §1：一个零参考元素把 max ulp 推到 1.66e35）。
幅值界把分母换成“这个输出**本可以**有多大”，既不看运气（参考恰好为 0），
也不迁就设备（mag 由 f64 从 bf16 输入独立重算，与设备的分块顺序无关）——
所以门没有被“调松”，**2.000 是一个真实、离散、可复现的边界值**。

---

## 3. 零余量意味着什么（以及什么时候必须重门）

* 门是 `<= 2.0`，实测**恰好 2.000**：通过，但**没有任何余量**。
* ulp 是 2 的幂、误差落在 bf16 网格上，所以 2.000 是离散可达值，不是浮点噪声；
  也因此“再多半个 ulp”就会直接翻红。
* 已经做过的事：该行在 1/10 抽样中被选为跨进程确定性样本，两个全新进程各跑一次，
  raw 的 sha256 相同（`detpass.json`）——所以 2.000 **不是**一次偶然读数。
* **会先翻这一行的动作**（下一个单若做，需重门）：换参考模型、换 eps 口径、
  改 chunk/group 划分（f32 累加顺序）、换 bf16 舍入模式、或换 `_bf16_ulp` 的 floor 约定。
* 3b3 的现状：`webui/evidence/T21-3B3-VAE.md:165` 已按“id 79 本来就在 2.000 零余量上”处理，
  与本归档一致。

---

## 4. 结论

| 问题 | 回答 |
|---|---|
| 门限改了吗 | **没有**（`<= 2.0` 原样） |
| 数值改了吗 | 没有；2.000 是 2026-10-19 的实测值，本次仅归档 |
| 这条记录能复算吗 | 能：证据三个 JSON + 上面的源码行号 |
| 风险 | 该行零余量；任何口径变动都要重门，不能默认继承 |
