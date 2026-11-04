# t22_bigpipe — 预测先行（写于任何器件动作之前）

时间：2026-10-30 22:0x (+08:00)｜owner：t22_bigpipe｜器件窗：webui 7874（单实例，seed 7）

## 0. 口径

- 全部按 **512²/1024²/2048²** 计；小图只作对照，不作速度结论。
- 512²：tokens=1024，M=SEQ=1056（cap 32 + image 1024）。
- 768²：M=2336（25 个 96 行块）；1024²：M=4128（43 块）；2048²：M=16416（171 块）。
- 当前产品路径 = webui/chunked.py（exec_mode=chunked）：投影/FFN/o 投影走已交付 M=96 镜像，
  **attention 在 host f32 torch**。manifest 512x512/manifest.json 已写死 attention=host-f32-torch。

## 1. 512² 全跑（8 步）预测

| 量 | 预测 | 依据 |
|---|---|---|
| 每步 loop wall | **35–40 s** | 2026-10-21 同路径 1 步实测 37.49 s（job 20261021-190057-055663） |
| 8 步 loop | **285–320 s** | 8 × 37.5 s，含首层冷启动 |
| 整任务 total | **340–380 s** | + startup ~14 s + VAE 5 s + 收尾 |
| dispatches | **13,200**（1650/步，0 retry） | 30 层 × 5 × 11 块 × 8 步 |
| device | **17–19 s/步** | 同 1 步实测 18.06 s |
| staging | **5.0–5.8 s/步** | 同 1 步实测 5.43 s |
| pack（host 打包） | **2.5–3.0 s/步** | 同 1 步实测 2.78 s |
| host 桶（含 attention） | **52–58 s/步** | 同 1 步实测 55.45 s；attention 约 7.3 s/步（0.243 s/层） |
| **attention（host f32）** | **6.5–8.0 s/步** | chunked.py 文档：480p 0.243 s/层；512² 同量级 |
| proj 器件（att/q,k,v,o） | **6.5–7.2 s/步** | 1 步实测 att/q 1768.8 + k 1716.4 + v 1711.9 + o 1632.8 ms |
| FFN 器件 | **10.8–11.5 s/步** | 1 步实测 11232.1 ms |

## 2. 768² / 1024² 缩样预测（各 1 步）

| 尺寸 | M | 块数 | 预测 loop wall | 预测 dispatches | 预测 device |
|---|---|---|---|---|---|
| 768² | 2336 | 25 | **55–65 s** | 3,750 | 36–42 s |
| 1024² | 4128 | 43 | **95–115 s** | 6,450 | 62–72 s |

若 1024² 的实测落在预测带内，则 2048² 用 **线性项（proj/FFN/host glue ∝ M）+ 平方项
（attention ∝ M²）** 外推，外推口径单独标注；否则只用 768² 做单点缩放。

## 3. 先写死的判据

1. **attention 归属**：512²/768²/1024²/2048² 都必须是 **host f32 torch**（NPU attention_core 只在
   M≤288/SEQ≤304 的交付尺寸可用）；若实测发现任何一段走 device ip，本预测作废并重跑。
2. **N² 项**：host attention 的 S/P 矩阵按 M²×HEADS×4 B 计，512² 每层 267.6 MB，8 步 64.2 GB；
   这是“数据在哪落地”的第一硬账。
3. **dispatch 数**：chunked 路径 dispatches = 30 × 5 × ceil(M/96) × steps；不含 VAE。
4. 所有时间以 job.json 的 timing + bridge log 的 [t3d]/[t22g] 为准；看门狗/ERT 状态不作为墙钟。