# BIGPIPE3-PREDICTIONS — 记录与缺陷声明

日期：2026-10-31｜owner：t22_bigpipe3

**本单没有在首跑前把预测写成文件，这是流程缺陷**；任务消息里的四个目标值（0.999 / 0.999 / 150 s+0.99 / 75 s）不是本文件事后补的。

首跑前实际使用的模型（口头）：

- A：devhand 把 FFN host 550–600 ms/层砍到 ~180 ms/层，512² 8 步 ≤150 s。
- B：`qkv_to_image` 修 72 B 子块 scatter 后单层 corr ≥0.999；1024² 1 步设备 9.4 s + attention 37.9 + 其余 27 ≈ 75 s。
- C：state 放 ObjectFifo 或整 q-block 单 kernel 后 seq128 ≥0.999。

实测对账见 `BIGPIPE3-REPORT.md` §5.1。**A 的 host 项低估 2×，B 的 host 重排未定价，C 的 carry 算术残差未定位到行。**
