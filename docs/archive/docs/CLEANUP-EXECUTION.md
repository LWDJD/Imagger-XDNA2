# CLEANUP-EXECUTION — 最小化清理台账(2026)

**原则**:只删不属于当前路线(IRON+XRT)的大件(单件 >= 400 MB),最小化操作次数;长尾小文件、日志、存疑项一律不动。
**前置盘点**:CLEANUP-INVENTORY.md / CLEANUP-INVENTORY-2.md(可再生性分级与 NEVER 名单)。
**范围**:仅 <workspace>\zimage-port\_probe_out。

## 删除结果(释放 17,825,508,926 B ≈ 16.6 GiB;12,278 文件/33.48 GB -> 5,477 文件/15.66 GB)

| # | 删除目标 | 大小 | 性质 | 再生成依据 |
|---|---|---|---|---|
| 1 | zimage_loop\ 36 个非基线子目录(gx*/out_*/tmp 等) | ~14.4 GB | DD 路线迭代产物(非当前路线) | zimage-port\_GL2-FINAL-HANDOVER.md(partition 配方) |
| 2 | fresh_cache\(vendor .rai 原件) | 418,881,856 B | **哈希核验通过后删**:与 vendor_artifacts 备份 sha256 一致(DCE407A3…412A10) | 不可再生成,但备份在 zimage-bfp16-npu\vendor_artifacts\ |
| 3 | packverify\(65 个 gemm_*.onnx) | 1,205,478,876 B | 探针 ONNX | 探针脚本 + 结论已入 KNOWN-ISSUES/FINDINGS |
| 4 | p2_wholegraph\ | 814,842,660 B | partition const 转储 | 重跑 partition 探针(分钟级) |
| 5 | bfp16_deadlock\ | 550,264,384 B | 死锁复现日志 | 结论已入 _gl*_REPORT.md |
| 6 | gl_work\ | 441,278,209 B | GL 路线编译缓存 | _GL2-FINAL-HANDOVER.md |

**哈希门(执行前置条件)**:
- zimage_loop\weights.bfp16:哈希核验时路径已不存在(早前清点的 6.91 GB 副本),**自动跳过,未删任何权重**;
- fresh_cache\*.rai:与 vendor_artifacts 备份哈希一致 -> 删。

## 保留(未动)

- 当前路线全部产物:zimage-port\weights.bfp16、_probe_out\weights.bfp16(check_handoff.py 期望位)、skeleton_all_v2\(含 weight_map.json 506 权重名注册表)、mem_profile\、shards_i1024_c32\(32.98 MB io 引用件在内);
- 备注(清理后核实,非本次删除所致):_probe_out 根下未见 verify_all.json、shards_i1024_c32\io.npz 路径与 HANDOFF-2026-10-12.md 的记载不一致(该目录有 1 个 32,983,238 B 文件)——是清理前既有状态,下次冷启动时向文档作者澄清;
- zimage_loop\ 8 个作者基线(out_op1/out_fix2/out_r1/t_a4/t_w1/g_ca1/out_nomul2/out_nomul2_keep)+ plugins\(存疑项,保守保留);
- 作者 NEVER 名单:aiert_build、cast_nb3、dd103t、plugins_clean 等;
- ryzenai_nuget\(765 MB 供应商包缓存:无下载政策下不可再获);
- 一切代码/脚本/文档;_models\(检查点);vendor_artifacts\;
- 所有 < 400 MB 的探测目录、全部散落日志(最小化原则,不动)。

## 再生成文档

CLEANUP-INVENTORY.md / CLEANUP-INVENTORY-2.md(分级与名单)、zimage-port\_gen_cleanlist.py(作者分类器)、zimage-port\_GL2-FINAL-HANDOVER.md(zimage_loop 配方)、zimage-port\build_bfp16_weights.py(weights.bfp16)、encode_zimage_bfp16_v2.py + _bfp16_pack.py(bfp16 v2)、export_shards_fp32.py(shard)、_dd4_iter.py/_dd5_vi.py/_probe_merge_shards.py(DD 系)、HANDOFF-2026-10-12.md(必留产物清单)。
---

## 补记(2026-10-19):漏记的 `bfp16_weights_v2` 删除

上面的"删除结果"表只记了 6 项、合计 17,825,508,926 B。实测**还有一次删除未入账**:

| # | 删除目标 | 大小 | 性质 | 再生成依据 |
|---|---|---|---|---|
| 7 | `zimage-port\_probe_out\bfp16_weights_v2\` | **6,924,417,834 B(6.449 GiB),2 文件** | 派生权重编码:`weights.bfp16` 6,924,272,328 B(**521 张量** / 6,154,908,736 params)+ `manifest.json` 145,506 B;2026/10/09 16:37 | `zimage-port\encode_zimage_bfp16_v2.py` + `_bfp16_pack.py`,确定性纯 NumPy 重编(REGENERABLE-SLOW) |

**为什么漏记**:本台账范围写作"仅 `zimage-port\_probe_out`",而 6 项清单只枚举了
`zimage_loop\`、`fresh_cache\`、`packverify\`、`p2_wholegraph\`、`bfp16_deadlock\`、`gl_work\`;
`bfp16_weights_v2` 在 `_probe_out` **根**下,属同一次清理却未回填台账。

**证据**:该目录现已不存在(`Test-Path` = False);字节/张量数取自 `CLEANUP-INVENTORY.md`:95-101、
`CLEANUP-INVENTORY-2.md`:39/:92(记为 DELETABLE-CANDIDATE,6.449 GiB)——即**盘点记了、执行台账没记**。

**能否从 `_backup_staging` 02 恢复**:**不能**。`02_weights.bfp16` 与 `zimage-port\weights.bfp16`
同大小(6,913,965,970 B,按 MANIFEST 为硬链接副本),是**交付格式的 506 张量**版本
(6,143,056,384 params,1.125 B/param);v2 是 **521 张量**、6,924,272,328 B 的另一个文件。
两者不是同一产物,互相不能替代。

**实际可恢复路径**:源权重仍在 `zimage-port\_models\Z-Image-Turbo\transformer\`
(3 片 safetensors,合计 24,619,690,888 B ≈ 24.62 GB),用 `encode_zimage_bfp16_v2.py` 重编即可。
耗时**只有估计**(几十分钟至小时级),未实测——`CLEANUP-INVENTORY-2.md`:183 已声明该估计无测量支撑。

**影响**:不影响 IRON + XRT 当前路线;不是 shard/VAIP 再生的输入(INVENTORY:100);不影响 WebUI 运行。

**更正后合计**:17,825,508,926 + 6,924,417,834 = **24,749,926,760 B ≈ 23.05 GiB**(约 24.75 GB)。

---

## 补记 2026-10-19(t21_clean_b8):`_probe_out\weights.bfp16` **门禁未过,未执行删除**

**结论:本单执行了验证、未执行删除。** 目标文件原封未动,大小与 inode 不变,`Remove-Item` 未调用。

| 项 | 实测值 |
|---|---|
| 目标(解析后绝对路径) | `<workspace>\zimage-port\_probe_out\weights.bfp16` |
| 字节数 | 6,913,965,970 B(6,593.67 MiB) |
| st_ino / st_nlink | 17,169,973,579,517,768 / **1** |
| 文件属性 | `st_file_attributes=0x20`(ARCHIVE);`FILE_ATTRIBUTE_REPARSE_POINT=False`;`st_reparse_tag=0`;`S_ISLNK=False`;`S_ISREG=True` → 真文件,非符号链接/Junction |
| 批准人 | 负责人 2026-10-19 勾选 B8(授权已下达;因门禁未过而未执行) |
| 执行结果 | **未删除** |

### 门禁 U2(同内容)—— 四重证据,第 ④ 门失败

| 证据 | 实测 | 判定 |
|---|---|---|
| ① 非重解析点 | `st_file_attributes=0x20`;`FILE_ATTRIBUTE_REPARSE_POINT=False`;`st_reparse_tag=0`;`S_ISLNK=False`;`S_ISREG=True` | PASS |
| ② nlink | `st_nlink=1` | PASS |
| ③ inode 互异 | TARGET=`17169973579517768`;CANON=`2251799813816683`;BACKUP=`2251799813816683`(CANON 与 BACKUP 同 inode 即硬链接定义本身;`fsutil hardlink list` 亦证实:canon 列 2 名、target 列 1 名) | PASS |
| ④ SHA256 三者全等 | **不等** | **FAIL** |

SHA256 实测(均 6,913,965,970 B,Python `hashlib` 64 MiB 流式):

| 文件 | SHA256 |
|---|---|
| `zimage-port\weights.bfp16` | `4ff5b6a66ea3b3ae5a29ecc601097d50eade61979c222f8e0c58d3a272cb8042` |
| `_backup_staging\02_weights.bfp16` | `4ff5b6a66ea3b3ae5a29ecc601097d50eade61979c222f8e0c58d3a272cb8042` |
| `_probe_out\weights.bfp16`(目标) | `bfac45f89fe302c4e3f2840d7c223ab2c17bb0f492434e9780e574cf2ed1cb71` |

**关键更正**:`T21-DISKAUDIT.md`:440 的"与 `zimage-port\weights.bfp16` **同尺寸同内容推定**(未做 sha256)"**不成立**。全量逐字节比对结果:
- 头部 87,946 B **完全相同**(头部 SHA 同、506 张量、张量表 **0** 条差异);
- payload 差异 **27,429,272 B = 0.397%**,散布 413 个 64 MiB 块,首个差异偏移 **89,158**(payload 起点 87,954,即 payload 内 +1,204);
- mtime:目标 `2026-10-12 16:23:43`(旧);`zimage-port\weights.bfp16` 与 02 备份 `2026-10-14 22:37:24`(新)。

即:目标疑为 **10-12 旧构建**,canon 为 **10-14 重建**;0.397% 的量级与 `zimage-port\weights_audit.py`:3-6 记载的"`_quantise` 2 的幂修复影响部分张量旧字节"吻合。**旧构建无法由新版反推还原**——按原计划删除,销毁的不是"冗余副本",而是唯一存留的旧版构建。

### 门禁 U1(引用复核)—— 真命中

`*.py` 硬编码读路径 4 处:

| 位置 | 代码 | 性质 |
|---|---|---|
| `zimage-port\skeleton_runner.py`:193 | `bfp = HERE / "_probe_out" / "weights.bfp16"` | 运行器权重回退路径 |
| `zimage-port\build_rai_weights.py`:49 | `WFP = HERE / "_probe_out" / "weights.bfp16"` | 重建 `.rai` payload 的源 |
| `zimage-port\verify_bfp16_weights.py`:33 | `DEFAULT_FILE = HERE / "_probe_out" / "weights.bfp16"` | 权重校验默认文件 |
| `zimage-port\check_handoff.py`:24 | `"zimage-port/_probe_out/weights.bfp16"` | 交付物必检项 |

另 `zimage-port\dedup_models.py`:42 将其列为去重候选。`CLEANUP-INVENTORY.md`:79 的作者 NEVER 名单**未**包含它(不构成额外拦截,亦未提供保护)。

加重情节:`_probe_out\skeleton_all_v2\manifest.json` 的 `skeleton_dir` 即 `_probe_out\skeleton_all_v2`,该目录下**无** `weights.bfp16`;故 `skeleton_runner.py` 走回退分支时**读的正是本目标文件**。

### 可恢复性(未删除,故无需恢复)

若日后仍要删除,前置条件:(a) 先处置上述 4 处读路径;(b) 负责人明确接受"10-12 旧构建永久丢失"。
`zimage-port\weights.bfp16`(仍在)与 `_backup_staging\02_weights.bfp16`(仍在,同 inode)只能恢复 **10-14 新版**,**不能**恢复旧版。

### 一并观察到、**未动**的事实

- 另有 2 个同尺寸(6,913,965,970 B)、各 `nlink=1` 的独立副本:`zimage-port\npu-conversion\_probe_out\weights.bfp16`、`zimage-bfp16-repo\_probe_out\weights.bfp16`。故实测**2 个物理 inode**(④⑤ 为指向③的 Junction 别名)。原文此处记"4 个物理 inode、T21-DISKAUDIT.md:164 计数偏低",系按路径名计数导致的误判。【更正 2026-10-19 t21_wcensus】依据:Get-Item `.LinkType`=Junction、三路径目录同 ino(14918173765808039)、三文件同 ino(17169973579517768)/nlink=1;三路径内容同源(旧版 SHA256 bfac45f8…),故"2 个物理副本"正确。
- 本单未触碰 `_scratch/vae`(并发代理 3b1 在用),未动回收站、A 档、E 项及其余任何项;除本文件与该观察外零修改。
- `docs/zimage-bfp16-npu-questions.md`:8/:83 的"5 项已落实/已落实 = 5/15"汇总数,因 E 行更正后实为 4,本次**未改**(越界),留待负责人处置。


