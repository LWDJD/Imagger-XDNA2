# README-WEBUI — Z-Image-Turbo NPU 生图 WebUI

生产可用的单页 WebUI,把已验证的 t13 8 步 × 30 层 + 4 refiner + VAE 链路包装成
参数化界面。纯 Python stdlib HTTP server + 单文件 HTML/JS,零 pip、零下载、零 CDN。

    repo : <checkout>
    入口 : webui.py          (py312, HTTP + 调度)
    器件 : webui/bridge.py   (py310, import t13_run.py,不改驱动)

---

## 1. 启动

    $py312 = '<python312>\python.exe'
    cd <checkout>
    & $py312 webui.py --host 127.0.0.1 --port 7860

打开 http://127.0.0.1:7860 。首次请求校验/复制 128 构建缓存(<1 s),之后按任务
依次启动文本编码器、器件循环、VAE。文本编码器 GGUF(8.05 GB,第三方权重,不在
git 树)搜索顺序:环境变量 ZIMAGE_TE_GGUF;webui/models/;%TEMP%\zimage-te\;
%TEMP%\dsh-*\zimage-te\。找不到时 UI 禁用提交并显示提示,绝不静默退回 null caption。

停止:Ctrl+C。清理缓存:

    & $py312 webui.py --clean-cache          # 只删 webui/cache
    & $py312 webui.py --prepare-cache        # 只校验/准备 128 缓存

依赖:py312 需标准库 + numpy + diffusers/torch/PIL(本机已有);py310 需 numpy +
numba + pyxrt。PEANO_INSTALL_DIR 和 driver376 PATH 由 webui.py 注入器件子进程。

---

## 2. 参数表

| 参数 | UI | 默认 | 范围/说明 |
|---|---|---|---|
| 正向提示词 | textarea | a mountain lake at sunrise | llama.cpp Qwen3 编码;总 token <= 32(含模板) |
| 负向提示词 | textarea | 空 | 不参与计算:Turbo 无 CFG;只写元数据 |
| 高 H / 宽 W | number ×2 | 128 / 128 | 64..1024,16 的倍数;M=32+ceil((H/16)(W/16)/32)×32;预设 128/256/512/192×128/144×128/128×256/64×256/256×384;已构建:128×128、64×256、144×128、192×128、**128×256**(T20)、**256×256**(T19/T20);其余尺寸提交时返回 400 + 构建计划(不静默降级) |
| 批量 batch | number | 1 | 1..16(T16 放开原 4 的上限);批 N 每图 = seed+b;--batch N --share-weights |
| 扩散步数 steps | number | 8 | 1..16;!=8 显著提示分布外、质量不可保证 |
| seed | number | 0 | 0..2^31-1;batch 第 b 张 = seed+b |
| 调度器 | readonly | Euler | FlowMatchEulerDiscreteScheduler,固定只读 |
| VAE | checkbox | 开 | 关 = 只保存 *_latents.npy,不出 PNG |
| 输出目录 | text | webui/outputs | 必须位于项目树内 |
| 前缀 | text | zimage | [A-Za-z0-9._-],<=64 |
| 模式 | checkbox | 试跑 | 勾选 = seed 复现门(双跑 latents 逐字节) |
| 内存模式 | select | auto | auto / fast(cache-all)/ low(--no-cache-all) |
| BO 常驻 | select | 关 | 实验;60 个 BO;默认关 |

UI 徽标:门内可复现 = 同参数两次 latents 逐字节相同(且 pe 的 sha 相同);
试跑 = 单次未过门;复现门 FAIL = 两次不同,图保留但标红。

---

## 3. 架构 / 一次任务的数据流

    browser
      | POST /api/jobs
      v
    webui.py (py312, stdlib ThreadingHTTPServer)
      | 串行队列,一次一个 NPU 任务
      +- cache.ensure(128)            校验/复制 8 个构建产物(0.93 MB)
      +- textenc.run                    py312 te_worker.py
      |    +- imaggers/textenc/encode_api.py (只读 import)
      |         pe = hidden_states[-2] (1,L,2560)
      +- schedule.get_schedule(steps)   8 步用 t13b_traj 逐字节;其它步数现算
      +- webui/bridge.py (py310)
      |    +- import t13_run.py        (只读,不修改)
      |    +- patch t3_real.cap_embedding  (32x3840,见 §4)
      |    +- patch t3_real.load_schedule  (WebUI 的 timesteps/sigmas)
      |    +- patch t3_real.patchify      (T16 P0-1:patch-major,见 §1.3/§15)
      |    +- t13_run.T4_OUT = webui/cache/128  (只注册校验过的副本)
      |    +- t13_run.main()  -> 30 层 x steps,1440 dispatch/8 步
      +- vae_local.py (py312)          常驻 CPU VAE,首次加载后 0.35 s/批(T16)

不改 t13_run.py、不改 t3_real.py、不改三个交付 xclbin。新代码只在 webui/ 与
webui.py;运行时产物在 webui/cache|jobs|outputs。

【更正 2026-10-19】上图里的 `import t13_run.py` / `t13_run.main()` **已不是产品默认**。
`webui/bridge.py` 的驱动程序选择默认返回 **t18_run**(bridge.py:63-69, `name = force or "t18_run"`),
`--driver t13_run` 只留 A/B。依据 `webui/evidence/T18-P0-5-ARBITRARY-M.md` §5.1(:89-95):
交付 t13_run 的 dispatch 把输出 BO 又追加一次(6 个声明参数 → 7 个 BO),M=128 的 FFN
镜像在这第 7 个 BO 上直接 0xC0000005;当前环境里 t13_run 路径本身在首 dispatch 就硬崩
(fresh numba cache 也一样),而 t18_run 只传声明参数、同图同参数正常。切换后的字节门
全部与生产 latents 逐字节相同:128×128 `6004035d0cb97250`、64×256 `0df40ca3c18baa0a`、
192×128 `1e8a3ddc1b65de55`。t13_run 仍是只读基线实现,A/B 时显式 `--driver t13_run`。

---

## 4. 文本编码接入(替换 null caption)

已验证路径(见 imaggers/textenc/FINDINGS.md):

* llama.cpp Vulkan build llama.dll + GGUF Qwen3-4B;
* 取 layer-input tap lid=35 = HF hidden_states[-2],(1,L,2560) f32;
* chat 模板 = <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n;
* 已知 13-token 参考 prompt 的 pe 与缓存 ab1b0a2d17949d71.npz cosine 0.9999855,
  逐行 min 0.999846。

接入点 webui/cap.py 逐项复刻 zimage-port/generate.py 的 cap_prelude:

    pad 到 32 行(重复最后一行)
    -> RMSNorm(cap_embedder.0.weight, eps 1e-5)
    -> Linear 2560->3840 (cap_embedder.1)
    -> 第 L..31 行覆盖为 cap_pad_token

所以 L=13 的真实 caption 写入前 13 行,其余 19 行仍是模型自己的 pad token。
t_emb 仍由 t3_real.t_emb 计算(diffusers TimestepEmbedder 的 (1000-t)/1000 数学),
WebUI 不动它。

换 caption 换 latent 属预期。本交付的 PNG 与 T13 的 null-caption 基线
(PSNR 28.2052 / sha256 c7e0a890adf1c819)不可比;不要拿两者做逐字节或 PSNR 比较。

### GPU 显存不够时的 CPU 回退

TE 默认先走 Vulkan。若浏览器/其它进程占用 iGPU 导致
vk::Device::allocateMemory: ErrorOutOfDeviceMemory,te_worker 自动改用同一
llama.cpp 构建的 CPU 后端(只加载 ggml-cpu-*.dll,n_gpu_layers=0,GGUF 与
lid=35 tap 不变)。实测 CPU vs Vulkan 的 pe:cosine 1.000000,
max abs 0.37(|x|max 13,730),mean abs 0.0275。同一 server 记住该失败,后续
任务直接走 CPU(重启 server 才重试 GPU)。复现门在第一次编码后锁定后端,第二次
强制同一后端,避免把 GPU/CPU 的 pe 差异误报成 seed 不可复现。

---

## 5. 文本编码器 GGUF 重建(零下载,只用本机权重)

GGUF 8,051,284,800 B,由 zimage-port\_models\Z-Image-Turbo\text_encoder
(3 个 safetensors,7.5 GiB)转换。FINDINGS.md 的原命令可复现;要点:staging
放 config/tokenizer/index,model.safetensors 用 junction 指向 text_encoder
(不复制 7.5 GiB),converter 用 --outtype bf16。

webui/rebuild_te_gguf.ps1 封装了这些动作(默认输出
$env:TEMP\zimage-te\zimage-te-qwen3-4b-bf16.gguf):

    powershell -ExecutionPolicy Bypass -File webui\rebuild_te_gguf.ps1
    $env:ZIMAGE_TE_GGUF = "$env:TEMP\zimage-te\zimage-te-qwen3-4b-bf16.gguf"

输出约 8.05 GB,需要 C: 至少 9 GB 空闲。D: 只剩 0.8 GB,不要把 GGUF 放回项目树。

---

## 6. 分辨率 / 构建缓存

| 尺寸 H×W | latent | patch grid | image tokens | M | 状态 |
|---|---|---|---|---:|---|
| 64×64 | 8×8 | 4×4 | 16→32 | 64 | 未构建(下界) |
| 64×256 | 8×32 | 4×16 | 64 | **96** | **已构建(切片 1,host geometry)**;PNG 256×64;两次 latents 逐字节;host f32 PSNR 20.70 dB |
| 128×128 | 16×16 | 8×8 | 64 | **96** | **已构建**(交付镜像;T16 修 token 布局后肉眼可用) |
| **144×128** | 18×16 | 9×8 | 72(+24 pad) | **128** | **已构建(T18,padding 分支)**;run-to-run `d1a5f5fa`;vs host f32 corr 0.98241 / 25.00 dB |
| **192×128** | 24×16 | 12×8 | 96 | **128** | **已构建(T18)**;run-to-run 逐字节;vs host f32 corr 0.99422 / PSNR 29.86 dB;四件套过 |
| **128×256** | 16×32 | 8×16 | 128 | **160** | **已构建(T20)**;ip ND=160/Mq=32(5 块)+ perm 32×128;FFN 2×96 行块;run-to-run 逐字节 `7fa580b136b65830`;一步 vs host f32 corr **0.999975**;audit 3/3+13/17;**ready=true** |
| **256×256** | 32×32 | 16×16 | 256 | **288** | **已构建(T19),T20 修复 state-8**;FFN 3×96 行块(块间 1.7 ms relaunch barrier);run-to-run 逐字节 `683050a95f4bd357`;一步 vs host f32 corr **0.999964**;audit 13/17+3/3;**8 步 0 次 state-8、142.7 s**;ready=true |
| 256×384 | 32×48 | 16×24 | 384 | 416 | 未构建 |
| 512×512 | 64×64 | 32×32 | 1024 | 1056 | 未构建(原文档误写 1152,T16 修正) |
| 1024×1024 | 128×128 | 64×64 | 4096 | 4128 | 未构建(纸面上限) |

M = 32 + ceil(((H/16)×(W/16))/32)×32。新尺寸 = 每个受 M 约束的阶段 + perm +
host 常量 + 每尺寸四件套;详见 webui/evidence/T18-P0-5-ARBITRARY-M.md。
T18 实测:FFN/投影都按"行独立"分块,所以新 M **不需要改设计源码**——FFN 用交付
源码按 M 编译一次或直接 96 行块;投影用交付 t7_p5 的 96 行块;只有 ip 需要按
ND=SEQ 新编一个镜像。**T16 的"M=128 放置失败"是 kgu=64 假象**(见 T18 文档 §1)。
UI 不再锁死分辨率:输入任意合法 H/W 会显示 M、预算与构建计划;未构建尺寸提交
返回 400 + missing 列表,不静默放大/降级。

webui/cache.py 按尺寸缓存构建产物:每个 (H,W) 一个 webui/cache/{H}x{W}/,逐文件
sha256 写 manifest v2(shape/h/w/m);命中只校验 manifest。M=96 的尺寸
(128×128、64×256)复用同一批交付文件;64×256 只额外需要 bridge 的 host geometry
patch(IMG_GRID / unpatchify / 初始 latent 经 --resume 注入),不重建任何镜像。
新 M 的实际构成(T18):FFN 镜像(交付源码,--kgu 8)+ ip 镜像(ND=SEQ)+
perm(t18_perm.py 提取)+ host 驱动(t18_run.py)。192×128 的 4 个镜像 + 3 个 perm
共 984 KB;构建/审计/缓存由 webui/build_shape.py 串起来(plan → compile → audit →
cache → manifest 阶段流)。**T16 的"M=128 L1 墙"更正**:那次命令漏了 --kgu 8,
kgu=64 的 midB 对象正好 2,048 B;交付旋钮下 M=128/288 都编得过。证据:
webui/evidence/T18-P0-5-ARBITRARY-M.md §1。

**T19 起 256×256 已可用**(ready=true,镜像 = 交付 t7_p5 + T18 的 ip ND=288 +
交付 t11_nh5 行块;FFN 的 M=288 单 dispatch 在本设备上超时,见 T19 文档 §2/§4)。
**T20 把它的 state-8 定案并修掉**:同镜像背靠背的第二次 launch 不产生完成信号
(ERT 看门狗 7.0 s 报 state 8),主机间隔 0–2000 ms 无效;FFN 块之间插一次别的镜像的
launch(ip,1.7 ms)即可,见 `webui/evidence/T20-STATE8.md`。256² 8 步现在
**0 次重试、loop 89.4 s、total 142.7 s**(T19 为 3551/4651 s),latents 逐字节不变。
**T20 同时补建 128×256(M=160)**:ip ND=160/Mq=32 + perm 32×128 + FFN 2×96 行块,
四件套全过,见 `webui/evidence/T20-128x256.md`。
512/1024 的 UI 选项仍禁用并给出原因;没有半建——不生成半套镜像,也不把 128/256 的
图放大冒充。要开 512,仍需 ip 的键分块/在线 softmax 设计与 4 个投影镜像(T18 §11
的 L1 账),不是 --M 一个参数。

**T16 分辨率结论**:任务书原本把"host f32 也抽象"当成 128² 分布外的证据,从而要求
把 256/512 升为必选。T16 证明了相反的事:官方 ZImagePipeline 在 128² 同参数下出
清晰的湖景,host f32 与 WebUI 同时抽象是**共享的 patch token 布局 bug**(§15)。
修掉后 128² 肉眼可用,所以 256/512 不因图像质量而必选,T15 的镜像重建没有被半建。
硬件预算口径(256×256):**T20 修复后 8 步 loop 89.4–90.8 s、total 142.7 s**
(设备 52.6 s = ffn 29.0 + att 15.9 + ip 7.7;barrier 1.25 s;staging 7.0 s;
其余是 host 侧打包与逐层权重读)。T19 时点同一形状是 3501–4599 s,其中 96–97 %
是 FFN 块 dispatch 的 ERT state-8 重试(7 s 等待 + 10 s 退避,200/264 次)——
那部分现在归零,根因与修复见 `webui/evidence/T20-STATE8.md`。
权重按图重读、不随 M 摊薄,所以 M 越大每 token 成本越低;启动/TE/VAE 不变。

---

## 7. seed 复现门

勾选“运行 seed 复现门”后:

1. 第一次完整执行(含 TE、器件、VAE);
2. 第二次用同 prompt/steps/seed/batch、新 tag、同一 TE 后端、同一 schedule
   再跑一遍(不重复 VAE);
3. 比较两个 *_latents.npy 的 sha256 与字节;同时记录两次 pe 的 sha256;
4. latents 相同 -> PASS / 门内可复现;不同 -> FAIL,图保留并标红。

门是同机、同 session、同 TE 后端的逐字节门;跨机器/跨后端不承诺。门耗时约
2 倍。试跑图不会被自动标成“门内可复现”;UI 与元数据分开标记。

---

## 8. 内存模式 / 实测计时

完整内存路径需要约 6 GB private commit(W 像 1.85 GB + cache-all 3.71 GB)。
WebUI 按 GlobalMemoryStatusEx 的 commit 余量自动选择:

* fast(commit >= 9.0 GB):--cache-all,目标 24.5-25.1 s loop(快状态);
* low(commit < 9.0 GB):--no-cache-all --prefetch,更稳,loop 更慢;
* fast 仍 MemoryError -> 自动在新进程用 low 重跑,UI 记录 fallback 原因。

### T16 计时闭合(2026-10-18,修复后)

job 20261018-170459-ed2a32(8 步、batch 1、seed 0、128²,prompt a mountain lake
at sunrise,memory auto→fast,TE GPU,**常驻 VAE 第一次加载**):

| 阶段 | 秒 | 说明 |
|---|---:|---|
| 排队 queue | 0.005 | |
| 构建缓存 prepare | 0.012 | manifest 校验 |
| 文本编码 textenc | 14.895 | load 11.10 + decode 1.99 |
| 调度 schedule | 0.004 | 8 步 t13b 轨迹 |
| 冷启动重试 retry | 0.000 | 本次 0 次 |
| 启动/注册/载荷 startup | 37.682 | warm + cache-all + refinement |
| 扩散 loop | 27.728 | 1440 dispatch,0 retry |
| 器件尾 device_tail | 0.000 | |
| VAE 加载 | 15.284 | 常驻,仅本 server 第一次 |
| VAE 解码 | 0.455 | |
| 保存 PNG | 0.018 | |
| 任务收尾 post | 0.003 | |
| **Σ桶** | **96.086** | |
| **总计** | **96.857** | 排队起算 |
| **未归账** | **0.771** | **≤ 1 s,PASS** |

第二个 job 起 VAE 加载为 0(常驻解码器),同参数 job 的 VAE 段只剩
decode 0.34–0.46 s/图 + save 0.02 s;实测同进程第 2 次调用 total 0.350 s。

证据:webui/jobs/20261018-170459-ed2a32/job.json 的 timing.closure;
webui/evidence/T16-REPORT.md §4。

**T20 修了一个计时口径缺陷**:runner 的"扩散"桶原来取 driver 各步墙钟之和,而第 0 步的
计时起点在 `startup_done` 之前(cap 构建发生在第 0 步内部),这段被 startup 与 loop
**重复计入**(实测 128² 2.24 s、256² 1.74 s)。T19 的 128² job 以 −0.65 s 擦过门限,
256² 的第一个 job 直接 −1.05 s FAIL。现在扩散桶是**区间**(startup_done → 最后一行 step),
driver 自己的步和与重复量另记为 `loop_steps_sum_s` / `loop_overlap_s`,门限仍是 ±1 s。
复测 256² **+0.08 s**、128² **+0.08 s**,均 PASS(job 20261019-023928-614999 /
20261019-024206-00bac8)。

### (历史)本次交付 session 的实测(1 步 smoke,job 20261018-105612-416632):

    prompt   : a mountain lake at sunrise
    steps    : 1  (smoke;默认 8)
    seed     : 123
    memory   : low (--no-cache-all)
    TE       : GPU, 9.86 s (load 6.70 + decode 1.67)
    startup  : 30.38 s (注册/warm/cache)
    loop     : 7.83 s (device 2.31 + staging 0.25;180 dispatch,0 retry)
    VAE      : load 17.28 + decode 0.43 = 17.73 s
    total    : 72.74 s
    PNG      : 128x128,34,003 B
    sha256   : ffbabab73cbb9731f94d61ed886c2367d3b57fae3b83d3d56b026b2310d15946

证据:webui/evidence/E2E-prompt-1step.png 与 E2E-prompt-1step.log.txt。该图是
1 步分布外 smoke,只证明 prompt->latent->PNG 全链路可用,不是 8 步质量图。

### 重启后最终验收(2026-10-18)

8 步正式图(job 20261018-121643-c13f4e,默认参数,prompt a mountain lake at
sunrise,seed 0,batch 1,128x128,memory fast):

    TE       : 10.86 s (GPU load 7.80 + decode 1.57)
    startup  : 37.53 s
    loop     : 27.70 s (device 14.98 + staging 1.98;1440 dispatch,0 retry)
    VAE      : load 20.86 + decode 0.43 = 21.31 s
    total    : 199.03 s (含前两次 cold-start 重试 49.72 + 47.12 s;成功进程 64.45 s)
    PNG      : 128x128,39,282 B,sha256 90f0e2d628f5064e...
    calls    : steps_rec=8,dispatches=1440,retries=0

证据:webui/evidence/E2E-prompt-8step.png 与 E2E-prompt-8step.log.txt。该图与
T13 null-caption 基线不可比(换 caption 属预期);latents corr 0.907、PNG PSNR
14.39 dB vs 基线,只说明同链路不同 caption。

seed 复现性:两次独立完整 8 步运行(上述 job 的成功 pass 与 gate job 的 pass0)
latents 文件逐字节相同,sha256 a25920e724ab983c...;两次 PE npz 文件 sha256
437634b665...、pe 数组 sha256 90514d08... 均相同 -> 复现性 PASS。自动 gate job
20261018-122043-b83527 的第一次 pass 成功,第二次 pass 连续 4 个新进程都在第一个
dispatch 0xC0000005 失败,所以该 job 的 gate 字段是 FAIL(基础设施失败),不是
latents 不一致;两次完整运行已给出 PASS 证据。

### 安全模式与根因隔离(2026-10-18 重启后)

重启后按单变量矩阵重测(每格 2 个冷启动进程起步;1 步,成功后
latents 均为 sha16 065b52398a8d8e35,算术不变):

| 配置 | runs | 成功 | 首 dispatch 崩 |
|---|---:|---:|---:|
| safe low(no fast-codec/no mmap) | 4 | 2 | 2 |
| fast-codec only, low | 4 | 0 | 4 |
| refiner mmap only, low | 4 | 2 | 2 |
| fast-codec + mmap, low | 4 | 2 | 2 |
| fast-codec + cache-all | 2 | 1 | 1 |
| 60 BO + wpack + mmap | 2 | 1 | 1 |
| 60 BO + wpack(no mmap) | 2 | 1 | 1 |
| cache-all + mmap(no fast-codec) | 2 | 0 | 2 |
| cache-all safe(no fast-codec/no mmap) | 2 | 0 | 2 |

结论:**没有单一确定性根因**。首 dispatch 0xC0000005 在所有配置都出现,
是进程级冷启动竞态;fast-codec 在 low 臂最差(0/4),但 cache-all 臂能成功(1/2);
mmap 与 safe 同量级;60 BO + wpack + mmap 也能完成(1/2),没有独立内存阈值签名。
官方 8 步 safe/cache-all 重试 2 次成功,gate 第二次 pass 重试 3 次失败,
进一步说明是环境竞态而非算术问题。

因此**维持安全模式默认**:fast-codec OFF、refiner mmap OFF、fast-host ON、
wpack-array ON、memory auto;两条 fast 路径留作 opt-in。安全模式 8 步 loop
27.70 s,**24.5 s fast 路径未恢复**。完整矩阵见
webui/evidence/CONFIG-MATRIX.md。
* 首 dispatch 冷启动崩溃仍会偶发:官方 8 步 job 重试 2 次后成功;gate 第二次
  pass 重试 3 次仍失败。UI 自动换新进程重试;写盘后 bridge 用 os._exit 跳过
  teardown,成功进程退出码 0。

---

## 9. 限制清单(逐条)

1. 有器件构建的尺寸:128×128(M=96)、64×256(M=96,host geometry)、
   **192×128(M=128)**、**144×128(M=128,72+24 padding)**、**128×256(M=160,T20)**、
   **256×256(M=288,T19/T20)**。256 的 FFN 走 3×96 行块(单 dispatch M=288 超时),
   块间插一条 relaunch barrier 后 **0 次 state-8**(T20);两次独立跑逐字节相同,
   **ready=true**。其余尺寸(256×384、512×512、1024×1024)对任意合法 H/W 给出 M、
   预算与构建计划;不静默降级/放大,也不做半套镜像。512 的 M 已从错误的 1152
   修正为 1056;T18 更正见 webui/evidence/T18-P0-5-ARBITRARY-M.md。
   **2026-10-20 更正**:512 的 L1 预算文案原写 83,456 B,那其实是 256×384(SEQ=416)
   的值;M=1056 在 Mq=16 下是 196,096 B。已按新值更正 config 文案与前端兜底句,
   结论不变(仍 > 64 KB、不可用);出处 webui/evidence/T22-PRODLARGE.md §7.1。
2. cap_tokens = 32(含 chat 模板 9 个 token,正文约 23 token)。超长 prompt 在
   编码后按 token 数拒绝;不截断、不静默换 null caption。
3. 无 CFG,负向提示词不参与计算;仅记录。guidance 固定 turbo 行为。
4. steps != 8 是分布外。用同一 scheduler 现算 schedule,允许试,质量不保证;
   UI 红字警告,元数据 out_of_distribution=true。
5. 调度器固定 Euler;UI 只读。没有 DPM++/UniPC。
6. TE 后端回退:GPU 显存被占时自动 CPU;两者 pe 有约 0.37 绝对差(cosine 1.0)。
   门会锁定后端,但不要跨后端比较 PNG/latents。
7. 门只比较 latents 字节,不比较 PNG(VAE 可能因 torch 版本变化)。
8. 队列不持久:server 重启时 queued/running 标为 error,不自动续跑。
9. BO 常驻是实验:60 个 BO,首 dispatch 崩溃概率更高;默认关。
10. 输出目录必须在项目树内(默认 webui/outputs);不允许写系统目录。
11. 单租户:器件一次一个任务;WebUI 跑时不要另开 t13/image_run。
12. 首次 8 步启动约 30-50 s:warm 30 层投影 + refiner cache + cache-all/BO;
    这段不计入 loop,UI 显示“启动/载荷”。
13. VAE 默认常驻(py312 server 进程内,首次加载一次;之后 decode 0.34-0.46 s/图
   + save 0.02 s)。失败自动回退到一次性 vae_worker.py。**这是 CPU 解码器**;
   NPU VAE 只有 kernel 级物料,整链未建(见 T16-REPORT.md §2)。
14. 0xC0000005 首 dispatch 冷启动崩溃仍会偶发:bridge 自动换新进程重试 3 次;
    写盘完成后 os._exit 避免 teardown 崩溃。官方 8 步 job 重试 2 次成功;gate
    第二次 pass 重试 3 次未成功。若重试耗尽,按设备状态重跑;复现性可由两次
    独立完整运行对比证明。
15. 磁盘:PNG 34-39 KB/张;cache 0.93 MB;refiner cache 2.67 GB 只读、不复制;
    GGUF 8.05 GB 在 C: temp,不在 git。--clean-cache 只删 webui/cache。
16. fast-codec / refiner mmap 默认关闭(2026-10-18 重启后隔离出的首 dispatch
    崩溃源);仅作为 opt-in,重新验证前不要打开。

---

## 10. 磁盘 / 缓存清理 / git

    webui/cache/       128 的 8 个构建产物 + manifest + 现算 schedule(可删,自动重建)
    webui/jobs/        每个 job 的 job.json / 日志 / pe(可删,历史丢失)
    webui/outputs/     PNG / latents(交付物,按需保留)
    webui/evidence/    本次 E2E 证据(提交)

    & $py312 webui.py --clean-cache
    Remove-Item webui\jobs -Recurse -Force        # 可选
    Remove-Item webui\outputs -Recurse -Force     # 可选

git:新代码仅在 webui/、webui.py、README-WEBUI.md、.gitignore。
webui/cache|jobs|outputs|models|_selftest 已 ignore;不提交 xclbin/bin/GGUF/npy;
evidence 里的 34 KB smoke PNG 是交付证据。imaggers/block/t4_compile.py 的既有
未提交改动不是本任务产生,未纳入提交。

---

## 11. 故障排查

| 现象 | 处理 |
|---|---|
| UI 顶部 TE missing | 设 ZIMAGE_TE_GGUF 或跑 webui/rebuild_te_gguf.ps1 |
| POST 400 resolution 256 ... | 正常:本版本只交付 128;见 §6 【更正 2026-10-19:此句**作废**——现交付 6 个 ready 尺寸 128²、64×256、144×128、192×128、128×256、256×256(见 :39 的已构建列表与 §6 形状表 :153-154 两行;行号随本文改版会漂,以 `ready=true` 为准);仍返回 400 的只有未构建尺寸 512×512、384×256、256×384、1024×1024 等,应答附 M 与构建计划。】 |
| 400 prompt is N tokens > 32 | 缩短 prompt;固定 M=96 的 caption 位只有 32 |
| 任务 error,raw 里 MemoryError | 切 memory_mode=low;关闭 Chrome/Edge 释放 commit |
| 首 dispatch 0xC0000005 | UI 自动重试 3 次;仍失败则重启机器(上下文卡死) |
| 卡在“文本编码” | GPU 被占,te_worker 正在回退 CPU;看 job log 的 [te:cpu] |
| VAE 很慢/加载 17 s | 正常,每 job 新进程;批量只摊薄 decode |
| 队列不动 | /api/status 看 runner.current;一次只跑一个 |
| 端口占用 | --port 7861 |

---

## 12. HTTP API

    GET  /api/status                 版本/内存/缓存/TE/队列
    GET  /api/jobs                   最近 40 个 job
    POST /api/jobs                   提交,返回 202 + job
    GET  /api/jobs/<id>              状态/阶段/计时/图/门/日志尾
    GET  /api/jobs/<id>/log          完整日志 text/plain
    GET  /api/jobs/<id>/images/<idx> PNG
    POST /api/jobs/<id>/cancel       取消
    POST /api/jobs/<id>/retry        用同参数新建 job
    GET  /api/cache                  按尺寸的缓存状态
    GET  /api/shapes                 预设 + M + ready + 构建计划 + 限制
    POST /api/shapes/prebuild        常用尺寸一键预建(已构建=暖缓存;未构建=返回计划)
    POST /api/cache/prepare          校验/准备 128 缓存
    POST /api/cache/clean            删除 webui/cache

提交示例(PowerShell):

    $body = @{
      prompt='a mountain lake at sunrise'; negative_prompt=''
      height=128; width=128; batch=1; steps=8; seed=0; vae=$true
      output_dir='webui/outputs'; prefix='zimage'
      mode='trial'          # trial | gate
      memory_mode='auto'    # auto | fast | low
      bo_resident=$false
    } | ConvertTo-Json
    Invoke-RestMethod -Method Post -Uri http://127.0.0.1:7860/api/jobs -ContentType 'application/json' -Body $body

---

## 13. 复现门命令(绕过 UI)

    # 1) 文本编码(自动 GPU->CPU)
    & $py312 webui\te_worker.py --gguf $env:ZIMAGE_TE_GGUF --prompt-file webui\_selftest\prompt.txt --out pe.npz --backend cpu
    # 2) 器件(py310);schedule.npz 可由 webui.schedule.get_schedule(8) 生成
    $env:PEANO_INSTALL_DIR='...\llvm-aie\llvm-aie'
    $env:PATH='...\driver376\npu_mcdm_stack_prod;'+$env:PATH
    & $py310 webui\bridge.py --tag my8 --pe pe.npz --schedule schedule.npz --cache-dir webui\cache\128 --steps 8 --seed 0 --batch 1 --no-cache-all
    # 3) VAE
    & $py312 webui\vae_worker.py --latents imaggers\image\my8_latents.npy --out-dir webui\outputs --prefix my8 --seeds 0 --result vae.json

正常判据:imaggers/image/my8_calls.json 的 steps_rec 长度 = 8、dispatches = 1440、
retries = 0;*_latents.npy 存在。进程若在写完后以 3221225477 退出,bridge 已用
os._exit 避免。

---

## 14. 本次交付的验证边界

* WebUI 启动:OK,py312 webui.py --port 7860,API 正常;TE 自动命中
  webui/models/ 硬链接。
* 真实端到端:OK,8 步 128x128,prompt->latents->PNG->计时(§8);
  PNG 39,282 B sha256 90f0e2d6...
* 8 步 calls.json:steps_rec=8、dispatches=1440、retries=0。
* seed 复现性:两次独立完整 8 步运行 latents 逐字节相同、PE sha 相同 -> PASS;
  自动 gate job 第二次 pass 因首 dispatch 崩溃未完成,gate job 字段 FAIL(基础设施),
  不代表复现性失败。
* 交付镜像/参考工件:t3d_baseline.py --check 前后均为 ALL REFERENCE BYTES
  UNCHANGED(收尾已跑)。
* batch 4:上述"post-reboot 批 4 未成功"的记录已被后续运行推翻:job
  20261018-163636-f54297(4×128²,8 步)在两轮首 dispatch 冷启动重试后成功出 4 图,
  loop 99.80 s = 4×24.95 s/图,总计 262.96 s;263 s 的完整归账见
  webui/evidence/T16-REPORT.md §3.2。
* T16 新增:128² 图像肉眼可用(并排对照 §15);计时全闭合(§8);batch 上限 16;
  常驻 VAE;批语义门与 N>4 证据见 webui/evidence/T16-REPORT.md。

---

## 15. T16 产品返工(P0 四项)摘要

完整报告:webui/evidence/T16-REPORT.md;并排图:
webui/evidence/T16-P0-1-side-by-side.png;自查清单:
webui/evidence/PRODUCT-CHECKLIST.md。

**P0-1 图像混乱 = token 布局 bug,不是 128² 分布外。** 官方 ZImagePipeline 在
128²/同 prompt/seed/8 步下是清晰湖景(zimage-port/_out/zimage_128_fixed.png),
官方最终 latent 用交付 CPU VAE 解码逐字节相同 -> VAE 没错。根因是
t3_real.patchify 把 2×2 patch 按 channel-major 摊平,而 checkpoint 的
all_x_embedder 按官方 patch-major 训练;unpatchify 本来就是 patch-major。单变量
A/B:corr 0.432 -> 0.9986,l2rel 109.4% -> 5.27%。修复在
webui/bridge.py::_install_token_layout_fix();修复后 8 步器件图 22,926 B、湖景,
latent rms 1.434 -> 0.939(官方 0.821)。

**P0-2 VAE。** 工作区不存在 t3_real_vae.onnx;"NPU VAE"只有 kernel 级物料
(35 conv GEMM、GroupNorm 1.0–1.5 ulp、upsample 逐位、整链投影 37.32 dB),
整解码器未串上器件,也没有 1.8 s/图计量 -> 验收 ③ 未达成,已在报告里写明剩余工作。
已交付常驻 CPU VAE:加载从每 job 15–20 s 降为每 server 一次,之后 0.35 s/批。

**P0-3 批量。** MAX_BATCH 4 -> 16;4×128² 的 263 s 归账:TE 12.30 +
startup 33.31 + loop 99.80 + VAE 21.81 + 冷启动重试 91.45 + 编排 ~4.3 = 262.96。
**N>4 的 post-fix 逐字节门 PASS**:batch 5 五图、batch 8 八图全部与各自单图
latents 逐字节相同(gate.pass=true);每图 loop 26.4(N=1)→23.3(N=5)→22.8(N=8) s,
wall/图 50.5→28.7→26.9 s。设备中途卡死一次(0xC0000005),恢复后补跑通过;
细节与曲线见 T16-REPORT.md §3.3。

**P0-4 计时。** 13 个桶 + closure(Σ桶 vs 总计,±1 s);UI 显示每一项与未归账,
重试单独列出;本轮 job 未归账 0.77 s(新口径复跑 -0.28 s)。

**P0-5 任意宽高。** UI 改为 H/W 两个输入 + 预设;M = 32 + ceil(((H/16)(W/16))/32)*32;
按尺寸缓存 webui/cache/{H}x{W}/ + sha256 manifest + LRU 10 / 1 GiB;未构建尺寸
返回 400 + 精确构建计划(镜像 + perm + host 常量 + 四件套),不假装无限制。
**T18:192×128(M=128)已过四件套**(run-to-run 逐字节 + vs host f32 corr 0.99422 /
29.86 dB + audit + 并排图)。**T19:256×256(M=288)收口**——run-to-run 逐字节
`683050a95f4bd357`、一步数值门 vs host f32 **0.999964**、audit 13/17+3/3、并排图、
计时闭合 +0.05 s,ready=true;八步终图 17.19 dB 的量化解释与混沌对照见
webui/evidence/T19-256x256.md §3。构建器 webui/build_shape.py 把
plan→compile→audit→cache→manifest 串成阶段流。
**T20:256×256 的 state-8 定案**——同镜像背靠背第二次 launch 不产生完成信号(ERT
7.0 s 看门狗),FFN 块间插一次别的镜像的 launch(1.7 ms)即消除;8 步 0 次重试、
total 142.7 s、latents 仍 `683050a95f4bd357`(webui/evidence/T20-STATE8.md)。
**T20:128×256(M=160)补建**——ip ND=160/Mq=32 + perm 32×128 + FFN 2×96 行块,
run-to-run `7fa580b136b65830`、一步 corr **0.999975**、audit 3/3+13/17、并排图、
ready=true(webui/evidence/T20-128x256.md)。

**【口径注 2026-10-20】512²+ 的数值门换了参考**:512² 及更大尺寸改判**对 BFP16
精确仿真参考**(逐位/corr),**对 host f32 的 corr 降为信息项**;**128–256² 口径不变**。
依据:投影/attention 的 BFP16 地板实证(`8e5fdc7`、`fbe6b24`)——512² 的 f32 corr 已经
贴到量化地板,继续拿 f32 当阈值等于把地板当缺陷;负责人 2026-10-20 裁决。纪律承诺:
新参考自身 100% 门、独立门(逐字节/audit/计时闭合/批语义)保持 100%、报告与 UI 全程
声明口径。卡片:`docs/KNOWN-PLATFORM-FACTS.md` F-23;清单:
`webui/evidence/PRODUCT-CHECKLIST.md` §5。
