# vLLM Ascend profiling 操作手册（面向 cp_balance 开/关 A/B 对比）

本手册基于本机三份只读代码树（`D:\code\cp_balance\vllm-ascend`、`vllm-ascend-base`、`vllm`）+
`torch-npu==2.10.0.post2` 官方 wheel 源码（见第 8 节）静态阅读整理。所有结论都给出
`文件:行号` 或文档出处；没有实测过的部分明确标注为"未验证/需现场确认"。

---

## 0. 先看结论（关键事实纠正）

1. **环境变量方式已经不存在了**。`VLLM_TORCH_PROFILER_DIR` 等一组变量在 vLLM 主线被
   重构进 CLI 配置（PR #29912，2025-12-09 提交 `e858bfe051`），随后被删除
   （PR #33536/`ef248ff740` 2026-02-03、PR #33722/`d88a1df699` 2026-02-04）。
   本机 `vllm` 树里 grep `VLLM_TORCH_PROFILER` 已无命中（只有 `vllm/envs.py:290` 的
   内存 profiler 开关）。vllm-ascend 明确公告过这一点：
   `vllm-ascend/docs/source/user_guide/release_notes.md:1089`、
   `vllm-ascend/.agents/skills/vllm-ascend-release/references/ref-past-release-notes-highlight.md:44`。

2. **当前唯一可用的开关是 `--profiler-config`（在线）/ `profiler_config=`（离线）**
   （`vllm/engine/arg_utils.py:1571-1572`、`vllm/entrypoints/llm.py:211,276`）。
   历史变量名（现在无效，仅用于识别旧文档/旧 yaml）：`VLLM_TORCH_PROFILER_DIR`、
   `VLLM_TORCH_PROFILER_RECORD_SHAPES`、`VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY`、
   `VLLM_TORCH_PROFILER_WITH_STACK`、`VLLM_TORCH_PROFILER_WITH_FLOPS`、
   `VLLM_TORCH_PROFILER_USE_GZIP`、`VLLM_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL`、
   `VLLM_TORCH_PROFILER_DISABLE_ASYNC_LLM`（`git show e858bfe051^:vllm/envs.py` 第 93-105 行）。
   注意：`vllm-ascend/tests/e2e/nightly/single_node/models/configs/MiniMax-M2.5-w8a8-QuaRot-A2.yaml:18`
   仍在设置 `VLLM_TORCH_PROFILER_DIR`，这是**失效残留**。

3. **Ascend 上只有 `profiler: "torch"` 可用**，`"cuda"` 会在 worker 里抛
   `RuntimeError: Unrecognized profiler: cuda`
   （`vllm_ascend/profiler/torch_npu_profiler.py:39-40`）。

4. **导出类型只有两种：`text` 和 `db`**（`torch_npu/profiler/experimental_config.py:31-32,54-56`，
   `torch_npu/profiler/analysis/prof_common_func/_constant.py:132-133`）。不存在
   "text/csv/json 三种 export_type"；CSV 和 `trace_view.json` 都是 `text` 型导出后
   由解析器产出的文件（第 3、4 节）。

5. **采集是"每 rank 一个目录、不合并"**：每个 worker 在自己进程里写
   `<torch_profiler_dir>/<trace_name>_<pid>_<时间戳>_ascend_pt/`
   （`torch_npu/profiler/_profiler_path_creator.py:54-78`）。

6. **在 `--distributed-executor-backend mp` 下，stop 时的自动解析会失败**，因为 vLLM 的
   worker 进程是 daemon 进程（`"mp"` → `MultiprocExecutor`：
   `vllm/v1/executor/abstract.py:69-72`；worker 进程 `daemon=True`：
   `vllm/v1/executor/multiproc_executor.py:694-698`），而
   torch_npu 的解析器拒绝在 daemon 进程里跑
   （`torch_npu/profiler/analysis/_npu_profiler.py:22-28`）。因此**必须手工补一次离线解析**
   （第 4.1 节）。这一点是本手册最重要的操作结论。

---

## 1. 两条 profiling 路线与命名陷阱

### 1.1 Ascend PyTorch Profiler（= `torch_npu.profiler`，本手册主线）

- 采集粒度：PyTorch 算子/设备 kernel 级；产物是 `*_ascend_pt` 目录。
- 控制方式：API（`/start_profile`、`/stop_profile`）+ 配置。
- vllm-ascend 侧的接线：`vllm_ascend/profiler/torch_npu_profiler.py:30-79`
  （`TorchNPUProfilerWrapper` 继承上游 `WorkerProfiler`，内部用 `torch_npu.profiler.profile`）。

### 1.2 MS Service Profiler（框架函数级，另一条路）

- 由 `ms_service_profiler` 工具打点，靠 YAML 符号表（本仓库的
  `vllm_ascend/profiling_config.py` 就是自动生成这张符号表的，见 1.3）。
- 采集粒度：服务框架函数 + 可选算子级（`acl_task_time=3` 时走 torch profiler dump）。
- 出处：`vllm-ascend/docs/source/developer_guide/performance_and_debug/service_profiling_guide.md`
  （下称 "service_profiling_guide"），摘录见第 7 节。

### 1.3 三个容易混淆的名字

| 名字 | 实际含义 | 出处 |
|---|---|---|
| `profiling_chunk_config` | **不是采集**，是 PP 场景下"用启动期测量拟合动态 chunk 大小"的调度器功能 | `vllm_ascend/ascend_config.py:606-645`；也正是 `tests/e2e/pull_request/four_card/test_profiling_chunk_performance.py` 测的东西 |
| `profile_run()` / `determine_available_memory()` | **不是采集**，是启动期跑 dummy batch 量可用显存 | `vllm_ascend/worker/model_runner_v1.py:3418-3425` |
| `MSMONITOR_USE_DAEMON` / `torch_npu.profiler.dynamic_profile` | 另一套"daemon 模式"监控（`dp.step()`），与 torch profiler **互斥** | `vllm_ascend/worker/worker.py:595-599`、`vllm_ascend/envs.py:74` |

`vllm_ascend/profiling_config.py` 在 `vllm_ascend` import 时把符号表写到
`~/.config/vllm_ascend/service_profiling_symbols.<vllm版本>.yaml`
（`vllm_ascend/__init__.py:67-69`、`vllm_ascend/profiling_config.py:32-33,518-528,542`）。
里面已经包含 cp_balance 相关的点，例如
`vllm_ascend.patch.platform.patch_balance_schedule:BalanceScheduler.schedule`
（`vllm_ascend/profiling_config.py:197-203`）和
`NPUModelRunner._model_forward`（domain=ModelForward，带 dp_rank/npu_id 属性，
`vllm_ascend/profiling_config.py:275-279`）——如果只想看"这一段 forward 在 host 侧的
时间跨度"，MS Service Profiler 更省事；但看通信/算子归因还是走第 1 节主线。

---

## 2. 打开 profiling 的所有方式

### 2.1 环境变量方式：不可用

结论见第 0 节第 1、2 条。旧文档里"export VLLM_TORCH_PROFILER_DIR=./profile 就自动开"
的写法对当前代码树无效；`/start_profile` 之所以 404，最常见原因就是没给
`--profiler-config`（见 2.3）。

### 2.2 配置/CLI 方式（推荐）

两种等价写法，都来自 `vllm/engine/arg_utils.py:1571-1572` +
`vllm/utils/argparse_utils.py:389-426`（`.field value` 会被折成 JSON）：

```bash
# 写法 1：整段 JSON（文档和教程都用这种）
vllm serve <model> --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/abs/path/prof"}'

# 写法 2：点号逐个字段（值会被 json.loads，false/1 等直接写）
vllm serve <model> \
  --profiler-config.profiler torch \
  --profiler-config.torch_profiler_dir /abs/path/prof \
  --profiler-config.torch_profiler_with_stack false \
  --profiler-config.max_iterations 1
```

校验规则（`vllm/config/profiler.py:137-162`）：
- `profiler="torch"` 时 `torch_profiler_dir` 必填，否则启动直接报
  `torch_profiler_dir must be set when profiler is 'torch'`；
- 只要给了 `torch_profiler_dir`，`profiler` 必须是 `"torch"`；
- 相对路径会被 `os.path.abspath(os.path.expanduser(...))` 展开成**相对于服务进程 CWD**
  的绝对路径（`:154-156`）→ 现场一律写绝对路径。

`ProfilerConfig` 全字段与在 Ascend 上是否生效：

| 字段（`vllm/config/profiler.py` 行号） | 默认 | Ascend 是否生效 | 说明 |
|---|---|---|---|
| `profiler` (:37) | None | 是 | 只能填 `torch` |
| `torch_profiler_dir` (:43) | "" | 是 | 所有 rank 共用的根目录 |
| `torch_profiler_with_stack` (:48) | True | 是 | Ascend 上被映射成 `torch_npu.profiler.with_modules`（`torch_npu_profiler.py:68-70`），开启会显著增大数据量 |
| `torch_profiler_with_memory` (:65) | False | 是 | 映射成 `profile_memory`（`torch_npu_profiler.py:67`） |
| `delay_iterations` (:87) | 0 | 是 | 由基类 `WorkerProfiler.step()` 实现（`vllm/profiler/wrapper.py:83-114`） |
| `max_iterations` (:92) | 0 | 是 | 同上，到点上界后自动 stop 并落盘 |
| `ignore_frontend` (:80) | False | 是（只影响 API server 侧） | True 时不启动 AsyncLLM 进程内的 CPU profiler（`vllm/v1/engine/async_llm.py:178-200`） |
| `torch_profiler_record_shapes` (:62) | False | **否** | NPU wrapper 没传（等价能力是硬件级的 `record_op_args`，被硬编码为 False） |
| `torch_profiler_with_flops` (:53) | False | **否** | 同上 |
| `torch_profiler_use_gzip` (:56) | True | **否** | `torch_npu.profiler.tensorboard_trace_handler` 没有 gzip 参数（`torch_npu/profiler/profiler.py:171-185`） |
| `torch_profiler_dump_cuda_time_total` (:59) | True | **否** | 只有 GPU 的 `TorchProfilerWrapper._stop()` 用它写 `profiler_out_*.txt`（`vllm/profiler/wrapper.py:266-286`） |
| `capture_torch_profiler` (:69) | False | **否** | 只在 GPU model runner 的图捕获里用（`vllm/v1/worker/gpu_model_runner.py:6769-6775`） |
| `detailed_trace_annotation` (:74) | False | **否** | vllm-ascend 的 wrapper 不实现 `annotate_context_manager`，基类返回 `nullcontext()`（`vllm/profiler/wrapper.py:146-148`） |
| `warmup_iterations` (:97) / `active_iterations` (:105) / `wait_iterations` (:111) | 0/5/0 | **否** | schedule 型参数，只在 GPU 的 `TorchProfilerWrapper.__init__` 里构造 `torch.profiler.schedule`；vllm-ascend 的 PR #8953 明确写了 "Schedule-only options (e.g. warmup/wait) remain out of scope"（`git log -1 ada08174f`），代码里 `_profiler_step()` 恒返回 True（`torch_npu_profiler.py:74-75`） |

**采集参数（export_type / profiler_level / aic_metrics / data_simplification / l2_cache /
op_attr 等）在 vllm-ascend 里是硬编码的**，只能改代码
（`vllm_ascend/profiler/torch_npu_profiler.py:48-61`）：

```python
experimental_config = torch_npu.profiler._ExperimentalConfig(
    export_type=torch_npu.profiler.ExportType.Text,     # 只导出 text 型
    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    msprof_tx=False, aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    l2_cache=False, op_attr=False, data_simplification=True,
    record_op_args=False, gc_detect_threshold=None)
```

`aic_metrics=PipeUtilization` 是 PR #10778 改的（此前是 `AiCoreNone`，导致 trace 里没有
计算/流水利用率数据，`git show 12c8da7a1` 的 commit message）。Level1 也是硬编码的，
它决定了后面能拿到哪些产物：Level1 才没有 trace 裁剪、才有 `op_statistic.csv`
（见 3.4）。

### 2.3 在线 API 方式

- 端点在 vLLM 侧定义：`POST /start_profile`、`POST /stop_profile`
  （`vllm/entrypoints/serve/profile/api_router.py:21-34`）。
- **没有 query 参数、没有 body 字段**，两个 handler 只做
  `await engine_client(raw_request).start_profile()`，成功返回
  `Response(status_code=200)`（空 body）。
- **只有带了 `--profiler-config` 且 `profiler` 非 None 才会注册这两个路由**
  （`vllm/entrypoints/serve/profile/api_router.py:37-45`），注册时还会打一条
  "Profiler with mode 'torch' is enabled in the API server. This should ONLY be used for
  local development!"。没配置就 curl → 404。
- **是否所有 rank 都会 start/stop：会。** 调用链是
  HTTP → `AsyncLLM.start_profile()`（`vllm/v1/engine/async_llm.py:905-910`）→
  `EngineCore.profile()`（`vllm/v1/engine/core.py:761-762`）→
  `Executor.profile()`（`vllm/v1/executor/abstract.py:256-257`）→
  `collective_rpc("profile", args=(is_start, profile_prefix))`
  （`vllm/v1/executor/multiproc_executor.py:343-377` 广播给全部 worker）→
  `NPUWorker.profile()`（`vllm_ascend/worker/worker.py:925-953`）。
- `start_profile` 还额外启动 API server 进程内的 AsyncLLM CPU profiler
  （`vllm/v1/engine/async_llm.py:178-200`），产物目录名
  `{hostname}_{pid}.async_llm`；`ignore_frontend=true` 可关掉（推荐关，见 5.1）。
- **HTTP 方式无法传 `profile_prefix`**：路由不接收参数（`:21-26`），
  于是 worker 端 `trace_name = get_worker_rank_suffix(global_rank)`，形如
  `dp0_pp0_tp0_dcp0_ep0_rank7`（`vllm-ascend/worker/worker.py:935-944`、
  `vllm/distributed/utils.py:696-745`）。需要自定义前缀只有离线 API：
  `llm.start_profile("warmup")` → trace_name 变成 `warmup_dp0_pp0_tp0_dcp0_ep0_rank0`
  （`vllm/entrypoints/llm.py:776-787`、`tests/ut/worker/a2/test_worker_v1.py:343-360`）。
- 失败模式：
  - 404 → 没配 `--profiler-config`；
  - 500 / RPC 报错 → `MSMONITOR_USE_DAEMON=1` 或 `additional_config.msmonitor_use_daemon=true`
    时构造 profiler 抛 `RuntimeError: MSMONITOR_USE_DAEMON and torch profiler cannot be
    both enabled at the same time.`（`torch_npu_profiler.py:43-47`）；
  - `/stop_profile` 可能阻塞很久：worker 在 stop 里做落盘（甚至解析），`curl` 要放宽超时
    （客户端 `--max-time`；`profiler` 相关超时改看 `harness.json limits`。
    历史上配置里放过 `VLLM_RPC_TIMEOUT`，该变量在 vllm/vllm-ascend 里都没有消费者，已从 `configs/_base.json` 删掉）。

### 2.4 代码/离线方式

```python
from vllm import LLM
llm = LLM(model="...", tensor_parallel_size=16,
          profiler_config={"profiler": "torch",
                           "torch_profiler_dir": "/abs/prof",
                           "torch_profiler_with_stack": False,
                           "max_iterations": 1})
llm.start_profile()          # 可选前缀：llm.start_profile("cp_on")
out = llm.generate(prompts, sampling_params)
llm.stop_profile()
```

- `LLM(profiler_config=...)`：`vllm/entrypoints/llm.py:211,276,325`；
- `llm.start_profile(prefix)` / `stop_profile()`：`vllm/entrypoints/llm.py:776-787`；
- 官方示例：`vllm/examples/features/profiling/simple_profiling_offline.py`（30-36 行采集）、
  `vllm/examples/features/profiling/run_one_batch_offline.py`（50-72 行用
  `delay_iterations/max_iterations` 区分 prefill/decode 窗口，正是我们要的用法）；
- `vllm bench serve --profile`：会在压测前后自动 curl `/start_profile`、`/stop_profile`
  （`vllm/benchmarks/serve.py:921-935,1343-1355`，文档 `vllm/docs/contributing/profiling.md:52-62`）。

**`additional_config` 里不能传 profiler 配置**：`vllm_ascend/ascend_config.py` 只解析
Ascend 自有键（`vllm_ascend/ascend_config.py:80-200`），没有任何读取
`profiler_config` 的代码路径（`grep -rn profiler_config vllm_ascend/` 只命中
`worker.py:140` 和 `profiler/torch_npu_profiler.py`）。所以在
`cp_balance/configs/*.json` 里要通过 `server_args` 追加 `--profiler-config`。

### 2.5 与 `--enforce-eager` / torch.compile（graph mode）的关系

- 现场 `configs/_common.json` 的 `server_args` 已含 `--enforce-eager`，
  即 ACLGraph/Npugraph_ex 图模式关闭，每个算子单独下发，trace 里能看到逐算子 kernel，
  这是做"每层 attention/MLP 归因"的前提（graph mode 概念见
  `vllm-ascend/docs/source/user_guide/feature_guide/graph_mode.md:17-51,124`）。
- 如果开了图模式（`cudagraph_mode != NONE`，或 `ascend_compilation_config.enable_npugraph_ex`），
  模型 forward 会变成图捕获/回放，profiler 里看到的是回放粒度而不是逐算子粒度，
  逐层对比会失真。**A/B 两次必须都用 `--enforce-eager`，不能只关一边**。
- `--enforce-eager` 与 profiling 本身不冲突；`enforce_eager` 只影响调度/执行形态
  （`vllm-ascend/docs/source/user_guide/feature_guide/graph_mode.md:276,292`）。
- `torch.compile`/静态 kernel 的验证方法官方就是用 profiler 产物
  `op_statistic.csv` 查 `static_kernel` 关键字
  （`vllm-ascend/docs/source/user_guide/feature_guide/graph_mode.md:214-224`）。

### 2.6 与 `MSMONITOR_USE_DAEMON` 的冲突

- 变量/配置：`MSMONITOR_USE_DAEMON=1`（`vllm_ascend/envs.py:74`）或
  `additional_config.msmonitor_use_daemon=true`
  （`vllm_ascend/ascend_config.py:179-184`、`docs/source/user_guide/configuration/additional_config.md:88`）；
  二者同时给出时 **additional_config 优先**（`torch_npu_profiler.py:43-45` 先读 env，
  再用 `get_ascend_config().msmonitor_use_daemon` 覆盖）。
- 冲突行为：只要为真，创建 profiler 就抛
  `MSMONITOR_USE_DAEMON and torch profiler cannot be both enabled at the same time.`
  （`torch_npu_profiler.py:46-47`），`/start_profile` 失败。
  单测覆盖了两种取值方向的组合：`tests/ut/profiler/test_torch_npu_profiler.py:144-223`。
- 采集前先确认未开：`unset MSMONITOR_USE_DAEMON`，且 `additional_config` 里没有
  `msmonitor_use_daemon`（现场 `_common.json` 没有，安全）。

---

## 3. 采集产物

### 3.1 目录结构

`torch_profiler_dir` 下，**每个 rank 一个目录**：

```
<torch_profiler_dir>/
  <trace_name>_<pid>_<YYYYMMDDHHMMSSmmm>_ascend_pt/        # 每 rank 每轮采集
      FRAMEWORK/                     # torch 侧原始数据
      PROF_<pid>_<时间戳>_<随机串>/   # CANN 侧原始数据
          host/                      # host_start.log, start_info, info.json, data/
          device_<id>/               # data/ , summary/ , timeline/ , sqlite/
          mindstudio_profiler_output/# msprof 导出的中间文件（op_summary_*.csv、msprof_*.json 等）
          analyze/                   # msprof --analyze 产物（communication.json 等）
      profiler_info.json             # 或 profiler_info_<rank>.json（有 rank 时）
      profiler_metadata.json
      ASCEND_PROFILER_OUTPUT/        # 解析后的最终产物（analyse 之后才出现）
          analyse.done               # 解析完成的标记文件
  {hostname}_{pid}.async_llm/        # 仅 ignore_frontend=false 时，API server 进程的 CPU trace
```

- 命名规则：`worker_name = "<worker_name或hostname>_<pid>"`，
  `span_name = "<worker_name>_<%Y%m%d%H%M%S%f 取到毫秒>_ascend_pt"`
  （`torch_npu/profiler/_profiler_path_creator.py:54-78`）。
- 解析输出固定在 `<prof_dir>/ASCEND_PROFILER_OUTPUT/`，且每次解析前会先删除旧目录
  （`torch_npu/profiler/analysis/_profiling_parser.py:27-30`）。
- 结束标记：`ASCEND_PROFILER_OUTPUT/analyse.done`
  （`torch_npu/profiler/analysis/_profiling_parser.py:108-109`）。
- `FRAMEWORK/`、`PROF_*/` 的识别规则见
  `torch_npu/profiler/analysis/prof_common_func/_path_manager.py:11-25,107-121`。
- 同一个进程反复 start/stop：worker 只创建一次 profiler 对象、trace_name 固定
  （`vllm_ascend/worker/worker.py:941-948`），但每次 `start()` 会新开一个带新时间戳的
  `*_ascend_pt` 目录（`torch_npu/profiler/_profiler_path_creator.py:54-78` 在
  `profile.start()` 路径上被调用）→ 多轮采集靠时间戳区分。

### 3.2 每 rank 一个还是合并？

每 rank 一个目录，**没有任何自动合并**。rank 从目录名里的 `rank<N>` 就能读出来
（`get_worker_rank_suffix` 会拼上 `dp/pp/tp/dcp/ep/rank`，`vllm/distributed/utils.py:718-737`）。
TP16 单机就是 16 个 `*_ascend_pt` 目录写在同一台机器同一个根目录下；
多机 DP 时该根目录必须是共享存储（教程示例用 `/mnt/share/...`，
`docs/source/tutorials/models/GLM5.2.md:947`），否则只有本机 rank 的文件。

### 3.3 export_type 到底是什么，产出什么

`ExportType` 只有两个值（`torch_npu/profiler/experimental_config.py:31-32,54-56`；
字符串值为 `"text"` / `"db"`，见
`torch_npu/profiler/analysis/prof_common_func/_constant.py:132-133`）：

- `ExportType.Text`（vllm-ascend 硬编码值）：用 `msprof --export=on` 导出文本型数据，
  再由解析器生成 CSV / `trace_view.json`（`torch_npu/profiler/analysis/prof_view/cann_parse/_cann_export.py:66-71`）。
- `ExportType.Db`：额外用 `msprof --export=on --type=db` 导出 db，并生成
  `ascend_pytorch_profiler_<rank>.db`
  （`torch_npu/profiler/analysis/prof_view/prof_db_parse/_db_parser.py:44`）；
  需要 CANN 版本支持（否则 `RuntimeError: ... does not support export db`，
  `torch_npu/profiler/analysis/_profiling_parser.py:72-76`）。

`ASCEND_PROFILER_OUTPUT/` 的文件清单（producer → 文件）：

| 文件 | 内容 | 生成代码 |
|---|---|---|
| `step_trace_time.csv` | 每 step 的 Computing / Communication(Not Overlapped) / Overlapped / Communication / Free / Stage / Bubble / Preparing | `analysis/prof_view/_trace_step_time_parser.py:44-48` |
| `kernel_details.csv` | 设备 kernel 明细（Level1 用 msprof 原始 op_summary 的全部列，含 aicore 相关列） | `analysis/prof_view/_kernel_view_parser.py:15,57-68`；列映射见 `analysis/prof_common_func/_csv_headers.py:1-7`；Level1 → `is_all_kernel_headers()=True`，见 `analysis/_profiler_config.py:206-210` |
| `operator_details.csv` | torch 算子级：Name / Input Shapes / Call Stack / Host Self,Total Duration(us) / Device Self,Total Duration(us) / Device Self,Total Duration With AICore(us) | `analysis/prof_view/_operator_view_parser.py:17-19` |
| `op_statistic.csv` | 按 OP Type + Core Type 汇总：Count / Total Time(us) / Min / Avg / Max / Ratio(%) | `analysis/prof_view/_integrate_parser.py:23`；列名定义 `analysis/prof_bean/_op_statistic_bean.py:22-23` |
| `api_statistic.csv` | CANN API 调用统计（Level1/Level2 才有） | `analysis/_profiler_config.py:30-36`、`_integrate_parser.py:22` |
| `npu_module_mem.csv` | 模块级 NPU 内存（Level0/1/2 都有） | `analysis/_profiler_config.py:27-37` |
| `communication.json` | 按 step 分组的 HCCL 通信算子：通信时间信息 + 带宽信息（Size 分布、Wait Time(ms)、Transit Time(ms)、Transit Size(MB)、Bandwidth(GB/s)、同步时间及占比、Transport Type） | `analysis/prof_view/_communication_parser.py:19-42,85-95` |
| `communication_matrix.json` | 按 step / 算子 / link 的通信矩阵（逐 rank 对） | `analysis/prof_view/_communication_parser.py:97-104` |
| `trace_view.json` | Chrome tracing：CANN timeline + torch 算子 + flow 事件（framework ↔ device 关联） | `analysis/prof_view/_trace_view_parser.py:19,66-78` |
| `memory_record.csv` / `operator_memory.csv` | 内存记录（需要 profile_memory） | `analysis/prof_view/_memory_view_parser.py:26-27` |
| `analysis.db` / `ascend_pytorch_profiler_<rank>.db` / `msprof_<n>.db` | db 型产物 | `analysis/prof_common_func/_constant.py:304`、`prof_db_parse/_db_parser.py:44`、`_cann_file_parser.py:69` |
| `nic.csv` / `roce.csv` / `pcie.csv` / `hccs.csv` / `l2_cache.csv` / `data_preprocess.csv` | 需要 `sys_io`/`sys_interconnection`/`l2_cache`，vllm-ascend 默认都没开 | `_integrate_parser.py:15-25` |

> 注意：现场 `service_profiling_guide.md` 的产物列表里列了 `analysis.db`，
> 但按 2.2 的硬编码 `export_type=Text`，只有 `msprof` 自己支持"默认导出 db"时
> 才会附带 db（`analysis/_profiling_parser.py:135-143`）。以实际目录为准，
> 不要假设一定有 db。

### 3.4 `data_simplification` 与 Level 的副作用（重要）

- `data_simplification=True`（硬编码）：**解析完成后会删原始数据**——设备/host 下的
  `sqlite`、`summary`、`timeline`，以及 `PROF_*/` 下的 `analyze/`、
  `mindstudio_profiler_log/`、`mindstudio_profiler_output/`
  （`torch_npu/profiler/analysis/_profiling_parser.py:33-60,102-104`）。
  即：一旦解析成功，就不能再用别的参数重解析，`ASCEND_PROFILER_OUTPUT/` 成为唯一证据，
  **务必回传**；反过来，如果解析失败（见 4.1 的 daemon 情况），原始数据还在，可以重来。
- `profiler_level=Level1`（硬编码）→ `trace_view.json` 不做裁剪
  （`analysis/_profiler_config.py:38-43`）；如果是 Level0 会裁掉
  `Hccl`、`Communication`@、`CANN`、`GE` 等前缀的事件，通信分析会残废。要改 level
  只能改 `torch_npu_profiler.py:52`。
- 数据量告警：单目录数据 >1GB 时会打印"解析时间预计超过 30 分钟"
  （`torch_npu/profiler/analysis/prof_view/cann_parse/_cann_export.py:147-149`，
  阈值 `_constant.py:34`）。一次只抓 1 个 step 就是为了避免这个。

---

## 4. 产物处理与分析

### 4.1 官方解析入口：`torch_npu.profiler.profiler.analyse`

```python
from torch_npu.profiler.profiler import analyse     # 注意：不是 torch_npu.profiler.analyse
analyse("/abs/path/prof/cp_on")                      # 可以给父目录，会解析下面所有 *_ascend_pt
analyse("/abs/path/prof/cp_on/<某rank>_ascend_pt", export_type="text")
```

- 函数签名：`analyse(profiler_path, max_process_number=cpu_count()//2, export_type=None)`
  （`torch_npu/profiler/profiler.py:327-348`）。`export_type` 只能是 `"text"`/`"db"`
  或其列表，非法值会重置为 None（用采集时记录的值）。
- 输入路径可以是单个 `*_ascend_pt`，也可以是包含多个 `*_ascend_pt` 的父目录
  （`analysis/prof_common_func/_path_manager.py:107-121`）→ **一条命令解析 16 个 rank**。
- 输出固定写到每个 `*_ascend_pt/ASCEND_PROFILER_OUTPUT/`，完成后留
  `analyse.done`（`analysis/_profiling_parser.py:27-30,108-109`）。
- **依赖 CANN**：解析里会 `shutil.which("msprof")` 并执行
  `msprof --export=on --output=<cann路径>` 和 `msprof --analyze=on ...`
  （`analysis/prof_view/cann_parse/_cann_export.py:46,61-71`、
  `_cann_analyze.py:36,46-59`），还要校验 `msprof.py` 存在与路径权限。
  所以**解析必须在装了 CANN 工具链的 Linux 机器上做**；本机 Windows 不行
  （另外 `analysis/_npu_profiler.py:31` 强制 `set_start_method("fork")`，Windows 没有 fork）。
  官方离线解析说明：<https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/devaids/profiling/atlasprofiling_16_0034.html>（网络来源）。
- **不能是 daemon 进程**：`analysis/_npu_profiler.py:22-28` 会直接返回并提示
  "The profiling data cannot be parsed during the daemon process ... use an offline
  parsing interface"。而现场用的 `--distributed-executor-backend mp` 正是这种情况：
  `"mp"` → `MultiprocExecutor`（`vllm/v1/executor/abstract.py:69-72`），
  其 worker 进程以 `daemon=True` 创建（`vllm/v1/executor/multiproc_executor.py:694-698`）
  →
  **`/stop_profile` 之后不会自动产生 `ASCEND_PROFILER_OUTPUT`，必须按上面的方式补跑一次**；
  日志里会看到 `[ERROR] [... profiler.py: The profiling data cannot be parsed during the
  daemon process ...` 。判断是否已经自动解析过：看目录里有没有 `ASCEND_PROFILER_OUTPUT/`
  和 `analyse.done`。
- 单进程内多处调用有坑：`ProfilerConfig` 是 Singleton 且有 `_is_load` 只加载一次
  （`analysis/_profiler_config.py:25-26,147-156`）→
  不要在一个 python 进程里循环 `analyse()` 多个不同 rank 目录（每个都会被 fork 成子进程，
  单进程循环时会串配置）。稳妥做法是每个目录起一个进程，或一次给父目录交给官方并行池。
- 解析前的现场检查：目录属主必须是当前用户或 root、不能 other-writable，否则会报
  `Please execute 'chown -R $(id -un) ...'` / `chmod -R 755 ...`
  （`analysis/prof_view/cann_parse/_cann_export.py:82-120`）。

第三方/可视化：
- `trace_view.json` 用 MindStudio Insight 打开（`service_profiling_guide` 第 5 节）。
- `trace_view.json` 也能直接喂 <https://ui.perfetto.dev/>（同格式），但 Ascend 的
  通信/流水信息在 MindStudio Insight 里更全（`vllm/docs/contributing/profiling.md:28` 只提了 perfetto）。

### 4.2 上游 vLLM 自带的分析脚本：没有

- grep `vllm/` 全树，没有解析 `trace_view.json` / `kernel_details.csv` 的脚本
  （`grep -rln "chrome_tracing\|traceEvents" --include=*.py vllm/` 无命中）。
- vLLM 侧的分析产物只有 GPU 上的 `profiler_out_<rank>.txt`（`key_averages().table()`，
  仅当 `torch_profiler_dump_cuda_time_total`，GPU-only：
  `vllm/profiler/wrapper.py:266-286`）——Ascend 不会生成。
- `vllm/profiler/layerwise_profile.py` 是 GPU-only 的离线分层分析
  （`_ModuleTreeNode.is_cuda` 判 `DeviceType.CUDA`，`layerwise_profile.py:43-52`），
  不能用于 NPU trace。
- 因此**离线分析要自己写**，第 4.5 节给了可直接用的脚本。

### 4.3 纯离线（只有 trace 文件、没有 NPU）能算什么

前提：解析（4.1）已经在有 CANN 的机器上完成，拿到 `ASCEND_PROFILER_OUTPUT/`。
以下都能在 Windows 上用纯 Python 完成：

能算：
- kernel/算子设备时长、次数、占比（`kernel_details.csv`、`op_statistic.csv`）；
- torch 算子 host/device 时长（`operator_details.csv`）；
- 通信算子（HCCL）耗时、次数、单次带宽、wait/transit 时间（`communication.json`、
  `communication_matrix.json`），或在 `kernel_details.csv` 里按名字前缀筛 `hcom_*`；
- 每 step 的 compute / communication / overlapped / free / bubble
  （`step_trace_time.csv`）；
- 算子间 gap / 空洞（从 `trace_view.json` 按 stream(pid,tid) 排序相减，见 4.5）；
- aicore 相关指标（`aic_metrics=PipeUtilization` + Level1 → `kernel_details.csv` 的
  aicore 列，能近似算"aicore 忙时间/利用率"；`op_statistic.csv` 给到算子级总量）。

不能算 / 有限制：
- 不能重新生成 CSV（`data_simplification=True` 已删原始数据）；
- 不能做 `export_stacks` / `export_memory_timeline` 之类的二次导出；
- `ProfilerActivity.NPU` 的字段缺失时解析器会直接把 NPU activity 从 activities 里删掉
  （只对已有产物动手，不需要 NPU 卡，`analysis/_profiler_config.py:158-171`）；
- 真机联动的指标（NPU 频率、syscnt 对齐）来自 `profiler_info*.json` 与
  `PROF_*/host/info.json`，如果没回传这些，跨 rank 时间轴对齐只能靠 msprof 已写入的
  `ts` 字段（同一台机器上是同一时钟源，够用）。

### 4.4 `trace_view.json` 结构（Chrome tracing）

- 顶层：`{"traceEvents": [...], ...}`；框架事件由
  `TraceEventManager.create_x_event/create_m_event` 生成，字段是标准 chrome tracing：
  `ph`（`X` 完整事件 / `M` 元数据 / `s`、`f` flow 起止）、`name`、`cat`、`pid`、`tid`、
  `ts`、`dur`、`args`，flow 事件额外有 `id`、`bp`
  （`analysis/prof_common_func/_trace_event_manager.py:18-60`）。
- **单位是微秒**：`ts` 用 `convert_ns2us_str`、`dur` 用 `convert_ns2us_float`
  （同上 21-22 行），CANN 侧 timeline 也是 us（`_cann_file_parser.py:127-163` 里
  对 `ts` 做 `convert_us2ns` 反推）。
- `M` 事件给出 track 名称：`process_name` / `process_labels` / `thread_name` /
  `thread_sort_index`（`_constant.py:64-68`）。
- 通信算子名字约定：`hcom_send*` / `hcom_receive*` / `hcom_batchsendrecv*` 是 P2P，
  其余含 `allreduce/allgather/reducescatter/alltoall/broadcast/scatter/reduce` 的归为集合通信
  （`analysis/prof_view/_communication_parser.py:24-28,170-183`）。
- framework ↔ device 关联用 flow 事件（`cat="async_npu"` 的 `torch_to_npu`，
  `cat="HostToDevice"` 的 `acl_to_npu`），见
  `_trace_event_manager.py:52-60`、`_cann_file_parser.py:127-163`。

### 4.5 可直接用的离线脚本

**(a) trace_view.json 快速统计 + gap 分析**（纯标准库，Windows 可直接跑）：

```python
#!/usr/bin/env python3
"""统计 chrome tracing (trace_view.json) 的 per-category/per-name 耗时，并算 stream 间 gap。"""
import argparse, collections, json, sys

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("traceEvents", []) if isinstance(data, dict) else data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--filter", default="", help="只统计 name 含该子串的事件")
    ap.add_argument("--gaps", action="store_true", help="按 (pid,tid) 输出最大 gap")
    a = ap.parse_args()
    ev = load(a.trace)

    # 1) track 名字表（M 事件）
    tracks = collections.defaultdict(dict)
    for e in ev:
        if e.get("ph") == "M":
            args = e.get("args") or {}
            tracks[(e.get("pid"), e.get("tid"))][e.get("name")] = args.get("name") or args.get("labels") or ""

    # 2) X 事件聚合
    agg = collections.defaultdict(lambda: [0, 0.0, 0.0])   # count, total_us, max_us
    xev = []
    for e in ev:
        if e.get("ph") != "X" or not isinstance(e.get("dur"), (int, float)):
            continue
        if a.filter and a.filter not in str(e.get("name", "")):
            continue
        xev.append(e)
        k = (e.get("cat"), e.get("name"))
        agg[k][0] += 1
        agg[k][1] += float(e["dur"])
        agg[k][2] = max(agg[k][2], float(e["dur"]))
    print("events=%d  X=%d" % (len(ev), len(xev)))
    print("%-26s %-46s %8s %14s %14s" % ("cat", "name", "count", "total_us", "max_us"))
    for (cat, name), (c, t, mx) in sorted(agg.items(), key=lambda kv: -kv[1][1])[: a.top]:
        print("%-26s %-46s %8d %14.1f %14.1f" % (str(cat), str(name)[:46], c, t, mx))

    # 3) 每个 stream 内部的事件间隔（gap / bubble）
    if a.gaps:
        by_stream = collections.defaultdict(list)
        for e in xev:
            by_stream[(e.get("pid"), e.get("tid"))].append(e)
        print("\n# per-stream gaps (top 20 by total gap)")
        rows = []
        for k, evs in by_stream.items():
            evs.sort(key=lambda e: float(e["ts"]))
            gaps, prev_end = 0.0, None
            for e in evs:
                ts, dur = float(e["ts"]), float(e["dur"])
                if prev_end is not None and ts > prev_end:
                    gaps += ts - prev_end
                prev_end = max(prev_end or 0.0, ts + dur)
            rows.append((gaps, len(evs), k, tracks.get(k, {})))
        for gaps, n, k, label in sorted(rows, reverse=True)[:20]:
            print("pid=%s tid=%s events=%d total_gap_us=%.1f track=%s" % (k[0], k[1], n, gaps, label))

if __name__ == "__main__":
    sys.exit(main())
```

用法与判读：

```bash
python prof_trace_stats.py ASCEND_PROFILER_OUTPUT/trace_view.json --top 40
python prof_trace_stats.py trace_view.json --filter hcom_ --gaps      # 只看通信
python prof_trace_stats.py trace_view.json --filter aclnn --gaps      # 算子空洞
```

注意：NPU 上同一 stream 内的 kernel 之间还会有真实的数据搬运/依赖，gap 不全是空洞；
要区分"空洞"和"重叠"建议同时看 `step_trace_time.csv` 的
`Communication(Not Overlapped)` / `Overlapped` / `Bubble` 列。

**(b) `op_statistic.csv` 跨 rank 聚合 + 两次采集 diff**（纯标准库）：

```python
#!/usr/bin/env python3
"""聚合多个 rank 的 op_statistic.csv，并对比两次采集的算子总时长差异。"""
import argparse, collections, csv, glob, os, sys

COL_OP, COL_CORE, COL_TOTAL, COL_COUNT, COL_RATIO = "OP Type", "Core Type", "Total Time(us)", "Count", "Ratio(%)"

def read_run(root):
    """root 下所有 *_ascend_pt/ASCEND_PROFILER_OUTPUT/op_statistic.csv -> {(op,core): [total,count]}"""
    out = collections.defaultdict(lambda: [0.0, 0])
    files = glob.glob(os.path.join(root, "*_ascend_pt", "ASCEND_PROFILER_OUTPUT", "op_statistic.csv"))
    files += glob.glob(os.path.join(root, "ASCEND_PROFILER_OUTPUT", "op_statistic.csv"))
    for path in files:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                k = (row.get(COL_OP, ""), row.get(COL_CORE, ""))
                try:
                    out[k][0] += float(row.get(COL_TOTAL) or 0.0)
                    out[k][1] += int(float(row.get(COL_COUNT) or 0))
                except ValueError:
                    continue
    return out, len(files)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", required=True, help="cp_balance=1 的根目录（内含多个 *_ascend_pt）")
    ap.add_argument("--off", required=True, help="cp_balance=0 的根目录")
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()
    on, n_on = read_run(a.on)
    off, n_off = read_run(a.off)
    print("ranks with op_statistic.csv: on=%d off=%d" % (n_on, n_off))
    keys = set(on) | set(off)
    rows = [(k, on[k][0] - off[k][0], off[k][0], on[k][0], on[k][1] - off[k][1]) for k in keys]
    print("%-40s %-14s %12s %12s %12s %8s" % ("OP Type", "Core Type", "d_on-off(us)", "off(us)", "on(us)", "d_count"))
    for (op, core), d, o, n, dc in sorted(rows, key=lambda r: -abs(r[1]))[: a.top]:
        print("%-40s %-14s %12.1f %12.1f %12.1f %8d" % (op[:40], core[:14], d, o, n, dc))

if __name__ == "__main__":
    sys.exit(main())
```

**(c) `communication.json` 摘要**（结构是 `{step: {op_name: {"Communication Time Info": {...},
"Communication Bandwidth Info": {...}}}}`，op_name 形如 `<Op>@<link>`，
见 `_communication_parser.py:85-95,170-183`）：

```python
import json, sys, collections
data = json.load(open(sys.argv[1], encoding="utf-8"))
for step, ops in data.items():
    tot = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0])   # count, transit_ms, wait_ms, size_mb
    for name, info in ops.items():
        base = name.split("@")[0]
        bw = info.get("Communication Bandwidth Info", {})
        tot[base][0] += 1
        tot[base][1] += float(bw.get("Transit Time(ms)", 0) or 0)
        tot[base][2] += float(bw.get("Wait Time(ms)", 0) or 0)
        tot[base][3] += float(bw.get("Transit Size(MB)", 0) or 0)
    print("== %s ==" % step)
    for base, (c, t, w, s) in sorted(tot.items(), key=lambda kv: -kv[1][1]):
        print("  %-46s count=%-5d transit=%9.3f ms  wait=%9.3f ms  size=%8.3f MB" % (base[:46], c, t, w, s))
```

**(d) `step_trace_time.csv` 对比**：直接按 `Step` 行比两边的 `Computing` /
`Communication(Not Overlapped)` / `Overlapped` / `Communication` / `Free` / `Bubble`。
列名见 `analysis/prof_view/_trace_step_time_parser.py:47-48`。

---

## 5. 本项目（cp_balance 开/关）具体建议

### 5.1 采集窗口怎么控制

目标窗口 = **一次 prefill forward**（CP_BALANCE=0/1 两种实现都在 prefill 阶段分叉，
decode 阶段两边都会拒绝 zigzag：`docs/cp_balance_analysis.md` 第 3 节、
`README.md` 的"分支证明"一节）。

推荐做法（组合拳）：

1. 服务启动加 `--profiler-config '{"profiler":"torch", "torch_profiler_dir":"<abs>",
   "torch_profiler_with_stack":false, "ignore_frontend":true,
   "delay_iterations":0, "max_iterations":1}'`。
   - `max_iterations=1`：`WorkerProfiler` 在记录到 1 个 execute_model 后在**下一步开始前**
     自动 stop 并落盘（`vllm/profiler/wrapper.py:83-114`；语义有单测
     `tests/ut/profiler/test_torch_npu_profiler.py:265-290`）→ 恰好 1 个 engine step。
   - `ignore_frontend=true`：不产生 API server 的 CPU trace，避免额外开销/噪声文件，
     也避免 `vllm/config/profiler.py:139-145` 的 "high overhead" 警告。
   - `torch_profiler_with_stack=false`：Ascend 上等于关掉 `with_modules`，
     数据量显著下降（`service_profiling_guide` 0 节、`torch_npu_profiler.py:68-70`）。
2. 请求用**纯 prefill**：`prompt` 取 `questions.json` 的 `items[20]`（第一个 long 用例，
   约 4400 字符），`max_completion_tokens=1`（或 `max_tokens=1`）——只要 1 个 token，
   引擎不会再排 decode step，窗口里就只有 prefill。**先确认这条 prompt 的 token 数
   ≥ `min_tokens`**：`check_branch.py` 会打印
   `[check] long: tokens=<N> (min_tokens=2048) -> expect ZIGZAG`，N 必须大于 min_tokens
   （`cp_balance/check_branch.py:134-139`）；采集脚本用同一下标（`--index 20`），
   保证"验证过的那条 prompt"就是"采集用的那条"。
3. 序列固定为：**服务就绪 → 同一 prompt 先跑 1 次 warmup（不吃污染窗口）→
   `POST /start_profile` → 发同一条请求 → 收到响应后 `POST /stop_profile`**。
4. 采集前必须确认引擎**没有在途请求**（上一轮请求已经收完，避免把别人的 decode
   step 记进来；`vllm/v1/engine/core.py:583-587` 在没有请求时不执行模型，
   所以空闲期不会消耗窗口）。
5. **每轮采集结束后一定要显式调一次 `/stop_profile`**：`max_iterations` 触发的是
   内部 stop，wrapper 的 `_active` 仍为 True，而 `WorkerProfiler.start()` 在 `_active`
   时会直接忽略下一次 start（`vllm/profiler/wrapper.py:71-79,126-140`）→
   不补 stop，第二次 `/start_profile` 会静默失效。
6. 想保留完整 trace（不自动停）时把 `max_iterations` 设为 0，靠手工 `/stop_profile`
   控制；但那样窗口里会混入响应返回后的空闲/其他 step，只建议做辅助验证。

### 5.2 一次采集多少 step、怎么保证两次条件一致

- **一次采集 = 1 个 step**，但**每组开关采 3 轮**（3 次 start/stop，产生 3 个
  `*_ascend_pt` 目录），用来判断"差异是否大于轮间噪声"——这与项目现有的
  `base_off` / `base_off2` 噪声地板思路一致（`README.md` 的"一条命令跑完四组实验"）。
- 一致性清单（两边必须逐项相同）：
  1. 同一份 `configs/*.json`，只有 `cp_balance`（和 `port`、profiler 目录）不同；
     同一 `repo_tree`（`cur`，路径由 `harness.json trees` 解析）、同一 `server_args`
     （特别注意 `--enforce-eager`、`--async-scheduling`、`--no-enable-prefix-caching`、
     `--max-num-batched-tokens`、`--max-num-seqs`、`--quantization` 一致）；
  2. 同一 `additional_config`（`enable_dsa_cp`、`enable_sparse_sfa_c8/li_c8`、
     `enable_flashcomm1` 等）与同一 `env`（`HCCL_ALGO/BUFFSIZE`、`VLLM_ASCEND_ENABLE_FLASHCOMM1`）；
  3. 同一 prompt 文本、同一 `max_completion_tokens=1`、`temperature=0`、
     同一请求顺序（先 warmup 再采集）；
  4. 同一 profiler 配置（含 `max_iterations`）；
  5. 采集前确认走了目标分支：跑 `python accuracy/check_branch.py --url ... --log <log>`，
     long 用例必须出现 zigzag 证据（默认 `branch=ZIGZAG` 或 `[CP_BALANCE][plan]`，
     可在 `harness.json verify.diagnose.zigzag_evidence` 配置；开关关时只有 `branch=CONTINUOUS`），
     `README.md` 判据 `[check] RESULT: PASS`；
  6. 不要让 `VLLM_ASCEND_CP_BALANCE_MIN_TOKENS` 落在 prompt token 数的临界点上
     （现场 2048），否则两次采集可能一个进 zigzag 一个不进。
- 明确一个方法学限制：**profiler 本身会改变时序**（尤其开了 `with_modules`、
  memory profiling 时）。所以：
  - 结论量化（"快了多少毫秒"）用项目现有的 TTFT/端到端测量（如
    `tests/e2e/pull_request/four_card/test_profiling_chunk_performance.py` 的写法）；
  - profiler 只用于**归因对比**（哪类算子/通信/空洞变了），
    并且在两边配置完全相同时比较**相对差异**。

### 5.3 应该对比哪些指标（可计算清单）

| # | 指标 | 数据来源 | 算法 | 关注点 |
|---|---|---|---|---|
| 1 | 单 step 总时长（forward 时间） | `step_trace_time.csv` 的 `Stage` 列；或 `trace_view.json` 里 `ProfilerStep#`/forward 区间 | 直接读列 | cp_balance 是否真的缩短 prefill |
| 2 | Computing / Communication / Overlapped / Free / Bubble | `step_trace_time.csv` | 直接读列 | 是"计算变少"还是"重叠变差" |
| 3 | 每层 attention / MLP / MoE 时间 | `kernel_details.csv`（按名字里的 layer 序号 + 算子类型筛选）、`operator_details.csv` | 正则分组求和 | zigzag 是否把 attention 的 SFA/LI 时间拉平 |
| 4 | 每层 attention 在 rank 间的均衡度 | 同 3，按 rank 目录汇总 | `max/mean`、`min/max` | cp_balance 的核心收益指标 |
| 5 | HCCL 集合通信耗时与次数（all_gather / reduce_scatter / allreduce / alltoall） | `communication.json`、`communication_matrix.json`；或 `op_statistic.csv` / `kernel_details.csv` 里 `hcom_*` | 注：本仓库的 export/ 默认**不**回传 communication*.json（每 rank 上 GB，脚本不解析），看 op_statistic.csv 的 hcom_* 行；真要链路带宽矩阵就从原始 trace 目录里取 | 按 Op 名 + link 求和/计数 | cp_balance 会引入额外通信块交换，要看净收益 |
| 6 | 通信带宽 / 单次传输量 / wait 时间 | `communication.json` 的 `Communication Bandwidth Info` | 求和 + Bandwidth(GB/s) | 通信是否退化成小包多次 |
| 7 | 算子间 gap / 空洞 | `trace_view.json`（4.5a 的 `--gaps`） | 同 stream 排序相减 | 额外通信是否被计算掩盖 |
| 8 | 算子耗时占比 Top-N 变化 | `op_statistic.csv` | 两次 diff（4.5b） | 哪个算子被引入/消除 |
| 9 | aicore 利用率 | `kernel_details.csv`（Level1 + `PipeUtilization`）；`op_statistic.csv` 的 Total Time/Ratio | aicore 列 / task duration | 计算是否更"实" |
| 10 | host 侧算子耗时（python/CANN API） | `operator_details.csv`、`api_statistic.csv` | 求和/计数 | 是否只是把时间挪到 host |
| 11 | NPU 内存 | `npu_module_mem.csv`（Level1 自带） | 直接读 | gather/index_select 缓冲是否涨显存 |
| 12 | 分支证据（必须留档） | 服务日志 `[CP_BALANCE][branch]` / `[CP_BALANCE][plan]` 行 | `check_branch.py` | 证明这次采集真的走了目标分支 |

### 5.4 远端执行命令清单（可直接抄）

约定：远端项目目录 `/opt/its/z30055003/cp_balance`，采集产物 `/opt/its/z30055003/cp_balance/prof/`；
节点 IP/NIC 见 `configs/_common.json`（`7.246.78.75` / `eth2`）。

**步骤 1：直接用仓库里的 `prof_*` 配置（不要派生）**

profiling 的配置已经在 `configs/` 里维护好了：`prof_cur_cp0/cp1`（A3）、`prof_a5_cur_cp0/cp1`（A5）、
还有 repeat / base / a2a 变体；它们继承 `_profile_base.json`（profiler 段 + `lengths`）与各自的 `_common*.json`。

- `profiler` 段自动生成 `--profiler-config`，输出目录默认 `<harness>/<配置名>`（不用手写 `torch_profiler_dir`，
  也不用绝对路径）；
- 采集轮 `debug: 0`（`_profile_base.json` 定的，避免 `[CP_BALANCE][plan]` 的 `.tolist()` D2H 污染 trace）；
  分支证据另外取证：`bash verify.sh --family <a5|a3>`，或起一次 `glm52_*_cur_cp1` 跑 `accuracy/check_branch.py`；
- 树用 `repo_tree: cur|base`，不要再往配置里写 `/opt/...` 这类绝对路径。

```bash
cd <harness>
bash run.sh prof_a5_cur_cp1 --dry-run --print-env | tail -5    # 确认 argv 里有 --profiler-config
bash tests/run_tests.sh --only perf/p10_capture --skip perf/p10_capture#prof_a2a
```

（下面这段是历史做法，仅作参考——它按旧的 `server_args` list / `repo` 路径 / 手写 `--profiler-config` 写，
现在配置 schema 已经变了，派生出来跑不起来）：

```bash
cd /opt/its/z30055003/cp_balance
mkdir -p prof/cp_on prof/cp_off
chmod 755 prof prof/cp_on prof/cp_off

python3 - <<'PY'
import json, pathlib
d = pathlib.Path('/opt/its/z30055003/cp_balance/configs')
common = json.loads((d / '_common.json').read_text(encoding='utf-8'))
prof_root = '/opt/its/z30055003/cp_balance/prof'

def derive(name, port, cp_balance, prof_dir):
    out = dict(common)
    out['name'] = name
    out['port'] = port
    out['cp_balance'] = cp_balance
    out['debug'] = 1                       # 让服务日志打 [CP_BALANCE][branch]
    out['repo'] = '/opt/its/z30055003/vllm-ascend'
    out['server_args'] = list(common.get('server_args') or []) + [
        '--profiler-config',
        json.dumps({"profiler": "torch", "torch_profiler_dir": prof_dir,
                    "torch_profiler_with_stack": False, "ignore_frontend": True,
                    "delay_iterations": 0, "max_iterations": 1},
                   separators=(',', ':')),
    ]
    return out

(d / 'prof_cp1.json').write_text(
    json.dumps(derive('prof_cp1', 8035, 1, prof_root + '/cp_on'), indent=2, ensure_ascii=False),
    encoding='utf-8')
(d / 'prof_cp0.json').write_text(
    json.dumps(derive('prof_cp0', 8034, 0, prof_root + '/cp_off'), indent=2, ensure_ascii=False),
    encoding='utf-8')
print(json.dumps(json.loads((d / 'prof_cp1.json').read_text(encoding='utf-8'))['server_args'][-2:], ensure_ascii=False))
PY

bash run.sh prof_cp1 --dry-run --print-env | tail -5   # 确认 argv 里带了 --profiler-config
bash run.sh prof_cp0 --dry-run | head -3               # 确认除 port/cp_balance 外与上面一致
```

**步骤 2：写采集脚本 `prof_capture.py`（放远端项目目录）**

```bash
cat > /opt/its/z30055003/cp_balance/prof_capture.py <<'PY'
#!/usr/bin/env python3
"""cp_balance profiling: 一次 = 1 个 prefill step。先 warmup，再 start_profile -> 请求 -> stop_profile。"""
import argparse, json, time, urllib.request
from pathlib import Path

def post(url, payload=None, timeout=1800):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode(errors="replace")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--questions", default="/opt/its/z30055003/cp_balance/questions.json")
    ap.add_argument("--index", type=int, default=20, help="questions.json items 下标，>=20 是 long")
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    base = a.url.rstrip("/")
    items = json.loads(Path(a.questions).read_text(encoding="utf-8"))["items"]
    item = items[a.index]
    prompt = item["prompt"]
    payload = {"model": "glm-52", "prompt": prompt,
               "max_completion_tokens": a.max_tokens, "temperature": 0}
    print("[capture] prompt chars=%d kind=%s" % (len(prompt), item.get("kind")))
    for i in range(a.warmup):
        t0 = time.perf_counter()
        post(base + "/v1/completions", payload)
        print("[capture] warmup %d done in %.3fs" % (i, time.perf_counter() - t0))
    records = []
    for r in range(a.rounds):
        time.sleep(3)                                     # 确认引擎空闲
        st, _ = post(base + "/start_profile")
        t0 = time.perf_counter()
        _, body = post(base + "/v1/completions", payload)
        dt = time.perf_counter() - t0
        time.sleep(1)
        post(base + "/stop_profile")                      # 阻塞到落盘完成
        first = json.loads(body)["choices"][0]["text"]
        print("[capture] round %d start_profile=%s forward_wall=%.3fs first_token=%r"
              % (r, st, dt, first))
        records.append({"round": r, "wall_s": dt, "first_token": first})
    if a.out:
        Path(a.out).write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[capture] wrote", a.out)

if __name__ == "__main__":
    main()
PY
```

**步骤 3：跑 cp_balance=1 并采集**（建议在 tmux/screen 会话里执行，服务用 `&` 放后台）

```bash
cd /opt/its/z30055003/cp_balance
bash run.sh prof_cp1 > prof_cp1.log 2>&1 &
until curl -sf http://127.0.0.1:8035/v1/models >/dev/null; do sleep 10; done
echo "[ready] cp_on"

# 先证明走的是 ZIGZAG 分支（需要 debug=1，日志里会打 [CP_BALANCE][branch]）
python check_branch.py --url http://127.0.0.1:8035 --log prof_cp1.log | tail -5

rm -rf prof/cp_on/*        # 清掉上一次的产物
python prof_capture.py --url http://127.0.0.1:8035 --out cp_on_wall.json

ls -d prof/cp_on/*_ascend_pt | wc -l        # 期望 16（TP16）
du -sh prof/cp_on

# 停服务：serve_config.py 不做进程组隔离，要按端口/命令行杀掉真正的 vllm serve
pkill -f -- "--port 8035"
until ! curl -sf http://127.0.0.1:8035/v1/models >/dev/null 2>&1; do sleep 5; done
echo "[stopped] cp_on"
```

**步骤 4：解析（在同一台有 CANN 的机器上）**

```bash
cd /opt/its/z30055003/cp_balance
python3 - <<'PY'
from torch_npu.profiler.profiler import analyse
analyse("/opt/its/z30055003/cp_balance/prof/cp_on")     # 一次解析 16 个 rank 目录
PY

ls prof/cp_on/*_ascend_pt/ASCEND_PROFILER_OUTPUT/analyse.done | wc -l   # 期望 16
```

如果日志里出现 `The profiling data cannot be parsed during the daemon process...`，
说明 stop 时没自动解析，上面这一步就是**必须**的（mp 后端下是常态）。
解析后原始 timeline/summary 会被 `data_simplification` 删掉，先别急着清目录。

**步骤 5：跑 cp_balance=0 并采集（同 4 步，换 config/port/目录）**

```bash
cd /opt/its/z30055003/cp_balance
bash run.sh prof_cp0 > prof_cp0.log 2>&1 &
until curl -sf http://127.0.0.1:8034/v1/models >/dev/null; do sleep 10; done
python check_branch.py --url http://127.0.0.1:8034 --log prof_cp0.log | tail -5
rm -rf prof/cp_off/*
python prof_capture.py --url http://127.0.0.1:8034 --out cp_off_wall.json
python3 - <<'PY'
from torch_npu.profiler.profiler import analyse
analyse("/opt/its/z30055003/cp_balance/prof/cp_off")
PY
pkill -f -- "--port 8034"
until ! curl -sf http://127.0.0.1:8034/v1/models >/dev/null 2>&1; do sleep 5; done
```

**步骤 6：远端分析摘要（可选，先出个粗结论）**

```bash
cd /opt/its/z30055003/cp_balance
for r in prof/cp_on/*_ascend_pt; do
  echo "== $r"; cat "$r/ASCEND_PROFILER_OUTPUT/step_trace_time.csv"; done | head -80
python prof_compare.py --on prof/cp_on --off prof/cp_off --top 40      # 把 4.5(b) 存成这个文件
```

**步骤 7：打包回传**

```bash
cd /opt/its/z30055003/cp_balance
tar czf cp_prof_small.tgz \
  --exclude='*/FRAMEWORK/*' \
  --exclude='*/PROF_*' \
  prof/cp_on/*_ascend_pt/ASCEND_PROFILER_OUTPUT \
  prof/cp_off/*_ascend_pt/ASCEND_PROFILER_OUTPUT \
  prof/cp_on/*_ascend_pt/profiler_info*.json \
  prof/cp_off/*_ascend_pt/profiler_info*.json \
  prof_cp1.log prof_cp0.log cp_on_wall.json cp_off_wall.json
ls -lh cp_prof_small.tgz

# 完整 trace（可能很大，只挑代表性 rank；trace_view.json 压缩后仍然可观）
tar czf cp_prof_trace_r0.tgz prof/cp_on/*rank0*_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json
tar czf cp_prof_trace_r0_off.tgz prof/cp_off/*rank0*_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json
```

**步骤 8：本机（Windows Git Bash）拉回**

```bash
mkdir -p /d/code/cp_balance/prof
scp <user>@7.246.78.75:/opt/its/z30055003/cp_balance/cp_prof_small.tgz /d/code/cp_balance/prof/
tar xzf /d/code/cp_balance/prof/cp_prof_small.tgz -C /d/code/cp_balance/prof/
# 或者按需 rsync 只拉小文件：
# rsync -avm --include='*/' --include='ASCEND_PROFILER_OUTPUT/*.csv' \
#   --include='ASCEND_PROFILER_OUTPUT/*.json' --exclude='*' \
#   <user>@7.246.78.75:/opt/its/z30055003/cp_balance/prof/cp_on/ /d/code/cp_balance/prof/cp_on/
```

### 5.5 需要回传的目录/文件清单

必回传（小，判据所需）：
- `prof/cp_on/*_ascend_pt/ASCEND_PROFILER_OUTPUT/step_trace_time.csv`（16 份）
- `.../op_statistic.csv`、`.../operator_details.csv`、`.../kernel_details.csv`
- `.../communication.json`、`.../communication_matrix.json`
- `.../api_statistic.csv`、`.../npu_module_mem.csv`
- 每个 `*_ascend_pt/profiler_info*.json`（含 rank、level、export_type、起止时间）
- `prof_cp1.log` / `prof_cp0.log`（含 `[cp_balance] CONFIG=...` 指纹、
  `[CP_BALANCE][branch]` 行、profiler 的 INFO/ERROR 行）
- `cp_on_wall.json` / `cp_off_wall.json`（每轮 wall time + 首 token）

按需回传（大）：
- `ASCEND_PROFILER_OUTPUT/trace_view.json`（先只回 rank0；必要时再回 rank7/rank15）
- 若解析还没跑成，则回整个 `*_ascend_pt/`（含 `PROF_*`）；**注意**：
  `/stop_profile` 时自动解析成功的话，`PROF_*/` 里的
  `timeline/`、`summary/`、`sqlite/` 已被删掉，别以为丢数据了。

### 5.6 判读时的坑

- `/start_profile` 静默无效：多半是上一轮没调 `/stop_profile`
  （`vllm/profiler/wrapper.py:71-79`）。
- 只拿到 1 个 rank 的目录：`torch_profiler_dir` 是相对路径，
  被解析成每个进程 CWD 下的不同绝对路径 → 一律写绝对路径
  （`vllm/config/profiler.py:154-156`）。
- 目录里出现 `*.async_llm/`：`ignore_frontend` 没开，属于 API server 的 CPU trace。
- `ASCEND_PROFILER_OUTPUT` 目录不存在但 `PROF_*` 在：解析没跑（daemon 进程限制），
  按 4.1 手工补跑。
- stop 卡很久：stopping 时要 flush（甚至触发 msprof 解析），
  `curl`/客户端超时要放到分钟级（`vllm/docs/contributing/profiling.md:36-38` 也提醒了这点）。
- 只看 trace 里的算子名可能找不到"每层"：`kernel_details.csv` 的 `Name` 里带层序号，
  用正则分组；如果开了图模式就分不出来，必须先回到 `--enforce-eager`。

---

## 6. 时间线（把窗口对齐到"这一步"）

- worker 每个 `execute_model` 前调用一次 `self.profiler.step()`
  （`vllm_ascend/worker/worker.py:625-627`），这就是 `delay_iterations`/`max_iterations`
  的计数单位（engine step，不是 token、不是请求）。
- `profiler.start()` 在 `/start_profile` 的 RPC 处理里同步执行；`profiler.stop()`
  在 `/stop_profile`（或 max_iterations 到点）的 RPC 里同步执行并落盘。
- 由于 `torch_npu` 每次 `start()` 新建带毫秒时间戳的 `*_ascend_pt` 目录，
  一进程多轮采集可以用目录名时间戳对应 `cp_*_wall.json` 里的轮次顺序。

---

## 7. `service_profiling_guide.md` 关键内容摘录（译）

文档：`vllm-ascend/docs/source/developer_guide/performance_and_debug/service_profiling_guide.md`

- 提供两套方案：Ascend PyTorch Profiler（算子级、API 控制、无需额外安装）与
  MS Service Profiler（服务框架函数级、配置文件控制、需要 `msserviceprofiler`，
  CANN Toolkit 自带、也可源码构建）。
- Ascend PyTorch Profiler：
  - 启动时用 `--profiler-config` 指定落盘路径即开启；默认会采集 python stack，
    可用 `torch_profiler_with_stack=false` 关闭以显著减小数据量；
  - 采集由 `/start_profile`、`/stop_profile` 控制；**只有启动时设置了
    `--profiler-config` 才会注册这两个端点**，否则 404；
  - 数据落到 `*ascend_pt` 目录，需要先解析：
    `from torch_npu.profiler.profiler import analyse; analyse("<dir>/*_ascend_pt/")`；
  - 解析后看 `ASCEND_PROFILER_OUTPUT/`：`analysis.db`、`api_statistic.csv`、
    `ascend_pytorch_profiler_0.db`、`kernel_details.csv`、`operator_details.csv`、
    `op_statistic.csv`、`step_trace_time.csv`、`trace_view.json`（后者可用 MindStudio Insight 打开）；
  - PD 分离：P/D 各自独立配 `--profiler-config` 并**各自 curl 自己的端口**；
    主负载均衡代理不转发这两个端点；EPD 代理（`epd_load_balance_proxy_layerwise_server_example.py`）会广播。
- MS Service Profiler（摘要）：
  - 启动前 `export SERVICE_PROF_CONFIG_PATH=ms_service_profiler_config.json`、
    `export PROFILING_SYMBOLS_PATH=service_profiling_symbols.yaml`，然后正常起服务；
  - 采集开关：配置文件里 `"enable": 0 → 1`（`sed -i 's/"enable":\s*0/"enable": 1/'`）；
  - 解析：`msserviceprofiler parse --input-path=./ --output-path output`，
    产物 `chrome_tracing.json`、`profiler.db`、`request.csv`、`kvcache.csv`、`batch.csv`；
  - 主要配置项：`enable`、`prof_dir`（默认 `${HOME}/.ms_server_profiler`）、
    `profiler_level`、`acl_task_time`（0/1/2/3，3 表示走 Torch Profiler dump）、
    `acl_prof_task_time_level`（L0/L1[;秒数]）、`timelimit`（秒，0=不限）、
    `domain`（Request/KVCache/ModelExecute/BatchSchedule/Communication）、
    `torch_prof_stack`、`torch_prof_step_num`、`profiler_step_num`；
  - 符号表字段：`symbol`/`handler`/`domain`/`name`/`min_version`/`max_version`/`attributes`
    （`attributes` 的表达式见该文档 6.2 节；本仓库默认表见
    `vllm_ascend/profiling_config.py`）。

---

## 8. 来源与引用清单

本地代码（本机只读树）：
- `vllm/`（vLLM v0.26.0 上游）：`vllm/config/profiler.py`、`vllm/profiler/wrapper.py`、
  `vllm/v1/engine/async_llm.py`、`vllm/v1/executor/abstract.py`、
  `vllm/v1/executor/multiproc_executor.py`、`vllm/entrypoints/serve/profile/api_router.py`、
  `vllm/entrypoints/serve/__init__.py`、`vllm/distributed/utils.py`、
  `vllm/engine/arg_utils.py`、`vllm/utils/argparse_utils.py`、`vllm/benchmarks/serve.py`、
  `vllm/examples/features/profiling/*.py`、`vllm/docs/contributing/profiling.md`
- `vllm-ascend/`：`vllm_ascend/profiler/torch_npu_profiler.py`、`vllm_ascend/profiling_config.py`、
  `vllm_ascend/worker/worker.py`、`vllm_ascend/envs.py`、`vllm_ascend/ascend_config.py`、
  `vllm_ascend/__init__.py`、`tests/ut/profiler/test_torch_npu_profiler.py`、
  `tests/ut/worker/a2/test_worker_v1.py`、`requirements.txt`
- 本地文档：`docs/source/developer_guide/performance_and_debug/service_profiling_guide.md`、
  `docs/source/user_guide/feature_guide/graph_mode.md`、
  `docs/source/user_guide/configuration/additional_config.md`、
  `docs/source/user_guide/release_notes.md`、`docs/source/tutorials/models/GLM5.2.md`
- 项目侧：`cp_balance/README.md`、`configs/_base.json`（公共段）、`configs/_profile_base.json`、
  `serve_config.py`、`accuracy/run_matrix.py`、`accuracy/check_branch.py`、
  `accuracy/compare_first_token.py`、`docs/scripts_review.md`（当前状态）

torch-npu 官方源码（本机下载的 wheel，用于确认 API 语义与产物清单）：
- `torch-npu==2.10.0.post2`（版本号与 `vllm-ascend/requirements.txt:17` 一致），
  wheel 缓存于 `D:\code\cp_balance\.scratch\tnp_dl\`，
  已解出 `torch_npu/profiler/`（`ext/torch_npu/profiler/`）用于阅读；
  本文档所有 `torch_npu/profiler/...` 行号均指该 wheel 内文件。

网络来源（本地无法访问时用于交叉验证，未逐条实测）：
- `torch_npu.profiler.ExportType`：
  <https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/torchnpuCustomsapi/docs/zh/custom_APIs/torch_npu-profiler/torch_npu-profiler-ExportType.md>
- `torch_npu.profiler.profiler.analyse`：
  <https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/torchnpuCustomsapi/docs/zh/custom_APIs/torch_npu-profiler/torch_npu-profiler-profiler-analyse.md>
- `torch_npu.profiler._ExperimentalConfig`：
  <https://www.hiascend.com/document/detail/zh/Pytorch/60RC2/apiref/apilist/ptaoplist_000597.html>
- `torch_npu.profiler.profile`：
  <https://www.hiascend.com/document/detail/zh/Pytorch/2600/apiref/torchnpuCustomsapi/docs/zh/custom_APIs/torch_npu-profiler/torch_npu-profiler-profile.md>
- CANN 性能数据离线解析：<https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/devaids/profiling/atlasprofiling_16_0034.html>
- CANN 数据目录说明：<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta1/devaids/Profiling/atlasprofiling_16_0035.html>
- vLLM profiling env 重构 PR #29912（提交 `e858bfe051`）、移除 PR #33536（`ef248ff740`）、
  弃用 PR #33722（`d88a1df699`）；vllm-ascend PR #5928、#8953、#10778、#6141（本地 git 历史可查）
- vllm-ascend RFC #6954（NPUWorker Profiler profile_prefix 适配）：
  <https://github.com/vllm-project/vllm-ascend/issues/6954>
