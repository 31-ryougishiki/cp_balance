# cp_balance 脚本分阶段审查（2026-09-21）

对象：`run.sh` / `serve_config.py` / `configs/*` / `harness.json` / `verify.sh` / `tests/**` / `accuracy/*.py` / `perf/*.py`。
方式：4 个只读 reviewer 分阶段看（启动层、harness 层、测试与驱动、运行期契约），本地只做静态检查
（`bash -n`、`py_compile`、`--dry-run`、改动前后 36 份 dry-run 指纹/参数集合对比、`verify.sh --diag-only` 合成证据）。
**服务/NPU 相关的结论一条都没有在本地验证过，全部列在"需远端确认"。**

## 0. 结论

阻塞 3 条，都已处理（细节见各节）：

1. `enable_dsa_cp` 在 dp=1 下被 `ascend_config.py:664` 的 `use_sequence_parallel_moe` 前置关掉 →
   已确认 dp=1 是成立配置并改掉（被测树 commit、参照树走 `base-dp1` 分支），见 §4.1。
2. 诊断/测试要的 `branch=ZIGZAG` 日志代码从来没打过（只有 `branch=CONTINUOUS` 与 `[CP_BALANCE][plan]`）→
   harness 侧两种都认，代码侧补了一行对称日志，见 §4.2。
3. `harness.json` 曾被 `.gitignore` 的 `/*.json` 吞掉，远端 pull 下来会没有这个文件 → 已加白名单，见 §2。

仍未处理、需要你决策的：`_profile_a5_common.json` 的 `deterministic:false`、死变量
`VLLM_RPC_TIMEOUT`/`VLLM_ASCEND_ENABLE_PREFETCH_MLP`、`p21` 的 Step 列、`round2_verify` 锚点重写、
MTP draft 的 `for_draft` 死代码（都列在对应小节的"待定"）。

脚本本身的配置化已经做完：机器差异、服务参数、profile 公共段、树清单、验证步骤与判据全部进
`configs/*.json` + `harness.json`，脚本里只剩控制流。

## 1. 阶段 A：启动层（run.sh / serve_config.py / configs）

已修：

| 问题 | 修法 |
| --- | --- |
| `bash run.sh --dry-run <cfg>`（flag 在前）把 `default` 和用户的位置参数一起传下去，argparse 退出 2 | run.sh 只做 `exec serve_config.py "$@"` |
| `--print-env` 不带 `--dry-run` 被忽略，直接起真服务 | `--print-env` 自动隐含 `--dry-run` 并警告 |
| `--set repo=../xxx` 在 absolutize 之后应用，相对路径不绝对化 | 先 `--set` 再 `absolutize` |
| profiler 目录绝对化少 `startswith("/")` 判断，Windows 下把 `/opt/...` 拼成 `D:\opt\...` | 统一的 `_is_abs()` |
| `PYTHONPATH=<repo>:` 结尾空项等于把 cwd 加进 `sys.path` | 只 join 非空项 |
| `--set port=` 之类给原始 ValueError traceback | 数值字段给出可读报错 |
| 配置 `env` 与顶层字段（debug/devices/...）冲突时静默后者赢 | 打 WARN 列出冲突键 |
| auto IP/NIC 识别失败后无任何提示，`HCCL_IF_IP`/`*_SOCKET_IFNAME` 全不设 | 识别不出来时 WARN 并给出修法 |
| `git_head()` 无超时（仓库卡住会拖死启动） | `timeout=5` |
| 子进程收到 SIGTERM/SIGINT 不转发，`wait()` 负数状态码直接 `SystemExit(-15)` | 转发信号 + `os.waitstatus_to_exitcode` |
| 指纹只有 CONFIG/MODEL/PORT 等，env/参数不同的配置指纹相同 | 指纹加 `ARGS=<sha1(env+server_args+additional_config+speculative+hf_overrides)>` |
| `name` 在合并之后才 setdefault，子配置不写 name 会继承父名 | 合并前 setdefault |
| `resolve()` 先看 cwd，同名文件会顶掉 `configs/` 里的 | 裸名字优先 `configs/`；矩阵配置不再能被当服务配置启动 |
| `--print-env` 的变量白名单（REPORT_ENV）写死，新加 env 看不到 | 由 `build_env(cfg, managed)` 自动收集 |
| `deterministic_env`（LCCL/HCCL/ATB 4 个）在代码里写死 | 进 `configs/_base.json`，`harness.json serve.deterministic_env` 与配置都可覆盖 |

配置化（去重）：

- `configs/_base.json`：所有线共用的 served name / port / cp_balance / min_tokens / reduce_mode / debug /
  deterministic / additional_config / env / server_args（dict 形式，`true`=裸开关，`false`=丢弃继承项）。
- `configs/_common.json`（A3）、`_common_a5.json`（A5）：只剩机器差异（model / ip / nic / devices / tp / prelude / env 增量 / 覆盖项）。
- `configs/_profile_base.json`：profiler 段 + `lengths`，两个族的 profile 公共配置只写 `extends`。
- `extends` 支持数组（`["_common_a5.json", "_profile_base.json"]`，后者覆盖前者）。
- 26 条配置里的 `repo: ../vllm-ascend*` 改成 `repo_tree: cur|base`，路径只在 `harness.json trees` 一处；
  矩阵的 `static_check.repo/base_repo` 同样改成 `repo_tree/base_tree`。
- 14 个 `prof_*` 配置里的 `profiler.dir`（9 条是 `/opt/its/z30055003/...` 绝对路径）删掉，
  默认就是 `<harness>/<配置名>`，与测试里按配置名找产物一致。
- 36 条配置改动前后 dry-run 对比：指纹除 `ARGS`/`--print-env` 新增行外一致，vLLM 参数集合逐条相同
  （唯一差异是 `--max-num-seqs` 位置移到列表末尾，argparse 与顺序无关）。

待定（未改，列出来给决策）：

- `configs/_profile_a5_common.json` 的 `deterministic:false` 与 A5 服务配置的 `deterministic:true` 不一致，
  A3 的 profile 却保持 true。性能数字是否要按服务同款 setting 采，需要定一下。
- `VLLM_RPC_TIMEOUT`（vllm 里已无消费者）与 `VLLM_ASCEND_ENABLE_PREFETCH_MLP`（两个树里都没有读者）
  疑似死变量，删之前想在远端确认一次（CANN/vendor 侧可能读）。
- A3 的 svc 端口（8034/8035/8036/8037）与 prof 端口重叠，两轮服务不能同时跑；要不要把端口分配也做成表。

## 2. 阶段 B：harness 层（verify.sh / run_tests.sh / tests/lib）

已修：

| 问题 | 修法 |
| --- | --- |
| `harness.json` 被 `/*.json` 忽略，远端不会有 | `.gitignore` 加 `!/harness.json`；文件本身已进版本库 |
| `verify_a5.sh` 两个失败账本（自己的 `EXIT_CODE` + common.sh 的 `HX_FAILED`），`hx_need_dir` 的 FAIL 不参与判定 → 可能报 PASS | 改为通用 `verify.sh`，统一用 common.sh 的 `hx_fail` |
| 每次 `run_tests.sh` 生成新的时间戳目录，诊断只看"最新一个" → `--skip-smoke` 时永远诊断失败、打包漏证据 | 一次验证一个 `tests/_out/<stamp>_<family>`，两个 stage 都传 `--out` |
| 诊断里 `[^\r]*` 在 BRE 里不是"回车"，`grep -o` 到第一个 `r` 就截断；另有一处把 `\r` 写成了真 CR 字节（git 把脚本判成二进制） | 判据改到 `harness.json`（`.*`），解析改成 Python 正则（`tests/lib/diagnose.py`），不再依赖外部 grep |
| `hx_service_up` 忽略 `hx_service_down` 的返回值，旧服务还在时新服务起不来却"就绪" | 停不干净直接 FAIL |
| 起服务用 `setsid ... &` 丢掉 PID，清理只靠 `pkill -f -- "--port N"`（匹配不到 mp worker） | 记 PID，停服时按进程组 kill，pkill 仅兜底 |
| 就绪轮询不看子进程死没死，启动即崩也要等满 30 分钟 | 子进程退出立即 FAIL 并打印日志尾部 |
| `setsid` 不存在时静默失效 | 没有 `setsid` 退回 `nohup`；`smoke/s01` 检查 curl/pkill/python3/setsid |
| `sync_tree.sh`：dirty 检查在版本检查之前（树已经对了也会因为一个未跟踪文件整体失败）；fetch 失败时退回本地同名分支还报"已在 <branch>"；`CP_BALANCE_AUTO_CHECKOUT=0` 说要"只报告"实际直接失败 | 先比版本、fetch 失败不再退回本地分支、角色清单为空时明确失败 |
| 打包 `[ -s "$TAR" ]` 永远为真（空 tar 也有 10KB） | 改成 `tar tf \| grep -q .`；产物名按 family 区分 |
| family 白名单/默认值、chips 映射、超时阈值、目录名等散在脚本里 | `harness.json families/limits/expect`；`hx.py limit/families/stages/stage-run` 读 |
| 静态测试超时预算（1800s）小于矩阵的 `ready_timeout`（2400s） | `limits.ready_tries=480`（2400s），`limits.ready_timeout_s` 供驱动默认值 |
| `a5|a3` 白名单在 runner 里重复实现 | runner 的 `--family` 校验读 `harness.json families` |

结构调整：

- `verify.sh`（新，family 无关）：前置检查 → `harness.json verify.stages` 逐条执行（`run` 命令或 `builtin: diagnose/pack`）
  → 诊断 → 打包。`verify_a5.sh` 变成 3 行兼容入口。
- `tests/lib/diagnose.py`（新）：报告与判据的唯一实现（证据串、已知失败表都从 `harness.json verify.diagnose` 读），
  退出码 0=有 zigzag 证据 / 1=无证据 / 3=命中已知失败。
- `tests/lib/trees.py` 改读 `harness.json`（`tests/lib/trees.json` 已删）。

待定：

- `run_tests.sh --list` 忽略 `--from`；`--list/--dry-run` 仍会建产物目录。
- `hx_resolve` 分不清"角色存在但配置是 `-`"和"角色不存在"（都是 rc=1），README 说前者 SKIP。
- `HX_PY` 未加引号，路径带空格会坏。

## 3. 阶段 C：测试与驱动（tests/**、accuracy/*.py、perf/*.py）

已修（都是"假 PASS / 假 FAIL"类）：

| 问题 | 修法 |
| --- | --- |
| `compare_first_token.py`：两侧空结果 → `first-token match: 0/0` + `RESULT: PASS`；`first_token=None` 也算相同 | 空数据 FAIL；任一侧 `first_token=None` 记为不一致 |
| `run_matrix.py`：`static_check` 的返回码被丢掉，静态门控失败仍 `RESULT: PASS` | 计入 verdict |
| `run_matrix.py`：起服务前不查端口，残留服务会被当成本次"就绪"；指纹不一致只是 WARNING | 端口被占直接 FAIL；日志指纹不是 `CONFIG=<name>` 计入 verdict |
| `run_matrix.py`：Popen 段没有 try/finally，异常/中断会漏掉 8 卡服务 | try/finally 里 stop + 等端口释放 |
| `run_matrix.py`：只有 compare 结果参与判定，谁都拦不住"两边都没进 zigzag" | 矩阵配置新增 `require_log`：cp1 必须出现 zigzag 证据、cp0 不许出现 plan 行 |
| `profile_forward.py`：所有配置失败也返回 0（p10 的 rc 断言是死代码），且旧 `windows.json` 会冒充本轮产物 | 有失败返回 1；开跑前删 `windows.json`；ready-timeout/settle/requests 默认值取 `harness.json limits` |
| perf 测试用"配置名"当产物目录（A3 配置里写的是绝对路径，会假 FAIL/假 SKIP） | 新增 `hx_profdir`，p10/p20/p21/p23 都按 `profiler.dir` 定位 |
| `a30_slot_filter_ab`：`round2_verify.py` 的补丁锚点已随源码变化失效，`RuntimeError` → FAIL | 锚点找不到时降级为 SKIP 并提示"driver stale" |
| `check_branch.py`：`CONTINUOUS` 是 `logger.info_once`（每进程一次），只看窗口字节差会误判短 prompt；长 prompt token 不够 `min_tokens` 时报"没进 ZIGZAG" | CONTINUOUS 改看全日志；token 不够时输出 `RESULT: INCONCLUSIVE`（s11 记 SKIP）；证据串可配 |
| 三处驱动把 `/home/z30055003/...`、`/opt/its/...` 写死为默认 `--repo` | 默认取 `harness.json trees.cur` |
| `s06` 把 `ZigzagPlan fields=13` 写死在测试里 | 读 `harness.json expect.zigzag_plan_fields` |

待定（未改）：

- `tests/perf/p21_window_single_step`：`kernel_details.csv` 没有 `Step` 列，`kernel_steps` 恒为 None → 该测试恒 SKIP，
  多步窗口抓不到。要真判它得从 `step_trace_time.csv`/`api_statistic` 数 step，需拿一份真实 trace 定列名。
- `p24_collect`：只断言"有 tgz"，空包也算 PASS。
- `/v1/completions` 在 pinned vLLM 上忽略 `max_completion_tokens`（只有 `max_tokens`），
  fixed vllm 会退化成默认 16 token；`profile_forward.py` 没有回退分支。远端确认后统一改 `max_tokens`。
- `round2_verify.py` 的三个补丁锚点需要按当前 `attention/context_parallel/sfa_cp.py` + `attention/indexer.py` 重写。
- `run_matrix.py` 的 `COUNTS` 里 `branch=ZIGZAG` 永远不会出现、`path=native` 只在没进 zigzag 时打，容易误导（现由 `require_log` 承担判据）。
- `perf/profile_analyse.py` 的列选择（`Communication` vs `Communication(Not Overlapped)`、`type` vs `name`）、
  `profile_order.py --rank rank0` 关闭窗口自动选择、`profile_l6.sh` 的 `|| true` 吞错。
- 驱动层 4 套服务起停/fingerprint/questions 解析重复，可合并成 1 个 helper（本轮没动，避免大改驱动）。

## 4. 阶段 D：运行期契约（configs ↔ vllm-ascend 代码）

### 4.1 已定案并修掉：dp=1 要能开 DSA-CP

结论（按你的判断查代码确认，文档不作为依据）：**dp=1 + DSA-CP 是成立配置，之前是 main 线的门写窄了。**

代码证据链：

- 现状 `vllm_ascend/ascend_config.py:664` 把 `enable_dsa_cp` AND 上 `parallel_config.use_sequence_parallel_moe`；
  上游 `vllm/config/parallel.py:711-726` 的 `use_sequence_parallel_moe` 要求
  `all2all_backend∈(...) 且 enable_expert_parallel 且 tp>1 且 data_parallel_size>1` —— 这是"EP 下避免重复算 MoE"
  的优化开关，不是 DSA-CP 的前提。
- git 历史：`cee6c1bc6`（#15549）把这一行从 `self.enable_dsa_cp = self.enable_dsa_cp and has_indexer`
  改成现在这样；提交说明自己写的是"dsacp 以前依赖 flashcomm"，即 dp=1 本来是可用的。
- DSA-CP 自己负责序列切分：`attention/context_parallel/sfa_cp.py:618-639` 在 attention 内部
  pad 到 `num_tokens_pad` 后按 rank 取连续切片；输出在 `o_proj.reduce_results` 为真
  （= 模型没做序列并行，也就是 dp=1）时用 `tp_group.all_gather` 还原成 replicated 状态
  （`sfa_cp.py:830-848`），为假时才把 reduce-scatter 交给 decoder 的 SP 路径。
- `platform.py:1182` 也是 `enable_sp(vllm_config) or enable_shared_expert_dp or enable_dsa_cp` 的 OR 关系，
  说明 DSA-CP 本就按"可以独立于 SP"来对待。

处理（两个树都改了，改动在各自的 git 里）：

- 被测树 `vllm-ascend`（分支 cp_balance，commit `1b638aa2a`）：
  `enable_dsa_cp = enable_dsa_cp and has_indexer`；没有 SP-MoE 时打 `info_once`
  "using the native DSA-CP sharding path"（原来那句 "Disabling DSA-CP" 的 warning 去掉）。
- 参照树 `vllm-ascend-base`：同样的一行改动提交在 fork 的 `base-dp1` 分支（commit `1bf45408f`），
  `harness.json trees.base.ref=base-dp1`，verify.sh 自动对齐到它（`main` 保持与上游一致）。
  这样参照树也能真的跑 DSA-CP，B 等价性才比较的是"原版 DSA-CP vs cp_balance 关"，
  而不是"DSA-CP 关 vs DSA-CP 关"。参照树被 `verify.sh` 对齐（`reset --hard origin/main`）之后，
  补丁会重新打上，不会丢。
- 如果更希望参照树保持"原版 main 不动"，把补丁前提到你 fork 的 main 上，然后删掉
  `trees.base.ref` 指回 `main` 即可（两种方式都行，别两边同时做）。`trees.<角色>.patches` 机制保留但当前没人用。

顺带补的日志：`sfa_cp.py` 在 zigzag plan 成功后补 `[CP_BALANCE][branch] rank=%d branch=ZIGZAG reason=-`
（DEBUG 门控，与资格门拒绝时的 CONTINUOUS 行对称），见 §4.2。

远端仍需确认的事实（改完就跑得到）：服务日志里不再出现
"Disabling DSA-CP"，而是出现 "DSA-CP is enabled without sequence-parallel MoE"，
且长 prompt 至少有一行 `branch=ZIGZAG`/`[CP_BALANCE][plan]`、cp0 一行都没有。

### 4.1c 分阶段核对：use_sequence_parallel_moe 为什么要求 dp>1

结论：dp>1 不是为了 DSA-CP 的正确性，而是这个属性服务的目标部署形态（WideEP：DP>1 + EP 的 MoE + TP 的 attention）自带的条件；
dp=1 时它只是不生效，不代表 DSA-CP 不能跑。分阶段证据：

阶段 1 定义（vllm/config/parallel.py:699-726）：条件是 all2all_backend 在许可集合内 且 EP 且 tp>1 且 dp>1。
属性上面的注释写明动机：attention 出口 o_proj 的 all_reduce 让输入在 TP 组内是复制的，如果 MoE 是 EP 的，
复制的 token 会造成重复计算与重复通信，所以让进入 experts 的输入变成 sequence-parallel。

阶段 2 历史：这个属性由 a5354b3ed2（[Bugfix][WideEP] Apply TP Attn + EP MoE fix to other models，PR #24982）引入，
面向的就是 WideEP（DP 组里做专家并行）；同一提交配套改了 device_communicators/all2all.py 等通信实现。
也就是说它的设计场景从一开始就是 DP>1。

阶段 3 与 dp 绑定的其它证据：parallel.py:726-737 use_all2all = dp>1 or use_sequence_parallel_moe or (EP 且 pcp>1)；
use_batched_dp_moe 直接要求 dp>1 且 backend 是 deepep_low_latency/nixl_ep；
许可集合里的 deepep_/mori_/nixl 都是按 DP 组做专家分发的后端。也就是说“dp>1”里既有“优化意义”也有“这些后端的硬要求”，
两者合在了同一个属性里。

阶段 4 对照我们自己的部署：tp16/dp1 + EP + backend=allgather_reducescatter。
allgather_reducescatter 是 all_ranks 的 allgather/reduce-scatter 后端，不需要 DP 组；
dp=1 时属性为 False，模型侧走未切分的（replicated）路径——而 DSA-CP 移植版本来就是按这个状态写的：
attention 自己 pad+切 rank-local 行，输出在 o_proj.reduce_results 为真时用 tp all_gather 还原（4.1b 修的就是 zigzag 下的这一处）。

阶段 5 判断：把该属性当 DSA-CP 的前置属于“拿另一个部署形态的优化开关当正确性前提”（#15549 引入），
改成只判 has_indexer 是恢复原意；dp>1 时 SP 仍会被自动打开，不受影响。
（如果上游以后要收紧，正确做法是按 backend 区分：deepep/mori/nixl 这类要 DP，allgather_reducescatter 不需要。）

### 4.1b 已修：zigzag 的 o_proj 出口（dp=1 放开 DSA-CP 之后才可达的 B3）

dp=1 一旦能开 DSA-CP，zigzag 就真的会被选中，而移植时留下的 o_proj 出口回归会静默算错：

- 模型边界 patch/worker/patch_deepseek_v2.py:365/379 把 hidden_states/positions 切成 rank-local 的
  [prev, next] 块（= L 行），层循环结束后由它 zigzag_gather_hidden_states_and_aux 做 all_gather + 反排列回自然序。
- 但 attention/context_parallel/sfa_cp.py 的 _finalize_o_proj 在 gather_full_o_proj=True
  （prefill + 全权重 o_proj）且 o_proj.reduce_results=True（= dp=1，没有 SP）时，
  又做了一次 tp_group.all_gather(local_output) 然后 output[...] = full_output[:L]：
  按 rank 连接后的前 L 行是 rank0 的数据，于是每个 rank 都写 rank0 的行，形状检查还过得去。
- 老线语义（cp_balance_v0.26.0rc:attention/sfa_v1.py:1530-1537）是
  output[...] = 全权重 o_proj(attn_output) 直接返回（rank-local 行），由模型边界收尾。

已改（被测树 commit 6ca45f53e）：zigzag_active() 时把本 rank 的投影结果直接写回 output 并返回，
不参与连续切片的 all_gather；另外 zigzag 若落到 all_to_all 分支直接 raise，绝不静默算错。
参照树不需要这个补丁（它没有 VLLM_ASCEND_CP_BALANCE，不会走 zigzag）。

远端建议顺序（先便宜后贵）：

1. cp_balance=0、DEBUG 开着起服务发长 prompt：确认日志是
   DSA-CP is enabled without sequence-parallel MoE + [CP_BALANCE][branch] branch=CONTINUOUS，
   且 cp0 与 base 的 40 条首 token 一致（这一段是连续切片路径，本来就是对的）。
2. 再跑 cp_balance=1（zigzag）：确认出现 branch=ZIGZAG/[CP_BALANCE][plan]，且 C 矩阵首 token 一致；
   不一致就带着 [CP_BALANCE][plan] 附近的日志回传。

### 4.2 证据契约：`branch=ZIGZAG` 不存在

- 代码只打 `branch=CONTINUOUS reason=...`（`sfa_cp.py:489-495`，`logger.info_once`）与 `[CP_BALANCE][plan]`（`sfa_cp.py:532-534`）。
- 文档（`docs/cp_balance_main_port.md:124`）、`accuracy/check_branch.py` 的说明、`verify_a5.sh` 的判据都按
  `[CP_BALANCE][branch] ... branch=ZIGZAG/CONTINUOUS` 写的。
- 处理：harness 侧证据串改为可配（默认 `branch=ZIGZAG` 或 `[CP_BALANCE][plan]`），诊断/矩阵门/s11 都按这个来；
  建议代码侧在 plan 成功后补一行 `[CP_BALANCE][branch] rank=%d branch=ZIGZAG`（纯日志、DEBUG 门控），
  这样 runbook 里的 `grep -c "branch=ZIGZAG"` 也成立。

### 4.3 已核对一致的项

- `additional_config` 五个键都是 `AscendConfig` 的真实字段（`extra="forbid"`，拼错会直接报错）；`enable_sparse_*_c8`
  在 GLM-5.2 上不会被静默丢弃（`model_uses_sfa_sparse()=True`）。
- `VLLM_USE_V2_MODEL_RUNNER=0` 是 load-bearing：GLM 的模型类在 V2 白名单里，而 zigzag 门拒绝 V2。
- A3 权重/参数：`max_position_embeddings=1048576 > 135000`、`num_hidden_layers=78` 与 `indexer_types` 一致、
  `num_nextn_predict_layers=1 == num_speculative_tokens`、`qk_rope_head_dim=64`（不触发 SFA c8 的 NotImplementedError）、
  `served_model_name` 含 `glm-52`（questions.json 里的模型名）。
- 两个静态门控在本地树上 PASS：`check_cp_balance_fields.py`（ZigzagPlan fields=13）、`check_b_path.py`（8/8）。

### 4.3b 放开 dp=1 之后仍然存在的其它风险（本轮没改）

- MC2/FUSED_MC2 的 MoE prepare 把输入当整批并 `tensor_split(...)[tp_rank]`（ops/fused_moe/prepare_finalize.py:252-311、
  ascend_forward_context.py:353-362），而 zigzag 下每 rank 只有 L 行 → 若该配置选到 MC2，rank>0 会读到 padding。
  诊断报告现在会一起抓 moe_comm_type/MC2 行，远端先看它选的是哪条 MoE 路径；真要 long-run 得在 zigzag 下禁 MC2。
- zigzag 回退路径只清理带 dsa_cp_context 的元数据，indexer 的元数据没清（ascend_forward_context.py:118-174）→
  draft/V2/dp>1 的回退可能出现 SFA 连续切片 + indexer zigzag 的单侧布局。
- zigzag 资格门没看 PCP，indexer 在 PCP 下又跳过 zigzag → 也是单侧布局（H2）。
- 行为差异（与老的 dp1+FlashComm1 基线比，不是 bug）：enable_dsa_cp 打开后
  `_pad_for_sequence_parallelism` 会把每次 forward 补到 tp 对齐（worker/model_runner_v1.py:3159），
  platform.py:1182 也会把 cudagraph capture size 改成 tp 对齐（我们的配置都是 --enforce-eager，这条用不上）；
  集合通信次数与老线不可直接对比。

### 4.3c 已修：MTP draft 的 for_draft 接上了（定案 5）

llm_base_proposer 在 MTP draft 的 metadata 构建处本来就传了 for_draft=True（注释写着：不要让
cp_balance 的资格门按 target batch 的 plan 去规划 draft 张量），但 sfa_v1.build() 把它吞进 **kwargs 里，
只往下传 draft_index=None，于是资格门的 speculative 永远 False——那条守卫是死的。

已改（被测树 commit 99d03e477），把 for_draft 从入口透传到底：

- sfa_v1.build() -> _build() -> _prepare_parallel_metadata()（base 与 CP 覆写都加了参数）->
  _prepare_zigzag_layout() -> zigzag_gate_reason()；
- indexer 侧 builder 同样透传（它是独立的 attention group builder，不传就会出现
  SFA 连续切片 + indexer zigzag 的单侧布局）；
- 判据改成 speculative = draft_index is not None or for_draft；draft_index 的原有语义没动
  （dspark 的 spec 数组仍按 draft_index-1 取）。

预期（远端 DEBUG 打开时）：draft 步出现 branch=CONTINUOUS reason=draft，主 prefill 仍出现 branch=ZIGZAG；
诊断报告的 branch 段现在能直接看出 draft 有没有被挡住。

### 4.3d for_draft 的复核（第三方考证）与补的两个边角

结论：4.3c 的改法与老线完全一致——cp_balance_v0.26.0rc 的 sfa_v1.py:474 就是 for_draft = bool(kwargs.pop(for_draft, False))，
:510 的 _build(..., for_draft=False)，:617 的 speculative = draft_index is not None or for_draft；
llm_base_proposer.py:2131-2140 也传同一个参数。新线 port 时只搬了 proposer 的调用，中间透传丢了，所以那条守卫是死的。

补的两个边角（被测树 commit b64c9569b）：

- PCP+DSA-CP 的 builder（AscendSFAPCPDCPMetadataBuilder.build）同样透传 for_draft（今天这条 MRO 没有 zigzag 门，属同坑预防）；
- indexer 的资格门拒绝时补 [CP_BALANCE][branch] ... reason=<门> site=indexer 一行：indexer metadata 没有 dsa_cp_context，
  前向兜底回滚不到它，SFA 连续 + indexer 排列的单侧布局只能靠这行日志才看得见。

明确的边界（不改，知道即可）：

- build 期的显式信号是唯一可靠的门：v2_model_runner / dp>1 / dcp_replicated / o_proj_not_full / state 这些对 prefill 步的 draft 挡不住，
  query_len<2*cp_size、actual<min_tokens、not_all_prefilling 只是巧合式阻挡；
- MTP 的 graph capture 路径（sfa_v1.build_for_graph_capture）仍不带标记：我们的配置都是 --enforce-eager，不触发；
- 既有隐患（老线与 port 都有）：proposer 对每个非 dspark draft group 都传 for_draft=True，而部分 Ascend builder
  （MLA/GQA/310P 那几个）没有 **kwargs → 用 eagle3/MLA 做 draft 时会 TypeError；GLM-5.2 的 MTP 走 SFA builder，不受影响。

### 4.4 待远端确认（可能影响成败）

- 容器里的 vllm 是否就是 pinned `84030bbe3d`（4.1 的前提）。
- 长 prompt 是否够 `min_tokens=2048`：`[check] long: tokens=N (min_tokens=2048)`；不够会被设计性拒绝（现在记 INCONCLUSIVE 而不是 FAIL）。
- 短 prompt 是否是第一个走到 DSA-CP builder 的 batch（`info_once` 的 CONTINUOUS 行只在进程内打一次）。
- A5 权重目录 `config.json`/`quant_model_description.json` 是否齐全；`--max-model-len 135000` + `--max-num-seqs 500`
  在 8 卡上的 KV 是否放得下（看启动日志的 KV cache 行）。
- MTP 的 draft 步：`sfa_v1.py:470-480` 吞掉 `for_draft`，`zigzag_cp.py` 的 draft 门实际不会触发（现在靠 draft 的
  query_len 很小才没进 zigzag）。要不要把 `for_draft` 接上，需要你定。
- `_profile_l6_*`：只加载 6 层时 LI-C8 的层过滤匹配不到任何层，`enable_sparse_li_c8` 在 L6 跑里实际是关的。
- A5 的 `VLLM_USE_FASTOKENS=1`、auto 识别出的网卡是否就是 RoCE 网卡。

## 5. 这次改动之后，远端第一条命令

```bash
# 前提：两个仓都 push 到 origin 之后远端才拉得到
#   cp_balance  (harness)   -> origin/main
#   vllm-ascend (被测树)    -> origin/cp_balance（否则 verify.sh 会把本地 commit reset 掉）
cd <harness 目录>
git pull                      # harness.json / verify.sh / docs/ 一起更新
export CP_BALANCE_LOCAL_IP=<本机 IP> CP_BALANCE_NIC_NAME=<网卡>
bash tests/run_tests.sh --tag fast          # 秒级自检（s03 会检查权重/树是否就位）
bash verify.sh --family a5                  # 前置 -> 静态 -> 冒烟 -> 诊断 -> 打包
# 诊断只看已有证据（不起服务）：
bash verify.sh --family a5 --diag-only
```

回传：`verify_a5_*.tar.gz`（里面是报告 + 这一轮的 `tests/_out/<stamp>_a5/`，含服务日志与 status.tsv）。
若诊断是 `VERDICT=KNOWN ... Disabling DSA-CP`，先把 §4.1 的远端确认命令结果发回来，再决定改门的方式。

## 6. 本地做过的静态验证（说明哪些结论不依赖远端）

- 36 份 `configs/*.json`：改动前后 `serve_config.py --dry-run --print-env` 对比，vLLM 参数集合逐条相同，
  指纹一致（只多了 `ARGS` 与更全的 env 打印）。
- `bash -n` 全部 shell、`py_compile` 全部 python 通过；`tests/run_tests.sh --list/--dry-run` 正常。
- `smoke/s02_config_resolve`（a3 family，24 条配置）PASS；`s06`/`s07` 两个静态门控 PASS。
- `verify.sh --diag-only` 用合成日志跑通三种结论：有 `[CP_BALANCE][plan]` → OK；有 `Disabling DSA-CP` → KNOWN（rc=1）；
  什么都没有 → MISSING。
- 未验证：任何起服务、NPU、profiler、精度/性能路径。

## 7. 5 项待定项的定案（2026-09-21）

1. A5 profile 的 deterministic:false —— 不动。性能采集与 A5 服务配置不是同一款设置，结论里要写明（对比数字时别当成服务同款）。
2. VLLM_RPC_TIMEOUT / VLLM_ASCEND_ENABLE_PREFETCH_MLP —— 已删（configs/_base.json 少两个 env；指纹里的 ARGS 会变，属预期）。
3. 参照树的门修 —— 提交在 fork 的 `base-dp1` 分支（`trees.base.ref=base-dp1`），fork 的 `main` 保持与上游一致；两条路只走一条。
4. round2_verify 的补丁锚点 —— 不修。tests/accuracy/a30_slot_filter_ab 保持 SKIP（提示 driver stale），可选 A/B 不再纳入验收；a10/a20 的 C/B/噪声地板不受影响。
5. MTP draft 的 for_draft —— 接上，见 4.3c。

## 8. 追加发现（2026-09-21 晚）：dp=1 下 zigzag 还有第三处结构性问题（MoE 布局）

来源：dp>1 的分阶段考证（阶段性结论见 4.1c）+ 随后的代码核对。

事实链：

1. 属性为 False（dp=1）时，vLLM 的 MoE 走 no-DP-EP：没有 dispatch/combine，每 rank 用自己的专家对**输入的所有行**
   算 partial，最后由 moe_runner._maybe_reduce_final_output 做一次 all_reduce（条件 tp>1 or ep>1）。
   这套只有当输入是**全量（replicated）行**时才正确。
2. zigzag 的布局恰恰相反：模型边界 patch_deepseek_v2.py 在进入层循环前把 hidden_states 切成 rank-local 的
   [prev,next] 块，出口才 all_gather + 反排列。也就是说层循环里的 MLP/MoE 看到的是**切片行**。
3. 移植版为 zigzag 写的 MoE/MLP 适配只存在于 SP 路径里：ops/linear_op.py:192 的 MLPRowParallelOp
   （fine-grained mlp tp）与 ops/fused_moe/shared_experts.py:246（注释写明 “SP-only path”）。
   dp=1 下这些都不生效（MLPRowParallelOp 只在 mlp_tp/SP 配置里被替换，shared_experts 的 parallel_mode 不是
   SEQUENCE_PARALLEL_ONLY）。
4. 结论：连续切片路径没问题（attention 出口 _finalize_o_proj 在 o_proj.reduce_results=True 时 all_gather 回全量行，
   MoE 拿到的是全量行）；但 **zigzag 下 MoE 会拿到切片行，而 MoE 侧要么走需要全量行的 no-DP-EP，要么走按整批
   tensor_split 的 MC2（同样假设整批）** → 结果不对。这不是 4.1b 那处能覆盖的，属于布局层面的不匹配。

三条出路（需要定）：

- (a) dp=1 先不开 zigzag：加一条显式门（not use_sequence_parallel_moe → 资格门拒绝，reason=no_sp），
  只验收连续切片 DSA-CP；代价是 cp_balance 在这批机器上暂时不可用，但不会算出错的数。
- (b) 让 dp=1 也开 SP（上游 #47070 的 “SP without DP”，HEAD 里 forward_context 的 dp=1 兜底与 AgRs
   [num_local]*world_size 短路都还在），模型侧 SP 分支一开，MoE 的 chunk/all_gather 与 reduce_scatter 就位，
  zigzag 布局自洽；风险：上游因 DSv3.2+MTP 精度问题 (#47902/#48849) 把它挡回去了，且移植注释明说
  “如果那道门放开，本文件要重新打层补丁”。
- (c) 自己补 zigzag 的层内边界：在 MLP/MoE 前 all_gather、之后再切回 rank-local（每层两趟通信），
  或把 3 里的 row-parallel 路径在 zigzag 下强制生效；工作量大、需要远端逐层验证。

建议顺序：先按 (a) 跑连续切片的验收（cp0 vs base 的 B 等价性 + 噪声地板），把“DSA-CP 在 dp=1 成立”钉死；
再用一次 cp1 起停确认 C 的行为（预期 FAIL，用来验证上面第 4 条的判断是否成立，日志里的 moe_comm_type 会直接告诉我们
走的是哪条 MoE 路径）；之后在 (b)/(c) 之间选一条投入。
