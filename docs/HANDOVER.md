# 交接：cp_balance on vllm-ascend main（GLM-5.2，A5/A3）

写给下一位接手的人/AI。目的：不用翻历史，按本文就能继续把 cp_balance 在 dp=1 上做完。
所有"当前状态"以 `docs/scripts_review.md` 为准；验证步骤见 `docs/verify_steps.md`；文档索引 `docs/README.md`。

## 1. 现状（2026-09-21，全部已 push）

| 仓 | 分支 | tip | 说明 |
| --- | --- | --- | --- |
| harness（本仓） | `main` | `522fe76` | 启动/配置/测试脚手架 + 工程文档 |
| vllm-ascend（被测） | `cp_balance` | `b64c9569b` | 移植版 + 本地 4 个修复（见 §3） |
| vllm-ascend-base（参照） | `main` | `aff1b74b6` + 补丁 | 补丁 = `patches/dsa_cp_dp1.patch`（dp=1 放开门），harness 自动打 |
| vllm | 固定 commit | `84030bbe3d` | `harness.json trees.vllm`，`required=false` |

- 机器：默认 A5（8 卡，`GLM-5.2-w4a4c8-mxfp4`，TP8）；A3（16 卡，`GLM-5.2-W4A8C8`，TP16）用 `--family a3`。
- 上游现状（2026-09-21 核对）：vllm-ascend main `d05255207`、vllm main `9a70c233cd`，**都还没修 dp>1 那道门**；
  上游文档 `context_parallel.md` 的 DSA-CP 示例却是 `--tensor-parallel-size <N>`（dp=1）——代码与文档矛盾，属上游 bug。
- 只做过静态验证：36 条配置 dry-run 参数集合一致、`bash -n`/`py_compile` 全过、`s02/s06/s07` 本地 PASS。
  **改完 dp=1 门之后还没在机器上跑过一次服务**（上一次远端失败是缺 `_build_info.py`，见 `log/field_error.log`）。

## 2. 怎么跑（远端，A5 默认）

```bash
cd <harness>            # 例如 /home/z30055003/cp_balance
git pull                # 只拉 harness；两棵代码树由 verify.sh 自动 clone/对齐/打补丁
export CP_BALANCE_LOCAL_IP=<本机 IP> CP_BALANCE_NIC_NAME=<网卡>   # 不设也行（配置默认 auto）
bash verify.sh                  # = --family a5：前置 -> 静态 -> 冒烟 -> 诊断 -> 打包
bash tests/run_tests.sh --list  # 看全部测试；--only/--tag/--from/--skip 逐层跑
```

判据（见 `docs/verify_steps.md`）：

1. 启动日志应是 `DSA-CP is enabled without sequence-parallel MoE (data_parallel_size=1)`，**不是** `Disabling DSA-CP`；
2. 长 prompt 要出现 zigzag 证据（`branch=ZIGZAG` 或 `[CP_BALANCE][plan]`），短 prompt 只有 `CONTINUOUS`；draft 步 `reason=draft`；
3. 诊断 `[diagnose] VERDICT=OK`；证据包 `verify_a5_*.tar.gz`。

注意：`verify.sh` 会把两棵树 `reset --hard` 到远端 tip（本地未 push 的改动会丢）；`tests/run_tests.sh` 本身不对齐树。

## 3. 代码改了什么（被测树 4 个提交）

1. `1b638aa2a` dp=1 也能开 DSA-CP：`ascend_config.py` 的 `enable_dsa_cp = enable_dsa_cp and has_indexer`（原来 AND 上 `use_sequence_parallel_moe`，
   而上游该属性要求 `data_parallel_size > 1`）；同时 zigzag 成功时补 `[CP_BALANCE][branch] branch=ZIGZAG reason=-`。参照树同一行改动走补丁。
2. `6ca45f53e` zigzag 的 o_proj 出口：`sfa_cp.py::_finalize_o_proj` 在 `zigzag_active()` 时写回 rank-local 行（原来按 rank 连接后取前 L 行 = 每 rank 拿到 rank0 数据）；
   落到 `all_to_all` 分支则直接报错。
3. `99d03e477` 接上 MTP draft 的 `for_draft`：`sfa_v1.build()` -> `_build` -> `_prepare_parallel_metadata` -> `_prepare_zigzag_layout` -> `zigzag_gate_reason`，
   判据 `speculative = draft_index is not None or for_draft`（老线本来就是这样，port 时丢了中间透传）。
4. `b64c9569b` 边角：PCP builder 同样透传；indexer 侧被门拒绝时补 `[CP_BALANCE][branch] ... site=indexer` 日志（indexer metadata 没有 `dsa_cp_context`，前向兜底回滚不到它）。

harness 侧（本仓，主要提交）：`e9e943a` 配置化 + 一批假 PASS/假 FAIL 修复、`7da9a55` dp=1 门/参照树补丁、`ad866e5` 诊断 KNOWN 规则、`dc8bff6` 文档入仓 + 默认 A5、`522fe76` 前置顺序修正。

## 4. 未决问题（按优先级）

### 4.1 【阻塞精度】dp=1 下 zigzag 的 MoE 布局（`scripts_review.md` §8）

- zigzag 在模型边界把 hidden_states 切成 rank-local 行；而 SP 关闭时（dp=1 就是）MoE 走 no-DP-EP：
  每 rank 用自己的专家对**输入的每一行**算 partial，最后做一次 TP all_reduce —— 这套需要**全量行**。
  移植版为 zigzag 写的 MoE/MLP 适配（`ops/linear_op.py::MLPRowParallelOp`、`ops/fused_moe/shared_experts.py`）只在 SP 路径里生效。
- 后果：`cp_balance=1`（zigzag）的数值很可能不对；连续切片（`cp_balance=0`）没问题。
- 三条出路：(a) dp=1 显式不开 zigzag（资格门加 `reason=no_sp`）；(b) 让 dp=1 也开 SP（上游 #47070 的 "SP without DP"，脚手架在 HEAD 还在，
  但上游因 DSv3.2+MTP 精度问题又挡回去了）；(c) 自己补层内边界（MoE 前 gather / 之后切回，或让 row-parallel 路径在 zigzag 下强制生效）。
- 建议：先按 `docs/verify_steps.md` 跑阶段 1（cp0 vs base）钉死 DSA-CP 成立，再跑阶段 2（C 矩阵）确认 zigzag 的真实行为，
  把 `[CP_BALANCE][plan]` 前后日志 + `moe_comm_type/MC2` 行一起带回，再定 (a)/(b)/(c)，别重复烧机时。

### 4.2 其它已记录风险

- MC2/FUSED_MC2 的 MoE prepare 按整批 `tensor_split`（`ops/fused_moe/prepare_finalize.py`），zigzag 下 rank>0 会读到 padding（诊断会抓 `moe_comm_type/MC2`）。
- zigzag 回退只清 `dsa_cp_context`，indexer metadata 没清 → 单侧布局；PCP 下 indexer 跳过 zigzag（H2）。
- `p21_window_single_step` 恒 SKIP（trace 没有 Step 列）、`p24_collect` 只断言"有 tgz"、`round2_verify` 锚点失效（a30 已定案 SKIP）。
- `/v1/completions` 在 pinned vLLM 上忽略 `max_completion_tokens`（只有 `max_tokens`），远端确认后统一改调用。
- 上游要把 DSA-CP 迁到 PCP（`ascend_config.py` 有 deprecation 警告），PCP 与 DSA-CP 互斥。

## 5. 已定案（别推翻，除非有新证据）

1. A5 profile 的 `deterministic=false` 不动（结论里注明"性能数字不是服务同款设置"）。
2. `VLLM_RPC_TIMEOUT` / `VLLM_ASCEND_ENABLE_PREFETCH_MLP` 已删（两个树里都没有消费者）。
3. 参照树补丁走 `harness.json trees.base.patches`，不改 fork 的 main。
4. `round2_verify` 的补丁锚点不修，a30 永久 SKIP。
5. MTP draft 的 `for_draft` 已接上（见 §3.3）。

## 6. 接手第一小时的建议动作

1. 读 `docs/scripts_review.md`（§1-§4 是脚本/配置、§4.1c 是 dp>1 考证、§8 是 MoE 布局）与 `docs/verify_steps.md`；
2. 本地静态自检：`python3 serve_config.py default --dry-run --print-env`、`bash verify.sh --help`、`bash tests/run_tests.sh --list`；
3. 远端（需要人配合）：`git pull && bash verify.sh`，把 `verify_a5_*.tar.gz` 带回来；
4. 按判据确认 §3.1/§3.2 的日志；再决定 §4.1 的 (a)/(b)/(c)；
5. 改完跑阶段 2 → 阶段 3（性能），更新 `docs/scripts_review.md` 的状态与本文件。

## 7. 约束

- 本地（Windows 开发机）**不跑服务**：服务/NPU/远端路径只能在远端验证，任何"能不能跑"的结论都要靠远端日志。
- 本机可以 push（github 凭据可用）；远端只 pull harness，两棵代码树由 `verify.sh` 自动对齐。
- 别手改参照树：`reset --hard` 后补丁由 harness 重打；要长期改就提到 fork 的 main（并删掉 `trees.base.patches`）。
- 文档已入仓 `docs/`，改文档请改仓内副本。
