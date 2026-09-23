# 交接：cp_balance on vllm-ascend main（GLM-5.2，A5/A3）

写给下一位接手的人/AI。目的：不用翻历史，按本文就能继续把 cp_balance 在 dp=1 上做完。
所有"当前状态"以 `docs/scripts_review.md` 为准；验证步骤见 `docs/verify_steps.md`；文档索引 `docs/README.md`。

## 1. 现状（2026-09-23）

| 仓 | 分支 | tip | 说明 |
| --- | --- | --- | --- |
| harness（本仓） | `main` | `a2fc9e6`（本次改动） | 启动/配置/测试脚手架 + 工程文档 |
| vllm-ascend（被测） | `cp_balance` | `c9f6b455c` | 移植版 + 本地修复 + 2026-09-23 zigzag 修复（§3b/§4.1） |
| vllm-ascend-base（参照） | `base-dp1` | `1bf45408f` | = main aff1b74b6 + dp=1 放开门一行；harness 自动对齐到这个分支 |
| vllm | 固定 commit | `84030bbe3d` | `harness.json trees.vllm`，`required=false` |

- 机器：默认 A5（8 卡，`GLM-5.2-w4a4c8-mxfp4`，TP8）；A3（16 卡，`GLM-5.2-W4A8C8`，TP16）用 `--family a3`。
- 已跑过远端（2026-09-22，`log/acc_0922_222922`）：B 等价性 PASS（40/40），C 验收 FAIL（长 prompt 0/20，全是换行）——
  根因是"zigzag 把行布局搬到模型边界"，2026-09-23 已修（见 §3b 与 `docs/dp1_zigzag_acc_fix.md`），**修复本身还没上机**。
- dp=1 下 DSA-CP 实际走什么分支、精度怎么测、怎么复跑：全部写在 `docs/dp1_zigzag_acc_fix.md`（先看它）。

## 2. 怎么跑（远端，A5 默认）

```bash
cd <harness>            # 例如 /home/z30055003/cp_balance
git pull                # 只拉 harness；两棵代码树由 verify.sh 自动 clone/对齐/打补丁
export CP_BALANCE_LOCAL_IP=<本机 IP> CP_BALANCE_NIC_NAME=<网卡>   # 不设也行（配置默认 auto）
bash verify.sh                  # = --family a5：前置 -> 静态 -> 冒烟(会拉起模型) -> 诊断 -> 打包
bash verify.sh --live-log       # 同上，并把测试与模型服务日志实时打屏
bash verify.sh --skip-smoke     # 不起服务（只做前置/静态）
bash tests/run_tests.sh --list  # 看全部测试；--only/--tag/--from/--skip 逐层跑
```

判据（见 `docs/verify_steps.md`）：

1. 启动日志应是 `DSA-CP is enabled without sequence-parallel MoE (data_parallel_size=1)`，**不是** `Disabling DSA-CP`；
2. 长 prompt 要出现 zigzag 证据（`branch=ZIGZAG` 或 `[CP_BALANCE][plan]`），短 prompt 只有 `CONTINUOUS`；draft 步 `reason=draft`；
3. 诊断 `[diagnose] VERDICT=OK`；证据包 `verify_a5_*.tar.gz`。

注意：`verify.sh` 会把两棵树 `reset --hard` 到远端 tip（本地未 push 的改动会丢）；`tests/run_tests.sh` 本身不对齐树。

## 3. 代码改了什么（被测树 5 个提交）

1. `1b638aa2a` dp=1 也能开 DSA-CP：`ascend_config.py` 的 `enable_dsa_cp = enable_dsa_cp and has_indexer`（原来 AND 上 `use_sequence_parallel_moe`，
   而上游该属性要求 `data_parallel_size > 1`）；同时 zigzag 成功时补 `[CP_BALANCE][branch] branch=ZIGZAG reason=-`。参照树同一行改动走补丁。
2. `6ca45f53e` zigzag 的 o_proj 出口：`sfa_cp.py::_finalize_o_proj` 在 `zigzag_active()` 时写回 rank-local 行（原来按 rank 连接后取前 L 行 = 每 rank 拿到 rank0 数据）；
   落到 `all_to_all` 分支则直接报错。
3. `99d03e477` 接上 MTP draft 的 `for_draft`：`sfa_v1.build()` -> `_build` -> `_prepare_parallel_metadata` -> `_prepare_zigzag_layout` -> `zigzag_gate_reason`，
   判据 `speculative = draft_index is not None or for_draft`（老线本来就是这样，port 时丢了中间透传）。
4. `b64c9569b` 边角：PCP builder 同样透传；indexer 侧被门拒绝时补 `[CP_BALANCE][branch] ... site=indexer` 日志（indexer metadata 没有 `dsa_cp_context`，前向兜底回滚不到它）。
5. `c9f6b455c` + 后续（2026-09-23，**修精度**）zigzag 收进 attention 内部选行：模型主流不再被切成 rank-local 行
   （`_prepare_native_hidden_states` 用本层 `zigzag_index` 选行、`_finalize_o_proj` 用本层 `inv_gather_index` 排回自然序），
   同时删掉模型边界 shard、SP-only 的 owner-independent 归约、`reduce_mode`/`EMBED_LOCAL` 两个开关、
   MoE `mc2_mask` 重排，以及全局 `zigzag_active()`/`_EXTRA_CTX.zigzag_cp_*`（布局唯一来源 = 每层元数据）。
   详见 `docs/dp1_zigzag_acc_fix.md`。

harness 侧（本仓，主要提交）：`e9e943a` 配置化 + 一批假 PASS/假 FAIL 修复、`7da9a55` dp=1 门/参照树补丁、`ad866e5` 诊断 KNOWN 规则、`dc8bff6` 文档入仓 + 默认 A5、`522fe76` 前置顺序修正。

## 4. 未决问题（按优先级）

### 4.1 已解决（2026-09-23）：dp=1 下 zigzag 的模型级行布局

- 2026-09-22 远端实测证实：zigzag 在模型边界把 hidden_states 切成 rank-local 行，而 dp=1 的 MoE/MLP 集合通信是
  element-wise 的（需要全量行）→ 长 prompt 首 token 全空。
- 定案：zigzag 收回 attention 内部（选行 + 出口反排列），模型主流恢复 replicated 全量行；三条老出路
  (a) 不开 zigzag / (b) dp=1 开 SP / (c) 层内补 gather 都不采用。
- 细节与复跑命令见 `docs/dp1_zigzag_acc_fix.md`；静态门控见 `docs/scripts_review.md` §8-§9。

### 4.2 其它已记录风险

- ~~MC2/FUSED_MC2 的 MoE prepare 按整批 `tensor_split`~~：2026-09-23 后 MoE 拿到的是 replicated 全量行，这条风险随行布局修复消失。
- zigzag 回退只清带 `dsa_cp_context` 的元数据，indexer 独立建的元数据没清 → draft/V2/dp>1 回退时可能出现
  SFA 连续切片 + indexer zigzag 的单侧布局；PCP 下 indexer 跳过 zigzag（H2）。两条都还没在远端复现过。
- `p21_window_single_step` 恒 SKIP（trace 没有 Step 列）、`p24_collect` 只断言"有 tgz"、`round2_verify` 锚点失效（a30 已定案 SKIP）。
- `/v1/completions` 在 pinned vLLM 上忽略 `max_completion_tokens`（只有 `max_tokens`），远端确认后统一改调用。
- 上游要把 DSA-CP 迁到 PCP（`ascend_config.py` 有 deprecation 警告），PCP 与 DSA-CP 互斥。

## 5. 已定案（别推翻，除非有新证据）

1. A5 profile 的 `deterministic=false` 不动（结论里注明"性能数字不是服务同款设置"）。
2. `VLLM_RPC_TIMEOUT` / `VLLM_ASCEND_ENABLE_PREFETCH_MLP` 已删（两个树里都没有消费者）。
3. 参照树的 dp=1 门修提交在 fork 的 `base-dp1` 分支（`harness.json trees.base.ref=base-dp1`），fork 的 main 保持与上游一致；要跟上游同步就把 main 快进后 rebase 这个分支。`trees.<角色>.patches` 机制保留但当前没人用。
4. `round2_verify` 的补丁锚点不修，a30 永久 SKIP。
5. MTP draft 的 `for_draft` 已接上（见 §3.3）。
6. **zigzag 不得出现在模型主流里，且布局只能由每层 DSA-CP 元数据描述**（2026-09-23 定案）：dp=1 下 `use_sequence_parallel_moe=False`，
   层内 TP/EP 集合通信是 element-wise 的，任何"层间只有 rank-local 行"的布局都会算错。
   要做 zigzag 就在 attention 内部换工作窗口（选行 + 出口反排列）。
   `owner-independent 归约` / `reduce_mode` / `EMBED_LOCAL` 这类补偿手段不要重新引入。

## 6. 接手第一小时的建议动作

1. 先读 `docs/dp1_zigzag_acc_fix.md`（dp=1 的分支现状 + 精度方法 + 修复），
   再看 `docs/scripts_review.md`（§1-§4 是脚本/配置、§4.1c 是 dp>1 考证、§8-§9 是行布局问题与修复）与 `docs/verify_steps.md`；
2. 本地静态自检：`python3 serve_config.py default --dry-run --print-env`、`bash verify.sh --help`、`bash tests/run_tests.sh --list`；
3. 远端（需要人配合）：`git pull && bash verify.sh`，把 `verify_a5_*.tar.gz` 带回来；
4. 按判据确认 §3.1/§3.2 的日志；行布局相关的结论已经定案（§4.1/§5.6），不用再选 (a)/(b)/(c)；
5. 改完跑阶段 2 → 阶段 3（性能），更新 `docs/scripts_review.md` 的状态与本文件。

## 7. 约束

- 本地（Windows 开发机）**不跑服务**：服务/NPU/远端路径只能在远端验证，任何"能不能跑"的结论都要靠远端日志。
- 本机可以 push（github 凭据可用）；远端只 pull harness，两棵代码树由 `verify.sh` 自动对齐。
- 别手改参照树：harness 会 `reset --hard origin/base-dp1`；要改就改 `base-dp1` 分支并 push。
- 文档已入仓 `docs/`，改文档请改仓内副本。
