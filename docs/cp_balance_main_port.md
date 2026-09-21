> 历史文档（2026-09 上旬的交接/评审期记录）：结论可能已过时，当前状态以 `docs/scripts_review.md` 为准。
>
> 更正（2026-09-21）：文中「先做 A-B1 远端取证」已经做完，结论与处理见 `scripts_review.md` §4.1 / §4.1c；
> 上游两个仓的最新 main 仍未修那道门（§4.1c）。

# cp_balance 移植到 vllm 主仓 main 线

## 1. 现状（refs）

| 代码树 | 分支 / commit | 说明 |
| --- | --- | --- |
| vllm-ascend | `cp_balance` = `b64c9569b` | 本次移植结果：upstream vllm-ascend main + zigzag 实现 + 本地 dp=1 门修/o_proj 出口/for_draft |
| vllm-ascend | `cp_balance_v0.26.0rc` = `23ff2c23c` | 老实现（releases/v0.26.0rc 线），原样保留 |
| vllm-ascend | `main`（本地） = `bb570591f` | 已不用；新基线是 `upstream/main` |
| vllm-ascend-base | `main` = `aff1b74b6` + `patches/dsa_cp_dp1.patch` | 对照树（原版 main + dp=1 门修；补丁由 harness 在树对齐后自动打，别手改这棵树） |
| vllm-ascend-base | `base_layer3` = `c7990e5e4` | 老的打点树，保留备用 |
| vllm | 分离头 `84030bbe3d` | vllm-ascend main 验证过的 vLLM commit |
| vllm | `origin/main` = `9a70c233cd` | vllm main 最新（仍要求 `use_sequence_parallel_moe` 带 dp>1；pin 保持 84030bbe3d） |
| upstream vllm-ascend | `main` = `d05255207` | 上游最新；dp>1 那道门仍未修，dp=1 放行是本地补丁（见 `scripts_review.md` §4.1c） |

三条远端分支都已推送：`origin/cp_balance`（force 更新）、`origin/cp_balance_v0.26.0rc`（新增）、
`vllm-ascend-base:origin/main`（快进）。

为什么抽到 `aff1b74b6`：upstream vllm-ascend main 的 `.github/vllm-main-verified.commit` = `84030bbe3d`，
即“main 分支对应的 vllm 主仓 main 版本”。老 cp_balance 在 v0.26.0rc 线上（对应 vllm `568afb3a13` = v0.26.0）。

## 2. 移植内容

新增：

| 文件 | 作用 |
| --- | --- |
| `vllm_ascend/layers/cp_zigzag.py` | 计划/资格判定/shard/gather 原语（逻辑照搬；gather 不再读已删除的 `forward_context.pad_size`） |
| `vllm_ascend/attention/context_parallel/zigzag_cp.py` | 一个 batch 的布局单一来源：资格门、`ZigzagPlan`、设备张量（zigzag/gather/逆置换索引、合并的 2B 长度表、翻倍 block table、重排 slot mapping） |

改动：

| 文件 | 改了什么 |
| --- | --- |
| `attention/context_parallel/sfa_cp.py` | `DSACPContext` 增加 zigzag 字段与连续切片 fallback；`_prepare_parallel_metadata` 选布局；`_update_parallel_slot_mapping` 重放置换；`AscendSFADSACPImpl` 用合并长度、top-k 用翻倍 block table、KV 全量 gather 用重排 slot 写回 |
| `attention/indexer.py` | indexer/LI 自己的 DSA-CP 元数据与 cache 写回用同一个计划 |
| `ascend_forward_context.py` | 逐 forward 的 `zigzag_cp_active`（draft / V2 / DP>1 否决）、fallback 还原、`mc2_mask` 重排、`zigzag_active()` |
| `patch/worker/patch_deepseek_v2.py` | 模型边界：embedding + positions 切片、出口 all-gather + 逆置换（runner 已不再负责） |
| `ops/linear_op.py`、`ops/fused_moe/shared_experts.py` | row-parallel 归约按 `zigzag_active()` 门控并换 owner 无关归约 |
| `ops/vocab_parallel_embedding.py` | 拆出 `_embed_partial`，新增 `forward_zigzag_local`（`EMBED_LOCAL` 实验开关） |
| `device/device_op.py` | indexer post-process 增加可选 `block_table` |
| `envs.py` | `VLLM_ASCEND_CP_BALANCE`、`_MIN_TOKENS`、`_REDUCE_MODE`、`_DEBUG`、`_EMBED_LOCAL` |
| `attention/utils.py`、`worker/model_runner_v1.py` | `is_prefilling_cpu` 主机侧副本（资格门不做 device→host 同步） |
| `spec_decode/llm_base_proposer.py` | draft 元数据显式排除在 zigzag 之外 |
| `tests/ut/ops/test_vocab_parallel_embedding.py` | `forward_zigzag_local` 的用例 |

## 3. 关键设计决定

1. **布局单一来源**：新 main 里 DSA-CP 的分片算了两遍（SFA `sfa_cp.py` 与 indexer `indexer.py`）。
   两边都调 `zigzag_cp.py` 的同一个 planner，避免“只改一边 → indexer cache 静默写错”。
2. **V2 model runner 不参与**：新 main 对 `GlmMoeDsaForCausalLM` 默认开 MRV2，而 zigzag 挂在 V1 的
   SFA/DSA-CP 元数据与 impl 上。`zigzag_ineligible_reason(v2_model_runner=...)` 在 V2 下直接否决，
   harness 两条线都显式 `VLLM_USE_V2_MODEL_RUNNER=0`，保证 B/C 比的是同一个 runner。
3. **出口 gather 反转**：新 main 删了 runner 的 `_all_gather_hidden_states*`，出口拼接在
   `patch_deepseek_v2.py::_patched_forward` 里。zigzag 的逆置换就放在同一处：一次 TP all-gather +
   `inv_gather_index`，再按非 zigzag 路径相同的行数截断。
4. **归约**：门控 `zigzag_active()` 后走 `distributed/utils.fixed_order_reduce_scatter`（默认 allreduce +
   切自己那块）。老树上的 `SequenceColumnParallelOp/SequenceRowParallelOp` 在 main 已被 FlashComm v1
   清理删掉，没有重新引入。
5. **没有一起搬**：base_layer3 的“3 层/减层权重过滤”和 SFA-5.3 `record_function` 打点属于 base 线
   （不在 `cp_balance` 的 15 文件差异里），本次不移植；老 `main` 上的 matplotlib/其它无关改动同理。

## 4. 本地做了什么验证 / 没做什么

已验证（纯静态，本机没有 NPU）：

- 全部改动文件 `python -m compileall` 通过；
- `pyflakes` 无新增告警（`worker/model_runner_v1.py:259` 的 `SchedulerOutput` 重定义是 main 里既有的）；
- `DSACPContext` 的构造关键字与 dataclass 字段一一对应（AST 校验）；
- 老实现完整保留在两个仓库的分支上（老分支可随时 diff 对照）。

没验证（必须在远端跑）：

- 任何运行时行为：KV/indexer 写回顺序、top-k 合并 2B metadata 的 kernel 语义、归约位一致性、
  出口 gather 行数、`ZigzagPlan` 的 max-flow 余数分配在多请求下的表现。

## 5. 远端怎么验（A3：16 卡，GLM-5.2-W4A8C8）

### 5.1 更新代码树

```bash
cd /opt/its/z30055003

# vllm：切到 vllm-ascend main 验证过的 commit，并按安装文档重装（这步不做后面 API 全不匹配）
git -C vllm fetch origin
git -C vllm checkout 84030bbe3d74d99bad477a3d2e37a973ccd8865c

# vllm-ascend：cp_balance 被 force 更新过，用 reset 而不是 pull
git -C vllm-ascend fetch origin
git -C vllm-ascend checkout cp_balance
git -C vllm-ascend reset --hard origin/cp_balance      # b64c9569b

# 对照树：交给 harness 对齐更省事（git clone/fetch + reset 到 origin/main + 自动打 patches/dsa_cp_dp1.patch）
git -C vllm-ascend-base fetch origin
git -C vllm-ascend-base checkout main
git -C vllm-ascend-base reset --hard origin/main       # aff1b74b6（补丁由 harness 在 verify.sh 里补打）

git -C vllm-ascend log -1 --oneline
git -C vllm-ascend-base log -1 --oneline
```

`VLLM_USE_V2_MODEL_RUNNER=0` 现在集中在 `configs/_base.json` 的 `env` 里（族配置继承它）：
新 main 对 GLM-5.2 默认开 MRV2，而 zigzag 只在 V1 路径上生效；两条线必须同 runner。

### 5.2 先跑不起服务的检查

```bash
cd /opt/its/z30055003/cp_balance
export CP_BALANCE_FAMILY=a3 CP_BALANCE_LOCAL_IP=7.246.78.75 CP_BALANCE_NIC_NAME=eth2
bash tests/run_tests.sh --tag fast
```

两个静态门控已经按新 main 线适配过（18a8cff），本地实测 PASS：
`perf/check_cp_balance_fields.py` 报 `ZigzagPlan fields=13 / ZigzagCPPlan fields=13 / DSACPContext fields=19`，
`accuracy/check_b_path.py` 是 8/8。它们默认取 `harness.json trees.cur`，可以直接当门控用。

### 5.3 冒烟（2 次起停）

```bash
bash tests/run_tests.sh --only smoke/s10_service_ready,smoke/s11_zigzag_request
```

判据：s10 服务起来且 `/v1/models` 有模型名；s11 由 `accuracy/check_branch.py` 给 `RESULT: PASS`，
证据串 = `branch=ZIGZAG` **或** `[CP_BALANCE][plan]`（`harness.json verify.diagnose.zigzag_evidence` 可配）；
短 prompt 只有 `CONTINUOUS`。长 prompt 的 token 数不到 `min_tokens` 时脚本输出 INCONCLUSIVE、测试记 SKIP。
打点行来自 `attention/context_parallel/sfa_cp.py`（zigzag 成功时补了 `branch=ZIGZAG reason=-`），indexer 侧
拒绝时会多一行 `… site=indexer`。

### 5.4 精度 / 性能

```bash
bash tests/run_tests.sh --only accuracy/a10_matrix_gate --keep-going   # B==base + C 验收 + 噪声地板
bash tests/run_tests.sh --only perf/p10_capture --skip perf/p10_capture#prof_a2a
```

性能侧先看 `#prof_cp0 vs #prof_base`（噪声地板）再看 `#prof_cp1`，判读顺序见 `docs/perf_plan.md`。

### 5.5 回传

```bash
tar czf port_$(date +%m%d_%H%M).tgz tests/_out/*/status.tsv tests/_out/*/results.json \
    tests/_out/*/smoke tests/_out/*/accuracy vllm-ascend.log
```

## 6. 已知风险与缺口（按影响排序）

1. **indexer 侧的合并 metadata**：zigzag 下 LI top-k 需要 2B 长度 + 2B 行 block table。本次让 indexer
   builder 与 SFA builder 用同一个 planner 生成，但 kernel 是否接受“2B 行 block table + 2B 长度”只能
   在远端确认；现象是精度不对或直接报错。
2. **KV 写回的 scatter**：zigzag 时用全 padded 的 gather + `slot_mapping_cp_gathered`（padding 行为 -1）。
   若 kernel 不接受 -1 槽位，写入会越界/报错。
3. **归约点清单变化**：老树的三个门控点里 `maybe_pad_and_reduce` 在新 main 变成了 EP 集合通信，
   embedding 走 `all_reduce`（owner 无关，本来就安全）。现在门控的是 `MLPRowParallelOp` 与
   MoE shared expert 的 reduce-scatter；`OProjRowParallelOp` 与 `mmrs_fusion` 分支仍未门控（老树同样如此，
   本配置 fine-grained TP 没开）。
4. **V2 / DP>1 / draft 一律回退**连续切片：回退路径依赖 `fallback_cos/sin/slot_mapping`，缺一个会抛
   `RuntimeError`（有意为之，不静默降级）。
5. **`MIN_TOKENS` 默认 8192**，harness 配置显式 2048；prompt 短于阈值时不会进 zigzag（s11 会因此失败）。
6. 老 base_layer3 的打点/减层加载没有搬过来；如果精度排查还要用 3 层/6 层调试，需要单独在 base 线上补。
7. **dp=1 放开 DSA-CP 之后 zigzag 的 MoE 布局仍未决**：zigzag 给层内 MoE 的是 rank-local 切片行，
   而 SP 关时的 MoE（no-DP-EP）需要全量行 —— 所以 cp1 的精度失败可能是预期内的结构性问题，不是回归。
   三条出路 (a) dp=1 显式不开 zigzag /(b) dp=1 也开 SP /(c) 自己补层内 gather，见 `scripts_review.md` §8。
8. **上游要废弃 DSA-CP、迁到 PCP**：`ascend_config.py` 已打 “enable_dsa_cp will be fully deprecated once PCP is ready”，
   文档给的迁移方式是 `--tensor-parallel-size 1 --prefill-context-parallel-size N`；PCP 与 DSA-CP 互斥。
9. **min_tokens 的 s11 判据**：长 prompt token 不够时 check_branch 输出 INCONCLUSIVE、测试记 SKIP（不是 FAIL）。
