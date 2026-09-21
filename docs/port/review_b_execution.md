> 历史文档（2026-09-18 移植期取证/评审）：结论可能已过时；dp>1 那道门与 zigzag 的现状见 `docs/scripts_review.md` §4.1/§4.1c/§8。
>
> 已变事实（2026-09-21 更正）：
> - §0.1「当前提交里 zigzag 不会真正生效」与 B1 的不达性已解（dp=1 定案 + 两个树改门，§4.1）。
> - §5 B5「`for_draft=True` 是死参数」已修：`build()` 取出并透传 SFA/indexer/PCP 三个 builder，draft 步 `reason=draft`（b64c9569b，§4.3c/§4.3d）。
> - §7 Q3 的「靠 `attn_state`/MIN_TOKENS 偶然成立」已有显式信号；未覆盖的形状见 §4.3d。
> - B6（MC2/MoE prepare 假设整批 vs zigzag 局部行）仍未修——它正是 §8 的来源。

# cp_balance 移植评审 · Stage B（执行阶段）问题清单

- **审查对象**：`D:/code/cp_balance/vllm-ascend` 分支 `cp_balance` @ **d9dfc4497**
  （= 任务书里的 `612d03a8b` 之后又落了 2 个提交：`8912139f9` indexer zigzag gate / `d9dfc4497` seq_lens 热路径，
  评审开始时工作树已推进到这里；全部结论按 d9dfc4497 复核行号）。
  工作树在评审期间仍在被其它 agent 修改（`.scratch/fix_*.py`、`.scratch/c_review/`），**行号以 d9dfc4497 为准**。
- **基线**：`aff1b74b6`（upstream/main，无 cp_balance）；**老实现**：`23ff2c23c`（v0.26.0rc 线，含 zigzag）；
  **对照原版**：`vllm-ascend-base`（`base_layer3`=c7990e5e4 / `main`=aff1b74b6）。
- **只读手段**：`git show/diff/grep`、`sed`、`python -c`（AST 名字解析 + 槽位/行号算术复算）；未运行任何 torch/NPU 代码。
- **范围**：`context_parallel/sfa_cp.py` impl 段、`ascend_forward_context.py`、
  `patch/worker/patch_deepseek_v2.py`、`spec_decode/llm_base_proposer.py`（draft 排除），
  以及被它们咬到的 `attention/sfa_v1.py`、`attention/indexer.py`、`ops/mla.py`、`ops/fused_moe/prepare_finalize.py`。

一句话结论：**布局（plan/置换）本身自洽，但「执行阶段」只改了一半**——metadata 侧切到 zigzag 局部行，
而 SFA impl 的输入切片、o_proj 出口、MoE 侧入口、回退路径仍然按新 main 的
「模型级 = 全量自然序、attention 只算自己那一段」老契约写。zigzag 一旦真正生效（cp=8、纯 prefill、≥MIN_TOKENS），
rank>0 拿到的是零行/别人行。另有 1 处 import 级 BLOCKER 使模块根本加载不了。

### 0.1 与 Stage A 的关系 / 可达性前提

Stage A 已给出一条决定性结论（其 B1）：新 main 上
`ascend_config.py:664` 把 `enable_dsa_cp` AND 上 `use_sequence_parallel_moe`，而后者要求
`data_parallel_size > 1`（`vllm/config/parallel.py:711-725`）；而 zigzag gate 又把 `dp_size > 1`
直接判为不可用（`layers/cp_zigzag.py:571-572`、
`context_parallel/sfa_cp.py:484`、`attention/indexer.py:799`）。两者合起来：
**当前提交里 zigzag 不会真正生效**。
本文所有执行阶段问题（B2/B3/B6…）因此是「一旦 dp 门被放开 ⼀
修复完 Stage A B1 ⼀ 就立刻命中」，而不是当前就在线上爆炸。
建议修复顺序：B1（import）→ 决定 dp 门（Stage A B1）→ 同时修 B2/B3/B4/B6，
否则一旦该特性可达就是稳定出错。

---

## 0. 结论摘要

| 级别 | 编号 | 文件:行号 | 现象 | 建议 | 本地能否判定 |
| --- | --- | --- | --- | --- | --- |
| BLOCKER | B1 | `context_parallel/sfa_cp.py:48-50`、`attention/indexer.py:48` | 从 `vllm_ascend.utils` import 了**不存在**的 `dsa_cp_with_o_proj_tp_for_config`（全仓库无定义）与 `use_v2_model_runner`（实际在 `mrv2_utils`） | 改成已有符号：`enable_dsa_cp_full_o_proj()` / `from vllm_ascend.mrv2_utils import use_v2_model_runner` | **能**（AST 名字解析 + 全仓库 grep，见 §1） |
| BLOCKER | B2 | `context_parallel/sfa_cp.py:612-633`（调用 `sfa_v1.py:1707`，强制 native `sfa_v1.py:1663-1669`） | zigzag 下每层输入已是 `[prev,next]` 局部行，但 impl 仍 `pad 到 num_tokens_pad` 再切 `[local_start, local_end_with_pad)` → cp>1 时 **rank>0 全 0**（当前不可达，见 §0.1） | zigzag（`dsa_cp_context.zigzag_index is not None`）时直接返回 `hidden_states`（等价于用 `zigzag_index` 分片） | **能**（纯布局算术） |
| BLOCKER | B3 | `context_parallel/sfa_cp.py:637-645`、`:809-845`（输出张量 `ops/mla.py:250-255`） | 出口 o_proj 仍按「拼回全量自然序」处理：all_gather 后取前 `output.shape[0]` 行 = 各 rank 都写 r0 的行；`reduce_results=False` 分支按 `rank*local` 连续槽写入 → rank>0 全 0（当前不可达，见 §0.1） | zigzag 时 `gather_full_o_proj=False` 且仿射输出保持局部行（用全量权重、不做 TP 汇总） | **能**（代码路径判定；`reduce_results` 取值需再核一次） |
| BLOCKER | B4 | `ascend_forward_context.py:118-174`、`:273-292`；`attention/indexer.py:67-98` | 回退只还原 SFA metadata（`dsa_cp_context`），**indexer metadata 没有该字段 → 不还原** → 同一 forward 里 SFA 走连续切片、indexer 仍走 zigzag | 回退需同时还原 indexer metadata（cos/sin/slot/block_table/seq_lengths）；或在 indexer metadata 上也挂 fallback | **能**（dataclass 字段 + 迭代逻辑） |
| 高 | B5 | `spec_decode/llm_base_proposer.py:2528-2535` vs `sfa_v1.py:470-481`、`indexer.py:1006-1017` | `for_draft=True` 不是新 main 的 builder API：SFA `build()` 把 `**kwargs` 直接吞掉（draft veto 收不到）；indexer `build()` 转发 `**kwargs` 进 `_build`，`for_draft` 落进 `**kwargs` 也被丢。该调用点的 draft 排除实际不生效 | 删掉该 kwarg 并让这条路径走 `build_for_drafting(draft_index=...)`；若要保留语义就在 `build()` 里显式 `for_draft = bool(kwargs.pop("for_draft", False))` 并透传给 gate | **能**（静态） |
| 高 | B6 | `ops/fused_moe/prepare_finalize.py:252-311` + `ascend_forward_context.py:353-362` | MC2 路径的 `prepare` 假设输入是「全量 padded batch」：pad 到 `padded_num_tokens` 再 `tensor_split(...)[tp_rank]`；zigzag 下输入只有 1/tp 局部行 → rank>0 取到补齐区。mc2_mask 已被重排成 rank 拼接序（局部切片=本 rank 真掩码），与 hidden states 错位 | MoE 入口在 zigzag 时要么先把局部行 all-gather 成 rank 拼接序（`_prepare_with_ep_group` 的语义），要么明确禁用 MC2/FUSED_MC2 | 机制**能**判定；是否命中取决于运行时 comm type（见 §3.6） |
| 中 | B7 | `context_parallel/sfa_cp.py:752-800`（zigzag 分支 `:773-780`，C8 scatter `:784-789`，非 C8 `:796-800`）| `_store_parallel_kv` 在 zigzag 下改为「全 padded gather + `slot_mapping_cp_gathered`」scatter，**新依赖 slot==-1 被 CANN scatter 跳过**（`npu_scatter_nd_update_` / `npu_scatter_pa_kv_cache`） | 目标 CANN 上做一次定向 A/B（或先用 `slot>=0` 掩码过滤）；老分支（23ff2c23c）已有同样假设，但新 main 把 `exec_kv` 本地写也带进 -1 语义 | **不能**（需 NPU） |
| 中 | B8 | `ascend_forward_context.py:101-141`、`:273-292` | `zigzag_cp_active`/`zigzag_cp_context` 是**整个 forward 的单值**，而 metadata 可能是 list-of-dicts（ubatch）里多个不同 plan；`_find_zigzag_cp_context` 只取第一个 | 记录并断言「同一 forward 所有 metadata 的 zigzag plan 一致」，或把 plan 挂到 forward_context 逐 ubatch 取 | **能**（静态） |
| 中 | B9 | `ascend_forward_context.py:353-362`、`layers/cp_zigzag.py:79-93` | mc2_mask 重排只写 `forward_context.mc2_mask`（新张量），全局 `_reserved_mc2_mask` 仍是自然序；形状不匹配（`padded_num_tokens != gather_index.shape[0]`）直接 `RuntimeError` 而不是回退 | 统一从 `_EXTRA_CTX.mc2_mask` 读；形状不匹配时降级为连续路径而非抛错 | **能**（静态；消费方已确认只读 `_EXTRA_CTX.mc2_mask`） |
| 低 | B10 | `layers/cp_zigzag.py:79-87` | docstring 仍称 `set_ascend_forward_context` 会重排 `input_ids`；实际没有（新 main 的 MoE 根本不收 input_ids） | 删掉该段描述（结论本身是对的：不需要重排） | **能** |
| 低 | B11 | `ascend_forward_context.py:118-174` | 回退**原地重写** metadata/DSACPContext；ACL graph 抓取或 metadata 复用时该改写会固化 | 改成构造期决定，或回退时替换 metadata 对象而非改字段 | **能**（部分：是否命中取决于是否开 graph） |

---

## 1. B1：`dsa_cp_with_o_proj_tp_for_config` 不存在，两个模块 import 即失败（BLOCKER）

- **现象**：`vllm_ascend/attention/context_parallel/sfa_cp.py:46-60` 与 `vllm_ascend/attention/indexer.py:43-50`
  都从 `vllm_ascend.utils` 导入 `dsa_cp_with_o_proj_tp_for_config`；`sfa_cp.py` 还从 `vllm_ascend.utils`
  导入 `use_v2_model_runner`。
- **原因 / 证据（本地可判）**：
  - `python -c "ast 遍历 vllm_ascend/utils.py"`：顶层及任意层级都**没有** `dsa_cp_with_o_proj_tp_for_config`、
    `use_v2_model_runner`、`enable_dsa_cp_with_o_proj_tp` 的定义/别名，也没有 `import *` / `__getattr__`。
  - 全仓库 grep（含 `vllm/`、`tests/`）：该名字只出现在
    `sfa_cp.py:49,486`、`indexer.py:48,801` 以及 `./.scratch/fix_sfa_cp.py`、`./.scratch/patch_indexer.py`
    （即移植时的临时补丁脚本，没把符号一起带上）。
  - `use_v2_model_runner` 的正确定义在 `vllm_ascend/mrv2_utils.py:178`（`ascend_forward_context.py:20`、
    `indexer.py:43` 都是从那里导入的）。
- **影响**：`import vllm_ascend.attention.context_parallel.sfa_cp` 直接 `ImportError`，engine 起不来。
  这是移植脚本的半成品，说明**当前提交从未被真正 import 过**。
- **建议**：`from vllm_ascend.utils import enable_dsa_cp_full_o_proj` 用作 gate 的 `full_o_proj` 值
  （老实现里对应的是 `enable_dsa_cp_with_o_proj_tp()`；新 main 的等价物是 `enable_dsa_cp_full_o_proj()`，
  见 `utils.py:1523-1538`），`use_v2_model_runner` 改从 `mrv2_utils` 导入。

---

## 2. B2：attention 输入仍是「连续切片」假设 → zigzag 下 rank>0 全 0（BLOCKER）

- **现象**：`AscendSFADSACPImpl._prepare_native_hidden_states`（`sfa_cp.py:612-633`）在
  `actual_tokens < context.num_tokens_pad` 时**补零到全量长度**，再返回 `hidden_states[local_start:local_end_with_pad]`。
- **原因链（逐文件实证）**：
  1. 模型边界在 zigzag 下把 embedding 分片成局部行：
     `patch/worker/patch_deepseek_v2.py:353-377`（`zigzag_shard_tensor(hidden_states)`、`positions = zigzag_shard_tensor(positions)`），
     所以逐层 `hidden_states.shape[0] == num_tokens_pad / cp_size`。
  2. 本地行数 `local_tokens = num_tokens_pad // tp_size`（`zigzag_cp.py:166-175`、`ZigzagPlan.local_tokens`），
     而 `local_start = rank * num_tokens_per_device`（`sfa_cp.py:347-349`）。
  3. 因此对 rank>0：`local_start >= local_tokens`，切片落在**补零区** → 该 rank 的 q/kv 全 0；
     rank0 只是凑巧取到 `[0, local_tokens)`。
     例：A5 配置 tp=8、16K token prefill → `local_tokens=2048`、`num_tokens_pad=16384`，
     rank7 取 `[14336:16384)`，而真实数据只在 `[0:2048)`。
  4. 该函数在 pure-prefill 必然被调用：`sfa_v1.py:1663-1669` 对
     `attn_state not in (DecodeOnly, SpecDecoding)`（zigzag 只允许 `PrefillNoCache/PrefillCacheHit/ChunkedPrefill`）
     强制 `PreprocessType.NATIVE`，随后 `sfa_v1.py:1707` 调用它。fused 分支（mlapo/prolog_v3）在 zigzag 下不可达。
- **补充（说明这不是「本来就这样」）**：`aff1b74b6` 的 base 主线上模型级张量是**全量**的
  （`ops/vocab_parallel_embedding.py:306-325` `_forward_origin` 做 TP all-reduce，注释明写
  “first decoder layer must receive the complete token sequence”），attention 只切自己那段正好对；
  老实现（23ff2c23c / base_layer3）**根本没有这个函数**（grep 全树无 `_prepare_native_hidden_states`），
  因为那时模型级就是局部的。移植把模型级改成局部后漏改了这里。
- **建议**：`if context.zigzag_index is not None: return hidden_states`（必要时按 `zigzag_index` 校验行数），
  并删除 pad+连续切片分支在 zigzag 下的可达性。
- **本地能否判定**：**能**，纯布局算术 + 调用点可达性。

---

## 3. B3：o_proj 出口按「拼回全量自然序」写 → 局部行输出写错（BLOCKER）

- **现象**：`_get_parallel_forward_context` 在 zigzag prefill 下 `gather_full_o_proj=True`
  （`sfa_cp.py:637-645`：tp>1 且 `enable_dsa_cp_full_o_proj` 且状态非 decode/spec），
  于是 `_finalize_o_proj`（`sfa_cp.py:809-845`）走 gather 分支：
  - `o_proj.reduce_results == True`（GLM-5.2 未见置 False 的补丁；`models/*` 的置 False 只针对
    glm5next/kimi 等其它模型）：`full_output = tp_group.all_gather(local_output)` 得到的是
    **rank 拼接序** `[r0_prev,r0_next,r1_prev,...]`，然后 `output[...] = full_output[:output.shape[0]]`；
  - `reduce_results == False`：`local_start = rank*local_output.shape[0]` 是**连续槽**，rank>0 时
    `local_end == output.shape[0] < local_start` → `output` 保持全 0。
- **为什么在 zigzag 下必然错**：输出缓冲由 `ops/mla.py:250-255` 按 `hidden_states.shape[0]`（= 局部行数）
  分配；zigzag 下模型级必须保持局部行，出口拼装由模型边界统一做
  （`patch_deepseek_v2.py:428-436` `zigzag_gather_hidden_states_and_aux`）。impl 里再做一次
  「全量自然序」拼装既改变行数、又用错行号。
- **旁证（移植内部自相矛盾）**：`sfa_cp.py:819-825` 注释说“The decoder's sequence-parallel path will
  reduce-scatter this tensor”，而 `patch_deepseek_v2.py:442-449` 的注释说 SP 分支**不会**触发
  （`use_sequence_parallel_moe` 要求 `data_parallel_size>1`，见 `vllm/config/parallel.py:711-725`，
  本配置 dp=1）。两处对同一事实的假设相反，说明这一段没有跟着布局一起重审。
- **建议**：zigzag 时 `gather_full_o_proj=False`，并新增「用全量权重做局部输出、不做 TP 汇总」的分支
  （即在 `_use_full_o_proj_weights()` 下 `output[...] = local_output`），
  同时把 SP 相关注释按 dp=1 的实际事实改掉。
- **本地能否判定**：**能**（分支可达性与行号语义都是静态的；`reduce_results` 的实际取值可再确认一次）。

---

## 4. B4：回退只还原 SFA metadata，indexer metadata 留在 zigzag（BLOCKER）

- **现象**：`_disable_zigzag_metadata_for_fallback`（`ascend_forward_context.py:118-174`）遍历
  `attn_metadata` 里所有对象，但只处理 `getattr(meta, "dsa_cp_context", None)` 且 `zigzag_index is not None`
  的对象。indexer 的 metadata 是 `AscendSFAIndexerMetadata`（`attention/indexer.py:67-98`），
  **没有 `dsa_cp_context` 字段**，zigzag 结果（cos/sin/block_table/slot_mapping/actual_seq_lengths_*）
  是直接存在 dataclass 字段上的 → 回退时被跳过。
- **触发路径**：`ascend_forward_context.py:273-292`
  - `is_draft_model=True`（V1 MTP proposer：`llm_base_proposer.py:844-857`、`:1215-1226` 都是 `is_draft_model=True`）；
  - `_USE_V2_EXTRA_KWARGS`（V2 runner）；
  - `dp_world_size > 1`（`:320-327`）。
  此时 SFA metadata 被还原成连续切片（`slot_mapping_cp`、`cos/sin` 恢复，`zigzag_*` 清空），
  但同一个 `attn_metadata` dict 里 indexer 层的 metadata 仍是 zigzag：
  `slot_mapping` 是局部的、`block_table` 是 2B 的、`actual_seq_lengths_*` 是合并的。
- **影响**：SFA 走连续、indexer 走 zigzag。轻则 indexer 写 cache 的槽位/行数与 `hidden_states` 行数不符
  （形状不匹配直接抛错），重则 SFA 与 indexer 的 topk/cache 语义不一致导致数值错。
- **建议**：给 indexer metadata 也保留一份连续 fallback（或在 `_disable_...` 里按类分支还原），
  并在回退后断言「同 forward 内 SFA 与 indexer 的 layout 标记一致」。
- **本地能否判定**：**能**（字段与迭代逻辑静态可见）。

---

## 5. B5：`for_draft=True` 在新 main 的 builder API 里是死参数（高）

- **现象**：`spec_decode/llm_base_proposer.py:2528-2535` 调
  `builder.build(0, common_attn_metadata, self.runner.get_model(), for_draft=True, **extra)`。
- **原因**：
  - `AscendSFAMetadataBuilder.build`（`sfa_v1.py:470-481`）签名是
    `build(self, common_prefix_len, common_attn_metadata, fast_build=False, **kwargs)`，
    body 直接 `self._build(common_attn_metadata, draft_index=None)`——`**kwargs` 被丢弃，
    gate 里的 `speculative=draft_index is not None`（`context_parallel/zigzag_cp.py:155`）永远看不到 draft。
  - indexer 侧 `AscendSFAIndexerMetadataBuilder.build`（`indexer.py:1006-1017`）会
    `self._build(..., **kwargs)`，但 `_build`（`indexer.py:1073-1086`）的形参里没有 `for_draft`，
    于是它落进 `**kwargs` 同样被丢。
  - 老实现（23ff2c23c `sfa_v1.py:469-477`）里 `build()` 会 `kwargs.pop("for_draft", False)` 并传给 gate，
    所以这是移植到新签名时刻意保留、但下游没接的一条线。
  - 新 main 真正的 draft 入口是 `build_for_drafting(common_attn_metadata, draft_index, **kwargs)`
    （proposer 在 `:811-815`、`:1184`、`:2084` 用它；`8912139f9` 也正是把 `draft_index` 透传进 indexer `_build`）。
- **影响**：这条 metadata 构建路径的 draft 排除完全依赖 `attn_state / is_prefilling / MIN_TOKENS` 兜底；
  一旦 draft batch 命中 pure-prefill gate（本仓库 A5 配置 `min_tokens=2048`，MTP 大 batch 有可能），
  drafter 会拿到 zigzag metadata，再叠加 B4 的半还原 → 数值错。
- **建议**：删掉 `for_draft=True`，让该调用点走 `build_for_drafting(draft_index=...)`；
  若必须保留旧语义，则在 `build()` 里显式 pop 并透传到 gate（老实现的做法）。
- **本地能否判定**：**能**。

---

## 6. B6：MoE（MC2 家族）入口与 zigzag 局部行不兼容（高，取决于 comm type）

- **现象**：`ascend_forward_context.py:353-362` 把 `mc2_mask` 按 `zigzag_gather_index` 重排成
  「rank 拼接序」，隐含假设 MoE 看到的全局 batch = 各 rank 局部行的拼接（这正是局部切片
  `tensor_split(mask, tp)[tp_rank]` 能得到本 rank 真掩码的原因）。
- **但 hidden states 没有任何地方做这个拼接**：
  - `PrepareAndFinalizeWithAllGather._prepare_with_ep_group`（`prepare_finalize.py:401-428`，gather 在 `:420-421`）才是
    「本地行 → `maybe_all_gather_and_maybe_unpad` → rank 拼接序」的那一步，它要求
    `moe_config.is_sequence_parallel`（`prepare_finalize.py:378-380`）；
  - 该开关来自 `parallel_config.use_sequence_parallel_moe`，它在 vLLM 里要求
    `data_parallel_size > 1`（`vllm/config/parallel.py:711-725`），本配置 dp=1 → False；
  - 于是走 `PrepareAndFinalizeWithMC2.prepare`（`prepare_finalize.py:252-311`）时：
    `self.num_tokens = hidden_states.shape[0]`（局部行）→ `pad_size = padded_num_tokens - 局部行 > 0`
    → pad → `tensor_split(hidden_states, tp)[tp_rank]`：rank0 拿 `[0,local)`（恰好是自己的行），
    rank>0 拿的是补齐区；而被切分的 `mc2_mask` 却是各 rank 的真掩码。
- **是否命中**：A5 profile = `MoECommPolicy.CAPACITY_AND_WORLD_SIZE`
  （`device/hardware_profile.py:203`、`ascend_forward_context.py:466-493`）：
  `enable_fused_mc2` 默认 0（`ascend_config.py:490`，A5 config 未覆盖）→
  `num_tokens <= mc2_tokens_capacity(=min(ceil(max_num_tokens/tp),512)*tp=4096)` 时选 **MC2**；
  否则 `world_size(8) <= num_experts_per_tok(8)` → **ALLGATHER**（该路径 `_prepare_with_dp_group`
  在 dp=1/pcp=1 时是 no-op，**不做** gather，各 rank 只算自己的局部行，与 zigzag 自洽）。
  也就是说：全局 batch ≤4096 token 的 zigzag prefill 会踩 MC2（2048~4096 正好落在
  `VLLM_ASCEND_CP_BALANCE_MIN_TOKENS=2048` 之上），更大的 batch 走 ALLGATHER 反而没问题。
- **建议**：zigzag 时要么显式 all-gather 局部行到 rank 拼接序再进 MoE，要么在 zigzag 下禁用 MC2/FUSED_MC2
  （并把该限制写进 gate）；同时确认 ALLGATHER 路径下 `mc2_mask` 是否被消费（若不被消费，
  B9 的重排是无用功，真正生效的只是「不 gather」这条隐含前提）。
- **本地能否判定**：机制**能**；是否命中需要按 `num_experts_per_tok`/`max_num_batched_tokens`/
  `_MC2_TOKENS_PER_RANK_LIMIT` 具体算（本文件已给出算式），最终仍需 NPU 复现确认。

---

## 7. 必答问题逐条回答

### Q1 KV 写回两条路径：slot 集合 / 行数 / -1 语义是否自洽？C8 与非 C8 都对吗？

- **路径 A `exec_kv`（本地行）**：入参 `slots = parallel_context.kv_slot_mapping = context.slot_mapping_cp`
  （`sfa_v1.py:1726-1733` → `_get_parallel_forward_context`），zigzag 下 = `slot_mapping[zigzag_index]`，
  长度 `local_tokens` = `kv_no_split` 行数（**前提是 B2 修好**——当前 `_prepare_native_hidden_states`
  给的 Q/KV 行内容与槽位不匹配）。槽集合 = 本 rank 的全局自然位置集合，与 `zigzag_index` 一一对应，**自洽**。
  非 C8 走 `npu_kv_rmsnorm_rope_cache(..., slots, ...)`（`sfa_cp.py:696-723`）；C8 走
  `super().exec_kv` → `custom_kv_rmsnorm_rope`（`sfa_v1.py:1112-1124`，**只返回、不落盘**）。
- **路径 B `_store_parallel_kv`（全量 gather）**：`_prepare_kv_for_parallel`
  （`sfa_cp.py:725-750`）用 `all_gather_async` 做 rank 拼接，得到
  `[r0_prev,r0_next,r1_prev,...]`；`slot_mapping_cp_gathered = slot_mapping[zigzag_gather_index]`
  （`context_parallel/zigzag_cp.py:232`）与之逐行对应，行数 = `num_tokens_pad` = `kv_to_write` 行数，**自洽**。
- **-1 语义**：zigzag 下行里含 pad 行（natural 尾部，slot=-1），路径 B 把**全 padded** 张量做 scatter：
  C8 用 `torch_npu.npu_scatter_nd_update_`（`sfa_cp.py:784-789`）、非 C8 用
  `DeviceOperator.reshape_and_cache` → `npu_scatter_pa_kv_cache`（`device/device_op.py:42-70`），
  这两处都需要 CANN「slot<0 跳过」的语义。基线的连续路径只写 `[:num_actual_tokens]`（无 -1），
  所以这是 zigzag 新引入的依赖；老实现（23ff2c23c `sfa_v1.py:2251-2260`）同样依赖它，但**本地无法判定**，
  必须目标 CANN 上定向 A/B（见 B7）。
  另注：`slot_mapping_cp` 本身也可能含 -1（pad 行落在任意 rank 的 prev/next 块里），
  所以路径 A 的 -1 语义在 base 主线连续切片下**已经**存在，不算新增。
- **C8 与非 C8 的差异**：C8 只有路径 B 落盘（`exec_kv` 只算不写），非 C8 是「路径 A 落盘 + 路径 B 再落一次」。
  两条路径的值应逐行相同（同一批 k_pe/k_nope 经 gather 回写），属于冗余但不致错；
  但**当前 B2/B3 未修时，路径 B 会把 rank0 的正确行也覆盖成各 rank 的垃圾行**，会放大故障。
- **额外核对**：非 C8 分支 `kv_to_write = fused_kv_no_split`（全量）后 `split` 出来的
  k_pe/k_nope 行数已是 gather 全量，而 forward 里此后**不再使用**
  （`sfa_v1.py:1737-1746` 之后只有 indexer 调用），所以不会外溢；这属于历史遗留，不是新问题。

### Q2 `_get_parallel_forward_context` 返回合并 2B 长度后，forward 里其它用「连续切片 local 长度」的地方有没有漏换？

**有，而且是致命的**。逐点核对 `sfa_v1.py:1650-1795` 的 forward：

| 位置 | zigzag 下现状 | 判定 |
| --- | --- | --- |
| `_prepare_native_hidden_states`（`sfa_cp.py:612-633`） | 仍 pad+连续切片 | **错**（B2） |
| `exec_kv` 的 `kv_slot_mapping` | `slot_mapping_cp`（zigzag 序） | 对（前提 B2） |
| `actual_seq_lengths_query/key` | 合并 2B 长度 | 对 |
| `topk_num_tokens = local_end_with_pad - local_start` | 数值上 == `local_tokens` | 对（但语义上应直接用 `local_tokens`，避免读者误解成连续切片） |
| `_finalize_o_proj` / `gather_full_o_proj` | 拼回全量自然序 | **错**（B3） |
| `_execute_sparse_flash_attention_process(block_table=block_table_zigzag)` | 2B 行、与合并长度/合并 topk 对齐 | 对 |
| fused 分支（PROLOG_V3/MLAPO）用 `slot_mapping_sfa`（全量 natural） | zigzag 不可达（prefill 强制 NATIVE） | 暂不影响，但注释/断言（`sfa_v1.py:1672-1676`）会误导后继修改 |
| indexer 调用 | 用 indexer 自己的 metadata（独立 plan） | 对——**前提是两份 plan 不漂移**（B4/B5 正是漂移点） |

### Q3 出口 gather：行数与 runner 期望是否一致？aux / PP 中间张量 / MTP draft 步会不会走错分支？

- **行数一致**：`patch_deepseek_v2.py:317` 用 `full_num_tokens = positions.shape[0]`（= padded 输入行数），
  `zigzag_gather_tensor(..., num_tokens)` 在 `cp_zigzag.py:630-651` 里 all_gather + `inv_gather_index` + `[:num_tokens]`，
  与非 zigzag 的 `combined_states[: positions.shape[0]]` 语义一致（runner 之后还会按
  `num_scheduled_tokens` 截断，见 `worker/model_runner_v1.py:2020`）。**一致**。
- **aux hidden states**：zigzag 下逐层不做 gather（`patch_deepseek_v2.py:404` 加了 `not zigzag_active`），
  出口统一走 `zigzag_gather_hidden_states_and_aux`（`:433-436`），行数/顺序与非 zigzag 相同。**对**。
  唯一细节：非 zigzag 是“cat(hidden,residual) → gather → norm”，zigzag 是“norm 后 gather”，
  RMSNorm 逐行且出口丢掉第二个返回值，等价。**不构成问题**。
- **PP 中间张量**：`patch_deepseek_v2.py:410-411` 对非末 PP rank 直接返回**局部行**的
  `IntermediateTensors`；下游 stage 也按局部行处理，前提是**所有 PP rank 的 zigzag 判定一致**。
  gate 里含 rank/角色相关的 `full_o_proj=enable_dsa_cp_full_o_proj()`（与 kv producer 角色有关），
  PD 分离时不同实例角色不同——同一实例内一致，但跨 PP stage 的一致性没有任何断言。**需补断言**（中危）。
- **MTP draft 步**：正常路径由 `set_ascend_forward_context(is_draft_model=True)` 否决
  （`ascend_forward_context.py:278-292`），且 metadata 的 gate 还看 `draft_index`；
  但 (a) proposer 的 `for_draft=True` 不生效（B5），(b) 回退不覆盖 indexer metadata（B4），
  所以「draft 步绝不走 zigzag」这条不变量目前**靠 `attn_state`/`MIN_TOKENS` 偶然成立**，不是显式保证。

### Q4 mc2_mask 重排用的置换与 zigzag_gather_index 是否同一张？input_ids 重排还需要吗？

- **同一张**：`ascend_forward_context.py:353-362` → `layers/cp_zigzag.py:75-93` 直接用
  `ctx.zigzag_gather_index`（`_EXTRA_CTX.zigzag_cp_context`，即 SFA planner 产出的那张张量），
  与模型边界 all_gather 的 rank 拼接序、以及 MoE 的 `tensor_split(mask, tp)[r]` 局部切片同源。**对**。
  但注意它是「全局 mask 重排」的前提是 MoE 看到 rank 拼接序的全局 batch —— 见 B6。
- **input_ids 重排**：新 main **不需要**。`DeepseekV2MoE.forward(hidden_states, already_sequence_parallel)`
  （`vllm/model_executor/models/deepseek_v2.py:406-408`）不接收 input_ids，MoE runner 的 `input_ids`
  默认 `None`（`ops/fused_moe/fused_moe.py:154/416`）；`forward_context.input_ids` 在整个 vllm-ascend 里
  没有任何读点（grep 确认，老实现的 `ascend_forward_context.py:333-342` 那套在新 main 没有对应入参）。
  所以移植里没做重排是**正确**的，只是 `cp_zigzag.py:79-87` 的 docstring 还写着会重排，需要删（B10）。

### Q5 回退（draft/V2/DP>1）时：cos/sin/slot 还原了，indexer metadata / 模型 boundary / KV 写回是否也回到连续切片？有没有只还原一半？

- **SFA 侧**：全（`slot_mapping_cp`、`meta.cos/sin` 还原，`zigzag_*`/`actual_seq_lengths_*_zigzag`/
  `block_table_zigzag`/`slot_mapping_cp_gathered` 清空 → `_execute_sparse_flash_attention_process`
  回落 `attn_metadata.block_table`、`_store_parallel_kv` 回落
  `slot_mapping_sfa[:num_actual_tokens]`、`zigzag_active()=False`）。
- **模型 boundary**：`zigzag_cp_active=False` → `patch_deepseek_v2.py:313` 之后所有 zigzag 分支都不进，
  `_EXTRA_CTX.zigzag_cp_context` 也被置 None（`:288-290`）→ `zigzag_shard_tensor` 不会被调用。**一致**。
- **indexer metadata**：**没还原**（B4）→ **存在「只还原一半」的路径**，这是本节的核心答案。
- **KV 写回**：SFA 侧随 metadata 一起回落；indexer 的 LI cache 写仍按 zigzag 局部 slot/2B block_table → 不一致。
- **补充风险（B11）**：回退是**原地改对象**；若 metadata 被复用（graph capture / 缓存），
  这次改写会固化到后续 replay。

### Q6 `zigzag_active()` 的读取时机（metadata 先建、forward context 后建）是否存在不一致窗口？

- **存在，但当前被两件事兜住**：
  1. vLLM 的 `set_forward_context` 每次 **新建** `ForwardContext`（`vllm/forward_context.py:326-347`
     `create_forward_context` + `override_forward_context`），退出后恢复旧对象，所以上一个 forward 的
     `zigzag_cp_active` 不会渗到下一个 forward；缺省读写走
     `_ExtraForwardContextProxy.__getattr__`（`getattr(ctx, name, None)`）→ False，安全。
  2. `_EXTRA_CTX.zigzag_cp_active` 在 `set_ascend_forward_context` 内的赋值点
     （`ascend_forward_context.py:344-345`）位于所有引用者（模型 forward、linear_op、shared_experts）之前。
- **真正的窗口**：metadata 在 **build 阶段**就按 gate 决定了布局（`sfa_cp.py:448-552`、`indexer.py:762-850`），
  而 `zigzag_cp_active` 到 **set_ascend_forward_context** 才决定（`ascend_forward_context.py:273-292`）。
  两者判定输入不同（前者没有 `is_draft_model`/`_USE_V2_EXTRA_KWARGS`/`dp_world_size`），
  只能靠「回退时把 metadata 改回来」弥合 —— 这就把 B4（indexer 不还原）、
  B5（draft 标记没送到 gate）、B6（MoE 看到的张量不是 rank 拼接序）都变成了同一个窗口的衍生问题。
  另外 `_find_zigzag_cp_context` 只看**第一个**含 zigzag 的 metadata（ubatching 时是 list-of-dicts），
  单值 flag 无法表达「多个 microbatch 各自 plan」（B8）。
- **建议**：把布局决策上移到「先算 forward 级 veto，再 build metadata」（或在 metadata build 里也接受
  `is_draft_model`/V2/dp 信息并断言两侧一致），至少加一条
  「SFA/Indexer/forward_context 三处 layout 标记一致」的断言。

---

## 8. 本地判定手段与未覆盖点

- **已做**：`git show/diff` 逐文件比对 `aff1b74b6`/`23ff2c23c`；`python -c` AST 名字解析（B1）；
  行号/槽位算术复算（B2/B3）；dataclass 字段与调用点 grep（B4/B5/B6/B9/B10）。
- **本地不能判定**：B7（CANN scatter 对 slot=-1 的行为）、B6 的最终命中（需实测 comm type），
  以及任何涉及 `torch_npu` 算子的语义（本次未运行 torch/NPU）。
- **未覆盖（超出本次范围）**：`ops/linear_op.py` / `distributed/utils.py` 的 fixed-order reduce
  契约与宽异常（老评审 A1/A2 已提，本文件 B6 会放大其前提）、PCP/DCP 组合下的
  `_update_parallel_slot_mapping`、以及 debug 打点本身。
