#!/usr/bin/env bash
# A5 一键验证：新 main 线上的 cp_balance（移植版）
#
#   bash verify_a5.sh                 # 前置 -> 静态 -> 起服务冒烟 -> 分支诊断 -> 打包证据
#   bash verify_a5.sh --skip-fast     # 跳过 --tag fast
#   bash verify_a5.sh --skip-smoke    # 不起服务
#   bash verify_a5.sh --diag-only     # 只从最近一次 tests/_out 抓分支日志
#
# 代码版本：verify_a5.sh 会比对 tests/lib/targets.tsv（cur=cp_balance / base=main），
# 分支或 commit 不对时自动 checkout + reset 到 origin/<分支>（脏树会拒绝并列出改动，
# 想只检查不切换就 CP_BALANCE_AUTO_CHECKOUT=0）
#
# 环境变量：CP_BALANCE_LOCAL_IP / CP_BALANCE_NIC_NAME 未设置时自动识别
#           （默认路由接口优先，容器内没有 ip/ifconfig 也能用；多网卡建议显式指定）
# 可选：CP_BALANCE_DEVICES / CP_BALANCE_REPO / CP_BALANCE_BASE_REPO / HX_READY_TRIES
set -o pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

SKIP_FAST=0; SKIP_SMOKE=0; DIAG_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --skip-fast)  SKIP_FAST=1 ;;
    --skip-smoke) SKIP_SMOKE=1 ;;
    --diag-only)  DIAG_ONLY=1; SKIP_FAST=1; SKIP_SMOKE=1 ;;
    -h|--help)    sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done

STAMP=$(date +%m%d_%H%M%S)
BRANCH_REPORT="$HERE/verify_a5_branch_$STAMP.txt"
EXIT_CODE=0; FAILED_STEPS=()
step() { printf '\n===== %s =====\n' "$*"; }
ok()   { printf '[ok]   %s\n' "$*"; }
note() { printf '[note] %s\n' "$*"; }
warn() { printf '[warn] %s\n' "$*"; }
bad()  { printf '[FAIL] %s\n' "$*"; EXIT_CODE=1; }

step "0. 前置检查"
export CP_BALANCE_FAMILY=a5
missing=()
[ -n "${CP_BALANCE_LOCAL_IP:-}" ] || missing+=(CP_BALANCE_LOCAL_IP)
[ -n "${CP_BALANCE_NIC_NAME:-}" ] || missing+=(CP_BALANCE_NIC_NAME)
if [ ${#missing[@]} -gt 0 ]; then
  note "未设置 ${missing[*]}：尝试自动识别（容器内没有 ip/ifconfig 也可用）"
  BEST=$(python3 tests/lib/netif.py --best 2>/dev/null || true)
  if [ -n "$BEST" ]; then
    DET_NIC=$(printf '%s' "$BEST" | cut -f1)
    DET_IP=$(printf '%s' "$BEST" | cut -f2)
    [ -n "${CP_BALANCE_NIC_NAME:-}" ] || export CP_BALANCE_NIC_NAME="$DET_NIC"
    [ -n "${CP_BALANCE_LOCAL_IP:-}" ] || export CP_BALANCE_LOCAL_IP="$DET_IP"
    note "识别到 NIC=$CP_BALANCE_NIC_NAME IP=$CP_BALANCE_LOCAL_IP"
    note "全部候选（网卡 / IP / default=默认路由接口）："
    python3 tests/lib/netif.py | sed 's/^/       /'
    note "多网卡/多网段（RoCE 与业务网分开）时请显式 export CP_BALANCE_LOCAL_IP / CP_BALANCE_NIC_NAME"
  else
    bad "没有识别到可用 IPv4 网卡"
    cat <<'EOF'
  容器里请显式指定（候选可用 python3 tests/lib/netif.py 查看）：
    export CP_BALANCE_LOCAL_IP=<本机 IP>
    export CP_BALANCE_NIC_NAME=<网卡名>
  或把 configs/_common_a5.json 的 local_ip / nic_name 改成具体值（就不再走 auto）。
EOF
    exit 2
  fi
fi
[ -f tests/run_tests.sh ] || { bad "tests/run_tests.sh 不存在"; exit 2; }
[ -f tests/lib/roles.tsv ] || { bad "tests/lib/roles.tsv 不存在"; exit 2; }

. "$HERE/tests/lib/common.sh"

SVC_CFG=$(hx_role svc_cp1)
[ -n "$SVC_CFG" ] || { bad "roles.tsv 里 a5 没有 svc_cp1"; exit 2; }
CUR_REPO=${CP_BALANCE_REPO:-$(hx_path "$SVC_CFG" repo)}
BASE_REPO=${CP_BALANCE_BASE_REPO:-}
[ -n "$BASE_REPO" ] || BASE_REPO=$(python3 -c "import json;print(json.load(open('configs/matrix_a5_b_vs_base.json')).get('static_check',{}).get('base_repo',''))" 2>/dev/null)
MODEL=$(hx_cfg_field "$SVC_CFG" model)
PORT=$(hx_cfg_field "$SVC_CFG" port)

note "family=a5 配置=$SVC_CFG port=$PORT"
note "IP=$CP_BALANCE_LOCAL_IP NIC=$CP_BALANCE_NIC_NAME DEVICES=${CP_BALANCE_DEVICES:-<配置默认>}"
note "被测树=$CUR_REPO"
note "对照树=$BASE_REPO"
hx_need_dir "被测代码树" "$CUR_REPO"
hx_need_dir "对照代码树" "$BASE_REPO"
hx_need_dir "权重" "$MODEL"
if [ -n "$MODEL" ] && [ ! -d "$MODEL" ]; then
  cand=$(python3 "$HERE/tests/lib/trees.py" model-candidates 2>/dev/null | head -5)
  if [ -n "$cand" ]; then
    note "本机找到这些权重候选（可用 CP_BALANCE_MODEL=<路径> 覆盖）："
    printf '%s
' "$cand" | sed 's/^/       /'
  fi
fi
# 代码版本对齐（tests/lib/trees.json 里维护）：树不在就 clone，版本不对就切到目标
. "$HERE/tests/lib/sync_tree.sh"
if [ "${CP_BALANCE_AUTO_CHECKOUT:-1}" = "1" ]; then
  while IFS=$'	' read -r role tpath _tremote _tref _tkind _treq _texists; do
    [ -n "$role" ] || continue
    override=""
    case "$role" in
      cur)  override=$CUR_REPO ;;
      base) override=$BASE_REPO ;;
    esac
    if ! hx_ensure_tree "$role" "$override"; then
      bad "代码树就绪失败：$role"
    fi
  done < <(hx_tree_roles)
  if [ -d "$HERE/.git" ] && [ -z "$(git -C "$HERE" status --porcelain 2>/dev/null | head -5)" ]; then
    if git -C "$HERE" fetch --quiet origin main 2>/dev/null        && [ "$(git -C "$HERE" rev-parse HEAD)" != "$(git -C "$HERE" rev-parse origin/main 2>/dev/null)" ]; then
      note "harness 落后 origin/main，执行 git pull --ff-only"
      git -C "$HERE" pull --ff-only --quiet || warn "harness 自更新失败，继续用当前版本"
    fi
  fi
else
  note "CP_BALANCE_AUTO_CHECKOUT=0：只显示当前版本，不自动拉取/切换"
fi

for r in "$CUR_REPO" "$BASE_REPO"; do
  [ -n "$r" ] && [ -d "$r/.git" ] && note "$r -> $(git -C "$r" log -1 --oneline 2>/dev/null)"
done
if [ -f "$CUR_REPO/vllm_ascend/envs.py" ]; then
  grep -q VLLM_ASCEND_CP_BALANCE "$CUR_REPO/vllm_ascend/envs.py" \
    && ok "被测树含 VLLM_ASCEND_CP_BALANCE（移植版）" \
    || bad "被测树没有 CP_BALANCE：不是移植后的树（应为 cp_balance 分支）"
fi
if [ -f "$BASE_REPO/vllm_ascend/envs.py" ] && grep -q VLLM_ASCEND_CP_BALANCE "$BASE_REPO/vllm_ascend/envs.py"; then
  bad "对照树里也有 CP_BALANCE：对照树应是原版 main"
fi
# 构建产物：_build_info.py 由 setup.py 生成，缺了服务会在 import 阶段就死
if [ -f "$CUR_REPO/vllm_ascend/__init__.py" ] && [ ! -f "$CUR_REPO/vllm_ascend/_build_info.py" ]; then
  bad "被测树缺构建产物 vllm_ascend/_build_info.py（服务会 ImportError: cannot import name '_build_info'）"
  if [ "${CP_BALANCE_AUTO_BUILD:-0}" = "1" ]; then
    note "CP_BALANCE_AUTO_BUILD=1：在 $CUR_REPO 里执行 pip install -e . --no-build-isolation（几分钟）"
    ( cd "$CUR_REPO"       && { [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && . /usr/local/Ascend/ascend-toolkit/set_env.sh || true; }       && pip install -e . --no-build-isolation ) >"$HERE/verify_a5_build_$STAMP.log" 2>&1       && ok "构建完成（日志 verify_a5_build_$STAMP.log）"       || bad "自动构建失败，看 verify_a5_build_$STAMP.log"
    [ -f "$CUR_REPO/vllm_ascend/_build_info.py" ] && ok "构建产物已生成"
  fi
  cat <<'EOF'
  说明这棵树没有在本机构建/安装过（harness 用 PYTHONPATH 直接指到这棵树）。修法任选：
    a) 重新构建（推荐，同时生成 C 扩展）：
         cd <被测树> && source <CANN 路径>/set_env.sh && pip install -e . --no-build-isolation
    b) 临时从同机对照树拷（该文件只跟芯片型号有关，与本仓代码无关）：
         cp <对照树>/vllm_ascend/_build_info.py <被测树>/vllm_ascend/
  验证：PYTHONPATH=<被测树> python3 -c "import vllm_ascend._build_info as b; print(b.__device_type__)"
EOF
fi
if [ -d "$CUR_REPO/vllm_ascend" ] && ! ls "$CUR_REPO"/vllm_ascend/vllm_ascend_C*.so >/dev/null 2>&1; then
  warn "被测树里没有 vllm_ascend_C 扩展（vllm_ascend_C*.so）：如果后续报自定义算子缺失，按上面 a) 重新构建"
fi

note "vllm 版本: $(python3 -c 'import vllm;print(vllm.__version__)' 2>&1 | tail -1)"
note "SOC: $(python3 -c 'import torch_npu;print(torch_npu.npu.get_soc_version())' 2>&1 | tail -1)"
df -h "$HERE" | tail -1

if [ "$EXIT_CODE" -ne 0 ]; then
  bad "前置检查未通过，后续大概率失败；修完再跑（--diag-only 可只抓日志）"
  [ "$DIAG_ONLY" -eq 0 ] && exit 2
fi

if [ "$SKIP_FAST" -eq 0 ]; then
  step "1. 静态测试（tests/run_tests.sh --tag fast）"
  if bash tests/run_tests.sh --tag fast; then ok "静态测试全过"; else bad "静态测试有 FAIL（看 tests/_out/*/ 下各 id.log）"; FAILED_STEPS+=(fast); fi
fi

if [ "$SKIP_SMOKE" -eq 0 ]; then
  step "2. 起服务冒烟（s10 就绪 + s11 走 ZIGZAG）"
  if bash tests/run_tests.sh --only smoke/s10_service_ready,smoke/s11_zigzag_request --keep-going; then
    ok "冒烟通过"
  else
    bad "冒烟未通过（看 tests/_out/*/smoke 下的服务日志）"; FAILED_STEPS+=(smoke)
  fi
fi

step "3. 分支诊断（A-B1：zigzag 是否被 dp 前提挡住）"
OUT_DIR=$(ls -1dt tests/_out/*/ 2>/dev/null | head -1)
if [ -z "$OUT_DIR" ]; then
  warn "没有 tests/_out 目录，跳过诊断"
else
  note "最新一轮证据：$OUT_DIR"
  {
    echo "# cp_balance 分支日志（$(date '+%F %T')）"
    echo "# 证据目录: $OUT_DIR"
    echo "# 配置: $SVC_CFG  IP=$CP_BALANCE_LOCAL_IP NIC=$CP_BALANCE_NIC_NAME"
    echo; echo "## [CP_BALANCE][branch] 行（按次数排序）"
    grep -rho '\[CP_BALANCE\]\[branch\][^\r]*' "$OUT_DIR" 2>/dev/null | sort | uniq -c | sort -rn
    echo; echo "## [CP_BALANCE][plan] 行（最多 5 条）"
    grep -rho '\[CP_BALANCE\]\[plan\][^\r]*' "$OUT_DIR" 2>/dev/null | head -5
    echo; echo "## DSA-CP / runner 相关行（去重，最多 20 条）"
    grep -rhoE '(DSA[- _]?CP|dsa_cp|Model Runner V[12]|data_parallel_size|sequence_parallel_moe)[^\r]*' "$OUT_DIR" 2>/dev/null | sort -u | head -20
  } > "$BRANCH_REPORT" 2>&1
  cat "$BRANCH_REPORT"; echo
  BRANCH_LINES=$(grep -rho '\[CP_BALANCE\]\[branch\][^]*' "$OUT_DIR" 2>/dev/null)
  if printf '%s' "$BRANCH_LINES" | grep -q 'branch=ZIGZAG'; then
    ok "出现 branch=ZIGZAG：资格门通过，可以往精度/性能走"
  elif printf '%s' "$BRANCH_LINES" | grep -q 'reason=dp>1'; then
    bad "只有 reason=dp>1：确认 A-B1（main 上 enable_dsa_cp 要求 data_parallel_size>1，见 vllm_ascend/ascend_config.py 的 use_sequence_parallel_moe 校验；而 zigzag 资格门拒绝 dp>1）"
    echo "      二选一后再往下：a) 放宽 ascend_config 回到 tp+dp1+FlashComm1；b) 让 zigzag 支持 DP>1"
    FAILED_STEPS+=(ab1-dp-gate)
  elif [ -n "$BRANCH_LINES" ]; then
    bad "有 branch 日志但既不是 ZIGZAG 也不是 dp>1：看报告里的 reason 字段"; FAILED_STEPS+=(branch-reason)
  else
    bad "没有任何 [CP_BALANCE][branch] 行：确认 debug 打开（VLLM_ASCEND_CP_BALANCE_DEBUG=1）且真进了 DSA-CP 分支"; FAILED_STEPS+=(no-branch-log)
  fi
fi

step "4. 汇总"
if [ "$EXIT_CODE" -eq 0 ]; then ok "一键验证：PASS"; else bad "一键验证：FAIL（${FAILED_STEPS[*]}）"; fi
TAR="$HERE/verify_a5_$STAMP.tar"
tar cf "$TAR" -C "$HERE" --files-from /dev/null 2>/dev/null
for f in "verify_a5_branch_$STAMP.txt"; do
  [ -f "$HERE/$f" ] && tar rf "$TAR" -C "$HERE" "$f" 2>/dev/null
done
if [ -n "$OUT_DIR" ] && [ -d "$OUT_DIR" ]; then
  for f in status.tsv results.json; do
    [ -f "$OUT_DIR$f" ] && tar rf "$TAR" -C "$HERE" "${OUT_DIR#./}$f" 2>/dev/null
  done
  tar rf "$TAR" -C "$HERE" "${OUT_DIR#./}" 2>/dev/null
fi
if [ -s "$TAR" ]; then
  gzip -f "$TAR"
  note "证据包：$TAR.gz ($(du -h "$TAR.gz" 2>/dev/null | cut -f1))"
fi

cat <<EOF

下一步：
  1) 诊断是 reason=dp>1 时先定 dp 方案，再重跑本脚本；
  2) 出现 ZIGZAG 后：bash tests/run_tests.sh --only accuracy/a10_matrix_gate --keep-going
  3) 性能：bash tests/run_tests.sh --only perf/p10_capture --skip perf/p10_capture#prof_a2a
  4) 回传 verify_a5_*.tgz（原始 trace 不用拷）
EOF
exit "$EXIT_CODE"
