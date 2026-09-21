#!/usr/bin/env bash
# 一键验证（机器族无关）：前置检查 -> harness.json verify.stages -> 分支诊断 -> 打包证据
#
#   bash verify.sh                    # family 取 CP_BALANCE_FAMILY，默认 a5
#   bash verify.sh --family a3
#   bash verify.sh --skip-fast        # 跳过任一 stage（名字见 harness.json verify.stages[].skip_flag）
#   bash verify.sh --skip-smoke
#   bash verify.sh --diag-only        # 只对最近一轮 tests/_out 出诊断报告
#   bash verify.sh --live-log         # 测试与模型服务日志实时打屏（默认只写文件）
#
# 判据、步骤、日志格式全部来自 harness.json（verify.stages / verify.diagnose），
# 机器身份来自环境变量（CP_BALANCE_LOCAL_IP / CP_BALANCE_NIC_NAME / CP_BALANCE_DEVICES），
# 代码树来自 harness.json trees（CP_BALANCE_REPO / CP_BALANCE_BASE_REPO 可覆盖）。
set -o pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

FAMILY=${CP_BALANCE_FAMILY:-a5}
DIAG_ONLY=0
declare -A SKIP=()
HX_PY="python3 $HERE/tests/lib/hx.py"

stages_tsv=$($HX_PY stages 2>/dev/null)
[ -n "$stages_tsv" ] || { echo "verify: 读不到 harness.json 的 verify.stages（文件缺失或损坏？）" >&2; exit 2; }
KNOWN_SKIP_FLAGS=$(printf '%s\n' "$stages_tsv" | cut -d'|' -f2 | tr '\n' ' ')

while [ $# -gt 0 ]; do
  case "$1" in
    --family)    FAMILY=${2:-}; shift 2 ;;
    --diag-only) DIAG_ONLY=1; shift ;;
    --live-log)  HX_LIVE_LOG=1; shift ;;
    -h|--help)   sed -n '2,13p' "$0"; exit 0 ;;
    --skip-*)
      case " $KNOWN_SKIP_FLAGS " in
        *" $1 "*) SKIP[$1]=1 ;;
        *) echo "verify: 未知的跳过开关 $1（可选：$KNOWN_SKIP_FLAGS）" >&2; exit 2 ;;
      esac
      shift ;;
    *) echo "verify: unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
done

FAMILIES=$($HX_PY families | cut -f1 | tr '\n' ' ')
case " $FAMILIES " in
  *" $FAMILY "*) ;;
  *) echo "verify: --family 只能是 $FAMILIES（当前 '$FAMILY'）" >&2; exit 2 ;;
esac
export CP_BALANCE_FAMILY=$FAMILY
# 远端有 http_proxy 时，本机地址必须绕开代理（否则就绪探测被发到代理）
no_proxy="127.0.0.1,localhost,::1${no_proxy:+,$no_proxy}"; export no_proxy
NO_PROXY="$no_proxy"; export NO_PROXY
export HX_LIVE_LOG=${HX_LIVE_LOG:-0}

STAMP=$(date +%m%d_%H%M%S)_$$
PREFIX=$($HX_PY harness verify.artifact_prefix)
[ -n "$PREFIX" ] || PREFIX=verify
REPORT="$HERE/${PREFIX}_${FAMILY}_branch_$STAMP.txt"
VERIFY_OUT="$HERE/tests/_out/${STAMP}_$FAMILY"
FAILED_STAGES=()
SKIPPED_STAGES=()

step() { printf '\n===== %s =====\n' "$*"; }

if [ "$DIAG_ONLY" = 1 ]; then
  VERIFY_OUT=$(ls -1dt "$HERE"/tests/_out/*/ 2>/dev/null | head -1)
  VERIFY_OUT=${VERIFY_OUT%/}
fi

# ---------------- 0. 前置检查 ----------------
[ "${HX_LIVE_LOG:-0}" = "1" ] && printf "[note] live-log：测试与模型服务日志实时打屏（同时写入 tests/_out）\n"
step "0. 前置检查"
missing=()
[ -n "${CP_BALANCE_LOCAL_IP:-}" ] || missing+=(CP_BALANCE_LOCAL_IP)
[ -n "${CP_BALANCE_NIC_NAME:-}" ] || missing+=(CP_BALANCE_NIC_NAME)
if [ ${#missing[@]} -gt 0 ]; then
  hx_note_unset="未设置 ${missing[*]}：尝试自动识别（容器内没有 ip/ifconfig 也能用）"
  printf '[note] %s\n' "$hx_note_unset"
  BEST=$(python3 tests/lib/netif.py --best 2>/dev/null || true)
  if [ -n "$BEST" ]; then
    DET_NIC=$(printf '%s' "$BEST" | cut -f1)
    DET_IP=$(printf '%s' "$BEST" | cut -f2)
    [ -n "${CP_BALANCE_NIC_NAME:-}" ] || export CP_BALANCE_NIC_NAME="$DET_NIC"
    [ -n "${CP_BALANCE_LOCAL_IP:-}" ] || export CP_BALANCE_LOCAL_IP="$DET_IP"
    printf '[note] 识别到 NIC=%s IP=%s\n' "$CP_BALANCE_NIC_NAME" "$CP_BALANCE_LOCAL_IP"
    printf '[note] 全部候选（网卡 / IP / default=默认路由接口）：\n'
    python3 tests/lib/netif.py | sed 's/^/       /'
  else
    printf '[FAIL] 没有识别到可用 IPv4 网卡\n'
    printf '  容器里请显式指定（候选可用 python3 tests/lib/netif.py 查看）：\n'
    printf '    export CP_BALANCE_LOCAL_IP=<本机 IP>\n'
    printf '    export CP_BALANCE_NIC_NAME=<网卡名>\n'
    exit 2
  fi
fi

[ -f tests/run_tests.sh ] || { echo "[FAIL] tests/run_tests.sh 不存在" >&2; exit 2; }
[ -f tests/lib/roles.tsv ] || { echo "[FAIL] tests/lib/roles.tsv 不存在" >&2; exit 2; }
[ -f harness.json ] || { echo "[FAIL] harness.json 不存在（树清单/阶段/判据都在里面）" >&2; exit 2; }

. "$HERE/tests/lib/common.sh"

SVC_ROLE=$($HX_PY harness "families.$FAMILY.svc_role")
[ -n "$SVC_ROLE" ] || { echo "[FAIL] harness.json families.$FAMILY.svc_role 没配" >&2; exit 2; }
SVC_CFG=$(hx_role "$SVC_ROLE") || { echo "[FAIL] roles.tsv 里 $FAMILY 没有角色 $SVC_ROLE" >&2; exit 2; }
CUR_REPO=${CP_BALANCE_REPO:-$(hx_cur_repo)}
BASE_REPO=${CP_BALANCE_BASE_REPO:-$(hx_base_repo)}
MODEL=$(hx_cfg_field "$SVC_CFG" model)
PORT=$(hx_cfg_field "$SVC_CFG" port)

hx_note "family=$FAMILY ($($HX_PY harness "families.$FAMILY.label")) 配置=$SVC_CFG port=$PORT"
hx_note "IP=$CP_BALANCE_LOCAL_IP NIC=$CP_BALANCE_NIC_NAME DEVICES=${CP_BALANCE_DEVICES:-<配置默认>}"
hx_note "被测树=$CUR_REPO"
hx_note "对照树=$BASE_REPO"
# 代码版本对齐：树不在就 clone，版本不对就切（tests/lib/sync_tree.sh，清单在 harness.json trees）
. "$HERE/tests/lib/sync_tree.sh"
roles_seen=0
if [ "${CP_BALANCE_AUTO_CHECKOUT:-1}" = "1" ]; then
  while IFS=$'\t' read -r role _tpath _tremote _tref _tkind _treq _texists; do
    [ -n "$role" ] || continue
    roles_seen=$((roles_seen + 1))
    override=""
    case "$role" in
      cur)  override=$CUR_REPO ;;
      base) override=$BASE_REPO ;;
    esac
    if ! hx_ensure_tree "$role" "$override"; then
      hx_fail "代码树就绪失败：$role"
    fi
  done < <(hx_tree_roles)
  [ "$roles_seen" -gt 0 ] || hx_fail "harness.json trees 里没有任何角色（拉不到树清单）"
  if [ -d "$HERE/.git" ] && [ -z "$(git -C "$HERE" status --porcelain 2>/dev/null | head -5)" ]; then
    branch=$($HX_PY harness verify.self_update.branch)
    if [ -n "$branch" ] && git -C "$HERE" fetch --quiet origin "$branch" 2>/dev/null \
       && [ "$(git -C "$HERE" rev-parse HEAD)" != "$(git -C "$HERE" rev-parse "origin/$branch" 2>/dev/null)" ]; then
      hx_note "harness 落后 origin/$branch，执行 git pull --ff-only"
      git -C "$HERE" pull --ff-only --quiet || hx_warn "harness 自更新失败，继续用当前版本"
    fi
  fi
else
  hx_note "CP_BALANCE_AUTO_CHECKOUT=0：只显示当前版本，不自动拉取/切换"
fi

hx_need_dir "被测代码树" "$CUR_REPO"
hx_need_dir "对照代码树" "$BASE_REPO"
hx_need_dir "权重" "$MODEL"
if [ -n "$MODEL" ] && [ ! -d "$MODEL" ]; then
  cand=$(python3 "$HERE/tests/lib/trees.py" model-candidates 2>/dev/null | head -5)
  if [ -n "$cand" ]; then
    hx_note "本机找到这些权重候选（可用 CP_BALANCE_MODEL=<路径> 覆盖）："
    printf '%s\n' "$cand" | sed 's/^/       /'
  fi
fi

for r in "$CUR_REPO" "$BASE_REPO"; do
  [ -n "$r" ] && [ -d "$r/.git" ] && hx_note "$r -> $(git -C "$r" log -1 --oneline 2>/dev/null)"
done
if [ -n "$CUR_REPO" ] && [ -f "$CUR_REPO/vllm_ascend/envs.py" ]; then
  grep -q VLLM_ASCEND_CP_BALANCE "$CUR_REPO/vllm_ascend/envs.py" \
    && hx_ok "被测树含 VLLM_ASCEND_CP_BALANCE（移植版）" \
    || hx_fail "被测树没有 CP_BALANCE：不是移植后的树（应为 cp_balance 分支）"
fi
if [ -n "$BASE_REPO" ] && [ -f "$BASE_REPO/vllm_ascend/envs.py" ] \
   && grep -q VLLM_ASCEND_CP_BALANCE "$BASE_REPO/vllm_ascend/envs.py"; then
  hx_fail "对照树里也有 CP_BALANCE：对照树应是原版 main"
fi
# 构建产物：_build_info.py 是 setup.py 在安装/构建时生成的（只有一行 __device_type__，只跟芯片型号有关），
# 它不进版本库，所以新 clone / 换机器都没有它；而 vllm_ascend 在 import 阶段就要读它 → 服务直接 ImportError。
# 注意：只改 py 不需要重编译；这里缺的是生成物，只要让这个文件存在即可。
if [ -n "$CUR_REPO" ] && [ -f "$CUR_REPO/vllm_ascend/__init__.py" ] \
   && [ ! -f "$CUR_REPO/vllm_ascend/_build_info.py" ]; then
  hx_warn "被测树缺生成物 vllm_ascend/_build_info.py（只有一行 __device_type__，import 阶段就要用）"
  auto=${CP_BALANCE_AUTO_BUILD:-0}
  if [ "$auto" = "copy" ]; then
    src=""
    while IFS=$'\t' read -r role tpath _r _f _k _q _e; do
      [ -n "$role" ] || continue
      [ "$role" = "cur" ] && continue
      if [ -f "$tpath/vllm_ascend/_build_info.py" ]; then src="$tpath"; break; fi
    done < <(hx_tree_roles)
    if [ -n "$src" ]; then
      cp "$src/vllm_ascend/_build_info.py" "$CUR_REPO/vllm_ascend/_build_info.py" \
        && hx_ok "已从同芯片的兄弟树拷 _build_info.py：$src" \
        || hx_fail "拷贝 _build_info.py 失败"
    else
      hx_fail "CP_BALANCE_AUTO_BUILD=copy 但没找到带 _build_info.py 的兄弟树"
    fi
  elif [ "$auto" = "1" ]; then
    build_cmd=$($HX_PY harness verify.auto_build.command)
    build_env=$($HX_PY harness verify.auto_build.env_script)
    [ -n "$build_cmd" ] || build_cmd="pip install -e . --no-build-isolation"
    prelude=$($HX_PY path "$SVC_CFG" prelude)
    [ -n "$prelude" ] && build_env=$prelude
    hx_note "CP_BALANCE_AUTO_BUILD=1：在 $CUR_REPO 里执行 $build_cmd（首次 clone 才需要，几分钟；顺带生成 C 扩展）"
    (
      cd "$CUR_REPO" || exit 1
      [ -n "$build_env" ] && [ -f "$build_env" ] && . "$build_env"
      $build_cmd
    ) >"$HERE/${PREFIX}_build_$STAMP.log" 2>&1 \
      && hx_ok "构建完成（日志 ${PREFIX}_build_$STAMP.log）" \
      || hx_fail "自动构建失败，看 ${PREFIX}_build_$STAMP.log"
    [ -f "$CUR_REPO/vllm_ascend/_build_info.py" ] && hx_ok "构建产物已生成"
  else
    hx_note "只改过 py 的话不需要重编译，只要补这个文件：三选一"
    hx_note "  1) 从同芯片的兄弟树拷（最快，内容只有一行）：cp <另一棵树>/vllm_ascend/_build_info.py $CUR_REPO/vllm_ascend/"
    hx_note "  2) 在本树跑一次安装（首次 clone 推荐，顺带生成 vllm_ascend_C*.so）：cd $CUR_REPO && source <CANN>/set_env.sh && pip install -e . --no-build-isolation"
    hx_note "  3) 让脚本自动做：CP_BALANCE_AUTO_BUILD=copy（拷兄弟树）或 =1（跑 pip install）"
  fi
  if [ -f "$CUR_REPO/vllm_ascend/_build_info.py" ]; then
    hx_ok "生成物已就位（只改过 py 的话不需要重编译）"
  else
    hx_fail "生成物仍缺失：服务会在 import 阶段 ImportError，按上面的修法补上再跑"
  fi
fi
if [ -n "$CUR_REPO" ] && [ -d "$CUR_REPO/vllm_ascend" ]; then
  compgen -G "$CUR_REPO/vllm_ascend/vllm_ascend_C*.so" >/dev/null \
    || hx_warn "被测树里没有 vllm_ascend_C 扩展（后续报自定义算子缺失时重新构建）"
fi

hx_note "vllm 版本: $(python3 -c 'import vllm;print(vllm.__version__)' 2>&1 | tail -1)"
hx_note "SOC: $(python3 -c 'import torch_npu;print(torch_npu.npu.get_soc_version())' 2>&1 | tail -1)"
df -h "$HERE" | tail -1

if [ "$HX_FAILED" -ne 0 ] && [ "$DIAG_ONLY" -eq 0 ]; then
  hx_fail "前置检查未通过，后续大概率失败；修完再跑（--diag-only 可只抓日志）"
  exit 2
fi
if [ "$DIAG_ONLY" -eq 1 ]; then
  hx_note "diag-only：前置检查的 FAIL 不计入结论（只对已有证据出诊断）"
  HX_FAILED=0
fi

# ---------------- 1..n 阶段（harness.json verify.stages） ----------------
stage_diagnose() {
  local out=$VERIFY_OUT
  if [ "$DIAG_ONLY" -eq 0 ]; then
    for skipped in "${SKIPPED_STAGES[@]:-}"; do
      case "$skipped" in
        *smoke*) hx_note "smoke 被跳过：没有服务日志可诊断（--diag-only 可对最近一轮出报告）"; return 0 ;;
      esac
    done
  fi
  [ -n "$out" ] && [ -d "$out" ] || { hx_warn "没有证据目录，跳过诊断"; return 0; }
  hx_note "证据目录：$out"
  python3 "$HERE/tests/lib/diagnose.py" report "$out" "$REPORT"
  case "$?" in
    0) hx_ok "出现 zigzag 证据，可以往精度/性能走" ;;
    3) hx_fail "命中的是一个已知失败（见报告末尾 VERDICT=KNOWN）"; FAILED_STAGES+=(diagnose) ;;
    *) hx_fail "没有任何 zigzag 证据（报告：$REPORT）"; FAILED_STAGES+=(diagnose) ;;
  esac
}

stage_pack() {
  local tar="$HERE/${PREFIX}_${FAMILY}_$STAMP.tar" items=()
  [ -f "$REPORT" ] && items+=("${REPORT#"$HERE"/}")
  [ -n "$VERIFY_OUT" ] && [ -d "$VERIFY_OUT" ] && items+=("${VERIFY_OUT#"$HERE"/}")
  if [ ${#items[@]} -eq 0 ]; then
    hx_warn "没有可打包的产物"
    return 0
  fi
  tar cf "$tar" -C "$HERE" "${items[@]}" || { hx_fail "tar 打包失败"; return 1; }
  if tar tf "$tar" | grep -q .; then
    gzip -f "$tar"
    hx_ok "证据包：$tar.gz ($(du -h "$tar.gz" 2>/dev/null | cut -f1))"
  else
    hx_fail "证据包是空的"
    return 1
  fi
}

while IFS='|' read -r id skip_flag builtin title; do
  [ -n "$id" ] || continue
  step "$title"
  if [ "$DIAG_ONLY" = 1 ]; then
    case "$builtin" in
      diagnose|pack) ;;
      *) hx_note "diag-only：跳过 $id"; SKIPPED_STAGES+=("$id"); continue ;;
    esac
  fi
  if [ -n "$skip_flag" ] && [ -n "${SKIP[$skip_flag]:-}" ]; then
    hx_note "按 $skip_flag 跳过 $id"
    SKIPPED_STAGES+=("$id")
    continue
  fi
  case "$builtin" in
    diagnose) stage_diagnose ;;
    pack)     stage_pack ;;
    *)
      cmd=$($HX_PY stage-run "$id") || { hx_fail "stage $id 没有 run 命令"; continue; }
      cmd=${cmd//\{out\}/$VERIFY_OUT}
      if bash -c "$cmd"; then
        hx_ok "$id 通过"
      else
        hx_fail "$id 失败（日志在 $VERIFY_OUT/$id 下）"
        FAILED_STAGES+=("$id")
      fi ;;
  esac
done < <(printf '%s\n' "$stages_tsv")

step "汇总"
if [ "$HX_FAILED" -eq 0 ]; then
  hx_ok "一键验证：PASS（family=$FAMILY）"
else
  hx_fail "一键验证：FAIL（stage: ${FAILED_STAGES[*]:-preflight}）"
fi

cat <<EOF

下一步：
  1) 诊断没通过先看它的 VERDICT：KNOWN 行会直接指出是哪个门（例如 Disabling DSA-CP）；
  2) zigzag 证据出现后：bash tests/run_tests.sh --only accuracy/a10_matrix_gate --keep-going
  3) 性能：bash tests/run_tests.sh --only perf/p10_capture --skip perf/p10_capture#prof_a2a
  4) 把 ${PREFIX}_${FAMILY}_*.tar.gz 回传（原始 trace 不用拷）
EOF

[ "$HX_FAILED" -eq 0 ]
