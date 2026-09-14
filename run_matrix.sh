#!/usr/bin/env bash
# cp_balance 四组实验 + 三组对比（串行执行，避免 16 卡互抢）
#
#   R1 cur_off    当前分支 CP_BALANCE=0      <- 与 base 的等价性主判据
#   R2 cur_on     当前分支 CP_BALANCE=1      <- 精度验收（C vs B）
#   R3 base_off   base 分支（原版 DSA-CP）
#   R4 base_off2  base 分支重复一次          <- 噪声地板
#
# usage: bash run_matrix.sh <nic> [base_port] [det]
#   det=1（默认）导出 vllm-ascend 文档推荐的确定性环境变量；det=0 关闭
# 可选环境变量：
#   VLLM_ASCEND_REPO_CUR / VLLM_ASCEND_REPO_BASE / RUNS / OUT_DIR / READY_TIMEOUT
set -o pipefail

NIC=$1
PORT0=$2
DET=$3
[ -n "$NIC" ] || NIC=eth2
[ -n "$PORT0" ] || PORT0=8034
[ -n "$DET" ] || DET=1

REPO_CUR=$VLLM_ASCEND_REPO_CUR
[ -n "$REPO_CUR" ] || REPO_CUR=/opt/its/z30055003/vllm-ascend
REPO_BASE=$VLLM_ASCEND_REPO_BASE
[ -n "$REPO_BASE" ] || REPO_BASE=/opt/its/z30055003/vllm-ascend-base
RUNS=$RUNS
[ -n "$RUNS" ] || RUNS=cur_off,cur_on,base_off,base_off2
OUT=$OUT_DIR
[ -n "$OUT" ] || OUT=matrix_$(date +%m%d_%H%M)
TIMEOUT=$READY_TIMEOUT
[ -n "$TIMEOUT" ] || TIMEOUT=1800

set -u

mkdir -p "$OUT"
SUMMARY=$OUT/summary.txt
: > "$SUMMARY"
log() { echo "[matrix] $*" | tee -a "$SUMMARY"; }
want() { case ",$RUNS," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

if [ "$DET" = "1" ]; then
  export LCCL_DETERMINISTIC=1
  export HCCL_DETERMINISTIC=true
  export ATB_MATMUL_SHUFFLE_K_ENABLE=0
  export ATB_LLM_LCOC_ENABLE=0
fi

log "nic=$NIC base_port=$PORT0 det=$DET out=$OUT"
log "det: LCCL_DETERMINISTIC=$(printenv LCCL_DETERMINISTIC || echo unset) HCCL_DETERMINISTIC=$(printenv HCCL_DETERMINISTIC || echo unset) ATB_MATMUL_SHUFFLE_K_ENABLE=$(printenv ATB_MATMUL_SHUFFLE_K_ENABLE || echo unset)"
log "repos: cur=$REPO_CUR base=$REPO_BASE"

log "step 0: static gate check"
python check_b_path.py --repo "$REPO_CUR" --base-repo "$REPO_BASE" 2>&1 | tee -a "$SUMMARY" | tail -2

stop_one() {
  local pid=$1
  kill -TERM -- -"$pid" 2>/dev/null
  local i
  for i in $(seq 1 60); do
    sleep 2
    kill -0 -- -"$pid" 2>/dev/null || break
  done
  kill -KILL -- -"$pid" 2>/dev/null
  sleep 10
}

run_one() {
  local name=$1
  local repo=$2
  local cp=$3
  local port=$4
  local logf=$OUT/$name.log
  log "run $name: repo=$repo CP_BALANCE=$cp port=$port"
  set -m
  env VLLM_ASCEND_REPO=$repo VLLM_ASCEND_CP_BALANCE=$cp VLLM_ASCEND_CP_BALANCE_DEBUG=1 PYTHONUNBUFFERED=1 bash run.sh "$NIC" "$port" >"$logf" 2>&1 &
  local pid=$!
  set +m

  local waited=0 ok=0
  while [ "$waited" -lt "$TIMEOUT" ]; do
    if curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then ok=1; break; fi
    if ! kill -0 -- -"$pid" 2>/dev/null; then log "  service exited early, see $logf"; break; fi
    sleep 10
    waited=$((waited + 10))
  done
  if [ "$ok" != "1" ]; then
    log "  NOT READY after $waited seconds"
    stop_one "$pid"
    return 1
  fi

  log "  ready after $waited seconds, collecting 40 prompts"
  python compare_first_token.py collect --url "http://127.0.0.1:$port" --out "$OUT/$name.json" >"$OUT/$name.collect.txt" 2>&1
  local rc=$?
  tail -1 "$OUT/$name.collect.txt" | tee -a "$SUMMARY"
  grep -F -m 1 "[cp_balance] REPO=" "$logf" >>"$SUMMARY" || log "  WARNING: no [cp_balance] fingerprint in $logf"
  local fixed native zig plan cont
  fixed=$(grep -cF "path=fixed_order" "$logf")
  native=$(grep -cF "path=native" "$logf")
  zig=$(grep -cF "branch=ZIGZAG" "$logf")
  plan=$(grep -cF "[CP_BALANCE][plan]" "$logf")
  cont=$(grep -cF "branch=CONTINUOUS" "$logf")
  log "  $name log: fixed_order=$fixed native=$native zigzag=$zig plan=$plan continuous=$cont"

  stop_one "$pid"
  return $rc
}

compare() {
  local label=$1 mode=$2 left=$3 right=$4
  if [ ! -f "$left" ] || [ ! -f "$right" ]; then
    log "skip compare $label (missing json)"
    return 2
  fi
  log "compare $label"
  python compare_first_token.py compare $mode "$left" "$right" >"$OUT/cmp_$label.txt" 2>&1
  local rc=$?
  grep "^[compare]" "$OUT/cmp_$label.txt" | tee -a "$SUMMARY"
  echo "  exit=$rc" >>"$SUMMARY"
  return $rc
}

want cur_off && { run_one cur_off "$REPO_CUR" 0 "$PORT0" || log "run cur_off FAILED"; }
want cur_on && { run_one cur_on "$REPO_CUR" 1 "$((PORT0 + 1))" || log "run cur_on FAILED"; }
want base_off && { run_one base_off "$REPO_BASE" 0 "$((PORT0 + 2))" || log "run base_off FAILED"; }
want base_off2 && { run_one base_off2 "$REPO_BASE" 0 "$((PORT0 + 3))" || log "run base_off2 FAILED"; }

R1=2
R2=2
R3=2
compare b_vs_base --require-text "$OUT/cur_off.json" "$OUT/base_off.json"
R1=$?
compare c_vs_b "" "$OUT/cur_on.json" "$OUT/cur_off.json"
R2=$?
compare noise_floor --require-text "$OUT/base_off.json" "$OUT/base_off2.json"
R3=$?

verdict() { if [ "$1" = "0" ]; then echo PASS; else echo FAIL; fi; }
log "---- verdict ----"
log "R1 cur_off == base_off   : $(verdict $R1)   (主判据: 非 cp_balance 与原版一致)"
log "R2 cur_on  == cur_off    : $(verdict $R2)   (C 验收: 首 token)"
log "R3 base_off == base_off2 : $(verdict $R3)   (噪声地板)"
if [ "$R3" != "0" ]; then
  log "注意: 噪声地板不为 0, 测量本身不可复现, 先确认 det=1 的确定性变量生效"
fi
if [ "$R1" = "0" ] && [ "$R3" = "0" ]; then
  log "RESULT: PASS"
else
  log "RESULT: FAIL"
fi
log "artifacts: $OUT (summary.txt, *.log, *.json, cmp_*.txt)"
