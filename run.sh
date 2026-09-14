# 自动获取配置
nic_name=eth2
local_ip=7.246.78.75

# 以下环境变量无需修改
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export HCCL_ALGO=level0:fullmesh

export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=180
export HCCL_BUFFSIZE=1200

export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export TASK_QUEUE_ENABLE=1

# 缓存禁用
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_ASCEND_REPO=${VLLM_ASCEND_REPO:-/opt/its/z30055003/vllm-ascend}
export PYTHONPATH=${VLLM_ASCEND_REPO}:${PYTHONPATH}

export VLLM_ASCEND_CP_BALANCE=${VLLM_ASCEND_CP_BALANCE:-1}
export VLLM_ASCEND_CP_BALANCE_MIN_TOKENS=${VLLM_ASCEND_CP_BALANCE_MIN_TOKENS:-2048}
export VLLM_ASCEND_CP_BALANCE_REDUCE_MODE=${VLLM_ASCEND_CP_BALANCE_REDUCE_MODE:-allreduce}

echo "[cp_balance] REPO=${VLLM_ASCEND_REPO} CP_BALANCE=${VLLM_ASCEND_CP_BALANCE} MIN_TOKENS=${VLLM_ASCEND_CP_BALANCE_MIN_TOKENS} REDUCE_MODE=${VLLM_ASCEND_CP_BALANCE_REDUCE_MODE} DEBUG=${VLLM_ASCEND_CP_BALANCE_DEBUG:-0}"
vllm serve /opt/its/model/GLM-5.2-W4A8C8 \
  --host 0.0.0.0 \
  --port $2 \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --distributed-executor-backend mp \
  --max_model_len 135000 \
  --max-num-batched-tokens 16384 \
  --served-model-name glm glm-52 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 32 \
  --trust-remote-code \
  --enforce-eager \
  --quantization ascend \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --async-scheduling \
  --no-enable-prefix-caching \
  --additional_config '{"enable_cpu_binding": "true", "multistream_overlap_shared_expert": "true", "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "enable_dsa_cp": true}'