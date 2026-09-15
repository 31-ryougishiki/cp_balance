# 自动获取配置
nic_name=eth0
local_ip=141.61.133.112

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

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TASK_QUEUE_ENABLE=1

dir="$PWD/$(date +%Y%m%d_%H%M%S)/plog" && mkdir -p "$dir"
export ASCEND_PROCESS_LOG_PATH="$dir"

export VLLM_USE_FASTOKENS=1
source /mnt/share/l00622059/vendors/custom_transformer/bin/set_env.bash

# 缓存禁用
export VLLM_DISABLE_COMPILE_CACHE=1
export PYTHONPATH=/home/z30055003/vllm-ascend:${PYTHONPATH}

vllm serve /mnt/share/weights/GLM-5.2-w4a4c8-mxfp4 \
  --host 0.0.0.0 \
  --port $2 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --distributed-executor-backend mp \
  --max_model_len 135000 \
  --max-num-batched-tokens 16384 \
  --served-model-name glm \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 500 \
  --trust-remote-code \
  --enforce-eager \
  --quantization ascend \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --async-scheduling \
  --no-enable-prefix-caching \
  --speculative-config '{"num_speculative_tokens": 1,"method": "deepseek_mtp", "enforce_eager":true}' \
  --additional_config '{"enable_cpu_binding": "true", "multistream_overlap_shared_expert": "true", "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "enable_dsa_cp": true}'