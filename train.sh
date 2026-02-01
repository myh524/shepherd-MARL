export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=0 python -u onpolicy/scripts/train_mpe.py \
  --use_valuenorm --use_popart \
  --project_name "informarl_debug" \
  --env_name "GraphMPE" \
  --algorithm_name "rmappo" \
  --seed 0 \
  --experiment_name "1v1_02_01_06" \
  --scenario_name "navigation_graph" \
  --num_agents 1 \
  --collision_rew 3 \
  --n_training_threads 1 \
  --n_rollout_threads 32 \
  --num_mini_batch 2 \
  --episode_length 300 \
  --num_env_steps 600000 \
  --ppo_epoch 5 \
  --use_ReLU \
  --lr 3e-4 --critic_lr 1e-3 \
  --max_grad_norm 0.5 \
  --user_name "marl" \
  --use_cent_obs True \
  --graph_feat_type relative \
  --auto_mini_batch_size False \
  --target_mini_batch_size 64 \
  --use_wandb False


# #!/usr/bin/env bash
# set -e

# ############################
# # 基本配置
# ############################
# SESSION_NAME="train_mpe"
# CONDA_ENV_NAME="informarl38"     # ← 改成你的 conda 环境名
# CUDA_VISIBLE_DEVICES=0

# LOG_DIR="logs"
# LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

# ############################
# # 准备环境
# ############################
# echo ">>> Checking GPU status..."
# nvidia-smi || { echo "nvidia-smi failed"; exit 1; }

# echo ">>> Activating conda environment: ${CONDA_ENV_NAME}"
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate "${CONDA_ENV_NAME}"

# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}

# mkdir -p "${LOG_DIR}"

# ############################
# # 检查 tmux 会话
# ############################
# if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
#   echo ">>> tmux session '${SESSION_NAME}' already exists"
#   echo ">>> Attach with: tmux attach -t ${SESSION_NAME}"
#   exit 0
# fi

# ############################
# # 启动 tmux 训练
# ############################
# echo ">>> Starting tmux session: ${SESSION_NAME}"
# tmux new-session -d -s "${SESSION_NAME}"

# tmux send-keys -t "${SESSION_NAME}" "
# echo '>>> Training started at: \$(date)';
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -u onpolicy/scripts/train_mpe.py \
#   --use_valuenorm --use_popart \
#   --project_name 'informarl_debug' \
#   --env_name 'GraphMPE' \
#   --algorithm_name 'rmappo' \
#   --seed 0 \
#   --experiment_name 'experiment01_24_01' \
#   --scenario_name 'navigation_graph' \
#   --num_agents 2 \
#   --collision_rew 3 \
#   --n_training_threads 1 \
#   --n_rollout_threads 4 \
#   --num_mini_batch 4 \
#   --episode_length 50 \
#   --num_env_steps 400000 \
#   --ppo_epoch 5 \
#   --use_ReLU \
#   --lr 3e-4 --critic_lr 1e-3 \
#   --max_grad_norm 0.5 \
#   --user_name 'marl' \
#   --use_cent_obs True \
#   --graph_feat_type relative \
#   --auto_mini_batch_size False \
#   --target_mini_batch_size 64 \
#   --use_wandb False \
#   > ${LOG_FILE} 2>&1;
# echo '>>> Training finished at: \$(date)'
# " C-m

# echo ">>> Training launched successfully!"
# echo ">>> tmux attach -t ${SESSION_NAME}"
# echo ">>> log file: ${LOG_FILE}"
