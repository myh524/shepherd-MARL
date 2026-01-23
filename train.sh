# CUDA_VISIBLE_DEVICES=0 python onpolicy/scripts/train_mpe.py \
#   --env_name GraphMPE \
#   --scenario_name navigation_graph \
#   --algorithm_name rmappo \
#   --experiment_name gpu_test \
#   --num_agents 3 \
#   --episode_length 25 \
#   --num_env_steps 200000 \
#   --use_wandb False

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=0 python -u onpolicy/scripts/train_mpe.py \
  --use_valuenorm --use_popart \
  --project_name "informarl_debug" \
  --env_name "GraphMPE" \
  --algorithm_name "rmappo" \
  --seed 0 \
  --experiment_name "debug_15min" \
  --scenario_name "navigation_graph" \
  --num_agents 3 \
  --collision_rew 5 \
  --n_training_threads 1 \
  --n_rollout_threads 32 \
  --num_mini_batch 1 \
  --episode_length 25 \
  --num_env_steps 300000 \
  --ppo_epoch 5 \
  --use_ReLU --gain 0.01 \
  --lr 7e-4 --critic_lr 7e-4 \
  --user_name "marl" \
  --use_cent_obs False \
  --graph_feat_type relative \
  --auto_mini_batch_size \
  --target_mini_batch_size 64 \
  --use_wandb False

