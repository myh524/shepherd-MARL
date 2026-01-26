# CUDA_VISIBLE_DEVICES=0 xvfb-run -a -s "-screen 0 1400x900x24" python onpolicy/scripts/train_mpe.py \
#     --env_name GraphMPE \
#     --scenario_name navigation_graph \
#     --algorithm_name rmappo \
#     --experiment_name gpu_test \
#     --num_agents 3 \
#     --episode_length 25 \
#     --use_render True \
#     --save_gifs True \
#     --render_episodes 5 \
#     --n_render_rollout_threads 1 \
#     --model_dir onpolicy/results/GraphMPE/navigation_graph/rmappo/gpu_test/run10/models \
#     --use_wandb False \
#     --num_env_steps 0

# tensorboard --logdir ./onpolicy/results/GraphMPE/navigation_graph/rmappo/experiment01_24_03/run3/logs

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python onpolicy/scripts/eval_mpe.py \
    --env_name GraphMPE \
    --scenario_name navigation_graph \
    --num_agents 1 \
    --model_dir onpolicy/results/GraphMPE/navigation_graph/rmappo/experiment01_24_03/run1/models \
    --episode_length 1000 \      





