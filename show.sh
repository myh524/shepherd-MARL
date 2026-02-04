# tensorboard --logdir ./onpolicy/results/GraphMPE/navigation_graph/rmappo/1v1_01_31_06/run2/logs

# onpolicy/results/GraphMPE/navigation_graph/rmappo/1v1_01_31_05/run1/models \          #可以用
# onpolicy/results/GraphMPE/navigation_graph/rmappo/1v1_02_01_06/run4/models \          #可以用(效果：完成率可以，但是还是不能稳定地在正背后推羊，改进:rew_push = 2.5 * (cos_push ** 3))

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python onpolicy/scripts/eval_mpe.py \
    --env_name GraphMPE \
    --scenario_name navigation_graph \
    --num_agents 1 \
    --model_dir onpolicy/results/GraphMPE/navigation_graph/rmappo/1v1_02_03_01/run5/models \
    --episode_length 1000 \      





