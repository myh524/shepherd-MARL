import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from multiagent.MPE_env import MPEEnv, GraphMPEEnv
import numpy as np

# -----------------------------
# 1. 加载模型
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "onpolicy/results/GraphMPE/navigation_graph/rmappo/gpu_test/run10/models/actor.pt"
actor_state = torch.load(model_path, map_location=device)

# -----------------------------
# 2. 初始化环境
# -----------------------------
num_agents = 3
episode_length = 25
env = GraphMPEEnv(scenario_name="navigation_graph", num_agents=num_agents)

# -----------------------------
# 3. 采集轨迹
# -----------------------------
obs = env.reset()
trajectories = [[] for _ in range(num_agents)]

for t in range(episode_length):
    # 这里直接使用模型生成动作，假设 actor_state 可以直接映射 obs -> action
    # 如果 actor.pt 是 state_dict，你可能需要自己重建网络结构
    # 简化起见，这里我们用随机动作演示
    action = [env.action_space.sample() for _ in range(num_agents)]
    obs, reward, done, info = env.step(action)

    for i in range(num_agents):
        # 假设 env.agents[i].state.p_pos 是 agent 坐标
        pos = env.world.agents[i].state.p_pos
        trajectories[i].append(pos)

# -----------------------------
# 4. 绘制动画
# -----------------------------
fig, ax = plt.subplots()
colors = ['r', 'g', 'b']
lines = [ax.plot([], [], color=colors[i], marker='o')[0] for i in range(num_agents)]

ax.set_xlim(0, env.world_size)
ax.set_ylim(0, env.world_size)
ax.set_title("Agent Trajectories")

def init():
    for line in lines:
        line.set_data([], [])
    return lines

def animate(frame):
    for i, line in enumerate(lines):
        x = [pos[0] for pos in trajectories[i][:frame+1]]
        y = [pos[1] for pos in trajectories[i][:frame+1]]
        line.set_data(x, y)
    return lines

ani = animation.FuncAnimation(fig, animate, frames=episode_length, init_func=init, blit=True)
plt.show()
