import gym
from gym import spaces
import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import pathlib
from compas_assembly.datastructures import Assembly
from compas_assembly.algorithms import assembly_interfaces_numpy
from compas_rbe.equilibrium import compute_interface_forces_cvx

class TrainingCallback(BaseCallback):
    def __init__(self, check_freq):
        super(TrainingCallback, self).__init__()
        self.check_freq = check_freq
        self.rewards = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self.rewards.append(self.model.get_env().get_attr("reward", 0)[0])
        return True

class CustomEnv(gym.Env):
    """自定义环境，继承自gym.Env"""

    # Visualize the assembly
    def load_and_process_assembly(file_path):
        # Load the assembly from JSON
        assembly = Assembly.from_json(file_path)
        # Compute the assembly interfaces
        assembly_interfaces_numpy(assembly, nmax=10, amin=0.0001)

        # Compute the equilibrium geometry
        compute_interface_forces_cvx(assembly, solver='CPLEX', verbose=True)

        return assembly

    def __init__(self, assembly):
        super(CustomEnv, self).__init__()
        self.assembly = assembly
        self.nodes = list(assembly.nodes())
        self.num_nodes = len(self.nodes)
        self.forces_cache = {}  # 初始化forces_cache

        # 定义动作空间和观察空间
        self.action_space = spaces.MultiDiscrete([self.num_nodes, self.num_nodes])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_nodes, 3), dtype=np.float32)
        self.current_state = self._next_observation()

    def step(self, action):
        node_to_move, new_position_index = action
        if 0 <= node_to_move < self.num_nodes and 0 <= new_position_index < self.num_nodes:
            moved_node = self.nodes.pop(node_to_move)
            self.nodes.insert(new_position_index, moved_node)
        else:
            reward = -10
            return self.current_state, reward, True, {}

        reward = self.calculate_reward()
        self.current_state = self._next_observation()
        done = False
        info = {}
        return self.current_state, reward, done, info

    def reset(self):
        np.random.shuffle(self.nodes)
        self.current_state = self._next_observation()
        return self.current_state

    def _next_observation(self):
        # 获取assembly对象中所有节点的信息
        observation = []
        for node in self.nodes:
            block = self.assembly.graph.node_attribute(node, 'block')
            centroid = block.centroid()
            observation.append([centroid[0], centroid[1], centroid[2]])
        
        return np.array(observation)


    def calculate_reward(self):
        # 根据您的獎勵和懲罰設計计算奖励
        reward = 0
        
        # 1. 根据interface的各种force计算奖励
        forces = self.calculate_forces_on_interfaces()
        reward += -sum(forces.values())  # 如果力量减少，则增加奖励

        # 2. 计算nodes间的距离，奖励距离减少
        reward += -self.calculate_total_distance()

        # 3. 根据是否为支撑点进行奖励
        for node in self.nodes:
            if self.graph.nodes[node]['is_support']:
                reward += 10

        return reward

    def calculate_forces_on_interfaces(self):
        # 计算接口上的各种力
        total_forces = {"contact": 0, "compression": 0, "tension": 0, "friction": 0, "resultant": 0}
        for node, attr in self.assembly.graph.nodes(data=True):
            interface = attr.get('interface')
            if interface:
                if interface not in self.forces_cache:
                    # 计算并缓存结果
                    self.forces_cache[interface] = {
                        "contact": sum(interface.contactforces()),
                        "compression": sum(interface.compressionforces()),
                        "tension": sum(interface.tensionforces()),
                        "friction": sum(interface.frictionforces()),
                        "resultant": sum(interface.resultantforce())
                    }
                # 从缓存中获取结果
                total_forces["contact"] += self.forces_cache[interface]["contact"]
                total_forces["compression"] += self.forces_cache[interface]["compression"]
                total_forces["tension"] += self.forces_cache[interface]["tension"]
                total_forces["friction"] += self.forces_cache[interface]["friction"]
                total_forces["resultant"] += self.forces_cache[interface]["resultant"]
        return total_forces

    def calculate_total_distance(self):
        # 修改此方法以直接从assembly对象获取位置信息
        total_distance = 0
        for i in range(len(self.nodes) - 1):
            node_i = self.nodes[i]
            node_j = self.nodes[i + 1]
            pos_i = self.assembly.graph.node_attribute(node_i, 'block').centroid()
            pos_j = self.assembly.graph.node_attribute(node_j, 'block').centroid()
            total_distance += np.linalg.norm(np.array(pos_i) - np.array(pos_j))
        return total_distance

    def calculate_loss(self, assembly):
        # 计算损失
        loss = 0
        for node, attr in assembly.graph.nodes(data=True):
            interface = attr.get('interface')
            if interface:
                loss += sum(interface.contactforces()) + sum(interface.compressionforces()) + sum(interface.tensionforces()) + sum(interface.frictionforces()) + sum(interface.resultantforce())
        return loss

    def render(self, mode='human'):
        if mode == 'human':
            plt.figure(figsize=(8, 6))
            for node in self.nodes:
                x, y, z = self.graph.nodes[node]['position']
                plt.scatter(x, y, c='blue')
                plt.text(x, y, str(node), fontsize=12)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Current Nodes Configuration')
            plt.show()

    def close(self):
        pass

# 载入组装
CWD = pathlib.Path(__file__).parent.absolute()
FILE = CWD.parent.parent / "output" / "assembly_interface_from_rhino_test_2.json"
assembly = Assembly.from_json(FILE)

# 创建环境实例
env = CustomEnv(assembly)

callback = TrainingCallback(check_freq=1000)

model = PPO("MlpPolicy", env, device='cuda', verbose=1)

model.learn(total_timesteps=10000, callback=callback)

# 绘制累积奖励
plt.plot(callback.rewards)
plt.title('Cumulative Reward Over Time')
plt.xlabel('Training Steps')
plt.ylabel('Cumulative Reward')
plt.show()

# 保存模型
model.save("ppo_model")