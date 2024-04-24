import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gym
from stable_baselines3 import PPO
from gym import spaces
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from compas.geometry import Point
from compas_view2.app import App
from compas_view2.objects import Pointcloud
import compas.datastructures as cd
import compas.geometry as cg
import json
import time
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)


parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)

args = parser.parse_args()

# 解析JSON數據
# TODO: 將JSON數據解析成可用於環境的數據結構
class BrickWall:
    def __init__(self, json_data):
        self.assembly_name = json_data["assembly"]["attributes"]["name"]
        self.bricks = self.parse_bricks(json_data["assembly"]["edge"])

    def parse_bricks(self, edge_data):
        bricks = []
        for edge_id, brick_data in edge_data.items():
            for interface_id, interface_info in brick_data.items():
                brick = {
                    "edge_id": edge_id,
                    "interface_id": interface_id,
                    "interface_forces": interface_info["interface_forces"],
                    "interface_origin": interface_info["interface_origin"],
                    "interface_points": interface_info["interface_points"],
                    "interface_size": interface_info["interface_size"],
                    "interface_type": interface_info["interface_type"],
                    "interface_uvw": interface_info["interface_uvw"]
                }
                bricks.append(brick)
        return bricks


# 使用JSON數據初始化BrickWall對象
root_dir = os.path.dirname(os.path.abspath(__file__))
filename = "assembly_interface_from_rhino.json"
json_data = os.path.join(root_dir, filename)

brick_wall = BrickWall(json_data)

# 訪問BrickWall對象中的數據
print("Assembly Name:", brick_wall.assembly_name)
print("Number of Bricks:", len(brick_wall.bricks))

for idx, brick in enumerate(brick_wall.bricks):
    print("----------------------------------------------------")
    print(f"Brick {idx + 1}:")
    print("Edge ID:", brick["edge_id"])
    print("Interface ID:", brick["interface_id"])
     
    print("Interface Size:", brick["interface_size"])
    print("Interface Type:", brick["interface_type"])
    print("Interface UVW:", brick["interface_uvw"])
    
    # 訪問其他數據字段
    print("Interface Forces:")
    for force in brick["interface_forces"]:
        print(" ---- c_nn:", force["c_nn"])
        print(" ---- c_np:", force["c_np"])
        print(" ---- c_u:", force["c_u"])
        print(" ---- c_v:", force["c_v"])
    
    print("Interface Points:")
    for point in brick["interface_points"]:
        print(" ---- Point:", point)
    
    print("Interface Origin:", brick["interface_origin"])

M = 3  # 觀察空間的維度
N = 6  # 動作空間的大小(上下前後左右六個方向)
input_size = M  # 神經網絡輸入的大小
output_size = N  # 神經網絡輸出的大小
learning_rate = 0.001  # 學習率
epochs = 1000  # 訓練迴圈的迭代次數
node_num = len(json_data["assembly"]["edge"]["i"]) # the node numbers in the graph in json_data

# 定義自定義環境
class CustomEnvironment(gym.Env):
    def __init__(self, json_data):
        # 初始化環境
        self.start_time = time.time()
        self.max_brick_num = node_num
        self.json_data = json_data
        self.current_state = None  # 當前狀態，根據需要初始化

        # 定義動作空間和觀察空間
        self.action_space = spaces.Discrete(N)  # 定義動作空間的大小（N是動作數量）
        self.observation_space = spaces.Box(low=0, high=1, shape=(M,))  # 定義觀察空間的形狀（M是觀察空間的維度）
        self.locations = [] # 假設環境具有位置數據，存儲在self.locations中

    def initial_state(self):
        # 返回初始狀態，可以根據需要初始化
        initial_state = {
            "brick_id": 0,  # 假設從第一個磚體開始
            "current_position": np.zeros(3),  # 假設初始位置為原點
            "downward_force": 0,  # 初始向下力為0
            "upward_force": 0  # 初始向上力為0
        }
        return initial_state
    
    def calculate_goal_positions(self):
        # 根據JSON數據中的信息計算每個磚體的目標位置
        goal_positions = {}
        for brick in self.json_data["bricks"]:
            # 假設JSON數據中包含目標位置的信息
            goal_positions[brick["id"]] = brick["goal_position"]
        return goal_positions
    
    def is_done(self):
        # 判斷是否終止
        # 在這裡實現終止條件的邏輯
        # 例如，如果疊加的磚體數量超過了最大疊加數量，則終止
        # 如果疊加的磚體數量沒有超過最大疊加數量，則不終止
        if self.current_state["brick_id"] > self.max_brick_num:
            return True
        else:
            return False
    
    def is_wall_complete(self):
        # 檢查是否疊加成功，即所有磚體都已疊加到其目標位置
        return all(np.allclose(self.current_state["current_position"], goal) for goal in self.goal_positions.values())


    def step(self, action):

        # 執行動作，更新狀態並返回獎勵、狀態等訊息
        # 在這裡實現環境的動力學模型，根據動作更新狀態，計算獎勵等
        # 返回觀察、獎勵、是否終止、其他信息（可根據需要自定義）

        # 假設以下是示例代碼，請根據您的環境需求自定義
        current_state = self.current_state  # 當前狀態

        # 假設根據action更新狀態，例如更新疊加位置等
        # 假設有一個函數update_state，根據action更新狀態
        updated_state = self.update_state(current_state, action)

        reward = self.calculate_reward(updated_state, action)  # 自定義獎勵計算函數
        done = self.is_done()  # 是否終止
        info = {}  # 額外的訊息

        # 更新當前狀態
        self.current_state = updated_state

        return updated_state, reward, done, info
    
    def update_state(self, current_state, action):
        # 根據動作更新狀態，例如更新疊加位置等
        updated_state = current_state.copy()

        # 假設這裡有一個函數update_position，根據action更新疊加位置
        updated_state["current_position"] = self.update_position(current_state["current_position"], action)

        # 假設這裡有一個函數update_forces，根據action更新向下力和向上力
        updated_state["downward_force"], updated_state["upward_force"] = self.update_forces(current_state, action)

        return updated_state

    def update_position(self, current_position, action):
        # 根據action更新疊加位置
        # 假設 action 是一個 (dx, dy, dz) 的位移向量
        return current_position + np.array(action)

    def update_forces(self, current_state, action):
        # 根據action更新向下力和向上力
        # 假設 action 影響了向下力和向上力
        downward_force = current_state["downward_force"] + action[0]  # 假設 action 的第一個元素影響向下力
        upward_force = current_state["upward_force"] - action[0]  # 假設 action 的第一個元素影響向上力
        return downward_force, upward_force
    
    def calculate_reward(self, updated_state, action):
        # 自定義獎勵計算函數
        # 根據您的環境需求計算並返回獎勵

        # 假設以下是示例代碼，請根據您的環境需求自定義
        # 假設磚體目標位置為一個字典，其中鍵是磚體的ID，值是目標位置的坐標
        brick_id = self.current_state["brick_id"]  # 當前疊加的磚體ID
        current_position = self.current_state["current_position"]  # 當前疊加的磚體的位置
        goal_position = self.goal_positions[brick_id]  # 當前疊加的磚體的目標位置
        reward = 0

        # 2.4.1 以符合力學模型為首要目標
        # 假設向下力和向上力已知，可以進行簡單的獎勵計算
        downward_force = self.current_state["downward_force"]
        upward_force = self.current_state["upward_force"]

        if downward_force > upward_force:
            reward += downward_force - upward_force

        # 2.4.2 以疊加完全部磚體成一個磚牆為第二重要目標
        # 如果疊加成功，給予指數級倍增的獎勵
        if self.is_wall_complete():
            reward *= 2**len(goal_position)

        # 2.4.3 第三重要目標是以較短時間完成所有磚體的疊加
        if self.current_state["brick_id"] > 0:
            time = time.time() - self.start_time
            # 如果疊加成功，給予指數級倍增的獎勵
            reward *= 2**self.max_brick_num / time
        elif self.is_done():
            reward *= 2**self.max_brick_num # 如果疊加完成，給予指數級倍增的獎勵
        else:
            reward = 0

        return reward
    
    def reset(self):
        # 重置環境狀態
        # 在這裡實現將環境重置為初始狀態的邏輯
        self.current_state = self.initial_state()  # 假設有一個initial_state方法返回初始狀態
        return self.current_state

    def render(self, mode='human'):
        # 可視化環境（可選）
        if mode == 'human':
            # 在這裡實現環境的可視化，以便您可以觀察環境的運行
            #node_num=the node numbers in the graph in json_data
            if self.json_data["assembly"]["edge"]["i"] == None:
                node_num = 0
            else:
                node_num = len(self.json_data["assembly"]["edge"]["i"])

            # 創建一個NetworkX圖
            G = nx.Graph()

            # 將位置數據添加為點
            for i, location in enumerate(self.locations):
                G.add_node(i, pos=location)

            # 可視化環境
            pos = nx.get_node_attributes(G, 'pos')
            nx.draw(G, pos, with_labels=True, node_size = 10, node_color='black')

            viewer = App()
            # 環境具有位置數據，存儲在self.locations中
            # self.locations是一個包含位置坐標的列表，每個位置是一個(x, y)元組
            for location in self.locations:
                x, y, z = location
                box = cg.Box(cg.Point(x, y, z), width=23.0, depth=11.0, height=6.0)  # 創建一個Box
                viewer.add(box)
            viewer.run()

        elif mode == 'robot_mode':
            # local mode for robot
            pass
        else:
            super(CustomEnvironment, self).render(mode=mode)

# 創建PPO模型
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        # self.linear3 = nn.Linear(hidden_size, hidden_size)
        # self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.output = nn.Linear(hidden_size, num_actions)

        self.num_actions = num_actions
        
    def forward(self, state, softmax_dim=-1):
        x = F.tanh(self.linear1(state))
        x = F.tanh(self.linear2(x))
        # x = F.tanh(self.linear3(x))
        # x = F.tanh(self.linear4(x))

        probs = F.softmax(self.output(x), dim=softmax_dim)
        
        return probs
    
    def evaluate(self, state, epsilon=1e-8):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        probs = self.forward(state, softmax_dim=-1)
        log_probs = torch.log(probs)

        # Avoid numerical instability. Ref: https://github.com/ku2482/sac-discrete.pytorch/blob/40c9d246621e658750e0a03001325006da57f2d4/sacd/model.py#L98
        z = (probs == 0.0).float() * epsilon
        log_probs = torch.log(probs + z)

        return log_probs
        
    def get_action(self, state, deterministic):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        probs = self.forward(state)
        dist = Categorical(probs)

        if deterministic:
            action = np.argmax(probs.detach().cpu().numpy())
        else:
            action = dist.sample().squeeze().detach().cpu().numpy()
        return action

# 保存损失历史图表
def save_loss_history_plot(loss_history, epoch):
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss History (Epoch {epoch})")
    plt.savefig(f"loss_history_epoch_{epoch}.png")
    plt.close()
    
# 定義訓練過程
def train_ppo(env, epochs, input_size, output_size, learning_rate, save_interval):
    policy_net = PolicyNetwork(input_size, output_size)
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    ppo_agent = PPO("MlpPolicy", env, verbose=1, policy_kwargs={"net_arch": [64, 64]})  # 根据策略网络结构调整net_arch

    writer = SummaryWriter()  # 创建Tensorboard的SummaryWriter

    loss_history = []  # 用于保存损失历史

    for epoch in range(epochs):
        obs = env.reset()
        epoch_rewards = 0

        while True:
            # 收集数据
            with torch.no_grad():
                action = ppo_agent.predict(obs)[0]

            # 执行动作并观察环境
            new_obs, reward, done, _ = env.step(action)

            # 计算梯度并更新模型权重
            loss = -ppo_agent.policy.log_prob(action, obs).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            obs = new_obs
            epoch_rewards += reward

            if done:
                break

        loss_history.append(loss.item())  # 将损失添加到历史列表中

        print(f"Episode {epoch + 1}/{epochs}, Total Reward: {epoch_rewards}")

        # 每save_interval个周期保存损失历史图表
        if (epoch + 1) % save_interval == 0:
            save_loss_history_plot(loss_history, epoch + 1)

    ppo_agent.save("ppo_model")  # 保存训练好的模型

    writer.close()  # 关闭Tensorboard的SummaryWriter


# 定义验证过程和测试过程，类似于训练过程，但不进行梯度更新
def validate_ppo(env, epochs, input_size, output_size):
    num_val = 10  # 设置验证周期的数量
    policy_net = PolicyNetwork(input_size, output_size)
    epoch_rewards = []

    for epoch in range(epochs):
        obs = env.reset()
        total_reward = 0

        while True:
            with torch.no_grad():
                action = policy_net(obs)  # 使用策略网络进行预测，不进行梯度更新

            new_obs, reward, done, _ = env.step(action)

            obs = new_obs
            total_reward += reward

            if done:
                break

        epoch_rewards.append(total_reward)

    average_reward = np.mean(epoch_rewards)
    print(f"Validation Average Reward: {average_reward}")

# 定义测试过程
def test_ppo(env, epochs, input_size, output_size):
    num_test = 10  # 设置测试周期的数量
    policy_net = PolicyNetwork(input_size, output_size)
    epoch_rewards = []

    for epoch in range(epochs):
        obs = env.reset()
        total_reward = 0

        while True:
            with torch.no_grad():
                action = policy_net(obs)  # 使用策略网络进行预测，不进行梯度更新

            new_obs, reward, done, _ = env.step(action)

            obs = new_obs
            total_reward += reward

            if done:
                break

        epoch_rewards.append(total_reward)

    average_reward = np.mean(epoch_rewards)
    print(f"Test Average Reward: {average_reward}")




# 主程序入口
if __name__ == "__main__":

    self = argparse.ArgumentParser()
    args = self.parse_args()
    root_dir = os.path.dirname(os.path.abspath(__file__))
    filename = self.add_argument('--filename', type=str, default='equilibrium_arch_20230608-203748.json')
    epoch = self.add_argument('--epoch', type=int, default=300)
    input_size = self.add_argument('--input_size', type=int, default=3)
    output_size = self.add_argument('--output_size', type=int, default=6)
    learning_rate = self.add_argument('--learning_rate', type=float, default=0.001)
    save_interval = self.add_argument('--save_interval', type=int, default=10)
    json_data = json.load(open('{}/{}'.format(root_dir, filename)))
    env = CustomEnvironment(json_data)
    # 訓練PPO模型
    train_ppo(env, epoch, input_size, output_size, learning_rate, save_interval)
    # 调用验证过程
    validate_ppo(env, epoch, input_size, output_size)

    # 调用测试过程
    test_ppo(env, epoch, input_size, output_size)

