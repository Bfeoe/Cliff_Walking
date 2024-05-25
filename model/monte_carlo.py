import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from maze import Maze_config
import os
from collections import defaultdict
from typing import Tuple


# 让他在GPU上运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 定义模型
class MonteCarloModel(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 64) -> None:
        super(MonteCarloModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)


    # 前向传播
    def forward(self, x) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 定义Agent
class MonteCarloAgent(object):
    def __init__(self, config: Maze_config, epsilon: float = 1.0, hidden_size: int = 64) -> None:
        # 初始化基本参数
        self.state_size = config.NUM_STATES
        self.action_size = config.NUM_ACTIONS

        # 定义超参数
        learning_rate = 0.001       # 学习率
        self.epsilon = epsilon      # 探索率
        self.epsilon_min = 0.01     # 探索率下限
        self.epsilon_decay = 0.995  # 探索率的衰减因子
        self.gamma = 0.99           # 折扣因子决定了智能体对未来奖励的重视程度

        self.model_path = config.save_dir + "monte_carlo_model.pth"

        # 初始化蒙特卡洛模型
        self.model = MonteCarloModel(self.state_size, self.action_size, hidden_size).to(device)
        # 加载模型
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            print(f"加载了训练好的模型")

        # 初始化Adam优化器和均方误差损失函数
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # 初始化访问次数和回报
        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(float)
        self.q_values = defaultdict(lambda: np.zeros(self.action_size))

        self.episode = []


    # 将int转为one-hot向量
    def turn_to_tensor(self, state: int) -> torch.Tensor:
        state_one_hot = np.zeros(self.state_size)
        state_one_hot[state] = 1
        state_tensor = torch.tensor(state_one_hot, dtype=torch.float32).unsqueeze(0).to(device)
        return state_tensor


    # 根据当前策略选择动作
    def choose_action(self, state: int) -> int:
        # 以epsilon的概率随机选择动作
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # 选择Q值最大的动作
        state = self.turn_to_tensor(state)
        q_values = self.model(state)  # 前向传播，通过神经网络计算 Q 值
        return torch.argmax(q_values).item()



    # 更新模型
    def update(self):
        # 计算每个状态-动作对的回报
        G = 0
        for t in reversed(range(len(self.episode))):
            state, action, reward = self.episode[t]
            G = self.gamma * G + reward
            sa_pair = (state, action)

            if sa_pair not in [(x[0], x[1]) for x in self.episode[:t]]:
                self.returns_sum[sa_pair] += G
                self.returns_count[sa_pair] += 1
                self.q_values[sa_pair][action] = self.returns_sum[sa_pair] / self.returns_count[sa_pair]

                # 更新模型
                state_tensor = self.turn_to_tensor(state)
                q_values = self.model(state_tensor).detach().cpu().numpy().flatten()
                q_values[action] = self.q_values[sa_pair][action]
                q_values_tensor = torch.tensor(q_values, dtype=torch.float32).unsqueeze(0).to(device)

                # 计算损失
                predicted_q_values = self.model(state_tensor)
                loss = self.criterion(predicted_q_values, q_values_tensor)

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.episode = []

        # 更新探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    # 训练函数
    def train_model(self, config: Maze_config, iteration: int = None) -> Tuple[int, bool]:
        state = config.current_state
        action = self.choose_action(state)
        config.get_next_state(action)
        reward = config.get_reward()

        self.episode.append((state, action, reward))
        judge_position = config.get_judgement()

        return reward, judge_position


    def save_model(self) -> None:
        if not os.path.exists(os.path.dirname(self.model_path)):
            os.makedirs(os.path.dirname(self.model_path))
        torch.save(self.model.state_dict(), self.model_path)
