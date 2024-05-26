import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from maze import Maze_config
import os


# 让它在 GPU 上运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 定义模型
class DQN_Model(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 64) -> None:
        super(DQN_Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)


    # 前向传播
    def forward(self, x) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# 定义 Agent
class DQN(object):
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

        # 判敛
        self.threshold = 10
        self.window = 20

        self.model_path = config.save_dir + "dqn_model.pth"

        # 初始化DQN模型
        self.model = DQN_Model(self.state_size, self.action_size, hidden_size).to(device)
        # 加载模型
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            print(f"加载了训练好的模型")

        # 初始化 Adam 优化器和均方误差损失函数
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()


    # 将 int 转为 one-hot 向量
    def turn_to_tensor(self, state: int) -> torch.Tensor:
        state_one_hot = np.zeros(self.state_size)
        state_one_hot[state] = 1
        state_tensor = torch.tensor(state_one_hot, dtype=torch.float32).unsqueeze(0).to(device)
        return state_tensor


    # 根据当前策略选择动作
    def choose_action(self, state: int) -> int:
        # 以 epsilon 的概率随机选择动作
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # 选择 Q 值最大的动作
        state = self.turn_to_tensor(state)
        q_values = self.model(state)  # 通过神经网络计算 Q 值
        return torch.argmax(q_values).item()


    # 判敛
    def converge(self, rewards: list) -> bool:
        recent_rewards = rewards[-self.window:]
        return np.std(recent_rewards) < self.threshold


    # 训练函数
    def train_model(self, config: Maze_config, iteration: int, rewards: list) -> tuple[float, bool]:
        state = config.current_state
        action = self.choose_action(state)

        # 计算出 next_state
        config.get_next_state(action)
        next_state = config.next_state

        # 计算 reward
        reward = config.get_reward()

        judge_position = config.get_judgement()

        # 将两个状态转为向量形式
        state_tensor =  self.turn_to_tensor(state)
        next_state_tensor = self.turn_to_tensor(next_state)

        # 计算当前状态的 Q 值
        q_values = self.model(state_tensor)
        target = q_values.clone()

        # 目标 Q 值为即时奖励加上折扣后的未来奖励
        next_q_values = self.model(next_state_tensor)
        target[0][action] = reward + self.gamma * torch.max(next_q_values).item()

        # 计算损失
        loss = self.criterion(q_values, target.detach())

        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return reward, judge_position


    # 保存模型
    def save_model(self) -> None:
        if not os.path.exists(os.path.dirname(self.model_path)):
            os.makedirs(os.path.dirname(self.model_path))
        torch.save(self.model.state_dict(), self.model_path)
