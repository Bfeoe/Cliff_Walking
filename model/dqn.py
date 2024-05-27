import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from maze import Maze_config
import os
from collections import deque


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

        # 初始化DQN模型
        self.model_path = config.save_dir + "dqn_model.pth"
        self.model = DQN_Model(self.state_size, self.action_size, hidden_size).to(device)
        # 加载模型
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            print(f"加载了训练好的模型")

        # 初始化 Adam 优化器和均方误差损失函数
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # 设置目标网络
        self.target_model = DQN_Model(self.state_size, self.action_size, hidden_size).to(device)
        self.update_target_model()

        # 初始化经验回放池
        self.memory = deque(maxlen=2000)
        self.batch_size = 64


    # 更新目标网络
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


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


    # 记忆
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    # 经验回放
    def replay(self) -> None:
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        state_batch = torch.cat([self.turn_to_tensor(state) for state, _, _, _, _ in minibatch])
        next_state_batch = torch.cat([self.turn_to_tensor(next_state) for _, _, _, next_state, _ in minibatch])

        q_values = self.model(state_batch)
        next_q_values = self.target_model(next_state_batch)

        target = q_values.clone()
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            # G 了不考虑未来回报
            if done:
                target[i][action] = reward
            # Q 是折扣后的未来奖励
            else:
                target[i][action] = reward + self.gamma * torch.max(next_q_values[i]).item()

            # 计算损失和优化
            loss = self.criterion(q_values, target.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 更新epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    # 训练函数
    def train_model(self, config: Maze_config, iteration: int) -> tuple[float, bool]:
        state = config.current_state
        action = self.choose_action(state)

        # 计算出 next_state
        config.get_next_state(action)
        next_state = config.next_state

        # 计算 reward
        reward = config.get_reward()

        # 判断位置
        done = config.get_judgement()

        # 经验回放
        self.remember(state, action, reward, next_state, done)
        self.replay()

        return reward, done


    # 保存模型
    def save_model(self) -> None:
        if not os.path.exists(os.path.dirname(self.model_path)):
            os.makedirs(os.path.dirname(self.model_path))
        torch.save(self.model.state_dict(), self.model_path)
