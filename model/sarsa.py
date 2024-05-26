from typing import Any

import numpy as np
import pandas as pd
from maze import Maze_config
import os


# Sarsa 算法
class Sarsa(object):
    def __init__(self, config: Maze_config, epsilon: float = 1.0) -> None:
        # 定义超参数
        self.alpha = 0.1                # 学习率
        self.gamma = 0.9                # 折扣因子
        self.epsilon = epsilon // 10    # 探索率

        # 判敛
        self.threshold = 0.01
        self.window = 20

        self.model_path = config.save_dir + "sarsa_table.csv"

        # 如果没有初始化的 Q 表则生成个空表
        if not os.path.exists(self.model_path):
            self.Q = np.zeros([config.NUM_STATES, config.NUM_ACTIONS])
        # 将 DataFrame 转换为 NumPy 数组
        else:
            df = pd.read_csv(self.model_path)
            self.Q = df.values
            print(f"加载了训练好的模型")


    # 选择下一个行动
    def choose_action(self, config: Maze_config, iteration: int) -> int:
        exploration_rate = self.epsilon / (iteration + 1)
        if np.random.rand() < exploration_rate:
            action = np.random.randint(config.NUM_ACTIONS)
        else:
            action = np.argmax(self.Q[config.current_state, :] + np.random.randn(1, config.NUM_ACTIONS) * (1. / (iteration + 1)))
        return action


    # 判敛
    def converge(self, delta_q_values: list) -> bool:
        recent_changes = delta_q_values[-self.window:]
        return max(recent_changes) < self.threshold


    # 更新 Q 表
    def update(self, config: Maze_config, action: int, reward: float, next_action: int) -> float:
        next_state = config.next_state
        current_state = config.current_state
        # 更新 Q 表
        delta_q_value = self.alpha * (reward + self.gamma * self.Q[next_state, next_action] - self.Q[current_state, action])
        self.Q[current_state, action] += delta_q_value

        return delta_q_value.item()


    # 训练模型
    def train_model(self, config: Maze_config, iteration: int, delta_q_values: list) -> float and bool:

        action = self.choose_action(config, iteration)
        config.get_next_state(action)
        reward = config.get_reward()
        next_action = self.choose_action(config, iteration + 1)

        delta_q_value = self.update(config, action, reward, next_action)
        delta_q_values.append(delta_q_value)
        if self.converge(delta_q_values):
            config.converge_epoch = iteration + 1

        judge_position = config.get_judgement()

        return reward, judge_position


    # 保存 Q 表
    def save_model(self) -> None:
        df = pd.DataFrame(self.Q)
        df.to_csv(self.model_path, index=False)