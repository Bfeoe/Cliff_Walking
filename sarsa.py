import numpy as np
import pandas as pd
from maze import Maze_config


# Sarsa算法
class Sarsa(object):
    def __init__(self, config: Maze_config, q_table: str = None):
        # 定义超参数
        self.alpha = 0.1    # 学习率
        self.gamma = 0.9    # 折扣因子
        self.epsilon = 0.1  # 探索率

        # 如果没有初始化的Q表则生成个空表
        if q_table is None:
            self.Q = np.zeros([config.NUM_STATES, config.NUM_ACTIONS])
        # 将DataFrame转换为NumPy数组，并确保其尺寸匹配
        else:
            df = pd.read_csv(q_table)
            self.Q = df.values


    # 选择下一个行动
    def choose_action(self, config: Maze_config, iteration: int) -> int:
        exploration_rate = self.epsilon / (iteration + 1)
        if np.random.rand() < exploration_rate:
            action = np.random.randint(config.NUM_ACTIONS)
        else:
            action = np.argmax(self.Q[config.current_state, :] + np.random.randn(1, config.NUM_ACTIONS) * (1. / (iteration + 1)))
        return action


    # 更新Q表
    def update(self, config: Maze_config, action: int, reward: int) -> None:
        next_state = config.next_state
        current_state = config.current_state
        self.Q[current_state, action] += self.alpha * (reward + self.gamma * self.Q[next_state, next_action] - self.Q[current_state, action])


    # 保存Q表
    def save_model(self) -> None:
        df = pd.DataFrame(self.Q)
        df.to_csv('sarsa_table.csv', index=False)
