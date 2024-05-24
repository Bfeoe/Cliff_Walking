import numpy as np
import pandas as pd
from maze import Maze_config


# Q_Learning算法
class Q_Learning(object):
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
        # 使用epsilon贪心策略选择动作，其中探索率随迭代次数递减
        exploration_rate = self.epsilon / (iteration + 1)
        if np.random.rand() < exploration_rate:
            action = np.random.randint(config.NUM_ACTIONS)
        else:
            action = np.argmax(self.Q[config.current_state, :] + np.random.randn(1, config.NUM_ACTIONS) * (1. / (iteration + 1)))
        return action


    # 更新Q表
    def update(self, config: Maze_config, action: int, reward: int) -> None:
        self.Q[config.current_state, action] += (self.alpha * (reward + self.gamma * np.max(self.Q[config.next_state, :]) - self.Q[config.current_state, action]))


    # 保存Q表
    def save_model(self) -> None:
        df = pd.DataFrame(self.Q)
        df.to_csv('Q_learning_table.csv', index=False)
