import numpy as np
import pandas as pd
from maze import Maze_config
import os


# Q_Learning 算法
class Q_Learning(object):
    def __init__(self, config: Maze_config, epsilon: float = 1.0) -> None:
        # 定义超参数
        self.alpha = 0.05               # 学习率
        self.gamma = 0.9                # 折扣因子
        self.epsilon = epsilon // 10    # 探索率

        self.model_path = config.save_dir + "q_learning_table.csv"

        # 如果没有初始化的 Q 表则生成个空表
        if not os.path.exists(self.model_path):
            self.Q = np.zeros([config.NUM_STATES, config.NUM_ACTIONS])
        # 将 DataFrame 转换为 NumPy 数组
        else:
            df = pd.read_csv(self.model_path, encoding="utf-8")
            self.Q = df.values
            print(f"加载了训练好的模型")


    # 选择下一个行动
    def choose_action(self, config: Maze_config, iteration: int) -> int:
        # 使用 epsilon 贪心策略选择动作，其中探索率随迭代次数递减
        exploration_rate = self.epsilon / (iteration + 1)
        if np.random.rand() < exploration_rate:
            action = np.random.randint(config.NUM_ACTIONS)
        else:
            action = np.argmax(self.Q[config.current_state, :] + np.random.randn(1, config.NUM_ACTIONS) * (1. / (iteration + 1)))
        return action


    # 更新 Q 表
    def update(self, config: Maze_config, action: int, reward: float) -> None:
        next_state = config.next_state
        current_state = config.current_state
        # 更新 Q 表
        self.Q[current_state, action] += (self.alpha * (reward + self.gamma * np.max(self.Q[next_state, :]) - self.Q[current_state, action]))


    # 训练
    def train_model(self, config: Maze_config, iteration: int) -> float and bool:
        # 选则下一步的行动,并更新模型
        action = self.choose_action(config, iteration)
        config.get_next_state(action)
        reward = config.get_reward()

        self.update(config, action, reward)

        judge_position = config.get_judgement()

        return reward, judge_position


    # 保存 Q 表
    def save_model(self) -> None:
        df = pd.DataFrame(self.Q)
        df.to_csv(self.model_path, index=False, encoding="utf-8")
