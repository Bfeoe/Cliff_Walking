import numpy as np
import pandas as pd
from maze import Maze_config
import os


# Sarsa(λ) 算法
class SarsaLambda(object):
    def __init__(self, config: Maze_config, epsilon: float = 1.0) -> None:
        # 定义超参数
        self.alpha = 0.05               # 学习率
        self.gamma = 0.9                # 折扣因子
        self.epsilon = epsilon // 10    # 探索率
        self.lambda_ = 0.9          # 资格迹折扣因子

        self.model_path = config.save_dir + "sarsa_lambda_table.csv"

        # 如果没有初始化的 Q 表则生成个空表
        if not os.path.exists(self.model_path):
            self.Q = np.zeros([config.NUM_STATES, config.NUM_ACTIONS])
        else:
            df = pd.read_csv(self.model_path)
            self.Q = df.values
            print(f"加载了训练好的模型")

        # 初始化资格迹表
        self.E = np.zeros_like(self.Q)


    # 选择下一个行动
    def choose_action(self, config: Maze_config, iteration: int) -> int:
        exploration_rate = self.epsilon / (iteration + 1)
        if np.random.rand() < exploration_rate:
            action = np.random.randint(config.NUM_ACTIONS)
        else:
            action = np.argmax(self.Q[config.current_state, :] + np.random.randn(1, config.NUM_ACTIONS) * (1. / (iteration + 1)))
        return action


    # 更新 Q 表和资格迹表
    def update(self, config: Maze_config, action: int, reward: float, next_action: int) -> None:
        next_state = config.next_state
        current_state = config.current_state

        # 计算TD误差
        td_error = reward + self.gamma * self.Q[next_state, next_action] - self.Q[current_state, action]

        # 更新资格迹表
        self.E[current_state, action] += 1

        # 更新 Q 表和资格迹表
        self.Q += self.alpha * td_error * self.E
        self.E *= self.gamma * self.lambda_


    # 训练模型
    def train_model(self, config: Maze_config, iteration: int) -> float and bool:
        action = self.choose_action(config, iteration)
        config.get_next_state(action)
        reward = config.get_reward()
        next_action = self.choose_action(config, iteration + 1)

        self.update(config, action, reward, next_action)

        judge_position = config.get_judgement()

        return reward, judge_position


    # 保存 Q 表
    def save_model(self) -> None:
        df = pd.DataFrame(self.Q)
        df.to_csv(self.model_path, index=False)


    # 重置资格迹表
    def reset_eligibility(self) -> None:
        self.E = np.zeros_like(self.Q)
