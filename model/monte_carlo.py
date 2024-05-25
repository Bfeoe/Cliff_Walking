import numpy as np
import pandas as pd
from maze import Maze_config


class MonteCarlo:
    def __init__(self, config: Maze_config) -> None:
        self.Q = np.zeros([config.NUM_STATES, config.NUM_ACTIONS])  # 初始化状态动作值函数
        self.returns = {}  # 用于存储每个状态动作对的回报和计数
        self.policy = {}  # 用于存储策略，即每个状态下选择的动作

    def choose_action(self, config: Maze_config) -> int:
        state = config.current_state
        if state not in self.policy:
            self.policy[state] = np.random.randint(config.NUM_ACTIONS)
        return self.policy[state]

    def update(self, episode) -> None:
        G = 0  # 初始化回报
        for step in reversed(range(len(episode))):
            state, action, reward = episode[step]
            G = reward + config.GAMMA * G  # 计算回报
            sa_pair = (state, action)
            if sa_pair not in [x[0] for x in episode[0:step]]:  # 确保不重复计算同一状态动作对的回报
                if sa_pair in self.returns:
                    self.returns[sa_pair].append(G)
                else:
                    self.returns[sa_pair] = [G]
                self.Q[state, action] = np.mean(self.returns[sa_pair])  # 更新状态动作值函数
