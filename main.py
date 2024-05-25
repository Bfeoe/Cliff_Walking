import argparse
import pygame
from model.dqn import DQNAgent
from game import Game_Visual
from maze import Maze_config
from model.q_learning import Q_Learning
from train import Model_Interface
from model.sarsa import Sarsa
from model.monte_carlo import MonteCarloAgent


# 主函数呦
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["Q-Learning", "Sarsa", "DQN", "MC"], default='Q-Learning', nargs='?',
                        required=False, help="模型的选择")
    parser.add_argument("--epoch", type=int, default=1000, nargs='?', required=False, help="训练的轮次")
    parser.add_argument("--maze", type=str, default="maze_map.csv", nargs='?', required=False, help="导入地图")
    parser.add_argument("--model_dir", type=str, default="result\\", nargs='?', required=False, help="模型保存的根目录")
    parser.add_argument("--visual", type=bool, default=False, nargs='?', required=False, help="是否需要可视化")

    args = parser.parse_args()

    # 初始化迷宫
    config = Maze_config(args.maze, args.mode, args.model_dir)
    config.update_maze()

    # 选用的算法
    # Q-Learning
    if args.mode == "Q-Learning":
        if args.visual:
            agent = Q_Learning(config)
        else:
            agent = Q_Learning(config, 0)
    # Sarsa
    elif args.mode == "Sarsa":
        if args.visual:
            agent = Sarsa(config)
        else:
            agent = Sarsa(config, 0)
    # DQN
    elif args.mode == "DQN":
        agent = DQNAgent(config, 0.01)
    # Monte-Carlo
    elif args.mode == "MC":
        agent = MonteCarloAgent(config, 0.01)
    else:
        agent = None


    # 是否启用可视化界面
    game_config = None
    if args.visual:
        pygame.init()
        game_config = Game_Visual(config)

    model = Model_Interface(args.visual, args.epoch)
    model.train(config, agent, game_config)


if __name__ == "__main__":
    main()