import argparse
import pygame
from game import Game_Visual
from maze import Maze_config
from q_learning import Q_Learning
from train import Train_Model
from sarsa import Sarsa


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["Q", "Sarsa", "DQN", "MC"], default='Q', nargs='?',
                        required=False, help="模型的选择")
    parser.add_argument("--epoch", type=int, default=1000, nargs='?', required=False, help="轮次")
    parser.add_argument("--maze_map", type=str, default="maze_map.csv", nargs='?', required=False, help="导入地图")
    parser.add_argument("--q_table", type=str, default=None, nargs='?', required=False, help="导入训练好的Q表")
    parser.add_argument("--visual", type=bool, default=False, nargs='?', required=False, help="是否需要可视化")

    args = parser.parse_args()

    # 初始化迷宫
    config = Maze_config(args.maze_map)
    config.update_maze()

    # 选用的算法
    if args.mode == "Q":
        agent = Q_Learning(config ,args.q_table)
    elif args.mode == "Sarsa":
        agent = Sarsa(config ,args.q_table)
    else:
        agent = None

    # 是否启用可视化界面
    game_config = None
    if args.visual:
        pygame.init()
        game_config = Game_Visual(config ,args.mode)

    model = Train_Model(args.visual, args.epoch)
    model.train(config, agent, game_config)


if __name__ == "__main__":
    main()