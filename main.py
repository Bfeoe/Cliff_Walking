import argparse
import pygame
# 其他组件
from game import Game_Visual
from maze import Maze_config
from train import Model_Interface
# 我的模型
from model.q_learning import Q_Learning
from model.dqn import DQN
from model.sarsa import Sarsa
from model.monte_carlo import Monte_Carlo
from model.sarsa_lambda import SarsaLambda


# 主函数呦
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", choices=["Q-Learning", "Sarsa", "DQN", "MC", "L"], default='Q-Learning', nargs='?',
                        required=False, help="模型的选择")
    parser.add_argument("--epoch", type=int, default=1000, nargs='?', required=False, help="训练的轮次")

    parser.add_argument("--maze", type=str, default="maze_map.csv", nargs='?', required=False, help="导入地图")
    parser.add_argument("--model_dir", type=str, default="result\\", nargs='?', required=False, help="模型保存的根目录")

    parser.add_argument("--visual", type=bool, default=False, nargs='?', required=False, help="是否需要可视化")
    parser.add_argument("--explore", type=bool, default=True, nargs='?', required=False, help="是否启用探索")
    parser.add_argument("--epsilon", type=float, default=1.0, nargs='?', required=False, help="探索率的设置")

    parser.add_argument("--tactic", type=bool, default=True, nargs='?', required=False, help="一些reward的策略")

    args = parser.parse_args()

    # 初始化迷宫和目标点
    config = Maze_config(args.model, args.maze, args.model_dir)
    config.update_goal()

    # 处理危险点的策略
    if args.tactic:
        config.update_maze()

    # 可视化情况之下默认禁用探索以保证准确预测
    if args.visual:
        args.explore = False

    epsilon = args.epsilon   # Q-Learning 和 Sarsa 算法整除了10
    # 探索率置为0
    if not args.explore:
        epsilon = 0
    print("epsilon:", epsilon, "\n")

    # 选用的算法
    # Q-Learning
    if args.model == "Q-Learning":
        agent = Q_Learning(config, epsilon)
    # Sarsa
    elif args.model == "Sarsa":
        agent = Sarsa(config, epsilon)
    # DQN
    elif args.model == "DQN":
        agent = DQN(config, epsilon)
    # Monte-Carlo
    elif args.model == "MC":
        agent = Monte_Carlo(config, epsilon)
    # Sarsa_lambda
    elif args.model == "L":
        agent = SarsaLambda(config, epsilon)
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
