import matplotlib.pyplot as plt
from maze import Maze_config
from game import Game_Visual


# 画图
def draw_reward(config: Maze_config, rewards: list) -> None:
    episode_numbers = list(range(1, len(rewards) + 1))
    avg_reward = sum(rewards) / len(rewards)

    plt.figure(figsize=(10, 5))
    plt.plot(episode_numbers, rewards, marker='o')  # 绘制折线图，使用圆圈标记每个点
    plt.axhline(y=avg_reward, color='green')  # 绿色的平均 reward 线

    plt.title(config.model)
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    # 保存图像为PNG文件
    plt.savefig(f'result\\{config.model}_reward_plot.png')


# 为多个模型定义一个共同的接口
class Model_Interface(object):
    def __init__(self, visual_mode: bool, epochs: int) -> None:
        self.visual_mode = visual_mode
        self.epochs = epochs


    # 训练函数
    def train(self, config: Maze_config, agent, game_config: Game_Visual = None) -> None:
        total_rewards = []

        # 训练的轮数
        for epoch in range(self.epochs):
            print(f'第 {epoch+1} 轮')

            i = 0
            rewards = 0

            # 每次最大步数
            for i in range(500):

                # 训练模型
                reward, judgement = agent.train_model(config, epoch)
                rewards += reward

                # 用于判断当前位置是否符合规则
                if not judgement:
                    break

                # 可以可视化时随时更新模型
                if self.visual_mode:
                    game_config.update_screen(config, i, epoch)

                # 每移动 20 步则更改一次目标点位置
                if i % 20 == 0:
                    config.update_goal()

            # Monte_Carlo 方法是训练完一条完整路径后一并更新
            if config.model == "MC":
                agent.update()
            # Sarsa_Lambda 更新资格迹表
            elif config.model == "L":
                agent.reset_eligibility()

            print(f'本轮总步数为: {i+1}\t本轮总奖励为: {rewards}\n')
            total_rewards.append(rewards)

            # 每十轮更新一次目标网络
            if config.model == "DQN":
                if epoch % 10 == 0:
                    agent.update_target_model()

            # 置为原位
            config.current_state = 0
            config.maze[config.desire_point[0]][config.desire_point[1]] = 6
            config.visited_positions = set()

        # 存模型画图
        agent.save_model()
        draw_reward(config, total_rewards)
