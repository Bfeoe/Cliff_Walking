import matplotlib.pyplot as plt
from maze import Maze_config


# 画图
def draw_reward(config: Maze_config, rewards: list) -> None:
    episode_numbers = list(range(1, len(rewards) + 1))
    avg_reward = sum(rewards) / len(rewards)

    plt.figure(figsize=(10, 5))
    plt.plot(episode_numbers, rewards, marker='o')  # 绘制折线图，使用圆圈标记每个点
    plt.axvline(x = config.converge_epoch, color='red')  # 红色的收敛线
    plt.axhline(y = avg_reward, color='green')  # 绿色的平均 reward 线

    plt.title(config.mode)
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.show()


# 为多个模型定义一个共同的接口
class Model_Interface(object):
    def __init__(self, visual_mode: bool, epochs: int) -> None:
        self.visual_mode = visual_mode
        self.epochs = epochs


    # 训练函数
    def train(self, config: Maze_config, agent, game_config = None) -> None:
        total_rewards = []

        # 训练的轮数
        for epoch in range(self.epochs):
            print(f'第 {epoch+1} 轮')
            i = 0
            rewards = 0
            delta_q_values = []

            # 每次最大步数
            for i in range(500):

                # 训练模型
                reward, judgement = agent.train_model(config, epoch, delta_q_values)
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
            if config.mode == "MC":
                agent.update()

            # 置为原位
            config.current_state = 0
            config.maze[config.desire_point[0]][config.desire_point[1]] = 6
            config.visited_positions = set()

            print(f'本轮总步数为: {i+1}\t本轮总奖励为: {rewards}\n')
            total_rewards.append(rewards)

            # 多轮的平均损失值
            if config.mode == "DQN" or config.mode == "MC":
                agent.converge(total_rewards)
                config.converge_epoch = epoch + 1

        if config.converge_epoch == 0:
            print("并没有完全收敛或运行轮次过低")
        else:
            print(f'收敛于epoch数为: {config.converge_epoch} 处')

        # 存模型画图
        agent.save_model()
        draw_reward(config, total_rewards)