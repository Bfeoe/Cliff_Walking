import matplotlib.pyplot as plt
from maze import Maze_config


# 画图
def draw_reward(rewards: list, mode: str) -> None:
    episode_numbers = list(range(1, len(rewards) + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(episode_numbers, rewards, marker='o')  # 绘制折线图，使用圆圈标记每个点

    plt.title(mode)
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
        total_reward = []

        # 训练的轮数
        for epoch in range(self.epochs):
            print(f'第 {epoch+1} 轮')
            i = 0
            rewards = 0

            # 每次最大步数
            for i in range(500):
                # 每移动20步则更改一次目标点位置
                if i % 20 == 0:
                    config.update_maze()

                # 训练模型
                reward, judgement = agent.train_model(config, i)
                rewards += reward

                # 用于判断当前位置是否符合规则
                if not judgement:
                    break

                # 可以可视化时随时更新模型
                if self.visual_mode:
                    game_config.update_screen(config, i, epoch)

            # 置为原位
            config.current_state = 0

            print(f'本轮总步数为: {i+1}\t本轮总奖励为: {rewards}\n')
            total_reward.append(rewards)

        agent.save_model()
        draw_reward(total_reward, config.mode)