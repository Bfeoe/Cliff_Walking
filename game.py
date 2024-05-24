import pygame
from maze import Maze_config


# 颜色字典
color_dir = {
    0: (255, 255, 255),   # white
    1: (0, 0, 0),         # black
    2: (255, 215, 0),     # golden
    3: (72, 118, 255),    # blue
    5: (255, 0, 0),       # red
}


# 使用pygame实现可视化
class Game_Visual(object):
    def __init__(self, config: Maze_config, mode: str) -> None:
        self.mode = mode

        # 设置可视化视窗大小
        SCREEN_SIZE = [800, 800]
        self.BLOCK_SIZE = SCREEN_SIZE[0] // config.NUM_COLS

        # 初始化Pygame
        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        self.font = pygame.font.SysFont(None, 30)    # 字体


    # 填某一块的颜色
    def draw_block(self, color: tuple, current_position: tuple) -> None:
        block = pygame.Surface((self.BLOCK_SIZE, self.BLOCK_SIZE))
        block.fill(color)
        rect = block.get_rect()
        rect.topleft = (current_position[1] * self.BLOCK_SIZE, current_position[0] * self.BLOCK_SIZE)
        self.screen.blit(block, rect)


    # 区分颜色然后画图
    def draw_maze(self, config: Maze_config, current_position: tuple, goal_position: tuple) -> None:
        for i in range(len(config.maze)):
            for j in range(len(config.maze[i])):
                color = color_dir[config.maze[i][j]]
                self.draw_block(color, (i, j))

        self.draw_block((0, 255, 0), current_position)
        self.draw_block((255, 215, 0), goal_position)


    # 更新界面
    def update_screen(self, config: Maze_config, steps: int, current_epochs: int) -> None:
        self.screen.fill((255, 255, 255))

        # 当前坐标与此时目标的坐标
        current_position = config.turn_to_position(config.current_state)
        goal_position = config.turn_to_position(config.goal_state)

        # 画迷宫图像
        self.draw_maze(config, current_position, goal_position)

        # 显示步数
        text = self.font.render("Step: {}".format(steps), True, (0, 0, 0))
        self.screen.blit(text, (10, 0))

        # 显示轮次
        text = self.font.render("Epoch: {}".format(current_epochs), True, (0, 0, 0))
        self.screen.blit(text, (210, 0))

        # 显示mode
        text = self.font.render(f"Mode: {self.mode}", True, (0, 0, 0))
        self.screen.blit(text, (410, 0))

        pygame.display.update()