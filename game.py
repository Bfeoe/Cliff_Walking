import pygame
from maze import Maze_config


# 颜色字典
color_dir = {
    0: (255, 255, 255),     # white
    1: (0, 0, 0),           # black
    2: (255, 215, 0),       # golden
    3: (0,0,255),           # blue
    4: (192,192,192),       # grey
    5: (255, 0, 0),         # red
    6: (0,0,255),           # blue
}


# 使用 pygame 实现可视化
class Game_Visual(object):
    def __init__(self, config: Maze_config) -> None:
        # 设置可视化视窗大小
        SCREEN_SIZE = [800, 800]
        self.BLOCK_SIZE = SCREEN_SIZE[0] // config.NUM_COLS

        # 初始化 Pygame
        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        self.font = pygame.font.SysFont(None, 30)    # 字体


    # 填某一块的颜色
    def draw_block(self, color: tuple, position: tuple) -> None:
        block = pygame.Surface((self.BLOCK_SIZE, self.BLOCK_SIZE))
        block.fill(color)
        rect = block.get_rect()
        rect.topleft = (position[1] * self.BLOCK_SIZE, position[0] * self.BLOCK_SIZE)

        self.screen.blit(block, rect)


    # 区分颜色然后画图
    def draw_maze(self, config: Maze_config) -> None:
        for i in range(len(config.maze)):
            for j in range(len(config.maze[i])):
                color = color_dir[config.maze[i][j]]
                self.draw_block(color, (i, j))


    # 更新界面
    def update_screen(self, config: Maze_config, steps: int, current_epochs: int) -> None:
        self.screen.fill((255, 255, 255))

        # 画迷宫图像
        self.draw_maze(config)

        # 当前坐标
        current_position = config.turn_to_position(config.current_state)
        self.draw_block((0, 255, 0), current_position)    # green

        # 显示步数
        text_steps = self.font.render(f"Step: {steps}", True, (0, 0, 0))
        self.screen.blit(text_steps, (10, 0))

        # 显示轮次
        text_epochs = self.font.render(f"Epoch: {current_epochs}", True, (0, 0, 0))
        self.screen.blit(text_epochs, (210, 0))

        # 显示 mode
        text_mode = self.font.render(f"Mode: {config.model}", True, (0, 0, 0))
        self.screen.blit(text_mode, (410, 0))

        pygame.display.update()