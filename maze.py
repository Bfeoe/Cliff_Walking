import random
import pandas as pd

# reward索引表
reward_dir = {
    0: -1,    # 走到空地
    1: -100,  # 走到墙壁时
    2: 100,   # 走到终点位置
    3: -1,    # 走到终点可能存在的区域
    5: -1000  # 走到悬崖
}

# 定义迷宫类
class Maze_config(object):
    def __init__(self, maze_map: str, mode: str, save_dir: str) -> None:
        self.mode = mode.upper()
        self.save_dir = save_dir

        # 定义地图大小
        self.NUM_ROWS = 40  # 行数
        self.NUM_COLS = 40  # 列数

        # 定义状态动作
        self.NUM_ACTIONS = 4
        self.NUM_STATES = self.NUM_ROWS * self.NUM_COLS

        # 初始化迷宫
        df = pd.read_csv(maze_map, header=None)
        self.maze = df.values

        # 初始化起始点与目标点(最右下角)
        self.current_state = 0
        self.next_state = None
        self.goal_state = self.NUM_STATES - 1

        # 迷宫终点可能的区域
        self.positions_of_goal = [(i, j) for i in range(29, 40) for j in range(29, 40)]


    # 将状态转为坐标位置
    def turn_to_position(self, state: int) -> tuple:
        x, y = state // self.NUM_ROWS, state % self.NUM_COLS
        return x, y


    # 将坐标位置转为状态
    def turn_to_state(self, x: int , y: int) -> int:
        state = x * self.NUM_COLS + y
        return state


    # 定义reward函数
    def get_reward(self) -> int:
        x, y = self.turn_to_position(self.next_state)
        reward = reward_dir[self.maze[x][y]]
        return reward


    # 定义下一个状态函数
    def get_next_state(self, action: int) -> None:
        x, y = self.turn_to_position(self.current_state)

        # 防止超脱边界
        if action == 0:
            x = max(x - 1, 0)
        elif action == 1:
            x = min(x + 1, self.NUM_COLS - 1)
        elif action == 2:
            y = max(y - 1, 0)
        elif action == 3:
            y = min(y + 1, self.NUM_ROWS - 1)

        self.next_state = self.turn_to_state(x, y)


    #  更新迷宫的终点
    def update_maze(self) -> None:
        goal_x, goal_y = self.turn_to_position(self.goal_state)
        self.maze[goal_x][goal_y] = 3

        next_goal_x, next_goal_y = random.choice(self.positions_of_goal)
        self.maze[next_goal_x][next_goal_y] = 2

        # 更新
        self.goal_state = self.turn_to_state(next_goal_x, next_goal_y)


    # 判断下一个位置是否合规
    def get_judgement(self) -> bool:
        x, y = self.turn_to_position(self.next_state)
        # 撞悬崖直接结束
        if self.maze[x][y] == 5:
            print("掉下悬崖")
            return False
        # 撞墙不更改当前坐标
        elif self.maze[x][y] == 1:
            return True
        # 到达终点也直接结束
        elif self.maze[x][y] == 2:
            print("成功到达")
            return False
        # 其余位置更新当前位置状态后继续
        else:
            self.current_state = self.next_state
            return True
