import random
import pandas as pd
import numpy as np


# 鼓励在终点的区域之内探索
random_reward = np.random.uniform(0, 4)

# reward 索引表
reward_dir = {
    0: -1,              # 空地
    1: -1000,            # 墙壁时
    2: 15,              # 终点位置
    3: random_reward,   # 终点可能存在的区域
    4: -4,              # 危险区域(贴近墙壁或悬崖的位置)
    5: -1000,           # 悬崖
    6: 4                # 手动引导点
}


# 计算曼哈顿距离
def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)


# 定义迷宫类
class Maze_config(object):
    def __init__(self, model: str, maze_map: str, save_dir: str) -> None:
        self.model = model.upper()
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

        # 记录 Agent 走过的路径
        self.visited_positions = set()
        self.distances = 78

        # 初始化起始点与目标点(最右下角)
        self.current_state = 0
        self.next_state = None
        self.goal_state = self.NUM_STATES - 1

        # 迷宫终点可能的区域
        self.positions_of_goal = [(i, j) for i in range(30, 40) for j in range(30, 40)]

        # 设置引导点(终点区域的最左上角)
        self.desire_point = (30, 30)


    # 将状态转为坐标位置
    def turn_to_position(self, state: int) -> tuple:
        x, y = state // self.NUM_ROWS, state % self.NUM_COLS
        return x, y


    # 将坐标位置转为状态
    def turn_to_state(self, x: int , y: int) -> int:
        state = x * self.NUM_COLS + y
        return state


    # 处理危险点的策略
    def update_maze(self) -> None:
        # 设置期望到达的点
        print("启用适当的引导策略")
        self.maze[self.desire_point[0]][self.desire_point[1]] = 6

        # 获取值为 1 或 5 的危险点
        dangerous_point = np.argwhere((self.maze == 1) | (self.maze == 5))

        # 遍历这些坐标点及其周围的点
        for (x, y) in dangerous_point:
            # 上下左右四个点的坐标
            neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
            for (nx, ny) in neighbors:
                # 检查是否在数组范围内且值为 0
                if 0 <= nx < 40 and 0 <= ny < 40:
                    if self.maze[nx][ny] == 0:
                        self.maze[nx][ny] = 4


    # 定义 reward 函数
    def get_reward(self) -> float:
        # 基准奖励
        x, y = self.turn_to_position(self.next_state)
        base_reward = reward_dir.get(self.maze[x][y])

        # 接近奖励
        goal_x, goal_y = self.turn_to_position(self.goal_state)
        distance = manhattan_distance(x, y, goal_x, goal_y)
        # 距离越近则奖励越高
        if distance <= self.distances:
            proximity_reward = 10 / (distance + 1)
        # 否则予与负的奖励
        else:
            proximity_reward = - (5 / (distance + 1))
        self.distances = distance

        # proximity_reward = 10 / (distance + 1)  # 距离越近，奖励越高

        # 重复路径惩罚
        if (x, y) in self.visited_positions:
            repeat_penalty = -5
        else:
            repeat_penalty = 0.1
            self.visited_positions.add((x, y))

        total_reward = base_reward + proximity_reward + repeat_penalty
        return total_reward


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
    def update_goal(self) -> None:
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
        # 引导点只允许到一次
        elif self.maze[x][y] == 6:
            self.current_state = self.next_state
            self.maze[x][y] = 3
            return True
        # 其余位置更新当前位置状态后继续
        else:
            self.current_state = self.next_state
            return True
