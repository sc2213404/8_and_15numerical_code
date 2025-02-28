import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageTk
import time
import heapq
import copy
from collections import deque
import os
import random

class Puzzle:
    def __init__(self, initial, goal):
        self.initial = initial
        self.goal = goal
        self.size = len(initial)

    def find_blank(self, state):
        for i in range(self.size):
            for j in range(self.size):
                if state[i][j] == 0:
                    return i, j

    def is_goal(self, state):
        return state == self.goal

    def get_neighbors(self, state):
        neighbors = []
        x, y = self.find_blank(state)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                new_state = copy.deepcopy(state)
                new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
                neighbors.append(new_state)
        return neighbors

    def manhattan_distance(self, state):
        distance = 0
        for i in range(self.size):
            for j in range(self.size):
                value = state[i][j]
                if value == 0:
                    continue
                target_x, target_y = divmod(value - 1, self.size)
                distance += abs(i - target_x) + abs(j - target_y)
        return distance

    def misplaced_tiles(self, state):
        return sum(
            1 for i in range(self.size) for j in range(self.size)
            if state[i][j] != 0 and state[i][j] != self.goal[i][j]
        )

    def is_solvable(self, state):
        flat_state = [num for row in state for num in row if num != 0]
        inversions = sum(
            1 for i in range(len(flat_state)) for j in range(i + 1, len(flat_state))
            if flat_state[i] > flat_state[j]
        )
        if self.size % 2 == 1:
            return inversions % 2 == 0
        else:
            blank_row = next(i for i, row in enumerate(state) if 0 in row)
            blank_row_from_bottom = self.size - blank_row
            return (inversions + blank_row_from_bottom) % 2 == 0


class PuzzleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("8数码 / 15数码问题")
        self.size = 3
        self.initial_entries = []
        self.goal_entries = []
        self.frames = []  # Stores frames for the GIF animation

        self.create_widgets()

    def create_widgets(self):
        # Puzzle size selection
        size_frame = tk.LabelFrame(self.root, text="拼图设置")
        size_frame.pack(padx=10, pady=5, fill="x")
        tk.Label(size_frame, text="拼图大小 (3x3 or 4x4): ").pack(side=tk.LEFT, padx=5)
        self.size_var = tk.StringVar(value="3")
        size_menu = tk.OptionMenu(size_frame, self.size_var, "3", "4", command=self.update_grid_size)
        size_menu.pack(side=tk.LEFT)

        # Input initial and goal state
        self.input_frame = tk.LabelFrame(self.root, text="输入初始状态和目标状态")
        self.input_frame.pack(padx=10, pady=5, fill="x")
        self.create_grid()

        # Search buttons
        search_frame = tk.LabelFrame(self.root, text="选择算法")
        search_frame.pack(padx=10, pady=5, fill="x")
        tk.Button(search_frame, text="深度优先搜索 (DFS)", command=self.run_dfs).pack(side=tk.LEFT, padx=5)
        tk.Button(search_frame, text="广度优先搜索 (BFS)", command=self.run_bfs).pack(side=tk.LEFT, padx=5)
        tk.Button(search_frame, text="A* 搜索 (曼哈顿)", command=self.run_astar_manhattan).pack(side=tk.LEFT, padx=5)
        tk.Button(search_frame, text="A* 搜索 (错位)", command=self.run_astar_misplaced).pack(side=tk.LEFT, padx=5)
        tk.Button(search_frame, text="随机决策搜索", command=self.run_random_decision_search).pack(side=tk.LEFT, padx=5)

        # Results display
        self.results_frame = tk.LabelFrame(self.root, text="结果展示")
        self.results_frame.pack(padx=10, pady=5, fill="both", expand=True)
        self.results_text = tk.Text(self.results_frame, height=10, width=50, wrap="word")
        self.results_text.pack(padx=5, pady=5, fill="both", expand=True)

        # GIF Display
        self.gif_label = tk.Label(self.root)
        self.gif_label.pack(padx=10, pady=10)

    def create_grid(self):
        """根据拼图大小动态创建输入网格，清除旧网格并重新生成。"""
        for widget in self.input_frame.winfo_children():
            widget.destroy()

        # 创建两个子框架来分别容纳初始状态和目标状态
        initial_frame = tk.Frame(self.input_frame)
        initial_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        goal_frame = tk.Frame(self.input_frame)
        goal_frame.pack(side=tk.LEFT, padx=10, pady=5)

        tk.Label(initial_frame, text="初始状态:").pack()
        self.initial_entries = self.create_entry_grid(initial_frame)

        tk.Label(goal_frame, text="目标状态:").pack()
        self.goal_entries = self.create_entry_grid(goal_frame)

    def create_entry_grid(self, parent):
        """动态创建输入框网格，每个格子对应一个数字输入框。"""
        frame = tk.Frame(parent)
        frame.pack()
        entries = []
        for i in range(self.size):
            row_entries = []
            for j in range(self.size):
                entry = tk.Entry(frame, width=3, justify="center", font=("Arial", 14))
                entry.grid(row=i, column=j, padx=2, pady=2)

                # 限制输入为合法数字
                def validate_input(action, value):
                    if action == "1":  # 输入字符时
                        return value.isdigit() and 0 <= int(value) <= self.size ** 2 - 1
                    return True

                validate_cmd = parent.register(validate_input)
                entry.config(validate="key", validatecommand=(validate_cmd, "%d", "%P"))
                row_entries.append(entry)
            entries.append(row_entries)
        return entries

    def get_grid_values(self, entries):
        """从输入框中提取数字，返回拼图的二维数组。"""
        try:
            return [[int(entries[i][j].get()) for j in range(self.size)] for i in range(self.size)]
        except ValueError:
            messagebox.showerror("输入错误", "请确保所有格子里填的是数字！")
            return None

    def update_grid_size(self, size):
        """更新拼图大小并重新生成输入网格。"""
        self.size = int(size)
        self.create_grid()

    def display_results(self, algorithm, nodes, steps, time_taken, longest_path):
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"算法: {algorithm}\n")
        self.results_text.insert(tk.END, f"总节点数: {nodes}\n")
        self.results_text.insert(tk.END, f"步骤数: {steps}\n")
        self.results_text.insert(tk.END, f"用时: {time_taken:.4f} 秒\n")
        self.results_text.insert(tk.END, f"最长搜索链: {len(longest_path)} 步\n")
        self.results_text.insert(tk.END, "具体路径:\n")
        for state in longest_path:
            self.results_text.insert(tk.END, f"{state}\n")

    def create_frame_image(self, state):
        """创建一个表示当前状态的图像帧。"""
        img_size = 300
        cell_size = img_size // self.size
        img = Image.new("RGB", (img_size, img_size), "white")
        draw = ImageDraw.Draw(img)
        for i in range(self.size):
            for j in range(self.size):
                value = state[i][j]
                if value != 0:
                    x0, y0 = j * cell_size, i * cell_size
                    x1, y1 = x0 + cell_size, y0 + cell_size
                    draw.rectangle([x0, y0, x1, y1], fill="lightblue", outline="black")
                    draw.text((x0 + cell_size // 3, y0 + cell_size // 3), str(value), fill="black")
        return img

    def save_gif(self, frames):
        """保存GIF文件，设置循环次数为1，确保GIF能够持续播放"""
        if frames:
            frames[0].save("solution.gif", save_all=True, append_images=frames[1:], duration=1000, loop=0)

    # def play_gif(self):
    #     """播放生成的GIF。"""
    #     if os.path.exists("solution.gif"):
    #         img = Image.open("solution.gif")
    #         img_tk = ImageTk.PhotoImage(img)
    #         self.gif_label.config(image=img_tk)
    #         self.gif_label.image = img_tk
    #         self.root.after(1000, self.play_gif)  # Refresh the image every 1000 ms
    def play_gif(self):
        """播放生成的GIF。"""
        if os.path.exists("solution.gif"):
            gif = Image.open("solution.gif")
            self.gif_frames = []
            try:
                while True:
                    self.gif_frames.append(ImageTk.PhotoImage(gif.copy()))
                    gif.seek(len(self.gif_frames))
            except EOFError:
                pass

            def update_frame(index=0):
                if self.gif_frames:
                    self.gif_label.config(image=self.gif_frames[index])
                    self.root.after(1000, update_frame, (index + 1) % len(self.gif_frames))

            update_frame()

    def run_search(self, search_func, heuristic=None):
        """运行给定的搜索算法，并展示结果。"""
        self.initial_state = self.get_grid_values(self.initial_entries)
        self.goal_state = self.get_grid_values(self.goal_entries)

        if not self.initial_state or not self.goal_state:
            return

        puzzle = Puzzle(self.initial_state, self.goal_state)

        if not puzzle.is_solvable(self.initial_state):
            messagebox.showerror("无解", "此初始状态无法到达目标状态！")
            return

        start_time = time.time()
        if heuristic:
            nodes, steps, longest_path = search_func(puzzle, heuristic)
        else:
            nodes, steps, longest_path = search_func(puzzle)
        end_time = time.time()

        self.display_results(search_func.__name__, nodes, steps, end_time - start_time, longest_path)

        # 创建动画帧并保存为GIF
        self.frames = [self.create_frame_image(state) for state in longest_path]
        self.save_gif(self.frames)
        self.play_gif()

    def run_random_decision_search(self):
        """触发随机决策搜索算法，保持与其他搜索一致的格式。"""
        self.run_search(random_decision_search)

    def run_bfs(self):
        self.run_search(bfs)

    def run_dfs(self):
        self.run_search(dfs)

    def run_astar_manhattan(self):
        self.run_search(a_star, heuristic=lambda puzzle, state: puzzle.manhattan_distance(state))

    def run_astar_misplaced(self):
        self.run_search(a_star, heuristic=lambda puzzle, state: puzzle.misplaced_tiles(state))


def dfs(puzzle):
    stack = [(puzzle.initial, 0, [puzzle.initial])]
    visited = set()
    visited.add(tuple(tuple(row) for row in puzzle.initial))
    nodes = 0
    longest_path = []  # 追踪最长路径

    while stack:
        state, depth, path = stack.pop()  # 出栈操作
        nodes += 1

        # 更新最长路径
        if len(path) > len(longest_path):
            longest_path = path

        if puzzle.is_goal(state):
            return nodes, depth, longest_path

        for neighbor in puzzle.get_neighbors(state):
            neighbor_tuple = tuple(tuple(row) for row in neighbor)
            if neighbor_tuple not in visited:
                visited.add(neighbor_tuple)  # 标记邻居为已访问
                stack.append((neighbor, depth + 1, path + [neighbor]))  # 更新路径并加入栈

    return nodes, -1, longest_path  # 如果没有找到解，返回无解


def bfs(puzzle):
    queue = deque([(puzzle.initial, 0, [puzzle.initial])])
    visited = set()
    visited.add(tuple(tuple(row) for row in puzzle.initial))
    nodes = 0
    longest_path = []  # 追踪最长路径

    while queue:
        state, depth, path = queue.popleft()
        nodes += 1

        # 更新最长路径
        if len(path) > len(longest_path):
            longest_path = path

        if puzzle.is_goal(state):
            return nodes, depth, path
        for neighbor in puzzle.get_neighbors(state):
            neighbor_tuple = tuple(tuple(row) for row in neighbor)
            if neighbor_tuple not in visited:
                visited.add(neighbor_tuple)
                queue.append((neighbor, depth + 1, path + [neighbor]))

    return nodes, -1, [],longest_path


def a_star(puzzle, heuristic):
    open_set = []
    heapq.heappush(open_set, (heuristic(puzzle, puzzle.initial), 0, puzzle.initial, [puzzle.initial]))
    visited = set()
    visited.add(tuple(tuple(row) for row in puzzle.initial))
    nodes = 0
    longest_path = []

    while open_set:
        _, depth, state, path = heapq.heappop(open_set)
        nodes += 1
        if len(path) > len(longest_path):  # Track longest path
            longest_path = path
        if puzzle.is_goal(state):
            return nodes, depth, longest_path
        for neighbor in puzzle.get_neighbors(state):
            neighbor_tuple = tuple(tuple(row) for row in neighbor)
            if neighbor_tuple not in visited:
                visited.add(neighbor_tuple)
                g = depth + 1
                f = g + heuristic(puzzle, neighbor)
                heapq.heappush(open_set, (f, g, neighbor, path + [neighbor]))

    return nodes, -1, longest_path


def random_decision_search(puzzle):
    """改进版的随机决策搜索算法，加入启发式选择"""
    current_state = puzzle.initial
    path = [current_state]
    visited = set()
    visited.add(tuple(tuple(row) for row in current_state))
    nodes = 1

    max_steps = 1000  # 最大步数，避免无限循环
    max_stagnation = 100  # 如果状态不再变化超过100步，则重新选择起始状态
    
    stagnation_count = 0  # 记录状态不变的次数

    while len(path) <= max_steps:
        if puzzle.is_goal(current_state):
            return nodes, len(path), path
        
        # 获取所有可能的邻居
        neighbors = puzzle.get_neighbors(current_state)
        
        # 计算每个邻居的曼哈顿距离（启发式）
        neighbor_scores = [(neighbor, puzzle.manhattan_distance(neighbor)) for neighbor in neighbors]
        
        # 选择曼哈顿距离最小的邻居（可以调整为随机选择，但使用启发式有助于更快解决）
        next_state = min(neighbor_scores, key=lambda x: x[1])[0]
        
        # 如果该状态未被访问过，则添加到路径中
        if tuple(tuple(row) for row in next_state) not in visited:
            visited.add(tuple(tuple(row) for row in next_state))
            path.append(next_state)
            nodes += 1
            current_state = next_state
            stagnation_count = 0  # 状态有变化，重置不变计数
        else:
            stagnation_count += 1
        
        # 如果多次没有进展，则随机重启（避免死循环）
        if stagnation_count > max_stagnation:
            current_state = random.choice(neighbors)  # 重启时随机选择一个邻居
            stagnation_count = 0

    return nodes, -1, path  # 如果超过最大步数或无法找到解，返回无解



if __name__ == "__main__":
    root = tk.Tk()
    app = PuzzleApp(root)
    root.mainloop()
