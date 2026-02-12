import math
import heapq
import random
import numpy as np
import matplotlib.pyplot as plt

# --- 参数配置 ---
XY_RESOLUTION = 0.05  # 网格分辨率 [m]
YAW_RESOLUTION = np.deg2rad(15.0)  # 角度分辨率 [rad]
MOTION_RESOLUTION = 0.1  # 运动积分步长 [m]
MIN_TURN_RADIUS = 0.2  # 最小转弯半径 [m]
GOAL_TOLERANCE_XY = 0.2  # 到达目标的允许误差 [m]
GOAL_TOLERANCE_YAW = np.deg2rad(15.0) # 到达目标的角度误差 [rad]

# 车辆参数
WB = 0.2  # 轴距 (Wheelbase) [m] (这里设小一点以适配小半径)
MAX_STEER = math.atan(WB / MIN_TURN_RADIUS)  # 最大转向角

class Node:
    def __init__(self, x_ind, y_ind, yaw_ind, direction, x_list, y_list, yaw_list, directions, steer, cost, parent_index):
        self.x_ind = x_ind
        self.y_ind = y_ind
        self.yaw_ind = yaw_ind
        self.direction = direction
        self.x_list = x_list
        self.y_list = y_list
        self.yaw_list = yaw_list
        self.directions = directions
        self.steer = steer
        self.cost = cost
        self.parent_index = parent_index

class Path:
    def __init__(self, x_list, y_list, yaw_list, direction_list, cost):
        self.x_list = x_list
        self.y_list = y_list
        self.yaw_list = yaw_list
        self.direction_list = direction_list
        self.cost = cost

class Config:
    def __init__(self, ox, oy, xy_res, yaw_res, min_x_m=None, max_x_m=None, min_y_m=None, max_y_m=None):
        if min_x_m is None or max_x_m is None or min_y_m is None or max_y_m is None:
            min_x_m = min(ox)
            min_y_m = min(oy)
            max_x_m = max(ox)
            max_y_m = max(oy)

        self.min_x = round(min_x_m / xy_res)
        self.min_y = round(min_y_m / xy_res)
        self.max_x = round(max_x_m / xy_res)
        self.max_y = round(max_y_m / xy_res)
        
        self.x_w = round(self.max_x - self.min_x)
        self.y_w = round(self.max_y - self.min_y)

        self.min_x_m = min_x_m
        self.min_y_m = min_y_m
        self.xy_res = xy_res
        self.yaw_res = yaw_res

def calc_motion_inputs():
    # 简单的运动原语：左转，直行，右转
    for steer in [0, -MAX_STEER, MAX_STEER]:
        yield steer

def get_neighbors(curr_node, config, obstacle_map):
    # 生成下一步的节点
    for steer in calc_motion_inputs():
        # 模拟一小段轨迹
        x, y, yaw = curr_node.x_list[-1], curr_node.y_list[-1], curr_node.yaw_list[-1]
        
        # 预测一段路径 (这里简化为一步长，也可以积分多步)
        dist = MOTION_RESOLUTION
        theta = yaw
        
        # 车辆运动学模型 (Bicycle Model)
        x += dist * math.cos(theta)
        y += dist * math.sin(theta)
        theta += dist / WB * math.tan(steer)
        theta = pi_2_pi(theta)

        # 检查是否碰壁
        if not check_collision(x, y, config, obstacle_map):
            continue

        # 计算离散索引
        x_ind = round(x / config.xy_res)
        y_ind = round(y / config.xy_res)
        yaw_ind = round(theta / config.yaw_res)

        # 构造新节点
        # 代价 G = 父节点代价 + 移动代价 + 转向惩罚
        move_cost = MOTION_RESOLUTION
        steer_cost = 0.0 if steer == 0 else 0.1 # 稍微惩罚转向
        new_cost = curr_node.cost + move_cost + steer_cost

        yield Node(x_ind, y_ind, yaw_ind, 1, 
                   [x], [y], [theta], [1], steer, new_cost, None)

def check_collision(x, y, config, obstacle_map):
    ix = round((x - config.min_x_m) / config.xy_res)
    iy = round((y - config.min_y_m) / config.xy_res)

    if ix < 0 or ix >= config.x_w or iy < 0 or iy >= config.y_w:
        return False  # 超出边界

    if obstacle_map[ix][iy]:
        return False  # 碰撞

    return True

def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def calc_heuristic(n1, n2):
    # 启发式函数 H
    # 这里使用 欧几里得距离 + 角度差惩罚
    # 在完整的 Hybrid A* 中，通常使用 Holonomic Heuristic (2D BFS) 来避障
    w = 0.5 # 角度权重的 heuristic
    d = math.hypot(n1.x_list[-1] - n2.x_list[-1], n1.y_list[-1] - n2.y_list[-1])
    angle_diff = abs(pi_2_pi(n1.yaw_list[-1] - n2.yaw_list[-1]))
    return d + w * angle_diff

def hybrid_astar_planning(sx, sy, syaw, gx, gy, gyaw, ox, oy, xy_res, yaw_res,
                          min_x_m=None, max_x_m=None, min_y_m=None, max_y_m=None):
    # 1. 配置空间构建
    config = Config(ox, oy, xy_res, yaw_res, min_x_m, max_x_m, min_y_m, max_y_m)
    
    # 构建栅格地图 (简单起见，仅标记中心点)
    obstacle_map = [[False for _ in range(config.y_w)] for _ in range(config.x_w)]
    for x, y in zip(ox, oy):
        ix = round((x - config.min_x_m) / xy_res)
        iy = round((y - config.min_y_m) / xy_res)
        if 0 <= ix < config.x_w and 0 <= iy < config.y_w:
            obstacle_map[ix][iy] = True

    # 2. 初始化
    start_node = Node(round(sx/xy_res), round(sy/xy_res), round(syaw/yaw_res), 1, 
                      [sx], [sy], [syaw], [1], 0, 0.0, -1)
    goal_node = Node(round(gx/xy_res), round(gy/xy_res), round(gyaw/yaw_res), 1, 
                     [gx], [gy], [gyaw], [1], 0, 0.0, -1)

    open_set = {}
    closed_set = {}
    
    # 索引生成器 key
    def calc_index(node):
        return (node.x_ind, node.y_ind, node.yaw_ind)

    open_set[calc_index(start_node)] = start_node
    
    # 优先队列: (Total Cost F, Index)
    pq = []
    heapq.heappush(pq, (calc_heuristic(start_node, goal_node), calc_index(start_node)))

    final_path = None

    print("开始搜索...")
    
    iter_count = 0
    while True:
        if not pq:
            print("Open set is empty, cannot find path.")
            break
            
        _, c_id = heapq.heappop(pq)
        
        if c_id in open_set:
            current = open_set[c_id]
            del open_set[c_id]
            closed_set[c_id] = current
        else:
            continue
            
        iter_count += 1
        # 可视化搜索过程 (可选，会变慢)
        if iter_count % 200 == 0:
            plt.plot(current.x_list[-1], current.y_list[-1], "xc")
            plt.pause(0.001)

        # 检查是否到达目标区域 (Goal Region)
        # 完整的 Hybrid A* 这里会尝试连接 Dubins 曲线直接射击目标 (Analytic Expansion)
        # 这里简化为：只要进入目标容差范围即认为到达
        dist_to_goal = math.hypot(current.x_list[-1] - gx, current.y_list[-1] - gy)
        ang_diff = abs(pi_2_pi(current.yaw_list[-1] - gyaw))
        
        if dist_to_goal <= GOAL_TOLERANCE_XY and ang_diff <= GOAL_TOLERANCE_YAW:
            print(f"Goal Found! Cost: {current.cost}")
            final_path = current
            break

        # 扩展节点
        for neighbor in get_neighbors(current, config, obstacle_map):
            neighbor_index = calc_index(neighbor)
            
            if neighbor_index in closed_set:
                continue
            
            neighbor.cost += current.cost # 累加路径代价
            neighbor.parent_index = c_id
            
            # 如果不在 open set 或者找到了更优路径
            if neighbor_index not in open_set or open_set[neighbor_index].cost > neighbor.cost:
                neighbor.f = neighbor.cost + calc_heuristic(neighbor, goal_node)
                open_set[neighbor_index] = neighbor
                heapq.heappush(pq, (neighbor.f, neighbor_index))

    # 3. 路径回溯
    if final_path:
        rx, ry, ryaw = [final_path.x_list[-1]], [final_path.y_list[-1]], [final_path.yaw_list[-1]]
        parent_idx = final_path.parent_index
        
        while parent_idx != -1:
            n = closed_set[parent_idx]
            rx.append(n.x_list[-1])
            ry.append(n.y_list[-1])
            ryaw.append(n.yaw_list[-1])
            parent_idx = n.parent_index
            
        rx = rx[::-1]
        ry = ry[::-1]
        ryaw = ryaw[::-1]

        # If we stopped inside the goal tolerance, try to snap a straight segment to the exact goal.
        if math.hypot(rx[-1] - gx, ry[-1] - gy) <= GOAL_TOLERANCE_XY:
            if is_straight_path_collision_free(rx[-1], ry[-1], gx, gy, config, obstacle_map):
                append_straight_segment(rx, ry, ryaw, gx, gy, gyaw, xy_res)

        # Smooth the path using collision-checked shortcutting, then densify for visualization.
        rx, ry, ryaw = smooth_path_shortcut(rx, ry, ryaw, config, obstacle_map, iterations=200)
        rx, ry, ryaw = densify_path(rx, ry, ryaw, xy_res)

        return Path(rx, ry, ryaw, [], final_path.cost)
    
    return None

def is_straight_path_collision_free(sx, sy, gx, gy, config, obstacle_map):
    length = math.hypot(gx - sx, gy - sy)
    if length == 0.0:
        return True
    step = max(config.xy_res * 0.5, 0.01)
    steps = int(math.ceil(length / step))
    for i in range(1, steps + 1):
        ratio = i / steps
        x = sx + (gx - sx) * ratio
        y = sy + (gy - sy) * ratio
        if not check_collision(x, y, config, obstacle_map):
            return False
    return True

def append_straight_segment(rx, ry, ryaw, gx, gy, gyaw, xy_res):
    length = math.hypot(gx - rx[-1], gy - ry[-1])
    if length == 0.0:
        return
    step = max(xy_res * 0.5, 0.01)
    steps = int(math.ceil(length / step))
    for i in range(1, steps + 1):
        ratio = i / steps
        x = rx[-1] + (gx - rx[-1]) * ratio
        y = ry[-1] + (gy - ry[-1]) * ratio
        rx.append(x)
        ry.append(y)
        ryaw.append(gyaw)

def smooth_path_shortcut(rx, ry, ryaw, config, obstacle_map, iterations=200):
    if len(rx) < 3:
        return rx, ry, ryaw

    rng = random.Random(0)
    for _ in range(iterations):
        i = rng.randint(0, len(rx) - 3)
        j = rng.randint(i + 2, len(rx) - 1)

        if is_straight_path_collision_free(rx[i], ry[i], rx[j], ry[j], config, obstacle_map):
            new_rx = rx[:i + 1] + rx[j:]
            new_ry = ry[:i + 1] + ry[j:]
            new_ryaw = rebuild_yaw_from_xy(new_rx, new_ry, ryaw[-1])
            rx, ry, ryaw = new_rx, new_ry, new_ryaw

    return rx, ry, ryaw

def densify_path(rx, ry, ryaw, xy_res):
    if len(rx) < 2:
        return rx, ry, ryaw

    step = max(xy_res * 0.5, 0.01)
    out_x = [rx[0]]
    out_y = [ry[0]]

    for i in range(1, len(rx)):
        sx, sy = out_x[-1], out_y[-1]
        gx, gy = rx[i], ry[i]
        length = math.hypot(gx - sx, gy - sy)
        if length == 0.0:
            continue
        steps = int(math.ceil(length / step))
        for k in range(1, steps + 1):
            ratio = k / steps
            out_x.append(sx + (gx - sx) * ratio)
            out_y.append(sy + (gy - sy) * ratio)

    out_yaw = rebuild_yaw_from_xy(out_x, out_y, ryaw[-1])
    return out_x, out_y, out_yaw

def rebuild_yaw_from_xy(rx, ry, final_yaw):
    if len(rx) == 1:
        return [final_yaw]

    yaw_list = []
    for i in range(1, len(rx)):
        yaw_list.append(math.atan2(ry[i] - ry[i - 1], rx[i] - rx[i - 1]))
    yaw_list.append(final_yaw)
    return yaw_list

# --- 辅助：生成矩形障碍物的点云 ---
def get_rect_points(center_x, center_y, width, height, resolution):
    ox, oy = [], []
    # 填充矩形区域
    min_x = center_x - width / 2.0
    max_x = center_x + width / 2.0
    min_y = center_y - height / 2.0
    max_y = center_y + height / 2.0
    
    for x in np.arange(min_x, max_x, resolution):
        for y in np.arange(min_y, max_y, resolution):
            ox.append(x)
            oy.append(y)
    return ox, oy

# --- 主程序 ---
def main():
    print("Hybrid A* Path Planning Start")
    
    # 1. 定义起点终点
    sx, sy, syaw = 0.2, 0.2, np.deg2rad(0.0)
    gx, gy, gyaw = 1.8, 0.8, np.deg2rad(0.0)
    
    # 2. 定义障碍物 (生成离散点云来模拟栅格)
    ox, oy = [], []
    
    # 矩形障碍物 [x, y, w, h]
    rects = [
        [1.0, 0.5, 0.15, 0.2],
        [0.5, 0.7, 0.1, 0.1]
    ]
    
    for r in rects:
        rx, ry = get_rect_points(r[0], r[1], r[2], r[3], XY_RESOLUTION)
        ox.extend(rx)
        oy.extend(ry)

    # 3. 运行规划
    plt.figure(figsize=(8, 4))
    plt.plot(ox, oy, ".k", label="Obstacle")
    plt.plot(sx, sy, "xr", label="Start")
    plt.plot(gx, gy, "xb", label="Goal")
    plt.grid(True)
    plt.axis("equal")
    plt.xlim(0.0, 2.0)
    plt.ylim(0.0, 1.0)

    # 绘制车辆初始和目标方向
    plt.arrow(sx, sy, 0.1 * math.cos(syaw), 0.1 * math.sin(syaw), fc='r', ec='r', head_width=0.05)
    plt.arrow(gx, gy, 0.1 * math.cos(gyaw), 0.1 * math.sin(gyaw), fc='b', ec='b', head_width=0.05)

    path = hybrid_astar_planning(
        sx, sy, syaw, gx, gy, gyaw, ox, oy, XY_RESOLUTION, YAW_RESOLUTION,
        min_x_m=0.0, max_x_m=2.0, min_y_m=0.0, max_y_m=1.0
    )

    if path:
        plt.plot(path.x_list, path.y_list, "-r", linewidth=2.5, label="Hybrid A* Path")
        plt.legend()
        plt.title("Hybrid A* Path Planning (RRT Dubins Scenario)")
        plt.show()
    else:
        print("Failed to find path")
        plt.show()

if __name__ == '__main__':
    main()