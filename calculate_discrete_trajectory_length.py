"""
计算warehouse场景下所有机器人的离散轨迹总长度
通过计算相邻点之间的欧几里得距离求和
"""

import json
import numpy as np
from pathlib import Path

# 数据路径
data_dir = Path(r"d:\Data_visualization_code\result\MAPS\warehouse")
robot_files = [
    "robot0_discrete_trajectory_warehouse.json",
    "robot1_discrete_trajectory_warehouse.json",
    "robot2_discrete_trajectory_warehouse.json",
    "robot3_discrete_trajectory_warehouse.json"
]

print("=" * 70)
print("机器人轨迹长度计算 - Warehouse场景 (离散轨迹数据)")
print("=" * 70)

total_all_robots = 0.0
robot_lengths = []

for robot_file in robot_files:
    file_path = data_dir / robot_file

    # 读取数据
    with open(file_path, 'r') as f:
        data = json.load(f)

    robot_id = data['robot_id']
    x_positions = np.array(data['discrete_trajectory']['x_positions'])
    y_positions = np.array(data['discrete_trajectory']['y_positions'])
    total_points = data['discrete_trajectory']['total_points']

    # 计算轨迹长度: 相邻点之间的欧几里得距离求和
    distances = []
    for i in range(len(x_positions) - 1):
        dx = x_positions[i+1] - x_positions[i]
        dy = y_positions[i+1] - y_positions[i]
        dist = np.sqrt(dx**2 + dy**2)
        distances.append(dist)

    trajectory_length = sum(distances)
    robot_lengths.append(trajectory_length)
    total_all_robots += trajectory_length

    # 输出单个机器人信息
    print(f"\n机器人 {robot_id}:")
    print(f"  路径点数量: {total_points}")
    print(f"  实际点数量: {len(x_positions)}")
    print(f"  路径段数量: {len(distances)}")
    print(f"  轨迹总长度: {trajectory_length:.6f} m ({trajectory_length * 100:.2f} cm)")

    # 起点和终点
    print(f"  起点: ({x_positions[0]:.4f}, {y_positions[0]:.4f})")
    print(f"  终点: ({x_positions[-1]:.4f}, {y_positions[-1]:.4f})")

    # 直线距离
    straight_distance = np.sqrt((x_positions[-1] - x_positions[0])**2 +
                                 (y_positions[-1] - y_positions[0])**2)
    print(f"  起终点直线距离: {straight_distance:.6f} m")
    print(f"  轨迹/直线比率: {trajectory_length/straight_distance:.3f}")

    # 统计信息
    avg_segment = np.mean(distances)
    max_segment = np.max(distances)
    min_segment = np.min(distances)
    print(f"  平均段长度: {avg_segment:.6f} m")
    print(f"  最大段长度: {max_segment:.6f} m")
    print(f"  最小段长度: {min_segment:.6f} m")

# 总结
print("\n" + "=" * 70)
print("总体统计:")
print("=" * 70)
for i, length in enumerate(robot_lengths):
    percentage = (length / total_all_robots) * 100
    print(f"机器人 {i}: {length:10.6f} m ({length*100:8.2f} cm) - {percentage:5.2f}%")

print("-" * 70)
print(f"总轨迹长度: {total_all_robots:10.6f} m ({total_all_robots*100:8.2f} cm)")
print(f"平均每机器人: {total_all_robots/len(robot_files):10.6f} m ({total_all_robots/len(robot_files)*100:8.2f} cm)")
print("=" * 70)

# 额外分析
print("\n轨迹复杂度分析:")
print("-" * 70)
for i, robot_file in enumerate(robot_files):
    file_path = data_dir / robot_file
    with open(file_path, 'r') as f:
        data = json.load(f)

    waypoints = data.get('waypoints', [])
    print(f"机器人 {i}: {len(waypoints)} 个航点 (waypoint IDs: {waypoints})")