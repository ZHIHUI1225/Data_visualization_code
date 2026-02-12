"""检查warehouse环境的RRT*失败原因"""

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

# 像素到米的转换
PIXEL_TO_METER = 0.0023
OBSTACLE_EXPANSION = 0.05  # 米

# 起点和终点
start = np.array([100, 150]) * PIXEL_TO_METER
goal = np.array([700, 570]) * PIXEL_TO_METER

print(f'起点 (米): ({start[0]:.3f}, {start[1]:.3f})')
print(f'终点 (米): ({goal[0]:.3f}, {goal[1]:.3f})')
print(f'直线距离: {np.linalg.norm(goal - start):.3f} m')
print()

# 构建障碍物
obstacles_data = [
    [[28, 48], [800, 48], [800, 18], [28, 18]],  # 顶部墙
    [[28, 650], [800, 650], [800, 620], [28, 620]],  # 底部墙
    [[28, 650], [58, 650], [58, 18], [28, 18]],  # 左墙
    [[770, 650], [800, 650], [800, 18], [770, 18]],  # 右墙
    [[168, 109], [268, 109], [268, 199], [168, 199]],  # 货架1
    [[168, 299], [268, 299], [268, 389], [168, 389]],  # 货架2
    [[168, 489], [268, 489], [268, 579], [168, 579]],  # 货架3
    [[350, 109], [450, 109], [450, 199], [350, 199]],  # 货架4
    [[350, 299], [450, 299], [450, 389], [350, 389]],  # 货架5
    [[350, 489], [450, 489], [450, 579], [350, 579]],  # 货架6
    [[532, 109], [632, 109], [632, 199], [532, 199]],  # 货架7
    [[532, 299], [632, 299], [632, 389], [532, 389]],  # 货架8
    [[532, 489], [632, 489], [632, 579], [532, 579]],  # 货架9
]

obstacles = []
for i, verts in enumerate(obstacles_data):
    verts_m = np.array(verts) * PIXEL_TO_METER
    poly = Polygon(verts_m)
    obstacles.append(poly)

obstacles_union = unary_union(obstacles)
obstacles_expanded = obstacles_union.buffer(OBSTACLE_EXPANSION)

# 检查起点终点是否在膨胀后的障碍物内
start_pt = Point(start)
goal_pt = Point(goal)

start_collision = obstacles_expanded.contains(start_pt)
goal_collision = obstacles_expanded.contains(goal_pt)

print(f'起点碰撞检测: {"碰撞!" if start_collision else "安全"}')
print(f'终点碰撞检测: {"碰撞!" if goal_collision else "安全"}')
print()

# 检查起点到最近障碍物的距离
start_dist = obstacles_union.distance(start_pt)
goal_dist = obstacles_union.distance(goal_pt)

print(f'起点到障碍物最近距离: {start_dist:.4f} m')
print(f'终点到障碍物最近距离: {goal_dist:.4f} m')
print(f'障碍物膨胀距离: {OBSTACLE_EXPANSION} m')
print()

if start_dist < OBSTACLE_EXPANSION:
    print(f'⚠️ 起点太靠近障碍物！膨胀后会被覆盖')
if goal_dist < OBSTACLE_EXPANSION:
    print(f'⚠️ 终点太靠近障碍物！膨胀后会被覆盖')

# 计算通道宽度
print('\n货架布局分析（米）：')
shelves_x = [
    (168 * PIXEL_TO_METER, 268 * PIXEL_TO_METER, '第1列'),
    (350 * PIXEL_TO_METER, 450 * PIXEL_TO_METER, '第2列'),
    (532 * PIXEL_TO_METER, 632 * PIXEL_TO_METER, '第3列'),
]

for i in range(len(shelves_x) - 1):
    gap = shelves_x[i+1][0] - shelves_x[i][1]
    print(f'  {shelves_x[i][2]} -> {shelves_x[i+1][2]}: 间距 = {gap:.4f} m')
    effective_gap = gap - 2 * OBSTACLE_EXPANSION
    print(f'    膨胀后有效宽度: {effective_gap:.4f} m')
    if effective_gap < 0:
        print(f'    ❌ 膨胀后通道完全堵塞!')