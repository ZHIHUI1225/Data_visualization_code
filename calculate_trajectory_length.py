"""
计算优化轨迹的总长度
基于Optimization_withSC_path4warehouse.json数据
"""

import json
import numpy as np

# 读取数据
json_path = r"d:\Data_visualization_code\result\MAPS\warehouse\Optimization_withSC_path4warehouse.json"
with open(json_path, 'r') as f:
    data = json.load(f)

# 提取参数
phi_values = np.array(data["Optimization_phi"])  # 弧线转角 (radians)
l_values = np.array(data["Optimization_l"])      # 直线段长度 (pixels)
r_values = np.array(data["Optimization_r"])      # 弧线半径 (pixels)

# 像素到米的转换比例
PIXEL_TO_METER = 0.0023

print("=" * 60)
print("轨迹长度计算 - Warehouse场景")
print("=" * 60)
print(f"\n数据文件: {json_path}")
print(f"坐标系: {data['coordinate_frame']}")
print(f"转换比例: {PIXEL_TO_METER} m/pixel\n")

# 分析数据结构
print(f"Phi数组长度: {len(phi_values)}")
print(f"L数组长度: {len(l_values)}")
print(f"R数组长度: {len(r_values)}")

# 计算弧线段长度
# Arc length = |radius × angle|
# 使用前6个phi值对应6个r值
arc_lengths_pixel = []
arc_lengths_meter = []

print("\n" + "-" * 60)
print("弧线段 (Arc Segments):")
print("-" * 60)
for i in range(min(len(r_values), len(phi_values))):
    arc_length_px = abs(r_values[i] * phi_values[i])
    arc_length_m = arc_length_px * PIXEL_TO_METER
    arc_lengths_pixel.append(arc_length_px)
    arc_lengths_meter.append(arc_length_m)
    print(f"Arc {i+1}: r={r_values[i]:8.2f} px, φ={phi_values[i]:7.4f} rad "
          f"→ L={arc_length_px:8.2f} px = {arc_length_m:6.4f} m")

total_arc_pixel = sum(arc_lengths_pixel)
total_arc_meter = sum(arc_lengths_meter)
print(f"\n弧线段总长度: {total_arc_pixel:.2f} pixels = {total_arc_meter:.4f} m")

# 计算直线段长度
print("\n" + "-" * 60)
print("直线段 (Line Segments):")
print("-" * 60)
line_lengths_meter = []
for i, l in enumerate(l_values):
    l_meter = l * PIXEL_TO_METER
    line_lengths_meter.append(l_meter)
    print(f"Line {i+1}: {l:8.2f} px = {l_meter:6.4f} m")

total_line_pixel = sum(l_values)
total_line_meter = sum(line_lengths_meter)
print(f"\n直线段总长度: {total_line_pixel:.2f} pixels = {total_line_meter:.4f} m")

# 总长度
print("\n" + "=" * 60)
print("总轨迹长度 (TOTAL TRAJECTORY LENGTH):")
print("=" * 60)
total_pixel = total_arc_pixel + total_line_pixel
total_meter = total_arc_meter + total_line_meter

print(f"弧线段:  {total_arc_pixel:8.2f} px = {total_arc_meter:7.4f} m ({total_arc_meter/total_meter*100:.1f}%)")
print(f"直线段:  {total_line_pixel:8.2f} px = {total_line_meter:7.4f} m ({total_line_meter/total_meter*100:.1f}%)")
print(f"{'─' * 60}")
print(f"总长度:  {total_pixel:8.2f} px = {total_meter:7.4f} m")
print("=" * 60)

# 额外信息
if len(phi_values) > len(r_values):
    print(f"\n注意: Phi数组有{len(phi_values)}个值，但只有{len(r_values)}个半径值")
    print(f"未使用的Phi值: {phi_values[len(r_values):]}")