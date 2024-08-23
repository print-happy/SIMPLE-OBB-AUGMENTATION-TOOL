import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def normalize_coords(coords, img_shape):
  """将归一化坐标转换为像素坐标"""
  height, width = img_shape[:2]
  return [(int(x * width), int(y * height)) for x, y in coords]

def draw_obb_on_image(image_path, coords_list, output_filename):
  """在图片上绘制 OBB 列表并保存"""
  # 读取图片
  image = cv2.imread(image_path)
  
  # 遍历坐标列表，绘制每个 OBB
  for coords in coords_list:
    pixel_coords = normalize_coords(coords, image.shape)
    # 使用 cv2.line 函数绘制 OBB 的四条边
    cv2.line(image, tuple(pixel_coords[0]), tuple(pixel_coords[1]), (0, 255, 0), 2)
    cv2.line(image, tuple(pixel_coords[1]), tuple(pixel_coords[2]), (0, 255, 0), 2)
    cv2.line(image, tuple(pixel_coords[2]), tuple(pixel_coords[3]), (0, 255, 0), 2)
    cv2.line(image, tuple(pixel_coords[3]), tuple(pixel_coords[0]), (0, 255, 0), 2)
  
  # 使用 matplotlib 显示并保存图片
  plt.imshow(image)
  plt.axis('off')  # 关闭坐标轴
  plt.savefig(output_filename)  # 保存图片

def process_images_and_coords(coords_path, output_dir):
  """处理图片和坐标数据，并保存标记后的图片"""
  # 读取坐标文件
  coords_dict = {}
  with open(coords_path, 'r') as f:
    for line in f:
      parts = line.strip().split()
      filename = parts[0]
      coords = list(map(float, parts[1:]))
      print(parts[0])
      coords = np.array(coords).reshape(4, 2)
      if filename not in coords_dict:
        coords_dict[filename] = []
      coords_dict[filename].append(coords)
  
  # 确保输出目录存在
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  
  # 遍历字典，处理每个图片和坐标列表
  for filename, coords_list in coords_dict.items():
    image_path = os.path.join('try', filename)
    output_filename = os.path.join(output_dir, filename)
    draw_obb_on_image(image_path, coords_list, output_filename)

# 调用函数
process_images_and_coords('path/to/input/label', 'path/to/output/image')
