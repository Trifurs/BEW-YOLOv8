# coding=utf-8
# @Time : 2024/4/12 15:32
# @Author : Trifurs
# @File : test_model.py
# @Software : PyCharm

from ultralytics import YOLO
from PIL import Image
import cv2
import os

img_path = r"..\data\new_car\images\train"
img_list = os.listdir(img_path)

# 加载计划使用的模型
# model = YOLO(r"../YOLO_BiFPN_EffectiveSE/runs/detect/train-all/weights/best.pt")
model = YOLO(r"..\YOLO_BiFPN_EffectiveSE\runs\detect\train-ori\weights\best.pt")

for img in img_list:
    # 加载照片
    im1 = Image.open(os.path.join(img_path, img))

    # 完成标注并保存到本地
    result = model.predict(source=[im1], save=True, conf=0.5)  # save plotted images

    print(result)

