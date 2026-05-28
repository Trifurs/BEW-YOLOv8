# coding=utf-8
# @Time : 2024/1/27 16:58
# @Author : Trifurs
# @File : test1.py
# @Software : PyCharm


from ultralytics import YOLO

# Load a model
model = YOLO("../weights/yolov8n.pt")  # load a pretrained model (recommended for training)

results = model("../data/bus.jpg")  # predict on an image

