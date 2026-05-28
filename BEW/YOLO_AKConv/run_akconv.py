# coding=utf-8
# @Time : 2024/1/29 11:15
# @Author : Trifurs
# @File : code.py
# @Software : PyCharm


import torch
from ultralytics import YOLO


def main():
    torch.cuda.empty_cache()
    # Load a model
    model = YOLO('yolov8_akconv_ori.yaml').load('yolov8n.pt')  # build a new model from YAML
    # model = YOLO('yolov8.yaml').load('yolov8n.pt')

    # Train the model
    results = model.train(data='../data/flood_data.yaml',
                          epochs=100, imgsz=640, batch=8)


if __name__ == '__main__':
    main()
