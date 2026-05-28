# coding=utf-8
# @Time : 2024/3/14 13:04
# @Author : Trifurs
# @File : run_botnet.py
# @Software : PyCharm


import torch
from ultralytics import YOLO


def main():
    torch.cuda.empty_cache()
    # Load a model
    model = YOLO('yolo_ese.yaml').load('yolov8n.pt')  # build a new model from YAML

    # Train the model
    results = model.train(data='../data/flood_data_car1.yaml',
                          epochs=1000, imgsz=640, batch=-1)


if __name__ == '__main__':
    main()


