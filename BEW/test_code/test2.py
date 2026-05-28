# coding=utf-8
# @Time : 2024/1/28 14:46
# @Author : Trifurs
# @File : test2.py
# @Software : PyCharm


from ultralytics import YOLO


def main():
    # Load a model
    # model = YOLO('yolov8n.yaml')  # build a new model from YAML
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    # model = YOLO('../data/flood_data.yaml').load('../weights/yolov8n.pt')  # build from YAML and transfer weights

    # Train the model
    results = model.train(data='../data/flood_data_car1.yaml', epochs=1000, imgsz=640, batch=-1)


if __name__ == '__main__':
    main()
