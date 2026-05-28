import sys
import torch
import cv2
from typing import Optional
from ultralytics import YOLO
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMainWindow, QApplication, QFileDialog
from PySide6.QtGui import QPixmap, QImage
from main_window_ui import Ui_MainWindow


def convert2QImage(img):
    height, width, channel = img.shape
    return QImage(img, width, height, width * channel, QImage.Format_RGB888)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        # self.model = torch.hub.load("./", "custom", "runs/detect/train5/weights/best.pt", source="local")
        self.model = YOLO("runs/detect/train5/weights/best.pt")
        self.timer = QTimer()
        self.timer.setInterval(1)
        self.video = None
        self.bind_slots()

    def image_pred(self, file_path):
        results = self.model(file_path)
        print(results)
        image = results.render()[0]
        # image = results.orig_img
        return convert2QImage(image)

    def open_image(self):
        print("点击了检测图片!")
        self.timer.stop()
        file_path = QFileDialog.getOpenFileName(
            self, dir="../data/images", filter="*.jpg;*.png;*.jpeg"
        )
        if file_path[0]:
            file_path = file_path[0]
            qimage = self.image_pred(file_path)
            self.input.setPixmap(QPixmap(file_path))
            self.output.setPixmap(QPixmap.fromImage(qimage))

    def video_pred(self):
        ret, frame = self.video.read()
        if not ret:
            self.timer.stop()
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.input.setPixmap(QPixmap.fromImage(convert2QImage(frame)))
            results = self.model(frame)
            image = results.render()[0]
            self.output.setPixmap(QPixmap.fromImage(convert2QImage(image)))
    
    def open_video(self):
        print("点击了检测视频!")
        self.timer.stop()
        file_path = QFileDialog.getOpenFileName(
            self, dir="../data", filter="*.mp4"
        )
        if file_path[0]:
            file_path = file_path[0]
            self.video = cv2.VideoCapture(file_path)
            self.timer.start()
                
    def bind_slots(self):
        self.det_img.clicked.connect(self.open_image)
        self.det_video.clicked.connect(self.open_video)
        self.timer.timeout.connect(self.video_pred)




if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec()
