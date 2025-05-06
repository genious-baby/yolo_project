import sys
import cv2
import os
import pandas as pd
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QFileDialog, QHBoxLayout
)
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import QTimer, Qt
from ultralytics import YOLO

class YOLOApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚗 YOLOv8 车流量检测系统")
        self.resize(1000, 800)

        # UI组件
        self.image_label = QLabel()
        self.image_label.setFixedSize(960, 540)
        self.label_up = QLabel("Bottom to Top: 0")
        self.label_down = QLabel("Top to Bottom: 0")
        self.label_up.setFont(QFont("Arial", 24))
        self.label_down.setFont(QFont("Arial", 24))
        self.btn_select = QPushButton("选择视频")
        self.btn_export = QPushButton("导出数据")

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        count_layout = QHBoxLayout()
        count_layout.addWidget(self.label_down)
        count_layout.addWidget(self.label_up)
        layout.addLayout(count_layout)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_select)
        button_layout.addWidget(self.btn_export)
        layout.addLayout(button_layout)
        self.setLayout(layout)

        # 模型加载（直接使用本地路径）
        self.model = YOLO("yolov8n.pt")

        # 初始化
        self.cap = None
        self.car_count_down = 0
        self.car_count_up = 0
        self.track_history = {}
        self.crossed_ids = set()
        self.log_data = []

        # 定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # 按钮连接
        self.btn_select.clicked.connect(self.select_video)
        self.btn_export.clicked.connect(self.export_data)

    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.avi)")
        if file_path:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(file_path)
            self.reset_counters()
            self.timer.start(30)

    def reset_counters(self):
        self.car_count_down = 0
        self.car_count_up = 0
        self.track_history.clear()
        self.crossed_ids.clear()
        self.log_data.clear()
        self.label_down.setText("Top to Bottom: 0")
        self.label_up.setText("Bottom to Top: 0")

    def export_data(self):
        df = pd.DataFrame(self.log_data, columns=["Timestamp", "Direction", "Total_Up", "Total_Down"])
        df.to_csv("traffic_log.csv", index=False)
        print("数据已导出至 traffic_log.csv")

    def update_frame(self):
        if not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            return

        results = self.model.track(frame, persist=True, verbose=False)
        frame_height, frame_width = frame.shape[:2]
        middle_line_y = frame_height // 2
        cv2.line(frame, (0, middle_line_y), (frame_width, middle_line_y), (0, 0, 255), 2)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()

            for box, track_id, cls in zip(boxes, track_ids, clss):
                x, y, w, h = box
                y_center = int(y)
                current_zone = "top" if y_center < middle_line_y else "bottom"

                if track_id not in self.track_history:
                    self.track_history[track_id] = current_zone
                    continue

                last_zone = self.track_history[track_id]
                if track_id not in self.crossed_ids:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    if last_zone == "top" and current_zone == "bottom":
                        self.car_count_down += 1
                        self.label_down.setText(f"Top to Bottom: {self.car_count_down}")
                        self.crossed_ids.add(track_id)
                        self.log_data.append([timestamp, "Down", self.car_count_up, self.car_count_down])
                    elif last_zone == "bottom" and current_zone == "top":
                        self.car_count_up += 1
                        self.label_up.setText(f"Bottom to Top: {self.car_count_up}")
                        self.crossed_ids.add(track_id)
                        self.log_data.append([timestamp, "Up", self.car_count_up, self.car_count_down])
                self.track_history[track_id] = current_zone

                cv2.rectangle(frame,
                              (int(x - w / 2), int(y - h / 2)),
                              (int(x + w / 2), int(y + h / 2)),
                              (0, 255, 0), 2)

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio)
        self.image_label.setPixmap(scaled_pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOApp()
    window.show()
    sys.exit(app.exec())
