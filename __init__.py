from . import darknet
import os
import numpy as np
import cv2


class Detector:
    def __init__(self, img_class: str):
        """첫 번째 인자로 이미지 클래스(디렉터리명(pet 등)) 을 받음"""
        self.image_class = img_class
        self.network, self.class_names, self.class_colors = darknet.load_network(
            f"./SmartBin-Detector/{img_class}/{img_class}.cfg",
            f"./SmartBin-Detector/{img_class}/{img_class}.data",
            f"./SmartBin-Detector/{img_class}/{img_class}.weights"
        )
        self.width = darknet.network_width(self.network)
        self.height = darknet.network_height(self.network)

    def _load_image(self, image):
        darknet_image = darknet.make_image(self.width, self.height, 3)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(darknet_image, image.tobytes())
        return darknet_image

    def detect(self, image: np.ndarray):
        """opencv로 불러온 이미지 numpy array를 인지로 받아 결과를 돌려줌"""
        darknet_image = self._load_image(image)
        detections = darknet.detect_image(self.network, self.class_names, darknet_image)
        darknet.free_image(darknet_image)
        return detections
