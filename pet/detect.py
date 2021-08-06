import cv2
import darknet
import numpy as np

ConfigFile = "./yolo.cfg"
DataFile = "./pet.data"
WeightsFile = "./pet.weights"
ImgW = 224
ImgH = 224

Net, ClassNames, ClassColors = darknet.load_network(ConfigFile, DataFile, WeightsFile)


def detect(network, class_names, img):
    darknet_img = darknet.make_image(ImgH, ImgW, 3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (ImgH, ImgW))
    darknet.copy_image_from_bytes(darknet_img, img.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_img)
    darknet.free_image(darknet_img)
    return detections


if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    detection = detect(Net, ClassNames, frame)
    print(detection)
