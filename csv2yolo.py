import csv
import os
from PIL import Image

classes = input("input classes(split class name by comma)\n>>").split(",")

def get_class(label: list) -> int:
    for obj_class in classes:
        if label[1] == obj_class:
            return classes.index(obj_class)


def get_regularized_pos(label: list) -> (int, int):
    im = Image.open(img_source_path + label[0])
    img_w, img_h = im.size
    x = ((int(label[3]) + (int(label[3]) + int(label[5]))) / 2) / img_w
    y = ((int(label[4]) + (int(label[4]) + int(label[6]))) / 2) / img_h
    return round(x, 6), round(y, 6)


def get_regularized_size(label: list) -> (int, int):
    im = Image.open(img_source_path + label[0])
    img_w, img_h = im.size
    x = int(label[5]) / img_w
    y = int(label[6]) / img_h
    return round(x, 6), round(y, 6)


csv_file_path = input("input origin index file path\n>>")
img_source_path = input("input image source path\n>>") + "/"
img_path = input("input where images/indexes should be in darknet\n>>") + "/"
with open(csv_file_path) as data:
    reader = csv.reader(data, delimiter=",")
    print(reader)
    files = list()
    try:
        os.makedirs("./index")
    except FileExistsError:
        pass

    for Label in reader:
        files.append(img_path + Label[0])

        print(Label)
        print(get_regularized_pos(Label))
        print(get_regularized_size(Label))
        pos_x, pos_y = get_regularized_pos(Label)
        width, height = get_regularized_size(Label)

        with open("./index/" + Label[0].replace(".jpg", ".txt"), "a") as file:
            file.write(f"{get_class(Label)} {pos_x} {pos_y} {width} {height}\n")

    with open("train.txt", "w") as file:
        print("\n".join(files))
        file.write("\n".join(files))

        # with open("./index/" + label[0].replace(".jpg", "") + ".txt", "a") as file:
        #     file.write(f"{get_class(label)} {label[3]} {label[4]} {label[5]} {label[6]}\n")

with open("./obj.names", "w") as file:
    file.write("\n".join(classes))
