import os
import glob
import cv2 as cv
import numpy as np
import shutil
import json

sample = 2

target_path = ""

path_list = glob.glob("key-frame-new\\*")

i = 0

for work_path in path_list:
    # 检索json文件
    json_list = glob.glob(os.path.join(work_path, "*.json"))

    if json_list.__len__() != 0 and "frame" in json_list[0]:
        json_list.sort(key=lambda x: int(x.split(os.path.sep)[-1].split(".")[0].split("frame")[-1]))
    else:
        json_list.sort(key=lambda x: int(x.split(os.path.sep)[-1].split(".")[0]))
    if len(json_list) == 0:
        continue

    json_list = json_list[0::sample] # 采样为21帧

    json_path = json_list[i % len(json_list)]
    i = i + 1

    # 准备目标目录
    if "Adenocarcinoma" in work_path:
        name = str(i) + "-0"
    else:
        name = str(i) + "-1"
    work_target_path = os.path.join(target_path, name)
    if os.path.exists(work_target_path):
        shutil.rmtree(work_target_path)
    os.makedirs(work_target_path)

    # 保存超声图像 (430, 146) width=357
    USpath = os.path.join(work_path, json_path.split(os.path.sep)[-1].split(".")[0] + ".jpg")
    img = cv.imread(USpath)
    if img is None:
        print(USpath)
        USpath = os.path.join(work_path, json_path.split(os.path.sep)[-1].split(".")[0] + ".png")
        img = cv.imread(USpath)

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    patch = img[146: 146 + 357, 430: 430 + 357]
    cv.imwrite(os.path.join(work_target_path, "US.jpg"), patch)

    # 保存造影图像 (67, 145) width=357   保存mask (67, 145) width=357
    CEUSpath = os.path.join(work_path, json_path.split(os.path.sep)[-1].split(".")[0] + ".jpg")
    img = cv.imread(CEUSpath)
    patch = img[145: 145 + 357, 67: 67 + 357, :]

    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    f = open(json_path, 'r')
    f_json = json.load(f)
    points = f_json["shapes"][0]["points"]

    cv.fillPoly(mask, np.array([points], dtype=np.int32), 255)
    mask_patch = mask[145: 145 + 357, 67: 67 + 357]

    cv.imwrite(os.path.join(work_target_path, "img.jpg"), patch)
    cv.imwrite(os.path.join(work_target_path, "mask.jpg"), mask_patch)


