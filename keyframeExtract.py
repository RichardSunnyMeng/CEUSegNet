import cv2 as cv
import numpy as np
import os
import glob
import shutil

datapath = "I:\\10-11\\ceusIMAGE\\XIANAI2\\*"

target_path = "I:\\10-11\\key-frame\\XIANAI2"

w = 2
n = 30

work_dirs = glob.glob(datapath)

for work_dir in work_dirs:
    targetwork_dir = target_path
    for sub in work_dir.split(os.path.sep)[4:]:
        targetwork_dir = os.path.join(targetwork_dir, sub)

    #计算灰度图
    img_list = glob.glob(os.path.join(work_dir, '*.jpg'))
    # img_prefix = img_list[0].split(os.path.sep)[-1].split('frame')[0]
    hist_list = {}
    for img_path in img_list:
        number = img_path.split(os.path.sep)[-1].split('frame')[-1].split('.')[0]
        number = int(number)
        img = cv.imread(img_path)
        while img is None:
            img = cv.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
            print("img_path ", img_path, " open error")
        img = img[144:591, 65: 425, :]
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        hist = cv.calcHist([img], [0], None, [256], [0.0, 255.0]) # 256 * 1
        hist_list[number] = hist

    # 关键帧提取
    delta_f = {}
    key_number = []
    ordered_hist_list = sorted(hist_list)
    for i, img_number in enumerate(ordered_hist_list):
        if i < w:
            key_number.append(img_number)
            continue
        if i + w >= len(hist_list.keys()):
            break
        sum_i = 0
        for neighborhood in range(-w, w + 1, 1):
            sum_i = sum_i + abs(hist_list[ordered_hist_list[i + neighborhood]] - hist_list[ordered_hist_list[i]]).sum()
        delta_f[img_number] = sum_i
    ordered_deltaf = sorted(delta_f.items(), key=lambda x: x[1], reverse=True)
    ordered_deltaf = [list(ech)[0] for ech in ordered_deltaf]
    key_number.extend(ordered_deltaf[:n])

    # 关键帧保存
    if os.path.exists(targetwork_dir):
        shutil.rmtree(targetwork_dir)
    os.makedirs(targetwork_dir)

    for number in key_number:
        # img_name = img_prefix + 'frame' + str(number) + '.jpg'
        img_name = 'frame' + str(number) + '.jpg'
        img = cv.imread(os.path.join(work_dir, img_name))
        cv.imwrite(os.path.join(targetwork_dir, str(number) + '.jpg'), img)

