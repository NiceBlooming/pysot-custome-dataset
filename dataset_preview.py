from os.path import join, isdir
from os import listdir, mkdir, makedirs
import cv2
import numpy as np
import glob
import xml.etree.ElementTree as ET
from concurrent import futures
import sys
import time

dataset_base_path = 'E:\Datasets\\tracking\\UOT32'


def main(video=""):
    video_base_path = join(dataset_base_path, video)

    # 读取目录下的img目录下所有图片
    img_base_path = join(video_base_path, "img")
    # img_file = glob.glob(join(video_base_path, 'img', '*.jpg'))
    # 提取目录下的groundtruth_rect.txt文件
    txt_file = open(join(video_base_path, "groundtruth_rect.txt"))
    # 以行的形式读取文件
    lines = txt_file.readlines()

    for i, line in enumerate(lines):
        img_path = join(img_base_path, str(i+1)+'.jpg')
        im = cv2.imread(img_path)
        # 将数据保存在data数据中
        data = line.split()
        # 转化为int类型
        bndbox = list(map(float, data))

        bbox = [int(bndbox[0]), int(bndbox[1]),
                int(bndbox[0] + bndbox[2] + 1), int(bndbox[1] + bndbox[3] + 1)]
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.imshow(video, im)
        cv2.waitKey(40)


if __name__ == '__main__':
    since = time.time()
    main("ArmyDiver1")
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
