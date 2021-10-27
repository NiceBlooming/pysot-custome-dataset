from os.path import join
from os import listdir
import json
import glob
import xml.etree.ElementTree as ET
from PIL import Image

UOT_base_path = './uot100'
ann_base_path = UOT_base_path
img_base_path = UOT_base_path

uot = []
videos = sorted(listdir(ann_base_path))
s = []
for vi, video in enumerate(videos):
    print('video id : {:03d} / {:03d}'.format(vi, len(videos)))
    v = dict()
    v['base_path'] = video
    v['frame'] = []
    video_base_path = join(ann_base_path, video)
    # 读取目录下的img目录下所有图片
    img_base_path = join(video_base_path, "img")
    img_file = listdir(img_base_path)
    # 获取img的大小
    img = Image.open(join(img_base_path, img_file[0]))
    frame_sz = list(img.size)
    # 提取目录下的groundtruth_rect.txt文件
    txt_file = open(join(video_base_path, "groundtruth_rect.txt"))
    # 以行的形式读取文件
    lines = txt_file.readlines()

    for i, line in enumerate(lines):
        f = dict()
        objs = []
        # 将数据保存在data数据中
        data = line.split()
        # 转化为int类型
        bndbox = list(map(float, data))
        o = dict()
        o['c'] = video
        o['bbox'] = [int(bndbox[0]), int(bndbox[1]), int(bndbox[0]+bndbox[2]+1), int(bndbox[1]+bndbox[3]+1)]
        o['trackid'] = 0
        o['occ'] = 0
        objs.append(o)

        f['frame_sz'] = frame_sz
        f['img_path'] = str(i+1)+'.jpg'
        f['objs'] = objs
        v['frame'].append(f)
    s.append(v)
uot.append(s)

print('save json (raw vid info), please wait 1 min~')
json.dump(uot, open('uot.json', 'w'), indent=4, sort_keys=True)
print('done!')


def process_data(input):
    data = [input[0], input[0] + input[2] + 1, input[1], input[1] + input[3] + 1]
    return data