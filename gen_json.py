from os.path import join
from os import listdir
import json
import numpy as np
import random

print('load json (raw vid info), please wait 20 seconds~')
uot = json.load(open('uot.json', 'r'))


def check_size(frame_sz, bbox):
    min_ratio = 0.1
    max_ratio = 0.75
    # only accept objects >10% and <75% of the total frame
    area_ratio = np.sqrt((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / float(np.prod(frame_sz)))
    ok = (area_ratio > min_ratio) and (area_ratio < max_ratio)
    return ok


def check_borders(frame_sz, bbox):
    dist_from_border = 0.05 * (bbox[2] - bbox[0] + bbox[3] - bbox[1]) / 2
    ok = (bbox[0] > dist_from_border) and (bbox[1] > dist_from_border) and \
         ((frame_sz[0] - bbox[2]) > dist_from_border) and \
         ((frame_sz[1] - bbox[3]) > dist_from_border)
    return ok


snippets = dict()
n_snippets = 0
n_videos = 0

for subset in uot:
    for video in subset:
        n_videos += 1
        frames = video['frame']
        id_set = []
        id_frames = [[]] * 60  # at most 60 objects
        # 是否为训练集
        train_flag = False
        for f, frame in enumerate(frames):
            objs = frame['objs']
            frame_sz = frame['frame_sz']
            for obj in objs:
                trackid = obj['trackid']
                occluded = obj['occ']
                bbox = obj['bbox']

                if trackid not in id_set:
                    id_set.append(trackid)
                    id_frames[trackid] = []
                id_frames[trackid].append(f)
        if len(id_set) > 0:
            # 随机生成训练集和测试集，比例为0.8和0.2
            if random.randint(1, 10) <= 8:
                snippets[video['base_path']+'_train'] = dict()
                train_flag = True
            else:
                snippets[video['base_path'] + '_val'] = dict()
        for selected in id_set:
            frame_ids = sorted(id_frames[selected])
            sequences = np.split(frame_ids, np.array(np.where(np.diff(frame_ids) > 1)[0]) + 1)
            sequences = [s for s in sequences if len(s) > 1]  # remove isolated frame.
            for seq in sequences:
                snippet = dict()
                for frame_id in seq:
                    frame = frames[frame_id]
                    for obj in frame['objs']:
                        if obj['trackid'] == selected:
                            o = obj
                            continue
                    snippet[frame['img_path'].split('.')[0]] = o['bbox']
                # 随机生成训练集和测试集，比例为0.8和0.2
                if train_flag:
                    snippets[video['base_path']+'_train']['{:d}'.format(selected)] = snippet
                else:
                    snippets[video['base_path'] + '_val']['{:d}'.format(selected)] = snippet
                n_snippets += 1
        print('video: {:d} snippets_num: {:d}'.format(n_videos, n_snippets))

train = {k[:-6]: v for (k, v) in snippets.items() if 'train' in k}
val = {k[:-4]: v for (k, v) in snippets.items() if 'val' in k}

json.dump(train, open('train.json', 'w'), indent=4, sort_keys=False)
json.dump(val, open('val.json', 'w'), indent=4, sort_keys=False)
print('done!')
