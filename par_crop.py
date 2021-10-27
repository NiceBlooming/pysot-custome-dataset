from os.path import join, isdir
from os import listdir, mkdir, makedirs
import cv2
import numpy as np
import glob
import xml.etree.ElementTree as ET
from concurrent import futures
import sys
import time

UOT_base_path = 'E:\\Datasets\\tracking\\UOT100'
ann_base_path = UOT_base_path


# Print iterations progress (thanks StackOverflow)
def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr = "{0:." + str(decimals) + "f}"
    percents = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()


def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz - 1) / (bbox[2] - bbox[0])
    b = (out_sz - 1) / (bbox[3] - bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop


def pos_s_2_bbox(pos, s):
    return [pos[0] - s / 2, pos[1] - s / 2, pos[0] + s / 2, pos[1] + s / 2]


def crop_like_SiamFC(image, bbox, context_amount=0.5, exemplar_size=127, instanc_size=255, padding=(0, 0, 0)):
    target_pos = [(bbox[2] + bbox[0]) / 2., (bbox[3] + bbox[1]) / 2.]
    target_size = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (instanc_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    z = crop_hwc(image, pos_s_2_bbox(target_pos, s_z), exemplar_size, padding)
    x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), instanc_size, padding)
    return z, x


def crop_video(video, crop_path, instanc_size):
    video_crop_base_path = join(crop_path, video)
    if not isdir(video_crop_base_path):
        makedirs(video_crop_base_path)

    video_base_path = join(ann_base_path, video)
    # 读取目录下的img目录下所有图片
    img_base_path = join(video_base_path, "img")
    # img_file = sorted(listdir(join(video_base_path, "img")))
    img_file = sorted(glob.glob(join(video_base_path, 'img', '*.jpg')))
    # 提取目录下的groundtruth_rect.txt文件
    txt_file = open(join(video_base_path, "groundtruth_rect.txt"))
    # 以行的形式读取文件
    lines = txt_file.readlines()
    print('lines: ', len(lines))

    for i, line in enumerate(lines):
        img_path = join(img_base_path, str(i + 1) + '.jpg')
        im = cv2.imread(img_path)
        avg_chans = np.mean(im, axis=(0, 1))
        # for object_iter in objects:
        trackid = 0
        # 将数据保存在data数据中
        data = line.split()
        # 转化为int类型
        bndbox = list(map(float, data))

        bbox = [int(bndbox[0]), int(bndbox[1]),
                int(bndbox[0]+bndbox[2]+1), int(bndbox[1]+bndbox[3]+1)]
        z, x = crop_like_SiamFC(im, bbox, instanc_size=instanc_size, padding=avg_chans)
        cv2.imwrite(join(video_crop_base_path, '{:d}.{:d}.z.jpg'.format(i+1, trackid)), z)
        cv2.imwrite(join(video_crop_base_path, '{:d}.{:d}.x.jpg'.format(i+1, trackid)), x)


def main(instanc_size=511, num_threads=24):
    uot_dataset = ['AntiguaTurtle', 'ArmyDiver1', 'ArmyDiver2', 'ArmyDiver3', 'Ballena', 'BallisticMissile1',
                   'BallisticMissile2', 'BlueFish1', 'BlueFish2', 'BoySwimming', 'CenoteAngelita', 'CleverOctopus',
                   'ClickerAndTarget', 'CoconutOctopus1', 'CoconutOctopus2', 'ColourChangingSquid', 'CoralGardenSea1',
                   'CoralGardenSea2', 'CrabTrap', 'CrayFish', 'CressiGuillaumeNeri1', 'CressiGuillaumeNeri2',
                   'Cuttlefish', 'DeepOceanLostWorld', 'DeepSeaFish', 'DefenseInTheSea1', 'DefenseInTheSea2',
                   'Diving360Degree1', 'Diving360Degree2', 'Diving360Degree3', 'Dolphin1', 'Dolphin2',
                   'ElephantSeals', 'FightToDeath', 'FightingEels1', 'FightingEels2', 'FishFollowing', 'Fisherman',
                   'FishingAdventure', 'FishingBait', 'FlukeFishing1', 'FlukeFishing2', 'FreeDiver1', 'FreeDiver2',
                   'GarryFish', 'GiantCuttlefish1', 'GiantCuttlefish2', 'GreenMoreyEel1', 'GreenMoreyEel2',
                   'GreenMoreyEel3',
                   'GuillaumeNery', 'HappyTurtle1', 'HappyTurtle2', 'HappyTurtle3', 'HeartShape', 'HoverFish1',
                   'HoverFish2',
                   'JerkbaitBites', 'Kleptopus1', 'Kleptopus2', 'LargemouthBass', 'LittleMonster', 'Lobsters1',
                   'Lobsters2',
                   'MantaRescue1', 'MantaRescue2', 'MantaRescue3', 'MantaRescue4', 'MantisShrimp', 'MississippiFish',
                   'MonsterCreature1', 'MonsterCreature2', 'MuckySecrets1', 'MuckySecrets2', 'MythBusters',
                   'NeryClimbing',
                   'OceanFloorSensor', 'Octopus1', 'Octopus2', 'PinkFish', 'PlayingTurtle', 'RedSeaReptile',
                   'Rocketman',
                   'ScubaDiving1', 'ScubaDiving2', 'SeaDiver', 'SeaDragon', 'SeaTurtle1', 'SeaTurtle2', 'SeaTurtle3',
                   'SharkCloseCall1', 'SharkCloseCall2', 'SharkSuckers1', 'SharkSuckers2', 'Skagerrak', 'SofiaRocks1',
                   'SofiaRocks2',
                   'Steinlager', 'Submarine', 'ThePassage']
    if "AntiguaTurtle" in uot_dataset:
        print("sss")
    else:
        print("sssssss")

    crop_path = './crop{:d}'.format(instanc_size)
    if not isdir(crop_path):
        mkdir(crop_path)

    videos = sorted(listdir(ann_base_path))
    # videos = ["SofiaRocks2"]
    n_videos = len(videos)
    with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        fs = [executor.submit(crop_video, video, crop_path, instanc_size) for video in videos]
        for i, f in enumerate(futures.as_completed(fs)):
            # Write progress to error so that it can be seen
            printProgress(i, n_videos, suffix='Done ', barLength=40)


if __name__ == '__main__':
    since = time.time()
    main(511, 12)
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

