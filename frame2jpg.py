from os.path import join
from os import listdir
import cv2

UOT_base_path = 'E:\\Datasets\\tracking\\UOT100'
subset = ["CenoteAngelita", "LargemouthBass", "Octopus1", "Octopus2", "SeaDiver", "Steinlager",
          "WhaleAtBeach1", "WhaleAtBeach2", "WhaleDiving"]

videos = sorted(listdir(UOT_base_path))
for sub in subset:
    print(sub)
    video_base_path = join(UOT_base_path, sub)
    # 读取目录下的img目录下所有图片
    img_base_path = join(video_base_path, "img")
    img_file = listdir(img_base_path)
    # 处理命名错误的图片
    for img_name in img_file:
        cha_img_name = img_name[5:]
        cha_img_name = int(cha_img_name.split('.')[0]) + 1
        im = cv2.imread(join(img_base_path, img_name))
        cv2.imwrite(join(img_base_path, str(cha_img_name) + ".jpg"), im)



print('done!')
