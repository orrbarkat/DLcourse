import os
import random
from glob import glob

import numpy as np
import scipy.misc
from tqdm import tqdm
project_root = dict(ben='/home/ben/PycharmProjects/DLcourse',
                    orrbarkat='/Users/orrbarkat/repos/deep_learning')
voc_root = os.path.join(project_root[os.getlogin()], 'VOCdevkit/VOC2007')
person_images = set()

with open(os.path.join(voc_root, 'ImageSets/Main/person_trainval.txt')) as f:
    for line in f:
        im, has_person = line.split()
        if int(has_person) == 1:
            person_images.add(im)

background_images = []
for im_path in tqdm(glob(os.path.join(voc_root, 'JPEGImages/*.jpg')), desc='Reading images'):
    image_id, _ = os.path.splitext(os.path.basename(im_path))
    if image_id not in person_images:
        im = scipy.misc.imread(im_path)
        background_images.append(im)
        for _ in range(5):
            background_images.append(scipy.misc.imresize(im, 0.5))

n_crops = 300000
crop_size = 12
random_crops = np.empty([n_crops, 3, crop_size, crop_size], dtype=np.float32)
for i in tqdm(range(n_crops), desc='Generating crops'):
    image = random.choice(background_images)

    crop_up = random.randint(0, image.shape[0] - crop_size)
    crop_left = random.randint(0, image.shape[1] - crop_size)
    crop = image[crop_up:crop_up + crop_size, crop_left:crop_left + crop_size]
    random_crops[i] = np.array(crop).transpose(2, 0, 1).astype(np.float32) / 255

np.savez_compressed(os.path.join(project_root[os.getlogin()], 'EX2/EX2_data/bg_12'), bg_12=random_crops)
