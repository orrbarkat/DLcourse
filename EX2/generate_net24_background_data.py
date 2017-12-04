import os
from glob import glob

import numpy as np
import scipy.misc
import scipy.ndimage
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn as nn
# import matplotlib.pyplot as plt


from net12 import Net12

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
        # for _ in range(5):
        #     background_images.append(scipy.misc.imresize(im, 0.5))
        #     break

net12 = Net12()
net12.load_state_dict(torch.load(os.path.join(project_root[os.getlogin()], 'EX2/log/model.checkpoint')))
net12.eval()
softmax_2d = nn.Softmax2d()

rect_size = crop_size = 12
fp_crops = []
image_count = 200000

# loop
for im in tqdm(background_images, desc='Predicting image crops'):
    im = im.transpose(2, 0, 1).astype(np.float32) / 255.
    out = net12(Variable(torch.Tensor(im), volatile=True).unsqueeze(0))
    out = softmax_2d(out)
    scores = out.data.numpy()[0, 1, ...]
    ys, xs = np.where(scores > 0.5)
    for y, x in zip(ys, xs):
        score = scores[y, x]
        crop = im[:,y:y + crop_size, x:x + crop_size]
        crop = scipy.ndimage.interpolation.zoom(crop, (1., 2., 2.), order=1)
        fp_crops.append(crop)
        if len(fp_crops) >= image_count:
            print('cropped {} images'.format(len(fp_crops)))
            np.savez_compressed(os.path.join(project_root[os.getlogin()], 'EX2/EX2_data/bg_24'), bg_24=fp_crops)
            exit(0)
