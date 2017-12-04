import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from torch.autograd import Variable
import torch.nn as nn
import scipy.misc
from tqdm import tqdm

from net12 import Net12
from nms import non_max_suppression


def run_detector(model, image, min_face_size=24):
    resize_factor = 12 / min_face_size
    image = scipy.misc.imresize(image, resize_factor)
    pyramid_factor = 0.8
    pyramid_size = 100
    image_pyramid = [image]
    for i in range(pyramid_size - 1):
        if image_pyramid[-1].shape[0] > 16 and image_pyramid[-1].shape[1] > 16:
            image_pyramid.append(scipy.misc.imresize(image_pyramid[-1], pyramid_factor))

    softmax_2d = nn.Softmax2d()
    rects = []
    for pyramid_i, im in enumerate(image_pyramid):
        im = im.transpose(2, 0, 1).astype(np.float32) / 255.
        out = model(Variable(torch.Tensor(im), volatile=True).unsqueeze(0))
        out = softmax_2d(out)
        scores = out.data.numpy()[0, 1, ...]
        ys, xs = np.where(scores > 0.1)
        rect_size = min_face_size * (1/pyramid_factor) ** pyramid_i

        cur_rects = []
        for y, x in zip(ys, xs):
            score = scores[y, x]
            y *= 2 * (1/pyramid_factor) ** pyramid_i / resize_factor
            x *= 2 * (1/pyramid_factor) ** pyramid_i / resize_factor
            cur_rects.append([x,
                              y,
                              x + rect_size,
                              y + rect_size,
                              score])
        rects.extend(non_max_suppression(np.array(cur_rects, np.float32), overlap_thresh=0.4))

    return rects


def main():
    fddb_root = '/home/ben/PycharmProjects/DLcourse/EX2/EX2_data/fddb/images/'

    net12 = Net12()
    net12.load_state_dict(torch.load('/home/ben/PycharmProjects/DLcourse/EX2/log/model.checkpoint'))
    net12.eval()

    with open('/home/ben/PycharmProjects/DLcourse/EX2/EX2_data/fddb/FDDB-folds/FDDB-fold-01.txt') as f:
        file_list = f.read().split('\n')[:-1]

    gt = dict()
    with open('/home/ben/PycharmProjects/DLcourse/EX2/EX2_data/fddb/FDDB-folds/FDDB-fold-01-ellipseList.txt') as f:
        state = 'header'
        cur_file = None
        n = None
        count = 0
        for line in f:
            if state == 'header':
                cur_file = line[:-1]
                gt[cur_file] = []
                state = 'count'
            elif state == 'count':
                n = int(line)
                count = 0
                state = 'ellipses'
            elif state == 'ellipses':
                major_axis_radius, minor_axis_radius, angle, center_x, center_y, score = map(float, line.split())
                gt[cur_file].append(Ellipse([center_x, center_y], major_axis_radius * 2, minor_axis_radius * 2, np.rad2deg(angle), fc='none', lw=3, ec='r'))
                count += 1

                if count == n:
                    state = 'header'

    output_lines = []
    n_rects = []
    for image_path in tqdm(file_list):
        image_path = os.path.join(fddb_root, image_path) + '.jpg'
        image = scipy.misc.imread(image_path, mode='RGB')
        rects = run_detector(net12, image)
        n_rects.append(len(rects))
        output_lines.append(image_path[len(fddb_root):-len('.jpg')])
        output_lines.append(str(len(rects)))
        ellipses = []
        for x1, y1, x2, y2, score in rects:
            major_axis_radius = (x2 - x1) * 0.5
            minor_axis_radius = (y2 - y1) * 0.5 * 1.2
            angle = 0.0
            center_x = (x1 + x2) * 0.5
            center_y = (y1 + y2) * 0.5 - minor_axis_radius * 0.2
            ellipses.append(Ellipse([center_x, center_y], major_axis_radius * 2, minor_axis_radius * 2, angle, fc='none', lw=int(2**(2*score)), ec='b'))
            output_lines.append('{} {} {} {} {} {}'.format(
                major_axis_radius, minor_axis_radius, angle, center_x, center_y, score))
            # left_x = x1
            # top_y = y1
            # width = x2 - x1
            # height = y2 - y1
            # output_lines.append('{} {} {} {} {}'.format(
            #     left_x, top_y, width, height, score))

        # fix, ax = plt.subplots(1)
        # ax.imshow(image)
        # for e in ellipses:
        #     ax.add_patch(e)
        # for e in gt[image_path[len(fddb_root):-len('.jpg')]]:
        #     ax.add_patch(e)
        # # ax.add_patch()
        # plt.show()
    # plt.hist(n_rects, 50)
    # plt.show()

    with open('/home/ben/PycharmProjects/DLcourse/EX2/log/fold-01-out.txt', 'w') as f:
        f.write('\n'.join(output_lines))


if __name__ == '__main__':
    main()