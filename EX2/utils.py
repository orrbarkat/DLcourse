import os
from glob import glob

import numpy as np
import torch
import torchfile
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm
import scipy.misc

from nms import non_max_suppression


def convert_torchfile_to_npz(torchfile_path):
    # convert torchfile to npz
    faces_data_ = torchfile.load(torchfile_path)
    faces_data = np.empty((len(faces_data_), *faces_data_[1].shape), dtype='float32')
    for i, im in enumerate(faces_data_.values()):
        faces_data[i] = im
    np.savez_compressed(os.path.splitext(torchfile_path)[0], data=faces_data)


def get_background_image_pyramids(voc_root):
    person_images = set()
    with open(os.path.join(voc_root, 'ImageSets/Main/person_trainval.txt')) as f:
        for line in f:
            input_image, has_person = line.split()
            if int(has_person) == 1:
                person_images.add(input_image)

    bg_image_pyramids = []
    for im_path in tqdm(glob(os.path.join(voc_root, 'JPEGImages/*.jpg')), desc='Reading images'):
        image_id, _ = os.path.splitext(os.path.basename(im_path))
        if image_id not in person_images:
            input_image = scipy.misc.imread(im_path)
            pyramid = [input_image]
            for _ in range(5):
                if pyramid[-1].shape[0] > 24 and pyramid[-1].shape[1] > 24:
                    pyramid.append(scipy.misc.imresize(pyramid[-1], 0.5))
            bg_image_pyramids.append(pyramid)

    return bg_image_pyramids


def read_gt(gt_path):
    gt = dict()
    with open(gt_path) as f:
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
                gt[cur_file].append(
                    [center_x, center_y, major_axis_radius, minor_axis_radius, np.rad2deg(angle)])
                count += 1

                if count == n:
                    state = 'header'

    return gt


def gt_to_bounding_boxes(gt):
    for image_id, ellipses in gt.items():
        bounding_boxes = [(cx - w/2, cy-h/2, cx + w/2, cy+h/2) for cx, cy, w, h, a in ellipses]
        gt[image_id] = np.array(bounding_boxes)

    return gt


def to_variables(*tensors, cuda=None, **kwargs):
    if cuda is None:
        cuda = torch.cuda.is_available()

    variables = []
    for t in tensors:
        if cuda:
            t = t.cuda()
        variables.append(Variable(t, **kwargs))

    return variables


def run_detector(model, image, image_scale, rect_size, threshold=0.05, nms=True, cuda=None):
    """

    :param model: Detector model to run
    :param image: Input image
    :param image_scale: (input_image_size / original_image_size), detections will be scaled by this factor
    :param rect_size: Detection rect size
    :param threshold: Detection score threshold
    :param nms: If true, apply non-max-suppression
    :return:
    """
    softmax_2d = nn.Softmax2d()

    image = image.transpose(2, 0, 1).astype(np.float32) / 255.
    image = torch.Tensor(image)
    if cuda:
        image = image.cuda()
    out = model(Variable(image, volatile=True).unsqueeze(0))
    out = softmax_2d(out)
    scores = out.data.cpu().numpy()[0, 1, ...]
    ys, xs = np.where(scores > threshold)

    rects = []
    for y, x in zip(ys, xs):
        score = scores[y, x]
        # multipy x,y by 2 because output is half the input
        x, y = 2*x, 2*y
        rects.append([x, y, x + rect_size, y + rect_size, score])
    rects = np.array(rects).reshape(-1, 5)

    # scale rects
    rects[:, :4] /= image_scale

    if nms:
        rects = non_max_suppression(rects, overlap_thresh=0.4)

    return rects


def run_detector_pyramid(model, image, rect_size, threshold=0.05, nms=True, pyramid_factor=0.8, min_face_size=None, cuda=None):
    if min_face_size:
        resize_factor = rect_size / min_face_size
    else:
        resize_factor = 1
    image = scipy.misc.imresize(image, resize_factor)

    image_pyramid = [image]
    min_image_size = rect_size / pyramid_factor
    while True:
        if image_pyramid[-1].shape[0] < min_image_size or image_pyramid[-1].shape[1] < min_image_size:
            break
        image_pyramid.append(scipy.misc.imresize(image_pyramid[-1], pyramid_factor))

    rects = []
    for pyramid_i, im in enumerate(image_pyramid):
        image_scale = pyramid_factor ** pyramid_i * resize_factor
        rects.extend(run_detector(model, im, image_scale, rect_size, threshold, nms, cuda))

    return rects
