import os
from subprocess import call

import matplotlib.pyplot as plt
import scipy.misc
import torch
from matplotlib.patches import Ellipse
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np

from nms import non_max_suppression
from train import Net12, Net24
from utils import run_detector_pyramid, read_gt


def filter_rects_with_24net(net24, image, rects):
    softmax = torch.nn.Softmax()

    filtered_rects = []
    for x1, y1, x2, y2, _ in rects:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        crop = scipy.misc.imresize(image[y1:y2, x1:x2, :], [24, 24])

        crop = crop.transpose(2, 0, 1).astype(np.float32) / 255.
        crop = torch.Tensor(crop)
        if args.cuda:
            crop = crop.cuda()
        out = net24(Variable(crop, volatile=True).unsqueeze(0))
        out = softmax(out)
        score = out.data.cpu().numpy()[0, 1]

        if score > 0.1:
            filtered_rects.append((x1, y1, x2, y2, score))

    return np.array(filtered_rects).reshape(-1, 5)


def main():
    net12 = Net12()
    net12.load_state_dict(torch.load(args.net12_checkpoint, map_location=lambda storage, loc: storage))
    net12.eval()

    if args.cuda:
        net12.cuda()

    net24 = None
    if args.net24_checkpoint:
        net24 = Net24()
        net24.load_state_dict(torch.load(args.net24_checkpoint, map_location=lambda storage, loc: storage))
        net24.eval()

        if args.cuda:
            net24.cuda()

    with open(os.path.join(args.fddb_dir, 'FDDB-folds/FDDB-fold-01.txt')) as f:
        file_list = f.read().split('\n')[:-1]

    gt = read_gt(os.path.join(args.fddb_dir, 'FDDB-folds/FDDB-fold-01-ellipseList.txt'))

    output_lines = []
    n_rects = []
    for image_path in tqdm(file_list):
        image = scipy.misc.imread(os.path.join(args.fddb_dir, 'images', image_path) + '.jpg', mode='RGB')
        rects = run_detector_pyramid(net12, image, 12, min_face_size=24, threshold=0.05)

        if args.net24_checkpoint:
            rects = filter_rects_with_24net(net24, image, rects)
            rects = non_max_suppression(rects, overlap_thresh=0.4)

        n_rects.append(len(rects))
        output_lines.append(image_path)
        output_lines.append(str(len(rects)))
        ellipses = []
        for x1, y1, x2, y2, score in rects:
            major_axis_radius = (x2 - x1) * 0.5
            minor_axis_radius = (y2 - y1) * 0.5 * 1.2
            angle = 0.0
            center_x = (x1 + x2) * 0.5
            center_y = (y1 + y2) * 0.5 - minor_axis_radius * 0.2
            output_lines.append('{} {} {} {} {} {}'.format(
                major_axis_radius, minor_axis_radius, angle, center_x, center_y, score))

            ellipses.append(Ellipse([center_x, center_y], major_axis_radius * 2, minor_axis_radius * 2, angle, fc='none', lw=int(2**(2*score)), ec='b'))

        if args.debug:
            fix, ax = plt.subplots(1)
            ax.imshow(image)
            for e in ellipses:
                ax.add_patch(e)
            for center_x, center_y, major_axis_radius, minor_axis_radius, angle in gt[image_path]:
                e = Ellipse([center_x, center_y], major_axis_radius * 2, minor_axis_radius * 2, angle,
                            fc='none', lw=3, ec='r')
                ax.add_patch(e)
            plt.show()

    print('falses per image: ', sum(n_rects) / len(n_rects))

    with open(os.path.join(args.output_dir, 'fold-01-out.txt'), 'w') as f:
        f.write('\n'.join(output_lines))

    call([os.path.join(args.fddb_dir, 'evaluation/runEvaluate.pl'), args.output_dir])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--fddb-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--net12-checkpoint', required=True)
    parser.add_argument('--net24-checkpoint')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--no-cuda', action='store_true')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if not args.output_dir.endswith('/'):
        args.output_dir += '/'

    main()
