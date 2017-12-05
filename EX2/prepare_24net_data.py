import random
from itertools import count

import numpy as np
import torch
from tqdm import tqdm

from train import Net12
from utils import convert_torchfile_to_npz, get_background_image_pyramids, run_detector


def main():
    # generate faces data
    convert_torchfile_to_npz(args.aflw_path)

    # generate bg data
    bg_image_pyramids = get_background_image_pyramids(args.voc_dir)

    net12 = Net12()
    net12.load_state_dict(torch.load(args.net12_checkpoint, map_location=lambda storage, loc: storage))
    if args.cuda:
        net12.cuda()
    net12.eval()

    crop_size = 24
    random_crops = np.empty([args.n_bg, 3, crop_size, crop_size], dtype=np.float32)
    t = tqdm(count(), desc='Generating crops', total=args.n_bg)
    i = 0
    for _ in t:
        pyramid = random.choice(bg_image_pyramids)
        pyramid_i = random.choice(range(1, len(pyramid)))
        rects = run_detector(net12, pyramid[pyramid_i], image_scale=0.5, rect_size=12, cuda=args.cuda)
        for x1, y1, x2, y2, score in rects:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if y2 >= pyramid[pyramid_i - 1].shape[0] or x2 >= pyramid[pyramid_i - 1].shape[1]:
                continue
            random_crops[i] = pyramid[pyramid_i - 1][y1:y2, x1:x2, :].transpose(2, 0, 1).astype(np.float32) / 255.
            i += 1
            t.update()

            if i >= args.n_bg:
                t.close()
                np.savez_compressed(args.bg_output_path, data=random_crops)
                return


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--aflw-path', required=True)
    parser.add_argument('--bg-output-path', required=True)
    parser.add_argument('--voc-dir', required=True)
    parser.add_argument('--net12-checkpoint', required=True)
    parser.add_argument('--n-bg', default=200000)
    parser.add_argument('--no-cuda', action='store_true')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    main()
