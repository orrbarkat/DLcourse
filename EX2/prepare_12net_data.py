import random

import numpy as np
from tqdm import tqdm

from utils import convert_torchfile_to_npz, get_background_image_pyramids


def main():
    # generate faces data
    convert_torchfile_to_npz(args.aflw_path)

    # generate bg data
    bg_image_pyramids = get_background_image_pyramids(args.voc_dir)

    crop_size = 12
    random_crops = np.empty([args.n_bg, 3, crop_size, crop_size], dtype=np.float32)
    for i in tqdm(range(args.n_bg), desc='Generating crops'):
        pyramid = random.choice(bg_image_pyramids)
        image = random.choice(pyramid)

        crop_up = random.randint(0, image.shape[0] - crop_size)
        crop_left = random.randint(0, image.shape[1] - crop_size)
        crop = image[crop_up:crop_up + crop_size, crop_left:crop_left + crop_size]
        random_crops[i] = np.array(crop).transpose(2, 0, 1).astype(np.float32) / 255

    np.savez_compressed(args.bg_output_path, data=random_crops)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--aflw-path', required=True)
    parser.add_argument('--bg-output-path', required=True)
    parser.add_argument('--voc-dir', required=True)
    parser.add_argument('--n-bg', default=200000)
    args = parser.parse_args()

    main()
