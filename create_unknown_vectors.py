# -*- coding: utf-8 -*-

import argparse
import cv2
import utils
import sys
import numpy as np
import random
import os

from dlib_api import get_face_reps
from os.path import join, exists


def get_rep(img_path):
    bgr_img = cv2.imread(img_path)

    if bgr_img is None:
        return None

    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    rep = get_face_reps(rgb_img)
    return rep


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_dir', type=str, default=utils.get_conf_prop('dataset', 'unknown_dir'))
    parser.add_argument('--nrof_images', type=int, default=200)
    parser.add_argument('--output_file', type=str, default=utils.get_conf_prop('model', 'unknown_file'))

    return parser.parse_args(argv)


def main(args):
    if exists(args.image_dir):
        imgs = [join(args.image_dir, img_name) for img_name in os.listdir(args.image_dir) if utils.is_image(img_name)]

        samples = random.sample(imgs, args.nrof_images)

        reps = []
        for sample in samples:
            # extract 128-D features
            rep = get_rep(sample)

            if rep is None:
                print 'Skip %s because rep is None' % sample
                continue

            reps.append(rep)

        print '%d image has been represented in total' % len(reps)

        # matrix: 200L(row) * 128L(column)
        all_reps = np.row_stack(reps)

        # save npy file
        np.save(args.output_file, all_reps)
    else:
        print 'Directory not found: ', args.image_dir


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))