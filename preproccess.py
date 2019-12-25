# -*- coding: utf-8 -*-

import argparse
import cv2
import numpy as np
import os
import sys
import utils

from dlib_api import get_largest_face_bounding_box
from os.path import join, exists, isfile, isdir, basename, splitext


def flip(rgb_img, filename):
    """
    flip an image horizontally

    :param rgb_img: an image to process
    :param filename: the name of image
    :return: （filename, transform_img)
    """
    transform_img = np.fliplr(rgb_img)
    base, ext = splitext(filename)

    return base + '_flip' + ext, cv2.cvtColor(transform_img, cv2.COLOR_RGB2BGR)


def color_jitter(rgb_img, filename):
    """
    color jittering an image

    :param rgb_img: an image to process
    :param filename: the name of image
    :return: filename, transform_img)
    """
    h, w, c = rgb_img.shape

    # random generate jitter/noise from -10px to 10px here
    noise = np.random.randint(0, 10, (h, w, c))

    jitter_img = np.zeros_like(rgb_img)
    jitter_img[:, :, 0] = noise[:, :, 0]
    jitter_img[:, :, 1] = noise[:, :, 1]
    jitter_img[:, :, 2] = noise[:, :, 2]

    transform_img = cv2.add(rgb_img, jitter_img)
    # clip image
    np.clip(transform_img, 0, 255, out=transform_img)

    base, ext = splitext(filename)
    return base + '_jitter' + ext, cv2.cvtColor(transform_img, cv2.COLOR_RGB2BGR)


def process_images(cls, items, output_path):
    if not exists(output_path):
        print 'create director {} for class {}'.format(output_path, cls)
        os.mkdir(output_path)

    for item in items:
        img = cv2.imread(item)

        # from BGR to RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # find largest face in a given image
        bb = get_largest_face_bounding_box(rgb_img)

        if bb is None:
            print "skip image '{}' because no face found!".format(item)
            continue

        # change dlib rectangle to (left, top, right, bottom)
        rect = utils.dlib_rect_to_rect(bb, rgb_img.shape)

        cropped_img = rgb_img[rect[1]:rect[3], rect[0]:rect[2]].copy()
        cropped_bgr = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)

        # save raw image in output directory
        filename = basename(item)
        cv2.imwrite(join(output_path, filename), cropped_bgr)

        # flip the image and save
        flip_name, flip_img = flip(cropped_img, filename)
        cv2.imwrite(join(output_path, flip_name), flip_img)

        # color jittering the image and save
        jitter_name, jitter_img = color_jitter(cropped_img, filename)
        cv2.imwrite(join(output_path, jitter_name), jitter_img)

    print 'class {} done!'.format(cls)


# list all images for given path
def list_all_images(path):
    return [join(path, item) for item in os.listdir(path) if utils.is_image(item)]


def main(args):
    root = args.input_dir
    assert exists(root)

    output = args.output_dir
    if not exists(output):
        print 'create output dir: ', output
        os.mkdir(output)

    # list subdirectories
    for cls in os.listdir(root):
        abspath = join(root, cls)

        if not isdir(abspath):
            print 'skip non-directory file: ', cls
            continue

        raw_image_paths = list_all_images(abspath)

        nrof_raw_images = len(raw_image_paths)
        print '{} has {} images in total'.format(abspath, nrof_raw_images)

        if nrof_raw_images == 0:
            continue

        cls_output_path = join(output, cls)

        if (args.override and exists(cls_output_path)) or not exists(cls_output_path):
            # pre-process images
            process_images(cls, raw_image_paths, cls_output_path)
        else:
            print 'Skip handle class `%s`' % cls


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    input_dir = utils.get_conf_prop('dataset', 'raw_dir')
    output_dir = utils.get_conf_prop('dataset', 'output_dir')

    parser.add_argument('--input_dir', type=str, default=input_dir,
                        help='directory that containing raw pictures')

    parser.add_argument('--output_dir', type=str, default=output_dir,
                        help='directory that saving generated pics')

    parser.add_argument('--override', type=bool, default=False)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
