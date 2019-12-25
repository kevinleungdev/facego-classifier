# -*- coding: utf-8 -*-

import ConfigParser
import dlib
import os


# global variables
IMAGE_TYPES = ['.jpg', '.jpeg', '.png']

# global configuration
config_parser = ConfigParser.ConfigParser()

config_filename = os.path.expandvars('./conf.ini')

# load config file
config_parser.read(config_filename)


def is_image(filename):
    _, ext = os.path.splitext(filename)
    # check a file is image or not by its postfix
    return ext.lower() in IMAGE_TYPES


def get_conf_prop(section, option):
    return config_parser.get(section, option)


def dlib_rect_to_rect(rect, shape):
    return [max(0, rect.left()), max(0, rect.top()),
            min(rect.right(), shape[1]), min(rect.bottom(), shape[0])]


def rect_to_dlib_rect(rect):
    return dlib.rectangle(rect[0], rect[1], rect[2], rect[3])