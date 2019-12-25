# -*- coding: utf-8 -*-

import dlib
import numpy as np
import utils

dlib_face_predictor_model_location = utils.get_conf_prop('dlib', 'dlib_face_predictor_model_location')
dlib_face_recognition_model_location = utils.get_conf_prop('dlib', 'dlib_face_recognition_model_location')

# face detector
detector = dlib.get_frontal_face_detector()

# face landmark predictor
predictor = dlib.shape_predictor(dlib_face_predictor_model_location)

# face encoder
encoder = dlib.face_recognition_model_v1(dlib_face_recognition_model_location)


def get_all_face_bounding_box(rgb_img):
    """
    find all face bounding box in an image

    :param rgb_img: RGB image to process. Shape: (height, width, 3)
    :type rgb_img: numpy.ndarray
    :return: All face bounding boxes in an image
    :rtype: dlib.rectangles
    """
    assert rgb_img is not None

    try:
        return detector(rgb_img, 1)
    except Exception as e:
        print 'Warning: {}'.format(e)
        # In rare cases, exceptions are thrown.
        return []


def get_largest_face_bounding_box(rgb_img):
    """
    find the largest face bounding box in an image.

    :param rgb_img: RGB image to process. Shape: (height, width, 3)
    :type rgb_img: numpy.ndarray
    :return: The largest face bounding boxes in an image
    :rtype: dlib.rectangles
    """
    faces = get_all_face_bounding_box(rgb_img)

    if len(faces) == 0:
        return None
    else:
        return max(faces, key=lambda rect: rect.width() * rect.height())


def get_face_landmarks(rgb_img, bb=None):
    if bb is None:
        bb = get_largest_face_bounding_box(rgb_img)

    try:
        return predictor(rgb_img, bb)
    except Exception as e:
        print 'Warning {}'.format(e)
        return None


def get_face_reps(rgb_img, bb=None):
    landmarks = get_face_landmarks(rgb_img, bb)

    if landmarks is None:
        return None

    try:
        return encoder.compute_face_descriptor(rgb_img, landmarks, 1)
    except Exception as e:
        print 'Warning {}'.format(e)
        return None









