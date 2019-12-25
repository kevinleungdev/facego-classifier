# -*- coding: utf-8 -*-

import argparse
import cv2
import dlib_api
import numpy as np
import os
import cPickle as pickle
import cStringIO as StringIO
import random
import sys
import time
import utils

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from os.path import join, exists, isdir

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

TRAINING_SAMPLES_FILE = 'training_samples.pkl'
TESTING_SAMPLES_FILE = 'testing_samples.pkl'


class Image(object):

    def __init__(self, cls, name, path):
        self.cls = cls
        self.name = name
        self.path = path

        rgb = self.get_rgb()
        bb = utils.rect_to_dlib_rect([0, 0, rgb.shape[1], rgb.shape[0]])

        # generate face reps
        encodings = dlib_api.get_face_reps(rgb, bb)
        if encodings is not None:
            file_like = StringIO.StringIO()

            np.save(file_like, encodings)
            file_like.seek(0)

            self.reps = file_like.read().decode('latin-1')
        else:
            self.reps = None

    def __repr__(self):
        return 'image class: {}, path: {}'.format(self.cls, self.path)

    def get_bgr(self):
        try:
            bgr = cv2.imread(self.path)
        except Exception as e:
            print 'Warning: {}'.format(e)
            bgr = None

        return bgr

    def get_rgb(self):
        bgr = self.get_bgr()

        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb = None

        return rgb

    def load_reps(self):
        if self.reps is None:
            return None

        reps_file = StringIO.StringIO()
        reps_file.write(self.reps.encode('latin-1'))
        reps_file.seek(0)

        return np.load(reps_file)


def iter_images(cls, directory):
    for filename in os.listdir(directory):
        if not utils.is_image(filename):
            continue

        img = Image(cls, filename, join(directory, filename))
        if img.reps is not None:
            yield img
        else:
            print 'Skip image {} for its reps is none'.format(img.path)


def get_data(data_file):
    x = []
    y = []

    with open(data_file, 'rb') as infile:
        training_samples_set = pickle.load(infile)

    for sample_list in training_samples_set.values():
        for sample in sample_list:
            sample_reps = sample.load_reps()
            print 'cls: {}, reps[0:5]: {}'.format(sample.cls, sample_reps[0:5])

            x.append(sample_reps)
            y.append(sample.cls)

    return x, y


def get_training_data(data_dir, unknown):
    training_samples_file = join(data_dir, TRAINING_SAMPLES_FILE)

    if exists(training_samples_file):
        x, y = get_data(training_samples_file)

        # total number of classes
        nrof_identities = len(set(y + [-1])) - 1

        # no data
        if nrof_identities == 0:
            return None, None

        # add unknown class
        if unknown:
            nrof_unknown = y.count(-1)
            nrof_identified = len(y) - nrof_unknown

            nrof_unknown_add = (nrof_identified / nrof_identities) - nrof_unknown
            if nrof_unknown_add > 0:
                # load unknown images
                unknown_file = utils.get_conf_prop('model', 'unknown_file')
                unknown_imgs = np.load(unknown_file)

                nrof_unknown_add = min(nrof_unknown_add, len(unknown_imgs))
                print "+ Augmenting with {} unknown images.".format(nrof_unknown_add)

                for rep in unknown_imgs[:nrof_unknown_add]:
                    x.append(rep)
                    y.append(-1)

        return x, y
    else:
        print '{} not found'.format(training_samples_file)
        return None, None


def get_validate_data(data_dir):
    test_samples_file = join(data_dir, TESTING_SAMPLES_FILE)

    if exists(test_samples_file):
        x, y = get_data(test_samples_file)

        if len(x) == 0 or len(y) == 0:
            return None
        else:
            return x, y
    else:
        print '{} not found'.format(test_samples_file)
        return None


def prepare_data(data_dir, train_test_split_ratio):
    """
    The images should be organized in subdirectories
    named by the image's class (who the person is)::
        $ tree directory

        emp_id
        ├── 000001.jpg
        ├── 000002.png
        ...
        └── 00000p.png
        ...
        emp_id_m
        ├── 000001.png
        ├── 000002.jpg
        ...
        └── 00000p.png

    :param data_dir: images directory
    :param train_test_split_ratio: ratio of training
    """
    train_sample_set = {}
    test_sample_set = {}

    start = time.time()

    print 'Preparing data...'
    for cls in os.listdir(data_dir):
        cls_path = join(data_dir, cls)

        if not isdir(cls_path):
            continue

        samples = list(iter_images(cls, cls_path))

        if len(samples) == 0:
            print 'No valid training and testing samples for class %s' % cls
            return

        # shuffle samples
        random.shuffle(samples)

        nrof_samples = len(samples)
        nrof_training_samples = int(nrof_samples * train_test_split_ratio)
        nrof_testing_samples = nrof_samples - nrof_training_samples

        if nrof_training_samples == 0:
            print 'No enough testing samples for class %s' % cls
            nrof_training_samples = nrof_samples
            nrof_testing_samples = 0

        print 'Class `{}` - number of training samples: {}, number of testing samples: {}'\
            .format(cls, nrof_training_samples, nrof_testing_samples)

        train_sample_set[cls] = samples[0:nrof_training_samples]
        if nrof_testing_samples > 0:
            test_sample_set[cls] = samples[nrof_training_samples:]

    # dump samples into pkl
    training_samples_filename = join(data_dir, TRAINING_SAMPLES_FILE)
    with open(training_samples_filename, 'wb') as outfile:
        pickle.dump(train_sample_set, outfile)
    print 'save training samples to %s' % training_samples_filename

    testing_samples_filename = join(data_dir, TESTING_SAMPLES_FILE)
    with open(testing_samples_filename, 'wb') as outfile:
        pickle.dump(test_sample_set, outfile)
    print 'save testing samples to %s' % testing_samples_filename

    duration = time.time() - start
    print 'Prepare data done! it takes %d seconds' % duration


def add_class(data_dir, train_test_split_ratio, classes):
    training_samples_filename = join(data_dir, TRAINING_SAMPLES_FILE)
    testing_samples_filename = join(data_dir, TESTING_SAMPLES_FILE)

    if not exists(training_samples_filename) or not exists(testing_samples_filename):
        print 'training or testing samples not found'
        return

    with open(training_samples_filename, 'rb') as infile:
        training_samples_set = pickle.load(infile)

    with open(testing_samples_filename, 'rb') as infile:
        testing_samples_set = pickle.load(infile)

    for cls in classes:
        cls_path = join(data_dir, cls)

        if not exists(cls_path):
            print 'the directory of class not found: ', cls_path
            continue

        samples = list(iter_images(cls, cls_path))

        if len(samples) == 0:
            print 'No valid training and testing samples for class %s' % cls
            return

        # shuffle samples
        random.shuffle(samples)

        nrof_samples = len(samples)
        nrof_training_samples = int(nrof_samples * train_test_split_ratio)
        nrof_testing_samples = nrof_samples - nrof_training_samples

        if nrof_training_samples == 0:
            print 'No enough testing samples for class %s' % cls
            nrof_training_samples = nrof_samples
            nrof_testing_samples = 0

        print 'Class `{}` - number of training samples: {}, number of testing samples: {}'\
            .format(cls, nrof_training_samples, nrof_testing_samples)

        training_samples_set[cls] = samples[0:nrof_training_samples]
        if nrof_testing_samples > 0:
            testing_samples_set[cls] = samples[nrof_training_samples:]

    with open(training_samples_filename, 'wb') as outfile:
        pickle.dump(training_samples_set, outfile)
    print 'save training samples to %s' % training_samples_filename

    testing_samples_filename = join(data_dir, TESTING_SAMPLES_FILE)
    with open(testing_samples_filename, 'wb') as outfile:
        pickle.dump(testing_samples_set, outfile)
    print 'save testing samples to %s' % testing_samples_filename


def remove_classes(data_dir, classes):
    training_samples_filename = join(data_dir, TRAINING_SAMPLES_FILE)
    testing_samples_filename = join(data_dir, TESTING_SAMPLES_FILE)

    if not exists(training_samples_filename) or not exists(testing_samples_filename):
        print 'training or testing samples not found'
        return

    with open(training_samples_filename, 'rb') as infile:
        training_samples_set = pickle.load(infile)

    with open(testing_samples_filename, 'rb') as infile:
        testing_samples_set = pickle.load(infile)

    for cls in classes:
        if training_samples_set.has_key(cls):
            print 'delete cls %s from training samples' % cls
            del training_samples_set[cls]
        else:
            print 'cls %s not found in training samples' % cls

        if testing_samples_set.has_key(cls):
            print 'delete cls %s from testing samples' % cls
            del testing_samples_set[cls]
        else:
            print 'cls %s not found in testing samples' % cls

    with open(training_samples_filename, 'wb') as outfile:
        pickle.dump(training_samples_set, outfile)
    print 'save training samples to %s' % training_samples_filename

    testing_samples_filename = join(data_dir, TESTING_SAMPLES_FILE)
    with open(testing_samples_filename, 'wb') as outfile:
        pickle.dump(testing_samples_set, outfile)
    print 'save testing samples to %s' % testing_samples_filename


def train_model(data_dir, model_dir, classifier_model_filename, unknown):
    x, y = get_training_data(data_dir, unknown)

    if x is None:
        print 'No training data. Please run `python classifier.py --mode PREPARE` firstly.'

    classes = list(np.unique(y))
    print '{} classes, {} training samples'.format(len(classes), len(x))

    # start to training classifier
    print 'Training classifier...'

    start = time.time()

    param_grid = [
        {'C': [1, 10, 100, 1000],
         'kernel': ['linear']},
        {'C': [1, 10, 100, 1000],
         'gamma': [0.001, 0.0001],
         'kernel': ['rbf']}
    ]
    model = GridSearchCV(SVC(C=1, probability=True), param_grid, cv=5).fit(x, y)

    duration = (time.time() - start)
    print 'Train classifier model done! it takes %d seconds' % duration

    # saving classifier model
    classifier_model_file = join(model_dir, classifier_model_filename)
    with open(classifier_model_file, 'wb') as outfile:
        pickle.dump((model, classes), outfile)
    print 'Saved classifier model to file "%s"' % classifier_model_file


def validate_model(data_dir, model_dir, classifier_model_filename):
    x, y = get_validate_data(data_dir)

    if x is None:
        print 'No training data. Please run `python classifier.py --mode PREPARE` firstly.'

    classifier_model_file = join(model_dir, classifier_model_filename)
    if not exists(classifier_model_file):
        print 'Classifier model file not found: %s' % classifier_model_file

    print 'Validating model...'
    start = time.time()

    print 'Expected values: {}'.format(y)

    with open(classifier_model_file, 'rb') as infile:
        model, classes = pickle.load(infile)

        predictions = model.predict_proba(x)
        # predictions = model.decision_function(x)

        # each row for one sample, get the best class index by column
        best_class_indices = np.argmax(predictions, axis=1)

        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        for i in range(len(best_class_indices)):
            print '%4d %s: %.3f' % (i, classes[best_class_indices[i]], best_class_probabilities[i])

        expected_class_indices = [classes.index(cls) for cls in y]
        accuracy = np.mean(np.equal(best_class_indices, expected_class_indices))

        print('Accuracy: %.3f' % accuracy)

    duration = (time.time() - start)
    print 'Validate model done! it takes %d seconds' % duration


def classify():
    pass


def show_tsne(data_dir, show_legend=False):
    x, y = get_training_data(data_dir, False)

    if x is None:
        print 'No data'
        return

    x_pca = PCA(n_components=50).fit_transform(x, x)
    tsne = TSNE(n_components=2, init='random', random_state=0)
    x_r = tsne.fit_transform(x_pca)

    y_vals = list(np.unique(y))
    colors = cm.rainbow(np.linspace(0, 1, len(y_vals)))

    plt.figure()
    for c, i in zip(colors, y_vals):
        indices = [idx for idx, value in enumerate(y) if value == i]

        plt.scatter(x_r[indices, 0], x_r[indices, 1], c=c, label=i)

        if show_legend:
            plt.legend()

    plt.show()


def main(args):
    if args.mode == 'PREPARE':
        if args.add_classes is not None:
            add_class(args.data_dir, args.train_test_split_ratio, args.add_classes)
        elif args.remove_classes is not None:
            remove_classes(args.data_dir, args.remove_classes)
        else:
            prepare_data(args.data_dir, args.train_test_split_ratio)
    elif args.mode == 'TRAIN':
        train_model(args.data_dir, args.model_dir, args.classifier_model_filename, args.unknown)
    elif args.mode == 'VALIDATE':
        validate_model(args.data_dir, args.model_dir, args.classifier_model_filename)
    elif args.mode == 'CLASSIFY':
        classify()
    elif args.mode == 'TSNE':
        show_tsne(args.data_dir)
    elif args.mode == 'MODIFY':
        if args.remove_classes is not None:
            pass
    else:
        print 'Unknown mode: {}'.format(args.mode)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    raw_dir = utils.get_conf_prop('dataset', 'raw_dir')
    data_dir = utils.get_conf_prop('dataset', 'output_dir')
    model_dir = utils.get_conf_prop('model', 'model_dir')
    classifier_model_filename = utils.get_conf_prop('model', 'classifier_model_file')

    parser.add_argument('--mode', type=str, choices=['PREPARE','TRAIN','VALIDATE','CLASSIFY','TSNE'], default='TRAIN')
    parser.add_argument('--data_dir', type=str, default=data_dir)
    parser.add_argument('--train_test_split_ratio', type=float, default=0.9)
    parser.add_argument('--model_dir', type=str, default=model_dir)
    parser.add_argument('--classifier_model_filename', type=str, default=classifier_model_filename)
    parser.add_argument('--add_classes', nargs='+', type=str, default=None, help='only handle specified classes')
    parser.add_argument('--remove_classes', nargs='+', type=str, default=None, help='remove classes in pickle')
    parser.add_argument('--unknown', type=bool, default=False, help='Try to predict unknown people')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
