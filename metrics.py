# -*- coding: utf-8 -*-

import os
import sys
import argparse
import utils
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt


from os.path import join, exists, isdir
from sklearn import metrics
from classifier import Image

default_data_dir = 'metrics'
default_data_file = 'data'


def get_data(samples):
    x = []
    y = []

    for sample_list in samples:
        for sample in sample_list:
            sample_reps = sample.load_reps()

            x.append(sample_reps)
            y.append(sample.cls)

    return x, y


def prepare_data(base_dir, model_file, batches, nrof_samples_per_batch):
    sample_file = join(base_dir, 'gen/testing_samples.pkl')

    if not exists(sample_file):
        print 'sample file not found: %s' % sample_file
        return

    if not exists(model_file):
        print 'model file not found: %s' % model_file
        return

    with open(sample_file, 'rb') as infile:
        samples = pickle.load(infile).values()

    with open(model_file, 'rb') as infile:
        (model, classes) = pickle.load(infile)

    nrof_samples = len(samples)

    if nrof_samples <= nrof_samples_per_batch:
        nrof_samples_per_batch = nrof_samples
        batches = 1

    output_dir = join(base_dir, default_data_dir)
    if not exists(output_dir):
        os.mkdir(output_dir)

    for i in range(batches):
        print '-------- batch %d --------' % i

        batch_samples = np.random.choice(samples, nrof_samples_per_batch, replace=False)
        x, y = get_data(batch_samples)

        predictions = model.predict_proba(x)
        # predictions = model.decision_function(x)

        classes_bit = np.zeros_like(predictions).astype(int)

        best_classes_indices = np.argmax(predictions, axis=1)
        classes_bit[np.arange(len(best_classes_indices)), best_classes_indices] = 1

        labels, scores = [], []

        for b in classes_bit:
            labels.extend(b.astype(str))

        for s in predictions:
            scores.extend(s.astype(str))

        data_file = join(output_dir, ('%02d.txt' % (i + 1)))
        with open(data_file, 'w') as outfile:
            outfile.writelines([','.join(labels) + '\n', ','.join(scores)])
            print 'save metrics data to: ', data_file


def load_labels_and_scores(file_path):
    # load labels and scores from file
    with open(file_path, mode='r') as f:
        lines = f.readlines()

        labels = [int(label.strip()) for label in lines[0].split(',')]
        scores = [float(score.strip()) for score in lines[1].split(',')]

        return labels, scores


def draw_plots(data_dir):
    data = []

    for f in os.listdir(data_dir):
        sub_item = join(data_dir, f)

        if isdir(sub_item):
            continue

        labels, scores = load_labels_and_scores(sub_item)

        fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
        # calculate auc score
        auc = metrics.roc_auc_score(labels, scores)

        data.append((fpr, tpr, thresholds, auc))

    # draw plot
    plt.figure(1)

    nrof_rows, nrof_cols, data_len = 1, 1, len(data)
    if data_len > 1:
        nrof_rows = (data_len / 2) + (data_len % 2)
        nrof_cols = 2

    for idx, (fpr, tpr, thresholds, auc) in enumerate(data):
        ax = plt.subplot(nrof_rows, nrof_cols, idx + 1)

        # ax.plot([0, 1], [0, 1], 'k--')
        ax.plot(fpr, tpr)

    plt.show()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, choices=['prepare', 'plot'], default='plot')
    parser.add_argument('--base_dir', type=str, default=utils.get_conf_prop('dataset', 'base_dir'))
    parser.add_argument('--model', type=str,
                        default=join(utils.get_conf_prop('model', 'model_dir'),
                                     utils.get_conf_prop('model', 'classifier_model_file')))
    parser.add_argument('--batches', type=int, default=3)
    parser.add_argument('--nrof_samples_per_batch', type=int, default=100)

    return parser.parse_args(argv)


def main(args):
    if args.mode == 'prepare':
        prepare_data(args.base_dir, args.model, args.batches, args.nrof_samples_per_batch)
    elif args.mode == 'plot':
        data_dir = join(args.base_dir, default_data_dir)
        if exists(data_dir):
           draw_plots(data_dir)
        else:
            print 'Plot data dir not found: ', data_dir
    else:
        print 'Unknown mode: ', args.mode


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))