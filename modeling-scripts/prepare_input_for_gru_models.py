#!/usr/bin/env python3
# coding: utf-8

'''
Prepare the per-trial time series for the GRU models, directly from the dataset:

  <out>/for_rnn/{ad_type}/{duration}s/{1,0}/*.csv

One file per trial, holding the mouse trajectory over the first `duration`
seconds, with the target ad boxes repeated on every row:

  t, x, y, aoi_info0 ... aoi_info7

A trial goes to directory `1` when its attention value exceeds the threshold
tau (the median attention of that configuration, rounded to one decimal), and
to `0` otherwise.

Reads:
  <dataset>/ad-boundary-data/<trial>.json    ad bounding boxes
  <dataset>/fixation-data/<trial>.csv        eye fixations
  <dataset>/mouse-movement-data/<trial>.csv  mouse cursor log

Usage:
  python3 prepare_input_for_gru_models.py --dataset dataset --out data/for_rnn
'''

import os
import json
import glob
import shutil
import argparse
import warnings

import numpy as np
import pandas as pd

from prepare_input_for_svm_knn_models import (
    AD_TYPES, DURATIONS, get_ad_boundary, attention_regions, attention_value,
)


def build_trial(mouse, ad_boxes, duration):
    '''The raw trajectory over the first `duration` seconds, with the ad boxes attached.'''
    start = mouse['timestamp'].min()
    window = mouse[mouse['timestamp'] <= start + duration * 1000]
    if len(window) == 0:
        return None
    trial = pd.DataFrame()
    trial['t'] = window['timestamp']
    trial['x'] = window['xpos']
    trial['y'] = window['ypos']
    for i, value in enumerate(ad_boxes):
        trial[f'aoi_info{i}'] = value
    return trial


def main():
    parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default='dataset', help='directory of the unzipped dataset')
    parser.add_argument('--out', default='data/for_rnn', help='directory to write the trials to')
    args = parser.parse_args()

    warnings.filterwarnings('ignore')
    trials = sorted(os.path.basename(f)[:-5]
                    for f in glob.glob(os.path.join(args.dataset, 'ad-boundary-data', '*.json')))
    print(f'{len(trials)} trials in the dataset')

    for ad_type in AD_TYPES:
        for duration in DURATIONS:
            prepared = []                        # (attention_value, trial_dataframe)
            for trial in trials:
                fixation_file = os.path.join(args.dataset, 'fixation-data', f'{trial}.csv')
                mouse_file = os.path.join(args.dataset, 'mouse-movement-data', f'{trial}.csv')
                ad_file = os.path.join(args.dataset, 'ad-boundary-data', f'{trial}.json')
                if not (os.path.exists(fixation_file) and os.path.exists(mouse_file)):
                    continue

                with open(ad_file) as f:
                    ad_data = json.load(f)
                boxes = get_ad_boundary(ad_data, ad_type)
                if boxes is None:
                    continue
                fixations = pd.read_csv(fixation_file)
                mouse = pd.read_csv(mouse_file)
                if len(fixations) == 0 or len(mouse) == 0:
                    continue

                value = attention_value(fixations, mouse['timestamp'].iloc[0], duration,
                                        attention_regions(ad_data, ad_type))
                if value is None:
                    continue
                frame = build_trial(mouse, boxes, duration)
                if frame is None:
                    continue
                prepared.append((value, frame))

            # Binarize the attention values at their median, rounded to one decimal.
            tau = np.round(np.median([v for v, _ in prepared]), 1)

            counters = {1: 1, 0: 1}
            for label in (1, 0):
                directory = os.path.join(args.out, ad_type, f'{duration}s', str(label))
                if os.path.isdir(directory):
                    shutil.rmtree(directory)
                os.makedirs(directory)
            for value, frame in prepared:
                label = 1 if value > tau else 0
                frame.to_csv(os.path.join(args.out, ad_type, f'{duration}s', str(label),
                                          f'{counters[label]}.csv'), index=False)
                counters[label] += 1
            print(f'{ad_type} {duration}s: {counters[1] - 1} attended, {counters[0] - 1} unattended')


if __name__ == '__main__':
    main()
