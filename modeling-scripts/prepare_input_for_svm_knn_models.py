#!/usr/bin/env python3
# coding: utf-8

'''
Prepare the input files for the SVM and k-NN classifiers, directly from the
dataset:

  <out>/ad_locs_{ad_type}_{duration}s.npy   bounding boxes of the target ads
  <out>/y_{ad_type}_{duration}s.npy         attention value of the target ad
  <out>/m2v_features_{duration}s.npy        Mouse2Vec embeddings

Reads three dataset directories:

  <dataset>/ad-boundary-data/<trial>.json    ad bounding boxes
  <dataset>/fixation-data/<trial>.csv        eye fixations (FPOGX, FPOGY, FPOGD)
  <dataset>/mouse-movement-data/<trial>.csv  mouse cursor log

Usage:
  python3 prepare_input_for_svm_knn_models.py --dataset dataset --out data
'''

import os
import sys
import glob
import json
import argparse
import warnings

import numpy as np
import pandas as pd

AD_TYPES = ['native_ad', 'dd_both']
DURATIONS = [5, 10, 15, 20]

SEGMENT_SECONDS = 5
SAMPLING_RATE = 20                          # Hz
WINDOW_SAMPLES = SEGMENT_SECONDS * SAMPLING_RATE
MIN_FIXATION_DURATION = 100                 # ms
NATIVE_AD_MERGE_THRESHOLD = 50              # px
FIXATION_TOLERANCE = 15                     # px, added around each ad box


# --- ad bounding boxes ----------------------------------------------------

def merge_ads(ad_boundaries, vertical_threshold=NATIVE_AD_MERGE_THRESHOLD):
    '''Merge ads separated by at most `vertical_threshold` pixels vertically.'''
    ad_boundaries = sorted(ad_boundaries, key=lambda ad: ad['location']['y'])

    def merge_group(group):
        min_x = min(a['location']['x'] for a in group)
        min_y = min(a['location']['y'] for a in group)
        max_x = max(a['location']['x'] + a['size']['width'] for a in group)
        max_y = max(a['location']['y'] + a['size']['height'] for a in group)
        return {'location': {'x': min_x, 'y': min_y},
                'size': {'width': max_x - min_x, 'height': max_y - min_y}}

    merged, group = [], []
    for ad in ad_boundaries:
        if not group:
            group = [ad]
            continue
        previous = group[-1]
        if ad['location']['y'] - (previous['location']['y'] + previous['size']['height']) <= vertical_threshold:
            group.append(ad)
        else:
            merged.append(merge_group(group))
            group = [ad]
    if group:
        merged.append(merge_group(group))
    return merged


def as_box(element):
    return [element['location']['x'], element['location']['y'],
            element['size']['width'], element['size']['height']]


def get_ad_boundary(ad_data, ad_type):
    '''The target ad boxes as [x, y, w, h, x, y, w, h], or None if the trial has none.'''
    if ad_type == 'dd_both':
        if ad_data['dd_right']:
            return as_box(ad_data['dd_right'][0]) + [0, 0, 0, 0]
        if ad_data['dd_top']:
            return as_box(ad_data['dd_top'][0]) + [0, 0, 0, 0]
        return None

    if ad_type == 'native_ad' and ad_data['native_ad']:
        wrappers = merge_ads(ad_data['native_ad'])
        second = as_box(wrappers[1]) if len(wrappers) > 1 else [0, 0, 0, 0]
        return as_box(wrappers[0]) + second

    return None


def attention_regions(ad_data, ad_type):
    '''The ad boxes a fixation is tested against.'''
    if ad_type == 'dd_both':
        return [as_box(ad) for ad in ad_data['dd_top'] + ad_data['dd_right']]
    return [as_box(ad) for ad in merge_ads(ad_data['native_ad'])]


# --- attention value ------------------------------------------------------

def is_on_ad(x, y, regions, tolerance=FIXATION_TOLERANCE):
    '''Whether a fixation falls within `tolerance` pixels of any region.'''
    if not (x > 0 and y > 0):
        return False
    for bx, by, bw, bh in regions:
        if bx - tolerance < x < bx + bw + tolerance and by - tolerance < y < by + bh + tolerance:
            return True
    return False


def attention_value(fixations, origin, duration, regions):
    '''Share of fixation time spent on the target ad within the first `duration` seconds.'''
    elapsed = fixations['timestamp'].values - origin
    fpogd = fixations['FPOGD'].values
    keep = (elapsed <= duration * 1000) & (fpogd >= MIN_FIXATION_DURATION)
    if keep.sum() == 0 or fpogd[keep].sum() == 0:
        return None

    xs, ys = fixations['FPOGX'].values, fixations['FPOGY'].values
    on_ad = np.array([is_on_ad(xs[i], ys[i], regions) for i in range(len(fixations))])
    return round(fpogd[keep & on_ad].sum() / fpogd[keep].sum(), 2)


# --- Mouse2Vec embeddings -------------------------------------------------

def load_encoder(model_dir):
    import torch
    import torch.nn as nn
    sys.path.insert(0, os.path.abspath(model_dir))
    model = torch.load(os.path.join(model_dir, 'pretrained_model.pkl'),
                       map_location='cpu', weights_only=False)

    def qualify(name):
        parts = name.split('.')
        for i, part in enumerate(parts):
            if part.isnumeric():
                parts[i] = "_modules['" + part + "']"
        return '.'.join(parts)

    for name, module in model.named_modules():
        if isinstance(module, nn.GELU):
            exec('model.' + qualify(name) + '=nn.GELU()')

    model.eval()
    return model


def preprocess_mouse(mouse_file):
    '''Resample a mouse log to a regular 20 Hz signal with a relative clock.'''
    df = pd.read_csv(mouse_file).drop(columns=['xpath']).drop_duplicates(subset='timestamp')
    df['event'] = df['event'].apply(lambda e: 1 if e == 'click' else 0)
    df = df.rename(columns={'xpos': 'x', 'ypos': 'y', 'timestamp': 't'})[['x', 'y', 't', 'event']]
    df['datetime'] = pd.to_datetime(df['t'], unit='ms')
    df = df.set_index('datetime')
    df = df.resample(f'{1000 // SAMPLING_RATE}ms').ffill().reset_index()
    df['t'] = df['datetime'].astype('int64') // 10 ** 6
    df = df.dropna()
    df['t'] = df['t'] - df['t'].iloc[0]
    return df.drop(columns=['datetime'])


def to_frequency_domain(window):
    import torch
    data = window[None]
    mags = phases = None
    for channel in range(data.shape[-1]):
        freq = np.fft.fft(data[:, :, channel])[:, :data.shape[1] // 2]
        magnitude, phase = np.abs(freq), np.angle(freq)
        magnitude = (magnitude - np.min(magnitude)) / ((np.max(magnitude) - np.min(magnitude)) + 1e-10)
        phase = (phase - np.min(phase)) / ((np.max(phase) - np.min(phase)) + 1e-10)
        if mags is None:
            mags, phases = magnitude, phase
        else:
            mags = np.stack([mags, magnitude], axis=-1)
            phases = np.stack([phases, phase], axis=-1)
    data = np.concatenate([data, mags, phases], axis=1)
    return torch.tensor(data.reshape(1, 2, WINDOW_SAMPLES * 2), dtype=torch.float32)


def embed_trial(model, opt, resampled, max_duration):
    '''Encode each consecutive 5 s window into a 128-d embedding, in order.'''
    import torch
    from sklearn.preprocessing import MinMaxScaler
    embeddings = []
    for i in range(max_duration // SEGMENT_SECONDS):
        window = resampled[(resampled['t'] >= i * SEGMENT_SECONDS * 1000) &
                           (resampled['t'] < (i + 1) * SEGMENT_SECONDS * 1000)]
        if len(window) != WINDOW_SAMPLES:
            embeddings.append((i, None))
            continue
        scaled = MinMaxScaler().fit_transform(window[['x', 'y']].values)
        with torch.no_grad():
            embeddings.append((i, model.encoder(to_frequency_domain(scaled), opt).numpy()[0]))
    return embeddings


# --- main -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default='dataset', help='directory of the unzipped dataset')
    parser.add_argument('--model_dir', default='mouse2vec', help='directory with pretrained_model.pkl')
    parser.add_argument('--out', default='data', help='directory to write the .npy files')
    args = parser.parse_args()

    warnings.filterwarnings('ignore')
    os.makedirs(args.out, exist_ok=True)

    import argparse as _argparse
    model = load_encoder(args.model_dir)
    opt = _argparse.Namespace(patch_len=5, seq_len=WINDOW_SAMPLES, device='cpu')

    trials = sorted(os.path.basename(f)[:-5]
                    for f in glob.glob(os.path.join(args.dataset, 'ad-boundary-data', '*.json')))
    print(f'{len(trials)} trials in the dataset')

    ad_locs = {(a, d): {} for a in AD_TYPES for d in DURATIONS}
    attention = {(a, d): {} for a in AD_TYPES for d in DURATIONS}
    features = {d: {} for d in DURATIONS}

    for n, trial in enumerate(trials, 1):
        if n % 250 == 0:
            print(f'  {n}/{len(trials)}')
        fixation_file = os.path.join(args.dataset, 'fixation-data', f'{trial}.csv')
        mouse_file = os.path.join(args.dataset, 'mouse-movement-data', f'{trial}.csv')
        ad_file = os.path.join(args.dataset, 'ad-boundary-data', f'{trial}.json')
        if not (os.path.exists(fixation_file) and os.path.exists(mouse_file)):
            continue

        with open(ad_file) as f:
            ad_data = json.load(f)
        fixations = pd.read_csv(fixation_file)
        mouse = pd.read_csv(mouse_file)
        if len(fixations) == 0 or len(mouse) == 0:
            continue
        origin = mouse['timestamp'].iloc[0]

        resampled = preprocess_mouse(mouse_file)
        embedded = embed_trial(model, opt, resampled, max(DURATIONS))

        for duration in DURATIONS:
            usable = [v for i, v in embedded if i < duration // SEGMENT_SECONDS and v is not None]
            if usable:
                features[duration][trial] = list(np.concatenate(usable).astype(np.float32))

            for ad_type in AD_TYPES:
                boxes = get_ad_boundary(ad_data, ad_type)
                if boxes is None:
                    continue
                value = attention_value(fixations, origin, duration, attention_regions(ad_data, ad_type))
                if value is None:
                    continue
                ad_locs[(ad_type, duration)][trial] = boxes
                attention[(ad_type, duration)][trial] = value

    for duration in DURATIONS:
        path = os.path.join(args.out, f'm2v_features_{duration}s.npy')
        np.save(path, features[duration])
        print(f'wrote {path} ({len(features[duration])} trials)')
        for ad_type in AD_TYPES:
            for name, data in (('ad_locs', ad_locs[(ad_type, duration)]),
                               ('y', attention[(ad_type, duration)])):
                path = os.path.join(args.out, f'{name}_{ad_type}_{duration}s.npy')
                np.save(path, data)
                print(f'wrote {path} ({len(data)} trials)')


if __name__ == '__main__':
    main()
