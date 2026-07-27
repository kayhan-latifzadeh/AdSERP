#!/usr/bin/env python3
# coding: utf-8

'''
Train a GRU classifier to predict, from mouse cursor movements, whether
a SERP advertisement attracted the user's visual attention.

At every timestep the model sees the mouse position (x, y) concatenated with
the bounding boxes of the target ads, each encoded as (x, y, w, h) — hence
--input_size 10 for the two boxes stored in the CSV files. Sequences are
truncated to the first 5, 10, 15 or 20 seconds of the trial, and zero-padded.

The architecture and its hyperparameters: a GRU layer with 150 units
and tanh activation, dropout 0.25, Adam with lr = 1e-3, batches of 32,
at most 100 epochs, and early stopping with a patience of 10 monitoring
validation accuracy.

Usage:
  python3 train_gru_models.py \
    --attended_dir data/for_rnn/<ad_type>/<duration>/1 \
    --unattended_dir data/for_rnn/<ad_type>/<duration>/0 \
    --input_size 10
'''

# Load std libs.
import sys
import os
import argparse
import random
from time import time

# Display only TF errors, if any. NB: must be set before TensorFlow is imported.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure CLI parser early.
parser = argparse.ArgumentParser(description='Train a GRU classifier to predict \
  visual attention on SERP ads from mouse movements',
  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Define CLI options.
parser.add_argument('--attended_dir', help='path to the directory of trials where the ad DID attract attention (label 1)')
parser.add_argument('--unattended_dir', help='path to the directory of trials where the ad did NOT attract attention (label 0)')
parser.add_argument('--attended_files', nargs='+', help='paths to individual attended-ad trials, instead of --attended_dir')
parser.add_argument('--unattended_files', nargs='+', help='paths to individual unattended-ad trials, instead of --unattended_dir')
parser.add_argument('--sort_files', action='store_true', help='sort data, to promote test participants not to be seen during model training')
parser.add_argument('--epochs', default=100, type=int, help='maximum number of training epochs')
parser.add_argument('--patience', default=10, type=int, help='number of consecutive epochs without improvement to stop training')
parser.add_argument('--batch_size', default=32, type=int, help='training batch size')
parser.add_argument('--training_ratio', default=0.7, type=float, help='training partition size')
parser.add_argument('--validation_ratio', default=0.1, type=float, help='validation partition size, relative to training partition')
parser.add_argument('--activation', choices=['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu', 'elu'], default='tanh', help='activation function in the GRU layer. NB: the output layer has ALWAYS sigmoid activation')
parser.add_argument('--input_size', default=10, type=int, help='number of features per timestep: mouse (x, y) plus the ad bounding boxes')
parser.add_argument('--out_dir', help='path to write the trained model')
parser.add_argument('--verbose', default=1, type=int, help='display more information to stdout')

args = parser.parse_args()
if args.verbose:
    print(args, file=sys.stderr)

# Load 3rd party libs.
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from utils import eval_binary_classifier

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print('We got a GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    print('Sorry, no GPU for you...')

# Use current timestamp to keep individual runs of each experiment.
NOW = int(time())

# Makes results reproducible.
RANDOM_SEED = 123456
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Model hyperparameters
RNN_UNITS = 150
DROPOUT_RATE = 0.25
LEARNING_RATE = 1e-3

# Sanity checks.
if (args.attended_dir and not os.path.isdir(args.attended_dir)) or (args.unattended_dir and not os.path.isdir(args.unattended_dir)):
    print('Input data not found! Please revise your attended and unattended directories.')
    parser.print_help()
    exit()


def get_label(filepath):
    return os.path.basename(os.path.dirname(filepath))


def read_sequence(filepath):
    '''Read one trial as the mouse position (x, y) per timestep,
    concatenated with the (x, y, w, h) bounding boxes of the target ads.'''
    df = pd.read_csv(filepath)
    df = df.drop_duplicates(subset='t')
    t = df['t'].tolist()
    xs = df['x'].tolist()
    ys = df['y'].tolist()
    # The bounding boxes are constant along the trial, so we read them once.
    aoi_info_columns = ['aoi_info0', 'aoi_info1', 'aoi_info2', 'aoi_info3', 'aoi_info4', 'aoi_info5', 'aoi_info6', 'aoi_info7']
    aoi_infos = df[aoi_info_columns].values.tolist()[0] # 8 numbers

    moves = []
    for i in range(1, len(xs)):
        # Skip non-monotonic timestamps.
        if t[i] - t[i-1] <= 0:
            continue

        entry = [int(xs[i]), int(ys[i])]
        for j in range(len(aoi_infos)):
            entry.append(aoi_infos[j])
        moves.append(entry)
    return np.array(moves)


def list_files(directory):
    files = []
    for r, d, f in os.walk(directory):
        for file in f:
            if file.endswith('.csv'):
                files.append(os.path.join(r, file))
    assert len(files) > 0
    return files


def sequence_loader(files):
    batch_input, batch_output = [], []
    for filepath in files:
        label = LABELS[get_label(filepath)]
        moves = read_sequence(filepath)
        # Discard trials with no usable timestep, otherwise they would be fed
        # to the model as all-zero samples. NB: this drops ~16% of the 5 s files.
        if len(moves) == 0:
            continue

        batch_input.append(moves)
        batch_output.append(label)

    # NB: two padding passes, on purpose. The first one post-pads every sequence
    # to the longest one in the batch; the second one pre-pads up to MAX_LENGTH.
    X, y = np.array(pad_sequences(batch_input, padding='post')), np.array(batch_output)
    X = pad_sequences(X, maxlen=MAX_LENGTH, dtype='float32')
    # Ensure we pass in the right data type. Is this a TF2 issue?
    X = tf.cast(X, tf.float32).numpy()
    y = tf.cast(y, tf.int32).numpy()

    assert len(X) > 0
    assert len(y) > 0
    return X, y


def get_max_lines_in_csv(directory):
    max_lines = 0
    for file_name in os.listdir(directory):
        if file_name.endswith('.csv'):
            df = pd.read_csv(os.path.join(directory, file_name))
            df = df.drop_duplicates(subset='t')
            max_lines = max(max_lines, len(df))
    return max_lines


def build_gru_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(None, args.input_size)),
        tf.keras.layers.GRU(units=RNN_UNITS, activation=args.activation, return_sequences=True),
        tf.keras.layers.Dropout(DROPOUT_RATE),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    return model


def train_model(X_train, y_train, X_test, y_test):
    model = build_gru_model()

    model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_ratio,
        verbose=args.verbose,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=args.patience, monitor='val_accuracy')],
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save(os.path.join(OUTPUT_DIR, f'{NOW}_gru.h5'))
    print(f'Best model saved to {OUTPUT_DIR}')

    # Evaluate best model
    y_pred = model.predict(X_test).ravel()
    res = eval_binary_classifier(y_test, y_pred)
    precision, recall, f1, _ = res['prf_binary']
    adj_precision, adj_recall, adj_f1, _ = res['prf_weighted']
    accuracy, auc = res['acc'], res['auc_macro']

    print(f'''
    | Mode         | Precision | Recall | F-measure | Accuracy | AUC ROC |
    |---           |---        |---     |---        |---       |---      |
    | non-weighted | {precision:.4f}    | {recall:.4f} | {f1:.4f}    | {accuracy:.4f}   | {auc:.4f}  |
    | weighted     | {adj_precision:.4f}    | {adj_recall:.4f} | {adj_f1:.4f}    | {accuracy:.4f}   | {auc:.4f}  |
    ''')


# General configuration.
OUTPUT_DIR = args.out_dir if args.out_dir else 'saved_models/model-{}'.format(NOW)

# Load filenames.
attended_files = list_files(args.attended_dir) if args.attended_dir else args.attended_files
unattended_files = list_files(args.unattended_dir) if args.unattended_dir else args.unattended_files

# Map directory names to class names.
LABELS = {
  get_label(attended_files[0]) : 1,
  get_label(unattended_files[0]) : 0,
}
print('Classes:', LABELS, file=sys.stderr)

# All sequences are padded to the longest trial found in the dataset.
MAX_LENGTH = max(get_max_lines_in_csv(args.attended_dir), get_max_lines_in_csv(args.unattended_dir))

if args.sort_files:
    # Interleave attended/unattended data while ensuring file order, so that
    # in the later splits some participants are not seen during training.
    attended_files = sorted(attended_files)
    unattended_files = sorted(unattended_files)
    train_files = np.array([f for files in zip(attended_files, unattended_files) for f in files])
else:
    # Collect data at random.
    train_files = np.array(attended_files + unattended_files)
    np.random.shuffle(train_files)

# Reserve some part of the training data for testing.
# Later we split the training data again for validation.
train_files, test_files = train_test_split(train_files, train_size=args.training_ratio,
  random_state=RANDOM_SEED, shuffle=not(args.sort_files))

X_train, y_train = sequence_loader(train_files)
X_test, y_test = sequence_loader(test_files)
print(X_train.shape)
print(X_test.shape)
train_model(X_train, y_train, X_test, y_test)
