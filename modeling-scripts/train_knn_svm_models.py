#!/usr/bin/env python3
# coding: utf-8

'''
Train SVM and k-NN classifiers to predict, from mouse cursor movements alone,
whether a SERP advertisement attracted the user's visual attention.

Each trial is represented by a 128-dim Mouse2Vec embedding concatenated with the
bounding box of the target ad, encoded as (x, y, w, h). Each trial is labelled
1 if the ad's attention value exceeds the median attention of that configuration,
0 otherwise; the labels come from eye fixations, the features from mouse data.

Both classifiers use scikit-learn's default parameters. This script loops over
all 16 configurations (2 classifiers x 2 ad types x 4 trajectory durations) and
prints a report for each, so it takes no arguments:

  python3 train_knn_svm_models.py
'''

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

DURATIONS = [5, 10, 15, 20]          # trajectory duration, in seconds
CLASSIFIERS = ['svm', 'knn']
AD_TYPES = ['native_ad', 'dd_both']
FEATURE_TYPE = 'm2v'
RANDOM_SEED = 42
CLASS_NAMES = ['unattended', 'attended']


def pad_lists_in_dict(data):
    '''Zero-pad every feature vector up to the longest one.'''
    max_len = max(len(lst) for lst in data.values())
    for key in data:
        data[key].extend([0] * (max_len - len(data[key])))
    return data


def load_m2v_data(ad_type, duration):
    '''Return the Mouse2Vec embeddings, the ad bounding boxes,
    and the raw (continuous) attention values, aligned per trial.'''
    m2v_features_dict = np.load(f'data/m2v_features_{duration}s.npy', allow_pickle=True).item()
    ad_locs_dict = np.load(f'data/ad_locs_{ad_type}_{duration}s.npy', allow_pickle=True).item()
    attention_dict = np.load(f'data/y_{ad_type}_{duration}s.npy', allow_pickle=True).item()
    m2v_features_dict = pad_lists_in_dict(m2v_features_dict)

    m2v_features, ad_locs, attention = [], [], []
    for trial_id, features in m2v_features_dict.items():
        if trial_id not in attention_dict:
            continue
        m2v_features.append(features)
        ad_locs.append(ad_locs_dict[trial_id])
        attention.append(attention_dict[trial_id])

    return np.array(m2v_features), np.array(ad_locs), np.array(attention)


def train_and_evaluate(clf_name, X_train, X_test, y_train, y_test):
    if clf_name == 'knn':
        model = KNeighborsClassifier()
    elif clf_name == 'svm':
        model = SVC(probability=True, random_state=RANDOM_SEED)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        (clf_name, model)
    ])
    pipeline.fit(X_train, y_train)

    if clf_name == 'svm':
        y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_prob > 0.5).astype(int)
        auc_roc = roc_auc_score(y_test, y_pred_prob)
    else:
        y_pred = pipeline.predict(X_test)
        auc_roc = roc_auc_score(y_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)

    print('Test Accuracy:', accuracy)
    print('AUC-ROC:', auc_roc)
    print('Classification Report:\n', report)


for clf_name in CLASSIFIERS:
    for ad_type in AD_TYPES:
        for duration in DURATIONS:

            m2v_features, ad_locs, attention = load_m2v_data(ad_type, duration)

            # Binarize attention at the median
            threshold = np.median(attention)
            labels = attention > threshold

            # Drop trials whose embeddings could not be computed.
            nan_indices = np.isnan(m2v_features).any(axis=1)
            m2v_features = m2v_features[~nan_indices]
            ad_locs = ad_locs[~nan_indices]
            labels = labels[~nan_indices]

            if ad_type == 'dd_both':
                ad_locs = ad_locs[:, 0:4]

            X = np.hstack([m2v_features, ad_locs])
            print(X.shape)

            X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3,
              random_state=RANDOM_SEED, stratify=labels)
            print(y_train.shape[0], y_test.shape[0], len(labels))

            train_and_evaluate(clf_name, X_train, X_test, y_train, y_test)

            print(f'CLF = {clf_name}, THRESHOLD = {threshold} MAX_DUR = {duration}, '
                  f'TARGET_AD_TYPE = {ad_type}, FEATURE_TYPE = {FEATURE_TYPE}')
