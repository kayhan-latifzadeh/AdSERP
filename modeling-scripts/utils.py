#!/usr/bin/env python3

'''
Evaluation helpers: binary classification metrics used by the GRU models.
'''

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix


def eval_binary_classifier(y_true, y_pred):
    '''Compute the usual binary classification metrics.

    NB: `y_true = [1, 0, 0, ...]` but `y_pred = [[0.2324], [0.8731], ...]`.
    Remember we're dealing with 1 output neuron with sigmoid activation.
    '''
    prob_labels = np.array(y_pred).ravel()
    pred_labels = prob_labels > 0.5

    true_labels = y_true.astype(int)
    pred_labels = pred_labels.astype(int)

    return {
        'acc': accuracy_score(true_labels, pred_labels),
        'prf_weighted': precision_recall_fscore_support(true_labels, pred_labels, average='weighted'),
        'prf_binary': precision_recall_fscore_support(true_labels, pred_labels, average='binary'),
        'prf_micro': precision_recall_fscore_support(true_labels, pred_labels, average='micro'),
        'prf_macro': precision_recall_fscore_support(true_labels, pred_labels, average='macro'),
        'auc_weighted': roc_auc_score(y_true, y_pred, average='weighted'),
        'auc_micro': roc_auc_score(y_true, y_pred, average='micro'),
        'auc_macro': roc_auc_score(y_true, y_pred, average='macro'),
        'conf_matrix': confusion_matrix(true_labels, pred_labels)
    }
