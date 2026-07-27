# Modeling scripts

Predict whether a SERP advertisement attracted the user's visual attention, from
mouse cursor movements. Two model families: SVM / k-NN on Mouse2Vec features, and
a GRU on raw trajectories.

## Requirements

```
python >= 3.10
numpy, pandas, scikit-learn
torch, einops           # Mouse2Vec encoder
tensorflow >= 2.10      # GRU models
```

The Mouse2Vec encoder under `mouse2vec/` is bundled from
https://git.cai.simtech.uni-stuttgart.de/public-projects/Mouse2Vec
(see `mouse2vec/NOTICE.md`); check there for the latest version.

## Dataset

Download the dataset from https://zenodo.org/records/15236546 and unzip these
into a folder named `dataset/`:

```
dataset/
  ad-boundary-data/<trial>.json     ad bounding boxes
  fixation-data/<trial>.csv         eye fixations
  mouse-movement-data/<trial>.csv   mouse cursor logs
```

Everything below is run from this directory.

---

## SVM / k-NN models

**1. Prepare the inputs** — writes `ad_locs`, `y` and `m2v_features` to `data/`:

```bash
python3 prepare_input_for_svm_knn_models.py --dataset dataset --out data
```

**2. Train and evaluate** — loops over all 16 configurations (2 classifiers ×
2 ad types × 4 durations) and prints a report for each:

```bash
python3 train_knn_svm_models.py
```

---

## GRU models

**1. Prepare the inputs** — writes the per-trial time series to `data/for_rnn/`:

```bash
python3 prepare_input_for_gru_models.py --dataset dataset --out data/for_rnn
```

**2. Train and evaluate** — one ad type and duration per run:

```bash
python3 train_gru_models.py \
  --attended_dir   data/for_rnn/native_ad/5s/1 \
  --unattended_dir data/for_rnn/native_ad/5s/0 \
  --input_size 10
```

`<ad_type>` is `native_ad` or `dd_both`; `<duration>` is `5s`, `10s`, `15s` or
`20s`. Directory `1` holds trials where the ad attracted attention, `0` the rest.

---

## Files

```
prepare_input_for_svm_knn_models.py   dataset -> ad_locs, y, m2v_features
prepare_input_for_gru_models.py       dataset -> for_rnn/ time series
train_knn_svm_models.py               SVM and k-NN
train_gru_models.py                   GRU
utils.py                              evaluation helpers
mouse2vec/                            pretrained Mouse2Vec encoder
```
