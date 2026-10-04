# Network Intrusion Detection with Machine Learning

A machine-learning pipeline for network intrusion detection: **classical ML
baselines → systematic feature selection → deep learning**, benchmarked on
the UNSW-NB15 and NSL-KDD datasets with emphasis on the **false-positive
rate** — the metric that decides whether an IDS is usable in practice.

> **About this repository.** This is a clean reference implementation
> reconstructing the approach of a 2024 University at Albany research
> internship (Research Intern / Lab Associate, Cybersecurity & Intrusion
> Detection). The original project code is no longer available, so the
> pipeline here was rewritten from the research description. The
> performance figures quoted below are the outcomes reported from that
> research — they describe the original study, not runs of this code.
> The bundled synthetic demo produces illustrative numbers only.

## Research context

Network intrusion detection systems must catch attacks without drowning
analysts in false alarms. This project explores that trade-off in three
stages:

1. **Classical baselines** — Random Forest, XGBoost and SVM benchmarked on
   the full feature sets to establish a reference point.
2. **Systematic feature selection** — mutual information, chi-squared,
   RFECV and tree-importance methods isolate the traffic signals that
   separate attacks from normal flows before model benchmarking.
3. **Deep learning** — PyTorch MLP and 1-D CNN classifiers tuned to cut
   the false-positive rate while holding detection sensitivity.

### Key results (from the original research)

| Outcome | Detail |
|---|---|
| **+27% detection accuracy** | over the baseline, via systematic feature selection and model benchmarking |
| **−18% false-positive rate** | via deep learning models, with detection sensitivity maintained |
| **Publication** | *"Intrusion Detection System,"* University at Albany Workshop, December 2024 |

## Datasets

Both benchmarks are public and **not** included in this repo (see
`data/`). Download them once, then point `--data-dir` at the folder.

### UNSW-NB15

Modern hybrid traffic (real benign + synthetic attacks), 49 features,
9 attack families plus normal traffic.

- Official source: <https://research.unsw.edu.au/projects/unsw-nb15-dataset>
- Files needed: `UNSW_NB15_training-set.csv` (175,341 records),
  `UNSW_NB15_testing-set.csv` (82,332 records)
- Citation: N. Moustafa and J. Slay, "UNSW-NB15: A Comprehensive Data Set
  for Network Intrusion Detection Systems," *Proc. MilCIS*, 2015.

### NSL-KDD

De-duplicated refinement of KDD'99, 41 features, 4 attack families
(DoS, Probe, R2L, U2R). Its test set deliberately contains attack types
absent from training — a hard generalisation test.

- Official source: <https://www.unb.ca/cic/datasets/nsl.html>
  (also mirrored on Kaggle: <https://www.kaggle.com/datasets/hassan06/nslkdd>)
- Files needed: `KDDTrain+.txt` (125,973 records),
  `KDDTest+.txt` (22,544 records)
- Citation: M. Tavallaee, E. Bagheri, W. Lu, and A. Ghorbani,
  "A Detailed Analysis of the KDD CUP 99 Data Set," *Proc. IEEE CISDA*,
  2009.

## Repository structure

```
intrusion-detection-system/
├── src/
│   ├── data_loader.py       # UNSW-NB15 / NSL-KDD loaders + synthetic generator
│   ├── preprocessing.py     # train-fit encoding, scaling, imbalance handling
│   ├── feature_selection.py # mutual info, chi2, RFECV, tree importance
│   ├── models.py            # RF / XGBoost / SVM baselines, PyTorch MLP & CNN
│   ├── train.py             # end-to-end CLI: load → train → evaluate
│   └── evaluate.py          # accuracy, precision/recall, FPR, confusion matrix
├── notebooks/
│   └── ids_walkthrough.ipynb  # narrated demo on synthetic data
├── requirements.txt
└── .gitignore
```

## Quickstart

```bash
pip install -r requirements.txt

# 1) Synthetic smoke-test — no download needed, runs in ~1 minute
python src/train.py --dataset synthetic --model all

# 2) NSL-KDD with mutual-information feature selection (top 25)
python src/train.py --dataset nsl-kdd --data-dir data --model all \
    --feature-selection mutual_info --k 25

# 3) UNSW-NB15, deep model only
python src/train.py --dataset unsw-nb15 --data-dir data --model mlp --epochs 30
```

Useful flags: `--feature-selection {mutual_info,chi2,rfecv,tree_importance,none}`,
`--imbalance {none,undersample,class_weight}`, `--model {rf,xgboost,svm,mlp,cnn,all}`.
Metrics, confusion matrices and trained models land in `outputs/`
(`metrics.json` includes per-model accuracy, FPR and gains vs. baseline).

The walkthrough notebook (`notebooks/ids_walkthrough.ipynb`) narrates the
same pipeline step by step on synthetic data — start there for the tour.

On headless servers, export `MPLBACKEND=Agg` so plots save without a display.

## Why the false-positive rate matters

An IDS that cries wolf burns out the SOC: every false alarm costs analyst
time, and chronic noise trains people to ignore real alerts. Accuracy can
hide this — a model can be 95% accurate while flagging benign traffic at a
rate no team would tolerate. That is why `evaluate.py` reports FPR
(`FP / (FP + TN)`) next to every other metric, and why the deep-learning
stage of this project optimised it directly without sacrificing recall.

## Reproducibility & limitations

- All modules compile cleanly (`python3 -m py_compile`); the full pipeline
  is exercised via the synthetic path. End-to-end runs against the real
  benchmarks need the datasets above and a machine with PyTorch.
- The synthetic generator exists only so the code runs anywhere — its
  numbers say nothing about real IDS performance.
- Research figures (+27% accuracy, −18% FPR) are the reported outcomes of
  the original 2024 study, reproduced here for context, not as claims
  about this reimplementation.
