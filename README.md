# Geometric Path Features (GPF)

Code for *Geometric Path Features (GPF) via the Generalized Cross Product for Time Series Embeddings*
(Y. C. Kahalan and S. B. Belhaouari). Preprint: [doi:10.20944/preprints202603.1778.v1](https://doi.org/10.20944/preprints202603.1778.v1).

| File | Contents |
|---|---|
| `gpf.py` | The features: `gpf` (main experiments), `gpf_aug` (channel-augmented GPF<sub>SU</sub>+Sig of the baseline comparison), `gpf_w` (windowed GPF-W), `sig` (signature baseline) |
| `run.py` | The experiments, dataset lists and summaries |
| `baselines.py` | The ten established feature sets of the baseline comparison |

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python run.py uni-cls      # Table 1: 116 univariate UCR classification datasets
python run.py multi-cls    # Table 1: 21 multivariate UEA classification datasets
python run.py uni-reg      # Table 1: 26 univariate TSER regression datasets
python run.py multi-reg    # Table 1: 19 multivariate TSER regression datasets
python run.py compare      # baseline comparison: 10 feature sets, GPF_SU+Sig and GPF-W on 19 UEA datasets
python run.py multi-cls BasicMotions ERing    # only some datasets
```

Datasets are downloaded by `aeon`. Each dataset is evaluated on its archive's official train/test split: a Random Forest
(100 trees, seed 42) is fit once on the training partition and scored once on the test partition (accuracy; MSE for
regression). Results are written to `results_<task>.csv` as each dataset finishes, and a re-run skips datasets already in
the file. The summary prints the mean per feature set, the share of datasets where GPF<sub>SU</sub>+Sig beats Sig, and the
Wilcoxon signed-rank p-value. For `compare` it prints the mean accuracy, mean rank, feature count, and GPF-W's
win/tie/loss record and Wilcoxon p-value against each feature set.

## Features

For each series `x` with `d` channels, `gpf` sums a curve over time (signed) and the absolute value of the curve (unsigned):

- **k = 1:** each channel minus its chord, and each channel minus its curvature transform.
- **k = 2:** `x_i - x_j` for every pair of channels.
- **k = 3, 4:** the product of every group of `k` channels. When 3 ≤ d ≤ 4, the product of all `d` channels is appended once more, as in the paper's runs.

The signature is truncated at degree 8 for univariate series and degree 4 for multivariate series.

## Equivalence with the original notebooks

The features from `gpf`, `gpf_aug` and `gpf_w` were compared with those of the notebooks that produced the paper's
results on 27 UCR, UEA and TSER datasets, including the 19 of the baseline comparison, and are bit-identical. The Random
Forest scores can differ slightly with other versions of scikit-learn, numpy or aeon, or with other copies of the
datasets: with scikit-learn 1.9, numpy 2.3 and aeon 1.6, 8 of 10 datasets tested reproduced the paper's scores exactly,
and the other two differed in the same way with the original notebooks.

For the baselines, nine of the ten feature sets in `baselines.py` give feature matrices bit-identical to those of the
original baseline notebook (checked on BasicMotions, ERing and AtrialFibrillation). TSFresh differs on purpose: tsfresh's
worker processes return the features in the order in which they finish, so the original code could put the training and
test features in different column orders. `baselines.py` puts them in tsfresh's sequential order. With this order, the
TSFresh accuracy equals the paper's on six of the eight datasets tested (BasicMotions, AtrialFibrillation, Epilepsy,
HandMovementDirection, Libras and RacketSports). The paper's RacketSports value (0.875) was computed with this code: in
the original run the training and test columns of that dataset were misaligned, which gave 0.250.

## License

MIT (see `LICENSE`).
