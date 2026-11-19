"""The ten established feature sets of the comparison (aeon transformers, plus an SFA dictionary built on sktime).

baselines(X_train, y_train, X_test) -> {name: (train features, test features)}; each transformer is fit on the training
partition only. NaN/inf values are median-imputed (dense) or dropped (sparse).
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from aeon.transformations.collection.convolution_based import HydraTransformer, MiniRocket, MultiRocket, Rocket
from aeon.transformations.collection.feature_based import Catch22, TSFresh
from aeon.transformations.collection.interval_based import RandomIntervals
from aeon.transformations.collection.shapelet_based import RandomDilatedShapeletTransform, RandomShapeletTransform
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import chi2
from sklearn.impute import SimpleImputer


class SFA:
    """Dictionary baseline (WEASEL/MUSE protocol): for each channel and each of up to 10 window sizes, an SFA bag of words
    (alphabet 4, z-normalised windows, equi-depth bins); the 200 words with the highest chi2 score on the training set are kept."""

    @staticmethod
    def _words(sfa, X2):
        raw = sfa.transform(pd.DataFrame({"dim_0": [pd.Series(r) for r in X2]}))
        return [d if isinstance(d, dict) else {} for d in np.array(raw).ravel()]

    def fit(self, X, y):
        from sktime.transformations.panel.dictionary_based import SFA as SkSFA
        T = X.shape[2]
        lo = min(max(4, T // 10), T)
        wins = sorted(set(range(lo, T + 1, max(1, (T - lo) // 9))))[:10] or [T]
        wl = max(2, min(6, (wins[0] // 2) // 2 * 2))
        self.models = []
        for c in range(X.shape[1]):
            for w in wins:
                sfa = SkSFA(word_length=max(2, min(wl, (w // 2) // 2 * 2)), alphabet_size=4, window_size=w, norm=True,
                            binning_method="equi-depth", anova=False, bigrams=False, return_pandas_data_series=False, n_jobs=1)
                df = pd.DataFrame({"dim_0": [pd.Series(r) for r in X[:, c]]})
                sfa.fit(df, y)
                dv = DictVectorizer(sparse=True)
                B = dv.fit_transform(self._words(sfa, X[:, c]))
                keep = np.zeros(B.shape[1], bool)
                if B.shape[1]:
                    k = min(200, B.shape[1])
                    keep[np.argpartition(np.nan_to_num(chi2(B, y)[0], nan=-1.0), -k)[-k:]] = True
                self.models.append((c, sfa, dv, np.flatnonzero(keep)))
        return self

    def transform(self, X):
        return sp.hstack([dv.transform(self._words(sfa, X[:, c]))[:, k].astype(np.float64) for c, sfa, dv, k in self.models],
                         format="csr")


FACTORIES = {
    "Catch22": lambda T: Catch22(catch24=True, replace_nans=True),
    "ROCKET": lambda T: Rocket(n_kernels=1_000, random_state=42),
    "MiniRocket": lambda T: MiniRocket(n_kernels=1_000, random_state=42),
    "MultiRocket": lambda T: MultiRocket(n_kernels=1_000, random_state=42),
    "Hydra": lambda T: HydraTransformer(n_kernels=8, n_groups=64, random_state=42),
    "TSFresh": lambda T: TSFresh(default_fc_parameters="comprehensive", n_jobs=-1, show_warnings=False),
    "Shapelet": lambda T: RandomShapeletTransform(n_shapelet_samples=10_000, max_shapelets=None, remove_self_similar=True,
                                                  n_jobs=-1, random_state=42),
    "RDST": lambda T: RandomDilatedShapeletTransform(max_shapelets=10_000, proba_normalization=0.8, n_jobs=-1, random_state=42),
    "Random Interval": lambda T: RandomIntervals(n_intervals=max(1, int(np.sqrt(T))), random_state=42, n_jobs=-1),
    "SFA (dict.)": lambda T: SFA(),
}


def clean(F):
    if sp.issparse(F):
        F = F.astype(np.float64).tocsr()
        if not np.isfinite(F.data).all():
            F.data[~np.isfinite(F.data)] = 0.0
            F.eliminate_zeros()
        return F
    F = np.asarray(F, dtype=float)
    return F if np.isfinite(F).all() else SimpleImputer(strategy="median").fit_transform(np.where(np.isfinite(F), F, np.nan))


def baselines(Xa, ya, Xb):
    out = {}
    for name, make in FACTORIES.items():
        trf = make(Xa.shape[2]).fit(Xa, ya)
        out[name] = clean(trf.transform(Xa)), clean(trf.transform(Xb))
    return out
