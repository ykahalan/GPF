"""Reproduce the experiments of the paper.

    python run.py TASK [DATASET ...]      TASK: uni-cls | multi-cls | uni-reg | multi-reg | compare

Every dataset is evaluated on its archive's official train/test split: a Random Forest (100 trees, seed 42) is fit once on
the training partition and scored once on the test partition (accuracy, or MSE for regression). Rows are appended to
results_TASK.csv as each dataset finishes (datasets already in the file are skipped), then a summary is printed.
"""
import os
import sys

import numpy as np
import pandas as pd
from aeon.datasets import load_classification, load_regression
from scipy.stats import rankdata, wilcoxon
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

from gpf import gpf, gpf_aug, gpf_w, sig

DATASETS = {k: v.split() for k, v in {
    "uni-cls": """ACSF1 Adiac ArrowHead Beef BeetleFly BirdChicken BME Car CBF Chinatown ChlorineConcentration CinCECGTorso
        Coffee Computers CricketX CricketY CricketZ Crop DiatomSizeReduction DistalPhalanxOutlineAgeGroup
        DistalPhalanxOutlineCorrect DistalPhalanxTW DodgerLoopDay DodgerLoopGame DodgerLoopWeekend Earthquakes ECG200 ECG5000
        ECGFiveDays ElectricDevices EOGHorizontalSignal EOGVerticalSignal EthanolLevel FaceAll FaceFour FacesUCR FiftyWords Fish
        FordA FordB FreezerRegularTrain FreezerSmallTrain Fungi GunPoint GunPointAgeSpan GunPointMaleVersusFemale
        GunPointOldVersusYoung Ham HandOutlines Haptics Herring HouseTwenty InlineSkate InsectEPGRegularTrain
        InsectEPGSmallTrain InsectWingbeatSound ItalyPowerDemand LargeKitchenAppliances Lightning2 Lightning7 Mallat Meat
        MedicalImages MiddlePhalanxOutlineCorrect MiddlePhalanxOutlineAgeGroup MiddlePhalanxTW MixedShapesRegularTrain
        MixedShapesSmallTrain MoteStrain NonInvasiveFetalECGThorax1 NonInvasiveFetalECGThorax2 OliveOil OSULeaf
        PhalangesOutlinesCorrect Phoneme PigAirwayPressure PigArtPressure PigCVP Plane PowerCons ProximalPhalanxOutlineCorrect
        ProximalPhalanxOutlineAgeGroup ProximalPhalanxTW RefrigerationDevices Rock ScreenType SemgHandGenderCh2
        SemgHandMovementCh2 SemgHandSubjectCh2 ShapeletSim ShapesAll SmallKitchenAppliances SmoothSubspace
        SonyAIBORobotSurface1 SonyAIBORobotSurface2 StarLightCurves Strawberry SwedishLeaf Symbols SyntheticControl
        ToeSegmentation1 ToeSegmentation2 Trace TwoLeadECG TwoPatterns UMD UWaveGestureLibraryAll UWaveGestureLibraryX
        UWaveGestureLibraryY UWaveGestureLibraryZ Wafer Wine WordSynonyms Worms WormsTwoClass Yoga""",
    "multi-cls": """ArticularyWordRecognition AtrialFibrillation BasicMotions Cricket EigenWorms Epilepsy EthanolConcentration
        ERing FingerMovements HandMovementDirection Handwriting Libras LSST NATOPS PenDigits PhonemeSpectra RacketSports
        SelfRegulationSCP1 SelfRegulationSCP2 StandWalkJump UWaveGestureLibrary""",
    "uni-reg": """AcousticContaminationMadrid AluminiumConcentration BoronConcentration CalciumConcentration CopperConcentration
        Covid19Andalusia Covid3Month DhakaHourlyAirQuality FloodModeling1 FloodModeling2 FloodModeling3 GasSensorArrayAcetone
        GasSensorArrayEthanol IronConcentration LPGasMonitoringHomeActivity MagnesiumConcentration ManganeseConcentration
        MethaneMonitoringHomeActivity NaturalGasPricesSentiment PhosphorusConcentration PotassiumConcentration
        SodiumConcentration SulphurConcentration WaveDataTension WindTurbinePower ZincConcentration""",
    "multi-reg": """BarCrawl6min BeijingIntAirportPM25Quality BinanceCoinSentiment BitcoinSentiment CardanoSentiment
        ChilledWaterPredictor DailyOilGasPrices ElectricityPredictor EthereumSentiment HotwaterPredictor
        HouseholdPowerConsumption1 HouseholdPowerConsumption2 IEEEPPG MetroInterstateTrafficVolume OccupancyDetectionLight
        PrecipitationAndalusia SierraNevadaMountainsSnow SolarRadiationAndalusia SteamPredictor""",
    "compare": """ArticularyWordRecognition AtrialFibrillation BasicMotions Cricket Epilepsy EthanolConcentration ERing
        FingerMovements HandMovementDirection Handwriting Libras LSST NATOPS PhonemeSpectra RacketSports SelfRegulationSCP1
        SelfRegulationSCP2 StandWalkJump UWaveGestureLibrary""",
}.items()}


def main_sets(X, level):
    """The five feature sets of Table 1."""
    G, B = gpf(X), sig(X, level)
    h = G.shape[1] // 2
    return {"Sig": B, "GPF_S": G[:, :h], "GPF_U": G[:, h:], "GPF_SU": G, "GPF_SU+Sig": np.concatenate([B, G], 1)}


def evaluate(task, name):
    reg = task.endswith("reg")
    (Xa, ya), (Xb, yb) = [(load_regression if reg else load_classification)(name, split=s) for s in ("train", "test")]
    if task == "compare":
        from baselines import baselines
        W = gpf_w(np.concatenate([Xa, Xb]))
        sets = {**baselines(Xa, ya, Xb), "GPF_SU+Sig": (gpf_aug(Xa), gpf_aug(Xb)), "GPF-W": (W[:len(Xa)], W[len(Xa):])}
    else:
        level = 8 if task.startswith("uni") else 4  # signature degree: 8 univariate, 4 multivariate
        a, b = main_sets(Xa, level), main_sets(Xb, level)
        sets = {k: (a[k], b[k]) for k in a}
    RF, score = (RandomForestRegressor, mean_squared_error) if reg else (RandomForestClassifier, accuracy_score)
    row = {"dataset": name}
    for k, (Fa, Fb) in sets.items():
        row[k] = score(yb, RF(n_estimators=100, random_state=42, n_jobs=-1).fit(Fa, ya).predict(Fb))
        row[k + " features"] = Fa.shape[1]
    return row


def summary(task, df):
    sets = [c for c in df.columns if c != "dataset" and not c.endswith(" features")]
    if task != "compare":
        a, b = df["GPF_SU+Sig"], df["Sig"]
        better = (a < b) if task.endswith("reg") else (a > b)
        print(df[sets].mean().to_string(), f"\nGPF_SU+Sig beats Sig on {better.mean():.0%} of {len(df)} datasets, "
              f"Wilcoxon p = {wilcoxon(a, b).pvalue:.1e}")
        return
    acc, w = df[sets], df["GPF-W"]
    rank = acc.apply(lambda r: pd.Series(rankdata(-r.values), index=sets), axis=1).mean()
    rows = [{"feature set": s, "acc": acc[s].mean(), "rank": rank[s], "features": df[s + " features"].mean(),
             "W/T/L": f"{(w > acc[s]).sum()}/{(w == acc[s]).sum()}/{(w < acc[s]).sum()}",
             "p": wilcoxon(w, acc[s]).pvalue if s != "GPF-W" else np.nan} for s in sets]
    print(pd.DataFrame(rows).sort_values("acc", ascending=False).round(3).to_string(index=False))


if __name__ == "__main__":
    task, out = sys.argv[1], f"results_{sys.argv[1]}.csv"
    done = set(pd.read_csv(out).dataset) if os.path.exists(out) else set()
    for name in [d for d in sys.argv[2:] or DATASETS[task] if d not in done]:
        print(name, flush=True)
        pd.DataFrame([evaluate(task, name)]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
    summary(task, pd.read_csv(out))
