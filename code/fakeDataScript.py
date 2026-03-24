from __future__ import annotations


import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import clone
from sklearn.inspection import permutation_importance

# ---- Survival libraries (recommended) ----
# pip install scikit-survival
try:
    from sksurv.util import Surv
    from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
except ImportError as e:
    raise ImportError(
        "This skeleton expects scikit-survival. Install with: pip install scikit-survival"
    ) from e

@dataclass(frozen=True) 
class Config:
    data_path: str = "data/fake_data/mgus2_june_paper_placeholder_cleaned.csv"  # TODO
    id_col: str = "participant_id"

    ace_col: str = "baseline_biomarker_1"

    # Endpoint column names
    time_npe: str = "time_to_primary_event_months"
    event_npe: str = "primary_event_indicator"
    time_hosp: str = "time_to_competing_event_or_censor_months"
    event_hosp: str = "competing_event_indicator"

    # Horizons for time-dependent AUC (in same units as time columns)
    # In Config — change horizons to months
    horizons: Tuple[float, ...] = (12.0, 24.0, 36.0, 60.0)  # 1, 2, 3, 5 years in months

    # CV
    n_splits: int = 5
    n_repeats: int = 10
    random_state: int = 42

    # Coxnet regularisation path
    coxnet_l1_ratio: float = 0.01
    coxnet_alphas: Optional[np.ndarray] = 0.1 * np.logspace(-4, 4, 50)  # if None, will use default path from data

    # RSF / GBS (optional sensitivity)
    rsf_n_estimators: int = 500
    rsf_min_samples_leaf: int = 15

    gbs_n_estimators: int = 500

    # Bootstrap for CIs on metrics from out-of-fold predictions
    n_bootstrap: int = 2000
    bootstrap_seed: int = 123

    # Risk binning for calibration
    n_calib_bins: int = 5  # quintiles


CFG = Config()

class NumpyEncoder(json.JSONEncoder):
    """Serialise numpy scalars and arrays transparently."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)
# cfg = CFG

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def infer_feature_types(
    df: pd.DataFrame,
    candidate_features: List[str]
) -> Tuple[List[str], List[str]]:
    """Split features into numeric vs categorical by dtype."""
    numeric, categorical = [], []
    for c in candidate_features:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)
        else:
            categorical.append(c)
    return numeric, categorical

def make_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """Impute + scale numeric; impute + one-hot categorical."""
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")), # replace with SMOTE or remove completely
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")), # replace with SMOTE or remove completely
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )
    return pre

def make_y_surv(df: pd.DataFrame, time_col: str, event_col: str):
    """Create scikit-survival structured array."""
    # event must be boolean
    event = df[event_col].astype(bool).values
    time = df[time_col].astype(float).values
    return Surv.from_arrays(event=event, time=time)

def get_primary_cohort(df: pd.DataFrame, ace_col: str) -> pd.DataFrame:
    """Primary cohort is ACE-available subset."""
    return df.loc[df[ace_col].notna()].copy()


def build_models(cfg: Config) -> Dict[str, object]:
    """
    Return estimators (not pipelines): preprocessing is applied separately.
    For Coxnet, we will pick alpha via internal CV *or* by a simple heuristic
    (here: use default alpha path then select best by CV in training folds).
    """
    models = {}

    # Cox elastic-net
    models["coxnet"] = CoxnetSurvivalAnalysis(
        l1_ratio=cfg.coxnet_l1_ratio,
        alphas=cfg.coxnet_alphas
    )

    # Optional nonlinear models (for supplementary robustness)
    models["rsf"] = RandomSurvivalForest(
        n_estimators=cfg.rsf_n_estimators,
        min_samples_leaf=cfg.rsf_min_samples_leaf,
        n_jobs=-1,
        random_state=cfg.random_state
    )

    models["gbs"] = GradientBoostingSurvivalAnalysis(
        n_estimators=cfg.gbs_n_estimators,
        random_state=cfg.random_state
    )

    return models

def select_best_coxnet_alpha(
    X_train, y_train, base_estimator: CoxnetSurvivalAnalysis, inner_splits: int = 5, seed: int = 0
) -> CoxnetSurvivalAnalysis:
    """
    Minimal inner-CV selection for Coxnet alpha using c-index on inner folds.
    This is intentionally lightweight; you can replace with a more rigorous selection.
    """
    rkf = RepeatedKFold(n_splits=inner_splits, n_repeats=1, random_state=seed)
    # Fit once to get alpha path if not supplied
    est0 = clone(base_estimator).fit(X_train, y_train)
    alphas = est0.alphas_

    best_alpha = None
    best_score = -np.inf

    for a in alphas:
        scores = []
        for tr, va in rkf.split(X_train):
            est = clone(base_estimator)
            est.set_params(alphas=np.array([a]))
            est.fit(X_train[tr], y_train[tr])
            # Coxnet returns risk scores via predict (higher = higher risk)
            risk = est.predict(X_train[va])
            cidx = concordance_index_censored(
                y_train[va]["event"], y_train[va]["time"], risk
            )[0]
            scores.append(cidx)
        m = float(np.mean(scores))
        if m > best_score:
            best_score = m
            best_alpha = float(a)

    final_est = clone(base_estimator)
    final_est.set_params(alphas=np.array([best_alpha]))
    return final_est

def eval_fold(
    preprocessor, estimator,
    X_train_df, y_train,
    X_test_df, y_test,
    horizons, choose_alpha_for_coxnet=True, inner_seed=0,
):
    from sksurv.nonparametric import kaplan_meier_estimator

    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    est = clone(estimator)
    if isinstance(est, CoxnetSurvivalAnalysis) and choose_alpha_for_coxnet:
        est = select_best_coxnet_alpha(
            X_train, y_train, base_estimator=est, inner_splits=5, seed=inner_seed
        )

    est.fit(X_train, y_train)
    risk_test = est.predict(X_test)

    cidx = concordance_index_censored(y_test["event"], y_test["time"], risk_test)[0]

    # --- Determine the safe upper time limit from the censoring KM on training data ---
    # Censoring indicator is the *inverse* of the event indicator
    cens_times, cens_surv = kaplan_meier_estimator(
        ~y_train["event"],   # True = censored
        y_train["time"]
    )
    # Find the last time point where the censoring survival is still > 0
    nonzero_mask = cens_surv > 0
    if not nonzero_mask.any():
        # Degenerate fold — skip AUC entirely
        return risk_test, float(cidx), np.full(len(horizons), np.nan)

    max_cens_time = float(cens_times[nonzero_mask].max())

    # Safe upper bound: must be < max test time AND within censoring support
    max_test_time = float(y_test["time"].max())
    upper = min(max_test_time, max_cens_time)

    valid_horizons = np.array([h for h in horizons if h < upper], dtype=float)

    if len(valid_horizons) == 0:
        return risk_test, float(cidx), np.full(len(horizons), np.nan)

    # Drop test subjects whose times fall beyond the censoring support
    # (their IPCW weights would be undefined)
    keep = y_test["time"] <= max_cens_time
    if keep.sum() < 2:
        return risk_test, float(cidx), np.full(len(horizons), np.nan)
    
    print(f"Fold: keeping {keep.sum()}/{len(keep)} test subjects for AUC")

    aucs, _ = cumulative_dynamic_auc(
        y_train, y_test[keep], risk_test[keep], valid_horizons
    )

    full_aucs = np.full(len(horizons), np.nan)
    valid_mask = np.array([h < upper for h in horizons])
    full_aucs[valid_mask] = aucs.astype(float)

    return risk_test, float(cidx), full_aucs

def cross_val_oof_predictions(
    df: pd.DataFrame,
    feature_cols: List[str],
    time_col: str,
    event_col: str,
    estimator,
    cfg: Config
) -> Dict[str, np.ndarray]:
    """
    Repeated K-fold CV out-of-fold (OOF) predictions for one endpoint and one model.
    Returns dict with:
      - oof_risk: length n array (mean across repeats for each sample if repeats>1)
      - oof_cindex_folds: list of fold c-index values
      - oof_auc_folds: list of fold AUC arrays
      - horizons: horizons array
    """
    Xdf = df[feature_cols].copy()
    y = make_y_surv(df, time_col, event_col)

    numeric, categorical = infer_feature_types(df, feature_cols)
    pre = make_preprocessor(numeric, categorical)

    rkf = RepeatedKFold(
        n_splits=cfg.n_splits,
        n_repeats=cfg.n_repeats,
        random_state=cfg.random_state
    )

    n = len(df)
    oof_risk_sum = np.zeros(n, dtype=float)
    oof_risk_count = np.zeros(n, dtype=int)

    fold_cidx = []
    fold_aucs = []

    split_id = 0
    for tr_idx, te_idx in rkf.split(Xdf):
        split_id += 1
        X_train_df, X_test_df = Xdf.iloc[tr_idx], Xdf.iloc[te_idx]
        y_train, y_test = y[tr_idx], y[te_idx]

        risk_test, cidx, aucs = eval_fold(
            preprocessor=clone(pre),
            estimator=estimator,
            X_train_df=X_train_df,
            y_train=y_train,
            X_test_df=X_test_df,
            y_test=y_test,
            horizons=cfg.horizons,
            choose_alpha_for_coxnet=True,
            inner_seed=cfg.random_state + split_id,
        )

        oof_risk_sum[te_idx] += risk_test
        oof_risk_count[te_idx] += 1

        fold_cidx.append(cidx)
        fold_aucs.append(aucs)

    oof_risk = oof_risk_sum / np.maximum(oof_risk_count, 1)

    return {
        "oof_risk": oof_risk,
        "fold_cindex": np.array(fold_cidx, dtype=float),
        "fold_auc": np.vstack(fold_aucs),  # shape: n_folds_total x n_horizons
        "horizons": np.array(cfg.horizons, dtype=float)
    }


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int,
    seed: int,
    alpha: float = 0.05
) -> Tuple[float, float]:
    """Percentile bootstrap CI for a 1D array of bootstrap statistics."""
    rng = np.random.default_rng(seed)
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return lo, hi


def bootstrap_delta_ci(
    stat_a: np.ndarray,
    stat_b: np.ndarray,
    n_boot: int,
    seed: int,
    alpha: float = 0.05
) -> Tuple[float, float]:
    """
    Bootstrap CI for mean difference (b - a) over paired arrays.
    Useful for delta AUC / delta c-index over folds.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(stat_a))
    deltas = []
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        deltas.append(float(np.mean(stat_b[s] - stat_a[s])))
    deltas = np.array(deltas)
    lo = float(np.quantile(deltas, alpha / 2))
    hi = float(np.quantile(deltas, 1 - alpha / 2))
    return lo, hi


def calibration_table_at_horizon(
    df: pd.DataFrame,
    oof_risk: np.ndarray,
    time_col: str,
    event_col: str,
    horizon: float,
    n_bins: int
) -> pd.DataFrame:
    """
    Simple calibration diagnostic:
    - Bin participants by predicted risk (quantiles)
    - Compare mean predicted risk vs observed event rate by horizon

    Note: For strict survival calibration you’d use predicted survival probabilities,
    but many survival models here give risk scores. You can still do a pragmatic
    calibration-style check by comparing:
      - higher risk bins should have higher observed event by horizon.
    If you switch to models that can output survival functions, replace this with
    predicted S(t) vs KM S(t) per bin at horizon.
    """
    tmp = df[[time_col, event_col]].copy()
    tmp["risk"] = oof_risk
    tmp["bin"] = pd.qcut(tmp["risk"], q=n_bins, labels=False, duplicates="drop")

    # Observed event by horizon: event occurred and time <= horizon
    tmp["event_by_h"] = (tmp[event_col].astype(int) == 1) & (tmp[time_col] <= horizon)

    out = (
        tmp.groupby("bin", dropna=True)
        .agg(
            n=("risk", "size"),
            mean_risk=("risk", "mean"),
            obs_event_rate=("event_by_h", "mean"),
        )
        .reset_index()
    )
    out["horizon"] = horizon
    return out

demo_cols = [
        "baseline_age", "sex", "baseline_biomarker_2", "baseline_biomarker_3",
        "baseline_biomarker_1_missing", "baseline_biomarker_2_missing", "baseline_biomarker_3_missing"
    ]

ace_col = "baseline_biomarker_1"
time_col = "time_to_primary_event_months"
event_col = "primary_event_indicator"
dat = pd.read_csv("data/fake_data/mgus2_june_paper_placeholder_cleaned.csv")
# drop rows with missing values
dat = dat.dropna()
df_primary = dat

df_primary.head()

def run_endpoint(
    df_primary: pd.DataFrame,
    endpoint_name: str,
    time_col: str,
    event_col: str,
    demo_cols: List[str],
    ace_col: str,
    #mri_cols: List[str],
    cfg: Config
) -> Dict:
    """
    Run primary vs secondary models for a single endpoint on ACE subset.
    Returns a results dict with fold metrics + deltas + calibration tables.
    """
    models = build_models(cfg)

    # ---- Define feature sets ----
    primary_features = demo_cols + [ace_col]
    secondary_features = demo_cols + [ace_col] #+ mri_cols

    # ---- Fit / evaluate ----
    # Main model choice: Coxnet (keep story clean). Others can be supplementary.
    est = models["coxnet"]

    res_primary = cross_val_oof_predictions(
        df=df_primary,
        feature_cols=primary_features,
        time_col=time_col,
        event_col=event_col,
        estimator=est,
        cfg=cfg
    )
        

    res_secondary = cross_val_oof_predictions(
        df=df_primary,
        feature_cols=secondary_features,
        time_col=time_col,
        event_col=event_col,
        estimator=est,
        cfg=cfg
    )
    

    # ---- Summaries across folds ----
    primary_cidx_mean = float(res_primary["fold_cindex"].mean())
    secondary_cidx_mean = float(res_secondary["fold_cindex"].mean())
    delta_cidx_mean = secondary_cidx_mean - primary_cidx_mean

    # Fold-level AUC summaries (mean over horizons and folds)
    primary_auc_mean_by_h = np.nanmean(res_primary["fold_auc"], axis=0)
    secondary_auc_mean_by_h = np.nanmean(res_secondary["fold_auc"], axis=0)
    delta_auc_mean_by_h = secondary_auc_mean_by_h - primary_auc_mean_by_h

   # ---- Bootstrap CIs over folds (paired deltas) ----
    cidx_delta_ci = bootstrap_delta_ci(
        res_primary["fold_cindex"],
        res_secondary["fold_cindex"],
        n_boot=cfg.n_bootstrap,
        seed=cfg.bootstrap_seed
    )

    auc_delta_ci_by_h = []
    for j, h in enumerate(cfg.horizons):
        ci = bootstrap_delta_ci(
            res_primary["fold_auc"][:, j],
            res_secondary["fold_auc"][:, j],
            n_boot=cfg.n_bootstrap,
            seed=cfg.bootstrap_seed + j + 1
        )
        auc_delta_ci_by_h.append((float(h), ci))

    # ---- Calibration diagnostics at 12 months (or closest horizon) ----
    h_cal = float(cfg.horizons[-1])  # e.g., 365
    calib_primary = calibration_table_at_horizon(
        df=df_primary,
        oof_risk=res_primary["oof_risk"],
        time_col=time_col,
        event_col=event_col,
        horizon=h_cal,
        n_bins=cfg.n_calib_bins
    )
    calib_secondary = calibration_table_at_horizon(
        df=df_primary,
        oof_risk=res_secondary["oof_risk"],
        time_col=time_col,
        event_col=event_col,
        horizon=h_cal,
        n_bins=cfg.n_calib_bins
    )

    return {
        "endpoint": endpoint_name,
        "n": int(len(df_primary)),
        "primary": {
            "features": primary_features,
            "cindex_mean": primary_cidx_mean,
            "auc_mean_by_horizon": primary_auc_mean_by_h.tolist(),
            "oof_risk": res_primary["oof_risk"],  # keep arrays if you want downstream plots
        },
        "secondary": {
            "features": secondary_features,
            "cindex_mean": secondary_cidx_mean,
            "auc_mean_by_horizon": secondary_auc_mean_by_h.tolist(),
            "oof_risk": res_secondary["oof_risk"],
        },
        "delta": {
            "cindex_mean": delta_cidx_mean,
            "cindex_delta_ci": cidx_delta_ci,
            "auc_delta_mean_by_horizon": delta_auc_mean_by_h.tolist(),
            "auc_delta_ci_by_horizon": auc_delta_ci_by_h,
        },
        "horizons": cfg.horizons,
        "calibration_12m": {
            "primary": calib_primary,
            "secondary": calib_secondary,
        }
    }

def main(cfg: Config):
    df = load_data(cfg.data_path)
    df.columns

    demo_cols = [
        "baseline_age", "sex", "baseline_biomarker_2", "baseline_biomarker_3",
        "baseline_biomarker_1_missing", "baseline_biomarker_2_missing", "baseline_biomarker_3_missing"
    ]
    # mri_cols = [
    #     # volumetric MRI columns...
    #     "hippocampus_vol", "entorhinal_vol", "ventricles_vol"
    # ]

    # ---- Primary cohort (ACE subset) ----
    df_primary = get_primary_cohort(df, cfg.ace_col)

    # Basic sanity checks
    required_cols = (
        [cfg.id_col, cfg.ace_col]
        + demo_cols
        # + mri_cols
        + [cfg.time_npe, cfg.event_npe, cfg.time_hosp, cfg.event_hosp]
    )
    missing = [c for c in required_cols if c not in df_primary.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")

    # ---- Run both endpoints ----
    results = {}
    results["npe"] = run_endpoint(
        df_primary=df_primary,
        endpoint_name="neuropsychiatric_event",
        time_col=cfg.time_npe,
        event_col=cfg.event_npe,
        demo_cols=demo_cols,
        ace_col=cfg.ace_col,
       # mri_cols=mri_cols,
        cfg=cfg
    )
    results["hosp"] = run_endpoint(
        df_primary=df_primary,
        endpoint_name="hospitalisation",
        time_col=cfg.time_hosp,
        event_col=cfg.event_hosp,
        demo_cols=demo_cols,
        ace_col=cfg.ace_col,
        # mri_cols=mri_cols,
        cfg=cfg
    )

    # ---- Save light summary (avoid writing huge arrays to JSON) ----
    summary = {
        "config": asdict(cfg),
        "n_primary_cohort": int(len(df_primary)),
        "results": {}
    }
    for k, res in results.items():
        summary["results"][k] = {
            "endpoint": res["endpoint"],
            "n": res["n"],
            "horizons": res["horizons"],
            "primary_cindex_mean": res["primary"]["cindex_mean"],
            "secondary_cindex_mean": res["secondary"]["cindex_mean"],
            "delta_cindex_mean": res["delta"]["cindex_mean"],
            "delta_cindex_ci": res["delta"]["cindex_delta_ci"],
            "primary_auc_mean_by_horizon": res["primary"]["auc_mean_by_horizon"],
            "secondary_auc_mean_by_horizon": res["secondary"]["auc_mean_by_horizon"],
            "delta_auc_mean_by_horizon": res["delta"]["auc_delta_mean_by_horizon"],
            "delta_auc_ci_by_horizon": res["delta"]["auc_delta_ci_by_horizon"],
        }

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/summary.json", "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    # Save calibration tables
    for key, res in results.items():
        res["calibration_12m"]["primary"].to_csv(f"outputs/calibration_{key}_primary.csv", index=False)
        res["calibration_12m"]["secondary"].to_csv(f"outputs/calibration_{key}_secondary.csv", index=False)

    print("Done. Wrote outputs/summary.json and calibration CSVs.")


if __name__ == "__main__":
    main(CFG)




