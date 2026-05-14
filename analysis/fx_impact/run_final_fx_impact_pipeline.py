from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, ElasticNetCV, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.statespace.sarimax import SARIMAX


warnings.filterwarnings("ignore")

RANDOM_STATE = 42
MAX_LAG = 6
TEST_OBS = 24
MIN_TARGET_OBS = 48
FINAL_TARGET_LIMIT = 8
LP_HORIZONS = tuple(range(1, 7))
LP_TEST_FRACTION = 0.25

SCRIPT_PATH = Path(__file__).resolve()
FX_IMPACT_DIR = SCRIPT_PATH.parent
BASE_DIR = SCRIPT_PATH.parents[2]
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = FX_IMPACT_DIR / "reports"
TARGET_REPORT_DIR = REPORT_DIR / "target_selection"
FX_MODEL_REPORT_DIR = REPORT_DIR / "fx_model_selection"
FINAL_REPORT_DIR = REPORT_DIR / "final"
ANOMALY_SET_REPORT_DIR = FINAL_REPORT_DIR / "anomaly_set"
EVENT_PANEL_REPORT_DIR = FINAL_REPORT_DIR / "event_panel"
EVENT_PANEL_PLOT_DIR = EVENT_PANEL_REPORT_DIR / "plots"

MACRO_PATH = DATA_DIR / "integrated_macro_targets.csv"
PERIOD_DEF_PATH = BASE_DIR / "analysis" / "anomaly" / "period_definition.json"

EXCLUDED_LEVEL_COLUMNS = {
    "Date",
    "USD_KRW",
    "FX_Surge",
    "Is_Abnormal_Period",
    "Period_Type",
}

EXTERNAL_CONTROL_COLUMNS = {
    "Policy_Rate_USA",
    "DXY",
    "VIX",
    "WTI_Oil",
    "Rate_Spread_KOR_USA",
}

THEORY_TARGETS = {
    "Import_Price_Index",
    "CPI_KOR",
    "Trade_Balance",
    "Foreign_Stock_Investment",
    "Foreign_Bond_Investment",
    "KOSPI",
}

DIFF_KEYWORDS = (
    "Balance",
    "Account",
    "Investment",
    "Rate",
    "Spread",
    "Unemployment",
    "BSI",
    "CSI",
)


@dataclass
class TargetTransform:
    name: str
    transform: str
    unit_label: str


def ensure_dirs() -> None:
    for path in [
        TARGET_REPORT_DIR,
        TARGET_REPORT_DIR / "plots",
        FX_MODEL_REPORT_DIR,
        FINAL_REPORT_DIR,
        FINAL_REPORT_DIR / "plots",
        ANOMALY_SET_REPORT_DIR,
        ANOMALY_SET_REPORT_DIR / "plots",
        EVENT_PANEL_REPORT_DIR,
        EVENT_PANEL_PLOT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def safe_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def load_macro_dataset(path: Path = MACRO_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError(f"{path} does not contain Date column")
    df["Date"] = pd.to_datetime(df["Date"]) + pd.offsets.MonthEnd(0)
    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    return df


def load_period_definition(path: Path = PERIOD_DEF_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_derived_column(column: str) -> bool:
    return (
        column.endswith("_MoM")
        or column.endswith("_YoY")
        or "_MoM_lag" in column
        or "_YoY_lag" in column
        or column.endswith("_lag1")
        or column.endswith("_lag3")
        or column.endswith("_lag6")
    )


def get_level_target_candidates(df: pd.DataFrame) -> list[str]:
    candidates = []
    for column in df.columns:
        if column in EXCLUDED_LEVEL_COLUMNS:
            continue
        if is_derived_column(column):
            continue
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        candidates.append(column)
    return candidates


def choose_target_transform(series: pd.Series, target: str) -> TargetTransform:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return TargetTransform(target, "diff", "unit_change")
    if any(keyword in target for keyword in DIFF_KEYWORDS):
        return TargetTransform(target, "diff", "unit_change")
    if (clean > 0).all():
        return TargetTransform(target, "log_diff", "log_change")
    return TargetTransform(target, "diff", "unit_change")


def transform_series(series: pd.Series, transform: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if transform == "log_diff":
        numeric = numeric.where(numeric > 0)
        return np.log(numeric).diff()
    return numeric.diff()


def invert_transformed_forecast(
    start_level: float,
    transformed_forecast: np.ndarray,
    transform: str,
) -> np.ndarray:
    levels = []
    prev = float(start_level)
    for value in transformed_forecast:
        if not np.isfinite(value):
            levels.append(np.nan)
            continue
        if transform == "log_diff":
            prev = prev * float(np.exp(value))
        else:
            prev = prev + float(value)
        levels.append(prev)
    return np.array(levels, dtype=float)


def build_analysis_frame(
    df: pd.DataFrame,
    target: str,
    max_lag: int = MAX_LAG,
    subset: str = "full",
) -> tuple[pd.DataFrame, TargetTransform]:
    tmp = df[["Date", "USD_KRW", target, "Is_Abnormal_Period"]].copy()
    transform = choose_target_transform(tmp[target], target)
    tmp["fx_change"] = transform_series(tmp["USD_KRW"], "log_diff")
    tmp["target_change"] = transform_series(tmp[target], transform.transform)
    if subset == "anomaly":
        tmp = tmp[tmp["Is_Abnormal_Period"].eq(1)].copy()
    out = tmp[["Date", "fx_change", "target_change"]].dropna().copy()
    for lag in range(0, max_lag + 1):
        out[f"fx_lag{lag}"] = out["fx_change"].shift(lag)
    out["target_lag1"] = out["target_change"].shift(1)
    return out.dropna().reset_index(drop=True), transform


def corr_pvalue(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    clean = pd.concat([x, y], axis=1).dropna()
    if len(clean) < 8 or clean.iloc[:, 0].nunique() <= 1 or clean.iloc[:, 1].nunique() <= 1:
        return np.nan, np.nan
    corr, pvalue = pearsonr(clean.iloc[:, 0], clean.iloc[:, 1])
    return float(corr), float(pvalue)


def cross_correlation_by_lag(frame: pd.DataFrame, max_lag: int = MAX_LAG) -> pd.DataFrame:
    rows = []
    for lag in range(0, max_lag + 1):
        corr, pvalue = corr_pvalue(frame["target_change"], frame[f"fx_lag{lag}"])
        rows.append({"lag": lag, "corr": corr, "pvalue": pvalue})
    return pd.DataFrame(rows)


def granger_summary(frame: pd.DataFrame, max_lag: int = MAX_LAG) -> dict[str, float | int]:
    data = frame[["target_change", "fx_change"]].dropna()
    if len(data) < max(36, max_lag * 8) or data.nunique().min() <= 1:
        return {"best_lag": np.nan, "min_pvalue": np.nan}
    try:
        results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
    except Exception:
        return {"best_lag": np.nan, "min_pvalue": np.nan}
    pvalues = {
        lag: float(test_result[0]["ssr_ftest"][1])
        for lag, test_result in results.items()
        if np.isfinite(test_result[0]["ssr_ftest"][1])
    }
    if not pvalues:
        return {"best_lag": np.nan, "min_pvalue": np.nan}
    best_lag = min(pvalues, key=pvalues.get)
    return {"best_lag": int(best_lag), "min_pvalue": float(pvalues[best_lag])}


def distributed_lag_regression(frame: pd.DataFrame, max_lag: int = MAX_LAG) -> tuple[dict[str, Any], pd.DataFrame]:
    feature_cols = ["target_lag1"] + [f"fx_lag{lag}" for lag in range(1, max_lag + 1)]
    data = frame[["target_change", *feature_cols]].dropna()
    if len(data) < max(36, len(feature_cols) * 5):
        empty = pd.DataFrame(columns=["lag", "coef", "pvalue", "tvalue"])
        return {"best_lag": np.nan, "coef": np.nan, "pvalue": np.nan, "tvalue": np.nan}, empty

    y = data["target_change"]
    x = add_constant(data[feature_cols], has_constant="add")
    try:
        model = OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": max_lag})
    except Exception:
        model = OLS(y, x).fit()

    rows = []
    for lag in range(1, max_lag + 1):
        col = f"fx_lag{lag}"
        rows.append(
            {
                "lag": lag,
                "coef": float(model.params.get(col, np.nan)),
                "pvalue": float(model.pvalues.get(col, np.nan)),
                "tvalue": float(model.tvalues.get(col, np.nan)),
            }
        )
    detail = pd.DataFrame(rows)
    valid = detail.dropna(subset=["pvalue", "coef"])
    if valid.empty:
        return {"best_lag": np.nan, "coef": np.nan, "pvalue": np.nan, "tvalue": np.nan}, detail
    best = valid.sort_values(["pvalue", "tvalue"], ascending=[True, False]).iloc[0]
    return (
        {
            "best_lag": int(best["lag"]),
            "coef": float(best["coef"]),
            "pvalue": float(best["pvalue"]),
            "tvalue": float(best["tvalue"]),
        },
        detail,
    )


def regularized_lag_model(frame: pd.DataFrame, max_lag: int = MAX_LAG) -> tuple[dict[str, Any], pd.DataFrame]:
    feature_cols = ["target_lag1"] + [f"fx_lag{lag}" for lag in range(1, max_lag + 1)]
    data = frame[["target_change", *feature_cols]].dropna()
    if len(data) < 48:
        empty = pd.DataFrame(columns=["lag", "coef_abs"])
        return {"top_lag": np.nan, "fx_importance": np.nan}, empty

    x = data[feature_cols].to_numpy(dtype=float)
    y = data["target_change"].to_numpy(dtype=float).reshape(-1, 1)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    xs = scaler_x.fit_transform(x)
    ys = scaler_y.fit_transform(y).ravel()
    n_splits = min(5, max(2, len(data) // 36))
    try:
        model = ElasticNetCV(
            l1_ratio=[0.2, 0.5, 0.8, 1.0],
            alphas=np.logspace(-4, 1, 30),
            cv=TimeSeriesSplit(n_splits=n_splits),
            random_state=RANDOM_STATE,
            max_iter=20000,
        )
        model.fit(xs, ys)
        coefs = pd.Series(model.coef_, index=feature_cols)
    except Exception:
        return {"top_lag": np.nan, "fx_importance": np.nan}, pd.DataFrame(columns=["lag", "coef_abs"])

    rows = []
    for lag in range(1, max_lag + 1):
        rows.append({"lag": lag, "coef_abs": float(abs(coefs.get(f"fx_lag{lag}", 0.0)))})
    detail = pd.DataFrame(rows)
    if detail["coef_abs"].sum() <= 0:
        return {"top_lag": np.nan, "fx_importance": 0.0}, detail
    top = detail.sort_values("coef_abs", ascending=False).iloc[0]
    return {"top_lag": int(top["lag"]), "fx_importance": float(detail["coef_abs"].sum())}, detail


def tree_lag_importance(frame: pd.DataFrame, max_lag: int = MAX_LAG) -> tuple[dict[str, Any], pd.DataFrame]:
    feature_cols = ["target_lag1"] + [f"fx_lag{lag}" for lag in range(1, max_lag + 1)]
    data = frame[["target_change", *feature_cols]].dropna()
    if len(data) < 60:
        empty = pd.DataFrame(columns=["lag", "feature_importance", "permutation_importance", "shap_importance"])
        return {"top_lag": np.nan, "fx_importance": np.nan, "shap_importance": np.nan}, empty

    split = max(int(len(data) * 0.8), len(data) - 24)
    split = min(split, len(data) - 12)
    if split <= 20:
        empty = pd.DataFrame(columns=["lag", "feature_importance", "permutation_importance", "shap_importance"])
        return {"top_lag": np.nan, "fx_importance": np.nan, "shap_importance": np.nan}, empty

    train = data.iloc[:split]
    test = data.iloc[split:]
    model = ExtraTreesRegressor(
        n_estimators=250,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(train[feature_cols], train["target_change"])

    perm_values = {col: np.nan for col in feature_cols}
    try:
        perm = permutation_importance(
            model,
            test[feature_cols],
            test["target_change"],
            n_repeats=10,
            random_state=RANDOM_STATE,
        )
        perm_values = dict(zip(feature_cols, perm.importances_mean))
    except Exception:
        pass

    shap_values_by_col = {col: np.nan for col in feature_cols}
    try:
        import shap

        sample = test[feature_cols].iloc[-min(len(test), 80) :]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        mean_abs = np.abs(np.asarray(shap_values)).mean(axis=0)
        shap_values_by_col = dict(zip(feature_cols, mean_abs))
    except Exception:
        pass

    feature_importance = dict(zip(feature_cols, model.feature_importances_))
    rows = []
    for lag in range(1, max_lag + 1):
        col = f"fx_lag{lag}"
        rows.append(
            {
                "lag": lag,
                "feature_importance": float(feature_importance.get(col, np.nan)),
                "permutation_importance": float(perm_values.get(col, np.nan)),
                "shap_importance": float(shap_values_by_col.get(col, np.nan)),
            }
        )
    detail = pd.DataFrame(rows)
    importance_col = "permutation_importance"
    if detail[importance_col].isna().all() or detail[importance_col].abs().sum() <= 0:
        importance_col = "feature_importance"
    valid = detail.dropna(subset=[importance_col])
    if valid.empty:
        return {"top_lag": np.nan, "fx_importance": np.nan, "shap_importance": np.nan}, detail
    top = valid.assign(abs_imp=valid[importance_col].abs()).sort_values("abs_imp", ascending=False).iloc[0]
    shap_sum = detail["shap_importance"].abs().sum()
    return (
        {
            "top_lag": int(top["lag"]),
            "fx_importance": float(valid[importance_col].abs().sum()),
            "shap_importance": float(shap_sum) if np.isfinite(shap_sum) else np.nan,
        },
        detail,
    )


def summarize_subset_methods(
    df: pd.DataFrame,
    target: str,
    subset: str,
    max_lag: int = MAX_LAG,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    frame, transform = build_analysis_frame(df, target, max_lag=max_lag, subset=subset)
    detail_rows = []
    if len(frame) < MIN_TARGET_OBS:
        summary = {
            "target": target,
            "subset": subset,
            "transform": transform.transform,
            "unit_label": transform.unit_label,
            "observations": len(frame),
            "sample_note": "sample_limited",
        }
        return summary, pd.DataFrame(), pd.DataFrame(detail_rows)

    ccf_df = cross_correlation_by_lag(frame, max_lag=max_lag)
    ccf_positive_lags = ccf_df[ccf_df["lag"].gt(0)].dropna(subset=["corr"])
    if ccf_positive_lags.empty:
        ccf_best = {"lag": np.nan, "corr": np.nan, "pvalue": np.nan}
    else:
        ccf_best = ccf_positive_lags.assign(abs_corr=ccf_positive_lags["corr"].abs()).sort_values(
            "abs_corr", ascending=False
        ).iloc[0].to_dict()

    for row in ccf_df.to_dict("records"):
        detail_rows.append(
            {
                "target": target,
                "subset": subset,
                "method": "cross_correlation",
                "lag": row["lag"],
                "value": row["corr"],
                "pvalue": row["pvalue"],
            }
        )

    granger = granger_summary(frame, max_lag=max_lag)
    ardl, ardl_detail = distributed_lag_regression(frame, max_lag=max_lag)
    elastic, elastic_detail = regularized_lag_model(frame, max_lag=max_lag)
    tree, tree_detail = tree_lag_importance(frame, max_lag=max_lag)

    for row in ardl_detail.to_dict("records"):
        detail_rows.append(
            {
                "target": target,
                "subset": subset,
                "method": "distributed_lag_ols",
                "lag": row["lag"],
                "value": row["coef"],
                "pvalue": row["pvalue"],
                "tvalue": row["tvalue"],
            }
        )
    for row in elastic_detail.to_dict("records"):
        detail_rows.append(
            {
                "target": target,
                "subset": subset,
                "method": "elasticnet_abs_coef",
                "lag": row["lag"],
                "value": row["coef_abs"],
            }
        )
    for row in tree_detail.to_dict("records"):
        detail_rows.append(
            {
                "target": target,
                "subset": subset,
                "method": "tree_permutation_importance",
                "lag": row["lag"],
                "value": row["permutation_importance"],
            }
        )
        detail_rows.append(
            {
                "target": target,
                "subset": subset,
                "method": "tree_feature_importance",
                "lag": row["lag"],
                "value": row["feature_importance"],
            }
        )
        detail_rows.append(
            {
                "target": target,
                "subset": subset,
                "method": "tree_shap_abs",
                "lag": row["lag"],
                "value": row["shap_importance"],
            }
        )

    summary = {
        "target": target,
        "subset": subset,
        "transform": transform.transform,
        "unit_label": transform.unit_label,
        "observations": len(frame),
        "sample_note": "ok",
        "ccf_best_lag": ccf_best.get("lag", np.nan),
        "ccf_best_corr": ccf_best.get("corr", np.nan),
        "ccf_best_pvalue": ccf_best.get("pvalue", np.nan),
        "granger_best_lag": granger["best_lag"],
        "granger_min_pvalue": granger["min_pvalue"],
        "ardl_best_lag": ardl["best_lag"],
        "ardl_coef": ardl["coef"],
        "ardl_pvalue": ardl["pvalue"],
        "ardl_tvalue": ardl["tvalue"],
        "elasticnet_top_lag": elastic["top_lag"],
        "elasticnet_fx_importance": elastic["fx_importance"],
        "tree_top_lag": tree["top_lag"],
        "tree_fx_importance": tree["fx_importance"],
        "shap_fx_importance": tree["shap_importance"],
    }
    return summary, frame, pd.DataFrame(detail_rows)


def plot_target_selection_diagnostics(target: str, detail: pd.DataFrame) -> None:
    if detail.empty:
        return
    target_safe = safe_filename(target)
    ccf = detail[(detail["target"].eq(target)) & (detail["method"].eq("cross_correlation"))]
    if not ccf.empty:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for subset, subset_df in ccf.groupby("subset"):
            ax.plot(subset_df["lag"], subset_df["value"], marker="o", label=subset)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"Lead-lag cross-correlation: USD/KRW -> {target}")
        ax.set_xlabel("FX lag in months")
        ax.set_ylabel("Correlation")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(TARGET_REPORT_DIR / "plots" / f"lead_lag_{target_safe}.png", dpi=140)
        plt.close(fig)

    methods = [
        "distributed_lag_ols",
        "elasticnet_abs_coef",
        "tree_permutation_importance",
        "tree_shap_abs",
    ]
    imp = detail[(detail["target"].eq(target)) & (detail["subset"].eq("full")) & (detail["method"].isin(methods))].copy()
    if imp.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    axes = axes.flatten()
    labels = {
        "distributed_lag_ols": "ARDL coefficient",
        "elasticnet_abs_coef": "ElasticNet abs coef",
        "tree_permutation_importance": "Tree permutation",
        "tree_shap_abs": "SHAP mean abs",
    }
    for ax, method in zip(axes, methods):
        subset = imp[imp["method"].eq(method)].sort_values("lag")
        if subset.empty:
            ax.axis("off")
            continue
        ax.bar(subset["lag"], subset["value"].fillna(0.0), color="#4c78a8")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(labels[method])
        ax.grid(axis="y", alpha=0.25)
    for ax in axes:
        ax.set_xlabel("FX lag in months")
    fig.suptitle(f"FX lag importance by method: {target}", y=0.995)
    fig.tight_layout()
    fig.savefig(TARGET_REPORT_DIR / "plots" / f"importance_{target_safe}.png", dpi=140)
    plt.close(fig)


def normalize_signal(series: pd.Series) -> pd.Series:
    clean = series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    max_value = clean.max()
    if max_value <= 0:
        return clean * 0.0
    return (clean / max_value).clip(0, 1)


def add_composite_scores(ranking: pd.DataFrame) -> pd.DataFrame:
    out = ranking.copy()
    out["ccf_signal"] = out["full_ccf_best_corr"].abs()
    out["granger_signal"] = -np.log10(out["full_granger_min_pvalue"].clip(lower=1e-12))
    out["ardl_signal"] = out["full_ardl_tvalue"].abs()
    out["elastic_signal"] = out["full_elasticnet_fx_importance"].abs()
    out["tree_signal"] = out["full_tree_fx_importance"].abs()
    out["shap_signal"] = out["full_shap_fx_importance"].abs()

    sign_full = np.sign(out["full_ardl_coef"].fillna(out["full_ccf_best_corr"]))
    sign_anom = np.sign(out["anomaly_ardl_coef"].fillna(out["anomaly_ccf_best_corr"]))
    sign_agree = (sign_full.eq(sign_anom) & sign_full.ne(0)).astype(float)
    lag_full = out["full_ardl_best_lag"].fillna(out["full_ccf_best_lag"])
    lag_anom = out["anomaly_ardl_best_lag"].fillna(out["anomaly_ccf_best_lag"])
    lag_distance = (lag_full - lag_anom).abs()
    lag_score = (1.0 - (lag_distance / MAX_LAG)).clip(lower=0.0, upper=1.0).fillna(0.0)
    out["stability_score"] = (0.6 * sign_agree + 0.4 * lag_score).where(
        out["anomaly_observations"].ge(MIN_TARGET_OBS), 0.25
    )

    weights = {
        "ccf_signal_norm": 0.18,
        "granger_signal_norm": 0.20,
        "ardl_signal_norm": 0.22,
        "elastic_signal_norm": 0.13,
        "tree_signal_norm": 0.13,
        "shap_signal_norm": 0.04,
        "stability_score": 0.10,
    }
    for col in ["ccf_signal", "granger_signal", "ardl_signal", "elastic_signal", "tree_signal", "shap_signal"]:
        out[f"{col}_norm"] = normalize_signal(out[col])
    out["composite_score"] = sum(out[col] * weight for col, weight in weights.items())

    out["is_external_control"] = out["target"].isin(EXTERNAL_CONTROL_COLUMNS)
    out["is_theory_target"] = out["target"].isin(THEORY_TARGETS)
    out["sample_limited"] = out["full_observations"].lt(96)
    out = out.sort_values("composite_score", ascending=False).reset_index(drop=True)

    eligible = out[
        (~out["is_external_control"])
        & (~out["sample_limited"])
        & out["full_observations"].ge(96)
    ].head(FINAL_TARGET_LIMIT)["target"].tolist()

    theory_with_signal = out[
        out["target"].isin(THEORY_TARGETS)
        & out["full_observations"].ge(96)
        & (
            out["full_granger_min_pvalue"].lt(0.10)
            | out["full_ardl_pvalue"].lt(0.10)
            | out["full_ccf_best_pvalue"].lt(0.10)
        )
    ]["target"].tolist()

    selected = []
    for target in [*theory_with_signal, *eligible]:
        if target not in selected:
            selected.append(target)
    selected = selected[:FINAL_TARGET_LIMIT]
    out["selected_for_final_model"] = out["target"].isin(selected)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def run_target_selection(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = get_level_target_candidates(df)
    summaries: list[dict[str, Any]] = []
    detail_frames = []

    for target in candidates:
        subset_summaries = {}
        for subset in ["full", "anomaly"]:
            summary, _, detail = summarize_subset_methods(df, target, subset=subset, max_lag=MAX_LAG)
            subset_summaries[subset] = summary
            if not detail.empty:
                detail_frames.append(detail)
        merged: dict[str, Any] = {"target": target}
        for subset, summary in subset_summaries.items():
            for key, value in summary.items():
                if key in {"target", "subset"}:
                    continue
                merged[f"{subset}_{key}"] = value
        summaries.append(merged)

    detail_df = pd.concat(detail_frames, axis=0, ignore_index=True) if detail_frames else pd.DataFrame()
    ranking = add_composite_scores(pd.DataFrame(summaries))

    if not detail_df.empty:
        detail_df.to_csv(TARGET_REPORT_DIR / "target_lag_details.csv", index=False)
        for target in ranking["target"].tolist():
            plot_target_selection_diagnostics(target, detail_df)

    ranking.to_csv(TARGET_REPORT_DIR / "target_ranking.csv", index=False)
    with (TARGET_REPORT_DIR / "target_ranking.json").open("w", encoding="utf-8") as f:
        json.dump(ranking.replace({np.nan: None}).to_dict("records"), f, indent=2, ensure_ascii=False)
    write_target_selection_summary(ranking)
    return ranking, detail_df


def write_target_selection_summary(ranking: pd.DataFrame) -> None:
    selected = ranking[ranking["selected_for_final_model"]].copy()
    top = ranking.head(12).copy()
    lines = [
        "# Target Selection Summary",
        "",
        "## Method",
        "",
        "- Target candidates are source level columns from `data/integrated_macro_targets.csv`; derived `_MoM`, `_YoY`, and `_lag` columns are excluded from the target universe.",
        "- USD/KRW is transformed as monthly log change. Positive index/flow targets are log-differenced; balances, rates, spreads, survey indexes, and capital-flow variables are first-differenced.",
        "- Evidence combines lagged cross-correlation, Granger causality, distributed-lag OLS/ARDL, ElasticNet lag selection, ExtraTrees permutation importance, and Tree SHAP where available.",
        "- Full-period and anomaly-month results are compared, but anomaly-only estimates are treated cautiously because several targets have limited overlap.",
        "",
        "## Selected Final Targets",
        "",
    ]
    for row in selected.to_dict("records"):
        lag = row.get("full_ardl_best_lag")
        if not np.isfinite(lag):
            lag = row.get("full_ccf_best_lag")
        lines.append(
            f"- `{row['target']}`: lag {int(lag) if np.isfinite(lag) else 'n/a'}, "
            f"score={row['composite_score']:.3f}, transform={row['full_transform']}"
        )
    lines.extend(["", "## Top Ranking", ""])
    for row in top.to_dict("records"):
        lines.append(
            f"- #{int(row['rank'])} `{row['target']}`: score={row['composite_score']:.3f}, "
            f"Granger p={row['full_granger_min_pvalue']:.4g} if available, "
            f"ARDL lag={row['full_ardl_best_lag']}, selected={bool(row['selected_for_final_model'])}"
        )
    (TARGET_REPORT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_hybrid_results(result_path: Path, source_prefix: str) -> pd.DataFrame:
    if not result_path.exists():
        return pd.DataFrame()
    with result_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for period in data:
        for model_key in ["a", "b"]:
            rows.append(
                {
                    "source_model": f"{source_prefix}_model_{model_key}",
                    "family": source_prefix,
                    "hybrid_component": f"model_{model_key}",
                    "period": period["period"],
                    "rmse": period.get(f"rmse_model_{model_key}"),
                    "mae": period.get(f"mae_model_{model_key}"),
                    "better_model_for_period": period.get("better_model"),
                    "eligible_final": True,
                }
            )
    return pd.DataFrame(rows)


def direction_accuracy(actual: pd.Series, pred: pd.Series, groups: pd.Series | None = None) -> float:
    data = pd.DataFrame({"actual": actual, "pred": pred})
    if groups is not None:
        data["group"] = groups.values
        hits = []
        for _, part in data.groupby("group", sort=False):
            if len(part) < 2:
                continue
            actual_dir = np.sign(part["actual"].diff().dropna())
            pred_dir = np.sign(part["pred"].diff().dropna())
            aligned = pd.concat([actual_dir, pred_dir], axis=1).dropna()
            if not aligned.empty:
                hits.extend((aligned.iloc[:, 0].eq(aligned.iloc[:, 1])).tolist())
        return float(np.mean(hits)) if hits else np.nan
    diffs = data.diff().dropna()
    if diffs.empty:
        return np.nan
    return float(np.mean(np.sign(diffs["actual"]).eq(np.sign(diffs["pred"]))))


def load_prediction_candidates() -> dict[str, pd.DataFrame]:
    sources = {
        "hybrid_mmf": BASE_DIR / "analysis" / "LSTM" / "Hybrid" / "hybrid_mmf" / "eval" / "predictions.csv",
        "hybrid_m2": BASE_DIR / "analysis" / "LSTM" / "Hybrid" / "hybrid_m2" / "eval" / "predictions.csv",
    }
    candidates = {}
    for family, path in sources.items():
        if not path.exists():
            continue
        pred_df = pd.read_csv(path)
        pred_df["date"] = pd.to_datetime(pred_df["date"])
        for col in ["pred_model_a", "pred_model_b"]:
            if col not in pred_df.columns:
                continue
            source_model = f"{family}_{col.replace('pred_', '')}"
            out = pred_df[["date", "actual_fx", col]].rename(columns={col: "pred_fx"}).copy()
            out["source_model"] = source_model
            for optional in ["block_index", "block_start", "block_end"]:
                if optional in pred_df.columns:
                    out[optional] = pred_df[optional]
            candidates[source_model] = out
        for baseline_col in ["pred_arima", "pred_naive"]:
            if baseline_col in pred_df.columns:
                source_model = f"{family}_{baseline_col.replace('pred_', '')}"
                out = pred_df[["date", "actual_fx", baseline_col]].rename(columns={baseline_col: "pred_fx"}).copy()
                out["source_model"] = source_model
                for optional in ["block_index", "block_start", "block_end"]:
                    if optional in pred_df.columns:
                        out[optional] = pred_df[optional]
                candidates[source_model] = out
    return candidates


def monthly_prediction_path(macro_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.Series:
    pred_df = pred_df.copy()
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    monthly = pred_df.set_index("date")["pred_fx"].resample("ME").mean()
    monthly.index = monthly.index + pd.offsets.MonthEnd(0)
    macro_fx = macro_df.set_index("Date")["USD_KRW"].copy()
    out = macro_fx.copy()
    aligned = monthly.reindex(out.index)
    out.loc[aligned.notna()] = aligned.dropna()
    return out


def evaluate_downstream_fx_candidate(
    macro_df: pd.DataFrame,
    selected_targets: list[str],
    pred_df: pd.DataFrame,
    test_obs: int = TEST_OBS,
) -> float:
    fx_path = monthly_prediction_path(macro_df, pred_df)
    actual_fx = macro_df.set_index("Date")["USD_KRW"]
    scores = []
    for target in selected_targets:
        transform = choose_target_transform(macro_df[target], target)
        y = transform_series(macro_df.set_index("Date")[target], transform.transform)
        actual_fx_change = transform_series(actual_fx, "log_diff")
        pred_fx_change = transform_series(fx_path, "log_diff")

        frame = pd.DataFrame({"y": y, "fx_actual": actual_fx_change, "fx_pred": pred_fx_change})
        for lag in range(1, MAX_LAG + 1):
            frame[f"fx_actual_lag{lag}"] = frame["fx_actual"].shift(lag)
            frame[f"fx_pred_lag{lag}"] = frame["fx_pred"].shift(lag)
        frame["y_lag1"] = frame["y"].shift(1)
        frame = frame.dropna()
        if len(frame) < test_obs + 36:
            continue
        train = frame.iloc[:-test_obs]
        test = frame.iloc[-test_obs:]
        feature_actual = ["y_lag1"] + [f"fx_actual_lag{lag}" for lag in range(1, MAX_LAG + 1)]
        feature_pred = ["y_lag1"] + [f"fx_pred_lag{lag}" for lag in range(1, MAX_LAG + 1)]
        model = Ridge(alpha=1.0)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(train[feature_actual])
        model.fit(x_train, train["y"])
        pred = model.predict(scaler.transform(test[feature_pred].rename(columns=lambda c: c.replace("fx_pred", "fx_actual"))))
        scores.append(float(np.sqrt(mean_squared_error(test["y"], pred))))
    return float(np.mean(scores)) if scores else np.nan


def run_fx_model_selection(
    macro_df: pd.DataFrame,
    selected_targets: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    candidates = load_prediction_candidates()
    metric_rows = []
    for source_model, pred_df in candidates.items():
        groups = pred_df["block_index"] if "block_index" in pred_df.columns else None
        daily_rmse = np.sqrt(mean_squared_error(pred_df["actual_fx"], pred_df["pred_fx"]))
        daily_mae = mean_absolute_error(pred_df["actual_fx"], pred_df["pred_fx"])
        monthly = pred_df.set_index("date")[["actual_fx", "pred_fx"]].resample("ME").mean().dropna()
        monthly_dir = direction_accuracy(monthly["actual_fx"], monthly["pred_fx"]) if len(monthly) >= 3 else np.nan
        daily_dir = direction_accuracy(pred_df["actual_fx"], pred_df["pred_fx"], groups=groups)
        downstream = evaluate_downstream_fx_candidate(macro_df, selected_targets, pred_df)
        metric_rows.append(
            {
                "source_model": source_model,
                "daily_available_rmse": float(daily_rmse),
                "daily_available_mae": float(daily_mae),
                "daily_direction_accuracy": daily_dir,
                "monthly_direction_accuracy": monthly_dir,
                "downstream_avg_rmse": downstream,
                "available_daily_rows": len(pred_df),
                "available_months": int(monthly.shape[0]),
                "eligible_final": source_model.endswith("model_a") or source_model.endswith("model_b"),
            }
        )

    comparison = pd.DataFrame(metric_rows)
    result_metrics = pd.concat(
        [
            read_hybrid_results(
                BASE_DIR / "analysis" / "LSTM" / "Hybrid" / "hybrid_mmf" / "results.json",
                "hybrid_mmf",
            ),
            read_hybrid_results(
                BASE_DIR / "analysis" / "LSTM" / "Hybrid" / "hybrid_m2" / "results.json",
                "hybrid_m2",
            ),
        ],
        axis=0,
        ignore_index=True,
    )
    if not result_metrics.empty:
        pivot = result_metrics.pivot_table(index="source_model", columns="period", values=["rmse", "mae"], aggfunc="first")
        pivot.columns = [f"{metric}_{period}" for metric, period in pivot.columns]
        pivot = pivot.reset_index()
        comparison = comparison.merge(pivot, on="source_model", how="left")

    eligible = comparison[comparison["eligible_final"]].copy()
    rank_cols = [
        "rmse_full_1995_2026",
        "rmse_anomaly_concatenated_blocks",
        "daily_available_rmse",
        "downstream_avg_rmse",
    ]
    for col in rank_cols:
        if col not in eligible.columns:
            eligible[col] = np.nan
        eligible[f"{col}_rank"] = eligible[col].rank(method="dense", ascending=True, na_option="bottom")
    eligible["selection_score"] = eligible[[f"{col}_rank" for col in rank_cols]].sum(axis=1)
    selected_source = eligible.sort_values(["selection_score", "rmse_anomaly_concatenated_blocks"]).iloc[0]["source_model"]
    comparison = comparison.merge(
        eligible[["source_model", "selection_score"]],
        on="source_model",
        how="left",
    )
    comparison["selected"] = comparison["source_model"].eq(selected_source)
    comparison = comparison.sort_values(["eligible_final", "selection_score", "daily_available_rmse"], ascending=[False, True, True])
    comparison.to_csv(FX_MODEL_REPORT_DIR / "fx_model_comparison.csv", index=False)

    selected_predictions = candidates[selected_source].copy()
    selected_predictions = selected_predictions[["date", "actual_fx", "pred_fx", "source_model"] + [
        col for col in ["block_index", "block_start", "block_end"] if col in selected_predictions.columns
    ]]
    selected_predictions.to_csv(FX_MODEL_REPORT_DIR / "selected_fx_predictions.csv", index=False)
    write_fx_model_selection_summary(comparison, selected_source)
    plot_fx_model_selection(comparison)
    return comparison, selected_predictions, selected_source


def write_fx_model_selection_summary(comparison: pd.DataFrame, selected_source: str) -> None:
    selected = comparison[comparison["selected"]].iloc[0]
    lines = [
        "# FX Model Selection",
        "",
        "## Compared Inputs",
        "",
        "- Existing `hybrid_mmf` and `hybrid_m2` result JSON files are compared for full-period and anomaly-block RMSE/MAE.",
        "- Daily anomaly-block prediction CSV files are also compared by direct RMSE/MAE, direction accuracy, monthly direction accuracy, and downstream distributed-lag RMSE.",
        "- The selected file `selected_fx_predictions.csv` keeps the daily prediction path; the final macro pipeline resamples it to month-end by monthly mean and fills non-predicted months with actual USD/KRW for controlled error-propagation tests.",
        "",
        "## Selected FX Input",
        "",
        f"- Selected source: `{selected_source}`",
        f"- Full-period RMSE: {selected.get('rmse_full_1995_2026', np.nan):.4f}",
        f"- Anomaly-block RMSE: {selected.get('rmse_anomaly_concatenated_blocks', np.nan):.4f}",
        f"- Downstream average RMSE: {selected.get('downstream_avg_rmse', np.nan):.6f}",
        "",
        "## Ranking",
        "",
    ]
    for row in comparison.to_dict("records"):
        lines.append(
            f"- `{row['source_model']}`: selected={bool(row['selected'])}, "
            f"daily RMSE={row['daily_available_rmse']:.4f}, "
            f"downstream RMSE={row.get('downstream_avg_rmse', np.nan):.6f}, "
            f"score={row.get('selection_score', np.nan)}"
        )
    (FX_MODEL_REPORT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_fx_model_selection(comparison: pd.DataFrame) -> None:
    eligible = comparison[comparison["eligible_final"]].copy()
    if eligible.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    eligible = eligible.sort_values("daily_available_rmse")
    ax.barh(eligible["source_model"], eligible["daily_available_rmse"], color="#4c78a8")
    ax.set_title("FX prediction candidates: available anomaly-block RMSE")
    ax.set_xlabel("USD/KRW RMSE")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FX_MODEL_REPORT_DIR / "fx_candidate_rmse.png", dpi=140)
    plt.close(fig)


def build_fx_level_path(
    macro_df: pd.DataFrame,
    selected_predictions: pd.DataFrame,
    mode: str,
    scenario_shock_pct: float = 0.05,
    scenario_start: pd.Timestamp | None = None,
    custom_shock_path: Path | None = None,
) -> pd.Series:
    dates = pd.to_datetime(macro_df["Date"])
    actual = macro_df.set_index("Date")["USD_KRW"].copy()
    if mode == "actual":
        return actual
    predicted = monthly_prediction_path(macro_df, selected_predictions)
    if mode == "predicted":
        return predicted
    if mode != "scenario":
        raise ValueError(f"Unsupported FX mode: {mode}")

    scenario = predicted.copy()
    if custom_shock_path is not None and custom_shock_path.exists():
        custom = pd.read_csv(custom_shock_path)
        date_col = "date" if "date" in custom.columns else "Date"
        value_col = "fx" if "fx" in custom.columns else "pred_fx" if "pred_fx" in custom.columns else "USD_KRW"
        custom[date_col] = pd.to_datetime(custom[date_col]) + pd.offsets.MonthEnd(0)
        custom_monthly = custom.set_index(date_col)[value_col].resample("ME").mean()
        custom_monthly.index = custom_monthly.index + pd.offsets.MonthEnd(0)
        aligned = custom_monthly.reindex(scenario.index)
        scenario.loc[aligned.notna()] = aligned.dropna()
        return scenario

    if scenario_start is None:
        scenario_start = dates.iloc[-TEST_OBS]
    scenario_start = pd.to_datetime(scenario_start) + pd.offsets.MonthEnd(0)
    scenario.loc[scenario.index >= scenario_start] = scenario.loc[scenario.index >= scenario_start] * (1.0 + scenario_shock_pct)
    return scenario


def get_target_lag_map(ranking: pd.DataFrame, selected_targets: list[str]) -> dict[str, int]:
    lag_map = {}
    indexed = ranking.set_index("target")
    for target in selected_targets:
        row = indexed.loc[target]
        lag = row.get("full_ardl_best_lag")
        if not np.isfinite(lag):
            lag = row.get("full_ccf_best_lag")
        if not np.isfinite(lag) or int(lag) <= 0:
            lag = 1
        lag_map[target] = int(min(max(lag, 1), MAX_LAG))
    return lag_map


def build_model_frame(
    macro_df: pd.DataFrame,
    target: str,
    fx_level_path: pd.Series,
    max_lag: int = MAX_LAG,
) -> tuple[pd.DataFrame, TargetTransform]:
    idx = pd.to_datetime(macro_df["Date"])
    target_series = pd.Series(macro_df[target].values, index=idx)
    transform = choose_target_transform(target_series, target)
    y = transform_series(target_series, transform.transform)
    fx = transform_series(fx_level_path.reindex(idx), "log_diff")
    frame = pd.DataFrame({"target_change": y, "fx_change": fx}, index=idx)
    for lag in range(1, max_lag + 1):
        frame[f"fx_lag{lag}"] = frame["fx_change"].shift(lag)
    frame["target_lag1"] = frame["target_change"].shift(1)
    return frame, transform


def evaluate_prediction_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    forecast_df = pd.DataFrame(records)
    if forecast_df.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in forecast_df.dropna(subset=["actual_transform", "forecast_transform"]).groupby(
        ["model", "fx_mode", "target", "transform"], dropna=False
    ):
        model, fx_mode, target, transform = keys
        rows.append(
            {
                "model": model,
                "fx_mode": fx_mode,
                "target": target,
                "transform": transform,
                "rmse_transform": float(np.sqrt(mean_squared_error(group["actual_transform"], group["forecast_transform"]))),
                "mae_transform": float(mean_absolute_error(group["actual_transform"], group["forecast_transform"])),
                "rmse_level": float(np.sqrt(mean_squared_error(group["actual_level"], group["forecast_level"]))),
                "mae_level": float(mean_absolute_error(group["actual_level"], group["forecast_level"])),
                "nrmse_transform": float(
                    np.sqrt(mean_squared_error(group["actual_transform"], group["forecast_transform"]))
                    / max(group["actual_transform"].std(ddof=0), 1e-12)
                ),
                "nrmse_level": float(
                    np.sqrt(mean_squared_error(group["actual_level"], group["forecast_level"]))
                    / max(group["actual_level"].std(ddof=0), 1e-12)
                ),
                "rows": len(group),
            }
        )
    metrics = pd.DataFrame(rows)
    overall = (
        metrics.groupby(["model", "fx_mode"], as_index=False)
        .agg(
            avg_rmse_transform=("rmse_transform", "mean"),
            avg_mae_transform=("mae_transform", "mean"),
            avg_rmse_level=("rmse_level", "mean"),
            avg_mae_level=("mae_level", "mean"),
            avg_nrmse_transform=("nrmse_transform", "mean"),
            avg_nrmse_level=("nrmse_level", "mean"),
            targets=("target", "nunique"),
        )
        .assign(target="__ALL__", transform="mixed", rows=np.nan)
    )
    return pd.concat([metrics, overall], axis=0, ignore_index=True)


def run_arimax_models(
    macro_df: pd.DataFrame,
    selected_targets: list[str],
    lag_map: dict[str, int],
    fx_paths: dict[str, pd.Series],
    test_obs: int = TEST_OBS,
) -> list[dict[str, Any]]:
    records = []
    actual_fx_path = fx_paths["actual"]
    for target in selected_targets:
        lag = lag_map[target]
        actual_frame, transform = build_model_frame(macro_df, target, actual_fx_path)
        level_series = macro_df.set_index("Date")[target]
        model_data = actual_frame[["target_change", f"fx_lag{lag}"]].dropna()
        if len(model_data) < test_obs + 36:
            continue
        train = model_data.iloc[:-test_obs]
        test = model_data.iloc[-test_obs:]
        try:
            model = SARIMAX(
                train["target_change"],
                exog=train[[f"fx_lag{lag}"]],
                order=(1, 0, 1),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False, maxiter=300)
        except Exception:
            continue
        train_end = train.index[-1]
        start_level = float(level_series.loc[train_end])
        actual_level = level_series.reindex(test.index).to_numpy(dtype=float)
        actual_transform = actual_frame["target_change"].reindex(test.index).to_numpy(dtype=float)
        for mode, fx_path in fx_paths.items():
            mode_frame, _ = build_model_frame(macro_df, target, fx_path)
            exog_future = mode_frame[[f"fx_lag{lag}"]].reindex(test.index)
            if exog_future.isna().any().any():
                continue
            try:
                forecast = fitted.forecast(steps=len(test), exog=exog_future)
            except Exception:
                continue
            forecast_values = np.asarray(forecast, dtype=float)
            forecast_level = invert_transformed_forecast(start_level, forecast_values, transform.transform)
            for horizon, date in enumerate(test.index, start=1):
                records.append(
                    {
                        "date": date,
                        "horizon": horizon,
                        "target": target,
                        "model": "ARIMAX",
                        "fx_mode": mode,
                        "lag_used": lag,
                        "transform": transform.transform,
                        "unit_label": transform.unit_label,
                        "forecast_transform": forecast_values[horizon - 1],
                        "actual_transform": actual_transform[horizon - 1],
                        "forecast_level": forecast_level[horizon - 1],
                        "actual_level": actual_level[horizon - 1],
                    }
                )
    return records


def recursive_per_target_forecast(
    macro_df: pd.DataFrame,
    target: str,
    fx_paths: dict[str, pd.Series],
    model_kind: str,
    test_obs: int = TEST_OBS,
) -> list[dict[str, Any]]:
    actual_frame, transform = build_model_frame(macro_df, target, fx_paths["actual"])
    feature_cols = ["target_lag1"] + [f"fx_lag{lag}" for lag in range(1, MAX_LAG + 1)]
    data = actual_frame[["target_change", *feature_cols]].dropna()
    if len(data) < test_obs + 36:
        return []
    train = data.iloc[:-test_obs]
    test = data.iloc[-test_obs:]
    if model_kind == "RIDGE_DLM":
        model = Ridge(alpha=1.0)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(train[feature_cols])
        model.fit(x_train, train["target_change"])
    elif model_kind == "TREE_DLM":
        model = ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        scaler = None
        model.fit(train[feature_cols], train["target_change"])
    else:
        raise ValueError(model_kind)

    level_series = macro_df.set_index("Date")[target]
    train_end = train.index[-1]
    start_level = float(level_series.loc[train_end])
    actual_level = level_series.reindex(test.index).to_numpy(dtype=float)
    actual_transform = actual_frame["target_change"].reindex(test.index).to_numpy(dtype=float)
    records = []

    for mode, fx_path in fx_paths.items():
        mode_frame, _ = build_model_frame(macro_df, target, fx_path)
        y_history = actual_frame["target_change"].loc[:train_end].dropna().to_dict()
        forecasts = []
        for date in test.index:
            feature_values = {"target_lag1": y_history.get(date - pd.offsets.MonthEnd(1), np.nan)}
            for lag in range(1, MAX_LAG + 1):
                feature_values[f"fx_lag{lag}"] = mode_frame.at[date, f"fx_lag{lag}"] if date in mode_frame.index else np.nan
            x_row = pd.DataFrame([feature_values], columns=feature_cols)
            if x_row.isna().any().any():
                pred = np.nan
            elif model_kind == "RIDGE_DLM":
                pred = float(model.predict(scaler.transform(x_row))[0])
            else:
                pred = float(model.predict(x_row)[0])
            forecasts.append(pred)
            y_history[date] = pred
        forecast_values = np.asarray(forecasts, dtype=float)
        forecast_level = invert_transformed_forecast(start_level, forecast_values, transform.transform)
        for horizon, date in enumerate(test.index, start=1):
            records.append(
                {
                    "date": date,
                    "horizon": horizon,
                    "target": target,
                    "model": model_kind,
                    "fx_mode": mode,
                    "lag_used": np.nan,
                    "transform": transform.transform,
                    "unit_label": transform.unit_label,
                    "forecast_transform": forecast_values[horizon - 1],
                    "actual_transform": actual_transform[horizon - 1],
                    "forecast_level": forecast_level[horizon - 1],
                    "actual_level": actual_level[horizon - 1],
                }
            )
    return records


def run_dlm_models(
    macro_df: pd.DataFrame,
    selected_targets: list[str],
    fx_paths: dict[str, pd.Series],
    test_obs: int = TEST_OBS,
) -> list[dict[str, Any]]:
    records = []
    for target in selected_targets:
        records.extend(recursive_per_target_forecast(macro_df, target, fx_paths, "RIDGE_DLM", test_obs=test_obs))
        records.extend(recursive_per_target_forecast(macro_df, target, fx_paths, "TREE_DLM", test_obs=test_obs))
    return records


def run_varx_model(
    macro_df: pd.DataFrame,
    selected_targets: list[str],
    fx_paths: dict[str, pd.Series],
    test_obs: int = TEST_OBS,
) -> list[dict[str, Any]]:
    var_targets = selected_targets[: min(6, len(selected_targets))]
    idx = pd.to_datetime(macro_df["Date"])
    level_df = macro_df.set_index("Date")[var_targets]
    transform_map = {target: choose_target_transform(level_df[target], target) for target in var_targets}
    endog = pd.DataFrame(index=idx)
    for target in var_targets:
        endog[target] = transform_series(level_df[target], transform_map[target].transform)
    actual_fx = transform_series(fx_paths["actual"].reindex(idx), "log_diff")
    exog = pd.DataFrame(index=idx)
    for lag in range(1, MAX_LAG + 1):
        exog[f"fx_lag{lag}"] = actual_fx.shift(lag)
    combined = pd.concat([endog, exog], axis=1).dropna()
    if len(combined) < test_obs + 48:
        return []
    train = combined.iloc[:-test_obs]
    test = combined.iloc[-test_obs:]
    endog_train = train[var_targets]
    exog_train = train[[f"fx_lag{lag}" for lag in range(1, MAX_LAG + 1)]]
    try:
        model = VAR(endog_train, exog=exog_train)
        lag_selection = model.select_order(maxlags=min(3, max(1, len(endog_train) // 20)))
        selected_lag = int(lag_selection.aic)
        selected_lag = max(selected_lag, 1)
        fitted = model.fit(selected_lag)
    except Exception:
        return []

    records = []
    actual_levels = level_df.reindex(test.index)
    actual_transforms = endog.reindex(test.index)
    start_levels = level_df.reindex(train.index).iloc[-1]
    lagged_values = endog_train.values[-selected_lag:]

    for mode, fx_path in fx_paths.items():
        fx_change = transform_series(fx_path.reindex(idx), "log_diff")
        exog_future = pd.DataFrame(index=idx)
        for lag in range(1, MAX_LAG + 1):
            exog_future[f"fx_lag{lag}"] = fx_change.shift(lag)
        exog_test = exog_future.reindex(test.index)
        if exog_test.isna().any().any():
            continue
        try:
            forecast = fitted.forecast(lagged_values, steps=len(test), exog_future=exog_test.values)
        except Exception:
            continue
        forecast_df = pd.DataFrame(forecast, index=test.index, columns=var_targets)
        level_forecasts = {}
        for target in var_targets:
            level_forecasts[target] = invert_transformed_forecast(
                float(start_levels[target]),
                forecast_df[target].to_numpy(dtype=float),
                transform_map[target].transform,
            )
        for target in var_targets:
            transform = transform_map[target]
            for horizon, date in enumerate(test.index, start=1):
                records.append(
                    {
                        "date": date,
                        "horizon": horizon,
                        "target": target,
                        "model": "VARX",
                        "fx_mode": mode,
                        "lag_used": selected_lag,
                        "transform": transform.transform,
                        "unit_label": transform.unit_label,
                        "forecast_transform": float(forecast_df.at[date, target]),
                        "actual_transform": float(actual_transforms.at[date, target]),
                        "forecast_level": float(level_forecasts[target][horizon - 1]),
                        "actual_level": float(actual_levels.at[date, target]),
                    }
                )
    return records


def build_scenario_forecasts(forecast_df: pd.DataFrame) -> pd.DataFrame:
    baseline = forecast_df[forecast_df["fx_mode"].eq("predicted")].copy()
    scenario = forecast_df[forecast_df["fx_mode"].eq("scenario")].copy()
    if baseline.empty or scenario.empty:
        return pd.DataFrame()
    key_cols = ["date", "horizon", "target", "model", "transform", "unit_label"]
    merged = scenario.merge(
        baseline[key_cols + ["forecast_transform", "forecast_level"]],
        on=key_cols,
        how="inner",
        suffixes=("_scenario", "_baseline"),
    )
    merged["impact_transform_delta"] = merged["forecast_transform_scenario"] - merged["forecast_transform_baseline"]
    merged["impact_level_delta"] = merged["forecast_level_scenario"] - merged["forecast_level_baseline"]
    merged["impact_level_pct_delta"] = np.where(
        merged["forecast_level_baseline"].abs() > 1e-12,
        merged["impact_level_delta"] / merged["forecast_level_baseline"] * 100.0,
        np.nan,
    )
    return merged[
        [
            "date",
            "horizon",
            "target",
            "model",
            "transform",
            "unit_label",
            "forecast_transform_baseline",
            "forecast_transform_scenario",
            "impact_transform_delta",
            "forecast_level_baseline",
            "forecast_level_scenario",
            "impact_level_delta",
            "impact_level_pct_delta",
        ]
    ]


def build_lag_effect_summary(
    ranking: pd.DataFrame,
    selected_targets: list[str],
    scenario_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    indexed = ranking.set_index("target")
    for target in selected_targets:
        if target not in indexed.index:
            continue
        row = indexed.loc[target]
        lag = row.get("full_ardl_best_lag")
        if not np.isfinite(lag):
            lag = row.get("full_ccf_best_lag")
        effect_per_1pct = row.get("full_ardl_coef", np.nan)
        if np.isfinite(effect_per_1pct):
            effect_per_1pct = effect_per_1pct * 0.01
        target_scenario = scenario_df[scenario_df["target"].eq(target)].copy()
        if target_scenario.empty:
            peak = {}
        else:
            peak_row = target_scenario.assign(abs_delta=target_scenario["impact_level_delta"].abs()).sort_values(
                "abs_delta", ascending=False
            ).iloc[0]
            peak = peak_row.to_dict()
        rows.append(
            {
                "target": target,
                "selected_lag_months": int(lag) if np.isfinite(lag) else np.nan,
                "transform": row.get("full_transform"),
                "unit_label": row.get("full_unit_label"),
                "ardl_coef": row.get("full_ardl_coef"),
                "effect_per_1pct_fx_move": effect_per_1pct,
                "ardl_pvalue": row.get("full_ardl_pvalue"),
                "granger_min_pvalue": row.get("full_granger_min_pvalue"),
                "ccf_best_corr": row.get("full_ccf_best_corr"),
                "tree_fx_importance": row.get("full_tree_fx_importance"),
                "stability_score": row.get("stability_score"),
                "peak_scenario_model": peak.get("model"),
                "peak_scenario_horizon": peak.get("horizon"),
                "peak_scenario_level_delta": peak.get("impact_level_delta"),
                "peak_scenario_pct_delta": peak.get("impact_level_pct_delta"),
            }
        )
    return pd.DataFrame(rows)


def select_plot_models_by_target(forecast_df: pd.DataFrame, selected_targets: list[str]) -> pd.DataFrame:
    rows = []
    predicted = forecast_df[forecast_df["fx_mode"].eq("predicted")].copy()
    for target in selected_targets:
        target_predicted = predicted[predicted["target"].eq(target)]
        if target_predicted.empty:
            continue
        for model, group in target_predicted.groupby("model"):
            clean = group.dropna(subset=["actual_level", "forecast_level"])
            if clean.empty:
                continue
            rmse_level = float(np.sqrt(mean_squared_error(clean["actual_level"], clean["forecast_level"])))
            nrmse_level = rmse_level / max(float(clean["actual_level"].std(ddof=0)), 1e-12)
            rmse_transform = float(np.sqrt(mean_squared_error(clean["actual_transform"], clean["forecast_transform"])))
            nrmse_transform = rmse_transform / max(float(clean["actual_transform"].std(ddof=0)), 1e-12)
            rows.append(
                {
                    "target": target,
                    "model": model,
                    "rmse_level": rmse_level,
                    "nrmse_level": nrmse_level,
                    "rmse_transform": rmse_transform,
                    "nrmse_transform": nrmse_transform,
                    "rows": len(clean),
                    "start_date": clean["date"].min(),
                    "end_date": clean["date"].max(),
                }
            )
    selection = pd.DataFrame(rows)
    if selection.empty:
        return selection
    selection["selected_for_plot"] = False
    latest_end = selection.groupby("target")["end_date"].transform("max")
    same_latest_window = selection[selection["end_date"].eq(latest_end)].copy()
    best_idx = (
        same_latest_window.sort_values(["target", "nrmse_level", "nrmse_transform"])
        .groupby("target")
        .head(1)
        .index
    )
    selection.loc[best_idx, "selected_for_plot"] = True
    return selection.sort_values(["target", "selected_for_plot", "nrmse_level"], ascending=[True, False, True])


def plot_final_forecasts(forecast_df: pd.DataFrame, selected_targets: list[str]) -> None:
    if forecast_df.empty:
        return
    for old_plot in (FINAL_REPORT_DIR / "plots").glob("forecast_*.png"):
        old_plot.unlink()
    plot_selection = select_plot_models_by_target(forecast_df, selected_targets)
    if not plot_selection.empty:
        plot_selection.to_csv(FINAL_REPORT_DIR / "plot_model_selection.csv", index=False)
    selected_model_by_target = (
        plot_selection[plot_selection["selected_for_plot"]].set_index("target")["model"].to_dict()
        if not plot_selection.empty
        else {}
    )
    for target in selected_targets:
        preferred_model = selected_model_by_target.get(target, "RIDGE_DLM")
        part = forecast_df[(forecast_df["target"].eq(target)) & (forecast_df["model"].eq(preferred_model))].copy()
        if part.empty:
            part = forecast_df[forecast_df["target"].eq(target)].copy()
        if part.empty:
            continue
        model_label = part["model"].iloc[0]
        fig, ax = plt.subplots(figsize=(11, 5.5))
        actual = part.drop_duplicates("date").sort_values("date")
        ax.plot(actual["date"], actual["actual_level"], color="black", linewidth=1.8, marker="o", label="Actual target")
        for mode, style in [("actual", "--"), ("predicted", ":"), ("scenario", "-.")]:
            mode_part = part[part["fx_mode"].eq(mode)].sort_values("date")
            if mode_part.empty:
                continue
            ax.plot(
                mode_part["date"],
                mode_part["forecast_level"],
                linestyle=style,
                linewidth=1.5,
                marker="x",
                label=f"{model_label} with {mode} FX",
            )
        ax.set_title(f"Final impact forecast: {target} ({model_label}, best predicted-FX level fit)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Level")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FINAL_REPORT_DIR / "plots" / f"forecast_{safe_filename(target)}.png", dpi=140)
        plt.close(fig)


def build_anomaly_concat_macro(macro_df: pd.DataFrame) -> pd.DataFrame:
    anomaly = macro_df[macro_df["Is_Abnormal_Period"].eq(1)].copy()
    anomaly = anomaly.sort_values("Date").dropna(subset=["USD_KRW"]).reset_index(drop=True)
    anomaly["concat_step"] = np.arange(1, len(anomaly) + 1)
    return anomaly


def build_anomaly_fx_paths(
    anomaly_df: pd.DataFrame,
    selected_predictions: pd.DataFrame,
    scenario_shock_pct: float,
    test_obs: int = TEST_OBS,
) -> dict[str, pd.Series]:
    idx = anomaly_df.index
    dates = pd.to_datetime(anomaly_df["Date"])
    actual = pd.Series(anomaly_df["USD_KRW"].to_numpy(dtype=float), index=idx)

    pred_daily = selected_predictions.copy()
    pred_daily["date"] = pd.to_datetime(pred_daily["date"])
    monthly_pred = pred_daily.set_index("date")["pred_fx"].resample("ME").mean()
    monthly_pred.index = monthly_pred.index + pd.offsets.MonthEnd(0)
    monthly_pred_by_date = monthly_pred.reindex(dates).to_numpy(dtype=float)

    predicted = actual.copy()
    mask = np.isfinite(monthly_pred_by_date)
    predicted.iloc[mask] = monthly_pred_by_date[mask]
    scenario = predicted.copy()
    shock_window = max(8, min(test_obs, 12))
    shock_start = max(1, len(scenario) - shock_window)
    scenario.iloc[shock_start:] = scenario.iloc[shock_start:] * (1.0 + scenario_shock_pct)
    return {"actual": actual, "predicted": predicted, "scenario": scenario}


def build_anomaly_target_frame(
    anomaly_df: pd.DataFrame,
    target: str,
    fx_level: pd.Series,
    max_lag: int = MAX_LAG,
) -> tuple[pd.DataFrame, TargetTransform]:
    transform = choose_target_transform(anomaly_df[target], target)
    idx = anomaly_df.index
    target_level = pd.Series(pd.to_numeric(anomaly_df[target], errors="coerce").to_numpy(dtype=float), index=idx)
    fx_level = fx_level.reindex(idx)
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(anomaly_df["Date"]).to_numpy(),
            "concat_step": anomaly_df["concat_step"].to_numpy(dtype=int),
            "target_level": target_level,
            "target_change": transform_series(target_level, transform.transform),
            "fx_level": fx_level,
            "fx_change": transform_series(fx_level, "log_diff"),
        },
        index=idx,
    )
    for lag in range(1, max_lag + 1):
        frame[f"fx_lag{lag}"] = frame["fx_change"].shift(lag)
    frame["target_lag1"] = frame["target_change"].shift(1)
    return frame, transform


def choose_anomaly_test_obs(row_count: int, requested_test_obs: int = TEST_OBS) -> int:
    if row_count < 36:
        return max(6, row_count // 4)
    return min(requested_test_obs, max(12, int(row_count * 0.25)))


def build_anomaly_model_panel(
    anomaly_df: pd.DataFrame,
    selected_targets: list[str],
    fx_paths: dict[str, pd.Series],
) -> pd.DataFrame:
    rows = []
    for target in selected_targets:
        if target not in anomaly_df.columns:
            continue
        actual_frame, transform = build_anomaly_target_frame(anomaly_df, target, fx_paths["actual"])
        panel = actual_frame[["date", "concat_step", "target_level", "target_change", "target_lag1"]].copy()
        panel["target"] = target
        panel["transform"] = transform.transform
        panel["unit_label"] = transform.unit_label
        for mode, fx_path in fx_paths.items():
            mode_frame, _ = build_anomaly_target_frame(anomaly_df, target, fx_path)
            panel[f"{mode}_fx_level"] = mode_frame["fx_level"]
            panel[f"{mode}_fx_change"] = mode_frame["fx_change"]
            for lag in range(1, MAX_LAG + 1):
                panel[f"{mode}_fx_lag{lag}"] = mode_frame[f"fx_lag{lag}"]
        rows.append(panel)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, axis=0, ignore_index=True)


def run_anomaly_arimax_models(
    anomaly_df: pd.DataFrame,
    selected_targets: list[str],
    lag_map: dict[str, int],
    fx_paths: dict[str, pd.Series],
    test_obs: int = TEST_OBS,
) -> list[dict[str, Any]]:
    records = []
    for target in selected_targets:
        lag = lag_map[target]
        actual_frame, transform = build_anomaly_target_frame(anomaly_df, target, fx_paths["actual"])
        data = actual_frame[["target_change", f"fx_lag{lag}", "date", "concat_step", "target_level"]].dropna()
        if len(data) < 36:
            continue
        target_test_obs = choose_anomaly_test_obs(len(data), test_obs)
        if len(data) - target_test_obs < 24:
            continue
        train = data.iloc[:-target_test_obs]
        test = data.iloc[-target_test_obs:]
        try:
            fitted = SARIMAX(
                train["target_change"],
                exog=train[[f"fx_lag{lag}"]],
                order=(1, 0, 1),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=300)
        except Exception:
            continue
        start_level = float(train["target_level"].iloc[-1])
        for mode, fx_path in fx_paths.items():
            mode_frame, _ = build_anomaly_target_frame(anomaly_df, target, fx_path)
            exog_future = mode_frame[[f"fx_lag{lag}"]].reindex(test.index)
            if exog_future.isna().any().any():
                continue
            try:
                forecast = np.asarray(fitted.forecast(steps=len(test), exog=exog_future), dtype=float)
            except Exception:
                continue
            forecast_level = invert_transformed_forecast(start_level, forecast, transform.transform)
            for horizon, (idx, row) in enumerate(test.iterrows(), start=1):
                records.append(
                    {
                        "date": row["date"],
                        "concat_step": int(row["concat_step"]),
                        "horizon": horizon,
                        "target": target,
                        "model": "ARIMAX",
                        "fx_mode": mode,
                        "lag_used": lag,
                        "transform": transform.transform,
                        "unit_label": transform.unit_label,
                        "forecast_transform": forecast[horizon - 1],
                        "actual_transform": row["target_change"],
                        "forecast_level": forecast_level[horizon - 1],
                        "actual_level": row["target_level"],
                        "sample_scope": "anomaly_concatenated",
                    }
                )
    return records


def run_anomaly_dlm_models(
    anomaly_df: pd.DataFrame,
    selected_targets: list[str],
    fx_paths: dict[str, pd.Series],
    test_obs: int = TEST_OBS,
) -> list[dict[str, Any]]:
    records = []
    feature_cols = ["target_lag1"] + [f"fx_lag{lag}" for lag in range(1, MAX_LAG + 1)]
    for target in selected_targets:
        actual_frame, transform = build_anomaly_target_frame(anomaly_df, target, fx_paths["actual"])
        data = actual_frame[["target_change", "date", "concat_step", "target_level", *feature_cols]].dropna()
        if len(data) < 36:
            continue
        target_test_obs = choose_anomaly_test_obs(len(data), test_obs)
        if len(data) - target_test_obs < 24:
            continue
        train = data.iloc[:-target_test_obs]
        test = data.iloc[-target_test_obs:]
        models: list[tuple[str, Any, StandardScaler | None]] = []

        scaler = StandardScaler()
        ridge = Ridge(alpha=1.0)
        ridge.fit(scaler.fit_transform(train[feature_cols]), train["target_change"])
        models.append(("RIDGE_DLM", ridge, scaler))

        tree = ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        tree.fit(train[feature_cols], train["target_change"])
        models.append(("TREE_DLM", tree, None))

        start_level = float(train["target_level"].iloc[-1])
        for model_name, model, fitted_scaler in models:
            for mode, fx_path in fx_paths.items():
                mode_frame, _ = build_anomaly_target_frame(anomaly_df, target, fx_path)
                y_history = actual_frame["target_change"].loc[: train.index[-1]].dropna().to_dict()
                forecasts = []
                for idx, row in test.iterrows():
                    feature_values = {"target_lag1": y_history.get(idx - 1, np.nan)}
                    for lag in range(1, MAX_LAG + 1):
                        feature_values[f"fx_lag{lag}"] = mode_frame.at[idx, f"fx_lag{lag}"] if idx in mode_frame.index else np.nan
                    x_row = pd.DataFrame([feature_values], columns=feature_cols)
                    if x_row.isna().any().any():
                        pred = np.nan
                    elif fitted_scaler is not None:
                        pred = float(model.predict(fitted_scaler.transform(x_row))[0])
                    else:
                        pred = float(model.predict(x_row)[0])
                    forecasts.append(pred)
                    y_history[idx] = pred
                forecast = np.asarray(forecasts, dtype=float)
                forecast_level = invert_transformed_forecast(start_level, forecast, transform.transform)
                for horizon, (idx, row) in enumerate(test.iterrows(), start=1):
                    records.append(
                        {
                            "date": row["date"],
                            "concat_step": int(row["concat_step"]),
                            "horizon": horizon,
                            "target": target,
                            "model": model_name,
                            "fx_mode": mode,
                            "lag_used": np.nan,
                            "transform": transform.transform,
                            "unit_label": transform.unit_label,
                            "forecast_transform": forecast[horizon - 1],
                            "actual_transform": row["target_change"],
                            "forecast_level": forecast_level[horizon - 1],
                            "actual_level": row["target_level"],
                            "sample_scope": "anomaly_concatenated",
                        }
                    )
    return records


def plot_anomaly_forecasts(forecast_df: pd.DataFrame, selected_targets: list[str]) -> None:
    if forecast_df.empty:
        return
    for old_plot in (FINAL_REPORT_DIR / "plots").glob("forecast_*.png"):
        old_plot.unlink()
    for old_plot in (ANOMALY_SET_REPORT_DIR / "plots").glob("anomaly_forecast_*.png"):
        old_plot.unlink()

    plot_selection = select_plot_models_by_target(forecast_df, selected_targets)
    if not plot_selection.empty:
        plot_selection.to_csv(ANOMALY_SET_REPORT_DIR / "plot_model_selection.csv", index=False)
        plot_selection.to_csv(FINAL_REPORT_DIR / "plot_model_selection.csv", index=False)
    selected_model_by_target = (
        plot_selection[plot_selection["selected_for_plot"]].set_index("target")["model"].to_dict()
        if not plot_selection.empty
        else {}
    )

    for target in selected_targets:
        preferred_model = selected_model_by_target.get(target, "RIDGE_DLM")
        part = forecast_df[(forecast_df["target"].eq(target)) & (forecast_df["model"].eq(preferred_model))].copy()
        if part.empty:
            continue
        model_label = part["model"].iloc[0]
        fig, ax = plt.subplots(figsize=(11, 5.5))
        actual = part.drop_duplicates("concat_step").sort_values("concat_step")
        ax.plot(
            actual["concat_step"],
            actual["actual_level"],
            color="black",
            linewidth=1.8,
            marker="o",
            label="Actual target",
        )
        for mode, style in [("actual", "--"), ("predicted", ":"), ("scenario", "-.")]:
            mode_part = part[part["fx_mode"].eq(mode)].sort_values("concat_step")
            if mode_part.empty:
                continue
            ax.plot(
                mode_part["concat_step"],
                mode_part["forecast_level"],
                linestyle=style,
                linewidth=1.5,
                marker="x",
                label=f"{model_label} with {mode} FX",
            )
        ax.set_title(f"Anomaly-set forecast: {target} ({model_label}, concatenated abnormal months)")
        ax.set_xlabel("Concatenated anomaly step")
        ax.set_ylabel("Level")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        out_name = f"forecast_{safe_filename(target)}.png"
        fig.savefig(FINAL_REPORT_DIR / "plots" / out_name, dpi=140)
        fig.savefig(ANOMALY_SET_REPORT_DIR / "plots" / f"anomaly_{out_name}", dpi=140)
        plt.close(fig)


def run_anomaly_concat_analysis(
    macro_df: pd.DataFrame,
    ranking: pd.DataFrame,
    selected_predictions: pd.DataFrame,
    selected_source: str,
    scenario_shock_pct: float,
    test_obs: int = TEST_OBS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_targets = ranking[ranking["selected_for_final_model"]]["target"].tolist()
    lag_map = get_target_lag_map(ranking, selected_targets)
    anomaly_df = build_anomaly_concat_macro(macro_df)
    fx_paths = build_anomaly_fx_paths(anomaly_df, selected_predictions, scenario_shock_pct, test_obs=test_obs)

    model_panel = build_anomaly_model_panel(anomaly_df, selected_targets, fx_paths)
    model_panel.to_csv(ANOMALY_SET_REPORT_DIR / "anomaly_model_panel.csv", index=False)

    records = []
    records.extend(run_anomaly_arimax_models(anomaly_df, selected_targets, lag_map, fx_paths, test_obs=test_obs))
    records.extend(run_anomaly_dlm_models(anomaly_df, selected_targets, fx_paths, test_obs=test_obs))
    forecast_df = pd.DataFrame(records)
    if not forecast_df.empty:
        forecast_df["source_model"] = selected_source
    forecast_df.to_csv(ANOMALY_SET_REPORT_DIR / "anomaly_impact_forecasts.csv", index=False)

    model_comparison = evaluate_prediction_records(records)
    model_comparison.to_csv(ANOMALY_SET_REPORT_DIR / "anomaly_model_comparison.csv", index=False)

    scenario_df = build_scenario_forecasts(forecast_df)
    scenario_df.to_csv(ANOMALY_SET_REPORT_DIR / "anomaly_scenario_forecasts.csv", index=False)

    lag_summary = build_lag_effect_summary(ranking, selected_targets, scenario_df)
    lag_summary.to_csv(ANOMALY_SET_REPORT_DIR / "anomaly_lag_effect_summary.csv", index=False)

    plot_anomaly_forecasts(forecast_df, selected_targets)
    return model_comparison, forecast_df, lag_summary, scenario_df


def lp_feature_columns() -> list[str]:
    return ["fx_shock_t"] + [f"fx_lag{lag}" for lag in range(1, MAX_LAG + 1)] + ["target_lag1"]


def response_to_level_delta(start_level: float, response: float, transform: str) -> float:
    if not np.isfinite(start_level) or not np.isfinite(response):
        return np.nan
    if transform == "log_diff":
        if start_level <= 0:
            return np.nan
        return float(start_level * (np.exp(response) - 1.0))
    return float(response)


def build_event_fx_feature_frame(
    macro_df: pd.DataFrame,
    selected_predictions: pd.DataFrame,
    scenario_shock_pct: float,
) -> pd.DataFrame:
    idx = pd.to_datetime(macro_df["Date"]) + pd.offsets.MonthEnd(0)
    actual_fx = pd.Series(pd.to_numeric(macro_df["USD_KRW"], errors="coerce").to_numpy(dtype=float), index=idx)
    predicted_fx = monthly_prediction_path(macro_df, selected_predictions).reindex(idx)

    actual_fx_change = transform_series(actual_fx, "log_diff")
    pred_fx_change = transform_series(predicted_fx, "log_diff")
    scenario_fx = predicted_fx * (1.0 + scenario_shock_pct)
    scenario_fx_change = np.log(scenario_fx.where(scenario_fx > 0)) - np.log(predicted_fx.shift(1).where(predicted_fx.shift(1) > 0))

    frame = pd.DataFrame(
        {
            "actual_fx_t": actual_fx,
            "pred_fx_t": predicted_fx,
            "scenario_fx_t": scenario_fx,
            "actual_fx_change_t": actual_fx_change,
            "pred_fx_change_t": pred_fx_change,
            "scenario_fx_change_t": scenario_fx_change,
        },
        index=idx,
    )
    for lag in range(1, MAX_LAG + 1):
        frame[f"fx_lag{lag}_actual"] = actual_fx_change.shift(lag)
        frame[f"fx_lag{lag}_predicted"] = pred_fx_change.shift(lag)
        frame[f"fx_lag{lag}_scenario"] = pred_fx_change.shift(lag)
    return frame


def build_event_time_panel(
    macro_df: pd.DataFrame,
    selected_targets: list[str],
    selected_predictions: pd.DataFrame,
    selected_source: str,
    scenario_shock_pct: float,
    horizons: tuple[int, ...] = LP_HORIZONS,
) -> pd.DataFrame:
    macro = macro_df.copy()
    macro["Date"] = pd.to_datetime(macro["Date"]) + pd.offsets.MonthEnd(0)
    macro = macro.sort_values("Date").drop_duplicates("Date").set_index("Date")
    fx_features = build_event_fx_feature_frame(macro.reset_index(), selected_predictions, scenario_shock_pct)
    anomaly_dates = macro.index[macro["Is_Abnormal_Period"].eq(1)]

    rows: list[dict[str, Any]] = []
    for target in selected_targets:
        if target not in macro.columns:
            continue
        target_level = pd.to_numeric(macro[target], errors="coerce")
        transform = choose_target_transform(target_level, target)
        target_change = transform_series(target_level, transform.transform)
        target_lag1 = target_change.shift(1)
        unit_label = "log_change_horizon" if transform.transform == "log_diff" else "unit_delta_horizon"

        for event_date in anomaly_dates:
            if event_date not in fx_features.index:
                continue
            target_t = target_level.get(event_date, np.nan)
            if not np.isfinite(target_t):
                continue
            feature_row = fx_features.loc[event_date]
            base_values = {
                "target_lag1": target_lag1.get(event_date, np.nan),
                "target_change_t": target_change.get(event_date, np.nan),
                **feature_row.to_dict(),
            }
            required_base = [
                "target_lag1",
                "target_change_t",
                "actual_fx_t",
                "pred_fx_t",
                "scenario_fx_t",
                "actual_fx_change_t",
                "pred_fx_change_t",
                "scenario_fx_change_t",
                *[f"fx_lag{lag}_actual" for lag in range(1, MAX_LAG + 1)],
                *[f"fx_lag{lag}_predicted" for lag in range(1, MAX_LAG + 1)],
                *[f"fx_lag{lag}_scenario" for lag in range(1, MAX_LAG + 1)],
            ]
            if any(not np.isfinite(base_values.get(col, np.nan)) for col in required_base):
                continue

            for horizon in horizons:
                response_date = event_date + pd.offsets.MonthEnd(horizon)
                if response_date not in macro.index:
                    continue
                target_future = target_level.get(response_date, np.nan)
                if not np.isfinite(target_future):
                    continue
                if transform.transform == "log_diff":
                    if target_t <= 0 or target_future <= 0:
                        continue
                    actual_response = float(np.log(target_future) - np.log(target_t))
                else:
                    actual_response = float(target_future - target_t)
                level_delta = float(target_future - target_t)
                rows.append(
                    {
                        "event_date": event_date,
                        "response_date": response_date,
                        "target": target,
                        "horizon": int(horizon),
                        "actual_fx_t": base_values["actual_fx_t"],
                        "pred_fx_t": base_values["pred_fx_t"],
                        "scenario_fx_t": base_values["scenario_fx_t"],
                        "actual_fx_change_t": base_values["actual_fx_change_t"],
                        "pred_fx_change_t": base_values["pred_fx_change_t"],
                        "scenario_fx_change_t": base_values["scenario_fx_change_t"],
                        **{f"fx_lag{lag}_actual": base_values[f"fx_lag{lag}_actual"] for lag in range(1, MAX_LAG + 1)},
                        **{
                            f"fx_lag{lag}_predicted": base_values[f"fx_lag{lag}_predicted"]
                            for lag in range(1, MAX_LAG + 1)
                        },
                        **{
                            f"fx_lag{lag}_scenario": base_values[f"fx_lag{lag}_scenario"]
                            for lag in range(1, MAX_LAG + 1)
                        },
                        "target_level_t": float(target_t),
                        "target_level_t_plus_h": float(target_future),
                        "target_lag1": base_values["target_lag1"],
                        "target_change_t": base_values["target_change_t"],
                        "actual_response": actual_response,
                        "actual_response_level_delta": level_delta,
                        "transform": transform.transform,
                        "unit_label": unit_label,
                        "is_abnormal_period": int(macro.at[event_date, "Is_Abnormal_Period"]),
                        "source_model": selected_source,
                    }
                )

    panel = pd.DataFrame(rows)
    if panel.empty:
        return panel
    return panel.sort_values(["target", "horizon", "event_date"]).reset_index(drop=True)


def build_lp_features(panel: pd.DataFrame, fx_mode: str) -> pd.DataFrame:
    mode_change_cols = {
        "actual": "actual_fx_change_t",
        "predicted": "pred_fx_change_t",
        "scenario": "scenario_fx_change_t",
    }
    if fx_mode not in mode_change_cols:
        raise ValueError(f"Unsupported LP FX mode: {fx_mode}")
    out = pd.DataFrame(index=panel.index)
    out["fx_shock_t"] = pd.to_numeric(panel[mode_change_cols[fx_mode]], errors="coerce")
    for lag in range(1, MAX_LAG + 1):
        out[f"fx_lag{lag}"] = pd.to_numeric(panel[f"fx_lag{lag}_{fx_mode}"], errors="coerce")
    out["target_lag1"] = pd.to_numeric(panel["target_lag1"], errors="coerce")
    return out[lp_feature_columns()]


def split_lp_train_test(group: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = pd.Series(pd.to_datetime(group["event_date"]).unique()).sort_values().to_list()
    feature_count = len(lp_feature_columns())
    min_train_events = max(feature_count + 8, 20)
    if len(unique_dates) < min_train_events + 3:
        return pd.DataFrame(), pd.DataFrame()
    test_events = max(3, int(math.ceil(len(unique_dates) * LP_TEST_FRACTION)))
    if len(unique_dates) - test_events < min_train_events:
        test_events = len(unique_dates) - min_train_events
    if test_events < 3:
        return pd.DataFrame(), pd.DataFrame()
    test_start = unique_dates[-test_events]
    train = group[pd.to_datetime(group["event_date"]).lt(test_start)].copy()
    test = group[pd.to_datetime(group["event_date"]).ge(test_start)].copy()
    return train, test


def original_scale_linear_coefficients(
    model: Ridge | ElasticNet | ElasticNetCV,
    scaler: StandardScaler,
    feature_cols: list[str],
) -> dict[str, float]:
    scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    coefficients = np.asarray(model.coef_, dtype=float) / scale
    intercept = float(model.intercept_ - np.sum(np.asarray(model.coef_, dtype=float) * scaler.mean_ / scale))
    out = {"intercept": intercept}
    out.update({feature: float(value) for feature, value in zip(feature_cols, coefficients)})
    return out


def fit_local_projection_estimators(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    horizon: int,
) -> list[dict[str, Any]]:
    feature_cols = lp_feature_columns()
    estimators: list[dict[str, Any]] = []
    if len(train_x) <= len(feature_cols) + 2:
        return estimators

    x_ols = add_constant(train_x[feature_cols], has_constant="add")
    try:
        cov_lags = max(1, min(int(horizon), len(train_x) // 4))
        ols_model = OLS(train_y, x_ols).fit(cov_type="HAC", cov_kwds={"maxlags": cov_lags})
        coefficients = {"intercept": float(ols_model.params.get("const", np.nan))}
        coefficients.update({feature: float(ols_model.params.get(feature, np.nan)) for feature in feature_cols})
        pvalues = {"intercept": float(ols_model.pvalues.get("const", np.nan))}
        pvalues.update({feature: float(ols_model.pvalues.get(feature, np.nan)) for feature in feature_cols})
        estimators.append(
            {
                "name": "LocalProjection_OLS",
                "kind": "ols",
                "model": ols_model,
                "scaler": None,
                "coefficients": coefficients,
                "pvalues": pvalues,
            }
        )
    except Exception:
        pass

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(train_x[feature_cols])

    ridge = Ridge(alpha=1.0)
    ridge.fit(x_scaled, train_y)
    estimators.append(
        {
            "name": "LocalProjection_Ridge",
            "kind": "scaled_linear",
            "model": ridge,
            "scaler": scaler,
            "coefficients": original_scale_linear_coefficients(ridge, scaler, feature_cols),
            "pvalues": {feature: np.nan for feature in ["intercept", *feature_cols]},
        }
    )

    try:
        if len(train_x) >= 30:
            n_splits = min(5, max(2, len(train_x) // 12))
            elastic = ElasticNetCV(
                l1_ratio=[0.1, 0.3, 0.5, 0.7],
                alphas=np.logspace(-5, 1, 40),
                cv=TimeSeriesSplit(n_splits=n_splits),
                random_state=RANDOM_STATE,
                max_iter=30000,
            )
        else:
            elastic = ElasticNet(alpha=0.01, l1_ratio=0.3, random_state=RANDOM_STATE, max_iter=30000)
        elastic.fit(x_scaled, train_y)
        estimators.append(
            {
                "name": "LocalProjection_ElasticNet",
                "kind": "scaled_linear",
                "model": elastic,
                "scaler": scaler,
                "coefficients": original_scale_linear_coefficients(elastic, scaler, feature_cols),
                "pvalues": {feature: np.nan for feature in ["intercept", *feature_cols]},
            }
        )
    except Exception:
        pass

    return estimators


def predict_local_projection(estimator: dict[str, Any], x: pd.DataFrame) -> np.ndarray:
    feature_cols = lp_feature_columns()
    if estimator["kind"] == "ols":
        x_model = add_constant(x[feature_cols], has_constant="add")
        return np.asarray(estimator["model"].predict(x_model), dtype=float)
    scaler = estimator["scaler"]
    return np.asarray(estimator["model"].predict(scaler.transform(x[feature_cols])), dtype=float)


def run_event_time_local_projection_models(
    event_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coefficient_rows: list[dict[str, Any]] = []
    forecast_rows: list[dict[str, Any]] = []
    if event_panel.empty:
        return pd.DataFrame(coefficient_rows), pd.DataFrame(forecast_rows)

    for (target, horizon), group in event_panel.groupby(["target", "horizon"], sort=True):
        group = group.sort_values("event_date").reset_index(drop=True)
        train, test = split_lp_train_test(group)
        if train.empty or test.empty:
            continue

        train_x = build_lp_features(train, "actual")
        train_y = pd.to_numeric(train["actual_response"], errors="coerce")
        valid_train = train_x.notna().all(axis=1) & train_y.notna()
        train_x = train_x.loc[valid_train]
        train_y = train_y.loc[valid_train]
        if len(train_x) <= len(lp_feature_columns()) + 2:
            continue

        estimators = fit_local_projection_estimators(train_x, train_y, int(horizon))
        test_rows = len(test)
        for estimator in estimators:
            for feature, coefficient in estimator["coefficients"].items():
                coefficient_rows.append(
                    {
                        "target": target,
                        "horizon": int(horizon),
                        "model": estimator["name"],
                        "feature": feature,
                        "coefficient": coefficient,
                        "pvalue": estimator["pvalues"].get(feature, np.nan),
                        "train_rows": int(len(train_x)),
                        "test_rows": int(test_rows),
                    }
                )

            for fx_mode in ["actual", "predicted", "scenario"]:
                test_x = build_lp_features(test, fx_mode)
                valid_test = test_x.notna().all(axis=1) & pd.to_numeric(test["actual_response"], errors="coerce").notna()
                if not valid_test.any():
                    continue
                test_valid = test.loc[valid_test].copy()
                x_valid = test_x.loc[valid_test]
                predictions = predict_local_projection(estimator, x_valid)
                for row, prediction in zip(test_valid.to_dict("records"), predictions):
                    predicted_level_delta = response_to_level_delta(
                        float(row["target_level_t"]),
                        float(prediction),
                        row["transform"],
                    )
                    forecast_rows.append(
                        {
                            "event_date": row["event_date"],
                            "response_date": row["response_date"],
                            "target": target,
                            "horizon": int(horizon),
                            "model": estimator["name"],
                            "fx_mode": fx_mode,
                            "actual_response": row["actual_response"],
                            "predicted_response": float(prediction),
                            "actual_level_delta": row["actual_response_level_delta"],
                            "predicted_level_delta": predicted_level_delta,
                            "target_level_t": row["target_level_t"],
                            "transform": row["transform"],
                            "unit_label": row["unit_label"],
                            "source_model": row["source_model"],
                            "train_rows": int(len(train_x)),
                            "test_rows": int(test_rows),
                        }
                    )

    return pd.DataFrame(coefficient_rows), pd.DataFrame(forecast_rows)


def build_event_lp_metrics(forecast_df: pd.DataFrame) -> pd.DataFrame:
    if forecast_df.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in forecast_df.groupby(["target", "horizon", "model", "fx_mode", "transform", "unit_label"], dropna=False):
        target, horizon, model, fx_mode, transform, unit_label = keys
        clean = group.dropna(
            subset=["actual_response", "predicted_response", "actual_level_delta", "predicted_level_delta"]
        ).copy()
        if clean.empty:
            continue
        rmse_response = float(np.sqrt(mean_squared_error(clean["actual_response"], clean["predicted_response"])))
        rmse_level = float(np.sqrt(mean_squared_error(clean["actual_level_delta"], clean["predicted_level_delta"])))
        response_std = max(float(clean["actual_response"].std(ddof=0)), 1e-12)
        rows.append(
            {
                "target": target,
                "horizon": int(horizon),
                "model": model,
                "fx_mode": fx_mode,
                "transform": transform,
                "unit_label": unit_label,
                "rmse_response": rmse_response,
                "mae_response": float(mean_absolute_error(clean["actual_response"], clean["predicted_response"])),
                "nrmse_response": rmse_response / response_std,
                "rmse_level_delta": rmse_level,
                "mae_level_delta": float(mean_absolute_error(clean["actual_level_delta"], clean["predicted_level_delta"])),
                "test_rows": int(len(clean)),
            }
        )
    return pd.DataFrame(rows)


def build_event_scenario_response_forecasts(forecast_df: pd.DataFrame, scenario_shock_pct: float) -> pd.DataFrame:
    if forecast_df.empty:
        return pd.DataFrame()
    key_cols = ["event_date", "response_date", "target", "horizon", "model", "transform", "unit_label", "source_model"]
    baseline = forecast_df[forecast_df["fx_mode"].eq("predicted")].copy()
    scenario = forecast_df[forecast_df["fx_mode"].eq("scenario")].copy()
    if baseline.empty or scenario.empty:
        return pd.DataFrame()
    merged = scenario.merge(
        baseline[key_cols + ["actual_response", "predicted_response", "actual_level_delta", "predicted_level_delta"]],
        on=key_cols,
        how="inner",
        suffixes=("_scenario", "_baseline"),
    )
    merged["scenario_shock_pct"] = scenario_shock_pct
    merged["response_delta_vs_predicted"] = merged["predicted_response_scenario"] - merged["predicted_response_baseline"]
    merged["level_delta_vs_predicted"] = merged["predicted_level_delta_scenario"] - merged["predicted_level_delta_baseline"]
    return merged[
        [
            *key_cols,
            "scenario_shock_pct",
            "actual_response_baseline",
            "actual_level_delta_baseline",
            "predicted_response_baseline",
            "predicted_response_scenario",
            "response_delta_vs_predicted",
            "predicted_level_delta_baseline",
            "predicted_level_delta_scenario",
            "level_delta_vs_predicted",
        ]
    ].rename(
        columns={
            "actual_response_baseline": "actual_response",
            "actual_level_delta_baseline": "actual_level_delta",
        }
    )


def select_event_panel_models(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    predicted = metrics[metrics["fx_mode"].eq("predicted")].copy()
    if predicted.empty:
        return pd.DataFrame()

    target_rows = (
        predicted.groupby(["target", "model"], as_index=False)
        .agg(
            avg_rmse_response=("rmse_response", "mean"),
            avg_nrmse_response=("nrmse_response", "mean"),
            avg_rmse_level_delta=("rmse_level_delta", "mean"),
            total_test_rows=("test_rows", "sum"),
            horizons=("horizon", "nunique"),
        )
        .assign(selection_scope="target")
    )
    target_rows["rank"] = target_rows.groupby("target")["avg_nrmse_response"].rank(method="first")
    target_rows["selected_for_plot"] = target_rows["rank"].eq(1)

    overall_rows = (
        predicted.groupby("model", as_index=False)
        .agg(
            avg_rmse_response=("rmse_response", "mean"),
            avg_nrmse_response=("nrmse_response", "mean"),
            avg_rmse_level_delta=("rmse_level_delta", "mean"),
            total_test_rows=("test_rows", "sum"),
            horizons=("horizon", "nunique"),
            targets=("target", "nunique"),
        )
        .assign(target="__ALL__", selection_scope="overall")
    )
    overall_rows["rank"] = overall_rows["avg_nrmse_response"].rank(method="first")
    overall_rows["selected_for_plot"] = overall_rows["rank"].eq(1)
    if "targets" not in target_rows.columns:
        target_rows["targets"] = 1
    return pd.concat([target_rows, overall_rows], axis=0, ignore_index=True).sort_values(
        ["selection_scope", "target", "rank"]
    )


def plot_event_response_curves(
    forecast_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    model_selection: pd.DataFrame,
    selected_targets: list[str],
    scenario_shock_pct: float,
) -> None:
    if forecast_df.empty or model_selection.empty:
        return

    selected_models = (
        model_selection[
            model_selection["selection_scope"].eq("target")
            & model_selection["selected_for_plot"].eq(True)
            & ~model_selection["target"].eq("__ALL__")
        ]
        .set_index("target")["model"]
        .to_dict()
    )
    mode_labels = {
        "actual": "LP prediction using actual FX",
        "predicted": "LP prediction using hybrid FX",
        "scenario": f"LP prediction using +{scenario_shock_pct * 100:.1f}% FX scenario",
    }
    mode_styles = {
        "actual": {"color": "#4c78a8", "linestyle": "--", "marker": "s"},
        "predicted": {"color": "#f58518", "linestyle": ":", "marker": "^"},
        "scenario": {"color": "#54a24b", "linestyle": "-.", "marker": "D"},
    }

    for target in selected_targets:
        model = selected_models.get(target)
        if model is None:
            continue
        part = forecast_df[forecast_df["target"].eq(target) & forecast_df["model"].eq(model)].copy()
        if part.empty:
            continue
        transform = part["transform"].iloc[0]
        unit_label = part["unit_label"].iloc[0]
        unit_description = (
            "cumulative log change over h months"
            if transform == "log_diff"
            else "level delta over h months"
        )
        test_event_count = part["event_date"].nunique()

        actual = part[["event_date", "horizon", "actual_response"]].drop_duplicates()
        actual_summary = actual.groupby("horizon")["actual_response"].agg(["mean", lambda x: x.quantile(0.10), lambda x: x.quantile(0.90)])
        actual_summary.columns = ["mean", "p10", "p90"]

        fig, ax = plt.subplots(figsize=(9.5, 5.6))
        horizons = np.array(LP_HORIZONS, dtype=int)
        actual_summary = actual_summary.reindex(horizons)
        ax.fill_between(
            horizons,
            actual_summary["p10"].to_numpy(dtype=float),
            actual_summary["p90"].to_numpy(dtype=float),
            color="#b8b8b8",
            alpha=0.25,
            label="Actual test event P10-P90",
        )
        ax.plot(
            horizons,
            actual_summary["mean"].to_numpy(dtype=float),
            color="black",
            linewidth=2.0,
            marker="o",
            label="Actual mean response on test events",
        )

        for mode in ["actual", "predicted", "scenario"]:
            mode_summary = part[part["fx_mode"].eq(mode)].groupby("horizon")["predicted_response"].mean().reindex(horizons)
            if mode_summary.dropna().empty:
                continue
            ax.plot(
                horizons,
                mode_summary.to_numpy(dtype=float),
                linewidth=1.8,
                label=mode_labels[mode],
                **mode_styles[mode],
            )

        ax.axhline(0, color="#333333", linewidth=0.9)
        ax.set_title(f"{target}: event-time local projection ({unit_label}, {model}, n={test_event_count} test events)")
        ax.set_xlabel("Horizon after anomaly event month")
        ax.set_ylabel(unit_description)
        ax.set_xticks(horizons)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        out_name = f"response_{safe_filename(target)}.png"
        fig.savefig(EVENT_PANEL_PLOT_DIR / out_name, dpi=150)
        fig.savefig(FINAL_REPORT_DIR / "plots" / out_name, dpi=150)
        plt.close(fig)

    if scenario_df.empty:
        return
    peak_rows = []
    for target in selected_targets:
        model = selected_models.get(target)
        part = scenario_df[scenario_df["target"].eq(target) & scenario_df["model"].eq(model)].copy()
        if part.empty:
            continue
        peak = part.assign(abs_delta=part["response_delta_vs_predicted"].abs()).sort_values("abs_delta", ascending=False).iloc[0]
        peak_rows.append(
            {
                "target": target,
                "model": model,
                "horizon": int(peak["horizon"]),
                "peak_response_delta": float(peak["response_delta_vs_predicted"]),
            }
        )
    if not peak_rows:
        return
    peak_df = pd.DataFrame(peak_rows).sort_values("peak_response_delta")
    colors = np.where(peak_df["peak_response_delta"].ge(0), "#54a24b", "#e45756")
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.barh(peak_df["target"], peak_df["peak_response_delta"], color=colors)
    for y_pos, row in enumerate(peak_df.to_dict("records")):
        offset = 0.01 * max(peak_df["peak_response_delta"].abs().max(), 1e-12)
        x_pos = row["peak_response_delta"] + (offset if row["peak_response_delta"] >= 0 else -offset)
        ha = "left" if row["peak_response_delta"] >= 0 else "right"
        ax.text(x_pos, y_pos, f"h={row['horizon']}", va="center", ha=ha, fontsize=8)
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.set_title(f"Scenario-baseline peak response delta by target (+{scenario_shock_pct * 100:.1f}% USD/KRW at event month)")
    ax.set_xlabel("Peak response delta vs hybrid FX baseline")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(EVENT_PANEL_PLOT_DIR / "response_summary_top_targets.png", dpi=150)
    fig.savefig(FINAL_REPORT_DIR / "plots" / "response_summary_top_targets.png", dpi=150)
    plt.close(fig)


def run_event_time_local_projection_analysis(
    macro_df: pd.DataFrame,
    ranking: pd.DataFrame,
    selected_predictions: pd.DataFrame,
    selected_source: str,
    scenario_shock_pct: float,
) -> dict[str, pd.DataFrame]:
    selected_targets = ranking[ranking["selected_for_final_model"]]["target"].tolist()
    event_panel = build_event_time_panel(
        macro_df,
        selected_targets,
        selected_predictions,
        selected_source,
        scenario_shock_pct=scenario_shock_pct,
    )
    event_panel.to_csv(EVENT_PANEL_REPORT_DIR / "anomaly_event_panel.csv", index=False)

    coefficients, forecasts = run_event_time_local_projection_models(event_panel)
    coefficients.to_csv(EVENT_PANEL_REPORT_DIR / "local_projection_coefficients.csv", index=False)
    forecasts.to_csv(EVENT_PANEL_REPORT_DIR / "event_response_forecasts.csv", index=False)

    metrics = build_event_lp_metrics(forecasts)
    metrics.to_csv(EVENT_PANEL_REPORT_DIR / "local_projection_metrics.csv", index=False)

    scenario = build_event_scenario_response_forecasts(forecasts, scenario_shock_pct)
    scenario.to_csv(EVENT_PANEL_REPORT_DIR / "scenario_response_forecasts.csv", index=False)

    model_selection = select_event_panel_models(metrics)
    model_selection.to_csv(EVENT_PANEL_REPORT_DIR / "event_panel_model_selection.csv", index=False)

    plot_event_response_curves(forecasts, scenario, model_selection, selected_targets, scenario_shock_pct)
    return {
        "event_panel": event_panel,
        "coefficients": coefficients,
        "forecasts": forecasts,
        "metrics": metrics,
        "scenario": scenario,
        "model_selection": model_selection,
    }


def run_final_impact_models(
    macro_df: pd.DataFrame,
    ranking: pd.DataFrame,
    selected_predictions: pd.DataFrame,
    selected_source: str,
    scenario_shock_pct: float,
    custom_shock_path: Path | None = None,
    test_obs: int = TEST_OBS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_targets = ranking[ranking["selected_for_final_model"]]["target"].tolist()
    lag_map = get_target_lag_map(ranking, selected_targets)

    test_start = pd.to_datetime(macro_df["Date"]).iloc[-test_obs]
    fx_paths = {
        "actual": build_fx_level_path(macro_df, selected_predictions, "actual"),
        "predicted": build_fx_level_path(macro_df, selected_predictions, "predicted"),
        "scenario": build_fx_level_path(
            macro_df,
            selected_predictions,
            "scenario",
            scenario_shock_pct=scenario_shock_pct,
            scenario_start=test_start,
            custom_shock_path=custom_shock_path,
        ),
    }

    records = []
    records.extend(run_arimax_models(macro_df, selected_targets, lag_map, fx_paths, test_obs=test_obs))
    records.extend(run_dlm_models(macro_df, selected_targets, fx_paths, test_obs=test_obs))
    records.extend(run_varx_model(macro_df, selected_targets, fx_paths, test_obs=test_obs))
    forecast_df = pd.DataFrame(records)
    if not forecast_df.empty:
        forecast_df["source_model"] = selected_source
        forecast_df.to_csv(FINAL_REPORT_DIR / "impact_forecasts.csv", index=False)
    else:
        forecast_df.to_csv(FINAL_REPORT_DIR / "impact_forecasts.csv", index=False)

    model_comparison = evaluate_prediction_records(records)
    model_comparison.to_csv(FINAL_REPORT_DIR / "model_comparison.csv", index=False)

    scenario_df = build_scenario_forecasts(forecast_df)
    scenario_df.to_csv(FINAL_REPORT_DIR / "scenario_forecasts.csv", index=False)

    lag_summary = build_lag_effect_summary(ranking, selected_targets, scenario_df)
    lag_summary.to_csv(FINAL_REPORT_DIR / "lag_effect_summary.csv", index=False)

    # Calendar-time comparison tables are preserved here. The final result.md and
    # user-facing plots are written by the event-time local projection layer.
    return model_comparison, forecast_df, lag_summary, scenario_df


def format_float(value: Any, digits: int = 4) -> str:
    try:
        if value is None or not np.isfinite(float(value)):
            return "n/a"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "n/a"


def write_final_result_md(
    ranking: pd.DataFrame,
    selected_source: str,
    model_comparison: pd.DataFrame,
    lag_summary: pd.DataFrame,
    scenario_shock_pct: float,
    anomaly_model_comparison: pd.DataFrame | None = None,
    anomaly_lag_summary: pd.DataFrame | None = None,
) -> None:
    selected = ranking[ranking["selected_for_final_model"]].copy()
    overall = model_comparison[model_comparison["target"].eq("__ALL__")].copy() if not model_comparison.empty else pd.DataFrame()
    overall_pred = overall[overall["fx_mode"].eq("predicted")].sort_values("avg_nrmse_transform") if not overall.empty else pd.DataFrame()
    best_model = overall_pred.iloc[0]["model"] if not overall_pred.empty else "n/a"

    lines = [
        "# FX Impact Final Pipeline Result",
        "",
        "## Purpose",
        "",
        "This final pipeline estimates how domestic macro/financial variables respond with lags when USD/KRW follows an actual, predicted, or scenario shock path during anomaly-style conditions.",
        "",
        "## Reproduction Commands",
        "",
        "```bash",
        "python analysis/fx_impact/run_final_fx_impact_pipeline.py",
        "python analysis/fx_impact/lead_lag_causality_analysis.py",
        "python analysis/fx_impact/predict_arimax.py",
        "python analysis/fx_impact/predict_varx.py",
        "```",
        "",
        "## Target Selection",
        "",
        "Targets are selected from source level columns only. Derived `_MoM`, `_YoY`, and `_lag` columns are excluded as target candidates to avoid duplicate targets and leakage.",
        "",
    ]
    for row in selected.to_dict("records"):
        lag = row.get("full_ardl_best_lag")
        if not np.isfinite(lag):
            lag = row.get("full_ccf_best_lag")
        direction = "positive" if row.get("full_ardl_coef", 0) > 0 else "negative"
        lines.append(
            f"- `{row['target']}`: lag={int(lag) if np.isfinite(lag) else 'n/a'} months, "
            f"direction={direction}, score={format_float(row['composite_score'], 3)}, "
            f"transform={row['full_transform']}"
        )

    lines.extend(
        [
            "",
            "## Selected FX Input",
            "",
            f"- Selected FX source: `{selected_source}`",
            "- Daily hybrid prediction outputs are resampled to month-end using monthly means. Months without an available hybrid prediction retain actual USD/KRW in the predicted path, so the comparison isolates error propagation where predictions exist.",
            "",
            "## Final Impact Models",
            "",
            "- `ARIMAX`: per-target SARIMAX on transformed target with target-specific selected USD/KRW lag.",
            "- `RIDGE_DLM`: recursive regularized distributed-lag model using target lag 1 and USD/KRW lags 1-6.",
            "- `TREE_DLM`: recursive ExtraTrees distributed-lag model using the same lag structure.",
            "- `VARX`: multivariate VAR with USD/KRW lags 1-6 as exogenous inputs for the top selected targets.",
            "- Calendar-time forecast tables are still saved for comparison, but final forecast plots now use the concatenated anomaly-month set.",
            "- In the anomaly-set run, `Is_Abnormal_Period == 1` months are filtered first and then treated as one stitched sequence; lag distortion across gaps is intentionally ignored for this prototype pass.",
            "",
            f"- Scenario path: persistent `{scenario_shock_pct * 100:.1f}%` upward shock to the selected predicted FX level from the first test month.",
            f"- Best predicted-FX model by average transformed NRMSE: `{best_model}`",
            "",
            "## Model Comparison",
            "",
        ]
    )
    if not overall.empty:
        for row in overall.sort_values(["fx_mode", "avg_rmse_transform"]).to_dict("records"):
            lines.append(
                f"- `{row['model']}` with `{row['fx_mode']}` FX: "
                f"avg transform RMSE={format_float(row['avg_rmse_transform'], 6)}, "
                f"avg transform NRMSE={format_float(row['avg_nrmse_transform'], 4)}, "
                f"avg level RMSE={format_float(row['avg_rmse_level'], 4)}, "
                f"targets={int(row['targets'])}"
            )

    if anomaly_model_comparison is not None and not anomaly_model_comparison.empty:
        anomaly_overall = anomaly_model_comparison[anomaly_model_comparison["target"].eq("__ALL__")].copy()
        lines.extend(["", "## Concatenated Anomaly-Set Model Comparison", ""])
        for row in anomaly_overall.sort_values(["fx_mode", "avg_nrmse_transform"]).to_dict("records"):
            lines.append(
                f"- `{row['model']}` with `{row['fx_mode']}` FX: "
                f"avg transform NRMSE={format_float(row['avg_nrmse_transform'], 4)}, "
                f"avg level NRMSE={format_float(row['avg_nrmse_level'], 4)}, "
                f"targets={int(row['targets'])}"
            )

    lines.extend(["", "## Lag Effect Summary", ""])
    for row in lag_summary.to_dict("records"):
        lines.append(
            f"- `{row['target']}`: lag={row['selected_lag_months']}, "
            f"effect per +1% FX move={format_float(row['effect_per_1pct_fx_move'], 6)} "
            f"({row['unit_label']}), peak scenario delta={format_float(row['peak_scenario_level_delta'], 4)}"
        )

    if anomaly_lag_summary is not None and not anomaly_lag_summary.empty:
        lines.extend(["", "## Concatenated Anomaly-Set Scenario Peaks", ""])
        for row in anomaly_lag_summary.to_dict("records"):
            lines.append(
                f"- `{row['target']}`: peak model={row.get('peak_scenario_model')}, "
                f"peak horizon={row.get('peak_scenario_horizon')}, "
                f"peak scenario level delta={format_float(row.get('peak_scenario_level_delta'), 4)}"
            )

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `analysis/fx_impact/reports/target_selection/target_ranking.csv`",
            "- `analysis/fx_impact/reports/target_selection/target_ranking.json`",
            "- `analysis/fx_impact/reports/fx_model_selection/fx_model_comparison.csv`",
            "- `analysis/fx_impact/reports/fx_model_selection/selected_fx_predictions.csv`",
            "- `analysis/fx_impact/reports/final/model_comparison.csv`",
            "- `analysis/fx_impact/reports/final/impact_forecasts.csv`",
            "- `analysis/fx_impact/reports/final/lag_effect_summary.csv`",
            "- `analysis/fx_impact/reports/final/scenario_forecasts.csv`",
            "- `analysis/fx_impact/reports/final/plot_model_selection.csv`",
            "- `analysis/fx_impact/reports/final/anomaly_set/anomaly_model_panel.csv`",
            "- `analysis/fx_impact/reports/final/anomaly_set/anomaly_impact_forecasts.csv`",
            "- `analysis/fx_impact/reports/final/anomaly_set/anomaly_model_comparison.csv`",
            "- `analysis/fx_impact/reports/final/anomaly_set/anomaly_scenario_forecasts.csv`",
            "- `analysis/fx_impact/reports/final/anomaly_set/anomaly_lag_effect_summary.csv`",
            "",
            "## Interpretation Notes",
            "",
            "- `log_diff` forecasts are monthly log changes. Level forecasts are recursively inverse-transformed with `level_t = level_{t-1} * exp(log_change_t)`.",
            "- `diff` forecasts are monthly unit changes. Level forecasts are recursively inverse-transformed with `level_t = level_{t-1} + change_t`.",
            "- Anomaly-only target selection is reported but not over-weighted because overlap can be small for some targets.",
        ]
    )
    (FX_IMPACT_DIR / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_event_panel_result_md(
    ranking: pd.DataFrame,
    selected_source: str,
    scenario_shock_pct: float,
    event_outputs: dict[str, pd.DataFrame],
) -> None:
    selected = ranking[ranking["selected_for_final_model"]].copy()
    selected_targets = selected["target"].tolist()
    event_panel = event_outputs["event_panel"]
    metrics = event_outputs["metrics"]
    scenario = event_outputs["scenario"]
    model_selection = event_outputs["model_selection"]

    event_months = int(event_panel["event_date"].nunique()) if not event_panel.empty else 0
    panel_rows = int(len(event_panel))
    horizons = (
        ", ".join(str(int(h)) for h in sorted(event_panel["horizon"].dropna().unique()))
        if not event_panel.empty
        else "1-6"
    )
    source_models = sorted(event_panel["source_model"].dropna().unique().tolist()) if not event_panel.empty else [selected_source]
    selected_model_rows = (
        model_selection[
            model_selection["selection_scope"].eq("target")
            & model_selection["selected_for_plot"].eq(True)
            & ~model_selection["target"].eq("__ALL__")
        ].copy()
        if not model_selection.empty
        else pd.DataFrame()
    )
    selected_model_by_target = selected_model_rows.set_index("target")["model"].to_dict() if not selected_model_rows.empty else {}
    overall_rows = (
        model_selection[model_selection["selection_scope"].eq("overall")].sort_values("rank")
        if not model_selection.empty
        else pd.DataFrame()
    )

    scenario_peaks: dict[str, dict[str, Any]] = {}
    if not scenario.empty:
        for target, model in selected_model_by_target.items():
            part = scenario[scenario["target"].eq(target) & scenario["model"].eq(model)].copy()
            if part.empty:
                continue
            peak = part.assign(abs_delta=part["response_delta_vs_predicted"].abs()).sort_values(
                "abs_delta", ascending=False
            ).iloc[0]
            scenario_peaks[target] = peak.to_dict()

    median_nrmse = (
        float(selected_model_rows["avg_nrmse_response"].median())
        if not selected_model_rows.empty
        else np.nan
    )
    stable_targets = []
    unstable_targets = []
    for row in selected_model_rows.to_dict("records"):
        label = row["target"]
        if np.isfinite(median_nrmse) and row["avg_nrmse_response"] <= median_nrmse:
            stable_targets.append(label)
        else:
            unstable_targets.append(label)

    lines = [
        "# FX Impact Final Pipeline Result",
        "",
        "## Purpose",
        "",
        "Estimate how domestic macro/financial targets respond after an anomaly-month USD/KRW movement, with the model answering: if FX changes at event month `t`, what is the target response at horizons `h=1..6` months?",
        "",
        "## Why The Previous Anomaly-Concat Forecast Was Problematic",
        "",
        "- A continuous monthly forecast mostly measured ordinary calendar-time forecasting skill, not anomaly-conditional spillovers.",
        "- Filtering anomaly months and stitching them into one sequence removed real calendar gaps, so `shift(1)` could jump across disconnected months and distort lags.",
        "- Level-path forecast plots made the core question hard to read because they emphasized path fit rather than `h`-month response after an FX event.",
        "",
        "## Event-Time Local Projection",
        "",
        "- The new final layer builds an event-time panel at `(event_date, target, horizon)` granularity.",
        "- `event_date` rows are limited to `Is_Abnormal_Period == 1` months.",
        "- All lag and response values are read from the original monthly calendar index: `t-1`, `t-6`, and `t+h` are never computed on a stitched anomaly-only sequence.",
        "- The response is cumulative from event month `t` to `t+h`: log-level targets use `log(target_{t+h}) - log(target_t)`; differenced targets use `target_{t+h} - target_t`.",
        "",
        "## Data Configuration",
        "",
        f"- Anomaly event months used by the panel: `{event_months}`",
        f"- Event-time panel rows after horizon and missing-value drops: `{panel_rows}`",
        f"- Horizons: `{horizons}` months",
        f"- Selected targets: {', '.join(f'`{target}`' for target in selected_targets)}",
        f"- FX input source: `{', '.join(source_models)}`",
        "- Daily selected FX predictions are resampled to month-end monthly means and merged to the macro panel. Months without a hybrid prediction keep actual USD/KRW, matching the earlier controlled error-propagation policy.",
        f"- Scenario shock: predicted USD/KRW at each event month is multiplied by `{1.0 + scenario_shock_pct:.3f}`. The scenario current-month log shock is computed against the unshocked predicted `t-1`, so the event shock is present in `scenario_fx_change_t`.",
        "",
        "## Models",
        "",
        "- `LocalProjection_OLS`: separate OLS for each target and horizon with HAC covariance for p-values.",
        "- `LocalProjection_Ridge`: separate Ridge model with features scaled on the train split only.",
        "- `LocalProjection_ElasticNet`: separate ElasticNet/ElasticNetCV model using time-series CV when the event sample is large enough.",
        "- Feature set: `fx_shock_t`, `fx_lag1..fx_lag6`, and `target_lag1`. External controls were intentionally left out because the event sample is small after target/horizon filtering.",
        "- Split: target/horizon-specific time split by `event_date`; the last roughly 25% of valid anomaly event dates are held out for test.",
        "- Leakage control: scalers are fit only on train rows, target future values are used only as `actual_response`, and lag/response lookup always uses the original calendar index.",
        "",
        "## Performance Summary",
        "",
    ]

    if overall_rows.empty:
        lines.append("- No event-time LP metrics were generated.")
    else:
        lines.append("Overall predicted-FX test metrics by model:")
        for row in overall_rows.to_dict("records"):
            lines.append(
                f"- `{row['model']}`: avg RMSE={format_float(row['avg_rmse_response'], 6)}, "
                f"avg NRMSE={format_float(row['avg_nrmse_response'], 4)}, "
                f"avg level-delta RMSE={format_float(row['avg_rmse_level_delta'], 4)}, "
                f"targets={int(row.get('targets', 0))}"
            )

    lines.extend(["", "Target-level selected plot models:"])
    if selected_model_rows.empty:
        lines.append("- n/a")
    else:
        for row in selected_model_rows.sort_values("target").to_dict("records"):
            peak = scenario_peaks.get(row["target"], {})
            lines.append(
                f"- `{row['target']}`: `{row['model']}`, "
                f"avg NRMSE={format_float(row['avg_nrmse_response'], 4)}, "
                f"avg RMSE={format_float(row['avg_rmse_response'], 6)}, "
                f"peak scenario-baseline response={format_float(peak.get('response_delta_vs_predicted'), 6)} "
                f"at h={peak.get('horizon', 'n/a')}"
            )

    lines.extend(["", "Required target comments:"])
    for target in ["KOSPI", "Import_Price_Index", "Industrial_Production", "Trade_Balance"]:
        row = selected_model_rows[selected_model_rows["target"].eq(target)]
        peak = scenario_peaks.get(target, {})
        if row.empty:
            lines.append(f"- `{target}`: no selected event-time model was available.")
            continue
        row_dict = row.iloc[0].to_dict()
        try:
            peak_delta = float(peak.get("response_delta_vs_predicted", np.nan))
        except Exception:
            peak_delta = np.nan
        if not np.isfinite(peak_delta):
            direction = "n/a"
        elif abs(peak_delta) <= 1e-10:
            direction = "near-zero"
        else:
            direction = "positive" if peak_delta > 0 else "negative"
        stability = "stable" if target in stable_targets else "less stable"
        lines.append(
            f"- `{target}`: best `{row_dict['model']}` with avg NRMSE={format_float(row_dict['avg_nrmse_response'], 4)}; "
            f"scenario effect is {direction} and peaks around h={peak.get('horizon', 'n/a')} ({stability})."
        )

    lines.extend(["", "## Main Interpretation", ""])
    if scenario_peaks:
        peak_table = pd.DataFrame(scenario_peaks.values())
        strongest = peak_table.assign(abs_delta=peak_table["response_delta_vs_predicted"].abs()).sort_values(
            "abs_delta", ascending=False
        )
        lines.append("Largest scenario-baseline event responses by absolute response delta:")
        for row in strongest.head(5).to_dict("records"):
            direction = "up" if row["response_delta_vs_predicted"] > 0 else "down"
            lines.append(
                f"- `{row['target']}` moves {direction} most at h={int(row['horizon'])}: "
                f"response delta={format_float(row['response_delta_vs_predicted'], 6)}, "
                f"level-delta effect={format_float(row['level_delta_vs_predicted'], 4)}"
            )
    else:
        lines.append("- Scenario peaks were not available.")
    lines.append(
        f"- More stable target fits by predicted-FX NRMSE: {', '.join(f'`{t}`' for t in stable_targets) if stable_targets else 'n/a'}."
    )
    lines.append(
        f"- Less stable target fits needing cautious interpretation: {', '.join(f'`{t}`' for t in unstable_targets) if unstable_targets else 'n/a'}."
    )

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `analysis/fx_impact/reports/final/event_panel/anomaly_event_panel.csv`",
            "- `analysis/fx_impact/reports/final/event_panel/local_projection_coefficients.csv`",
            "- `analysis/fx_impact/reports/final/event_panel/event_response_forecasts.csv`",
            "- `analysis/fx_impact/reports/final/event_panel/local_projection_metrics.csv`",
            "- `analysis/fx_impact/reports/final/event_panel/scenario_response_forecasts.csv`",
            "- `analysis/fx_impact/reports/final/event_panel/event_panel_model_selection.csv`",
            "- `analysis/fx_impact/reports/final/event_panel/plots/response_*.png`",
            "- `analysis/fx_impact/reports/final/event_panel/plots/response_summary_top_targets.png`",
            "- `analysis/fx_impact/reports/final/plots/response_*.png` contains the latest event-time response plots for quick access. Existing calendar-time output tables are preserved.",
            "",
            "## Reproduction Command",
            "",
            "```bash",
            "python analysis/fx_impact/run_final_fx_impact_pipeline.py",
            "```",
        ]
    )
    (FX_IMPACT_DIR / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the final USD/KRW FX impact modeling pipeline.")
    parser.add_argument("--macro-path", type=Path, default=MACRO_PATH)
    parser.add_argument("--period-definition", type=Path, default=PERIOD_DEF_PATH)
    parser.add_argument("--scenario-shock-pct", type=float, default=0.05)
    parser.add_argument("--shock-path", type=Path, default=None, help="Optional CSV with date and fx/pred_fx/USD_KRW columns.")
    parser.add_argument("--test-obs", type=int, default=TEST_OBS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    macro_df = load_macro_dataset(args.macro_path)
    _ = load_period_definition(args.period_definition)

    print("[1/4] Selecting FX-sensitive macro/financial targets...")
    ranking, _ = run_target_selection(macro_df)
    selected_targets = ranking[ranking["selected_for_final_model"]]["target"].tolist()
    print(f"Selected targets: {', '.join(selected_targets)}")

    print("[2/4] Comparing hybrid FX prediction paths...")
    fx_comparison, selected_predictions, selected_source = run_fx_model_selection(macro_df, selected_targets)
    print(f"Selected FX source: {selected_source}")
    print(fx_comparison[["source_model", "selected", "daily_available_rmse", "downstream_avg_rmse"]].to_string(index=False))

    print("[3/4] Preserving calendar-time comparison models...")
    model_comparison, forecast_df, lag_summary, scenario_df = run_final_impact_models(
        macro_df,
        ranking,
        selected_predictions,
        selected_source,
        scenario_shock_pct=args.scenario_shock_pct,
        custom_shock_path=args.shock_path,
        test_obs=args.test_obs,
    )
    print("Final model comparison:")
    if not model_comparison.empty:
        print(model_comparison[model_comparison["target"].eq("__ALL__")].to_string(index=False))
    print(f"Generated {len(forecast_df)} forecast rows and {len(scenario_df)} scenario rows.")

    print("[4/4] Running event-time local projection final pipeline...")
    event_outputs = run_event_time_local_projection_analysis(
        macro_df,
        ranking,
        selected_predictions,
        selected_source,
        scenario_shock_pct=args.scenario_shock_pct,
    )
    write_event_panel_result_md(
        ranking,
        selected_source,
        scenario_shock_pct=args.scenario_shock_pct,
        event_outputs=event_outputs,
    )
    event_panel = event_outputs["event_panel"]
    event_metrics = event_outputs["metrics"]
    print(
        "Event-time LP outputs: "
        f"{len(event_panel)} panel rows, "
        f"{event_panel['event_date'].nunique() if not event_panel.empty else 0} anomaly event months, "
        f"{len(event_metrics)} metric rows."
    )
    print(f"Reports saved under {REPORT_DIR.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
