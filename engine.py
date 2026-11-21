"""Model comparison engine: given a table + a target column, benchmark several models.

Everything here is plain pandas + scikit-learn (+ xgboost / lightgbm / catboost).
The single public function is `run_comparison(df, target, task="auto")`, which returns a
JSON-safe dict that the web page turns into tables and charts.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR

warnings.filterwarnings("ignore")

MAX_ROWS = 5000          # default cap on rows used for training/CV
TIME_BUDGET_S = 600      # stop starting new models after this many seconds
RANDOM_STATE = 42


def _is_text(s: pd.Series) -> bool:
    """True for object/string/category columns (pandas 3 uses a dedicated string dtype)."""
    return (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
    ) and not pd.api.types.is_bool_dtype(s)


class DataError(ValueError):
    """A problem with the user's data that we can explain in plain words."""


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def _prepare(df: pd.DataFrame, target: str, task: str, max_rows: int = MAX_ROWS):
    if target not in df.columns:
        raise DataError(f"Target column '{target}' was not found in the file.")

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    target = str(target).strip()
    df = df.loc[:, ~df.columns.duplicated()]
    n_raw = len(df)

    df = df.dropna(subset=[target]).reset_index(drop=True)
    if len(df) < 30:
        raise DataError("Need at least 30 rows with a non-empty target to compare models.")

    y_raw = df[target]
    X = df.drop(columns=[target])
    if X.shape[1] == 0:
        raise DataError("The file needs at least one feature column besides the target.")

    # Turn "numbers stored as text" into real numbers.
    for col in X.columns:
        if _is_text(X[col]):
            conv = pd.to_numeric(X[col], errors="coerce")
            if conv.notna().sum() >= 0.9 * X[col].notna().sum() and X[col].notna().sum() > 0:
                X[col] = conv

    # ---- decide the task ----
    if task == "auto":
        if _is_text(y_raw) or pd.api.types.is_bool_dtype(y_raw):
            task = "classification"
        else:
            vals = pd.to_numeric(y_raw, errors="coerce")
            is_int = (vals.dropna() % 1 == 0).all()
            task = "classification" if (is_int and vals.nunique() <= 10) else "regression"

    if task == "regression":
        y_num = pd.to_numeric(y_raw, errors="coerce")
        if y_num.notna().mean() < 0.95:
            raise DataError("The target is not numeric, so it cannot be used for regression.")
        keep = y_num.notna()
        X, y_series = X.loc[keep].reset_index(drop=True), y_num.loc[keep].reset_index(drop=True)
    else:
        y_series = y_raw.astype(str).reset_index(drop=True)
        counts = y_series.value_counts()
        if len(counts) < 2:
            raise DataError("The target has only one distinct value - nothing to predict.")
        if len(counts) > 30:
            raise DataError(
                f"The target has {len(counts)} distinct values - that looks like a continuous "
                "number or an ID. Pick another column or force 'Regression'."
            )
        rare = counts[counts < 3].index
        if len(rare):
            keep = ~y_series.isin(rare)
            X, y_series = X.loc[keep].reset_index(drop=True), y_series.loc[keep].reset_index(drop=True)
            if y_series.nunique() < 2:
                raise DataError("Too few rows per class to run cross-validation.")

    # ---- cap the row count ----
    sampled = False
    if len(X) > max_rows:
        sampled = True
        if task == "classification":
            idx, _ = train_test_split(
                np.arange(len(X)), train_size=max_rows, stratify=y_series, random_state=RANDOM_STATE
            )
        else:
            idx = np.random.RandomState(RANDOM_STATE).choice(len(X), max_rows, replace=False)
        idx = np.sort(idx)
        X, y_series = X.iloc[idx].reset_index(drop=True), y_series.iloc[idx].reset_index(drop=True)

    # ---- drop columns that cannot help ----
    dropped = []
    keep_cols = []
    for col in X.columns:
        s = X[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            dropped.append((col, "date/time column"))
        elif s.nunique(dropna=True) <= 1:
            dropped.append((col, "constant or empty"))
        elif _is_text(s) and s.nunique() > 20 and s.nunique() / max(len(s), 1) > 0.5:
            dropped.append((col, "ID-like / free text"))
        elif s.isna().mean() > 0.6:
            dropped.append((col, "more than 60% missing"))
        else:
            keep_cols.append(col)
    if not keep_cols:
        raise DataError("After removing unusable columns, no feature columns were left.")
    X = X[keep_cols]

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and X[c].dtype != bool]
    cat_cols = [c for c in X.columns if c not in num_cols]
    for c in cat_cols:
        X[c] = X[c].astype(str).where(X[c].notna(), np.nan)

    return X, y_series, task, target, num_cols, cat_cols, dropped, n_raw, sampled


def _preprocessor(num_cols, cat_cols):
    parts = []
    if num_cols:
        parts.append(
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols)
        )
    if cat_cols:
        parts.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        (
                            "oh",
                            OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=15, sparse_output=False),
                        ),
                    ]
                ),
                cat_cols,
            )
        )
    return ColumnTransformer(parts, sparse_threshold=0.0)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def _build_models(task: str):
    """Return (models, skipped) where models is {name: estimator}."""
    clf = task == "classification"
    models, skipped = {}, {}

    if clf:
        models["Logistic Regression"] = LogisticRegression(max_iter=1000)
        skipped["Linear Regression"] = "Predicts continuous numbers, so it only runs on regression targets."
    else:
        models["Linear Regression"] = LinearRegression()
        skipped["Logistic Regression"] = "Predicts categories, so it only runs on classification targets."

    models["SVM"] = SVC() if clf else SVR()
    models["KNN"] = KNeighborsClassifier(n_neighbors=7) if clf else KNeighborsRegressor(n_neighbors=7)
    models["Random Forest"] = (
        RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=RANDOM_STATE)
        if clf
        else RandomForestRegressor(n_estimators=150, n_jobs=-1, random_state=RANDOM_STATE)
    )

    try:
        import xgboost as xgb

        models["XGBoost"] = (
            xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, tree_method="hist",
                              n_jobs=-1, verbosity=0, random_state=RANDOM_STATE)
            if clf
            else xgb.XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.1, tree_method="hist",
                                  n_jobs=-1, verbosity=0, random_state=RANDOM_STATE)
        )
    except Exception:
        skipped["XGBoost"] = "Library is not installed in this deployment."

    try:
        import lightgbm as lgb

        models["LightGBM"] = (
            lgb.LGBMClassifier(n_estimators=150, learning_rate=0.1, verbose=-1, n_jobs=-1, random_state=RANDOM_STATE)
            if clf
            else lgb.LGBMRegressor(n_estimators=150, learning_rate=0.1, verbose=-1, n_jobs=-1, random_state=RANDOM_STATE)
        )
    except Exception:
        skipped["LightGBM"] = "Library is not installed in this deployment."

    try:
        import catboost as cb

        common = dict(iterations=200, learning_rate=0.1, depth=6, verbose=0, allow_writing_files=False,
                      random_seed=RANDOM_STATE, thread_count=-1)
        models["CatBoost"] = cb.CatBoostClassifier(**common) if clf else cb.CatBoostRegressor(**common)
    except Exception:
        skipped["CatBoost"] = "Library is not installed in this deployment."

    return models, skipped


# --------------------------------------------------------------------------- #
# Scoring helpers
# --------------------------------------------------------------------------- #
def _scores(model, Xte, yte, task, n_classes):
    pred = model.predict(Xte)
    if task == "regression":
        return (
            {
                "R²": r2_score(yte, pred),
                "RMSE": float(np.sqrt(mean_squared_error(yte, pred))),
                "MAE": mean_absolute_error(yte, pred),
            },
            pred,
            None,
        )

    out = {
        "Accuracy": accuracy_score(yte, pred),
        "F1": f1_score(yte, pred, average="weighted", zero_division=0),
        "Precision": precision_score(yte, pred, average="weighted", zero_division=0),
        "Recall": recall_score(yte, pred, average="weighted", zero_division=0),
        "ROC AUC": np.nan,
    }
    score = None
    try:
        if hasattr(model, "predict_proba"):
            proba = np.asarray(model.predict_proba(Xte))
            if n_classes == 2:
                score = proba[:, 1]
                out["ROC AUC"] = roc_auc_score(yte, score)
            else:
                out["ROC AUC"] = roc_auc_score(yte, proba, multi_class="ovr", average="weighted",
                                               labels=list(range(n_classes)))
        elif n_classes == 2 and hasattr(model, "decision_function"):
            score = np.asarray(model.decision_function(Xte))
            out["ROC AUC"] = roc_auc_score(yte, score)
    except Exception:
        pass
    return out, pred, score


def _clean(o):
    """Make anything JSON-safe (numpy -> python, NaN/inf -> None)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if (np.isnan(o) or np.isinf(o)) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def run_comparison(df: pd.DataFrame, target: str, task: str = "auto", max_rows: int = MAX_ROWS) -> dict:
    t_start = time.time()
    X, y_series, task, target, num_cols, cat_cols, dropped, n_raw, sampled = _prepare(df, target, task, max_rows)
    clf = task == "classification"
    n = len(X)

    if clf:
        le = LabelEncoder()
        y = le.fit_transform(y_series)
        classes = [str(c) for c in le.classes_]
        n_classes = len(classes)
        n_splits = int(min(5, np.bincount(y).min()))
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        primary = "F1"
        metric_names = ["Accuracy", "F1", "Precision", "Recall", "ROC AUC"]
    else:
        y = y_series.to_numpy(dtype=float)
        classes, n_classes = [], 0
        n_splits = 5
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        primary = "R²"
        metric_names = ["R²", "RMSE", "MAE"]
    folds = list(splitter.split(X, y))

    models, skipped = _build_models(task)
    baseline = DummyClassifier(strategy="most_frequent") if clf else DummyRegressor(strategy="mean")
    run_list = [("Baseline", baseline)] + list(models.items())

    rows, oof, failed = [], {}, {}
    for name, est in run_list:
        if time.time() - t_start > TIME_BUDGET_S and name != "Baseline":
            skipped[name] = "Skipped: the time budget for this run was used up."
            continue
        pipe = Pipeline([("prep", _preprocessor(num_cols, cat_cols)), ("model", est)])
        fold_scores, fit_times = [], []
        pred_all = np.zeros(n, dtype=float)
        score_all = np.full(n, np.nan)
        try:
            for tr, te in folds:
                m = clone(pipe)
                t0 = time.time()
                m.fit(X.iloc[tr], y[tr])
                fit_times.append(time.time() - t0)
                s, pred, score = _scores(m, X.iloc[te], y[te], task, n_classes)
                fold_scores.append(s)
                pred_all[te] = pred
                if score is not None:
                    score_all[te] = score
        except Exception as exc:  # a model failing should never kill the whole run
            failed[name] = f"{type(exc).__name__}: {str(exc)[:120]}"
            continue

        row = {"model": name, "is_baseline": name == "Baseline",
               "fit_time": float(np.mean(fit_times)), "total_time": float(np.sum(fit_times))}
        for mn in metric_names:
            vals = np.array([f[mn] for f in fold_scores], dtype=float)
            row[mn] = float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else None
            row[mn + "_std"] = float(np.nanstd(vals)) if not np.all(np.isnan(vals)) else None
            row[mn + "_folds"] = [None if np.isnan(v) else float(v) for v in vals]
        rows.append(row)
        oof[name] = (pred_all, score_all)

    contenders = [r for r in rows if not r["is_baseline"]]
    if not contenders:
        raise DataError("No model could be trained on this data. " + "; ".join(failed.values()))

    higher_better = primary in ("F1", "R²")
    key = (lambda r: (-(r[primary] if r[primary] is not None else -1e18), r["fit_time"]))
    contenders.sort(key=key)
    for i, r in enumerate(contenders, 1):
        r["rank"] = i
    best = contenders[0]
    best_name = best["model"]
    # Models that are within one standard deviation of the best are statistically close.
    tol = best[primary + "_std"] or 0.0
    close = [r["model"] for r in contenders[1:]
             if r[primary] is not None and abs(best[primary] - r[primary]) <= tol]

    # ---------------- best-model diagnostics ----------------
    pred_best, score_best = oof[best_name]
    diag = {}
    if clf:
        pred_i = pred_best.astype(int)
        cm = confusion_matrix(y, pred_i, labels=list(range(n_classes)))
        p, r_, f, s = precision_recall_fscore_support(y, pred_i, labels=list(range(n_classes)), zero_division=0)
        diag["confusion"] = {"labels": classes, "matrix": cm.tolist()}
        diag["class_report"] = [
            {"class": classes[i], "precision": p[i], "recall": r_[i], "f1": f[i], "support": int(s[i])}
            for i in range(n_classes)
        ]
    else:
        resid = y - pred_best
        rs = np.random.RandomState(0)
        idx = rs.choice(n, min(n, 800), replace=False)
        diag["scatter"] = {"actual": y[idx], "pred": pred_best[idx]}
        diag["residuals"] = resid
        diag["resid_summary"] = {"mean": float(resid.mean()), "std": float(resid.std())}

    roc = {}
    if clf and n_classes == 2:
        for r in contenders:
            sc = oof[r["model"]][1]
            if not np.all(np.isnan(sc)):
                fpr, tpr, _ = roc_curve(y, sc)
                step = max(1, len(fpr) // 120)
                roc[r["model"]] = {"fpr": fpr[::step], "tpr": tpr[::step],
                                   "auc": r["ROC AUC"]}

    # Permutation importance on original columns for the winner (hold-out set).
    importance = []
    try:
        strat = y if clf else None
        tr, te = train_test_split(np.arange(n), test_size=0.25, random_state=RANDOM_STATE, stratify=strat)
        te = te[:1000]
        pipe = Pipeline([("prep", _preprocessor(num_cols, cat_cols)), ("model", clone(models[best_name]))])
        pipe.fit(X.iloc[tr], y[tr])
        scoring = "f1_weighted" if clf else "r2"
        pi = permutation_importance(pipe, X.iloc[te], y[te], n_repeats=3, random_state=0, scoring=scoring, n_jobs=1)
        order = np.argsort(pi.importances_mean)[::-1][:12]
        importance = [{"feature": X.columns[i], "importance": float(pi.importances_mean[i]),
                       "std": float(pi.importances_std[i])} for i in order]
    except Exception:
        importance = []

    # ---------------- dataset profile ----------------
    types = {c: ("numeric" if c in num_cols else "categorical") for c in X.columns}
    col_table = [{"column": target, "role": "target", "type": "categorical" if clf else "numeric",
                  "missing": float(df[target].isna().mean() * 100), "unique": int(df[target].nunique())}]
    for c in X.columns:
        col_table.append({"column": c, "role": "feature", "type": types[c],
                          "missing": float(df[c].isna().mean() * 100) if c in df else 0.0,
                          "unique": int(X[c].nunique())})
    for c, why in dropped:
        col_table.append({"column": c, "role": f"dropped ({why})", "type": "-",
                          "missing": float(df[c].isna().mean() * 100) if c in df else 0.0,
                          "unique": int(df[c].nunique()) if c in df else 0})

    if clf:
        vc = pd.Series(y_series).value_counts().head(15)
        target_dist = {"kind": "classes", "labels": [str(i) for i in vc.index], "counts": vc.values}
    else:
        counts, edges = np.histogram(y, bins=30)
        target_dist = {"kind": "hist", "edges": edges, "counts": counts}

    corr = None
    if len(num_cols) >= 2 or (num_cols and not clf):
        cdf = X[num_cols].copy()
        if not clf:
            cdf[target] = y
        if cdf.shape[1] >= 2:
            variances = cdf.var().sort_values(ascending=False)
            keep = list(variances.index[:12])
            if not clf and target not in keep:
                keep = keep[:11] + [target]
            c = cdf[keep].corr().round(2)
            corr = {"labels": keep, "matrix": c.where(c.notna(), None).values.tolist()}

    miss = (df.drop(columns=[target]).isna().mean() * 100).sort_values(ascending=False)
    miss = miss[miss > 0].head(10)
    missing = {"labels": list(miss.index), "values": miss.values}

    summary = {
        "rows_uploaded": int(n_raw), "rows_used": int(n), "sampled": bool(sampled),
        "n_features": int(X.shape[1]), "n_numeric": len(num_cols), "n_categorical": len(cat_cols),
        "missing_cells_pct": float(df.drop(columns=[target]).isna().mean().mean() * 100),
        "target": target, "task": task, "classes": classes, "n_splits": n_splits,
        "primary_metric": primary, "metrics": metric_names, "higher_better": higher_better,
        "elapsed": time.time() - t_start,
        "class_balance": (float(np.bincount(y).max() / n) if clf else None),
    }
    skipped_list = [{"model": k, "reason": v} for k, v in skipped.items()]
    skipped_list += [{"model": k, "reason": "Failed: " + v} for k, v in failed.items()]

    return _clean({
        "summary": summary,
        "results": [r for r in rows],
        "ranking": [r["model"] for r in contenders],
        "best": best_name,
        "close": close,
        "diag": diag,
        "roc": roc,
        "importance": importance,
        "columns": col_table,
        "target_dist": target_dist,
        "corr": corr,
        "missing": missing,
        "skipped": skipped_list,
    })
