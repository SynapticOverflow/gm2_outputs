"""
train_timing_model.py

Trains a regression model that predicts LIFU therapy efficacy (day-365 GM2
percent reduction vs. natural history) from acoustic parameters, target
depth, and *session timing* -- answering "how well will this protocol
work" for any candidate (where, how, when) combination, without having to
run the 8760-hour SDE simulator at inference time.

Input features (7):
    depth_mm, f_mhz, P_mpa, DC, eta, t_first_hr, n_sessions, spacing_hr
    (eta is included as an engineered feature -- it's a deterministic
    function of f/P/DC/depth, but it captures the acoustic model's
    nonlinear cavitation-threshold sigmoid directly and reliably improves
    fit over the raw acoustic parameters alone.)

Target: reduction_pct (from generate_timing_data.py's simulation output).

Model: Gradient Boosting Regressor (scikit-learn). Chosen over a linear
model because the underlying relationship is known to be nonlinear (the
cavitation sigmoid in eta, saturating dose-response in the SDE dynamics)
and there isn't remotely enough data here (3,000 points) to justify a deep
network -- gradient boosting is the standard, hard-to-beat baseline for
tabular regression at this scale.

Outputs:
    lifu_timing_model.joblib      -- trained model + feature list (for reuse)
    timing_model_metrics.json     -- train/test R^2, MAE, feature importances
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(OUT_DIR, "timing_training_data.csv")
MODEL_PATH = os.path.join(OUT_DIR, "lifu_timing_model.joblib")
METRICS_PATH = os.path.join(OUT_DIR, "timing_model_metrics.json")

FEATURES = ["depth_mm", "f_mhz", "P_mpa", "DC", "eta", "t_first_hr", "n_sessions", "spacing_hr"]
TARGET = "reduction_pct"


def load_data():
    df = pd.read_csv(DATA_CSV)
    return df


def train(df, test_size=0.2, random_state=0):
    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = GradientBoostingRegressor(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    metrics = dict(
        n_train=len(X_train),
        n_test=len(X_test),
        r2_train=r2_score(y_train, pred_train),
        r2_test=r2_score(y_test, pred_test),
        mae_train=mean_absolute_error(y_train, pred_train),
        mae_test=mean_absolute_error(y_test, pred_test),
        feature_importances={
            f: float(imp) for f, imp in zip(FEATURES, model.feature_importances_)
        },
    )
    return model, metrics


def main():
    df = load_data()
    print(f"Loaded {len(df)} training rows from {DATA_CSV}")

    model, metrics = train(df)

    joblib.dump(dict(model=model, features=FEATURES, target=TARGET), MODEL_PATH)
    with open(METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"Wrote {MODEL_PATH}")
    print(f"Wrote {METRICS_PATH}")
    print()
    print(f"Train R^2 = {metrics['r2_train']:.3f}   Test R^2 = {metrics['r2_test']:.3f}")
    print(f"Train MAE = {metrics['mae_train']:.2f} pct points   "
          f"Test MAE = {metrics['mae_test']:.2f} pct points")
    print()
    print("Feature importances (higher = more predictive of efficacy):")
    for f, imp in sorted(metrics["feature_importances"].items(), key=lambda kv: -kv[1]):
        print(f"  {f:12s} {imp:.3f}")


if __name__ == "__main__":
    main()
