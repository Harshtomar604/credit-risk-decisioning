from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from cv_compare import load_train
from pipelines import SEED, make_gbm, make_logreg

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "calibration_output.txt"
PLOT_PATH = ROOT / "reports" / "calibration.png"
N_BINS = 10


def expected_calibration_error(y, p, n_bins=N_BINS):
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return ece


def main():
    X, y = load_train()
    y = y.to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    models = {
        "logistic_regression": make_logreg(),
        "gradient_boosting": make_gbm(),
        "gbm_isotonic": CalibratedClassifierCV(make_gbm(), method="isotonic", cv=5),
        "gbm_sigmoid": CalibratedClassifierCV(make_gbm(), method="sigmoid", cv=5),
    }

    lines = [f"training rows: {len(y)}, actual default rate: {y.mean():.4f}", ""]
    PLOT_PATH.parent.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect calibration")

    for name, model in models.items():
        p = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=N_BINS, strategy="quantile")

        lines += [
            f"=== {name} ===",
            f"AUC (out-of-fold):   {roc_auc_score(y, p):.4f}",
            f"Brier (out-of-fold): {brier_score_loss(y, p):.4f}",
            f"ECE:                 {expected_calibration_error(y, p):.4f}",
            f"mean predicted:      {p.mean():.4f}",
            "bin  mean_predicted  actual_rate",
        ]
        for i, (mp, fp) in enumerate(zip(mean_pred, frac_pos)):
            lines.append(f"{i:>3}  {mp:>14.4f}  {fp:>11.4f}")
        lines.append("")

        ax.plot(mean_pred, frac_pos, marker="o", label=name)

    ax.set_xlabel("Mean predicted default probability")
    ax.set_ylabel("Actual default rate")
    ax.set_title("Calibration (5-fold out-of-fold, training set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)

    text = "\n".join(lines)
    OUT_PATH.write_text(text)
    print(text)
    print(f"saved to {OUT_PATH.name} and {PLOT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
