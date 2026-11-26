from pathlib import Path

import duckdb
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from pipelines import FEATURES

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "credit.duckdb"
MODEL_DIR = ROOT / "models"
OUT_PATH = ROOT / "profit_output.txt"
PLOT_PATH = ROOT / "reports" / "profit_curve.png"

BASE_R = 0.15
BASE_LGD = 0.75
R_GRID = [0.10, 0.15, 0.20]
LGD_GRID = [0.45, 0.75, 0.90]
THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.005), 3)


def load_test():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("""
        SELECT f.*
        FROM features f
        JOIN split s USING (client_id)
        WHERE s.split = 'test'
        ORDER BY client_id
    """).df()
    con.close()
    return df


def realised_profit(p, y, ead, threshold, r, lgd):
    approved = p < threshold
    gain = np.where(y == 0, r * ead, -lgd * ead)
    return gain[approved].sum(), approved.mean(), (y[approved].mean() if approved.any() else np.nan)


def sweep(p, y, ead, r, lgd):
    rows = []
    for t in THRESHOLDS:
        profit, appr, dr = realised_profit(p, y, ead, t, r, lgd)
        rows.append({"threshold": t, "profit": profit, "approval_rate": appr, "default_rate_approved": dr})
    return pd.DataFrame(rows)


def fmt_row(label, t, profit, appr, dr):
    return f"{label:<36} threshold {t:.3f} | profit {profit / 1e6:>8.1f}m | approval {appr:.3f} | default among approved {dr:.3f}"


def main():
    df = load_test()
    X, y, ead = df[FEATURES], df["default"].to_numpy(), df["limit_bal"].to_numpy(dtype=float)

    models = {
        "gbm_isotonic": joblib.load(MODEL_DIR / "gbm_isotonic.joblib"),
        "logistic_regression": joblib.load(MODEL_DIR / "logistic_regression.joblib"),
    }
    preds = {name: m.predict_proba(X)[:, 1] for name, m in models.items()}

    lines = [f"test rows: {len(y)}, default rate: {y.mean():.4f}", "", "=== test-set performance ==="]
    for name, p in preds.items():
        lines.append(f"{name:<20} AUC {roc_auc_score(y, p):.4f} | Brier {brier_score_loss(y, p):.4f} | mean predicted {p.mean():.4f}")

    breakeven = BASE_R / (BASE_R + BASE_LGD)
    approve_all = np.where(y == 0, BASE_R * ead, -BASE_LGD * ead).sum()
    lines += [
        "",
        f"=== base case: r={BASE_R}, LGD={BASE_LGD}, EAD=limit_bal (profit in NT$ millions) ===",
        f"break-even threshold r/(r+LGD) = {breakeven:.3f}",
        f"{'approve everyone':<36} profit {approve_all / 1e6:>8.1f}m | approval 1.000 | default among approved {y.mean():.3f}",
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, p in preds.items():
        curve = sweep(p, y, ead, BASE_R, BASE_LGD)
        best = curve.loc[curve["profit"].idxmax()]
        be_profit, be_appr, be_dr = realised_profit(p, y, ead, breakeven, BASE_R, BASE_LGD)
        lines.append(fmt_row(f"{name} @ break-even", breakeven, be_profit, be_appr, be_dr))
        lines.append(fmt_row(f"{name} @ best on test", best["threshold"], best["profit"], best["approval_rate"], best["default_rate_approved"]))
        ax.plot(curve["threshold"], curve["profit"] / 1e6, label=name)

    ax.axvline(breakeven, linestyle="--", color="grey", label=f"break-even {breakeven:.3f}")
    ax.axhline(approve_all / 1e6, linestyle=":", color="black", label="approve everyone")
    ax.set_xlabel("Approval threshold (approve if predicted PD < threshold)")
    ax.set_ylabel("Realised profit on test set (NT$ m)")
    ax.set_title(f"Profit vs threshold (r={BASE_R}, LGD={BASE_LGD})")
    ax.legend()
    fig.tight_layout()
    PLOT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150)

    lines += ["", "=== sensitivity grid: gbm_isotonic (break-even rule vs best on test) ==="]
    lines.append(f"{'r':>5} {'LGD':>5} | {'break-even':>10} {'profit':>9} {'approval':>8} | {'best t':>7} {'profit':>9} {'approval':>8}")
    p = preds["gbm_isotonic"]
    for r in R_GRID:
        for lgd in LGD_GRID:
            be = r / (r + lgd)
            be_profit, be_appr, _ = realised_profit(p, y, ead, be, r, lgd)
            curve = sweep(p, y, ead, r, lgd)
            best = curve.loc[curve["profit"].idxmax()]
            lines.append(
                f"{r:>5.2f} {lgd:>5.2f} | {be:>10.3f} {be_profit / 1e6:>8.1f}m {be_appr:>8.3f} | "
                f"{best['threshold']:>7.3f} {best['profit'] / 1e6:>8.1f}m {best['approval_rate']:>8.3f}"
            )

    con = duckdb.connect(str(DB_PATH))
    out = pd.DataFrame({"client_id": df["client_id"], "limit_bal": ead, "default": y,
                        "p_gbm": preds["gbm_isotonic"], "p_logreg": preds["logistic_regression"]})
    con.execute("CREATE OR REPLACE TABLE test_predictions AS SELECT * FROM out")
    con.close()
    lines += ["", "test predictions saved to DuckDB table test_predictions"]

    text = "\n".join(lines)
    OUT_PATH.write_text(text)
    print(text)
    print(f"\nsaved to {OUT_PATH.name} and {PLOT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
