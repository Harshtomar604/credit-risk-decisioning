from pathlib import Path

import duckdb
from sklearn.model_selection import StratifiedKFold, cross_validate

from pipelines import FEATURES, SEED, make_gbm, make_logreg

DB_PATH = Path("data/credit.duckdb")
OUT_PATH = Path("cv_output.txt")


def load_train():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("""
        SELECT f.*
        FROM features f
        JOIN split s USING (client_id)
        WHERE s.split = 'train'
    """).df()
    con.close()
    return df[FEATURES], df["default"]


def main():
    X, y = load_train()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    lines = [f"training rows: {len(X)}, default rate: {y.mean():.4f}", ""]
    for name, model in [("logistic_regression", make_logreg()), ("gradient_boosting", make_gbm())]:
        res = cross_validate(model, X, y, cv=cv, scoring=["roc_auc", "neg_brier_score"])
        auc = res["test_roc_auc"]
        brier = -res["test_neg_brier_score"]
        lines += [
            f"=== {name} ===",
            f"AUC per fold:   {[round(float(a), 4) for a in auc]}",
            f"AUC mean (std): {auc.mean():.4f} ({auc.std():.4f})",
            f"Brier per fold: {[round(float(b), 4) for b in brier]}",
            f"Brier mean (std): {brier.mean():.4f} ({brier.std():.4f})",
            "",
        ]

    text = "\n".join(lines)
    OUT_PATH.write_text(text)
    print(text)
    print(f"saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
