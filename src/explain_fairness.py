from pathlib import Path

import duckdb
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from pipelines import FEATURES, SEED

DB_PATH = Path("data/credit.duckdb")
MODEL_PATH = Path("models/gbm_isotonic.joblib")
OUT_PATH = Path("explain_output.txt")
PLOT_PATH = Path("reports/permutation_importance.png")
THRESHOLD = 0.15 / (0.15 + 0.75)

SEX_LABELS = {1: "male", 2: "female"}
MARRIAGE_LABELS = {1: "married", 2: "single", 3: "other"}


def load_test():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("""
        SELECT f.*,
               c.sex,
               CASE WHEN c.marriage IN (1, 2) THEN c.marriage ELSE 3 END AS marriage,
               t.p_gbm
        FROM features f
        JOIN split s USING (client_id)
        JOIN clients c USING (client_id)
        JOIN test_predictions t USING (client_id)
        WHERE s.split = 'test'
    """).df()
    con.close()
    return df


def group_table(df, col, labels):
    df = df.assign(group=df[col].map(labels), approved=df["p_gbm"] < THRESHOLD)
    rows = []
    for g, d in df.groupby("group"):
        appr = d[d["approved"]]
        rows.append({
            "group": g,
            "n": len(d),
            "actual_default": d["default"].mean(),
            "mean_predicted": d["p_gbm"].mean(),
            "approval_rate": d["approved"].mean(),
            "default_among_approved": appr["default"].mean(),
        })
    table = pd.DataFrame(rows).set_index("group")
    table["approval_ratio_vs_max"] = table["approval_rate"] / table["approval_rate"].max()
    return table


def main():
    df = load_test()
    X, y = df[FEATURES], df["default"]
    model = joblib.load(MODEL_PATH)

    pi = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=10, random_state=SEED)
    imp = pd.DataFrame({"feature": FEATURES, "auc_drop_mean": pi.importances_mean, "auc_drop_std": pi.importances_std})
    imp = imp.sort_values("auc_drop_mean", ascending=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_df = imp.iloc[::-1]
    ax.barh(plot_df["feature"], plot_df["auc_drop_mean"], xerr=plot_df["auc_drop_std"])
    ax.set_xlabel("Drop in test AUC when feature is shuffled")
    ax.set_title("Permutation importance (GBM + isotonic, test set)")
    fig.tight_layout()
    PLOT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150)

    lines = [
        f"test rows: {len(df)}, threshold: {THRESHOLD:.3f}",
        "",
        "=== permutation importance (drop in AUC, 10 repeats) ===",
        imp.round(4).to_string(index=False),
        "",
        "=== fairness by sex (not a model input) ===",
        group_table(df, "sex", SEX_LABELS).round(3).to_string(),
        "",
        "=== fairness by marriage (not a model input; 0 mapped to other) ===",
        group_table(df, "marriage", MARRIAGE_LABELS).round(3).to_string(),
    ]
    text = "\n".join(lines)
    OUT_PATH.write_text(text)
    print(text)
    print(f"\nsaved to {OUT_PATH} and {PLOT_PATH}")


if __name__ == "__main__":
    main()
