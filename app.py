import sys
from pathlib import Path

import altair as alt
import duckdb
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from pipelines import FEATURES  # noqa: E402

DB_PATH = ROOT / "data" / "credit.duckdb"
MODEL_DIR = ROOT / "models"
MODELS = {
    "Gradient boosting + isotonic calibration": "gbm_isotonic",
    "Logistic regression (benchmark)": "logistic_regression",
}
THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.005), 3)


def table_exists(name):
    if not DB_PATH.exists():
        return False
    con = duckdb.connect(str(DB_PATH), read_only=True)
    found = name in [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    con.close()
    return found


@st.cache_resource(show_spinner="First run: downloading the data and building features…")
def ensure_data():
    if not table_exists("clients"):
        import load_data
        load_data.main()
    if not table_exists("features"):
        import build_features
        build_features.main()
    if not table_exists("split"):
        import split_data
        split_data.main()
    return True


@st.cache_resource
def load_models():
    return {key: joblib.load(MODEL_DIR / f"{key}.joblib") for key in MODELS.values()}


@st.cache_data
def load_scored_test():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("""
        SELECT f.*
        FROM features f
        JOIN split s USING (client_id)
        WHERE s.split = 'test'
        ORDER BY client_id
    """).df()
    con.close()
    models = load_models()
    for key, model in models.items():
        df[f"p_{key}"] = model.predict_proba(df[FEATURES])[:, 1]
    return df


def profit_curve(p, y, ead, r, lgd):
    gain = np.where(y == 0, r * ead, -lgd * ead)
    approved = p[None, :] < THRESHOLDS[:, None]
    n_approved = approved.sum(axis=1)
    defaults_approved = (approved & (y[None, :] == 1)).sum(axis=1)
    return pd.DataFrame({
        "threshold": THRESHOLDS,
        "profit_m": (approved * gain[None, :]).sum(axis=1) / 1e6,
        "approval_rate": n_approved / len(y),
        "default_rate_approved": np.divide(defaults_approved, n_approved,
                                           out=np.full(len(THRESHOLDS), np.nan),
                                           where=n_approved > 0),
    })


st.set_page_config(page_title="Credit risk decisioning", layout="wide")
st.title("Credit risk decisioning: where should the lender draw the line?")
st.caption(
    "A default model scores 6,000 held-out card clients. Choose the approval threshold and the "
    "economics to see the approval rate, default rate and profit the policy would have produced."
)

try:
    ensure_data()
except Exception as e:
    st.error("Could not download the UCI dataset on first start, so the app cannot score clients yet. "
             f"Please reload in a few minutes. ({type(e).__name__}: {e})")
    st.stop()
df = load_scored_test()
y = df["default"].to_numpy()
ead = df["limit_bal"].to_numpy(dtype=float)

with st.sidebar:
    st.header("Policy settings")
    model_label = st.selectbox("Model", list(MODELS))
    r = st.slider("Net interest margin (r)", 0.05, 0.30, 0.15, 0.01,
                  help="Profit earned on each client who repays, as a share of their credit limit.")
    lgd = st.slider("Loss given default (LGD)", 0.30, 1.00, 0.75, 0.05,
                    help="Share of the credit limit lost when a client defaults.")
    breakeven = r / (r + lgd)
    # Fixed labels and keys keep these widgets (and the user's chosen threshold) stable when r or LGD changes.
    use_breakeven = st.checkbox("Use break-even threshold r/(r+LGD)", value=True, key="use_breakeven")
    st.caption(f"Break-even threshold at these settings: {breakeven:.3f}")
    if use_breakeven:
        threshold = breakeven
        st.session_state["manual_threshold"] = round(breakeven, 3)
    else:
        threshold = st.slider("Approval threshold (approve if PD < threshold)", 0.01, 0.99, step=0.001,
                              key="manual_threshold", help="PD = predicted probability of default.")

p = df[f"p_{MODELS[model_label]}"].to_numpy()
approved = p < threshold
gain = np.where(y == 0, r * ead, -lgd * ead)
profit = gain[approved].sum() / 1e6
approve_all = gain.sum() / 1e6
approval_rate = approved.mean()
default_approved = y[approved].mean() if approved.any() else float("nan")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Approval rate", f"{approval_rate:.1%}")
c2.metric("Default rate (approved)", f"{default_approved:.1%}",
          delta=f"{default_approved - y.mean():+.1%} vs all clients", delta_color="inverse")
c3.metric("Profit (NT$ m)", f"{profit:,.1f}")
c4.metric("Gain vs approve-all", f"{profit - approve_all:+,.1f}m",
          help=f"Profit if every client is approved: NT$ {approve_all:,.1f}m.")

curve = profit_curve(p, y, ead, r, lgd)
best = curve.loc[curve["profit_m"].idxmax()]

line = alt.Chart(curve).mark_line().encode(
    x=alt.X("threshold:Q", title="Approval threshold (approve if predicted PD < threshold)"),
    y=alt.Y("profit_m:Q", title="Profit on test set (NT$ m)"),
    tooltip=[alt.Tooltip("threshold:Q", format=".3f"),
             alt.Tooltip("profit_m:Q", title="profit (m)", format=",.1f"),
             alt.Tooltip("approval_rate:Q", title="approval", format=".1%"),
             alt.Tooltip("default_rate_approved:Q", title="default rate among approved", format=".1%")],
)
marker = alt.Chart(pd.DataFrame({"threshold": [threshold]})).mark_rule(color="red", strokeDash=[4, 4]).encode(x="threshold:Q")
baseline = alt.Chart(pd.DataFrame({"y": [approve_all]})).mark_rule(color="grey").encode(y="y:Q")
st.subheader("Profit across thresholds")
st.altair_chart((line + baseline + marker).properties(height=360), width="stretch")
st.caption(
    f"Red dashed line: your threshold ({threshold:.3f}). Grey line: profit if everyone is approved. "
    f"Most profitable threshold on this test set: {best['threshold']:.3f} (NT$ {best['profit_m']:,.1f}m)."
)

with st.expander("Method and assumptions"):
    st.markdown(f"""
- **Data:** UCI *Default of Credit Card Clients* (Yeh & Lien, 2009): 30,000 card clients of a Taiwanese bank in 2005, licensed CC BY 4.0.
- **Target:** default in October 2005, predicted from April–September repayment history. The overall default rate is {y.mean():.1%}.
- **Model:** gradient boosting with isotonic calibration, chosen by 5-fold cross-validation, with logistic regression kept as an explainable benchmark.
- **Evaluation:** 6,000 clients were held out before any modelling and used only for this evaluation.
- **Profit:** a client who repays earns r × their credit limit; a client who defaults costs LGD × their credit limit. A client is approved when the expected profit is positive, i.e. when PD < r/(r+LGD).
- **Simplifications:** exposure equals the credit limit, a single period is modelled, and the economics are illustrative rather than a real lender's.
- **Fairness:** sex and marital status are excluded from the model and audited separately (see the README).
""")
