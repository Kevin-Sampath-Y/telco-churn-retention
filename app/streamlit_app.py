"""
Telco churn app: explore churn, score one customer, plan a retention campaign.

    streamlit run app/streamlit_app.py

Needs the artefacts created by `python run_pipeline.py` (model, scored customers, figures).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import business, config, data_prep, explain

st.set_page_config(page_title="Telco Churn Retention", page_icon=":telephone_receiver:", layout="wide")

RISK_COLORS = {"Low": "#2E8B57", "Medium": "#F28E2B", "High": "#C0392B"}


# ----------------------------------------------------------------------------- data loading
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return joblib.load(config.MODEL_FILE)


@st.cache_data(show_spinner=False)
def load_meta() -> dict:
    return json.loads(config.META_FILE.read_text())


@st.cache_data(show_spinner=False)
def load_scored() -> pd.DataFrame:
    df = pd.read_csv(config.SCORED_FILE)
    return df


def artefacts_ready() -> bool:
    return all(p.exists() for p in (config.MODEL_FILE, config.META_FILE, config.SCORED_FILE))


# ----------------------------------------------------------------------------- shared sidebar
def sidebar_assumptions() -> business.Assumptions:
    st.sidebar.header("Business assumptions")
    st.sidebar.caption("Not in the dataset: adjust them to your business. The save rate should come from an A/B test.")
    margin = st.sidebar.slider("Gross margin", 0.1, 0.9, 0.5, 0.05, format="%.2f")
    horizon = st.sidebar.slider("Value horizon (months)", 3, 36, 12)
    cost = st.sidebar.slider("Cost per contacted customer ($)", 5, 150, 30)
    save = st.sidebar.slider("Save rate (share of would-be churners kept)", 0.05, 0.80, 0.30, 0.05, format="%.2f")
    return business.Assumptions(horizon_months=horizon, gross_margin=margin, offer_cost=float(cost), save_rate=save)


def risk_band(p: float, thr: float) -> str:
    return "High" if p >= 0.5 else "Medium" if p >= thr else "Low"


# ----------------------------------------------------------------------------- pages
def page_overview(scored: pd.DataFrame, meta: dict) -> None:
    st.title("Customer churn overview")
    churned = scored[scored[config.TARGET] == 1]
    c = st.columns(4)
    c[0].metric("Customers", f"{len(scored):,}")
    c[1].metric("Churn rate", f"{scored[config.TARGET].mean():.1%}")
    c[2].metric("Monthly revenue lost to churn", f"${churned['MonthlyCharges'].sum():,.0f}")
    c[3].metric("Model ROC-AUC (hold-out)", f"{meta['test_metrics']['roc_auc']:.2f}")

    st.subheader("Where does churn concentrate?")
    col = st.selectbox("Split customers by", ["Contract", "InternetService", "PaymentMethod", "TechSupport",
                                              "OnlineSecurity", "PaperlessBilling", "SeniorCitizen", "Partner", "Dependents"])
    t = scored.groupby(col)[config.TARGET].agg(["mean", "count"]).rename(columns={"mean": "churn_rate", "count": "customers"})
    left, right = st.columns([2, 1])
    left.bar_chart(t["churn_rate"], horizontal=True)
    right.dataframe(t.style.format({"churn_rate": "{:.1%}", "customers": "{:,}"}))

    st.subheader("Tenure")
    img = config.FIGURES_DIR / "02_churn_by_tenure.png"
    if img.exists():
        st.image(str(img))


def _customer_form(scored: pd.DataFrame) -> pd.DataFrame:
    ids = ["Custom customer"] + scored[config.ID_COL].head(300).tolist()
    pick = st.selectbox("Start from", ids, help="Pick a real customer to pre-fill the form, then change anything.")
    base = scored[scored[config.ID_COL] == pick].iloc[0] if pick != ids[0] else scored[scored["Contract"] == "Month-to-month"].iloc[0]
    k = pick  # widget keys include the pick so defaults reset when it changes

    def sel(label, options, field):
        cur = base[field]
        return st.selectbox(label, options, index=options.index(cur) if cur in options else 0, key=f"{field}_{k}")

    a, b, c = st.columns(3)
    with a:
        tenure = st.slider("Tenure (months)", 0, 72, int(base["tenure"]), key=f"tenure_{k}")
        contract = sel("Contract", ["Month-to-month", "One year", "Two year"], "Contract")
        payment = sel("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                                         "Credit card (automatic)"], "PaymentMethod")
        paperless = sel("Paperless billing", ["Yes", "No"], "PaperlessBilling")
        monthly = st.number_input("Monthly charges ($)", 15.0, 130.0, float(base["MonthlyCharges"]), 0.5, key=f"mc_{k}")
    with b:
        internet = sel("Internet service", ["DSL", "Fiber optic", "No"], "InternetService")
        phone = sel("Phone service", ["Yes", "No"], "PhoneService")
        lines = "No phone service" if phone == "No" else sel("Multiple lines", ["No", "Yes"], "MultipleLines")
        senior = sel("Senior citizen", ["No", "Yes"], "SeniorCitizen")
        partner = sel("Partner", ["No", "Yes"], "Partner")
        dep = sel("Dependents", ["No", "Yes"], "Dependents")
    addons = {}
    with c:
        for name in config.ADDON_SERVICES:
            if internet == "No":
                addons[name] = "No internet service"
                st.text_input(name, "No internet service", disabled=True, key=f"{name}_{k}_off")
            else:
                addons[name] = sel(name, ["No", "Yes"], name)
    total = round(monthly * tenure, 2)
    row = pd.DataFrame([{
        "customerID": "form", "gender": "n/a", "SeniorCitizen": senior, "Partner": partner, "Dependents": dep,
        "tenure": tenure, "PhoneService": phone, "MultipleLines": lines, "InternetService": internet, **addons,
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total}])
    return data_prep.add_features(row)


def page_predict(scored: pd.DataFrame, meta: dict, a: business.Assumptions) -> None:
    st.title("Score a customer")
    row = _customer_form(scored)
    model = load_model()
    p = float(model.predict_proba(row[config.FEATURES])[:, 1][0])
    thr = meta["operating_threshold"]
    band = risk_band(p, thr)
    value = float(business.customer_value(row["MonthlyCharges"], a)[0])
    benefit = float(business.expected_benefit(p, row["MonthlyCharges"], a)[0])
    be = float(business.break_even_probability(row["MonthlyCharges"], a)[0])

    st.divider()
    c = st.columns(4)
    c[0].metric("Churn probability", f"{p:.0%}")
    c[1].markdown(f"**Risk band**<br><span style='font-size:1.8rem;color:{RISK_COLORS[band]}'>{band}</span>", unsafe_allow_html=True)
    c[2].metric("Value if retained", f"${value:,.0f}")
    c[3].metric("Expected net benefit of contacting", f"${benefit:,.0f}",
                delta="worth contacting" if benefit > 0 else "not worth contacting", delta_color="normal" if benefit > 0 else "off")
    st.caption(f"Contacting this customer pays off when churn probability is above {min(be, 1):.0%} "
               f"(break-even for their monthly charge under the sidebar assumptions).")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Why this score?")
        background = scored[config.FEATURES].sample(min(300, len(scored)), random_state=0)
        drivers = explain.top_drivers(model, row[config.FEATURES], background, k=8).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.barh(drivers["feature"], drivers["impact"], color=["#C0392B" if v > 0 else "#2E8B57" for v in drivers["impact"]])
        ax.axvline(0, color="#1B2430", lw=0.8)
        ax.set_xlabel("Effect on churn risk (SHAP, log-odds)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        st.pyplot(fig, clear_figure=True)
        st.caption("Red bars push the customer towards churning, green bars towards staying.")
    with right:
        st.subheader("Suggested actions")
        for text in business.recommend_actions(row.iloc[0]):
            st.markdown(f"- {text}")
        st.caption("Rules of thumb based on the strongest segments in the data, not a causal guarantee.")


def page_planner(scored: pd.DataFrame, meta: dict, a: business.Assumptions) -> None:
    st.title("Retention campaign planner")
    st.write("Back-test on every customer using **out-of-fold** predictions (each customer was scored by a model that never saw them).")
    strat = business.strategy_comparison(scored, a)
    curve = business.campaign_curve(scored, a)
    k_best = int(curve["cum_profit"].idxmax()) + 1

    c = st.columns(3)
    c[0].metric("Best campaign size", f"{k_best:,} customers")
    c[1].metric("Back-tested net profit", f"${curve['cum_profit'].max():,.0f}")
    c[2].metric("Contact everyone instead", f"${strat.set_index('strategy').loc['Contact everyone', 'net_profit']:,.0f}")

    k = st.slider("How many customers can your team contact?", 50, len(scored), min(k_best, len(scored)), 50)
    chosen = curve.head(k)
    st.write(f"Contacting the top **{k:,}** customers by expected benefit gives a back-tested profit of "
             f"**${chosen['real_benefit'].sum():,.0f}** and reaches **{int(chosen[config.TARGET].sum()):,}** actual churners "
             f"({chosen[config.TARGET].sum() / curve[config.TARGET].sum():.0%} of all).")

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(curve["k"], curve["cum_profit"], color="#1F4E79", lw=2.4)
    ax.axvline(k, color="#F28E2B", ls="--")
    ax.axhline(0, color="#1B2430", lw=0.8)
    ax.set_xlabel("Customers contacted (best first)")
    ax.set_ylabel("Net profit ($)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    st.pyplot(fig, clear_figure=True)

    st.subheader("Strategy comparison")
    st.dataframe(strat.style.format({"net_profit": "${:,.0f}", "customers_contacted": "{:,}"}), hide_index=True)

    st.subheader("Call list")
    cols = [config.ID_COL, "churn_probability", "MonthlyCharges", "Contract", "InternetService", "tenure", "exp_benefit"]
    lst = chosen[cols].copy()
    st.dataframe(lst.head(200).style.format({"churn_probability": "{:.0%}", "exp_benefit": "${:,.0f}", "MonthlyCharges": "${:,.2f}"}),
                 hide_index=True)
    st.download_button("Download call list (CSV)", lst.to_csv(index=False), "retention_call_list.csv", "text/csv")
    st.info("Ranking uses P(churn) x save rate x customer value - cost. Profit figures depend on the sidebar assumptions; "
            "the save rate must be confirmed with a controlled experiment before acting on the money numbers.")


def page_model(meta: dict) -> None:
    st.title("Model card")
    m = meta["test_metrics"]
    st.write(f"**Selected model:** {meta['selected_model']} · trained on {meta['n_train']:,} customers, "
             f"evaluated once on {meta['n_test']:,} held-out customers.")
    c = st.columns(5)
    for col, (lab, key) in zip(c, [("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"), ("Recall", "recall"),
                                    ("Precision", "precision"), ("Brier score", "brier")]):
        col.metric(lab, f"{m[key]:.2f}" if key != "brier" else f"{m[key]:.3f}")
    comp = pd.DataFrame(meta["comparison"])[["model", "cv_roc_auc", "test_roc_auc", "test_pr_auc", "test_brier"]]
    st.dataframe(comp, hide_index=True)
    for name, cap in [("07_roc_pr.png", "Ranking quality"), ("09_lift_gain.png", "Lift by risk decile"),
                      ("12_shap_summary.png", "What drives predictions"), ("08_calibration.png", "Calibration")]:
        p = config.FIGURES_DIR / name
        if p.exists():
            st.image(str(p), caption=cap)
    fair = config.REPORTS_DIR / "fairness_check.csv"
    if fair.exists():
        st.subheader("Group check")
        st.dataframe(pd.read_csv(fair), hide_index=True)
    st.subheader("Limitations")
    st.markdown(
        "- Trained on one telecom sample (7,043 customers, one snapshot): it will not transfer to another company unchanged.\n"
        "- Predicts **who is likely to churn**, not **who can be persuaded to stay**. An uplift model or A/B test is needed for that.\n"
        "- Profit figures rest on assumed margin, cost and save rate.\n"
        "- `gender` is excluded from the model; `SeniorCitizen` is included, so monitor group performance before any real use.\n"
        "- SHAP and importance show association, not causation.")


# ----------------------------------------------------------------------------- main
def main() -> None:
    if not artefacts_ready():
        st.error("Model files not found. Run `python run_pipeline.py` first, then reload this page.")
        st.stop()
    meta, scored = load_meta(), load_scored()
    if meta.get("synthetic"):
        st.warning("DEMO DATA: this app is showing a model trained on synthetic data. Re-run the pipeline on the real dataset.")
    page = st.sidebar.radio("Page", ["Overview", "Score a customer", "Retention planner", "Model card"])
    a = sidebar_assumptions()
    if page == "Overview":
        page_overview(scored, meta)
    elif page == "Score a customer":
        page_predict(scored, meta, a)
    elif page == "Retention planner":
        page_planner(scored, meta, a)
    else:
        page_model(meta)


main()
