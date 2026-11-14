"""
Automated checks. They use a small synthetic table with the Telco schema, so they run in
seconds, need no download and never touch your real data:

    pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from src import business, data_prep, evaluate, train
from src.config import CATEGORICAL, FEATURES, NUMERIC, TARGET
from src.synthetic import generate


@pytest.fixture(scope="module")
def raw():
    return generate(n=2500, seed=7)


@pytest.fixture(scope="module")
def df(raw):
    return data_prep.add_features(data_prep.clean(raw))


@pytest.fixture(scope="module")
def trained(df):
    return train.train(df, fast=True, verbose=False)


# ------------------------------------------------------------------ data cleaning
def test_blank_total_charges_become_zero_for_new_customers(raw):
    assert (raw["TotalCharges"].str.strip() == "").any()          # the quirk exists in the raw table
    d = data_prep.clean(raw)
    assert d["TotalCharges"].notna().all()
    assert (d.loc[d["tenure"] == 0, "TotalCharges"] == 0).all()


def test_target_is_binary_and_ids_unique(df):
    assert set(df[TARGET].unique()) <= {0, 1}
    assert df["customerID"].is_unique


def test_validation_rejects_missing_columns(raw):
    with pytest.raises(ValueError, match="missing columns"):
        data_prep.clean(raw.drop(columns=["Contract"]))


def test_validation_rejects_duplicate_ids(raw):
    bad = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="not unique"):
        data_prep.clean(bad)


def test_blank_total_charges_for_established_customer_is_an_error(raw):
    bad = raw.copy()
    i = bad.index[bad["tenure"] > 5][0]
    bad.loc[i, "TotalCharges"] = " "
    with pytest.raises(ValueError, match="tenure > 0"):
        data_prep.clean(bad)


def test_cleaning_does_not_modify_input(raw):
    before = raw.copy()
    data_prep.clean(raw)
    pd.testing.assert_frame_equal(raw, before)


# ------------------------------------------------------------------ features
def test_engineered_features(df):
    assert df["n_addons"].between(0, 6).all()
    new = df[df["tenure"] == 0]
    assert np.allclose(new["avg_monthly_charge"], new["MonthlyCharges"])     # no divide-by-zero
    assert set(df["autopay"].unique()) <= {"Yes", "No"}
    assert np.isfinite(df[NUMERIC].to_numpy(dtype=float)).all()


def test_gender_is_not_a_model_feature():
    assert "gender" not in FEATURES and "customerID" not in FEATURES and TARGET not in FEATURES


# ------------------------------------------------------------------ modelling
def test_train_test_split_is_disjoint_and_stratified(df):
    X_tr, X_te, y_tr, y_te = train.split(df)
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert len(X_tr) + len(X_te) == len(df)
    assert abs(y_tr.mean() - y_te.mean()) < 0.01


def test_model_beats_baseline_and_probabilities_valid(trained):
    m = trained["meta"]
    comp = {c["model"]: c for c in m["comparison"]}
    assert comp[m["selected_model"]]["test_roc_auc"] > 0.70
    assert comp["Baseline (predict base rate)"]["test_roc_auc"] == pytest.approx(0.5)
    p = trained["model"].predict_proba(trained["X_test"])[:, 1]
    assert ((p >= 0) & (p <= 1)).all()


def test_selection_never_uses_test_scores(trained):
    best_cv = max((c for c in trained["meta"]["comparison"] if not c["model"].startswith("Baseline")),
                  key=lambda c: c["cv_roc_auc"])
    assert trained["meta"]["selected_model"] == best_cv["model"]


def test_out_of_fold_scores_cover_every_customer_once(trained, df):
    s = trained["scored"]
    assert len(s) == len(df) and s["customerID"].is_unique
    assert s["churn_probability"].between(0, 1).all()
    assert set(s["split"]) == {"train", "test"}


@pytest.mark.filterwarnings("ignore:Found unknown categories")
def test_unseen_category_does_not_crash(trained):
    row = trained["X_test"].iloc[[0]].copy()
    row["PaymentMethod"] = "Crypto"                         # never seen in training
    p = trained["model"].predict_proba(row)[:, 1]
    assert np.isfinite(p).all()


def test_metrics_helper_matches_manual_counts():
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([.9, .4, .3, .2, .8, .6])
    m = evaluate.metrics_at(y, p, 0.5)
    assert m["confusion"] == [[2, 1], [1, 2]] and m["recall"] == pytest.approx(2 / 3, abs=1e-3)


def test_decile_table_shape(trained):
    p = trained["model"].predict_proba(trained["X_test"])[:, 1]
    t = evaluate.decile_table(trained["y_test"], p)
    assert len(t) == 10 and t["customers"].sum() == len(p)
    assert t["cum_capture"].iloc[-1] == pytest.approx(1.0)
    assert t["churn_rate"].iloc[0] >= t["churn_rate"].iloc[-1]       # riskiest decile churns more


# ------------------------------------------------------------------ business logic
A = business.Assumptions()


def test_zero_save_rate_means_every_contact_loses_money():
    a = business.Assumptions(save_rate=0.0)
    assert (business.expected_benefit([0.1, 0.9], [50, 100], a) == -a.offer_cost).all()


def test_break_even_probability_gives_zero_benefit():
    monthly = 80.0
    p = business.break_even_probability(monthly, A)
    assert business.expected_benefit(p, monthly, A) == pytest.approx(0.0)


def test_benefit_increases_with_probability_and_value():
    assert business.expected_benefit(0.8, 70, A) > business.expected_benefit(0.2, 70, A)
    assert business.expected_benefit(0.5, 100, A) > business.expected_benefit(0.5, 40, A)


def test_model_targeting_beats_random_and_everyone(trained):
    s = trained["scored"][trained["scored"]["split"] == "test"]
    t = business.strategy_comparison(s, A).set_index("strategy")["net_profit"]
    model = t["Model: contact only where expected benefit > 0"]
    assert model > t["Contact a random group of the same size"]
    assert model > t["Do nothing"]


def test_campaign_curve_is_sorted_by_expected_benefit(trained):
    c = business.campaign_curve(trained["scored"], A)
    assert c["exp_benefit"].is_monotonic_decreasing
    assert c["cum_profit"].iloc[-1] == pytest.approx(c["real_benefit"].sum())


def test_recommend_actions_always_returns_something():
    row = pd.Series({"Contract": "Two year", "InternetService": "DSL", "TechSupport": "Yes",
                     "PaymentMethod": "Mailed check", "tenure": 60, "n_addons": 3})
    assert business.recommend_actions(row)
