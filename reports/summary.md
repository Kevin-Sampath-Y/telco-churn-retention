> Figures below were computed on the public **IBM Telco Customer Churn** dataset (7,043 customers).


### Model comparison (5-fold CV on training data, then hold-out test)

| Model | CV ROC-AUC | Test ROC-AUC | Test PR-AUC | Test Brier |
|---|---|---|---|---|
| Baseline (predict base rate) | 0.500 | 0.500 | 0.265 | 0.195 |
| Logistic regression | 0.846 | 0.841 | 0.627 | 0.139 |
| Random forest | 0.847 | 0.844 | 0.655 | 0.136 |
| XGBoost **(selected)** | 0.850 | 0.847 | 0.665 | 0.135 |

### Key findings

1. **The model ranks churners well.** Hold-out ROC-AUC 0.85 and PR-AUC 0.67 (guessing gives 0.27). Probabilities are well calibrated (Brier 0.135).
2. **A small group holds most of the risk.** The riskiest 10% of customers churn at 76% (2.9x the average); contacting the riskiest 30% reaches 68% of churners.
3. **Top drivers.** *Contract*, *tenure*, *InternetService* cause the biggest drop in accuracy when shuffled. This is association, not proof of causation.
4. **Segments.** Churn is highest for *Contract = Month-to-month* (43%) and lowest for *Two year* (3%).
5. **Business impact (back-test, assumptions below).** Targeting the 573 customers whose expected benefit is positive gives $23,644 net profit vs $6,717 for contacting everyone and $2,732 for a same-size random group.

*Assumptions: 12-month horizon, 50% margin, $30 per contact, 30% of would-be churners saved. These are NOT in the data; validate the save rate with an A/B test.*

### Decision at the operating threshold

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.19 | 0.47 | 0.86 | 0.61 |

