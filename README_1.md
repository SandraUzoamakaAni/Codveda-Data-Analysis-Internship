# Codveda Data Analysis Internship — Project Summary
**Sandra Uzoamaka Ani**

## Level 1 (Basic) — House Price Dataset

**Data Cleaning:** Loaded the Boston Housing data (no header row, whitespace-delimited despite the `.csv` extension — a real inconsistent-format issue), verified no missing values or duplicates, cast the `CHAS` river-adjacency flag to a proper binary type.

**EDA:** Home value (`MEDV`) is right-skewed, median ~$21,200. The two strongest relationships with price:
- **% lower-status population (LSTAT): -0.74 correlation** — the single strongest predictor
- **Average rooms (RM): +0.70 correlation** — more rooms, higher value, as expected

**Business insight:** Neighborhood socioeconomic composition predicts home value more strongly than almost any physical property feature — useful framing for a real-estate pricing or investment tool.

## Level 2 (Intermediate)

**Regression (house prices):** Linear regression explains **66.9%** of price variance (R²=0.669), average error ~$4,900. Pollution (NOX) and room count (RM) are the strongest price levers in the model.

**Clustering (iris):** K-Means with k=3 (elbow-confirmed) nearly perfectly recovered the three true flower species from measurements alone — 50/50 correct on setosa, ~89% correct on the other two. **Business insight:** shows clustering can recover meaningful real-world groupings from features alone, without labels — the same technique businesses use for customer segmentation.

## Level 3 (Advanced) — Telecom Churn

**Classification:** Compared Logistic Regression, Decision Tree, and tuned Random Forest. The tuned Random Forest won clearly:

| Metric | Score |
|---|---|
| Accuracy | 95.5% |
| Precision | 92.2% |
| Recall | 74.7% |
| F1-score | 82.6% |

**Top churn drivers:** Total day minutes, customer service calls, day charge, and having an international plan.

**Business insight:** Customers who call support frequently *and* have an international plan are the highest churn risk — that's a specific, actionable segment for a retention campaign, not a vague "engaged customers churn less" statement.

**Dashboard:** Built in Power BI/Tableau from `churn_dashboard_data.csv` — see `DASHBOARD_GUIDE.md`. Surfaces churn rate by state and plan type for filtering by the business.

---

## Deliverables checklist
- [x] `Level1_Basic.py` — cleaning + EDA code
- [x] `Level2_Intermediate.py` — regression + clustering code
- [x] `Level3_Advanced.py` — classification + dashboard data prep code
- [x] `house_data_cleaned.csv`, `churn_dashboard_data.csv` — cleaned datasets
- [x] 7 plots in `/plots`
- [x] Dashboard (build from guide — the one manual step left)
- [ ] LinkedIn post + video (per Codveda's instructions) — talking points above
- [ ] Submit via the Codveda submission form
