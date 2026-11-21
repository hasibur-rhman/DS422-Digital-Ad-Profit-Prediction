# DS422 Machine Learning Lab Project: Digital Ad Profit Prediction
**Course:** DS422 — Machine Learning I  
**Evaluation Date:** 22nd November, 2026  
**Dataset:** Digital Advertising Campaign Performance Dataset (2024–2026)  
**Dataset Source:** [Kaggle](https://www.kaggle.com/datasets/juniornsa/digital-advertising-campaign-performance-dataset) (Author: Junior Nsa, 2026)  
**Target Variable:** `profit` (USD, defined as `revenue - ad_spend`)  

---

## 1. Executive Summary & Verified Metrics

| Metric Category | Verified Value | Analytical Interpretation |
| :--- | :--- | :--- |
| **Total Campaigns** | 10,000 campaigns | Full 2-year sample (Jan 2024 – Jan 2026) |
| **Original Schema** | 41 columns (35 base + 6 derived) | 0 missing values, 0 duplicates |
| **Profitable Campaigns** | 78.31% ($7,831$ campaigns) | Net positive cash flow ($> \$0$) |
| **Loss-Making Campaigns** | 21.69% ($2,169$ campaigns) | Negative return campaigns ($< \$0$) |
| **Target Profit Mean** | $\$24,070.20$ | Substantially pulled by viral mega-campaigns |
| **Target Profit Median** | $\$3,302.54$ | Typical median campaign baseline profit |
| **Target Profit Range** | $-\$36,939.84$ to $\$2,750,398.00$ | Positive skewness ($13.03$), leptokurtic ($265.43$) |
| **Zero Conversions Rate**| 5.94% | Complete ad burn without single user conversion |

---

## 2. Rigorous Data Leakage & Feature Selection Audit

To prevent synthetic over-optimism and ensure realistic deployment generalization, **13 columns were strictly dropped** prior to feature matrix construction:
1. `profit`: Target variable.
2. `revenue` & `ad_spend`: Mathematical identity leakage ($profit = revenue - ad_spend$).
3. `clicks` & `conversions`: Post-campaign outcome metrics that mechanically correlate with revenue.
4. `actual_cpc`: Exact duplicate of derived CPC column ($100.0\%$ identity).
5. `CTR`, `CPC`, `conversion_rate`, `CPA`, `ROAS`: Derived post-hoc marketing KPIs.
6. `campaign_id`: High-cardinality unique identifier.
7. `start_date`: Raw timestamp string (temporal signals captured by `quarter`, `day_of_week`, `campaign_day`).

**Remaining Actionable Predictors:** **28 features** (9 numeric + 19 categorical) $\rightarrow$ **78 one-hot encoded features**.

---

## 3. Comprehensive Model Evaluation Table (Sorted by Test $R^2$)

| model_name | slug | train_r2 | test_r2 | test_mae_usd | test_mse | test_rmse_usd | cv_5fold_mean_r2 | cv_5fold_std_r2 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| XGBoost | xgboost | 0.8588 | 0.3414 | $21,982.56 | $6,318,963,083.71 | $79,491.91 | 0.4058 | 0.0826 |
| Random Forest | random_forest | 0.8808 | 0.3327 | $24,457.87 | $6,401,809,858.27 | $80,011.31 | 0.3051 | 0.0621 |
| Lasso Regression | lasso_regression | 0.2442 | 0.1913 | $31,896.25 | $7,758,979,417.93 | $88,085.07 | 0.2276 | 0.0272 |
| Ridge Regression | ridge_regression | 0.2442 | 0.1913 | $31,859.09 | $7,759,059,842.75 | $88,085.53 | 0.2277 | 0.0271 |
| Multiple Linear Regression | linear_regression | 0.2443 | 0.1910 | $31,971.88 | $7,761,507,191.22 | $88,099.42 | 0.2272 | 0.0273 |
| Decision Tree | decision_tree | 0.5799 | -0.1572 | $28,909.31 | $11,101,976,246.24 | $105,365.92 | -0.2404 | 0.1570 |

### Hyperparameter Tuned Model (Tuned XGBoost):
- **Best Parameters:** `{'subsample': 0.95, 'reg_lambda': 3.0, 'reg_alpha': 1.0, 'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.1, 'colsample_bytree': 0.85}`
- **Test $R^2$:** **0.3726**
- **Test MAE:** **$22,545.67**
- **Test RMSE:** **$77,586.79**
- **5-Fold CV $R^2$:** **0.3940**

---

## 4. Key Marketing & Business Insights

1. **Purchase Intent is the Strongest Profit Lever:** High-intent audiences generate an average profit of $\$47,652.10$ vs. $\$10,294.34$ for low-intent audiences (a **4.6x profitability multiplier**).
2. **Retargeting Yields Massive Premium:** Retargeted campaigns average $\$37,040.07$ vs $\$19,815.81$ for cold prospecting (**+86.9% lift**).
3. **Desktop Superiority in High-Ticket Tech:** Desktop ad campaigns average $\$44,985.04$ vs $\$11,848.56$ for Mobile (**3.8x higher profit**).
4. **Creative Format Hierarchy:** Video ($\$35,242.08$) and Interactive ($\$27,575.92$) formats vastly outperform static Text ($\$8,928.76$).
5. **Q4 Seasonality:** Holiday shopping in Q4 averages $\$36,850.91$ profit per campaign, nearly double Q3 ($\$18,593.55$).

---

## 5. Artifacts and Generated Figures
All 9 figures saved at 300 DPI:
- `01_profit_distribution.png`
- `02_categorical_profit_drivers.png`
- `03_platform_and_seasonality.png`
- `04_correlation_heatmap.png`
- `05_outlier_boxplots.png`
- `06_model_comparison.png`
- `07_actual_vs_predicted.png`
- `08_residual_diagnostics.png`
- `09_feature_importance.png`
