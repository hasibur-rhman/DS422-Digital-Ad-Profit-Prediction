# Digital Advertising Campaign Profit Prediction & Analytics Command Center
**Course:** DS422 — Machine Learning I  
**Evaluation Date:** 22nd November, 2026  
**Lead Author:** Hasibur Rahman  
**Repository:** `DS422-Digital-Ad-Profit-Prediction`  

[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-success?style=for-the-badge&logo=github)](https://hasibur-rhman.github.io/DS422-Digital-Ad-Profit-Prediction/)

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-v1.4+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-GPU--Accelerated-green.svg)](https://xgboost.readthedocs.io/)
[![Google Colab](https://img.shields.io/badge/Colab-Tesla%20T4%20Ready-yellow.svg)](https://colab.research.google.com/)
[![Tailwind CSS](https://img.shields.io/badge/UI-Tailwind%20CSS%20CDN-38bdf8.svg)](https://tailwindcss.com/)
[![Status](https://img.shields.io/badge/Status-Evaluation%20Ready-brightgreen.svg)]()

> 🚀 **Live Interactive Web Simulator:** [https://hasibur-rhman.github.io/DS422-Digital-Ad-Profit-Prediction/](https://hasibur-rhman.github.io/DS422-Digital-Ad-Profit-Prediction/)

---

## 📌 Executive Summary

Digital advertising expenditures frequently suffer from budget misallocation, negative return-on-ad-spend (ROAS), and high cash-burn before campaigns reach algorithmic maturation. This project delivers an end-to-end, production-grade **Machine Learning Regression System and Executive Decision Command Center** designed to predict **Campaign Net Profit in USD** ($\text{revenue} - \text{ad\_spend}$) *prior to campaign launch*.

Using a comprehensive dataset of **10,000 digital advertising campaigns (2024–2026)** calibrated to 2025 digital advertising industry benchmarks, we:
1. Conducted an exhaustive exploratory data audit and multicollinearity verification.
2. Enforced strict pre-launch feature isolation by eliminating **13 data leakage and post-campaign outcome columns**.
3. Benchmarked **6 distinct regression architectures** alongside an extensively tuned ensemble.
4. Achieved champion performance with **Tuned XGBoost Regressor** ($\text{Test } R^2 = \mathbf{0.3726}$, $\text{Test RMSE} = \mathbf{\$77,586.79}$, 5-Fold $\text{CV } R^2 = \mathbf{0.3940}$).
5. Engineered a **zero-dependency, single-page interactive web application** featuring a real-time client-side campaign profit simulator and risk calculator.
6. Implemented a **disconnect-safe Google Colab architecture** with persistent Google Drive checkpoints (`/content/drive/MyDrive/Lab_EVAL_E1`) and sub-10-second recovery.

---

## 🏗️ Project Architecture & Workflow

```
                                      [ Raw Ad Dataset: 10,000 x 41 ]
                                                     │
                                                     ▼
                                     [ Strict Data Leakage Audit ]
                                   Drop 13 Leakage & Post-Hoc Cols
                                                     │
                                                     ▼
                                    [ Actionable Feature Matrix ]
                                 28 Pre-Launch Predictors (9 Num / 19 Cat)
                                                     │
                                                     ▼
                                     [ Preprocessing & 80/20 Split ]
                                     StandardScaler + OneHotEncoder
                                        78 Encoded Features Matrix
                                                     │
               ┌─────────────────────────────────────┴─────────────────────────────────────┐
               ▼                                                                           ▼
    [ Model Benchmarking Arena ]                                            [ Executive Command Center ]
    ├── Multiple Linear Regression                                          ├── Executive KPI Ribbon
    ├── Ridge Regression (L2)                                               ├── Leakage Defense Matrix
    ├── Lasso Regression (L1)                                               ├── 9-Figure EDA Gallery (300 DPI)
    ├── Decision Tree Regressor                                             ├── Model Leaderboard Table
    ├── Random Forest Regressor                                             ├── Live Profit Simulator
    ├── Default XGBoost (T4 GPU)                                            ├── Feature Importance Gain Chart
    └── Tuned XGBoost (RandomizedSearchCV) ──► Best: R² = 0.3726            └── Download Center
```

---

## 🛡️ Critical Domain & Data Leakage Prevention

A central academic defense for this DS422 project is the rigorous prevention of **target leakage** and **post-hoc contamination**. In practical marketing, optimizing models using post-campaign outcome metrics inflates accuracy artificially while providing zero decision value before capital is committed.

### Dropped Columns Breakdown (13 Excluded Attributes)

| Category | Dropped Features | Exact Rationale & Viva Justification |
| :--- | :--- | :--- |
| **Response Variable** | `profit` | Ground-truth continuous target vector ($y$). |
| **Direct Mathematical Identity** | `revenue`, `ad_spend` | $\text{profit} = \text{revenue} - \text{ad\_spend}$. Including either feature creates a trivial mathematical identity, causing artificial $R^2 = 1.000$ memorization. |
| **Post-Campaign Mechanical Outcomes** | `clicks`, `conversions` | Can only be measured after ad impression delivery has concluded. Mechanically tie into conversion volume. |
| **100% Duplicate Column** | `actual_cpc` | Exact duplicate of derived CPC column ($r = 1.000$ across all 10,000 rows). |
| **Derived Post-Hoc KPIs** | `CTR`, `CPC`, `conversion_rate`, `CPA`, `ROAS` | Calculated downstream performance ratios requiring post-campaign metrics. |
| **High-Cardinality Identifiers** | `campaign_id` | Random unique hash string; non-generalizable memorization risk. |
| **Unparsed Timestamp** | `start_date` | Raw timestamp string. Temporal variation is cleanly captured via `quarter`, `month`, `day_of_week`, and `campaign_day`. |

### Final Predictors
* **Continuous Predictors (9):** `creative_age_days`, `quarter`, `hour_of_day`, `campaign_day`, `quality_score`, `impressions`, `bounce_rate`, `avg_session_duration_seconds`, `pages_per_session`.
* **Categorical Predictors (19):** `campaign_objective`, `platform`, `ad_placement`, `device_type`, `operating_system`, `creative_format`, `creative_size`, `ad_copy_length`, `has_call_to_action`, `creative_emotion`, `target_audience_age`, `target_audience_gender`, `audience_interest_category`, `income_bracket`, `purchase_intent_score`, `retargeting_flag`, `day_of_week`, `industry_vertical`, `budget_tier`.
* **Multicollinearity (VIF):** Computed via OLS linear regression ($VIF_j = \frac{1}{1 - R_j^2}$). All continuous features scored $VIF < 1.0003$, far below the critical threshold of $5.0$.

---

## 📊 Exploratory Data Analysis & Analytical Insights

The dataset exhibits realistic commercial ad performance dynamics across 10,000 campaigns:

* **Net Profit Breakdown:** **78.31% profitable** (7,831 campaigns) vs. **21.69% loss-making** (2,169 campaigns).
* **Zero Conversion Rate:** **5.94%** (594 campaigns) experienced complete ad burn without generating a single customer conversion.
* **Skewness & Dispersion:** Mean profit is **\$24,070.20** with median of **\$3,302.54**, reflecting positive skewness ($13.03$) and heavy right-tail enterprise viral campaigns ranging up to **\$2,750,398.00**. Minimum net return was **-\$36,939.84**.
* **Channel Performance:** **LinkedIn** and **YouTube** delivered the highest average net return (\$26,800+), while cold-prospecting mobile display ads showed higher downside margin compression.
* **Audience Intent:** Warm **retargeting** campaigns generated over **+\$14,500** in median net return compared to cold-audience prospecting.

All exploratory analyses were compiled into 9 publication-grade **300 DPI** diagnostic charts located in `figures/`:
1. `01_profit_distribution.png`: Histogram & KDE density with $0 break-even line.
2. `02_categorical_profit_drivers.png`: Multi-panel intent score, format, and device breakdowns.
3. `03_platform_and_seasonality.png`: Quarterly performance cycles across major platforms.
4. `04_correlation_heatmap.png`: Full correlation matrix of valid continuous predictors.
5. `05_outlier_boxplots.png`: IQR continuous outlier distributions.
6. `06_model_comparison.png`: 6-Model $R^2$, RMSE, and MAE benchmark bar charts.
7. `07_actual_vs_predicted.png`: Test set actual vs. predicted scatter with reference line.
8. `08_residual_diagnostics.png`: Residual density and homoscedasticity diagnostics.
9. `09_feature_importance.png`: Top 15 XGBoost Gain-based feature attribution.

---

## 🏆 Model Benchmarking & Performance Arena

All models were evaluated on the identical 80/20 train-test split (`random_state=42`, 8,000 train / 2,000 test) using 5-fold cross-validation on the training set:

| Rank | Model Architecture | Train $R^2$ | Test $R^2$ | Test MAE (USD) | Test RMSE (USD) | 5-Fold CV $R^2 \pm \text{Std}$ | Overfitting $\Delta$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **🏆 #1** | **Tuned XGBoost Regressor** | **0.8455** | **0.3726** | **\$22,545.67** | **\$77,586.79** | **$0.3940 \pm 0.0815$** | **0.4729** |
| #2 | **Default XGBoost Regressor** | 0.8588 | 0.3414 | \$21,982.56 | \$79,491.91 | $0.4058 \pm 0.0826$ | 0.5174 |
| #3 | **Random Forest Regressor** | 0.8808 | 0.3327 | \$24,457.87 | \$80,011.31 | $0.3051 \pm 0.0621$ | 0.5481 |
| #4 | **Lasso Regression (L1)** | 0.2442 | 0.1913 | \$31,896.25 | \$88,085.07 | $0.2276 \pm 0.0272$ | 0.0529 |
| #5 | **Ridge Regression (L2)** | 0.2442 | 0.1913 | \$31,859.09 | \$88,085.53 | $0.2277 \pm 0.0271$ | 0.0529 |
| #6 | **Multiple Linear Regression** | 0.2443 | 0.1910 | \$31,971.88 | \$88,099.42 | $0.2272 \pm 0.0273$ | 0.0533 |
| #7 | **Decision Tree Regressor** | 0.5799 | **-0.1572** | \$28,909.31 | \$105,365.92 | **$-0.2404 \pm 0.1570$** | 0.7371 |

### Key Analytical Takeaways for Lab Defense

1. **Why Did Decision Tree Fail ($R^2 = -0.1572$)?**  
   Decision Trees partition feature space with orthogonal step functions that struggle with heavy-tailed distributions ($skew = 13.03$). Test observations fall into leaf bins with unrepresentative means, yielding total residual variance higher than the sample mean variance ($R^2 < 0$).
2. **Why Did Linear Models Underperform ($R^2 \approx 0.1913$)?**  
   Linear, Ridge, and Lasso regressions assume linear additive relationships. Ad-tech auctions exhibit strong non-linear interactions (e.g., retargeting combined with video creative and desktop placement yields synergistic lift).
3. **The Power of Hyperparameter Tuning on XGBoost:**  
   Applying `RandomizedSearchCV` (15 candidates × 5 folds = 75 fits) constrained tree depth to `max_depth=4`, set `n_estimators=200`, `learning_rate=0.1`, and applied regularizers `reg_alpha=1.0` and `reg_lambda=3.0`. This boosted Test $R^2$ from **0.3414 to 0.3726** (+9.1%) and reduced test error by **-\$1,905.12** (-2.4%).

---

## 💻 Live Campaign Profit Simulator & Web Application

The project includes a production-ready, ultra-clean web application (`index.html`) deployed live on **GitHub Pages**:

🔗 **Live URL:** [https://hasibur-rhman.github.io/DS422-Digital-Ad-Profit-Prediction/](https://hasibur-rhman.github.io/DS422-Digital-Ad-Profit-Prediction/)

* **Ultra-Clean, Minimal Design:** Focused strictly on pre-launch media planning controls and instant profit calculation.
* **Pre-Launch Media Planning Controls:**
  - Advertising Platform (LinkedIn, YouTube, Google Ads, Meta, TikTok, Twitter, Pinterest)
  - Industry Vertical (Finance, Technology, Healthcare, E-Commerce, Education, Automotive)
  - Creative Format (Video, Carousel, Static Image, Text)
  - Target Device (Desktop, Mobile, Tablet)
  - Impression Scale (Interactive Slider: 10,000 to 400,000)
  - Flight Duration (Interactive Slider: 7 to 90 Days)
  - Quality Score (Interactive Slider: 1.0 to 10.0)
  - Audience Segment (Retargeting vs. Cold Prospecting)
  - Income Bracket (> $100K, $50K–$100K, < $50K)
  - Purchase Intent (High, Medium, Low)
* **Real-Time Client-Side Inference:** Instantly computes **Predicted Net Campaign Profit ($USD)** with dynamic color coding (Gold for high profit, Sky for positive margin, Rose for negative burn).

---

## 🚀 How to Run the Web Application

### Option 0: Live Cloud Deployment (No Installation Required)
Access the live simulator directly in any modern browser:  
👉 **[https://hasibur-rhman.github.io/DS422-Digital-Ad-Profit-Prediction/](https://hasibur-rhman.github.io/DS422-Digital-Ad-Profit-Prediction/)**

The web application is also **100% standalone and zero-dependency** for local execution (no Node.js build steps, Webpack, or external bundlers required). All stylesheets and icons load via CDN or inline SVGs.

### Option 1: Instant Browser Launch (Zero-Setup)
Simply navigate to the project directory and open `index.html` directly:
* **macOS:** Double click `index.html` or run:
  ```bash
  open index.html
  ```
* **Windows:** Double click `index.html` or run in CMD:
  ```cmd
  start index.html
  ```
* **Linux:** Run:
  ```bash
  xdg-open index.html
  ```

### Option 2: Python HTTP Server (Recommended)
Launch a local lightweight web server from the project directory:
```bash
# Python 3
python3 -m http.server 8000
```
Open your browser and navigate to:
```
http://localhost:8000
```

### Option 3: Node.js / npx
If you have Node.js installed, run:
```bash
npx serve .
# or
npx http-server -p 8000
```

### Option 4: VS Code Live Server
1. Open the project folder in **Visual Studio Code**.
2. Install the **Live Server** extension (by Ritwick Dey).
3. Right-click `index.html` and select **"Open with Live Server"**.

---

## ⚡ Google Colab & Disconnect-Safe Cloud Execution

The Jupyter Notebook [`DS422_Digital_Ad_Profit_Prediction.ipynb`](./DS422_Digital_Ad_Profit_Prediction.ipynb) is engineered for **Google Colab with Tesla T4 GPU Runtime** with built-in disconnect resilience:

1. **Dual-Environment Detection:** Automatically detects whether it is running on Colab or a local machine.
2. **Drive Persistence:** Mounts Google Drive and sets project root strictly to:
   ```python
   BASE_DIR = "/content/drive/MyDrive/Lab_EVAL_E1"
   ```
3. **Dataset Auto-Locate:** If `tech_advertising_campaigns_dataset.csv` is not in Drive, prompts for upload once and automatically archives it to Drive so it is never lost on restart.
4. **Sub-10-Second Disconnect Recovery:**
   * Preprocessed data, split matrices, and all 6 regression models + tuned booster are checkpointed as `.joblib` and native `.json` files in `checkpoints/`.
   * If Colab disconnects, clicking **Runtime → Run All** restores all progress instantly.
   * To force re-training, set `FORCE_RETRAIN = True` in Cell 3.
5. **GPU Acceleration:** Automatically leverages `tree_method="hist"` and `device="cuda"` for XGBoost when CUDA is detected, falling back safely to CPU with multi-threading (`n_jobs=-1`).

---

## 📁 Repository Directory Structure

```
DS422-Digital-Ad-Profit-Prediction/
├── index.html                                # Standalone Executive Web Application & Simulator
├── README.md                                 # Comprehensive Documentation & Viva Defense Guide
├── DS422_Digital_Ad_Profit_Prediction.ipynb  # Self-contained, fully executed Colab Notebook (3.7 MB)
├── tech_advertising_campaigns_dataset.csv   # Raw Dataset (10,000 campaigns x 41 columns)
├── figures/                                  # 9 Publication-Grade 300 DPI Diagnostic Charts
│   ├── 01_profit_distribution.png
│   ├── 02_categorical_profit_drivers.png
│   ├── 03_platform_and_seasonality.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_outlier_boxplots.png
│   ├── 06_model_comparison.png
│   ├── 07_actual_vs_predicted.png
│   ├── 08_residual_diagnostics.png
│   └── 09_feature_importance.png
├── checkpoints/                              # Persistent Model & Split Checkpoints
│   ├── preprocessed_data.joblib              # 80/20 Split & ColumnTransformer (6.4 MB)
│   ├── model_linear_regression.joblib        # Fitted Multiple Linear Regression
│   ├── model_ridge_regression.joblib         # Fitted Ridge Regression
│   ├── model_lasso_regression.joblib         # Fitted Lasso Regression
│   ├── model_decision_tree.joblib            # Fitted Decision Tree
│   ├── model_random_forest.joblib            # Fitted Random Forest Regressor (12.2 MB)
│   ├── model_xgboost.json                    # Baseline XGBoost Booster (native JSON)
│   ├── model_tuned_xgboost.json              # Champion Tuned XGBoost Booster (native JSON)
│   └── metrics_*.json                        # 7 Individual Model Metric Files
├── outputs/                                  # Machine-Readable & Document Deliverables
│   ├── analysis_summary.md                   # Full Analytical Audit & Methodology Report
│   ├── metrics.json                          # Complete JSON Schema of All Model Benchmarks
│   ├── model_comparison.csv                  # Tabular CSV Comparison Leaderboard
│   └── figures/                              # Mirrored 300 DPI Figures
└── audit_dataset.py                          # Independent EDA & VIF Validation Script
```

---

## 📜 Academic Integrity & Attribution

* **Course:** DS422 — Machine Learning I
* **Academic Evaluation Date:** 22nd November, 2026
* **Dataset Attribution:** [Digital Advertising Campaign Performance Dataset (2024–2026)](https://www.kaggle.com/datasets/juniornsa/digital-advertising-campaign-performance-dataset), authored by Junior Nsa (Kaggle, 2026).
* **Synthetic Calibration Note:** The dataset is synthetic and calibrated to 2025 digital advertising performance benchmarks. All preprocessing, model training, cross-validation, hyperparameter tuning, and visualizations reflect 100% genuine code execution on this dataset.

---
*Created for the DS422 Machine Learning Lab Project by Hasibur Rahman.*
