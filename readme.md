#  Supply Chain Delivery Risk Predictor

### *Zero-leakage ML pipeline that flags delivery risk at order placement — before the warehouse even moves*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)]()
[![ML Framework](https://img.shields.io/badge/ML-XGBoost%20%7C%20LightGBM%20%7C%20CatBoost-orange)]()
[![Optimization](https://img.shields.io/badge/HPO-Optuna-blueviolet)]()
[![Interpretability](https://img.shields.io/badge/Explainability-SHAP-yellow)]()

---

##  The Problem

Supply chain delays cost companies billions annually — not because they're unpredictable, but because they're detected too late.

This project answers a single, high-stakes business question:

> **Can we predict, at the moment an order is placed, whether it will be delivered late?**

Using the **DataCo Smart Supply Chain** dataset (~180K transactions), this pipeline builds a robust binary classifier that flags delivery risk in real time — before the warehouse even processes the shipment.

---

##  Business Impact

| Outcome | Value |
|---|---|
|  **Early Risk Detection** | Flag at-risk orders at placement, not post-shipment |
|  **Cost Avoidance** | Redirect logistics resources proactively before SLA breaches |
|  **Operational Intelligence** | Understand which shipping modes, regions, and time windows drive delay |
|  **Scalable Inference** | One-call `pipeline.predict(X_new)` — deployable as a REST API in minutes |
|  **Recall-Optimized** | Tuned to minimize missed delays (false negatives), not just maximize accuracy |

---

##  Pipeline Architecture

```
RAW DATA (180K rows, 53 columns)
    │
    ▼
[1] LEAKAGE REMOVAL          → Drop 12 post-shipment columns (Delivery_Status, etc.)
    │
    ▼
[2] STRICT TRAIN / VAL / TEST SPLIT (70 / 15 / 15, stratified)
    │                         ← All subsequent steps fit on TRAIN ONLY
    ▼
[3] FEATURE ENGINEERING       → Temporal features, financial ratios,
    │                           interaction terms, log & Yeo-Johnson transforms
    ▼
[4] PREPROCESSING             → Median imputation | Rare-category grouping (<0.5%)
    │                           OrdinalEncoder | StandardScaler (fit on train)
    ▼
[5] SMOTE (train only)        → Balance classes without contaminating val/test
    │
    ▼
[6] FEATURE SELECTION         → LightGBM SHAP importance
    │                           + Bayesian threshold search (Optuna, 50 trials)
    │                           → 38 features → optimal subset, +1.21 AUC pts
    ▼
[7] MODEL TRAINING & TUNING   → XGBoost | LightGBM | CatBoost | Random Forest
    │                           Optuna HPO on validation set (not test)
    ▼
[8] STACKING ENSEMBLE         → SkLearn StackingClassifier (CV=3)
    │                           + Optuna threshold tuning (F2-score optimized)
    ▼
[9] TEST EVALUATION           → Single blind evaluation on held-out test set
    │
    ▼
[10] SHAP INTERPRETABILITY    → TreeExplainer on XGBoost base learner
         + Individual prediction breakdowns
```

---

##  Results

### Model Leaderboard — Test Set

| Model | Recall | F2-Score | F1-Score | AUC | Threshold |
|---|---|---|---|---|---|
| 🥇 **Stacking** | **0.8507** | **0.8506** | **0.8505** | **0.9293** | **0.37** |
| Random Forest | 0.8476 | 0.8475 | 0.8473 | 0.9268 | 0.40 |
| XGBoost | 0.8073 | 0.8072 | 0.8076 | 0.8912 | 0.50 |
| CatBoost | 0.8008 | 0.7995 | 0.8010 | 0.8924 | 0.48 |
| LightGBM | 0.7926 | 0.7926 | 0.7931 | 0.8803 | 0.45 |

### Why Recall over Accuracy?

In a supply chain context, a **missed delay (false negative)** is far more costly than a false alarm (false positive). The pipeline is explicitly optimized for **Recall** and **F2-score** — both weight false negatives more heavily. The decision threshold is tuned via Optuna, not left at the default 0.50.

### Cross-Validation Stability

The final model was validated with stratified 5-fold CV on the training set. CV → Test gap < 0.03 on all metrics, confirming **no overfitting**.

---

##  Key Insights

**1. Scheduled Shipping Days dominates the signal.**
`Days_for_shipment_scheduled` is the single most important feature by SHAP margin. Longer promised windows correlate strongly with actual delays — suggesting systemic overcommitment in certain shipping tiers.

**2. Shipping Mode is a structural risk factor.**
The mode of transport (Standard, First Class, Same Day, etc.) carries significant predictive weight, independent of the scheduled time. This points to carrier-level reliability differences.

**3. Order time-of-day captures operational patterns.**
Orders placed in late hours (after 14:00) show measurably higher delay rates — likely due to warehouse cutoff times and batch processing constraints.

**4. Feature selection pays off.**
Bayesian SHAP-based selection reduced the feature space from 38 to an optimal subset, improving AUC by +1.21 points vs using all features — less noise, more signal.

**5. Leakage is a silent killer.**
12 columns (`Delivery_Status`, `Days_for_shipping_real`, `Order_Status`, etc.) were removed before any modeling. Keeping them would have inflated performance to ~99% — and produced a model useless in production.

---

##  Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **Data** | Pandas, NumPy |
| **ML Models** | XGBoost, LightGBM, CatBoost, Scikit-learn |
| **Imbalanced Data** | imbalanced-learn (SMOTE) |
| **Hyperparameter Optimization** | Optuna (TPE Sampler) |
| **Interpretability** | SHAP (TreeExplainer) |
| **Visualization** | Matplotlib, Seaborn |
| **Serialization** | Joblib |
| **Data Source** | KaggleHub (`shashwatwork/dataco-smart-supply-chain-for-big-data-analysis`) |

---

## 📁 Project Structure

```
supply-chain-ml-pipeline/
│
├── Supply_Chain_ML_Pipeline_final.ipynb   # Self-contained notebook — all cells ordered
│                                          # Covers: EDA → FE → Preprocessing → SMOTE
│                                          #         → Feature Selection → HPO → Stacking
│                                          #         → Evaluation → SHAP → Inference
│
└── README.md
```

> **Self-contained design:** The notebook handles everything end-to-end. Artifacts (graphs, `.pkl` files) are generated and saved locally when cells are executed — no external scripts required.

---

## ⚙️ Installation & Usage

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/supply-chain-ml-pipeline.git
cd supply-chain-ml-pipeline

pip install -r requirements.txt
# or manually:
pip install catboost xgboost lightgbm optuna imbalanced-learn shap kagglehub scikit-learn pandas numpy matplotlib seaborn joblib
```

### 2. Run the Notebook

```bash
jupyter notebook Supply_Chain_ML_Pipeline_final.ipynb
```

> The notebook auto-downloads the dataset via `kagglehub`. Kaggle API credentials required.

### 3. Run Inference on New Data

```python
import joblib
import pandas as pd

# Load the full pipeline bundle
pipeline = joblib.load('pipeline_final_complet.pkl')
model     = pipeline['model']
threshold = pipeline['threshold']
features  = pipeline['features']

# Prepare your new order data (same feature schema)
X_new = pd.DataFrame([{
    'Days_for_shipment_scheduled': 5,
    'Shipping_Mode': 0,
    'order_hour': 22,
    # ... other features
}], columns=features)

proba  = model.predict_proba(X_new.values)[:, 1][0]
pred   = int(proba >= threshold)

print(f"Delay Risk: {'⚠️ HIGH' if pred == 1 else '✅ LOW'} ({proba:.1%} probability)")
```

---

## 🧠 Key Learnings

**On pipeline discipline:**
Enforcing a strict split-first protocol — then fitting every transformer (imputer, encoder, scaler, SMOTE) exclusively on training data — is the difference between a research demo and a trustworthy model. Leakage detection before modeling is non-negotiable.

**On threshold optimization:**
Default 0.50 decision thresholds are a hyperparameter in disguise. Using Optuna to tune the classification threshold on the validation set (optimizing F2, not accuracy) produced measurable recall gains without retraining any model.

**On feature selection:**
Permutation-based or correlation-based selection often underperforms. SHAP-based importance, combined with Bayesian threshold search, yielded a smaller, stronger feature set that generalizes better — confirming that fewer, meaningful features beat noisy completeness.

**On ensemble design:**
Stacking outperformed all individual models — but the meta-learner's quality depends on diversity among base learners. Using XGBoost, LightGBM, CatBoost, and Random Forest as bases provided sufficient diversity for meaningful stacking gains.

**On business framing:**
Choosing Recall as the primary metric (over F1 or accuracy) was a deliberate business decision. In supply chain, the cost of an undetected delay dwarfs the cost of a false alarm. The model is explicitly optimized for operational cost minimization, not statistical elegance.

---

##  Future Improvements

- [ ] **Probability calibration** — apply `CalibratedClassifierCV` so scores reflect true delay likelihood
- [ ] **Richer features** — supplier reliability history, public holidays, weather signals
- [ ] **REST API** — wrap the pipeline in FastAPI (`/predict`, `/health`) + Docker for easy deployment
- [ ] **Drift monitoring** — trigger retraining automatically when production Recall drops below 0.80

---

## 📄 Dataset

**DataCo Smart Supply Chain for Big Data Analysis**
- Source: [Kaggle](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)
- Size: ~180,000 transactions, 53 raw columns
- Target: `Late_delivery_risk` (binary — 1 = late, 0 = on time)
- Class distribution: approximately 55% late / 45% on time

---


Built with rigorous ML engineering practices. Feedback and contributions welcome.


---

*"A model that leaks is a model that lies. Build clean or don't build at all."*
