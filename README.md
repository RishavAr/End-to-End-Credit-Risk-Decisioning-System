# End-to-End Credit Risk & Decisioning System

## Production-Grade Credit Default Risk Model

A complete implementation of a FICO-style credit scoring system with model comparison, production monitoring, and score distribution analysis.

---

## 📌 Problem Statement

Build a credit default risk model and deploy it with production validation + score distribution monitoring, exactly like FICO models.

---

## 🏗️ Project Structure

```
credit_risk_system/
├── README.md                    # This file
├── main.py                      # Main execution script
├── requirements.txt             # Python dependencies
├── configs/
│   └── config.py               # Configuration settings
├── src/
│   ├── __init__.py
│   ├── data_generator.py       # Synthetic LendingClub-style data
│   ├── feature_engineering.py  # Feature engineering + WoE encoding
│   ├── model_training.py       # Model training & comparison
│   ├── scoring.py              # FICO-style scoring (300-850)
│   ├── monitoring.py           # PSI, CSI, drift detection
│   └── visualization.py        # All plotting functions
├── data/
│   └── synthetic_credit_data.csv  # Pre-generated dataset
└── outputs/                    # Generated outputs (after running)
    ├── model_comparison.csv
    ├── feature_importance.csv
    ├── score_distribution.csv
    ├── scorecard.csv
    ├── csi_analysis.csv
    ├── roc_curves.png
    ├── ks_curve.png
    └── ... (more visualizations)
```

---

## 📊 Data Specification

### Synthetic Dataset (LendingClub/FICO-style)
- **100,000 records** (scalable to millions)
- **~15% default rate** (realistic class imbalance)
- **14 numeric features** + **5 categorical features**

### Numeric Features
| Feature | Description | Range |
|---------|-------------|-------|
| annual_income | Yearly income | $15K - $500K |
| loan_amount | Requested loan amount | $1K - $100K |
| debt_to_income | DTI ratio | 0% - 60% |
| credit_utilization | Credit card utilization | 0% - 100% |
| num_credit_lines | Number of credit accounts | 1 - 30 |
| total_credit_limit | Total available credit | $5K - $200K |
| revolving_balance | Current revolving debt | $0 - $150K |
| num_delinquencies | Past delinquencies | 0 - 10 |
| months_since_delinquency | Time since last delinquency | 0 - 120+ |
| num_inquiries_6m | Credit inquiries (6 months) | 0 - 15 |
| months_employed | Employment duration | 0 - 480 |
| interest_rate | Assigned interest rate | 5.5% - 30% |
| loan_term | Loan term in months | 36 or 60 |
| installment_amount | Monthly payment | Calculated |

### Categorical Features
| Feature | Categories |
|---------|------------|
| home_ownership | RENT, MORTGAGE, OWN, OTHER |
| employment_status | Employed, Self-Employed, Unemployed, Retired, Student |
| loan_purpose | debt_consolidation, credit_card, home_improvement, etc. |
| verification_status | Verified, Source Verified, Not Verified |
| grade | A, B, C, D, E, F, G |

---

## 🧠 Models Compared

| Model | Description | Strengths |
|-------|-------------|-----------|
| **Logistic Regression** | Linear baseline | Interpretable, coefficients → scorecard |
| **Random Forest** | Ensemble of trees | Handles non-linearity, feature importance |
| **Gradient Boosting** | Sequential boosting | High accuracy, handles imbalance |
| **Hist Gradient Boosting** | Histogram-based GB | Fast, handles missing values natively |

---

## 📈 Expected Results

### Model Performance (Test Set)

| Model | AUC-ROC | Gini | KS Statistic | Precision | Recall |
|-------|---------|------|--------------|-----------|--------|
| Logistic Regression | 0.78-0.82 | 0.56-0.64 | 0.42-0.48 | 0.35-0.45 | 0.55-0.65 |
| Random Forest | 0.82-0.86 | 0.64-0.72 | 0.48-0.54 | 0.40-0.50 | 0.58-0.68 |
| **Gradient Boosting** | **0.84-0.88** | **0.68-0.76** | **0.50-0.58** | **0.42-0.52** | **0.60-0.70** |
| Hist Gradient Boosting | 0.83-0.87 | 0.66-0.74 | 0.49-0.56 | 0.41-0.51 | 0.59-0.69 |

**Expected Best Model:** Gradient Boosting or Hist Gradient Boosting

### FICO-Style Credit Score Distribution

| Score Band | Score Range | Expected % | Expected Default Rate |
|------------|-------------|------------|----------------------|
| Exceptional | 800-850 | 5-10% | 1-3% |
| Very Good | 740-799 | 15-20% | 3-6% |
| Good | 670-739 | 25-30% | 8-12% |
| Fair | 580-669 | 25-30% | 15-25% |
| Poor | 300-579 | 15-25% | 30-50% |

### Score Statistics
- **Mean Score:** 650-680
- **Median Score:** 660-690
- **Standard Deviation:** 70-90
- **Score Range:** 350-820

### Feature Importance (Top 10 Expected)

1. **grade** - Lender's risk assessment
2. **interest_rate** - Proxy for risk
3. **debt_to_income** - Payment capacity
4. **credit_utilization** - Credit behavior
5. **num_delinquencies** - Past payment issues
6. **loan_to_income** - Derived feature
7. **payment_to_income** - Affordability
8. **annual_income** - Repayment capacity
9. **num_inquiries_6m** - Credit seeking behavior
10. **months_employed** - Stability

### Information Value (IV) by Category

| Feature | Expected IV | Interpretation |
|---------|-------------|----------------|
| grade | 0.40-0.60 | Strong predictor |
| home_ownership | 0.05-0.15 | Weak-Medium |
| employment_status | 0.08-0.20 | Weak-Medium |
| loan_purpose | 0.03-0.10 | Weak |
| verification_status | 0.02-0.08 | Weak |

### Production Monitoring Expected Results

| Metric | Baseline vs Production (15% drift) |
|--------|-----------------------------------|
| **Score PSI** | 0.08-0.15 (Moderate change) |
| **Features with drift** | 3-6 features |
| **AUC degradation** | 1-3% |

---

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

### Quick Start
```python
# Run the complete pipeline
python main.py

# Or import and run programmatically
from main import main
results = main()

# Access results
print(results['eval_df'])  # Model comparison
print(results['score_stats'])  # Score statistics
```

### Generate Data Only
```python
from src.data_generator import CreditDataGenerator

generator = CreditDataGenerator(n_samples=100000, default_rate=0.15)
df = generator.generate()
df.to_csv('credit_data.csv', index=False)
```

### Train Models Only
```python
from src.model_training import CreditRiskModelTrainer
from src.feature_engineering import CreditFeatureEngineer

# Prepare data
fe = CreditFeatureEngineer()
X_train, y_train = fe.fit_transform(train_df)
X_test, y_test = fe.transform(test_df)

# Train all models
trainer = CreditRiskModelTrainer()
trainer.train_all_models(X_train, y_train, quick_mode=False, cv=5)

# Evaluate
eval_df = trainer.evaluate_on_test(X_test, y_test)
```

### Generate Scores
```python
from src.scoring import FICOStyleScorer

scorer = FICOStyleScorer(score_min=300, score_max=850)
probabilities = model.predict_proba(X_test)[:, 1]
credit_scores = scorer.probability_to_score(probabilities)
```

---

## 📋 Output Files Description

| File | Description |
|------|-------------|
| `model_comparison.csv` | All metrics for all models |
| `feature_importance.csv` | Feature importance by model |
| `logistic_coefficients.csv` | LR coefficients + odds ratios |
| `score_distribution.csv` | Score bands with default rates |
| `scorecard.csv` | Traditional scorecard points |
| `information_value.csv` | IV for categorical features |
| `csi_analysis.csv` | Feature-level drift analysis |
| `roc_curves.png` | ROC curves comparison |
| `pr_curves.png` | Precision-Recall curves |
| `calibration_curves.png` | Probability calibration |
| `feature_importance.png` | Visual feature importance |
| `score_distribution.png` | Score histogram & bands |
| `ks_curve.png` | KS statistic visualization |
| `gains_chart.png` | Cumulative gains chart |
| `psi_analysis.png` | PSI comparison |
| `model_comparison.png` | Summary comparison |

---

## 🔬 Key Metrics Explained

### AUC-ROC (Area Under ROC Curve)
- Measures discrimination ability
- 0.5 = random, 1.0 = perfect
- **Target: > 0.80**

### Gini Coefficient
- Gini = 2 × AUC - 1
- Measures inequality in predictions
- **Target: > 0.60**

### KS Statistic (Kolmogorov-Smirnov)
- Maximum separation between cumulative distributions
- **Target: > 0.40**

### PSI (Population Stability Index)
- Measures score distribution shift
- < 0.10: No change
- 0.10-0.25: Moderate change
- > 0.25: Significant change

### IV (Information Value)
- < 0.02: Not useful
- 0.02-0.10: Weak
- 0.10-0.30: Medium
- 0.30-0.50: Strong
- > 0.50: Suspicious (overfitting)

---

## 📝 Notes

1. **Class Imbalance**: The ~15% default rate creates realistic imbalance. Consider SMOTE or class weights for production.

2. **Feature Engineering**: 14 derived features are created including ratios, buckets, and interaction terms.

3. **Scorecard**: The logistic regression coefficients are converted to a traditional points-based scorecard for transparency.

4. **Monitoring**: PSI and CSI are industry-standard metrics for detecting model drift in production.

5. **Scalability**: The data generator can create millions of records. For large datasets, use `HistGradientBoostingClassifier` which is optimized for speed.

---

## 📧 Author

Generated by Claude (Anthropic) as a complete ML project template for credit risk modeling.
