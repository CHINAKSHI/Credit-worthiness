# 🏦 Creditworthiness Prediction Report

## 📌 1. Problem Statement
**Objective:**  
Predict whether a village resident is creditworthy based on socio-economic and loan-related data.

---

## 📁 2. Dataset Overview

**Dataset Name:** `trainingData.csv`  
**Records:** 40,000 individuals  
**Columns:** 21 original features, including:

- **Demographics:** `age`, `sex`, `social_class`, `city`
- **Financials:** `annual_income`, `monthly_expenses`, `loan_amount`, `loan_tenure`
- **Assets & Facilities:** `house_area`, `home_ownership`, `sanitary_availability`, `water_availabity`

**Target Variable:** `creditworthy` (Binary: `1` = Creditworthy, `0` = Not Creditworthy)

---

## 🧼 3. Data Preprocessing

### 🧪 Exploratory Data Analysis (EDA)

- Visualized gender distribution, loan purpose distribution, income vs loan amount, etc.
- Identified outliers in `age`, `annual_income`, and other numeric features.

### 🧹 Missing Value Handling

- **Numerical columns:** Filled using **median**
- **Categorical columns:** Filled using **mode**

### 🚫 Outlier Removal

- Applied **IQR** and **Z-score** methods to eliminate anomalies.

---

## 🧮 4. Feature Engineering

Derived domain-relevant financial metrics:

### 📊 Derived Metrics

| Metric | Formula |
|--------|---------|
| **DTI** (Debt-to-Income Ratio) | `(loan_installments × 12) / annual_income` |
| **Disposable Income** | `annual_income - (monthly_expenses × 12)` |
| **Dependents** | `old_dependents + young_dependents` |
| **Dependent Ratio** | `Dependents / annual_income` |
| **LTV** (Loan-to-Value Ratio) | `loan_amount / house_area` *(only if `home_ownership == 1`)* |

---

## 🧠 5. Credit Scoring Logic

Each satisfied rule scores **+1 point**. Final score is between `0` to `7`.

| Rule No. | Condition | Description |
|----------|-----------|-------------|
| 1 | `DTI < 0.4` | Low debt burden |
| 2 | `Disposable_Income > loan_amount` | Can afford the loan |
| 3 | `Dependent_Ratio < 0.5` | Fewer dependents per income |
| 4 | `LTV < 0.8` | Safer loan relative to house value |
| 5 | `sanitary_availability == 1` **AND** `water_availabity == 1` | Has basic amenities |
| 6 | `loan_tenure <= 24` | Shorter repayment period |
| 7 | `social_class == 'A'` | Higher, stable social class |

### ✅ Final Labeling
- **Creditworthy (1):** If total score **≥ 4**
- **Not Creditworthy (0):** If total score **< 4**

---

## 🤖 6. Modeling Techniques

### 🔧 Preprocessing
- **Feature Encoding:** One-hot encoding
- **Imputation:** Median (numerical), Mode (categorical)
- **Outlier Handling:** IQR & Z-score methods

### 🔍 Models Trained

1. **Random Forest Classifier**
2. **Logistic Regression**
3. **XGBoost Classifier**

---

## 📈 7. Model Evaluation

### 🧪 Why these metrics?
Used **Accuracy, Precision, Recall, F1-Score** due to potential **imbalance** in target classes.

### 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** | 0.98 | 0.97 | 0.99 | 0.98 |
| **Logistic Regression** | 0.92 | 0.91 | 0.94 | 0.92 |
| **XGBoost** | 1.00 | 1.00 | 1.00 | 1.00 |

Other evaluations:
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve

---

## 💾 8. Final Output

- ✅ **Processed Output File:** `village_creditworthy.csv`
- 🧠 **Best Trained Model:** `creditworthiness_model.pkl`
- 🔄 **Full Pipeline:** `creditworthiness_pipeline.pkl`

---

## 🏁 Conclusion

The XGBoost model achieved **100% accuracy** with complete feature engineering and domain-inspired credit scoring. This robust system is now capable of classifying the creditworthiness of rural residents using a transparent, rule-based + machine learning hybrid approach.


