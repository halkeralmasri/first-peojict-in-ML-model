# first-peojict-in-ML-model
:)
# 🚗 Car Price Prediction Pipeline

A machine learning pipeline designed to predict used car prices using feature engineering, categorical target encoding, and ensemble learning with Scikit-Learn.

---

## 📌 Project Overview

Predicting car prices involves handling non-linear features, skewed target distributions, and high-cardinality categorical variables (such as vehicle brand and model). This project implements an end-to-end, leak-free machine learning workflow that cleans data, handles outliers, transforms categorical features, and evaluates model performance using regression metrics.

---

## 🛠️ Key Technical Features

* **Data Cleaning & Leak-Free Pipeline:** Used Scikit-Learn `Pipeline` and `ColumnTransformer` to prevent data leakage between training and testing splits.
* **Feature Engineering:** Derived `Car_Age` from manufacturing years and filtered high-quantile price outliers to stabilize model training.
* **Skewness Correction:** Applied a logarithmic transformation (`np.log1p`) to the target variable (`Selling_Price`) to handle right-skewed price distributions.
* **Categorical Encoding:** Utilized `TargetEncoder` for high-cardinality features (`Brand`, `Model`) and `OneHotEncoder` for low-cardinality categorical variables.
* **Outlier Mitigation:** Used `RobustScaler` on numerical features to reduce the sensitivity of continuous variables to extreme values.

---

## 📊 Model & Performance

The baseline model relies on a **Random Forest Regressor** trained on engineered features.

| Metric | Score / Value | Description |
| :--- | :--- | :--- |
| **$R^2$ Score** | ~0.85+ | Percentage of variance explained by the model |
| **MAE** | Lowest Dollar Margin | Average absolute error in predictions |
| **RMSE** | Penalized Score | Root Mean Squared Error accounting for larger deviations |

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed along with the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
