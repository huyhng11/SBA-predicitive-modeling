# 🏦 SBA Loan Default Prediction - Profit-Optimized Machine Learning

# Overview
The Small Business Administration (SBA), established in 1953, has played a crucial role in supporting small businesses and entrepreneurship in the United States. Through its loan guarantee programs, the SBA helps reduce lending risks, encouraging banks to extend credit to small enterprises that drive job creation and economic development. With a network of 10 regional offices and a workforce exceeding 8,000, the SBA facilitated over 103,000 small-business financings in fiscal year 2024, a 22% increase from the previous year, contributing a total capital impact of $56 billion.

However, not all loans are successfully repaid. Loan defaults present significant financial risks to lenders, making effective risk assessment a crucial part of the lending process. Traditionally, loan decisions were often based on credit history and qualitative assessments. Yet, as financial markets become more complex, data-driven approaches have become increasingly important. Data analysis allows lenders to move beyond intuition, leveraging historical trends and statistical models to improve the accuracy of credit risk evaluations and make more informed, objective lending decisions.

This project aims to develop a comprehensive predictive framework for small-business loan defaults using various machine learning algorithms, including Logistic Regression, Random Forest, XGBoost, Neural Network, Multilayer Perceptron. Beyond classification accuracy, we incorporate a cost-benefit analysis through a customized cost matrix to better reflect the financial consequences of loan decisions. By aligning model outputs with profit-maximization goals, the final recommendation will not only predict default risks but also support lenders in optimizing their approval strategies for maximum financial return.

## 📊 Overview

- **Dataset**: 899,000+ SBA loan records (1987–2014), enriched with macroeconomic indicators (GDP growth, interest rate, inflation, recession)
- **Objective**: Binary classification of loan status (Paid in Full vs. Charged Off)
- **Unique Contribution**: Custom cost matrix and profit-maximization framework to guide real-world lending decisions

## 🧠 Key Features

- **Preprocessing**:
  - One-Hot Encoding for categorical variables
  - Standardization of numerical features
  - Median imputation for missing values
  - Feature engineering:
    - SBA guarantee ratio
    - Real estate collateral indicator
    - Recession flags and macroeconomic enrichment

- **Models Implemented**:
  - Logistic Regression (Lasso, Ridge)
  - Random Forest
  - XGBoost (best performing)
  - Multilayer Perceptron (highest ROI)

- **Evaluation Strategy**:
  - ROC AUC, Accuracy, Precision, Recall, F1-Score
  - **Custom Cost Matrix**:
    - +5% profit for fully repaid loans
    - −25% cost for defaulted loans
    - −5% opportunity cost for false rejections
  - Threshold tuning to optimize **net profit**
  - Gain and Lift charts to identify ROI-maximizing lending policy

## 📈 Results

| Model           | ROC AUC | Net Profit | ROI    | Optimal Threshold |
|----------------|---------|------------|--------|-------------------|
| XGBoost         | 0.978   | $3.566B    | 3.49%  | 0.27              |
| Random Forest   | 0.975   | $3.484B    | 3.41%  | 0.27              |
| MLP (Neural Net)| 0.945   | $3.383B    | **3.97%**  | 0.15              |
| Logistic (Lasso/Ridge)| ~0.892 | ~$2.71B | 2.66% | ~0.34–0.35        |

> **Key Insight**: Approving only the top ~76–80% of applications based on repayment likelihood **maximizes profit**. This is operationalized using model-specific "P(Paid in Full)" thresholds.

## 🚀 How to Use

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/sba-loan-default-prediction.git

# 2. Install requirements
pip install -r requirements.txt

# 3. Run preprocessing
python scripts/preprocess.py

# 4. Train model (example: XGBoost)
python scripts/train_xgb.py


