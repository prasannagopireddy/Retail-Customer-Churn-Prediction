## 📌 Problem Statement

Retail businesses lose significant revenue when customers stop purchasing. The goal is to predict churn early using behavioral and transactional data, so businesses can act before it's too late.

---

## 📂 Dataset

**Source:** [Kaggle – Online Retail Customer Churn Prediction](https://www.kaggle.com/datasets/sahilislam007/online-retail-customer-churn-prediction-dataset)
**9,000 rows · 17 features · 0 missing values · ~19.7% churn rate**

Features span customer demographics, transaction history, engagement behavior, satisfaction scores, and loyalty signals.

---

## 🛠️ Feature Engineering

Beyond raw features, I engineered 6 new signals:

| Feature | Formula | Purpose |
|---|---|---|
| `recency` | Days since last purchase | Captures how recently active |
| `frequency` | Total purchases | Purchase habit |
| `monetary` | purchases × avg value | Total spend contribution |
| `engagement_score` | visits × avg session time | Platform interaction depth |
| `complaint_ratio` | tickets / (purchases + 1) | Friction per transaction |
| `loyalty_index` | referrals × satisfaction | Advocacy + happiness combined |

---

## 🤖 Models Trained

| Model | Configuration |
|---|---|
| Logistic Regression | `max_iter=1000`, scaled features |
| Random Forest | `n_estimators=200` |
| XGBoost | `n_estimators=500, lr=0.01, max_depth=4` |

Evaluated using Accuracy, ROC-AUC, F1-Score, and 5-Fold Stratified Cross-Validation.

---

## 📊 Results & What I Learned

All three models achieved ~80.27% accuracy — but this was misleading. The model only predicted class `0` (retained), never flagging a single churner. With a ROC-AUC of 0.531, the model performed barely above random.

**Root cause:** The 80/20 class imbalance was not addressed. The model took the easy route — predicting the majority class every time.

**What I'll do differently in future projects:**
- Apply `SMOTE` or `class_weight='balanced'` to handle imbalance
- Optimize the decision threshold for better recall on the minority class
- Evaluate with Precision-Recall curves, not just accuracy

> This was my first ML project. The results weren't perfect — but understanding *why* they failed is exactly what builds stronger intuition for the next one.

---

## 🔎 SHAP Explainability

Used `shap.TreeExplainer` on XGBoost to interpret feature-level contributions, making predictions transparent and explainable to business stakeholders.

---

## 🚀 Run Locally

```bash
git clone https://github.com/prasannagopireddy/Retail-Customer-Churn-Prediction.git
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap joblib
# Add dataset CSV from Kaggle, then open the notebook
jupyter notebook Retail_Customer_churn_Prediction.ipynb
```

---

## 📁 Repository Structure

```
Retail-Customer-Churn-Prediction/
├── Retail_Customer_churn_Prediction.ipynb
└── README.md
```

---

## 👨‍💻 Author

**Prasanna Gopireddy** · [GitHub](https://github.com/prasannagopireddy) · [LinkedIn](https://www.linkedin.com/in/prasannagopireddy)
