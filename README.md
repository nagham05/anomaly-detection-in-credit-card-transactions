# 💳 Unsupervised Anomaly Detection in Credit Card Transactions

## 🎯 Project Goal

The primary objective of this project was to utilize unsupervised machine learning techniques to identify potential fraudulent transactions within a highly imbalanced dataset. The model was trained without access to the true fraud labels, demonstrating the feasibility of deploying anomaly detection systems in real-world scenarios where labeled data for rare events is scarce.

## 📦 Dataset

* **Source:** Credit Card Fraud Detection – Kaggle
* **Size:** 284,807 transactions.
* **Features:** 30 numerical features:

  * **V1 to V28:** Anonymized PCA components.
  * **Time:** Seconds elapsed since the first transaction.
  * **Amount:** Transaction amount.
* **Target:** **Class** (0: Normal, 1: Fraud). This column was **hidden from the model** during training and used *only* for evaluation.

### Data Imbalance

Only **492 transactions (0.17%)** were fraudulent. This extreme imbalance made anomaly detection essential.

## 🛠️ Methodology

The project followed a structured ML workflow:

### 1. Preprocessing

* **Feature Scaling** for `Time` and `Amount` to align with PCA-scaled components.
* **Contamination Rate:** Empirically calculated at **0.001727**, used as the `contamination` parameter.

### 2. Unsupervised Models Compared

| Model                | Technique                                 | Summary                                               |
| -------------------- | ----------------------------------------- | ----------------------------------------------------- |
| **Isolation Forest** | Random partitioning to isolate anomalies. | Best performance, effective in high-dimensional data. |
| **One-Class SVM**    | Kernel-based boundary modeling.           | Good baseline, but poor precision in this case.       |

### 3. Evaluation Metrics (Fraud Only)

| Model                 | Recall     | Precision  | Notes                                            |
| --------------------- | ---------- | ---------- | ------------------------------------------------ |
| **Isolation Forest**  | **0.3313** | **0.3313** | Selected for best balance of recall & precision. |
| **OCSVM (ν = 0.005)** | 0.33       | 0.10       | Too many false positives → impractical.          |

## 💡 Key Results and Anomaly Profile

The Isolation Forest model detected **33.13%** of all fraud cases. Key anomaly characteristics:

### 1. Higher Transaction Amounts

* Anomaly transactions had noticeably higher mean and maximum `Amount` values.
* Transactions above the maximum scaled amount of normal transactions (>0.30 scaled) were **all** flagged as anomalies.

### 2. Rare Structural PCA Pattern

Strong deviations between anomaly and normal transactions across specific PCA components:

| Feature           | Mean Deviation (Anomaly – Normal) | Interpretation                     |
| ----------------- | --------------------------------- | ---------------------------------- |
| **V1**            | **−14.76**                        | Largest negative deviation.        |
| **V3**            | **−10.24**                        | Strong anomaly indicator.          |
| **V10, V14, V17** | Negative                          | Known fraud-related in literature. |
| **V4**            | **+5.13**                         | Largest positive deviation.        |

**Insight:** Fraudulent transactions show a *unique structural fingerprint* — high positive **V4** values combined with strongly negative features like **V1** and **V3**. This allows the model to catch even low-value fraud.

## 🚀 How to Run the Project

### Requirements

Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib
```

### Steps

1. Download `creditcard.csv` from Kaggle and place it in a `data/` folder.
2. Open `credit_card.ipynb` in Jupyter.
3. Run all cells to reproduce preprocessing, modeling, evaluation, and insights.

## 🎓 Learning Outcomes

* Understanding of unsupervised fraud detection using **Isolation Forest** and **One-Class SVM**.
* Hands-on experience with extreme data imbalance scenarios.
* Ability to interpret PCA-driven anomaly patterns and extract business insights.
