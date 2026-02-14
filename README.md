# ML_Assignment-2

## Problem Statement
The objective of this assignment is to implement six machine learning classification models on a chosen dataset, evaluate their performance using standard metrics, and deploy an interactive Streamlit web application. This simulates a real-world ML workflow: modeling, evaluation, UI design, and deployment.

---

## Dataset Description
- **Dataset**: [Letter Recognition - UCI Repository](https://archive.ics.uci.edu/dataset/59/letter+recognition)  
- **Instances**: 20,000  
- **Features**: 16 numerical attributes (statistical features of letter images)  
- **Target**: 26 classes (A–Z letters)  
- **Task**: Multi-class classification  

---

## Models Used and Evaluation Metrics

| ML Model Name       | Accuracy | AUC     | Precision | Recall  | F1      | MCC     |
|---------------------|----------|---------|-----------|---------|---------|---------|
| Logistic Regression | 0.77425  | 0.98052 | 0.774817  | 0.772974| 0.772762| 0.765303|
| Decision Tree       | 0.88100  | 0.93802 | 0.881082  | 0.880794| 0.880669| 0.876258|
| KNN                 | 0.94875  | 0.99285 | 0.949097  | 0.948578| 0.948611| 0.946717|
| Naive Bayes         | 0.65225  | 0.95728 | 0.664067  | 0.651197| 0.647879| 0.639111|
| Random Forest       | 0.96150  | 0.99943 | 0.961996  | 0.961175| 0.961349| 0.959978|
| XGBoost             | 0.96425  | 0.99969 | 0.964183  | 0.963996| 0.963992| 0.962827|

---

## Observations

| ML Model Name       | Observation about model performance |
|---------------------|-------------------------------------|
| Logistic Regression | Performs reasonably well, but lower accuracy compared to tree-based models. High AUC indicates good separability. |
| Decision Tree       | Strong performance, but slightly less robust than ensemble methods due to possible overfitting. |
| KNN                 | Very strong results; dataset structure is well-suited for distance-based classification. |
| Naive Bayes         | Weakest performer; independence assumption does not hold well for this dataset, though AUC remains high. |
| Random Forest       | Excellent performance; balanced metrics and near-perfect AUC. Very reliable. |
| XGBoost             | Best overall performer; slightly outperforms Random Forest in all metrics, showing the strength of gradient boosting. |

---