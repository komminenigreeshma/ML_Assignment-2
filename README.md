# ML_Assignment-2

## Problem Statement
The goal of this assignment is to apply six different machine learning classification algorithms to the UCI Letter Recognition dataset. This dataset contains thousands of samples of alphabet characters represented by numerical features. Each model is trained and tested on the dataset, and their performance is compared using evaluation metrics such as Accuracy, AUC, Precision, Recall, F1 Score, and MCC. The outcomes are presented through an interactive Streamlit application that enables users to upload test data, choose models, view performance metrics, visualize confusion matrices, and download predictions. The assignment simulates a practical ML workflow that includes model building, evaluation, interface design, and deployment on Streamlit Community Cloud.

---

## Dataset Description
- **Dataset**: [Letter Recognition - UCI Repository](https://archive.ics.uci.edu/dataset/59/letter+recognition)  
- **Instances**: 20,000  
- **Features**: 16 numerical attributes (statistical features of letter images)  
- **Target**: 26 classes (A–Z letters)  
- **Task**: Multi-class classification
- **Why This Dataset**:
    - The dataset is sufficiently large (20,000 samples), which ensures reliable training and testing across multiple models.
    - Its numeric attributes make it compatible with a wide range of algorithms, from simple linear classifiers to advanced ensemble methods, allowing meaningful performance       comparisons.
    - With 26 output classes and several letters that share similar visual characteristics (e.g., O vs. Q, C vs. G), the dataset introduces complexity that challenges models       to distinguish subtle differences.
    - This variety provides an excellent opportunity to evaluate how simpler approaches (such as Logistic Regression and Naive Bayes) perform relative to more sophisticated        ensemble techniques (like Random Forest and XGBoost).

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
| Logistic Regression | Performs reasonably well, but lower accuracy compared to tree-based models. High AUC indicates good separability, meaning the model distinguishes classes well even if raw accuracy is lower. However, being a linear model, it struggles with complex non-linear boundaries in the dataset. |
| Decision Tree       | Strong performance with good accuracy. Easy to interpret, but prone to overfitting — especially on large datasets like this one. The slightly lower AUC compared to ensembles shows that single trees are less robust. |
| KNN                 | Very strong results; dataset structure is well-suited for distance-based classification. High accuracy and AUC suggest that local neighborhood information is highly predictive. However, KNN can be computationally expensive on large datasets and sensitive to scaling. |
| Naive Bayes         | Weakest performer overall. The independence assumption does not hold well for this dataset, leading to lower accuracy and F1. Interestingly, the AUC remains relatively high, showing that while predictions are noisy, the model still ranks classes reasonably well. |
| Random Forest       | Excellent performance; balanced metrics and near-perfect AUC. By averaging across many trees, it reduces overfitting and variance. It is more stable than a single decision tree and handles feature interactions effectively. |
| XGBoost             | Best overall performer; slightly outperforms Random Forest in all metrics. Gradient boosting captures subtle patterns and optimizes errors iteratively, giving it an edge. It is computationally heavier but provides the most reliable predictions in this task. |

---
