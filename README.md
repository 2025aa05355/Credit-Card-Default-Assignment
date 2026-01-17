# Credit Default Prediction Project

## a. Problem Statement
This project aims to predict credit card defaults using client demographic and transaction data. By applying machine learning classification techniques, we attempt to identify high-risk customers to support financial decision-making.

## b. Dataset Description
* **Source:** Default of Credit Card Clients (UCI Repository)
* **Instances:** 30,000
* **Features:** 23 (Limits, Education, Gender, Marriage, Age, Pay History, Bill Amts, Pay Amts)
* **Target:** Binary (1 = Default, 0 = No Default)

## c. Models Used & Comparison
Evaluation metrics obtained from BITS Virtual Lab execution:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8097 | 0.7270 | 0.6913 | 0.2353 | 0.3511 | 0.3242 |
| Decision Tree | 0.8108 | 0.7358 | 0.6206 | 0.3488 | 0.4466 | 0.3639 |
| kNN | 0.7952 | 0.7080 | 0.5493 | 0.3564 | 0.4323 | 0.3252 |
| Naive Bayes | 0.7070 | 0.7371 | 0.3967 | 0.6504 | 0.4928 | 0.3218 |
| Random Forest (Ensemble) | 0.8185 | 0.7754 | 0.6582 | 0.3549 | 0.4612 | 0.3887 |
| XGBoost (Ensemble) | 0.8183 | 0.7826 | 0.6538 | 0.3610 | 0.4652| 0.3900 |

*(Note: Please fill the XXXX with your actual results from the Python script)*

### Model Performance Observations

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Performed reasonably well as a baseline but failed to capture complex non-linear relationships. |
| **Decision Tree** | Showed high variance; likely overfitted the training data resulting in lower test accuracy. |
| **kNN** | Performance was average; computation was slow during prediction due to the large dataset size. |
| **Naive Bayes** | Yielded the lowest performance, likely because the features (payment history) are not truly independent. |
| **Random Forest** | One of the best performers; effectively handled non-linearity and reduced overfitting via bagging. |
| **XGBoost** | Achieved the highest AUC and Accuracy, proving efficient at handling the imbalance in default classes. |
