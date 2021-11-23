# Santader-Customer-Transaction-Classification

## Problem Statement: Project Overview

In this problem, Santander Bank poses a challenge,  in order to help them with the problem of identification of the customers who will make a transaction with the bank in future, irrespective of the amount of money transacted previously with the bank. The data set provided is similar to the real data that is available to solve the problem, although the data that is provided to us for solving the problem is masked completely with only numeric values. The data is anonymous with no Customer details been revealed to the participants of the competition. The data sheet contains 200000 rows for both train and test data. The Train Data set has 202 columns with 200 columns having values for var_1 to var_200, one column for ID code and one column for target, which are the outcome of the transaction. The same columns are present for test data except for the target.

## Data Acquisition:

Data has been taken from Kaggle: https://www.kaggle.com/c/santander-customer-transaction-prediction/overview

## Languages Used 
**Python Version:** 3.9.0

## Resources and Tools Used
**Tools:** Jupyter Notebook

**Packages:** Pandas, NumPy, sklearn, Matplotlib, missingno, scipy, seaborn, imblearn and counter.

## EDA

* No null values found.
* No duplicates found.
* Outliers were present, they were removed. Data points inside 2 standard deviations are kept.
* Skewness was observed in the features. 88 features are left-skewed and 112 features are right skewed.
* Class imbalance observed. For every observation of class 1, nearly 9 samples of class 0 were there.




## Feature Engineering

* I.d column was dropped as it was irrelevant.
* Data was scaled using Standard Scaler.
* Principal component analysis was done, as some overfitting was observed. 
* For 160 components, explained variance goes as high as 81.86 which is pretty good. We have dropped down from 199 features.
* Balanced data with random undersampling and oversampling to find a better class distribution.

## Model Building

* Built the baseline model using LGBM.
* Built different combinations of model and data like:

1) LGBM + PCA data
2) LGBM + No PCA data
3) Catboost + PCA data
4) Catboost + No PCA data
5) A stacked classifier of LGBM and Catboost

## HyperParameter Tuning

* Hyperparameters were tuned using grid search.
* Optimal models were found for each one of them.

## Model Interpreation 

* Model interpretation was done using SHAP.

## Metric For Evaluation

* ROC-AUC
* Plotted ROC-AUC curves and confusion matrix.

## Conclusion

* All models are performing almost the same.
* I tried to do a PCA to deal with overfiitting but that did not help the cause.
* Tried a stacked model, so as to get a better test score but that was also nearly same.
* LGBM performed the best with train roc: 0.92 and test roc: 0.86.
