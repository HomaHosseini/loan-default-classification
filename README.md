# Loan Default Classification

An educational machine learning project comparing four classification algorithms for loan-default prediction. The project explores data preprocessing, feature engineering, hyperparameter selection, and model performance analysis using Python and scikit-learn.

## Overview

The goal is to compare different machine learning approaches for classifying loans into two categories based on borrower and loan characteristics.

The implementation includes data preprocessing, engineered financial features, validation-based hyperparameter selection, and visualizations for comparing model performance.

## Models

| Model | Description |
|---|---|
| **Decision Tree** | Uses a tree of feature-based decisions to classify loan outcomes. |
| **K-Nearest Neighbors (KNN)** | Predicts a class using the majority label among the nearest training examples. Includes a custom KNN implementation. |
| **AdaBoost** | Combines sequentially trained weak learners into an ensemble classifier. |
| **Random Forest** | Combines multiple decision trees to improve generalization and provide feature-importance estimates. |

## Features

- Data loading and preprocessing with Pandas.
- Financial feature engineering, including credit age, credit risk score, loan burden, and employment stability.
- Standardization using `StandardScaler`.
- Training, validation, and test splits.
- Hyperparameter selection using validation accuracy and `GridSearchCV`.
- Comparison of model accuracy, training time, and prediction time.
- Decision-tree visualization.
- Random Forest feature-importance visualization.

## Dataset

The program expects a CSV file named `loan_sub.csv` in the project root directory.

The dataset must contain the columns used by the preprocessing function, including loan interest rate, debt-to-income ratio, revolving credit utilization, employment information, loan purpose, and the `bad_loans` target.

The target variable is interpreted as:

- `0`: Loan is not classified as a bad loan.
- `1`: Loan is classified as a bad loan.

Note: The dataset (loan_sub.csv) is not included in this repository and must be obtained separately.

## Requirements

- Python 3.9 or newer
- NumPy
- Pandas
- scikit-learn
- Matplotlib

## Experimental Workflow

1. Load the loan dataset and select relevant features.
2. Handle missing values and construct additional features.
3. Encode categorical variables and standardize numerical features.
4. Create training, validation, and test sets.
5. Select hyperparameters using validation accuracy.
6. Train the selected models.
7. Evaluate the models on the test set.
8. Visualize model performance and Random Forest feature importance.

## Visualizations

The script produces:

- **Model comparison:** Test accuracy, training time, and prediction time.
- **Decision tree:** A visualization of the trained Decision Tree classifier.
- **Feature importance:** A ranking of features according to the trained Random Forest model.


## Technologies

- Python
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Supervised Machine Learning

