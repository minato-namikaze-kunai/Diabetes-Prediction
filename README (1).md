# Diabetes Prediction Model

This project aims to predict whether a patient has diabetes based on medical diagnostic measurements. The prediction is made using a machine learning model trained on the **Pima Indians Diabetes Dataset**. The dataset contains various features related to patient health, such as glucose levels, BMI, and age.

The model uses **StandardScaler** to standardize the feature values before training to improve the model's performance.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Dependencies](#dependencies)
- [Setup](#setup)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Diabetes is a chronic disease that occurs either when the pancreas does not produce enough insulin or when the body cannot effectively use the insulin it produces. Early diagnosis can help in controlling the disease more effectively. This project implements a machine learning pipeline to predict the onset of diabetes in a patient.

We use **StandardScaler** to standardize the data and a variety of machine learning models to make predictions, such as:

1. **Logistic Regression**
2. **Support Vector Machine (SVM)**
3. **Random Forest Classifier**

## Dataset

The dataset used for training and testing the models is the **Pima Indians Diabetes Dataset**.

- **Source**: [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
- **Features**:
  - `Pregnancies`: Number of times pregnant
  - `Glucose`: Plasma glucose concentration after 2 hours in an oral glucose tolerance test
  - `BloodPressure`: Diastolic blood pressure (mm Hg)
  - `SkinThickness`: Triceps skin fold thickness (mm)
  - `Insulin`: 2-hour serum insulin (mu U/ml)
  - `BMI`: Body mass index (weight in kg/(height in m)^2)
  - `DiabetesPedigreeFunction`: A function which scores likelihood of diabetes based on family history
  - `Age`: Age (years)
- **Target**:
  - `Outcome`: Class variable (0 or 1) representing whether the patient has diabetes (1) or not (0)

## Methodology

### StandardScaler

Before applying machine learning models, the data is standardized using the **StandardScaler** from `sklearn.preprocessing`. Standardization ensures that the data has a mean of 0 and a standard deviation of 1, which is especially useful for models sensitive to the scale of input data (e.g., SVM, Logistic Regression).

### Models

Three different machine learning algorithms were used:

1. **Logistic Regression**: A linear model that performs well for binary classification tasks.
2. **Support Vector Machine (SVM)**: A model that finds the hyperplane which maximally separates the classes.
3. **Random Forest Classifier**: An ensemble learning method based on decision trees.

### Evaluation Metrics

To evaluate the model's performance, we use the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

## Dependencies

Ensure the following dependencies are installed before running the project:

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter (for notebooks)


| Model                     | Accuracy | Precision | Recall | F1 Score |
|----------------------------|----------|-----------|--------|----------|
| Logistic Regression         | 0.78     | 0.76      | 0.72   | 0.74     |
| Support Vector Machine (SVM)| 0.80     | 0.79      | 0.75   | 0.77     |
| Random Forest               | 0.82     | 0.81      | 0.78   | 0.79     |


You can install the dependencies using:
```bash
pip install -r requirements.txt
