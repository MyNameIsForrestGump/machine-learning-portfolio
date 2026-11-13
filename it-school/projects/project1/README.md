# Customer Churn Prediction — Machine Learning

**IT School | Machine Learning — Project 1**

## Project Overview

This project focuses on predicting customer churn in the telecommunications industry using supervised machine learning.

Customer churn refers to customers discontinuing their relationship with a service provider. Predicting churn can help businesses identify customers at risk of leaving and develop more effective retention strategies.

The objective is to develop and compare multiple classification algorithms, evaluate their predictive performance, and identify a suitable model for customer churn prediction.

The project covers the complete machine learning workflow, from exploratory data analysis and preprocessing to model training, hyperparameter optimization, evaluation, and model serialization.

## Dataset

The project uses a telecommunications customer churn dataset containing **7,043 customer records**.

The dataset includes information about:

- Customer demographics
- Contract type and duration
- Internet and phone services
- Payment methods
- Monthly and total charges
- Customer tenure
- Churn status

**Target variable:** `Churn`

- `0` — Customer stays
- `1` — Customer leaves

The dataset is imbalanced, with approximately 26.5% of customers belonging to the churn class.

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

Performed exploratory analysis to understand the structure and characteristics of the dataset.

Key steps included:

- Inspecting features and data types
- Identifying missing values and inconsistencies
- Checking for duplicate records
- Examining class imbalance
- Investigating numerical and categorical variables

The analysis identified 11 records with missing `TotalCharges` values, corresponding to customers with zero tenure.

**Notebook:** [EDA.ipynb](notebooks/EDA.ipynb)

### 2. Data Preprocessing

Prepared the dataset for machine learning through:

- Handling missing values
- Removing records with zero tenure
- Converting categorical binary variables into numerical representations
- Applying one-hot encoding to categorical features
- Analyzing feature correlations with churn
- Removing selected features
- Excluding customer identifiers

A feature engineering step was also included to construct a binary indicator based on the relationship between monthly charges, total charges, and customer tenure.

**Notebook:** [Preprocessing.ipynb](notebooks/Preprocessing.ipynb)

### 3. Model Training

Implemented and compared four classification approaches:

| Model | Description |
|---|---|
| Dummy Classifier | Majority-class baseline |
| Logistic Regression | Linear classification with regularization |
| Random Forest | Ensemble of decision trees |
| XGBoost | Gradient-boosted decision trees |

The dataset was split into training and testing subsets using a stratified 75/25 split.

For hyperparameter optimization, `GridSearchCV` was applied with five-fold cross-validation and F1-score as the optimization metric.

StandardScaler was incorporated into the Logistic Regression pipeline.

**Notebook:** [Model_Training.ipynb](notebooks/Model_Training.ipynb)

### 4. Model Evaluation

Because the dataset is imbalanced, accuracy alone is not sufficient to evaluate model performance.

The primary evaluation metric was the **F1-score**, which balances precision and recall for the positive churn class.

The saved experiment comparison reports the following values:

| Model | Without Feature Engineering | With Feature Engineering |
|---|---:|---:|
| Dummy Classifier | 0.0000 | 0.0000 |
| Logistic Regression | 0.6192 | 0.6247 |
| Random Forest | **0.6285** | **0.6282** |
| XGBoost | 0.6038 | 0.6017 |

*These values are interpreted as F1-scores based on the evaluation code. The saved comparison text labels them as recall scores, so they should be verified when reproducing the experiments.*

Random Forest achieved the highest recorded F1-score in both configurations.

Feature engineering produced a small improvement for Logistic Regression but did not improve the recorded Random Forest or XGBoost results.

### 5. Model Serialization

The best-performing model was saved using `joblib`, allowing the trained estimator to be loaded for future inference or experimentation.

**Saved model:** `models/best_churn_model.pkl`

## Project Structure

```text
project-1/
├── README.md
├── requirements.txt
├── data/
│   ├── raw_data.csv
│   └── processed_data.csv
├── notebooks/
│   ├── EDA.ipynb
│   ├── Preprocessing.ipynb
│   ├── Model_Training.ipynb
│   └── comparison.txt
├── models/
│   └── best_churn_model.pkl
└── main/
    └── main.ipynb
```

## Technologies Used

- **Python** — Programming language
- **Pandas** — Data manipulation and preprocessing
- **NumPy** — Numerical computing
- **Matplotlib & Seaborn** — Data visualization
- **Scikit-learn** — Classification, pipelines, cross-validation, and model evaluation
- **XGBoost** — Gradient boosting
- **Joblib** — Model serialization
- **Jupyter Notebook** — Experimentation and analysis

## Running the Project

Clone the portfolio repository:

```bash
git clone https://github.com/MyNameIsForrestGump/machine-learning-portfolio.git
```

Navigate to this project's directory, then install the dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Run the notebooks in the following order:

1. `notebooks/EDA.ipynb`
2. `notebooks/Preprocessing.ipynb`
3. `notebooks/Model_Training.ipynb`

The preprocessing notebook generates `processed_data.csv`, which is subsequently used for model training.

The model training notebook performs hyperparameter optimization, compares the classifiers, and saves the selected model.

## Key Takeaways

This project provided practical experience in:

- Building an end-to-end machine learning workflow
- Handling missing values and imbalanced classification datasets
- Encoding categorical variables and engineering features
- Comparing linear and ensemble classification models
- Applying cross-validation and hyperparameter optimization
- Evaluating models using the F1-score
- Saving trained models for later use

## Limitations and Future Improvements

The current preprocessing workflow performs feature selection using correlations computed before the train/test split. This can introduce data leakage into the evaluation process.

Potential improvements include:

- Moving data-dependent feature selection into the training pipeline
- Using a separate final test set for unbiased model comparison
- Adding precision-recall curves and confusion matrix visualizations
- Evaluating threshold selection and probability calibration
- Improving the inference interface
- Investigating model interpretability and feature importance

## Context

This project was completed as part of a three-month Machine Learning training programme at IT School.

It is included in my [Machine Learning Portfolio](https://github.com/MyNameIsForrestGump/machine-learning-portfolio), alongside university coursework and other applied AI projects.
