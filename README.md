# Predictive-Modeling-for-Sales-Forecasting

# Project description
This project implements various machine learning techniques to predict daily sales for Rossmann stores using the Kaggle Rossmann Store Sales dataset. The goal is to preprocess the data, engineer features, and train models to accurately forecast sales. The models used include Linear Regression, Random Forest, and XGBoost.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Processing](#data-processing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [License](#license)

# Installation
To run this project, you need to have Python installed along with the following libraries:
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
statsmodels
prophet
darts
You can install the required libraries using the following command:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels prophet darts
```

# Usage
1. Clone the Repository
```bash
git clone https://github.com/your_username/Sales-Forecasting.git
cd Sales-Forecasting
```

2. run the script
```bash
pmsf_model.ipynb
```

# Project Structure
```bash
Sales-Forecasting/
├── pmsf_model.ipynb                   # Main script with the code
├── train.csv                          # Training dataset
├── test.csv                           # Test dataset
├── store.csv                          # Store dataset
├── README.md                          # Project README file
```

# Data Preprocessing
The dataset is loaded using pandas and merged with store information. Missing values are handled by filling them with appropriate values. Date features are extracted, and categorical variables are one-hot encoded.
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb

# Load the datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
store_df = pd.read_csv('store.csv')

# Merge train and store data
train_df = train_df.merge(store_df, on='Store', how='left')
test_df = test_df.merge(store_df, on='Store', how='left')

# Handle missing values
train_df['CompetitionDistance'].fillna(train_df['CompetitionDistance'].max(), inplace=True)
test_df['CompetitionDistance'].fillna(test_df['CompetitionDistance'].max(), inplace=True)
train_df['Promo2SinceYear'].fillna(0, inplace=True)
train_df['Promo2SinceWeek'].fillna(0, inplace=True)
test_df['Promo2SinceYear'].fillna(0, inplace=True)
test_df['Promo2SinceWeek'].fillna(0, inplace=True)
train_df['PromoInterval'].fillna('None', inplace=True)
test_df['PromoInterval'].fillna('None', inplace=True)

# Extract date features
train_df['Date'] = pd.to_datetime(train_df['Date'])
test_df['Date'] = pd.to_datetime(test_df['Date'])
train_df['Year'] = train_df['Date'].dt.year
train_df['Month'] = train_df['Date'].dt.month
train_df['Day'] = train_df['Date'].dt.day
train_df['WeekOfYear'] = train_df['Date'].dt.isocalendar().week
test_df['Year'] = test_df['Date'].dt.year
test_df['Month'] = test_df['Date'].dt.month
test_df['Day'] = test_df['Date'].dt.day
test_df['WeekOfYear'] = test_df['Date'].dt.isocalendar().week

# Feature engineering
train_df['CompetitionOpenSince'] = 12 * (train_df['Year'] - train_df['CompetitionOpenSinceYear']) + (train_df['Month'] - train_df['CompetitionOpenSinceMonth'])
test_df['CompetitionOpenSince'] = 12 * (test_df['Year'] - test_df['CompetitionOpenSinceYear']) + (test_df['Month'] - test_df['CompetitionOpenSinceMonth'])
train_df['Promo2Since'] = 12 * (train_df['Year'] - train_df['Promo2SinceYear']) + (train_df['WeekOfYear'] - train_df['Promo2SinceWeek'])
test_df['Promo2Since'] = 12 * (test_df['Year'] - test_df['Promo2SinceYear']) + (test_df['WeekOfYear'] - test_df['Promo2SinceWeek'])
train_df["CompetitionOpenSince"] = train_df['CompetitionOpenSince'].apply(lambda x: x if x > 0 else 0)
test_df['CompetitionOpenSince'] = test_df['CompetitionOpenSince'].apply(lambda x: x if x > 0 else 0)

# One hot encode categorical values
train_df = pd.get_dummies(train_df, columns=['StoreType', 'Assortment', 'StateHoliday', 'PromoInterval'])
test_df = pd.get_dummies(test_df, columns=['StoreType', 'Assortment', 'StateHoliday', 'PromoInterval'])

# Ensure both train and test have the same columns after one-hot encoding
train_columns = set(train_df.columns)
test_columns = set(test_df.columns)
for col in train_columns - test_columns:
    test_df[col] = 0
for col in test_columns - train_columns:
    train_df[col] = 0
test_df = test_df[train_df.columns]

# Drop unnecessary columns
columns_to_drop = ['Date', 'Customers', 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth', 'Promo2SinceYear', 'Promo2SinceWeek']
train_df.drop(columns=columns_to_drop, inplace=True)
test_df.drop(columns=columns_to_drop, inplace=True)
```

# Feature Engineering
Additional features are created to capture the duration of competition and promotions. Negative values are replaced with zero.
```python
# Feature engineering
train_df['CompetitionOpenSince'] = 12 * (train_df['Year'] - train_df['CompetitionOpenSinceYear']) + (train_df['Month'] - train_df['CompetitionOpenSinceMonth'])
test_df['CompetitionOpenSince'] = 12 * (test_df['Year'] - test_df['CompetitionOpenSinceYear']) + (test_df['Month'] - test_df['CompetitionOpenSinceMonth'])
train_df['Promo2Since'] = 12 * (train_df['Year'] - train_df['Promo2SinceYear']) + (train_df['WeekOfYear'] - train_df['Promo2SinceWeek'])
test_df['Promo2Since'] = 12 * (test_df['Year'] - test_df['Promo2SinceYear']) + (test_df['WeekOfYear'] - test_df['Promo2SinceWeek'])
train_df["CompetitionOpenSince"] = train_df['CompetitionOpenSince'].apply(lambda x: x if x > 0 else 0)
test_df['CompetitionOpenSince'] = test_df['CompetitionOpenSince'].apply(lambda x: x if x > 0 else 0)
```

# Model Training
The data is split into training and testing sets. Linear Regression, Random Forest, and XGBoost models are trained on the preprocessed data.
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Define features and target
X_train = train_df.drop(columns=['Sales'])
y_train = train_df['Sales']
X_test = test_df[X_train.columns]

# Preprocess the data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), X_train.select_dtypes(include=['int64', 'float64']).columns),
        ('cat', SimpleImputer(strategy='most_frequent'), X_train.select_dtypes(include=['object', 'category']).columns)
    ],
    remainder='passthrough'
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler())
])

X_train_preprocessed = pipeline.fit_transform(X_train)
X_test_preprocessed = pipeline.transform(X_test)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_preprocessed, y_train)
y_pred_lr = lr_model.predict(X_test_preprocessed)
mse_lr = mean_squared_error(y_train, lr_model.predict(X_train_preprocessed))
rmse_lr = np.sqrt(mse_lr)
print(f"Linear Regression RMSE: {rmse_lr}")

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_preprocessed, y_train)
y_pred_rf = rf_model.predict(X_test_preprocessed)
mse_rf = mean_squared_error(y_train, rf_model.predict(X_train_preprocessed))
rmse_rf = np.sqrt(mse_rf)
print(f"Random Forest RMSE: {rmse_rf}")

# XGBoost Regressor
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train_preprocessed, y_train)
y_pred_xgb = xgb_model.predict(X_test_preprocessed)
mse_xgb = mean_squared_error(y_train, xgb_model.predict(X_train_preprocessed))
rmse_xgb = np.sqrt(mse_xgb)
print(f"XGBoost RMSE: {rmse_xgb}")
```

# Model Evaluation
The models are evaluated using Root Mean Squared Error (RMSE). The results are as follows:
```bash
Linear Regression RMSE: 2525.77
Random Forest RMSE: 312.74
XGBoost RMSE: 1179.86
```

# License
This README template includes all the pertinent information about your project, such as installation instructions, usage, project structure, data processing, model training, model evaluation, and details about the web application. It also includes sections for contributing and licensing, which are important for open-source projects.

