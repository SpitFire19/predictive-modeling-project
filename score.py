import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import os

df_train = pd.read_csv('./Data/net-load-forecasting-during-soberty-period/train.csv')
df_test = pd.read_csv('./Data/net-load-forecasting-during-soberty-period/test.csv')

df_train['Time'] = (
    pd.to_datetime(df_train['Date']) - pd.Timestamp("1970-01-01")
).dt.days
df_test['Time'] = (
    pd.to_datetime(df_test['Date']) - pd.Timestamp("1970-01-01")
).dt.days

# Perform variable selection
# LightGBM
# XGBoost
# Les variables pertinentes, feature engineering
# Ecrire les tests pour les modeles. model.score
# Time series forecasting != tabular data
# sktime: Time series forecasting
feature_list = ['Time', 'toy', 'Temp', 'Net_demand.1', 'Net_demand.7', 'Temp_s99', 
                'WeekDays', 'BH', 'Temp_s95_max', 'Temp_s99_max', 'Summer_break', 
                'Christmas_break', 'Temp_s95_min', 'Temp_s99_min', 'DLS']

X_train, y_train = df_train[feature_list], df_train['Net_demand']
X_test = df_test[feature_list]

reg = LinearRegression().fit(X_train, y_train)
print(reg.score(X_train, y_train))

# print(reg.predict(X_test))
