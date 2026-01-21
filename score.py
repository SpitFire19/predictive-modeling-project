import numpy as np
from sklearn.model_selection import PredefinedSplit
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_pinball_loss, mean_squared_error
import pandas as pd

from data_utils import date_raw_to_numeric
from data_utils import write_predictions_csv


df_train = pd.read_csv('./Data/net-load-forecasting-during-soberty-period/train.csv')
df_test = pd.read_csv('./Data/net-load-forecasting-during-soberty-period/test.csv')

df_train = date_raw_to_numeric(df_train)
df_test  = date_raw_to_numeric(df_test)


# Perform variable selection
# LightGBM
# XGBoost
# Les variables pertinentes, feature engineering
# Ecrire les tests pour les modeles. model.score
# Time series forecasting != tabular data
# sktime: Time series forecasting
# Pygamme
# OPERA
# AUTOARIMA
# VIKING

feature_list = ['Year', 'Time', 'toy', 'Temp', 'Net_demand.1', 'Net_demand.7', 'Temp_s99', 
                'WeekDays', 'BH', 'Temp_s95_max', 'Temp_s99_max', 'Summer_break', 
                'Christmas_break', 'Temp_s95_min', 'Temp_s99_min', 'DLS', ]


target = 'Net_demand'

X = df_train[feature_list]
y = df_train[target]

# Create boolean masks
mask_train = df_train['Year'] <= 2021
mask_test  = df_train['Year'] > 2021

# Build PredefinedSplit
# -1 = training, 0 = test
test_fold = np.where(mask_test, 0, -1)

ps = PredefinedSplit(test_fold)

# Extract train/test indices
train_idx, test_idx = next(ps.split())

# Index the split data
X_train = X.iloc[train_idx]
y_train = y.iloc[train_idx]

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

# Apply Quantile Regressor
qr = QuantileRegressor(quantile=0.8, alpha=0.1)
qr.fit(X_train, y_train)

# Predictions
y_train_pred = qr.predict(X_train)
y_test_pred  = qr.predict(X_test)


# Collect metrics
train_pinball = mean_pinball_loss(y_train, y_train_pred, alpha=0.8)
test_pinball  = mean_pinball_loss(y_test, y_test_pred, alpha=0.8)

train_rmse = mean_squared_error(y_train, y_train_pred) ** 0.5
test_rmse  = mean_squared_error(y_test, y_test_pred) ** 0.5

print(f"CV Train error: {train_pinball}")
print(f"CV Test error: {test_pinball}")

ids = df_test["Id"].values

X_test = df_test[feature_list]

y_test_pred  = qr.predict(X_test)

write_predictions_csv(ids, y_test_pred, "Data/net_demand_forecast.csv")

"""
Try to simulate the following
Data0$WeekDays2 <- weekdays(Data0$Date)
Data0$WeekDays3 <- forcats::fct_recode(Data0$WeekDays2, 'WorkDay'='Thursday' ,'WorkDay'='Tuesday', 'WorkDay' = 'Wednesday')
"""