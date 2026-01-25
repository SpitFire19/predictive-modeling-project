import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_pinball_loss, mean_squared_error

from scipy.stats import norm

from data_utils import date_raw_to_numeric
from data_utils import write_predictions_csv


df_train = pd.read_csv('./Data/net-load-forecasting-during-soberty-period/train.csv')
df_test = pd.read_csv('./Data/net-load-forecasting-during-soberty-period/test.csv')

# Preprocess table data



df_train = date_raw_to_numeric(df_train)
df_test  = date_raw_to_numeric(df_test)


df_train['WeekDays'] = df_train['WeekDays'].astype("category")
df_test['WeekDays'] = df_test['WeekDays'].astype("category")

df_train["Temp_trunc1"] = np.maximum(df_train["Temp"] - 286, 0)
df_train["Temp_trunc2"] = np.maximum(df_train["Temp"] - 290, 0)
df_test["Temp_trunc1"] = np.maximum(df_test["Temp"] - 286, 0)
df_test["Temp_trunc2"] = np.maximum(df_test["Temp"] - 290, 0)

df_train["WeekDays3"] = (
    pd.to_datetime(df_train["Date"])
      .dt.day_name()
      .replace({"Tuesday":"WorkDay","Wednesday":"WorkDay","Thursday":"WorkDay"})
      .astype("category")
)

df_test["WeekDays3"] = (
    pd.to_datetime(df_test["Date"])
      .dt.day_name()
      .replace({"Tuesday":"WorkDay","Wednesday":"WorkDay","Thursday":"WorkDay"})
      .astype("category")
)

w = 2 * np.pi / 365
Nfourier = 10   # R settles near this

for i in range(1, Nfourier + 1):
    df_train[f"cos{i}"] = np.cos(w * df_train["Time"] * i)
    df_train[f"sin{i}"] = np.sin(w * df_train["Time"] * i)
    df_test[f"cos{i}"]  = np.cos(w * df_test["Time"] * i)
    df_test[f"sin{i}"]  = np.sin(w * df_test["Time"] * i)


features = (
    ["Temp", "Temp_trunc1", "Temp_trunc2", "Net_demand.1", "Net_demand.7"]
    + [f"cos{i}" for i in range(1, Nfourier + 1)]
    + [f"sin{i}" for i in range(1, Nfourier + 1)]
)


preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(
            drop='first',
            sparse_output=True,
            handle_unknown='ignore'
        ), ['WeekDays3']),
        ('num', 'passthrough', features)
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(
            drop='first',
            handle_unknown='ignore',
            sparse_output=True
        ), ['WeekDays3']),
        ('num', 'passthrough', features)
    ]
)

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



target = 'Net_demand'

X = preprocess.fit_transform(df_train)
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
X_train = X[train_idx]
y_train = y[train_idx]

X_test_CV = X[test_idx]
y_test_CV = y[test_idx]


X_test = preprocess.transform(df_test)

# Apply regression

model = QuantileRegressor(
    quantile=0.8,   # τ = 0.8
    alpha=0.01,      # IMPORTANT: matches R's rq()
    solver="highs"  # most stable
)
model.fit(X, y)

# Predictions
y_pred = model.predict(X_test)
y_CV_pred = model.predict(X_test_CV)
test_pinball  = mean_pinball_loss(y_test_CV, y_CV_pred, alpha=0.8)
print(f"CV Test error: {test_pinball}")


# Submit

submit = pd.read_csv("Data/sample_submission.csv")
submit["Net_demand"] = y_pred
submit.to_csv("Data/submission_lm_python.csv", index=False)

write_predictions_csv(
    y_pred, 
    "Data/sample_submission.csv",
     "Data/submission_lm_python.csv" 
)