import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from scipy.stats import norm

from data_utils import date_raw_to_numeric

# -------------------------------
# utilities
# -------------------------------

def pinball_loss(y, yhat, tau):
    return np.mean(np.maximum(tau * (y - yhat),
                              (tau - 1) * (y - yhat)))

train = pd.read_csv("Data/net-load-forecasting-during-soberty-period/train.csv")
test = pd.read_csv("Data/net-load-forecasting-during-soberty-period/test.csv")

print(train.dtypes)

train = date_raw_to_numeric(train)
test = date_raw_to_numeric(test)

train["Date"] = pd.to_datetime(train["Date"])
test["Date"] = pd.to_datetime(test["Date"])

train["Date"] = train["Date"].dt.dayofyear
test["Date"] = test["Date"].dt.dayofyear

train["WeekDays3"] = train["WeekDays"].replace(
    {"Tuesday": "WorkDay", "Wednesday": "WorkDay", "Thursday": "WorkDay"}
)
test["WeekDays3"] = test["WeekDays"].replace(
    {"Tuesday": "WorkDay", "Wednesday": "WorkDay", "Thursday": "WorkDay"}
)

train["Temp_trunc1"] = np.maximum(train["Temp"] - 286, 0)
train["Temp_trunc2"] = np.maximum(train["Temp"] - 290, 0)
test["Temp_trunc1"] = np.maximum(test["Temp"] - 286, 0)
test["Temp_trunc2"] = np.maximum(test["Temp"] - 290, 0)

w = 2 * np.pi / 365
Nfourier = 10

for i in range(1, Nfourier + 1):
    train[f"cos{i}"] = np.cos(w * train["Time"] * i)
    train[f"sin{i}"] = np.sin(w * train["Time"] * i)
    test[f"cos{i}"]  = np.cos(w * test["Time"] * i)
    test[f"sin{i}"]  = np.sin(w * test["Time"] * i)

print(train.dtypes)

train["T1_cos1"] = train["Temp_trunc1"] * train["cos1"]
train["T2_cos1"] = train["Temp_trunc2"] * train["cos1"]
test["T1_cos1"]  = test["Temp_trunc1"] * test["cos1"]
test["T2_cos1"]  = test["Temp_trunc2"] * test["cos1"]

TARGET = "Net_demand"
diff_features = ['Wind_power', 'Load', 'Net_demand', 'Solar_power']

FEATURES = [c for c in train.columns if c != TARGET and c not in diff_features]

FEATURES = ["WeekDays3", "Net_demand.1", "Net_demand.7", 
            "Temp", "Temp_trunc1", "Temp_trunc2", "Nebulosity"] + \
[f"cos{i}" for i in range(1, Nfourier + 1)] + \
[f"sin{i}" for i in range(1, Nfourier + 1)]
    
# Les jours feries en facteur
# Load avec vent el le soleil, avec lasso pour choisir les features. prevoir la moyenne (quantile gaussien), enlever bh et time et nebulosity
# GBM model complet, importance de variables dans le modele gb, permutation based performance, faire attention a overfit avec boosting
# (tree-based gradient boosting) + te(Net.demand.1, Net.demand.7)
# GAM a 15 variables + Kalman
# Utiliser plus de liaisons (BAM = Big Additive Model)
# Predire la mediane (ou moyenne) peut etre mieux que le quantile 0.8
# 117 variables apres le lasso
# Utiliser les GAM univaries univaries, tracer une courbe pour chacune des covariables

X = train[FEATURES]
y = train[TARGET]
X_test = test[FEATURES]
print(train.dtypes)
# ---------- time split ----------
tau = 0.8
split = int(0.8 * len(train))

X_tr, X_val = X.iloc[:split], X.iloc[split:]
y_tr, y_val = y.iloc[:split], y.iloc[split:]

# ---------- LightGBM quantile model ----------
params = {
    "objective": "quantile",
    "alpha": tau,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 54,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": 42
}

train_set = lgb.Dataset(X_tr, label=y_tr)
val_set   = lgb.Dataset(X_val, label=y_val)

model = lgb.train(
    params,
    train_set,
    num_boost_round=3000,
    valid_sets=[train_set, val_set],
    callbacks=[
        lgb.early_stopping(stopping_rounds=3),
    ]

)

# ---------- validation ----------
val_pred = model.predict(X_val)
print("Pinball loss:", pinball_loss(y_val.values, val_pred, tau))

# ---------- final training ----------
final_model = lgb.train(
    params,
    lgb.Dataset(X, label=y),
    num_boost_round=model.best_iteration
)

# ---------- test prediction ----------
test_pred = final_model.predict(X_test)

submit = pd.read_csv("Data/sample_submission.csv")
submit["Net_demand"] = test_pred
submit.to_csv("Data/submission_light_gbm_python.csv", index=False)
