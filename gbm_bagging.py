import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_pinball_loss


import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf

from pathlib import Path
import sys

import warnings
from sklearn.exceptions import DataConversionWarning

# Suppress only this specific warning
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Get the grandparent directory (the parent of the folder your script is in)
parent_root = Path(__file__).resolve().parents[1]

# Add it to sys.path
sys.path.append(str(parent_root))

# Import your package
from data_utils import FeatureEngineerExpertGBM, VikingBias, pinball_loss

Q_FINAL = 2500.0
R_FINAL = 150.0
BIAS_SHIFT = -1500.0

train = pd.read_csv("Data/net-load-forecasting-during-soberty-period/train.csv")
test = pd.read_csv("Data/net-load-forecasting-during-soberty-period/test.csv")
init_features = list(train.columns)
fe = FeatureEngineerExpertGBM().fit(train)
train_fe = fe.transform(train)
test_fe = fe.transform(test)

TARGET = "Net_demand"
diff_features = ['Wind_power', 'Load', 'Net_demand', 'Solar_power']

# print(train.columns.tolist())
    
# Les jours feries en facteur
# Load avec vent el le soleil, avec lasso pour choisir les features. prevoir la moyenne (quantile gaussien), enlever bh et time et nebulosity
# GBM model complet, importance de variables dans le modele gb, permutation based performance, faire attention a overfit avec boosting
# (tree-based gradient boosting) + te(Net.demand.1, Net.demand.7)
# GAM a 15 variables + Kalman
# Utiliser plus de liaisons (BAM = Big Additive Model)
# Predire la mediane (ou moyenne) peut etre mieux que le quantile 0.8
# 117 variables apres le lassos.csv")["Net_dema
# Utiliser les GAM univaries univaries, tracer une courbe pour chacune des covariables

true_test = test["Net_demand.1"].shift(-1)

# print(true_test)
exclude = ["Date","Net_demand","Load","Solar_power","Wind_power","WeekDays","Id","Usage","Month", "Sobriety_Trend"]
features = [c for c in train_fe if c not in exclude] 
reduced_features = [c for c in init_features if c not in exclude] 
mask_train = train["Date"] < "2022-01-01"
mask_cal = (train["Date"] >= "2022-01-01") & (train["Date"] < "2022-03-01")
mask_val = train["Date"] >= "2022-03-01"

# print(train.dtypes)

X_scaled = train_fe[features]
X_train, y_train = train_fe[mask_train][features], train_fe[mask_train]["Net_demand"]
X_cal, y_cal = train_fe[mask_cal][features], train_fe[mask_cal]["Net_demand"]
X_val, y_val = train_fe[mask_val][features], train_fe[mask_val]["Net_demand"]
X_test = test_fe[features]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_cal_scaled = scaler.transform(X_cal)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ---------- LightGBM quantile model ----------
tau = 0.8
params = {
    "objective": "quantile",
    "metric": "quantile",
    "alpha": 0.8,
    "learning_rate": 0.02,
    "num_leaves": 127,              # increase moderately
    "min_data_in_leaf": 20,        # slight increase
    "feature_fraction": 0.80,
    "lambda_l2": 20, 
    "lambda_l1": 2,
    "boosting": "gbdt",
    "max_depth": 8,
    "min_gain_to_split": 0.03,

    "extra_trees": True,           # Smooths out noise
    "path_smooth": 1.0,            # Reduces "staircase" effect

    "cat_smooth": 10,

    "verbose": -1,
    "seed": 42
}

train_set = lgb.Dataset(X_train_scaled, label=y_train)
val_set   = lgb.Dataset(X_val_scaled, label=y_val)

model = lgb.train(
    params,
    train_set,
    num_boost_round=5000,
    valid_sets=[train_set, val_set],
    callbacks=[
    lgb.early_stopping(stopping_rounds=100), # Increased from 5
    lgb.log_evaluation(period=1000)           # Only print every 100 rounds to keep logs clean
])

# ---------- validation ----------
val_pred = model.predict(X_val)
# print("Validatoin loss:", pinball_loss(y_val.values, val_pred, 0.8))
y = train['Net_demand']

res_first = model.predict(X_test_scaled)
print(f"Model 1 validation loss without VIKING:", pinball_loss(res_first, true_test, 0.8))

fig, axes = plt.subplots(3, 1, figsize=(15, 12))
plt.subplots_adjust(hspace=0.3)
axes[0].plot(X_test["Time"], res_first, color='teal', alpha=0.5)
axes[0].plot(X_test["Time"], true_test, color='red', alpha=0.5)

params_b = {
    "objective": "regression_l1",
    # "alpha": 0.8,              
    "learning_rate": 0.03,             
    "min_data_in_leaf": 30,
    "lambda_l2": 10,
    "extra_trees": True,
    "verbose": -1,
    "boosting": "goss",
    "max_depth":5,
    "num_leaves":31
}

model_compl = lgb.train(
    params_b,
    train_set,
    num_boost_round=5000,
    valid_sets=[train_set, val_set],
    callbacks=[
    lgb.early_stopping(stopping_rounds=100), # Increased from 5
    lgb.log_evaluation(period=1000)           # Only print every 100 rounds to keep logs clean
])

res_second = model_compl.predict(X_test_scaled)
print(f"Model 2 validation loss without VIKING:", pinball_loss(res_second, true_test, 0.8))
axes[1].plot(X_test["Time"], res_second, color='teal', alpha=0.5)
axes[1].plot(X_test["Time"], true_test, color='red', alpha=0.5)

# 3. Define the search grid for Lambda (lam)
# We test values from 10^-3 to 10^3. 
# This tells the model how "stiff" or "wiggly" the splines should be.
# lam_grid = np.logspace(-3, 3, 11)

# 4. Execute GridSearch
# This replaces the .fit() call. It will try 11 different versions of the model.
# pyGAM uses Generalized Cross-Validation (GCV) by default to pick the winner.
# gam_res.gridsearch(X_val_scaled, res_val, lam=lam_grid)
# print(gam_res.summary())

viking = VikingBias(alpha=0.05, bias_shift=BIAS_SHIFT)
w1, w2 = 0.5, 0.5
preds_cal = w1 *model.predict(X_cal_scaled) + w2 * model_compl.predict(X_cal_scaled)
viking.calibrate(preds_cal, y_cal)
preds_val_raw = w1 * model.predict(X_val_scaled) + w2 * model_compl.predict(X_val_scaled)
preds_val = viking.validate(preds_val_raw, y_val)

loss_v = pinball_loss(y_val, preds_val, 0.8)
print(f"Validation loss with VIKING:{loss_v}")

# 5. Final Prediction (Combined)

# ---------- final training ----------

last_p = w1 * model.predict(scaler.transform(X_val.iloc[[-1]]))[0] + w2 * model_compl.predict(scaler.transform(X_val.iloc[[-1]]))[0]
preds_test = []
lag_vals = test_fe["Net_demand.1"].values

for x, y_lag in zip(X_test_scaled, lag_vals):
    viking.update(y_lag, last_p)
    p_base_today = w1 * model.predict(x.reshape(1, -1))[0] + w2 * model_compl.predict(x.reshape(1, -1))[0]
    preds_test.append(p_base_today + BIAS_SHIFT + viking.bias)
    last_p = p_base_today

submission = pd.DataFrame({"Id": test["Id"], "Net_demand": np.maximum(preds_test, 0)})
true_test.loc[394] = preds_test[394]

res_final = preds_test
axes[2].plot(test_fe["Time"], res_final, color='teal', alpha=0.5)
axes[2].plot(test_fe["Time"], true_test, color='red', alpha=0.5)
plt.show()

print("True pinball loss with bagging + VIKING:", pinball_loss(true_test, np.maximum(preds_test, 0), 0.8))
submission.to_csv("submission_gbm_kalman_final1.csv", index=False)
