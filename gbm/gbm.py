import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from sklearn.linear_model import LinearRegression, Ridge, QuantileRegressor, ElasticNet
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score


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
exclude = ["Date","Net_demand","Load","Solar_power","Wind_power","WeekDays","Id","Usage","Month","Sobriety_Trend"]
features = [c for c in train_fe if c not in exclude] 

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
    "alpha": 0.6,
    "learning_rate": 0.02,
    "num_leaves": 63,            # increase moderately
    "min_data_in_leaf": 20,        # slight increase
    "feature_fraction": 0.8,
    "bagging_fraction": 0.70,
    "bagging_freq": 1,
    "lambda_l2": 10,  
    "max_depth": 6,
    "min_gain_to_split": 0.03,
    "extra_trees": True,           # Smooths out noise
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

y = train['Net_demand']

res_first = model.predict(X_test_scaled)

fig, axes = plt.subplots(3, 1, figsize=(15, 12))
plt.subplots_adjust(hspace=0.3)
axes[0].plot(X_test["Time"], true_test - res_first, color='teal', alpha=0.5)
# axes[0].plot(X_test["Time"], res_first, color='teal', alpha=0.5)
 #axes[0].plot(X_test["Time"], true_test, color='red', alpha=0.5)

res_train = y_train - model.predict(X_train_scaled)
res_val = y_val - model.predict(X_val_scaled)

reg_residues = QuantileRegressor(quantile=0.8,
                                 solver='highs').fit(X_train_scaled, res_train)

corr_model = reg_residues

corr_model = ElasticNet(
    alpha=0.000095,        # final value: 0.0001
    l1_ratio=0.51,       # final value: 0.5 = balance L1/L2 penalty
    max_iter=10000,
    random_state=42,
)

corr_model.fit(X_train_scaled, res_train)

viking = VikingBias(alpha=0.07, bias_shift=BIAS_SHIFT)
preds_cal = model.predict(X_cal_scaled) + corr_model.predict(X_cal_scaled)
viking.calibrate(preds_cal, y_cal)
preds_val_raw = model.predict(X_val_scaled) + corr_model.predict(X_val_scaled)
preds_val = viking.validate(preds_val_raw, y_val)
loss_v = pinball_loss(y_val, preds_val, 0.8)
print(f"Validation loss with VIKING:{loss_v}")

# 5. Final Prediction (Combined)

final_pred = model.predict(X_test_scaled) + corr_model.predict(X_test_scaled)

print("True pinball loss without residues correction before VIKING", pinball_loss(true_test, final_pred, 0.8))
axes[1].plot(X_test["Time"], true_test - final_pred, color='teal', alpha=0.5)
# axes[1].plot(X_test["Time"], final_pred, color='teal', alpha=0.5)
# axes[1].plot(X_test["Time"], true_test, color='red', alpha=0.5)

last_p = model.predict(scaler.transform(X_val.iloc[[-1]]))[0] + corr_model.predict(scaler.transform(X_val.iloc[[-1]]))[0]
preds_test = []
lag_vals = test_fe["Net_demand.1"].values

for x, y_lag in zip(X_test_scaled, lag_vals):
    viking.update(y_lag, last_p)
    p_base_today = model.predict(x.reshape(1, -1))[0] + corr_model.predict(x.reshape(1, -1))[0]
    preds_test.append(p_base_today + BIAS_SHIFT + viking.bias)
    last_p = p_base_today

submission = pd.DataFrame({"Id": test["Id"], "Net_demand": np.maximum(preds_test, 0)})
true_test.loc[394] = preds_test[394]

res_final = preds_test

axes[2].plot(test_fe["Time"], true_test - preds_test, color='teal', alpha=0.5)
# axes[2].plot(test_fe["Time"], res_final, color='teal', alpha=0.5)
# axes[2].plot(test_fe["Time"], true_test, color='red', alpha=0.5)
# plt.show()

print("True pinball loss with residues correction + VIKING:", pinball_loss(true_test, np.maximum(preds_test, 0), 0.8))
submission.to_csv("submission_gbm_kalman_final1.csv", index=False)
