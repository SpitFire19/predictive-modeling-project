import pandas as pd
import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
from data_utils import FeatureEngineerExpertReg
from sklearn.model_selection import TimeSeriesSplit
from data_utils import AdaptiveKalman
import lightgbm as lgb

# Ignore some warnings
warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-darkgrid')
QUANTILE = 0.8
ALPHA = 0.0004 # Value chosen by CV
# Kalman parameters
Q_FINAL = 10
R_FINAL = 1
BIAS_SHIFT = -500 # prioritize correcting 'negative' residues

train = pd.read_csv("Data/net-load-forecasting-during-soberty-period/train.csv")
test = pd.read_csv("Data/net-load-forecasting-during-soberty-period/test.csv")

mask_train = train["Date"] < "2022-04-01"
mask_cal = (train["Date"] >= "2022-04-01") & (train["Date"] < "2022-06-01")
mask_val = train["Date"] >= "2022-06-01"

df_train_raw = train[mask_train]
df_cal_raw = train[mask_cal]
df_val_raw = train[mask_val]

fe = FeatureEngineerExpertReg().fit(df_train_raw)

X_train = fe.transform(df_train_raw)
X_cal = fe.transform(df_cal_raw)
X_val = fe.transform(df_val_raw)

y_train = df_train_raw['Net_demand']
y_cal= df_cal_raw['Net_demand']
y_val = df_val_raw['Net_demand']

exclude = ["Date","Net_demand","Load","Solar_power", 'Wind',"Wind_power","WeekDays","Id","Usage","Year","Month"]
features = [c for c in X_train.columns if c not in exclude]

X_train, X_cal, X_val = X_train[features], X_cal[features], X_val[features]

test_fe = fe.transform(test)
X_test = test_fe[features]

# true_test = test["Net_demand.1"].shift(-1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_cal_scaled = scaler.transform(X_cal)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
model = QuantileRegressor(quantile=0.8, alpha=ALPHA, solver='highs')
model.fit(X_train_scaled, y_train)

pred_raw = model.predict(X_val_scaled)

res_first = model.predict(X_test_scaled)

fig, axes = plt.subplots(3, 1, figsize=(15, 12))
plt.subplots_adjust(hspace=0.3)
res_train = y_train - model.predict(X_train_scaled)

res_val = y_val - model.predict(X_val_scaled)
train_set = lgb.Dataset(X_train_scaled, label=res_train)
val_set   = lgb.Dataset(X_val_scaled, label=res_val)

tree_depth = 6
tree_leaves = 2 ** tree_depth - 1

params = {
    "objective": "quantile",
    "metric": "quantile",
    "alpha": 0.6,
    "learning_rate": 0.03,
    "num_leaves": tree_leaves,
    "min_data_in_leaf": 25,        
    "bagging_fraction": 0.70, # draw 70% of data at random
    "bagging_freq": 1,        # and do so at each iteration 
    "lambda_l2": 10,
    "max_depth": tree_depth,
    "extra_trees": True,      # Reducec overfitting and smoothes out noise
    "verbose": -1,
    "seed": 42
}

gbm_model = lgb.train(
    params,
    train_set,
    num_boost_round=5000,
    valid_sets=[train_set, val_set],
    callbacks=[
    lgb.early_stopping(stopping_rounds=100), # early stopping
    lgb.log_evaluation(period=1000)          # logging
])

corr_model = gbm_model

kalman = AdaptiveKalman(Q_FINAL,R_FINAL, BIAS_SHIFT, quantile=0.8)
preds_cal = model.predict(X_cal_scaled) + corr_model.predict(X_cal_scaled)
kalman.calibrate(preds_cal, y_cal)
preds_val_raw = model.predict(X_val_scaled) + corr_model.predict(X_val_scaled)
preds_val = kalman.validate(preds_val_raw, y_val)

model.fit(X_train_scaled, y_train)

# 5. Submission
pred_res = corr_model.predict(X_val_scaled)

residues_correction = corr_model.predict(X_val_scaled)
test_pred = pred_raw + residues_correction

last_pred = model.predict(scaler.transform(X_val.iloc[[-1]]))[0] + corr_model.predict(scaler.transform(X_val.iloc[[-1]]))[0]
last_pred -= - kalman.BIAS_SHIFT
preds_test = []
lag_vals = test_fe["Net_demand.1"].values

for x, y_lag in zip(X_test_scaled, lag_vals):
    kalman.update(y_lag, last_pred)
    pred_base_today = model.predict(x.reshape(1, -1))[0] + corr_model.predict(x.reshape(1, -1))[0]
    final_pred = pred_base_today + kalman.BIAS_SHIFT + kalman.bias
    preds_test.append(final_pred)
    last_pred = pred_base_today

submission = pd.DataFrame({"Id": test["Id"], "Net_demand": np.maximum(preds_test, 0)})
# print("True pinball loss with residues correction + smoothing:", pinball_loss(true_test, preds_test, 0.8))

submission.to_csv("submission_pca.csv", index=False)
