import pandas as pd
import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
from data_utils import FeatureEngineerExpertReg, pinball_loss, DefaultFeatureEngineerExpert, PrimaryFeatureEngineerExpert
from sklearn.model_selection import TimeSeriesSplit
from data_utils import AdaptiveKalman
import lightgbm as lgb
from sklearn.metrics import mean_pinball_loss

# Ignore some warnings
warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-darkgrid')
QUANTILE = 0.8
ALPHA = 0.0004 # Value chosen by CV

train = pd.read_csv("Data/net-load-forecasting-during-soberty-period/train.csv")
test = pd.read_csv("Data/net-load-forecasting-during-soberty-period/test.csv")

mask_train = train["Date"] < "2022-04-01"
mask_cal = (train["Date"] >= "2022-04-01") & (train["Date"] < "2022-06-01")
mask_val = train["Date"] >= "2022-06-01"

df_train_raw = train[mask_train]
df_cal_raw = train[mask_cal]
df_val_raw = train[mask_val]

exclude = ["Date","Net_demand","Load","Solar_power", 'Wind',"Wind_power","WeekDays","Id","Usage","Year","Month"]
fe = FeatureEngineerExpertReg().fit(df_train_raw)

X_train = fe.transform(df_train_raw)
X_cal = fe.transform(df_cal_raw)
X_val = fe.transform(df_val_raw)

y_train = df_train_raw['Net_demand']
y_cal= df_cal_raw['Net_demand']
y_val = df_val_raw['Net_demand']

features = [c for c in X_train.columns if c not in exclude]

X_train, X_cal, X_val = X_train[features], X_cal[features], X_val[features]

test_fe = fe.transform(test)
X_test = test_fe[features]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_cal_scaled = scaler.transform(X_cal)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
model = QuantileRegressor(quantile=0.8, alpha=ALPHA, solver='highs')
model.fit(X_train_scaled, y_train)

pred_raw = model.predict(X_val_scaled)
manual_loss = mean_pinball_loss(y_val, pred_raw, alpha=0.8)
print("Validation pinball loss :", manual_loss)

res_train = y_train - model.predict(X_train_scaled)
res_val = y_val - model.predict(X_val_scaled)

model_bias = res_val.mean()
train_set = lgb.Dataset(X_train_scaled, label=res_train)
val_set   = lgb.Dataset(X_val_scaled, label=res_val)

alpha = 0.8
individual_loss = np.maximum(alpha * res_val, (alpha - 1) * res_val)

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
    "extra_trees": True,      # Reduces overfitting and smoothes out noise
    "verbose": -1,
    "seed": 42
}

corr_model = lgb.train(
    params,
    train_set,
    num_boost_round=5000,
    valid_sets=[train_set, val_set],
    callbacks=[
    lgb.early_stopping(stopping_rounds=100), # early stopping
    lgb.log_evaluation(period=1000)          # logging
])

preds_train_raw = model.predict(X_train_scaled) + corr_model.predict(X_train_scaled)
hybrid_train_res = y_train - preds_train_raw
preds_val_raw = model.predict(X_val_scaled) + corr_model.predict(X_val_scaled)
hybrid_res = y_val - preds_val_raw

# Kalman parameters
Q_FINAL = 1 # trust the measurements as much
R_FINAL = 1 # as we trust the predictions
BIAS_SHIFT = np.mean(hybrid_train_res) # our past estimations have bias

kalman = AdaptiveKalman(Q_FINAL,R_FINAL, BIAS_SHIFT, quantile=0.8)
preds_cal = model.predict(X_cal_scaled) + corr_model.predict(X_cal_scaled)
kalman.calibrate(preds_cal, y_cal)
preds_val_raw = model.predict(X_val_scaled) + corr_model.predict(X_val_scaled)

last_val = X_cal_scaled[-1].reshape(1, -1)
last_pred = model.predict(last_val)[0] + corr_model.predict(last_val)[0]

lag_y_val = y_val
preds_val = []
for x, y_today in zip(X_val_scaled, lag_y_val):
    # get current features and actual value
    x_val = x.reshape(1, -1)

    # raw prediction (linear + gbm) for today
    pred_base_today = model.predict(x_val)[0] + corr_model.predict(x_val)[0]

    # update the filter for the next iteration
    final_pred = pred_base_today + kalman.BIAS_SHIFT + kalman.bias
    preds_val.append(final_pred)

    # update the filter for the next iteration
    kalman.update(y_today, pred_base_today)
    
print("Validation pinball loss :", pinball_loss(y_val, preds_val_raw, 0.8))
print("Validation pinball loss after Kalman filter:", pinball_loss(y_val, preds_val, 0.8))

print(f"Final Bias: {kalman.bias}")
print(f"Kalman Shift: {kalman.BIAS_SHIFT}")

preds_test = []

X_val_np = np.array(X_val_scaled)
last_val = X_val_np[-1].reshape(1, -1)
X_test_np = np.array(X_test_scaled)

first_x = X_test_np[0].reshape(1, -1)
last_pred = model.predict(last_val)[0] + corr_model.predict(last_val)[0]

y_true_lag = X_test['Net_demand.1'].shift(-1)
y_true_lag.iloc[-1] = 0

kalman.bias=0.0

for i in range(len(X_test)):
    # get current features and actual value
    x_today = X_test_np[i].reshape(1, -1)

    # raw prediction (linear + gbm) for today
    pred_base_today = model.predict(x_today)[0] + corr_model.predict(x_today)[0]
    # use the bias from previous step to obtain the final prediction
    final_pred = pred_base_today + kalman.BIAS_SHIFT + kalman.bias
    preds_test.append(final_pred)

    # update the filter for the next iteration
    y_true_today = y_true_lag[i]
    kalman.update(y_true_today, pred_base_today)

# convert results for plotting
preds_test = np.array(preds_test)
submission = pd.DataFrame({"Id": test["Id"], "Net_demand":preds_test})

submission.to_csv("submission_final.csv", index=False)
