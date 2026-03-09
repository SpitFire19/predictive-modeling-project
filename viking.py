import pandas as pd
import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from pygam import LinearGAM, s, f, te
import matplotlib.pyplot as plt
import warnings
from data_utils import FeatureEngineerExpertReg, PrimaryFeatureEngineerExpert, DefaultFeatureEngineerExpert, pinball_loss, rss, tss
from sklearn.model_selection import TimeSeriesSplit
from data_utils import ExponentialSmoothing
import lightgbm as lgb

# Ignore some warnings
warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-darkgrid')
QUANTILE = 0.8
ALPHA = 0.0004 # Value chosen by CV
# smoothing parameter
BIAS_SHIFT = -500.0

train = pd.read_csv("Data/net-load-forecasting-during-soberty-period/train.csv")
test = pd.read_csv("Data/net-load-forecasting-during-soberty-period/test.csv")

fe = FeatureEngineerExpertReg().fit(train)
train_fe = fe.transform(train)
test_fe = fe.transform(test)

# is removing wind really necessary ?
exclude = ["Date","Net_demand","Load","Solar_power", 'Wind',"Wind_power","WeekDays","Id","Usage","Year","Month"]
features = [c for c in train_fe.columns if c not in exclude]
f_map = {name: i for i, name in enumerate(features)}


mask_train = train_fe["Date"] < "2022-04-01"
mask_cal = (train_fe["Date"] >= "2022-04-01") & (train_fe["Date"] < "2022-06-01")
mask_val = train_fe["Date"] >= "2022-06-01"

X_train, y_train = train_fe[mask_train][features], train_fe[mask_train]["Net_demand"]
X_cal, y_cal = train_fe[mask_cal][features], train_fe[mask_cal]["Net_demand"]
X_val, y_val = train_fe[mask_val][features], train_fe[mask_val]["Net_demand"]
X_test = test_fe[features]

true_test = test["Net_demand.1"].shift(-1)

# print(len(train_fe[features].columns.tolist())) # 72

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_scaled = scaler.transform(train_fe[features])
X_cal_scaled = scaler.transform(X_cal)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
model = QuantileRegressor(quantile=0.8, alpha=ALPHA, solver='highs')
model.fit(X_train_scaled, y_train)

coefs = model.coef_

# Build dataframe
importance = pd.DataFrame({
    "feature": features,
    "coef": coefs,
    "abs_coef": np.abs(coefs)
})

# Sort
importance = importance.sort_values("abs_coef", ascending=False)

# Plot top important variables
k = 30
plt.figure(figsize=(8,6))
plt.barh(importance["feature"].head(k)[::-1],
         importance["abs_coef"].head(k)[::-1])
plt.title(f"Top {k} most important variables for QuantileRegressor", fontsize=20)
plt.xlabel("Absolute Coefficient", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)   # slightly smaller since there are many labels
# plt.show()

res_first = model.predict(X_test_scaled)

fig, axes = plt.subplots(3, 1, figsize=(15, 12))
plt.subplots_adjust(hspace=0.3)
# axes[0].plot(X_test["Time"], true_test - res_first, color='teal', alpha=0.5)
# axes[0].plot(X_test["Time"], true_test, color='red', alpha=0.5)

# Calculate residues
# print(*X_test.columns)
res_train = y_train - model.predict(X_train_scaled)
y = train_fe["Net_demand"]

res_x = y - model.predict(X_scaled)
res_val = y_val - model.predict(X_val_scaled)
terms = (
    te(f_map['Temp'], f_map['Nebulosity'], n_splines=[8, 8]) +
    te(f_map['Temp'], f_map['Time'], n_splines=[8, 10]) +
    te(f_map['Solar_power.1'], f_map['Nebulosity'], n_splines=[10, 10]) +    
    s(f_map['toy'], n_splines=25, penalties='periodic') + 
    te(f_map['Load.1'], f_map['Load.7'], n_splines=[7, 7]) +
    te(f_map['Temp_s95'], f_map['Time'], n_splines=[10, 10]) +
    f(f_map['Christmas_break']) +
    f(f_map['BH_Holiday']) +
    f(f_map['DLS']) # + f(f_map['Summer_break'])
)


# gam_res = LinearGAM(terms, lam = 10)
# gam_res.fit(X_train_scaled, res_train)
# gam_res.summary()

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
    "extra_trees": True,      # Reduce overfitting and smooth out noise
    "verbose": -1,
    "seed": 42
}

gbm_model = lgb.train(
    params,
    train_set,
    num_boost_round=5000,
    valid_sets=[train_set, val_set],
    callbacks=[
    lgb.early_stopping(stopping_rounds=100), # Increased from 5
    lgb.log_evaluation(period=1000)           # Only print every 100 rounds to keep logs clean
])

corr_model = gbm_model

# Apply error correction
# window = 1/alpha, so we choose 14 day cycle with 
viking = ExponentialSmoothing(alpha=0.07, bias_shift=BIAS_SHIFT)
preds_cal = model.predict(X_cal_scaled) + corr_model.predict(X_cal_scaled)
viking.calibrate(preds_cal, y_cal)
preds_val_raw = model.predict(X_val_scaled) + corr_model.predict(X_val_scaled)
preds_val = viking.validate(preds_val_raw, y_val)

scores = []
X = X_scaled

model.fit(X_train_scaled, y_train)
y_pred_full = []
viking1 = ExponentialSmoothing(alpha=0.07, bias_shift=BIAS_SHIFT)

k = 3470

for i in range(len(X[:k + 1])):
    x_scaled = X[i].reshape(1, -1)
    base_pred = model.predict(x_scaled)[0] + corr_model.predict(x_scaled)[0] 
    
    final_pred = base_pred + BIAS_SHIFT + viking1.bias
    y_pred_full.append(final_pred)
    
    # update bias using true value
    viking1.update(y[i], base_pred)

viking1 = ExponentialSmoothing(alpha=0.07, bias_shift=BIAS_SHIFT)


plt.plot(train_fe["Date"].loc[:k], train_fe["Net_demand"].loc[:k], color="red", label="Predicted")
# Points for actual values
# plt.scatter(train_fe["Date"].loc[:k], y.loc[:k], color="black", s=10, alpha=0.5, label="Actual")

plt.title("Electricity Demand: Actual vs Predicted")
plt.xlabel("Date")
plt.ylabel("Demand (MW)")
plt.legend()
# plt.show()


# 5. Submission

pred_res = corr_model.predict(X_val_scaled)
R2= r2_score(res_val, pred_res)
print("Residue correction model performance:", R2)

pred_raw = model.predict(X_val_scaled)
# true_test.loc[394] = pred_raw[394]
print("Validation pinball loss without residues correction before smoothing", pinball_loss(y_val, pred_raw, 0.8))

residues_correction = corr_model.predict(X_val_scaled)
test_pred = pred_raw + residues_correction
# true_test.loc[394] = test_pred[394]
print("Validation pinball loss with residues correction before smoothing", pinball_loss(y_val, test_pred, 0.8))
# axes[1].plot(X_test["Time"], true_test - test_pred, color='teal', alpha=0.5)
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
axes[2].plot(test_fe["Time"], true_test - preds_test, color='teal', alpha=0.5)
# axes[2].plot(test_fe["Time"], true_test, color='red', alpha=0.5)
# plt.show()
# true_test.loc[394] = preds_test[394]
print("True pinball loss with residues correction + smoothing::", pinball_loss(true_test, np.maximum(preds_test, 0), 0.8))
submission.to_csv("submission_pca.csv", index=False)