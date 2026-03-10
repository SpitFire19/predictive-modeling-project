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
BIAS_SHIFT = -500

train = pd.read_csv("Data/net-load-forecasting-during-soberty-period/train.csv")
test = pd.read_csv("Data/net-load-forecasting-during-soberty-period/test.csv")

mask_train = train["Date"] < "2018-06-01"
mask_cal = (train["Date"] >= "2022-04-01") & (train["Date"] < "2022-06-01")
mask_val = train["Date"] >= "2018-06-01"

df_train_raw = train[mask_train]
df_cal_raw = train[mask_cal]
df_val_raw = train[mask_val]

fe = DefaultFeatureEngineerExpert().fit(df_train_raw)

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

true_test = test["Net_demand.1"].shift(-1)

#print(len(train_fe[features].columns.tolist())) # 72

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_cal_scaled = scaler.transform(X_cal)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
model = QuantileRegressor(quantile=0.8, alpha=0.01, solver='highs')
model.fit(X_train_scaled, y_train)
pred_val = model.predict(X_val_scaled)
pb1 =  pinball_loss(pred_val, y_val, 0.8)
print(f'Model with default features performance: ', pb1)
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
k = 20
plt.figure(figsize=(8,6))
plt.barh(importance["feature"].head(k)[::-1],
         importance["abs_coef"].head(k)[::-1])
plt.title(f"Top {k} most important variables for QuantileRegressor", fontsize=20)
plt.xlabel("Absolute Coefficient", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.show()


fe = PrimaryFeatureEngineerExpert(drop_cols=exclude).fit(df_train_raw)
X_train_raw = fe.transform(df_train_raw)
X_val_raw = fe.transform(df_val_raw)

exclude = ["Date","Net_demand","Load","Solar_power", 'Wind',"Wind_power","WeekDays","Id","Usage","Year","Month"]
features = [c for c in X_train_raw.columns if c not in exclude]

X_train = X_train_raw[features]
X_val = X_val_raw[features]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

model = QuantileRegressor(quantile=0.8, alpha=0.01, solver='highs')
model.fit(X_train_scaled, y_train)
pred_val = model.predict(X_val_scaled)
# print(pred_val)
pb2 =  pinball_loss(pred_val, y_val, 0.8)
print(f'Model with primary features performance: ', pb2)


fe = FeatureEngineerExpertReg().fit(df_train_raw)
X_train_raw = fe.transform(df_train_raw)
X_val_raw = fe.transform(df_val_raw)

exclude = ["Date","Net_demand","Load","Solar_power", 'Wind',"Wind_power","WeekDays","Id","Usage","Year","Month"]
features = [c for c in X_train_raw.columns if c not in exclude]

X_train = X_train_raw[features]
X_val = X_val_raw[features]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

model = QuantileRegressor(quantile=0.8, alpha=0.01, solver='highs')
model.fit(X_train_scaled, y_train)
pred_val = model.predict(X_val_scaled)
# print(pred_val)
pb3 =  pinball_loss(pred_val, y_val, 0.8)
print(f'Model with RBF features performance: ', pb3)

coefs = model.coef_

print(f"Features list length: {len(features)}")
print(f"Model coefficients length: {len(model.coef_)}")

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

model_names = ['Default', 'Primary', 'RBF']
scores = [pb1, pb2, pb3]

plt.figure(figsize=(10, 6))
bars = plt.bar(model_names, scores, color=['#3498db', '#e67e22', '#e74c3c'])

# Add labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 4), ha='center', va='bottom', fontsize=12)

plt.ylabel('Pinball Loss', fontsize=14)
plt.title('Comparison of feature engineering strategies', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()