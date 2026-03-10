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

# true_test = test["Net_demand.1"].shift(-1)

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
model = QuantileRegressor(quantile=0.8, alpha=ALPHA, solver='highs')
model.fit(X_train_scaled, y_train)

pred_raw = model.predict(X_val_scaled)
# true_test.loc[394] = pred_raw[394]
print("Validation pinball loss without residues correction before smoothing", pinball_loss(y_val, pred_raw, 0.8))

