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
ALPHA = 0.00084 # in [0, 0.001]

# smoothing parameter
BIAS_SHIFT = 800.0 # -800
 
# ============================================================================
# 3. EXÉCUTION & CALCUL DES PERFORMANCES
# ============================================================================
train = pd.read_csv("Data/net-load-forecasting-during-soberty-period/train.csv")
test = pd.read_csv("Data/net-load-forecasting-during-soberty-period/test.csv")

fe = FeatureEngineerExpertReg().fit(train)
train_fe = fe.transform(train)
test_fe = fe.transform(test)

# is removing wind really necessary ?
exclude = ["Date","Net_demand","Load","Solar_power", 'Wind',"Wind_power","WeekDays","Id","Usage","Year","Month"]


mask_train = train_fe["Date"] < "2022-04-01"
mask_cal = (train_fe["Date"] >= "2022-04-01") & (train_fe["Date"] < "2022-06-01")
mask_val = train_fe["Date"] >= "2022-06-01"

X_train, y_train = train_fe[mask_train][features], train_fe[mask_train]["Net_demand"]
X_cal, y_cal = train_fe[mask_cal][features], train_fe[mask_cal]["Net_demand"]
X_val, y_val = train_fe[mask_val][features], train_fe[mask_val]["Net_demand"]
X_test = test_fe[features]

true_test = test["Net_demand.1"].shift(-1)

print(len(train_fe[features].columns.tolist())) # 103
# print(*train_fe[features].columns)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_scaled = scaler.transform(train_fe[features])
X_cal_scaled = scaler.transform(X_cal)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


X = X_scaled
y = train_fe["Net_demand"]

def k_CV(X, y, alpha, n_splits):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, validation_idx in tscv.split(X):
        X_train, X_valn = X[train_idx], X[validation_idx]
        y_train, y_valn = y[train_idx], y[validation_idx]
        model = QuantileRegressor(quantile=0.8, alpha=alpha, solver='highs')
        model.fit(X_train, y_train)
        preds = model.predict(X_valn)

        score = pinball_loss(y_valn, preds, 0.8)
        scores.append(score)

    # print("CV scores:", scores)
    print(f"Mean CV score for alpha =  {alpha}:", np.mean(scores))

k = 10
"""
alphas = np.logspace(-6, 0, 7) # the best are 0.0001, 0.001, 0.01
for alpha in alphas:
    k_CV(X, y, alpha = alpha, n_splits=k) # baseline = 305.739
"""
# We find out that reasonable values for alpha are in [0.0001, 0.01] or in [0.01, 0.1]
print(f'k={k}')
alphas = np.linspace(0.0001, 0.001, 10) # test from 0.0007 to 0.0008
print(*alphas)
for alpha in alphas:
    k_CV(X, y, alpha = alpha, n_splits=k) # baseline = 305.739


"""
alphas = np.linspace(0.001, 0.01, 10) # test from 0.0007 to 0.0008
print(*alphas)
for alpha in alphas:
    k_CV(X, y, alpha = alpha, n_splits=k) # baseline = 305.739
"""