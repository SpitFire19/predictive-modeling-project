import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_pinball_loss
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf

from pathlib import Path
import sys
# Get the grandparent directory (the parent of the folder your script is in)
parent_root = Path(__file__).resolve().parents[1]

# Add it to sys.path
sys.path.append(str(parent_root))

# Import your package
from data_utils import AdaptiveKalman, DefaultFeatureEngineerExpert, FeatureEngineerExpertGBM

Q_FINAL = 2500.0
R_FINAL = 150.0
BIAS_SHIFT = -1500.0

def pinball_loss(y, yhat, tau):
    return np.mean(np.maximum(tau * (y - yhat),
                              (tau - 1) * (y - yhat)))

train = pd.read_csv("Data/net-load-forecasting-during-soberty-period/train.csv")
test = pd.read_csv("Data/net-load-forecasting-during-soberty-period/test.csv")

fe = FeatureEngineerExpertGBM().fit(train)

train["WeekDays3"] = (
    pd.to_datetime(train["Date"])
      .dt.day_name()
      .replace({"Tuesday":"WorkDay","Wednesday":"WorkDay","Thursday":"WorkDay"})
      .astype("category")
)
test["WeekDays3"] = (
    pd.to_datetime(test["Date"])
      .dt.day_name()
      .replace({"Tuesday":"WorkDay","Wednesday":"WorkDay","Thursday":"WorkDay"})
      .astype("category")
)
le = LabelEncoder()
train["WeekDays3"] = le.fit_transform(train["WeekDays3"])
test["WeekDays3"] = le.fit_transform(test["WeekDays3"])

train = fe.transform(train)
test = fe.transform(test)

TARGET = "Net_demand"
diff_features = ['Wind_power', 'Load', 'Net_demand', 'Solar_power']

def stress_test(params, train_set, val_set):
    model = lgb.train(params, train_set, num_boost_round=1000, 
                             valid_sets=[train_set, val_set], 
                             callbacks=[ lgb.early_stopping(stopping_rounds=100)])
    

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
features = [c for c in train if c not in exclude]
f_map = {name: i for i, name in enumerate(features)}

mask_train = train["Date"] < "2022-01-01"
mask_cal = (train["Date"] >= "2022-01-01") & (train["Date"] < "2022-03-01")
mask_val = train["Date"] >= "2022-03-01"

# print(train.dtypes)

X_train, y_train = train[mask_train][features], train[mask_train]["Net_demand"]
X_cal, y_cal = train[mask_cal][features], train[mask_cal]["Net_demand"]
X_val, y_val = train[mask_val][features], train[mask_val]["Net_demand"]

# 1. The "Deep Explorer" - MSE, High Depth
params_deep = {
    'boosting_type': 'gbdt',
    'objective': 'mse',           # Changed from 'regression' to 'mse'
    'metric': 'rmse',
    'num_leaves': 128,
    'max_depth': 20,
    'min_child_samples': 20,      # Same as min_data_in_leaf but more standard
    'verbose': -1,                # Turn off internal logging to see Python errors
    'force_col_wise': True,       # Forces a specific memory mapper
    'seed': 42
}

# 2. The "Stochastic Regularizer" - MAE, Heavy Subsampling (Great for Over-prediction)
params_robust = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'num_leaves': 31,
    'max_depth': 5,
    'bagging_fraction': 0.7,
    'pos_bagging_fraction': 0.7,
    'neg_bagging_fraction': 0.7,
    'bagging_freq': 5,
    'feature_fraction': 0.6,
    'lambda_l1': 15,
    'seed': 43
}

# 3. The "Conservative Quantile" - Targets 60th percentile to lower predictions
params_quantile = {
    'objective': 'quantile',
    'alpha': 0.6,
    'metric': 'quantile',
    'num_leaves': 64,
    'max_depth': 8,
    'learning_rate': 0.05,
    'seed': 44
}

# 4. The "Growth-Focused" - DART (Prevents over-reliance on first trees)
params_dart = {
    'boosting_type': 'dart',
    'objective': 'regression',
    'num_leaves': 45,
    'learning_rate': 0.1,
    'drop_rate': 0.1,
    'skip_drop': 0.5,
    'max_drop': 50,
    'seed': 45
}

# 5. The "Leaf-Wise Asymmetric" - GOSS + Tweedie (Handles skewed demand)
params_goss = {
    'boosting_type': 'goss',
    'objective': 'tweedie',
    'tweedie_variance_power': 1.5,
    'metric': 'tweedie',
    'num_leaves': 80,
    'top_rate': 0.2,    # GOSS specific: keep top 20% gradients
    'other_rate': 0.1,  # GOSS specific: sample 10% of low gradients
    'seed': 46
}
train_set = lgb.Dataset(X_train, label=y_train)
val_set   = lgb.Dataset(X_val, label=y_val)
# Training Loop Example
models = {}
for name, p in [('Deep', params_deep), ('Robust', params_robust), 
                ('Quantile', params_quantile), ('Dart', params_dart), 
                ('Goss', params_goss)]:
    
    # Assuming dtrain is your lgb.Dataset
    models[name] = lgb.train(p, train_set, num_boost_round=1000, 
                             valid_sets=[train_set, val_set], 
                             callbacks=[ lgb.early_stopping(stopping_rounds=100)])

meta_features = np.column_stack([
    models['Deep'].predict(X_cal),
    models['Robust'].predict(X_cal),
    models['Quantile'].predict(X_cal),
    models['Dart'].predict(X_cal),
    models['Goss'].predict(X_cal)
])

# 2. Train a Ridge model to find the best weights
# Set positive=True so models don't get "negative" weights
meta_model = Ridge(alpha=1.0, positive=True)
meta_model.fit(meta_features, y_cal)

cal_features = np.column_stack([m.predict(X_cal) for m in models.values()])
val_features = np.column_stack([m.predict(X_val) for m in models.values()])

# 2. Fit the Ridge "Judge" on the calibration set
# positive=True ensures we don't 'invert' a model's logic

# 3. Get the "Base Ensemble" forecast
ensemble_cal = meta_model.predict(cal_features)
ensemble_val = meta_model.predict(val_features)

true_test = test["Net_demand.1"].shift(-1)
test_features = np.column_stack([m.predict(test[features]) for m in models.values()])
print(test_features.shape)
static_preds = meta_model.predict(test_features)
Q = 0.002 
R = 0.04  

# 3. Initialization 
# We start with the bias found at the end of your 'calibrate_kalman' window
# This ensures we don't start from zero on June 1st
val_static_preds = ensemble_cal
initial_error = y_cal.iloc[-1] - ensemble_cal[-1]
current_bias = initial_error 
P = 1.0  
kalman_val_preds = []

# 4. Sequential Loop (Simulating real-time updates)
actuals_val = y_cal.to_numpy()

for i in range(len(val_static_preds)):
    # --- PREDICT STEP ---
    P = P + Q 
    
    # Apply the current bias estimate to the static GBM ensemble prediction
    # If bias is negative (over-prediction), this pulls the forecast DOWN
    current_prediction = val_static_preds[i] + current_bias
    kalman_val_preds.append(current_prediction)
    
    # --- UPDATE STEP ---
    # In validation, we use the actual value to 'teach' the filter for the next step
    actual = actuals_val[i]
    residual = actual - current_prediction
    
    # Kalman Gain
    K = P / (P + R)
    
    # Update the bias for the NEXT hour
    current_bias = current_bias + K * residual
    P = (1 - K) * P

kalman_val_preds = np.array(kalman_val_preds)

print("Ensemble Loss:", pinball_loss(true_test, static_preds, 0.8))
print("Kalman Adjusted Loss:", pinball_loss(true_test, kalman_val_preds, 0.8))

print("True pinball loss with GBM 5 ensembling", pinball_loss(true_test, static_preds, 0.8))