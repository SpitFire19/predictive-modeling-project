import pandas as pd
import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import QuantileRegressor
import lightgbm as lgb

from pathlib import Path
import sys

# Get the grandparent directory
parent_root = Path(__file__).resolve().parents[1]
# Add it to sys.path
sys.path.append(str(parent_root))
from data_utils import RBFFeatureEngineerExpert, pinball_loss

# Ignore some warnings
warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-darkgrid')
QUANTILE = 0.8
ALPHA = 0.0008

train = pd.read_csv("Data/net-load-forecasting-during-soberty-period/train.csv")
test = pd.read_csv("Data/net-load-forecasting-during-soberty-period/test.csv")

mask_train = train["Date"] < "2022-04-01"
mask_val = train["Date"] >= "2022-04-01"

df_train_raw = train[mask_train]
df_val_raw = train[mask_val]

y_train = df_train_raw['Net_demand']
y_val = df_val_raw['Net_demand']

exclude = ["Date","Net_demand","Load","Solar_power", 'Wind',"Wind_power","WeekDays","Id","Usage","Year","Month"]
fe = RBFFeatureEngineerExpert().fit(df_train_raw)

X_train = fe.transform(df_train_raw)
X_val = fe.transform(df_val_raw)

features = [c for c in X_train.columns if c not in exclude]

X_train, X_val = X_train[features], X_val[features]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

model = QuantileRegressor(quantile=0.8, alpha=ALPHA, solver='highs')
model.fit(X_train_scaled, y_train)

res_train = y_train - model.predict(X_train_scaled)
raw_pred = model.predict(X_val_scaled)
res_val = y_val - raw_pred # calculate the residues of main model
alphas = [0.4, 0.5, 0.6] # GBM systematically over-predicts, so we don't set 
                                    # alpha > 0.6
depths = [4, 5, 6, 7] # Bigger depth leads to overfitting, we want the residue 
                   # correction model to stay as simple as possible
l2_regs = [0, 5, 10, 20] # bigger values tend to hurt the performance
min_leaf_sizes = [20, 25, 30, 35] # leaf sizes less that 20 tend to overfit

train_set = lgb.Dataset(X_train_scaled, label=res_train)
val_set   = lgb.Dataset(X_val_scaled, label=res_val)
best_loss =  1e9 
best_params = {}
"""
Our grid is more sparse than with linear model
as it takes more time to train GBM than QuantileRegressor
We don't use TimeSeriesSplit to spend feasible amount of 
time for model selection (10 minutes instead of several hours)
"""

for alpha in alphas:
    print("Test alpha=", alpha)
    for depth in depths:
        num_leaves = 2 ** depth - 1 
        for l2_reg in l2_regs:
            for min_leaf_size in min_leaf_sizes:
                params = {
                    "objective": "quantile",
                    "metric": "quantile",
                    "alpha": alpha,
                    "learning_rate": 0.03,
                    "num_leaves": num_leaves,
                    "min_data_in_leaf": min_leaf_size,
                    "bagging_fraction": 0.70, # draw 70% of data at random
                    "bagging_freq": 1,        # and do so at each iteration
                    "lambda_l2": l2_reg,
                    "max_depth": depth,
                    "extra_trees": True,      # Reduces overfitting and smoothes out noise
                                              # by intoducing more randomization
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
                    lgb.log_evaluation(period=0)          # logging
                ])
                res_pred = corr_model.predict(X_val_scaled)
                val_loss = pinball_loss(y_val, raw_pred + res_pred, 0.8)
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_params = {'alpha': alpha, 'max_depth': depth, 
                                   'min_leaf_size': min_leaf_size, 'l2_reg': l2_reg}
                
print(best_params)
# {'alpha': 0.6, 'max_depth': 6, 'min_leaf_size': 25, 'l2_reg': 10}