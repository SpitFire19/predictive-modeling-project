import pandas as pd
import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path
import sys

parent_root = Path(__file__).resolve().parents[1] # Get the grandparent directory
sys.path.append(str(parent_root)) # Add it to sys.path
from data_utils import RBFFeatureEngineerExpert, PrimaryFeatureEngineerExpert, DefaultFeatureEngineerExpert, pinball_loss

QUANTILE = 0.8
ALPHA = 0.0004 # Value chosen by CV
k = 20

def build_importance_plot(model, ax):
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
    ax.set_title("Absolute coefficient")
    ax.barh(importance["feature"].head(k)[::-1],
         importance["abs_coef"].head(k)[::-1],
         height=0.8)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 16), facecolor='white')
plt.style.use('seaborn-v0_8-darkgrid')

train = pd.read_csv("Data/net-load-forecasting-during-soberty-period/train.csv")

mask_train = train["Date"] < "2022-04-01"
mask_cal = (train["Date"] >= "2022-04-01") & (train["Date"] < "2022-06-01")
mask_val = train["Date"] >= "2022-06-01"

df_train_raw = train[mask_train]
df_cal_raw = train[mask_cal]
df_val_raw = train[mask_val]

y_train = df_train_raw['Net_demand']
y_cal= df_cal_raw['Net_demand']
y_val = df_val_raw['Net_demand']


exclude = ["Date","Net_demand","Load","Solar_power", 'Wind',"Wind_power","WeekDays","Id","Usage","Year","Month"]
fe = DefaultFeatureEngineerExpert().fit(df_train_raw)

X_train = fe.transform(df_train_raw)
X_cal = fe.transform(df_cal_raw)
X_val = fe.transform(df_val_raw)

features = [c for c in X_train.columns if c not in exclude]

X_train, X_cal, X_val = X_train[features], X_cal[features], X_val[features]

y_train = df_train_raw['Net_demand']
y_cal= df_cal_raw['Net_demand']
y_val = df_val_raw['Net_demand']

X_train, X_cal, X_val = X_train[features], X_cal[features], X_val[features]
features = [c for c in X_train.columns if c not in exclude]

X_train, X_cal, X_val = X_train[features], X_cal[features], X_val[features]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_cal_scaled = scaler.transform(X_cal)
X_val_scaled = scaler.transform(X_val)
model = QuantileRegressor(quantile=0.8, alpha=ALPHA, solver='highs')
model.fit(X_train_scaled, y_train)
pred_val = model.predict(X_val_scaled)
pb1 =  pinball_loss(pred_val, y_val, 0.8)
print(f'Model with default features performance: ', pb1)

build_importance_plot(model, ax1)

plt.title(f"Top {k} variables for different QuantileRegressors", fontsize=20)

fe = PrimaryFeatureEngineerExpert(drop_cols=exclude).fit(df_train_raw)
X_train_raw = fe.transform(df_train_raw)
X_val_raw = fe.transform(df_val_raw)

features = [c for c in X_train_raw.columns if c not in exclude]

X_train = X_train_raw[features]
X_val = X_val_raw[features]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

model = QuantileRegressor(quantile=0.8, alpha=ALPHA, solver='highs')
model.fit(X_train_scaled, y_train)
pred_val = model.predict(X_val_scaled)
pb2 =  pinball_loss(pred_val, y_val, 0.8)
print(f'Model with primary features performance: ', pb2)
build_importance_plot(model, ax2)
plt.show()

fe = RBFFeatureEngineerExpert().fit(df_train_raw)

X_train = fe.transform(df_train_raw)
X_cal = fe.transform(df_cal_raw)
X_val = fe.transform(df_val_raw)

features = [c for c in X_train.columns if c not in exclude]

X_train, X_cal, X_val = X_train[features], X_cal[features], X_val[features]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_cal_scaled = scaler.transform(X_cal)
X_val_scaled = scaler.transform(X_val)
model = QuantileRegressor(quantile=0.8, alpha=ALPHA, solver='highs')
model.fit(X_train_scaled, y_train)

pred_raw = model.predict(X_val_scaled)
pb3 = pinball_loss(y_val, pred_raw, 0.8)
print(f'Model with RBF features performance: ', pb3)
print(f"RBF model features count: {len(features)}")


# Build comparison between feature engineering strategies
model_names = ['Default', 'Primary', 'RBF']
scores = [pb1, pb2, pb3]
plt.figure(figsize=(10, 6))

colors = ['#7a8d9a', '#b7a290', '#95a18d']
bars = plt.bar(model_names, scores, color=colors, edgecolor='#cccccc', linewidth=0.6, width=0.7)

# --- 4. DATA LABELS ---
# Position labels neatly above each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 15,
            f'{height:.2f}',  # Keep two decimals for cleaner display
            ha='center', va='bottom',
            fontsize=10,
            color='#4d4d4d', # Dark gray for contrast, not pure black
            weight='normal')
    

plt.ylabel('Pinball Loss', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
