import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from scipy.stats import norm

# -------------------------------
# utilities
# -------------------------------

def pinball_loss(y, yhat, tau):
    return np.mean(np.maximum(tau * (y - yhat),
                              (tau - 1) * (y - yhat)))

def rolling_cv_splits(n_obs, n_folds=5):
    fold_size = n_obs // (n_folds + 1)
    splits = []
    for i in range(1, n_folds + 1):
        train_end = fold_size * i
        test_end = fold_size * (i + 1)
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, test_end)
        splits.append((train_idx, test_idx))
    return splits

# -------------------------------
# load data
# -------------------------------

train = pd.read_csv("Data/net-load-forecasting-during-soberty-period/train.csv")
test = pd.read_csv("Data/net-load-forecasting-during-soberty-period/test.csv")

train["Date"] = pd.to_datetime(train["Date"])
test["Date"] = pd.to_datetime(test["Date"])

# -------------------------------
# COVID weighting
# -------------------------------

train["weight"] = 1.0
covid_mask = train["Date"].between("2020-03-15", "2021-06-30")
train.loc[covid_mask, "weight"] = 0.5

# -------------------------------
# feature engineering
# -------------------------------

train["WeekDays3"] = train["WeekDays"].replace(
    {"Tuesday": "WorkDay", "Wednesday": "WorkDay", "Thursday": "WorkDay"}
)
test["WeekDays3"] = test["WeekDays"].replace(
    {"Tuesday": "WorkDay", "Wednesday": "WorkDay", "Thursday": "WorkDay"}
)
train["Temp_trunc1"] = np.maximum(train["Temp"] - 286, 0)
train["Temp_trunc2"] = np.maximum(train["Temp"] - 290, 0)
test["Temp_trunc1"] = np.maximum(test["Temp"] - 286, 0)
test["Temp_trunc2"] = np.maximum(test["Temp"] - 290, 0)

train["Temp2"] = train["Temp"] ** 2
test["Temp2"] = test["Temp"] ** 2

num_features = [
    "Temp", "Temp2", "Temp_trunc1", "Temp_trunc2",
    "Net_demand.1", "Net_demand.7"
]
cat_features = ["WeekDays3"]
all_features = num_features + cat_features

# -------------------------------
# cross-validation
# -------------------------------

tau = 0.8
splits = rolling_cv_splits(len(train), n_folds=5)
cv_scores = []

for train_idx, val_idx in splits:
    Xtr = train.iloc[train_idx][all_features]
    ytr = train.iloc[train_idx]["Net_demand"]
    wtr = train.iloc[train_idx]["weight"]
    print(train.iloc[val_idx]["Date"].max())
    Xva = train.iloc[val_idx][all_features]
    yva = train.iloc[val_idx]["Net_demand"]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features),
            ("num", "passthrough", num_features),
        ]
    )

    Xtr_enc = preprocess.fit_transform(Xtr)
    Xva_enc = preprocess.transform(Xva)
    
    model = LinearRegression()
    model.fit(Xtr_enc, ytr, sample_weight=wtr)

    yva_mean = model.predict(Xva_enc)

    res = ytr - model.predict(Xtr_enc)
    weighted_res = np.repeat(res, (wtr * 10).astype(int))
    q_shift = np.quantile(weighted_res, tau)

    yva_q = yva_mean + q_shift
    score = pinball_loss(yva, yva_q, tau) / len(Xva_enc)
    cv_scores.append(score)

cv_scores = np.array(cv_scores)
print("CV pinball loss per fold:", cv_scores)
print("Mean CV pinball loss:", cv_scores.mean())
print("Std CV pinball loss:", cv_scores.std())

# -------------------------------
# train final model
# -------------------------------

X_train = train[all_features]
y_train = train["Net_demand"]
w_train = train["weight"]
X_test = test[all_features]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features),
        ("num", "passthrough", num_features),
    ]
)

Xtr_enc = preprocess.fit_transform(X_train)
Xte_enc = preprocess.transform(X_test)

model = Ridge(alpha = 0.02)
model.fit(Xtr_enc, y_train, sample_weight=w_train)

y_test_mean = model.predict(Xte_enc)

residuals = y_train - model.predict(Xtr_enc)
weighted_residuals = np.repeat(residuals, (w_train * 10).astype(int))
q_shift = np.quantile(weighted_residuals, tau)

y_test_q = y_test_mean + q_shift
