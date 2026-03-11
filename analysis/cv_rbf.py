import pandas as pd
import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import make_scorer, mean_pinball_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin

from pathlib import Path
import sys

# Get the grandparent directory
parent_root = Path(__file__).resolve().parents[1]
# Add it to sys.path
sys.path.append(str(parent_root))
from data_utils import PrimaryFeatureEngineerExpert

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
fe = PrimaryFeatureEngineerExpert(drop_cols=exclude).fit(df_train_raw)

X_train = fe.transform(df_train_raw)
X_val = fe.transform(df_val_raw)

features = [c for c in X_train.columns if c not in exclude]

X_train, X_val = X_train[features], X_val[features]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

X = X_train
y = y_train

class AddRBFFeatures(BaseEstimator, TransformerMixin):
    def __init__(self,
                 gamma_temp=0.01,
                 gamma_neb=0.002,
                 gamma_wc=1e-6,
                 n_features=5):
        self.gamma_temp = gamma_temp
        self.gamma_neb = gamma_neb
        self.gamma_wc = gamma_wc
        self.n_features = n_features
        # fixed anchors
        self.temp_anchors = np.linspace(273.15 - 10, 273.15 + 35, 10)
        self.neb_anchors = np.array([0.0, 0.5, 1.0])
        self.wc_anchors = np.linspace(0, 2000, 2)

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("AddRBFFeatures must receive a pandas DataFrame")
        return self

    def _rbf(self, x, c, gamma):
        return np.exp(-gamma * (x - c) ** 2)

    def transform(self, X):
        X = X.copy()

        w = 2 * np.pi / 365.25
        for i in range(1, self.n_features + 1):
            X[f'cos{i}'] = np.cos(X['Time'] * w * i)
            X[f'sin{i}'] = np.sin(X['Time'] * w * i)
        # TEMP RBF FEATURES
        temp_vals = X["Temp"].values

        for i, c in enumerate(self.temp_anchors):
            X[f"Temp_K_RBF_{i}"] = self._rbf(temp_vals, c, self.gamma_temp)

        # NEBULOSITY RBF FEATURES
        neb_vals = X["Nebulosity"].values

        for i, c in enumerate(self.neb_anchors):
            X[f"Neb_RBF_{i}"] = self._rbf(neb_vals, c, self.gamma_neb)

        # WINDCHILL RBF FEATURES
        wc_vals = X["Wind_Chill"].values

        for i, c in enumerate(self.wc_anchors):
            X[f"WindChill_RBF_{i}"] = self._rbf(wc_vals, c, self.gamma_wc)

        return X 

temp_min = 273.15 - 10
temp_max = 273.15 + 35
temp_anchors = np.linspace(temp_min, temp_max, 10).reshape(-1,1)

neb_anchors = np.array([[0.0],[0.5],[1.0]])

wc_anchors = np.linspace(0, X_train['Wind_Chill'].max(), 2).reshape(-1,1)

pipeline = Pipeline([
    ("rbf", AddRBFFeatures()),
    ("scaler", StandardScaler()),
    ("model", QuantileRegressor())
])

print(pipeline.get_params().keys())


# we know the scale of hyperparameters and want to know the best values
param_grid = {
    "rbf__n_features": [5, 6, 7, 8],
    "rbf__gamma_temp": [0.002, 0.005, 0.01, 0.02],
    "rbf__gamma_neb":  [0.001, 0.005, 0.01],
    "rbf__gamma_wc":   [1e-7, 1e-6, 1e-5],
    "model__alpha": [0.0004, 0.0008]
}

quantile = 0.8

pinball_scorer = make_scorer(
    mean_pinball_loss,
    alpha=quantile,
    greater_is_better=False
)
cv = TimeSeriesSplit(n_splits=10)

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring=pinball_scorer,
    n_jobs=-1
)

grid.fit(X, y)

best_model = grid.best_estimator_

# takes around 5-7 min
print(grid.best_params_)
# {'model__alpha': 0.0004, 'rbf__gamma_neb': 0.001, 'rbf__gamma_temp': 0.01, 'rbf__gamma_wc': 1e-07, 'rbf__n_features': 7}
print(grid.best_score_)