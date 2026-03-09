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
import lightgbm as lgb
from sklearn.metrics.pairwise import rbf_kernel

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.kernel_approximation import RBFSampler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import GridSearchCV


# Ignore some warnings
warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-darkgrid')
QUANTILE = 0.8
ALPHA = 0.0008

train = pd.read_csv("Data/net-load-forecasting-during-soberty-period/train.csv")
test = pd.read_csv("Data/net-load-forecasting-during-soberty-period/test.csv")

drop = ["Date","Net_demand","Load","Solar_power", 'Wind',"Wind_power","WeekDays","Id","Usage","Year","Month"]

fe = PrimaryFeatureEngineerExpert(drop, n_Fourier=6).fit(train)
train_fe = fe.transform(train)
test_fe = fe.transform(test)

mask_train = train["Date"] < "2022-04-01"
mask_val = train["Date"] >= "2022-04-01"

X_train, y_train = train_fe[mask_train], train[mask_train]["Net_demand"]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_scaled = scaler.transform(train_fe)
X_val_scaled = scaler.transform(train_fe[mask_val])
X_test_scaled = scaler.transform(test_fe)

X = train_fe
y = train["Net_demand"]

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer, mean_pinball_loss


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

wc_anchors = np.linspace(0, train_fe['Wind_Chill'].max(), 2).reshape(-1,1)

pipeline = Pipeline([
    ("rbf", AddRBFFeatures()),
    ("scaler", StandardScaler()),
    ("model", QuantileRegressor())
])

print(pipeline.get_params().keys())

param_grid = {
    "rbf__n_features": [5, 6, 7, 8, 9],
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

print(grid.best_params_)
# {'model__alpha': 0.0004, 'rbf__gamma_neb': 0.001, 'rbf__gamma_temp': 0.01, 'rbf__gamma_wc': 1e-07, 'rbf__n_features': 7}
print(grid.best_score_)