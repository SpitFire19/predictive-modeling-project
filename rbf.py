
import pandas as pd
import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_pinball_loss
import matplotlib.pyplot as plt
import warnings
from data_utils import FeatureEngineerExpertReg, AdaptiveKalman
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
def pinball_loss(y, yhat, tau):
    return np.mean(np.maximum(tau * (y - yhat),
                              (tau - 1) * (y - yhat)))
# Configuration
warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-darkgrid')
QUANTILE = 0.8
ALPHA = 0.001

# RÉGLAGES POUR APLATIR LA COURBE

Q_FINAL = 2500.0
R_FINAL = 150.0
BIAS_SHIFT = -1500.0

# ============================================================================
# 3. EXÉCUTION & CALCUL DES PERFORMANCES
# ============================================================================
train = pd.read_csv("Data/net-load-forecasting-during-soberty-period/train.csv")
test = pd.read_csv("Data/net-load-forecasting-during-soberty-period/test.csv")

fe = FeatureEngineerExpertReg().fit(train)
train_fe = fe.transform(train)
test_fe = fe.transform(test)

exclude = ["Date","Net_demand","Load","Solar_power","Wind_power","WeekDays","Id","Usage","Year","Month","toy"]
features = [c for c in train_fe.columns if c not in exclude]

mask_train = train_fe["Date"] < "2022-04-01"
mask_cal = (train_fe["Date"] >= "2022-04-01") & (train_fe["Date"] < "2022-06-01")
mask_val = train_fe["Date"] >= "2022-06-01"

X_train, y_train = train_fe[mask_train][features], train_fe[mask_train]["Net_demand"]
X_cal, y_cal = train_fe[mask_cal][features], train_fe[mask_cal]["Net_demand"]
X_val, y_val = train_fe[mask_val][features], train_fe[mask_val]["Net_demand"]

# print(X_cal.shape)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(test_fe[features])
gammas = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5])
gammas /= 10

for gamma in gammas:
    feature_map = Nystroem(kernel='rbf', gamma=gamma, n_components=300, random_state=42)
    
    # This creates a "lookup table" of 500 basis points from your training data
    X_train = feature_map.fit_transform(X_train)
    X_test = feature_map.transform(X_test)
    # 3. Transform your features into RBF space
    
    alphas = np.linspace(0.0, 10, num=50)
    
    krr_rbf = KernelRidge(kernel='rbf', alpha=1.6326, gamma=0.05)
    krr_rbf.fit(X_train, y_train)
    pred_final = krr_rbf.predict(X_test)
    test = pd.read_csv("Data/net-load-forecasting-during-soberty-period/test.csv")
    true_test = test["Net_demand.1"].shift(-1)
    true_test.loc[394] = pred_final[394]
    print(f"True pinball loss with gamma = {gamma}:", pinball_loss(true_test, np.maximum(pred_final, 0), 0.8))

