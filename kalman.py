## score 558
import pandas as pd
import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_pinball_loss
import matplotlib.pyplot as plt
import warnings
import sklearn.metrics as metrics
from statsmodels.api import OLS
from pygam import LinearGAM, s, f, te
from sklearn.preprocessing import LabelEncoder
def regression_results(y_true, y_pred):
    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance,4))    
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))

# Configuration
warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-darkgrid')
QUANTILE = 0.8
ALPHA = 0.005 

# RÉGLAGES POUR APLATIR LA COURBE
Q_FINAL = 2500.0
R_FINAL = 150.0
BIAS_SHIFT = -1500.0 

# ============================================================================
# 1. FILTRE DE KALMAN
# ============================================================================
class AdaptiveKalman:
    def __init__(self, quantile=0.8, Q=2500.0, R=150.0):
        self.quantile = quantile
        self.bias = 0.0
        self.P = 1000.0
        self.Q = Q
        self.R = R

    def update(self, y_true, y_pred_base):
        residual = y_true - (y_pred_base + BIAS_SHIFT + self.bias)
        self.P += self.Q
        K = self.P / (self.P + self.R)
        weight = self.quantile if residual > 0 else (1 - self.quantile)
        self.bias += K * residual * weight
        self.P *= (1 - K)
        return self.bias

# ============================================================================
# 2. FEATURE ENGINEERING (VERSION ULTRA-ROBUSTE)
# ============================================================================
class FeatureEngineerExpert:
    def __init__(self):
        self.train_date_min = None

    def fit(self, df):
        self.train_date_min = pd.to_datetime(df["Date"]).min()
        return self

    def transform(self, df):
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df["Month"] = df["Date"].dt.month
        df["Time"] = (df["Date"] - self.train_date_min).dt.days
        mask_neb = df["Date"] < "2018-01-01"
        mask_neb_after = df["Date"] >= "2018-01-01"
        
        #df["Nebulosity"][mask_neb] = (df["Nebulosity"][mask_neb_after] - 
        #                                    df["Nebulosity"][mask_neb_after].mean()) / df["Nebulosity"][mask_neb_after].std()

        #df["Nebulosity"][mask_neb_after] = (df["Nebulosity"][mask_neb_after] - 
        #                                    df["Nebulosity"][mask_neb_after].mean()) / df["Nebulosity"][mask_neb_after].std()

        #df["Nebulosity_weighted"][mask_neb] = (df["Nebulosity_weighted"][mask_neb_after] - 
        #                                    df["Nebulosity_weighted"][mask_neb_after].mean()) / df["Nebulosity_weighted"][mask_neb_after].std()

        # df["Nebulosity_weighted"][mask_neb_after] = (df["Nebulosity_weighted"][mask_neb_after] - 
        #                                    df["Nebulosity_weighted"][mask_neb_after].mean()) / df["Nebulosity_weighted"][mask_neb_after].std()

        # Sobriété & Température
        inflection = (pd.to_datetime("2022-01-01") - self.train_date_min).days
        df["Sobriety_Trend"] = np.maximum(0, df["Time"] - inflection)
        df["is_holiday_season"] = df["Month"].isin([12, 1, 7, 8]).astype(int)
        df["Heating_Std"] = np.maximum(288.15 - df["Temp_s95"], 0)
        df["Thermal_Sobriety"] = df["Heating_Std"] * df["Sobriety_Trend"] / 1000
        df["Wind_Chill"] = df["Heating_Std"] * df["Wind_weighted"]
        
        # --- CORRECTION FINALE DU KEYERROR / ATTRIBUTEERROR ---
        df["is_weekend"] = df["WeekDays"].isin([5, 6]).astype(int)
        
        # Si Usage existe, on compare, sinon on met 1 par défaut pour les jours ouvrés
        if "Usage" in df.columns:
            df["work_activity"] = np.where((df["is_weekend"] == 0) & (df["Usage"] == "Public"), 1, 0)
        else:
            df["work_activity"] = np.where(df["is_weekend"] == 0, 1, 0)

        # Saisonnalité
        w = 2 * np.pi / 365.25
        for i in range(1, 7):
            df[f"cos{i}"] = np.cos(df["Time"] * w * i)
            df[f"sin{i}"] = np.sin(df["Time"] * w * i)
            
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].ffill().bfill().fillna(0)
        return df

# ============================================================================
# 3. EXÉCUTION & CALCUL DES PERFORMANCES
# ============================================================================
train = pd.read_csv("Data/net-load-forecasting-during-soberty-period/train.csv")
test = pd.read_csv("Data/net-load-forecasting-during-soberty-period/test.csv")


train["Date_num"] = pd.to_datetime(train["Date"])
test["Date_num"] = pd.to_datetime(test["Date"])

train["Date_num"] = train["Date_num"].dt.dayofyear
test["Date_num"] = test["Date_num"].dt.dayofyear

train["WeekDays3"] = (
    pd.to_datetime(train["Date_num"])
      .dt.day_name()
      .replace({"Tuesday":"WorkDay","Wednesday":"WorkDay","Thursday":"WorkDay"})
      .astype("category")
)
test["WeekDays3"] = (
    pd.to_datetime(test["Date_num"])
      .dt.day_name()
      .replace({"Tuesday":"WorkDay","Wednesday":"WorkDay","Thursday":"WorkDay"})
      .astype("category")
)

le = LabelEncoder()
train["WeekDays3"] = le.fit_transform(train["WeekDays3"])
test["WeekDays3"] = le.fit_transform(test["WeekDays3"])

fe = FeatureEngineerExpert().fit(train)
train_fe = fe.transform(train)
test_fe = fe.transform(test)

exclude = ["Date","Net_demand","Load", "Solar_power","Wind_power","WeekDays","Id","Usage","Year","Month","toy",
           "Holiday_zone_c", "is_holiday_season", "Temp_s95_max", "WeekDays3"]
gam_features = ["Date_num", "toy", "Temp", "Temp_s99", "Wind", 'Load.1', 'Load.7', 'Net_demand.1', 'Net_demand.7',
                'WeekDays', 'BH', 'Nebulosity_weighted', 'WeekDays3']

features = [c for c in train_fe.columns if c not in exclude]
print(len(test_fe[features].columns))


mask_train = train_fe["Date"] < "2022-04-01"
mask_cal = (train_fe["Date"] >= "2022-04-01") & (train_fe["Date"] < "2022-06-01")
mask_val = train_fe["Date"] >= "2022-06-01"

X_train_lm, y_train_lm = train_fe[mask_train][features], train_fe[mask_train]["Net_demand"]
X_train_gam, y_train_gam = train_fe[mask_train][gam_features], train_fe[mask_train]["Net_demand"]

X_cal_lm, y_cal_lm = train_fe[mask_cal][features], train_fe[mask_cal]["Net_demand"]
X_cal_gam, y_cal_gam = train_fe[mask_cal][gam_features], train_fe[mask_cal]["Net_demand"]


X_val_lm, y_val_lm = train_fe[mask_val][features], train_fe[mask_val]["Net_demand"]
X_val_gam, y_val_gam = train_fe[mask_val][gam_features], train_fe[mask_val]["Net_demand"]

scaler = StandardScaler()
X_train_lm_scaled = scaler.fit_transform(X_train_lm)

model = QuantileRegressor(quantile=0.8, alpha=ALPHA, solver='highs')
# features.append("Date")
X_train_gam = train_fe[mask_train][gam_features]
print(X_train_gam.dtypes)
X_train_gam_scaled = X_train_gam.copy()
print(X_train_lm.columns)
col = {name: i for i, name in enumerate(X_train_gam.columns)}
# print(X_train_gam.columns)
# print(X_train_gam_scaled.shape)

gam = LinearGAM(
    # Time and Trend
    s(col['Date_num'], n_splines=5) + 
    s(col['toy'], n_splines=30, basis='cp') + 
    
    # Weather and Variables
    s(col['Temp'], n_splines=10) +
    s(col['Temp_s99'], n_splines=10) +
    s(col['Wind']) +
    
    # Lags (Autoregressive terms)
    s(col['Load.1']) + 
    s(col['Load.7']) + 
    s(col['Net_demand.1']) + 
    s(col['Net_demand.7']) +
    
    # Categorical Factors
    f(col['WeekDays3']) + 
    f(col['BH']) +
    
    # Interaction (Tensor Product)
    te(col['Date_num'], col['Nebulosity_weighted'], n_splines=[5, 10])
)
model.fit(X_train_lm_scaled, y_train_lm)
lam = np.logspace(-3, 3, 11)
gam.fit(X_train_gam_scaled, y_train_gam)
gam.gridsearch(X_train_gam_scaled, y_train_gam, lam=lam)

# Calcul Pinball Loss Train
loss_tr_lm = mean_pinball_loss(y_train_lm, model.predict(X_train_lm_scaled), alpha=0.8)
loss_tr_gam = mean_pinball_loss(y_train_gam, gam.predict(X_train_gam_scaled), alpha=0.8)

# Application Kalman Adaptive
kalman_lm = AdaptiveKalman(quantile=0.8, Q=Q_FINAL, R=R_FINAL)
kalman_gam = AdaptiveKalman(quantile=0.8, Q=Q_FINAL, R=R_FINAL)

print(X_val_gam.shape)
for x, y_t in zip(scaler.transform(X_cal_lm), y_cal_lm):
    p_b_lm = model.predict(x.reshape(1,-1))[0]
    kalman_lm.update(y_t, p_b_lm)

for x, y_t in zip(X_cal_gam.values, y_cal_gam):
    p_b_gam = gam.predict(x.reshape(1,-1))[0]
    kalman_gam.update(y_t, p_b_gam)


preds_val_lm = []
preds_val_gam = []

bias_hist_lm = []
bias_hist_gam = []

for x, y_t in zip(scaler.transform(X_val_lm), y_val_lm):
    p_b_lm = model.predict(x.reshape(1,-1))[0]
    final_p_lm = p_b_lm + BIAS_SHIFT + kalman_lm.bias
    preds_val_lm.append(final_p_lm)
    bias_hist_lm.append(kalman_lm.bias)
    kalman_lm.update(y_t, p_b_lm)


for x, y_t in zip(X_val_gam.values, y_val_gam):
    p_b_gam = gam.predict(x.reshape(1,-1))[0]
    final_p_gam = p_b_gam + BIAS_SHIFT + kalman_gam.bias
    preds_val_gam.append(final_p_gam)
    bias_hist_gam.append(kalman_gam.bias)
    kalman_gam.update(y_t, p_b_gam)

loss_v_lm = mean_pinball_loss(y_val_lm, preds_val_lm, alpha=0.8)
loss_v_gam = mean_pinball_loss(y_val_gam, preds_val_gam, alpha=0.8)

# regression_results(preds_val_lm, y_val_lm)


# print(OLS(y_train,X_train).fit().summary())

# ============================================================================
# 4. RÉSULTATS & VISUALISATION
# ============================================================================
print(f"\n" + "="*30)
print(f"PINBALL LOSS TRAIN: {loss_tr_lm:.2f}")
print(f"PINBALL LOSS VAL:   {loss_v_lm:.2f}")
print(f"DIFFÉRENCE:         {loss_tr_lm - loss_v_lm:.2f}")
print("="*30)
# regression_results(preds_val_gam, y_val_gam)
print(f"\n" + "="*30)
print(f"GAM PINBALL LOSS TRAIN: {loss_tr_gam:.2f}")
print(f"GAM PINBALL LOSS VAL:   {loss_v_gam:.2f}")
print(f"DIFFÉRENCE:         {loss_tr_gam - loss_v_gam:.2f}")
print("="*30)


fig, ax = plt.subplots(3, 1, figsize=(12, 18))
"""
# Graphique 1 : Réel vs Prédit
ax[0].plot(train_fe[mask_val]["Date"], y_val, label="Réel", color='black', alpha=0.3)
ax[0].plot(train_fe[mask_val]["Date"], preds_val, label="Prédiction (Shift + Kalman)", color='red')
ax[0].set_title("Validation : Suivi de la demande")
ax[0].legend()

# Graphique 2 : Biais Kalman
ax[1].plot(train_fe[mask_val]["Date"], bias_hist, color='purple')
ax[1].axhline(0, color='black', linestyle='--')
ax[1].set_title("Évolution de la correction Kalman (Bias)")

# Graphique 3 : Résidus Cumulés

residus = np.array(y_val) - np.array(preds_val)
ax[2].plot(train_fe[mask_val]["Date"], np.cumsum(residus), color='green', linewidth=2)
ax[2].axhline(0, color='black', linestyle='-')
ax[2].set_title("Somme Cumulée des Résidus (Stabilité de l'erreur)")

plt.tight_layout()
plt.show()
"""

# ============================================================================
# 5. SOUMISSION
# ============================================================================
preds_test = []
last_p_base = model.predict(scaler.transform(X_val_lm.iloc[[-1]]))[0]

for i, row in test_fe.iterrows():
    kalman_lm.update(row["Net_demand.1"], last_p_base)
    x_scaled = scaler.transform(row[features].values.reshape(1, -1))
    p_base_today = model.predict(x_scaled)[0]
    preds_test.append(p_base_today + BIAS_SHIFT + kalman_lm.bias)
    last_p_base = p_base_today

submission = pd.DataFrame({"Id": test["Id"], "Net_demand": np.maximum(preds_test, 0)})
submission.to_csv("submission_v8_final.csv", index=False)