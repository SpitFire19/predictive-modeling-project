## score 558

import pandas as pd
import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_pinball_loss
import matplotlib.pyplot as plt
import warnings

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

fe = FeatureEngineerExpert().fit(train)
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
print(X_cal.shape)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = QuantileRegressor(quantile=0.8, alpha=ALPHA, solver='highs')
model.fit(X_train_scaled, y_train)

# Calcul Pinball Loss Train
loss_tr = mean_pinball_loss(y_train, model.predict(X_train_scaled), alpha=0.8)

# Application Kalman Adaptive
kalman = AdaptiveKalman(quantile=0.8, Q=Q_FINAL, R=R_FINAL)
for x, y_t in zip(scaler.transform(X_cal), y_cal):
    p_b = model.predict(x.reshape(1,-1))[0]
    kalman.update(y_t, p_b)

preds_val = []
bias_hist = []
for x, y_t in zip(scaler.transform(X_val), y_val):
    p_b = model.predict(x.reshape(1,-1))[0]
    final_p = p_b + BIAS_SHIFT + kalman.bias
    preds_val.append(final_p)
    bias_hist.append(kalman.bias)
    kalman.update(y_t, p_b)

loss_v = mean_pinball_loss(y_val, preds_val, alpha=0.8)

# ============================================================================
# 4. RÉSULTATS & VISUALISATION
# ============================================================================
print(f"\n" + "="*30)
print(f"PINBALL LOSS TRAIN: {loss_tr:.2f}")
print(f"PINBALL LOSS VAL:   {loss_v:.2f}")
print(f"DIFFÉRENCE:         {loss_tr - loss_v:.2f}")
print("="*30)

fig, ax = plt.subplots(3, 1, figsize=(12, 18))

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

# ============================================================================
# 5. SOUMISSION
# ============================================================================
preds_test = []
last_p_base = model.predict(scaler.transform(X_val.iloc[[-1]]))[0]

for i, row in test_fe.iterrows():
    kalman.update(row["Net_demand.1"], last_p_base)
    x_scaled = scaler.transform(row[features].values.reshape(1, -1))
    p_base_today = model.predict(x_scaled)[0]
    preds_test.append(p_base_today + BIAS_SHIFT + kalman.bias)
    last_p_base = p_base_today

submission = pd.DataFrame({"Id": test["Id"], "Net_demand": np.maximum(preds_test, 0)})
submission.to_csv("submission_v8_final.csv", index=False)