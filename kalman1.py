## score 558

import pandas as pd
import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_pinball_loss
import matplotlib.pyplot as plt
import warnings
from data_utils import FeatureEngineerExpertReg, AdaptiveKalman
# Configuration
warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-darkgrid')
QUANTILE = 0.8
ALPHA = 0.0009

# RÉGLAGES POUR APLATIR LA COURBE

Q_FINAL = 2550.0
R_FINAL = 250.0
BIAS_SHIFT = -2550.0

# ============================================================================
# 3. EXÉCUTION & CALCUL DES PERFORMANCES
# ============================================================================
train = pd.read_csv("Data/net-load-forecasting-during-soberty-period/train.csv")
test = pd.read_csv("Data/net-load-forecasting-during-soberty-period/test.csv")

fe = FeatureEngineerExpertReg().fit(train)
train_fe = fe.transform(train)
test_fe = fe.transform(test)

exclude = ["Date","Net_demand","Load","Solar_power","Wind_power","WeekDays","Id","Usage","Year","Month"]
features = [c for c in train_fe.columns if c not in exclude]

mask_train = train_fe["Date"] < "2022-04-01"
mask_cal = (train_fe["Date"] >= "2022-04-01") & (train_fe["Date"] < "2022-06-01")
mask_val = train_fe["Date"] >= "2022-06-01"

X_train, y_train = train_fe[mask_train][features], train_fe[mask_train]["Net_demand"]
X_cal, y_cal = train_fe[mask_cal][features], train_fe[mask_cal]["Net_demand"]
X_val, y_val = train_fe[mask_val][features], train_fe[mask_val]["Net_demand"]

# print(X_cal.shape)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = QuantileRegressor(quantile=0.8, alpha=ALPHA, solver='highs')
model.fit(X_train_scaled, y_train)

# Calcul Pinball Loss Train
loss_tr = mean_pinball_loss(y_train, model.predict(X_train_scaled), alpha=0.8)

# Application Kalman Adaptive
kalman = AdaptiveKalman(Q_FINAL,R_FINAL, BIAS_SHIFT, quantile=0.8)
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

"""
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
"""



# useful features to add:
# df["Time_Temp"] = df["Time"] * df["Temp"] ?
#df["Time_Cooling"] = df["Time"] * df["Cooling"]

def pinball_loss(y, yhat, tau):
    return np.mean(np.maximum(tau * (y - yhat),
                              (tau - 1) * (y - yhat)))

# ============================================================================
# 5. SUBMISSION
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
true_test = test["Net_demand.1"].shift(-1)
true_test.loc[394] = preds_test[394]
print("True pinball loss:", pinball_loss(true_test, np.maximum(preds_test, 0), 0.8))
submission.to_csv("submission_v8_final.csv", index=False)