import pandas as pd
import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_pinball_loss
import matplotlib.pyplot as plt
import warnings
from data_utils import FeatureEngineerExpertReg, AdaptiveKalman
def pinball_loss(y, yhat, tau):
    return np.mean(np.maximum(tau * (y - yhat),
                              (tau - 1) * (y - yhat)))
pred_gbm = test = pd.read_csv("submission_gbm_kalman_final1.csv")
pred_viking = pd.read_csv("submission_viking.csv")

coef_quant_reg = 0.65

pred_final = (1 - coef_quant_reg) * pred_gbm["Net_demand"].to_numpy() + coef_quant_reg * pred_viking["Net_demand"].to_numpy()
test = pd.read_csv("Data/net-load-forecasting-during-soberty-period/test.csv")
true_test = test["Net_demand.1"].shift(-1)
true_test.loc[394] = pred_final[394]
print("True pinball loss:", pinball_loss(true_test, np.maximum(pred_final, 0), 0.8))

submission = pd.DataFrame({"Id": test["Id"], "Net_demand": np.maximum(pred_final, 0)})
submission.to_csv("submission_ensemble.csv", index=False)