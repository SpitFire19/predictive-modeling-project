import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import QuantileRegressor
import lightgbm as lgb

def pinball_loss(y, yhat, tau):
    return np.mean(np.maximum(tau * (y - yhat),
                              (tau - 1) * (y - yhat)))

# Kalman filter
class AdaptiveKalman:
    def __init__(self, Q, R, B, quantile=0.8):
        """
        Q: Process Noise (How fast the bias can change)
        R: Measurement Noise (How much we trust the data)
        B: Initial static bias shift
        quantile: The target quantile for Pinball Loss (0.8)
        """
        self.quantile = quantile
        self.bias = 0.0
        self.P = 1.0     # Initial est. error covariance = identity
        self.Q = Q       # Process noise covariance
        self.R = R       # Measurement noise covariance
        self.BIAS_SHIFT = B
        self.bias_history = []
    def update(self, y_true, y_pred_base):
        """
        The 'Learning' Phase. 
        Updates the internal bias state based on the error observed at time t.
        """
        # predict next state covariance
        self.P += self.Q
        
        # calculate innovation (the error of the prediction we just made)
        # we use the bias that existed BEFORE this update
        residual = y_true - (y_pred_base + self.BIAS_SHIFT + self.bias)
        #  calculate Kalman gain
        K = self.P / (self.P + self.R)
        print(K)
        # asymmetric Weighting - this is our modification
        # to adjust better to pinball loss

        # residual > 0 -> under-prediction, weight is alpha
        # residual < 0 -> over-prediction, weight is 1-alpha
        weight = self.quantile if residual > 0 else (1 - self.quantile)
        
        # update state (Bias and Covariance)
        self.bias += K * residual * weight
        self.P *= (1 - K)
        
        self.bias_history.append(self.bias)
        return self.bias

    def calibrate(self, y_train, y_pred_train_base):
        """Warm up the filter using training data to settle the bias."""
        for y_t, p_t in zip(y_train, y_pred_train_base):
            self.update(y_t, p_t)
        return self.bias

# Default features (no processing except Date removal)
class DefaultFeatureEngineerExpert:
    def __init__(self):
        self.train_date_min = None

    def fit(self, df):
        self.train_date_min = pd.to_datetime(df['Date']).min()
        return self

    def transform(self, df_):
        df = df_.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df['Time'] = (df['Date'] - self.train_date_min).dt.days
        return df.drop(columns = 'Date')

# Add only basic features
class PrimaryFeatureEngineerExpert:
    def __init__(self, drop_cols, n_Fourier=None):
        self.drop_cols = drop_cols
        self.train_date_min = None
        self.n_Fourier = n_Fourier

    def fit(self, df):
        self.train_date_min = pd.to_datetime(df['Date']).min()
        return self

    def transform(self, df):
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df['Time'] = (df['Date'] - self.train_date_min).dt.days
        
        # Sobriété & Température
        inflection = (pd.to_datetime('2022-01-01') - self.train_date_min).days
        df['Sobriety_Trend'] = np.maximum(0,df['Time']-inflection)

        df['Heating_Std'] = np.maximum(288.15 - df['Temp_s95'], 0)
        df['Cooling_Std'] = np.maximum(df['Temp_s95'] - 295.15, 0) 
        
        df['Wind_Chill'] = df['Heating_Std'] * df['Wind_weighted']

        df['Week']=df['Date'].dt.isocalendar().week.astype(int)
        df['Temp_vs_Weekly_Max']=df['Temp']/df['Temp'].rolling(window=7,min_periods=1).max()
        
        df['Time_Cooling'] = df['Time'] * df['Wind_weighted']

        df['Is_Peak_Month'] = df['Date'].dt.month.isin([12, 1, 2]).astype(int)
        
        is_weekend = df['WeekDays'].isin([5, 6]).astype(int)
        if 'Usage' in df.columns:
            df['Work_activity'] = np.where((is_weekend == 0) & (df['Usage'] == 'Public'), 1, 0)
        else:
            df['Work_activity'] = np.where(is_weekend == 0, 1, 0)
        
        df['Year_Continuous'] = df['Date'].dt.year + (df['Date'].dt.dayofyear / 366)

        DayOfWeek = df['Date'].dt.dayofweek
    
        # create periodic weekly features (with period = 7)
        df['week_sin'] = np.sin(2 * np.pi * DayOfWeek / 7)
        df['week_cos'] = np.cos(2 * np.pi * DayOfWeek / 7)

        if self.n_Fourier != None:
            df = self.addPeriodicFourier(df, self.n_Fourier)

        df = df.drop(columns=[c for c in self.drop_cols if c in df.columns])
        return df
  
# Full features set (RBF, Fourier)
class RBFFeatureEngineerExpert:
    def __init__(self, gamma_temp = 0.01,
                gamma_neb = 0.001, gamma_wch = 1e-07,
                n_Fourier_features= 7):
        """
        Default gammas for RBF features were chosen by TimeSeriesSplit
        """
        self.train_date_min = None
        self.gamma_temp = gamma_temp
        self.gamma_neb = gamma_neb
        self.gamma_wch = gamma_wch
        self.n_Fourier_features = n_Fourier_features
        
    def fit(self, df):
        self.train_date_min = pd.to_datetime(df['Date']).min()
        return self

    def transform(self, df):
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df['Time'] = (df['Date'] - self.train_date_min).dt.days
        
        # Sobriété & Température
        inflection = (pd.to_datetime('2022-01-01') - self.train_date_min).days
        df['Sobriety_Trend'] = np.maximum(0,df['Time']-inflection)

        df['Heating_Std'] = np.maximum(288.15 - df['Temp_s95'], 0) # First cut at 15 C
        df['Cooling_Std'] = np.maximum(df['Temp_s95'] - 295.15, 0) # Second cut at 22 C
        
        df['Wind_Chill'] = df['Heating_Std'] * df['Wind_weighted']

        df['Week']=df['Date'].dt.isocalendar().week.astype(int)
        df['Temp_vs_Weekly_Max']=df['Temp']/df['Temp'].rolling(window=7,min_periods=1).max()
    
        df['Time_Cooling'] = df['Time'] * df['Wind_weighted']

        df['Is_Peak_Month'] = df['Date'].dt.month.isin([12, 1, 2]).astype(int)
        
        is_weekend = df['WeekDays'].isin([5, 6]).astype(int)
        if 'Usage' in df.columns:
            df['Work_activity'] = np.where((is_weekend == 0) & (df['Usage'] == 'Public'), 1, 0)
        else:
            df['Work_activity'] = np.where(is_weekend == 0, 1, 0)

        df['Year_Continuous'] = df['Date'].dt.year + (df['Date'].dt.dayofyear / 366)

        # Seasonality
        w = 2 * np.pi / 365.25
        n_fourier = self.n_Fourier_features
        for i in range(1, n_fourier + 1):
            df[f'cos{i}'] = np.cos(df['Time'] * w * i)
            df[f'sin{i}'] = np.sin(df['Time'] * w * i)

        temp_min, temp_max = 273.15 - 10, 273.15 + 35
        temp_anchors_k = np.linspace(temp_min, temp_max, 10).reshape(-1, 1)        
        temp_values = df['Temp'].values.reshape(-1, 1)
        rbf_features = rbf_kernel(temp_values, temp_anchors_k, gamma=self.gamma_temp)
        for i in range(rbf_features.shape[1]):
            df[f'Temp_K_RBF_{i}'] = rbf_features[:, i]
        
        DayOfWeek = df['Date'].dt.dayofweek

        # create periodic weekly features (with period = 7)
        df['week_sin'] = np.sin(2 * np.pi * DayOfWeek / 7)
        df['week_cos'] = np.cos(2 * np.pi * DayOfWeek / 7)
        
        neb_values = df['Nebulosity'].values.reshape(-1, 1)
    
        # Define Anchors (Clear, Scattered, Overcast)
        anchors = np.array([[0.0], [0.5], [1.0]])
        
        # Calculate RBF
        neb_rbf = rbf_kernel(neb_values, anchors, gamma=self.gamma_neb)
        
        # Map to DataFrame
        df['Neb_Clear'] = neb_rbf[:, 0]
        df['Neb_Partial'] = neb_rbf[:, 1]
        df['Neb_Overcast'] = neb_rbf[:, 2]

        wch_vals = df['Wind_Chill'].values.reshape(-1,1)
        wch_anchors = np.linspace(0, wch_vals.max(), 2).reshape(-1,1)
        wch_rbf = rbf_kernel(wch_vals, wch_anchors, gamma=self.gamma_wch)
        for i in range(wch_rbf.shape[1]):
            df[f'WindChill_RBF_{i}'] = wch_rbf[:, i]
        return df.drop(columns = 'Date')

def k_CV(X, y, alpha, n_splits):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, validation_idx in tscv.split(X):
        X_train, X_valn = X[train_idx], X[validation_idx]
        y_train, y_valn = y[train_idx], y[validation_idx]
        model = QuantileRegressor(quantile=0.8, alpha=alpha, solver='highs')
        model.fit(X_train, y_train)
        preds = model.predict(X_valn)

        score = pinball_loss(y_valn, preds, 0.8)
        scores.append(score)

    # print("CV scores:", scores)
    print(f"Mean CV score for alpha =  {alpha}:", np.mean(scores))

def k_CV_corr(X, y, model, params_corr_model, n_splits):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, validation_idx in tscv.split(X):
        X_train, X_valn = X[train_idx], X[validation_idx]
        y_train, y_valn = y[train_idx], y[validation_idx]
        model.fit(X_train, y_train)
        res_train = y_train - model.predict(X_train)
        res_val = y_valn - model.predict(X_valn)
        train_set = lgb.Dataset(X_train, label=res_train)
        val_set   = lgb.Dataset(X_valn, label=res_val)
        corr_model = lgb.train(
            params_corr_model,
            train_set,
            num_boost_round=5000,
            valid_sets=[train_set, val_set],
            callbacks=[
            lgb.early_stopping(stopping_rounds=100), # early stopping
            lgb.log_evaluation(period=0)            # logging

        ])
        preds = model.predict(X_valn) + corr_model.predict(X_valn)
        score = pinball_loss(y_valn, preds, 0.8)
        scores.append(score)

    # print("CV scores:", scores)
    print(f"Mean CV score for linear model + GBM correction:", np.mean(scores))
