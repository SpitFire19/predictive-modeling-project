import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import rbf_kernel

def pinball_loss(y, yhat, tau):
    return np.mean(np.maximum(tau * (y - yhat),
                              (tau - 1) * (y - yhat)))

def rss(y, yhat):
    return np.sum([(y0 - yh) ** 2 for y0, yh in zip(y, yhat)])

def tss(y):
    return np.sum((y - np.mean(y) ** 2))

def import_raw_from_csv(dir_path: str) -> pd.DataFrame:
    df_train = pd.read_csv(f'{dir_path}/train.csv')
    df_test = pd.read_csv(f'{dir_path}/test.csv')
    return df_train, df_test

def date_raw_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df['Time'] = (
    pd.to_datetime(df['Date']) - pd.Timestamp('1970-01-01')
    ).dt.days
    return df
    # return df.drop(columns = 'Date')

def write_predictions_csv(
    pred: list[float],
    sample_pred_path: str,
    output_path: str
):
    submit = pd.read_csv(sample_pred_path)
    submit['Net_demand'] = pred
    submit.to_csv(output_path, index=False)

class FeatureEngineerExpertGBM:
    def __init__(self):
        self.train_date_min = None

    def fit(self, df):
        self.train_date_min = pd.to_datetime(df['Date']).min()
        return self

    def transform(self, df):
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['Time'] = (df['Date'] - self.train_date_min).dt.days
        df['WeekDays3'] = (
            pd.to_datetime(df['Date'])
            .dt.day_name()
            .replace({'Tuesday':'WorkDay','Wednesday':'WorkDay','Thursday':'WorkDay'})
            .astype('category')
        )
        le = LabelEncoder()
        df['WeekDays3'] = le.fit_transform(df['WeekDays3'])
        # Sobriété & Température
        inflection = (pd.to_datetime('2022-01-01') - self.train_date_min).days
        df['Sobriety_Trend'] = np.maximum(0, df['Time'] - inflection)
        df['is_holiday_season'] = df['Month'].isin([12, 1, 7, 8]).astype(int)
        df['Heating_Std'] = np.maximum(288.15 - df['Temp_s95'], 0)
        df['Thermal_Sobriety'] = df['Heating_Std'] * df['Sobriety_Trend'] / 1000
        df['Wind_Chill'] = df['Heating_Std'] * df['Wind_weighted']
        df['Heat_index'] = df['Temp'] + (0.55 * df['Nebulosity'])
        df['Hour_Month'] = df['Time'] * 100 + df['Month']
        df['Week']=df['Date'].dt.isocalendar().week.astype(int)
        df['Is_Peak_Month'] = df['Date'].dt.month.isin([12, 1, 2]).astype(int)
        # --- CORRECTION FINALE DU KEYERROR / ATTRIBUTEERROR ---
        df['is_weekend'] = df['WeekDays'].isin([5, 6]).astype(int)
        df['Cooling'] = df['Wind'] / df['Temp']
        df['Time_Cooling'] = df['Time'] * df['Cooling']
        is_winter = df['Month'].isin([12, 1, 2]).astype(int)
        ## Are these two really necessary ?
        df['is_shoulder'] = df['Month'].isin([3, 4, 5, 9, 10, 11]).astype(int)
        df['Winter_Temp'] = df['Temp'] * is_winter
        
        # Weather_Demand_Driver = HDD + CDD - Solar_power_Forecast
        df['Wind_eff'] = df['Wind_weighted'] / df['Wind']
        # Si Usage existe, on compare, sinon on met 1 par défaut pour les jours ouvrés
        if 'Usage' in df.columns:
            df['Work_activity'] = np.where((df['is_weekend'] == 0) & (df['Usage'] == 'Public'), 1, 0)
        else:
            df['Work_activity'] = np.where(df['is_weekend'] == 0, 1, 0)

        # Saisonnalité
        w = 2 * np.pi / 365.25
        for i in range(1, 7):
            df[f'cos{i}'] = np.cos(df['Time'] * w * i)
            df[f'sin{i}'] = np.sin(df['Time'] * w * i)
                
        # 1. Calculate Temp_Delta (The 'Shock' component)
        # We use a 1-step difference (e.g., Change since last hour)

    # 3. Optional: Non-Linear 'Heat Index' Momentum
    # This captures the 'discomfort' factor more accurately
    # (High humidity matters more at high temperatures)
        target_time = df[df['Year'] == 2022]['Time'].mean()
        target_temp  = 35.0 # Max heat

        # Calculate distance to that 'event'
        dist = (df['Temp'] - target_temp)**2
        sigma = 10.0 # Control the width of the 'influence'

        # The RBF feature: 1.0 when exactly at that point, 0.0 far away
        df['RBF_2022_Heat_Regime'] = np.exp(-dist / (2 * sigma**2))

        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].ffill().bfill().fillna(0)
        return df

class FeatureEngineerExpertReg:
    def __init__(self, gamma_temp = 0.01,
                gamma_neb = 0.001, gamma_wch = 1e-07,
                n_Fourier_features= 7):
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

from sklearn.metrics import mean_squared_error, mean_absolute_error

# ============================================================================
# KALMAN FILTER
# ============================================================================

class AdaptiveKalman:
    def __init__(self, Q, R, B, quantile=0.8):
        self.quantile = quantile
        self.bias = 0.0
        self.P = 10 * R
        self.Q = Q
        self.R = R
        self.BIAS_SHIFT = B
        self.bias_history = []

    def update(self, y_true, y_pred_base):
        # 1. Prediction Phase
        self.P += self.Q
        
        # 2. Innovation (Residual relative to the shifted base)
        residual = y_true - (y_pred_base + self.BIAS_SHIFT + self.bias)
        
        # 3. Kalman Gain
        K = self.P / (self.P + self.R)
        
        # 4. Asymmetric Weighting (Quantile Logic)
        # This forces the bias to favor one side of the error distribution
        weight = self.quantile if residual > 0 else (1 - self.quantile)
        
        # 5. Update State and Covariance
        self.bias += K * residual * weight
        self.P *= (1 - K)
        
        self.bias_history.append(self.bias)
        return self.bias

    def calibrate(self, y_train, y_pred_train_base):
        """Warm up the filter using training data."""
        for y_t, p_t in zip(y_train, y_pred_train_base):
            self.update(y_t, p_t)
        return self.bias

    def validate(self, y_val, y_pred_val_base):
        """
        Returns predictions for the validation set. 
        Note: Prediction for step 't' uses bias from 't-1'.
        """
        corrected_preds = []
        for y_t, p_t in zip(y_val, y_pred_val_base):
            # Predict using existing bias + shift
            corrected_preds.append(p_t + self.BIAS_SHIFT + self.bias)
            
            # Update the state for the next step
            # self.update(y_t, p_t)
            
        return np.array(corrected_preds)

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
