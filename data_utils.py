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
    def __init__(self):
        self.train_date_min = None

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

        # Seasonality
        w = 2 * np.pi / 365.25
        for i in range(1, 7):
            df[f'cos{i}'] = np.cos(df['Time'] * w * i)
            df[f'sin{i}'] = np.sin(df['Time'] * w * i)
        
        temp_anchors_k = np.linspace(268.15, 308.15, 8).reshape(-1, 1)
        temp_values = df['Temp'].values.reshape(-1, 1)

        gamma_k = 0.018 # put 0.02 ?

        rbf_features = rbf_kernel(temp_values, temp_anchors_k, gamma=gamma_k)

        for i in range(rbf_features.shape[1]):
            df[f'Temp_K_RBF_{i}'] = rbf_features[:, i]
        
        DayOfWeek = df['Date'].dt.dayofweek
    
        # create periodic weekly features (with period = 7)
        df['week_sin'] = np.sin(2 * np.pi * DayOfWeek / 7)
        df['week_cos'] = np.cos(2 * np.pi * DayOfWeek / 7)

        #  we define 7 anchors for each day of the week
        angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        anchors = np.column_stack([np.sin(angles), np.cos(angles)])

        # periodic kernel (RBF on the sin and cos)
        periodic_week = rbf_kernel(df[['week_sin', 'week_cos']].values, anchors, gamma=0.25) # 0.25
        
        # Add features: Day_0, Day_1 ... Day_6
        for i in range(periodic_week.shape[1]):
            df[f'Weekly_Period_{i}'] = periodic_week[:, i] 
        
        all_years = np.arange(2013, 2023) # From start of train to end of test
        anchors = all_years.reshape(-1, 1) + 0.5

        # Convert current Date to continuous
        df['Year_Continuous'] = df['Date'].dt.year + (df['Date'].dt.dayofyear / 366)

        # Calculate Kernel
        year_rbf = rbf_kernel(df[['Year_Continuous']].values, anchors, gamma=0.005)

        # Add to DataFrame using the FIXED list of years
        for i, year in enumerate(all_years):
            df[f'RBF_Year_{year}'] = year_rbf[:, i]

        neb_values = df['Nebulosity'].values.reshape(-1, 1)
    
        # Define Anchors (Clear, Scattered, Overcast)
        anchors = np.array([[0.0], [0.5], [1.0]])
        
        # Calculate RBF
        neb_rbf = rbf_kernel(neb_values, anchors, gamma=0.002)
        
        # Map to DataFrame
        df['Neb_Clear'] = neb_rbf[:, 0]
        df['Neb_Partial'] = neb_rbf[:, 1]
        df['Neb_Overcast'] = neb_rbf[:, 2]

        wc_vals = df['Wind_Chill'].values.reshape(-1,1)
        wc_anchors = np.linspace(0, wc_vals.max(), 2).reshape(-1,1) # 
        wc_rbf = rbf_kernel(wc_vals, wc_anchors, gamma=1e-6)
        for i in range(wc_rbf.shape[1]):
            df[f'WindChill_RBF_{i}'] = wc_rbf[:, i]
        
        hdd_values = df['Heating_Std'].values.reshape(-1, 1)
        hdd_anchors = np.linspace(0, 25, 4).reshape(-1, 1) # 5 is a good value
        hdd_rbf = rbf_kernel(hdd_values, hdd_anchors, gamma=0.033) # 0.033
        # Adding the same for cooling_std won't have any positive effect
        for i in range(hdd_rbf.shape[1]):
            df[f'HDD_RBF_{i}'] = hdd_rbf[:, i]
        
        return df
    
class ExponentialSmoothing:
    def __init__(self, alpha=0.05, bias_shift=0.0):
        self.alpha = alpha
        self.bias = 0.0
        self.shift = bias_shift
        self.bias_hist = []
    def update(self, y_true, y_pred_base):
        err = y_true - (y_pred_base + self.shift + self.bias)
        self.bias = self.alpha * err + (1 - self.alpha) * self.bias
        # self.bias_hist.append(self.bias)

    def calibrate(self, y_hat, y):
        for y_pred, y_t in zip(y_hat, y):
            self.update(y_t, y_pred)

    def validate(self, y_hat, y):
        preds_val = []
        for y_pred, y_t in zip(y_hat, y):
            final_p = y_pred + self.shift + self.bias
            self.bias_hist.append(self.bias)
            self.update(y_t, y_pred)
            preds_val.append(final_p)
        return preds_val

# ============================================================================
# KALMAN FILTER
# ============================================================================

class AdaptiveKalman:
    def __init__(self,Q, R, B, quantile=0.8 ):
        self.quantile = quantile
        self.bias = 0.0
        self.P = 1000.0
        self.Q = Q
        self.R = R
        self.BIAS_SHIFT = B

    def update(self, y_true, y_pred_base):
        residual = y_true - (y_pred_base + self.BIAS_SHIFT + self.bias)
        self.P += self.Q
        K = self.P / (self.P + self.R)
        weight = self.quantile if residual > 0 else (1 - self.quantile)
        self.bias += K * residual * weight
        self.P *= (1 - K)
        return self.bias
    

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
        return df

class PrimaryFeatureEngineerExpert:
    def __init__(self):
        self.train_date_min = None

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
        return df


'''
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = []


for train_idx, val_idx in kf.split(train):
    X_train_cv, X_val_cv = train.iloc[train_idx], train.iloc[val_idx]
    y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

    train_set_cv = lgb.Dataset(X_train_cv, label=y_train_cv)
    val_set_cv   = lgb.Dataset(X_val_cv, label=y_val_cv)
    
    model = lgb.train(
    params,
    train_set_cv,
    num_boost_round=5000,
    valid_sets=[train_set_cv, val_set_cv],
    callbacks=[
        lgb.early_stopping(stopping_rounds=3),
    ]
    )
    preds = model.predict(X_val_cv)
    diff = y_val_cv - preds
    fold_loss = np.mean(np.maximum(tau * diff, (tau - 1) * diff))
    cv_scores.append(fold_loss)

print('Fold losses:', cv_scores)
print('Mean CV loss:', np.mean(cv_scores))
'''

