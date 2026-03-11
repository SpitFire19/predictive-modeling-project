import pandas as pd
import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from pygam import LinearGAM, s, f, te
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import TimeSeriesSplit
from dateutil.relativedelta import relativedelta
from scipy.stats import pearsonr
from pathlib import Path
import sys

# Get the grandparent directory
parent_root = Path(__file__).resolve().parents[1]
# Add it to sys.path
sys.path.append(str(parent_root))

from data_utils import RBFFeatureEngineerExpert

# Ignore some warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-darkgrid')
QUANTILE = 0.8


train = pd.read_csv("Data/net-load-forecasting-during-soberty-period/train.csv")
test = pd.read_csv("Data/net-load-forecasting-during-soberty-period/test.csv")

exp = RBFFeatureEngineerExpert().fit(train)
df = exp.transform(train)

corr = df[['Time', 'Week', 'Wind','Temp','Temp_s95','Time_Cooling', 'Temp_s99',
           'Nebulosity','Solar_power','Wind_power',
           'Load','Net_demand', 'Year_Continuous']].corr()


col = 'Temp'
print(corr[col].sort_values(ascending=False))
start = pd.Timestamp("2013-03-02")
end = pd.Timestamp("2022-09-01")
delta = relativedelta(end, start)
months = delta.years * 12 + delta.months
df['period'] = pd.qcut(df['Time'], months)
df_plot = df.copy()
corrs = []
for p in df['period'].unique():
    subset = df[df['period'] == p]
    val = subset[['Time_Cooling','Net_demand']].corr().iloc[0, 1]
    corrs.append(val)
    print(val)

print(f'Mean correlation: {np.mean(corrs)}')

plt.figure(figsize=(10,7))

for p in df_plot["period"].unique():
    subset = df_plot[df_plot["period"] == p]

    plt.scatter(subset["Wind"], subset["Net_demand"], 
                alpha=0.3, label=str(p))


for p in df_plot["period"].unique():
    subset = df_plot[df_plot["period"] == p]

    slope = np.polyfit(subset["Wind"], subset["Net_demand"], 1)[0]

    print(p, "slope:", slope)

columns = ['Sobriety_Trend', 'Heating_Std', 'Cooling_Std', 'Wind_Chill', 'Temp_vs_Weekly_Max', 
           'Week', 'Time_Cooling', 'Is_Peak_Month', 'Work_activity', 'Year_Continuous']
for colname in columns:
    r, p = pearsonr(df[colname], df["Net_demand"])
    print(f'{colname} p-value: {r:.2f}, {p}')


plt.xlabel("Wind")
plt.ylabel("Net demand")
plt.title("Wind vs Net demand across time periods")
plt.legend()
# plt.show()


print(f'train Temp min: {train['Temp'].min() - 273.15}')
print(f'train Temp max: {train['Temp'].max() - 273.15}')
