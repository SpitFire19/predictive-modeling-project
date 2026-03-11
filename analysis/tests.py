import pandas as pd
import numpy as np
import warnings
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

QUANTILE = 0.8

train = pd.read_csv("Data/net-load-forecasting-during-soberty-period/train.csv")
test = pd.read_csv("Data/net-load-forecasting-during-soberty-period/test.csv")

exp = RBFFeatureEngineerExpert().fit(train)
df = exp.transform(train)

corr = df[['Time', 'Week', 'Wind','Temp','Temp_s95','Time_Cooling', 'Temp_s99',
           'Nebulosity','Solar_power','Wind_power',
           'Load','Net_demand', 'Year_Continuous']].corr()

col = 'Temp'
# Print the correlation coefficients of cols with every other column from corr
print(corr[col].sort_values(ascending=False))
start = pd.Timestamp("2013-03-02")
end = pd.Timestamp("2022-09-01")
delta = relativedelta(end, start)
# Total months in train data
months = delta.years * 12 + delta.months

# Separate the data by months
df['period'] = pd.qcut(df['Time'], months)
df_plot = df.copy()
corrs = []
for p in df['period'].unique():
    subset = df[df['period'] == p]
    val = subset[['Time_Cooling','Net_demand']].corr().iloc[0, 1]
    corrs.append(val)
    print(val)

# Mean correlation Corr(Time_Cooling, Net_demand) over all months in train data
print(f'Mean correlation: {np.mean(corrs)}')

# Calculate how Wind feature importance over each month in train data
for p in df_plot["period"].unique():
    subset = df_plot[df_plot["period"] == p]

    slope = np.polyfit(subset["Wind"], subset["Net_demand"], 1)[0]
    print(p, "slope:", slope)

# Calculate Pearson correlation and perform simple statistical test
columns = ['Sobriety_Trend', 'Heating_Std', 'Cooling_Std', 'Wind_Chill', 'Temp_vs_Weekly_Max', 
           'Week', 'Time_Cooling', 'Is_Peak_Month', 'Work_activity', 'Year_Continuous']
for colname in columns:
    r, p = pearsonr(df[colname], df["Net_demand"])
    print(f'{colname} p-value: {r:.2f}, {p}')

# Print max and min Temperature for RBF feature construction
print(f'train Temp min: {train['Temp'].min() - 273.15}')
print(f'train Temp max: {train['Temp'].max() - 273.15}')
