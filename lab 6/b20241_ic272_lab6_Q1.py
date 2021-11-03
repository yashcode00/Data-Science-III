# Name: Yash Sharma
# Registration Number: B20241
# Mobile Number: 8802131138

# importing needed modules
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime
from sklearn.metrics import mean_squared_error


# importing dataset
df = pd.read_csv('daily_covid_cases.csv',sep=',')
df['Date']=pd.to_datetime(df['Date'])

# plotting line plot of original dataframe
plt.plot(df['Date'],df['new_cases'])
matplotlib.dates.DateFormatter('%d')
plt.xlabel("Month-Year")
plt.xticks(rotation=60)
plt.ylabel("New confirmed cases")
plt.title("Lineplot--Q1a.")
plt.tight_layout()
plt.show()

########################################################################################################################
# Part-Q1 (b) ---------------------------------------------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>
########################################################################################################################

# making lag series by 1 day from original series
df_lag_by1=df[:-1]
# finding pearson correlation (autocorrelation)
corr_lagby1=round(np.corrcoef(df_lag_by1['new_cases'],df['new_cases'][1:])[0,1],5)
print("1.(b) The Pearson correlation (autocorrelation) coefficient between the generated one-day lag time sequence "
      "and the given time sequence is: ",corr_lagby1)

########################################################################################################################
# Part-Q1 (c) ---------------------------------------------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>
########################################################################################################################

# generating scatter plot between original data and one-day lagged data
plt.scatter(df_lag_by1['new_cases'],df['new_cases'][1:],alpha=0.85)
plt.xlabel("One day lagged Sequence")
plt.ylabel("Original Sequence")
plt.title("Scatter plot b/w original and 1-day lagged time series")
plt.tight_layout()
plt.show()

########################################################################################################################
# Part-Q1 (d) ---------------------------------------------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>
########################################################################################################################

print('Q1.(d)')
lags=np.arange(2,7,1)
# array to store correlation values
lagged_corr=[]
original_data=df['new_cases']
for p in lags:
    df_lagbyp=df['new_cases'][:-p]
    corr_lagbyp=round(np.corrcoef(df_lagbyp,original_data[p:])[0,1],5)
    print("The Pearson correlation (autocorrelation) coefficient with ",p,"day lag is: ",corr_lagbyp)
    # appending values in array
    lagged_corr.append(corr_lagbyp)

# now plotting the graph b/w correlation and respective lag in sequence
plt.plot(lags,lagged_corr, linestyle='dotted',marker='o')
plt.xlabel("(p) Lag in time sequence (in days)")
plt.ylabel("Autocorrelation value")
plt.title("Line plot b/w obtained correlation coefficients and lagged values")
plt.show()

########################################################################################################################
# Part-Q1 (e) ---------------------------------------------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>
########################################################################################################################

sm.graphics.tsa.plot_acf(original_data)
plt.tight_layout()
plt.show()