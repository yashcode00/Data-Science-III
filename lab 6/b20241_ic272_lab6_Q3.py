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
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.ar_model import AutoReg as AR


# defining function to find rmse percentage
def rmsep(y_true, y_pred):
    return (mean_squared_error(y_true, y_pred,squared=False)/np.mean(y_true)) * 100

# importing dataset
df = pd.read_csv("daily_covid_cases.csv")
df['Date'] = pd.to_datetime(df['Date'])

# splitting data into train set (initial 65%) and test set (rest 35%)
train = df[:int(len(df) * 0.65)]
test = df[int(len(df) * 0.65):]
# resetting test set indexing
test.reset_index(inplace=True)

# Code snippet to train AR model and predict using the coefficients.
lags=[1,5,10,15,25]
# making arrays to store values of rmse percentage and ,mape for respective lags
rmse_a=[]
mape_a=[]
for p in lags:
    model = AR(train['new_cases'], lags=p).fit()  # fit/train the model
    coef = model.params  # Get the coefficients of AR model

    # now making prediction step by step
    # using these coefficients walk forward over time steps in test, one step each time
    history = (train['new_cases'][len(train) - p:]).tolist()
    history = [history[i] for i in range(len(history))]
    predictions = list()  # List to hold the predictions, 1 step at a time
    for t in range(len(test)):
        length = len(history)
        lag = [history[i] for i in range(length - p, length)]
        yhat = coef[0]  # Initialize to w0
        for d in range(p):
            yhat += coef[d + 1] * lag[p - d - 1]  # Add other values
        obs = test['new_cases'][t]
        predictions.append(yhat)  # Append predictions to compute RMSE later
        history.append(obs)  # Append actual test value to history, to be used in next step.

    rmse_a.append(round(rmsep(predictions, test['new_cases']), 3))
    mape_a.append(round(mean_absolute_percentage_error(predictions, test['new_cases'])*100, 3))
    print("For lag of "+str(p)+" days, RMSE (%): ", round(rmsep(predictions, test['new_cases']), 3), "And MAPE (%): ",
          round(mean_absolute_percentage_error(predictions, test['new_cases'])*100, 3))


# now plotting bar graphs for both rmse and mape
plt.bar(lags,rmse_a,width =3)
plt.title("Bar chart: lags v RMSE(%) ")
plt.xlabel("Lag in time sequence (in days)")
plt.ylabel("RMSE (in %)")
plt.xticks(lags)
plt.tight_layout()
plt.show()

plt.bar(lags,mape_a,width = 3)
plt.title("Bar chart: lags v MAPE(%) ")
plt.xticks(lags)
plt.xlabel("Lag in time sequence (in days)")
plt.ylabel("MAPE (in %)")
plt.tight_layout()
plt.show()

