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

# finding correct value of lag
l_bound=2/np.sqrt(len(train))
p=1
# print(l_bound)
original_data=train['new_cases']
while(p<=len(train)):
    df_lagbyp=train['new_cases'][:-p]
    corr_lagbyp=round(np.corrcoef(df_lagbyp,original_data[p:])[0,1],5)
    # print(p,",,,,,",corr_lagbyp)
    if abs(corr_lagbyp)<=l_bound:
        break
    p+=1

# Code snippet to train AR model and predict using the coefficients.
window = p-1  # The lag=p
model = AR(train['new_cases'], lags=window).fit()  # fit/train the model
coef = model.params  # Get the coefficients of AR model

# using these coefficients walk forward over time steps in test, one step each time
history = (train['new_cases'][len(train) - window:]).tolist()
history = [history[i] for i in range(len(history))]
predictions = list()  # List to hold the predictions, 1 step at a time
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length - window, length)]
    yhat = coef[0]  # Initialize to w0
    for d in range(window):
        yhat += coef[d + 1] * lag[window - d - 1]  # Add other values
    obs = test['new_cases'][t]
    predictions.append(yhat)  # Append predictions to compute RMSE later
    history.append(obs)  # Append actual test value to history, to be used in next step.


print("\nQ4 ")
print("The best lag value comes to be :",p-1)
print("The RMSE (%) is: ", round(rmsep(predictions, test['new_cases']), 3), "\nAnd MAPE (%) is: ",
      round(mean_absolute_percentage_error(predictions, test['new_cases'])*100, 3))

