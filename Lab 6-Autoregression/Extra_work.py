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
train = df
dict={'Date': pd.date_range(start='10/3/2021', end='1/31/2022'), 'Cases': np.zeros(121)}
test=pd.DataFrame(dict)
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
    test['Cases'].loc[t]=yhat
    history.append(yhat)  # Append actual test value to history, to be used in next step.


print("\nQ4 ")
print("The best lag value comes to be :",p-1)
print(test)
# plotting line plot of original dataframe
plt.plot(test['Date'],test['Cases'],label='future')
plt.plot(train['Date'],train['new_cases'],label='present and past')
matplotlib.dates.DateFormatter('%d')
plt.xlabel("Month-Year")
plt.xticks(rotation=60)
plt.legend()
plt.ylabel("Future confirmed cases")
plt.title("Future Predictions")
plt.tight_layout()
plt.show()

