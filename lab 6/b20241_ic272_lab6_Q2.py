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

# importing dataset
df = pd.read_csv("daily_covid_cases.csv")
df['Date'] = pd.to_datetime(df['Date'])

# splitting data into train set (initial 65%) and test set (rest 35%)
train = df[:int(len(df) * 0.65)]
test = df[int(len(df) * 0.65):]

# plotting the  train set
plt.plot(train['Date'],train['new_cases'],label='Training Set')
matplotlib.dates.DateFormatter('%d')
plt.xlabel("Month-Year")
plt.xticks(rotation=60)
plt.ylabel("New confirmed cases")
plt.title("Lineplot--Q2a. Training Set")
plt.tight_layout()
plt.legend()
plt.show()

# plotting the test set
plt.plot(test['Date'],test['new_cases'],label="Test Set")
matplotlib.dates.DateFormatter('%d')
plt.xlabel("Month-Year")
plt.xticks(rotation=60)
plt.ylabel("New confirmed cases")
plt.title("Lineplot--Q2a. Test Set")
plt.tight_layout()
plt.legend()
plt.show()

print("\nQ2 (a). ")
# Code snippet to train AR model and predict using the coefficients.
window = 5  # The lag=5
model = AR(train['new_cases'], lags=window).fit()  # fit/train the model
coef = model.params  # Get the coefficients of AR model
# printing the coefficients
for w in range(window + 1):
    print("The value of w" + str(w) + ": ", coef[w])

########################################################################################################################
# Part-Q2 (b) ---------------------------------------------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>
########################################################################################################################

# resetting test set indexing
test.reset_index(inplace=True)
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


# Q2 (b) (i)
# plotting scatter plot between actual and predicted cases frequency
plt.scatter(test['new_cases'],predictions,alpha=0.86)
plt.xlabel("Actual Cases")
plt.ylabel("Predicted Cases")
plt.title("Scatter plot between actual and predicted values")
plt.tight_layout()
plt.show()

# Q2 (b) (ii)
# plotting linear plot between actual and predicted cases frequency vs Date
plt.plot(test['Date'],test['new_cases'],alpha=0.86,label='Actual')
plt.plot(test['Date'],predictions,alpha=0.86,label='Predictions')
plt.legend()
matplotlib.dates.DateFormatter('%d')
plt.xlabel("Year-Month")
plt.ylabel("Cases")
plt.title("Line plot between actual and predicted values")
plt.tight_layout()
plt.show()

# Q2 (b) (ii)
# defining function to find rmse percentage
def rmsep(y_true, y_pred):
    return (mean_squared_error(y_true, y_pred,squared=False)/np.mean(y_true)) * 100

print("\nQ2 (b). (iii)")
print("The RMSE (%) is: ", round(rmsep(predictions, test['new_cases']), 3), "\nAnd MAPE (%) is: ",
      round(mean_absolute_percentage_error(predictions, test['new_cases'])*100, 3))

