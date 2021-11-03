# Name: Yash Sharma
# Registration Number: B20241
# Mobile Number: 8802131138

# importing needed modules
import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# importing dataset
df = pd.read_csv("abalone.csv")
inputs_cols = np.array(df.columns[:-1])
output_col = df.columns[-1]

# splitting the data as asked in the problem
inputs_train, inputs_test, target_train, target_test = train_test_split(df[inputs_cols],
                                                                                df[output_col], test_size=0.3,
                                                                                random_state=42, shuffle=True)
# saving test and train splits to csv
df_train = pd.concat([inputs_train, target_train], axis=1)
df_test = pd.concat([inputs_test, target_test], axis=1)
df_train.to_csv("abalone-train.csv", index=False)
df_test.to_csv("abalone-test.csv", index=False)

# finding pearson coefficient of every input with target attribute
# slicing the array so as to eliminate the value 1 of rings with itself :)
cooef=np.array(df_train.corr()["Rings"])[:-1]
input1=[inputs_cols[np.argmax(cooef)]]
print("The attribute having highest pearson correlation coefficient with  target attribute: 'Rings' is: ",input1)

# making and fitting the model
model1=LinearRegression().fit(df_train[input1],df_train[output_col])

# its like y=w1*x+w0 so extracting slope and bias as follow
m1=model1.coef_[0]
c1=model1.intercept_

# plotting the scatter plot with line of best fit
plt.scatter(df_train[input1],df_train[output_col],alpha=0.30)
plt.plot(df_train[input1],m1*df_train[input1]+c1,color='r',linestyle='-.',linewidth=2)
plt.title("The best fit line on the training data")
plt.xlabel("Input: "+str(input1[0]))
plt.ylabel("Output: "+str(output_col))
plt.show()

print("The prediction accuracy using RMSE when only one highly correlated input is taken.")
print("\nRMSE Error in training set is: ",mean_squared_error(model1.predict(df_train[input1]),df_train[output_col],squared=False))
print("RMSE error on test set is: ",mean_squared_error(model1.predict(df_test[input1]),df_test[output_col],squared=False))

# plotting scatter plot for actual Rings in test set vs predicted rings using above model1
plt.scatter(df_test[output_col],model1.predict(df_test[input1]))
plt.title("The scatter plot of actual Rings (x-axis) vs predicted Rings \n(y-axis) on the test data")
plt.xlabel("Actual Output: ")
plt.ylabel("Predicted Output: ")
plt.show()

# making and fitting the model
model2=LinearRegression().fit(df_train[inputs_cols],df_train[output_col])

print("\nThe prediction accuracy using RMSE when all available inputs are taken.")
print("RMSE Error in training set is: ",mean_squared_error(model2.predict(df_train[inputs_cols]),df_train[output_col],squared=False))
print("RMSE error on test set is: ",mean_squared_error(model2.predict(df_test[inputs_cols]),df_test[output_col],squared=False))

# plotting scatter plot for actual Rings in test set vs predicted rings using above model1
plt.scatter(df_test[output_col],model2.predict(df_test[inputs_cols]))
plt.title("The scatter plot of actual Rings (x-axis) vs predicted Rings \n(y-axis) on the test data")
plt.xlabel("Actual Output: ")
plt.ylabel("Predicted Output: ")
plt.show()

