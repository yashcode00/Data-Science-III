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
from sklearn.preprocessing import PolynomialFeatures
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

########################################################################################################################
# Question 4(i) ---------------------------------------------------------------------------------->>>>>>>>>>>>>>>>>>>>
########################################################################################################################

degrees=[2,3,4,5]
print("\nPart B: Q4 (i).")
accur_train=[]
print(df_train.info())
for p in degrees:
    # making and fitting the model
    model1=PolynomialFeatures(p)
    arr_poly=model1.fit_transform(df_train[inputs_cols],df_train[output_col])
    regressor=LinearRegression().fit(arr_poly,df_train[output_col])
    preds=regressor.predict(model1.fit_transform(df_train[inputs_cols]))
    accu=mean_squared_error(preds,df_train[output_col],squared=False)
    print("The prediction accuracy using RMSE on train set when degree p is "+str(p)+" is: ",round(accu,3))
    accur_train.append(accu)

# now plotting bar graph for rmse and respective degree of polynomial
plt.bar(degrees,accur_train,alpha=0.85)
plt.xticks(degrees)
plt.xlabel("Degree of polynomial: p")
plt.ylabel("RMSE")
plt.title("Bar graph of RMSE (y-axis) vs different values of degree \nof the polynomial (x-axis) on Train Set")
plt.show()


########################################################################################################################
# Question 4(ii) ---------------------------------------------------------------------------------->>>>>>>>>>>>>>>>>>>>
########################################################################################################################

# now predicting and plotting same for the test set also
accur_test=[]
print("\nPart B: Q4 (ii).")
for p in degrees:
    # making and fitting the model
    model2=PolynomialFeatures(p)
    arr_poly=model2.fit_transform(df_train[inputs_cols],df_train[output_col])
    regressor=LinearRegression().fit(arr_poly,df_train[output_col])
    preds=regressor.predict(model2.fit_transform(df_test[inputs_cols]))
    accu=mean_squared_error(preds,df_test[output_col],squared=False)
    print("The prediction accuracy on test set using RMSE when degree p is "+str(p)+" is: ",round(accu,3))
    accur_test.append(accu)

# now plotting bar graph for rmse and respective degree of polynomial
plt.bar(degrees,accur_test,alpha=0.85)
plt.xticks(degrees)
plt.xlabel("Degree of polynomial: p")
plt.ylabel("RMSE")
plt.title("Bar graph of RMSE (y-axis) vs different values of degree \nof the polynomial (x-axis) on Test Set")
plt.show()

########################################################################################################################
# Question 4(iii) ---------------------------------------------------------------------------------->>>>>>>>>>>>>>>>>>>>
########################################################################################################################

print("\nPart B: Q3 (iii).")
best_degree=degrees[np.argmin(np.array(accur_test))]
# making model for best degree model
model_best=PolynomialFeatures(best_degree)
poly_best=model_best.fit_transform(df_train[inputs_cols],df_train[output_col])
regressor_best=LinearRegression().fit(poly_best,df_train[output_col])
# making best degree predictions on test set
preds_best_test=regressor_best.predict(model_best.fit_transform(df_test[inputs_cols]))
# plotting scatter plot for actual Rings in test set vs predicted rings using above model1
plt.scatter(df_test[output_col],preds_best_test)
plt.title("The scatter plot of actual Rings (x-axis) vs predicted Rings \n(y-axis) on the test data")
plt.xlabel("Actual Output: ")
plt.ylabel("Predicted Output: ")
plt.tight_layout()
plt.show()
