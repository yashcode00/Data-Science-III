# Name: Yash Sharma
# Registration Number: B20241
# Mobile Number: 8802131138

# importing needed modules
import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as plt

# reading csv file" pima-indians-diabetes" with pandas
df=pd.read_csv("pima-indians-diabetes.csv",sep=',')

# function to find mean, median, mode, max, min, std

def tendencies_founder(arr,col):
    # convert data of given attribute to numpy array
    arr=arr.to_numpy()
    # forming a dictionary to place correct units at the end
    dict = {"pregs": "n", "plas": "n", "pres": "mm Hg", "skin": "mm", "test": "mu U/mL", "BMI": "weight in kg/(height in m)^2",
            "pedi": "n", "Age": "years"}
    # checking attributes and putting unit at end if needed
    if(dict[col]!="n"):
        col=col+"("+dict[col]+")"

    # main calculation
    print("The mean of  attribute: "+col+" is:",round(np.mean(arr),3))
    print("The mode of attribute: " + col + " is:", st.mode(arr))
    print("The median of attribute: " + col + " is:", round(np.median(arr),3))
    print("The minimum of attribute: " + col + " is:", min(arr))
    print("The maximum of attribute: " + col + " is:",max(arr))
    print("The standard deviation of attribute: " + col + " is:", round(np.std(arr),3))
    print()

# extracting columns from dataframe
df_columns=df.columns

# sending all attributes one by one to function tendencies_founder except attribute class
for val in df_columns:
    if(val!='class'):
        tendencies_founder(df[val],val)















