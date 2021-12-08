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


##############################################################################################################################################################
#Part A and Part B are combined as the attributes are taken in an array below
##############################################################################################################################################################

# forming array of attributes with which correlation has to be calculated
corr_base=['Age','BMI']
for base in corr_base:
    # extracting all columns from dataframe except class
    new_columns=df.columns.to_list()
    new_columns.remove('class')

    # forming a dictionary to place correct units at the end
    dict = {"pregs": "n", "plas": "n", "pres": " in mm Hg", "skin": "in mm", "test": "in mu U/mL",
            "BMI": "in kg/m^2",
            "pedi": "n", "Age": " in years"}

    # checking attribute if needed attaching unit to it
    if dict[base] != "n":
        basse = base + "(" + dict[base] + ")"
    else:
        basse=base
    # performing same procedure of attaching unit at end for the other attribute
    for cols in new_columns:
        if dict[cols] != "n":
            colls = cols + "(" + dict[cols] + ")"
        else:
            colls=cols
        # printing final answer rounded off to 3 decimal places
        print("Correlation coefficient between "+basse+" and "+colls+" is: ",round(np.corrcoef(df[base],df[cols])[0,1],3))
    print()
