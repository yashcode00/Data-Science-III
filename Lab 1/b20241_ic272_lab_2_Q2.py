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
#Part A
##############################################################################################################################################################

# extracting attribute "age" data
df_age=df['Age']
# extracting columns from dataframe and removing "age" and "class"
df_columns=df.columns
df_columns=df_columns.to_list()
df_columns.remove('class')
df_columns.remove('Age')


for val in df_columns:
    # extracting one by one data from each attribute fro further processing
    y=df[val]
    # forming a dictionary to place correct units at the end
    dict = {"pregs": "n", "plas": "n", "pres": " in mm Hg", "skin": "in mm", "test": "in mu U/mL",
            "BMI": "in kg/m^2",
            "pedi": "n", "Age": " in years"}
    # checking attributes and putting unit at end if needed
    if dict[val]!="n":
        val=val+"("+dict[val]+")"

    # plotting the scatter plot between "Age" and other attribute
    plt.scatter(df_age,y,alpha=0.8)
    plt.title("Scatter Plot of Age (in years) vs "+val)
    plt.xlabel("Age (in years)")
    plt.ylabel(val)
    plt.tight_layout()
    plt.show()
##############################################################################################################################################################
#Part B
##############################################################################################################################################################

# extracting attribute "bmi" data
df_bmi=df['BMI']
# extracting columns from dataframe and removing "bmi" and "class"
df_columns=df.columns
df_columns=df_columns.to_list()
df_columns.remove('class')
df_columns.remove('BMI')

for val in df_columns:
    # extracting one by one data from each attribute fro further processing
    y=df[val]
    # forming a dictionary to place correct units at the end
    dict = {"pregs": "n", "plas": "n", "pres": " in mm Hg", "skin": "in mm", "test": "in mu U/mL",
            "BMI": "in kg/m^2",
            "pedi": "n", "Age": " in years"}
    # checking attributes and putting unit at end if needed
    if dict[val] != "n":
        val = val + "(" + dict[val] + ")"
    # plotting the scatter plot between "BMI" and other attribute
    plt.scatter(df_bmi,y,alpha=0.8)
    plt.title("Scatter Plot of BMI(in kg/m^2) vs "+val)
    plt.xlabel("BMI(in kg/m^2)")
    plt.ylabel(val)
    plt.tight_layout()
    plt.show()
