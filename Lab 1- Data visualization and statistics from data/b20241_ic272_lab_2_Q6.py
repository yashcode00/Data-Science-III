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

# extracting columns from dataframe and removing attribute "class"
box_columns=df.columns.to_list()
box_columns.remove('class')
for val in box_columns:
    # making dictionary for y-label of the boxplot for all the attributes to be plotted
    dict2={"pregs": "Number of times pregnant","plas": "Plasma glucose concentration \n2 hours in an oral glucose tolerance test",
           "pres": "Diastolic blood pressure (mm Hg)","skin": "Triceps skin fold thickness (mm)",
            "test": "2-Hour serum insulin (mu U/mL)",
            "BMI": "Body mass index (weight in kg/(height in m)^2)",
            "pedi": "Diabetes pedigree function",
            "Age": "Age (years)"}

    # forming a dictionary to place correct units at the end
    dict = {"pregs": "n", "plas": "n", "pres": " in mm Hg", "skin": "in mm", "test": "in mu U/mL",
            "BMI": "in kg/m^2",
            "pedi": "n", "Age": " in years"}
    # checking and attaching unit at end of attribute name if needed
    if dict[val] != "n":
        val2 = val + "(" + dict[val] + ")"
    # else doing nothing
    else:
        val2=val
    # plotting boxplot with proper labels
    plt.boxplot(df[val])
    plt.title("Boxplot for "+val2)
    plt.ylabel(dict2[val])
    plt.xlabel("Experiment")
    plt.tight_layout()
    plt.show()