# Name: Yash Sharma
# Registration Number: B20241
# Mobile Number: 8802131138

# importing needed modules
import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as plt


# reading csv file" landslide_data3_after_Q2" from previous problem 2 with pandas
df=pd.read_csv("landslide_data3_after_Q2.csv",sep=',')

# saving array storing missing values count in each attribute
list_missing=df.isna().sum().to_list()[1:]
# combining this data with attribute name into dictionary
dict={"Attributes":df.columns.to_list()[1:],"Missing Values Count":list_missing}
dict=pd.DataFrame(dict)
# printing dictionary
print(dict)

# printing total number of missing values in the file
print("\nTotal number of missing values in file are: ",sum(list_missing))


