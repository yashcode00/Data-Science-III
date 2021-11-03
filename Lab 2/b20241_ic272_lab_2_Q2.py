# Name: Yash Sharma
# Registration Number: B20241
# Mobile Number: 8802131138

# importing needed modules
import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as plt


# reading csv file" landslide_data3_miss" with pandas
df=pd.read_csv("landslide_data3_miss.csv",sep=',')

#############################################################################################################################################################
#Part A
#############################################################################################################################################################

# calculating initial target values
initial_target_values=df['stationid'].isnull().sum() + df['stationid'].notnull().sum()
print("Part (a)")
print("Total Number of values Null or not-Null in the target attribute 'stationid' initially: ",initial_target_values)
# dropping missing values in target col-"stationid"
df.dropna(subset=['stationid'],inplace=True)
# calculating final target values
final_target_values=df['stationid'].isnull().sum() + df['stationid'].notnull().sum()

print("Total Number of values after droping null values in the target attribute 'stationid' finally: ",final_target_values)
print("Thus, number of deleted tuples are (or initially number of null values): ",initial_target_values-final_target_values)


#############################################################################################################################################################
#Part B
#############################################################################################################################################################

initial_tuples=df.shape[0]
# calculating total number of columns
attributes=len(df.columns)
# putting thresh argument in dropna its required value
thresh=attributes-(attributes/3)
# dropping the tuples with at least 3 missing attributes
df.dropna(thresh=thresh+1,inplace=True,axis=0)
print("\n\nPart (b)\nNumber of tuples or rows after Deleting (dropping) the tuples (rows) having equal to or more than one third of attributes \
with missing values are: ",initial_tuples-df.shape[0])
# saving dataset fro use in problem number 3 in csv format with name: "landslide_data3_after_Q2.csv"
df.to_csv("landslide_data3_after_Q2.csv")