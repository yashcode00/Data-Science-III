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

# using inbuilt  function isna() and sum() below to find total missing values
df_miss=df.isna().sum()
# making x ticks for x-axis
x_ticks=np.arange(0,85,4)
df_miss.plot.bar(label='Number of missing Values')
plt.yticks(x_ticks)
plt.xlabel('Attributes')
plt.ylabel('Count of missing values')
plt.title("Barplot for Count of missing values in all Attributes")
plt.legend()
plt.tight_layout()
plt.show()
