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

# using group by function to separate data based on class and then convert it to list
grouped_by_pref=df.groupby('class')['pregs'].agg(lambda x: x.tolist())

# separating the two datas
preg_class_0=grouped_by_pref[0]
preg_class_1=grouped_by_pref[1]

# making x_ticks for bin-size (pregs) with class as 0
x_ticks=np.arange(0,16,1)
# plotting histogram with proper labels
plt.hist(preg_class_0,bins=x_ticks,rwidth=0.95,alpha=0.8)
plt.xlabel("Classes")
plt.xticks(x_ticks)
plt.ylabel("Frequency")
plt.title("Histogram plot for (pregs) with class as 0")
plt.tight_layout()
plt.show()

# making x_ticks for bin-size (pregs) with class as 1
x_ticks=np.arange(0,20,1)
# plotting histogram with proper labels
plt.hist(preg_class_1,bins=x_ticks,rwidth=0.95,alpha=0.8)
plt.xlabel("Classes")
plt.xticks(x_ticks)
plt.ylabel("Frequency")
plt.title("Histogram plot for (pregs) with class as 1")
plt.tight_layout()
plt.show()