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


# making x-ticks for bin size for attribute preg
x_ticks=np.arange(0,20,1)
plt.hist(df['pregs'],bins=x_ticks,rwidth=0.95,alpha=0.8)
plt.xlabel("Classes")
plt.xticks(x_ticks)
plt.ylabel("Frequency")
plt.title("Histogram plot for column (pregs)")
# plotting histogram with proper labels
plt.show()

# making x-ticks for bin size for attribute skin
x_ticks=np.arange(0,100,5)
plt.hist(df['skin'],bins=x_ticks,rwidth=0.95,alpha=0.8)
plt.xlabel("Classes")
plt.xticks(x_ticks)
plt.ylabel("Frequency")
plt.title("Histogram plot for column skin (in mm)")
# plotting histogram with proper labels
plt.show()