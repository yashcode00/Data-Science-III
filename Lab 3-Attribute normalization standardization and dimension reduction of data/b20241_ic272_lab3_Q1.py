# Name: Yash Sharma
# Registration Number: B20241
# Mobile Number: 8802131138

# importing needed modules
import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 1000, 'display.max_columns', 1000)

# importing the dataframe
df = pd.read_csv("pima-indians-diabetes.csv", sep=",")
cols = df.columns[:-1]


# function to find top and bottom whisker
def find_boundary(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    top = Q3 + (1.5 * IQR)
    bottom = Q1 - (1.5 * IQR)
    return top, bottom


# function to replace the outliers with median
def replace_outliers(df, col):
    top, bottom = find_boundary(df, col)
    outliers = (df[col] < bottom) | (df[col] > top)
    # finding median of non-outliers
    arr = df[col].to_list()
    med = []
    for val in arr:
        if (val > bottom) and val < top:
            med.append(val)
    # finding median of values that are not outliers
    median = np.median(med)
    df[col] = np.where(outliers, median, df[col])


# making boxplot for finding outliers in each attribute
bp = plt.boxplot(df[cols])
plt.xticks(range(1, len(cols) + 1), cols)
plt.title("Original boxplots for all attributes before replacing outliers")
plt.ylabel("Values")
plt.show()

# loop to call function to replace outliers
for col in cols:
    replace_outliers(df, col)

# making boxplot after filling outliers
bp_2 = plt.boxplot(df[cols])
plt.xticks(range(1, len(cols) + 1), cols)
plt.title("Boxplots for all attributes after replacing outliers")
plt.ylabel("Values")
plt.show()

##############################################################################################################################################################
# Part (a)
##############################################################################################################################################################\

print("\nPart (a)")
print("The min and max values before Min-Max scaling: ")
print(df.describe().loc[['min', 'max'], cols])


# function to do min-max scaling of the data
def min_max_scaling(new_min, new_max, arr):
    arr = arr.to_numpy()
    mn = min(arr)
    mx = max(arr)
    arr = (new_max - new_min) * ((arr - mn) / (mx - mn)) + (new_min)
    return arr


# creating copy of original df
df_min_max = df.copy()

# now performing the scaling operation
for col in cols:
    df_min_max[col] = min_max_scaling(5, 12, df[col])

print("\nThe min and max values after Min-Max scaling: ")
print(df_min_max.describe().loc[['min', 'max'], cols])

##############################################################################################################################################################
# Part (b)
##############################################################################################################################################################

print("\nPart(b)")
print("The mean and standard deviations of each attribute before Standarization: ")
data_mean_std = df.describe().loc[['mean', 'std'], cols]
print(data_mean_std)

# now performing the standardize operation
df[cols] = (df[cols] - df[cols].mean()) / df[cols].std()

print("The mean and standard deviations of each attribute after Standarization: ")
data_mean_std = df.describe().loc[['mean', 'std'], cols]
print(data_mean_std)

# saving current dataframe to csv to use in next problem
df.to_csv("pima-indians-diabetes-after-question-1.csv", index=False)
