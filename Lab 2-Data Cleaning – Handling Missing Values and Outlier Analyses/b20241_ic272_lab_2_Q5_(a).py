# Name: Yash Sharma
# Registration Number: B20241
# Mobile Number: 8802131138

# importing needed modules
import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as plt


# defining a function below to find every quartile value of a box when a boxplot is supplied as an argument
def get_box_plot_data(labels, bp):
    rows_list = []

    for i in range(len(labels)):
        dict1 = {}
        dict1['label'] = labels[i]
        dict1['lower_whisker'] = bp['whiskers'][i * 2].get_ydata()[1]
        dict1['lower_quartile'] = bp['boxes'][i].get_ydata()[1]
        dict1['median'] = bp['medians'][i].get_ydata()[1]
        dict1['upper_quartile'] = bp['boxes'][i].get_ydata()[2]
        dict1['upper_whisker'] = bp['whiskers'][(i * 2) + 1].get_ydata()[1]
        rows_list.append(dict1)

    # returning all box plot values as dataframe
    return pd.DataFrame(rows_list)


# reading csv file" landslide_data3_after_Q2" from previous problem 2 with pandas
df = pd.read_csv("landslide_data3_interpolated.csv", sep=',')

# printing boxplot for temperature
plt.boxplot(df['temperature'], labels=['temperature'])
plt.title("Boxplot for 'Temperature'( in Celsius) ")
plt.xlabel("Experiment")
plt.ylabel('Values (in celsius)')
plt.show()

# printing boxplot for rain
plt.boxplot(df['rain'], labels=['rain'])
plt.title("Boxplot for 'rain'( in ml)")
plt.xlabel("Experiment")
plt.ylabel('Values (in ml)')
plt.show()

data = [df['temperature'], df['rain']]
label = ['temperature', 'rain']

bp = plt.boxplot(data, labels=label)

# storing boxlot information values
quartile_data = get_box_plot_data(label, bp)


# defining function to find and list all outliers for an attribute
def outliers(data, df, col):
    Q1 = df.loc[(df['label'] == col), 'lower_quartile'].to_list()[0]
    Q3 = df.loc[(df['label'] == col), 'upper_quartile'].to_list()[0]
    IQR = Q3 - Q1

    print("The list of outliers for attribute " + col + " is: ")
    outliers = []
    for val in data:
        if (val > (Q3 + (1.5 * IQR))) or (val < (Q1 - (1.5 * IQR))):
            outliers.append(val)
    outliers = pd.Series(outliers)
    print(outliers)


# finally calling outliers function for 2 given attributes and printing their respecting outliers
cols = ['temperature', 'rain']
for col in cols:
    outliers(df[col], quartile_data, col)
    print("\n\n")
