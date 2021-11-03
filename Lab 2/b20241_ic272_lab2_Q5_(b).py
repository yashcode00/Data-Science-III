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
        dict1['lower_whisker'] = bp['whiskers'][i*2].get_ydata()[1]
        dict1['lower_quartile'] = bp['boxes'][i].get_ydata()[1]
        dict1['median'] = bp['medians'][i].get_ydata()[1]
        dict1['upper_quartile'] = bp['boxes'][i].get_ydata()[2]
        dict1['upper_whisker'] = bp['whiskers'][(i*2)+1].get_ydata()[1]
        rows_list.append(dict1)

    return pd.DataFrame(rows_list)

# reading csv file" landslide_data3_after_Q2" from previous problem 2 with pandas
df=pd.read_csv("landslide_data3_interpolated.csv",sep=',')

# extracting column names and their respective data
data=[df['temperature'],df['rain']]
label=['temperature','rain']

# plotting boxplot combined for both attributes before replacing outliers
bp=plt.boxplot(data,labels=label)
quartile_data=get_box_plot_data(label,bp)
plt.title("Original boxplots for both attributes before replacing outliers")
plt.ylabel("Values")
plt.show()

# copying quartile boxplot information into new dataframe
data=quartile_data.copy()

# loop for replacing the outliers in a data with respective median
for col in ['temperature','rain']:
    Q1 = data.loc[(data['label'] == col),'lower_quartile'].to_list()[0]
    Q3 = data.loc[(data['label'] == col), 'upper_quartile'].to_list()[0]
    IQR = Q3 - Q1
    # boolean detection of outliers
    outliers = (df[col] > (Q3 + (1.5 * IQR))) | (df[col] < (Q1 - (1.5 * IQR)))
    # extracting median form boxplot
    median = data.loc[(data['label'] == col), 'median'].to_list()[0]
    df[outliers] = np.nan
    # replacing outliers with median
    df[col].fillna(median, inplace=True)
    # plotting the new boxlot
    plt.boxplot(df[col], labels=[col])
    plt.ylabel('values')
    plt.title("Boxplot for " + col + " after filling outliers")
    plt.show()


