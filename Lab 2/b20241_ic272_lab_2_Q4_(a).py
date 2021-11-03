# Name: Yash Sharma
# Registration Number: B20241
# Mobile Number: 8802131138

# importing needed modules
import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as plt

# reading csv file" landslide_data3_after_Q2" from previous problem 2 with pandas
df = pd.read_csv("landslide_data3_after_Q2.csv", sep=',')
print(df.info())

# reading csv file" landslide_data3_original" from previous problem 2 with pandas
df_original_1 = pd.read_csv("landslide_data3_original.csv", sep=',')


# rename index column in Q2 csv
df.rename(columns={'Unnamed: 0': 'previous index'}, inplace=True)

# Using original index in above file (column-Unamed) to remove rows from original csv data so to calculate accurate rmse
duplicates = df_original_1.index.isin(df['previous index'])
df_original = df_original_1[duplicates]

bool=np.where(df['dates'] == df['dates'], True, False)
print(bool)

# removing  extra inbuilt column 'previous index'
df.drop('previous index', inplace=True, axis=1)

df_columns = df.columns.to_list()
df_columns.remove('stationid')
df_columns.remove('dates')

# using fillna() to fill misiing values with mean of respective attributes

mean_fill = df[df_columns].mean().to_list()
dict_fillna = {}
for i in range(len(mean_fill)):
    dict_fillna[df_columns[i]] = round(mean_fill[i], 3)
# fillin  na values with respective mean values
df.fillna(value=dict_fillna, inplace=True)


def tendencies_founder(arr, arr_original, col):
    # convert data of given attribute to numpy array
    arr = arr.to_numpy()
    arr_original = arr_original.to_numpy()
    # main calculation of all the required central tendencies
    print("The mean of  attribute: " + col + " for our dataframe and original dataframe are respectively :",
          round(np.mean(arr), 3), "and", round(np.mean(arr_original), 3))
    print("The mode of attribute: " + col + " for our dataframe and original dataframe are respectively :",
          st.mode(arr), "and", st.mode(arr_original))
    print("The median of attribute: " + col + " for our dataframe and original dataframe are respectively :",
          round(np.median(arr), 3), "and", round(np.median(arr_original), 3))
    print(
        "The standard deviation of attribute: " + col + " for our dataframe and original dataframe are respectively :",
        round(np.std(arr), 3), "and", round(np.std(arr_original), 3))
    print()


# extracting columns from dataframe
df_columns = df.columns.to_list()
print(
    "As attributes: 'stationid' and 'dates' are unique ids or are primary keys thus mean, mode, median are note defined for them.\n")
df_columns.remove('stationid')
df_columns.remove('dates')
i = 1
for col in df_columns:
    print(str(i) + "." + " For attribute: " + col)
    tendencies_founder(df[col], df_original[col], col)
    i += 1

#############################################################################################################################################################
# Part (ii)
#############################################################################################################################################################


# defining a RMSE calculating function
def rmse(targets, predictions):
    #return np.sqrt(np.mean(np.square(targets - predictions)))
    return np.sqrt(np.mean(np.square(np.subtract(np.array(targets) , np.array(predictions)))))

print("\nPart (B) : RMSE calculation of all the attributes: \n")
dict_rmse = {}
for cols in df_columns:
    print("The RMSE for attribute " + cols + " is :", round(rmse(df[cols], df_original[cols]), 4))
    dict_rmse[cols] = round(rmse(df[cols], df_original[cols]), 4)

y = dict_rmse.values()

# plotting rmse values of respective attributes using matplot
plt.bar(df_columns, y, label='RMSE Values')
plt.xlabel("Attributes")
plt.xticks(rotation='65')
plt.ylabel('RMSE Value')
plt.legend()
plt.tight_layout()
plt.show()


