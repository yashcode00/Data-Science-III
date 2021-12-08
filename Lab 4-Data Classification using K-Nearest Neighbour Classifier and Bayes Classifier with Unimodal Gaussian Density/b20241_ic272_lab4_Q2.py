# Name: Yash Sharma
# Registration Number: B20241
# Mobile Number: 8802131138

# importing needed modules
import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

pd.set_option('display.max_rows', 1000,'display.max_columns', 1000)

# importing dataset both train and test splits from previous problems
df_train = pd.read_csv("SteelPlateFaults-train.csv")
df_test = pd.read_csv("SteelPlateFaults-test.csv")
inputs_cols = np.array(df_train.columns[:-1])
output_col = df_train.columns[-1]

# normalising dataset

# function to do min-max scaling of the data
def min_max_scaling(new_min, new_max, arr_train, arr_test):
    arr_train = arr_train.to_numpy()
    arr_test = arr_test.to_numpy()
    # checking if min or max value are present in test due to split
    if min(arr_train)>min(arr_test):
        mn=min(arr_test)
    else:
        mn = min(arr_train)
    if max(arr_train)<max(arr_test):
        mx=max(arr_test)
    else:
        mx = max(arr_train)

    arr_train = (new_max - new_min) * ((arr_train - mn) / (mx - mn)) + (new_min)
    arr_test=(new_max - new_min) * ((arr_test - mn) / (mx - mn)) + (new_min)
    return arr_train,arr_test

# now performing the scaling operation
for col in inputs_cols:
    a,b= min_max_scaling(0,1,df_train[col],df_test[col])
    df_train[col]=a
    df_test[col]=b


# saving normalised test and train splits to csv
df_train.to_csv("SteelPlateFaults-train-Normalised.csv", index=False)
df_test.to_csv("SteelPlateFaults-test-Normalised.csv", index=False)

# now we are ready to use are normalised data
inputs_test=df_test[inputs_cols]
target_test=df_test[output_col]


# defining function for k-nearest-neighbour
def knn(k, test, train):
    # making array to store final prediction on test set tuples
    pred_class = []
    test1=np.array(test[inputs_cols].copy())
    # iterate over test set rows
    for tuples in test1:
        # to store class with respective euclid error in two arrays
        train_class = []
        euclid_error = []
        train1=np.array(train.copy())
        # iterate over train set row to fin euclidean distance for each tuple
        for train_rows in train1:
            train_class.append(train_rows[-1])
            # transform to numpy
            train_rows = train_rows[:-1]
            # finding and storing Euclidean distance
            error = np.sqrt(np.sum(np.square(np.subtract(tuples, train_rows))))
            euclid_error.append(error)
        # making dataframe for these class and respective euclid error
        knn_df = pd.DataFrame({"Euclid Error": euclid_error, "Class": train_class})
        knn_df = knn_df.sort_values(by=["Euclid Error"])
        knn_df = knn_df.head(k)
        # append predicted class for that tuple via knn method
        pred_class.append(knn_df["Class"].value_counts().idxmax())
    return pred_class


arr=[]
kn=[1,3,5]
for k in kn:
    predictions=knn(k, inputs_test, df_train)
    arr.append(accuracy_score(predictions,target_test))
    print("For k="+str(k)+" Classification accuracy is: ",round(accuracy_score(predictions,target_test)*100,4)," %")
    print("The confusion matrix is : \n",confusion_matrix(target_test,predictions))

# printing max accuracy obtained and its respective k value
print("\nThe best accuracy is "+str(round(max(arr)*100,3))+"% for k="+str(kn[np.argmax(arr)]))
# importing dataframe for keeping best accuracies
data_best=pd.read_csv("Best_accuracies_Q4.csv")
data_best['KNN on normalised data']=round(max(arr)*100,3)
data_best.to_csv("Best_accuracies_Q4.csv",index=False)