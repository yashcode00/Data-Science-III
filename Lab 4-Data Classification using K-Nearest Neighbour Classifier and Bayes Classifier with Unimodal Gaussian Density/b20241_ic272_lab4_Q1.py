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

# importing dataset
df = pd.read_csv("SteelPlateFaults-2class.csv")
# remove 2 one-hot encoded column from input columns as they make cov matrix singular i.e det(cov)=0
inputs_cols = np.array(df.columns[:-1])
output_col = df.columns[-1]

df_class_0 = df[df['Class'] == 0].copy()
df_class_1 = df[df['Class'] == 1].copy()

# splitting the data as asked in the problem (Class-wise)
inputs_train_0, inputs_test_0, target_train_0, target_test_0 = train_test_split(df_class_0[inputs_cols],
                                                                                df_class_0[output_col], test_size=0.3,
                                                                                random_state=42, shuffle=True)
inputs_train_1, inputs_test_1, target_train_1, target_test_1 = train_test_split(df_class_1[inputs_cols],
                                                                                df_class_1[output_col], test_size=0.3,
                                                                                random_state=42, shuffle=True)

# concatenating splits done class-wise into one single train and test split
inputs_train = pd.concat([inputs_train_0, inputs_train_1], axis=0)
inputs_test = pd.concat([inputs_test_0, inputs_test_1], axis=0)
target_train = pd.concat([target_train_0, target_train_1], axis=0)
target_test = pd.concat([target_test_0, target_test_1], axis=0)

# saving test and train splits to csv
df_train = pd.concat([inputs_train, target_train], axis=1)
df_test = pd.concat([inputs_test, target_test], axis=1)
df_train.to_csv("SteelPlateFaults-train.csv", index=False)
df_test.to_csv("SteelPlateFaults-test.csv", index=False)


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
kn=[1, 3, 5]

for k in kn:
    predictions = knn(k, inputs_test, df_train)
    arr.append(accuracy_score(predictions,target_test))
    print("For k=" + str(k) + " Classification accuracy is: ", round(accuracy_score(predictions, target_test)*100,3)," %")
    print("The confusion matrix is : \n", confusion_matrix(target_test, predictions))

# printing max accuracy obtained and its respective k value
print("\nThe best accuracy is "+str(round(max(arr)*100,3))+" % for k="+str(kn[np.argmax(arr)]))
# storing best accuracies among all simulated
data_best=pd.DataFrame({'KNN':[round(max(arr)*100,3)]})
data_best.to_csv("Best_accuracies_Q4.csv",index=False)

