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
import scipy

#pd.set_option('display.max_rows', 1000,'display.max_columns', 1000)

# importing training and test data
train=pd.read_csv("SteelPlateFaults-train.csv")
test=pd.read_csv("SteelPlateFaults-test.csv")

# remove 2 one-hot encoded column from input columns as they make cov matrix singular i.e det(cov)=0
inputs_cols = np.array(train.columns[:-1])
index1 = np.argwhere(inputs_cols == "TypeOfSteel_A300")
index2 = np.argwhere(inputs_cols == "TypeOfSteel_A400")
index3 = np.argwhere(inputs_cols == "X_Minimum")
index4 = np.argwhere(inputs_cols == "Y_Minimum")
inputs_cols = np.delete(inputs_cols, [index1, index2,index3,index4])
output_col = train.columns[-1]

# separating the classes in train set
train_class_0=train[train["Class"]==0]
train_class_1=train[train["Class"]==1]

# finding mean and covariance matrix for each class
mean_class_0=np.array(train_class_0[inputs_cols].mean())
mean_class_1=np.array(train_class_1[inputs_cols].mean())
cov_class_0=np.array(train_class_0[inputs_cols].cov())
cov_class_1=np.array(train_class_1[inputs_cols].cov())
# dataframe to store mean for all classes
df_mean=pd.DataFrame({"Attributes":inputs_cols,"Class0":np.around(mean_class_0,3),"Class1":np.around(mean_class_1,3)})

# printing mean and covariance matrix for each class
print("Mean for class 0 and class 1 attributes")
print(df_mean)
print()
cov0=train_class_0[inputs_cols].cov()
cov1=train_class_1[inputs_cols].cov()
# rounding off decimals
cov0=cov0.round(3)
cov1=cov1.round(3)
# print(np.unravel_index(np.where(np.array(cov0)==sorted(list((np.array(cov0).flatten())))[0]),(23,23)),sorted(list((np.array(cov0).flatten())))[0],np.array(cov0)[1,5])
# print(np.unravel_index(np.where(np.array(cov1)==sorted(list((np.array(cov1).flatten())))[0]),(23,23)),sorted(list((np.array(cov1).flatten())))[0],np.array(cov1)[1,5])

print("\nCovariance matrix for class 0: ")
print(cov0)
print("\nCovariance matrix for class 1: ")
print(cov1)



# Prior Probability ( P(Ci) )
P_C0=len(train[train["Class"]==0])/(len((train["Class"])))
P_C1=len(train[train["Class"]==1])/(len((train["Class"])))

# function to find likelihood for a single sample
def multivariate_probb(x,m,c):
    mahalno0 = np.dot(np.dot((x - m).T, np.linalg.inv(c)), (x - m))  # Mahalanobis Distance
    return 1 / ((2 * np.pi) ** (len(inputs_cols) / 2) * np.linalg.det(c) ** 0.5) * np.e ** (-0.5 * mahalno0)  # Likelihood

# function to find posterior probability
def posterior_pro(x):
    # index of this array will be class 0 and class 1 also
    pred=[]
    # finding posterior probability for both classes wrt to given test tuple
    total_prob=multivariate_probb(x,mean_class_0,cov_class_0)*P_C0+multivariate_probb(x, mean_class_1, cov_class_1)*P_C1
    pred.append((multivariate_probb(x,mean_class_0,cov_class_0)*P_C0)/total_prob)
    pred.append((multivariate_probb(x, mean_class_1, cov_class_1)*P_C1)/total_prob)
    return pred

# calculating predicted classes
predictions=[]
for _, test_rows in test[inputs_cols].iterrows():
    # converting test input tuples to numpy for fast processing
    test_rows=np.array(test_rows)
    # calling function
    p=posterior_pro(test_rows)
    # storing predicted class via bayes in a array
    predictions.append(np.argmax(np.array(p)))

print(" Classification accuracy is: ",round(accuracy_score(predictions,test[output_col])*100,4)," %")
print("The confusion matrix is : \n",confusion_matrix(test[output_col],predictions))

# importing dataframe for keeping best accuracies
data_best=pd.read_csv("Best_accuracies_Q4.csv")
data_best['Bayes Classifier']=round(accuracy_score(predictions,test[output_col])*100,3)
# printing best accuracies so far data frame
print(data_best)
# saving best accuracies dataframe to csv file
data_best.to_csv("Best_accuracies_Q4.csv",index=False)






