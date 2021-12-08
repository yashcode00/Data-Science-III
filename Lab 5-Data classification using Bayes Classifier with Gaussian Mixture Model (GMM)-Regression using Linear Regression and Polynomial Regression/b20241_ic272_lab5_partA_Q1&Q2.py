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
from sklearn.mixture import GaussianMixture

########################################################################################################################
# Importing and splitting data into 70-30 manner (train and test sets)------------------------------>>>>>>>>>>>>>>>>>>>>
########################################################################################################################

# importing and splitting of data
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
train = pd.concat([inputs_train, target_train], axis=1)
test = pd.concat([inputs_test, target_test], axis=1)
train.to_csv("SteelPlateFaults-train.csv", index=False)
test.to_csv("SteelPlateFaults-test.csv", index=False)

########################################################################################################################
# Splitting ends here------------------------------------------------------------------------------->>>>>>>>>>>>>>>>>>>>
########################################################################################################################

########################################################################################################################
# A. Question 1------------------------------------------------------------------------------------->>>>>>>>>>>>>>>>>>>>
########################################################################################################################

print("\nPart A: Q1.")

# making input and output attribute names arrays
inputs_cols = np.array(train.columns[:-1])
output_col = train.columns[-1]
test_inputs = np.array(test[inputs_cols])
test_targets = np.array(test[output_col])

# dividing the data class wise
df_class_0 = train[train['Class'] == 0].copy()
df_class_1 = train[train['Class'] == 1].copy()

# Prior Probability ( P(Ci) )
P_C0=len(train[train["Class"]==0])/(len((train["Class"])))
P_C1=len(train[train["Class"]==1])/(len((train["Class"])))


# array to store best accuracy
best=[]
# q is array to store all component values to be used in GMM
q = [2, 4, 8, 16]
for x in q:
    # making gaussian model for both given classes i.e. 0 and 1 and then fitting the model to its respective data
    # GMM model for class 0
    gmm_0 = GaussianMixture(n_components=x, covariance_type='full', random_state=42,reg_covar=1e-5)
    gmm_0.fit(np.array(df_class_0[inputs_cols]))
    # GMM model for class 1
    gmm_1 = GaussianMixture(n_components=x, covariance_type='full', random_state=42,reg_covar=1e-5)
    gmm_1.fit(np.array(df_class_1[inputs_cols]))

    # calculating the log-likelihoods and taking exponential and then multiply it with prior probabilities
    # of its respective class to have numerator of probability
    log_likelihoods_0 = np.exp(np.array(gmm_0.score_samples(test_inputs))).reshape(-1, 1)*P_C0
    log_likelihoods_1 = np.exp(np.array(gmm_1.score_samples(test_inputs))).reshape(-1, 1)*P_C1

    # finding total prob (denominator)
    total_prob=log_likelihoods_0+log_likelihoods_1

    # final probability numerator/denominator
    for val in total_prob:
        # checking if total prob is not zero
        if val!=0:
            prob_0=log_likelihoods_0/val
            prob_1=log_likelihoods_1/val

    # array to store final prediction on test samples predicted class
    predictions=[]
    for i in range(len(log_likelihoods_0)):
        # array to store temporarily the probabilities of given sample belonging to any of given
        # class (has 2 values as 2 classes)
        pred_each=[]
        pred_each.extend([prob_0[i],prob_1[i]])
        # appending the class 0 or class 1 based on the higher probability (np.argmax used for index 0 or 1)
        predictions.append(np.argmax(pred_each))

    # finding accuracy
    accuracy=round(accuracy_score(predictions,test_targets)*100,4)
    # appending calculated accuracy to this array for later comparision for best accuracy
    best.append(accuracy)
    print("Classification accuracy for "+str(x)+" Gaussian components (modes) is: ",accuracy," %")
    print("The confusion matrix is : \n", confusion_matrix(test_targets, predictions))

# printing best accuracy obtained and its respective Q value
print("\nThe best classification accuracy is "+str(max(best))+" % observed when Q="+str(q[np.argmax(best)]))
print()

########################################################################################################################
# A. Question 2------------------------------------------------------------------------------------->>>>>>>>>>>>>>>>>>>>
########################################################################################################################

print("\nPart A: Q2.")
# reading best accuracies from assignment 4
best_accu=pd.read_csv("Best_accuracies_Q4.csv")
best_accu["Bayes classifier using GMM"]=max(best)
print(best_accu)
print("\nResult: KNN on normalised data gives the best classification accuracy of ~97% among all other classifiers.")
# saving best accuracies dataframe to csv file
best_accu.to_csv("Best_accuracies_Q4.csv",index=False)


