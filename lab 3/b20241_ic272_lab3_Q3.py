# Name: Yash Sharma
# Registration Number: B20241
# Mobile Number: 8802131138

# importing needed modules
import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pd.set_option('display.max_rows', 1000,'display.max_columns', 1000)

# importing dataset processed in problem 1(b) final
df = pd.read_csv("pima-indians-diabetes-after-question-1.csv")
cols = df.columns[:-1].to_list() # for columns except "class"

# creating pca object
pca = PCA(n_components=2)
# fitting our data
pca.fit(df[cols])
# transforming
transformed_df = pca.transform(df[cols])
print("The variance of column 1 is: ", st.variance(transformed_df[:, 0]))
print("The variance of column 2 is: ", st.variance(transformed_df[:, 1]))
# printing all eigen values associated
transformed_df[:, 0] = transformed_df[:, 0] - np.mean(transformed_df[:, 0])
transformed_df[:, 1] = transformed_df[:, 1] - np.mean(transformed_df[:, 1])
cov_mat = np.cov(transformed_df[:, 0], transformed_df[:, 1])
eigen_values, eigen_vectors = np.linalg.eig(cov_mat)
print(" The eigenvalues associated with the directions are:\n ", sorted(eigen_values))

# plotting the scatter plot of this data
plt.scatter(transformed_df[:, 0], transformed_df[:, 1], alpha=0.85)
plt.title("The scatter plot of reduced dimensional data")
plt.xlabel("X----->")
plt.ylabel("Y----->")
plt.show()
print()

##############################################################################################################################################################
# Part (b)
##############################################################################################################################################################\

# extracting all eigen values
eigen_values_all, eigen_vectors_all=np.linalg.eig(df[cols].cov())
# x-values
x = [i for i in range(len(eigen_values_all))]
# corresponding y-values
y = sorted(eigen_values_all,reverse=True)
# plotting the points
plt.plot(x, y, color='green', linestyle='dashed', linewidth=3, marker='o', markerfacecolor='blue', markersize=12)
plt.xlabel("Index of eigen-value")
plt.ylabel("Eigen-Value")
plt.xticks(x)
plt.title("All the eigenvalues in the descending order")
plt.show()

##############################################################################################################################################################
# Part (c)
##############################################################################################################################################################\

# Reconstruction of data
l=[i for i in range(1,9)]
Euclidean_error=[]
for n in l:
    # making pca object
    pca=PCA(n_components=n)
    # fitting the data of 8 dimensions
    pca.fit(df[cols])
    # transforming data to required dimensions
    reduced_df=pca.transform(df[cols])
    # reconstructed the complete data from reduced dimension data
    reconstructed_data=pca.inverse_transform(reduced_df)
    # dict to store reconstructed data as dataframe
    dict = {}
    for i in range(8):
        dict[cols[i]]=reconstructed_data[:,i]
    # making dataframe of dict
    df_reconstruct=pd.DataFrame(dict)
    print("The covariance matrix for l=",n, " is: ")
    # making dataframe of reduced data to find covariance matrix
    dict_cov={}
    for i in range(n):
        dict_cov["Column"+str(i)]=reduced_df[:,i]
    cov_mat=pd.DataFrame(dict_cov)
    cov_mat =cov_mat.cov()
    print(cov_mat,"\n")
    # finding reconstruction error via Euclidean distance
    df_error=df_reconstruct.subtract(df[cols])
    df_error["Error"]=((df_error[cols]**2).sum(axis=1))**0.5
    # taking mean of all tuple-wise euclidean errors
    Euclidean_error.append(df_error["Error"].mean(axis=0))

# plotting all reconstruction error whit value of respective reduced dimension on x-axis
plt.plot(l, Euclidean_error, color='grey', linestyle='dashed', linewidth=1, marker='o', markerfacecolor='blue',alpha=0.85, markersize=8)
plt.xlabel("Values of l")
plt.ylabel("Euclidean Error")
plt.xticks(l)
plt.title("The reconstruction errors in terms of Euclidean-Error")
plt.show()

##############################################################################################################################################################
# Part (d)
##############################################################################################################################################################

print()
print("The covariance matrix for the original data (8-dimensional) is: ")
cov_mat_original=df[cols].cov()
print(cov_mat_original)
# finding cov_matrix for l=8 using pca again
pca=PCA(n_components=8)
pca.fit(df[cols])
reduced_df=pca.transform(df[cols])
dict_cov={}
for i in range(8):
    dict_cov[cols[i]]=reduced_df[:,i]
cov_mat=pd.DataFrame(dict_cov)
cov_mat =cov_mat.cov()
print("\nThe covariance matrix for 8-dimensional representation obtained using PCA with l = 8 is: ")
print(cov_mat)



