# Name: Yash Sharma
# Registration Number: B20241
# Mobile Number: 8802131138

# importing needed modules
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
import scipy as sp
from scipy import spatial as spatial

# importing the dataframe
df=pd.read_csv("Iris.csv")
cols = df.columns[:-1].to_list() # for columns except "class"

# creating pca object
pca = PCA(n_components=2)
# fitting our data
pca.fit(df[cols])
# transforming
transformed_df = pca.transform(df[cols])
df2=pd.DataFrame({'Column1': transformed_df[:, 0], 'Column2': transformed_df[:, 1]})

# printing all eigen values associated
transformed_df[:, 0] = transformed_df[:, 0] - np.mean(transformed_df[:, 0])
transformed_df[:, 1] = transformed_df[:, 1] - np.mean(transformed_df[:, 1])
cov_mat = np.cov(transformed_df[:, 0], transformed_df[:, 1])
eigen_values, eigen_vectors = np.linalg.eig(cov_mat)
print(" The eigenvalues associated with the directions are:\n ", sorted(eigen_values))
#
# # plotting the eigenvalues
# plt.plot([0,1],sorted(eigen_values),marker='o', markerfacecolor='blue')
# plt.xlabel("Index of eigenvalue")
# plt.ylabel("Eigenvalue")
# plt.title("All the eigenvalues in the ascending order")
# plt.show()
#
# # plotting the scatter plot of this data
# plt.scatter(transformed_df[:, 0], transformed_df[:, 1], alpha=0.85)
# plt.title("The scatter plot of reduced dimensional data")
# plt.xlabel("X----->")
# plt.ylabel("Y----->")
# plt.show()

##############################################################################################################################################################
# Question 2
##############################################################################################################################################################

# for K = 3
k=3
kmeans = KMeans(n_clusters=k)
kmeans.fit(df2)
kmeans_prediction = pd.DataFrame({'class': kmeans.predict(df2)})
combined=pd.concat([df2,kmeans_prediction],axis=1)

# cluster centres
centres=kmeans.cluster_centers_

# # plotting the data
# for i in range(k):
#     label='class '+str(i)
#     plt.scatter(combined.loc[combined['class']==i,'Column1'],combined.loc[combined['class']==i,'Column2'],label=label, alpha=0.85)
#     # plotting cluster centre
#     plt.scatter(centres[i][0],centres[i][1],color='black',marker="X",label="Centroids" if i == 0 else "")
# plt.title("Scatter plot of Data")
# plt.legend()
# plt.show()

# shortcut for plotting
# plt.scatter(df2.iloc[:,0],df2.iloc[:,1], c=kmeans.labels_, cmap='rainbow')
# plt.show()

# Save the distortion for each final cluster chosen by KMeans above.
distortion=kmeans.inertia_

# function to find distortion
def find_distortion(k,combined,clusters):
    dist=0
    for i in range(k):
        dist+=np.sum(np.subtract(np.array(combined.loc[combined['class'] == i, 'Column1']),clusters[i][0])**2)
        dist+=np.sum(np.subtract(np.array(combined.loc[combined['class'] == i, 'Column2']),clusters[i][1])**2)
    return dist

print(distortion)
# print(find_distortion(3,combined,centres))

# function to find purity score
def find_purity(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred)
    #print(contingency_matrix)
    # Find optimal one-to-one mapping between cluster labels and true labels
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    # Return cluster accuracy
    return contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix)

print(find_purity(kmeans_prediction,df['Species']))

##############################################################################################################################################################
# Question 3
##############################################################################################################################################################

# # initializing all k values in an array
# arr=[2,3,4,5,6,7]
# J=[]
# purity_scores=[]
# for k in arr:
#     kmeans = KMeans(n_clusters=k)
#     kmeans.fit(df2)
#     kmeans_prediction = pd.DataFrame({'class': kmeans.predict(df2)})
#     combined=pd.concat([df2,kmeans_prediction],axis=1)
#     # Save the distortion for each final cluster chosen by KMeans above.
#     distortion=kmeans.inertia_
#     # cluster centres
#     centres = kmeans.cluster_centers_
#
#     print("The Distortion measure and Purity score for k= "+str(k)+" is: ",round(distortion,3)," and ",round(find_purity(kmeans_prediction,df['Species'])*100,3),"%")
#
#     # saving value of distortion in J=[]
#     J.append(find_distortion(k,combined,centres))
#     # saving purity score in purity_score=[]
#     purity_scores.append(round(find_purity(kmeans_prediction,df['Species']),3)*100)
#
# # plotting scatter graph between distortion measure and k value (ni. of clusters)
# plt.plot(arr,J,marker='o',linestyle='dashed')
# plt.xlabel("Number of Cluster (K)")
# plt.ylabel("Distortion Measure")
# plt.title("The plot of K vs distortion measure")
# plt.show()
#
# # plot between purity score and number of clusters (K)
# plt.plot(arr,purity_scores,linestyle='dashed',marker='o')
# plt.xlabel("Number of Cluster (K)")
# plt.ylabel("Purity Score (%)")
# plt.title("The plot between K and Purity Score ")
# plt.show()
#
#
# print("The optimal number of clusters using Elbow method are: 3 ")
# print("The purity score for optimal value of cluster (K=4) is 88.70 %")

##############################################################################################################################################################
# Question 4
##############################################################################################################################################################

# # making GMM model
# K = 3
# gmm = GaussianMixture(n_components = K)
# gmm.fit(df2)
#
# gmm_prediction = pd.DataFrame({'class': gmm.predict(df2)})
# combined=pd.concat([df2,gmm_prediction],axis=1)
# # calculating cluster centre
# centres_gmm=[]
# for i in range(K):
#     vectors=np.array(combined.loc[combined['class'] == i,['Column1','Column2']])
#     index=np.argmin(metrics.pairwise_distances(vectors).sum(axis=0))
#     centres_gmm.append(vectors[index])
#
# # plotting the data clusters
# for i in range(K):
#     label='class '+str(i)
#     plt.scatter(combined.loc[combined['class']==i,'Column1'],combined.loc[combined['class']==i,'Column2'],label=label, alpha=0.85)
#     # plotting cluster centre
#     plt.scatter(centres[i][0],centres[i][1],color='black',marker="X",label="Centroids" if i == 0 else "")
# plt.title("Scatter plot of Data using GMM")
# plt.legend()
# plt.show()
#
# distortion=sum(gmm.score_samples(df2))
# print("The total log likelihood of the data as Distortion measure for k=3 is: ",distortion)
# # calculating the purity score and printing it
# print("The Purity score for GMM clustering for k=3 is: ",round(find_purity(gmm.predict(df2),df['Species'])*100,3),"%")

##############################################################################################################################################################
# Question 5
##############################################################################################################################################################

# # initializing all k values in an array
# arr=[2,3,4,5,6,7]
# J=[]
# purity_scores=[]
# for k in arr:
#     # making GMM model
#     gmm = GaussianMixture(n_components=k)
#     gmm.fit(df2)
#     gmm_prediction = pd.DataFrame({'class': gmm.predict(df2)})
#     combined=pd.concat([df2,gmm_prediction],axis=1)
#
#     # calculating distortion measure
#     distortion = sum(gmm.score_samples(df2))
#     # saving value of distortion in J=[]
#     J.append(distortion)
#
#     # saving purity score in purity_score=[]
#     purity_scores.append(round(find_purity(gmm.predict(df2),df['Species']),3)*100)
#
#     print("The Distortion measure and Purity score for k= "+str(k)+" is: ",round(distortion,3)," and ",round(find_purity(gmm.predict(df2),df['Species'])*100,3),"%")
#
# # plotting scatter graph between distortion measure and k value (ni. of clusters)
# plt.plot(arr,J,marker='o',linestyle='dashed')
# plt.xlabel("Number of Cluster (K)")
# plt.ylabel("Distortion Measure")
# plt.title("The plot of K vs distortion measure (log-likelihood)")
# plt.show()
#
# # plot between purity score and number of clusters (K)
# plt.plot(arr,purity_scores,linestyle='dashed',marker='o')
# plt.xlabel("Number of Cluster (K)")
# plt.ylabel("Purity Score (%)")
# plt.title("The plot between K and Purity Score ")
# plt.show()
#
#
# print("The optimal number of clusters using Elbow method are: 3 ")
# print("The purity score for optimal value of cluster (K=4) is 98 %")

##############################################################################################################################################################
# Question 6
##############################################################################################################################################################

def dbscan_predict(dbscan_model, X_new, metric=spatial.distance.euclidean):
    # Result is noise by default
    y_new = np.ones(shape=len(X_new), dtype=int)*-1
    # Iterate all input samples for a label
    for j, x_new in enumerate(X_new):
        # Find a core sample closer than EPS
        for i, x_core in enumerate(dbscan_model.components_):
            if metric(x_new, x_core) < dbscan_model.eps:
                # Assign label of x_core to x_new
                y_new[j] =dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                break
    return y_new


# making all possible combination between epsilon and Minpoints
possible=[[1,4],[1,10],[5,4],[5,10]]

for c in possible:
    eps=c[0]
    min_samples=c[1]
    # Now making DBSCAN model
    dbscan_model=DBSCAN(eps=eps, min_samples=min_samples).fit(df2)
    DBSCAN_predictions = dbscan_model.labels_
    dbtest = dbscan_predict(dbscan_model, np.array(df2), metric=spatial.distance.euclidean)
    dbtest=pd.DataFrame({'class':dbtest})
    combined = pd.concat([df2, dbtest], axis=1)

    # plotting the data
    for i in sorted(np.unique(dbtest),reverse=True):
        label='class '+str(i)
        plt.scatter(combined.loc[combined['class']==i,'Column1'],combined.loc[combined['class']==i,'Column2'],label=label, alpha=0.85)
    plt.title("Scatter plot of Data (using DBSCAN)\nFor epsilon="+str(eps)+" and min_samples="+str(min_samples))
    plt.legend(loc='lower right')
    plt.show()

    # Now computing Purity Scores
    print("The Purity score for epsilon=" + str(eps) + " and min_samples= "+str(min_samples)+" is: ", round(find_purity(dbtest, df['Species'])*100, 3) , "%")









