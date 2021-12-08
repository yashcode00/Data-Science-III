# Name: Yash Sharma
# Registration Number: B20241
# Mobile Number: 8802131138

# importing needed modules
import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as plt

# importing the dataframe
df = pd.read_csv("pima-indians-diabetes.csv", sep=",")
cols = df.columns[:-1]

##############################################################################################################################################################
# Part (a)
##############################################################################################################################################################\

# generating 2d random bi-variate gaussian distributed data
mean_matrix = np.array([0, 0])
# This is my covariance matrix obtained from 2 x 1000 points
cov_matrix = np.array([[13, -3], [-3, 5]])

# generating the random 2x1000 matrix
D = np.random.multivariate_normal(mean_matrix, cov_matrix, 1000)

# separating X and Y coordinates for plotting later
x = D[:, 0]
y = D[:, 1]

# plotting the scatter plot for this data
plt.scatter(x,y,alpha=0.85)
plt.xlabel("X ----->")
plt.ylabel("Y ----->")
plt.title("Scatter plot of the data samples")
plt.show()

##############################################################################################################################################################
# Part (b)
##############################################################################################################################################################\

# calculating eigen-values and vectors
eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
eig_vec1 = eigen_vectors[:, 0]
eig_vec2 = eigen_vectors[:, 1]
origin = [0, 0]


# plotting the scatter plot for this data with eigen directions
plt.scatter(x,y,alpha=0.85)
plt.xlabel("X ----->")
plt.ylabel("Y ----->")
plt.title("Scatter plot of the data samples with Eigen directions (with arrows/lines)")
plt.quiver(*origin, *eig_vec1, color=['r'], scale=eigen_values[1])
plt.quiver(*origin, *eig_vec2, color=['r'], scale=eigen_values[0])
plt.show()

##############################################################################################################################################################
# Part (c)
##############################################################################################################################################################

# defining function to project data on given vector and then plot it
def project_data(vec, data):
    # finding vector magnitude
    vec_magnitude = np.sqrt(sum(vec ** 2))
    # projecting the data on given vector
    data= np.transpose(data)
    scalar_data=(np.dot(vec,data)/np.dot(vec,vec))
    data_projected=list(map(lambda x:x*vec,scalar_data))
    data_projected=np.array(data_projected)
    # returning projected data for further investigations
    return data_projected

# for vector title
i=1
# loop for calling function to project then plotting them individually
for vector in [eig_vec1, eig_vec2]:
    data_projected = project_data(vector, D)
    # now plotting all projected data using scatter plot superimposed on all eigenvectors
    # extracting x and y components from projected data
    x = data_projected[:, 0]
    y = data_projected[:, 1]
    # plotting the scatter plot for this data with eigen directions
    # plotting original D matrix also along with it
    plt.scatter(D[:,0],D[:,1], alpha=0.65)
    plt.scatter(x, y, alpha=0.75, color=['lawngreen'])
    plt.xlim(-10,10)
    plt.xlabel("X ----->")
    plt.ylabel("Y ----->")
    plt.title("Scatter plot of the data samples projected on Eigen vector "+str(i))
    plt.quiver(*origin, *eig_vec1, color=['r'], scale=eigen_values[1])
    plt.quiver(*origin, *eig_vec2, color=['r'], scale=eigen_values[0])
    plt.tight_layout()
    plt.show()
    # increase vector number just for plt.title
    i+=1

##############################################################################################################################################################
# Part (d)
##############################################################################################################################################################

# reconstructing data and finding Euclidean distance error
# we will find error for one of the projected data let us say on eigen vector 1
# calling function for projected data
data_on_vec1=project_data(eig_vec1,D)
data_on_vec2=project_data(eig_vec2,D)
data_projected_reconstruct = data_on_vec1+data_on_vec2
# making dataframe to ease our work
D_reconstructed=pd.DataFrame({"X":data_projected_reconstruct[:,0],"Y":data_projected_reconstruct[:,1]})
D_original=pd.DataFrame({"X":D[:,0],"Y":D[:,1]})
# finding tuple wise Euclidean error and storing in new column names "Error"
D_error=D_original-D_reconstructed
D_error["Error"]=(D_error['X']**2+D_error["Y"]**2)**0.5

print("The reconstruction error between 𝐃̂ and D using Euclid distance is: ",round(D_error['Error'].mean(axis=0),3))


