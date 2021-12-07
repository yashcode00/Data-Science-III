import numpy as np
import pandas as pd



# function to find likelihood for a single sample
def multivariate_probb(x,m,c):
    dimension=2
    mahalno0 = np.dot(np.dot((x - m).T, np.linalg.inv(c)), (x - m))  # Mahalanobis Distance
    return 1 / ((2 * np.pi) ** (dimension / 2) * np.linalg.det(c) ** 0.5) * np.e ** (-0.5 * mahalno0)  # Likelihood


a=[12.5, 35.75, 38.25, 52.5, 53, 63]
a=[52.5,63,38.25,12.5,35.75,53]
b=[15.75, 29.25, 35.75, 54.25, 63, 67.5]
b=[63,54.25,29.25,35.75,67.5,15.75]

data = np.array([[x, y] for x in a for y in b])
data=pd.DataFrame(data)
m=np.array(data.mean())
c=np.array(data.cov())
c=np.cov(a,b)
print(m)
print(c)
print(multivariate_probb([150,50],m,c))