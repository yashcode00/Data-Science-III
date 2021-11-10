import numpy as np
import pandas as pd

count=0
for j in range(1,1000):
    for i in range(1,j+1):
        count+=1
        print(i," ",j)
print(count)
