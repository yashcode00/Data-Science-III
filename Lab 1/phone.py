import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as plt

df=pd.read_csv("phone_data.csv",sep=',')
df1=df.groupby(['network','item'])
voda=df1.get_group(('Vodafone','call'))

for name,group in df1:
    print(name)

# print(voda)