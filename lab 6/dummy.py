import numpy as np
import pandas as pd

dict={'Date': pd.date_range(start='10/3/2021', end='1/31/2022'), 'Cases': np.zeros(121)}
test=pd.DataFrame(dict)
print(test)