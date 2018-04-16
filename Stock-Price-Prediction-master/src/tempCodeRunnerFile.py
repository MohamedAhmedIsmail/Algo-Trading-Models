import numpy as np
import pandas as pd
from pandas import DataFrame
data = pd.read_excel('../data/Suez Cement.xls')
data = np.array(data)

print(data[0, :])