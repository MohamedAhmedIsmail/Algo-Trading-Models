

import numpy as np
import pandas as pd
from pandas import DataFrame
files = ['Oriental Weavers.xls', 'T M G Holding.xls', 'Telecom Egypt.xls']
outs = ['Oriental Weavers.csv', 'T M G Holding.csv', 'Telecom Egypt.csv']

for file, out in zip(files, outs):
    data = pd.read_excel('../data/'+file)
    data = np.array(data)
    data = data[:, 2:]
    columns = ['TRADE_VOLUME'	, 'TRADE_VALUE'	, 'TRADE_COUNT'	,
               'OPEN_PRICE',	'HIGH_PRICE'	, 'LOW_PRICE', 'CLOSE_PRICE']
    data = DataFrame(data=data, columns=columns)
    DataFrame.to_csv(data, out)
