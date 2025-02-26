import pandas as pd
from pandas_datareader import wb

df = wb.get_indicators()[['id','name']]
df = df[df.name == 'Mortality rate, under-5 (per 1,000 live births)']
print(df)