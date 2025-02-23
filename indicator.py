import pandas as pd
from pandas_datareader import wb

df = wb.get_indicators()[['id','name']]
df = df[df.name == 'Birth rate, crude (per 1,000 people)']
print(df)