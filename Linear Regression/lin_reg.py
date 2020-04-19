import numpy as np
from sklearn import linear_model
import pandas as pd

df = pd.read_csv('homeprices.csv')

reg = linear_model.LinearRegression()

reg.fit(df[['area']], df.price)
print(df)
print(reg.predict([[3100]]))

print('In mx+b format, m = {}, b = {}'.format(reg.coef_,reg.intercept_))

df2 = pd.read_csv('areas.csv')

df2['prices'] = reg.predict(df2)

df2.to_csv('areas&prices.csv',index = False)
