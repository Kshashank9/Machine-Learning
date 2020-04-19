import pandas as pd
import numpy as np
from sklearn import linear_model
import math

df = pd.read_csv('homeprices.csv')

print(df)

median_bedrms = math.floor(df.bedrooms.median())
df_bed = df.bedrooms.fillna(median_bedrms)

df.bedrooms = df_bed
reg = linear_model.LinearRegression()

reg.fit(df[['area','bedrooms','age']],df.price)

print(reg.predict([[3000,3.0,40]]))
