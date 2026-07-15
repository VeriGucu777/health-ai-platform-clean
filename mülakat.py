import pandas as pd
df = pd.read_csv("diabetes.csv")
print(df.head(10))

df.info()
print(df.describe())

print(df.isnull().sum())
print((df == 0).sum())