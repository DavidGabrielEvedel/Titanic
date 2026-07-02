import pandas as pd
df=pd.read_csv("data/titanic.csv")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())