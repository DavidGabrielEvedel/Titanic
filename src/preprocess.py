import pandas as pd
df = pd.read_csv("data/titanic.csv")
print(df.info())
print(df.describe())
print(df.isnull().sum())
df.drop("Cabin", axis=1, inplace=True)
df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
print("\nMissing Values After Cleaning:\n")
print(df.isnull().sum())

# Feature Engineering
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

print(df[["SibSp", "Parch", "FamilySize"]].head())

df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

print(df[["FamilySize", "IsAlone"]].head())
df["Sex"] = df["Sex"].map({
    "male": 0,
    "female": 1
})
df = pd.get_dummies(
    df,
    columns=["Embarked"],
    drop_first=True
)
df["Embarked_Q"] = df["Embarked_Q"].astype(int)
df["Embarked_S"] = df["Embarked_S"].astype(int)
print(df.head())
print(df.info())
df.to_csv("data/cleaned_titanic.csv", index=False)

print("Cleaned dataset saved successfully!")