
import numpy as np
import pandas as pd

# df = pd.read_csv(r'../data/dataset_traffic_accident_prediction1.csv')
df = pd.read_csv(r'C:\Users\ADMIN\Documents\VScode\Python\github\machinelearning\data\dataset_traffic_accident_prediction1.csv')



df1 = df.drop(["Accident_Severity"], axis =1 )

df1['Road_Light_Condition'].fillna(df1['Road_Light_Condition'].mode()[0], inplace=True)
df1['Road_Type'].fillna(df['Road_Type'].mode()[0], inplace= True)
df1['Driver_Alcohol'].fillna(df['Driver_Alcohol'].mode()[0], inplace= True)







# thinh
out = df1.describe()

nan = df1.isnull().sum()

# tinh thu gini index

print(df1["Weather"].unique())
test = df1.dropna()
print(test)