import numpy as np
import pandas as pd

df = pd.read_csv(r'C:\Users\ADMIN\Documents\VScode\Python\github\machinelearning\data\dataset_traffic_accident_prediction1.csv')

df = df.drop(df['Accident_Severity'],axis=1)

df1 = df['Road_Light_Condition'].fillna(df['Road_Light_Condition'].mode(), inplace = True)
df1 = df['Road_Type'] = df['Road_Type'].fillna(df['Road_Type'].mode()[0])
df1 = df['Driver_Alcohol'] = df['Driver_Alcohol'].fillna(df['Driver_Alcohol'].mode()[0])


