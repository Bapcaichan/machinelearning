# thinh
import operator
import csv
import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv('../data/dataset_traffic_accident_prediction1.csv')
# df = pd.read_csv(r'C:\Users\ADMIN\Documents\VScode\Python\github\machinelearning\data\dataset_traffic_accident_prediction1.csv')



df.describe()

df.isnull().sum()
columns_with_missing = ['Traffic_Density', 'Speed_Limit', 'Number_of_Vehicles', 'Driver_Alcohol',
                       'Driver_Age', 'Driver_Experience',
                       'Accident']

for i in columns_with_missing:
    print(f"{i} - TrungBinh: {df[i].mean()}, TrungVi: {df[i].median()}")
with open('../data/input.csv', mode='w', newline='', encoding='utf-8') as file: writer = csv.writer(file) 
writer.writerows(df3)
