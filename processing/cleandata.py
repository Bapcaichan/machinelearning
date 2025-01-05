# thinh
import operator
import csv
import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv('../data/dataset_traffic_accident_prediction1.csv')
# df = pd.read_csv(r'C:\Users\ADMIN\Documents\VScode\Python\github\machinelearning\data\dataset_traffic_accident_prediction1.csv')

df1 = df.drop(["Accident_Severity"], axis =1 )

out = df1.describe()

nan = df1.isnull().sum() # co nhieu gia tri nan khong cung 1 record

#xu ly cac hang co 2 gia tri null
df2 = df1.dropna(thresh = 11)
out1 = df2.describe()

obj = df2.select_dtypes(include = 'object').columns

#xu ly du lieu nominal bang tan suat
for i in obj:

    df2[i] = df2[i].fillna(df2[i].mode()[0])

#xu ly du lieu so bang trung vi

flt = df2.select_dtypes(include = 'float').columns

for i in flt:
    df2[i] = df2[i].fillna(df2[i].median())

out2 = df2.isna().sum()

# Bo du lieu trung lap
df3 = df.drop_duplicates()

# profile = ProfileReport(df3, title="Traffic accident", explorative=True)
# profile.to_file("Traffic.html")

with open('../data/input.csv', mode='w', newline='', encoding='utf-8') as file: writer = csv.writer(file) 
writer.writerows(df3)
