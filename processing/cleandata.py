# thinh
import operator
import csv
import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt

# df = pd.read_csv('../data/dataset_traffic_accident_prediction1.csv')
df = pd.read_csv(r'C:\Users\ADMIN\Documents\VScode\Python\github\machinelearning\data\dataset_traffic_accident_prediction1.csv')



df.describe()

df.isnull().sum()
#kiểm tra các thuộc tính
columns_with_missing = ['Traffic_Density', 'Speed_Limit', 'Number_of_Vehicles', 'Driver_Alcohol',
                       'Driver_Age', 'Driver_Experience',
                       'Accident']

for i in columns_with_missing:
    print(f"{i} - TrungBinh: {df[i].mean()}, TrungVi: {df[i].median()}")
    df[i].plot(kind='hist', bins=30, title=i)
    plt.axvline(df[i].mean(), color='red', linestyle='dashed', linewidth=2)
    plt.axvline(df[i].median(), color='blue', linestyle='dashed', linewidth=2)
    plt.show()
#Xử lý các dữ liệu thiếu
df['Traffic_Density'].fillna(df['Traffic_Density'].median(), inplace=True)
df['Speed_Limit'].fillna(df['Speed_Limit'].median(), inplace=True)
df['Number_of_Vehicles'].fillna(df['Number_of_Vehicles'].median(), inplace=True)
df['Driver_Alcohol'].fillna(df['Driver_Alcohol'].median(), inplace=True)
df['Driver_Age'].fillna(df['Driver_Age'].median(), inplace=True)
df['Driver_Experience'].fillna(df['Driver_Experience'].median(), inplace=True)
df['Accident'].fillna(df['Accident'].median(), inplace=True)
# with open('../data/input.csv', mode='w', newline='', encoding='utf-8') as file: writer = csv.writer(file) 
# writer.writerows(df3)
