# thinh
import operator
import csv
import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns

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



#Kiểm tra dữ liệu
columns_miss = df.columns[df.isnull().sum() > 0]
df[columns_miss].info()
#Xử lý dữ liệu thiếu bằng trung bình
df['Weather'] = df['Weather'].fillna(df['Weather'].mode()[0])
df['Road_Type'] = df['Road_Type'].fillna(df['Road_Type'].mode()[0])
df['Time_of_Day'] = df['Time_of_Day'].fillna(df['Time_of_Day'].mode()[0])
df['Accident_Severity'] = df['Accident_Severity'].fillna(df['Accident_Severity'].mode()[0])
df['Road_Condition'] = df['Road_Condition'].fillna(df['Road_Condition'].mode()[0])
df['Vehicle_Type'] = df['Vehicle_Type'].fillna(df['Vehicle_Type'].mode()[0])
df['Road_Light_Condition'] = df['Road_Light_Condition'].fillna(df['Road_Light_Condition'].mode()[0])
#Kiểm tra dữ liệu thiếu
df.isnull().sum()


#Kiểm tra bản ghi trùng lặp
duplicates_count = df.duplicated().sum()
print(f"Ban ghi trung lap: {duplicates_count}")
#Xóa các bản ghi trùng lặp
df = df.drop_duplicates()
duplicates_count1 = df.duplicated().sum()
print(f"Ban ghi trung lap: {duplicates_count1}")

#Kiểm tra ngoại lệ

# Vẽ biểu đồ boxplot cho Speed_Limit
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['Speed_Limit'])
plt.title('Speed_Limit')
plt.show()

# Vẽ biểu đồ boxplot cho Number_of_Vehicles
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['Number_of_Vehicles'])
plt.title('Number_of_Vehicles')
plt.show()

# Vẽ biểu đồ boxplot cho Driver_Age
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['Driver_Age'])
plt.title('Driver_Age')
plt.show()

# Vẽ biểu đồ boxplot cho Driver_Experience
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['Driver_Experience'])
plt.title('Driver_Experience')
plt.show()

# Vẽ biểu đồ boxplot cho Traffic_Density
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['Traffic_Density'])
plt.title('Traffic_Density')
plt.show()

cols = ['Speed_Limit', 'Number_of_Vehicles', 'Driver_Age', 'Driver_Experience']
sns_plot = sns.pairplot(df[cols])
plt.show()

# with open('../data/input.csv', mode='w', newline='', encoding='utf-8') as file: writer = csv.writer(file) 
# writer.writerows(df3)