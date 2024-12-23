# thinh

import numpy as np
import pandas as pd

df = pd.read_csv('../data/dataset_traffic_accident_prediction1.csv')

df1 = df.drop(["Accident_Severity"], axis =1 )

out = df1.describe()

nan = df1.isnull().sum()

# tinh thu gini index

print(df1["Weather"].unique())
test = df1.dropna()
