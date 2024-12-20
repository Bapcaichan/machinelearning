import numpy as np
import pandas as pd

df = pd.read_csv("dataset_traffic_accident_prediction1.csv")

data = df.describe()

print(data)

