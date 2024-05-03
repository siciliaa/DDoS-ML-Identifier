import pandas as pd
import numpy as np
path = '../final_dataset.csv/final_dataset.csv'
data = pd.read_csv(path)


df = pd.read_csv(path)
missing = df.isna().sum()
print("Valores perdidos: ", missing)

