import pandas as pd

path = '../final_dataset.csv/final_dataset.csv'
data = pd.read_csv(path)

data1 = data['Bwd IAT Std']
null_count = data1.isna().sum()
print("Número de valores nulos: ", null_count)