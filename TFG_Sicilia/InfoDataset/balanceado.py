import pandas as pd

path = '../final_dataset.csv/final_dataset.csv'
data = pd.read_csv(path)
"""
counts = data['Label'].value_counts()

print("Cantidad de muestras por clase: ")
print(counts)
"""

ddos_count = (data['Label'] == 'ddos').sum()
bening_count = (data['Label'] == 'Benign').sum()

print("DDoS: ", ddos_count)
print("Benign: ", bening_count)


