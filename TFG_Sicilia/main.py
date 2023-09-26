import copy

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

path = r'final_dataset.csv\final_dataset.csv'
df = pd.read_csv(path)

data_X = df.drop('Label', axis=1)
data_y = df['Label']

data_X_train, data_X_test = train_test_split(data_X, test_size=0.2, random_state=42)
data_Y_train, data_y_test = train_test_split(data_y, test_size=0.2, random_state=42)

# Vamos a hacer una copia de data_y_test profunda, para que podamos eliminar las etiquetas:

data_y_pred = copy.deepcopy(data_y_test)  # Si hiciéramos data_y_pred = data_y_test, lo que pasaría es que estaríamos
# haciendo una copia por referencia

data_y_pred = data_y_pred.apply(lambda x: np.nan)


# Comprobar si hemos dividido bien el dataset:

print(len(data_X_train))
print(len(data_Y_train))

print(len(data_X_test))
print(len(data_y_test))

"""
for i in range(min(10, len(data_X))):
    print(data_X.iloc[i])
    
"""
ddos = 0
no_ddos = 0
t = 0
for i in range(len(data_y)):
    if data_y[i] == 'ddos':
        ddos += 1
    else:
        if t == 0:
            print(i)
            print(data_y.iloc[i])
            t = 1
        no_ddos += 1


print("DDoS: ", ddos)
print("No DDoS: ", no_ddos)


"""
print("X e y del train: ")
print(len(data_X_train))
print(len(data_y_train))

print("X e y del test: ")
print(len(data_X_test))
print(len(data_y_test))
"""
