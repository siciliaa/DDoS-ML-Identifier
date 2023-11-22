import copy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import csv
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import random

# Leemos el DF:

path = r'../final_dataset.csv/final_dataset.csv'
df = pd.read_csv(path)
print("He leído ya el dataset.")


# Cogemos todas las columnas menos la etiqueta, num y flowID:
columnas_eliminar = ['Label', 'Flow ID']

data_X = df.drop(columnas_eliminar, axis=1)  # All columns except "label"
data_y = df['Label']

print("Vamos a quitarle el punto a las IPs")

for index, value in data_X['Src IP'].items():
    data_X.at[index, 'Src IP'] = value.replace('.', '')

for index, value in data_X['Dst IP'].items():
    data_X.at[index, 'Dst IP'] = value.replace('.', '')

print("Ahora vamos a cambiar 'ddos' a 1 y cualquier otro caso a 0 ")
data_y = data_y.apply(lambda x: 1 if x == 'ddos' else 0)

print("Vamos a tratar los NaN")

# Columna 1:

for index, value in data_X['Src IP'].items():
    if pd.isna(data_X.at[index, 'Src IP']):
        data_X.at[index, 'Src IP'] = 0


# Columna 2:

for index, value in data_X['Src Port'].items():
    if pd.isna(data_X.at[index, 'Src Port']):
        data_X.at[index, 'Src Port'] = 0


# Columna 3:

for index, value in data_X['Dst Port'].items():
    if pd.isna(data_X.at[index, 'Dst Port']):
        data_X.at[index, 'Dst Port'] = 0



# Columna 4:

for index, value in data_X['Dst IP'].items():
    if pd.isna(data_X.at[index, 'Dst IP']):
        data_X.at[index, 'Dst IP'] = 0


# Columna 5:

for index, value in data_X['Protocol'].items():
    if pd.isna(data_X.at[index, 'Protocol']):
        data_X.at[index, 'Protocol'] = 0


# Dividimos los datos del DF:

print("Voy a dividir los datos del dataset")
data_X_train, data_X_test = train_test_split(data_X, test_size=0.2, random_state=42)
data_Y_train, data_y_test = train_test_split(data_y, test_size=0.2, random_state=42)


"""for index, value in data_X_train['Src IP'].items():
    data_X_train.at[index, 'Src IP'] = value.replace('.', '')
print(data_X['Src IP'])"""




# Eliminar NaN de las primeras 5 columnas




"""

print(data_X_train)

subset_X_train = data_X_train.iloc[:, :5]
print("Hola")
print(subset_X_train)"""

# Entrenar el modelo
k = 5
clf = KNeighborsClassifier(n_neighbors=k)


subset_X_train = data_X_train.iloc[:, :5]
subset_X_test = data_X_test.iloc[:, :5]

"""print(subset_X_train)

print("********************************************")
print("********************************************")
print("********************************************")
print("********************************************")

print(subset_X_test)"""


clf.fit(subset_X_train, data_Y_train)

print("Vamos a predecir")
data_y_pred = clf.predict(subset_X_test)

accuracy = accuracy_score(data_y_test, data_y_pred)
print(accuracy)
