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
import random


path = r'final_dataset.csv/final_dataset.csv'
df = pd.read_csv(path)
print("He leído ya el dataset. Procedo a eliminar y tratar los datos.")
"""
 df['Flow ID'] = df['Flow ID'].astype(str)
df['Src IP'] = df['Src IP'].astype(str)
df['Dst IP'] = df['Dst IP'].astype(str)
"""
scaler = RobustScaler()

columnas_string = []

for columna in df.columns:
    if df[columna].apply(lambda x: isinstance(x, str)).all():
        columnas_string.append(columna)
print("Eliminando valores faltantes")
df = df.dropna()
data_X = df.drop(columnas_string, axis=1)  # All columns except "label"
data_y = df['Label']
# scaler = StandardScaler()



print("Voy a dividir los datos del dataset")
data_X_train, data_X_test = train_test_split(data_X, test_size=0.2, random_state=42)
data_Y_train, data_y_test = train_test_split(data_y, test_size=0.2, random_state=42)
data_X_train_scaled = scaler.fit_transform(data_X_train)
data_X_test_scaled = scaler.transform(data_X_test)

# data_y_pred = [1] * len(data_y_test)

# Change 'ddos' to 1 and any other value to 0:
print("Cambiando 'ddos' a 1 y cualquier otro caso a 0")
#data_X_train_encoded = pd.get_dummies(data_X_train)
#data_X_test_encoded = pd.get_dummies(data_X_test)
data_Y_train = data_Y_train.apply(lambda x: 1 if x == 'ddos' else 0)
data_y_test = data_y_test.apply(lambda x: 1 if x == 'ddos' else 0)

print("Vamos a entrenar el modelo")

# Training the model:
k = 5
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(data_X_train_scaled, data_Y_train)


# Doing the predictions :
print("Vamos a predecir")
data_y_pred = clf.predict(data_X_test_scaled)


# Printing the result of our model:
accuracy = accuracy_score(data_y_test, data_y_pred)


print("Precisión: ", accuracy)
