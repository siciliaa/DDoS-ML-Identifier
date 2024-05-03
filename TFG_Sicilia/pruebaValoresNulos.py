import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

# Leemos el DF:

path = r'final_dataset.csv/final_dataset.csv'
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

for index, value in data_X['Timestamp'].items():
    if "AM" in value or "PM" in value:
        fecha_hora_obj = datetime.strptime(value, "%d/%m/%Y %I:%M:%S %p")
        timestamp = fecha_hora_obj.timestamp()
        data_X.at[index, 'Timestamp'] = timestamp
    else:
        fecha_hora_obj = datetime.strptime(value, "%d/%m/%Y %H:%M:%S")
        timestamp = fecha_hora_obj.timestamp()
        data_X.at[index, 'Timestamp'] = timestamp

print("Ahora vamos a cambiar 'ddos' a 1 y cualquier otro caso a 0 ")
data_y = data_y.apply(lambda x: 1 if x == 'ddos' else 0)

print("Vamos a tratar los NaN")
# Variable de control para saber por que columna acabamos de analizar:


# Dividimos los datos del DF:

print("Voy a dividir los datos del dataset")
data_X_train, data_X_test = train_test_split(data_X, test_size=0.2, random_state=42)
data_Y_train, data_y_test = train_test_split(data_y, test_size=0.2, random_state=42)

# Entrenar el modelo
k = 5
clf = KNeighborsClassifier(n_neighbors=k)

print("Creación de subset.")

subset_X_train = data_X_train.iloc[:, :15]
subset_X_test = data_X_test.iloc[:, :15]

print("Entrenar modelo.")
clf.fit(subset_X_train, data_Y_train)

print("Vamos a predecir.")
data_y_pred = clf.predict(subset_X_test)

accuracy = accuracy_score(data_y_test, data_y_pred)
print(accuracy)