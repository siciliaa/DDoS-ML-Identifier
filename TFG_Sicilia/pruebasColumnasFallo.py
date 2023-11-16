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

path = r'final_dataset.csv/final_dataset.csv'
df = pd.read_csv(path)
print("He leído ya el dataset. Procedo a eliminar y tratar los datos.")

# Cogemos todas las columnas menos la etiqueta:

data_X = df.drop('Label', axis=1)  # All columns except "label"
data_X = pd.get_dummies(data_X)
data_y = df['Label']

# Dividimos los datos del DF:

print("Voy a dividir los datos del dataset")
data_X_train, data_X_test = train_test_split(data_X, test_size=0.2, random_state=42)
data_Y_train, data_y_test = train_test_split(data_y, test_size=0.2, random_state=42)

# Change 'ddos' to 1 and any other value to 0:
print("Cambiando 'ddos' a 1 y cualquier otro caso a 0")
data_Y_train = data_Y_train.apply(lambda x: 1 if x == 'ddos' else 0)
data_y_test = data_y_test.apply(lambda x: 1 if x == 'ddos' else 0)


categorical_columns = data_X.select_dtypes(include=['object']).columns
informe = []


# Bucle para iterar sobre las columnas
for i in range(1, len(data_X.columns) + 1):
    # Generar todas las combinaciones posibles de columnas
    column_combinations = [data_X.columns[j:j+i] for j in range(len(data_X.columns) - i + 1)]

    # Iterar sobre las combinaciones de columnas
    for columns_subset in column_combinations:
        # Seleccionar el subconjunto de columnas
        subconjunto_X_train = data_X_train[columns_subset]
        subconjunto_X_test = data_X_test[columns_subset]

        # Aplicar la transformación one-hot solo a las columnas categóricas
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(), categorical_columns)
            ],
            remainder='passthrough'
        )
        subconjunto_X_train_encoded = preprocessor.fit_transform(subconjunto_X_train)
        subconjunto_X_test_encoded = preprocessor.transform(subconjunto_X_test)

        # Entrenar el modelo
        k = 5
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(subconjunto_X_train_encoded, data_Y_train)

        # Hacer las predicciones
        data_y_pred = clf.predict(subconjunto_X_test_encoded)

        # Calcular la precisión
        accuracy = accuracy_score(data_y_test, data_y_pred)

        # Almacenar resultados en el informe
        informe.append({
            'Num_Columnas': i,
            'Columnas': columns_subset,
            'Precisión': accuracy
        })


for resultado in informe:
    print(f"\nUtilizando las columnas: {resultado['Columnas']}")
    print(f"Precisión: {resultado['Precisión']}")
