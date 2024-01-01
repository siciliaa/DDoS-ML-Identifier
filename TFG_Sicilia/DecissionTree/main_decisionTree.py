import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
from sklearn import tree
import graphviz


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

# Variable de control para saber por que columna acabamos de analizar:

a = 0

for i in data_X.columns[:25]:
    for index, value in data_X[i].items():
            if pd.isna(data_X.at[index, i]):
                data_X.at[index, i] = 0
            elif value == np.inf:
                data_X.at[index, i] = int(100000000000)
            elif isinstance(value, float):
                data_X.at[index, i] = int(value * 100000)

    print(a)
    a += 1

print("Timestamp")

for index, value in data_X['Timestamp'].items():
    if "AM" in value or "PM" in value:
        fecha_hora_obj = datetime.strptime(value, "%d/%m/%Y %I:%M:%S %p")
        timestamp = fecha_hora_obj.timestamp()
        data_X.at[index, 'Timestamp'] = timestamp
    else:
        fecha_hora_obj = datetime.strptime(value, "%d/%m/%Y %H:%M:%S")
        timestamp = fecha_hora_obj.timestamp()
        data_X.at[index, 'Timestamp'] = timestamp


# Dividimos los datos del DF:

print("Voy a dividir los datos del dataset")

data_X_train, data_X_test = train_test_split(data_X, test_size=0.2, random_state=42)
data_Y_train, data_y_test = train_test_split(data_y, test_size=0.2, random_state=42)

# Entrenar el modelo
print("Entrenando el modelo")
clf = tree.DecisionTreeClassifier()
print("Creando subset")
subset_X_train = data_X_train.iloc[:, :25]
subset_X_test = data_X_test.iloc[:, :25]


#Train with decision tree
print("Vamos a entrenar el modelo")
clf = clf.fit(subset_X_train, data_Y_train)

#Predict
print("Vamos a predecir")
prediction = clf.predict(subset_X_test)

accuracy = accuracy_score(data_y_test, prediction)
print("Precisión: ", accuracy)

print("Creando el árbol")
dot_data = tree.export_graphviz(clf, out_file=None, filled=True, rounded=True,special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('test.gv', view=True).replace('\\', '/')
graph.format = 'png'

