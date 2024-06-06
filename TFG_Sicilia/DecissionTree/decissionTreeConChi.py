import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
from sklearn import tree
from sklearn.feature_selection import chi2, SelectKBest
import graphviz
import ipaddress
import time
import os

# Leemos el DF
start_time = time.time()
path = r'../final_dataset.csv/final_dataset.csv'
df = pd.read_csv(path)
print("He leído ya el dataset.")
leido = time.time()

tiempo_lectura_dataset = leido - start_time

# Cogemos todas las columnas menos la etiqueta y flowID
columnas_eliminar = ['Label', 'Flow ID', 'Fwd Seg Size Avg', 'Subflow Fwd Byts', 'Bwd Pkt Len Mean', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
                     'TotLen Bwd Pkts', 'Bwd Header Len', 'TotLen Fwd Pkts', 'Idle Mean', 'Subflow Fwd Pkts', 'Pkt Len Mean',
                     'Subflow Bwd Pkts', 'Flow IAT Mean', 'Idle Max', 'Flow Duration', 'Flow IAT Min', 'Pkt Len Max',
                     'Fwd Pkt Len Max', 'Fwd IAT Mean', 'Bwd Pkt Len Max', 'Fwd Pkt Len Mean', 'Fwd Header Len', 'Active Mean',
                     'Flow IAT Max', 'Pkt Size Avg', 'Protocol', 'Bwd IAT Std', 'Bwd Pkt Len Std', 'Fwd Pkt Len Min',
                     'Fwd IAT Max', 'Flow IAT Std', 'Bwd IAT Mean', 'Fwd IAT Tot', 'Bwd Pkt Len Min', 'Active Std',
                     'SYN Flag Cnt', 'Bwd IAT Tot', 'Fwd Pkt Len Std', 'Pkt Len Std', 'Fwd IAT Std', 'Active Max',
                     'RST Flag Cnt', 'Bwd IAT Max', 'Src Port', 'PSH Flag Cnt', 'CWE Flag Count', 'Fwd IAT Min', 'Flow Byts/s',
                     'ACK Flag Cnt', 'Dst Port', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg',
                     'Bwd Pkts/b Avg', 'Fwd URG Flags', 'Bwd URG Flags', 'Bwd Blk Rate Avg']

data_X = df.drop(columnas_eliminar, axis=1)  # Todas las columnas excepto "label"
data_y = df['Label']

print("Vamos a convertir las IPs a números enteros")
data_X['Src IP'] = data_X['Src IP'].apply(lambda x: int(ipaddress.ip_address(x)) if pd.notnull(x) else 0)
data_X['Dst IP'] = data_X['Dst IP'].apply(lambda x: int(ipaddress.ip_address(x)) if pd.notnull(x) else 0)

print("Ahora vamos a cambiar 'ddos' a 1 y cualquier otro caso a 0")
data_y = data_y.apply(lambda x: 1 if x == 'ddos' else 0)

print("Vamos a tratar los NaN y los valores infinitos")

# Se tratan los valores infinitos y los NaN
for i in data_X.columns:
    data_X[i].replace([np.inf, -np.inf], 100000000000, inplace=True)
    data_X[i].fillna(0, inplace=True)
    if data_X[i].dtype == float:
        data_X[i] = data_X[i].apply(lambda x: int(x * 100000))

# Convertir timestamps a segundos desde epoch
print("Se trata timestamp")
for index, value in data_X['Timestamp'].items():
    if "AM" in value or "PM" in value:
        fecha_hora_obj = datetime.strptime(value, "%d/%m/%Y %I:%M:%S %p")
    else:
        fecha_hora_obj = datetime.strptime(value, "%d/%m/%Y %H:%M:%S")
    timestamp = fecha_hora_obj.timestamp()
    data_X.at[index, 'Timestamp'] = timestamp

# Asegurarse de que todos los valores sean no negativos
print("Asegurando que todos los valores sean no negativos")
data_X = data_X.apply(lambda x: x - x.min() if x.min() < 0 else x)

# Selección de características usando Chi-Square
print("Seleccionando características con Chi-Square")
selector = SelectKBest(score_func=chi2, k=20)  # Selecciona las 10 mejores características

data_X_selected = selector.fit_transform(data_X, data_y)

# Obtener los nombres de las características seleccionadas
selected_features = data_X.columns[selector.get_support()]

print(f"Características seleccionadas: {selected_features}")

# Dividimos los datos del DF
print("Voy a dividir los datos del dataset")
data_X_train, data_X_test, data_y_train, data_y_test = train_test_split(data_X_selected, data_y, test_size=0.2, random_state=42)

# Entrenar el modelo
print("Entrenando el modelo")
entrenar = time.time()
clf = tree.DecisionTreeClassifier(random_state=42, max_depth=20)

# Entrenar el modelo
print("Vamos a entrenar el modelo")
clf = clf.fit(data_X_train, data_y_train)

# Predecir
predecir = time.time()
print("Vamos a predecir")
prediction = clf.predict(data_X_test)
fin_predecir = time.time()
tiempo_en_predecir = fin_predecir - predecir
tiempo_en_entrenar_y_predecir = fin_predecir - entrenar
accuracy = accuracy_score(data_y_test, prediction)

precision = precision_score(data_y_test, prediction)
recall = recall_score(data_y_test, prediction)
f1 = f1_score(data_y_test, prediction)
total_time = fin_predecir - start_time

print("********************************")
print(f"[*]Tiempo total de ejecución: {total_time}\n[*]Tiempo tardado en leer el dataset: {tiempo_lectura_dataset}\n[*]Tiempo en entrenar y predecir: {tiempo_en_entrenar_y_predecir}\n[*]Tiempo en predecir: {tiempo_en_predecir}")
print("********************************")

print("[*]Accuracy:", accuracy)
print("[*]Precision:", precision)
print("[*]Recall (Sensibilidad):", recall)
print("[*]F1 Score:", f1)

# Crear el árbol y guardarlo
print("Creando el árbol")
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=selected_features, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)

# Guardar el árbol en formato PDF
output_dir = 'trees'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
graph.render(os.path.join(output_dir, 'decision_tree_20'), format='pdf', view=True)
