import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from subprocess import call
from datetime import datetime
import numpy as np
import ipaddress
import os

# Leemos el DF
path = r'../final_dataset.csv/final_dataset.csv'
df = pd.read_csv(path)
print("He leído ya el dataset.")

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

# Dividimos los datos del DF
print("Voy a dividir los datos del dataset")
data_X_train, data_X_test, data_y_train, data_y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=42)

# Entrenar el modelo
clf = RandomForestClassifier(max_depth=24, random_state=42, n_estimators=10)

print("Entrenar modelo")
clf = clf.fit(data_X_train, data_y_train)

print("Vamos a predecir")
# Predict
prediction = clf.predict(data_X_test)

accuracy = accuracy_score(data_y_test, prediction)
print("Precisión: ", accuracy)

# Crear directorio para almacenar los árboles si no existe
if not os.path.exists('trees'):
    os.makedirs('trees')

# Exportar y guardar todos los árboles en imágenes PNG
feature_names = data_X.columns
for i, estimator in enumerate(clf.estimators_):
    export_graphviz(estimator, out_file=f'trees/tree_{i}.dot',
                    feature_names=feature_names,
                    rounded=True, proportion=False,
                    precision=2, filled=True)
    call(['dot', '-Tpng', f'trees/tree_{i}.dot', '-o', f'trees/tree_{i}.png', '-Gdpi=600'])
    print(f'Árbol {i} guardado como trees/tree_{i}.png')
