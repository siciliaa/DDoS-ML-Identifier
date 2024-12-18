import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import chi2, SelectKBest
from subprocess import call
from datetime import datetime
import time
import numpy as np
import ipaddress
import os

# Leemos el DF
path = r'../final_dataset.csv/final_dataset.csv'
start_time = time.time()
df = pd.read_csv(path)
print("He leído ya el dataset.")
leido = time.time()
tiempo_lectura_dataset = leido - start_time


columnas_eliminar = ['Label', 'Flow ID', 'Fwd Seg Size Avg', 'Subflow Fwd Byts', 'Bwd Pkt Len Mean', 'Tot Fwd Pkts',
                     'Tot Bwd Pkts', 'TotLen Bwd Pkts', 'Bwd Header Len', 'TotLen Fwd Pkts', 'Idle Mean',
                     'Subflow Fwd Pkts', 'Pkt Len Mean', 'Subflow Bwd Pkts', 'Flow IAT Mean', 'Idle Max',
                     'Flow Duration', 'Flow IAT Min', 'Pkt Len Max','Fwd Pkt Len Max', 'Fwd IAT Mean', 'Bwd Pkt Len Max',
                     'Fwd Pkt Len Mean', 'Fwd Header Len', 'Active Mean','Flow IAT Max', 'Pkt Size Avg', 'Protocol',
                     'Bwd IAT Std', 'Bwd Pkt Len Std', 'Fwd Pkt Len Min','Fwd IAT Max', 'Flow IAT Std', 'Bwd IAT Mean',
                     'Fwd IAT Tot', 'Bwd Pkt Len Min', 'Active Std','SYN Flag Cnt', 'Bwd IAT Tot', 'Fwd Pkt Len Std',
                     'Pkt Len Std', 'Fwd IAT Std', 'Active Max','RST Flag Cnt', 'Bwd IAT Max', 'Src Port',
                     'PSH Flag Cnt', 'CWE Flag Count', 'Fwd IAT Min','Flow Byts/s', 'ACK Flag Cnt', 'Dst Port',
                     'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg','Bwd Byts/b Avg','Bwd Pkts/b Avg',
                     'Fwd URG Flags', 'Bwd URG Flags', 'Bwd Blk Rate Avg']


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

# Convertir timestamps a segundos
print("Se trata timestamp")
for index, value in data_X['Timestamp'].items():
    if "AM" in value or "PM" in value:
        fecha_hora_obj = datetime.strptime(value, "%d/%m/%Y %I:%M:%S %p")
    else:
        fecha_hora_obj = datetime.strptime(value, "%d/%m/%Y %H:%M:%S")
    timestamp = fecha_hora_obj.timestamp()
    data_X.at[index, 'Timestamp'] = timestamp

print("Asegurando que todos los valores sean no negativos")
data_X = data_X.apply(lambda x: x - x.min() if x.min() < 0 else x)

# Selección de características usando Chi-Square
print("Seleccionando características más importantes usando Chi-Square")
selector = SelectKBest(score_func=chi2, k=25)
data_X_selected = selector.fit_transform(data_X, data_y)
selected_features = data_X.columns[selector.get_support()]
print(f"Características seleccionadas: {selected_features}")

# Dividimos los datos del DF
print("Voy a dividir los datos del dataset")
data_X_train, data_X_test, data_y_train, data_y_test = train_test_split(data_X_selected, data_y, test_size=0.2,
                                                                        random_state=42)

# Entrenar el modelo
print("Entrenando el modelo con validación cruzada")
kf = KFold(n_splits=7, shuffle=True, random_state=68)
clf = RandomForestClassifier(max_depth=25, random_state=42, n_estimators=25)
cv_scores = cross_val_score(clf, data_X_train, data_y_train, cv=kf, scoring='accuracy')

print('Se obtienen los siguientes coeficientes de determinación:')
print(cv_scores, '\n')
print(f'Max Accuracy: {max(cv_scores)}')
print(f'Min Accuracy: {min(cv_scores)}')
print('Promedio Accuracy: {:.3f}'.format(np.mean(cv_scores)))
print('Desviación Estándar: {:.3f}'.format(np.std(cv_scores)))
print(f'Intervalo de confianza 95%: {np.quantile(cv_scores, [0.025, 0.975])}')

print("Entrenar modelo")
entrenar = time.time()
clf = clf.fit(data_X_train, data_y_train)

print("Vamos a predecir")
# Predict
predecir = time.time()
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
print(
    f"[*]Tiempo total de ejecución: {total_time}\n[*]Tiempo tardado en leer el dataset: {tiempo_lectura_dataset}\n[*]Tiempo en entrenar y predecir: {tiempo_en_entrenar_y_predecir}\n[*]Tiempo en predecir: {tiempo_en_predecir}")
print("********************************")

print("[*]Accuracy:", accuracy)
print("[*]Precision:", precision)
print("[*]Recall (Sensibilidad):", recall)
print("[*]F1 Score:", f1)

# Crear directorio para almacenar los árboles si no existe
output_dir = 'trees_Cross_Validation/trees_crossVal_25'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Exportar y guardar todos los árboles
feature_names = selected_features  # Cambia esto para que coincida con las características seleccionadas
for i, estimator in enumerate(clf.estimators_):
    dot_file = f'{output_dir}/tree_{i}.dot'
    pdf_file = f'{output_dir}/tree_{i}.pdf'

    export_graphviz(estimator, out_file=dot_file,
                    feature_names=feature_names,
                    rounded=True, proportion=False,
                    precision=2, filled=True)


    call(['dot', '-Tpdf', dot_file, '-o', pdf_file])
    print(f'Árbol {i} guardado como {pdf_file}')
