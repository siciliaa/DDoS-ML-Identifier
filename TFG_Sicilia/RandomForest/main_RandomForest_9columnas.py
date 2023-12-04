import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from subprocess import call
from datetime import datetime


# Leemos el DF:

path = r'../final_dataset.csv/final_dataset.csv'
df = pd.read_csv(path)
print("He leído ya el dataset.")


# Cogemos todas las columnas menos la etiqueta y flowID:
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
for index, value in data_X['Timestamp'].items():
    if pd.isna(data_X.at[index, 'Timestamp']):
        data_X.at[index, 'Timestamp'] = 0
    else:
        if "AM" in value or "PM" in value:
            fecha_hora_obj = datetime.strptime(value, "%d/%m/%Y %I:%M:%S %p")
            timestamp = fecha_hora_obj.timestamp()
            data_X.at[index, 'Timestamp'] = timestamp
        else:
            fecha_hora_obj = datetime.strptime(value, "%d/%m/%Y %H:%M:%S")
            timestamp = fecha_hora_obj.timestamp()
            data_X.at[index, 'Timestamp'] = timestamp

# Columna 2:

for index, value in data_X['Src Port'].items():
    if pd.isna(data_X.at[index, 'Src Port']):
        data_X.at[index, 'Src Port'] = 0

for index, value in data_X['Flow Duration'].items():
    if pd.isna(data_X.at[index, 'Flow Duration']):
        data_X.at[index, 'Flow Duration'] = 0

for index, value in data_X['Tot Fwd Pkts'].items():
    if pd.isna(data_X.at[index, 'Tot Fwd Pkts']):
        data_X.at[index, 'Tot Fwd Pkts'] = 0

for index, value in data_X['Tot Bwd Pkts'].items():
    if pd.isna(data_X.at[index, 'Tot Bwd Pkts']):
        data_X.at[index, 'Tot Bwd Pkts'] = 0
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


# Entrenar el modelo
clf = RandomForestClassifier(max_depth=2, random_state=0,n_estimators=10)


subset_X_train = data_X_train.iloc[:, :5]
subset_X_test = data_X_test.iloc[:, :5]


#Train with decision tree
clf = clf.fit(subset_X_train, data_Y_train)

#Predict
prediction = clf.predict(subset_X_test)

accuracy = accuracy_score(data_y_test, prediction)
print("Precisión: ", accuracy)

for i in range(len(clf.estimators_)):
    estimator = clf.estimators_[i]
    export_graphviz(estimator,
    out_file='tree.dot', rounded=True, proportion=False,precision=2, filled=True)
call(['dot','-Tpng', 'tree.dot','-o','tree' + str(i) + '.png','-Gdpi=600'])

