import copy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import csv
from sklearn.metrics import accuracy_score
import random
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

path = r'../final_dataset.csv/final_dataset.csv'
df = pd.read_csv(path)

print("He leído ya los datos del dataset")

data_X = df.drop('Label', axis=1)  # All columns except "label"
data_y = df['Label']

print("Voy a dividir los datos del dataset")
data_X_train, data_X_test = train_test_split(data_X, test_size=0.2, random_state=42)
data_Y_train, data_y_test = train_test_split(data_y, test_size=0.2, random_state=42)

semilla = 42
random.seed(semilla)

print("Creando array de predicción")
data_y_pred = [random.randint(0, 1) for _ in range(len(data_y_test))]
# data_y_pred = [1] * len(data_y_test)

# Change 'ddos' to 1 and any other value to 0:
print("Cambiado 'ddos' a 1 y cualquier otro caso a 0")
data_Y_train = data_Y_train.apply(lambda x: 1 if x == 'ddos' else 0)
data_y_test = data_y_test.apply(lambda x: 1 if x == 'ddos' else 0)


accuracy = accuracy_score(data_y_test, data_y_pred)
print("Precisión: ", accuracy)
f1 = f1_score(data_y_test, data_y_pred, zero_division=1.0)

print("F1_Score: ", f1)

target_names = ['class 0', 'class 1']

print(classification_report(data_y_test, data_y_pred, target_names=target_names, zero_division=1.0))
