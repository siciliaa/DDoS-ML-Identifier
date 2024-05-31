import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import ipaddress
from matplotlib.backends.backend_pdf import PdfPages
import csv

# Cargar el dataset
path = '../final_dataset.csv/final_dataset.csv'
df = pd.read_csv(path)
print("Leído el dataset")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Eliminar columnas innecesarias
columnas_eliminar = ['Label', 'Flow ID']
data_X = df.drop(columnas_eliminar, axis=1)

# Convertir direcciones IP a números enteros
print("Vamos a convertir las IPs a números enteros")
data_X['Src IP'] = data_X['Src IP'].apply(lambda x: int(ipaddress.ip_address(x)))
data_X['Dst IP'] = data_X['Dst IP'].apply(lambda x: int(ipaddress.ip_address(x)))

# Tratar valores infinitos y NaN
print("Se tratan los valores infinitos y los NaN")
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

# Crear una figura con subplots para la tabla y el heatmap
fig, ax = plt.subplots(2, 1, figsize=(36, 48))

# Mostrar el heatmap de la matriz de correlación en el primer subplot
print("Generando el heatmap")
correlation_matrix = data_X.corr()
half_correlation_matrix = np.tril(correlation_matrix, k=-1)  # Obtener la mitad inferior de la matriz de correlación
sns.heatmap(half_correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax[0])
ax[0].set_title('Heatmap de la Mitad Inferior de la Matriz de Correlación')

# Guardar la mitad de la matriz de correlación en un archivo CSV
corr_path = 'half_correlation_matrix.csv'
pd.DataFrame(half_correlation_matrix).to_csv(corr_path, index=True)
print(f"Mitad de la matriz de correlación guardada en {corr_path}")

# Encontrar correlaciones mayores o iguales a 0.5 y menores o iguales a -0.5
resultados_mayor = []
resultados_menor = []

for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        column1 = correlation_matrix.columns[i]
        column2 = correlation_matrix.columns[j]
        correlation = correlation_matrix.iloc[i, j]
        if correlation >= 0.5:
            resultados_mayor.append([column1, column2, correlation])
        if correlation <= -0.5:
            resultados_menor.append([column1, column2, correlation])

# Convertir los resultados a dataframes de pandas
df_resultados_mayor = pd.DataFrame(resultados_mayor, columns=['Columna1', 'Columna2', "Correlacion"])
df_resultados_menor = pd.DataFrame(resultados_menor, columns=['Columna1', 'Columna2', "Correlacion"])

# Guardar los resultados en archivos CSV
output_mayor = 'correlacion_mayor.csv'
output_menor = 'correlacion_menor.csv'
df_resultados_mayor.to_csv(output_mayor, index=False)
df_resultados_menor.to_csv(output_menor, index=False)

print(f"Correlaciones mayores o iguales a 0.5 guardadas en {output_mayor}")
print(f"Correlaciones menores o iguales a -0.5 guardadas en {output_menor}")

# Guardar la figura en un archivo PDF
pdf_path = 'output.pdf'
with PdfPages(pdf_path) as pdf:
    pdf.savefig(fig)

print(f"Figura guardada en {pdf_path}")

# Mostrar la figura (opcional, para verificar visualmente durante el desarrollo)
plt.show()
