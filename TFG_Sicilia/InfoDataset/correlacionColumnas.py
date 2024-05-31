import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages

path = '../final_dataset.csv/final_dataset.csv'
df = pd.read_csv(path)
df = df.head(100)
print("Leído el dataset")

columnas_eliminar = ['Label', 'Flow ID']
data_X = df.drop(columnas_eliminar, axis=1)

print("Vamos a quitarle el punto a las IPs")
for index, value in data_X['Src IP'].items():
    data_X.at[index, 'Src IP'] = value.replace('.', '')

for index, value in data_X['Dst IP'].items():
    data_X.at[index, 'Dst IP'] = value.replace('.', '')

print("Se tratan los valores infinitos y los NaN")

for i in data_X.columns[:25]:
    for index, value in data_X[i].items():
            if pd.isna(data_X.at[index, i]):
                data_X.at[index, i] = 0
            elif value == np.inf:
                data_X.at[index, i] = int(100000000000)
            elif isinstance(value, float):
                data_X.at[index, i] = int(value * 100000)

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


print("Generando el heatmap")
sns.heatmap(data_X.corr(), annot=True, cmap='coolwarm', center=0, ax=axs[0])
axs[0].set_title('Heatmap de la Matriz de Correlación')

# Mostrar la tabla de datos en el segundo subplot
print("Mostrando la tabla de datos")
axs[1].axis('off')  # Ocultar el eje del segundo subplot
table = axs[1].table(cellText=data_X.head(10).values, colLabels=data_X.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
axs[1].set_title('Tabla de Datos (Primeras 10 Filas)')

# Ajustar la disposición de los subplots
plt.tight_layout()

# Guardar la figura en un archivo PDF
pdf_path = 'output.pdf'
with PdfPages(pdf_path) as pdf:
    pdf.savefig(fig)

print(f"Figura guardada en {pdf_path}")

# Mostrar la figura (opcional, para verificar visualmente durante el desarrollo)
plt.show()