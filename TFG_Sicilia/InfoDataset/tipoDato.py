import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


path = '../final_dataset.csv/final_dataset.csv'
df = pd.read_csv(path)

pd.set_option('display.max_columns', None)

dtypes = df.dtypes


# Contar los tipos de datos
dtype_counts = df.dtypes.value_counts()

# Imprimir los resultados
print("Contador de tipos de datos en el DataFrame:")
print(dtype_counts)

# Opcionalmente, puedes imprimir solo los tipos de datos específicos
num_ints = df.select_dtypes(include='int64').shape[1]
num_floats = df.select_dtypes(include='float64').shape[1]
num_objects = df.select_dtypes(include='object').shape[1]

labels = ['int', 'float', 'string']
sizes = [num_ints, num_floats, num_objects]
colors = ['#333333','#707070','#B7B6B6']
explode = (0.1, 0.1, 0.1)  # "explode" la primera porción

# Crear el gráfico de anillo
plt.figure(figsize=(8, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=140, wedgeprops=dict(width=0.3))

with PdfPages('grafico_tipos_de_datos.pdf') as pdf:
    pdf.savefig()  # Guarda la figura actual en el archivo PDF
    plt.close()
