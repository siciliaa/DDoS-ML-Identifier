import pandas as pd

# Leer el dataset desde un archivo CSV (puedes cambiar esto a tu fuente de datos)
df = pd.read_csv('final_dataset.csv/final_dataset.csv', usecols=[0])

# Mostrar las primeras filas del DataFrame para confirmar que se ha leído correctamente
print(df.head())

# Obtener el nombre de la primera columna
primer_columna_nombre = df.columns[0]

# Contar los elementos diferentes en la primera columna
num_elementos_diferentes = df[primer_columna_nombre].nunique()

print(f'El número de elementos diferentes en la primera columna ({primer_columna_nombre}) es: {num_elementos_diferentes}')
