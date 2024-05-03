import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Supongamos que tienes un DataFrame llamado 'df' con tus datos
# y la columna que quieres verificar se llama 'columna'

# Por ejemplo:
path = 'final_dataset.csv/final_dataset.csv'
df = pd.read_csv(path)

plt.figure(figsize=(12, 6))
#sns.set_style("whitegrid")  # Set seaborn style

distribucion = df['Label'].value_counts(normalize=True) * 100  # Calculate percentages
colors = ['gray', 'gray']  # List of colors for bars

barplot = distribucion.plot(kind='bar', color=colors)

plt.xlabel('Clase', fontsize=14)
plt.ylabel('Porcentaje de los puntos', fontsize=14)
plt.title('Distribución de los datos', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=12)  # Rotate x-axis labels for better readability
plt.yticks(fontsize=12)

# Add percentage labels on top of each bar
for i, value in enumerate(distribucion):
    barplot.text(i, value + 0.5, f'{value:.2f}%', ha='center', fontsize=10, color='black')

plt.tight_layout()

# Exportar la imagen a un archivo PDF
plt.savefig('distribucion_datos.pdf')

# Opcionalmente, puedes especificar la calidad y otros parámetros de guardado:
# plt.savefig('distribucion_datos.pdf', dpi=300, bbox_inches='tight')

plt.show()
