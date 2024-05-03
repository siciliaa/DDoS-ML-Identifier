import pandas as pd


def elementos_unicos_columa(PATH, nombre_archivo):
    df = pd.read_csv(PATH)
    resultado = pd.DataFrame(columns=["Columna", "Numero_valores_unicos", "Numero_elementos_no_unicos", "Total_Elementos"])
    for columna in df.columns:
        valores_unicos = set()
        valores_no_unicos = []
        for valor in df[columna]:
            valores_unicos.add(valor)

        filas_totales = 12794627
        num_valores_unicos = len(valores_unicos)
        num_valores_no_unicos = filas_totales - num_valores_unicos
        total_elementos = num_valores_no_unicos + num_valores_unicos
        resultado = resultado._append({"Columna": columna,
                                        "Numero_valores_unicos": num_valores_unicos,
                                        "Numero_elementos_no_unicos": num_valores_no_unicos,
                                        "Total_Elementos": total_elementos},
                                       ignore_index=True)

    resultado.to_csv(nombre_archivo, index=False)


PATH = '../final_dataset.csv/final_dataset.csv'

elementos_unicos_columa(PATH, "resultados.csv")
