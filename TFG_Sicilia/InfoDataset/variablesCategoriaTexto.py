import pandas as pd


def categoria_texto(PATH):
    df = pd.read_csv(PATH)
    df_export = pd.DataFrame(columns=["Nombre_columna"])
    columnas_texto = df.select_dtypes(include=['object']).columns
    cantidad = len(columnas_texto)
    print("La cantidad de columnas cuya categoría es texto es: ", cantidad)

    if cantidad > 0:
        print("Las columnas de tipo texto son: ")
        for columna in columnas_texto:
            df_export = df_export._append({"Nombre_columna": columna}, ignore_index=True)

    df_export.to_csv("columnas_texto.csv", index=False)


PATH = '../final_dataset.csv/final_dataset.csv'
categoria_texto(PATH)
