def calcular_media(numeros):
    if not numeros:
        return 0
    suma = sum(numeros)
    cantidad = len(numeros)
    media = suma / cantidad
    return media

# Pedir al usuario que ingrese una serie de números separados por espacios
entrada = input("Introduce una serie de números separados por espacios: ")

# Convertir la entrada en una lista de números
numeros = list(map(float, entrada.split()))

# Calcular la media
media = calcular_media(numeros)

# Mostrar el resultado
print(f"La media de los números es {media}")
