import pandas as pd
import math
path = '../final_dataset.csv/final_dataset.csv'
data = pd.read_csv(path)


rows = data.sample(frac=.50)
path_csv_sampling = 'csv_sampling.csv'
df_rows = pd.DataFrame(rows)

print(0.5*len(data), len(rows))

longitud_data = len(data)
longitud_rows = len(rows)

if longitud_data // 2 == longitud_rows:
    print("Cool")
    print(len(data), len(rows))
    df_rows.to_csv(path_csv_sampling, index=False)
    ddos_count = (df_rows['Label'] == 'ddos').sum()
    bening_count = (df_rows['Label'] == 'Benign').sum()
    print("DDoS: ", ddos_count)
    print("Benign: ", bening_count)


if int(longitud_data * 0.5) == longitud_rows:
    print("\nSegundo IF\n")
    print("***********************************\n")
    path2 = 'csv2.csv'
    print("Cool")
    print(len(data), len(rows))
    df_rows.to_csv(path2, index=False)
    ddos_count = (df_rows['Label'] == 'ddos').sum()
    bening_count = (df_rows['Label'] == 'Benign').sum()
    print("DDoS: ", ddos_count)
    print("Benign: ", bening_count)

if math.ceil(longitud_data * 0.5) == longitud_rows:
    print("\nTercer IF\n")
    print("***********************************\n")
    path2 = 'csv_math_ceil.csv'
    print("Cool")
    print(math.ceil(longitud_data * 0.5), len(rows))
    df_rows.to_csv(path2, index=False)
    ddos_count = (df_rows['Label'] == 'ddos').sum()
    bening_count = (df_rows['Label'] == 'Benign').sum()
    print("DDoS: ", ddos_count)
    print("Benign: ", bening_count)
