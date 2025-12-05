import pandas as pd
from image_fetcher import get_image_url
from time import sleep

# Cargar el dataset
df = pd.read_csv("../data/sample_dataset.csv")


# Selección y limpieza
df = df[["destination", "country", "description", "category", "average_cost", "weather", "activities"]]
df = df.dropna()

# Añadir columna de imágenes si no existe
if 'image_url' not in df.columns:
    df['image_url'] = ''

# Agregar imágenes automáticamente
for i in range(len(df)):
    if df.loc[i, 'image_url'] == '':
        search_term = f"{df.loc[i, 'destination']} {df.loc[i, 'country']}"
        print(f"{i+1}/{len(df)} 🔍 Buscando imagen para: {search_term}")
        df.loc[i, 'image_url'] = get_image_url(search_term)
        sleep(1)

        # Guardar progreso cada 50 filas
        if i % 50 == 0:
            df.to_csv("../processed/cleaned_data.csv", index=False)

print("Proceso completado. Dataset guardado en '../processed/cleaned_data.csv'")


