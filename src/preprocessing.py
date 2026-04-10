import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_hotels(df):
    df = df.copy()

    # Limpiar nombres
    df.columns = df.columns.str.strip()

    # Eliminar columnas irrelevantes
    df = df.drop(columns=["url", "phone", "details", "address", "name"], errors="ignore")

    # Convertir categóricas
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category").cat.codes

    # 🚨 SOLUCIÓN CLAVE: eliminar NaN
    df = df.replace([-1], 0)  # por si category generó -1
    df = df.fillna(0)

    # Escalar
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    return df_scaled, df