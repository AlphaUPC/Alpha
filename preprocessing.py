import pandas as pd

def reemplazar_outliers(df, columnas):
    for col in columnas:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        inf = Q1 - 3 * IQR
        sup = Q3 + 3 * IQR
        df.loc[df[col] < inf, col] = Q1
        df.loc[df[col] > sup, col] = Q3
    return df

def preprocesar_datos(df):
    fechas = ["timeOpen", "timeClose", "timeHigh", "timeLow", "timestamp"]
    for col in fechas:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    columnas_outliers = ["open", "high", "low", "close", "volume", "marketCap"]
    df = reemplazar_outliers(df, columnas_outliers)

    df = df[["timestamp", "close", "volume", "marketCap"]].sort_values("timestamp").dropna().reset_index(drop=True)
    return df
