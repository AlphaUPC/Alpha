import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

def cargar_modelo(path):
    return load_model(path, compile=False)

def predecir_siguiente_7_dias(df, modelo, lookback):
    df = df[["close", "volume", "marketCap"]].copy()

    if len(df) < lookback:
        raise ValueError(f"Se requieren al menos {lookback} días de datos. Actualmente hay {len(df)}.")

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_raw = df[["close", "volume", "marketCap"]].values
    y_raw = df["close"].values.reshape(-1, 1)

    scaler_X.fit(X_raw)
    scaler_y.fit(y_raw)

    input_seq = df[-lookback:].copy()
    predicciones = []

    for _ in range(7):
        X_input_scaled = scaler_X.transform(input_seq)
        X_input_reshaped = X_input_scaled.reshape(1, lookback, 3)

        y_scaled_pred = modelo.predict(X_input_reshaped, verbose=0)[0][0]
        y_real = scaler_y.inverse_transform([[y_scaled_pred]])[0][0]
        predicciones.append(y_real)

        new_row = pd.DataFrame([[y_real, input_seq.iloc[-1]["volume"], input_seq.iloc[-1]["marketCap"]]],
                               columns=["close", "volume", "marketCap"])
        input_seq = pd.concat([input_seq.iloc[1:], new_row], ignore_index=True)

    return pd.DataFrame({
        "Día": [f"Día {i+1}" for i in range(7)],
        "Predicción": predicciones
    })
