import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib

TIPOS_MODELOS = {
    "modelo_bitcoin.h5": "lstm",
    "modelo_ethereum.h5": "linear_regression", 
    "modelo_polkadot.h5": "random_forest",
    "modelo_oasis.h5": "lstm",
    "modelo_nervos.h5": "linear_regression",
    "modelo_terra_classic.h5": "random_forest",
    "modelo_the_graph.h5": "lstm",
    "modelo_algorand.h5": "lstm",
}

def obtener_tipo_modelo(path):
    nombre_archivo = path.split("/")[-1]
    return TIPOS_MODELOS.get(nombre_archivo, "lstm")

def cargar_modelo(path):
    """Carga el modelo según su tipo"""
    tipo = obtener_tipo_modelo(path)
    
    if tipo == "lstm":
        return load_model(path, compile=False)
    elif tipo in ["random_forest", "linear_regression"]:
        return joblib.load(path)
    else:
        raise ValueError(f"Tipo de modelo no soportado: {tipo}")

def predecir_siguiente_7_dias(df, modelo, lookback):
    df = df[["close", "volume", "marketCap"]].copy()
    
    if len(df) < lookback:
        raise ValueError(f"Se requieren al menos {lookback} días de datos. Actualmente hay {len(df)}.")
    
    tipo_modelo = "lstm" 
    if hasattr(modelo, 'predict') and hasattr(modelo, 'n_estimators'):
        tipo_modelo = "random_forest"
    elif hasattr(modelo, 'coef_'):
        tipo_modelo = "linear_regression"
    
    if tipo_modelo == "lstm":
        return _predecir_lstm(df, modelo, lookback)
    elif tipo_modelo == "random_forest":
        return _predecir_random_forest(df, modelo, lookback)
    elif tipo_modelo == "linear_regression":
        return _predecir_linear_regression(df, modelo, lookback)

def _predecir_lstm(df, modelo, lookback):
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

def _predecir_random_forest(df, modelo, lookback):
    input_data = df[-lookback:].copy()
    predicciones = []
    
    for _ in range(7):
        # Para RF usamos las últimas características directamente
        X_input = input_data[["close", "volume", "marketCap"]].values[-1].reshape(1, -1)
        y_pred = modelo.predict(X_input)[0]
        predicciones.append(y_pred)
        
        # Actualizar datos para siguiente predicción
        new_row = pd.DataFrame([[y_pred, input_data.iloc[-1]["volume"], input_data.iloc[-1]["marketCap"]]],
                               columns=["close", "volume", "marketCap"])
        input_data = pd.concat([input_data.iloc[1:], new_row], ignore_index=True)
    
    return pd.DataFrame({
        "Día": [f"Día {i+1}" for i in range(7)],
        "Predicción": predicciones
    })

def _predecir_linear_regression(df, modelo, lookback):
    input_data = df[-lookback:].copy()
    predicciones = []
    
    for _ in range(7):
        # Para LR usamos las últimas características
        X_input = input_data[["close", "volume", "marketCap"]].values[-1].reshape(1, -1)
        y_pred = modelo.predict(X_input)[0]
        predicciones.append(y_pred)
        
        # Actualizar datos para siguiente predicción
        new_row = pd.DataFrame([[y_pred, input_data.iloc[-1]["volume"], input_data.iloc[-1]["marketCap"]]],
                               columns=["close", "volume", "marketCap"])
        input_data = pd.concat([input_data.iloc[1:], new_row], ignore_index=True)
    
    return pd.DataFrame({
        "Día": [f"Día {i+1}" for i in range(7)],
        "Predicción": predicciones
    })