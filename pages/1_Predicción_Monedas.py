import streamlit as st
import pandas as pd
from utils import cargar_modelo, predecir_siguiente_7_dias
from preprocessing import preprocesar_datos
import plotly.graph_objects as go

st.title("📈 Predicción de Monedas")

monedas = ["Bitcoin", "Ethereum", "Solana", "Cardano", "BNB", "XRP", "Dogecoin", "Polkadot"]
modelos_disponibles = {
    "Bitcoin": "modelo/modelo_bitcoin.h5"
    # Agrega más rutas de modelos aquí
}

archivo = st.file_uploader("📤 Sube tu archivo CSV (últimos 60 días)", type=["csv"])
moneda = st.selectbox("Selecciona una criptomoneda", monedas)

if archivo and moneda in modelos_disponibles:
    df = pd.read_csv(archivo, sep=";")
    df = preprocesar_datos(df)

    modelo = cargar_modelo(modelos_disponibles[moneda])
    st.success("✅ Archivo cargado correctamente. Modelo listo para usar.")

    if len(df) < 60:
        st.error("❌ Se requieren al menos 60 registros para hacer la predicción.")
    else:
        resultados = predecir_siguiente_7_dias(df, modelo, lookback=60)

        st.subheader("📊 Predicción para los próximos 7 días")
        st.dataframe(resultados)

        # --- PREPARAR GRÁFICA ---
        reales = df["close"].tolist()
        predichos = resultados["Predicción"].tolist()
        todos = reales + predichos

        x_hist = list(range(len(reales)))
        x_pred = [len(reales)-1] + list(range(len(reales), len(reales) + 7))
        y_pred = [reales[-1]] + predichos  # unimos con el último valor real

        fig = go.Figure()

        # Histórico
        fig.add_trace(go.Scatter(
            x=x_hist,
            y=reales,
            mode="lines+markers",
            name="Histórico (Usuario)",
            line=dict(color="blue"),
            marker=dict(size=6),
            hovertemplate='Histórico: %{y:.2f}<extra></extra>'
        ))

        # Predicción
        fig.add_trace(go.Scatter(
            x=x_pred,
            y=y_pred,
            mode="lines+markers",
            name="Predicción (7 días)",
            line=dict(color="red"),
            marker=dict(size=6),
            hovertemplate='Predicción: %{y:.2f}<extra></extra>'
        ))

        fig.update_layout(
            title="Evolución del Precio y Predicción (7 días)",
            xaxis_title="Día",
            yaxis_title="Precio de Cierre",
            yaxis=dict(range=[min(todos) * 0.98, max(todos) * 1.02]),
            hovermode="closest", 
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)
