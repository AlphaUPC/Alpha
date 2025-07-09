import streamlit as st

st.set_page_config(page_title="Predicción Cripto", layout="centered")

st.title("🔮 Predicción de Criptomonedas")
st.subheader("Proyecto de IA aplicado a series temporales")

st.markdown("""
Bienvenido a esta app donde podrás predecir el valor de diferentes criptomonedas usando modelos avanzados de Machine Learning.

---

📌 **Cómo usar la app**:
1. Dirígete a la sección "Predicción de Monedas"
2. Sube tu archivo `.csv` con **60 días recientes** de la moneda seleccionada.
3. Selecciona la moneda y el modelo.
4. Visualiza la predicción para los próximos 7 días.

⚠️ Para que la predicción funcione, tu archivo debe tener como mínimo:
- Columnas: `timestamp`, `close`, `volume`, `marketCap`
- Datos consecutivos (sin huecos)
""")
