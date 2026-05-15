import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from scipy.signal import savgol_filter

# --- CONFIGURACIÓN DE LA PANTALLA ---
st.set_page_config(page_title="Radar SIGINT", layout="wide")
st.title("🛰️ Centro de Mando: Radar de Exoplanetas")

# --- LOGÍSTICA: CARGA DE DATOS Y CEREBRO ---
@st.cache_data
def cargar_datos():
    import zipfile
    with zipfile.ZipFile('exoTest.csv.zip', 'r') as z:
        df = pd.read_csv(z.open('exoTest.csv'))
    return df

@st.cache_resource
def cargar_cerebro():
    import os
    import gdown
    from tensorflow.keras.models import load_model
    
    archivo_modelo = 'modelo_exoplanetas_v4.h5'
    
    # Si el cerebro no está en el servidor de Streamlit, gdown vuela a Drive a por él
    if not os.path.exists(archivo_modelo):
        # ⚠️ ¡ATENCIÓN! Sustituya esto por el ID que copió de su enlace de Drive:
        id_drive = '1xiFhDxGBDAQqqZm_puITBEE0ZzkMKYpT' 
        
        url = f'https://drive.google.com/uc?id={id_drive}'
        try:
            gdown.download(url, archivo_modelo, quiet=False)
        except Exception as e:
            st.error(f"Error en el puente aéreo: No se pudo descargar de Drive. ID: {id_drive}")
            st.stop()
            
    return load_model(archivo_modelo)

try:
    df = cargar_datos()
    modelo = cargar_cerebro()
except Exception as e:
    st.error(f"Error logístico: Asegúrese de tener 'exoTest.csv' y 'modelo_exoplanetas_ciencia.h5' en la misma carpeta. Detalle: {e}")
    st.stop()

# --- PANEL DE CONTROL ---
st.sidebar.header("Panel de Control Táctico")
estrella_idx = st.sidebar.number_input("ID de Estrella (0 al 569)", min_value=0, max_value=len(df)-1, value=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Calibración del Radar")
umbral = st.sidebar.slider("Sensibilidad (Gatillo de Alarma)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# --- PROCESAMIENTO ---
fila = df.iloc[estrella_idx]
etiqueta_real = fila['LABEL'] - 1 
flujo_luz = fila[1:].values.astype(float)

flujo_norm = (flujo_luz - np.mean(flujo_luz)) / (np.std(flujo_luz) + 1e-8)
flujo_suave = savgol_filter(flujo_norm, window_length=15, polyorder=2)

entrada_modelo = flujo_suave.reshape(1, len(flujo_suave), 1)
probabilidad = modelo.predict(entrada_modelo)[0][0]

# --- PANTALLA TÁCTICA ---
st.subheader(f"Telemetría Óptica: Objetivo #{estrella_idx}")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(flujo_norm, label="Señal Bruta", color="gray", alpha=0.5)
ax.plot(flujo_suave, label="Señal Limpia", color="cyan")
ax.set_facecolor('#0e1117') 
fig.patch.set_facecolor('#0e1117')
ax.tick_params(colors='white')
ax.legend(facecolor='#0e1117', labelcolor='white')
st.pyplot(fig)

# --- REPORTE ---
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.write("### Veredicto del Radar:")
    if probabilidad >= umbral:
        st.error(f"🚨 ¡ALERTA! Posible Exoplaneta Detectado")
        st.write(f"**Nivel de Certeza:** {probabilidad*100:.2f}%")
    else:
        st.success(f"✅ Sector Despejado. Sin anomalías.")
        st.write(f"**Nivel de Certeza:** {probabilidad*100:.2f}%")

with col2:
    st.write("### Realidad Confirmada:")
    if etiqueta_real == 1:
        st.warning("🪐 EXOPLANETA REAL EN ESTE SECTOR")
    else:
        st.info("⭐ Estrella Vacía")

# --- NUEVA FUNCIÓN: ESCANEO DE SECTOR COMPLETO ---
st.divider()
st.subheader("🔍 Escaneo Automático de Sector")

if st.button("Iniciar Escaneo de Todas las Estrellas"):
    with st.spinner("Escaneando el firmamento..."):
        # Preparamos todos los datos
        x_total = df.values.astype(float)
        x_total_norm = (x_total - np.mean(x_total, axis=1, keepdims=True)) / (np.std(x_total, axis=1, keepdims=True) + 1e-8)
        x_total_suave = np.apply_along_axis(lambda m: savgol_filter(m, window_length=15, polyorder=2), 1, x_total_norm)
        x_total_final = x_total_suave.reshape(x_total_suave.shape[0], x_total_suave.shape[1], 1)
        
        # Predicción masiva
        predicciones = modelo.predict(x_total_final)
        
        # Filtrar por el umbral que el usuario puso en el sidebar
        hallazgos = []
        for i, prob in enumerate(predicciones):
            if prob[0] >= umbral:
                hallazgos.append({"Fila (ID)": i, "Certeza": f"{prob[0]*100:.2f}%"})
        
        if hallazgos:
            st.success(f"¡Se han detectado {len(hallazgos)} posibles anomalías!")
            st.table(hallazgos) # Muestra la lista de filas sospechosas
        else:
            st.warning("No se han detectado anomalías con el umbral actual.")
