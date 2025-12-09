import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# CONFIGURACIÓN Y ESTILO MINIMALISTA (FLAT DARK)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Tendencia Central", layout="centered")

st.markdown("""
    <style>
    /* Reset y Fuente */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        color: #e0e0e0;
        background-color: #050505;
    }
    
    header, footer {visibility: hidden;}
    
    /* Fondo Principal */
    .stApp {
        background-color: #050505;
    }

    /* Botones */
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: 500;
        transition: background-color 0.2s;
        width: 100%;
        margin-top: 10px;
    }
    .stButton > button:hover {
        background-color: #2563eb;
    }

    /* Inputs */
    .stTextArea textarea {
        background-color: #111;
        border: 1px solid #333;
        color: white;
        border-radius: 4px;
    }
    
    /* Tarjetas de Resultados */
    div[data-testid="stMetric"] {
        background-color: #0a0a0a;
        border: 1px solid #222;
        padding: 15px;
        border-radius: 6px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetricLabel"] {
        color: #888;
        font-size: 0.9rem;
    }
    div[data-testid="stMetricValue"] {
        color: #fff;
        font-size: 1.8rem;
        font-weight: 600;
    }

    /* Caja de Interpretación */
    .interpretation {
        margin-top: 25px;
        padding: 20px;
        border-left: 4px solid #3b82f6;
        background-color: #0a101f;
        color: #ccc;
        font-size: 0.95rem;
        line-height: 1.5;
        border-radius: 0 6px 6px 0;
    }
    
    h1 { color: white !important; font-weight: 300; text-align: center; margin-bottom: 30px; }
    h3 { color: #3b82f6 !important; font-weight: 400; margin-top: 20px; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# LÓGICA DE LA APLICACIÓN
# -----------------------------------------------------------------------------

st.title("Medidas de Tendencia Central")

# --- ENTRADA DE DATOS ---
st.write("Ingrese su conjunto de datos numéricos:")
data_input = st.text_area("", "10, 12, 15, 14, 13, 16, 10, 12, 12, 14", height=100, placeholder="Ej: 5, 10, 15...")

if st.button("Calcular Indicadores"):
    try:
        # 1. Procesamiento
        raw_data = [float(x.strip()) for x in data_input.split(",") if x.strip()]
        data = np.array(raw_data)
        n = len(data)
        
        if n == 0:
            st.warning("El conjunto de datos está vacío.")
        else:
            # 2. Cálculos Matemáticos
            media = np.mean(data)
            mediana = np.median(data)
            
            # Cálculo de Moda (puede haber múltiples)
            mode_result = stats.mode(data, keepdims=True)
            moda_val = mode_result.mode[0]
            count_moda = mode_result.count[0]
            
            # 3. Visualización de Resultados (Tarjetas)
            st.write("---")
            col1, col2, col3 = st.columns(3)
            
            col1.metric("Media (Promedio)", f"{media:.2f}")
            col2.metric("Mediana (Centro)", f"{mediana:.2f}")
            col3.metric("Moda (Más frecuente)", f"{moda_val:.2f}")

            # 4. Interpretación Inteligente
            interpretacion_txt = f"""
            <strong>Análisis de los Datos:</strong><br><br>
            
            • <strong>La Media ({media:.2f}):</strong> Es el promedio aritmético. Si estuviéramos repartiendo el valor total equitativamente entre los {n} elementos, cada uno tendría {media:.2f}.<br>
            
            • <strong>La Mediana ({mediana:.2f}):</strong> Divide a los datos exactamente por la mitad. El 50% de los datos es menor o igual a {mediana:.2f}.<br>
            
            • <strong>La Moda ({moda_val:.2f}):</strong> Es el valor que más se repite (aparece {count_moda} veces).
            """
            
            # Comparación Media vs Mediana para sesgo
            if abs(media - mediana) < 0.1:
                interpretacion_txt += "<br><br>👉 <strong>Simetría:</strong> La media y la mediana son casi iguales, lo que sugiere una distribución simétrica (normal)."
            elif media > mediana:
                interpretacion_txt += "<br><br>👉 <strong>Sesgo a la Derecha:</strong> La media es mayor que la mediana, indicando que hay valores muy altos (atípicos) jalando el promedio hacia arriba."
            else:
                interpretacion_txt += "<br><br>👉 <strong>Sesgo a la Izquierda:</strong> La media es menor que la mediana, indicando valores muy bajos jalando el promedio hacia abajo."

            st.markdown(f"<div class='interpretation'>{interpretacion_txt}</div>", unsafe_allow_html=True)
            
            # 5. Gráfico de Referencia (Histograma simple)
            st.write("### Visualización")
            fig, ax = plt.subplots(figsize=(8, 3))
            fig.patch.set_facecolor('#050505')
            ax.set_facecolor('#050505')
            
            # Histograma
            ax.hist(data, bins='auto', color='#1e293b', edgecolor='#333', alpha=0.8)
            
            # Líneas verticales
            ax.axvline(media, color='#3b82f6', linestyle='--', linewidth=2, label=f'Media: {media:.1f}')
            ax.axvline(mediana, color='#10b981', linestyle='-', linewidth=2, label=f'Mediana: {mediana:.1f}')
            
            ax.legend(facecolor='#111', edgecolor='#333', labelcolor='white')
            ax.tick_params(colors='#888')
            ax.spines['bottom'].set_color('#333')
            ax.spines['left'].set_color('#333')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            st.pyplot(fig)

    except ValueError:
        st.error("Error: Asegúrese de ingresar solo números separados por comas.")
