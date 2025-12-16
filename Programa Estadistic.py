import re
import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN E IMPORTACIONES
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Calculadora de estadistica",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. ESTILOS CSS (TEMA DARK + SIN FLECHAS)
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

        /* RESET GENERAL */
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
            background-color: #050505;
            color: #e5e7eb;
        }
        
        header, footer {visibility: hidden;}
        .stApp { background-color: #050505; }

        /* PESTAÑAS ESTILIZADAS */
        .stTabs [data-baseweb="tab-list"] {
            gap: 15px;
            background-color: #0a0a0a;
            padding: 15px;
            border-radius: 16px;
            border: 1px solid #222;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            border-radius: 8px;
            font-weight: 600;
            color: #888;
            border: none;
            transition: all 0.3s;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #222;
            color: white;
            border-bottom: 2px solid #3b82f6;
        }

        /* Ocultar flechas de los inputs numéricos */
        input[type=number]::-webkit-inner-spin-button, 
        input[type=number]::-webkit-outer-spin-button { 
            -webkit-appearance: none; 
            margin: 0; 
        }

        input[type=number] {
            -moz-appearance: textfield;
        }
        
        /* ESTILO DE LOS CAMPOS */
        .stNumberInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
            background-color: #111 !important;
            color: white !important;
            border: 1px solid #333 !important;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Calculadora de estadistica")

# Función auxiliar para tarjeta en HTML
def card(label, value, sub="", color="border-blue"):
    return f"""
    <div class="result-card {color}">
        <div class="card-label">{label}</div>
        <div class="card-value">{value}</div>
        <div class="card-sub">{sub}</div>
    </div>
    """

# -----------------------------------------------------------------------------
# ESTRUCTURA DE PESTAÑAS
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["🟣 Estadística Descriptiva", "🧪 Opciones Extra"])

# =============================================================================
# TAB 1: Estadística Descriptiva
# =============================================================================
with tab1:
    st.markdown("<h3 style='color:#a855f7'>Análisis de Datos</h3>", unsafe_allow_html=True)
    
    col_in, col_out = st.columns([1, 2], gap="large")
    with col_in:
        st.info("Introduce tus números. Usa PUNTO (.) para decimales.")
        input_desc = st.text_area("Datos Numéricos:", height=150, placeholder="Ej: 10.5, 15, 12.0; 18 20")
        btn_calc_desc = st.button("Analizar Datos", key="btn1")

    with col_out:
        if btn_calc_desc:
            if not input_desc.strip():
                st.warning("⚠️ El campo está vacío.")
            else:
                if re.search(r'\d+,\d+', input_desc):
                    st.error("Por favor usa PUNTO (.) para decimales. Reemplaza '10,5' por '10.5' y vuelve a intentar.")
                else:
                    try:
                        parts = re.split(r'[,\;\s]+', input_desc.strip())
                        tokens = [p for p in parts if p != '']
                        nums = [float(t) for t in tokens]
                        data = np.array(nums, dtype=float)
                        n = data.size
                        media = float(np.mean(data))
                        mediana = float(np.median(data))
                        desv = float(np.std(data, ddof=1)) if n >= 2 else float(np.std(data, ddof=0))
                        rango = float(np.max(data) - np.min(data)) if n > 0 else 0.0

                        # Resultados en tarjetas
                        c1, c2, c3 = st.columns(3)
                        c1.markdown(card("Promedio", f"{media:.2f}", "", "border-purple"), unsafe_allow_html=True)
                        c2.markdown(card("Mediana", f"{mediana:.2f}", "", "border-purple"), unsafe_allow_html=True)
                        c3.markdown(card("Rango", f"{rango:.2f}", "", "border-purple"), unsafe_allow_html=True)
                        
                        # Histograma con número controlado de bins
                        st.write("#### Histograma de Frecuencias")
                        fig, ax = plt.subplots(figsize=(10, 3))
                        fig.patch.set_facecolor('#050505')
                        ax.set_facecolor('#111')

                        # Número dinámico de bins
                        num_bins = max(5, int(np.sqrt(len(data))))
                        counts, bins, patches = ax.hist(data, bins=num_bins, color='#a855f7', edgecolor='black', alpha=0.9)

                        # Etiquetar las barras
                        try:
                            ax.bar_label(patches, fmt='%.0f', color='white', padding=3, fontweight='bold')
                        except Exception as e:
                            st.warning(f"No se pudieron etiquetar las barras: {e}")

                        # Línea de promedio
                        ax.axvline(media, color='white', linestyle='--', label='Promedio')
                        ax.legend(facecolor='#222', labelcolor='white', frameon=False)
                        ax.axis('off')
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error al procesar los datos: {e}")
