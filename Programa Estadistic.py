import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="StatSuite Ultra",
    page_icon="🔮",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. ESTILOS CSS AVANZADOS (NEGRO, NEÓN Y BLANCO)
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
        /* FUENTE E IMPORTACIONES */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

        /* RESET GENERAL */
        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
            color: #e5e7eb;
        }

        /* FONDO PRINCIPAL NEGRO PROFUNDO CON TOQUE MORADO */
        .stApp {
            background-color: #050505;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(124, 58, 237, 0.15) 0%, transparent 20%),
                radial-gradient(circle at 90% 80%, rgba(59, 130, 246, 0.15) 0%, transparent 20%);
            background-attachment: fixed;
        }

        /* TÍTULOS */
        h1, h2, h3 {
            color: white !important;
            font-weight: 700;
            letter-spacing: -0.5px;
        }

        /* PESTAÑAS (TABS) PERSONALIZADAS */
        .stTabs [data-baseweb="tab-list"] {
            gap: 15px;
            background-color: rgba(255, 255, 255, 0.03);
            padding: 15px;
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .stTabs [data-baseweb="tab"] {
            height: 50px;
            border-radius: 12px;
            font-weight: 600;
            color: #9ca3af;
            border: none;
            transition: all 0.3s ease;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: white;
            background-color: rgba(255, 255, 255, 0.05);
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, #7c3aed 0%, #3b82f6 100%);
            color: white;
            box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4);
        }

        /* INPUTS (TEXT AREA, NUMBER INPUT) */
        .stTextArea textarea, .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
            background-color: #111 !important;
            color: white !important;
            border: 1px solid #333 !important;
            border-radius: 12px;
        }
        
        /* BOTONES PRINCIPALES */
        div.stButton > button {
            background: linear-gradient(90deg, #7c3aed 0%, #6d28d9 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 12px;
            font-weight: 600;
            width: 100%;
            transition: transform 0.2s, box-shadow 0.2s;
            box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3);
        }
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(124, 58, 237, 0.5);
            color: white;
        }

        /* TARJETAS BLANCAS DE RESULTADOS (White Cards) */
        .result-card {
            background-color: #ffffff;
            color: #1f2937; /* Texto oscuro para contraste */
            padding: 20px;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            margin-bottom: 15px;
            border-top: 4px solid #7c3aed; /* Borde superior de color */
            transition: transform 0.2s;
        }
        .result-card:hover {
            transform: scale(1.02);
        }
        .card-label {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #6b7280;
            margin-bottom: 5px;
            font-weight: 600;
        }
        .card-value {
            font-size: 1.8rem;
            font-weight: 800;
            color: #111827;
        }
        .card-sub {
            font-size: 0.75rem;
            color: #9ca3af;
        }

        /* CAJA DE INTERPRETACIÓN */
        .interpretation {
            background: rgba(255, 255, 255, 0.05);
            border-left: 4px solid #3b82f6;
            padding: 20px;
            border-radius: 0 12px 12px 0;
            margin-top: 20px;
            color: #d1d5db;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. LÓGICA Y ESTRUCTURA
# -----------------------------------------------------------------------------

st.title("🔮 StatSuite Ultra")
st.markdown("Plataforma de análisis estadístico integral.")

# Pestañas Principales
tab1, tab2, tab3 = st.tabs([
    "📊 Estadística Descriptiva", 
    "🧮 Inferencia (1 Población)", 
    "⚖️ Comparación (2 Poblaciones)"
])

# =============================================================================
# PESTAÑA 1: DESCRIPTIVA (CON GRÁFICO INTEGRADO)
# =============================================================================
with tab1:
    # Contenedor con fondo sutilmente diferente (CSS via st.container no es directo, usamos layout)
    col_input, col_res = st.columns([1, 2], gap="large")

    with col_input:
        st.subheader("Entrada de Datos")
        st.markdown("<small style='color:#888'>Ingrese números separados por comas:</small>", unsafe_allow_html=True)
        input_desc = st.text_area("Datos:", "10, 12, 15, 14, 13, 16, 10, 12, 25, 30", height=150)
        
        calc_btn = st.button("Analizar Datos", key="btn_desc")

    with col_res:
        if calc_btn:
            try:
                data = [float(x.strip()) for x in input_desc.split(",") if x.strip()]
                arr = np.array(data)
                
                # Cálculos
                media = np.mean(arr)
                mediana = np.median(arr)
                desv = np.std(arr, ddof=1)
                varianza = np.var(arr, ddof=1)
                n = len(arr)
                rango = np.max(arr) - np.min(arr)

                st.subheader("Resultados")
                
                # Función para tarjetas blancas
                def white_card(label, value, sub=""):
                    return f"""
                    <div class="result-card">
                        <div class="card-label">{label}</div>
                        <div class="card-value">{value}</div>
                        <div class="card-sub">{sub}</div>
                    </div>
                    """

                # Grid de Resultados
                c1, c2, c3 = st.columns(3)
                with c1: st.markdown(white_card("Media", f"{media:.2f}", "Promedio"), unsafe_allow_html=True)
                with c2: st.markdown(white_card("Mediana", f"{mediana:.2f}", "Centro"), unsafe_allow_html=True)
                with c3: st.markdown(white_card("Muestra (n)", f"{n}", "Datos"), unsafe_allow_html=True)

                c4, c5, c6 = st.columns(3)
                with c4: st.markdown(white_card("Desviación Std", f"{desv:.2f}", "Dispersión"), unsafe_allow_html=True)
                with c5: st.markdown(white_card("Varianza", f"{varianza:.2f}", "S²"), unsafe_allow_html=True)
                with c6: st.markdown(white_card("Rango", f"{rango:.2f}", "Max - Min"), unsafe_allow_html=True)

                # --- GRÁFICO INTEGRADO AQUÍ ---
                st.markdown("### 📈 Visualización de Distribución")
                fig, ax = plt.subplots(figsize=(10, 3))
                # Fondo negro para el plot para que combine con la app
                fig.patch.set_facecolor('#050505')
                ax.set_facecolor('#111')
                
                # Histograma
                counts, bins, patches = ax.hist(arr, bins='auto', color='#7c3aed', alpha=0.7, rwidth=0.9, edgecolor='black')
                
                # Líneas de referencia
                ax.axvline(media, color='#3b82f6', linestyle='--', label=f'Media: {media:.1f}')
                ax.axvline(mediana, color='#ffffff', linestyle=':', label=f'Mediana: {mediana:.1f}')
                
                ax.legend(facecolor='#222', edgecolor='#444', labelcolor='white')
                ax.tick_params(colors='white')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#444')
                
                st.pyplot(fig)
                
            except ValueError:
                st.error("Error: Asegúrate de usar solo números y comas.")

# =============================================================================
# PESTAÑA 2: INFERENCIA (1 POBLACIÓN)
# =============================================================================
with tab2:
    st.subheader("Cálculos Inferenciales")
    
    opcion = st.selectbox("Seleccione herramienta:", [
        "Intervalo de Confianza (Media)",
        "Intervalo de Confianza (Proporción)",
        "Puntaje Z (Score)",
        "Tamaño de Muestra"
    ])
    
    st.write("---")

    # --- LÓGICA UNIFICADA ---
    if opcion == "Intervalo de Confianza (Media)":
        c1, c2, c3, c4 = st.columns(4)
        x_bar = c1.number_input("Media (x̄)", value=50.0)
        s = c2.number_input("Desviación (s)", value=5.0)
        n = c3.number_input("N", value=30)
        conf = c4.slider("Confianza", 0.80, 0.99, 0.95)

        if st.button("Calcular Intervalo"):
            se = s / np.sqrt(n)
            dist = stats.t if n < 30 else stats.norm
            crit = dist.ppf((1 + conf)/2, df=n-1) if n < 30 else dist.ppf((1 + conf)/2)
            margin = crit * se
            
            # Tarjeta blanca para el resultado
            st.markdown(f"""
            <div class="result-card" style="border-top-color: #3b82f6;">
                <div class="card-label">Intervalo de Confianza ({conf*100:.0f}%)</div>
                <div class="card-value">[{x_bar - margin:.3f}, {x_bar + margin:.3f}]</div>
                <div class="card-sub">Margen de error: ±{margin:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Gráfico de la campana integrado
            fig, ax = plt.subplots(figsize=(8, 2))
            fig.patch.set_facecolor('#050505')
            ax.set_facecolor('#050505')
            x = np.linspace(x_bar - 4*se, x_bar + 4*se, 100)
            y = stats.norm.pdf(x, x_bar, se)
            ax.plot(x, y, color='white')
            ax.fill_between(x, y, where=((x > x_bar-margin) & (x < x_bar+margin)), color='#3b82f6', alpha=0.5)
            ax.axis('off')
            st.pyplot(fig)

    elif opcion == "Puntaje Z (Score)":
        c1, c2, c3 = st.columns(3)
        val = c1.number_input("Valor X", 0.0)
        mu = c2.number_input("Media µ", 0.0)
        sigma = c3.number_input("Sigma σ", 1.0)
        
        if st.button("Calcular Z"):
            z = (val - mu) / sigma
            st.markdown(f"""
            <div class="result-card" style="max-width: 400px; margin: 0 auto;">
                <div class="card-label">Puntaje Z</div>
                <div class="card-value">{z:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# PESTAÑA 3: COMPARACIÓN (2 POBLACIONES)
# =============================================================================
with tab3:
    st.subheader("Prueba de Hipótesis (Comparativa)")
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("##### Grupo A")
        m1 = st.number_input("Media A", 50.0)
        s1 = st.number_input("Desv A", 5.0)
        n1 = st.number_input("N A", 30)
    with col_g2:
        st.markdown("##### Grupo B")
        m2 = st.number_input("Media B", 48.0)
        s2 = st.number_input("Desv B", 5.0)
        n2 = st.number_input("N B", 30)
    
    alpha = st.slider("Alpha (Nivel de Significancia)", 0.01, 0.10, 0.05)
    
    if st.button("Ejecutar Prueba T"):
        se = np.sqrt((s1**2/n1) + (s2**2/n2))
        t_stat = (m1 - m2) / se
        # Grados de libertad simplificados
        df = n1 + n2 - 2
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        c_res1, c_res2 = st.columns(2)
        
        with c_res1:
            st.markdown(f"""
            <div class="result-card">
                <div class="card-label">Estadístico T</div>
                <div class="card-value">{t_stat:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c_res2:
            color_res = "#ef4444" if p_val < alpha else "#22c55e" # Rojo si rechaza, Verde si no
            decision = "Rechazar H0 (Diferencia Real)" if p_val < alpha else "No Rechazar H0 (Iguales)"
            
            st.markdown(f"""
            <div class="result-card" style="border-top-color: {color_res};">
                <div class="card-label">Valor P</div>
                <div class="card-value">{p_val:.4f}</div>
                <div class="card-sub" style="color: {color_res}; font-weight:bold;">{decision}</div>
            </div>
            """, unsafe_allow_html=True)
