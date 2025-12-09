import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN Y ESTILO (ESTABLE)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="StatSuite Pro", page_icon="💠", layout="wide")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
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
            color: #666;
            transition: all 0.3s;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #222;
            color: white;
            border-bottom: 2px solid white;
        }

        /* INPUTS */
        .stNumberInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
            background-color: #111 !important;
            color: white !important;
            border: 1px solid #333 !important;
            border-radius: 8px;
        }

        /* TARJETAS DE RESULTADOS (Blancas para contraste) */
        .result-card {
            background-color: #ffffff;
            color: #1f2937;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
            margin-bottom: 15px;
            border-top: 5px solid;
        }
        .card-label { font-size: 0.8rem; text-transform: uppercase; font-weight: 700; color: #6b7280; margin-bottom: 5px; }
        .card-value { font-size: 1.8rem; font-weight: 800; color: #111; }

        /* COLORES TEMÁTICOS */
        .neon-purple { border-color: #a855f7; }
        .neon-blue { border-color: #3b82f6; }
        .neon-red { border-color: #ef4444; }
        .neon-green { border-color: #22c55e; }

        /* CAJA DE TEXTO SIMPLE */
        .simple-text {
            background: rgba(255,255,255,0.03);
            border-left: 3px solid #666;
            padding: 15px;
            border-radius: 0 8px 8px 0;
            margin-top: 15px;
            color: #ccc;
        }
        
        /* BOTONES */
        div.stButton > button {
            width: 100%;
            border-radius: 8px;
            font-weight: 600;
            padding: 12px;
            background-color: #222;
            color: white;
            border: 1px solid #444;
        }
        div.stButton > button:hover {
            border-color: #fff;
        }
    </style>
""", unsafe_allow_html=True)

st.title("💠 StatSuite Pro")

# Función de tarjeta
def card(label, value, theme="neon-blue"):
    return f"""
    <div class="result-card {theme}">
        <div class="card-label">{label}</div>
        <div class="card-value">{value}</div>
    </div>
    """

# -----------------------------------------------------------------------------
# ESTRUCTURA DE PESTAÑAS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🟣 Descriptiva (Datos)", 
    "🔵 Inferencia (Inteligente)", 
    "🔴 Comparación (2 Pob)", 
    "🟢 Tamaño Muestra"
])

# =============================================================================
# PESTAÑA 1: DESCRIPTIVA (Clásica y Sencilla)
# =============================================================================
with tab1:
    st.markdown("<h3 style='color:#a855f7'>Estadística Descriptiva</h3>", unsafe_allow_html=True)
    
    col_in, col_out = st.columns([1, 2], gap="large")
    with col_in:
        st.info("Pega tus datos numéricos separados por comas.")
        input_desc = st.text_area("Datos:", height=150, placeholder="Ej: 10, 15, 12, 18, 20")
        btn_calc_desc = st.button("Analizar Datos", key="btn1")

    with col_out:
        if btn_calc_desc:
            if not input_desc.strip():
                st.warning("Escribe algunos números para comenzar.")
            else:
                try:
                    data = np.array([float(x.strip()) for x in input_desc.split(",") if x.strip()])
                    if len(data) > 0:
                        media = np.mean(data)
                        mediana = np.median(data)
                        desv = np.std(data, ddof=1)
                        var = np.var(data, ddof=1)
                        rango = np.max(data) - np.min(data)
                        
                        # Tarjetas
                        c1, c2, c3 = st.columns(3)
                        c1.markdown(card("Promedio (Media)", f"{media:.2f}", "neon-purple"), unsafe_allow_html=True)
                        c2.markdown(card("Centro (Mediana)", f"{mediana:.2f}", "neon-purple"), unsafe_allow_html=True)
                        c3.markdown(card("Total Datos", f"{len(data)}", "neon-purple"), unsafe_allow_html=True)
                        
                        c4, c5, c6 = st.columns(3)
                        c4.markdown(card("Desviación Estándar", f"{desv:.2f}", "neon-purple"), unsafe_allow_html=True)
                        c5.markdown(card("Varianza", f"{var:.2f}", "neon-purple"), unsafe_allow_html=True)
                        c6.markdown(card("Rango", f"{rango:.2f}", "neon-purple"), unsafe_allow_html=True)

                        # Gráfico Integrado
                        fig, ax = plt.subplots(figsize=(10, 3))
                        fig.patch.set_facecolor('#050505')
                        ax.set_facecolor('#111')
                        counts, bins, patches = ax.hist(data, bins='auto', color='#a855f7', edgecolor='black', alpha=0.9)
                        ax.bar_label(patches, fmt='%.0f', color='white', padding=3, fontweight='bold')
                        ax.axvline(media, color='white', linestyle='--', label='Media')
                        ax.axis('off')
                        st.pyplot(fig)
                except:
                    st.error("Revisa que solo haya números y comas.")

# =============================================================================
# PESTAÑA 2: INFERENCIA (LÓGICA AUTOMÁTICA AQUÍ SOLAMENTE)
# =============================================================================
with tab2:
    st.markdown("<h3 style='color:#3b82f6'>Inferencia Estadística</h3>", unsafe_allow_html=True)
    st.markdown("Elige qué tipo de dato tienes y el sistema elegirá la fórmula correcta (Z o T) automáticamente.")

    # Selector simple para limpiar la vista
    tipo_dato = st.radio("Tipo de Dato:", ["Promedio (Media)", "Porcentaje (Proporción)", "Posición (Z)"], horizontal=True)
    
    st.markdown("---")

    # --- LÓGICA AUTOMÁTICA PARA MEDIAS (Z vs T) ---
    if tipo_dato == "Promedio (Media)":
        c1, c2, c3 = st.columns(3)
        media = c1.number_input("Promedio Muestral", 0.0)
        n = c2.number_input("Tamaño de Muestra", 0)
        conf = c3.selectbox("Nivel de Confianza", [0.90, 0.95, 0.99], index=1)
        
        st.markdown("##### Desviación Estándar (Llena solo una)")
        c_sig, c_s = st.columns(2)
        # Aquí está la magia: Detectamos cuál llenó el usuario
        sigma = c_sig.number_input("Poblacional (σ) -> Usa Z", 0.0, help="Si conoces la desviación histórica/poblacional.")
        s = c_s.number_input("Muestral (s) -> Usa T", 0.0, help="Si la desviación viene de la propia muestra.")
        
        if st.button("Calcular Intervalo", key="btn_inf_smart"):
            if n > 1:
                margen = 0
                metodo = ""
                
                # AUTOMATIZACIÓN DE FÓRMULA
                if sigma > 0:
                    se = sigma / np.sqrt(n)
                    z_val = stats.norm.ppf((1 + conf)/2)
                    margen = z_val * se
                    metodo = "Fórmula Z (Sigma conocida)"
                elif s > 0:
                    se = s / np.sqrt(n)
                    # Si n >= 30 usa Z, si n < 30 usa T (Estándar general)
                    if n >= 30:
                        z_val = stats.norm.ppf((1 + conf)/2)
                        margen = z_val * se
                        metodo = "Fórmula Z (Muestra grande)"
                    else:
                        t_val = stats.t.ppf((1 + conf)/2, df=n-1)
                        margen = t_val * se
                        metodo = "Fórmula T-Student (Muestra pequeña)"
                else:
                    st.warning("⚠️ Debes ingresar alguna desviación estándar (Sigma o S).")

                if margen > 0:
                    col1, col2 = st.columns(2)
                    col1.markdown(card("Límite Inferior", f"{media - margen:.4f}", "neon-blue"), unsafe_allow_html=True)
                    col2.markdown(card("Límite Superior", f"{media + margen:.4f}", "neon-blue"), unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="simple-text" style="border-color: #3b82f6;">
                        <strong>Interpretación:</strong> Estamos un {conf*100:.0f}% seguros de que el promedio real está entre esos valores.<br>
                        <em>Sistema: Se detectó y usó {metodo}.</em>
                    </div>
                    """, unsafe_allow_html=True)

    # --- LÓGICA ESTÁNDAR PARA PROPORCIONES ---
    elif tipo_dato == "Porcentaje (Proporción)":
        c1, c2, c3 = st.columns(3)
        prop = c1.number_input("Proporción (0.0 a 1.0)", 0.0, 1.0, 0.0)
        n = c2.number_input("Tamaño Muestra", 0)
        conf = c3.selectbox("Nivel de Confianza", [0.90, 0.95, 0.99], index=1)
        
        if st.button("Calcular Intervalo"):
            if n > 0:
                q = 1 - prop
                se = np.sqrt((prop*q)/n)
                z_val = stats.norm.ppf((1+conf)/2)
                margen = z_val * se
                
                col1, col2 = st.columns(2)
                col1.markdown(card("Mínimo", f"{(prop-margen)*100:.2f}%", "neon-blue"), unsafe_allow_html=True)
                col2.markdown(card("Máximo", f"{(prop+margen)*100:.2f}%", "neon-blue"), unsafe_allow_html=True)
    
    # --- LÓGICA SIMPLE PARA Z ---
    elif tipo_dato == "Posición (Z)":
        c1, c2, c3 = st.columns(3)
        val = c1.number_input("Dato", 0.0)
        mu = c2.number_input("Promedio Población", 0.0)
        sig = c3.number_input("Desviación Población", 1.0)
        
        if st.button("Calcular Z"):
            z = (val - mu) / sig
            st.markdown(card("Puntaje Z", f"{z:.4f}", "neon-blue"), unsafe_allow_html=True)

# =============================================================================
# PESTAÑA 3: COMPARACIÓN (Clásica por Menú)
# =============================================================================
with tab3:
    st.markdown("<h3 style='color:#ef4444'>Comparación de Dos Grupos</h3>", unsafe_allow_html=True)
    
    modo_comp = st.selectbox("Selecciona Análisis:", ["Diferencia de Medias", "Diferencia de Proporciones"])
    st.markdown("---")
    
    c1, c2 = st.columns(2)
    
    if modo_comp == "Diferencia de Medias":
        with c1:
            st.write("Grupo 1")
            m1 = st.number_input("Media 1", 0.0)
            s1 = st.number_input("Desv 1", 0.0)
            n1 = st.number_input("N 1", 0)
        with c2:
            st.write("Grupo 2")
            m2 = st.number_input("Media 2", 0.0)
            s2 = st.number_input("Desv 2", 0.0)
            n2 = st.number_input("N 2", 0)
            
        alpha = st.slider("Alpha", 0.01, 0.10, 0.05)
        
        if st.button("Comparar Medias"):
            if n1 > 1 and n2 > 1:
                se = np.sqrt((s1**2/n1) + (s2**2/n2))
                t_stat = (m1 - m2) / se
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n1+n2-2))
                
                conclusion = "Son Diferentes" if p_val < alpha else "Son Iguales"
                st.markdown(card("Conclusión", conclusion, "neon-red"), unsafe_allow_html=True)
                st.info(f"Valor P: {p_val:.4f}")

    else: # Proporciones
        with c1:
            x1 = st.number_input("Éxitos 1", 0)
            nt1 = st.number_input("Total 1", 0)
        with c2:
            x2 = st.number_input("Éxitos 2", 0)
            nt2 = st.number_input("Total 2", 0)
            
        alpha = st.slider("Alpha", 0.01, 0.10, 0.05)
        
        if st.button("Comparar Porcentajes"):
            if nt1 > 0 and nt2 > 0:
                p1, p2 = x1/nt1, x2/nt2
                pp = (x1+x2)/(nt1+nt2)
                se = np.sqrt(pp*(1-pp)*(1/nt1 + 1/nt2))
                z = (p1 - p2) / se
                p_val = 2 * (1 - stats.norm.cdf(abs(z)))
                
                conclusion = "Son Diferentes" if p_val < alpha else "Son Iguales"
                st.markdown(card("Conclusión", conclusion, "neon-red"), unsafe_allow_html=True)
                st.info(f"Valor P: {p_val:.4f}")

# =============================================================================
# PESTAÑA 4: TAMAÑO DE MUESTRA (Verde)
# =============================================================================
with tab4:
    st.markdown("<h3 style='color:#22c55e'>Calculadora de Muestra</h3>", unsafe_allow_html=True)
    
    target = st.radio("Objetivo:", ["Estimar un Promedio", "Estimar un Porcentaje"])
    
    col_a, col_b = st.columns(2)
    error = col_a.number_input("Error Aceptable (Ej: 0.05)", 0.0, format="%.4f")
    conf = col_b.selectbox("Confianza", [0.90, 0.95, 0.99], index=1, key="size_conf")
    
    sigma = 0
    p_est = 0.5
    
    if target == "Estimar un Promedio":
        sigma = st.number_input("Desviación Estimada", 0.0)
    else:
        p_est = st.number_input("Proporción Estimada (0.5 si no sabes)", 0.0, 1.0, 0.5)
        
    if st.button("Calcular N"):
        z = stats.norm.ppf((1+conf)/2)
        n = 0
        if error > 0:
            if target == "Estimar un Promedio":
                n = (z**2 * sigma**2) / error**2
            else:
                n = (z**2 * p_est * (1-p_est)) / error**2
            
            st.markdown(card("Personas a Encuestar", f"{math.ceil(n)}", "neon-green"), unsafe_allow_html=True)
        else:
            st.warning("El error debe ser mayor a 0.")
