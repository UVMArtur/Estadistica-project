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
    page_icon="💎",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. ESTILOS CSS (COLORES Y DISEÑO REDONDEADO)
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
            background-color: #050505;
            color: #e5e7eb;
        }

        header {visibility: hidden;}
        .stApp { background-color: #050505; }

        /* INPUTS ESTILIZADOS */
        .stTextArea textarea, .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
            background-color: #111 !important;
            color: white !important;
            border: 1px solid #333 !important;
            border-radius: 12px;
        }

        /* PESTAÑAS */
        .stTabs [data-baseweb="tab-list"] {
            gap: 15px;
            background-color: #0a0a0a;
            padding: 15px;
            border-radius: 20px;
            border: 1px solid #222;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            font-weight: 600;
            color: #888;
            border: none;
            transition: all 0.2s;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #222;
            color: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.5);
        }

        /* TARJETAS DE RESULTADO */
        .result-card {
            background-color: #ffffff;
            color: #1f2937;
            padding: 20px;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
            margin-bottom: 15px;
            transition: transform 0.2s;
        }
        .result-card:hover { transform: translateY(-3px); }
        .card-label { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; color: #6b7280; font-weight: 700; margin-bottom: 5px; }
        .card-value { font-size: 1.8rem; font-weight: 800; color: #111; }
        
        /* BORDES DE COLOR SEGÚN SECCIÓN */
        .btn-purple { border-top: 6px solid #a855f7; }
        .btn-blue { border-top: 6px solid #3b82f6; }
        .btn-red { border-top: 6px solid #ef4444; }
        .btn-green { border-top: 6px solid #10b981; }

        /* BOTONES */
        div.stButton > button {
            background-color: #1f1f1f;
            color: white;
            border: 1px solid #333;
            border-radius: 10px;
            font-weight: 600;
            width: 100%;
            padding: 12px;
        }
        div.stButton > button:hover {
            border-color: #888;
            color: #fff;
        }
        
        /* TEXTO EXPLICATIVO SENCILLO */
        .simple-text {
            font-size: 1.1rem;
            color: #d1d5db;
            background: rgba(255,255,255,0.05);
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid;
            margin-top: 15px;
        }

    </style>
""", unsafe_allow_html=True)

st.title("💎 StatSuite Ultra")

# Función para crear tarjetas
def card(label, value, color_class):
    return f"""
    <div class="result-card {color_class}">
        <div class="card-label">{label}</div>
        <div class="card-value">{value}</div>
    </div>
    """

# -----------------------------------------------------------------------------
# ESTRUCTURA DE PESTAÑAS
# -----------------------------------------------------------------------------
tab_desc, tab_inf, tab_comp, tab_size = st.tabs([
    "🟣 Estadística Descriptiva", 
    "🔵 Inferencia (1 Población)", 
    "🔴 Comparación (2 Poblaciones)",
    "🟢 Calculadora de Muestra"
])

# =============================================================================
# 1. ESTADÍSTICA DESCRIPTIVA (MORADO)
# =============================================================================
with tab_desc:
    st.markdown("<h3 style='color: #a855f7;'>Análisis de Datos</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.write("Pega tus números aquí:")
        input_data = st.text_area("Datos (Separados por coma):", placeholder="Ej: 10, 15.5, 20, 12", height=150)
        calcular_desc = st.button("Analizar Datos", key="btn_desc")

    with col2:
        if calcular_desc:
            if not input_data.strip():
                st.warning("⚠️ Escribe algunos números primero.")
            else:
                try:
                    raw_data = [float(x.strip()) for x in input_data.split(",") if x.strip()]
                    data = np.array(raw_data)
                    n = len(data)

                    if n < 1:
                        st.error("Necesitas al menos 1 número.")
                    else:
                        # Cálculos
                        media = np.mean(data)
                        mediana = np.median(data)
                        desviacion = np.std(data, ddof=1)
                        varianza = np.var(data, ddof=1)
                        rango = np.max(data) - np.min(data)

                        # Tarjetas
                        c1, c2, c3 = st.columns(3)
                        c1.markdown(card("Promedio (Media)", f"{media:.2f}", "btn-purple"), unsafe_allow_html=True)
                        c2.markdown(card("Centro (Mediana)", f"{mediana:.2f}", "btn-purple"), unsafe_allow_html=True)
                        c3.markdown(card("Total de Datos", f"{n}", "btn-purple"), unsafe_allow_html=True)

                        c4, c5, c6 = st.columns(3)
                        c4.markdown(card("Variación (Desv. Std)", f"{desviacion:.2f}", "btn-purple"), unsafe_allow_html=True)
                        c5.markdown(card("Varianza", f"{varianza:.2f}", "btn-purple"), unsafe_allow_html=True)
                        c6.markdown(card("Diferencia Max-Min", f"{rango:.2f}", "btn-purple"), unsafe_allow_html=True)

                        # Interpretación Sencilla
                        st.markdown(f"""
                        <div class="simple-text" style="border-color: #a855f7;">
                            <strong>¿Qué significan estos datos?</strong><br>
                            El valor típico es <b>{media:.2f}</b>. <br>
                            La mayoría de los datos se alejan unos <b>{desviacion:.2f}</b> puntos de ese promedio (hacia arriba o hacia abajo).
                        </div>
                        """, unsafe_allow_html=True)

                        # Gráfico
                        st.write("#### Gráfico de tus datos")
                        fig, ax = plt.subplots(figsize=(10, 3))
                        fig.patch.set_facecolor('#050505')
                        ax.set_facecolor('#111')
                        
                        counts, bins, patches = ax.hist(data, bins='auto', color='#a855f7', edgecolor='black', alpha=0.9)
                        ax.bar_label(patches, fmt='%.0f', color='white', fontsize=11, padding=3, weight='bold') # Dígitos visibles

                        ax.axvline(media, color='white', linestyle='--', linewidth=1, label='Promedio')
                        ax.legend(facecolor='#222', labelcolor='white', frameon=False)
                        ax.axis('off') # Limpiar ejes feos
                        st.pyplot(fig)

                except ValueError:
                    st.error("Solo se permiten números y comas.")

# =============================================================================
# 2. INFERENCIA (AZUL)
# =============================================================================
with tab_inf:
    st.markdown("<h3 style='color: #3b82f6;'>Estimaciones y Posición</h3>", unsafe_allow_html=True)
    
    tipo_inf = st.radio("Elige una opción:", 
                       ["Estimar un Promedio (Intervalo)", "Estimar un Porcentaje (Proporción)", "Calcular Posición (Puntaje Z)"], 
                       horizontal=True)
    
    st.markdown("---")
    
    # 2.1 INTERVALO MEDIA
    if "Promedio" in tipo_inf:
        c1, c2, c3, c4 = st.columns(4)
        media = c1.number_input("Promedio de la Muestra", value=0.0)
        desv = c2.number_input("Desviación Estándar", value=0.0)
        n_size = c3.number_input("Tamaño de Muestra", value=0, min_value=0)
        confianza = c4.selectbox("Seguridad (Confianza)", [0.90, 0.95, 0.99], index=1)
        
        if st.button("Calcular Rango", key="btn_inf_mean"):
            if n_size <= 1:
                st.error("El tamaño de muestra debe ser mayor a 1.")
            elif desv <= 0:
                st.error("La desviación debe ser mayor a 0.")
            else:
                se = desv / np.sqrt(n_size)
                # T o Z automático
                critico = stats.t.ppf((1 + confianza)/2, df=n_size-1) if n_size < 30 else stats.norm.ppf((1 + confianza)/2)
                margen = critico * se
                
                c_res1, c_res2 = st.columns(2)
                c_res1.markdown(card("Mínimo Esperado", f"{media - margen:.2f}", "btn-blue"), unsafe_allow_html=True)
                c_res2.markdown(card("Máximo Esperado", f"{media + margen:.2f}", "btn-blue"), unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="simple-text" style="border-color: #3b82f6;">
                    <strong>Interpretación:</strong><br>
                    Estamos un {confianza*100:.0f}% seguros de que el verdadero promedio está entre <b>{media - margen:.2f}</b> y <b>{media + margen:.2f}</b>.
                </div>
                """, unsafe_allow_html=True)

    # 2.2 INTERVALO PROPORCIÓN
    elif "Porcentaje" in tipo_inf:
        c1, c2, c3 = st.columns(3)
        prop = c1.number_input("Porcentaje (decimal, ej: 0.5 para 50%)", 0.0, 1.0, 0.0)
        n_size = c2.number_input("Total de personas/datos", value=0, min_value=0)
        confianza = c3.selectbox("Seguridad (Confianza)", [0.90, 0.95, 0.99], index=1)
        
        if st.button("Calcular Rango", key="btn_inf_prop"):
            if n_size <= 0:
                st.error("Necesitas ingresar el tamaño total.")
            else:
                q = 1 - prop
                se = np.sqrt((prop * q) / n_size)
                z = stats.norm.ppf((1 + confianza) / 2)
                margen = z * se
                
                p_min = max(0, prop - margen)
                p_max = min(1, prop + margen)
                
                c_res1, c_res2 = st.columns(2)
                c_res1.markdown(card("Porcentaje Mínimo", f"{p_min*100:.1f}%", "btn-blue"), unsafe_allow_html=True)
                c_res2.markdown(card("Porcentaje Máximo", f"{p_max*100:.1f}%", "btn-blue"), unsafe_allow_html=True)

                st.markdown(f"""
                <div class="simple-text" style="border-color: #3b82f6;">
                    <strong>Interpretación:</strong><br>
                    El porcentaje real está entre el <b>{p_min*100:.1f}%</b> y el <b>{p_max*100:.1f}%</b> (Margen de error: {margen*100:.1f}%).
                </div>
                """, unsafe_allow_html=True)

    # 2.3 PUNTAJE Z
    elif "Posición" in tipo_inf:
        c1, c2, c3 = st.columns(3)
        valor = c1.number_input("Tu Dato", value=0.0)
        media_pob = c2.number_input("Promedio General", value=0.0)
        desv_pob = c3.number_input("Desviación General", value=1.0)
        
        if st.button("Calcular Posición", key="btn_z"):
            z = (valor - media_pob) / desv_pob
            st.markdown(card("Puntaje Z", f"{z:.2f}", "btn-blue"), unsafe_allow_html=True)
            
            explicacion = "por encima" if z > 0 else "por debajo"
            st.markdown(f"""
            <div class="simple-text" style="border-color: #3b82f6;">
                Este dato está <b>{abs(z):.2f}</b> desviaciones {explicacion} del promedio normal.
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# 3. COMPARACIÓN (ROJO)
# =============================================================================
with tab_comp:
    st.markdown("<h3 style='color: #ef4444;'>Comparar dos Grupos</h3>", unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("🅰️ **Grupo 1**")
        m1 = st.number_input("Promedio (1)", 0.0)
        s1 = st.number_input("Desviación (1)", 0.0)
        n1 = st.number_input("Total Datos (1)", 0, step=1)
    
    with col_b:
        st.write("🅱️ **Grupo 2**")
        m2 = st.number_input("Promedio (2)", 0.0)
        s2 = st.number_input("Desviación (2)", 0.0)
        n2 = st.number_input("Total Datos (2)", 0, step=1)
    
    if st.button("Ver si son diferentes", key="btn_comp"):
        if n1 < 2 or n2 < 2:
            st.error("Faltan datos en los grupos.")
        else:
            se_diff = np.sqrt((s1**2/n1) + (s2**2/n2))
            t_stat = (m1 - m2) / se_diff
            df = n1 + n2 - 2 
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            
            # Lógica simple de decisión (usando 0.05 estándar)
            es_diferente = p_val < 0.05
            titulo = "SÍ son diferentes" if es_diferente else "NO son diferentes"
            color_res = "#ef4444" if es_diferente else "#10b981" # Rojo alerta si son distintos, verde si iguales
            
            st.markdown(f"""
            <div class="result-card" style="border-top: 5px solid {color_res};">
                <h2 style="color: {color_res}; margin:0;">{titulo}</h2>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="simple-text" style="border-color: {color_res};">
                <strong>Explicación directa:</strong><br>
                {'Hay una diferencia real y significativa entre el Grupo 1 y el Grupo 2.' if es_diferente else 'La diferencia es tan pequeña que podría ser casualidad. Se consideran iguales.'}
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# 4. TAMAÑO DE MUESTRA (VERDE)
# =============================================================================
with tab_size:
    st.markdown("<h3 style='color: #10b981;'>¿A cuántos debo encuestar?</h3>", unsafe_allow_html=True)
    
    col_inp, col_out = st.columns(2)
    
    with col_inp:
        target = st.radio("Quiero calcular:", ["Para un Promedio", "Para un Porcentaje"])
        conf_lvl = st.selectbox("Nivel de Seguridad", [0.90, 0.95, 0.99], index=1)
        error = st.number_input("Error máximo aceptable (ej: 0.05 para 5%)", value=0.05, format="%.4f")
        
        sigma = 0
        if "Promedio" in target:
            sigma = st.number_input("Desviación estimada (si no sabes pon 10)", value=0.0)
        
        btn_calc_size = st.button("Calcular Cantidad", key="btn_size")

    with col_out:
        if btn_calc_size:
            z = stats.norm.ppf((1 + conf_lvl) / 2)
            n_final = 0
            
            if "Promedio" in target:
                if sigma > 0 and error > 0:
                    n_final = (z**2 * sigma**2) / error**2
            else:
                if error > 0:
                    n_final = (z**2 * 0.5 * 0.5) / error**2 # Usando 0.5 como peor escenario
            
            if n_final > 0:
                st.markdown(card("Debes encuestar a:", f"{math.ceil(n_final)} personas", "btn-green"), unsafe_allow_html=True)
            else:
                st.warning("Revisa que el error y la desviación sean mayores a 0.")
