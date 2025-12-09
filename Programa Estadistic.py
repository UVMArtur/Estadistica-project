import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math

# -----------------------------------------------------------------------------
# CONFIGURACIÓN Y ESTILO MINIMALISTA (FLAT)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Calculadora Estadística", layout="wide")

# CSS para forzar estilo "Dark & Blue Flat"
st.markdown("""
    <style>
    /* Reset básico y fuente limpia */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        color: #e0e0e0;
    }

    /* Ocultar elementos predeterminados de Streamlit */
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Fondo Principal: Negro/Gris muy oscuro sólido */
    .stApp {
        background-color: #050505;
    }

    /* Pestañas (Tabs) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: transparent;
        border-bottom: 1px solid #333;
        padding-bottom: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border: none;
        color: #666;
        font-size: 16px;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #3b82f6; /* Azul Flat */
        border-bottom: 2px solid #3b82f6;
    }

    /* Botones: Azul sólido, sin degradados, bordes sutiles */
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: 500;
        transition: background-color 0.2s;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #2563eb;
        color: white;
    }

    /* Inputs y Selects */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div, .stTextArea textarea {
        background-color: #111;
        border: 1px solid #333;
        color: white;
        border-radius: 4px;
    }
    
    /* Métricas y Resultados */
    div[data-testid="stMetric"] {
        background-color: #0a0a0a;
        border: 1px solid #222;
        padding: 10px;
        border-radius: 4px;
    }
    div[data-testid="stMetricLabel"] {
        color: #888;
        font-size: 0.8rem;
    }
    div[data-testid="stMetricValue"] {
        color: #fff;
        font-size: 1.2rem;
    }

    /* Caja de Interpretación */
    .interpretation {
        margin-top: 20px;
        padding: 15px;
        border-left: 3px solid #3b82f6;
        background-color: #0a101f;
        color: #ccc;
        font-size: 0.9rem;
    }
    
    h1, h2, h3 { color: white !important; font-weight: 400; }
    </style>
""", unsafe_allow_html=True)

st.title("Calculadora Estadística")

# Estructura de Pestañas
tab1, tab2, tab3, tab4 = st.tabs([
    "Descriptiva", 
    "Inferencia (1 Población)", 
    "Dos Poblaciones", 
    "Gráficos & TLC"
])

# =============================================================================
# PESTAÑA 1: MEDIDAS DE TENDENCIA CENTRAL (DESCRIPTIVA)
# =============================================================================
with tab1:
    col_izq, col_der = st.columns([1, 2], gap="large")
    
    with col_izq:
        st.subheader("Entrada de Datos")
        input_text = st.text_area("Ingrese números separados por coma", "10, 12, 15, 14, 13, 16, 10, 12")
        
        if st.button("Calcular Descriptiva"):
            try:
                # Procesar datos
                lista = [float(x.strip()) for x in input_text.split(",") if x.strip()]
                arr = np.array(lista)
                st.session_state['data_desc'] = arr
            except:
                st.error("Formato inválido.")

    with col_der:
        st.subheader("Resultados")
        if 'data_desc' in st.session_state:
            data = st.session_state['data_desc']
            n = len(data)
            media = np.mean(data)
            mediana = np.median(data)
            desv = np.std(data, ddof=1)
            varianza = np.var(data, ddof=1)
            error_est = desv / np.sqrt(n)
            rango = np.max(data) - np.min(data)
            
            # Grid de resultados
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Media", f"{media:.2f}")
            c2.metric("Mediana", f"{mediana:.2f}")
            c3.metric("Mínimo", f"{np.min(data)}")
            c4.metric("Máximo", f"{np.max(data)}")
            
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Desviación Std", f"{desv:.2f}")
            c6.metric("Varianza", f"{varianza:.2f}")
            c7.metric("Error Estándar", f"{error_est:.2f}")
            c8.metric("Rango", f"{rango:.2f}")

            st.markdown(f"""
            <div class="interpretation">
                <strong>Interpretación:</strong><br>
                El conjunto tiene {n} observaciones. El valor central promedio es {media:.2f}. 
                La dispersión de los datos (Desviación Estándar) es de {desv:.2f}, lo que genera un Error Estándar de {error_est:.4f}.
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# PESTAÑA 2: INFERENCIA ESTADÍSTICA (1 POBLACIÓN)
# =============================================================================
with tab2:
    st.subheader("Herramientas de Inferencia")
    
    opcion = st.selectbox("Seleccione cálculo:", [
        "Intervalo de Confianza (Media)",
        "Intervalo de Confianza (Proporción)",
        "Cálculo de Puntaje Z",
        "Cálculo de Puntaje T",
        "Tamaño de Muestra (Media)",
        "Tamaño de Muestra (Proporción)"
    ])
    
    st.write("---") # Separador sutil

    # --- IC MEDIA ---
    if opcion == "Intervalo de Confianza (Media)":
        c1, c2, c3, c4 = st.columns(4)
        media = c1.number_input("Media Muestral", value=100.0)
        desv = c2.number_input("Desviación Std", value=15.0)
        n = c3.number_input("Tamaño Muestra (n)", value=30, step=1)
        conf = c4.slider("Nivel Confianza", 0.80, 0.99, 0.95)

        if n > 1:
            se = desv / np.sqrt(n)
            # Elección automática de T o Z según n
            if n < 30:
                crit = stats.t.ppf((1 + conf)/2, df=n-1)
                metodo = "T-Student (n<30)"
            else:
                crit = stats.norm.ppf((1 + conf)/2)
                metodo = "Normal Z (n>=30)"
            
            margen = crit * se
            lim_inf, lim_sup = media - margen, media + margen
            
            st.info(f"Resultado: [{lim_inf:.4f}, {lim_sup:.4f}]")
            st.markdown(f"<div class='interpretation'>Con un {conf*100:.0f}% de confianza, la media poblacional está en este rango. Método usado: {metodo}.</div>", unsafe_allow_html=True)

    # --- IC PROPORCIÓN ---
    elif opcion == "Intervalo de Confianza (Proporción)":
        c1, c2, c3 = st.columns(3)
        p = c1.number_input("Proporción muestral (decimal)", 0.0, 1.0, 0.5)
        n = c2.number_input("Tamaño Muestra (n)", value=100)
        conf = c3.slider("Nivel Confianza", 0.80, 0.99, 0.95)
        
        if n > 0:
            q = 1 - p
            se = np.sqrt((p*q)/n)
            z = stats.norm.ppf((1 + conf)/2)
            margen = z * se
            st.info(f"Resultado: [{p - margen:.4f}, {p + margen:.4f}]")
            st.markdown(f"<div class='interpretation'>Se estima que la proporción real está entre el {(p-margen)*100:.2f}% y el {(p+margen)*100:.2f}%.</div>", unsafe_allow_html=True)

    # --- Z SCORE ---
    elif opcion == "Cálculo de Puntaje Z":
        c1, c2, c3 = st.columns(3)
        x = c1.number_input("Valor (x)", 0.0)
        mu = c2.number_input("Media Poblacional (µ)", 0.0)
        sigma = c3.number_input("Desviación Poblacional (σ)", 1.0)
        
        if sigma != 0:
            z = (x - mu) / sigma
            st.metric("Z Score", f"{z:.4f}")
            prob = stats.norm.cdf(z)
            st.markdown(f"<div class='interpretation'>El valor está a {z:.2f} desviaciones estándar de la media. Probabilidad acumulada: {prob:.4f}.</div>", unsafe_allow_html=True)

    # --- T SCORE ---
    elif opcion == "Cálculo de Puntaje T":
        c1, c2, c3, c4 = st.columns(4)
        x_bar = c1.number_input("Media Muestral (x̄)", 0.0)
        mu = c2.number_input("Media Hipotética (µ)", 0.0)
        s = c3.number_input("Desviación Muestral (s)", 1.0)
        n = c4.number_input("Tamaño Muestra (n)", 10)

        if s != 0 and n > 0:
            se = s / np.sqrt(n)
            t = (x_bar - mu) / se
            st.metric("T Score", f"{t:.4f}")
            st.markdown(f"<div class='interpretation'>Estadístico T calculado con {n-1} grados de libertad.</div>", unsafe_allow_html=True)

    # --- TAMAÑO DE MUESTRA ---
    elif opcion.startswith("Tamaño de Muestra"):
        col1, col2, col3 = st.columns(3)
        conf = col1.slider("Confianza deseada", 0.80, 0.99, 0.95)
        error = col2.number_input("Error máximo permitido (E)", value=0.05)
        z = stats.norm.ppf((1 + conf)/2)

        if opcion == "Tamaño de Muestra (Media)":
            sigma = col3.number_input("Desviación estimada (σ)", value=10.0)
            if error > 0:
                n_res = (z**2 * sigma**2) / error**2
                st.metric("Muestra necesaria (n)", f"{math.ceil(n_res)}")
        else:
            p = col3.number_input("Proporción estimada (p)", value=0.5)
            if error > 0:
                n_res = (z**2 * p * (1-p)) / error**2
                st.metric("Muestra necesaria (n)", f"{math.ceil(n_res)}")

# =============================================================================
# PESTAÑA 3: DOS POBLACIONES
# =============================================================================
with tab3:
    st.subheader("Comparación de Grupos")
    tipo_prueba = st.radio("Tipo de Análisis:", ["Diferencia de Medias", "Diferencia de Proporciones"], horizontal=True)
    
    col_g1, col_g2 = st.columns(2)
    
    if tipo_prueba == "Diferencia de Medias":
        with col_g1:
            st.markdown("GRUPO 1")
            m1 = st.number_input("Media 1", 50.0)
            s1 = st.number_input("Desv 1", 5.0)
            n1 = st.number_input("N 1", 30)
        with col_g2:
            st.markdown("GRUPO 2")
            m2 = st.number_input("Media 2", 45.0)
            s2 = st.number_input("Desv 2", 5.0)
            n2 = st.number_input("N 2", 30)
            
        alpha = st.slider("Alpha (Significancia)", 0.01, 0.10, 0.05)
        
        if st.button("Calcular Prueba de Hipótesis"):
            # Welch's t-test
            se_diff = np.sqrt((s1**2/n1) + (s2**2/n2))
            t_stat = (m1 - m2) / se_diff
            df = ((s1**2/n1 + s2**2/n2)**2) / ( ((s1**2/n1)**2/(n1-1)) + ((s2**2/n2)**2/(n2-1)) )
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            
            res_txt = "Se rechaza H0 (Diferencia Significativa)" if p_val < alpha else "No se rechaza H0 (Sin evidencia de diferencia)"
            
            c_res1, c_res2 = st.columns(2)
            c_res1.metric("Estadístico t", f"{t_stat:.4f}")
            c_res2.metric("Valor P", f"{p_val:.4f}")
            
            st.markdown(f"<div class='interpretation'><strong>Conclusión:</strong> {res_txt} al nivel {alpha}.</div>", unsafe_allow_html=True)

    else: # Diferencia Proporciones
        with col_g1:
            x1 = st.number_input("Éxitos Grupo 1", 40)
            nt1 = st.number_input("Total Grupo 1", 100)
        with col_g2:
            x2 = st.number_input("Éxitos Grupo 2", 30)
            nt2 = st.number_input("Total Grupo 2", 100)
            
        alpha = st.slider("Alpha", 0.01, 0.10, 0.05)
        
        if st.button("Calcular Z"):
            p1, p2 = x1/nt1, x2/nt2
            pp = (x1+x2)/(nt1+nt2)
            se = np.sqrt(pp*(1-pp)*(1/nt1 + 1/nt2))
            z = (p1 - p2) / se
            p_val = 2 * (1 - stats.norm.cdf(abs(z)))
            
            res_txt = "Diferencia Significativa" if p_val < alpha else "No hay diferencia significativa"
            
            st.metric("Diferencia observada", f"{(p1-p2)*100:.2f}%")
            st.metric("Valor P", f"{p_val:.4f}")
            st.markdown(f"<div class='interpretation'><strong>Conclusión:</strong> {res_txt}.</div>", unsafe_allow_html=True)


# =============================================================================
# PESTAÑA 4: GRÁFICOS Y TLC
# =============================================================================
with tab4:
    st.subheader("Visualización")
    
    viz_mode = st.selectbox("Seleccione visualización:", ["Histograma de Datos", "Simulación Teorema Límite Central"])
    
    if viz_mode == "Histograma de Datos":
        if 'data_desc' in st.session_state:
            data = st.session_state['data_desc']
            
            fig, ax = plt.subplots(figsize=(10, 3))
            fig.patch.set_facecolor('#050505')
            ax.set_facecolor('#050505')
            
            # Histograma
            ax.hist(data, bins='auto', color='#3b82f6', alpha=0.7, rwidth=0.9)
            ax.set_title("Distribución de Frecuencias", color='white')
            
            ax.tick_params(colors='#888')
            ax.spines['bottom'].set_color('#333')
            ax.spines['left'].set_color('#333')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            st.pyplot(fig)
            st.markdown("<div class='interpretation'>Visualización de la distribución de los datos ingresados en la pestaña 1.</div>", unsafe_allow_html=True)
        else:
            st.warning("Ingrese datos en la pestaña Descriptiva primero.")
            
    elif viz_mode == "Simulación Teorema Límite Central":
        col_sim1, col_sim2 = st.columns(2)
        n_sim = col_sim1.slider("Tamaño de cada muestra (n)", 1, 100, 30)
        reps = col_sim2.slider("Número de repeticiones", 100, 2000, 500)
        
        if st.button("Ejecutar Simulación"):
            # Población Uniforme (No normal)
            poblacion = np.random.uniform(0, 100, 10000)
            medias = [np.mean(np.random.choice(poblacion, n_sim)) for _ in range(reps)]
            
            fig, ax = plt.subplots(figsize=(10, 3))
            fig.patch.set_facecolor('#050505')
            ax.set_facecolor('#050505')
            
            ax.hist(medias, bins=30, color='#3b82f6', alpha=0.7)
            ax.set_title(f"Distribución de Medias Muestrales (n={n_sim})", color='white')
            ax.tick_params(colors='#888')
            ax.spines['bottom'].set_color('#333')
            ax.spines['left'].set_color('#333')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            st.pyplot(fig)
            st.markdown("<div class='interpretation'>Observe cómo la distribución de las medias tiende a formarse como una campana (Normal) a medida que aumenta 'n', sin importar la población original.</div>", unsafe_allow_html=True)
