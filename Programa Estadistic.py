import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILO
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="StatSuite Aesthetic",
    page_icon="🌌",
    layout="wide"
)

# CSS AVANZADO: Colores Neón por Categoría y Diseño Minimalista Oscuro
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

        /* RESET GENERAL */
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
            background-color: #050505;
            color: #e5e7eb;
        }
        
        /* OCULTAR ELEMENTOS DEFAULT */
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .stApp { background-color: #050505; }

        /* ESTILO DE PESTAÑAS (TABS) */
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
            border: none;
            transition: all 0.3s;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: white;
            background-color: #1a1a1a;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #222;
            color: white;
            border-bottom: 3px solid white;
        }

        /* INPUTS PERSONALIZADOS */
        .stTextArea textarea, .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
            background-color: #111 !important;
            color: white !important;
            border: 1px solid #333 !important;
            border-radius: 8px;
        }
        
        /* TARJETAS DE RESULTADO */
        .result-card {
            background-color: #121212;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            border: 1px solid #333;
            margin-bottom: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            transition: transform 0.2s;
        }
        .result-card:hover { transform: translateY(-3px); }
        
        .card-label {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 8px;
            font-weight: 600;
        }
        .card-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #fff;
        }

        /* CLASES DE COLOR POR TEMA */
        .theme-purple .card-label { color: #d8b4fe; }
        .theme-purple { border-top: 4px solid #a855f7; }

        .theme-blue .card-label { color: #93c5fd; }
        .theme-blue { border-top: 4px solid #3b82f6; }

        .theme-red .card-label { color: #fca5a5; }
        .theme-red { border-top: 4px solid #ef4444; }

        .theme-green .card-label { color: #86efac; }
        .theme-green { border-top: 4px solid #22c55e; }

        /* TEXTO DE INTERPRETACIÓN */
        .interpretation-box {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 8px;
            padding: 15px;
            margin-top: 20px;
            font-size: 1rem;
            line-height: 1.6;
            color: #d1d5db;
            border-left: 4px solid;
        }

        /* BOTONES DE ACCIÓN */
        div.stButton > button {
            width: 100%;
            background-color: #222;
            color: white;
            border: 1px solid #444;
            padding: 12px;
            font-weight: 600;
            border-radius: 8px;
        }
        div.stButton > button:hover {
            border-color: #888;
            color: white;
            background-color: #333;
        }

    </style>
""", unsafe_allow_html=True)

st.title("🌌 StatSuite Aesthetic")

# Función auxiliar para generar tarjetas HTML
def card(label, value, theme="theme-blue"):
    return f"""
    <div class="result-card {theme}">
        <div class="card-label">{label}</div>
        <div class="card-value">{value}</div>
    </div>
    """

# -----------------------------------------------------------------------------
# PESTAÑAS PRINCIPALES
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🟣 Medidas de Tendencia Central", 
    "🔵 Inferencia Estadística", 
    "🔴 Dos Poblaciones", 
    "🟢 Gráficos y Distribuciones"
])

# =============================================================================
# PESTAÑA 1: TENDENCIA CENTRAL (Morado)
# =============================================================================
with tab1:
    st.markdown("<h3 style='color:#a855f7'>Estadística Descriptiva</h3>", unsafe_allow_html=True)
    
    col_in, col_out = st.columns([1, 2], gap="large")
    
    with col_in:
        st.write("Ingrese sus datos numéricos:")
        input_desc = st.text_area("Datos (separados por coma):", placeholder="Ej: 10, 15, 12, 18, 20", height=150)
        btn_calc_desc = st.button("Analizar Datos", key="btn1")

    with col_out:
        if btn_calc_desc:
            if not input_desc.strip():
                st.warning("⚠️ Por favor ingrese al menos un número.")
            else:
                try:
                    data = np.array([float(x.strip()) for x in input_desc.split(",") if x.strip()])
                    if len(data) == 0:
                        st.error("No se encontraron datos válidos.")
                    else:
                        # Cálculos
                        media = np.mean(data)
                        mediana = np.median(data)
                        moda_res = stats.mode(data, keepdims=True)
                        moda = moda_res.mode[0] if len(moda_res.mode) > 0 else "N/A"
                        desv = np.std(data, ddof=1)
                        var = np.var(data, ddof=1)
                        rango = np.max(data) - np.min(data)
                        
                        # Resultados
                        c1, c2, c3 = st.columns(3)
                        c1.markdown(card("Media (Promedio)", f"{media:.2f}", "theme-purple"), unsafe_allow_html=True)
                        c2.markdown(card("Mediana", f"{mediana:.2f}", "theme-purple"), unsafe_allow_html=True)
                        c3.markdown(card("Moda", f"{moda}", "theme-purple"), unsafe_allow_html=True)
                        
                        c4, c5, c6 = st.columns(3)
                        c4.markdown(card("Desviación Estándar", f"{desv:.2f}", "theme-purple"), unsafe_allow_html=True)
                        c5.markdown(card("Varianza", f"{var:.2f}", "theme-purple"), unsafe_allow_html=True)
                        c6.markdown(card("Rango Total", f"{rango:.2f}", "theme-purple"), unsafe_allow_html=True)
                        
                        # Interpretación
                        st.markdown(f"""
                        <div class="interpretation-box" style="border-color: #a855f7;">
                            <strong>Interpretación de los datos:</strong><br>
                            El valor promedio de su conjunto de datos es <b>{media:.2f}</b>. 
                            La dispersión indica que los datos se alejan típicamente <b>{desv:.2f}</b> unidades de este promedio.
                            El dato central que divide la muestra es <b>{mediana:.2f}</b>.
                        </div>
                        """, unsafe_allow_html=True)

                        # Gráfico Histograma con Dígitos
                        st.write("#### Visualización")
                        fig, ax = plt.subplots(figsize=(10, 3))
                        fig.patch.set_facecolor('#050505')
                        ax.set_facecolor('#111')
                        
                        counts, bins, patches = ax.hist(data, bins='auto', color='#a855f7', edgecolor='black', alpha=0.8)
                        
                        # Agregar etiquetas de valor a las barras
                        ax.bar_label(patches, fmt='%.0f', color='white', padding=3, fontweight='bold')
                        
                        ax.axvline(media, color='white', linestyle='--', label=f'Media: {media:.1f}')
                        ax.legend(facecolor='#222', labelcolor='white', frameon=False)
                        ax.axis('off')
                        st.pyplot(fig)

                except ValueError:
                    st.error("Error: Asegúrate de usar solo números y comas.")

# =============================================================================
# PESTAÑA 2: INFERENCIA ESTADÍSTICA (Azul)
# =============================================================================
with tab2:
    st.markdown("<h3 style='color:#3b82f6'>Herramientas de Inferencia</h3>", unsafe_allow_html=True)
    
    # Menú de selección dentro de la pestaña
    opcion_inf = st.selectbox("Seleccione el cálculo que desea realizar:", 
                             ["Error Estándar", 
                              "Intervalo de Confianza (Media)", 
                              "Intervalo de Confianza (Proporción)", 
                              "Puntaje Z (Distribución Normal)", 
                              "Puntaje T (T-Student)",
                              "Tamaño de Muestra"])
    
    st.markdown("---")

    if opcion_inf == "Error Estándar":
        c1, c2 = st.columns(2)
        s = c1.number_input("Desviación Estándar Muestral", 0.0)
        n = c2.number_input("Tamaño de la Muestra", 0, step=1)
        
        if st.button("Calcular Error Estándar"):
            if n > 0:
                ee = s / np.sqrt(n)
                st.markdown(card("Error Estándar (EE)", f"{ee:.4f}", "theme-blue"), unsafe_allow_html=True)
                st.markdown(f"<div class='interpretation-box' style='border-color: #3b82f6;'>Esto indica cuánto varía la media muestral respecto a la media real de la población.</div>", unsafe_allow_html=True)
            else:
                st.warning("El tamaño de muestra debe ser mayor a 0.")

    elif opcion_inf == "Intervalo de Confianza (Media)":
        c1, c2, c3, c4 = st.columns(4)
        media = c1.number_input("Media Muestral", 0.0)
        s = c2.number_input("Desviación Estándar", 0.0)
        n = c3.number_input("Tamaño de Muestra", 0, step=1)
        conf = c4.selectbox("Nivel de Confianza", [0.90, 0.95, 0.99], index=1)
        
        if st.button("Calcular Intervalo"):
            if n > 1:
                se = s / np.sqrt(n)
                # Auto-selección T o Z
                crit = stats.t.ppf((1+conf)/2, n-1) if n < 30 else stats.norm.ppf((1+conf)/2)
                margen = crit * se
                
                col_res1, col_res2 = st.columns(2)
                col_res1.markdown(card("Límite Inferior", f"{media - margen:.4f}", "theme-blue"), unsafe_allow_html=True)
                col_res2.markdown(card("Límite Superior", f"{media + margen:.4f}", "theme-blue"), unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="interpretation-box" style="border-color: #3b82f6;">
                    <strong>Interpretación Directa:</strong><br>
                    Con una seguridad del {conf*100:.0f}%, el verdadero valor promedio de la población se encuentra entre <b>{media-margen:.2f}</b> y <b>{media+margen:.2f}</b>.
                </div>
                """, unsafe_allow_html=True)

    elif opcion_inf == "Intervalo de Confianza (Proporción)":
        c1, c2, c3 = st.columns(3)
        p = c1.number_input("Proporción (decimal, ej. 0.5)", 0.0, 1.0, 0.0)
        n = c2.number_input("Tamaño Total de Muestra", 0, step=1)
        conf = c3.selectbox("Confianza", [0.90, 0.95, 0.99], index=1)
        
        if st.button("Calcular Intervalo Proporción"):
            if n > 0:
                q = 1 - p
                se = np.sqrt((p*q)/n)
                z = stats.norm.ppf((1+conf)/2)
                margen = z * se
                
                col_res1, col_res2 = st.columns(2)
                col_res1.markdown(card("Límite Inferior", f"{(p-margen)*100:.2f}%", "theme-blue"), unsafe_allow_html=True)
                col_res2.markdown(card("Límite Superior", f"{(p+margen)*100:.2f}%", "theme-blue"), unsafe_allow_html=True)
                
                st.markdown(f"<div class='interpretation-box' style='border-color: #3b82f6;'>El porcentaje real poblacional está entre {(p-margen)*100:.1f}% y {(p+margen)*100:.1f}%.</div>", unsafe_allow_html=True)

    elif opcion_inf == "Puntaje Z (Distribución Normal)":
        c1, c2, c3 = st.columns(3)
        x = c1.number_input("Valor a evaluar (x)", 0.0)
        mu = c2.number_input("Media Poblacional", 0.0)
        sigma = c3.number_input("Desviación Poblacional", 1.0)
        
        if st.button("Calcular Z"):
            if sigma != 0:
                z = (x - mu) / sigma
                st.markdown(card("Puntaje Z", f"{z:.4f}", "theme-blue"), unsafe_allow_html=True)
                pos = "por encima" if z > 0 else "por debajo"
                st.markdown(f"<div class='interpretation-box' style='border-color: #3b82f6;'>Este dato se encuentra a <b>{abs(z):.2f}</b> desviaciones estándar {pos} del promedio.</div>", unsafe_allow_html=True)

    elif opcion_inf == "Puntaje T (T-Student)":
        c1, c2, c3, c4 = st.columns(4)
        x_bar = c1.number_input("Media Muestral", 0.0)
        mu = c2.number_input("Media Hipotética (Población)", 0.0)
        s = c3.number_input("Desviación Muestral", 1.0)
        n = c4.number_input("Tamaño de Muestra", 0, step=1)
        
        if st.button("Calcular T"):
            if s != 0 and n > 0:
                se = s / np.sqrt(n)
                t = (x_bar - mu) / se
                st.markdown(card("Puntaje T", f"{t:.4f}", "theme-blue"), unsafe_allow_html=True)

    elif opcion_inf == "Tamaño de Muestra":
        target = st.radio("¿Qué desea estimar?", ["Una Media (Promedio)", "Una Proporción (Porcentaje)"], horizontal=True)
        col_s1, col_s2, col_s3 = st.columns(3)
        conf = col_s1.selectbox("Confianza Requerida", [0.90, 0.95, 0.99], index=1)
        error = col_s2.number_input("Margen de Error Aceptable", 0.01, format="%.4f")
        
        sigma = 0
        p_est = 0.5
        
        if "Media" in target:
            sigma = col_s3.number_input("Desviación Estándar Estimada", 0.0)
        else:
            p_est = col_s3.number_input("Proporción Estimada (0.5 si desconoce)", 0.0, 1.0, 0.5)
            
        if st.button("Calcular Tamaño Muestra"):
            z = stats.norm.ppf((1+conf)/2)
            if "Media" in target:
                if sigma > 0 and error > 0:
                    n = (z**2 * sigma**2) / error**2
                    st.markdown(card("Muestra Necesaria", f"{math.ceil(n)}", "theme-blue"), unsafe_allow_html=True)
                    st.caption("Número mínimo de sujetos a encuestar.")
            else:
                if error > 0:
                    n = (z**2 * p_est * (1-p_est)) / error**2
                    st.markdown(card("Muestra Necesaria", f"{math.ceil(n)}", "theme-blue"), unsafe_allow_html=True)

# =============================================================================
# PESTAÑA 3: DOS POBLACIONES (Rojo)
# =============================================================================
with tab3:
    st.markdown("<h3 style='color:#ef4444'>Comparación y Pruebas de Hipótesis</h3>", unsafe_allow_html=True)
    
    opcion_comp = st.selectbox("Seleccione Análisis:", 
                              ["Diferencia de Medias (Prueba T Independiente)", 
                               "Diferencia de Proporciones"])

    st.markdown("---")
    
    if opcion_comp == "Diferencia de Medias (Prueba T Independiente)":
        c1, c2 = st.columns(2)
        with c1:
            st.write("🅰️ **Grupo 1**")
            m1 = st.number_input("Media Grupo 1", 0.0)
            s1 = st.number_input("Desviación Grupo 1", 0.0)
            n1 = st.number_input("Tamaño Grupo 1", 0, step=1)
        with c2:
            st.write("🅱️ **Grupo 2**")
            m2 = st.number_input("Media Grupo 2", 0.0)
            s2 = st.number_input("Desviación Grupo 2", 0.0)
            n2 = st.number_input("Tamaño Grupo 2", 0, step=1)
            
        alpha = st.slider("Nivel de Significancia (Alpha)", 0.01, 0.10, 0.05)
        
        if st.button("Realizar Prueba de Hipótesis", key="btn_test_means"):
            if n1 > 1 and n2 > 1 and s1 > 0 and s2 > 0:
                se_diff = np.sqrt((s1**2/n1) + (s2**2/n2))
                t_stat = (m1 - m2) / se_diff
                # Grados de libertad (simplificado)
                df = n1 + n2 - 2
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                
                col_res1, col_res2 = st.columns(2)
                col_res1.markdown(card("Estadístico T", f"{t_stat:.4f}", "theme-red"), unsafe_allow_html=True)
                col_res2.markdown(card("Valor P", f"{p_val:.4f}", "theme-red"), unsafe_allow_html=True)
                
                conclusion = "EXISTE una diferencia significativa" if p_val < alpha else "NO EXISTE diferencia significativa"
                st.markdown(f"""
                <div class="interpretation-box" style="border-color: #ef4444;">
                    <strong>Conclusión Directa:</strong><br>
                    {conclusion} entre los dos grupos (basado en un alpha de {alpha}).
                    {'La diferencia observada es real.' if p_val < alpha else 'La diferencia es probablemente casualidad.'}
                </div>
                """, unsafe_allow_html=True)

    elif opcion_comp == "Diferencia de Proporciones":
        c1, c2 = st.columns(2)
        with c1:
            st.write("🅰️ **Grupo 1**")
            x1 = st.number_input("Éxitos Grupo 1", 0)
            nt1 = st.number_input("Total Grupo 1", 0)
        with c2:
            st.write("🅱️ **Grupo 2**")
            x2 = st.number_input("Éxitos Grupo 2", 0)
            nt2 = st.number_input("Total Grupo 2", 0)
            
        alpha = st.slider("Alpha", 0.01, 0.10, 0.05)
        
        if st.button("Realizar Prueba Z", key="btn_test_props"):
            if nt1 > 0 and nt2 > 0:
                p1, p2 = x1/nt1, x2/nt2
                pp = (x1+x2)/(nt1+nt2)
                se = np.sqrt(pp*(1-pp)*(1/nt1 + 1/nt2))
                z = (p1 - p2) / se
                p_val = 2 * (1 - stats.norm.cdf(abs(z)))
                
                st.markdown(card("Valor P", f"{p_val:.4f}", "theme-red"), unsafe_allow_html=True)
                
                conclusion = "Diferencia Significativa" if p_val < alpha else "Sin Diferencia Significativa"
                st.markdown(f"<div class='interpretation-box' style='border-color: #ef4444;'>Resultado: <b>{conclusion}</b>.</div>", unsafe_allow_html=True)

# =============================================================================
# PESTAÑA 4: GRÁFICOS Y TLC (Verde)
# =============================================================================
with tab4:
    st.markdown("<h3 style='color:#22c55e'>Gráficos y Teorema del Límite Central</h3>", unsafe_allow_html=True)
    
    viz_type = st.radio("Seleccione herramienta visual:", 
                        ["Simulación TLC (Teorema Límite Central)", "Visualizar Distribución Normal"])
    
    st.markdown("---")
    
    if viz_type == "Simulación TLC (Teorema Límite Central)":
        c1, c2 = st.columns(2)
        n_sim = c1.slider("Tamaño de cada muestra (n)", 1, 100, 30)
        reps = c2.slider("Cantidad de muestras simuladas", 100, 5000, 1000)
        
        if st.button("Simular TLC"):
            # Población Uniforme
            pop = np.random.uniform(0, 100, 10000)
            means = [np.mean(np.random.choice(pop, n_sim)) for _ in range(reps)]
            
            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor('#050505')
            ax.set_facecolor('#111')
            
            ax.hist(means, bins=30, color='#22c55e', alpha=0.7, edgecolor='black')
            ax.set_title(f"Distribución de {reps} medias muestrales (n={n_sim})", color='white')
            ax.axis('off')
            st.pyplot(fig)
            st.markdown(f"<div class='interpretation-box' style='border-color: #22c55e;'>Note como la distribución de las medias forma una campana perfecta, validando el Teorema del Límite Central.</div>", unsafe_allow_html=True)

    elif viz_type == "Visualizar Distribución Normal":
        mu_viz = st.slider("Media (Centro)", -10.0, 10.0, 0.0)
        sigma_viz = st.slider("Desviación Estándar (Ancho)", 0.5, 5.0, 1.0)
        
        x = np.linspace(mu_viz - 4*sigma_viz, mu_viz + 4*sigma_viz, 200)
        y = stats.norm.pdf(x, mu_viz, sigma_viz)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#050505')
        ax.set_facecolor('#111')
        
        ax.plot(x, y, color='#22c55e', linewidth=3)
        ax.fill_between(x, y, color='#22c55e', alpha=0.2)
        ax.set_title("Curva de Densidad Normal", color='white')
        ax.axis('off')
        st.pyplot(fig)
