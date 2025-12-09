import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA Y ESTÉTICA (NEGRO Y AZUL MINIMALISTA)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="StatSuite Pro", layout="wide", page_icon="📊")

# CSS Personalizado para forzar el tema Azul/Negro
st.markdown("""
    <style>
    /* Fondo principal */
    .stApp {
        background-color: #050505;
        color: #e0e0e0;
    }
    /* Headers */
    h1, h2, h3 {
        color: #29b5e8 !important;
        font-family: 'Helvetica', sans-serif;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #0e1117;
        border-radius: 4px;
        color: #888;
        border: 1px solid #333;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #29b5e8;
        color: #000;
        font-weight: bold;
    }
    /* Botones */
    .stButton>button {
        background-color: #0e1117;
        color: #29b5e8;
        border: 1px solid #29b5e8;
        border-radius: 5px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #29b5e8;
        color: #000;
        box-shadow: 0 0 10px #29b5e8;
    }
    /* Inputs */
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #161b22;
        color: white;
        border: 1px solid #30363d;
    }
    /* Tarjetas de interpretación */
    .interpretation-box {
        background-color: #0d1b26;
        border-left: 5px solid #29b5e8;
        padding: 15px;
        margin-top: 15px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ StatSuite: Análisis Estadístico Integral")
st.markdown("---")

# -----------------------------------------------------------------------------
# LÓGICA DE PESTAÑAS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📂 Descriptiva & Datos", 
    "🧮 Inferencia (1 Población)", 
    "⚖️ Dos Poblaciones & Hipótesis", 
    "📈 Gráficos & TLC"
])

# =============================================================================
# PESTAÑA 1: ESTADÍSTICA DESCRIPTIVA Y CARGA DE DATOS
# =============================================================================
with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Entrada de Datos")
        data_input = st.text_area("Ingresa datos separados por comas:", "10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 10.0, 15.2")
        
        if st.button("Procesar Datos", key="btn_desc"):
            try:
                raw_data = [float(x.strip()) for x in data_input.split(",") if x.strip()]
                st.session_state["dataset"] = np.array(raw_data)
                st.success("Datos cargados al sistema.")
            except ValueError:
                st.error("Formato incorrecto. Usa números separados por comas.")

    with col2:
        if "dataset" in st.session_state:
            data = st.session_state["dataset"]
            st.subheader("Medidas de Tendencia Central y Dispersión")
            
            # Cálculos
            media = np.mean(data)
            mediana = np.median(data)
            moda_result = stats.mode(data, keepdims=True)
            moda = moda_result.mode[0] if len(moda_result.mode) > 0 else "N/A"
            desv = np.std(data, ddof=1)
            var = np.var(data, ddof=1)
            n = len(data)
            ee = desv / np.sqrt(n)

            # Métricas visuales
            c1, c2, c3 = st.columns(3)
            c1.metric("Media", f"{media:.4f}")
            c2.metric("Mediana", f"{mediana:.4f}")
            c3.metric("Moda", f"{moda}")
            
            c4, c5, c6 = st.columns(3)
            c4.metric("Desviación Estándar", f"{desv:.4f}")
            c5.metric("Varianza", f"{var:.4f}")
            c6.metric("Error Estándar", f"{ee:.4f}")

            # Interpretación
            st.markdown(f"""
            <div class="interpretation-box">
                <strong>💡 Interpretación:</strong><br>
                El centro de gravedad de tus datos se encuentra en <b>{media:.2f}</b>. 
                La dispersión promedio respecto a este centro es de <b>{desv:.2f}</b> unidades.
                El error estándar de <b>{ee:.2f}</b> sugiere cuánto podría variar la media muestral respecto a la media real de la población.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Carga los datos en el panel izquierdo para ver el análisis.")

# =============================================================================
# PESTAÑA 2: INFERENCIA (UNA POBLACIÓN)
# =============================================================================
with tab2:
    st.subheader("Cálculos de Una Sola Población")
    option = st.selectbox("Selecciona el cálculo:", 
                          ["Intervalo de Confianza (Media)", 
                           "Intervalo de Confianza (Proporción)", 
                           "Cálculo de Z / T (Puntajes)",
                           "Tamaño de Muestra (n)"])

    # --- IC MEDIA ---
    if option == "Intervalo de Confianza (Media)":
        c1, c2, c3, c4 = st.columns(4)
        x_bar = c1.number_input("Media Muestral (x̄)", value=100.0)
        s = c2.number_input("Desv. Estándar (s)", value=15.0)
        n_val = c3.number_input("Tamaño Muestra (n)", value=30, step=1)
        conf = c4.slider("Nivel de Confianza (%)", 80, 99, 95) / 100

        if n_val > 1:
            se = s / np.sqrt(n_val)
            if n_val < 30:
                t_crit = stats.t.ppf((1 + conf) / 2, df=n_val-1)
                margin = t_crit * se
                dist_used = "t-Student"
            else:
                z_crit = stats.norm.ppf((1 + conf) / 2)
                margin = z_crit * se
                dist_used = "Normal (Z)"

            lower = x_bar - margin
            upper = x_bar + margin

            st.write(f"**Margen de Error:** {margin:.4f} | **Distribución usada:** {dist_used}")
            st.markdown(f"""
            <div class="interpretation-box">
                <strong>💡 Interpretación:</strong><br>
                Con un nivel de confianza del {conf*100:.0f}%, estimamos que la verdadera media de la población se encuentra entre 
                <b>[{lower:.4f} y {upper:.4f}]</b>.
            </div>
            """, unsafe_allow_html=True)

    # --- IC PROPORCION ---
    elif option == "Intervalo de Confianza (Proporción)":
        c1, c2, c3 = st.columns(3)
        p_hat = c1.number_input("Proporción muestral (p)", 0.0, 1.0, 0.5)
        n_val = c2.number_input("Tamaño Muestra (n)", value=100, step=1)
        conf = c3.slider("Confianza (%)", 80, 99, 95) / 100

        if n_val > 0:
            q_hat = 1 - p_hat
            se_p = np.sqrt((p_hat * q_hat) / n_val)
            z_crit = stats.norm.ppf((1 + conf) / 2)
            margin = z_crit * se_p
            lower = max(0, p_hat - margin)
            upper = min(1, p_hat + margin)

            st.markdown(f"""
            <div class="interpretation-box">
                <strong>💡 Interpretación:</strong><br>
                Estamos un {conf*100:.0f}% seguros de que la verdadera proporción poblacional está entre 
                <b>{lower:.2%} y {upper:.2%}</b>.
            </div>
            """, unsafe_allow_html=True)

    # --- CALCULO Z / T ---
    elif option == "Cálculo de Z / T (Puntajes)":
        tipo = st.radio("Tipo de puntaje:", ["Z (Población conocida)", "T (Población desconocida/n<30)"], horizontal=True)
        c1, c2, c3 = st.columns(3)
        val_obs = c1.number_input("Valor Observado / Media Muestral", value=0.0)
        val_esp = c2.number_input("Media Poblacional (µ)", value=0.0)
        desv_in = c3.number_input("Desviación (σ o s)", value=1.0)
        
        n_in = 1
        if "T" in tipo:
            n_in = st.number_input("Tamaño de muestra (n)", value=10, step=1)
            denom = desv_in / np.sqrt(n_in)
        else:
            denom = desv_in # Si es Z individual. Si es Z de medias, sería sigma/sqrt(n) pero usaremos el caso simple.

        if st.button("Calcular Puntaje"):
            score = (val_obs - val_esp) / denom
            
            p_val_text = ""
            if "Z" in tipo:
                p_accum = stats.norm.cdf(score)
                p_val_text = f"Probabilidad acumulada (Area a la izquierda): {p_accum:.4f}"
            else:
                p_accum = stats.t.cdf(score, df=n_in-1)
                p_val_text = f"Probabilidad acumulada (Grados de libertad {n_in-1}): {p_accum:.4f}"

            st.metric("Puntaje Calculado", f"{score:.4f}")
            st.markdown(f"""
            <div class="interpretation-box">
                <strong>💡 Interpretación:</strong><br>
                El valor observado se encuentra a <b>{abs(score):.2f}</b> desviaciones estándar de la media.
                {p_val_text}.
            </div>
            """, unsafe_allow_html=True)

    # --- TAMAÑO DE MUESTRA ---
    elif option == "Tamaño de Muestra (n)":
        target = st.radio("Objetivo:", ["Estimar Media", "Estimar Proporción"], horizontal=True)
        conf = st.slider("Nivel de Confianza (%)", 80, 99, 95) / 100
        z = stats.norm.ppf((1 + conf) / 2)
        error = st.number_input("Margen de Error deseado (E)", value=0.05, format="%.4f")

        n_result = 0
        txt = ""
        
        if target == "Estimar Media":
            sigma = st.number_input("Desviación estándar estimada (σ)", value=10.0)
            if error > 0:
                n_result = (z**2 * sigma**2) / error**2
                txt = "Para mantener la media dentro del error establecido."
        else:
            p_est = st.number_input("Proporción estimada (p) [Dejar 0.5 si se desconoce]", value=0.5)
            q_est = 1 - p_est
            if error > 0:
                n_result = (z**2 * p_est * q_est) / error**2
                txt = "Para estimar la proporción con máxima incertidumbre (p=0.5) o basada en estudios previos."

        if error > 0:
            st.metric("Tamaño de Muestra Requerido", f"{np.ceil(n_result):.0f}")
            st.markdown(f"""
            <div class="interpretation-box">
                <strong>💡 Interpretación:</strong><br>
                Necesitas encuestar o medir al menos a <b>{int(np.ceil(n_result))}</b> sujetos {txt}
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# PESTAÑA 3: DOS POBLACIONES Y PRUEBA DE HIPÓTESIS
# =============================================================================
with tab3:
    st.subheader("Comparación de Dos Poblaciones & Hipótesis")
    
    test_type = st.selectbox("Selecciona la prueba:", 
                             ["Diferencia de Medias (Prueba Z/T)", 
                              "Diferencia de Proporciones"])
    
    alpha = st.number_input("Nivel de Significancia (α)", 0.01, 0.10, 0.05, step=0.01)

    if test_type == "Diferencia de Medias (Prueba Z/T)":
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Grupo 1**")
            m1 = st.number_input("Media 1", value=50.0)
            s1 = st.number_input("Desv. 1", value=5.0)
            n1 = st.number_input("n1", value=30)
        with c2:
            st.markdown("**Grupo 2**")
            m2 = st.number_input("Media 2", value=45.0)
            s2 = st.number_input("Desv. 2", value=5.0)
            n2 = st.number_input("n2", value=30)
        
        if st.button("Ejecutar Prueba de Hipótesis"):
            # Error estándar de la diferencia
            se_diff = np.sqrt((s1**2/n1) + (s2**2/n2))
            # Estadístico T (asumiendo varianzas no iguales - Welch)
            t_stat = (m1 - m2) / se_diff
            
            # Grados de libertad (Aprox Welch-Satterthwaite)
            df = ((s1**2/n1 + s2**2/n2)**2) / ( ((s1**2/n1)**2/(n1-1)) + ((s2**2/n2)**2/(n2-1)) )
            
            # P-value (dos colas)
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            
            decision = "RECHAZAR H0" if p_val < alpha else "NO RECHAZAR H0"
            color_dec = "red" if p_val < alpha else "green"

            st.write("---")
            col_res1, col_res2 = st.columns(2)
            col_res1.metric("Estadístico t", f"{t_stat:.4f}")
            col_res2.metric("Valor P", f"{p_val:.4f}")

            st.markdown(f"""
            <div class="interpretation-box">
                <strong>💡 Interpretación / Decisión:</strong><br>
                Dado que el valor P ({p_val:.4f}) es {'menor' if p_val < alpha else 'mayor'} que alfa ({alpha}), 
                la decisión es: <b style='color:{color_dec}'>{decision}</b>.<br><br>
                {'Existe evidencia estadística suficiente para afirmar que las medias de los grupos son diferentes.' if p_val < alpha else 'No hay evidencia suficiente para afirmar que las medias son diferentes.'}
            </div>
            """, unsafe_allow_html=True)

    elif test_type == "Diferencia de Proporciones":
        c1, c2 = st.columns(2)
        with c1:
            x1 = st.number_input("Éxitos Grupo 1", value=40)
            n1 = st.number_input("Total Grupo 1", value=100)
        with c2:
            x2 = st.number_input("Éxitos Grupo 2", value=30)
            n2 = st.number_input("Total Grupo 2", value=100)

        if st.button("Comparar Proporciones"):
            p1 = x1/n1
            p2 = x2/n2
            p_pool = (x1 + x2) / (n1 + n2)
            se_diff = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
            
            z_stat = (p1 - p2) / se_diff
            p_val = 2 * (1 - stats.norm.cdf(abs(z_stat))) # Dos colas

            decision = "RECHAZAR H0" if p_val < alpha else "NO RECHAZAR H0"

            st.metric("Estadístico Z", f"{z_stat:.4f}")
            st.metric("Valor P", f"{p_val:.4f}")

            st.markdown(f"""
            <div class="interpretation-box">
                <strong>💡 Interpretación:</strong><br>
                La diferencia observada entre proporciones es de {p1-p2:.2%}.<br>
                Estadísticamente: <b>{decision}</b>. {'Las proporciones son significativamente diferentes.' if p_val < alpha else 'La diferencia observada puede deberse al azar.'}
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# PESTAÑA 4: GRÁFICOS Y TLC (VISUALIZACIÓN)
# =============================================================================
with tab4:
    st.subheader("Visualización y Teorema del Límite Central")
    
    viz_type = st.radio("Herramienta Visual:", 
                        ["Histograma (Datos Cargados)", "Simulación TLC (Distribuciones Muestrales)", "Gráfico Normal Estándar"], 
                        horizontal=True)

    if viz_type == "Histograma (Datos Cargados)":
        if "dataset" in st.session_state:
            data = st.session_state["dataset"]
            fig, ax = plt.subplots(facecolor='#0e1117')
            ax.set_facecolor('#0e1117')
            
            # Histograma
            n_bins, bins, patches = ax.hist(data, bins='auto', color='#29b5e8', alpha=0.7, edgecolor='black')
            
            # Línea de densidad
            density = stats.gaussian_kde(data)
            xs = np.linspace(min(data), max(data), 200)
            ax.plot(xs, density(xs)*len(data)*(bins[1]-bins[0]), color='white', linestyle='--')

            ax.set_title("Distribución de Frecuencias", color='white')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            
            st.pyplot(fig)
            st.markdown(f"""
            <div class="interpretation-box">
                <strong>💡 Interpretación:</strong><br>
                Visualiza la forma de tus datos. Si la línea punteada (densidad) se asemeja a una campana simétrica, tus datos podrían seguir una distribución normal.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Ve a la pestaña 1 y carga datos primero.")

    elif viz_type == "Simulación TLC (Distribuciones Muestrales)":
        st.write("Demostración de cómo las medias muestrales se distribuyen normalmente aunque la población no lo sea.")
        
        col_tlc1, col_tlc2 = st.columns(2)
        sz = col_tlc1.slider("Tamaño de cada muestra (n)", 2, 100, 30)
        sims = col_tlc2.slider("Número de simulaciones (muestras)", 100, 5000, 1000)
        
        # Generar población asimétrica (Exponencial)
        pop_data = np.random.exponential(scale=1.0, size=10000)
        
        # Muestreo
        sample_means = [np.mean(np.random.choice(pop_data, sz)) for _ in range(sims)]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), facecolor='#0e1117')
        
        # Plot Población
        ax1.set_facecolor('#0e1117')
        ax1.hist(pop_data, bins=30, color='#333', alpha=0.8)
        ax1.set_title("Población Original (Asimétrica)", color='white', fontsize=10)
        ax1.tick_params(colors='white')

        # Plot Distribución Muestral
        ax2.set_facecolor('#0e1117')
        ax2.hist(sample_means, bins=30, color='#29b5e8', alpha=0.8, density=True)
        ax2.set_title(f"Distribución de Medias (n={sz})", color='white', fontsize=10)
        
        # Curva Normal Teórica sobrepuesta
        mu_x = np.mean(pop_data)
        sigma_x = np.std(pop_data) / np.sqrt(sz)
        x = np.linspace(min(sample_means), max(sample_means), 100)
        ax2.plot(x, stats.norm.pdf(x, mu_x, sigma_x), 'w--', lw=2)
        ax2.tick_params(colors='white')
        
        st.pyplot(fig)
        
        st.markdown(f"""
        <div class="interpretation-box">
            <strong>💡 Interpretación del TLC:</strong><br>
            Observa cómo, aunque la población original (izquierda) es muy sesgada, la distribución de las medias (derecha) se vuelve una campana perfecta (Normal) a medida que aumentas <b>n</b>. El error estándar disminuye (la campana se hace más angosta).
        </div>
        """, unsafe_allow_html=True)

    elif viz_type == "Gráfico Normal Estándar":
        z_cut = st.slider("Valor Z crítico", -3.0, 3.0, 1.96)
        
        x = np.linspace(-4, 4, 1000)
        y = stats.norm.pdf(x)
        
        fig, ax = plt.subplots(facecolor='#0e1117')
        ax.set_facecolor('#0e1117')
        ax.plot(x, y, color='white', lw=2)
        
        # Relleno del área
        mask = (x > -z_cut) & (x < z_cut)
        ax.fill_between(x, y, where=mask, color='#29b5e8', alpha=0.4, label='Nivel de Confianza')
        ax.fill_between(x, y, where=~mask, color='#ff4b4b', alpha=0.4, label='Zona de Rechazo (Alpha)')
        
        ax.axvline(x=z_cut, color='white', linestyle=':')
        ax.axvline(x=-z_cut, color='white', linestyle=':')
        
        ax.set_title(f"Curva Normal Estándar (Área central para Z={z_cut})", color='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        
        st.pyplot(fig)
        st.markdown(f"""
        <div class="interpretation-box">
            <strong>💡 Interpretación Visual:</strong><br>
            El área azul representa la probabilidad acumulada entre -{z_cut} y {z_cut}.
            Las áreas rojas ("Colas") representan el riesgo o el nivel de significancia (alpha) complementario.
        </div>
        """, unsafe_allow_html=True)
