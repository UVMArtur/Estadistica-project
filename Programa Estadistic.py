import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN E IMPORTACIONES
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="StatSuite Final",
    page_icon="🧬",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. ESTILOS CSS (SIN FLECHAS, TEMA DARK)
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
            border-bottom: 2px solid #3b82f6; /* Azul por defecto */
        }

        /* INPUTS LIMPIOS (SIN FLECHAS +/-) */
        /* Esto oculta los steppers en navegadores Webkit (Chrome, Safari) y Firefox */
        input[type=number]::-webkit-inner-spin-button, 
        input[type=number]::-webkit-outer-spin-button { 
            -webkit-appearance: none; 
            margin: 0; 
        }
        input[type=number] {
            -moz-appearance: textfield;
        }
        
        .stNumberInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
            background-color: #111 !important;
            color: white !important;
            border: 1px solid #333 !important;
            border-radius: 8px;
        }
        .stNumberInput input:focus {
            border-color: #555 !important;
        }

        /* TARJETAS DE RESULTADOS */
        .result-card {
            background-color: #ffffff;
            color: #1f2937;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
            margin-bottom: 15px;
            border-top: 5px solid;
            transition: transform 0.2s;
        }
        .result-card:hover { transform: translateY(-3px); }
        .card-label { font-size: 0.8rem; text-transform: uppercase; font-weight: 700; color: #6b7280; margin-bottom: 5px; }
        .card-value { font-size: 1.6rem; font-weight: 800; color: #111; }
        .card-sub { font-size: 0.8rem; color: #666; margin-top: 5px; font-style: italic; }

        /* COLORES */
        .border-purple { border-color: #a855f7; }
        .border-blue { border-color: #3b82f6; }
        .border-red { border-color: #ef4444; }
        .border-green { border-color: #22c55e; }

        /* CAJAS DE TEXTO */
        .simple-text {
            background: rgba(255,255,255,0.03);
            border-left: 3px solid #666;
            padding: 15px;
            border-radius: 0 8px 8px 0;
            margin-top: 15px;
            color: #ccc;
            font-size: 0.95rem;
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
            background-color: #333;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🧬 StatSuite Final")

# Función de tarjeta HTML
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🟣 Estadística Descriptiva", 
    "🔵 Inferencia Inteligente", 
    "🔴 Comparación (2 Pob)", 
    "🟢 Tamaño Muestra",
    "🧪 Laboratorio Visual (TLC)"
])

# =============================================================================
# 1. DESCRIPTIVA (Morado)
# =============================================================================
with tab1:
    st.markdown("<h3 style='color:#a855f7'>Análisis de Datos</h3>", unsafe_allow_html=True)
    
    col_in, col_out = st.columns([1, 2], gap="large")
    with col_in:
        st.info("Introduce tus números separados por comas.")
        input_desc = st.text_area("Datos Numéricos:", height=150, placeholder="Ej: 10, 15, 12, 18, 20")
        btn_calc_desc = st.button("Analizar Datos", key="btn1")

    with col_out:
        if btn_calc_desc:
            if not input_desc.strip():
                st.warning("⚠️ El campo está vacío.")
            else:
                try:
                    data = np.array([float(x.strip()) for x in input_desc.split(",") if x.strip()])
                    if len(data) > 0:
                        n = len(data)
                        media = np.mean(data)
                        mediana = np.median(data)
                        desv = np.std(data, ddof=1)
                        var = np.var(data, ddof=1)
                        ee = desv / np.sqrt(n)
                        rango = np.max(data) - np.min(data)
                        
                        # Resultados
                        c1, c2, c3 = st.columns(3)
                        c1.markdown(card("Promedio (Media)", f"{media:.2f}", "", "border-purple"), unsafe_allow_html=True)
                        c2.markdown(card("Valor Central (Mediana)", f"{mediana:.2f}", "", "border-purple"), unsafe_allow_html=True)
                        c3.markdown(card("Error Estándar (EE)", f"{ee:.4f}", "", "border-purple"), unsafe_allow_html=True)
                        
                        c4, c5, c6 = st.columns(3)
                        c4.markdown(card("Desviación Estándar (s)", f"{desv:.2f}", "Muestral", "border-purple"), unsafe_allow_html=True)
                        c5.markdown(card("Varianza ($s^2$)", f"{var:.2f}", "", "border-purple"), unsafe_allow_html=True)
                        c6.markdown(card("Rango", f"{rango:.2f}", "", "border-purple"), unsafe_allow_html=True)

                        # Interpretación del Sesgo
                        sesgo = ""
                        if abs(media - mediana) < (desv/10):
                            sesgo = "Los datos son bastante simétricos (Media ≈ Mediana)."
                        elif media > mediana:
                            sesgo = "Hay sesgo positivo (Cola a la derecha). El promedio es jalado por valores altos."
                        else:
                            sesgo = "Hay sesgo negativo (Cola a la izquierda). El promedio es jalado por valores bajos."

                        st.markdown(f"""
                        <div class="simple-text" style="border-left-color: #a855f7;">
                            <strong>Interpretación:</strong><br>
                            Con una muestra de <b>{n}</b> datos, el centro se ubica en <b>{media:.2f}</b>. 
                            La dispersión promedio es de <b>{desv:.2f}</b>.<br>
                            <em>Análisis de Forma:</em> {sesgo}
                        </div>
                        """, unsafe_allow_html=True)

                        # Histograma
                        st.write("#### Histograma de Frecuencias")
                        fig, ax = plt.subplots(figsize=(10, 3))
                        fig.patch.set_facecolor('#050505')
                        ax.set_facecolor('#111')
                        counts, bins, patches = ax.hist(data, bins='auto', color='#a855f7', edgecolor='black', alpha=0.9)
                        ax.bar_label(patches, fmt='%.0f', color='white', padding=3, fontweight='bold')
                        ax.axvline(media, color='white', linestyle='--', label='Promedio')
                        ax.legend(facecolor='#222', labelcolor='white', frameon=False)
                        ax.axis('off')
                        st.pyplot(fig)
                except:
                    st.error("Error: Revisa que solo haya números y comas.")

# =============================================================================
# 2. INFERENCIA INTELIGENTE (Azul)
# =============================================================================
with tab2:
    st.markdown("<h3 style='color:#3b82f6'>Inferencia de Una Población</h3>", unsafe_allow_html=True)
    st.write("El sistema detecta automáticamente si usar Z o T según los datos ingresados.")

    # Selector de Tipo
    tipo_dato = st.radio("¿Qué tipo de dato tienes?", ["Promedio (Media)", "Porcentaje (Proporción)", "Posición Individual (Z)"], horizontal=True)
    st.markdown("---")

    # --- LÓGICA MEDIA ---
    if tipo_dato == "Promedio (Media)":
        c1, c2, c3 = st.columns(3)
        media = c1.number_input("Promedio Muestral ($\overline{x}$)", step=0.0)
        n = c2.number_input("Tamaño de Muestra ($n$)", step=0.0)
        # Nivel de confianza editable
        conf = c3.number_input("Nivel de Confianza ($1-\\alpha$)", value=0.95, min_value=0.0, max_value=1.0, step=0.0, help="Por defecto es 0.95 (95%)")

        st.markdown("##### Desviación Estándar (Llena solo una)")
        col_sig, col_s = st.columns(2)
        sigma = col_sig.number_input("Poblacional ($\sigma$) -> Activa Z", step=0.0, help="Usa esta si conoces la desviación histórica.")
        s = col_s.number_input("Muestral ($s$) -> Activa T", step=0.0, help="Usa esta si calculaste la desviación de la muestra actual.")

        st.markdown("##### Prueba de Hipótesis (Opcional)")
        mu_hyp = st.number_input("Valor Hipotético ($\mu_0$)", value=0.0, step=0.0, help="Si llenas esto (y es diferente a 0), se calculará la prueba de hipótesis.")

        if st.button("Calcular Inferencia", key="btn_inf_smart"):
            if n > 1:
                se = 0
                dist_label = ""
                margen = 0
                
                # Detectar Z o T
                if sigma > 0:
                    se = sigma / np.sqrt(n)
                    z_val = stats.norm.ppf((1 + conf)/2)
                    margen = z_val * se
                    dist_label = "Normal (Z) - Sigma Conocida"
                    test_stat = (media - mu_hyp) / se if mu_hyp != 0 else 0
                    p_val = 2 * (1 - stats.norm.cdf(abs(test_stat)))
                elif s > 0:
                    se = s / np.sqrt(n)
                    if n >= 30:
                        z_val = stats.norm.ppf((1 + conf)/2)
                        margen = z_val * se
                        dist_label = "Normal (Z) - Muestra Grande"
                        test_stat = (media - mu_hyp) / se if mu_hyp != 0 else 0
                        p_val = 2 * (1 - stats.norm.cdf(abs(test_stat)))
                    else:
                        t_val = stats.t.ppf((1 + conf)/2, df=n-1)
                        margen = t_val * se
                        dist_label = "T-Student - Muestra Pequeña"
                        test_stat = (media - mu_hyp) / se if mu_hyp != 0 else 0
                        p_val = 2 * (1 - stats.t.cdf(abs(test_stat), df=n-1))
                else:
                    st.error("⚠️ Debes ingresar alguna desviación ($\sigma$ o $s$).")
                    st.stop()

                # Mostrar Resultados Intervalo
                c_res1, c_res2 = st.columns(2)
                c_res1.markdown(card("Límite Inferior", f"{media - margen:.4f}", "", "border-blue"), unsafe_allow_html=True)
                c_res2.markdown(card("Límite Superior", f"{media + margen:.4f}", "", "border-blue"), unsafe_allow_html=True)

                st.markdown(f"""
                <div class="simple-text" style="border-left-color: #3b82f6;">
                    <strong>Interpretación del Intervalo:</strong><br>
                    Con un {conf*100:.1f}% de confianza, el verdadero promedio poblacional está entre <b>{media - margen:.4f}</b> y <b>{media + margen:.4f}</b>.<br>
                    <em>Método usado: {dist_label}. Error Estándar: {se:.4f}.</em>
                </div>
                """, unsafe_allow_html=True)

                # Mostrar Resultados Hipótesis (Si aplica)
                if mu_hyp != 0:
                    st.markdown("#### Resultado de Prueba de Hipótesis")
                    st.write(f"Hipótesis Nula $H_0: \mu = {mu_hyp}$")
                    
                    alpha_calc = 1 - conf
                    conclusion = "Rechazar $H_0$ (Diferencia Significativa)" if p_val < alpha_calc else "No Rechazar $H_0$ (Sin Diferencia Significativa)"
                    color_hyp = "border-red" if p_val < alpha_calc else "border-green"
                    
                    h1, h2 = st.columns(2)
                    h1.markdown(card("Estadístico de Prueba", f"{test_stat:.4f}", "Z o T calculado", "border-blue"), unsafe_allow_html=True)
                    h2.markdown(card("Valor P (P-Value)", f"{p_val:.4f}", conclusion, color_hyp), unsafe_allow_html=True)

                # Gráfico
                fig, ax = plt.subplots(figsize=(8, 2))
                fig.patch.set_facecolor('#050505')
                ax.set_facecolor('#050505')
                x_axis = np.linspace(media - 4*se, media + 4*se, 200)
                y_axis = stats.norm.pdf(x_axis, media, se) # Aprox normal para visualización
                ax.plot(x_axis, y_axis, color='#3b82f6', lw=2)
                
                # Sombrear Area
                ax.fill_between(x_axis, y_axis, where=((x_axis >= media-margen) & (x_axis <= media+margen)), color='#3b82f6', alpha=0.3)
                ax.axvline(media, color='white', linestyle=':')
                ax.axis('off')
                st.pyplot(fig)

    # --- LÓGICA PROPORCIÓN ---
    elif tipo_dato == "Porcentaje (Proporción)":
        c1, c2, c3 = st.columns(3)
        prop = c1.number_input("Proporción ($p$) [Ej: 0.5]", 0.0, 1.0, step=0.0)
        n = c2.number_input("Tamaño de Muestra ($n$)", step=0.0)
        conf = c3.number_input("Nivel de Confianza ($1-\\alpha$)", value=0.95, step=0.0)

        p_hyp = st.number_input("Proporción Hipotética ($p_0$)", value=0.0, step=0.0)

        if st.button("Calcular Intervalo"):
            if n > 0:
                q = 1 - prop
                se = np.sqrt((prop*q)/n)
                z_val = stats.norm.ppf((1+conf)/2)
                margen = z_val * se
                
                c1, c2 = st.columns(2)
                c1.markdown(card("Límite Inferior", f"{(prop-margen)*100:.2f}%", f"{prop-margen:.4f}", "border-blue"), unsafe_allow_html=True)
                c2.markdown(card("Límite Superior", f"{(prop+margen)*100:.2f}%", f"{prop+margen:.4f}", "border-blue"), unsafe_allow_html=True)

                if p_hyp > 0:
                    se_hyp = np.sqrt((p_hyp*(1-p_hyp))/n)
                    z_stat = (prop - p_hyp) / se_hyp
                    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                    st.markdown("#### Prueba de Hipótesis")
                    st.write(f"Probabilidad (Valor P): {p_val:.4f}")

    # --- LÓGICA Z SCORE ---
    elif tipo_dato == "Posición Individual (Z)":
        c1, c2, c3 = st.columns(3)
        val = c1.number_input("Valor a Evaluar ($x$)", step=0.0)
        mu = c2.number_input("Promedio Población ($\mu$)", step=0.0)
        sig = c3.number_input("Desviación Población ($\sigma$)", step=0.0)
        
        if st.button("Calcular Z"):
            if sig != 0:
                z = (val - mu) / sig
                st.markdown(card("Puntaje Z", f"{z:.4f}", "Desviaciones estándar", "border-blue"), unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="simple-text" style="border-left-color: #3b82f6;">
                    El dato {val} se encuentra a <b>{abs(z):.2f}</b> desviaciones estándar {'por encima' if z>0 else 'por debajo'} del promedio.
                </div>
                """, unsafe_allow_html=True)

# =============================================================================
# 3. COMPARACIÓN (Rojo)
# =============================================================================
with tab3:
    st.markdown("<h3 style='color:#ef4444'>Comparación de Dos Grupos</h3>", unsafe_allow_html=True)
    
    opcion = st.selectbox("Seleccione Análisis:", ["Diferencia de Medias", "Diferencia de Proporciones"])
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    
    if opcion == "Diferencia de Medias":
        with col_a:
            st.write("🅰️ **Grupo 1**")
            m1 = st.number_input("Media 1 ($\overline{x}_1$)", step=0.0)
            s1 = st.number_input("Desviación 1 ($s_1$)", step=0.0)
            n1 = st.number_input("Tamaño 1 ($n_1$)", step=0.0)
        with col_b:
            st.write("🅱️ **Grupo 2**")
            m2 = st.number_input("Media 2 ($\overline{x}_2$)", step=0.0)
            s2 = st.number_input("Desviación 2 ($s_2$)", step=0.0)
            n2 = st.number_input("Tamaño 2 ($n_2$)", step=0.0)
            
        alpha = st.number_input("Nivel de Significancia (Alpha $\\alpha$)", value=0.05, step=0.0, help="Nivel de riesgo aceptado, usualmente 0.05")

        if st.button("Comparar Grupos"):
            if n1 > 1 and n2 > 1:
                se = np.sqrt((s1**2/n1) + (s2**2/n2))
                t_stat = (m1 - m2) / se
                df = n1 + n2 - 2
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                
                conclusion = "DIFERENCIA SIGNIFICATIVA" if p_val < alpha else "NO HAY DIFERENCIA SIGNIFICATIVA"
                color = "border-red" if p_val < alpha else "border-green"
                
                c1, c2 = st.columns(2)
                c1.markdown(card("Diferencia (Media 1 - Media 2)", f"{m1-m2:.2f}", "", "border-red"), unsafe_allow_html=True)
                c2.markdown(card("Valor P", f"{p_val:.4f}", conclusion, color), unsafe_allow_html=True)

    else: # Proporciones
        with col_a:
            st.write("🅰️ **Grupo 1**")
            x1 = st.number_input("Éxitos 1 ($x_1$)", step=0.0)
            nt1 = st.number_input("Total 1 ($n_1$)", step=0.0)
        with col_b:
            st.write("🅱️ **Grupo 2**")
            x2 = st.number_input("Éxitos 2 ($x_2$)", step=0.0)
            nt2 = st.number_input("Total 2 ($n_2$)", step=0.0)
            
        alpha = st.number_input("Alpha ($\alpha$)", value=0.05, step=0.0)
        
        if st.button("Comparar Porcentajes"):
            if nt1 > 0 and nt2 > 0:
                p1 = x1/nt1
                p2 = x2/nt2
                pp = (x1+x2)/(nt1+nt2)
                se = np.sqrt(pp*(1-pp)*(1/nt1 + 1/nt2))
                z = (p1 - p2) / se
                p_val = 2 * (1 - stats.norm.cdf(abs(z)))
                
                conclusion = "DIFERENCIA SIGNIFICATIVA" if p_val < alpha else "NO HAY DIFERENCIA"
                color = "border-red" if p_val < alpha else "border-green"
                
                c1, c2 = st.columns(2)
                c1.markdown(card("Diferencia %", f"{(p1-p2)*100:.2f}%", "", "border-red"), unsafe_allow_html=True)
                c2.markdown(card("Valor P", f"{p_val:.4f}", conclusion, color), unsafe_allow_html=True)

# =============================================================================
# 4. TAMAÑO DE MUESTRA (Verde)
# =============================================================================
with tab4:
    st.markdown("<h3 style='color:#22c55e'>Calculadora de Muestra ($n$)</h3>", unsafe_allow_html=True)
    
    target = st.radio("Objetivo:", ["Estimar Promedio", "Estimar Proporción"])
    c1, c2 = st.columns(2)
    error = c1.number_input("Margen de Error (ME)", value=0.05, step=0.0, format="%.4f")
    conf = c2.number_input("Confianza ($1-\\alpha$)", value=0.95, step=0.0)
    
    if target == "Estimar Promedio":
        sigma = st.number_input("Desviación Estimada ($\sigma$)", value=10.0, step=0.0)
        if st.button("Calcular N"):
            z = stats.norm.ppf((1+conf)/2)
            if error > 0:
                n = (z**2 * sigma**2) / error**2
                st.markdown(card("Tamaño de Muestra ($n$)", f"{math.ceil(n)}", "Personas/Datos", "border-green"), unsafe_allow_html=True)
    else:
        p_est = st.number_input("Proporción Estimada ($p$)", value=0.5, step=0.0)
        if st.button("Calcular N"):
            z = stats.norm.ppf((1+conf)/2)
            if error > 0:
                n = (z**2 * p_est * (1-p_est)) / error**2
                st.markdown(card("Tamaño de Muestra ($n$)", f"{math.ceil(n)}", "Personas/Datos", "border-green"), unsafe_allow_html=True)

# =============================================================================
# 5. LABORATORIO VISUAL (TLC y Conceptos)
# =============================================================================
with tab5:
    st.markdown("<h3 style='color:#ffffff'>Laboratorio Visual</h3>", unsafe_allow_html=True)
    
    tool = st.selectbox("Seleccione Simulación:", ["Teorema del Límite Central (TLC)", "Comportamiento Error Estándar"])
    
    if tool == "Teorema del Límite Central (TLC)":
        st.info("Simula cómo el promedio de muchas muestras forma una campana, sin importar la población original.")
        c1, c2 = st.columns(2)
        n_sim = c1.number_input("Tamaño de cada muestra ($n$)", value=30, step=0.0)
        reps = c2.number_input("Cantidad de muestras (Repeticiones)", value=1000, step=0.0)
        
        if st.button("Simular"):
            # Población muy sesgada (Exponencial)
            pop = np.random.exponential(scale=1.0, size=10000)
            means = [np.mean(np.random.choice(pop, int(n_sim))) for _ in range(int(reps))]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            fig.patch.set_facecolor('#050505')
            
            ax1.set_facecolor('#111')
            ax1.hist(pop, bins=30, color='#444')
            ax1.set_title("Población Original (Sesgada)", color='white')
            ax1.axis('off')
            
            ax2.set_facecolor('#111')
            ax2.hist(means, bins=30, color='#22c55e', alpha=0.8)
            ax2.set_title(f"Distribución de Medias (n={int(n_sim)})", color='white')
            ax2.axis('off')
            
            st.pyplot(fig)
            st.success("¡Observe cómo la gráfica derecha se vuelve una campana perfecta!")

    elif tool == "Comportamiento Error Estándar":
        st.info("Mira cómo el error disminuye al aumentar la muestra.")
        sigma_sim = st.number_input("Desviación Poblacional Simulada", value=10.0, step=0.0)
        
        if st.button("Generar Curva"):
            ns = np.arange(1, 200)
            ees = sigma_sim / np.sqrt(ns)
            
            fig, ax = plt.subplots(figsize=(8, 3))
            fig.patch.set_facecolor('#050505')
            ax.set_facecolor('#111')
            
            ax.plot(ns, ees, color='#3b82f6', lw=3)
            ax.set_xlabel("Tamaño de Muestra (n)", color='white')
            ax.set_ylabel("Error Estándar", color='white')
            ax.grid(color='#333', linestyle='--')
            ax.tick_params(colors='white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            
            st.pyplot(fig)
