import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
import re

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="StatSuite - Calculadora Estadística",
    page_icon="🧬",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. ESTILOS CSS (TEMA VISUAL IDENTICO AL PDF)
# -----------------------------------------------------------------------------
# Este bloque CSS fuerza el modo oscuro de fondo, pero crea tarjetas blancas
# para los resultados con los bordes de colores (Morado, Azul, Rojo, Verde)
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

        /* FONDO PRINCIPAL OSCURO (#050505) */
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
            background-color: #050505;
            color: #e5e7eb;
        }
        
        .stApp { background-color: #050505; }
        
        /* OCULTAR ELEMENTOS INNECESARIOS */
        header, footer {visibility: hidden;}

        /* PESTAÑAS (TABS) ESTILIZADAS */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: #0a0a0a;
            padding: 10px;
            border-radius: 12px;
            border: 1px solid #222;
        }
        .stTabs [data-baseweb="tab"] {
            height: 45px;
            border-radius: 6px;
            font-weight: 600;
            color: #888;
            border: none;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #222;
            color: white;
            border-bottom: 2px solid #3b82f6;
        }

        /* INPUTS NUMÉRICOS (Ocultar flechas y estilo dark) */
        input[type=number]::-webkit-inner-spin-button, 
        input[type=number]::-webkit-outer-spin-button { 
            -webkit-appearance: none; 
            margin: 0; 
        }
        input[type=number] { -moz-appearance: textfield; }
        
        .stNumberInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
            background-color: #111 !important;
            color: white !important;
            border: 1px solid #333 !important;
            border-radius: 8px;
        }
        .stNumberInput input:focus { border-color: #555 !important; }

        /* TARJETAS DE RESULTADOS (Estilo visual clave del PDF) */
        .result-card {
            background-color: #ffffff; /* Fondo blanco */
            color: #1f2937;            /* Texto oscuro */
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
            margin-bottom: 15px;
            border-top: 5px solid;     /* Borde superior grueso de color */
            transition: transform 0.2s;
        }
        .result-card:hover { transform: translateY(-3px); }
        
        .card-label { 
            font-size: 0.85rem; 
            text-transform: uppercase; 
            font-weight: 700; 
            color: #6b7280; 
            margin-bottom: 5px; 
        }
        .card-value { 
            font-size: 1.8rem; 
            font-weight: 800; 
            color: #111; 
        }
        .card-sub { 
            font-size: 0.8rem; 
            color: #666; 
            margin-top: 5px; 
            font-style: italic; 
        }

        /* COLORES DE BORDES (Coinciden con las secciones del PDF) */
        .border-purple { border-color: #a855f7; } /* Descriptiva */
        .border-blue { border-color: #3b82f6; }   /* Inferencia */
        .border-red { border-color: #ef4444; }    /* Comparación */
        .border-green { border-color: #22c55e; }  /* Muestra */

        /* CAJAS DE TEXTO EXPLICATIVO */
        .simple-text {
            background: rgba(255,255,255,0.05);
            border-left: 4px solid #666;
            padding: 15px;
            border-radius: 0 8px 8px 0;
            margin-top: 15px;
            color: #ddd;
            font-size: 0.95rem;
        }

        /* BOTONES PERSONALIZADOS */
        div.stButton > button {
            width: 100%;
            border-radius: 8px;
            font-weight: 700;
            padding: 12px;
            background-color: #222;
            color: white;
            border: 1px solid #444;
            transition: all 0.3s;
        }
        div.stButton > button:hover {
            border-color: #fff;
            background-color: #333;
            transform: scale(1.02);
        }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.title("🧬 StatSuite Final")

# Función auxiliar para generar el HTML de las tarjetas
def card(label, value, sub="", color="border-blue"):
    return f"""
    <div class="result-card {color}">
        <div class="card-label">{label}</div>
        <div class="card-value">{value}</div>
        <div class="card-sub">{sub}</div>
    </div>
    """

# -----------------------------------------------------------------------------
# ESTRUCTURA DE PESTAÑAS PRINCIPALES
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🟣 Estadística Descriptiva", 
    "🔵 Inferencia Inteligente", 
    "🔴 Comparación (2 Pob)", 
    "🟢 Tamaño Muestra",
    "🧪 Laboratorio Visual"
])

# =============================================================================
# [cite_start]PESTAÑA 1: DESCRIPTIVA (Morado) [cite: 1, 2, 3]
# =============================================================================
with tab1:
    st.markdown("<h3 style='color:#a855f7'>Análisis de Datos</h3>", unsafe_allow_html=True)
    
    col_in, col_out = st.columns([1, 2], gap="large")
    with col_in:
        st.info("Introduce tus números. Usa PUNTO (.) para decimales. Separa con comas, espacios o saltos de línea.")
        # Entrada de texto grande como en el PDF
        input_desc = st.text_area("Datos Numéricos:", height=150, placeholder="Ej: 10.5, 15, 12.0; 18 20")
        # Botón morado (por contexto de la pestaña)
        btn_calc_desc = st.button("Analizar Datos", key="btn1")

    with col_out:
        if btn_calc_desc:
            if not input_desc.strip():
                st.warning("⚠️ El campo está vacío.")
            else:
                # Validación: Prohibir comas decimales (ej 10,5)
                if re.search(r'\d+,\d+', input_desc):
                    st.error("Por favor usa PUNTO (.) para decimales. Reemplaza '10,5' por '10.5'.")
                else:
                    try:
                        # Procesamiento robusto de texto a números
                        parts = re.split(r'[,\;\s]+', input_desc.strip())
                        tokens = [p for p in parts if p != '']
                        nums = []
                        valid = True
                        for t in tokens:
                            if re.fullmatch(r'[-+]?(?:\d+|\d+\.\d+|\.\d+)', t):
                                nums.append(float(t))
                            else:
                                st.error(f"Token inválido: '{t}'. Usa solo números.")
                                valid = False
                                break
                        
                        if valid and len(nums) > 0:
                            data = np.array(nums, dtype=float)
                            n = data.size
                            media = float(np.mean(data))
                            mediana = float(np.median(data))
                            
                            # Lógica para n < 2
                            if n >= 2:
                                desv = float(np.std(data, ddof=1))
                                var = float(np.var(data, ddof=1))
                            else:
                                desv = float(np.std(data, ddof=0))
                                var = float(np.var(data, ddof=0))
                                
                            ee = desv / math.sqrt(n) if n > 0 else 0.0
                            rango = float(np.max(data) - np.min(data)) if n > 0 else 0.0

                            # -- Visualización de Tarjetas (Fila 1) --
                            c1, c2, c3 = st.columns(3)
                            c1.markdown(card("Promedio (Media)", f"{media:.2f}", "", "border-purple"), unsafe_allow_html=True)
                            c2.markdown(card("Mediana", f"{mediana:.2f}", "", "border-purple"), unsafe_allow_html=True)
                            c3.markdown(card("Error Estándar", f"{ee:.4f}", "", "border-purple"), unsafe_allow_html=True)
                            
                            # -- Visualización de Tarjetas (Fila 2) --
                            c4, c5, c6 = st.columns(3)
                            c4.markdown(card("Desviación Estándar", f"{desv:.2f}", "Muestral" if n>=2 else "Poblacional", "border-purple"), unsafe_allow_html=True)
                            c5.markdown(card("Varianza", f"{var:.2f}", "", "border-purple"), unsafe_allow_html=True)
                            c6.markdown(card("Rango", f"{rango:.2f}", "", "border-purple"), unsafe_allow_html=True)

                            # Interpretación de Sesgo
                            sesgo = ""
                            if desv == 0:
                                sesgo = "Datos idénticos (sin variación)."
                            elif abs(media - mediana) < (desv/10):
                                sesgo = "Distribución simétrica (Media ≈ Mediana)."
                            elif media > mediana:
                                sesgo = "Sesgo positivo (Cola a la derecha)."
                            else:
                                sesgo = "Sesgo negativo (Cola a la izquierda)."

                            st.markdown(f"""
                            <div class="simple-text" style="border-left-color: #a855f7;">
                                <strong>Interpretación:</strong><br>
                                Con n=<b>{n}</b>, el centro es <b>{media:.2f}</b> y la dispersión es <b>{desv:.2f}</b>.<br>
                                <em>Forma:</em> {sesgo}
                            </div>
                            """, unsafe_allow_html=True)

                            # Histograma Estilizado Dark
                            st.write("#### Histograma de Frecuencias")
                            fig, ax = plt.subplots(figsize=(10, 3))
                            fig.patch.set_facecolor('#050505')
                            ax.set_facecolor('#111')
                            counts, bins, patches = ax.hist(data, bins='auto', color='#a855f7', edgecolor='black', alpha=0.9)
                            ax.axvline(media, color='white', linestyle='--', label='Promedio')
                            ax.legend(facecolor='#222', labelcolor='white', frameon=False)
                            ax.axis('off') # Ocultar ejes para estilo minimalista
                            st.pyplot(fig)
                            
                    except Exception as e:
                        st.error(f"Error: {e}")

# =============================================================================
# [cite_start]PESTAÑA 2: INFERENCIA INTELIGENTE (Azul) [cite: 51, 68, 81]
# =============================================================================
with tab2:
    st.markdown("<h3 style='color:#3b82f6'>Inferencia de Una Población</h3>", unsafe_allow_html=True)
    
    # Sub-navegación interna
    tipo_dato = st.radio("¿Qué tipo de dato tienes?", ["Promedio (Media)", "Porcentaje (Proporción)", "Posición Individual (Z)"], horizontal=True)
    st.markdown("---")

    # CASO 1: MEDIA
    if tipo_dato == "Promedio (Media)":
        c1, c2, c3 = st.columns(3)
        media = c1.number_input("Promedio Muestral (x̄)", step=0.01, format="%.4f")
        n = c2.number_input("Tamaño de Muestra (n)", value=30.0, step=1.0)
        conf = c3.number_input("Nivel de Confianza (0.90 - 0.99)", value=0.95, step=0.01)

        col_sig, col_s = st.columns(2)
        sigma = col_sig.number_input("Poblacional (σ) -> Usa Z", step=0.01, format="%.4f", help="Si conoces la historia poblacional")
        s = col_s.number_input("Muestral (s) -> Usa T (o Z aprox)", step=0.01, format="%.4f", help="Calculado de estos datos")

        realizar_prueba = st.checkbox("Calcular prueba de hipótesis (H0)", value=False)
        mu_hyp = 0.0
        if realizar_prueba:
            mu_hyp = st.number_input("Valor Hipotético (μ0)", value=0.0, step=0.01)

        if st.button("Calcular Inferencia", key="btn_inf"):
            try:
                n_int = int(n)
                if n_int <= 0: raise ValueError("n > 0")
                
                # Lógica de decisión Z vs T
                se = 0
                dist_label = ""
                margen = 0
                
                if sigma > 0:
                    se = sigma / math.sqrt(n_int)
                    z_val = stats.norm.ppf((1 + conf)/2)
                    margen = z_val * se
                    dist_label = "Normal (Z) - Sigma Conocida"
                    test_func = stats.norm
                elif s > 0:
                    se = s / math.sqrt(n_int)
                    if n_int >= 30:
                        z_val = stats.norm.ppf((1 + conf)/2)
                        margen = z_val * se
                        dist_label = "Normal (Z) - Muestra Grande"
                        test_func = stats.norm
                    else:
                        t_val = stats.t.ppf((1 + conf)/2, df=n_int-1)
                        margen = t_val * se
                        dist_label = "T-Student - Muestra Pequeña"
                        test_func = stats.t
                else:
                    st.error("Ingresa alguna desviación (σ o s).")
                    st.stop()

                # Resultados Intervalo
                c_res1, c_res2 = st.columns(2)
                c_res1.markdown(card("Límite Inferior", f"{media - margen:.4f}", "", "border-blue"), unsafe_allow_html=True)
                c_res2.markdown(card("Límite Superior", f"{media + margen:.4f}", "", "border-blue"), unsafe_allow_html=True)
                
                st.markdown(f"<div class='simple-text' style='border-left-color:#3b82f6'>Confianza: {conf*100:.1f}%<br>Método: {dist_label}</div>", unsafe_allow_html=True)

                # Resultados Hipótesis
                if realizar_prueba:
                    st.markdown("#### Prueba de Hipótesis")
                    test_stat = (media - mu_hyp) / se
                    if dist_label.startswith("T-Student"):
                        p_val = 2 * (1 - test_func.cdf(abs(test_stat), df=n_int-1))
                    else:
                        p_val = 2 * (1 - test_func.cdf(abs(test_stat)))
                        
                    alpha = 1 - conf
                    sig = "Diferencia Significativa (Rechazar H0)" if p_val < alpha else "No Significativa"
                    col_h = "border-red" if p_val < alpha else "border-green"
                    
                    h1, h2 = st.columns(2)
                    h1.markdown(card("Estadístico", f"{test_stat:.4f}", "", "border-blue"), unsafe_allow_html=True)
                    h2.markdown(card("Valor P", f"{p_val:.4f}", sig, col_h), unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error en cálculo: {e}")

    # CASO 2: PROPORCIÓN
    elif tipo_dato == "Porcentaje (Proporción)":
        c1, c2, c3 = st.columns(3)
        prop = c1.number_input("Proporción (p) [0.0 - 1.0]", value=0.5, step=0.01)
        n = c2.number_input("Muestra (n)", value=100.0, step=1.0)
        conf = c3.number_input("Confianza", value=0.95, step=0.01)
        
        if st.button("Calcular Intervalo Prop"):
            n_int = int(n)
            se = math.sqrt((prop * (1-prop)) / n_int)
            z = stats.norm.ppf((1+conf)/2)
            margen = z * se
            
            c1, c2 = st.columns(2)
            c1.markdown(card("Límite Inf %", f"{(prop-margen)*100:.2f}%", f"{max(0, prop-margen):.4f}", "border-blue"), unsafe_allow_html=True)
            c2.markdown(card("Límite Sup %", f"{(prop+margen)*100:.2f}%", f"{min(1, prop+margen):.4f}", "border-blue"), unsafe_allow_html=True)

    # CASO 3: PUNTUACIÓN Z
    elif tipo_dato == "Posición Individual (Z)":
        c1, c2, c3 = st.columns(3)
        val = c1.number_input("Valor (x)", 0.0)
        mu = c2.number_input("Media (μ)", 0.0)
        sig = c3.number_input("Desviación (σ)", 1.0)
        if st.button("Calcular Z"):
            z = (val - mu) / sig
            st.markdown(card("Puntaje Z", f"{z:.4f}", "Desviaciones estándar", "border-blue"), unsafe_allow_html=True)

# =============================================================================
# [cite_start]PESTAÑA 3: COMPARACIÓN (Rojo) [cite: 142, 175]
# =============================================================================
with tab3:
    st.markdown("<h3 style='color:#ef4444'>Comparación de Dos Grupos</h3>", unsafe_allow_html=True)
    opcion = st.selectbox("Tipo de Análisis:", ["Diferencia de Medias", "Diferencia de Proporciones"])
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    
    if opcion == "Diferencia de Medias":
        with col_a:
            st.write("🅰️ **Grupo 1**")
            m1 = st.number_input("Media 1", step=0.1)
            s1 = st.number_input("Desviación 1", step=0.1, value=1.0)
            n1 = st.number_input("Tamaño 1", value=30.0)
        with col_b:
            st.write("🅱️ **Grupo 2**")
            m2 = st.number_input("Media 2", step=0.1)
            s2 = st.number_input("Desviación 2", step=0.1, value=1.0)
            n2 = st.number_input("Tamaño 2", value=30.0)
            
        alpha = st.number_input("Significancia (α)", value=0.05)

        if st.button("Comparar Grupos"):
            se = math.sqrt((s1**2/n1) + (s2**2/n2))
            t_stat = (m1 - m2) / se
            df = n1 + n2 - 2
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            
            res_txt = "DIFERENCIA SIGNIFICATIVA" if p_val < alpha else "NO HAY DIFERENCIA"
            color = "border-red" if p_val < alpha else "border-green"
            
            k1, k2 = st.columns(2)
            k1.markdown(card("Diferencia", f"{(m1-m2):.2f}", f"t = {t_stat:.3f}", "border-red"), unsafe_allow_html=True)
            k2.markdown(card("Valor P", f"{p_val:.4f}", res_txt, color), unsafe_allow_html=True)

    else: # Proporciones
        with col_a:
            x1 = st.number_input("Éxitos 1", 0.0)
            nt1 = st.number_input("Total 1", 100.0)
        with col_b:
            x2 = st.number_input("Éxitos 2", 0.0)
            nt2 = st.number_input("Total 2", 100.0)
        
        if st.button("Comparar %"):
            p1, p2 = x1/nt1, x2/nt2
            pp = (x1+x2)/(nt1+nt2)
            se = math.sqrt(pp*(1-pp)*(1/nt1 + 1/nt2))
            z = (p1-p2)/se
            p_val = 2*(1-stats.norm.cdf(abs(z)))
            
            res_txt = "SIGNIFICATIVA" if p_val < 0.05 else "NO SIGNIFICATIVA"
            color = "border-red" if p_val < 0.05 else "border-green"
            
            k1, k2 = st.columns(2)
            k1.markdown(card("Diferencia %", f"{(p1-p2)*100:.2f}%", "", "border-red"), unsafe_allow_html=True)
            k2.markdown(card("Valor P", f"{p_val:.4f}", res_txt, color), unsafe_allow_html=True)

# =============================================================================
# [cite_start]PESTAÑA 4: TAMAÑO MUESTRA (Verde) [cite: 45]
# =============================================================================
with tab4:
    st.markdown("<h3 style='color:#22c55e'>Calculadora de Muestra (n)</h3>", unsafe_allow_html=True)
    target = st.radio("Objetivo:", ["Estimar Promedio", "Estimar Proporción"])
    
    col1, col2 = st.columns(2)
    error = col1.number_input("Margen de Error (E)", value=0.05, format="%.4f")
    conf = col2.number_input("Confianza (1-α)", value=0.95)
    
    if target == "Estimar Promedio":
        sigma = st.number_input("Desviación Estimada (σ)", value=10.0)
        if st.button("Calcular N Promedio"):
            z = stats.norm.ppf((1+conf)/2)
            n_res = (z**2 * sigma**2) / error**2
            st.markdown(card("Muestra Necesaria", f"{math.ceil(n_res)}", "Registros", "border-green"), unsafe_allow_html=True)
    else:
        p_est = st.number_input("Proporción Estimada (p)", value=0.5)
        if st.button("Calcular N Proporción"):
            z = stats.norm.ppf((1+conf)/2)
            n_res = (z**2 * p_est * (1-p_est)) / error**2
            st.markdown(card("Muestra Necesaria", f"{math.ceil(n_res)}", "Encuestados", "border-green"), unsafe_allow_html=True)

# =============================================================================
# PESTAÑA 5: LABORATORIO VISUAL (TLC)
# =============================================================================
with tab5:
    st.markdown("<h3 style='color:#ffffff'>Laboratorio Visual</h3>", unsafe_allow_html=True)
    st.info("Simulación del Teorema del Límite Central")
    
    col1, col2 = st.columns(2)
    n_sim = int(col1.number_input("Tamaño de muestra (n)", value=30, min_value=1))
    reps = int(col2.number_input("Repeticiones", value=500, min_value=10))
    
    if st.button("Simular TLC"):
        # Población exponencial (sesgada)
        pop = np.random.exponential(scale=1.0, size=10000)
        means = [np.mean(np.random.choice(pop, n_sim)) for _ in range(reps)]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        fig.patch.set_facecolor('#050505')
        
        ax1.set_facecolor('#111')
        ax1.hist(pop, bins=30, color='#444')
        ax1.set_title("Población Original (Sesgada)", color='white')
        ax1.axis('off')
        
        ax2.set_facecolor('#111')
        ax2.hist(means, bins=30, color='#22c55e', alpha=0.8)
        ax2.set_title(f"Distribución de Medias (Normal)", color='white')
        ax2.axis('off')
        
        st.pyplot(fig)
