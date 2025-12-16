import re
import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
from collections import Counter

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN E IMPORTACIONES
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Calculadora de Estadística Pro",
    page_icon="📊",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. ESTILOS CSS (TEMA DARK + FUENTE INTER)
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* RESET GENERAL */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #000000;
            color: #ffffff;
        }
        
        header, footer {visibility: hidden;}
        .stApp { 
            background-color: #000000;
        }

        /* TÍTULO PRINCIPAL */
        h1 {
            font-size: 2.5rem !important;
            font-weight: 800 !important;
            color: #ffffff !important;
            margin-bottom: 2rem !important;
            letter-spacing: -1px;
            text-align: center;
        }

        /* PESTAÑAS ESTILIZADAS */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            background-color: transparent;
            padding: 0;
            border-radius: 0;
            border: none;
            border-bottom: 1px solid #333;
            margin-bottom: 2rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 60px;
            border-radius: 0;
            font-weight: 500;
            font-size: 0.95rem;
            color: #666;
            border: none;
            transition: all 0.3s;
            background-color: transparent;
            padding: 0 2rem;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #999;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: transparent;
            color: white;
            border-bottom: 3px solid;
        }
        
        /* Colores específicos por pestaña */
        .stTabs [data-baseweb="tab"]:nth-child(1)[aria-selected="true"] { border-bottom-color: #a855f7; }
        .stTabs [data-baseweb="tab"]:nth-child(2)[aria-selected="true"] { border-bottom-color: #3b82f6; }
        .stTabs [data-baseweb="tab"]:nth-child(3)[aria-selected="true"] { border-bottom-color: #ef4444; }
        .stTabs [data-baseweb="tab"]:nth-child(4)[aria-selected="true"] { border-bottom-color: #22c55e; }
        .stTabs [data-baseweb="tab"]:nth-child(5)[aria-selected="true"] { border-bottom-color: #f59e0b; }

        /* Ocultar flechas de los inputs numéricos */
        input[type=number]::-webkit-inner-spin-button, 
        input[type=number]::-webkit-outer-spin-button { 
            -webkit-appearance: none; 
            margin: 0; 
        }
        input[type=number] { -moz-appearance: textfield; }
        
        /* ESTILO DE LOS CAMPOS */
        .stNumberInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
            background-color: #0a0a0a !important;
            color: white !important;
            border: 1px solid #222 !important;
            border-radius: 12px;
            font-size: 1rem;
        }
        .stNumberInput input:focus, .stTextArea textarea:focus {
            border-color: #444 !important;
            box-shadow: 0 0 0 1px #444 !important;
        }

        /* Labels de inputs */
        .stNumberInput label, .stTextArea label, .stSelectbox label {
            color: #999 !important;
            font-weight: 500 !important;
            font-size: 0.9rem !important;
        }

        /* TARJETAS DE RESULTADOS */
        .result-card {
            background: linear-gradient(135deg, #ffffff 0%, #f5f5f5 100%);
            color: #000000;
            padding: 2rem;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
            margin-bottom: 1rem;
            border: 1px solid rgba(255,255,255,0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .result-card:hover { 
            transform: translateY(-5px);
            box-shadow: 0 12px 48px rgba(0,0,0,0.6);
        }
        .card-label { 
            font-size: 0.75rem; 
            text-transform: uppercase; 
            font-weight: 700; 
            color: #666; 
            margin-bottom: 0.75rem;
            letter-spacing: 1px;
        }
        .card-value { 
            font-size: 2.5rem; 
            font-weight: 800; 
            color: #000;
            line-height: 1;
        }
        .card-sub { 
            font-size: 0.85rem; 
            color: #666; 
            margin-top: 0.5rem; 
            font-weight: 500;
        }

        /* BORDES DE COLOR */
        .border-purple { border-top: 6px solid #a855f7; }
        .border-blue { border-top: 6px solid #3b82f6; }
        .border-red { border-top: 6px solid #ef4444; }
        .border-green { border-top: 6px solid #22c55e; }
        .border-yellow { border-top: 6px solid #f59e0b; }

        /* TEXTO EXPLICATIVO */
        .simple-text {
            background: #0a0a0a;
            border-left: 4px solid;
            padding: 1.5rem;
            border-radius: 8px;
            margin-top: 1.5rem;
            color: #ccc;
            font-size: 0.95rem;
            line-height: 1.6;
        }

        /* BOTONES */
        div.stButton > button {
            width: 100%;
            border-radius: 12px;
            font-weight: 600;
            padding: 1rem;
            background: linear-gradient(135deg, #1a1a1a 0%, #0a0a0a 100%);
            color: white;
            border: 1px solid #333;
            font-size: 1rem;
            transition: all 0.3s;
        }
        div.stButton > button:hover {
            border-color: #555;
            background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        }

        /* SECTION HEADERS */
        h3 {
            font-weight: 300 !important;
            font-size: 1.8rem !important;
            margin-bottom: 1.5rem !important;
            letter-spacing: -0.5px !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Calculadora de Estadística Pro")

# Función auxiliar de tarjeta HTML
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
    "Medidas de tendencia central", 
    "Inferencia estadística", 
    "Comparación de dos poblaciones", 
    "Tamaño de muestra",
    "Visual LAB"
])

# =============================================================================
# 1. DESCRIPTIVA (Morado)
# =============================================================================
with tab1:
    st.markdown("<h3 style='color:#a855f7'>Analizar datos</h3>", unsafe_allow_html=True)
    
    col_in, col_out = st.columns([1, 2], gap="large")
    with col_in:
        st.info("Usa PUNTO (.) para decimales. Separa números con comas, espacios o saltos de línea.")
        input_desc = st.text_area("Datos:", height=200, placeholder="Ej: 3.2, 4.5, 7.8, 9.1")
        btn_calc_desc = st.button("Analizar datos", key="btn1")

    with col_out:
        if btn_calc_desc:
            if not input_desc.strip():
                st.warning("⚠️ El campo está vacío.")
            else:
                if re.search(r'\d+,\d+', input_desc):
                    st.error("Por favor usa PUNTO (.) para decimales. Reemplaza '10,5' por '10.5'.")
                else:
                    try:
                        parts = re.split(r'[,\;\s]+', input_desc.strip())
                        tokens = [p for p in parts if p != '']
                        nums = []
                        for t in tokens:
                            if re.fullmatch(r'[-+]?(?:\d+|\d+\.\d+|\.\d+)', t):
                                nums.append(float(t))
                            else:
                                st.error(f"Token inválido: '{t}'")
                                nums = None
                                break

                        if nums:
                            data = np.array(nums, dtype=float)
                            n = data.size
                            media = float(np.mean(data))
                            mediana = float(np.median(data))
                            
                            if n >= 2:
                                desv = float(np.std(data, ddof=1))
                                var = float(np.var(data, ddof=1))
                            else:
                                desv = float(np.std(data, ddof=0))
                                var = float(np.var(data, ddof=0))
                            
                            ee = desv / math.sqrt(n) if n > 0 else 0.0

                            # Moda
                            contador = Counter(data)
                            max_freq = max(contador.values())
                            modas = [k for k, v in contador.items() if v == max_freq]
                            
                            if len(modas) == len(data):
                                moda_str = "—"
                                moda_sub = "No hay moda (valores únicos)"
                            elif len(modas) == 1:
                                moda_str = f"{modas[0]:.2f}"
                                moda_sub = f"Frecuencia: {max_freq}"
                            else:
                                moda_str = "Múltiple"
                                moda_sub = f"{len(modas)} modas encontradas"

                            # Tarjetas
                            c1, c2, c3 = st.columns(3)
                            c1.markdown(card("Promedio", f"{media:.2f}", "", "border-purple"), unsafe_allow_html=True)
                            c2.markdown(card("Mediana", f"{mediana:.2f}", "", "border-purple"), unsafe_allow_html=True)
                            c3.markdown(card("Moda", moda_str, moda_sub, "border-purple"), unsafe_allow_html=True)
                            
                            c4, c5, c6 = st.columns(3)
                            c4.markdown(card("Varianza", f"{var:.2f}", "", "border-purple"), unsafe_allow_html=True)
                            c5.markdown(card("Desviación Std", f"{desv:.2f}", "Muestral" if n>=2 else "Poblacional", "border-purple"), unsafe_allow_html=True)
                            c6.markdown(card("Error Estándar", f"{ee:.4f}", "", "border-purple"), unsafe_allow_html=True)

                            # Histograma
                            st.write("#### Histograma:")
                            fig, ax = plt.subplots(figsize=(10, 3))
                            fig.patch.set_facecolor('#000000')
                            ax.set_facecolor('#000000')
                            counts, bins, patches = ax.hist(data, bins='auto', color='#a855f7', edgecolor='black', alpha=0.9)
                            ax.axvline(media, color='white', linestyle='--', label='Promedio')
                            ax.legend(facecolor='#222', labelcolor='white', frameon=False)
                            ax.axis('off')
                            st.pyplot(fig)
                            
                    except Exception as e:
                        st.error(f"Error al procesar: {e}")

# =============================================================================
# 2. INFERENCIA (Azul)
# =============================================================================
with tab2:
    st.markdown("<h3 style='color:#3b82f6'>Inferencia de Una Población</h3>", unsafe_allow_html=True)
    
    tipo_dato = st.radio("Tipo de dato:", ["Promedio (Media)", "Porcentaje (Proporción)", "Posición Individual (Z)"], horizontal=True, key="inferencia_radio")
    st.markdown("---")

    if tipo_dato == "Promedio (Media)":
        c1, c2, c3 = st.columns(3)
        media = c1.number_input("Promedio Muestral (x̄)", step=0.01, format="%.4f")
        n = c2.number_input("Muestra (n)", value=30.0, step=1.0)
        conf = c3.number_input("Confianza (%)", value=95.0, step=1.0)

        col_sig, col_s = st.columns(2)
        sigma = col_sig.number_input("Poblacional (σ) - (Usa Z)", step=0.01, format="%.4f")
        s = col_s.number_input("Muestral (s) - (Usa T)", step=0.01, format="%.4f")

        realizar_prueba = st.checkbox("Prueba de hipótesis (H0)")
        mu_hyp = 0.0
        if realizar_prueba:
            mu_hyp = st.number_input("Valor Hipotético (μ0)", value=0.0, step=0.01)

        if st.button("Calcular Inferencia", key="btn_inf"):
            conf_dec = conf / 100.0
            n_int = int(n)
            
            # Lógica Z vs T
            if sigma > 0:
                se = sigma / math.sqrt(n_int)
                z_val = stats.norm.ppf((1 + conf_dec)/2)
                margen = z_val * se
                dist_label = "Normal (Z) - σ conocida"
                p_func = lambda x: 2 * (1 - stats.norm.cdf(abs(x)))
            elif s > 0:
                se = s / math.sqrt(n_int)
                if n_int >= 30:
                    z_val = stats.norm.ppf((1 + conf_dec)/2)
                    margen = z_val * se
                    dist_label = "Normal (Z) - n grande"
                    p_func = lambda x: 2 * (1 - stats.norm.cdf(abs(x)))
                else:
                    t_val = stats.t.ppf((1 + conf_dec)/2, df=n_int-1)
                    margen = t_val * se
                    dist_label = "T-Student - n pequeña"
                    p_func = lambda x: 2 * (1 - stats.t.cdf(abs(x), df=n_int-1))
            else:
                st.error("Ingresa una desviación (σ o s).")
                st.stop()

            c_res1, c_res2 = st.columns(2)
            c_res1.markdown(card("Límite Inferior", f"{media - margen:.4f}", "", "border-blue"), unsafe_allow_html=True)
            c_res2.markdown(card("Límite Superior", f"{media + margen:.4f}", "", "border-blue"), unsafe_allow_html=True)

            if realizar_prueba:
                test_stat = (media - mu_hyp) / se
                p_val = p_func(test_stat)
                alpha = 1 - conf_dec
                conclusion = "Rechazar H0" if p_val < alpha else "No rechazar H0"
                color_h = "border-red" if p_val < alpha else "border-green"
                
                h1, h2 = st.columns(2)
                h1.markdown(card("Estadístico", f"{test_stat:.4f}", "", "border-blue"), unsafe_allow_html=True)
                h2.markdown(card("Valor P", f"{p_val:.4f}", conclusion, color_h), unsafe_allow_html=True)

    elif tipo_dato == "Porcentaje (Proporción)":
        c1, c2, c3 = st.columns(3)
        prop = c1.number_input("Proporción (p)", value=0.5, step=0.01)
        n = c2.number_input("Muestra (n)", value=100.0, step=1.0)
        conf = c3.number_input("Confianza (%)", value=95.0, step=1.0)
        
        if st.button("Calcular Intervalo Prop"):
            n_int = int(n)
            se = math.sqrt((prop * (1-prop)) / n_int)
            z = stats.norm.ppf((1 + conf/100.0)/2)
            margen = z * se
            
            c1, c2 = st.columns(2)
            c1.markdown(card("Límite Inferior", f"{(prop-margen)*100:.2f}%", f"{prop-margen:.4f}", "border-blue"), unsafe_allow_html=True)
            c2.markdown(card("Límite Superior", f"{(prop+margen)*100:.2f}%", f"{prop+margen:.4f}", "border-blue"), unsafe_allow_html=True)

    elif tipo_dato == "Posición Individual (Z)":
        c1, c2, c3 = st.columns(3)
        val = c1.number_input("Valor (x)", 0.0)
        mu = c2.number_input("Media (μ)", 0.0)
        sig = c3.number_input("Desviación (σ)", 1.0)
        
        if st.button("Calcular Z"):
            if sig == 0: st.error("Sigma no puede ser 0")
            else:
                z = (val - mu) / sig
                st.markdown(card("Puntaje Z", f"{z:.4f}", "Desviaciones estándar", "border-blue"), unsafe_allow_html=True)

# =============================================================================
# 3. COMPARACIÓN (Rojo)
# =============================================================================
with tab3:
    st.markdown("<h3 style='color:#ef4444'>Comparación de Dos Grupos</h3>", unsafe_allow_html=True)
    
    opcion = st.selectbox("Análisis:", ["Diferencia de Medias", "Diferencia de Proporciones"])
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    
    if opcion == "Diferencia de Medias":
        with col_a:
            st.write("🅰️ **Grupo 1**")
            m1 = st.number_input("Media 1", step=0.1, key="m1")
            s1 = st.number_input("Desv 1", step=0.1, key="s1")
            n1 = st.number_input("Tamaño 1", value=30.0, key="n1")
        with col_b:
            st.write("🅱️ **Grupo 2**")
            m2 = st.number_input("Media 2", step=0.1, key="m2")
            s2 = st.number_input("Desv 2", step=0.1, key="s2")
            n2 = st.number_input("Tamaño 2", value=30.0, key="n2")
            
        alpha = st.number_input("Significancia (α)", value=0.05, step=0.01)

        if st.button("Comparar Medias", key="btn_comp_m"):
            se = math.sqrt((s1**2 / n1) + (s2**2 / n2))
            if se == 0:
                st.error("Error estándar 0.")
            else:
                t_stat = (m1 - m2) / se
                df = n1 + n2 - 2
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                
                conclusion = "DIFERENCIA SIGNIFICATIVA" if p_val < alpha else "NO HAY DIFERENCIA"
                color = "border-red" if p_val < alpha else "border-green"
                
                c1, c2 = st.columns(2)
                c1.markdown(card("Diferencia", f"{(m1-m2):.2f}", "", "border-red"), unsafe_allow_html=True)
                c2.markdown(card("Valor P", f"{p_val:.4f}", conclusion, color), unsafe_allow_html=True)

    else:
        with col_a:
            st.write("🅰️ **Grupo 1**")
            x1 = st.number_input("Éxitos 1", 0.0)
            nt1 = st.number_input("Total 1", 100.0)
        with col_b:
            st.write("🅱️ **Grupo 2**")
            x2 = st.number_input("Éxitos 2", 0.0)
            nt2 = st.number_input("Total 2", 100.0)
        
        alpha = st.number_input("Significancia (α)", value=0.05, step=0.01, key="alpha_prop")

        if st.button("Comparar Proporciones", key="btn_comp_p"):
            p1 = x1/nt1
            p2 = x2/nt2
            pp = (x1 + x2) / (nt1 + nt2)
            se = math.sqrt(pp*(1-pp) * (1/nt1 + 1/nt2))
            
            if se == 0:
                st.error("Error estándar 0.")
            else:
                z = (p1 - p2) / se
                p_val = 2 * (1 - stats.norm.cdf(abs(z)))
                
                conclusion = "DIFERENCIA SIGNIFICATIVA" if p_val < alpha else "NO HAY DIFERENCIA"
                color = "border-red" if p_val < alpha else "border-green"
                
                c1, c2 = st.columns(2)
                c1.markdown(card("Dif %", f"{(p1-p2)*100:.2f}%", "", "border-red"), unsafe_allow_html=True)
                c2.markdown(card("Valor P", f"{p_val:.4f}", conclusion, color), unsafe_allow_html=True)

# =============================================================================
# 4. TAMAÑO DE MUESTRA (Verde)
# =============================================================================
with tab4:
    st.markdown("<h3 style='color:#22c55e'>Calculadora de Muestra (n)</h3>", unsafe_allow_html=True)
    
    target = st.radio("Objetivo:", ["Estimar Promedio", "Estimar Proporción"], key="target_n")
    
    col1, col2 = st.columns(2)
    error = col1.number_input("Margen de Error (E)", value=0.05, format="%.4f")
    conf = col2.number_input("Confianza (%)", value=95.0, step=1.0, key="conf_n")
    
    if target == "Estimar Promedio":
        sigma = st.number_input("Desviación Estimada (σ)", value=10.0)
        if st.button("Calcular N Promedio"):
            z = stats.norm.ppf((1 + conf/100.0)/2)
            n_res = (z**2 * sigma**2) / (error**2)
            st.markdown(card("Muestra Necesaria", f"{math.ceil(n_res)}", "Registros", "border-green"), unsafe_allow_html=True)
    else:
        p_est = st.number_input("Proporción Estimada (p)", value=0.5)
        if st.button("Calcular N Proporción"):
            z = stats.norm.ppf((1 + conf/100.0)/2)
            n_res = (z**2 * p_est * (1-p_est)) / (error**2)
            st.markdown(card("Muestra Necesaria", f"{math.ceil(n_res)}", "Encuestados", "border-green"), unsafe_allow_html=True)

# =============================================================================
# 5. VISUAL LAB (Amarillo)
# =============================================================================
with tab5:
    st.markdown("<h3 style='color:#f59e0b'>Laboratorio Visual</h3>", unsafe_allow_html=True)
    
    tool = st.selectbox("Herramienta:", ["Teorema del Límite Central (TLC)", "Error Estándar vs n"])
    
    if tool == "Teorema del Límite Central (TLC)":
        st.info("Simula cómo el promedio de muchas muestras forma una campana, aunque la población original no lo sea.")
        c1, c2 = st.columns(2)
        n_sim = c1.number_input("Tamaño de muestra (n)", value=30, min_value=1)
        reps = c2.number_input("Repeticiones", value=1000, min_value=10)
        
        if st.button("Simular TLC"):
            pop = np.random.exponential(scale=1.0, size=10000)
            means = [np.mean(np.random.choice(pop, int(n_sim))) for _ in range(int(reps))]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            fig.patch.set_facecolor('#000000')
            
            # Gráfico 1: Población
            ax1.set_facecolor('#0a0a0a')
            ax1.hist(pop, bins=30, color='#666', edgecolor='#999')
            ax1.set_title("Población Original (Sesgada)", color='white')
            ax1.axis('off')
            
            # Gráfico 2: Medias
            ax2.set_facecolor('#0a0a0a')
            ax2.hist(means, bins=30, color='#f59e0b', alpha=0.8, edgecolor='black')
            ax2.set_title(f"Distribución de Medias (Normal)", color='white')
            ax2.axis('off')
            
            st.pyplot(fig)

    elif tool == "Error Estándar vs n":
        sigma_sim = st.number_input("Desviación Poblacional", value=10.0)
        
        if st.button("Generar Curva"):
            ns = np.arange(1, 200)
            ees = sigma_sim / np.sqrt(ns)
            
            fig, ax = plt.subplots(figsize=(8, 3))
            fig.patch.set_facecolor('#000000')
            ax.set_facecolor('#0a0a0a')
            
            ax.plot(ns, ees, color='#f59e0b', lw=3)
            ax.set_xlabel("Tamaño de Muestra (n)", color='white')
            ax.set_ylabel("Error Estándar", color='white')
            ax.grid(color='#333', linestyle='--', alpha=0.3)
            
            # Estilo ejes
            ax.spines['bottom'].set_color('#666')
            ax.spines['left'].set_color('#666')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(colors='#999')
            
            st.pyplot(fig)
