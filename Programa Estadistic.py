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
    page_title="StatSuite Pro",
    page_icon="🧬",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. ESTILOS CSS
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
            background-color: #050505;
            color: #e5e7eb;
        }
        
        .stApp { background-color: #050505; }
        header, footer {visibility: hidden;}

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

        .border-purple { border-color: #a855f7; }
        .border-blue { border-color: #3b82f6; }
        .border-red { border-color: #ef4444; }
        .border-green { border-color: #22c55e; }

        .simple-text {
            background: rgba(255,255,255,0.05);
            border-left: 4px solid #666;
            padding: 15px;
            border-radius: 0 8px 8px 0;
            margin-top: 15px;
            color: #ddd;
            font-size: 0.95rem;
        }

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

st.title("🧬 StatSuite Pro")

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
    "🔵 Inferencia Estadística", 
    "🔴 Comparación (2 Pob)", 
    "🟢 Tamaño Muestra",
    "🧪 Visual LAB"
])

# =============================================================================
# PESTAÑA 1: DESCRIPTIVA
# =============================================================================
with tab1:
    st.markdown("<h3 style='color:#a855f7'>Medidas de Tendencia Central</h3>", unsafe_allow_html=True)
    
    col_in, col_out = st.columns([1, 2], gap="large")
    with col_in:
        st.info("Introduce tus números. Usa PUNTO (.) para decimales.")
        input_desc = st.text_area("Datos Numéricos:", height=150, placeholder="Ej: 3.2, 4.5, 7.8")
        btn_calc_desc = st.button("Analizar Datos", key="btn1")

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
                            
                            if n >= 2:
                                desv = float(np.std(data, ddof=1)) 
                                var = float(np.var(data, ddof=1))
                            else:
                                desv = float(np.std(data, ddof=0))
                                var = float(np.var(data, ddof=0))
                                
                            ee = desv / math.sqrt(n) if n > 0 else 0.0
                            rango = float(np.max(data) - np.min(data)) if n > 0 else 0.0

                            c1, c2, c3 = st.columns(3)
                            c1.markdown(card("Promedio (Media)", f"{media:.2f}", "", "border-purple"), unsafe_allow_html=True)
                            c2.markdown(card("Mediana", f"{mediana:.2f}", "", "border-purple"), unsafe_allow_html=True)
                            c3.markdown(card("Error Estándar (EE)", f"{ee:.4f}", "", "border-purple"), unsafe_allow_html=True)
                            
                            c4, c5, c6 = st.columns(3)
                            c4.markdown(card("Desviación Estándar", f"{desv:.2f}", "Muestral" if n>=2 else "Poblacional", "border-purple"), unsafe_allow_html=True)
                            c5.markdown(card("Varianza", f"{var:.2f}", "", "border-purple"), unsafe_allow_html=True)
                            c6.markdown(card("Rango", f"{rango:.2f}", "", "border-purple"), unsafe_allow_html=True)

                            sesgo = ""
                            if desv == 0:
                                sesgo = "Datos idénticos."
                            elif abs(media - mediana) < (desv/10):
                                sesgo = "Los datos son bastante simétricos (Media ≈ Mediana)."
                            elif media > mediana:
                                sesgo = "Sesgo positivo (Cola a la derecha)."
                            else:
                                sesgo = "Sesgo negativo (Cola a la izquierda)."

                            st.markdown(f"""
                            <div class="simple-text" style="border-left-color: #a855f7;">
                                <strong>Interpretación:</strong><br>
                                Con n=<b>{n}</b>, el centro se ubica en <b>{media:.2f}</b>. La dispersión (s) es de <b>{desv:.2f}</b>.<br>
                                {sesgo}
                            </div>
                            """, unsafe_allow_html=True)

                            st.write("#### Histograma de Frecuencias")
                            fig, ax = plt.subplots(figsize=(10, 3))
                            fig.patch.set_facecolor('#050505')
                            ax.set_facecolor('#111')
                            counts, bins, patches = ax.hist(data, bins='auto', color='#a855f7', edgecolor='black', alpha=0.9)
                            ax.axvline(media, color='white', linestyle='--', label='Promedio')
                            ax.legend(facecolor='#222', labelcolor='white', frameon=False)
                            ax.axis('off') 
                            st.pyplot(fig)
                            
                    except Exception as e:
                        st.error(f"Error: {e}")

# =============================================================================
# PESTAÑA 2: INFERENCIA
# =============================================================================
with tab2:
    st.markdown("<h3 style='color:#3b82f6'>Inferencia de Una Población</h3>", unsafe_allow_html=True)
    
    tipo_dato = st.radio("¿Qué tipo de dato tienes?", ["Promedio (Media)", "Porcentaje (Proporción)", "Posición Individual (Z)"], horizontal=True)
    st.markdown("---")

    if tipo_dato == "Promedio (Media)":
        c1, c2, c3 = st.columns(3)
        media = c1.number_input("Promedio Muestral (x̄)", step=0.01, format="%.4f")
        n = c2.number_input("Tamaño de Muestra (n)", value=30.0, step=1.0)
        conf = c3.number_input("Nivel de Confianza (90-99%)", value=95.0, step=0.1) / 100.0

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

                c_res1, c_res2 = st.columns(2)
                c_res1.markdown(card("Límite Inferior", f"{media - margen:.4f}", "", "border-blue"), unsafe_allow_html=True)
                c_res2.markdown(card("Límite Superior", f"{media + margen:.4f}", "", "border-blue"), unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class='simple-text' style='border-left-color:#3b82f6'>
                <strong>Interpretación del Intervalo:</strong><br>
                Con un {conf*100:.1f}% de confianza, el verdadero promedio poblacional está entre <b>{media - margen:.4f}</b> y <b>{media + margen:.4f}</b>.<br>
                Método usado: {dist_label}. Error Estándar: {se:.4f}.
                </div>""", unsafe_allow_html=True)

                if realizar_prueba:
                    st.markdown("#### Resultado de Prueba de Hipótesis")
                    test_stat = (media - mu_hyp) / se
                    if dist_label.startswith("T-Student"):
                        p_val = 2 * (1 - test_func.cdf(abs(test_stat), df=n_int-1))
                    else:
                        p_val = 2 * (1 - test_func.cdf(abs(test_stat)))
                        
                    alpha = 1 - conf
                    sig = "Rechazar H0 (Diferencia Significativa)" if p_val < alpha else "No Rechazar H0"
                    col_h = "border-red" if p_val < alpha else "border-green"
                    
                    h1, h2 = st.columns(2)
                    h1.markdown(card("Estadístico de Prueba", f"{test_stat:.4f}", "", "border-blue"), unsafe_allow_html=True)
                    h2.markdown(card("Valor P (P-Value)", f"{p_val:.4f}", sig, col_h), unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error en cálculo: {e}")

    elif tipo_dato == "Porcentaje (Proporción)":
        c1, c2, c3 = st.columns(3)
        prop = c1.number_input("Proporción (p) [0.0 - 1.0]", value=0.5, step=0.01)
        n = c2.number_input("Muestra (n)", value=100.0, step=1.0)
        conf = c3.number_input("Nivel de Confianza %", value=95.0, step=0.1) / 100.0
        
        realizar_prueba_p = st.checkbox("Calcular prueba de hipótesis (H0)", value=False)
        p_hyp = 0.5
        if realizar_prueba_p:
            p_hyp = st.number_input("Proporción Hipotética (p0)", value=0.5, step=0.01)

        if st.button("Calcular Intervalo"):
            n_int = int(n)
            se = math.sqrt((prop * (1-prop)) / n_int)
            z = stats.norm.ppf((1+conf)/2)
            margen = z * se
            
            c1, c2 = st.columns(2)
            c1.markdown(card("Límite Inferior", f"{(prop-margen)*100:.2f}%", f"{max(0, prop-margen):.4f}", "border-blue"), unsafe_allow_html=True)
            c2.markdown(card("Límite Superior", f"{(prop+margen)*100:.2f}%", f"{min(1, prop+margen):.4f}", "border-blue"), unsafe_allow_html=True)

            if realizar_prueba_p:
                 se_hyp = math.sqrt((p_hyp * (1-p_hyp)) / n_int)
                 z_stat = (prop - p_hyp) / se_hyp
                 p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                 sig_txt = "Significativa" if p_val < (1-conf) else "No significativa"
                 st.markdown(f"<div class='simple-text'>Prueba de Hipótesis: Valor P = {p_val:.4f} ({sig_txt})</div>", unsafe_allow_html=True)

    elif tipo_dato == "Posición Individual (Z)":
        c1, c2, c3 = st.columns(3)
        val = c1.number_input("Valor a Evaluar (x)", 0.0)
        mu = c2.number_input("Promedio Población (μ)", 0.0)
        sig = c3.number_input("Desviación Población (σ)", 1.0)
        if st.button("Calcular Z"):
            z = (val - mu) / sig
            st.markdown(card("Puntaje Z", f"{z:.4f}", "Desviaciones estándar", "border-blue"), unsafe_allow_html=True)

# =============================================================================
# PESTAÑA 3: COMPARACIÓN
# =============================================================================
with tab3:
    st.markdown("<h3 style='color:#ef4444'>Comparación de Dos Poblaciones</h3>", unsafe_allow_html=True)
    opcion = st.selectbox("Seleccione Análisis:", ["Diferencia de Medias", "Diferencia de Proporciones"])
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    
    if opcion == "Diferencia de Medias":
        with col_a:
            st.write("🅰️ **Grupo 1**")
            m1 = st.number_input("Media 1 (x̄1)", step=0.1)
            s1 = st.number_input("Desviación 1 (s1)", step=0.1, value=1.0)
            n1 = st.number_input("Tamaño 1 (n1)", value=30.0)
        with col_b:
            st.write("🅱️ **Grupo 2**")
            m2 = st.number_input("Media 2 (x̄2)", step=0.1)
            s2 = st.number_input("Desviación 2 (s2)", step=0.1, value=1.0)
            n2 = st.number_input("Tamaño 2 (n2)", value=30.0)
            
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

    else:
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
# PESTAÑA 4: TAMAÑO MUESTRA
# =============================================================================
with tab4:
    st.markdown("<h3 style='color:#22c55e'>Calculadora de Muestra (n)</h3>", unsafe_allow_html=True)
    target = st.radio("Objetivo:", ["Estimar Promedio", "Estimar Proporción"])
    
    col1, col2 = st.columns(2)
    error = col1.number_input("Margen de Error (E)", value=0.05, format="%.4f")
    conf = col2.number_input("Confianza (1-α) [0-1]", value=0.95)
    
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
# PESTAÑA 5: VISUAL LAB
# =============================================================================
with tab5:
    st.markdown("<h3 style='color:#ffffff'>Laboratorio Visual</h3>", unsafe_allow_html=True)
    st.info("Simulación del Teorema del Límite Central")
    
    col1, col2 = st.columns(2)
    n_sim = int(col1.number_input("Tamaño de muestra (n)", value=30, min_value=1))
    reps = int(col2.number_input("Repeticiones", value=500, min_value=10))
    
    if st.button("Simular TLC"):
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
