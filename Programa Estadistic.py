import re
import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
from collections import Counter

st.set_page_config(
    page_title="Calculadora de estadística",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

        .stApp {
            background-color: #000000;
            font-family: 'Inter', sans-serif;
            color: #ffffff;
        }

        h1 {
            color: #ffffff !important;
            font-weight: 800 !important;
            font-size: 3rem !important;
            margin-bottom: 1rem !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 20px;
            background-color: transparent;
            border-bottom: 1px solid #333;
        }
        .stTabs [data-baseweb="tab"] {
            color: #888;
            font-size: 1rem;
            border: none;
            background: transparent;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            color: #ffffff;
            font-weight: bold;
            border-bottom: 3px solid #a855f7;
        }

        .stTextArea textarea {
            background-color: #000000 !important;
            color: white !important;
            border-radius: 10px;
            border: 1px solid #333;
        }
        .stNumberInput input {
            background-color: #1a1a1a !important;
            color: white !important;
            border-radius: 8px;
            border: 1px solid #333;
        }

        div.stButton > button {
            background-color: #6d28d9;
            color: white;
            border-radius: 30px;
            font-weight: bold;
            border: none;
            padding: 10px 20px;
            font-size: 1rem;
            transition: 0.3s;
        }
        div.stButton > button:hover {
            background-color: #7c3aed;
        }

        .result-card {
            background-color: #1a1a1a;
            border: 1px solid #333;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            margin-bottom: 15px;
        }
        .card-label {
            color: #888;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }
        .card-value {
            color: #fff;
            font-size: 2rem;
            font-weight: 700;
        }
        .card-sub {
            color: #666;
            font-size: 0.8rem;
            font-style: italic;
        }
        
        .border-purple { border-top: 4px solid #a855f7; }
        .border-blue { border-top: 4px solid #3b82f6; }
        .border-red { border-top: 4px solid #ef4444; }
        .border-green { border-top: 4px solid #22c55e; }
        .border-yellow { border-top: 4px solid #f59e0b; }

    </style>
""", unsafe_allow_html=True)

st.title("Calculadora de estadística")

def card(label, value, sub="", color="border-blue"):
    return f"""
    <div class="result-card {color}">
        <div class="card-label">{label}</div>
        <div class="card-value">{value}</div>
        <div class="card-sub">{sub}</div>
    </div>
    """

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Medidas de tendencia central", 
    "Inferencia estadística", 
    "Comparación de dos poblaciones", 
    "Tamaño de muestra",
    "Visual LAB"
])

with tab1:
    st.markdown("""
        <div style="background-color: white; border-radius: 15px 15px 0 0; padding: 15px;">
            <h3 style="color: #6d28d9; margin:0; font-weight:800;">Datos:</h3>
            <div style="height: 4px; width: 50px; background-color: #6d28d9; margin-top:5px; border-radius: 2px;"></div>
        </div>
        <div style="background-color: #3b82f6; color: white; padding: 10px; font-size: 0.9rem; text-align: center;">
            Usa PUNTO (.) para decimales. Separa números con comas, punto y coma, espacios o saltos de línea.
        </div>
    """, unsafe_allow_html=True)
    
    input_desc = st.text_area("", height=150, placeholder="Ej: 3.2, 4.5, 7.8", label_visibility="collapsed")
    btn_calc_desc = st.button("Analizar datos", key="btn1")

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
                        
                        contador = Counter(data)
                        max_freq = max(contador.values())
                        modas = [k for k, v in contador.items() if v == max_freq]
                        
                        if len(modas) == len(data):
                            moda_str = "—"
                            moda_sub = "No hay moda"
                        elif len(modas) == 1:
                            moda_str = f"{modas[0]:.2f}"
                            moda_sub = f"Freq: {max_freq}"
                        else:
                            moda_str = "Múltiple"
                            moda_sub = f"{len(modas)} modas"

                        st.markdown("---")
                        c1, c2, c3 = st.columns(3)
                        c1.markdown(card("Promedio", f"{media:.2f}", "", "border-purple"), unsafe_allow_html=True)
                        c2.markdown(card("Mediana", f"{mediana:.2f}", "", "border-purple"), unsafe_allow_html=True)
                        c3.markdown(card("Moda", moda_str, moda_sub, "border-purple"), unsafe_allow_html=True)
                        
                        c4, c5, c6 = st.columns(3)
                        c4.markdown(card("Varianza", f"{var:.2f}", "", "border-purple"), unsafe_allow_html=True)
                        c5.markdown(card("Desviación Std", f"{desv:.2f}", "Muestral" if n>=2 else "Poblacional", "border-purple"), unsafe_allow_html=True)
                        c6.markdown(card("Error Estándar", f"{ee:.4f}", "", "border-purple"), unsafe_allow_html=True)

                        fig, ax = plt.subplots(figsize=(10, 3))
                        fig.patch.set_facecolor('#000000')
                        ax.set_facecolor('#000000')
                        ax.hist(data, bins='auto', color='#a855f7', edgecolor='black', alpha=0.8)
                        ax.axvline(media, color='white', linestyle='--', label='Promedio')
                        ax.axis('off')
                        st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error: {e}")

with tab2:
    st.markdown("<h3 style='color:#3b82f6'>Inferencia de Una Población</h3>", unsafe_allow_html=True)
    
    tipo_dato = st.radio("Tipo:", ["Promedio (Media)", "Porcentaje (Proporción)", "Posición Individual (Z)"], horizontal=True)
    st.markdown("---")

    if tipo_dato == "Promedio (Media)":
        c1, c2, c3 = st.columns(3)
        media = c1.number_input("Promedio Muestral (x̄)", step=0.01, format="%.4f")
        n = c2.number_input("Muestra (n)", value=30.0, step=1.0)
        conf = c3.number_input("Confianza (%)", value=95.0, step=1.0)

        col_sig, col_s = st.columns(2)
        sigma = col_sig.number_input("Poblacional (σ) [Usa Z]", step=0.01)
        s = col_s.number_input("Muestral (s) [Usa T]", step=0.01)

        realizar_prueba = st.checkbox("Prueba de hipótesis (H0)")
        mu_hyp = 0.0
        if realizar_prueba:
            mu_hyp = st.number_input("Valor Hipotético (μ0)", value=0.0, step=0.01)

        if st.button("Calcular Inferencia"):
            conf_dec = conf / 100.0
            n_int = int(n)
            
            if sigma > 0:
                se = sigma / math.sqrt(n_int)
                z_val = stats.norm.ppf((1 + conf_dec)/2)
                margen = z_val * se
            elif s > 0:
                se = s / math.sqrt(n_int)
                if n_int >= 30:
                    z_val = stats.norm.ppf((1 + conf_dec)/2)
                    margen = z_val * se
                else:
                    t_val = stats.t.ppf((1 + conf_dec)/2, df=n_int-1)
                    margen = t_val * se
            else:
                st.error("Falta desviación (σ o s).")
                st.stop()

            c1, c2 = st.columns(2)
            c1.markdown(card("Límite Inferior", f"{media - margen:.4f}", "", "border-blue"), unsafe_allow_html=True)
            c2.markdown(card("Límite Superior", f"{media + margen:.4f}", "", "border-blue"), unsafe_allow_html=True)

            if realizar_prueba:
                test_stat = (media - mu_hyp) / se
                p_val = 2 * (1 - stats.norm.cdf(abs(test_stat))) 
                conclusion = "Rechazar H0" if p_val < (1-conf_dec) else "No rechazar"
                st.markdown(card("Valor P", f"{p_val:.4f}", conclusion, "border-red"), unsafe_allow_html=True)

    elif tipo_dato == "Porcentaje (Proporción)":
        c1, c2, c3 = st.columns(3)
        prop = c1.number_input("Proporción (p)", value=0.5)
        n = c2.number_input("Muestra (n)", value=100.0)
        conf = c3.number_input("Confianza (%)", value=95.0)
        
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
            z = (val - mu) / sig
            st.markdown(card("Puntaje Z", f"{z:.4f}", "", "border-blue"), unsafe_allow_html=True)

with tab3:
    st.markdown("<h3 style='color:#ef4444'>Comparación de Dos Grupos</h3>", unsafe_allow_html=True)
    opcion = st.selectbox("Análisis:", ["Diferencia de Medias", "Diferencia de Proporciones"])
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    if opcion == "Diferencia de Medias":
        with col_a:
            st.write("Grupo 1")
            m1 = st.number_input("Media 1", step=0.1)
            s1 = st.number_input("Desv 1", step=0.1)
            n1 = st.number_input("Tamaño 1", value=30.0)
        with col_b:
            st.write("Grupo 2")
            m2 = st.number_input("Media 2", step=0.1)
            s2 = st.number_input("Desv 2", step=0.1)
            n2 = st.number_input("Tamaño 2", value=30.0)
        
        if st.button("Comparar Medias"):
            se = math.sqrt((s1**2/n1) + (s2**2/n2))
            t = (m1-m2)/se
            p = 2*(1-stats.norm.cdf(abs(t)))
            st.markdown(card("Valor P", f"{p:.4f}", "Significativo" if p<0.05 else "No sig", "border-red"), unsafe_allow_html=True)
    else:
        with col_a:
            x1 = st.number_input("Éxitos 1", 0.0)
            nt1 = st.number_input("Total 1", 100.0)
        with col_b:
            x2 = st.number_input("Éxitos 2", 0.0)
            nt2 = st.number_input("Total 2", 100.0)
        if st.button("Comparar Prop"):
            p1, p2 = x1/nt1, x2/nt2
            pp = (x1+x2)/(nt1+nt2)
            se = math.sqrt(pp*(1-pp)*(1/nt1 + 1/nt2))
            z = (p1-p2)/se
            p = 2*(1-stats.norm.cdf(abs(z)))
            st.markdown(card("Valor P", f"{p:.4f}", "Significativo" if p<0.05 else "No sig", "border-red"), unsafe_allow_html=True)

with tab4:
    st.markdown("<h3 style='color:#22c55e'>Calculadora de Muestra</h3>", unsafe_allow_html=True)
    target = st.radio("Objetivo:", ["Estimar Promedio", "Estimar Proporción"])
    col1, col2 = st.columns(2)
    error = col1.number_input("Error (E)", value=0.05)
    conf = col2.number_input("Confianza (%)", value=95.0)
    
    if target == "Estimar Promedio":
        sigma = st.number_input("Desviación (σ)", value=10.0)
        if st.button("Calcular n (Promedio)"):
            z = stats.norm.ppf((1 + conf/100)/2)
            n = (z**2 * sigma**2) / error**2
            st.markdown(card("n", f"{math.ceil(n)}", "", "border-green"), unsafe_allow_html=True)
    else:
        p = st.number_input("Proporción (p)", value=0.5)
        if st.button("Calcular n (Proporción)"):
            z = stats.norm.ppf((1 + conf/100)/2)
            n = (z**2 * p * (1-p)) / error**2
            st.markdown(card("n", f"{math.ceil(n)}", "", "border-green"), unsafe_allow_html=True)

with tab5:
    st.markdown("<h3 style='color:#f59e0b'>Laboratorio Visual</h3>", unsafe_allow_html=True)
    tool = st.selectbox("Herramienta:", ["Teorema del Límite Central (TLC)", "Error Estándar vs n"])

    if tool == "Teorema del Límite Central (TLC)":
        c1, c2 = st.columns(2)
        n_sim = c1.number_input("Tamaño de muestra (n)", value=30, min_value=1)
        reps = c2.number_input("Repeticiones", value=1000, min_value=10)
        
        if st.button("Simular TLC"):
            pop = np.random.exponential(scale=1.0, size=10000)
            means = [np.mean(np.random.choice(pop, int(n_sim))) for _ in range(int(reps))]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            fig.patch.set_facecolor('#000000')
            
            ax1.set_facecolor('#0a0a0a')
            ax1.hist(pop, bins=30, color='#666', edgecolor='#999')
            ax1.set_title("Población Original (Sesgada)", color='white')
            ax1.axis('off')
            
            ax2.set_facecolor('#0a0a0a')
            ax2.hist(means, bins=30, color='#f59e0b', alpha=0.8, edgecolor='black')
            ax2.set_title(f"Distribución de Medias", color='white')
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
            
            ax.spines['bottom'].set_color('#666')
            ax.spines['left'].set_color('#666')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(colors='#999')
            
            st.pyplot(fig)
