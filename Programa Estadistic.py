import re
import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math

# -----------------------------------------------------------------------------
# CONFIGURACIÓN
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Calculadora de Estadística",
    page_icon="📊",
    layout="wide"
)

# -----------------------------------------------------------------------------
# ESTILOS CSS - TEMA OSCURO
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* RESET GENERAL */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: #000000;
            color: #e5e7eb;
        }
        
        header, footer {visibility: hidden;}
        .stApp { 
            background-color: #000000;
        }

        /* TÍTULO PRINCIPAL */
        h1 {
            font-size: 1.5rem !important;
            font-weight: 600 !important;
            color: #ffffff !important;
            margin-bottom: 2rem !important;
        }

        /* PESTAÑAS MEJORADAS */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: transparent;
            padding: 0;
            border-bottom: 1px solid #262626;
            margin-bottom: 2rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 48px;
            border-radius: 0;
            font-weight: 500;
            font-size: 0.875rem;
            color: #6b7280;
            border: none;
            border-bottom: 2px solid transparent;
            background-color: transparent;
            padding: 0 16px;
            transition: all 0.2s;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #ffffff;
            background-color: #1a1a1a;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: transparent;
            color: #3b82f6;
            border-bottom: 2px solid #3b82f6;
        }

        /* Ocultar flechas de inputs numéricos */
        input[type=number]::-webkit-inner-spin-button, 
        input[type=number]::-webkit-outer-spin-button { 
            -webkit-appearance: none; 
            margin: 0; 
        }
        input[type=number] {
            -moz-appearance: textfield;
        }
        
        /* INPUTS MEJORADOS */
        .stNumberInput > div > div > input,
        .stTextArea textarea,
        .stSelectbox select {
            background-color: #1a1a1a !important;
            color: #ffffff !important;
            border: 1px solid #333333 !important;
            border-radius: 8px !important;
            font-size: 0.875rem !important;
            padding: 10px 12px !important;
            transition: all 0.2s !important;
        }
        
        .stNumberInput > div > div > input:focus,
        .stTextArea textarea:focus {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
            outline: none !important;
        }

        /* LABELS */
        .stNumberInput label,
        .stTextArea label,
        .stSelectbox label {
            font-size: 0.875rem !important;
            font-weight: 500 !important;
            color: #d1d5db !important;
            margin-bottom: 6px !important;
        }

        /* TARJETAS DE RESULTADOS - TEMA OSCURO */
        .metric-card {
            background-color: #1a1a1a;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #262626;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
            margin-bottom: 16px;
            transition: all 0.2s;
        }
        .metric-card:hover {
            border-color: #333333;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
            transform: translateY(-2px);
        }
        .metric-label {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #9ca3af;
            margin-bottom: 8px;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #ffffff;
            line-height: 1;
        }
        .metric-sub {
            font-size: 0.8rem;
            color: #6b7280;
            margin-top: 4px;
        }

        /* COLORES DE ACENTO */
        .accent-blue { color: #3b82f6; }
        .accent-green { color: #10b981; }
        .accent-red { color: #ef4444; }
        .accent-purple { color: #a855f7; }
        .border-blue { border-left: 4px solid #3b82f6 !important; }
        .border-green { border-left: 4px solid #10b981 !important; }
        .border-red { border-left: 4px solid #ef4444 !important; }
        .border-purple { border-left: 4px solid #a855f7 !important; }

        /* CAJAS DE INFORMACIÓN */
        .info-box {
            background: #0f172a;
            border-left: 4px solid #3b82f6;
            padding: 16px;
            border-radius: 8px;
            margin: 16px 0;
            font-size: 0.875rem;
            color: #93c5fd;
            line-height: 1.6;
        }

        .interpretation-box {
            background: #1a1a1a;
            border: 1px solid #262626;
            padding: 20px;
            border-radius: 12px;
            margin-top: 24px;
            font-size: 0.875rem;
            line-height: 1.7;
            color: #d1d5db;
        }

        /* BOTONES MEJORADOS */
        div.stButton > button {
            width: 100%;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.875rem;
            padding: 12px 24px;
            background-color: #3b82f6;
            color: white;
            border: none;
            transition: all 0.2s;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
        }
        div.stButton > button:hover {
            background-color: #2563eb;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
            transform: translateY(-1px);
        }

        /* SECCIONES */
        .section-header {
            font-size: 1.125rem;
            font-weight: 600;
            color: #ffffff;
            margin: 32px 0 16px 0;
            padding-bottom: 8px;
            border-bottom: 2px solid #262626;
        }

        /* RADIO BUTTONS Y CHECKBOXES */
        .stRadio > label,
        .stCheckbox > label {
            font-size: 0.875rem !important;
            font-weight: 500 !important;
            color: #d1d5db !important;
        }

        /* DIVISOR */
        hr {
            margin: 32px 0;
            border: none;
            border-top: 1px solid #262626;
        }

        /* ALERTS */
        .stAlert {
            background-color: #1a1a1a !important;
            border: 1px solid #333333 !important;
            color: #e5e7eb !important;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TÍTULO
# -----------------------------------------------------------------------------
st.title("📊 Calculadora de Estadística")

# Función auxiliar de tarjeta HTML
def metric_card(label, value, sub="", accent="blue"):
    return f"""
    <div class="metric-card border-{accent}">
        <div class="metric-label">{label}</div>
        <div class="metric-value accent-{accent}">{value}</div>
        <div class="metric-sub">{sub}</div>
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
# 1. MEDIDAS DE TENDENCIA CENTRAL
# =============================================================================
with tab1:
    st.markdown("<div class='section-header'>Analizar datos</div>", unsafe_allow_html=True)
    
    col_in, col_out = st.columns([1, 2], gap="large")
    
    with col_in:
        st.markdown("""
        <div class='info-box'>
            <strong>📌 Instrucciones:</strong><br>
            Usa PUNTO (.) para decimales. Separa números con comas, punto y coma, espacios o saltos de línea.
        </div>
        """, unsafe_allow_html=True)
        
        input_desc = st.text_area(
            "Datos:", 
            height=200, 
            placeholder="Ejemplo: 3.2, 4.5, 7.8, 9.1, 0.6, 12.3, 14.7",
            help="Ingresa tus datos separados por comas, punto y coma, espacios o saltos de línea"
        )
        btn_calc_desc = st.button("Analizar datos", key="btn1", type="primary")

    with col_out:
        if btn_calc_desc:
            if not input_desc.strip():
                st.warning("⚠️ El campo de datos está vacío.")
            else:
                if re.search(r'\d+,\d+', input_desc):
                    st.error("❌ Por favor usa PUNTO (.) para decimales. Reemplaza '10,5' por '10.5' y vuelve a intentar.")
                else:
                    try:
                        parts = re.split(r'[,\;\s]+', input_desc.strip())
                        tokens = [p for p in parts if p != '']
                        nums = []
                        for t in tokens:
                            if re.fullmatch(r'[-+]?(?:\d+|\d+\.\d+|\.\d+)', t):
                                nums.append(float(t))
                            else:
                                st.error(f"❌ Token inválido detectado: '{t}'. Usa sólo números con '.' como decimal.")
                                nums = None
                                break

                        if nums is None or len(nums) == 0:
                            if nums is not None:
                                st.error("❌ No se detectaron números válidos.")
                        else:
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

                            # Resultados principales
                            c1, c2, c3 = st.columns(3)
                            c1.markdown(metric_card("Promedio (media)", f"{media:.2f}", f"n = {n}", "blue"), unsafe_allow_html=True)
                            c2.markdown(metric_card("Mediana", f"{mediana:.2f}", "Valor central", "blue"), unsafe_allow_html=True)
                            c3.markdown(metric_card("Moda", "—", "No hay moda", "blue"), unsafe_allow_html=True)
                            
                            c4, c5, c6 = st.columns(3)
                            c4.markdown(metric_card("Varianza (s²)", f"{var:.2f}", "Muestral", "purple"), unsafe_allow_html=True)
                            c5.markdown(metric_card("Desviación estándar (s)", f"{desv:.2f}", "Muestral", "purple"), unsafe_allow_html=True)
                            c6.markdown(metric_card("Error estándar (EE)", f"{ee:.4f}", "", "purple"), unsafe_allow_html=True)

                            # Interpretación
                            sesgo = ""
                            if desv == 0:
                                sesgo = "No hay variación (todos los valores son iguales)."
                            else:
                                if abs(media - mediana) < (desv/10):
                                    sesgo = "Los datos son bastante simétricos (Media ≈ Mediana)."
                                elif media > mediana:
                                    sesgo = "Hay sesgo positivo (cola a la derecha)."
                                else:
                                    sesgo = "Hay sesgo negativo (cola a la izquierda)."

                            st.markdown(f"""
                            <div class="interpretation-box">
                                <strong>📊 Interpretación:</strong><br><br>
                                Con una muestra de <strong>{n}</strong> datos, el centro se ubica en <strong>{media:.2f}</strong>. 
                                La dispersión (s) es de <strong>{desv:.2f}</strong>.<br><br>
                                <em>{sesgo}</em>
                            </div>
                            """, unsafe_allow_html=True)

                            # Histograma
                            st.markdown("<div class='section-header'>Histograma de Frecuencias</div>", unsafe_allow_html=True)
                            fig, ax = plt.subplots(figsize=(10, 4))
                            fig.patch.set_facecolor('#000000')
                            ax.set_facecolor('#0a0a0a')
                            counts, bins, patches = ax.hist(data, bins='auto', color='#3b82f6', edgecolor='#1a1a1a', alpha=0.9, linewidth=1.5)
                            try:
                                ax.bar_label(patches, fmt='%.0f', color='#ffffff', padding=3, fontweight='600', fontsize=9)
                            except Exception:
                                pass
                            ax.axvline(media, color='#ef4444', linestyle='--', linewidth=2, label=f'Promedio: {media:.2f}')
                            ax.legend(loc='upper right', frameon=True, facecolor='#1a1a1a', edgecolor='#333333', labelcolor='white')
                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)
                            ax.spines['left'].set_color('#333333')
                            ax.spines['bottom'].set_color('#333333')
                            ax.tick_params(colors='#9ca3af')
                            ax.grid(axis='y', alpha=0.1, color='#333333')
                            st.pyplot(fig)
                    except Exception as e:
                        st.error(f"❌ Error al procesar los datos: {e}")

# =============================================================================
# 2. INFERENCIA ESTADÍSTICA
# =============================================================================
with tab2:
    st.markdown("<div class='section-header'>Inferencia de Una Población</div>", unsafe_allow_html=True)
    
    tipo_dato = st.radio("¿Qué tipo de dato tienes?", 
                        ["Promedio (Media)", "Porcentaje (Proporción)", "Posición Individual (Z)"], 
                        horizontal=True)
    
    st.markdown("---")

    # --- PROMEDIO (MEDIA) ---
    if tipo_dato == "Promedio (Media)":
        st.markdown("### Datos")
        c1, c2, c3 = st.columns(3)
        media = c1.number_input("Promedio Muestral (x̄)", step=0.01, format="%.4f", value=15.0)
        n = c2.number_input("Tamaño de Muestra (n)", value=30.0, step=1.0)
        conf = c3.number_input("Nivel de Confianza (1−α)", value=95.0, min_value=0.0, max_value=100.0, step=1.0) / 100

        st.markdown("### Desviación Estándar")
        col_sig, col_s = st.columns(2)
        sigma = col_sig.number_input("Poblacional (σ)", step=0.01, format="%.4f", value=10.0, help="Usa esta si conoces σ")
        s = col_s.number_input("Muestral (s)", step=0.01, format="%.4f", value=0.0, help="Usa esta si solo tienes s")

        realizar_prueba = st.checkbox("Calcular prueba de hipótesis (H₀)", value=False)
        mu_hyp = 0.0
        if realizar_prueba:
            mu_hyp = st.number_input("Valor Hipotético (μ₀)", value=20.0, step=0.01, format="%.4f")

        if st.button("Calcular Inferencia", key="btn_inf", type="primary"):
            try:
                n_int = int(n)
                if n_int <= 0:
                    st.error("❌ n debe ser mayor que 0.")
                    st.stop()
            except Exception:
                st.error("❌ Tamaño de muestra inválido.")
                st.stop()

            se = None
            dist_label = ""
            margen = None
            test_stat = None
            p_val = None

            # Detectar Z o T
            if sigma and sigma > 0:
                se = sigma / math.sqrt(n_int)
                z_val = stats.norm.ppf((1 + conf)/2)
                margen = z_val * se
                dist_label = "Normal (Z) - Sigma Conocida"
                if realizar_prueba:
                    test_stat = (media - mu_hyp) / se
                    p_val = 2 * (1 - stats.norm.cdf(abs(test_stat)))
            elif s and s > 0:
                se = s / math.sqrt(n_int)
                if n_int >= 30:
                    z_val = stats.norm.ppf((1 + conf)/2)
                    margen = z_val * se
                    dist_label = "Normal (Z) - Muestra Grande"
                    if realizar_prueba:
                        test_stat = (media - mu_hyp) / se
                        p_val = 2 * (1 - stats.norm.cdf(abs(test_stat)))
                else:
                    t_val = stats.t.ppf((1 + conf)/2, df=n_int-1)
                    margen = t_val * se
                    dist_label = "T-Student - Muestra Pequeña"
                    if realizar_prueba:
                        test_stat = (media - mu_hyp) / se
                        p_val = 2 * (1 - stats.t.cdf(abs(test_stat), df=n_int-1))
            else:
                st.error("⚠️ Debes ingresar alguna desviación positiva (σ o s).")
                st.stop()

            # Mostrar Intervalo
            st.markdown("### Resultados")
            c_res1, c_res2 = st.columns(2)
            c_res1.markdown(metric_card("Límite Inferior", f"{media - margen:.4f}", "", "green"), unsafe_allow_html=True)
            c_res2.markdown(metric_card("Límite Superior", f"{media + margen:.4f}", "", "green"), unsafe_allow_html=True)

            st.markdown(f"""
            <div class="interpretation-box">
                <strong>📊 Interpretación del Intervalo:</strong><br><br>
                Con un <strong>{conf*100:.1f}%</strong> de confianza, el verdadero promedio poblacional está entre 
                <strong>{media - margen:.4f}</strong> y <strong>{media + margen:.4f}</strong>.<br><br>
                <em>Método usado: {dist_label}. Error Estándar: {se:.4f}.</em>
            </div>
            """, unsafe_allow_html=True)

            # Mostrar Hipótesis
            if realizar_prueba:
                st.markdown("### Resultado de Prueba de Hipótesis")
                alpha_calc = 1 - conf
                conclusion = "Rechazar H₀ (Diferencia Significativa)" if (p_val is not None and p_val < alpha_calc) else "No Rechazar H₀"
                color_hyp = "red" if (p_val is not None and p_val < alpha_calc) else "green"
                
                h1, h2 = st.columns(2)
                h1.markdown(metric_card("Valor P (P-Value)", f"{p_val:.4f}" if p_val is not None else "N/A", conclusion, color_hyp), unsafe_allow_html=True)
                h2.markdown(metric_card("Estadístico de Prueba", f"{test_stat:.4f}" if test_stat is not None else "N/A", conclusion, color_hyp), unsafe_allow_html=True)

    # --- PROPORCIÓN ---
    elif tipo_dato == "Porcentaje (Proporción)":
        st.markdown("### Datos")
        c1, c2, c3 = st.columns(3)
        prop = c1.number_input("Proporción (p) [Ej: 0.5]", value=0.5, min_value=0.0, max_value=1.0, step=0.01)
        n = c2.number_input("Tamaño de Muestra (n)", value=100.0, step=1.0)
        conf = c3.number_input("Nivel de Confianza (1−α)", value=95.0, step=1.0) / 100

        realizar_prueba_p = st.checkbox("Calcular prueba de hipótesis", value=False)
        p_hyp = 0.0
        if realizar_prueba_p:
            p_hyp = st.number_input("Proporción Hipotética (p₀)", value=0.5, min_value=0.0, max_value=1.0, step=0.01)

        if st.button("Calcular Intervalo", key="btn_prop", type="primary"):
            try:
                n_int = int(n)
                if n_int <= 0:
                    st.error("❌ n debe ser mayor que 0.")
                    st.stop()
            except Exception:
                st.error("❌ Tamaño de muestra inválido.")
                st.stop()

            q = 1 - prop
            se = math.sqrt((prop * q) / n_int) if n_int > 0 else 0.0
            z_val = stats.norm.ppf((1+conf)/2)
            margen = z_val * se

            st.markdown("### Resultados")
            c1, c2 = st.columns(2)
            c1.markdown(metric_card("Límite Inferior", f"{max(0, prop-margen)*100:.2f}%", f"{max(0,prop-margen):.4f}", "blue"), unsafe_allow_html=True)
            c2.markdown(metric_card("Límite Superior", f"{min(1, prop+margen)*100:.2f}%", f"{min(1,prop+margen):.4f}", "blue"), unsafe_allow_html=True)

            if realizar_prueba_p:
                se_hyp = math.sqrt((p_hyp*(1-p_hyp))/n_int) if n_int > 0 else 0.0
                if se_hyp == 0:
                    st.error("❌ No es posible calcular la prueba (desviación bajo H₀ = 0).")
                else:
                    z_stat = (prop - p_hyp) / se_hyp
                    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                    alpha_calc = 1 - conf
                    conclusion = "Diferencia Significativa" if p_val < alpha_calc else "No significativa"
                    st.markdown(f"<div class='interpretation-box'><strong>Prueba de Hipótesis:</strong> Valor P = {p_val:.4f} ({conclusion})</div>", unsafe_allow_html=True)

    # --- Z SCORE ---
    elif tipo_dato == "Posición Individual (Z)":
        st.markdown("### Datos")
        c1, c2, c3 = st.columns(3)
        val = c1.number_input("Valor a Evaluar (x)", step=0.01, format="%.4f", value=20.0)
        mu = c2.number_input("Promedio Población (μ)", step=0.01, format="%.4f", value=12.0)
        sig = c3.number_input("Desviación Población (σ)", step=0.01, format="%.4f", value=1.2)
        
        if st.button("Calcular z", key="btn_z", type="primary"):
            if sig == 0:
                st.error("❌ La desviación poblacional (σ) debe ser mayor que 0.")
            else:
                z = (val - mu) / sig
                st.markdown(metric_card("Puntaje Z", f"{z:.4f}", "Desviaciones estándar", "purple"), unsafe_allow_html=True)

# =============================================================================
# 3. COMPARACIÓN DE DOS POBLACIONES
# =============================================================================
with tab3:
    st.markdown("<div class='section-header'>Comparación de Dos Grupos</div>", unsafe_allow_html=True)
    
    opcion = st.selectbox("Seleccione Análisis:", ["Diferencia de Medias", "Diferencia de Proporciones"])
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    
    if opcion == "Diferencia de Medias":
        with col_a:
            st.markdown("### 🅰️ Grupo 1")
            m1 = st.number_input("Media 1 (x̄₁)", step=0.01, format="%.4f", key="m1")
            s1 = st.number_input("Desviación 1 (s₁)", step=0.01, format="%.4f", key="s1")
            n1 = st.number_input("Tamaño 1 (n₁)", value=30.0, step=1.0, key="n1")
        with col_b:
            st.markdown("### 🅱️ Grupo 2")
            m2 = st.number_input("Media 2 (x̄₂)", step=0.01, format="%.4f", key="m2")
            s2 = st.number_input("Desviación 2 (s₂)", step=0.01, format="%.4f", key="s2")
            n2 = st.number_input("Tamaño 2 (n₂)", value=30.0, step=1.0, key="n2")
            
        alpha = st.number_input("Nivel de Significancia (α)", value=0.05, step=0.01)

        if st.button("Comparar Grupos", type="primary"):
            try:
                n1_int = int(n1)
                n2_int = int(n2)
                if n1_int <= 1 or n2_int <= 1:
                    st.error("❌ Cada grupo debe tener n > 1.")
                    st.stop()
            except Exception:
                st.error("❌ Tamaños de muestra inválidos.")
                st.stop()

            se = math.sqrt((s1**2 / n1_int) + (s2**2 / n2_int))
            if se == 0:
                st.error("❌ Error estándar = 0. Revisa desviaciones o tamaños.")
            else:
                t_stat = (m1 - m2) / se
                df = max(1, n1_int + n2_int - 2)
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                
                conclusion = "DIFERENCIA SIGNIFICATIVA" if p_val < alpha else "NO HAY DIFERENCIA SIGNIFICATIVA"
                color = "red" if p_val < alpha else "green"
                
                st.markdown("### Resultados")
                c1, c2 = st.columns(2)
                c1.markdown(metric_card("Diferencia de Medias", f"{(m1-m2):.2f}", "", "blue"), unsafe_allow_html=True)
                c2.markdown(metric_card("Valor P", f"{p_val:.4f}", conclusion, color), unsafe_allow_html=True)

    else: 
        with col_a:
            st.markdown("### 🅰️ Grupo 1")
            x1 = st.number_input("Éxitos 1 (x₁)", step=1.0, format="%.0f", key="x1")
            nt1 = st.number_input("Total 1 (n₁)", value=30.0, step=1.0, key="nt1")
        with col_b:
            st.markdown("### 🅱️ Grupo 2")
            x2 = st.number_input("Éxitos 2 (x₂)", step=1.0, format="%.0f", key="x2")
            nt2 = st.number_input("Total 2 (n₂)", value=30.0, step=1.0, key="nt2")
            
        alpha = st.number_input
