```python name=stat_suite_final_enhanced.py
import re
import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN E IMPORTACIONES
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Calculadora de estaditica",
    page_icon="🧬",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. ESTILOS CSS (TEMA DARK + INPUTS/RESULTS EN BLANCO + TABS CENTRADAS)
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
        header, footer { visibility: hidden; }
        .stApp { background-color: #050505; }

        /* TITULO CENTRADO Y MÁS GRANDE */
        .app-title {
            text-align: center;
            font-weight: 900;
            font-size: 36px;
            margin-bottom: 10px;
            color: #ffffff;
        }

        /* LÍNEA DECORATIVA */
        .page-line {
            height: 6px;
            width: 70%;
            margin: 8px auto 22px auto;
            border-radius: 6px;
            background: linear-gradient(90deg, #a855f7, #3b82f6);
            box-shadow: 0 4px 12px rgba(59,130,246,0.12);
        }

        /* PESTAÑAS ESTILIZADAS CENTRADAS */
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px;
            background-color: transparent;
            padding: 8px 10px;
            border-radius: 12px;
            border: none;
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 0 auto 20px auto;
        }
        .stTabs [data-baseweb="tab"] {
            min-width: 220px;
            height: 46px;
            border-radius: 12px;
            font-weight: 700;
            color: #888;
            background-color: #0a0a0a;
            border: 1px solid #222;
            transition: all 0.25s;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #111;
            color: #fff;
            border: 1px solid #3b82f6;
            box-shadow: 0 6px 20px rgba(59,130,246,0.08);
        }

        /* INPUTS Y TEXTAREAS: FONDO BLANCO REDONDEADO (mantener fondo negro general) */
        .stNumberInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div, textarea {
            background-color: #ffffff !important;
            color: #111 !important;
            border: 1px solid #e6e6e6 !important;
            border-radius: 12px !important;
            padding: 10px !important;
        }
        .stNumberInput input:focus, .stTextArea textarea:focus {
            border-color: #3b82f6 !important;
            outline: none !important;
            box-shadow: 0 6px 18px rgba(59,130,246,0.08);
        }

        /* RESULTADOS EN TARJETAS BLANCAS REDONDEADAS */
        .result-card {
            background-color: #ffffff;
            color: #111;
            padding: 18px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 6px 18px rgba(0,0,0,0.45);
            margin-bottom: 16px;
            border-top: 6px solid;
            transition: transform 0.15s;
        }
        .result-card:hover { transform: translateY(-3px); }
        .card-label { font-size: 0.75rem; text-transform: uppercase; font-weight: 800; color: #6b7280; margin-bottom: 6px; }
        .card-value { font-size: 1.6rem; font-weight: 900; color: #111; }
        .card-sub { font-size: 0.85rem; color: #444; margin-top: 6px; font-style: italic; }

        .border-purple { border-top-color: #a855f7; }
        .border-blue { border-top-color: #3b82f6; }
        .border-red { border-top-color: #ef4444; }
        .border-green { border-top-color: #22c55e; }

        /* CAJA BLANCA PARA SECCIONES (INPUT Y RESULTADOS) */
        .panel-box {
            background: #ffffff;
            color: #111;
            border-radius: 12px;
            padding: 14px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.45);
            margin-bottom: 18px;
        }

        /* SEPARADOR ENTRE DATOS Y RESPUESTAS */
        .sep-line {
            height: 2px;
            background: linear-gradient(90deg, rgba(168,85,247,0.9), rgba(59,130,246,0.9));
            margin: 12px 0 18px 0;
            border-radius: 4px;
        }

        /* TEXTO EXPLICATIVO (oscuro) */
        .simple-text {
            background: rgba(255,255,255,0.03);
            border-left: 3px solid #666;
            padding: 12px;
            border-radius: 0 8px 8px 0;
            margin-top: 12px;
            color: #ccc;
            font-size: 0.95rem;
        }

        /* BOTONES ESTÉTICOS */
        div.stButton > button {
            width: 100%;
            border-radius: 8px;
            font-weight: 700;
            padding: 10px;
            background-color: #111;
            color: white;
            border: 1px solid #444;
        }
        div.stButton > button:hover {
            border-color: #fff;
            background-color: #222;
        }

    </style>
""", unsafe_allow_html=True)

# TÍTULO PRINCIPAL (centrado, grande)
st.markdown("<div class='app-title'>Calculadora de estaditica</div>", unsafe_allow_html=True)
st.markdown("<div class='page-line'></div>", unsafe_allow_html=True)

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
# ESTRUCTURA DE PESTAÑAS (ahora sin emojis y con nombres claros)
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Estadística Descriptiva",
    "Inferencia Inteligente",
    "Comparación de Dos Poblaciones",
    "Tamaño de Muestra",
    "Laboratorio Visual"
])

# ------------------
# UTIL: calcular moda(s)
# ------------------
def compute_modes(arr):
    vals, counts = np.unique(arr, return_counts=True)
    maxc = counts.max()
    if maxc == 1:
        return []  # no moda
    modes = vals[counts == maxc].tolist()
    return modes

# =============================================================================
# 1. DESCRIPTIVA
# =============================================================================
with tab1:
    st.markdown("<h3 style='color:#a855f7'>Análisis de Datos</h3>", unsafe_allow_html=True)

    # panel input (blanco)
    st.markdown("<div class='panel-box'>", unsafe_allow_html=True)
    st.info("Usa PUNTO (.) para decimales. Separa números con comas, punto y coma, espacios o saltos de línea.")
    input_desc = st.text_area("Datos Numéricos:", height=150, placeholder="Ej: 10.5, 15, 12.0; 18 20")
    btn_calc_desc = st.button("Analizar Datos", key="btn1")
    st.markdown("</div>", unsafe_allow_html=True)

    # separación visual
    st.markdown("<div class='sep-line'></div>", unsafe_allow_html=True)

    # resultados
    if btn_calc_desc:
        if not input_desc.strip():
            st.warning("⚠️ El campo está vacío.")
        else:
            # Rechazar si detecta coma decimal (forzar punto)
            if re.search(r'\d+,\d+', input_desc):
                st.error("Por favor usa PUNTO (.) para decimales. Reemplaza '10,5' por '10.5' y vuelve a intentar.")
            else:
                try:
                    parts = re.split(r'[,\;\s]+', input_desc.strip())
                    tokens = [p for p in parts if p != '']
                    nums = []
                    for t in tokens:
                        if re.fullmatch(r'[-+]?(?:\d+|\d+\.\d+|\.\d+)', t):
                            nums.append(float(t))
                        else:
                            st.error(f"Token inválido detectado: '{t}'. Usa sólo números con '.' como decimal. Ej: 10.5")
                            nums = None
                            break

                    if nums is None:
                        pass
                    elif len(nums) == 0:
                        st.error("No se detectaron números válidos. Asegúrate de usar puntos para decimales.")
                    else:
                        data = np.array(nums, dtype=float)
                        n = data.size
                        media = float(np.mean(data))
                        mediana = float(np.median(data))
                        modes = compute_modes(data)
                        if n >= 2:
                            desv = float(np.std(data, ddof=1))
                            var = float(np.var(data, ddof=1))
                        else:
                            desv = float(np.std(data, ddof=0))
                            var = float(np.var(data, ddof=0))
                        ee = desv / math.sqrt(n) if n > 0 else 0.0

                        # Mostrar tarjetas (3 columnas)
                        c1, c2, c3 = st.columns(3)
                        c1.markdown(card("Promedio (media)", f"{media:.2f}", "", "border-purple"), unsafe_allow_html=True)
                        c2.markdown(card("Mediana", f"{mediana:.2f}", "", "border-purple"), unsafe_allow_html=True)
                        # moda
                        if len(modes) == 0:
                            moda_text = "No hay moda (todos los valores son únicos)"
                            moda_val = "—"
                        else:
                            moda_text = f"Moda(s): {', '.join([str(round(m,4)) for m in modes])}"
                            moda_val = ", ".join([str(round(m,4)) for m in modes])
                        c3.markdown(card("Moda", moda_val, moda_text, "border-purple"), unsafe_allow_html=True)

                        c4, c5, c6 = st.columns(3)
                        c4.markdown(card("Desviación estándar (s)", f"{desv:.2f}", "Muestral" if n>=2 else "Poblacional (n<2)", "border-blue"), unsafe_allow_html=True)
                        c5.markdown(card("Varianza (s^2)", f"{var:.2f}", "", "border-blue"), unsafe_allow_html=True)
                        c6.markdown(card("Error estándar (EE)", f"{ee:.4f}", "", "border-blue"), unsafe_allow_html=True)

                        # Interpretación resumida
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
                        <div class="simple-text" style="border-left-color: #a855f7;">
                            <strong>Interpretación:</strong><br>
                            Con una muestra de <b>{n}</b> datos, el centro se ubica en <b>{media:.2f}</b>. La dispersión (s) es de <b>{desv:.2f}</b>.<br>
                            <em>{sesgo}</em>
                        </div>
                        """, unsafe_allow_html=True)

                        # Histograma
                        st.write("#### Histograma de Frecuencias")
                        fig, ax = plt.subplots(figsize=(10, 3))
                        fig.patch.set_facecolor('#050505')
                        ax.set_facecolor('#111')
                        counts, bins, patches = ax.hist(data, bins='auto', color='#a855f7', edgecolor='black', alpha=0.9)
                        try:
                            ax.bar_label(patches, fmt='%.0f', color='white', padding=3, fontweight='bold')
                        except Exception:
                            pass
                        ax.axvline(media, color='white', linestyle='--', label='Promedio')
                        ax.legend(facecolor='#222', labelcolor='white', frameon=False)
                        ax.axis('off')
                        st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error al procesar los datos: {e}")

# =============================================================================
# 2. INFERENCIA INTELIGENTE
# =============================================================================
with tab2:
    st.markdown("<h3 style='color:#3b82f6'>Inferencia de Una Población</h3>", unsafe_allow_html=True)
    st.write("El sistema detecta Z o T según lo ingresado. Se muestra también el margen de error usado en el intervalo.")

    tipo_dato = st.radio("Tipo de dato:", ["Promedio (media)", "Proporción", "Posición individual (Z)"], horizontal=True)
    st.markdown("---")

    if tipo_dato == "Promedio (media)":
        c1, c2, c3 = st.columns(3)
        media = c1.number_input("Promedio muestral (x̄)", step=0.01, format="%.4f")
        n = c2.number_input("Tamaño de muestra (n)", value=30.0, step=1.0)
        conf = c3.number_input("Confianza (1-α)", value=0.95, min_value=0.0, max_value=1.0, step=0.01)

        st.markdown("##### Desviación estándar (llenar solo una)")
        col_sig, col_s = st.columns(2)
        sigma = col_sig.number_input("Poblacional (σ) -> Z", step=0.01, format="%.4f")
        s = col_s.number_input("Muestral (s) -> T", step=0.01, format="%.4f")

        realizar_prueba = st.checkbox("Calcular prueba de hipótesis (H0)")
        mu_hyp = 0.0
        if realizar_prueba:
            mu_hyp = st.number_input("Valor hipotético (μ0)", value=0.0, step=0.01, format="%.4f")

        if st.button("Calcular Inferencia", key="btn_inf"):
            try:
                n_int = int(n)
                if n_int <= 0:
                    st.error("n debe ser mayor que 0.")
                    st.stop()
            except Exception:
                st.error("Tamaño de muestra inválido.")
                st.stop()

            se = None
            dist_label = ""
            margen = None
            test_stat = None
            p_val = None

            if sigma and sigma > 0:
                se = sigma / math.sqrt(n_int)
                z_val = stats.norm.ppf((1 + conf) / 2)
                margen = z_val * se
                dist_label = "Z (σ conocida)"
                if realizar_prueba:
                    test_stat = (media - mu_hyp) / se
                    p_val = 2 * (1 - stats.norm.cdf(abs(test_stat)))
            elif s and s > 0:
                se = s / math.sqrt(n_int)
                if n_int >= 30:
                    z_val = stats.norm.ppf((1 + conf) / 2)
                    margen = z_val * se
                    dist_label = "Z (muestra grande)"
                    if realizar_prueba:
                        test_stat = (media - mu_hyp) / se
                        p_val = 2 * (1 - stats.norm.cdf(abs(test_stat)))
                else:
                    t_val = stats.t.ppf((1 + conf) / 2, df=n_int - 1)
                    margen = t_val * se
                    dist_label = "T-Student (muestra pequeña)"
                    if realizar_prueba:
                        test_stat = (media - mu_hyp) / se
                        p_val = 2 * (1 - stats.t.cdf(abs(test_stat), df=n_int - 1))
            else:
                st.error("Ingresa σ o s (positivo).")
                st.stop()

            # Mostrar intervalo y margen
            li = media - margen
            ls = media + margen
            c1, c2, c3 = st.columns(3)
            c1.markdown(card("Límite inferior", f"{li:.4f}", "", "border-blue"), unsafe_allow_html=True)
            c2.markdown(card("Límite superior", f"{ls:.4f}", "", "border-blue"), unsafe_allow_html=True)
            c3.markdown(card("Margen de error", f"{margen:.4f}", f"{conf*100:.0f}% de confianza", "border-blue"), unsafe_allow_html=True)

            # frase explícita
            st.markdown(f"<div class='simple-text' style='border-left-color: #3b82f6;'><strong>Con un {conf*100:.0f}% de confianza, el promedio poblacional está entre {li:.4f} y {ls:.4f}.</strong></div>", unsafe_allow_html=True)

            # prueba de hipótesis (si solicitada)
            if realizar_prueba:
                alpha_calc = 1 - conf
                rej = (p_val is not None) and (p_val < alpha_calc)
                conclusion = "Rechazar H0: hay diferencia significativa." if rej else "No se rechaza la hipótesis nula, no hay diferencia significativa."
                color = "border-red" if rej else "border-green"
                h1, h2 = st.columns(2)
                h1.markdown(card("Estadístico de prueba", f"{test_stat:.4f}" if test_stat is not None else "N/A", dist_label, "border-blue"), unsafe_allow_html=True)
                h2.markdown(card("Valor P", f"{p_val:.4f}" if p_val is not None else "N/A", conclusion, color), unsafe_allow_html=True)

    elif tipo_dato == "Proporción":
        c1, c2, c3 = st.columns(3)
        prop = c1.number_input("Proporción (p)", value=0.5, min_value=0.0, max_value=1.0, step=0.01)
        n = c2.number_input("Tamaño de muestra (n)", value=100.0, step=1.0)
        conf = c3.number_input("Confianza (1-α)", value=0.95, step=0.01)

        hacer_prueba = st.checkbox("Calcular prueba de hipótesis para proporción")
        p_hyp = 0.0
        if hacer_prueba:
            p_hyp = st.number_input("Proporción hipotética (p0)", value=0.5, min_value=0.0, max_value=1.0, step=0.01)

        if st.button("Calcular Intervalo"):
            try:
                n_int = int(n)
                if n_int <= 0:
                    st.error("n debe ser mayor que 0.")
                    st.stop()
            except Exception:
                st.error("n inválido.")
                st.stop()

            q = 1 - prop
            se = math.sqrt((prop * q) / n_int) if n_int > 0 else 0.0
            z_val = stats.norm.ppf((1 + conf) / 2)
            margen = z_val * se
            li = max(0, prop - margen)
            ls = min(1, prop + margen)

            c1, c2, c3 = st.columns(3)
            c1.markdown(card("Límite inferior", f"{li*100:.2f}%", f"{li:.4f}", "border-blue"), unsafe_allow_html=True)
            c2.markdown(card("Límite superior", f"{ls*100:.2f}%", f"{ls:.4f}", "border-blue"), unsafe_allow_html=True)
            c3.markdown(card("Margen de error", f"{margen*100:.4f}%", f"{conf*100:.0f}% confianza", "border-blue"), unsafe_allow_html=True)

            st.markdown(f"<div class='simple-text' style='border-left-color: #3b82f6;'>Con un {conf*100:.0f}% de confianza, la proporción poblacional está entre {li:.4f} y {ls:.4f}.</div>", unsafe_allow_html=True)

            if hacer_prueba:
                se_hyp = math.sqrt((p_hyp * (1 - p_hyp)) / n_int) if n_int > 0 else 0.0
                if se_hyp == 0:
                    st.error("No se puede calcular la prueba (se bajo H0 = 0).")
                else:
                    z_stat = (prop - p_hyp) / se_hyp
                    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                    alpha_calc = 1 - conf
                    rej = p_val < alpha_calc
                    conclusion = "Diferencia significativa" if rej else "No significativa"
                    st.markdown(f"<div class='simple-text'><strong>Prueba de Hipótesis:</strong> Valor P = {p_val:.4f} ({conclusion})</div>", unsafe_allow_html=True)

    elif tipo_dato == "Posición individual (Z)":
        c1, c2, c3 = st.columns(3)
        val = c1.number_input("Valor (x)", step=0.01, format="%.4f")
        mu = c2.number_input("Media poblacional (μ)", step=0.01, format="%.4f")
        sig = c3.number_input("Desviación poblacional (σ)", step=0.01, format="%.4f")

        if st.button("Calcular Z"):
            if sig == 0:
                st.error("σ debe ser mayor que 0.")
            else:
                z = (val - mu) / sig
                st.markdown(card("Puntaje Z", f"{z:.4f}", "Desviaciones estándar", "border-blue"), unsafe_allow_html=True)

# =============================================================================
# 3. COMPARACIÓN DE DOS POBLACIONES
# =============================================================================
with tab3:
    st.markdown("<h3 style='color:#ef4444'>Comparación de dos poblaciones</h3>", unsafe_allow_html=True)
    opcion = st.selectbox("Seleccione análisis:", ["Diferencia de medias", "Diferencia de proporciones"])
    st.markdown("---")

    if opcion == "Diferencia de medias":
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("Grupo 1")
            m1 = st.number_input("Media 1", step=0.01, format="%.4f", key="m1")
            s1 = st.number_input("Desviación 1 (s1)", step=0.01, format="%.4f", key="s1")
            n1 = st.number_input("Tamaño 1 (n1)", value=30.0, step=1.0, key="n1")
        with col_b:
            st.write("Grupo 2")
            m2 = st.number_input("Media 2", step=0.01, format="%.4f", key="m2")
            s2 = st.number_input("Desviación 2 (s2)", step=0.01, format="%.4f", key="s2")
            n2 = st.number_input("Tamaño 2 (n2)", value=30.0, step=1.0, key="n2")

        alpha = st.number_input("Nivel de significancia (α)", value=0.05, step=0.01, format="%.4f")

        if st.button("Comparar grupos"):
            try:
                n1_int = int(n1)
                n2_int = int(n2)
                if n1_int <= 1 or n2_int <= 1:
                    st.error("Cada grupo debe tener n > 1.")
                    st.stop()
            except Exception:
                st.error("Tamaños inválidos.")
                st.stop()

            se = math.sqrt((s1 ** 2 / n1_int) + (s2 ** 2 / n2_int))
            if se == 0:
                st.error("Error estándar = 0 (revisa s o n).")
            else:
                t_stat = (m1 - m2) / se
                df = max(1, n1_int + n2_int - 2)
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                rej = p_val < alpha
                conclusion_text = ""
                if rej:
                    if m1 > m2:
                        conclusion_text = "Grupo 1 tiene una media significativamente mayor que Grupo 2."
                    elif m2 > m1:
                        conclusion_text = "Grupo 2 tiene una media significativamente mayor que Grupo 1."
                    else:
                        conclusion_text = "Se detecta diferencia significativa, dirección neutra (medias iguales dentro de redondeo)."
                else:
                    conclusion_text = "No se rechaza la hipótesis nula, no hay diferencia significativa."

                c1, c2 = st.columns(2)
                c1.markdown(card("Diferencia de medias (m1 - m2)", f"{(m1 - m2):.4f}", "", "border-red"), unsafe_allow_html=True)
                c2.markdown(card("Valor P", f"{p_val:.4f}", conclusion_text, "border-green" if not rej else "border-red"), unsafe_allow_html=True)

                # opcional: gráfico de columnas con error bars
                if st.checkbox("Mostrar gráfica comparativa con EE"):
                    means = [m1, m2]
                    ses = [s1 / math.sqrt(n1_int) if n1_int > 0 else 0, s2 / math.sqrt(n2_int) if n2_int > 0 else 0]
                    labels = ['Grupo 1', 'Grupo 2']
                    fig, ax = plt.subplots(figsize=(6, 4))
                    fig.patch.set_facecolor('#050505')
                    ax.set_facecolor('#111')
                    ax.bar(labels, means, yerr=ses, color=['#a855f7', '#3b82f6'], capsize=8)
                    ax.set_ylabel("Media")
                    ax.set_title("Comparación de medias (con EE)")
                    ax.tick_params(colors='white')
                    st.pyplot(fig)

    else:  # Diferencia de proporciones
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("Grupo 1")
            x1 = st.number_input("Éxitos 1 (x1)", step=1.0, format="%.0f", key="x1")
            nt1 = st.number_input("Total 1 (n1)", value=30.0, step=1.0, key="nt1")
        with col_b:
            st.write("Grupo 2")
            x2 = st.number_input("Éxitos 2 (x2)", step=1.0, format="%.0f", key="x2")
            nt2 = st.number_input("Total 2 (n2)", value=30.0, step=1.0, key="nt2")

        alpha = st.number_input("Alpha (α)", value=0.05, step=0.01, format="%.4f")

        if st.button("Comparar proporciones"):
            try:
                nt1_int = int(nt1)
                nt2_int = int(nt2)
                if nt1_int <= 0 or nt2_int <= 0:
                    st.error("Totales deben ser > 0.")
                    st.stop()
            except Exception:
                st.error("Totales inválidos.")
                st.stop()

            p1 = x1 / nt1_int if nt1_int > 0 else 0
            p2 = x2 / nt2_int if nt2_int > 0 else 0
            pp = (x1 + x2) / (nt1_int + nt2_int) if (nt1_int + nt2_int) > 0 else 0
            se = math.sqrt(pp * (1 - pp) * (1 / nt1_int + 1 / nt2_int)) if (nt1_int + nt2_int) > 0 else 0
            if se == 0:
                st.error("Error estándar = 0; no se puede comparar (posiblemente proporciones extremas 0 o 1).")
            else:
                z = (p1 - p2) / se
                p_val = 2 * (1 - stats.norm.cdf(abs(z)))
                rej = p_val < alpha
                if rej:
                    direction = "mayor" if p1 > p2 else "menor" if p1 < p2 else "igual"
                    conclusion_text = f"La diferencia de proporciones es significativa al nivel del {alpha*100:.1f}%. Grupo 1 tiene proporción {direction} que Grupo 2."
                else:
                    conclusion_text = "No se rechaza la hipótesis nula, no hay diferencia significativa."

                c1, c2 = st.columns(2)
                c1.markdown(card("Diferencia % (p1 - p2)", f"{(p1 - p2) * 100:.3f}%", "", "border-red"), unsafe_allow_html=True)
                c2.markdown(card("Valor P", f"{p_val:.4f}", conclusion_text, "border-green" if not rej else "border-red"), unsafe_allow_html=True)

                if st.checkbox("Mostrar gráfica comparativa de proporciones"):
                    labels = ['Grupo 1', 'Grupo 2']
                    props = [p1 * 100, p2 * 100]
                    ses = [math.sqrt(p1 * (1 - p1) / nt1_int) * 100 if nt1_int > 0 else 0, math.sqrt(p2 * (1 - p2) / nt2_int) * 100 if nt2_int > 0 else 0]
                    fig, ax = plt.subplots(figsize=(6, 4))
                    fig.patch.set_facecolor('#050505')
                    ax.set_facecolor('#111')
                    ax.bar(labels, props, yerr=ses, color=['#a855f7', '#3b82f6'], capsize=8)
                    ax.set_ylabel("Proporción (%)")
                    ax.set_title("Comparación de proporciones (con EE)")
                    ax.tick_params(colors='white')
                    st.pyplot(fig)

# =============================================================================
# 4. TAMAÑO DE MUESTRA
# =============================================================================
with tab4:
    st.markdown("<h3 style='color:#22c55e'>Calculadora de tamaño de muestra</h3>", unsafe_allow_html=True)
    target = st.radio("Objetivo:", ["Estimar promedio", "Estimar proporción"], horizontal=True)
    st.markdown("---")
    c1, c2 = st.columns(2)
    error = c1.number_input("Margen de error (ME)", value=0.05, step=0.001, format="%.4f")
    conf = c2.number_input("Confianza (1-α)", value=0.95, step=0.01)

    if target == "Estimar promedio":
        sigma = st.number_input("Desviación estimada (σ)", value=10.0, step=0.1)
        if st.button("Calcular N"):
            if error <= 0:
                st.error("ME debe ser > 0.")
            else:
                z = stats.norm.ppf((1 + conf) / 2)
                n = (z ** 2 * sigma ** 2) / (error ** 2)
                st.markdown(card("Tamaño de muestra (n)", f"{math.ceil(n)}", "Personas/Datos", "border-green"), unsafe_allow_html=True)
    else:
        p_est = st.number_input("Proporción estimada (p)", value=0.5, min_value=0.0, max_value=1.0, step=0.01)
        if st.button("Calcular N (proporción)"):
            if error <= 0:
                st.error("ME debe ser > 0.")
            else:
                z = stats.norm.ppf((1 + conf) / 2)
                n = (z ** 2 * p_est * (1 - p_est)) / (error ** 2)
                st.markdown(card("Tamaño de muestra (n)", f"{math.ceil(n)}", "Personas/Datos", "border-green"), unsafe_allow_html=True)

# =============================================================================
# 5. LABORATORIO VISUAL (TLC)
# =============================================================================
with tab5:
    st.markdown("<h3 style='color:#ffffff'>Laboratorio Visual</h3>", unsafe_allow_html=True)
    tool = st.selectbox("Seleccione simulación:", ["Teorema del Límite Central (TLC)", "Comportamiento Error Estándar"])
    st.markdown("---")

    if tool == "Teorema del Límite Central (TLC)":
        c1, c2 = st.columns(2)
        n_sim = c1.number_input("Tamaño de cada muestra (n)", value=30.0, step=1.0)
        reps = c2.number_input("Cantidad de muestras (repeticiones)", value=1000.0, step=1.0)
        if st.button("Simular"):
            try:
                n_sim_int = max(1, int(n_sim))
                reps_int = max(1, int(reps))
                pop = np.random.exponential(scale=1.0, size=10000)
                means = [np.mean(np.random.choice(pop, n_sim_int)) for _ in range(reps_int)]
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                fig.patch.set_facecolor('#050505')
                ax1.set_facecolor('#111')
                ax1.hist(pop, bins=30, color='#444')
                ax1.set_title("Población original (sesgada)", color='white')
                ax1.axis('off')
                ax2.set_facecolor('#111')
                ax2.hist(means, bins=30, color='#22c55e', alpha=0.8)
                ax2.set_title(f"Distribución de medias (n={n_sim_int})", color='white')
                ax2.axis('off')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error en la simulación: {e}")

    else:
        sigma_sim = st.number_input("Desviación poblacional simulada", value=10.0, step=0.1)
        if st.button("Generar curva"):
            ns = np.arange(1, 200)
            ees = sigma_sim / np.sqrt(ns)
            fig, ax = plt.subplots(figsize=(8, 3))
            fig.patch.set_facecolor('#050505')
            ax.set_facecolor('#111')
            ax.plot(ns, ees, color='#3b82f6', lw=3)
            ax.set_xlabel("Tamaño de muestra (n)", color='white')
            ax.set_ylabel("Error estándar", color='white')
            ax.grid(color='#333', linestyle='--')
            ax.tick_params(colors='white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            st.pyplot(fig)
```
