import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import math
from scipy import stats

st.set_page_config(page_title="Calculadora de Estadística", layout="wide", page_icon="📊")

# === ESTILOS ===
st.markdown("""
<style>
    .stApp { background-color: #000000; color: white; }
    h1, h2, h3 { color: white !important; text-align: center; font-family: 'Arial', sans-serif; }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 20px; justify-content: center; }
    .stTabs [data-baseweb="tab"] { color: white; }
    .stTabs [aria-selected="true"] { background-color: rgba(124, 77, 255, 0.28); border-bottom: 3px solid #7C4DFF; }
    /* Gradiente superior */
    .gradient-line { height: 8px; background: linear-gradient(90deg, #7C4DFF 0%, #00B0FF 100%); border-radius: 4px; margin-bottom: 20px; }
    /* Inputs */
    .stTextArea textarea, input[type=text] {
        background-color: #111111; color: white; border: 1px solid #7C4DFF;
    }
    /* Botones morados */
    .stButton>button {
        background-color: #7C4DFF; color: white; border-radius: 12px; width: 100%;
        border: none; font-weight: bold; padding: 14px 0;
    }
    .stButton>button:hover { background-color: #9b6bff; }
    /* Tarjetas */
    div[data-testid="metric-container"] {
        background-color: white !important; color: black !important;
        border-radius: 14px; padding: 14px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.15); text-align: center; border: 2px solid #5A86FF;
    }
    div[data-testid="metric-container"] label { color: #444 !important; font-size: 0.9rem; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: black !important; font-weight: 800; }
    /* Centrados */
    .centered { display: flex; justify-content: center; align-items: center; }
    /* Cards */
    .card-white {
        background: white; color: black; border-radius: 24px; padding: 18px 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    .card-red {
        background: white; color: #c8102e; border-radius: 18px; padding: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.20); border: 2px solid #c8102e;
    }
    .card-green {
        background: white; color: #0c7a43; border-radius: 18px; padding: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.20); border: 2px solid #0c7a43;
    }
</style>
""", unsafe_allow_html=True)

st.title("Calculadora de estadítica")
st.markdown('<div class="gradient-line"></div>', unsafe_allow_html=True)

tabs = st.tabs([
    "Medidas de tendencia central",
    "Inferencia estadística",
    "Comparación de dos poblaciones",
    "Tamaño de muestra",
    "Visual LAB"
])

# ---------------------------------------------------------------------
# PESTAÑA 1: Medidas de tendencia central
# ---------------------------------------------------------------------
with tabs[0]:
    col_izq, col_der = st.columns([1, 2])
    data_list = []

    with col_izq:
        st.markdown("### Datos:")
        st.markdown("""
        <div style="background-color: white; padding: 10px; border-radius: 10px; color: black; margin-bottom: 10px;">
            <p style="font-size: 0.8rem; margin: 0; color: #6200EA; font-weight: bold;">
                Usa PUNTO (.) para decimales. Separa números con comas, espacios o saltos de línea.
            </p>
        </div>
        """, unsafe_allow_html=True)

        input_data = st.text_area("Ingresa tus datos aquí", height=150,
                                  placeholder="Ej: 13, 19, 25, 31 ...",
                                  label_visibility="collapsed")
        st.markdown('<div class="centered"><b>¿Qué tipo de datos son?</b></div>', unsafe_allow_html=True)
        tipo_datos = st.radio("", ["Muestra", "Población"], horizontal=True)
        calcular = st.button("Analizar datos")

    if input_data:
        try:
            raw_text = input_data.replace(',', ' ').replace(';', ' ').replace('\n', ' ')
            data_list = [float(x) for x in raw_text.split() if x.strip()]
        except ValueError:
            st.error("⚠️ Error: Uno de los valores ingresados no es un número válido.")
            data_list = []

    if calcular and input_data and len(data_list) > 0:
        df = pd.DataFrame(data_list, columns=['Valor'])
        arr = np.array(data_list)
        n = len(arr)

        media = np.mean(arr)
        mediana = np.median(arr)
        skewness = stats.skew(arr)
        if skewness > 0.3:
            sesgo = "Sesgo a la derecha (cola a la derecha)."
        elif skewness < -0.3:
            sesgo = "Sesgo a la izquierda (cola a la izquierda)."
        else:
            sesgo = "Distribución aproximadamente simétrica."
        vals, counts = np.unique(arr, return_counts=True)
        max_count = np.max(counts)
        if max_count == 1:
            moda_str = "No hay moda"
            moda_subtext = "(Todos únicos)"
        else:
            modas = vals[counts == max_count]
            moda_str = "Multimodal" if len(modas) > 5 else ", ".join(map(str, modas))
            moda_subtext = f"(Repite {max_count} veces)"

        rango = np.ptp(arr)
        ddof = 1 if tipo_datos == "Muestra" else 0
        varianza = np.var(arr, ddof=ddof)
        desviacion_std = np.std(arr, ddof=ddof)

        with col_der:
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Promedio (media)", f"{media:.2f}")
            with c2: st.metric("Mediana", f"{mediana:.2f}")
            with c3: st.metric("Moda", moda_str, delta=moda_subtext, delta_color="off")

            st.markdown("<br>", unsafe_allow_html=True)
            c4, c5, c6 = st.columns(3)
            with c4:
                lbl_std = "Desviación estándar (s)" if tipo_datos == "Muestra" else "Desviación estándar (σ)"
                st.metric(lbl_std, f"{desviacion_std:.2f}", delta=tipo_datos, delta_color="off")
            with c5:
                lbl_var = "Varianza (s²)" if tipo_datos == "Muestra" else "Varianza (σ²)"
                st.metric(lbl_var, f"{varianza:.2f}")
            with c6:
                st.metric("Rango", f"{rango:.2f}")

            interpretation_html = f"""
            <div style="background-color: white; color: black; padding: 15px; border-radius: 5px; margin-top: 20px; border-left: 5px solid #6200EA;">
                <strong>Interpretación:</strong><br>
                Con una {tipo_datos.lower()} de <strong>{n}</strong> datos, el centro se ubica en <strong>{media:.2f}</strong>.
                La dispersión es de <strong>{desviacion_std:.2f}</strong>.<br>
                {sesgo}
            </div>
            """
            st.markdown(interpretation_html, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Histograma y tabla de Frecuencias")

        # Regla de Sturges
        if n > 1:
            k = math.ceil(1 + 3.322 * math.log10(n))
        else:
            k = 1

        val_min = np.min(arr)
        val_max = np.max(arr)

        if val_min == val_max:
            width = 1
            display_edges = [math.floor(val_min), math.floor(val_min) + width]
        else:
            val_min_int = math.floor(val_min)
            val_max_int = math.ceil(val_max)
            rango_int = val_max_int - val_min_int
            width = math.ceil(rango_int / k)
            display_edges = [val_min_int + i * width for i in range(k + 1)]
            display_edges[-1] = val_max_int

        bin_edges = [display_edges[0] - 0.5] + [edge + 0.5 for edge in display_edges[1:]]
        counts, _ = np.histogram(arr, bins=bin_edges)

        grupos = np.arange(1, k + 1)
        tabla_freq = pd.DataFrame({
            'Grupo': grupos,
            'Límite Inferior': display_edges[:-1],
            'Límite Superior': display_edges[1:],
            'Frecuencia Absoluta (fi)': counts
        })
        tabla_freq['Marca de Clase (xi)'] = (tabla_freq['Límite Inferior'] + tabla_freq['Límite Superior']) / 2
        tabla_freq['Frecuencia Relativa (hi)'] = tabla_freq['Frecuencia Absoluta (fi)'] / n if n > 0 else 0
        tabla_freq['Frecuencia Acumulada (Fi)'] = tabla_freq['Frecuencia Absoluta (fi)'].cumsum()

        col_hist, col_tabla = st.columns([2, 1])

        with col_hist:
            fig = px.bar(
                tabla_freq,
                x='Grupo',
                y='Frecuencia Absoluta (fi)',
                title=f"Histograma (k={k}, ancho={width:.4f})",
                text='Frecuencia Absoluta (fi)',
            )
            fig.update_traces(marker_color='#7C4DFF', textposition='outside')
            fig.update_layout(
                plot_bgcolor='black',
                paper_bgcolor='black',
                font=dict(color='white'),
                xaxis=dict(title='Grupo', tickmode='linear'),
                yaxis=dict(title='Frecuencia', showgrid=True, gridcolor='#333'),
                bargap=0.05
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_tabla:
            st.markdown("#### Tabla de Frecuencias")
            st.dataframe(tabla_freq.style.format({
                'Límite Inferior': '{:.0f}',
                'Límite Superior': '{:.0f}',
                'Marca de Clase (xi)': '{:.2f}',
                'Frecuencia Relativa (hi)': '{:.4f}'
            }), height=400)
            st.info(f"**N:** {n} | **Grupos (k):** {k} | **Ancho:** {width}")

# ---------------------------------------------------------------------
# PESTAÑA 2: Inferencia estadística (Media y Proporción) con subíndices
# ---------------------------------------------------------------------
with tabs[1]:
    st.markdown("## Inferencia de Una Población")
    st.markdown('<div class="gradient-line"></div>', unsafe_allow_html=True)

    st.markdown('<div class="centered"><b>¿Qué tipo de dato tienes?</b></div>', unsafe_allow_html=True)
    tipo_inferencia = st.radio(
        "",
        ["Promedio (Media)", "Porcentaje (Proporción)"],
        horizontal=True,
        index=0
    )

    st.markdown("### Datos:")

    def to_float(txt, default=None):
        txt = str(txt).strip()
        if txt == "":
            return default
        try:
            return float(txt)
        except:
            return None

    colA, colB, colC = st.columns(3)
    if tipo_inferencia == "Promedio (Media)":
        with colA:
            x_bar_txt = st.text_input("Promedio muestral (x̄)", value="0")
        with colB:
            n_txt = st.text_input("Tamaño de muestra (n)", value="30")
        with colC:
            nivel_txt = st.text_input("Nivel de confianza (1−α) %", value="95", key="nivel_conf_media_inf")

        colD, colE = st.columns(2)
        with colD:
            sigma_txt = st.text_input("Desviación estándar poblacional (σ) (opcional)", value="")
        with colE:
            s_txt = st.text_input("Desviación estándar muestral (s) (opcional)", value="")

        colF, colG = st.columns(2)
        with colF:
            usar_hipotesis = st.checkbox("Calcular prueba de hipótesis (H₀)")
        with colG:
            mu0_txt = st.text_input("Valor hipotético (μ₀)", value="0", disabled=not usar_hipotesis)
    else:  # Proporción
        with colA:
            x_success_txt = st.text_input("Número de éxitos (x)", value="0")
        with colB:
            n_txt = st.text_input("Tamaño de muestra (n)", value="30")
        with colC:
            nivel_txt = st.text_input("Nivel de confianza (1−α) %", value="95", key="nivel_conf_prop_inf")

        colF, colG = st.columns(2)
        with colF:
            usar_hipotesis = st.checkbox("Calcular prueba de hipótesis (H₀)")
        with colG:
            mu0_txt = st.text_input("Proporción hipotética (p₀)", value="0.5", disabled=not usar_hipotesis)

    calcular_inf = st.button("Calcular Inferencia")

    if calcular_inf:
        n_inf = to_float(n_txt, None)
        nivel_conf = to_float(nivel_txt, 95)

        if n_inf is None or n_inf <= 0:
            st.error("n debe ser > 0")
        else:
            alpha = 1 - (nivel_conf / 100.0)
            se = None
            crit = None
            df = None
            lower = None
            upper = None
            metodo = ""
            stat_test = None
            decision = ""
            p_value = None

            if tipo_inferencia == "Promedio (Media)":
                x_bar = to_float(x_bar_txt, 0)
                sigma = to_float(sigma_txt, 0)
                s_muestral = to_float(s_txt, 0)
                mu0 = to_float(mu0_txt, 0)

                if sigma and sigma > 0:
                    se = sigma / math.sqrt(n_inf)
                    metodo = "Normal (Z) - σ conocida"
                    crit = stats.norm.ppf(1 - alpha/2)
                    lower = x_bar - crit * se
                    upper = x_bar + crit * se
                    if usar_hipotesis:
                        stat_test = (x_bar - mu0) / se
                        p_value = 2 * (1 - stats.norm.cdf(abs(stat_test)))
                        decision = "Rechaza H₀" if abs(stat_test) > crit else "No se rechaza H₀"
                else:
                    if s_muestral is None or s_muestral <= 0:
                        st.error("Falta σ o s para media.")
                    else:
                        se = s_muestral / math.sqrt(n_inf)
                        df = n_inf - 1
                        crit = stats.t.ppf(1 - alpha/2, df)
                        metodo = "t-student - σ desconocida"
                        lower = x_bar - crit * se
                        upper = x_bar + crit * se
                        if usar_hipotesis:
                            stat_test = (x_bar - mu0) / se
                            p_value = 2 * (1 - stats.t.cdf(abs(stat_test), df))
                            decision = "Rechaza H₀" if abs(stat_test) > crit else "No se rechaza H₀"

            else:  # Proporción
                x_success = to_float(x_success_txt, 0)
                mu0 = to_float(mu0_txt, 0.5)
                p_hat = x_success / n_inf if n_inf else 0
                se = math.sqrt(p_hat * (1 - p_hat) / n_inf) if n_inf else 0
                crit = stats.norm.ppf(1 - alpha/2)
                metodo = "Normal (Z) - Proporción"
                lower = p_hat - crit * se
                upper = p_hat + crit * se
                if usar_hipotesis and se > 0:
                    stat_test = (p_hat - mu0) / se
                    p_value = 2 * (1 - stats.norm.cdf(abs(stat_test)))
                    decision = "Rechaza H₀" if abs(stat_test) > crit else "No se rechaza H₀"

            if se is not None:
                st.markdown("---")
                st.markdown("### Resultados")

                row1 = st.columns(3)
                if tipo_inferencia == "Promedio (Media)":
                    with row1[0]: st.metric("Promedio muestral (x̄)", f"{x_bar:.4f}")
                else:
                    with row1[0]: st.metric("Proporción muestral (p̂)", f"{p_hat:.4f}")
                with row1[1]: st.metric("Error estándar", f"{se:.4f}")
                if df is not None:
                    with row1[2]: st.metric("Valor crítico t", f"{crit:.4f}" if crit else "—")
                else:
                    with row1[2]: st.metric("Valor crítico Z", f"{crit:.4f}" if crit else "—")

                if lower is not None and upper is not None:
                    row2 = st.columns(2)
                    with row2[0]: st.metric("Límite Inferior", f"{lower:.4f}")
                    with row2[1]: st.metric("Límite Superior", f"{upper:.4f}")

                if stat_test is not None and p_value is not None:
                    row3 = st.columns(3)
                    with row3[0]: st.metric("Estadístico de prueba", f"{stat_test:.4f}")
                    with row3[1]: st.metric("p-value (bilateral)", f"{p_value:.4f}")
                    with row3[2]: st.metric("Decisión", decision)

                interp = []
                if lower is not None and upper is not None:
                    interp.append(f"Con un {nivel_conf:.1f}% de confianza, el verdadero parámetro está entre {lower:.4f} y {upper:.4f}.")
                interp.append(f"Método usado: {metodo}. Error estándar: {se:.4f}.")
                st.markdown("#### Interpretación / Prueba")
                st.markdown(f"<div class='card-white'>{'<br>'.join(interp)}</div>", unsafe_allow_html=True)

                if stat_test is not None:
                    st.markdown(
                        f"<div style='background:#1b3a90;color:white;padding:12px;border-radius:12px;"
                        f"border:2px solid #7C4DFF;font-weight:700;margin-top:12px;'>"
                        f"Prueba de hipótesis: estadístico = {stat_test:.4f}. Decisión: {decision}."
                        f"</div>", unsafe_allow_html=True)

                # Curva
                if lower is not None and upper is not None:
                    center = (lower + upper) / 2
                    xs = np.linspace(center - 4*(upper-lower), center + 4*(upper-lower), 400)
                    ys = stats.norm.pdf(xs, loc=center, scale=se if se>0 else 1)
                    fig_curve = go.Figure()
                    fig_curve.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line=dict(color='#5A86FF'), name='Distribución'))
                    fig_curve.add_vrect(x0=lower, x1=upper, fillcolor='rgba(124,77,255,0.25)', line_width=0, annotation_text="IC")
                    fig_curve.add_vline(x=center, line_width=2, line_dash="dash", line_color="white", annotation_text="Centro")
                    fig_curve.update_layout(
                        plot_bgcolor='black',
                        paper_bgcolor='black',
                        font=dict(color='white'),
                        xaxis_title="Valor", yaxis_title="Densidad"
                    )
                    st.plotly_chart(fig_curve, use_container_width=True)

# ---------------------------------------------------------------------
# PESTAÑA 3: Comparación de dos poblaciones (medias y proporciones)
# ---------------------------------------------------------------------
with tabs[2]:
    st.markdown("## Comparación de Dos Grupos")
    st.markdown('<div class="gradient-line" style="background: linear-gradient(90deg,#c8102e 0%,#ff6b6b 100%);"></div>', unsafe_allow_html=True)

    st.markdown('<div class="centered"><b>Seleccione Análisis</b></div>', unsafe_allow_html=True)
    analisis = st.radio("", ["Diferencia de Medias", "Diferencia de Proporciones"], horizontal=True, index=0)

    st.markdown("### Datos:")

    def tf(txt, default=None):
        txt = str(txt).strip()
        if txt == "":
            return default
        try:
            return float(txt)
        except:
            return None

    if analisis == "Diferencia de Medias":
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("<div class='card-red'>", unsafe_allow_html=True)
            st.markdown("**Grupo 1**")
            x1_txt = st.text_input("Media 1 (x̄₁)", value="0")
            s1_txt = st.text_input("Desviación 1 (s₁)", value="1")
            n1_txt = st.text_input("Tamaño 1 (n₁)", value="30")
            st.markdown("</div>", unsafe_allow_html=True)
        with g2:
            st.markdown("<div class='card-red'>", unsafe_allow_html=True)
            st.markdown("**Grupo 2**")
            x2_txt = st.text_input("Media 2 (x̄₂)", value="0")
            s2_txt = st.text_input("Desviación 2 (s₂)", value="1")
            n2_txt = st.text_input("Tamaño 2 (n₂)", value="30")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("<div class='card-red'>", unsafe_allow_html=True)
            st.markdown("**Grupo 1**")
            x1_txt = st.text_input("Número de éxitos (x₁)", value="0")
            n1_txt = st.text_input("Tamaño 1 (n₁)", value="30")
            p1_txt = st.text_input("Proporción muestral (p̂₁) (opcional)", value="")
            st.markdown("</div>", unsafe_allow_html=True)
        with g2:
            st.markdown("<div class='card-red'>", unsafe_allow_html=True)
            st.markdown("**Grupo 2**")
            x2_txt = st.text_input("Número de éxitos (x₂)", value="0")
            n2_txt = st.text_input("Tamaño 2 (n₂)", value="30")
            p2_txt = st.text_input("Proporción muestral (p̂₂) (opcional)", value="")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="centered"><b>Nivel de Confianza (1−α) %</b></div>', unsafe_allow_html=True)
    nivel_cmp_txt = st.text_input("", value="95", key="nivel_cmp_tab3")

    calcular_cmp = st.button("Calcular comparación")

    if calcular_cmp:
        nivel_cmp = tf(nivel_cmp_txt, 95)
        alpha = 1 - (nivel_cmp / 100.0)

        if analisis == "Diferencia de Medias":
            x1 = tf(x1_txt, 0); x2 = tf(x2_txt, 0)
            s1 = tf(s1_txt, 1); s2 = tf(s2_txt, 1)
            n1 = tf(n1_txt, None); n2 = tf(n2_txt, None)
            if not n1 or not n2 or n1 <=0 or n2<=0:
                st.error("n₁ y n₂ deben ser > 0")
            else:
                diff = x1 - x2
                se = math.sqrt((s1**2 / n1) + (s2**2 / n2))
                crit = stats.norm.ppf(1 - alpha/2)
                me = crit * se
                lower = diff - me
                upper = diff + me
                if se > 0:
                    z_stat = diff / se
                    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                else:
                    z_stat = 0; p_val = 1

                st.markdown("### Resultados", unsafe_allow_html=True)
                r1 = st.columns(3)
                with r1[0]: st.metric("Diferencia de medias", f"{diff:.4f}")
                with r1[1]: st.metric("Error estándar combinado", f"{se:.4f}")
                with r1[2]: st.metric("Margen de error (ME)", f"{me:.4f}")

                r2 = st.columns(2)
                with r2[0]: st.metric("Límite Inferior", f"{lower:.4f}")
                with r2[1]: st.metric("Límite Superior", f"{upper:.4f}")

                st.markdown("#### Interpretación del Intervalo:")
                texto = (
                    f"Con un {nivel_cmp:.1f}% de confianza, la diferencia verdadera entre las medias poblacionales "
                    f"está entre {lower:.4f} y {upper:.4f}.<br>"
                    f"Método usado: Normal (Z) – muestras independientes, comparación de medias. "
                    f"Error estándar combinado: {se:.4f}."
                )
                st.markdown(f"<div class='card-white'>{texto}</div>", unsafe_allow_html=True)

                st.markdown(
                    f"<div style='background:#1b3a90;color:white;padding:12px;border-radius:12px;"
                    f"border:2px solid #7C4DFF;font-weight:700;margin-top:12px;'>"
                    f"Prueba de hipótesis (H₀: μ₁ − μ₂ = 0): Z = {z_stat:.4f}, p = {p_val:.4f}. "
                    f"Decisión: {'Rechaza H₀' if p_val < alpha else 'No se rechaza H₀'}."
                    f"</div>", unsafe_allow_html=True)

        else:  # Diferencia de Proporciones
            x1 = tf(x1_txt, 0); x2 = tf(x2_txt, 0)
            n1 = tf(n1_txt, None); n2 = tf(n2_txt, None)
            if not n1 or not n2 or n1 <=0 or n2<=0:
                st.error("n₁ y n₂ deben ser > 0")
            else:
                if p1_txt.strip():
                    p1_hat = tf(p1_txt, 0)
                else:
                    p1_hat = x1 / n1 if n1 else 0
                if p2_txt.strip():
                    p2_hat = tf(p2_txt, 0)
                else:
                    p2_hat = x2 / n2 if n2 else 0

                diff = p1_hat - p2_hat
                se = math.sqrt(p1_hat*(1-p1_hat)/n1 + p2_hat*(1-p2_hat)/n2)
                crit = stats.norm.ppf(1 - alpha/2)
                me = crit * se
                lower = diff - me
                upper = diff + me
                if se > 0:
                    z_stat = diff / se
                    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                else:
                    z_stat = 0; p_val = 1

                st.markdown("### Resultados", unsafe_allow_html=True)
                r1 = st.columns(3)
                with r1[0]: st.metric("Estimador puntual (p̂₁ − p̂₂)", f"{diff:.4f}")
                with r1[1]: st.metric("Error estándar combinado", f"{se:.4f}")
                with r1[2]: st.metric("Valor crítico Z", f"{crit:.4f}")

                r2 = st.columns(3)
                with r2[0]: st.metric("Margen de error (ME)", f"{me:.4f}")
                with r2[1]: st.metric("Límite Inferior", f"{lower:.4f}")
                with r2[2]: st.metric("Límite Superior", f"{upper:.4f}")

                st.markdown("#### Interpretación del Intervalo:")
                texto = (
                    f"Con un {nivel_cmp:.1f}% de confianza, la diferencia verdadera entre las proporciones poblacionales "
                    f"está entre {lower:.4f} y {upper:.4f}.<br>"
                    f"Método usado: Normal (Z) – muestras independientes, diferencia de proporciones. "
                    f"Error estándar combinado: {se:.4f}."
                )
                st.markdown(f"<div class='card-white'>{texto}</div>", unsafe_allow_html=True)

                st.markdown(
                    f"<div style='background:#1b3a90;color:white;padding:12px;border-radius:12px;"
                    f"border:2px solid #7C4DFF;font-weight:700;margin-top:12px;'>"
                    f"Prueba de hipótesis (H₀: p₁ − p₂ = 0): Z = {z_stat:.4f}, p = {p_val:.4f}. "
                    f"Decisión: {'Rechaza H₀' if p_val < alpha else 'No se rechaza H₀'}."
                    f"</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# PESTAÑA 4: Tamaño de muestra (media / proporción, con corrección finita)
# ---------------------------------------------------------------------
with tabs[3]:
    st.markdown("## Tamaño de Muestra")
    st.markdown('<div class="gradient-line" style="background: linear-gradient(90deg,#d7df01 0%,#00e6b8 100%);"></div>', unsafe_allow_html=True)

    st.markdown('<div class="centered"><b>Seleccione Análisis</b></div>', unsafe_allow_html=True)
    tipo_n = st.radio("", ["Por media", "Por proporción"], horizontal=True, index=0)

    st.markdown("### Datos:")

    def tf2(txt, default=None):
        txt = str(txt).strip()
        if txt == "":
            return default
        try:
            return float(txt)
        except:
            return None

    if tipo_n == "Por media":
        c1, c2 = st.columns(2)
        with c1:
            sigma_txt = st.text_input("Desviación estándar poblacional (σ)", value="", key="sigma_media")
        with c2:
            s_txt = st.text_input("Desviación estándar muestral (s)", value="", key="s_media")

        c3, c4 = st.columns(2)
        with c3:
            e_txt = st.text_input("Margen de error deseado (E)", value="1", key="E_media")
        with c4:
            nivel_txt = st.text_input("Nivel de confianza (1−α) %", value="95", key="nivel_conf_media_n")

    else:  # Por proporción
        c1, c2 = st.columns(2)
        with c1:
            p_hat_txt = st.text_input("Proporción esperada (p̂) (0-1)", value="0.5", key="phat_prop")
        with c2:
            e_txt = st.text_input("Margen de error deseado (E)", value="0.05", key="E_prop")

        c3, c4 = st.columns(2)
        with c3:
            nivel_txt = st.text_input("Nivel de confianza (1−α) %", value="95", key="nivel_conf_prop_n")
        with c4:
            pass

    st.markdown("<br>", unsafe_allow_html=True)
    colF1, colF2 = st.columns(2)
    with colF1:
        calc_finite = st.checkbox("Calcular por población finita")
    with colF2:
        N_txt = st.text_input("Tamaño de población (N)", value="", disabled=not calc_finite, key="N_finite")

    calcular_n = st.button("Calcular Muestra")

    if calcular_n:
        nivel_conf = tf2(nivel_txt, 95)
        alpha = 1 - (nivel_conf / 100.0)
        z = stats.norm.ppf(1 - alpha/2)

        n0 = None
        metodo = ""

        if tipo_n == "Por media":
            sigma = tf2(sigma_txt, 0)
            s = tf2(s_txt, 0)
            E = tf2(e_txt, None)
            if E is None or E <= 0:
                st.error("Margen de error debe ser > 0")
            else:
                sd_use = sigma if sigma and sigma > 0 else s
                if sd_use is None or sd_use <= 0:
                    st.error("Proporciona σ o s para estimar la media.")
                else:
                    n0 = (z * sd_use / E) ** 2
                    metodo = f"Normal (Z) – Planeación de tamaño de muestra para media (σ/s={sd_use})."
        else:
            p_hat = tf2(p_hat_txt, 0.5)
            E = tf2(e_txt, None)
            if E is None or E <= 0:
                st.error("Margen de error debe ser > 0")
            else:
                n0 = (z**2 * p_hat * (1 - p_hat)) / (E**2)
                metodo = f"Normal (Z) – Planeación de tamaño de muestra para proporción (p̂={p_hat})."

        if n0 is not None:
            n_req = math.ceil(n0)
            usado_finite = False
            if calc_finite:
                N = tf2(N_txt, None)
                if N and N > 0:
                    n_req = math.ceil((N * n0) / (N + n0 - 1))
                    usado_finite = True

            st.markdown("---")
            st.markdown("## Resultado")
            cres = st.columns([1, 2])
            with cres[0]:
                st.markdown(
                    f"<div class='card-green' style='text-align:center;'>"
                    f"<div style='font-weight:700;'>Tamaño de la muestra requerida (n)</div>"
                    f"<div style='font-size:32px;'>{n_req}</div>"
                    f"<div style='font-size:12px;'>Participantes</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with cres[1]:
                detalle = f"Con un {nivel_conf:.1f}% de confianza, se requieren al menos {n_req} observaciones"
                if tipo_n == "Por media":
                    detalle += f" para estimar la media con un margen de error de {E}."
                else:
                    detalle += f" para estimar la proporción con un margen de error de {E}."
                if usado_finite:
                    detalle += f" Considerando población finita N={int(tf2(N_txt,0))}."
                interp = (
                    f"{detalle}<br>"
                    f"Método usado: {metodo}"
                )
                st.markdown(f"<div class='card-white'>{interp}</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# PESTAÑA 5: Visual LAB (placeholder)
# ---------------------------------------------------------------------
with tabs[4]:
    st.markdown("### Visual LAB")
    st.info("En construcción. Aquí podrás agregar visualizaciones personalizadas.")
