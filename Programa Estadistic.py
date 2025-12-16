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
    page_title="Calculadora de estadistica",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. ESTILOS CSS (TEMA DARK + SIN FLECHAS)
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
            border-bottom: 2px solid #3b82f6;
        }

        /* Ocultar flechas de los inputs numéricos (Chrome, Safari, Edge, Opera) */
        input[type=number]::-webkit-inner-spin-button, 
        input[type=number]::-webkit-outer-spin-button { 
            -webkit-appearance: none; 
            margin: 0; 
        }
        /* Ocultar flechas en Firefox */
        input[type=number] {
            -moz-appearance: textfield;
        }
        
        /* ESTILO DE LOS CAMPOS */
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

        /* BORDES DE COLOR */
        .border-purple { border-color: #a855f7; }
        .border-blue { border-color: #3b82f6; }
        .border-red { border-color: #ef4444; }
        .border-green { border-color: #22c55e; }

        /* TEXTO EXPLICATIVO */
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

st.title("Calculadora de estadistica")

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
    "Estadística Descriptiva", 
    "Inferencia de una población", 
    "Comparación (2 Pob)", 
    "Tamaño Muestra",
    "Laboratorio Visual (TLC)"
])

# =============================================================================
# 1. DESCRIPTIVA (Morado) - AHORA: obliga punto decimal
# =============================================================================
with tab1:
    st.markdown("<h3 style='color:#a855f7'>Análisis de Datos</h3>", unsafe_allow_html=True)
    
    col_in, col_out = st.columns([1, 2], gap="large")
    with col_in:
        st.info("Introduce tus números. Usa PUNTO (.) para decimales (ej: 10.5). Puedes separar números con comas, punto y coma, espacios o saltos de línea.")
        input_desc = st.text_area("Datos Numéricos:", height=150, placeholder="Ej: 10.5, 15, 12.0; 18 20")
        btn_calc_desc = st.button("Analizar Datos", key="btn1")

    with col_out:
        if btn_calc_desc:
            if not input_desc.strip():
                st.warning("⚠️ El campo está vacío.")
            else:
                # Verificación: si se detecta un número con coma decimal, forzar uso de punto
                # (rechazamos entradas tipo 10,5 que usan coma como decimal)
                if re.search(r'\d+,\d+', input_desc):
                    st.error("Por favor usa PUNTO (.) para decimales. Reemplaza '10,5' por '10.5' y vuelve a intentar.")
                else:
                    try:
                        # Dividir usando separadores comunes: coma, punto y coma, espacios y saltos de línea.
                        parts = re.split(r'[,\;\s]+', input_desc.strip())
                        tokens = [p for p in parts if p != '']
                        # Buscamos sólo valores válidos con punto decimal opcional
                        nums = []
                        for t in tokens:
                            # Validar que el token es un número con punto decimal (si tiene parte decimal)
                            if re.fullmatch(r'[-+]?(?:\d+|\d+\.\d+|\.\d+)', t):
                                nums.append(float(t))
                            else:
                                # token inválido -> mostrar mensaje y abortar
                                st.error(f"Token inválido detectado: '{t}'. Usa sólo números con '.' como decimal. Ej: 10.5")
                                nums = None
                                break

                        if nums is None or len(nums) == 0:
                            if nums is None:
                                # ya mostramos el error específico
                                pass
                            else:
                                st.error("No se detectaron números válidos. Asegúrate de usar puntos para decimales.")
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
                            rango = float(np.max(data) - np.min(data)) if n > 0 else 0.0

                            # Resultados en tarjetas
                            c1, c2, c3 = st.columns(3)
                            c1.markdown(card("Promedio (Media)", f"{media:.2f}", "", "border-purple"), unsafe_allow_html=True)
                            c2.markdown(card("Valor Central (Mediana)", f"{mediana:.2f}", "", "border-purple"), unsafe_allow_html=True)
                            c3.markdown(card("Error Estándar (EE)", f"{ee:.4f}", "", "border-purple"), unsafe_allow_html=True)
                            
                            c4, c5, c6 = st.columns(3)
                            c4.markdown(card("Desviación Estándar (s)", f"{desv:.2f}", "Muestral" if n>=2 else "Poblacional (n<2)", "border-purple"), unsafe_allow_html=True)
                            c5.markdown(card("Varianza (s^2)", f"{var:.2f}", "", "border-purple"), unsafe_allow_html=True)
                            c6.markdown(card("Rango", f"{rango:.2f}", "", "border-purple"), unsafe_allow_html=True)

                            # Análisis de Sesgo
                            sesgo = ""
                            if desv == 0:
                                sesgo = "No hay variación (todos los valores son iguales)."
                            else:
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
# 2. INFERENCIA INTELIGENTE (Azul)
# =============================================================================
with tab2:
    st.markdown("<h3 style='color:#3b82f6'>Inferencia de Una Población</h3>", unsafe_allow_html=True)
    st.write("El sistema detecta automáticamente si usar Z o T según los datos ingresados.")

    tipo_dato = st.radio("¿Qué tipo de dato tienes?", ["Promedio (Media)", "Porcentaje (Proporción)", "Posición Individual (Z)"], horizontal=True)
    st.markdown("---")

    # --- LÓGICA MEDIA ---
    if tipo_dato == "Promedio (Media)":
        c1, c2, c3 = st.columns(3)
        media = c1.number_input("Promedio Muestral ($\\overline{x}$)", step=0.01, format="%.4f")
        n = c2.number_input("Tamaño de Muestra ($n$)", value=30.0, step=1.0)
        conf = c3.number_input("Nivel de Confianza ($1-\\alpha$)", value=0.95, min_value=0.0, max_value=1.0, step=0.01, help="Por defecto es 0.95 (95%)")

        st.markdown("##### Desviación Estándar (Llena solo una)")
        col_sig, col_s = st.columns(2)
        sigma = col_sig.number_input("Poblacional ($\\sigma$) -> Activa Z", step=0.01, format="%.4f", help="Usa esta si conoces la desviación histórica.")
        s = col_s.number_input("Muestral ($s$) -> Activa T", step=0.01, format="%.4f", help="Usa esta si calculaste la desviación de la muestra actual.")

        realizar_prueba = st.checkbox("Calcular prueba de hipótesis (H0)", value=False)
        mu_hyp = 0.0
        if realizar_prueba:
            mu_hyp = st.number_input("Valor Hipotético ($\\mu_0$)", value=0.0, step=0.01, format="%.4f", help="Si llena esto, se calculará la prueba de hipótesis.")

        if st.button("Calcular Inferencia", key="btn_inf_smart"):
            # Validaciones
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
                st.error("⚠️ Debes ingresar alguna desviación positiva ($\\sigma$ o $s$).")
                st.stop()

            # Mostrar Intervalo
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

            # Mostrar Hipótesis
            if realizar_prueba:
                st.markdown("#### Resultado de Prueba de Hipótesis")
                st.write(f"Hipótesis Nula $H_0: \\mu = {mu_hyp}$")
                
                alpha_calc = 1 - conf
                conclusion = "Rechazar $H_0$ (Diferencia Significativa)" if (p_val is not None and p_val < alpha_calc) else "No Rechazar $H_0$ (Sin Diferencia Significativa)"
                color_hyp = "border-red" if (p_val is not None and p_val < alpha_calc) else "border-green"
                
                h1, h2 = st.columns(2)
                h1.markdown(card("Estadístico de Prueba", f"{test_stat:.4f}" if test_stat is not None else "N/A", "Z o T calculado", "border-blue"), unsafe_allow_html=True)
                h2.markdown(card("Valor P (P-Value)", f"{p_val:.4f}" if p_val is not None else "N/A", conclusion, color_hyp), unsafe_allow_html=True)

            # Gráfico del intervalo alrededor de la media
            fig, ax = plt.subplots(figsize=(8, 2))
            fig.patch.set_facecolor('#050505')
            ax.set_facecolor('#050505')
            if se is None:
                se = 1.0
            x_axis = np.linspace(media - 4*se, media + 4*se, 200)
            y_axis = stats.norm.pdf(x_axis, media, se)
            ax.plot(x_axis, y_axis, color='#3b82f6', lw=2)
            ax.fill_between(x_axis, y_axis, where=((x_axis >= media-margen) & (x_axis <= media+margen)), color='#3b82f6', alpha=0.3)
            ax.axvline(media, color='white', linestyle=':')
            ax.axis('off')
            st.pyplot(fig)

    # --- LÓGICA PROPORCIÓN ---
    elif tipo_dato == "Porcentaje (Proporción)":
        c1, c2, c3 = st.columns(3)
        prop = c1.number_input("Proporción ($p$) [Ej: 0.5]", value=0.5, min_value=0.0, max_value=1.0, step=0.01)
        n = c2.number_input("Tamaño de Muestra ($n$)", value=100.0, step=1.0)
        conf = c3.number_input("Nivel de Confianza ($1-\\alpha$)", value=0.95, step=0.01)

        realizar_prueba_p = st.checkbox("Calcular prueba de hipótesis para proporción", value=False)
        p_hyp = 0.0
        if realizar_prueba_p:
            p_hyp = st.number_input("Proporción Hipotética ($p_0$)", value=0.5, min_value=0.0, max_value=1.0, step=0.01)

        if st.button("Calcular Intervalo"):
            try:
                n_int = int(n)
                if n_int <= 0:
                    st.error("n debe ser mayor que 0.")
                    st.stop()
            except Exception:
                st.error("Tamaño de muestra inválido.")
                st.stop()

            q = 1 - prop
            se = math.sqrt((prop * q) / n_int) if n_int > 0 else 0.0
            z_val = stats.norm.ppf((1+conf)/2)
            margen = z_val * se

            c1, c2 = st.columns(2)
            c1.markdown(card("Límite Inferior", f"{max(0, prop-margen)*100:.2f}%", f"{max(0,prop-margen):.4f}", "border-blue"), unsafe_allow_html=True)
            c2.markdown(card("Límite Superior", f"{min(1, prop+margen)*100:.2f}%", f"{min(1,prop+margen):.4f}", "border-blue"), unsafe_allow_html=True)

            if realizar_prueba_p:
                # Usar se bajo H0
                se_hyp = math.sqrt((p_hyp*(1-p_hyp))/n_int) if n_int > 0 else 0.0
                if se_hyp == 0:
                    st.error("No es posible calcular la prueba (desviación bajo H0 = 0).")
                else:
                    z_stat = (prop - p_hyp) / se_hyp
                    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                    alpha_calc = 1 - conf
                    conclusion = "Diferencia Significativa" if p_val < alpha_calc else "No significativa"
                    st.markdown(f"<div class='simple-text'><strong>Prueba de Hipótesis:</strong> Valor P = {p_val:.4f} ({conclusion})</div>", unsafe_allow_html=True)

    # --- LÓGICA Z SCORE ---
    elif tipo_dato == "Posición Individual (Z)":
        c1, c2, c3 = st.columns(3)
        val = c1.number_input("Valor a Evaluar ($x$)", step=0.01, format="%.4f")
        mu = c2.number_input("Promedio Población ($\\mu$)", step=0.01, format="%.4f")
        sig = c3.number_input("Desviación Población ($\\sigma$)", step=0.01, format="%.4f")
        
        if st.button("Calcular Z"):
            if sig == 0:
                st.error("La desviación poblacional (σ) debe ser mayor que 0.")
            else:
                z = (val - mu) / sig
                st.markdown(card("Puntaje Z", f"{z:.4f}", "Desviaciones estándar", "border-blue"), unsafe_allow_html=True)

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
            m1 = st.number_input("Media 1 ($\\overline{x}_1$)", step=0.01, format="%.4f")
            s1 = st.number_input("Desviación 1 ($s_1$)", step=0.01, format="%.4f")
            n1 = st.number_input("Tamaño 1 ($n_1$)", value=30.0, step=1.0)
        with col_b:
            st.write("🅱️ **Grupo 2**")
            m2 = st.number_input("Media 2 ($\\overline{x}_2$)", step=0.01, format="%.4f")
            s2 = st.number_input("Desviación 2 ($s_2$)", step=0.01, format="%.4f")
            n2 = st.number_input("Tamaño 2 ($n_2$)", value=30.0, step=1.0)
            
        alpha = st.number_input("Nivel de Significancia (Alpha $\\alpha$)", value=0.05, step=0.01)

        if st.button("Comparar Grupos"):
            try:
                n1_int = int(n1)
                n2_int = int(n2)
                if n1_int <= 1 or n2_int <= 1:
                    st.error("Cada grupo debe tener n > 1 para comparar medias.")
                    st.stop()
            except Exception:
                st.error("Tamaños de muestra inválidos.")
                st.stop()

            se = math.sqrt((s1**2 / n1_int) + (s2**2 / n2_int))
            if se == 0:
                st.error("Error estándar = 0. Revisa desviaciones o tamaños.")
            else:
                t_stat = (m1 - m2) / se
                df = max(1, n1_int + n2_int - 2)
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                
                conclusion = "DIFERENCIA SIGNIFICATIVA" if p_val < alpha else "NO HAY DIFERENCIA SIGNIFICATIVA"
                color = "border-red" if p_val < alpha else "border-green"
                
                c1, c2 = st.columns(2)
                c1.markdown(card("Diferencia Medias", f"{(m1-m2):.2f}", "", "border-red"), unsafe_allow_html=True)
                c2.markdown(card("Valor P", f"{p_val:.4f}", conclusion, color), unsafe_allow_html=True)

    else: 
        with col_a:
            st.write("🅰️ **Grupo 1**")
            x1 = st.number_input("Éxitos 1 ($x_1$)", step=1.0, format="%.0f")
            nt1 = st.number_input("Total 1 ($n_1$)", value=30.0, step=1.0)
        with col_b:
            st.write("🅱️ **Grupo 2**")
            x2 = st.number_input("Éxitos 2 ($x_2$)", step=1.0, format="%.0f")
            nt2 = st.number_input("Total 2 ($n_2$)", value=30.0, step=1.0)
            
        alpha = st.number_input("Alpha ($\\alpha$)", value=0.05, step=0.01)
        
        if st.button("Comparar Porcentajes"):
            try:
                nt1_int = int(nt1)
                nt2_int = int(nt2)
                if nt1_int <= 0 or nt2_int <= 0:
                    st.error("Los totales deben ser mayores que 0.")
                    st.stop()
            except Exception:
                st.error("Totales inválidos.")
                st.stop()

            p1 = x1/nt1_int
            p2 = x2/nt2_int
            pp = (x1 + x2) / (nt1_int + nt2_int)
            se = math.sqrt(pp*(1-pp) * (1/nt1_int + 1/nt2_int))
            if se == 0:
                st.error("Error estándar = 0; no se puede comparar (posiblemente proporciones en {0,1}).")
            else:
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
    error = c1.number_input("Margen de Error (ME)", value=0.05, step=0.001, format="%.4f")
    conf = c2.number_input("Confianza ($1-\\alpha$)", value=0.95, step=0.01)
    
    if target == "Estimar Promedio":
        sigma = st.number_input("Desviación Estimada ($\\sigma$)", value=10.0, step=0.1)
        if st.button("Calcular N"):
            if error <= 0:
                st.error("El margen de error debe ser mayor que 0.")
            else:
                z = stats.norm.ppf((1+conf)/2)
                n = (z**2 * sigma**2) / (error**2)
                st.markdown(card("Tamaño de Muestra ($n$)", f"{math.ceil(n)}", "Personas/Datos", "border-green"), unsafe_allow_html=True)
    else:
        p_est = st.number_input("Proporción Estimada ($p$)", value=0.5, min_value=0.0, max_value=1.0, step=0.01)
        if st.button("Calcular N"):
            if error <= 0:
                st.error("El margen de error debe ser mayor que 0.")
            else:
                z = stats.norm.ppf((1+conf)/2)
                n = (z**2 * p_est * (1-p_est)) / (error**2)
                st.markdown(card("Tamaño de Muestra ($n$)", f"{math.ceil(n)}", "Personas/Datos", "border-green"), unsafe_allow_html=True)

# =============================================================================
# 5. LABORATORIO VISUAL (TLC)
# =============================================================================
with tab5:
    st.markdown("<h3 style='color:#ffffff'>Laboratorio Visual</h3>", unsafe_allow_html=True)
    
    tool = st.selectbox("Seleccione Simulación:", ["Teorema del Límite Central (TLC)", "Comportamiento Error Estándar"])
    
    if tool == "Teorema del Límite Central (TLC)":
        st.info("Simula cómo el promedio de muchas muestras forma una campana, sin importar la población original.")
        c1, c2 = st.columns(2)
        n_sim = c1.number_input("Tamaño de cada muestra ($n$)", value=30.0, step=1.0)
        reps = c2.number_input("Cantidad de muestras (Repeticiones)", value=1000.0, step=1.0)
        
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
                ax1.set_title("Población Original (Sesgada)", color='white')
                ax1.axis('off')
                
                ax2.set_facecolor('#111')
                ax2.hist(means, bins=30, color='#22c55e', alpha=0.8)
                ax2.set_title(f"Distribución de Medias (n={n_sim_int})", color='white')
                ax2.axis('off')
                
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error en la simulación: {e}")

    elif tool == "Comportamiento Error Estándar":
        sigma_sim = st.number_input("Desviación Poblacional Simulada", value=10.0, step=0.1)
        
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
