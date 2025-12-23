import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import math
from scipy import stats

st.set_page_config(page_title="Calculadora de Estadística", layout="wide", page_icon="📊")

# Estilos CSS
st.markdown("""
<style>
    .stApp { background-color: #000000; color: white; }
    h1, h2, h3 { color: white !important; text-align: center; font-family: 'Arial', sans-serif; }
    .stTextArea textarea { background-color: #111111; color: white; border: 1px solid #4B0082; }
    .stButton>button {
        background-color: #6200EA; color: white; border-radius: 20px; width: 100%;
        border: none; font-weight: bold;
    }
    .stButton>button:hover { background-color: #7C4DFF; }
    /* Tarjetas blancas */
    div[data-testid="metric-container"] {
        background-color: white !important; color: black !important;
        border-radius: 10px; padding: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; border: 2px solid #6200EA;
    }
    div[data-testid="metric-container"] label {
        color: black !important; font-size: 0.9rem;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: black !important; font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 20px; justify-content: center; }
    .stTabs [data-baseweb="tab"] { color: white; }
    .stTabs [aria-selected="true"] { background-color: rgba(98, 0, 234, 0.2); border-bottom: 2px solid #6200EA; }
    .gradient-line {
        height: 8px; background: linear-gradient(90deg, #6200EA 0%, #00B0FF 100%);
        border-radius: 4px; margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Calculadora de estadística")
st.markdown('<div class="gradient-line"></div>', unsafe_allow_html=True)

tabs = st.tabs(["Medidas de tendencia central", "Inferencia estadística", "Comparación de poblaciones", "Tamaño de muestra", "Visual LAB"])

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
                                  placeholder="Ej: 3.2, 4.5, 7.8, 9.1...",
                                  label_visibility="collapsed")
        tipo_datos = st.radio("¿Qué tipo de datos son?", ["Muestra", "Población"], horizontal=True)
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

        # Medidas
        media = np.mean(arr)
        mediana = np.median(arr)
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
                La dispersión es de <strong>{desviacion_std:.2f}</strong>.
            </div>
            """
            st.markdown(interpretation_html, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Histograma de Frecuencias y Regla de Sturges")

        # Regla de Sturges
        if n > 1:
            k = math.ceil(1 + 3.322 * math.log10(n))
        else:
            k = 1

        val_min = np.min(arr)
        val_max = np.max(arr)

        # Construcción de intervalos sin solape (enteros)
        # Ancho entero (ceil), límites mostrados como enteros cerrados [LI, LS]
        if val_min == val_max:
            width = 1
            display_edges = [math.floor(val_min), math.floor(val_min) + width]
        else:
            val_min_int = math.floor(val_min)
            val_max_int = math.ceil(val_max)
            rango_int = val_max_int - val_min_int
            width = math.ceil(rango_int / k)  # amplitud (entera)
            display_edges = [val_min_int + i * width for i in range(k + 1)]
            # Forzar que el último cubra el máximo redondeado
            display_edges[-1] = val_max_int

        # Bordes continuos para histogram (para contar sin solape): [-0.5, +0.5] alrededor de enteros
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
