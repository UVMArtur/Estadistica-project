import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import math
from scipy import stats

# Configuración de la página y tema visual
st.set_page_config(page_title="Calculadora de Estadística", layout="wide", page_icon="📊")

# Estilos CSS personalizados para imitar el diseño de la imagen (tema oscuro/morado)
st.markdown("""
<style>
    /* Fondo general */
    .stApp {
        background-color: #000000;
        color: white;
    }
    
    /* Títulos */
    h1, h2, h3 {
        color: white !important;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    
    /* Input de texto */
    .stTextArea textarea {
        background-color: #111111;
        color: white;
        border: 1px solid #4B0082;
    }
    
    /* Botones */
    .stButton>button {
        background-color: #6200EA;
        color: white;
        border-radius: 20px;
        width: 100%;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #7C4DFF;
    }
    
    /* Tarjetas de resultados (Metric Cards) */
    div[data-testid="metric-container"] {
        background-color: white;
        color: black;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border: 2px solid #6200EA;
    }
    div[data-testid="metric-container"] label {
        color: #555 !important; /* Color del título pequeño (label) */
        font-size: 0.9rem;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: black !important; /* Color del número grande */
        font-weight: bold;
    }

    /* Tabs personalizados (intentando imitar la barra de navegación) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(98, 0, 234, 0.2);
        border-bottom: 2px solid #6200EA;
    }
    
    /* Línea decorativa degradada */
    .gradient-line {
        height: 8px;
        background: linear-gradient(90deg, #6200EA 0%, #00B0FF 100%);
        border-radius: 4px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- Encabezado ---
st.title("Calculadora de estadística")
st.markdown('<div class="gradient-line"></div>', unsafe_allow_html=True)

# --- Navegación (Tabs) ---
tabs = st.tabs(["Medidas de tendencia central", "Inferencia estadística", "Comparación de poblaciones", "Tamaño de muestra", "Visual LAB"])

with tabs[0]: # Trabajaremos principalmente en la primera pestaña según tu imagen
    
    col_izq, col_der = st.columns([1, 2])
    
    with col_izq:
        st.markdown("### Datos:")
        st.markdown("""
        <div style="background-color: white; padding: 10px; border-radius: 10px; color: black; margin-bottom: 10px;">
            <p style="font-size: 0.8rem; margin: 0; color: #6200EA; font-weight: bold;">
                Usa PUNTO (.) para decimales. Separa números con comas, espacios o saltos de línea.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        input_data = st.text_area("Ingresa tus datos aquí", height=150, placeholder="Ej: 3.2, 4.5, 7.8, 9.1...", label_visibility="collapsed")
        
        # Selección de tipo de datos (Muestra vs Población)
        tipo_datos = st.radio("¿Qué tipo de datos son?", ["Muestra", "Población"], horizontal=True)
        
        calcular = st.button("Analizar datos")

    if calcular and input_data:
        try:
            # 1. Procesamiento de datos
            # Reemplazar comas por espacios para facilitar el split, o manejar ambos
            raw_text = input_data.replace(',', ' ').replace(';', ' ').replace('\n', ' ')
            data_list = [float(x) for x in raw_text.split() if x.strip()]
            
            if len(data_list) == 0:
                st.error("No se encontraron números válidos.")
            else:
                df = pd.DataFrame(data_list, columns=['Valor'])
                arr = np.array(data_list)
                n = len(arr)

                # 2. Cálculos Estadísticos
                media = np.mean(arr)
                mediana = np.median(arr)
                
                # Moda (puede haber múltiples o ninguna)
                mode_result = stats.mode(arr, keepdims=True)
                # Si todos los valores son únicos, scipy a veces devuelve el más bajo. 
                # Verificamos si realmente hay moda contando frecuencias.
                vals, counts = np.unique(arr, return_counts=True)
                max_count = np.max(counts)
                
                if max_count == 1:
                    moda_str = "No hay moda"
                    moda_subtext = "(Todos los valores son únicos)"
                else:
                    modas = vals[counts == max_count]
                    moda_str = ", ".join(map(str, modas))
                    moda_subtext = f"(Se repite {max_count} veces)"

                rango = np.ptp(arr) # Peak to peak (max - min)
                
                # Grados de libertad para varianza/std: Muestra (ddof=1), Población (ddof=0)
                ddof = 1 if tipo_datos == "Muestra" else 0
                
                varianza = np.var(arr, ddof=ddof)
                desviacion_std = np.std(arr, ddof=ddof)
                
                # Error estándar (solo suele aplicar conceptualmente a muestras para inferir medias, pero lo calculamos)
                error_estandar = stats.sem(arr, ddof=ddof) if n > 1 else 0

                # 3. Mostrar Tarjetas de Resultados (estilo visual de la imagen)
                with col_der:
                    st.markdown("<br>", unsafe_allow_html=True) # Espacio
                    
                    # Fila 1
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric(label="Promedio (media)", value=f"{media:.2f}")
                    with c2:
                        st.metric(label="Mediana", value=f"{mediana:.2f}")
                    with c3:
                        st.metric(label="Moda", value=moda_str, delta=moda_subtext, delta_color="off")

                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Fila 2
                    c4, c5, c6 = st.columns(3)
                    with c4:
                        lbl_std = "Desviación estándar (s)" if tipo_datos == "Muestra" else "Desviación estándar (σ)"
                        st.metric(label=lbl_std, value=f"{desviacion_std:.2f}", delta=tipo_datos, delta_color="off")
                    with c5:
                        lbl_var = "Varianza (s²)" if tipo_datos == "Muestra" else "Varianza (σ²)"
                        st.metric(label=lbl_var, value=f"{varianza:.2f}")
                    with c6:
                        st.metric(label="Rango", value=f"{rango:.2f}")

                    # Interpretación simple (caja blanca de texto)
                    interpretation_html = f"""
                    <div style="background-color: white; color: black; padding: 15px; border-radius: 5px; margin-top: 20px; border-left: 5px solid #6200EA;">
                        <strong>Interpretación:</strong><br>
                        Con una {tipo_datos.lower()} de <strong>{n}</strong> datos, el centro se ubica en <strong>{media:.2f}</strong>. 
                        La dispersión es de <strong>{desviacion_std:.2f}</strong>.
                        {'Los datos son simétricos' if abs(media - mediana) < 0.1 * desviacion_std else 'Los datos presentan asimetría'}.
                    </div>
                    """
                    st.markdown(interpretation_html, unsafe_allow_html=True)

    # --- Sección inferior: Histograma y Frecuencias (Ancho completo) ---
    if calcular and input_data and len(data_list) > 0:
        st.markdown("---")
        st.markdown("### Histograma de Frecuencias y Regla de Sturges")
        
        # Calcular Regla de Sturges
        # k = 1 + 3.322 * log10(n)
        k = 1 + 3.322 * math.log10(n)
        num_clases = math.ceil(k) # Redondear hacia arriba para número de intervalos
        
        val_min = np.min(arr)
        val_max = np.max(arr)
        ancho_clase = (val_max - val_min) / num_clases
        
        # Ajuste pequeño para asegurar que el último valor entre
        bins = np.linspace(val_min, val_max, num_clases + 1)
        
        # Crear tabla de frecuencias
        counts, bin_edges = np.histogram(arr, bins=bins)
        
        tabla_freq = pd.DataFrame({
            'Límite Inferior': bin_edges[:-1],
            'Límite Superior': bin_edges[1:],
            'Frecuencia Absoluta (fi)': counts
        })
        tabla_freq['Marca de Clase (xi)'] = (tabla_freq['Límite Inferior'] + tabla_freq['Límite Superior']) / 2
        tabla_freq['Frecuencia Relativa (hi)'] = tabla_freq['Frecuencia Absoluta (fi)'] / n
        tabla_freq['Frecuencia Acumulada (Fi)'] = tabla_freq['Frecuencia Absoluta (fi)'].cumsum()

        col_hist, col_tabla = st.columns([2, 1])

        with col_hist:
            # Crear Histograma con Plotly
            fig = px.bar(
                tabla_freq, 
                x='Marca de Clase (xi)', 
                y='Frecuencia Absoluta (fi)',
                title=f"Histograma (Regla de Sturges: k={num_clases})",
                text='Frecuencia Absoluta (fi)', # Mostrar número encima de la barra
            )
            
            # Personalizar diseño para que parezca al de la imagen
            fig.update_traces(marker_color='#7C4DFF', textposition='outside')
            fig.update_layout(
                plot_bgcolor='black',
                paper_bgcolor='black',
                font=dict(color='white'),
                xaxis=dict(title='Grupos / Clases', showgrid=False),
                yaxis=dict(title='Frecuencia', showgrid=True, gridcolor='#333'),
                bargap=0.05 # Barras casi pegadas
            )
            
            # Añadir línea de promedio
            fig.add_vline(x=media, line_width=2, line_dash="dash", line_color="white", annotation_text="Promedio", annotation_position="top right")
            
            st.plotly_chart(fig, use_container_width=True)

        with col_tabla:
            st.markdown("#### Tabla de Frecuencias")
            st.dataframe(tabla_freq.style.format({
                'Límite Inferior': '{:.2f}',
                'Límite Superior': '{:.2f}',
                'Marca de Clase (xi)': '{:.2f}',
                'Frecuencia Relativa (hi)': '{:.4f}'
            }), height=400)
            
            st.info(f"""
            **Detalles del cálculo:**
            - **N (Datos):** {n}
            - **K (Sturges):** {num_clases} grupos
            - **Ancho calculado:** {ancho_clase:.4f}
            """)

        except ValueError:
            st.error("Error al procesar los datos. Asegúrate de ingresar solo números separados correctamente.")
