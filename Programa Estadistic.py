import re
import math
import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Calculadora de Estadística",
    layout="wide",
    page_icon="🧬"
)

# -----------------------------------------------------------------------------
# 2. ESTILOS CSS AVANZADOS (REPLICA EXACTA)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;700;900&display=swap');

    /* FONDO GENERAL */
    .stApp {
        background-color: #000000 !important;
        font-family: 'Outfit', sans-serif;
    }

    /* OCULTAR ELEMENTOS NATIVOS MOLESTOS */
    header, footer {visibility: hidden;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* --- TITULO PRINCIPAL --- */
    .main-title {
        text-align: center;
        color: white;
        font-size: 3.5rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
    }
    
    .gradient-bar {
        height: 12px;
        width: 100%;
        background: linear-gradient(90deg, #7c3aed 0%, #3b82f6 50%, #2dd4bf 100%);
        border-radius: 10px;
        margin-bottom: 3rem;
        box-shadow: 0 0 15px rgba(124, 58, 237, 0.3);
    }

    /* --- PESTAÑAS (TABS) PERSONALIZADAS --- */
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        gap: 2rem;
        background-color: transparent;
        border: none;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border: none;
        color: #888;
        font-size: 1.1rem;
        font-weight: 400;
        padding-bottom: 10px;
    }
    /* Colores específicos para la línea activa de cada tab */
    .stTabs [data-baseweb="tab-list"] button:nth-child(1)[aria-selected="true"] {
        color: white; border-bottom: 4px solid #7c3aed !important; font-weight: 700;
    }
    .stTabs [data-baseweb="tab-list"] button:nth-child(2)[aria-selected="true"] {
        color: white; border-bottom: 4px solid #3b82f6 !important; font-weight: 700;
    }
    .stTabs [data-baseweb="tab-list"] button:nth-child(3)[aria-selected="true"] {
        color: white; border-bottom: 4px solid #ef4444 !important; font-weight: 700;
    }
    .stTabs [data-baseweb="tab-list"] button:nth-child(4)[aria-selected="true"] {
        color: white; border-bottom: 4px solid #22c55e !important; font-weight: 700;
    }
    .stTabs [data-baseweb="tab-list"] button:nth-child(5)[aria-selected="true"] {
        color: white; border-bottom: 4px solid #ffffff !important; font-weight: 700;
    }

    /* --- PANEL IZQUIERDO (CONTENEDOR BLANCO) --- */
    /* Hack para estilizar la columna específica de Streamlit */
    div[data-testid="column"]:nth-of-type(1) > div {
        background-color: #ffffff;
        border-radius: 30px;
        padding: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }

    /* Titulo Datos */
    .datos-title {
        color: #6d28d9;
        font-size: 2.2rem;
        font-weight: 900;
        margin-bottom: 5px;
        border-bottom: 4px solid #8b5cf6;
        display: inline-block;
        width: 60%;
    }

    /* Caja azul de instrucción */
    .info-box {
        background-color: #5b68e6;
        color: white;
        padding: 12px;
        font-size: 0.9rem;
        border-radius: 8px;
        margin-top: 15px;
        margin-bottom: 15px;
        line-height: 1.4;
    }

    /* TEXTAREA (Input negro) */
    .stTextArea textarea {
        background-color: #000000 !important;
        color: white !important;
        border: 1px solid #333 !important;
        border-radius: 0px !important; /* Cuadrado como en la imagen */
        font-family: monospace;
        height: 150px;
    }
    .stTextArea label { display: none; } /* Ocultar label nativo */

    /* BOTÓN ANALIZAR DATOS */
    div.stButton > button {
        background-color: #7c3aed !important;
        color: white !important;
        border-radius: 50px !important;
        border: none !important;
        padding: 12px 30px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        width: 100%;
        margin-top: 10px;
        transition: transform 0.2s;
    }
    div.stButton > button:hover {
        background-color: #6d28d9 !important;
        transform: scale(1.02);
    }

    /* --- SEPARADOR VERTICAL --- */
    .vertical-line {
        border-left: 3px solid #6d28d9;
        height: 100%;
        margin: 0 auto;
        opacity: 0.8;
    }

    /* --- TARJETAS DE RESULTADOS (DERECHA) --- */
    .result-grid {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 15px;
    }
    
    .stat-card {
        background-color: white;
        border-radius: 15px;
        padding: 15px 5px;
        text-align: center;
        color: black;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .stat-label {
        font-size: 0.8rem;
        color: #888;
        font-weight: 700;
        margin-bottom: 5px;
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: 900;
        color: #000;
    }
    
    .stat-sub {
        font-size: 0.7rem;
        color: #666;
        font-style: italic;
        margin-top: 2px;
    }

    /* Caja de Interpretación */
    .interpret-box {
        background-color: #f0f0f0; /* Un blanco grisaceo ligero */
        background: linear-gradient(90deg, #ffffff 0%, #f9fafb 100%);
        border-left: 6px solid #7c3aed;
        color: #000;
        padding: 15px;
        margin-top: 20px;
        font-size: 0.9rem;
    }
    .interpret-title {
        font-weight: 900;
        margin-bottom: 5px;
        color: #000;
    }

    /* --- HISTOGRAMA --- */
    .hist-header {
        color: #7c3aed;
        font-size: 2rem;
        font-weight: 900;
        text-decoration: underline;
        text-decoration-thickness: 3px;
        margin-top: 40px;
        margin-bottom: 20px;
    }
    
    /* Ajustes para errores */
    .stAlert {
        background-color: #222 !important;
        color: #ffcccc !important;
        border: 1px solid #ff4444 !important;
    }

</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. LÓGICA DE NEGOCIO (FUNCIONES PYTHON)
# -----------------------------------------------------------------------------
def parse_data(text):
    # Validar que NO haya comas decimales (ej: 10,5)
    if re.search(r'\d+,\d+', text):
        return None, "Error: Se detectaron comas decimales. Usa PUNTO (.) ej: 10.5"
    
    # Reemplazar caracteres raros y separar
    clean_text = text.replace(';', ' ').replace(',', ' ').replace('\n', ' ')
    tokens = clean_text.split()
    
    nums = []
    for t in tokens:
        try:
            nums.append(float(t))
        except ValueError:
            return None, f"Error: '{t}' no es un número válido."
            
    if not nums:
        return None, "El campo está vacío."
        
    return np.array(nums), None

def get_modes(data):
    vals, counts = np.unique(data, return_counts=True)
    max_count = counts.max()
    if max_count == 1:
        return []
    return vals[counts == max_count].tolist()

# -----------------------------------------------------------------------------
# 4. ESTRUCTURA DE LA INTERFAZ
# -----------------------------------------------------------------------------

# Encabezado
st.markdown('<div class="main-title">Calculadora de estaditica</div>', unsafe_allow_html=True)
st.markdown('<div class="gradient-bar"></div>', unsafe_allow_html=True)

# Pestañas
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Medidas de tendencia central", 
    "Inferencia estadística", 
    "Comparación de dos poblaciones", 
    "Tamaño de muestra", 
    "Visual LAB"
])

# --- PESTAÑA 1: DESCRIPTIVA (DISEÑO IDÉNTICO IMAGEN) ---
with tab1:
    # Creamos columnas: Panel Izquierdo | Separador | Panel Derecho
    # Usamos ratios para ajustar el ancho
    col_left, col_sep, col_right = st.columns([1, 0.05, 1.5])

    # --- PANEL IZQUIERDO (Inputs) ---
    with col_left:
        # Título y caja azul (HTML puro)
        st.markdown("""
            <div class="datos-title">Datos:</div>
            <div class="info-box">
                Usa PUNTO (.) para decimales. Separa números con
                comas, punto y coma, espacios o saltos de línea.
            </div>
        """, unsafe_allow_html=True)
        
        # Widget de Streamlit (Text Area)
        # Nota: El CSS lo vuelve negro y cuadrado
        input_data = st.text_area("input_label", value="3.2, 4.5, 7.8, 9.1, 0.6, 12.3, 14.7", label_visibility="collapsed")
        
        # Botón (Streamlit button)
        # El CSS lo vuelve morado y redondo
        calc_btn = st.button("Analizar datos")

    # --- SEPARADOR VERTICAL ---
    with col_sep:
        st.markdown('<div class="vertical-line"></div>', unsafe_allow_html=True)

    # --- PANEL DERECHO (Resultados) ---
    with col_right:
        if calc_btn:
            data, error = parse_data(input_data)
            
            if error:
                st.error(error)
            else:
                # Cálculos
                n = len(data)
                media = np.mean(data)
                mediana = np.median(data)
                modas = get_modes(data)
                
                # Desviación y Varianza (Muestral si n > 1)
                ddof = 1 if n > 1 else 0
                desv = np.std(data, ddof=ddof)
                var = np.var(data, ddof=ddof)
                ee = desv / np.sqrt(n)
                
                # Formateo de moda
                if not modas:
                    moda_val = "—"
                    moda_sub = "No hay moda<br>(todos los valores son únicos)"
                else:
                    moda_val = ", ".join([f"{m:.2f}" for m in modas])
                    moda_sub = "Moda(s)"

                # --- GRID DE TARJETAS (HTML PURO) ---
                # Usamos f-strings de Python para inyectar los valores
                st.markdown(f"""
                <div class="result-grid">
                    <div class="stat-card">
                        <div class="stat-label">Promedio (media)</div>
                        <div class="stat-value">{media:.2f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Mediana</div>
                        <div class="stat-value">{mediana:.2f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Moda</div>
                        <div class="stat-value">{moda_val}</div>
                        <div class="stat-sub">{moda_sub}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Desviación estándar (s)</div>
                        <div class="stat-value">{desv:.2f}</div>
                        <div class="stat-sub">Muestral</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Varianza (s^2)</div>
                        <div class="stat-value">{var:.2f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Error estándar (EE)</div>
                        <div class="stat-value">{ee:.4f}</div>
                    </div>
                </div>
                
                <div class="interpret-box">
                    <div class="interpret-title">Interpretación:</div>
                    Con una muestra de <b>{n}</b> datos, el centro se ubica en <b>{media:.2f}</b>. 
                    La dispersión (s) es de <b>{desv:.2f}</b>. 
                    Los datos son {"bastante simétricos" if abs(media-mediana) < desv/10 else "sesgados"}.
                </div>
                """, unsafe_allow_html=True)
                
                # Guardar en session state para el histograma
                st.session_state['last_data'] = data
                st.session_state['last_mean'] = media

        else:
            # Estado vacio (Placeholder visual para que no se vea feo al inicio)
             st.markdown("""
                <div class="result-grid" style="opacity: 0.5;">
                    <div class="stat-card"><div class="stat-label">Promedio</div><div class="stat-value">-</div></div>
                    <div class="stat-card"><div class="stat-label">Mediana</div><div class="stat-value">-</div></div>
                    <div class="stat-card"><div class="stat-label">Moda</div><div class="stat-value">-</div></div>
                </div>
                <div style="text-align:center; margin-top:20px; color:#666;">
                    Presiona "Analizar datos" para ver resultados
                </div>
            """, unsafe_allow_html=True)

# --- HISTOGRAMA (FUERA DE LAS COLUMNAS, ANCHO COMPLETO) ---
# Solo se muestra en la Tab 1 si hay datos calculados
if calc_btn and 'last_data' in st.session_state:
    st.markdown('<div class="hist-header">Histograma de Frecuencias:</div>', unsafe_allow_html=True)
    
    data = st.session_state['last_data']
    media = st.session_state['last_mean']
    
    # Crear figura con fondo negro para igualar la imagen
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # Histograma morado
    counts, bins, patches = ax.hist(data, bins='auto', color='#8b5cf6', edgecolor='black', alpha=0.9)
    
    # Etiquetas de valor encima de las barras (blanco)
    for count, patch in zip(counts, patches):
        if count > 0:
            height = patch.get_height()
            ax.text(patch.get_x() + patch.get_width() / 2, height + 0.1, 
                    str(int(count)), ha='center', color='white', fontweight='bold', fontsize=12)
    
    # Línea promedio
    ax.axvline(media, color='white', linestyle='--', linewidth=2)
    ax.text(media + (max(data)*0.02), max(counts), 'Promedio', color='white', fontsize=12)
    
    # Eliminar ejes para que parezca "flotante" como en la imagen
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#444')
    ax.tick_params(axis='x', colors='white')
    ax.set_yticks([]) # Quitar eje Y
    
    st.pyplot(fig)

# --- OTRAS PESTAÑAS (PLACEHOLDERS CON ESTILO) ---
# Se mantiene la lógica simple pero respetando el tema oscuro
with tab2:
    st.header("Inferencia Estadística")
    st.info("Funcionalidad disponible en versiones completas.")
with tab3:
    st.header("Comparación de Poblaciones")
    st.info("Funcionalidad disponible en versiones completas.")
with tab4:
    st.header("Tamaño de Muestra")
    st.info("Funcionalidad disponible en versiones completas.")
with tab5:
    st.header("Laboratorio Visual")
    st.info("Funcionalidad disponible en versiones completas.")
