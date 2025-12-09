import re
import math
import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# -------------------------
# Configuración de página
# -------------------------
st.set_page_config(page_title="Calculadora Estadística", layout="wide", page_icon="🧮")

# -------------------------
# Lógica de Cálculo (Python)
# -------------------------
def parse_strict_point(text: str):
    if re.search(r'\d+,\d+', text):
        return None, "Error: Usa PUNTO (.) para decimales. Ejemplo: 10.5 (no 10,5)."
    parts = re.split(r'[,\;\n\r]+|\s+', text.strip())
    tokens = [p for p in parts if p != ""]
    nums = []
    for t in tokens:
        if re.fullmatch(r'[-+]?(?:\d+|\d+\.\d+|\.\d+)', t):
            nums.append(float(t))
        else:
            return None, f"Error: Token inválido '{t}'."
    if not nums:
        return None, "No hay datos."
    return np.array(nums, dtype=float), None

def compute_modes(arr):
    vals, counts = np.unique(arr, return_counts=True)
    maxc = counts.max()
    if maxc == 1:
        return []
    return vals[counts == maxc].tolist()

# -------------------------
# Generación de Gráfico (Matplotlib -> Base64)
# -------------------------
def get_hist_image(data, mean_val):
    fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
    fig.patch.set_facecolor('#000000') # Fondo negro total
    ax.set_facecolor('#000000')
    
    # Histograma morado exacto
    counts, bins, patches = ax.hist(data, bins='auto', color='#8b5cf6', edgecolor='black', alpha=1.0)
    
    # Números encima de las barras (blancos y negrita)
    for p, c in zip(patches, counts):
        if c > 0:
            x = p.get_x() + p.get_width() / 2
            y = p.get_height()
            ax.text(x, y + 0.05 * max(counts), f"{int(c)}", ha='center', va='bottom', 
                    color='white', fontweight='bold', fontsize=12)
    
    # Línea promedio punteada blanca
    ax.axvline(mean_val, color='white', linestyle='--', linewidth=2)
    ax.text(mean_val + (max(data)-min(data))*0.02, max(counts)*0.95, "Promedio", color='white', fontsize=10)

    # Quitar ejes para que se vea limpio como la imagen
    ax.axis('off')
    
    # Guardar en memoria
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", facecolor='#000000')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return img_str

# -------------------------
# ESTILOS CSS (REPLICA EXACTA)
# -------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&display=swap');
    
    /* Reset total */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        background-color: #000000;
        color: white;
    }
    
    /* Ocultar elementos nativos de Streamlit */
    header, footer {visibility: hidden;}
    .stApp { margin-top: -50px; }
    
    /* TITULO */
    .main-title {
        font-size: 48px;
        font-weight: 900;
        text-align: center;
        margin-bottom: 10px;
    }
    
    /* BARRA GRADIENTE */
    .gradient-bar {
        height: 12px;
        width: 100%;
        max-width: 1200px;
        margin: 0 auto 40px auto;
        background: linear-gradient(90deg, #8b5cf6 0%, #3b82f6 50%, #2dd4bf 100%);
        border-radius: 10px;
    }
    
    /* MENU DE PESTAÑAS PERSONALIZADO (CSS GRID) */
    .nav-container {
        display: flex;
        justify-content: center;
        gap: 40px;
        margin-bottom: 40px;
        border-bottom: 1px solid #333;
        padding-bottom: 0px;
    }
    .nav-item {
        font-size: 16px;
        color: #cccccc;
        text-align: center;
        padding-bottom: 10px;
        cursor: pointer;
        max-width: 150px;
    }
    .nav-item.active {
        color: white;
        border-bottom: 4px solid #8b5cf6; /* Morado activo */
        font-weight: bold;
    }

    /* CONTENEDOR PRINCIPAL (GRID) */
    .dashboard-grid {
        display: grid;
        grid-template-columns: 350px 20px 1fr; /* Panel izq | Separador | Panel der */
        gap: 0px;
        max-width: 1100px;
        margin: 0 auto;
        align-items: start;
    }

    /* --- PANEL IZQUIERDO (BLANCO) --- */
    .input-panel {
        background-color: white;
        border-radius: 30px;
        padding: 25px;
        color: black;
        box-shadow: 0 0 20px rgba(255,255,255,0.1);
    }
    .input-title {
        color: #7c3aed; /* Morado fuerte */
        font-size: 28px;
        font-weight: 900;
        margin-bottom: 5px;
        border-bottom: 4px solid #7c3aed;
        display: inline-block;
        width: 100px;
    }
    .input-instruction {
        background-color: #5b6af0; /* Azul similar imagen */
        color: white;
        font-size: 12px;
        padding: 8px;
        margin-top: 15px;
        margin-bottom: 15px;
        text-align: center;
    }
    /* Estilizar el textarea nativo inyectado */
    .stTextArea textarea {
        background-color: #000000 !important;
        color: white !important;
        border: none !important;
        height: 100px;
    }
    
    /* --- SEPARADOR VERTICAL --- */
    .vertical-line {
        width: 4px;
        height: 350px;
        background: linear-gradient(to bottom, #7c3aed, #000);
        margin: 0 auto;
    }

    /* --- PANEL DERECHO (TARJETAS) --- */
    .cards-grid {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 15px;
        margin-left: 20px;
    }
    .stat-card {
        background-color: white;
        border-radius: 15px;
        padding: 15px;
        text-align: center;
        color: black;
        min-height: 90px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .card-label {
        font-size: 11px;
        color: #888;
        font-weight: 700;
        margin-bottom: 2px;
    }
    .card-value {
        font-size: 24px;
        font-weight: 900;
        color: #000;
    }
    .card-sub {
        font-size: 10px;
        color: #666;
        font-style: italic;
    }
    
    /* CAJA INTERPRETACION */
    .interp-box {
        background-color: white;
        margin-top: 20px;
        margin-left: 20px;
        padding: 15px;
        color: black;
        font-size: 13px;
        border-left: 6px solid #7c3aed; /* Borde morado izq */
    }

    /* TITULO HISTOGRAMA */
    .hist-header {
        color: #7c3aed;
        font-size: 24px;
        font-weight: 900;
        margin-top: 40px;
        margin-bottom: 10px;
        text-decoration: underline;
        max-width: 1100px;
        margin-left: auto;
        margin-right: auto;
    }

    /* BOTON PERSONALIZADO */
    div.stButton > button {
        background-color: #7c3aed;
        color: white;
        border-radius: 20px;
        border: none;
        width: 100%;
        font-weight: bold;
        padding: 10px 0;
    }
    div.stButton > button:hover {
        background-color: #6d28d9;
        color: white;
    }

</style>
""", unsafe_allow_html=True)

# -------------------------
# INTERFAZ
# -------------------------

# 1. Título y Barra
st.markdown('<div class="main-title">Calculadora de estaditica</div>', unsafe_allow_html=True)
st.markdown('<div class="gradient-bar"></div>', unsafe_allow_html=True)

# 2. Navegación (Visual, simula pestañas)
st.markdown("""
<div class="nav-container">
    <div class="nav-item active">Medidas de<br>tendencia central</div>
    <div class="nav-item">Inferencia<br>estadística</div>
    <div class="nav-item">Comparación de<br>dos poblaciones</div>
    <div class="nav-item">Tamaño de<br>muestra</div>
    <div class="nav-item">Visual<br>LAB</div>
</div>
""", unsafe_allow_html=True)


# -------------------------
# CUERPO PRINCIPAL (Simulación Grid)
# -------------------------

# Contenedor Grid usando columnas de Streamlit para inyectar HTML en bloques
col_izq, col_sep, col_der = st.columns([0.8, 0.1, 1.4])

# --- Variables de Estado para mantener datos ---
if 'res_stats' not in st.session_state:
    st.session_state.res_stats = None
if 'res_hist' not in st.session_state:
    st.session_state.res_hist = None

with col_izq:
    st.markdown('<div class="input-panel">', unsafe_allow_html=True)
    st.markdown('<div class="input-title">Datos:</div>', unsafe_allow_html=True)
    st.markdown('<div class="input-instruction">Usa PUNTO (.) para decimales. Separa números con<br>comas, punto y coma, espacios o saltos de línea.</div>', unsafe_allow_html=True)
    
    # Input nativo (se estiliza con CSS para ser negro)
    input_text = st.text_area("input_hidden", value="3.2, 4.5, 7.8, 9.1, 0.6, 12.3, 14.7", label_visibility="collapsed")
    
    st.markdown('<br>', unsafe_allow_html=True)
    if st.button("Analizar datos"):
        data, err = parse_strict_point(input_text)
        if err:
            st.error(err)
            st.session_state.res_stats = None
        else:
            # Calcular Estadísticas
            stats_dict = {
                'n': len(data),
                'media': np.mean(data),
                'mediana': np.median(data),
                'modas': compute_modes(data),
                'desv': np.std(data, ddof=1) if len(data)>1 else 0,
                'var': np.var(data, ddof=1) if len(data)>1 else 0,
                'ee': (np.std(data, ddof=1)/np.sqrt(len(data))) if len(data)>1 else 0
            }
            st.session_state.res_stats = stats_dict
            st.session_state.res_hist = get_hist_image(data, stats_dict['media'])

    st.markdown('</div>', unsafe_allow_html=True) # Cierre input-panel

with col_sep:
    st.markdown('<div class="vertical-line"></div>', unsafe_allow_html=True)

with col_der:
    res = st.session_state.res_stats
    
    # Valores por defecto (vacíos) o calculados
    v_media = f"{res['media']:.2f}" if res else ""
    v_mediana = f"{res['mediana']:.2f}" if res else ""
    v_desv = f"{res['desv']:.2f}" if res else ""
    v_var = f"{res['var']:.2f}" if res else ""
    v_ee = f"{res['ee']:.4f}" if res else ""
    
    # Moda lógica
    if res:
        if not res['modas']:
            v_moda = "—"
            sub_moda = "No hay moda<br>(todos los valores son únicos)"
        else:
            v_moda = ", ".join([str(m) for m in res['modas'][:2]]) # Mostrar máx 2
            sub_moda = "Multimodal" if len(res['modas']) > 1 else ""
    else:
        v_moda = ""
        sub_moda = ""

    # Interpretación
    if res:
        interp_txt = f"Con una muestra de <b>{res['n']}</b> datos, el centro se ubica en <b>{res['media']:.2f}</b>. La dispersión (s) es de <b>{res['desv']:.2f}</b>.<br>Los datos son bastante simétricos (Media ≈ Mediana)."
    else:
        interp_txt = "Esperando datos..."

    # HTML Grid de tarjetas
    html_cards = f"""
    <div class="cards-grid">
        <!-- Fila 1 -->
        <div class="stat-card">
            <div class="card-label">Promedio (media)</div>
            <div class="card-value">{v_media}</div>
        </div>
        <div class="stat-card">
            <div class="card-label">Mediana</div>
            <div class="card-value">{v_mediana}</div>
        </div>
        <div class="stat-card">
            <div class="card-label">Moda</div>
            <div class="card-value">{v_moda}</div>
            <div class="card-sub">{sub_moda}</div>
        </div>
        
        <!-- Fila 2 -->
        <div class="stat-card">
            <div class="card-label">Desviación estándar (s)</div>
            <div class="card-value">{v_desv}</div>
            <div class="card-sub">Muestral</div>
        </div>
        <div class="stat-card">
            <div class="card-label">Varianza (s^2)</div>
            <div class="card-value">{v_var}</div>
        </div>
        <div class="stat-card">
            <div class="card-label">Error estándar (EE)</div>
            <div class="card-value">{v_ee}</div>
        </div>
    </div>
    
    <div class="interp-box">
        <strong>Interpretación:</strong><br>
        {interp_txt}
    </div>
    """
    st.markdown(html_cards, unsafe_allow_html=True)

# -------------------------
# HISTOGRAMA
# -------------------------
st.markdown('<div class="hist-header">Histograma de Frecuencias:</div>', unsafe_allow_html=True)

if st.session_state.res_hist:
    # Centrar la imagen
    col_izq_h, col_centro_h, col_der_h = st.columns([1, 8, 1])
    with col_centro_h:
        st.markdown(f'<img src="data:image/png;base64,{st.session_state.res_hist}" style="width:100%; border-radius:10px;">', unsafe_allow_html=True)
