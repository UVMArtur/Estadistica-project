import streamlit as st
import numpy as np

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="StatCalc Dark",
    page_icon="💜",
    layout="centered"
)

# -----------------------------------------------------------------------------
# 2. ESTILOS CSS (TEMA NEGRO Y MORADO)
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
        /* Importar fuente Inter */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

        /* RESET GENERAL */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #e5e7eb; /* Texto blanco suave */
        }

        /* FONDO NEGRO */
        .stApp {
            background-color: #050505;
            background-image: radial-gradient(circle at 50% 0%, #2e1065 0%, #050505 50%); /* Luz morada arriba */
        }

        /* TITULOS */
        h1, h2, h3 {
            color: #ffffff !important;
            font-weight: 800;
        }

        /* PESTAÑAS (TABS) PERSONALIZADAS */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: #000000;
            padding: 10px;
            border-radius: 20px;
            border: 1px solid #333;
        }

        .stTabs [data-baseweb="tab"] {
            height: 45px;
            border-radius: 12px;
            font-weight: 600;
            color: #9ca3af; /* Gris */
            border: none;
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #7c3aed; /* Morado Intenso */
            color: white;
        }

        /* INPUTS (TEXT AREA) */
        .stTextArea textarea {
            background-color: #111111 !important;
            color: white !important;
            border: 1px solid #333 !important;
            border-radius: 15px;
            padding: 15px;
        }
        .stTextArea textarea:focus {
            border-color: #7c3aed !important;
            box-shadow: 0 0 10px rgba(124, 58, 237, 0.5);
        }

        /* BOTÓN PRINCIPAL (GRADIENTE MORADO) */
        div.stButton > button {
            background: linear-gradient(90deg, #7c3aed 0%, #6d28d9 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 15px;
            font-weight: 600;
            transition: all 0.3s ease;
            width: 100%;
            box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3);
        }
        div.stButton > button:hover {
            transform: scale(1.02);
            box-shadow: 0 6px 20px rgba(124, 58, 237, 0.5);
            color: white;
        }

        /* TARJETAS DE ESTADÍSTICAS (CLASE PERSONALIZADA) */
        .stat-card {
            background-color: #111111;
            border: 1px solid #222;
            padding: 20px;
            border-radius: 18px;
            text-align: center;
            margin-bottom: 15px;
            transition: transform 0.2s;
        }
        .stat-card:hover {
            border-color: #7c3aed; /* Borde morado al pasar mouse */
            transform: translateY(-5px);
        }
        .stat-title {
            color: #9ca3af;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }
        .stat-value {
            color: #ffffff;
            font-size: 2rem;
            font-weight: 700;
            text-shadow: 0 2px 10px rgba(124, 58, 237, 0.4);
        }
        .stat-desc {
            color: #6b7280;
            font-size: 0.8rem;
        }

        /* CAJA DE INFORMACIÓN */
        .info-box {
            background-color: #111;
            border-left: 4px solid #7c3aed;
            padding: 20px;
            border-radius: 0 15px 15px 0;
            color: #d1d5db;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. LÓGICA DE LA APLICACIÓN
# -----------------------------------------------------------------------------

st.title("💜 Calculadora Estadística")
st.markdown("---")

# Creación de Pestañas
tab1, tab2, tab3 = st.tabs(["Calculadora", "Estadísticos", "Acerca de"])

# --------------------
# PESTAÑA 1: CALCULADORA
# --------------------
with tab1:
    st.header("Ingreso de Datos")
    
    st.markdown("""
    <div style="margin-bottom: 15px; color: #aaa;">
        Pega tus datos numéricos separados por comas.<br>
        <span style="color: #7c3aed;">Ejemplo:</span> <code>10, 20.5, 15, 30, 25</code>
    </div>
    """, unsafe_allow_html=True)

    data_input = st.text_area("Datos:", height=120, placeholder="Escribe aquí...")

    if st.button("✨ Procesar Datos"):
        try:
            # Convertir texto a lista numérica
            if data_input.strip():
                data = [float(x.strip()) for x in data_input.split(",") if x.strip()]
                
                st.session_state["datos"] = data
                st.success(f"¡Correcto! Se cargaron {len(data)} datos.")
            else:
                st.warning("El campo está vacío.")

        except ValueError:
            st.error("Error: Asegúrate de escribir solo números separados por comas.")

# --------------------
# PESTAÑA 2: ESTADÍSTICOS
# --------------------
with tab2:
    st.header("Resultados del Análisis")

    if "datos" in st.session_state:
        data = st.session_state["datos"]
        
        # Cálculos Matemáticos
        media = np.mean(data)
        mediana = np.median(data)
        desviacion = np.std(data, ddof=1)
        varianza = np.var(data, ddof=1)
        minimo = np.min(data)
        maximo = np.max(data)
        rango = maximo - minimo

        # FUNCIÓN PARA GENERAR TARJETAS HTML
        def card(title, value, desc=""):
            return f"""
            <div class="stat-card">
                <div class="stat-title">{title}</div>
                <div class="stat-value">{value}</div>
                <div class="stat-desc">{desc}</div>
            </div>
            """

        # VISUALIZACIÓN EN GRID (Rejilla)
        
        # Fila 1
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(card("Media", f"{media:.2f}", "Promedio"), unsafe_allow_html=True)
        with c2:
            st.markdown(card("Mediana", f"{mediana:.2f}", "Valor Central"), unsafe_allow_html=True)

        # Fila 2
        c3, c4 = st.columns(2)
        with c3:
            st.markdown(card("Desviación Std", f"{desviacion:.3f}", "Dispersión"), unsafe_allow_html=True)
        with c4:
            st.markdown(card("Varianza", f"{varianza:.3f}", "S²"), unsafe_allow_html=True)

        # Fila 3
        st.markdown("##### Extremos")
        c5, c6, c7 = st.columns(3)
        with c5:
            st.markdown(card("Mínimo", f"{minimo:.2f}"), unsafe_allow_html=True)
        with c6:
            st.markdown(card("Máximo", f"{maximo:.2f}"), unsafe_allow_html=True)
        with c7:
            st.markdown(card("Rango", f"{rango:.2f}"), unsafe_allow_html=True)

    else:
        st.info("👈 Por favor, carga los datos en la pestaña 'Calculadora' primero.")

# --------------------
# PESTAÑA 3: ACERCA DE
# --------------------
with tab3:
    st.header("Acerca de")
    
    st.markdown("""
    <div class="info-box">
        <h4 style="color:white;">StatCalc Dark Edition</h4>
        <p>Herramienta diseñada para análisis estadístico rápido con una interfaz optimizada para modo oscuro.</p>
        <br>
        <strong>Funciones:</strong>
        <ul>
            <li>Cálculo de tendencias centrales.</li>
            <li>Análisis de dispersión.</li>
            <li>Interfaz visual con tarjetas.</li>
        </ul>
        <br>
        <small style="color: #666;">Diseñado con Python & Streamlit</small>
    </div>
    """, unsafe_allow_html=True)
