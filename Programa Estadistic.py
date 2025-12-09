import streamlit as st
import numpy as np

# 1. CONFIGURACIÓN DE PÁGINA
st.set_page_config(
    page_title="StatCalc Pro",
    page_icon="📊",
    layout="centered"
)

# 2. INYECCIÓN DE ESTILOS CSS (Aquí ocurre la magia visual)
st.markdown("""
    <style>
        /* Importar fuente moderna */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        /* Aplicar fuente a todo */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #1f2937;
        }

        /* Fondo general suave */
        .stApp {
            background-color: #f3f4f6;
        }

        /* Estilo de los Títulos */
        h1, h2, h3 {
            color: #111827;
            font-weight: 700;
        }

        /* Personalización de las Pestañas (Tabs) */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: white;
            padding: 10px 20px;
            border-radius: 15px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        .stTabs [data-baseweb="tab"] {
            height: 50px;
            border-radius: 10px;
            font-weight: 600;
            color: #6b7280;
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #eef2ff;
            color: #4f46e5;
        }

        /* Estilo del Botón Principal */
        div.stButton > button {
            background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
            color: white;
            font-size: 16px;
            font-weight: 600;
            padding: 0.75rem 1.5rem;
            border-radius: 12px;
            border: none;
            box-shadow: 0 4px 6px rgba(79, 70, 229, 0.3);
            transition: all 0.3s ease;
            width: 100%;
        }

        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(79, 70, 229, 0.4);
            background: linear-gradient(135deg, #4338ca 0%, #3730a3 100%);
            color: white;
        }

        /* Estilo del Área de Texto */
        .stTextArea textarea {
            border-radius: 12px;
            border: 1px solid #e5e7eb;
            background-color: white;
            padding: 15px;
            font-size: 16px;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
        }

        /* Clases personalizadas para las tarjetas de estadísticas (HTML Injection) */
        .stat-card {
            background-color: white;
            padding: 20px;
            border-radius: 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            text-align: center;
            margin-bottom: 15px;
            transition: transform 0.2s;
            border: 1px solid #f3f4f6;
        }
        .stat-card:hover {
            transform: translateY(-3px);
            border-color: #c7d2fe;
        }
        .stat-title {
            font-size: 0.85rem;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 5px;
            font-weight: 600;
        }
        .stat-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #1f2937;
        }
        .stat-desc {
            font-size: 0.8rem;
            color: #9ca3af;
        }

        /* Contenedor de "Acerca de" */
        .info-box {
            background-color: white;
            padding: 25px;
            border-radius: 16px;
            border-left: 5px solid #4f46e5;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }
    </style>
""", unsafe_allow_html=True)

# 3. INTERFAZ PRINCIPAL

st.title("📊 Calculadora Estadística")
st.markdown("---")

# Pestañas
tab1, tab2, tab3 = st.tabs(["📝 Calculadora", "📈 Resultados", "ℹ️ Acerca de"])

# --------------------
# PESTAÑA 1: CALCULADORA
# --------------------
with tab1:
    st.markdown("### Ingreso de Datos")
    
    st.markdown("""
    <div style="background-color: white; padding: 15px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #e5e7eb;">
        <span style="color:#6b7280; font-size: 0.9rem;">
        Introduce tus números separados por comas. <br>
        <b>Ejemplo:</b> <code>10.5, 20.2, 15, 30.1, 25</code>
        </span>
    </div>
    """, unsafe_allow_html=True)

    data_input = st.text_area("Datos Numéricos:", height=150, placeholder="Escribe aquí tus números...")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Procesar Datos"):
            try:
                # Limpieza y conversión de datos
                data = [float(x.strip()) for x in data_input.split(",") if x.strip()]
                
                if len(data) > 0:
                    st.session_state["datos"] = data
                    st.balloons()
                    st.success(f"✅ ¡Éxito! Se cargaron {len(data)} valores.")
                else:
                    st.warning("⚠️ La lista está vacía.")
            except ValueError:
                st.error("❌ Error: Asegúrate de usar solo números y comas.")

# --------------------
# PESTAÑA 2: ESTADÍSTICOS
# --------------------
with tab2:
    if "datos" in st.session_state:
        data = st.session_state["datos"]
        
        # Cálculos
        media = np.mean(data)
        mediana = np.median(data)
        desviacion = np.std(data, ddof=1)
        varianza = np.var(data, ddof=1)
        minimo = np.min(data)
        maximo = np.max(data)
        rango = maximo - minimo

        st.markdown("### 📊 Panel de Control Estadístico")
        st.write("") # Espacio

        # Función auxiliar para dibujar tarjetas HTML
        def card(title, value, desc=""):
            return f"""
            <div class="stat-card">
                <div class="stat-title">{title}</div>
                <div class="stat-value">{value}</div>
                <div class="stat-desc">{desc}</div>
            </div>
            """

        # Fila 1: Tendencia Central
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(card("Media Aritmética", f"{media:.2f}", "Promedio"), unsafe_allow_html=True)
        with c2:
            st.markdown(card("Mediana", f"{mediana:.2f}", "Valor central"), unsafe_allow_html=True)

        # Fila 2: Dispersión
        c3, c4 = st.columns(2)
        with c3:
            st.markdown(card("Desviación Estándar", f"{desviacion:.3f}", "Dispersión"), unsafe_allow_html=True)
        with c4:
            st.markdown(card("Varianza", f"{varianza:.3f}", "Desv²"), unsafe_allow_html=True)

        # Fila 3: Rango
        st.markdown("#### Detalles del Rango")
        c5, c6, c7 = st.columns(3)
        with c5:
            st.markdown(card("Mínimo", f"{minimo:.2f}"), unsafe_allow_html=True)
        with c6:
            st.markdown(card("Máximo", f"{maximo:.2f}"), unsafe_allow_html=True)
        with c7:
            st.markdown(card("Rango Total", f"{rango:.2f}"), unsafe_allow_html=True)

    else:
        # Mensaje de estado vacío elegante
        st.markdown("""
        <div style="text-align: center; padding: 50px; color: #6b7280;">
            <h3>🔍 No hay datos cargados</h3>
            <p>Por favor, ve a la pestaña <b>Calculadora</b> e ingresa tus datos para ver el análisis.</p>
        </div>
        """, unsafe_allow_html=True)

# --------------------
# PESTAÑA 3: ACERCA DE
# --------------------
with tab3:
    st.markdown("""
    <div class="info-box">
        <h3>ℹ️ Sobre StatCalc Pro</h3>
        <p style="color: #4b5563; line-height: 1.6;">
            Esta aplicación ha sido diseñada para facilitar el análisis estadístico descriptivo 
            de manera rápida y visual. Es ideal para estudiantes y profesionales que necesitan 
            verificar cálculos básicos al instante.
        </p>
        <hr style="border-top: 1px solid #e5e7eb; margin: 20px 0;">
        <h4 style="margin-bottom: 10px;">Funcionalidades:</h4>
        <ul style="color: #4b5563; margin-left: 20px;">
            <li>Cálculo de medidas de tendencia central (Media, Mediana).</li>
            <li>Análisis de dispersión (Desviación Estándar, Varianza).</li>
            <li>Identificación de extremos (Mínimo, Máximo, Rango).</li>
        </ul>
        <br>
        <small style="color: #9ca3af;">Versión 2.0 | Diseño Minimalista</small>
    </div>
    """, unsafe_allow_html=True)
