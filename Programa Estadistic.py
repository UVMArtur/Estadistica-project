import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN E INICIALIZACIÓN
# -----------------------------------------------------------------------------
st.set_page_config(page_title="StatSuite Reactiva", page_icon="⚡", layout="wide")

# Inicializar estados para la lógica reactiva si no existen
def init_state(keys):
    for k in keys:
        if k not in st.session_state:
            st.session_state[k] = 0.0

# Claves de los inputs para controlar su estado
state_keys = [
    'inf_media', 'inf_prop', 'inf_sigma', 'inf_s', # Inferencia
    'size_sigma', 'size_prop', # Tamaño Muestra
    'comp_s1', 'comp_s2' # Comparación
]
init_state(state_keys)

# -----------------------------------------------------------------------------
# 2. ESTILOS CSS (DARK NEON + UI REACTIVA)
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
            background-color: #050505;
            color: #e5e7eb;
        }
        
        header, footer {visibility: hidden;}
        .stApp { background-color: #050505; }

        /* PESTAÑAS */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: #0f0f0f;
            padding: 10px;
            border-radius: 12px;
            border: 1px solid #333;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 6px;
            color: #666;
            font-weight: 600;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #222;
            color: white;
            border-bottom: 2px solid white;
        }

        /* INPUTS - Estado Normal */
        .stNumberInput input {
            background-color: #111 !important;
            color: white !important;
            border: 1px solid #333 !important;
            border-radius: 8px;
        }
        
        /* INPUTS - Deshabilitados (Visualmente grisáceos) */
        div[data-testid="stNumberInput"] div[aria-disabled="true"] input {
            background-color: #1a1a1a !important;
            color: #444 !important; /* Texto oscuro para indicar inactivo */
            border-color: #222 !important;
            cursor: not-allowed;
        }

        /* TARJETAS DE RESULTADO */
        .result-card {
            background-color: #151515;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            border: 1px solid #333;
            margin-bottom: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }
        .card-label { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; opacity: 0.8; }
        .card-value { font-size: 1.8rem; font-weight: 700; color: #fff; }

        /* COLORES DE ACENTO */
        .neon-blue { border-top: 3px solid #3b82f6; }
        .neon-green { border-top: 3px solid #22c55e; }
        .neon-red { border-top: 3px solid #ef4444; }

        /* TEXTO DE INTERPRETACIÓN */
        .smart-text {
            background: rgba(255,255,255,0.03);
            border-left: 3px solid #666;
            padding: 15px;
            border-radius: 0 8px 8px 0;
            margin-top: 15px;
            color: #ccc;
        }

        /* BOTONES */
        div.stButton > button {
            width: 100%;
            border-radius: 8px;
            font-weight: 600;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ StatSuite Reactiva")
st.markdown("El sistema detecta automáticamente qué fórmula usar según los campos que llenes.")

# Función auxiliar para tarjetas
def card(label, value, style="neon-blue"):
    return f"""
    <div class="result-card {style}">
        <div class="card-label">{label}</div>
        <div class="card-value">{value}</div>
    </div>
    """

# -----------------------------------------------------------------------------
# ESTRUCTURA DE PESTAÑAS
# -----------------------------------------------------------------------------
tab_inf, tab_size, tab_comp = st.tabs([
    "🔍 Inferencia Inteligente (1 Pob)", 
    "📏 Tamaño de Muestra", 
    "⚖️ Comparación (2 Pob)"
])

# =============================================================================
# PESTAÑA 1: INFERENCIA INTELIGENTE (Lógica Z/T y Media/Prop Automática)
# =============================================================================
with tab_inf:
    st.markdown("<h3 style='color:#3b82f6'>Inferencia de Una Población</h3>", unsafe_allow_html=True)
    
    # --- CAMPOS COMPARTIDOS ---
    c_common1, c_common2 = st.columns(2)
    n = c_common1.number_input("Tamaño de Muestra (n)", min_value=0, step=1, key="inf_n")
    conf = c_common2.selectbox("Nivel de Confianza", [0.90, 0.95, 0.99], index=1, key="inf_conf")
    
    st.markdown("---")
    
    # --- LÓGICA DE EXCLUSIÓN MUTUA (MEDIA VS PROPORCIÓN) ---
    # Si hay valor en Media, bloqueamos Proporción. Y viceversa.
    has_media_data = st.session_state.inf_media != 0.0
    has_prop_data = st.session_state.inf_prop != 0.0

    col_A, col_B = st.columns(2)
    
    with col_A:
        st.markdown("##### 📊 Datos Cuantitativos (Media)")
        # Campo A: Media Muestral
        media = st.number_input(
            "Media Muestral (x̄)", 
            key="inf_media", 
            disabled=has_prop_data, # SE BLOQUEA SI HAY PROPORCIÓN
            help="Si escribes aquí, se desactivará la sección de Proporción."
        )
        
        # --- LÓGICA Z vs T ---
        # Si hay Sigma, bloqueamos S. Si hay S, bloqueamos Sigma.
        has_sigma = st.session_state.inf_sigma != 0.0
        has_s = st.session_state.inf_s != 0.0
        
        # Campo C: Sigma (Poblacional)
        sigma = st.number_input(
            "Desviación Estándar Poblacional (σ)", 
            key="inf_sigma",
            disabled=has_s or has_prop_data, # Se bloquea si hay 's' o si estamos en modo proporción
            help="Llenar esto activa la fórmula Z."
        )
        
        # Campo D: S (Muestral)
        s = st.number_input(
            "Desviación Estándar Muestral (s)", 
            key="inf_s",
            disabled=has_sigma or has_prop_data, # Se bloquea si hay 'sigma' o si estamos en modo proporción
            help="Llenar esto activa la fórmula T-Student."
        )

    with col_B:
        st.markdown("##### 🥧 Datos Cualitativos (Proporción)")
        # Campo B: Proporción
        prop = st.number_input(
            "Proporción de Éxito (p) [0.0 - 1.0]", 
            min_value=0.0, max_value=1.0, step=0.01,
            key="inf_prop",
            disabled=has_media_data, # SE BLOQUEA SI HAY MEDIA
            help="Si escribes aquí, se desactivará la sección de Media."
        )
        
        st.write("")
        st.write("")
        # Botón de Reset para limpiar campos trabados
        if st.button("🔄 Limpiar Todos los Campos", key="reset_inf"):
            for k in ['inf_media', 'inf_prop', 'inf_sigma', 'inf_s']:
                st.session_state[k] = 0.0
            st.rerun()

    # --- RESULTADOS AUTOMÁTICOS ---
    st.markdown("### Resultados Calculados")
    
    if n > 0:
        # CASO 1: INTERVALO DE PROPORCIÓN
        if prop > 0:
            q = 1 - prop
            se = np.sqrt((prop * q) / n)
            z_score = stats.norm.ppf((1 + conf) / 2)
            margen = z_score * se
            
            c1, c2 = st.columns(2)
            c1.markdown(card("Límite Inferior", f"{(prop - margen)*100:.2f}%"), unsafe_allow_html=True)
            c2.markdown(card("Límite Superior", f"{(prop + margen)*100:.2f}%"), unsafe_allow_html=True)
            st.success(f"✅ Detectado: Intervalo para Proporción (Usando Z). Margen de error: {margen:.4f}")

        # CASO 2: INTERVALO DE MEDIA (Z o T)
        elif media != 0:
            # Sub-caso: Z (Con Sigma)
            if sigma > 0:
                se = sigma / np.sqrt(n)
                z_score = stats.norm.ppf((1 + conf) / 2)
                margen = z_score * se
                tipo = "Normal (Z) - Sigma Conocida"
            # Sub-caso: T (Con S)
            elif s > 0:
                se = s / np.sqrt(n)
                t_score = stats.t.ppf((1 + conf) / 2, df=n-1)
                margen = t_score * se
                tipo = "T-Student - Sigma Desconocida"
            else:
                st.warning("⚠️ Falta la Desviación Estándar (Poblacional o Muestral) para calcular.")
                margen = 0
                tipo = "Pendiente"

            if margen > 0:
                c1, c2 = st.columns(2)
                c1.markdown(card("Límite Inferior", f"{media - margen:.4f}"), unsafe_allow_html=True)
                c2.markdown(card("Límite Superior", f"{media + margen:.4f}"), unsafe_allow_html=True)
                st.info(f"ℹ️ Método Detectado: {tipo}")
        else:
            st.info("👈 Ingresa datos en el panel izquierdo (Media o Proporción) para comenzar.")
    else:
        st.warning("El tamaño de muestra (n) debe ser mayor a 0.")

# =============================================================================
# PESTAÑA 2: TAMAÑO DE MUESTRA (Exclusión Sigma vs Proporción)
# =============================================================================
with tab_size:
    st.markdown("<h3 style='color:#22c55e'>Calculadora de 'n'</h3>", unsafe_allow_html=True)

    # CAMPOS COMPARTIDOS
    c1, c2 = st.columns(2)
    error = c1.number_input("Margen de Error Deseado (E)", format="%.4f", value=0.05)
    conf_z = c2.selectbox("Nivel de Confianza", [0.90, 0.95, 0.99], index=1, key="size_conf")

    st.markdown("---")
    
    # LÓGICA DE EXCLUSIÓN
    has_sigma_size = st.session_state.size_sigma != 0.0
    has_prop_size = st.session_state.size_prop != 0.0

    col_media, col_prop = st.columns(2)
    
    with col_media:
        st.markdown("##### Opción A: Para estimar un Promedio")
        sigma_est = st.number_input(
            "Desviación Estándar Estimada (σ)", 
            key="size_sigma",
            disabled=has_prop_size,
            help="Al llenar esto, calcularemos n para medias."
        )
    
    with col_prop:
        st.markdown("##### Opción B: Para estimar una Proporción")
        prop_est = st.number_input(
            "Proporción Estimada (p)", 
            min_value=0.0, max_value=1.0, step=0.01,
            key="size_prop",
            disabled=has_sigma_size,
            help="Al llenar esto, calcularemos n para proporciones."
        )
        
        if st.button("🔄 Limpiar", key="reset_size"):
            st.session_state.size_sigma = 0.0
            st.session_state.size_prop = 0.0
            st.rerun()

    # CÁLCULO AUTOMÁTICO
    st.markdown("### Resultado")
    z_val = stats.norm.ppf((1 + conf_z) / 2)
    
    if error > 0:
        n_final = 0
        formula = ""
        
        # Caso Media
        if sigma_est > 0:
            n_final = (z_val**2 * sigma_est**2) / error**2
            formula = "Media: (Z·σ / E)²"
            
        # Caso Proporción
        elif prop_est > 0:
            n_final = (z_val**2 * prop_est * (1 - prop_est)) / error**2
            formula = "Proporción: (Z² · p · q) / E²"
            
        if n_final > 0:
            st.markdown(card("Muestra Necesaria", f"{math.ceil(n_final)}", "neon-green"), unsafe_allow_html=True)
            st.caption(f"Fórmula aplicada: {formula}")
        else:
            st.info("Ingresa la Desviación o la Proporción estimada.")
    else:
        st.warning("El margen de error debe ser mayor a 0.")

# =============================================================================
# PESTAÑA 3: COMPARACIÓN (2 POBLACIONES)
# =============================================================================
with tab_comp:
    st.markdown("<h3 style='color:#ef4444'>Comparación de Grupos</h3>", unsafe_allow_html=True)
    st.markdown("El sistema detecta si comparas Medias o Proporciones basándose en si llenas o no las desviaciones estándar.")

    c_g1, c_g2 = st.columns(2)
    
    with c_g1:
        st.markdown("#### Grupo 1")
        n1 = st.number_input("Tamaño (n1)", 0, key="c_n1")
        val1 = st.number_input("Media o Éxitos (x1)", 0.0, key="c_v1")
        # El campo mágico
        s1 = st.number_input("Desviación Estándar (s1) [Dejar 0 si es Proporción]", 0.0, key="comp_s1")

    with c_g2:
        st.markdown("#### Grupo 2")
        n2 = st.number_input("Tamaño (n2)", 0, key="c_n2")
        val2 = st.number_input("Media o Éxitos (x2)", 0.0, key="c_v2")
        # El campo mágico
        s2 = st.number_input("Desviación Estándar (s2) [Dejar 0 si es Proporción]", 0.0, key="comp_s2")

    alpha = st.slider("Alpha", 0.01, 0.10, 0.05)
    
    st.markdown("---")

    # LÓGICA DE DETECCIÓN
    if n1 > 0 and n2 > 0:
        # Si las desviaciones son 0, asumimos que el usuario quiere PROPORCIONES
        if s1 == 0 and s2 == 0:
            st.markdown("⚙️ **Modo Detectado: Diferencia de Proporciones** (Porque no hay desviaciones)")
            # Asumimos que val1 y val2 son conteos de éxitos
            p1 = val1 / n1
            p2 = val2 / n2
            
            # Validación simple
            if p1 > 1 or p2 > 1:
                st.error("Error: Si es proporción, los éxitos (x) no pueden ser mayores que la muestra (n).")
            else:
                pp = (val1 + val2) / (n1 + n2)
                se = np.sqrt(pp * (1 - pp) * (1/n1 + 1/n2))
                if se > 0:
                    z = (p1 - p2) / se
                    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
                    
                    lbl = "Diferente" if p_val < alpha else "Iguales"
                    color = "neon-red" if p_val < alpha else "neon-green"
                    
                    c1, c2 = st.columns(2)
                    c1.markdown(card("Valor P", f"{p_val:.4f}", color), unsafe_allow_html=True)
                    c2.markdown(card("Conclusión", lbl, color), unsafe_allow_html=True)
        
        # Si hay desviaciones, asumimos MEDIAS
        else:
            st.markdown("⚙️ **Modo Detectado: Diferencia de Medias** (Se detectaron desviaciones)")
            se = np.sqrt((s1**2/n1) + (s2**2/n2))
            t_stat = (val1 - val2) / se
            df = n1 + n2 - 2
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            
            lbl = "Diferente" if p_val < alpha else "Iguales"
            color = "neon-red" if p_val < alpha else "neon-green"
            
            c1, c2 = st.columns(2)
            c1.markdown(card("Valor P", f"{p_val:.4f}", color), unsafe_allow_html=True)
            c2.markdown(card("Conclusión", lbl, color), unsafe_allow_html=True)
