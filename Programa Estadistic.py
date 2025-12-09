import streamlit as st

import numpy as np
 
st.title("Calculadora Estadística con Pestañas")
 
# Creamos las pestañas

tab1, tab2, tab3 = st.tabs(["Calculadora", "Estadísticos", "Acerca de"])
 
# --------------------

# PESTAÑA 1: CALCULADORA

# --------------------

with tab1:

    st.header("Calculadora de Datos")
 
    st.write("Ingresa una lista de números separados por comas. Ejemplo:")

    st.code("10, 20, 15, 30, 25")
 
    data_input = st.text_area("Datos:")
 
    if st.button("Cargar datos"):

        try:

            # Convertir texto a lista numérica

            data = [float(x.strip()) for x in data_input.split(",")]
 
            st.success("Datos cargados correctamente.")

            st.write("Tamaño de la muestra:", len(data))

            st.write("Primeros valores:", data[:5])
 
            # Guardamos los datos para usarlos en otras pestañas

            st.session_state["datos"] = data
 
        except:

            st.error("Error: revisa que los datos estén escritos correctamente.")
 
# --------------------

# PESTAÑA 2: ESTADÍSTICOS

# --------------------

with tab2:

    st.header("Resultados Estadísticos")
 
    if "datos" in st.session_state:

        data = st.session_state["datos"]
 
        media = np.mean(data)

        mediana = np.median(data)

        desviacion = np.std(data, ddof=1)

        varianza = np.var(data, ddof=1)

        minimo = np.min(data)

        maximo = np.max(data)

        rango = maximo - minimo
 
        st.write(f"**Media:** {media:.4f}")

        st.write(f"**Mediana:** {mediana:.4f}")

        st.write(f"**Desviación estándar (muestral):** {desviacion:.4f}")

        st.write(f"**Varianza (muestral):** {varianza:.4f}")

        st.write(f"**Mínimo:** {minimo:.4f}")

        st.write(f"**Máximo:** {maximo:.4f}")

        st.write(f"**Rango:** {rango:.4f}")
 
    else:

        st.warning("Primero ingresa los datos en la pestaña 'Calculadora'.")
 
# --------------------

# PESTAÑA 3: ACERCA DE

# --------------------

with tab3:

    st.header("Acerca de la App")

    st.write("""

    Esta app fue creada para practicar el análisis estadístico básico.
 
    **Funciones:**

    - Ingreso de datos numéricos

    - Cálculo de media, mediana, desviación estándar y varianza

    - Cálculo de mínimo, máximo y rango

    - Organización en pestañas (entrada de datos, resultados e información)

    """)

 
