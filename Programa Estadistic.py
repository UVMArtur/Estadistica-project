import streamlit as st
import numpy as np
 
st.title("Calculadora de Media de un Conjunto de Datos")
 
st.write("Ingresa una lista de números separados por comas. Ejemplo:")
st.code("10, 20, 15, 30, 25")
 
# Entrada de datos
data_input = st.text_area("Datos:")
 
if st.button("Calcular"):
    try:
        # Convertir texto a lista numérica
        data = [float(x.strip()) for x in data_input.split(",")]
 
        media = np.mean(data)
        mediana = np.median(data)
        desviacion = np.std(data, ddof=1)
        varianza = np.var(data, ddof=1)
 
        st.success("Resultados:")
        st.write(f"**Media:** {media:.4f}")
        st.write(f"**Mediana:** {mediana:.4f}")
        st.write(f"**Desviación estándar (muestral):** {desviacion:.4f}")
        st.write(f"**Varianza (muestral):** {varianza:.4f}")
 
    except:
        st.error("Revisa que los datos estén escritos correctamente.")
    - Organización en pestañas (entrada de datos, resultados e información)

    """)

 
