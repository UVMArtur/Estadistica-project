# Versión inicial con manejo amigable de dependencias faltantes.
# Si scipy no está instalado, la app mostrará instrucciones en lugar de romperse.

try:
    import streamlit as st
    import numpy as np
    import pandas as pd
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception as e:
    # ImportError u otro problema con dependencias
    try:
        import streamlit as st  # intentamos al menos importar streamlit para mostrar el error
    except Exception:
        # Si ni siquiera streamlit está disponible, levantar el error original para debug local
        raise e

    st.set_page_config(page_title="Dependencia faltante", layout="centered")
    st.title("Dependencia faltante: scipy (u otra librería)")
    st.error("La aplicación necesita paquetes que no están instalados en este entorno.")
    st.markdown("Para solucionar esto puedes:")
    st.markdown("- Si trabajas localmente: crear/activar un entorno virtual e instalar dependencias:\n\n```bash\npython -m venv .venv\nsource .venv/bin/activate  # (Linux/Mac)\n.venv\\Scripts\\activate     # (Windows)\npip install -r requirements.txt\nstreamlit run \"Programa Estadistic.py\"\n```")
    st.markdown("- Si usas Streamlit Cloud: añade este archivo `requirements.txt` en la raíz del repositorio, haz commit y push. Streamlit instalará las dependencias automáticamente en el despliegue.")
    st.markdown("Contenido sugerido para requirements.txt:")
    st.code("streamlit\nnumpy\npandas\nscipy\nmatplotlib\nseaborn")
    st.markdown("Si la instalación falla localmente al compilar scipy, intenta actualizar pip primero:\n\n```bash\npip install --upgrade pip wheel setuptools\npip install scipy\n```")
    st.stop()

# Si llegamos aquí, las dependencias están disponibles y puedes importar el resto del código
import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Calculadora Estadística", layout="wide")
st.title("Calculadora Estadística (dependencias cargadas)")

st.write("Si ves este mensaje, scipy y las demás dependencias están instaladas correctamente.")
# Aquí pondrías el resto de tu código de la aplicación original.
# -----------------------
st.markdown("---")
st.write("App creada para facilitar cálculos estadísticos comunes y su interpretación. Modifique los ejemplos, pruebe con sus datos y revise las suposiciones antes de usar resultados en decisiones.")
