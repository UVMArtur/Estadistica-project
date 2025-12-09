import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import io
import math

st.set_page_config(layout="wide", page_title="Calculadora Estadística")

st.title("Calculadora Estadística (pestañas) — UVMArtur")

# Barra superior para cargar datos (común)
st.markdown("## Entrada de datos (opcional)")
col1, col2 = st.columns([3, 1])
with col1:
    st.write("Puedes ingresar datos numéricos separados por comas o subir un CSV con una columna numérica.")
    data_input = st.text_area("Datos (ej: 10, 20, 15, 30, 25)", height=80)
    upload = st.file_uploader("O subir CSV (una columna numérica)", type=["csv", "txt"])
with col2:
    load_btn = st.button("Cargar datos")

# Helper para cargar datos desde entrada o CSV
def parse_text_to_list(text):
    try:
        arr = [float(x.strip()) for x in text.split(",") if x.strip() != ""]
        return np.array(arr)
    except Exception:
        return None

if load_btn:
    ds = None
    if upload is not None:
        try:
            df = pd.read_csv(upload, header=0)
            # If multiple columns, let user pick
            if df.shape[1] > 1:
                st.session_state["_df_uploaded"] = df
                col = st.selectbox("Selecciona columna para usar", options=df.columns, key="col_pick")
                arr = pd.to_numeric(df[col], errors="coerce").dropna().values
                ds = np.array(arr, dtype=float)
            else:
                colname = df.columns[0]
                arr = pd.to_numeric(df[colname], errors="coerce").dropna().values
                ds = np.array(arr, dtype=float)
        except Exception as e:
            st.error(f"Error leyendo CSV: {e}")
    if data_input and (not upload):
        arr = parse_text_to_list(data_input)
        if arr is None:
            st.error("Error: revisa que los datos estén bien escritos (números separados por comas).")
        else:
            ds = arr

    if ds is not None:
        st.session_state["datos"] = ds.tolist()
        st.success(f"Datos cargados. Tamaño: {len(ds)}")
    else:
        st.warning("No se cargaron datos. Introduce texto o sube un CSV válido.")

# Pestañas principales
tab1, tab2, tab3, tab4 = st.tabs(["Tendencia central", "Inferencia estadística", "Dos poblaciones y pruebas", "TLC / Gráficos"])

# -------------------------
# PESTAÑA 1: Tendencia central
# -------------------------
with tab1:
    st.header("Medidas de tendencia central y resumen")
    if "datos" not in st.session_state:
        st.info("Primero carga los datos en la sección superior (texto o CSV).")
    else:
        data = np.array(st.session_state["datos"], dtype=float)
        n = data.size
        media = np.mean(data)
        mediana = np.median(data)
        try:
            moda_val = stats.mode(data, keepdims=True).mode[0]
            moda_count = stats.mode(data, keepdims=True).count[0]
            moda = f"{moda_val} (freq={moda_count})"
        except Exception:
            moda = "No definida"
        var_m = np.var(data, ddof=1)
        sd_m = np.std(data, ddof=1)
        minimo = np.min(data)
        maximo = np.max(data)
        rango = maximo - minimo
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1

        st.write(f"Tamaño de la muestra (n): {n}")
        st.write(f"Media: {media:.4f}")
        st.write(f"Mediana: {mediana:.4f}")
        st.write(f"Moda: {moda}")
        st.write(f"Desviación estándar (muestral): {sd_m:.4f}")
        st.write(f"Varianza (muestral): {var_m:.4f}")
        st.write(f"Mínimo: {minimo:.4f}  | Máximo: {maximo:.4f}  | Rango: {rango:.4f}")
        st.write(f"Q1: {q1:.4f}  | Q3: {q3:.4f}  | IQR: {iqr:.4f}")

        # Histogram
        fig = px.histogram(data, nbins=20, labels={'value':'Valor'}, title="Histograma de los datos", marginal="box")
        st.plotly_chart(fig, use_container_width=True)

        # Interpretación
        st.subheader("Interpretación")
        interp = f"""
        La media de la muestra es {media:.3f}, mientras que la mediana es {mediana:.3f}. 
        Si la media y la mediana son similares, la distribución es aproximadamente simétrica; 
        si difieren considerablemente, puede haber asimetría. La desviación estándar muestral es {sd_m:.3f}, 
        indicando la dispersión típica alrededor de la media. El rango es {rango:.3f} y el IQR (rango intercuartílico) es {iqr:.3f}, 
        lo cual ayuda a entender la variabilidad robusta frente a valores extremos.
        """
        st.write(interp)

# -------------------------
# PESTAÑA 2: Inferencia estadística
# -------------------------
with tab2:
    st.header("Inferencia estadística")
    st.write("Elija si usar los datos cargados (recomendado) o ingrese estadísticas resumen manualmente.")
    use_data = st.checkbox("Usar los datos cargados como muestra", value=True)
    if use_data and ("datos" not in st.session_state):
        st.warning("No hay datos cargados. Desmarque 'Usar los datos cargados' o cargue datos.")
        use_data = False

    if use_data:
        sample = np.array(st.session_state["datos"], dtype=float)
        n = sample.size
        xbar = np.mean(sample)
        s = np.std(sample, ddof=1)
    else:
        st.write("Introduce resumen de la muestra:")
        n = st.number_input("Tamaño de la muestra n", min_value=1, value=30, step=1)
        xbar = st.number_input("Media muestral (x̄)", value=0.0, format="%.6f")
        s = st.number_input("Desviación estándar muestral (s)", value=1.0, format="%.6f")

    st.markdown("---")
    st.subheader("Error estándar")
    se = s / np.sqrt(n)
    st.write(f"Error estándar de la media (SE) = s / sqrt(n) = {s:.4f}/sqrt({n}) = {se:.6f}")
    st.write("Interpretación: El SE estima la precisión de la media muestral como estimador de la media poblacional; valores menores indican estimaciones más precisas.")

    st.markdown("---")
    st.subheader("Intervalo de confianza de la media")
    conf = st.slider("Nivel de confianza (%)", min_value=80, max_value=99, value=95, step=1)
    alpha = 1 - conf/100
    use_t = st.checkbox("Usar t (desconozco sigma poblacional / n pequeño)", value=True)
    if use_t:
        df = n - 1
        t_crit = stats.t.ppf(1 - alpha/2, df)
        moe = t_crit * se
        lower = xbar - moe
        upper = xbar + moe
        st.write(f"IC {conf}% para la media (t_{df}): [{lower:.6f}, {upper:.6f}] (±{moe:.6f})")
    else:
        # usar z
        z_crit = stats.norm.ppf(1 - alpha/2)
        moe = z_crit * se
        lower = xbar - moe
        upper = xbar + moe
        st.write(f"IC {conf}% para la media (z): [{lower:.6f}, {upper:.6f}] (±{moe:.6f})")
    st.write("Interpretación: Con un nivel de confianza del", conf, "%, esperamos que el intervalo contenga la media poblacional en aproximadamente ese porcentaje de repeticiones del experimento.")

    st.markdown("---")
    st.subheader("Intervalo de confianza para una proporción")
    p_input_mode = st.radio("Ingresar proporción desde:", options=["Datos (promedio de 0/1)", "Resumen (p̂ y n)"])
    if p_input_mode == "Datos (promedio de 0/1)":
        if "datos" in st.session_state:
            # interprete datos como 0/1 if only 0/1 present; else ask user to specify successes and n
            sample_p = np.array(st.session_state["datos"], dtype=float)
            # If sample only zeros and ones consider that; else ask manual
            if set(np.unique(sample_p)) <= {0.0, 1.0}:
                p_hat = sample_p.mean()
                n_p = sample_p.size
            else:
                st.info("Los datos no son 0/1. Introduce p̂ y n manualmente.")
                p_hat = st.number_input("Proporción muestral p̂", min_value=0.0, max_value=1.0, value=0.5)
                n_p = st.number_input("n", min_value=1, value=30, step=1)
        else:
            st.info("No hay datos. Introduce p̂ y n manualmente.")
            p_hat = st.number_input("Proporción muestral p̂", min_value=0.0, max_value=1.0, value=0.5)
            n_p = st.number_input("n", min_value=1, value=30, step=1)
    else:
        p_hat = st.number_input("Proporción muestral p̂", min_value=0.0, max_value=1.0, value=0.5)
        n_p = st.number_input("n", min_value=1, value=100, step=1)

    conf_p = st.slider("Nivel de confianza para proporción (%)", min_value=80, max_value=99, value=95, step=1, key="conf_p")
    alpha_p = 1 - conf_p/100
    zc = stats.norm.ppf(1 - alpha_p/2)
    se_p = math.sqrt(p_hat*(1-p_hat)/n_p)
    moe_p = zc * se_p
    low_p = max(0.0, p_hat - moe_p)
    up_p = min(1.0, p_hat + moe_p)
    st.write(f"IC {conf_p}% para proporción: [{low_p:.6f}, {up_p:.6f}] (p̂={p_hat:.4f}, n={n_p})")
    st.write("Interpretación: Intervalo de confianza para la proporción poblacional. Si n*p̂ y n*(1-p̂) son pequeños, resultados pueden ser poco fiables.")

    st.markdown("---")
    st.subheader("Cálculo de z y t (prueba puntual vs H0)")
    st.write("Proporcione la hipótesis nula mu0 y el tipo de desviación (sigma poblacional conocido o no).")
    mu0 = st.number_input("Valor bajo H0 (μ0)", value=0.0)
    sigma_known = st.checkbox("Asumir sigma poblacional conocido?", value=False)
    if sigma_known:
        sigma = st.number_input("σ poblacional", value=1.0)
        z_stat = (xbar - mu0) / (sigma / math.sqrt(n))
        p_two = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        st.write(f"z = {z_stat:.6f}, p-valor (dos colas) = {p_two:.6f}")
        st.write("Interpretación: Si p-valor < α, rechazamos H0 (evidencia contra H0).")
    else:
        t_stat = (xbar - mu0) / (s / math.sqrt(n))
        p_two_t = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
        st.write(f"t = {t_stat:.6f}, grados de libertad = {n-1}, p-valor (dos colas) = {p_two_t:.6f}")
        st.write("Interpretación: prueba t para una media cuando σ desconocida; compare p-valor con α para concluir.")

    st.markdown("---")
    st.subheader("Tamaño de la muestra")
    st.write("Calcule n requerido para estimar una media o una proporción con error máximo deseado (E).")
    choice = st.radio("Calcular tamaño para:", options=["Media", "Proporción"])
    if choice == "Media":
        E = st.number_input("Margen de error deseado E", min_value=0.0001, value=0.5, format="%.6f")
        sigma_est = st.number_input("Estimación de σ poblacional (si no la tienes, usa s muestra)", value=float(s) if s>0 else 1.0, format="%.6f")
        conf_n = st.slider("Nivel confianza (%)", 80, 99, 95, key="n_media")
        zc_n = stats.norm.ppf(1 - (1 - conf_n/100)/2)
        n_req = (zc_n * sigma_est / E)**2
        n_req = math.ceil(n_req)
        st.write(f"Tamaño de muestra requerido (redondeado arriba): n = {n_req}")
        st.write("Interpretación: Con n observaciones se espera que el margen de error de la media sea ≤ E al nivel de confianza especificado.")
    else:
        E_p = st.number_input("E (margen de error) para proporción", min_value=0.0001, value=0.05, format="%.6f")
        p0 = st.number_input("Estimación de p (usar 0.5 si no sabes)", min_value=0.0, max_value=1.0, value=0.5)
        conf_np = st.slider("Nivel confianza (%)", 80, 99, 95, key="n_prop")
        zc_np = stats.norm.ppf(1 - (1 - conf_np/100)/2)
        n_req_p = (zc_np**2 * p0 * (1-p0)) / (E_p**2)
        n_req_p = math.ceil(n_req_p)
        st.write(f"Tamaño de muestra requerido (redondeado arriba): n = {n_req_p}")
        st.write("Interpretación: Este n garantiza, aproximadamente, que el intervalo para la proporción tenga margen de error ≤ E.")

# -------------------------
# PESTAÑA 3: Dos poblaciones y pruebas
# -------------------------
with tab3:
    st.header("Dos poblaciones — diferencias y pruebas de hipótesis")
    st.write("Seleccione el tipo de análisis:")
    choice = st.radio("", options=["Diferencia de medias (independientes)", "Diferencia de proporciones", "Prueba de hipótesis (medias)", "Prueba de hipótesis (proporciones)"])

    if choice == "Diferencia de medias (independientes)":
        st.subheader("Diferencia de medias — datos o resumen")
        use_groups_data = st.checkbox("Usar datos (dos columnas en CSV previamente cargado)", value=False)
        if use_groups_data and ("_df_uploaded" in st.session_state):
            df = st.session_state["_df_uploaded"]
            st.write("Selecciona columnas para grupo 1 y grupo 2")
            colA = st.selectbox("Columna grupo 1", df.columns, key="g1")
            colB = st.selectbox("Columna grupo 2", df.columns, key="g2")
            g1 = pd.to_numeric(df[colA], errors="coerce").dropna().values
            g2 = pd.to_numeric(df[colB], errors="coerce").dropna().values
            n1 = len(g1); n2 = len(g2)
            x1 = g1.mean(); x2 = g2.mean()
            s1 = g1.std(ddof=1); s2 = g2.std(ddof=1)
        else:
            n1 = st.number_input("n1", min_value=1, value=30, step=1)
            n2 = st.number_input("n2", min_value=1, value=30, step=1)
            x1 = st.number_input("Media muestral grupo 1", value=0.0, format="%.6f")
            x2 = st.number_input("Media muestral grupo 2", value=0.0, format="%.6f")
            s1 = st.number_input("s1 (desv. grupo1)", value=1.0, format="%.6f")
            s2 = st.number_input("s2 (desv. grupo2)", value=1.0, format="%.6f")

        diff = x1 - x2
        # Welch CI
        alpha_dp = 1 - st.slider("Nivel de confianza (%)", 80, 99, 95, key="dp_conf")/100
        conf_level = 1 - alpha_dp
        se_diff = math.sqrt(s1**2 / n1 + s2**2 / n2)
        # Welch df
        num = (s1**2 / n1 + s2**2 / n2)**2
        denom = (s1**4 / (n1**2 * (n1 - 1))) + (s2**4 / (n2**2 * (n2 - 1)))
        df_w = num / denom if denom != 0 else min(n1-1, n2-1)
        tcrit = stats.t.ppf(1 - alpha_dp/2, df=df_w)
        moe = tcrit * se_diff
        lower = diff - moe
        upper = diff + moe
        st.write(f"Diferencia de medias (x̄1 - x̄2) = {diff:.6f}")
        st.write(f"IC {100*(1-alpha_dp)}% (Welch) para la diferencia: [{lower:.6f}, {upper:.6f}]")
        st.write(f"SE diferencia = {se_diff:.6f}, grados de libertad aproximados = {df_w:.2f}")
        st.write("Interpretación: Si el intervalo contiene 0, no hay evidencia concluyente de diferencia de medias.")

    elif choice == "Diferencia de proporciones":
        st.subheader("Diferencia de proporciones: p1 - p2")
        p1 = st.number_input("p̂1", min_value=0.0, max_value=1.0, value=0.5, format="%.6f")
        n1 = st.number_input("n1", min_value=1, value=100, step=1)
        p2 = st.number_input("p̂2", min_value=0.0, max_value=1.0, value=0.4, format="%.6f")
        n2 = st.number_input("n2", min_value=1, value=100, step=1)
        conf_dp = st.slider("Nivel de confianza (%)", 80, 99, 95, key="diffprop_conf")
        alpha_dp = 1 - conf_dp/100
        zc = stats.norm.ppf(1 - alpha_dp/2)
        se_diffp = math.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
        diffp = p1 - p2
        moe = zc * se_diffp
        low = diffp - moe
        up = diffp + moe
        st.write(f"Diferencia de proporciones p̂1 - p̂2 = {diffp:.6f}")
        st.write(f"IC {conf_dp}%: [{low:.6f}, {up:.6f}] (SE = {se_diffp:.6f})")
        st.write("Interpretación: Si el intervalo contiene 0, no se evidencia diferencia significativa entre proporciones.")

    elif choice == "Prueba de hipótesis (medias)":
        st.subheader("Prueba de hipótesis para medias (una o dos muestras)")
        test_type = st.selectbox("Tipo de prueba", options=["Una muestra", "Dos muestras independientes (Welch)"])
        alpha = st.number_input("Nivel de significancia α", min_value=0.0001, max_value=0.5, value=0.05, format="%.4f")
        if test_type == "Una muestra":
            if "datos" in st.session_state:
                use_this = st.checkbox("Usar datos cargados para prueba", value=True)
            else:
                use_this = False
            if use_this:
                sample = np.array(st.session_state["datos"], dtype=float)
                n = sample.size
                xbar = sample.mean()
                s = sample.std(ddof=1)
            else:
                n = st.number_input("n", min_value=2, value=30, step=1)
                xbar = st.number_input("x̄", value=0.0, format="%.6f")
                s = st.number_input("s", value=1.0, format="%.6f")
            mu0 = st.number_input("Valor H0 (μ0)", value=0.0)
            t_stat = (xbar - mu0) / (s / math.sqrt(n))
            p_two = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
            st.write(f"t = {t_stat:.6f}, df = {n-1}, p-valor (dos colas) = {p_two:.6f}")
            if p_two < alpha:
                st.success(f"Rechazamos H0 al nivel α={alpha}.")
            else:
                st.info(f"No rechazamos H0 al nivel α={alpha}.")
            st.write("Interpretación: usar resultado y contexto para decidir si la diferencia es práctica además de estadística.")
        else:
            # Two-sample Welch test
            st.write("Proporciona medias, desviaciones y tamaños para ambos grupos.")
            n1 = st.number_input("n1", min_value=2, value=30, step=1, key="hp_n1")
            x1 = st.number_input("x̄1", value=0.0, format="%.6f", key="hp_x1")
            s1 = st.number_input("s1", value=1.0, format="%.6f", key="hp_s1")
            n2 = st.number_input("n2", min_value=2, value=30, step=1, key="hp_n2")
            x2 = st.number_input("x̄2", value=0.0, format="%.6f", key="hp_x2")
            s2 = st.number_input("s2", value=1.0, format="%.6f", key="hp_s2")
            # H0: mu1 - mu2 = 0
            diff = x1 - x2
            se = math.sqrt(s1**2/n1 + s2**2/n2)
            num = (s1**2 / n1 + s2**2 / n2)**2
            denom = (s1**4 / (n1**2 * (n1 - 1))) + (s2**4 / (n2**2 * (n2 - 1)))
            dfw = num / denom if denom != 0 else min(n1-1, n2-1)
            t_stat = diff / se
            p_two = 2 * (1 - stats.t.cdf(abs(t_stat), df=dfw))
            st.write(f"t (Welch) = {t_stat:.6f}, df ≈ {dfw:.2f}, p-valor (dos colas) = {p_two:.6f}")
            if p_two < alpha:
                st.success(f"Rechazamos H0 al nivel α={alpha}.")
            else:
                st.info(f"No rechazamos H0 al nivel α={alpha}.")
            st.write("Interpretación: revise magnitud del efecto y confianza del intervalo además del p-valor.")

    else:  # Prueba de hipótesis (proporciones)
        st.subheader("Prueba de hipótesis para proporciones (una o dos muestras)")
        test_kind = st.selectbox("Tipo", options=["Una proporción", "Dos proporciones (independientes)"])
        alpha = st.number_input("Nivel de significancia α", min_value=0.0001, max_value=0.5, value=0.05, format="%.4f", key="hp_prop_alpha")
        if test_kind == "Una proporción":
            x = st.number_input("Número de éxitos (x)", min_value=0, value=30, step=1)
            n = st.number_input("n", min_value=1, value=100, step=1)
            p0 = st.number_input("p0 (valor bajo H0)", min_value=0.0, max_value=1.0, value=0.5, format="%.6f")
            phat = x / n
            se0 = math.sqrt(p0*(1-p0)/n)
            z = (phat - p0) / se0
            p_two = 2 * (1 - stats.norm.cdf(abs(z)))
            st.write(f"p̂ = {phat:.6f}, z = {z:.6f}, p-valor (dos colas) = {p_two:.6f}")
            if p_two < alpha:
                st.success("Rechazamos H0.")
            else:
                st.info("No rechazamos H0.")
            st.write("Interpretación: la prueba usa la aproximación normal bajo H0; cuidado si n es pequeño.")
        else:
            x1 = st.number_input("x1 (éxitos grupo1)", min_value=0, value=30, step=1, key="xp_x1")
            n1 = st.number_input("n1", min_value=1, value=100, step=1, key="xp_n1")
            x2 = st.number_input("x2 (éxitos grupo2)", min_value=0, value=20, step=1, key="xp_x2")
            n2 = st.number_input("n2", min_value=1, value=100, step=1, key="xp_n2")
            phat1 = x1 / n1
            phat2 = x2 / n2
            phat_pool = (x1 + x2) / (n1 + n2)
            se_pool = math.sqrt(phat_pool*(1-phat_pool)*(1/n1 + 1/n2))
            z = (phat1 - phat2) / se_pool
            p_two = 2 * (1 - stats.norm.cdf(abs(z)))
            st.write(f"p̂1 = {phat1:.6f}, p̂2 = {phat2:.6f}")
            st.write(f"z (prueba de diferencia con pooled) = {z:.6f}, p-valor (dos colas) = {p_two:.6f}")
            if p_two < alpha:
                st.success("Rechazamos H0 de igualdad de proporciones.")
            else:
                st.info("No rechazamos H0.")
            st.write("Interpretación: pruebe también sin agrupar si supuestos no se cumplen, o calcule IC para diferencia de proporciones.")

# -------------------------
# PESTAÑA 4: TLC / Gráficos
# -------------------------
with tab4:
    st.header("Teorema central del límite y gráficos")
    st.write("Simula distribuciones muestrales de la media a partir de una población (o datos cargados).")
    population_choice = st.radio("Población a usar:", options=["Usar datos cargados (si existen)", "Generar distribución"], index=0)
    if population_choice == "Usar datos cargados" and ("datos" not in st.session_state):
        st.warning("No hay datos. Cambia a 'Generar distribución' o carga datos.")
        population_choice = "Generar distribución"

    if population_choice == "Generar distribución":
        dist = st.selectbox("Selecciona una distribución poblacional", options=["Normal", "Exponencial", "Log-normal", "Uniforme", "Binomial"])
        pop_n = st.number_input("Tamaño población (simulado)", min_value=1000, value=10000, step=1000)
        if dist == "Normal":
            mu_pop = st.number_input("μ poblacional", value=10.0)
            sigma_pop = st.number_input("σ poblacional", value=5.0)
            population = np.random.normal(loc=mu_pop, scale=sigma_pop, size=pop_n)
        elif dist == "Exponencial":
            lam = st.number_input("Lambda (rate)", value=1.0)
            population = np.random.exponential(scale=1/lam, size=pop_n)
        elif dist == "Log-normal":
            mu_l = st.number_input("mu (log-space)", value=0.0)
            sigma_l = st.number_input("sigma (log-space)", value=1.0)
            population = np.random.lognormal(mean=mu_l, sigma=sigma_l, size=pop_n)
        elif dist == "Uniforme":
            a = st.number_input("a", value=0.0)
            b = st.number_input("b", value=1.0)
            population = np.random.uniform(a, b, size=pop_n)
        else:  # Binomial
            trials = st.number_input("ensayos (n)", min_value=1, value=1)
            pbin = st.number_input("p", min_value=0.0, max_value=1.0, value=0.5)
            population = np.random.binomial(trials, pbin, size=pop_n)
    else:
        population = np.array(st.session_state["datos"], dtype=float)

    st.write(f"Población usada: tamaño {len(population)}; media poblacional ≈ {population.mean():.4f}; sd ≈ {population.std(ddof=0):.4f}")
    st.markdown("---")
    st.subheader("Simular distribución muestral de la media")
    k = st.number_input("Número de réplicas (simulaciones)", min_value=100, value=2000, step=100)
    sample_size = st.number_input("Tamaño de muestra en cada réplica (n)", min_value=2, value=30, step=1)
    run_sim = st.button("Simular distribución muestral")
    if run_sim:
        means = []
        for _ in range(int(k)):
            s = np.random.choice(population, size=int(sample_size), replace=True)
            means.append(s.mean())
        means = np.array(means)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=means, nbinsx=40, name="Medias muestrales", histnorm='probability density'))
        # superponer normal aprox
        mu_means = means.mean()
        sd_means = means.std(ddof=1)
        x_axis = np.linspace(mu_means - 4*sd_means, mu_means + 4*sd_means, 200)
        y_axis = stats.norm.pdf(x_axis, loc=mu_means, scale=sd_means)
        fig.add_trace(go.Scatter(x=x_axis, y=y_axis, mode='lines', name='Normal aprox', line=dict(color='red')))
        fig.update_layout(title=f"Distribución muestral de la media (n={sample_size}, réplicas={k})",
                          xaxis_title="Media muestral", yaxis_title="Densidad")
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Media de las medias ≈ {mu_means:.6f}; SE empírico ≈ {sd_means:.6f}")
        st.write("Interpretación: según el TLC, si n es suficientemente grande, la distribución de medias tiende a normal aun si la población no lo es.")

    st.markdown("---")
    st.subheader("Complementario del Error Estándar (áreas)")
    z_val = st.number_input("Valor z (ej: 1.96)", value=1.96, format="%.4f")
    tail = st.selectbox("Área a calcular", options=["Área a la izquierda (P(Z < z))", "Área a la derecha (P(Z > z))", "Área entre -z y z (P(|Z| < z))"])
    if st.button("Calcular área z"):
        if tail == "Área a la izquierda (P(Z < z))":
            area = stats.norm.cdf(z_val)
            st.write(f"P(Z < {z_val}) = {area:.6f}")
        elif tail == "Área a la derecha (P(Z > z))":
            area = 1 - stats.norm.cdf(z_val)
            st.write(f"P(Z > {z_val}) = {area:.6f}")
        else:
            area = stats.norm.cdf(z_val) - stats.norm.cdf(-z_val)
            st.write(f"P(|Z| < {z_val}) = {area:.6f}")
        st.write("Interpretación: estas áreas se usan al construir intervalos y pruebas de hipótesis.")

    st.markdown("---")
    st.subheader("Histograma de la población")
    bins = st.slider("Número de bins", 5, 100, 30)
    fig2 = px.histogram(population, nbins=bins, marginal="box", title="Histograma de la población (o datos cargados)")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.write("App creada: funciones para cálculo y visualización de técnicas estadísticas básicas. Interpretaciones en cada sección ayudan a contextualizar resultados.")
st.write("Si quieres que adapte la app (p.ej. más opciones, exportar resultados, tests exactos, o soporte para datos emparejados), dime qué agregar.")
