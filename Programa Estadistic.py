import streamlit as st
import numpy as np
import pandas as pd
import math
import traceback

# librerías opcionales
try:
    from scipy import stats
except Exception:
    stats = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None

st.set_page_config(layout="wide", page_title="Calculadora Estadística (debug)")

st.title("Calculadora Estadística (debug) — si falla, pega el error aquí")

st.write("Primero: si la app se cae, copia y pega el mensaje rojo (traceback) aquí o en el chat para que lo revise.")

# Mostrar versiones (útil para diagnóstico)
st.caption(f"Python y paquetes (diagnóstico): numpy {np.__version__}, pandas {pd.__version__}, scipy {'no instalado' if stats is None else getattr(stats, '__version__', 'ok')}, streamlit (ver terminal)")

# Entrada de datos
st.markdown("## Entrada de datos")
data_input = st.text_area("Datos (numeros separados por comas), o sube CSV abajo", height=80)
upload = st.file_uploader("Subir CSV (una columna numérica)", type=["csv", "txt"])
load_btn = st.button("Cargar datos")

def parse_text_to_array(text):
    try:
        arr = [float(x.strip()) for x in text.split(",") if x.strip() != ""]
        return np.array(arr, dtype=float)
    except Exception:
        return None

if load_btn:
    try:
        ds = None
        if upload is not None:
            df = pd.read_csv(upload)
            st.session_state["_df_uploaded"] = df
            if df.shape[1] == 1:
                col = df.columns[0]
                arr = pd.to_numeric(df[col], errors="coerce").dropna().values
                ds = np.array(arr, dtype=float)
            else:
                st.info("CSV con múltiples columnas cargado. Usa la pestaña correspondiente o selecciona columna más abajo.")
                ds = None
        elif data_input and not upload:
            arr = parse_text_to_array(data_input)
            if arr is None:
                st.error("No se pudieron parsear los datos. Revisa el formato (ej: 10, 20, 15).")
            else:
                ds = arr
        if ds is not None:
            st.session_state["datos"] = ds.tolist()
            st.success(f"Datos cargados correctamente (n={len(ds)}).")
    except Exception as e:
        st.error("Error cargando datos. Ver traceback:")
        st.text(traceback.format_exc())

# Pestañas
tab1, tab2, tab3, tab4 = st.tabs(["Tendencia central", "Inferencia", "Dos poblaciones y pruebas", "TLC / Gráficos"])

# ---------- PESTAÑA 1 ----------
with tab1:
    st.header("Medidas de tendencia central")
    try:
        if "datos" not in st.session_state:
            st.info("Carga datos en la parte superior de la app.")
        else:
            data = np.array(st.session_state["datos"], dtype=float)
            if data.size == 0:
                st.warning("Los datos están vacíos.")
            else:
                n = data.size
                media = np.mean(data)
                mediana = np.median(data)
                # modo compatible
                moda = "No definida"
                try:
                    if stats is not None:
                        mres = stats.mode(data, keepdims=True)
                        moda = f"{mres.mode[0]} (freq={mres.count[0]})"
                except Exception:
                    # fallback sencillo
                    vals, counts = np.unique(data, return_counts=True)
                    idx = np.argmax(counts)
                    moda = f"{vals[idx]} (freq={counts[idx]})"
                var_m = np.var(data, ddof=1) if n > 1 else 0.0
                sd_m = np.std(data, ddof=1) if n > 1 else 0.0
                minimo = np.min(data)
                maximo = np.max(data)
                rango = maximo - minimo
                q1 = np.percentile(data, 25)
                q3 = np.percentile(data, 75)
                iqr = q3 - q1

                st.write(f"n = {n}")
                st.write(f"Media = {media:.6f}")
                st.write(f"Mediana = {mediana:.6f}")
                st.write(f"Moda = {moda}")
                st.write(f"Desviación estándar (muestral) = {sd_m:.6f}")
                st.write(f"Varianza (muestral) = {var_m:.6f}")
                st.write(f"Mínimo = {minimo:.6f}  Máximo = {maximo:.6f}  Rango = {rango:.6f}")
                st.write(f"Q1 = {q1:.6f}  Q3 = {q3:.6f}  IQR = {iqr:.6f}")

                if px is not None:
                    fig = px.histogram(data, nbins=20, title="Histograma")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("plotly no está disponible para graficar el histograma.")
                st.subheader("Interpretación")
                st.write(f"La media es {media:.3f} y la mediana {mediana:.3f}. La desviación estándar muestral es {sd_m:.3f}.")
    except Exception:
        st.error("Error en la pestaña de tendencia central:")
        st.text(traceback.format_exc())

# ---------- PESTAÑA 2 ----------
with tab2:
    st.header("Inferencia estadística")
    try:
        use_data = st.checkbox("Usar datos cargados", value=True)
        if use_data and ("datos" not in st.session_state):
            st.warning("No hay datos; desmarca 'Usar datos cargados' o carga datos.")
            use_data = False

        if use_data:
            sample = np.array(st.session_state["datos"], dtype=float)
            n = sample.size
            xbar = sample.mean()
            s = sample.std(ddof=1) if n > 1 else 0.0
        else:
            n = st.number_input("n (tamaño)", min_value=1, value=30, step=1)
            xbar = st.number_input("x̄ (media muestral)", value=0.0, format="%.6f")
            s = st.number_input("s (desviación muestral)", value=1.0, format="%.6f")

        st.subheader("Error estándar")
        if n > 0:
            se = s / math.sqrt(n) if n > 0 else float("nan")
            st.write(f"SE = {se:.6f}")
        else:
            st.warning("n debe ser > 0 para calcular SE.")

        st.subheader("Intervalo de confianza de la media")
        conf = st.slider("Nivel confianza (%)", 80, 99, 95)
        alpha = 1 - conf/100
        use_t = st.checkbox("Usar t en lugar de z (por defecto True)", value=True)
        if n <= 1:
            st.warning("n debe ser >1 para IC de la media.")
        else:
            if use_t:
                if stats is None:
                    st.error("scipy no instalado: no puedo calcular t-crit.")
                else:
                    df = n - 1
                    tcrit = stats.t.ppf(1 - alpha/2, df)
                    moe = tcrit * se
                    st.write(f"IC {conf}% (t, df={df}): [{xbar - moe:.6f}, {xbar + moe:.6f}]")
            else:
                zcrit = 1.96 if stats is None else stats.norm.ppf(1 - alpha/2)
                moe = zcrit * se
                st.write(f"IC {conf}% (z): [{xbar - moe:.6f}, {xbar + moe:.6f}]")

        st.subheader("Intervalo de confianza para proporción")
        p_hat = st.number_input("p̂ (proporción muestral)", min_value=0.0, max_value=1.0, value=0.5, format="%.6f")
        n_p = st.number_input("n (para proporción)", min_value=1, value=100, step=1)
        conf_p = st.slider("Nivel confianza para proporción (%)", 80, 99, 95, key="conf_p_tab2")
        alpha_p = 1 - conf_p/100
        if n_p * p_hat < 5 or n_p * (1-p_hat) < 5:
            st.caption("Advertencia: aproximación normal puede no ser adecuada (n*p̂ o n*(1-p̂) pequeño).")
        zc = 1.96 if stats is None else stats.norm.ppf(1 - alpha_p/2)
        se_p = math.sqrt(p_hat*(1-p_hat)/n_p)
        moe_p = zc * se_p
        low_p = max(0.0, p_hat - moe_p)
        up_p = min(1.0, p_hat + moe_p)
        st.write(f"IC {conf_p}% para proporción: [{low_p:.6f}, {up_p:.6f}] (±{moe_p:.6f})")

        st.subheader("Cálculo z/t para prueba vs H0")
        mu0 = st.number_input("μ0 (valor bajo H0)", value=0.0, key="mu0_tab2")
        sigma_known = st.checkbox("σ poblacional conocido (usar z)", value=False, key="sigma_known_tab2")
        if sigma_known:
            sigma = st.number_input("σ poblacional", value=1.0, format="%.6f", key="sigma_tab2")
            if sigma <= 0:
                st.error("σ debe ser > 0")
            else:
                z_stat = (xbar - mu0) / (sigma / math.sqrt(max(1, n)))
                p_two = 2 * (1 - (stats.norm.cdf(abs(z_stat)) if stats is not None else 0.5))
                st.write(f"z = {z_stat:.6f}, p-valor (dos colas) ≈ {p_two:.6f}")
        else:
            if n <= 1:
                st.warning("n debe ser >1 para calcular t.")
            else:
                t_stat = (xbar - mu0) / (s / math.sqrt(n)) if s > 0 else float("inf")
                p_two_t = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1)) if stats is not None else float("nan")
                st.write(f"t = {t_stat:.6f}, p-valor (dos colas) ≈ {p_two_t:.6f}")
    except Exception:
        st.error("Error en la pestaña de inferencia:")
        st.text(traceback.format_exc())

# ---------- PESTAÑA 3 ----------
with tab3:
    st.header("Dos poblaciones y pruebas")
    try:
        analysis = st.selectbox("Análisis", ["Diferencia de medias", "Diferencia de proporciones", "Prueba de hipótesis para proporciones"])
        if analysis == "Diferencia de medias":
            st.write("Introduce resúmenes para grupo 1 y grupo 2")
            n1 = st.number_input("n1", min_value=1, value=30, key="d_n1")
            x1 = st.number_input("x̄1", value=0.0, format="%.6f", key="d_x1")
            s1 = st.number_input("s1", value=1.0, format="%.6f", key="d_s1")
            n2 = st.number_input("n2", min_value=1, value=30, key="d_n2")
            x2 = st.number_input("x̄2", value=0.0, format="%.6f", key="d_x2")
            s2 = st.number_input("s2", value=1.0, format="%.6f", key="d_s2")
            diff = x1 - x2
            se = math.sqrt((s1**2)/max(1, n1) + (s2**2)/max(1, n2))
            st.write(f"Diferencia de medias = {diff:.6f}, SE ≈ {se:.6f}")
            conf = st.slider("Nivel confianza (%)", 80, 99, 95, key="d_conf")
            alpha = 1 - conf/100
            if stats is not None and n1>1 and n2>1:
                # Welch df
                num = (s1**2 / n1 + s2**2 / n2)**2
                denom = (s1**4 / (n1**2 * (n1 - 1))) + (s2**4 / (n2**2 * (n2 - 1)))
                dfw = num / denom if denom != 0 else min(n1-1, n2-1)
                tcrit = stats.t.ppf(1 - alpha/2, df=dfw)
                moe = tcrit * se
                st.write(f"IC {conf}% (Welch): [{diff - moe:.6f}, {diff + moe:.6f}] (df≈{dfw:.2f})")
            else:
                st.write("No es posible calcular IC con t (scipy ausente o n<=1).")
        elif analysis == "Diferencia de proporciones":
            p1 = st.number_input("p̂1", min_value=0.0, max_value=1.0, value=0.5, key="dp_p1")
            n1 = st.number_input("n1", min_value=1, value=100, key="dp_n1")
            p2 = st.number_input("p̂2", min_value=0.0, max_value=1.0, value=0.4, key="dp_p2")
            n2 = st.number_input("n2", min_value=1, value=100, key="dp_n2")
            diffp = p1 - p2
            se = math.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
            conf = st.slider("Nivel confianza (%)", 80, 99, 95, key="dp_conf")
            alpha = 1 - conf/100
            zc = 1.96 if stats is None else stats.norm.ppf(1 - alpha/2)
            moe = zc * se
            st.write(f"Diferencia p̂1-p̂2 = {diffp:.6f}, IC {conf}%: [{diffp - moe:.6f}, {diffp + moe:.6f}]")
        else:
            st.write("Prueba de hipótesis para proporciones (dos muestras):")
            x1 = st.number_input("x1 éxitos", min_value=0, value=30, key="hp_x1")
            n1 = st.number_input("n1", min_value=1, value=100, key="hp_n1")
            x2 = st.number_input("x2 éxitos", min_value=0, value=20, key="hp_x2")
            n2 = st.number_input("n2", min_value=1, value=100, key="hp_n2")
            phat1 = x1 / n1
            phat2 = x2 / n2
            pool = (x1 + x2) / (n1 + n2)
            se_pool = math.sqrt(pool * (1 - pool) * (1/n1 + 1/n2))
            if se_pool == 0:
                st.error("SE pooled = 0 (revisa valores).")
            else:
                z = (phat1 - phat2) / se_pool
                pval = 2 * (1 - (stats.norm.cdf(abs(z)) if stats is not None else 0.5))
                st.write(f"z = {z:.6f}, p-valor ≈ {pval:.6f}")
    except Exception:
        st.error("Error en la pestaña de dos poblaciones:")
        st.text(traceback.format_exc())

# ---------- PESTAÑA 4 ----------
with tab4:
    st.header("TLC y gráficos")
    try:
        choice = st.radio("Usar:", ["Datos cargados", "Generar población"], index=0)
        if choice == "Datos cargados" and ("datos" not in st.session_state):
            st.warning("No hay datos cargados.")
            choice = "Generar población"
        if choice == "Generar población":
            dist = st.selectbox("Distribución", ["Normal", "Exponencial", "Uniforme"])
            pop_n = st.number_input("Tamaño población simulado", min_value=1000, value=10000, step=1000)
            if dist == "Normal":
                mu = st.number_input("μ", value=0.0)
                sigma = st.number_input("σ", value=1.0)
                population = np.random.normal(mu, sigma, size=pop_n)
            elif dist == "Exponencial":
                scale = st.number_input("scale", value=1.0)
                population = np.random.exponential(scale, size=pop_n)
            else:
                a = st.number_input("a", value=0.0)
                b = st.number_input("b", value=1.0)
                population = np.random.uniform(a, b, size=pop_n)
        else:
            population = np.array(st.session_state["datos"], dtype=float)

        st.write(f"Población: n={len(population)}, media≈{population.mean():.4f}, sd≈{population.std(ddof=0):.4f}")
        repl = st.number_input("Réplicas", min_value=100, value=1000, step=100)
        sample_n = st.number_input("Tamaño de muestra (por réplica)", min_value=2, value=30, step=1)
        if st.button("Simular"):
            means = []
            for _ in range(int(repl)):
                s = np.random.choice(population, size=int(sample_n), replace=True)
                means.append(s.mean())
            means = np.array(means)
            st.write(f"Medias simuladas: media≈{means.mean():.6f}, sd≈{means.std(ddof=1):.6f}")
            if px is not None:
                fig = px.histogram(means, nbins=40, title="Distribución muestral de la media", marginal="box")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("plotly no disponible.")
    except Exception:
        st.error("Error en TLC / gráficos:")
        st.text(traceback.format_exc())

st.markdown("---")
st.write("Si hay algún error, pega aquí el traceback completo (texto rojo) y te lo arreglaré. Además dime la versión de Python y Streamlit.")
