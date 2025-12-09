import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

sns.set_style("whitegrid")

# -----------------------
# Helpers
# -----------------------

def parse_numeric_list(text: str):
    """Convierte una cadena con números separados por comas, espacios o saltos de línea a lista de floats."""
    if not text:
        return []
    # Reemplazar saltos de línea por comas, luego dividir por comas
    items = [item.strip() for item in text.replace("\n", ",").split(",") if item.strip() != ""]
    nums = []
    for it in items:
        try:
            nums.append(float(it))
        except:
            # intentar interpretar como punto decimal con coma
            try:
                nums.append(float(it.replace(",", ".")))
            except:
                raise ValueError(f"Valor no numérico: '{it}'")
    return nums

def se_mean(sd: float, n: int) -> float:
    return sd / np.sqrt(n) if n > 0 else np.nan

def ci_mean_sample(data: np.ndarray, conf=0.95) -> Tuple[float,float,float]:
    n = len(data)
    mean = np.mean(data)
    sd = np.std(data, ddof=1)
    se = se_mean(sd, n)
    t_crit = stats.t.ppf((1 + conf) / 2, df=n-1)
    lo = mean - t_crit * se
    hi = mean + t_crit * se
    return mean, lo, hi

def ci_mean_known_sigma(mean: float, sigma: float, n: int, conf=0.95):
    z = stats.norm.ppf((1 + conf) / 2)
    se = sigma / np.sqrt(n)
    lo = mean - z * se
    hi = mean + z * se
    return lo, hi, se

def ci_prop(p_hat: float, n: int, conf=0.95, method="wald"):
    z = stats.norm.ppf((1 + conf) / 2)
    if method == "wald":
        se = np.sqrt(p_hat * (1 - p_hat) / n)
        lo = p_hat - z * se
        hi = p_hat + z * se
    elif method == "wilson":
        # Wilson score interval
        denom = 1 + z**2 / n
        center = p_hat + z**2/(2*n)
        adj = z * np.sqrt((p_hat*(1-p_hat) + z**2/(4*n))/n)
        lo = (center - adj) / denom
        hi = (center + adj) / denom
        se = np.sqrt(p_hat*(1-p_hat)/n)
    else:
        raise ValueError("Método desconocido")
    lo = max(0, lo)
    hi = min(1, hi)
    return lo, hi, se

def z_test_mean(mean_sample, mu0, sigma, n):
    """Z test cuando sigma poblacional conocido."""
    se = sigma / np.sqrt(n)
    z = (mean_sample - mu0) / se
    p_two = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_two

def t_test_one_sample(data, mu0):
    n = len(data)
    mean = np.mean(data)
    sd = np.std(data, ddof=1)
    se = sd / np.sqrt(n)
    t = (mean - mu0) / se
    p_two = 2 * (1 - stats.t.cdf(abs(t), df=n-1))
    return t, p_two

def sample_size_mean(sigma, E, conf=0.95):
    z = stats.norm.ppf((1 + conf) / 2)
    n = (z * sigma / E) ** 2
    return int(np.ceil(n))

def sample_size_prop(p, E, conf=0.95):
    z = stats.norm.ppf((1 + conf) / 2)
    n = (z**2 * p * (1-p)) / (E**2)
    return int(np.ceil(n))

def two_sample_ttest(x, y, equal_var=False):
    res = stats.ttest_ind(x, y, equal_var=equal_var)
    # res.statistic is t, res.pvalue is two-sided p
    return res.statistic, res.pvalue

def diff_means_ci(x, y, conf=0.95, equal_var=False):
    nx = len(x); ny = len(y)
    mx = np.mean(x); my = np.mean(y)
    if equal_var:
        # pooled sd
        sx2 = np.var(x, ddof=1); sy2 = np.var(y, ddof=1)
        sp2 = ((nx-1)*sx2 + (ny-1)*sy2) / (nx+ny-2)
        se = np.sqrt(sp2*(1/nx + 1/ny))
        df = nx + ny - 2
    else:
        sx2 = np.var(x, ddof=1); sy2 = np.var(y, ddof=1)
        se = np.sqrt(sx2/nx + sy2/ny)
        # Welch-Satterthwaite df
        num = (sx2/nx + sy2/ny)**2
        den = (sx2**2)/((nx**2)*(nx-1)) + (sy2**2)/((ny**2)*(ny-1))
        df = num/den if den>0 else min(nx, ny)-1
    diff = mx - my
    tcrit = stats.t.ppf((1+conf)/2, df=df)
    lo = diff - tcrit*se
    hi = diff + tcrit*se
    return diff, lo, hi, se, df

def diff_prop_test(x1_success, n1, x2_success, n2, conf=0.95):
    p1 = x1_success / n1
    p2 = x2_success / n2
    diff = p1 - p2
    # pooled for test
    p_pool = (x1_success + x2_success) / (n1 + n2)
    se_pool = np.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2))
    z = diff / se_pool
    p_two = 2*(1 - stats.norm.cdf(abs(z)))
    # CI (unpooled)
    se_unpooled = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    zcrit = stats.norm.ppf((1+conf)/2)
    lo = diff - zcrit*se_unpooled
    hi = diff + zcrit*se_unpooled
    return diff, z, p_two, lo, hi

def format_pct(x):
    return f"{100*x:.3f}%"

# -----------------------
# Streamlit App
# -----------------------

st.set_page_config(page_title="Calculadora Estadística", layout="wide")
st.title("Calculadora Estadística con Pestañas")
st.write("Aplicación en español que organiza cálculos por temas y ofrece interpretaciones al final de cada sección.")

# Tabs
tabs = st.tabs([
    "Datos",
    "Medidas de tendencia central",
    "Inferencia estadística (1 muestra)",
    "Dos poblaciones",
    "Gráficos y distribuciones",
    "Ejemplos / Interpretación"
])

# -----------------------
# PESTAÑA: Datos
# -----------------------
with tabs[0]:
    st.header("Carga y gestión de datos")
    st.write("Ingrese una muestra numérica como lista separada por comas o pegue una columna de números.")
    st.code("10, 20, 15, 30, 25\n(ó una columna copiando desde Excel)")
    col1, col2 = st.columns([2,1])
    with col1:
        data_text = st.text_area("Datos (lista):", height=140, placeholder="10, 20, 15, 30, 25")
        uploaded_file = st.file_uploader("O sube un archivo CSV (primera columna usada)", type=["csv","txt"], accept_multiple_files=False)
        if st.button("Cargar datos"):
            try:
                data = []
                if uploaded_file is not None:
                    df = pd.read_csv(uploaded_file, header=None)
                    # tomar la primera columna y descartar NaN
                    series = pd.to_numeric(df.iloc[:,0], errors="coerce").dropna()
                    data = series.tolist()
                else:
                    data = parse_numeric_list(data_text)
                if len(data) == 0:
                    st.error("No se detectaron datos válidos.")
                else:
                    st.session_state["datos"] = data
                    st.success(f"Datos cargados: {len(data)} observaciones.")
            except Exception as e:
                st.error(f"Error al parsear datos: {e}")
    with col2:
        if "datos" in st.session_state:
            d = np.array(st.session_state["datos"])
            st.write("Vista previa (primeros 10):")
            st.write(d[:10])
            if st.button("Borrar datos"):
                del st.session_state["datos"]
                st.success("Datos eliminados.")

# -----------------------
# PESTAÑA: Medidas de tendencia central
# -----------------------
with tabs[1]:
    st.header("Medidas de tendencia central y dispersión")
    if "datos" not in st.session_state:
        st.warning("Carga los datos en la pestaña 'Datos' para calcular estadísticas.")
    else:
        data = np.array(st.session_state["datos"])
        n = len(data)
        mean = np.mean(data)
        median = np.median(data)
        try:
            mode_val = stats.mode(data, keepdims=True).mode[0]
            mode_count = stats.mode(data, keepdims=True).count[0]
        except:
            mode_val, mode_count = np.nan, 0
        var = np.var(data, ddof=1)
        sd = np.std(data, ddof=1)
        skew = stats.skew(data)
        kurt = stats.kurtosis(data)
        mn = np.min(data)
        mx = np.max(data)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        st.subheader("Resultados numéricos")
        st.write(f"N = {n}")
        st.write(f"Media = {mean:.6f}")
        st.write(f"Mediana = {median:.6f}")
        if not np.isnan(mode_val):
            st.write(f"Moda = {mode_val} (frecuencia = {mode_count})")
        st.write(f"Varianza (muestral) = {var:.6f}")
        st.write(f"Desviación estándar (muestral) = {sd:.6f}")
        st.write(f"Min = {mn:.6f}, Max = {mx:.6f}, Rango = {mx-mn:.6f}")
        st.write(f"Q1 = {q1:.6f}, Q3 = {q3:.6f}, IQR = {iqr:.6f}")
        st.write(f"Asimetría (skewness) = {skew:.6f}")
        st.write(f"Curtosis (exceso) = {kurt:.6f}")

        st.subheader("Interpretación")
        interp = []
        interp.append(f"La media ({mean:.3f}) resume el valor central de la muestra; la mediana ({median:.3f}) muestra la tendencia central robusta ante valores extremos.")
        if abs(mean-median) > 0.1*abs(mean) if mean!=0 else (abs(mean-median)>0.1):
            interp.append("Diferencia notable entre media y mediana sugiere asimetría en la distribución.")
        if skew > 0.5:
            interp.append("La asimetría positiva indica cola derecha más pronunciada.")
        elif skew < -0.5:
            interp.append("La asimetría negativa indica cola izquierda más pronunciada.")
        else:
            interp.append("Asimetría cercana a 0, distribución relativamente simétrica.")
        interp.append(f"La desviación estándar ({sd:.3f}) mide dispersión; comparada con la media da idea de variabilidad relativa.")
        st.write("\n\n".join(interp))

# -----------------------
# PESTAÑA: Inferencia (1 muestra)
# -----------------------
with tabs[2]:
    st.header("Inferencia estadística (una muestra)")
    st.write("Aquí puedes calcular error estándar, intervalos de confianza, pruebas para la media o proporción y tamaño de muestra.")
    st.markdown("---")
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Error estándar y CI para la media")
        use_data = st.checkbox("Usar datos cargados para la media", value=True)
        if use_data and "datos" not in st.session_state:
            st.warning("Carga datos en la pestaña Datos para usar esta opción.")
        if use_data and "datos" in st.session_state:
            data = np.array(st.session_state["datos"])
            n = len(data)
            mean = np.mean(data)
            sd = np.std(data, ddof=1)
            se = se_mean(sd, n)
            conf = st.slider("Confianza (%)", 80, 99, 95)
            mean_val, ci_lo, ci_hi = ci_mean_sample(data, conf/100)
            st.write(f"Media muestral = {mean:.6f}")
            st.write(f"Desviación estándar (muestral) = {sd:.6f}")
            st.write(f"Error estándar = {se:.6f}")
            st.write(f"Intervalo de confianza {conf}% para la media (t): [{ci_lo:.6f}, {ci_hi:.6f}]")
            st.write("Interpretación:")
            st.write(f"Con un {conf}% de confianza, el intervalo contiene el valor real de la media poblacional bajo supuestos de muestreo aleatorio. El error estándar indica cuánto varía la media muestral entre muestras.")
        else:
            st.subheader("CI para la media (con sigma conocida)")
            mean_input = st.number_input("Media muestral (si no usa datos)", value=0.0, format="%.6f")
            sigma = st.number_input("Sigma poblacional (σ). Si se desconoce, use t con los datos.", value=1.0, format="%.6f")
            n = st.number_input("Tamaño de la muestra n", value=30, min_value=1)
            conf = st.slider("Confianza (%)", 80, 99, 95, key="conf_mean")
            lo, hi, se = ci_mean_known_sigma(mean_input, sigma, int(n), conf/100)
            st.write(f"Error estándar = {se:.6f}")
            st.write(f"Intervalo de confianza {conf}% (z, σ conocido): [{lo:.6f}, {hi:.6f}]")
            st.write("Interpretación:")
            st.write("Si σ es conocido y el muestreo es aleatorio, el intervalo indica el rango plausible para la media poblacional con el nivel de confianza dado.")

    with colB:
        st.subheader("Intervalo de confianza y prueba para proporción")
        method = st.selectbox("Método para IC de proporción", ["wald","wilson"])
        successes = st.number_input("Número de éxitos", min_value=0, value=5, step=1)
        n_prop = st.number_input("Tamaño de la muestra n", min_value=1, value=20, step=1)
        conf_p = st.slider("Confianza (%)", 80, 99, 95, key="conf_prop")
        if successes > n_prop:
            st.error("Éxitos no puede ser mayor que n.")
        else:
            p_hat = successes / n_prop
            lo, hi, se = ci_prop(p_hat, int(n_prop), conf=conf_p/100, method=method)
            st.write(f"Proporción muestral p̂ = {p_hat:.6f}")
            st.write(f"Error estándar aproximado = {se:.6f}")
            st.write(f"IC {conf_p}% ({method}): [{lo:.6f}, {hi:.6f}]")
            st.write("Interpretación:")
            st.write("El intervalo proporciona un rango plausible para la proporción poblacional. Si se usa 'wald' con p̂ cercano a 0 o 1 o n pequeño, el intervalo puede ser poco preciso; Wilson es más robusto.")

    st.markdown("---")
    st.subheader("Prueba de hipótesis para la media (una muestra)")
    col1, col2 = st.columns(2)
    with col1:
        use_data_test = st.checkbox("Usar datos cargados para prueba de media", value=True, key="use_data_test")
        mu0 = st.number_input("Valor bajo H0 (μ0)", value=0.0, format="%.6f", key="mu0")
        alpha = st.slider("Alfa", 1, 10, 5)/100.0
        alt = st.selectbox("Hipótesis alternativa", ["two-sided","greater","less"])

        if use_data_test and "datos" in st.session_state:
            t_stat, pval = t_test_one_sample(np.array(st.session_state["datos"]), mu0)
            st.write(f"t = {t_stat:.6f}, p-valor (two-sided) = {pval:.6f}")
            if alt == "two-sided":
                p_adj = pval
            elif alt == "greater":
                # one-sided: P(T >= t) if mean > mu0
                df = len(st.session_state["datos"]) - 1
                if t_stat > 0:
                    p_adj = 1 - stats.t.cdf(t_stat, df)
                else:
                    p_adj = stats.t.cdf(t_stat, df)
            else: # less
                df = len(st.session_state["datos"]) - 1
                if t_stat < 0:
                    p_adj = 1 - stats.t.cdf(abs(t_stat), df)
                else:
                    p_adj = stats.t.cdf(t_stat, df)
            decision = "Rechazar H0" if p_adj < alpha else "No rechazar H0"
            st.write(f"P-valor ajustado para alternativa '{alt}': {p_adj:.6f}")
            st.write("Conclusión:", decision)
            st.write("Interpretación:")
            if decision.startswith("Rechazar"):
                st.write(f"Existe evidencia estadística (alfa={alpha}) para decir que la media poblacional difiere de {mu0} según la alternativa seleccionada.")
            else:
                st.write("No hay evidencia suficiente para rechazar H0 con el nivel de significancia dado.")
        else:
            st.info("Activa 'Usar datos cargados' y carga datos en la pestaña Datos o desactiva para ingresar valores manuales.")
    with col2:
        st.subheader("Prueba para proporción (una muestra)")
        p0 = st.number_input("Proporción bajo H0 (p0)", min_value=0.0, max_value=1.0, value=0.5, format="%.4f")
        succ = st.number_input("Número de éxitos", min_value=0, value=10, step=1, key="succ_test")
        n_test = st.number_input("n", min_value=1, value=50, step=1, key="n_test")
        alt_p = st.selectbox("Alternativa (proporción)", ["two-sided","greater","less"], key="alt_p")
        if succ > n_test:
            st.error("Éxitos mayor que n.")
        else:
            phat = succ / n_test
            se_pool = np.sqrt(p0*(1-p0)/n_test)
            z = (phat - p0) / se_pool
            if alt_p == "two-sided":
                pval = 2*(1 - stats.norm.cdf(abs(z)))
            elif alt_p == "greater":
                pval = 1 - stats.norm.cdf(z)
            else:
                pval = stats.norm.cdf(z)
            decision = "Rechazar H0" if pval < alpha else "No rechazar H0"
            st.write(f"p̂ = {phat:.6f}, z = {z:.6f}, p-valor = {pval:.6f}")
            st.write("Conclusión:", decision)
            st.write("Interpretación:")
            if decision.startswith("Rechazar"):
                st.write("La proporción observada es estadísticamente diferente a p0 según la alternativa y el alfa seleccionados.")
            else:
                st.write("No hay evidencia suficiente para afirmar diferencia respecto a p0.")

    st.markdown("---")
    st.subheader("Tamaño de muestra requerido")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Por media (σ conocido o usar estimado):")
        sigma = st.number_input("Desvío poblacional (σ)", value=1.0, format="%.6f", key="sigma_ss")
        E = st.number_input("Margen de error deseado (E)", value=0.5, format="%.6f", key="E_ss")
        conf_ss = st.slider("Confianza (%)", 80, 99, 95, key="conf_ss")
        if st.button("Calcular n (media)"):
            n_req = sample_size_mean(sigma, E, conf=conf_ss/100)
            st.write(f"Tamaño de muestra requerido: n = {n_req}")
            st.write("Interpretación: con este n y el sigma asumido, el margen de error máximo será E con el nivel de confianza solicitado.")
    with col2:
        st.write("Por proporción:")
        p_guess = st.number_input("Estimación de p (usar 0.5 para máxima varianza)", min_value=0.0, max_value=1.0, value=0.5, format="%.4f", key="p_guess")
        E_p = st.number_input("Margen de error (proporcional)", value=0.05, format="%.4f", key="E_p")
        conf_pn = st.slider("Confianza (%)", 80, 99, 95, key="conf_pn")
        if st.button("Calcular n (proporción)"):
            n_req_p = sample_size_prop(p_guess, E_p, conf=conf_pn/100)
            st.write(f"Tamaño de muestra requerido: n = {n_req_p}")
            st.write("Interpretación: este n garantiza el margen de error E para la proporción con la confianza dada bajo la suposición de p.")

# -----------------------
# PESTAÑA: Dos poblaciones
# -----------------------
with tabs[3]:
    st.header("Comparaciones entre dos poblaciones")
    st.write("Diferencia de medias, diferencia de proporciones y pruebas de hipótesis para dos muestras.")
    st.markdown("---")
    st.subheader("Diferencia de medias (dos muestras)")
    col1, col2 = st.columns(2)
    with col1:
        data_x_text = st.text_area("Muestra A (x):", value="", height=120)
        data_y_text = st.text_area("Muestra B (y):", value="", height=120)
        use_equal_var = st.checkbox("Asumir varianzas iguales (prueba t pooled)", value=False)
        conf_diff = st.slider("Confianza (%)", 80, 99, 95, key="conf_diff")
    with col2:
        if st.button("Calcular diferencia de medias"):
            try:
                x = np.array(parse_numeric_list(data_x_text))
                y = np.array(parse_numeric_list(data_y_text))
                if len(x) < 2 or len(y) < 2:
                    st.error("Cada muestra debe tener al menos 2 observaciones.")
                else:
                    t_stat, pval = two_sample_ttest(x, y, equal_var=use_equal_var)
                    diff, lo, hi, se_diff, df = diff_means_ci(x, y, conf=conf_diff/100, equal_var=use_equal_var)
                    st.write(f"Media A = {np.mean(x):.6f} (n={len(x)}), Media B = {np.mean(y):.6f} (n={len(y)})")
                    st.write(f"Diferencia (A - B) = {diff:.6f}")
                    st.write(f"IC {conf_diff}% para la diferencia: [{lo:.6f}, {hi:.6f}] (df aprox. = {df:.2f})")
                    st.write(f"t = {t_stat:.6f}, p-valor (two-sided) = {pval:.6f}")
                    st.write("Interpretación:")
                    if pval < 0.05:
                        st.write("Existe evidencia estadística de diferencia entre las medias de las dos poblaciones.")
                    else:
                        st.write("No hay evidencia suficiente para afirmar una diferencia en las medias con el nivel de significancia típico.")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    st.subheader("Diferencia de proporciones (dos muestras)")
    col1, col2 = st.columns(2)
    with col1:
        succ1 = st.number_input("Éxitos en muestra 1", min_value=0, value=20, step=1, key="s1")
        n1 = st.number_input("Tamaño n1", min_value=1, value=100, step=1, key="n1")
    with col2:
        succ2 = st.number_input("Éxitos en muestra 2", min_value=0, value=15, step=1, key="s2")
        n2 = st.number_input("Tamaño n2", min_value=1, value=120, step=1, key="n2")
    conf_p = st.slider("Confianza (%)", 80, 99, 95, key="conf_diff_prop")
    if st.button("Calcular diferencia de proporciones"):
        if succ1 > n1 or succ2 > n2:
            st.error("Éxitos no pueden exceder n.")
        else:
            diff, z, pval, lo, hi = diff_prop_test(succ1, int(n1), succ2, int(n2), conf=conf_p/100)
            st.write(f"p̂1 = {succ1/n1:.6f} (n1={n1}), p̂2 = {succ2/n2:.6f} (n2={n2})")
            st.write(f"Diferencia p̂1 - p̂2 = {diff:.6f}")
            st.write(f"IC {conf_p}% para la diferencia (z, no pool): [{lo:.6f}, {hi:.6f}]")
            st.write(f"z = {z:.6f}, p-valor (two-sided) = {pval:.6f}")
            st.write("Interpretación:")
            if pval < 0.05:
                st.write("Hay evidencia de diferencia entre las proporciones de las dos poblaciones.")
            else:
                st.write("No hay evidencia suficiente de diferencia entre las proporciones con el nivel de significación usual.")

# -----------------------
# PESTAÑA: Gráficos y distribuciones
# -----------------------
with tabs[4]:
    st.header("Gráficos y distribuciones")
    st.write("Visualizaciones para entender la distribución muestral, histogramas, QQ-plot y comportamiento del EE con n.")
    if "datos" not in st.session_state:
        st.warning("Carga datos en la pestaña Datos para usar las visualizaciones.")
    else:
        data = np.array(st.session_state["datos"])
        col1, col2 = st.columns([2,1])
        with col1:
            st.subheader("Histograma y KDE")
            fig, ax = plt.subplots(figsize=(6,3.5))
            sns.histplot(data, kde=True, ax=ax, stat="density", bins="auto", color="C0")
            ax.axvline(np.mean(data), color="red", linestyle="--", label=f"Media = {np.mean(data):.3f}")
            ax.legend()
            st.pyplot(fig)

            st.subheader("QQ-plot (normalidad)")
            fig2, ax2 = plt.subplots(figsize=(6,3.5))
            stats.probplot(data, dist="norm", plot=ax2)
            st.pyplot(fig2)

        with col2:
            st.subheader("Distribución muestral de la media (simulación)")
            n_sim = st.number_input("Tamaño de cada muestra para simular (n)", min_value=1, value=min(30, max(1, len(data))), step=1)
            n_rep = st.number_input("Número de réplicas", min_value=100, max_value=20000, value=2000, step=100)
            if st.button("Simular distribución muestral"):
                rng = np.random.default_rng()
                # Bootstrap-like or sampling with replacement from observed data
                sims = [np.mean(rng.choice(data, size=int(n_sim), replace=True)) for _ in range(int(n_rep))]
                sims = np.array(sims)
                fig3, ax3 = plt.subplots(figsize=(5,3.5))
                sns.histplot(sims, kde=True, ax=ax3, color="C2", bins="auto")
                ax3.axvline(np.mean(sims), color="black", linestyle="--", label=f"Mean sim = {np.mean(sims):.3f}")
                ax3.legend()
                st.pyplot(fig3)
                st.write(f"Media de las medias simuladas = {np.mean(sims):.6f}, SD de la distribución muestral = {np.std(sims, ddof=1):.6f}")
                st.write("Interpretación: la distribución de la media muestral tiende a la normal si n es suficientemente grande (TCL).")

        st.markdown("---")
        st.subheader("Cómo varía el Error Estándar (EE) con n")
        st.write("Seleccione una desviación estándar (σ) y vea cómo disminuye el EE al aumentar n (EE = σ / sqrt(n)).")
        sigma_sel = st.number_input("Valor de σ hipotético", value=float(np.std(data, ddof=1)), format="%.6f", key="sigma_plot")
        ns = np.arange(1, 2001)
        ees = sigma_sel / np.sqrt(ns)
        fig4, ax4 = plt.subplots(figsize=(8,2.5))
        ax4.plot(ns[:500], ees[:500])
        ax4.set_xlabel("n")
        ax4.set_ylabel("Error estándar (σ/√n)")
        ax4.set_title("EE vs n (primeros 500 valores)")
        st.pyplot(fig4)
        st.write("Interpretación: el EE decrece proporcional a 1/√n; duplicar la precisión requiere cuadruplicar n.")

# -----------------------
# PESTAÑA: Ejemplos / Interpretación
# -----------------------
with tabs[5]:
    st.header("Ejemplos y ejercicios resueltos")
    st.write("Algunos ejemplos pre-cargados para probar la aplicación. Seleccione un ejemplo para cargarlo en 'Datos'.")
    ex = st.selectbox("Elegir ejemplo:", ["-- seleccionar --","Ejemplo 1: alturas (simuladas)","Ejemplo 2: proporciones (éxitos)","Ejemplo 3: dos muestras"])
    if st.button("Cargar ejemplo"):
        if ex == "-- seleccionar --":
            st.info("Elija un ejemplo.")
        elif ex == "Ejemplo 1: alturas (simuladas)":
            rng = np.random.default_rng(123)
            alturas = (rng.normal(loc=170, scale=7, size=100)).round(2).tolist()
            st.session_state["datos"] = alturas
            st.success("Ejemplo de alturas cargado (n=100).")
            st.write("Interpretación corta: media ~170 cm, varianza moderada; se puede usar t o z según σ conocido.")
        elif ex == "Ejemplo 2: proporciones (éxitos)":
            # no cargamos datos numéricos, mostramos ejemplo de conteos
            st.session_state["datos"] = [1]*30 + [0]*70  # 30 éxitos en 100
            st.success("Ejemplo de proporciones cargado (30 éxitos en 100).")
            st.write("Interpretación corta: p̂ = 0.30; revisar IC y pruebas para proporciones.")
        elif ex == "Ejemplo 3: dos muestras":
            rng = np.random.default_rng(7)
            a = (rng.normal(loc=5, scale=1.2, size=50)).round(3).tolist()
            b = (rng.normal(loc=4.6, scale=1.5, size=55)).round(3).tolist()
            # concatenar con separación para que el usuario copie a ambas cajas si desea
            st.session_state["datos"] = a  # cargamos A por defecto
            st.success("Se cargó muestra A en 'Datos'. Copie B en la sección 'Dos poblaciones' si desea comparar.")
            st.write("Interpretación corta: medias cercanas; realice t-test (independiente) para evaluar diferencia estadística.")

    st.markdown("---")
    st.subheader("Sobre las interpretaciones")
    st.write("""
    - Las interpretaciones provistas son orientativas y su validez depende de los supuestos (muestreo aleatorio, independencia, distribución) que deben evaluarse antes de concluir causalidad o generalización.
    - Para tamaños pequeños se recomiendan métodos exactos o t en lugar de z cuando σ es desconocido.
    - Para proporciones con p cercano a 0 o 1 y n pequeño, Wilson o métodos exactos son preferibles a Wald.
    """)

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.write("App creada para facilitar cálculos estadísticos comunes y su interpretación. Modifique los ejemplos, pruebe con sus datos y revise las suposiciones antes de usar resultados en decisiones.")
