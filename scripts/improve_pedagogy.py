"""
improve_pedagogy.py — Inyecta celdas markdown interpretativas y corrige numeración interna
=====================================
Notebooks afectados:
  02_Estacionalidad      – 4 celdas interpretativas entre code cells consecutivos
  05_Prophet             – Definición formal del modelo + fix numeración 06→05
  09_Benchmarking        – 8 celdas interpretativas entre code cells consecutivos
  05_SARIMAX, 06_XGBoost, 07_LSTM, 08_Comparacion – Fix numeración interna (+1)
"""
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"

def md_cell(source_lines):
    """Crea una celda markdown con las líneas dadas."""
    if isinstance(source_lines, str):
        source_lines = [l + "\n" for l in source_lines.split("\n")]
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines
    }

def load_nb(name):
    fp = NB_DIR / name
    return fp, json.loads(fp.read_text(encoding="utf-8"))

def save_nb(fp, nb):
    fp.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

changes = []

# ===================================================================
# 1. 02_Estacionalidad — Insertar celdas interpretativas
# ===================================================================
fp, nb = load_nb("02_Estacionalidad.ipynb")
cells = nb["cells"]

# Celdas a insertar DESPUÉS de ciertos code cells (por contenido para robustez)
interp_02 = [
    # Después de FASE I.1 (CCF) y antes de FASE I.2 (Heatmap)
    (4, md_cell(
        "### Interpretación — Correlación Cruzada (CCF)\n"
        "\n"
        "La **Cross-Correlation Function (CCF)** mide la correlación entre la serie de recaudo y cada variable macroeconómica en diferentes *lags*.\n"
        "\n"
        "- Un pico significativo en $\\text{lag} = k$ indica que la variable exógena *adelanta* al recaudo por $k$ meses.\n"
        "- Esto permite identificar **indicadores líderes** útiles como regresores en SARIMAX y XGBoost.\n"
        "- Los intervalos de Bartlett (bandas azules) marcan el umbral de significancia al 95%."
    )),
    # Después de FASE II.1 (STL) y antes de FASE II.2 (Heterocedasticidad)
    (7, md_cell(
        "### Interpretación — Descomposición STL\n"
        "\n"
        "La descomposición **STL** (*Seasonal and Trend decomposition using Loess*; Cleveland et al., 1990) separa la serie en tres componentes:\n"
        "\n"
        "$$Y_t = T_t + S_t + R_t$$\n"
        "\n"
        "| Componente | Qué revela | Implicación |\n"
        "|-----------|-----------|-------------|\n"
        "| $T_t$ (tendencia) | Dirección de largo plazo del recaudo | Define $d$ y $D$ en SARIMAX |\n"
        "| $S_t$ (estacionalidad) | Ciclo anual Ene–Jul con picos fijos | Confirma $s=12$ para modelos estacionales |\n"
        "| $R_t$ (residuo) | Variabilidad no explicada | Si es homocedástico, SARIMAX es apropiado; si no, se requiere log1p |"
    )),
    # Después de FASE III.1 (Licores) y antes de FASE III.2 (Juegos Azar)
    (10, md_cell(
        "### Interpretación — Dinámica Sectorial: Licores\n"
        "\n"
        "La serie **deflactada** de licores (dividida por IPC) revela el crecimiento **real** del recaudo eliminando el efecto inflacionario.\n"
        "\n"
        "- Si la tendencia real es plana pero la nominal crece, el aumento se explica solo por inflación.\n"
        "- La **elasticidad** mide cuánto varía el recaudo ante cambios en el IPC: $\\beta > 1$ indica que el recaudo amplifica los movimientos de precios."
    )),
    # Después de FASE IV.1 (Change Points) y antes de FASE IV.2 (Valores Negativos)
    (13, md_cell(
        "### Interpretación — Detección de Quiebres Estructurales\n"
        "\n"
        "El test **CUSUM** detecta puntos donde la media de la serie cambia abruptamente.\n"
        "\n"
        "- Los quiebres identificados corresponden a **eventos reales**: cambios regulatorios, efectos residuales del COVID-19, o reformas tributarias.\n"
        "- Los tests de **Welch** y **Levene** verifican si la media y la varianza difieren significativamente antes y después del quiebre.\n"
        "- Estos puntos deben excluirse del entrenamiento o incorporarse como *dummies* en SARIMAX."
    )),
]

# Insertar en orden inverso para no alterar índices
offset = 0
for after_idx, cell in interp_02:
    cells.insert(after_idx + offset, cell)
    offset += 1

save_nb(fp, nb)
changes.append(f"02_Estacionalidad: +{len(interp_02)} celdas interpretativas")

# ===================================================================
# 2. 05_Prophet — Definición formal + fix numeración
# ===================================================================
fp, nb = load_nb("05_Prophet.ipynb")
cells = nb["cells"]

# Fix numeración: "# 06 —" → "# 05 —"
for cell in cells:
    if cell["cell_type"] == "markdown":
        for i, line in enumerate(cell["source"]):
            if "# 06 —" in line:
                cell["source"][i] = line.replace("# 06 —", "# 05 —")

# Insertar definición formal del modelo Prophet después de la celda 0 (intro)
prophet_formal = md_cell(
    "## Definición Formal del Modelo\n"
    "\n"
    "**Prophet** (Taylor & Letham, 2018) es un modelo aditivo bayesiano diseñado para series temporales con estacionalidades fuertes:\n"
    "\n"
    "$$y(t) = g(t) + s(t) + h(t) + \\varepsilon_t$$\n"
    "\n"
    "| Componente | Significado | En nuestro contexto |\n"
    "|-----------|-----------|--------------------|\n"
    "| $g(t)$ | Tendencia (*piecewise linear* o logística) | Crecimiento sostenido del recaudo post-pandemia |\n"
    "| $s(t)$ | Estacionalidad (series de Fourier) | Ciclo anual Ene→Jul con picos de recaudo mes vencido |\n"
    "| $h(t)$ | Efecto de festivos/eventos especiales | Festivos colombianos (Ley 51/83 que traslada festivos) |\n"
    "| $\\varepsilon_t$ | Error irreducible | Variabilidad no capturada por los componentes anteriores |\n"
    "\n"
    "**Ventajas clave para esta aplicación:**\n"
    "1. **Robusto con datos escasos** — funciona bien con 36–50 observaciones mensuales\n"
    "2. **Detección automática de changepoints** — identifica cambios de régimen sin intervención manual\n"
    "3. **Incorpora conocimiento del dominio** — permite especificar festivos y topes de capacidad\n"
    "\n"
    "> **Referencia:** Taylor, S.J. & Letham, B. (2018). *Forecasting at Scale*. The American Statistician, 72(1), 37–45."
)

cells.insert(1, prophet_formal)
save_nb(fp, nb)
changes.append("05_Prophet: +1 celda definición formal + fix numeración 06→05")

# ===================================================================
# 3. 09_Benchmarking — Insertar celdas interpretativas
# ===================================================================
fp, nb = load_nb("09_Benchmarking_Territorial.ipynb")
cells = nb["cells"]

# Fix numeración: "# 10 —" → "# 09 —"
for cell in cells:
    if cell["cell_type"] == "markdown":
        for i, line in enumerate(cell["source"]):
            if "# 10 —" in line:
                cell["source"][i] = line.replace("# 10 —", "# 09 —")

interp_09 = [
    # Después de Fase I imports (idx 1) y antes de carga dataset (idx 2)
    (2, md_cell(
        "### Fase I — Carga y Preparación del Dataset Territorial\n"
        "\n"
        "Se carga el dataset granular de recaudo por departamento y concepto. Este contiene las liquidaciones individuales que permiten analizar la **distribución territorial** del recaudo de rentas cedidas a nivel departamental.\n"
        "\n"
        "La granularidad departamental es fundamental para identificar asRIMETRÍAS ESTRUCTURALES en el sistema de transferencias fiscales colombiano."
    )),
    # Después de Gini/Pareto cálculo (idx 4→5) y antes de visualización Lorenz (idx 5→6)
    (6, md_cell(
        "### Interpretación — Concentración Fiscal\n"
        "\n"
        "El **coeficiente de Gini** mide la desigualdad en la distribución del recaudo:\n"
        "\n"
        "- $G = 0$: distribución perfectamente equitativa entre departamentos\n"
        "- $G = 1$: todo el recaudo concentrado en un solo departamento\n"
        "\n"
        "La **curva de Lorenz** visualiza esta concentración: cuanto más se aleja de la diagonal, mayor es la inequidad. El **principio de Pareto** (80/20) indica qué proporción del recaudo generan los pocos departamentos de mayor tamaño."
    )),
    # Después de K-Means (idx 8→9+1) y antes de scatter (idx 9→10+1)
    (10, md_cell(
        "### Interpretación — Tipologías K-Means\n"
        "\n"
        "El clustering **K-Means** ($k=4$) agrupa departamentos según sus patrones de recaudo, creando tipologías que reflejan realidades fiscales diferentes:\n"
        "\n"
        "| Tipología | Característica | Ejemplo |\n"
        "|-----------|---------------|--------|\n"
        "| **Tipo A** — Consolidado | Alto recaudo, baja variabilidad | Antioquia, Valle, Bogotá |\n"
        "| **Tipo B** — Emergente | Recaudo medio, crecimiento sostenido | Santander, Atlántico |\n"
        "| **Tipo C** — Volátil | Recaudo medio, alta variabilidad | Departamentos con economías extractivas |\n"
        "| **Tipo D** — Vulnerable | Bajo recaudo, alta dependencia | Chocó, Vaupés, Guainía |\n"
        "\n"
        "El **CV interanual** (no mensual) es la métrica de variabilidad correcta, ya que elimina la estacionalidad intra-año."
    )),
    # Después de Asimetría Bogotá vs Chocó cálculo (idx 11→13) y antes de visualización (idx 12→14)
    (14, md_cell(
        "### Interpretación — Asimetría Estructural\n"
        "\n"
        "La brecha entre el departamento de mayor recaudo (Bogotá) y el de menor recaudo (Chocó) revela la **asimetría estructural** del sistema fiscal colombiano.\n"
        "\n"
        "- La brecha **per cápita** corrige por población, permitiendo una comparación justa\n"
        "- Una brecha creciente sugiere que las políticas de redistribución no están compensando las diferencias de base económica\n"
        "- Este indicador es clave para la evaluación de la política de **rentas cedidas** como mecanismo de equidad territorial"
    )),
    # Después de deflactación (idx 14→17) y antes de visualización nominal vs real (idx 15→18)
    (18, md_cell(
        "### Interpretación — Deflactación y Valores Reales\n"
        "\n"
        "La **deflactación** por IPC permite distinguir entre crecimiento **nominal** (incluye inflación) y crecimiento **real** (poder adquisitivo efectivo):\n"
        "\n"
        "$$\\text{Valor Real}_t = \\frac{\\text{Valor Nominal}_t}{\\text{IPC}_t / \\text{IPC}_{\\text{base}}}$$\n"
        "\n"
        "Si el recaudo nominal crece pero el real se mantiene estable, el crecimiento es ilusorio — solo refleja inflación."
    )),
    # Después de ACF cálculo (idx 17→21) y antes de visualización ACF (idx 18→22)
    (22, md_cell(
        "### Interpretación — Autocorrelación Lag-12\n"
        "\n"
        "La autocorrelación en **lag 12** mide cuánto se parece el recaudo de cada mes al del mismo mes del año anterior.\n"
        "\n"
        "- $\\rho_{12} > 0.7$: Estacionalidad fuerte → modelos SARIMAX/Prophet capturan bien este patrón\n"
        "- $\\rho_{12} < 0.3$: Estacionalidad débil → se necesitan features de ML (XGBoost)\n"
        "- **Anomalías**: Departamentos donde $\\rho_{12}$ es atípicamente bajo pueden tener eventos disruptivos no modelados"
    )),
    # Después de SAT cálculo (idx 20→25) y antes de visualización SAT (idx 21→26)
    (26, md_cell(
        "### Interpretación — Sistema de Alertas Tempranas (SAT-STAR)\n"
        "\n"
        "El SAT combina dos indicadores propios:\n"
        "\n"
        "- **IEP** (Índice de Estrés Presupuestario): detecta desviaciones del recaudo respecto a la meta\n"
        "- **ERS** (Estabilidad del Recaudo Subyacente): mide la volatilidad estructural\n"
        "\n"
        "El **semáforo adaptativo** clasifica cada departamento en:\n"
        "- 🟢 Verde: recaudo estable, sin alertas\n"
        "- 🟡 Amarillo: desviación moderada, requiere monitoreo\n"
        "- 🔴 Rojo: alerta crítica, posible colapso del recaudo"
    )),
    # Después de Mapa calor (idx 23→29) y antes de informe narrativo (idx 24→30)
    (30, md_cell(
        "### Interpretación — Mapa de Calor y Box-Plots\n"
        "\n"
        "El **mapa de calor** permite visualizar simultáneamente el comportamiento temporal de todos los departamentos:\n"
        "\n"
        "- Columnas = meses; filas = departamentos\n"
        "- Colores cálidos = meses de alto recaudo; fríos = bajo recaudo\n"
        "- Patrones verticales = estacionalidad compartida; horizontales = diferencias estructurales\n"
        "\n"
        "Los **box-plots** por departamento revelan la dispersión y los outliers de cada entidad territorial."
    )),
]

offset = 0
for after_idx, cell in interp_09:
    cells.insert(after_idx + offset, cell)
    offset += 1

save_nb(fp, nb)
changes.append(f"09_Benchmarking: +{len(interp_09)} celdas interpretativas + fix numeración 10→09")

# ===================================================================
# 4. Fix numeración interna en notebooks restantes
# ===================================================================
NUM_FIXES = {
    "04_SARIMAX.ipynb": ("# 04 —", None),        # Ya está correcto
    "05_SARIMAX.ipynb": ("# 05 —", None),          # Ya está correcto
    "05_SARIMAX_2_Alterno.ipynb": ("# 05 —", None),  # Ya está correcto
    "06_XGBoost.ipynb": ("# 07 —", "# 06 —"),
    "07_LSTM.ipynb": ("# 08 —", "# 07 —"),
    "08_Comparacion_Modelos.ipynb": ("# 09 —", "# 08 —"),
}

for name, (expected, fix_to) in NUM_FIXES.items():
    if fix_to is None:
        continue
    fpath = NB_DIR / name
    if not fpath.exists():
        continue
    fp, nb = load_nb(name)
    fixed = False
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            for i, line in enumerate(cell["source"]):
                if expected in line:
                    cell["source"][i] = line.replace(expected, fix_to)
                    fixed = True
    if fixed:
        save_nb(fp, nb)
        changes.append(f"{name}: fix numeración {expected} → {fix_to}")

# ===================================================================
# REPORTE
# ===================================================================
print("\n" + "=" * 60)
print("  MEJORA PEDAGÓGICA — Celdas interpretativas + numeración")
print("=" * 60 + "\n")
for c in changes:
    print(f"  ✓ {c}")
print(f"\n  Total: {len(changes)} notebooks modificados")
