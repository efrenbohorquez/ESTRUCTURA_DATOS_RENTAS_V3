"""
fix_sarima_references.py — Elimina rastros de SARIMA puro del sistema
=======================================================================
SARIMA puro nunca se entrenó; solo SARIMAX (con exógenas IPC) existe.

Regla:
  - Referencias a "SARIMA" como modelo evaluado → eliminar o cambiar a SARIMAX
  - Referencias conceptuales a la familia ARIMA/SARIMA → cambiar a ARIMA/SARIMAX
  - Filas de tabla que listan SARIMA como modelo separado → eliminar
"""
import json, re, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
REPORTS = ROOT / "outputs" / "reports"

changes_log = []

def log(file, desc):
    changes_log.append(f"  {file}: {desc}")

# ═══════════════════════════════════════════════════════════════
# 1. NOTEBOOKS — Reemplazos en JSON
# ═══════════════════════════════════════════════════════════════
NB_REPLACEMENTS = {
    # 01_EDA_Completo.ipynb — tabla de implicaciones y referencias
    "ARIMA/SARIMA": "ARIMA/SARIMAX",
    "modelos SARIMA.": "modelos SARIMAX.",
    "modelos SARIMA ": "modelos SARIMAX ",
    "| Estacionalidad fuerte (s=12) | SARIMA, Prophet |":
        "| Estacionalidad fuerte (s=12) | SARIMAX, Prophet |",
    "| Tendencia + estacionalidad | SARIMA con $(D=1)$, Prophet (piecewise linear) |":
        "| Tendencia + estacionalidad | SARIMAX con $(D=1)$, Prophet (piecewise linear) |",
    "| Serie corta (51 obs) | SARIMA, Prophet (robustos con pocas observaciones) |":
        "| Serie corta (51 obs) | SARIMAX, Prophet (robustos con pocas observaciones) |",
    "Modelado SARIMA con selección automática de órdenes.":
        "Modelado SARIMAX con selección automática de órdenes.",

    # 02_Estacionalidad — sugerencia de orden
    "SARIMA/SARIMAX": "SARIMAX",
    "para SARIMA/SARIMAX": "para SARIMAX",
    "para SARIMA": "para SARIMAX",
    "Orden sugerido: SARIMA(": "Orden sugerido: SARIMAX(",
    "transformación para SARIMA": "transformación para SARIMAX",

    # 03_Correlacion — referencias metodológicas
    "STL/SARIMA la capturan": "STL/SARIMAX la capturan",
    "SARIMA(d": "SARIMAX(d",
    "SARIMA/SARIMAX": "SARIMAX",

    # 04_SARIMAX — referencia interna
    "modelos SARIMA.\\n": "modelos SARIMAX.\\n",

    # 06_XGBoost — nota de transformación
    "con SARIMA y Prophet": "con SARIMAX y Prophet",

    # 09_Benchmarking — referencia
    "modelos SARIMA/Prophet": "modelos SARIMAX/Prophet",
}


def fix_notebooks():
    total = 0
    for nb_path in sorted(NB_DIR.glob("*.ipynb")):
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        fixes = 0
        for cell in nb.get("cells", []):
            for i, line in enumerate(cell.get("source", [])):
                orig = line
                for old, new in NB_REPLACEMENTS.items():
                    if old in line:
                        line = line.replace(old, new)
                if line != orig:
                    cell["source"][i] = line
                    fixes += 1
        if fixes:
            nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1),
                               encoding="utf-8")
            log(nb_path.name, f"{fixes} líneas")
            total += fixes
    return total

# ═══════════════════════════════════════════════════════════════
# 2. README.md — Eliminar fila SARIMA de tabla de ranking
# ═══════════════════════════════════════════════════════════════
def fix_readme():
    fp = ROOT / "README.md"
    txt = fp.read_text(encoding="utf-8")
    # Eliminar fila SARIMA (que es duplicado de SARIMAX con métricas idénticas)
    old = "| SARIMA | 13.99 | 42.5 | 39.6 | 2.69 |\n"
    if old in txt:
        txt = txt.replace(old, "")
        # Renumerar: cambiar "los cinco paradigmas" → "los cuatro modelos"
        txt = txt.replace("los cinco paradigmas convergen", 
                          "los cuatro modelos convergen")
        fp.write_text(txt, encoding="utf-8")
        log("README.md", "Eliminada fila SARIMA de tabla de ranking")
        return 1
    return 0

# ═══════════════════════════════════════════════════════════════
# 3. AUDITORIA_SISTEMA_COMPLETO_2026.md
# ═══════════════════════════════════════════════════════════════
def fix_auditoria():
    fp = REPORTS / "AUDITORIA_SISTEMA_COMPLETO_2026.md"
    if not fp.exists():
        return 0
    txt = fp.read_text(encoding="utf-8")
    count = 0

    # Eliminar fila SARIMA de tabla de desempeño
    old = "| — | SARIMA | 13.99% | 42.5 | 39.6 | 2.46% | -6.8 | ⚠️ Limitado |\n"
    if old in txt:
        txt = txt.replace(old, "")
        count += 1

    # Eliminar referencia a sarima_metricas.csv inexistente
    old2 = "│   ├─ sarima_metricas.csv               ✅ Config + params\n"
    if old2 in txt:
        txt = txt.replace(old2, "")
        count += 1

    # Corregir nota sobre SARIMAX vs SARIMA
    old3 = "   - SARIMAX y SARIMA = MAPE idéntico 13.99% (variable exógena no contribuyó)"
    new3 = "   - SARIMAX MAPE 13.99% — variable exógena IPC no mejoró significativamente el error"
    if old3 in txt:
        txt = txt.replace(old3, new3)
        count += 1

    # Cambiar "5 modelos registrados" → "4 modelos registrados"
    txt = txt.replace("5 modelos registrados", "4 modelos registrados")

    if count:
        fp.write_text(txt, encoding="utf-8")
        log("AUDITORIA_SISTEMA_COMPLETO_2026.md", f"{count} correcciones")
    return count

# ═══════════════════════════════════════════════════════════════
# 4. DASHBOARDS_AUDITORIA_VISUAL.md
# ═══════════════════════════════════════════════════════════════
def fix_dashboards():
    fp = REPORTS / "DASHBOARDS_AUDITORIA_VISUAL.md"
    if not fp.exists():
        return 0
    txt = fp.read_text(encoding="utf-8")
    count = 0

    # Eliminar fila SARIMA duplicada
    old = "║  SARIMA     13.99%   $42.5MM   $39.6MM    2.46%   17.09   ⚠️ RESPALDO ║\n"
    if old in txt:
        txt = txt.replace(old, "")
        count += 1

    # Cambiar SARIMAX/SARIMA → SARIMAX
    old2 = "SARIMAX/SARIMA"
    if old2 in txt:
        txt = txt.replace(old2, "SARIMAX")
        count += 1

    if count:
        fp.write_text(txt, encoding="utf-8")
        log("DASHBOARDS_AUDITORIA_VISUAL.md", f"{count} correcciones")
    return count

# ═══════════════════════════════════════════════════════════════
# 5. evaluacion_cualitativa.csv — Eliminar fila SARIMA
# ═══════════════════════════════════════════════════════════════
def fix_evaluacion():
    fp = REPORTS / "evaluacion_cualitativa.csv"
    if not fp.exists():
        return 0
    txt = fp.read_text(encoding="utf-8")
    old = "SARIMA,Econométrico,⭐⭐⭐⭐⭐,No,Baja (≥24 obs),Baja,Alta\n"
    if old in txt:
        txt = txt.replace(old, "")
        fp.write_text(txt, encoding="utf-8")
        log("evaluacion_cualitativa.csv", "Eliminada fila SARIMA puro")
        return 1
    return 0

# ═══════════════════════════════════════════════════════════════
# 6. Reportes temáticos — Actualizar SARIMA → SARIMAX
# ═══════════════════════════════════════════════════════════════
REPORT_REPLACEMENTS = {
    # Solo en contexto de modelo evaluado, no concepto genérico
    "SARIMA puro": "SARIMAX",
    "SARIMA como modelo base": "SARIMAX como modelo base",
    "Modelo SARIMA (Notebook 04)": "Modelo SARIMAX (Notebook 04)",
    "SARIMA es apropiado": "SARIMAX es apropiado",
    "SARIMA como modelo base |": "SARIMAX como modelo base |",
    "modelos puramente estacionales (SARIMA)": "modelos estacionales con exógenas (SARIMAX)",
    "modelos **estacionales puros** (SARIMA) como línea base": 
        "modelos **estacionales con exógenas** (SARIMAX) como línea base",
    "Un modelo SARIMA(p,d,q)": "Un modelo SARIMAX(p,d,q)",
    "Diagnóstico para Modelado SARIMA": "Diagnóstico para Modelado SARIMAX",
    "órdenes del modelo SARIMA(": "órdenes del modelo SARIMAX(",
    "SARIMA(1,1,1)(1,1,1)[12]": "SARIMAX(1,1,1)(1,1,1)[12]",
    "Comparar SARIMA(1,1,1)": "Comparar SARIMAX(1,1,1)",
    "(2,1,1)(1,1,1)[12] y (1,1,2)": "(2,1,1)(1,1,1)[12] y (1,1,2)",
    "Alternativa: SARIMA con s=12": "Alternativa: SARIMAX con s=12",
    "ajustar el SARIMA": "ajustar el SARIMAX",
    "SARIMA(p,d,q)": "SARIMAX(p,d,q)",
    "Después de SARIMA": "Después de SARIMAX",
    "varianza antes de SARIMA": "varianza antes de SARIMAX",
    "| **SARIMA**": "| **SARIMAX**",
}

def fix_reports():
    total = 0
    for md_path in sorted(REPORTS.glob("*.md")):
        txt = md_path.read_text(encoding="utf-8")
        orig = txt
        for old, new in REPORT_REPLACEMENTS.items():
            if old in txt:
                txt = txt.replace(old, new)
        if txt != orig:
            md_path.write_text(txt, encoding="utf-8")
            log(md_path.name, "Referencias SARIMA→SARIMAX")
            total += 1
    return total


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  LIMPIEZA SARIMA PURO — Solo SARIMAX en producción")
    print("=" * 60 + "\n")

    t = 0
    t += fix_notebooks()
    t += fix_readme()
    t += fix_auditoria()
    t += fix_dashboards()
    t += fix_evaluacion()
    t += fix_reports()

    print("\n".join(changes_log))
    print(f"\n{'=' * 60}")
    print(f"  TOTAL: {t} correcciones aplicadas")
    print("=" * 60)
