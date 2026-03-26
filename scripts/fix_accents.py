"""
fix_accents.py — Corrige tildes faltantes en texto visible de figuras
Solo modifica texto en contextos de visualización (títulos, ejes, leyendas, print, markdown),
NO modifica nombres de columnas de DataFrames ni variables.
"""
import json, re, pathlib

NB_DIR = pathlib.Path(__file__).resolve().parent.parent / "notebooks"

# ── Reemplazos SEGUROS: strings que solo aparecen en display text ──
SAFE_REPLACEMENTS = {
    # 07_LSTM
    "label='Validacion'":           "label='Validación'",
    "label='Validacion')":          "label='Validación')",
    "set_xlabel('Epoca'":           "set_xlabel('Época'",
    "label='Normal teorica'":       "label='Normal teórica'",
    "Entrenamiento (ultimos 18m)'": "Entrenamiento (últimos 18m)'",
    "LSTM - Validacion OOS":        "LSTM - Validación OOS",
    "Curva de Aprendizaje - Loss":  "Curva de Aprendizaje - Pérdida",
    "Curva de Aprendizaje - MAE":   "Curva de Aprendizaje - MAE",
    "LSTM — Pronostico de Produccion 2026": "LSTM — Pronóstico de Producción 2026",
    "LSTM — Pronostico 2026":       "LSTM — Pronóstico 2026",
    "label='LSTM Pronostico 2026'": "label='LSTM Pronóstico 2026'",
    "PRONOSTICO OUT-OF-SAMPLE":     "PRONÓSTICO OUT-OF-SAMPLE",
    "PRONOSTICO DE PRODUCCION":     "PRONÓSTICO DE PRODUCCIÓN",
    "PRONOSTICO MENSUAL":           "PRONÓSTICO MENSUAL",
    "Pronostico OOS guardado":      "Pronóstico OOS guardado",
    "Pronostico 2026 exportado":    "Pronóstico 2026 exportado",
    "Pronostico de Produccion":     "Pronóstico de Producción",
    "Modelo ACEPTABLE para pronostico": "Modelo ACEPTABLE para pronóstico",
    "Pronostico vs Real":           "Pronóstico vs Real",

    # 08_Comparacion
    "label='Historico (ultimos 12m)'": "label='Histórico (últimos 12m)'",
    "Distribucion de Residuos":      "Distribución de Residuos",
    "Historico (ultimos 12m)":       "Histórico (últimos 12m)",

    # 09_Benchmarking — textos de ejes y títulos de figuras
    "set_ylabel('Autocorrelacion'":          "set_ylabel('Autocorrelación'",
    "Autocorrelacion Lag-12":                "Autocorrelación Lag-12",
    "Distribucion CV Interanual por Tipologia": "Distribución CV Interanual por Tipología",
    "(Comparacion de Forma)":                "(Comparación de Forma)",
    "set_xlabel('Anio'":                     "set_xlabel('Año'",
    "Distribucion del Recaudo Anual por Tipologia y Anio": "Distribución del Recaudo Anual por Tipología y Año",
    "Autocorrelacion Lag-12 + Deteccion de Anomalias": "Autocorrelación Lag-12 + Detección de Anomalías",
    "AUTOCORRELACION Y ANOMALIAS":           "AUTOCORRELACIÓN Y ANOMALÍAS",
    "Fundamentacion Teorica":                "Fundamentación Teórica",
    "Analisis y Pronostico de Rentas":       "Análisis y Pronóstico de Rentas",
    "distribucion y giro de Rentas":         "distribución y giro de Rentas",
    "Distribucion y giro de Rentas":         "Distribución y giro de Rentas",
    "Reglamenta la distribucion":            "Reglamenta la distribución",
    "Validacion Orozco-Gallo":              "Validación Orozco-Gallo",
    "Validacion Santamaria":                "Validación Santamaría",
    "Distribucion CV Interanual":           "Distribución CV Interanual",
    "Distribucion Semaforo SAT":            "Distribución Semáforo SAT",
    "Distribucion R_Lag12":                 "Distribución R_Lag12",
    "Autocorrelacion de la serie":          "Autocorrelación de la serie",
    "autocorrelacion de la serie":          "autocorrelación de la serie",
    "patron de autocorrelacion":            "patrón de autocorrelación",
    "Correlacion estacional":               "Correlación estacional",
    "correlacion entre patrones":           "correlación entre patrones",
    "distribucion de Rentas Cedidas hacia": "distribución de Rentas Cedidas hacia",
    "autocorrelacion = mayor":              "autocorrelación = mayor",

    # 05_SARIMAX_2 / 04_SARIMAX — print text
    "'Pronostico':":  "'Pronóstico':",
}

# También: texto en celdas markdown donde 'Correlacion' aparece como referencia de notebook
MARKDOWN_REPLACEMENTS = {
    "## Fase VI — Autocorrelacion Lag-12 y Deteccion de Anomalias":
        "## Fase VI — Autocorrelación Lag-12 y Detección de Anomalías",
    "analisis de autocorrelacion": "análisis de autocorrelación",
}


def fix_notebook(path: pathlib.Path) -> int:
    nb = json.loads(path.read_text(encoding="utf-8"))
    fixes = 0
    for cell in nb.get("cells", []):
        src = cell.get("source", [])
        new_src = []
        for line in src:
            original = line
            # Apply safe replacements
            for old, new in SAFE_REPLACEMENTS.items():
                if old in line:
                    line = line.replace(old, new)
            # Apply markdown replacements for markdown cells
            if cell.get("cell_type") == "markdown":
                for old, new in MARKDOWN_REPLACEMENTS.items():
                    if old in line:
                        line = line.replace(old, new)
            if line != original:
                fixes += 1
            new_src.append(line)
        cell["source"] = new_src
    if fixes > 0:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    return fixes


notebooks = sorted(NB_DIR.glob("*.ipynb"))
total = 0
for nb_path in notebooks:
    n = fix_notebook(nb_path)
    tag = "✅" if n > 0 else "⏭️ "
    print(f"  {tag} {nb_path.name}: {n} líneas corregidas")
    total += n
print(f"\nTOTAL: {total} líneas con tildes corregidas en {len(notebooks)} notebooks")
