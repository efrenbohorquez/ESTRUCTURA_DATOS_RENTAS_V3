"""
fix_sarima_pass2.py — Segunda pasada: build scripts, reports restantes, dashboard
"""
import pathlib, re

ROOT = pathlib.Path(__file__).resolve().parent.parent
changes = []

def fix_file(fp, replacements):
    """Aplica un dict {old: new} sobre el archivo. Retorna #cambios."""
    txt = fp.read_text(encoding="utf-8")
    orig = txt
    for old, new in replacements.items():
        txt = txt.replace(old, new)
    if txt != orig:
        fp.write_text(txt, encoding="utf-8")
        changes.append(fp.name)
        return 1
    return 0

SCRIPTS = ROOT / "scripts"
REPORTS = ROOT / "outputs" / "reports"
NB = ROOT / "notebooks"

# ═══════════════════════════════════════════════════════════════
# 1. build scripts — mirror cambios en notebooks
# ═══════════════════════════════════════════════════════════════
fix_file(SCRIPTS / "build_01_eda.py", {
    "ARIMA/SARIMA": "ARIMA/SARIMAX",
    "modelos SARIMA.": "modelos SARIMAX.",
    "| Estacionalidad fuerte (s=12) | SARIMA, Prophet |":
        "| Estacionalidad fuerte (s=12) | SARIMAX, Prophet |",
    "| Tendencia + estacionalidad | SARIMA con $(D=1)$, Prophet (piecewise linear) |":
        "| Tendencia + estacionalidad | SARIMAX con $(D=1)$, Prophet (piecewise linear) |",
    "| Serie corta (51 obs) | SARIMA, Prophet (robustos con pocas observaciones) |":
        "| Serie corta (51 obs) | SARIMAX, Prophet (robustos con pocas observaciones) |",
    "Modelado SARIMA con selección automática de órdenes.":
        "Modelado SARIMAX con selección automática de órdenes.",
})

fix_file(SCRIPTS / "build_02_estacionalidad.py", {
    "para SARIMA/SARIMAX": "para SARIMAX",
    "para SARIMA\"": "para SARIMAX\"",
    "Orden sugerido: SARIMA(": "Orden sugerido: SARIMAX(",
    "transformación para SARIMA": "transformación para SARIMAX",
    "SARIMA/SARIMAX": "SARIMAX",
})

fix_file(SCRIPTS / "build_04_sarimax.py", {
    "modelos SARIMA.": "modelos SARIMAX.",
})

fix_file(SCRIPTS / "build_06_xgboost.py", {
    "con SARIMA y Prophet": "con SARIMAX y Prophet",
})

fix_file(SCRIPTS / "build_09_benchmarking.py", {
    "modelos SARIMA/Prophet": "modelos SARIMAX/Prophet",
})

# ═══════════════════════════════════════════════════════════════
# 2. Reports restantes
# ═══════════════════════════════════════════════════════════════
fix_file(REPORTS / "explicacion_grafica_serie_tiempo.md", {
    "los modelos SARIMA asumen": "los modelos SARIMAX asumen",
    "Usar SARIMA con componente estacional": "Usar SARIMAX con componente estacional",
    "transformación logarítmica para SARIMA": "transformación logarítmica para SARIMAX",
})

fp_indice = REPORTS / "INDICE_MAESTRO_AUDITORIA.md"
if fp_indice.exists():
    fix_file(fp_indice, {
        "SARIMA          13.99%   ⚠️ RESPALDO\n": "",
        "SARIMA/SARIMAX": "SARIMAX",
    })

fp_macro = REPORTS / "informe_entorno_macroeconomico.md"
if fp_macro.exists():
    fix_file(fp_macro, {
        "modelos SARIMA, SARIMAX,": "modelos SARIMAX,",
    })

# ═══════════════════════════════════════════════════════════════
# 3. dashboard_rentas.py — SARIMA → SARIMAX en selección
# ═══════════════════════════════════════════════════════════════
fix_file(SCRIPTS / "dashboard_rentas.py", {
    '"SARIMA"': '"SARIMAX"',
    'Parámetros SARIMA': 'Parámetros SARIMAX',
    '# SARIMA': '# SARIMAX',
    'else: # SARIMA': 'else: # SARIMAX',
})

# ═══════════════════════════════════════════════════════════════
# 4. 00_config.py — Clave 'sarima' → 'sarimax' (ya existe sarimax,
#    pero hay dos; eliminamos la de sarima puro)
# ═══════════════════════════════════════════════════════════════
cfg = NB / "00_config.py"
txt = cfg.read_text(encoding="utf-8")
# Solo eliminamos la clave 'sarima' si hay duplicado con 'sarimax'
if "'sarima'" in txt and "'sarimax'" in txt:
    txt = txt.replace("'sarima': C_SECONDARY, ", "")
    cfg.write_text(txt, encoding="utf-8")
    changes.append("00_config.py")

print("\n=== SARIMA Pass 2 ===")
for c in changes:
    print(f"  ✓ {c}")
print(f"\n  Total: {len(changes)} archivos corregidos")
