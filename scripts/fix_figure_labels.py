"""
fix_figure_labels.py — Parche automático de figuras en todos los notebooks
==========================================================================
Corrige:
  1. Títulos montados (suptitle y=1.01/1.02 → y=0.98, tight_layout con rect)
  2. Texto en inglés → español en etiquetas de ejes, títulos, leyendas
  3. Espaciado adecuado entre títulos y subgráficos
"""
import json
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
NB_DIR = PROJECT / "notebooks"

# ────────────────────────────────────────────────────────────
# 1. Traducciones de texto en inglés → español
# ────────────────────────────────────────────────────────────
TEXT_REPLACEMENTS = [
    # --- Ejes ---
    ("'Lag (meses)'",           "'Rezago (meses)'"),
    ("'Lag k (meses)'",        "'Rezago k (meses)'"),
    ('"Lag (meses)"',           '"Rezago (meses)"'),
    ("set_xlabel('Lags'",      "set_xlabel('Rezagos'"),
    ("set_xlabel('Lag'",       "set_xlabel('Rezago'"),

    # --- Títulos ACF/PACF ---
    ("'ACF — Serie Original'",           "'Autocorrelación (ACF) — Serie Original'"),
    ("'PACF — Serie Original'",          "'Autocorrelación Parcial (PACF) — Serie Original'"),
    ("'ACF — Primera Diferencia (d=1)'", "'Autocorrelación (ACF) — Primera Diferencia (d=1)'"),
    ("'PACF — Primera Diferencia (d=1)'","'Autocorrelación Parcial (PACF) — Primera Diferencia (d=1)'"),
    ("'ACF — Primera Diferencia'",       "'Autocorrelación (ACF) — Primera Diferencia'"),
    ("'PACF — Primera Diferencia'",      "'Autocorrelación Parcial (PACF) — Primera Diferencia'"),
    ("'ACF — Serie log1p'",              "'Autocorrelación (ACF) — Serie log1p'"),
    ("'PACF — Serie log1p'",             "'Autocorrelación Parcial (PACF) — Serie log1p'"),

    # --- Títulos generales ---
    ("'Q-Q Plot'",              "'Gráfico Q-Q'"),
    ("'QQ-Plot Residuos'",     "'Gráfico Q-Q de Residuos'"),
    ("'QQ Plot'",               "'Gráfico Q-Q'"),

    # --- Leyendas comunes ---
    ("label='Original'",                "label='Serie Original'"),
    ("label='Forecast'",                "label='Pronóstico'"),
    ("label='Training'",                "label='Entrenamiento'"),
    ("label='Test'",                    "label='Prueba'"),
    ("label='Residuals'",              "label='Residuos'"),
    ("label='Observed'",               "label='Observado'"),
    ("label='Trend'",                  "label='Tendencia'"),
    ("label='Threshold'",              "label='Umbral'"),
    ("label='Score'",                  "label='Puntaje'"),
    ("label='Upper bound'",           "label='Límite superior'"),
    ("label='Lower bound'",           "label='Límite inferior'"),
    ("label='Best fit'",              "label='Mejor ajuste'"),
    ("label='Actual'",                "label='Real'"),
    ("label='Predicted'",             "label='Predicho'"),
    ("label='Confidence interval'",   "label='Intervalo de confianza'"),

    # --- Ejes xlabel/ylabel ---
    ("set_xlabel('Date'",      "set_xlabel('Fecha'"),
    ("set_xlabel('Month'",     "set_xlabel('Mes'"),
    ("set_xlabel('Year'",      "set_xlabel('Año'"),
    ("set_xlabel('Epoch'",     "set_xlabel('Época'"),
    ("set_xlabel('Epochs'",    "set_xlabel('Épocas'"),
    ("set_xlabel('Iteration'", "set_xlabel('Iteración'"),
    ("set_xlabel('Feature'",   "set_xlabel('Variable'"),
    ("set_xlabel('Features'",  "set_xlabel('Variables'"),
    ("set_xlabel('Step'",      "set_xlabel('Paso'"),
    ("set_xlabel('Steps'",     "set_xlabel('Pasos'"),
    ("set_xlabel('Time'",      "set_xlabel('Tiempo'"),  
    ("set_ylabel('Loss'",      "set_ylabel('Pérdida'"),
    ("set_ylabel('Score'",     "set_ylabel('Puntaje'"),
    ("set_ylabel('Error'",     "set_ylabel('Error'"),
    ("set_ylabel('Value'",     "set_ylabel('Valor'"),
    ("set_ylabel('Count'",     "set_ylabel('Conteo'"),
    ("set_ylabel('Frequency'", "set_ylabel('Frecuencia'"),
    ("set_ylabel('Importance'","set_ylabel('Importancia'"),
    ("set_ylabel('Weight'",    "set_ylabel('Peso'"),
    ("set_ylabel('Forecast'",  "set_ylabel('Pronóstico'"),

    # --- Títulos de gráfico ---
    ("'Feature Importance'",           "'Importancia de Variables'"),
    ("'Learning Curve'",               "'Curva de Aprendizaje'"),
    ("'Learning Curves'",              "'Curvas de Aprendizaje'"),
    ("'Residual Analysis'",            "'Análisis de Residuos'"),
    ("'Forecast vs Actual'",           "'Pronóstico vs Real'"),
    ("'Training vs Validation Loss'",  "'Pérdida: Entrenamiento vs Validación'"),
    ("'Training Loss'",                "'Pérdida de Entrenamiento'"),
    ("'Validation Loss'",              "'Pérdida de Validación'"),
    ("'Out-of-Sample'",               "'Fuera de Muestra'"),

    # --- Heatmap xticklabels Lag ---
    ("f'Lag {i+1}'",   "f'Rezago {i+1}'"),
    ("f'Lag {i}'",     "f'Rezago {i}'"),
    ("f'Lag {k}'",     "f'Rezago {k}'"),
    ("'Lag '",         "'Rezago '"),

    # --- Anotaciones ---
    ("'Lag negativo = consumo lidera recaudo'",  "'Rezago negativo = consumo lidera recaudo'"),
    ("'Lag negativo = IPC lidera recaudo'",      "'Rezago negativo = IPC lidera recaudo'"),
    ("'Lag negativo = feature lidera target'",   "'Rezago negativo = variable lidera objetivo'"),
    ("'Lag óptimo'",                             "'Rezago óptimo'"),
    ("f'Lag óptimo",                             "f'Rezago óptimo"),

    # --- Misc ---
    ("'Real OOS'",      "'Real fuera de muestra'"),
    ("label='Real OOS'","label='Real (fuera de muestra)'"),
]

# ────────────────────────────────────────────────────────────
# 2. Regex para arreglar suptitle y tight_layout
# ────────────────────────────────────────────────────────────
# Patron: y=1.01 ó y=1.02 ó y=1.03 en suptitle
RE_SUPTITLE_Y = re.compile(r'(fig\.suptitle\([^)]*), *y *= *1\.0[0-9]([^)]*\))')
# Patron: plt.tight_layout() sin argumentos
RE_TIGHT_PLAIN = re.compile(r'plt\.tight_layout\(\)')


def has_suptitle(source_lines):
    """Detecta si el bloque de código contiene fig.suptitle()"""
    code = "".join(source_lines)
    return "fig.suptitle(" in code or "suptitle(" in code


def fix_cell_source(source_lines):
    """Aplica correcciones a la lista de líneas de una celda."""
    changed = False
    new_lines = []

    cell_has_suptitle = has_suptitle(source_lines)

    for line in source_lines:
        original = line

        # ── 2a. Fix suptitle y=1.0x → y=0.98
        if RE_SUPTITLE_Y.search(line):
            line = RE_SUPTITLE_Y.sub(r'\1, y=0.98\2', line)

        # ── 2b. Fix tight_layout() → tight_layout(rect=...) cuando hay suptitle
        if cell_has_suptitle and RE_TIGHT_PLAIN.search(line):
            line = RE_TIGHT_PLAIN.sub('plt.tight_layout(rect=[0, 0, 1, 0.95])', line)

        # ── 2c. Traducciones de texto
        for old, new in TEXT_REPLACEMENTS:
            if old in line:
                line = line.replace(old, new)

        if line != original:
            changed = True
        new_lines.append(line)

    return new_lines, changed


def process_notebook(nb_path):
    """Procesa un archivo .ipynb y corrige todas las celdas de código."""
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    total_fixed = 0
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        source = cell.get('source', [])
        if not source:
            continue

        new_source, changed = fix_cell_source(source)
        if changed:
            cell['source'] = new_source
            total_fixed += 1

    if total_fixed > 0:
        with open(nb_path, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print(f"  ✅ {nb_path.name}: {total_fixed} celdas corregidas")
    else:
        print(f"  ⏭️  {nb_path.name}: sin cambios necesarios")

    return total_fixed


def main():
    notebooks = sorted(NB_DIR.glob("*.ipynb"))
    print(f"\n{'='*60}")
    print(f"  PARCHE DE FIGURAS — {len(notebooks)} notebooks")
    print(f"{'='*60}\n")

    grand_total = 0
    for nb in notebooks:
        grand_total += process_notebook(nb)

    print(f"\n{'='*60}")
    print(f"  TOTAL: {grand_total} celdas corregidas en {len(notebooks)} notebooks")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
