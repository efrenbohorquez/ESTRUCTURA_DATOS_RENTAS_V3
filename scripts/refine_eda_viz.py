import nbformat as nbf
from pathlib import Path
import re

# Configuración de rutas
project_root = Path(r'C:\Users\efren\Music\ESTRUCTURA DATOS RENTAS')
notebook_path = project_root / 'notebooks' / '01_EDA_Completo.ipynb'

print(f"Iniciando refinamiento de: {notebook_path.name}")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

modified = False

for cell in nb.cells:
    if cell.cell_type == 'code':
        # Refinar Gráfico de Serie de Tiempo (General)
        if "plt.subplots(figsize=(15, 6))" in cell.source or "grafica_serie_tiempo" in cell.source:
            if "ticker" not in cell.source:
                cell.source += "\n# Refinamiento solicitado: Formato de pesos y grid\nimport matplotlib.ticker as ticker\nax = plt.gca()\nax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: formato_pesos(x)))\nplt.grid(True, alpha=0.3, linestyle='--')\n"
                modified = True
                print("  OK: Refinado grafico de serie temporal.")

        # Refinar Box Plot Estacional
        if "sns.boxplot" in cell.source and "Mes" in cell.source:
            # Asegurar orden de meses y paleta profesional
            if "order=" not in cell.source:
                cell.source = cell.source.replace(
                    "sns.boxplot(", 
                    "sns.boxplot(order=range(1, 13), palette='viridis', "
                )
                cell.source += "\nplt.title('Estacionalidad Mensual del Recaudo (Ene-Dic)', fontweight='bold')\nplt.xlabel('Mes')\n"
                modified = True
                print("  OK: Refinado boxplot estacional.")

# Agregar nueva sección para Procesamiento Multi-Frecuencia al final
new_md = nbf.v4.new_markdown_cell(
    "## 6. Procesamiento Multi-Frecuencia (Bimestral y Trimestral)\n"
    "Siguiendo los requerimientos tecnicos, generamos las series agregadas para comparar la estabilidad del error en horizontes mas amplios."
)

new_code = nbf.v4.new_code_cell(
    "from utils import agregar_bimestral, agregar_trimestral\n\n"
    "# 6.1 Generacion de series\n"
    "df_bim = agregar_bimestral(df)\n"
    "df_tri = agregar_trimestral(df)\n\n"
    "# 6.2 Visualizacion comparativa\n"
    "fig, axes = plt.subplots(2, 1, figsize=(15, 10))\n\n"
    "axes[0].plot(df_bim.index, df_bim['Recaudo_Neto'], marker='o', color='orange')\n"
    "axes[0].set_title('Serie Bimestral', fontweight='bold')\n\n"
    "axes[1].plot(df_tri.index, df_tri['Recaudo_Neto'], marker='s', color='green')\n"
    "axes[1].set_title('Serie Trimestral', fontweight='bold')\n\n"
    "for ax in axes:\n"
    "    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: formato_pesos(x)))\n"
    "    ax.grid(True, alpha=0.2)\n\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

nb.cells.append(new_md)
nb.cells.append(new_code)
modified = True

if modified:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Notebook actualizado exitosamente.")
else:
    print("No se realizaron cambios.")
