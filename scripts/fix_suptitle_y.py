"""
fix_suptitle_y.py — Corrige posición y de suptitle en llamadas multilínea
Reemplaza y=1.01 y y=1.02 por y=0.98 en todos los notebooks
"""
import json, pathlib

NB_DIR = pathlib.Path(__file__).resolve().parent.parent / "notebooks"

REPLACEMENTS = [
    ("y=1.01)",  "y=0.98)"),
    ("y=1.01,",  "y=0.98,"),
    ("y=1.02)",  "y=0.98)"),
    ("y=1.02,",  "y=0.98,"),
    (" y=1.01 ", " y=0.98 "),
    (" y=1.02 ", " y=0.98 "),
]

total = 0
for nb_path in sorted(NB_DIR.glob("*.ipynb")):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    fixes = 0
    for cell in nb.get("cells", []):
        for i, line in enumerate(cell.get("source", [])):
            orig = line
            for old, new in REPLACEMENTS:
                if old in line:
                    line = line.replace(old, new)
            if line != orig:
                cell["source"][i] = line
                fixes += 1
    if fixes:
        nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  ✅ {nb_path.name}: {fixes} líneas")
    total += fixes
print(f"\nTOTAL: {total} líneas con y=1.0x → y=0.98")
