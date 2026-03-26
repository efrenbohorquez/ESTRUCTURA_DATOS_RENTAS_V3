"""
run_all.py — Ejecuta todos los notebooks en orden y reporta resultado.
"""
import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"

NOTEBOOKS = [
    "01_EDA_Completo.ipynb",
    "02_Estacionalidad.ipynb",
    "03_Correlacion_Macro.ipynb",
    "04_SARIMAX.ipynb",
    "05_SARIMAX.ipynb",
    "05_SARIMAX_2_Alterno.ipynb",
    "05_Prophet.ipynb",
    "06_XGBoost.ipynb",
    "07_LSTM.ipynb",
    "08_Comparacion_Modelos.ipynb",
    "09_Benchmarking_Territorial.ipynb",
]

results = []

for nb in NOTEBOOKS:
    fp = NB_DIR / nb
    if not fp.exists():
        results.append((nb, "SKIP", 0, "no existe"))
        continue
    print(f"\n{'='*60}")
    print(f"  Ejecutando: {nb}")
    print(f"{'='*60}")
    t0 = time.time()
    proc = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=600",
            "--ExecutePreprocessor.kernel_name=python3",
            str(fp),
        ],
        capture_output=True,
        text=True,
        cwd=str(NB_DIR),
    )
    elapsed = time.time() - t0
    if proc.returncode == 0:
        results.append((nb, "OK", elapsed, ""))
        print(f"  ✓ OK ({elapsed:.0f}s)")
    else:
        # Extraer última línea de error
        err = proc.stderr.strip().split("\n")[-1] if proc.stderr else "unknown"
        results.append((nb, "FAIL", elapsed, err))
        print(f"  ✗ FAIL ({elapsed:.0f}s)")
        print(f"    {err[:200]}")

print(f"\n\n{'='*60}")
print(f"  RESUMEN DE EJECUCIÓN")
print(f"{'='*60}")
ok = sum(1 for _, s, _, _ in results if s == "OK")
fail = sum(1 for _, s, _, _ in results if s == "FAIL")
skip = sum(1 for _, s, _, _ in results if s == "SKIP")
for nb, status, t, err in results:
    icon = "✓" if status == "OK" else ("✗" if status == "FAIL" else "⊘")
    line = f"  {icon} {nb:<40} {status:>4}  {t:5.0f}s"
    if err:
        line += f"  [{err[:80]}]"
    print(line)
print(f"\n  Total: {ok} OK, {fail} FAIL, {skip} SKIP")
