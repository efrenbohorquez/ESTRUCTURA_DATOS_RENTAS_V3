import pandas as pd
from pathlib import Path

# Paths
FORECAST_DIR = Path(r'C:\Users\efren\Music\ESTRUCTURA DATOS RENTAS\outputs\forecasts')

def fix_csv(filename):
    path = FORECAST_DIR / filename
    if path.exists():
        print(f"Fixing {filename}...")
        df = pd.read_csv(path)
        if 'FechaRecaudo' in df.columns:
            df = df.rename(columns={'FechaRecaudo': 'Fecha'})
            df.to_csv(path, index=False)
            print(f"  ✅ Renamed 'FechaRecaudo' to 'Fecha' in {filename}")
        else:
            print(f"  ℹ️ 'FechaRecaudo' not found in {filename}")

fix_csv('sarima_forecast.csv')
fix_csv('sarimax_forecast.csv')
fix_csv('sarima_futuro.csv')
fix_csv('sarima_produccion_12m.csv')

print("\nDone fixing CSV files.")
