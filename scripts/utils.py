"""
utils.py — Funciones Reutilizables para el Sistema de Análisis de Rentas Cedidas
=================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.stattools import adfuller, kpss
import sys
from pathlib import Path

# Agregar notebooks/ al path para importar config
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "notebooks"))
try:
    from importlib.machinery import SourceFileLoader
    config = SourceFileLoader("config", str(_project_root / "notebooks" / "00_config.py")).load_module()
except Exception:
    config = None


# ============================================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ============================================================

def cargar_datos(ruta=None):
    """
    Carga el archivo Excel principal y retorna el DataFrame crudo.
    
    Parameters
    ----------
    ruta : str or Path, optional
        Ruta al archivo Excel. Si None, usa la ruta de config.
    
    Returns
    -------
    pd.DataFrame
    """
    if ruta is None:
        if config:
            ruta = config.DATA_FILE
        else:
            raise ValueError("No se proporcionó ruta y no se pudo cargar config.")
    
    df = pd.read_excel(ruta)
    
    # Validaciones básicas
    col_fecha = 'FechaRecaudo'
    col_valor = 'ValorRecaudo'
    
    assert col_fecha in df.columns, f"Columna '{col_fecha}' no encontrada en el archivo."
    assert col_valor in df.columns, f"Columna '{col_valor}' no encontrada en el archivo."
    
    df[col_fecha] = pd.to_datetime(df[col_fecha])
    df[col_valor] = pd.to_numeric(df[col_valor], errors='coerce')
    
    # Nota: Los valores negativos se CONSERVAN — representan anulaciones y
    # ajustes fiscales validos que forman parte del analisis.
    n_negativos = (df[col_valor] < 0).sum()
    if n_negativos > 0:
        print(f"Aviso: {n_negativos} registros con ValorRecaudo < 0 (anulaciones fiscales, se conservan).")
    
    print(f"Datos cargados: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    print(f"   Periodo: {df[col_fecha].min().strftime('%Y-%m-%d')} a {df[col_fecha].max().strftime('%Y-%m-%d')}")
    
    return df


def agregar_mensual(df, col_fecha='FechaRecaudo', col_valor='ValorRecaudo'):
    """
    Agrega datos transaccionales a nivel mensual.
    
    Returns
    -------
    pd.DataFrame con índice DatetimeIndex y columna 'Recaudo_Neto'
    """
    df = df.copy()
    df[col_fecha] = pd.to_datetime(df[col_fecha])
    df = df.set_index(col_fecha)
    
    df_mensual = df[col_valor].resample('MS').sum().to_frame(name='Recaudo_Neto')
    
    print(f"✅ Serie mensual: {len(df_mensual)} observaciones")
    return df_mensual


def agregar_bimestral(df, col_fecha='FechaRecaudo', col_valor='ValorRecaudo'):
    """Agrega datos a nivel bimestral (cada 2 meses)."""
    df = df.copy()
    df[col_fecha] = pd.to_datetime(df[col_fecha])
    df = df.set_index(col_fecha)
    df_bim = df[col_valor].resample('2MS').sum().to_frame(name='Recaudo_Neto')
    print(f"✅ Serie bimestral: {len(df_bim)} observaciones")
    return df_bim


def agregar_trimestral(df, col_fecha='FechaRecaudo', col_valor='ValorRecaudo'):
    """Agrega datos a nivel trimestral (cada 3 meses)."""
    df = df.copy()
    df[col_fecha] = pd.to_datetime(df[col_fecha])
    df = df.set_index(col_fecha)
    df_tri = df[col_valor].resample('3MS').sum().to_frame(name='Recaudo_Neto')
    print(f"✅ Serie trimestral: {len(df_tri)} observaciones")
    return df_tri


def preparar_features_ml(df_mensual):
    """
    Genera features para modelos de ML (XGBoost, LSTM).
    
    Features generados:
    - Lags: 1, 2, 3, 6, 12
    - Medias móviles: 3, 6
    - Desviación estándar móvil: 3
    - Mes, Año, Trimestre
    - Dummies de picos (Ene, Jul)
    - Variables macro si están disponibles
    
    Returns
    -------
    pd.DataFrame con features, sin NaNs
    """
    df = df_mensual.copy()
    
    # Lags
    for lag in [1, 2, 3, 6, 12]:
        df[f'lag_{lag}'] = df['Recaudo_Neto'].shift(lag)
    
    # Medias móviles
    df['rolling_mean_3'] = df['Recaudo_Neto'].rolling(window=3).mean()
    df['rolling_mean_6'] = df['Recaudo_Neto'].rolling(window=6).mean()
    df['rolling_std_3'] = df['Recaudo_Neto'].rolling(window=3).std()
    
    # Variables calendáricas
    df['Mes'] = df.index.month
    df['Año'] = df.index.year
    df['Trimestre'] = df.index.quarter
    
    # Dummies de picos
    df['es_enero'] = (df.index.month == 1).astype(int)
    df['es_julio'] = (df.index.month == 7).astype(int)
    df['es_festividad'] = df.index.month.isin([6, 12]).astype(int)
    
    # Variables macro
    if config:
        df['IPC'] = df['Año'].map(lambda y: config.MACRO_DATA.get(y, {}).get('IPC', np.nan))
        df['Salario_Minimo'] = df['Año'].map(lambda y: config.MACRO_DATA.get(y, {}).get('Salario_Minimo', np.nan))
        df['UPC'] = df['Año'].map(lambda y: config.MACRO_DATA.get(y, {}).get('UPC', np.nan))
    
    # YoY change
    df['YoY_Recaudo'] = df['Recaudo_Neto'].pct_change(12) * 100
    
    # Limpiar NaNs
    df.dropna(inplace=True)
    
    print(f"✅ Features ML generados: {df.shape[1]} columnas, {len(df)} filas")
    return df


# ============================================================
# 2. TESTS ESTADÍSTICOS
# ============================================================

def test_estacionariedad(serie, nombre='Serie'):
    """
    Ejecuta pruebas ADF y KPSS sobre la serie.
    
    Returns
    -------
    dict con resultados
    """
    print(f"\n{'='*60}")
    print(f"  PRUEBAS DE ESTACIONARIEDAD: {nombre}")
    print(f"{'='*60}")
    
    # ADF
    adf_result = adfuller(serie.dropna(), autolag='AIC')
    adf_stat, adf_pval = adf_result[0], adf_result[1]
    adf_estacionaria = adf_pval < 0.05
    
    print(f"\n📊 Prueba ADF (H0: raíz unitaria)")
    print(f"   Estadístico: {adf_stat:.4f}")
    print(f"   p-valor:     {adf_pval:.4f}")
    print(f"   Resultado:   {'✅ Estacionaria' if adf_estacionaria else '⚠️ NO estacionaria'}")
    
    # KPSS
    try:
        kpss_result = kpss(serie.dropna(), regression='c', nlags='auto')
        kpss_stat, kpss_pval = kpss_result[0], kpss_result[1]
        kpss_estacionaria = kpss_pval > 0.05
        
        print(f"\n📊 Prueba KPSS (H0: estacionaria)")
        print(f"   Estadístico: {kpss_stat:.4f}")
        print(f"   p-valor:     {kpss_pval:.4f}")
        print(f"   Resultado:   {'✅ Estacionaria' if kpss_estacionaria else '⚠️ NO estacionaria'}")
    except Exception:
        kpss_stat, kpss_pval, kpss_estacionaria = None, None, None
    
    return {
        'adf_stat': adf_stat, 'adf_pval': adf_pval, 'adf_estacionaria': adf_estacionaria,
        'kpss_stat': kpss_stat, 'kpss_pval': kpss_pval, 'kpss_estacionaria': kpss_estacionaria,
    }


# ============================================================
# 3. MÉTRICAS DE EVALUACIÓN
# ============================================================

def calcular_metricas(y_real, y_pred, modelo_nombre='Modelo'):
    """
    Calcula RMSE, MAPE, MAE y R² para un pronóstico.
    
    Returns
    -------
    dict con las métricas
    """
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)
    
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mae = mean_absolute_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)
    
    # MAPE (evitando división por cero)
    mask = y_real != 0
    mape = np.mean(np.abs((y_real[mask] - y_pred[mask]) / y_real[mask])) * 100
    
    print(f"\n📈 Métricas — {modelo_nombre}")
    print(f"   RMSE:  {rmse:,.0f}")
    print(f"   MAE:   {mae:,.0f}")
    print(f"   MAPE:  {mape:.2f}%")
    print(f"   R²:    {r2:.4f}")
    
    return {
        'Modelo': modelo_nombre,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE (%)': mape,
        'R²': r2,
    }


# ============================================================
# 4. VISUALIZACIÓN
# ============================================================

def plot_serie_tiempo(serie, titulo='Serie de Tiempo', color='#2C3E50', save_path=None):
    """Gráfico estándar de una serie de tiempo."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(serie.index, serie.values, color=color, linewidth=1.5)
    ax.set_title(titulo, fontweight='bold', fontsize=14)
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Recaudo Neto ($)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Gráfico guardado: {save_path}")
    
    plt.show()
    return fig


def plot_forecast(y_real, predicciones_dict, titulo='Comparación de Pronósticos', save_path=None):
    """
    Gráfico con múltiples pronósticos superpuestos.
    
    Parameters
    ----------
    y_real : pd.Series
        Serie real
    predicciones_dict : dict
        {nombre_modelo: pd.Series_pronostico}
    """
    colores_default = ['#E74C3C', '#3498DB', '#27AE60', '#F39C12', '#9B59B6', '#1ABC9C']
    
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(y_real.index, y_real.values, 'o-', color='#2C3E50', linewidth=2, markersize=5, label='Real', zorder=5)
    
    for i, (nombre, pred) in enumerate(predicciones_dict.items()):
        color = colores_default[i % len(colores_default)]
        ax.plot(pred.index, pred.values, '--', color=color, linewidth=1.8, label=nombre, alpha=0.8)
    
    ax.set_title(titulo, fontweight='bold', fontsize=14)
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Recaudo Neto ($)')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Gráfico guardado: {save_path}")
    
    plt.show()
    return fig


def plot_residuos(residuos, titulo='Diagnóstico de Residuos', save_path=None):
    """Gráfico de diagnóstico de residuos: histograma + serie temporal."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Serie temporal de residuos
    axes[0].plot(residuos, color='#E74C3C', linewidth=1)
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[0].axhline(y=residuos.std() * 2, color='gray', linestyle='--', alpha=0.5, label='±2σ')
    axes[0].axhline(y=-residuos.std() * 2, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_title('Residuos en el Tiempo')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Histograma
    axes[1].hist(residuos, bins=15, color='#3498DB', edgecolor='white', alpha=0.8, density=True)
    axes[1].set_title('Distribución de Residuos')
    axes[1].set_xlabel('Residuo')
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(titulo, fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig


def formato_pesos(valor):
    """Formatea un valor numérico a pesos colombianos."""
    if abs(valor) >= 1e12:
        return f"${valor/1e12:,.2f}B"
    elif abs(valor) >= 1e9:
        return f"${valor/1e9:,.2f}MM"
    elif abs(valor) >= 1e6:
        return f"${valor/1e6:,.2f}M"
    else:
        return f"${valor:,.0f}"
