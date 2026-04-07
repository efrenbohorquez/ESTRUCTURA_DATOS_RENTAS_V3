"""
agente_rentas.py — Agente Autónomo de Pronóstico de Rentas Cedidas
===================================================================
Implementa un agente ReAct (Reason + Act) que orquesta el pipeline
completo de pronóstico: valida datos, ejecuta modelos, consolida
resultados y genera reportes ejecutivos.

Arquitectura:
    ┌─────────────────────────────────┐
    │  AGENTE STAR                    │
    │  (Sistema Territorial de        │
    │   Análisis de Rentas)           │
    ├─────────────────────────────────┤
    │  Cerebro: Loop ReAct            │
    │  ┌─────────────────────┐        │
    │  │ 1. Observar estado  │        │
    │  │ 2. Razonar          │        │
    │  │ 3. Seleccionar tool │        │
    │  │ 4. Ejecutar         │        │
    │  │ 5. Evaluar          │◄─ loop │
    │  └─────────────────────┘        │
    ├─────────────────────────────────┤
    │  Tools:                         │
    │  🔍 diagnosticar_datos          │
    │  📊 cargar_pronosticos          │
    │  ⚖️  comparar_modelos           │
    │  🚨 detectar_alertas            │
    │  📈 consolidar_forecast         │
    │  📋 reporte_ejecutivo           │
    └─────────────────────────────────┘

Uso:
    from agente_rentas import AgenteRentas
    agente = AgenteRentas()
    resultado = agente.ejecutar("diagnosticar y generar reporte consolidado 2026")
"""

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import json
import traceback
from typing import Callable

# ============================================================
# RUTAS
# ============================================================
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS_FORECASTS = PROJECT_ROOT / "outputs" / "forecasts"
OUTPUTS_REPORTS = PROJECT_ROOT / "outputs" / "reports"
OUTPUTS_FIGURES = PROJECT_ROOT / "outputs" / "figures"

# ============================================================
# REGISTRO DE HERRAMIENTAS (TOOL REGISTRY)
# ============================================================

@dataclass
class Tool:
    """Definición de una herramienta del agente."""
    nombre: str
    descripcion: str
    funcion: Callable
    categoria: str = "general"

@dataclass
class Observacion:
    """Resultado de ejecutar una herramienta."""
    tool: str
    exito: bool
    datos: dict = field(default_factory=dict)
    mensaje: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))

@dataclass
class Paso:
    """Un paso en el razonamiento del agente."""
    pensamiento: str
    accion: str
    observacion: Observacion = None

# ============================================================
# HERRAMIENTAS DEL AGENTE
# ============================================================

def tool_diagnosticar_datos(**kwargs) -> Observacion:
    """Diagnostica la calidad y completitud del dataset principal."""
    try:
        serie_path = DATA_PROCESSED / "serie_mensual.csv"
        if not serie_path.exists():
            return Observacion("diagnosticar_datos", False,
                               mensaje="No se encontro serie_mensual.csv")

        df = pd.read_csv(serie_path, parse_dates=['Fecha'])
        n_meses = len(df)
        fecha_min = df['Fecha'].min()
        fecha_max = df['Fecha'].max()
        nulls = df.isnull().sum().to_dict()
        total_nulls = sum(nulls.values())

        # Verificar continuidad temporal
        expected_months = pd.date_range(fecha_min, fecha_max, freq='MS')
        meses_faltantes = len(expected_months) - n_meses

        # Estadisticas basicas
        stats = {
            'media': float(df['Recaudo_Total'].mean()),
            'std': float(df['Recaudo_Total'].std()),
            'min': float(df['Recaudo_Total'].min()),
            'max': float(df['Recaudo_Total'].max()),
            'cv': float(df['Recaudo_Total'].std() / df['Recaudo_Total'].mean() * 100),
        }

        # Detectar outliers (IQR)
        Q1 = df['Recaudo_Total'].quantile(0.25)
        Q3 = df['Recaudo_Total'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df['Recaudo_Total'] < Q1 - 1.5*IQR) |
                      (df['Recaudo_Total'] > Q3 + 1.5*IQR)]

        # Ultimo dato disponible
        ultimo = df.iloc[-1]

        diagnostico = {
            'n_meses': n_meses,
            'rango': f"{fecha_min.strftime('%b %Y')} - {fecha_max.strftime('%b %Y')}",
            'fecha_inicio': str(fecha_min.date()),
            'fecha_fin': str(fecha_max.date()),
            'nulls_totales': total_nulls,
            'meses_faltantes': meses_faltantes,
            'columnas': list(df.columns),
            'estadisticas': stats,
            'n_outliers': len(outliers),
            'outliers_fechas': [d.strftime('%b %Y') for d in outliers['Fecha']] if len(outliers) > 0 else [],
            'ultimo_dato': {
                'fecha': ultimo['Fecha'].strftime('%b %Y'),
                'recaudo': float(ultimo['Recaudo_Total']),
            },
            'calidad': 'OPTIMA' if total_nulls == 0 and meses_faltantes == 0 else 'REQUIERE_ATENCION',
        }

        return Observacion("diagnosticar_datos", True, diagnostico,
                           f"Serie: {n_meses} meses | {diagnostico['rango']} | "
                           f"Calidad: {diagnostico['calidad']}")

    except Exception as e:
        return Observacion("diagnosticar_datos", False, mensaje=f"Error: {e}")


def tool_cargar_pronosticos(**kwargs) -> Observacion:
    """Carga todos los pronosticos disponibles de cada modelo."""
    try:
        modelos = {}
        archivos_oos = {
            'SARIMAX': 'sarimax_forecast.csv',
            'Prophet': 'prophet_forecast.csv',
            'XGBoost': 'xgboost_forecast.csv',
            'LSTM': 'lstm_forecast.csv',
            'Ensemble_Foundation': 'foundation_ensemble_forecast.csv',
        }
        archivos_2026 = {
            'SARIMAX': ('sarimax_forecast_2026.csv', 'Pronóstico'),
            'Prophet': ('prophet_forecast_2026.csv', 'Pronóstico'),
            'XGBoost': ('xgboost_forecast_2026.csv', 'Pronóstico'),
            'LSTM': ('lstm_forecast_2026.csv', 'Pronóstico'),
            'Ensemble_Foundation': ('foundation_ensemble_forecast_2026.csv', 'Pronostico_Ensemble'),
            'Distributional_P50': ('distributional_forecast_2026.csv', 'P50_Mediana'),
        }

        # OOS (Oct-Dic 2025)
        for modelo, archivo in archivos_oos.items():
            path = OUTPUTS_FORECASTS / archivo
            if path.exists():
                df = pd.read_csv(path, parse_dates=['Fecha'])
                # Buscar columna de pronostico
                pred_col = [c for c in df.columns if 'Pronostico' in c or 'Pronóstico' in c]
                if pred_col:
                    mape_vals = df.get('Error_Pct', pd.Series(dtype=float))
                    mape = float(mape_vals.abs().mean()) if len(mape_vals) > 0 else None
                    modelos[modelo] = {
                        'oos_disponible': True,
                        'mape_oos': mape,
                        'col_pred': pred_col[0],
                    }

        # 2026
        for modelo, (archivo, col) in archivos_2026.items():
            path = OUTPUTS_FORECASTS / archivo
            if path.exists():
                df = pd.read_csv(path, parse_dates=['Fecha'])
                if col in df.columns:
                    total_anual = float(df[col].sum())
                    if modelo not in modelos:
                        modelos[modelo] = {}
                    modelos[modelo]['forecast_2026_disponible'] = True
                    modelos[modelo]['total_anual_2026'] = total_anual
                    modelos[modelo]['mensual_2026'] = [float(v) for v in df[col].values]

        return Observacion("cargar_pronosticos", True, modelos,
                           f"Modelos cargados: {len(modelos)} | "
                           f"Con OOS: {sum(1 for m in modelos.values() if m.get('oos_disponible'))} | "
                           f"Con 2026: {sum(1 for m in modelos.values() if m.get('forecast_2026_disponible'))}")

    except Exception as e:
        return Observacion("cargar_pronosticos", False, mensaje=f"Error: {e}")


def tool_comparar_modelos(**kwargs) -> Observacion:
    """Compara todos los modelos y genera un ranking."""
    try:
        pronosticos = kwargs.get('pronosticos', {})
        if not pronosticos:
            # Cargar si no se proporcionaron
            obs = tool_cargar_pronosticos()
            pronosticos = obs.datos

        # Ranking por MAPE OOS
        ranking_oos = []
        for modelo, info in pronosticos.items():
            if info.get('mape_oos') is not None:
                ranking_oos.append({
                    'modelo': modelo,
                    'mape_oos': info['mape_oos'],
                })
        ranking_oos.sort(key=lambda x: x['mape_oos'])

        # Ranking por total anual 2026
        ranking_2026 = []
        for modelo, info in pronosticos.items():
            if info.get('total_anual_2026') is not None:
                ranking_2026.append({
                    'modelo': modelo,
                    'total_B': info['total_anual_2026'] / 1e9,
                    'total_T': info['total_anual_2026'] / 1e12,
                })
        ranking_2026.sort(key=lambda x: x['total_B'])

        # Consensus forecast (mediana de todos los modelos)
        totales = [r['total_B'] for r in ranking_2026]
        consensus = {
            'mediana_B': float(np.median(totales)) if totales else 0,
            'media_B': float(np.mean(totales)) if totales else 0,
            'std_B': float(np.std(totales)) if totales else 0,
            'min_B': float(np.min(totales)) if totales else 0,
            'max_B': float(np.max(totales)) if totales else 0,
            'rango_B': float(np.max(totales) - np.min(totales)) if totales else 0,
            'cv_pct': float(np.std(totales) / np.mean(totales) * 100) if totales else 0,
        }

        # Mejor modelo
        mejor = ranking_oos[0] if ranking_oos else None

        return Observacion("comparar_modelos", True,
                           {'ranking_oos': ranking_oos,
                            'ranking_2026': ranking_2026,
                            'consensus': consensus,
                            'mejor_modelo': mejor},
                           f"Mejor modelo OOS: {mejor['modelo']} (MAPE {mejor['mape_oos']:.2f}%)" if mejor else "Sin datos OOS")

    except Exception as e:
        return Observacion("comparar_modelos", False, mensaje=f"Error: {e}")


def tool_detectar_alertas(**kwargs) -> Observacion:
    """Detecta anomalias y genera alertas del sistema."""
    try:
        pronosticos = kwargs.get('pronosticos', {})
        comparacion = kwargs.get('comparacion', {})
        diagnostico = kwargs.get('diagnostico', {})
        alertas = []

        # Alerta 1: Divergencia entre modelos
        consensus = comparacion.get('consensus', {})
        if consensus.get('cv_pct', 0) > 15:
            alertas.append({
                'nivel': 'ALTO',
                'tipo': 'DIVERGENCIA_MODELOS',
                'mensaje': f"Los modelos divergen significativamente (CV={consensus['cv_pct']:.1f}%). "
                           f"Rango: ${consensus['min_B']:.0f}B - ${consensus['max_B']:.0f}B",
                'accion': 'Revisar supuestos de cada modelo y recalibrar.',
            })
        elif consensus.get('cv_pct', 0) > 8:
            alertas.append({
                'nivel': 'MEDIO',
                'tipo': 'DIVERGENCIA_MODERADA',
                'mensaje': f"Divergencia moderada entre modelos (CV={consensus['cv_pct']:.1f}%)",
                'accion': 'Monitorear, ponderar por MAPE OOS.',
            })
        else:
            alertas.append({
                'nivel': 'BAJO',
                'tipo': 'MODELOS_CONSISTENTES',
                'mensaje': f"Los modelos convergen bien (CV={consensus.get('cv_pct', 0):.1f}%)",
                'accion': 'Ninguna requerida.',
            })

        # Alerta 2: Datos desactualizados
        if diagnostico:
            fecha_fin_str = diagnostico.get('fecha_fin', '')
            if fecha_fin_str:
                fecha_fin = pd.Timestamp(fecha_fin_str)
                dias_sin_actualizar = (pd.Timestamp.now() - fecha_fin).days
                if dias_sin_actualizar > 60:
                    alertas.append({
                        'nivel': 'ALTO',
                        'tipo': 'DATOS_DESACTUALIZADOS',
                        'mensaje': f"Ultimo dato: {diagnostico.get('ultimo_dato', {}).get('fecha', '?')} "
                                   f"({dias_sin_actualizar} dias sin actualizar)",
                        'accion': 'Cargar datos recientes y re-entrenar modelos.',
                    })

        # Alerta 3: Modelo con alto error
        ranking = comparacion.get('ranking_oos', [])
        for r in ranking:
            if r['mape_oos'] > 10:
                alertas.append({
                    'nivel': 'MEDIO',
                    'tipo': 'MODELO_IMPRECISO',
                    'mensaje': f"{r['modelo']}: MAPE = {r['mape_oos']:.2f}% (> 10%)",
                    'accion': f'Considerar excluir {r["modelo"]} del ensemble o reoptimizar.',
                })

        # Alerta 4: Calidad de datos
        if diagnostico.get('calidad') == 'REQUIERE_ATENCION':
            alertas.append({
                'nivel': 'ALTO',
                'tipo': 'CALIDAD_DATOS',
                'mensaje': f"Nulls: {diagnostico.get('nulls_totales', 0)}, "
                           f"Meses faltantes: {diagnostico.get('meses_faltantes', 0)}",
                'accion': 'Revisar pipeline de ETL.',
            })

        # Alerta 5: Outliers
        if diagnostico.get('n_outliers', 0) > 0:
            alertas.append({
                'nivel': 'INFO',
                'tipo': 'OUTLIERS_DETECTADOS',
                'mensaje': f"{diagnostico['n_outliers']} outlier(s): {diagnostico.get('outliers_fechas', [])}",
                'accion': 'Verificar si corresponden a eventos reales (picos fiscales).',
            })

        n_altos = sum(1 for a in alertas if a['nivel'] == 'ALTO')
        return Observacion("detectar_alertas", True,
                           {'alertas': alertas, 'n_total': len(alertas), 'n_criticas': n_altos},
                           f"Alertas: {len(alertas)} ({n_altos} criticas)")

    except Exception as e:
        return Observacion("detectar_alertas", False, mensaje=f"Error: {e}")


def tool_consolidar_forecast(**kwargs) -> Observacion:
    """Genera un pronostico consolidado ponderado por desempeno OOS."""
    try:
        pronosticos = kwargs.get('pronosticos', {})
        if not pronosticos:
            obs = tool_cargar_pronosticos()
            pronosticos = obs.datos

        # Modelos con datos OOS Y forecast 2026
        modelos_validos = {}
        for nombre, info in pronosticos.items():
            if info.get('mape_oos') is not None and info.get('mensual_2026') is not None:
                modelos_validos[nombre] = info

        if not modelos_validos:
            return Observacion("consolidar_forecast", False,
                               mensaje="No hay modelos con OOS + forecast 2026")

        # Pesos inversamente proporcionales al MAPE
        mapes = {n: info['mape_oos'] for n, info in modelos_validos.items()}
        inv_mapes = {n: 1.0 / max(m, 0.01) for n, m in mapes.items()}
        total_inv = sum(inv_mapes.values())
        pesos = {n: v / total_inv for n, v in inv_mapes.items()}

        # Forecast ponderado
        n_meses = 12
        forecast_ponderado = np.zeros(n_meses)
        contribuciones = {}
        for nombre, info in modelos_validos.items():
            vals = np.array(info['mensual_2026'][:n_meses])
            peso = pesos[nombre]
            forecast_ponderado += vals * peso
            contribuciones[nombre] = {
                'peso': float(peso),
                'mape_oos': float(mapes[nombre]),
                'total_anual_B': float(vals.sum() / 1e9),
            }

        # Intervalo de confianza (min-max de modelos)
        all_forecasts = np.array([
            np.array(info['mensual_2026'][:n_meses])
            for info in modelos_validos.values()
        ])
        ic_inferior = all_forecasts.min(axis=0)
        ic_superior = all_forecasts.max(axis=0)

        # Fechas 2026
        fechas = pd.date_range('2026-01-01', periods=12, freq='MS')
        df_consolidado = pd.DataFrame({
            'Fecha': fechas,
            'Pronostico_Consolidado': forecast_ponderado,
            'IC_Inferior_Modelos': ic_inferior,
            'IC_Superior_Modelos': ic_superior,
        })

        # Guardar CSV
        out_path = OUTPUTS_FORECASTS / 'consolidado_agente_2026.csv'
        df_consolidado.to_csv(out_path, index=False)

        total_anual = float(forecast_ponderado.sum())
        resultado = {
            'total_anual_B': total_anual / 1e9,
            'total_anual_T': total_anual / 1e12,
            'mensual': [float(v) for v in forecast_ponderado],
            'ic_inferior_total_B': float(ic_inferior.sum() / 1e9),
            'ic_superior_total_B': float(ic_superior.sum() / 1e9),
            'contribuciones': contribuciones,
            'n_modelos': len(modelos_validos),
            'archivo_csv': str(out_path),
        }

        return Observacion("consolidar_forecast", True, resultado,
                           f"Consolidado 2026: ${resultado['total_anual_T']:.2f}T "
                           f"[${resultado['ic_inferior_total_B']:.0f}B - ${resultado['ic_superior_total_B']:.0f}B] | "
                           f"{len(modelos_validos)} modelos")

    except Exception as e:
        return Observacion("consolidar_forecast", False, mensaje=f"Error: {e}")


def tool_reporte_ejecutivo(**kwargs) -> Observacion:
    """Genera un reporte ejecutivo consolidado."""
    try:
        diagnostico = kwargs.get('diagnostico', {})
        comparacion = kwargs.get('comparacion', {})
        alertas = kwargs.get('alertas', {})
        consolidado = kwargs.get('consolidado', {})

        lineas = []
        sep = '=' * 72
        sub = '-' * 72

        lineas.append(sep)
        lineas.append('  REPORTE EJECUTIVO — AGENTE STAR')
        lineas.append('  Sistema Territorial de Analisis de Rentas')
        lineas.append(f'  Generado: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
        lineas.append(sep)

        # 1. Estado de datos
        lineas.append('\n  1. ESTADO DE LOS DATOS')
        lineas.append(f'  {sub}')
        if diagnostico:
            lineas.append(f'     Serie temporal: {diagnostico.get("n_meses", "?")} meses')
            lineas.append(f'     Periodo: {diagnostico.get("rango", "?")}')
            lineas.append(f'     Calidad: {diagnostico.get("calidad", "?")}')
            stats = diagnostico.get('estadisticas', {})
            if stats:
                lineas.append(f'     Recaudo medio: ${stats.get("media", 0)/1e9:.1f}B/mes')
                lineas.append(f'     Coef. variacion: {stats.get("cv", 0):.1f}%')
            lineas.append(f'     Outliers: {diagnostico.get("n_outliers", 0)}')

        # 2. Ranking de modelos
        lineas.append('\n  2. RANKING DE MODELOS (MAPE OOS Oct-Dic 2025)')
        lineas.append(f'  {sub}')
        ranking = comparacion.get('ranking_oos', [])
        for i, r in enumerate(ranking, 1):
            medalla = ['   ', '   ', '   '][min(i-1, 2)] if i <= 3 else '     '
            lineas.append(f'     {i}. {r["modelo"]:<25} MAPE = {r["mape_oos"]:>6.2f}%')

        # 3. Pronostico consolidado
        lineas.append('\n  3. PRONOSTICO CONSOLIDADO 2026')
        lineas.append(f'  {sub}')
        if consolidado:
            lineas.append(f'     Total anual:  ${consolidado.get("total_anual_T", 0):.2f} billones')
            lineas.append(f'     Rango modelos: ${consolidado.get("ic_inferior_total_B", 0):.0f}B'
                          f' - ${consolidado.get("ic_superior_total_B", 0):.0f}B')
            lineas.append(f'     Modelos en consensus: {consolidado.get("n_modelos", 0)}')
            contribs = consolidado.get('contribuciones', {})
            if contribs:
                lineas.append(f'\n     Pesos del ensemble ponderado:')
                for modelo, info in sorted(contribs.items(), key=lambda x: -x[1]['peso']):
                    lineas.append(f'       {modelo:<25} peso={info["peso"]:.1%}  '
                                  f'(MAPE={info["mape_oos"]:.2f}%)')

        # 4. Alertas
        lineas.append('\n  4. ALERTAS DEL SISTEMA')
        lineas.append(f'  {sub}')
        alertas_list = alertas.get('alertas', [])
        if alertas_list:
            for a in alertas_list:
                icono = {'ALTO': '!!!', 'MEDIO': ' ! ', 'BAJO': ' . ', 'INFO': ' i '}
                lineas.append(f'     [{icono.get(a["nivel"], "?")}] {a["tipo"]}: {a["mensaje"]}')
                lineas.append(f'           Accion: {a["accion"]}')
        else:
            lineas.append('     Sin alertas activas.')

        # 5. Consenso inter-modelos
        lineas.append('\n  5. CONSENSUS INTER-MODELOS')
        lineas.append(f'  {sub}')
        consensus = comparacion.get('consensus', {})
        if consensus:
            lineas.append(f'     Mediana: ${consensus.get("mediana_B", 0):.1f}B')
            lineas.append(f'     Media:   ${consensus.get("media_B", 0):.1f}B')
            lineas.append(f'     Desv. Est: ${consensus.get("std_B", 0):.1f}B')
            lineas.append(f'     Coef. Variacion: {consensus.get("cv_pct", 0):.1f}%')
            lineas.append(f'     Dispersion: '
                          + ('BAJA' if consensus['cv_pct'] < 5
                             else 'MODERADA' if consensus['cv_pct'] < 10
                             else 'ALTA'))

        lineas.append(f'\n{sep}')
        lineas.append('  Fin del reporte — Agente STAR v1.0')
        lineas.append(sep)

        reporte_texto = '\n'.join(lineas)
        return Observacion("reporte_ejecutivo", True,
                           {'texto': reporte_texto},
                           f"Reporte generado ({len(lineas)} lineas)")

    except Exception as e:
        return Observacion("reporte_ejecutivo", False, mensaje=f"Error: {e}")


# ============================================================
# CLASE PRINCIPAL: AgenteRentas
# ============================================================

class AgenteRentas:
    """
    Agente autónomo ReAct para el Sistema de Pronóstico de Rentas Cedidas.

    Ciclo:  Percibir → Razonar → Actuar → Observar → (repetir)

    El agente mantiene un estado interno (memoria de trabajo) y un log
    de todos los pasos ejecutados para auditabilidad completa.
    """

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.estado = {}        # Memoria de trabajo
        self.log = []           # Traza completa de ejecucion
        self.resultados = {}    # Resultados de cada tool

        # Registrar herramientas
        self.tools = {
            'diagnosticar_datos': Tool(
                'diagnosticar_datos',
                'Analiza calidad, completitud y estadisticas del dataset',
                tool_diagnosticar_datos,
                'datos'
            ),
            'cargar_pronosticos': Tool(
                'cargar_pronosticos',
                'Carga pronosticos OOS y 2026 de todos los modelos',
                tool_cargar_pronosticos,
                'modelos'
            ),
            'comparar_modelos': Tool(
                'comparar_modelos',
                'Compara modelos, genera ranking y consensus forecast',
                tool_comparar_modelos,
                'analisis'
            ),
            'detectar_alertas': Tool(
                'detectar_alertas',
                'Detecta anomalias, divergencias y genera alertas',
                tool_detectar_alertas,
                'monitoreo'
            ),
            'consolidar_forecast': Tool(
                'consolidar_forecast',
                'Genera pronostico ponderado por MAPE de todos los modelos',
                tool_consolidar_forecast,
                'pronostico'
            ),
            'reporte_ejecutivo': Tool(
                'reporte_ejecutivo',
                'Genera reporte ejecutivo consolidado en texto',
                tool_reporte_ejecutivo,
                'reportes'
            ),
        }

    def _log(self, msg):
        """Imprime y registra un mensaje."""
        ts = datetime.now().strftime("%H:%M:%S")
        entrada = f"[{ts}] {msg}"
        self.log.append(entrada)
        if self.verbose:
            print(entrada)

    def _razonar(self, objetivo: str) -> list:
        """
        Determina la secuencia de acciones basada en el objetivo.

        Este es el 'cerebro' del agente: un planificador basado en reglas
        que descompone objetivos complejos en pasos ejecutables.
        """
        objetivo_lower = objetivo.lower()
        plan = []

        # Siempre empezar con diagnostico
        plan.append(('diagnosticar_datos', {}))

        # Cargar pronosticos si se necesita analisis/comparacion
        if any(kw in objetivo_lower for kw in
               ['pronostico', 'forecast', 'modelo', 'comparar', 'consolidar',
                'reporte', 'completo', 'todo', 'ejecutar', 'diagnosticar']):
            plan.append(('cargar_pronosticos', {}))
            plan.append(('comparar_modelos', {}))
            plan.append(('detectar_alertas', {}))

        # Consolidar si se pide pronostico
        if any(kw in objetivo_lower for kw in
               ['consolidar', 'pronostico', 'forecast', '2026',
                'reporte', 'completo', 'todo', 'ejecutar']):
            plan.append(('consolidar_forecast', {}))

        # Reporte siempre al final si es completo
        if any(kw in objetivo_lower for kw in
               ['reporte', 'completo', 'todo', 'ejecutar', 'informe']):
            plan.append(('reporte_ejecutivo', {}))

        # Si no se activo nada especifico, hacer todo
        if len(plan) <= 1:
            plan = [
                ('diagnosticar_datos', {}),
                ('cargar_pronosticos', {}),
                ('comparar_modelos', {}),
                ('detectar_alertas', {}),
                ('consolidar_forecast', {}),
                ('reporte_ejecutivo', {}),
            ]

        return plan

    def ejecutar(self, objetivo: str = "diagnosticar y generar reporte consolidado 2026") -> dict:
        """
        Punto de entrada principal del agente.

        Ejecuta el ciclo ReAct completo:
        1. Razona sobre el objetivo
        2. Ejecuta las herramientas en secuencia
        3. Propaga resultados entre herramientas
        4. Retorna resumen consolidado
        """
        self._log(f"{'='*60}")
        self._log(f"AGENTE STAR — Inicio de ejecucion")
        self._log(f"Objetivo: {objetivo}")
        self._log(f"{'='*60}")

        # Fase 1: RAZONAR — planificar acciones
        plan = self._razonar(objetivo)
        self._log(f"\nPlan de ejecucion ({len(plan)} pasos):")
        for i, (tool_name, _) in enumerate(plan, 1):
            self._log(f"  {i}. {tool_name}")

        # Fase 2: ACTUAR — ejecutar cada herramienta
        self._log(f"\n{'─'*60}")
        for i, (tool_name, params) in enumerate(plan, 1):
            self._log(f"\nPaso {i}/{len(plan)}: {tool_name}")
            self._log(f"  Pensamiento: Necesito {self.tools[tool_name].descripcion}")

            # Inyectar resultados previos como contexto
            kwargs = dict(params)
            if 'diagnostico' in self.resultados:
                kwargs['diagnostico'] = self.resultados['diagnostico']
            if 'pronosticos' in self.resultados:
                kwargs['pronosticos'] = self.resultados['pronosticos']
            if 'comparacion' in self.resultados:
                kwargs['comparacion'] = self.resultados['comparacion']
            if 'alertas' in self.resultados:
                kwargs['alertas'] = self.resultados['alertas']
            if 'consolidado' in self.resultados:
                kwargs['consolidado'] = self.resultados['consolidado']

            # Ejecutar
            try:
                obs = self.tools[tool_name].funcion(**kwargs)
                self._log(f"  Resultado: {'OK' if obs.exito else 'FALLO'} — {obs.mensaje}")

                # Guardar en memoria de trabajo
                key_map = {
                    'diagnosticar_datos': 'diagnostico',
                    'cargar_pronosticos': 'pronosticos',
                    'comparar_modelos': 'comparacion',
                    'detectar_alertas': 'alertas',
                    'consolidar_forecast': 'consolidado',
                    'reporte_ejecutivo': 'reporte',
                }
                self.resultados[key_map.get(tool_name, tool_name)] = obs.datos

            except Exception as e:
                self._log(f"  ERROR: {e}")
                self._log(f"  {traceback.format_exc()}")

        # Fase 3: REPORTAR
        self._log(f"\n{'='*60}")
        self._log(f"AGENTE STAR — Ejecucion completada")
        self._log(f"Pasos ejecutados: {len(plan)}")
        self._log(f"{'='*60}")

        return self.resultados

    def get_reporte_texto(self) -> str:
        """Retorna el reporte ejecutivo como texto."""
        reporte = self.resultados.get('reporte', {})
        return reporte.get('texto', '(no generado)')

    def get_log(self) -> str:
        """Retorna el log completo de ejecucion."""
        return '\n'.join(self.log)

    def get_forecast_consolidado(self) -> pd.DataFrame:
        """Retorna el forecast consolidado como DataFrame."""
        path = OUTPUTS_FORECASTS / 'consolidado_agente_2026.csv'
        if path.exists():
            return pd.read_csv(path, parse_dates=['Fecha'])
        return pd.DataFrame()

    def __repr__(self):
        return (f"AgenteRentas(tools={len(self.tools)}, "
                f"resultados={list(self.resultados.keys())})")


# ============================================================
# CLI — Ejecucion directa
# ============================================================
if __name__ == '__main__':
    import sys
    objetivo = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "diagnosticar y generar reporte consolidado 2026"
    agente = AgenteRentas(verbose=True)
    resultados = agente.ejecutar(objetivo)
    print('\n' + agente.get_reporte_texto())
