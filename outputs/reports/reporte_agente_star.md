# Reporte Ejecutivo — Agente STAR

Generado automaticamente por el Agente STAR

```
========================================================================
  REPORTE EJECUTIVO — AGENTE STAR
  Sistema Territorial de Analisis de Rentas
  Generado: 2026-04-07 10:48
========================================================================

  1. ESTADO DE LOS DATOS
  ------------------------------------------------------------------------
     Serie temporal: 51 meses
     Periodo: Oct 2021 - Dec 2025
     Calidad: REQUIERE_ATENCION
     Recaudo medio: $256.9B/mes
     Coef. variacion: 27.2%
     Outliers: 2

  2. RANKING DE MODELOS (MAPE OOS Oct-Dic 2025)
  ------------------------------------------------------------------------
     1. Ensemble_Foundation       MAPE =   1.06%
     2. XGBoost                   MAPE =   3.36%
     3. Prophet                   MAPE =   6.30%
     4. SARIMAX                   MAPE =   9.75%
     5. LSTM                      MAPE =  23.52%

  3. PRONOSTICO CONSOLIDADO 2026
  ------------------------------------------------------------------------
     Total anual:  $3.19 billones
     Rango modelos: $2968B - $4078B
     Modelos en consensus: 5

     Pesos del ensemble ponderado:
       Ensemble_Foundation       peso=61.1%  (MAPE=1.06%)
       XGBoost                   peso=19.2%  (MAPE=3.36%)
       Prophet                   peso=10.3%  (MAPE=6.30%)
       SARIMAX                   peso=6.6%  (MAPE=9.75%)
       LSTM                      peso=2.7%  (MAPE=23.52%)

  4. ALERTAS DEL SISTEMA
  ------------------------------------------------------------------------
     [ . ] MODELOS_CONSISTENTES: Los modelos convergen bien (CV=6.8%)
           Accion: Ninguna requerida.
     [!!!] DATOS_DESACTUALIZADOS: Ultimo dato: Dec 2025 (127 dias sin actualizar)
           Accion: Cargar datos recientes y re-entrenar modelos.
     [ ! ] MODELO_IMPRECISO: LSTM: MAPE = 23.52% (> 10%)
           Accion: Considerar excluir LSTM del ensemble o reoptimizar.
     [!!!] CALIDAD_DATOS: Nulls: 24, Meses faltantes: 0
           Accion: Revisar pipeline de ETL.
     [ i ] OUTLIERS_DETECTADOS: 2 outlier(s): ['Jan 2025', 'Jul 2025']
           Accion: Verificar si corresponden a eventos reales (picos fiscales).

  5. CONSENSUS INTER-MODELOS
  ------------------------------------------------------------------------
     Mediana: $3363.2B
     Media:   $3338.0B
     Desv. Est: $227.4B
     Coef. Variacion: 6.8%
     Dispersion: MODERADA

========================================================================
  Fin del reporte — Agente STAR v1.0
========================================================================
```
