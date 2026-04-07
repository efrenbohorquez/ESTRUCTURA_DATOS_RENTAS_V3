======================================================================
REPORTE TECNICO: LSTM CON AUMENTACION TEMPORAL
======================================================================

Fecha: 2026-04-07 11:53

1. AUMENTACION DE DATOS
   Backcast: 36 meses (Oct 2018 – Sep 2021)
   Metodo: STL decomposition + trend extrapolation
   Serie total: 87 meses
   Entrenamiento: 83 meses
   Ventanas: 71 (vs ~38 original)

2. ARQUITECTURA (identica a NB 07)
   LSTM(64) -> Dropout(0.2) -> LSTM(32) -> Dropout(0.2) -> Dense(16) -> Dense(1)
   Parametros: 31,905

3. METRICAS OOS (Oct-Dic 2025)
   MAPE aumentado: 23.45%
   MAPE original:  23.52%
   Mejora:         0.07pp
   RMSE: $72.1MM COP
   MAE:  $59.8MM COP

4. DIAGNOSTICO DE RESIDUOS
   3/4 pruebas superadas

5. PRONOSTICO EXTENDIDO
   2026: $3,614MM
   2027: $3,915MM
   2028: $4,088MM
   2029: $4,219MM
   2030: $4,344MM
   Total 2026–2030: $20,179MM

======================================================================