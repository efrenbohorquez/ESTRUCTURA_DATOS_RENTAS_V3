# BORRADOR INFORME TÉCNICO: SISTEMA DE ANÁLISIS Y PRONÓSTICO DE RENTAS CEDIDAS
**Entrega de Refinamiento Final - 4 de Marzo**

---

## 1. Resumen de Estabilización de Datos
Se realizó una auditoría de integridad sobre el dataset `BaseRentasVF`, detectando **26 registros con valores negativos** en la columna `ValorRecaudo` (mínimo detectado: -$483M). 
- **Acción**: Se eliminaron estos registros por considerarse errores de anulación o ruido estadístico que distorsiona la tendencia real.
- **Resultado**: Dataset estabilizado con 149,384 observaciones para el periodo Ene-2022 a Dic-2025.

## 2. Hallazgos de Priorización (Análisis de Pareto)
Para enfocar el modelado predictivo, se identificaron las fuentes que componen el **85% del recaudo total**:
1. **Consumo Cigarrillos y Tabacos**: $3.18B (24.3%)
2. **COLJUEGOS (Régimen Subsidiado)**: $2.48B (19.0%)
3. **Impoconsumo Licores y Vinos**: $2.14B (16.3%)
4. **Monopolio de Juegos de Suerte y Azar**: $1.88B (14.4%)

*Implicación Estratégica*: El 50% del recaudo depende de solo 3 rubros, lo cual justifica el uso de modelos robustos (XGBoost/Prophet) que manejen shocks en estas industrias.

## 3. Análisis de Estacionalidad Mensual
Se ha procesado la serie en temporalidad mensual para capturar la dinámica de recaudación:
- **Mensual**: Captura el pico de Enero (impuestos anuales) y la caída en la actividad durante Diciembre.

## 4. Estrategia de Validación de Cierre
Siguiendo las mejores prácticas de "Backtesting", se configuró una **ventana ciega de validación**:
- **Train Set**: Ene-2022 a Sep-2025.
- **Validación Final**: Nov-2025 y Dic-2025 (Cierre de año).
Esta prueba de "estrés" permite verificar la precisión del modelo en el periodo de mayor incertidumbre (cierre fiscal).

## 5. Próximos Pasos (Sistema STAR)
Los modelos ajustados alimentarán el **Sistema de Alerta y Recomendación Territorial (STAR)**, permitiendo a los departamentos:
- Detectar desviaciones superiores al 10% en tiempo real.
- Analizar el impacto de variables macro (IPC/Salario Mínimo) en las rentas específicas.

---
*Nota: Este borrador está listo para ser traspasado a Word según la estructura de tesis.*
