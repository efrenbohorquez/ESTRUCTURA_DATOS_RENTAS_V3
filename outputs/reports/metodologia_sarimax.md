# Metodología: Modelo SARIMAX

El modelo **SARIMAX** (*Seasonal AutoRegressive Integrated Moving Average with eXogenous factors*) es una extensión del modelo ARIMA que permite manejar tanto la estacionalidad como variables externas (exógenas) que influyen en la serie de tiempo.

## 1. Estructura del Modelo
El modelo se define por los parámetros $(p, d, q) \times (P, D, Q)_s$:
- **Componentes no estacionales:** 
  - $p$: Orden de autorregresión (relación con valores pasados).
  - $d$: Orden de diferenciación (para hacer la serie estacionaria).
  - $q$: Orden de media móvil (relación con errores pasados).
- **Componentes estacionales:** 
  - $(P, D, Q)$: Versión estacional de los parámetros anteriores.
  - $s$: Periodicidad (en este caso, $s=12$ para periodicidad mensual).

## 2. Variables Exógenas ($X$)
Para este proyecto, se han integrado variables del entorno macroeconómico colombiano que afectan directamente el recaudo:
- **IPC (Índice de Precios al Consumidor):** Monitorea la inflación y el poder adquisitivo.
- **Salario Mínimo:** Factor determinante en el consumo y base gravable.
- **UPC (Unidad de Pago por Capitación):** Crítica para rentas relacionadas con el sector salud y seguridad social.

## 3. Proceso de Ajuste
1. **Identificación:** Análisis de las funciones de autocorrelación (ACF y PACF).
2. **Estimación:** Uso de algoritmos de optimización para encontrar los coeficientes que minimizan el Criterio de Información de Akaike (AIC).
3. **Validación:** Análisis de residuos para asegurar que se comporten como "ruido blanco" (sin patrones remanentes).

Este modelo es fundamental para capturar el comportamiento cíclico observado en el recaudo de Rentas Cedidas de Quibdó.
