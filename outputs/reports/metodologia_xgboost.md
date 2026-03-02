# Metodología: Modelo XGBoost para Series Temporales

**XGBoost** (*Extreme Gradient Boosting*) es un algoritmo de aprendizaje supervisado basado en árboles de decisión que utiliza la técnica de *gradient boosting* para obtener predicciones precisas.

## 1. El "Comité de Expertos"
XGBoost funciona como un comité de especialistas. En lugar de una sola predicción, construye cientos de árboles de decisión en secuencia. Cada nuevo árbol se enfoca exclusivamente en corregir los errores que cometió el anterior, logrando así captar relaciones súper complejas y no lineales que un analista humano pasaría por alto. Este enfoque permitió lograr la mayor precisión del sistema (MAPE ~14%).

## 2. Enfoque de Aprendizaje Supervisado
A diferencia de los modelos estadísticos tradicionales (como SARIMA), XGBoost no entiende la noción de "tiempo" intrínsecamente. Por ello, transformamos la serie en un problema de aprendizaje supervisado mediante:
- **Lags (Retardos):** Usar valores de meses anteriores ($t-1, t-2, ...$). El análisis reveló que el **Lag 12** (mismo mes del año anterior) es la variable más decisiva, confirmando que el comportamiento humano estacional (ej. Navidad) es el motor principal del recaudo.
- **Variables de Tiempo:** Extracción de mes, año y trimestre para capturar estacionalidad.
- **Variables Macro:** Incorporación de IPC, Salario Mínimo y UPC como factores externos.

## 3. Ventajas del Modelo
- **No Linealidad:** Capaz de capturar relaciones complejas entre las variables macro y el recaudo.
- **Manejo de Outliers:** Alta robustez frente a valores atípicos mediante regularización.
- **Importancia de Variables:** Permite cuantificar qué factores tienen mayor peso en la predicción.


## 3. Entrenamiento y Ajuste
Se utiliza una validación cruzada de ventana expandida (*Time Series Split*) para evitar el *data leakage* (uso de información futura para predecir el pasado) y optimizar hiperparámetros como la tasa de aprendizaje (*learning rate*) y la profundidad de los árboles.
