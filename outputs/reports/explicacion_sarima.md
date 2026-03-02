# 📈 Documentación del Modelo SARIMA — Pronóstico de Rentas Cedidas

**Proyecto de Tesis de Maestría**
**Municipio de Quibdó — Rentas Cedidas (2021–2025)**

---

## 1. Introducción

El modelo SARIMA (*Seasonal AutoRegressive Integrated Moving Average*) constituye el **modelo econométrico base** (benchmark) del proyecto. Su propósito es doble:

1. **Pronóstico**: Generar predicciones de recaudo mensual de rentas cedidas
2. **Línea base**: Servir como punto de comparación para modelos más avanzados (SARIMAX, Random Forest, XGBoost, LSTM)

### ¿Por qué SARIMA para rentas municipales?

El recaudo tributario municipal presenta tres características que SARIMA captura de forma natural:

| Característica | Componente SARIMA | Ejemplo en Quibdó |
|---------------|-------------------|-------------------|
| **Tendencia** | Diferenciación regular ($d$) | Crecimiento por inflación y economía |
| **Estacionalidad** | Componentes estacionales ($P,D,Q$) | Picos en enero (plazos fiscales) y diciembre |
| **Autocorrelación** | Componentes AR y MA ($p,q$) | Meses consecutivos están relacionados |

---

## 2. Formulación Matemática

### 2.1 Notación SARIMA$(p,d,q)(P,D,Q)_m$

El modelo se compone de **7 parámetros**:

| Parámetro | Nombre | Descripción |
|-----------|--------|-------------|
| $p$ | Orden AR (autorregresivo) | Nº de rezagos de la serie usados como predictores |
| $d$ | Orden de diferenciación | Nº de veces que se diferencia la serie para hacerla estacionaria |
| $q$ | Orden MA (media móvil) | Nº de rezagos del error usados como predictores |
| $P$ | Orden SAR (AR estacional) | Igual que $p$, pero a nivel estacional (cada $m$ periodos) |
| $D$ | Orden de diferenciación estacional | Diferenciación para eliminar estacionalidad |
| $Q$ | Orden SMA (MA estacional) | Igual que $q$, pero a nivel estacional |
| $m$ | Periodo estacional | Longitud del ciclo (12 para datos mensuales) |

### 2.2 Ecuación General

Usando el **operador de retardo** $B$ (donde $B^k y_t = y_{t-k}$):

$$\phi_p(B) \cdot \Phi_P(B^m) \cdot (1-B)^d \cdot (1-B^m)^D \cdot y_t = \theta_q(B) \cdot \Theta_Q(B^m) \cdot \varepsilon_t$$

**Desglose de cada componente:**

- **Polinomio AR**: $\phi_p(B) = 1 - \phi_1 B - \phi_2 B^2 - \cdots - \phi_p B^p$
  - Captura la dependencia lineal del valor actual con sus valores pasados recientes
  
- **Polinomio SAR**: $\Phi_P(B^m) = 1 - \Phi_1 B^m - \Phi_2 B^{2m} - \cdots - \Phi_P B^{Pm}$
  - Captura la dependencia con el mismo mes de años anteriores
  
- **Diferenciación regular**: $(1-B)^d$
  - Elimina la tendencia. Con $d=1$: $y_t' = y_t - y_{t-1}$ (primera diferencia)
  
- **Diferenciación estacional**: $(1-B^m)^D$
  - Elimina la estacionalidad. Con $D=1$, $m=12$: $y_t'' = y_t - y_{t-12}$
  
- **Polinomio MA**: $\theta_q(B) = 1 + \theta_1 B + \theta_2 B^2 + \cdots + \theta_q B^q$
  - Captura el efecto de shocks pasados (errores anteriores)
  
- **Polinomio SMA**: $\Theta_Q(B^m) = 1 + \Theta_1 B^m + \Theta_2 B^{2m} + \cdots + \Theta_Q B^{Qm}$
  - Captura el efecto de shocks estacionales pasados
  
- **Ruido blanco**: $\varepsilon_t \sim \mathcal{N}(0, \sigma^2)$ — innovaciones independientes

### 2.3 Ejemplo: SARIMA(1,1,1)(1,1,1)₁₂

Para el caso específico con datos mensuales:

$$(1 - \phi_1 B)(1 - \Phi_1 B^{12})(1-B)(1-B^{12}) y_t = (1 + \theta_1 B)(1 + \Theta_1 B^{12}) \varepsilon_t$$

**Interpretación práctica**: El recaudo del mes actual depende de:
- El recaudo del **mes anterior** (componente $\phi_1$)
- El recaudo del **mismo mes del año anterior** (componente $\Phi_1$)
- El error de pronóstico del **mes anterior** (componente $\theta_1$)
- El error de pronóstico del **mismo mes del año anterior** (componente $\Theta_1$)

---

## 3. Metodología de Estimación

### 3.1 Selección del Orden — Criterio de Información de Akaike (AIC)

El orden óptimo se selecciona minimizando el AIC:

$$\text{AIC} = -2 \ln(\hat{L}) + 2k$$

donde:
- $\hat{L}$ = valor máximo de la función de verosimilitud
- $k$ = número total de parámetros estimados

**Criterio de Información Bayesiano (BIC)** — más conservador:

$$\text{BIC} = -2 \ln(\hat{L}) + k \ln(n)$$

| Criterio | Penalización | Tiende a... |
|----------|-------------|-------------|
| AIC | $2k$ | Modelos más complejos (menor sesgo) |
| BIC | $k \ln(n)$ | Modelos más simples (mayor parsimonia) |

El algoritmo `auto_arima` de `pmdarima` usa búsqueda **stepwise** para encontrar eficientemente el mejor modelo sin evaluar todas las combinaciones posibles.

### 3.2 Estimación de Parámetros — Máxima Verosimilitud (MLE)

Los parámetros se estiman maximizando la **log-verosimilitud**:

$$\ell(\boldsymbol{\theta}) = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2}\sum_{t=1}^n \varepsilon_t^2$$

Se utiliza la representación en **espacio de estados** con el **filtro de Kalman** para calcular eficientemente la verosimilitud, manejar valores iniciales y permitir predicciones multistep.

### 3.3 Partición Temporal (Train/Test)

Para series de tiempo, la validación respeta el orden temporal:

$$\text{Train} = \{y_1, \ldots, y_T\}, \quad \text{Test} = \{y_{T+1}, \ldots, y_n\}$$

- **Regla empírica**: El entrenamiento debe contener al menos 3 ciclos completos (36 meses para $m=12$)
- **Prohibido**: Validación cruzada aleatoria (violaría la causalidad temporal)

---

## 4. Diagnóstico de Residuos

### 4.1 Condiciones de un buen modelo

Si el modelo es adecuado, los residuos $\hat{\varepsilon}_t = y_t - \hat{y}_t$ deben ser **ruido blanco gaussiano**:

1. **Media cero**: $E[\hat{\varepsilon}_t] = 0$
2. **Varianza constante**: $\text{Var}(\hat{\varepsilon}_t) = \sigma^2$ (homocedasticidad)
3. **Sin autocorrelación**: $\text{Cov}(\hat{\varepsilon}_t, \hat{\varepsilon}_{t-k}) = 0, \;\forall k \neq 0$
4. **Normalidad**: $\hat{\varepsilon}_t \sim \mathcal{N}(0, \sigma^2)$

### 4.2 Test de Ljung-Box

Evalúa la autocorrelación conjunta hasta el rezago $h$:

$$Q_{LB}(h) = n(n+2)\sum_{k=1}^{h}\frac{\hat{\rho}_k^2}{n-k}$$

| Hipótesis | Descripción |
|-----------|-------------|
| $H_0$ | Los residuos son independientes (no autocorrelación) |
| $H_1$ | Existe autocorrelación en al menos un rezago |
| Distribución bajo $H_0$ | $Q_{LB} \sim \chi^2_{h-p-q}$ |
| **Decisión** | Si $p > 0.05$ → no se rechaza $H_0$ → modelo adecuado ✅ |

Se evalúa en los rezagos 6, 12 y 24 para cubrir medio año, un año y dos años.

### 4.3 Test de Jarque-Bera

Evalúa normalidad usando asimetría ($S$) y curtosis en exceso ($K-3$):

$$JB = \frac{n}{6}\left(S^2 + \frac{(K-3)^2}{4}\right)$$

| Hipótesis | Descripción |
|-----------|-------------|
| $H_0$ | Los residuos provienen de una distribución normal |
| $H_1$ | Los residuos NO son normales |
| **Nota** | Para $n > 30$, el modelo sigue siendo válido por el **Teorema Central del Límite** |

### 4.4 Gráficos de Diagnóstico

El panel de 4 gráficos de `statsmodels` incluye:

1. **Residuos estandarizados vs tiempo**: Detección visual de heterocedasticidad, tendencia residual y outliers
2. **Histograma + densidad KDE**: Evaluación visual de normalidad y sesgo
3. **Q-Q Plot normal**: Comparación de cuantiles empíricos vs teóricos; desviaciones en las colas indican no normalidad
4. **Correlograma de residuos**: Bandas de confianza para detectar autocorrelación residual significativa

---

## 5. Métricas de Evaluación

### 5.1 Métricas In-Sample vs Out-of-Sample

| Tipo | Datos usados | Propósito |
|------|-------------|-----------|
| **In-Sample** | Entrenamiento | Evaluar qué tan bien el modelo captura patrones históricos |
| **Out-of-Sample** | Test | Evaluar capacidad predictiva real (lo más importante) |

### 5.2 Definición de Métricas

**RMSE** — Raíz del Error Cuadrático Medio:
$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{t=1}^n(y_t - \hat{y}_t)^2}$$
- Mismas unidades que $y$ (pesos colombianos)
- Penaliza más los errores grandes (por el cuadrado)

**MAE** — Error Absoluto Medio:
$$\text{MAE} = \frac{1}{n}\sum_{t=1}^n |y_t - \hat{y}_t|$$
- Interpretación directa: "en promedio, el pronóstico se desvía en $X pesos"

**MAPE** — Error Porcentual Absoluto Medio:
$$\text{MAPE} = \frac{100}{n}\sum_{t=1}^n \left|\frac{y_t - \hat{y}_t}{y_t}\right|$$
- Independiente de la escala, permite comparar entre series

| MAPE | Calidad del pronóstico |
|------|----------------------|
| < 10% | Excelente |
| 10–20% | Bueno |
| 20–50% | Razonable |
| > 50% | Impreciso |

**R²** — Coeficiente de Determinación:
$$R^2 = 1 - \frac{\sum(y_t - \hat{y}_t)^2}{\sum(y_t - \bar{y})^2}$$
- Proporción de varianza explicada por el modelo
- $R^2 = 1$: ajuste perfecto; $R^2 = 0$: equivalente a predicir la media

### 5.3 Intervalos de Confianza

$$\hat{y}_{T+h} \pm z_{\alpha/2} \cdot \hat{\sigma}_h$$

donde:
- $z_{0.025} \approx 1.96$ para IC al 95%
- $\hat{\sigma}_h^2 = \sigma^2 \sum_{j=0}^{h-1} \psi_j^2$ (crece con $h$)

Los $\psi_j$ son los coeficientes de la representación MA($\infty$) del proceso. Esto explica por qué los intervalos de confianza **se ensanchan** conforme aumenta el horizonte de pronóstico.

---

## 6. Pronóstico Futuro

### 6.1 Estrategia de Re-estimación

Para el pronóstico operativo de planeación fiscal:

1. Se **re-ajusta** el modelo con **toda la serie disponible** (train + test)
2. Se generan pronósticos para los próximos `HORIZONTE_PRONOSTICO` meses
3. Se incluyen intervalos de confianza al 95%

**Justificación**: Al usar toda la información disponible, los parámetros estimados son más precisos y los pronósticos más robustos.

### 6.2 Limitaciones del Pronóstico Futuro

- La **incertidumbre crece** con el horizonte (IC más amplios)
- SARIMA asume que los **patrones pasados se repiten** — no captura cambios estructurales
- No incorpora **variables exógenas** (inflación, población, política fiscal) → ver SARIMAX

---

## 7. Archivos Generados

| Archivo | Directorio | Contenido |
|---------|-----------|-----------|
| `04_sarima_insample.png` | `outputs/figures/` | Gráfico de ajuste in-sample |
| `04_sarima_diagnostico.png` | `outputs/figures/` | Panel de diagnóstico de residuos |
| `04_sarima_pronostico.png` | `outputs/figures/` | Pronóstico test vs real |
| `04_sarima_futuro.png` | `outputs/figures/` | Pronóstico futuro con IC |
| `sarima_forecast.csv` | `outputs/forecasts/` | Pronósticos test con IC |
| `sarima_futuro.csv` | `outputs/forecasts/` | Pronósticos futuros con IC |
| `sarima_metricas.csv` | `outputs/reports/` | Métricas (RMSE, MAE, MAPE, R², AIC, BIC) |

---

## 8. Conclusiones y Siguiente Paso

El modelo SARIMA proporciona una **línea base sólida** para el pronóstico de rentas cedidas municipales. Sus fortalezas:

- ✅ Captura estacionalidad mensual de forma natural
- ✅ Interpretable (coeficientes tienen significado económico)
- ✅ Intervalos de confianza paramétricos bien fundamentados
- ✅ Selección automática de orden con criterios de información

Sus limitaciones motivan el uso de modelos complementarios:

- ⚠️ No incorpora variables macroeconómicas → **SARIMAX** (notebook 05)
- ⚠️ Relaciones lineales únicamente → **Random Forest / XGBoost** (notebooks 06–07)
- ⚠️ No captura patrones no lineales complejos → **LSTM** (notebook 08)

---

*Documento generado como parte del proyecto de tesis de Maestría — Análisis de Rentas Cedidas, Municipio de Quibdó.*
