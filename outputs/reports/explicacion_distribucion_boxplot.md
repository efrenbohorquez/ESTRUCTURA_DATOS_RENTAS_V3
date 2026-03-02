# 📊 Explicación: Distribución y Boxplot del Recaudo Mensual

## Gráficas Analizadas
Panel izquierdo: **Distribución del Recaudo Mensual** (histograma de densidad)
Panel derecho: **Recaudo por Año** (boxplot comparativo 2021–2025)

---

## Panel Izquierdo — Histograma de Distribución

### ¿Qué muestra?
La frecuencia relativa (densidad) de los valores de recaudo mensual. Cada barra representa un rango de valores y su altura indica qué tan común es observar esa magnitud de recaudo.

### Hallazgos

| Característica | Valor Aproximado | Interpretación |
|---|---|---|
| **Moda** | ~$200–220 mil millones | El recaudo más frecuente está en este rango |
| **Rango total** | $150MM – $470MM | Amplitud de $320MM entre mínimo y máximo |
| **Forma** | Asimetría positiva (cola derecha) | Hay meses con recaudo extraordinariamente alto |

### Asimetría Positiva (sesgo a la derecha)
- La **mayoría** de los meses recaudan entre **$180MM y $260MM** (concentración a la izquierda)
- Existe una **cola derecha larga** hasta $470MM — son los meses de **Enero y Julio** con picos estacionales
- Esta distribución **no es normal** → confirma que la serie tiene estacionalidad fuerte

### Implicaciones para el Modelado
- Los modelos que asumen normalidad (SARIMA puro) pueden subestimar los picos
- Una **transformación logarítmica** podría normalizar la distribución
- XGBoost y LSTM son más robustos a distribuciones asimétricas

---

## Panel Derecho — Boxplot por Año

### ¿Qué muestra?
La distribución del recaudo para cada año: mediana (línea central), rango intercuartílico (caja), bigotes (valores típicos) y outliers (círculos).

### Lectura año por año

| Año | Mediana (~) | Dispersión | Outliers | Interpretación |
|---|---|---|---|---|
| **2021** | ~$220MM | Baja (caja compacta) | Ninguno | Solo Oct-Dic, pocos datos, comportamiento estable |
| **2022** | ~$220MM | Moderada | Ninguno | Primer año completo; recaudo estable con variación estacional |
| **2023** | ~$230MM | **Alta** (caja más ancha) | 1 alto (~$375MM) | Mayor dispersión por IPC alto (9.28%) e incremento salarial (16%) |
| **2024** | ~$230MM | **Muy alta** | 1 alto (~$410MM) | Máxima dispersión — picos de Ene/Jul cada vez más pronunciados |
| **2025** | ~$250MM | **Extrema** | 2 altos (~$430MM, $470MM) | Los valores atípicos son los picos estacionales que crecen año tras año |

### Patrones clave del Boxplot

#### 1. Mediana estable, dispersión creciente
La mediana crece **lentamente** (~$220MM→$250MM), pero la **dispersión se amplifica** cada año. Esto significa:
- Los meses "normales" suben poco
- Los meses pico (Ene/Jul) suben **mucho más rápido** que el promedio
- Se confirma **heterocedasticidad** (varianza no constante en el tiempo)

#### 2. Los outliers son los picos estacionales
Los círculos marcados como "outliers" en 2023–2025 **no son anomalías** — son los meses de Enero y Julio donde el recaudo se dispara por:
- Renovación de impuestos e indexación de tarifas al IPC
- Efecto de primas salariales de mitad de año
- Estos "outliers" son **predecibles** y deben modelarse, no eliminarse

#### 3. El bigote inferior baja cada año
El valor mínimo de recaudo tiende a mantenerse bajo (~$175–190MM), indicando que los meses de menor recaudo (Feb, Ago, Sep) no crecen proporcionalmente.

---

## Conclusiones Combinadas

> [!IMPORTANT]
> **Hallazgo principal:** El recaudo de Rentas Cedidas presenta una distribución cada vez más dispersa, con una brecha creciente entre los meses pico (Ene/Jul) y los meses valle (Feb/Ago). Esto sugiere que el efecto de indexación anual al IPC impacta desproporcionadamente los meses de renovación tributaria.

### Recomendaciones para la Tesis

1. **No eliminar outliers** — son picos estacionales predecibles, no errores
2. **Considerar transformación logarítmica** para estabilizar la varianza antes de SARIMA
3. **Variables dummy para Ene/Jul** en SARIMAX y Prophet para capturar los picos
4. **Reportar el coeficiente de variación (CV)** por año como evidencia de heterocedasticidad creciente

---

*Documento generado como parte del Sistema de Análisis de Rentas Cedidas — Febrero 2026*
