# 📊 Explicación de la Gráfica: Serie de Tiempo del Recaudo Mensual

## Gráfica Analizada
**Título:** "Serie de Tiempo: Recaudo Mensual de Rentas Cedidas (Oct 2021 – Dic 2025)"

---

## ¿Qué muestra la gráfica?

La gráfica presenta la evolución del **recaudo neto mensual** de Rentas Cedidas para Departamentos y Distritos de Colombia, con 4 elementos visuales:

| Elemento | Descripción |
|---|---|
| 🔵 **Línea azul (Recaudo Mensual)** | Valor real del recaudo agregado mes a mes |
| 🔴 **Línea roja punteada (MA-6)** | Media móvil de 6 meses — suaviza la volatilidad para mostrar la tendencia subyacente |
| ⬜ **Línea gris punteada (Media $256.44MM)** | Promedio histórico de todo el periodo como línea de referencia |
| 🔺 **Triángulos rojos (Picos Ene/Jul)** | Marcan los meses de Enero y Julio donde históricamente se dispara el recaudo |

---

## Patrones Clave Identificados

### 1. Estacionalidad fuerte (ciclo semestral)
La serie muestra un **patrón repetitivo cada 6 meses** con picos pronunciados en:
- **Enero**: Inicio de vigencia fiscal, ajuste de tarifas, pago de primas y renovaciones de impuestos
- **Julio**: Segundo semestre, primas de mitad de año, ajustes salariales

Estos picos son consistentes en todos los años (2022, 2023, 2024, 2025), lo que confirma estacionalidad determinística.

### 2. Tendencia creciente
La línea de tendencia (MA-6, roja punteada) muestra un **crecimiento sostenido** del recaudo entre 2022 y mediados de 2025:
- 2022: Tendencia estable alrededor de $240-250MM
- 2023: Leve aumento a $260-270MM
- 2024-2025: Aceleración hasta $290-300MM

Este crecimiento se explica por la **indexación anual de tarifas** al IPC y al incremento del salario mínimo.

### 3. Volatilidad creciente (efecto abanico)
La amplitud de las oscilaciones **aumenta con el tiempo**:
- 2022: Picos de ~$300MM, valles de ~$180MM (amplitud ~$120MM)
- 2025: Picos de ~$470MM, valles de ~$210MM (amplitud ~$260MM)

Esto indica **heterocedasticidad** — la varianza no es constante. Es importante para el modelado: los modelos SARIMAX asumen varianza constante, mientras que LSTM y XGBoost son más robustos a este efecto.

---

## ⚠️ ¿Por qué "cae" la curva al final (2025-2026)?

La **caída abrupta** que se observa al final de la gráfica (últimos 2-3 puntos de datos, hacia finales de 2025 — principios de 2026) se explica por **tres factores combinados**:

### Factor 1: Patrón estacional natural (la razón principal)
Después de cada pico de Julio, la serie **siempre cae** durante los meses siguientes (agosto, septiembre, octubre). Esto ocurre todos los años:
- Jul 2022 (pico ~$330MM) → Ago-Sep 2022 (~$180MM) ⬇️ **-45%**
- Jul 2023 (pico ~$350MM) → Ago-Sep 2023 (~$200MM) ⬇️ **-43%**
- Jul 2024 (pico ~$400MM) → Ago-Sep 2024 (~$190MM) ⬇️ **-53%**
- **Jul 2025 (pico ~$425MM) → Ago-Sep 2025: se observa la misma caída estacional**

> **Conclusión:** La "caída" en 2025-2026 NO indica un problema real de recaudo. Es el **ciclo estacional normal** repitiéndose. Después de esta caída, se espera una recuperación hacia Enero 2026 (próximo pico).

### Factor 2: Datos parciales o incompletos
Si los últimos meses del dataset (Nov-Dic 2025 o Ene 2026) contienen **datos parciales** (no se han registrado todas las transacciones del mes), el total mensual aparecerá artificialmente bajo.

### Factor 3: Efecto de la media móvil (MA-6)
La tendencia roja (MA-6) también desciende al final porque:
- La media móvil **pierde 3 puntos** en los extremos (ventana centrada de 6 meses)
- Los últimos valores calculados promedian datos incompletos, arrastrando la tendencia hacia abajo

---

## Interpretación para la Tesis

| Hallazgo | Implicación |
|---|---|
| Estacionalidad s=12 confirmada | Usar SARIMAX con componente estacional $(P,D,Q)_{12}$ |
| Tendencia creciente | La serie NO es estacionaria en nivel → requiere diferenciación (d≥1) |
| Heterocedasticidad | Considerar transformación logarítmica para SARIMAX; XGBoost y LSTM la manejan naturalmente |
| Picos Ene/Jul | Incluir como variables dummy o festivos en Prophet/SARIMAX |
| Caída final es estacional | **NO alarmar** — es el comportamiento esperado. Validar que los datos del último mes estén completos |

---

## Recomendación

> [!IMPORTANT]
> La caída observada al final de la serie es **comportamiento estacional esperado**, no una anomalía. Para confirmar, se recomienda verificar que el último mes de datos tenga el conteo completo de transacciones comparado con meses anteriores equivalentes.

**Fórmula de verificación:**
```
Si transacciones(mes_actual) ≈ transacciones(mismo_mes_año_anterior)
    → Dato completo, caída es estacional
Si transacciones(mes_actual) << transacciones(mismo_mes_año_anterior)
    → Dato parcial, excluir del modelado hasta completar
```

---

*Documento generado como parte del Sistema de Análisis de Rentas Cedidas — Febrero 2026*
