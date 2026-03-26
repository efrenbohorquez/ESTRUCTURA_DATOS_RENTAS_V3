# 📊 Análisis de Estacionalidad del Recaudo de Rentas Cedidas

**Documento de soporte para tesis de maestría**
*Periodo: Octubre 2021 – Diciembre 2025*

---

## 1. Descripción de las Gráficas

### Panel Izquierdo: "Recaudo Promedio por Mes (Estacionalidad)"

Gráfico de barras que muestra el **promedio histórico del recaudo neto** para cada mes del año, calculado sobre todos los años del periodo (2021–2025).

- **Eje X:** Meses del año (Ene–Dic)
- **Eje Y:** Recaudo Neto en pesos colombianos (escala ×10¹¹, es decir, centenas de miles de millones)
- **Barras rojas:** Meses pico (Enero y Julio)
- **Barras azules:** Meses regulares
- **Línea punteada gris:** Media general (~$260.000 millones)

### Panel Derecho: "Patrón Estacional por Año"

Gráfico de líneas superpuestas que muestra el **perfil mensual de recaudo para cada año individual**, permitiendo ver la evolución y consistencia del patrón estacional.

- **Cada línea** representa un año: 2021 (rosa), 2022 (naranja), 2023 (verde), 2024 (cian), 2025 (azul)
- **Eje X:** Meses del año
- **Eje Y:** Recaudo Neto (misma escala ×10¹¹)

---

## 2. Hallazgos Clave

### 2.1 Dos picos dominantes: Enero y Julio

| Mes Pico | Recaudo Promedio | vs. Media General | Causa Probable |
|----------|-----------------|-------------------|----------------|
| **Enero** | ~$375.000M | **+44%** por encima | Vencimiento de impuestos anuales, renovación de licencias, indexación IPC año nuevo |
| **Julio** | ~$375.000M | **+44%** por encima | Segundo semestre fiscal, renovación semestral de permisos, pagos semestrales |

> **Hallazgo:** Los dos picos son **simétricos** — prácticamente igual magnitud cada 6 meses, confirmando un ciclo **semestral** (s=6) dentro del ciclo anual (s=12).

### 2.2 Dos valles pronunciados: Abril–Mayo y Septiembre

| Mes Valle | Recaudo Promedio | vs. Media General | Interpretación |
|-----------|-----------------|-------------------|----------------|
| **Abril** | ~$175.000M | **-33%** por debajo | Pos-pico trimestral, menor actividad fiscal |
| **Mayo** | ~$190.000M | **-27%** por debajo | Continuación del valle Q2 |
| **Septiembre** | ~$175.000M | **-33%** por debajo | Valle pos-julio, ciclo simétrico con abril |

### 2.3 Meses de transición

Los meses **Febrero–Marzo**, **Junio**, **Agosto** y **Octubre–Diciembre** oscilan entre $200.000M y $260.000M, formando una "meseta" alrededor de la media.

### 2.4 Evolución interanual (Panel Derecho)

| Observación | Detalle |
|-------------|---------|
| **2022 (naranja)** es el año más plano | Menor amplitud estacional, recaudo estable ~$200.000M |
| **2023 (verde)** introduce los picos | Enero y Julio empiezan a crecer significativamente |
| **2024 (cian)** amplifica el patrón | Julio 2024 alcanza ~$400.000M, el pico más alto hasta ese año |
| **2025 (azul)** rompe récords | Enero 2025 llega a ~$470.000M y Julio 2025 a ~$410.000M |
| **Amplificación creciente** | La brecha pico-valle se **expande cada año**, no es constante |

### 2.5 Heterocedasticidad confirmada

```
Amplitud pico-valle por año:
  2022: ~$150.000M   (rango estrecho)
  2023: ~$200.000M   (crece)
  2024: ~$250.000M   (crece más)
  2025: ~$320.000M   (máximo histórico)
```

> **Implicación:** La varianza del recaudo **NO es constante** — crece con el nivel. Esto es **heterocedasticidad**, y tiene consecuencias directas para los modelos.

---

## 3. Patrón Estacional Formalizado

```
    Ene     Feb     Mar     Abr     May     Jun     Jul     Ago     Sep     Oct     Nov     Dic
    ▲▲▲     ──      ──      ▼▼      ▼▼      ──      ▲▲▲     ──      ▼▼      ──      ──      ──
    PICO    TRANS   TRANS   VALLE   VALLE   TRANS   PICO    TRANS   VALLE   TRANS   TRANS   TRANS
```

**Frecuencia dominante:** Semestral (cada 6 meses), con subarmonía anual.

---

## 4. Recomendaciones para los Modelos de Pronóstico

### 4.1 Configuración del componente estacional

| Modelo | Recomendación | Justificación |
|--------|--------------|---------------|
| **SARIMAX** | Usar `s=12` con `D=1` | Diferenciación estacional anual captura el ciclo completo |
| **SARIMAX** | Añadir variable dummy `mes_pico` (Ene=1, Jul=1) | Refuerza los picos como efecto exógeno |
| **Prophet** | Configurar `seasonality_mode='multiplicative'` | La amplitud crece con el nivel → estacionalidad multiplicativa, NO aditiva |
| **XGBoost** | Crear features: `mes_sin`, `mes_cos`, `es_enero`, `es_julio` | Captura el ciclo + los picos explícitamente |
| **LSTM** | lookback=12 meses mínimo | Necesita ver un ciclo completo para aprender el patrón |

### 4.2 Tratamiento de la heterocedasticidad

> [!IMPORTANT]
> **La varianza creciente es el hallazgo más relevante para los modelos.**

| Estrategia | Aplicación |
|------------|-----------|
| **Transformación log** | Aplicar `log(Recaudo)` antes de modelar para estabilizar varianza |
| **Estacionalidad multiplicativa** | Prophet: `seasonality_mode='multiplicative'` |
| **Intervalos de confianza dinámicos** | Los IC deben **ensancharse** en Ene/Jul y **estrecharse** en meses valle |
| **Evaluación por MAPE** | Preferir MAPE sobre RMSE; RMSE penaliza excesivamente los picos |

### 4.3 Tratamiento de los picos en el set de Test (Oct–Dic 2025)

El set de test reservado (**Oct, Nov, Dic 2025**) corresponde a meses de **transición** (no picos). Esto tiene implicaciones:

| Factor | Impacto |
|--------|---------|
| **Ventaja:** Evalúa precisión en meses "normales" | Los modelos deben acertar valores medios |
| **Limitación:** No prueba la predicción de picos | No se valida Ene/Jul donde el error tiende a ser mayor |
| **Recomendación:** Complementar con cross-validation temporal | Usar walk-forward validation para evaluar también los picos |

### 4.4 Variables derivadas sugeridas

```python
# Features para capturar la estacionalidad observada
df['mes_pico'] = df.index.month.isin([1, 7]).astype(int)        # Pico binario
df['mes_valle'] = df.index.month.isin([4, 5, 9]).astype(int)    # Valle binario
df['ciclo_semestral'] = np.sin(2 * np.pi * df.index.month / 6)  # Armonía s=6
df['ciclo_anual'] = np.sin(2 * np.pi * df.index.month / 12)     # Armonía s=12
df['tendencia_amplitud'] = df['año'] - 2021                     # Proxy de heterocedasticidad
```

---

## 5. Conclusión para la Tesis

La serie de recaudo de Rentas Cedidas presenta un **patrón estacional doble** (s=6 y s=12) con **amplificación progresiva** de los picos. Este comportamiento:

1. **Descarta modelos lineales simples** — la varianza no constante viola los supuestos de homogeneidad
2. **Favorece modelos multiplicativos** — Prophet multiplicativo y transformaciones log
3. **Justifica el ensemble** — ningún modelo individual captura bien tanto los picos como los valles
4. **Exige validación especial** — el test Oct–Dic 2025 es conservador; se recomienda complementar con walk-forward validation que incluya meses pico

> [!TIP]
> **Cita recomendada para la tesis:**
> "La serie presenta estacionalidad multiplicativa con periodo dominante s=6 (semestral) y s=12 (anual), donde la amplitud pico-valle crece interanualmente de $150.000M (2022) a $320.000M (2025), confirmando heterocedasticidad que requiere transformación logarítmica o modelos multiplicativos para pronóstico robusto."
