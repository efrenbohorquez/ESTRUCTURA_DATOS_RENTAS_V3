# Evidencia Estratégica y Soporte Académico

Este documento consolida los hallazgos cuantitativos del proyecto y los vincula con las referencias estratégicas y estudios de soporte que fundamentan el sistema STAR.

## 📊 Porcentajes y Métricas Clave

| Métrica | Valor | Hallazgo Estratégico |
| :--- | :--- | :--- |
| **Volatilidad (Std Dev)** | **42%** | Las Rentas Cedidas son 2.3x más inestables que las participaciones nacionales (SGP). |
| **Concentración (Top 5)** | **48%** | El financiamiento de la salud colombiana depende de un "hilo" en 5 departamentos líderes. |
| **Importancia Lag 12** | **65%** | El recaudo es un fenómeno de "memoria anual". El mejor predictor es el mismo mes del año anterior. |
| **Error (ADRES vs STAR)** | **-10%** | Reducción del error proyectado del 25% (lineal) al 14-15% (ML/Sistema STAR). |
| **Asimetría Extrema** | **0.04%** | La mitad de los municipios no tienen capacidad real de financiamiento propio para salud. |

## � Variables Macroeconómicas — Verificación y Fuentes Oficiales

### IPC — Variación Anual (dic-dic)
| Año | Valor (%) | Fuente | Resolución / Referencia |
|:---:|:---------:|:------:|:------------------------|
| 2021 | 5.62 | DANE | Boletín Técnico IPC dic-2021 |
| 2022 | 13.12 | DANE | Boletín Técnico IPC dic-2022 |
| 2023 | 9.28 | DANE | Boletín Técnico IPC dic-2023 |
| 2024 | **5.20** | DANE | Boletín Técnico IPC dic-2024 |
| 2025 | **5.10** | DANE | Boletín Técnico IPC dic-2025 |
| 2026 | 4.00 | BanRep | Proyección (meta 3%±1pp) |

**URL fuente:** https://www.dane.gov.co/index.php/estadisticas-por-tema/precios-y-costos/indice-de-precios-al-consumidor-ipc
**Ref. cruzada:** Wikipedia "Anexo:Indicadores económicos de Colombia"

### Salario Mínimo Legal Mensual Vigente (SMLMV)
| Año | SMLMV (COP) | Aumento (%) | Decreto / Fuente |
|:---:|:-----------:|:-----------:|:-----------------|
| 2021 | $908,526 | 3.50 | Decreto dic-2020, Diario Oficial |
| 2022 | $1,000,000 | 10.07 | Decreto dic-2021, Diario Oficial |
| 2023 | $1,160,000 | 16.00 | Decreto dic-2022, Diario Oficial |
| 2024 | $1,300,000 | **12.07** | Decreto dic-2023, Diario Oficial |
| 2025 | $1,423,500 | **9.54** | Decreto 24-dic-2024, Diario Oficial |
| 2026 | $1,750,905 | **23.00** | Decreto 30-dic-2025, Presidencia.gov.co |

**Nota 2026:** El Gobierno creó el concepto "Salario mínimo vital" = $2,000,000 (SMLMV + auxilio de transporte de $249,095).
**URL fuente:** https://es.wikipedia.org/wiki/Salario_m%C3%ADnimo_en_Colombia (verificado con decretos oficiales)

### UPC — Unidad de Pago por Capitación (% incremento anual)
| Año | Valor (%) | Fuente | Estado |
|:---:|:---------:|:------:|:-------|
| 2021 | 5.00 | MinSalud Res. 2503/2020 | ✅ Verificado |
| 2022 | 5.42 | MinSalud Res. 2481/2021 | ✅ Verificado |
| 2023 | 16.23 | MinSalud Res. 2808/2022 | ✅ Verificado |
| 2024 | 12.01 | MinSalud Res. 2807/2023 | ✅ Verificado |
| 2025 | 8.00 | MinSalud Res. 1879/2024 | ⚠️ Pendiente verificación |
| 2026 | 7.00 | MinSalud (estimado) | ⚠️ Pendiente verificación |

**Nota UPC 2025–2026:** Los valores exactos no pudieron ser verificados dado que la fuente especializada (consultorsalud.com) requiere suscripción de pago. Se recomienda verificar contra la Resolución MinSalud vigente.

### Consumo de los Hogares — Variación Anual Real (%)
| Año | Valor (%) | Fuente | Estado |
|:---:|:---------:|:------:|:-------|
| 2021 | 14.72 | Banco Mundial / DANE Cuentas Nacionales | ✅ Confirmado (WB API, lastupdated 2026-02-24) |
| 2022 | 10.79 | Banco Mundial / DANE Cuentas Nacionales | ✅ Confirmado |
| 2023 | 0.38 | Banco Mundial / DANE Cuentas Nacionales | ✅ Confirmado |
| 2024 | 1.60 | Banco Mundial / DANE Cuentas Nacionales | ✅ Confirmado |
| 2025 | 2.60 | Estimación = PIB 2025pr (DANE, 16-feb-2026) | ⚠️ Estimado — WB sin dato |
| 2026 | 2.50 | Promedio de proyecciones | ⚠️ Proyección — Año en curso |

**Indicador World Bank:** NE.CON.PRVT.KD.ZG — *Households and NPISHs Final consumption expenditure (annual % growth)*
**API:** `https://api.worldbank.org/v2/country/COL/indicator/NE.CON.PRVT.KD.ZG?format=json&date=2019:2024`
**Fuente primaria:** DANE — Cuentas Nacionales Trimestrales, PIB por enfoque del gasto.
**Contexto:** El consumo de hogares representó ~68% del PIB colombiano en 2024. La fuerte caída post-pandémica (2020: −5.01%) fue seguida por un rebote excepcional (2021: +14.72%) y posterior normalización.

---

## �📚 Referencias y Estudios de Soporte

### 1. Marco Institucional: La Fragilidad del Régimen Subsidiado
*   **Referencia:** *Informes de la ADRES sobre el financiamiento de la UPC.*
*   **Sustentación:** Se evidencia que las Rentas Cedidas no son solo una cifra fiscal, sino el flujo de caja para la atención de **24 millones de personas**. El sistema STAR actúa como un "Monitor de Signos Vitales" para este flujo.

### 2. Metodología de Pronóstico: El Comité de Expertos (XGBoost)
*   **Referencia:** *Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.*
*   **Sustentación:** La arquitectura del "Comité de Expertos" implementada se sustenta en la robustez del gradiente boosting para manejar valores atípicos (outliers) causados por la estacionalidad de licores y cigarrillos.

### 3. Teoría del Tiempo: El Relojero (Prophet)
*   **Referencia:** *Taylor, S. J., & Letham, B. (2018). Forecasting at Scale.*
*   **Sustentación:** La descomposición aditiva/multiplicativa permite al administrador público "ver debajo del capó" del recaudo, separando el crecimiento real (Tendencia) del ruido estacional.

### 4. Pragmática de Datos: El Ferrari (LSTM)
*   **Referencia:** *Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.*
*   **Sustentación:** La lección del "Ferrari sin combustible" se alinea con la literatura de *Small Data Learning*, donde la sintonización experta (tuning) es más valiosa que la complejidad computacional pura.

## 💡 Insight de la Red (Podcast)
"La ciencia de datos en el sector público no se trata de reemplazar al político, sino de darle el mapa para que no navegue a ciegas." – Esta premisa fundamenta el **Motor de Recomendaciones** del sistema STAR, que traduce el % de desviación en planes de acción tácticos.
