# Modelo de Pronóstico: El "Relojero" (Facebook Prophet)

El modelo **Prophet**, desarrollado por Facebook, aborda el problema del recaudo no con fuerza bruta, sino con la precisión de un **relojero**. Su filosofía consiste en desarmar cuidadosamente la serie de tiempo en sus piezas fundamentales: tendencia, estacionalidad y efectos de calendario.

## 1. Desglose de Componentes
- **Tendencia a Largo Plazo:** Identifica si el recaudo departamental está en una trayectoria creciente o decreciente, independientemente del ruido mensual.
- **Estacionalidad (El Pulso de la Salud):** Captura con precisión los picos de diciembre por consumo festivo y los valles de inicio de año.
- **Efectos de Calendario:** Modela el impacto de los puentes festivos y eventos especiales de Colombia.

## 2. Ventajas Estratégicas y Sustentación
*   **Interpretabilidad:** A diferencia de las "cajas negras", Prophet permite que un tomador de decisiones gubernamental entienda exactamente *por qué* el modelo predice un aumento o caída (referencia: "Artesanía y Pragmatismo en Gestión Pública").
*   **Robustez:** Maneja de forma excelente los cambios de régimen (*changepoints*) causados por reformas tributarias o eventos macroeconómicos.
*   **Precisión Equilibrada:** Con un MAPE de **14.72%**, ofrece un balance óptimo entre precisión predictiva y transparencia administrativa.

## 3. Configuración para Rentas Cedidas
Se seleccionó una estacionalidad **multiplicativa**, reconociendo que la amplitud de las variaciones del recaudo (especialmente en diciembre) crece proporcionalmente con el nivel total de la serie.

