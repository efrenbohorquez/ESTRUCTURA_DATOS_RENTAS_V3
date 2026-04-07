import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

# Ajustar path para importar módulos locales
ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
NOTEBOOKS = ROOT / "notebooks"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS))

# Importar configuración y auxiliares
try:
    import utils
    import model_helpers
    # Cargar 00_config usando exec para que las variables estén en el namespace global si es necesario
    config_path = NOTEBOOKS / "00_config.py"
    config_globals = dict(globals())
    config_globals['__file__'] = str(config_path)
    with open(config_path, encoding='utf-8-sig') as f:
        exec(f.read(), config_globals)
    globals().update(config_globals)
except Exception as e:
    st.error(f"Error cargando módulos: {e}")

# Configuración de página Streamlit
st.set_page_config(
    page_title="Tablero de Rentas Cedidas",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo personalizado (CSS)
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    h1, h2, h3 {
        color: #1B2A4A;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("💎 Configuración")
st.sidebar.markdown("Ajuste los parámetros de los modelos para explorar diferentes escenarios.")

# Selección de Modelo
modelo_selected = st.sidebar.selectbox(
    "Seleccione Modelo",
    ["SARIMAX", "Prophet", "XGBoost", "Comparativo"]
)

# Parámetros por modelo
params = {}
if modelo_selected == "SARIMAX":
    st.sidebar.subheader("Parámetros SARIMAX")
    p = st.sidebar.slider("p (AR)", 0, 5, 1)
    d = st.sidebar.slider("d (I)", 0, 2, 1)
    q = st.sidebar.slider("q (MA)", 0, 5, 1)
    P = st.sidebar.slider("P (SAR)", 0, 2, 1)
    D = st.sidebar.slider("D (SI)", 0, 1, 1)
    Q = st.sidebar.slider("Q (SMA)", 0, 2, 1)
    s = st.sidebar.selectbox("Frecuencia (s)", [6, 12], index=1)
    params = {'order': (p, d, q), 'seasonal_order': (P, D, Q, s)}

elif modelo_selected == "Prophet":
    st.sidebar.subheader("Parámetros Prophet")
    cps = st.sidebar.slider("Changepoint Prior Scale", 0.001, 0.5, 0.05, step=0.005)
    params = {'changepoint_prior_scale': cps}

elif modelo_selected == "XGBoost":
    st.sidebar.subheader("Parámetros XGBoost")
    n_est = st.sidebar.slider("N Estimators", 10, 500, 100)
    depth = st.sidebar.slider("Max Depth", 1, 10, 3)
    params = {'n_estimators': n_est, 'max_depth': depth}

st.sidebar.divider()
horizonte = st.sidebar.number_input("Meses a Predecir", 1, 36, 12)

# --- Main App ---
st.title("📊 Análisis y Pronóstico de Rentas Cedidas")
st.markdown(f"**Proyecto:** {PROYECTO_NOMBRE}")

@st.cache_data
def load_and_preprocess():
    df = utils.cargar_datos(DATA_FILE)
    df_m = utils.agregar_mensual(df, COL_FECHA, COL_VALOR)
    return df_m

try:
    df_mensual = load_and_preprocess()
    
    # Mostrar filtros temporales
    fecha_min, fecha_max = df_mensual.index.min(), df_mensual.index.max()
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Fecha Inicio Análisis", value=fecha_min, min_value=fecha_min, max_value=fecha_max)
    end_date = col2.date_input("Fecha Fin Análisis", value=fecha_max, min_value=fecha_min, max_value=fecha_max)
    
    # Filtrar datos
    df_filtered = df_mensual.loc[str(start_date):str(end_date)]
    
    # --- Ejecutar Modelos ---
    if modelo_selected != "Comparativo":
        st.subheader(f"Evolución y Pronóstico - {modelo_selected}")
        
        with st.spinner(f"Entrenando {modelo_selected}..."):
            if modelo_selected == "SARIMAX":
                y_hat, conf_int = model_helpers.entrenar_predict_sarima(df_filtered[COL_VALOR], params['order'], params['seasonal_order'], steps=horizonte)
            elif modelo_selected == "Prophet":
                y_hat, conf_int = model_helpers.entrenar_predict_prophet(df_filtered, COL_VALOR, params['changepoint_prior_scale'], steps=horizonte)
            elif modelo_selected == "XGBoost":
                y_hat, conf_int = model_helpers.entrenar_predict_xgboost(df_filtered, COL_VALOR, params['n_estimators'], params['max_depth'], steps=horizonte)
        
        # Gráfica Interactiva con Plotly
        fig = go.Figure()
        
        # Serie Real
        fig.add_trace(go.Scatter(
            x=df_filtered.index, y=df_filtered[COL_VALOR],
            mode='lines+markers', name='Histórico Real',
            line=dict(color=C_PRIMARY, width=2)
        ))
        
        # Predicción
        fig.add_trace(go.Scatter(
            x=y_hat.index, y=y_hat,
            mode='lines+markers', name=f'Pronóstico {modelo_selected}',
            line=dict(color=C_SECONDARY, width=3, dash='dash')
        ))
        
        # Intervalos de confianza (si existen)
        if conf_int is not None:
            # Manejar diferentes formatos de conf_int (SARIMAX vs Prophet)
            if isinstance(conf_int, pd.DataFrame):
                lower = conf_int.iloc[:, 0]
                upper = conf_int.iloc[:, 1]
            else: # SARIMAXX
                lower = conf_int.iloc[:, 0]
                upper = conf_int.iloc[:, 1]
                
            fig.add_trace(go.Scatter(
                x=y_hat.index.tolist() + y_hat.index.tolist()[::-1],
                y=upper.tolist() + lower.tolist()[::-1],
                fill='toself',
                fillcolor='rgba(192, 57, 43, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                name='Intervalo Confianza'
            ))
            
        fig.update_layout(
            title=f"Pronóstico de Rentas Cedidas: {modelo_selected}",
            xaxis_title="Fecha",
            yaxis_title="Recaudo (COP)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            height=600,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Métricas (si hay datos suficientes para validación, aquí simplificamos mostrando promedio futuro)
        m1, m2, m3 = st.columns(3)
        m1.metric("Promedio Histórico", f"${df_filtered[COL_VALOR].mean():,.0f}")
        m2.metric("Promedio Pronóstico", f"${y_hat.mean():,.0f}")
        crecimiento = (y_hat.mean() - df_filtered[COL_VALOR].tail(12).mean()) / df_filtered[COL_VALOR].tail(12).mean() if len(df_filtered) >= 12 else 0
        m3.metric("Crecimiento Proyectado vs AA", f"{crecimiento:.2%}")

    else:
        st.subheader("Comparativa de Modelos")
        st.info("Esta sección comparará los modelos con parámetros base por defecto para una vista rápida.")
        # Aquí se podría implementar una lógica para correr todos y comparar
        st.write("Cargando métricas de modelos pre-entrenados...")
        
        try:
            # Intentar cargar métricas existentes
            metrics_files = list(OUTPUTS_REPORTS.glob("*_metricas.csv"))
            if metrics_files:
                all_metrics = []
                for f in metrics_files:
                    m_df = pd.read_csv(f)
                    m_df['Modelo'] = f.name.replace('_metricas.csv', '').upper()
                    all_metrics.append(m_df)
                
                metrics_summary = pd.concat(all_metrics).set_index('Modelo')
                st.table(metrics_summary[['RMSE', 'MAE', 'MAPE', 'R2']])
                
                # Gráfica de Radar para métricas
                fig_radar = go.Figure()
                for modelo in metrics_summary.index:
                    # Normalizar para radar (inverso para que más afuera sea mejor)
                    # Nota: Simplificado para demo
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[1/metrics_summary.loc[modelo, 'MAPE'], metrics_summary.loc[modelo, 'R2']],
                        theta=['Eficiencia (1/MAPE)', 'Ajuste (R2)'],
                        fill='toself',
                        name=modelo
                    ))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
                # st.plotly_chart(fig_radar) # Reactivar cuando se normalicen métricas
            else:
                st.warning("No se encontraron archivos de métricas generados previamente.")
        except Exception as e:
            st.error(f"Error cargando comparativa: {e}")

except Exception as e:
    st.error(f"Error en la aplicación: {e}")
    st.exception(e)
