import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# Importar configuraciones locales
import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import cargar_datos, agregar_mensual, calcular_metricas

def entrenar_predict_sarima(y, order, seasonal_order, steps=12):
    """Ajusta y predice con SARIMA."""
    model = SARIMAX(y, order=order, seasonal_order=seasonal_order, 
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)
    forecast = results.get_forecast(steps=steps)
    return forecast.predicted_mean, forecast.conf_int()

def entrenar_predict_prophet(df_mensual, col_recaudo, changepoint_prior_scale=0.05, steps=12):
    """Ajusta y predice con Prophet."""
    df_p = df_mensual[[col_recaudo]].reset_index()
    df_p.columns = ['ds', 'y']
    
    m = Prophet(changepoint_prior_scale=changepoint_prior_scale, yearly_seasonality=True)
    m.fit(df_p)
    
    future = m.make_future_dataframe(periods=steps, freq='MS')
    forecast = m.predict(future)
    
    return forecast.tail(steps).set_index('ds')['yhat'], forecast.tail(steps).set_index('ds')[['yhat_lower', 'yhat_upper']]

def entrenar_predict_xgboost(df_mensual, col_recaudo, n_estimators=100, max_depth=3, steps=12):
    """Ajusta y predice con XGBoost (recursivo simple)."""
    # Feature engineering básico para el dashboard
    df = df_mensual[[col_recaudo]].copy()
    df['Mes'] = df.index.month
    df['Anio'] = df.index.year
    for i in range(1, 4):
        df[f'Lag_{i}'] = df[col_recaudo].shift(i)
    
    df = df.dropna()
    X = df.drop(columns=[col_recaudo])
    y = df[col_recaudo]
    
    model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X, y)
    
    # Predicción recursiva
    last_val = df.tail(1)
    predictions = []
    current_date = df.index.max()
    
    # Simplificación para el dashboard: solo componentes de calendario para evitar recursividad compleja
    future_index = pd.date_range(start=current_date + pd.DateOffset(months=1), periods=steps, freq='MS')
    X_future = pd.DataFrame(index=future_index)
    X_future['Mes'] = X_future.index.month
    X_future['Anio'] = X_future.index.year
    # Lags promedio para no complicar el loop
    for i in range(1, 4):
        X_future[f'Lag_{i}'] = y.mean()
        
    preds = model.predict(X_future)
    return pd.Series(preds, index=future_index), None
