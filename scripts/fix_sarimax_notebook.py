import nbformat as nbf
from pathlib import Path

notebook_path = Path(r'c:\Users\efren\Music\ESTRUCTURA DATOS RENTAS\notebooks\05_SARIMAX.ipynb')

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        # 1. Corregir preparación de datos
        if 'pd.read_csv(DATA_PROCESSED' in cell.source:
            cell.source = (
                "# 1. Reconstrucción robusta de la serie mensual (asegurando 2021-2025)\n"
                "df = cargar_datos(DATA_FILE)\n"
                "df_mensual = agregar_mensual(df)\n"
                "\n"
                "# Mapear variables macro usando el diccionario de config\n"
                "df_mensual['Año'] = df_mensual.index.year\n"
                "for var in ['IPC', 'Salario_Minimo', 'UPC']:\n"
                "    df_mensual[var] = df_mensual['Año'].map(lambda y: MACRO_DATA.get(y, {}).get(var, np.nan))\n"
                "\n"
                "# Rellenar 2021 si hay NaNs\n"
                "df_mensual.fillna(method='bfill', inplace=True)\n"
                "df_mensual.dropna(inplace=True)\n"
                "\n"
                "# Variables exógenas\n"
                "exog_cols = ['IPC', 'Salario_Minimo', 'UPC']\n"
                "\n"
                "# Split Train/Test\n"
                "train = df_mensual.loc[:TRAIN_END]\n"
                "test = df_mensual.loc[TEST_START:]\n"
                "\n"
                "y_train = train['Recaudo_Neto']\n"
                "y_test = test['Recaudo_Neto']\n"
                "X_train = train[exog_cols]\n"
                "X_test = test[exog_cols]\n"
                "\n"
                "print(f'📊 Train: {len(train)} meses | Test: {len(test)} meses')\n"
                "print(f'📊 Rango: {train.index.min().date()} a {train.index.max().date()}')\n"
                "print(f'📊 Variables exógenas: {exog_cols}')"
            )
        
        # 2. Corregir Ajuste del Modelo
        elif 'order = (1, 1, 1)' in cell.source and 'seasonal_order = (1, 1, 1,' in cell.source:
            cell.source = (
                "# Simplificamos el modelo para series cortas (evita ValueError en diagnósticos)\n"
                "# Usamos el orden exitoso del SARIMA puro: (1,0,0)x(0,1,0,12) + Variables Exógenas\n"
                "order = (1, 0, 0)\n"
                "seasonal_order = (0, 1, 0, ESTACIONALIDAD)\n"
                "\n"
                "modelo_sarimax = SARIMAX(\n"
                "    y_train,\n",
                "    exog=X_train,\n",
                "    order=order,\n",
                "    seasonal_order=seasonal_order,\n",
                "    enforce_stationarity=False,\n",
                "    enforce_invertibility=False\n",
                ")\n"
                "resultado_sarimax = modelo_sarimax.fit(disp=False, maxiter=500)\n"
                "\n"
                "print(resultado_sarimax.summary())\n"
                "print(f'\\n📊 AIC: {resultado_sarimax.aic:.2f} | BIC: {resultado_sarimax.bic:.2f}')"
            )

        # 3. Corregir Diagnóstico
        elif 'plot_diagnostics' in cell.source:
            cell.source = (
                "try:\n"
                "    # Reducimos lags a 5 para mayor compatibilidad con series cortas\n"
                "    fig = resultado_sarimax.plot_diagnostics(figsize=(14, 10), lags=5)\n"
                "    fig.suptitle('Diagnóstico de Residuos — SARIMAX', fontweight='bold', fontsize=14, y=1.01)\n"
                "    plt.tight_layout()\n"
                "    fig.savefig(OUTPUTS_FIGURES / '05_sarimax_diagnostico.png', dpi=150, bbox_inches='tight')\n"
                "    plt.show()\n"
                "except Exception as e:\n"
                "    print(f'⚠️ No se pudo generar el plot_diagnostics automático: {e}')\n"
                "\n"
                "lb_test = acorr_ljungbox(resultado_sarimax.resid, lags=[min(12, len(resultado_sarimax.resid)-1)], return_df=True)\n"
                "print('\\n📊 Test Ljung-Box:')\n"
                "print(lb_test)"
            )

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"✅ Cuaderno '{notebook_path.name}' corregido exitosamente.")
