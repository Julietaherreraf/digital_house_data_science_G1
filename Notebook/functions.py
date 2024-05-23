## Funciones necesarias

def dropear_columnas(df_x, columnas):
    """
    Realiza modificaciones en un DataFrame dado, incluyendo eliminación de columnas específicas y
    eliminación de filas con valores NaN en 'fake_positive'.
    """
    df = df_x.copy()
    df = df.drop(columns=columnas, axis=1)
    return df


def fix_types_and_nan(df: pd.DataFrame, config: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Función para preprocesar un dataframe según una configuración proporcionada.
    Rellena los valores NA y convierte los tipos de datos. Genera un error si se
    encuentran valores NA en una columna donde 'na_fill_value' está configurado como
    'NOT_ALLOWED'.

    Nota: Devuelve un dataframe que SOLO contiene las columnas especificadas.

    El parámetro config debe ser un diccionario que mapea nombres de columnas a
    otro diccionario que especifica el tipo de dato para la columna ('dtype') y
    cómo manejar los valores NA ('na_fill_value'). Si 'na_fill_value' está
    configurado como 'NOT_ALLOWED', la función generará un ValueError si se
    encuentran valores NA en esa columna.

    Ejemplo:

    config = {
        'promotion_id': {'dtype': str, 'na_fill_value': 'NOT_ALLOWED'},
        'days_of_week': {'dtype': str, 'na_fill_value': 'NOT_ALLOWED'},
        'cap_amount': {'dtype': np.float32, 'na_fill_value': 0},
        'slug': {'dtype': str, 'na_fill_value': 'NOT_ALLOWED'},
        'promotion_banks': {'dtype': str, 'na_fill_value': 'NOT_ALLOWED'},
        'pct_promo_value': {'dtype': np.float32, 'na_fill_value': 0},
        'visibility_start_date': {'dtype': 'datetime64[ns]', 'na_fill_value': 'NOT_ALLOWED'},
        'visibility_stop_date': {'dtype': 'datetime64[ns]', 'na_fill_value': 'NOT_ALLOWED'}
    }

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame de entrada a ser procesado.

    config : dict
        Diccionario que especifica los tipos de datos y el manejo de valores NA para cada columna.

    Returns
    -------
    df : pandas.DataFrame
        Dateframe procesado
    """
    df_copy = df.copy()
    for col, conf in config.items():
        if col in df_copy.columns:
            if conf['na_fill_value'] == "NOT_ALLOWED":
                if df_copy[col].isnull().any():
                    raise ValueError(f"NA valores encontrados en columna {col} que no admite valores NA.")
                df_copy[col] = df_copy[col].astype(conf['dtype'])
            else:
                df_copy[col] = df_copy[col].fillna(conf['na_fill_value']).astype(conf['dtype'])
        else:
            raise KeyError(f"{col} no encontrado en dataframe.")

    return df_copy[list(config.keys())]


def plot_acf_pacf(df, title='FAS y FACP de la Serie Diferenciada'):
    # Set up the plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(title)

    # Plot ACF and PACF
    plot_acf(df, ax=axes[0], title='FAS')
    plot_pacf(df, ax=axes[1], title='FACP')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_fac(df, title='FAC de la Serie Diferenciada'):
    # Set up the plot
    fig, ax = plt.subplots(figsize=(15, 5))
    fig.suptitle(title)

    # Plot FAC
    fac_values = acovf(df, adjusted=True)
    sns.lineplot(x=range(len(fac_values)), y=fac_values, ax=ax)
    ax.set_title('FAC')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def stationarity_tests(df, title=''):
    print(f'Stationarity Tests for: {title}')

    # ADF Test
    print('ADF Test:')
    result_adf = adfuller(df, autolag='AIC')
    labels_adf = ['ADF Test Statistic', 'p-value', '# Lags Used', '# Observations']
    out_adf = pd.Series(result_adf[0:4], index=labels_adf)
    for key, val in result_adf[4].items():
        out_adf[f'Critical Value ({key})'] = val
    print(out_adf.to_string())
    print('\n')

    # KPSS Test
    print('KPSS Test:')
    result_kpss = kpss(df, regression='c', nlags='auto')
    labels_kpss = ['KPSS Test Statistic', 'p-value', '# Lags Used']
    out_kpss = pd.Series(result_kpss[0:3], index=labels_kpss)
    for key, val in result_kpss[3].items():
        out_kpss[f'Critical Value ({key})'] = val
    print(out_kpss.to_string())
    print('\n')


def test_stationarity(timeseries):
    print('Stationarity Tests:')

    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    # Graficar:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    timeseries = timeseries.iloc[:,0].values
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


def tes_optimizer(train, test, abg, trend_mode='add', seasonal_mode='add', seasonal_period=12, step=5):
    """
    This function optimizes hyperparameters for the TES model.
    """
    best_alpha, best_beta, best_gamma, best_mae = None, None, None, float("inf")
    for comb in abg:
        tes_model = ExponentialSmoothing(train, trend=trend_mode, seasonal=seasonal_mode, seasonal_periods=seasonal_period).\
            fit(smoothing_level=comb[0], smoothing_trend=comb[1], smoothing_seasonal=comb[2])
        y_pred = tes_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)
        if mae < best_mae:
            best_alpha, best_beta, best_gamma, best_mae = comb[0], comb[1], comb[2], mae
        print([round(comb[0], 2), round(comb[1], 2), round(comb[2], 2), round(mae, 2)])
    print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_gamma:", round(best_gamma, 2),
          "best_mae:", round(best_mae, 4))
    return best_alpha, best_beta, best_gamma, best_mae


def date_filter(df, column, inf, sup):
    return df[ (df[column].dt.year >= inf) & (df[column].dt.year <= sup) ]


def busqueda_sarimax(ts, valores_p, valores_d, valores_q, valores_P, valores_D, valores_Q, valores_s, exog=None):
    """
    Realiza una búsqueda en cuadrícula para encontrar los mejores parámetros del modelo SARIMAX.

    Parámetros:
    ts (pd.Series): Datos de la serie temporal.
    valores_p (list): Parte AR(p) del modelo.
    valores_d (list): Parte I(d) del modelo.
    valores_q (list): Parte MA(q) del modelo.
    valores_P (list): Parte AR(P) estacional del modelo.
    valores_D (list): Parte I(D) estacional del modelo.
    valores_Q (list): Parte MA(Q) estacional del modelo.
    valores_s (list): Períodos estacionales.
    exog (pd.DataFrame, opcional): Variables exógenas.

    Retorna:
    dict: Mejores parámetros y el AIC correspondiente.
    """
    warnings.filterwarnings("ignore")
    mejor_aic = float("inf")
    mejores_params = None

    pdq = list(itertools.product(valores_p, valores_d, valores_q))
    seasonal_pdq = list(itertools.product(valores_P, valores_D, valores_Q, valores_s))

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                model = SARIMAX(ts, order=param, seasonal_order=param_seasonal, exog=exog)
                results = model.fit(disp=False)
                if results.aic < mejor_aic:
                    mejor_aic = results.aic
                    mejores_params = (param, param_seasonal)
                print(f'SARIMAX{param}x{param_seasonal} - AIC:{results.aic}')
            except Exception as e:
                continue

    return {"params": mejores_params, "aic": mejor_aic}