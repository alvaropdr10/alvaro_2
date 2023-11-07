#!/usr/bin/env python
# coding: utf-8

# ## Sección 1: Introducción
# Introducción:
# 
# En este trabajo, vamos a hablar sobre Stack Overflow, esa web donde la gente va a buscar respuestas cuando se atasca con el código. Imagina que en un archivo llamado StackOverflow.csv hay datos sobre cuántas preguntas nuevas se hacen cada mes sobre diferentes lenguajes de programación.
# 
# Lo que nos interesa es ver cómo ha ido cambiando el número de preguntas sobre MATLAB a lo largo del tiempo en Stack Overflow. Queremos predecir cómo serán esas preguntas en el futuro, como adivinar un poco basándonos en lo que ha pasado antes.
# 
# Objetivo:
# 
# Queremos ver cómo han ido creciendo o bajando esas preguntas sobre MATLAB. Para hacerlo, vamos a mirar esos datos y a tratar de entender cómo cambian con el tiempo. Queremos usar lo que hemos aprendido para intentar adivinar cómo serán las preguntas de MATLAB más adelante.
# 
# Vamos a usar unos truquitos matemáticos para intentar predecir esas preguntas. Así, podemos entender mejor si hay patrones o tendencias que nos ayuden a ver cómo será el futuro.
# 
# La idea es contar cómo ha sido todo esto de las preguntas sobre MATLAB en Stack Overflow y tratar de predecir un poco cómo será en el futuro. Queremos explicar paso a paso todo lo que hacemos para entender mejor este rollo de las predicciones en la programación.
# 
# + AÑADIR APUNTES QUE TENGO Y COSAS DICHAS EN CLASE!!! 
# 
# 

# Hacer un Dataframe con matlab y con date y pasar a date time MONTH

# # Sección 2: Preparación y Análisis Inicial de Datos -- Jueves TBC Retocar
# 
# ## Análisis Inicial de Datos: 
# Vamos a abrir el archivo llamado "StackOverflow.csv" que tiene un montón de datos sobre preguntas de programación. Pero nosotros nos interesamos solo en las preguntas sobre MATLAB, así que vamos a buscar esas y dejar "al margen" las demás.
# 
# ## Visualización y Análisis Exploratorio: 
# Después de seleccionar las preguntas de Matlab, lo siguiente será respresentar. Haciendo gráficos que muestren cómo cambian las preguntas mes a mes a lo largo del tiempo. Así, vemos si hay algún patrón/tendencia o si suben y bajan mucho. 
# 
# ## Preparación de Datos: 
# Antes de hacer nuestros cálculos y predicciones, es necesario organizar bien los datos. Es como si cocinaramos, primero se junta los ingredientes y se mezclan antes de empezar a ejecutar. Con los datos, los limpiamos y separamos en 3 grupos: uno para aprender cómo predecir, otro para probar si lo que hemos aprendido funciona y el último para ver cómo de bien hemos 'adivinado el futuro'.

# In[1]:


import warnings
warnings.filterwarnings('ignore')
# version 2.03


# In[2]:


# Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


# Leo los datos y separo en columnas
stackoverf_df = pd.read_csv('/Users/paoladiezmartinez/Downloads/StackOverflow (1).csv', sep=';', header=None)
stackoverf_df = stackoverf_df[0].str.split(',', expand=True)

# Establezco la primera fila como nombres de columna
stackoverf_df.columns = stackoverf_df.iloc[0]

# Selecciono únicamente la columna 'matlab' y las fechas que se llaman 'month'
matlab_dates_df = stackoverf_df[['month', 'matlab']]

# Elimino la primera fila ya que se ha utilizado como nombres de columna
matlab_dates_df = matlab_dates_df.iloc[1:]

# El DataFrame resultante
print(matlab_dates_df)


# ## Graficar Serie Temporal

# In[4]:


# Convertimos las fechas al formato adecuado (asumiendo '09-Jan', '09-Feb', ...)
matlab_dates_df['month'] = pd.to_datetime(matlab_dates_df['month'], format='%y-%b')
# Establecemos 'month' como índice para facilitar el análisis de series temporales
matlab_dates_df.set_index('month', inplace=True)

# Mostramos los primeros registros para verificar la conversión
print(matlab_dates_df.head())


# In[5]:


# Eliminar la segunda columna por posición
matlab_dates_df = matlab_dates_df.iloc[:, [0]]  # Mantenemos solo la primera columna


# In[6]:


matlab_dates_df


# In[7]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.plot(matlab_dates_df.index, matlab_dates_df['matlab'], marker='o')

plt.title('Serie Temporal de Preguntas Mensuales de MATLAB')

plt.xlabel('Fecha')
plt.ylabel('Número de Preguntas')

plt.grid(True)

plt.show()


# ## Tendencia, estacionalidad y estacionariedad con descomposición estacional:

# In[11]:


get_ipython().system('pip install statsmodels')


# In[8]:


import statsmodels.api as sm

# Realiza la descomposición estacional
decomposition = sm.tsa.seasonal_decompose(matlab_dates_df['matlab'], model='additive')

# Grafica los componentes: tendencia, estacionalidad y residuos
fig, axes = plt.subplots(4, 1, figsize=(10, 8))

axes[0].plot(matlab_dates_df.index, decomposition.observed, label='Original')
axes[0].legend(loc='upper left')

axes[1].plot(matlab_dates_df.index, decomposition.trend, label='Tendencia')
axes[1].legend(loc='upper left')

axes[2].plot(matlab_dates_df.index, decomposition.seasonal, label='Estacionalidad')
axes[2].legend(loc='upper left')

axes[3].plot(matlab_dates_df.index, decomposition.resid, label='Residuos')
axes[3].legend(loc='upper left')

plt.tight_layout()
plt.show()


# In[9]:


import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Realizar la descomposición estacional
result = seasonal_decompose(matlab_dates_df['matlab'], model='additive')

# Graficar la descomposición
plt.figure(figsize=(10, 8))

plt.subplot(411)
plt.plot(result.observed, label='Observado')
plt.legend()

plt.subplot(412)
plt.plot(result.trend, label='Tendencia')
plt.legend()

plt.subplot(413)
plt.plot(result.seasonal, label='Estacionalidad')
plt.legend()

plt.subplot(414)
plt.plot(result.resid, label='Residuo')
plt.legend()

plt.tight_layout()
plt.show()


# ## Preprocesamiento de datos: División Train y Test

# 1. División en conjuntos de entrenamiento y prueba

# In[10]:


from sklearn.model_selection import train_test_split

# Suponiendo que tienes tu serie temporal 'matlab_dates_df' preparada y lista para el modelado

# Dividir los datos en entrenamiento y prueba (por ejemplo, 70% para entrenamiento y 30% para prueba)
train_size = int(len(matlab_dates_df) * 0.7)
train, test = matlab_dates_df.iloc[:train_size], matlab_dates_df.iloc[train_size:]

print(f"Tamaño del conjunto de entrenamiento: {len(train)}")
print(f"Tamaño del conjunto de prueba: {len(test)}")


# 2. División en conjuntos de entrenamiento, validación y prueba

# In[11]:


# Suponiendo que deseas tener un conjunto de validación además del conjunto de entrenamiento y prueba

# Dividir los datos en entrenamiento, validación y prueba
train_size = int(len(matlab_dates_df) * 0.7)
val_size = int(len(matlab_dates_df) * 0.15)  # 15% para validación
train, val, test = (
    matlab_dates_df.iloc[:train_size],
    matlab_dates_df.iloc[train_size:train_size + val_size],
    matlab_dates_df.iloc[train_size + val_size:]
)

print(f"Tamaño del conjunto de entrenamiento: {len(train)}")
print(f"Tamaño del conjunto de validación: {len(val)}")
print(f"Tamaño del conjunto de prueba: {len(test)}")


# # Sección 3: Modelado - ETS & ARIMA -- Viernes TBC (Retocar)
# 
# ## Modelo ETS (Error, Trend, Seasonality):
# El modelo ETS es una forma sencilla y poderosa de predecir valores en series temporales. Se enfoca en capturar el Error, la Tendencia y la Estacionalidad. El error se refiere a las diferencias entre los valores reales y los valores predichos. La tendencia describe la dirección general de los datos a lo largo del tiempo (por ejemplo, si la serie temporal muestra un aumento o disminución constante). La estacionalidad refleja patrones repetitivos que ocurren a intervalos regulares.
# 
# ## Modelo ARIMA (AutoRegressive Integrated Moving Average):
# El modelo ARIMA es otra técnica útil para predecir series temporales. Este modelo se enfoca en la autocorrelación (las relaciones entre observaciones en diferentes momentos) y se compone de tres partes: Auto Regresión (AR), Integración (I) y Media Móvil (MA). El componente AR considera la relación entre una observación actual y las anteriores, la Integración es la diferenciación de la serie para hacerla estacionaria, y la Media Móvil modela el error de predicción a partir de un promedio móvil de observaciones anteriores.
# 
# ## Validación y Ajuste de Parámetros:
# La validación es un proceso crucial para verificar qué tan bien los modelos se desempeñan con datos nuevos. En este caso, se usará un conjunto de datos de validación para probar la capacidad predictiva de los modelos ETS y ARIMA. Ajustar los parámetros significa encontrar la mejor configuración para los modelos, como la longitud de la ventana temporal, la inclusión o exclusión de la estacionalidad, la regularización y otros hiperparámetros. Posteriormente, se comparará el rendimiento de ambos modelos para evaluar su precisión en la predicción de la serie temporal de MATLAB.

# ## ETS GUIA DEL PROCESO (QUITARLO)
# 
# 1. Preprocesamiento de Datos (ETL):
#    - Asegurarse de que tus fechas estén en el formato adecuado.
#    - Revisar y limpiar los datos, si es necesario.
#    - Asegurarse de que tus datos estén en el formato de serie temporal (fechas y valores).
# 
# 2. División en Entrenamiento y Prueba:
#    - Separar los datos en un conjunto de entrenamiento y uno de prueba.
# 
# 3. Implementación del Modelo ETS:
#    - Utilizar la biblioteca `sktime`.
#    - Crear un modelo ETS.
#    - Ajustar el modelo ETS con los datos de entrenamiento.
# 
# 4. Predicción con el Modelo ETS:
#    - Hacer predicciones para el conjunto de prueba.
#    - Visualizar los resultados y comparar las predicciones con los valores reales.
# 
# 5. Evaluación del Modelo:
#    - Calcular métricas como el Error Cuadrático Medio (MSE) y el Error Porcentual Absoluto Medio (MAPE).
#    - Analizar y entender qué tan bien se ajusta el modelo a tus datos de prueba.

# ## desde la práctica 3 modelo exponencial??

# Graficar los Ingresos
# - Tendencia - Componente Estacional - Varianza no constante

# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt

# Ajustamos el tamaño predeterminado de la figura usando Seaborn
sns.set(rc={'figure.figsize':(11, 4)})

ax = sns.lineplot(data=matlab_dates_df, x='month', y='matlab', marker='o', linestyle='-')
ax.set_ylabel('Preguntas Matlab')

plt.show()  # Muestra la visualización



# In[14]:


import statsmodels.api as sm


# In[28]:


import seaborn as sns
import matplotlib.pyplot as plt

# Ajustamos el tamaño predeterminado de la figura usando Seaborn
sns.set(rc={'figure.figsize': (11, 4)})

# Creamos el gráfico de líneas para los datos trimestrales sin variables de agrupación
ax = sns.lineplot(data=matlab_quarterly['matlab'], marker='o', linestyle='-')
ax.set_ylabel('Preguntas Matlab por Trimestre')

plt.show()  



# ## Modelos de Suavizado Exponencial
# 
# Vamos a separar la muestra en la parte de estimación (Training) y la parte de predicción/Verificación (Testing). Quitamos 4 trimestres.
# 
# Vamos a predecir 4 periodos (un año) (h=4)
# 

# In[25]:


from sktime.forecasting.base import ForecastingHorizon
from sktime.utils.plotting import plot_series
from sktime.forecasting.model_selection import temporal_train_test_split


# In[33]:


# Convertir el índice 'month' a tipo timestamp
matlab_dates_df.index = matlab_dates_df.index.to_timestamp()

# Convertir a trimestres
matlab_dates_df.index = matlab_dates_df.index.to_period('Q')
print(matlab_dates_df)


# In[49]:


from sktime.forecasting.model_selection import temporal_train_test_split

# Asumiendo que 'matlab_dates_df' tiene un índice ajustado a trimestres
y = matlab_dates_df.index  # Utilizar el índice trimestral como la variable objetivo

# Dividir los datos en conjuntos de entrenamiento y prueba
y_train, y_test = temporal_train_test_split(y=y, test_size=4)

# Mostrar la forma de los conjuntos de entrenamiento y prueba
print(y_train.shape[0], y_test.shape[0])


# In[50]:


print(y_train)


# In[51]:


from sktime.forecasting.ets import AutoETS


# In[52]:


# step 2: specifying forecasting horizon
fh = np.arange(1, 17)

# step 3: specifying the forecasting algorithm
matlab_auto_model = AutoETS(auto=True, sp=4, n_jobs=-1)


# In[54]:


# Convertimo el índice a un DataFrame
y_train_df = pd.DataFrame(index=y_train)
y_train_df['time'] = y_train_df.index


# In[58]:


import pandas as pd

# Convertir y_train a un DataFrame de Pandas con un índice de tipo periodo
y_train_df = pd.DataFrame(y_train, index=y_train.index)


# In[59]:


print(y_train_df.index)


# In[75]:


matlab_auto_model.fit(y_train_df.index)


# In[57]:


print(matlab_auto_model.summary())


# In[ ]:


# #predicciones
y_pred = matlab_auto_model.predict(fh)
print(y_pred)


# In[ ]:


y_pred_ints = matlab_auto_model.predict_interval(fh, coverage=0.9)
y_pred_ints


# In[ ]:


# optional: plotting predictions and past data
plot_series(y_train, y_pred,y_test, labels=["Matlab", "Matlab pred", "Matlab REAL"])


# In[ ]:


fig, ax = plot_series(y_train, y_pred, y_test, labels=["Matlab", "Matlab pred", "Matlab REAL"])
ax.fill_between(
    ax.get_lines()[-2].get_xdata(),
    ko_pred_ints[('Coverage', 0.9, 'lower')],
    ko_pred_ints[('Coverage', 0.9, 'upper')],
    alpha=0.2,
    color=ax.get_lines()[-2].get_c(),
    label=f"90% prediction intervals",
)
ax.legend(loc='upper left')


# In[ ]:


plot_series(y_train["2013":], ko_pred,y_test, labels=["Matlab", "Matlab pred", "Matlab REAL"])


# In[ ]:


fig, ax = plot_series(y_train["2013":], ko_pred, y_test, labels=["Matlab", "Matlab pred", "Matlab REAL"])
ax.fill_between(
    ax.get_lines()[-2].get_xdata(),
    ko_pred_ints[('Coverage', 0.9, 'lower')],
    ko_pred_ints[('Coverage', 0.9, 'upper')],
    alpha=0.2,
    color=ax.get_lines()[-2].get_c(),
    label=f"90% prediction intervals",
)
ax.legend(loc='upper left');


# In[ ]:


from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
# option 1: using the lean function interface
mean_absolute_percentage_error(y_test, y_pred[0:4])


# In[ ]:


from sktime.performance_metrics.forecasting import MeanSquaredError
mse = MeanSquaredError()
mse(y_test, y_pred[0:4])


# In[ ]:


rmse = MeanSquaredError(square_root=True)
rmse(y_test, y_pred[0:4])


# In[ ]:


# step 2: specifying forecasting horizon = SE REPITE
fh = np.arange(1, 7)

# step 3: specifying the forecasting algorithm
matlab_auto_model = AutoETS(auto=True, sp=4, n_jobs=-1)

y = matlab_ts['matlab'].astype('float64').to_period('Q')

matlab_auto_model.fit(y)

print(matlab_auto_model.summary())


# In[76]:


from sktime.forecasting.ets import AutoETS

# Step 2: Specify the forecasting algorithm
matlab_auto_model = AutoETS(auto=True, sp=4, n_jobs=-1)

# Ensure that 'matlab_dates_df' has an index adjusted to quarters
# ...

# Convert the 'matlab_dates_df' index to a pandas PeriodIndex if it's not already in that format
# You might need to adjust this conversion based on your current DataFrame structure
matlab_dates_df.index = pd.PeriodIndex(matlab_dates_df.index, freq='Q')

# Train the model
matlab_auto_model.fit(matlab_dates_df)

# Show a summary of the model
print(matlab_auto_model.summary())



# In[ ]:


# step 5: querying predictions
matlab_pred = matlab_auto_model.predict(fh)
print(matlab_pred)


# In[ ]:


plot_series(y, matlab_pred, labels=["Matlab", "Matlab pred"])


# In[77]:


from sktime.forecasting.exp_smoothing import ExponentialSmoothing
forecaster = ExponentialSmoothing(trend='additive', seasonal='multiplicative', sp=4)
forecaster.fit(y)


# In[ ]:


y_pred = forecaster.predict(fh)
y_pred


# In[ ]:


print(forecaster._fitted_forecaster.summary())


# In[ ]:


forecaster.get_fitted_params()


# In[ ]:


plot_series(y, y_pred, labels=["Matlab", "Matlab pred"])


# In[ ]:


forecaster = ExponentialSmoothing(trend='additive',seasonal=None, sp=4)
forecaster.fit(y)


# In[ ]:


y_pred = forecaster.predict(fh)


# In[ ]:


plot_series(y, y_pred, labels=["Matlab", "Matlab pred"])


# In[ ]:


forecaster = ExponentialSmoothing(trend='mul',seasonal=None, sp=4)
forecaster.fit(y)
y_pred = forecaster.predict(fh)
plot_series(y, y_pred, labels=["Matlab", "Matlab pred"])


# In[ ]:


forecaster = ExponentialSmoothing(trend=None,seasonal=None, sp=4)
forecaster.fit(y)
y_pred = forecaster.predict(fh)
plot_series(y, y_pred, labels=["Matlab", "Matlab pred"])


# In[ ]:


forecaster = ExponentialSmoothing(trend=None,seasonal="mul", sp=4)
forecaster.fit(y)
y_pred = forecaster.predict(fh)
plot_series(y, y_pred, labels=["Matlab", "Matlab pred"])


# In[ ]:


forecaster = ExponentialSmoothing(trend=None,seasonal="add", sp=4)
forecaster.fit(y)
y_pred = forecaster.predict(fh)
plot_series(y, y_pred, labels=["Matlab", "Matlab pred"])


# In[ ]:


forecaster = ExponentialSmoothing(trend="add",seasonal="add",damped_trend=False, sp=4)
forecaster.fit(y)
y_pred = forecaster.predict(fh)

forecaster = ExponentialSmoothing(trend="add",seasonal="add",damped_trend=True, sp=4)
forecaster.fit(y)
y_pred_dump = forecaster.predict(fh)
plot_series(y["2009":], y_pred, y_pred_dump,labels=["Matlab", "Matlab pred","Matlab Pred Dumpeded"])


# hasta aqui el codigo de la sesion 3!! Usarlo
# y adapatarlo

# In[16]:


get_ipython().system('pip install sktime')


# In[78]:


from sklearn.model_selection import train_test_split

y = matlab_dates_df['matlab']  # Suponiendo que 'matlab' es tu columna de datos

# Divide los datos en conjuntos de entrenamiento y prueba
y_train, y_test = train_test_split(y, test_size=12, shuffle=False)  # Asegúrate de deshabilitar el mezclado para mantener el orden temporal


# In[79]:


# Suponiendo que 'matlab' es tu columna de datos
y = matlab_dates_df['matlab']

# Transformar a un formato aceptado por sktime (por ejemplo, un DataFrame de Pandas con un índice de tiempo)
y.index = pd.date_range(start='2009-01-01', periods=len(y), freq='MS')


# In[80]:


import numpy as np
import pandas as pd

# Suponiendo que 'y_test' y 'y_pred' son arrays o series de NumPy
y_test = np.array([2, 4, 6, 8, 10])  # Valores reales
y_pred = np.array([3, 3, 7, 8, 12])  # Predicciones del modelo

def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

smape_value = smape(y_test, y_pred)
print(f"SMAPE: {smape_value}")


# In[21]:


from sktime.utils.data_processing import from_nested_to_2d_array

y = matlab_dates_df['matlab']

y_nested = y.to_frame()

y_2d = from_nested_to_2d_array(y_nested)


# In[22]:


#incluirrr
from sktime.utils.data_processing import from_2d_array_to_nested
y = matlab_dates_df['matlab']
y_train, y_test = temporal_train_test_split(from_series_to_2d_array(y), test_size=12)


# In[81]:


import pandas as pd
from sktime.forecasting.ets import AutoETS
from sktime.utils.plotting import plot_series
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error


# Separación en entrenamiento y prueba
y = matlab_dates_df['matlab']  # Asumiendo 'matlab' como la columna de datos
y_train, y_test = temporal_train_test_split(y, test_size=12)  # Separando los últimos 12 meses para prueba


# In[82]:


# Creación y ajuste del modelo ETS
fh = list(range(1, len(y_test) + 1))
ets_model = AutoETS(auto=True, sp=12)  # Se utiliza la estacionalidad del año
ets_model.fit(y_train)

# Realizando predicciones
y_pred = ets_model.predict(fh)

# Visualizando resultados
plot_series(y_train, y_test, y_pred, labels=["Train", "Test", "ETS Predictions"])

# Calculando métricas de evaluación
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"MAPE (Error Porcentual Absoluto Medio): {mape:.2f}%")


# # Modelo ARIMA - Procedimiento TBC (quitarlo)
# 
# # 1. Exploración de datos:
# 
# ## Análisis de la serie temporal: 
# - Observa la distribución de los datos a lo largo del tiempo. Verifica si hay tendencias, patrones estacionales o ciclos.
# ## Identificación de estacionalidad: 
# - Busca patrones repetitivos que podrían indicar estacionalidad en la serie.
# ## Verificación de la estacionariedad: 
# - Comprueba si la media y la varianza de los datos son constantes a lo largo del tiempo. 
# - Si no lo son, considera transformaciones como diferenciación para hacer la serie temporal estacionaria.
# ## Preprocesamiento:
# - Tratamiento de datos faltantes: Asegúrate de que no haya datos faltantes en tu serie temporal.
# - Conversión de fechas: Verifica si las fechas están en un formato temporal adecuado para su procesamiento.
# 
# # 2. Modelado ARIMA:
# 
# ## a. División en conjuntos de entrenamiento y prueba: 
# Separa los datos en dos partes: una para entrenar el modelo y otra para evaluar su desempeño.
# 
# ## b. Transformación de la serie temporal: 
# Considera aplicar una transformación logarítmica a tus datos si la serie muestra una tendencia clara o una varianza no constante.
# 
# ## c. Ajuste del modelo ARIMA: 
# Utiliza la librería SKtime para ajustar un modelo ARIMA a tus datos. Experimenta con distintos valores para los parámetros p, d, y q del ARIMA.
# 
# ## d. Predicciones y evaluación: 
# Realiza predicciones utilizando tu modelo ARIMA ajustado y evalúa su rendimiento utilizando métricas de error como el MAE, el MSE y el MAPE.
# 
# ## e. Ajuste y validación: 
# Realiza ajustes en los parámetros y en la estructura del modelo para mejorar su desempeño, y verifica su validez utilizando el conjunto de prueba.
# 
# ## f. Visualización y Análisis:
# Graficación de predicciones: Visualiza las predicciones realizadas por el modelo ARIMA en comparación con los valores reales de la serie temporal.
# 
# ## g. Análisis de precisión: 
# Analiza y compara las predicciones con los valores reales para determinar la precisión y eficacia del modelo.

# ## COPIA practica 4 modelos arima

# In[83]:


import warnings
warnings.filterwarnings('ignore')


# In[84]:


# Styling notebook
from IPython.core.display import HTML
def css_styling():
    styles = open("style.css", "r",encoding="utf-8").read()
    return HTML(styles)
css_styling()


# In[90]:


# Resample to Quarterly I
matlab_ts=matlab_dates_df.resample("q").last()
matlab_ts.tail()


# In[88]:


# Resample to Quarterly II
# SKtime format
matlab_ts_q=matlab_dates_df['matlab'].astype('float64').to_period('Q').sort_index()
matlab_ts_q.tail()


# In[92]:


# GRAFICAR 
import seaborn as sns
sns.set(rc={'figure.figsize':(11, 4)})
ax = matlab_ts_q.plot(marker='o', linestyle='-')
ax.set_ylabel('Preguntas Matlab');


# In[93]:


# Plot Data
# Use Sktime style 
from sktime.utils.plotting import plot_series
plot_series(matlab_ts_q, labels=["Ventas"])


# No estacionariedad en Varianza
# Calculamos la transformacion logarítmica de la Serie Original¶
# 

# In[94]:


# Log Transformer Function
from sktime.transformations.series.boxcox import LogTransformer


# In[95]:


# Apply Log Transformer

transformer = LogTransformer()
log_matlab_ts_q= transformer.fit_transform(matlab_ts_q)
log_matlab_ts_q.tail()


# ## Lo he dividio en Q, pero es mejor por Month "M"

# In[96]:


# Plot Log Data
ax = matlab_ts_q.plot(marker='o', linestyle='-')
ax.set_ylabel('Preguntas Matlab')
ax.set_title('Preguntas Matlab: Transformación LOG')


# In[97]:


# Plot Log Data
fig, ax =plot_series(log_matlab_ts_q, labels=["Preguntas"])
ax.set_title('Preguntas Matlab: Transformación LOG')


# todo lo he ido haciendo con matlab_ts_q pero en realidad era con matlab_ts --> corregirloo!!

# No estacionariedad en Varianza
# ## Comparamos la transformacion logarítmica de la **Serie Original** y la **Serie en Logs**

# In[ ]:


# Plot Log Data & Original Data
fig, ax =plot_series(log_matlab_ts_q, labels=["Preguntas"])
ax.set_title('Matlab Preguntas: Serie Original')
fig, ax =plot_series(log_matlab_ts, labels=["Preguntas"])
ax.set_title('Matlab Preguntas: Transformación LOG')


# Autocorrelación
# Calculamos la Autocorrelación de la Serie en Logs

# In[98]:


# Autocorrelation Fuction Package
from sktime.utils.plotting import plot_correlations


# In[ ]:


# Autocorrelation Fuction Original Time Series = ESTE HAY QUE USAR
plot_correlations(log_matlab_ts)


# In[100]:


# Autocorrelation Fuction Original Time Series
plot_correlations(log_matlab_ts_q)


# In[101]:


# Difference Fuction Package
from sktime.transformations.series.difference import Differencer


# In[102]:


# Autocorrelation Fuction
# d=1; D=0; S=4
# 
transf_diff=Differencer(lags=[1])
plot_correlations(transf_diff.fit_transform(log_matlab_ts_q))#usar log_matlab_ts


# In[103]:


# Autocorrelation Fuction
# d=0; D=1; S=4
# 
transf_diff=Differencer(lags=[4])
plot_correlations(transf_diff.fit_transform(log_matlab_ts_q))#usar log_matlab_ts


# In[104]:


# Autocorrelation Fuction
# d=1; D=1; S=4
# 
transf_diff=Differencer(lags=[1,4])
plot_correlations(transf_diff.fit_transform(log_matlab_ts_q)) #usar log_matlab_ts


# ## Continuación: Modelos ARIMA
# 
# Vamos a separar la muestra en la parte de estiamción (Training) y la parte de predicción/Verificación (Testing). Quitamos 8 trimestres.
# 
# Vamos a predecir 8 periodos (h=8)
# 
# 

# In[105]:


# Sktime fucntions
# Forecast horizon and Split function 
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split


# In[106]:


# Select Train & test sample
# we will try to forecast y_test from y_train
# plotting for illustration
# original and log samples
y_train, y_test = temporal_train_test_split(y =matlab_ts_q, test_size=8)
log_y_train, log_y_test = temporal_train_test_split(y =log_matlab_ts, test_size=8)
plot_series(y_train, y_test, labels=["Train", "Test"])
# Time Series Size
print(y_train.shape[0], y_test.shape[0])


# In[107]:


# Forecast Horizon
fh = np.arange(len(y_test)) + 1  # forecasting horizon
fh


# In[108]:


# Sktime Auto ARIMA Function
from sktime.forecasting.arima import AutoARIMA


# In[109]:


#  Auto ARIMA Model

forecaster = AutoARIMA(sp=4,suppress_warnings=True)
forecaster.fit(log_y_train)


# In[110]:


#  Auto ARIMA Model Summary
print(forecaster.summary())


# In[ ]:


#  Auto ARIMA Forecast
log_y_pred = forecaster.predict(fh)
log_y_pred


# In[ ]:


#  Auto ARIMA Forecast
# Original Time series (Invert log transformation)
np.exp(log_y_pred)


# In[ ]:


# Sktime fucntions
# Forecast Accuracy MAPE & MSE & RMSE
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.performance_metrics.forecasting import MeanSquaredError


# In[ ]:


# Forecast Accuracy
# MAPE
mean_absolute_percentage_error(log_y_test, log_y_pred)


# In[ ]:


# Forecast Accuracy
# MAPE 
# Orignal Time Serie

mean_absolute_percentage_error(y_test, np.exp(log_y_pred))


# In[ ]:


# Forecast Accuracy
#Mean Squared Error
rmse = MeanSquaredError(square_root=True)
rmse(log_y_test, log_y_pred)


# In[ ]:


# Forecast Accuracy
#Mean Squared Error
rmse = MeanSquaredError(square_root=True)
rmse(y_test, np.exp(log_y_pred))


# In[ ]:


# Forecast Accuracy Plot
# Plotting predictions and past data
plot_series(y_train, np.exp(log_y_pred),y_test, labels=["Matlab", "Matlab pred", "Matlab REAL"])


# In[ ]:


# Forecast Accuracy Plot
# Plotting predictions and past data
# Zoom 2013 -2021
plot_series(y_train["2013":], np.exp(log_y_pred),y_test, labels=["Matlab", "Matlab pred", "Matlab REAL"])


# In[ ]:


# Save Plot
plot_series(y_train["2013":], np.exp(log_y_pred),y_test, labels=["Matlab", "Matlab pred", "Matlab REAL"])
plt.savefig('img/04/predict-AutoARIMA-airline-data-plot.png',
            dpi=300, bbox_inches='tight')
plt.close('all')


# In[ ]:


# Forecast Horizon
fh = np.arange(6) + 1  # forecasting horizon
fh


# In[ ]:


#  Auto ARIMA Model

forecaster = AutoARIMA(sp=4,suppress_warnings=True)
forecaster.fit(log_matlab_ts)


# In[ ]:


#  Auto ARIMA Model Summary
print(forecaster.summary())


# In[ ]:


#  Auto ARIMA Forecast
log_y_pred = forecaster.predict(fh)
log_y_pred


# In[ ]:


#  Auto ARIMA Forecast
# Original Time series (Invert log transformation)
np.exp(log_y_pred)


# In[ ]:


# Forecast Accuracy Plot
# Plotting predictions and past data
plot_series(matlab_ts_q, np.exp(log_y_pred), labels=["Matlab", "Matlab pred"])


# In[ ]:


# Forecast Accuracy Plot
# Plotting predictions and past data, el año ???
plot_series(matlab_ts_q["2016":], np.exp(log_y_pred), labels=["Matlab", "Matlab pred"])


# ## A partir de aqui forma Chat GPT

# In[64]:


pip install -U sktime


# In[113]:


import matplotlib.pyplot as plt

# ... (código previo para entrenar el modelo ARIMA y hacer predicciones)

# Visualizar las predicciones junto con los datos de entrenamiento y prueba
plt.figure(figsize=(10, 6))

# Datos de entrenamiento
plt.plot(y_train, label='Train', color='blue')

# Datos de prueba
plt.plot(y_test.index, y_test, label='Test', color='green')

# Predicciones del modelo ARIMA
plt.plot(y_test.index, y_pred, label='ARIMA Predictions', color='red')

plt.legend()
plt.title('ARIMA Model Predictions')
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.show()


# # Validación y Ajuste de Parametros - ETS & ARIMA.
# 
# 1. Selección de métricas de rendimiento: Define las métricas de evaluación adecuadas para comparar y evaluar los modelos. Las métricas comunes incluyen el Error Cuadrático Medio (MSE), el Error Absoluto Medio (MAE), y el Error Porcentual Absoluto Medio (MAPE).
# 2. Ajuste de parámetros del modelo:
#     - **ARIMA:** Experimenta con distintas combinaciones de los parámetros p, d y q para encontrar la que mejor se ajuste a los datos. Utiliza métodos como la búsqueda en cuadrícula (grid search) o el método auto-ARIMA para encontrar la combinación óptima de parámetros.
#     - **ETS:** Ajusta los modelos ETS probando diferentes configuraciones de los componentes de error, tendencia y estacionalidad.
# 
# 3. División de los datos: Divide tu conjunto de datos en tres partes: entrenamiento, validación y prueba. Por ejemplo, puedes usar una proporción del 70-15-15, dejando un 70% para entrenar, un 15% para la validación y un 15% para probar.
# 
# 4. Entrenamiento y ajuste del modelo:
#     - Utiliza la parte de entrenamiento para ajustar los modelos ARIMA y ETS con los parámetros seleccionados.
#     - Aplica estos modelos a la muestra de validación para evaluar su rendimiento.
# 
# 5. Comparación y selección del mejor modelo:
#     - Evalúa el desempeño de los modelos utilizando las métricas definidas.
#     - Compara los resultados de ambos modelos para determinar cuál ofrece un mejor rendimiento en la muestra de validación.
# 
# 6. Refinamiento y reajuste del modelo:
#     - Si ninguno de los modelos alcanza un rendimiento satisfactorio, ajusta los parámetros o considera transformaciones adicionales en tus datos.
#     - Repite el proceso de ajuste y validación hasta que estés satisfecho con el rendimiento de uno de los modelos.
# 
# El objetivo principal es identificar el modelo que mejor se ajuste (TENER EN CUENTA QUE VAMOS A VER DOS MODELOS MÁS)a los datos y ofrezca predicciones precisas. Este proceso de ajuste de parámetros y validación te permitirá seleccionar el modelo más adecuado para predecir tu serie temporal.

# # Sección 4: Modelado - 4Theta & TBATS -- Domingo TBC (Retocar)
# 
# ## Modelo 4Theta: S06
# 
# El modelo 4Theta nos ayuda a predecir datos futuros en una serie temporal. Funciona mirando hacia el pasado para identificar patrones y tendencias en los datos. Usando estos patrones, intenta predecir qué sucederá en el futuro.
# El modelo 4Theta es un enfoque avanzado para predecir series temporales. Se basa en cuatro componentes principales: Tendencia, Estacionalidad, Ciclos y Eventos Irregulares. La Tendencia se refiere a la dirección general de la serie temporal; la Estacionalidad captura patrones que se repiten a lo largo del tiempo; los Ciclos representan cambios a largo plazo; y los Eventos Irregulares cubren fluctuaciones que no se pueden asignar a los componentes anteriores.
# 
# ## Modelo TBATS: S07
# 
# Al igual que el 4Theta, el modelo TBATS se utiliza para hacer predicciones en series temporales. Sin embargo, este modelo es bueno para manejar series temporales con patrones estacionales más complejos, como los datos que suben y bajan a lo largo del tiempo de una manera menos predecible.
# TBATS (Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend, and Seasonal components) es otro modelo avanzado de series temporales. Utiliza transformaciones Box-Cox para estabilizar la varianza y se enfoca en los componentes de tendencia y estacionalidad. La 'T' en TBATS indica la presencia de estacionalidad trigonométrica, que maneja estacionalidades múltiples y complejas.
# 
# ## Ajuste y Evaluación:
# 
# Al ajustar los modelos 4Theta y TBATS, estamos tratando de encontrar la mejor configuración para que funcionen bien con nuestros datos. Luego, usamos una muestra de datos que ya conocemos (datos de validación) para ver qué tan buenas son las predicciones de estos modelos. Si hacen buenas predicciones en datos que ya conocemos, es probable que hagan buenas predicciones sobre datos futuros que aún no hemos visto.
# 
# Al igual que en los modelos ETS y ARIMA, el ajuste de parámetros es crucial para afinar el rendimiento de los modelos 4Theta y TBATS. Es fundamental encontrar la configuración óptima para estos modelos, lo que incluye la identificación de la mejor manera de manejar la estacionalidad, los ajustes en la diferenciación y el manejo de los eventos irregulares. Luego, se evaluará el rendimiento de ambos modelos utilizando datos de validación para comprender su precisión predictiva y compararlos con los modelos anteriores.

# # Modelo 4Theta
# 
# ## Entrenamiento del modelo FourTheta:
# - Se define y se entrena el modelo utilizando la serie temporal de entrenamiento (train). Este modelo ajusta los parámetros y encuentra el patrón subyacente en los datos históricos.
# - Se efectúa una predicción sobre el conjunto de validación (val) para evaluar el rendimiento del modelo.
# 
# ## Ajuste de hiperparámetros:
# - Se realiza una búsqueda de hiperparámetros con el fin de encontrar la configuración más óptima para el modelo FourTheta. Se exploran diferentes combinaciones de parámetros, como el "theta", "model_mode", "season_mode" y "trend_mode".
# - Se ajusta un modelo usando los mejores hiperparámetros encontrados.
# 
# ## Validación cruzada:
# - Se realiza una validación cruzada utilizando la función historical_forecasts para verificar el rendimiento del modelo a lo largo de diferentes ventanas de tiempo históricas.
# - Se obtienen los errores de pronóstico y se muestran mediante un histograma para comprender la distribución de los errores.
# ## Backtesting:
# - Se efectúa una evaluación de backtesting con la función backtest, lo que significa generar pronósticos retrospectivos sobre la serie temporal y compararlos con los valores reales, obteniendo una medida del error para cada punto de predicción.
# - Se calcula el error promedio de estos puntos de predicción.

# ## Apuntes de la sesion 6: Copiarlo tal cual y adaptarlo

# In[114]:


conda update conda


# In[115]:


python3.8 -m venv myenv


# In[26]:


source myenv/bin/activate


# In[27]:


pip install virtualenv


# In[28]:


virtualenv -p python3.8 myenv


# In[29]:


source myenv/bin/activate


# In[30]:


myenv\Scripts\activate


# In[31]:


pip install darts


# In[116]:


import warnings
warnings.filterwarnings('ignore')


# Es necesario tener instalado DARTS
# 
# Darts: https://unit8co.github.io/darts/index.html
# 

# In[117]:


import darts


# In[118]:


#%%
import darts 
from darts import TimeSeries
from darts.datasets import AirPassengersDataset


# In[120]:


from darts.models import FourTheta

model = FourTheta(seasonality_period=12)
model.fit(train)
forecast = model.predict(len(val))

print("model {} obtains MAPE: {:.2f}%".format(model, mape(val, forecast)))


# In[121]:


# %%
train.plot(label="train")
val.plot(label="true")
forecast.plot(label="prediction")


# In[ ]:


from darts.utils.utils import SeasonalityMode, TrendMode, ModelMode
theta_grid = {
    #'theta':2- np.linspace(-10, 10, 10),
    'theta':[-4,-3,-2,-1,1,2,3,4],
    'model_mode': [ModelMode.ADDITIVE,ModelMode.MULTIPLICATIVE],
    'season_mode': [SeasonalityMode.MULTIPLICATIVE,SeasonalityMode.ADDITIVE],
    'trend_mode': [TrendMode.EXPONENTIAL,TrendMode.LINEAR]
}

best_grid_model=FourTheta.gridsearch(parameters=theta_grid,
                                series=train,
                                forecast_horizon=36, # 12
                                start=0.5,
                                last_points_only=False,
                                metric=mape,
                                reduction=np.mean,
                                verbose=True,
                                n_jobs=-1)

best_grid_model


# In[ ]:


modelo=FourTheta(theta=best_grid_model[1]['theta'],
                 model_mode=ModelMode.ADDITIVE,
                 season_mode=SeasonalityMode.MULTIPLICATIVE,
                 trend_mode=TrendMode.LINEAR
                 )
modelo.fit(train)
pred_modelo = modelo.predict(len(val))

train.plot(label="train")
val.plot(label="true")
pred_modelo.plot(label="prediction")


# In[ ]:


print("model {} obtains MAPE: {:.2f}%".format(modelo, mape(val, pred_modelo)))


# In[ ]:


# %%
train.plot(label="train")
val.plot(label="true")
pred_modelo.plot(label="prediction 4Theta")
pred_best_theta.plot(label="prediction Theta")


# ## Cross Validation Historical

# In[ ]:


# %%
historical_fcast_theta = best_theta_model.historical_forecasts(
    series, start=0.5, forecast_horizon=36, verbose=True, stride=1
)

series.plot(label="data")
historical_fcast_theta.plot(label="backtest 12-months ahead forecast (Theta)")
print("MAPE = {:.2f}%".format(mape(historical_fcast_theta, series)))


# ## BackTest

# In[ ]:


# %%


raw_errors = best_theta_model.backtest(
    series, 
    start=0.4, 
    forecast_horizon=12, 
    metric=mape, 
    reduction=None, # None: return errors
    verbose=True
)

from darts.utils.statistics import plot_hist

plot_hist(
    raw_errors,
    bins=np.arange(0, max(raw_errors), 1),
    title="Individual backtest error scores (histogram)",
)



# In[ ]:


# %%
average_error = best_theta_model.backtest(
    series,
    start=0.4,
    forecast_horizon=12,
    metric=mape,
    reduction=np.mean,  # this is actually the default
    verbose=True,
)

print("Average error (MAPE) over all historical forecasts: %.2f" % average_error)




# In[ ]:


average_error


# In[ ]:


raw_errors


# ## Hasta aqui los apuntes de la sesion 6 -- apartir de aqui otras ideas

# # 1. Preparación de los datos

# In[123]:


from darts import TimeSeries
# Suponiendo que 'matlab_date_df' tiene una columna 'value' para los datos de la serie temporal
ts = TimeSeries.from_dataframe(matlab_dates_df, time_col='datetime_column', value_cols='value_column')


# # 2. Entrenamiento del modelo 4Theta

# In[124]:


from darts.models import FourTheta

# Divide los datos en conjuntos de entrenamiento y validación
train, val = ts.split_before(pd.Timestamp('fecha de separacion'))

# Crea y entrena el modelo FourTheta
model = FourTheta()
model.fit(train)


# # 3. Predicción y Evaluación del modelo

# In[ ]:


# Realiza predicciones con el modelo
forecast = model.predict(len(val))

# Evalúa el rendimiento del modelo utilizando MAPE (Error Porcentual Absoluto Medio)
from darts.metrics import mape
error = mape(val, forecast)
print(f"Modelo FourTheta obtiene un MAPE: {error:.2f}%")


# # 4. Ajuste de Hiperparámetros

# In[ ]:


from darts.models import FourThetaParams

# Define la cuadrícula de búsqueda de hiperparámetros
theta_grid = {
    'theta': [-4, -3, -2, -1, 1, 2, 3, 4],
    'model_mode': ['additive', 'multiplicative'],
    'season_mode': ['multiplicative', 'additive'],
    'trend_mode': ['exponential', 'linear']
}

# Realiza la búsqueda de cuadrícula para encontrar los mejores hiperparámetros
best_grid_model = FourThetaParams.gridsearch(
    theta_grid,
    train,
    forecast_horizon=12,
    metric=mape,
    reduction=np.mean,
    verbose=True
)


# # Modelo TBATS - Procedimiento TBC (Editar)

# ## 1. Lectura y Preparación de Datos

# In[126]:


from darts import TimeSeries

# Cargar tu conjunto de datos a una serie temporal
# Reemplaza 'time_column' por el nombre de tu columna de tiempo y 'value_column' por la columna de valores en tu conjunto de datos
series = TimeSeries.from_dataframe(matlab_dates_df, time_col='time_column', value_cols='value_column')

# Dividir los datos en conjunto de entrenamiento y prueba
train, test = series.split_before(pd.Timestamp('fecha de división'))


# ## 2. Entrenamiento del Modelo TBATS

# ## Apuntes sesion 7 TBATS - copiar y adaptarlos tal cual

# In[128]:


# Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[129]:


# %%
from darts import TimeSeries
from darts.datasets import AirPassengersDataset


# In[130]:


# %%
series = AirPassengersDataset().load()
series


# In[131]:


import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,6)
series.plot()


# In[132]:


# %%
train, val = series.split_before(pd.Timestamp("19580101"))
train.plot(label="training")
val.plot(label="validation")


# In[133]:


from darts.models import TBATS
from darts.metrics import mape


# In[134]:


#@title

# %%
#from darts.models import TBATS
#from darts.metrics import mape

model =TBATS(
use_box_cox=None,
box_cox_bounds=(0, 1),
use_trend=None,
use_damped_trend=None,
seasonal_periods="freq",
use_arma_errors=True,
show_warnings=False,
multiprocessing_start_method='spawn',
random_state=0)
model.fit(train)
forecast = model.predict(len(val))
print("model {} obtains MAPE: {:.2f}%".format(model, mape(val, forecast)))




# In[135]:


model.model.params.summary()


# In[136]:


# %%
train.plot(label="train")
val.plot(label="true")
forecast.plot(label="prediction")


# In[137]:


from sktime.datasets import load_airline
from sktime.forecasting.tbats import TBATS # MODELO TBATS
from sktime.forecasting.base import ForecastingHorizon
from sktime.utils.plotting import plot_series
from sktime.forecasting.model_selection import temporal_train_test_split
y = load_airline()

y_train, y_test = temporal_train_test_split(y =y , test_size=36)
# we will try to forecast y_test from y_train
# plotting for illustration
plot_series(y_train, y_test, labels=["y_train", "y_test"])
print(y_train.shape[0], y_test.shape[0])


# In[138]:


forecaster = TBATS(  
    use_box_cox=None,
    use_trend=None,
    use_damped_trend=None,
    sp=12,
    use_arma_errors=True,
    n_jobs=1)
forecaster.fit(y_train)  
# TBATS(...)


# In[139]:


import numpy as np
y_pred = forecaster.predict(fh=np.arange(1, 37))


# In[140]:


# optional: plotting predictions and past data
plot_series(y_train, y_pred,y_test, labels=["y", "pred", "REAL"])


# In[141]:


# Sktime fucntions
# Forecast Accuracy MAPE & MSE & RMSE
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
# Forecast Accuracy
# MAPE
mean_absolute_percentage_error(y_test, y_pred)


# ## Cross Validations Historical

# In[142]:


from darts.models import TBATS
from darts.metrics import mape


# In[143]:


# %%
historical_fcast_tbats = model.historical_forecasts(
    series, start=0.4, forecast_horizon=12, verbose=True, stride=1
)

series.plot(label="data")
historical_fcast_tbats.plot(label="backtest 12-months ahead forecast (Tbats)")
print("MAPE = {:.2f}%".format(mape(historical_fcast_tbats, series)))




# In[148]:


model_best =TBATS(
use_box_cox=True,
box_cox_bounds=(0, 0),
use_trend=True,
use_damped_trend=False,
seasonal_periods="freq",
use_arma_errors=False,
show_warnings=False,
multiprocessing_start_method='spawn',
random_state=0)


# In[151]:


# %%
historical_fcast_tbats_best = model_best.historical_forecasts(
    series, start=0.4, forecast_horizon=12, verbose=True, stride=1
)

series.plot(label="data")
historical_fcast_tbats_best.plot(label="backtest 12-months ahead forecast (Tbats)")
print("MAPE = {:.2f}%".format(mape(historical_fcast_tbats_best, series)))


# # BackTest

# In[150]:


# %%

raw_errors = model_best.backtest(
    series, 
    start=0.4, 
    forecast_horizon=12, 
    metric=mape, 
    reduction=None, # None: return errors
    verbose=True
)

from darts.utils.statistics import plot_hist

plot_hist(
    raw_errors,
    bins=np.arange(0, max(raw_errors), 1),
    title="Individual backtest error scores (histogram)",
)



# In[152]:


# %%
average_error = model.backtest(
    series,
    start=0.4,
    forecast_horizon=12,
    metric=mape,
    reduction=np.mean,  # this is actually the default
    verbose=True,
)
print("Average error (MAPE) over all historical forecasts: %.2f" % average_error)


# In[153]:


average_error


# In[154]:


raw_errors


# ### Paso 1: Entrenamiento del Modelo TBATS
# #### 1. Definición y Entrenamiento del Modelo:
# Define y entrena el modelo TBATS utilizando la serie temporal de entrenamiento (train). Este modelo busca ajustar los parámetros y encontrar los patrones subyacentes en los datos históricos.
# Ajusta el modelo al conjunto de entrenamiento para que capture la estacionalidad, tendencias y otros patrones en los datos.

# In[155]:


from darts.models import TBATS
from darts.metrics import mape

# Define el modelo TBATS con los parámetros adecuados para tu problema
model_tbats = TBATS(
    use_box_cox=None,
    box_cox_bounds=(0, 1),
    use_trend=None,
    use_damped_trend=None,
    seasonal_periods=("diario", "anual"),  # Define las estacionalidades apropiadas
    use_arma_errors=True,
    show_warnings=False,
    multiprocessing_start_method='spawn',
    random_state=0
)

# Entrena el modelo con los datos de entrenamiento
model_tbats.fit(train)


# #### 2. Predicción y Evaluación:
# - Realiza predicciones sobre el conjunto de validación (val) para evaluar el rendimiento del modelo entrenado.
# - Calcula el Error Porcentual Absoluto Medio (MAPE) para evaluar el rendimiento del modelo.

# In[157]:


# Realiza predicciones con el modelo sobre el conjunto de validación
forecast_tbats = model_tbats.predict(len(val))

# Calcula el Error Porcentual Absoluto Medio (MAPE) para evaluar el rendimiento
error_tbats = mape(val, forecast_tbats)
print(f"El modelo TBATS obtiene un MAPE de: {error_tbats:.2f}%")


# ### Paso 2: Ajuste de Hiperparámetros
# 
# #### Búsqueda y Ajuste de Hiperparámetros:
# Realiza una búsqueda exhaustiva de hiperparámetros para encontrar la configuración óptima del modelo TBATS.
# 
# Explora diferentes combinaciones de parámetros como el "use_box_cox", "use_trend", "use_damped_trend", entre otros.
# Ajusta un modelo con los mejores hiperparámetros encontrados.
# (La búsqueda de hiperparámetros puede variar según la implementación de la librería Darts para el modelo TBATS.)
# 

# In[ ]:


from darts.models import TBATS
from darts.metrics import mape

# Definir una grilla de parámetros para buscar
tbats_params_grid = {
    'use_box_cox': [True, False],
    'use_trend': [True, False],
    'use_damped_trend': [True, False],
    # Agregar otros parámetros que desees buscar
    # ...
}

# Implementar la búsqueda de hiperparámetros
best_error = float('inf')
best_params = None

for box_cox in tbats_params_grid['use_box_cox']:
    for trend in tbats_params_grid['use_trend']:
        for damped_trend in tbats_params_grid['use_damped_trend']:
            # Crea y entrena el modelo con una combinación de parámetros
            model = TBATS(
                use_box_cox=box_cox,
                use_trend=trend,
                use_damped_trend=damped_trend,
                # Otros parámetros de la grilla
                # ...
            )
            model.fit(train)
            
            # Realiza predicciones y calcula el MAPE
            forecast = model.predict(len(val))
            error = mape(val, forecast)
            
            # Actualiza si se encuentra un nuevo mejor modelo
            if error < best_error:
                best_error = error
                best_params = {
                    'use_box_cox': box_cox,
                    'use_trend': trend,
                    'use_damped_trend': damped_trend,
                    # Otras configuraciones
                    # ...
                }

# Entrena un modelo con los mejores parámetros encontrados
best_model = TBATS(
    use_box_cox=best_params['use_box_cox'],
    use_trend=best_params['use_trend'],
    use_damped_trend=best_params['use_damped_trend'],
    # Otras configuraciones
    # ...
)
best_model.fit(train)


# ### Paso 3: Validación Cruzada y Backtesting
# 
# #### Validación Cruzada:
# Utiliza la función historical_forecasts para realizar una validación cruzada y verificar el rendimiento del modelo a lo largo de diferentes ventanas de tiempo históricas.
# 
# Obtiene los errores de pronóstico y muestra la distribución de los errores mediante un histograma para comprender su variabilidad.
# 
# #### Backtesting:
# Realiza una evaluación de backtesting con la función backtest, generando pronósticos retrospectivos sobre la serie temporal y comparándolos con los valores reales para obtener medidas de error para cada punto de predicción.
# 
# Calcula el error promedio de estos puntos de predicción para evaluar la precisión del modelo en un contexto retrospectivo.

# In[158]:


# Validación Cruzada
# En darts se usa la función 'historical_forecasts' para la validación cruzada

historical_forecasts = best_model.historical_forecasts(
    series, start=0.4, forecast_horizon=12, verbose=True, stride=1
)
# Visualización de los resultados
series.plot(label="data")
historical_forecasts.plot(label="backtest 12-months ahead forecast (TBATS)")

# Backtesting
# Para backtest en darts se emplea la función 'backtest'

raw_errors = best_model.backtest(
    series,
    start=0.4,
    forecast_horizon=12,
    metric=mape,
    reduction=None,  # None: return errors
    verbose=True
)

# Muestra de los errores a través de un histograma
from darts.utils.statistics import plot_hist
plot_hist(
    raw_errors,
    bins=np.arange(0, max(raw_errors), 1),
    title="Individual backtest error scores (histogram)",
)

# Cálculo del error promedio
average_error = best_model.backtest(
    series,
    start=0.4,
    forecast_horizon=12,
    metric=mape,
    reduction=np.mean,  # this is actually the default
    verbose=True
)

print("Error promedio (MAPE) en todos los pronósticos históricos: %.2f" % average_error)


# ## 3. Realizar Predicciones y Evaluación

# In[159]:


# Realizar predicciones con el modelo entrenado
forecast = model.predict(len(test))

# Calcular el MAPE (Error Porcentual Absoluto Medio)
error = mape(test, forecast)
print(f"Modelo TBATS obtiene MAPE: {error:.2f}%")

# Visualizar los resultados
train.plot(label="train")
test.plot(label="true")
forecast.plot(label="prediction")


# # Sección 4. Ajuste y Evaluación: 
# Ajustamos los parámetros y evalúa el rendimiento de estos modelos utilizando la muestra de validación.

# In[160]:


# Modelo 4Theta - Ajuste y evaluación
from darts.models import FourTheta

# Ajuste del modelo 4Theta
model_4theta = FourTheta(seasonality_period=12)
model_4theta.fit(train)

# Predicción del modelo 4Theta en la muestra de validación
forecast_4theta = model_4theta.predict(len(val))

# Evaluación del rendimiento del modelo 4Theta
mape_4theta = mape(val, forecast_4theta)
print(f"El modelo 4Theta obtiene un MAPE: {mape_4theta:.2f}%")

# Modelo TBATS - Ajuste y evaluación
from darts.models import TBATS

# Ajuste del modelo TBATS
model_tbats = TBATS(use_box_cox=True, use_trend=True, use_damped_trend=True)  # Ajusta los parámetros apropiados
model_tbats.fit(train)

# Predicción del modelo TBATS en la muestra de validación
forecast_tbats = model_tbats.predict(len(val))

# Evaluación del rendimiento del modelo TBATS
mape_tbats = mape(val, forecast_tbats)
print(f"El modelo TBATS obtiene un MAPE: {mape_tbats:.2f}%")


# # Sección Final : Comparación de Modelos
# Vamos a observar diferentes maneras de predecir el futuro.
# Seleccionaremos el método que creemos funcionará mejor para hacer predicciones correctas.
# 
# ## Predicción de 12 Meses:
# Usaremos el método seleccionado para adivinar qué pasará en los próximos 12 meses.
# Luego, verificaremos si nuestras suposiciones son correctas o si necesitamos ajustar nuestras predicciones.
# ## Documentación y Presentación:
# cómo hicimos las predicciones y qué hemos aprendido.
# Responder a la pregunta de la introducción: ¿Podríamos vivir sin StackOverflow?

# ### A. Predicción de 12 Meses:
# Suponiendo que has seleccionado el modelo o método basado en los resultados anteriores, puedes realizar la predicción para los próximos 12 meses.

# In[165]:


# Predicción de 12 meses con el modelo seleccionado
modelo_seleccionado = model_tbats  # Coloca aquí el modelo que hayas seleccionado (4Theta, TBATS, etc.)

# Ajusta el modelo seleccionado a todos los datos disponibles
model_tbats.fit(train) #train!!!???

# Realiza la predicción para los próximos 12 meses
prediccion_12_meses = model_tbats.predict(12)  # 12 meses hacia adelante

# Muestra la predicción
print(prediccion_12_meses)


# In[166]:


# Predicción de 12 meses con el modelo seleccionado # Coloca aquí el modelo que hayas seleccionado (4Theta, TBATS, etc.)

# Ajusta el modelo seleccionado a todos los datos disponibles
model_4theta.fit(train) #train!!!???

# Realiza la predicción para los próximos 12 meses
prediccion_12_meses = model_4theta.predict(12)  # 12 meses hacia adelante

# Muestra la predicción
print(prediccion_12_meses)


# ### B. Documentación:
# #### Métodos utilizados: 
# Explica brevemente los modelos y métodos que has aplicado para hacer predicciones.
# #### Resultados de la predicción: 
# Muestra las predicciones realizadas por el modelo seleccionado para los próximos 12 meses. Puedes mostrar gráficamente estas predicciones para que sean visualmente claras.
# #### Evaluación del rendimiento:
# Comenta sobre la precisión de las predicciones en función de las métricas de rendimiento que hayas utilizado anteriormente, como el MAPE.
# #### Lecciones aprendidas: 
# Discute las lecciones que has aprendido durante este proceso. Esto podría incluir la identificación de modelos que funcionan mejor, desafíos encontrados, métodos de ajuste de parámetros y selección de modelos, etc.
# #### Respuesta a la pregunta de la introducción: 
# 
# Reflexiona sobre si pudieras vivir sin StackOverflow. ¿Cómo ha influido o facilitado la búsqueda de soluciones durante este proceso?
# 
# SMART METHOD Y REP GRAF DE RESULTADOS , PUEDE INCLUIR:
# Explicación con gráficos, métricas de rendimiento y reflexiones sobre el proceso es fundamental para comunicar hallazgos y conclusiones.
