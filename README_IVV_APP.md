# IVV Tactical Monte Carlo

Aplicacion de investigacion para evaluar una estrategia de compras escalonadas
durante caidas de IVV y venta durante la recuperacion.

## Modelo

- Horizonte de 63 sesiones, aproximadamente tres meses.
- Escenarios: 15% adverso, 70% central y 15% favorable.
- Innovaciones Student-t para colas pesadas.
- Volatilidad recursiva tipo GARCH.
- Saltos de precio condicionados al escenario.
- Supuestos de inflacion, tasas, petroleo y USD/MXN.
- Resultado de la estrategia en USD y MXN.
- Comparacion contra comprar y mantener IVV.
- Calibracion historica de retorno, volatilidad, colas y escenarios.
- Backtesting walk-forward sin utilizar informacion futura.
- Paper trading operativo sin conexion a un broker.
- Ordenes limite con vigencia, bid/ask, deslizamiento y comisiones simuladas.
- Limites de exposicion y perdida maxima con bitacora de controles.
- Graficos Plotly interactivos con zoom, hover y descarga.

La app permite alternar entre supuestos manuales y parametros historicamente
calibrados. La inflacion permanece como supuesto de baja frecuencia.

## Instalacion

```bash
python -m pip install -r requirements-ivv-app.txt
```

## Ejecucion

```bash
streamlit run app_ivv_trading.py
```

## Publicacion

La ruta recomendada para una primera version publica es Streamlit Community
Cloud. Consulta `DEPLOY_STREAMLIT.md` para subirla desde GitHub y obtener una
URL compartible.

## Calibracion y backtesting

La calibracion usa IVV, WTI, Treasury a 10 anos y USD/MXN. Estima:

- Retorno y volatilidad anualizados.
- Grados de libertad de Student-t a partir de curtosis.
- Sensibilidades macro mediante regresion ridge.
- Correlacion entre IVV y USD/MXN.
- Regimen adverso, central y favorable mediante retornos moviles.
- Frecuencia, media y dispersion de saltos.

El walk-forward calibra cada origen exclusivamente con observaciones anteriores,
simula los siguientes 63 dias y compara P5-P95 con el resultado observado.

## Paper trading operativo

La seccion de paper trading recorre una trayectoria adversa, mediana o favorable
como si fueran sesiones de mercado. Genera apertura, maximo, minimo, bid y ask
simulados para evaluar:

- Creacion, ejecucion, vencimiento y rechazo de ordenes limite.
- Participaciones completas o fraccionadas.
- Costos de operacion y deslizamiento.
- Take-profit, stop-loss y trailing stop.
- Exposicion maxima y detencion por perdida de cartera.
- Evolucion del patrimonio y bitacora de cada decision.
- Comparacion contra una compra en el minimo y venta en el maximo posterior,
  calculadas retrospectivamente con los mismos costos.
- Porcentaje del beneficio teorico capturado por las reglas ejecutables.

Este modulo no contiene credenciales, integraciones ni funciones para enviar
ordenes a un broker. Los datos intradia y las ejecuciones son simulados. La
operacion teorica perfecta usa informacion futura y solo funciona como referencia.

## Siguiente etapa profesional

1. Sustituir Yahoo Finance por fuentes oficiales y almacenamiento versionado.
2. Incorporar CPI y expectativas de inflacion respetando su frecuencia mensual.
3. Estimar GARCH con maxima verosimilitud y validar residuos.
4. Etiquetar eventos geopoliticos historicos y calibrar saltos condicionados.
5. Incorporar impuestos, deslizamiento y reglas de administracion de riesgo.
