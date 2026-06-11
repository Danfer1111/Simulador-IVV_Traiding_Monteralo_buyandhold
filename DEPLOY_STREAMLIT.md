# Publicar en Streamlit Community Cloud

## Archivos necesarios

Sube estos archivos al mismo repositorio de GitHub:

- `app_ivv_trading.py`
- `ivv_montecarlo_engine.py`
- `ivv_calibration_backtest.py`
- `requirements.txt`
- `.streamlit/config.toml`
- `README_IVV_APP.md`

Los simuladores GNP y los archivos de pruebas pueden conservarse localmente,
pero no son necesarios para ejecutar esta aplicacion.

## Crear el repositorio

1. Crea un repositorio nuevo en GitHub.
2. Puede ser publico o privado si tu cuenta de Streamlit tiene acceso.
3. Sube los archivos indicados.
4. No subas contrasenas, tokens ni `.streamlit/secrets.toml`.

## Desplegar

1. Entra a `https://share.streamlit.io`.
2. Inicia sesion con GitHub.
3. Selecciona `Create app`.
4. Elige el repositorio y la rama principal.
5. En `Main file path` escribe `app_ivv_trading.py`.
6. Elige una URL disponible y presiona `Deploy`.

La aplicacion quedara disponible en una direccion similar a:

```text
https://nombre-de-tu-app.streamlit.app
```

## Limitaciones del plan gratuito

- La aplicacion puede entrar en reposo despues de varias horas sin visitas.
- La primera visita posterior puede tardar mientras vuelve a iniciar.
- Varias simulaciones simultaneas pueden consumir la memoria disponible.
- Para uso compartido se recomiendan 10,000 o 20,000 trayectorias.
- El backtesting es mas pesado que una simulacion individual.

## Actualizaciones

Cada cambio enviado a la rama desplegada de GitHub actualizara la aplicacion.
Si cambia `requirements.txt`, Streamlit reinstalara las dependencias.
