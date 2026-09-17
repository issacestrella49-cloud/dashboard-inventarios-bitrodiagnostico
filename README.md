# Dashboard inteligente de inventarios — Bitrodiagnóstico

## Ejecución local
1. Instalar Python 3.10 o superior.
2. Abrir terminal en esta carpeta.
3. Ejecutar `pip install -r requirements.txt`.
4. Ejecutar `streamlit run app.py`.
5. Abrir la dirección local que muestra Streamlit.

## Ejecución y verificación
Primero usar **Resultados congelados de la tesis** y comprobar que aparecen los 12 SKU. Contrastar obligatoriamente BR 004015-1, BR 001235 y BR 002124/BR 004014. Después usar **Cargar archivos actualizados** con histórico, pronósticos y parámetros/inventario.

## Flujo de carga de datos (modo "Cargar archivos actualizados")
1. **Datos base (una sola vez)**: histórico semanal y parámetros de costo/lead time. Se guardan en el servidor (`data_base/`) para no tener que volver a subirlos.
2. **Actualizar inventario (cada vez)**: solo un archivo liviano con Código + cantidad en stock (opcionalmente pedidos en tránsito/pendientes). El sistema recalcula todo usando los datos base ya guardados.

Nota: en hosting gratuito (Streamlit Community Cloud) los datos base guardados pueden perderse si la app se reinicia por inactividad prolongada; en ese caso hay que volver a subirlos una vez.

## Pronóstico con Machine Learning
Ya no se sube un archivo de pronósticos. Con el histórico guardado, el sistema reentrena automáticamente el modelo ganador que la Fase 2 de la tesis validó para cada SKU (Ingenuo, Promedio histórico, Croston, Holt-Winters, XGBoost o Random Forest — ver columna `Modelo` en `METRICAS`) y genera el pronóstico de las siguientes 12 semanas. La elección del modelo ganador por SKU no se vuelve a hacer (viene de la validación estadística ya hecha en la tesis); solo se automatiza su ejecución sobre el histórico vigente. El resultado se cachea (`st.cache_data`) para no reentrenar en cada clic.

## Panel de Decisión (pestaña nueva)
Muestra por SKU: clasificación ABC/FSN (fuente: análisis histórico de la tesis, Fase 1), demanda predicha (semanal/anual), cantidad óptima a pedir (EOQ), punto de reorden (ROP), stock actual, riesgo de sobrestock, riesgo de quiebre y fecha sugerida de pedido. Los indicadores de riesgo y la fecha sugerida son reglas adicionales de apoyo a la decisión — no forman parte de ni modifican las fórmulas congeladas de la metodología.

## Fórmulas congeladas
- D = promedio del pronóstico de 12 semanas × 52.
- e_t = y_t - yhat_t.
- sigma_e = SD(e_t).
- SS = Z × sigma_e × sqrt(LT en semanas).
- ROP = demanda pronosticada durante LT + SS.
- H = costo unitario × tasa anual de mantenimiento.
- EOQ = sqrt(2 × D × S / H).
- IP = inventario físico + pedidos en tránsito - pedidos pendientes/reservados.

RMSE se conserva como métrica de pronóstico y no reemplaza sigma_e en el SS. Si no hay pedidos en tránsito/pendientes, la alerta basada en inventario físico es preliminar.

## Validación que deben documentar
Registrar para cada prueba: SKU, valor en tesis, valor en web, diferencia, causa y corrección. No declarar superioridad del sistema hasta ejecutar el backtesting comparativo de la siguiente fase.
