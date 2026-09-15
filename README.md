# Dashboard inteligente de inventarios — Bitrodiagnóstico

## Ejecución local
1. Instalar Python 3.10 o superior.
2. Abrir terminal en esta carpeta.
3. Ejecutar `pip install -r requirements.txt`.
4. Ejecutar `streamlit run app.py`.
5. Abrir la dirección local que muestra Streamlit.

## Ejecución y verificación
Primero usar **Resultados congelados de la tesis** y comprobar que aparecen los 12 SKU. Contrastar obligatoriamente BR 004015-1, BR 001235 y BR 002124/BR 004014. Después usar **Cargar archivos actualizados** con histórico, pronósticos y parámetros/inventario.

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
