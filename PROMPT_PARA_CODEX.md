# Prompt para Codex

Revisa `app.py` como ingeniero senior de Python/Streamlit y auditor de reproducibilidad. Verifica ejecución, imports, Excel, nulos, tipos de datos y las 8 pestañas. NO rediseñes la metodología ni cambies modelos ganadores.

Comprueba exactamente: D=promedio pronóstico 12 semanas×52; sigma_e=SD de residuos fuera de muestra; SS=Z×sigma_e×sqrt(LT semanas); ROP=demanda pronosticada durante LT+SS; H=costo unitario×tasa anual; EOQ=sqrt(2DS/H); IP=inventario físico+pedidos en tránsito-pedidos pendientes. RMSE solo es métrica predictiva, nunca sustituto de sigma_e.

Prueba BR 004015-1 contra la versión congelada: demanda LT≈2927.14; SS≈29110.30; ROP≈32037.44; inventario físico=683; EOQ reportado≈1667. Confirma que REABASTECER sea alerta, no orden automática. Si faltan datos de IP, debe decir alerta preliminar.

Si encuentras errores, indica componente, causa, corrección y efecto. Devuelve lista de pruebas superadas/fallidas y modifica `app.py` únicamente si es necesario. No inventes datos.
