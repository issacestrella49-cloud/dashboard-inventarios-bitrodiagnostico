import io, math, re, unicodedata
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

st.set_page_config(page_title='Inventario Inteligente | Bitrodiagnóstico',page_icon='📦',layout='wide')
st.title('Sistema inteligente de gestión de inventarios')
st.caption('Modelo híbrido basado en Machine Learning y EOQ dinámico · Bitrodiagnóstico Cía. Ltda.')

PRODUCTOS={
'BR 001029':'ID CARD NEWBORD BIO RAD','BR 001235':'ID CARD DIACLON ABODVI-A1B BIO RAD','BR 001255':'ID CARD ABD/ABD BIORAD','BR 002124':'ID CARD RH-SUBGROUP+ K 4x12 BIO-RAD','BR 002125':'ID CARD RH-SUBGROUP+K BIO RAD','BR 004014':'ID CARD LISS/COOMBS KIT 4 x 12 BIO RAD','BR 004015':'ID CARD LISS/COOMBS BIO RAD','BR 004015-1':'TARJETA PARA PRUEBA CRUZADA / COMPATIBILIDAD','BR 004310':'ID DIACELL I II III 3X10ML BIO RAD','BR 004851':'ID DC SCREENING I BIO RAD','BR 009280':'ID DILUENT 2 500ML BIO RAD','DM BT248DOPE':'EQUIPO DE TRASFUSION DE SANGRE DEMOTECK'}
METRICAS=pd.DataFrame([
['BR 001029','Promedio histórico',815.19,1072.84,72.59,.675],['BR 001235','Croston',3014.80,4187.64,54.50,.74037],['BR 001255','Ingenuo',1411.12,2096.19,79.33,.553],['BR 002124','XGBoost',8.83,12.60,31.43,.902],['BR 002125','Ingenuo',1766.73,2606.30,133.93,.915],['BR 004014','XGBoost',4.38,5.65,23.22,.613],['BR 004015','Ingenuo',3850.04,5463.68,100.15,.962],['BR 004015-1','Ingenuo',6161.08,8429.30,103.53,.755],['BR 004310','Holt-Winters',16.33,21.63,75.39,.737],['BR 004851','XGBoost',240.20,326.00,120.44,.932],['BR 009280','Random Forest',16.46,18.04,74.28,.723],['DM BT248DOPE','Promedio histórico',2325.50,2923.39,113.57,.997]],columns=['Codigo','Modelo','MAE','RMSE','sMAPE','MASE'])
BASE=pd.DataFrame([
['BR 001029',4968.02,4439.08,9407.10,2008],['BR 001235',29746.90,17155.21,46902.10,6848],['BR 001255',128.57,7065.68,7194.26,30],['BR 002124',94.26,50.81,145.07,15],['BR 002125',7187.14,6675.98,13863.13,9610],['BR 004014',93.96,26.78,120.73,20],['BR 004015',16315.71,16553.23,32868.94,3807],['BR 004015-1',2927.14,29110.30,32037.44,683],['BR 004310',91.92,88.13,180.05,63],['BR 004851',827.71,1108.42,1936.13,29],['BR 009280',130.15,71.57,201.73,52],['DM BT248DOPE',8894.29,9859.32,18753.60,220]],columns=['Codigo','Demanda_LT','SS','ROP','Inventario_Fisico'])
SERV={'BR 001029':(.975,'Media'),'BR 001235':(.975,'Media'),'BR 001255':(.95,'Alta'),'BR 002124':(.99,'Baja'),'BR 002125':(.95,'Alta'),'BR 004014':(.99,'Baja'),'BR 004015':(.975,'Media'),'BR 004015-1':(.95,'Alta'),'BR 004310':(.975,'Media'),'BR 004851':(.95,'Alta'),'BR 009280':(.975,'Media'),'DM BT248DOPE':(.95,'Alta')}
FCONST={'BR 001029':1159.2143,'BR 001235':6940.94,'BR 001255':30.,'BR 002125':1677.,'BR 004015':3807.,'BR 004015-1':683.}
DATES=pd.date_range('2026-08-01',periods=12,freq='7D')
FD=[]
for c,v in FCONST.items():
 m=METRICAS.loc[METRICAS.Codigo.eq(c),'Modelo'].iloc[0]
 for d in DATES: FD.append([c,d,m,v])
FORECAST_DEMO=pd.DataFrame(FD,columns=['Codigo','Fecha','Modelo','Pronostico'])

def clean(x):
 x=''.join(c for c in unicodedata.normalize('NFD',str(x).lower()) if unicodedata.category(c)!='Mn')
 return re.sub(r'[^a-z0-9]+','_',x).strip('_')
def col(df,names):
 M={clean(c):c for c in df.columns}
 for n in names:
  if clean(n) in M:return M[clean(n)]
 for n in names:
  for k,v in M.items():
   if clean(n) in k:return v
 return None
def sheets(up):
 if up is None:return {}
 b=up.getvalue(); x=pd.ExcelFile(io.BytesIO(b)); return {s:pd.read_excel(io.BytesIO(b),sheet_name=s) for s in x.sheet_names}
def best(sh,groups):
 ans=None;bs=-1
 for n,d in sh.items():
  sc=sum(col(d,g) is not None for g in groups)
  if sc>bs:ans=d.copy();bs=sc
 return ans,bs
def std_forecast(sh):
 if not sh:return None
 d,s=best(sh,[['codigo','producto'],['fecha'],['pronostico','demanda semanal pronosticada','demanda pronosticada']])
 if s<2:return None
 cc=col(d,['codigo','producto']); cp=col(d,['pronostico','demanda semanal pronosticada','demanda pronosticada','forecast']); cf=col(d,['fecha','fecha futura']); cm=col(d,['modelo'])
 if not cc or not cp:return None
 o=pd.DataFrame({'Codigo':d[cc].astype(str).str.strip().str.upper(),'Pronostico':pd.to_numeric(d[cp],errors='coerce')})
 o['Fecha']=pd.to_datetime(d[cf],errors='coerce',dayfirst=True) if cf else pd.NaT;o['Modelo']=d[cm].astype(str) if cm else ''
 return o.dropna(subset=['Pronostico'])
def std_history(sh):
 if not sh:return None
 d,s=best(sh,[['codigo','producto'],['fecha'],['demanda','ventas']])
 if s<3:return None
 cc=col(d,['codigo','producto']);cf=col(d,['fecha']);cy=col(d,['demanda','ventas','cantidad'])
 o=pd.DataFrame({'Codigo':d[cc].astype(str).str.strip().str.upper(),'Fecha':pd.to_datetime(d[cf],errors='coerce',dayfirst=True),'Demanda':pd.to_numeric(d[cy],errors='coerce')})
 return o.dropna()
def std_params(sh):
 if not sh:return None
 d,s=best(sh,[['codigo'],['costo unitario'],['lead time'],['inventario actual']])
 if s<2:return None
 amap={'Codigo':['codigo'],'Costo_Unitario':['costo unitario'],'Tasa_Mantenimiento':['tasa mantenimiento anual'],'H':['h anual por unidad'],'Lead_Time_Dias':['lead time proveedor','lead time'],'Inventario_Fisico':['inventario actual fisico','inventario actual'],'MOQ':['moq','cantidad minima de compra'],'Pedidos_Transito':['pedidos en transito'],'Pedidos_Pendientes':['pedidos pendientes','backorders'],'Costo_Orden':['costo orden','costo de ordenar']}
 o=pd.DataFrame()
 for new,a in amap.items():
  c=col(d,a)
  if c:o[new]=d[c]
 if 'Codigo' not in o:return None
 o['Codigo']=o['Codigo'].astype(str).str.strip().str.upper()
 for c in o.columns[1:]:o[c]=pd.to_numeric(o[c],errors='coerce')
 return o
def integrate(vals,days):
 vals=np.asarray(vals,float);w=days/7;full=int(math.floor(w));frac=w-full
 if not len(vals):return np.nan
 if len(vals)<=full:vals=np.pad(vals,(0,full+1-len(vals)),mode='edge')
 return vals[:full].sum()+(vals[full]*frac if frac else 0)
def alert(ip,rop):
 if pd.isna(ip) or pd.isna(rop):return 'SIN DATOS'
 if ip<=rop:return 'REABASTECER'
 if ip<=1.2*rop:return 'VIGILANCIA'
 return 'NORMAL'
def xlsx_bytes(dic):
 b=io.BytesIO()
 with pd.ExcelWriter(b,engine='openpyxl') as w:
  for n,d in dic.items():d.to_excel(w,sheet_name=n[:31],index=False)
 return b.getvalue()

st.sidebar.header('Configuración')
modo=st.sidebar.radio('Modo',['Resultados congelados de la tesis','Cargar archivos actualizados'])
uh=uf=up=None
if modo.startswith('Cargar'):
 uh=st.sidebar.file_uploader('1. Histórico semanal (.xlsx)',type='xlsx')
 uf=st.sidebar.file_uploader('2. Pronósticos (.xlsx)',type='xlsx')
 up=st.sidebar.file_uploader('3. Parámetros / inventario (.xlsx)',type='xlsx')
Sdefault=st.sidebar.number_input('Costo por orden S ($)',.01,value=15.0)
LTdefault=st.sidebar.number_input('Lead time por defecto (días)',1,value=30)
hist=None;forecast=FORECAST_DEMO.copy();params=None
if modo.startswith('Cargar'):
 try:
  hist=std_history(sheets(uh)) if uh else None
  ff=std_forecast(sheets(uf)) if uf else None
  params=std_params(sheets(up)) if up else None
  if ff is not None and len(ff):forecast=ff
 except Exception as e:st.sidebar.error(str(e))

master=BASE.merge(METRICAS,on='Codigo',how='left');master['Producto']=master.Codigo.map(PRODUCTOS)
master['Nivel_Servicio']=master.Codigo.map(lambda c:SERV[c][0]);master['Variabilidad']=master.Codigo.map(lambda c:SERV[c][1]);master['Z']=master.Nivel_Servicio.map(lambda x:norm.ppf(x));master['Lead_Time_Dias']=30.;master['Lead_Time_Semanas']=master.Lead_Time_Dias/7
master['Sigma_e']=master.SS/(master.Z*np.sqrt(master.Lead_Time_Semanas));master['Posicion_Inventario']=master.Inventario_Fisico;master['Fuente_IP']='Inventario físico (alerta preliminar)';master['EOQ']=np.nan;master.loc[master.Codigo.eq('BR 004015-1'),'EOQ']=1667
if params is not None and len(params):
 p=params.drop_duplicates('Codigo',keep='last');master=master.merge(p,on='Codigo',how='left',suffixes=('','_new'))
 for c in ['Inventario_Fisico','Lead_Time_Dias']:
  if c+'_new' in master:master[c]=master[c+'_new'].fillna(master[c])
 master['Lead_Time_Semanas']=master.Lead_Time_Dias/7
 tr=master.Pedidos_Transito.fillna(0) if 'Pedidos_Transito' in master else 0;pe=master.Pedidos_Pendientes.fillna(0) if 'Pedidos_Pendientes' in master else 0
 if 'Pedidos_Transito' in master or 'Pedidos_Pendientes' in master:
  master['Posicion_Inventario']=master.Inventario_Fisico+tr-pe;master['Fuente_IP']='Posición de inventario'
 else:master['Posicion_Inventario']=master.Inventario_Fisico;master['Fuente_IP']='Inventario físico (alerta preliminar)'
 for i,r in master.iterrows():
  f=forecast[forecast.Codigo.eq(r.Codigo)].sort_values('Fecha')
  if len(f):
   vals=f.Pronostico.dropna().values;D=float(vals.mean()*52);dlt=integrate(vals,r.Lead_Time_Dias);ss=r.Z*r.Sigma_e*math.sqrt(r.Lead_Time_Dias/7);master.at[i,'Demanda_LT']=dlt;master.at[i,'SS']=ss;master.at[i,'ROP']=dlt+ss;master.at[i,'Demanda_Anualizada']=D
   H=r.get('H',np.nan);C=r.get('Costo_Unitario',np.nan);rate=r.get('Tasa_Mantenimiento',np.nan)
   if pd.isna(H) and pd.notna(C) and pd.notna(rate):H=C*rate
   So=r.get('Costo_Orden',np.nan);So=Sdefault if pd.isna(So) else So;moq=r.get('MOQ',1);moq=1 if pd.isna(moq) else moq
   if pd.notna(H) and H>0:master.at[i,'EOQ']=math.ceil(max(math.sqrt(2*D*So/H),moq))
master['Alerta']=[alert(a,b) for a,b in zip(master.Posicion_Inventario,master.ROP)];master['Brecha_ROP']=master.Posicion_Inventario-master.ROP
fm=forecast.groupby('Codigo').Pronostico.mean();master['Pronostico_Medio']=master.Codigo.map(fm);master['Advertencia']=np.where((master.Pronostico_Medio>0)&(master.Sigma_e/master.Pronostico_Medio>=1),'ALTA INCERTIDUMBRE','')

if modo.startswith('Resultados'):st.info('Modo académico: reproduce la Fase 3 congelada. Las alertas usan inventario físico y son preliminares.')
else:st.info('Modo operativo: recalcula con archivos cargados. Verificar fecha de corte y posición de inventario.')

tabs=st.tabs(['📊 Resumen','📦 Portafolio','🔮 Pronóstico','🚨 Inventario','🔎 Producto','🧪 Sensibilidad','🧠 Metodología','⬇️ Exportar'])
with tabs[0]:
 cs=st.columns(5);cs[0].metric('SKU',len(master));cs[1].metric('Reabastecer',(master.Alerta=='REABASTECER').sum());cs[2].metric('Vigilancia',(master.Alerta=='VIGILANCIA').sum());cs[3].metric('Normal',(master.Alerta=='NORMAL').sum());cs[4].metric('Alta incertidumbre',(master.Advertencia!='').sum())
 m=master[['Codigo','Posicion_Inventario','ROP']].melt('Codigo',var_name='Indicador',value_name='Unidades');st.plotly_chart(px.bar(m,x='Codigo',y='Unidades',color='Indicador',barmode='group',title='Inventario/IP frente al ROP'),use_container_width=True)
 st.write('EOQ = cuánto pedir; ROP = cuándo revisar/activar reabastecimiento; SS = protección frente a incertidumbre del pronóstico.')
with tabs[1]:
 sh=master[['Codigo','Producto','Modelo','MASE','sMAPE','Variabilidad','Nivel_Servicio','Sigma_e','SS','ROP','Inventario_Fisico','Posicion_Inventario','EOQ','Alerta','Advertencia']].copy();sh.Nivel_Servicio=(sh.Nivel_Servicio*100).round(1).astype(str)+' %';st.dataframe(sh,use_container_width=True,hide_index=True)
 f=px.bar(master.sort_values('MASE'),x='Codigo',y='MASE',color='Modelo',title='MASE por SKU');f.add_hline(y=1,line_dash='dash');st.plotly_chart(f,use_container_width=True)
with tabs[2]:
 c=st.selectbox('Producto',master.Codigo,key='f');ff=forecast[forecast.Codigo.eq(c)].sort_values('Fecha');hh=hist[hist.Codigo.eq(c)].sort_values('Fecha') if hist is not None else None;g=go.Figure()
 if hh is not None and len(hh):g.add_trace(go.Scatter(x=hh.Fecha,y=hh.Demanda,name='Histórico'))
 if len(ff):g.add_trace(go.Scatter(x=ff.Fecha,y=ff.Pronostico,name='Pronóstico',mode='lines+markers'))
 st.plotly_chart(g,use_container_width=True);st.dataframe(METRICAS[METRICAS.Codigo.eq(c)],hide_index=True,use_container_width=True);st.caption('RMSE es métrica predictiva; el SS final utiliza σₑ.')
with tabs[3]:
 st.dataframe(master[['Codigo','Demanda_LT','Sigma_e','Z','SS','ROP','Inventario_Fisico','Posicion_Inventario','Fuente_IP','EOQ','Brecha_ROP','Alerta']],use_container_width=True,hide_index=True);f=px.bar(master.sort_values('Brecha_ROP'),x='Codigo',y='Brecha_ROP',title='Brecha IP − ROP');f.add_hline(y=0,line_dash='dash');st.plotly_chart(f,use_container_width=True);st.warning('REABASTECER es una alerta de apoyo a la decisión, no una orden automática de compra.')
with tabs[4]:
 c=st.selectbox('SKU',master.Codigo,key='p');r=master[master.Codigo.eq(c)].iloc[0];st.subheader(c+' — '+r.Producto);cs=st.columns(4);cs[0].metric('Modelo',r.Modelo);cs[1].metric('MASE',f'{r.MASE:.3f}');cs[2].metric('ROP',f'{r.ROP:,.0f}');cs[3].metric('EOQ','N/D' if pd.isna(r.EOQ) else f'{r.EOQ:,.0f}');st.write(f"Inventario/IP: **{r.Posicion_Inventario:,.0f}** · SS: **{r.SS:,.0f}** · σₑ: **{r.Sigma_e:,.2f}** · Servicio: **{r.Nivel_Servicio*100:.1f}%**");st.error('⚠️ '+r.Alerta if r.Alerta=='REABASTECER' else r.Alerta)
with tabs[5]:
 c=st.selectbox('SKU',master.Codigo,key='s');r=master[master.Codigo.eq(c)].iloc[0];rows=[]
 for lv in [.95,.975,.99]:
  z=norm.ppf(lv);ss=z*r.Sigma_e*math.sqrt(r.Lead_Time_Semanas);rows.append([lv*100,z,ss,r.Demanda_LT+ss])
 se=pd.DataFrame(rows,columns=['Nivel_servicio_pct','Z','SS','ROP']);st.dataframe(se,hide_index=True,use_container_width=True);st.plotly_chart(px.line(se,x='Nivel_servicio_pct',y=['SS','ROP'],markers=True),use_container_width=True);st.caption('95 %, 97,5 % y 99 % son escenarios técnicos; la selección definitiva requiere validación empresarial.')
with tabs[6]:
 st.markdown(r'''**Cadena:** `histórico → modelo → pronóstico 12 semanas → σₑ → SS → ROP → EOQ → IP → alerta`

- $D=promedio(\hat y_{12})\times52$
- $e_t=y_t-\hat y_t$ y $\sigma_e=SD(e_t)$
- $SS=Z\sigma_e\sqrt{LT}$
- $ROP=\hat D_{LT}+SS$
- $H=C\times i$
- $EOQ=\sqrt{2DS/H}$
- $IP=OH+OO-BO$

Actualización propuesta: enero, abril, julio y octubre. Incorporar datos reales, reevaluar candidatos, ratificar/sustituir ganador y recalcular pronóstico, σₑ, SS, ROP y EOQ.

**Alcance:** no afirmar superioridad frente al sistema tradicional hasta completar backtesting/validación comparativa.''')
with tabs[7]:
 d={'Politica_Inventario':master,'Metricas_Modelos':METRICAS,'Pronosticos':forecast}
 if hist is not None:d['Historico']=hist
 st.download_button('Descargar resultados Excel',xlsx_bytes(d),'resultados_dashboard.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet');st.download_button('Descargar política CSV',master.to_csv(index=False).encode('utf-8-sig'),'politica_inventario.csv','text/csv')
st.caption('Prototipo académico · La decisión final de compra requiere validación empresarial.')
