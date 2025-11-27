''''
DAY_OF_MONTH': Dia do mês.
'DAY_OF_WEEK': Dia da semana.
'OP_UNIQUE_CARRIER': Código único da companhia aérea.
'OP_CARRIER_AIRLINE_ID': ID único do operador aéreo.
'OP_CARRIER': Código IATA da companhia aérea.
'TAIL_NUM': Número da cauda (registro da aeronave).
'OP_CARRIER_FL_NUM': Número do voo.
'ORIGIN_AIRPORT_ID': ID do aeroporto de origem.
'ORIGIN_AIRPORT_SEQ_ID': ID sequencial do aeroporto de origem.
'ORIGIN': Aeroporto de origem.
'DEST_AIRPORT_ID': ID do aeroporto de destino.
'DEST_AIRPORT_SEQ_ID': ID sequencial do aeroporto de destino.
'DEST': Aeroporto de destino.
'DEP_TIME': Horário de partida do voo.
'DEP_DEL15': Indicador de atraso na partida (1 = atraso ≥ 15 minutos).
'DEP_TIME_BLK': Faixa de horário (hora) em que o voo partiu.
'ARR_TIME': Horário de chegada do voo.
'ARR_DEL15': Indicador de atraso na chegada (1 = atraso ≥ 15 minutos).
'CANCELLED': Indicador de cancelamento do voo.
'DIVERTED': Indicador se o voo foi desviado.
'DISTANCE': Distância entre os aeroportos.
'''

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%%
pd.set_option('display.max_columns', None)

df_jan = pd.read_csv('Jan_2019_ontime.csv')
df_fev = pd.read_csv('Feb_2019_ontime.csv')

df_jan.shape
df_fev.shape

print(df_fev.columns.tolist())
print(df_jan.columns.tolist())

df = pd.concat([df_jan,df_jan], axis=0).reset_index(drop=True)
#%%
'''Arrumar os nulos nas colunas TAIL_NUM , DEP_TIME, DEP_DEL15, ARR_TIME, ARR_DEL15, Unnamed'''
df.isnull().sum()

#%% Matrícula da aeronave (ex: N123AA)
df['TAIL_NUM'].value_counts()
df['TAIL_NUM'] = df['TAIL_NUM'].fillna('SEM_NUM')
df[df['TAIL_NUM'] == 'SEM_NUM']

#%% Hora real da decolagem (formato HHMM)
df['DEP_TIME'].value_counts()


#%%
#%% Indicador de atraso na partida:
# 1 = atraso > 15 minutos
# 0 = sem atraso significativo
df['DEP_DEL15'].value_counts()



#%% Hora real da chegada (formato HHMM).
df['ARR_TIME'].value_counts()

#%% Indicador de atraso na chegada (> 15 minutos):
# 1 = atrasado
# 0 = não atrasado
df['ARR_DEL15'].value_counts()

#%%
df.dropna(subset=['DEP_TIME','DEP_DEL15','ARR_TIME','ARR_DEL15'], inplace=True)
#%%
df.info()

#%%
df.drop(columns=['Unnamed: 21'], axis=1, inplace= True)
#%%
df.isnull().sum()

#%% 
#criar gafico da porcentagem de atrasos
df['TOTAL_ATRASOS'] = df['ARR_DEL15'] + df['DEP_DEL15']
df['TOTAL_ATRASOS'] = df['TOTAL_ATRASOS'].replace(2,1)
total_atrasos = df['TOTAL_ATRASOS'].value_counts(normalize=True) * 100
total_atrasos = pd.DataFrame(total_atrasos)
total_atrasos
#22,12% teve algum tipo de atrado, seja na chegada ou na partida
#%%
plt.figure(figsize=[14,10])
sns.barplot(data=total_atrasos, x=total_atrasos.index, y=total_atrasos['proportion'])
plt.show()

#%%
#PERCURSO DE DESTINOS COM MAIS ATRASOS
df['PERCURSO'] = df['ORIGIN'] + " - " + df['DEST'] 

rotas_atraso = pd.crosstab(df['PERCURSO'], df['TOTAL_ATRASOS']).sort_values(by=[1], ascending=False).head(10)
#AS 10 ROTAS COM MAIOR ATRASADO SENDO NA PARTIDA OU NA CHEGADA

plt.figure(figsize=[14,10])
sns.barplot(data=rotas_atraso, x=rotas_atraso.index, y=1.0)
plt.show()

#%%
#ATRASO DAS PARTIDAS ATRAVES DAS ORIGENS NA PARTIDA
atraso_partida = df[df['DEP_DEL15'] == 1.0]
atraso_partida = atraso_partida.groupby('ORIGIN').agg({'DEP_DEL15': 'sum'}).sort_values(by='DEP_DEL15', ascending=False).head(10)
atraso_partida
#%%
plt.figure(figsize=[14,10])
sns.barplot(data=atraso_partida, x=atraso_partida.index, y='DEP_DEL15')
plt.show()
#%%
#ATRASO DAS PARTIDAS ATRAVES DAS CHEGADAS
atraso_chegada = df[df['ARR_DEL15'] == 1.0]
atraso_chegada = atraso_partida.groupby('DEST').agg({'ARR_DEL15': 'sum'}).sort_values(by='ARR_DEL15', ascending=False).head(10)
atraso_chegada
#%%
plt.figure(figsize=[14,10])
sns.barplot(data=atraso_chegada, x=atraso_chegada.index, y='ARR_DEL15')
plt.show()

#%%
#COLUNA COM APENAS HORA DE PARTIDA, PARA VER QUAL HORARIO TEM MAIS ATRASOS
df['DEP_TIME'] = df['DEP_TIME'].astype(int)
df['HORA_PARTIDA'] = df['DEP_TIME'].astype(str).str.zfill(4).str[:2]
#ATRASO DE ACORDO COM AS HORAS

hora_partida_atrasos = pd.crosstab(df['HORA_PARTIDA'], df['TOTAL_ATRASOS']).sort_values(by=[1], ascending=False).head(10)
hora_partida_atrasos

#%%
#COLUNA COM APENAS HORA DE CHEGADA, SEM CONTAR AS PARTIDAS ATRASADAS

df['ARR_TIME'] = df['ARR_TIME'].astype(int)
df['HORA_CHEGADA'] = df['ARR_TIME'].astype(str).str.zfill(4).str[:2]
#ATRASO DE ACORDO COM AS HORAS
df_chegada_atrasada = df[df['DEP_DEL15']==0.0]
hora_chegada_atrasos = pd.crosstab(df_chegada_atrasada['HORA_CHEGADA'], df_chegada_atrasada['TOTAL_ATRASOS']).sort_values(by=[1], ascending=False)  
hora_chegada_atrasos
#%%

