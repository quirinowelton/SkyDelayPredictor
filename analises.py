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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
pd.set_option('display.max_columns', None)
#%%
'''
#Dataframe summary
pd.DataFrame({'unicos':data.nunique(),
              'missing': data.isna().sum()/data.count(),
              'tipo':data.dtypes})
              
              
              
              DEPOIS TRANSFORMAR ESSAS COLUNAS EM CATEGORICAS 
              ['DAY_OF_WEEK','DAY_OF_MONTH','DEP_DEL15','ARR_DEL15','CANCELLED','DIVERTED']
              '''

pd.set_option('display.max_columns', None)

df_jan = pd.read_csv('Jan_2019_ontime.csv')
df_fev = pd.read_csv('Feb_2019_ontime.csv')

df_jan.shape
df_fev.shape

print(df_fev.columns.tolist())
print(df_jan.columns.tolist())

df = pd.concat([df_jan,df_jan], axis=0).reset_index(drop=True)
#%%
pd.DataFrame({'unicos':df.nunique(),
              'missing': df.isna().sum(),
              'tipo':df.dtypes})
#%%
#Separando parte dos dados que estão com valores nulos na coluna de atraso na chegada
#Para posteiormente podermos prever seus resutados como mais um teste do modelo
separado_para_previsao = df[df['ARR_DEL15'].isnull()].reset_index(drop=True) 
#%%
#Apagando valores nulos de 'ARR_DEL15' que ja separamos e coincidentemente ja retira todos os nulos da base
df = df.dropna(subset='ARR_DEL15')

df[['DAY_OF_WEEK','DAY_OF_MONTH','DEP_DEL15','ARR_DEL15','CANCELLED','DIVERTED']] = df[['DAY_OF_WEEK','DAY_OF_MONTH','DEP_DEL15','ARR_DEL15','CANCELLED','DIVERTED']].astype('category')
#%%
df['ARR_TIME'] = df['ARR_TIME'].astype(str).str.zfill(4)
df['DEP_TIME'] = df['DEP_TIME'].astype(str).str.zfill(4)

#%%
#Analizando a distribuição dos valores de chegadas atrasadas normalizados e não normalizados
atrasos_proporcao = pd.DataFrame(round(df['ARR_DEL15'].value_counts(normalize=True)*100, 2))
atrasos_count= pd.DataFrame(df['ARR_DEL15'].value_counts())
atrasos = pd.merge(atrasos_count, atrasos_proporcao, left_index=True, right_index=True).reset_index()
atrasos

plt.figure(figsize=[14,10])
sns.barplot(data=atrasos, x=atrasos.index, y=atrasos['count'])
plt.show()
#%%