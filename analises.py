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


pd.set_option('display.max_columns', None)

df_jan = pd.read_csv('Jan_2019_ontime.csv')
df_fev = pd.read_csv('Feb_2019_ontime.csv')

df_jan.shape
df_fev.shape

print(df_fev.columns.tolist())
print(df_jan.columns.tolist())

df = pd.concat([df_jan,df_jan], axis=0).reset_index(drop=True)

pd.DataFrame({'unicos':df.nunique(),
              'missing': df.isna().sum(),
              'tipo':df.dtypes})

#Separando parte dos dados que estão com valores nulos na coluna de atraso na chegada
#Para posteiormente podermos prever seus resutados como mais um teste do modelo
separado_para_previsao = df[df['ARR_DEL15'].isnull()].reset_index(drop=True) 

#Apagando valores nulos de 'ARR_DEL15' que ja separamos e coincidentemente ja retira todos os nulos da base
df = df.dropna(subset='ARR_DEL15')
df = df.drop(columns=['Unnamed: 21'])

df[['DAY_OF_WEEK','DAY_OF_MONTH','DEP_DEL15','ARR_DEL15','CANCELLED','DIVERTED']] = df[['DAY_OF_WEEK','DAY_OF_MONTH','DEP_DEL15','ARR_DEL15','CANCELLED','DIVERTED']].astype('category')

df['ARR_TIME'] = pd.to_numeric(df['ARR_TIME'], errors='coerce')
df['DEP_TIME'] = pd.to_numeric(df['DEP_TIME'], errors='coerce')

df['HORA_PARTIDA'] = (df['DEP_TIME'] // 100).astype('Int64')
df['HORA_CHEGADA'] = (df['ARR_TIME'] // 100).astype('Int64')


#Analizando a distribuição dos valores de chegadas atrasadas normalizados e não normalizados
atrasos_proporcao = pd.DataFrame(round(df['ARR_DEL15'].value_counts(normalize=True)*100, 2))
atrasos_count= pd.DataFrame(df['ARR_DEL15'].value_counts())
atrasos = pd.merge(atrasos_count, atrasos_proporcao, left_index=True, right_index=True).reset_index()
print(atrasos)

plt.figure(figsize=[14,10])
sns.barplot(data=atrasos, x=atrasos.index, y=atrasos['count'])
plt.show()

df['PERCURSO'] = df['ORIGIN'] + " - " + df['DEST'] 
#AS 10 ROTAS COM MAIOR ATRASADO SENDO NA PARTIDA OU NA CHEGADA
rotas_atraso = pd.crosstab(df['PERCURSO'], df['ARR_DEL15']).sort_values(by=[1], ascending=False).head(10)

plt.figure(figsize=[14,10])
sns.barplot(data=rotas_atraso, x=rotas_atraso.index, y=1.0)
plt.show()

#ATRASO DAS PARTIDAS ATRAVES DAS ORIGENS NA PARTIDA
atraso_partida = df[df['DEP_DEL15'] == 1.0]
atraso_partida = atraso_partida.groupby('ORIGIN').agg({'DEP_DEL15': 'count'}).sort_values(by='DEP_DEL15', ascending=False).head(10)
print(atraso_partida)
plt.figure(figsize=[14,10])
sns.barplot(data=atraso_partida, x=atraso_partida.index, y='DEP_DEL15')
plt.show()

######################################REVER PQ O NUMERO TOTAL DA ERRADO NO FINAL
#HORARIO DE PARTIDA QUE MAIS TEM CHEGADAS ATRASADAS
df_chegada_atrasada = df[['ARR_DEL15', 'HORA_PARTIDA']]
hora_chegada_atrasos = df_chegada_atrasada[df_chegada_atrasada['ARR_DEL15']==1.0]
hora_chegada_atrasos = hora_chegada_atrasos.groupby('HORA_PARTIDA').agg({'ARR_DEL15':'count'}).sort_values(by='ARR_DEL15', ascending=False).head(10)

# dia da semana e dia do mes com mais voos atrasados
dia_atraso = df[['DAY_OF_WEEK', 'PERCURSO', 'ARR_DEL15']]
dia_atraso_count = dia_atraso.groupby('DAY_OF_WEEK').agg({'ARR_DEL15':'count'}).sort_values(by='ARR_DEL15', ascending=False).head(10)
dia_atraso_count
#dia do mes por percurso com mais voos atrasados
dia_atraso_percurso = dia_atraso.groupby(['PERCURSO','DAY_OF_WEEK']).agg({'ARR_DEL15':'count'}).sort_values(by='ARR_DEL15', ascending=False).head(10)
dia_atraso_percurso

# analise de dia por numero de atrasos
dia_atraso = df[['DAY_OF_MONTH', 'PERCURSO', 'ARR_DEL15']]
dia_atraso_count = dia_atraso.groupby('DAY_OF_MONTH').agg({'ARR_DEL15':'count'}).sort_values(by='ARR_DEL15', ascending=False).head(10)
dia_atraso_count
#analise de dia e percurso por numero de atrasos
dia_atraso_percurso = dia_atraso.groupby(['PERCURSO','DAY_OF_MONTH']).agg({'ARR_DEL15':'count'}).sort_values(by='ARR_DEL15', ascending=False).head(10)
dia_atraso_percurso


X = df[['DAY_OF_MONTH','DAY_OF_WEEK','DEP_TIME_BLK','HORA_PARTIDA','OP_UNIQUE_CARRIER','ORIGIN','DEST','DISTANCE']]
y = df['ARR_DEL15']

model = LogisticRegression(random_state=42)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)

num_cols = X.select_dtypes(include=['int', 'float']).columns

cat_cols = X.select_dtypes(include=['object', 'category']).columns


X_transformer = ColumnTransformer(transformers=[
    ("int", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=True), cat_cols)
])


#%%
model_reg = LogisticRegression(n_jobs=-1,verbose=1,random_state=42)

pipe = Pipeline(steps=[
    ("preprocessor", X_transformer),
    ("model", model_reg)
                
                ])

params = {
    "model__C": [0.001, 0.01, 0.1, 1.0, 10.0],
    "model__max_iter": [100, 200, 300],
    'model__class_weight': ['balanced'],
    'model__solver': ['lbfgs']
}

grid = GridSearchCV(pipe, param_grid=params, cv=3, scoring="roc_auc", verbose=2)
grid.fit(X_train,y_train)
print(grid.best_params_)
print(grid.best_estimator_)
print(grid.best_score_)

#%%
y_test_predict = grid.predict(X_test) #tESTANDO A ACURACIA
y_test_proba = grid.predict_proba(X_test)[:,1] #TESTANDO A CURVA ROC

roc_test = metrics.roc_curve(y_test, y_test_proba)
acc_test = metrics.accuracy_score(y_test, y_test_predict)
auc_test = metrics.roc_auc_score(y_test, y_test_proba)

roc = metrics.roc_curve(y_test_predict, y_test_proba)
cm = confusion_matrix(y_test, y_test_predict)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Curva ROC
ax[0].plot(roc_test[0], roc_test[1], label=f"Teste AUC = {auc_test:.3f}")
ax[0].plot([0, 1], [0, 1], 'k--')
ax[0].set_title("Curva ROC")
ax[0].set_xlabel("Falso Positivo (1 - Especificidade)")
ax[0].set_ylabel("Verdadeiro Positivo (Sensibilidade)")
ax[0].legend()
ax[0].grid(True)

# Matriz de Confusão
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Não Churn", "Churn"]).plot(
    cmap="Blues", ax=ax[1], values_format="d"
)
ax[1].set_title("Matriz de Confusão")

plt.tight_layout()
plt.show()

#%%
model = RandomForestClassifier(random_state=42, n_jobs=-1)

pipe_model = Pipeline(steps=[
    ("preprocesso", X_transformer),
    ("model", model)
])

params = {
    "model__min_samples_leaf": [10],
    "model__n_estimators": [100],
    "model__class_weight": ['balanced'],
    }

pipe_model.fit(X_train, y_train)


grid = GridSearchCV(pipe_model, param_grid=params, cv=3, scoring="roc_auc", verbose=2)
grid.fit(X_train, y_train)

print("\nMelhores parâmetros encontrados:")
print(grid.best_params_)
print(grid.best_score_)
print(grid.best_estimator_)

#%%
y_test_predict = grid.predict(X_test) #tESTANDO A ACURACIA
y_test_proba = grid.predict_proba(X_test)[:,1] #TESTANDO A CURVA ROC

roc_test = metrics.roc_curve(y_test, y_test_proba)
acc_test = metrics.accuracy_score(y_test, y_test_predict)
auc_test = metrics.roc_auc_score(y_test, y_test_proba)

roc = metrics.roc_curve(y_test_predict, y_test_proba)
cm = confusion_matrix(y_test, y_test_predict)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Curva ROC
ax[0].plot(roc_test[0], roc_test[1], label=f"Teste AUC = {auc_test:.3f}")
ax[0].plot([0, 1], [0, 1], 'k--')
ax[0].set_title("Curva ROC")
ax[0].set_xlabel("Falso Positivo (1 - Especificidade)")
ax[0].set_ylabel("Verdadeiro Positivo (Sensibilidade)")
ax[0].legend()
ax[0].grid(True)

# Matriz de Confusão
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Não Churn", "Churn"]).plot(
    cmap="Blues", ax=ax[1], values_format="d"
)
ax[1].set_title("Matriz de Confusão")

plt.tight_layout()
plt.show()