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
#%%
pd.set_option('display.max_columns', None)

df_jan = pd.read_csv('Jan_2019_ontime.csv')
df_fev = pd.read_csv('Feb_2019_ontime.csv')

df_jan.shape
df_fev.shape

print(df_fev.columns.tolist())
print(df_jan.columns.tolist())

df = pd.concat([df_jan,df_jan], axis=0).reset_index(drop=True)

'''Arrumar os nulos nas colunas TAIL_NUM , DEP_TIME, DEP_DEL15, ARR_TIME, ARR_DEL15, Unnamed'''
df.isnull().sum()

#%% Matrícula da aeronave (ex: N123AA)
df['TAIL_NUM'].value_counts()
df['TAIL_NUM'] = df['TAIL_NUM'].fillna('SEM_NUM')
df[df['TAIL_NUM'] == 'SEM_NUM']

#%% Hora real da decolagem (formato HHMM)
df['DEP_TIME'].value_counts()

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
atraso_chegada = atraso_chegada.groupby('DEST').agg({'ARR_DEL15': 'sum'}).sort_values(by='ARR_DEL15', ascending=False).head(10)
atraso_chegada
#%%
plt.figure(figsize=[14,10])
sns.barplot(data=atraso_chegada, x=atraso_chegada.index, y='ARR_DEL15')
plt.show()

#%%
'''REVER ISSO AQUI, EXISTE A FAIXA DE HORARIO DE SAIDA'''

#COLUNA COM APENAS HORA DE PARTIDA, PARA VER QUAL HORARIO TEM MAIS ATRASOS
df['DEP_TIME'] = df['DEP_TIME'].astype(int)
df['HORA_PARTIDA'] = df['DEP_TIME'].astype(str).str.zfill(4).str[:2]
#ATRASO DE ACORDO COM AS HORAS

hora_partida_atrasos = pd.crosstab(df['HORA_PARTIDA'], df['TOTAL_ATRASOS']).sort_values(by=1, ascending=False).head(10)
hora_partida_atrasos

#%%
#COLUNA COM APENAS HORA DE CHEGADA, SEM CONTAR AS PARTIDAS ATRASADAS

df['ARR_TIME'] = df['ARR_TIME'].astype(int)
df['HORA_CHEGADA'] = df['ARR_TIME'].astype(str).str.zfill(4).str[:2]
#ATRASO DE ACORDO COM AS HORAS
df_chegada_atrasada = df[df['DEP_DEL15']==0.0]
hora_chegada_atrasos = pd.crosstab(df_chegada_atrasada['HORA_CHEGADA'], df_chegada_atrasada['TOTAL_ATRASOS']).sort_values(by=[1], ascending=False)  
hora_chegada_atrasos
#%% dia da semana e dia do mes com mais voos atrasados
dia_atraso = df[['DAY_OF_WEEK', 'PERCURSO', 'TOTAL_ATRASOS']]
dia_atraso_count = dia_atraso.groupby('DAY_OF_WEEK').agg({'TOTAL_ATRASOS':'count'}).sort_values(by='TOTAL_ATRASOS', ascending=False).head(10)
dia_atraso_count
#%%dia do mes por percurso com mais voos atrasados
dia_atraso_percurso = dia_atraso.groupby(['PERCURSO','DAY_OF_WEEK']).agg({'TOTAL_ATRASOS':'count'}).sort_values(by='TOTAL_ATRASOS', ascending=False).head(10)
dia_atraso_percurso

#%% analise de dia por numero de atrasos
dia_atraso = df[['DAY_OF_MONTH', 'PERCURSO', 'TOTAL_ATRASOS']]
dia_atraso_count = dia_atraso.groupby('DAY_OF_MONTH').agg({'TOTAL_ATRASOS':'count'}).sort_values(by='TOTAL_ATRASOS', ascending=False).head(10)
dia_atraso_count
#%% analise de dia e percurso por numero de atrasos
dia_atraso_percurso = dia_atraso.groupby(['PERCURSO','DAY_OF_MONTH']).agg({'TOTAL_ATRASOS':'count'}).sort_values(by='TOTAL_ATRASOS', ascending=False).head(10)
dia_atraso_percurso

#%%
origin_delay_rate = df.groupby("ORIGIN")["TOTAL_ATRASOS"].mean()
df["ORIGIN_DELAY_RATE"] = df["ORIGIN"].map(origin_delay_rate)


dest_delay_rate = df.groupby("DEST")["TOTAL_ATRASOS"].mean()
df["DEST_DELAY_RATE"] = df["DEST"].map(dest_delay_rate)

hour_delay_rate = df.groupby("HORA_PARTIDA")["TOTAL_ATRASOS"].mean()
df["HOUR_DELAY_RATE"] = df["HORA_PARTIDA"].map(hour_delay_rate)

route_delay_rate = df.groupby("PERCURSO")["TOTAL_ATRASOS"].mean()
df["ROUTE_DELAY_RATE"] = df["PERCURSO"].map(route_delay_rate)
#%%


X = df[['DAY_OF_MONTH', 'DAY_OF_WEEK', 'ORIGIN', 'DEST', 'CANCELLED', 'DIVERTED', 'DISTANCE',  'HORA_PARTIDA',
        'ORIGIN_DELAY_RATE','DEST_DELAY_RATE', 'HOUR_DELAY_RATE','ROUTE_DELAY_RATE' ]].copy()
y = df[['TOTAL_ATRASOS']].copy()


X.info()
#%%
X[['CANCELLED', 'DIVERTED', 'DISTANCE', 'HORA_PARTIDA']] = X[['CANCELLED', 'DIVERTED', 'DISTANCE' ,'HORA_PARTIDA']].astype(int)
#%%

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)
#%%
num_cols = X.select_dtypes(include=['int', 'float']).columns

cat_cols = X.select_dtypes(include='object').columns

#%%
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

#%%
from xgboost import XGBClassifier

model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    tree_method='hist',      # mais rápido
    random_state=42
)

pipe_model = Pipeline(steps=[
    ("preprocesso", X_transformer),
    ("model", model)
])

params_xgb = {
    'model__n_estimators': [300, 600],
    'model__learning_rate': [0.05, 0.1],
    'model__max_depth': [3, 5, 7],
    'model__subsample': [0.8, 1],
    'model__colsample_bytree': [0.8],
    'model__gamma': [0, 1]
}

pipe_model.fit(X_train, y_train)

grid = GridSearchCV(pipe_model, param_grid=params_xgb, cv=3, scoring="roc_auc", verbose=2)
grid.fit(X_train, y_train)

print("\nMelhores parâmetros encontrados:")
print(grid.best_params_)

#{'model__colsample_bytree': 0.8, 'model__gamma': 0, 'model__learning_rate': 0.1, 'model__max_depth': 7, 'model__n_estimators': 600, 'model__subsample': 0.8}
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

#%%

print(acc_test)
print(roc_test)
print(auc_test)
#%%
from sklearn.model_selection import StratifiedKFold, cross_val_score,cross_val_predict
import numpy as np
lr_model_final = LogisticRegression(C=1.0,n_jobs=-1,verbose=1, random_state=154)
lr_model_final.fit(X_train, y_train)


#%%
cv = StratifiedKFold(n_splits=3, shuffle=True)
result = cross_val_score(lr_model_final,X_train,y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f'A média: {np.mean(result)}')
print(f'Limite Inferior: {np.mean(result)-2*np.std(result)}')
print(f'Limite Superior: {np.mean(result)+2*np.std(result)}')

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

#%%
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from sklearn.model_selection import RandomizedSearchCV
model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    tree_method='hist',      # mais rápido
    random_state=42
)

pipe_model = Pipeline(steps=[
    ("preprocesso", X_transformer),
    ("model", model)
])

param_dist = {

    'model__max_depth': [3, 5, 7, 9],
    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'model__colsample_bytree': [0.6, 0.8, 1.0],
    'model__gamma': [0, 1, 5],
    'model__min_child_weight': [1, 3, 5]
}




random_search = RandomizedSearchCV(
    pipe,
    param_distributions=param_dist,
    n_iter=30,                 # tenta 30 combinações – leve e eficiente
    scoring='f1',
    cv=3,
    verbose=2,
    n_jobs=-1
)

print("\nMelhores parâmetros encontrados:")
random_search.fit(X_train, y_train)


#%% MELHOR MODELO
best_model = random_search.best_estimator_
print("Melhores parâmetros:", random_search.best_params_)


#%% AVALIAÇÃO
y_pred = best_model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))