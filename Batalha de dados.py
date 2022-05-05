#!/usr/bin/env python
# coding: utf-8

# # Import Functions and Data Loading

# In[2]:


pip install umap-learn


# In[1]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np


# In[348]:


df_agro = pd.read_csv('./shared/bases_budokai_ufpr/agroclimatology_budokai.csv')
df_agro


# In[349]:


df_prod = pd.read_csv('./shared/bases_budokai_ufpr/produtividade_soja_modelagem.csv')
df_prod


# # Data Preparation

# In[2]:


# Seleciona apenas as cidades que serão avaliadas
submit_cities = [4102000,4104303,4104428,4104808,4104907,4109401,4113205,4113700,4113734,4114005,4114401,4117701,4117909,4119608,4119905,4127007,4127403,4127502,4127700,4128005]


# In[350]:


# Calcula a média movel da produção dos últimos 6 anos para cada cidade

df_prod_pct = df_prod.copy()
df_prod_pct.loc[:, '2004':] = df_prod_pct.loc[:, '2004':] / df_prod_pct.loc[:, '2004':].rolling(window=6,axis=1).mean()
df_prod_pct


# In[351]:


#transforma as colunas com os anos de cada cidade em linhas
df_model = df_prod_pct[df_prod.columns[1:]].melt(id_vars=['codigo_ibge', 'name'], var_name='Ano',value_name='Prod').copy()
df_model


# In[352]:


#Seleção dos anos de 2009 a 2017
df_model = df_model.dropna()
df_model


# In[353]:


#Funcao que seleciona os meses entre a semeadura e colheita
def data_to_safra_and_mes(data:int) -> [int, int, int]:
    data = str(data)
    mes = int(data[4:6])
    if mes >= 4:
        safra = int(data[:4]) + 1
        mes = mes - 4
        dia = int(data[6:])
        return [safra, mes, dia]
    else:
        safra = int(data[:4])
        mes = mes + 8
        dia = int(data[6:])
        return [safra, mes, dia]


# In[354]:


df_agro[['Ano', 'Mes', 'Dia']] = df_agro.data.apply(lambda x: pd.Series(data_to_safra_and_mes(x)))
df_agro


# In[355]:


df_agro_meses = df_agro.loc[df_agro.Mes > 4].copy()
df_agro_meses


# In[356]:


#Seleção das features
original_vars = df_agro_meses.loc[:,'T2M_RANGE':'ALLSKY_SFC_UV_INDEX'].columns.tolist()


# In[357]:


#Utilização de medidas estatísticas para criação de novas features
df_agro_meses_grouped = df_agro_meses[original_vars + ['codigo_ibge', 'Ano']].groupby(
    ['codigo_ibge', 'Ano'], 
    as_index=False).agg(['mean', 'max', 'min', 'std', 'median', 'sem', 'var'])
df_agro_meses_grouped


# In[358]:


df_agro_meses_grouped.columns = ['_'.join(col) for col in df_agro_meses_grouped.columns]
df_agro_meses_grouped = df_agro_meses_grouped.reset_index()
df_agro_meses_grouped


# In[359]:


df_model.Ano = df_model.Ano.astype(int).copy()
df_model


# In[360]:


df_model_x_y = pd.merge(df_model, df_agro_meses_grouped, on=['codigo_ibge', 'Ano'])
df_model_x_y


# In[361]:


X, Y = df_model_x_y[df_model_x_y.columns[4:]], df_model_x_y[df_model_x_y.columns[3]]


# In[362]:


X.columns


# In[363]:


Y


# # Dimensionality reduction

# Utiliza-se da técnica de redução de dimensionalidade a fim de transformar os dados de um espaço de alta dimensão em um espaço de baixa dimensão, de modo que a representação de baixa dimensão retenha algumas propriedades significativas dos dados originais.

# In[396]:


reducer = umap.UMAP(random_state=42)


# In[365]:


scaled_X = StandardScaler().fit_transform(X)


# In[401]:


embedding = reducer.fit_transform(scaled_X)
embedding.shape


# # Modeling

# In[409]:


# Verifica os melhores parâmetros a serem utilizados no modelo escolhido
gb = RandomForestRegressor()
parameters = {
    'random_state': [0],
    'n_estimators': range(10,100,10),
    'max_depth': range(5, 15,1),
    'criterion': ['mae'],
    'max_samples': np.linspace(0.1, 1, 9),
    'max_features': ['auto', 'sqrt', 'log2']
}
clf = GridSearchCV(gb, parameters, cv=5, n_jobs=-1, scoring='neg_mean_absolute_error')
_ = clf.fit(embedding, Y)
clf.best_score_


# In[412]:


clf.best_estimator_


# ## Test data transformation

# In[414]:


#Transformação dos dados de teste igualando aos dados de treino

df_to_predict = df_agro_meses_grouped.loc[(df_agro_meses_grouped.Ano > 2017) &
                                          (df_agro_meses_grouped.Ano < 2021) &
                                          (df_agro_meses_grouped.codigo_ibge.isin(submit_cities))].copy()
df_to_predict


# In[415]:


scaled_df_to_predict = StandardScaler().fit_transform(df_to_predict.loc[:, 'T2M_RANGE_mean':])
scaled_df_to_predict


# In[416]:


embedding_to_predict = reducer.transform(scaled_df_to_predict)
embedding_to_predict.shape


# In[417]:


#Predição da média móvel dos demais anos utilzando os melhores parâmetros segundo o modelo

df_to_predict['predicted'] = clf.best_estimator_.predict(embedding_to_predict)


# In[418]:


df_submit = df_to_predict[['codigo_ibge', 'Ano', 'predicted']].pivot_table(index='codigo_ibge', columns='Ano')
df_submit.columns = [i[1] for i in df_submit.columns]
df_submit


# In[419]:


#Transformação da média móvel para a real previsão de produção e soja

for ano in [2018,2019,2020 ]:
    base_value = df_prod.loc[df_prod.codigo_ibge.isin(df_submit.index),
                             str(ano-6):str(ano-1)].rolling(window=6, axis=1).mean()[str(ano-1)]
    df_prod.loc[df_prod.codigo_ibge.isin(df_submit.index), str(ano)] = base_value.values * df_submit[ano].values
df_prod.loc[df_prod.codigo_ibge.isin(df_submit.index)]


# In[420]:


df_prod.loc[df_prod.codigo_ibge.isin(df_submit.index), ['codigo_ibge', '2018', '2019', '2020']]


# In[ ]:




