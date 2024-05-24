import numpy as np
import pandas as pd
import seaborn as sns
import math


# Base de dados

tamanho = np.array([30,39,49,60])

preco = np.array([57000,69000,77000,90000])

dataset = pd.DataFrame({'Tamanho': tamanho, 'Preço': preco})
dataset

media_tamanho = dataset['Tamanho'].mean()

media_preco = dataset['Preço'].mean()

media_tamanho, media_preco

dp_tamanho = dataset['Tamanho'].std()
dp_preco = dataset['Preço'].std()
dp_tamanho, dp_preco

# Correlação

dataset['dif'] = (dataset['Tamanho'] - media_tamanho) * (dataset['Preço'] - media_preco)

dataset

soma_dif = dataset['dif'].sum()

soma_dif

covariancia = soma_dif / (len(dataset) - 1)

covariancia

coeficiente_correlacao = covariancia / (dp_tamanho * dp_preco)

coeficiente_correlacao

sns.heatmap(dataset.corr(), annot = True)

sns.scatterplot(x = 'Tamanho', y = 'Preço', data = dataset)

coeficiente_determinacao = math.pow(coeficiente_correlacao, 2)

coeficiente_determinacao


# Correlação - cálculo com numpy e pandas

np.cov(tamanho, preco)

dataset.cov()

np.corrcoef(tamanho, preco)

dataset.corr()


# EXERCÍCIO

dataset = pd.read_csv('Bases de dados/house_prices.csv')

dataset.drop(labels = ['id', 'date', 'sqft_living', 'sqft_lot'], axis = 1, inplace = True)
dataset.head()

dataset.corr()

sns.scatterplot(x=dataset['sqft_living15'], y=dataset['price'])

sns.scatterplot(x=dataset['grade'], y=dataset['price'])

sns.scatterplot(x=dataset['long'], y=dataset['price'])

sns.heatmap(dataset.corr(), annot=True)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(15,15))
ax = sns.heatmap(dataset.corr(), annot=True)


# Correlação com Yellowbrick

from yellowbrick.target import FeatureCorrelation

dataset.columns[1:]

grafico = FeatureCorrelation(labels = dataset.columns[1:])
grafico.fit(dataset.iloc[:,1:16].values, dataset.iloc[:, 0].values)
grafico.show();


# Regressão Linear Simples

dataset = pd.read_csv('Bases de dados/house_prices.csv')

dataset.drop(labels=['id', 'date'], axis = 1, inplace = True)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(15,15))
ax = sns.heatmap(dataset.corr(), annot=True)

X = dataset['sqft_living'].values

x = X.reshape(-1, 1)

y = dataset['price'].values

from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.2, random_state=1)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_treino, y_treino)

previsoes = regressor.predict(x_teste)

from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_teste, previsoes)

import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.plot(x, regressor.predict(x), color = 'red')

regressor.score(x_treino, y_treino)

regressor.score(x_teste, y_teste)


# Métricas de erros

previsoes = regressor.predict(x_teste)

previsoes, y_teste

from sklearn.metrics import mean_absolute_error, mean_squared_error

mean_absolute_error(y_teste, previsoes)

mean_squared_error(y_teste, previsoes)

math.sqrt(mean_squared_error(y_teste, previsoes)) # RMSE


# Regressão Linear Múltipla

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(15,15))
ax = sns.heatmap(dataset.corr(), annot=True)

x = dataset.iloc[:, [2,3,9,10]].values

y = dataset.iloc[:, 0].values


from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.2, random_state=1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_treino, y_treino)

regressor.score(x_treino, y_treino)

regressor.score(x_teste, y_teste)

previsoes = regressor.predict(x_teste)
mean_absolute_error(y_teste, previsoes)

import matplotlib.pyplot as plt
f, ax = plt.subplots(2, 2, figsize=(10, 10))  # Cria uma matriz 2x2 de gráficos
ax[0, 0].hist(x[0])
ax[0, 1].hist(x[1])
ax[1, 0].hist(x[2])
ax[1, 1].hist(x[3])
plt.show()


plt.hist(y);

y = np.log1p(y)

plt.hist(y);


from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.2, random_state=1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_treino, y_treino)

regressor.score(x_treino, y_treino)

regressor.score(x_teste, y_teste)

previsoes = regressor.predict(x_teste)
mean_absolute_error(y_teste, previsoes)


# EXERCICIOS
dataset = pd.read_csv('Bases de dados/house_prices.csv')
dataset.head()
dataset.shape

dataset.drop(labels = ['sqft_living15', 'sqft_lot15'], axis = 1, inplace = True)

x = dataset.iloc[:, 1:17].values
y = dataset.iloc[:, 0].values

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.2, random_state = 0)

regressor = LinearRegression()
regressor.fit(x_treino, y_treino)

regressor.score(x_treino, y_treino)

regressor.score(x_teste, y_teste)


# Seleção de atributos

from sklearn.feature_selection import SelectFdr, f_regression

selecao = SelectFdr(f_regression, alpha = 0.0)

X_novo = selecao.fit_transform(x, y)
x.shape, X_novo.shape

selecao.pvalues_

colunas = selecao.get_support()
colunas

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x_treino, x_teste, y_treino, y_teste = train_test_split(X_novo, y, test_size = 0.2, random_state = 0)

regressor = LinearRegression()
regressor.fit(x_treino, y_treino)

regressor.score(x_treino, y_treino)

regressor.score(x_teste, y_teste)