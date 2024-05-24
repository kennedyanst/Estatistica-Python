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

