import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statistics
import scipy.stats as stats
import math


dados =  np.array([160,165,167,164,160,166,160,161,150,152,173,160,155,164,168,162,161,168,163,156,155,169,151,170,164,155,152,163,160,155,157,156,158,158,161,154,161,156,172,153])

# Média aritmética simples

dados.sum() / len(dados)

dados.mean()

statistics.mean(dados)


# Moda

statistics.mode(dados)

stats.mode(dados)

# Mediana

dados_impar = [150, 151, 152, 152, 153, 154, 155, 155, 155]

posicao = len(dados_impar) // 2

posicao = math.ceil(posicao)

dados_impar[posicao]

posicao = len(np.sort(dados)) // 2

dados[posicao - 1], dados[posicao]

np.median(dados_impar)

np.median(dados)

statistics.median(dados)

# Média aritmética ponderada

notas = np.array([9, 8, 7, 3])
pesos = np.array([1, 2, 3, 4])

media_ponderada = (notas * pesos).sum() / pesos.sum()

np.average(notas, weights=pesos)

# Média aritmética, moda e mediana com distribuição de frequência (Dados agrupados)

dados = {'inferior': [150, 154, 158, 162, 166, 170],
         'superior': [154, 158, 162, 166, 170, 174],
         'fi': [5, 9, 11, 7, 5, 3]}

dataset = pd.DataFrame(dados)
dataset

dataset['xi'] = (dataset['superior'] + dataset['inferior']) / 2

dataset

dataset['fi.xi'] = dataset['fi'] * dataset['xi']

dataset

dataset['Fi'] = 0


frequencia_acumulada = []
somatorio = 0
for linha in dataset.iterrows():
    somatorio += linha[1][2]
    frequencia_acumulada.append(somatorio)

frequencia_acumulada

dataset['Fi'] = frequencia_acumulada

dataset

# Média

media = dataset['fi.xi'].sum() / dataset['fi'].sum()

media

# Moda

dataset[dataset['fi'] == dataset['fi'].max()]

dataset[dataset['fi'] == dataset['fi'].max()]['xi'].values[0]

# Mediana

fi_2 = dataset['fi'].sum() / 2

fi_2


limite_inferior, frequencia_classes, id_frequencia_anterior = 0, 0, 0
for linha in dataset.iterrows():
    limite_inferior = linha[1][0]
    frequencia_classes = linha[1][2]
    id_frequencia_anterior = linha[0]
    if linha[1][5] >= fi_2:
        id_frequencia_anterior -= 1
        break

limite_inferior, frequencia_classes, id_frequencia_anterior

Fi_anterior = dataset.iloc[[id_frequencia_anterior]]['Fi'].values[0]

mediana = limite_inferior + ((fi_2 - Fi_anterior) * 4) / frequencia_classes


def get_estatisticas(dados):
    media = dados['fi.xi'].sum() / dados['fi'].sum()
    moda = dados[dados['fi'] == dados['fi'].max()]['xi'].values[0]

    fi_2 = dados['fi'].sum() / 2

    limite_inferior, frequencia_classes, id_frequencia_anterior = 0, 0, 0
    for linha in dados.iterrows():
        limite_inferior = linha[1][0]
        frequencia_classes = linha[1][2]
        id_frequencia_anterior = linha[0]
        if linha[1][5] >= fi_2:
            id_frequencia_anterior -= 1
            break
    Fi_anterior = dados.iloc[[id_frequencia_anterior]]['Fi'].values[0]
    mediana = limite_inferior + ((fi_2 - Fi_anterior) * 4) / frequencia_classes

    return media, moda, mediana

get_estatisticas(dataset)

# Média geométrica

from scipy.stats.mstats import gmean, hmean

gmean([2, 3, 4])

# Média harmônica

hmean([2, 3, 4])

# Média quadrática

np.sqrt(np.square([2, 3, 4]).sum() / len([2, 3, 4]))


dados_impar = [150,151,152,152,153,154,155,155,155]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import math

np.median(dados_impar)

posicao_mediana = len(dados_impar) // 2
posicao_mediana

esquerda = dados_impar[:posicao_mediana]
direita = dados_impar[posicao_mediana + 1:]

np.median(esquerda)

direita

np.median(direita)

np.quantile(dados_impar, 0.25)

np.quantile(dados_impar, 0.5)

np.quantile(dados_impar, 0.75)

np.quantile(dados_impar, 0.25), np.quantile(dados_impar, 0.5), np.quantile(dados_impar, 0.75)

stats.scoreatpercentile(dados_impar, 25), stats.scoreatpercentile(dados_impar, 50), stats.scoreatpercentile(dados_impar, 75)

dataset = pd.DataFrame(dados_impar)

dataset.quantile([0.25,0.5,0.75])

dados = {'inferior': [150, 154, 158, 162, 166, 170],
         'superior': [154, 158, 162, 166, 170, 174],
         'fi': [5, 9, 11, 7, 5, 3]}

dataset = pd.DataFrame(dados)
dataset

dataset['xi'] = (dataset['superior'] + dataset['inferior']) / 2

dataset

dataset['fi.xi'] = dataset['fi'] * dataset['xi']

dataset

dataset['Fi'] = 0


frequencia_acumulada = []
somatorio = 0
for linha in dataset.iterrows():
    somatorio += linha[1][2]
    frequencia_acumulada.append(somatorio)

frequencia_acumulada

dataset['Fi'] = frequencia_acumulada

dataset

# Média

media = dataset['fi.xi'].sum() / dataset['fi'].sum()

media

# Moda

dataset[dataset['fi'] == dataset['fi'].max()]

dataset[dataset['fi'] == dataset['fi'].max()]['xi'].values[0]

# Mediana

fi_2 = dataset['fi'].sum() / 2

fi_2


limite_inferior, frequencia_classes, id_frequencia_anterior = 0, 0, 0
for linha in dataset.iterrows():
    limite_inferior = linha[1][0]
    frequencia_classes = linha[1][2]
    id_frequencia_anterior = linha[0]
    if linha[1][5] >= fi_2:
        id_frequencia_anterior -= 1
        break

dataset

def get_quartil(dataframe, q1 = True):
  if q1 == True:
    fi_4 = dataset['fi'].sum() / 4
  else:
    fi_4 = (3 * dataset['fi'].sum()) / 4

  limite_inferior, frequencia_classe, id_frequencia_anterior = 0, 0, 0
  for linha in dataset.iterrows():
      limite_inferior = linha[1][0]
      frequencia_classes = linha[1][2]
      id_frequencia_anterior = linha[0]
      if linha[1][5] >= fi_4:
        id_frequencia_anterior -= 1
        break

  Fi_anterior = dataset.iloc[[id_frequencia_anterior]]['Fi'].values[0]
  q = limite_inferior + ((fi_4 - Fi_anterior) * 4) / frequencia_classes
  return q

get_quartil(dados), get_quartil(dados, False)

"""Percentis"""

dados =  np.array([160,165,167,164,160,166,160,161,150,152,173,160,155,
                   164,168,162,161,168,163,156,155,169,151,170,164,155,
                   152,163,160,155,157,156,158,158,161,154,161,156,172,153])

np.median(dados)

np.quantile(dados, 0.5)

np.percentile(dados, 50)

np.percentile(dados, 5), np.percentile(dados, 10), np.percentile(dados, 90)

stats.scoreatpercentile(dados, 5), stats.scoreatpercentile(dados, 10), stats.scoreatpercentile(dados, 90)

dataset = pd.DataFrame(dados)

dataset.quantile([0.05,0.10,0.90])

# Amplitude total e diferença interquartil

print(dados)

dados.max() - dados.min()

q1 = np.quantile(dados, 0.25)
q3 = np.quantile(dados, 0.75)
q1, q3

diferenca_interquartil = q3 - q1
diferenca_interquartil

inferior = q1 - 1.5 * diferenca_interquartil
superior = q3 + 1.5 * diferenca_interquartil
inferior, superior

# Variância e Desvio Padrão

dados_impar = np.array(dados_impar)

# Calculo manual

media = dados_impar.sum() / len(dados_impar)
media

desvio = abs(dados_impar - media)
desvio

desvio = desvio ** 2
desvio

soma_desvio = desvio.sum()
soma_desvio

v = soma_desvio / len(dados_impar)
v

dp = math.sqrt(v)
dp

cv = (dp / media) * 100
cv

def get_varianca_desvio_padrao_coeficiente(dataset):
  media = dataset.sum() / len(dataset)
  desvio = abs(dataset - media)
  desvio = desvio ** 2
  soma_desvio = desvio.sum()
  variancia = soma_desvio / len(dataset)
  dp = math.sqrt(variancia)
  cv = (dp / media) * 100
  return variancia, dp, cv

get_varianca_desvio_padrao_coeficiente(dados_impar)

np.var(dados_impar)

np.std(dados_impar)

np.var(dados)

np.std(dados)

import statistics

from scipy import ndimage

ndimage.variance(dados)

stats.tstd(dados, ddof = 0)

statistics.stdev(dados)

stats.variation(dados) * 100

# Desvio padrão com dados agrupados

dataset

dataset['xi_2'] = dataset['xi']  * dataset['xi']
dataset

dataset['fi_xi_2'] = dataset['fi'] * dataset['xi_2']
dataset

dataset.columns

colunas_ordenadas = ['inferior', 'superior', 'fi', 'xi', 'fi.xi', 'xi_2', 'fi_xi_2','Fi']

dataset = dataset[colunas_ordenadas]
dataset

dp = math.sqrt(dataset['fi_xi_2'].sum() / dataset['fi'].sum() - math.pow(dataset['fi.xi'].sum( ) / dataset['fi'].sum(), 2))
dp


# Avaliação de algoritmos de machine learning

import pandas as pd

dataset = pd.read_csv('Bases de dados/credit_data.csv')

dataset.head()

dataset.dropna(inplace = True)

X = dataset.iloc[:, 1:4].values

Y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import scipy.stats as stats

resultados_naive_bayes = []
resultados_logistica = []
resultados_forest = []

for i in range(30):
   X_treinamento, X_teste, Y_treinamento, Y_teste = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = i)

   naive_bayes = GaussianNB()
   naive_bayes.fit(X_treinamento, Y_treinamento)
   resultados_naive_bayes.append(accuracy_score(Y_teste, naive_bayes.predict(X_teste)))

   logistica = LogisticRegression()
   logistica.fit(X_treinamento, Y_treinamento)
   resultados_logistica.append(accuracy_score(Y_teste, logistica.predict(X_teste)))
   
   forest = RandomForestClassifier()
   forest.fit(X_treinamento, Y_treinamento)
   resultados_forest.append(accuracy_score(Y_teste, forest.predict(X_teste)))

print(resultados_naive_bayes)
print(resultados_logistica)
print(resultados_forest)


# Média
np.mean(resultados_naive_bayes), np.mean(resultados_logistica), np.mean(resultados_forest)

# Moda
stats.mode(resultados_naive_bayes), stats.mode(resultados_logistica), stats.mode(resultados_forest)

# Medianas
np.median(resultados_naive_bayes), np.median(resultados_logistica), np.median(resultados_forest)

# Desvio padrão
np.std(resultados_naive_bayes), np.std(resultados_logistica), np.std(resultados_forest)

# Variancia
np.var(resultados_naive_bayes), np.var(resultados_logistica), np.var(resultados_forest)

# Coeficiente de variação
cv_naive_bayes = (np.std(resultados_naive_bayes) / np.mean(resultados_naive_bayes)) * 100
cv_logistica = (np.std(resultados_logistica) / np.mean(resultados_logistica)) * 100
cv_forest = (np.std(resultados_forest) / np.mean(resultados_forest)) * 100

cv_naive_bayes, cv_logistica, cv_forest


# Validação cruzada

from sklearn.model_selection import cross_val_score, KFold

resultados_naive_bayes_cv = []
resultados_logistica_cv = []
resultados_forest_cv = []

for i in range(30):
  kfold = KFold(n_splits = 10, shuffle = True, random_state = i)

  naive_bayes = GaussianNB()
  scores = cross_val_score(naive_bayes, X, Y, cv = kfold)
  resultados_naive_bayes_cv.append(scores.mean())

  logistica = LogisticRegression()
  scores = cross_val_score(logistica, X, Y, cv = kfold)
  resultados_logistica_cv.append(scores.mean())

  forest = RandomForestClassifier()
  scores = cross_val_score(forest, X, Y, cv = kfold)
  resultados_forest_cv.append(scores.mean())

scores.mean()

print(resultados_naive_bayes_cv)
print(resultados_logistica_cv)
print(resultados_forest_cv)

stats.variation(resultados_naive_bayes_cv) * 100, stats.variation(resultados_logistica_cv) * 100, stats.variation(resultados_forest_cv) * 100


# Seleção de atributos utilizando variância

np.random.rand(50)

base_selecao = {'a': np.random.rand(20),
                'b': np.array([0.5] * 20),
                'classe': np.random.randint(0, 2, 20)}

dataset = pd.DataFrame(base_selecao)

dataset

dataset.describe()

np.var(dataset['a']), np.var(dataset['b'])

X = dataset.iloc[:, 0:2].values

from sklearn.feature_selection import VarianceThreshold

selecao = VarianceThreshold(threshold = 0.06)
X_novo = selecao.fit_transform(X)

X_novo, X_novo.shape

selecao.variances_

indices = np.where(selecao.variances_ > 0.06)
indices


# EXERCICIO DE ATRIBUTOS UTILIZANDO VARIÂNCIA

dataset = pd.read_csv('Bases de dados/credit_data.csv')

dataset.dropna(inplace = True)

dataset.head()

X = dataset.iloc[:, 1:4].values

Y = dataset.iloc[:, 4].values

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

selecao = VarianceThreshold(threshold=0.027)
X_novo = selecao.fit_transform(X)
X_novo

np.var(X[:,0]), np.var(X[:,1]), np.var(X[:,2])

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
naive_bayes_sem_selecao = GaussianNB()
naive_bayes_sem_selecao.fit(X, Y)
previsoes = naive_bayes_sem_selecao.predict(X)
accuracy_score(Y, previsoes)

naive_bayes_com_selecao = GaussianNB()
naive_bayes_com_selecao.fit(X_novo, Y)
previsoes = naive_bayes_com_selecao.predict(X_novo)
accuracy_score(Y, previsoes)

# Valores faltantes com média e moda

dataset = pd.read_csv('Bases de dados/credit_data.csv')

dataset.isnull().sum()

nulos = dataset[dataset.isnull().any(axis = 1)]

nulos

dataset['age'].mean(), dataset['age'].median()

dataset['age'] = dataset['age'].replace(to_replace = np.nan, value = dataset['age'].mean())


# Moda

dataset = pd.read_csv('Bases de dados/autos.csv', encoding='ISO-8859-1')

dataset.head()

dataset.shape

dataset.isnull().sum()

dataset['fuelType'].unique()

import statistics

statistics.mode(dataset['fuelType'])

dataset['fuelType'] = dataset['fuelType'].replace(to_replace = np.nan, value = statistics.mode(dataset['fuelType']))


