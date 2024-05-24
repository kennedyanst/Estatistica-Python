import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats

# Variáveis contínuas

# Distribuição normal - Gaussian distribution

dados_nomrmal = stats.norm.rvs(size = 1000, random_state = 1)
dados_nomrmal

min(dados_nomrmal), max(dados_nomrmal)

sns.histplot(dados_nomrmal, bins = 50, kde = True)

dados_nomrmal.mean(), np.median(dados_nomrmal), stats.mode(dados_nomrmal), np.var(dados_nomrmal), np.std(dados_nomrmal)

np.sum(((dados_nomrmal >= 0.9810041339322116) & (dados_nomrmal <= 0.9810041339322116 + 1)))

np.sum(((dados_nomrmal <= 0.9810041339322116) & (dados_nomrmal >= 0.9810041339322116 - 1)))

(148 + 353) / 1000


# Distribuição normal com dados de alturas

dados = np.array([126. , 129.5, 133. , 133. , 136.5, 136.5, 140. , 140. , 140. ,
                  140. , 143.5, 143.5, 143.5, 143.5, 143.5, 143.5, 147. , 147. ,
                  147. , 147. , 147. , 147. , 147. , 150.5, 150.5, 150.5, 150.5,
                  150.5, 150.5, 150.5, 150.5, 154. , 154. , 154. , 154. , 154. ,
                  154. , 154. , 154. , 154. , 157.5, 157.5, 157.5, 157.5, 157.5,
                  157.5, 157.5, 157.5, 157.5, 157.5, 161. , 161. , 161. , 161. ,
                  161. , 161. , 161. , 161. , 161. , 161. , 164.5, 164.5, 164.5,
                  164.5, 164.5, 164.5, 164.5, 164.5, 164.5, 168. , 168. , 168. ,
                  168. , 168. , 168. , 168. , 168. , 171.5, 171.5, 171.5, 171.5,
                  171.5, 171.5, 171.5, 175. , 175. , 175. , 175. , 175. , 175. ,
                  178.5, 178.5, 178.5, 178.5, 182. , 182. , 185.5, 185.5, 189., 192.5])

min(dados), max(dados)

dados.mean(), np.median(dados), stats.mode(dados), np.var(dados), np.std(dados)

sns.histplot(dados, bins = 20, kde = True)


# Enviesamento

from scipy.stats import skewnorm

dados_normal = skewnorm.rvs(a = 0, size = 1000)
sns.histplot(dados_normal, bins = 20, kde = True)

dados_normal.mean(), np.median(dados_normal), stats.mode(dados_normal), np.var(dados_normal), np.std(dados_normal)

dados_normal_positivo = skewnorm.rvs(a = 10, size = 1000)
sns.histplot(dados_normal_positivo, bins = 20, kde = True)
dados_normal_positivo.mean(), np.median(dados_normal_positivo), stats.mode(dados_normal_positivo), np.var(dados_normal_positivo), np.std(dados_normal_positivo)

dados_normal_negativo = skewnorm.rvs(a = -10, size = 1000)
sns.histplot(dados_normal_negativo, bins = 20, kde = True)
dados_normal_negativo.mean(), np.median(dados_normal_negativo), stats.mode(dados_normal_negativo), np.var(dados_normal_negativo), np.std(dados_normal_negativo)

# Distribuição normal padrão

dados_normal_padronizada = np.random.standard_normal(1000)
sns.histplot(dados_normal_padronizada, bins = 10, kde = True)
dados_normal_padronizada.mean(), np.median(dados_normal_padronizada), stats.mode(dados_normal_padronizada), np.var(dados_normal_padronizada), np.std(dados_normal_padronizada)

dados

media = dados.mean()
desvio_padrao = dados.std()
dados_padronizados = (dados - media) / desvio_padrao
dados_padronizados

dados_padronizados.mean(), np.median(dados_padronizados), stats.mode(dados_padronizados), np.var(dados_padronizados), np.std(dados_padronizados)

sns.histplot(dados_padronizados, bins = 20, kde = True)

# Teorema Central do Limite - Quando a amostra é grande, a distribuição das médias amostrais se aproxima de uma distribuição normal

alturas = np.random.randint(126, 192, 500)
alturas.mean()

sns.histplot(alturas, bins = 20, kde = True)

medias = [np.random.randint(126, 192, 500).mean() for i in range(1000)]

type(medias), len(medias)

sns.histplot(medias, bins = 20, kde = True)


# Distribuição GAMA - Valores assimétricos para o lado direito
import seaborn as sns
from scipy.stats import gamma

dados_gama = gamma.rvs(a = 5, size = 1000)

sns.histplot(dados_gama, bins = 20, kde = True)

min(dados_gama), max(dados_gama)


#Distriuição Exponencial - Valores assimétricos para o lado direito / Um tipo de distribuição GAMA
from scipy.stats import expon

dados_exponencial = expon.rvs(size = 1000)

sns.histplot(dados_exponencial, bins = 20, kde = True)


# Distribuição Uniforme
import pandas as pd
import numpy as np
from scipy.stats import uniform

dados_uniforme = uniform.rvs(size = 1000)

sns.histplot(dados_uniforme, bins = 20, kde = True)

min(dados_uniforme), max(dados_uniforme)

dataset = pd.read_csv('Bases de dados/credit_data.csv')
dataset.dropna(inplace = True)
dataset.shape

dataset.head()

X = dataset.iloc[:, 1:4].values
y = dataset.iloc[:, 4].values

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

resultados_naive_bayes = []
for i in range(30):
    X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size = 0.2, stratify = y, random_state=i)

    naive_bayes = GaussianNB()
    naive_bayes.fit(X_treinamento, y_treinamento)
    resultados_naive_bayes.append(accuracy_score(y_teste, naive_bayes.predict(X_teste)))

print(resultados_naive_bayes)

sns.histplot(resultados_naive_bayes, bins = 10, kde = True)


# Distribuição de Bernoulli - Variável aleatória discreta - Sim ou Não

from scipy.stats import bernoulli

dados_bernoulli = bernoulli.rvs(size = 1000, p = 0.8)
dados_bernoulli

np.unique(dados_bernoulli, return_counts = True)

sns.histplot(dados_bernoulli, bins = 20, kde = True)


# Distribuição Binomial - Variável aleatória discreta - Sucesso ou Fracasso

from scipy.stats import binom

dados_binomial = binom.rvs(size = 1000, n = 10, p = 0.8)

np.unique(dados_binomial, return_counts = True)

sns.histplot(dados_binomial, bins = 20, kde = True)


# Distribuição de Poisson - Variável aleatória discreta - Número de eventos em um intervalo de tempo

from scipy.stats import poisson

dados_poisson = poisson.rvs(size = 1000, mu = 1)

np.unique(dados_poisson, return_counts = True)

sns.histplot(dados_poisson, bins = 20, kde = True)

# EXERCICIO 1

dataset = pd.read_csv('Bases de dados/census.csv')
dataset.head()

dataset.dtypes

sns.displot(dataset['age']);

sns.displot(dataset['final-weight']);

sns.displot(dataset['education-num']);

sns.displot(dataset['capital-gain']);

sns.displot(dataset['capital-loos']);

sns.displot(dataset['hour-per-week']);

sns.countplot(dataset['marital-status']);

sns.countplot(dataset['sex']);

sns.countplot(dataset['income']);

sns.countplot(dataset['native-country']);

sns.countplot(dataset['workclass']);

sns.countplot(dataset['education']);

sns.countplot(dataset['occupation']);


# Bernoulli Naive Bayes

from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('Bases de dados/census.csv')

dataset['sex'].unique()

X = dataset['sex'].values

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

X = label_encoder.fit_transform(X)

X

sns.histplot(X, bins = 20, kde = True)

X.shape

X = X.reshape(-1, 1)

Y = dataset['income'].values

bernoulli_naive_bayes = BernoulliNB()
bernoulli_naive_bayes.fit(X, Y)

previsoes = bernoulli_naive_bayes.predict(X)

accuracy_score(Y, previsoes)


# MultiNomial Naive Bayes

from sklearn.naive_bayes import MultinomialNB

dataset = pd.read_csv('Bases de dados/census.csv')

dataset.head()

from sklearn.preprocessing import LabelEncoder

label_encoder0 = LabelEncoder()
label_encoder1 = LabelEncoder()
label_encoder2 = LabelEncoder()
label_encoder3 = LabelEncoder()
label_encoder4 = LabelEncoder()
label_encoder5 = LabelEncoder()
label_encoder6 = LabelEncoder()

dataset['workclass'] = label_encoder0.fit_transform(dataset['workclass'])
dataset['education'] = label_encoder1.fit_transform(dataset['education'])
dataset['marital-status'] = label_encoder2.fit_transform(dataset['marital-status'])
dataset['occupation'] = label_encoder3.fit_transform(dataset['occupation'])
dataset['relationship'] = label_encoder4.fit_transform(dataset['relationship'])
dataset['race'] = label_encoder5.fit_transform(dataset['race'])
dataset['native-country'] = label_encoder6.fit_transform(dataset['native-country'])

dataset.head()

X = dataset.iloc[:, [1, 3, 5, 6, 7, 8, 13]].values
Y = dataset['income'].values

multinomial_naive_bayes = MultinomialNB()
multinomial_naive_bayes.fit(X, Y)

previsoes = multinomial_naive_bayes.predict(X)

accuracy_score(Y, previsoes)


# Aprendizagem baseada em Instancias - KNN

# Padronização (z-score) e knn

dataset = pd.read_csv('Bases de dados/credit_data.csv')
dataset.dropna(inplace = True)
dataset.head()

''' Sem padronização'''

X = dataset.iloc[:, 1:4].values
y = dataset['c#default'].values

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 1)

knn = KNeighborsClassifier()
knn.fit(X_treinamento, y_treinamento)
previsoes = knn.predict(X_teste)
accuracy_score(y_teste, previsoes)

''' Com padronização'''

from sklearn.preprocessing import StandardScaler

z_score_treinamento = StandardScaler()
z_score_teste = StandardScaler()

X_treinamento_p = z_score_treinamento.fit_transform(X_treinamento)
X_teste_p = z_score_teste.fit_transform(X_teste)

knn = KNeighborsClassifier()
knn.fit(X_treinamento_p, y_treinamento)
previsoes = knn.predict(X_teste_p)
accuracy_score(y_teste, previsoes)


# Dados enviesados e machine learning

dataset = pd.read_csv('Bases de dados/house_prices.csv')

dataset.head()

dataset.shape

sns.histplot(dataset['price'], bins = 20, kde = True)

sns.histplot(dataset['sqft_living'], bins = 20, kde = True)

''''Sem tratamento'''

from sklearn.linear_model import LinearRegression

X = dataset['sqft_living'].values
X = X.reshape(-1, 1)

y = dataset['price'].values

regressor = LinearRegression()
regressor.fit(X, y)

previsoes = regressor.predict(X)

from sklearn.metrics import mean_absolute_error, r2_score
mean_absolute_error(y, previsoes), r2_score(y, previsoes)

'''Com tratamento'''

X_novo = np.log(X) # Transformação logarítmica criação de uma distribuição normal
y_novo = np.log(y)

sns.histplot(X_novo, bins = 20, kde = True)
sns.histplot(y_novo, bins = 20, kde = True)

regressor = LinearRegression()
regressor.fit(X_novo, y_novo)
previsor = regressor.predict(X_novo)
mean_absolute_error(y_novo, previsor), r2_score(y_novo, previsor)


# Inicialização de pesos em redes neurais

import tensorflow as tf

tf.__version__

'''Inicializadores'''

import numpy as np
from tensorflow.keras import initializers
import seaborn as sns

# Random normal
normal = initializers.RandomNormal()
dados_normal = normal(shape = [1000])

np.mean(dados_normal), np.std(dados_normal)

sns.histplot(dados_normal);

# Random uniform
uniforme = initializers.RandomUniform()
dados_uniforme = uniforme(shape = [1000])

np.mean(dados_uniforme), np.std(dados_uniforme)

sns.histplot(dados_uniforme);

# Glorot normal - Xavier
glorot_normal = initializers.GlorotNormal()
dados_glorot_normal = glorot_normal(shape = [1000])

np.mean(dados_glorot_normal), np.std(dados_glorot_normal)

sns.histplot(dados_glorot_normal);

# Glorot uniform - Xavier
glorot_uniform = initializers.GlorotUniform()
dados_glorot_uniform = glorot_uniform(shape = [1000])

np.mean(dados_glorot_uniform), np.std(dados_glorot_uniform)

sns.histplot(dados_glorot_uniform);


# Testes de normalidades

from scipy.stats import skewnorm
dados_normais = skewnorm.rvs(size = 1000)
dados_nao_normais = skewnorm.rvs(a=10, size = 1000)

# Histograma

sns.histplot(dados_normais, bins = 20, kde = True)
sns.histplot(dados_nao_normais, bins = 20, kde = True)

# Quantile-Quantile plot

from statsmodels.graphics.gofplots import qqplot

qqplot(dados_normais, line = 's')
qqplot(dados_nao_normais, line = 's')

# Shapiro-Wilk

from scipy.stats import shapiro

_, p = shapiro(dados_normais)
p

alpha = 0.05
if p > alpha:
    print('Distribuição normal')
else:
    print('Distribuição não normal')

_, p = shapiro(dados_nao_normais)
p

alpha = 0.05
if p > alpha:
    print('Distribuição normal')
else:
    print('Distribuição não normal')