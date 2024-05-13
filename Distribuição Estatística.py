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

