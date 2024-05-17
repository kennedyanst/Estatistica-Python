import numpy as np
from scipy.stats import norm
from scipy import stats
import seaborn as sns
import math


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

n = len(dados)
media = np.mean(dados)
desvio_padrao = np.std(dados, ddof=1)


# Calculos do intervalo de confiança - Manual
alpha = 0.05 / 2

1 - alpha

z = norm.ppf(1 - alpha)

x_inferior = media - z * (desvio_padrao / math.sqrt(n))

x_superior = media + z * (desvio_padrao / math.sqrt(n))

(x_inferior, x_superior)

margem_error = abs(media - x_superior)

margem_error

# Calculos do intervalo de confiança - Scipy

stats.sem(dados)

intervalos = norm.interval(0.95, media, stats.sem(dados))

intervalos

margem_error = abs(media - intervalos[0])
margem_error


# Diferentes níveis de confiança

intervalos = norm.interval(0.99, media, stats.sem(dados))
intervalos

margem_error = abs(media - intervalos[0])
margem_error


# Distribuição T de Student

# Intervalo de confiança e classificação

dados = np.array([149., 160., 147., 189., 175., 168., 156., 160., 152.])

n = len(dados)
media = np.mean(dados)
desvio_padrao = np.std(dados)

(n, media, desvio_padrao)

from scipy.stats import t

intervalos = t.interval(0.95, n - 1, media, stats.sem(dados, ddof = 0))
intervalos

margem_error = media - intervalos[0]
margem_error


# Intervalo de confiança e classificação

# Accuracy

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold

dataset = pd.read_csv('Bases de dados/credit_data.csv')
dataset.dropna(inplace=True)
dataset.head()

X = dataset.iloc[:, 1:4].values
y = dataset.iloc[:, 4].values

resultados_naive_bayes_cv = []
resultados_naive_bayes_cv_300 = []
resultados_logistica_cv = []
resultados_logistica_cv_300 = []
resultados_forest_cv = []
resultados_forest_cv_300 = []

for i in range(30):
    kfold = KFold(n_splits = 10, shuffle = True, random_state = i)


    naive_bayes = GaussianNB()
    scores_naive = cross_val_score(naive_bayes, X, y, cv = kfold)
    resultados_naive_bayes_cv.append(scores_naive.mean())
    resultados_naive_bayes_cv_300.append(scores_naive)
    
    logistica = LogisticRegression()
    scores_logistic = cross_val_score(logistica, X, y, cv = kfold)
    resultados_logistica_cv.append(scores_logistic.mean())
    resultados_logistica_cv_300.append(scores_logistic)
   
    forest = RandomForestClassifier()
    scores_random = cross_val_score(forest, X, y, cv = kfold)
    resultados_forest_cv.append(scores_random.mean())
    resultados_forest_cv_300.append(scores_random)

len(resultados_naive_bayes_cv), len(resultados_naive_bayes_cv_300)
print(resultados_naive_bayes_cv_300)
np.asarray(resultados_naive_bayes_cv_300).shape

resultados_naive_bayes_cv = np.array(resultados_naive_bayes_cv)
resultados_naive_bayes_cv_300 = np.array(np.asarray(resultados_naive_bayes_cv_300).reshape(-1))
resultados_logistica_cv = np.array(resultados_logistica_cv)
resultados_logistica_cv_300 = np.array(np.asarray(resultados_logistica_cv_300).reshape(-1))
resultados_forest_cv = np.array(resultados_forest_cv)
resultados_forest_cv_300 = np.array(np.asarray(resultados_forest_cv_300).reshape(-1))

sns.histplot(resultados_naive_bayes_cv, kde = True)
sns.histplot(resultados_naive_bayes_cv_300, kde = True)
sns.histplot(resultados_logistica_cv, kde = True)
sns.histplot(resultados_logistica_cv_300, kde = True)
sns.histplot(resultados_forest_cv, kde = True)
sns.histplot(resultados_forest_cv_300, kde = True)

resultados_naive_bayes_cv.mean(), resultados_logistica_cv.mean(), resultados_forest_cv.mean()

stats.variation(resultados_naive_bayes_cv) * 100, stats.variation(resultados_logistica_cv) * 100, stats.variation(resultados_forest_cv) * 100


# Intervalo de confiança

from scipy.stats import t
from scipy.stats import norm

# Naive bayes

intervalos_naive_bayes_t = t.interval(0.95, len(resultados_naive_bayes_cv) - 1, resultados_naive_bayes_cv.mean(), stats.sem(resultados_naive_bayes_cv, ddof = 0))

intervalos_naive_bayes_t

# margem de erro
abs(resultados_naive_bayes_cv.mean() - intervalos_naive_bayes_t[1])

intervalos_naive_bayes_norm = norm.interval(0.95, resultados_naive_bayes_cv_300.mean(), 
                                            stats.sem(resultados_naive_bayes_cv_300))

intervalos_naive_bayes_norm

# margem de erro
abs(resultados_naive_bayes_cv_300.mean() - intervalos_naive_bayes_norm[1])

# Regressão logística

intervalos_logistica_t = t.interval(0.95, len(resultados_logistica_cv) - 1, resultados_logistica_cv.mean(), stats.sem(resultados_logistica_cv, ddof = 0))

intervalos_logistica_t

# margem de erro
abs(resultados_logistica_cv.mean() - intervalos_logistica_t[1])

intervalos_logistica_norm = norm.interval(0.95, resultados_logistica_cv_300.mean(),
                                            stats.sem(resultados_logistica_cv_300))

intervalos_logistica_norm

# margem de erro
abs(resultados_logistica_cv_300.mean() - intervalos_logistica_norm[1])


# Random Forest

intervalos_forest_t = t.interval(0.95, len(resultados_forest_cv) - 1, resultados_forest_cv.mean(), stats.sem(resultados_forest_cv, ddof = 0))

intervalos_forest_t

# margem de erro
abs(resultados_forest_cv.mean() - intervalos_forest_t[1])

intervalos_forest_norm = norm.interval(0.95, resultados_forest_cv_300.mean(),
                                            stats.sem(resultados_forest_cv_300))

intervalos_forest_norm

# margem de erro
abs(resultados_forest_cv_300.mean() - intervalos_forest_norm[1])


# Teste de hipótese - Z

import numpy as np
import math
from scipy.stats import norm

dados_originais = np.array([126. , 129.5, 133. , 133. , 136.5, 136.5, 140. , 140. , 140. ,
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
dados_originais

H0_media = np.mean(dados_originais)
H0_media

H0_desvio_padrao = np.std(dados_originais)
H0_desvio_padrao


dados_novos = dados_originais * 1.03
dados_novos

H1_media = np.mean(dados_novos)
H1_media

H1_desvio_padrao = np.std(dados_novos)
H1_desvio_padrao

H1_n = len(dados_originais)

alpha = 0.05


# Teste Z

z = (H1_media - H0_media) / (H1_desvio_padrao / math.sqrt(H1_n))
z

norm.cdf(3.398058252427187), norm.ppf(0.9996606701617486)

z = norm.cdf(z)

z

p = 1 - z

if p < alpha:
    print('Hipótese nula rejeitada.')
else:
    print('Hipótese alternativa rejeitada.')


# Statsmodels

from statsmodels.stats.weightstats import ztest

_, p = ztest(dados_originais, dados_novos,
             value= H1_media - H0_media, alternative = 'larger')

p