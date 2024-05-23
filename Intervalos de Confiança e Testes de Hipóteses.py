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


# Distribuição T Student

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


# Teste de hipótese T - Usa quando a base possui menos de 30 amostras

dados_originais = np.array([149., 160., 147., 189., 175., 168., 156., 160., 152.]) 

dados_originais.mean(), np.std(dados_originais)

dados_novos = dados_originais * 1.02
dados_novos

dados_novos.mean(), np.std(dados_novos)

from scipy.stats import ttest_rel


_, p = ttest_rel(dados_originais, dados_novos)

p

alpha = 0.01
if p <= alpha:
    print('Hipótese nula rejeitada.')
else:
    print('Hipótese alternativa rejeitada.')



# Qui Quadrado - Categorias

from scipy.stats import chi2_contingency

tabela = np.array([[30, 20], [22, 28]])

tabela.shape

_, p, _, _ = chi2_contingency(tabela)

p

alpha = 0.05
if p <= alpha:
    print('Hipótese nula rejeitada.')
else:
    print('Hipótese alternativa rejeitada.')


# Seleção de atributos com testes de hipóteses - univariate SelectFdr (FDR: Teste estatísticos univariados são aqueles que envolvem apenas uma variável dependente, por exemplo: teste t ou teste z para comparação de médias)


import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFdr, chi2
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('Bases de dados/ad.data', header = None)
dataset

dataset.shape

X = dataset.iloc[:, 0:1558].values
y = dataset.iloc[:, 1558].values
np.unique(y, return_counts = True)

X.shape, y.shape

naive1 = GaussianNB()
naive1.fit(X, y)
previsoes1 = naive1.predict(X)
accuracy_score(y, previsoes1)

selecao = SelectFdr(chi2, alpha = 0.01)
X_novo = selecao.fit_transform(X, y)

X.shape, X_novo.shape

selecao.pvalues_, len(selecao.pvalues_)

colunas = selecao.get_support()
colunas

indices = np.where(colunas == True)
indices

naive2 = GaussianNB()
naive2.fit(X_novo, y)
previsoes2 = naive2.predict(X_novo)
accuracy_score(y, previsoes2) # ANTES ERA 0.78 AGORA 0.97


# Teste de hipótese - ANOVA (Análise de Variação)

# Comparativo entre 3 ou mais grupos (amostras independentes)

# Distribuição normal

# Variação entre os grupos comparado a variação dentro dos grupos

# H0: As médias são iguais (Não há diferença estatística)

# H1: Pelo menos uma média é diferente (Existe diferença estatística)

# ANOVA

grupo_a = np.array([165,152,143,140,155])
grupo_b = np.array([130,169,164,143,154])
grupo_c = np.array([163,158,154,149,156])

from scipy.stats import f, f_oneway

f.ppf(1 - 0.05, dfn = 2, dfd = 12)

_, p = f_oneway(grupo_a, grupo_b, grupo_c)
p

alpha = 0.05
if p <= alpha:
    print('Hipótese nula rejeitada.') # Não existia diferença estatística entre os dados
else:
    print('Hipótese alternativa rejeitada.') # Existia diferença estatística entre os dados


# TESTE DE TUKEY

dados = {'valores': [165, 152, 143, 140, 155, 130, 169, 164, 143, 154, 163, 158, 154, 149, 156],
         'grupo': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C']}

dados = {'valores': [70, 90, 80, 50, 20, 130, 169, 164, 143, 154, 163, 158, 154, 149, 156],
         'grupo': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C']}


import pandas as pd
dados_pd = pd.DataFrame(dados)
dados_pd

from statsmodels.stats.multicomp import MultiComparison

compara_grupos = MultiComparison(dados_pd['valores'], dados_pd['grupo'])

teste_tukey = compara_grupos.tukeyhsd()

print(teste_tukey)

teste_tukey.plot_simultaneous()


# Seleção de atributos com ANOVA

from sklearn.feature_selection import f_classif

selecao = SelectFdr(f_classif, alpha = 0.01)
X_novo_2 = selecao.fit_transform(X, y)

X.shape, X_novo.shape, X_novo_2.shape

selecao.pvalues_

np.sum(selecao.pvalues_ < 0.01)

naive3 = GaussianNB()
naive3.fit(X_novo_2, y)
previsoes3 = naive3.predict(X_novo_2)
accuracy_score(y, previsoes3)

# Resultados dos algoritmos de machine learning

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold

dataset = pd.read_csv('Bases de dados/credit_data.csv')
dataset.dropna(inplace=True)
dataset.head()

X = dataset.iloc[:, 1:4].values
y = dataset.iloc[:, 4].values

min(X[0]), max(X[0])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

min(X[0]), max(X[0])

resultados_naive_cv = []
resultados_logistica_cv = []
resultados_forest_cv = []

for i in range(30):
    kfold = KFold(n_splits = 10, shuffle = True, random_state = i)

    naive = GaussianNB()
    scores_naive = cross_val_score(naive, X, y, cv = kfold)
    resultados_naive_cv.append(scores_naive.mean())
    
    logistica = LogisticRegression()
    scores_logistica = cross_val_score(logistica, X, y, cv = kfold)
    resultados_logistica_cv.append(scores_logistica.mean())
    
    forest = RandomForestClassifier()
    scores_forest = cross_val_score(forest, X, y, cv = kfold)
    resultados_forest_cv.append(scores_forest.mean())

resultados_naive_cv = np.array(resultados_naive_cv)
resultados_logistica_cv = np.array(resultados_logistica_cv)
resultados_forest_cv = np.array(resultados_forest_cv)

resultados_naive_cv.mean(), resultados_logistica_cv.mean(), resultados_forest_cv.mean()

# Qui Quadrado e ANOVA são testes parametricos de distribuição normal

# Testes para dados que não estão em uma distribuição normal (Shapiro-Wilk, D'Agostinho K^2, Anderson-Darling)

# SHAPIRO

alpha = 0.05
from scipy.stats import shapiro
shapiro(resultados_naive_cv), shapiro(resultados_logistica_cv), shapiro(resultados_forest_cv)

import seaborn as sns
sns.histplot(resultados_naive_cv, kde = True)
sns.histplot(resultados_logistica_cv, kde = True)
sns.histplot(resultados_forest_cv, kde = True)

# D'Agostinho K^2

from scipy.stats import normaltest
normaltest(resultados_naive_cv), normaltest(resultados_logistica_cv), normaltest(resultados_forest_cv)

# Anderson-Darling

from scipy.stats import anderson
anderson(resultados_naive_cv).statistic, anderson(resultados_logistica_cv).statistic, anderson(resultados_forest_cv).statistic


# Testes não paramétricos

# Teste de Wilcoxon Signed-Rank

alpha = 0.05

from scipy.stats import wilcoxon

_, p = wilcoxon(resultados_naive_cv, resultados_logistica_cv)
p

_, p = wilcoxon(resultados_naive_cv, resultados_forest_cv)
p

_, p = wilcoxon(resultados_logistica_cv, resultados_forest_cv)
p

# Teste de Friedman

from scipy.stats import friedmanchisquare

_, p = friedmanchisquare(resultados_naive_cv, resultados_logistica_cv, resultados_forest_cv)
p


# Exercicio - ANOVA E TUKEY

from scipy.stats import f_oneway

_, p = f_oneway(resultados_naive_cv, resultados_logistica_cv, resultados_forest_cv)
p

alpha = 0.05
if p <= alpha:
    print('Hipótese nula rejeitada. Dados são diferentes')
else:
    print('Hipótese alternativa rejeitada.')

resultados_algoritmos = {'accuracy': np.concatenate([resultados_naive_cv, resultados_logistica_cv, resultados_forest_cv]),
                         'algoritmo': ['naive', 'naive', 'naive', 'naive', 'naive', 'naive', 'naive', 'naive', 'naive', 'naive', 
                                       'naive', 'naive', 'naive', 'naive', 'naive', 'naive', 'naive', 'naive', 'naive', 'naive', 
                                       'naive', 'naive', 'naive', 'naive', 'naive', 'naive', 'naive', 'naive', 'naive', 'naive', 
                                       'logistica', 'logistica', 'logistica', 'logistica', 'logistica', 'logistica', 'logistica', 'logistica', 'logistica', 'logistica',
                                       'logistica', 'logistica', 'logistica', 'logistica', 'logistica', 'logistica', 'logistica','logistica', 'logistica', 'logistica',
                                       'logistica', 'logistica', 'logistica', 'logistica', 'logistica', 'logistica', 'logistica','logistica', 'logistica', 'logistica',
                                       'forest', 'forest', 'forest', 'forest', 'forest', 'forest', 'forest', 'forest', 'forest', 'forest',
                                       'forest', 'forest', 'forest', 'forest', 'forest', 'forest', 'forest', 'forest', 'forest', 'forest',
                                       'forest', 'forest', 'forest', 'forest', 'forest', 'forest', 'forest', 'forest', 'forest', 'forest']}

resultados_algoritmos

import pandas as pd
resultados_df = pd.DataFrame(resultados_algoritmos)
resultados_df

from statsmodels.stats.multicomp import MultiComparison

compara_grupos = MultiComparison(resultados_df['accuracy'], resultados_df['algoritmo'])

teste = compara_grupos.tukeyhsd()
print(teste)

teste.plot_simultaneous()


# Teste de Nemenyi
resultados_algoritmos = {'naive_bayes': resultados_naive_cv,
                         'logistica': resultados_logistica_cv,
                         'forest': resultados_forest_cv}

resultados_df = pd.DataFrame(resultados_algoritmos)
resultados_df

resultados_df.to_excel('resultados-excel.xlsx', index = False, sheet_name='resultados')


# Dados não normais

dataset = pd.read_csv('Bases de dados/trip_d1_d2.csv', sep=';')
dataset.head()

sns.histplot(dataset['D1'], kde = True)
sns.histplot(dataset['D2'], kde = True)

# Teste de Shapito
alpha = 0.05
from scipy.stats import shapiro
shapiro(dataset['D1']), shapiro(dataset['D2'])

# Teste de Friedman
from scipy.stats import friedmanchisquare
_, p = friedmanchisquare(dataset['D1'], dataset['D2']) # Precisa de 3 bases de dados

# Teste de Wilcoxon
from scipy.stats import wilcoxon
_, p = wilcoxon(dataset['D1'], dataset['D2'])
p

dataset['D1'].mean(), dataset['D2'].mean()

