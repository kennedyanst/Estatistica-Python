import numpy as np
from scipy import stats
import seaborn as sns

# Permutação

import math

math.factorial(3)

math.factorial(36) / math.factorial(36 - 5)

math.pow(36, 5)

# Combinação

math.factorial(6) / (math.factorial(2) * math.factorial(6-2))

math.factorial(6+2-1) / (math.factorial(2) * math.factorial(6-1))

# Interseção, união e diferença

# Interseção
a = (0,1,2,3,4,5,6,7)
b = (0,2,4,6,8)

set(a) and set(b)

# União
set(a) or set(b)

# Diferença
set(a).difference(set(b))
set(b).difference(set(a))

# Problabilidade e distribuição normal

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

sns.histplot(dados, kde=True)

# Média e desvio padrão
media = np.mean(dados)
desvio_padrao = np.std(dados)
media, desvio_padrao

np.quantile(dados, 0.25), np.quantile(dados, 0.5), np.quantile(dados, 0.75), np.quantile(dados, 1)

# Calculando a probabilidade de selecionar uma pessoa em Q1
q1 = (np.quantile(dados, 0.25) - media) / desvio_padrao # Valor padronizado
q1

stats.norm.cdf(q1)


# Calculando a probabilidade de selecionar uma pessoa em Q3
q3 = (np.quantile(dados, 0.75) - media) / desvio_padrao
q3

1 - stats.norm.cdf(q3)
stats.norm.sf(q3)

# Calculando a probabilidade de selecionar uma pessoa entre o Q2 e Q3
(168 - media) / desvio_padrao

0.73891

stats.norm.cdf(q3)

(159.25 - media) / desvio_padrao

0.50

stats.norm.cdf(159.25, media, desvio_padrao)

stats.norm.cdf(q3) - stats.norm.cdf(159.25, media, desvio_padrao)

0.73891 - 0.5


# Calculando a probabilidade de selecionar uma pessoa entre o Q1 ou Q3

0.26109 + 0.26109

# Calculando a probabilidade de não selecionar uma pessoa entre o Q1 ou Q3
1 - 0.52218


# Exercicio 1

# Uma empresa faz um concurso para seleção de novos funcionários. A prova tinha 50 questões e o Pedro acertou 40 questões. Considerando uma distribuição normal com média 24 e desvio padrão 8, qual a probabilidade dele ser contratado?
x = 40
media = 24
desvio_padrao = 8

padronizado = (x - media) / desvio_padrao
padronizado 

stats.norm.cdf(padronizado)
stats.norm.cdf(x, media, desvio_padrao)
stats.norm.ppf(stats.norm.cdf(padronizado))


# Exercicio 2

# A vida útil de uma marca de pneus é representada por uma distrinuição normal com média de 38.000 Km e desvio padrão de 3.000 Km.
# 1. Qual a probabilidade de que um pneu escolhido aleatoriamente tenha vida útil de 35.000 Km?
1 - stats.norm.cdf((35000 - 38000) / 3000)
stats.norm.sf(35000, 38000, 3000)

# 2. Qual a probabilidade de que ele dure mais do que 44.000 Km?
1 - stats.norm.cdf((44000 - 38000) / 3000)
stats.norm.sf(44000, 38000, 3000)


# Probabilidade - Distribuição Binomial

# Exemplo das moedas - Jogando a moeda 10m vezes, qual a probabilidade de obter 5 "caras"?

n = 10
x = 5
p = 0.5

import math
(math.factorial(n) / (math.factorial(x) * math.factorial(n - x))) * math.pow(p, x) * math.pow(1 - p, n - x)

stats.binom.pmf(x, n, p)

# Exercicio 1

# 70% das pessoas que compraram o livro de python são mulheres. Se 10 leitores forem selecionados randomicamente, qual a probabilidade de selecionarmos 7 mulheres?
n = 10
x = 7
p = 0.7

stats.binom.pmf(x, n, p)

# Exercicio 2

# Em uma linha de produção de uma fábrica de parafusos, a probabilidade de obter um parafuso defeituoso é 0,05. Tendo uma amostra de 50 peças, qual a probabilidade de obter?
# 1. Um parafuso defeituoso
n = 50
x = 1
p = 0.05

stats.binom.pmf(x, n, p)

# 2. Nenhum parafuso defeituoso
n = 50
x = 0
p = 0.05

stats.binom.pmf(x, n, p)


# Probabilidade - Distribuição de Poisson

# O número médio de carros vendidos por dia é 10. Qual a probabilidade de vender 14 carros amanhã?

x = 14
media = 10

math.e

math.pow(math.e, - media) * (math.pow(media, x) / math.factorial(x))

stats.poisson.pmf(x, media)


# Exercicio 1
# Em uma linha de produção de uma fábrica de parafusos, a probabilidade é obter 0,05 defeitos por UNIDADE. Qual a probabilidade de uma unidade apresentar

# 1. Um defeito
x = 1
media = 0.05
stats.poisson.pmf(x, media)

# 2. Nenhum defeito
x = 0
media = 0.05
stats.poisson.pmf(x, media)

# Exercicio 2
# Um vendedor de uma loja vende em média 50 produtos por dia. Qual a probabilidade de vender somente 5 produtos no próximo dia?

x = 5
media = 50
stats.poisson.pmf(x, media)