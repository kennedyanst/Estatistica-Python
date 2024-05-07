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

