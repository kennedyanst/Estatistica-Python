# Amostragem 

import pandas as pd
import random
import numpy as np

dataset = pd.read_csv('Bases de Dados/census.csv')

dataset.shape

dataset.head()

dataset.tail()


# Amostragem aleatória simples

df_amostra_aleatoria_simples = dataset.sample(n = 1000, random_state = 1)

df_amostra_aleatoria_simples.shape

df_amostra_aleatoria_simples.head()


# Criando a função de amostragem aleatória simples
def amostra_aleatoria_simples(dataset, amostras):
    return dataset.sample(n = amostras, random_state = 1)

df_amostra_aleatoria_simples = amostra_aleatoria_simples(dataset, 1000)

df_amostra_aleatoria_simples.shape

df_amostra_aleatoria_simples.head()

# Amostragem sistemática

dataset.shape

len(dataset) // 1000

random.seed(1)
random.randint(0, 32561)

np.arange(68, len(dataset), step = 325)

# Função de amostragem sistemática

def amostragem_sistematica(dataset, amostras):
    intervalo = len(dataset) // amostras
    random.seed(1)
    inicio = random.randint(0, intervalo)
    indices = np.arange(inicio, len(dataset), step = intervalo)
    amostra_sistematica = dataset.iloc[indices]
    return amostra_sistematica

df_amostra_sistematica = amostragem_sistematica(dataset, 1000)

df_amostra_sistematica.shape


# Amostragem por grupos

len(dataset) / 10

grupos = []
id_grupo = 0
contagem = 0

for _ in dataset.iterrows():
    grupos.append(id_grupo)
    contagem += 1
    if contagem > 3256:
        contagem = 0
        id_grupo += 1

np.unique(grupos, return_counts = True)

np.shape(grupos), dataset.shape

dataset['grupo'] = grupos

dataset.tail()

random.randint(0, 9)

df_agrupamento = dataset[dataset['grupo'] == 5]

df_agrupamento.shape

df_agrupamento['grupo'].value_counts()

# Função de amostragem por grupos

def amostragem_agrupamento(dataset, numero_grupos):
    intervalo = len(dataset) / numero_grupos
    
    grupos = []
    id_grupo = 0
    contagem = 0

    for _ in dataset.iterrows():
        grupos.append(id_grupo)
        contagem += 1
        if contagem > intervalo:
            contagem = 0
            id_grupo += 1

    dataset['grupo'] = grupos
    random.seed(1)
    grupo_selecionado = random.randint(0, numero_grupos)
    return dataset[dataset['grupo'] == grupo_selecionado]

df_amostra_agrupamento = amostragem_agrupamento(dataset, 100)

df_amostra_agrupamento.shape


# Amostragem estratificada

from sklearn.model_selection import StratifiedShuffleSplit

dataset['income'].value_counts()

split = StratifiedShuffleSplit(test_size = 0.0030)
for x, y in split.split(dataset, dataset['income']):
    df_x = dataset.iloc[x]
    df_y = dataset.iloc[y]

df_x.shape, df_y.shape

# Função de amostragem estratificada

def amostragem_estratificada(dataset, percentual):
    split = StratifiedShuffleSplit(test_size = percentual, random_state = 1)
    for _, y in split.split(dataset, dataset['income']):
        df_y = dataset.iloc[y]
    return df_y

df_amostra_estratificada = amostragem_estratificada(dataset, 0.0030)


# Amostagem de reservatório

stream = []
for i in range(len(dataset)):
    stream.append(i)

def amostragem_reservatorio(dataset, amostras):
    stream = []
    for i in range(len(dataset)):
        stream.append(i)
    
    i = 0
    tamanho = len(dataset)
    reservatorio = [0] * amostras
    for i in range(amostras):
        reservatorio[i] = stream[i]
    
    while i < tamanho:
        j = random.randrange(i + 1)
        if j < amostras:
            reservatorio[j] = stream[i]
        i += 1
    
    return dataset.iloc[reservatorio]

df_amostragem_reservatorio = amostragem_reservatorio(dataset, 1000)

df_amostragem_reservatorio.shape

df_amostragem_reservatorio.head()