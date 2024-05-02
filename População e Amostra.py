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