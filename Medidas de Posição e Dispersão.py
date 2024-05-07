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



