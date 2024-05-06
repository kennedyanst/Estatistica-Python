# Distribuyição de Frequência

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dados =  np.array([160,165,167,164,160,166,160,161,150,152,173,160,155,164,168,162,161,168,163,156,155,169,151,170,164,155,152,163,160,155,157,156,158,158,161,154,161,156,172,153])

# Ordenação
dados = np.sort(dados)

minimo = dados.min()

maximo = dados.max()

np.unique(dados, return_counts=True)

plt.bar(dados, dados)

# Número de classes i=1+3,3*log(n)

n = len(dados)

i = 1 + 3.3 * np.log10(n)

i = round(i)

# Amplitude = h = (maximo - minimo) / i ---- AA = Xmax - Xmin

AA = maximo - minimo

h = AA / i

import math

h = math.ceil(h)

# Construção da distribuição de frequência

intervalos = np.arange(minimo, maximo + 2, step = h)

intervalo1, intervalo2, intervalo3, intervalo4, intervalo5, intervalo6 = 0,0,0,0,0,0

for i in range(n):
    if dados[i] >= intervalos[0] and dados[i] < intervalos[1]:
        intervalo1 += 1
    elif dados[i] >= intervalos[1] and dados[i] < intervalos[2]:
        intervalo2 += 1
    elif dados[i] >= intervalos[2] and dados[i] < intervalos[3]:
        intervalo3 += 1
    elif dados[i] >= intervalos[3] and dados[i] < intervalos[4]:
        intervalo4 += 1
    elif dados[i] >= intervalos[4] and dados[i] < intervalos[5]:
        intervalo5 += 1
    elif dados[i] >= intervalos[5] and dados[i] < intervalos[6]:
        intervalo6 += 1

lista_intervalos = []

lista_intervalos.append(intervalo1)
lista_intervalos.append(intervalo2)
lista_intervalos.append(intervalo3)
lista_intervalos.append(intervalo4)
lista_intervalos.append(intervalo5)
lista_intervalos.append(intervalo6)

lista_intervalos

lista_classes = []
for i in range(len(intervalos)-1):
    lista_classes.append(str(intervalos[i]) + ' - ' + str(intervalos[i+1]))

lista_classes

plt.bar(lista_classes, lista_intervalos)
plt.title('Distribuição de Frequência')
plt.xlabel('intervalos')
plt.ylabel('Frequência')

plt.show()

# Distribuição de Frequência com Matplotlib

frequencia, classes = np.histogram(dados)

frequencia

plt.hist(dados, bins = classes);

frequencia, classes = np.histogram(dados, bins = 5)

plt.hist(dados, bins = classes);

frequencia, classes = np.histogram(dados, bins = 'sturges')

plt.hist(dados, bins = classes);


# Distribuição de Frequência com Pandas

dataset = pd.DataFrame({'dados': dados})

dataset.plot.hist();

sns.displot(dados, kde = True);



