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


# Comparando os métodos de amostragem

dataset['age'].mean()

df_amostra_aleatoria_simples['age'].mean()

df_amostra_sistematica['age'].mean()

df_amostra_agrupamento['age'].mean()

df_amostra_estratificada['age'].mean()

df_amostragem_reservatorio['age'].mean()

# É preciso fazer mais testes (30x no minimo) para ter uma média mais precisa, modificando o random_state.


# EXERCICIO 01

data = pd.read_csv('Bases de Dados/credit_data.csv')

data.shape

data.head()

amostra_simples = amostra_aleatoria_simples(data, 1000)

amostra_sistematica = amostragem_sistematica(data, 1000)

amostra_agrupamento = amostragem_agrupamento(data, 2)

def amostragem_estratificada(dataset, percentual, campo):
    split = StratifiedShuffleSplit(test_size = percentual, random_state = 1)
    for _, y in split.split(dataset, dataset[campo]):
        df_y = dataset.iloc[y]
    return df_y
amostra_estratificado = amostragem_estratificada(data, 0.5, 'c#default')

amostra_reservatorio = amostragem_reservatorio(data, 1000)

# Comparativo dos resultados

data['age'].mean(), data['income'].mean(), data['loan'].mean()

amostra_simples['age'].mean(), amostra_simples['income'].mean(), amostra_simples['loan'].mean()

amostra_sistematica['age'].mean(), amostra_sistematica['income'].mean(), amostra_sistematica['loan'].mean()

amostra_agrupamento['age'].mean(), amostra_agrupamento['income'].mean(), amostra_agrupamento['loan'].mean()

amostra_estratificado['age'].mean(), amostra_estratificado['income'].mean(), amostra_estratificado['loan'].mean()

amostra_reservatorio['age'].mean(), amostra_reservatorio['income'].mean(), amostra_reservatorio['loan'].mean()


# Classificação com Naive Bayes

dataset_2 = pd.read_csv('Bases de Dados/credit_data.csv')

dataset_2.shape

dataset_2.head()

dataset_2.dropna(inplace = True)

dataset_2.shape

import seaborn as sns
sns.countplot(dataset_2['c#default']);

dataset_2['c#default'].value_counts()

# Base está desbalanceada

X = dataset_2.iloc[:, 1:4].values

X.shape

y = dataset_2.iloc[:, 4].values

y.shape

# Base de treinamento e teste

from sklearn.model_selection import train_test_split

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size = 0.2, stratify = y)

X_treinamento.shape, X_teste.shape

y_treinamento.shape, y_teste.shape

np.unique(y, return_counts=True)

np.unique(y_treinamento, return_counts = True)

np.unique(y_teste, return_counts = True)

# Classificação com Naive Bayes

from sklearn.naive_bayes import GaussianNB

modelo = GaussianNB()
modelo.fit(X_treinamento, y_treinamento)

previsoes = modelo.predict(X_teste)
previsoes

y_teste

from sklearn.metrics import accuracy_score, classification_report

accuracy_score(y_teste, previsoes)

print(classification_report(y_teste, previsoes))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_teste, previsoes)

sns.heatmap(cm, annot = True);


# Subamostragem (undersampling) - Tomek Links

from imblearn.under_sampling import TomekLinks

tl = TomekLinks(sampling_strategy='majority')

X_under, y_under = tl.fit_resample(X, y)

X_under.shape, y_under.shape

np.unique(y, return_counts=True)

np.unique(y_under, return_counts=True)

X_treinamento_u, X_teste_u, y_treinamento_u, y_teste_u = train_test_split(X_under, y_under, test_size = 0.2, stratify = y_under)

X_treinamento_u.shape, X_teste_u.shape

modelo_u = GaussianNB()
modelo_u.fit(X_treinamento_u, y_treinamento_u)
previsoes_u = modelo_u.predict(X_teste_u)
accuracy_score(y_teste_u, previsoes_u)

cm_u = confusion_matrix(y_teste_u, previsoes_u)

sns.heatmap(cm_u, annot = True);