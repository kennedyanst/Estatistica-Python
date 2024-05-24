# Gráfico de Dispersão

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

dataset = pd.read_csv('Bases de dados/census.csv')

sns.relplot(x='age', y = 'final-weight', data = dataset, hue='income', style='sex', size='education-num')

# Gráfico de Barras e setor

sns.barplot(x= 'sex', y='final-weight', data = dataset, hue='income')

dados_agrupados = dataset.groupby('income')['education-num'].mean()
dados_agrupados


dados_agrupados.plot.bar()

dados_agrupados.plot.pie()


# Gráfico de Linhas

vendas = {'mes': np.array([1,2,3,4,5,6,7,8,9,10,11,12]),
          'vendas': np.array([100, 200, 200, 110, 500, 600, 300, 50, 220, 1000, 1100, 1200])}

vendas = pd.DataFrame(vendas)

sns.lineplot(x='mes', y='vendas', data = vendas)


# Boxplot

sns.boxplot(dataset['age'])

sns.boxplot(dataset['education-num'])

dataset2 = dataset.iloc[:, [0,4,12]]

sns.boxplot(data=dataset2)


# Gráficos com atributos categóricos

sns.catplot(x = 'income', y= 'hour-per-week', data= dataset, hue='sex')  


sns.catplot(x = 'income', y= 'hour-per-week', data= dataset.query('age < 30'), hue='sex')  


#Subplots

g = sns.FacetGrid(dataset, col='sex', hue='income')
g.map(sns.scatterplot, 'age', 'final-weight')

g = sns.FacetGrid(dataset, col= 'workclass', hue='income')
g.map(sns.scatterplot, 'age', 'final-weight')

g = sns.FacetGrid(dataset, col= 'sex', hue='income')
g.map(sns.histplot, 'age')

g = sns.PairGrid(dataset2)
g.map(sns.scatterplot)

g = sns.PairGrid(dataset2)
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)