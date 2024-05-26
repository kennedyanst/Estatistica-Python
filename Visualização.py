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

import pandas as pd
from mpl_toolkits.basemap import Basemap

dataset = pd.read_csv('Bases de dados/house_prices.csv')

dataset['lat'].describe()
dataset['long'].describe()

lat1, lat2 = dataset['lat'].min(), dataset['lat'].max()
long1, long2 = dataset['long'].min(), dataset['long'].max()

import matplotlib.pyplot as plt

dataset = dataset.sort_values(by = 'price', ascending=True)


dataset_caros = dataset.iloc[0:-1000]
dataset_baratos = dataset.iloc[0:1000]

plt.figure(figsize=(10, 10))
m = Basemap(projection='cyl', resolution='h',
            llcrnrlat=lat1, urcrnrlat=lat2, 
            llcrnrlon=long1, urcrnrlon=long2
            )

m.drawcoastlines()
m.fillcontinents(color = 'palegoldenrod', lake_color='lightskyblue')
m.drawmapboundary(fill_color='lightskyblue')
m.scatter(dataset['long'], dataset['lat'], s= 5, c='green', alpha=0.1, zorder=2)
m.scatter(dataset_caros['long'], dataset_caros['lat'], s= 10, c='red', alpha=0.1, zorder=3)
m.scatter(dataset_baratos['long'], dataset_baratos['lat'], s= 10, c='blue', alpha=0.1, zorder=3)

m.scatter(dataset_caros['long'], dataset_caros['lat'], s= 10, c='red', alpha=0.1, zorder=3)
m.scatter(dataset_baratos['long'], dataset_baratos['lat'], s= 10, c='blue', alpha=0.1, zorder=3)