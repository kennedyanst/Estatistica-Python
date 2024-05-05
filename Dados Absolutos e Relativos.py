import pandas as pd

dados = {'emprego': ['Administrador_banco_dados', 'Programador', 'Arquiteto_redes'],
         'nova_jersey': [97350, 82080, 112840],
         'florida': [77140, 71540, 62310]
         }

df = pd.DataFrame(dados)

df['nova_jersey'].sum()

df['florida'].sum()

df['%_nova_jersey'] = (df['nova_jersey'] / df['nova_jersey'].sum()) * 100

df['%_florida'] = (df['florida'] / df['florida'].sum()) * 100

df


# Exercicio 1

dataset = pd.read_csv('Bases de dados/census.csv')

dataset2 = dataset[['income', 'education']]

dataset3 = dataset2.groupby(['education', 'income'])['education'].count()

dataset3

dataset3.index