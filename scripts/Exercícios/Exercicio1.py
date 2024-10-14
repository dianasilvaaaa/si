"""
## Exercise 1: NumPy array Indexing/Slicing

1.1) Neste exercício, vamos utilizar o conjunto de dados da íris. Carregue
o ficheiro “iris.csv” utilizando o método apropriado para este tipo de ficheiro
(utilize as novas funções do pacote).

"""
import os
from si.io.csv_file import read_csv

# Usando o caminho correto no Windows
file_path = r"C:\Users\diana\OneDrive\Ambiente de Trabalho\Sist. Inteligentes\23 set\session2-notebooks\iris.csv"
# Carregando o dataset iris
dataset = read_csv(file_path, sep=',', features=True, label=True)

# Verificar se o dataset foi carregado corretamente
print(dataset.X)
print(dataset.y)



"""
1.2) Selecione a penúltima variável independente. Qual é a dimensão da matriz resultante?
"""
# Selecionando a penúltima variável (penúltima coluna)
# dataset.X : features 
penultimate_feature = dataset.X[:, -2]

# Dimensão do array resultante
print("Dimensão da penúltima variável independente:", penultimate_feature.shape)


"""
1.3) Selecione as últimas 10 amostras do conjunto de dados da íris. 
Qual é a média das últimas 10 amostras para cada variável independente/caraterística?

"""

# Selecionando as últimas 10 amostras
last_10_samples = dataset.X[-10:, :]

# Calculando a média para cada variável independente
mean_last_10 = last_10_samples.mean(axis=0)

print("Média das últimas 10 amostras por variável independente:", mean_last_10)

"""
1.4) Selecione todas as amostras do conjunto de dados com valores menores ou iguais a 6 para todas 
as variáveis independentes/caraterísticas. Quantas amostras obtém?
"""

import numpy as np

# Selecionar amostras onde todas as variáveis são <= 6
samples_less_equal_6 = dataset.X[np.all(dataset.X <= 6, axis=1)]

# Quantidade de amostras selecionadas
print("Número de amostras com todas as variáveis <= 6:", samples_less_equal_6.shape[0])



"""
1.5) Selecione todas as amostras com uma classe/rótulo diferente de diferente de “Iris-setosa”.
 Quantas amostras obtém?

"""

# Selecionar amostras com classe diferente de 'Iris-setosa'
samples_not_setosa = dataset.X[dataset.y != 'Iris-setosa']

# Quantidade de amostras selecionadas
print("Número de amostras com classe diferente de 'Iris-setosa':", samples_not_setosa.shape[0])
