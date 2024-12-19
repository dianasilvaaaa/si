from si.data.dataset import Dataset 
import numpy as np
from si.model_selection.split import stratified_train_test_split




# Criando os dados para teste
X = np.array([
    [1, 2], [3, 4], [5, 6], [7, 8],  # Classe 0
    [9, 10], [11, 12], [13, 14],     # Classe 1
    [15, 16], [17, 18]               # Classe 2
])
y = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2])  # Rótulos (classes)

# Criando o Dataset
dataset = Dataset(X=X, y=y)

# Rodando a função para dividir em treino e teste
train, test = stratified_train_test_split(dataset, test_size=0.2, random_state=42)

# Imprimindo os resultados
print("Treino:")
print(train.X, train.y)

print("\nTeste:")
print(test.X, test.y)
