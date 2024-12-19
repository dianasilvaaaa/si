from typing import Tuple

import numpy as np

from si.data.dataset import Dataset

# O codigo define duas dunções principais para dividir um conjunto de dados em conjunto de treino e teste
#train_test_split: Divide aleatoriamente o conjunto de dados em treino e teste de acordo com uma proporção especificada, sem considerar a distribuição de classes
#stratified_train_test_split: Faz a divisão de treino e teste, mas garantindo que a distribuição das classes no conjunto original seja preservada em ambos os subconjuntos (estratificação)

#ambas as funções recebem como entrada um objeto Dataset contendo as características (X) e os rotulos (Y) retornando dois novos datasets um para treino e outro para teste.



def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
#train_test_split: Simplesmente divide os dados aleatoriamente, sem considerar a distribuição das classes.
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # set random state
    np.random.seed(random_state) #Define a semente do gerador de números aleatórios para garantir reprodutibilidade
    # get dataset size
    n_samples = dataset.shape()[0] #Calcula o número total de amostras no conjunto de dados. 
    # get number of samples in the test set
    n_test = int(n_samples * test_size) #Determina quantas amostras devem estar no conjunto de teste.
    # get the dataset permutations
    permutations = np.random.permutation(n_samples) #Gera uma permutação aleatória dos índices das amostras.
    # get samples in the test set
    test_idxs = permutations[:n_test] #Seleciona os primeiros índices como os do conjunto de teste.
    # get samples in the training set
    train_idxs = permutations[n_test:]#Seleciona os índices restantes como os do conjunto de treino.

    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    #Cria um novo objeto Dataset contendo apenas as amostras e rótulos dos índices de treino

    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    #Cria um objeto similar para o conjunto de teste.

    return train, test #retorna os dois subconjuntos


def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int= 42) -> tuple[Dataset, Dataset]:
#Preserva a distribuição das classes nos subconjuntos de treino e teste, garantindo uma amostragem mais representativa.
    #stratified train-test split : divide os dados em treino e teste

    """

    Splits the dataset into training and testing sets while preserving the class distribution in each subset.

    Parameters
    ----------
    dataset: Dataset
        - Dataset object to split
    test_size: float
        - Size of the test set. By default, 20%
    random_state: int
        - Random seed for reproducibility
    
    Returns
    -------
    Tuple[Dataset, Dataset]
        - A tuple where the first element is the training dataset and the second element is the testing dataset

    Raises
    -------
    ValueError
        - If test_size is not a float between 0 and 1
    
"""


# verifica se test_size é valido

    if test_size <0 or test_size > 1: #Objetivo: Garante que o tamanho de test_size esteja entre 0 e 1.
      raise ValueError #Se o valor for inválido, lança um erro.


    np.random.seed(random_state) #configuração da semente aleatória, garantindo a reprodutibilidade dos dados.
    #ao usar o random_state as divisões são geradas sempre da mesma forma

# Extract class labels and their corresponding counts
    class_labels, class_counts = np.unique(dataset.y, return_counts= True) #Obtém os rótulos únicos das classes (class_labels) e a contagem de amostras de cada classe (class_counts).

# Initialize lists for storing indices of train and test samples  
    train_indices = [] #prepara listas vazias para armazenar os indices que irão compor os conjuntos de treino
    test_indices = [] #  e teste. 

#Efetuar uma amostragem estratificada para cada classe


    # A função entra em um loop para processar cada classe separadamente:   
    for label, count in zip(class_labels, class_counts):
        
        #Obtém todos os índices das amostras pertencentes à classe atual
        label_indices = np.where(dataset.y == label)[0]

        #garantir aleatoriedade
        np.random.shuffle(label_indices) #Baralha os índices para garantir aleatoriedade.

        # Calcular o número de amostras a atribuir ao conjunto de teste para a classe atual
        split_point = int(count * test_size)#Calcula quantas amostras dessa classe irão para o conjunto de teste.

        #Atribuir a primeira parte dos índices baralhados para testar e o resto para treinar
        test_indices.extend(label_indices[:split_point])#Adiciona os índices para teste.
        train_indices.extend(label_indices[split_point:]) #Adiciona os índices para treino.

    # Convert lists to numpy arrays
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    # Create new Dataset objects for training and testing:

    #Adiciona os índices para treino.
    train_dataset = Dataset(X=dataset.X[train_indices, :], y=dataset.y[train_indices], features=dataset.features, label=dataset.label)
    
    #Adiciona os índices para treino.
    test_dataset = Dataset(X=dataset.X[test_indices, :], y=dataset.y[test_indices], features=dataset.features, label=dataset.label)

    return train_dataset, test_dataset #retorna os dois subconjuntos
    