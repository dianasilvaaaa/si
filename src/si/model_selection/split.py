from typing import Tuple

import numpy as np

from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
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
    np.random.seed(random_state)
    # get dataset size
    n_samples = dataset.shape()[0]
    # get number of samples in the test set
    n_test = int(n_samples * test_size)
    # get the dataset permutations
    permutations = np.random.permutation(n_samples)
    # get samples in the test set
    test_idxs = permutations[:n_test]
    # get samples in the training set
    train_idxs = permutations[n_test:]
    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test


def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int= 42) -> tuple[Dataset, Dataset]:

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

    if test_size <0 or test_size > 1:
      raise ValueError


    np.random.seed(random_state)

# Extract class labels and their corresponding counts
    class_labels, class_counts = np.unique(dataset.y, return_counts= True)

# Initialize lists for storing indices of train and test samples  
    train_indices = []
    test_indices = []

#Efetuar uma amostragem estratificada para cada classe
       
    for label, count in zip(class_labels, class_counts):
        #Obtém todos os índices em que a etiqueta corresponde à classe atual
        label_indices = np.where(dataset.y == label)[0]
        #garantir aleatoriedade
        np.random.shuffle(label_indices)
        # Calcular o número de amostras a atribuir ao conjunto de teste para a classe atual
        split_point = int(count * test_size)
        #Atribuir a primeira parte dos índices baralhados para testar e o resto para treinar
        test_indices.extend(label_indices[:split_point])
        train_indices.extend(label_indices[split_point:])

    # Convert lists to numpy arrays
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    # Create new Dataset objects for training and testing
    train_dataset = Dataset(X=dataset.X[train_indices, :], y=dataset.y[train_indices], features=dataset.features, label=dataset.label)
    test_dataset = Dataset(X=dataset.X[test_indices, :], y=dataset.y[test_indices], features=dataset.features, label=dataset.label)

    return train_dataset, test_dataset
    