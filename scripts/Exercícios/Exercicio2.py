"""
Exercise 2: NumPy array Indexing/Slicing
2.1) Adicione um método à classe Dataset que remova todas as amostras que contenham pelo menos um valor nulo (NaN).
 Note que o objeto resultante não deve conter valores nulos em nenhuma caraterística/variável independente. 
 Além disso, note que deve atualizar o vetor y removendo as entradas associadas às amostras a serem removidas. 
 Deve utilizar apenas funções NumPy. Nome do método: dropna

def dropna
argumentos:
None (Nenhum)
resultado esperado:
self (objeto Dataset modificado)

"""

import numpy as np

class Dataset:
    def __init__(self, X: np.ndarray, y: np.ndarray = None, features: list = None, label: str = None):
        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def dropna(self) -> 'Dataset':
        """
        Remove all samples that contain at least one NaN value.
        Also updates the target variable (y) by removing the corresponding entries.

        Returns
        -------
        Dataset
            The Dataset object with NaN samples removed.
        """
        # Identify rows with NaN values
        mask = ~np.isnan(self.X).any(axis=1)
#np.isnan(self.X): Identifica todas as posições na matriz X que contêm NaN
#any(axis=1): Verifica, para cada linha, se existe pelo menos um valor NaN        
#~np.isnan(self.X).any(axis=1): cria uma máscara que retorna True para as linhas que não têm valores NaN


        # Filter the data
        self.X = self.X[mask]
#self.X[mask]: Mantém apenas as linhas onde mask é True e descarta aquelas onde mask é False, removendo assim as amostras que contêm NaN


        # If y exists, filter it as well
        if self.y is not None:
            self.y = self.y[mask]
#self.y[mask]: Se o vetor de rótulos y existir, ele também é filtrado de acordo com a mesma máscara, para garantir que as amostras associadas aos valores NaN também sejam removidas de y.
            
            
        # Return the updated Dataset object
        return self


"""
2.2) Adicionar um método à classe Dataset que substitui todos os valores nulos por outro valor ou 
pela média ou mediana da caraterística/variável. 
Note que o objeto resultante não deve conter valores nulos em nenhuma caraterística/variável independente. 
Deve-se utilizar apenas funções NumPy. Nome do método: fillna

def fillna
argumentos:
value - float ou “mean” ou “median”
resultado esperado:
self (objeto Dataset modificado)

"""

import numpy as np

class Dataset:
    
    def fillna(self, value) -> 'Dataset':
        """
        Replaces all NaN values in the dataset with a specified value, or with the mean or median of the corresponding feature.
        
        Parameters
        ----------
        value : float or str ('mean' or 'median')
            The value to replace NaN values with. If 'mean', the NaN values will be replaced by the mean of the corresponding feature.
            If 'median', the NaN values will be replaced by the median of the corresponding feature.

        Returns
        -------
        Dataset
            The Dataset object with NaN values replaced.
        """
        # Iterating over each feature (column)
        for i in range(self.X.shape[1]):  # Looping over columns (features)
            col = self.X[:, i]  # Select the i-th feature (column)
            nan_mask = np.isnan(col)  # Mask identifying NaN values

            if np.any(nan_mask):  # Check if there are any NaN values
                if value == 'mean':
                    fill_value = np.nanmean(col)  # Calculate mean ignoring NaN
                elif value == 'median':
                    fill_value = np.nanmedian(col)  # Calculate median ignoring NaN
                else:
                    fill_value = value  # Use the specified value directly

                # Replace NaN values with the fill_value
                col[nan_mask] = fill_value
                self.X[:, i] = col  # Update the dataset with the modified column

        return self  # Return the updated Dataset object
    

"""
2.3) Adicione um método à classe Dataset que remove uma amostra pelo seu índice. 
Note que deve também atualizar o vetor y, removendo a entrada associado à amostra a ser removida. 
Deve utilizar apenas funções NumPy. Nome do método: remove_by_index

def remove_by_index
argumentos:
index - número inteiro correspondente à amostra a remover
resultado esperado:
self (objeto Dataset modificado)
"""

import numpy as np

class Dataset:

    def remove_by_index(self, index: int) -> 'Dataset':
        """
        Removes a sample by its index from the dataset and updates the corresponding target (y).

        Parameters
        ----------
        index : int
            The index of the sample to remove.

        Returns
        -------
        Dataset
            The Dataset object with the specified sample removed.
        """
        # Remove the sample at the given index from X
        self.X = np.delete(self.X, index, axis=0)

        # If y exists, remove the corresponding value in y
        if self.y is not None:
            self.y = np.delete(self.y, index)

        # Return the updated Dataset object
        return self
