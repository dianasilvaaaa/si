import numpy as np
from si.data.dataset import Dataset

class PCA: 
    def _init_(self, n_components, **kwargs):


        """
Principal Component Analysis (PCA)

        Parameters
        ----------
        n_components: int
            Number of principal components to retain.

        """
        self.n_components = n_components
        self.mean = None
        n_components = None
        self.explained_variance = None

    def _fit(self, dataset: Dataset) -> "PCA":
        """
        Estimates the mean, principal components, and explained variance.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the PCA on.

        Returns
        -------
        self: PCA
            The fitted PCA object.
        """
        # 1. Centralizar os dados
        X = dataset.X
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # 2. Calcular a matriz de covariância e decomposição de autovalores
        covariance_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # 3. Ordenar autovalores e autovetores em ordem decrescente
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # 4. Inferir os componentes principais
        self.components = eigenvectors[:, :self.n_components]

        # 5. Calcular a variância explicada
        total_variance = np.sum(eigenvalues)
        self.explained_variance = eigenvalues[:self.n_components] / total_variance

        return self

    
def _transform (self, dataset: Dataset) -> Dataset:

    """
        Transforms the dataset using the principal components.

        Parameters
        ----------
        dataset: Dataset
            The dataset to transform.

        Returns
        -------
        Dataset
            Transformed dataset with principal components.
        """
    
    #centralizar os dados
    X_centered = dataset.X - self.mean

    #reduzir os dados multiplicando pelos componentes principais
    x_reduced = np.dot(X_centered, self.components)

    #retorna um novo dataset com os dados transformados~
    return Dataset(x_reduced, features=[f"PC{i+1}" for i in range(self.n_components)], labels=dataset.y)