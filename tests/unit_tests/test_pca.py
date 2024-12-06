import numpy as np
from si.data.dataset import Dataset
import unittest
from si.io.csv_file import read_csv
import os
import pandas as pd
from sklearn.decomposition import PCA as PCA_sk

class PCA:
    def __init__(self, n_components):
        """
        Análise de Componentes Principais (PCA)

        Parâmetros
        ----------
        n_components: int
            Número de componentes principais a serem retidos.
        """
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def _validate_input(self, dataset: Dataset):
        """
        Valida o dataset de entrada e o parâmetro n_components.

        Parâmetros
        ----------
        dataset: Dataset
            O dataset a ser validado.

        Erros
        ------
        ValueError:
            Se n_components for inválido ou se o dataset for incompatível.
        """
        if not isinstance(dataset, Dataset):
            raise ValueError("A entrada deve ser uma instância de Dataset.")

        if self.n_components <= 0 or self.n_components > dataset.X.shape[1]:
            raise ValueError("n_components deve ser um número inteiro positivo e menor ou igual ao número de atributos.")

    def _fit(self, dataset: Dataset) -> "PCA":
        """
        Estima a média, os componentes principais e a variância explicada.

        Parâmetros
        ----------
        dataset: Dataset
            O dataset no qual o PCA será ajustado.

        Retornos
        -------
        self: PCA
            O objeto PCA ajustado.
        """
        self._validate_input(dataset)

        # Passo 1: Centralizar os dados
        X = dataset.X
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Passo 2: Calcular a matriz de covariância e decomposição dos autovalores
        matriz_covariancia = np.cov(X_centered, rowvar=False)
        autovalores, autovetores = np.linalg.eig(matriz_covariancia)

        # Passo 3: Ordenar autovalores e autovetores em ordem decrescente
        indices_ordenados = np.argsort(autovalores)[::-1]
        autovalores = autovalores[indices_ordenados]
        autovetores = autovetores[:, indices_ordenados]

        # Passo 4: Selecionar os componentes principais
        self.components = autovetores[:, :self.n_components]

        # Passo 5: Calcular a variância explicada
        variancia_total = np.sum(autovalores)
        self.explained_variance = autovalores[:self.n_components] / variancia_total

        return self

    def fit(self, dataset: Dataset) -> "PCA":
        """
        Método público para ajustar o PCA a um dataset.

        Parâmetros
        ----------
        dataset: Dataset
            O dataset no qual o PCA será ajustado.

        Retornos
        -------
        self: PCA
            O objeto PCA ajustado.
        """
        return self._fit(dataset)

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Transforma o dataset usando os componentes principais.

        Parâmetros
        ----------
        dataset: Dataset
            O dataset a ser transformado.

        Retornos
        -------
        Dataset
            Dataset transformado com os componentes principais.
        """
        if self.components is None or self.mean is None:
            raise ValueError("O PCA deve ser ajustado antes de chamar o método transform.")

        # Passo 1: Centralizar os dados
        X_centered = dataset.X - self.mean

        # Passo 2: Projetar os dados nos componentes principais
        X_reduced = np.dot(X_centered, self.components)

        # Passo 3: Criar um novo objeto Dataset
        return Dataset(X_reduced, features=[f"PC{i+1}" for i in range(self.n_components)], y=dataset.y)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Ajusta o PCA ao dataset e o transforma.

        Parâmetros
        ----------
        dataset: Dataset
            O dataset a ser ajustado e transformado.

        Retornos
        -------
        Dataset
            Dataset transformado com os componentes principais.
        """
        self.fit(dataset)
        return self.transform(dataset)

class TestPCA(unittest.TestCase):

    def setUp(self):
        """
        Configura os dados de teste carregando o dataset de Iris.
        """
        datasets_path = os.getenv("DATASETS_PATH", "./datasets")
        self.csv_file = os.path.join(datasets_path, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)
        self.dataset_sk = pd.read_csv(self.csv_file)

    def test_pca_fit(self):
        """
        Testa o método fit do PCA comparando os resultados com a implementação do scikit-learn.
        """
        pca = PCA(n_components=2)
        iris_fit = pca.fit(self.dataset)
        iris_fit_sklearn = PCA_sk(n_components=2).fit(self.dataset_sk.iloc[:, :4])

        # Testa se a variância explicada está próxima
        self.assertTrue(np.allclose(iris_fit.explained_variance, iris_fit_sklearn.explained_variance_ratio_))
        # Testa o número de componentes
        self.assertEqual(len(iris_fit.explained_variance), 2)
        self.assertEqual(iris_fit.components.shape[1], 2)
        # Testa a média
        self.assertTrue(np.allclose(iris_fit.mean, iris_fit_sklearn.mean_))

    def test_pca_transform(self):
        """
        Testa o método transform do PCA verificando o tamanho das dimensões transformadas.
        """
        pca = PCA(n_components=2)
        x_reduced = pca.fit_transform(self.dataset)
        self.assertEqual(x_reduced.X.shape[1], 2)
        self.assertEqual(x_reduced.X.shape[0], self.dataset.X.shape[0])

if __name__ == "__main__":
    unittest.main()
