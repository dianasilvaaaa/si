import numpy as np
from datasets import DATASETS_PATH
import os
from si.decomposition.pca import PCA
from si.io.csv_file import read_csv
from sklearn.decomposition import PCA as PCA_sk
import pandas as pd


class TestPCA(TestCase):

    def setUp(self):
        # Definir os arquivos e datasets
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)
        self.dataset_sk = pd.read_csv(self.csv_file)

    def test_pca_fit(self):
        # Teste de ajuste (fit)
        pca_custom = PCA(n_components=2).fit(self.dataset)
        pca_sklearn = PCA_sk(n_components=2).fit(self.dataset_sk.iloc[:, :4])

        # Comparando a variância explicada
        explained_variance_custom = pca_custom.explained_variance
        explained_variance_sklearn = pca_sklearn.explained_variance_ratio_

        # Verifica se as variâncias explicadas estão próximas
        self.assertTrue(np.allclose(explained_variance_custom, explained_variance_sklearn[:2], rtol=1e-5))

    def test_pca_transform(self):
        # Teste de transformação (transform)
        pca_custom = PCA(n_components=2).fit(self.dataset)
        x_reduced_custom = pca_custom.transform(self.dataset)

        # Redução de dimensionalidade com sklearn
        pca_sklearn = PCA_sk(n_components=2).fit(self.dataset_sk.iloc[:, :4])
        x_reduced_sklearn = pca_sklearn.transform(self.dataset_sk.iloc[:, :4])

        # Verifica as formas das matrizes transformadas
        self.assertEqual(x_reduced_custom.shape[1], 2)
        self.assertEqual(x_reduced_custom.shape[0], self.dataset.shape()[0])

        # Verifica se a transformação está correta comparando com sklearn
        self.assertTrue(np.allclose(x_reduced_custom, x_reduced_sklearn, rtol=1e-5))
