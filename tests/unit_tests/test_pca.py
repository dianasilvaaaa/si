import numpy as np
from si.data.dataset import Dataset
from unittest import TestCase
from si.io.csv_file import read_csv
import os
import pandas as pd
from sklearn.decomposition import PCA as PCA_sk
from si.decomposition.pca import PCA
import unittest

class TestPCA(TestCase):

    def setUp(self): #Define o caminho para o dataset Iris 
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

        O test_pca_fit compara o ajuste (cálculo da média, variância explicada, componentes principais) da implementação customizada com o scikit-learn.
        """
        pca = PCA(n_components=2) 
        iris_fit = pca.fit(self.dataset)
        #Ajusta o PCA aos dados no formato do objeto Dataset
        iris_fit_sklearn = PCA_sk(n_components=2).fit(self.dataset_sk.iloc[:, :4]) 
        #Ajusta o PCA aos mesmos dados, mas no formato do Pandas (somente as 4 primeiras colunas, que são os atributos numéricos)

        # Testa se a variância explicada está próxima
        self.assertTrue(np.allclose(iris_fit.explained_variance, iris_fit_sklearn.explained_variance_ratio_)) #Variância Explicada: Compara a variância explicada pelo PCA customizado e pelo scikit-learn.
        
        # Testa o número de componentes garantindo que o número de componentes principais ajustados seja o esperado.
        self.assertEqual(len(iris_fit.explained_variance), 2)
        self.assertEqual(iris_fit.components.shape[1], 2)
        #Garante que a quantidade de componentes principais ajustados é 2
        
        # Testa a média. Verifica se as médias dos atributos calculadas são semelhantes.
        self.assertTrue(np.allclose(iris_fit.mean, iris_fit_sklearn.mean_))
        #Verifica se a média calculada pelo PCA customizado é semelhante à do scikit-learn

    def test_pca_transform(self):
        """
        Testa o método transform do PCA verificando o tamanho das dimensões transformadas
        
        O test_pca_transform verifica se a transformação reduz corretamente as dimensões dos dados.
        """
        pca = PCA(n_components=2)
        x_reduced = pca.fit_transform(self.dataset) #Ajusta e transforma os dados em uma única etapa usando o método fit_transform da implementação customizada.
        #Ajusta o PCA e transforma os dados em uma única etapa

        self.assertEqual(x_reduced.X.shape[1], 2) #Verifica se os dados foram reduzidos para 2 dimensões (especificado por n_components).
        self.assertEqual(x_reduced.X.shape[0], self.dataset.X.shape[0]) #Garante que o número de linhas (amostras) permaneça o mesmo após a transformação

if __name__ == "__main__":
    unittest.main()
