from unittest import TestCase
from datasets import DATASETS_PATH
import os
from si.models.random_forest_classifier import RandomForestClassifier
from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split
from si.models.decision_tree_classifier import DecisionTreeClassifier

class TestRandomForest(TestCase):

    def setUp(self):
        # Caminho para o arquivo CSV do dataset
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        # Carregar o dataset
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        # Dividir em conjunto de treino e teste
        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_fit(self):
        """
        Testa o método _fit para garantir que os parâmetros de inicialização
        sejam corretamente atribuídos e as árvores sejam treinadas.
        """
        random_forest = RandomForestClassifier(n_estimators=10, max_depth=10, min_sample_split=5, mode='gini') #Cria uma instância de RandomForestClassifier com parâmetros específicos
        random_forest.fit(self.train_dataset) #Ajusta o modelo ao conjunto de treino

        # Verifica se o modelo foi ajustado com os parâmetros certos
        self.assertEqual(random_forest.min_sample_split, 5) #verifica se o parâmetro min_sample_split foi configurado corretamente
        self.assertEqual(random_forest.max_depth, 10) #verifica se o parâmetro max_depth foi configurado corretamente.
        self.assertEqual(len(random_forest.trees), 10)  # Verificar se 10 árvores foram criadas, ou seja, Se o número de árvores treinadas corresponde ao especificado (n_estimators=10).

    def test_predict(self):
        """
        Testa o método _predict para garantir que as previsões tenham o mesmo
        tamanho que o número de amostras no conjunto de teste.
        """
        #Objetivo: Garantir que o método predict retorna o número correto de previsões.
        random_forest = RandomForestClassifier(n_estimators=10, max_depth=10, min_sample_split=5, mode='gini')
        random_forest.fit(self.train_dataset) #Ajusta o modelo ao conjunto de treino.

        predictions = random_forest.predict(self.test_dataset)  #Gera as previsões para o conjunto de teste.

        # Verifica se o número de previsões é igual ao número de amostras no conjunto de teste
        self.assertEqual(predictions.shape[0], self.test_dataset.shape()[0])

    def test_score(self):
        """
        Testa o método _score para garantir que a precisão esteja sendo calculada corretamente.
        """
        #Objetivo: Testar o cálculo da precisão (score)
        random_forest = RandomForestClassifier(n_estimators=10, max_depth=10, min_sample_split=5, mode='gini')
        #Ajusta o modelo ao conjunto de treino
        random_forest.fit(self.train_dataset)
        
        accuracy_ = random_forest.score(self.test_dataset) #Calcula a precisão usando o método score com o conjunto de teste.

        # Verifica se a precisão está próxima do valor esperado
        self.assertEqual(round(accuracy_, 2), 0.95) #Verifica se a precisão calculada é aproximadamente igual a 0.95.
        

