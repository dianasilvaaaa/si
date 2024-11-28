from unittest import TestCase
from datasets import DATASETS_PATH
import os
from si.models.random_forest_classifier import RandomForestClassifier
from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split


class TestRandomForest(TestCase):
    def setUp(self):
        # Carregar o dataset
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")
        
        # Dividir o dataset em treino e teste
        self.train_dataset, self.test_dataset = train_test_split(self.dataset, test_size=0.2, random_state=42)

    def test_fit(self):
        # Configurar o RandomForestClassifier
        random_forest = RandomForestClassifier(n_estimators=10, max_features=None, min_sample_split=5, max_depth=10, mode="gini", seed=42)
        
        # Treinar o modelo
        random_forest.fit(self.train_dataset)
        
        # Verificar o número de árvores treinadas
        self.assertEqual(len(random_forest.trees), 10, "O número de árvores treinadas deve ser 10.")
        
        # Verificar os parâmetros configurados
        self.assertEqual(random_forest.min_sample_split, 5, "O min_sample_split não está configurado corretamente.")
        self.assertEqual(random_forest.max_depth, 10, "O max_depth não está configurado corretamente.")

    def test_predict(self):
        # Configurar e treinar o RandomForestClassifier
        random_forest = RandomForestClassifier(n_estimators=10, max_features=None, min_sample_split=5, max_depth=10, mode="gini", seed=42)
        random_forest.fit(self.train_dataset)
        
        # Fazer previsões
        predictions = random_forest.predict(self.test_dataset)
        
        # Verificar o tamanho das previsões
        self.assertEqual(predictions.shape[0], self.test_dataset.X.shape[0], "O número de previsões não corresponde ao número de amostras de teste.")
        
        # Validar que as previsões estão nas classes esperadas
        unique_classes = np.unique(self.train_dataset.y)
        self.assertTrue(all(pred in unique_classes for pred in predictions), "Previsões contêm classes inválidas.")

    def test_score(self):
        # Configurar e treinar o RandomForestClassifier
        random_forest = RandomForestClassifier(n_estimators=10, max_features=None, min_sample_split=5, max_depth=10, mode="gini", seed=42)
        random_forest.fit(self.train_dataset)
        
        # Calcular a precisão
        accuracy_ = random_forest.score(self.test_dataset)
        
        # Garantir que a precisão está dentro do esperado
        self.assertGreaterEqual(accuracy_, 0.8, "A precisão do modelo é inferior ao esperado (0.8).")
