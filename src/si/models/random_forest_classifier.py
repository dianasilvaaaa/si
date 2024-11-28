import numpy as np
from si.metrics.accuracy import accuracy
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.data.dataset import Dataset
from si.base.model import Model

"""
    Random Forest Classifier: Uma técnica de aprendizado de máquina de conjunto
    que combina várias árvores de decisão para melhorar a precisão do modelo,
    reduzindo o overfitting.
    """

class RandomForestClassifier(Model):

    def __init__(self, 
                 n_estimators: int = 100,
                 max_features: int = None,
                 min_sample_split: int = 2,
                 max_depth: int = 10,
                 mode: str = 'gini',
                 seed: int = 42,
                 **kwargs):

   
    
    
        """
        Parâmetros:
        ----------
        n_estimators : int
            Número de árvores de decisão no modelo.
        max_features : int
            Número máximo de características a serem usadas por árvore.
        min_sample_split : int
            Número mínimo de amostras necessário para dividir um nó.
        max_depth : int
            Profundidade máxima das árvores.
        mode : str
            Modo de cálculo de impureza (gini ou entropy).
        seed : int
            Semente para gerar resultados reprodutíveis.
        """
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed
        self.trees = []  # Lista para armazenar árvores e recursos usados.
    

    def _fit(self, dataset: Dataset) -> 'RandomForestClassifier':

        """
        Treina o modelo usando o conjunto de dados fornecido.

        Parâmetros:
        ----------
        dataset : Dataset
            Conjunto de dados de treinamento.

        Retorna:
        -------
        self : RandomForestClassifier
            O modelo treinado.
        """

        np.random.seed(self.seed)

        # Definir max_features se não especificado
        if self.max_features is None:
            self.max_features = int(np.sqrt(dataset.X.shape[1]))

        for _ in range(self.n_estimators):
            # Criar um dataset bootstrap com reposição
            bootstrap_indices = np.random.choice(dataset.X.shape[0], size=dataset.X.shape[0], replace=True)
            bootstrap_X = dataset.X[bootstrap_indices]
            bootstrap_y = dataset.y[bootstrap_indices]

            # Selecionar um subconjunto de características
            feature_indices = np.random.choice(dataset.X.shape[1], size=self.max_features, replace=False)
            bootstrap_features = [dataset.features[i] for i in feature_indices]

            # Criar o Dataset com as características e amostras selecionadas
            bootstrap_data = Dataset(X=bootstrap_X[:, feature_indices],
                                     y=bootstrap_y,
                                     features=bootstrap_features,
                                     label=dataset.label)

            # Treinar uma árvore de decisão com o dataset bootstrap
            tree = DecisionTreeClassifier(
                min_sample_split=self.min_sample_split,
                max_depth=self.max_depth,
                mode=self.mode
            )
            tree.fit(bootstrap_data)

            # Armazenar a árvore e os índices das características usadas
            self.trees.append((tree, feature_indices))

        return self
    
    def _predict(self, dataset) -> np.ndarray:
        """
    Prediz as classes usando a floresta de árvores treinada.

    Parameters
    ----------
    dataset : Dataset
        O conjunto de dados de teste.

    Returns
    -------
    np.ndarray
        As previsões das classes para cada amostra no dataset.
    """
        
        all_predictions = []

        for tree, features in self.trees:
            # Se features contém índices, precisamos convertê-los para os nomes das características
            if isinstance(features[0], int):
                feature_indices = features
                # Convertendo índices para nomes das características
                feature_names = [dataset.features[i] for i in feature_indices]
            else:
                # Caso já esteja com nomes, usamos diretamente
                feature_names = features
                feature_indices = [dataset.features.index(f) if isinstance(f, str) else f for f in feature_names]
            # Selecionar as colunas corretas do dataset
            X_subset = dataset.X[:, feature_indices]

            # Criar um novo dataset com as características corretas para a árvore
            tree_predictions = tree.predict(Dataset(X_subset, dataset.y, features=feature_names, label=dataset.label))
        
            # Adicionar as previsões da árvore atual
            all_predictions.append(tree_predictions)

        # Transpor para obter previsões para cada amostra
        all_predictions = np.array(all_predictions).T
    
        # Para cada linha, encontrar o valor mais comum (classe mais frequente)
        final_predictions = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=all_predictions)

        return np.array(final_predictions)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:

        """
        Calcula a precisão do modelo no conjunto de dados fornecido.

        Parâmetros:
        ----------
        dataset : Dataset
            Conjunto de dados de teste.
        predictions : np.ndarray
            Predições feitas pelo modelo.

        Retorna:
        -------
        float
            A precisão do modelo.
        """

        return accuracy(dataset.y, predictions)
