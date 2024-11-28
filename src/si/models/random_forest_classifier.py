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

        np.random.seed(self.seed) #Define uma semente aleatória

        # Definir max_features se não especificado como a raiz quadrada do número total de características no dataset.
        if self.max_features is None:
            self.max_features = int(np.sqrt(dataset.X.shape[1]))
        # Construção das Árvores de Decisão
        for _ in range(self.n_estimators):
            # Criar um dataset bootstrap com reposição
            bootstrap_indices = np.random.choice(dataset.X.shape[0], size=dataset.X.shape[0], replace=True)
            #Seleciona as amostras e rótulos correspondentes aos índices gerados.
            bootstrap_X = dataset.X[bootstrap_indices]
            bootstrap_y = dataset.y[bootstrap_indices]

            # Selecionar um subconjunto de características
            #Seleciona aleatoriamente um subconjunto de índices de características (colunas do dataset) com tamanho igual a max_features.
            #replace=False garante que cada índice seja único.
            feature_indices = np.random.choice(dataset.X.shape[1], size=self.max_features, replace=False)
            bootstrap_features = [dataset.features[i] for i in feature_indices] #Converte os índices das características para os nomes das mesmas

            # Criar o Dataset com as características e amostras selecionadas
            bootstrap_data = Dataset(X=bootstrap_X[:, feature_indices],
                                     y=bootstrap_y,
                                     features=bootstrap_features,
                                     label=dataset.label)

            # Treinar uma árvore de decisão com o dataset bootstrap
            #Cria uma nova árvore de decisão com os parâmetros definidos para o Random Forest
            tree = DecisionTreeClassifier(
                min_sample_split=self.min_sample_split,
                max_depth=self.max_depth,
                mode=self.mode
            )
            #Treina a árvore com o dataset reduzido criado na etapa anterior.
            tree.fit(bootstrap_data)

            # Armazenar a árvore e os índices das características usadas
            self.trees.append((tree, feature_indices)) #Cada árvore treinada e os índices das características usadas são armazenados na lista

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
        #Cria uma lista para armazenar as predições feitas por cada árvore da floresta para todas as amostras
        all_predictions = []

        #O loop for tree, features in self.trees itera sobre todas as árvores da floresta
        for tree, features in self.trees:
            # Se features contém índices, precisamos convertê-los para os nomes das características
            if isinstance(features[0], int): #Se features[0] for um índice inteiro, significa que as características estão armazenadas como índices das colunas.
                feature_indices = features
                # Convertendo índices para nomes das características
                feature_names = [dataset.features[i] for i in feature_indices]
            else:
                # Caso já esteja com nomes, usamos diretamente
                feature_names = features
                feature_indices = [dataset.features.index(f) if isinstance(f, str) else f for f in feature_names]
            # Selecionar as colunas corretas do dataset
            X_subset = dataset.X[:, feature_indices] #Seleciona apenas as colunas do conjunto de dados correspondentes às características usadas pela árvore.

            # Criar um novo dataset com as características corretas para a árvore
            #novo Dataset com o subconjunto de dados e características (X_subset e feature_names)
            tree_predictions = tree.predict(Dataset(X_subset, dataset.y, features=feature_names, label=dataset.label)) ## árvore gera as predições para todas as amostras.
        
            # Adicionar as previsões da árvore atual
            all_predictions.append(tree_predictions) #Adiciona as predições da árvore à lista:
        
        # Transpor para obter previsões para cada amostra
        all_predictions = np.array(all_predictions).T # Transpor Previsões: Cada linha de all_predictions agora corresponde a uma amostra do conjunto de dados, e cada coluna contém a predição feita por uma árvore
    
        # Para cada linha, encontrar o valor mais comum (classe mais frequente)
        final_predictions = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=all_predictions)
# np.bincount(x.astype(int)).argmax(): Conta a frequência de cada classe nas predições das árvores e seleciona a classe com maior frequência (votação majoritária).
#O resultado é um array com as classes finais preditas para cada amostra.

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
