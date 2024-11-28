import numpy as np
from si.metrics.accuracy import accuracy
from si.models.decision_tree_classifier import DecisionTreeClassifier


class RandomForestClassifier(Model):

    def __init__(self, n_estimators: int = 100, max_features: int = None, min_sample_split: int = 2, max_depth: int = 10, mode: str = 'gini', seed: int = 42,  **kwargs):
        super().__init__(**kwargs)

        """
A Random Forest classifier that uses an ensemble of decision trees.

 Parameters
----------
n_estimators : int
number of decision trees to use

max_features : int
maximum number of features to use per tree

min_sample_split :int
minimum samples allowed in a split

max_depth : int
maximum depth of the trees

mode : str
impurity calculation mode (gini or entropy)

seed : str
random seed to use to assure reproducibility
"""
        self.n_estimators = n_estimators
        self.max_features = max_features 
        self. min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed 

        self.trees = []



    def _fit(self, dataset: Dataset) -> 'RandomForestClassifier':
    
        """
train the decision trees of the random forest
Parameters
        ----------
        dataset : Dataset
            The training dataset.
        
        Returns
        -------
        self : RandomForestClassifier
            The fitted model.

        """


        np.random.seed(self.seed) # Define a semente aleatória

        # Se max_features for None, define-o como sqrt do número de caraterísticas
        if self.max_features is None:
            self.max_features = int(np.sqrt(dataset.X.shape[1]))

        for _ in range(self.n_estimators):
            #passo 3: criar  dataset bootstrap
            bootstrap_indices = np.random.choice(dataset.X.shape[0], size = dataset.X.shape[0], replace = True)
            bootstrap_data = Dataset(dataset.X[bootstrap_indices], dataset.y[bootstrap_indices], features = dataset.features, label = dataseet.label)

            #passo 4: criar e treinar a arvore de decisão com dataset bootstrapped
            #Limit the number of features the tree can use based on max_features
            tree = DecisionTreeClassifier(min_sample_split= self.min_sample_split, max_depth= self.max_depth, mode = self.mode)
            tree.fit(bootstrap_data)

            #passo 5: Anexar a árvore treinada e as caraterísticas utilizadas
            used_features = np.random.choice(dataset.features, size=self.max_features, replace=False)
            self.trees.append((tree, used_features))

        return self
    
    def _predict(self, dataset) -> np.ndarray:
        """
Predict the class labels using the trained random forest.
        
        Parameters
        ----------
        dataset : Dataset
            The test dataset.
        
        Returns
        -------
        np.ndarray
            The predicted class labels for each sample in the dataset.
"""

        all_predictions = []

        for tree, features in self.trees:
            #Selecionar caraterísticas para a árvore atual
            tree_predictions = tree.predict(Dataset(dataset.X[:, [dataset.features.index(f) for f in features]], dataset.y, features = features, label = dataset.label))
            all_predictions.append(tree_predictions)

            #passo 2: obter a classe prevista mais comum para cada amostra
            all_predictions = np.array(all_predictions).T #transpor para obter previsões para cada amostra
            final_predictions = [np.bincount(pred).argmax() for pred in all_predictions]

        return np.array(final_predictions)
    
    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:

        """

        Computes the accuracy of the random forest model.
        
        Parameters
        ----------
        dataset : Dataset
            The test dataset.
        predictions : np.ndarray
            The predicted class labels.
        
        Returns
        -------
        float
            The accuracy of the model on the test dataset.
        
"""

        return accuracy(dataset.y, predictions)
