import numpy as np
from si.base.estimator import Estimator
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy

class StackingClassifier(Model):

    """
Stacking classifier that combines multiple models using a final model.

    Parameters
    ----------
    models : list
        The initial set of models to be used in the stacking ensemble.
    final_model : Model
        The final model to make predictions based on the output of the initial models.

"""

    def __init__(self, models, final_models, **kwargs):
        super().__init__(**kwargs)

        self.models = models #models: Lista de modelos base (inicialmente treinados nos dados originais).
        self.final_models = final_models #final_model: Modelo que combinará as previsões dos modelos base.

    def _fit(self, dataset: Dataset) -> 'StackingClassifier':

        """
 Train the initial models and the final model.

        Parameters
        ----------
        dataset : Dataset
            The training data.

        Returns
        -------
        self : StackingClassifier
            The fitted stacking model.

"""
        #treina os modelos iniciais (base)
        for model in self.models: #Itera sobre cada modelo em self.models e chama o método fit para treiná-los nos dados originais.
            model.fit(dataset)

        #obtem as previsões dos modelos iniciais
        #Gera previsões para cada modelo base e organiza-as em colunas de uma matriz (np.column_stack).
        predictions = np.column_stack([model.predict(dataset) for model in self.models]) # Essas previsões são organizadas em colunas (matriz predictions), onde cada coluna representa as previsões de um modelo.

        #treina o modelo final com as previsões dos modelos iniciais
        final_dataset = Dataset(X = predictions, y = dataset.y) #As previsões (X) são combinadas com os rótulos verdadeiros (y) para formar um novo dataset (final_dataset).
        self.final_models.fit(final_dataset) #O modelo final é treinado para combinar essas previsões numa decisão final
        #O modelo final (self.final_models) é treinado usando o final_dataset.

        return self
    
    def _predict(self, dataset) -> np.ndarray:
        
        """
 Predict the class labels using the stacking ensemble.

        Parameters
        ----------
        dataset : Dataset
            The test data.

        Returns
        -------
        np.ndarray
            Predicted class labels.


"""

        #obtem as previsões dos modelos iniciais (base)
        #Cada modelo base faz previsões para os dados de entrada.
        #As previsões são organizadas em uma matriz.
        predicitions = np.column_stack([model.predict(dataset) for model in self.models]) # as previsões dos modelos base são combinadas numa matriz

        #Obtem as previsões finais utilizando o modelo final
        final_dataset = Dataset(X = predicitions, y=None) #Cria um dataset usando as previsões como X (atributos) e None para y (sem rótulos).

        # As previsões dos modelos base são usadas como entrada para o modelo final, que faz a previsão final
        return self.final_models.predict(final_dataset) #O modelo final (self.final_models) usa essas previsões como entrada para gerar as previsões finais.
        
       




    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        
        """
 Compute the accuracy of the stacking model.

        Parameters
        ----------
        dataset : Dataset
            The test data.
        predictions : np.ndarray
            Predicted class labels.

        Returns
        -------
        float
            Accuracy of the stacking model.


"""

        return accuracy(dataset.y, predictions)
    #Calcula a acurácia comparando os rótulos verdadeiros (dataset.y) com as previsões geradas.
    #Usa os rótulos verdadeiros (dataset.y) e as previsões (predictions) para calcular a acurácia do modelo.