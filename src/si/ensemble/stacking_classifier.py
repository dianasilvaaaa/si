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

        self.models = models
        self.final_models = final_models

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
        #treina os modelos iniciais
        for model in self.models:
            model.fit(dataset)

        #obtem as previsões dos modelos iniciais
        predictions = np.column_stack([model.predict(dataset) for model in self.models])

        ##treina o modelo final com as previsões dos modelos iniciais
        final_dataset = Dataset(X = predictions, y = dataset.y)
        self.final_models.fit(final_dataset)

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

        #obtem as previsões dos modelos iniciais
        predicitions = np.column_stack([model.predict(dataset) for model in self.models])

        #Obtem as previsões finais utilizando o modelo final
        final_dataset = Dataset(X = predicitions, y=None)

        return self.final_models.predict(final_dataset)
    


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
    