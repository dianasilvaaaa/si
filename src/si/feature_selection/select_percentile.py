import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics import f_classification


class SelectPercentile(Transformer):
    
    def __init__(self, percentile:float, score_func:callable = f_classification,**kwargs):
        """
        Selects features from the given percentile of a score function and returns a new Dataset object with the selected features

        Parameters
        ----------
        percentile: float
            Percentile for selecting features
        
        score_func: callable, optional
            Variance analysis function. Use the f_classification by default for
        """
        super().__init__(**kwargs)
        if isinstance(percentile,int):
            self.percentile = percentile
        else:
            raise ValueError("Percentile must be a integer between 0 and 100")
        self.score_func = score_func
        self.F = None
        self.p = None

    def _fit(self,dataset:Dataset) -> 'SelectPercentile':

        """
        Estimate the F and P values for each feature using the scoring function

        Parameters
        ----------
        dataset: Dataset
            - Dataset object where is intended to select features
        
        Returns
        ----------
        self: object
            - Returns self instance with the F and P values for each feature calculated using the scoring function.
        """

        self.F,self.p = self.score_func(dataset) #A função score_func calcula os escores (F) e p-values (p) para todas as características.
        
        return self
    
    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Selects features with the highest F value up to the specified percentile

        Parameters
        ----------
        dataset: Dataset
            - Dataset object where is intended to select features
        
        Returns
        ----------
        dataset: Dataset
            - A new Dataset object with the selected features
        
        """
        # calculata o threshold para os scores
        threshold= np.percentile(self.F,100-self.percentile)
        # selecionar as caraterísticas com score superior ao threshold
        mask = self.F > threshold
        # Verificar se existem caraterísticas com o mesmo score que o threshould
        ties = np.where(self.F == threshold)[0]
        if len(ties) != 0:
            # calcula o número máximo de caraterísticas que podem ser mantidas sem que o número total de características selecionadas não exceda o percentual especificado.
            max_features = int (len(self.F)*self.percentile/100)
            # seleciona os ties que devem integrar as caraterísticas
            # Altera o valor destas caraterísticas para Verdadeiro na máscara
            # para garantir que o número total de características selecionadas não ultrapasse o máximo permitido (max_features).
            mask[ties[: max_features -mask.sum()]] = True

        #Filtra os nomes das características com base na máscara (mask).
        #Apenas as características que têm True na máscara serão mantidas.
        features = np.array(dataset.features)[mask]
        
        return Dataset(X=dataset.X[:, mask], y=dataset.y, features=list(features), label=dataset.label) #Filtra as características e retorna um novo Dataset com os dados correspondentes
        #Cria um novo Dataset com:
        #Apenas as colunas de X correspondentes às características selecionadas (dataset.X[:, mask]).
        #O alvo (y) e o rótulo do dataset (label) permanecem inalterados.
