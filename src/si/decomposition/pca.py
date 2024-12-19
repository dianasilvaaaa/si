import numpy as np
from si.data.dataset import Dataset
from si.base.transformer import Transformer

class PCA:
    def __init__(self, n_components):
# Método inicializador da classe. Define o número de componentes principais a serem retidos e inicializa outras propriedades.

        """
        Análise de Componentes Principais (PCA)

        Parâmetros
        ----------
        n_components: int
            Número de componentes principais a serem retidos.
        """
        self.n_components = n_components #número de componentes principais que a classe deve calcular.
        self.mean = None #Média de cada atributo (usada para centralizar os dados)
        self.components = None #Inicializa o atributo components, que armazenará os componentes principais calculados.
        self.explained_variance = None #Fração da variância total explicada por cada componente principal

    def _validate_input(self, dataset: Dataset):

    #Método privado para validar o dataset e o número de componentes principais.

        """
        Valida o dataset de entrada e o parâmetro n_components.

        Parâmetros
        ----------
        dataset: Dataset
            O dataset a ser validado.

        Erros
        ------
        ValueError:
            Se n_components for inválido ou se o dataset for incompatível.
        """
        # verifica se o dataset é uma instância da classe Dataset
        if not isinstance(dataset, Dataset):
            raise ValueError("A entrada deve ser uma instância de Dataset.") # Garante que o objeto fornecido seja uma instância da classe Datase
        
        # verifica se n_components é válido:maior que 0 e menor ou igual ao número de atributos  
        #Verifica se o número de componentes principais é válido (maior que 0 e menor ou igual ao número de atributos do dataset).
        if self.n_components <= 0 or self.n_components > dataset.X.shape[1]:
            raise ValueError("n_components deve ser um número inteiro positivo e menor ou igual ao número de atributos.")

    def _fit(self, dataset: Dataset) -> "PCA":
        # método fit ajusta o PCA ao dataset
        """
        Estima a média, os componentes principais e a variância explicada.

        Parâmetros
        ----------
        dataset: Dataset
            O dataset no qual o PCA será ajustado.

        Retornos
        -------
        self: PCA
            O objeto PCA ajustado.
        """
        self._validate_input(dataset)

        # Passo 1: Centralizar os dados (subtraindo a média de cada atributo)
        X = dataset.X #Obtém as features do dataset.

        self.mean = np.mean(X, axis=0) #calcula a média de cada atributo (coluna) do dataset X ao longo das observações (linhas)
        #axis=0 (ao longo das linhas)
        
        X_centered = X - self.mean # Centraliza os dados subtraindo a média.

        # Passo 2: Calcular a matriz de covariância dos dados centrados e efetuar a decomposição dos valores próprios
        matriz_covariancia = np.cov(X_centered, rowvar=False) # Calcula a matriz de covariância para entender as relações entre os atributos
        # rowvar=False: inversão e as variáveis estão nas colunas e as observações estão nas linhas 
        
        autovalores, autovetores = np.linalg.eig(matriz_covariancia) #Autovalores representam a variância explicada em cada direção dos dados.
        #Autovetores representam as direções dos componentes principais

        # Ordenar autovalores e autovetores em ordem decrescente de variancia 
        indices_ordenados = np.argsort(autovalores)[::-1]
        autovalores = autovalores[indices_ordenados] # Autovalores representam a variância em cada direção
        autovetores = autovetores[:, indices_ordenados] #Autovetores são as direções principais

        # Passo 3: Selecionar os componentes principais
        self.components = autovetores[:, :self.n_components] #leciona os autovetores correspondentes aos maiores autovalores.

        # Passo 4: Calcular a variância explicada
        variancia_total = np.sum(autovalores) #Calcula a variância total dos dados.
        self.explained_variance = autovalores[:self.n_components] / variancia_total #Calcula a fração de variância explicada por cada componente principal.

        return self

    def fit(self, dataset: Dataset) -> "PCA":
        """
        Método público para ajustar o PCA a um dataset.

        Parâmetros
        ----------
        dataset: Dataset
            O dataset no qual o PCA será ajustado.

        Retornos
        -------
        self: PCA
            O objeto PCA ajustado.
        """
        return self._fit(dataset)

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Transforma o dataset usando os componentes principais.

        Parâmetros
        ----------
        dataset: Dataset
            O dataset a ser transformado.

        Retornos
        -------
        Dataset
            Dataset transformado com os componentes principais.
        """
        if self.components is None or self.mean is None: #if self.components is None or self.mean is None:
            raise ValueError("O PCA deve ser ajustado antes de chamar o método transform.")

        # Passo 1: Centralizar os dados
        X_centered = dataset.X - self.mean #Os dados devem ser centralizados novamente, usando a média calculada durante o ajuste.

        # Passo 2: Projetar os dados nos componentes principais
        X_reduced = np.dot(X_centered, self.components) #A projeção é feita usando o produto escalar dos dados centralizados com os componentes principais
        #Essa projeção reduz os dados originais para o número de dimensões especificado por n_components

        # Passo 3: Criar um novo objeto Dataset com os dados reduzidos.
        return Dataset(X_reduced, features=[f"PC{i+1}" for i in range(self.n_components)], y=dataset.y)
        #Um novo objeto Dataset é criado com os dados reduzidos.


    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Ajusta o PCA ao dataset e o transforma.

        Parâmetros
        ----------
        dataset: Dataset
            O dataset a ser ajustado e transformado.

        Retornos
        -------
        Dataset
            Dataset transformado com os componentes principais.
        """
        self.fit(dataset) #Ajusta o modelo.
        return self.transform(dataset) #Transforma os dados e retorna o dataset reduzido.
#Este método combina os passos de ajuste (fit) e projeção (transform) numa única operação. É útil para reduzir o número de chamadas no código