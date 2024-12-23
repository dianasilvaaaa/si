import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class LassoRegression(Model):
    """
    Regressão Lasso com regularização L1 para ajustar coeficientes.
    """

    def __init__(self, l1_penalty: float = 1, max_iter: int = 1000, patience: int = 5, scale: bool = True, **kwargs):
        """
        Inicializa o modelo Lasso Regression.

        Parâmetros:
        ----------
        l1_penalty : float
            Parâmetro de regularização L1.A penalização L1 (parâmetro de regularização) para controle da magnitude dos coeficientes.
        max_iter : int
            Número máximo de iterações que o modelo deve realizar durante o ajuste dos parâmetros.
        patience : int
            Número de iterações sem melhoria no erro para ativar o early stopping (interrupção antecipada).
        scale : bool
           e verdadeiro, os dados são normalizados (média 0 e desvio padrão 1).

          Atributos:
        ----------
        theta : np.ndarray
            Coeficientes do modelo (parâmetros ajustados durante o treinamento).
        theta_zero : float
            Intercepto (coeficiente de viés).
        mean : np.ndarray
            Média de cada coluna dos dados (usada para normalização).
        std : np.ndarray
            Desvio padrão de cada coluna dos dados (usado para normalização).
        cost_history : dict
            Histórico do custo (erro) durante as iterações.
        """

        super().__init__(**kwargs) # Inicializa qualquer comportamento herdado de Model
        self.l1_penalty = l1_penalty # Define o parâmetro de penalização L1
        self.max_iter = max_iter # Define o número máximo de iterações
        self.patience = patience # Define o limite de iterações sem melhoria
        self.scale = scale # Indica se os dados devem ser normalizados

        # Inicializa os atributos do modelo
        self.theta = None # Coeficientes iniciais (não definidos até o treinamento)
        self.theta_zero = None # Intercept inicial
        self.mean = None # Média para normalização (calculada no treinamento)
        self.std = None # Desvio padrão para normalização (calculado no treinamento)
        self.cost_history = {} # Dicionário para armazenar o histórico do custo

    def _fit(self, dataset: Dataset) -> 'LassoRegression':
        """
        Ajusta o modelo aos dados fornecidos.

        Parâmetros:
        ----------
        dataset : Dataset
            Conjunto de dados de entrada.

        Retorna:
        -------
        self : LassoRegression
            O modelo ajustado.
        """
        # Verifica se os dados precisam ser normalizados
        # A normalização é feita subtraindo a média e dividindo pelo desvio padrão.
        """
        Se o parâmetro scale=True, os dados de entrada (dataset.X) são normalizados (cada variável tem média 0 e desvio padrão 1). 
        A variável mean armazena a média de cada coluna de X, e std armazena o desvio padrão de cada coluna.
        Se a normalização não for necessária (scale=False), a variável X simplesmente recebe os dados de entrada como estão

        """

#1 NORMALIZAÇÃO DOS DADOS: 
#Objetivo: Garantir que todas as variáveis 𝑋𝑗 tenham média zero e desvio padrão unitário, evitando que variáveis com escalas maiores dominem o ajuste do modelo.
# Normaliza os dados (X) se necessário, garantindo que tenham média zero e desvio padrão unitário.
        if self.scale:
            self.mean, self.std = dataset.X.mean(axis=0), dataset.X.std(axis=0) # Calcula média e desvio padrão
            X = (dataset.X - self.mean) / (self.std + 1e-8)   # Normaliza os dados 
        else:
            X = dataset.X  
#calcula-se a media e o desvio padrão. Xnormalizado = (X-media)/desvio padrao



#2 INICIALIZAÇÃO DOS PARÂMETROS:
#Objetivo: Começar com valores neutros para os coeficientes (𝜃𝑗) e o intercepto (𝜃0).
#Inicializa os coeficientes (theta) e o intercepto (theta_zero) com zeros.
        m, n = X.shape # Obtém o número de amostras (m) e características (n)
        # Inicializa os parâmetros
        # m e n representam, respectivamente, o número de amostras (linhas) 
        # e n número de características (colunas) de X.
        self.theta = np.zeros(n) # é um vetor de coeficientes do modelo, e é inicializado com zeros. Ele terá o mesmo número de elementos que o número de características em X.
        self.theta_zero = 0 #é o intercepto do modelo (o termo constante), que também é inicializado com 0.
        
        
        """
O modelo é treinado utilizando gradiente descendente, um algoritmo de otimização usado para minimizar a função de custo. 
O gradiente é calculado em cada iteração para ajustar os coeficientes de modo que o erro seja minimizado.

A cada iteração, o modelo faz uma previsão (y_pred) com os coeficientes theta e o intercept theta_zero.
A previsão y_pred é calculada como o produto escalar entre os dados X e os coeficientes theta, somando o intercept theta_zero.
        """

#3 COORDENADA DESCENDENTE
#Loop de treinamento que ajusta os coeficientes
#Calcula as previsões (y_pred) com base nos coeficientes atuais.         

        early_stopping_counter = 0 # Contador para o mecanismo de early stopping
        for iteration in range(self.max_iter): # Loop principal para ajustar os parâmetros

    #3.1    #calculo das previsões: ypred = 𝜃0 + media(𝜃𝑗 x Xj)
            y_pred = np.dot(X, self.theta) + self.theta_zero # Calcula as previsões baseadas no modelo atual

    #3.2    # Atualiza o theta_zero com base na média dos resíduos: media (y-ypred)
            self.theta_zero = np.mean(dataset.y - y_pred) #O intercepto é ajustado calculando a média da diferença entre as previsões e os valores reais (dataset.y).

    #3.3    #Atualização de cada coeficiente (𝜃𝑗)
            # Atualiza os coeficientes usando o gradiente descendente com regualarização L1. 
            for j in range(n): # Para cada coeficiente:
                X_feature = X[:, j] # Seleciona a característica j
    
    #3.3.1  #Cálculo do resíduo ajustado para 𝜃𝑗
                residual = dataset.y - (y_pred - X_feature * self.theta[j]) # Calcula o resíduo excluindo a contribuição de j
                gradient = np.dot(X_feature, residual) / m # Gradiente para o coeficiente j

    #3.3.2  #Atualização usando o soft thresholding
                 # Atualiza o coeficiente aplicando a regularização 𝐿1(penalização Lasso):
                self.theta[j] = self._apply_soft_threshold(gradient, self.l1_penalty) / (np.dot(X_feature, X_feature) / m)
            """
Para cada variável (ou coluna de X), é calculado o resíduo (a diferença entre os valores reais e as previsões, excluindo a contribuição da variável j).
O gradiente é a derivada do erro em relação ao coeficiente theta[j], e é calculado como o produto escalar entre a variável X_feature e o resíduo, normalizado pelo número de amostras m.
O coeficiente theta[j] é atualizado aplicando o soft thresholding (que faz a regularização L1) no gradiente, com a penalização L1 sendo controlada pelo parâmetro self.l1_penalty.
A regularização L1 ajuda a "zerar" coeficientes que são pequenos, forçando o modelo a ser mais simples e com menos variáveis ativas.
            """


#4 CALCULO DO CUSTO 
#O custo combina o erro quadrático médio e a penalização 𝐿1

            # Calcula o custo atual
            #O custo é uma combinação do erro quadrático médio e da penalização L1
            cost = self.cost(dataset)
            self.cost_history[iteration] = cost # Armazena o custo no histórico

#5 VERIFICAÇÃO EARLY STOPPING
#Objetivo: Parar o treinamento se o custo não melhorar por várias iterações consecutivas.
#Se o custo da iteração atual for maior ou igual ao da iteração anterior, incrementa-se um contador. Quando esse contador atinge o valor de patience, o treinamento é interrompido

            # Verifica se houve melhora no custo: early stopping
            # Early stopping baseado na ausência de melhoria no custo
            if iteration > 0 and cost >= self.cost_history[iteration - 1]: # Verifica se o custo não melhorou
                early_stopping_counter += 1 # Incrementa o contador
                if early_stopping_counter >= self.patience: # Interrompe se atingir o limite de patience
                    break
            else:
                early_stopping_counter = 0 # Reseta o contador se o custo melhorar
        """
Se o custo da iteração atual não for menor do que o custo da iteração anterior, o contador early_stopping_counter é incrementado. Se o número de iterações sem melhora (early_stopping_counter) atingir o valor de patience, o treinamento é interrompido.
Caso o custo melhore, o contador é resetado para 0.
        """
        return self  # Retorna o modelo ajustado

    def cost(self, dataset: Dataset) -> float:
        """
        Calcula o custo do modelo.

        Parâmetros:
        ----------
        dataset : Dataset
            Dados de entrada para cálculo do custo.

        Retorna:
        -------
        cost : float
            Valor do custo (erro quadrático médio + regularização L1).
        """
#1 GERAÇÃO DAS PREVISÕES
#Aqui, o modelo utiliza os coeficientes atuais (𝜃) e o intercepto (𝜃0) para gerar as previsões (𝑦pred) a partir dos dados 𝑋 fornecidos no conjunto dataset.
        y_pred = self.predict(dataset)  # Gera previsões para os dados
#A função predict calcula essa fórmula internamente, utilizando os coeficientes 𝜃 e 𝜃0


#2 CALCULO DO ERRO QUADRATICO MEDIO(MSE)
        mse_error = np.mean((dataset.y - y_pred) ** 2) # Calcula o erro quadrático médio
#Objetivo: Minimizar o MSE para que as previsões 𝑦pred fiquem o mais próximo possível dos valores reais 𝑦

#3. PENALIZAÇÃO L1
        l1_term = self.l1_penalty * np.sum(np.abs(self.theta)) # Calcula a penalização L1
#Objetivo: Adicionar um custo ao uso de coeficientes maiores


#4. CALCULO DO CUSTO TOTAL
        return mse_error / 2 + l1_term # Retorna o custo total


    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Faz previsões para novos dados.

        Parâmetros:
        ----------
        dataset : Dataset
            Dados de entrada.

        Retorna:
        -------
        predictions : np.ndarray
            Previsões geradas pelo modelo.
        """
#1 NORMALIZAÇÁO DOS DADOS
#Caso o parâmetro scale seja True, os dados são normalizados com base na média (self.mean) e no desvio padrão (self.std) calculados durante o treinamento.
#Caso scale seja False, os dados são usados como estão.
        X = (dataset.X - self.mean) / (self.std + 1e-8) if self.scale else dataset.X # Normaliza se necessário

#2 CALCULO DAS PREVISÕES
#X⋅θ: Produto escalar entre os dados normalizados (𝑋) e os coeficientes (𝜃).
#θ : Intercepto adicionado ao resultado.
        return np.dot(X, self.theta) + self.theta_zero  # Calcula as previsões


    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Avalia o modelo com base no erro médio quadrático (MSE).

        Parâmetros:
        ----------
        dataset : Dataset
            Conjunto de dados para avaliação.
        predictions : np.ndarray
            Previsões do modelo.

        Retorna:
        -------
        mse : float
            Erro médio quadrático.
        """

        return mse(dataset.y, predictions) # Retorna o MSE entre as previsões e os valores reais
#Entrada: Valores reais (𝑦) e previsões (𝑦pred)
#Saída: Erro médio quadrático (𝑀𝑆𝐸)


    def _apply_soft_threshold(self, value: float, penalty: float) -> float:
        """
        Aplica a técnica de soft thresholding para regularização L1.

        Parâmetros:
        ----------
        value : float
            Gradiente calculado para um coeficiente.
        penalty : float
            Penalização L1.

        Retorna:
        -------
        thresholded_value : float
            Valor ajustado após a regularização.
        """
        if abs(value) > penalty: # Verifica se o valor está fora do intervalo de penalização
            return np.sign(value) * (abs(value) - penalty)   # Aplica a redução pelo soft thresholding
        return 0.0 # Zera o valor se estiver dentro do intervalo de penalização
#Entrada: Valor do coeficiente (value) e a penalidade (penalty).
#Saída: Valor ajustado após aplicação do Soft Thresholding.