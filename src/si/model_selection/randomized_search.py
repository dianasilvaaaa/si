from typing import Dict, Tuple, Callable, Union
import numpy as np
import itertools
from si.data.dataset import Dataset
from si.model_selection.cross_validate import k_fold_cross_validation

def randomized_search_cv(model, # modelo a ser ajustado
                         dataset: Dataset, #Conjunto de dados utilizado para validação
                         parameter_distribution: Dict[str, Tuple], #Dicionário com nomes dos hiperparâmetros e seus valores possíveis.
                         scoring: Callable = None, #Função que calcula a métrica de avaliação
                         cv: int = 3, #Número de divisões (folds) para validação cruzada
                         n_iter: int = 10, #Número de combinações aleatórias a serem testadas
                         test_size: float = 0.3) -> Dict[str, Tuple[str, Union[int, float]]]:
                        # test_size: Proporção dos dados reservados para teste
                        # Retorno: Um dicionário com os resultados da busca aleatória

    """                 
    Realiza uma busca aleatória sobre os hiperparâmetros e avalia o desempenho do modelo.

    Parâmetros:
    ----------
    model: Modelo a ser validado.
    dataset: Conjunto de dados para validação.
    parameter_distribution: Dicionário com os nomes dos hiperparâmetros e seus possíveis valores.
    scoring: Função para avaliar o modelo.
    cv: Número de divisões (folds) na validação cruzada.
    n_iter: Número de combinações aleatórias de hiperparâmetros a serem testadas.
    test_size: Tamanho do conjunto de teste.

    Retorno:
    ----------
    Um dicionário com os resultados da busca aleatória e validação cruzada.
    """
    scores = {'parameters': [], 'seed': [], 'train': [], 'test': []}

    # Verifica se os parâmetros existem no modelo
    for parameter in parameter_distribution: #Verifica se os parâmetros no parameter_distribution existem no modelo (hasattr).
        if not hasattr(model, parameter):
            raise AttributeError(f"O modelo {model} não possui o parâmetro {parameter}.") #Erro: Caso algum parâmetro seja inválido, uma exceção é levantada.

    # Gera n_iter combinações aleatórias de parâmetros
    for _ in range(n_iter):
        random_state = np.random.randint(0, 1000)
        scores['seed'].append(random_state) #Gera um random_state (semente aleatória) para cada iteração e armazena no dicionário de scores.

        # Seleciona os parâmetros de forma aleatória
        parameters = {param: np.random.choice(values) for param, values in parameter_distribution.items()} #Seleciona valores aleatórios para cada hiperparâmetro dentro das distribuições especificadas.

        # Configura os parâmetros no modelo
        for param, value in parameters.items():
            setattr(model, param, value) #Define os hiperparâmetros no modelo com setattr.

        # Realiza a validação cruzada e obtém as métricas
        score = k_fold_cross_validation(model=model, dataset=dataset, scoring=scoring, cv=cv) #Realiza validação cruzada com os hiperparâmetros definidos.

        # Armazena os resultados. 
        #Armazena os parâmetros e as métricas obtidas na validação cruzada.
        scores['parameters'].append(parameters)
        scores['train'].append(score)
        scores['test'].append(score)

    return scores


def random_combinations(hyperparameter_grid: dict, n_iter: int) -> list:

    #Gera todas as combinações possíveis de hiperparâmetros utilizando o produto cartesiano (itertools.product).
    """
    Seleciona combinações aleatórias de hiperparâmetros.

    Parâmetros:
    ----------
    hyperparameter_grid: dict
        Dicionário com os nomes dos hiperparâmetros e os valores de busca.
    n_iter: int
        Número de combinações a serem selecionadas aleatoriamente.

    Retorno:
    ----------
    random_combinations: list
        Lista de combinações aleatórias de hiperparâmetros.
    """
    all_combinations = list(itertools.product(*hyperparameter_grid.values()))
    
    #Valida se o número de combinações aleatórias solicitadas (n_iter) é menor ou igual ao número total de combinações possíveis.
    if n_iter > len(all_combinations):
        raise ValueError(f"n_iter ({n_iter}) não pode ser maior que o número total de combinações ({len(all_combinations)}).")


    # Seleciona aleatoriamente n_iter combinações
    #Seleciona n_iter combinações aleatórias de índices e retorna as combinações correspondentes.
    selected_indices = np.random.choice(len(all_combinations), n_iter, replace=False)
    return [all_combinations[i] for i in selected_indices]


def randomized_search_cv_v2(model, dataset, hyperparameter_grid, scoring=None, cv=3, n_iter=10, test_size=0.3):

    #Variante da função randomized_search_cv: Usa a função random_combinations para selecionar combinações de hiperparâmetros.
    #Calcula a média das pontuações obtidas na validação cruzada.
    """
    Realiza uma busca aleatória de hiperparâmetros utilizando validação cruzada, adaptada para busca baseada em grid.

    Parâmetros:
    ----------
    model : Modelo
        O modelo a ser ajustado.
    dataset : Dataset
        O conjunto de dados para validação cruzada.
    hyperparameter_grid : dict
        Um dicionário contendo os hiperparâmetros e seus valores possíveis.
    scoring : callable, opcional
        Uma função para avaliar o desempenho do modelo.
    cv : int
        Número de divisões (folds) na validação cruzada.
    n_iter : int
        Número de combinações aleatórias de hiperparâmetros a serem testadas.

    Retorno:
    ----------
    results : dict
        Dicionário com os resultados da busca aleatória.
    """
    # Verifica se o modelo possui os parâmetros especificados
    for parameter in hyperparameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f"O modelo {model} não possui o parâmetro '{parameter}'.")

    # Gera combinações aleatórias de hiperparâmetros
    combinations = random_combinations(hyperparameter_grid, n_iter)

    # Inicializa o dicionário de resultados
    results = {
        "scores": [],
        "hyperparameters": [],
        "best_hyperparameters": None,
        "best_score": -np.inf,
    }

    # Itera sobre as combinações aleatórias
    for combination in combinations:
        hyperparameters = dict(zip(hyperparameter_grid.keys(), combination))

        # Configura os hiperparâmetros no modelo
        for param, value in hyperparameters.items():
            setattr(model, param, value) #a função setattr(object, name, value) é uma função que recebe um objeto, define ou codifica um atributo name desse objeto e atribui a esse objeto um valor value
        #Um dicionário onde cada chave (param) é o nome de um hiperparâmetro e cada valor (value) é o valor correspondente para aquele hiperparâmetro.
        #setattr: Atualiza o modelo, atribuindo o valor value ao hiperparâmetro param

        # Realiza a validação cruzada K-fold
        cv_results = k_fold_cross_validation(model=model, dataset=dataset, scoring=scoring, cv=cv)

        # Calcula a pontuação média
        #Calcula a pontuação média sobre os folds da validação cruzada.       
        mean_score = np.mean(cv_results if isinstance(cv_results, list) else cv_results.get("test_scores", []))

        # Armazena os hiperparâmetros e a pontuação
        results['hyperparameters'].append(hyperparameters)
        results['scores'].append(mean_score)

        # Atualiza os melhores hiperparâmetros e a melhor pontuação
        #Atualiza o melhor score e os melhores hiperparâmetros encontrados.
        if mean_score > results['best_score']:
            results['best_score'] = mean_score
            results['best_hyperparameters'] = hyperparameters

    return results

