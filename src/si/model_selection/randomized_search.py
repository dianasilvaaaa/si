from typing import Dict, Tuple, Callable, Union
import numpy as np
import itertools
from si.data.dataset import Dataset
from si.model_selection.cross_validate import k_fold_cross_validation

def randomized_search_cv(model,
                         dataset: Dataset,
                         parameter_distribution: Dict[str, Tuple],
                         scoring: Callable = None,
                         cv: int = 3,
                         n_iter: int = 10,
                         test_size: float = 0.3) -> Dict[str, Tuple[str, Union[int, float]]]:
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
    for parameter in parameter_distribution:
        if not hasattr(model, parameter):
            raise AttributeError(f"O modelo {model} não possui o parâmetro {parameter}.")

    # Gera n_iter combinações aleatórias de parâmetros
    for _ in range(n_iter):
        random_state = np.random.randint(0, 1000)
        scores['seed'].append(random_state)

        # Seleciona os parâmetros de forma aleatória
        parameters = {param: np.random.choice(values) for param, values in parameter_distribution.items()}

        # Configura os parâmetros no modelo
        for param, value in parameters.items():
            setattr(model, param, value)

        # Realiza a validação cruzada e obtém as métricas
        score = k_fold_cross_validation(model=model, dataset=dataset, scoring=scoring, cv=cv)

        # Armazena os resultados
        scores['parameters'].append(parameters)
        scores['train'].append(score)
        scores['test'].append(score)

    return scores


def random_combinations(hyperparameter_grid: dict, n_iter: int) -> list:
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
    
    if n_iter > len(all_combinations):
        raise ValueError(f"n_iter ({n_iter}) não pode ser maior que o número total de combinações ({len(all_combinations)}).")

    # Seleciona aleatoriamente n_iter combinações
    selected_indices = np.random.choice(len(all_combinations), n_iter, replace=False)
    return [all_combinations[i] for i in selected_indices]


def randomized_search_cv_v2(model, dataset, hyperparameter_grid, scoring=None, cv=3, n_iter=10, test_size=0.3):
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
            setattr(model, param, value)

        # Realiza a validação cruzada K-fold
        cv_results = k_fold_cross_validation(model=model, dataset=dataset, scoring=scoring, cv=cv)

        # Calcula a pontuação média
        mean_score = np.mean(cv_results if isinstance(cv_results, list) else cv_results.get("test_scores", []))

        # Armazena os hiperparâmetros e a pontuação
        results['hyperparameters'].append(hyperparameters)
        results['scores'].append(mean_score)

        # Atualiza os melhores hiperparâmetros e a melhor pontuação
        if mean_score > results['best_score']:
            results['best_score'] = mean_score
            results['best_hyperparameters'] = hyperparameters

    return results

