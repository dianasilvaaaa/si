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
    Perform a randomized search over hyperparameters and evaluate model performance.

    param model: Model to validate
    param dataset: Validation dataset
    param parameter_distribution: Dictionary with hyperparameter names and their possible values
    param scoring: Scoring function
    param cv: Number of folds
    param n_iter: Number of random hyperparameter combinations to test
    param test_size: Test set size

    return: Dictionary with the results of the randomized search cross-validation.
    """
    scores = {'parameters': [], 'seed': [], 'train': [], 'test': []}

    # Check if parameters exist in the model
    for parameter in parameter_distribution:
        if not hasattr(model, parameter):
            raise AttributeError(f"The model {model} does not have parameter {parameter}.")

    # Generate n_iter random combinations of parameters
    for _ in range(n_iter):
        random_state = np.random.randint(0, 1000)
        scores['seed'].append(random_state)

        # Randomly select parameters
        parameters = {param: np.random.choice(values) for param, values in parameter_distribution.items()}

        # Set the model's parameters
        for param, value in parameters.items():
            setattr(model, param, value)

        # Perform cross-validation and get the scores
        score = k_fold_cross_validation(model=model, dataset=dataset, scoring=scoring, cv=cv)

        # Store the results
        scores['parameters'].append(parameters)
        scores['train'].append(score)
        scores['test'].append(score)

    return scores


def random_combinations(hyperparameter_grid: dict, n_iter: int) -> list:
    """
    Select random combinations of hyperparameters.

    Parameters:
    ----------
    hyperparameter_grid: dict
        Dictionary with hyperparameter names and search values.
    n_iter: int
        Number of combinations to randomly select.

    Returns:
    ----------
    random_combinations: list
        List of randomly selected combinations of hyperparameters.
    """
    all_combinations = list(itertools.product(*hyperparameter_grid.values()))
    
    if n_iter > len(all_combinations):
        raise ValueError(f"n_iter ({n_iter}) cannot be greater than the total number of combinations ({len(all_combinations)}).")

    # Randomly select n_iter combinations
    selected_indices = np.random.choice(len(all_combinations), n_iter, replace=False)
    return [all_combinations[i] for i in selected_indices]


def randomized_search_cv_v2(model, dataset, hyperparameter_grid, scoring=None, cv=3, n_iter=10, test_size=0.3):
    """
    Perform randomized hyperparameter search using cross-validation, adapted for grid-based search.
    
    Parameters:
    ----------
    model : Model
        The model to be tuned.
    dataset : Dataset
        The dataset for cross-validation.
    hyperparameter_grid : dict
        A dictionary containing the hyperparameters and their search space.
    scoring : callable, optional
        A function for scoring the model's performance.
    cv : int
        Number of cross-validation folds.
    n_iter : int
        Number of random hyperparameter combinations to test.
    
    Returns:
    ----------
    results : dict
        Dictionary with results from the randomized search.
    """
    # Check if the model has the parameters specified
    for parameter in hyperparameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f"The model {model} does not have the parameter '{parameter}'.")

    # Generate random combinations of hyperparameters
    combinations = random_combinations(hyperparameter_grid, n_iter)

    # Initialize results dictionary
    results = {
        "scores": [],
        "hyperparameters": [],
        "best_hyperparameters": None,
        "best_score": -np.inf,
    }

    # Iterate over random combinations
    for combination in combinations:
        hyperparameters = dict(zip(hyperparameter_grid.keys(), combination))

        # Set the hyperparameters in the model
        for param, value in hyperparameters.items():
            setattr(model, param, value)

        # Perform k-fold cross-validation
        cv_results = k_fold_cross_validation(model=model, dataset=dataset, scoring=scoring, cv=cv)

        # Calculate mean score
        mean_score = np.mean(cv_results if isinstance(cv_results, list) else cv_results.get("test_scores", []))

        # Store the hyperparameters and score
        results['hyperparameters'].append(hyperparameters)
        results['scores'].append(mean_score)

        # Update best score and hyperparameters
        if mean_score > results['best_score']:
            results['best_score'] = mean_score
            results['best_hyperparameters'] = hyperparameters

    return results
