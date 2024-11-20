import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
import itertools
#from si.model_selection.cross_validate import k_fold_cross_validation

def randomized_search_cv(
    model: Model,
    dataset: Dataset,
    hyperparameter_grid: dict,
    cv: int,
    n_iter: int,
    scoring: callable = None,
) -> dict:
    """
    Performs randomized grid search.

    Randomized Grid Search is a hyperparameter tuning technique that explores a specified
    parameter space by randomly sampling combinations of hyperparameter values.

    Parameters:
    ----------
    model : Model
        The model to perform the hyperparameter tuning.
    dataset : Dataset
        The dataset to use for validation.
    hyperparameter_grid : dict
        Dictionary with hyperparameter names and search values.
    scoring : callable
        Scoring function to evaluate the model's performance during the hyperparameter tuning.
    cv : int
        Number of folds for cross-validation.
    n_iter : int
        Number of random hyperparameter combinations to search.

    Returns:
    ----------
    results : dict
        Dictionary with the results of the grid search cross-validation. Includes:
        - 'hyperparameters': List of hyperparameter combinations tested.
        - 'scores': List of mean scores obtained for each combination.
        - 'best_hyperparameters': Best combination of hyperparameters.
        - 'best_score': Best score obtained.
"""
    # Validate hyperparameter existence in the model
    for parameter in hyperparameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} does not have parameter '{parameter}'.")

    # Generate random combinations
    combinations = random_combinations(hyperparameter_grid, n_iter)

    # Initialize results
    results = {
        "scores": [],
        "hyperparameters": [],
        "best_hyperparameters": None,
        "best_score": -np.inf,
    }

    # Perform randomized search
    for combination in combinations:
        # Set model hyperparameters
        parameters = {}
        for parameter, value in zip(hyperparameter_grid.keys(), combination):
            setattr(model, parameter, value)
            parameters[parameter] = value

        # Perform cross-validation
        cv_results = k_fold_cross_validation(model=model, dataset=dataset, scoring=scoring, cv=cv)
        mean_score = np.mean(cv_results["test_scores"])

        # Save results
        results["scores"].append(mean_score)
        results["hyperparameters"].append(parameters)

        # Update best score and hyperparameters if necessary
        if mean_score > results["best_score"]:
            results["best_score"] = mean_score
            results["best_hyperparameters"] = parameters

    return results


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
    # Compute all possible combinations of hyperparameters
    all_combinations = list(itertools.product(*hyperparameter_grid.values()))

    # Validate n_iter
    if n_iter > len(all_combinations):
        raise ValueError(
            f"n_iter ({n_iter}) cannot exceed the number of total combinations ({len(all_combinations)})."
        )

    # Select random combinations
    random_indices = np.random.choice(len(all_combinations), n_iter, replace=False)
    return [all_combinations[idx] for idx in random_indices]
