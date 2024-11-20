import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
import itertools
from si.model_selection.cross_validate import k_fold_cross_validation

def randomized_search_cv(model: Model, dataset: Dataset, hyperparameter_grid: dict, scoring: callable = None, cv =3 , n_iter = 10)-> dict:

    """
Perform randomized search cross-validation for hyperparameter tuning.

    Parameters
    ----------
    model : Estimator
        Model to validate.
    dataset : Dataset
        Validation dataset.
    hyperparameter_grid : dict
        Dictionary with hyperparameter names as keys and lists of possible values as values.
    scoring : callable
        Scoring function to evaluate model performance.
    cv : int, optional
        Number of folds for cross-validation, by default 3.
    n_iter : int, optional
        Number of random hyperparameter combinations to test, by default 10.

    Returns
    -------
    dict
        Dictionary with the results of the randomized search cross-validation. Includes:
        - 'hyperparameters': list of hyperparameter combinations tested.
        - 'scores': list of mean scores obtained for each combination.
        - 'best_hyperparameters': the combination of hyperparameters with the best score.
        - 'best_score': the best score obtained.


"""

    # Validate hyperparameter grid keys
    for param in hyperparameter_grid.keys():
        if not hasattr(model, param):
            raise ValueError(f"The model does not have the parameter '{param}'.")

    # Generate random combinations of hyperparameters
    param_names = list(hyperparameter_grid.keys())
    param_values = list(hyperparameter_grid.values())
    all_combinations = [
        dict(zip(param_names, combo))
        for combo in np.array(np.meshgrid(*param_values)).T.reshape(-1, len(param_names))
    ]
    
    if n_iter > len(all_combinations):
        raise ValueError("n_iter cannot be greater than the total number of hyperparameter combinations.")
    
    random_combinations = np.random.choice(all_combinations, size=n_iter, replace=False)
    
    # Initialize results
    results = {
        'hyperparameters': [],
        'scores': [],
        'best_hyperparameters': None,
        'best_score': -np.inf,
    }

    # Perform randomized search
    for combination in random_combinations:
        # Set model hyperparameters
        for param, value in combination.items():
            setattr(model, param, value)

        # Perform cross-validation
        cv_results = k_fold_cross_validation(model, dataset, scoring=scoring, cv=cv)
        mean_score = np.mean(cv_results['test_scores'])

        # Save results
        results['hyperparameters'].append(combination)
        results['scores'].append(mean_score)

        # Update best score and hyperparameters if needed
        if mean_score > results['best_score']:
            results['best_score'] = mean_score
            results['best_hyperparameters'] = combination

    return results