"""
Machine learning models.
"""

import numpy as np


def find_optimal_params(optimal_depths: dict, oob_errors: dict) -> dict:
    """
    Retrieve the optimal parameters computed during the validation step to use them for model
    fitting.

    Parameters:
        optimal_depths: dict
            Mapping of model name to computed depths and metrics
        oob_errors: dict
            Mapping of model name to computed estimators and metrics

    Return:
        params: dict
            Mapping of model name to the depth and estimators that should be used.

    """
    params = {}
    for (name, opt_depths), (
        _,
        (
            (_, _),
            (optimal_estimator, _),
        ),
    ) in zip(optimal_depths.items(), oob_errors.items()):
        params[name] = {}
        optimal_depth = opt_depths["mse"][0][0]
        params[name] = (optimal_depth, optimal_estimator)
    return params


def run_all_models(
    models: dict,
    x_train: np.array,
    x_test: np.array,
    y_train: np.array,
    y_test: np.array,
    params=None,
) -> dict:
    """
    Runs all the models in the input model dictionnary.

    Parameters:
        models: dict
            Dictionnary of model name and model class to use.
        x_train: np.array
            Training feature data.
        x_test: np.array
            Testing feature data.
        y_train: np.array
            Training prediction data
        y_test: np.array
            Testing prediciton data
        models: dict
            Dictionnary of model name and model class to use.
        params: bool default None
            Parameters to pass to the model.

    Return:
        ran_models: dict
            Dictionnary containing the instantiated model objects that have been ran.
    """

    ran_models = {}
    for name, model in models.items():
        if params is not None:
            try:
                p = params[name]
                print(params)
                print(f"Running {name}")
                _model = model(
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    name,
                    depth=p[0],
                    n_estimators=p[1],
                )
                _model.run_model()
                ran_models[name] = _model
            except KeyError:
                print(f"Running {name}")
                _model = model(x_train, y_train, x_test, y_test, name)
                _model.run_model()
                ran_models[name] = _model
        else:
            print(f"Running {name}")
            _model = model(x_train, y_train, x_test, y_test, name)
            _model.run_model()
            ran_models[name] = _model
    return ran_models
