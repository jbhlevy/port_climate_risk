"""
Module for models.
"""

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)
from sklearn.tree import DecisionTreeRegressor
import numpy as np


class Model:
    def __init__(self, x_train, y_train, x_test, y_test, name) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = None
        self.name = name
        self.y_pred = None
        self.metrics = {}
        print(f"============= Instantiated {self.name} model =============")

    def fit(self):
        self.model.fit(self.x_train, self.y_train)

    def fit_and_predict(self) -> np.array:
        self.model.fit(self.x_train, self.y_train)
        self.y_pred = self.model.predict(self.x_test)

    def compute_metrics(self) -> None:
        self.metrics["mse"] = mean_squared_error(self.y_test, self.y_pred)
        self.metrics["mae"] = mean_absolute_error(self.y_test, self.y_pred)
        self.metrics["r2"] = r2_score(self.y_test, self.y_pred)
        self.metrics["rmse"] = root_mean_squared_error(self.y_test, self.y_pred)
        self.metrics["mape"] = mean_absolute_percentage_error(self.y_test, self.y_pred)

    def print_metrics(self) -> None:
        print("======= Key Metrics =======")
        for k, v in self.metrics.items():
            print(f"{k}: {v}")
        print("===========================\n")

    def run_model(self) -> None:
        self.fit_and_predict()
        self.compute_metrics()


# --------------------------------------------------------------------------------------------------

models = {}


class SimpleRegression(Model):
    def __init__(self, x_train, y_train, x_test, y_test, name) -> None:
        super().__init__(x_train, y_train, x_test, y_test, name)
        self.model = LinearRegression()


class SimpleDecisionTree(Model):
    def __init__(self, x_train, y_train, x_test, y_test, name) -> None:
        super().__init__(x_train, y_train, x_test, y_test, name)
        self.model = DecisionTreeRegressor()


class ExtraTrees(Model):
    def __init__(
        self,
        x_train,
        y_train,
        x_test,
        y_test,
        name,
        depth=18,
        oob_score=False,
        n_estimators=68,
        bootstrap=False,
    ) -> None:
        super().__init__(x_train, y_train, x_test, y_test, name)
        self.depth = depth
        self.model = ExtraTreesRegressor(
            max_depth=self.depth,
            oob_score=oob_score,
            n_estimators=n_estimators,
            bootstrap=bootstrap,
        )


class RandomForest(Model):
    def __init__(
        self,
        x_train,
        y_train,
        x_test,
        y_test,
        name,
        depth=18,
        oob_score=False,
        n_estimators=80,
        bootstrap=False,
    ) -> None:
        super().__init__(x_train, y_train, x_test, y_test, name)
        self.depth = depth
        self.model = RandomForestRegressor(
            max_depth=self.depth,
            oob_score=oob_score,
            n_estimators=n_estimators,
            bootstrap=bootstrap,
        )


class GradientBoost(Model):
    def __init__(
        self,
        x_train,
        y_train,
        x_test,
        y_test,
        name,
        depth=18,
        n_estimators=50,
    ) -> None:
        super().__init__(x_train, y_train, x_test, y_test, name)
        self.model = GradientBoostingRegressor(
            max_depth=depth, n_estimators=n_estimators
        )


models.update(
    {
        "linear_regression": SimpleRegression,
        "decison_tree": SimpleDecisionTree,
        "extra_trees": ExtraTrees,
        "random_forest": RandomForest,
        "gradient_boos": GradientBoost,
    }
)

# ----------------------------------------------------------------


def optimal_depth_trees(
    x_train: np.array,
    x_test: np.array,
    y_train: np.array,
    y_test: np.array,
    models_to_test: dict,
    max_depth_tested: int = 26,
) -> dict:
    """
    Find the optimal depth for tree models.

    Parameters:
        x_train: np.array
            Training feature data.
        x_test: np.array
            Testing feature data.
        y_train: np.array
            Training prediction data
        y_test: np.array
            Testing prediciton data
        max_depth_tested: int default 26
            Limit on the max depth that will be tested.

    Return:
        optimal_depths: dict
            Dictionnary mapping each model to the optimal parameters.
    """
    optimal_depths = {}
    depth_models = models_to_test.copy()

    for name, model in depth_models.items():
        print(f"Computing metrics for {name}")
        metrics = {}
        for depth in range(1, max_depth_tested):
            print(f"Depth = {depth}", end=" ")
            tree_model = model(x_train, y_train, x_test, y_test, name, depth)
            tree_model.run_model()
            metrics[depth] = tree_model.metrics
            optimal_depths[name] = {}
        print(f"Finished computing metrics for {name}")
        print("=====================================")
        metrics_scores = {}

        for key, sub_dict in metrics.items():
            for sub_key, value in sub_dict.items():
                if sub_key not in metrics_scores:
                    metrics_scores[sub_key] = {}
                metrics_scores[sub_key][key] = value

        for metric, scores in metrics_scores.items():
            optimal_depths[name][metric] = {}
            print(f"Finding optimal depth for {metric}")
            depths = list(scores.keys())
            values = list(scores.values())

            # For R2, we seek the maximum, for others, the minimum
            if metric == "r2":
                optimal_index = np.argmax(values)
            else:
                optimal_index = np.argmin(values)

            optimal_depth = depths[optimal_index]
            optimal_value = values[optimal_index]
            optimal_depths[name][metric] = (optimal_depth, optimal_value), (
                depths,
                values,
            )
        print(f"======== Finished assesing {name} performance ========")
        print()
    print("Found all optimal depths sucessfully!")
    return optimal_depths


def oob_erros_trees(
    x_train: np.array,
    x_test: np.array,
    y_train: np.array,
    y_test: np.array,
    models_to_test: dict,
    max_estimators: int = 101,
):
    """
    Compute out-of-bag error.

    Parameters:
        x_train: np.array
            Training feature data.
        x_test: np.array
            Testing feature data.
        y_train: np.array
            Training prediction data
        y_test: np.array
            Testing prediciton data
        max_estimators: int default 100
            Limit on the max estimators that will be tested.

    Return:
        optimal_depths: dict
            Dictionnary mapping each model to the computed errors and optimal parameters.
    """
    oob_errors = {}
    for name, model in models_to_test.items():
        if name == "gradient_boost":
            continue
        oob_errors[name] = ()
        errors = []
        print(f"Computing oob error for {name}")
        for n_estimators in range(18, max_estimators):
            print(f"Estimators = {n_estimators}", end="")
            tree_model = model(
                x_train,
                y_train,
                x_test,
                y_test,
                name,
                oob_score=True,
                n_estimators=n_estimators,
                bootstrap=True,
            )
            tree_model.fit()
            oob_error = 1 - tree_model.model.oob_score_
            errors.append(oob_error)
        optimal_estimator = np.argmin(np.array(errors))
        print(optimal_estimator)
        print(len(errors))
        optimal_value = errors[optimal_estimator]
        print(optimal_value)
        oob_errors[name] = (range(18, max_estimators), errors), (
            optimal_estimator + 18,
            optimal_value,
        )
        print("Computed all oob errors sucessfully!")
    return oob_errors
