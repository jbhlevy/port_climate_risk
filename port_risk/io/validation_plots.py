"""
Module to plot validation tests for hyperparameters tunning.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from port_risk import data_path


def make_depth_plots(
    models_stats: dict, folder: str = None, show: bool = False, save: bool = False
) -> None:
    """
    Make depth plots used to find optimal depths for forest and boosting methods.
    Parameters:
        models_stats: dict
            Dictionnary containing the best statistics associated to each model.
        folder: str, default None
            Folder where to save the file.
        show: bool default False:
           Boolean flag set to to show plot.
        save: bool default False:
           Boolean flag set to to save plot.

    Return:
        None
    """
    plt.figure(figsize=(10, 8))
    for name, optimal_depths in models_stats.items():
        for (
            metric,
            ((optimal_depth, optimal_value), (depths, values)),
        ) in optimal_depths.items():
            plt.plot(
                depths,
                values,
                label=f"{metric} (Optimal Depth: {optimal_depth})",
                marker="o",
            )
            plt.scatter(optimal_depth, optimal_value, color="black", zorder=5)
            plt.text(
                optimal_depth,
                optimal_value,
                f" {optimal_depth}",
                verticalalignment="bottom",
                horizontalalignment="right",
            )

        # Adding plot details
        plt.title(
            f"Model: {name} Performance vs. Tree Depth with Optimal Depths Highlighted"
        )
        plt.xlabel("Maximum Tree Depth")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        if show:
            plt.show()
        if save:
            if folder:
                folder += "/"
            else:
                folder = ""
            plt.savefig(Path(data_path["plots"], f"{folder}metrics_depths_{name}"))

        plt.close()


def make_oob_error_plots(
    models_errors: dict, folder=None, show: bool = False, save: bool = False
):
    """
    Makes the out-of-bag error plots for all models in models_erros dictionnary.

    Parameters:
        models_erros: dict
            Dictionnary contianing the models name, (estimators range, errors) and best values for
            each pair.
        folder: str, default None
            Folder where to save the file.
        show: bool default False:
           Boolean flag set to to show plot.
        save: bool default False:
           Boolean flag set to to save plot.

    Return:
        None
    """
    plt.figure(figsize=(10, 8))

    for name, (
        (estimators_range, errors),
        (optimal_estimator, optimal_value),
    ) in models_errors.items():
        plt.plot(
            estimators_range,
            errors,
            "-o",
            label=f"OOB Error {name} (Optimal Estimator: {optimal_estimator})",
        )
        plt.scatter(optimal_estimator, optimal_value, color="black", zorder=5)

        plt.xlabel("Number of Trees (n_estimators)")
        plt.ylabel("Out-of-bag classification error")
        plt.title("OOB Error Rate Across Different Numbers of Trees")
        plt.legend()
        plt.grid(True)

    if show:
        plt.show()

    if save:
        if folder:
            folder += "/"
        else:
            folder = ""
        plt.savefig(Path(data_path["plots"], "oob_erros"))
