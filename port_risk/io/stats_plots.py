"""
Module to plot statistical data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
from pathlib import Path

from port_risk import data_path


def plot_histogram(data: pd.Series, show: bool = False, save: bool = False) -> None:
    """
    Plots the histogram of the input data.

    Parameters:
        data: pd.Series
            The data to plot.
        show: bool default False
            Boolean flag set True to display to show the plot.
        save: bool default False
            Boolean flag set to True to save the plot.

    Return:
        None
    """
    plt.figure(figsize=(8, 12))
    bins = np.arange(0, 10, 0.1)
    data.hist(bins=bins)
    if show:
        plt.show()
    if save:
        path = ""
        plt.savefig(path)
    plt.close()


def plot_qq_plot(
    data: pd.Series,
    name: str,
    func: stats.distributions,
    show: bool = False,
    save: bool = False,
) -> None:
    """
    Plots the Q-Q plot of the input data.

    Parameters:
        data: pd.Series
            The data to plot.
        name: str
            The name of the distribution to compare to.
        func: stats.distributions
            The distribution to compare to.
        show: bool default False
            Boolean flag set True to display to show the plot.
        save: bool default False
            Boolean flag set to True to save the plot.

    Return:
        None
    """
    sm.qqplot(data, dist=func, line="45")
    plt.title(name)

    if show:
        plt.show()
    if save:
        path = Path(data_path["plots"], f"qq_plot_{name}")
        plt.savefig(path)
    plt.close()


def plot_correlation_matrix(
    corr_matrix: np.array, name: str, show: bool = False, save: bool = False
) -> None:
    """
    Plots the input correlation matrix

    Parameters:
        corr_matrix: np.array
            The matrix to plot.
        name: str
            The name of the dataset.
        show: bool default False
            Boolean flag set True to display to show the plot.
        save: bool default False
            Boolean flag set to True to save the plot.
    Return:
        None
    """
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr_matrix, annot=True)
    if show:
        plt.show()
    if save:
        path = Path(data_path["plots"], f"corr_matrix_{name}")
        plt.savefig(path)
    plt.close()
