""""
Statistics data evaluation module.
"""

import pandas as pd
import scipy.stats as stats
from port_risk.io.stats_plots import plot_qq_plot


def compute_statistics(data: pd.Series, tested_functions: dict, f_name: str) -> dict:
    """
    Computes the statistical hypothesis Kolmogorov-Smirnow test for the given data and distributions
    to test and plots the corresponding Q-Q plot.

    Parameters:
        data: pd.Series
            The data for which we seek the best fitting distribution.
        tested_functions: dict
            A dictionnary containing the distributions to test and their name.

    Return:
        statistics: dict
            A dictionnary mapping the name of each distribution to the results of the test.
    """
    statistics = {}
    for name, func in tested_functions.items():
        params = func.fit(data)
        if len(params) == 2:
            loc, scale = params
            dist = func(loc=loc, scale=scale)
            cdf = dist.cdf
        elif len(params) == 3:
            a, loc, scale = params
            dist = func(a, loc=loc, scale=scale)
            cdf = dist.cdf
        else:
            a, b, loc, scale = params
            cdf = func(a, b, loc=loc, scale=scale).cdf
        result = stats.ks_1samp(data, cdf)
        statistics[name] = (result.statistic, result.pvalue)
        plot_qq_plot(data, f"{f_name}_{name}", dist, save=True)
    return statistics
