"""
Pre-processing module for the machine learning model.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from port_risk.io.stats_plots import plot_correlation_matrix


def ml_preprocessing(data: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Applies pre-processing steps to the data before feeding it to a learning model.
    Computes the data correlation_matrix and saves it.

    Parameters:
        data: pd.DataFrame
            The input data to preprocess.
        name: str
            The name of the dataset.

    Return:
        data: pd.DataFrame
            The ppreprocessed modified dataframe.
    """
    data = data.drop(["geometry", "iso3", "port_name", "port_iso3"], axis=1)
    data["id"] = data["id"].apply(lambda x: x[4:])

    data = data[(data["downtime_q_weighted"] != 0) & (data["downtime_v_weighted"] != 0)]
    # print(data.describe())
    data = data.loc[data["downtime_q_weighted"] < np.power(10, float(-4))]
    corr_matrix = data.corr()
    plot_correlation_matrix(corr_matrix, name, save=True)
    return data


def get_data_model_ready(data: pd.DataFrame, feature_to_predict: str) -> tuple:
    """
    Converts the dataframe into feature and feature to predict arrays and
    splits the dataset into train and testing sets.

    Parameters:
        data: pd.DataFrame
            The data that will be used to train the model.
        feature_to_predict: str
            The feature to predict

    Return:
        X_train, X_test, y_train, y_test: tuple
            The training/testing X and y arrays.
    """
    y = data[feature_to_predict]
    X = data.drop(feature_to_predict, axis=1).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    return X_train, X_test, y_train, y_test


def make_validation_data(x_train, y_train) -> tuple:

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )
    return x_train, x_val, y_train, y_val
