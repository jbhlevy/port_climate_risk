"""
IO Module for loading the data
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
from port_risk import data_path


def load_risk_data() -> pd.DataFrame:
    """
    Loads the climate risk data and applies some simple renaming steps for later use.

    Parameters:
        None

    Returns:
        df: pd.DataFrame
            The loaded data.
    """
    path = Path(data_path["climate_data"])
    df = pd.read_csv(path)

    df = df.rename(columns={"port_name": "name"})
    df["port_name"] = df["name"] + "_" + df["country"]
    df = df.drop(["country", "name"], axis=1)

    return df


def load_ports_data() -> gpd.GeoDataFrame:
    """
    Loads the ports data and applies some simple preproecessing steps to preprare for later merging.

    Parameters:
        None

    Returns:
        df: gpd.GeoDataFrame
            The loaded data.
    """
    path = Path(data_path["ports"])
    df = gpd.read_file(path)
    df = df.loc[df["infra"] == "port"].reset_index(drop=True)
    df = df.rename(columns={"name": "port_name"})
    df = df.drop(["infra", "Continent_Code"], axis=1)

    return df


def load_industries_data() -> pd.DataFrame:
    """
    Loads the industry <-> sector mapping.

    Parameters:
        None

    Returns:
        df: pd.DataFrame
            The loaded data.
    """
    path = Path(data_path["industries"])
    df = pd.read_csv(path)
    return df


def load_maritime_network() -> pd.DataFrame:
    """
    Loads the maritime trade network.

    Parameters:
        None

    Returns:
        df: pd.DataFrame
            The loaded data.
    """
    path = Path(data_path["network"])
    df = pd.read_csv(path)

    return df


def load_countries() -> gpd.GeoDataFrame:
    """
    Loads a geopandas dataframe with all countries studied for plotting.

    Parameters:
        None

    Returns:
        df: gpd.GeoDataFrame
            The loaded countries frame
    """
    path = Path(data_path["countries"])
    df = gpd.read_file(path)
    df = df.rename(columns={"ISO_A3": "iso3"})
    return df


def load_world() -> gpd.GeoDataFrame:
    """
    Loads the geopandas world map dataset for plotting.

    Paramaters:
        None

    Return:
        df: gpd.GeoDataFrame
            The loaded world map.
    """
    path = Path(gpd.datasets.get_path("naturalearth_lowres"))
    df = gpd.read_file(path)
    return df


def load_optimal_params() -> dict:
    return 0
