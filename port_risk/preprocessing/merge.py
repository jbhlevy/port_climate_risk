""""
Preprocessing module for merging data.
"""

import numpy as np
import pandas as pd
import geopandas as gpd


def merge_ports_risk(risks: pd.DataFrame, ports: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Merges the climate risk data frame to the ports data frame so that each port
    is associated with its natural disaster caused downtime.

    Parameters:
        risk: pd.DataFrame
            The climate risk dataframe.
        ports: gpd.GeoDataFrame
            The ports geopandas dataframe.

    Returns:
        ports_risk: pd.DataFrame
            The combined data.
    """
    port_risks = risks.merge(ports, on="port_name")

    port_risks = (
        port_risks.set_index(["id", "iso3", "geometry", "port_name", "hazard"])
        .unstack(level="hazard")
        .replace(np.nan, 0)
    )
    port_risks.columns = [
        "downtime_TC",
        "downtime_coastal",
        "downtime_earthquake",
        "downtime_fluvial",
        "downtime_operational",
        "downtime_pluvial",
    ]
    port_risks["downtime_total"] = port_risks.sum(axis=1)
    port_risks = port_risks.reset_index()

    return port_risks


def merge_network_industries(
    network: pd.DataFrame, industries: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges the maritime network with the different sectors available to perform specific analysis
    if required.

    Parameters:
        network: pd.DataFrame
            The maritime tarde network.
        industries: pd.DataFrame
            The industries hot-encoding to sector name mapping.

    Return:
        network: pd.DataFrame
            The combined data.
    """

    network = network.merge(industries, on="Industries")
    return network
