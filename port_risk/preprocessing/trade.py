"""
Preprecesssing module to create a trade dataframe.
"""

import pandas as pd
import geopandas as gpd


def preprocess_trade(
    network: pd.DataFrame, ports: gpd.GeoDataFrame, iso3: str, industries: bool = False
) -> pd.DataFrame:
    """
    Creates a data frame containing trade flows for each country using the q_sea_flow and
    v_sea_flow features. Performs the computation on import/export flows depending on the input
    value of the iso3 parameter. Can perform the computation based on all trade or on each industry
    sector seperatly depending on the industries flag.

    Parameters:
        network: pd.DataFrame
            The maritime trade network dataframe
        ports: gpd.GeoDataFrame
            The port location and information
        iso3: str
            Possible values are: {"iso3_O", iso3_D}. Specifies if process is conducted on improts or
            exports.
        industries: bool, default False
            Boolean flag set to true if process should be done on each sector seperatly.

    Return:
        country_flows_total: pd.DataFrame
            A DataFrame object contaning the aggregated flows of trade associated to each country.
    """
    groupby_list = [iso3]
    if industries:
        groupby_list += ["Industries", "sector"]

    country_flows = (
        network.loc[network["flow"] == "port_import"]
        .groupby(groupby_list)[["q_sea_flow", "v_sea_flow"]]
        .sum()
    )
    country_flows = country_flows.rename(
        columns={"q_sea_flow": "q_sea_flow_total", "v_sea_flow": "v_sea_flow_total"}
    )
    country_flows = country_flows.reset_index()

    country_flows_total = network.groupby(
        [groupby_list[0]] + ["id"] + groupby_list[1:]
    )[["q_sea_flow", "v_sea_flow"]].sum()
    country_flows_total = country_flows_total.reset_index()

    country_flows_total = country_flows_total.merge(country_flows, on=groupby_list)

    country_flows_total = country_flows_total.merge(ports, on="id")
    country_flows_total = country_flows_total.rename(
        columns={iso3: "iso3", "iso3": "port_iso3"}
    )

    return country_flows_total


def create_trade_dataframe(
    network: pd.DataFrame, ports: gpd.GeoDataFrame
) -> dict[str, pd.DataFrame]:
    """
    Creates the trade dataframe by combining data from the maritime network and the ports risks.

    Parameters:
        network: pd.DataFrame
            The maritime trade network.
        ports: gpd.GeoDataFrame
            The ports and their associated downtime risk.

    Return:;
        trade: dict[str, pd.DataFrame]
            Dictionnary mapping trade type (import/export) to the associated trade dataframe.
    """
    trade = {}

    trade["export_trade"] = preprocess_trade(network, ports, "iso3_O", False)
    trade["import_trade"] = preprocess_trade(network, ports, "iso3_D", False)

    trade["export_trade_sector"] = preprocess_trade(network, ports, "iso3_O", True)
    trade["import_trade_sector"] = preprocess_trade(network, ports, "iso3_D", True)

    return trade
