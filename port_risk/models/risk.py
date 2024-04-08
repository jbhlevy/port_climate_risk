"""
Models submodule to compute risk
"""

import numpy as np
import pandas as pd


def compute_trade_risk(
    trade: pd.DataFrame,
    risk: pd.DataFrame,
    hazard: str = "downtime_total",
    industries: bool = False,
) -> pd.DataFrame:
    """
    Computes the trade risk by applying the downtime penalty corresponding to the hazard specified
    as argument to the trade flows.
    Performs additional steps to compute a downtime weighted by the amount of trade flow.
    Can perform the computation based on all trade or on each industry sector seperatly depending on
    the industries flag.

    Parameters:
        trade: pd.DataFrame
            The trade dataframe.
        risk:
            The port_risk data.
        hazard: str default 'downtime_total'
            The hazard to consider, default to all hazards.
        industries: bool default False
            Boolean flag set to true if process should be done on each sector seperatly.

    Return:
        trade_risk: pd.DataFrame
            A DataFrame object containing trade network and additional column containing the
            risk associated to each trade flow.

    """
    trade_risk = trade.merge(risk[["id", hazard]], on="id")
    trade_risk["q_flow_at_risk"] = (trade_risk["q_sea_flow"] / 365) * trade_risk[hazard]
    trade_risk["v_flow_at_risk"] = (trade_risk["v_sea_flow"] / 365) * trade_risk[hazard]

    trade_risk["downtime_q_weighted"] = (
        trade_risk["q_sea_flow"] / trade_risk["q_sea_flow_total"]
    ) * trade_risk[hazard]
    trade_risk["downtime_v_weighted"] = (
        trade_risk["v_sea_flow"] / trade_risk["v_sea_flow_total"]
    ) * trade_risk[hazard]

    groupby_list = ["iso3"]
    if industries:
        groupby_list += ["sector", "Industries"]

    network_trade_risk = (
        trade_risk.groupby(groupby_list)[["downtime_q_weighted", "downtime_v_weighted"]]
        .sum()
        .reset_index()
    )
    network_trade_risk = network_trade_risk.rename(
        columns={
            "downtime_q_weighted": "downtime_q_network",
            "downtime_v_weighted": "downtime_v_network",
        }
    )
    trade_risk = trade_risk.merge(network_trade_risk, on=groupby_list)

    return trade_risk


def merge_risk(
    import_risk: pd.DataFrame,
    export_risk: pd.DataFrame,
    quantity: bool = False,
    industries: bool = False,
) -> pd.DataFrame:
    """
    Merges the import and export risk to obtain a compounded gloabl risk value for each
    country by looking at domestic and foreign import/export risk. Can perform the analysis on
    quantity of trade flow or value of trade flow.

    Parameters:
        import_risk: pd.DataFrame
            The DataFrame of import risks.
        export_risk: pd.DataFrame
            The DataFrame of export risk.
        quantity: bool default False
            Boolean flag set to true when the analysis should be performed on quantity of trade
            flows.
        industries: bool default False
            Boolean flag set to true if process should be done on each sector seperatly.

    Return:
        global_risk: pd.DataFrame
            A DataFrame object containing the compounded annual domestic, foreign and global
            import/export risk for each country.
    """

    col_name = f"downtime_{'q' if quantity else 'v'}_weighted"
    sea_flow_total = f"{'q' if quantity else 'v'}_sea_flow_total"

    iso3 = "iso3"
    if industries:
        import_risk["iso3_sector"] = import_risk["iso3"] + "_" + import_risk["sector"]
        export_risk["iso3_sector"] = export_risk["iso3"] + "_" + export_risk["sector"]
        iso3 = "iso3_sector"

    domestic_import_risk = (
        import_risk.loc[import_risk["iso3"] == import_risk["port_iso3"]]
        .groupby(iso3)[col_name]
        .sum()
    )
    domestic_import_risk = domestic_import_risk.rename("domestic_import_risk")
    foreign_import_risk = (
        import_risk.loc[import_risk["iso3"] != import_risk["port_iso3"]]
        .groupby(iso3)[col_name]
        .sum()
    )
    foreign_import_risk = foreign_import_risk.rename("foreign_import_risk")

    domestic_export_risk = (
        export_risk.loc[export_risk["iso3"] == export_risk["port_iso3"]]
        .groupby(iso3)[col_name]
        .sum()
    )
    domestic_export_risk = domestic_export_risk.rename("domestic_export_risk")
    foreign_export_risk = (
        export_risk.loc[export_risk["iso3"] != export_risk["port_iso3"]]
        .groupby(iso3)[col_name]
        .sum()
    )
    foreign_export_risk = foreign_export_risk.rename("foreign_export_risk")

    global_risk = pd.concat(
        [
            domestic_import_risk,
            foreign_export_risk,
            domestic_export_risk,
            foreign_import_risk,
        ],
        axis=1,
    )

    total_import = (
        import_risk[[iso3, sea_flow_total]]
        .drop_duplicates()
        .rename(columns={sea_flow_total: "imports"})
        .set_index(iso3)
    )
    total_exports = (
        export_risk[[iso3, sea_flow_total]]
        .drop_duplicates()
        .rename(columns={sea_flow_total: "exports"})
        .set_index(iso3)
    )

    global_risk = pd.concat([global_risk, total_import, total_exports], axis=1)

    ports_import = import_risk.groupby(iso3)["id"].nunique().rename("n_ports_import")
    ports_export = export_risk.groupby(iso3)["id"].nunique().rename("n_ports_import")

    global_risk = pd.concat([global_risk, ports_import, ports_export], axis=1)

    global_risk = global_risk.replace(np.nan, 0.0).reset_index()

    global_risk["total_import_risk"] = (
        global_risk["domestic_import_risk"] + global_risk["foreign_import_risk"]
    )
    global_risk["total_export_risk"] = (
        global_risk["domestic_export_risk"] + global_risk["foreign_export_risk"]
    )
    global_risk["total_risk"] = (
        global_risk["total_import_risk"] * global_risk["imports"]
    ) / global_risk[["imports", "exports"]].sum(axis=1)
    global_risk["total_risk"] += (
        global_risk["total_export_risk"] * global_risk["exports"]
    ) / global_risk[["imports", "exports"]].sum(axis=1)

    if industries:
        global_risk["iso3"] = global_risk["iso3_sector"].apply(
            lambda x: x.split("_")[0]
        )
        global_risk["sector"] = global_risk["iso3_sector"].apply(
            lambda x: x.split("_")[1]
        )

    return global_risk


def create_risk_dataframe(
    trade: dict[str, pd.DataFrame], ports_risk: pd.DataFrame, hazards: list[str]
) -> dict:
    """
    Creates trade risk dataframe by computing trade risk for all hazards in input list.
    Computes import/export risk and merges them together.

    Parameters:
        trade: dict[str, pd.DataFrame]
            Dictionnary mapping name of dataset to the associated trade. Should be iterated two by
            two to obtain both import and export trade values.
        ports_risk: pd.DataFrame
            Risk dataframe to be considered.
        hazards: list[str]
            Hazards for which to compute risk.

    Return:
        risk: dict
            Dictionnary mapping name of dataset to the computed risk.
    """
    risk = {}
    for hazard in hazards:
        risk[hazard] = {}

        for i in range(0, len(trade), 2):
            k1 = list(trade.keys())[i]
            k2 = list(trade.keys())[i + 1]
            v1, v2 = trade[k1], trade[k2]

            industries = "sector" in k1 and "sector" in k2

            export_trade_risk = compute_trade_risk(
                v1, ports_risk, hazard=hazard, industries=industries
            )
            import_trade_risk = compute_trade_risk(
                v2, ports_risk, hazard=hazard, industries=industries
            )
            risk[hazard][f"import{'_sector' if industries else ''}"] = import_trade_risk
            risk[hazard][f"export{'_sector' if industries else ''}"] = export_trade_risk

            risk[hazard][f"value{'_sector' if industries else ''}"] = merge_risk(
                import_trade_risk, export_trade_risk, industries=industries
            )
            risk[hazard][f"quantity{'_sector' if industries else ''}"] = merge_risk(
                import_trade_risk,
                export_trade_risk,
                industries=industries,
                quantity=True,
            )
    return risk
