"""
IO module used for plotting the results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors


from pathlib import Path

from port_risk import data_path


def plot_downtime_risk(
    world,
    countries,
    risk_df: pd.DataFrame,
    name: str,
    hazard: str,
    show: bool = False,
    save: bool = False,
) -> None:
    """
    Plots the dowtime_risk on a world map for the input risk_df and hazard value. Saves to name.

    Parameters:
        world : gpd.GeoDataFrame,
            Geopandas world map.
        countries: gpd.GeoDataFrame,
            Countries identification.
        risk_df: pd.DataFrame,
            Risk dataframe containing risk exposure for eac country.
        name: str,
            Name of file to save to.
        hazard: str,
            Name of climate change hazard to consider.
        show: bool default False
            Boolean flag set True to display to show the plot.
        save: bool default False
            Boolean flag set to True to save the plot.

    Return:
        None
    """
    colorbar = "magma_r"
    vmin = 0
    vmax = 6
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = plt.cm.ScalarMappable(norm=norm, cmap=colorbar)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    columns = [
        "domestic_import_risk",
        "domestic_export_risk",
        "foreign_import_risk",
        "foreign_export_risk",
    ]
    i = 0
    for ax in axes:
        for sub_ax in ax:
            world.plot(color="grey", alpha=0.3, ax=sub_ax)
            countries.merge(risk_df, on="iso3").plot(
                column=columns[i],
                vmin=vmin,
                vmax=vmax,
                cmap=colorbar,
                ax=sub_ax,
            )
            # add colorbar
            ax_cbar = fig.colorbar(
                cbar,
                fraction=0.012,
                pad=-0.16,
                ticks=[0, 2, 4, 6],
                orientation="horizontal",
                ax=sub_ax,
            )
            ax_cbar.ax.set_xticklabels(["0", "2", "4", "6"], fontsize=11)
            ax_cbar.set_label("Downtime risk (days per year)", fontsize=11)
            sub_ax.set_xlim(-180, 180)
            sub_ax.set_ylim(-75, 80)
            plt.rcParams["axes.spines.right"] = True
            plt.rcParams["axes.spines.top"] = True
            plt.rcParams["axes.spines.left"] = True
            plt.rcParams["axes.spines.bottom"] = True
            sub_ax.title.set_text(columns[i])

            sub_ax.title.set_text(" ".join(columns[i].split("_")))
            i += 1
    fig.suptitle(f"Downtime risk {hazard.split('_')[1]} ({name})")
    if show:
        plt.show()
    if save:
        path = Path(data_path["plots"], f"risk_{name}_{hazard}")
        plt.savefig(path)
    plt.close()
