"""
Main module to be executed when the project is run.
"""

from port_risk.io.latex import make_metrics_table, make_stats_table
import scipy.stats as stats
import pandas as pd
from port_risk.models.machine_learning import run_all_models
from port_risk.preprocessing.machine_learning import (
    get_data_model_ready,
    ml_preprocessing,
    make_validation_data,
)

from port_risk.io.data import (
    load_risk_data,
    load_ports_data,
    load_maritime_network,
    load_industries_data,
    load_countries,
    load_world,
)
from port_risk.io.plots import plot_downtime_risk
from port_risk.io.validation_plots import make_oob_error_plots, make_depth_plots

from port_risk.preprocessing.merge import merge_ports_risk, merge_network_industries
from port_risk.preprocessing.statistics import compute_statistics
from port_risk.preprocessing.trade import create_trade_dataframe
from port_risk.models.risk import create_risk_dataframe
from port_risk.models.machine_learning import find_optimal_params
from port_risk.models.models import (
    ExtraTrees,
    RandomForest,
    GradientBoost,
    models,
    optimal_depth_trees,
    oob_erros_trees,
)


def main(args, argv) -> None:
    """
    Main function to be ran when the package is called.
    """
    risks = load_risk_data()
    ports = load_ports_data()
    ports_risk = merge_ports_risk(risks, ports)
    industries = load_industries_data()
    network = load_maritime_network()

    print("Loaded All required data")
    print("========================")

    # =============
    # Setting flags
    # =============
    if not len(argv) > 1:
        validation_flag = True
        stats_flag = True
        running_flag = True
    else:
        validation_flag = args.validation
        stats_flag = args.stats
        running_flag = args.models

    network = merge_network_industries(network, industries)
    trade = create_trade_dataframe(network, ports)

    print("Create trade dataframes")
    print("=======================")

    # hazards = list(
    #     map(lambda x: f"downtime_{x}", risks["hazard"].drop_duplicates().values)
    # )
    # hazards += ["downtime_total"]
    hazards = ["downtime_total"]
    risk = create_risk_dataframe(trade, ports_risk, hazards)
    print("Create risk dataframes")
    print("======================")

    # for k, v in risk.items():
    #     print(k)
    #     print(v.keys())

    world = load_world()
    countries = load_countries()

    for hazard, risk_dict in risk.items():
        for name, risk_df in risk_dict.items():
            if name in ("value", "quantity"):
                plot_downtime_risk(world, countries, risk_df, name, hazard, save=True)

    # ===================
    # Statistical testing
    # ===================

    tested_functions = {
        "Gamma Distribution": stats.gamma,
        "Exponential Distribution": stats.expon,
        "Normal Distribution": stats.norm,
        "Levy Distribution": stats.levy,
        "Lognormal Distribution": stats.lognorm,
        "Chi2 Distribtuion": stats.ncx2,
        "Student-t Distribution": stats.t,
    }
    if stats_flag:
        data = risk["downtime_total"]["value"]["total_risk"]
        statistics = compute_statistics(data, tested_functions, "var")
        make_stats_table(statistics, "var")

    # ================
    # Machine learning
    # ================
    ml_data = pd.concat(
        [risk["downtime_total"]["import"], risk["downtime_total"]["export"]], axis=0
    )
    # print(ml_data.columns)
    ml_data = ml_preprocessing(ml_data, "trade")
    y = ml_data["downtime_q_weighted"]
    if stats_flag:
        statistics = compute_statistics(y, tested_functions, "risk")
        make_stats_table(statistics, "risk")

    X_train, X_test, Y_train, Y_test = get_data_model_ready(
        ml_data, "downtime_q_weighted"
    )

    params = None
    if validation_flag:
        models_to_test = {
            "extra_trees": ExtraTrees,
            "random_forest": RandomForest,
            "gradient_boost": GradientBoost,
        }
        x_train, y_train, x_val, y_val = make_validation_data(X_train, Y_train)
        optimal_depths = optimal_depth_trees(
            x_train, y_train, x_val, y_val, models_to_test
        )
        oob_errors = oob_erros_trees(x_train, y_train, x_val, y_val, models_to_test)
        make_depth_plots(optimal_depths, save=True)
        make_oob_error_plots(oob_errors, save=True)

        params = find_optimal_params(optimal_depths, oob_errors)

    if running_flag:

        ran_models = run_all_models(
            models, X_train, X_test, Y_train, Y_test, params=params
        )
        make_metrics_table(ran_models)
        for name, model in ran_models.items():
            model.print_metrics()
