from pkg_resources import resource_filename

data_path = {
    "climate_data": resource_filename("port_risk", "data/downtime_risk_present.csv"),
    "network": resource_filename("port_risk", "data/port_trade_network.csv"),
    "industries": resource_filename("port_risk", "data/sector_df.csv"),
    "ports": resource_filename("port_risk", "data/nodes_maritime.gpkg"),
    "countries": resource_filename("port_risk", "data/country_file.gpkg"),
    "plots": resource_filename("port_risk", "data/plots/"),
    "latex": resource_filename("port_risk", "data/latex/"),
}
