"""
Module to make latex table summary of results.
"""

from pathlib import Path

from port_risk import data_path


def format_float(x: float) -> str:
    """
    Rounds float to 3 decimals for string representation.

    Parameters:
        x: float
            Float to round and convert to string.

    Return:
        s: str
            Formatted string.
    """
    s = str(round(x, 3))
    return s


def format_scientific(x: float) -> str:
    """
    Convets float to 3 decimals scientif notationfor string representation.

    Parameters:
        x: float
            Float to round and convert to string.

    Return:
        s: str
            Formatted string.
    """
    x = "{:.3e}".format(x)
    s = str(x)
    return s


def format_name(name: str) -> str:
    """
    Formats names containing _ for latex table display

    Parameters:
        name: str
            Name to format

    Return:
        s: str
            Formatted name.
    """
    s = " ".join(list(map(str.title, name.split("_"))))
    return s


def format_header(s: str) -> str:
    """
    Formats s to bold font in latex.

    Parameters:
        name: str
            String to format

    Return:
        s: str
            Formatted string.
    """
    s = s.upper()
    return "\\textbf{" + s + "}"


def make_stats_table(statistics: dict, f_name: str) -> None:
    """
    Makes the table of computed statistics and saves to f_name.

    Parameters:
        statistics: dict
            A dictionnary containing the tested function name and results of Kolomogorov test.
        f_name: str
            Name of the file to save to.

    Return:
        None
    """
    headers = ["p-value", "statistic"]
    textabular = f"l {'r' * len(headers)}"
    textheader = " Distribution &  " + " & ".join(map(str.title, headers)) + "\\\\"
    textdata = "\\midrule\n"

    for name, (stat, p_value) in statistics.items():
        textdata += f"{name.title()} & {format_scientific(p_value)} & {format_scientific(stat)} \\\\\n"

    text = (
        "\\begin{tabular}{"
        + textabular
        + "}\n"
        + "\\toprule"
        + textheader
        + "\n"
        + textdata
        + "\\bottomrule\n"
        + "\\end{tabular}"
    )
    with open(
        Path(data_path["latex"], f"{f_name}_stats_table.tex"), "w+", encoding="utf-8"
    ) as latex_file:
        latex_file.write(text)


def make_metrics_table(all_models: dict) -> None:
    """
    Makes the table of computed model metrics.

    Parameters:
        all_models: dict
            A dictionnary containing model names mapped to the ran models to acess the statistics
            attribute.

    Return:
        None
    """
    headers = list(all_models[list(all_models)[0]].metrics.keys())
    headers = list(map(format_header, headers))
    print(headers)
    textabular = f"l {'r' * len(headers)}"
    textheader = " Model &  " + " & ".join(map(str.title, headers)) + "\\\\"
    textdata = "\\midrule\n"

    for name, model in all_models.items():
        metrics = model.metrics
        textdata += (
            f"{format_name(name)} & "
            + " & ".join(list(map(format_scientific, metrics.values())))
            + " \\\\\n"
        )

    text = (
        "\\begin{tabular}{"
        + textabular
        + "}\n"
        + "\\toprule"
        + textheader
        + "\n"
        + textdata
        + "\\bottomrule\n"
        + "\\end{tabular}"
    )
    with open(
        Path(data_path["latex"], "metrics_table.tex"), "w+", encoding="utf-8"
    ) as latex_file:
        latex_file.write(text)
