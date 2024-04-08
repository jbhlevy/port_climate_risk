""""
Main file executed when the package is called at the command line.
Handles the different functionality of the package.
"""

import sys
from argparse import ArgumentParser
from port_risk.main import main as p_main


def main() -> None:
    """
    Main function executed when the package is used as a command line tool. Parses user argument and
    executes the sub-module corresponding to the input argument.

    Parameters:
        None

    Return:
        None
    """
    print("Welcome the the port_risk package for risk assessment of maritime trade !")
    try:
        parser = ArgumentParser(
            add_help=True,
            usage="python port_risk [option1]... [option2]",
            description="""port_risk is a tool to compute study the global maritime trade and
            reproduce the results of the project report of COMP0047. Different options are available
            after the inital pipeline of risk computation is ran: make some statistical tests; 
            comapre machine learning models to learn the data distribution with (or without) 
            validation tuning of hyperparamters.
            """,
        )
        parser.add_argument(
            "-s", "--stats", action="store_true", help="Run statistical tests"
        )
        parser.add_argument(
            "-v",
            "--validation",
            action="store_true",
            help="Run validation of hyper-parameters",
        )
        parser.add_argument(
            "-m",
            "--models",
            action="store_true",
            help="Run the models with no validation",
        )
        args = parser.parse_args()
        p_main(args, sys.argv)
    except KeyboardInterrupt:
        print("\nScript Interrupted by user. Exiting...")
        sys.exit(1)
    sys.exit(1)


if __name__ == "__main__":
    main()
