from logging_config import setup_logging
from visualization.cli import main as _main
from visualization.plotting import plot_geometry


if __name__ == "__main__":
    setup_logging("membrane_solver.log")
    _main()
