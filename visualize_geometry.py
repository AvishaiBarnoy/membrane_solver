from runtime.logging_config import setup_logging
from visualization.cli import main as _main

if __name__ == "__main__":
    setup_logging("membrane_solver.log")
    _main()
