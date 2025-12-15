import logging

"""
Example for usage of logger.*
from runtime.logging_config import setup_logging

logger = setup_logging()
logger.info('Starting geometry calculation.')
logger.debug('Loaded vertices data successfully.')
logger.warning('Potential issue detected: vertex count mismatch.')
logger.error('Error while computing volume constraint.')
logger.critical('Critical error! Cannot proceed further.')
"""


def setup_logging(
    log_file: str = "membrane_solver.log",
    quiet: bool = False,
    debug: bool = False,
) -> logging.Logger:
    """Configure and return the shared membrane_solver logger.

    - By default logs at INFO level.
    - When ``debug`` is True, logs at DEBUG level.
    - Uses a FileHandler opened with ``mode='w'`` so each run overwrites
      any existing log file.
    """
    logger = logging.getLogger("membrane_solver")
    level = logging.DEBUG if debug else logging.INFO

    # If logger already has handlers, just adjust levels / quietness.
    if logger.handlers:
        logger.setLevel(level)
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                if quiet:
                    logger.removeHandler(handler)
                else:
                    handler.setLevel(logging.INFO)
            else:
                handler.setLevel(level)
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File handler: overwrite on each run instead of appending/rotating.
    file_handler = None
    try:
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as exc:
        print(f"[logging] Could not open log file '{log_file}': {exc}")

    # Optional console handler for interactive feedback.
    if not quiet:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
