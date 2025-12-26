import logging
from typing import Optional


def setup_logging(
    log_file: Optional[str],
    *,
    quiet: bool = False,
    debug: bool = False,
) -> logging.Logger:
    """Configure and return the shared `membrane_solver` logger.

    By default, no file is written. Pass `log_file` to enable file logging.
    """
    logger = logging.getLogger("membrane_solver")
    # Keep propagation enabled so test harnesses (e.g. pytest caplog) can capture
    # records even when we suppress console output.
    logger.propagate = True

    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, mode="w")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError as exc:
            print(f"[logging] Could not open log file '{log_file}': {exc}")

    if not quiet:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
