import logging
import logging.handlers
'''
Example for usage of logger.*
from logging_config import setup_logging

logger = setup_logging()
logger.info('Starting geometry calculation.')
logger.debug('Loaded vertices data successfully.')
logger.warning('Potential issue detected: vertex count mismatch.')
logger.error('Error while computing volume constraint.')
logger.critical('Critical error! Cannot proceed further.')
'''

def setup_logging(log_file='membrane_solver.log', quiet: bool = False):
    logger = logging.getLogger('membrane_solver')

    if logger.handlers:
        if quiet:
            logger.handlers = [h for h in logger.handlers
                               if not isinstance(h, logging.StreamHandler)]
        return logger

    logger.setLevel(logging.DEBUG)

    # Log format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler
    file_handler = logging.handlers.RotatingFileHandler(log_file,
                                                        maxBytes=5_000_000,
                                                        backupCount=0)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    if not quiet:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
