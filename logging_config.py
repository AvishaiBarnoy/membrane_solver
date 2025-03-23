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

def setup_logging(log_file='membrane_solver.log'):
    logger = logging.getLogger('membrane_solver')
    logger.setLevel(logging.DEBUG)

    # Log format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler
    file_handler = logging.handlers.RotatingFileHandler(log_file,
                                                        maxBytes=5_000_000,
                                                        backupCount=3)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Adding handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
