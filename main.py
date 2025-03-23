from logging_config import setup_logging
import argparse

if __name__ == "__main__":
    logger = setup_logging()

    # TODO: add argparse to get input file

    logger.info('Starting geometry calculation.')
    # TODO: add loading geometry file
    logger.debug('Loaded vertices data successfully.')
    logger.warning('Potential issue detected: vertex count mismatch.')
    logger.error('Error while computing volume constraint.')
    logger.critical('Critical error! Cannot proceed further.')
